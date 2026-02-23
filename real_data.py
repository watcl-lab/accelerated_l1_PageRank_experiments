#!/usr/bin/env python3
"""
real_l1_ppr_ista_fista_sweeps_kkt.py

Real-data ISTA vs FISTA benchmarks for the ℓ1-regularized PageRank objective,
using ONLY "work vs parameter" sweeps:

  1) Work vs alpha
  2) Work vs epsilon   (epsilon = KKT/prox-gradient tolerance)
  3) Work vs rho

For each sweep we produce a 2×2 figure (4 datasets), with:
  - mean work over randomly sampled seed nodes
  - IQR (25%–75%) shaded band

Datasets (SNAP, commonly used in local graph clustering / network science):
  - com-Amazon  (co-purchase)  
  - com-DBLP    (collaboration) 
  - com-Youtube (social) 
  - email-Enron (email)

Work metric:
  work += sum_{i in supp(y)} d_i  +  sum_{i in supp(x_next)} d_i

Notes on the termination residual:
  - We use step-size 1, consistent with the ISTA update in your code.
  - For this objective, x* is optimal iff x* = prox_g(x* - ∇f(x*)),
    so the residual r(x) is a standard stationarity/KKT surrogate.

Outputs (default: ./outputs):
  - work_vs_alpha_4datasets.(png/pdf)
  - work_vs_epsilon_4datasets.(png/pdf)
  - work_vs_rho_4datasets.(png/pdf)

Run:
  python3 real_l1_ppr_ista_fista_sweeps_kkt.py --help
"""

from __future__ import annotations

import argparse
import gzip
import math
import re
import shutil
import urllib.request
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# -----------------------------
# Dataset metadata (SNAP)
# -----------------------------
@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    url_gz: str
    filename: str  # uncompressed filename (we'll keep .txt after extracting)
    n_hint: int | None = None  # optional node-count hint for large graphs


DATASETS: List[DatasetSpec] = [
    DatasetSpec(
        key="com-Amazon",
        name="com-Amazon (co-purchase)",
        url_gz="https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz",
        filename="com-amazon.ungraph.txt",
    ),
    DatasetSpec(
        key="com-DBLP",
        name="com-DBLP (collaboration)",
        url_gz="https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz",
        filename="com-dblp.ungraph.txt",
    ),
    DatasetSpec(
        key="com-Youtube",
        name="com-Youtube (social)",
        url_gz="https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz",
        filename="com-youtube.ungraph.txt",
    ),
    DatasetSpec(
        key="com-Orkut",
        name="com-Orkut (social)",
        url_gz="https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz",
        filename="com-orkut.ungraph.txt",
        # SNAP reports ~3,072,441 nodes.
        n_hint=3_072_441,
    ),
]


# -----------------------------
# IO: download + parse
# -----------------------------
def download_if_needed(url: str, dst_gz: Path) -> None:
    dst_gz.parent.mkdir(parents=True, exist_ok=True)
    if dst_gz.exists() and dst_gz.stat().st_size > 0:
        return
    print(f"[download] {url}")
    tmp = dst_gz.with_suffix(dst_gz.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 (expected)
    tmp.replace(dst_gz)


def gunzip_if_needed(src_gz: Path, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    print(f"[extract] {src_gz.name} -> {dst.name}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src_gz, "rb") as fin, open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout)


def load_undirected_unweighted_graph_from_edgelist(path: Path, comment: str = "#") -> sp.csr_matrix:
    """
    Load an edge list into an undirected, unweighted simple graph adjacency matrix (CSR).
    - Removes self-loops
    - Symmetrizes
    - Coalesces duplicates into 1s

    Memory-conscious reader: uses array('I') instead of Python int lists.
    """
    rows = array("I")
    cols = array("I")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith(comment):
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                u = int(parts[0])
                v = int(parts[1])
            except ValueError:
                continue
            if u == v:
                continue
            if u < 0 or v < 0:
                continue
            rows.append(u)
            cols.append(v)

    if len(rows) == 0:
        raise ValueError(f"No edges parsed from {path}")

    rows_np = np.frombuffer(rows, dtype=np.uint32).astype(np.int64, copy=False)
    cols_np = np.frombuffer(cols, dtype=np.uint32).astype(np.int64, copy=False)

    # Reindex nodes to 0..n-1 to avoid huge IDs / gaps (robustness).
    all_nodes = np.concatenate([rows_np, cols_np])
    uniq, inv = np.unique(all_nodes, return_inverse=True)
    m = rows_np.size
    r = inv[:m].astype(np.int32, copy=False)
    c = inv[m:].astype(np.int32, copy=False)
    n = int(uniq.size)

    rr = np.concatenate([r, c])
    cc = np.concatenate([c, r])
    data = np.ones(rr.size, dtype=np.uint8)

    A = sp.coo_matrix((data, (rr, cc)), shape=(n, n)).tocsr()
    A.sum_duplicates()
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1  # binary
    return A


# -----------------------------
# Local P*x (sparse-vector times normalized adjacency)
# -----------------------------
def p_dot_sparse_from_A(A: sp.csr_matrix, inv_sqrt_d: np.ndarray, x: Dict[int, float]) -> Dict[int, float]:
    """
    Compute Px = (D^{-1/2} A D^{-1/2}) x, expanding only rows for j in supp(x).
    Returns a dict {i: (Px)_i}. Complexity ~ sum_{j in supp(x)} deg(j).
    """
    if not x:
        return {}

    idx_chunks: List[np.ndarray] = []
    val_chunks: List[np.ndarray] = []

    for j, xj in x.items():
        start, end = A.indptr[j], A.indptr[j + 1]
        nbrs = A.indices[start:end]
        if nbrs.size == 0:
            continue
        vals = (inv_sqrt_d[j] * xj) * inv_sqrt_d[nbrs]
        idx_chunks.append(nbrs)
        val_chunks.append(vals)

    if not idx_chunks:
        return {}

    idx = np.concatenate(idx_chunks)
    val = np.concatenate(val_chunks)

    # coalesce duplicates by sorting
    order = np.argsort(idx, kind="mergesort")
    idx = idx[order]
    val = val[order]

    uniq, first = np.unique(idx, return_index=True)
    summed = np.add.reduceat(val, first)

    out: Dict[int, float] = {}
    for i, vi in zip(uniq, summed):
        if vi != 0.0:
            out[int(i)] = float(vi)
    return out


def dict_axpby(a: float, x: Dict[int, float], b: float, y: Dict[int, float]) -> Dict[int, float]:
    """
    out = a*x + b*y for sparse dict vectors.
    """
    out: Dict[int, float] = {}
    if a != 0.0:
        for i, xi in x.items():
            v = a * xi
            if v != 0.0:
                out[i] = v
    if b != 0.0:
        for i, yi in y.items():
            v = out.get(i, 0.0) + b * yi
            if v != 0.0:
                out[i] = v
            else:
                out.pop(i, None)
    return out


def soft_threshold_dict_scaled(u: Dict[int, float], scale: float, sqrt_d: np.ndarray) -> Dict[int, float]:
    """
    Prox of scale * sum_i sqrt(d_i) |x_i|  applied to vector u:
      prox(u)_i = sign(u_i) * max(|u_i| - scale*sqrt(d_i), 0)
    Only over indices present in u (others stay 0).
    """
    out: Dict[int, float] = {}
    if scale <= 0.0:
        # no thresholding
        for i, ui in u.items():
            if ui != 0.0:
                out[i] = float(ui)
        return out

    for i, ui in u.items():
        t = scale * float(sqrt_d[i])
        aui = abs(ui)
        if aui > t:
            out[i] = math.copysign(aui - t, ui)
    return out


def sum_degrees(d: np.ndarray, x: Dict[int, float]) -> float:
    s = 0.0
    for i in x.keys():
        s += float(d[i])
    return s


def prox_step_from_xPx(
    x: Dict[int, float],
    Px: Dict[int, float],
    *,
    c: float,
    b_seed: float,
    seed: int,
    l1_scale: float,
    sqrt_d: np.ndarray,
) -> Dict[int, float]:
    """
    Compute T(x) = prox_g( x - ∇f(x) ) for step size 1.
    Given Px = P x, and using:
      x - ∇f(x) = c*(x + P x) + b_seed*e_seed
    """
    u: Dict[int, float] = {}
    for i, xi in x.items():
        u[i] = c * xi
    for i, pxi in Px.items():
        u[i] = u.get(i, 0.0) + c * pxi
        if u[i] == 0.0:
            u.pop(i, None)
    u[seed] = u.get(seed, 0.0) + b_seed

    return soft_threshold_dict_scaled(u, l1_scale, sqrt_d)


def prox_grad_inf_norm(x: Dict[int, float], Tx: Dict[int, float]) -> float:
    """
    Infinity-norm of x - T(x) for dict vectors.
    """
    m = 0.0
    for i, xi in x.items():
        v = abs(xi - Tx.get(i, 0.0))
        if v > m:
            m = v
    for i, yi in Tx.items():
        if i not in x:
            v = abs(yi)
            if v > m:
                m = v
    return float(m)


# -----------------------------
# Algorithms: run until KKT/prox-grad tolerance
# -----------------------------
def beta_sc(alpha: float) -> float:
    return (1.0 - math.sqrt(alpha)) / (1.0 + math.sqrt(alpha))


def run_ista_until_kkt(
    A: sp.csr_matrix,
    d: np.ndarray,
    inv_sqrt_d: np.ndarray,
    sqrt_d: np.ndarray,
    *,
    seed: int,
    alpha: float,
    rho: float,
    kkt_eps: float,
    max_iter: int,
) -> float:
    """
    Return work to reach prox-grad residual <= kkt_eps, or NaN if not reached by max_iter.
    """
    c = (1.0 - alpha) / 2.0
    l1_scale = rho * alpha
    b_seed = alpha / float(sqrt_d[seed])

    x: Dict[int, float] = {}
    Px: Dict[int, float] = {}
    work = 0.0

    # residual at x0
    Tx = prox_step_from_xPx(x, Px, c=c, b_seed=b_seed, seed=seed, l1_scale=l1_scale, sqrt_d=sqrt_d)
    r = prox_grad_inf_norm(x, Tx)
    if r <= kkt_eps:
        return 0.0

    for _ in range(max_iter):
        # ISTA update: x_next = T(x)
        x_next = Tx
        Px_next = p_dot_sparse_from_A(A, inv_sqrt_d, x_next)

        # work increment uses y=x and x_next
        work += sum_degrees(d, x) + sum_degrees(d, x_next)

        x, Px = x_next, Px_next

        # compute residual at new x
        Tx = prox_step_from_xPx(x, Px, c=c, b_seed=b_seed, seed=seed, l1_scale=l1_scale, sqrt_d=sqrt_d)
        r = prox_grad_inf_norm(x, Tx)
        if r <= kkt_eps:
            return float(work)

    return float("nan")


def run_fista_until_kkt(
    A: sp.csr_matrix,
    d: np.ndarray,
    inv_sqrt_d: np.ndarray,
    sqrt_d: np.ndarray,
    *,
    seed: int,
    alpha: float,
    rho: float,
    beta: float,
    kkt_eps: float,
    max_iter: int,
) -> float:
    """
    Return work to reach prox-grad residual <= kkt_eps, or NaN if not reached by max_iter.

    Termination is evaluated on the current iterate x using the SAME residual as ISTA:
      r(x) = ||x - T(x)||_∞, where T is the ISTA prox-step mapping.
    """
    c = (1.0 - alpha) / 2.0
    l1_scale = rho * alpha
    b_seed = alpha / float(sqrt_d[seed])

    x_prev: Dict[int, float] = {}
    x: Dict[int, float] = {}
    Px_prev: Dict[int, float] = {}
    Px: Dict[int, float] = {}
    work = 0.0

    # residual at x0
    Tx = prox_step_from_xPx(x, Px, c=c, b_seed=b_seed, seed=seed, l1_scale=l1_scale, sqrt_d=sqrt_d)
    r = prox_grad_inf_norm(x, Tx)
    if r <= kkt_eps:
        return 0.0

    for _ in range(max_iter):
        # y = (1+beta)x - beta x_prev
        y = dict_axpby(1.0 + beta, x, -beta, x_prev)
        # Py = (1+beta)Px - beta Px_prev
        Py = dict_axpby(1.0 + beta, Px, -beta, Px_prev)

        x_next = prox_step_from_xPx(y, Py, c=c, b_seed=b_seed, seed=seed, l1_scale=l1_scale, sqrt_d=sqrt_d)
        Px_next = p_dot_sparse_from_A(A, inv_sqrt_d, x_next)

        work += sum_degrees(d, y) + sum_degrees(d, x_next)

        x_prev, x = x, x_next
        Px_prev, Px = Px, Px_next

        Tx = prox_step_from_xPx(x, Px, c=c, b_seed=b_seed, seed=seed, l1_scale=l1_scale, sqrt_d=sqrt_d)
        r = prox_grad_inf_norm(x, Tx)
        if r <= kkt_eps:
            return float(work)

    return float("nan")


def run_ista_kkt_history(
    A: sp.csr_matrix,
    d: np.ndarray,
    inv_sqrt_d: np.ndarray,
    sqrt_d: np.ndarray,
    *,
    seed: int,
    alpha: float,
    rho: float,
    max_iter: int,
    stop_tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (works, residuals) for ISTA, starting at x0, stopping early when residual <= stop_tol.

    works[k] and residuals[k] correspond to iterate x_k.
    """
    c = (1.0 - alpha) / 2.0
    l1_scale = rho * alpha
    b_seed = alpha / float(sqrt_d[seed])

    x: Dict[int, float] = {}
    Px: Dict[int, float] = {}
    work = 0.0

    works: List[float] = []
    resids: List[float] = []

    Tx = prox_step_from_xPx(x, Px, c=c, b_seed=b_seed, seed=seed, l1_scale=l1_scale, sqrt_d=sqrt_d)
    r = prox_grad_inf_norm(x, Tx)
    works.append(0.0)
    resids.append(r)
    if r <= stop_tol:
        return np.asarray(works, dtype=float), np.asarray(resids, dtype=float)

    for _ in range(max_iter):
        x_next = Tx
        Px_next = p_dot_sparse_from_A(A, inv_sqrt_d, x_next)

        work += sum_degrees(d, x) + sum_degrees(d, x_next)

        x, Px = x_next, Px_next

        Tx = prox_step_from_xPx(x, Px, c=c, b_seed=b_seed, seed=seed, l1_scale=l1_scale, sqrt_d=sqrt_d)
        r = prox_grad_inf_norm(x, Tx)

        works.append(work)
        resids.append(r)

        if r <= stop_tol:
            break

    return np.asarray(works, dtype=float), np.asarray(resids, dtype=float)


def run_fista_kkt_history(
    A: sp.csr_matrix,
    d: np.ndarray,
    inv_sqrt_d: np.ndarray,
    sqrt_d: np.ndarray,
    *,
    seed: int,
    alpha: float,
    rho: float,
    beta: float,
    max_iter: int,
    stop_tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (works, residuals) for FISTA, starting at x0, stopping early when residual <= stop_tol.

    works[k] and residuals[k] correspond to iterate x_k.
    Termination residual is always computed at the iterate x_k (same as ISTA).
    """
    c = (1.0 - alpha) / 2.0
    l1_scale = rho * alpha
    b_seed = alpha / float(sqrt_d[seed])

    x_prev: Dict[int, float] = {}
    x: Dict[int, float] = {}
    Px_prev: Dict[int, float] = {}
    Px: Dict[int, float] = {}
    work = 0.0

    works: List[float] = []
    resids: List[float] = []

    Tx = prox_step_from_xPx(x, Px, c=c, b_seed=b_seed, seed=seed, l1_scale=l1_scale, sqrt_d=sqrt_d)
    r = prox_grad_inf_norm(x, Tx)
    works.append(0.0)
    resids.append(r)
    if r <= stop_tol:
        return np.asarray(works, dtype=float), np.asarray(resids, dtype=float)

    for _ in range(max_iter):
        y = dict_axpby(1.0 + beta, x, -beta, x_prev)
        Py = dict_axpby(1.0 + beta, Px, -beta, Px_prev)

        x_next = prox_step_from_xPx(y, Py, c=c, b_seed=b_seed, seed=seed, l1_scale=l1_scale, sqrt_d=sqrt_d)
        Px_next = p_dot_sparse_from_A(A, inv_sqrt_d, x_next)

        work += sum_degrees(d, y) + sum_degrees(d, x_next)

        x_prev, x = x, x_next
        Px_prev, Px = Px, Px_next

        Tx = prox_step_from_xPx(x, Px, c=c, b_seed=b_seed, seed=seed, l1_scale=l1_scale, sqrt_d=sqrt_d)
        r = prox_grad_inf_norm(x, Tx)

        works.append(work)
        resids.append(r)

        if r <= stop_tol:
            break

    return np.asarray(works, dtype=float), np.asarray(resids, dtype=float)


def first_work_below_tol(residuals: np.ndarray, works: np.ndarray, eps: float) -> float:
    idx = np.where(residuals <= eps)[0]
    if idx.size == 0:
        return float("nan")
    return float(works[idx[0]])


# -----------------------------
# Sweeps + aggregation
# -----------------------------
def agg_work_stats(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate over seeds for each x-grid point.
    Returns (mean, q25, q75), ignoring NaNs.
    """
    mean = np.nanmean(W, axis=0)
    q25 = np.nanquantile(W, 0.25, axis=0)
    q75 = np.nanquantile(W, 0.75, axis=0)
    return mean, q25, q75


def sweep_alpha(
    A: sp.csr_matrix,
    d: np.ndarray,
    inv_sqrt_d: np.ndarray,
    sqrt_d: np.ndarray,
    *,
    dataset_name: str,
    seeds: np.ndarray,
    alpha_grid: np.ndarray,
    rho: float,
    kkt_eps: float,
    max_iter: int,
) -> dict:
    W_ista = np.full((seeds.size, alpha_grid.size), np.nan, dtype=float)
    W_fista = np.full((seeds.size, alpha_grid.size), np.nan, dtype=float)

    print(f"\n[{dataset_name}] alpha sweep: rho={rho:g}, kkt_eps={kkt_eps:g}, max_iter={max_iter}")
    for ai, alpha in enumerate(alpha_grid):
        beta = beta_sc(float(alpha))
        print(f"  alpha={alpha:.3g}  beta={beta:.3g}")
        for si, seed in enumerate(seeds):
            W_ista[si, ai] = run_ista_until_kkt(
                A, d, inv_sqrt_d, sqrt_d,
                seed=int(seed), alpha=float(alpha), rho=float(rho),
                kkt_eps=float(kkt_eps), max_iter=int(max_iter)
            )
            W_fista[si, ai] = run_fista_until_kkt(
                A, d, inv_sqrt_d, sqrt_d,
                seed=int(seed), alpha=float(alpha), rho=float(rho), beta=float(beta),
                kkt_eps=float(kkt_eps), max_iter=int(max_iter)
            )

    mI, qI1, qI3 = agg_work_stats(W_ista)
    mF, qF1, qF3 = agg_work_stats(W_fista)
    return {
        "dataset": dataset_name,
        "x": alpha_grid,
        "xlabel": r"$\alpha$",
        "ISTA": {"mean": mI, "q25": qI1, "q75": qI3},
        "FISTA": {"mean": mF, "q25": qF1, "q75": qF3},
    }


def sweep_rho(
    A: sp.csr_matrix,
    d: np.ndarray,
    inv_sqrt_d: np.ndarray,
    sqrt_d: np.ndarray,
    *,
    dataset_name: str,
    seeds: np.ndarray,
    rho_grid: np.ndarray,
    alpha: float,
    kkt_eps: float,
    max_iter: int,
) -> dict:
    W_ista = np.full((seeds.size, rho_grid.size), np.nan, dtype=float)
    W_fista = np.full((seeds.size, rho_grid.size), np.nan, dtype=float)

    beta = beta_sc(float(alpha))

    print(f"\n[{dataset_name}] rho sweep: alpha={alpha:g}, kkt_eps={kkt_eps:g}, max_iter={max_iter}")
    for ri, rho in enumerate(rho_grid):
        print(f"  rho={rho:.2e}")
        for si, seed in enumerate(seeds):
            W_ista[si, ri] = run_ista_until_kkt(
                A, d, inv_sqrt_d, sqrt_d,
                seed=int(seed), alpha=float(alpha), rho=float(rho),
                kkt_eps=float(kkt_eps), max_iter=int(max_iter)
            )
            W_fista[si, ri] = run_fista_until_kkt(
                A, d, inv_sqrt_d, sqrt_d,
                seed=int(seed), alpha=float(alpha), rho=float(rho), beta=float(beta),
                kkt_eps=float(kkt_eps), max_iter=int(max_iter)
            )

    mI, qI1, qI3 = agg_work_stats(W_ista)
    mF, qF1, qF3 = agg_work_stats(W_fista)
    return {
        "dataset": dataset_name,
        "x": rho_grid,
        "xlabel": r"$\rho$",
        "ISTA": {"mean": mI, "q25": qI1, "q75": qI3},
        "FISTA": {"mean": mF, "q25": qF1, "q75": qF3},
    }


def sweep_epsilon(
    A: sp.csr_matrix,
    d: np.ndarray,
    inv_sqrt_d: np.ndarray,
    sqrt_d: np.ndarray,
    *,
    dataset_name: str,
    seeds: np.ndarray,
    eps_grid: np.ndarray,
    alpha: float,
    rho: float,
    max_iter: int,
) -> dict:
    """
    epsilon sweep: epsilon is the KKT/prox-grad tolerance.
    For each seed, run each method once (up to max_iter or until eps_min),
    then compute work-to-first-hit for every eps in eps_grid.
    """
    eps_grid = np.asarray(eps_grid, dtype=float)
    eps_min = float(eps_grid.min())
    beta = beta_sc(float(alpha))

    W_ista = np.full((seeds.size, eps_grid.size), np.nan, dtype=float)
    W_fista = np.full((seeds.size, eps_grid.size), np.nan, dtype=float)

    print(f"\n[{dataset_name}] epsilon sweep: alpha={alpha:g}, rho={rho:g}, max_iter={max_iter}")
    print(f"  eps grid: [{eps_grid.min():.1e}, ..., {eps_grid.max():.1e}]  (stop_tol={eps_min:.1e})")

    for si, seed in enumerate(seeds):
        if (si + 1) % 10 == 0 or si == 0:
            print(f"  seed {si+1:3d}/{seeds.size}: node={int(seed)} deg={d[int(seed)]:.0f}")

        works_I, resids_I = run_ista_kkt_history(
            A, d, inv_sqrt_d, sqrt_d,
            seed=int(seed), alpha=float(alpha), rho=float(rho),
            max_iter=int(max_iter), stop_tol=eps_min
        )
        works_F, resids_F = run_fista_kkt_history(
            A, d, inv_sqrt_d, sqrt_d,
            seed=int(seed), alpha=float(alpha), rho=float(rho), beta=float(beta),
            max_iter=int(max_iter), stop_tol=eps_min
        )

        for ei, eps in enumerate(eps_grid):
            W_ista[si, ei] = first_work_below_tol(resids_I, works_I, float(eps))
            W_fista[si, ei] = first_work_below_tol(resids_F, works_F, float(eps))

    mI, qI1, qI3 = agg_work_stats(W_ista)
    mF, qF1, qF3 = agg_work_stats(W_fista)
    return {
        "dataset": dataset_name,
        "x": eps_grid,
        "xlabel": r"$\epsilon$",
        "ISTA": {"mean": mI, "q25": qI1, "q75": qI3},
        "FISTA": {"mean": mF, "q25": qF1, "q75": qF3},
    }


# -----------------------------
# NEW: Scatter experiment — volume of S* vs volume of "extra" nodes activated by FISTA
# -----------------------------
def run_ista_until_kkt_return_x(
    A: sp.csr_matrix,
    d: np.ndarray,
    inv_sqrt_d: np.ndarray,
    sqrt_d: np.ndarray,
    *,
    seed: int,
    alpha: float,
    rho: float,
    kkt_eps: float,
    max_iter: int,
) -> Tuple[float, int, float, float, Dict[int, float], bool]:
    """Run ISTA until KKT/prox-grad tol and return extra diagnostics.

    Returns:
        work_total:   total work (same definition used everywhere else)
        iters:        number of ISTA prox-steps performed until r(x)<=kkt_eps
        work_y:       cumulative work contributed by y (=x for ISTA)
        work_xnext:   cumulative work contributed by x_next
        x_star:       final iterate (dict-sparse)
        reached:      whether tolerance was reached within max_iter

    Notes:
      * This function is ONLY used by the scatter/diagnostic experiments.
        The main sweep experiments still call run_ista_until_kkt unchanged.
    """
    c = (1.0 - alpha) / 2.0
    l1_scale = rho * alpha
    b_seed = alpha / float(sqrt_d[seed])

    x: Dict[int, float] = {}
    Px: Dict[int, float] = {}

    work_total = 0.0
    work_y = 0.0
    work_xnext = 0.0
    iters = 0

    # residual at x0
    Tx = prox_step_from_xPx(x, Px, c=c, b_seed=b_seed, seed=seed, l1_scale=l1_scale, sqrt_d=sqrt_d)
    r = prox_grad_inf_norm(x, Tx)
    if r <= kkt_eps:
        return 0.0, 0, 0.0, 0.0, x, True

    for _ in range(max_iter):
        # ISTA update: x_next = T(x)
        x_next = Tx
        Px_next = p_dot_sparse_from_A(A, inv_sqrt_d, x_next)

        # work increment uses y=x and x_next
        wy = sum_degrees(d, x)
        wx = sum_degrees(d, x_next)
        work_total += wy + wx
        work_y += wy
        work_xnext += wx

        x, Px = x_next, Px_next
        iters += 1

        # compute residual at new x
        Tx = prox_step_from_xPx(x, Px, c=c, b_seed=b_seed, seed=seed, l1_scale=l1_scale, sqrt_d=sqrt_d)
        r = prox_grad_inf_norm(x, Tx)
        if r <= kkt_eps:
            return float(work_total), int(iters), float(work_y), float(work_xnext), x, True

    return float("nan"), int(iters), float(work_y), float(work_xnext), x, False


def run_fista_until_kkt_union_support(
    A: sp.csr_matrix,
    d: np.ndarray,
    inv_sqrt_d: np.ndarray,
    sqrt_d: np.ndarray,
    *,
    seed: int,
    alpha: float,
    rho: float,
    beta: float,
    kkt_eps: float,
    max_iter: int,
    S_star: "set[int]",
) -> Tuple[float, int, float, float, float, float, float, float, float, float, bool]:
    """Run FISTA until KKT tol while tracking *where* the work is spent.

    This is a diagnostic variant used ONLY by the scatter/diagnostic experiments.

    We track two different kinds of "extra" exploration relative to the (ISTA) support S*:

      extra_x := union_k supp(x_k) \ S*
      extra_y := union_k supp(y_k) \ S*    where y_k = (1+beta)x_k - beta x_{k-1}

    Note: Your "work" metric charges *both* supp(y_k) and supp(x_{k+1}), so even if
    extra_x is small, a large extra_y (or large *cumulative* outside-S* volume over many
    iterations) can still make FISTA much more expensive.

    Returns:
      work_total, iters,
      work_y_total, work_x_total,
      extra_x_vol, extra_y_vol,
      outside_work_y, outside_work_x,
      max_vol_y, max_vol_xnext,
      reached
    """
    c = (1.0 - alpha) / 2.0
    l1_scale = rho * alpha
    b_seed = alpha / float(sqrt_d[seed])

    x_prev: Dict[int, float] = {}
    x: Dict[int, float] = {}
    Px_prev: Dict[int, float] = {}
    Px: Dict[int, float] = {}

    work_total = 0.0
    work_y_total = 0.0
    work_x_total = 0.0
    outside_work_y = 0.0
    outside_work_x = 0.0
    max_vol_y = 0.0
    max_vol_xnext = 0.0
    iters = 0

    extra_x_nodes: set[int] = set()
    extra_y_nodes: set[int] = set()
    extra_x_vol = 0.0
    extra_y_vol = 0.0

    # residual at x0
    Tx = prox_step_from_xPx(x, Px, c=c, b_seed=b_seed, seed=seed, l1_scale=l1_scale, sqrt_d=sqrt_d)
    r = prox_grad_inf_norm(x, Tx)
    if r <= kkt_eps:
        return 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True

    for _ in range(max_iter):
        # y = (1+beta)x - beta x_prev
        y = dict_axpby(1.0 + beta, x, -beta, x_prev)
        # Py = (1+beta)Px - beta Px_prev
        Py = dict_axpby(1.0 + beta, Px, -beta, Px_prev)

        x_next = prox_step_from_xPx(y, Py, c=c, b_seed=b_seed, seed=seed, l1_scale=l1_scale, sqrt_d=sqrt_d)
        Px_next = p_dot_sparse_from_A(A, inv_sqrt_d, x_next)

        # work increment
        vy = sum_degrees(d, y)
        vx = sum_degrees(d, x_next)
        work_total += vy + vx
        work_y_total += vy
        work_x_total += vx
        if vy > max_vol_y:
            max_vol_y = vy
        if vx > max_vol_xnext:
            max_vol_xnext = vx

        # (A) unique extra nodes (union support) in y and x_next
        for j in y.keys():
            if j not in S_star and j not in extra_y_nodes:
                extra_y_nodes.add(j)
                extra_y_vol += float(d[j])
        for j in x_next.keys():
            if j not in S_star and j not in extra_x_nodes:
                extra_x_nodes.add(j)
                extra_x_vol += float(d[j])

        # (B) cumulative outside-S* work (counts multiplicity across iterations)
        oy = 0.0
        for j in y.keys():
            if j not in S_star:
                oy += float(d[j])
        ox = 0.0
        for j in x_next.keys():
            if j not in S_star:
                ox += float(d[j])
        outside_work_y += oy
        outside_work_x += ox

        x_prev, x = x, x_next
        Px_prev, Px = Px, Px_next
        iters += 1

        # residual at current iterate x
        Tx = prox_step_from_xPx(x, Px, c=c, b_seed=b_seed, seed=seed, l1_scale=l1_scale, sqrt_d=sqrt_d)
        r = prox_grad_inf_norm(x, Tx)
        if r <= kkt_eps:
            return (
                float(work_total), int(iters),
                float(work_y_total), float(work_x_total),
                float(extra_x_vol), float(extra_y_vol),
                float(outside_work_y), float(outside_work_x),
                float(max_vol_y), float(max_vol_xnext),
                True,
            )

    return (
        float("nan"), int(iters),
        float(work_y_total), float(work_x_total),
        float(extra_x_vol), float(extra_y_vol),
        float(outside_work_y), float(outside_work_x),
        float(max_vol_y), float(max_vol_xnext),
        False,
    )


def volume_of_node_set(d: np.ndarray, nodes: Iterable[int]) -> float:
    """
    Volume = sum_{i in nodes} d_i.
    """
    # Use a loop to avoid creating large temporary arrays for huge sets.
    s = 0.0
    for i in nodes:
        s += float(d[int(i)])
    return s


def scatter_volumes_Sstar_vs_fista_extra(
    A: sp.csr_matrix,
    d: np.ndarray,
    inv_sqrt_d: np.ndarray,
    sqrt_d: np.ndarray,
    *,
    dataset_key: str,
    dataset_name: str,
    seeds: np.ndarray,
    alpha: float,
    rho: float,
    kkt_eps: float,
    max_iter: int,
) -> dict:
    """Scatter experiment (plus extra diagnostics) for a fixed (alpha,rho,kkt_eps)."""
    seeds = np.asarray(seeds, dtype=np.int64)

    vol_Sstar = np.full(seeds.size, np.nan, dtype=float)
    vol_extra_x = np.full(seeds.size, np.nan, dtype=float)
    vol_extra_y = np.full(seeds.size, np.nan, dtype=float)

    work_ista = np.full(seeds.size, np.nan, dtype=float)
    work_fista = np.full(seeds.size, np.nan, dtype=float)
    iters_ista = np.full(seeds.size, np.nan, dtype=float)
    iters_fista = np.full(seeds.size, np.nan, dtype=float)

    outside_work_frac_fista = np.full(seeds.size, np.nan, dtype=float)
    max_vol_y_fista = np.full(seeds.size, np.nan, dtype=float)

    alpha = float(alpha)
    rho = float(rho)
    kkt_eps = float(kkt_eps)
    beta = beta_sc(alpha)

    print(f"\n[{dataset_name}] scatter+diagnostics at alpha={alpha:.3g}, rho={rho:g}, kkt_eps={kkt_eps:g}")
    for si, seed in enumerate(seeds):
        if (si + 1) % 10 == 0 or si == 0:
            print(f"  seed {si+1:3d}/{seeds.size}: node={int(seed)}")

        wI, itI, wIy, wIx, x_star, reached_I = run_ista_until_kkt_return_x(
            A, d, inv_sqrt_d, sqrt_d,
            seed=int(seed), alpha=alpha, rho=rho,
            kkt_eps=kkt_eps, max_iter=int(max_iter)
        )
        if not reached_I or not np.isfinite(wI):
            continue

        S_star = set(x_star.keys())
        volS = volume_of_node_set(d, S_star)

        (
            wF, itF,
            wFy, wFx,
            extra_x_vol, extra_y_vol,
            out_y, out_x,
            max_vy, _max_vx,
            reached_F,
        ) = run_fista_until_kkt_union_support(
            A, d, inv_sqrt_d, sqrt_d,
            seed=int(seed), alpha=alpha, rho=rho, beta=float(beta),
            kkt_eps=kkt_eps, max_iter=int(max_iter),
            S_star=S_star,
        )
        if not reached_F or not np.isfinite(wF):
            continue

        vol_Sstar[si] = volS
        vol_extra_x[si] = extra_x_vol
        vol_extra_y[si] = extra_y_vol

        work_ista[si] = wI
        work_fista[si] = wF
        iters_ista[si] = float(itI)
        iters_fista[si] = float(itF)

        denom = float(wF)
        outside_work_frac_fista[si] = float((out_y + out_x) / denom) if denom > 0 else 0.0
        max_vol_y_fista[si] = float(max_vy)

    return {
        "dataset_key": dataset_key,
        "seeds": seeds.copy(),
        "dataset": dataset_name,
        "alpha": float(alpha),
        "rho": float(rho),
        "kkt_eps": float(kkt_eps),
        "vol_Sstar": vol_Sstar,
        "vol_extra": vol_extra_x,
        "vol_fista_extra": vol_extra_x,
        "vol_extra_y": vol_extra_y,
        "work_ista": work_ista,
        "work_fista": work_fista,
        "iters_ista": iters_ista,
        "iters_fista": iters_fista,
        "outside_work_frac_fista": outside_work_frac_fista,
        "max_vol_y_fista": max_vol_y_fista,
    }


# -----------------------------
# Plot styling helpers (match B600 style)
# -----------------------------
PLOT_FIGSIZE = (8.0, 4.8)
PLOT_LW = 6.0
PLOT_MS = 12
FS_LABEL = 23
FS_TICK = 21
FS_LEGEND = 24


def _kfmt(x, pos):
    """Compressed tick labels: 1.2K, 50K, 3.4M, ... (same as B600 script)."""
    x = float(x)
    axabs = abs(x)
    if axabs >= 1_000_000:
        v = x / 1_000_000
        return f"{v:.1f}M" if abs(v) < 10 else f"{v:.0f}M"
    if axabs >= 1_000:
        v = x / 1_000
        return f"{v:.1f}K" if abs(v) < 10 else f"{v:.0f}K"
    if axabs >= 1:
        return f"{x:.0f}"
    if axabs == 0:
        return "0"
    return f"{x:.0e}"


def _plain_log_ticks(ax) -> None:
    # Keep log scaling, but show ticks as plain decimals (1, 1.1, 2, 10, ...)
    sf = mticker.ScalarFormatter()
    sf.set_scientific(False)
    sf.set_useOffset(False)
    ax.xaxis.set_major_formatter(sf)
    ax.yaxis.set_major_formatter(sf)


def _plain_decimal_ticks(ax, *, labelsize: int) -> None:
    # Plain decimal labels on (possibly log) axes: 0.1, 1, 2, 10, ...
    fmt = mticker.FuncFormatter(lambda v, pos: f"{v:g}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.tick_params(axis="both", which="major", labelsize=labelsize)


# -----------------------------
# NEW (scatter only): decimal ticks with 1 digit that DO NOT OVERLAP
# -----------------------------
def _scatter_decimal_ticks_1(ax, *, labelsize: int) -> None:
    """
    Scatter plots ONLY:
      - x and y major tick labels: fixed-point decimals with exactly 1 digit after the dot
      - choose a small set of major ticks so labels do not overlap
      - does NOT change axis scales (log stays log)
    """
    fmt = mticker.FuncFormatter(lambda v, pos: f"{v:.1f}")

    ax.tick_params(axis="both", which="major", labelsize=labelsize)

    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    max_ticks = 5  # keep small to prevent overlap

    def _unique_by_label(ticks: List[float]) -> List[float]:
        seen: set[str] = set()
        out: List[float] = []
        for t in ticks:
            lab = f"{float(t):.1f}"
            if lab in seen:
                continue
            seen.add(lab)
            out.append(float(t))
        return out

    def _apply_log_ticks(axis, lo: float, hi: float) -> None:
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return
        vmin, vmax = (lo, hi) if lo <= hi else (hi, lo)
        if vmin <= 0.0 or vmax <= 0.0:
            return

        # candidate ticks at {1..9} * 10^k (safe for 1-decimal labels; no duplicate rounding)
        kmin = int(math.floor(math.log10(vmin))) - 1
        kmax = int(math.ceil(math.log10(vmax))) + 1

        ticks: List[float] = []
        for k in range(kmin, kmax + 1):
            base = 10.0 ** k
            for m in (1, 2, 3, 4, 5, 6, 7, 8, 9):
                t = m * base
                if vmin <= t <= vmax:
                    ticks.append(t)
        ticks.sort()

        # If range is narrow and would yield too few ticks, add a couple of log-space points.
        if len(ticks) < 3:
            extra = np.geomspace(vmin, vmax, num=3)
            ticks.extend([float(z) for z in extra])
            ticks.sort()

        # Remove ticks that would collide after formatting to 1 decimal.
        ticks = _unique_by_label(ticks)

        # Subsample to keep labels from overlapping.
        if len(ticks) > max_ticks:
            idx = np.linspace(0, len(ticks) - 1, max_ticks)
            idx = np.round(idx).astype(int)
            idx = np.unique(idx)
            ticks = [ticks[i] for i in idx]

        axis.set_major_locator(mticker.FixedLocator(ticks))
        axis.set_minor_locator(mticker.NullLocator())

    # x-axis
    if ax.get_xscale() == "log":
        _apply_log_ticks(ax.xaxis, *ax.get_xlim())
    else:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=max_ticks))
        ax.xaxis.set_minor_locator(mticker.NullLocator())

    # y-axis
    if ax.get_yscale() == "log":
        _apply_log_ticks(ax.yaxis, *ax.get_ylim())
    else:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=max_ticks))
        ax.yaxis.set_minor_locator(mticker.NullLocator())


# -----------------------------
# Plotting
# -----------------------------
def _plot_series(
    ax,
    x: np.ndarray,
    series: dict,
    label: str,
    *,
    marker: str,
    linestyle: str,
) -> None:
    m = np.asarray(series["mean"], dtype=float)
    q25 = np.asarray(series["q25"], dtype=float)
    q75 = np.asarray(series["q75"], dtype=float)

    mask = np.isfinite(x) & np.isfinite(m) & np.isfinite(q25) & np.isfinite(q75)
    if mask.sum() == 0:
        return

    ax.plot(
        x[mask],
        m[mask],
        marker=marker,
        linestyle=linestyle,
        linewidth=PLOT_LW,
        markersize=PLOT_MS,
        label=label,
        alpha=0.9,
    )
    ax.fill_between(x[mask], q25[mask], q75[mask], alpha=0.18, linewidth=0)


def plot_sweep_4datasets(results: List[dict], *, title: str, out_png: Path, out_pdf: Path) -> None:
    """
    Previously: one 2×2 figure (4 datasets).
    Now: one figure per dataset (4 figures total).

    Style updated to match B600 script (fonts/linewidth/markers/ticks).
    """
    out_png = Path(out_png)
    out_pdf = Path(out_pdf)
    out_dir = out_png.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    base_png = out_png.stem.replace("_4datasets", "")
    base_pdf = out_pdf.stem.replace("_4datasets", "")

    def _slug_from_dataset_name(name: str) -> str:
        name = str(name).strip()
        head = name.split()[0] if name else name
        head = head.lower().replace("com-", "").replace("email-", "")
        head = re.sub(r"[^a-z0-9]+", "_", head).strip("_")
        return head or "dataset"

    for res in results:
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

        x = np.asarray(res["x"], dtype=float)

        _plot_series(ax, x, res["ISTA"], "ISTA", marker="o", linestyle="-")
        _plot_series(ax, x, res["FISTA"], "FISTA", marker="s", linestyle="--")

        ax.set_xlabel(res["xlabel"], fontsize=32)
        if res is results[0]:
            ax.set_ylabel("Work to reach tol.", fontsize=32)
        else:
            ax.set_ylabel("")

        ax.set_xscale("log")
        ax.set_yscale("log", nonpositive="clip")

        ax.grid(True, which="both", alpha=0.3)

        # Tick style (match B600)
        ax.tick_params(axis="both", which="major", labelsize=26)
        ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext())
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_kfmt))

        ax.legend(loc="lower left", fontsize=32)

        fig.tight_layout()

        slug = _slug_from_dataset_name(res.get("dataset", ""))
        out_png_i = out_dir / f"{base_png}_{slug}.png"
        out_pdf_i = out_dir / f"{base_pdf}_{slug}.pdf"

        fig.savefig(out_png_i, dpi=220)
        fig.savefig(out_pdf_i)
        print(f"[saved] {out_png_i}")
        print(f"[saved] {out_pdf_i}")
        plt.close(fig)


# -----------------------------
# NEW: Degree distribution (per dataset)
# -----------------------------
def compute_degree_ccdf(deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute empirical CCDF of integer degrees.

    ccdf(k) = P(D >= k)

    Returns (k_vals, ccdf_vals) restricted to k>=1 where the degree has nonzero mass.
    """
    deg = np.asarray(deg, dtype=np.int64)
    if deg.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    deg = deg[deg >= 0]
    if deg.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    max_deg = int(deg.max())
    counts = np.bincount(deg, minlength=max_deg + 1)
    tail = np.cumsum(counts[::-1])[::-1]  # tail[k] = #{i : deg_i >= k}

    k_vals = np.nonzero(counts)[0]
    k_vals = k_vals[k_vals >= 1]  # log-axis friendly

    if k_vals.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    ccdf = tail[k_vals] / float(deg.size)
    return k_vals.astype(float), ccdf.astype(float)


def compute_degree_distribution_summary(deg: np.ndarray, *, dataset_name: str) -> dict:
    """Compact degree distribution summary for plotting (purely additive)."""
    deg_int = np.asarray(deg, dtype=np.int64)
    deg_int = deg_int[deg_int >= 0]
    n = int(deg_int.size)
    if n == 0:
        return {
            "dataset": dataset_name,
            "k": np.zeros(0, dtype=float),
            "ccdf": np.zeros(0, dtype=float),
            "n": 0,
            "m": 0,
            "mean_deg": float("nan"),
            "max_deg": 0,
        }

    k, ccdf = compute_degree_ccdf(deg_int)
    m = int(deg_int.sum() // 2)  # undirected edges

    return {
        "dataset": dataset_name,
        "k": k,
        "ccdf": ccdf,
        "n": n,
        "m": m,
        "mean_deg": float(deg_int.mean()),
        "max_deg": int(deg_int.max()),
    }


def plot_degree_distribution_4datasets(results: List[dict], *, title: str, out_png: Path, out_pdf: Path) -> None:
    """
    Single figure: degree CCDF for ALL datasets on one log-log plot.

    Style updated to match B600 script (fonts/linewidth/ticks).
    """
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    linestyles = ["-", "--", "-.", ":"]

    any_data = False
    for i, res in enumerate(results):
        k = np.asarray(res.get("k", []), dtype=float)
        ccdf = np.asarray(res.get("ccdf", []), dtype=float)
        ds = str(res.get("dataset", ""))

        if k.size == 0 or ccdf.size == 0:
            continue

        ax.plot(
            k,
            ccdf,
            linestyle=linestyles[i % len(linestyles)],
            linewidth=PLOT_LW,
            label=ds,
            alpha=0.9,
        )
        any_data = True

    if not any_data:
        ax.text(0.5, 0.5, "no degree data", ha="center", va="center", transform=ax.transAxes, fontsize=FS_LABEL)
        ax.axis("off")
    else:
        ax.set_xscale("log")
        ax.set_yscale("log", nonpositive="clip")

        ax.set_xlabel("degree k", fontsize=FS_LABEL)
        ax.set_ylabel("P(D ≥ k)", fontsize=FS_LABEL)

        ax.grid(True, which="both", alpha=0.3)

        ax.tick_params(axis="both", which="major", labelsize=FS_TICK)
        ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext())
        ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())

        ax.legend(loc="best", fontsize=FS_LEGEND - 7, frameon=True)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    print(f"[saved] {out_png}")
    print(f"[saved] {out_pdf}")
    plt.close(fig)


# -----------------------------
# Orkut slowdown diagnostics (why can FISTA be slower?)
# -----------------------------
def analyze_orkut_fista_slowdown(scatter_results: List[dict], *, out_dir: Path) -> None:
    """Post-process the per-seed scatter diagnostics to explain Orkut slowdowns."""
    target = None
    for res in scatter_results:
        key = str(res.get("dataset_key", ""))
        name = str(res.get("dataset", ""))
        if key == "com-Orkut" or "Orkut" in name:
            target = res
            break

    if target is None:
        print("[diagnostics] No Orkut entry found in scatter_results; skipping.")
        return

    vS = np.asarray(target.get("vol_Sstar"), dtype=float)
    vX = np.asarray(target.get("vol_extra"), dtype=float)
    vY = np.asarray(target.get("vol_extra_y"), dtype=float)

    wI = np.asarray(target.get("work_ista"), dtype=float)
    wF = np.asarray(target.get("work_fista"), dtype=float)
    itI = np.asarray(target.get("iters_ista"), dtype=float)
    itF = np.asarray(target.get("iters_fista"), dtype=float)

    outside_frac = np.asarray(target.get("outside_work_frac_fista"), dtype=float)
    max_vy = np.asarray(target.get("max_vol_y_fista"), dtype=float)

    mask = (
        np.isfinite(vS) & np.isfinite(vX) & np.isfinite(vY) &
        np.isfinite(wI) & np.isfinite(wF) & np.isfinite(itI) & np.isfinite(itF) &
        (wI > 0) & (wF > 0) & (itI > 0) & (itF > 0)
    )
    n_total = int(vS.size)
    n_valid = int(mask.sum())

    if n_valid == 0:
        print("[diagnostics] Orkut: no valid seeds where both methods reached tol; skipping.")
        return

    vS = vS[mask]
    vX = vX[mask]
    vY = vY[mask]
    wI = wI[mask]
    wF = wF[mask]
    itI = itI[mask]
    itF = itF[mask]
    outside_frac = outside_frac[mask]
    max_vy = max_vy[mask]

    work_ratio = wF / wI
    iter_ratio = itF / itI

    incI = wI / itI
    incF = wF / itF
    inc_ratio = incF / incI

    slower = wF > wI

    def _summ(x: np.ndarray) -> str:
        x = np.asarray(x, dtype=float)
        return (
            f"mean={np.mean(x):.3g}, median={np.median(x):.3g}, "
            f"q25={np.quantile(x,0.25):.3g}, q75={np.quantile(x,0.75):.3g}"
        )

    rx = vX / vS
    ry = vY / vS

    def _corr_log(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        if int(m.sum()) < 3:
            return float("nan")
        la = np.log10(a[m])
        lb = np.log10(b[m])
        return float(np.corrcoef(la, lb)[0, 1])

    corr_wr_ir = _corr_log(work_ratio, iter_ratio)
    corr_wr_pr = _corr_log(work_ratio, inc_ratio)
    corr_wr_ry = _corr_log(work_ratio, ry)

    lines = []
    lines.append("Orkut FISTA slowdown diagnostics")
    lines.append(f"  alpha={target.get('alpha'):.3g}, rho={target.get('rho'):g}, kkt_eps={target.get('kkt_eps'):g}")
    lines.append(f"  seeds: total={n_total}, valid={n_valid}")
    lines.append(f"  frac(FISTA slower in work) = {slower.mean():.3f}")
    lines.append("")
    lines.append("Work ratio (FISTA/ISTA): " + _summ(work_ratio))
    lines.append("Iter ratio (FISTA/ISTA): " + _summ(iter_ratio))
    lines.append("Avg per-iter work ratio:  " + _summ(inc_ratio))
    lines.append("")
    lines.append("Extra support ratios (volume-based):")
    lines.append("  extra_x / vol(S*): " + _summ(rx))
    lines.append("  extra_y / vol(S*): " + _summ(ry))
    lines.append("")
    lines.append("FISTA outside-work fraction (relative to S*): " + _summ(outside_frac))
    lines.append("FISTA max vol(supp(y_k)) over iterations:      " + _summ(max_vy))
    lines.append("")
    lines.append("Correlations (log-space):")
    lines.append(f"  corr(log work_ratio, log iter_ratio) = {corr_wr_ir:.3f}")
    lines.append(f"  corr(log work_ratio, log inc_ratio)  = {corr_wr_pr:.3f}")
    lines.append(f"  corr(log work_ratio, log (extra_y/volS)) = {corr_wr_ry:.3f}")

    stats_txt = Path(out_dir) / "orkut_fista_slowdown_stats.txt"
    stats_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"[saved] {stats_txt}")

    # ---- Plot: iteration-count vs per-iteration-cost tradeoff ----
    try:
        fig1, ax1 = plt.subplots(figsize=PLOT_FIGSIZE)

        xtr = np.asarray(iter_ratio, dtype=float)
        ytr = np.asarray(inc_ratio, dtype=float)
        m_tr = np.isfinite(xtr) & np.isfinite(ytr) & (xtr > 0) & (ytr > 0)
        xtr = xtr[m_tr]
        ytr = ytr[m_tr]
        slower_tr = slower[m_tr]

        ax1.scatter(xtr[~slower_tr], ytr[~slower_tr], marker="o", s=160, alpha=0.9, label="FISTA ≤ ISTA (work)")
        ax1.scatter(xtr[slower_tr],  ytr[slower_tr],  marker="x", s=220, linewidths=3.5, alpha=0.9, label="FISTA > ISTA (work)")

        ax1.axvline(1.0, linestyle="--", linewidth=2.0)
        ax1.axhline(1.0, linestyle="--", linewidth=2.0)

        ax1.set_xscale("log")
        ax1.set_yscale("log", nonpositive="clip")

        ax1.set_xlabel("Iteration ratio", fontsize=FS_LABEL)
        ax1.set_ylabel("Per-iteration work ratio", fontsize=FS_LABEL)

        ax1.grid(True, which="both", alpha=0.3)

        # ✅ scatter ticks: 1 decimal, and limited count (no overlap)
        _scatter_decimal_ticks_1(ax1, labelsize=FS_TICK)

        ax1.legend(loc="upper left", fontsize=FS_LEGEND)

        fig1.tight_layout(rect=[0, 0, 1, 0.95])

        out_png = out_dir / "orkut_iters_vs_cost_tradeoff.png"
        out_pdf = out_dir / "orkut_iters_vs_cost_tradeoff.pdf"
        fig1.savefig(out_png, dpi=220)
        fig1.savefig(out_pdf)
        print(f"[saved] {out_png}")
        print(f"[saved] {out_pdf}")
        plt.close(fig1)
    except Exception as e:
        print(f"[diagnostics] Orkut tradeoff plot failed: {e}")


def plot_iters_vs_cost_tradeoff_4datasets(
    results: List[dict],
    *,
    out_png: Path,
    out_pdf: Path,
) -> None:
    """
    One plot per dataset (4 total).
    """
    out_png = Path(out_png)
    out_pdf = Path(out_pdf)
    out_dir = out_png.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    base_png = out_png.stem.replace("_4datasets", "")
    base_pdf = out_pdf.stem.replace("_4datasets", "")

    def _slug(s: str) -> str:
        s = str(s).lower()
        s = s.replace("com-", "").replace("email-", "")
        s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
        return s or "dataset"

    for res in results:
        wI = np.asarray(res.get("work_ista"), dtype=float)
        wF = np.asarray(res.get("work_fista"), dtype=float)
        itI = np.asarray(res.get("iters_ista"), dtype=float)
        itF = np.asarray(res.get("iters_fista"), dtype=float)

        dataset_name = str(res.get("dataset", ""))
        dataset_key = str(res.get("dataset_key", dataset_name))
        slug = _slug(dataset_key if dataset_key else dataset_name)

        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

        mask = (
            np.isfinite(wI) & np.isfinite(wF) &
            np.isfinite(itI) & np.isfinite(itF) &
            (wI > 0) & (wF > 0) &
            (itI > 0) & (itF > 0)
        )

        if int(mask.sum()) == 0:
            ax.text(0.5, 0.5, "no valid seeds", ha="center", va="center", transform=ax.transAxes, fontsize=FS_LABEL)
            ax.axis("off")
        else:
            wI_m = wI[mask]
            wF_m = wF[mask]
            itI_m = itI[mask]
            itF_m = itF[mask]

            iter_ratio = itF_m / itI_m
            inc_ratio = (wF_m / itF_m) / (wI_m / itI_m)
            slower = wF_m > wI_m

            m = np.isfinite(iter_ratio) & np.isfinite(inc_ratio) & (iter_ratio > 0) & (inc_ratio > 0)
            iter_ratio = iter_ratio[m]
            inc_ratio = inc_ratio[m]
            slower = slower[m]

            ax.scatter(iter_ratio[~slower], inc_ratio[~slower], marker="o", s=160, alpha=0.9, label="FISTA ≤ ISTA (work)")
            ax.scatter(iter_ratio[slower], inc_ratio[slower], marker="x", s=220, linewidths=3.5, alpha=0.9, label="FISTA > ISTA (work)")

            ax.axvline(1.0, linestyle="--", linewidth=3.0)
            ax.axhline(1.0, linestyle="--", linewidth=3.0)

            ax.set_xscale("log")
            ax.set_yscale("log", nonpositive="clip")

            ax.set_xlabel("Iteration ratio", fontsize=FS_LABEL)
            #if res is results[0]:
            ax.set_ylabel("Per-iteration work ratio", fontsize=FS_LABEL)
            #else:
            #    ax.set_ylabel("")

            ax.grid(True, which="both", alpha=0.3)

            # ✅ scatter ticks: 1 decimal, and limited count (no overlap)
            _scatter_decimal_ticks_1(ax, labelsize=FS_TICK)

            ax.legend(loc="upper left", fontsize=FS_LEGEND)

        fig.tight_layout()

        out_png_i = out_dir / f"{base_png}_{slug}.png"
        out_pdf_i = out_dir / f"{base_pdf}_{slug}.pdf"
        fig.savefig(out_png_i, dpi=220)
        fig.savefig(out_pdf_i)
        print(f"[saved] {out_png_i}")
        print(f"[saved] {out_pdf_i}")
        plt.close(fig)


def _cache_paths(out_dir: Path) -> dict:
    return {
        "alpha": out_dir / "cache_results_alpha.npz",
        "eps": out_dir / "cache_results_eps.npz",
        "rho": out_dir / "cache_results_rho.npz",
        "scatter": out_dir / "cache_scatter_results.npz",
        "degree": out_dir / "cache_degree_results.npz",
        "meta": out_dir / "cache_meta.txt",
    }


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data_snap", help="Where to download/extract datasets")
    p.add_argument("--out-dir", type=str, default="outputs", help="Where to write plots")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip all downloads/sweeps; load cached results from out-dir and only render plots")

    p.add_argument("--rng-seed", type=int, default=123, help="Global RNG seed (entire script)")
    p.add_argument("--num-seeds", type=int, default=300, help="Random seed nodes per dataset (work averages)")
    p.add_argument("--max-iter", type=int, default=50000, help="Large max iteration cap for all runs")

    p.add_argument("--alpha0", type=float, default=0.2, help="Base alpha used in epsilon/rho sweeps")
    p.add_argument("--rho0", type=float, default=1e-4, help="Base rho used in alpha/epsilon sweeps")
    p.add_argument("--kkt-eps", type=float, default=1e-8, help="Fixed KKT tolerance for alpha and rho sweeps")

    p.add_argument("--alpha-min", type=float, default=1e-3, help="Min alpha in alpha sweep")
    p.add_argument("--alpha-max", type=float, default=0.9, help="Max alpha in alpha sweep")
    p.add_argument("--alpha-points", type=int, default=15, help="Number of alpha grid points")

    p.add_argument("--eps-min", type=float, default=1e-8, help="Min epsilon in epsilon sweep")
    p.add_argument("--eps-max", type=float, default=1e-2, help="Max epsilon in epsilon sweep")
    p.add_argument("--eps-points", type=int, default=15, help="Number of epsilon grid points")

    p.add_argument("--rho-min", type=float, default=1e-6, help="Min rho in rho sweep")
    p.add_argument("--rho-max", type=float, default=1e-2, help="Max rho in rho sweep")
    p.add_argument("--rho-points", type=int, default=15, help="Number of rho grid points")

    args = p.parse_args()

    np.random.seed(int(args.rng_seed))
    rng_master = np.random.default_rng(int(args.rng_seed))

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = _cache_paths(out_dir)

    if args.plot_only:
        if not (cache["alpha"].exists() and cache["eps"].exists() and cache["rho"].exists()
                and cache["scatter"].exists() and cache["degree"].exists()):
            raise FileNotFoundError(
                "plot-only requested, but cache files were not found in out-dir.\n"
                "Run once normally to generate caches, then re-run with --plot-only.\n"
                f"Expected:\n"
                f"  {cache['alpha']}\n  {cache['eps']}\n  {cache['rho']}\n  {cache['scatter']}\n  {cache['degree']}\n"
            )

        results_alpha = list(np.load(cache["alpha"], allow_pickle=True)["results"])
        results_eps = list(np.load(cache["eps"], allow_pickle=True)["results"])
        results_rho = list(np.load(cache["rho"], allow_pickle=True)["results"])
        scatter_results = list(np.load(cache["scatter"], allow_pickle=True)["results"])
        degree_results = list(np.load(cache["degree"], allow_pickle=True)["results"])

        alpha_grid = np.logspace(math.log10(float(args.alpha_min)), math.log10(float(args.alpha_max)), int(args.alpha_points))

        plot_sweep_4datasets(
            results_alpha,
            title=f"Work vs alpha (rho={args.rho0:g}, KKT tol={args.kkt_eps:g}, max_iter={args.max_iter})",
            out_png=out_dir / "work_vs_alpha_4datasets.png",
            out_pdf=out_dir / "work_vs_alpha_4datasets.pdf",
        )
        plot_sweep_4datasets(
            results_eps,
            title=f"Work vs epsilon (KKT tol) (alpha={args.alpha0:g}, rho={args.rho0:g}, max_iter={args.max_iter})",
            out_png=out_dir / "work_vs_epsilon_4datasets.png",
            out_pdf=out_dir / "work_vs_epsilon_4datasets.pdf",
        )
        plot_sweep_4datasets(
            results_rho,
            title=f"Work vs rho (alpha={args.alpha0:g}, KKT tol={args.kkt_eps:g}, max_iter={args.max_iter})",
            out_png=out_dir / "work_vs_rho_4datasets.png",
            out_pdf=out_dir / "work_vs_rho_4datasets.pdf",
        )

        analyze_orkut_fista_slowdown(scatter_results, out_dir=out_dir)

        plot_iters_vs_cost_tradeoff_4datasets(
            scatter_results,
            out_png=out_dir / "iters_vs_cost_tradeoff_4datasets.png",
            out_pdf=out_dir / "iters_vs_cost_tradeoff_4datasets.pdf",
        )

        plot_degree_distribution_4datasets(
            degree_results,
            title="Degree distribution (CCDF) for each dataset",
            out_png=out_dir / "degree_distribution_4datasets.png",
            out_pdf=out_dir / "degree_distribution_4datasets.pdf",
        )

        return

    alpha_grid = np.logspace(math.log10(float(args.alpha_min)), math.log10(float(args.alpha_max)), int(args.alpha_points))
    eps_grid = np.logspace(math.log10(float(args.eps_min)), math.log10(float(args.eps_max)), int(args.eps_points))
    rho_grid = np.logspace(math.log10(float(args.rho_min)), math.log10(float(args.rho_max)), int(args.rho_points))

    results_alpha: List[dict] = []
    results_eps: List[dict] = []
    results_rho: List[dict] = []
    scatter_results: List[dict] = []
    degree_results: List[dict] = []

    for spec in DATASETS:
        gz_path = data_dir / (spec.filename + ".gz")
        txt_path = data_dir / spec.filename
        download_if_needed(spec.url_gz, gz_path)
        gunzip_if_needed(gz_path, txt_path)

        A = load_undirected_unweighted_graph_from_edgelist(txt_path)
        print(f"[loaded] {spec.key:10s}  n={A.shape[0]:,}  m={A.nnz//2:,} (undirected edges)")

        d = np.asarray(A.sum(axis=1)).ravel().astype(np.float64)
        sqrt_d = np.sqrt(d, dtype=np.float64)
        inv_sqrt_d = np.zeros_like(sqrt_d)
        mask = sqrt_d > 0
        inv_sqrt_d[mask] = 1.0 / sqrt_d[mask]

        degree_results.append(
            compute_degree_distribution_summary(d, dataset_name=spec.name)
        )

        dataset_rng_seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(dataset_rng_seed)
        candidates = np.where(d > 0)[0]
        if candidates.size == 0:
            raise ValueError(f"{spec.name}: all nodes have degree 0?")
        replace = candidates.size < int(args.num_seeds)
        seeds = rng.choice(candidates, size=int(args.num_seeds), replace=replace)
        print(f"[{spec.name}] sampled {seeds.size} seed nodes (dataset_rng_seed={dataset_rng_seed})")

        results_alpha.append(
            sweep_alpha(
                A, d, inv_sqrt_d, sqrt_d,
                dataset_name=spec.name,
                seeds=seeds,
                alpha_grid=alpha_grid,
                rho=float(args.rho0),
                kkt_eps=float(args.kkt_eps),
                max_iter=int(args.max_iter),
            )
        )

        results_eps.append(
            sweep_epsilon(
                A, d, inv_sqrt_d, sqrt_d,
                dataset_name=spec.name,
                seeds=seeds,
                eps_grid=eps_grid,
                alpha=float(args.alpha0),
                rho=float(args.rho0),
                max_iter=int(args.max_iter),
            )
        )

        results_rho.append(
            sweep_rho(
                A, d, inv_sqrt_d, sqrt_d,
                dataset_name=spec.name,
                seeds=seeds,
                rho_grid=rho_grid,
                alpha=float(args.alpha0),
                kkt_eps=float(args.kkt_eps),
                max_iter=int(args.max_iter),
            )
        )

        scatter_res = scatter_volumes_Sstar_vs_fista_extra(
            A, d, inv_sqrt_d, sqrt_d,
            dataset_key=spec.key,
            dataset_name=spec.name,
            seeds=seeds,
            alpha=float(alpha_grid[0]),
            rho=float(args.rho0),
            kkt_eps=float(args.kkt_eps),
            max_iter=int(args.max_iter),
        )
        scatter_results.append(scatter_res)

        del A

    np.savez(cache["alpha"], results=np.array(results_alpha, dtype=object))
    np.savez(cache["eps"], results=np.array(results_eps, dtype=object))
    np.savez(cache["rho"], results=np.array(results_rho, dtype=object))
    np.savez(cache["scatter"], results=np.array(scatter_results, dtype=object))
    np.savez(cache["degree"], results=np.array(degree_results, dtype=object))
    cache["meta"].write_text(
        f"rng_seed={args.rng_seed}\nnum_seeds={args.num_seeds}\nmax_iter={args.max_iter}\n"
        f"alpha0={args.alpha0}\nrho0={args.rho0}\nkkt_eps={args.kkt_eps}\n",
        encoding="utf-8",
    )
    print(f"[cache] wrote cached results to: {out_dir}")

    plot_sweep_4datasets(
        results_alpha,
        title=f"Work vs alpha (rho={args.rho0:g}, KKT tol={args.kkt_eps:g}, max_iter={args.max_iter})",
        out_png=out_dir / "work_vs_alpha_4datasets.png",
        out_pdf=out_dir / "work_vs_alpha_4datasets.pdf",
    )
    plot_sweep_4datasets(
        results_eps,
        title=f"Work vs epsilon (KKT tol) (alpha={args.alpha0:g}, rho={args.rho0:g}, max_iter={args.max_iter})",
        out_png=out_dir / "work_vs_epsilon_4datasets.png",
        out_pdf=out_dir / "work_vs_epsilon_4datasets.pdf",
    )
    plot_sweep_4datasets(
        results_rho,
        title=f"Work vs rho (alpha={args.alpha0:g}, KKT tol={args.kkt_eps:g}, max_iter={args.max_iter})",
        out_png=out_dir / "work_vs_rho_4datasets.png",
        out_pdf=out_dir / "work_vs_rho_4datasets.pdf",
    )

    analyze_orkut_fista_slowdown(scatter_results, out_dir=out_dir)

    plot_iters_vs_cost_tradeoff_4datasets(
        scatter_results,
        out_png=out_dir / "iters_vs_cost_tradeoff_4datasets.png",
        out_pdf=out_dir / "iters_vs_cost_tradeoff_4datasets.pdf",
    )

    plot_degree_distribution_4datasets(
        degree_results,
        title="Degree distribution (CCDF) for each dataset",
        out_png=out_dir / "degree_distribution_4datasets.png",
        out_pdf=out_dir / "degree_distribution_4datasets.pdf",
    )


if __name__ == "__main__":
    main()