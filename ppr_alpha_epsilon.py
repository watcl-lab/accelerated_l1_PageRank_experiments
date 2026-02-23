#!/usr/bin/env python3
"""
UNWEIGHTED-ONLY VERSION (eta > 0, no-percolation enforced, small-alpha start).

(A) Alpha sweep: MODIFIED ONLY HERE.
    - still UNWEIGHTED only
    - still enforces no-percolation
    - still requires eta>0
    - now AUTO-TUNES the alpha-sweep-only unweighted graph parameters to create a regime
      where FISTA is slower than ISTA in WORK over a wide small-alpha window).

(B) Epsilon sweep: UNCHANGED (same cached unweighted graph builder as before).

Outputs:
- B600_work_vs_alpha.png / .pdf
- B600_work_vs_epsilon.png / .pdf

Dependencies: numpy, scipy, matplotlib
"""

import math
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# -----------------------------
# Graph construction helpers (epsilon sweep uses these unchanged)
# -----------------------------
def circulant_edges(nodes, k):
    n = len(nodes)
    if k == 0:
        return []
    assert k % 2 == 0 and 0 < k < n
    half = k // 2
    edges = set()
    for idx, u in enumerate(nodes):
        for j in range(1, half + 1):
            v = nodes[(idx + j) % n]
            a, b = (u, v) if u < v else (v, u)
            edges.add((a, b))
    return list(edges)


# -----------------------------
# Graph construction (ORIGINAL, UNWEIGHTED) — used by epsilon sweep (UNCHANGED)
# -----------------------------
def build_graph_fixed_core_degree(
    core_size: int,
    boundary_size: int,
    ext_size: int,
    seed: int = 0,
    c_boundary_per_core: int = 20,
    deg_b_internal: int = 82,
    deg_ext: int = 998,
):
    assert 0 <= seed < core_size
    assert deg_ext % 2 == 0 and deg_ext < ext_size

    core = list(range(core_size))
    boundary = list(range(core_size, core_size + boundary_size))
    exterior = list(range(core_size + boundary_size, core_size + boundary_size + ext_size))
    n = core_size + boundary_size + ext_size

    edges = []
    # core clique
    for i in range(core_size):
        for j in range(i + 1, core_size):
            edges.append((i, j))

    # exterior internal
    edges += circulant_edges(exterior, deg_ext)

    if boundary_size > 0:
        # core-boundary edges: each core has c_boundary_per_core boundary neighbors
        for u in core:
            for j in range(c_boundary_per_core):
                b = boundary[(u * c_boundary_per_core + j) % boundary_size]
                a, c = (u, b) if u < b else (b, u)
                edges.append((a, c))

        # boundary internal circulant
        deg_b = min(deg_b_internal, boundary_size - 1)
        if deg_b % 2 == 1:
            deg_b -= 1
        if deg_b >= 2:
            edges += circulant_edges(boundary, deg_b)

        # exterior-to-boundary (1 per exterior node)  <-- IMPORTANT: epsilon sweep uses this
        for idx, x in enumerate(exterior):
            b = boundary[idx % boundary_size]
            a, c = (x, b) if x < b else (b, x)
            edges.append((a, c))

    # adjacency
    if edges:
        rows = np.fromiter((e[0] for e in edges), dtype=int)
        cols = np.fromiter((e[1] for e in edges), dtype=int)
        data = np.ones(len(edges), dtype=float)
        A = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
        A = (A + A.T).tocsr()
        A.sum_duplicates()
    else:
        A = sp.csr_matrix((n, n), dtype=float)

    d = np.array(A.sum(axis=1)).ravel()
    inv_sqrt_d = 1.0 / np.sqrt(d)
    P = (sp.diags(inv_sqrt_d) @ A @ sp.diags(inv_sqrt_d)).tocsr()

    S = set(core)
    B = set(boundary)
    Ext = set(exterior)
    return P, d, S, B, Ext, seed


@lru_cache(maxsize=None)
def build_graph_cached(core_size, boundary_size, ext_size,
                       seed, c_boundary_per_core, deg_b_internal, deg_ext):
    # Epsilon sweep uses this exact builder/caching (UNCHANGED)
    return build_graph_fixed_core_degree(
        core_size=core_size,
        boundary_size=boundary_size,
        ext_size=ext_size,
        seed=seed,
        c_boundary_per_core=c_boundary_per_core,
        deg_b_internal=deg_b_internal,
        deg_ext=deg_ext,
    )


# -----------------------------
# Optimization helpers (UNCHANGED)
# -----------------------------
def beta_sc(alpha: float) -> float:
    return (1.0 - math.sqrt(alpha)) / (1.0 + math.sqrt(alpha))


def soft_threshold(u: np.ndarray, tau: np.ndarray) -> np.ndarray:
    return np.sign(u) * np.maximum(np.abs(u) - tau, 0.0)


def ppr_objective(x, Px, d, seed, alpha, rho) -> float:
    c = (1.0 - alpha) / 2.0
    qdiag = (1.0 + alpha) / 2.0
    xQx = qdiag * float(x @ x) - c * float(x @ Px)
    bTx = alpha * (x[seed] / math.sqrt(d[seed]))
    g = rho * alpha * float((np.sqrt(d) * np.abs(x)).sum())
    return 0.5 * xQx - bTx + g


# ============================================================
# NEW stopping criterion: prox-gradient fixed-point residual
# ============================================================
def prox_map_T(x: np.ndarray, Px: np.ndarray, *, c: float, tau: np.ndarray, b_seed: float, seed: int) -> np.ndarray:
    """
    Prox-gradient fixed-point mapping with unit step:
        T(x) = prox_g( x - ∇f(x) )

    For this objective, with Px = P x:
        x - ∇f(x) = c * (x + P x) + b_seed * e_seed
        prox_g is weighted soft-threshold with tau = rho*alpha*sqrt(d)
    """
    u = c * (x + Px)
    u[seed] += b_seed
    return soft_threshold(u, tau)


def fixed_point_residual_inf(x: np.ndarray, Tx: np.ndarray) -> float:
    """r(x) = ||x - T(x)||_∞."""
    return float(np.max(np.abs(x - Tx)))


def cum_work_update(work, d, y, x_next):
    return work + float(d[y != 0].sum()) + float(d[x_next != 0].sum())


def first_work_below(objs, works, target):
    idx = np.where(objs <= target)[0]
    if idx.size == 0:
        return np.nan
    return float(works[idx[0]])


def run_ista_history(P, d, seed, alpha, rho, max_iter):
    c = (1.0 - alpha) / 2.0
    tau = rho * alpha * np.sqrt(d)
    b_seed = alpha / math.sqrt(d[seed])

    n = d.size
    x = np.zeros(n)
    Px = np.zeros(n)

    work = 0.0
    # NOTE: "objs" now stores the fixed-point residual r(x_k) = ||x_k - T(x_k)||_inf
    objs = np.empty(max_iter + 1, dtype=float)
    works = np.empty(max_iter + 1, dtype=float)

    # residual at x0
    Tx = prox_map_T(x, Px, c=c, tau=tau, b_seed=b_seed, seed=seed)
    objs[0] = fixed_point_residual_inf(x, Tx)
    works[0] = 0.0

    for k in range(1, max_iter + 1):
        y = x

        # ISTA update: x_next = T(x)
        x_next = Tx
        Px_next = P.dot(x_next)

        work = cum_work_update(work, d, y, x_next)

        x, Px = x_next, Px_next

        # residual at current iterate x_k
        Tx = prox_map_T(x, Px, c=c, tau=tau, b_seed=b_seed, seed=seed)
        objs[k] = fixed_point_residual_inf(x, Tx)
        works[k] = work

    return objs, works


def run_fista_until(P, d, seed, alpha, rho, beta, target, max_iter):
    c = (1.0 - alpha) / 2.0
    tau = rho * alpha * np.sqrt(d)
    b_seed = alpha / math.sqrt(d[seed])

    n = d.size
    x_prev = np.zeros(n)
    x = np.zeros(n)
    Px_prev = np.zeros(n)
    Px = np.zeros(n)

    work = 0.0

    # residual at x0
    Tx = prox_map_T(x, Px, c=c, tau=tau, b_seed=b_seed, seed=seed)
    r = fixed_point_residual_inf(x, Tx)
    if r <= target:
        return 0, 0.0, r, True

    for k in range(1, max_iter + 1):
        y = x + beta * (x - x_prev)
        Py = Px + beta * (Px - Px_prev)

        x_next = prox_map_T(y, Py, c=c, tau=tau, b_seed=b_seed, seed=seed)
        Px_next = P.dot(x_next)

        work = cum_work_update(work, d, y, x_next)

        x_prev, x = x, x_next
        Px_prev, Px = Px, Px_next

        # residual at current iterate x_k (same mapping as ISTA)
        Tx = prox_map_T(x, Px, c=c, tau=tau, b_seed=b_seed, seed=seed)
        r = fixed_point_residual_inf(x, Tx)
        if r <= target:
            return k, work, r, True

    return max_iter, work, r, False


def run_fista_history(P, d, seed, alpha, rho, beta, max_iter):
    c = (1.0 - alpha) / 2.0
    tau = rho * alpha * np.sqrt(d)
    b_seed = alpha / math.sqrt(d[seed])

    n = d.size
    x_prev = np.zeros(n)
    x = np.zeros(n)
    Px_prev = np.zeros(n)
    Px = np.zeros(n)

    work = 0.0
    # NOTE: "objs" now stores the fixed-point residual r(x_k) = ||x_k - T(x_k)||_inf
    objs = np.empty(max_iter + 1, dtype=float)
    works = np.empty(max_iter + 1, dtype=float)

    # residual at x0
    Tx = prox_map_T(x, Px, c=c, tau=tau, b_seed=b_seed, seed=seed)
    objs[0] = fixed_point_residual_inf(x, Tx)
    works[0] = 0.0

    for k in range(1, max_iter + 1):
        y = x + beta * (x - x_prev)
        Py = Px + beta * (Px - Px_prev)

        x_next = prox_map_T(y, Py, c=c, tau=tau, b_seed=b_seed, seed=seed)
        Px_next = P.dot(x_next)

        work = cum_work_update(work, d, y, x_next)

        x_prev, x = x, x_next
        Px_prev, Px = Px, Px_next

        # residual at current iterate x_k
        Tx = prox_map_T(x, Px, c=c, tau=tau, b_seed=b_seed, seed=seed)
        objs[k] = fixed_point_residual_inf(x, Tx)
        works[k] = work

    return objs, works


# -----------------------------
# No-percolation diagnostic (UNCHANGED)
# -----------------------------
def nopercolation_LHS(alpha, beta, eta, d_v, d_min):
    return ((1.0 - alpha) / 2.0) * eta * (3.0 + 2.0 * beta) / math.sqrt(d_v * d_min)


# =============================================================
# ALPHA SWEEP ONLY: UNWEIGHTED operator (eta>0), + AUTO-TUNING
# =============================================================

def choose_ext_size_for_nopercolation(
    *,
    alpha_start: float,
    rho: float,
    d_v: float,
    d_min: float,
    safety: float = 1.05,
    ext_max: int = 2_000_000,
) -> int:
    """
    Choose ext_size so that with an UNWEIGHTED COMPLETE exterior (degree ext_size-1),
    any exterior node that has ONE boundary neighbor has exposure:
        eta = 1 / ext_size   (>0)
    and the no-percolation inequality holds at alpha_start.

    If it holds at alpha_start, it holds for all alpha >= alpha_start.
    """
    beta0 = beta_sc(alpha_start)
    numer = ((1.0 - alpha_start) / 2.0) * (3.0 + 2.0 * beta0)
    denom = (rho * alpha_start) * math.sqrt(d_v * d_min)
    req = numer / denom  # need ext_size >= req since eta = 1/ext_size

    ext_size = int(math.ceil(float(safety) * float(req)))
    ext_size = max(ext_size, 3)

    if ext_size > ext_max:
        raise RuntimeError(
            f"[alpha sweep] Required ext_size≈{ext_size} exceeds ext_max={ext_max}. "
            f"Try increasing alpha_start, increasing rho, or increasing degrees."
        )
    return ext_size


@lru_cache(maxsize=None)
def _boundary_internal_adj(boundary_size: int, deg_b_internal: int) -> sp.csr_matrix:
    """
    UNWEIGHTED adjacency of the boundary circulant (shape BxB) as CSR.
    """
    B = boundary_size
    if B <= 1:
        return sp.csr_matrix((B, B), dtype=float)

    deg_b = min(int(deg_b_internal), B - 1)
    if deg_b % 2 == 1:
        deg_b -= 1
    if deg_b < 2:
        return sp.csr_matrix((B, B), dtype=float)

    nodes = list(range(B))
    edges = circulant_edges(nodes, deg_b)
    rows = np.fromiter((e[0] for e in edges), dtype=int)
    cols = np.fromiter((e[1] for e in edges), dtype=int)
    data = np.ones(len(edges), dtype=float)
    A = sp.coo_matrix((data, (rows, cols)), shape=(B, B))
    A = (A + A.T).tocsr()
    A.sum_duplicates()
    return A


@lru_cache(maxsize=None)
def _core_boundary_bipartite(core_size: int, boundary_size: int, c_boundary_per_core: int) -> sp.csr_matrix:
    """
    UNWEIGHTED core-to-boundary adjacency (shape core_size x B) as CSR,
    using the SAME round-robin pattern as your original construction.
    NOTE: duplicates can appear when c_boundary_per_core > B; we cap in caller.
    """
    C = int(c_boundary_per_core)
    if core_size <= 0 or boundary_size <= 0 or C <= 0:
        return sp.csr_matrix((core_size, boundary_size), dtype=float)

    rows = np.repeat(np.arange(core_size, dtype=int), C)
    u = np.arange(core_size, dtype=int)[:, None]
    j = np.arange(C, dtype=int)[None, :]
    cols = (u * C + j) % boundary_size
    cols = cols.reshape(-1)

    data = np.ones(rows.shape[0], dtype=float)
    M = sp.coo_matrix((data, (rows, cols)), shape=(core_size, boundary_size))
    M = M.tocsr()
    M.sum_duplicates()
    return M


def build_alpha_sweep_operator_unweighted(
    *,
    core_size: int,
    boundary_size: int,
    ext_size: int,
    seed: int,
    c_boundary_per_core: int,
    deg_b_internal: int,
    ext_bnd_edges: int,
):
    """
    Alpha-sweep-only UNWEIGHTED operator P = D^{-1/2} A D^{-1/2}.

    Blocks (UNWEIGHTED):
      - core: clique
      - core-boundary: round-robin (c_boundary_per_core per core node; capped at B)
      - boundary internal: circulant degree deg_b_internal
      - exterior internal: COMPLETE graph on ext_size nodes (implicit)
      - boundary-exterior: ONLY the first ext_bnd_edges exterior nodes each have 1 boundary neighbor
                           (round-robin); the rest have 0 boundary neighbors.

    eta is the *maximum* exposure among exterior nodes, which equals 1/ext_size as long as ext_bnd_edges>=1.
    """
    if not (0 <= seed < core_size):
        raise ValueError("seed must be a core node index.")
    if ext_bnd_edges <= 0:
        raise ValueError("ext_bnd_edges must be >= 1 to ensure eta>0.")
    B = int(boundary_size)
    # cap core-boundary fanout at B (distinct neighbors)
    C = int(min(max(0, c_boundary_per_core), B))
    ext_bnd_edges = int(min(ext_bnd_edges, ext_size))
    if C == 0:
        raise ValueError("c_boundary_per_core must be >= 1 for alpha sweep (or increase it).")

    n = core_size + B + ext_size

    CB = _core_boundary_bipartite(core_size, B, C)
    cb_counts = np.array(CB.sum(axis=0)).ravel()

    A_bb = _boundary_internal_adj(B, deg_b_internal)
    deg_b_used = int(A_bb[0].sum()) if B > 0 else 0

    m = int(ext_bnd_edges)
    ext_to_bnd = (np.arange(m, dtype=np.int32) % B)
    be_counts = np.bincount(ext_to_bnd, minlength=B).astype(float)

    # Degrees (UNWEIGHTED)
    d_core = float((core_size - 1) + C)
    d_bnd = float(deg_b_used) + cb_counts + be_counts

    # Exterior: complete graph degree (ext_size-1), plus +1 for the m nodes attached to boundary
    d_ext = (ext_size - 1) * np.ones(ext_size, dtype=float)
    d_ext[:m] += 1.0

    d = np.empty(n, dtype=float)
    d[:core_size] = d_core
    d[core_size:core_size + B] = d_bnd
    d[core_size + B:] = d_ext

    inv_sqrt_d = 1.0 / np.sqrt(d)

    eta = 1.0 / float(ext_size)  # >0

    c0, c1 = 0, core_size
    b0, b1 = core_size, core_size + B
    e0, e1 = b1, n

    def A_matvec(v: np.ndarray) -> np.ndarray:
        v_core = v[c0:c1]
        v_bnd  = v[b0:b1]
        v_ext  = v[e0:e1]

        # core clique
        s_core = float(v_core.sum())
        w_core = (s_core - v_core)

        # boundary internal
        w_bnd = A_bb.dot(v_bnd)

        # exterior internal COMPLETE graph (implicit)
        s_ext = float(v_ext.sum())
        w_ext = (s_ext - v_ext)

        # core <-> boundary
        w_core = w_core + CB.dot(v_bnd)
        w_bnd  = w_bnd  + CB.T.dot(v_core)

        # boundary <-> exterior (only first m exterior nodes)
        w_bnd = w_bnd + np.bincount(ext_to_bnd, weights=v_ext[:m], minlength=B)
        w_ext[:m] = w_ext[:m] + v_bnd[ext_to_bnd]

        out = np.empty_like(v)
        out[c0:c1] = w_core
        out[b0:b1] = w_bnd
        out[e0:e1] = w_ext
        return out

    def P_matvec(x: np.ndarray) -> np.ndarray:
        v = inv_sqrt_d * x
        w = A_matvec(v)
        return inv_sqrt_d * w

    P = spla.LinearOperator((n, n), matvec=P_matvec, dtype=float)
    return P, d, float(eta), seed


def _score_candidate_on_window(
    *,
    P,
    d,
    eta,
    seed,
    rho,
    eps_gap,
    alpha_list,
    ista_cap,
    log_factor,
    fista_cap,
):
    """
    Evaluate a candidate alpha-sweep graph on a list of alphas using the SAME alpha-sweep protocol:
      - ISTA reference history length ista_ref = min(ista_cap, ceil(log_factor/alpha))
      - measure work for ISTA and FISTA to reach the fixed-point residual tolerance:
            r(x_k) = ||x_k - T(x_k)||_inf  <= eps_gap
        (FISTA max fista_cap)
    Returns: (fraction_FISTA_slower, mean_log_ratio, details)
    """
    ratios = []
    slower = 0
    total = 0
    details = []

    d_v = float(d[seed])
    d_min = float(d.min())

    for alpha in alpha_list:
        alpha = float(alpha)
        beta = beta_sc(alpha)

        lhs = nopercolation_LHS(alpha, beta, float(eta), d_v, d_min)
        rhs = rho * alpha
        if lhs > rhs:
            # invalid candidate (violates no-perc in the window)
            return -1.0, -np.inf, [("violates_no_perc", alpha, lhs, rhs)]

        ista_ref = int(min(ista_cap, max(50, math.ceil(log_factor / alpha))))
        objs_I, works_I = run_ista_history(P, d, seed, alpha, rho, max_iter=ista_ref)

        # Termination target is the KKT / prox-gradient tolerance (absolute)
        target = float(eps_gap)
        wI = first_work_below(objs_I, works_I, target)

        itF, wF, rF, reached = run_fista_until(P, d, seed, alpha, rho, beta, target, max_iter=fista_cap)

        if not np.isfinite(wI) or wI <= 0 or (not reached) or (not np.isfinite(wF)):
            ratio = np.nan
        else:
            ratio = float(wF / wI)

        total += 1
        if np.isfinite(ratio):
            ratios.append(ratio)
            if ratio > 1.0:
                slower += 1

        details.append((alpha, ista_ref, wI, wF if reached else np.nan, reached, lhs, rhs, ratio))

    if len(ratios) == 0:
        return 0.0, -np.inf, details

    frac = float(slower) / float(total)
    mean_log = float(np.mean(np.log(np.maximum(1e-300, np.array(ratios)))))
    return frac, mean_log, details


def _auto_tune_alpha_sweep_graph(
    *,
    alpha_start,
    alpha_window_hi,
    rho,
    eps_gap,
    core_size,
    B,
    seed,
    ext_safety,
    ext_max,
    ista_cap,
    log_factor,
    fista_cap,
    # user-provided baselines
    c_boundary_per_core_baseline,
    ext_bnd_edges_target_baseline,
    deg_b_internal_base,
):
    """
    Alpha-sweep-only auto-tuner.
    Searches over UNWEIGHTED parameters to find a graph where FISTA is slower in WORK
    over a calibration alpha list in [alpha_start, alpha_window_hi].
    """
    # calibration grid inside the window (log-spaced across entire range)
    n_calib = 12
    calib = np.logspace(np.log10(alpha_start), np.log10(alpha_window_hi), n_calib)
    calib = np.unique(np.clip(calib.astype(float), alpha_start, alpha_window_hi))

    # Candidate lists (tight-ish but actually useful)
    # We expand around user baselines while staying within [1, B] for c.
    c0 = int(min(max(1, c_boundary_per_core_baseline), B))
    c_candidates = sorted(set([
        c0,
        min(B, 2 * c0),
        min(B, 4 * c0),
        B,              # allow full fanout (often needed to force boundary transients)
    ]))

    # ext_bnd_edges candidates: allow boundary degree inflation (can be >B) WITHOUT changing eta.
    # This is one of the only reliable unweighted levers to make boundary work dominate.
    m0 = int(max(1, ext_bnd_edges_target_baseline))
    m_candidates = sorted(set([
        m0,
        5 * B,
        20 * B,
        50 * B,
    ]))

    # deg_b_internal candidates (even, within [2, B-2])
    if deg_b_internal_base is None:
        deg_candidates = [20, 40, 82, 120, 160, 240, 320, 400, 500]
    else:
        deg_candidates = [int(deg_b_internal_base)]
    # normalize to even and within range
    degB_candidates = []
    for g in deg_candidates:
        g = int(g)
        if g % 2 == 1:
            g -= 1
        g = max(2, min(g, B - 2))
        if g % 2 == 1:
            g -= 1
        g = max(2, g)
        if g not in degB_candidates:
            degB_candidates.append(g)

    print("\n[alpha sweep auto-tune] searching candidates:")
    print(f"  calib alphas: {calib}")
    print(f"  c_boundary_per_core candidates: {c_candidates}")
    print(f"  ext_bnd_edges candidates: {m_candidates}")
    print(f"  deg_b_internal candidates: {degB_candidates}")
    print("  (all unweighted; eta>0 enforced; no-perc enforced)\n")

    best = None
    best_score = (-1.0, -np.inf)
    best_details = None

    for C in c_candidates:
        # seed degree estimate uses capped C
        d_core = float((core_size - 1) + C)

        for degB in degB_candidates:
            for m in m_candidates:
                # Need ext_size for no-perc at alpha_start.
                # We need d_min too; boundary min degree depends on cb_counts and be_counts,
                # but we can conservatively lower-bound d_min by min(d_core, degB + floor(core_size*C/B) + floor(m/B)).
                cb_min = (core_size * C) // B
                be_min = (m // B)  # since round-robin distributes m ext edges over B boundary nodes
                d_bnd_min = float(degB + cb_min + be_min)
                d_min_est = float(min(d_core, d_bnd_min))

                try:
                    ext_size = choose_ext_size_for_nopercolation(
                        alpha_start=float(alpha_start),
                        rho=float(rho),
                        d_v=float(d_core),
                        d_min=float(d_min_est),
                        safety=float(ext_safety),
                        ext_max=int(ext_max),
                    )
                except RuntimeError:
                    continue

                # ensure enough exteriors to attach m boundary edges
                if ext_size <= m:
                    ext_size = m + 1
                    if ext_size > ext_max:
                        continue

                # build operator
                try:
                    P, d, eta, _ = build_alpha_sweep_operator_unweighted(
                        core_size=core_size,
                        boundary_size=B,
                        ext_size=ext_size,
                        seed=seed,
                        c_boundary_per_core=C,
                        deg_b_internal=degB,
                        ext_bnd_edges=m,
                    )
                except Exception:
                    continue

                frac, mean_log, details = _score_candidate_on_window(
                    P=P, d=d, eta=eta, seed=seed,
                    rho=rho, eps_gap=eps_gap,
                    alpha_list=calib,
                    ista_cap=min(ista_cap, 800),      # keep tuning inexpensive
                    log_factor=log_factor,
                    fista_cap=min(fista_cap, 800),    # keep tuning inexpensive
                )

                if frac < 0:
                    continue  # violated no-perc somewhere

                score = (frac, mean_log)
                if score > best_score:
                    best_score = score
                    best = (C, degB, m, ext_size, float(eta))
                    best_details = details

                    print(f"[tune best] frac_slower={frac:.2f}, mean_log_ratio={mean_log:.3f} "
                          f"with C={C}, degB={degB}, m={m}, ext_size={ext_size}, eta={eta:.2e}")

    if best is None:
        raise RuntimeError("[alpha sweep auto-tune] No valid candidate found that satisfies no-percolation.")

    C, degB, m, ext_size, eta = best
    frac, mean_log = best_score

    print("\n[alpha sweep auto-tune] selected graph params:")
    print(f"  C=c_boundary_per_core={C}")
    print(f"  deg_b_internal={degB}")
    print(f"  ext_bnd_edges={m}")
    print(f"  ext_size={ext_size} => eta={eta:.3e}")
    print(f"  calibration: frac(FISTA slower)={frac:.2f}, mean_log_ratio={mean_log:.3f}")
    print("  calibration details (alpha, ista_ref, workI, workF, reachedF, lhs, rhs, ratio):")
    for row in best_details:
        if isinstance(row, tuple) and len(row) >= 8:
            a, ista_ref, wI, wF, reached, lhs, rhs, ratio = row[:8]
            print(f"    a={a:.3g} ista_ref={ista_ref:4d} wI={wI:.3g} wF={wF:.3g} "
                  f"reached={reached} lhs={lhs:.2e} rhs={rhs:.2e} ratio={ratio:.3g}")
    print("")
    return {
        "c_boundary_per_core": C,
        "deg_b_internal": degB,
        "ext_bnd_edges": m,
        "ext_size": ext_size,
        "eta": eta,
        "calib_frac_slower": frac,
        "calib_mean_log_ratio": mean_log,
    }


def alpha_sweep_experiment(
    alpha_grid,
    *,
    B=600,
    eps_gap=1e-6,
    rho=1e-4,
    core_size=60,
    seed=0,
    alpha_start=1e-3,
    ext_safety=1.05,
    ext_max=2_000_000,
    ista_cap=50000,
    log_factor=25.0,
    fista_cap=50000,
    # --- alpha-sweep-only knobs (unweighted) ---
    c_boundary_per_core=20,
    ext_bnd_edges_target=600,
    deg_b_internal_base=None,  # None => auto-tune deg_b_internal (and other params)
):
    """
    Alpha sweep on UNWEIGHTED graphs with eta>0 and no-percolation enforced.
    This version AUTO-TUNES alpha-sweep-only graph params to create a regime where FISTA
    is slower than ISTA in WORK on a wide small-alpha window.

    Epsilon sweep remains untouched.
    """
    results = []

    # Sanity: alpha grid respects alpha_start
    if float(np.min(alpha_grid)) < float(alpha_start) - 1e-15:
        raise RuntimeError(f"alpha_grid contains values below alpha_start={alpha_start:g}.")

    # Auto-tune parameters for alpha sweep only
    tune = _auto_tune_alpha_sweep_graph(
        alpha_start=float(alpha_start),
        alpha_window_hi=9.0e-1,
        rho=float(rho),
        eps_gap=float(eps_gap),
        core_size=int(core_size),
        B=int(B),
        seed=int(seed),
        ext_safety=float(ext_safety),
        ext_max=int(ext_max),
        ista_cap=int(ista_cap),
        log_factor=float(log_factor),
        fista_cap=int(fista_cap),
        c_boundary_per_core_baseline=int(c_boundary_per_core),
        ext_bnd_edges_target_baseline=int(ext_bnd_edges_target),
        deg_b_internal_base=deg_b_internal_base,
    )

    C = int(tune["c_boundary_per_core"])
    degB = int(tune["deg_b_internal"])
    m = int(tune["ext_bnd_edges"])
    ext_size = int(tune["ext_size"])

    # Build the chosen alpha-sweep-only operator ONCE
    P, d, eta, _ = build_alpha_sweep_operator_unweighted(
        core_size=core_size,
        boundary_size=B,
        ext_size=ext_size,
        seed=seed,
        c_boundary_per_core=C,
        deg_b_internal=degB,
        ext_bnd_edges=m,
    )

    print("[alpha sweep graph info]")
    print(f"  core_size={core_size}, B={B}, ext_size={ext_size}")
    print(f"  C={C}, deg_b_internal={degB}, ext_bnd_edges={m}")
    print(f"  eta={eta:.3e} (>0)")
    print(f"  d_seed={float(d[seed]):.1f}, d_min={float(d.min()):.1f}\n")

    # Run the actual alpha sweep
    for alpha in alpha_grid:
        alpha = float(alpha)
        beta = beta_sc(alpha)

        lhs = nopercolation_LHS(alpha, beta, float(eta), float(d[seed]), float(d.min()))
        rhs = rho * alpha
        ok = (lhs <= rhs)
        if not ok:
            raise RuntimeError(
                f"alpha={alpha:.6g} violates no-percolation on tuned alpha-sweep graph: "
                f"LHS={lhs:.3e} > RHS={rhs:.3e}. "
                f"Increase ext_safety/ext_max or raise alpha_start."
            )

        # adaptive ISTA reference iterations (UNCHANGED protocol)
        ista_ref = int(min(ista_cap, max(50, math.ceil(log_factor / alpha))))

        objs_I, works_I = run_ista_history(P, d, seed, alpha, rho, max_iter=ista_ref)

        # Termination target is the KKT / prox-gradient tolerance (absolute)
        target = float(eps_gap)
        work_I = first_work_below(objs_I, works_I, target)

        itF, work_F, rF, reached_F = run_fista_until(
            P, d, seed, alpha, rho, beta, target, max_iter=fista_cap
        )

        results.append({
            "alpha": float(alpha),
            "beta": float(beta),
            "c_boundary_per_core_used": int(C),
            "deg_b_internal_used": int(degB),
            "ext_bnd_edges_used": int(m),
            "ext_size_used": int(ext_size),
            "ista_ref_iters": int(ista_ref),
            "eta": float(eta),
            "lhs": float(lhs),
            "rhs": float(rhs),
            "nopercolation_ok": bool(ok),
            # Kept for compatibility with prior output schema:
            # now stores the minimum observed residual over the ISTA reference run
            "F_est": float(objs_I.min()),
            # now stores the residual tolerance
            "target": float(target),
            "work_ista": float(work_I),
            "work_fista": float(work_F) if reached_F else np.nan,
            "iters_fista": int(itF),
            "fista_reached": bool(reached_F),
        })

        ratio = (work_F / work_I) if (reached_F and work_I > 0) else float("nan")
        print(f"[alpha={alpha:.3g}] ista_ref={ista_ref:5d}  "
              f"noperc_ok={ok} eta={eta:.2e}  LHS={lhs:.2e} rhs={rhs:.2e}  "
              f"workI={work_I:.3g} workF={work_F:.3g}  F/I={ratio:.3g} reachedF={reached_F}")

    # window report
    alphas = np.array([r["alpha"] for r in results], dtype=float)
    wI = np.array([r["work_ista"] for r in results], dtype=float)
    wF = np.array([r["work_fista"] for r in results], dtype=float)
    mask = (alphas >= 1.0e-3) & (alphas <= 9.0e-1) & np.isfinite(wF) & np.isfinite(wI)
    if mask.any():
        frac = float(np.mean(wF[mask] > wI[mask]))
        print(f"\n[window check] alpha in [1e-3, 9e-1]: FISTA slower on {frac*100:.1f}% of sampled points.\n")

    return results


# -----------------------------
# Epsilon sweep (UNCHANGED)
# -----------------------------
def epsilon_sweep_experiment(
    eps_grid,
    *,
    alpha=0.20,
    B=600,
    rho=1e-4,
    core_size=60,
    ext_size=1000,
    seed=0,
    c_boundary_per_core=20,
    deg_b_internal=82,
    deg_ext=998,
    max_iter=50000,
):
    # UNCHANGED: reference graph for the actual history runs (original unweighted cached graph)
    P, d, *_ = build_graph_cached(core_size, B, ext_size, seed,
                                  c_boundary_per_core, deg_b_internal, deg_ext)
    beta = beta_sc(alpha)

    # NOTE: histories now store residuals r(x_k) (termination is r(x_k) <= eps)
    objs_I, works_I = run_ista_history(P, d, seed, alpha, rho, max_iter=max_iter)
    objs_F, works_F = run_fista_history(P, d, seed, alpha, rho, beta, max_iter=max_iter)

    out = []
    for eps in eps_grid:
        # "Generate a new graph" per epsilon (cached) + re-check inequality (UNCHANGED)
        P2, d2, *_ = build_graph_cached(core_size, B, ext_size, seed,
                                        c_boundary_per_core, deg_b_internal, deg_ext)
        eta = 1.0 / (deg_ext + 1) if B > 0 else 0.0
        lhs = nopercolation_LHS(alpha, beta, eta, float(d2[seed]), float(d2.min()))
        ok = (lhs <= rho * alpha)

        target = float(eps)  # KKT / prox-gradient tolerance
        wI = first_work_below(objs_I, works_I, target)
        wF = first_work_below(objs_F, works_F, target)

        out.append({
            "eps": float(eps),
            "target": float(target),
            "nopercolation_ok": bool(ok),
            "lhs": float(lhs),
            "rhs": float(rho * alpha),
            "work_ista": float(wI),
            "work_fista": float(wF),
        })
    return out


def plot_alpha_sweep(results, save_base="B600_work_vs_alpha"):

    alphas = np.array([r["alpha"] for r in results])
    wI = np.array([r["work_ista"] for r in results], dtype=float)
    wF = np.array([r["work_fista"] for r in results], dtype=float)

    # Compressed tick labels: 1.2K, 50K, 3.4M, ...
    def kfmt(x, pos):
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

    fig, ax = plt.subplots(figsize=(8, 4.8))
    # ax.set_title("B=600: Work vs alpha (UNWEIGHTED; eta>0; no-percolation enforced; auto-tuned)")
    ax.set_xlabel(r"$\alpha$", fontsize=32)
    ax.set_ylabel("Total work", fontsize=32)

    ax.plot(
        alphas, wI,
        marker="o",
        linestyle="-",
        linewidth=6.0,
        markersize=12,
        label="ISTA",
        alpha=0.9,
    )
    ax.plot(
        alphas, wF,
        marker="s",
        linestyle="--",
        linewidth=6.0,
        markersize=12,
        label="FISTA",
        alpha=0.9,
    )

    ax.grid(True, which="both", alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Larger tick label font
    ax.tick_params(axis="both", which="major", labelsize=26)

    # Keep x-axis log labels as 10^k
    ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext())

    # Compressed y ticks with K/M
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(kfmt))

    ax.legend(loc="lower left", fontsize=32)
    fig.tight_layout()
    fig.savefig(f"{save_base}.png", dpi=200)
    fig.savefig(f"{save_base}.pdf")
    print(f"[saved] {save_base}.png and .pdf")
    plt.show()
    return fig


def plot_epsilon_sweep(results, save_base="B600_work_vs_epsilon"):
    import matplotlib.ticker as mticker

    eps = np.array([r["eps"] for r in results])
    wI = np.array([r["work_ista"] for r in results], dtype=float)
    wF = np.array([r["work_fista"] for r in results], dtype=float)
    ok = np.array([r["nopercolation_ok"] for r in results], dtype=bool)

    # Compressed tick labels: 1.2K, 50K, 3.4M, ...
    def kfmt(x, pos):
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

    fig, ax = plt.subplots(figsize=(8, 4.8))
    # ax.set_title("B=600, alpha=0.2: Work vs epsilon")
    ax.set_xlabel(r"$\epsilon$", fontsize=32)
    ax.set_ylabel("Total work", fontsize=32)

    ax.semilogx(
        eps, wI,
        marker="o",
        linestyle="-",
        linewidth=6.0,
        markersize=12,
        label="ISTA",
    )
    ax.semilogx(
        eps, wF,
        marker="s",
        linestyle="--",
        linewidth=6.0,
        markersize=12,
        label="FISTA",
    )

    if (~ok).any():
        ax.scatter(eps[~ok], np.maximum(wI[~ok], wF[~ok]), marker="x", s=90, linewidths=2, label="no-perc violated")

    ax.grid(True, which="both", alpha=0.3)

    # Larger tick label font
    ax.tick_params(axis="both", which="major", labelsize=26)

    # Keep x-axis log labels as 10^k
    ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext())

    # Compressed y ticks with K/M
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(kfmt))

    ax.legend(loc="upper right", fontsize=32)
    fig.tight_layout()
    fig.savefig(f"{save_base}.png", dpi=200)
    fig.savefig(f"{save_base}.pdf")
    print(f"[saved] {save_base}.png and .pdf")
    plt.show()
    return fig


def main():
    # Fixed graph params (same as before for epsilon sweep)
    core_size = 60
    B = 600
    ext_size_eps = 1000
    seed = 0
    c_boundary_per_core_eps = 20
    deg_b_internal_eps = 82
    deg_ext_eps = 998

    rho = 1e-4

    # ===== (A) alpha sweep =====
    eps_gap = 1e-6
    alpha_start = 1.0e-2

    alpha_small = np.logspace(math.log10(alpha_start), math.log10(0.19), 10)
    alpha_large = np.linspace(0.20, 1.00, 12)
    alpha_grid = np.unique(np.concatenate([alpha_small, alpha_large]))

    print("\n=== Alpha sweep (B=600) UNWEIGHTED, eta>0, no-percolation enforced (auto-tuned) ===")
    res_alpha = alpha_sweep_experiment(
        alpha_grid,
        B=B,
        eps_gap=eps_gap,
        rho=rho,
        core_size=core_size,
        seed=seed,
        alpha_start=alpha_start,
        ext_safety=1.05,
        ext_max=2_000_000,   # can be bumped if needed
        ista_cap=50000,
        log_factor=25.0,
        fista_cap=50000,
        # alpha-sweep-only baselines (the tuner expands around these)
        c_boundary_per_core=20,
        ext_bnd_edges_target=B,
        deg_b_internal_base=None,   # None => tune deg_b_internal too
    )
    plot_alpha_sweep(res_alpha, save_base="B600_work_vs_alpha")

    # ===== (B) epsilon sweep ===== (UNCHANGED)
    alpha0 = 0.20
    eps_grid = np.logspace(-12, -1, 30)

    print("\n=== Epsilon sweep (B=600, alpha=0.2) ===")
    res_eps = epsilon_sweep_experiment(
        eps_grid,
        alpha=alpha0,
        B=B,
        rho=rho,
        core_size=core_size,
        ext_size=ext_size_eps,
        seed=seed,
        c_boundary_per_core=c_boundary_per_core_eps,
        deg_b_internal=deg_b_internal_eps,
        deg_ext=deg_ext_eps,
        max_iter=50000,
    )
    plot_epsilon_sweep(res_eps, save_base="B600_work_vs_epsilon")


if __name__ == "__main__":
    main()
