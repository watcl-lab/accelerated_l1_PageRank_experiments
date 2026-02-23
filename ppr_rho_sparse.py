#!/usr/bin/env python3
"""
Experiment: Fix B=600 and all other parameters as in your previous setup,
and sweep rho from small to large (into the regime where x* becomes identically 0).

CRITICAL (your requirement):
- For each rho, we generate a new graph (we do not reuse a single adjacency / P).
- We also check (and enforce via the rho grid) that the no-percolation inequality holds.

How we create a "new graph per rho" while keeping the same *structure*:
- We keep the same 3-block node partition (core | boundary | exterior) and the same wiring rules,
  but we randomly permute the order of boundary nodes and exterior nodes before forming
  the circulant edges + deterministic cross-edges. This changes the actual adjacency pattern.

Outputs:
- B600_work_vs_rho.png
- B600_work_vs_rho.pdf

Dependencies: numpy, scipy, matplotlib
"""

import math
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# -----------------------------
# Graph construction (same structure, randomized per call)
# -----------------------------
def circulant_edges(nodes, k):
    n = len(nodes)
    if k == 0:
        return []
    assert k % 2 == 0 and 0 < k < n
    half = k // 2
    edges = []
    for idx, u in enumerate(nodes):
        for j in range(1, half + 1):
            v = nodes[(idx + j) % n]
            edges.append((u, v) if u < v else (v, u))
    return edges

def sparsified_core_edges_connected(core_nodes, rng, keep_pct=20):
    """
    Keep ~keep_pct% of core-core edges while guaranteeing the core is connected.

    Method:
      1) Build a random spanning tree on the core (ensures connectivity).
      2) Add uniformly-sampled extra edges until we reach the target count.
    Returns edges as (u,v) with u < v.
    """
    n = len(core_nodes)
    if n <= 1:
        return []

    total = n * (n - 1) // 2
    # Round-to-nearest number of edges corresponding to keep_pct%
    target = (keep_pct * total + 50) // 100
    target = max(n - 1, min(target, total))

    # --- 1) Random spanning tree (connectivity guarantee) ---
    perm = core_nodes.copy()
    rng.shuffle(perm)

    chosen = set()
    for idx in range(1, n):
        u = perm[idx]
        v = perm[int(rng.integers(0, idx))]  # connect to an earlier node
        a, b = (u, v) if u < v else (v, u)
        chosen.add((a, b))

    # --- 2) Add extra edges to reach target density ---
    k_add = target - len(chosen)
    if k_add > 0:
        remaining = []
        for i in range(n):
            u = core_nodes[i]
            for j in range(i + 1, n):
                v = core_nodes[j]
                e = (u, v)  # u < v since core_nodes is sorted in your code
                if e not in chosen:
                    remaining.append(e)

        idxs = rng.choice(len(remaining), size=k_add, replace=False)
        for t in idxs:
            chosen.add(remaining[int(t)])

    return list(chosen)


def build_graph_fixed_core_degree_randomized(
    core_size: int,
    boundary_size: int,
    ext_size: int,
    *,
    seed_graph: int,
    seed_node: int = 0,
    c_boundary_per_core: int = 20,
    deg_b_internal: int = 82,
    deg_ext: int = 998,
):
    """
    Same high-level structure as before, but we shuffle the ORDER of boundary/exterior nodes
    so each call produces a different adjacency pattern.

    Returns:
      P: symmetric normalized adjacency (CSR)
      d: degree vector (numpy array)
    """
    assert 0 <= seed_node < core_size
    assert deg_ext % 2 == 0 and deg_ext < ext_size

    rng = np.random.default_rng(int(seed_graph))

    core = list(range(core_size))
    boundary = list(range(core_size, core_size + boundary_size))
    exterior = list(range(core_size + boundary_size, core_size + boundary_size + ext_size))

    rng.shuffle(boundary)
    rng.shuffle(exterior)

    n = core_size + boundary_size + ext_size
    edges = []

    # core: keep ~20% of clique edges, but ensure the core stays connected
    edges += sparsified_core_edges_connected(core, rng=rng, keep_pct=20)

    # exterior internal (circulant)
    edges += circulant_edges(exterior, deg_ext)

    if boundary_size > 0:
        # core-boundary: each core has c_boundary_per_core boundary neighbors
        for u in core:
            base = u * c_boundary_per_core
            for j in range(c_boundary_per_core):
                b = boundary[(base + j) % boundary_size]
                edges.append((u, b) if u < b else (b, u))

        # boundary internal (circulant)
        deg_b = min(deg_b_internal, boundary_size - 1)
        if deg_b % 2 == 1:
            deg_b -= 1
        if deg_b >= 2:
            edges += circulant_edges(boundary, deg_b)

        # exterior-to-boundary: 1 boundary neighbor per exterior node
        for idx, x in enumerate(exterior):
            b = boundary[idx % boundary_size]
            edges.append((x, b) if x < b else (b, x))

    # Build adjacency
    rows = np.fromiter((e[0] for e in edges), dtype=int)
    cols = np.fromiter((e[1] for e in edges), dtype=int)
    data = np.ones(len(edges), dtype=np.float64)

    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    A = (A + A.T).tocsr()
    A.sum_duplicates()
    A.data[:] = 1.0

    d = np.array(A.sum(axis=1)).ravel()
    inv_sqrt_d = 1.0 / np.sqrt(d)

    # Build P by scaling A's COO entries (fast)
    Acoo = A.tocoo()
    Pdata = inv_sqrt_d[Acoo.row] * Acoo.data * inv_sqrt_d[Acoo.col]
    P = sp.coo_matrix((Pdata, (Acoo.row, Acoo.col)), shape=A.shape).tocsr()
    P.sum_duplicates()

    return P, d


# -----------------------------
# Prox + objective (same as before)
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


# -----------------------------
# NEW stopping criterion: prox-gradient fixed point residual
# -----------------------------
def prox_map_T(x, Px, *, c, tau, b_seed, seed):
    """
    Prox-gradient fixed-point mapping with unit step:
        T(x) = prox_g( x - ∇f(x) )

    For this objective, given Px = P x:
        x - ∇f(x) = c * (x + Px) + b_seed * e_seed
        prox_g is weighted soft-threshold with tau = rho*alpha*sqrt(d)
    """
    u = c * (x + Px)
    u[seed] += b_seed
    return soft_threshold(u, tau)


def fixed_point_residual_inf(x, Tx):
    """r(x) = ||x - T(x)||_∞."""
    return float(np.max(np.abs(x - Tx)))


def cum_work_update(work, d, y, x_next):
    return work + float(d[y != 0].sum()) + float(d[x_next != 0].sum())


def first_work_below(objs, works, target):
    idx = np.where(objs <= target)[0]
    if idx.size == 0:
        return np.nan
    return float(works[idx[0]])


# -----------------------------
# ISTA / FISTA (same logic)
# -----------------------------
def run_ista_history(P, d, seed, alpha, rho, max_iter):
    c = (1.0 - alpha) / 2.0
    tau = rho * alpha * np.sqrt(d)
    b_seed = alpha / math.sqrt(d[seed])

    n = d.size
    x = np.zeros(n)
    Px = np.zeros(n)

    work = 0.0
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

        u = c * (y + Py)
        u[seed] += b_seed

        x_next = soft_threshold(u, tau)
        Px_next = P.dot(x_next)

        work = cum_work_update(work, d, y, x_next)

        x_prev, x = x, x_next
        Px_prev, Px = Px, Px_next

        Tx = prox_map_T(x, Px, c=c, tau=tau, b_seed=b_seed, seed=seed)
        r = fixed_point_residual_inf(x, Tx)
        if r <= target:
            return k, work, r, True

    return max_iter, work, r, False


# -----------------------------
# No-percolation inequality check
# -----------------------------
def nopercolation_LHS(alpha, beta, eta, d_v, d_min):
    return ((1.0 - alpha) / 2.0) * eta * (3.0 + 2.0 * beta) / math.sqrt(d_v * d_min)

def degrees_only_for_nopercolation(
    core_size: int,
    boundary_size: int,
    ext_size: int,
    *,
    seed_graph: int,
    seed_node: int = 0,
    c_boundary_per_core: int = 20,
    deg_b_internal: int = 82,
    deg_ext: int = 998,
    keep_pct: int = 20,
):
    """
    Compute (d_seed, d_min) for the graph produced by
    build_graph_fixed_core_degree_randomized(...), WITHOUT building the full adjacency.

    This is ONLY used to choose a rho grid that satisfies no-percolation after
    core sparsification. It matches the RNG consumption of the real builder by:
      - shuffling boundary + exterior first (same as build_graph...),
      - then generating sparsified core edges with the same rng state.

    Returns:
      d_seed: degree of the seed node
      d_min:  global minimum degree in the whole graph
    """
    rng = np.random.default_rng(int(seed_graph))

    # Advance RNG state exactly as in build_graph_fixed_core_degree_randomized(...)
    boundary = np.arange(boundary_size, dtype=int)
    exterior = np.arange(ext_size, dtype=int)
    rng.shuffle(boundary)   # boundary order shuffle (even if empty)
    rng.shuffle(exterior)   # exterior order shuffle

    # Generate ONLY the sparsified core and compute core degrees
    core_nodes = list(range(core_size))
    core_edges = sparsified_core_edges_connected(core_nodes, rng=rng, keep_pct=keep_pct)

    core_deg = np.zeros(core_size, dtype=int)
    for u, v in core_edges:
        core_deg[u] += 1
        core_deg[v] += 1

    # Core nodes also have deterministic core->boundary edges (if boundary_size>0)
    d_core_total = core_deg.astype(float)
    if boundary_size > 0:
        d_core_total += float(c_boundary_per_core)

    d_seed = float(d_core_total[seed_node])
    d_min = float(d_core_total.min())

    # Boundary / exterior degrees (deterministic, and typically much larger than core min)
    if boundary_size > 0:
        deg_b = min(deg_b_internal, boundary_size - 1)
        if deg_b % 2 == 1:
            deg_b -= 1

        # core->boundary edges are assigned round-robin; minimum inbound is floor(total/B)
        total_core_bnd = core_size * c_boundary_per_core
        core_in_min = total_core_bnd // boundary_size

        # exterior->boundary is 1 per exterior node, assigned round-robin
        ext_in_min = ext_size // boundary_size

        boundary_min = float(deg_b + core_in_min + ext_in_min)
        exterior_min = float(deg_ext + 1)  # +1 boundary neighbor

        d_min = float(min(d_min, boundary_min, exterior_min))
    else:
        # No boundary edges => exterior has only circulant degree
        exterior_min = float(deg_ext)
        d_min = float(min(d_min, exterior_min))

    return d_seed, d_min


# -----------------------------
# Rho sweep
# -----------------------------
def rho_sweep(
    rho_grid,
    *,
    alpha=0.20,
    eps_gap=1e-10,
    core_size=60,
    B=600,
    ext_size=1000,
    seed_node=0,
    c_boundary_per_core=20,
    deg_b_internal=82,
    deg_ext=998,
    ista_ref=50000,
    fista_cap=50000,
    base_graph_seed=2026001,
):
    beta = beta_sc(alpha)
    eta = 1.0 / (deg_ext + 1)

    out = []
    for i, rho in enumerate(rho_grid):
        # NEW graph per rho
        P, d = build_graph_fixed_core_degree_randomized(
            core_size, B, ext_size,
            seed_graph=base_graph_seed + i,
            seed_node=seed_node,
            c_boundary_per_core=c_boundary_per_core,
            deg_b_internal=deg_b_internal,
            deg_ext=deg_ext,
        )

        lhs = nopercolation_LHS(alpha, beta, eta, float(d[seed_node]), float(d.min()))
        rhs = rho * alpha
        noperc_ok = (lhs <= rhs)

        # For this family, degrees are fixed, so noperc_ok depends mainly on rho.
        # We enforce the grid in main() so noperc_ok should be True for all points.
        if not noperc_ok:
            print(f"[warning] no-percolation FAILED at rho={rho:.3g}: LHS={lhs:.2e} > rhs={rhs:.2e}")

        d_seed = float(d[seed_node])
        zero_solution = (rho >= 1.0 / d_seed)  # x stays identically zero

        if zero_solution:
            out.append((rho, noperc_ok, lhs, rhs, 0.0, 0.0, True))
            continue

        # Common termination: r(x_k) = ||x_k - T(x_k)||_inf <= eps_gap
        target = eps_gap

        objs_I, works_I = run_ista_history(P, d, seed_node, alpha, rho, max_iter=ista_ref)
        work_I = first_work_below(objs_I, works_I, target)

        itF, work_F, rF, reached = run_fista_until(P, d, seed_node, alpha, rho, beta, target, max_iter=fista_cap)
        work_F = float(work_F) if reached else float("nan")

        out.append((rho, noperc_ok, lhs, rhs, float(work_I), work_F, False))

    return out


def main():
    # Fixed params (same as before, except rho sweep)
    alpha = 0.20
    eps_gap = 1e-6

    core_size = 60
    B = 600
    ext_size = 1000
    seed_node = 0
    c_boundary_per_core = 20
    deg_b_internal = 82
    deg_ext = 998

    beta = beta_sc(alpha)
    eta = 1.0 / (deg_ext + 1)

    # ------------------------------------------------------------
    # No-percolation threshold (UPDATED for sparsified core)
    # ------------------------------------------------------------
    # With a sparsified core, degrees are NOT fixed, so we compute the maximum
    # no-percolation LHS over the exact randomized graphs we will use in the sweep
    # (same base_graph_seed + i).
    base_graph_seed = 2026001
    n_points = 20

    lhs_max = 0.0
    d_seed_min = float("inf")

    for i in range(n_points):
        d_seed_i, d_min_i = degrees_only_for_nopercolation(
            core_size, B, ext_size,
            seed_graph=base_graph_seed + i,
            seed_node=seed_node,
            c_boundary_per_core=c_boundary_per_core,
            deg_b_internal=deg_b_internal,
            deg_ext=deg_ext,
            keep_pct=20,
        )
        lhs_i = nopercolation_LHS(alpha, beta, eta, d_seed_i, d_min_i)
        lhs_max = max(lhs_max, lhs_i)
        d_seed_min = min(d_seed_min, d_seed_i)

    rho_min_noperc = lhs_max / alpha

    # For the dotted vertical line in the plot: a conservative (guaranteed) "x*=0" threshold
    # across these randomized graphs (since d_seed varies with the sparsified core).
    rho_zero = 1.0 / d_seed_min

    # Pick rho grid so that no-percolation holds for ALL points, and we cross into x*=0 regime.
    rho_lo = max(1e-4, 1.05 * rho_min_noperc)
    rho_hi = 5e-2
    rho_grid = np.logspace(math.log10(rho_lo), math.log10(rho_hi), n_points)

    print(f"rho_min(no-perc) ≈ {rho_min_noperc:.3g}")
    print(f"rho_zero (x*=0)  ≈ {rho_zero:.3g}")
    print(f"rho grid: [{rho_grid[0]:.3g}, ..., {rho_grid[-1]:.3g}]  (n={rho_grid.size})")

    res = rho_sweep(
        rho_grid,
        alpha=alpha,
        eps_gap=eps_gap,
        core_size=core_size,
        B=B,
        ext_size=ext_size,
        seed_node=seed_node,
        c_boundary_per_core=c_boundary_per_core,
        deg_b_internal=deg_b_internal,
        deg_ext=deg_ext,
        ista_ref=50000,
        fista_cap=50000,
        base_graph_seed=base_graph_seed,
    )

    # Unpack
    rhos = np.array([r[0] for r in res])
    ok = np.array([r[1] for r in res], dtype=bool)
    wI = np.array([r[4] for r in res], dtype=float)
    wF = np.array([r[5] for r in res], dtype=float)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4.8))
    # ax.set_title("B=600, alpha=0.2: Work vs rho (new graph per rho)")
    ax.set_xlabel(r"$\rho$", fontsize=32)
    ax.set_ylabel("Total work", fontsize=32)

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
        return f"{x:.0f}"

    ax.plot(
        rhos, wI,
        marker="o",
        linestyle="-",
        linewidth=6.0,
        markersize=12,
        label="ISTA",
    )
    ax.plot(
        rhos, wF,
        marker="s",
        linestyle="--",
        linewidth=6.0,
        markersize=12,
        label="FISTA",
    )

    if (~ok).any():
        ax.scatter(rhos[~ok], wI[~ok], marker="x", s=80, linewidths=2, label="no-perc violated")
        ax.scatter(rhos[~ok], wF[~ok], marker="x", s=80, linewidths=2)

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(-1, 10000)

    # Larger tick label font
    ax.tick_params(axis="both", which="major", labelsize=26)

    # Keep x-axis log labels as 10^k
    ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext())

    # Compressed y ticks with K/M
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(kfmt))

    ax.grid(True, which="both", alpha=0.3)

    # ax.axvline(rho_min_noperc, linestyle="--", linewidth=4.0, alpha=0.6, label="no-perc threshold")
    # ax.axvline(rho_zero, linestyle=":",  linewidth=4.0, alpha=0.8, label="rho (x*=0)")

    ax.legend(loc="lower center", fontsize=32)
    fig.tight_layout()

    fig.savefig("B600_work_vs_rho_sparse.png", dpi=200)
    fig.savefig("B600_work_vs_rho_sparse.pdf")
    print("[saved] B600_work_vs_rho_sparse.png and B600_work_vs_rho_sparse.pdf")
    plt.show()

if __name__ == "__main__":
    main()
