#!/usr/bin/env python3
"""
This is your experiment code (ISTA/FISTA work vs vol(B)) unchanged,
PLUS a binned adjacency-density visualization saved to a PDF for zooming.

Dependencies: numpy, scipy, matplotlib
"""

import math
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker


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


def build_graph_fixed_core_degree(
    core_size: int,
    boundary_size: int,
    ext_size: int,
    seed: int = 0,
    c_boundary_per_core: int = 20,
    deg_b_internal: int = 82,
    deg_ext: int = 998,
):
    """
    Core: clique on core_size nodes.
    Boundary: boundary_size nodes.
      - each core node has exactly c_boundary_per_core boundary neighbors (deterministic assignment),
        so core degrees are constant across boundary_size (for boundary_size>0).
      - boundary internal circulant with degree deg_b_internal (even, capped).
    Exterior: ext_size nodes.
      - exterior internal circulant degree deg_ext (even, <ext_size)
      - each exterior node connects to 1 boundary node (round-robin) if boundary_size>0
    """
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
        # core-boundary edges: each core has c_boundary_per_core neighbors in boundary
        for u in core:
            for j in range(c_boundary_per_core):
                b = boundary[(u * c_boundary_per_core + j) % boundary_size]
                a, c = (u, b) if u < b else (b, u)
                edges.append((a, c))

        # boundary internal
        deg_b = min(deg_b_internal, boundary_size - 1)
        if deg_b % 2 == 1:
            deg_b -= 1
        if deg_b >= 2:
            edges += circulant_edges(boundary, deg_b)

        # exterior-to-boundary (1 per exterior node)
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


def beta_sc(alpha: float) -> float:
    return (1 - math.sqrt(alpha)) / (1 + math.sqrt(alpha))


def soft_threshold(u: np.ndarray, tau: np.ndarray) -> np.ndarray:
    return np.sign(u) * np.maximum(np.abs(u) - tau, 0.0)


def ppr_objective(x, Px, d, seed, alpha, rho) -> float:
    c = (1 - alpha) / 2.0
    qdiag = (1 + alpha) / 2.0
    xQx = qdiag * float(x @ x) - c * float(x @ Px)
    bTx = alpha * (x[seed] / math.sqrt(d[seed]))
    g = rho * alpha * float((np.sqrt(d) * np.abs(x)).sum())
    return 0.5 * xQx - bTx + g

# ============================================================
# NEW stopping criterion: prox-gradient fixed point residual
# ============================================================

def prox_map_T(x: np.ndarray, Px: np.ndarray, *, c: float, tau: np.ndarray, b_seed: float, seed: int) -> np.ndarray:
    """
    Prox-gradient fixed-point mapping with unit step:
        T(x) = prox_g( x - ∇f(x) )

    For this objective, with Px = P x:
        x - ∇f(x) = c * (x + Px) + b_seed * e_seed
        prox_g is weighted soft-threshold with tau = rho*alpha*sqrt(d)
    """
    u = c * (x + Px)
    u[seed] += b_seed
    return soft_threshold(u, tau)


def fixed_point_residual_inf(x: np.ndarray, Tx: np.ndarray) -> float:
    """r(x) = ||x - T(x)||_∞."""
    return float(np.max(np.abs(x - Tx)))


def run_ista_until(P, d, seed, alpha, rho, kkt_eps, max_iter=50000):
    """
    ISTA with the SAME update as before, but stopping when:
        r(x_k) = ||x_k - T(x_k)||_∞ <= kkt_eps
    """
    c = (1 - alpha) / 2.0
    tau = rho * alpha * np.sqrt(d)
    b_seed = alpha / math.sqrt(d[seed])

    x = np.zeros_like(d)
    Px = np.zeros_like(d)
    work = 0.0

    # residual at x0
    Tx = prox_map_T(x, Px, c=c, tau=tau, b_seed=b_seed, seed=seed)
    r = fixed_point_residual_inf(x, Tx)
    if r <= kkt_eps:
        return 0, work, r

    for k in range(1, max_iter + 1):
        # ISTA update: x_next = T(x)
        x_next = Tx
        Px_next = P.dot(x_next)

        # work metric (unchanged): y=x for ISTA
        work += float(d[x != 0].sum()) + float(d[x_next != 0].sum())

        x, Px = x_next, Px_next

        # residual at current iterate x_k
        Tx = prox_map_T(x, Px, c=c, tau=tau, b_seed=b_seed, seed=seed)
        r = fixed_point_residual_inf(x, Tx)
        if r <= kkt_eps:
            return k, work, r

    return max_iter, work, r



def run_fista_until(P, d, seed, alpha, rho, beta, kkt_eps, max_iter=50000):
    """
    FISTA with the SAME update as before, but stopping when:
        r(x_k) = ||x_k - T(x_k)||_∞ <= kkt_eps

    IMPORTANT:
      - The FISTA update uses y_k extrapolates (unchanged).
      - The termination residual is evaluated at the current iterate x_k (not y_k),
        using the SAME mapping T as ISTA.
    """
    c = (1 - alpha) / 2.0
    tau = rho * alpha * np.sqrt(d)
    b_seed = alpha / math.sqrt(d[seed])

    x_prev = np.zeros_like(d)
    x = np.zeros_like(d)
    Px_prev = np.zeros_like(d)
    Px = np.zeros_like(d)
    work = 0.0

    # residual at x0
    Tx = prox_map_T(x, Px, c=c, tau=tau, b_seed=b_seed, seed=seed)
    r = fixed_point_residual_inf(x, Tx)
    if r <= kkt_eps:
        return 0, work, r

    for k in range(1, max_iter + 1):
        y = x + beta * (x - x_prev)
        Py = Px + beta * (Px - Px_prev)

        u = c * (y + Py)
        u[seed] += b_seed

        x_next = soft_threshold(u, tau)
        Px_next = P.dot(x_next)

        # work metric (unchanged): charges supp(y_k) and supp(x_{k+1})
        work += float(d[y != 0].sum()) + float(d[x_next != 0].sum())

        x_prev, x = x, x_next
        Px_prev, Px = Px, Px_next

        # residual at current iterate x_k
        Tx = prox_map_T(x, Px, c=c, tau=tau, b_seed=b_seed, seed=seed)
        r = fixed_point_residual_inf(x, Tx)
        if r <= kkt_eps:
            return k, work, r

    return max_iter, work, r



def nopercolation_LHS(alpha, beta, eta, d_v, d_min):
    return ((1 - alpha) / 2.0) * eta * (3 + 2 * beta) / math.sqrt(d_v * d_min)


def compute_volB(d, Bset):
    if len(Bset) == 0:
        return 0.0
    idx = np.fromiter(Bset, dtype=int)
    return float(d[idx].sum())


def estimate_F_star(P, d, seed, alpha, rho, iters=50000):
    c = (1 - alpha) / 2.0
    tau = rho * alpha * np.sqrt(d)
    b_seed = alpha / math.sqrt(d[seed])

    x = np.zeros_like(d)
    Px = np.zeros_like(d)
    best = ppr_objective(x, Px, d, seed, alpha, rho)

    for _ in range(iters):
        u = c * (x + Px)
        u[seed] += b_seed
        x = soft_threshold(u, tau)
        Px = P.dot(x)
        best = min(best, ppr_objective(x, Px, d, seed, alpha, rho))
    return best


# ============================================================
# Adjacency binned-density visualization (post-processing only)
# ============================================================

def build_adjacency_fixed_core_degree(
    core_size: int,
    boundary_size: int,
    ext_size: int,
    c_boundary_per_core: int = 20,
    deg_b_internal: int = 82,
    deg_ext: int = 998,
):
    """
    Rebuild the ADJACENCY MATRIX A (CSR) for the same deterministic construction used in
    build_graph_fixed_core_degree(...).

    IMPORTANT: This is for visualization only (does not change experiment).
    """
    core = list(range(core_size))
    boundary = list(range(core_size, core_size + boundary_size))
    exterior = list(range(core_size + boundary_size, core_size + boundary_size + ext_size))
    n = core_size + boundary_size + ext_size

    edges = []
    # core clique
    for i in range(core_size):
        for j in range(i + 1, core_size):
            edges.append((i, j))

    # exterior internal (circulant)
    edges += circulant_edges(exterior, deg_ext)

    if boundary_size > 0:
        # core-boundary edges
        for u in core:
            for j in range(c_boundary_per_core):
                b = boundary[(u * c_boundary_per_core + j) % boundary_size]
                a, c = (u, b) if u < b else (b, u)
                edges.append((a, c))

        # boundary internal (circulant)
        deg_b = min(deg_b_internal, boundary_size - 1)
        if deg_b % 2 == 1:
            deg_b -= 1
        if deg_b >= 2:
            edges += circulant_edges(boundary, deg_b)

        # exterior-to-boundary (1 per exterior node)
        for idx, x in enumerate(exterior):
            b = boundary[idx % boundary_size]
            a, c = (x, b) if x < b else (b, x)
            edges.append((a, c))

    if edges:
        rows = np.fromiter((e[0] for e in edges), dtype=int)
        cols = np.fromiter((e[1] for e in edges), dtype=int)
        data = np.ones(len(edges), dtype=np.uint8)
        A = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
        A = (A + A.T).tocsr()
        A.sum_duplicates()
    else:
        A = sp.csr_matrix((n, n), dtype=np.uint8)

    A.data[:] = 1
    return A


def _binned_edge_density(A_upper: sp.spmatrix, bin_size: int) -> np.ndarray:
    """
    Compute a BINNED edge-density matrix for an undirected graph.

    Input:
      A_upper: sparse upper-triangular adjacency (no diagonal), so each undirected edge appears once.
      bin_size: number of nodes per bin

    Output:
      D: (m x m) matrix where D[p,q] = (# edges between bin p and q) / (# possible pairs)
         with possible pairs = s_p*s_q if p!=q, else s_p*(s_p-1)/2.
    """
    n = A_upper.shape[0]
    m = int(math.ceil(n / bin_size))

    # Bin sizes
    sizes = np.full(m, bin_size, dtype=int)
    sizes[-1] = n - bin_size * (m - 1)

    # COO for edges
    A_coo = A_upper.tocoo()
    bi = A_coo.row // bin_size
    bj = A_coo.col // bin_size

    # Count edges per (bin_i, bin_j)
    counts = np.zeros((m, m), dtype=np.int64)
    for p, q in zip(bi, bj):
        counts[p, q] += 1

    # Symmetrize counts (since A_upper only has p<=q)
    counts = counts + counts.T

    # Possible pairs matrix
    possible = np.outer(sizes, sizes).astype(np.float64)
    diag_possible = sizes * (sizes - 1) / 2.0
    np.fill_diagonal(possible, diag_possible)

    # Density
    D = np.zeros_like(possible, dtype=np.float64)
    mask = possible > 0
    D[mask] = counts[mask] / possible[mask]
    return D


def visualize_adjacency_binned_density(
    B_list,
    core_size,
    ext_size,
    c_boundary_per_core,
    deg_b_internal,
    deg_ext,
    bin_size=20,
    cmap_name="magma",
    vmin=1e-4,
    save_pdf_path="adjacency_Blist_binned_density_colored.pdf",
):
    """
    Previous-style adjacency visualization:
      - For each B, build the TRUE adjacency A (same deterministic construction),
        then compute a binned edge-density matrix.
      - Plot a panel over B_list with:
          * log-scaled density heatmap (fancy colormap)
          * dashed block boundaries for core|boundary|exterior
          * labels and one shared colorbar

    Saves to a PDF (single page) for zooming.
    """
    fig, axes = plt.subplots(1, len(B_list), figsize=(3.2 * len(B_list), 4.2), constrained_layout=True)

    # Ensure axes is always an iterable (handles len(B_list)==1)
    axes = np.atleast_1d(axes).ravel()

    # Use a copy so we can set "under" color
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_under("white")

    # We'll use a shared norm across panels for comparability
    norm = LogNorm(vmin=vmin, vmax=1.0)

    ims = []
    for ax, B in zip(axes, B_list):
        A = build_adjacency_fixed_core_degree(
            core_size, B, ext_size,
            c_boundary_per_core=c_boundary_per_core,
            deg_b_internal=deg_b_internal,
            deg_ext=deg_ext,
        )
        # upper triangle, no diagonal (each edge once)
        A_upper = sp.triu(A, k=1).tocsr()

        D = _binned_edge_density(A_upper, bin_size=bin_size)

        # Mask zeros so LogNorm works nicely
        Dm = np.ma.masked_less_equal(D, 0.0)
        im = ax.imshow(Dm, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
        ims.append(im)

        m = D.shape[0]

        # Block boundaries (in bin coordinates)
        core_bins = core_size // bin_size
        boundary_bins = B // bin_size if B > 0 else 0
        b1 = core_bins + boundary_bins

        ax.axvline(core_bins - 0.5, linestyle="--", linewidth=1.0, alpha=0.8)
        ax.axhline(core_bins - 0.5, linestyle="--", linewidth=1.0, alpha=0.8)
        if B > 0:
            ax.axvline(b1 - 0.5, linestyle="--", linewidth=1.0, alpha=0.8)
            ax.axhline(b1 - 0.5, linestyle="--", linewidth=1.0, alpha=0.8)

        # ax.set_title(f"B={B}\n(binned density)")
        ax.set_title(r"$\mathcal{B}$=" + f"{B}", fontsize=18)
        ax.set_xticks([])
        ax.set_yticks([])

        # ax.text(core_bins / 2, m - 1.5, "core", ha="center", va="top", fontsize=18, color="white")
        # if B > 0:
        #     ax.text((core_bins + b1) / 2, m - 1.5, "boundary", ha="center", va="top", fontsize=18, color="white")
        # ax.text((b1 + m) / 2, m - 1.5, "exterior", ha="center", va="top", fontsize=18, color="white")

    # Shared colorbar
    cbar = fig.colorbar(ims[0], ax=axes, fraction=0.025, pad=0.02)
    # cbar.set_label("edge density (fraction of possible edges, log scale)")
    cbar.set_label("edge density", fontsize=18)
    cbar.ax.tick_params(labelsize=18)

    # --- NEW: force the colorbar to match the subplot (adjacency) height exactly ---
    # constrained_layout can make the colorbar span the full figure height; draw once,
    # then freeze layout and manually set the cbar axis height to match the panels.
    fig.canvas.draw()
    fig.set_constrained_layout(False)

    y0 = min(ax.get_position().y0 for ax in axes)
    y1 = max(ax.get_position().y1 for ax in axes)
    cbpos = cbar.ax.get_position()
    cbar.ax.set_position([cbpos.x0, y0, cbpos.width, y1 - y0])
    # ---------------------------------------------------------------------------

    # fig.suptitle(
    #    f"Binned adjacency density (bin_size={bin_size}) — core clique | boundary band | exterior dense | cross edges",
    #    fontsize=12,
    # )

    fig.savefig(save_pdf_path, bbox_inches="tight")
    print(f"[saved] Binned-density adjacency PDF -> {save_pdf_path}")
    plt.show()
    plt.close(fig)
    return save_pdf_path



def main():
    # ---- Parameters (edit as needed) ----
    alpha = 0.20
    rho = 1e-4
    beta = beta_sc(alpha)

    core_size = 60
    c_boundary_per_core = 20

    ext_size = 1000
    deg_ext = 998  # eta = 1/(deg_ext+1)
    deg_b_internal = 82

    eps = 1e-6
    B_list = [320, 400, 600, 800, 1000]

    # ---- Sweep ----
    vols = []
    work_ista = []
    work_fista = []
    lhs_list = []

    for B in B_list:
        P, d, S, Bset, Ext, seed = build_graph_fixed_core_degree(
            core_size, B, ext_size,
            c_boundary_per_core=c_boundary_per_core,
            deg_b_internal=deg_b_internal,
            deg_ext=deg_ext,
        )
        volB = compute_volB(d, Bset)
        vols.append(volB)

        eta = 0.0 if B == 0 else 1.0 / (deg_ext + 1)
        lhs = nopercolation_LHS(alpha, beta, eta, float(d[seed]), float(d.min()))
        lhs_list.append(lhs)

        # eps is now the KKT / prox-gradient fixed-point tolerance
        _, wI, _ = run_ista_until(P, d, seed, alpha, rho, eps)
        _, wF, _ = run_fista_until(P, d, seed, alpha, rho, beta, eps)

        work_ista.append(wI)
        work_fista.append(wF)

    # ---- Plot ----
    fig, ax = plt.subplots()

    ax.plot(
        vols, work_ista,
        marker="o",
        linestyle="-",
        linewidth=4.0,
        markersize=8,
        label="ISTA",
    )

    ax.plot(
        vols, work_fista,
        marker="s",
        linestyle="--",
        linewidth=4.0,
        markersize=8,
        label="FISTA",
    )

    ax.set_xlabel(r"$\operatorname{vol}(\mathcal{B})$", fontsize=18)
    ax.set_ylabel("Total work", fontsize=18)

    # Larger tick label font
    ax.tick_params(axis="both", which="major", labelsize=16)

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

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(kfmt))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(kfmt))

    ax.grid(True)
    ax.legend(fontsize=18)
    fig.tight_layout()
    
    # --- NEW: save the figure (does not affect the experiment) ---
    out_png = "work_vs_volumeB.png"
    out_pdf = "work_vs_volumeB.pdf"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    print(f"[saved] {out_png}")
    print(f"[saved] {out_pdf}")

    # Diagnostic print: no-percolation check
    print("max no-percolation LHS =", max(lhs_list))
    print("rho*alpha =", rho * alpha)

    # ============================================================
    # Post-processing adjacency visualizations (DO NOT affect experiment)
    # ============================================================
    visualize_adjacency_binned_density(
        B_list=B_list,
        core_size=core_size,
        ext_size=ext_size,
        c_boundary_per_core=c_boundary_per_core,
        deg_b_internal=deg_b_internal,
        deg_ext=deg_ext,
        bin_size=20,
        cmap_name="magma",
        vmin=1e-4,
        save_pdf_path="adjacency_Blist_binned_density_colored.pdf",
    )


if __name__ == "__main__":
    main()
