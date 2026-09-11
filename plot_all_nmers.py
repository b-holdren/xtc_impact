#!/usr/bin/env python3
import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from plot_FEL_2D import FELResult, compute_fel_2d

# Global Config
# ==============================================================================
BINS = 30
FIGSIZE_PER_PANEL = 6.0
CMAP = "jet"
DPI = 300
DEFAULT_ZMAX = 18.0

NMER_NAMES = {
    2: "Dimer (N=2)",
    3: "Trimer (N=3)",
    4: "Tetramer (N=4)",
    5: "Pentamer (N=5)",
    6: "Hexamer (N=6)",
}

NMER_COLORS = {
    2: "#1f77b4",  # Blue
    3: "#00a88f",  # Teal
    4: "#2ca02c",  # Green
    5: "#ff7f0e",  # Orange
    6: "#d62728",  # Red
}

R_KJ = 8.314e-3
R_KCAL = 1.987204259e-3


@dataclass
class NmerData:
    n: int
    name: str
    times: np.ndarray
    ccs: np.ndarray
    beta: np.ndarray
    n_frames: int


#  [1/6] Data loading 
# ==============================================================================
def load_aligned_nmer_data(base_dir: str, mol: str, n_range: Tuple[int, int] = (2, 6)) -> List[NmerData]:
    nmers: List[NmerData] = []

    for n in range(n_range[0], n_range[1] + 1):
        ccs_path = os.path.join(base_dir, "collected_xvg_files", f"{mol}_n{n}_ccs.xvg")
        beta_path = os.path.join(base_dir, "collected_xvg_files", f"{mol}_n{n}_beta.xvg")
        if not os.path.exists(ccs_path) or os.path.getsize(ccs_path) == 0:
            continue
        if not os.path.exists(beta_path) or os.path.getsize(beta_path) == 0:
            continue

        try:
            ccs_data = np.loadtxt(ccs_path, comments=["#", "@"])
            beta_data = np.loadtxt(beta_path, comments=["#", "@"])
            if ccs_data.ndim == 1:
                ccs_data = ccs_data.reshape(1, -1)
            if beta_data.ndim == 1:
                beta_data = beta_data.reshape(1, -1)

            min_len = min(len(ccs_data), len(beta_data))
            if min_len > 0:
                times_arr = ccs_data[:min_len, 0]
                ccs_arr = ccs_data[:min_len, 1]
                beta_arr = beta_data[:min_len, 1]

                nmers.append(
                    NmerData(
                        n=n,
                        name=NMER_NAMES.get(n, f"N={n}"),
                        times=times_arr,
                        ccs=ccs_arr,
                        beta=beta_arr,
                        n_frames=min_len,
                    )
                )
        except Exception as e:
            print(f"  [N={n}] Error loading data for {mol}: {e}")

    return nmers
# ==============================================================================


#  [2/6] Direct FEL computation
# ==============================================================================
def compute_fel_from_arrays(
    x: np.ndarray,
    y: np.ndarray,
    temp: float = 300.0,
    bins: int = BINS,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    eps: float = 1e-10,
) -> Optional[FELResult]:
    if len(x) == 0 or len(y) == 0:
        return None

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())

    if xlim is None:
        if xmin == xmax:
            xlim = (xmin - 20.0, xmax + 20.0)
        else:
            x_pad = max(10.0, (xmax - xmin) * 0.05)
            xlim = (xmin - x_pad, xmax + x_pad)

    if ylim is None:
        if ymin == ymax:
            ylim = (max(0.0, ymin - 5.0), ymin + 15.0)
        else:
            y_pad = max(3.0, (ymax - ymin) * 0.05)
            ylim = (max(0.0, ymin - y_pad), min(100.0, ymax + y_pad))

    actual_bins = min(bins, max(5, len(x)))
    hist_range = [xlim, ylim]
    H, xedges, yedges = np.histogram2d(x, y, bins=actual_bins, range=hist_range, density=True)

    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    extent = [float(xedges[0]), float(xedges[-1]), float(yedges[0]), float(yedges[-1])]

    P = np.where(H > 0, H, eps)
    fel_kj = -R_KJ * temp * np.log(P / P.max())
    fel_kcal = -R_KCAL * temp * np.log(P / P.max())

    return FELResult(
        prob_density=H.T,
        fel_kj=fel_kj.T,
        fel_kcal=fel_kcal.T,
        extent=extent,
        xedges=xedges,
        yedges=yedges,
        xc=xc,
        yc=yc,
    )
# ==============================================================================


#  [3/6] Master grid (all systems in one PNG)
# ==============================================================================
def plot_master_grid_all_systems(
    mols: List[str],
    base_dir: str,
    n_range: Tuple[int, int] = (2, 6),
    zmax_kj: float = DEFAULT_ZMAX,
):
    n_sys = len(mols)
    if n_sys <= 3:
        nrows, ncols = 1, n_sys
    elif n_sys <= 6:
        nrows, ncols = 2, 3
    elif n_sys <= 9:
        nrows, ncols = 3, 3
    else:
        ncols = 4
        nrows = int(np.ceil(n_sys / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.2 * ncols, 6.5 * nrows + 1.2),
        squeeze=False,
    )

    im_ref = None

    for idx, mol in enumerate(mols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        try:
            nmers = load_aligned_nmer_data(base_dir, mol, n_range)
            if not nmers:
                ax.text(0.5, 0.5, f"No Data for {mol}", ha="center", va="center", transform=ax.transAxes, fontsize=14)
                continue

            all_x = np.concatenate([item.ccs for item in nmers])
            all_y = np.concatenate([item.beta for item in nmers])
            total_frames = len(all_x)

            # Global FEL for this system across all oligomers
            res = compute_fel_from_arrays(all_x, all_y, temp=300.0, bins=BINS)
            if res is not None:
                F_g = np.clip(res.fel_kj, 0, zmax_kj)
                im = ax.imshow(
                    F_g,
                    interpolation="gaussian",
                    origin="lower",
                    extent=res.extent,
                    aspect="auto",
                    cmap=CMAP,
                    vmin=0,
                    vmax=zmax_kj,
                    alpha=0.88,
                )
                im_ref = im

                # Add subtle contour lines
                X, Y = np.meshgrid(res.xc, res.yc)
                cs = ax.contour(X, Y, F_g, levels=np.arange(0, zmax_kj + 1, 4), colors="black", linewidths=0.4, alpha=0.35)

                # Overlay color-coded scatter points for each N-mer
                for item in nmers:
                    color = NMER_COLORS.get(item.n, "black")
                    
                    if len(item.ccs) > 800:
                        sub_idx = np.random.choice(len(item.ccs), size=800, replace=False)
                        x_pts, y_pts = item.ccs[sub_idx], item.beta[sub_idx]
                        alpha_val, s_val = 0.40, 14
                    else:
                        x_pts, y_pts = item.ccs, item.beta
                        alpha_val = 0.70 if len(item.ccs) < 50 else 0.50
                        s_val = 28 if len(item.ccs) < 50 else 18

                    ax.scatter(
                        x_pts,
                        y_pts,
                        color=color,
                        edgecolor="black",
                        linewidth=0.3,
                        s=s_val,
                        alpha=alpha_val,
                        zorder=4,
                    )

                ax.set_title(f"{mol.upper()}  ({total_frames:,} frames)", fontsize=16, fontweight="bold", pad=8)
                if r == nrows - 1:
                    ax.set_xlabel("CCS (Å²)", fontsize=14, fontweight="bold")
                if c == 0:
                    ax.set_ylabel("$\\beta$-Strand (%)", fontsize=14, fontweight="bold")
                ax.tick_params(labelsize=11)
                ax.grid(True, linestyle="--", alpha=0.25)

        except Exception as e:
            print(f"Error rendering master grid panel for {mol}: {e}")
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes, fontsize=12)

    # Hide unused axes
    for idx in range(n_sys, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    # Global Top Legend for N-mer classes
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label=f"Dimer ($N=2$)", markerfacecolor=NMER_COLORS[2], markeredgecolor="black", markersize=10),
        Line2D([0], [0], marker="o", color="w", label=f"Trimer ($N=3$)", markerfacecolor=NMER_COLORS[3], markeredgecolor="black", markersize=10),
        Line2D([0], [0], marker="o", color="w", label=f"Tetramer ($N=4$)", markerfacecolor=NMER_COLORS[4], markeredgecolor="black", markersize=10),
        Line2D([0], [0], marker="o", color="w", label=f"Pentamer ($N=5$)", markerfacecolor=NMER_COLORS[5], markeredgecolor="black", markersize=10),
        Line2D([0], [0], marker="o", color="w", label=f"Hexamer ($N=6$)", markerfacecolor=NMER_COLORS[6], markeredgecolor="black", markersize=10),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.47, 0.99),
        ncol=5,
        fontsize=13,
        framealpha=0.92,
        edgecolor="gray",
    )

    fig.suptitle("Free Energy Landscapes Across All Molecular Systems and Oligomers ($N=2$ to $6$)", fontsize=22, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 0.91, 0.96])

    # Shared colorbar on the right
    if im_ref is not None:
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.70])
        ticks = list(np.arange(0, zmax_kj + 1, 3.0))
        cbar = fig.colorbar(im_ref, cax=cbar_ax)
        cbar.set_label("$\\Delta F$ (kJ/mol)", fontsize=16, fontweight="bold")
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize=12)

    out_png = os.path.join(base_dir, "FEL_2D_master_all_systems_combined.png")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    print(f"\n>>> Saved Master Grid (All Systems on One PNG) -> {out_png}")
    plt.close(fig)



#  [4/6] Single graph per system (all N-mers N=2..6 on one plot)
# ==============================================================================
def plot_system_all_nmers_single_graph(
    mol: str,
    base_dir: str,
    n_range: Tuple[int, int] = (2, 6),
    zmax_kj: float = DEFAULT_ZMAX,
):
    nmers = load_aligned_nmer_data(base_dir, mol, n_range)
    if not nmers:
        print(f"No valid data found for system: {mol}. Skipping...")
        return

    all_x = np.concatenate([item.ccs for item in nmers])
    all_y = np.concatenate([item.beta for item in nmers])
    total_frames = len(all_x)

    res = compute_fel_from_arrays(all_x, all_y, temp=300.0, bins=BINS)
    if res is None:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    F_g = np.clip(res.fel_kj, 0, zmax_kj)
    im = ax.imshow(
        F_g,
        interpolation="gaussian",
        origin="lower",
        extent=res.extent,
        aspect="auto",
        cmap=CMAP,
        vmin=0,
        vmax=zmax_kj,
        alpha=0.88,
    )

    X, Y = np.meshgrid(res.xc, res.yc)
    cs = ax.contour(X, Y, F_g, levels=np.arange(0, zmax_kj + 1, 3), colors="black", linewidths=0.5, alpha=0.4)
    ax.clabel(cs, fmt="%.0f", fontsize=8, inline=True)

    legend_elements = []
    for item in nmers:
        color = NMER_COLORS.get(item.n, "black")
        
        if len(item.ccs) > 1000:
            sub_idx = np.random.choice(len(item.ccs), size=1000, replace=False)
            x_pts, y_pts = item.ccs[sub_idx], item.beta[sub_idx]
            alpha_val, s_val = 0.45, 18
        else:
            x_pts, y_pts = item.ccs, item.beta
            alpha_val = 0.75 if len(item.ccs) < 50 else 0.55
            s_val = 36 if len(item.ccs) < 50 else 22

        ax.scatter(
            x_pts,
            y_pts,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            s=s_val,
            alpha=alpha_val,
            zorder=4,
        )

        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"{item.name} ({item.n_frames:,} frames)",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=10,
            )
        )

    ax.set_title(f"{mol.upper()} — All Oligomer Complexes ($N=2$ to $6$)\nTotal Sampled: {total_frames:,} Frames", fontsize=18, fontweight="bold", pad=12)
    ax.set_xlabel("CCS (Å²)", fontsize=16, fontweight="bold")
    ax.set_ylabel("$\\beta$-Strand (%)", fontsize=16, fontweight="bold")
    ax.tick_params(labelsize=13)
    ax.grid(True, linestyle="--", alpha=0.3)

    ax.legend(handles=legend_elements, loc="upper right", fontsize=11, framealpha=0.88, edgecolor="gray")

    cbar = fig.colorbar(im, ax=ax, pad=0.03, shrink=0.92)
    cbar.set_label("$\\Delta F$ (kJ/mol)", fontsize=15, fontweight="bold")
    cbar.set_ticks(list(np.arange(0, zmax_kj + 1, 3.0)))
    cbar.ax.tick_params(labelsize=12)

    fig.tight_layout()

    out_png = os.path.join(base_dir, f"FEL_2D_{mol}_all_nmers_single.png")
    fig.savefig(out_png, dpi=DPI)
    print(f"Saved Single Graph -> {out_png}")
    plt.close(fig)
# ==============================================================================


#  [5/6] Multi-panel grid per system (individual panels N=2..6)
# ==============================================================================
def plot_system_multi_panel_grid(
    mol: str,
    base_dir: str,
    n_range: Tuple[int, int] = (2, 6),
    zmax_kj: float = DEFAULT_ZMAX,
):
    nmers = load_aligned_nmer_data(base_dir, mol, n_range)
    if not nmers:
        return

    n_panels = len(nmers)
    if n_panels <= 3:
        nrows, ncols = 1, n_panels
    elif n_panels <= 6:
        nrows, ncols = 2, 3
    else:
        ncols = 3
        nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(FIGSIZE_PER_PANEL * ncols, FIGSIZE_PER_PANEL * nrows + 1.0),
        squeeze=False,
    )

    im = None
    for idx, item in enumerate(nmers):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        res = compute_fel_from_arrays(item.ccs, item.beta, temp=300.0, bins=BINS)
        if res is not None:
            F_g = np.clip(res.fel_kj, 0, zmax_kj)
            im = ax.imshow(
                F_g,
                interpolation="gaussian",
                origin="lower",
                extent=res.extent,
                aspect="auto",
                cmap=CMAP,
                vmin=0,
                vmax=zmax_kj,
            )

            if item.n_frames < 40:
                ax.scatter(item.ccs, item.beta, color="white", edgecolor="black", s=40, zorder=5, alpha=0.9, label="Observed")
                ax.legend(loc="upper right", fontsize=10, framealpha=0.7)

            color = NMER_COLORS.get(item.n, "black")
            ax.set_title(f"{item.name}\n({item.n_frames:,} frame{'s' if item.n_frames != 1 else ''})", fontsize=16, fontweight="bold", color=color)
            ax.set_xlabel("CCS (Å²)", fontsize=14)
            if c == 0:
                ax.set_ylabel("$\\beta$-Strand (%)", fontsize=14)
            ax.tick_params(labelsize=11)

    for idx in range(len(nmers), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle(f"Free Energy Landscapes by Oligomer Complex: {mol.upper()}", fontsize=22, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.90, 0.95])

    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.70])
        ticks = list(np.arange(0, zmax_kj + 1, 3.0))
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("$\\Delta F$ (kJ/mol)", fontsize=16)
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize=12)

    out_png = os.path.join(base_dir, f"FEL_2D_{mol}_all_nmers_grid.png")
    fig.savefig(out_png, dpi=DPI)
    print(f"Saved Multi-Panel Grid -> {out_png}")
    plt.close(fig)
# ==============================================================================


#  [6/6] Cross-system comparison grids per N-mer size
# ==============================================================================
def plot_cross_system_grid_for_nmer(
    n: int,
    base_dir: str,
    mols: List[str],
    zmax_kj: float = DEFAULT_ZMAX,
):
    systems_data = []

    for mol in mols:
        nmers = load_aligned_nmer_data(base_dir, mol, (n, n))
        if nmers:
            systems_data.append((mol, nmers[0]))

    if not systems_data:
        print(f"No valid data found across systems for N={n}. Skipping...")
        return

    n_sys = len(systems_data)
    if n_sys <= 3:
        nrows, ncols = 1, n_sys
    elif n_sys <= 6:
        nrows, ncols = 2, 3
    elif n_sys <= 9:
        nrows, ncols = 3, 3
    else:
        ncols = 4
        nrows = int(np.ceil(n_sys / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(FIGSIZE_PER_PANEL * ncols, FIGSIZE_PER_PANEL * nrows + 1.0),
        squeeze=False,
    )

    im = None
    for idx, (mol, item) in enumerate(systems_data):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        res = compute_fel_from_arrays(item.ccs, item.beta, temp=300.0, bins=BINS)
        if res is not None:
            F_g = np.clip(res.fel_kj, 0, zmax_kj)
            im = ax.imshow(
                F_g,
                interpolation="gaussian",
                origin="lower",
                extent=res.extent,
                aspect="auto",
                cmap=CMAP,
                vmin=0,
                vmax=zmax_kj,
            )

            if item.n_frames < 40:
                ax.scatter(item.ccs, item.beta, color="white", edgecolor="black", s=40, zorder=5, alpha=0.9)

            ax.set_title(f"{mol.upper()}\n({item.n_frames:,} frames)", fontsize=16, fontweight="bold")
            ax.set_xlabel("CCS (Å²)", fontsize=14)
            if c == 0:
                ax.set_ylabel("$\\beta$-Strand (%)", fontsize=14)
            ax.tick_params(labelsize=11)

    for idx in range(len(systems_data), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    n_name = NMER_NAMES.get(n, f"N={n}")
    fig.suptitle(f"Comparison of {n_name} Complexes Across Molecular Systems", fontsize=22, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.90, 0.95])

    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.70])
        ticks = list(np.arange(0, zmax_kj + 1, 3.0))
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("$\\Delta F$ (kJ/mol)", fontsize=16)
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize=12)

    out_png = os.path.join(base_dir, f"FEL_2D_grid_n{n}.png")
    fig.savefig(out_png, dpi=DPI)
    print(f"Saved Cross-System Grid -> {out_png}")
    plt.close(fig)
# ==============================================================================


# Driver
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Plot 2D FEL for oligomer complexes with system-wide beta strand.")
    parser.add_argument("--base-dir", default="/home/bholdren/systems", help="Base directory")
    parser.add_argument(
        "--mols",
        nargs="+",
        default=["mol-13", "mol-14", "mol-20", "mol-22", "mol-26", "mol-36", "mol-37", "mol-53", "hexamer"],
        help="List of molecules/systems to plot",
    )
    parser.add_argument(
        "--mode",
        choices=["master", "single", "grid", "by-nmer", "all"],
        default="master",
        help="Plot mode: 'master' (all systems in one master PNG), 'single' (individual single graph per system), 'grid', 'by-nmer', or 'all'",
    )
    parser.add_argument("--zmax", type=float, default=DEFAULT_ZMAX, help="Max free energy cutoff (kJ/mol)")
    args = parser.parse_args()

    #  Master Grid (All Systems on ONE PNG)
    if args.mode in ["master", "all"]:
        print("==========================================================")
        print("1. Generating Master Grid (All Systems on ONE PNG)...")
        print("==========================================================")
        plot_master_grid_all_systems(args.mols, args.base_dir, n_range=(2, 6), zmax_kj=args.zmax)

    # 2. Individual Single Graph per system
    if args.mode in ["single", "all"]:
        print("\n==========================================================")
        print("2. Generating Individual Single Graphs per system...")
        print("==========================================================")
        for mol in args.mols:
            print(f"--- Processing System: {mol} ---")
            plot_system_all_nmers_single_graph(mol, args.base_dir, n_range=(2, 6), zmax_kj=args.zmax)

    # 3. Multi-Panel Grid per system
    if args.mode in ["grid", "all"]:
        print("\n==========================================================")
        print("3. Generating Multi-Panel Grid per system...")
        print("==========================================================")
        for mol in args.mols:
            print(f"--- Processing Grid for: {mol} ---")
            plot_system_multi_panel_grid(mol, args.base_dir, n_range=(2, 6), zmax_kj=args.zmax)

    # 4. Cross-System comparison grids per N-mer
    if args.mode in ["by-nmer", "all"]:
        print("\n==========================================================")
        print("4. Generating Cross-System comparison grids per N-mer...")
        print("==========================================================")
        for n in range(2, 7):
            print(f"--- Generating comparison grid for N={n} ---")
            plot_cross_system_grid_for_nmer(n, args.base_dir, args.mols, zmax_kj=args.zmax)
# ==============================================================================


if __name__ == "__main__":
    main()
