#!/usr/bin/env python3
"""2D free energy landscape from two GROMACS XVG files."""

from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import os
# from systems_config import BASE_DIR, SYSTEMS


R_KJ = 8.314e-3
R_KCAL = 1.987204259e-3


@dataclass
class FELResult:
    """2D FEL computation results. All grids are [ny, nx] (image orientation)."""
    prob_density: np.ndarray
    fel_kj: np.ndarray
    fel_kcal: np.ndarray
    extent: list
    xedges: np.ndarray
    yedges: np.ndarray
    xc: np.ndarray
    yc: np.ndarray


def compute_fel_2d(
    xvg1: str,
    xvg2: str,
    temp: float,
    bins: int = 100,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    eps: float = 1e-10,
) -> FELResult:
    """Compute 2D FEL from two GROMACS XVG reaction coordinate files."""
    rc1 = np.loadtxt(xvg1, comments=["#", "@"])[:, 1]
    rc2 = np.loadtxt(xvg2, comments=["#", "@"])[:, 1]
    assert len(rc1) == len(rc2), "XVG files have different numbers of frames."

    hist_range = [
        xlim if xlim else (rc1.min(), rc1.max()),
        ylim if ylim else (rc2.min(), rc2.max()),
    ]
    H, xedges, yedges = np.histogram2d(rc1, rc2, bins=bins, range=hist_range, density=True) 
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    extent = [float(xedges[0]), float(xedges[-1]), float(yedges[0]), float(yedges[-1])]

    P = np.where(H > 0, H, eps)
    fel_kj = -R_KJ * temp * np.log(P / P.max())
    fel_kcal = -R_KCAL * temp * np.log(P / P.max())

    return FELResult(
        prob_density=H.T, fel_kj=fel_kj.T, fel_kcal=fel_kcal.T,
        extent=extent, xedges=xedges, yedges=yedges, xc=xc, yc=yc,
    )


def plot_fel_2d(
    result: FELResult,
    label1: str,
    label2: str,
    title: str = "",
    zmax_kj: Optional[float] = None,
    out_contour: Optional[str] = None,
    out_gaussian: Optional[str] = None,
    out_density: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    cmap: str = "jet",
    dpi: int = 300,
    show: bool = False,
):
    """Plot 2D FEL as contourf and Gaussian-interpolated imshow, plus the
    normalised probability density (the data written by write_norm_dat)."""
    X, Y = np.meshgrid(result.xc, result.yc)
    F = result.fel_kj

    if zmax_kj is None:
        F_sampled = F[result.prob_density > 0]
        zmax_kj = float(np.percentile(F_sampled, 99)) if F_sampled.size else 10.0

    F_c = F.copy()
    F_c[result.prob_density == 0] = np.nan
    F_c = np.clip(F_c, 0, zmax_kj)

    F_g = np.clip(F, 0, zmax_kj)

    ticks = list(np.arange(0, zmax_kj, 1.5)) + [zmax_kj]

    fig, ax = plt.subplots(figsize=(6, 5))
    cf = ax.contourf(X, Y, F_c, levels=np.linspace(0, zmax_kj, 41),
                     cmap=cmap, vmin=0, vmax=zmax_kj)
    cs = ax.contour(X, Y, F_c, levels=np.arange(0, zmax_kj + 1, 5),
                    colors="black", linewidths=0.4, alpha=0.5)
    ax.clabel(cs, fmt="%.0f", fontsize=7, inline=True)
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("$\\Delta F$ (kJ/mol)", fontsize=11)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
    ax.set_xlabel(label1, fontsize=11)
    ax.set_ylabel(label2, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=9)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    fig.tight_layout()
    if out_contour:
        fig.savefig(out_contour, dpi=dpi)
        print(f"Saved -> {out_contour}")

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im = ax2.imshow(F_g, interpolation="gaussian", origin="lower",
                    extent=result.extent, aspect="auto", cmap=cmap, vmin=0, vmax=zmax_kj)
    cbar2 = fig2.colorbar(im, ax=ax2, pad=0.02)
    cbar2.set_label("$\\Delta F$ (kJ/mol)", fontsize=11)
    cbar2.set_ticks(ticks)
    cbar2.set_ticklabels([f"{t:.1f}" for t in ticks])
    ax2.set_xlabel(label1, fontsize=11)
    ax2.set_ylabel(label2, fontsize=11)
    if title:
        ax2.set_title(title, fontsize=12)
    ax2.tick_params(labelsize=9)
    if xlim:
        ax2.set_xlim(xlim)
    if ylim:
        ax2.set_ylim(ylim)

    # Add boxes and labels for regions of interest
    # from matplotlib.patches import Rectangle
    # (x0, y0, width, height), label
    # regions = [  # S305I neutral
    #     ((0.51, 0.53, 0.15, 0.10), "A"),
    #     ((0.80, 0.58, 0.15, 0.10), "B"),
    #     ((0.84, 0.80, 0.15, 0.20), "C"),
    # ]

    # regions = [  # S305I charged
    #     ((0.54, 0.48, 0.12, 0.10), "D"),
    #     ((0.82, 0.60, 0.10, 0.10), "E"),
    #     ((1.86, 1.36, 0.20, 0.20), "F"),
    # ]

    # regions = [  # S305N neutral
    #     ((0.54, 0.41, 0.08, 0.07), "A"),
    #     ((0.80, 0.54, 0.15, 0.10), "B"),
    #     ((1.90, 1.45, 0.20, 0.20), "C"),
    # ]

    # regions = [  # S305N charged
    #     ((0.39, 0.39, 0.10, 0.05), "D"),
    #     ((0.84, 0.92, 0.10, 0.10), "E"),
    #     ((0.93, 0.65, 0.15, 0.15), "F"),
    # ]

    # regions = [  # pS305 neutral
    #     ((0.55, 0.52, 0.10, 0.13), "A"),
    #     ((0.51, 1.04, 0.11, 0.15), "B"),
    #     ((0.86, 0.90, 0.12, 0.12), "C"),
    # ]

    # regions = [  # pS305 charged
    #     ((0.54, 0.63, 0.08, 0.20), "D"),
    #     ((0.82, 0.57, 0.12, 0.10), "E"),
    #     ((0.88, 0.80, 0.14, 0.25), "F"),
    # ]

    # for (x0, y0, w, h), label in regions:
    #     ax2.add_patch(Rectangle(
    #         (x0, y0), w, h,
    #         fill=False, edgecolor="white", linewidth=1.5, zorder=5,
    #     ))
    #     ax2.text(
    #         x0-0.03, y0-0.03, label,
    #         color="white", ha="center", va="center",
    #         fontsize=12, fontweight="bold", zorder=6,
    #     )

    fig2.tight_layout()
    if out_gaussian:
        fig2.savefig(out_gaussian, dpi=dpi)
        print(f"Saved -> {out_gaussian}")

    fig3, ax3 = plt.subplots(figsize=(6, 5))
    im3 = ax3.imshow(result.prob_density, interpolation="nearest", origin="lower",
                     extent=result.extent, aspect="auto", cmap=cmap,
                     vmin=0, vmax=result.prob_density.max())
    cbar3 = fig3.colorbar(im3, ax=ax3, pad=0.02)
    cbar3.set_label("Probability Density", fontsize=11)
    ax3.set_xlabel(label1, fontsize=11)
    ax3.set_ylabel(label2, fontsize=11)
    if title:
        ax3.set_title(title, fontsize=12)
    ax3.tick_params(labelsize=9)
    if xlim:
        ax3.set_xlim(xlim)
    if ylim:
        ax3.set_ylim(ylim)
    fig3.tight_layout()
    if out_density:
        fig3.savefig(out_density, dpi=dpi)
        print(f"Saved -> {out_density}")

    if show:
        plt.show()


def write_norm_dat(result: FELResult, path: str):
    """Write normalised probability density grid to file (x, y, prob format)."""
    with open(path, "w") as fout:
        for i, x in enumerate(result.xc):
            for j, y in enumerate(result.yc):
                fout.write(f"{x:.6f} {y:.6f} {result.prob_density[j, i]:.6e}\n")
    print(f"Saved -> {path}")


if __name__ == "__main__":
    for system in SYSTEMS:
        # XVG1 = "/home/jsheppard/research/tau/a99sbdisp_pS/charged_hip/Analysis_200-400ns/dist_k298_q307.xvg"
        # XVG2 = "/home/jsheppard/research/tau/a99sbdisp_pS/charged_hip/Analysis_200-400ns/dist_v300_ps305.xvg"
        # XVG1 = "/home/jsheppard/research/tau/a99sbdisp_pS/charged_hip/Analysis_200-400ns/axis1.xvg"
        # XVG2 = "/home/jsheppard/research/tau/a99sbdisp_pS/charged_hip/Analysis_200-400ns/axis2.xvg"

        # XVG1 = "/home/jsheppard/research/tau/a99sbdisp/s305i/neutral_hie/Analysis_200-400ns/mindist_sidechain_K298_I308.dat"
        # XVG2 = "/home/jsheppard/research/tau/a99sbdisp/s305i/neutral_hie/Analysis_200-400ns/mindist_sidechain_V300_Q307.dat"
        # XVG1 = "/home/jsheppard/research/tau/a99sbdisp/s305i/charged_hip/Analysis_200-400ns/mindist_sidechain_K298_I308.dat"
        # XVG2 = "/home/jsheppard/research/tau/a99sbdisp/s305i/charged_hip/Analysis_200-400ns/mindist_sidechain_V300_Q307.dat"
        # XVG1 = "/home/jsheppard/research/tau/a99sbdisp/s305n/neutral_hie/Analysis_200-400ns/mindist_sidechain_K298_I308.dat"
        # XVG2 = "/home/jsheppard/research/tau/a99sbdisp/s305n/neutral_hie/Analysis_200-400ns/mindist_sidechain_V300_Q307.dat"
        # XVG1 = "/home/jsheppard/research/tau/a99sbdisp/s305n/charged_hip/Analysis_200-400ns/mindist_sidechain_K298_I308.dat"
        # XVG2 = "/home/jsheppard/research/tau/a99sbdisp/s305n/charged_hip/Analysis_200-400ns/mindist_sidechain_V300_Q307.dat"
        # XVG1 = "/home/jsheppard/research/tau/a99sbdisp_pS/neutral_hie/Analysis_200-400ns/mindist_sidechain_K298_I308.dat"
        # XVG2 = "/home/jsheppard/research/tau/a99sbdisp_pS/neutral_hie/Analysis_200-400ns/mindist_sidechain_V300_Q307.dat"
        # XVG1 = "/home/bholdren/systems/collected_xvg_files/mol-13_final_ccs.xvg"
        # XVG2 = "/home/bholdren/systems/collected_xvg_files/beta_strand_pct.xvg"
        system_dir = os.path.join(BASE_DIR, system)
        XVG1 = os.path.join(system_dir, f"{system}_final_lig_ccs.xvg")
        XVG2 = os.path.join(system_dir, "beta_strand_pct.xvg")
        TEMP = 300.0
        # BINS = 20
        BINS = 20
        ZMAX = 24.0
        # ZMAX = None

        # OUT = "/home/jsheppard/research/tau/a99sbdisp/s305i/neutral_hie/Analysis_200-400ns/FEL_2D_sc_mindist.png"
        # OUT2 = "/home/jsheppard/research/tau/a99sbdisp/s305i/neutral_hie/Analysis_200-400ns/FEL_2D_gaussian_sc_mindist.png"
        # OUT3 = "/home/jsheppard/research/tau/a99sbdisp/s305i/neutral_hie/Analysis_200-400ns/FEL_2D_density_sc_mindist.png"
        # OUTDAT = "/home/jsheppard/research/tau/a99sbdisp/s305i/neutral_hie/Analysis_200-400ns/normFEL_sc_mindist.dat"
        # OUT = "/home/jsheppard/research/tau/a99sbdisp/s305i/charged_hip/Analysis_200-400ns/FEL_2D_sc_mindist.png"
        # OUT2 = "/home/jsheppard/research/tau/a99sbdisp/s305i/charged_hip/Analysis_200-400ns/FEL_2D_gaussian_sc_mindist.png"
        # OUT3 = "/home/jsheppard/research/tau/a99sbdisp/s305i/charged_hip/Analysis_200-400ns/FEL_2D_density_sc_mindist.png"
        # OUTDAT = "/home/jsheppard/research/tau/a99sbdisp/s305i/charged_hip/Analysis_200-400ns/normFEL_sc_mindist.dat"
        # OUT = "/home/jsheppard/research/tau/a99sbdisp/s305n/neutral_hie/Analysis_200-400ns/FEL_2D_sc_mindist.png"
        # OUT2 = "/home/jsheppard/research/tau/a99sbdisp/s305n/neutral_hie/Analysis_200-400ns/FEL_2D_gaussian_sc_mindist.png"
        # OUT3 = "/home/jsheppard/research/tau/a99sbdisp/s305n/neutral_hie/Analysis_200-400ns/FEL_2D_density_sc_mindist.png"
        # OUTDAT = "/home/jsheppard/research/tau/a99sbdisp/s305n/neutral_hie/Analysis_200-400ns/normFEL_sc_mindist.dat"
        # OUT = "/home/jsheppard/research/tau/a99sbdisp/s305n/charged_hip/Analysis_200-400ns/FEL_2D_sc_mindist.png"
        # OUT2 = "/home/jsheppard/research/tau/a99sbdisp/s305n/charged_hip/Analysis_200-400ns/FEL_2D_gaussian_sc_mindist.png"
        # OUT3 = "/home/jsheppard/research/tau/a99sbdisp/s305n/charged_hip/Analysis_200-400ns/FEL_2D_density_sc_mindist.png"
        # OUTDAT = "/home/jsheppard/research/tau/a99sbdisp/s305n/charged_hip/Analysis_200-400ns/normFEL_sc_mindist.dat"
        # OUT = "/home/jsheppard/research/tau/a99sbdisp_pS/neutral_hie/Analysis_200-400ns/FEL_2D_sc_mindist.png"
        # OUT2 = "/home/jsheppard/research/tau/a99sbdisp_pS/neutral_hie/Analysis_200-400ns/FEL_2D_gaussian_sc_mindist.png"
        # OUT3 = "/home/jsheppard/research/tau/a99sbdisp_pS/neutral_hie/Analysis_200-400ns/FEL_2D_density_sc_mindist.png"
        # OUTDAT = "/home/jsheppard/research/tau/a99sbdisp_pS/neutral_hie/Analysis_200-400ns/normFEL_sc_mindist.dat"
        OUT = os.path.join(system_dir, "FEL_2D_lig_sc_mindist.png")
        OUT2 = os.path.join(system_dir, "FEL_2D_lig_gaussian_sc_mindist.png")
        OUT3 = os.path.join(system_dir, "FEL_2D_lig_density_sc_mindist.png")
        OUTDAT = os.path.join(system_dir, "normFEL_lig_sc_mindist.dat")

        # OUT = "/home/tank/jsheppardshare/alpha_syn/NACore/r4d/Analysis_200-500ns/zinc-2/FEL_2D.png"
        # OUT2 = "/home/tank/jsheppardshare/alpha_syn/NACore/r4d/Analysis_200-500ns/zinc-2/FEL_2D_gaussian.png"
        # OUT3 = "/home/tank/jsheppardshare/alpha_syn/NACore/r4d/Analysis_200-500ns/zinc-2/FEL_2D_density.png"
        # OUTDAT = "/home/tank/jsheppardshare/alpha_syn/NACore/r4d/Analysis_200-500ns/zinc-2/sasa_beta_normFEL.dat"
        # LABEL1 = "K298-Q307 C$\\alpha$ Distance (nm)"
        # LABEL2 = "V300-pS305 C$\\alpha$ Distance (nm)"
        # TITLE = "pS305 (Charged) Free Energy Landscape"
        # LABEL1 = "d_VIY - d_QVK (A) [Face to R2 Preference]"
        # LABEL2 = "((1/2)(d_VIY + d_QVK)) (A) [PHF6 to R2 Proximity]"
        LABEL1 = "Collision Cross Section (CCS)"
        LABEL2 = "Beta Strand Content (%)"

        # TITLE = "S305I (Neutral) Free Energy Landscape"
        # TITLE = "S305I (Charged) Free Energy Landscape"
        # TITLE = "S305N (Neutral) Free Energy Landscape"
        # TITLE = "S305N (Charged) Free Energy Landscape"
        # TITLE = "pS305 (Neutral) Free Energy Landscape"
        TITLE = "pS305 (Charged) Free Energy Landscape"

        # LABEL1 = "SASA (A^2)"
        # LABEL2 = "Beta Strand Content (%)"
        # TITLE = "ZINC-2 Free Energy Landscape"
        XLIM = None
        YLIM = None
        # XLIM = None
        # YLIM = None
        print(f"\n=== Processing {system} ===")
        result = compute_fel_2d(XVG1, XVG2, TEMP, bins=BINS, xlim=XLIM, ylim=YLIM)

        rc1 = np.loadtxt(XVG1, comments=["#", "@"])[:, 1]
        rc2 = np.loadtxt(XVG2, comments=["#", "@"])[:, 1]
        print(f"Frames loaded : {len(rc1):,}")
        print(f"RC1  [{LABEL1}]  min={rc1.min():.3f}  max={rc1.max():.3f}  mean={rc1.mean():.3f}")
        print(f"RC2  [{LABEL2}]  min={rc2.min():.3f}  max={rc2.max():.3f}  mean={rc2.mean():.3f}")

        write_norm_dat(result, OUTDAT)
        plot_fel_2d(result, LABEL1, LABEL2, title=TITLE, zmax_kj=ZMAX,
                    out_contour=OUT, out_gaussian=OUT2, out_density=OUT3,
                    xlim=XLIM, ylim=YLIM)

        plt.close('all')
