import argparse
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Make sure this is imported from wherever your plot_FEL_2D.py script is located
from plot_FEL_2D import compute_fel_2d

# =========================
# CONFIG
# =========================
BINS = 20
FIGSIZE_PER_PANEL = 8.0
CMAP = "jet"
DPI = 300

@dataclass(frozen=True)

class SystemSpec:
    title: str
    xvg1: str
    xvg2: str
    T: float
    label1: str = 'CCS (Å²)'
    label2: str = 'Beta Strand (%)'
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    zmax_kj: Optional[float] = None

# =========================
# PLOTTING
# =========================
def resolve_zmax(spec: SystemSpec, result) -> float:
    if spec.zmax_kj is not None:
        return spec.zmax_kj
    F_sampled = result.fel_kj[result.prob_density > 0]
    return float(np.percentile(F_sampled, 99)) if F_sampled.size else 10.0


def panel(ax, spec: SystemSpec, result, zmax_kj: float,
          cmap: str = CMAP, show_ylabel: bool = True):
    F = result.fel_kj
    F_g = np.clip(F, 0, zmax_kj)
    
    im = ax.imshow(F_g, interpolation="gaussian", origin="lower",
                    extent=result.extent, aspect="auto", cmap=cmap,
                    vmin=0, vmax=zmax_kj)

    ax.set_title(spec.title, fontsize=28)
    ax.set_xlabel(spec.label1, fontsize=24)
    if show_ylabel:
        ax.set_ylabel(spec.label2, fontsize=24)
    ax.tick_params(labelsize=14)
    if spec.xlim:
        ax.set_xlim(spec.xlim)
    if spec.ylim:
        ax.set_ylim(spec.ylim)

    file_count = 0
    best_match_dir = os.path.join(BASE_DIR, "best_match")
    if os.path.exists(best_match_dir):
        for dirpath, dirnames, filenames in os.walk(best_match_dir):
            for fname in filenames:
                if re.search(spec.title.lower()[:6], fname) and re.search(r"cluster1", fname):
                    pattern = r"\d+\.\d+"
                    files = re.findall(pattern, fname)
                    if len(files) >= 2:
                        x0, y0 = float(files[0]), float(files[1])
                        file_count += 1
                        draw_box(ax, x0, y0, 100, 5, f"{file_count}")

    return im

def draw_box(ax, x0, y0, width, height, label):
    x0 -= 0.5*width
    y0 -= 0.5*height
    ax.add_patch(Rectangle(
        (x0, y0), width, height,
        fill=False, edgecolor="white", linewidth=1.5, zorder=5,
    ))
    ax.text(
        x0-0.5*width, y0+0.5*height, label,
        color="white", ha="center", va="center",
        fontsize=25, fontweight="bold", zorder=6,
    )

def plot_for_nmer(n: int, base_dir: str, mols: List[str]):
    # Automatically build the SYSTEMS list for this N-mer size
    SYSTEMS: List[SystemSpec] = []
    for mol in mols:
        xvg1_path = os.path.join(base_dir, "collected_xvg_files", f"{mol}_n{n}_ccs.xvg")
        xvg2_path = os.path.join(base_dir, "collected_xvg_files", f"{mol}_n{n}_beta.xvg")
        
        # Only add it if the file exists and has size > 0
        if os.path.exists(xvg1_path) and os.path.getsize(xvg1_path) > 0:
            SYSTEMS.append(
                SystemSpec(
                    title=f"{mol.upper()} (N={n})",
                    xvg1=xvg1_path,
                    xvg2=xvg2_path,
                    T=300.0,
                    zmax_kj=18.0,
                )
            )

    if not SYSTEMS:
        print(f"No valid .xvg files found for N={n}. Skipping...")
        return

    n_sys = len(SYSTEMS)
    
    # Calculate grid size dynamically
    if n_sys <= 4:
        ncols = 2
    elif n_sys <= 6:
        ncols = 3
    elif n_sys <= 8:
        ncols = 4
    elif n_sys == 9:
        ncols = 3
    else:
        ncols = int(np.ceil(np.sqrt(n_sys)))
        
    nrows = int(np.ceil(n_sys / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(FIGSIZE_PER_PANEL * ncols, FIGSIZE_PER_PANEL * nrows))
    axes_2d = np.atleast_2d(axes)
    if nrows == 1 and ncols == 1:
        axes_2d = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes_2d = axes_2d.reshape(nrows, ncols)

    results = []
    valid_systems = []
    
    for spec in SYSTEMS:
        try:
            res = compute_fel_2d(spec.xvg1, spec.xvg2, spec.T, bins=BINS,
                                 xlim=spec.xlim, ylim=spec.ylim)
            results.append(res)
            valid_systems.append(spec)
        except Exception as e:
            print(f"Error processing {spec.title}: {e}")

    if not results:
        print(f"Failed to process any systems for N={n}. Skipping...")
        plt.close(fig)
        return

    zmax_kj = max(resolve_zmax(spec, result)
                  for spec, result in zip(valid_systems, results))

    im = None
    for idx, (spec, result) in enumerate(zip(valid_systems, results)):
        r, c = divmod(idx, ncols)
        im = panel(axes_2d[r, c], spec, result, zmax_kj, show_ylabel=(c == 0))

    # Delete any unused subplots
    for idx in range(len(valid_systems), nrows * ncols):
        r, c = divmod(idx, ncols)
        fig.delaxes(axes_2d[r, c])

    fig.tight_layout()

    if im is not None:
        ticks = list(np.arange(0, zmax_kj, 3.0)) + [zmax_kj]
        cbar = fig.colorbar(im, ax=axes_2d, location="right", pad=0.02, shrink=0.9)
        cbar.set_label("$\\Delta F$ (kJ/mol)", fontsize=24)
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize=14)

    OUT = os.path.join(base_dir, f"FEL_2D_grid_n{n}.png")
    fig.savefig(OUT, dpi=DPI)
    print(f"Saved -> {OUT}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot FEL grids for different N-mers.")
    parser.add_argument("--base-dir", required=True, help="Base directory of the project")
    parser.add_argument("--mols", nargs="+", required=True, help="List of molecules/systems to plot")
    args = parser.parse_args()

    print("Starting generation of multi-panel FEL plots for N=2 to 6...")
    for n in range(2, 7):
        print(f"\n--- Generating FEL grid for N={n} ---")
        plot_for_nmer(n, args.base_dir, args.mols)


if __name__ == "__main__":
    main()
