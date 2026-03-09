#!/usr/bin/env python3
"""
plot_heatmaps.py - Effect size and p-value heatmaps for manuscript
"""

import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from scipy import stats
from typing import Optional, Sequence, Tuple
import os

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# =============================================================================
# STYLE - exactly from heatmaps.py
# =============================================================================

def set_publication_style(font_size: int = 8):
    """Set publication-quality figure style."""
    if HAS_SEABORN:
        sns.set(
            style="ticks",
            context="paper",
            font="serif",
            rc={
                "font.size": font_size,
                "axes.titlesize": font_size,
                "axes.labelsize": font_size,
                "axes.linewidth": 0.5,
                "lines.linewidth": 1.5,
                "xtick.labelsize": font_size,
                "ytick.labelsize": font_size,
                "legend.fontsize": font_size,
                "legend.frameon": False,
                "xtick.major.size": 2.5,
                "ytick.major.size": 2.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "savefig.transparent": True,
            }
        )
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

# =============================================================================
# WALD TEST - exactly from coupling_stats.py
# =============================================================================

def wald_test(beta_samples, J, burn_in_frac=0.5):
    """Compute Wald statistics and p-values."""
    n_samples, S, P = beta_samples.shape
    burn_in = int(burn_in_frac * n_samples)
    samples = beta_samples[burn_in:]

    beta_R = samples[:, :, 1 : 1 + J]
    beta_I = samples[:, :, 1 + J : 1 + 2 * J]

    mean_R = beta_R.mean(axis=0)
    mean_I = beta_I.mean(axis=0)
    var_R = beta_R.var(axis=0)
    var_I = beta_I.var(axis=0)

    W = np.zeros((S, J))
    pval = np.ones((S, J))

    for s in range(S):
        for j in range(J):
            if var_R[s, j] > 1e-10 and var_I[s, j] > 1e-10:
                W[s, j] = (mean_R[s, j] ** 2 / var_R[s, j] + mean_I[s, j] ** 2 / var_I[s, j])
                pval[s, j] = 1 - stats.chi2.cdf(W[s, j], df=2)
            else:
                W[s, j] = 0.0
                pval[s, j] = 1.0

    return W, pval

# =============================================================================
# HEATMAP FUNCTIONS - from heatmaps.py, star fixed
# =============================================================================

def plot_effect_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    freqs: np.ndarray,
    title: str,
    *,
    true_freqs: Optional[np.ndarray] = None,
    masks: Optional[np.ndarray] = None,
    log_scale: bool = False,
    cmap: str = "Reds",
    vmax_percentile: float = 99,
    show_stars: bool = True,
) -> plt.cm.ScalarMappable:
    """Plot effect size heatmap with optional ground truth markers."""
    if log_scale:
        plot_values = np.log10(values + 1)
        vmax = np.percentile(plot_values[np.isfinite(plot_values)], vmax_percentile)
    else:
        plot_values = values
        vmax = np.percentile(plot_values[np.isfinite(plot_values)], vmax_percentile)

    freq_step = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    extent = [
        freqs[0] - freq_step / 2,
        freqs[-1] + freq_step / 2,
        values.shape[0] - 0.5,
        -0.5,
    ]

    im = ax.imshow(plot_values, aspect="auto", cmap=cmap, vmin=0, vmax=vmax, extent=extent)

    ax.set_ylabel("Unit")

    # Show all y-ticks (all units)
    n_units = values.shape[0]
    ax.set_yticks(np.arange(n_units))
    ax.set_yticklabels([str(unit) for unit in range(n_units)])

    # Title inside subplot, upper right
    ax.text(
        0.97,
        0.92,
        title,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="right",
        va="top",
        path_effects=[patheffects.withStroke(linewidth=2, foreground="white")],
    )

    # Star markers using scatter (not text)
    if show_stars and masks is not None and true_freqs is not None:
        for s in range(masks.shape[0]):
            for j, f in enumerate(true_freqs):
                if j < masks.shape[1] and masks[s, j]:
                    ax.scatter(
                        f,
                        s,
                        marker="*",
                        s=15,
                        c="white",
                        edgecolors="black",
                        linewidths=0.5,
                        zorder=10,
                    )

    return im


def plot_pval_heatmap(
    ax: plt.Axes,
    pvals: np.ndarray,
    freqs: np.ndarray,
    title: str,
    *,
    true_freqs: Optional[np.ndarray] = None,
    masks: Optional[np.ndarray] = None,
    cmap: str = "hot_r",
    vmin: float = 0,
    vmax: float = 5,
    alpha: float = 0.05,
    show_stars: bool = True,
) -> plt.cm.ScalarMappable:
    """Plot -log10(p-value) heatmap."""
    log_p = -np.log10(np.clip(pvals, 1e-20, 1))

    freq_step = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    extent = [
        freqs[0] - freq_step / 2,
        freqs[-1] + freq_step / 2,
        pvals.shape[0] - 0.5,
        -0.5,
    ]

    im = ax.imshow(log_p, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)

    ax.set_ylabel("Unit")

    # Title inside subplot, upper right
    ax.text(
        0.97,
        0.92,
        title,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="right",
        va="top",
        path_effects=[patheffects.withStroke(linewidth=2, foreground="white")],
    )

    # Star markers using scatter (not text)
    if show_stars and masks is not None and true_freqs is not None:
        for s in range(masks.shape[0]):
            for j, f in enumerate(true_freqs):
                if j < masks.shape[1] and masks[s, j]:
                    ax.scatter(
                        f,
                        s,
                        marker="*",
                        s=30,
                        c="white",
                        edgecolors="black",
                        linewidths=0.5,
                        zorder=10,
                    )

    return im

# =============================================================================
# MANUSCRIPT FIGURE: 3 rows x 2 columns
# =============================================================================

def plot_heatmaps_comparison(
    effect_dict: dict,
    pval_dict: dict,
    freqs: np.ndarray,
    output_path: str,
    *,
    true_freqs: Optional[np.ndarray] = None,
    masks: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    log_scale_keys: Sequence[str] = (),
    figsize: Tuple[float, float] = (7.0, 2.8),
):
    """
    Plot effect size and p-value heatmaps in 3x2 grid.
    """
    set_publication_style()

    method_names = list(effect_dict.keys())
    n_methods = len(method_names)

    fig, axes = plt.subplots(n_methods, 2, figsize=figsize, sharex=True)

    # First pass: compute global vmax for p-values
    vmax_p_global = 0
    for name in method_names:
        log_p = -np.log10(np.clip(pval_dict[name], 1e-10, 1))
        vmax_p = max(np.percentile(log_p[np.isfinite(log_p)], 95), -np.log10(alpha) + 1)
        vmax_p_global = max(vmax_p_global, vmax_p)
    print(f"vmax_p_global: {vmax_p_global}")
    vmax_p_global = min(vmax_p_global, 5)
    print(f"vmax_p_global: {vmax_p_global}")

    pval_im_list = []

    for row_idx, name in enumerate(method_names):
        # ----- Left: Effect size -----
        ax_effect = axes[row_idx, 0]
        use_log = name in log_scale_keys

        im_effect = plot_effect_heatmap(
            ax_effect,
            effect_dict[name],
            freqs,
            name,
            true_freqs=true_freqs,
            masks=masks,
            log_scale=use_log,
        )
        cbar_effect = fig.colorbar(im_effect, ax=ax_effect, fraction=0.02, pad=0.01)
        cbar_effect.ax.tick_params(labelsize=6)
        if use_log:
            cbar_effect.set_label(r"$\log_{10}$", fontsize=6)

        # ----- Right: P-value (use global vmax) -----
        ax_pval = axes[row_idx, 1]

        im_pval = plot_pval_heatmap(
            ax_pval,
            pval_dict[name],
            freqs,
            name,
            true_freqs=true_freqs,
            masks=masks,
            vmin=0,
            vmax=vmax_p_global,
            alpha=alpha,
        )
        pval_im_list.append(im_pval)

        # Add simple y-ticks without labels on the right column
        n_units = effect_dict[name].shape[0]
        ax_pval.set_yticks(np.arange(n_units))
        ax_pval.set_yticklabels([])  # No numbers
        ax_pval.set_ylabel("")
        # Optionally, make ticks visible but without numbers (already handled above)

    # X-axis label only on bottom row
    axes[-1, 0].set_xlabel("Frequency (Hz)")
    axes[-1, 1].set_xlabel("Frequency (Hz)")

    # Column headers
    axes[0, 0].annotate(
        "Effect Size",
        xy=(0.5, 1.08),
        xycoords="axes fraction",
        fontsize=9,
        fontweight="bold",
        ha="center",
    )
    axes[0, 1].annotate(
        r"$p$-value",
        xy=(0.5, 1.08),
        xycoords="axes fraction",
        fontsize=9,
        fontweight="bold",
        ha="center",
    )

    if HAS_SEABORN:
        sns.despine()
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.25, wspace=0.15, right=0.88)

    # Add shared colorbar for p-value column spanning all rows
    # Get positions of top and bottom right axes after tight_layout
    pos_top = axes[0, 1].get_position()
    pos_bot = axes[-1, 1].get_position()
    
    # Create colorbar axis: [left, bottom, width, height]
    plt.subplots_adjust(top=0.92, hspace=0.25, wspace=0.2, right=0.88)

    # Add shared colorbar for p-value column spanning all rows
    pos_top = axes[0, 1].get_position()
    pos_bot = axes[-1, 1].get_position()
    
    # Create colorbar axis: [left, bottom, width, height] - width=0.012 to match left column
    cax = fig.add_axes([0.90, pos_bot.y0 + 0.05, 0.005, pos_top.y1 - pos_bot.y0 - 0.05])
    cbar_pval = fig.colorbar(pval_im_list[0], cax=cax)
    cbar_pval = fig.colorbar(pval_im_list[0], cax=cax)
    cbar_pval.ax.tick_params(labelsize=6)
    cbar_pval.set_label(r"$-\log_{10}(p)$", fontsize=6)

    # Mark significance threshold on colorbar
    sig_line = -np.log10(alpha)
    if sig_line <= vmax_p_global:
        cbar_pval.ax.axhline(sig_line, color="cyan", lw=1.5)
        cbar_pval.ax.text(1.5, sig_line, f"α={alpha}", color="cyan", fontsize=5, va="center")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")
    return output_path

# =============================================================================
# MAIN
# =============================================================================
import argparse
# pasrse in base_dir 

def main():
    # Paths
    parser = argparse.ArgumentParser(description="Plot effect size and p-value heatmaps for manuscript.")
    parser.add_argument('--base_dir', type=str, default="./final_figure_data", help="Base directory for data/results (default: current directory)")
    args = parser.parse_args()
    base_dir = args.base_dir
    
    results_dir = os.path.join(base_dir, "results")
    output_dir = os.path.join(base_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    ALPHA = 0.05
    BURNIN = 0.6

    # Load simulation data (ground truth)
    sim_data_path = os.path.join(base_dir, "sim_data.pkl")
    print(f"Loading simulation data from: {sim_data_path}")
    with open(sim_data_path, "rb") as f:
        sim_data = pickle.load(f)

    freqs_true = np.asarray(sim_data["freqs_hz"], dtype=float)
    masks = np.asarray(sim_data["masks"], dtype=bool)

    # Load coupling results
    print(f"Loading coupling results from: {results_dir}")
    coupling = np.load(os.path.join(results_dir, "coupling.npz"), allow_pickle=True)
    beta_trace = coupling["beta_trace"]
    beta_mag = coupling["beta_mag"]
    freqs = coupling["freqs"]

    # Compute Wald p-values
    print("Computing Wald test...")
    wald_W, wald_pval = wald_test(beta_trace, len(freqs), burn_in_frac=BURNIN)

    # Load traditional methods
    trad_path = os.path.join(results_dir, "traditional_methods.pkl")
    print(f"Loading traditional methods from: {trad_path}")
    with open(trad_path, "rb") as f:
        trad = pickle.load(f)

    plv_val = trad["plv"]["values"]
    plv_pval = trad["plv"]["pval"]
    sfc_val = trad["sfc"]["values"]
    sfc_pval = trad["sfc"]["pval"]

    # Resample if frequency grids don't match
    J = len(freqs)
    if plv_val.shape[1] != J:
        from scipy.interpolate import interp1d

        plv_freqs = trad.get("freqs", np.linspace(freqs[0], freqs[-1], plv_val.shape[1]))
        S = plv_val.shape[0]

        plv_val_new = np.zeros((S, J))
        plv_pval_new = np.zeros((S, J))
        sfc_val_new = np.zeros((S, J))
        sfc_pval_new = np.zeros((S, J))

        for s in range(S):
            plv_val_new[s] = interp1d(plv_freqs, plv_val[s], kind="nearest", fill_value="extrapolate")(freqs)
            plv_pval_new[s] = interp1d(plv_freqs, plv_pval[s], kind="nearest", fill_value="extrapolate")(freqs)
            sfc_val_new[s] = interp1d(plv_freqs, sfc_val[s], kind="nearest", fill_value="extrapolate")(freqs)
            sfc_pval_new[s] = interp1d(plv_freqs, sfc_pval[s], kind="nearest", fill_value="extrapolate")(freqs)

        plv_val, plv_pval = plv_val_new, plv_pval_new
        sfc_val, sfc_pval = sfc_val_new, sfc_pval_new

    print(f"  PLV: {plv_val.shape}, SFC: {sfc_val.shape}, beta_mag: {beta_mag.shape}")

    # Build dicts
    effect_dict = {
        "PLV": plv_val,
        "SFC": sfc_val,
        r"Joint $|\mathrm{E}[\beta]|$": beta_mag,
    }

    pval_dict = {
        "PLV": plv_pval,
        "SFC": sfc_pval,
        r"Joint $|\mathrm{E}[\beta]|$": wald_pval,
    }

    # Generate figure
    print("\nGenerating heatmaps...")
    plot_heatmaps_comparison(
        effect_dict=effect_dict,
        pval_dict=pval_dict,
        freqs=freqs,
        output_path=os.path.join(output_dir, "heatmaps_comparison.pdf"),
        true_freqs=freqs_true,
        masks=masks,
        alpha=ALPHA,
        log_scale_keys=(),
    )

    print("\nDone!")


if __name__ == "__main__":
    main()