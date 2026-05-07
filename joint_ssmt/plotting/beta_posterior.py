"""
Beta Posterior Scatter Plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Ellipse
from scipy import stats
from typing import Optional, List

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


COLORS = {
    'plv': '#2E86AB',
    'sfc': '#A23B72',
    'joint': '#F18F01',
    'coupled': '#F18F01',
    'uncoupled': '#888888',
}


def set_style(font_size=9):
    if HAS_SEABORN:
        sns.set(style="ticks", context="paper", font="sans-serif",
                rc={"font.size": font_size,
                    "axes.titlesize": font_size,
                    "axes.labelsize": font_size,
                    "axes.linewidth": 0.5,
                    "lines.linewidth": 1.5,
                    "xtick.labelsize": font_size,
                    "ytick.labelsize": font_size,
                    "legend.fontsize": font_size,
                    "legend.frameon": False})
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42


def plot_single_panel(ax, beta_R_samples, beta_I_samples, freq_hz, unit_idx,
                      is_top_row, label_top, label_bottom):
    mean_R = np.mean(beta_R_samples)
    mean_I = np.mean(beta_I_samples)

    scatter_color = COLORS['coupled'] if is_top_row else COLORS['uncoupled']
    ax.scatter(beta_R_samples, beta_I_samples, alpha=0.3, s=8, c=scatter_color)

    ax.scatter([mean_R], [mean_I], c='#333333', s=80, marker='x', linewidths=1.5,
               label='E[β]', zorder=10)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.scatter([0], [0], c='black', s=60, marker='o', zorder=9, label='Origin')

    cov = np.cov(beta_R_samples, beta_I_samples)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    chi2_val = stats.chi2.ppf(0.95, 2)
    width, height = 2 * np.sqrt(eigenvalues * chi2_val)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    ellipse = Ellipse((mean_R, mean_I), width, height, angle=angle,
                      fill=False, color=scatter_color, linestyle='-', linewidth=1.5,
                      label='95% CI')
    ax.add_patch(ellipse)

    status = label_top if is_top_row else label_bottom
    ax.set_title(f'Unit {unit_idx}, {freq_hz:.0f} Hz\n({status})',
                 fontsize=9, color=scatter_color, fontweight='bold')
    ax.set_xlabel('βR')
    ax.set_ylabel('βI')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(fontsize=7, loc='upper right')


def plot_beta_posterior_scatter(
    beta_trace: np.ndarray,
    freqs: np.ndarray,
    output_path: str,
    unit_idx: int = 0,
    burn_in_frac: float = 0.7,
    freqs_true: Optional[np.ndarray] = None,
    masks: Optional[np.ndarray] = None,
    freq_list: Optional[List[float]] = None,
    test_type: str = 'wald',
    alpha: float = 0.05,
    n_top: int = 3,
    n_bottom: int = 3,
) -> str:
    set_style(font_size=9)

    n_samples, S, P = beta_trace.shape
    J = len(freqs)
    burn = int(burn_in_frac * n_samples)
    post = beta_trace[burn:]

    bR = post[:, :, 1:1+J]
    bI = post[:, :, 1+J:1+2*J]

    # Case 1: User-specified frequency list
    if freq_list is not None:
        idx_list = [np.argmin(np.abs(freqs - f)) for f in freq_list]
        n_cols = max(len(idx_list), 1)
        fig, axes = plt.subplots(1, n_cols, figsize=(3.5*n_cols, 3.5))
        if n_cols == 1:
            axes = [axes]

        for i, j in enumerate(idx_list):
            plot_single_panel(axes[i], bR[:, unit_idx, j], bI[:, unit_idx, j],
                              freqs[j], unit_idx, is_top_row=True,
                              label_top="Selected", label_bottom="Selected")

        plt.suptitle(f'β Posterior Samples (Unit {unit_idx})\nEllipse = 95% CI, × = posterior mean',
                     fontsize=10, y=0.98)
        if HAS_SEABORN:
            sns.despine()
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        if output_path is not None:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        return plt.gcf()

    # Case 2: Ground truth available
    if freqs_true is not None and masks is not None:
        idx_map = np.array([np.argmin(np.abs(freqs - f)) for f in freqs_true])

        top_indices = []
        bottom_indices = []
        for jt, ft in enumerate(freqs_true):
            j_dense = idx_map[jt]
            if masks[unit_idx, jt]:
                top_indices.append((j_dense, ft))
            else:
                bottom_indices.append((j_dense, ft))

        noise_freqs = [f for f in freqs if not any(np.abs(f - ft) < 1.5 for ft in freqs_true)]
        for f in noise_freqs:
            j_dense = np.argmin(np.abs(freqs - f))
            bottom_indices.append((j_dense, f))

        top_indices = top_indices[:n_top]
        bottom_indices = bottom_indices[:n_bottom]
        label_top, label_bottom = "Coupled", "Uncoupled"

    # Case 3: No ground truth - use statistical test
    else:
        from .coupling_stats import wald_test, phase_concentration_test

        if test_type == 'rayleigh':
            _, pvals, _ = phase_concentration_test(beta_trace, J, burn_in_frac=burn_in_frac)
        else:
            _, pvals = wald_test(beta_trace, J, burn_in_frac=burn_in_frac)

        pval_unit = pvals[unit_idx]
        sig_idx = np.where(pval_unit < alpha)[0]
        nonsig_idx = np.where(pval_unit >= alpha)[0]

        sig_order = np.argsort(pval_unit[sig_idx])
        sig_idx = sig_idx[sig_order]

        top_indices = [(j, freqs[j]) for j in sig_idx[:n_top]]
        bottom_indices = [(j, freqs[j]) for j in nonsig_idx[:n_bottom]]
        label_top, label_bottom = "Significant", "Non-significant"

    # Plot
    n_top_plot = len(top_indices)
    n_bottom_plot = len(bottom_indices)
    n_cols = max(n_top_plot, n_bottom_plot, 1)

    fig, axes = plt.subplots(2, n_cols, figsize=(3.5*n_cols, 7))

    for i, (j, freq_hz) in enumerate(top_indices):
        ax = axes[0, i]
        plot_single_panel(ax, bR[:, unit_idx, j], bI[:, unit_idx, j],
                          freq_hz, unit_idx, is_top_row=True,
                          label_top=label_top, label_bottom=label_bottom)

    for i in range(n_top_plot, n_cols):
        axes[0, i].axis('off')

    for i, (j, freq_hz) in enumerate(bottom_indices):
        ax = axes[1, i]
        plot_single_panel(ax, bR[:, unit_idx, j], bI[:, unit_idx, j],
                          freq_hz, unit_idx, is_top_row=False,
                          label_top=label_top, label_bottom=label_bottom)

    for i in range(n_bottom_plot, n_cols):
        axes[1, i].axis('off')

    if n_top_plot > 0:
        axes[0, 0].annotate(label_top, xy=(-0.35, 0.5), xycoords='axes fraction',
                            fontsize=11, fontweight='bold', color=COLORS['coupled'],
                            rotation=90, va='center', ha='center')
    if n_bottom_plot > 0:
        axes[1, 0].annotate(label_bottom, xy=(-0.35, 0.5), xycoords='axes fraction',
                            fontsize=11, fontweight='bold', color=COLORS['uncoupled'],
                            rotation=90, va='center', ha='center')

    plt.suptitle(f'β Posterior Samples (Unit {unit_idx})\nEllipse = 95% CI, × = posterior mean',
                 fontsize=10, y=0.98)

    if HAS_SEABORN:
        sns.despine()
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, top=0.88)

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    return plt.gcf()