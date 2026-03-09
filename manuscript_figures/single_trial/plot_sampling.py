#!/usr/bin/env python3
"""
plot_sampling.py - Beta posterior scatter for manuscript
"""

import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats
import os

# =============================================================================
# STYLE SETUP - LaTeX fonts
# =============================================================================

def setup_latex_style():
    """Configure matplotlib for LaTeX-style fonts."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
        'mathtext.fontset': 'cm',
        'font.size': 8,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'axes.linewidth': 0.5,
        'lines.linewidth': 1.0,
        'patch.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 2,
        'ytick.major.size': 2,
        'xtick.major.pad': 2,
        'ytick.major.pad': 2,
        'axes.labelpad': 1,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })

COLORS = {
    'coupled': '#F18F01',
    'uncoupled': '#888888',
}

# =============================================================================
# BETA POSTERIOR SCATTER
# =============================================================================

def plot_beta_posterior_scatter_manuscript(
    beta_trace: np.ndarray,
    freqs: np.ndarray,
    output_path: str,
    unit_idx: int = 0,
    burn_in_frac: float = 0.6,
    freqs_true: np.ndarray = None,
    masks: np.ndarray = None,
    n_cols: int = 3,
    coupled_freqs_plot: list = None,
    uncoupled_freqs_plot: list = None,
):
    """
    Plot beta posterior scatter for manuscript.
    """
    setup_latex_style()
    
    n_samples, S, P = beta_trace.shape
    J = len(freqs)
    burn = int(burn_in_frac * n_samples)
    post = beta_trace[burn:]
    
    bR = post[:, :, 1:1+J]
    bI = post[:, :, 1+J:1+2*J]
    
    # Map true frequencies to analysis grid
    idx_map = np.array([np.argmin(np.abs(freqs - f)) for f in freqs_true])
    
    # Build coupled/uncoupled indices
    if coupled_freqs_plot is not None:
        coupled_indices = []
        for f in coupled_freqs_plot:
            j_dense = np.argmin(np.abs(freqs - f))
            coupled_indices.append((j_dense, f))
    else:
        coupled_indices = []
        for jt, ft in enumerate(freqs_true):
            j_dense = idx_map[jt]
            if masks[unit_idx, jt]:
                coupled_indices.append((j_dense, ft))
        coupled_indices = coupled_indices[:n_cols]
    
    if uncoupled_freqs_plot is not None:
        uncoupled_indices = []
        for f in uncoupled_freqs_plot:
            j_dense = np.argmin(np.abs(freqs - f))
            uncoupled_indices.append((j_dense, f))
    else:
        uncoupled_indices = []
        for jt, ft in enumerate(freqs_true):
            j_dense = idx_map[jt]
            if not masks[unit_idx, jt]:
                uncoupled_indices.append((j_dense, ft))
        noise_freqs = [f for f in freqs if not any(np.abs(f - ft) < 2 for ft in freqs_true)]
        for f in noise_freqs[:max(0, n_cols - len(uncoupled_indices))]:
            j_dense = np.argmin(np.abs(freqs - f))
            uncoupled_indices.append((j_dense, f))
        uncoupled_indices = uncoupled_indices[:n_cols]
    
    # Figure
    fig_width = 7.0
    fig_height = 2.3
    fig, axes = plt.subplots(2, n_cols, figsize=(fig_width, fig_height))
    
    def plot_panel(ax, j, freq_hz, is_coupled, show_legend=False, show_ylabel=False, show_xlabel=True, show_title=True):
        samples_R = bR[:, unit_idx, j]
        samples_I = bI[:, unit_idx, j]
        
        mean_R = np.mean(samples_R)
        mean_I = np.mean(samples_I)
        
        color = COLORS['coupled'] if is_coupled else COLORS['uncoupled']
        
        ax.scatter(samples_R, samples_I, alpha=0.3, s=6, c=color,
                   edgecolors='none', rasterized=True)
        
        ax.scatter([mean_R], [mean_I], c='#333333', s=50, marker='x',
                   linewidths=1.2, zorder=10, label=r'E[$\beta$]')
        
        ax.axhline(0, color='gray', linestyle='--', alpha=0.4, linewidth=0.4)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.4, linewidth=0.4)
        ax.scatter([0], [0], c='black', s=40, marker='o', zorder=9, label='Origin')
        
        cov = np.cov(samples_R, samples_I)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        chi2_val = stats.chi2.ppf(0.95, 2)
        width = 2 * np.sqrt(eigenvalues[0] * chi2_val)
        height = 2 * np.sqrt(eigenvalues[1] * chi2_val)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        
        ellipse = Ellipse((mean_R, mean_I), width, height, angle=angle,
                          fill=False, color=color, linestyle='-', linewidth=1.0,
                          label='95% CI')
        ax.add_patch(ellipse)
        
        # Title only shows frequency (row label shows Coupled/Uncoupled)
        if show_title:
            ax.set_title(f'{freq_hz:.0f} Hz', fontsize=8, color=color, fontweight='bold')
        
        if show_xlabel:
            ax.set_xlabel(r'$\beta_R$')
        else:
            ax.set_xticklabels([])
        
        if show_ylabel:
            ax.set_ylabel(r'$\beta_I$', labelpad=0)
        else:
            ax.set_ylabel('')
        
        ax.set_aspect('equal', adjustable='datalim')
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        
        if show_legend:
            ax.legend(fontsize=5, loc='upper right', frameon=True,
                      fancybox=False, edgecolor='gray', framealpha=0.9,
                      handletextpad=0.3, borderpad=0.3)
        
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    
    # Plot coupled (top row) - no x labels, with titles
    for i, (j, freq_hz) in enumerate(coupled_indices):
        show_legend = (i == n_cols - 1)
        show_ylabel = (i == 0)
        plot_panel(axes[0, i], j, freq_hz, is_coupled=True,
                   show_legend=show_legend, show_ylabel=show_ylabel, 
                   show_xlabel=False, show_title=True)
    
    for i in range(len(coupled_indices), n_cols):
        axes[0, i].axis('off')
    
    # Plot uncoupled (bottom row) - with x labels, with titles
    for i, (j, freq_hz) in enumerate(uncoupled_indices):
        show_legend = (i == n_cols - 1)
        show_ylabel = (i == 0)
        plot_panel(axes[1, i], j, freq_hz, is_coupled=False,
                   show_legend=show_legend, show_ylabel=show_ylabel, 
                   show_xlabel=True, show_title=True)
    
    for i in range(len(uncoupled_indices), n_cols):
        axes[1, i].axis('off')

    # Row labels
    axes[0, 0].annotate('Coupled', xy=(-0.35, 0.5), xycoords='axes fraction',
                        fontsize=9, fontweight='bold', color=COLORS['coupled'],
                        rotation=90, va='center', ha='center')
    axes[1, 0].annotate('Uncoupled', xy=(-0.35, 0.5), xycoords='axes fraction',
                        fontsize=9, fontweight='bold', color=COLORS['uncoupled'],
                        rotation=90, va='center', ha='center')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, wspace=0.30, hspace=0.35)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Paths
    base_dir = './final_figure_data'
    results_dir = os.path.join(base_dir, 'results')
    output_dir = os.path.join(base_dir, 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load simulation data (ground truth)
    sim_data_path = os.path.join(base_dir, 'sim_data.pkl')
    print(f"Loading simulation data from: {sim_data_path}")
    with open(sim_data_path, 'rb') as f:
        sim_data = pickle.load(f)
    
    freqs_true = np.asarray(sim_data['freqs_hz'], dtype=float)
    masks = np.asarray(sim_data['masks'], dtype=bool)
    
    # Load coupling results
    print(f"Loading coupling results from: {results_dir}")
    coupling = np.load(os.path.join(results_dir, 'coupling.npz'), allow_pickle=True)
    beta_trace = coupling['beta_trace']
    freqs = coupling['freqs']
    
    print(f"  beta_trace: {beta_trace.shape}")
    print(f"  freqs: {freqs.shape}")
    print(f"  freqs_true: {freqs_true}")
    print(f"  masks: {masks.shape}")
    
    # Generate figure
    print("\nGenerating beta posterior scatter...")
    plot_beta_posterior_scatter_manuscript(
        beta_trace=beta_trace,
        freqs=freqs,
        output_path=os.path.join(output_dir, 'beta_posterior_scatter.pdf'),
        unit_idx=0,
        burn_in_frac=0.6,
        freqs_true=freqs_true,
        masks=masks,
        n_cols=3,
        coupled_freqs_plot=[19, 27, 43],
        uncoupled_freqs_plot=[5, 15, 35],
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()