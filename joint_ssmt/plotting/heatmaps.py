"""
Coupling Detection Heatmap Plots

Visualize spike-field coupling effect sizes and p-values as heatmaps.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from typing import Optional, Sequence, Tuple

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# =============================================================================
# Style
# =============================================================================

def set_publication_style(font_size: int = 9):
    """Set publication-quality figure style."""
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
                    "legend.frameon": False,
                    "xtick.major.size": 2.5,
                    "ytick.major.size": 2.5,
                    "xtick.major.width": 0.5,
                    "ytick.major.width": 0.5,
                    "savefig.transparent": True,
                })
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42


# Default colors
COLORS = {
    'plv': '#2E86AB',       # Teal blue
    'sfc': '#A23B72',       # Magenta/pink
    'joint': '#F18F01',     # Orange
    'phase': '#28A745',     # Green
}


# =============================================================================
# Single Heatmap Row
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
    cmap: str = 'Reds',
    vmax_percentile: float = 99,
    show_stars: bool = True,
) -> plt.cm.ScalarMappable:
    """
    Plot effect size heatmap with optional ground truth markers.
    
    Parameters
    ----------
    ax : matplotlib Axes
    values : (S, J) array
        Effect sizes
    freqs : (J,) array
        Frequency grid
    title : str
    true_freqs : (J_true,) array, optional
        Ground truth coupled frequencies
    masks : (S, J_true) bool array, optional
        Ground truth coupling mask
    log_scale : bool
        Plot log10(values + 1)
    cmap : str
    vmax_percentile : float
    show_stars : bool
        Show ★ markers at true couplings
    
    Returns
    -------
    im : ScalarMappable for colorbar
    """
    if log_scale:
        plot_values = np.log10(values + 1)
        vmax = np.percentile(plot_values[np.isfinite(plot_values)], vmax_percentile)
    else:
        plot_values = values
        vmax = np.percentile(plot_values[np.isfinite(plot_values)], vmax_percentile)
    
    freq_step = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    extent = [freqs[0] - freq_step/2, freqs[-1] + freq_step/2,
              values.shape[0] - 0.5, -0.5]
    
    im = ax.imshow(plot_values, aspect='auto', cmap=cmap,
                   vmin=0, vmax=vmax, extent=extent)
    
    ax.set_ylabel('Unit')
    ax.set_title(title, fontweight='bold')
    
    # Mark true frequencies
    if true_freqs is not None:
        for f in true_freqs:
            ax.axvline(f, color='cyan', linestyle='--', alpha=0.5, lw=0.8)
    
    # Star markers
    if show_stars and masks is not None and true_freqs is not None:
        for s in range(masks.shape[0]):
            for j, f in enumerate(true_freqs):
                if j < masks.shape[1] and masks[s, j]:
                    ax.text(f, s, '★', ha='center', va='center',
                            fontsize=10, color='white', fontweight='bold',
                            path_effects=[patheffects.withStroke(linewidth=1.5, foreground='black')])
    
    return im


def plot_pval_heatmap(
    ax: plt.Axes,
    pvals: np.ndarray,
    freqs: np.ndarray,
    title: str,
    *,
    true_freqs: Optional[np.ndarray] = None,
    masks: Optional[np.ndarray] = None,
    cmap: str = 'hot_r',
    vmin: float = 0,
    vmax: float = 10,
    alpha: float = 0.05,
    show_stars: bool = True,
) -> plt.cm.ScalarMappable:
    """
    Plot -log10(p-value) heatmap.
    
    Parameters
    ----------
    ax : matplotlib Axes
    pvals : (S, J) array
    freqs : (J,) array
    title : str
    true_freqs, masks : optional ground truth
    cmap : str
    vmin, vmax : float
        Color scale range for -log10(p)
    alpha : float
        Significance threshold (for colorbar annotation)
    show_stars : bool
    
    Returns
    -------
    im : ScalarMappable for colorbar
    """
    log_p = -np.log10(np.clip(pvals, 1e-20, 1))
    
    freq_step = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    extent = [freqs[0] - freq_step/2, freqs[-1] + freq_step/2,
              pvals.shape[0] - 0.5, -0.5]
    
    im = ax.imshow(log_p, aspect='auto', cmap=cmap,
                   vmin=vmin, vmax=vmax, extent=extent)
    
    ax.set_ylabel('Unit')
    ax.set_title(title, fontweight='bold')
    
    # Mark true frequencies
    if true_freqs is not None:
        for f in true_freqs:
            ax.axvline(f, color='cyan', linestyle='--', alpha=0.5, lw=0.8)
    
    # Star markers
    if show_stars and masks is not None and true_freqs is not None:
        for s in range(masks.shape[0]):
            for j, f in enumerate(true_freqs):
                if j < masks.shape[1] and masks[s, j]:
                    ax.text(f, s, '★', ha='center', va='center',
                            fontsize=10, color='white', fontweight='bold',
                            path_effects=[patheffects.withStroke(linewidth=1.5, foreground='black')])
    
    return im


# =============================================================================
# Multi-Panel Heatmap Figures
# =============================================================================

def plot_effect_comparison(
    effect_dict: dict,
    freqs: np.ndarray,
    output_path: str,
    *,
    true_freqs: Optional[np.ndarray] = None,
    masks: Optional[np.ndarray] = None,
    log_scale_keys: Sequence[str] = ('wald_W',),
    figsize: Tuple[float, float] = (10, 9),
    suptitle: str = 'Effect Size Comparison',
) -> str:
    """
    Plot multiple effect size heatmaps in a column.
    
    Parameters
    ----------
    effect_dict : dict
        {name: (S, J) array} of effect sizes
    freqs : (J,) array
    output_path : str
    true_freqs, masks : optional ground truth
    log_scale_keys : sequence of str
        Keys to plot in log scale
    figsize : tuple
    suptitle : str
    
    Returns
    -------
    output_path : str
    """
    set_publication_style()
    
    n_panels = len(effect_dict)
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    if n_panels == 1:
        axes = [axes]
    
    for ax, (name, values) in zip(axes, effect_dict.items()):
        use_log = name in log_scale_keys
        im = plot_effect_heatmap(
            ax, values, freqs, name,
            true_freqs=true_freqs, masks=masks,
            log_scale=use_log,
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        if use_log:
            cbar.set_label('log₁₀(value + 1)')
    
    axes[-1].set_xlabel('Frequency (Hz)')

    gt_suffix = ' (★ = true coupling)' if (true_freqs is not None or masks is not None) else ''
    plt.suptitle(f'{suptitle}{gt_suffix}', fontsize=10, y=0.995)
    if HAS_SEABORN:
        sns.despine()
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    return plt.gcf()


def plot_pval_comparison(
    pval_dict: dict,
    freqs: np.ndarray,
    output_path: str,
    *,
    true_freqs: Optional[np.ndarray] = None,
    masks: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    figsize: Tuple[float, float] = (10, 6.5),
    suptitle: str = 'P-value Comparison',
) -> str:
    """
    Plot multiple p-value heatmaps in a column.
    
    Parameters
    ----------
    pval_dict : dict
        {name: (S, J) array} of p-values
    freqs : (J,) array
    output_path : str
    true_freqs, masks : optional ground truth
    alpha : float
        Significance threshold
    figsize : tuple
    suptitle : str
    
    Returns
    -------
    output_path : str
    """
    set_publication_style()
    
    n_panels = len(pval_dict)
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    if n_panels == 1:
        axes = [axes]
    
    for ax, (name, pvals) in zip(axes, pval_dict.items()):
        log_p = -np.log10(np.clip(pvals, 1e-10, 1))
        vmax = max(np.percentile(log_p[np.isfinite(log_p)], 99), -np.log10(alpha) + 1)
        
        im = plot_pval_heatmap(
            ax, pvals, freqs, name,
            true_freqs=true_freqs, masks=masks,
            vmax=vmax, alpha=alpha,
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cbar.set_label('-log₁₀(p)')
        
        # Mark significance threshold
        sig_line = -np.log10(alpha)
        if sig_line <= vmax:
            cbar.ax.axhline(sig_line, color='cyan', lw=1.5)
            cbar.ax.text(1.5, sig_line, f'α={alpha}', color='cyan', fontsize=7, va='center')
    
    axes[-1].set_xlabel('Frequency (Hz)')

    gt_suffix = ' (★ = true coupling)' if (true_freqs is not None or masks is not None) else ''
    plt.suptitle(f'{suptitle}{gt_suffix}', fontsize=10, y=0.995)
    if HAS_SEABORN:
        sns.despine()
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    return plt.gcf()


# =============================================================================
# Side-by-Side Comparison
# =============================================================================

def plot_side_by_side_heatmaps(
    left_values: np.ndarray,
    right_values: np.ndarray,
    freqs: np.ndarray,
    output_path: str,
    *,
    left_title: str = 'Original',
    right_title: str = 'Standardized',
    true_freqs: Optional[np.ndarray] = None,
    masks: Optional[np.ndarray] = None,
    log_scale: bool = False,
    figsize: Tuple[float, float] = (12, 4),
    suptitle: str = 'Comparison',
) -> str:
    """
    Plot two heatmaps side by side.
    
    Returns output_path.
    """
    set_publication_style()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    im1 = plot_effect_heatmap(
        axes[0], left_values, freqs, left_title,
        true_freqs=true_freqs, masks=masks, log_scale=log_scale,
    )
    fig.colorbar(im1, ax=axes[0], fraction=0.02, pad=0.01)
    
    im2 = plot_effect_heatmap(
        axes[1], right_values, freqs, right_title,
        true_freqs=true_freqs, masks=masks, log_scale=log_scale,
    )
    fig.colorbar(im2, ax=axes[1], fraction=0.02, pad=0.01)
    
    axes[0].set_xlabel('Frequency (Hz)')
    axes[1].set_xlabel('Frequency (Hz)')

    gt_suffix = ' (★ = true coupling)' if (true_freqs is not None or masks is not None) else ''
    plt.suptitle(f'{suptitle}{gt_suffix}', fontsize=10, y=1.02)
    if HAS_SEABORN:
        sns.despine()
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    return plt.gcf()