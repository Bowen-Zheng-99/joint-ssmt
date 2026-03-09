"""
Scatter Plots and Metrics Visualization

- Magnitude correlation scatter plots
- Phase recovery scatter plots  
- Detection metrics bar charts
- ROC and PR curves
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, Sequence

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# Default colors
COLORS = {
    'plv': '#2E86AB',
    'sfc': '#A23B72',
    'joint': '#F18F01',
    'phase': '#28A745',
    'wald': '#F18F01',  # Same as joint
}


def get_color(method_name: str) -> str:
    """Get color for method, handling compound names like 'Joint (Wald)'."""
    name_lower = method_name.lower()
    # Direct match
    if name_lower in COLORS:
        return COLORS[name_lower]
    # Check if any key is contained in the name
    for key, color in COLORS.items():
        if key in name_lower:
            return color
    return '#333333'


def set_publication_style(font_size: int = 9):
    """Set publication-quality figure style."""
    if HAS_SEABORN:
        sns.set(style="ticks", context="paper", font="sans-serif",
                rc={"font.size": font_size,
                    "axes.titlesize": font_size,
                    "axes.labelsize": font_size})
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42


# =============================================================================
# Magnitude Correlation
# =============================================================================

def plot_magnitude_scatter(
    gt_mag: np.ndarray,
    est_dict: Dict[str, np.ndarray],
    coupled_mask: np.ndarray,
    output_path: str,
    *,
    figsize: Tuple[float, float] = (12, 4),
    suptitle: str = 'Magnitude Correlation',
) -> Tuple[str, Dict[str, float]]:
    """
    Plot scatter of ground truth vs estimated magnitudes.
    
    Parameters
    ----------
    gt_mag : (S, J_true) array
        Ground truth |β|
    est_dict : dict
        {method_name: (S, J_true) array} estimated magnitudes
    coupled_mask : (S, J_true) bool array
        True for coupled pairs
    output_path : str
    
    Returns
    -------
    output_path : str
    correlations : dict
        {method: r} correlation coefficients for coupled pairs
    """
    set_publication_style()
    
    n_methods = len(est_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    if n_methods == 1:
        axes = [axes]
    
    gt_flat = gt_mag.flatten()
    coupled_flat = coupled_mask.flatten()
    
    correlations = {}
    
    for ax, (name, est) in zip(axes, est_dict.items()):
        est_flat = est.flatten()
        color = get_color(name)
        
        # Plot uncoupled (background)
        ax.scatter(gt_flat[~coupled_flat], est_flat[~coupled_flat],
                   c=color, alpha=0.3, s=30, label='Uncoupled')
        
        # Plot coupled (foreground)
        ax.scatter(gt_flat[coupled_flat], est_flat[coupled_flat],
                   c=color, alpha=0.8, s=50, label='Coupled')
        
        # Correlation for coupled
        if coupled_flat.sum() > 2:
            r = np.corrcoef(gt_flat[coupled_flat], est_flat[coupled_flat])[0, 1]
            correlations[name] = r
            ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes,
                    fontsize=11, va='top', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Ground Truth |β|')
        ax.set_ylabel('Estimated')
        ax.set_title(name, fontweight='bold', color=color)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(suptitle, fontsize=10, y=1.02)
    if HAS_SEABORN:
        sns.despine()
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path, correlations


# =============================================================================
# Phase Recovery
# =============================================================================

def circular_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular difference."""
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def plot_phase_recovery(
    gt_phase: np.ndarray,
    est_dict: Dict[str, np.ndarray],
    coupled_mask: np.ndarray,
    output_path: str,
    *,
    figsize: Tuple[float, float] = (12, 4),
) -> Tuple[str, Dict[str, float]]:
    """
    Plot phase recovery scatter and error histogram.
    
    Parameters
    ----------
    gt_phase : (S, J_true) array
        Ground truth phases (radians)
    est_dict : dict
        {method: (S, J_true) array} estimated phases
    coupled_mask : (S, J_true) bool array
    output_path : str
    
    Returns
    -------
    output_path : str
    mae_dict : dict
        {method: MAE in degrees}
    """
    set_publication_style()
    
    true_ph = gt_phase.flatten()[coupled_mask.flatten()]
    
    n_methods = len(est_dict)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=figsize)
    
    mae_dict = {}
    errors = {}
    
    for i, (name, est) in enumerate(est_dict.items()):
        ax = axes[i]
        color = get_color(name)
        
        est_ph = est.flatten()[coupled_mask.flatten()]
        err = circular_diff(est_ph, true_ph)
        mae = np.abs(err).mean()
        mae_dict[name] = np.degrees(mae)
        errors[name] = err
        
        ax.scatter(np.degrees(true_ph), np.degrees(est_ph),
                   c=color, s=40, alpha=0.7, edgecolors='white', lw=0.5)
        ax.plot([-180, 180], [-180, 180], 'k--', lw=0.8)
        ax.set_xlabel('True Phase (°)')
        ax.set_ylabel('Estimated Phase (°)')
        ax.set_title(f'{name} (MAE={np.degrees(mae):.1f}°)')
        ax.set_xlim([-180, 180])
        ax.set_ylim([-180, 180])
        ax.set_aspect('equal')
    
    # Error histogram
    ax = axes[-1]
    bins = np.linspace(-180, 180, 25)
    for name, err in errors.items():
        color = get_color(name)
        ax.hist(np.degrees(err), bins, alpha=0.5, color=color,
                label=f'{name} ({mae_dict[name]:.1f}°)', edgecolor='white', linewidth=0.5)
    ax.axvline(0, color='k', ls='--', lw=0.8)
    ax.set_xlabel('Phase Error (°)')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.legend(fontsize=7, loc='upper right')
    
    if HAS_SEABORN:
        sns.despine()
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path, mae_dict


# =============================================================================
# Detection Metrics Bar Chart
# =============================================================================

def plot_metrics_bars(
    metrics_dict: Dict[str, dict],
    output_path: str,
    *,
    alpha: float = 0.05,
    figsize: Tuple[float, float] = (10, 3),
) -> str:
    """
    Plot detection metrics as bar charts.
    
    Parameters
    ----------
    metrics_dict : dict
        {method: {'sensitivity': x, 'specificity': y, ...}}
    output_path : str
    alpha : float
        For title
    
    Returns
    -------
    output_path : str
    """
    set_publication_style()
    
    metric_names = ['sensitivity', 'specificity', 'precision', 'f1']
    metric_labels = ['Sensitivity', 'Specificity', 'Precision', 'F1']
    
    methods = list(metrics_dict.keys())
    n_methods = len(methods)
    
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    for ax, key, label in zip(axes, metric_names, metric_labels):
        vals = [metrics_dict[m].get(key, 0) for m in methods]
        colors = [get_color(m) for m in methods]
        
        bars = ax.bar(methods, vals, color=colors, width=0.6)
        ax.set_ylim([0, 1.1])
        ax.set_title(label)
        
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.03, f'{v:.2f}',
                    ha='center', fontsize=8)
    
    plt.suptitle(f'Detection Metrics (α = {alpha})', y=1.02, fontsize=10)
    if HAS_SEABORN:
        sns.despine()
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


# =============================================================================
# ROC and PR Curves
# =============================================================================

def plot_roc_pr_curves(
    y_true: np.ndarray,
    scores_dict: Dict[str, np.ndarray],
    output_path: str,
    *,
    figsize: Tuple[float, float] = (10, 4),
) -> Tuple[str, Dict[str, float]]:
    """
    Plot ROC and PR curves.
    
    Parameters
    ----------
    y_true : (S, J) bool array
        Ground truth
    scores_dict : dict
        {method: (S, J) array} scores (higher = more likely positive)
    output_path : str
    
    Returns
    -------
    output_path : str
    auc_dict : dict
        {method: AUC}
    """
    if not HAS_SKLEARN:
        print("Warning: sklearn not available, skipping ROC/PR curves")
        return output_path, {}
    
    set_publication_style()
    
    y = y_true.flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    auc_dict = {}
    
    # ROC
    ax = axes[0]
    for name, scores in scores_dict.items():
        color = get_color(name)
        score = scores.flatten()
        
        fpr, tpr, _ = roc_curve(y, score)
        a = auc(fpr, tpr)
        auc_dict[name] = a
        
        ax.plot(fpr, tpr, color=color, lw=1.5, label=f'{name} (AUC={a:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_aspect('equal')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # PR
    ax = axes[1]
    for name, scores in scores_dict.items():
        color = get_color(name)
        score = scores.flatten()
        
        prec, rec, _ = precision_recall_curve(y, score)
        ap = average_precision_score(y, score)
        
        ax.plot(rec, prec, color=color, lw=1.5, label=f'{name} (AP={ap:.3f})')
    
    ax.axhline(y.mean(), color='gray', ls='--', lw=0.8, label=f'Baseline ({y.mean():.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    if HAS_SEABORN:
        sns.despine()
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path, auc_dict


# =============================================================================
# Beta Posterior Scatter
# =============================================================================

def plot_beta_posterior_scatter(
    beta_samples: np.ndarray,
    J: int,
    freqs: np.ndarray,
    output_path: str,
    *,
    gt_phase: Optional[np.ndarray] = None,
    gt_mag: Optional[np.ndarray] = None,
    burn_in_frac: float = 0.5,
    figsize_per_band: Tuple[float, float] = (2.5, 2.5),
    max_bands: int = 12,
) -> str:
    """
    Plot posterior scatter in (Re, Im) plane for each frequency band.
    
    Parameters
    ----------
    beta_samples : (n_samples, S, P) array
    J : int
        Number of frequency bands
    freqs : (J,) array
    output_path : str
    gt_phase, gt_mag : optional (S, J) arrays
        Ground truth for overlay
    burn_in_frac : float
    figsize_per_band : tuple
    max_bands : int
        Maximum bands to plot
    
    Returns
    -------
    output_path : str
    """
    set_publication_style()
    
    n_samples, S, P = beta_samples.shape
    burn_in = int(burn_in_frac * n_samples)
    samples = beta_samples[burn_in:]
    
    bR = samples[:, :, 1:1+J]
    bI = samples[:, :, 1+J:1+2*J]
    
    J_plot = min(J, max_bands)
    n_cols = min(4, J_plot)
    n_rows = (J_plot + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * figsize_per_band[0], n_rows * figsize_per_band[1]))
    axes = np.atleast_2d(axes).flatten()
    
    for j in range(J_plot):
        ax = axes[j]
        
        for s in range(S):
            ax.scatter(bR[:, s, j], bI[:, s, j], alpha=0.1, s=5, c=f'C{s}')
        
        # Ground truth ray
        if gt_phase is not None and gt_mag is not None:
            for s in range(S):
                if gt_mag[s, j] > 0:
                    r = gt_mag[s, j]
                    theta = gt_phase[s, j]
                    ax.plot([0, r * np.cos(theta)], [0, r * np.sin(theta)],
                            'k-', lw=2, alpha=0.7)
        
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='gray', lw=0.5)
        ax.set_aspect('equal')
        ax.set_title(f'{freqs[j]:.0f} Hz', fontsize=9)
        ax.set_xlabel('Re(β)', fontsize=8)
        ax.set_ylabel('Im(β)', fontsize=8)
    
    # Hide unused axes
    for j in range(J_plot, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path