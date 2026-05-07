"""
Spike-Field Coupling Plotting Module

Submodules:
- heatmaps: Effect size and p-value heatmaps
- scatter_metrics: Magnitude scatter, phase recovery, metrics bars, ROC/PR
- beta_posterior: Beta posterior scatter plots
- coupling_stats: Statistical tests (Wald, Rayleigh)
- spectral_dynamics: Spectrogram comparison, correlation plots
- spectral_dynamics_trials: Trial-structured visualizations
"""

# Heatmaps
from .heatmaps import (
    set_publication_style,
    COLORS,
    plot_effect_heatmap,
    plot_pval_heatmap,
    plot_effect_comparison,
    plot_pval_comparison,
    plot_side_by_side_heatmaps,
)

# Scatter and metrics
from .scatter_metrics import (
    plot_magnitude_scatter,
    plot_phase_recovery,
    plot_metrics_bars,
    plot_roc_pr_curves,
)

# Beta posterior
from .beta_posterior import (
    plot_beta_posterior_scatter,
)

# Statistical tests
from .coupling_stats import (
    wald_test,
    wald_test_band_selection,
    phase_concentration_test,
    compute_detection_metrics,
    compute_roc_auc,
    circular_difference,
    compute_phase_mae,
    summarize_posterior,
)

# Spectral dynamics (single-trial)
from .spectral_dynamics import (
    METHOD_CONFIG,
    generate_spectral_dynamics_figures,
)

# Summary plots (high-level, user-facing)
from .summary import (
    plot_coupling_summary,
    plot_spectrogram,
    plot_trial_averaged_dynamics,
    plot_all_default,
)

# Spectral dynamics (trial-structured)
from .spectral_dynamics_trials import (
    set_style as set_style_trials,
    METHOD_CONFIG as TRIAL_METHOD_CONFIG,
    compute_optimal_scale,
    compute_global_scale,
    resample_to_target,
    extract_complex_from_separated,
    extract_complex_from_interleaved,
    extract_variance_from_interleaved,
    derotate_tfr,
    compute_trial_correlations,
    plot_trial_specific_comparison,
    plot_deviation_comparison,
    plot_trial_averaged_comparison,
    plot_correlation_boxplot,
    plot_psd_comparison,
    plot_spectrogram_trial_specific,
    plot_spectrogram_trial_averaged,
    plot_spectrogram_deviation,
    generate_trial_dynamics_figures,
)

__all__ = [
    # Style
    'set_publication_style',
    'COLORS',
    'METHOD_CONFIG',
    
    # Heatmaps
    'plot_effect_heatmap',
    'plot_pval_heatmap',
    'plot_effect_comparison',
    'plot_pval_comparison',
    'plot_side_by_side_heatmaps',
    
    # Scatter and metrics
    'plot_magnitude_scatter',
    'plot_phase_recovery',
    'plot_metrics_bars',
    'plot_roc_pr_curves',
    
    # Beta posterior
    'plot_beta_posterior_scatter',
    
    # Statistical tests
    'wald_test',
    'wald_test_band_selection',
    'phase_concentration_test',
    'compute_detection_metrics',
    'compute_roc_auc',
    'circular_difference',
    'compute_phase_mae',
    'summarize_posterior',
    
    # Summary (user-facing)
    'plot_coupling_summary',
    'plot_spectrogram',
    'plot_trial_averaged_dynamics',
    'plot_all_default',

    # Spectral dynamics
    'generate_spectral_dynamics_figures',
    'generate_trial_dynamics_figures',
    
    # Trial dynamics helpers
    'compute_optimal_scale',
    'compute_global_scale',
    'resample_to_target',
    'extract_complex_from_separated',
    'extract_complex_from_interleaved',
    'extract_variance_from_interleaved',
    'derotate_tfr',
    'compute_trial_correlations',
    'plot_trial_specific_comparison',
    'plot_deviation_comparison',
    'plot_trial_averaged_comparison',
    'plot_correlation_boxplot',
    'plot_psd_comparison',
    'plot_spectrogram_trial_specific',
    'plot_spectrogram_trial_averaged',
    'plot_spectrogram_deviation',
]