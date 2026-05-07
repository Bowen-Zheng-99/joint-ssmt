"""
Statistical Analysis Module

Re-exports from coupling_stats.py
"""

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

__all__ = [
    'wald_test',
    'wald_test_band_selection',
    'phase_concentration_test',
    'compute_detection_metrics',
    'compute_roc_auc',
    'circular_difference',
    'compute_phase_mae',
    'summarize_posterior',
]