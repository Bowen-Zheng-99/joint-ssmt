# joint_ssmt/io/__init__.py
"""
Spike-Field Coupling I/O

Single-trial inference:
    from joint_ssmt.io import run_inference
    
    saved_files = run_inference(
        lfp, spikes,
        ctssmt_config={'freq_min': 1, 'freq_max': 61},
        mcmc_config={'fixed_iter': 1000},
        output_config={'output_dir': './results', 'save_spectral': True},
    )

Trial-structured inference:
    from joint_ssmt.io import run_inference_trials
    
    saved_files = run_inference_trials(
        lfp, spikes,  # (R, T) and (R, S, T_fine)
        ctssmt_config={'freq_min': 1, 'freq_max': 61, 'window_sec': 0.4},
        mcmc_config={'fixed_iter': 500, 'n_refreshes': 10},
        output_config={'output_dir': './results', 'downsample_factor': 10},
    )

Loading results:
    from joint_ssmt.io import load_results, load_results_trials
    
    # Single-trial
    results = load_results('./results')
    
    # Trial-structured
    results = load_results_trials('./results')
"""

# Single-trial inference
from .runner import (
    run_inference,
    load_coupling,
    load_spectral,
    load_metadata,
    load_results,
    results_to_legacy_dict,
    DEFAULT_CTSSMT_CONFIG,
    DEFAULT_MCMC_CONFIG,
    DEFAULT_OUTPUT_CONFIG,
)

# Trial-structured inference
from .runner_trials import (
    run_inference_trials,
    load_coupling_trials,
    load_spectral_trials,
    load_lfp_only_trials,
    load_metadata_trials,
    load_results_trials,
    results_to_legacy_dict_trials,
    DEFAULT_CTSSMT_CONFIG_TRIALS,
    DEFAULT_MCMC_CONFIG_TRIALS,
    DEFAULT_OUTPUT_CONFIG_TRIALS,
)

__all__ = [
    # Single-trial
    'run_inference',
    'load_coupling',
    'load_spectral',
    'load_metadata',
    'load_results',
    'results_to_legacy_dict',
    'DEFAULT_CTSSMT_CONFIG',
    'DEFAULT_MCMC_CONFIG',
    'DEFAULT_OUTPUT_CONFIG',
    
    # Trial-structured
    'run_inference_trials',
    'load_coupling_trials',
    'load_spectral_trials',
    'load_lfp_only_trials',
    'load_metadata_trials',
    'load_results_trials',
    'results_to_legacy_dict_trials',
    'DEFAULT_CTSSMT_CONFIG_TRIALS',
    'DEFAULT_MCMC_CONFIG_TRIALS',
    'DEFAULT_OUTPUT_CONFIG_TRIALS',
]