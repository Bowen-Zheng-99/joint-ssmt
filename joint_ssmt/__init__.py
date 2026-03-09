# joint_ssmt/__init__.py
"""
Joint SSMT: Bayesian state-space model for joint inference of
oscillatory dynamics and point-process coupling.
"""

__version__ = "0.1.0"

# JAX backend configuration (must run before any jax imports)
# The Apple-Silicon Metal backend does not support complex numbers or float64.
# On macOS + ARM, we force the CPU backend. On Linux/Windows this is a no-op.
import platform as _platform
if _platform.system() == "Darwin" and _platform.machine() == "arm64":
    import os as _os
    # Only override if the user hasn't explicitly chosen a platform already
    if "JAX_PLATFORMS" not in _os.environ:
        _os.environ["JAX_PLATFORMS"] = "cpu"
    del _os
del _platform

import logging
logging.getLogger("joint_ssmt").addHandler(logging.NullHandler())

# Configuration (user-facing)
from joint_ssmt.config import SpectralConfig, InferenceConfig, OutputConfig

# High-level I/O runners (user-facing)
from joint_ssmt.io import (
    run_inference,
    run_inference_trials,
    load_results,
    load_results_trials,
)

# Summary plotting (user-facing)
from joint_ssmt.plotting.summary import (
    plot_coupling_summary,
    plot_spectrogram,
    plot_trial_averaged_dynamics,
    plot_all_default,
)

# Simulation
from joint_ssmt.simulate_single_trial import (
    SingleTrialSimConfig,
    simulate_single_trial,
    build_history_design_single,
)

# Core data structures (for advanced users)
from joint_ssmt.params import OUParams
from joint_ssmt.utils_joint import Trace
from joint_ssmt.state_index import StateIndex

# Low-level inference (for advanced users)
from joint_ssmt.run_joint_inference_single_trial import (
    run_joint_inference_single_trial,
    SingleTrialInferenceConfig,
)
from joint_ssmt.run_joint_inference_trials import (
    run_joint_inference_trials_hier,
    InferenceTrialsHierConfig,
)

# EM fitting (for advanced users)
from joint_ssmt.em_ct_single_jax import em_ct_single_jax, EMSingleResult
from joint_ssmt.em_ct_hier_jax import em_ct_hier_jax, EMHierResult
from joint_ssmt.upsample_ct_single_fine import upsample_ct_single_fine, UpsampleSingleResult

# Utilities
from joint_ssmt.utils_multitaper import derotate_tfr_align_start
from joint_ssmt.utils_common import (
    centres_from_win,
    map_blocks_to_fine,
    build_t2k,
    normalize_Y_to_RJMK,
    separated_to_interleaved,
    interleaved_to_separated,
)
from joint_ssmt.ou import _phi_q, kalman_filter_ou, kalman_filter_ou_numba

__all__ = [
    # Version
    "__version__",
    # Config (user-facing)
    "SpectralConfig",
    "InferenceConfig",
    "OutputConfig",
    # High-level I/O (user-facing)
    "run_inference",
    "run_inference_trials",
    "load_results",
    "load_results_trials",
    # Summary plotting (user-facing)
    "plot_coupling_summary",
    "plot_spectrogram",
    "plot_trial_averaged_dynamics",
    "plot_all_default",
    # Simulation
    "SingleTrialSimConfig",
    "simulate_single_trial",
    "build_history_design_single",
    # Core
    "OUParams",
    "Trace",
    "StateIndex",
    # Low-level inference
    "run_joint_inference_single_trial",
    "SingleTrialInferenceConfig",
    "run_joint_inference_trials_hier",
    "InferenceTrialsHierConfig",
    # EM
    "em_ct_single_jax",
    "EMSingleResult",
    "em_ct_hier_jax",
    "EMHierResult",
    "upsample_ct_single_fine",
    "UpsampleSingleResult",
    # Utilities
    "derotate_tfr_align_start",
    "centres_from_win",
    "map_blocks_to_fine",
    "build_t2k",
    "normalize_Y_to_RJMK",
    "separated_to_interleaved",
    "interleaved_to_separated",
]
