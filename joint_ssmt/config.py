"""
Configuration system for Joint SSMT.

Supports loading from YAML files, dicts, or dataclass defaults.
"""

import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


# Spectral config

@dataclass
class SpectralConfig:
    """Multitaper spectrogram parameters.

    Attributes
    ----------
    freq_min : float
        Lower bound of frequency grid (Hz).
    freq_max : float
        Upper bound of frequency grid (Hz).
    freq_step : float
        Grid spacing (Hz).
    window_sec : float
        Multitaper window length (seconds).
    time_bandwidth : float
        NW product (controls resolution vs. variance).
    em_max_iter : int
        Maximum EM iterations for warm-starting OU parameters.
    em_tol : float
        EM convergence tolerance.
    """

    freq_min: float = 1.0
    freq_max: float = 61.0
    freq_step: float = 1.0
    window_sec: float = 2.0
    time_bandwidth: float = 2.0
    em_max_iter: int = 2000
    em_tol: float = 1e-6

    # Trial-specific EM extras (only used by trial runner)
    sig_eps_init: float = 5.0
    log_every: int = 1000

    def to_ctssmt_dict(self) -> Dict[str, Any]:
        """Convert to the legacy ctssmt_config dict format."""
        return {
            "freq_min": self.freq_min,
            "freq_max": self.freq_max,
            "freq_step": self.freq_step,
            "window_sec": self.window_sec,
            "NW": self.time_bandwidth,
            "em_max_iter": self.em_max_iter,
            "em_tol": self.em_tol,
            "sig_eps_init": self.sig_eps_init,
            "log_every": self.log_every,
        }


# Inference config

@dataclass
class InferenceConfig:
    """MCMC / Gibbs sampler parameters.

    Attributes
    ----------
    warmup_iterations : int
        Gibbs iterations during warmup phase.
    n_refresh_cycles : int
        Number of KF/RTS latent refresh cycles.
    iterations_per_refresh : int
        Gibbs iterations per refresh cycle.
    trace_thinning : int
        Keep every Nth sample in the trace.
    burn_in_fraction : float
        Fraction of trace to discard as burn-in.
    n_history_lags : int
        Number of spike-history lags.
    """

    # MCMC structure
    warmup_iterations: int = 1000
    n_refresh_cycles: int = 5
    iterations_per_refresh: int = 200
    trace_thinning: int = 2
    burn_in_fraction: float = 0.6

    # Spike history
    n_history_lags: int = 20

    # Priors
    omega_floor: float = 1e-3
    tau2_intercept: float = 25.0
    tau2_gamma: float = 625.0
    a0_ard: float = 0.5
    b0_ard: float = 0.5

    # Trial-specific prior (flat Normal on beta)
    sigma_u: float = 1.0

    # Detection
    use_wald_band_selection: bool = True
    wald_alpha: float = 0.1
    use_shrinkage: bool = True

    # Latent inference
    standardize_latents: bool = True
    freeze_intercept_during_refresh: bool = True
    enable_latent_refresh: bool = True

    # Winsorization (trial-structured only)
    use_winsorization: bool = True
    winsorize_percentile: float = 95.0
    winsorize_after_warmup: bool = True
    winsorize_after_refresh: bool = True

    def to_mcmc_dict(self) -> Dict[str, Any]:
        """Convert to the legacy mcmc_config dict format."""
        return {
            "fixed_iter": self.warmup_iterations,
            "n_refreshes": self.n_refresh_cycles,
            "inner_steps": self.iterations_per_refresh,
            "trace_thin": self.trace_thinning,
            "burn_in_frac": self.burn_in_fraction,
            "n_history_lags": self.n_history_lags,
            "omega_floor": self.omega_floor,
            "tau2_intercept": self.tau2_intercept,
            "tau2_gamma": self.tau2_gamma,
            "a0_ard": self.a0_ard,
            "b0_ard": self.b0_ard,
            "sigma_u": self.sigma_u,
            "use_wald_selection": self.use_wald_band_selection,
            "wald_alpha": self.wald_alpha,
            "use_shrinkage": self.use_shrinkage,
            "standardize_latents": self.standardize_latents,
            "freeze_beta0": self.freeze_intercept_during_refresh,
            "enable_latent_refresh": self.enable_latent_refresh,
            "use_winsorization": self.use_winsorization,
            "winsorize_percentile": self.winsorize_percentile,
            "winsorize_after_warmup": self.winsorize_after_warmup,
            "winsorize_after_refresh": self.winsorize_after_refresh,
        }


# Output config

@dataclass
class OutputConfig:
    """Output and plotting options."""

    output_dir: str = "./results"
    save_spectral: bool = True
    save_fine: bool = False
    downsample_factor: int = 10
    plot: bool = True
    plot_format: str = "pdf"
    plot_dpi: int = 300

    def to_output_dict(self) -> Dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "save_spectral": self.save_spectral,
            "save_fine": self.save_fine,
            "downsample_factor": self.downsample_factor,
        }


# YAML loading

def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    import yaml

    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


_INFERENCE_ALIASES = {
    # Legacy / shorthand → canonical dataclass field name
    "fixed_iter": "warmup_iterations",
    "n_refreshes": "n_refresh_cycles",
    "inner_steps": "iterations_per_refresh",
    "trace_thin": "trace_thinning",
    "burn_in_frac": "burn_in_fraction",
    "use_wald_selection": "use_wald_band_selection",
    "freeze_beta0": "freeze_intercept_during_refresh",
}

_SPECTRAL_ALIASES = {
    "NW": "time_bandwidth",
}

_ALIAS_MAPS = {
    "InferenceConfig": _INFERENCE_ALIASES,
    "SpectralConfig": _SPECTRAL_ALIASES,
}


def _dict_to_dataclass(cls, d: Dict[str, Any]):
    """Create a dataclass instance from a dict, ignoring unknown keys.

    Recognises common aliases (e.g. ``inner_steps`` → ``iterations_per_refresh``)
    so that both the human-readable and legacy key names work.
    """
    aliases = _ALIAS_MAPS.get(cls.__name__, {})
    valid = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {}
    unknown = []
    for k, v in d.items():
        canon = aliases.get(k, k)
        if canon in valid:
            filtered[canon] = v
        else:
            unknown.append(k)
    if unknown:
        logger.warning(f"Unknown keys for {cls.__name__}: {unknown}")
    return cls(**filtered)


def load_config(
    source: Union[str, Dict[str, Any], None],
    config_class: type,
) -> Any:
    """Load a config from a YAML path, dict, dataclass, or None (defaults).

    Parameters
    ----------
    source : str, dict, dataclass, or None
        - str: path to a YAML file
        - dict: config keys
        - dataclass instance: returned as-is
        - None: return defaults
    config_class : type
        The dataclass type (SpectralConfig, InferenceConfig, OutputConfig).
    """
    if source is None:
        return config_class()
    if isinstance(source, config_class):
        return source
    if isinstance(source, str):
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Config file not found: {source}")
        source = load_yaml(source)
    if isinstance(source, dict):
        return _dict_to_dataclass(config_class, source)
    raise TypeError(f"Expected str, dict, {config_class.__name__}, or None; got {type(source)}")
