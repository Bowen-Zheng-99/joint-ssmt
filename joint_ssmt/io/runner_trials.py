"""
Spike-Field Coupling Inference Runner for Trial-Structured Data

Main entry point: run_inference_trials()

Usage (new config-based API)::

    from joint_ssmt.io.runner_trials import run_inference_trials

    saved = run_inference_trials(
        lfp=lfp_trials,          # (R, T)
        spikes=spikes_trials,    # (R, S, T_fine)
        spectral_config="configs/spectral_trials.yaml",
        inference_config="configs/inference_trials.yaml",
        output_config={"output_dir": "./results_trials"},
        fs=1000.0,
        plot=True,
    )
"""

import logging
import os
import json
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass

from joint_ssmt.config import (
    SpectralConfig,
    InferenceConfig,
    OutputConfig,
    load_config,
)

logger = logging.getLogger(__name__)


# DEFAULT CONFIGS

DEFAULT_CTSSMT_CONFIG_TRIALS = {
    'freq_min': 1.0,
    'freq_max': 61.0,
    'freq_step': 2.0,       # [1, 3, 5, ..., 59] Hz
    'window_sec': 0.4,
    'NW': 1,                # NW_product = 1
    'em_max_iter': 5000,
    'em_tol': 1e-3,
    'sig_eps_init': 5.0,
    'log_every': 1000,
}

DEFAULT_MCMC_CONFIG_TRIALS = {
    'fixed_iter': 500,            # Gibbs iterations during warmup phase
    'n_refreshes': 5,             # Number of hierarchical KF/RTS refresh cycles
    'inner_steps': 200,           # Gibbs iterations per refresh cycle
    'trace_thin': 2,              # Keep every N-th sample in the trace
    'burn_in_frac': 0.5,          # Fraction of trace to discard as burn-in
    'n_history_lags': 20,         # Number of spike-history lags
    # Regularization
    'omega_floor': 1e-3,          # Floor for Polya-Gamma auxiliary variable (numerical stability)
    'sigma_u': 1.0,               # Prior std on beta (coupling coefficients) — flat-ish Normal
    # Priors
    'tau2_intercept': 100.0 ** 2, # Prior variance on beta_0 (baseline firing rate)
    'tau2_gamma': 25.0 ** 2,      # Prior variance on gamma (spike-history weights)
    'a0_ard': 1e-2,               # ARD half-Cauchy shape: small → aggressive sparsity
    'b0_ard': 1e-2,               # ARD half-Cauchy scale
    # Features
    'use_wald_selection': True,   # Enable Wald-test band selection after warmup
    'wald_alpha': 0.05,           # Significance level for Wald test (per-band)
    'use_shrinkage': False,       # Enable ARD shrinkage (maps to internal use_beta_shrinkage)
    'standardize_latents': True,  # Z-score latent amplitudes before coupling regression
    'freeze_beta0': False,        # Fix intercept during refresh cycles
    # Winsorization (cap extreme magnitudes per frequency band)
    'use_winsorization': True,    # Cap extreme beta magnitudes per frequency band
    'winsorize_percentile': 95.0, # Percentile threshold for magnitude capping
    'winsorize_after_warmup': True,  # Apply winsorization after warmup phase
    'winsorize_after_refresh': True, # Apply winsorization after each KF refresh
}

DEFAULT_OUTPUT_CONFIG_TRIALS = {
    'output_dir': './results',
    'save_spectral': True,
    'downsample_factor': 10,  # Downsample fine states for storage
}


# HELPER FUNCTIONS

def _merge_config(user_config: Optional[Dict], default_config: Dict) -> Dict:
    """Merge user config with defaults."""
    if user_config is None:
        return default_config.copy()
    merged = default_config.copy()
    merged.update(user_config)
    return merged


def _numpy_to_python(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_numpy_to_python(v) for v in obj]
    return obj


def _theta_to_dict(theta) -> dict:
    """Convert OUParams to dict."""
    return {
        'lam': np.asarray(theta.lam),
        'sig_v': np.asarray(theta.sig_v),
        'sig_eps': np.asarray(theta.sig_eps),
    }


def _downsample_array(arr: np.ndarray, factor: int, axis: int = -1) -> np.ndarray:
    """Downsample array along specified axis."""
    if factor <= 1:
        return arr
    slices = [slice(None)] * arr.ndim
    slices[axis] = slice(None, None, factor)
    return arr[tuple(slices)]

def _compute_multitaper_trials(
    lfp: np.ndarray,
    fs: float,
    freqs: np.ndarray,
    window_sec: float,
    NW: int,
) -> np.ndarray:
    """Compute complex multitaper spectrogram for trial data."""
    import mne
    from joint_ssmt.utils_multitaper import derotate_tfr_align_start
    
    R, T = lfp.shape
    M_samples = int(window_sec * fs)
    n_tapers = int(2 * NW - 1) if NW > 0.5 else 1
    
    tfr_raw = mne.time_frequency.tfr_array_multitaper(
        lfp[:, None, :],
        sfreq=fs,
        freqs=freqs,
        n_cycles=freqs * window_sec,
        time_bandwidth=2 * NW,
        output="complex",
        zero_mean=False,
    ).squeeze()  # After squeeze: (R, M, J, T) for multi-taper, (R, J, T) for single
    
    # Handle single vs multiple tapers
    # MNE output order: (n_epochs, n_channels, n_tapers, n_freqs, n_times)
    # After squeeze: (R, M, J, T) for multi-taper
    if tfr_raw.ndim == 3:
        # Single taper: (R, J, T) -> (R, J, 1, T)
        tfr_raw = tfr_raw[:, :, None, :]
    else:
        # Multiple tapers: (R, M, J, T) -> (R, J, M, T)
        tfr_raw = tfr_raw.transpose(0, 2, 1, 3)
    
    # Derotate - expects (R, J, M, T) or compatible shape
    tfr = derotate_tfr_align_start(tfr_raw, freqs, fs, 1, M_samples)
    
    # Scale by taper normalization
    tapers, _ = mne.time_frequency.multitaper.dpss_windows(M_samples, NW, Kmax=n_tapers)
    scaling = 2.0 / tapers[0].sum()
    tfr = tfr * scaling
    
    # Downsample to blocks
    Y_trials = tfr[:, :, :, ::M_samples]  # (R, J, M, K)
    
    return Y_trials


def _build_history_design_trials(spikes: np.ndarray, n_lags: int = 20) -> np.ndarray:
    """Build spike history design matrix for trial data.
    
    Input: spikes (R, S, T)
    Output: H_SRTL (S, R, T, L)
    """
    R, S, T = spikes.shape
    H = np.zeros((S, R, T, n_lags), dtype=np.float32)
    
    for s in range(S):
        for r in range(R):
            for lag in range(n_lags):
                if lag + 1 < T:
                    H[s, r, lag+1:, lag] = spikes[r, s, :T-lag-1]
    
    return H


def _extract_beta_stats(
    beta_trace: np.ndarray,
    J: int,
    burn_in_frac: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract beta statistics from trace.
    
    Returns: beta_mag, beta_phase, wald_W, wald_pval, beta_std
    """
    from scipy import stats as sp_stats
    
    n_samples, S, D = beta_trace.shape
    burn = int(burn_in_frac * n_samples)
    post = beta_trace[burn:]
    
    # Extract real/imag parts
    bR = post[:, :, 1:1+J]
    bI = post[:, :, 1+J:1+2*J]
    
    # Posterior mean magnitude and phase (matching notebook)
    mR, mI = bR.mean(axis=0), bI.mean(axis=0)
    beta_mag = np.sqrt(mR**2 + mI**2)
    beta_phase = np.arctan2(mI, mR)
    
    # Posterior std
    beta_std = beta_trace[burn:].std(axis=0)
    
    # Wald test
    wald_W = np.zeros((S, J))
    wald_pval = np.zeros((S, J))
    
    for s in range(S):
        for j in range(J):
            br, bi = bR[:, s, j], bI[:, s, j]
            mu = np.array([br.mean(), bi.mean()])
            Sig = np.cov(np.column_stack([br, bi]), rowvar=False) + 1e-10*np.eye(2)
            wald_W[s, j] = mu @ np.linalg.solve(Sig, mu)
            wald_pval[s, j] = 1 - sp_stats.chi2.cdf(wald_W[s, j], df=2)
    
    return beta_mag, beta_phase, wald_W, wald_pval, beta_std


def _extract_phase_stats(
    beta_trace: np.ndarray,
    J: int,
    burn_in_frac: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract phase concentration statistics."""
    n_samples, S, D = beta_trace.shape
    burn = int(burn_in_frac * n_samples)
    post = beta_trace[burn:]
    
    bR = post[:, :, 1:1+J]
    bI = post[:, :, 1+J:1+2*J]
    
    phase_R = np.zeros((S, J))
    phase_pval = np.zeros((S, J))
    phase_est = np.zeros((S, J))
    
    for s in range(S):
        for j in range(J):
            br, bi = bR[:, s, j], bI[:, s, j]
            phases = np.arctan2(bi, br)
            n = len(phases)
            C = np.mean(np.cos(phases))
            S_sin = np.mean(np.sin(phases))
            phase_R[s, j] = np.sqrt(C**2 + S_sin**2)
            phase_est[s, j] = np.arctan2(S_sin, C)
            Z = n * phase_R[s, j]**2
            phase_pval[s, j] = max(np.exp(-Z), 1e-300)
    
    return phase_R, phase_pval, phase_est


# MAIN FUNCTION

def run_inference_trials(
    lfp: np.ndarray,
    spikes: np.ndarray,
    spectral_config: Union[str, Dict, SpectralConfig, None] = None,
    inference_config: Union[str, Dict, InferenceConfig, None] = None,
    output_config: Union[str, Dict, OutputConfig, None] = None,
    fs: float = 1000.0,
    delta_spk: Optional[float] = None,
    ground_truth: Optional[Dict[str, Any]] = None,
    plot: bool = True,
    verbose: bool = True,
    # Legacy aliases
    ctssmt_config: Optional[Dict[str, Any]] = None,
    mcmc_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Run spike-field coupling inference on trial-structured data.

    Parameters
    ----------
    lfp : (R, T) array
        LFP signals per trial.
    spikes : (R, S, T_fine) array
        Spike trains per trial (binary).
    spectral_config : str, dict, SpectralConfig, or None
        Spectral / multitaper config.
    inference_config : str, dict, InferenceConfig, or None
        MCMC / Gibbs sampler config.
    output_config : str, dict, OutputConfig, or None
        Output and plotting config.
    fs : float
        LFP sampling rate (Hz).
    delta_spk : float, optional
        Spike time resolution (default: 1/fs).
    ground_truth : dict, optional
        Ground truth for metadata (simulation only).
    plot : bool
        Generate default summary plots.
    verbose : bool
        Print progress.
    ctssmt_config : dict, optional
        **Legacy alias** for ``spectral_config``.
    mcmc_config : dict, optional
        **Legacy alias** for ``inference_config``.

    Returns
    -------
    saved_files : dict
        ``{'coupling': path, 'spectral': path, 'metadata': path, ...}``
    """
    start_time = time.time()

    # --- Resolve configs (new system or legacy dicts) ---
    if ctssmt_config is not None:
        ct_cfg = _merge_config(ctssmt_config, DEFAULT_CTSSMT_CONFIG_TRIALS)
    else:
        spec = load_config(spectral_config, SpectralConfig)
        ct_cfg = _merge_config(spec.to_ctssmt_dict(), DEFAULT_CTSSMT_CONFIG_TRIALS)

    if mcmc_config is not None:
        mc_cfg = _merge_config(mcmc_config, DEFAULT_MCMC_CONFIG_TRIALS)
    else:
        inf = load_config(inference_config, InferenceConfig)
        mc_cfg = _merge_config(inf.to_mcmc_dict(), DEFAULT_MCMC_CONFIG_TRIALS)

    if isinstance(output_config, OutputConfig):
        out_cfg = _merge_config(output_config.to_output_dict(), DEFAULT_OUTPUT_CONFIG_TRIALS)
    elif isinstance(output_config, dict):
        out_cfg = _merge_config(output_config, DEFAULT_OUTPUT_CONFIG_TRIALS)
    else:
        out_obj = load_config(output_config, OutputConfig)
        out_cfg = _merge_config(out_obj.to_output_dict(), DEFAULT_OUTPUT_CONFIG_TRIALS)

    if plot:
        out_cfg['save_spectral'] = True

    # --- Input validation ---
    lfp = np.asarray(lfp)
    spikes = np.asarray(spikes)
    if lfp.ndim != 2:
        raise ValueError(f"lfp must be 2-D (R, T), got shape {lfp.shape}")
    if spikes.ndim != 3:
        raise ValueError(f"spikes must be 3-D (R, S, T_fine), got shape {spikes.shape}")
    if np.any(np.isnan(lfp)) or np.any(np.isinf(lfp)):
        raise ValueError("lfp contains NaN or Inf values")
    if np.any(np.isnan(spikes)) or np.any(np.isinf(spikes)):
        raise ValueError("spikes contains NaN or Inf values")
    # Warn on unrecognized config keys
    for name, user_cfg, defaults in [
        ("ctssmt_config", ctssmt_config, DEFAULT_CTSSMT_CONFIG_TRIALS),
        ("mcmc_config", mcmc_config, DEFAULT_MCMC_CONFIG_TRIALS),
        ("output_config", output_config, DEFAULT_OUTPUT_CONFIG_TRIALS),
    ]:
        if user_cfg is not None:
            unknown = set(user_cfg) - set(defaults)
            if unknown:
                logger.warning(f"Unrecognized keys in {name}: {unknown}")

    # Infer delta_spk if not provided
    if delta_spk is None:
        delta_spk = 1.0 / fs

    # Data dimensions
    R, T_lfp = lfp.shape
    R_spk, S, T_fine = spikes.shape

    if R != R_spk:
        raise ValueError(f"Trial count mismatch: LFP has {R}, spikes has {R_spk}")
    
    if verbose:
        logger.info("=" * 60)
        logger.info("TRIAL-STRUCTURED SPIKE-FIELD COUPLING INFERENCE")
        logger.info("=" * 60)
        logger.info(f"LFP: {lfp.shape} (R={R} trials, T={T_lfp})")
        logger.info(f"Spikes: {spikes.shape} (S={S} units, T_fine={T_fine})")
        logger.info(f"fs: {fs} Hz, delta_spk: {delta_spk}")
    
    # =================================================================
    # 1. COMPUTE SPECTROGRAM
    # =================================================================
    if verbose:
        logger.info("\n[1/5] Computing multitaper spectrogram...")
    
    freqs = np.arange(ct_cfg['freq_min'], ct_cfg['freq_max'], ct_cfg['freq_step'])
    J = len(freqs)
    NW = ct_cfg['NW']
    n_tapers = int(2 * NW - 1) if NW > 0.5 else 1
    M = n_tapers
    window_sec = ct_cfg['window_sec']
    
    if verbose:
        logger.info(f"  Frequency grid: {J} bands ({freqs[0]:.0f}-{freqs[-1]:.0f} Hz, step={ct_cfg['freq_step']})")
        logger.info(f"  NW={NW}, n_tapers={n_tapers}, window_sec={window_sec}")
    
    Y_trials = _compute_multitaper_trials(lfp, fs, freqs, window_sec, NW)
    R, J, M, K = Y_trials.shape
    
    if verbose:
        logger.info(f"  Y_trials: {Y_trials.shape} (R={R}, J={J}, M={M}, K={K} blocks)")
    
    # =================================================================
    # 2. BUILD HISTORY DESIGN
    # =================================================================
    if verbose:
        logger.info("\n[2/5] Building history design matrix...")
    
    H_SRTL = _build_history_design_trials(spikes, n_lags=mc_cfg['n_history_lags'])
    L = H_SRTL.shape[-1]
    
    if verbose:
        logger.info(f"  H_SRTL: {H_SRTL.shape}")
    
    # =================================================================
    # 3. RUN HIERARCHICAL JOINT INFERENCE
    # =================================================================
    if verbose:
        logger.info("\n[3/5] Running hierarchical joint inference...")
    
    from joint_ssmt.run_joint_inference_trials import (
        run_joint_inference_trials_hier,
        InferenceTrialsHierConfig,
    )
    
    # Prepare spikes: (R, S, T) -> (S, R, T)
    spikes_SRT = np.swapaxes(spikes, 0, 1).astype(np.float32)
    
    inference_config = InferenceTrialsHierConfig(
        fixed_iter=mc_cfg['fixed_iter'],
        n_refreshes=mc_cfg['n_refreshes'],
        inner_steps_per_refresh=mc_cfg['inner_steps'],
        trace_thin=mc_cfg['trace_thin'],
        omega_floor=mc_cfg['omega_floor'],
        sigma_u=mc_cfg['sigma_u'],
        tau2_intercept=mc_cfg['tau2_intercept'],
        tau2_gamma=mc_cfg['tau2_gamma'],
        a0_ard=mc_cfg['a0_ard'],
        b0_ard=mc_cfg['b0_ard'],
        use_wald_band_selection=mc_cfg['use_wald_selection'],
        wald_alpha=mc_cfg['wald_alpha'],
        use_beta_shrinkage=mc_cfg['use_shrinkage'],
        standardize_latents=mc_cfg['standardize_latents'],
        freeze_beta0=mc_cfg['freeze_beta0'],
        # Winsorization parameters
        use_winsorization=mc_cfg['use_winsorization'],
        winsorize_percentile=mc_cfg['winsorize_percentile'],
        winsorize_after_warmup=mc_cfg['winsorize_after_warmup'],
        winsorize_after_refresh=mc_cfg['winsorize_after_refresh'],
        em_kwargs=dict(
            max_iter=ct_cfg['em_max_iter'],
            tol=ct_cfg['em_tol'],
            sig_eps_init=ct_cfg.get('sig_eps_init', 5.0),
            log_every=ct_cfg.get('log_every', 1000),
        ),
    )
    
    beta, gamma, theta_X, theta_D, trace = run_joint_inference_trials_hier(
        Y_trials=Y_trials,
        spikes_SRT=spikes_SRT,
        H_SRTL=H_SRTL,
        all_freqs=freqs,
        delta_spk=delta_spk,
        window_sec=window_sec,
        config=inference_config,
    )
    
    if verbose:
        logger.info(f"  beta: {beta.shape}, gamma: {gamma.shape}")
    
    # =================================================================
    # 4. EXTRACT STATISTICS
    # =================================================================
    if verbose:
        logger.info("\n[4/5] Computing statistics...")
    
    # Stack traces
    beta_trace = np.stack(trace.beta, axis=0) if trace.beta else None
    gamma_trace = np.stack(trace.gamma, axis=0) if trace.gamma else None
    
    # Handle gamma shape: may be (n, S, R, L) -> take mean over R
    if gamma_trace is not None and gamma_trace.ndim == 4:
        gamma_trace = gamma_trace.mean(axis=2)  # (n, S, L)
    
    # Extract beta stats
    if beta_trace is not None:
        beta_mag, beta_phase, wald_W, wald_pval, beta_std = _extract_beta_stats(
            beta_trace, J, mc_cfg['burn_in_frac']
        )
        phase_R, phase_pval, phase_est = _extract_phase_stats(
            beta_trace, J, mc_cfg['burn_in_frac']
        )
        wald_significant = (wald_pval < mc_cfg['wald_alpha']).any(axis=0)
        
        if verbose:
            logger.info(f"  Wald significant bands: {wald_significant.sum()}/{J}")
            logger.info(f"  Phase R range: [{phase_R.min():.3f}, {phase_R.max():.3f}]")
    else:
        # Fallback to point estimate
        beta_R = beta[:, 1:1+J]
        beta_I = beta[:, 1+J:1+2*J]
        beta_mag = np.sqrt(beta_R**2 + beta_I**2)
        beta_phase = np.arctan2(beta_I, beta_R)
        wald_W = wald_pval = wald_significant = None
        phase_R = phase_pval = phase_est = None
        beta_std = None
    
    # =================================================================
    # 5. SAVE RESULTS
    # =================================================================
    if verbose:
        logger.info("\n[5/5] Saving results...")
    
    output_dir = out_cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = {}
    elapsed = time.time() - start_time
    downsample_factor = out_cfg.get('downsample_factor', 10)
    
    # -----------------------------------------------------------------
    # coupling.npz - Main coupling results
    # -----------------------------------------------------------------
    coupling_data = {
        # Point estimates
        'beta': beta,
        'gamma': gamma if gamma.ndim == 2 else gamma.mean(axis=1),  # (S, L)
        'freqs': freqs,
        'beta_mag': beta_mag,
        'beta_phase': beta_phase,
        
        # Full trace
        'beta_trace': beta_trace,
        'gamma_trace': gamma_trace,
    }
    
    # Add optional arrays
    if beta_std is not None:
        coupling_data['beta_std'] = beta_std
    if wald_W is not None:
        coupling_data['wald_W'] = wald_W
        coupling_data['wald_pval'] = wald_pval
        coupling_data['wald_significant'] = wald_significant
    if phase_R is not None:
        coupling_data['phase_R'] = phase_R
        coupling_data['phase_pval'] = phase_pval
        coupling_data['phase_est'] = phase_est
    
    # Standardization
    if hasattr(trace, 'latent_scale_factors') and trace.latent_scale_factors is not None:
        coupling_data['latent_scale_factors'] = np.asarray(trace.latent_scale_factors)
    
    # Shrinkage
    if hasattr(trace, 'shrinkage_factors') and trace.shrinkage_factors:
        coupling_data['shrinkage_factors'] = np.stack(trace.shrinkage_factors)
    
    # Winsorization thresholds
    if hasattr(trace, 'winsorize_thresholds') and trace.winsorize_thresholds:
        coupling_data['winsorize_thresholds'] = np.stack(trace.winsorize_thresholds)
    
    # Wald mask from trace (if present)
    if hasattr(trace, 'wald_significant_mask') and trace.wald_significant_mask is not None:
        coupling_data['wald_significant_mask_trace'] = np.asarray(trace.wald_significant_mask)

    # Add standardized versions
    if hasattr(trace, 'beta_standardized') and trace.beta_standardized is not None:
        coupling_data['beta_standardized'] = trace.beta_standardized
    if hasattr(trace, 'beta_trace_standardized') and trace.beta_trace_standardized:
        coupling_data['beta_trace_standardized'] = np.stack(trace.beta_trace_standardized, axis=0)
        
    coupling_path = os.path.join(output_dir, 'coupling.npz')
    np.savez_compressed(coupling_path, **coupling_data)
    saved_files['coupling'] = coupling_path
    
    if verbose:
        size_mb = os.path.getsize(coupling_path) / (1024 * 1024)
        logger.info(f"  coupling.npz: {size_mb:.2f} MB")
    
    # -----------------------------------------------------------------
    # spectral.npz - Latent dynamics and OU parameters
    # -----------------------------------------------------------------
    if out_cfg.get('save_spectral', True):
        spectral_data = {
            'Y_trials': np.asarray(Y_trials),
            'freqs': freqs,
            'downsample_factor': downsample_factor,
        }
        
        # OU parameters for X (shared)
        theta_X_dict = _theta_to_dict(theta_X)
        spectral_data['theta_X_lam'] = theta_X_dict['lam']
        spectral_data['theta_X_sig_v'] = theta_X_dict['sig_v']
        spectral_data['theta_X_sig_eps'] = theta_X_dict['sig_eps']
        
        # OU parameters for D (trial-specific)
        theta_D_dict = _theta_to_dict(theta_D)
        spectral_data['theta_D_lam'] = theta_D_dict['lam']
        spectral_data['theta_D_sig_v'] = theta_D_dict['sig_v']
        spectral_data['theta_D_sig_eps'] = theta_D_dict['sig_eps']
        
        # X (shared) fine states
        if hasattr(trace, 'X_fine') and len(trace.X_fine) > 0:
            X_fine = np.asarray(trace.X_fine[-1])
            spectral_data['X_fine'] = _downsample_array(X_fine, downsample_factor, axis=0)
            if verbose:
                logger.info(f"  X_fine: {X_fine.shape} -> {spectral_data['X_fine'].shape} (ds={downsample_factor})")
        
        if hasattr(trace, 'X_var_fine') and len(trace.X_var_fine) > 0:
            X_var_fine = np.asarray(trace.X_var_fine[-1])
            spectral_data['X_var_fine'] = _downsample_array(X_var_fine, downsample_factor, axis=0)
        
        # D (trial-specific) fine states
        if hasattr(trace, 'D_fine') and len(trace.D_fine) > 0:
            D_fine = np.asarray(trace.D_fine[-1])
            spectral_data['D_fine'] = _downsample_array(D_fine, downsample_factor, axis=1)
            if verbose:
                logger.info(f"  D_fine: {D_fine.shape} -> {spectral_data['D_fine'].shape} (ds={downsample_factor})")
        
        if hasattr(trace, 'D_var_fine') and len(trace.D_var_fine) > 0:
            D_var_fine = np.asarray(trace.D_var_fine[-1])
            spectral_data['D_var_fine'] = _downsample_array(D_var_fine, downsample_factor, axis=1)
        
        # Latent (combined X+D for backward compatibility)
        if hasattr(trace, 'latent') and len(trace.latent) > 0:
            latent = np.asarray(trace.latent[-1])
            spectral_data['latent'] = _downsample_array(latent, downsample_factor, axis=1)
        
        # Latent scale factors
        if hasattr(trace, 'latent_scale_factors') and trace.latent_scale_factors is not None:
            spectral_data['latent_scale_factors'] = np.asarray(trace.latent_scale_factors)
        
        spectral_path = os.path.join(output_dir, 'spectral.npz')
        np.savez_compressed(spectral_path, **spectral_data)
        saved_files['spectral'] = spectral_path
        
        if verbose:
            size_mb = os.path.getsize(spectral_path) / (1024 * 1024)
            logger.info(f"  spectral.npz: {size_mb:.2f} MB")
        
        # -----------------------------------------------------------------
        # ctssmt_lfp_only.npz - LFP-only (EM) estimates for comparison
        # -----------------------------------------------------------------
        # Check if LFP-only estimates are stored in trace (from EM before MCMC)
        lfp_only_data = {}
        has_lfp_only = False
        
        # Check for explicit LFP-only attributes in trace
        if hasattr(trace, 'X_fine_lfp_only') and trace.X_fine_lfp_only is not None:
            lfp_only_data['X_fine'] = np.asarray(trace.X_fine_lfp_only)
            has_lfp_only = True
        if hasattr(trace, 'D_fine_lfp_only') and trace.D_fine_lfp_only is not None:
            lfp_only_data['D_fine'] = np.asarray(trace.D_fine_lfp_only)
            has_lfp_only = True
        if hasattr(trace, 'X_var_fine_lfp_only') and trace.X_var_fine_lfp_only is not None:
            lfp_only_data['X_var_fine'] = np.asarray(trace.X_var_fine_lfp_only)
        if hasattr(trace, 'D_var_fine_lfp_only') and trace.D_var_fine_lfp_only is not None:
            lfp_only_data['D_var_fine'] = np.asarray(trace.D_var_fine_lfp_only)
        
        # If not explicitly stored, compute LFP-only from EM
        if not has_lfp_only:
            if verbose:
                logger.info("  Computing LFP-only estimates from EM...")
            try:
                from joint_ssmt.em_ct_hier_jax import em_ct_hier_jax
                from joint_ssmt.upsample_ct_hier_fine import upsample_ct_hier_fine
                
                # Re-run EM (fast, already converged in joint inference)
                em_kwargs_lfp = dict(max_iter=5000, tol=1e-3, sig_eps_init=5.0, log_every=5000)
                res_lfp = em_ct_hier_jax(Y_trials=Y_trials, db=ct_cfg['window_sec'], **em_kwargs_lfp)
                
                # Upsample to fine grid
                ups_lfp = upsample_ct_hier_fine(
                    Y_trials=Y_trials, res=res_lfp, delta_spk=delta_spk,
                    win_sec=ct_cfg['window_sec'], offset_sec=0.0, T_f=None
                )
                
                # Extract X and D in fine format (T_fine, 2*J*M)
                X_mean = np.asarray(ups_lfp.X_mean)  # (J, M, T_fine)
                D_mean = np.asarray(ups_lfp.D_mean)  # (R, J, M, T_fine)
                X_var = np.asarray(ups_lfp.X_var)    # (J, M, T_fine)
                D_var = np.asarray(ups_lfp.D_var)    # (R, J, M, T_fine)
                
                J_em, M_em, T_fine = X_mean.shape
                R_em = D_mean.shape[0]
                
                # Convert to interleaved format (T_fine, 2*J*M) for X
                X_fine_lfp = np.zeros((T_fine, 2 * J_em * M_em), dtype=float)
                X_var_fine_lfp = np.zeros((T_fine, 2 * J_em * M_em), dtype=float)
                for j in range(J_em):
                    for m in range(M_em):
                        col = 2 * (j * M_em + m)
                        X_fine_lfp[:, col] = X_mean[j, m, :].real
                        X_fine_lfp[:, col + 1] = X_mean[j, m, :].imag
                        X_var_fine_lfp[:, col] = X_var[j, m, :]
                        X_var_fine_lfp[:, col + 1] = X_var[j, m, :]
                
                # Convert to interleaved format (R, T_fine, 2*J*M) for D
                D_fine_lfp = np.zeros((R_em, T_fine, 2 * J_em * M_em), dtype=float)
                D_var_fine_lfp = np.zeros((R_em, T_fine, 2 * J_em * M_em), dtype=float)
                for r in range(R_em):
                    for j in range(J_em):
                        for m in range(M_em):
                            col = 2 * (j * M_em + m)
                            D_fine_lfp[r, :, col] = D_mean[r, j, m, :].real
                            D_fine_lfp[r, :, col + 1] = D_mean[r, j, m, :].imag
                            D_var_fine_lfp[r, :, col] = D_var[r, j, m, :]
                            D_var_fine_lfp[r, :, col + 1] = D_var[r, j, m, :]
                
                # Downsample if needed
                lfp_only_data['X_fine'] = _downsample_array(X_fine_lfp, downsample_factor, axis=0)
                lfp_only_data['D_fine'] = _downsample_array(D_fine_lfp, downsample_factor, axis=1)
                lfp_only_data['X_var_fine'] = _downsample_array(X_var_fine_lfp, downsample_factor, axis=0)
                lfp_only_data['D_var_fine'] = _downsample_array(D_var_fine_lfp, downsample_factor, axis=1)
                has_lfp_only = True
                
                if verbose:
                    logger.info(f"    X_fine_lfp_only: {lfp_only_data['X_fine'].shape}")
                    logger.info(f"    D_fine_lfp_only: {lfp_only_data['D_fine'].shape}")
                    
            except Exception as e:
                logger.warning(f"Could not compute LFP-only estimates: {e}")
        
        if has_lfp_only:
            lfp_only_data['freqs'] = freqs
            lfp_only_data['downsample_factor'] = downsample_factor
            
            lfp_only_path = os.path.join(output_dir, 'ctssmt_lfp_only.npz')
            np.savez_compressed(lfp_only_path, **lfp_only_data)
            saved_files['lfp_only'] = lfp_only_path
            
            if verbose:
                size_mb = os.path.getsize(lfp_only_path) / (1024 * 1024)
                logger.info(f"  ctssmt_lfp_only.npz: {size_mb:.2f} MB")
    
    # -----------------------------------------------------------------
    # metadata.json - Config and summary
    # -----------------------------------------------------------------
    metadata = {
        'data': {
            'n_trials': int(R),
            'n_units': int(S),
            'n_timepoints_lfp': int(T_lfp),
            'n_timepoints_spk': int(T_fine),
            'n_blocks': int(K),
            'n_history_lags': int(L),
            'fs': float(fs),
            'delta_spk': float(delta_spk),
        },
        'ctssmt': {
            'freq_min': float(ct_cfg['freq_min']),
            'freq_max': float(ct_cfg['freq_max']),
            'freq_step': float(ct_cfg['freq_step']),
            'n_freqs': int(J),
            'window_sec': float(window_sec),
            'NW': float(NW),
            'n_tapers': int(n_tapers),
            'em_max_iter': int(ct_cfg['em_max_iter']),
        },
        'mcmc': {
            'fixed_iter': int(mc_cfg['fixed_iter']),
            'n_refreshes': int(mc_cfg['n_refreshes']),
            'inner_steps': int(mc_cfg['inner_steps']),
            'trace_thin': int(mc_cfg['trace_thin']),
            'burn_in_frac': float(mc_cfg['burn_in_frac']),
            'n_samples': int(len(trace.beta)) if trace.beta else 0,
        },
        'detection': {
            'use_wald_selection': bool(mc_cfg['use_wald_selection']),
            'wald_alpha': float(mc_cfg['wald_alpha']),
            'use_shrinkage': bool(mc_cfg['use_shrinkage']),
            'standardize_latents': bool(mc_cfg['standardize_latents']),
            'use_winsorization': bool(mc_cfg['use_winsorization']),
            'winsorize_percentile': float(mc_cfg['winsorize_percentile']),
            'n_significant_bands': int(wald_significant.sum()) if wald_significant is not None else 0,
        },
        'output': {
            'save_spectral': bool(out_cfg.get('save_spectral', True)),
            'downsample_factor': int(downsample_factor),
        },
        'timing': {
            'elapsed_seconds': float(elapsed),
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        },
    }
    
    # Add ground truth info
    if ground_truth is not None:
        metadata['ground_truth'] = {
            'freqs_hz': _numpy_to_python(ground_truth.get('freqs_hz', [])),
            'freqs_hz_extra': _numpy_to_python(ground_truth.get('freqs_hz_extra', [])),
            'has_masks': 'masks' in ground_truth,
            'has_beta_mag': 'beta_mag' in ground_truth,
        }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    saved_files['metadata'] = metadata_path
    
    if verbose:
        logger.info(f"  metadata.json")

    # =================================================================
    # 6. DEFAULT PLOTS (optional)
    # =================================================================
    if plot:
        if verbose:
            logger.info("\n[6/6] Generating default plots...")
        try:
            results_for_plot = load_results_trials(output_dir)
            from joint_ssmt.plotting.summary import plot_all_default
            plot_paths = plot_all_default(
                results_for_plot,
                save_dir=output_dir,
                fmt="pdf",
                dpi=300,
            )
            saved_files['figures'] = plot_paths
        except Exception as e:
            logger.warning(f"Plotting failed: {e}")

    if verbose:
        logger.info(f"\nTotal time: {elapsed:.1f} seconds")
        logger.info("=" * 60)

    return saved_files


# LOAD FUNCTIONS

def load_coupling_trials(filepath: str) -> Dict[str, np.ndarray]:
    """Load coupling results from npz file."""
    data = np.load(filepath, allow_pickle=False)
    return {key: data[key] for key in data.files}


def load_spectral_trials(filepath: str) -> Dict[str, np.ndarray]:
    """Load spectral results from npz file."""
    data = np.load(filepath, allow_pickle=False)
    return {key: data[key] for key in data.files}


def load_lfp_only_trials(filepath: str) -> Dict[str, np.ndarray]:
    """Load LFP-only (EM) estimates from npz file."""
    data = np.load(filepath, allow_pickle=False)
    return {key: data[key] for key in data.files}


def load_metadata_trials(filepath: str) -> Dict[str, Any]:
    """Load metadata from json file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_results_trials(output_dir: str) -> Dict[str, Any]:
    """
    Load all results from output directory.
    
    Returns dict with 'coupling', 'spectral' (if exists), 'lfp_only' (if exists), 'metadata'.
    """
    results = {}
    
    coupling_path = os.path.join(output_dir, 'coupling.npz')
    if os.path.exists(coupling_path):
        results['coupling'] = load_coupling_trials(coupling_path)
    
    spectral_path = os.path.join(output_dir, 'spectral.npz')
    if os.path.exists(spectral_path):
        results['spectral'] = load_spectral_trials(spectral_path)
    
    lfp_only_path = os.path.join(output_dir, 'ctssmt_lfp_only.npz')
    if os.path.exists(lfp_only_path):
        results['lfp_only'] = load_lfp_only_trials(lfp_only_path)
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        results['metadata'] = load_metadata_trials(metadata_path)
    
    return results


def results_to_legacy_dict_trials(output_dir: str) -> Dict[str, Any]:
    """
    Load results and convert to legacy pickle format.
    
    This allows using existing plotting scripts unchanged.
    """
    results = load_results_trials(output_dir)
    coupling = results['coupling']
    metadata = results['metadata']
    
    freqs = coupling['freqs']
    J = len(freqs)
    
    legacy = {
        # Basic
        'freqs': freqs,
        'freqs_dense': freqs,
        'beta': coupling['beta'],
        'gamma': coupling['gamma'],
        
        # Derived
        'coupling': {
            'beta_mag_mean': coupling['beta_mag'],
            'beta_phase_mean': coupling['beta_phase'],
        },
        
        # Trace (in format expected by plotting)
        'trace': {
            'beta': coupling['beta_trace'],
            'gamma': coupling['gamma_trace'],
        },
        
        # Metadata
        'window_sec': metadata['ctssmt']['window_sec'],
        'NW': metadata['ctssmt']['NW'],
        'n_tapers': metadata['ctssmt']['n_tapers'],
        'delta_spk': metadata['data']['delta_spk'],
        'fs': metadata['data']['fs'],
        'config': metadata['mcmc'],
    }
    
    # Optional coupling stats
    if 'beta_std' in coupling:
        legacy['beta_std'] = coupling['beta_std']
    if 'latent_scale_factors' in coupling:
        legacy['latent_scale_factors'] = coupling['latent_scale_factors']
        legacy['trace']['latent_scale_factors'] = coupling['latent_scale_factors']
    if 'shrinkage_factors' in coupling:
        legacy['shrinkage_factors'] = coupling['shrinkage_factors']
    if 'winsorize_thresholds' in coupling:
        legacy['winsorize_thresholds'] = coupling['winsorize_thresholds']
    
    # Wald
    if 'wald_W' in coupling:
        legacy['wald'] = {
            'W': coupling['wald_W'],
            'pval_wald': coupling['wald_pval'],
            'significant_mask': coupling['wald_significant'],
        }
    
    # Spectral
    if 'spectral' in results:
        spectral = results['spectral']
        legacy['Y_trials'] = spectral.get('Y_trials')
        
        # OU params
        legacy['theta_X'] = {
            'lam': spectral.get('theta_X_lam'),
            'sig_v': spectral.get('theta_X_sig_v'),
            'sig_eps': spectral.get('theta_X_sig_eps'),
        }
        legacy['theta_D'] = {
            'lam': spectral.get('theta_D_lam'),
            'sig_v': spectral.get('theta_D_sig_v'),
            'sig_eps': spectral.get('theta_D_sig_eps'),
        }
        
        # X/D fine states
        if 'X_fine' in spectral:
            legacy['trace']['X_fine_final'] = spectral['X_fine']
        if 'X_var_fine' in spectral:
            legacy['trace']['X_var_fine_final'] = spectral['X_var_fine']
        if 'D_fine' in spectral:
            legacy['trace']['D_fine_final'] = spectral['D_fine']
        if 'D_var_fine' in spectral:
            legacy['trace']['D_var_fine_final'] = spectral['D_var_fine']
        if 'latent' in spectral:
            legacy['trace']['latent'] = [spectral['latent']]
        if 'latent_scale_factors' in spectral:
            legacy['trace']['latent_scale_factors'] = spectral['latent_scale_factors']
        
        legacy['downsample_factor'] = int(spectral.get('downsample_factor', 10))
    
    return legacy