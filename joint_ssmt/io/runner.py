"""
Spike-Field Coupling Inference Runner

Main entry point: run_inference()

Usage (new config-based API)::

    from joint_ssmt.io.runner import run_inference

    saved = run_inference(
        lfp=lfp_array,
        spikes=spikes_array,
        spectral_config="configs/spectral_default.yaml",
        inference_config="configs/inference_default.yaml",
        output_config={"output_dir": "./results"},
        fs=1000.0,
        plot=True,
    )

Legacy dict-based calls still work::

    saved = run_inference(
        lfp, spikes,
        ctssmt_config={'freq_min': 1, 'freq_max': 61},
        mcmc_config={'fixed_iter': 1000, 'n_refreshes': 5},
        output_config={'output_dir': './results'},
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

DEFAULT_CTSSMT_CONFIG = {
    'freq_min': 1.0,
    'freq_max': 61.0,
    'freq_step': 1.0,
    'window_sec': 2.0,
    'NW': 2.0,
    'em_max_iter': 2000,
    'em_tol': 1e-6,
}

DEFAULT_MCMC_CONFIG = {
    'fixed_iter': 1000,           # Gibbs iterations during warmup phase
    'n_refreshes': 5,             # Number of KF/RTS latent refresh cycles
    'inner_steps': 200,           # Gibbs iterations per refresh cycle
    'trace_thin': 2,              # Keep every N-th sample in the trace
    'burn_in_frac': 0.6,          # Fraction of trace to discard as burn-in
    'n_history_lags': 20,         # Number of spike-history lags for autoregressive term
    # Priors
    'omega_floor': 1e-3,          # Floor for Polya-Gamma auxiliary variable (numerical stability)
    'tau2_intercept': 25.0,       # Prior variance on beta_0 (intercept / baseline firing)
    'tau2_gamma': 625.0,          # Prior variance on gamma (spike-history coefficients)
    'a0_ard': 0.5,                # ARD half-Cauchy shape: small → stronger sparsity on coupling
    'b0_ard': 0.5,                # ARD half-Cauchy scale: controls prior width on |beta_j|
    # Features
    'use_wald_selection': True,   # Enable Wald-test band selection after warmup
    'wald_alpha': 0.1,            # Significance level for Wald test (per-band)
    'use_shrinkage': True,        # Enable ARD shrinkage on coupling coefficients
                                  # NOTE: maps to internal `use_beta_shrinkage` parameter
    'standardize_latents': True,  # Z-score latent amplitudes before coupling regression
    'freeze_beta0': True,         # Fix intercept at posterior mean during refresh cycles
    'enable_latent_refresh': True, # Enable KF/RTS refresh of spectral latents
}

DEFAULT_OUTPUT_CONFIG = {
    'output_dir': './results',
    'save_spectral': False,
    'save_fine': True,           # save fine-resolution latent states (needed for spectral dynamics plots)
}


# HELPER FUNCTIONS

def _merge_config(user_config: Optional[Dict], default_config: Dict) -> Dict:
    """Merge user config with defaults."""
    if user_config is None:
        return default_config.copy()
    merged = default_config.copy()
    merged.update(user_config)
    return merged


def _convert_fine_to_block(Z_fine, Z_var_fine, J, M, K, window_sec, delta_spk):
    """Convert fine state (1, T, 2*J*M) to block format (J, M, K) complex."""
    T_fine = Z_fine.shape[1]
    block_size = int(window_sec / delta_spk)
    
    Z_complex = np.zeros((J, M, T_fine), dtype=complex)
    Z_var_out = np.zeros((J, M, T_fine), dtype=float)
    
    for j in range(J):
        for m in range(M):
            col_re = 2 * (j * M + m)
            col_im = col_re + 1
            Z_complex[j, m, :] = Z_fine[0, :, col_re] + 1j * Z_fine[0, :, col_im]
            Z_var_out[j, m, :] = Z_var_fine[0, :, col_re] + Z_var_fine[0, :, col_im]
    
    K_actual = min(K, T_fine // block_size)
    Z_mean = Z_complex[:, :, ::block_size][:, :, :K_actual]
    Z_var = Z_var_out[:, :, ::block_size][:, :, :K_actual]
    
    return Z_mean, Z_var


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


# MAIN FUNCTION

def run_inference(
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
    # Legacy aliases (take precedence if provided)
    ctssmt_config: Optional[Dict[str, Any]] = None,
    mcmc_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Run spike-field coupling inference.

    Parameters
    ----------
    lfp : (T,) array
        Local field potential signal.
    spikes : (S, T) array
        Spike trains (binary).
    spectral_config : str, dict, SpectralConfig, or None
        Spectral / multitaper config.  Accepts a YAML path, a dict,
        or a ``SpectralConfig`` dataclass.
    inference_config : str, dict, InferenceConfig, or None
        MCMC / Gibbs sampler config.
    output_config : str, dict, OutputConfig, or None
        Output and plotting config.
    fs : float
        Sampling rate (Hz).
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
        # Legacy path: caller used old dict API
        ct_cfg = _merge_config(ctssmt_config, DEFAULT_CTSSMT_CONFIG)
    else:
        spec = load_config(spectral_config, SpectralConfig)
        ct_cfg = _merge_config(spec.to_ctssmt_dict(), DEFAULT_CTSSMT_CONFIG)

    if mcmc_config is not None:
        mc_cfg = _merge_config(mcmc_config, DEFAULT_MCMC_CONFIG)
    else:
        inf = load_config(inference_config, InferenceConfig)
        mc_cfg = _merge_config(inf.to_mcmc_dict(), DEFAULT_MCMC_CONFIG)

    if isinstance(output_config, OutputConfig):
        out_obj = output_config
        out_cfg = _merge_config(out_obj.to_output_dict(), DEFAULT_OUTPUT_CONFIG)
    elif isinstance(output_config, dict):
        out_obj = None
        out_cfg = _merge_config(output_config, DEFAULT_OUTPUT_CONFIG)
    else:
        out_obj = load_config(output_config, OutputConfig)
        out_cfg = _merge_config(out_obj.to_output_dict(), DEFAULT_OUTPUT_CONFIG)

    # Always save spectral data so spectrograms can be plotted later
    if plot:
        out_cfg['save_spectral'] = True

    # --- Input validation ---
    lfp = np.asarray(lfp)
    spikes = np.asarray(spikes)
    if lfp.ndim != 1:
        raise ValueError(f"lfp must be 1-D (T,), got shape {lfp.shape}")
    if spikes.ndim != 2:
        raise ValueError(f"spikes must be 2-D (S, T), got shape {spikes.shape}")
    if np.any(np.isnan(lfp)) or np.any(np.isinf(lfp)):
        raise ValueError("lfp contains NaN or Inf values")
    if np.any(np.isnan(spikes)) or np.any(np.isinf(spikes)):
        raise ValueError("spikes contains NaN or Inf values")
    # Warn on unrecognized config keys
    for name, user_cfg, defaults in [
        ("ctssmt_config", ctssmt_config, DEFAULT_CTSSMT_CONFIG),
        ("mcmc_config", mcmc_config, DEFAULT_MCMC_CONFIG),
        ("output_config", output_config, DEFAULT_OUTPUT_CONFIG),
    ]:
        if user_cfg is not None:
            unknown = set(user_cfg) - set(defaults)
            if unknown:
                logger.warning(f"Unrecognized keys in {name}: {unknown}")

    # Infer delta_spk if not provided
    if delta_spk is None:
        delta_spk = 1.0 / fs

    # Data dimensions
    S, T_spk = spikes.shape
    T_lfp = len(lfp)
    
    if verbose:
        logger.info("=" * 60)
        logger.info("SPIKE-FIELD COUPLING INFERENCE")
        logger.info("=" * 60)
        logger.info(f"LFP: {lfp.shape}, Spikes: {spikes.shape}")
        logger.info(f"fs: {fs} Hz, delta_spk: {delta_spk}")
    
    # =================================================================
    # 1. COMPUTE SPECTROGRAM
    # =================================================================
    if verbose:
        logger.info("\n[1/5] Computing multitaper spectrogram...")
    
    import mne
    from joint_ssmt.utils_multitaper import derotate_tfr_align_start
    
    freqs = np.arange(ct_cfg['freq_min'], ct_cfg['freq_max'], ct_cfg['freq_step'])
    J = len(freqs)
    NW = ct_cfg['NW']
    n_tapers = int(2 * NW - 1)
    M = n_tapers
    window_sec = ct_cfg['window_sec']
    
    if verbose:
        logger.info(f"  Frequency grid: {J} bands ({freqs[0]:.0f}-{freqs[-1]:.0f} Hz)")
        logger.info(f"  NW={NW}, n_tapers={n_tapers}")
    
    tfr_raw = mne.time_frequency.tfr_array_multitaper(
        lfp[None, None, :],
        sfreq=fs,
        freqs=freqs,
        n_cycles=freqs * window_sec,
        time_bandwidth=2 * NW,
        output='complex',
        zero_mean=False,
    )
    
    # Reshape to (J, M, T)
    if tfr_raw.ndim == 5:
        tfr = tfr_raw[0, 0, :, :, :].transpose(1, 0, 2)
    else:
        tfr = tfr_raw[0, 0, :, :][:, None, :]
    
    J, M, T = tfr.shape
    M_samples = int(window_sec * fs)
    
    # Derotate and scale
    tfr = derotate_tfr_align_start(tfr, freqs, fs, 1, M_samples)
    tapers, _ = mne.time_frequency.multitaper.dpss_windows(M_samples, NW, Kmax=n_tapers)
    scaling = 2.0 / tapers[0].sum()
    tfr = tfr * scaling
    
    # Downsample to blocks
    Y_cube = tfr[:, :, ::M_samples]
    J, M_tapers, K = Y_cube.shape
    
    if verbose:
        logger.info(f"  Y_cube: {Y_cube.shape} ({K} time blocks)")
    
    # =================================================================
    # 2. BUILD HISTORY DESIGN
    # =================================================================
    if verbose:
        logger.info("\n[2/5] Building history design matrix...")
    
    from joint_ssmt.simulate_single_trial import build_history_design_single
    
    H_hist = build_history_design_single(spikes, n_lags=mc_cfg['n_history_lags'])
    L = H_hist.shape[-1]
    
    if verbose:
        logger.info(f"  H_hist: {H_hist.shape}")
    
    # =================================================================
    # 3. RUN JOINT INFERENCE
    # =================================================================
    if verbose:
        logger.info("\n[3/5] Running joint inference...")
    
    from joint_ssmt.run_joint_inference_single_trial import (
        run_joint_inference_single_trial,
        SingleTrialInferenceConfig
    )
    
    inference_config = SingleTrialInferenceConfig(
        fixed_iter=mc_cfg['fixed_iter'],
        n_refreshes=mc_cfg['n_refreshes'],
        inner_steps_per_refresh=mc_cfg['inner_steps'],
        trace_thin=mc_cfg['trace_thin'],
        omega_floor=mc_cfg['omega_floor'],
        tau2_intercept=mc_cfg['tau2_intercept'],
        tau2_gamma=mc_cfg['tau2_gamma'],
        a0_ard=mc_cfg['a0_ard'],
        b0_ard=mc_cfg['b0_ard'],
        use_wald_band_selection=mc_cfg['use_wald_selection'],
        wald_alpha=mc_cfg['wald_alpha'],
        use_beta_shrinkage=mc_cfg['use_shrinkage'],
        standardize_latents=mc_cfg['standardize_latents'],
        freeze_beta0=mc_cfg['freeze_beta0'],
        enable_latent_refresh=mc_cfg['enable_latent_refresh'],
        em_kwargs=dict(
            max_iter=ct_cfg['em_max_iter'],
            tol=ct_cfg['em_tol'],
        ),
    )
    
    beta, gamma, theta, trace = run_joint_inference_single_trial(
        Y_cube=Y_cube,
        spikes_ST=spikes,
        H_STL=H_hist,
        all_freqs=freqs,
        delta_spk=delta_spk,
        window_sec=window_sec,
        config=inference_config,
    )
    
    if verbose:
        logger.info(f"  beta: {beta.shape}, gamma: {gamma.shape}")
    
    # =================================================================
    # 4. EXTRACT AND COMPUTE STATISTICS
    # =================================================================
    if verbose:
        logger.info("\n[4/5] Computing statistics...")
    
    # Stack trace
    beta_trace = np.stack(trace.beta, axis=0) if trace.beta else None
    gamma_trace = np.stack(trace.gamma, axis=0) if trace.gamma else None
    
    # Derived quantities from TRACE MEAN (matches compare_coupling_methods_single_trial.py)
    # The old code computes: mR, mI = bR.mean(0), bI.mean(0); joint_mag = np.sqrt(mR**2 + mI**2)
    if beta_trace is not None:
        burn = int(mc_cfg['burn_in_frac'] * beta_trace.shape[0])
        post = beta_trace[burn:]
        bR = post[:, :, 1:1+J]
        bI = post[:, :, 1+J:1+2*J]
        mR, mI = bR.mean(axis=0), bI.mean(axis=0)
        beta_mag = np.sqrt(mR**2 + mI**2)
        beta_phase = np.arctan2(mI, mR)
    else:
        # Fallback to point estimate
        beta_R = beta[:, 1:1+J]
        beta_I = beta[:, 1+J:1+2*J]
        beta_mag = np.sqrt(beta_R**2 + beta_I**2)
        beta_phase = np.arctan2(beta_I, beta_R)
    
    # Wald test from trace
    if beta_trace is not None:
        from scipy import stats as sp_stats
        
        burn = int(mc_cfg['burn_in_frac'] * beta_trace.shape[0])
        post = beta_trace[burn:]
        
        bR = post[:, :, 1:1+J]
        bI = post[:, :, 1+J:1+2*J]
        
        # Compute Wald stats
        wald_W = np.zeros((S, J))
        wald_pval = np.zeros((S, J))
        
        for s in range(S):
            for j in range(J):
                br, bi = bR[:, s, j], bI[:, s, j]
                mu = np.array([br.mean(), bi.mean()])
                Sig = np.cov(np.column_stack([br, bi]), rowvar=False) + 1e-10*np.eye(2)
                wald_W[s, j] = mu @ np.linalg.solve(Sig, mu)
                wald_pval[s, j] = 1 - sp_stats.chi2.cdf(wald_W[s, j], df=2)
        
        wald_significant = (wald_pval < mc_cfg['wald_alpha']).any(axis=0)
        
        # Phase concentration test
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
        
        # Posterior std
        beta_std = beta_trace[burn:].std(axis=0)
        
        if verbose:
            logger.info(f"  Wald significant bands: {wald_significant.sum()}/{J}")
            logger.info(f"  Phase R range: [{phase_R.min():.3f}, {phase_R.max():.3f}]")
    else:
        wald_W = wald_pval = wald_significant = None
        phase_R = phase_pval = phase_est = None
        beta_std = None
    
    # Extract spectral dynamics
    Z_mean_em = Z_var_em = Z_smooth_em = None
    Z_mean_joint = Z_var_joint = Z_smooth_joint = None
    Z_fine_em = Z_var_fine_em = None
    Z_fine_joint = Z_var_fine_joint = None
    
    if hasattr(trace, 'Z_fine_em') and trace.Z_fine_em is not None:
        Z_mean_em, Z_var_em = _convert_fine_to_block(
            trace.Z_fine_em, trace.Z_var_em, J, M, K, window_sec, delta_spk
        )
        Z_smooth_em = Z_mean_em.mean(axis=1)
        
        if out_cfg['save_fine']:
            Z_fine_em = np.asarray(trace.Z_fine_em)
            Z_var_fine_em = np.asarray(trace.Z_var_em)
    
    if hasattr(trace, 'Z_fine_joint') and trace.Z_fine_joint is not None:
        Z_mean_joint, Z_var_joint = _convert_fine_to_block(
            trace.Z_fine_joint, trace.Z_var_joint, J, M, K, window_sec, delta_spk
        )
        Z_smooth_joint = Z_mean_joint.mean(axis=1)
        
        if out_cfg['save_fine']:
            Z_fine_joint = np.asarray(trace.Z_fine_joint)
            Z_var_fine_joint = np.asarray(trace.Z_var_joint)
    
    # =================================================================
    # 5. SAVE RESULTS
    # =================================================================
    if verbose:
        logger.info("\n[5/5] Saving results...")
    
    output_dir = out_cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = {}
    elapsed = time.time() - start_time
    
    # -----------------------------------------------------------------
    # coupling.npz - Main results
    # -----------------------------------------------------------------
    coupling_data = {
        # Point estimates
        'beta': beta,
        'gamma': gamma,
        'freqs': freqs,
        'beta_mag': beta_mag,
        'beta_phase': beta_phase,
        
        # Full trace (for posterior plots)
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
    if hasattr(trace, 'beta_standardized') and trace.beta_standardized is not None:
        coupling_data['beta_standardized'] = np.asarray(trace.beta_standardized)
    if hasattr(trace, 'latent_scale_factors') and trace.latent_scale_factors is not None:
        coupling_data['latent_scale_factors'] = np.asarray(trace.latent_scale_factors)
    
    # Shrinkage
    if hasattr(trace, 'shrinkage_factors') and trace.shrinkage_factors:
        coupling_data['shrinkage_factors'] = np.stack(trace.shrinkage_factors)
    
    # EM parameters
    coupling_data['theta_lam'] = np.asarray(theta.lam)
    coupling_data['theta_sig_v'] = np.asarray(theta.sig_v)
    coupling_data['theta_sig_eps'] = np.asarray(theta.sig_eps)
    
    coupling_path = os.path.join(output_dir, 'coupling.npz')
    np.savez_compressed(coupling_path, **coupling_data)
    saved_files['coupling'] = coupling_path
    
    if verbose:
        size_mb = os.path.getsize(coupling_path) / (1024 * 1024)
        logger.info(f"  coupling.npz: {size_mb:.2f} MB")
    
    # -----------------------------------------------------------------
    # spectral.npz - Latent dynamics (optional)
    # -----------------------------------------------------------------
    if out_cfg['save_spectral']:
        spectral_data = {
            'Y_cube': np.asarray(Y_cube),
            'freqs': freqs,
        }
        
        if Z_mean_em is not None:
            spectral_data['Z_mean_em'] = Z_mean_em
            spectral_data['Z_var_em'] = Z_var_em
            spectral_data['Z_smooth_em'] = Z_smooth_em
        
        if Z_mean_joint is not None:
            spectral_data['Z_mean_joint'] = Z_mean_joint
            spectral_data['Z_var_joint'] = Z_var_joint
            spectral_data['Z_smooth_joint'] = Z_smooth_joint
        
        if out_cfg['save_fine']:
            if Z_fine_em is not None:
                spectral_data['Z_fine_em'] = Z_fine_em
                spectral_data['Z_var_fine_em'] = Z_var_fine_em
            if Z_fine_joint is not None:
                spectral_data['Z_fine_joint'] = Z_fine_joint
                spectral_data['Z_var_fine_joint'] = Z_var_fine_joint
        
        spectral_path = os.path.join(output_dir, 'spectral.npz')
        np.savez_compressed(spectral_path, **spectral_data)
        saved_files['spectral'] = spectral_path
        
        if verbose:
            size_mb = os.path.getsize(spectral_path) / (1024 * 1024)
            logger.info(f"  spectral.npz: {size_mb:.2f} MB")
    
    # -----------------------------------------------------------------
    # metadata.json - Config and summary
    # -----------------------------------------------------------------
    metadata = {
        'data': {
            'n_units': int(S),
            'n_timepoints_lfp': int(T_lfp),
            'n_timepoints_spk': int(T_spk),
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
            'n_significant_bands': int(wald_significant.sum()) if wald_significant is not None else 0,
        },
        'output': {
            'save_spectral': bool(out_cfg['save_spectral']),
            'save_fine': bool(out_cfg['save_fine']),
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
            'has_masks': 'masks' in ground_truth,
            'has_beta_mag': 'beta_mag' in ground_truth,
            'has_beta_phase': 'beta_phase' in ground_truth,
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
            results_for_plot = load_results(output_dir)
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

def load_coupling(filepath: str) -> Dict[str, np.ndarray]:
    """Load coupling results from npz file."""
    data = np.load(filepath, allow_pickle=False)
    return {key: data[key] for key in data.files}


def load_spectral(filepath: str) -> Dict[str, np.ndarray]:
    """Load spectral results from npz file."""
    data = np.load(filepath, allow_pickle=False)
    return {key: data[key] for key in data.files}


def load_metadata(filepath: str) -> Dict[str, Any]:
    """Load metadata from json file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_results(output_dir: str) -> Dict[str, Any]:
    """
    Load all results from output directory.
    
    Returns dict with 'coupling', 'spectral' (if exists), 'metadata'.
    """
    results = {}
    
    coupling_path = os.path.join(output_dir, 'coupling.npz')
    if os.path.exists(coupling_path):
        results['coupling'] = load_coupling(coupling_path)
    
    spectral_path = os.path.join(output_dir, 'spectral.npz')
    if os.path.exists(spectral_path):
        results['spectral'] = load_spectral(spectral_path)
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        results['metadata'] = load_metadata(metadata_path)
    
    return results


def results_to_legacy_dict(output_dir: str) -> Dict[str, Any]:
    """
    Load results and convert to legacy pickle format.
    
    This allows using existing plotting scripts unchanged.
    """
    results = load_results(output_dir)
    coupling = results['coupling']
    metadata = results['metadata']
    
    freqs = coupling['freqs']
    J = len(freqs)
    
    legacy = {
        # Basic
        'freqs': freqs,
        'beta': coupling['beta'],
        'gamma': coupling['gamma'],
        
        # Derived
        'coupling': {
            'beta_mag': coupling['beta_mag'],
            'beta_phase': coupling['beta_phase'],
        },
        
        # Trace
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
    
    # Optional
    if 'beta_standardized' in coupling:
        legacy['beta_standardized'] = coupling['beta_standardized']
    if 'latent_scale_factors' in coupling:
        legacy['latent_scale_factors'] = coupling['latent_scale_factors']
    if 'shrinkage_factors' in coupling:
        legacy['shrinkage_factors'] = coupling['shrinkage_factors']
    
    # Wald
    if 'wald_W' in coupling:
        legacy['wald'] = {
            'W_stats': coupling['wald_W'],
            'pval': coupling['wald_pval'],
            'significant_mask': coupling['wald_significant'],
        }
    
    # EM params
    if 'theta_lam' in coupling:
        legacy['theta_em'] = {
            'lam': coupling['theta_lam'],
            'sig_v': coupling['theta_sig_v'],
            'sig_eps': coupling['theta_sig_eps'],
        }
    
    # Spectral
    if 'spectral' in results:
        spectral = results['spectral']
        legacy['Y_cube'] = spectral.get('Y_cube')
        legacy['Z_mean_em'] = spectral.get('Z_mean_em')
        legacy['Z_mean_joint'] = spectral.get('Z_mean_joint')
        legacy['Z_var_em'] = spectral.get('Z_var_em')
        legacy['Z_var_joint'] = spectral.get('Z_var_joint')
        legacy['Z_smooth_em'] = spectral.get('Z_smooth_em')
        legacy['Z_smooth_joint'] = spectral.get('Z_smooth_joint')
        legacy['Z_smooth'] = spectral.get('Z_smooth_joint', spectral.get('Z_smooth_em'))
        
        # Fine resolution
        legacy['Z_fine_em'] = spectral.get('Z_fine_em')
        legacy['Z_fine_joint'] = spectral.get('Z_fine_joint')
        legacy['Z_var_fine_em'] = spectral.get('Z_var_fine_em')
        legacy['Z_var_fine_joint'] = spectral.get('Z_var_fine_joint')
    
    return legacy