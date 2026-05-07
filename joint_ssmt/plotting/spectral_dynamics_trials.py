"""
Spectral Dynamics Comparison for Trial-Structured Data

Compares three methods for estimating trial-specific spectral dynamics:
1. **Multi-taper** (derotated to baseband)
2. **CT-SSMT (LFP-only)**
3. **CT-SSMT (Joint LFP + Spikes)**

Generates (matching notebook figures):
- trial_specific_comparison.png   (Fig 1: sample trials x frequencies)
- deviation_comparison.png        (Fig 2: trial deviation from mean)
- trial_averaged_comparison.png   (Fig 3: trial-averaged line plots)
- spectrogram_trial_specific.png  (Fig 4: heatmaps per trial)
- spectrogram_trial_averaged.png  (Fig 5: heatmaps averaged)
- spectrogram_deviation.png       (Fig 6: deviation spectrograms)
- correlation_boxplot.png         (Fig 7: per-trial correlation boxplot)
- psd_comparison.png              (Fig 8: power spectral density)

Works with or without ground truth.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import interp1d
from typing import Optional, Dict, List, Tuple, Sequence

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False


# =============================================================================
# FIGURE STYLE + METHOD COLORS
# =============================================================================

def set_style(font_size: int = 7):
    """Set publication-quality figure style."""
    if HAS_SEABORN:
        sns.set(
            style="ticks",
            context="paper",
            font="sans-serif",
            rc={
                "font.size": font_size,
                "figure.titlesize": font_size,
                "axes.titlesize": font_size,
                "axes.labelsize": font_size,
                "axes.linewidth": 0.5,
                "lines.linewidth": 1,
                "lines.markersize": 3,
                "xtick.labelsize": font_size,
                "ytick.labelsize": font_size,
                "savefig.transparent": True,
                "xtick.major.size": 2.5,
                "ytick.major.size": 2.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "legend.fontsize": font_size,
                "legend.title_fontsize": font_size,
                "legend.frameon": False,
            },
        )
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42


METHOD_CONFIG = {
    "mt":  {"label": "Multi-taper",        "color": "#2E86AB", "linestyle": "-", "linewidth": 1.5},
    "lfp": {"label": "CT-SSMT (LFP-only)", "color": "#A23B72", "linestyle": "-", "linewidth": 1.5},
    "spk": {"label": "CT-SSMT (Joint)",    "color": "#F18F01", "linestyle": "-", "linewidth": 1.5},
    "gt":  {"label": "Ground Truth",       "color": "#333333", "linestyle": "-", "linewidth": 1.0},
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_optimal_scale(estimate: np.ndarray, target: np.ndarray) -> float:
    """Compute optimal scaling factor to match estimate to target (least squares)."""
    est_flat = np.abs(estimate).flatten()
    tgt_flat = np.abs(target).flatten()
    denom = np.dot(est_flat, est_flat)
    if denom < 1e-10:
        return 1.0
    return np.dot(tgt_flat, est_flat) / denom


def compute_global_scale(
    Z_est: np.ndarray,
    Z_gt: np.ndarray,
    freq_indices: Sequence[int],
) -> float:
    """Compute global scale factor across all trials and selected frequencies."""
    est_flat = np.abs(Z_est[:, freq_indices, :]).flatten()
    tgt_flat = np.abs(Z_gt[:, freq_indices, :]).flatten()
    denom = np.dot(est_flat, est_flat)
    if denom < 1e-10:
        return 1.0
    return np.dot(tgt_flat, est_flat) / denom


def resample_to_target(Z: np.ndarray, T_target: int, axis: int = -1) -> np.ndarray:
    """Resample array along time axis to target length."""
    T_source = Z.shape[axis]
    if T_source == T_target:
        return Z
    
    t_source = np.linspace(0, 1, T_source)
    t_target = np.linspace(0, 1, T_target)
    
    # Move axis to last position
    Z = np.moveaxis(Z, axis, -1)
    shape = Z.shape[:-1]
    
    Z_flat = Z.reshape(-1, T_source)
    Z_resampled = np.zeros((Z_flat.shape[0], T_target), dtype=Z.dtype)
    
    for i in range(Z_flat.shape[0]):
        if np.iscomplexobj(Z):
            f_re = interp1d(t_source, Z_flat[i].real, kind='linear', fill_value='extrapolate')
            f_im = interp1d(t_source, Z_flat[i].imag, kind='linear', fill_value='extrapolate')
            Z_resampled[i] = f_re(t_target) + 1j * f_im(t_target)
        else:
            f = interp1d(t_source, Z_flat[i], kind='linear', fill_value='extrapolate')
            Z_resampled[i] = f(t_target)
    
    Z_resampled = Z_resampled.reshape(shape + (T_target,))
    return np.moveaxis(Z_resampled, -1, axis)


def extract_complex_from_separated(lat_reim: np.ndarray, J: int) -> np.ndarray:
    """
    Convert SEPARATED format [Re_0..Re_{J-1}, Im_0..Im_{J-1}] to complex.
    
    Input: (R, T, 2*J)
    Output: (R, J, T)
    """
    Z_re = lat_reim[:, :, :J]
    Z_im = lat_reim[:, :, J:]
    Z_complex = Z_re + 1j * Z_im
    return np.transpose(Z_complex, (0, 2, 1))


def extract_complex_from_interleaved(fine_state: np.ndarray, J: int, M: int = 1) -> np.ndarray:
    """
    Convert INTERLEAVED format [Re_0, Im_0, Re_1, Im_1, ...] to complex.
    
    Input: (T, 2*J*M) or (R, T, 2*J*M)
    Output: (J, T) or (R, J, T)
    """
    fine_state = np.asarray(fine_state)
    
    if fine_state.ndim == 2:
        T_fine, D = fine_state.shape
        Z = np.zeros((J, T_fine), dtype=complex)
        for j in range(J):
            for m in range(M):
                base = (j * M + m) * 2
                Z[j, :] += fine_state[:, base] + 1j * fine_state[:, base + 1]
        return Z / M
    
    elif fine_state.ndim == 3:
        R_dim, T_fine, D = fine_state.shape
        Z = np.zeros((R_dim, J, T_fine), dtype=complex)
        for j in range(J):
            for m in range(M):
                base = (j * M + m) * 2
                Z[:, j, :] += fine_state[:, :, base] + 1j * fine_state[:, :, base + 1]
        return Z / M
    
    else:
        raise ValueError(f"Unexpected shape: {fine_state.shape}")


def extract_variance_from_interleaved(var_state: np.ndarray, J: int, M: int = 1) -> np.ndarray:
    """Extract variance for each frequency from INTERLEAVED format."""
    var_state = np.asarray(var_state)
    
    if var_state.ndim == 2:
        T_fine, D = var_state.shape
        V = np.zeros((J, T_fine), dtype=float)
        for j in range(J):
            for m in range(M):
                base = (j * M + m) * 2
                V[j, :] += var_state[:, base] + var_state[:, base + 1]
        return V / (M * M)
    
    elif var_state.ndim == 3:
        R_dim, T_fine, D = var_state.shape
        V = np.zeros((R_dim, J, T_fine), dtype=float)
        for j in range(J):
            for m in range(M):
                base = (j * M + m) * 2
                V[:, j, :] += var_state[:, :, base] + var_state[:, :, base + 1]
        return V / (M * M)
    
    else:
        raise ValueError(f"Unexpected shape: {var_state.shape}")


def derotate_tfr(tfr_rotated: np.ndarray, freqs: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Derotate TFR from rotated (carrier) form to baseband."""
    R, J, T = tfr_rotated.shape
    tfr_baseband = np.zeros_like(tfr_rotated)
    for j, f in enumerate(freqs):
        phase = 2.0 * np.pi * f * time
        tfr_baseband[:, j, :] = tfr_rotated[:, j, :] * np.exp(-1j * phase)[None, :]
    return tfr_baseband


def compute_trial_correlations(
    Z_est: np.ndarray,
    Z_gt: np.ndarray,
    freq_indices: Sequence[int],
    time_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute correlation between estimate and ground truth for each trial."""
    R = Z_est.shape[0]
    n_freqs = len(freq_indices)
    correlations = np.zeros((R, n_freqs))
    
    for i, j in enumerate(freq_indices):
        for r in range(R):
            if time_mask is not None:
                est = np.abs(Z_est[r, j, time_mask])
                gt = np.abs(Z_gt[r, j, time_mask])
            else:
                est = np.abs(Z_est[r, j, :])
                gt = np.abs(Z_gt[r, j, :])
            
            if est.std() > 1e-10 and gt.std() > 1e-10:
                correlations[r, i] = np.corrcoef(est, gt)[0, 1]
    
    return correlations


# =============================================================================
# FIGURE 1: TRIAL-SPECIFIC COMPARISON
# =============================================================================

def plot_trial_specific_comparison(
    Z_gt: Optional[np.ndarray],  # (R, J, T)
    Z_mt: np.ndarray,            # (R, J, T)
    Z_lfp: Optional[np.ndarray], # (R, J, T)
    Z_spk: Optional[np.ndarray], # (R, J, T)
    freqs: np.ndarray,
    time: np.ndarray,
    output_path: str,
    *,
    idx_sig: Sequence[int],
    freqs_true: np.ndarray,
    sample_trials: List[int] = [0, 25, 57, 85],
    time_range: Tuple[float, float] = (1.0, 8.0),
    Z_spk_var: Optional[np.ndarray] = None,
    scale_mt: float = 1.0,
    scale_lfp: float = 1.0,
    scale_spk: float = 1.0,
) -> str:
    """
    Plot trial-specific amplitude comparison (Fig 1 in notebook).
    
    Layout: frequencies (rows) x sample trials (columns)
    """
    set_style(font_size=8)
    
    R, J, T = Z_mt.shape
    has_gt = Z_gt is not None
    has_lfp = Z_lfp is not None
    has_spk = Z_spk is not None
    
    # Filter valid trials
    sample_trials = [t for t in sample_trials if t < R]
    if len(sample_trials) == 0:
        sample_trials = list(np.linspace(0, R-1, min(4, R), dtype=int))
    
    n_freqs = len(idx_sig)
    n_trials = len(sample_trials)
    
    # Time mask
    t_mask = (time >= time_range[0]) & (time <= time_range[1])
    t_plot = time[t_mask]
    
    fig, axes = plt.subplots(n_freqs, n_trials, figsize=(3.5*n_trials, 2.5*n_freqs), squeeze=False)
    
    for i, (j_true, j_dense) in enumerate(zip(range(len(idx_sig)), idx_sig)):
        freq_hz = freqs_true[j_true] if j_true < len(freqs_true) else freqs[j_dense]
        
        for col, trial_idx in enumerate(sample_trials):
            ax = axes[i, col]
            
            # Ground truth
            if has_gt:
                amp_gt = np.abs(Z_gt[trial_idx, j_dense, t_mask])
                ax.plot(t_plot, amp_gt, color=METHOD_CONFIG['gt']['color'],
                       lw=METHOD_CONFIG['gt']['linewidth'], label=METHOD_CONFIG['gt']['label'], zorder=5)
                
                # Add std band across trials for GT
                amp_gt_all = np.abs(Z_gt[:, j_dense, t_mask])
                mean_gt = amp_gt_all.mean(axis=0)
                std_gt = amp_gt_all.std(axis=0)
                ax.fill_between(t_plot, mean_gt - std_gt, mean_gt + std_gt,
                               color=METHOD_CONFIG['gt']['color'], alpha=0.15, zorder=0)
            
            # Multitaper
            amp_mt = np.abs(Z_mt[trial_idx, j_dense, t_mask]) * scale_mt
            ax.plot(t_plot, amp_mt, color=METHOD_CONFIG['mt']['color'],
                   lw=METHOD_CONFIG['mt']['linewidth'], label=METHOD_CONFIG['mt']['label'], alpha=0.6, zorder=1)
            
            # LFP-only
            if has_lfp:
                amp_lfp = np.abs(Z_lfp[trial_idx, j_dense, t_mask]) * scale_lfp
                ax.plot(t_plot, amp_lfp, color=METHOD_CONFIG['lfp']['color'],
                       lw=METHOD_CONFIG['lfp']['linewidth'], label=METHOD_CONFIG['lfp']['label'], alpha=0.7, zorder=2)
            
            # Joint
            if has_spk:
                amp_spk = np.abs(Z_spk[trial_idx, j_dense, t_mask]) * scale_spk
                ax.plot(t_plot, amp_spk, color=METHOD_CONFIG['spk']['color'],
                       lw=METHOD_CONFIG['spk']['linewidth'], label=METHOD_CONFIG['spk']['label'], alpha=0.8, zorder=3)
                
                # Posterior variance band
                if Z_spk_var is not None:
                    std_spk = np.sqrt(Z_spk_var[trial_idx, j_dense, t_mask]) * scale_spk
                    ax.fill_between(t_plot, amp_spk - std_spk, amp_spk + std_spk,
                                   color=METHOD_CONFIG['spk']['color'], alpha=0.2, zorder=0)
            
            ax.set_xlim(time_range)
            if i == 0:
                ax.set_title(f'Trial {trial_idx}', fontsize=9, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{freq_hz:.0f} Hz\nAmplitude', fontsize=8)
            if i == 0 and col == n_trials - 1:
                ax.legend(loc='upper right', fontsize=6)
            if i == n_freqs - 1:
                ax.set_xlabel('Time (s)', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


# =============================================================================
# FIGURE 2: DEVIATION COMPARISON
# =============================================================================

def plot_deviation_comparison(
    Z_gt: Optional[np.ndarray],
    Z_mt: np.ndarray,
    Z_lfp: Optional[np.ndarray],
    Z_spk: Optional[np.ndarray],
    freqs: np.ndarray,
    time: np.ndarray,
    output_path: str,
    *,
    idx_sig: Sequence[int],
    freqs_true: np.ndarray,
    sample_trials: List[int] = [0, 25, 57, 85],
    time_range: Tuple[float, float] = (1.0, 8.0),
    Z_spk_var: Optional[np.ndarray] = None,
) -> str:
    """
    Plot deviation term comparison (Fig 2 in notebook).
    
    Deviation = |Z_trial| - mean(|Z|) across trials
    """
    set_style(font_size=8)
    
    R, J, T = Z_mt.shape
    has_gt = Z_gt is not None
    has_lfp = Z_lfp is not None
    has_spk = Z_spk is not None
    
    # Compute trial means
    Z_gt_mean = np.abs(Z_gt).mean(axis=0) if has_gt else None
    Z_mt_mean = np.abs(Z_mt).mean(axis=0)
    Z_lfp_mean = np.abs(Z_lfp).mean(axis=0) if has_lfp else None
    Z_spk_mean = np.abs(Z_spk).mean(axis=0) if has_spk else None
    
    # Compute deviations
    Z_gt_dev = np.abs(Z_gt) - Z_gt_mean[None, :, :] if has_gt else None
    Z_mt_dev = np.abs(Z_mt) - Z_mt_mean[None, :, :]
    Z_lfp_dev = np.abs(Z_lfp) - Z_lfp_mean[None, :, :] if has_lfp else None
    Z_spk_dev = np.abs(Z_spk) - Z_spk_mean[None, :, :] if has_spk else None
    
    # Compute deviation scales
    def compute_dev_scale(dev_est, dev_gt, freq_indices):
        est = dev_est[:, freq_indices, :].flatten()
        gt = dev_gt[:, freq_indices, :].flatten()
        denom = np.dot(np.abs(est), np.abs(est))
        if denom < 1e-10:
            return 1.0
        return np.dot(np.abs(gt), np.abs(est)) / denom
    
    if has_gt:
        scale_mt_dev = compute_dev_scale(Z_mt_dev, Z_gt_dev, idx_sig)
        scale_lfp_dev = compute_dev_scale(Z_lfp_dev, Z_gt_dev, idx_sig) if has_lfp else 1.0
        scale_spk_dev = compute_dev_scale(Z_spk_dev, Z_gt_dev, idx_sig) if has_spk else 1.0
    else:
        scale_mt_dev = scale_lfp_dev = scale_spk_dev = 1.0
    
    # Filter valid trials
    sample_trials = [t for t in sample_trials if t < R]
    if len(sample_trials) == 0:
        sample_trials = list(np.linspace(0, R-1, min(4, R), dtype=int))
    
    n_freqs = len(idx_sig)
    n_trials = len(sample_trials)
    
    t_mask = (time >= time_range[0]) & (time <= time_range[1])
    t_plot = time[t_mask]
    
    fig, axes = plt.subplots(n_freqs, n_trials, figsize=(3.5*n_trials, 2.5*n_freqs), squeeze=False)
    
    for i, (j_true, j_dense) in enumerate(zip(range(len(idx_sig)), idx_sig)):
        freq_hz = freqs_true[j_true] if j_true < len(freqs_true) else freqs[j_dense]
        
        for col, trial_idx in enumerate(sample_trials):
            ax = axes[i, col]
            
            # Ground truth deviation
            if has_gt:
                dev_gt = Z_gt_dev[trial_idx, j_dense, t_mask]
                ax.plot(t_plot, dev_gt, color=METHOD_CONFIG['gt']['color'],
                       lw=METHOD_CONFIG['gt']['linewidth'], label='GT Deviation', zorder=5)
            
            # MT deviation
            dev_mt = Z_mt_dev[trial_idx, j_dense, t_mask] * scale_mt_dev
            ax.plot(t_plot, dev_mt, color=METHOD_CONFIG['mt']['color'],
                   lw=METHOD_CONFIG['mt']['linewidth'], label='MT Deviation', alpha=0.6, zorder=1)
            
            # LFP deviation
            if has_lfp:
                dev_lfp = Z_lfp_dev[trial_idx, j_dense, t_mask] * scale_lfp_dev
                ax.plot(t_plot, dev_lfp, color=METHOD_CONFIG['lfp']['color'],
                       lw=METHOD_CONFIG['lfp']['linewidth'], label='LFP Deviation', alpha=0.7, zorder=2)
            
            # Joint deviation
            if has_spk:
                dev_spk = Z_spk_dev[trial_idx, j_dense, t_mask] * scale_spk_dev
                ax.plot(t_plot, dev_spk, color=METHOD_CONFIG['spk']['color'],
                       lw=METHOD_CONFIG['spk']['linewidth'], label='Joint Deviation', alpha=0.8, zorder=3)
                
                if Z_spk_var is not None:
                    std_spk = np.sqrt(Z_spk_var[trial_idx, j_dense, t_mask]) * scale_spk_dev
                    ax.fill_between(t_plot, dev_spk - std_spk, dev_spk + std_spk,
                                   color=METHOD_CONFIG['spk']['color'], alpha=0.2, zorder=0)
            
            ax.axhline(0, color='gray', linestyle='--', lw=0.8, alpha=0.5)
            ax.set_xlim(time_range)
            if i == 0:
                ax.set_title(f'Trial {trial_idx}', fontsize=9, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{freq_hz:.0f} Hz\nDeviation', fontsize=8)
            if i == 0 and col == n_trials - 1:
                ax.legend(loc='upper right', fontsize=6)
            if i == n_freqs - 1:
                ax.set_xlabel('Time (s)', fontsize=8)
    
    plt.suptitle('Trial-Specific Deviation: |Z_r| − mean(|Z|)', fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


# =============================================================================
# FIGURE 3: TRIAL-AVERAGED LINE PLOTS
# =============================================================================

def plot_trial_averaged_comparison(
    Z_gt: Optional[np.ndarray],
    Z_mt: np.ndarray,
    Z_lfp: Optional[np.ndarray],
    Z_spk: Optional[np.ndarray],
    freqs: np.ndarray,
    time: np.ndarray,
    output_path: str,
    *,
    idx_sig: Sequence[int],
    freqs_true: np.ndarray,
    freqs_extra: Optional[np.ndarray] = None,
    time_range: Tuple[float, float] = (1.0, 8.0),
) -> str:
    """Plot trial-averaged amplitude with std bands (Fig 3 in notebook)."""
    set_style(font_size=8)
    
    R, J, T = Z_mt.shape
    has_gt = Z_gt is not None
    has_lfp = Z_lfp is not None
    has_spk = Z_spk is not None
    
    # Build set of signal-only frequencies
    freqs_extra_set = set()
    if freqs_extra is not None:
        freqs_extra_set = set(int(f) for f in freqs_extra)
    
    n_freqs = len(idx_sig)
    t_mask = (time >= time_range[0]) & (time <= time_range[1])
    t_plot = time[t_mask]
    
    fig, axes = plt.subplots(n_freqs, 1, figsize=(6, 2.5*n_freqs), squeeze=False)
    
    for i, (j_true, j_dense) in enumerate(zip(range(len(idx_sig)), idx_sig)):
        freq_hz = freqs_true[j_true] if j_true < len(freqs_true) else freqs[j_dense]
        ax = axes[i, 0]
        
        # Check if signal-only
        tag = ' (signal-only)' if int(freq_hz) in freqs_extra_set else ''
        
        # Ground truth
        if has_gt:
            amp_gt = np.abs(Z_gt[:, j_dense, t_mask])
            mean_gt = amp_gt.mean(axis=0)
            std_gt = amp_gt.std(axis=0)
            ax.plot(t_plot, mean_gt, color=METHOD_CONFIG['gt']['color'], lw=1.0, label='Ground Truth')
            ax.fill_between(t_plot, mean_gt - std_gt, mean_gt + std_gt,
                           color=METHOD_CONFIG['gt']['color'], alpha=0.15)
        
        # Multitaper
        amp_mt = np.abs(Z_mt[:, j_dense, t_mask])
        mean_mt = amp_mt.mean(axis=0)
        if has_gt:
            scale_mt = compute_optimal_scale(mean_mt, mean_gt)
        else:
            scale_mt = 1.0
        ax.plot(t_plot, mean_mt * scale_mt, color=METHOD_CONFIG['mt']['color'], lw=1.2,
               label=METHOD_CONFIG['mt']['label'], alpha=0.7)
        
        # LFP-only
        if has_lfp:
            amp_lfp = np.abs(Z_lfp[:, j_dense, t_mask])
            mean_lfp = amp_lfp.mean(axis=0)
            if has_gt:
                scale_lfp = compute_optimal_scale(mean_lfp, mean_gt)
            else:
                scale_lfp = 1.0
            ax.plot(t_plot, mean_lfp * scale_lfp, color=METHOD_CONFIG['lfp']['color'], lw=1.2,
                   label=METHOD_CONFIG['lfp']['label'], alpha=0.8)
        
        # Joint
        if has_spk:
            amp_spk = np.abs(Z_spk[:, j_dense, t_mask])
            mean_spk = amp_spk.mean(axis=0)
            if has_gt:
                scale_spk = compute_optimal_scale(mean_spk, mean_gt)
            else:
                scale_spk = 1.0
            ax.plot(t_plot, mean_spk * scale_spk, color=METHOD_CONFIG['spk']['color'], lw=1.2,
                   label=METHOD_CONFIG['spk']['label'], alpha=0.9)
        
        ax.set_ylabel('Amplitude')
        ax.set_title(f'{freq_hz:.0f} Hz{tag} - Trial Averaged')
        ax.legend(loc='upper right', fontsize=7)
        ax.set_xlim(time_range)
    
    axes[-1, 0].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


# =============================================================================
# FIGURE 7: CORRELATION BOXPLOT
# =============================================================================

def plot_correlation_boxplot(
    Z_gt: np.ndarray,
    Z_mt: np.ndarray,
    Z_lfp: Optional[np.ndarray],
    Z_spk: Optional[np.ndarray],
    freqs: np.ndarray,
    time: np.ndarray,
    output_path: str,
    *,
    idx_sig: Sequence[int],
    freqs_true: np.ndarray,
    time_range: Tuple[float, float] = (1.0, 8.0),
) -> str:
    """Plot correlation boxplot across all trials (Fig 7 in notebook)."""
    set_style(font_size=8)
    
    if not HAS_SEABORN or not HAS_PANDAS:
        print("  [SKIP] correlation_boxplot: seaborn/pandas not available")
        return output_path
    
    R, J, T = Z_mt.shape
    has_lfp = Z_lfp is not None
    has_spk = Z_spk is not None
    
    t_mask = (time >= time_range[0]) & (time <= time_range[1])
    
    # Compute correlations
    corr_mt = compute_trial_correlations(Z_mt, Z_gt, idx_sig, t_mask)
    corr_lfp = compute_trial_correlations(Z_lfp, Z_gt, idx_sig, t_mask) if has_lfp else None
    corr_spk = compute_trial_correlations(Z_spk, Z_gt, idx_sig, t_mask) if has_spk else None
    
    # Build dataframe
    data_list = []
    for i, j_dense in enumerate(idx_sig):
        freq_hz = freqs_true[i] if i < len(freqs_true) else freqs[j_dense]
        for r in range(R):
            data_list.append({
                'Frequency (Hz)': f'{int(freq_hz)}',
                'Correlation': corr_mt[r, i],
                'Method': METHOD_CONFIG['mt']['label']
            })
            if has_lfp:
                data_list.append({
                    'Frequency (Hz)': f'{int(freq_hz)}',
                    'Correlation': corr_lfp[r, i],
                    'Method': METHOD_CONFIG['lfp']['label']
                })
            if has_spk:
                data_list.append({
                    'Frequency (Hz)': f'{int(freq_hz)}',
                    'Correlation': corr_spk[r, i],
                    'Method': METHOD_CONFIG['spk']['label']
                })
    
    df_corr = pd.DataFrame(data_list)
    
    # Build method order and palette
    method_order = [METHOD_CONFIG['mt']['label']]
    palette = {METHOD_CONFIG['mt']['label']: METHOD_CONFIG['mt']['color']}
    if has_lfp:
        method_order.append(METHOD_CONFIG['lfp']['label'])
        palette[METHOD_CONFIG['lfp']['label']] = METHOD_CONFIG['lfp']['color']
    if has_spk:
        method_order.append(METHOD_CONFIG['spk']['label'])
        palette[METHOD_CONFIG['spk']['label']] = METHOD_CONFIG['spk']['color']
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.boxplot(data=df_corr, x='Frequency (Hz)', y='Correlation', hue='Method',
                hue_order=method_order, palette=palette, width=0.6, linewidth=0.8, fliersize=2, ax=ax)
    
    ax.set_xlabel('Frequency (Hz)', fontsize=9)
    ax.set_ylabel('Correlation with Ground Truth', fontsize=9)
    ax.set_ylim([-0.5, 1.05])
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(title=None, loc='upper center', bbox_to_anchor=(0.5, -0.12),
             frameon=False, fontsize=7, ncol=3)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Print summary
    print("\n  Correlation Summary (median ± IQR):")
    for i, j_dense in enumerate(idx_sig):
        freq_hz = freqs_true[i] if i < len(freqs_true) else freqs[j_dense]
        print(f"    {freq_hz:.0f} Hz:")
        print(f"      {METHOD_CONFIG['mt']['label']:<25}: {np.median(corr_mt[:, i]):.3f} ± {np.percentile(corr_mt[:, i], 75) - np.percentile(corr_mt[:, i], 25):.3f}")
        if has_lfp:
            print(f"      {METHOD_CONFIG['lfp']['label']:<25}: {np.median(corr_lfp[:, i]):.3f} ± {np.percentile(corr_lfp[:, i], 75) - np.percentile(corr_lfp[:, i], 25):.3f}")
        if has_spk:
            print(f"      {METHOD_CONFIG['spk']['label']:<25}: {np.median(corr_spk[:, i]):.3f} ± {np.percentile(corr_spk[:, i], 75) - np.percentile(corr_spk[:, i], 25):.3f}")
    
    return output_path


# =============================================================================
# FIGURE 8: PSD COMPARISON
# =============================================================================

def plot_psd_comparison(
    Z_mt: np.ndarray,
    Z_lfp: Optional[np.ndarray],
    Z_spk: Optional[np.ndarray],
    freqs: np.ndarray,
    output_path: str,
    *,
    idx_sig: Sequence[int],
    scale_mt: float = 1.0,
    scale_lfp: float = 1.0,
    scale_spk: float = 1.0,
) -> str:
    """Plot power spectral density comparison (Fig 8 in notebook)."""
    set_style(font_size=8)
    
    R, J, T = Z_mt.shape
    has_lfp = Z_lfp is not None
    has_spk = Z_spk is not None
    
    def compute_psd_stats(Z):
        power_per_trial = np.mean(np.abs(Z)**2, axis=2)  # (R, J)
        psd_mean = np.mean(power_per_trial, axis=0)      # (J,)
        psd_std = np.std(power_per_trial, axis=0)        # (J,)
        return psd_mean, psd_std
    
    psd_mt_mean, psd_mt_std = compute_psd_stats(Z_mt)
    psd_mt_mean *= (scale_mt ** 2)
    psd_mt_std *= (scale_mt ** 2)
    
    if has_lfp:
        psd_lfp_mean, psd_lfp_std = compute_psd_stats(Z_lfp)
        psd_lfp_mean *= (scale_lfp ** 2)
        psd_lfp_std *= (scale_lfp ** 2)
    
    if has_spk:
        psd_spk_mean, psd_spk_std = compute_psd_stats(Z_spk)
        psd_spk_mean *= (scale_spk ** 2)
        psd_spk_std *= (scale_spk ** 2)
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    # Multitaper
    ax.fill_between(freqs, psd_mt_mean - psd_mt_std, psd_mt_mean + psd_mt_std,
                   color=METHOD_CONFIG['mt']['color'], alpha=0.2)
    ax.plot(freqs, psd_mt_mean, color=METHOD_CONFIG['mt']['color'],
           linewidth=2, label=METHOD_CONFIG['mt']['label'], zorder=2)
    
    # LFP-only
    if has_lfp:
        ax.fill_between(freqs, psd_lfp_mean - psd_lfp_std, psd_lfp_mean + psd_lfp_std,
                       color=METHOD_CONFIG['lfp']['color'], alpha=0.2)
        ax.plot(freqs, psd_lfp_mean, color=METHOD_CONFIG['lfp']['color'],
               linewidth=2, label=METHOD_CONFIG['lfp']['label'], zorder=3)
    
    # Joint
    if has_spk:
        ax.fill_between(freqs, psd_spk_mean - psd_spk_std, psd_spk_mean + psd_spk_std,
                       color=METHOD_CONFIG['spk']['color'], alpha=0.2)
        ax.plot(freqs, psd_spk_mean, color=METHOD_CONFIG['spk']['color'],
               linewidth=2, label=METHOD_CONFIG['spk']['label'], zorder=4)
    
    # Mark signal frequencies
    for j in idx_sig:
        ax.axvline(freqs[j], color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
    
    ax.set_xlabel('Frequency (Hz)', fontsize=9)
    ax.set_ylabel('Mean Power (a.u.)', fontsize=9)
    ax.set_xlim([freqs[0] - 1, freqs[-1] + 1])
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=7, ncol=3)
    
    if HAS_SEABORN:
        sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


# =============================================================================
# SPECTROGRAM FIGURES
# =============================================================================

def plot_spectrogram_trial_specific(
    Z_gt: Optional[np.ndarray],
    Z_mt: np.ndarray,
    Z_lfp: Optional[np.ndarray],
    Z_spk: Optional[np.ndarray],
    freqs: np.ndarray,
    time: np.ndarray,
    output_path: str,
    *,
    freqs_true: np.ndarray,
    freqs_coupled: Optional[np.ndarray] = None,
    freqs_extra: Optional[np.ndarray] = None,
    sample_trials: List[int] = [0, 5, 10],
    time_range: Tuple[float, float] = (1.0, 8.0),
) -> str:
    """Plot trial-specific spectrograms (Fig 4 in notebook)."""
    set_style(font_size=8)
    
    R, J, T = Z_mt.shape
    has_gt = Z_gt is not None
    has_lfp = Z_lfp is not None
    has_spk = Z_spk is not None
    
    sample_trials = [t for t in sample_trials if t < R]
    n_trials = len(sample_trials)
    
    t_mask = (time >= time_range[0]) & (time <= time_range[1])
    t_start = np.where(t_mask)[0][0]
    t_end = np.where(t_mask)[0][-1] + 1
    t_plot = time[t_mask]
    
    # Default: all true frequencies are coupled
    if freqs_coupled is None:
        freqs_coupled = freqs_true
    freqs_extra_set = set(int(f) for f in freqs_extra) if freqs_extra is not None else set()
    
    # Build method list
    methods = []
    method_data = []
    if has_gt:
        methods.append(('gt', 'Ground Truth'))
        method_data.append(Z_gt)
    methods.append(('mt', METHOD_CONFIG['mt']['label']))
    method_data.append(Z_mt)
    if has_lfp:
        methods.append(('lfp', METHOD_CONFIG['lfp']['label']))
        method_data.append(Z_lfp)
    if has_spk:
        methods.append(('spk', METHOD_CONFIG['spk']['label']))
        method_data.append(Z_spk)
    
    n_methods = len(methods)
    
    # Compute vmin/vmax per method
    method_vmax = []
    method_vmin = []
    for data in method_data:
        all_amps = [np.abs(data[r, :, t_start:t_end]) for r in sample_trials]
        method_vmax.append(np.percentile(np.concatenate([a.flatten() for a in all_amps]), 99.9))
        method_vmin.append(np.percentile(np.concatenate([a.flatten() for a in all_amps]), 5))
    
    fig, axes = plt.subplots(n_trials, n_methods, figsize=(3.5*n_methods, 2.5*n_trials))
    extent = [t_plot[0], t_plot[-1], freqs[0], freqs[-1]]
    
    for i, trial_idx in enumerate(sample_trials):
        for j, ((key, label), data, vmax, vmin) in enumerate(zip(methods, method_data, method_vmax, method_vmin)):
            ax = axes[i, j] if n_trials > 1 else axes[j]
            amp = np.abs(data[trial_idx, :, t_start:t_end])
            im = ax.imshow(amp, aspect='auto', origin='lower', extent=extent,
                          cmap='viridis', vmin=vmin, vmax=vmax, interpolation='none')
            
            # Mark true frequencies: dashed for coupled, dotted for extra
            for freq in freqs_true:
                if int(freq) in freqs_extra_set:
                    ax.axhline(freq, color='white', linestyle=':', lw=1.0, alpha=0.9)
                else:
                    ax.axhline(freq, color='white', linestyle='--', lw=0.8, alpha=0.7)
            
            if i == 0:
                ax.set_title(label, fontsize=10, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'Trial {trial_idx}\nFreq (Hz)', fontsize=9)
            else:
                ax.set_yticks([])
            if i == n_trials - 1:
                ax.set_xlabel('Time (s)', fontsize=9)
            else:
                ax.set_xticks([])
    
    fig.suptitle('Trial-Specific Spectrograms', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def plot_spectrogram_trial_averaged(
    Z_gt: Optional[np.ndarray],
    Z_mt: np.ndarray,
    Z_lfp: Optional[np.ndarray],
    Z_spk: Optional[np.ndarray],
    freqs: np.ndarray,
    time: np.ndarray,
    output_path: str,
    *,
    freqs_true: np.ndarray,
    freqs_coupled: Optional[np.ndarray] = None,
    freqs_extra: Optional[np.ndarray] = None,
    time_range: Tuple[float, float] = (1.0, 8.0),
    scale_mt: float = 1.0,
    scale_lfp: float = 1.0,
    scale_spk: float = 1.0,
) -> str:
    """Plot trial-averaged spectrograms (Fig 5 in notebook)."""
    set_style(font_size=8)
    
    R, J, T = Z_mt.shape
    has_gt = Z_gt is not None
    has_lfp = Z_lfp is not None
    has_spk = Z_spk is not None
    
    t_mask = (time >= time_range[0]) & (time <= time_range[1])
    t_start = np.where(t_mask)[0][0]
    t_end = np.where(t_mask)[0][-1] + 1
    t_plot = time[t_mask]
    
    # Default: all true frequencies are coupled
    if freqs_coupled is None:
        freqs_coupled = freqs_true
    freqs_extra_set = set(int(f) for f in freqs_extra) if freqs_extra is not None else set()
    
    # Compute trial-averaged amplitudes
    Z_gt_avg = np.abs(Z_gt).mean(axis=0) if has_gt else None
    Z_mt_avg = np.abs(Z_mt).mean(axis=0) * scale_mt
    Z_lfp_avg = np.abs(Z_lfp).mean(axis=0) * scale_lfp if has_lfp else None
    Z_spk_avg = np.abs(Z_spk).mean(axis=0) * scale_spk if has_spk else None
    
    # Build method list
    methods = []
    method_data = []
    if has_gt:
        methods.append('Ground Truth')
        method_data.append(Z_gt_avg)
    methods.append(METHOD_CONFIG['mt']['label'])
    method_data.append(Z_mt_avg)
    if has_lfp:
        methods.append(METHOD_CONFIG['lfp']['label'])
        method_data.append(Z_lfp_avg)
    if has_spk:
        methods.append(METHOD_CONFIG['spk']['label'])
        method_data.append(Z_spk_avg)
    
    n_methods = len(methods)
    
    # Compute global vmin/vmax
    all_data = np.concatenate([d[:, t_start:t_end].flatten() for d in method_data])
    vmax = np.percentile(all_data, 99)
    vmin = np.percentile(all_data, 5)
    
    fig, axes = plt.subplots(1, n_methods, figsize=(3.5*n_methods, 3))
    extent = [t_plot[0], t_plot[-1], freqs[0], freqs[-1]]
    
    for j, (label, data) in enumerate(zip(methods, method_data)):
        ax = axes[j]
        amp = data[:, t_start:t_end]
        im = ax.imshow(amp, aspect='auto', origin='lower', extent=extent,
                      cmap='viridis', vmin=vmin, vmax=vmax, interpolation='none')
        
        # Mark frequencies: dashed for coupled, dotted for extra
        for freq in freqs_true:
            if int(freq) in freqs_extra_set:
                ax.axhline(freq, color='white', linestyle=':', lw=1.2, alpha=0.95)
            else:
                ax.axhline(freq, color='white', linestyle='--', lw=1.0, alpha=0.8)
        
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=9)
        if j == 0:
            ax.set_ylabel('Frequency (Hz)', fontsize=9)
        else:
            ax.set_yticks([])
    
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='|Z| (scaled)')
    
    plt.suptitle(f'Trial-Averaged Spectrograms (n={R} trials)', fontsize=12)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


# =============================================================================
# SPECTROGRAM DEVIATION (Fig 6)
# =============================================================================

def plot_spectrogram_deviation(
    Z_gt: Optional[np.ndarray],
    Z_mt: np.ndarray,
    Z_lfp: Optional[np.ndarray],
    Z_spk: Optional[np.ndarray],
    freqs: np.ndarray,
    time: np.ndarray,
    output_path: str,
    *,
    freqs_true: np.ndarray,
    sample_trials: List[int] = [0, 5, 10],
    time_range: Tuple[float, float] = (1.0, 8.0),
    scale_mt: float = 1.0,
    scale_lfp: float = 1.0,
    scale_spk: float = 1.0,
) -> str:
    """Plot deviation spectrograms (Fig 6 in notebook)."""
    set_style(font_size=8)
    
    R, J, T = Z_mt.shape
    has_gt = Z_gt is not None
    has_lfp = Z_lfp is not None
    has_spk = Z_spk is not None
    
    sample_trials = [t for t in sample_trials if t < R]
    n_trials = len(sample_trials)
    
    t_mask = (time >= time_range[0]) & (time <= time_range[1])
    t_start = np.where(t_mask)[0][0]
    t_end = np.where(t_mask)[0][-1] + 1
    t_plot = time[t_mask]
    
    # Compute trial-averaged amplitudes
    Z_gt_avg = np.abs(Z_gt).mean(axis=0) if has_gt else None
    Z_mt_avg = np.abs(Z_mt).mean(axis=0)
    Z_lfp_avg = np.abs(Z_lfp).mean(axis=0) if has_lfp else None
    Z_spk_avg = np.abs(Z_spk).mean(axis=0) if has_spk else None
    
    # Compute deviations
    Z_gt_dev = np.abs(Z_gt) - Z_gt_avg[None, :, :] if has_gt else None
    Z_mt_dev = np.abs(Z_mt) - Z_mt_avg[None, :, :]
    Z_lfp_dev = np.abs(Z_lfp) - Z_lfp_avg[None, :, :] if has_lfp else None
    Z_spk_dev = np.abs(Z_spk) - Z_spk_avg[None, :, :] if has_spk else None
    
    # Build method list
    dev_data = []
    dev_labels = []
    dev_scales = []
    if has_gt:
        dev_data.append(Z_gt_dev)
        dev_labels.append('GT Deviation')
        dev_scales.append(1.0)
    dev_data.append(Z_mt_dev)
    dev_labels.append('MT Deviation')
    dev_scales.append(scale_mt)
    if has_lfp:
        dev_data.append(Z_lfp_dev)
        dev_labels.append('LFP Deviation')
        dev_scales.append(scale_lfp)
    if has_spk:
        dev_data.append(Z_spk_dev)
        dev_labels.append('Joint Deviation')
        dev_scales.append(scale_spk)
    
    n_methods = len(dev_data)
    
    # Compute global vmin/vmax
    all_devs = []
    for data, scale in zip(dev_data, dev_scales):
        for trial_idx in sample_trials:
            all_devs.append(data[trial_idx, :, t_start:t_end] * scale)
    vmax = np.percentile(np.abs(np.concatenate([d.flatten() for d in all_devs])), 95)
    vmin = -vmax
    
    fig, axes = plt.subplots(n_trials, n_methods, figsize=(3.5*n_methods + 0.5, 2.5*n_trials))
    extent = [t_plot[0], t_plot[-1], freqs[0], freqs[-1]]
    
    im = None
    for i, trial_idx in enumerate(sample_trials):
        for j, (data, label, scale) in enumerate(zip(dev_data, dev_labels, dev_scales)):
            ax = axes[i, j] if n_trials > 1 else axes[j]
            dev = data[trial_idx, :, t_start:t_end] * scale
            im = ax.imshow(dev, aspect='auto', origin='lower', extent=extent,
                          cmap='RdBu_r', vmin=vmin, vmax=vmax, interpolation='none')
            
            # Mark true frequencies
            for freq in freqs_true:
                ax.axhline(freq, color='black', linestyle='--', lw=0.8, alpha=0.7)
            
            if i == 0:
                ax.set_title(label, fontsize=10, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'Trial {trial_idx}\nFreq (Hz)', fontsize=9)
            else:
                ax.set_yticks([])
            if i == n_trials - 1:
                ax.set_xlabel('Time (s)', fontsize=9)
            else:
                ax.set_xticks([])
    
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Deviation (scaled)')
    
    plt.suptitle('Deviation Spectrograms: |Z_r| − mean(|Z|)', fontsize=12)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def generate_trial_dynamics_figures(
    sim_data: dict,
    joint_results: dict,
    lfp_results: Optional[dict],
    output_dir: str,
    *,
    freqs_dense: Optional[np.ndarray] = None,
    plot_freqs: Optional[Sequence[float]] = None,
    time_range: Tuple[float, float] = (1.0, 8.0),
    sample_trials: List[int] = [0, 25, 57, 85],
    fs: float = 1000.0,
    window_sec: float = 0.4,
    NW: float = 1.0,
) -> Dict[str, str]:
    """
    Generate all trial-structured dynamics figures.
    
    Parameters
    ----------
    sim_data : dict
        - LFP: (R, T) array
        - Z_lat: (R, J_true, T) ground truth (optional)
        - freqs_hz: true signal frequencies (optional, used for GT comparison)
        - time: time array
    joint_results : dict
        - trace: dict with 'latent', 'latent_scale_factors', etc.
        - freqs_dense: analysis frequencies
    lfp_results : dict (optional)
        - Z_smooth or Z_smooth_full: (R, J, T) LFP-only estimates
    output_dir : str
    freqs_dense : array (optional)
        Full analysis frequency grid (default: from joint_results or [1,3,5,...,59])
    plot_freqs : sequence of float (optional)
        Specific frequencies to plot. If None, uses freqs_hz from sim_data
        (ground truth) or defaults to [5, 15, 25, 35] Hz.
    time_range : tuple
        Time range for plotting (start, end) in seconds
    sample_trials : list
        Trial indices to show in trial-specific plots
    
    Returns
    -------
    saved_files : dict of {name: path}
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = {}
    
    print("=" * 60)
    print("GENERATING TRIAL DYNAMICS FIGURES")
    print("=" * 60)
    
    # ==========================================================================
    # EXTRACT DATA
    # ==========================================================================
    
    LFP = sim_data['LFP']
    R, T = LFP.shape
    time = sim_data.get('time', np.linspace(0, T/fs, T))
    
    # Analysis frequencies (full grid)
    if freqs_dense is None:
        freqs_dense = joint_results.get('freqs_dense', np.arange(1, 61, 2, dtype=float))
    freqs_dense = np.asarray(freqs_dense, dtype=float)
    J = len(freqs_dense)
    
    # Ground truth frequencies (for comparison metrics, may be empty)
    freqs_gt = np.asarray(sim_data.get('freqs_hz', []), dtype=float)
    
    # Signal-only frequencies (no coupling, just LFP signal)
    freqs_extra = np.asarray(sim_data.get('freqs_hz_extra', []), dtype=float)
    
    # ==========================================================================
    # DETERMINE WHICH FREQUENCIES TO PLOT
    # ==========================================================================
    
    if plot_freqs is not None:
        # User specified custom frequencies
        freqs_to_plot = np.asarray(plot_freqs, dtype=float)
        print(f"  Using custom plot frequencies: {freqs_to_plot}")
    elif len(freqs_gt) > 0:
        # Use ground truth frequencies
        freqs_to_plot = freqs_gt
        print(f"  Using ground truth frequencies: {freqs_to_plot}")
    else:
        # Default: pick representative frequencies from the analysis grid
        default_freqs = [5.0, 15.0, 25.0, 35.0]
        freqs_to_plot = np.array([f for f in default_freqs if f <= freqs_dense.max()], dtype=float)
        if len(freqs_to_plot) == 0:
            # Fallback: use first 4 frequencies from dense grid
            freqs_to_plot = freqs_dense[:min(4, J)]
        print(f"  Using default plot frequencies: {freqs_to_plot}")
    
    # Map plot frequencies to dense grid indices
    idx_plot = [int(np.argmin(np.abs(freqs_dense - f))) for f in freqs_to_plot]
    
    # Also map GT frequencies to dense grid (for metrics/comparison)
    if len(freqs_gt) > 0:
        idx_gt_to_dense = [int(np.argmin(np.abs(freqs_dense - f))) for f in freqs_gt]
    else:
        idx_gt_to_dense = idx_plot  # Use plot indices as fallback
    
    print(f"  R={R} trials, T={T} time points, J={J} freqs")
    print(f"  Frequencies to plot: {freqs_to_plot}")
    print(f"  Plot freq -> dense idx: {dict(zip(freqs_to_plot, idx_plot))}")
    
    # ==========================================================================
    # GROUND TRUTH
    # ==========================================================================
    
    Z_gt = None
    has_gt = 'Z_lat' in sim_data and sim_data['Z_lat'] is not None
    if has_gt:
        Z_lat = sim_data['Z_lat']  # (R, J_true, T)
        if Z_lat.shape[-1] != T:
            Z_lat = resample_to_target(Z_lat, T)
        
        # Expand to full grid using GT frequency mapping
        Z_gt = np.zeros((R, J, T), dtype=complex)
        for j_gt, j_dense in enumerate(idx_gt_to_dense):
            if j_gt < Z_lat.shape[1]:
                Z_gt[:, j_dense, :] = Z_lat[:, j_gt, :]
        print(f"  Ground truth: {Z_gt.shape} (BASEBAND)")
    else:
        print("  Ground truth: NOT AVAILABLE")
    
    # ==========================================================================
    # MULTITAPER
    # ==========================================================================
    
    print("  Computing multitaper...")
    if HAS_MNE:
        tfr_raw = mne.time_frequency.tfr_array_multitaper(
            LFP[:, None, :], sfreq=fs, freqs=freqs_dense,
            n_cycles=freqs_dense * window_sec,
            time_bandwidth=2 * NW, output='complex', zero_mean=False,
        ).squeeze()  # (R, J, T)
        Z_mt = tfr_raw
        if Z_mt.shape[-1] != T:
            Z_mt = resample_to_target(Z_mt, T)
        print(f"  Z_mt: {Z_mt.shape}")
    else:
        print("  [WARN] MNE not available, using zeros for multitaper")
        Z_mt = np.zeros((R, J, T), dtype=complex)
    
    # ==========================================================================
    # CT-SSMT (LFP-ONLY)
    # ==========================================================================
    
    Z_lfp = None
    has_lfp = False
    if lfp_results is not None:
        # Try Z_smooth format first (legacy pickle format)
        if 'Z_smooth_full' in lfp_results:
            Z_lfp = lfp_results['Z_smooth_full']
        elif 'Z_smooth' in lfp_results:
            Z_lfp = lfp_results['Z_smooth']
        
        # Try X_fine + D_fine format (new npz format from runner_trials)
        elif 'X_fine' in lfp_results and 'D_fine' in lfp_results:
            print("  Loading LFP-only from X_fine + D_fine format")
            X_fine_lfp = lfp_results['X_fine']  # (T_ds, 2*J*M)
            D_fine_lfp = lfp_results['D_fine']  # (R, T_ds, 2*J*M)
            
            J_lfp = len(lfp_results.get('freqs', freqs_dense))
            M_lfp = 1  # Assume M=1 for trial data
            
            # Convert from interleaved format to complex
            X_complex = extract_complex_from_interleaved(X_fine_lfp, J_lfp, M_lfp)  # (J, T_ds)
            D_complex = extract_complex_from_interleaved(D_fine_lfp, J_lfp, M_lfp)  # (R, J, T_ds)
            
            # Z = X + D (already in baseband)
            Z_lfp = X_complex[None, :, :] + D_complex  # (R, J, T_ds)
            print(f"    Z_lfp shape before resampling: {Z_lfp.shape}")
        
        if Z_lfp is not None:
            if Z_lfp.shape[-1] != T:
                Z_lfp = resample_to_target(Z_lfp, T)
            has_lfp = True
            print(f"  Z_lfp (LFP-only): {Z_lfp.shape}")
    
    if not has_lfp:
        print("  CT-SSMT (LFP-only): NOT AVAILABLE")
    
    # ==========================================================================
    # CT-SSMT (JOINT) - Extract from X_fine + D_fine (preferred) or latent
    # ==========================================================================
    
    Z_spk = None
    Z_spk_var = None
    has_spk = False
    
    # Also extract LFP-only from EM estimates if available
    Z_lfp_from_em = None
    
    if 'trace' in joint_results:
        trace = joint_results['trace']
        freqs_joint = joint_results.get('freqs_dense', freqs_dense)
        J_joint = len(freqs_joint)
        
        # Determine number of tapers (M=1 typically for trial data)
        M = 1
        
        # PREFERRED: Use X_fine + D_fine (these are in BASEBAND, no derotation needed)
        X_key = 'X_fine_final' if 'X_fine_final' in trace else ('X_fine_avg' if 'X_fine_avg' in trace else None)
        D_key = 'D_fine_final' if 'D_fine_final' in trace else ('D_fine_avg' if 'D_fine_avg' in trace else None)
        
        if X_key and D_key and X_key in trace and D_key in trace:
            print(f"  Using {X_key} + {D_key} (BASEBAND, no derotation needed)")
            X_fine_raw = trace[X_key]
            D_fine_raw = trace[D_key]
            
            X_fine = np.asarray(X_fine_raw[-1] if isinstance(X_fine_raw, list) else X_fine_raw)
            D_fine = np.asarray(D_fine_raw[-1] if isinstance(D_fine_raw, list) else D_fine_raw)
            
            print(f"    X_fine: {X_fine.shape}, D_fine: {D_fine.shape}")
            
            # X_fine: (T_fine, 2*J*M) INTERLEAVED format
            # D_fine: (R, T_fine, 2*J*M) INTERLEAVED format
            # Convert to complex and sum: Z = X + D
            X_complex = extract_complex_from_interleaved(X_fine, J_joint, M)  # (J, T_fine)
            D_complex = extract_complex_from_interleaved(D_fine, J_joint, M)  # (R, J, T_fine)
            
            # Z = X (shared) + D (trial-specific) - already in BASEBAND
            Z_spk_fine = X_complex[None, :, :] + D_complex  # (R, J, T_fine)
            
            # Resample to display resolution
            Z_spk = resample_to_target(Z_spk_fine, T)
            has_spk = True
            print(f"  Z_spk (Joint from X+D): {Z_spk.shape}")
            
            # Also compute LFP-only as just X (shared component, averaged over trials)
            # Actually for LFP-only comparison, we want the full Z from EM (before spike refinement)
            # This should be stored separately, but if not, X is a reasonable approximation
            Z_lfp_from_em = resample_to_target(X_complex[None, :, :].repeat(R, axis=0), T)
            print(f"  Z_lfp (LFP-only from X): {Z_lfp_from_em.shape}")
            
            # Extract posterior variance
            X_var_key = 'X_var_fine_final' if 'X_var_fine_final' in trace else 'X_var_fine'
            D_var_key = 'D_var_fine_final' if 'D_var_fine_final' in trace else 'D_var_fine'
            
            if X_var_key in trace and D_var_key in trace:
                X_var_raw = trace[X_var_key]
                D_var_raw = trace[D_var_key]
                X_var_fine = np.asarray(X_var_raw[-1] if isinstance(X_var_raw, list) else X_var_raw)
                D_var_fine = np.asarray(D_var_raw[-1] if isinstance(D_var_raw, list) else D_var_raw)
                
                X_var_freq = extract_variance_from_interleaved(X_var_fine, J_joint, M)
                D_var_freq = extract_variance_from_interleaved(D_var_fine, J_joint, M)
                Z_spk_var = X_var_freq[None, :, :] + D_var_freq
                Z_spk_var = resample_to_target(Z_spk_var, T)
                print(f"  Z_spk_var (posterior variance): {Z_spk_var.shape}")
        
        # FALLBACK: Use latent (which is ROTATED and needs derotation)
        elif 'latent' in trace and len(trace['latent']) > 0:
            print("  Using trace['latent'] (ROTATED, needs derotation)")
            lat_reim = np.asarray(trace['latent'][-1])  # (R, T_fine, 2*J)
            R_lat, T_lat, twoJ = lat_reim.shape
            J_lat = twoJ // 2
            
            print(f"    lat_reim: {lat_reim.shape}")
            
            # Rescale if standardized
            if 'latent_scale_factors' in trace:
                scale_factors = np.asarray(trace['latent_scale_factors'])
                lat_reim = lat_reim * scale_factors[None, None, :]
                print(f"    Applied scale factors")
            
            # Convert to complex (SEPARATED format: [Re_0..Re_{J-1}, Im_0..Im_{J-1}])
            Z_joint_rotated = extract_complex_from_separated(lat_reim, J_lat)  # (R, J, T_fine)
            
            # CRITICAL: Derotate at FINE resolution FIRST, then resample
            # Use the same time grid that was used for rotation
            delta_spk = sim_data.get('delta_spk', 0.001)
            t_fine = np.arange(T_lat) * delta_spk
            Z_joint_baseband = derotate_tfr(Z_joint_rotated, freqs_joint[:J_lat], t_fine)
            
            # NOW resample to display resolution
            Z_spk = resample_to_target(Z_joint_baseband, T)
            has_spk = True
            print(f"  Z_spk (Joint from latent): {Z_spk.shape}")
    
    if not has_spk:
        print("  CT-SSMT (Joint): NOT AVAILABLE")
    
    # ==========================================================================
    # CT-SSMT (LFP-ONLY) - From lfp_results or from EM estimates in joint trace
    # ==========================================================================
    
    # If we computed LFP-only from EM above, use it
    if Z_lfp_from_em is not None and Z_lfp is None:
        # Actually, Z_lfp_from_em is just the shared X component
        # For true LFP-only, we'd need the EM estimates before spike refinement
        # For now, skip this to avoid confusion - LFP-only should come from lfp_results
        pass
    
    # ==========================================================================
    # COMPUTE GLOBAL SCALES
    # ==========================================================================
    
    # Use GT indices for scaling if available, otherwise use plot indices
    idx_for_scale = idx_gt_to_dense if has_gt else idx_plot
    
    if has_gt:
        scale_mt = compute_global_scale(Z_mt, Z_gt, idx_for_scale)
        scale_lfp = compute_global_scale(Z_lfp, Z_gt, idx_for_scale) if has_lfp else 1.0
        scale_spk = compute_global_scale(Z_spk, Z_gt, idx_for_scale) if has_spk else 1.0
        print(f"  Global scales: MT={scale_mt:.4f}, LFP={scale_lfp:.4f}, Joint={scale_spk:.4f}")
    else:
        scale_mt = scale_lfp = scale_spk = 1.0
    
    # ==========================================================================
    # GENERATE FIGURES
    # ==========================================================================
    
    print("\n  Generating figures...")
    
    # Fig 1: Trial-specific comparison
    print("    [1/9] trial_specific_comparison")
    saved_files['trial_specific'] = plot_trial_specific_comparison(
        Z_gt, Z_mt, Z_lfp if has_lfp else None, Z_spk if has_spk else None,
        freqs_dense, time,
        os.path.join(output_dir, 'trial_specific_comparison.png'),
        idx_sig=idx_plot, freqs_true=freqs_to_plot,
        sample_trials=sample_trials, time_range=time_range,
        Z_spk_var=Z_spk_var,
        scale_mt=scale_mt, scale_lfp=scale_lfp, scale_spk=scale_spk,
    )
    
    # Fig 2: Deviation comparison
    print("    [2/9] deviation_comparison")
    saved_files['deviation'] = plot_deviation_comparison(
        Z_gt, Z_mt, Z_lfp if has_lfp else None, Z_spk if has_spk else None,
        freqs_dense, time,
        os.path.join(output_dir, 'deviation_comparison.png'),
        idx_sig=idx_plot, freqs_true=freqs_to_plot,
        sample_trials=sample_trials, time_range=time_range,
        Z_spk_var=Z_spk_var,
    )
    
    # Fig 3: Trial-averaged
    print("    [3/9] trial_averaged_comparison")
    saved_files['trial_averaged'] = plot_trial_averaged_comparison(
        Z_gt, Z_mt, Z_lfp if has_lfp else None, Z_spk if has_spk else None,
        freqs_dense, time,
        os.path.join(output_dir, 'trial_averaged_comparison.png'),
        idx_sig=idx_plot, freqs_true=freqs_to_plot,
        freqs_extra=freqs_extra if len(freqs_extra) > 0 else None,
        time_range=time_range,
    )
    
    # Fig 4: Spectrogram trial-specific
    print("    [4/9] spectrogram_trial_specific")
    saved_files['spec_trial'] = plot_spectrogram_trial_specific(
        Z_gt, Z_mt, Z_lfp if has_lfp else None, Z_spk if has_spk else None,
        freqs_dense, time,
        os.path.join(output_dir, 'spectrogram_trial_specific.png'),
        freqs_true=freqs_to_plot, freqs_extra=freqs_extra if len(freqs_extra) > 0 else None,
        sample_trials=sample_trials[:3], time_range=time_range,
    )
    
    # Fig 5: Spectrogram trial-averaged
    print("    [5/9] spectrogram_trial_averaged")
    saved_files['spec_avg'] = plot_spectrogram_trial_averaged(
        Z_gt, Z_mt, Z_lfp if has_lfp else None, Z_spk if has_spk else None,
        freqs_dense, time,
        os.path.join(output_dir, 'spectrogram_trial_averaged.png'),
        freqs_true=freqs_to_plot, freqs_extra=freqs_extra if len(freqs_extra) > 0 else None,
        time_range=time_range,
        scale_mt=scale_mt, scale_lfp=scale_lfp, scale_spk=scale_spk,
    )
    
    # Fig 6: Spectrogram deviation
    print("    [6/9] spectrogram_deviation")
    saved_files['spec_dev'] = plot_spectrogram_deviation(
        Z_gt, Z_mt, Z_lfp if has_lfp else None, Z_spk if has_spk else None,
        freqs_dense, time,
        os.path.join(output_dir, 'spectrogram_deviation.png'),
        freqs_true=freqs_to_plot, sample_trials=sample_trials[:3], time_range=time_range,
        scale_mt=scale_mt, scale_lfp=scale_lfp, scale_spk=scale_spk,
    )
    
    # Fig 7: Correlation boxplot (requires GT)
    if has_gt:
        print("    [7/9] correlation_boxplot")
        saved_files['corr_boxplot'] = plot_correlation_boxplot(
            Z_gt, Z_mt, Z_lfp if has_lfp else None, Z_spk if has_spk else None,
            freqs_dense, time,
            os.path.join(output_dir, 'correlation_boxplot.png'),
            idx_sig=idx_gt_to_dense, freqs_true=freqs_gt,
            time_range=time_range,
        )
    else:
        print("    [7/9] correlation_boxplot SKIPPED (no GT)")
    
    # Fig 8: PSD comparison
    print("    [8/9] psd_comparison")
    saved_files['psd'] = plot_psd_comparison(
        Z_mt, Z_lfp if has_lfp else None, Z_spk if has_spk else None,
        freqs_dense,
        os.path.join(output_dir, 'psd_comparison.png'),
        idx_sig=idx_plot,
        scale_mt=scale_mt, scale_lfp=scale_lfp, scale_spk=scale_spk,
    )
    
    # Summary metrics
    print("\n  " + "=" * 50)
    print("  AGGREGATE AMPLITUDE CORRELATIONS")
    print("  " + "=" * 50)
    
    if has_gt:
        for j_gt, (freq_hz, j_dense) in enumerate(zip(freqs_gt, idx_gt_to_dense)):
            gt_flat = np.abs(Z_gt[:, j_dense, :]).flatten()
            r_mt = np.corrcoef(gt_flat, np.abs(Z_mt[:, j_dense, :]).flatten())[0, 1]
            r_lfp = np.corrcoef(gt_flat, np.abs(Z_lfp[:, j_dense, :]).flatten())[0, 1] if has_lfp else np.nan
            r_spk = np.corrcoef(gt_flat, np.abs(Z_spk[:, j_dense, :]).flatten())[0, 1] if has_spk else np.nan
            print(f"  {freq_hz:.0f} Hz: MT={r_mt:.3f}, LFP={r_lfp:.3f}, Joint={r_spk:.3f}")
        
        gt_all = np.abs(Z_gt[:, idx_gt_to_dense, :]).flatten()
        mt_all = np.abs(Z_mt[:, idx_gt_to_dense, :]).flatten()
        r_mt_all = np.corrcoef(gt_all, mt_all)[0, 1]
        r_lfp_all = np.corrcoef(gt_all, np.abs(Z_lfp[:, idx_gt_to_dense, :]).flatten())[0, 1] if has_lfp else np.nan
        r_spk_all = np.corrcoef(gt_all, np.abs(Z_spk[:, idx_gt_to_dense, :]).flatten())[0, 1] if has_spk else np.nan
        print("-" * 50)
        print(f"  OVERALL: MT={r_mt_all:.3f}, LFP={r_lfp_all:.3f}, Joint={r_spk_all:.3f}")
    else:
        print("  (No ground truth - skipping correlation metrics)")
    
    print("\n  Done!")
    return saved_files