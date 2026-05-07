"""
Spectral Dynamics Comparison Plots (Single-Trial)

Ground-truth agnostic: All plots work without ground truth.
When ground truth is available, additional panels and correlations are shown.

For trial-structured data, see spectral_dynamics_trials.py

Generates:
- spectrogram_comparison.png (GT panel only if available)
- correlation_over_time.png (only if GT available)
- correlation_boxplot.png (only if GT available)
- correlation_heatmaps.png (only if GT available)
- timeseries_snapshots.png (works without GT)
- method_comparison.png (always works - compares MT, LFP-only, Joint)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import pearsonr, norm
from typing import Optional, Dict, Tuple, List

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import mne
    from mne.time_frequency.multitaper import dpss_windows
    HAS_MNE = True
except ImportError:
    HAS_MNE = False


# =============================================================================
# FIGURE STYLE + COLORS
# =============================================================================

def set_style(font_size=7):
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

# Aliases
METHOD_CONFIG["multitaper"] = METHOD_CONFIG["mt"]
METHOD_CONFIG["lfp_only"] = METHOD_CONFIG["lfp"]
METHOD_CONFIG["joint"] = METHOD_CONFIG["spk"]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def db_scale(power, ref=None):
    if ref is None:
        ref = np.nanmean(power)
    return 10 * np.log10(power / ref + 1e-10)


def find_optimal_scale(gt, est):
    """Find alpha such that alpha*est best matches gt (least squares)."""
    valid = (gt > 0) & (est > 0) & ~np.isnan(gt) & ~np.isnan(est)
    if valid.sum() < 10:
        return 1.0
    gt_valid = gt[valid]
    est_valid = est[valid]
    alpha = np.sum(gt_valid * est_valid) / (np.sum(est_valid**2) + 1e-10)
    return alpha


def fine_to_amplitude_JT(Z_fine, J, M):
    """Convert fine state (1, T, 2*J*M) to amplitude (J, T)."""
    T = Z_fine.shape[1]
    amplitude = np.zeros((J, T))

    for j in range(J):
        amp_tapers = np.zeros((M, T))
        for m in range(M):
            col_re = 2 * (j * M + m)
            col_im = col_re + 1
            amp_tapers[m, :] = np.sqrt(Z_fine[0, :, col_re]**2 + Z_fine[0, :, col_im]**2)
        amplitude[j, :] = amp_tapers.mean(axis=0)

    return amplitude


def fine_to_amplitude_with_uncertainty(Z_fine, Z_var_fine, J, M):
    """Convert fine state to amplitude with uncertainty."""
    T = Z_fine.shape[1]
    amplitude = np.zeros((J, T))
    amplitude_var = np.zeros((J, T))

    for j in range(J):
        amp_tapers = np.zeros((M, T))
        var_tapers = np.zeros((M, T))

        for m in range(M):
            col_re = 2 * (j * M + m)
            col_im = col_re + 1

            x = Z_fine[0, :, col_re]
            y = Z_fine[0, :, col_im]
            var_x = Z_var_fine[0, :, col_re]
            var_y = Z_var_fine[0, :, col_im]

            amp = np.sqrt(x**2 + y**2)
            amp_tapers[m, :] = amp

            denom = x**2 + y**2 + 1e-10
            var_amp = (x**2 * var_x + y**2 * var_y) / denom
            var_tapers[m, :] = var_amp

        amplitude[j, :] = amp_tapers.mean(axis=0)
        amplitude_var[j, :] = var_tapers.mean(axis=0) / M

    amplitude_std = np.sqrt(amplitude_var)
    return amplitude, amplitude_std


def downsample_to_blocks(data_JT, block_size):
    """Downsample (J, T) to (J, K) by averaging over blocks."""
    J, T = data_JT.shape
    K = T // block_size
    data_JK = np.zeros((J, K))
    for k in range(K):
        start = k * block_size
        end = start + block_size
        data_JK[:, k] = data_JT[:, start:end].mean(axis=1)
    return data_JK


# =============================================================================
# SPECTROGRAM COMPARISON
# =============================================================================

def plot_spectrogram_comparison(
    amp_gt_fine: Optional[np.ndarray],
    amp_mt_scaled: np.ndarray,
    amp_lfp_scaled: np.ndarray,
    amp_joint_scaled: np.ndarray,
    freqs: np.ndarray,
    duration: float,
    block_samples: int,
    output_path: str,
    freqs_coupled: Optional[np.ndarray] = None,
    freqs_all_signal: Optional[np.ndarray] = None,
) -> str:
    """
    Plot spectrogram comparison.
    If amp_gt_fine is None, only shows 3 panels (MT, LFP-only, Joint).
    """
    set_style(font_size=8)
    
    has_gt = amp_gt_fine is not None and amp_gt_fine.max() > 0
    n_panels = 4 if has_gt else 3
    
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 2.5 * n_panels), sharex=True, sharey=True)
    
    def to_db(power):
        return 10 * np.log10(power + 1e-10)
    
    power_mt_block = downsample_to_blocks(amp_mt_scaled**2, block_samples)
    power_lfp_block = downsample_to_blocks(amp_lfp_scaled**2, block_samples)
    power_joint_block = downsample_to_blocks(amp_joint_scaled**2, block_samples)
    
    db_mt = to_db(power_mt_block)
    db_lfp = to_db(power_lfp_block)
    db_joint = to_db(power_joint_block)
    
    extent = [0, duration, freqs[0], freqs[-1]]
    
    if has_gt:
        power_gt_block = downsample_to_blocks(amp_gt_fine**2, block_samples)
        db_gt = to_db(power_gt_block)
        db_gt[power_gt_block == 0] = np.nan
        
        panels = [
            ('gt', db_gt, 'Ground Truth'),
            ('mt', db_mt, METHOD_CONFIG['mt']['label']),
            ('lfp', db_lfp, METHOD_CONFIG['lfp']['label']),
            ('spk', db_joint, METHOD_CONFIG['spk']['label']),
        ]
    else:
        panels = [
            ('mt', db_mt, METHOD_CONFIG['mt']['label']),
            ('lfp', db_lfp, METHOD_CONFIG['lfp']['label']),
            ('spk', db_joint, METHOD_CONFIG['spk']['label']),
        ]
    
    for ax, (key, db_power, title) in zip(axes, panels):
        valid = db_power[~np.isnan(db_power)]
        if len(valid) > 0:
            vmax = np.percentile(valid, 100)
            vmin = vmax - 40
        else:
            vmin, vmax = -50, 50

        im = ax.imshow(
            db_power,
            aspect='auto',
            origin='lower',
            extent=extent,
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            interpolation='none',
        )
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(title, fontsize=10, fontweight='bold')

        if freqs_coupled is not None:
            for f in freqs_coupled:
                ax.axhline(f, color='white', linestyle='--', lw=1.0, alpha=0.8)
        
        if freqs_all_signal is not None and freqs_coupled is not None:
            freqs_extra = np.array([f for f in freqs_all_signal 
                                   if not np.any(np.isclose(freqs_coupled, f))], dtype=float)
            for f in freqs_extra:
                ax.axhline(f, color='white', linestyle=':', lw=1.0, alpha=0.9)

    axes[-1].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


# =============================================================================
# METHOD COMPARISON (ALWAYS WORKS)
# =============================================================================

def plot_method_comparison(
    amp_mt: np.ndarray,
    amp_lfp: np.ndarray,
    amp_joint: np.ndarray,
    freqs: np.ndarray,
    delta_spk: float,
    output_path: str,
    idx_sig: Optional[List[int]] = None,
    n_snapshots: int = 4,
    snapshot_sec: float = 10.0,
) -> str:
    """
    Plot method comparison (MT vs LFP-only vs Joint).
    Works WITHOUT ground truth.
    """
    set_style(font_size=8)
    
    J, T = amp_mt.shape
    
    if idx_sig is None or len(idx_sig) == 0:
        idx_sig = list(np.linspace(0, J-1, min(4, J), dtype=int))
    
    n_freqs = len(idx_sig)
    snapshot_samples = int(snapshot_sec / delta_spk)
    
    if T <= snapshot_samples:
        n_snapshots = 1
        snapshot_starts = [0]
    else:
        total_samples = T - snapshot_samples
        snapshot_starts = np.linspace(0, total_samples, n_snapshots, dtype=int)
    
    fig, axes = plt.subplots(n_freqs, n_snapshots, figsize=(4 * n_snapshots, 3 * n_freqs), squeeze=False)
    
    for col, start_sample in enumerate(snapshot_starts):
        end_sample = min(start_sample + snapshot_samples, T)
        t_local = np.arange(end_sample - start_sample) * delta_spk
        
        for row, j in enumerate(idx_sig):
            ax = axes[row, col]
            
            ax.plot(t_local, amp_mt[j, start_sample:end_sample],
                   color=METHOD_CONFIG['mt']['color'], lw=1.0, alpha=0.6,
                   label=METHOD_CONFIG['mt']['label'])
            
            ax.plot(t_local, amp_lfp[j, start_sample:end_sample],
                   color=METHOD_CONFIG['lfp']['color'], lw=1.5,
                   label=METHOD_CONFIG['lfp']['label'])
            
            ax.plot(t_local, amp_joint[j, start_sample:end_sample],
                   color=METHOD_CONFIG['spk']['color'], lw=1.5,
                   label=METHOD_CONFIG['spk']['label'])
            
            ax.set_xlim(0, snapshot_sec)
            ax.set_ylim(0, None)
            
            if row == 0:
                start_sec = start_sample * delta_spk
                ax.set_title(f't = {start_sec:.0f}-{start_sec + snapshot_sec:.0f} s', fontweight='bold')
            if row == n_freqs - 1:
                ax.set_xlabel('Time (s)')
            if col == 0:
                ax.set_ylabel(f'{freqs[j]:.0f} Hz\nAmplitude')
            
            if row == 0 and col == 0:
                ax.legend(loc='upper right', fontsize=6)
    
    fig.suptitle('Method Comparison (No Ground Truth)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


# =============================================================================
# TIME SERIES SNAPSHOTS (GT OPTIONAL)
# =============================================================================

def plot_timeseries_snapshots(
    amp_gt_fine: Optional[np.ndarray],
    amp_mt_scaled: np.ndarray,
    amp_lfp_scaled: np.ndarray,
    amp_joint_scaled: np.ndarray,
    freqs: np.ndarray,
    idx_sig: List[int],
    delta_spk: float,
    output_path: str,
    amp_joint_std_scaled: Optional[np.ndarray] = None,
    n_snapshots: int = 4,
    snapshot_sec: float = 10.0,
    ci_level: float = 0.95,
) -> str:
    """
    Plot time series snapshots with optional uncertainty bands.
    Works without ground truth (just won't show GT line).
    """
    set_style(font_size=8)
    
    has_gt = amp_gt_fine is not None and amp_gt_fine.max() > 0
    ci_mult = norm.ppf((1 + ci_level) / 2)
    
    J, T_fine = amp_mt_scaled.shape
    snapshot_samples = int(snapshot_sec / delta_spk)
    n_signals = len(idx_sig)

    if n_signals == 0 or T_fine <= snapshot_samples:
        print("  [SKIP] timeseries_snapshots: insufficient data")
        return output_path

    total_samples = T_fine - snapshot_samples
    if n_snapshots > 1:
        snapshot_starts = np.linspace(0, total_samples, n_snapshots, dtype=int)
    else:
        snapshot_starts = [total_samples // 2]

    fig, axes = plt.subplots(
        n_signals, n_snapshots,
        figsize=(4 * n_snapshots, 3 * n_signals),
        squeeze=False
    )

    for col, start_sample in enumerate(snapshot_starts):
        end_sample = start_sample + snapshot_samples
        t_local = np.arange(snapshot_samples) * delta_spk

        for row, j in enumerate(idx_sig):
            ax = axes[row, col]

            if has_gt:
                gt = amp_gt_fine[j, start_sample:end_sample]
                if gt.max() > 0:
                    ax.plot(t_local, gt,
                           color=METHOD_CONFIG["gt"]["color"],
                           lw=METHOD_CONFIG["gt"]["linewidth"],
                           label=METHOD_CONFIG["gt"]["label"], alpha=0.9)

            ax.plot(t_local, amp_mt_scaled[j, start_sample:end_sample],
                   color=METHOD_CONFIG['mt']['color'], lw=1.0, alpha=0.5,
                   label=METHOD_CONFIG['mt']['label'])

            ax.plot(t_local, amp_lfp_scaled[j, start_sample:end_sample],
                   color=METHOD_CONFIG['lfp']['color'], lw=1.5,
                   label=METHOD_CONFIG['lfp']['label'])

            joint_mean = amp_joint_scaled[j, start_sample:end_sample]
            ax.plot(t_local, joint_mean,
                   color=METHOD_CONFIG['spk']['color'], lw=1.5,
                   label=METHOD_CONFIG['spk']['label'])

            if amp_joint_std_scaled is not None:
                joint_std = amp_joint_std_scaled[j, start_sample:end_sample]
                lower = np.maximum(joint_mean - ci_mult * joint_std, 0)
                upper = joint_mean + ci_mult * joint_std
                ax.fill_between(t_local, lower, upper,
                               color=METHOD_CONFIG['spk']['color'],
                               alpha=0.2, label=f'{int(ci_level*100)}% CI')

            ax.set_xlim(0, snapshot_sec)
            ax.set_ylim(0, None)

            if row == 0:
                start_sec = start_sample * delta_spk
                ax.set_title(f't = {start_sec:.0f}-{start_sec + snapshot_sec:.0f} s', fontweight='bold')
            if row == n_signals - 1:
                ax.set_xlabel('Time (s)')
            if col == 0:
                ax.set_ylabel(f'{freqs[j]:.0f} Hz\nAmplitude')

            if row == 0 and col == 0:
                ax.legend(loc='upper right', fontsize=6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


# =============================================================================
# CORRELATION PLOTS (REQUIRE GT)
# =============================================================================

def compute_correlations_over_time(
    amp_gt_fine: np.ndarray,
    method_data: Dict[str, np.ndarray],
    idx_sig: List[int],
    fs_fine: float,
    time_window: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
    """Compute correlations with ground truth over time windows."""
    T_fine = amp_gt_fine.shape[1]
    time_bins = int(time_window * fs_fine)
    total_duration = T_fine / fs_fine
    n_windows = int(total_duration / time_window)
    time_centers = np.arange(n_windows) * time_window + time_window / 2

    methods = list(method_data.keys())
    correlations = {m: np.zeros((len(idx_sig), n_windows)) for m in methods}
    correlation_pvals = {m: np.zeros((len(idx_sig), n_windows)) for m in methods}

    for win_idx in range(n_windows):
        start_sample = int(win_idx * time_bins)
        end_sample = int(start_sample + time_bins)
        if end_sample > T_fine:
            break

        for freq_idx, j in enumerate(idx_sig):
            gt_power = amp_gt_fine[j, start_sample:end_sample] ** 2
            if gt_power.std() < 1e-10:
                continue

            for method in methods:
                est_power = method_data[method][j, start_sample:end_sample] ** 2
                if est_power.std() < 1e-10:
                    continue
                try:
                    corr, pval = pearsonr(gt_power, est_power)
                    correlations[method][freq_idx, win_idx] = corr
                    correlation_pvals[method][freq_idx, win_idx] = pval
                except Exception:
                    pass

    return correlations, correlation_pvals, time_centers


def plot_correlation_over_time(
    correlations: Dict[str, np.ndarray],
    correlation_pvals: Dict[str, np.ndarray],
    time_centers: np.ndarray,
    freqs: np.ndarray,
    idx_sig: List[int],
    time_window: float,
    output_path: str,
) -> str:
    """Plot correlation with GT over time (REQUIRES GT)."""
    set_style(font_size=8)
    
    methods = list(correlations.keys())
    
    fig, axes = plt.subplots(len(idx_sig), 1, figsize=(12, 3*len(idx_sig)), sharex=True)
    if len(idx_sig) == 1:
        axes = [axes]

    for freq_idx, (ax, j) in enumerate(zip(axes, idx_sig)):
        freq_val = freqs[j]

        for method in methods:
            config = METHOD_CONFIG.get(method, {"label": method, "color": "gray", "linewidth": 1.5})
            corr_values = correlations[method][freq_idx, :]

            ax.plot(time_centers, corr_values,
                   color=config['color'], lw=config['linewidth'],
                   marker='o', markersize=3, label=config['label'], alpha=0.8)

        ax.set_ylabel('Correlation with GT')
        ax.set_title(f'Frequency: {freq_val:.1f} Hz', fontsize=10, fontweight='bold')
        ax.set_ylim([-1, 1])
        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)

        if freq_idx == 0:
            ax.legend(loc='upper right', fontsize='small')

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Correlation with Ground Truth (window = {time_window}s)', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_correlation_boxplot(
    correlations: Dict[str, np.ndarray],
    freqs: np.ndarray,
    idx_sig: List[int],
    time_window: float,
    output_path: str,
) -> str:
    """Plot correlation boxplot (REQUIRES GT)."""
    set_style(font_size=8)
    
    methods = list(correlations.keys())
    n_windows = correlations[methods[0]].shape[1]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    box_data = []
    box_positions = []
    box_colors = []
    tick_labels = []

    x_pos = 0
    freq_spacing = len(methods) + 1

    for freq_idx, j in enumerate(idx_sig):
        freq_val = freqs[j]

        for i, method in enumerate(methods):
            box_data.append(correlations[method][freq_idx, :])
            box_positions.append(x_pos + i)
            config = METHOD_CONFIG.get(method, {"color": "gray"})
            box_colors.append(config['color'])

        tick_labels.append(f'{freq_val:.1f} Hz')
        x_pos += freq_spacing

    bp = ax.boxplot(box_data, positions=box_positions, widths=0.6, patch_artist=True)

    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks([i * freq_spacing + (len(methods)-1)/2 for i in range(len(idx_sig))])
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel('Correlation with Ground Truth')
    ax.set_xlabel('Signal Frequency')
    ax.set_title(f'Correlation Distribution ({time_window}s windows)', fontsize=10, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1,
                      facecolor=METHOD_CONFIG.get(m, {"color": "gray"})['color'],
                      alpha=0.7, label=METHOD_CONFIG.get(m, {"label": m})['label'])
        for m in methods
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def generate_spectral_dynamics_figures(
    sim_data: dict,
    joint_results: dict,
    output_dir: str,
    window_sec: float = 20.0,
    n_snapshots: int = 4,
    snapshot_sec: float = 10.0,
    ci_level: float = 0.95,
) -> Dict[str, str]:
    """
    Generate all spectral dynamics figures.
    
    Works WITHOUT ground truth - GT-dependent plots are skipped.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = {}
    
    has_gt = 'A_t' in sim_data and sim_data['A_t'] is not None
    
    freqs_coupled = np.asarray(sim_data.get('freqs_hz_coupled', sim_data.get('freqs_hz', [])), float)
    freqs_all_signal = np.asarray(sim_data.get('freqs_hz', []), float)
    fs = sim_data.get('fs', 1000.0)
    delta_spk = sim_data.get('delta_spk', 0.001)
    lfp = sim_data['LFP']
    duration = len(lfp) / fs
    
    freqs = np.asarray(joint_results['freqs'], dtype=float)
    window_sec_res = joint_results['window_sec']
    J = len(freqs)
    M = joint_results.get('n_tapers', 3)
    NW = joint_results.get('NW', 2.0)
    
    T_fine = int(duration / delta_spk)
    block_samples = int(window_sec_res / delta_spk)
    fs_fine = 1.0 / delta_spk
    
    print(f"  Duration: {duration:.1f}s, J: {J}, M: {M}")
    print(f"  Ground truth available: {has_gt}")
    
    if has_gt:
        A_gt = sim_data['A_t']
        freqs_signal = sim_data['freqs_hz']
        
        amp_gt_fine = np.zeros((J, T_fine))
        signal_freq_to_j = {}
        for i, f_sig in enumerate(freqs_signal):
            j_idx = np.argmin(np.abs(freqs - f_sig))
            if np.abs(freqs[j_idx] - f_sig) < 1.5 and i < A_gt.shape[0]:
                amp_gt_fine[j_idx, :] = A_gt[i, :T_fine]
                signal_freq_to_j[f_sig] = j_idx
        
        idx_sig = list(signal_freq_to_j.values())
    else:
        amp_gt_fine = None
        idx_sig = list(range(min(4, J)))
    
    if 'Z_fine_em' in joint_results and joint_results['Z_fine_em'] is not None:
        Z_fine_em = joint_results['Z_fine_em']
        amp_lfp_fine = fine_to_amplitude_JT(Z_fine_em, J, M)
    else:
        amp_lfp_fine = np.zeros((J, T_fine))
    
    if 'Z_fine_joint' in joint_results and joint_results['Z_fine_joint'] is not None:
        Z_fine_joint = joint_results['Z_fine_joint']
        Z_var_fine_joint = joint_results.get('Z_var_fine_joint', None)
        
        if Z_var_fine_joint is not None:
            amp_joint_fine, amp_joint_std = fine_to_amplitude_with_uncertainty(
                Z_fine_joint, Z_var_fine_joint, J, M
            )
        else:
            amp_joint_fine = fine_to_amplitude_JT(Z_fine_joint, J, M)
            amp_joint_std = None
    else:
        amp_joint_fine = np.zeros((J, T_fine))
        amp_joint_std = None
    
    if HAS_MNE:
        print("  Computing multitaper...")
        n_tapers = int(2 * NW - 1)
        
        tfr_raw = mne.time_frequency.tfr_array_multitaper(
            lfp[None, None, :],
            sfreq=fs,
            freqs=freqs,
            n_cycles=freqs * window_sec_res,
            time_bandwidth=2 * NW,
            output='complex',
            zero_mean=False,
        )
        
        if tfr_raw.ndim == 5:
            tfr = tfr_raw[0, 0, :, :, :].transpose(1, 0, 2)
        else:
            tfr = tfr_raw[0, 0, :, :][:, None, :]
        
        amp_mt_fine = np.abs(tfr).mean(axis=1)
    else:
        amp_mt_fine = np.zeros((J, T_fine))
    
    T_min = min(amp_lfp_fine.shape[1], amp_joint_fine.shape[1], amp_mt_fine.shape[1])
    if has_gt:
        T_min = min(T_min, amp_gt_fine.shape[1])
        amp_gt_fine = amp_gt_fine[:, :T_min]
    amp_lfp_fine = amp_lfp_fine[:, :T_min]
    amp_joint_fine = amp_joint_fine[:, :T_min]
    amp_mt_fine = amp_mt_fine[:, :T_min]
    if amp_joint_std is not None:
        amp_joint_std = amp_joint_std[:, :T_min]
    T_fine = T_min
    
    if has_gt and len(idx_sig) > 0:
        gt_flat = amp_gt_fine[idx_sig, :].flatten()
        scale_mt = find_optimal_scale(gt_flat, amp_mt_fine[idx_sig, :].flatten())
        scale_lfp = find_optimal_scale(gt_flat, amp_lfp_fine[idx_sig, :].flatten())
        scale_joint = find_optimal_scale(gt_flat, amp_joint_fine[idx_sig, :].flatten())
    else:
        mt_std = amp_mt_fine.std() + 1e-10
        scale_mt = 1.0
        scale_lfp = mt_std / (amp_lfp_fine.std() + 1e-10)
        scale_joint = mt_std / (amp_joint_fine.std() + 1e-10)
    
    print(f"  Scales: MT={scale_mt:.3f}, LFP={scale_lfp:.3f}, Joint={scale_joint:.3f}")
    
    amp_mt_scaled = amp_mt_fine * scale_mt
    amp_lfp_scaled = amp_lfp_fine * scale_lfp
    amp_joint_scaled = amp_joint_fine * scale_joint
    amp_joint_std_scaled = amp_joint_std * scale_joint if amp_joint_std is not None else None
    
    print("  Generating spectrogram comparison...")
    saved_files['spectrogram_comparison'] = plot_spectrogram_comparison(
        amp_gt_fine, amp_mt_scaled, amp_lfp_scaled, amp_joint_scaled,
        freqs, duration, block_samples,
        os.path.join(output_dir, 'spectrogram_comparison.png'),
        freqs_coupled=freqs_coupled if len(freqs_coupled) > 0 else None,
        freqs_all_signal=freqs_all_signal if len(freqs_all_signal) > 0 else None,
    )
    
    print("  Generating time series snapshots...")
    saved_files['timeseries_snapshots'] = plot_timeseries_snapshots(
        amp_gt_fine, amp_mt_scaled, amp_lfp_scaled, amp_joint_scaled,
        freqs, idx_sig, delta_spk,
        os.path.join(output_dir, 'timeseries_snapshots.png'),
        amp_joint_std_scaled=amp_joint_std_scaled,
        n_snapshots=n_snapshots,
        snapshot_sec=snapshot_sec,
        ci_level=ci_level,
    )
    
    if has_gt:
        print(f"  Computing correlations (window = {window_sec}s)...")
        method_data = {'mt': amp_mt_scaled, 'lfp': amp_lfp_scaled, 'spk': amp_joint_scaled}
        
        correlations, correlation_pvals, time_centers = compute_correlations_over_time(
            amp_gt_fine, method_data, idx_sig, fs_fine, window_sec
        )
        
        print("  Generating correlation over time...")
        saved_files['correlation_over_time'] = plot_correlation_over_time(
            correlations, correlation_pvals, time_centers, freqs, idx_sig, window_sec,
            os.path.join(output_dir, 'correlation_over_time.png')
        )
        
        print("  Generating correlation boxplot...")
        saved_files['correlation_boxplot'] = plot_correlation_boxplot(
            correlations, freqs, idx_sig, window_sec,
            os.path.join(output_dir, 'correlation_boxplot.png')
        )
    else:
        print("  [SKIP] Correlation plots: no ground truth available")
    
    return saved_files