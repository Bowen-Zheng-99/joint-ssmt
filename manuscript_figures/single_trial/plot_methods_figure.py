#!/usr/bin/env python3
"""
Generate panels for the methods figure.

Data processing: Copied EXACTLY from src/plotting/spectral_dynamics.py
Plotting: Adapted for publication layout sizes

Layout (7" × 4.6"):
┌────────────────────────────────────────────────────────────┐
│                    (a) Raw Data - 7.0" × 1.3"              │
│                    LFP + S spike rasters + Δ_b, Δ_spk      │
├────────────┬───────────────────────────────────────────────┤
│    (b)     │        (c) Spectrogram - 5.7" × 1.8"          │
│  1.1"×1.8" │        4 rows: GT / MT / LFP-only / Joint     │
├────────────┴───────────────────────────┬───────────────────┤
│       (d) Snapshots - 5.0" × 1.5"      │(e) Corr 1.8"×1.5" │
└────────────────────────────────────────┴───────────────────┘
"""

import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr, norm
from typing import Optional, Dict, List, Tuple

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False


# =============================================================================
# STYLE + COLORS (copied exactly from spectral_dynamics.py)
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


# Copied exactly from spectral_dynamics.py
METHOD_CONFIG = {
    "mt":  {"label": "Multi-taper",        "color": "#2E86AB", "linestyle": "-", "linewidth": 1.5},
    "lfp": {"label": "CT-SSMT (LFP-only)", "color": "#A23B72", "linestyle": "-", "linewidth": 1.5},
    "spk": {"label": "Joint SSMT",    "color": "#F18F01", "linestyle": "-", "linewidth": 1.5},
    "gt":  {"label": "Ground Truth",       "color": "#333333", "linestyle": "-", "linewidth": 1.0},
}
METHOD_CONFIG["multitaper"] = METHOD_CONFIG["mt"]
METHOD_CONFIG["lfp_only"] = METHOD_CONFIG["lfp"]
METHOD_CONFIG["joint"] = METHOD_CONFIG["spk"]

# Colors from panel_a_data.py
PANEL_A_COLORS = {
    'lfp': '#2E86AB',
    'latent': '#7B2D8E',
    'spike': '#E94F37',
}


# =============================================================================
# HELPER FUNCTIONS (copied exactly from spectral_dynamics.py)
# =============================================================================

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


def compute_correlations_over_time(
    amp_gt_fine: np.ndarray,
    method_data: Dict[str, np.ndarray],
    idx_sig: List[int],
    fs_fine: float,
    time_window: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
    """
    Compute correlations with ground truth over non-overlapping time windows.
    
    For each window, compute Pearson correlation between GT power and estimated power.
    Collect all correlations across windows for boxplot.
    
    Parameters
    ----------
    amp_gt_fine : (J, T_fine) ground truth amplitude
    method_data : dict mapping method name -> (J, T_fine) estimated amplitude
    idx_sig : list of frequency indices to analyze
    fs_fine : fine sampling rate (1/delta_spk)
    time_window : window size in seconds (e.g., 20s)
    
    Returns
    -------
    correlations : dict mapping method -> (n_freqs, n_windows) correlation values
    correlation_pvals : dict mapping method -> (n_freqs, n_windows) p-values
    time_centers : (n_windows,) center time of each window
    """
    J, T_fine = amp_gt_fine.shape
    time_bins = int(time_window * fs_fine)
    total_duration = T_fine / fs_fine
    n_windows = int(total_duration / time_window)
    time_centers = np.arange(n_windows) * time_window + time_window / 2
    
    print(f"    compute_correlations_over_time:")
    print(f"      T_fine={T_fine}, fs_fine={fs_fine}, time_window={time_window}")
    print(f"      time_bins per window={time_bins}, n_windows={n_windows}")

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
            
            # Skip if GT has no variance
            if gt_power.std() < 1e-10:
                continue

            for method in methods:
                est_power = method_data[method][j, start_sample:end_sample] ** 2
                
                # Skip if estimate has no variance
                if est_power.std() < 1e-10:
                    continue
                    
                try:
                    corr, pval = pearsonr(gt_power, est_power)
                    correlations[method][freq_idx, win_idx] = corr
                    correlation_pvals[method][freq_idx, win_idx] = pval
                except Exception as e:
                    print(f"      Warning: pearsonr failed for {method}, freq_idx={freq_idx}, win={win_idx}: {e}")

    # Print summary
    for method in methods:
        nonzero = (correlations[method] != 0).sum()
        total = correlations[method].size
        print(f"      {method}: {nonzero}/{total} non-zero correlations")

    return correlations, correlation_pvals, time_centers


# =============================================================================
# PANEL A: RAW DATA (adapted from panel_a_data.py for multiple units)
# =============================================================================

def plot_panel_a(
    sim_data: dict,
    output_path: str,
    time_start: float = 0.0,
    time_duration: float = 3.0,
    window_sec: float = 2.0,
    figsize: tuple = (7.0, 1.4),
):
    """
    Plot raw data: LFP (observed + clean) + multiple spike rasters.
    
    LAYOUT (y-coordinates, carefully calculated):
    ==============================================
    0.95  Legend
    0.90  LFP top
    0.74  LFP center  
    0.58  LFP bottom
    0.52  Δ_b arrow
    0.48  Δ_b text (below arrow)
    0.38  Spikes top (gap from Δ_b text)
    0.24  Spikes center
    0.10  Spikes bottom
    0.04  Time arrow
    -0.02 Δ_spk
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 8,
    })
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    lfp_obs = sim_data['LFP']
    lfp_clean = sim_data.get('LFP_clean', None)
    spikes = sim_data['spikes']
    fs = sim_data.get('fs', 1000.0)
    delta_spk = sim_data.get('delta_spk', 0.001)
    S = spikes.shape[0]
    
    lfp_start = int(time_start * fs)
    lfp_end = int((time_start + time_duration) * fs)
    lfp_end = min(lfp_end, len(lfp_obs))
    
    t_lfp = np.linspace(0, time_duration, lfp_end - lfp_start)
    lfp_obs_snippet = lfp_obs[lfp_start:lfp_end]
    if lfp_clean is not None:
        lfp_clean_snippet = lfp_clean[lfp_start:lfp_end]
    
    # LAYOUT - precise y-coordinates
    LFP_TOP = 0.90
    LFP_BOTTOM = 0.58
    LFP_CENTER = (LFP_TOP + LFP_BOTTOM) / 2  # 0.74
    LFP_HALFRANGE = (LFP_TOP - LFP_BOTTOM) / 2 * 0.90
    
    DELTA_B_ARROW_Y = 0.52
    DELTA_B_TEXT_Y = 0.48  # Below arrow
    
    # Spikes moved DOWN to avoid overlap with Δ_b text
    SPIKE_TOP = 0.38
    SPIKE_BOTTOM = 0.10
    SPIKE_HEIGHT = (SPIKE_TOP - SPIKE_BOTTOM) / max(S, 1)
    
    TIME_ARROW_Y = 0.04
    DELTA_SPK_Y = -0.02
    
    def normalize_lfp(x, center, halfrange):
        x_c = x - x.mean()
        x_n = x_c / (np.abs(x_c).max() + 1e-10)
        return x_n * halfrange + center
    
    lfp_obs_plot = normalize_lfp(lfp_obs_snippet, LFP_CENTER, LFP_HALFRANGE)
    if lfp_clean is not None:
        lfp_clean_plot = normalize_lfp(lfp_clean_snippet, LFP_CENTER, LFP_HALFRANGE)
    
    spk_start = int(time_start / delta_spk)
    spk_end = int((time_start + time_duration) / delta_spk)
    spk_end = min(spk_end, spikes.shape[1])
    
    # Reduced margins for less white space
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.30, time_duration + 0.05)
    ax.set_ylim(-0.08, 0.98)
    ax.axis('off')
    
    # 1. LFP traces
    ax.plot(t_lfp, lfp_obs_plot, color=PANEL_A_COLORS['lfp'], 
            linewidth=0.6, alpha=0.7, label='Observed')
    if lfp_clean is not None:
        ax.plot(t_lfp, lfp_clean_plot, color=PANEL_A_COLORS['latent'], 
                linewidth=0.8, label='Noise-free')
    
    ax.text(-0.18, LFP_CENTER, 'LFP', fontsize=7, ha='right', va='center', 
            color=PANEL_A_COLORS['lfp'], fontweight='bold')
    
    # 2. Legend at top
    ax.legend(loc='upper center', fontsize=5, ncol=2, frameon=False,
              bbox_to_anchor=(0.5, 1.0), columnspacing=1.0)
    
    # 3. Window blocks on LFP only
    n_blocks = int(time_duration / window_sec)
    gap = 0.03
    for i in range(n_blocks):
        x0 = i * window_sec + gap
        w = window_sec - 2 * gap
        rect = Rectangle(
            (x0, LFP_BOTTOM), w, LFP_TOP - LFP_BOTTOM,
            facecolor=PANEL_A_COLORS['lfp'], alpha=0.10,
            edgecolor=PANEL_A_COLORS['lfp'], linewidth=0.5, linestyle='--'
        )
        ax.add_patch(rect)
    
    # 4. Δ_b annotation - arrow at DELTA_B_ARROW_Y, text BELOW at DELTA_B_TEXT_Y
    if n_blocks >= 1:
        ax.plot([gap, window_sec - gap], [DELTA_B_ARROW_Y, DELTA_B_ARROW_Y],
                color=PANEL_A_COLORS['lfp'], lw=0.8)
        ax.plot(gap, DELTA_B_ARROW_Y, '<', color=PANEL_A_COLORS['lfp'], markersize=3)
        ax.plot(window_sec - gap, DELTA_B_ARROW_Y, '>', color=PANEL_A_COLORS['lfp'], markersize=3)
        ax.text(window_sec / 2, DELTA_B_TEXT_Y, r'$\Delta_b$', fontsize=6,
                ha='center', va='top', color=PANEL_A_COLORS['lfp'])
    
    # 5. Spike rasters - moved DOWN
    for s in range(S):
        y_bot = SPIKE_BOTTOM + s * SPIKE_HEIGHT
        y_top = y_bot + SPIKE_HEIGHT * 0.7
        
        spike_idx = np.where(spikes[s, spk_start:spk_end])[0]
        spike_times = spike_idx * delta_spk
        
        for st in spike_times:
            ax.plot([st, st], [y_bot, y_top], color=PANEL_A_COLORS['spike'], lw=0.4)
    
    ax.text(-0.18, (SPIKE_BOTTOM + SPIKE_TOP) / 2, 'Spikes', fontsize=7,
            ha='right', va='center', color=PANEL_A_COLORS['spike'], fontweight='bold')
    
    # 6. Time arrow
    ax.annotate('', xy=(time_duration, TIME_ARROW_Y), xytext=(0, TIME_ARROW_Y),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    ax.text(time_duration + 0.02, TIME_ARROW_Y, 'time', fontsize=6, ha='left', va='center')
    
    # 7. Δ_spk - short, below time arrow
    dspk_x0 = 0.0
    dspk_x1 = 0.12
    
    ax.plot([dspk_x0, dspk_x1], [DELTA_SPK_Y, DELTA_SPK_Y],
            color=PANEL_A_COLORS['spike'], lw=0.8)
    ax.plot(dspk_x0, DELTA_SPK_Y, '<', color=PANEL_A_COLORS['spike'], markersize=2)
    ax.plot(dspk_x1, DELTA_SPK_Y, '>', color=PANEL_A_COLORS['spike'], markersize=2)
    ax.text((dspk_x0 + dspk_x1) / 2, DELTA_SPK_Y - 0.025, r'$\Delta_{\mathrm{spk}}$',
            fontsize=5, ha='center', va='top', color=PANEL_A_COLORS['spike'])
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0.02)
    plt.close()
    
    print(f"  Saved: {output_path}")
    return output_path


# =============================================================================
# PANEL C: SPECTROGRAM - 4 ROWS (GT / MT / LFP-only / Joint)
# =============================================================================

def plot_panel_c_spectrogram(
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
    figsize: tuple = (4.4, 2.3),
):
    """
    Plot 4-row spectrogram: GT / MT / LFP-only / Joint
    
    - Uses viridis colormap
    - INDIVIDUALIZED vmin/vmax per method: vmax=percentile(100), vmin=vmax-40
    - No frequency marker lines
    """
    set_style(font_size=6)
    
    has_gt = amp_gt_fine is not None and amp_gt_fine.max() > 0
    n_panels = 4 if has_gt else 3
    
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.15, left=0.10, right=0.98, top=0.95, bottom=0.12)
    
    def to_db(power):
        return 10 * np.log10(power + 1e-10)
    
    # Compute dB for each method
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
        
        # 4 rows: GT / MT / LFP-only / Joint
        panels = [
            ('gt', db_gt, METHOD_CONFIG['gt']['label']),
            ('mt', db_mt, METHOD_CONFIG['mt']['label']),
            ('lfp', db_lfp, METHOD_CONFIG['lfp']['label']),
            ('spk', db_joint, METHOD_CONFIG['spk']['label']),
        ]
    else:
        # 3 rows: MT / LFP-only / Joint
        panels = [
            ('mt', db_mt, METHOD_CONFIG['mt']['label']),
            ('lfp', db_lfp, METHOD_CONFIG['lfp']['label']),
            ('spk', db_joint, METHOD_CONFIG['spk']['label']),
        ]
    
    for ax, (key, db_power, title) in zip(axes, panels):
        # INDIVIDUALIZED vmin/vmax: vmax = max, vmin = vmax - 40 dB
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
            cmap='Reds',
            vmin=vmin,
            vmax=vmax,
            interpolation='none',
        )
        
        # Compact y-label
        ax.set_ylabel('Hz', fontsize=5)
        
        # Title as text inside plot
        ax.text(0.02, 0.85, title, transform=ax.transAxes, fontsize=5,
                fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.6))

        ax.tick_params(labelsize=5)

    axes[-1].set_xlabel('Time (s)', fontsize=6)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    return output_path


# =============================================================================
# PANEL D: TIME SNAPSHOTS (adapted from spectral_dynamics.py)
# =============================================================================

def plot_panel_d_snapshots(
    amp_gt_fine: Optional[np.ndarray],
    amp_mt_scaled: np.ndarray,
    amp_lfp_scaled: np.ndarray,
    amp_joint_scaled: np.ndarray,
    freqs: np.ndarray,
    idx_sig: List[int],
    delta_spk: float,
    output_path: str,
    amp_joint_std_scaled: Optional[np.ndarray] = None,
    n_snapshots: int = 2,
    snapshot_sec: float = 10.0,
    ci_level: float = 0.95,
    figsize: tuple = (3.0, 1.8),
):
    """
    Plot 2 freq × 2 time snapshots with 3 methods + CI bands.
    Height=1.8" and bottom=0.20 to align with Panel E.
    """
    set_style(font_size=6)
    
    has_gt = amp_gt_fine is not None and amp_gt_fine.max() > 0
    ci_mult = norm.ppf((1 + ci_level) / 2)
    
    J, T_fine = amp_mt_scaled.shape
    snapshot_samples = int(snapshot_sec / delta_spk)
    n_signals = len(idx_sig)

    if n_signals == 0 or T_fine <= snapshot_samples:
        print("  [SKIP] snapshots: insufficient data")
        return output_path

    total_samples = T_fine - snapshot_samples
    if n_snapshots > 1:
        snapshot_starts = np.linspace(0, total_samples, n_snapshots, dtype=int)
    else:
        snapshot_starts = [total_samples // 2]

    fig, axes = plt.subplots(n_signals, n_snapshots, figsize=figsize, squeeze=False)
    # bottom=0.20, top=0.88 to align with Panel E; 
    # set hspace slightly smaller for second row
    if n_signals == 2:
        # We'll set a smaller hspace just for this case
        fig.subplots_adjust(hspace=0.22, wspace=0.12, left=0.12, right=0.98, top=0.88, bottom=0.20)
    else:
        fig.subplots_adjust(hspace=0.40, wspace=0.12, left=0.12, right=0.98, top=0.88, bottom=0.20)

    legend_handles = []
    legend_labels = []

    for col, start_sample in enumerate(snapshot_starts):
        end_sample = start_sample + snapshot_samples
        t_local = np.arange(snapshot_samples) * delta_spk

        for row, j in enumerate(idx_sig):
            ax = axes[row, col]

            if has_gt:
                gt = amp_gt_fine[j, start_sample:end_sample]
                if gt.max() > 0:
                    line, = ax.plot(t_local, gt,
                           color=METHOD_CONFIG["gt"]["color"],
                           lw=0.8, linestyle='--', alpha=0.9)
                    if row == 0 and col == 0:
                        legend_handles.append(line)
                        legend_labels.append(METHOD_CONFIG["gt"]["label"])

            line_mt, = ax.plot(t_local, amp_mt_scaled[j, start_sample:end_sample],
                   color=METHOD_CONFIG['mt']['color'], lw=0.6, alpha=0.5)
            if row == 0 and col == 0:
                legend_handles.append(line_mt)
                legend_labels.append(METHOD_CONFIG['mt']['label'])

            line_lfp, = ax.plot(t_local, amp_lfp_scaled[j, start_sample:end_sample],
                   color=METHOD_CONFIG['lfp']['color'], lw=0.8)
            if row == 0 and col == 0:
                legend_handles.append(line_lfp)
                legend_labels.append(METHOD_CONFIG['lfp']['label'])

            joint_mean = amp_joint_scaled[j, start_sample:end_sample]
            line_joint, = ax.plot(t_local, joint_mean,
                   color=METHOD_CONFIG['spk']['color'], lw=1.0)
            if row == 0 and col == 0:
                legend_handles.append(line_joint)
                legend_labels.append(METHOD_CONFIG['spk']['label'])

            if amp_joint_std_scaled is not None:
                joint_std = amp_joint_std_scaled[j, start_sample:end_sample]
                lower = np.maximum(joint_mean - ci_mult * joint_std, 0)
                upper = joint_mean + ci_mult * joint_std
                ax.fill_between(t_local, lower, upper,
                               color=METHOD_CONFIG['spk']['color'],
                               alpha=0.2)

            ax.set_xlim(0, snapshot_sec)
            ax.set_ylim(0, None)
            ax.tick_params(labelsize=5)
            ax.set_yticks([])
            ax.set_yticklabels([])

            # Suppress x-ticks for the first row
            if row == 0:
                start_sec = start_sample * delta_spk
                ax.set_title(f't={start_sec:.0f}-{start_sec + snapshot_sec:.0f}s', 
                            fontsize=6, fontweight='bold')
                ax.set_xticklabels([])
                ax.set_xticks([])
            if row == n_signals - 1:
                ax.set_xlabel('Time (s)', fontsize=5, labelpad=1)
            if col == 0:
                ax.set_ylabel(f'{freqs[j]:.0f} Hz', fontsize=6)

    # Legend at y=0.02 to align with Panel E
    fig.legend(legend_handles, legend_labels, 
               loc='lower center', 
               bbox_to_anchor=(0.55, 0.00),
               ncol=len(legend_handles), 
               fontsize=5, 
               frameon=False,
               handlelength=1.2,
               columnspacing=0.8)
    
    # Shared y-axis label
    fig.text(0.01, 0.54, 'Amplitude (a.u.)', fontsize=6, 
            ha='left', va='center', rotation=90)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0.02)
    plt.close()
    
    print(f"  Saved: {output_path}")
    return output_path


# =============================================================================
# PANEL E: CORRELATION BOXPLOT (adapted from spectral_dynamics.py)
# =============================================================================

def plot_panel_e_correlation(
    correlations: Dict[str, np.ndarray],
    freqs: np.ndarray,
    idx_sig: List[int],
    time_window: float,
    output_path: str,
    figsize: tuple = (2.2, 1.8),
):
    """
    Compact correlation boxplot for all signal frequencies.
    Height=1.8" and bottom=0.20 to align with Panel D.
    """
    set_style(font_size=6)
    
    methods = list(correlations.keys())
    n_methods = len(methods)
    n_freqs = len(idx_sig)
    
    print(f"    Plotting {n_methods} methods x {n_freqs} frequencies")
    
    fig, ax = plt.subplots(figsize=figsize)
    # bottom=0.20, top=0.88 to align with Panel D
    fig.subplots_adjust(left=0.18, right=0.95, top=0.88, bottom=0.20)

    all_box_data = []
    all_positions = []
    all_colors = []
    
    method_order = ['mt', 'lfp', 'spk']
    methods_sorted = [m for m in method_order if m in methods]
    
    box_width = 0.6
    method_spacing = 0.8
    freq_spacing = n_methods * method_spacing + 1.0
    
    tick_positions = []
    tick_labels = []
    
    for freq_idx, j in enumerate(idx_sig):
        freq_val = freqs[j]
        base_pos = freq_idx * freq_spacing
        
        for method_idx, method in enumerate(methods_sorted):
            corr_values = correlations[method][freq_idx, :]
            pos = base_pos + method_idx * method_spacing
            
            all_box_data.append(corr_values)
            all_positions.append(pos)
            
            config = METHOD_CONFIG.get(method, {"color": "gray"})
            all_colors.append(config['color'])
        
        tick_positions.append(base_pos + (n_methods - 1) * method_spacing / 2)
        tick_labels.append(f'{freq_val:.0f}')
    
    if len(all_box_data) > 0:
        bp = ax.boxplot(
            all_box_data, 
            positions=all_positions, 
            widths=box_width, 
            patch_artist=True,
            showfliers=False,
            medianprops={'color': 'black', 'linewidth': 0.8},
            whiskerprops={'linewidth': 0.5},
            capprops={'linewidth': 0.5},
        )
        
        for patch, color in zip(bp['boxes'], all_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(0.5)
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=5)
    ax.set_ylabel('Corr. w/ GT', fontsize=6)
    ax.set_xlabel('Freq (Hz)', fontsize=5, labelpad=1)
    # ax.set_title(f'{time_window:.0f}s windows', fontsize=6, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3, lw=0.5)
    ax.tick_params(labelsize=5)
    ax.set_ylim([-0.3, 1.05])
    
    if len(all_positions) > 0:
        ax.set_xlim(min(all_positions) - 0.8, max(all_positions) + 0.8)

    # Legend at y=0.02 to align with Panel D
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1,
                      facecolor=METHOD_CONFIG.get(m, {"color": "gray"})['color'],
                      edgecolor='black',
                      linewidth=0.5,
                      alpha=0.7, 
                      label=METHOD_CONFIG.get(m, {"label": m})['label'])
        for m in methods_sorted
    ]
    fig.legend(handles=legend_elements, 
               loc='lower center',
               bbox_to_anchor=(0.56, 0.01),
               ncol=len(methods_sorted), 
               fontsize=4, 
               frameon=False,
               handlelength=0.8, 
               handleheight=0.6,
               columnspacing=0.5)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0.02)
    plt.close()
    
    print(f"  Saved: {output_path}")
    return output_path


# =============================================================================
# DATA PREPARATION (load from saved results, don't recompute)
# =============================================================================

def prepare_spectral_data(sim_data: dict, joint_results: dict) -> dict:
    """
    Prepare amplitude data for plotting.
    
    IMPORTANT: Loads Y_cube from saved results instead of recomputing multitaper.
    """
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
    
    # -------------------------------------------------------------------------
    # Ground truth amplitude
    # -------------------------------------------------------------------------
    if has_gt:
        A_gt = sim_data['A_t']
        freqs_signal = sim_data['freqs_hz']
        
        amp_gt_fine = np.zeros((J, T_fine))
        signal_freq_to_j = {}
        for i, f_sig in enumerate(freqs_signal):
            j_idx = np.argmin(np.abs(freqs - f_sig))
            if np.abs(freqs[j_idx] - f_sig) < 1.5 and i < A_gt.shape[0]:
                amp_gt_fine[j_idx, :min(T_fine, A_gt.shape[1])] = A_gt[i, :min(T_fine, A_gt.shape[1])]
                signal_freq_to_j[f_sig] = j_idx
        
        idx_sig = list(signal_freq_to_j.values())
        print(f"  Signal frequencies mapped: {list(signal_freq_to_j.keys())} -> indices {idx_sig}")
    else:
        amp_gt_fine = None
        idx_sig = list(range(min(4, J)))
    
    # -------------------------------------------------------------------------
    # LFP-only amplitude (from Z_fine_em)
    # -------------------------------------------------------------------------
    if 'Z_fine_em' in joint_results and joint_results['Z_fine_em'] is not None:
        Z_fine_em = joint_results['Z_fine_em']
        amp_lfp_fine = fine_to_amplitude_JT(Z_fine_em, J, M)
        print(f"  LFP-only (Z_fine_em): {amp_lfp_fine.shape}, range [{amp_lfp_fine.min():.3f}, {amp_lfp_fine.max():.3f}]")
    else:
        amp_lfp_fine = np.zeros((J, T_fine))
        print("  WARNING: Z_fine_em not found, using zeros")
    
    # -------------------------------------------------------------------------
    # Joint amplitude with uncertainty (from Z_fine_joint)
    # -------------------------------------------------------------------------
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
        print(f"  Joint (Z_fine_joint): {amp_joint_fine.shape}, range [{amp_joint_fine.min():.3f}, {amp_joint_fine.max():.3f}]")
    else:
        amp_joint_fine = np.zeros((J, T_fine))
        amp_joint_std = None
        print("  WARNING: Z_fine_joint not found, using zeros")
    
    # -------------------------------------------------------------------------
    # Multitaper amplitude (from Y_cube - LOAD, don't recompute!)
    # -------------------------------------------------------------------------
    if 'Y_cube' in joint_results and joint_results['Y_cube'] is not None:
        Y_cube = joint_results['Y_cube']  # (J, M, K) complex
        print(f"  Loading Y_cube from saved results: {Y_cube.shape}")
        
        # Convert to amplitude: |Y|, average over tapers, then upsample to fine grid
        amp_mt_block = np.abs(Y_cube).mean(axis=1)  # (J, K)
        K = amp_mt_block.shape[1]
        
        # Upsample block-level to fine-level by repeating
        amp_mt_fine = np.zeros((J, T_fine))
        for k in range(K):
            start = k * block_samples
            end = min((k + 1) * block_samples, T_fine)
            amp_mt_fine[:, start:end] = amp_mt_block[:, k:k+1]
        
        print(f"  Multitaper (Y_cube): {amp_mt_fine.shape}, range [{amp_mt_fine.min():.3f}, {amp_mt_fine.max():.3f}]")
    else:
        # Fallback: recompute if Y_cube not saved
        print("  WARNING: Y_cube not found, recomputing multitaper...")
        if HAS_MNE:
            import mne
            tfr_raw = mne.time_frequency.tfr_array_multitaper(
                lfp[None, None, :],
                sfreq=fs,
                freqs=freqs,
                n_cycles=freqs * window_sec_res,
                time_bandwidth=2 * NW,
                output='complex',
                zero_mean=False,
                verbose=False,
            )
            
            if tfr_raw.ndim == 5:
                tfr = tfr_raw[0, 0, :, :, :].transpose(1, 0, 2)
            else:
                tfr = tfr_raw[0, 0, :, :][:, None, :]
            
            amp_mt_fine = np.abs(tfr).mean(axis=1)
        else:
            amp_mt_fine = np.zeros((J, T_fine))
    
    # -------------------------------------------------------------------------
    # Align lengths
    # -------------------------------------------------------------------------
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
    
    print(f"  Aligned to T_fine = {T_fine}")
    
    # -------------------------------------------------------------------------
    # Scale to match ground truth
    # -------------------------------------------------------------------------
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
    
    return {
        'amp_gt_fine': amp_gt_fine,
        'amp_mt_scaled': amp_mt_scaled,
        'amp_lfp_scaled': amp_lfp_scaled,
        'amp_joint_scaled': amp_joint_scaled,
        'amp_joint_std_scaled': amp_joint_std_scaled,
        'freqs': freqs,
        'freqs_coupled': freqs_coupled,
        'freqs_all_signal': freqs_all_signal,
        'idx_sig': idx_sig,
        'duration': duration,
        'delta_spk': delta_spk,
        'block_samples': block_samples,
        'T_fine': T_fine,
        'fs_fine': fs_fine,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate methods figure panels')
    parser.add_argument('--data', type=str, required=True, help='Path to simulation data (.pkl)')
    parser.add_argument('--results', type=str, required=True, help='Path to results directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--freqs_to_show', type=float, nargs='+', default=[11, 27],
                        help='Frequencies for time snapshots')
    parser.add_argument('--plot_duration', type=float, default=3.0,
                        help='Duration of raw data to plot (s)')
    parser.add_argument('--snapshot_sec', type=float, default=10.0,
                        help='Duration of each snapshot window')
    parser.add_argument('--n_snapshots', type=int, default=2,
                        help='Number of time snapshots')
    parser.add_argument('--corr_window', type=float, default=20.0,
                        help='Correlation window size (s)')
    
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print("Generating Methods Figure Panels")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    with open(args.data, 'rb') as f:
        sim_data = pickle.load(f)
    
    joint_path = os.path.join(args.results, 'joint.pkl')
    with open(joint_path, 'rb') as f:
        joint_results = pickle.load(f)
    
    # Prepare spectral data (copied exactly from spectral_dynamics.py)
    print("\nPreparing spectral data...")
    data = prepare_spectral_data(sim_data, joint_results)
    
    # Frequency indices for snapshots
    freqs = data['freqs']
    idx_freqs_to_show = [np.argmin(np.abs(freqs - f)) for f in args.freqs_to_show]
    
    # =========================================================================
    # Panel A: Raw data
    # =========================================================================
    print("\n[1/4] Panel A: Raw data...")
    plot_panel_a(
        sim_data,
        os.path.join(args.output, 'panel_a_raw.png'),
        time_start=0.0,
        time_duration=args.plot_duration,
        window_sec=joint_results.get('window_sec', 2.0),
        figsize=(7.0, 1.3),
    )
    
    # =========================================================================
    # Panel C: Spectrogram (4 rows: GT / MT / LFP-only / Joint)
    # =========================================================================
    print("\n[2/4] Panel C: Spectrogram (4 rows)...")
    plot_panel_c_spectrogram(
        data['amp_gt_fine'],
        data['amp_mt_scaled'],
        data['amp_lfp_scaled'],
        data['amp_joint_scaled'],
        data['freqs'],
        data['duration'],
        data['block_samples'],
        os.path.join(args.output, 'panel_c_spectrogram.png'),
        freqs_coupled=data['freqs_coupled'] if len(data['freqs_coupled']) > 0 else None,
        freqs_all_signal=data['freqs_all_signal'] if len(data['freqs_all_signal']) > 0 else None,
        figsize=(4.4, 2.3),
    )
    
    # =========================================================================
    # Panel D: Time snapshots
    # =========================================================================
    print("\n[3/4] Panel D: Time snapshots...")
    plot_panel_d_snapshots(
        data['amp_gt_fine'],
        data['amp_mt_scaled'],
        data['amp_lfp_scaled'],
        data['amp_joint_scaled'],
        data['freqs'],
        idx_freqs_to_show,
        data['delta_spk'],
        os.path.join(args.output, 'panel_d_snapshots.png'),
        amp_joint_std_scaled=data['amp_joint_std_scaled'],
        n_snapshots=args.n_snapshots,
        snapshot_sec=args.snapshot_sec,
        figsize=(3.0, 1.8),
    )
    
    # =========================================================================
    # Panel E: Correlation boxplot
    # =========================================================================
    print("\n[4/4] Panel E: Correlation boxplot...")
    if data['amp_gt_fine'] is not None:
        method_data = {
            'mt': data['amp_mt_scaled'],
            'lfp': data['amp_lfp_scaled'],
            'spk': data['amp_joint_scaled'],
        }
        
        # Debug: print method data stats
        print(f"  Method data shapes:")
        for name, arr in method_data.items():
            print(f"    {name}: {arr.shape}, range [{arr.min():.3f}, {arr.max():.3f}]")
        print(f"  GT shape: {data['amp_gt_fine'].shape}, range [{data['amp_gt_fine'].min():.3f}, {data['amp_gt_fine'].max():.3f}]")
        print(f"  idx_sig: {data['idx_sig']}")
        print(f"  fs_fine: {data['fs_fine']}, corr_window: {args.corr_window}")
        
        correlations, _, time_centers = compute_correlations_over_time(
            data['amp_gt_fine'],
            method_data,
            data['idx_sig'],
            data['fs_fine'],
            args.corr_window,
        )
        
        # Debug: print correlation stats
        print(f"  Correlation results:")
        print(f"    n_windows: {len(time_centers)}")
        for method, corr_arr in correlations.items():
            print(f"    {method}: shape {corr_arr.shape}, mean {corr_arr.mean():.3f}, range [{corr_arr.min():.3f}, {corr_arr.max():.3f}]")
        
        plot_panel_e_correlation(
            correlations,
            data['freqs'],
            data['idx_sig'],
            args.corr_window,
            os.path.join(args.output, 'panel_e_correlation.png'),
            figsize=(2.2, 1.8),
        )
    else:
        print("  [SKIP] No ground truth for correlation plot")
    
    print("\n" + "=" * 60)
    print(f"Done! Panels saved to: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()