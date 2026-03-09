#!/usr/bin/env python3
"""
plot_trial_manuscript_figures.py

Generate publication-ready figures for trial-structured spike-field coupling analysis.
Takes REAL inference results as input - does NOT generate mock data.

Usage:
    python plot_trial_manuscript_figures.py \
        --data ./data/sim_with_trials.pkl \
        --joint ./results/joint.pkl \
        --traditional ./results/traditional_methods.pkl \
        --output ./figures/

Generates 4 separate figures:
    1. spectrogram.pdf - Trial-averaged spectrogram (GT | MT | Joint)
    2. correlation.pdf - Correlation boxplot for coupled bands
    3. dynamics.pdf - Trial-specific dynamics (2 freqs × 4 trials)
    4. heatmaps.pdf - Effect size and p-value heatmaps
"""

import os
import sys
import pickle
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('pdf')  # Force PDF backend
from scipy.stats import norm
from scipy.interpolate import interp1d
from dataclasses import dataclass, field

# Optional imports - MNE not needed since we load Y_trials from results

# For unpickling simulation data
@dataclass
class TrialSimConfig:
    freqs_hz: np.ndarray = field(default_factory=lambda: np.array([11.0, 19.0, 27.0, 43.0]))
    freqs_hz_extra: np.ndarray = field(default_factory=lambda: np.array([7.0, 35.0]))
    R: int = 100
    S: int = 5
    k_active: int = 3
    fs: float = 1000.0
    duration_sec: float = 10.0
    delta_spk: float = 0.001
    half_bw_hz: np.ndarray = field(default_factory=lambda: np.array([0.05]*4))
    lam_delta_scale: float = 4.0
    sigma_v_shared: np.ndarray = field(default_factory=lambda: np.array([4.0]*4))
    sigma_v_delta: np.ndarray = field(default_factory=lambda: np.array([4.0]*4))
    sigma_eps: np.ndarray = field(default_factory=lambda: np.array([5.0]*4))
    sigma_eps_other: float = 15.0
    noise_fmax_hz: int = 60
    n_lags: int = 20
    hist_scale: float = 1.5
    hist_tau: float = 0.03
    b0_mu: float = -2.0
    b0_sd: float = 0.4
    beta_mag_lo: float = 0.02
    beta_mag_hi: float = 0.15
    out_path: str = "./data/sim.pkl"


# =============================================================================
# STYLE
# =============================================================================
def set_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 7,
        'axes.titlesize': 7,
        'axes.labelsize': 7,
        'axes.linewidth': 0.5,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 5,
        'lines.linewidth': 0.8,
        'xtick.major.size': 2,
        'ytick.major.size': 2,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        # ADD THESE:
        'savefig.dpi': 300,
        'figure.dpi': 150,
        'image.interpolation': 'nearest',  # Crisp pixels in imshow
    })
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

COLORS = {
    'gt': '#333333',
    'mt': '#2E86AB',
    'lfp': '#A23B72', 
    'spk': '#F18F01',
}

LABELS = {
    'gt': 'Ground Truth',
    'mt': 'Multi-taper',
    'lfp': 'CT-SSMT (LFP only)',
    'spk': 'Joint SSMT',
}


# =============================================================================
# HELPERS
# =============================================================================

def optimal_scale(est, gt):
    """Least-squares scale factor."""
    e, g = np.abs(est).flatten(), np.abs(gt).flatten()
    d = np.dot(e, e)
    return np.dot(g, e) / d if d > 1e-10 else 1.0


def resample(Z, T_target):
    """Resample last axis to T_target."""
    T_src = Z.shape[-1]
    if T_src == T_target:
        return Z
    t_src = np.linspace(0, 1, T_src)
    t_tgt = np.linspace(0, 1, T_target)
    shape = Z.shape[:-1]
    Z_flat = Z.reshape(-1, T_src)
    out = np.zeros((Z_flat.shape[0], T_target), dtype=Z.dtype)
    for i in range(Z_flat.shape[0]):
        if np.iscomplexobj(Z):
            out[i] = np.interp(t_tgt, t_src, Z_flat[i].real) + 1j * np.interp(t_tgt, t_src, Z_flat[i].imag)
        else:
            out[i] = np.interp(t_tgt, t_src, Z_flat[i])
    return out.reshape(shape + (T_target,))


def interleaved_to_complex(X, J, M=1):
    """Convert interleaved (T, 2*J*M) or (R, T, 2*J*M) to complex."""
    X = np.asarray(X)
    if X.ndim == 2:
        T, D = X.shape
        Z = np.zeros((J, T), dtype=complex)
        for j in range(J):
            for m in range(M):
                b = (j * M + m) * 2
                Z[j] += X[:, b] + 1j * X[:, b+1]
        return Z / M
    elif X.ndim == 3:
        R, T, D = X.shape
        Z = np.zeros((R, J, T), dtype=complex)
        for j in range(J):
            for m in range(M):
                b = (j * M + m) * 2
                Z[:, j, :] += X[:, :, b] + 1j * X[:, :, b+1]
        return Z / M
    raise ValueError(f"Bad shape: {X.shape}")


def interleaved_to_var(V, J, M=1):
    """Extract variance from interleaved format."""
    V = np.asarray(V)
    if V.ndim == 2:
        T, D = V.shape
        out = np.zeros((J, T))
        for j in range(J):
            for m in range(M):
                b = (j * M + m) * 2
                out[j] += V[:, b] + V[:, b+1]
        return out / (M * M)
    elif V.ndim == 3:
        R, T, D = V.shape
        out = np.zeros((R, J, T))
        for j in range(J):
            for m in range(M):
                b = (j * M + m) * 2
                out[:, j, :] += V[:, :, b] + V[:, :, b+1]
        return out / (M * M)
    raise ValueError(f"Bad shape: {V.shape}")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(args):
    """Load all data and prepare for plotting."""
    
    print("Loading simulation data...")
    with open(args.data, 'rb') as f:
        sim = pickle.load(f)
    
    # Joint inference is optional
    joint = {}
    if args.joint and os.path.exists(args.joint):
        print("Loading joint inference results...")
        # Check if it's a directory (npz files) or pickle
        if os.path.isdir(args.joint):
            # Load from npz files
            coupling_path = os.path.join(args.joint, 'coupling.npz')
            spectral_path = os.path.join(args.joint, 'spectral.npz')
            lfp_only_path = os.path.join(args.joint, 'ctssmt_lfp_only.npz')
            metadata_path = os.path.join(args.joint, 'metadata.json')
            
            if os.path.exists(coupling_path):
                coupling = dict(np.load(coupling_path, allow_pickle=False))
                joint['coupling'] = coupling
                joint['freqs'] = coupling.get('freqs')
                joint['beta_mag'] = coupling.get('beta_mag')
                joint['wald_pval'] = coupling.get('wald_pval')
            
            if os.path.exists(spectral_path):
                spectral = dict(np.load(spectral_path, allow_pickle=False))
                joint['spectral'] = spectral
                joint['Y_trials'] = spectral.get('Y_trials')  # Load Y_trials from spectral!
                # Store X/D fine in trace-like structure
                joint['trace'] = {
                    'X_fine_final': spectral.get('X_fine'),
                    'D_fine_final': spectral.get('D_fine'),
                    'X_var_fine_final': spectral.get('X_var_fine'),
                    'D_var_fine_final': spectral.get('D_var_fine'),
                }
                joint['downsample_factor'] = int(spectral.get('downsample_factor', 10))
            
            if os.path.exists(lfp_only_path):
                lfp_only = dict(np.load(lfp_only_path, allow_pickle=False))
                joint['lfp_only'] = lfp_only
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    joint['metadata'] = json.load(f)
                    joint['n_tapers'] = joint['metadata']['ctssmt'].get('n_tapers', 1)
        else:
            # Load from pickle
            with open(args.joint, 'rb') as f:
                joint = pickle.load(f)
    else:
        print("Joint inference results not found, skipping...")
    
    # Traditional methods is optional
    trad = {}
    if args.traditional and os.path.exists(args.traditional):
        print("Loading traditional methods...")
        with open(args.traditional, 'rb') as f:
            trad = pickle.load(f)
    else:
        print("Traditional methods not found, skipping...")
    
    # Basic dimensions
    R, T_lfp = sim['LFP'].shape
    S = sim['spikes'].shape[1]
    fs = sim.get('fs', 1000.0)
    delta_spk = sim.get('delta_spk', 0.001)
    duration = T_lfp / fs
    T_fine = int(duration / delta_spk)
    
    # Frequency grids
    freqs = np.asarray(joint.get('freqs', trad.get('config', {}).get('freqs', np.arange(1, 61, 2))))
    J = len(freqs)
    freqs_coupled = np.asarray(sim.get('freqs_hz_coupled', sim.get('freqs_hz', [])))
    freqs_all = np.asarray(sim.get('freqs_hz', []))
    idx_coupled = [np.argmin(np.abs(freqs - f)) for f in freqs_coupled]
    
    print(f"  R={R}, S={S}, J={J}, T_fine={T_fine}")
    print(f"  Coupled: {freqs_coupled} Hz -> indices {idx_coupled}")
    
    # Ground truth Z_lat -> Z_gt
    Z_gt = None
    if 'Z_lat' in sim:
        Z_lat = sim['Z_lat']  # (R, J_true, T_lfp)
        if Z_lat.shape[-1] != T_fine:
            Z_lat = resample(Z_lat, T_fine)
        Z_gt = np.zeros((R, J, T_fine), dtype=complex)
        for jt, jd in enumerate([np.argmin(np.abs(freqs - f)) for f in freqs_all]):
            if jt < Z_lat.shape[1]:
                Z_gt[:, jd, :] = Z_lat[:, jt, :]
    
    # =================================================================
    # MULTITAPER: Load from Y_trials, DO NOT RECOMPUTE
    # =================================================================
    Z_mt = None
    if 'Y_trials' in joint:
        print("  Loading multitaper from Y_trials (not recomputing)...")
        Y_trials = joint['Y_trials']  # (R, J, M, K)
        R_y, J_y, M_y, K_y = Y_trials.shape
        
        # Y_trials is (R, J, M, K) complex, average over tapers
        Y_avg = Y_trials.mean(axis=2)  # (R, J, K)
        
        # Expand K blocks to T_fine by REPEATING (not interpolating!)
        # Each block represents a time window, so repeat the value across that window
        block_samples = T_fine // K_y
        Z_mt = np.zeros((R_y, J_y, T_fine), dtype=Y_avg.dtype)
        for k in range(K_y):
            start = k * block_samples
            end = (k + 1) * block_samples if k < K_y - 1 else T_fine
            Z_mt[:, :, start:end] = Y_avg[:, :, k:k+1]
        
        print(f"  Z_mt: {Z_mt.shape} (expanded from {K_y} blocks)")
    else:
        print("  WARNING: Y_trials not found, multitaper unavailable")
        Z_mt = np.zeros((R, J, T_fine), dtype=complex)
    
    # =================================================================
    # Joint inference results - load X_fine and D_fine
    # =================================================================
    Z_lfp, Z_spk, Z_spk_var = None, None, None
    trace = joint.get('trace', {})
    
    X_key = 'X_fine_final' if 'X_fine_final' in trace else 'X_fine'
    D_key = 'D_fine_final' if 'D_fine_final' in trace else 'D_fine'
    
    M = joint.get('n_tapers', 1)
    downsample_factor = joint.get('downsample_factor', 10)
    
    if X_key in trace and D_key in trace:
        print(f"  Loading from {X_key} + {D_key}...")
        X = np.asarray(trace[X_key][-1] if isinstance(trace[X_key], list) else trace[X_key])
        D = np.asarray(trace[D_key][-1] if isinstance(trace[D_key], list) else trace[D_key])
        
        X_c = interleaved_to_complex(X, J, M)
        D_c = interleaved_to_complex(D, J, M)
        
        # Get actual time dimension
        T_stored = X_c.shape[-1]
        T_target = T_fine // downsample_factor if downsample_factor > 1 else T_fine
        # test X_c only 
        Z_spk = resample(X_c[None, :, :] + D_c, T_fine)
        # Z_spk = resample(X_c[None, :, :], T_fine)
        Z_lfp = resample(np.tile(X_c[None, :, :], (R, 1, 1)), T_fine)
        
        # Variance
        Xv_key = X_key.replace('fine', 'var_fine')
        Dv_key = D_key.replace('fine', 'var_fine')
        if Xv_key in trace and Dv_key in trace:
            Xv = np.asarray(trace[Xv_key][-1] if isinstance(trace[Xv_key], list) else trace[Xv_key])
            Dv = np.asarray(trace[Dv_key][-1] if isinstance(trace[Dv_key], list) else trace[Dv_key])
            Z_spk_var = resample(interleaved_to_var(Xv, J, M)[None, :, :] + interleaved_to_var(Dv, J, M), T_fine)
    
    # LFP-only from separate file if available
    if 'lfp_only' in joint:
        print("  Loading LFP-only estimates...")
        lfp_only = joint['lfp_only']
        if 'X_fine' in lfp_only and 'D_fine' in lfp_only:
            X_lfp = np.asarray(lfp_only['X_fine'])
            D_lfp = np.asarray(lfp_only['D_fine'])
            X_c_lfp = interleaved_to_complex(X_lfp, J, M)
            D_c_lfp = interleaved_to_complex(D_lfp, J, M)
            Z_lfp = resample(X_c_lfp[None, :, :] + D_c_lfp, T_fine)
    
    # Compute scales
    if Z_gt is not None:
        gt_coupled = Z_gt[:, idx_coupled, :]
        scale_mt = optimal_scale(Z_mt[:, idx_coupled, :], gt_coupled) if Z_mt is not None else 1.0
        scale_lfp = optimal_scale(Z_lfp[:, idx_coupled, :], gt_coupled) if Z_lfp is not None else 1.0
        scale_spk = optimal_scale(Z_spk[:, idx_coupled, :], gt_coupled) if Z_spk is not None else 1.0
    else:
        scale_mt = scale_lfp = scale_spk = 1.0
    
    print(f"  Scales: MT={scale_mt:.4f}, LFP={scale_lfp:.4f}, Joint={scale_spk:.4f}")
    
    # Traditional methods
    plv_val = trad.get('plv', {}).get('values', np.zeros((S, J)))
    plv_pval = trad.get('plv', {}).get('pval_parametric', trad.get('plv', {}).get('pval', np.ones((S, J))))
    sfc_val = trad.get('sfc', {}).get('values', np.zeros((S, J)))
    sfc_pval = trad.get('sfc', {}).get('pval_parametric', trad.get('sfc', {}).get('pval', np.ones((S, J))))
    
    # Resample traditional if needed
    trad_freqs = trad.get('config', {}).get('freqs', freqs)
    if len(trad_freqs) != J and len(trad_freqs) > 0:
        print(f"  Resampling traditional: {len(trad_freqs)} -> {J}")
        plv_val = np.array([np.interp(freqs, trad_freqs, plv_val[s]) for s in range(S)])
        plv_pval = np.array([np.interp(freqs, trad_freqs, plv_pval[s]) for s in range(S)])
        sfc_val = np.array([np.interp(freqs, trad_freqs, sfc_val[s]) for s in range(S)])
        sfc_pval = np.array([np.interp(freqs, trad_freqs, sfc_pval[s]) for s in range(S)])
    
    # Joint beta - check top level first, then coupling sub-dict
    beta_mag = joint.get('beta_mag', None)
    if beta_mag is None:
        coupling = joint.get('coupling', {})
        beta_mag = coupling.get('beta_mag', coupling.get('beta_mag_mean', np.zeros((S, J))))
    
    wald_pval = joint.get('wald_pval', None)
    if wald_pval is None:
        wald_pval = joint.get('wald', {}).get('pval_wald', np.ones((S, J)))
    
    return {
        'Z_gt': Z_gt, 'Z_mt': Z_mt, 'Z_lfp': Z_lfp, 'Z_spk': Z_spk, 'Z_spk_var': Z_spk_var,
        'freqs': freqs, 'freqs_coupled': freqs_coupled, 'idx_coupled': idx_coupled,
        'scale_mt': scale_mt, 'scale_lfp': scale_lfp, 'scale_spk': scale_spk,
        'duration': duration, 'delta_spk': delta_spk, 'R': R, 'S': S, 'J': J, 'T_fine': T_fine,
        'plv_val': plv_val, 'plv_pval': plv_pval,
        'sfc_val': sfc_val, 'sfc_pval': sfc_pval,
        'beta_mag': beta_mag, 'wald_pval': wald_pval,
        'masks': sim['masks'],
    }


# =============================================================================
# FIGURE 1: SPECTROGRAM (GT | MT | Joint) - no LFP, no dashed lines
# =============================================================================
def plot_spectrogram(data, output_path, time_range=(0.5, 9.5), figsize=(5.5, 1.8)):
    """Plot 3 spectrograms side by side: GT, MT, Joint."""
    from scipy.ndimage import zoom
    
    set_style()
    
    R, J, T = data['Z_mt'].shape
    freqs = data['freqs']
    time = np.linspace(0, data['duration'], T)
    t_mask = (time >= time_range[0]) & (time <= time_range[1])
    t0, t1 = np.where(t_mask)[0][0], np.where(t_mask)[0][-1] + 1
    
    # Upsample factor for frequency axis (J is likely small)
    upsample_freq = max(1, 100 // J)  # Target ~100 pixels in freq direction
    
    def upsample(arr):
        """Upsample frequency axis using nearest-neighbor interpolation."""
        return zoom(arr, (upsample_freq, 1), order=0)
    
    def to_db(Z, scale=1.0):
        power = (np.abs(Z[:, :, t0:t1]) ** 2).mean(axis=0) * (scale ** 2)
        return 10 * np.log10(power + 1e-10)
    
    # Only 3 panels: GT, MT, Joint (no LFP)
    panels = [
        ('gt', data['Z_gt'], 1.0, 'Ground Truth'),
        ('mt', data['Z_mt'], data['scale_mt'], 'Multi-taper'),
        ('spk', data['Z_spk'], data['scale_spk'], 'Joint SSMT'),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    fig.subplots_adjust(wspace=0.08, left=0.08, right=0.98, top=0.88, bottom=0.18)
    
    extent = [time_range[0], time_range[1], freqs[0], freqs[-1]]
    
    for i, (ax, (key, Z, scale, title)) in enumerate(zip(axes, panels)):
        if Z is None:
            ax.set_visible(False)
            continue
            
        db = to_db(Z, scale)
        if key == 'gt':
            db[(np.abs(Z[:, :, t0:t1]) ** 2).mean(axis=0) == 0] = np.nan
        
        # Method-specific vmax and vmin = vmax - 30
        valid = db[~np.isnan(db)]
        vmax = np.percentile(valid, 100) if len(valid) > 0 else 50
        vmin = vmax - 30
        
        # Upsample for better PDF rendering
        db_up = upsample(db)
        
        ax.imshow(db_up, aspect='auto', origin='lower', extent=extent,
                  cmap='Reds', vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(title, fontsize=7, fontweight='bold')
        
        # Only middle plot gets x-label
        if i == 1:
            ax.set_xlabel('Time (s)', fontsize=7)
        else:
            ax.tick_params(labelbottom=True)
        
        # Only first plot gets y-label
        if i == 0:
            ax.set_ylabel('Freq (Hz)', fontsize=7)
    
    # Save
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    if output_path.endswith('.pdf'):
        fig.savefig(output_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

# =============================================================================
# FIGURE 2: CORRELATION BOXPLOT
# =============================================================================

def plot_correlation(data, output_path, figsize=(3.5, 2.2)):
    """Correlation boxplot for coupled bands."""
    set_style()
    
    if data['Z_gt'] is None:
        print("  No ground truth, skipping correlation plot")
        return
    
    idx = data['idx_coupled'][:4]  # Max 4 frequencies
    methods = ['mt', 'lfp', 'spk']
    
    # Compute correlations per trial
    correlations = {m: [] for m in methods}
    for m in methods:
        Z = data[f'Z_{m}']
        if Z is None:
            continue
        scale = data[f'scale_{m}']
        corrs = np.zeros((data['R'], len(idx)))
        for i, j in enumerate(idx):
            for r in range(data['R']):
                gt = np.abs(data['Z_gt'][r, j, :])
                est = np.abs(Z[r, j, :]) * scale
                if gt.std() > 1e-10 and est.std() > 1e-10:
                    corrs[r, i] = np.corrcoef(gt, est)[0, 1]
        correlations[m] = corrs
    
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.14, right=0.96, top=0.95, bottom=0.22)
    
    # Plot
    positions = []
    box_data = []
    colors = []
    
    for fi, j in enumerate(idx):
        for mi, m in enumerate(methods):
            if m not in correlations or len(correlations[m]) == 0:
                continue
            positions.append(fi * 4 + mi)
            box_data.append(correlations[m][:, fi])
            colors.append(COLORS[m])
    
    bp = ax.boxplot(box_data, positions=positions, widths=0.7, patch_artist=True,
                    showfliers=False, medianprops={'color': 'k', 'lw': 0.8})
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    
    ax.set_xticks([fi * 4 + 1 for fi in range(len(idx))])
    ax.set_xticklabels([f'{data["freqs"][j]:.0f}' for j in idx])
    ax.set_xlabel('Freq (Hz)', fontsize=7)
    ax.set_ylabel('Correlation with GT', fontsize=7)
    ax.set_ylim(-0.35, 1.08)
    ax.axhline(0, color='gray', ls='--', lw=0.5, alpha=0.5)
    
    # Legend right below x-label, tight spacing
    handles = [plt.Rectangle((0,0),1,1, fc=COLORS[m], alpha=0.7) for m in methods]
    fig.legend(handles, [LABELS[m] for m in methods], 
               loc='lower center', bbox_to_anchor=(0.55, 0.005),
               ncol=3, fontsize=5.5, frameon=False, columnspacing=0.6, handletextpad=0.3)
    
    # Save
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    if output_path.endswith('.pdf'):
        fig.savefig(output_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# FIGURE 3: TRIAL-SPECIFIC DYNAMICS
# =============================================================================

def plot_dynamics(data, output_path, idx_freqs, sample_trials, time_range=(1.0, 8.0), figsize=(7.0, 2.8)):
    """Trial-specific dynamics: n_freqs rows × n_trials cols with legend at bottom."""
    set_style()
    
    T = data['T_fine']
    time = np.linspace(0, data['duration'], T)
    t_mask = (time >= time_range[0]) & (time <= time_range[1])
    t_plot = time[t_mask]
    ci_mult = norm.ppf(0.975)
    
    n_freqs = len(idx_freqs)
    n_trials = len(sample_trials)
    
    fig, axes = plt.subplots(n_freqs, n_trials, figsize=figsize, squeeze=False)
    fig.subplots_adjust(hspace=0.15, wspace=0.08, left=0.06, right=0.98, top=0.90, bottom=0.22)
    
    for row, j in enumerate(idx_freqs):
        for col, trial in enumerate(sample_trials):
            ax = axes[row, col]
            
            # Ground truth (solid line, not dashed)
            if data['Z_gt'] is not None:
                gt = np.abs(data['Z_gt'][trial, j, t_mask])
                if gt.max() > 0:
                    ax.plot(t_plot, gt, color=COLORS['gt'], ls='-', lw=0.8, alpha=0.9)
            
            # Multi-taper
            ax.plot(t_plot, np.abs(data['Z_mt'][trial, j, t_mask]) * data['scale_mt'],
                    color=COLORS['mt'], lw=0.5, alpha=0.5)
            
            # LFP-only
            if data['Z_lfp'] is not None:
                ax.plot(t_plot, np.abs(data['Z_lfp'][trial, j, t_mask]) * data['scale_lfp'],
                        color=COLORS['lfp'], lw=0.8)
            
            # Joint with CI
            if data['Z_spk'] is not None:
                amp = np.abs(data['Z_spk'][trial, j, t_mask]) * data['scale_spk']
                ax.plot(t_plot, amp, color=COLORS['spk'], lw=1.0)
                
                if data['Z_spk_var'] is not None:
                    std = np.sqrt(data['Z_spk_var'][trial, j, t_mask]) * data['scale_spk']
                    ax.fill_between(t_plot, np.maximum(amp - ci_mult*std, 0), amp + ci_mult*std,
                                    color=COLORS['spk'], alpha=0.2)
            
            ax.set_xlim(time_range)
            ax.set_ylim(0, None)
            
            # Title on top row
            if row == 0:
                ax.set_title(f'Trial {trial}', fontsize=7, fontweight='bold')
            
            # Only show tick labels on bottom row
            if row == n_freqs - 1:
                ax.tick_params(labelbottom=True)
            else:
                ax.set_xticklabels([])
            
            # Y-label on left column
            if col == 0:
                ax.set_ylabel(f'{data["freqs"][j]:.0f} Hz', fontsize=7)
            else:
                ax.set_yticklabels([])
    
    # Add single centered x-label using fig.text (centered across all columns)
    fig.text(0.52, 0.10, 'Time (s)', ha='center', va='center', fontsize=7)
    
    # Legend at bottom center, well below the plots
    handles = [plt.Line2D([0], [0], color=COLORS[m], ls='-', lw=1.5) 
               for m in ['gt', 'mt', 'lfp', 'spk']]
    fig.legend(handles, [LABELS[m] for m in ['gt', 'mt', 'lfp', 'spk']], 
               loc='lower center', bbox_to_anchor=(0.5, 0.02),
               ncol=4, fontsize=6, frameon=False, columnspacing=1.5)
    
    # Save
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    if output_path.endswith('.pdf'):
        fig.savefig(output_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def plot_heatmaps(data, output_path, figsize=(7.0, 2.4)):
    """Heatmaps: 3 rows (PLV, SFC, Joint) × 2 cols (Effect, p-value)."""
    set_style()
    
    S, J = data['plv_val'].shape
    freqs = data['freqs']
    masks = data['masks']
    idx_true = [np.argmin(np.abs(freqs - f)) for f in data['freqs_coupled']]
    
    effect_data = [
        (data['plv_val'], 'PLV'),
        (data['sfc_val'], 'SFC'),
        (data['beta_mag'], r'Joint $|\mathrm{E}[\beta]|$'),
    ]
    
    pval_data = [
        (-np.log10(data['plv_pval'] + 1e-300), 'PLV'),
        (-np.log10(data['sfc_pval'] + 1e-300), 'SFC'),
        (-np.log10(data['wald_pval'] + 1e-300), r'Joint $|\mathrm{E}[\beta]|$'),
    ]
    
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.subplots_adjust(hspace=0.25, wspace=0.08, left=0.08, right=0.88, top=0.92, bottom=0.12)
    
    # Build cell edges for pcolormesh
    if len(freqs) > 1:
        df = freqs[1] - freqs[0]
    else:
        df = 2.0
    # X edges: J+1 values
    x_edges = np.concatenate([[freqs[0] - df/2], (freqs[:-1] + freqs[1:]) / 2, [freqs[-1] + df/2]])
    # Y edges: S+1 values (unit boundaries)
    y_edges = np.arange(S + 1) - 0.5
    
    # Common marker style
    marker_style = dict(marker='*', color='white', ms=5, mec='black', mew=0.5, zorder=10)
    
    # Effect size column
    for row, (vals, label) in enumerate(effect_data):
        ax = axes[row, 0]
        
        valid = vals[np.isfinite(vals)]
        if len(valid) > 0:
            vmax = np.percentile(valid, 100)
            vmin = max(0, np.percentile(valid, 0))
        else:
            vmax, vmin = 1, 0
        
        im = ax.pcolormesh(x_edges, y_edges, vals, cmap='Reds', vmin=vmin, vmax=vmax, shading='flat')
        
        # Mark true couplings
        for s in range(S):
            for jt, j in enumerate(idx_true):
                if jt < masks.shape[1] and masks[s, jt]:
                    ax.plot(freqs[j], s, **marker_style)

        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])
        ax.set_ylabel('Unit', fontsize=6)
        ax.set_yticks(range(S))
        ax.text(0.97, 0.85, label, transform=ax.transAxes, fontsize=5, ha='right', fontweight='bold')
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=5)
        
        if row == 0:
            ax.set_title('Effect Size', fontsize=7, fontweight='bold')
        if row == 2:
            ax.set_xlabel('Frequency (Hz)', fontsize=6)
        else:
            ax.set_xticklabels([])
    
    # P-value column
    for row, (vals, label) in enumerate(pval_data):
        ax = axes[row, 1]
        
        valid = vals[np.isfinite(vals)]
        if len(valid) > 0:
            vmax_p = min(np.percentile(valid, 100), 15)
            vmin_p = 0
        else:
            vmax_p, vmin_p = 10, 0
        
        im = ax.pcolormesh(x_edges, y_edges, vals, cmap='hot_r', vmin=vmin_p, vmax=vmax_p, shading='flat')
        
        # Mark true couplings - SAME style
        for s in range(S):
            for jt, j in enumerate(idx_true):
                if jt < masks.shape[1] and masks[s, jt]:
                    ax.plot(freqs[j], s, **marker_style)

        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])
        ax.set_yticks([])
        ax.text(0.97, 0.85, label, transform=ax.transAxes, fontsize=5, ha='right', fontweight='bold')
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=5)
        cbar.ax.axhline(-np.log10(0.05), color='cyan', lw=1)
        
        if row == 0:
            ax.set_title(r'$-\log_{10}(p)$', fontsize=7, fontweight='bold')
        if row == 2:
            ax.set_xlabel('Frequency (Hz)', fontsize=6)
        else:
            ax.set_xticklabels([])
    
    # Save at high DPI
    fig.savefig(output_path, dpi=600, bbox_inches='tight')
    if output_path.endswith('.pdf'):
        fig.savefig(output_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Trial manuscript figures')
    parser.add_argument('--data', required=True, help='Simulation pickle')
    parser.add_argument('--joint', required=False, help='Joint inference results (directory with npz files or pickle)')
    parser.add_argument('--traditional', required=False, help='Traditional methods pickle')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--sample_trials', type=int, nargs='+', default=[0, 25, 50, 99])
    parser.add_argument('--n_freq', type=int, default=2, help='Number of frequencies for dynamics')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print("="*60)
    print("GENERATING TRIAL MANUSCRIPT FIGURES")
    print("="*60)
    
    data = load_data(args)
    
    # Select frequencies and trials
    idx_freqs = data['idx_coupled'][:args.n_freq]
    sample_trials = [t for t in args.sample_trials if t < data['R']][:4]
    
    print(f"\nPlotting frequencies: {[data['freqs'][j] for j in idx_freqs]} Hz")
    print(f"Sample trials: {sample_trials}")
    
    # Generate 4 separate figures
    print("\n[1/4] Spectrogram...")
    plot_spectrogram(data, os.path.join(args.output, 'spectrogram.pdf'))
    
    print("\n[2/4] Correlation...")
    plot_correlation(data, os.path.join(args.output, 'correlation.pdf'))
    
    print("\n[3/4] Dynamics...")
    plot_dynamics(data, os.path.join(args.output, 'dynamics.pdf'), idx_freqs, sample_trials)
    
    print("\n[4/4] Heatmaps...")
    plot_heatmaps(data, os.path.join(args.output, 'heatmaps.pdf'))
    
    print("\n" + "="*60)
    print(f"Done! Figures saved to: {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()