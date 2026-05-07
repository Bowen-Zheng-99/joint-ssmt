#!/usr/bin/env python3
"""
Plot Posterior Spike Probability

Generates a visualization showing:
- Mean posterior spike probability at each fine time step
- 95% credible intervals (shaded region)
- Actual spike times marked as red dots
- Units separated on y-axis

The posterior is computed by iterating through MCMC samples of (β, γ) 
and computing p(spike|Z̃, β, γ) = σ(ψ) where:
    ψ_n = β₀ + Σⱼ(βR_j Z̃R_j,n + βI_j Z̃I_j,n) + Σₕ γₕ N_{n-h}

Key: β samples are in STANDARDIZED units, matching the standardized latent
stored in trace.latent[-1].
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
import argparse
from pathlib import Path


def build_history_design_single(spikes: np.ndarray, n_lags: int = 20) -> np.ndarray:
    """
    Build spike history design matrix for single-trial.
    
    Parameters
    ----------
    spikes : (S, T) spike trains
    n_lags : number of history lags
    
    Returns
    -------
    H : (S, T, n_lags) history design matrix
    """
    S, T = spikes.shape
    H = np.zeros((S, T, n_lags), dtype=np.float32)
    
    for s in range(S):
        for lag in range(n_lags):
            if lag + 1 < T:
                H[s, lag+1:, lag] = spikes[s, :T-lag-1]
    
    return H


def compute_posterior_spike_probability(
    beta_trace: np.ndarray,    # (n_samples, S, 1+2J)
    gamma_trace: np.ndarray,   # (n_samples, S, L) or (n_samples, S, R, L)
    lat_reim: np.ndarray,      # (R, T, 2J) - standardized latent
    spikes: np.ndarray,        # (S, T_fine)
    n_lags: int = 20,
    burn_in_frac: float = 0.6,
    verbose: bool = True,
) -> dict:
    """
    Compute posterior spike probability for each unit at each time step.
    
    Parameters
    ----------
    beta_trace : Posterior samples of β in STANDARDIZED units
    gamma_trace : Posterior samples of γ
    lat_reim : Standardized, taper-averaged, phase-rotated latent [Z̃R | Z̃I]
    spikes : Binary spike matrix
    n_lags : Number of history lags
    burn_in_frac : Fraction of samples to discard as burn-in
    
    Returns
    -------
    dict with:
        'prob_mean': (S, T) mean posterior probability
        'prob_lower': (S, T) 2.5th percentile
        'prob_upper': (S, T) 97.5th percentile
        'psi_mean': (S, T) mean linear predictor
    """
    n_samples, S, P = beta_trace.shape
    J = (P - 1) // 2
    T = spikes.shape[1]
    
    # Handle gamma shape
    if gamma_trace.ndim == 4:
        # (n_samples, S, R, L) -> (n_samples, S, L) take R=0
        gamma_trace = gamma_trace[:, :, 0, :]
    L = gamma_trace.shape[2]
    
    # Apply burn-in
    burn = int(burn_in_frac * n_samples)
    beta_post = beta_trace[burn:]
    gamma_post = gamma_trace[burn:]
    n_post = beta_post.shape[0]
    
    if verbose:
        print(f"[POSTERIOR] Using {n_post} post-burn-in samples (burn-in: {burn_in_frac:.0%})")
        print(f"[POSTERIOR] β shape: {beta_post.shape}, γ shape: {gamma_post.shape}")
        print(f"[POSTERIOR] Latent shape: {lat_reim.shape}, J={J}")
    
    # Extract standardized latent (R=0 for single trial)
    # lat_reim: (R, T, 2J) -> (T, 2J)
    lat = lat_reim[0] if lat_reim.ndim == 3 else lat_reim
    T_lat = lat.shape[0]
    T_use = min(T, T_lat)
    
    if verbose:
        print(f"[POSTERIOR] Using T={T_use} time points (spikes: {T}, latent: {T_lat})")
    
    # Build history design matrix
    H_hist = build_history_design_single(spikes[:, :T_use], n_lags=n_lags)
    # Ensure L matches
    H_hist = H_hist[:, :, :L]
    
    # Preallocate
    prob_samples = np.zeros((n_post, S, T_use), dtype=np.float32)
    
    # Extract latent components
    Ztil_R = lat[:T_use, :J]  # (T, J)
    Ztil_I = lat[:T_use, J:2*J]  # (T, J)
    
    if verbose:
        print(f"[POSTERIOR] Computing probabilities for {n_post} samples...")
    
    for i in range(n_post):
        if verbose and (i+1) % 100 == 0:
            print(f"  Sample {i+1}/{n_post}")
        
        beta_i = beta_post[i]   # (S, 1+2J)
        gamma_i = gamma_post[i]  # (S, L)
        
        for s in range(S):
            # Extract β components for unit s
            b0 = beta_i[s, 0]
            bR = beta_i[s, 1:1+J]       # (J,)
            bI = beta_i[s, 1+J:1+2*J]   # (J,)
            
            # Latent contribution: Σⱼ(βR_j Z̃R_j,t + βI_j Z̃I_j,t)
            # = bR @ Ztil_R.T + bI @ Ztil_I.T  but we want (T,)
            latent_term = Ztil_R @ bR + Ztil_I @ bI  # (T,)
            
            # History contribution: H[s, t, :] @ γ[s]
            hist_term = H_hist[s] @ gamma_i[s]  # (T,)
            
            # Linear predictor
            psi = b0 + latent_term + hist_term  # (T,)
            
            # Spike probability
            prob_samples[i, s, :] = sigmoid(psi)
    
    # Aggregate posterior
    prob_mean = prob_samples.mean(axis=0)
    prob_lower = np.percentile(prob_samples, 2.5, axis=0)
    prob_upper = np.percentile(prob_samples, 97.5, axis=0)
    
    # Also compute mean linear predictor for diagnostics
    psi_mean = np.zeros((S, T_use), dtype=np.float32)
    beta_mean = beta_post.mean(axis=0)
    gamma_mean = gamma_post.mean(axis=0)
    
    for s in range(S):
        b0 = beta_mean[s, 0]
        bR = beta_mean[s, 1:1+J]
        bI = beta_mean[s, 1+J:1+2*J]
        latent_term = Ztil_R @ bR + Ztil_I @ bI
        hist_term = H_hist[s] @ gamma_mean[s]
        psi_mean[s] = b0 + latent_term + hist_term
    
    if verbose:
        print(f"[POSTERIOR] Done. Mean prob range: [{prob_mean.min():.4f}, {prob_mean.max():.4f}]")
    
    return {
        'prob_mean': prob_mean,
        'prob_lower': prob_lower,
        'prob_upper': prob_upper,
        'psi_mean': psi_mean,
        'T_use': T_use,
    }


def plot_posterior_spike_probability(
    prob_mean: np.ndarray,      # (S, T)
    prob_lower: np.ndarray,     # (S, T)
    prob_upper: np.ndarray,     # (S, T)
    spikes: np.ndarray,         # (S, T)
    delta_spk: float,
    t_start: float = 0.0,
    t_duration: float = 5.0,
    figsize: tuple = (7, 2.5),
    output_path: str = None,
    adaptive_ylim: bool = True,
    unit_idx: int = None,
):
    """
    Plot posterior spike probability with credible intervals.
    
    Parameters
    ----------
    prob_mean : Mean posterior probability
    prob_lower : Lower credible bound (2.5%)
    prob_upper : Upper credible bound (97.5%)
    spikes : Binary spike matrix
    delta_spk : Spike time resolution (seconds)
    t_start : Start time for plot (seconds)
    t_duration : Duration to plot (seconds)
    figsize : Figure size
    output_path : Path to save figure
    adaptive_ylim : If True, zoom y-axis to show modulation clearly
    unit_idx : If specified, only plot this unit (0-indexed)
    """
    # Use serif fonts (LaTeX-like without requiring LaTeX installation)
    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
        'mathtext.fontset': 'cm',  # Computer Modern for math
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
    })
    
    S_total, T = prob_mean.shape
    
    # Select units to plot
    if unit_idx is not None:
        units_to_plot = [unit_idx]
    else:
        units_to_plot = list(range(S_total))
    
    S = len(units_to_plot)
    
    # Time vector
    t = np.arange(T) * delta_spk
    
    # Determine time indices to plot
    idx_start = int(t_start / delta_spk)
    idx_end = min(int((t_start + t_duration) / delta_spk), T)
    t_plot = t[idx_start:idx_end]
    
    # Create figure
    fig, axes = plt.subplots(S, 1, figsize=figsize, sharex=True)
    if S == 1:
        axes = [axes]
    
    # Color scheme
    prob_color = '#2166AC'  # Blue for probability
    ci_color = '#92C5DE'    # Light blue for CI
    spike_color = '#B2182B'  # Red for spikes
    
    for i, s in enumerate(units_to_plot):
        ax = axes[i]
        
        # Extract data for this unit
        p_mean = prob_mean[s, idx_start:idx_end]
        p_lower = prob_lower[s, idx_start:idx_end]
        p_upper = prob_upper[s, idx_start:idx_end]
        spk = spikes[s, idx_start:idx_end]
        
        # Plot credible interval
        ax.fill_between(t_plot, p_lower, p_upper, 
                       alpha=0.3, color=ci_color, label='95% CI')
        
        # Plot mean probability
        ax.plot(t_plot, p_mean, color=prob_color, linewidth=1.0, 
               label=r'$\mathrm{E}[p(\mathrm{spike})]$')
        
        # Mark spike times - place at top of visible range
        spike_times = t_plot[spk.astype(bool)]
        if adaptive_ylim:
            spike_y = np.full(len(spike_times), p_upper.max())
        else:
            spike_y = p_mean[spk.astype(bool)]
        
        ax.scatter(spike_times, spike_y, 
                  color=spike_color, s=15, zorder=5, marker='|',
                  linewidths=1.5, label='Spikes', alpha=0.8)
        
        # Compute baseline for reference line
        p_baseline = p_mean.mean()
        
        # Adaptive y-axis: zoom to show modulation
        if adaptive_ylim:
            y_margin = 0.2 * (p_upper.max() - p_lower.min())
            y_lo = max(0, p_lower.min() - y_margin)
            y_hi = min(1, p_upper.max() + y_margin)
            ax.set_ylim(y_lo, y_hi)
        else:
            ax.set_ylim(-0.05, 1.05)
        
        # Formatting
        ax.set_ylabel(r'$p(\mathrm{spike})$')
        ax.axhline(p_baseline, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        ax.set_xlim(t_plot[0], t_plot[-1])
        
        if i == 0:
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.005), 
                     frameon=False, ncol=3, fontsize=10, 
                     handlelength=1.5, handletextpad=0.4, columnspacing=1.0)
    
    axes[-1].set_xlabel(r'Time (s)')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Make room for legend above
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {output_path}")
    
    return fig, axes


def load_inference_results(joint_path: str) -> dict:
    """
    Load inference results from either .pkl or .npz files.
    
    Supports:
    - joint.pkl (legacy format)
    - coupling.npz + spectral.npz (new format from runner.py)
    - results directory with coupling.npz
    
    IMPORTANT: spectral.npz contains Z_fine_joint which is needed for the latent!
    """
    joint_path = Path(joint_path)
    
    # Determine results directory
    if joint_path.is_dir():
        results_dir = joint_path
    elif joint_path.suffix in ['.npz', '.pkl']:
        results_dir = joint_path.parent
    else:
        results_dir = joint_path
    
    coupling_path = results_dir / 'coupling.npz'
    spectral_path = results_dir / 'spectral.npz'
    
    result = {}
    
    # Try loading from .npz files first (much faster)
    if coupling_path.exists():
        print(f"  Loading coupling.npz...")
        coupling = np.load(coupling_path, allow_pickle=True)
        for k in coupling.files:
            result[k] = coupling[k]
        print(f"    Keys: {list(coupling.files)}")
        
        # Wrap trace arrays in expected format
        if 'beta_trace' in result:
            result['trace'] = {
                'beta': result['beta_trace'],
                'gamma': result['gamma_trace'],
            }
    
    # CRITICAL: Load spectral.npz for the latent trajectory
    if spectral_path.exists():
        print(f"  Loading spectral.npz (contains latent trajectory)...")
        spectral = np.load(spectral_path, allow_pickle=True)
        for k in spectral.files:
            result[k] = spectral[k]
        print(f"    Keys: {list(spectral.files)}")
        
        # Check for required latent data
        if 'Z_fine_joint' in result:
            print(f"    Z_fine_joint shape: {result['Z_fine_joint'].shape}")
        elif 'Z_fine_em' in result:
            print(f"    Z_fine_em shape: {result['Z_fine_em'].shape}")
        else:
            print("    WARNING: No fine-resolution latent found in spectral.npz")
    else:
        print(f"  WARNING: spectral.npz not found at {spectral_path}")
        print(f"           Latent trajectory may not be available!")
    
    if result:
        return result
    
    # Fall back to pickle if no .npz files found
    pkl_path = results_dir / 'joint.pkl' if joint_path.is_dir() else joint_path
    if pkl_path.suffix == '.pkl' and pkl_path.exists():
        print(f"  Loading from pickle (this may take a while)...")
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    
    raise FileNotFoundError(f"Could not find inference results at {joint_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot posterior spike probability from joint inference results'
    )
    parser.add_argument('--sim', type=str, required=True,
                       help='Path to sim_data.pkl')
    parser.add_argument('--joint', type=str, required=True,
                       help='Path to joint.pkl, coupling.npz, or results directory')
    parser.add_argument('--output', type=str, default='posterior_spike_prob.pdf',
                       help='Output figure path')
    parser.add_argument('--t-start', type=float, default=2.0,
                       help='Start time for plot (seconds)')
    parser.add_argument('--t-duration', type=float, default=5.0,
                       help='Duration to plot (seconds)')
    parser.add_argument('--burn-in', type=float, default=0.6,
                       help='Burn-in fraction')
    parser.add_argument('--n-lags', type=int, default=20,
                       help='Number of history lags')
    parser.add_argument('--no-adaptive-ylim', action='store_true',
                       help='Use fixed 0-1 y-axis instead of adaptive zoom')
    parser.add_argument('--unit', type=int, default=0,
                       help='Unit index to plot (default: 0)')
    args = parser.parse_args()
    
    # Load simulation data
    print(f"Loading simulation data from: {args.sim}")
    with open(args.sim, 'rb') as f:
        sim_data = pickle.load(f)
    
    spikes = sim_data['spikes']  # (S, T_fine)
    delta_spk = sim_data['delta_spk']
    S, T_fine = spikes.shape
    print(f"  Spikes: {spikes.shape}, delta_spk: {delta_spk}")
    print(f"  Total duration: {T_fine * delta_spk:.1f} s")
    
    # Load joint inference results
    print(f"\nLoading joint results from: {args.joint}")
    joint_res = load_inference_results(args.joint)
    
    # Extract trace
    if isinstance(joint_res.get('trace'), dict):
        # Pickle format: trace is a dict with 'beta' and 'gamma' keys
        beta_trace = joint_res['trace']['beta']  # (n_samples, S, 1+2J)
        gamma_trace = joint_res['trace']['gamma']  # (n_samples, S, L) or (n_samples, S, R, L)
    else:
        # Trace object format
        trace = joint_res['trace']
        beta_trace = np.stack(trace.beta) if hasattr(trace, 'beta') else joint_res['trace']['beta']
        gamma_trace = np.stack(trace.gamma) if hasattr(trace, 'gamma') else joint_res['trace']['gamma']
    
    print(f"  β trace: {beta_trace.shape}")
    print(f"  γ trace: {gamma_trace.shape}")
    
    # Check for beta_standardized to determine if trace is standardized
    beta_standardized = joint_res.get('beta_standardized', None)
    if beta_standardized is not None:
        print(f"  β standardized (point est): {beta_standardized.shape}")
        print("  NOTE: trace['beta'] contains STANDARDIZED samples (matching standardized latent)")
    else:
        print("  NOTE: beta_standardized not found - assuming trace['beta'] is standardized")
    
    # Load additional metadata
    freqs = joint_res.get('freqs', None)
    latent_scale_factors = joint_res.get('latent_scale_factors', None)
    if freqs is not None:
        print(f"  Frequencies: {len(freqs)} bands ({freqs[0]:.0f}-{freqs[-1]:.0f} Hz)")
    if latent_scale_factors is not None:
        print(f"  Scale factors: {latent_scale_factors.shape}")
    
    # Get standardized latent
    # Priority: trace.latent[-1] > Z_fine_joint (reconstruct) > error
    lat_reim = None
    
    # Check for pre-computed latent in trace object (from pickle)
    if 'latent' in joint_res and joint_res['latent'] is not None:
        lat_list = joint_res['latent']
        if isinstance(lat_list, list) and len(lat_list) > 0:
            lat_reim = np.array(lat_list[-1])
            print(f"  Latent from joint_res['latent'][-1]: {lat_reim.shape}")
    
    # Fallback: check trace object attribute
    if lat_reim is None:
        trace_obj = joint_res.get('trace')
        if hasattr(trace_obj, 'latent') and hasattr(trace_obj.latent, '__len__') and len(trace_obj.latent) > 0:
            lat_reim = np.array(trace_obj.latent[-1])
            print(f"  Latent from trace.latent[-1]: {lat_reim.shape}")
    
    # Fallback: look for pre-computed standardized latent
    if lat_reim is None:
        for key in ['lat_reim', 'latent_reim', 'Z_standardized']:
            if key in joint_res and joint_res[key] is not None:
                lat_reim = np.array(joint_res[key])
                print(f"  Latent from joint_res['{key}']: {lat_reim.shape}")
                break
    
    # MAIN PATH FOR NPZ FILES: Reconstruct from Z_fine_joint
    if lat_reim is None:
        # Try Z_fine_joint first, then Z_fine_em
        Z_fine = None
        Z_fine_key = None
        for key in ['Z_fine_joint', 'Z_fine_em']:
            if key in joint_res and joint_res[key] is not None:
                Z_fine = joint_res[key]
                Z_fine_key = key
                break
        
        if Z_fine is None:
            raise ValueError(
                "Could not find latent trajectory in results!\n"
                "Expected one of: trace.latent, Z_fine_joint, Z_fine_em\n"
                "If using runner.py, make sure save_fine=True in output_config.\n"
                f"Available keys: {list(joint_res.keys())}"
            )
        
        print(f"\n  Reconstructing standardized latent from {Z_fine_key}...")
        print(f"    Input shape: {Z_fine.shape}")
        
        # Infer dimensions
        J = len(freqs) if freqs is not None else (beta_trace.shape[2] - 1) // 2
        total_cols = Z_fine.shape[2]
        M = total_cols // (2 * J)
        T_lat = Z_fine.shape[1]
        
        print(f"    Inferred: J={J} freqs, M={M} tapers, T={T_lat} time points")
        
        # Step 1: Convert to taper-averaged Re/Im format
        # Z_fine layout: interleaved [Re, Im] for each (j, m) pair
        # Output: (1, T, 2J) with [Re_j=0, ..., Re_j=J-1, Im_j=0, ..., Im_j=J-1]
        lat_reim = np.zeros((1, T_lat, 2*J), dtype=np.float64)
        
        for j in range(J):
            re_j = np.zeros(T_lat)
            im_j = np.zeros(T_lat)
            for m in range(M):
                col_re = 2 * (j * M + m)
                col_im = col_re + 1
                re_j += Z_fine[0, :, col_re]
                im_j += Z_fine[0, :, col_im]
            re_j /= M
            im_j /= M
            lat_reim[0, :, j] = re_j
            lat_reim[0, :, J + j] = im_j
        
        print(f"    After taper-averaging: {lat_reim.shape}")
        
        # Step 2: Apply phase rotation e^{+i omega_j t}
        if freqs is not None:
            print(f"    Applying phase rotation...")
            t = np.arange(T_lat) * delta_spk
            for j, f_j in enumerate(freqs):
                phi = 2.0 * np.pi * f_j * t
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                Re_orig = lat_reim[0, :, j].copy()
                Im_orig = lat_reim[0, :, J + j].copy()
                # e^{+i omega t} * (Re + i*Im) = (Re*cos - Im*sin) + i*(Re*sin + Im*cos)
                lat_reim[0, :, j] = Re_orig * cos_phi - Im_orig * sin_phi
                lat_reim[0, :, J + j] = Re_orig * sin_phi + Im_orig * cos_phi
        else:
            print("    WARNING: freqs not found, skipping phase rotation!")
        
        # Step 3: Apply standardization
        if latent_scale_factors is not None:
            print(f"    Applying standardization with stored scale factors...")
            for col in range(2*J):
                lat_reim[0, :, col] /= latent_scale_factors[col]
        else:
            # Estimate scale factors from amplitude RMS (same method as inference code)
            print(f"    Estimating scale factors from amplitude RMS...")
            estimated_scales = np.zeros(2*J)
            for j in range(J):
                re_j = lat_reim[0, :, j]
                im_j = lat_reim[0, :, J + j]
                amp_rms = np.sqrt(np.mean(re_j**2 + im_j**2))
                scale = max(amp_rms, 0.01)
                estimated_scales[j] = scale
                estimated_scales[J + j] = scale  # Same scale for Re and Im
                lat_reim[0, :, j] /= scale
                lat_reim[0, :, J + j] /= scale
            print(f"    Estimated scale range: [{estimated_scales.min():.4f}, {estimated_scales.max():.4f}]")
        
        print(f"    Final latent shape: {lat_reim.shape}")
    
    # Ensure correct shape (R, T, 2J)
    if lat_reim.ndim == 2:
        lat_reim = lat_reim[None, :, :]
    
    print(f"  Final latent shape: {lat_reim.shape}")
    
    # Compute posterior spike probability
    print("\nComputing posterior spike probability...")
    result = compute_posterior_spike_probability(
        beta_trace=beta_trace,
        gamma_trace=gamma_trace,
        lat_reim=lat_reim,
        spikes=spikes,
        n_lags=args.n_lags,
        burn_in_frac=args.burn_in,
        verbose=True,
    )
    
    # Plot
    print(f"\nGenerating plot (t={args.t_start:.1f}s to {args.t_start + args.t_duration:.1f}s)...")
    
    # Print modulation diagnostics
    print("\n=== Modulation Diagnostics ===")
    for s in range(result['prob_mean'].shape[0]):
        p = result['prob_mean'][s]
        p_min, p_max, p_mean = p.min(), p.max(), p.mean()
        mod_pct = 100 * (p_max - p_min) / p_mean if p_mean > 0 else 0
        print(f"  Unit {s}: p(spike) in [{p_min:.4f}, {p_max:.4f}], "
              f"baseline={p_mean:.4f}, modulation={mod_pct:.1f}%")
    
    T_use = result['T_use']
    fig, axes = plot_posterior_spike_probability(
        prob_mean=result['prob_mean'],
        prob_lower=result['prob_lower'],
        prob_upper=result['prob_upper'],
        spikes=spikes[:, :T_use],
        delta_spk=delta_spk,
        t_start=args.t_start,
        t_duration=args.t_duration,
        output_path=args.output,
        adaptive_ylim=not args.no_adaptive_ylim,
        unit_idx=args.unit,
    )
    
    # Also save the computed probabilities for later use
    npz_path = args.output.replace('.pdf', '.npz').replace('.png', '.npz')
    np.savez_compressed(
        npz_path,
        prob_mean=result['prob_mean'],
        prob_lower=result['prob_lower'],
        prob_upper=result['prob_upper'],
        psi_mean=result['psi_mean'],
        delta_spk=delta_spk,
    )
    print(f"Saved probabilities to: {npz_path}")
    
    print("\nDone!")
    plt.show()


if __name__ == '__main__':
    main()