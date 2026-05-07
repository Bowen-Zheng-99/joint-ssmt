#!/usr/bin/env python3
"""
Compute traditional spike-field coupling measures (PLV, SFC) with both
permutation tests and parametric tests.

Usage:
    python compute_traditional_methods.py --data ./data/sim_with_trials.pkl --output ./results/traditional_methods.pkl

    # Or compute only one method:
    python compute_traditional_methods.py --data ./data/sim_with_trials.pkl --output ./results/plv_results.pkl --method plv
    python compute_traditional_methods.py --data ./data/sim_with_trials.pkl --output ./results/sfc_results.pkl --method sfc
    
    # Skip permutation tests (faster, parametric only):
    python compute_traditional_methods.py --data ./data/sim_with_trials.pkl --output ./results/traditional_methods.pkl --skip_permutation
"""

import os
import sys
import pickle
import argparse
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, csd, welch
from scipy.stats import chi2
from simulate_trial_data import TrialSimConfig
# Flush print to ensure SLURM output
def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)
    sys.stdout.flush()

# Redirect uncaught exception tracebacks to stderr
import traceback

def exception_hook(exc_type, exc_value, exc_tb):
    traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
    sys.stderr.flush()
    sys.stdout.flush()

sys.excepthook = exception_hook

# =============================================================================
# Configuration (must match inference parameters)
# =============================================================================
FS = 1000.0              # Sampling rate
WINDOW_SEC = 0.4         # Window duration (400ms)
NW_PRODUCT = 1           # Time-bandwidth product
FREQS_DENSE = np.arange(1, 61, 2, dtype=float)  # [1, 3, 5, ..., 59] Hz
BANDWIDTH = 2 * NW_PRODUCT / WINDOW_SEC  # Hz (for bandpass filter)


# =============================================================================
# Parametric Tests
# =============================================================================
def rayleigh_pval(plv, n_spikes):
    """
    Rayleigh test p-value for PLV.
    
    Tests whether spike phases are uniformly distributed (null) vs 
    concentrated around a preferred phase (alternative).
    
    Under null hypothesis (uniform phases):
        2 * n * PLV^2 ~ chi2(df=2)
    
    Parameters
    ----------
    plv : float or array
        Phase-locking value(s)
    n_spikes : int or array
        Number of spikes used to compute PLV
    
    Returns
    -------
    pval : float or array
        P-value from Rayleigh test
    """
    # Avoid division by zero
    n_spikes = np.maximum(n_spikes, 1)
    z = 2 * n_spikes * (plv ** 2)
    pval = chi2.sf(z, df=2)
    return pval


def coherence_pval(coh, n_independent):
    """
    Parametric p-value for coherence.
    
    Based on the null distribution of coherence when signals are independent.
    
    For L independent spectral estimates:
        P(C > c | null) = (1 - c^2)^(L-1)
    
    Parameters
    ----------
    coh : float or array
        Coherence value(s) in [0, 1]
    n_independent : int
        Number of independent spectral estimates (n_trials * n_segments)
    
    Returns
    -------
    pval : float or array
        P-value for coherence
    """
    # Clip coherence to valid range
    coh = np.clip(coh, 0, 0.9999)
    L = max(n_independent, 2)
    pval = (1 - coh ** 2) ** (L - 1)
    return pval


# =============================================================================
# PLV Computation
# =============================================================================
def compute_plv_all(lfp, spikes, freqs, fs, bandwidth, 
                    n_permutations=500, seed=42, skip_permutation=False):
    """
    Compute PLV with both Rayleigh test (parametric) and circular shift 
    permutation test for significance.
    
    Parameters
    ----------
    lfp : (R, T) array
        LFP signals (R trials, T timepoints)
    spikes : (R, S, T_fine) array
        Spike trains (binary, at finer resolution)
    freqs : (B,) array
        Frequencies to analyze
    fs : float
        Sampling rate of LFP
    bandwidth : float
        Bandwidth for bandpass filter (Hz)
    n_permutations : int
        Number of permutations for significance test
    seed : int
        Random seed for reproducibility
    skip_permutation : bool
        If True, skip permutation test (faster)
    
    Returns
    -------
    results : dict
        'values': (S, B) PLV values
        'pval_parametric': (S, B) Rayleigh test p-values
        'pval_permutation': (S, B) Permutation test p-values (if computed)
        'phase': (S, B) Preferred phase (circular mean)
        'n_spikes': (S, B) Number of spikes per unit/frequency
    """
    R, T = lfp.shape
    _, S, T_fine = spikes.shape
    B = len(freqs)
    
    ds = T_fine // T
    nyq = fs / 2
    
    plv = np.zeros((S, B))
    plv_pval_parametric = np.ones((S, B))
    plv_pval_permutation = np.ones((S, B))
    preferred_phase = np.zeros((S, B))
    n_spikes_arr = np.zeros((S, B), dtype=int)
    
    total_combinations = S * B
    completed = 0
    
    for j, f in enumerate(freqs):
        # Design bandpass filter
        low = max(f - bandwidth / 2, 0.5)
        high = min(f + bandwidth / 2, nyq - 1)
        
        if low >= high or high >= nyq:
            completed += S
            continue
        
        try:
            b, a = butter(2, [low / nyq, high / nyq], btype='band')
        except ValueError:
            completed += S
            continue
        
        # Precompute LFP phases for all trials
        lfp_phases = np.zeros((R, T))
        for r in range(R):
            lfp_filt = filtfilt(b, a, lfp[r])
            lfp_phases[r] = np.angle(hilbert(lfp_filt))
        
        for s in range(S):
            completed += 1
            
            # Progress logging
            if completed % max(1, total_combinations // 20) == 0 or (j == 0 and s == 0):
                flush_print(f"  PLV: freq {j+1}/{B} ({f:.0f} Hz), unit {s+1}/{S} "
                            f"[{100*completed/total_combinations:.0f}%]")
            
            # Get spike times and phases (observed) - pooled across trials
            all_phases = []
            
            for r in range(R):
                spk_fine = spikes[r, s, :]
                spk_coarse = spk_fine.reshape(-1, ds).max(axis=1)
                spike_idx = np.where(spk_coarse > 0)[0]
                
                if len(spike_idx) > 0:
                    all_phases.extend(lfp_phases[r, spike_idx])
            
            all_phases = np.array(all_phases)
            n_spikes = len(all_phases)
            n_spikes_arr[s, j] = n_spikes
            
            if n_spikes < 5:  # Need minimum spikes
                continue
            
            # Observed PLV
            z_obs = np.exp(1j * all_phases)
            plv[s, j] = np.abs(z_obs.mean())
            preferred_phase[s, j] = np.angle(z_obs.mean())
            
            # Parametric test: Rayleigh test
            plv_pval_parametric[s, j] = rayleigh_pval(plv[s, j], n_spikes)
            
            # Permutation test: circular shift spikes relative to LFP
            if not skip_permutation:
                null_plv = np.zeros(n_permutations)
                rng = np.random.default_rng(seed=seed + s * B + j)
                
                for perm in range(n_permutations):
                    perm_phases = []
                    for r in range(R):
                        spk_fine = spikes[r, s, :]
                        # Circular shift by random amount
                        shift = rng.integers(T_fine // 4, 3 * T_fine // 4)
                        spk_shifted = np.roll(spk_fine, shift)
                        spk_coarse = spk_shifted.reshape(-1, ds).max(axis=1)
                        spike_idx = np.where(spk_coarse > 0)[0]
                        
                        if len(spike_idx) > 0:
                            perm_phases.extend(lfp_phases[r, spike_idx])
                    
                    if len(perm_phases) > 0:
                        null_plv[perm] = np.abs(np.exp(1j * np.array(perm_phases)).mean())
                
                # P-value: proportion of null >= observed
                plv_pval_permutation[s, j] = (np.sum(null_plv >= plv[s, j]) + 1) / (n_permutations + 1)
    
    flush_print()
    
    results = {
        'values': plv,
        'pval_parametric': plv_pval_parametric,
        'phase': preferred_phase,
        'n_spikes': n_spikes_arr,
    }
    
    if not skip_permutation:
        results['pval_permutation'] = plv_pval_permutation
    
    return results


# =============================================================================
# SFC Computation
# =============================================================================
def compute_sfc_all(lfp, spikes, freqs, fs, window_sec, 
                    n_permutations=500, seed=42, skip_permutation=False):
    """
    Compute SFC with both parametric test and circular shift permutation test.
    
    Parameters
    ----------
    lfp : (R, T) array
    spikes : (R, S, T_fine) array
    freqs : (B,) array
    fs : float
    window_sec : float
    n_permutations : int
    seed : int
    skip_permutation : bool
        If True, skip permutation test (faster)
    
    Returns
    -------
    results : dict
        'values': (S, B) Coherence values
        'pval_parametric': (S, B) Parametric p-values
        'pval_permutation': (S, B) Permutation p-values (if computed)
        'n_independent': int - number of independent spectral estimates
    """
    R, T = lfp.shape
    _, S, T_fine = spikes.shape
    B = len(freqs)
    
    ds = T_fine // T
    spikes_ds = spikes.reshape(R, S, T, ds).mean(axis=-1)
    
    nperseg = int(window_sec * fs)
    noverlap = nperseg // 2
    
    # Calculate number of independent spectral estimates
    # n_segments per trial * n_trials
    n_segments_per_trial = max(1, (T - noverlap) // (nperseg - noverlap))
    n_independent = R * n_segments_per_trial
    
    sfc = np.zeros((S, B))
    sfc_pval_parametric = np.ones((S, B))
    sfc_pval_permutation = np.ones((S, B))
    
    def compute_coherence(x_trials, y_trials):
        """Compute coherence pooled across trials."""
        Pxy_sum = np.zeros(B, dtype=complex)
        Pxx_sum = np.zeros(B)
        Pyy_sum = np.zeros(B)
        
        for r in range(R):
            f_csd, Pxy = csd(x_trials[r], y_trials[r], fs=fs, nperseg=nperseg, noverlap=noverlap)
            _, Pxx = welch(x_trials[r], fs=fs, nperseg=nperseg, noverlap=noverlap)
            _, Pyy = welch(y_trials[r], fs=fs, nperseg=nperseg, noverlap=noverlap)
            
            Pxy_sum += np.interp(freqs, f_csd, Pxy)
            Pxx_sum += np.interp(freqs, f_csd, Pxx)
            Pyy_sum += np.interp(freqs, f_csd, Pyy)
        
        denom = np.sqrt(Pxx_sum * Pyy_sum)
        coh = np.zeros(B)
        valid = denom > 1e-10
        coh[valid] = np.abs(Pxy_sum[valid]) / denom[valid]
        return coh
    
    total_perms = S * n_permutations if not skip_permutation else S
    completed_perms = 0
    
    for s in range(S):
        flush_print(f"  SFC: unit {s+1}/{S}, computing observed coherence...")
        
        # Observed SFC
        sfc[s] = compute_coherence(lfp, spikes_ds[:, s, :])
        
        # Parametric p-value
        sfc_pval_parametric[s] = coherence_pval(sfc[s], n_independent)
        
        # Permutation test
        if not skip_permutation:
            null_sfc = np.zeros((n_permutations, B))
            rng = np.random.default_rng(seed=seed + s)
            
            for perm in range(n_permutations):
                completed_perms += 1
                
                # Progress logging every 5%
                if perm % max(1, n_permutations // 20) == 0:
                    flush_print(f"  SFC: unit {s+1}/{S}, permutation {perm+1}/{n_permutations} "
                                f"[{100*completed_perms/total_perms:.0f}%]")
                
                # Circular shift spikes for each trial
                spikes_perm = np.zeros_like(spikes_ds[:, s, :])
                for r in range(R):
                    shift = rng.integers(T // 4, 3 * T // 4)
                    spikes_perm[r] = np.roll(spikes_ds[r, s, :], shift)
                
                null_sfc[perm] = compute_coherence(lfp, spikes_perm)
            
            # P-value per frequency
            for j in range(B):
                sfc_pval_permutation[s, j] = (np.sum(null_sfc[:, j] >= sfc[s, j]) + 1) / (n_permutations + 1)
    
    flush_print()
    
    results = {
        'values': sfc,
        'pval_parametric': sfc_pval_parametric,
        'n_independent': n_independent,
    }
    
    if not skip_permutation:
        results['pval_permutation'] = sfc_pval_permutation
    
    return results


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Compute PLV and SFC with parametric and permutation tests'
    )
    parser.add_argument('--data', type=str, required=True,
                        help='Input path for simulated data')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for results')
    parser.add_argument('--method', type=str, default='both',
                        choices=['plv', 'sfc', 'both'],
                        help='Which method to compute (default: both)')
    parser.add_argument('--n_permutations', type=int, default=500,
                        help='Number of permutations (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--skip_permutation', action='store_true',
                        help='Skip permutation test (faster, parametric only)')
    
    args = parser.parse_args()
    
    # Load data
    flush_print(f"Loading data from {args.data}")
    with open(args.data, 'rb') as f:
        data = pickle.load(f)
    
    lfp = data['LFP']
    spikes = data['spikes']
    
    R, T = lfp.shape
    _, S, T_fine = spikes.shape
    
    flush_print(f"  LFP: {lfp.shape} ({R} trials, {T} timepoints)")
    flush_print(f"  Spikes: {spikes.shape} ({R} trials, {S} units)")
    flush_print(f"  Frequencies: {len(FREQS_DENSE)} from {FREQS_DENSE[0]} to {FREQS_DENSE[-1]} Hz")
    flush_print(f"  Permutations: {args.n_permutations}" + (" (SKIPPED)" if args.skip_permutation else ""))
    
    results = {
        'config': {
            'freqs': FREQS_DENSE,
            'fs': FS,
            'window_sec': WINDOW_SEC,
            'bandwidth': BANDWIDTH,
            'nw_product': NW_PRODUCT,
            'n_permutations': args.n_permutations,
            'seed': args.seed,
            'skip_permutation': args.skip_permutation,
        }
    }
    
    # Compute PLV
    if args.method in ['plv', 'both']:
        flush_print("\n" + "="*60)
        flush_print("Computing PLV with Rayleigh test" + 
                    (" and permutation test..." if not args.skip_permutation else "..."))
        flush_print("="*60)
        
        plv_results = compute_plv_all(
            lfp, spikes, FREQS_DENSE, FS, BANDWIDTH,
            n_permutations=args.n_permutations, 
            seed=args.seed,
            skip_permutation=args.skip_permutation
        )
        
        results['plv'] = plv_results
        
        flush_print(f"\n  PLV Results:")
        flush_print(f"    Shape: {plv_results['values'].shape}")
        flush_print(f"    Range: [{plv_results['values'].min():.4f}, {plv_results['values'].max():.4f}]")
        flush_print(f"    Significant (parametric, p<0.05): {(plv_results['pval_parametric'] < 0.05).sum()} / {plv_results['pval_parametric'].size}")
        if not args.skip_permutation:
            flush_print(f"    Significant (permutation, p<0.05): {(plv_results['pval_permutation'] < 0.05).sum()} / {plv_results['pval_permutation'].size}")
    
    # Compute SFC
    if args.method in ['sfc', 'both']:
        flush_print("\n" + "="*60)
        flush_print("Computing SFC with parametric test" + 
                    (" and permutation test..." if not args.skip_permutation else "..."))
        flush_print("="*60)
        
        sfc_results = compute_sfc_all(
            lfp, spikes, FREQS_DENSE, FS, WINDOW_SEC,
            n_permutations=args.n_permutations, 
            seed=args.seed,
            skip_permutation=args.skip_permutation
        )
        
        results['sfc'] = sfc_results
        
        flush_print(f"\n  SFC Results:")
        flush_print(f"    Shape: {sfc_results['values'].shape}")
        flush_print(f"    Range: [{sfc_results['values'].min():.4f}, {sfc_results['values'].max():.4f}]")
        flush_print(f"    Independent estimates: {sfc_results['n_independent']}")
        flush_print(f"    Significant (parametric, p<0.05): {(sfc_results['pval_parametric'] < 0.05).sum()} / {sfc_results['pval_parametric'].size}")
        if not args.skip_permutation:
            flush_print(f"    Significant (permutation, p<0.05): {(sfc_results['pval_permutation'] < 0.05).sum()} / {sfc_results['pval_permutation'].size}")
    
    # Save results
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    
    flush_print("\n" + "="*60)
    flush_print(f"Results saved to {args.output}")
    flush_print("="*60)
    
    # Summary comparison if both tests were run
    if not args.skip_permutation:
        flush_print("\n" + "="*60)
        flush_print("Summary: Parametric vs Permutation Agreement")
        flush_print("="*60)
        
        if 'plv' in results:
            plv_para = results['plv']['pval_parametric'] < 0.05
            plv_perm = results['plv']['pval_permutation'] < 0.05
            plv_agree = (plv_para == plv_perm).mean() * 100
            flush_print(f"  PLV agreement: {plv_agree:.1f}%")
        
        if 'sfc' in results:
            sfc_para = results['sfc']['pval_parametric'] < 0.05
            sfc_perm = results['sfc']['pval_permutation'] < 0.05
            sfc_agree = (sfc_para == sfc_perm).mean() * 100
            flush_print(f"  SFC agreement: {sfc_agree:.1f}%")


if __name__ == '__main__':
    main()