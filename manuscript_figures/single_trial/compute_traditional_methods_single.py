#!/usr/bin/env python3
"""
Compute traditional spike-field coupling measures (PLV, SFC) for single-trial data.

Supports two statistical testing strategies:
  - parametric:  PLV uses Rayleigh test; SFC uses coherence significance from DOF
  - permutation: Both use circular-shift surrogate tests

Usage:
    # Both test types (default) - saves parametric and permutation p-values
    python compute_traditional_methods_single.py --data ./data/sim.pkl \
        --output ./results/traditional_methods.pkl

    # Only parametric (fast)
    python compute_traditional_methods_single.py --data ./data/sim.pkl \
        --output ./results/traditional_methods.pkl --test_type parametric

    # Only permutation
    python compute_traditional_methods_single.py --data ./data/sim.pkl \
        --output ./results/traditional_methods.pkl --test_type permutation \
        --n_permutations 500
"""

import os
import pickle
import argparse
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, csd, welch


# =============================================================================
# Configuration
# =============================================================================
FS = 1000.0
WINDOW_SEC = 2
NW_PRODUCT = 2
FREQS_DENSE = np.arange(1, 61, 1, dtype=float)
BANDWIDTH = 2 * NW_PRODUCT / WINDOW_SEC


# =============================================================================
# Parametric helpers
# =============================================================================
def rayleigh_test(phases):
    """
    Rayleigh test for circular non-uniformity.

    Tests H0: phases are uniformly distributed on the circle.

    Returns (R, p_value, mean_phase).
    """
    n = len(phases)
    if n < 2:
        return 0.0, 1.0, 0.0

    z = np.exp(1j * phases)
    R = np.abs(z.mean())
    mean_phase = np.angle(z.mean())

    Z = n * R**2
    if n >= 50:
        p_value = np.exp(-Z)
    else:
        p_value = np.exp(-Z) * (1 + (2*Z - Z**2)/(4*n) -
                                (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4)/(288*n**2))

    return R, np.clip(p_value, 1e-300, 1.0), mean_phase


def coherence_pvalue(coh, n_segments):
    """
    Parametric p-value for coherence: P(|C|^2 > c) = (1 - c)^(n-1).
    """
    coh_sq = np.clip(np.asarray(coh)**2, 0, 0.9999)
    n = max(n_segments, 2)
    return np.clip((1 - coh_sq)**(n - 1), 1e-300, 1.0)


# =============================================================================
# PLV
# =============================================================================
def _bandpass_phase(lfp, freq, bandwidth, fs):
    """Return instantaneous phase of band-passed LFP, or None if filter invalid."""
    nyq = fs / 2
    low = max(freq - bandwidth / 2, 0.5)
    high = min(freq + bandwidth / 2, nyq - 1)
    if low >= high or high >= nyq:
        return None
    try:
        b, a = butter(2, [low / nyq, high / nyq], btype='band')
    except ValueError:
        return None
    return np.angle(hilbert(filtfilt(b, a, lfp)))


def compute_plv(lfp, spikes, freqs, fs, bandwidth,
                do_parametric=True, do_permutation=True,
                n_permutations=500, seed=42):
    """
    Compute PLV with parametric (Rayleigh) and/or permutation p-values.

    Returns dict with keys: values, phase, n_spikes,
    and pval_parametric / pval_permutation depending on flags.
    """
    T = len(lfp)
    S, T_fine = spikes.shape
    B = len(freqs)
    ds = T_fine // T

    plv = np.zeros((S, B))
    phase_out = np.zeros((S, B))
    n_spikes_out = np.zeros((S, B), dtype=int)
    pval_para = np.ones((S, B)) if do_parametric else None
    pval_perm = np.ones((S, B)) if do_permutation else None

    for j, f in enumerate(freqs):
        lfp_phase = _bandpass_phase(lfp, f, bandwidth, fs)
        if lfp_phase is None:
            continue

        for s in range(S):
            spk_fine = spikes[s]
            spk_coarse = spk_fine.reshape(-1, ds).max(axis=1)
            spike_idx = np.where(spk_coarse > 0)[0]
            n_spk = len(spike_idx)
            n_spikes_out[s, j] = n_spk

            if n_spk < 5:
                continue

            spike_phases = lfp_phase[spike_idx]
            z_obs = np.exp(1j * spike_phases)
            plv[s, j] = np.abs(z_obs.mean())
            phase_out[s, j] = np.angle(z_obs.mean())

            # Parametric
            if do_parametric:
                _, p, _ = rayleigh_test(spike_phases)
                pval_para[s, j] = p

            # Permutation
            if do_permutation:
                rng = np.random.default_rng(seed=seed + s * B + j)
                null_plv = np.zeros(n_permutations)
                for perm in range(n_permutations):
                    shift = rng.integers(T_fine // 4, 3 * T_fine // 4)
                    spk_shifted = np.roll(spk_fine, shift)
                    spk_coarse_perm = spk_shifted.reshape(-1, ds).max(axis=1)
                    idx_perm = np.where(spk_coarse_perm > 0)[0]
                    if len(idx_perm) > 0:
                        null_plv[perm] = np.abs(np.exp(1j * lfp_phase[idx_perm]).mean())
                pval_perm[s, j] = (np.sum(null_plv >= plv[s, j]) + 1) / (n_permutations + 1)

    out = {'values': plv, 'phase': phase_out, 'n_spikes': n_spikes_out}
    if do_parametric:
        out['pval_parametric'] = pval_para
    if do_permutation:
        out['pval_permutation'] = pval_perm
    return out


# =============================================================================
# SFC
# =============================================================================
def compute_sfc(lfp, spikes, freqs, fs, window_sec,
                do_parametric=True, do_permutation=True,
                n_permutations=500, seed=42):
    """
    Compute SFC (magnitude-squared coherence) with parametric and/or permutation p-values.

    Returns dict with keys: values, n_segments,
    and pval_parametric / pval_permutation depending on flags.
    """
    T = len(lfp)
    S, T_fine = spikes.shape
    B = len(freqs)
    ds = T_fine // T
    nperseg = int(window_sec * fs)
    noverlap = nperseg // 2
    n_segments = max(1, int((T - noverlap) / (nperseg - noverlap)))

    spikes_ds = spikes.reshape(S, T, ds).mean(axis=-1)

    sfc = np.zeros((S, B))
    pval_para = np.ones((S, B)) if do_parametric else None
    pval_perm = np.ones((S, B)) if do_permutation else None

    def _coherence(x, y):
        f_csd, Pxy = csd(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap)
        _, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
        _, Pyy = welch(y, fs=fs, nperseg=nperseg, noverlap=noverlap)
        Pxy_i = np.interp(freqs, f_csd, Pxy)
        Pxx_i = np.interp(freqs, f_csd, Pxx)
        Pyy_i = np.interp(freqs, f_csd, Pyy)
        denom = np.sqrt(Pxx_i * Pyy_i)
        coh = np.zeros(B)
        valid = denom > 1e-10
        coh[valid] = np.abs(Pxy_i[valid]) / denom[valid]
        return coh

    for s in range(S):
        print(f"  SFC: unit {s+1}/{S}")
        sfc[s] = _coherence(lfp, spikes_ds[s])

        if do_parametric:
            pval_para[s] = coherence_pvalue(sfc[s], n_segments)

        if do_permutation:
            rng = np.random.default_rng(seed=seed + s)
            null_sfc = np.zeros((n_permutations, B))
            for perm in range(n_permutations):
                shift = rng.integers(T // 4, 3 * T // 4)
                null_sfc[perm] = _coherence(lfp, np.roll(spikes_ds[s], shift))
            for j in range(B):
                pval_perm[s, j] = (np.sum(null_sfc[:, j] >= sfc[s, j]) + 1) / (n_permutations + 1)

    out = {'values': sfc, 'n_segments': n_segments}
    if do_parametric:
        out['pval_parametric'] = pval_para
    if do_permutation:
        out['pval_permutation'] = pval_perm
    return out


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Compute PLV and SFC for single-trial data'
    )
    parser.add_argument('--data', type=str, required=True,
                        help='Input path for simulated data')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for results')
    parser.add_argument('--method', type=str, default='both',
                        choices=['plv', 'sfc', 'both'],
                        help='Which measure to compute (default: both)')
    parser.add_argument('--test_type', type=str, default='both',
                        choices=['parametric', 'permutation', 'both'],
                        help='Statistical test type (default: both)')
    parser.add_argument('--n_permutations', type=int, default=500,
                        help='Number of permutations (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    do_para = args.test_type in ('parametric', 'both')
    do_perm = args.test_type in ('permutation', 'both')

    # Load data
    print(f"Loading data from {args.data}")
    with open(args.data, 'rb') as f:
        data = pickle.load(f)

    lfp = data['LFP']
    spikes = data['spikes']
    T = len(lfp)
    S, T_fine = spikes.shape

    print(f"  LFP: ({T},)")
    print(f"  Spikes: {spikes.shape}")
    print(f"  Frequencies: {len(FREQS_DENSE)} from {FREQS_DENSE[0]} to {FREQS_DENSE[-1]} Hz")
    print(f"  Test type: {args.test_type}")
    if do_perm:
        print(f"  Permutations: {args.n_permutations}")

    results = {
        'config': {
            'freqs': FREQS_DENSE,
            'fs': FS,
            'window_sec': WINDOW_SEC,
            'bandwidth': BANDWIDTH,
            'nw_product': NW_PRODUCT,
            'test_type': args.test_type,
            'n_permutations': args.n_permutations if do_perm else None,
            'seed': args.seed,
        }
    }

    # --- PLV ---
    if args.method in ('plv', 'both'):
        print(f"\n{'='*60}")
        print("Computing PLV...")
        print(f"{'='*60}")

        plv_out = compute_plv(
            lfp, spikes, FREQS_DENSE, FS, BANDWIDTH,
            do_parametric=do_para, do_permutation=do_perm,
            n_permutations=args.n_permutations, seed=args.seed,
        )

        # Default 'pval' = parametric when available, else permutation
        if do_para:
            plv_out['pval'] = plv_out['pval_parametric']
        else:
            plv_out['pval'] = plv_out['pval_permutation']

        results['plv'] = plv_out

        v = plv_out['values']
        print(f"  PLV shape: {v.shape}")
        print(f"  PLV range: [{v.min():.4f}, {v.max():.4f}]")
        if do_para:
            print(f"  Significant (parametric, p<0.05): "
                  f"{(plv_out['pval_parametric'] < 0.05).sum()} / {v.size}")
        if do_perm:
            print(f"  Significant (permutation, p<0.05): "
                  f"{(plv_out['pval_permutation'] < 0.05).sum()} / {v.size}")
        print(f"  Spike counts: min={plv_out['n_spikes'].min()}, "
              f"max={plv_out['n_spikes'].max()}, mean={plv_out['n_spikes'].mean():.1f}")

    # --- SFC ---
    if args.method in ('sfc', 'both'):
        print(f"\n{'='*60}")
        print("Computing SFC...")
        print(f"{'='*60}")

        sfc_out = compute_sfc(
            lfp, spikes, FREQS_DENSE, FS, WINDOW_SEC,
            do_parametric=do_para, do_permutation=do_perm,
            n_permutations=args.n_permutations, seed=args.seed,
        )

        if do_para:
            sfc_out['pval'] = sfc_out['pval_parametric']
        else:
            sfc_out['pval'] = sfc_out['pval_permutation']

        results['sfc'] = sfc_out

        v = sfc_out['values']
        print(f"  SFC shape: {v.shape}")
        print(f"  SFC range: [{v.min():.4f}, {v.max():.4f}]")
        if do_para:
            print(f"  Significant (parametric, p<0.05): "
                  f"{(sfc_out['pval_parametric'] < 0.05).sum()} / {v.size}")
        if do_perm:
            print(f"  Significant (permutation, p<0.05): "
                  f"{(sfc_out['pval_permutation'] < 0.05).sum()} / {v.size}")
        print(f"  Number of segments: {sfc_out['n_segments']}")

    results['freqs'] = FREQS_DENSE

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n{'='*60}")
    print(f"Results saved to {args.output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
