#!/usr/bin/env python3
"""
run_analysis.py - Complete analysis using modular API

Matches the output of:
- compare_coupling_methods_single_trial.py
- compare_spectral_dynamics.py  
- plot_beta_posterior.py

Usage:
    python run_analysis.py \
        --data ./data/sim.pkl \
        --output ./results/ \
        --traditional ./traditional_methods.pkl
"""

import os
import pickle
import argparse
import numpy as np
# I/O
from joint_ssmt.io import run_inference, load_results, results_to_legacy_dict

# Statistical tests (from analysis or plotting - same file)
from joint_ssmt.analysis import (
    wald_test,
    phase_concentration_test,
    compute_detection_metrics,
    compute_roc_auc,
    compute_phase_mae,
)

# Heatmaps
from joint_ssmt.plotting.heatmaps import (
    set_publication_style,
    plot_effect_comparison,
    plot_pval_comparison,
)

# Scatter and metrics
from joint_ssmt.plotting.scatter_metrics import (
    plot_magnitude_scatter,
    plot_phase_recovery,
    plot_metrics_bars,
    plot_roc_pr_curves,
)

# Beta posterior
from joint_ssmt.plotting.beta_posterior import (
    plot_beta_posterior_scatter,
)

# Spectral dynamics
from joint_ssmt.plotting.spectral_dynamics import (
    generate_spectral_dynamics_figures,
)


def main():
    parser = argparse.ArgumentParser(description='Run spike-field coupling analysis')
    parser.add_argument('--data', type=str, required=True, help='Path to simulation data (.pkl)')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--traditional', type=str, default=None, help='Path to PLV/SFC results')
    
    # CT-SSMT config
    parser.add_argument('--freq_min', type=float, default=1.0)
    parser.add_argument('--freq_max', type=float, default=61.0)
    parser.add_argument('--freq_step', type=float, default=1.0)
    parser.add_argument('--window_sec', type=float, default=2.0)
    parser.add_argument('--NW', type=float, default=2.0)
    
    # MCMC config
    parser.add_argument('--fixed_iter', type=int, default=1000)
    parser.add_argument('--n_refreshes', type=int, default=5)
    parser.add_argument('--inner_steps', type=int, default=200)
    parser.add_argument('--no_shrinkage', action='store_true')
    
    # Skip inference (use existing results)
    parser.add_argument('--skip_inference', action='store_true', help='Skip inference, use existing results')
    
    args = parser.parse_args()
    
    ALPHA = 0.05
    BURNIN = 0.6
    
    # =========================================================================
    # 1. LOAD SIMULATION DATA
    # =========================================================================
    print("=" * 60)
    print("1. LOADING DATA")
    print("=" * 60)
    
    with open(args.data, 'rb') as f:
        sim_data = pickle.load(f)
    
    lfp = sim_data['LFP']
    spikes = sim_data['spikes']
    fs = sim_data.get('fs', 1000.0)
    delta_spk = sim_data.get('delta_spk', 0.001)
    
    # Ground truth
    freqs_true = np.asarray(sim_data['freqs_hz'], dtype=float)
    masks = np.asarray(sim_data['masks'], dtype=bool)
    S, J_true = masks.shape
    
    if 'beta_mag' in sim_data:
        gt_mag = np.asarray(sim_data['beta_mag'])
        gt_phase = np.asarray(sim_data['beta_phase'])
    else:
        beta_true = np.asarray(sim_data['beta_true'])
        bR = beta_true[:, 1:1+J_true]
        bI = beta_true[:, 1+J_true:1+2*J_true]
        gt_mag = np.sqrt(bR**2 + bI**2)
        gt_phase = np.arctan2(bI, bR)
    
    print(f"  LFP: {lfp.shape}, Spikes: {spikes.shape}")
    print(f"  Units: {S}, True freqs: {freqs_true}")
    print(f"  Coupled pairs: {masks.sum()}, Uncoupled: {(~masks).sum()}")
    
    # =========================================================================
    # 2. RUN JOINT INFERENCE (or load existing)
    # =========================================================================
    results_dir = os.path.join(args.output, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    if args.skip_inference and os.path.exists(os.path.join(results_dir, 'coupling.npz')):
        print("\n" + "=" * 60)
        print("2. LOADING EXISTING RESULTS")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("2. RUNNING JOINT INFERENCE")
        print("=" * 60)
        
        run_inference(
            lfp=lfp,
            spikes=spikes,
            ctssmt_config={
                'freq_min': args.freq_min,
                'freq_max': args.freq_max,
                'freq_step': args.freq_step,
                'window_sec': args.window_sec,
                'NW': args.NW,
            },
            mcmc_config={
                'fixed_iter': args.fixed_iter,
                'n_refreshes': args.n_refreshes,
                'inner_steps': args.inner_steps,
                
                'use_shrinkage': not args.no_shrinkage,
            },
            output_config={
                'output_dir': results_dir,
                'save_spectral': True,
                'save_fine': True,
            },
            fs=fs,
            delta_spk=delta_spk,
        )
    
    # Load results and convert to legacy format
    results = load_results(results_dir)
    coupling = results['coupling']
    
    freqs = coupling['freqs']
    J = len(freqs)
    beta_trace = coupling['beta_trace']
    beta_mag = coupling['beta_mag']
    beta_phase = coupling['beta_phase']
    
    # Also save legacy format
    legacy = results_to_legacy_dict(results_dir)
    legacy_path = os.path.join(results_dir, 'joint.pkl')
    with open(legacy_path, 'wb') as f:
        pickle.dump(legacy, f)
    print(f"  Saved legacy format: {legacy_path}")
    
    # Map true frequencies to analysis grid
    idx_map = np.array([np.argmin(np.abs(freqs - f)) for f in freqs_true])
    
    # Build full ground truth matrix (S, J)
    y_true = np.zeros((S, J), dtype=bool)
    for s in range(S):
        for jt, ft in enumerate(freqs_true):
            if masks[s, jt]:
                y_true[s, idx_map[jt]] = True
    
    # =========================================================================
    # 3. COMPUTE STATISTICS
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. COMPUTING STATISTICS")
    print("=" * 60)
    
    # Wald test
    print("  Computing Wald test...")
    wald_W, wald_pval = wald_test(beta_trace, J, burn_in_frac=BURNIN)
    
    # Phase concentration test
    print("  Computing phase concentration test...")
    phase_R, phase_pval, phase_est = phase_concentration_test(beta_trace, J, burn_in_frac=BURNIN)
    
    # =========================================================================
    # 4. LOAD TRADITIONAL METHODS
    # =========================================================================
    HAS_TRADITIONAL = False
    if args.traditional and os.path.exists(args.traditional):
        print("\n  Loading traditional methods...")
        with open(args.traditional, 'rb') as f:
            trad = pickle.load(f)
        
        plv_val = trad['plv']['values']
        plv_pval = trad['plv']['pval']
        plv_phase = trad['plv']['phase']
        sfc_val = trad['sfc']['values']
        sfc_pval = trad['sfc']['pval']
        
        # Resample if needed
        if plv_val.shape[1] != J:
            from scipy.interpolate import interp1d
            plv_freqs = trad.get('freqs', np.linspace(freqs[0], freqs[-1], plv_val.shape[1]))
            
            plv_val_new = np.zeros((S, J))
            plv_pval_new = np.zeros((S, J))
            plv_phase_new = np.zeros((S, J))
            sfc_val_new = np.zeros((S, J))
            sfc_pval_new = np.zeros((S, J))
            
            for s in range(S):
                plv_val_new[s] = interp1d(plv_freqs, plv_val[s], kind='nearest', fill_value='extrapolate')(freqs)
                plv_pval_new[s] = interp1d(plv_freqs, plv_pval[s], kind='nearest', fill_value='extrapolate')(freqs)
                plv_phase_new[s] = interp1d(plv_freqs, plv_phase[s], kind='nearest', fill_value='extrapolate')(freqs)
                sfc_val_new[s] = interp1d(plv_freqs, sfc_val[s], kind='nearest', fill_value='extrapolate')(freqs)
                sfc_pval_new[s] = interp1d(plv_freqs, sfc_pval[s], kind='nearest', fill_value='extrapolate')(freqs)
            
            plv_val, plv_pval, plv_phase = plv_val_new, plv_pval_new, plv_phase_new
            sfc_val, sfc_pval = sfc_val_new, sfc_pval_new
        
        HAS_TRADITIONAL = True
        print(f"  PLV: {plv_val.shape}, SFC: {sfc_val.shape}")
    else:
        print("\n  No traditional methods provided")
    
    # =========================================================================
    # 5. DETECTION METRICS
    # =========================================================================
    print("\n" + "=" * 60)
    print("4. DETECTION METRICS")
    print("=" * 60)
    
    m_wald = compute_detection_metrics(y_true, wald_pval, alpha=ALPHA)
    m_phase = compute_detection_metrics(y_true, phase_pval, alpha=ALPHA)
    
    print(f"\nJoint Inference @ α={ALPHA}:")
    print(f"  {'Test':<12} {'Sens':>8} {'Spec':>8} {'Prec':>8} {'F1':>8}")
    print("-" * 50)
    print(f"  {'Wald':<12} {m_wald['sensitivity']:>8.3f} {m_wald['specificity']:>8.3f} {m_wald['precision']:>8.3f} {m_wald['f1']:>8.3f}")
    print(f"  {'Phase':<12} {m_phase['sensitivity']:>8.3f} {m_phase['specificity']:>8.3f} {m_phase['precision']:>8.3f} {m_phase['f1']:>8.3f}")
    
    if HAS_TRADITIONAL:
        m_plv = compute_detection_metrics(y_true, plv_pval, alpha=ALPHA)
        m_sfc = compute_detection_metrics(y_true, sfc_pval, alpha=ALPHA)
        print(f"  {'PLV':<12} {m_plv['sensitivity']:>8.3f} {m_plv['specificity']:>8.3f} {m_plv['precision']:>8.3f} {m_plv['f1']:>8.3f}")
        print(f"  {'SFC':<12} {m_sfc['sensitivity']:>8.3f} {m_sfc['specificity']:>8.3f} {m_sfc['precision']:>8.3f} {m_sfc['f1']:>8.3f}")
    
    # =========================================================================
    # 6. GENERATE FIGURES
    # =========================================================================
    print("\n" + "=" * 60)
    print("5. GENERATING FIGURES")
    print("=" * 60)
    
    figures_dir = os.path.join(args.output, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    set_publication_style()
    
    if HAS_TRADITIONAL:
        # ----- Effect heatmaps (Wald) -----
        print("  Effect size heatmaps (Wald)...")
        plot_effect_comparison(
            {'PLV': plv_val, 'SFC': sfc_val, 'Joint |β|': beta_mag, 'Wald W': wald_W},
            freqs,
            os.path.join(figures_dir, 'heatmap_effect_wald.png'),
            true_freqs=freqs_true,
            masks=masks,
            log_scale_keys=('Wald W',),
            suptitle='Effect Size Comparison (Wald)',
        )
        
        # ----- Effect heatmaps (Phase) -----
        print("  Effect size heatmaps (Phase)...")
        plot_effect_comparison(
            {'PLV': plv_val, 'SFC': sfc_val, 'Joint |β|': beta_mag, 'Phase R': phase_R},
            freqs,
            os.path.join(figures_dir, 'heatmap_effect_phase.png'),
            true_freqs=freqs_true,
            masks=masks,
            suptitle='Effect Size Comparison (Phase)',
        )
        
        # ----- P-value heatmaps (Wald) -----
        print("  P-value heatmaps (Wald)...")
        plot_pval_comparison(
            {'PLV': plv_pval, 'SFC': sfc_pval, 'Wald': wald_pval},
            freqs,
            os.path.join(figures_dir, 'heatmap_pval_wald.png'),
            true_freqs=freqs_true,
            masks=masks,
            alpha=ALPHA,
            suptitle='P-value Comparison (Wald)',
        )
        
        # ----- P-value heatmaps (Phase) -----
        print("  P-value heatmaps (Phase)...")
        plot_pval_comparison(
            {'PLV': plv_pval, 'SFC': sfc_pval, 'Phase R': phase_pval},
            freqs,
            os.path.join(figures_dir, 'heatmap_pval_phase.png'),
            true_freqs=freqs_true,
            masks=masks,
            alpha=ALPHA,
            suptitle='P-value Comparison (Phase)',
        )
        
        # ----- Magnitude scatter -----
        print("  Magnitude scatter...")
        joint_mag_at_true = beta_mag[:, idx_map]
        plv_at_true = plv_val[:, idx_map]
        sfc_at_true = sfc_val[:, idx_map]
        
        plot_magnitude_scatter(
            gt_mag,
            {'PLV': plv_at_true, 'SFC': sfc_at_true, 'Joint': joint_mag_at_true},
            masks,
            os.path.join(figures_dir, 'magnitude_scatter.png'),
        )
        
        # ----- Phase recovery -----
        print("  Phase recovery...")
        joint_phase_at_true = beta_phase[:, idx_map]
        plv_phase_at_true = plv_phase[:, idx_map]
        
        plot_phase_recovery(
            gt_phase,
            {'Joint': joint_phase_at_true, 'PLV': plv_phase_at_true},
            masks,
            os.path.join(figures_dir, 'phase_recovery.png'),
        )
        
        # ----- Metrics bars -----
        print("  Detection metrics bars...")
        plot_metrics_bars(
            {'PLV': m_plv, 'SFC': m_sfc, 'Joint (Wald)': m_wald},
            os.path.join(figures_dir, 'metrics_bars_wald.png'),
            alpha=ALPHA,
        )
        plot_metrics_bars(
            {'PLV': m_plv, 'SFC': m_sfc, 'Joint (Phase)': m_phase},
            os.path.join(figures_dir, 'metrics_bars_phase.png'),
            alpha=ALPHA,
        )
        
        # ----- ROC/PR curves -----
        print("  ROC/PR curves...")
        # Scores: higher = more likely positive, so use 1 - pval
        plot_roc_pr_curves(
            y_true,
            {'PLV': 1 - plv_pval, 'SFC': 1 - sfc_pval, 'Wald': 1 - wald_pval},
            os.path.join(figures_dir, 'roc_pr_curves.png'),
        )
    
    # ----- Beta posterior scatter -----
    print("  Beta posterior scatter...")
    plot_beta_posterior_scatter(
        beta_trace,
        freqs,
        os.path.join(figures_dir, 'beta_posterior_scatter.png'),
        unit_idx=0,
        burn_in_frac=BURNIN,
        freqs_true=freqs_true,
        masks=masks,
    )
    
    # ----- Spectral dynamics -----
    print("  Spectral dynamics figures...")
    dynamics_dir = os.path.join(figures_dir, 'dynamics')
    os.makedirs(dynamics_dir, exist_ok=True)
    
    generate_spectral_dynamics_figures(
        sim_data, legacy, dynamics_dir,
        window_sec=20.0,
        n_snapshots=4,
        snapshot_sec=10.0,
    )
    
    # =========================================================================
    # 7. SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nResults: {results_dir}/")
    print(f"  - coupling.npz")
    print(f"  - spectral.npz")
    print(f"  - metadata.json")
    print(f"  - joint.pkl (legacy format)")
    print(f"\nFigures: {figures_dir}/")
    for root, dirs, files in os.walk(figures_dir):
        for f in sorted(files):
            if f.endswith('.png'):
                rel_path = os.path.relpath(os.path.join(root, f), figures_dir)
                print(f"  - {rel_path}")


if __name__ == '__main__':
    main()