#!/usr/bin/env python3
"""
run_analysis_trials.py - Complete analysis for trial-structured spike-field data

Matches the structure of run_analysis.py but for trial-structured data.

Usage:
    python run_analysis_trials.py \
        --data ./data/sim_with_trials.pkl \
        --output ./results/ \
        --traditional ./traditional_methods.pkl

    # Skip inference (use existing results)
    python run_analysis_trials.py \
        --data ./data/sim_with_trials.pkl \
        --output ./results/ \
        --skip_inference
"""

import os
import pickle
import argparse
import numpy as np
from simulate_trial_data import TrialSimConfig
# I/O
from joint_ssmt.io import (
    run_inference_trials,
    load_results_trials,
    results_to_legacy_dict_trials,
)

# Statistical tests
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

# Spectral dynamics (trial-structured)
from joint_ssmt.plotting.spectral_dynamics_trials import (
    generate_trial_dynamics_figures,
)


def main():
    parser = argparse.ArgumentParser(description='Run trial-structured spike-field coupling analysis')
    parser.add_argument('--data', type=str, required=True, help='Path to simulation data (.pkl)')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--traditional', type=str, default=None, help='Path to PLV/SFC results')
    
    # CT-SSMT config (trial-structured defaults)
    parser.add_argument('--freq_min', type=float, default=1.0)
    parser.add_argument('--freq_max', type=float, default=61.0)
    parser.add_argument('--freq_step', type=float, default=2.0)
    parser.add_argument('--window_sec', type=float, default=0.4)
    parser.add_argument('--NW', type=float, default=1.0)
    
    # MCMC config
    parser.add_argument('--fixed_iter', type=int, default=1000)
    parser.add_argument('--n_refreshes', type=int, default=5)
    parser.add_argument('--inner_steps', type=int, default=200)
    parser.add_argument('--no_shrinkage', action='store_true')
    parser.add_argument('--no_wald', action='store_true')
    parser.add_argument('--wald_alpha', type=float, default=0.05)
    
    parser.add_argument('--lfp_only', type=str, default=None, 
                        help='Path to CT-SSMT LFP-only results (.pkl) for spectral dynamics comparison')
    
    # Output config
    parser.add_argument('--downsample_factor', type=int, default=10)
    
    # Plotting config
    parser.add_argument('--plot_freqs', type=float, nargs='+', default=None,
                        help='Specific frequencies (Hz) to plot. Default: use ground truth freqs or [5,15,25,35]')
    
    # Skip inference (use existing results)
    parser.add_argument('--skip_inference', action='store_true', help='Skip inference, use existing results')
    
    args = parser.parse_args()
    
    ALPHA = 0.05
    BURNIN = 0.7  # Trial-structured uses 0.5
    
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
    
    R, T_lfp = lfp.shape
    _, S, T_fine = spikes.shape
    
    # Ground truth
    freqs_true = np.asarray(sim_data.get('freqs_hz', []), dtype=float)
    freqs_extra = np.asarray(sim_data.get('freqs_hz_extra', []), dtype=float)
    masks = sim_data.get('masks')
    
    if masks is not None:
        masks = np.asarray(masks, dtype=bool)
        J_true = masks.shape[1]
    else:
        J_true = len(freqs_true)
    
    if 'beta_mag' in sim_data:
        gt_mag = np.asarray(sim_data['beta_mag'])
        gt_phase = np.asarray(sim_data['beta_phase'])
    elif 'beta_true' in sim_data:
        beta_true = np.asarray(sim_data['beta_true'])
        bR = beta_true[:, 1:1+J_true]
        bI = beta_true[:, 1+J_true:1+2*J_true]
        gt_mag = np.sqrt(bR**2 + bI**2)
        gt_phase = np.arctan2(bI, bR)
    else:
        gt_mag = gt_phase = None
    
    print(f"  LFP: {lfp.shape} (R={R} trials)")
    print(f"  Spikes: {spikes.shape} (S={S} units)")
    print(f"  True signal freqs: {freqs_true}")
    if len(freqs_extra) > 0:
        print(f"  Extra (signal-only) freqs: {freqs_extra}")
    if masks is not None:
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
        print("2. RUNNING HIERARCHICAL JOINT INFERENCE")
        print("=" * 60)
        
        run_inference_trials(
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
                'use_wald_selection': not args.no_wald,
                'wald_alpha': args.wald_alpha,
                'burn_in_frac': BURNIN,
            },
            output_config={
                'output_dir': results_dir,
                'save_spectral': True,
                'downsample_factor': args.downsample_factor,
            },
            fs=fs,
            delta_spk=delta_spk,
            ground_truth={
                'freqs_hz': freqs_true,
                'freqs_hz_extra': freqs_extra,
                'masks': masks,
            } if masks is not None else None,
        )
    
    # Load results and convert to legacy format
    results = load_results_trials(results_dir)
    coupling = results['coupling']
    
    freqs = coupling['freqs']
    J = len(freqs)
    beta_trace = coupling['beta_trace']
    beta_mag = coupling['beta_mag']
    beta_phase = coupling['beta_phase']
    
    # Also save legacy format
    legacy = results_to_legacy_dict_trials(results_dir)
    legacy_path = os.path.join(results_dir, 'joint.pkl')
    with open(legacy_path, 'wb') as f:
        pickle.dump(legacy, f)
    print(f"  Saved legacy format: {legacy_path}")
    
    # Map true frequencies to analysis grid
    all_true_freqs = np.concatenate([freqs_true, freqs_extra]) if len(freqs_extra) > 0 else freqs_true
    idx_map = np.array([np.argmin(np.abs(freqs - f)) for f in all_true_freqs])
    idx_map_coupled = np.array([np.argmin(np.abs(freqs - f)) for f in freqs_true])
    
    # Build full ground truth matrix (S, J) - only for coupled freqs
    if masks is not None:
        y_true = np.zeros((S, J), dtype=bool)
        for s in range(S):
            for jt, ft in enumerate(freqs_true):
                if jt < masks.shape[1] and masks[s, jt]:
                    y_true[s, idx_map_coupled[jt]] = True
    else:
        y_true = None
    
    # =========================================================================
    # 3. COMPUTE STATISTICS
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. COMPUTING STATISTICS")
    print("=" * 60)
    
    # Use pre-computed stats from coupling.npz if available
    if 'wald_W' in coupling:
        wald_W = coupling['wald_W']
        wald_pval = coupling['wald_pval']
        print("  Using pre-computed Wald statistics")
    else:
        print("  Computing Wald test...")
        wald_W, wald_pval = wald_test(beta_trace, J, burn_in_frac=BURNIN)
    
    if 'phase_R' in coupling:
        phase_R = coupling['phase_R']
        phase_pval = coupling['phase_pval']
        phase_est = coupling['phase_est']
        print("  Using pre-computed phase statistics")
    else:
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
        
        # Handle nested structure
        if 'plv' in trad:
            plv_val = trad['plv']['values']
            plv_pval = trad['plv']['pval']
            plv_phase = trad['plv']['phase']
        else:
            plv_val = trad.get('plv_values')
            plv_pval = trad.get('plv_pval')
            plv_phase = trad.get('plv_phase')
        
        if 'sfc' in trad:
            sfc_val = trad['sfc']['values']
            sfc_pval = trad['sfc']['pval']
        else:
            sfc_val = trad.get('sfc_values')
            sfc_pval = trad.get('sfc_pval')
        
        # Get traditional freqs
        trad_freqs = trad.get('config', {}).get('freqs', trad.get('freqs'))
        if trad_freqs is None:
            trad_freqs = np.arange(1, 61, 2, dtype=float)  # Default trial freqs
        
        # Resample if needed
        if plv_val is not None and plv_val.shape[1] != J:
            from scipy.interpolate import interp1d
            
            plv_val_new = np.zeros((S, J))
            plv_pval_new = np.zeros((S, J))
            plv_phase_new = np.zeros((S, J))
            sfc_val_new = np.zeros((S, J))
            sfc_pval_new = np.zeros((S, J))
            
            for s in range(S):
                plv_val_new[s] = interp1d(trad_freqs, plv_val[s], kind='nearest', fill_value='extrapolate')(freqs)
                plv_pval_new[s] = interp1d(trad_freqs, plv_pval[s], kind='nearest', fill_value='extrapolate')(freqs)
                plv_phase_new[s] = interp1d(trad_freqs, plv_phase[s], kind='nearest', fill_value='extrapolate')(freqs)
                sfc_val_new[s] = interp1d(trad_freqs, sfc_val[s], kind='nearest', fill_value='extrapolate')(freqs)
                sfc_pval_new[s] = interp1d(trad_freqs, sfc_pval[s], kind='nearest', fill_value='extrapolate')(freqs)
            
            plv_val, plv_pval, plv_phase = plv_val_new, plv_pval_new, plv_phase_new
            sfc_val, sfc_pval = sfc_val_new, sfc_pval_new
        
        if plv_val is not None:
            HAS_TRADITIONAL = True
            print(f"  PLV: {plv_val.shape}, SFC: {sfc_val.shape}")
    else:
        print("\n  No traditional methods provided")
        plv_val = plv_pval = plv_phase = None
        sfc_val = sfc_pval = None
    
    # =========================================================================
    # 5. DETECTION METRICS
    # =========================================================================
    if y_true is not None:
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
    else:
        m_wald = m_phase = m_plv = m_sfc = None
    
    # =========================================================================
    # 6. GENERATE FIGURES
    # =========================================================================
    print("\n" + "=" * 60)
    print("5. GENERATING FIGURES")
    print("=" * 60)
    
    figures_dir = os.path.join(args.output, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    set_publication_style()
    
    # ----- Coupling heatmaps -----
    if HAS_TRADITIONAL and masks is not None:
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
        
        print("  Effect size heatmaps (Phase)...")
        plot_effect_comparison(
            {'PLV': plv_val, 'SFC': sfc_val, 'Joint |β|': beta_mag, 'Phase R': phase_R},
            freqs,
            os.path.join(figures_dir, 'heatmap_effect_phase.png'),
            true_freqs=freqs_true,
            masks=masks,
            suptitle='Effect Size Comparison (Phase)',
        )
        
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
    if HAS_TRADITIONAL and gt_mag is not None and masks is not None:
        print("  Magnitude scatter...")
        joint_mag_at_true = beta_mag[:, idx_map_coupled]
        plv_at_true = plv_val[:, idx_map_coupled]
        sfc_at_true = sfc_val[:, idx_map_coupled]
        
        # Ensure masks match the coupled frequencies only
        masks_coupled = masks[:, :len(freqs_true)]
        
        plot_magnitude_scatter(
            gt_mag[:, :len(freqs_true)],
            {'PLV': plv_at_true, 'SFC': sfc_at_true, 'Joint': joint_mag_at_true},
            masks_coupled,
            os.path.join(figures_dir, 'magnitude_scatter.png'),
        )
    
    # ----- Phase recovery -----
    if HAS_TRADITIONAL and gt_phase is not None and masks is not None:
        print("  Phase recovery...")
        joint_phase_at_true = beta_phase[:, idx_map_coupled]
        plv_phase_at_true = plv_phase[:, idx_map_coupled]
        
        masks_coupled = masks[:, :len(freqs_true)]
        
        plot_phase_recovery(
            gt_phase[:, :len(freqs_true)],
            {'Joint': joint_phase_at_true, 'PLV': plv_phase_at_true},
            masks_coupled,
            os.path.join(figures_dir, 'phase_recovery.png'),
        )
    
    # ----- Metrics bars -----
    if m_wald is not None:
        print("  Detection metrics bars...")
        if HAS_TRADITIONAL:
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
        else:
            plot_metrics_bars(
                {'Joint (Wald)': m_wald, 'Joint (Phase)': m_phase},
                os.path.join(figures_dir, 'metrics_bars.png'),
                alpha=ALPHA,
            )
    
    # ----- ROC/PR curves -----
    if y_true is not None and HAS_TRADITIONAL:
        print("  ROC/PR curves...")
        plot_roc_pr_curves(
            y_true,
            {'PLV': 1 - plv_pval, 'SFC': 1 - sfc_pval, 'Wald': 1 - wald_pval},
            os.path.join(figures_dir, 'roc_pr_curves.png'),
        )
    
    # ----- Beta posterior scatter -----
    print("  Beta posterior scatter...")
    for s in range(min(S, 3)):  # First 3 units
        plot_beta_posterior_scatter(
            beta_trace,
            freqs,
            os.path.join(figures_dir, f'beta_posterior_unit{s}.png'),
            unit_idx=s,
            burn_in_frac=BURNIN,
            freqs_true=freqs_true,
            masks=masks,
        )
    
    # ----- Spectral dynamics (trial-structured) -----
    print("  Spectral dynamics figures...")
    dynamics_dir = os.path.join(figures_dir, 'dynamics')
    os.makedirs(dynamics_dir, exist_ok=True)
    
    try:
        # Prepare joint_results dict for generate_trial_dynamics_figures
        joint_results = {
            'trace': legacy['trace'],
            'freqs_dense': freqs,
        }
        
        # Load LFP-only results - first check if saved during inference
        lfp_results = None
        lfp_only_path = os.path.join(results_dir, 'ctssmt_lfp_only.npz')
        
        if os.path.exists(lfp_only_path):
            print(f"    Loading LFP-only results from {lfp_only_path}")
            lfp_only_data = np.load(lfp_only_path)
            
            # Convert to Z_smooth format expected by plotting
            # X_fine: (T_ds, 2*J*M), D_fine: (R, T_ds, 2*J*M)
            if 'X_fine' in lfp_only_data and 'D_fine' in lfp_only_data:
                X_fine_lfp = lfp_only_data['X_fine']
                D_fine_lfp = lfp_only_data['D_fine']
                J_lfp = len(lfp_only_data['freqs'])
                
                # Construct lfp_results dict with Z_smooth (using X+D)
                # The plotting code will extract and convert properly
                lfp_results = {
                    'X_fine': X_fine_lfp,
                    'D_fine': D_fine_lfp,
                    'freqs': lfp_only_data['freqs'],
                }
                if 'X_var_fine' in lfp_only_data:
                    lfp_results['X_var_fine'] = lfp_only_data['X_var_fine']
                if 'D_var_fine' in lfp_only_data:
                    lfp_results['D_var_fine'] = lfp_only_data['D_var_fine']
                    
                print(f"    LFP-only X_fine: {X_fine_lfp.shape}, D_fine: {D_fine_lfp.shape}")
        
        # Also check if provided via CLI
        elif args.lfp_only and os.path.exists(args.lfp_only):
            print(f"    Loading LFP-only results from {args.lfp_only}")
            with open(args.lfp_only, 'rb') as f:
                lfp_results = pickle.load(f)
            # Expected keys: 'Z_smooth_full' or 'Z_smooth'
            if 'Z_smooth_full' in lfp_results:
                print(f"    LFP-only Z_smooth_full: {lfp_results['Z_smooth_full'].shape}")
            elif 'Z_smooth' in lfp_results:
                print(f"    LFP-only Z_smooth: {lfp_results['Z_smooth'].shape}")
        
        # Get coupled frequencies (from masks or ground truth)
        coupled_freqs = freqs_true if masks is None else freqs_true[masks.any(axis=0)]
        if len(coupled_freqs) == 0:
            coupled_freqs = args.plot_freqs if args.plot_freqs else None
        
        # Generate figures for each coupled frequency separately
        # for freq_hz in (coupled_freqs if coupled_freqs is not None else [None]):
        #     if freq_hz is not None:
        #         freq_dir = os.path.join(dynamics_dir, f'{int(freq_hz)}Hz')
        #         plot_freqs_arg = [float(freq_hz)]
        #         print(f"    Generating figures for {freq_hz:.0f} Hz...")
        #     else:
        #         freq_dir = dynamics_dir
        #         plot_freqs_arg = args.plot_freqs
        #     
        #     os.makedirs(freq_dir, exist_ok=True)
        #     saved_dynamics = generate_trial_dynamics_figures(
        #         sim_data=sim_data,
        #         joint_results=joint_results,
        #         lfp_results=lfp_results,
        #         output_dir=freq_dir,
        #         freqs_dense=freqs,
        #         plot_freqs=plot_freqs_arg,
        #         sample_trials=[0, 25, 57, 85] if R >= 86 else list(range(min(4, R))),
        #         time_range=(1.0, 8.0),
        #     )
        # 
        # print(f"    Saved {len(saved_dynamics)} spectral dynamics figures")
    except Exception as e:
        import traceback
        print(f"    Warning: Could not generate spectral dynamics plots: {e}")
        traceback.print_exc()
    
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