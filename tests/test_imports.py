"""Verify all public API symbols import correctly."""
import pytest


def test_version():
    import joint_ssmt
    assert joint_ssmt.__version__ == "0.1.0"


def test_core_types():
    from joint_ssmt import OUParams, Trace, StateIndex
    assert OUParams is not None
    assert Trace is not None
    assert StateIndex is not None


def test_single_trial_inference():
    from joint_ssmt import (
        run_joint_inference_single_trial,
        SingleTrialInferenceConfig,
    )
    assert callable(run_joint_inference_single_trial)
    assert SingleTrialInferenceConfig is not None


def test_trial_structured_inference():
    from joint_ssmt import (
        run_joint_inference_trials_hier,
        InferenceTrialsHierConfig,
    )
    assert callable(run_joint_inference_trials_hier)
    assert InferenceTrialsHierConfig is not None


def test_io_runners():
    from joint_ssmt import run_inference, run_inference_trials
    from joint_ssmt import load_results, load_results_trials
    assert callable(run_inference)
    assert callable(run_inference_trials)
    assert callable(load_results)
    assert callable(load_results_trials)


def test_simulation():
    from joint_ssmt import (
        SingleTrialSimConfig,
        simulate_single_trial,
        build_history_design_single,
    )
    assert callable(simulate_single_trial)
    assert callable(build_history_design_single)
    assert SingleTrialSimConfig is not None


def test_em():
    from joint_ssmt import (
        em_ct_single_jax,
        EMSingleResult,
        em_ct_hier_jax,
        EMHierResult,
        upsample_ct_single_fine,
        UpsampleSingleResult,
    )
    assert callable(em_ct_single_jax)
    assert callable(em_ct_hier_jax)
    assert callable(upsample_ct_single_fine)


def test_utilities():
    from joint_ssmt import (
        derotate_tfr_align_start,
        centres_from_win,
        map_blocks_to_fine,
        build_t2k,
        normalize_Y_to_RJMK,
        separated_to_interleaved,
        interleaved_to_separated,
    )
    assert callable(derotate_tfr_align_start)
    assert callable(centres_from_win)


def test_io_configs():
    from joint_ssmt.io import (
        DEFAULT_CTSSMT_CONFIG,
        DEFAULT_MCMC_CONFIG,
        DEFAULT_OUTPUT_CONFIG,
        DEFAULT_CTSSMT_CONFIG_TRIALS,
        DEFAULT_MCMC_CONFIG_TRIALS,
        DEFAULT_OUTPUT_CONFIG_TRIALS,
    )
    assert isinstance(DEFAULT_CTSSMT_CONFIG, dict)
    assert isinstance(DEFAULT_MCMC_CONFIG_TRIALS, dict)


def test_plotting():
    from joint_ssmt.plotting import (
        plot_effect_heatmap,
        plot_magnitude_scatter,
        wald_test,
        generate_spectral_dynamics_figures,
    )
    assert callable(plot_effect_heatmap)
    assert callable(wald_test)


def test_analysis():
    from joint_ssmt.analysis import (
        wald_test,
        phase_concentration_test,
        compute_detection_metrics,
    )
    assert callable(wald_test)
    assert callable(phase_concentration_test)
