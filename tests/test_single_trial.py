"""Integration test: single-trial inference with run_inference()."""
import os
import tempfile
import numpy as np
import pytest


@pytest.mark.timeout(300)
def test_run_inference_minimal(single_trial_sim_data):
    """Run inference with minimal iterations and verify output structure."""
    from joint_ssmt.io import run_inference, load_results

    d = single_trial_sim_data

    with tempfile.TemporaryDirectory() as tmpdir:
        saved = run_inference(
            lfp=d["LFP"],
            spikes=d["spikes"],
            ctssmt_config={
                "freq_min": 5,
                "freq_max": 25,
                "freq_step": 5.0,
                "window_sec": 1.0,
                "NW": 1.0,
                "em_max_iter": 50,
                "em_tol": 1e-2,
            },
            mcmc_config={
                "fixed_iter": 10,
                "n_refreshes": 1,
                "inner_steps": 5,
                "trace_thin": 1,
                "n_history_lags": 3,
            },
            output_config={
                "output_dir": tmpdir,
                "save_spectral": True,
            },
            fs=d["fs"],
            delta_spk=d["delta_spk"],
            verbose=False,
        )

        # Check files exist
        assert "coupling" in saved
        assert "spectral" in saved
        assert "metadata" in saved
        assert os.path.isfile(saved["coupling"])
        assert os.path.isfile(saved["metadata"])

        # Check load_results
        results = load_results(tmpdir)
        assert "coupling" in results
        assert "metadata" in results

        coupling = results["coupling"]
        assert "beta" in coupling
        assert "beta_mag" in coupling
        assert "freqs" in coupling

        # Shapes should be consistent
        J = len(coupling["freqs"])
        S = d["spikes"].shape[0]
        assert coupling["beta_mag"].shape == (S, J)
