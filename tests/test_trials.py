"""Integration test: trial-structured inference with run_inference_trials()."""
import os
import tempfile
import numpy as np
import pytest


@pytest.mark.timeout(300)
def test_run_inference_trials_minimal(trial_sim_data):
    """Run trial-structured inference with minimal iterations."""
    from joint_ssmt.io import run_inference_trials, load_results_trials

    d = trial_sim_data

    with tempfile.TemporaryDirectory() as tmpdir:
        saved = run_inference_trials(
            lfp=d["lfp"],
            spikes=d["spikes"],
            ctssmt_config={
                "freq_min": 5,
                "freq_max": 25,
                "freq_step": 5.0,
                "window_sec": 0.4,
                "NW": 1,
                "em_max_iter": 50,
                "em_tol": 1e-2,
                "sig_eps_init": 5.0,
                "log_every": 5000,
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
                "downsample_factor": 5,
            },
            fs=d["fs"],
            delta_spk=d["delta_spk"],
            verbose=False,
        )

        # Check files exist
        assert "coupling" in saved
        assert "metadata" in saved
        assert os.path.isfile(saved["coupling"])
        assert os.path.isfile(saved["metadata"])

        # Check load_results_trials
        results = load_results_trials(tmpdir)
        assert "coupling" in results
        assert "metadata" in results

        coupling = results["coupling"]
        assert "beta" in coupling
        assert "beta_mag" in coupling
        assert "freqs" in coupling

        # Shapes
        J = len(coupling["freqs"])
        S = d["spikes"].shape[1]
        assert coupling["beta_mag"].shape == (S, J)

        # Metadata should have trial count
        meta = results["metadata"]
        assert meta["data"]["n_trials"] == d["lfp"].shape[0]
