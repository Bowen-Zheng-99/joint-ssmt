"""Shared fixtures for joint_ssmt tests."""
import pytest
import numpy as np


@pytest.fixture
def single_trial_sim_data():
    """Small simulated single-trial dataset for fast tests."""
    from joint_ssmt import simulate_single_trial, SingleTrialSimConfig

    cfg = SingleTrialSimConfig(
        freqs_hz=np.array([11.0, 19.0]),
        freqs_hz_extra=np.array([]),
        S=2,
        k_active=1,
        duration_sec=2.0,
        fs=500.0,
        delta_spk=0.002,
        n_lags=5,
        half_bw_hz=np.array([0.05, 0.05]),
        sigma_v=np.array([4.0, 4.0]),
        sigma_eps=np.array([15.0, 15.0]),
    )
    return simulate_single_trial(cfg, seed=0)


@pytest.fixture
def trial_sim_data(single_trial_sim_data):
    """Small simulated trial-structured dataset (R=3 trials) derived from
    single-trial data by tiling with noise."""
    rng = np.random.default_rng(1)
    d = single_trial_sim_data
    R = 3
    T = len(d["LFP"])
    S, T_fine = d["spikes"].shape

    lfp = np.stack([d["LFP"] + rng.normal(0, 0.5, T) for _ in range(R)])  # (R, T)
    spikes = np.stack([d["spikes"] for _ in range(R)])  # (R, S, T_fine)

    return {
        "lfp": lfp,
        "spikes": spikes,
        "fs": d["fs"],
        "delta_spk": d["delta_spk"],
    }
