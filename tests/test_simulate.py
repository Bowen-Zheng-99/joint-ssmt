"""Verify simulator output shapes and keys."""
import numpy as np
import pytest


def test_simulate_single_trial_keys(single_trial_sim_data):
    d = single_trial_sim_data
    required_keys = [
        "LFP", "LFP_clean", "spikes", "Z_lat",
        "Ztil_R", "Ztil_I",
        "beta_true", "masks", "beta_mag", "beta_phase",
        "gamma_true", "freqs_hz", "freqs_hz_coupled",
        "time", "t_fine", "delta_spk", "fs",
    ]
    for key in required_keys:
        assert key in d, f"Missing key: {key}"


def test_simulate_single_trial_shapes(single_trial_sim_data):
    d = single_trial_sim_data
    S = 2   # from fixture
    J = 2   # freqs_hz = [11, 19], no extras
    T = int(2.0 * 500.0)   # duration_sec * fs
    T_fine = int(2.0 / 0.002)  # duration_sec / delta_spk

    assert d["LFP"].shape == (T,)
    assert d["spikes"].shape == (S, T_fine)
    assert d["beta_true"].shape == (S, 1 + 2 * J)
    assert d["masks"].shape == (S, J)
    assert d["beta_mag"].shape == (S, J)
    assert d["beta_phase"].shape == (S, J)
    assert d["Z_lat"].shape == (J, T)
    assert d["Ztil_R"].shape == (J, T_fine)
    assert d["gamma_true"].shape == (5,)  # n_lags=5


def test_simulate_spikes_are_binary(single_trial_sim_data):
    spikes = single_trial_sim_data["spikes"]
    assert set(np.unique(spikes)).issubset({0, 1})


def test_simulate_masks_respect_active_bands(single_trial_sim_data):
    masks = single_trial_sim_data["masks"]
    # k_active=1, so each unit should have exactly 1 active band
    assert np.all(masks.sum(axis=1) == 1)


def test_build_history_design_single():
    from joint_ssmt import build_history_design_single

    S, T, n_lags = 2, 100, 5
    spikes = np.random.default_rng(0).binomial(1, 0.1, (S, T)).astype(np.float32)
    H = build_history_design_single(spikes, n_lags=n_lags)

    assert H.shape == (S, T, n_lags)
    # First row should be all zeros (no history at t=0)
    assert np.all(H[:, 0, :] == 0)
