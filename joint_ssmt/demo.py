"""Bundled demo dataset and a small end-to-end test."""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

__all__ = ["load_demo_data", "test"]


_DATA_DIR = Path(__file__).resolve().parent / "data"
_DEMO_FILE = _DATA_DIR / "demo_dataset.npz"


def load_demo_data():
    """Return the bundled demo dataset as a dict.

    Keys: lfp (T,), spikes (S, T), fs, meta.
    """
    if not _DEMO_FILE.exists():
        raise FileNotFoundError(
            f"Demo dataset not found at {_DEMO_FILE}. "
            "Reinstalling joint-ssmt usually fixes this."
        )
    arr = np.load(_DEMO_FILE)
    return {
        "lfp": arr["lfp"],
        "spikes": arr["spikes"],
        "fs": float(arr["fs"]),
        "meta": {
            "duration_sec": float(arr["lfp"].shape[0]) / float(arr["fs"]),
            "n_units": int(arr["spikes"].shape[0]),
            "source": "synthetic 180 s LFP + 2 spike units, oscillations at 5/10/15/20/30 Hz",
        },
    }


def test(verbose: bool = True) -> bool:
    """Run a short end-to-end pipeline to confirm the install works.

    Generates a 10-second synthetic dataset, runs a short Joint SSMT
    inference, and checks the expected outputs are present. About a minute
    on CPU. Returns True on success, raises on failure.
    """
    from joint_ssmt import (
        SingleTrialSimConfig,
        simulate_single_trial,
        run_inference,
        load_results,
    )

    if verbose:
        print("joint-ssmt test")
        print("  1/3 simulating 10 s synthetic data ...")
    cfg = SingleTrialSimConfig(duration_sec=10.0, S=1, k_active=1)
    sim = simulate_single_trial(cfg, seed=0)

    with tempfile.TemporaryDirectory() as tmp:
        if verbose:
            print(f"  2/3 running short inference into {tmp} ...")
        run_inference(
            lfp=sim["LFP"],
            spikes=sim["spikes"],
            spectral_config={
                "freq_min": 5.0,
                "freq_max": 35.0,
                "freq_step": 2.0,
                "window_sec": 1.0,
                "time_bandwidth": 2.0,
            },
            inference_config={
                "warmup_iterations": 50,
                "n_refresh_cycles": 1,
                "iterations_per_refresh": 25,
            },
            output_config={"output_dir": tmp, "save_spectral": False},
            fs=cfg.fs,
            plot=False,
        )

        if verbose:
            print("  3/3 checking outputs ...")
        results = load_results(tmp)
        if "coupling" not in results:
            raise RuntimeError("test failed: 'coupling' missing from results.")
        for key in ("freqs", "beta_mag", "wald_pval", "wald_W", "beta_phase"):
            if key not in results["coupling"]:
                raise RuntimeError(f"test failed: 'coupling[{key!r}]' missing from results.")

    if verbose:
        print("\nOK. joint-ssmt is installed and working.")
    return True
