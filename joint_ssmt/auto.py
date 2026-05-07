"""Top-level wrapper around ``run_inference`` and ``run_inference_trials``.

Sanity-checks the inputs, prints a short pre-flight summary, asks for one
confirmation, and runs the pipeline. The output directory holds the
inference results and the figures derived from them; the raw inputs are
not duplicated.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from joint_ssmt.io import run_inference, run_inference_trials

__all__ = ["run_auto_inference"]


# Defaults tuned for 1-60 Hz analyses on 1 kHz recordings. Override per call.
_DEFAULT_SPECTRAL = {
    "freq_min": 1.0,
    "freq_max": 60.0,
    "freq_step": 1.0,
    "window_sec": 2.0,
    "time_bandwidth": 2.0,
}

_DEFAULT_INFERENCE = {
    "warmup_iterations": 300,
    "n_refresh_cycles": 2,
    "iterations_per_refresh": 100,
}


def _validate_single_trial(lfp: np.ndarray, spikes: np.ndarray, fs: float) -> None:
    if lfp.ndim != 1:
        raise ValueError(
            f"For trial_structure=False, LFP must be 1-D (T,) but got shape {lfp.shape}. "
            "If your data is shape (R, T), pass trial_structure=True."
        )
    if spikes.ndim != 2:
        raise ValueError(
            f"For trial_structure=False, spikes must be 2-D (S, T) but got shape {spikes.shape}. "
            "If your data is shape (R, S, T), pass trial_structure=True."
        )
    if spikes.shape[1] != lfp.shape[0]:
        raise ValueError(
            f"Spike bins ({spikes.shape[1]}) and LFP samples "
            f"({lfp.shape[0]}) must match. Both should be sampled at fs."
        )
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive (got {fs}).")


def _validate_trials(lfp: np.ndarray, spikes: np.ndarray, fs: float) -> None:
    if lfp.ndim != 2:
        raise ValueError(
            f"For trial_structure=True, LFP must be 2-D (R, T) but got shape {lfp.shape}. "
            "If your data is shape (T,), pass trial_structure=False."
        )
    if spikes.ndim != 3:
        raise ValueError(
            f"For trial_structure=True, spikes must be 3-D (R, S, T) but got shape {spikes.shape}. "
            "If your data is shape (S, T), pass trial_structure=False."
        )
    if lfp.shape[0] != spikes.shape[0]:
        raise ValueError(
            f"Number of trials must match: LFP has R={lfp.shape[0]}, "
            f"spikes has R={spikes.shape[0]}."
        )
    if spikes.shape[2] != lfp.shape[1]:
        raise ValueError(
            f"Spike bins ({spikes.shape[2]}) and LFP samples ({lfp.shape[1]}) "
            "must match within each trial. Both should be sampled at fs."
        )
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive (got {fs}).")


def _format_summary(lfp, spikes, fs, spectral, inference, output_dir, trial_structure):
    sep = "-" * 64
    lines = [sep, "Joint SSMT pre-flight", sep]
    if trial_structure:
        R, T = lfp.shape
        S = spikes.shape[1]
        duration = T / fs
        rates = spikes.sum(axis=(0, 2)) / max(R * duration, 1e-12)
        rates_str = ", ".join(f"{r:.1f}" for r in rates)
        lines += [
            f"  Mode            : trial-structured",
            f"  LFP shape       : {R} trials x {T} samples",
            f"  Sampling rate   : {fs:g} Hz",
            f"  Trial duration  : {duration:.2f} s",
            f"  Number of units : {S}",
            f"  Avg spike rates : {rates_str} Hz",
        ]
    else:
        T = lfp.shape[0]
        duration = T / fs
        rates = spikes.sum(axis=1) / max(duration, 1e-12)
        rates_str = ", ".join(f"{r:.1f}" for r in rates)
        lines += [
            f"  Mode            : single-trial",
            f"  LFP shape       : {T} samples",
            f"  Sampling rate   : {fs:g} Hz",
            f"  Duration        : {duration:.2f} s",
            f"  Number of units : {spikes.shape[0]}",
            f"  Spike rates     : {rates_str} Hz",
        ]
    lines += [
        "",
        "  Frequency grid  : "
        f"{spectral['freq_min']:g}-{spectral['freq_max']:g} Hz "
        f"(step {spectral['freq_step']:g} Hz)",
        "  Multitaper      : "
        f"window {spectral['window_sec']:g} s, "
        f"time-bandwidth {spectral['time_bandwidth']:g}",
        "  Inference       : "
        f"{inference['warmup_iterations']} warmup, "
        f"{inference['n_refresh_cycles']} refresh x "
        f"{inference['iterations_per_refresh']} samples",
        f"  Output dir      : {output_dir}",
        sep,
    ]
    return "\n".join(lines)


def _resolve_configs(overrides: Optional[Mapping], output_dir: str):
    overrides = dict(overrides or {})
    spectral = {**_DEFAULT_SPECTRAL, **dict(overrides.get("spectral", {}))}
    inference = {**_DEFAULT_INFERENCE, **dict(overrides.get("inference", {}))}
    output = {
        "output_dir": output_dir,
        "save_spectral": True,
        **dict(overrides.get("output", {})),
    }
    return spectral, inference, output


def run_auto_inference(
    lfp: np.ndarray,
    spikes: np.ndarray,
    fs: float,
    output_dir: str = "./results",
    *,
    trial_structure: bool = False,
    overrides: Optional[Mapping] = None,
    interactive: bool = True,
    plot: bool = True,
):
    """Run the full Joint SSMT pipeline with one confirmation step.

    lfp:             single trial: (T,) array. Trial-structured: (R, T).
    spikes:          single trial: (S, T) array of 0/1. Trial-structured: (R, S, T).
    fs:              sampling rate in Hz.
    output_dir:      where to save results and the default plots.
    trial_structure: False (default) for single-trial inference using
                     ``run_inference``. True for hierarchical trial-structured
                     inference using ``run_inference_trials``.
    overrides:       nested dict, e.g. {'spectral': {'freq_max': 100},
                     'inference': {'warmup_iterations': 800}}.
    interactive:     if True, print summary and wait for Enter (or 'abort').
    plot:            write the default summary plots.

    Returns the list of saved file paths, or None if the user aborted.
    """
    lfp = np.asarray(lfp)
    spikes = np.asarray(spikes)
    fs = float(fs)

    if trial_structure:
        _validate_trials(lfp, spikes, fs)
    else:
        _validate_single_trial(lfp, spikes, fs)

    spectral, inference, output = _resolve_configs(overrides, output_dir)

    if interactive:
        print(_format_summary(
            lfp, spikes, fs, spectral, inference,
            output["output_dir"], trial_structure,
        ))
        try:
            answer = input(
                "Press Enter to run, or type 'abort' to cancel: "
            ).strip().lower()
        except EOFError:
            answer = ""
        if answer in {"abort", "a", "quit", "q", "no", "n"}:
            print("Aborted.")
            return None

    out_path = Path(output["output_dir"])
    out_path.mkdir(parents=True, exist_ok=True)

    runner = run_inference_trials if trial_structure else run_inference
    saved = runner(
        lfp=lfp,
        spikes=spikes,
        spectral_config=spectral,
        inference_config=inference,
        output_config=output,
        fs=fs,
        plot=plot,
    )

    if interactive:
        print(f"\nDone. Outputs in {output['output_dir']}.")

    return saved
