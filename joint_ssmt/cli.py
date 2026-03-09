"""
Command-line interface for Joint SSMT.

Usage::

    joint-ssmt run \\
        --lfp data/lfp.npy --spikes data/spikes.npy \\
        --spectral-config configs/spectral.yaml \\
        --inference-config configs/inference.yaml \\
        --fs 1000 --output-dir results/ --plot

    joint-ssmt run-trials \\
        --lfp data/lfp_trials.npy --spikes data/spikes_trials.npy \\
        --spectral-config configs/spectral_trials.yaml \\
        --inference-config configs/inference_trials.yaml \\
        --fs 1000 --output-dir results_trials/ --plot
"""

import argparse
import logging
import sys
import numpy as np


def _load_array(path: str) -> np.ndarray:
    """Load a numpy array from .npy or .npz (first key)."""
    if path.endswith(".npz"):
        with np.load(path) as f:
            key = list(f.keys())[0]
            return f[key]
    return np.load(path)


def _run_single(args):
    from joint_ssmt.io.runner import run_inference

    lfp = _load_array(args.lfp)
    spikes = _load_array(args.spikes)

    saved = run_inference(
        lfp=lfp,
        spikes=spikes,
        spectral_config=args.spectral_config,
        inference_config=args.inference_config,
        output_config={
            "output_dir": args.output_dir,
            "save_spectral": True,
        },
        fs=args.fs,
        plot=args.plot,
        verbose=True,
    )

    for name, path in saved.items():
        print(f"  {name}: {path}")


def _run_trials(args):
    from joint_ssmt.io.runner_trials import run_inference_trials

    lfp = _load_array(args.lfp)
    spikes = _load_array(args.spikes)

    saved = run_inference_trials(
        lfp=lfp,
        spikes=spikes,
        spectral_config=args.spectral_config,
        inference_config=args.inference_config,
        output_config={
            "output_dir": args.output_dir,
            "save_spectral": True,
        },
        fs=args.fs,
        plot=args.plot,
        verbose=True,
    )

    for name, path in saved.items():
        print(f"  {name}: {path}")


def _plot_results(args):
    from joint_ssmt.io.runner import load_results
    from joint_ssmt.io.runner_trials import load_results_trials
    from joint_ssmt.plotting.summary import plot_all_default

    # Try trial results first, fall back to single
    results = load_results_trials(args.results_dir)
    if not results:
        results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    saved = plot_all_default(
        results,
        save_dir=args.results_dir,
        fmt=args.format,
        dpi=args.dpi,
    )
    print(f"Saved {len(saved)} figures:")
    for p in saved:
        print(f"  {p}")


def main():
    parser = argparse.ArgumentParser(
        prog="joint-ssmt",
        description="Joint SSMT: Bayesian spike-field coupling inference",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- run (single-trial) ---
    p_run = sub.add_parser("run", help="Single-trial inference")
    p_run.add_argument("--lfp", required=True, help="Path to LFP array (.npy / .npz)")
    p_run.add_argument("--spikes", required=True, help="Path to spike array (.npy / .npz)")
    p_run.add_argument("--spectral-config", default=None,
                       help="Path to spectral YAML config (optional)")
    p_run.add_argument("--inference-config", default=None,
                       help="Path to inference YAML config (optional)")
    p_run.add_argument("--fs", type=float, default=1000.0, help="Sampling rate (Hz)")
    p_run.add_argument("--output-dir", default="./results", help="Output directory")
    p_run.add_argument("--plot", action="store_true", help="Generate default plots")
    p_run.set_defaults(func=_run_single)

    # --- run-trials ---
    p_trials = sub.add_parser("run-trials", help="Trial-structured inference")
    p_trials.add_argument("--lfp", required=True, help="Path to LFP array (.npy / .npz)")
    p_trials.add_argument("--spikes", required=True, help="Path to spike array (.npy / .npz)")
    p_trials.add_argument("--spectral-config", default=None,
                          help="Path to spectral YAML config (optional)")
    p_trials.add_argument("--inference-config", default=None,
                          help="Path to inference YAML config (optional)")
    p_trials.add_argument("--fs", type=float, default=1000.0, help="Sampling rate (Hz)")
    p_trials.add_argument("--output-dir", default="./results_trials", help="Output directory")
    p_trials.add_argument("--plot", action="store_true", help="Generate default plots")
    p_trials.set_defaults(func=_run_trials)

    # --- plot (re-generate plots from saved results) ---
    p_plot = sub.add_parser("plot", help="Re-generate plots from saved results")
    p_plot.add_argument("results_dir", help="Directory with coupling.npz etc.")
    p_plot.add_argument("--format", default="pdf", help="Figure format (pdf, png, svg)")
    p_plot.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    p_plot.set_defaults(func=_plot_results)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args.func(args)


if __name__ == "__main__":
    main()
