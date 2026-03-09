"""
Default summary plots for Joint SSMT results.

Produces publication-quality figures from saved inference results,
designed for real-world use (no ground truth required).
"""

import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def _set_style(font_size: int = 9):
    """Minimal publication-quality style."""
    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "axes.linewidth": 0.6,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    if HAS_SEABORN:
        sns.set_style("ticks")


# =========================================================================
# Coupling heatmaps
# =========================================================================

def plot_coupling_summary(
    results: Dict[str, Any],
    *,
    effect_type: str = "wald",
    alpha: float = 0.05,
    unit_labels: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 5),
    save_dir: Optional[str] = None,
    fmt: str = "pdf",
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot effect-size and p-value heatmaps side by side.

    Works on the dict returned by ``load_results()`` or
    ``load_results_trials()``.  Does **not** require ground truth.

    Parameters
    ----------
    results : dict
        Must contain a ``'coupling'`` key with at least ``freqs``,
        ``beta_mag``, and (for Wald) ``wald_pval`` / ``wald_W``.
    effect_type : ``"wald"`` | ``"phase"``
        Which test statistic to display.
    alpha : float
        Significance threshold drawn on the p-value colorbar.
    unit_labels : list of str, optional
        Y-tick labels (one per unit).  Defaults to ``0, 1, 2, …``.
    figsize : tuple
    save_dir : str, optional
        Directory to save figures.  ``None`` → display only.
    fmt : str
        File format (``"pdf"``, ``"png"``, etc.).
    dpi : int

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _set_style()

    coupling = results["coupling"]
    freqs = coupling["freqs"]
    S = coupling["beta_mag"].shape[0]
    J = len(freqs)
    freq_step = freqs[1] - freqs[0] if J > 1 else 1.0

    # Pick effect size and p-value arrays
    if effect_type == "wald":
        effect = coupling["beta_mag"]
        pvals = coupling.get("wald_pval")
        effect_label = r"|$\mathbb{E}[\beta_C]$|"
        pval_label = "Wald test"
    elif effect_type == "phase":
        effect = coupling.get("phase_R", coupling["beta_mag"])
        pvals = coupling.get("phase_pval")
        effect_label = "Phase concentration R"
        pval_label = "Phase test"
    else:
        raise ValueError(f"Unknown effect_type: {effect_type!r}")

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    extent = [
        freqs[0] - freq_step / 2,
        freqs[-1] + freq_step / 2,
        S - 0.5,
        -0.5,
    ]

    # --- Effect size ---
    vmax_eff = np.percentile(effect[np.isfinite(effect)], 99)
    im_eff = axes[0].imshow(
        effect, aspect="auto", cmap="Reds",
        vmin=0, vmax=vmax_eff, extent=extent,
    )
    axes[0].set_title(f"Effect Size ({effect_label})", fontweight="bold")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Unit")
    cb_eff = fig.colorbar(im_eff, ax=axes[0], fraction=0.04, pad=0.02)
    cb_eff.set_label(effect_label)

    # --- P-value ---
    if pvals is not None:
        log_p = -np.log10(np.clip(pvals, 1e-20, 1.0))
        vmax_p = max(np.percentile(log_p[np.isfinite(log_p)], 99), -np.log10(alpha) + 1)
        im_p = axes[1].imshow(
            log_p, aspect="auto", cmap="hot_r",
            vmin=0, vmax=vmax_p, extent=extent,
        )
        axes[1].set_title(f"$-\\log_{{10}}(p)$  ({pval_label})", fontweight="bold")
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel("Unit")
        cb_p = fig.colorbar(im_p, ax=axes[1], fraction=0.04, pad=0.02)
        cb_p.set_label(r"$-\log_{10}(p)$")
        sig_line = -np.log10(alpha)
        if sig_line <= vmax_p:
            cb_p.ax.axhline(sig_line, color="cyan", lw=1.5)
            cb_p.ax.text(1.3, sig_line, f"α={alpha}", color="cyan",
                         fontsize=7, va="center")
    else:
        axes[1].text(0.5, 0.5, "p-values not available",
                     transform=axes[1].transAxes, ha="center", va="center")
        axes[1].set_axis_off()

    # Y-tick labels
    for ax in axes:
        if unit_labels is not None:
            ax.set_yticks(range(S))
            ax.set_yticklabels(unit_labels)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for ext in (fmt, ):
            path = os.path.join(save_dir, f"coupling_summary_{effect_type}.{ext}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            logger.info(f"Saved {path}")

    return fig


# =========================================================================
# Spectrogram
# =========================================================================

def plot_spectrogram(
    results: Dict[str, Any],
    *,
    method: str = "joint",
    db_scale: bool = True,
    figsize: Tuple[float, float] = (10, 4),
    save_dir: Optional[str] = None,
    fmt: str = "pdf",
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot the inferred spectrogram (taper-averaged power over time).

    Parameters
    ----------
    results : dict
        Must contain ``'spectral'`` with ``Y_cube`` or ``Z_smooth_joint``.
    method : ``"joint"`` | ``"em"`` | ``"multitaper"``
        Which estimate to plot.
    db_scale : bool
        Convert power to dB (10 log10).
    """
    _set_style()

    spectral = results.get("spectral", {})
    coupling = results.get("coupling", {})
    metadata = results.get("metadata", {})
    freqs = coupling.get("freqs", spectral.get("freqs"))
    if freqs is None:
        raise ValueError("No frequency grid found in results.")

    # Pick the right spectrogram array
    if method == "joint" and "Z_smooth_joint" in spectral:
        Z = spectral["Z_smooth_joint"]  # (J, K) complex
        title_suffix = "Joint SSMT"
    elif method == "em" and "Z_smooth_em" in spectral:
        Z = spectral["Z_smooth_em"]
        title_suffix = "CT-SSMT (LFP-only)"
    elif "Y_cube" in spectral:
        # Raw multitaper: average over tapers
        Y = spectral["Y_cube"]  # (J, M, K) or (R, J, M, K)
        if Y.ndim == 4:
            Y = Y.mean(axis=0)  # average over trials
        Z = Y.mean(axis=1)  # average over tapers -> (J, K)
        title_suffix = "Multitaper"
    elif "Y_trials" in spectral:
        Y = spectral["Y_trials"]  # (R, J, M, K)
        Z = Y.mean(axis=(0, 2))  # average trials + tapers -> (J, K)
        title_suffix = "Multitaper (trial-averaged)"
    else:
        raise ValueError("No spectrogram data found. Re-run with save_spectral=True.")

    power = np.abs(Z) ** 2
    if db_scale:
        power = 10 * np.log10(np.maximum(power, 1e-20))
        clabel = "Power (dB)"
    else:
        clabel = "Power"

    J, K = power.shape
    window_sec = metadata.get("ctssmt", {}).get("window_sec", 1.0)
    t_blocks = np.arange(K) * window_sec

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.pcolormesh(
        t_blocks, freqs, power,
        shading="auto", cmap="inferno",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Spectrogram — {title_suffix}", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(clabel)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"spectrogram_{method}.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved {path}")

    return fig


# =========================================================================
# Trial-averaged dynamics
# =========================================================================

def plot_trial_averaged_dynamics(
    results: Dict[str, Any],
    *,
    freq_indices: Optional[List[int]] = None,
    n_freqs_to_show: int = 6,
    figsize: Tuple[float, float] = (12, 6),
    save_dir: Optional[str] = None,
    fmt: str = "pdf",
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot trial-averaged amplitude trajectories at selected frequencies.

    For trial-structured data, shows the shared (X) component and
    optionally overlays individual trial deviations.

    Parameters
    ----------
    results : dict
        Must contain ``'spectral'`` with ``X_fine`` and optionally ``D_fine``.
    freq_indices : list of int, optional
        Which frequency indices to plot.  If *None*, selects the top
        ``n_freqs_to_show`` by coupling strength.
    n_freqs_to_show : int
        Number of frequencies to auto-select when *freq_indices* is None.
    """
    _set_style()

    spectral = results.get("spectral", {})
    coupling = results.get("coupling", {})
    metadata = results.get("metadata", {})
    freqs = coupling.get("freqs", spectral.get("freqs"))
    if freqs is None:
        raise ValueError("No frequency grid in results.")

    J = len(freqs)
    M = metadata.get("ctssmt", {}).get("n_tapers", 1)
    ds = int(spectral.get("downsample_factor", 10))
    delta_spk = metadata.get("data", {}).get("delta_spk", 0.001)
    dt_fine = delta_spk * ds

    # Auto-select frequencies by coupling strength
    if freq_indices is None:
        beta_mag = coupling.get("beta_mag")
        if beta_mag is not None:
            max_per_freq = beta_mag.max(axis=0)  # (J,)
            freq_indices = list(np.argsort(max_per_freq)[::-1][:n_freqs_to_show])
        else:
            freq_indices = list(range(min(n_freqs_to_show, J)))

    n_plot = len(freq_indices)

    # Decode X_fine → amplitude at each frequency
    # X_fine is (T_ds, 2*J*M) interleaved: [Re(j=0,m=0), Im(j=0,m=0), ...]
    X_fine = spectral.get("X_fine")
    D_fine = spectral.get("D_fine")  # (R, T_ds, 2*J*M)

    if X_fine is None:
        raise ValueError("No X_fine in spectral results. Re-run with save_spectral=True.")

    T_ds = X_fine.shape[0]
    t_axis = np.arange(T_ds) * dt_fine

    fig, axes = plt.subplots(n_plot, 1, figsize=figsize, sharex=True,
                             constrained_layout=True)
    if n_plot == 1:
        axes = [axes]

    for i, j_idx in enumerate(freq_indices):
        ax = axes[i]
        # Taper-averaged amplitude for X
        amp_x = np.zeros(T_ds)
        for m in range(M):
            col_re = 2 * (j_idx * M + m)
            col_im = col_re + 1
            amp_x += X_fine[:, col_re] ** 2 + X_fine[:, col_im] ** 2
        amp_x = np.sqrt(amp_x / M)

        ax.plot(t_axis, amp_x, color="#F18F01", lw=1.5, label="Shared (X)")

        # Overlay individual trial deviations if available
        if D_fine is not None and D_fine.ndim == 3:
            R = D_fine.shape[0]
            n_show = min(R, 4)  # show a few trials
            for r in range(n_show):
                amp_r = np.zeros(T_ds)
                for m in range(M):
                    col_re = 2 * (j_idx * M + m)
                    col_im = col_re + 1
                    z_re = X_fine[:, col_re] + D_fine[r, :, col_re]
                    z_im = X_fine[:, col_im] + D_fine[r, :, col_im]
                    amp_r += z_re ** 2 + z_im ** 2
                amp_r = np.sqrt(amp_r / M)
                ax.plot(t_axis, amp_r, color="gray", alpha=0.3, lw=0.6)

        ax.set_ylabel(f"{freqs[j_idx]:.0f} Hz")
        ax.set_title(f"{freqs[j_idx]:.0f} Hz", fontsize=8, loc="left")

        if i == 0:
            ax.legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Trial-Averaged Amplitude Dynamics", fontweight="bold", fontsize=10)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"trial_dynamics.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved {path}")

    return fig


# =========================================================================
# Master function — called by runners when plot=True
# =========================================================================

def plot_all_default(
    results: Dict[str, Any],
    save_dir: str,
    *,
    fmt: str = "pdf",
    dpi: int = 300,
    close: bool = True,
) -> List[str]:
    """
    Generate all default summary figures and save them.

    Called automatically by ``run_inference`` / ``run_inference_trials``
    when ``plot=True``.

    Returns a list of saved file paths.
    """
    saved = []
    figures_dir = os.path.join(save_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Coupling heatmap (Wald)
    try:
        fig = plot_coupling_summary(
            results, effect_type="wald", save_dir=figures_dir, fmt=fmt, dpi=dpi,
        )
        saved.append(os.path.join(figures_dir, f"coupling_summary_wald.{fmt}"))
        if close:
            plt.close(fig)
    except Exception as e:
        logger.warning(f"Skipping coupling heatmap (wald): {e}")

    # 2. Coupling heatmap (phase)
    try:
        fig = plot_coupling_summary(
            results, effect_type="phase", save_dir=figures_dir, fmt=fmt, dpi=dpi,
        )
        saved.append(os.path.join(figures_dir, f"coupling_summary_phase.{fmt}"))
        if close:
            plt.close(fig)
    except Exception as e:
        logger.debug(f"Skipping coupling heatmap (phase): {e}")

    # 3. Spectrogram
    if "spectral" in results:
        for method in ("joint", "multitaper"):
            try:
                fig = plot_spectrogram(
                    results, method=method, save_dir=figures_dir, fmt=fmt, dpi=dpi,
                )
                saved.append(os.path.join(figures_dir, f"spectrogram_{method}.{fmt}"))
                if close:
                    plt.close(fig)
            except Exception as e:
                logger.debug(f"Skipping spectrogram ({method}): {e}")

    # 4. Trial-averaged dynamics (if trial data)
    if "spectral" in results and "X_fine" in results.get("spectral", {}):
        try:
            fig = plot_trial_averaged_dynamics(
                results, save_dir=figures_dir, fmt=fmt, dpi=dpi,
            )
            saved.append(os.path.join(figures_dir, f"trial_dynamics.{fmt}"))
            if close:
                plt.close(fig)
        except Exception as e:
            logger.debug(f"Skipping trial dynamics: {e}")

    logger.info(f"Saved {len(saved)} default figures to {figures_dir}/")
    return saved
