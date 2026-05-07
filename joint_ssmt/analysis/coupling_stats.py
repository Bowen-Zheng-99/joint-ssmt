"""
Coupling Detection Statistics

Statistical tests for detecting significant spike-field coupling:
1. Wald test - Tests if β ≠ 0 using posterior mean and variance
2. Phase concentration test (Rayleigh) - Tests for unimodal phase distribution

Both tests work on posterior samples from the Gibbs sampler.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional, Sequence


# =============================================================================
# Wald Test
# =============================================================================

def wald_test(
    beta_samples: np.ndarray,
    J: int,
    *,
    burn_in_frac: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Wald statistics and p-values for each (neuron, band) pair.
    
    Tests H0: β_j = 0 (no coupling) using posterior samples.
    
    Wald statistic: W = β̂_R²/Var(β_R) + β̂_I²/Var(β_I) ~ χ²(2)
    
    Parameters
    ----------
    beta_samples : (n_samples, S, P) array
        Posterior samples of β coefficients
        Layout: [β₀, βR_0, ..., βR_{J-1}, βI_0, ..., βI_{J-1}]
    J : int
        Number of frequency bands
    burn_in_frac : float
        Fraction of samples to discard as burn-in
    
    Returns
    -------
    W : (S, J) array
        Wald statistics
    pval : (S, J) array
        P-values from χ²(2) distribution
    """
    n_samples, S, P = beta_samples.shape
    
    # Apply burn-in
    burn_in = int(burn_in_frac * n_samples)
    samples = beta_samples[burn_in:]
    
    # Extract real and imaginary parts
    # Layout: [β₀, βR_0, ..., βR_{J-1}, βI_0, ..., βI_{J-1}]
    beta_R = samples[:, :, 1:1+J]        # (n_post, S, J)
    beta_I = samples[:, :, 1+J:1+2*J]    # (n_post, S, J)
    
    # Posterior mean and variance
    mean_R = beta_R.mean(axis=0)  # (S, J)
    mean_I = beta_I.mean(axis=0)  # (S, J)
    var_R = beta_R.var(axis=0)    # (S, J)
    var_I = beta_I.var(axis=0)    # (S, J)
    
    # Wald statistic
    W = np.zeros((S, J))
    pval = np.ones((S, J))
    
    for s in range(S):
        for j in range(J):
            if var_R[s, j] > 1e-10 and var_I[s, j] > 1e-10:
                W[s, j] = (mean_R[s, j]**2 / var_R[s, j] +
                          mean_I[s, j]**2 / var_I[s, j])
                pval[s, j] = 1 - stats.chi2.cdf(W[s, j], df=2)
            else:
                W[s, j] = 0.0
                pval[s, j] = 1.0
    
    return W, pval


def wald_test_band_selection(
    beta_samples: np.ndarray,
    J: int,
    *,
    alpha: float = 0.05,
    burn_in_frac: float = 0.5,
    verbose: bool = True,
    freqs_hz: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Wald test and select significant bands.
    
    A band is significant if ANY neuron shows significant coupling.
    
    Parameters
    ----------
    beta_samples : (n_samples, S, P) array
    J : int
    alpha : float
        Significance level
    burn_in_frac : float
    verbose : bool
    freqs_hz : optional sequence
        Frequencies for verbose output
    
    Returns
    -------
    significant_mask : (J,) bool
        True if band has any significant coupling
    W : (S, J) array
        Wald statistics
    pval : (S, J) array
        P-values
    """
    W, pval = wald_test(beta_samples, J, burn_in_frac=burn_in_frac)
    
    # Band is significant if ANY neuron shows significance
    significant_mask = (pval < alpha).any(axis=0)  # (J,)
    
    if verbose:
        S = pval.shape[0]
        n_samples = beta_samples.shape[0]
        n_post = n_samples - int(burn_in_frac * n_samples)
        n_sig_bands = significant_mask.sum()
        n_sig_pairs = (pval < alpha).sum()
        total_pairs = S * J
        
        print(f"[WALD] Band selection (α={alpha}, burn-in={burn_in_frac}):")
        print(f"[WALD]   Using {n_post} post-burn-in samples")
        print(f"[WALD]   Significant (neuron,band) pairs: {n_sig_pairs}/{total_pairs}")
        print(f"[WALD]   Bands with ANY significant coupling: {n_sig_bands}/{J}")
        
        if freqs_hz is not None:
            sig_freqs = [f"{freqs_hz[j]:.1f}Hz" for j in range(J) if significant_mask[j]]
            if sig_freqs and len(sig_freqs) <= 10:
                print(f"[WALD]   Coupled bands: {', '.join(sig_freqs)}")
    
    return significant_mask, W, pval


# =============================================================================
# Phase Concentration Test (Rayleigh)
# =============================================================================

def phase_concentration_test(
    beta_samples: np.ndarray,
    J: int,
    *,
    burn_in_frac: float = 0.5,
    min_magnitude_percentile: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Phase Concentration Test (Rayleigh test on posterior phases).
    
    Key insight: Even when |β| is shrunk toward 0, the PHASE is preserved.
    This test detects UNIMODAL phase concentration:
    - Ellipse at origin: phases bimodal (θ and θ+π) → R ≈ 0 → NOT significant
    - Shifted cloud: phases unimodal (at θ only) → R > 0 → significant
    
    Parameters
    ----------
    beta_samples : (n_samples, S, P) array
        Posterior samples
    J : int
        Number of frequency bands
    burn_in_frac : float
        Fraction to discard as burn-in
    min_magnitude_percentile : float
        Filter out samples below this percentile in magnitude
    
    Returns
    -------
    R : (S, J) array
        Mean resultant length (concentration, 0-1)
    pval : (S, J) array
        P-values from Rayleigh test
    phase : (S, J) array
        Estimated coupling phase (radians)
    """
    n_samples, S, P = beta_samples.shape
    
    # Apply burn-in
    burn_in = int(burn_in_frac * n_samples)
    samples = beta_samples[burn_in:]
    
    # Extract real and imaginary parts
    bR = samples[:, :, 1:1+J]        # (n_post, S, J)
    bI = samples[:, :, 1+J:1+2*J]    # (n_post, S, J)
    
    R = np.zeros((S, J))
    pval = np.ones((S, J))
    phase = np.zeros((S, J))
    
    for s in range(S):
        for j in range(J):
            br = bR[:, s, j]
            bi = bI[:, s, j]
            
            # Compute magnitudes and phases
            magnitudes = np.sqrt(br**2 + bi**2)
            phases = np.arctan2(bi, br)
            
            # Filter out very small magnitude samples (noisy phases)
            if min_magnitude_percentile > 0:
                threshold = np.percentile(magnitudes, min_magnitude_percentile)
                mask = magnitudes >= threshold
                phases_filtered = phases[mask]
            else:
                phases_filtered = phases
            
            n = len(phases_filtered)
            if n < 10:
                R[s, j] = 0.0
                pval[s, j] = 1.0
                phase[s, j] = 0.0
                continue
            
            # Compute mean resultant
            C = np.mean(np.cos(phases_filtered))
            S_sin = np.mean(np.sin(phases_filtered))
            R[s, j] = np.sqrt(C**2 + S_sin**2)
            phase[s, j] = np.arctan2(S_sin, C)
            
            # Rayleigh test p-value
            Z = n * R[s, j]**2
            if n < 50:
                # Small sample correction
                pval[s, j] = np.exp(-Z) * (1 + (2*Z - Z**2)/(4*n) - 
                                           (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4)/(288*n**2))
            else:
                pval[s, j] = np.exp(-Z)
            pval[s, j] = max(pval[s, j], 1e-300)
    
    return R, pval, phase


# =============================================================================
# Detection Metrics
# =============================================================================

def compute_detection_metrics(
    y_true: np.ndarray,
    pval: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Compute detection metrics (TP, FP, TN, FN, sensitivity, specificity, etc.)
    
    Parameters
    ----------
    y_true : (S, J) bool array
        Ground truth coupling mask
    pval : (S, J) float array
        P-values from statistical test
    alpha : float
        Significance threshold
    
    Returns
    -------
    dict with keys: TP, FP, TN, FN, sensitivity, specificity, precision, f1
    """
    y_pred = (pval < alpha).flatten()
    y = y_true.flatten().astype(bool)
    
    TP = int((y_pred & y).sum())
    TN = int((~y_pred & ~y).sum())
    FP = int((y_pred & ~y).sum())
    FN = int((~y_pred & y).sum())
    
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    
    return {
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
    }


def compute_roc_auc(
    y_true: np.ndarray,
    pval: np.ndarray,
) -> float:
    """
    Compute ROC-AUC using -log10(p) as score.
    
    Parameters
    ----------
    y_true : (S, J) bool array
    pval : (S, J) float array
    
    Returns
    -------
    auc : float
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return np.nan
    
    y = y_true.flatten().astype(int)
    score = -np.log10(np.clip(pval.flatten(), 1e-10, 1))
    
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    
    return roc_auc_score(y, score)


# =============================================================================
# Phase Error
# =============================================================================

def circular_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute circular difference between angles (in radians)."""
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def compute_phase_mae(
    phase_true: np.ndarray,
    phase_est: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute mean absolute phase error (in radians).
    
    Parameters
    ----------
    phase_true : (S, J) array
        True coupling phases
    phase_est : (S, J) array
        Estimated phases
    mask : (S, J) bool array, optional
        Only compute for masked entries
    
    Returns
    -------
    mae : float
        Mean absolute error in radians
    """
    if mask is not None:
        true = phase_true[mask]
        est = phase_est[mask]
    else:
        true = phase_true.flatten()
        est = phase_est.flatten()
    
    if len(true) == 0:
        return np.nan
    
    err = circular_difference(est, true)
    return np.abs(err).mean()


# =============================================================================
# Summary Statistics from Posterior
# =============================================================================

def summarize_posterior(
    beta_samples: np.ndarray,
    J: int,
    *,
    burn_in_frac: float = 0.5,
    ci_level: float = 0.95,
) -> dict:
    """
    Compute summary statistics from posterior samples.
    
    Parameters
    ----------
    beta_samples : (n_samples, S, P) array
    J : int
    burn_in_frac : float
    ci_level : float
        Credible interval level (e.g., 0.95 for 95% CI)
    
    Returns
    -------
    dict with:
        beta_mean: (S, P)
        beta_std: (S, P)
        beta_ci_lower: (S, P)
        beta_ci_upper: (S, P)
        beta_mag_mean: (S, J)
        beta_mag_std: (S, J)
        beta_phase_mean: (S, J)  # circular mean
    """
    n_samples, S, P = beta_samples.shape
    burn_in = int(burn_in_frac * n_samples)
    samples = beta_samples[burn_in:]
    
    # Point estimates
    beta_mean = samples.mean(axis=0)
    beta_std = samples.std(axis=0)
    
    # Credible intervals
    alpha = 1 - ci_level
    beta_ci_lower = np.percentile(samples, 100 * alpha / 2, axis=0)
    beta_ci_upper = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)
    
    # Magnitude statistics
    bR = samples[:, :, 1:1+J]
    bI = samples[:, :, 1+J:1+2*J]
    mag = np.sqrt(bR**2 + bI**2)
    
    beta_mag_mean = mag.mean(axis=0)
    beta_mag_std = mag.std(axis=0)
    
    # Phase statistics (circular mean)
    phases = np.arctan2(bI, bR)
    C = np.cos(phases).mean(axis=0)
    S_sin = np.sin(phases).mean(axis=0)
    beta_phase_mean = np.arctan2(S_sin, C)
    
    return {
        'beta_mean': beta_mean,
        'beta_std': beta_std,
        'beta_ci_lower': beta_ci_lower,
        'beta_ci_upper': beta_ci_upper,
        'beta_mag_mean': beta_mag_mean,
        'beta_mag_std': beta_mag_std,
        'beta_phase_mean': beta_phase_mean,
    }
