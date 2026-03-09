# src/upsample_ct_hier_fine.py
"""
Upsample hierarchical CT-SSMT latents (X shared, D trial-specific) to fine grid.

Compatible with both zero-mean and constant-mean OU from em_ct_hier_jax.
Uses mu_X and mu_D from EMHierResult when available (backward compatible).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from joint_ssmt.utils_common import (
    centres_from_win,
    map_blocks_to_fine,
    build_t2k,
    normalize_Y_to_RJMK,
    EPS,
)


@dataclass
class UpsampleResult:
    X_mean: np.ndarray     # (J,M,Tf) complex
    X_var:  np.ndarray     # (J,M,Tf) real
    D_mean: np.ndarray     # (R,J,M,Tf) complex
    D_var:  np.ndarray     # (R,J,M,Tf) real
    Z_mean: np.ndarray     # (R,J,M,Tf) complex
    Z_var:  np.ndarray     # (R,J,M,Tf) real
    t_idx_of_k: np.ndarray # (K,)
    centres_sec: np.ndarray# (K,)


# ---------------- helpers (shape-strict) ----------------
def _expand_sig_eps_to_JMR(sig_eps_jmr: np.ndarray, J: int, M: int, R: int) -> np.ndarray:
    arr = np.asarray(sig_eps_jmr)
    if arr.shape == (J, M, R):  return arr
    if arr.shape == (1, M, R):  return np.tile(arr, (J, 1, 1))
    if arr.shape == (1, 1, R):  return np.tile(arr, (J, M, 1))
    raise AssertionError(f"sig_eps_jmr must be (J,M,R) or (1,M,R) or (1,1,R); got {arr.shape}")


# ---------------- scalar complex OU on fine grid (Numba + fallback) ----------------
_HAS_NUMBA = False
try:
    import numba as nb
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

if _HAS_NUMBA:
    @nb.njit(cache=True, fastmath=False)
    def _smooth_fine_ou_complex_numba(phi, q, R, yk, t2k, kcount, z0, P0):
        """
        Scalar complex OU RTS smoother on fine grid. Zero-mean OU.
        """
        Tf = t2k.shape[0]
        m_p = np.empty(Tf, np.complex128)
        P_p = np.empty(Tf, np.float64)
        m_f = np.empty(Tf, np.complex128)
        P_f = np.empty(Tf, np.float64)
        z = z0
        P = P0
        for t in range(Tf):
            z_pred = phi * z
            P_pred = phi*phi * P + q
            z_upd = z_pred
            P_upd = P_pred
            kc = kcount[t]
            for i in range(kc):
                k = t2k[t, i]
                if k < 0: break
                y = yk[k]
                S = P_upd + R
                K = P_upd / (S if S > 1e-18 else 1e-18)
                z_upd = z_upd + K * (y - z_upd)
                P_upd = (1.0 - K) * P_upd
                if P_upd < 1e-16: P_upd = 1e-16
            m_p[t] = z_pred; P_p[t] = P_pred
            m_f[t] = z_upd;  P_f[t] = P_upd
            z = z_upd; P = P_upd

        m_s = np.empty(Tf, np.complex128)
        P_s = np.empty(Tf, np.float64)
        m_s[Tf-1] = m_f[Tf-1]
        P_s[Tf-1] = P_f[Tf-1]
        for t in range(Tf-2, -1, -1):
            denom = P_p[t+1] if P_p[t+1] > 1e-16 else 1e-16
            Jt = (P_f[t] * phi) / denom
            m_s[t] = m_f[t] + Jt * (m_s[t+1] - m_p[t+1])
            P_s[t] = P_f[t] + Jt*Jt * (P_s[t+1] - P_p[t+1])
            if P_s[t] < 1e-16: P_s[t] = 1e-16
        return m_s, P_s

else:
    # Pure Python fallback (no Numba)
    def _smooth_fine_ou_complex_numba(phi, q, R, yk, t2k, kcount, z0, P0):
        """Scalar complex OU RTS smoother on fine grid (pure Python fallback). Zero-mean OU."""
        Tf = t2k.shape[0]
        m_p = np.empty(Tf, np.complex128)
        P_p = np.empty(Tf, np.float64)
        m_f = np.empty(Tf, np.complex128)
        P_f = np.empty(Tf, np.float64)
        z = complex(z0)
        P = float(P0)
        for t in range(Tf):
            z_pred = phi * z
            P_pred = phi*phi * P + q
            z_upd = z_pred
            P_upd = P_pred
            kc = kcount[t]
            for i in range(kc):
                k = t2k[t, i]
                if k < 0: break
                y = yk[k]
                S = P_upd + R
                K = P_upd / max(S, 1e-18)
                z_upd = z_upd + K * (y - z_upd)
                P_upd = max((1.0 - K) * P_upd, 1e-16)
            m_p[t] = z_pred; P_p[t] = P_pred
            m_f[t] = z_upd;  P_f[t] = P_upd
            z = z_upd; P = P_upd

        m_s = np.empty(Tf, np.complex128)
        P_s = np.empty(Tf, np.float64)
        m_s[-1] = m_f[-1]
        P_s[-1] = P_f[-1]
        for t in range(Tf - 2, -1, -1):
            denom = max(P_p[t+1], 1e-16)
            Jt = (P_f[t] * phi) / denom
            m_s[t] = m_f[t] + Jt * (m_s[t+1] - m_p[t+1])
            P_s[t] = max(P_f[t] + Jt*Jt * (P_s[t+1] - P_p[t+1]), 1e-16)
        return m_s, P_s


def _smooth_fine_ou_with_mean(phi, q, R, yk, t2k, kcount, z0, P0, mu):
    """
    Kalman smoother with constant-mean OU.
    
    Model: (Z - μ)_{t+1} = φ·(Z - μ)_t + η
    
    Implementation: center by μ, run zero-mean smoother, add μ back.
    """
    # Center observations and initial state
    yk_centered = yk - mu
    z0_centered = z0 - mu
    
    # Run zero-mean smoother on centered data
    xs_centered, Ps = _smooth_fine_ou_complex_numba(
        phi, q, R, yk_centered, t2k, kcount, z0_centered, P0
    )
    
    # Add mean back
    xs = xs_centered + mu
    
    return xs, Ps


# ---------------- main upsampler ----------------
def upsample_ct_hier_fine(
    *,
    Y_trials: np.ndarray,        # (R,J,M,K) or (R,M,J,K), complex
    res,                          # EMHierResult (jax/np arrays OK)
    delta_spk: float,
    win_sec: float,
    offset_sec: float,
    T_f: Optional[int] = None,
) -> UpsampleResult:
    """
    Upsample hierarchical CT-SSMT latents to fine (spike-resolution) grid.
    
    Automatically uses mu_X and mu_D from EMHierResult if available.
    Falls back to zero-mean OU if mu_X/mu_D not present (backward compatible).
    
    Parameters
    ----------
    Y_trials : (R,J,M,K) or (R,M,J,K) complex
        Multitaper coefficients per trial
    res : EMHierResult
        EM result containing parameters and block-level estimates
    delta_spk : float
        Fine grid time step (seconds), e.g. 0.001 for 1ms
    win_sec : float
        Block window duration (seconds)
    offset_sec : float
        Time offset to first block center
    T_f : int, optional
        Fine grid length (auto-computed if None)
        
    Returns
    -------
    UpsampleResult
        Contains X_mean, X_var, D_mean, D_var, Z_mean, Z_var at fine resolution
    """
    # (1) Read canonical (J,M) from EM (no guessing)
    lam_X = np.asarray(res.lam_X, float)
    assert lam_X.ndim == 2, "lam_X must be (J,M)"
    J, M = lam_X.shape

    # (2) Normalize Y to (R,J,M,K)
    Y_RJMK = normalize_Y_to_RJMK(Y_trials, J, M).astype(np.complex128)
    R, _, _, K = Y_RJMK.shape

    # (3) Strict shape checks
    sigv_X = np.asarray(res.sigv_X, float);  assert sigv_X.shape == (J, M)
    lam_D  = np.asarray(res.lam_D,  float);  assert lam_D.shape  == (J, M)
    sigv_D = np.asarray(res.sigv_D, float);  assert sigv_D.shape == (J, M)

    x0_X = np.asarray(res.x0_X, np.complex128); assert x0_X.shape == (J, M)
    P0_X = np.asarray(res.P0_X, float);         assert P0_X.shape == (J, M)
    x0_D = np.asarray(res.x0_D, np.complex128); assert x0_D.shape == (R, J, M)
    P0_D = np.asarray(res.P0_D, float);         assert P0_D.shape == (R, J, M)

    X_mean_block = np.asarray(res.X_mean, np.complex128); assert X_mean_block.shape == (J, M, K)
    D_mean_block = np.asarray(res.D_mean, np.complex128); assert D_mean_block.shape == (R, J, M, K)

    sig_eps_jmr = _expand_sig_eps_to_JMR(np.asarray(res.sig_eps_jmr, float), J, M, R)  # (J,M,R)

    # (4) Get mu_X and mu_D from EMHierResult (or default to 0 for backward compat)
    use_mean_reversion = False
    if hasattr(res, 'mu_X') and res.mu_X is not None:
        mu_X = np.asarray(res.mu_X, np.complex128)
        assert mu_X.shape == (J, M), f"mu_X shape mismatch: {mu_X.shape} vs ({J}, {M})"
        use_mean_reversion = True
    else:
        mu_X = np.zeros((J, M), dtype=np.complex128)
    
    if hasattr(res, 'mu_D') and res.mu_D is not None:
        mu_D = np.asarray(res.mu_D, np.complex128)
        assert mu_D.shape == (J, M), f"mu_D shape mismatch: {mu_D.shape} vs ({J}, {M})"
    else:
        # D is always zero-mean (deviations from X)
        mu_D = np.zeros((J, M), dtype=np.complex128)

    # (5) Fine grid length — use right edge of last block so 10s/1ms -> 10000
    centres_sec = centres_from_win(K, win_sec, offset_sec)
    t_end = centres_sec[-1] + 0.5 * float(win_sec)
    if T_f is None:
        T_f = int(round(t_end / float(delta_spk)))
    assert T_f > 0, "T_f must be positive"
    t_idx_of_k = map_blocks_to_fine(centres_sec, delta_spk, T_f)

    # (6) Build block->fine index table (once)
    t2k, kcount = build_t2k(centres_sec, delta_spk, T_f)
    
    # (7) OU params at fine step Δ
    phi_X = np.exp(-lam_X * float(delta_spk))
    q_X   = (sigv_X**2) * (1.0 - np.exp(-2.0 * lam_X * float(delta_spk))) / np.maximum(2.0 * lam_X, EPS)
    phi_D = np.exp(-lam_D * float(delta_spk))
    q_D   = (sigv_D**2) * (1.0 - np.exp(-2.0 * lam_D * float(delta_spk))) / np.maximum(2.0 * lam_D, EPS)

    # (8) Precision-pooled X observation per block
    W_jmr = 1.0 / np.maximum(sig_eps_jmr**2, EPS)                       # (J,M,R)
    Wsum_jm = W_jmr.sum(axis=2)                                         # (J,M)
    Y_res_RJMK = Y_RJMK - D_mean_block                                  # (R,J,M,K)
    num_JMK = (W_jmr.transpose(2,0,1)[:,:,:,None] * Y_res_RJMK).sum(0)  # (J,M,K)
    Y_pool_JMK = num_JMK / np.maximum(Wsum_jm[:, :, None], EPS)         # (J,M,K)
    Var_pool_jm = 1.0 / np.maximum(Wsum_jm, EPS)                        # (J,M)

    # (9) Upsample X on fine grid
    X_mean = np.zeros((J, M, T_f), dtype=np.complex128)
    X_var  = np.zeros((J, M, T_f), dtype=np.float64)
    
    for j in range(J):
        for m in range(M):
            if use_mean_reversion:
                xs, Ps = _smooth_fine_ou_with_mean(
                    float(phi_X[j, m]), float(q_X[j, m]), float(Var_pool_jm[j, m]),
                    Y_pool_JMK[j, m, :], t2k, kcount,
                    complex(x0_X[j, m]), float(P0_X[j, m]),
                    complex(mu_X[j, m])
                )
            else:
                xs, Ps = _smooth_fine_ou_complex_numba(
                    float(phi_X[j, m]), float(q_X[j, m]), float(Var_pool_jm[j, m]),
                    Y_pool_JMK[j, m, :], t2k, kcount,
                    complex(x0_X[j, m]), float(P0_X[j, m])
                )
            X_mean[j, m] = xs
            X_var [j, m] = Ps

    # (10) Upsample δ_r on fine grid
    D_mean = np.zeros((R, J, M, T_f), dtype=np.complex128)
    D_var  = np.zeros((R, J, M, T_f), dtype=np.float64)
    X_at_blocks = X_mean[:, :, t_idx_of_k]  # (J, M, K)

    for r in range(R):
        Y_r_JMK = Y_RJMK[r]
        Y_res_JMK = Y_r_JMK - X_at_blocks   # shape (J,M,K)
        for j in range(J):
            for m in range(M):
                R_obs = float(np.maximum(sig_eps_jmr[j, m, r]**2, EPS))
                # D is always zero-mean (mu_D = 0), so no need for _with_mean
                xs, Ps = _smooth_fine_ou_complex_numba(
                    float(phi_D[j, m]), float(q_D[j, m]), R_obs,
                    Y_res_JMK[j, m, :], t2k, kcount,
                    complex(x0_D[r, j, m]), float(P0_D[r, j, m])
                )
                D_mean[r, j, m] = xs
                D_var [r, j, m] = Ps

    # (11) Combine Z = X + δ (variances add; independent chains)
    Z_mean = X_mean[None, :, :, :] + D_mean
    Z_var  = X_var [None, :, :, :] + D_var

    return UpsampleResult(
        X_mean=X_mean, X_var=X_var,
        D_mean=D_mean, D_var=D_var,
        Z_mean=Z_mean, Z_var=Z_var,
        t_idx_of_k=map_blocks_to_fine(centres_sec, delta_spk, T_f),
        centres_sec=centres_sec,
    )