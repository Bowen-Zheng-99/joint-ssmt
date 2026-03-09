"""
JAX implementation of single-trial CT-SSMT latent upsampling.

Fully replicates upsample_ct_single_fine.py using JAX for GPU acceleration.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, NamedTuple
from functools import partial

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
import numpy as np

EPS = 1e-12


class UpsampleSingleResultJAX(NamedTuple):
    """Result of single-trial upsampling (JAX version)."""
    Z_mean: jnp.ndarray      # (J, M, Tf) complex
    Z_var: jnp.ndarray       # (J, M, Tf) real
    t_idx_of_k: jnp.ndarray  # (K,) block center indices
    centres_sec: jnp.ndarray # (K,) block center times


def centres_from_win(K: int, win_sec: float, offset_sec: float) -> jnp.ndarray:
    """Compute block centre times."""
    return offset_sec + jnp.arange(K, dtype=jnp.float64) * float(win_sec)


def map_blocks_to_fine(centres_sec: jnp.ndarray, delta_spk: float, T_f: int) -> jnp.ndarray:
    """Map block centres to fine-grid indices."""
    idx = jnp.round(centres_sec / float(delta_spk)).astype(jnp.int32)
    return jnp.clip(idx, 0, T_f - 1)


def build_t2k(centres_sec: jnp.ndarray, delta_spk: float, T_f: int):
    """
    Build fine-time to block-index lookup table.
    
    Returns
    -------
    t2k : (T_f, max_k) int32
        Lookup table, -1 for invalid entries
    kcount : (T_f,) int32
        Number of valid observations at each time
    """
    # Use numpy for index building (runs on CPU anyway)
    centres_np = np.asarray(centres_sec)
    t_idx = np.round(centres_np / float(delta_spk)).astype(np.int64)
    t_idx = np.clip(t_idx, 0, T_f - 1)
    T_f = int(T_f)

    buckets = [[] for _ in range(T_f)]
    for k, t in enumerate(t_idx):
        buckets[int(t)].append(k)

    kcount = np.array([len(b) for b in buckets], dtype=np.int32)
    max_k = int(kcount.max(initial=1))

    t2k = np.full((T_f, max_k), -1, dtype=np.int32)
    for t in range(T_f):
        row = buckets[t]
        if row:
            t2k[t, :len(row)] = np.asarray(row, dtype=np.int32)

    return jnp.array(t2k), jnp.array(kcount)


def _kalman_update_step(z_upd: jnp.ndarray, P_upd: jnp.ndarray, 
                        y: jnp.ndarray, R: float, valid: bool):
    """
    Single Kalman update step.
    
    Parameters
    ----------
    z_upd : complex
        Current state estimate
    P_upd : float
        Current variance estimate
    y : complex
        Observation
    R : float
        Observation noise variance
    valid : bool
        Whether this observation is valid
    """
    S = P_upd + R
    K = P_upd / jnp.maximum(S, 1e-18)
    
    z_new = z_upd + K * (y - z_upd)
    P_new = jnp.maximum((1.0 - K) * P_upd, 1e-16)
    
    # Only update if valid
    z_upd = jnp.where(valid, z_new, z_upd)
    P_upd = jnp.where(valid, P_new, P_upd)
    
    return z_upd, P_upd


def _smooth_fine_ou_complex_jax(phi: float, q: float, R: float, 
                                 yk: jnp.ndarray, t2k: jnp.ndarray, 
                                 kcount: jnp.ndarray, z0: complex, P0: float):
    """
    Kalman smoother on fine grid with block observations (JAX version).
    
    Parameters
    ----------
    phi : float
        AR coefficient
    q : float  
        Process noise variance
    R : float
        Observation noise variance
    yk : (K,) complex
        Block observations
    t2k : (Tf, max_k) int32
        Time-to-observation lookup table
    kcount : (Tf,) int32
        Observation count at each time
    z0 : complex
        Initial state
    P0 : float
        Initial variance
        
    Returns
    -------
    m_s : (Tf,) complex
        Smoothed means
    P_s : (Tf,) real
        Smoothed variances
    """
    Tf = t2k.shape[0]
    
    def forward_step(carry, t):
        z, P = carry
        t2k_row = t2k[t]
        
        # Predict
        z_pred = phi * z
        P_pred = phi * phi * P + q
        
        z_upd = z_pred
        P_upd = P_pred
        
        # Apply updates for all observations at this time step
        def update_loop(carry, k_idx):
            z_u, P_u = carry
            valid = k_idx >= 0
            safe_idx = jnp.maximum(k_idx, 0)
            y = yk[safe_idx]
            z_u, P_u = _kalman_update_step(z_u, P_u, y, R, valid)
            return (z_u, P_u), None
        
        (z_upd, P_upd), _ = lax.scan(update_loop, (z_upd, P_upd), t2k_row)
        
        return (z_upd, P_upd), (z_pred, P_pred, z_upd, P_upd)
    
    # Forward pass
    init_carry = (jnp.asarray(z0, dtype=jnp.complex128), 
                  jnp.asarray(P0, dtype=jnp.float64))
    
    (z_final, P_final), (m_p, P_p, m_f, P_f) = lax.scan(
        forward_step, init_carry, jnp.arange(Tf)
    )
    
    # Backward pass (RTS smoother)
    init_smooth = (m_f[-1], P_f[-1])
    
    def backward_step(carry, t):
        m_s_next, P_s_next = carry
        m_f_t = m_f[t]
        P_f_t = P_f[t]
        m_p_next = m_p[t + 1]
        P_p_next = P_p[t + 1]
        
        denom = jnp.maximum(P_p_next, 1e-16)
        Jt = (P_f_t * phi) / denom
        m_s_t = m_f_t + Jt * (m_s_next - m_p_next)
        P_s_t = jnp.maximum(P_f_t + Jt * Jt * (P_s_next - P_p_next), 1e-16)
        
        return (m_s_t, P_s_t), (m_s_t, P_s_t)
    
    # Scan backwards from Tf-2 to 0
    _, (m_s_rev, P_s_rev) = lax.scan(
        backward_step, init_smooth, jnp.arange(Tf - 2, -1, -1)
    )
    
    # Reverse to get forward order and append last element
    m_s = jnp.concatenate([m_s_rev[::-1], m_f[-1:]])
    P_s = jnp.concatenate([P_s_rev[::-1], P_f[-1:]])
    
    return m_s, P_s


def upsample_ct_single_fine_jax(
    *,
    Y: jnp.ndarray,           # (J, M, K) complex
    res,                       # EMSingleResult or compatible object
    delta_spk: float,
    win_sec: float,
    offset_sec: float = 0.0,
    T_f: Optional[int] = None,
) -> UpsampleSingleResultJAX:
    """
    Upsample single-trial CT-SSMT latents to fine grid (JAX version).
    
    Parameters
    ----------
    Y : (J, M, K) complex
        Multitaper spectrogram
    res : EMSingleResult
        EM result with lam, sigv, sig_eps, Z_mean, Z_var
    delta_spk : float
        Fine grid time step (seconds)
    win_sec : float
        Block window duration (seconds)
    offset_sec : float
        Time offset to first block
    T_f : int, optional
        Fine grid length (auto-computed if None)
        
    Returns
    -------
    UpsampleSingleResultJAX
        Z_mean: (J, M, Tf) complex smoothed means
        Z_var: (J, M, Tf) real smoothed variances
    """
    Y = jnp.asarray(Y, dtype=jnp.complex128)
    if Y.ndim != 3:
        raise ValueError(f"Y must be (J, M, K), got {Y.shape}")
    
    J, M, K = Y.shape
    
    # Extract EM parameters
    lam = jnp.asarray(res.lam, dtype=jnp.float64)
    sigv = jnp.asarray(res.sigv, dtype=jnp.float64)
    sig_eps = jnp.asarray(res.sig_eps, dtype=jnp.float64)
    
    # Handle shape - could be (J,M) or (J,) depending on EM output
    if lam.ndim == 1:
        lam = jnp.broadcast_to(lam[:, None], (J, M))
        sigv = jnp.broadcast_to(sigv[:, None], (J, M))
        sig_eps = jnp.broadcast_to(sig_eps[:, None], (J, M))
    
    assert lam.shape == (J, M), f"lam shape mismatch: {lam.shape} vs ({J}, {M})"
    
    # Get initial states if available
    if hasattr(res, 'x0') and res.x0 is not None:
        x0 = jnp.asarray(res.x0, dtype=jnp.complex128)
        if x0.ndim == 1:
            x0 = jnp.broadcast_to(x0[:, None], (J, M))
    else:
        x0 = jnp.zeros((J, M), dtype=jnp.complex128)
    
    if hasattr(res, 'P0') and res.P0 is not None:
        P0 = jnp.asarray(res.P0, dtype=jnp.float64)
        if P0.ndim == 1:
            P0 = jnp.broadcast_to(P0[:, None], (J, M))
    else:
        P0 = (sigv**2 / jnp.maximum(2.0 * lam, EPS))
    
    # Fine grid setup
    centres_sec = centres_from_win(K, win_sec, offset_sec)
    t_end = float(centres_sec[-1]) + 0.5 * float(win_sec)
    
    if T_f is None:
        T_f = int(round(t_end / float(delta_spk)))
    assert T_f > 0, "T_f must be positive"
    
    t_idx_of_k = map_blocks_to_fine(centres_sec, delta_spk, T_f)
    t2k, kcount = build_t2k(centres_sec, delta_spk, T_f)
    
    # OU parameters at fine step
    phi = jnp.exp(-lam * float(delta_spk))
    q = (sigv**2) * (1.0 - jnp.exp(-2.0 * lam * float(delta_spk))) / jnp.maximum(2.0 * lam, EPS)
    R = sig_eps**2
    
    # Upsample each (j, m) chain using vmap
    def smooth_chain(phi_jm, q_jm, R_jm, Y_jm, x0_jm, P0_jm):
        return _smooth_fine_ou_complex_jax(
            phi_jm, q_jm, R_jm, Y_jm, t2k, kcount, x0_jm, P0_jm
        )
    
    # vmap over m dimension, then over j dimension
    smooth_over_m = jax.vmap(smooth_chain, in_axes=(0, 0, 0, 0, 0, 0))
    smooth_over_jm = jax.vmap(smooth_over_m, in_axes=(0, 0, 0, 0, 0, 0))
    
    Z_mean, Z_var = smooth_over_jm(phi, q, R, Y, x0, P0)
    
    return UpsampleSingleResultJAX(
        Z_mean=Z_mean,
        Z_var=Z_var,
        t_idx_of_k=t_idx_of_k,
        centres_sec=centres_sec,
    )


# Convenience function for JIT compilation
def make_jit_upsampler(T_f: int, K: int, max_obs_per_time: int):
    """
    Create a JIT-compiled upsampler for fixed dimensions.
    
    Use this for repeated calls with same grid sizes.
    """
    @jax.jit
    def _upsample_jit(Y, lam, sigv, sig_eps, x0, P0, 
                      delta_spk, win_sec, offset_sec, t2k, kcount):
        J, M, _ = Y.shape
        
        # OU parameters at fine step
        phi = jnp.exp(-lam * delta_spk)
        q = (sigv**2) * (1.0 - jnp.exp(-2.0 * lam * delta_spk)) / jnp.maximum(2.0 * lam, EPS)
        R = sig_eps**2
        
        def smooth_chain(phi_jm, q_jm, R_jm, Y_jm, x0_jm, P0_jm):
            return _smooth_fine_ou_complex_jax(
                phi_jm, q_jm, R_jm, Y_jm, t2k, kcount, x0_jm, P0_jm
            )
        
        smooth_over_m = jax.vmap(smooth_chain, in_axes=(0, 0, 0, 0, 0, 0))
        smooth_over_jm = jax.vmap(smooth_over_m, in_axes=(0, 0, 0, 0, 0, 0))
        
        return smooth_over_jm(phi, q, R, Y, x0, P0)
    
    return _upsample_jit
