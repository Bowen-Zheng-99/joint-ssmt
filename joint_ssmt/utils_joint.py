"""
utils_joint.py - Trace container for MCMC inference.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from joint_ssmt.params import OUParams


@dataclass
class Trace:
    """Container for MCMC trace samples.

    Attributes
    ----------
    beta : list of (S, D) arrays
        Coupling coefficient samples.
    gamma : list of (S, L) arrays
        Spike-history coefficient samples.
    theta : list of OUParams
        OU parameter snapshots across iterations.
    latent : list of arrays
        Latent state snapshots (combined X+D for trial-structured).
    fine_latent : list of arrays
        Fine-resolution latent states.

    Notes
    -----
    Single-trial inference returns a 4-tuple ``(beta, gamma, theta, trace)``
    where *theta* is one :class:`OUParams`.

    Hierarchical (trial-structured) inference returns a 5-tuple
    ``(beta, gamma, theta_X, theta_D, trace)`` with separate OU parameters
    for the shared component (*theta_X*) and trial deviations (*theta_D*).
    """
    beta:        List[np.ndarray] = field(default_factory=list)
    gamma:       List[np.ndarray] = field(default_factory=list)
    theta:       List["OUParams"]  = field(default_factory=list)
    latent:      List[np.ndarray] = field(default_factory=list)
    fine_latent: List[np.ndarray] = field(default_factory=list)
