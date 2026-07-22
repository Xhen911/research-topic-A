"""
Dielectric observables — epsilon, epsilon^{-1}, and energy-loss function.

Split from response/dielectric.py during L4 refactor (PR-4).
These are observable quantities (L4) built from the bare polarisation Pi_0
and the interaction V_q.  The interaction itself (coulomb_2d, rpa_response)
lives in interactions/rpa.py (L3).
"""

import numpy as np
from typing import Tuple


def dielectric_function(
    pi0: np.ndarray, v_q: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Dielectric function: eps(q, w) = 1 - V_q * Pi_0, and its inverse.

    Parameters
    ----------
    pi0 : np.ndarray, shape (nw, nq), complex
        Lindhard polarisation Pi_0(q, w).
    v_q : np.ndarray, shape (nq,)
        Coulomb interaction V(q).

    Returns
    -------
    eps : np.ndarray, shape (nw, nq), complex
    eps_inv : np.ndarray, shape (nw, nq), complex
    """
    eps = np.zeros_like(pi0, dtype=complex)
    eps_inv = np.zeros_like(pi0, dtype=complex)
    for iq in range(len(v_q)):
        eps[:, iq] = 1.0 - v_q[iq] * pi0[:, iq]
        eps_inv[:, iq] = 1.0 / eps[:, iq]
    return eps, eps_inv


def energy_loss_function(pi0: np.ndarray, v_q: np.ndarray) -> np.ndarray:
    """Energy-loss function: -Im[1/eps(q, w)].

    Peaks at plasmon frequencies (eps -> 0).

    Parameters
    ----------
    pi0 : np.ndarray, shape (nw, nq), complex
    v_q : np.ndarray, shape (nq,)

    Returns
    -------
    elf : np.ndarray, shape (nw, nq)
    """
    _, eps_inv = dielectric_function(pi0, v_q)
    return -eps_inv.imag
