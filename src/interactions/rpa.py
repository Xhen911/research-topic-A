"""
RPA interaction — Coulomb potential and screened polarisation.

Split from response/dielectric.py during L3 refactor (PR-4).
The interaction kernel V(q) and the RPA screening Π_RPA = Π₀ / (1 − V·Π₀)
belong to L3 (Interactions), while the observable quantities ε, ε⁻¹, ELF
moved to observables/dielectric.py (L4).
"""

import numpy as np

PI = np.pi


def coulomb_2d(q_values: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """2D Coulomb interaction V(q) = 2*pi / q * alpha.

    Parameters
    ----------
    q_values : np.ndarray, shape (nq,)
        Momentum transfer magnitudes.
    alpha : float
        Coupling constant.  Suspended graphene alpha ~ 2.2,
        simplified models often use ~ 3.77.

    Returns
    -------
    v_q : np.ndarray, shape (nq,)
    """
    q_safe = np.where(q_values < 1e-12, 1e-12, q_values)
    return alpha * 2 * PI / q_safe


def rpa_response(pi0: np.ndarray, v_q: np.ndarray) -> np.ndarray:
    """RPA response: Pi_RPA = Pi_0 / (1 - V_q * Pi_0).

    Parameters
    ----------
    pi0 : np.ndarray, shape (nw, nq), complex
        Lindhard polarisation Pi_0(q, w).
    v_q : np.ndarray, shape (nq,)
        Coulomb interaction V(q).

    Returns
    -------
    pi_rpa : np.ndarray, shape (nw, nq), complex
    """
    pi_rpa = np.zeros_like(pi0, dtype=complex)
    for iq in range(len(v_q)):
        pi_rpa[:, iq] = pi0[:, iq] / (1.0 - v_q[iq] * pi0[:, iq])
    return pi_rpa
