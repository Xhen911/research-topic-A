"""
Dynamic structure factor S(q, w) = -(1/pi) * Im[chi_RPA(q, w)].

Nearly free given chi_RPA: the RPA dressed polarisation is already
computed by interactions/rpa.py.  This module wraps the final
observable extraction.
"""

import numpy as np


def structure_factor(pi_rpa: np.ndarray) -> np.ndarray:
    """Dynamic structure factor from dressed polarisation.

    S(q, w) = -(1/pi) * Im[Pi_RPA(q, w)]

    Parameters
    ----------
    pi_rpa : np.ndarray, shape (nw, nq), complex
        Dressed (RPA) polarisation.

    Returns
    -------
    S : np.ndarray, shape (nw, nq)
        Dynamic structure factor (real).
    """
    return -pi_rpa.imag / np.pi
