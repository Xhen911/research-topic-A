"""
Spectral weight transfer (SWT) — intra/inter band decomposition.

Moved from propagators/lindhard.py during L4 refactor (PR-4).
SWT is an observable (L4): it post-processes the polarisation Pi_0
to quantify how spectral weight is distributed between intra-band
(plasmon) and inter-band (single-particle) channels.

Note: the current implementation uses a free-electron vF boundary
to partition (q, w) space.  For TBG flat bands this needs to be
generalised — flagged for a future physics PR, not this refactor.
"""

import numpy as np


def compute_swt_1d(
    q_val: float,
    w_values: np.ndarray,
    pi0_dict: dict,
    q_values: np.ndarray,
    Ef: float,
    vf: float = 1.0,
):
    """1D spectral-weight integration at a selected q value.

    Uses Simpson integration along the omega axis to compute
    partial weights of -Im[Pi_0].

    Parameters
    ----------
    q_val : float
        Target q value.
    w_values : np.ndarray, shape (nw,)
        Frequency grid.
    pi0_dict : dict
        Dictionary {'intra': ..., 'inter': ...} from lindhard_polarization.
    q_values : np.ndarray, shape (nq,)
        q-value grid.
    Ef : float
        Fermi energy.
    vf : float
        Fermi velocity (used to set region boundaries).

    Returns
    -------
    weight_intra : float
    weight_inter : float
    total_weight : float
    """
    from scipy.integrate import simpson

    iq = np.argmin(np.abs(q_values - q_val))
    actual_q = q_values[iq]

    spec_intra = -pi0_dict['intra'][:, iq].imag
    spec_inter = -pi0_dict['inter'][:, iq].imag

    omega_intra_max = vf * actual_q
    omega_inter_min = max(2.0 * Ef - vf * actual_q, vf * actual_q)

    mask_intra = w_values <= omega_intra_max
    mask_inter = w_values >= omega_inter_min

    weight_intra = simpson(spec_intra[mask_intra], w_values[mask_intra])
    weight_inter = simpson(spec_inter[mask_inter], w_values[mask_inter])
    total_weight = simpson(spec_intra + spec_inter, w_values)

    return weight_intra, weight_inter, total_weight


def compute_swt_2d(
    q_values: np.ndarray,
    w_values: np.ndarray,
    pi0_dict: dict,
    Ef: float,
    vf: float = 1.0,
):
    """2D macroscopic-region spectral-weight integration.

    Partitions the (q, w) plane into intra/inter regions and
    integrates spectral weight separately.

    Parameters
    ----------
    q_values : np.ndarray, shape (nq,)
    w_values : np.ndarray, shape (nw,)
    pi0_dict : dict
    Ef : float
    vf : float

    Returns
    -------
    total_intra_weight : float
    total_inter_weight : float
    """
    Q, W = np.meshgrid(q_values, w_values)

    full_intra = -pi0_dict['intra'].imag
    full_inter = -pi0_dict['inter'].imag

    intra_mask = W <= vf * Q
    inter_mask = (W >= 2 * Ef - vf * Q) & (W >= vf * Q)

    dq = q_values[1] - q_values[0]
    dw = w_values[1] - w_values[0]
    da = dq * dw

    total_intra_weight = np.sum(full_intra[intra_mask]) * da
    total_inter_weight = np.sum(full_inter[inter_mask]) * da

    print("\n=== 2D macroscopic region integration ===")
    print(f"[*] Intra-band total spectral weight: {total_intra_weight:.4f}")
    print(f"[*] Inter-band total spectral weight: {total_inter_weight:.4f}")

    return total_intra_weight, total_inter_weight
