"""
Occupation numbers, filling factor, and CNP solver.

Extracted from response/dos.py during L1 refactor (PR-2).
These are band-level quantities (L1), not response functions (L2):
they depend only on eigenvalues E_k and the degeneracy factor,
not on any propagator or interaction.
"""

import numpy as np


# ============================================================
#  Filling factor & CNP  (flat-pair bisection)
# ============================================================

def compute_filling(E_k, Ef, band_slice=None, kBT=0.1e-3,
                    degeneracy=None, model=None):
    """Filling factor nu at Fermi energy Ef.

    nu(mu) = g * (<Sum_bands f(E_n(k), mu)>_k - <Sum_bands f(E_n(k), E_CNP)>_k)

    Returns nu, where nu=0 at CNP and nu in [-g, +g].
    """
    if degeneracy is None and model is not None:
        degeneracy = model.degeneracy_factor()
    if degeneracy is None:
        degeneracy = 4
    if band_slice is not None:
        E_k = E_k[:, band_slice]
    x = np.clip((E_k - Ef) / kBT, -80.0, 80.0)
    f = 1.0 / (1.0 + np.exp(x))
    occ_per_k = f.sum(axis=1)
    nb_sel = E_k.shape[1]
    return degeneracy * (np.mean(occ_per_k) - nb_sel / 2.0)


def compute_cnp(E_k, band_slice=None, kBT=0.1e-3,
                degeneracy=None, model=None):
    """CNP Fermi energy via bisection.

    Returns Ef such that compute_filling(E_k, Ef) ~ 0.
    """
    if degeneracy is None and model is not None:
        degeneracy = model.degeneracy_factor()
    if degeneracy is None:
        degeneracy = 4
    if band_slice is not None:
        E_k = E_k[:, band_slice]

    lo = float(E_k.min() - 50 * kBT)
    hi = float(E_k.max() + 50 * kBT)
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        x = np.clip((E_k - mid) / kBT, -80.0, 80.0)
        f = 1.0 / (1.0 + np.exp(x))
        occ_mean = np.mean(f.sum(axis=1))
        nb_sel = E_k.shape[1]
        if occ_mean < nb_sel / 2.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9:
            break
    return 0.5 * (lo + hi)
