"""
Density vertices and form factors.

The density vertex (form factor) M[m,n] = |<m, k+q | n, k>|^2 is the
overlap squared between eigenstates at k+q and k.  It enters the Lindhard
polarization as a band-pair weight.

Historically this computation lived inline in response/polarization.py.
It is promoted here to a first-class L1 quantity because it depends only
on the eigenstates (band structure), not on any propagator or interaction.
"""

import numpy as np


def density_form_factor(V_k, V_q):
    """Compute the density-vertex form factor M = |<m,k+q|n,k>|^2.

    Parameters
    ----------
    V_k : np.ndarray, shape (Nk, n_orbitals, nb)
        Eigenstates at k.
    V_q : np.ndarray, shape (Nk, n_orbitals, nb)
        Eigenstates at k+q.

    Returns
    -------
    M : np.ndarray, shape (Nk, nb, nb)
        M[k, m, n] = |<m, k+q | n, k>|^2, the squared overlap between
        band m at k+q and band n at k.
    """
    return np.abs(np.einsum('kbm,kbn->kmn', V_q.conj(), V_k)) ** 2
