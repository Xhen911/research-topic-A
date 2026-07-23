"""
Transition basis — electron-hole pair enumeration and band-pair slicing.

Provides the canonical band-pair slice helper ``make_bs_cache`` for the
central ``nb_cache`` bands around charge neutrality.  This is the SINGLE
source of truth for the ``half - nb_cache//2 : half + nb_cache//2`` slice;
``CachedModel`` (construction and ``.load``) and ``convergence._chi0_single_q``
both delegate to it, so the band-slicing convention lives in exactly one place.

The broader transition-basis enumeration (eig_at_q extraction) remains a
dedicated-PR item; only the slice helper has been consolidated here so far.
"""

import numpy as np


def make_bs_cache(n_bands, nb_cache):
    """Build the band-pair slice for the central nb_cache bands.

    Parameters
    ----------
    n_bands : int
        Total number of bands in the Hamiltonian.
    nb_cache : int
        Number of bands around CNP to retain (must be even).

    Returns
    -------
    slice
        Python slice object selecting the central nb_cache bands.
    """
    half = n_bands // 2
    return slice(half - nb_cache // 2, half + nb_cache // 2)
