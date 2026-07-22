"""
Transition basis — electron-hole pair enumeration and band-pair slicing.

This module will eventually hold the band-pair enumeration logic extracted
from CachedModel (bs_cache construction, eig_at_q).  For now it provides
a standalone helper so that transition-basis construction is not scattered
across multiple call sites.

During the PR-3 refactor the CachedModel class was moved wholesale to
core/cache.py without splitting its internals.  The extraction of
eig_at_q and bs_cache into this module is deferred to a dedicated PR
to keep PR-3 as a pure move (no logic changes).
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
