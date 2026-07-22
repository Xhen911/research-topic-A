"""
Symmetry utilities and gauge fixing.

- high_symmetry_points and check_hermitian remain as instance methods
  on the concrete model classes (graphene.py, tbg_bm.py).  They are NOT
  moved here because they depend on model-specific lattice conventions.
- fix_gauge: eigenstate gauge fixing for smooth q -> 0 behaviour of
  density vertices.  Currently a stub (identity); will be implemented
  when needed for vertex continuity.
"""

import numpy as np


def fix_gauge(eigvecs):
    """Fix the gauge of a set of eigenvectors for q-continuity.

    Applies a continuity (parallel-transport) gauge so that eigenvectors
    vary smoothly as k (or q) changes.  This prevents spurious sign flips
    and phase jumps in the density vertex M = |<m,k+q|n,k>|^2.

    Parameters
    ----------
    eigvecs : np.ndarray, shape (n_orbitals, n_bands) or (Nk, n_orbitals, n_bands)
        Eigenvector matrix (columns are eigenstates).

    Returns
    -------
    np.ndarray, same shape
        Gauge-fixed eigenvectors.

    Notes
    -----
    Currently returns the input unchanged (identity gauge).  The proper
    implementation will maximise <psi_k | psi_{k+dk}> overlap and apply
    the optimal phase factor per band.  Deferred to a dedicated PR.
    """
    return eigvecs
