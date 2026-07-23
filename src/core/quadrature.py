"""
quadrature.py — generic Brillouin-zone integration utilities.
===============================================================

Moved from src/propagators/dos.py during dos.py slimming (2026-07-23).
"""
import numpy as np
from typing import Callable, Optional


def integrate_bz(
    model,
    integrand: Callable,
    nk: int = 200,
    k_cart: Optional[np.ndarray] = None,
    E_k: Optional[np.ndarray] = None,
    V_k: Optional[np.ndarray] = None,
) -> float:
    """Generic BZ integral with uniform k-mesh.

    ∫_BZ d²k/(2π)² f(k) ≈ |det(R)|/((2π)² Nk) Σ_i f(k_i)

    Parameters
    ----------
    model : HamiltonianModel
    integrand : callable
        Signature: f(k_i, E_k[i], V_k[i]) → float.
        Receives a single k-point's coordinates, eigenvalues, eigenvectors.
    nk : int
        k-points per reciprocal direction (total = nk²).
    k_cart : np.ndarray, shape (Nk, 2) or None
    E_k : np.ndarray, shape (Nk, n_bands) or None
    V_k : np.ndarray, shape (Nk, n_orbitals, n_bands) or None

    Returns
    -------
    result : float
    """
    from ..propagators.lindhard import generate_k_mesh
    from ..propagators.dos import compute_eigenvalues

    if k_cart is None:
        _, k_cart = generate_k_mesh(nk, model.reciprocal_vectors)
    Nk = len(k_cart)

    if E_k is None or V_k is None:
        E_k, V_k = compute_eigenvalues(model, k_cart)

    vol_BZ = abs(np.linalg.det(model.reciprocal_vectors))
    prefactor = vol_BZ / ((2 * np.pi) ** 2 * Nk)

    total = 0.0
    for i in range(Nk):
        total += integrand(k_cart[i], E_k[i], V_k[i])

    return prefactor * total
