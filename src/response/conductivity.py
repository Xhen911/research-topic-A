"""
conductivity.py
===============
Kubo-formula optical conductivity for any HamiltonianModel.

Supports interband, intraband (Drude), and total conductivity tensors.
Works with both KP and TB models via finite-difference velocity matrix
elements — no reliance on TB-specific data structures.

Conventions
-----------
* ħ = 1 (energies in eV, velocities in eV·Å, conductivity in e²/ħ)
* SIGMA_0 = e²/ħ = 1.0  —  multiply externally for physical units:
    σ_physical (S) = σ × e²/ħ  ≈  σ × 3.874×10⁻⁵ S  (for 2D)
* ω, η in eV  |  k in 1/Å  |  T in eV (k_B = 1)
* Velocity operator  v_α = ∂H/∂k_α  via central differences

Physics
-------
Interband (ω > 0):

    σ_inter^{αβ}(ω) = (i·g/A) Σ_k Σ_{m≠n}
        (f_nk − f_mk) / (E_mk − E_nk)
        × v^α_nm(k) v^β_mn(k) / (ω + iη − (E_mk − E_nk))

Intraband (Drude):

    σ_intra^{αβ}(ω) = (i·g/A) Σ_k Σ_n
        (−∂f/∂E)(E_nk) · v^α_nn(k) v^β_nn(k) / (ω + iη)

The k-sum is a Riemann sum over a uniform BZ mesh (see generate_k_mesh).
Following the polarization.py convention, the discrete prefactor is

    i·g·|det(G)| / ((2π)²·Nk)

so that  (1/Nk) Σ_k → (1/|det(G)|) ∫_BZ d²k  recovers the continuum integral
∫ d²k/(2π)².

References
----------
* R. Kubo, J. Phys. Soc. Jpn. 12, 570 (1957)
* Compatible with src.models.base.HamiltonianModel interface

Author: 2026-06-01
"""

import numpy as np
from typing import Dict, Optional

try:
    from ..models.base import HamiltonianModel
except ImportError:
    from models.base import HamiltonianModel

try:
    from .polarization import generate_k_mesh, fermi_dirac
except ImportError:
    from polarization import generate_k_mesh, fermi_dirac

try:
    from .dos import compute_eigenvalues
except ImportError:
    from dos import compute_eigenvalues

# ────────────────────────────────────────────────────────────
#  Constants
# ────────────────────────────────────────────────────────────

SIGMA_0: float = 1.0
"""Conductivity quantum  e²/ħ  in natural units.

Multiply externally to convert to physical units:
    σ_S = σ × e²/ħ  ≈  σ × 3.874×10⁻⁵  S  (2D sheet)
"""

_DK_DEFAULT: float = 1e-4
"""Default finite-difference step for the velocity operator (1/Å)."""


# ────────────────────────────────────────────────────────────
#  Fermi-Dirac derivative  −∂f/∂E
# ────────────────────────────────────────────────────────────

def _fermi_derivative(E: np.ndarray, Ef: float, beta: float) -> np.ndarray:
    """−∂f/∂E = β·f(E)·(1 − f(E)).

    Positive-definite, peaked at E=Ef with width ~1/β.
    At very low T (beta > 1e4) falls back to a narrow Gaussian to avoid
    underflow in the product f(1−f).
    """
    if beta > 1e4:
        sigma_T = 1.0 / beta
        return np.exp(-((E - Ef) ** 2) / (2.0 * sigma_T ** 2)) / (
            np.sqrt(2.0 * np.pi) * sigma_T
        )
    f = fermi_dirac(E, Ef, beta)
    return beta * f * (1.0 - f)


# ────────────────────────────────────────────────────────────
#  Velocity matrix elements  v^α_{mn}(k)
# ────────────────────────────────────────────────────────────

def velocity_matrix_elements(
    model: HamiltonianModel,
    k_points: np.ndarray,
    dk: float = _DK_DEFAULT,
) -> np.ndarray:
    """速度矩阵元 v^α_{mn}(k)（本征基），使用模型的解析速度算符。

    对每个k点：
        1. 对角化 model.solve(k) → V (本征态)
        2. model.velocity_operator(k) → vx, vy (轨道基)
        3. 旋转变换：v^α_{mn} = V† v_α V

    Parameters
    ----------
    model : HamiltonianModel
    k_points : np.ndarray, shape (nk, 2)
    dk : float
        保留用于向后兼容，实际不使用（解析速度算符）。

    Returns
    -------
    v_matrix : np.ndarray, shape (nk, nb, nb, 2), dtype=complex128
    """
    nk = len(k_points)
    nb = model.n_bands

    v_matrix = np.zeros((nk, nb, nb, 2), dtype=np.complex128)

    for ik in range(nk):
        k = k_points[ik]
        _, V = model.solve(k)

        vx_orb, vy_orb = model.velocity_operator(k)

        v_matrix[ik, :, :, 0] = V.conj().T @ vx_orb @ V
        v_matrix[ik, :, :, 1] = V.conj().T @ vy_orb @ V

    return v_matrix


# ────────────────────────────────────────────────────────────
#  Interband conductivity kernel
# ────────────────────────────────────────────────────────────

def optical_conductivity_interband(
    model: HamiltonianModel,
    omega: np.ndarray,
    v_matrix: np.ndarray,
    E_k: np.ndarray,
    Ef: float = 0.0,
    beta: float = 100.0,
    eta: float = 0.003,
) -> np.ndarray:
    """Interband optical conductivity from the Kubo formula.

        σ_inter^{αβ}(ω) = (i·g/A) Σ_k Σ_{m≠n}
            (f_nk − f_mk) / (E_mk − E_nk)
            × v^α_nm(k) v^β_mn(k) / (ω + iη − (E_mk − E_nk))

    Parameters
    ----------
    model : HamiltonianModel
        Used for ``reciprocal_vectors`` and ``degeneracy_factor()``.
    omega : np.ndarray, shape (nω,)
        Frequency grid (eV).
    v_matrix : np.ndarray, shape (nk, nb, nb, 2), dtype=complex128
        Velocity matrix elements from :func:`velocity_matrix_elements`.
    E_k : np.ndarray, shape (nk, nb)
        Band energies (eV).  Assumed sorted ascending (band 0 = lowest).
    Ef : float
        Fermi energy (eV).
    beta : float
        Inverse temperature  1/(k_B T)  in eV⁻¹.
    eta : float
        Broadening η (eV).

    Returns
    -------
    sigma_inter : np.ndarray, shape (nω, 2, 2), dtype=complex128
        Interband conductivity tensor in units of e²/ħ.
    """
    nw = len(omega)
    nk, nb = E_k.shape
    area = abs(np.linalg.det(model.reciprocal_vectors))

    # prefactor: i·|det(G)| / ((2π)²·Nk)
    # (no extra degeneracy factor — BZ integration + band sum account for
    #  spin/valley implicitly; the universal conductivity g/16 emerges from
    #  the Kubo formula without an explicit g prefactor)
    prefactor = 1j * area / ((2.0 * np.pi) ** 2 * nk)

    sigma = np.zeros((nw, 2, 2), dtype=np.complex128)

    # Fermi occupations  f(E_nk)
    f = fermi_dirac(E_k, Ef, beta)

    for ik in range(nk):
        dE = E_k[ik, :, np.newaxis] - E_k[ik, np.newaxis, :]    # dE[m,n]
        df = f[ik, np.newaxis, :] - f[ik, :, np.newaxis]        # df[n,m]

        # m>n only (absorption, omega>0)
        mask = np.zeros((nb, nb), dtype=bool)
        for m in range(nb):
            for n in range(m):
                mask[m, n] = True

        dE[~mask] = np.inf
        df[~mask] = 0.0

        # ratio[n,m] = (f_n - f_m) / (E_m - E_n)
        ratio = np.where(np.abs(dE) > 1e-12, df / dE, 0.0)

        v_k = v_matrix[ik]                                       # (nb, nb, 2)
        v_prod = np.einsum('nma,nmb->abnm', v_k, v_k.conj())    # (2, 2, nb, nb)

        for iw, w in enumerate(omega):
            denom = w + 1j * eta - dE
            contrib = np.sum(ratio * v_prod / denom)             # (2, 2)
            sigma[iw] += prefactor * contrib

    return sigma


# ────────────────────────────────────────────────────────────
#  Intraband (Drude) conductivity kernel
# ────────────────────────────────────────────────────────────

def optical_conductivity_intraband(
    model: HamiltonianModel,
    omega: np.ndarray,
    v_matrix: np.ndarray,
    E_k: np.ndarray,
    Ef: float = 0.0,
    beta: float = 100.0,
    eta: float = 0.003,
) -> np.ndarray:
    """Intraband (Drude) optical conductivity.

        σ_intra^{αβ}(ω) = (i·g/A) Σ_k Σ_n
            (−∂f/∂E)(E_nk) · v^α_nn(k) v^β_nn(k) / (ω + iη)

    Parameters
    ----------
    model : HamiltonianModel
        Used for ``reciprocal_vectors`` and ``degeneracy_factor()``.
    omega : np.ndarray, shape (nω,)
        Frequency grid (eV).
    v_matrix : np.ndarray, shape (nk, nb, nb, 2), dtype=complex128
        Velocity matrix elements.
    E_k : np.ndarray, shape (nk, nb)
        Band energies (eV).
    Ef : float
        Fermi energy (eV).
    beta : float
        Inverse temperature (eV⁻¹).
    eta : float
        Broadening (eV).

    Returns
    -------
    sigma_intra : np.ndarray, shape (nω, 2, 2), dtype=complex128
        Intraband conductivity tensor in units of e²/ħ.
    """
    nw = len(omega)
    nk, nb = E_k.shape
    area = abs(np.linalg.det(model.reciprocal_vectors))

    prefactor = 1j * area / ((2.0 * np.pi) ** 2 * nk)

    sigma = np.zeros((nw, 2, 2), dtype=np.complex128)

    # −∂f/∂E  —  positive, weight for the Drude term
    dfde = _fermi_derivative(E_k, Ef, beta)                      # (nk, nb)

    for ik in range(nk):
        v_k = v_matrix[ik]                                       # (nb, nb, 2)
        # diagonal velocity elements (real, since v is Hermitian)
        v_diag = np.real(np.diagonal(v_k, axis1=0, axis2=1)).T   # (nb, 2)

        # Drude weight integrand per band:
        # W^{αβ}[n] = (−∂f/∂E)_n · v^α_nn · v^β_nn
        # sum over bands → Σ_n, then outer α,β
        W = np.einsum('n,na,nb->ab', dfde[ik], v_diag, v_diag)  # (2, 2)

        for iw, w in enumerate(omega):
            sigma[iw] += prefactor * W / (w + 1j * eta)

    return sigma


# ────────────────────────────────────────────────────────────
#  Main entry point — full conductivity
# ────────────────────────────────────────────────────────────

def optical_conductivity(
    model: HamiltonianModel,
    omega: np.ndarray,
    nk: int = 60,
    Ef: float = 0.0,
    T: float = 0.0,
    eta: float = 0.003,
    dk: float = _DK_DEFAULT,
) -> Dict[str, np.ndarray]:
    """Full optical conductivity tensor from the Kubo formula.

    Computes interband, intraband (Drude), and total conductivity
    on a uniform BZ mesh.  Works with **any** :class:`HamiltonianModel`
    (KP or TB).

    Parameters
    ----------
    model : HamiltonianModel
        Model providing ``hamiltonian(k)``, ``solve(k)``,
        ``reciprocal_vectors``, and ``degeneracy_factor()``.
    omega : np.ndarray, shape (nω,)
        Frequency grid (eV).
    nk : int
        k-points per reciprocal-lattice direction (total mesh = nk²).
    Ef : float
        Fermi energy (eV).  Default 0 (charge neutrality).
    T : float
        Temperature in eV (k_B·T).  T ≤ 0 uses a small effective
        temperature (1e-4 eV) for numerical stability.
    eta : float
        Broadening η (eV).  Controls peak / shoulder widths.
    dk : float
        Finite-difference step for the velocity operator (1/Å).
        Default 1e-4.

    Returns
    -------
    result : dict
        ``'inter'``  — ``(nω, 2, 2)`` complex128 interband tensor
        ``'intra'``  — ``(nω, 2, 2)`` complex128 intraband tensor
        ``'total'``  — ``(nω, 2, 2)`` complex128 total = inter + intra
        ``'omega'``  — ``(nω,)`` frequency grid
        ``'Ef'``     — Fermi energy used
        ``'T'``      — effective temperature used

    Notes
    -----
    * Conductivity is returned in units of e²/ħ (SIGMA_0 = 1).
      Multiply by e²/ħ ≈ 3.874×10⁻⁵ S to get physical 2D sheet
      conductivity in siemens, or by e²/h ≈ 2.434×10⁻⁴ Ω⁻¹ for
      conductance quantum units.
    * The intraband Drude term is purely real in the DC limit
      and falls off as ∼1/ω² for ω ≫ η.
    * For models with many bands (nb > 30), velocity matrix
      elements are computed once and stored.  Memory ~ 16·nk·nb² B.

    Examples
    --------
    >>> from src.models.graphene import SingleLayerGrapheneKP
    >>> from src.response.conductivity import optical_conductivity
    >>> model = SingleLayerGrapheneKP()
    >>> omega = np.linspace(0.01, 0.3, 200)
    >>> sigma = optical_conductivity(model, omega, nk=60)
    >>> Re_xx = sigma['total'][:, 0, 0].real  # real part, xx
    """
    # ── temperature ──
    T_eff = max(T, 1e-4)
    beta = 1.0 / T_eff

    # ── k-mesh ──
    _, k_cart = generate_k_mesh(nk, model.reciprocal_vectors)
    nk_tot = len(k_cart)

    # ── batch diagonalization ──
    E_k, _ = compute_eigenvalues(model, k_cart)

    # ── velocity matrix elements (analytical) ──
    v_matrix = velocity_matrix_elements(model, k_cart)

    # ── interband ──
    sigma_inter = optical_conductivity_interband(
        model, omega, v_matrix, E_k, Ef=Ef, beta=beta, eta=eta,
    )

    # ── intraband ──
    sigma_intra = optical_conductivity_intraband(
        model, omega, v_matrix, E_k, Ef=Ef, beta=beta, eta=eta,
    )

    sigma_total = sigma_inter + sigma_intra

    return {
        'inter': sigma_inter,
        'intra': sigma_intra,
        'total': sigma_total,
        'omega': omega,
        'Ef': Ef,
        'T': T_eff,
    }


# ────────────────────────────────────────────────────────────
#  Convenience: xx (longitudinal) diagonal
# ────────────────────────────────────────────────────────────

def optical_conductivity_xx(
    model: HamiltonianModel,
    omega: np.ndarray,
    nk: int = 60,
    Ef: float = 0.0,
    T: float = 0.0,
    eta: float = 0.003,
    dk: float = _DK_DEFAULT,
) -> Dict[str, np.ndarray]:
    """Convenience wrapper returning only the xx (longitudinal) component.

    Returns
    -------
    dict with keys ``'inter'``, ``'intra'``, ``'total'``, ``'omega'``.
    Each value except ``'omega'`` is a 1D ``(nω,)`` real array.
    """
    full = optical_conductivity(
        model, omega, nk=nk, Ef=Ef, T=T, eta=eta, dk=dk,
    )
    return {
        'inter': full['inter'][:, 0, 0].real,
        'intra': full['intra'][:, 0, 0].real,
        'total': full['total'][:, 0, 0].real,
        'omega': full['omega'],
    }


# ────────────────────────────────────────────────────────────
#  Public API
# ────────────────────────────────────────────────────────────

__all__ = [
    'SIGMA_0',
    'velocity_matrix_elements',
    'optical_conductivity_interband',
    'optical_conductivity_intraband',
    'optical_conductivity',
    'optical_conductivity_xx',
]
