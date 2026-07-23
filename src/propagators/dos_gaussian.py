"""
dos.py
======
k 积分模块：态密度 (Density of States) & 通用 BZ 积分。

核心函数
--------
- compute_dos(model, ...)          → (E_values, dos_values)
  主要入口：BZ 均匀采样 + 高斯展宽计算 DOS。
- compute_eigenvalues(model, k_cart) → E_k, V_k
  批量对角化（优先使用模型的 _compute_bands_batch）。
- integrate_bz(model, integrand, ...) → float
  通用 BZ 积分框架。

约定
----
- k 积分 d²k 采用均匀网格 Riemann 和：
    ∫_BZ d²k/(2π)² f(k) ≈ |det(R)|/(2π)² · (1/Nk) Σ_i f(k_i)
  其中 R = model.reciprocal_vectors。
- g = valley_degeneracy × spin_degeneracy，通过 model.degeneracy_factor() 获取。
- 能量单位：eV（由 model.hamiltonian 决定）。
- 展宽 σ：高斯标准差 (eV)。

验证
----
对于 Dirac 系统，低能 DOS 应满足：
    DOS(|E|) ≈ g·|E| / (2π ħ² vF²)
其中 vF = √3/2 · t，g = 4（自旋 × 谷）。
"""

import numpy as np
from typing import Callable, Optional, Tuple

from .lindhard import generate_k_mesh, fermi_dirac



# ============================================================
#  高斯展宽核
# ============================================================

def _gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    """归一化高斯：∫_{-∞}^{∞} dx g(x) = 1."""
    return np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))



from .dos import compute_eigenvalues

# ============================================================
#  Gaussian-broadened DOS / JDOS / O-JDOS
# ============================================================

def compute_dos(
    model,
    nk: int = 200,
    E_range: Optional[Tuple[float, float]] = None,
    nE: int = 500,
    sigma: float = 0.01,
    k_cart: Optional[np.ndarray] = None,
    E_k: Optional[np.ndarray] = None,
    k_area: Optional[float] = None,
    vectorize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Density of states DOS(E) via Gaussian broadening.

    DOS(E) = g/(2pi)^2 integral_BZ d^2k sum_n delta(E - E_n(k))
           ~ g|det(R)|/((2pi)^2 Nk) sum_{k_i,n} delta_sigma(E - E_n(k_i))

    where delta_sigma(x) = exp(-x^2/2sigma^2) / (sigma sqrt(2pi)).

    Parameters
    ----------
    model : HamiltonianModel
    nk : int
        k-points per reciprocal direction (total = nk^2).
    E_range : (E_min, E_max) or None
        Energy range (eV). Auto-detected if None.
    nE : int
        Number of energy grid points.
    sigma : float
        Gaussian broadening sigma (eV).
    k_cart : (Nk, 2) or None
        Pre-generated k-mesh in Cartesian coordinates.
    E_k : (Nk, n_bands) or None
        Pre-computed eigenvalues.
    k_area : float or None
        Total k-space area (1/A^2). Default: |det(reciprocal_vectors)|.
    vectorize : bool
        Use broadcasting (faster, more memory) instead of nested loops.

    Returns
    -------
    E_values : (nE,)  — energy grid (eV)
    dos_values : (nE,) — DOS(E) (states/eV/unit cell)
    """
    if k_cart is None:
        _, k_cart = generate_k_mesh(nk, model.reciprocal_vectors)
    Nk = len(k_cart)

    if E_k is None:
        E_k, _ = compute_eigenvalues(model, k_cart)

    g = model.degeneracy_factor()
    if k_area is None:
        k_area = abs(np.linalg.det(model.reciprocal_vectors))
    prefactor = g * k_area / ((2 * np.pi) ** 2 * Nk)

    if E_range is None:
        E_min = min(np.min(E_k) - 3 * sigma, -1.0)
        E_max = max(np.max(E_k) + 3 * sigma, 1.0)
    else:
        E_min, E_max = E_range

    E_values = np.linspace(E_min, E_max, nE)

    if vectorize:
        # Broadcasting: (nE, 1, 1) - (1, Nk, nb) -> (nE, Nk, nb)
        diff = E_values[:, np.newaxis, np.newaxis] - E_k[np.newaxis, :, :]
        dos_values = prefactor * np.sum(_gaussian(diff, sigma), axis=(1, 2))
    else:
        dos_values = np.zeros(nE)
        for n in range(model.n_bands):
            for i in range(Nk):
                dos_values += _gaussian(E_values - E_k[i, n], sigma)
        dos_values *= prefactor

    return E_values, dos_values


# Backward-compatible aliases
def compute_dos_vectorized(*args, **kwargs):
    """Deprecated: use compute_dos(..., vectorize=True) instead."""
    import warnings
    warnings.warn(
        "compute_dos_vectorized is deprecated; use compute_dos(vectorize=True).",
        DeprecationWarning, stacklevel=2,
    )
    kwargs['vectorize'] = True
    return compute_dos(*args, **kwargs)



# ============================================================
#  Joint DOS — 带间跃迁 JDOS(ω) 和 光学 JDOS(ω)
# ============================================================

def compute_jdos_q0(
    model,
    nk: int = 200,
    w_range: Optional[Tuple[float, float]] = None,
    nw: int = 500,
    sigma: float = 0.01,
    k_cart: Optional[np.ndarray] = None,
    E_k: Optional[np.ndarray] = None,
    interband_only: bool = True,
    k_area: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Joint Density of States at q=0 (optical transitions at Gamma).

    JDOS(omega) = g/(2pi)^2 integral_BZ d^2k sum_{m>n}
                  delta(omega - (E_m(k) - E_n(k)))

    NOTE: This is q=0 only — energy differences are taken at the SAME
    k-point.  For finite-q JDOS use jdos_triangle.compute_jdos_q_triangle.

    Sums m>n only to avoid (m,n) <-> (n,m) double counting.
    Broadening: Gaussian delta_sigma(x) = exp(-x^2/2sigma^2)/(sigma sqrt(2pi)).

    Parameters
    ----------
    model : HamiltonianModel
    nk : int
    w_range : (w_min, w_max) or None
    nw : int
    sigma : float
    k_cart, E_k : optional
    interband_only : bool
        True: m>n only (interband JDOS). False: all m>=n (includes intraband).
    k_area : float or None
        k-space sampling area. Default: |det(reciprocal_vectors)|.

    Returns
    -------
    w_values : (nw,)
    jdos_values : (nw,)
    """
    if k_cart is None:
        _, k_cart = generate_k_mesh(nk, model.reciprocal_vectors)
    Nk = len(k_cart)

    if E_k is None:
        E_k, _ = compute_eigenvalues(model, k_cart)

    g = model.degeneracy_factor()
    if k_area is None:
        k_area = abs(np.linalg.det(model.reciprocal_vectors))
    prefactor = g * k_area / ((2 * np.pi) ** 2 * Nk)

    nb = model.n_bands
    band_width = np.max(E_k) - np.min(E_k)

    if w_range is None:
        w_min = 0.0
        w_max = band_width
    else:
        w_min, w_max = w_range

    w_values = np.linspace(w_min, w_max, nw)

    jdos = np.zeros(nw)
    for m in range(nb):
        for n in range(m if interband_only else m + 1):
            for i in range(Nk):
                dE = E_k[i, m] - E_k[i, n]
                jdos += _gaussian(w_values - dE, sigma)
    jdos *= prefactor

    return w_values, jdos


# Backward-compatible alias
def compute_jdos(*args, **kwargs):
    """Deprecated: use compute_jdos_q0 instead."""
    import warnings
    warnings.warn(
        "compute_jdos is deprecated; use compute_jdos_q0. "
        "This alias will be removed in a future version.",
        DeprecationWarning, stacklevel=2,
    )
    return compute_jdos_q0(*args, **kwargs)


def compute_optical_jdos(
    model,
    nk: int = 200,
    w_range: Optional[Tuple[float, float]] = None,
    nw: int = 500,
    sigma: float = 0.01,
    k_cart: Optional[np.ndarray] = None,
    E_k: Optional[np.ndarray] = None,
    V_k: Optional[np.ndarray] = None,
    direction: Optional[str] = None,
    k_area: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算光学 Joint DOS —— 用速度矩阵元 |v_{mn}|² 加权的 JDOS。

    O-JDOS(ω) = g/(2π)² ∫ d²k Σ_{m≠n} |v_{mn}(k)|² δ(ω − (E_m−E_n))

    其中 v^α_{mn} = ⟨m,k| v_α |n,k⟩，即速度算符在本征基下的矩阵元。

    Parameters
    ----------
    model : HamiltonianModel
    nk : int
    w_range : (w_min, w_max) or None
    nw : int
    sigma : float
    k_cart, E_k, V_k : optional
    direction : str or None
        'x': 只使用 |v^x_{mn}|² 权重。
        'y': 只使用 |v^y_{mn}|² 权重。
        None: 用 (|v^x|² + |v^y|²)/2（平均）。
    k_area : float or None

    Returns
    -------
    w_values : np.ndarray, shape (nw,)
    ojdos_values : np.ndarray, shape (nw,)
    """
    if k_cart is None:
        _, k_cart = generate_k_mesh(nk, model.reciprocal_vectors)
    Nk = len(k_cart)

    if E_k is None or V_k is None:
        E_k, V_k = compute_eigenvalues(model, k_cart)

    g = model.degeneracy_factor()
    if k_area is None:
        k_area = abs(np.linalg.det(model.reciprocal_vectors))
    prefactor = g * k_area / ((2 * np.pi) ** 2 * Nk)

    nb = model.n_bands
    band_width = np.max(E_k) - np.min(E_k)

    if w_range is None:
        w_min = 0.0
        w_max = band_width
    else:
        w_min, w_max = w_range

    w_values = np.linspace(w_min, w_max, nw)

    # 速度矩阵元 |v_{mn}|²
    ojdos = np.zeros(nw)
    for i in range(Nk):
        vx, vy = model.velocity_operator(k_cart[i])
        Vi = V_k[i]  # (n_orb, n_bands)

        # 本征基: v^α_{mn} = V† v_α V
        vx_eig = Vi.conj().T @ vx @ Vi   # (nb, nb)
        vy_eig = Vi.conj().T @ vy @ Vi

        for m in range(nb):
            for n in range(m):           # m>n only — 避免对称双重计数
                dE = E_k[i, m] - E_k[i, n]  # 保证 dE ≥ 0

                if direction == 'x':
                    weight = np.abs(vx_eig[m, n]) ** 2
                elif direction == 'y':
                    weight = np.abs(vy_eig[m, n]) ** 2
                else:
                    weight = (np.abs(vx_eig[m, n]) ** 2 +
                              np.abs(vy_eig[m, n]) ** 2) / 2.0

                ojdos += weight * _gaussian(w_values - dE, sigma)
    ojdos *= prefactor

    return w_values, ojdos


# ============================================================
