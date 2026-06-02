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

from .polarization import generate_k_mesh, fermi_dirac


# ============================================================
#  高斯展宽核
# ============================================================

def _gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    """归一化高斯：∫_{-∞}^{∞} dx g(x) = 1."""
    return np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


# ============================================================
#  批量对角化
# ============================================================

def compute_eigenvalues(model, k_cart: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """对一组 k 点批量对角化，返回 (E_k, V_k)。

    优先使用模型的 _compute_bands_batch()（更快），
    否则回退到逐点 solve()。
    """
    Nk = len(k_cart)
    nb = model.n_bands
    no = model.n_orbitals

    if hasattr(model, '_compute_bands_batch'):
        return model._compute_bands_batch(k_cart)

    E = np.zeros((Nk, nb))
    V = np.zeros((Nk, no, nb), dtype=complex)
    for i in range(Nk):
        Ei, Vi = model.solve(k_cart[i])
        E[i] = Ei
        V[i] = Vi
    return E, V


# ============================================================
#  态密度 DOS(E)
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
) -> Tuple[np.ndarray, np.ndarray]:
    """计算态密度 DOS(E)。

    DOS(E) = g/(2π)² ∫_BZ d²k Σ_n δ(E − E_n(k))
           ≈ g·|det(R)|/((2π)² Nk) Σ_{k_i,n} δ_σ(E − E_n(k_i))

    其中 δ_σ 为高斯展宽：δ_σ(x) = exp(−x²/2σ²) / (σ√(2π)).

    Parameters
    ----------
    model : HamiltonianModel
        模型实例。
    nk : int
        每条倒格矢方向的 k 网格点数。总数 = nk²。
    E_range : (E_min, E_max) or None
        能量范围 (eV)。若为 None，自动从能带极值推定。
    nE : int
        能量网格点数。
    sigma : float
        高斯展宽 σ (eV)。
    k_cart : np.ndarray, shape (Nk, 2) or None
        预生成的 k 网格（直角坐标）。若为 None，自动生成。
    E_k : np.ndarray, shape (Nk, n_bands) or None
        预计算的本征值。
    k_area : float or None
        覆盖的 k 空间总面积 (1/Å²)。
        默认为 |det(reciprocal_vectors)|（完整 BZ）。
        对 Dirac 中心 KP 模型应设为网格实际面积，
        此时 degeneracy_factor 应包含谷简并。

    Returns
    -------
    E_values : np.ndarray, shape (nE,)
        能量网格点 (eV)。
    dos_values : np.ndarray, shape (nE,)
        DOS(E) (states/eV/unit cell)。
    """
    # ── k 网格 ────────────────────────────────────────────
    if k_cart is None:
        _, k_cart = generate_k_mesh(nk, model.reciprocal_vectors)
    Nk = len(k_cart)

    # ── 对角化 ────────────────────────────────────────────
    if E_k is None:
        E_k, _ = compute_eigenvalues(model, k_cart)

    # ── 积分预因子 ────────────────────────────────────────
    g = model.degeneracy_factor()
    if k_area is None:
        k_area = abs(np.linalg.det(model.reciprocal_vectors))
    prefactor = g * k_area / ((2 * np.pi) ** 2 * Nk)

    # ── 能量范围 ──────────────────────────────────────────
    if E_range is None:
        E_min = np.min(E_k) - 3 * sigma
        E_max = np.max(E_k) + 3 * sigma
        # 对 Dirac 系统自动扩展以覆盖低能区域
        E_min = min(E_min, -1.0)
        E_max = max(E_max, 1.0)
    else:
        E_min, E_max = E_range

    E_values = np.linspace(E_min, E_max, nE)

    # ── 高斯展宽累加 ──────────────────────────────────────
    # DOS(E) = prefactor · (1/Nk) Σ_{k_i,n} δ_σ(E − E_n(k_i))
    # prefactor 已含 1/Nk，此处直接乘
    dos_values = np.zeros(nE)
    for n in range(model.n_bands):
        for i in range(Nk):
            dos_values += _gaussian(E_values - E_k[i, n], sigma)
    dos_values *= prefactor

    return E_values, dos_values


# ============================================================
#  DOS 函数 — 矢量化优化版
# ============================================================

def compute_dos_vectorized(
    model,
    nk: int = 200,
    E_range: Optional[Tuple[float, float]] = None,
    nE: int = 500,
    sigma: float = 0.01,
    k_cart: Optional[np.ndarray] = None,
    E_k: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算态密度 — 矢量化版本（更大内存，更快速度）。

    与 compute_dos 相同接口，但使用广播代替循环。
    适合 nE × Nk × nb 不太大的情形（通常 < 10⁷）。
    """
    if k_cart is None:
        _, k_cart = generate_k_mesh(nk, model.reciprocal_vectors)
    Nk = len(k_cart)

    if E_k is None:
        E_k, _ = compute_eigenvalues(model, k_cart)

    g = model.degeneracy_factor()
    vol_BZ = abs(np.linalg.det(model.reciprocal_vectors))
    prefactor = g * vol_BZ / ((2 * np.pi) ** 2 * Nk)

    if E_range is None:
        E_min = min(np.min(E_k) - 3 * sigma, -1.0)
        E_max = max(np.max(E_k) + 3 * sigma, 1.0)
    else:
        E_min, E_max = E_range

    E_values = np.linspace(E_min, E_max, nE)

    # 广播: (nE, 1, 1) - (1, Nk, nb) → (nE, Nk, nb)
    diff = E_values[:, np.newaxis, np.newaxis] - E_k[np.newaxis, :, :]
    gaussians = _gaussian(diff, sigma)            # (nE, Nk, nb)
    dos_values = prefactor * np.sum(gaussians, axis=(1, 2))  # (nE,)

    return E_values, dos_values


# ============================================================
#  通用 BZ 积分
# ============================================================

def integrate_bz(
    model,
    integrand: Callable,
    nk: int = 200,
    k_cart: Optional[np.ndarray] = None,
    E_k: Optional[np.ndarray] = None,
    V_k: Optional[np.ndarray] = None,
) -> float:
    """通用 BZ 积分。

    ∫_BZ d²k/(2π)² f(k) ≈ |det(R)|/((2π)² Nk) Σ_i f(k_i)

    Parameters
    ----------
    model : HamiltonianModel
    integrand : callable
        签名: f(k_i, E_k[i], V_k[i]) → float
        接收单个 k 点的坐标、本征值、本征态，返回被积函数值。
    nk : int
    k_cart : np.ndarray, shape (Nk, 2) or None
    E_k : np.ndarray, shape (Nk, n_bands) or None
    V_k : np.ndarray, shape (Nk, n_orbitals, n_bands) or None

    Returns
    -------
    result : float
    """
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


# ============================================================
#  Dirac 点附近 k 网格（用于单谷 KP 模型）
# ============================================================

def generate_dirac_mesh(
    model,
    nk: int = 100,
    k_cut: float = 0.5,
    valley: str = 'K',
) -> np.ndarray:
    """生成 Dirac 点附近的均匀 k 网格（直角坐标）。

    用于 KP 模型等仅在 Dirac 点附近有效的模型。
    网格覆盖以 Dirac 点为中心、边长 2·k_cut 的正方形区域。

    Parameters
    ----------
    model : HamiltonianModel
        需有 K_point / Kp_point 属性（KP 模型）。
    nk : int
        每条边的采样点数。总数 = nk²。
    k_cut : float
        半边长 (1/Å)。Default 0.5。
    valley : str
        'K' 或 'K\''。

    Returns
    -------
    k_cart : np.ndarray, shape (nk², 2)
    """
    if valley == 'K':
        center = model.K_point
    else:
        center = model.Kp_point

    xs = np.linspace(-k_cut, k_cut, nk)
    ys = np.linspace(-k_cut, k_cut, nk)
    X, Y = np.meshgrid(xs, ys)
    k_cart = np.column_stack((X.ravel() + center[0], Y.ravel() + center[1]))
    return k_cart


def _dirac_mesh_area(model, k_cart: np.ndarray) -> float:
    """由均匀网格估算 k 空间采样面积。"""
    nk = int(np.sqrt(len(k_cart)))
    xs = np.unique(k_cart[:, 0])
    dx = xs[1] - xs[0] if len(xs) > 1 else 1.0
    ys = np.unique(k_cart[:, 1])
    dy = ys[1] - ys[0] if len(ys) > 1 else 1.0
    return nk * nk * dx * dy


# ============================================================
#  解析 Dirac DOS（用于验证）
# ============================================================

def dirac_dos_analytical(
    E_values: np.ndarray,
    vF: float,
    g: int = 4,
    vol_BZ: float = 1.0,
) -> np.ndarray:
    """2D Dirac 系统的解析态密度。

    DOS(E) = g·|E| / (2π ħ² vF²)

    注意：此公式假设 E ≪ 带宽，仅在 Dirac 锥有效范围内成立。

    Parameters
    ----------
    E_values : np.ndarray
        能量网格 (eV)。
    vF : float
        Fermi 速度 (eV·Å 或模型无量纲单位)。
    g : int
        简并因子 (自旋 × 谷)。Default 4。
    vol_BZ : float
        BZ 面积 |det(R)|。仅用于量纲一致性检查。

    Returns
    -------
    dos : np.ndarray
    """
    return g * np.abs(E_values) / (2 * np.pi * vF ** 2)
