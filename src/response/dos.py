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
#  Joint DOS — 带间跃迁 JDOS(ω) 和 光学 JDOS(ω)
# ============================================================

def compute_jdos(
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
    """计算 Joint Density of States。

    JDOS(ω) = g/(2π)² ∫_BZ d²k Σ_{m>n} δ(ω − (E_m − E_n))

    注意：只累加 m>n 以避免 (m,n) ↔ (n,m) 双重计数。

    展宽：高斯 δ_σ(x) = exp(−x²/2σ²) / (σ√(2π))。

    Parameters
    ----------
    model : HamiltonianModel
    nk : int
    w_range : (w_min, w_max) or None
    nw : int
    sigma : float
    k_cart, E_k : optional
    interband_only : bool
        True: 只 m>n（带间 JDOS）。False: 所有 m≥n。
    k_area : float or None
        k 空间采样总面积。默认 |det(reciprocal_vectors)|。

    Returns
    -------
    w_values : np.ndarray, shape (nw,)
    jdos_values : np.ndarray, shape (nw,)
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

    # 累加：m>n 保证 dE ≥ 0，避免对称双重计数
    jdos = np.zeros(nw)
    for m in range(nb):
        for n in range(m if interband_only else m + 1):
            for i in range(Nk):
                dE = E_k[i, m] - E_k[i, n]
                jdos += _gaussian(w_values - dE, sigma)
    jdos *= prefactor

    return w_values, jdos


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
#  解析 Dirac JDOS / O-JDOS（用于验证）
# ============================================================

def dirac_jdos_analytical(
    w_values: np.ndarray,
    vF: float,
    g: int = 4,
) -> np.ndarray:
    """2D Dirac 锥的解析带间 JDOS。

    JDOS(ω) = g·ω / (8π vF²)

    推导：ΔE = 2vF|k|, d(ΔE) = 2vF d|k|
    JDOS = g/(2π)² · 2π ∫ k dk · 2 · δ(ω − 2vF k)   [m↔n 对称计数 2]
         = g/(2π) · ∫ k dk δ(ω − 2vF k)
         = g/(2π) · (ω/(2vF)) · (1/(2vF))
         = g·ω / (8π vF²)

    Parameters
    ----------
    w_values : np.ndarray
    vF : float
    g : int
        简并因子 (自旋 × 谷)。Default 4。

    Returns
    -------
    jdos : np.ndarray
    """
    return g * w_values / (8 * np.pi * vF ** 2)


def dirac_optical_jdos_analytical(
    w_values: np.ndarray,
    vF: float,
    g: int = 4,
) -> np.ndarray:
    """2D Dirac 锥的解析光学 JDOS（各向同性平均）。

    O-JDOS(ω) = g·vF²·ω / (16π vF²) = g·ω / (16π)

    推导：|v_{+-}|² = vF²（各向同性，见光学矩阵元验证）
    O-JDOS = (1/2) Σ_α |v^α|² · JDOS_不带权重
            = vF² · JDOS（已考虑 1/2 平均）
    注意：compute_optical_jdos 的 (|vx|²+|vy|²)/2 也做了平均。
    """
    return g * w_values / (16 * np.pi)

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



# ============================================================
#  三角形 DOS (Lehmann–Taut) — 精确 η→0 公式
#  Ref: Lorentzian broadening convergence review, 2026-07-15
# ============================================================

def _triangles_for_kmesh(nk):
    """Triangle connectivity for a uniform nk×nk BZ parallelogram mesh."""
    tri_list = []
    for a in range(nk):
        for b in range(nk):
            v00 = a * nk + b
            v01 = a * nk + (b + 1) % nk
            v10 = (a + 1) % nk * nk + b
            v11 = (a + 1) % nk * nk + (b + 1) % nk
            tri_list.append([v00, v01, v11])
            tri_list.append([v00, v10, v11])
    return np.array(tri_list, dtype=int)


def _triangle_dos_exact(e1, e2, e3, E, area, tol=1e-12):
    """Exact η→0 DOS contribution from one triangle with linear dispersion.

    For sorted vertex energies e1 ≤ e2 ≤ e3, the DOS inside the triangle
    with uniform linear interpolation is exactly piecewise-linear:

        DOS(E) ∝ 2·(E−e1)/[(e2−e1)(e3−e1)]   for E ∈ [e1, e2]
        DOS(E) ∝ 2·(e3−E)/[(e3−e2)(e3−e1)]   for E ∈ [e2, e3]
        DOS(E) = 0                              otherwise

    The integral ∫ DOS(E) dE = 1 (per unit area; prefactor applied externally).

    The three-equal (e1≈e2≈e3) case gives a Dirac-delta — handled as
    Gaussian replacement with tiny width to keep the energy grid meaningful.

    Uses Taylor-blend for near-degenerate cases (|de| < 1e-6 eV by default)
    instead of hard thresholds, avoiding cliff-edge instability.
    """
    import warnings
    import numpy as np

    # Sort
    perm = np.argsort([e1, e2, e3])
    a, b, c = float(e1), float(e2), float(e3)
    a, b, c = (np.array([a, b, c])[perm]).tolist()
    de_ba = c - b              # e3 - e2 (after sort)
    de_ca = c - a              # e3 - e1
    de_cb = b - a              # e2 - e1

    dos = np.zeros_like(E)

    # --- Fully degenerate: replace with narrow Gaussian ---
    if de_ca < 1e-14:
        # All three equal → Dirac delta
        sigma = max(1e-14, area * 0.1)  # negligible but stable
        dos = area * np.exp(-0.5 * ((E - a) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        return dos

    # --- Taylor-blend threshold (padded window, no hard cliff) ---
    blend_width = max(1e-8, de_ca * 1e-6)

    # Helper: smooth 0→1 ramp over width w
    def _ramp(x, w):
        """Smooth ramp: 0 for x ≤ 0, 1 for x ≥ w, C¹ polynomial in between."""
        if w < 1e-15:
            return 1.0 if x > 0 else 0.0
        t = np.clip(x / w, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)   # smoothstep

    # Two-equal case: e1 = e2 < e3 (or e1 < e2 = e3)
    if abs(c - b) < blend_width:
        # e2 ≈ e3 — blend between the generic formula and the e2=e3 limit
        # e2=e3 limit: DOS = 2·A·(E-a)/(c-a)² for E∈[a,c], 0 otherwise
        # Generic:  DOS = 2·A·(E-a)/[(b-a)(c-a)] for E∈[a,b],
        #           DOS = 2·A·(c-E)/[(c-b)(c-a)] for E∈[b,c]
        r = _ramp(c - b, blend_width)
        # Limit formula (r=1 → fully degenerate e2=e3)
        mask_ac = (E >= a) & (E <= c)
        dos_limit = np.where(mask_ac, area * 2.0 * (E - a) / (de_ca ** 2), 0.0)
        # Generic formula
        de_cb_safe = max(de_cb, 1e-30)
        de_ba_safe = max(de_ba, 1e-30)
        mask_ab = (E >= a) & (E <= b)
        mask_bc = (E >= b) & (E <= c)
        dos_generic = np.where(mask_ab,
            area * 2.0 * (E - a) / (de_cb_safe * de_ca),
            np.where(mask_bc,
                area * 2.0 * (c - E) / (de_ba_safe * de_ca),
                0.0))
        dos = (1.0 - r) * dos_generic + r * dos_limit

    elif abs(b - a) < blend_width:
        # e1 ≈ e2
        r = _ramp(b - a, blend_width)
        mask_ac = (E >= a) & (E <= c)
        dos_limit = np.where(mask_ac, area * 2.0 * (c - E) / (de_ca ** 2), 0.0)
        de_cb_safe = max(de_cb, 1e-30)
        de_ba_safe = max(de_ba, 1e-30)
        mask_ab = (E >= a) & (E <= b)
        mask_bc = (E >= b) & (E <= c)
        dos_generic = np.where(mask_ab,
            area * 2.0 * (E - a) / (de_cb_safe * de_ca),
            np.where(mask_bc,
                area * 2.0 * (c - E) / (de_ba_safe * de_ca),
                0.0))
        dos = (1.0 - r) * dos_generic + r * dos_limit

    else:
        # Generic: all three distinct
        de_cb_safe = max(de_cb, 1e-30)
        de_ba_safe = max(de_ba, 1e-30)
        mask_ab = (E >= a) & (E <= b)
        mask_bc = (E >= b) & (E <= c)
        dos = np.where(mask_ab,
            area * 2.0 * (E - a) / (de_cb_safe * de_ca),
            np.where(mask_bc,
                area * 2.0 * (c - E) / (de_ba_safe * de_ca),
                0.0))

    return dos


def compute_dos_triangle(model, nk=24, E_range=None, nE=3000,
                         eta=0.05e-3, band_slice=None):
    """DOS via exact η→0 Lehmann-Taut triangle integration.

    Within each triangle the band energy is linearly interpolated from
    the three vertex values.  The DOS contribution is piecewise-linear
    and integrated analytically — no broadening, no branch-cut logarithms.
    This captures the logarithmic VHS divergence exactly (up to mesh
    discretisation, which converges O(1/Nk) in smooth regions).

    The ``eta`` parameter is kept for API compatibility but is **not used**
    in the core integration.  Physical broadening, if needed, should be
    applied as a separate post-processing Lorentzian convolution.

    Parameters
    ----------
    model : HamiltonianModel
    nk : int — k-points per reciprocal direction (total = nk²)
    E_range : (float, float) or None
    nE : int
    eta : float — *unused* in the exact formula; kept for compatibility.
    band_slice : slice or None — which bands to include (default: all)

    Returns
    -------
    E : (nE,)  energy grid (eV)
    dos : (nE,)  DOS (states/eV/unit cell)
    """
    _, k_cart = generate_k_mesh(nk, model.reciprocal_vectors)
    Nk = len(k_cart)
    assert int(np.sqrt(Nk)) ** 2 == Nk, f'nk² ≠ Nk={Nk}'
    nk_side = int(np.sqrt(Nk))

    E_k, _ = compute_eigenvalues(model, k_cart)
    if band_slice is None:
        band_slice = slice(None)
    E_k = E_k[:, band_slice]
    nb_sel = E_k.shape[1]

    if E_range is None:
        E_range = (float(E_k.min()), float(E_k.max()))
    E = np.linspace(*E_range, nE)

    area_BZ = abs(np.linalg.det(model.reciprocal_vectors))
    dk = np.sqrt(area_BZ / Nk)
    area_tri = dk ** 2 / 2.0
    g = model.degeneracy_factor()
    # prefactor: g / (2π)² — the triangle integral already gives ∫_T d²k δ(E−ε(k))
    # per unit area_tri, so total = Σ_T area_tri × DOS¹(E) = dk² × Σ_T DOS¹(E)
    prefactor = g / ((2 * np.pi) ** 2)

    tri_idx = _triangles_for_kmesh(nk_side)
    dos = np.zeros(nE)

    for i1, i2, i3 in tri_idx:
        for ib in range(nb_sel):
            e1 = float(E_k[i1, ib])
            e2 = float(E_k[i2, ib])
            e3 = float(E_k[i3, ib])
            dos += prefactor * _triangle_dos_exact(e1, e2, e3, E, area_tri)

    return E, dos


def compute_dos_triangle_broadened(model, nk=24, E_range=None, nE=3000,
                                    eta=0.05e-3, band_slice=None):
    """DOS via finite-η Green's-function triangle integration (legacy).

    Uses _lehmann_I0_broadened with a finite imaginary part η in
    z = E + iη.  This version is kept for regression testing against
    the exact η→0 implementation in ``compute_dos_triangle``.

    Parameters
    ----------
    model : HamiltonianModel
    nk : int
    E_range : (float, float) or None
    nE : int
    eta : float — Lorentzian broadening (eV).
    band_slice : slice or None

    Returns
    -------
    E : (nE,)
    dos : (nE,)
    """
    _, k_cart = generate_k_mesh(nk, model.reciprocal_vectors)
    Nk = len(k_cart)
    nk_side = int(np.sqrt(Nk))
    assert nk_side ** 2 == Nk

    E_k, _ = compute_eigenvalues(model, k_cart)
    if band_slice is None:
        band_slice = slice(None)
    E_k = E_k[:, band_slice]
    nb_sel = E_k.shape[1]

    if E_range is None:
        margin = 10 * eta
        E_range = (float(E_k.min()) - margin, float(E_k.max()) + margin)
    E = np.linspace(*E_range, nE)

    area_BZ = abs(np.linalg.det(model.reciprocal_vectors))
    dk = np.sqrt(area_BZ / Nk)
    area_tri = dk ** 2 / 2.0
    g = model.degeneracy_factor()
    prefactor = g / ((2 * np.pi) ** 2 * np.pi)

    tri_idx = _triangles_for_kmesh(nk_side)
    dos = np.zeros(nE)
    z_arr = E + 1j * eta

    for i1, i2, i3 in tri_idx:
        for ib in range(nb_sel):
            e1, e2, e3 = float(E_k[i1, ib]), float(E_k[i2, ib]), float(E_k[i3, ib])
            I0 = _lehmann_I0_broadened(e1, e2, e3, z_arr, area_tri)
            dos += prefactor * I0.imag

    return E, dos


# ── Legacy broadened Lehmann-Taut kernel (for regression testing) ──

def _lehmann_I0_broadened(e1, e2, e3, z, area, thr=1e-10):
    """Analytic 2D triangle integral ∫_T d²k / (ε(k) − z), finite η.

    Kept for regression testing.  Use ``_triangle_dos_exact`` for
    production (η→0 limit, no complex arithmetic).
    """
    def _L(e):
        val = e - z
        return val * np.log(val) - val

    def _D(ea, eb):
        if abs(ea - eb) < thr:
            return np.log(eb - z)
        return (_L(ea) - _L(eb)) / (ea - eb)

    de31 = e3 - e1
    if abs(de31) < thr:
        if abs(e2 - e1) < thr:
            return area / (e1 - z)
        return _lehmann_I0_broadened(e2, e1, e3, z, area, thr)
    return 2.0 * area * (_D(e2, e3) - _D(e2, e1)) / de31


# ============================================================
#  DOS 求和规则验证
# ============================================================

def check_dos_sum_rule(E, dos, g=None, nb=None, tol=1e-3):
    """Verify ∫ DOS(E) dE = g · nb (per unit cell).

    The integral of DOS over all energies must equal g (degeneracy)
    times nb (number of bands).  This holds for ANY k-mesh resolution
    once the triangle method uses exact η→0 integration.

    Parameters
    ----------
    E : (nE,) energy grid
    dos : (nE,) DOS array
    g : int or None — degeneracy factor
    nb : int or None — number of bands
    tol : float — relative tolerance

    Returns
    -------
    ok : bool — True if sum rule holds
    integral : float — ∫ DOS dE
    expected : float — g · nb
    """
    if g is None:
        g = 4
    if nb is None:
        nb = 1
    integral = float(np.trapz(dos, E))
    expected = float(g * nb)
    rel_err = abs(integral - expected) / max(expected, 1e-30)
    ok = rel_err < tol
    if not ok:
        import warnings
        warnings.warn(
            f"DOS sum rule violation: ∫DdE={integral:.4f} vs g·nb={expected:.1f} "
            f"(rel_err={rel_err:.2e})")
    return ok, integral, expected

# ============================================================
#  Van Hove 奇异点检测
# ============================================================

def find_vhs_peaks(E, dos, prominence=None, height=None, distance=None):
    """Locate VHS energies via peak-finding on DOS(E).
    
    Uses scipy.signal.find_peaks with optional prominence/height
    thresholds to reject noise.
    
    Returns list of VHS energies [eV].
    """
    from scipy.signal import find_peaks
    if prominence is None:
        prominence = 0.02 * (dos.max() - dos.min())
    if height is None:
        height = dos.min() + 0.05 * (dos.max() - dos.min())
    if distance is None:
        distance = max(1, len(E) // 50)
    peaks, _ = find_peaks(dos, prominence=prominence,
                           height=height, distance=distance)
    return [float(E[p]) for p in peaks]


def find_vhs_derivative(E, dos):
    """Locate VHS via dDOS/dE zero-crossings (sign + → −).
    
    Returns list of dicts with keys 'E_vhs' and 'dos'.
    """
    d_dos = np.gradient(dos, E)
    results = []
    for i in range(1, len(d_dos)):
        if d_dos[i - 1] > 0 and d_dos[i] <= 0:
            frac = d_dos[i - 1] / (d_dos[i - 1] - d_dos[i])
            E_cross = E[i - 1] + frac * (E[i] - E[i - 1])
            results.append({
                'E_vhs': float(E_cross),
                'dos': float(np.interp(E_cross, E, dos)),
            })
    return results


# ============================================================
#  Filling factor & CNP  (flat-pair bisection)
# ============================================================

def compute_filling(E_k, Ef, band_slice=None, kBT=0.1e-3,
                    degeneracy=None, model=None):
    """Filling factor ν at Fermi energy Ef.
    
    ν(μ) = g × (⟨Σ_bands f(E_n(k), μ)⟩_k − ⟨Σ_bands f(E_n(k), E_CNP)⟩_k)
    
    Returns ν, where ν=0 at CNP and ν ∈ [−g, +g].
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
    return degeneracy * (np.mean(occ_per_k) - occ_per_k.size / E_k.shape[1])


def compute_cnp(E_k, band_slice=None, kBT=0.1e-3,
                degeneracy=None, model=None):
    """CNP Fermi energy via bisection.
    
    Returns Ef such that compute_filling(E_k, Ef) ≈ 0.
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
        if occ_mean < 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9:
            break
    return 0.5 * (lo + hi)
