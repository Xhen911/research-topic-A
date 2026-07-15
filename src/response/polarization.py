"""
polarization.py
===============
Lindhard polarization function for TB/KP models, plus spectral-weight-transfer analysis.

Core function
-------------
- lindhard_polarization(model, q_values, w_values, ...) → dict
  Returns decoupled intra- and inter-band Π₀(q, ω).

Utilities
---------
- fermi_dirac(E, Ef, beta)          — numerically stable Fermi-Dirac
- generate_k_mesh(nk, reciprocal_vectors) — BZ parallelogram grid
- wrap_k_plus_q(k_frac, q_cart, reciprocal_vectors, lattice_vectors, wrap=True)
- compute_swt_1d(q_val, w_values, pi0_dict, q_values, Ef, vf=1.0)
- compute_swt_2d(q_values, w_values, pi0_dict, Ef, vf=1.0)

Author: Refactored from Sabio(2008) graphene_polarization_swt_tb_v2.py
Date:   2026-05-30
"""

import numpy as np

# ============================================================
#  数值稳定费米-狄拉克分布
# ============================================================

def fermi_dirac(E: np.ndarray, Ef: float, beta: float) -> np.ndarray:
    """费米-狄拉克分布 f(E) = 1 / (1 + exp(β(E - Ef))).

    对大正自变量使用零平台以避免溢出，对极小负值则自然趋近于 1。
    """
    x = beta * (E - Ef)
    f = np.zeros_like(x)
    mask = x < 100  # 远小于 exp 溢出极限 (~709)
    f[mask] = 1.0 / (1.0 + np.exp(x[mask]))
    return f


# ============================================================
#  BZ 平行四边形 k-网格
# ============================================================

def generate_k_mesh(nk: int, reciprocal_vectors: np.ndarray):
    """在第一布里渊区内生成均匀采样的平行四边形 k-网格。

    Parameters
    ----------
    nk : int
        每条倒格矢方向上的采样点数。
    reciprocal_vectors : np.ndarray, shape (2, 2)
        倒格矢基矢，每行是一个基矢 b_i。

    Returns
    -------
    k_frac : np.ndarray, shape (nk², 2)
        分数坐标 k = u·b₁ + v·b₂, 其中 (u,v) ∈ [0,1)×[0,1)。
    k_cart : np.ndarray, shape (nk², 2)
        直角坐标 k = k_frac @ reciprocal_vectors。
    """
    u = np.linspace(0, 1, nk, endpoint=False)
    v = np.linspace(0, 1, nk, endpoint=False)
    U, V = np.meshgrid(u, v)
    k_frac = np.column_stack((U.ravel(), V.ravel()))
    k_cart = k_frac @ reciprocal_vectors
    return k_frac, k_cart


# ============================================================
#  动量回绕 (wrap k+q → 第一布里渊区)
# ============================================================

def wrap_k_plus_q(
    k_frac: np.ndarray,
    q_cart: np.ndarray,
    reciprocal_vectors: np.ndarray,
    lattice_vectors: np.ndarray,
    wrap: bool = True,
) -> np.ndarray:
    """将直角坐标下的 q 矢量加到分数坐标的 k 上，必要时回绕至第一 BZ。

    Parameters
    ----------
    k_frac : np.ndarray, shape (Nk, 2)
        分数坐标下的 k 点。
    q_cart : np.ndarray, shape (2,)
        直角坐标下的 q 矢量。
    reciprocal_vectors : np.ndarray, shape (2, 2)
        倒格矢基矢。
    lattice_vectors : np.ndarray, shape (2, 2)
        实空间基矢（用于将 q 变换到分数坐标）。
    wrap : bool
        是否将结果取模 1.0 回到第一 BZ。

    Returns
    -------
    k_plus_q_cart : np.ndarray, shape (Nk, 2)
        直角坐标下的 k+q 点。
    """
    # q_frac = q_cart @ A.T / (2π), 其中 A = lattice_vectors
    q_frac = q_cart @ lattice_vectors.T / (2 * np.pi)

    k_plus_q_frac = (k_frac + q_frac) % 1.0 if wrap else (k_frac + q_frac)
    k_plus_q_cart = k_plus_q_frac @ reciprocal_vectors
    return k_plus_q_cart


# ============================================================
#  Lindhard 极化函数 (解耦 intra/inter)
# ============================================================

def lindhard_polarization(
    model,
    q_values: np.ndarray,
    w_values: np.ndarray,
    nk: int = 100,
    eta: float = 0.01,
    beta: float = 100.0,
    Ef: float = 0.0,
    use_tqdm: bool = True,
    form: bool = True,
) -> dict:
    """Lindhard 极化函数，分别返回带内 (intra) 和带间 (inter) 贡献。

    使用模型的哈密顿量对角化得到能带和本征态，通过形式因子
    |⟨m,k+q|n,k⟩|² 计算谱权重。仅支持 TB/KP 路径（无解析近似）。

    Parameters
    ----------
    model : HamiltonianModel
        遵循 src.models.base.HamiltonianModel 接口的模型实例。
    q_values : np.ndarray, shape (nq,)
        q 矢量模长数组（目前仅支持 q ∥ x̂）。
    w_values : np.ndarray, shape (nw,)
        频率数组。单位与模型哈密顿量一致。
    nk : int
        每条倒格矢方向的 k-网格点数（总数 = nk²）。
    eta : float
        展宽 η（eV 或模型能量单位）。
    beta : float
        逆温度 1/(k_B T)（eV⁻¹ 或模型能量单位倒数）。
    Ef : float
        费米能量（eV 或模型能量单位）。
    use_tqdm : bool
        是否显示进度条。
    form : bool
        是否使用形式因子。关闭时设为 1（用于调试）。

    Returns
    -------
    dict
        {'intra': pi0_intra, 'inter': pi0_inter}，
        每个均为 shape (nw, nq) 的 complex128 数组。
    """
    reciprocal_vectors = model.reciprocal_vectors
    lattice_vectors = model.lattice_vectors
    deg = model.degeneracy_factor()
    nb = model.n_bands

    # 预因子: deg/(2π)² · |det(reciprocal_vectors)| / nk²
    integral_const = (
        deg / (2 * np.pi) ** 2
        * abs(np.linalg.det(reciprocal_vectors))
        / nk ** 2
    )

    # 构建 k 网格
    k_frac, k_cart = generate_k_mesh(nk, reciprocal_vectors)
    Nk = len(k_cart)

    # ── 预计算所有 k 的本征值与 本征态 ──
    E_k = np.zeros((Nk, nb))
    V_k = np.zeros((Nk, model.n_orbitals, nb), dtype=complex)
    for i in range(Nk):
        Ei, Vi = model.solve(k_cart[i])
        E_k[i] = Ei
        V_k[i] = Vi

    nq, nw = len(q_values), len(w_values)
    pi0_intra = np.zeros((nw, nq), dtype=complex)
    pi0_inter = np.zeros((nw, nq), dtype=complex)

    _range = range(nq)
    if use_tqdm:
        try:
            from tqdm import tqdm
            _range = tqdm(_range, desc="Lindhard")
        except ImportError:
            pass

    for iq in _range:
        q_val = max(q_values[iq], 1e-12)
        q_cart = np.array([q_val, 0.0])

        # k+q (回绕至第一 BZ)
        k_q_cart = wrap_k_plus_q(
            k_frac, q_cart, reciprocal_vectors, lattice_vectors, wrap=True
        )

        # 计算 k+q 的本征值与 本征态
        E_q = np.zeros((Nk, nb))
        V_q = np.zeros((Nk, model.n_orbitals, nb), dtype=complex)
        for i in range(Nk):
            Ei, Vi = model.solve(k_q_cart[i])
            E_q[i] = Ei
            V_q[i] = Vi

        # 形式因子 M[m,n] = |⟨m,k+q|n,k⟩|²
        # V_q.conj() shape: (Nk, n_orbitals, nb)
        # V_k       shape: (Nk, n_orbitals, nb)
        # einsum 'kbm,kbn->kmn'  → (Nk, nb, nb)
        M = np.abs(np.einsum('kbm,kbn->kmn', V_q.conj(), V_k)) ** 2

        for m in range(nb):      # 末态能带 (k+q)
            for n in range(nb):  # 初态能带 (k)
                f_mn = M[:, m, n] if form else np.ones(Nk)

                f0 = fermi_dirac(E_k[:, n], Ef, beta)
                fq = fermi_dirac(E_q[:, m], Ef, beta)

                # denominator = -(E_{m,k+q} - E_{n,k}) + ω + iη
                dE = E_q[:, m] - E_k[:, n]
                denom = -dE[np.newaxis, :] + w_values[:, np.newaxis] + 1j * eta

                contribution = np.sum(
                    f_mn * (f0 - fq) / denom, axis=1
                ) * integral_const

                if m == n:
                    pi0_intra[:, iq] += contribution
                else:
                    pi0_inter[:, iq] += contribution

    return {'intra': pi0_intra, 'inter': pi0_inter}


# ============================================================
#  谱权重转移 (Spectral Weight Transfer, SWT)
# ============================================================

def compute_swt_1d(
    q_val: float,
    w_values: np.ndarray,
    pi0_dict: dict,
    q_values: np.ndarray,
    Ef: float,
    vf: float = 1.0,
):
    """对选定的 q 值求带内和带间谱权重的单维积分。

    使用辛普森积分沿 ω 轴计算 -Im[Π₀] 的分段权重。

    Parameters
    ----------
    q_val : float
        目标 q 值。
    w_values : np.ndarray, shape (nw,)
        频率网格。
    pi0_dict : dict
        lindhard_polarization 返回的字典 {'intra': ..., 'inter': ...}。
    q_values : np.ndarray, shape (nq,)
        q 值网格。
    Ef : float
        费米能量。
    vf : float
        费米速度（用于划定区域边界）。

    Returns
    -------
    weight_intra : float
        带内谱权重积分。
    weight_inter : float
        带间谱权重积分。
    total_weight : float
        全谱总权重积分。
    """
    from scipy.integrate import simpson

    iq = np.argmin(np.abs(q_values - q_val))
    actual_q = q_values[iq]

    spec_intra = -pi0_dict['intra'][:, iq].imag
    spec_inter = -pi0_dict['inter'][:, iq].imag

    omega_intra_max = vf * actual_q
    omega_inter_min = max(2.0 * Ef - vf * actual_q, vf * actual_q)

    mask_intra = w_values <= omega_intra_max
    mask_inter = w_values >= omega_inter_min

    weight_intra = simpson(spec_intra[mask_intra], w_values[mask_intra])
    weight_inter = simpson(spec_inter[mask_inter], w_values[mask_inter])
    total_weight = simpson(spec_intra + spec_inter, w_values)

    return weight_intra, weight_inter, total_weight


def compute_swt_2d(
    q_values: np.ndarray,
    w_values: np.ndarray,
    pi0_dict: dict,
    Ef: float,
    vf: float = 1.0,
):
    """二维全空间宏观区域谱权重积分。

    在 (q, ω) 平面上划分带内/带间区域并分别积分谱权重。

    Parameters
    ----------
    q_values : np.ndarray, shape (nq,)
    w_values : np.ndarray, shape (nw,)
    pi0_dict : dict
        lindhard_polarization 返回的字典。
    Ef : float
        费米能量。
    vf : float
        费米速度。

    Returns
    -------
    total_intra_weight : float
    total_inter_weight : float
    """
    Q, W = np.meshgrid(q_values, w_values)

    full_intra = -pi0_dict['intra'].imag
    full_inter = -pi0_dict['inter'].imag

    intra_mask = W <= vf * Q
    inter_mask = (W >= 2 * Ef - vf * Q) & (W >= vf * Q)

    dq = q_values[1] - q_values[0]
    dw = w_values[1] - w_values[0]
    da = dq * dw

    total_intra_weight = np.sum(full_intra[intra_mask]) * da
    total_inter_weight = np.sum(full_inter[inter_mask]) * da

    print("\n=== 2D macroscopic region integration ===")
    print(f"[*] Intra-band total spectral weight: {total_intra_weight:.4f}")
    print(f"[*] Inter-band total spectral weight: {total_inter_weight:.4f}")

    return total_intra_weight, total_inter_weight


# ============================================================
#  Cached Lindhard — 复用 CachedModel 的预计算 q-loop
# ============================================================

def lindhard_from_cache(
    cache,
    q_values,
    w_values,
    q_eps=0.0,
    eta=0.01,
    beta=100.0,
    Ef=0.0,
    form=True,
    use_tqdm=True,
):
    """Lindhard polarisation using pre-cached k+q eigenvalues/states.

    Avoids repeated diagonalisation — requires a CachedModel built
    with n_q > 0.  All q-point eigenvalues/states are read from
    ``cache.E_q`` and ``cache.V_q``.

    Parameters
    ----------
    cache : CachedModel
        Must have ``E_q`` and ``V_q`` (built with n_q > 0).
    q_values : np.ndarray, shape (nq,)
        q-vector magnitudes (Å⁻¹).  Must align with cache.q_norms.
    w_values : np.ndarray, shape (nw,)
        Frequency grid (eV).
    eta : float
        Lorentzian broadening (eV).
    beta : float
        Inverse temperature 1/(k_B T) (eV⁻¹).
    Ef : float
        Fermi energy (eV).
    form : bool
        Whether to use form factor |⟨k+q|k⟩|².
    use_tqdm : bool

    Returns
    -------
    dict  {'intra': pi0_intra, 'inter': pi0_inter}
        Each (nw, nq) complex128.
    """
    Nk = cache.Nk
    nb = cache.nb_cache
    bs = cache.bs_cache
    nq = len(q_values)
    nw = len(w_values)

    deg = cache.model.degeneracy_factor() if cache.model else 4
    vol_BZ = abs(np.linalg.det(cache.model.reciprocal_vectors)) if cache.model else 1.0
    integral_const = deg / (2 * np.pi) ** 2 * vol_BZ / Nk

    pi0_intra = np.zeros((nw, nq), dtype=complex)
    pi0_inter = np.zeros((nw, nq), dtype=complex)

    _range = range(nq)
    if use_tqdm:
        try:
            from tqdm import tqdm
            _range = tqdm(_range, desc="Lindhard(cached)")
        except ImportError:
            pass

    for iq in _range:
        # Optionally shift q=0 values by eps to avoid singularity
        _q_val = max(q_values[iq], q_eps) if q_eps > 0 else q_values[iq]
        E_q = cache.E_q[iq][:, bs]    # (Nk, nb_cache)
        V_q = cache.V_q[iq]              # (Nk, n_orbitals, nb_cache)
        E_k = cache.E_k[:, bs]         # (Nk, nb_cache)

        # Form factor |⟨m,k+q|n,k⟩|²
        if form and cache.V_k is not None:
            M = np.abs(np.einsum('kbm,kbn->kmn', V_q.conj(), cache.V_k)) ** 2
        else:
            M = np.ones((Nk, nb, nb))

        for m in range(nb):
            for n in range(nb):
                f_mn = M[:, m, n] if form else np.ones(Nk)
                f0 = fermi_dirac(E_k[:, n], Ef, beta)
                fq = fermi_dirac(E_q[:, m], Ef, beta)
                dE = E_q[:, m] - E_k[:, n]
                denom = -dE[np.newaxis, :] + w_values[:, np.newaxis] + 1j * eta
                contrib = np.sum(f_mn * (f0 - fq) / denom, axis=1) * integral_const
                if m == n:
                    pi0_intra[:, iq] += contrib
                else:
                    pi0_inter[:, iq] += contrib

    return {'intra': pi0_intra, 'inter': pi0_inter}


# ============================================================
#  Cached Lindhard — 复用 CachedModel 的预计算 q-loop
# ============================================================

def lindhard_from_cache(
    cache,
    q_values,
    w_values,
    q_eps=0.0,
    eta=0.01,
    beta=100.0,
    Ef=0.0,
    form=True,
    use_tqdm=True,
):
    """Lindhard polarisation using pre-cached k+q eigenvalues/states.

    Avoids repeated diagonalisation — requires a CachedModel built
    with n_q > 0.  All q-point eigenvalues/states are read from
    ``cache.E_q`` and ``cache.V_q``.

    Parameters
    ----------
    cache : CachedModel
        Must have ``E_q`` and ``V_q`` (built with n_q > 0).
    q_values : np.ndarray, shape (nq,)
        q-vector magnitudes (Å⁻¹).  Must align with cache.q_norms.
    w_values : np.ndarray, shape (nw,)
        Frequency grid (eV).
    eta : float
        Lorentzian broadening (eV).
    beta : float
        Inverse temperature 1/(k_B T) (eV⁻¹).
    Ef : float
        Fermi energy (eV).
    form : bool
        Whether to use form factor |⟨k+q|k⟩|².
    use_tqdm : bool

    Returns
    -------
    dict  {'intra': pi0_intra, 'inter': pi0_inter}
        Each (nw, nq) complex128.
    """
    Nk = cache.Nk
    nb = cache.nb_cache
    bs = cache.bs_cache
    nq = len(q_values)
    nw = len(w_values)

    deg = cache.model.degeneracy_factor() if cache.model else 4
    vol_BZ = abs(np.linalg.det(cache.model.reciprocal_vectors)) if cache.model else 1.0
    integral_const = deg / (2 * np.pi) ** 2 * vol_BZ / Nk

    pi0_intra = np.zeros((nw, nq), dtype=complex)
    pi0_inter = np.zeros((nw, nq), dtype=complex)

    _range = range(nq)
    if use_tqdm:
        try:
            from tqdm import tqdm
            _range = tqdm(_range, desc="Lindhard(cached)")
        except ImportError:
            pass

    for iq in _range:
        # Optionally shift q=0 values by eps to avoid singularity
        _q_val = max(q_values[iq], q_eps) if q_eps > 0 else q_values[iq]
        E_q = cache.E_q[iq][:, bs]    # (Nk, nb_cache)
        V_q = cache.V_q[iq]              # (Nk, n_orbitals, nb_cache)
        E_k = cache.E_k[:, bs]         # (Nk, nb_cache)

        # Form factor |⟨m,k+q|n,k⟩|²
        if form and cache.V_k is not None:
            M = np.abs(np.einsum('kbm,kbn->kmn', V_q.conj(), cache.V_k)) ** 2
        else:
            M = np.ones((Nk, nb, nb))

        for m in range(nb):
            for n in range(nb):
                f_mn = M[:, m, n] if form else np.ones(Nk)
                f0 = fermi_dirac(E_k[:, n], Ef, beta)
                fq = fermi_dirac(E_q[:, m], Ef, beta)
                dE = E_q[:, m] - E_k[:, n]
                denom = -dE[np.newaxis, :] + w_values[:, np.newaxis] + 1j * eta
                contrib = np.sum(f_mn * (f0 - fq) / denom, axis=1) * integral_const
                if m == n:
                    pi0_intra[:, iq] += contrib
                else:
                    pi0_inter[:, iq] += contrib

    return {'intra': pi0_intra, 'inter': pi0_inter}
