"""
graphene_tb_hybrid.py — TB 带结构 + Dirac 点极坐标积分 RPA
==========================================================

解决 graphene_tb_rpa.py 均匀 BZ 网格在低掺杂时效率低下的问题：
对每个 Dirac 点（K, K'）用极坐标 (k, θ) 采样，天然聚焦于
低能物理区域，避免在全 BZ 均匀撒点。

方法
----
全数值尾减法：
  χ₀ = χ₀_TB(doped, compact) + [χ₀_Dirac(large, undoped) − χ₀_Dirac(compact, undoped)]

- χ₀_TB(compact): TB 全响应（intra+inter），Dirac 点附近紧致区域
- χ₀_Dirac(large): 更大 k 范围数值 Dirac 无掺杂响应（同数值方案）
- Tail = large − compact: 同一数值方案、完美对消，无离散化不匹配
- 根因发现 (2026-07-23): 虚假正 Im 来自分析 χ₀^und 与数值 Dirac 的
  离散化不匹配（≫ TB 物理效应），用全数值替代消除。

与 graphene_rpa.py 的差别
--------------------------
- 带结构: TB E(k) vs Dirac E(k) = v_F·k
- 形式因子: |⟨k+q|k⟩|² from TB eigenstates vs analytic chirality
- 方法: TB(compact) + [analytical Dirac tail]，而非 Dirac(解析) 全空间 + 掺杂校正
  尾减法保证 TB→Dirac 极限下精确恢复 Dirac 解析结果
- 2026-07-23 fix: 改用数值尾减法替代无掺杂减除 —
  TB(compact) + [analytical − Dirac_numerical(compact)]，
  消除 TB−Dirac 在 interband 的不完全对消

Date: 2026-07-23
"""
import sys
import numpy as np

# ── 仓库路径 ──
# 优先使用本地 API 拉取的副本（沙箱 git clone 不可靠）
REPO_ROOT = "D:/erpt/dump/_rta_src"
# 回退: "D:/erpt/research-topic-A"  (可能被沙箱损坏)

# 物理常数
E2_4PIEPS0 = 14.3996
A_PHYS = 2.46
T_HOP = 2.78
G_DEG = 4  # g_s·g_v
VF_PHYS = np.sqrt(3) / 2 * T_HOP * A_PHYS


def _ensure_repo():
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)


def dirac_points(model):
    """返回两个不等价 Dirac 点的笛卡尔坐标 [K, K']。"""
    K = model.high_symmetry_points()['K']
    return np.array([K, -K])


def build_model(t=T_HOP):
    _ensure_repo()
    from src.bands.graphene import SingleLayerGrapheneTB
    return SingleLayerGrapheneTB(t=t)


def coulomb_phys(q_phys, kappa=1.0):
    """物理单位 2D Coulomb V(q) = 2π α̃/(κ q) [eV·Å²]"""
    q_safe = np.where(np.asarray(q_phys, dtype=float) < 1e-12, 1e-12,
                      np.asarray(q_phys, dtype=float))
    return 2 * np.pi * E2_4PIEPS0 / (kappa * q_safe)


# ============================================================
#  TB-hybrid χ₀
# ============================================================

def chi0_tb_hybrid(model, q_values, w_values, Ef, vf=None,
                   Nk=400, Ntheta=180, k_cut_factor=3.0, eta=0.005,
                   beta=200.0, verbose=True):
    """TB 带结构 + Dirac 点极坐标 Lindhard 极化（全数值尾减法）。

    方法：χ₀ = χ₀_TB(doped, compact) + [χ₀_Dirac(large, undoped) − χ₀_Dirac(compact, undoped)]

    - χ₀_TB(compact): TB 全响应，Dirac 点附近紧致区域
    - χ₀_Dirac(compact): 同网格数值 Dirac 无掺杂响应
    - χ₀_Dirac(large): 更大 k 范围数值 Dirac 无掺杂响应（同数值方案）
    - Tail = large − compact: 均为数值、同方案 → 完美对消，无离散化不匹配

    与旧版的关键差别：
    用全数值 Dirac(large) 替代解析 χ₀^und，消除分析解与数值积分的
    离散化不匹配——这是虚假正 Im χ₀ 的根因（≫ TB 三角翘曲效应）。

    Parameters
    ----------
    model : SingleLayerGrapheneTB
    q_values : array, shape (nq,)  模型无量纲 q 模长 (q || x̂)
    w_values : array, shape (nw,)  频率 [eV]
    Ef : float  费米能 [eV]（n型 Ef>0）
    vf : float or None  Dirac v_F 模型单位（None=√3/2·t）
    Nk : int  径向点数（compact 区域）
    Ntheta : int  角度点数
    k_cut_factor : float  紧致区域 = k_cut_factor × max(k_F, q)
    eta : float  展宽 [eV]
    beta : float  逆温度 [eV⁻¹]
    verbose : bool

    Returns
    -------
    chi0 : complex array, shape (nw, nq)  模型单位 [eV⁻¹]
    """
    _ensure_repo()
    from src.bands.vertices import density_form_factor
    from src.propagators.lindhard import fermi_dirac

    q_values = np.atleast_1d(np.asarray(q_values, dtype=float))
    w_values = np.atleast_1d(np.asarray(w_values, dtype=float))
    Ef = abs(Ef)
    nq, nw = len(q_values), len(w_values)

    if vf is None:
        vf = np.sqrt(3) / 2 * T_HOP  # 模型单位

    # Dirac 点
    D_pts = dirac_points(model)  # (2, 2): K, K'
    deg_spin = model.degeneracy_factor()  # 2
    prefactor = deg_spin / (2 * np.pi) ** 2
    kF = Ef / vf  # Dirac 近似，用于 k_cut 估算

    # ── 公共：Dirac 形式因子���算 ──
    def _make_dirac_ff(k_rel, kq_rel, cos_gamma):
        Fpp = 0.5 * (1.0 + cos_gamma)
        Fpm = 0.5 * (1.0 - cos_gamma)
        return np.array([[Fpp, Fpm], [Fpm, Fpp]])

    # ── 公共：单 (k,θ) 点的 Dirac 无掺杂贡献 ──
    def _dirac_undoped_contrib(k_rel, kq_rel, cos_gamma, wgt, w_block):
        Ek_d = np.array([-vf * k_rel, vf * k_rel])
        Ekq_d = np.array([-vf * kq_rel, vf * kq_rel])
        M_d = _make_dirac_ff(k_rel, kq_rel, cos_gamma)
        fk_und = np.array([1.0, 0.0])
        fkq_und = np.array([1.0, 0.0])
        contrib = np.zeros(len(w_block), dtype=complex)
        for m in range(2):
            for n in range(2):
                fdd = fk_und[n] - fkq_und[m]
                dEd = Ekq_d[m] - Ek_d[n]
                contrib += wgt * M_d[m, n] * fdd / (-dEd + w_block)
        return contrib

    # ── χ₀_Dirac (large, undoped) ──
    # 嵌套网格: compact k-points ⊂ large k-points → 精确对消
    chi0_dirac_large = np.zeros((nw, nq), dtype=complex)

    # ── χ₀_TB (compact) + χ₀_Dirac (compact) ──
    chi0_tb_compact = np.zeros((nw, nq), dtype=complex)
    chi0_dirac_compact = np.zeros((nw, nq), dtype=complex)

    # ── 网格构建 ──
    max_q = q_values.max()

    # Compact 网格
    R_compact = max(kF, max_q) * k_cut_factor
    k_edges_c = np.linspace(0, R_compact, Nk + 1)
    k_centers_c = 0.5 * (k_edges_c[:-1] + k_edges_c[1:])
    dk_c = np.diff(k_edges_c)

    # Large 嵌套网格: compact 点 + 外部补充点
    # 用更大的 R_large 和更多外层点确保 tail 收敛
    R_large_factor = 8.0  # 大半径因子
    R_large = max(kF, max_q) * R_large_factor
    Nk_outer = Nk * 3  # 外层点数更多以收敛
    k_edges_outer = np.linspace(R_compact, R_large, Nk_outer + 1)
    k_centers_outer = 0.5 * (k_edges_outer[:-1] + k_edges_outer[1:])
    dk_outer = np.diff(k_edges_outer)

    theta_arr = np.linspace(0, 2 * np.pi, Ntheta, endpoint=False)
    dtheta = 2 * np.pi / Ntheta

    w_block = w_values.astype(complex) + 1j * eta

    for iq in range(nq):
        q_val = max(q_values[iq], 1e-12)
        q_vec = np.array([q_val, 0.0])
        R_q_c = max(kF, q_val) * k_cut_factor
        mask_k_c = k_centers_c <= R_q_c

        if verbose:
            print(f"  [TB-hybrid] iq={iq + 1}/{nq}  q={q_val:.4f}")

        for iv, D in enumerate(D_pts):
            # ── TB(compact) + Dirac(compact): 同一网格 ──
            for ik in range(Nk):
                if not mask_k_c[ik]:
                    continue
                k_rel = k_centers_c[ik]
                for it in range(Ntheta):
                    wgt = prefactor * k_rel * dk_c[ik] * dtheta
                    if wgt == 0:
                        continue
                    theta = theta_arr[it]

                    # Dirac(compact)
                    kq_rel = np.sqrt(k_rel**2 + q_val**2
                                     + 2 * k_rel * q_val * np.cos(theta))
                    cos_g = np.clip(
                        (k_rel + q_val * np.cos(theta)) / max(kq_rel, 1e-30),
                        -1.0, 1.0)
                    chi0_dirac_compact[:, iq] += _dirac_undoped_contrib(
                        k_rel, kq_rel, cos_g, wgt, w_block)

                    # TB
                    kx = D[0] + k_rel * np.cos(theta)
                    ky = D[1] + k_rel * np.sin(theta)
                    k_vec = np.array([kx, ky])
                    Ek, Vk = model.solve(k_vec)
                    Ekq, Vkq = model.solve(k_vec + q_vec)
                    M_tb = density_form_factor(
                        Vk[np.newaxis, :, :], Vkq[np.newaxis, :, :])[0]
                    fk_tb = fermi_dirac(Ek, Ef, beta)
                    fkq_tb = fermi_dirac(Ekq, Ef, beta)
                    for m in range(model.n_bands):
                        for n in range(model.n_bands):
                            chi0_tb_compact[:, iq] += wgt * M_tb[m, n] * (
                                fk_tb[n] - fkq_tb[m]) / (
                                - (Ekq[m] - Ek[n]) + w_block)

            # ── Dirac(large) = Dirac(compact) + Dirac(outer tail) ──
            # 复用已计算的 compact 部分
            chi0_dirac_large[:, iq] += chi0_dirac_compact[:, iq]

            # 外层 tail 贡献
            R_q_outer = max(kF, q_val) * R_large_factor
            mask_k_outer = k_centers_outer <= R_q_outer
            for ik in range(Nk_outer):
                if not mask_k_outer[ik]:
                    continue
                k_rel = k_centers_outer[ik]
                for it in range(Ntheta):
                    wgt = prefactor * k_rel * dk_outer[ik] * dtheta
                    if wgt == 0:
                        continue
                    theta = theta_arr[it]
                    kq_rel = np.sqrt(k_rel**2 + q_val**2
                                     + 2 * k_rel * q_val * np.cos(theta))
                    cos_g = np.clip(
                        (k_rel + q_val * np.cos(theta)) / max(kq_rel, 1e-30),
                        -1.0, 1.0)
                    chi0_dirac_large[:, iq] += _dirac_undoped_contrib(
                        k_rel, kq_rel, cos_g, wgt, w_block)

    # ── 组合: χ₀ = TB(compact) + [Dirac(large) − Dirac(compact)]
    #     = TB(compact) + Dirac(outer tail)  ──
    chi0_total = chi0_tb_compact + (chi0_dirac_large - chi0_dirac_compact)
    return chi0_total


def chi0_to_physical(chi0_model):
    """χ₀ 模型单位 [eV⁻¹] → 物理单位 [eV⁻¹·Å⁻²]"""
    return np.asarray(chi0_model) / A_PHYS ** 2


def rpa_dielectric_phys(chi0_phys, q_phys, kappa=1.0):
    """从物理单位 χ₀ 计算 ε, ε⁻¹, ELF。

    复用 graphene_rpa.py 的 L3/L4 接口。
    """
    import graphene_rpa as gr
    eps, eps_inv, elf = gr.rpa_dielectric(
        chi0_phys, q_phys, kappa=kappa,
        repo_root=REPO_ROOT,
    )
    return eps, eps_inv, elf


# ============================================================
#  解析基准（Dirac 近似）
# ============================================================

def dos_dirac_model(Ef, vf=None):
    """Dirac DOS [模型单位 eV⁻¹]"""
    if vf is None:
        vf = np.sqrt(3) / 2 * T_HOP
    return G_DEG * abs(Ef) / (2.0 * np.pi * vf ** 2)


def kF_model(Ef, vf=None):
    """费米波矢 [模型单位]"""
    if vf is None:
        vf = np.sqrt(3) / 2 * T_HOP
    return abs(Ef) / vf


def plasmon_freq_phys(q_phys, Ef, kappa=1.0):
    """长波等离激元 [eV]"""
    q_phys = np.asarray(q_phys, dtype=float)
    return np.sqrt(G_DEG * E2_4PIEPS0 * abs(Ef) * q_phys / (2.0 * kappa))
