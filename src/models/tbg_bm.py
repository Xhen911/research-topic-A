# ============================================================
#  Bistritzer–MacDonald 连续体模型 — twisted bilayer graphene
# ============================================================
#
#  Reference:
#    R. Bistritzer and A.H. MacDonald,
#    "Moiré bands in twisted double-layer graphene",
#    PNAS 108, 12233 (2011).
#
#  物理约定（沿用项目标准单位：Å 和 eV）：
#    - 动量 k 以 Å⁻¹ 为单位，从 moiré Γ 点测量。
#    - 能量以 eV 为单位。
#    - ħv_F 以 eV·Å 为单位。
#    - 层间耦合 u, up 以 eV 为单位。
# ============================================================

import numpy as np
from .base import HamiltonianModel

# ── 物理常量 ────────────────────────────────────────────────
_A_G = 2.46                     # Å，石墨烯晶格常数
_G_NORM = 2 * np.pi / (np.cos(np.radians(30)) * _A_G)  # 1/Å, |G|
_VF_DEFAULT = np.sqrt(3) / 2 * 2.78 * _A_G             # ~5.92 eV·Å

# Moiré 跃迁相位因子  ω = exp(i·2π/3)
_OMEGA = np.exp(1j * 2 * np.pi / 3)


def _rotate(vec: np.ndarray, angle: float) -> np.ndarray:
    """将 2D 矢量逆时针旋转 angle 弧度。"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]]) @ vec


class BistritzMacDonaldTBG(HamiltonianModel):
    """
    Bistritzer–MacDonald 连续体模型 for twisted bilayer graphene.

    基矢排列: [|A₁⟩, |B₁⟩, |A₂⟩, |B₂⟩, …]，
    前 Nq₁ 个 2×2 块属于顶层 (layer 1)，后 Nq₂ 个块属于底层 (layer 2)。

    Parameters
    ----------
    theta : float
        扭转角（度）。默认 1.05°（魔角）。
    vF : float
        Fermi 速度 ħv_F (eV·Å)。默认 ~5.92。
    u : float
        AA 层间耦合 (eV)。默认 0.0797。
    up : float
        AB 层间耦合 (eV)。默认 0.0975。
    valley : int
        +1 为 K 谷，-1 为 K' 谷。默认 +1。
    n_shells : int
        倒空间壳层数（动量截断）。默认 4。
        小角度时需要更大的值。
    """

    n_bands: int
    n_orbitals: int
    n_valleys: int = 2
    n_spins: int = 1
    valley_degeneracy: int = 1
    spin_degeneracy: int = 2

    def __init__(
        self,
        theta: float = 1.05,
        vF: float = _VF_DEFAULT,
        u: float = 0.0797,
        up: float = 0.0975,
        valley: int = +1,
        n_shells: int = 4,
    ):
        self.theta = theta
        self.theta_rad = np.radians(theta)
        self.vF = vF
        self.u = u
        self.up = up
        self.xi = valley
        self.n_shells = n_shells

        # ── moiré 倒格子 ─────────────────────────────────
        g_norm = _G_NORM
        mg_norm = g_norm * 2 * np.sin(self.theta_rad / 2)
        mg1 = mg_norm * np.array([1.0, 0.0])
        mg2 = _rotate(mg1, np.radians(120.0))
        self.moire_reciprocal = np.array([mg1, mg2])   # 2×2
        self.reciprocal_vectors = self.moire_reciprocal.copy()

        # ── moiré 正格子 (L_i·mg_j = 2π δ_{ij}) ─────────
        # [mg1; mg2]^T · [L1; L2]^T = 2π I  →  L = 2π · (mg^T)^{-1}
        self.lattice_vectors = 2 * np.pi * np.linalg.inv(self.moire_reciprocal.T)

        # ── 构建 Q 点格点 ────────────────────────────────
        step = min(np.linalg.norm(mg1), np.linalg.norm(mg2))
        radius = n_shells * step + 0.5 * step
        axial = int(np.ceil(radius / step)) + 2

        pts1 = []          # top  layer  Q 点
        pts2 = []          # bottom layer Q 点
        # c = (mg1 + 2 * mg2) / 3   # 层间 offset

        # for i in range(-axial, axial + 1):
        #     for j in range(-axial, axial + 1):
        #         p = (i - 1 / 3) * mg1 + (j + 1 / 3) * mg2
        #         if np.linalg.norm(p) <= radius:
        #             pts1.append(p)
        #         if np.linalg.norm(p + c) <= radius:
        #             pts2.append(p + c)
        # 两层的种子偏移（约化坐标下 +K 和 -K 谷）
        offset1 = (mg1 + 2 * mg2) / 3    # = +K_M
        offset2 = -(mg1 + 2 * mg2) / 3   # = -K_M  ← 修复点

        for i in range(-axial, axial + 1):
            for j in range(-axial, axial + 1):
                p = i * mg1 + j * mg2     # 从 Γ 出发，无额外偏移
                
                p1 = p + offset1
                if np.linalg.norm(p1) <= radius:
                    pts1.append(p1)
                
                p2 = p + offset2
                if np.linalg.norm(p2) <= radius:
                    pts2.append(p2)

        Q = np.vstack([pts1, pts2]) if pts1 and pts2 else np.empty((0, 2))
        self._Q = Q
        self.Nq = len(Q)
        self._n_pts1 = len(pts1)

        # ── 轨道 / 能带计数 ──────────────────────────────
        self.n_orbitals = 2 * self.Nq
        self.n_bands = 2 * self.Nq

        # ── 最近邻连接（Q 格点上的 hop）─────────────────
        q_vectors = (
            np.array([
                [-1, -2],
                [2, 1],
                [-1, 1],
            ])
            @ self.moire_reciprocal
            / 3
        )   # 3 个 moiré 最近邻连接矢量

        Q_rounded = np.round(Q, decimals=3).tolist()
        self._Q_nn = {}
        for i in range(self.Nq):
            neighbors = []
            for p_idx, q_vec in enumerate(q_vectors):
                target = np.round(Q[i] + q_vec, decimals=3).tolist()
                if target in Q_rounded:
                    j = Q_rounded.index(target)
                    neighbors.append((j, p_idx))
            self._Q_nn[i] = neighbors

    # ────────────────────────────────────────────────────────
    #  辅助：层间跃迁 2×2 矩阵 T_{p}
    # ────────────────────────────────────────────────────────
    def _tunneling_matrix(self, p: int) -> np.ndarray:
        """
        返回 moiré 最近邻索引 p 对应的 2×2 层间跃迁矩阵。

        p = 0, 1, 2 对应三个最近邻连接方向。
        """
        u, up, xi = self.u, self.up, self.xi
        if p == 0:
            return np.array([[u, up], [up, u]])
        elif p == 1:
            return np.array([
                [u, up * _OMEGA ** (-xi)],
                [up * _OMEGA ** (xi), u],
            ])
        elif p == 2:
            return np.array([
                [u, up * _OMEGA ** (xi)],
                [up * _OMEGA ** (-xi), u],
            ])
        return np.zeros((2, 2), dtype=complex)

    # ────────────────────────────────────────────────────────
    #  哈密顿量
    # ────────────────────────────────────────────────────────
    def hamiltonian(self, k: np.ndarray) -> np.ndarray:
        """
        构建 2Nq × 2Nq BM 哈密顿量（厄米矩阵）。

        Parameters
        ----------
        k : np.ndarray, shape (2,)
            Bloch 波矢 (1/Å)，从 moiré Γ 点测量。

        Returns
        -------
        H : np.ndarray, shape (2Nq, 2Nq), dtype=complex
        """
        N = self.Nq
        H = np.zeros((2 * N, 2 * N), dtype=complex)

        for i in range(N):
            # 确定层指标: l = -1 (top), +1 (bottom)
            l = 1.0 if i >= self._n_pts1 else -1.0

            # 该层的旋转角（half-twist）: θ_l = l·ξ·θ_rad
            theta_l = l * self.xi * self.theta_rad
            sin_half = np.sin(theta_l / 2)
            rot = np.array([[1.0, -sin_half], [sin_half, 1.0]])

            # Dirac 锥展开点 (layer-resolved)
            kj = rot @ (k + self._Q[i])

            # 2×2 Dirac 对角块（上三角）
            km = self.xi * kj[0] - 1j * kj[1]
            H[2 * i, 2 * i + 1] = -self.vF * km

            # 2×2 层间（非对角）块
            for j, p in self._Q_nn.get(i, []):
                block = self._tunneling_matrix(p)
                H[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = block

        # 厄米对称化:  H_full = H + H†
        # 说明：
        #  - H 目前只填了上三角对角块 + 完整的非对角 2×2 块（仅在 Q_nn 方向）。
        #  - Q_nn 是有向的且双方不会互为邻居，因此每个 (i,j) 对只填一次。
        #  - H + H† 得到完整厄米矩阵。
        H = H + H.conj().T
        return H

    # ────────────────────────────────────────────────────────
    #  对角化
    # ────────────────────────────────────────────────────────
    def solve(self, k: np.ndarray) -> tuple:
        """
        对角化哈密顿量，返回 (energies, states)。

        Returns
        -------
        energies : np.ndarray, shape (2Nq,)
            升序本征值 (eV)
        states : np.ndarray, shape (2Nq, 2Nq), dtype=complex
            列向量为本征态，已固定相位规范。
        """
        H = self.hamiltonian(k)
        E, V = np.linalg.eigh(H)
        # 相位规范固定：每列最大绝对值分量的相位归零
        for n in range(V.shape[1]):
            idx = np.argmax(np.abs(V[:, n]))
            phase = np.angle(V[idx, n])
            V[:, n] *= np.exp(-1j * phase)
        return E, V

    # ────────────────────────────────────────────────────────
    #  moiré BZ 高对称点
    # ────────────────────────────────────────────────────────
    def high_symmetry_points(self) -> dict:
        """返回 moiré BZ 高对称点坐标 (1/Å)。"""
        mg1, mg2 = self.reciprocal_vectors
        return {
            "\u0393": np.zeros(2),               # Γ (moiré)
            "K": (2 * mg1 + mg2) / 3,           # K (moiré)
            "K'": (mg1 + 2 * mg2) / 3,          # K' (moiré)
            "M": (mg1 + mg2) / 2,               # M (moiré)
        }

    # ────────────────────────────────────────────────────────
    #  厄米性检验
    # ────────────────────────────────────────────────────────
    def check_hermitian(self, k: np.ndarray, tol: float = 1e-10) -> bool:
        H = self.hamiltonian(k)
        return np.max(np.abs(H - H.conj().T)) < tol
