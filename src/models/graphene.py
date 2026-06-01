from .base import HamiltonianModel
import numpy as np

# ============================================================
#  TB 模型用常量（无量纲化，a=1）
# ============================================================

# 实空间基矢矩阵 A (每行是一个基矢, a=1 无量纲)
_A = np.array([
    [0.5, np.sqrt(3)/2],   # a1
    [-0.5, np.sqrt(3)/2]   # a2
])

# 倒空间基矢矩阵 B (每行是一个基矢)
_B = 2 * np.pi * np.array([
    [1.0, 1.0/np.sqrt(3)],  # b1
    [-1.0, 1.0/np.sqrt(3)]  # b2
])

# 最邻近矢量 delta (A→B), |delta| = a_cc = a/√3 = 1/√3
_DELTA = np.array([
    [0, 1],
    [np.sqrt(3)/2, -0.5],
    [-np.sqrt(3)/2, -0.5]
]) / np.sqrt(3)


def _f_k(k: np.ndarray) -> complex:
    """
    紧束缚模型共用结构因子。

    f(k) = Σⱼ exp(i k·δⱼ),  j 遍历 3 个最近邻矢量 δⱼ。

    Parameters
    ----------
    k : np.ndarray, shape (2,)
        无量纲 k 矢量。

    Returns
    -------
    complex
    """
    return np.sum(np.exp(1j * (k @ _DELTA.T)))

# ============================================================
#  KP 模型用常量（物理单位）
# ============================================================

_A_GRAPHENE = 2.46   # Å
_VF_DEFAULT = 5.6    # eV·Å（ħv_F）


def _graphene_reciprocal(a=_A_GRAPHENE):
    """所有石墨烯类模型共用"""
    ...


# ============================================================
#  SingleLayerGrapheneTB — 单层石墨烯最近邻紧束缚模型
# ============================================================

class SingleLayerGrapheneTB(HamiltonianModel):
    """
    单层石墨烯最近邻紧束缚模型。
    H(k) = [  0,    -t·f(k)  ]
           [-t·f*(k),   0     ]
    其中 f(k) = Σ_j exp(i k·δ_j)，j 遍历 3 个最近邻矢量。

    本征值: E_±(k) = ± t·|f(k)|
    本征态: |±,k⟩ = 1/√2 (1, ∓e^{-iφ_k})^T, φ_k = arg(f(k))

    单位约定：
    - 能量: eV（或项目约定的无量纲单位，t 的取值决定）
    - k: 与倒格矢 B 一致的无量纲单位（a=1）
    - 该模型覆盖整个 BZ，K 和 K' 谷均包含在内
    """
    n_bands = 2
    n_orbitals = 2
    n_valleys = 2            # K 和 K' 均在 BZ 内
    n_spins = 1              # 不含自旋（手动乘 factor）
    valley_degeneracy = 1    # BZ 已包含两个谷
    spin_degeneracy = 2      # 自旋简并因子

    def __init__(self, t: float = 1.0):
        """
        Parameters
        ----------
        t : float
            最近邻跃迁能 (eV 或约定的无量纲单位)。
            默认 t=1.0 对应 v_F = (√3/2)·t ≈ 0.866，
            与项目已有的 vf=1.0 Dirac 近似接近。
        """
        self.t = t
        self.lattice_vectors = _A.copy()
        self.reciprocal_vectors = _B.copy()
        self._delta = _DELTA  # shape (3, 2)
        # 高对称点（K 点和 K' 点）
        self._K = np.array([4*np.pi/3, 0])
        self._Kp = np.array([-4*np.pi/3, 0])

    def hamiltonian(self, k: np.ndarray) -> np.ndarray:
        """计算 k 点的 2×2 TB 哈密顿量（厄米矩阵）."""
        f_k = _f_k(k)
        return self.t * np.array([
            [0.0, -f_k],
            [-np.conj(f_k), 0.0]
        ])

    def solve(self, k: np.ndarray) -> tuple:
        """
        对角化哈密顿量，返回 (energies, states)。

        Returns
        -------
        energies : np.ndarray, shape (2,)
            升序排列：E[0] = 价带 (-t|f|), E[1] = 导带 (+t|f|)
        states : np.ndarray, shape (2, 2)
            列向量为本征态，已固定相位规范：
            每列最大绝对值分量的相位归零（确保 k 空间连续性）
        """
        H = self.hamiltonian(k)
        E, V = np.linalg.eigh(H)
        # 相位规范固定：每列第一个非零元取实数正值
        for n in range(V.shape[1]):
            idx = np.argmax(np.abs(V[:, n]))
            phase = np.angle(V[idx, n])
            V[:, n] *= np.exp(-1j * phase)
        return E, V

    def _compute_bands_batch(self, kvec: np.ndarray) -> tuple:
        """
        批量计算 k 点的本征值和本征态。

        Parameters
        ----------
        kvec : np.ndarray, shape (Nk, 2)

        Returns
        -------
        E : np.ndarray, shape (Nk, 2)
            本征值，E[i, 0] = 价带, E[i, 1] = 导带
        V : np.ndarray, shape (Nk, 2, 2), dtype=complex
            本征态，V[i, :, n] 是第 i 个 k 点的第 n 个本征态
        """
        Nk = len(kvec)
        E = np.zeros((Nk, self.n_bands))
        V = np.zeros((Nk, self.n_orbitals, self.n_bands), dtype=complex)
        for i in range(Nk):
            Ei, Vi = self.solve(kvec[i])
            E[i] = Ei
            V[i] = Vi
        return E, V

    def high_symmetry_points(self) -> dict:
        """返回 BZ 高对称点坐标."""
        a = 1.0  # 无量纲
        return {
            'Γ': np.zeros(2),
            'K': np.array([4*np.pi/(3*a), 0]),
            "K'": np.array([-4*np.pi/(3*a), 0]),
            'M': np.array([np.pi/a, np.pi/(np.sqrt(3)*a)]),
        }

    def check_hermitian(self, k: np.ndarray, tol: float = 1e-10) -> bool:
        H = self.hamiltonian(k)
        return np.max(np.abs(H - H.conj().T)) < tol


# ============================================================
#  BilayerGrapheneTB — AB (Bernal) 堆叠双层石墨烯紧束缚模型
# ============================================================

class BilayerGrapheneTB(HamiltonianModel):
    """
    AB (Bernal) 堆叠双层石墨烯紧束缚模型。

    基矢顺序: [A1, B1, A2, B2]
    无量纲化约定同 SingleLayerGrapheneTB：a=1, k 无量纲。

    Parameters
    ----------
    t : float
        层内最近邻跃迁 γ₀ (eV). Default 2.78.
    gamma1 : float
        二聚体耦合 γ₁, B1↔A2 (eV). Default 0.39 for AB.
        设为 0 即退化为 AA 堆叠。
    gamma3 : float
        非二聚体耦合 γ₃, A1↔B2 (eV). Default 0.
    gamma4 : float
        二聚体-非二聚体耦合 γ₄ (eV). Default 0.
        非零时参与层内修正和层间耦合（AA 堆叠时提供主要层间耦合）。
    u : float
        层间偏压 (eV), 破缺反演对称性. Default 0.
    dp : float
        二聚体/非二聚体位能差 Δ' (eV). Default 0.
    dab : float
        子晶格不对称 Δ_AB (eV). Default 0.
    """
    n_bands = 4
    n_orbitals = 4
    n_valleys = 2
    n_spins = 1
    valley_degeneracy = 1
    spin_degeneracy = 2

    def __init__(
        self,
        t: float = 2.78,
        gamma1: float = 0.39,
        gamma3: float = 0.0,
        gamma4: float = 0.0,
        u: float = 0.0,
        dp: float = 0.0,
        dab: float = 0.0,
    ):
        self.t = t
        self.gamma1 = gamma1
        self.gamma3 = gamma3
        self.gamma4 = gamma4
        self.u = u
        self.dp = dp
        self.dab = dab

        self.lattice_vectors = _A.copy()
        self.reciprocal_vectors = _B.copy()

    def hamiltonian(self, k: np.ndarray) -> np.ndarray:
        """
        构造 4×4 双层石墨烯哈密顿量（厄米矩阵）。

        H = H_onsite + H_intra + H_inter

        AB 堆叠 (gamma1 ≠ 0):
          - H_intra 包含 gamma4/t 调制的层内交叉项
          - H_inter: gamma1 耦合 B1↔A2 (二聚体), gamma3 耦合 A1↔B2 (非二聚体)
        AA 堆叠 (gamma1 = 0):
          - H_intra 不含 gamma4 层内交叉项
          - H_inter: gamma4 提供常数层间耦合 A1↔A2, B1↔B2
        """
        f = _f_k(k)
        fc = np.conj(f)

        H = np.zeros((4, 4), dtype=complex)

        # --- onsite (diagonal) ---
        H[0, 0] = (-self.u + self.dab) / 2.0
        H[1, 1] = (-self.u - self.dab) / 2.0 + self.dp
        H[2, 2] = (self.u + self.dab) / 2.0 + self.dp
        H[3, 3] = (self.u - self.dab) / 2.0

        # --- intralayer ---
        H[0, 1] = -self.t * f              # A1-B1
        H[1, 0] = -self.t * fc
        H[2, 3] = -self.t * f              # A2-B2
        H[3, 2] = -self.t * fc

        if self.gamma1 != 0.0:
            # ---- AB stacking ----
            # intralayer gamma4 cross terms
            if self.gamma4 != 0.0:
                H[0, 2] += -self.gamma4 / self.t * f     # A1-A2
                H[2, 0] += -self.gamma4 / self.t * fc
                H[1, 3] += -self.gamma4 / self.t * f     # B1-B2
                H[3, 1] += -self.gamma4 / self.t * fc

            # interlayer
            H[1, 2] = self.gamma1            # B1-A2 (dimer)
            H[2, 1] = self.gamma1
            if self.gamma3 != 0.0:
                H[0, 3] = self.gamma3 / self.t * fc    # A1-B2 (skew)
                H[3, 0] = self.gamma3 / self.t * f
        else:
            # ---- AA stacking: gamma4 provides main interlayer coupling ----
            if self.gamma4 != 0.0:
                H[0, 2] = self.gamma4         # A1-A2
                H[2, 0] = self.gamma4
                H[1, 3] = self.gamma4         # B1-B2
                H[3, 1] = self.gamma4

        return H

    def solve(self, k: np.ndarray) -> tuple:
        """
        对角化哈密顿量，返回 (energies, states)。

        Returns
        -------
        energies : np.ndarray, shape (4,)
            升序排列
        states : np.ndarray, shape (4, 4)
            列向量为本征态，已固定相位规范：
            每列最大绝对值分量的相位归零（确保 k 空间连续性）
        """
        H = self.hamiltonian(k)
        E, V = np.linalg.eigh(H)
        # 相位规范固定
        for n in range(V.shape[1]):
            idx = np.argmax(np.abs(V[:, n]))
            phase = np.angle(V[idx, n])
            V[:, n] *= np.exp(-1j * phase)
        return E, V

    def _compute_bands_batch(self, kvec: np.ndarray) -> tuple:
        """
        批量计算 k 点的本征值和本征态。

        Parameters
        ----------
        kvec : np.ndarray, shape (Nk, 2)

        Returns
        -------
        E : np.ndarray, shape (Nk, 4)
        V : np.ndarray, shape (Nk, 4, 4), dtype=complex
        """
        Nk = len(kvec)
        E = np.zeros((Nk, self.n_bands))
        V = np.zeros((Nk, self.n_orbitals, self.n_bands), dtype=complex)
        for i in range(Nk):
            Ei, Vi = self.solve(kvec[i])
            E[i] = Ei
            V[i] = Vi
        return E, V

    def high_symmetry_points(self) -> dict:
        """返回 BZ 高对称点坐标."""
        a = 1.0  # 无量纲
        return {
            'Γ': np.zeros(2),
            'K': np.array([4 * np.pi / (3 * a), 0]),
            "K'": np.array([-4 * np.pi / (3 * a), 0]),
            'M': np.array([np.pi / a, np.pi / (np.sqrt(3) * a)]),
        }

    def check_hermitian(self, k: np.ndarray, tol: float = 1e-10) -> bool:
        H = self.hamiltonian(k)
        return np.max(np.abs(H - H.conj().T)) < tol


# ============================================================
#  SingleLayerGrapheneKP — 单层石墨烯 k·p 模型（Dirac 哈密顿量）
# ============================================================

class SingleLayerGrapheneKP(HamiltonianModel):
    """
    单层石墨烯 k·p 模型（一阶 Taylor，Dirac 哈密顿量）。

    H = vF * [ 0        xi*kx - i*ky ]
             [ xi*kx + i*ky    0    ]

    其中 xi = +1 (K valley) 或 -1 (K' valley)，
    (kx, ky) 是从 Dirac 点测量的相对波矢。

    Parameters
    ----------
    t : float
        最近邻跃迁 (eV). vF = (√3/2)·t. Default 2.78.
    valley : int
        +1 for K, -1 for K'. Default +1.
    """
    n_bands = 2
    n_orbitals = 2
    n_valleys = 2
    n_spins = 1
    valley_degeneracy = 1
    spin_degeneracy = 2

    def __init__(self, t: float = 2.78, valley: int = +1):
        self.t = t
        self.xi = valley
        self.vF = np.sqrt(3) / 2 * t

        # Dirac points in dimensionless units (a=1)
        self.K_point = np.array([4 * np.pi / 3, 0])
        self.Kp_point = np.array([-4 * np.pi / 3, 0])

        self.lattice_vectors = _A.copy()
        self.reciprocal_vectors = _B.copy()

    def hamiltonian(self, k: np.ndarray) -> np.ndarray:
        """计算 k 点的 2×2 Dirac 哈密顿量（厄米矩阵）."""
        valley_point = self.K_point if self.xi == +1 else self.Kp_point
        q = k - valley_point
        xi_qx = self.xi * q[0]
        qy = q[1]
        return self.vF * np.array([
            [0.0, xi_qx - 1j * qy],
            [xi_qx + 1j * qy, 0.0]
        ])

    def solve(self, k: np.ndarray) -> tuple:
        """
        对角化哈密顿量，返回 (energies, states)。

        Returns
        -------
        energies : np.ndarray, shape (2,)
            升序排列：E[0] = -vF|q|, E[1] = +vF|q|
        states : np.ndarray, shape (2, 2)
            列向量为本征态，已固定相位规范。
        """
        H = self.hamiltonian(k)
        E, V = np.linalg.eigh(H)
        for n in range(V.shape[1]):
            idx = np.argmax(np.abs(V[:, n]))
            phase = np.angle(V[idx, n])
            V[:, n] *= np.exp(-1j * phase)
        return E, V

    def _compute_bands_batch(self, kvec: np.ndarray) -> tuple:
        """批量计算 k 点的本征值和本征态。"""
        Nk = len(kvec)
        E = np.zeros((Nk, self.n_bands))
        V = np.zeros((Nk, self.n_orbitals, self.n_bands), dtype=complex)
        for i in range(Nk):
            Ei, Vi = self.solve(kvec[i])
            E[i] = Ei
            V[i] = Vi
        return E, V

    def high_symmetry_points(self) -> dict:
        """返回 BZ 高对称点坐标."""
        a = 1.0  # 无量纲
        return {
            'Γ': np.zeros(2),
            'K': np.array([4 * np.pi / (3 * a), 0]),
            "K'": np.array([-4 * np.pi / (3 * a), 0]),
            'M': np.array([np.pi / a, np.pi / (np.sqrt(3) * a)]),
        }

    def check_hermitian(self, k: np.ndarray, tol: float = 1e-10) -> bool:
        H = self.hamiltonian(k)
        return np.max(np.abs(H - H.conj().T)) < tol


# ============================================================
#  BilayerGrapheneKP — AB (Bernal) 堆叠双层石墨烯 k·p 模型
# ============================================================

class BilayerGrapheneKP(HamiltonianModel):
    """
    AB (Bernal) 堆叠双层石墨烯 k·p 模型。

    基矢顺序: [A1, B1, A2, B2]
    采用 McCann-Koshino 参数化，π = xi*kx + i*ky。

    Parameters
    ----------
    t : float
        层内最近邻 γ₀ (eV). Default 2.78.
    gamma1 : float
        二聚体耦合 γ₁ (eV). Default 0.39.
    gamma3 : float
        非二聚体斜向耦合 γ₃ (eV). Default 0.0.
    gamma4 : float
        二聚体-非二聚体耦合 γ₄ (eV). Default 0.0.
    u : float
        层间偏压 (eV). Default 0.0.
    delta : float
        在位能差 Δ, 破缺 AB 子晶格对称 (eV). Default 0.0.
    dp : float
        二聚体/非二聚体位能差 Δ' (eV). Default 0.0.
    valley : int
        +1 for K, -1 for K'. Default +1.
    """
    n_bands = 4
    n_orbitals = 4
    n_valleys = 2
    n_spins = 1
    valley_degeneracy = 1
    spin_degeneracy = 2

    def __init__(
        self,
        t: float = 2.78,
        gamma1: float = 0.39,
        gamma3: float = 0.0,
        gamma4: float = 0.0,
        u: float = 0.0,
        delta: float = 0.0,
        dp: float = 0.0,
        valley: int = +1,
    ):
        self.t = t
        self.gamma1 = gamma1
        self.gamma3 = gamma3
        self.gamma4 = gamma4
        self.u = u
        self.delta = delta
        self.dp = dp
        self.xi = valley

        # Velocity parameters: v = (√3/2) * energy
        self.v0 = np.sqrt(3) / 2 * t
        self.v3 = np.sqrt(3) / 2 * gamma3
        self.v4 = np.sqrt(3) / 2 * gamma4

        self.K_point = np.array([4 * np.pi / 3, 0])
        self.Kp_point = np.array([-4 * np.pi / 3, 0])

        self.lattice_vectors = _A.copy()
        self.reciprocal_vectors = _B.copy()

    def hamiltonian(self, k: np.ndarray) -> np.ndarray:
        """
        构造 4×4 双层 k·p 哈密顿量（厄米矩阵）。

        H = [  Δ+u/2      v0*π†       v4*π†       v3*π   ]
            [  v0*π     -Δ+u/2+dp      γ₁        v4*π†  ]
            [  v4*π        γ₁       -Δ-u/2+dp    v0*π†  ]
            [  v3*π†      v4*π        v0*π       Δ-u/2  ]
        """
        valley_point = self.K_point if self.xi == +1 else self.Kp_point
        q = k - valley_point
        pi = self.xi * q[0] + 1j * q[1]      # π  = xi*kx + i*ky
        pi_dag = self.xi * q[0] - 1j * q[1]  # π† = xi*kx - i*ky

        H = np.zeros((4, 4), dtype=complex)

        # Diagonal (onsite)
        H[0, 0] = self.delta + self.u / 2
        H[1, 1] = -self.delta + self.u / 2 + self.dp
        H[2, 2] = -self.delta - self.u / 2 + self.dp
        H[3, 3] = self.delta - self.u / 2

        # Intralayer hoppings
        H[0, 1] = self.v0 * pi_dag   # A1-B1
        H[1, 0] = self.v0 * pi
        H[2, 3] = self.v0 * pi_dag   # A2-B2
        H[3, 2] = self.v0 * pi

        # gamma4 interlayer
        if self.gamma4 != 0.0:
            H[0, 2] = self.v4 * pi_dag   # A1-A2
            H[2, 0] = self.v4 * pi
            H[1, 3] = self.v4 * pi_dag   # B1-B2
            H[3, 1] = self.v4 * pi

        # gamma1 (dimer)
        H[1, 2] = self.gamma1            # B1-A2
        H[2, 1] = self.gamma1

        # gamma3 (skew interlayer)
        if self.gamma3 != 0.0:
            H[0, 3] = self.v3 * pi       # A1-B2
            H[3, 0] = self.v3 * pi_dag

        return H

    def solve(self, k: np.ndarray) -> tuple:
        """
        对角化哈密顿量，返回 (energies, states)。

        Returns
        -------
        energies : np.ndarray, shape (4,)
            升序排列
        states : np.ndarray, shape (4, 4)
            列向量为本征态，已固定相位规范。
        """
        H = self.hamiltonian(k)
        E, V = np.linalg.eigh(H)
        for n in range(V.shape[1]):
            idx = np.argmax(np.abs(V[:, n]))
            phase = np.angle(V[idx, n])
            V[:, n] *= np.exp(-1j * phase)
        return E, V

    def _compute_bands_batch(self, kvec: np.ndarray) -> tuple:
        """批量计算 k 点的本征值和本征态。"""
        Nk = len(kvec)
        E = np.zeros((Nk, self.n_bands))
        V = np.zeros((Nk, self.n_orbitals, self.n_bands), dtype=complex)
        for i in range(Nk):
            Ei, Vi = self.solve(kvec[i])
            E[i] = Ei
            V[i] = Vi
        return E, V

    def high_symmetry_points(self) -> dict:
        """返回 BZ 高对称点坐标."""
        a = 1.0  # 无量纲
        return {
            'Γ': np.zeros(2),
            'K': np.array([4 * np.pi / (3 * a), 0]),
            "K'": np.array([-4 * np.pi / (3 * a), 0]),
            'M': np.array([np.pi / a, np.pi / (np.sqrt(3) * a)]),
        }

    def check_hermitian(self, k: np.ndarray, tol: float = 1e-10) -> bool:
        H = self.hamiltonian(k)
        return np.max(np.abs(H - H.conj().T)) < tol


# ============================================================
#  BilayerGrapheneKPAA — AA 堆叠双层石墨烯 k·p 模型
# ============================================================

class BilayerGrapheneKPAA(HamiltonianModel):
    """
    AA 堆叠双层石墨烯 k·p 模型。

    基矢顺序: [A1, B1, A2, B2]
    AA 堆叠：层间耦合发生在相同子晶格之间 (A1↔A2, B1↔B2)。

    Parameters
    ----------
    t : float
        层内最近邻 γ₀ (eV). Default 2.78.
    gamma_aa : float
        AA 层间耦合 (eV). Default 0.2.
    u : float
        层间偏压 (eV). Default 0.0.
    delta : float
        在位能差 Δ (eV). Default 0.0.
    valley : int
        +1 for K, -1 for K'. Default +1.
    """
    n_bands = 4
    n_orbitals = 4
    n_valleys = 2
    n_spins = 1
    valley_degeneracy = 1
    spin_degeneracy = 2

    def __init__(
        self,
        t: float = 2.78,
        gamma_aa: float = 0.2,
        u: float = 0.0,
        delta: float = 0.0,
        valley: int = +1,
    ):
        self.t = t
        self.gamma_aa = gamma_aa
        self.u = u
        self.delta = delta
        self.xi = valley

        self.v0 = np.sqrt(3) / 2 * t

        self.K_point = np.array([4 * np.pi / 3, 0])
        self.Kp_point = np.array([-4 * np.pi / 3, 0])

        self.lattice_vectors = _A.copy()
        self.reciprocal_vectors = _B.copy()

    def hamiltonian(self, k: np.ndarray) -> np.ndarray:
        """
        构造 4×4 AA 堆叠 k·p 哈密顿量（厄米矩阵）。

        H_AA = [  Δ+u/2   v0*π†    γ_aa      0   ]
               [  v0*π   -Δ+u/2      0     γ_aa ]
               [  γ_aa      0     Δ-u/2    v0*π† ]
               [    0     γ_aa     v0*π   -Δ-u/2 ]
        """
        valley_point = self.K_point if self.xi == +1 else self.Kp_point
        q = k - valley_point
        pi = self.xi * q[0] + 1j * q[1]      # π  = xi*kx + i*ky
        pi_dag = self.xi * q[0] - 1j * q[1]  # π† = xi*kx - i*ky

        H = np.zeros((4, 4), dtype=complex)

        # Diagonal (onsite)
        H[0, 0] = self.delta + self.u / 2
        H[1, 1] = -self.delta + self.u / 2
        H[2, 2] = self.delta - self.u / 2
        H[3, 3] = -self.delta - self.u / 2

        # Intralayer hoppings
        H[0, 1] = self.v0 * pi_dag   # A1-B1
        H[1, 0] = self.v0 * pi
        H[2, 3] = self.v0 * pi_dag   # A2-B2
        H[3, 2] = self.v0 * pi

        # Interlayer (AA coupling)
        H[0, 2] = self.gamma_aa      # A1-A2
        H[2, 0] = self.gamma_aa
        H[1, 3] = self.gamma_aa      # B1-B2
        H[3, 1] = self.gamma_aa

        return H

    def solve(self, k: np.ndarray) -> tuple:
        """
        对角化哈密顿量，返回 (energies, states)。

        Returns
        -------
        energies : np.ndarray, shape (4,)
            升序排列
        states : np.ndarray, shape (4, 4)
            列向量为本征态，已固定相位规范。
        """
        H = self.hamiltonian(k)
        E, V = np.linalg.eigh(H)
        for n in range(V.shape[1]):
            idx = np.argmax(np.abs(V[:, n]))
            phase = np.angle(V[idx, n])
            V[:, n] *= np.exp(-1j * phase)
        return E, V

    def _compute_bands_batch(self, kvec: np.ndarray) -> tuple:
        """批量计算 k 点的本征值和本征态。"""
        Nk = len(kvec)
        E = np.zeros((Nk, self.n_bands))
        V = np.zeros((Nk, self.n_orbitals, self.n_bands), dtype=complex)
        for i in range(Nk):
            Ei, Vi = self.solve(kvec[i])
            E[i] = Ei
            V[i] = Vi
        return E, V

    def high_symmetry_points(self) -> dict:
        """返回 BZ 高对称点坐标."""
        a = 1.0  # 无量纲
        return {
            'Γ': np.zeros(2),
            'K': np.array([4 * np.pi / (3 * a), 0]),
            "K'": np.array([-4 * np.pi / (3 * a), 0]),
            'M': np.array([np.pi / a, np.pi / (np.sqrt(3) * a)]),
        }

    def check_hermitian(self, k: np.ndarray, tol: float = 1e-10) -> bool:
        H = self.hamiltonian(k)
        return np.max(np.abs(H - H.conj().T)) < tol
