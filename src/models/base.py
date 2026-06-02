import numpy as np
from abc import ABC, abstractmethod

# ============================================================
#  哈密顿量模型抽象接口（遵循接口规范）
# ============================================================

class HamiltonianModel(ABC):
    """
    所有模型的抽象基类。
    响应函数求解器只与这个接口交互，不关心底层是 kp 还是 TB。
    """
    n_bands: int
    n_orbitals: int
    n_valleys: int
    n_spins: int
    valley_degeneracy: int
    spin_degeneracy: int
    lattice_vectors: np.ndarray
    reciprocal_vectors: np.ndarray

    @abstractmethod
    def hamiltonian(self, k: np.ndarray) -> np.ndarray:
        """k: shape (2,)，返回 shape (n_orbitals, n_orbitals) 厄米矩阵"""
        ...

    @abstractmethod
    def solve(self, k: np.ndarray) -> tuple:
        """k: shape (2,)，返回 (energies, states)
        energies: shape (n_bands,)，升序，单位 eV
        states: shape (n_orbitals, n_bands)，列向量为本征态
        """
        ...

    def degeneracy_factor(self) -> int:
        return self.valley_degeneracy * self.spin_degeneracy

    # ── 速度算符 ─────────────────────────────────────────
    def velocity_operator(self, k: np.ndarray, dk: float = 1e-4) -> tuple:
        """
        速度算符 v_α = ∂H/∂k_α，在轨道基下。
        默认实现：中心有限差分。
        子类应 override 为解析求导以提高精度和效率。

        Parameters
        ----------
        k : np.ndarray, shape (2,)
            波矢（模型约定单位）。
        dk : float
            有限差分步长。默认 1e-4。

        Returns
        -------
        v_x, v_y : tuple of np.ndarray
            各 shape (n_orbitals, n_orbitals)，dtype=complex。
        """
        H_plus_x = self.hamiltonian(k + np.array([dk, 0.0]))
        H_minus_x = self.hamiltonian(k - np.array([dk, 0.0]))
        H_plus_y = self.hamiltonian(k + np.array([0.0, dk]))
        H_minus_y = self.hamiltonian(k - np.array([0.0, dk]))

        v_x = (H_plus_x - H_minus_x) / (2.0 * dk)
        v_y = (H_plus_y - H_minus_y) / (2.0 * dk)

        return v_x, v_y
