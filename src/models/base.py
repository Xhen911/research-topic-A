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
