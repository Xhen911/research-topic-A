"""TDDFT kernel — time-dependent density functional theory.

Skeleton only.  Will implement the TDDFT kernel f_xc(q, w) that modifies
the bare polarisation: Pi_TDDFT = Pi_0 / (1 - (V_q + f_xc) * Pi_0).
"""

from .kernel import InteractionKernel
import numpy as np


class TDDFTKernel(InteractionKernel):
    """TDDFT interaction kernel (not yet implemented)."""

    @property
    def name(self) -> str:
        return 'TDDFT'

    def screen(self, pi0: np.ndarray, q_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TDDFT kernel not yet implemented")
