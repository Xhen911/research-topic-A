"""Electron-phonon coupling kernel.

Skeleton only.  Will implement the e-ph self-energy and phonon-mediated
interaction for moire systems.
"""

from .kernel import InteractionKernel
import numpy as np


class EPHKernel(InteractionKernel):
    """Electron-phonon interaction kernel (not yet implemented)."""

    @property
    def name(self) -> str:
        return 'e-ph'

    def screen(self, pi0: np.ndarray, q_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError("e-ph kernel not yet implemented")
