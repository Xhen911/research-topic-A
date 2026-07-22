"""BSE kernel — Bethe-Salpeter equation for excitonic effects.

Skeleton only.  Will implement the electron-hole ladder kernel
that goes beyond the independent-particle approximation.
"""

from .kernel import InteractionKernel
import numpy as np


class BSEKernel(InteractionKernel):
    """Bethe-Salpeter interaction kernel (not yet implemented)."""

    @property
    def name(self) -> str:
        return 'BSE'

    def screen(self, pi0: np.ndarray, q_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError("BSE kernel not yet implemented")
