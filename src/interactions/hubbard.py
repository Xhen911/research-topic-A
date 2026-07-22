"""Hubbard kernel — local on-site interaction for flat-band physics.

Skeleton only.  Will implement the Hubbard U correction relevant
for TBG flat bands at charge neutrality.
"""

from .kernel import InteractionKernel
import numpy as np


class HubbardKernel(InteractionKernel):
    """Hubbard interaction kernel (not yet implemented)."""

    @property
    def name(self) -> str:
        return 'Hubbard'

    def screen(self, pi0: np.ndarray, q_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Hubbard kernel not yet implemented")
