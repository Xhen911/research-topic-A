"""
InteractionKernel — abstract base for many-body interaction kernels.

Unifies the interface across RPA, TDDFT, BSE, Hubbard, and e-ph
so that observable modules can call ``kernel.screen(pi0, q)`` without
knowing which approximation is in use.

RPA is the first (and currently only) concrete implementation.
The others are stubs that raise NotImplementedError.
"""

from abc import ABC, abstractmethod
import numpy as np


class InteractionKernel(ABC):
    """Abstract interaction kernel.

    Subclasses must implement :meth:`screen`, which takes the bare
    polarisation Pi_0 and returns the dressed / screened response.
    """

    @abstractmethod
    def screen(self, pi0: np.ndarray, q_values: np.ndarray) -> np.ndarray:
        """Return the dressed polarisation given Pi_0.

        Parameters
        ----------
        pi0 : np.ndarray, shape (nw, nq), complex
            Bare Lindhard polarisation.
        q_values : np.ndarray, shape (nq,)
            Momentum transfer magnitudes.

        Returns
        -------
        np.ndarray, shape (nw, nq), complex
            Dressed polarisation.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier (e.g. 'RPA', 'TDDFT')."""
        ...
