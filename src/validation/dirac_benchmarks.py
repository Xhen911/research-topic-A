"""
dirac_benchmarks.py — analytical Dirac DOS/JDOS/O-JDOS benchmarks.
===================================================================

Reference formulas for the 2D massless Dirac cone (g = g_s·g_v = 4).
Used to validate numerical DOS/JDOS implementations.

Moved from src/propagators/dos.py during dos.py slimming (2026-07-23).
"""
import numpy as np


def dirac_dos_analytical(
    E_values: np.ndarray,
    vF: float,
    g: int = 4,
    vol_BZ: float = 1.0,
) -> np.ndarray:
    """Analytical DOS for a 2D Dirac system.

    DOS(E) = g·|E| / (2π vF²)

    Valid for |E| ≪ bandwidth (Dirac cone regime only).

    Parameters
    ----------
    E_values : np.ndarray  — energy grid (eV)
    vF : float             — Fermi velocity (eV·Å or model dimensionless)
    g : int                — degeneracy (spin × valley). Default 4.
    vol_BZ : float         — BZ area |det(R)| (for dimensional consistency only)

    Returns
    -------
    dos : np.ndarray
    """
    return g * np.abs(E_values) / (2 * np.pi * vF ** 2)


def dirac_jdos_analytical(
    w_values: np.ndarray,
    vF: float,
    g: int = 4,
) -> np.ndarray:
    """Analytical interband JDOS for a 2D Dirac cone.

    JDOS(ω) = g·ω / (8π vF²)

    Derivation:
      ΔE = 2vF|k|,  d(ΔE) = 2vF d|k|
      JDOS = g/(2π)² · 2π∫k dk · 2 · δ(ω − 2vF k)   [×2 for m↔n symmetry]
           = g/(2π) · ∫k dk δ(ω − 2vF k)
           = g/(2π) · (ω/(2vF)) · (1/(2vF))
           = g·ω / (8π vF²)
    """
    return g * w_values / (8 * np.pi * vF ** 2)


def dirac_optical_jdos_analytical(
    w_values: np.ndarray,
    vF: float,
    g: int = 4,
) -> np.ndarray:
    """Analytical optical JDOS for a 2D Dirac cone (isotropic average).

    O-JDOS(ω) = g·ω / (16π)

    Derivation:
      |v_{+-}|² = vF² (isotropic)
      O-JDOS = (1/2) Σ_α |v^α|² · JDOS_unweighted
             = vF² · JDOS  [after averaging over x,y directions]
    """
    return g * w_values / (16 * np.pi)
