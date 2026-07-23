"""
vhs_analysis.py — Van Hove singularity detection on DOS(E).
=============================================================

Moved from src/propagators/dos.py during dos.py slimming (2026-07-23).
These are post-processing analysis utilities, not propagator computations.
"""
import numpy as np


def find_vhs_peaks(E, dos, prominence=None, height=None, distance=None):
    """Locate VHS energies via peak-finding on DOS(E).

    Uses scipy.signal.find_peaks with optional prominence/height
    thresholds to reject noise.

    Returns list of VHS energies [eV].
    """
    from scipy.signal import find_peaks
    if prominence is None:
        prominence = 0.02 * (dos.max() - dos.min())
    if height is None:
        height = dos.min() + 0.05 * (dos.max() - dos.min())
    if distance is None:
        distance = max(1, len(E) // 50)
    peaks, _ = find_peaks(dos, prominence=prominence,
                           height=height, distance=distance)
    return [float(E[p]) for p in peaks]


def find_vhs_derivative(E, dos):
    """Locate VHS via dDOS/dE zero-crossings (sign + -> -).

    Returns list of dicts with keys 'E_vhs' and 'dos'.
    """
    d_dos = np.gradient(dos, E)
    results = []
    for i in range(1, len(d_dos)):
        if d_dos[i - 1] > 0 and d_dos[i] <= 0:
            frac = d_dos[i - 1] / (d_dos[i - 1] - d_dos[i])
            E_cross = E[i - 1] + frac * (E[i] - E[i - 1])
            results.append({
                'E_vhs': float(E_cross),
                'dos': float(np.interp(E_cross, E, dos)),
            })
    return results
