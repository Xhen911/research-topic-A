"""Grid and k-path utilities for band structure calculations.

Moved from utils/kpath.py to core/grids.py in PR-6 (layered architecture
cleanup).  `utils/` is now empty and removed.
"""
import numpy as np


def make_k_path(kpts, nkdensity=100):
    """
    Build a dense k-point path connecting a sequence of high-symmetry points.

    Parameters
    ----------
    kpts : list of np.ndarray
        Sequence of waypoint k-vectors, each of shape (2,).
    nkdensity : int, optional
        Approximate number of k-points per unit reciprocal length (default 100).

    Returns
    -------
    np.ndarray, shape (Nk, 2)
        Dense k-point path with smoothly interpolated segments.
    """
    seglen = [np.linalg.norm(kpts[i + 1] - kpts[i]) for i in range(len(kpts) - 1)]
    kpath_sector = []
    for i in range(len(kpts) - 1):
        npoints = max(2, int(round(nkdensity * seglen[i])))
        seg_pts = [(1 - t) * kpts[i] + t * kpts[i + 1] for t in np.linspace(0, 1, npoints)]
        if i > 0:
            seg_pts = seg_pts[1:]  # 避免重复点
        kpath_sector.extend(seg_pts)

    return np.array(kpath_sector)
