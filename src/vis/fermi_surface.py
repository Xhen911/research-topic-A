"""
fermi_surface.py — Fermi surface topology at Van Hove singularities
==================================================================

Two-panel visualisation:
  Left:  3D band-energy surface E(k_x, k_y) with a horizontal plane
         cutting through at the VHS saddle-point energy.
  Right: 2D constant-energy contours showing the Fermi-surface
         topology change (Lifshitz transition) near the VHS.

Works with any HamiltonianModel (no cached eig data needed).

Usage
-----
    from src.vis.fermi_surface import plot_fermi_surface_vhs
    fig, (ax3d, ax2d) = plot_fermi_surface_vhs(model, theta=1.05)

Author: ported from viz_fermi_surface_vhs.py, 2026-07-15
"""

import numpy as np
from ..response.polarization import generate_k_mesh, fermi_dirac


def plot_fermi_surface_vhs(
    model,
    band_idx=None,
    E_vhs=None,
    vhs_side='electron',
    nk=24,
    delta_e_range=None,
    n_contours=5,
    azimuth=25, elevation=35,
    figsize=(13, 5.2),
    E_k=None, k_cart=None,
):
    """Fermi-surface topology at the Van Hove singularity.

    Parameters
    ----------
    model : HamiltonianModel
        Any model implementing solve(k).
    band_idx : int or None
        Flat-band index. Default: conduction flat band for electron VHS,
        valence flat band for hole VHS (-1 from n_bands//2).
    E_vhs : float or None
        VHS energy (eV). Auto-detected via triangle DOS if None.
    vhs_side : 'electron' | 'hole'
        Which side of CNP to search.
    nk : int
        k-points per direction (total nk²).
    delta_e_range : (float, float) or None
        Energy range around E_vhs for contours.
    n_contours : int
        Number of contour levels.
    azimuth, elevation : float
        3D view angles (degrees).
    figsize : tuple
    E_k, k_cart : optional precomputed eigenvalues and k-mesh.

    Returns
    -------
    fig, (ax3d, ax2d)
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    # ── 1. Diagonalize on k-mesh ───────────────────────────
    if k_cart is None:
        _, k_cart = generate_k_mesh(nk, model.reciprocal_vectors)
    if E_k is None:
        from ..response.dos import compute_eigenvalues
        E_k, _ = compute_eigenvalues(model, k_cart)
    
    nk_side = int(np.sqrt(len(k_cart)))
    nb = model.n_bands
    half = nb // 2
    
    # Choose band
    if band_idx is None:
        band_idx = half if vhs_side == 'electron' else half - 1
    E_band = E_k[:, band_idx]
    
    # Reshape to 2D
    kx_2d = k_cart[:, 0].reshape(nk_side, nk_side)
    ky_2d = k_cart[:, 1].reshape(nk_side, nk_side)
    E_2d = E_band.reshape(nk_side, nk_side)
    
    # ── 2. VHS energy ──────────────────────────────────────
    if E_vhs is None:
        try:
            from .dos import compute_dos_triangle, find_vhs_peaks, compute_cnp
            kBT = 0.1e-3
            E_dos, dos = compute_dos_triangle(model, nk=nk_side,
                                              band_slice=slice(band_idx, band_idx+1))
            peaks = find_vhs_peaks(E_dos, dos)
            E_cnp = compute_cnp(E_k)
            if vhs_side == 'hole':
                candidates = [p for p in peaks if p < E_cnp]
            else:
                candidates = [p for p in peaks if p >= E_cnp]
            E_vhs = candidates[0] if candidates else E_cnp
        except Exception:
            E_cnp = (E_band.min() + E_band.max()) / 2
            E_vhs = E_band.max() if vhs_side == 'electron' else E_band.min()
    else:
        from .dos import compute_cnp
        E_cnp = compute_cnp(E_k)
    
    # ── 3. Contour range ───────────────────────────────────
    if delta_e_range is None:
        spread = max((E_band.max() - E_band.min()) / 4.0, 0.1e-3)
        spread = min(spread, 2.0e-3)
        delta_e_range = (-spread, spread)
    
    # ── 4. Build figure ────────────────────────────────────
    fig = plt.figure(figsize=figsize, layout="constrained")
    
    # Left: 3D surface
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax3d.plot_surface(
        kx_2d, ky_2d, E_2d * 1e3,
        cmap=mpl.colormaps['coolwarm'], alpha=1., linewidth=0,
        antialiased=True, shade=True,
    )
    # Horizontal VHS plane
    xx, yy = np.meshgrid(
        np.linspace(kx_2d.min(), kx_2d.max(), nk_side),
        np.linspace(ky_2d.min(), ky_2d.max(), nk_side),
    )
    zz = np.full_like(xx, E_vhs * 1e3)
    plane = ax3d.plot_surface(xx, yy, zz, color='b', alpha=0.3,
                               linewidth=0, antialiased=True)
    ax3d.set_xlabel(r'$k_x$ ($\mathrm{\AA}^{-1}$)')
    ax3d.set_ylabel(r'$k_y$ ($\mathrm{\AA}^{-1}$)')
    ax3d.set_zlabel(r'$E$ (meV)')
    ax3d.view_init(elev=elevation, azim=azimuth)
    
    # Right: 2D contours
    ax2d = fig.add_subplot(1, 2, 2)
    dE_lo, dE_hi = delta_e_range
    levels = np.linspace(E_vhs + dE_lo, E_vhs + dE_hi, n_contours + 2)
    cs = ax2d.contour(kx_2d, ky_2d, E_2d * 1e3,
                       levels=levels, linewidths=1.0, cmap='plasma')
    ax2d.clabel(cs, fmt='%.1f', fontsize=7)
    ax2d.contour(kx_2d, ky_2d, E_2d * 1e3,
                  levels=[E_vhs * 1e3], colors='red', linewidths=2.0)
    ax2d.set_xlabel(r'$k_x$ ($\mathrm{\AA}^{-1}$)')
    ax2d.set_ylabel(r'$k_y$ ($\mathrm{\AA}^{-1}$)')
    ax2d.set_aspect('equal')
    ax2d.grid(True, alpha=0.3)
    
    # Title
    # Compute ν at VHS
    from .dos import compute_filling
    nu_vhs = compute_filling(E_k, E_vhs, model=model)
    
    theta_str = getattr(model, 'theta', 0)
    ax3d.set_title(f'Band {band_idx} surface
$\theta={theta_str:.2f}^\circ$', fontsize=10)
    ax2d.set_title(f'Fermi contours ($E_{{\rm vhs}}={E_vhs*1e3:+.1f}$ meV)
$\nu={nu_vhs:+.2f}$', fontsize=10)
    fig.suptitle('Van Hove singularity & Fermi-surface topology', fontsize=12, y=1.02)
    
    return fig, (ax3d, ax2d)
