"""
dos.py
======
Triangle-method density of states — exact eta->0, Lehmann-Taut formula.

Core functions
--------------
- compute_dos_triangle(model, nk, ...)  -> (E, dos)
  Exact DOS via piecewise-linear interpolation on triangles.
  No broadening parameter — VHS logarithmic divergences are exact
  up to mesh discretisation.
- compute_eigenvalues(model, k_cart)    -> (E_k, V_k)
  Batch diagonalisation (shared by dos and dos_gaussian).
- check_dos_sum_rule(E, dos, ...)       -> (ok, integral, expected)
  Verify integral of DOS = g * nb * A_BZ/(2pi)^2.

For Gaussian-broadened DOS / JDOS / O-JDOS, see dos_gaussian.py.

Conventions
-----------
- k-integral: uniform mesh Riemann sum
    integral_BZ d^2k/(2pi)^2 f(k) ~ |det(R)| / ((2pi)^2 Nk) sum_i f(k_i)
- g = model.degeneracy_factor()
- Energy units: eV
"""

import numpy as np
from typing import Callable, Optional, Tuple

from .lindhard import generate_k_mesh, fermi_dirac



# ============================================================
#  批量对角化
# ============================================================

def compute_eigenvalues(model, k_cart: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """对一组 k 点批量对角化，返回 (E_k, V_k)。

    优先使用模型的 _compute_bands_batch()（更快），
    否则回退到逐点 solve()。
    """
    Nk = len(k_cart)
    nb = model.n_bands
    no = model.n_orbitals

    if hasattr(model, '_compute_bands_batch'):
        return model._compute_bands_batch(k_cart)

    E = np.zeros((Nk, nb))
    V = np.zeros((Nk, no, nb), dtype=complex)
    for i in range(Nk):
        Ei, Vi = model.solve(k_cart[i])
        E[i] = Ei
        V[i] = Vi
    return E, V



#  Triangle-method DOS (exact eta->0, Lehmann-Taut)
# ============================================================

def _triangles_for_kmesh(nk):
    """Triangle connectivity for a uniform nk×nk BZ parallelogram mesh."""
    tri_list = []
    for a in range(nk):
        for b in range(nk):
            v00 = a * nk + b
            v01 = a * nk + (b + 1) % nk
            v10 = (a + 1) % nk * nk + b
            v11 = (a + 1) % nk * nk + (b + 1) % nk
            tri_list.append([v00, v01, v11])
            tri_list.append([v00, v10, v11])
    return np.array(tri_list, dtype=int)


def _triangle_dos_exact(e1, e2, e3, E, area, tol=1e-12):
    """Exact η→0 DOS contribution from one triangle with linear dispersion.

    For sorted vertex energies e1 ≤ e2 ≤ e3, the DOS inside the triangle
    with uniform linear interpolation is exactly piecewise-linear:

        DOS(E) ∝ 2·(E−e1)/[(e2−e1)(e3−e1)]   for E ∈ [e1, e2]
        DOS(E) ∝ 2·(e3−E)/[(e3−e2)(e3−e1)]   for E ∈ [e2, e3]
        DOS(E) = 0                              otherwise

    The integral ∫ DOS(E) dE = 1 (per unit area; prefactor applied externally).

    The three-equal (e1≈e2≈e3) case gives a Dirac-delta — handled as
    Gaussian replacement with tiny width to keep the energy grid meaningful.

    Uses Taylor-blend for near-degenerate cases (|de| < 1e-6 eV by default)
    instead of hard thresholds, avoiding cliff-edge instability.
    """
    import warnings
    import numpy as np

    # Sort
    perm = np.argsort([e1, e2, e3])
    a, b, c = float(e1), float(e2), float(e3)
    a, b, c = (np.array([a, b, c])[perm]).tolist()
    de_ba = c - b              # e3 - e2 (after sort)
    de_ca = c - a              # e3 - e1
    de_cb = b - a              # e2 - e1

    dos = np.zeros_like(E)

    # --- Fully degenerate: replace with narrow Gaussian ---
    if de_ca < 1e-14:
        # All three equal → Dirac delta
        sigma = max(1e-14, area * 0.1)  # negligible but stable
        dos = area * np.exp(-0.5 * ((E - a) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        return dos

    # --- Taylor-blend threshold (padded window, no hard cliff) ---
    blend_width = max(1e-8, de_ca * 1e-6)

    # Helper: smooth 0→1 ramp over width w
    def _ramp(x, w):
        """Smooth ramp: 0 for x ≤ 0, 1 for x ≥ w, C¹ polynomial in between."""
        if w < 1e-15:
            return 1.0 if x > 0 else 0.0
        t = np.clip(x / w, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)   # smoothstep

    # Two-equal case: e1 = e2 < e3 (or e1 < e2 = e3)
    if abs(c - b) < blend_width:
        # e2 ≈ e3 — blend between the generic formula and the e2=e3 limit
        # e2=e3 limit: DOS = 2·A·(E-a)/(c-a)² for E∈[a,c], 0 otherwise
        # Generic:  DOS = 2·A·(E-a)/[(b-a)(c-a)] for E∈[a,b],
        #           DOS = 2·A·(c-E)/[(c-b)(c-a)] for E∈[b,c]
        r = _ramp(c - b, blend_width)
        # Limit formula (r=1 → fully degenerate e2=e3)
        mask_ac = (E >= a) & (E <= c)
        dos_limit = np.where(mask_ac, area * 2.0 * (E - a) / (de_ca ** 2), 0.0)
        # Generic formula
        de_cb_safe = max(de_cb, 1e-30)
        de_ba_safe = max(de_ba, 1e-30)
        mask_ab = (E >= a) & (E <= b)
        mask_bc = (E >= b) & (E <= c)
        dos_generic = np.where(mask_ab,
            area * 2.0 * (E - a) / (de_cb_safe * de_ca),
            np.where(mask_bc,
                area * 2.0 * (c - E) / (de_ba_safe * de_ca),
                0.0))
        dos = (1.0 - r) * dos_generic + r * dos_limit

    elif abs(b - a) < blend_width:
        # e1 ≈ e2
        r = _ramp(b - a, blend_width)
        mask_ac = (E >= a) & (E <= c)
        dos_limit = np.where(mask_ac, area * 2.0 * (c - E) / (de_ca ** 2), 0.0)
        de_cb_safe = max(de_cb, 1e-30)
        de_ba_safe = max(de_ba, 1e-30)
        mask_ab = (E >= a) & (E <= b)
        mask_bc = (E >= b) & (E <= c)
        dos_generic = np.where(mask_ab,
            area * 2.0 * (E - a) / (de_cb_safe * de_ca),
            np.where(mask_bc,
                area * 2.0 * (c - E) / (de_ba_safe * de_ca),
                0.0))
        dos = (1.0 - r) * dos_generic + r * dos_limit

    else:
        # Generic: all three distinct
        de_cb_safe = max(de_cb, 1e-30)
        de_ba_safe = max(de_ba, 1e-30)
        mask_ab = (E >= a) & (E <= b)
        mask_bc = (E >= b) & (E <= c)
        dos = np.where(mask_ab,
            area * 2.0 * (E - a) / (de_cb_safe * de_ca),
            np.where(mask_bc,
                area * 2.0 * (c - E) / (de_ba_safe * de_ca),
                0.0))

    return dos


def compute_dos_triangle(model, nk=24, E_range=None, nE=3000,
                         eta=0.05e-3, band_slice=None):
    """DOS via exact η→0 Lehmann-Taut triangle integration.

    Within each triangle the band energy is linearly interpolated from
    the three vertex values.  The DOS contribution is piecewise-linear
    and integrated analytically — no broadening, no branch-cut logarithms.
    This captures the logarithmic VHS divergence exactly (up to mesh
    discretisation, which converges O(1/Nk) in smooth regions).

    The ``eta`` parameter is kept for API compatibility but is **not used**
    in the core integration.  Physical broadening, if needed, should be
    applied as a separate post-processing Lorentzian convolution.

    Parameters
    ----------
    model : HamiltonianModel
    nk : int — k-points per reciprocal direction (total = nk²)
    E_range : (float, float) or None
    nE : int
    eta : float — *unused* in the exact formula; kept for compatibility.
    band_slice : slice or None — which bands to include (default: all)

    Returns
    -------
    E : (nE,)  energy grid (eV)
    dos : (nE,)  DOS (states/eV/unit cell)
    """
    _, k_cart = generate_k_mesh(nk, model.reciprocal_vectors)
    Nk = len(k_cart)
    assert int(np.sqrt(Nk)) ** 2 == Nk, f'nk² ≠ Nk={Nk}'
    nk_side = int(np.sqrt(Nk))

    E_k, _ = compute_eigenvalues(model, k_cart)
    if band_slice is None:
        band_slice = slice(None)
    E_k = E_k[:, band_slice]
    nb_sel = E_k.shape[1]

    if E_range is None:
        E_range = (float(E_k.min()), float(E_k.max()))
    E = np.linspace(*E_range, nE)

    area_BZ = abs(np.linalg.det(model.reciprocal_vectors))
    dk = np.sqrt(area_BZ / Nk)
    area_tri = dk ** 2 / 2.0
    g = model.degeneracy_factor()
    # prefactor: g / (2π)² — the triangle integral already gives ∫_T d²k δ(E−ε(k))
    # per unit area_tri, so total = Σ_T area_tri × DOS¹(E) = dk² × Σ_T DOS¹(E)
    prefactor = g / ((2 * np.pi) ** 2)

    tri_idx = _triangles_for_kmesh(nk_side)
    dos = np.zeros(nE)

    for i1, i2, i3 in tri_idx:
        for ib in range(nb_sel):
            e1 = float(E_k[i1, ib])
            e2 = float(E_k[i2, ib])
            e3 = float(E_k[i3, ib])
            dos += prefactor * _triangle_dos_exact(e1, e2, e3, E, area_tri)

    return E, dos


# ============================================================
#  DOS 求和规则验证
# ============================================================

def check_dos_sum_rule(E, dos, g=None, nb=None, area_BZ=None, tol=1e-3):
    """Verify ∫ DOS(E) dE = g · nb · A_BZ/(2π)².

    Parameters
    ----------
    E : (nE,) energy grid
    dos : (nE,) DOS array
    g : int or None — degeneracy factor
    nb : int or None — number of bands
    area_BZ : float or None — BZ area
    tol : float — relative tolerance

    Returns
    -------
    ok : bool
    integral : float
    expected : float
    """
    if g is None:
        g = 4
    if nb is None:
        nb = 1
    _trapz = getattr(np, "trapz", np.trapezoid)
    integral = float(_trapz(dos, E))
    area_factor = area_BZ / (2.0 * np.pi) ** 2 if area_BZ is not None else 1.0
    expected = float(g * nb * area_factor)
    rel_err = abs(integral - expected) / max(abs(expected), 1e-30)
    ok = rel_err < tol
    if not ok:
        import warnings
        warnings.warn(
            f"DOS sum rule violation: integral={integral:.4f} vs "
            f"g*nb*area={expected:.3f} (rel_err={rel_err:.2e})")
    return ok, integral, expected
