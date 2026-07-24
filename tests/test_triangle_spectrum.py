"""Unit tests for the batched Lehmann-Taut triangle-spectrum primitive.

These tests need no Hamiltonian model — they exercise triangle_spectrum
directly with synthetic vertex fields, following the self-check suggestions
in docs/三角形谱计算优化.md.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.propagators.triangle_core import triangle_spectrum, _triangles_for_kmesh
from src.propagators.dos import _triangle_dos_exact  # legacy scalar reference

_trapz = getattr(np, "trapz", np.trapezoid)


# ── T0.1  constant energy surface (fully degenerate) ───────────────
def test_constant_surface_sum_rule():
    Nv = 100
    nF = 2
    vf = np.zeros((Nv, nF))
    tri = _triangles_for_kmesh(10)
    E = np.linspace(-1.0, 1.0, 2001)
    dos = triangle_spectrum(vf, E, weights=None, triangles=tri,
                            area_BZ=1.0, prefactor=1.0, enforce_sum_rule=True)
    assert dos.shape == (E.size, nF)
    assert np.all(np.isfinite(dos))
    for f in range(nF):
        integral = float(_trapz(dos[:, f], E))
        assert abs(integral - 1.0) < 1e-6, f"field {f}: integral={integral}"


# ── T0.2  random weights satisfy sum rule ───────────────────────────
def test_random_weights_sum_rule():
    rng = np.random.default_rng(0)
    Nv = 144
    nF = 3
    vf = rng.standard_normal((Nv, nF)) * 0.3
    weights = rng.random((Nv, nF)) + 0.5
    tri = _triangles_for_kmesh(12)
    E = np.linspace(-2.0, 2.0, 3001)
    dos = triangle_spectrum(vf, E, weights=weights, triangles=tri,
                            area_BZ=1.0, prefactor=1.0, enforce_sum_rule=True)
    assert dos.shape == (E.size, nF)
    for f in range(nF):
        integral = float(_trapz(dos[:, f], E))
        target = float(weights[:, f].mean())
        assert np.allclose(integral, target, rtol=1e-8, atol=1e-10), \
            f"field {f}: integral={integral} target={target}"


# ── T0.3  shape contract for both input layouts ─────────────────────
def test_shape_contract():
    rng = np.random.default_rng(1)
    nk = 6
    tri = _triangles_for_kmesh(nk)
    nF = 4
    Nt = tri.shape[0]
    E = np.linspace(-1.0, 1.0, 500)

    # (Nv, nF) layout
    vf2 = rng.standard_normal((nk * nk, nF)) * 0.2
    s2 = triangle_spectrum(vf2, E, triangles=tri, area_BZ=1.0, prefactor=1.0)
    assert s2.shape == (E.size, nF)

    # (Nt, 3, nF) layout
    vf3 = rng.standard_normal((Nt, 3, nF)) * 0.2
    s3 = triangle_spectrum(vf3, E, area_BZ=1.0, prefactor=1.0)
    assert s3.shape == (E.size, nF)
    assert s3.dtype == float
    assert np.all(np.isfinite(s3))


# ── T0.4  E_grid validation ─────────────────────────────────────────
def test_egrid_validation():
    vf = np.zeros((100, 1))
    tri = _triangles_for_kmesh(10)
    with pytest.raises(ValueError):
        triangle_spectrum(vf, np.array([[1.0, 2.0]]), triangles=tri)  # 2D
    with pytest.raises(ValueError):
        triangle_spectrum(vf, np.array([3.0, 1.0, 2.0]), triangles=tri)  # not increasing
    with pytest.raises(ValueError):
        triangle_spectrum(vf, np.array([1.0, 2.0, 2.0]), triangles=tri)  # flat


# ── T0.5  degenerate branches: finite + conserve weight ─────────────
def test_degenerate_branches():
    nk = 5
    tri = _triangles_for_kmesh(nk)
    Nt = tri.shape[0]
    E = np.linspace(-1.0, 1.0, 4001)

    # Two-equal groups + one fully-degenerate group across triangles.
    vf = np.zeros((Nt, 3, 1))
    # group A: e1=e2 < e3  (a->b limit)
    vf[0::3, 0, 0] = -0.5
    vf[0::3, 1, 0] = -0.5
    vf[0::3, 2, 0] = 0.6
    # group B: e1 < e2=e3  (b->c limit)
    vf[1::3, 0, 0] = -0.6
    vf[1::3, 1, 0] = 0.5
    vf[1::3, 2, 0] = 0.5
    # group C: all equal  (Gaussian replacement)
    vf[2::3, :, 0] = 0.0

    dos = triangle_spectrum(vf, E, area_BZ=1.0, prefactor=1.0,
                            enforce_sum_rule=True)
    assert np.all(np.isfinite(dos))
    integral = float(_trapz(dos[:, 0], E))
    # sum rule: area_BZ * mean(weight) = 1 * 1 = 1
    assert abs(integral - 1.0) < 2e-2, f"degenerate integral={integral}"


# ── T0.6  per-triangle equivalence with legacy _triangle_dos_exact ──
def test_matches_legacy_per_triangle():
    """With enforce_sum_rule=False and well-separated energies (no
    near-degeneracy), the primitive must reproduce the legacy scalar
    `_triangle_dos_exact` summed over triangles (both use the identical
    generic piecewise-linear formula)."""
    rng = np.random.default_rng(7)
    nk = 8
    nE = 1000
    tri = _triangles_for_kmesh(nk)
    Nt = tri.shape[0]
    E = np.linspace(-3.0, 3.0, nE)

    # Per-vertex energies.  Draw Nv random values, then split into
    # well-separated clusters of three (sorted) so each triangle's three
    # vertices are ordered and non-degenerate (keeps legacy in generic branch).
    vE = np.zeros(nk * nk)
    for v in range(nk * nk):
        x = np.sort(rng.standard_normal(3) * 2.0)
        if (x[1] - x[0]) < 1e-2 or (x[2] - x[1]) < 1e-2:
            x = x + np.array([0.0, 0.03, 0.06])
        # pick one representative vertex energy for this mesh vertex
        vE[v] = x[1]

    area_tri = 1.0 / Nt  # area_BZ = 1, Nt triangles

    # legacy scalar sum — _triangle_dos_exact already applies `area_tri`
    legacy = np.zeros(nE)
    for i1, i2, i3 in tri:
        legacy += _triangle_dos_exact(vE[i1], vE[i2], vE[i3], E, area_tri)

    # primitive, (Nt, 3, 1) layout: gather the three vertex energies
    vf3 = vE[tri][:, :, None]  # (Nt, 3, 1)
    spec = triangle_spectrum(vf3, E, area_BZ=1.0, prefactor=1.0,
                             enforce_sum_rule=False)
    prim = spec[:, 0]

    np.testing.assert_allclose(prim, legacy, rtol=1e-8, atol=1e-10)
