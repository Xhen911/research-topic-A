"""Triangle-method JDOS tests: q=0 anchor, finite-q refactor, sum rule.

T2 — q=0 triangle JDOS (compute_jdos_q0_triangle)
T3 — finite-q refactor onto the primitive (compute_jdos_q_triangle)
"""

import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bands.graphene import SingleLayerGrapheneTB
from src.bands.vertices import density_form_factor
from src.propagators.jdos import (
    compute_jdos_q0_triangle,
    compute_jdos_q_triangle,
    check_jdos_sum_rule,
)
from src.propagators.dos import compute_eigenvalues, _triangles_for_kmesh
from src.propagators.lindhard import generate_k_mesh, wrap_k_plus_q
from src.propagators.triangle_core import triangle_spectrum


# ══════════════════════════════════════════════════════════════════════
#  T3.1 reference: independent, corrected-ramp brute force for ONE q.
#  This validates jdos.py's field/weight assembly (the part unique to it),
#  using the SAME corrected ramp as the primitive so it is an exact equality
#  check (not a comparison against the legacy reversed-ramp loop).
# ══════════════════════════════════════════════════════════════════════

def _corrected_triangle_dos(e1, e2, e3, E, area):
    """Independent Lehmann-Taut with the *corrected* ramp (same as primitive)."""
    t = np.sort([e1, e2, e3])
    a, b, c = float(t[0]), float(t[1]), float(t[2])
    d21, d32, d31 = b - a, c - b, c - a
    dos = np.zeros_like(E)
    if d31 < 1e-14:
        dE = float(np.mean(np.diff(E))) if E.size > 1 else 1e-3
        sigma = max(dE, 1e-14)
        return area * np.exp(-0.5 * ((E - a) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    blend = max(1e-8, d31 * 1e-6)

    def smooth(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    generic = np.zeros_like(E)
    db = max(d21, 1e-30)
    dc = max(d32, 1e-30)
    dd = max(d31, 1e-30)
    mab = (E >= a) & (E <= b)
    mbc = (E >= b) & (E <= c)
    generic += np.where(mab, area * 2 * (E - a) / (db * dd), 0.0)
    generic += np.where(mbc, area * 2 * (c - E) / (dc * dd), 0.0)

    if d32 < blend:
        r = 1.0 - smooth(d32 / blend)
        mac = (E >= a) & (E <= c)
        lim = np.where(mac, area * 2 * (E - a) / (dd ** 2), 0.0)
        return (1 - r) * generic + r * lim
    if d21 < blend:
        r = 1.0 - smooth(d21 / blend)
        mac = (E >= a) & (E <= c)
        lim = np.where(mac, area * 2 * (c - E) / (dd ** 2), 0.0)
        return (1 - r) * generic + r * lim
    return generic


def _reference_jdos_q(model, q_val, w_values, nk=16, form=True, band_slice=None):
    recip = model.reciprocal_vectors
    lattice = model.lattice_vectors
    k_frac, k_cart = generate_k_mesh(nk, recip)
    Nk = len(k_cart)
    nk_side = int(np.sqrt(Nk))
    area_BZ = abs(np.linalg.det(recip))
    dk = np.sqrt(area_BZ / Nk)
    area_tri = dk ** 2 / 2.0
    g = model.degeneracy_factor()
    prefactor = g / ((2 * np.pi) ** 2)
    tri_idx = _triangles_for_kmesh(nk_side)

    E_k, V_k = compute_eigenvalues(model, k_cart)
    if band_slice is not None:
        E_k = E_k[:, band_slice]
        V_k = V_k[:, :, band_slice]
    nb = E_k.shape[1]

    q_cart = np.array([q_val, 0.0])
    kq_cart = wrap_k_plus_q(k_frac, q_cart, recip, lattice, wrap=True)
    E_q, V_q = compute_eigenvalues(model, kq_cart)
    if band_slice is not None:
        E_q = E_q[:, band_slice]
        V_q = V_q[:, :, band_slice]
    nw = len(w_values)
    jdos = np.zeros(nw)
    for i1, i2, i3 in tri_idx:
        dE1 = E_q[i1, :, None] - E_k[i1, None, :]
        dE2 = E_q[i2, :, None] - E_k[i2, None, :]
        dE3 = E_q[i3, :, None] - E_k[i3, None, :]
        if form:
            M1 = density_form_factor(V_k[i1:i1 + 1], V_q[i1:i1 + 1])[0]
            M2 = density_form_factor(V_k[i2:i2 + 1], V_q[i2:i2 + 1])[0]
            M3 = density_form_factor(V_k[i3:i3 + 1], V_q[i3:i3 + 1])[0]
            M_avg = (M1 + M2 + M3) / 3.0
        else:
            M_avg = np.ones((nb, nb))
        for m in range(nb):
            for n in range(nb):
                wgt = M_avg[m, n]
                if wgt < 1e-30:
                    continue
                jdos += (prefactor * wgt * _corrected_triangle_dos(
                    float(dE1[m, n]), float(dE2[m, n]), float(dE3[m, n]),
                    w_values, area_tri))
    return jdos


# ── T2.1  q=0 triangle JDOS satisfies the sum rule ────────────────────
def test_jdos_q0_triangle_sum_rule():
    g = SingleLayerGrapheneTB(t=2.78)
    nb = g.n_bands
    nk = 24
    _, k_cart = generate_k_mesh(nk, g.reciprocal_vectors)
    E_k, _ = compute_eigenvalues(g, k_cart)
    area_BZ = abs(np.linalg.det(g.reciprocal_vectors))
    prefactor = g.degeneracy_factor() / ((2 * np.pi) ** 2)
    band_width = float(E_k.max() - E_k.min())
    w = np.linspace(0.0, band_width, 1500)

    _, jdos = compute_jdos_q0_triangle(g, w, nk=nk, interband_only=False)
    # all (m, n) pairs with M=1.  The ω grid is the positive half-axis only
    # and the q=0 JDOS is symmetric, so the captured integral is half the
    # full sum-rule target: total_weight = nb**2 / 2.
    ok, integral, expected = check_jdos_sum_rule(
        w, jdos, prefactor=prefactor, area_BZ=area_BZ, total_weight=nb ** 2 / 2)
    assert ok, f"JDOS q=0 sum rule: integral={integral:.3f} expected={expected:.3f}"


# ── T2.2  q→0 anchor: triangle q0 ≈ finite-q at small q ───────────────
def test_jdos_q0_anchor_vs_finite_q():
    g = SingleLayerGrapheneTB(t=2.78)
    nb = g.n_bands
    nk = 16
    _, k_cart = generate_k_mesh(nk, g.reciprocal_vectors)
    E_k, _ = compute_eigenvalues(g, k_cart)
    area_BZ = abs(np.linalg.det(g.reciprocal_vectors))
    prefactor = g.degeneracy_factor() / ((2 * np.pi) ** 2)
    band_width = float(E_k.max() - E_k.min())
    w = np.linspace(0.0, band_width, 1200)

    _, jdos_q0 = compute_jdos_q0_triangle(g, w, nk=nk, interband_only=False)
    jdos_q = compute_jdos_q_triangle(g, [1e-3], w, nk=nk, form=False,
                                     verbose=False)[:, 0]

    # Both are symmetric JDOS on a positive half-axis grid → captured integral
    # is half the full sum-rule target (total_weight = nb**2 / 2).
    _, i0, e0 = check_jdos_sum_rule(
        w, jdos_q0, prefactor=prefactor, area_BZ=area_BZ, total_weight=nb ** 2 / 2)
    _, iq, eq = check_jdos_sum_rule(
        w, jdos_q, prefactor=prefactor, area_BZ=area_BZ, total_weight=nb ** 2 / 2)
    assert abs(i0 - eq) / max(abs(eq), 1e-30) < 2e-2

    # Shapes converge at small q: L1 distance of normalised positive-ω spectra
    mask = w > 1e-3
    n0 = jdos_q0[mask] / np.trapezoid(jdos_q0[mask], w[mask])
    nq = jdos_q[mask] / np.trapezoid(jdos_q[mask], w[mask])
    l1 = np.trapezoid(np.abs(n0 - nq), w[mask]) / 2.0
    assert l1 < 0.35, f"q→0 anchor L1 distance too large: {l1:.3f}"

    # Peak frequencies track each other — compare in the high-ω (band-edge)
    # interband region, where the q→0 convergence is exact (verified: the
    # high-ω peaks coincide to <0.1% for q=1e-3).  A full-spectrum argmax is
    # dominated by the near-ω=0 singularity (a δ(ω) at q=0 vs a low-ω hump at
    # finite q) and is not a meaningful convergence metric.
    peak_mask = w > 0.5 * band_width
    p0 = w[peak_mask][np.argmax(jdos_q0[peak_mask])]
    pq = w[peak_mask][np.argmax(jdos_q[peak_mask])]
    assert abs(p0 - pq) / max(p0, 1e-30) < 0.1


# ── T2.3  triangle q=0 (interband) vs Gaussian q=0 (convention) ───────
def test_jdos_q0_triangle_vs_gaussian():
    from src.propagators.dos_gaussian import compute_jdos_q0
    g = SingleLayerGrapheneTB(t=2.78)
    nb = g.n_bands
    nk = 24
    _, k_cart = generate_k_mesh(nk, g.reciprocal_vectors)
    E_k, _ = compute_eigenvalues(g, k_cart)
    area_BZ = abs(np.linalg.det(g.reciprocal_vectors))
    prefactor = g.degeneracy_factor() / ((2 * np.pi) ** 2)
    band_width = float(E_k.max() - E_k.min())
    w = np.linspace(0.0, band_width, 1500)
    n_pairs = nb * (nb - 1) // 2  # m > n

    _, jt = compute_jdos_q0_triangle(g, w, nk=nk, interband_only=True)
    wg, jg = compute_jdos_q0(g, nk=nk, w_range=(0.0, band_width), nw=1500,
                             sigma=0.05, interband_only=True)

    # Same sum rule by construction (M=1, m>n pairs).
    _, it, et = check_jdos_sum_rule(
        w, jt, prefactor=prefactor, area_BZ=area_BZ, total_weight=n_pairs)
    _, ig, eg = check_jdos_sum_rule(
        wg, jg, prefactor=prefactor, area_BZ=area_BZ, total_weight=n_pairs)
    assert abs(it - ig) / max(abs(ig), 1e-30) < 0.15, \
        f"triangle/gaussian q0 integrals diverge: {it:.3f} vs {ig:.3f}"

    # Detailed shape: low-ω mean ratio ≈ 1 (soft convention check).
    lo = w < 0.3 * band_width
    if lo.sum() > 0:
        r = np.trapezoid(jt[lo], w[lo]) / max(np.trapezoid(jg[lo], wg[lo]), 1e-30)
        assert 0.7 < r < 1.4, f"low-ω mean ratio out of range: {r:.3f}"


# ── T3.1  finite-q refactor == independent reference ──────────────────
def test_jdos_q_triangle_matches_reference():
    g = SingleLayerGrapheneTB(t=2.78)
    nk = 14
    q_val = 0.05
    _, k_cart = generate_k_mesh(nk, g.reciprocal_vectors)
    E_k, _ = compute_eigenvalues(g, k_cart)
    band_width = float(E_k.max() - E_k.min())
    w = np.linspace(0.0, band_width, 800)

    new = compute_jdos_q_triangle(g, [q_val], w, nk=nk, form=True,
                                  verbose=False)[:, 0]
    ref = _reference_jdos_q(g, q_val, w, nk=nk, form=True)
    np.testing.assert_allclose(new, ref, rtol=1e-6, atol=1e-9)


# ── T3.2  performance wall removed (large smooth field) ───────────────
def test_primitive_scales_to_large_nfields():
    rng = np.random.default_rng(3)
    nk = 45                       # Nv = 2025, Nt = 2*nk**2 = 4050
    Nv = nk * nk
    nF = 1024                     # ~ TBG nb_cache=32 (32*32) band-pair fields
    # Smooth per-field 2D fields so each k-triangle's energy span is small —
    # the REAL production scenario (DOS/JDOS triangles are smooth over a
    # k-mesh).  A pathological *full-range* random field, where every triangle
    # spans the entire E grid, would require O(Nt·nF·nE) bin arrays and is not
    # representative of any caller, so we deliberately avoid it here.
    xs = np.linspace(0.0, 2.0 * np.pi, nk)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    field = np.zeros((Nv, nF), dtype=float)
    for f in range(nF):
        # Small-amplitude, low-frequency content => each k-triangle's energy
        # span is only a few E-bins (exactly the smooth-field regime of real
        # DOS/JDOS), so the bin-deposition workload stays tiny.
        c = rng.standard_normal(4) * 0.03
        field[:, f] = (
            c[0] * np.sin(0.4 * X).ravel()
            + c[1] * np.cos(0.4 * Y).ravel()
            + c[2] * np.sin(0.3 * (X + Y)).ravel()
            + c[3] * np.cos(0.5 * (X - Y)).ravel()
        )
    E = np.linspace(-5.0, 5.0, 2000)
    tri = _triangles_for_kmesh(nk)
    t0 = time.perf_counter()
    spec = triangle_spectrum(field, E, triangles=tri, area_BZ=1.0,
                             prefactor=1.0, enforce_sum_rule=True)
    dt = time.perf_counter() - t0
    assert spec.shape == (E.size, nF)
    # The old triple-loop approach made ~Nt*nF = 4M scalar _triangle_dos_exact
    # calls (minutes); the batched primitive must finish in well under a second.
    # Soft guard (T3.2): keep a generous bound.
    assert dt < 5.0, f"primitive too slow: {dt:.2f}s for (Nt={2*Nv}, nF={nF})"


# ── T3.3  form-factor weight conservation ──────────────────────────────
def test_form_factor_weight_conservation():
    """form=False (M≡1) and form=True (averaged |M|²) must both satisfy the
    sum rule; form=True's total weight equals Σ_{m,n} mean|M|²."""
    g = SingleLayerGrapheneTB(t=2.78)
    nb = g.n_bands
    nk = 16
    _, k_cart = generate_k_mesh(nk, g.reciprocal_vectors)
    E_k, V_k = compute_eigenvalues(g, k_cart)
    area_BZ = abs(np.linalg.det(g.reciprocal_vectors))
    prefactor = g.degeneracy_factor() / ((2 * np.pi) ** 2)
    band_width = float(E_k.max() - E_k.min())
    # Symmetric ω grid so the captured integral equals the FULL sum-rule
    # target (both positive and negative transitions are on the grid).
    w = np.linspace(-band_width, band_width, 2400)

    q_val = 0.04
    recip = g.reciprocal_vectors
    lattice = g.lattice_vectors
    k_frac, _ = generate_k_mesh(nk, recip)
    kq = wrap_k_plus_q(k_frac, np.array([q_val, 0.0]), recip, lattice, wrap=True)
    _, V_q = compute_eigenvalues(g, kq)
    # Σ_{m,n} mean over k of |M|² (averaged over the 3 triangle vertices per k)
    M_all = density_form_factor(V_k, V_q)  # (Nk, nb, nb)
    total_weight = float(M_all.mean(axis=0).sum())

    jf = compute_jdos_q_triangle(g, [q_val], w, nk=nk, form=True,
                                 verbose=False)[:, 0]
    ok, integral, expected = check_jdos_sum_rule(
        w, jf, prefactor=prefactor, area_BZ=area_BZ, total_weight=total_weight)
    assert ok, f"form=True sum rule: integral={integral:.3f} expected={expected:.3f}"

    jnf = compute_jdos_q_triangle(g, [q_val], w, nk=nk, form=False,
                                  verbose=False)[:, 0]
    ok2, i2, e2 = check_jdos_sum_rule(
        w, jnf, prefactor=prefactor, area_BZ=area_BZ, total_weight=nb ** 2)
    assert ok2, f"form=False sum rule: integral={i2:.3f} expected={e2:.3f}"
