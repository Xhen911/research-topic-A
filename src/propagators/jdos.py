"""
jdos.py
=======
Triangle-method Joint Density of States.

JDOS(q, omega) = g/(2pi)^2 sum_{m,n} integral d^2k
                 |<m,k+q|n,k>|^2  delta(omega - [E_m(k+q) - E_n(k)])

Two triangle-method integrators are provided:

- ``compute_jdos_q0_triangle``  — q = 0 only (single diagonalisation, no wrap,
  no form factor).  This is the natural q→0 anchor for the finite-q version and
  is the triangle-method analogue of ``dos_gaussian.compute_jdos_q0``.
- ``compute_jdos_q_triangle``   — finite q with form-factor weighting.

Both build the transition-energy field ``dE_{mn}(vertex)`` per triangle and
hand it to the vectorised primitive ``triangle_spectrum`` (triangle_core),
replacing the old triple Python loop (triangle × band × band) that called
``_triangle_dos_exact`` once per (triangle, m, n) — O(4.7M) scalar calls for a
TBG scan with ``nb_cache=32``.

See also
--------
- dos_gaussian.compute_jdos_q0  — Gaussian JDOS at q=0 only
- dos.compute_dos_triangle      — single-particle DOS (triangle method)
"""

import numpy as np

from .dos import compute_eigenvalues
from .triangle_core import triangle_spectrum, _triangles_for_kmesh
from ..bands.vertices import density_form_factor


def _prefactor_and_mesh(model, nk):
    """Shared BZ setup: k-mesh, triangle connectivity, area, prefactor."""
    from .lindhard import generate_k_mesh
    recip = model.reciprocal_vectors
    lattice = model.lattice_vectors
    k_frac, k_cart = generate_k_mesh(nk, recip)
    Nk = len(k_cart)
    nk_side = int(np.sqrt(Nk))
    assert nk_side ** 2 == Nk, f"nk² ≠ Nk={Nk}"

    area_BZ = abs(np.linalg.det(recip))
    dk = np.sqrt(area_BZ / Nk)
    area_tri = dk ** 2 / 2.0
    g = model.degeneracy_factor()
    prefactor = g / ((2 * np.pi) ** 2)
    tri_idx = _triangles_for_kmesh(nk_side)
    return k_frac, k_cart, lattice, recip, area_BZ, area_tri, prefactor, tri_idx, nk_side


def compute_jdos_q0_triangle(model, w_values, nk=48, *, band_slice=None,
                             interband_only=False):
    """Triangle-method JDOS at q = 0 (single diagonalisation).

    The transition energy field is ``dE_{mn}(k) = E_m(k) - E_n(k)`` taken at
    the *same* k-point, so there is no k+q wrap and no form-factor weighting.
    This is the q→0 anchor for ``compute_jdos_q_triangle`` and the triangle
    analogue of ``dos_gaussian.compute_jdos_q0``.

    Parameters
    ----------
    model : HamiltonianModel
    w_values : (nw,)  frequency grid [eV]
    nk : int  k-points per reciprocal direction (total = nk^2)
    band_slice : slice or None
    interband_only : bool
        True  → sum only pairs (m, n) with m > n (matches the Gaussian q=0
                convention, interband-only, ω ≥ 0).
        False → sum all (m, n) pairs (matches the finite-q construction at q→0).

    Returns
    -------
    w_values : (nw,)
    jdos : (nw,)  [states/eV/unit cell]
    """
    w_values = np.atleast_1d(np.asarray(w_values, dtype=float))
    nw = len(w_values)

    (_, k_cart, _, _, area_BZ, area_tri, prefactor,
     tri_idx, _) = _prefactor_and_mesh(model, nk)

    E_k, _ = compute_eigenvalues(model, k_cart)
    if band_slice is not None:
        E_k = E_k[:, band_slice]
    nb = E_k.shape[1]
    Nt = len(tri_idx)

    if interband_only:
        pairs = [(m, n) for m in range(nb) for n in range(m)]  # m > n
    else:
        pairs = [(m, n) for m in range(nb) for n in range(nb)]
    nf = len(pairs)

    # dE field: (Nt, 3, nf) — transition energy at the three triangle vertices
    dE = np.zeros((Nt, 3, nf))
    for v in range(3):
        Ek_v = E_k[tri_idx[:, v]]               # (Nt, nb)
        dE[:, v, :] = np.array(
            [Ek_v[:, m] - Ek_v[:, n] for (m, n) in pairs]).T  # (Nt, nf)

    spectrum = triangle_spectrum(
        dE, w_values, triangles=None, area_BZ=area_BZ,
        prefactor=prefactor, enforce_sum_rule=False)
    jdos = spectrum.sum(axis=1)
    return w_values, jdos


def compute_jdos_q_triangle(model, q_values, w_values, nk=48,
                            form=True, band_slice=None, verbose=True):
    """Triangle-method JDOS at finite q with form-factor weighting.

    Within each triangle the transition energy dE_{mn}(k) is linearly
    interpolated from the three vertex values and integrated analytically
    (piecewise-linear Lehmann-Taut) — no broadening, no branch-cut logarithms.

    Parameters
    ----------
    model : HamiltonianModel
    q_values : (nq,)  q-vector magnitudes (q || x-hat)
    w_values : (nw,)  frequency grid [eV]
    nk : int  k-points per reciprocal direction (total = nk^2)
    form : bool  include form factor |<m,k+q|n,k>|^2 weighting
    band_slice : slice or None  restrict band range
    verbose : bool

    Returns
    -------
    jdos : (nw, nq)  [states/eV/unit cell]
    """
    from .lindhard import wrap_k_plus_q

    q_values = np.atleast_1d(np.asarray(q_values, dtype=float))
    w_values = np.atleast_1d(np.asarray(w_values, dtype=float))
    nq, nw = len(q_values), len(w_values)

    (k_frac, k_cart, lattice, recip, area_BZ, area_tri, prefactor,
     tri_idx, nk_side) = _prefactor_and_mesh(model, nk)
    Nt = len(tri_idx)

    # k-point eigenvalues
    E_k, V_k = compute_eigenvalues(model, k_cart)
    if band_slice is not None:
        E_k = E_k[:, band_slice]
        V_k = V_k[:, :, band_slice]
    nb = E_k.shape[1]

    if verbose:
        print(f"  [jdos_q] nk={nk} (Nk={nk_side ** 2}), nq={nq}, nw={nw}, "
              f"nb={nb}, n_tri={Nt}")

    # Pre-gather vertex energies/eigenstates per triangle vertex (Nt ordering)
    E_k_t = np.stack([E_k[tri_idx[:, v]] for v in range(3)], axis=1)  # (Nt, 3, nb)
    V_k_t = [V_k[tri_idx[:, v]] for v in range(3)]                    # list of (Nt, no, nb)

    jdos = np.zeros((nw, nq))

    for iq in range(nq):
        q_val = max(q_values[iq], 1e-12)
        q_cart = np.array([q_val, 0.0])

        if verbose:
            print(f"  [jdos_q] iq={iq + 1}/{nq}  q={q_val:.4f}")

        # k+q eigenvalues
        kq_cart = wrap_k_plus_q(k_frac, q_cart, recip, lattice, wrap=True)
        E_q, V_q = compute_eigenvalues(model, kq_cart)
        if band_slice is not None:
            E_q = E_q[:, band_slice]
            V_q = V_q[:, :, band_slice]
        E_q_t = np.stack([E_q[tri_idx[:, v]] for v in range(3)], axis=1)  # (Nt, 3, nb)
        V_q_t = [V_q[tri_idx[:, v]] for v in range(3)]

        # Build the transition-energy field dE_{mn} and weight M_avg_{mn}
        # for every (m, n) pair: shape (Nt, 3, nb*nb).
        nf = nb * nb
        dE = np.zeros((Nt, 3, nf))
        Mw = np.zeros((Nt, nf)) if form else None
        for v in range(3):
            dE_v = (E_q_t[:, v, :, None] - E_k_t[:, v, None, :]).reshape(Nt, nf)
            dE[:, v, :] = dE_v
            if form:
                Mv = density_form_factor(V_k_t[v], V_q_t[v])  # (Nt, nb, nb)
                Mw += Mv.reshape(Nt, nf)
        if form:
            Mw /= 3.0  # average the form factor over the three vertices

        # One batched primitive call for all (m, n) pairs.
        # NOTE: enforce_sum_rule is False here on purpose.  The transition
        # field dE = E_m(k+q) - E_n(k) is signed (both (m,n) and (n,m)
        # orderings appear) so the ω grid is usually the positive half-axis
        # only.  Rescaling the captured half to the *full* sum-rule target
        # would double the physical positive-ω JDOS.  The primitive still
        # returns the exact (un-normalised) Lehmann-Taut JDOS, matching the
        # original triple-loop behaviour.
        spectrum = triangle_spectrum(
            dE, w_values, weights=Mw, triangles=None, area_BZ=area_BZ,
            prefactor=prefactor, enforce_sum_rule=False)
        jdos[:, iq] = spectrum.sum(axis=1)

    return jdos


def check_jdos_sum_rule(w_values, jdos, expected=None, *,
                        prefactor=None, area_BZ=None, total_weight=None,
                        tol=2e-2):
    """Verify the JDOS sum rule ∫ JDOS(ω) dω = prefactor·area_BZ·Σ_weight.

    Either pass ``expected`` directly, or the constituents
    ``(prefactor, area_BZ, total_weight)`` from which it is computed:

        expected = prefactor · area_BZ · total_weight

    where ``total_weight`` is the sum of per-field (per-transition-pair)
    weights (e.g. Σ_{m,n} M_avg[m,n], or nb² for M=1, all pairs).

    Returns
    -------
    ok : bool
    integral : float
    expected : float
    """
    _trapz = getattr(np, "trapz", np.trapezoid)
    integral = float(_trapz(jdos, w_values))
    if expected is None:
        if prefactor is None or area_BZ is None or total_weight is None:
            raise ValueError(
                "Provide either `expected` or (prefactor, area_BZ, total_weight).")
        expected = float(prefactor * area_BZ * total_weight)
    rel = abs(integral - expected) / max(abs(expected), 1e-30)
    ok = rel < tol
    return ok, integral, expected
