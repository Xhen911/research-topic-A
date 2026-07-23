"""
jdos.py
=======
Triangle-method Joint Density of States at FINITE q.

JDOS(q, omega) = g/(2pi)^2 sum_{m,n} integral d^2k
                 |<m,k+q|n,k>|^2  delta(omega - [E_m(k+q) - E_n(k)])

Uses the same triangle decomposition as compute_dos_triangle (exact eta->0
Lehmann-Taut formula), but with transition energies dE = E_m(k+q) - E_n(k)
instead of single-particle energies E_n(k).

Core functions
--------------
- compute_jdos_q_triangle(model, q_values, w_values, nk, ...) -> (nw, nq)
  JDOS at finite q with form-factor weighting.

See also
--------
- dos_gaussian.compute_jdos_q0  — Gaussian JDOS at q=0 only
- dos.compute_dos_triangle      — single-particle DOS (triangle method)
"""

import numpy as np


def compute_jdos_q_triangle(model, q_values, w_values, nk=48,
                            form=True, band_slice=None, verbose=True):
    """Triangle-method JDOS at finite q with form-factor weighting.

    Within each triangle the transition energy dE_{mn}(k) is linearly
    interpolated from the three vertex values.  The JDOS contribution
    is integrated analytically (piecewise-linear) — no broadening,
    no branch-cut logarithms.

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
    from .lindhard import generate_k_mesh, wrap_k_plus_q
    from .dos import _triangles_for_kmesh, _triangle_dos_exact, compute_eigenvalues
    from ..bands.vertices import density_form_factor

    q_values = np.atleast_1d(np.asarray(q_values, dtype=float))
    w_values = np.atleast_1d(np.asarray(w_values, dtype=float))
    nq, nw = len(q_values), len(w_values)

    # k-mesh
    recip = model.reciprocal_vectors
    lattice = model.lattice_vectors
    k_frac, k_cart = generate_k_mesh(nk, recip)
    Nk = len(k_cart)
    nk_side = int(np.sqrt(Nk))
    assert nk_side ** 2 == Nk

    area_BZ = abs(np.linalg.det(recip))
    dk = np.sqrt(area_BZ / Nk)
    area_tri = dk ** 2 / 2.0
    g = model.degeneracy_factor()
    prefactor = g / ((2 * np.pi) ** 2)

    # Triangle connectivity
    tri_idx = _triangles_for_kmesh(nk_side)

    # k-point eigenvalues
    E_k, V_k = compute_eigenvalues(model, k_cart)
    if band_slice is not None:
        E_k = E_k[:, band_slice]
        V_k = V_k[:, :, band_slice]
    nb = E_k.shape[1]

    if verbose:
        print(f"  [jdos_q] nk={nk} (Nk={Nk}), nq={nq}, nw={nw}, "
              f"nb={nb}, n_tri={len(tri_idx)}")

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

        # Triangle loop
        for i1, i2, i3 in tri_idx:
            dE1 = E_q[i1, :, np.newaxis] - E_k[i1, np.newaxis, :]  # (nb, nb)
            dE2 = E_q[i2, :, np.newaxis] - E_k[i2, np.newaxis, :]
            dE3 = E_q[i3, :, np.newaxis] - E_k[i3, np.newaxis, :]

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
                    jdos[:, iq] += (
                        prefactor * wgt
                        * _triangle_dos_exact(
                            float(dE1[m, n]), float(dE2[m, n]),
                            float(dE3[m, n]), w_values, area_tri)
                    )

    return jdos
