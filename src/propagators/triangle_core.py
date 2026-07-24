"""
triangle_core.py
=================
Batched Lehmann-Taut triangle-spectrum primitive.

This is the single source of truth for the exact η→0 triangle integration
(piecewise-linear interpolation inside each triangle of a uniform k-mesh),
used by both the single-particle DOS and the joint DOS (JDOS).

The old code implemented this as a triple Python loop over
(triangle, band, band) each calling ``_triangle_dos_exact`` once — that is
O(4.7M) scalar calls for a TBG scan with ``nb_cache=32`` and is unusable.
This module replaces it with a vectorised primitive:

    triangle_spectrum(vertex_fields, E_grid, weights=None, ...) -> (nE, n_fields)

which loops *only over fields* (bands / transition pairs) and, inside each
field, accumulates the Lehmann-Taut contribution of all triangles at once via
``np.searchsorted`` + ``np.add.at``.

Convention note (2026-07-24, KC-2)
----------------------------------
The Taylor-blend ramp direction here is the *corrected* one:

    r = 1 - smoothstep(delta / blend_width)     (delta -> 0  =>  r -> 1)

i.e. in the near-degenerate limit (delta small) we blend *towards* the
degenerate-limit formula, and far from degeneracy (delta >= blend_width) we
use the generic piecewise-linear formula.  The legacy ``_triangle_dos_exact``
in ``dos.py`` used the reversed direction (its ramp was 0 at degeneracy and 1
away from it, contradicting its own comment).  We fix the direction here and
do **not** keep a legacy switch.  See docs/三角形谱计算优化.md §"需要你特别注意".
"""

import numpy as np
from typing import Optional


# ============================================================
#  Triangle connectivity for a uniform nk×nk BZ parallelogram
#  (moved here so triangle_core has no dependency on dos.py)
# ============================================================

def _triangles_for_kmesh(nk):
    """Triangle connectivity for a uniform nk×nk BZ parallelogram mesh.

    Returns an ``(Nt, 3)`` integer array of vertex indices, ``Nt = 2*nk**2``.
    """
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


# ============================================================
#  Internal helpers
# ============================================================

def _smoothstep(t: np.ndarray) -> np.ndarray:
    """C¹ smoothstep: 0 for t<=0, 1 for t>=1."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _interval_indices(starts, ends):
    """Expand per-triangle [start, end) index ranges into global E-grid indices.

    Parameters
    ----------
    starts, ends : np.ndarray (int) of shape (N,)
        Inclusive lower / exclusive (or inclusive for side='right') bounds
        for each triangle.

    Returns
    -------
    idx : np.ndarray (int) or None
        Global E-grid indices, shape (total,).
    rep : np.ndarray (int) or None
        Which triangle each ``idx`` belongs to.
    counts : np.ndarray (int)
        Number of points contributed by each triangle.
    """
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)

    counts = np.maximum(ends - starts, 0).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return None, None, counts

    rep = np.repeat(np.arange(starts.size, dtype=np.int64), counts)

    cum = np.empty(counts.size + 1, dtype=np.int64)
    cum[0] = 0
    np.cumsum(counts, out=cum[1:])

    offset = np.arange(total, dtype=np.int64) - np.repeat(cum[:-1], counts)
    idx = starts[rep] + offset

    return idx, rep, counts


# ============================================================
#  The primitive
# ============================================================

def triangle_spectrum(
    vertex_fields: np.ndarray,
    E_grid: np.ndarray,
    weights: Optional[np.ndarray] = None,
    *,
    triangles: Optional[np.ndarray] = None,
    area_tri: Optional[float] = None,
    area_BZ: Optional[float] = None,
    prefactor: float = 1.0,
    deg_tol: float = 1e-14,
    blend_scale: float = 1e-6,
    enforce_sum_rule: bool = True,
    gauss_half_width: int = 8,
) -> np.ndarray:
    """Batched Lehmann-Taut triangle spectrum.

    For each field (band, transition pair, ...) the contribution of every
    triangle is the exact piecewise-linear Lehmann-Taut DOS of a linearly
    interpolated energy field, summed (with per-triangle weight) over the
    mesh.  The result is ``(nE, n_fields)``; the total 1D spectrum is
    ``spectrum.sum(axis=1)``.

    Parameters
    ----------
    vertex_fields : np.ndarray
        Vertex field, one of:
          - ``(Nv, n_fields)``: field value at each mesh vertex (then
            ``triangles`` is required, or ``Nv`` must equal ``nk**2`` so the
            regular mesh connectivity can be inferred).
          - ``(Nt, 3, n_fields)``: field already gathered per triangle.
    E_grid : np.ndarray
        1D, strictly increasing energy grid, shape ``(nE,)``.
    weights : np.ndarray, optional
        Per-vertex weight, matching the shape of ``vertex_fields`` (or
        ``(Nt, n_fields)`` / ``(n_fields,)`` broadcast).  For each triangle
        the three vertex weights are averaged: ``w_T = mean(w_v)``.
        Built-in sum rule: ``integral over field f = prefactor * area_BZ *
        mean(weight_f)``.
    triangles : np.ndarray, optional
        ``(Nt, 3)`` connectivity, used when ``vertex_fields`` is
        ``(Nv, n_fields)``.
    area_tri : float, optional
        Single-triangle area.  If omitted but ``area_BZ`` is given,
        ``area_tri = area_BZ / Nt``.  If both omitted, ``area_BZ = 1``.
    area_BZ : float, optional
        Brillouin-zone area.
    prefactor : float
        Global prefactor (e.g. ``model.degeneracy_factor() / (2*pi)**2`` for
        the physical DOS).
    deg_tol : float
        Fully-degenerate threshold (all three vertices equal).
    blend_scale : float
        Taylor-blend width as a fraction of the triangle energy span.
    enforce_sum_rule : bool
        If True, rescale each field after integration so that its integral
        equals ``prefactor * area_BZ * mean(weight_f)``.
    gauss_half_width : int
        For fully-degenerate triangles, the Gaussian replacement is added only
        within ``± gauss_half_width * sigma`` of the centre.

    Returns
    -------
    spectrum : np.ndarray, shape ``(nE, n_fields)``
    """
    E = np.asarray(E_grid, dtype=float)
    if E.ndim != 1:
        raise ValueError("E_grid must be a 1D array.")
    if E.size > 1 and np.any(np.diff(E) <= 0):
        raise ValueError("E_grid must be strictly increasing.")

    nE = E.size
    dE = float(np.mean(np.diff(E))) if nE > 1 else 1e-3

    F = np.asarray(vertex_fields, dtype=float)
    W_in = None if weights is None else np.asarray(weights, dtype=float)

    # ---- normalise to Fv, Wv : shape (Nt, 3, n_fields) ----
    if F.ndim == 2:
        Nv, nF = F.shape
        if triangles is None:
            nk = int(round(np.sqrt(Nv)))
            if nk * nk != Nv:
                raise ValueError(
                    "For vertex_fields with shape (Nv, n_fields), "
                    "either provide triangles=(Nt, 3), or ensure Nv=nk^2 "
                    "so connectivity can be inferred.")
            tri = _triangles_for_kmesh(nk)
        else:
            tri = np.asarray(triangles, dtype=int)
        Fv = F[tri]  # (Nt, 3, nF)
        if W_in is None:
            Wv = np.ones_like(Fv)
        elif W_in.ndim == 2 and W_in.shape == (Nv, nF):
            Wv = W_in[tri]
        elif W_in.shape == Fv.shape:
            Wv = W_in.astype(float, copy=False)
        elif W_in.ndim == 2 and W_in.shape == (tri.shape[0], nF):
            Wv = W_in[:, None, :]
        else:
            Wv = np.broadcast_to(W_in, Fv.shape).astype(float, copy=False)

    elif F.ndim == 3 and F.shape[1] == 3:
        Fv = F.astype(float, copy=False)
        nF = Fv.shape[2]
        if W_in is None:
            Wv = np.ones_like(Fv)
        elif W_in.shape == Fv.shape:
            Wv = W_in.astype(float, copy=False)
        elif W_in.ndim == 2 and W_in.shape == (Fv.shape[0], nF):
            Wv = W_in[:, None, :]
        else:
            Wv = np.broadcast_to(W_in, Fv.shape).astype(float, copy=False)
    else:
        raise ValueError(
            "vertex_fields must have shape (Nv, n_fields) "
            "or (Nt, 3, n_fields).")

    Nt = Fv.shape[0]

    # Ensure Wv always carries a full 3-vertex axis.  Some callers (e.g.
    # compute_jdos_q_triangle) pass a per-triangle ``(Nt, nF)`` weight that is
    # already the mean over the 3 vertices; broadcast it so the rest of the
    # primitive can treat Wv uniformly as ``(Nt, 3, nF)``.
    if Wv.shape[1] == 1:
        Wv = np.broadcast_to(Wv, (Wv.shape[0], 3, Wv.shape[2]))

    # ---- area / sum rule ----
    if area_tri is None:
        if area_BZ is not None:
            area_tri = float(area_BZ) / max(Nt, 1)
        else:
            area_tri = 1.0 / max(Nt, 1)
    if area_BZ is None:
        area_BZ = float(area_tri) * Nt
    area_tri = float(area_tri)
    area_BZ = float(area_BZ)
    prefactor = float(prefactor)

    spectrum = np.zeros((nE, nF), dtype=float)
    if nE == 0 or Nt == 0:
        return spectrum

    _trapz = np.trapezoid
    tiny = 1e-30
    sqrt2pi = np.sqrt(2.0 * np.pi)

    # ---- vectorised accumulation, chunked over fields ----
    # Eliminates the old ``for f in range(nF)`` Python loop.  Within a chunk
    # every Lehmann-Taut branch runs ONCE on a flattened ``(Nt*C, 3)`` array
    # (all triangles times the ``C`` fields in the chunk) using numpy
    # primitives, then the result is scattered back to the right columns via a
    # flat global index ``field_idx * nE + E_bin``.
    #
    # Triangle rows are laid out as ``r = i*C + f`` (``i`` = triangle,
    # ``f`` = local field index inside the chunk), so ``field_idx[r] = f``
    # maps any flattened point back to its destination column.  Chunking keeps
    # peak memory bounded: real DOS/JDOS triangles are smooth (small energy
    # span within one k-triangle), so the per-chunk bin arrays stay small.
    _MAX_TRI = 120_000
    chunk = max(1, min(nF, _MAX_TRI // max(Nt, 1)))

    for f0 in range(0, nF, chunk):
        f1 = min(f0 + chunk, nF)
        C = f1 - f0
        Fv_c = Fv[:, :, f0:f1]            # (Nt, 3, C)
        Wv_c = Wv[:, :, f0:f1]            # (Nt, 3, C)

        Ntc = Nt * C
        F2 = Fv_c.transpose(0, 2, 1).reshape(-1, 3)              # (Ntc, 3)
        W2 = Wv_c.transpose(0, 2, 1).reshape(-1, 3)              # (Ntc, 3)
        field_idx = np.broadcast_to(np.arange(C), (Nt, C)).reshape(-1)  # (Ntc,)

        e_sort = np.sort(F2, axis=1)
        a = e_sort[:, 0]
        b = e_sort[:, 1]
        c = e_sort[:, 2]

        d21 = b - a
        d32 = c - b
        d31 = c - a

        w_tri = W2.mean(axis=1)                              # (Ntc,)
        base_area = prefactor * area_tri * w_tri

        all_deg = d31 < deg_tol
        blend_width = np.maximum(1e-8, blend_scale * d31)

        # Corrected direction: delta -> 0  =>  r -> 1  (blend to degenerate limit)
        high = (d32 < blend_width) & (~all_deg)          # e2 ~ e3
        low = (d21 < blend_width) & (~all_deg) & (~high)  # e1 ~ e2

        r_high = np.zeros(Ntc, dtype=float)
        if np.any(high):
            r_high[high] = 1.0 - _smoothstep(d32[high] / blend_width[high])
        r_low = np.zeros(Ntc, dtype=float)
        if np.any(low):
            r_low[low] = 1.0 - _smoothstep(d21[low] / blend_width[low])

        generic_factor = np.ones(Ntc, dtype=float)
        generic_factor[all_deg] = 0.0
        generic_factor[high] = 1.0 - r_high[high]
        generic_factor[low] = 1.0 - r_low[low]

        d21s = np.maximum(d21, tiny)
        d32s = np.maximum(d32, tiny)
        d31s = np.maximum(d31, tiny)

        base_generic = base_area * generic_factor

        flat = np.zeros(C * nE, dtype=float)

        # 4. generic piecewise-linear contribution
        #    low interval [a, b)
        starts = np.searchsorted(E, a, side="left")
        ends = np.searchsorted(E, b, side="left")
        idx, rep, _ = _interval_indices(starts, ends)
        if idx is not None:
            Eg = E[idx]
            den = (d21s * d31s)[rep]
            val = base_generic[rep] * 2.0 * (Eg - a[rep]) / den
            flat += np.bincount(idx * C + field_idx[rep], weights=val,
                                minlength=C * nE)

        #    high interval [b, c]
        starts = np.searchsorted(E, b, side="left")
        ends = np.searchsorted(E, c, side="right")
        idx, rep, _ = _interval_indices(starts, ends)
        if idx is not None:
            Eg = E[idx]
            den = (d32s * d31s)[rep]
            val = base_generic[rep] * 2.0 * (c[rep] - Eg) / den
            flat += np.bincount(idx * C + field_idx[rep], weights=val,
                                minlength=C * nE)

        # 5. e2 ~ e3 limit (b -> c): DOS = 2*A*(E-a)/(c-a)^2 on [a, c]
        if np.any(high):
            rows = np.nonzero(high)[0]
            am = a[rows]
            cm = c[rows]
            d31m = d31s[rows]
            base_lim = base_area[rows] * r_high[rows]
            starts = np.searchsorted(E, am, side="left")
            ends = np.searchsorted(E, cm, side="right")
            idx, rep, _ = _interval_indices(starts, ends)
            if idx is not None:
                Eg = E[idx]
                val = base_lim[rep] * 2.0 * (Eg - am[rep]) / (d31m[rep] ** 2)
                flat += np.bincount(idx * C + field_idx[rows[rep]],
                                    weights=val, minlength=C * nE)

        # 6. e1 ~ e2 limit (a -> b): DOS = 2*A*(c-E)/(c-a)^2 on [a, c]
        if np.any(low):
            rows = np.nonzero(low)[0]
            am = a[rows]
            cm = c[rows]
            d31m = d31s[rows]
            base_lim = base_area[rows] * r_low[rows]
            starts = np.searchsorted(E, am, side="left")
            ends = np.searchsorted(E, cm, side="right")
            idx, rep, _ = _interval_indices(starts, ends)
            if idx is not None:
                Eg = E[idx]
                val = base_lim[rep] * 2.0 * (cm[rep] - Eg) / (d31m[rep] ** 2)
                flat += np.bincount(idx * C + field_idx[rows[rep]],
                                    weights=val, minlength=C * nE)

        # 7. fully degenerate: Gaussian replacement for the Dirac delta
        if np.any(all_deg):
            rows = np.nonzero(all_deg)[0]
            centers = a[rows]
            base_g = base_area[rows]
            sigma = max(abs(dE), 1e-14)
            half = gauss_half_width * sigma
            starts = np.searchsorted(E, centers - half, side="left")
            ends = np.searchsorted(E, centers + half, side="right")
            idx, rep, counts = _interval_indices(starts, ends)
            if idx is not None:
                Eg = E[idx]
                cen = centers[rep]
                val = (base_g[rep]
                       * np.exp(-0.5 * ((Eg - cen) / sigma) ** 2)
                       / (sigma * sqrt2pi))
                flat += np.bincount(idx * C + field_idx[rows[rep]],
                                    weights=val, minlength=C * nE)
            # nearest-neighbour patch-up for deltas that fell between bins
            missing = counts == 0
            if np.any(missing):
                dm = rows[missing]
                nearest = np.clip(
                    np.searchsorted(E, centers[missing]), 0, nE - 1)
                flat[nearest * C + field_idx[dm]] += base_g[missing] / sigma

        out_chunk = flat.reshape(nE, C)

        # 8. built-in sum rule (per field, inside the chunk)
        if enforce_sum_rule:
            target = prefactor * area_BZ * Wv_c.mean(axis=(0, 1))   # (C,)
            if nE > 1:
                integrals = _trapz(out_chunk, E, axis=0)
            else:
                integrals = out_chunk[0, :] * dE
            nz = np.abs(integrals) > 1e-300
            scale = np.ones(C, dtype=float)
            scale[nz] = target[nz] / integrals[nz]
            out_chunk = out_chunk * scale[None, :]

        spectrum[:, f0:f1] = out_chunk

    return spectrum
