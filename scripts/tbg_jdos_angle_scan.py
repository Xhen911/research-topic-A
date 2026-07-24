#!/usr/bin/env python
"""
tbg_jdos_angle_scan.py — Multi-angle TBG probe: DOS + JDOS (Lifshitz / vHS scan)
==================================================================================

For each twist angle θ this program evaluates
  * single-particle DOS              -> energy axis      E   (eV)
  * JDOS at q = 0                    -> transition-energy axis  ω  (eV)
  * JDOS at finite q (form factor)   -> (ω, q) grid

ALL raw axis data — the DOS curves and the transition-energy (JDOS) spectra —
are written to a single ``.npz`` file *before* any plotting, so the numerical
results are preserved for later re-visualisation or downstream analysis.  A
summary plot (DOS vs E and JDOS(q=0) vs ω across angles, with VHS/Lifshitz
markers, plus a JDOS(q,ω) heatmap) is produced afterwards.

It extends scripts/lifshitz_scan.py (DOS / VHS batch scan) with the
triangle-method JDOS transition-energy spectrum from src.propagators.jdos.

Usage
-----
    # full Lifshitz/vHS angle scan
    python scripts/tbg_jdos_angle_scan.py --theta 0.80 0.99 1.05 1.08 1.16 1.47 \
        --nk 24 --nq 12 --nw 400 --save-cache --plot

    # quick test
    python scripts/tbg_jdos_angle_scan.py --theta 1.05 1.16 --nk 16 --nq 6 --nw 200

Reference conventions
---------------------
    q-mesh, half-step offset, etc. follow the July-2026 revision
    (see scripts/scan_response.py).  Here q for JDOS is a small grid near Γ
    built from the k-space spacing dk = sqrt(A_BZ / nk^2).
"""

import numpy as np
import os, sys, argparse, time

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bands import BistritzMacDonaldTBG
from src.core.cache import CachedModel
from src.propagators.dos import compute_dos_triangle, check_dos_sum_rule
from src.propagators.jdos import (
    compute_jdos_q0_triangle, compute_jdos_q_triangle, check_jdos_sum_rule)
from src.bands.occupations import compute_filling, compute_cnp


# ── VHS / Lifshitz detection (lightweight, no scipy) ────────────────────────
def find_local_maxima(x, y, min_height_frac=0.04, min_sep=3):
    """Indices of local maxima with height > min_height_frac * global max,
    separated by at least ``min_sep`` grid points."""
    y = np.asarray(y, float)
    if y.size < 3:
        return np.array([], dtype=int)
    peak = (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])
    idx = np.where(peak)[0] + 1
    thr = min_height_frac * y.max()
    idx = idx[y[idx] > thr]
    out, last = [], -10 ** 9
    for i in idx:
        if i - last >= min_sep:
            out.append(int(i))
            last = i
    return np.array(out, dtype=int)


def _build_or_load(model, theta, nk, n_shells, nq, q_max_factor,
                   cache_dir, save_cache):
    """Return (CachedModel, E_k flat-bands slice, flat-band energy array)."""
    ttag = f'theta{theta:.2f}'.replace('.', 'p')
    cache_path = os.path.join(cache_dir, f'eig-cache-{ttag}-nk{nk}.npz')
    if os.path.exists(cache_path):
        print(f'  [cache] loading {cache_path}')
        cache = CachedModel.load(cache_path)
    else:
        cache = CachedModel(model, nk=nk, n_q=nq, nb_cache=32,
                            q_max_factor=q_max_factor)
        if save_cache:
            cache.save(cache_path)
    E_k = cache.E_k
    nb = E_k.shape[1]
    half = nb // 2
    flat_slice = slice(half - 1, half + 1)
    return cache, E_k, flat_slice


def analyze_theta(theta, nk=24, n_shells=2, nq=12, nw=400,
                  q_max_factor=1.2, nE=3000, cache_dir='.',
                  save_cache=False, do_jdos_q=True):
    """Full DOS + JDOS analysis for one twist angle.

    Returns dict with DOS curve, JDOS(q=0) spectrum, JDOS(q,ω) grid (or None),
    CNP, and detected VHS/Lifshitz markers.
    """
    model = BistritzMacDonaldTBG(theta=theta, n_shells=n_shells)
    cache, E_k, flat_slice = _build_or_load(
        model, theta, nk, n_shells, nq, q_max_factor, cache_dir, save_cache)
    nb_flat = flat_slice.stop - flat_slice.start

    E_cnp = compute_cnp(E_k, flat_slice)

    # ── DOS (single-particle state-density curve) ──
    E, dos = compute_dos_triangle(model, nk=nk, nE=nE, band_slice=flat_slice)
    ok, integ, expected = check_dos_sum_rule(
        E, dos, g=model.degeneracy_factor(), nb=nb_flat,
        area_BZ=abs(np.linalg.det(model.reciprocal_vectors)))
    if not ok:
        print(f'  ⚠ DOS sum rule: ∫={integ:.3f} vs g·nb·A/(2π)²={expected:.1f}')

    # ── VHS / Lifshitz markers from DOS peaks ──
    pk = find_local_maxima(E, dos)
    vhs_list = []
    for i in pk:
        ev = float(E[i])
        nu = float(compute_filling(E_k, ev, flat_slice))
        vhs_list.append({'E_vhs': ev, 'E_rel': float(ev - E_cnp), 'nu': nu,
                         'side': 'electron' if ev > E_cnp else 'hole'})

    # ── transition-energy axis from the flat-band manifold ──
    eflat = E_k[:, flat_slice]
    w_max = 1.2 * (eflat.max() - eflat.min())   # covers all intra-flat transitions
    w_q0 = np.linspace(0.0, max(w_max, 1e-4), nw)

    # ── JDOS at q = 0 (all pairs) ──
    _, jdos_q0 = compute_jdos_q0_triangle(
        model, w_q0, nk=nk, band_slice=flat_slice, interband_only=False)
    _trapz = np.trapezoid  # numpy 2.5 removed np.trapz; use trapezoid directly
    print(f'  JDOS(q=0) ∫dω = {float(_trapz(jdos_q0, w_q0)):.4f}')

    # ── JDOS at finite q (transition energy vs q) ──
    area_BZ = abs(np.linalg.det(model.reciprocal_vectors))
    dk = np.sqrt(area_BZ / nk ** 2)
    q_values = (np.arange(1, nq + 1) - 0.5) * dk * q_max_factor
    if do_jdos_q:
        jdos_q = compute_jdos_q_triangle(
            model, q_values, w_q0, nk=nk, form=True,
            band_slice=flat_slice, verbose=False)   # (nw, nq)
    else:
        jdos_q = None

    return dict(theta=theta, E_cnp=E_cnp, E=E, dos=dos, vhs=vhs_list,
                w=w_q0, jdos_q0=jdos_q0, q=q_values, jdos_q=jdos_q,
                nb_flat=nb_flat)


def _plot_summary(results, outdir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'  [plot] skipped ({e})')
        return

    nA = len(results)
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / max(nA - 1, 1)) for i in range(nA)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (1) DOS vs E
    ax = axes[0]
    for r, c in zip(results, colors):
        ax.plot(r['E'] * 1e3, r['dos'], color=c, lw=1.0,
                label=rf'$\theta={r["theta"]:.2f}^\circ$')
        ax.axvline(r['E_cnp'] * 1e3, color=c, ls=':', lw=0.6, alpha=0.6)
        for v in r['vhs']:
            ax.axvline(v['E_vhs'] * 1e3, color=c, ls='--', lw=0.5, alpha=0.5)
    ax.set_xlabel('E (meV)')
    ax.set_ylabel('DOS (states / eV / uc)')
    ax.set_title('TBG DOS vs twist angle')
    ax.legend(fontsize=7)

    # (2) JDOS(q=0) vs transition energy ω
    ax = axes[1]
    for r, c in zip(results, colors):
        ax.plot(r['w'] * 1e3, r['jdos_q0'], color=c, lw=1.0)
    ax.set_xlabel(r'$\omega$ transition energy (meV)')
    ax.set_ylabel('JDOS(q=0) (states / eV / uc)')
    ax.set_title('JDOS at q=0 vs twist angle')

    # (3) JDOS(q, ω) heatmap for the first angle (if computed)
    ax = axes[2]
    r0 = results[0]
    J = r0.get('jdos_q')
    if J is not None:
        ww = r0['w'] * 1e3
        qq = r0['q']
        Jpos = np.where(J > 0, J, np.nan)
        im = ax.pcolormesh(qq, ww, J, shading='auto', cmap='inferno')
        ax.set_xlabel(r'$q$ (Å$^{-1}$)')
        ax.set_ylabel(r'$\omega$ (meV)')
        ax.set_title(rf'JDOS($q,\omega$), $\theta={r0["theta"]:.2f}^\circ$')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'JDOS(q,ω) skipped\n(use --jdos-q)',
                ha='center', va='center', transform=ax.transAxes)

    fig.tight_layout()
    fig_dir = os.path.join(outdir, 'figures-jdos-angle')
    os.makedirs(fig_dir, exist_ok=True)
    fname = os.path.join(fig_dir, 'jdos-angle-scan.pdf')
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  [plot] {fname}')


def main():
    parser = argparse.ArgumentParser(description='TBG DOS+JDOS angle scan')
    parser.add_argument('--theta', type=float, nargs='+', required=True)
    parser.add_argument('--nk', type=int, default=24)
    parser.add_argument('--n-shells', type=int, default=2,
                        help='BM moiré shells. 2 (~84 bands) is fast and keeps '
                             'the flat pair; 3-4 is more accurate but ~30x slower '
                             'per diagonalisation (TBG has 4*n_shells_bands bands).')
    parser.add_argument('--nq', type=int, default=12)
    parser.add_argument('--nw', type=int, default=400)
    parser.add_argument('--q-max-factor', type=float, default=1.2)
    parser.add_argument('--nE', type=int, default=3000)
    parser.add_argument('--cache-dir', default='.')
    parser.add_argument('--save-cache', action='store_true')
    parser.add_argument('--out', default='jdos-angle-scan.npz')
    parser.add_argument('--plot', action='store_true', default=True)
    parser.add_argument('--no-plot', dest='plot', action='store_false')
    parser.add_argument('--jdos-q', dest='do_jdos_q', action='store_true', default=True,
                        help='Compute finite-q JDOS(q,ω) grid (slower).')
    parser.add_argument('--no-jdos-q', dest='do_jdos_q', action='store_false')
    args = parser.parse_args()

    print('=' * 64)
    print('  TBG DOS + JDOS angle scan (Lifshitz / vHS probe)')
    print('=' * 64)
    print(f'  theta: {args.theta}')
    print(f'  nk={args.nk}, nq={args.nq}, nw={args.nw}, nE={args.nE}')

    results = []
    for theta in args.theta:
        print(f'\n-- theta = {theta} --')
        t0 = time.time()
        r = analyze_theta(theta, nk=args.nk, n_shells=args.n_shells,
                          nq=args.nq, nw=args.nw,
                          q_max_factor=args.q_max_factor, nE=args.nE,
                          cache_dir=args.cache_dir, save_cache=args.save_cache,
                          do_jdos_q=args.do_jdos_q)
        print(f'  CNP: {r["E_cnp"]*1e3:.3f} meV')
        for v in r['vhs']:
            print(f'  VHS [{v["side"]:7s}]: E={v["E_vhs"]*1e3:+8.3f} meV  '
                  f'nu={v["nu"]:+.4f}')
        print(f'  time: {time.time()-t0:.1f}s')
        results.append(r)

    # ── PRESERVE AXIS DATA BEFORE any visualization (requirement) ──
    E_all = np.stack([r['E'] for r in results])          # (nA, nE)
    dos_all = np.stack([r['dos'] for r in results])      # (nA, nE)
    w_all = np.stack([r['w'] for r in results])          # (nA, nw)
    jdos_q0_all = np.stack([r['jdos_q0'] for r in results])   # (nA, nw)
    E_cnp_arr = np.array([r['E_cnp'] for r in results])
    q_values = results[0]['q']
    maxv = max(len(r['vhs']) for r in results)
    vhs_E = np.full((len(results), maxv), np.nan)
    vhs_nu = np.full((len(results), maxv), np.nan)
    for a, r in enumerate(results):
        for b, v in enumerate(r['vhs']):
            vhs_E[a, b] = v['E_vhs']
            vhs_nu[a, b] = v['nu']

    save_kwargs = dict(
        thetas=np.array([r['theta'] for r in results]),
        nk=args.nk, n_shells=args.n_shells, nq=args.nq, nw=args.nw,
        E=E_all, dos=dos_all, E_cnp=E_cnp_arr,
        vhs_E=vhs_E, vhs_nu=vhs_nu,
        w=w_all, jdos_q0=jdos_q0_all,
        q=q_values)
    if args.do_jdos_q:
        jdos_q_all = np.stack([r['jdos_q'] for r in results])   # (nA, nw, nq)
        save_kwargs['jdos_q'] = jdos_q_all

    np.savez(args.out, **save_kwargs)
    print(f'\n[save] axis data -> {args.out}')
    print(f'       DOS curves : E({E_all.shape[1]}) × {E_all.shape[0]} angles')
    print(f'       JDOS(q=0)  : ω({w_all.shape[1]}) × {w_all.shape[0]} angles')
    if args.do_jdos_q:
        print(f'       JDOS(q)    : ω({w_all.shape[1]}) × q({len(q_values)}) '
              f'× {jdos_q_all.shape[0]} angles')

    if args.plot:
        _plot_summary(results, args.cache_dir)


if __name__ == '__main__':
    main()
