#!/usr/bin/env python
"""
tbg_jdos_angle_scan.py — Multi-angle TBG probe: DOS + JDOS(q=0) (Lifshitz / vHS scan)
=====================================================================================

For each twist angle θ this program evaluates two single-particle *spectra*
(1D lines, directly comparable to one another):

  * single-particle DOS    -> energy axis                 E   (eV)
  * JDOS at q = 0          -> transition-energy axis       ω   (eV)

JDOS(q=0) is the optical joint density of states (vertical interband
transitions, q→0), so it reads like a DOS but on the *transition-energy*
axis — the natural "single spectrum line" companion to the DOS.  (Finite-q
JDOS(q,ω), which maps to finite-momentum loss / plasmon scattering, is not
included: it has no clean 1D reading and needs a heatmap, not a line.)

The DOS line duplicates what scripts/lifshitz_scan.py already produces, so
pass ``--no-dos`` to compute **only JDOS(q=0)** (a pure JDOS probe).  The
saved ``.npz`` then holds just the JDOS transition-energy axes.

ALL raw axis data — the DOS curves and/or the JDOS(q=0) spectra — are written
to a single ``.npz`` file *before* any plotting, so the numerical results are
preserved for later re-visualisation or downstream analysis.  A summary plot
(DOS vs E and JDOS(q=0) vs ω across angles, with VHS/Lifshitz markers, or just
the JDOS line in ``--no-dos`` mode) is produced afterwards.

It extends scripts/lifshitz_scan.py (DOS / VHS batch scan) with the
triangle-method JDOS transition-energy spectrum from src.propagators.jdos.

Usage
-----
    # full Lifshitz/vHS angle scan (DOS + JDOS)
    python scripts/tbg_jdos_angle_scan.py --theta 0.80 0.99 1.05 1.08 1.16 1.47 \
        --nk 24 --nw 400 --save-cache --plot

    # pure JDOS(q=0) probe (no DOS; lifshitz_scan already covers DOS)
    python scripts/tbg_jdos_angle_scan.py --theta 0.80 0.99 1.05 --no-dos \
        --nk 24 --nw 400 --save-cache --plot

    # quick test
    python scripts/tbg_jdos_angle_scan.py --theta 1.05 1.16 --nk 16 --nw 200
"""

import numpy as np
import os, sys, argparse, time

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bands import BistritzMacDonaldTBG
from src.core.cache import CachedModel
from src.propagators.dos import compute_dos_triangle, check_dos_sum_rule
from src.propagators.jdos import (
    compute_jdos_q0_triangle, check_jdos_sum_rule)
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


def _build_or_load(model, theta, nk, n_shells, cache_dir, save_cache):
    """Return (CachedModel, E_k flat-bands slice, flat-band energy array)."""
    ttag = f'theta{theta:.2f}'.replace('.', 'p')
    cache_path = os.path.join(cache_dir, f'eig-cache-{ttag}-nk{nk}.npz')
    if os.path.exists(cache_path):
        print(f'  [cache] loading {cache_path}')
        cache = CachedModel.load(cache_path)
    else:
        cache = CachedModel(model, nk=nk, nb_cache=32)
        if save_cache:
            cache.save(cache_path)
    E_k = cache.E_k
    nb = E_k.shape[1]
    half = nb // 2
    flat_slice = slice(half - 1, half + 1)
    return cache, E_k, flat_slice


def analyze_theta(theta, nk=24, n_shells=2, nw=400,
                  nE=3000, cache_dir='.', save_cache=False, do_dos=True):
    """JDOS(q=0) analysis for one twist angle, optionally with the DOS curve.

    With ``do_dos=True`` (default) it also returns the DOS curve and the
    VHS/Lifshitz markers derived from DOS peaks.  With ``do_dos=False`` it
    computes only JDOS(q=0) — a pure JDOS probe (DOS is already covered by
    scripts/lifshitz_scan.py).

    Returns dict with JDOS(q=0) spectrum, CNP, and (if do_dos) DOS curve +
    VHS/Lifshitz markers.  All returned spectra are 1D.
    """
    model = BistritzMacDonaldTBG(theta=theta, n_shells=n_shells)
    cache, E_k, flat_slice = _build_or_load(
        model, theta, nk, n_shells, cache_dir, save_cache)
    nb_flat = flat_slice.stop - flat_slice.start

    E_cnp = compute_cnp(E_k, flat_slice)

    E = dos = None
    vhs_list = []
    if do_dos:
        # ── DOS (single-particle state-density curve) ──
        E, dos = compute_dos_triangle(model, nk=nk, nE=nE, band_slice=flat_slice)
        ok, integ, expected = check_dos_sum_rule(
            E, dos, g=model.degeneracy_factor(), nb=nb_flat,
            area_BZ=abs(np.linalg.det(model.reciprocal_vectors)))
        if not ok:
            print(f'  ⚠ DOS sum rule: ∫={integ:.3f} vs g·nb·A/(2π)²={expected:.1f}')

        # ── VHS / Lifshitz markers from DOS peaks ──
        pk = find_local_maxima(E, dos)
        for i in pk:
            ev = float(E[i])
            nu = float(compute_filling(E_k, ev, flat_slice))
            vhs_list.append({'E_vhs': ev, 'E_rel': float(ev - E_cnp), 'nu': nu,
                             'side': 'electron' if ev > E_cnp else 'hole'})

    # ── transition-energy axis from the flat-band manifold ──
    eflat = E_k[:, flat_slice]
    w_max = 1.2 * (eflat.max() - eflat.min())   # covers all intra-flat transitions
    w_q0 = np.linspace(0.0, max(w_max, 1e-4), nw)

    # ── JDOS at q = 0 (vertical interband transitions -> optical JDOS) ──
    _, jdos_q0 = compute_jdos_q0_triangle(
        model, w_q0, nk=nk, band_slice=flat_slice, interband_only=False)
    _trapz = np.trapezoid  # numpy 2.5 removed np.trapz; use trapezoid directly
    print(f'  JDOS(q=0) ∫dω = {float(_trapz(jdos_q0, w_q0)):.4f}')

    return dict(theta=theta, E_cnp=E_cnp, E=E, dos=dos, vhs=vhs_list,
                w=w_q0, jdos_q0=jdos_q0, nb_flat=nb_flat, do_dos=do_dos)


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
    has_dos = results[0].get('dos') is not None

    if has_dos:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(7, 5))
        axes = [axes]

    # (1) DOS vs E  (only when DOS was computed)
    if has_dos:
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

    # JDOS(q=0) vs transition energy ω  (optical joint DOS — a 1D line)
    ax = axes[-1]
    for r, c in zip(results, colors):
        ax.plot(r['w'] * 1e3, r['jdos_q0'], color=c, lw=1.0,
                label=rf'$\theta={r["theta"]:.2f}^\circ$')
    ax.set_xlabel(r'$\omega$ transition energy (meV)')
    ax.set_ylabel('JDOS(q=0) (states / eV / uc)')
    ax.set_title(r'JDOS at q=0 (optical joint DOS) vs twist angle'
                 + ('  [--no-dos]' if not has_dos else ''))
    ax.legend(fontsize=7)

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
    parser.add_argument('--nw', type=int, default=400)
    parser.add_argument('--nE', type=int, default=3000)
    parser.add_argument('--cache-dir', default='.')
    parser.add_argument('--save-cache', action='store_true')
    parser.add_argument('--out', default='jdos-angle-scan.npz')
    parser.add_argument('--plot', action='store_true', default=True)
    parser.add_argument('--no-plot', dest='plot', action='store_false')
    parser.add_argument('--no-dos', dest='do_dos', action='store_false',
                        help='Skip the DOS curve (scripts/lifshitz_scan.py already '
                             'covers DOS). Compute only JDOS(q=0) — a pure JDOS probe.')
    args = parser.parse_args()

    print('=' * 64)
    print('  TBG DOS + JDOS(q=0) angle scan (Lifshitz / vHS probe)')
    print('=' * 64)
    print(f'  theta: {args.theta}')
    print(f'  nk={args.nk}, nw={args.nw}, nE={args.nE}, dos={args.do_dos}')

    results = []
    for theta in args.theta:
        print(f'\n-- theta = {theta} --')
        t0 = time.time()
        r = analyze_theta(theta, nk=args.nk, n_shells=args.n_shells,
                          nw=args.nw, nE=args.nE,
                          cache_dir=args.cache_dir, save_cache=args.save_cache,
                          do_dos=args.do_dos)
        print(f'  CNP: {r["E_cnp"]*1e3:.3f} meV')
        if args.do_dos:
            for v in r['vhs']:
                print(f'  VHS [{v["side"]:7s}]: E={v["E_vhs"]*1e3:+8.3f} meV  '
                      f'nu={v["nu"]:+.4f}')
        print(f'  time: {time.time()-t0:.1f}s')
        results.append(r)

    # ── PRESERVE AXIS DATA BEFORE any visualization (requirement) ──
    w_all = np.stack([r['w'] for r in results])          # (nA, nw)
    jdos_q0_all = np.stack([r['jdos_q0'] for r in results])   # (nA, nw)
    E_cnp_arr = np.array([r['E_cnp'] for r in results])
    save_kwargs = dict(
        thetas=np.array([r['theta'] for r in results]),
        nk=args.nk, n_shells=args.n_shells, nw=args.nw, do_dos=args.do_dos,
        E_cnp=E_cnp_arr, w=w_all, jdos_q0=jdos_q0_all)

    if args.do_dos:
        E_all = np.stack([r['E'] for r in results])      # (nA, nE)
        dos_all = np.stack([r['dos'] for r in results])  # (nA, nE)
        maxv = max(len(r['vhs']) for r in results)
        vhs_E = np.full((len(results), maxv), np.nan)
        vhs_nu = np.full((len(results), maxv), np.nan)
        for a, r in enumerate(results):
            for b, v in enumerate(r['vhs']):
                vhs_E[a, b] = v['E_vhs']
                vhs_nu[a, b] = v['nu']
        save_kwargs.update(E=E_all, dos=dos_all, vhs_E=vhs_E, vhs_nu=vhs_nu)

    np.savez(args.out, **save_kwargs)
    print(f'\n[save] axis data -> {args.out}')
    if args.do_dos:
        print(f'       DOS curves : E({E_all.shape[1]}) × {E_all.shape[0]} angles')
    print(f'       JDOS(q=0)  : ω({w_all.shape[1]}) × {w_all.shape[0]} angles')

    if args.plot:
        _plot_summary(results, args.cache_dir)


if __name__ == '__main__':
    main()
