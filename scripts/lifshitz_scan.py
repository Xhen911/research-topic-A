
#!/usr/bin/env python
"""
lifshitz_scan.py — Batch Van Hove singularity & Lifshitz transition analysis
============================================================================

P0 script: multi-angle scan of DOS, VHS detection, and filling factors.
Uses CachedModel to diagonalise once per angle, then re-evaluates
DOS and VHS for different broadening / band selections without
re-diagonalising.

Usage
-----
    python scripts/lifshitz_scan.py --theta 0.80 0.99 1.05 1.08 1.16 1.47 \\
        --method triangle --plot --save-cache

    python scripts/lifshitz_scan.py --theta 1.05 --method gaussian --sigma 0.3e-3

Dependencies
------------
    src.models, src.response.dos, src.response.cached_model
"""

import numpy as np
import os, sys, argparse, time

# Adjust path if run directly
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bands import BistritzMacDonaldTBG
from src.response.cached_model import CachedModel
from src.response.dos import (
    compute_dos, compute_dos_triangle,
    find_vhs_peaks, find_vhs_derivative,
    check_dos_sum_rule,
)
from src.bands.occupations import compute_filling, compute_cnp


def analyze_theta(theta, nk=24, n_shells=4, method='triangle',
                  sigma=0.2e-3, eta=0.05e-3, nE=3000,
                  cache_dir='.', save_cache=False, plot=False):
    """Full Lifshitz analysis for one twist angle.

    Returns dict with keys: theta, E_cnp, E, dos, vhs_list.
    """
    ttag = f'theta{theta:.2f}'.replace('.', 'p')
    cache_path = os.path.join(cache_dir, f'eig-cache-{ttag}-nk{nk}.npz')

    # ── Load or build cached eig ──
    if os.path.exists(cache_path):
        print(f'  [cache] Loading {cache_path}')
        cache = CachedModel.load(cache_path)
    else:
        model = BistritzMacDonaldTBG(theta=theta, n_shells=n_shells)
        cache = CachedModel(model, nk=nk)
        if save_cache:
            cache.save(cache_path)

    E_k = cache.E_k
    half = E_k.shape[1] // 2
    flat_slice = slice(half - 1, half + 1)

    # ── CNP ──
    E_cnp = compute_cnp(E_k, flat_slice)

    # ── DOS ──
    if method == 'triangle':
        model_dummy = cache.model or BistritzMacDonaldTBG(theta=theta)
        E, dos = compute_dos_triangle(model_dummy, nk=nk, nE=nE,
                                       eta=eta, band_slice=flat_slice)
    else:
        model_dummy = cache.model or BistritzMacDonaldTBG(theta=theta)
        E, dos = compute_dos(model_dummy, nk=nk, nE=nE, sigma=sigma,
                             E_k=E_k[:, flat_slice],
                             k_cart=cache.k_cart)

    # ── Sum-rule check ──
    nb_flat = flat_slice.stop - flat_slice.start
    ok, integral, expected = check_dos_sum_rule(
        E, dos, g=model_dummy.degeneracy_factor(), nb=nb_flat,
        area_BZ=abs(np.linalg.det(model_dummy.reciprocal_vectors)))
    if not ok:
        print(f'  ⚠ DOS sum-rule: ∫DdE={integral:.3f} vs g·nb={expected:.1f}')

    # ── VHS ──
    vhs_peaks = find_vhs_peaks(E, dos)
    vhs_deriv = find_vhs_derivative(E, dos)
    dedup_tol = max(3 * (E[1] - E[0]), 0.1e-3)
    all_vhs = []
    for ev in sorted(vhs_peaks + [v['E_vhs'] for v in vhs_deriv]):
        if all(abs(ev - e0) > dedup_tol for e0 in all_vhs):
            all_vhs.append(ev)

    vhs_results = []
    for ev in all_vhs:
        nu = compute_filling(E_k, ev, flat_slice)
        vhs_results.append({
            'E_vhs': ev,
            'E_rel': ev - E_cnp,
            'nu': nu,
            'side': 'electron' if ev > E_cnp else 'hole',
        })

    result = dict(theta=theta, E_cnp=E_cnp, E=E, dos=dos,
                  vhs=vhs_results, method=method, cache=cache)

    # ── Plot ──
    if plot:
        _plot_dos_vhs(result, cache_dir)

    return result


def _plot_dos_vhs(result, outdir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    theta = result['theta']
    E, dos = result['E'], result['dos']
    E_cnp = result['E_cnp']

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(E * 1e3, dos, alpha=0.25, color='C0')
    ax.plot(E * 1e3, dos, color='C0', lw=0.8)
    ax.axvline(E_cnp * 1e3, color='r', ls=':', lw=1, label='CNP (ν=0)')

    for v in result['vhs']:
        ax.axvline(v['E_vhs'] * 1e3, color='gray', ls='--', lw=0.7)
        yp = np.interp(v['E_vhs'], E, dos)
        ax.annotate(rf'$\nu={v["nu"]:+.2f}$', xy=(v['E_vhs']*1e3, yp),
                    xytext=(0, 12), textcoords='offset points',
                    fontsize=8, ha='center',
                    arrowprops=dict(arrowstyle='->', lw=0.5))

    ax.set_xlabel('E (meV)')
    ax.set_ylabel('DOS (states / eV / unit cell)')
    ax.set_title(rf'TBG $\theta={theta:.2f}^\circ$ — DOS & Van Hove singularities')
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()

    from pathlib import Path
    fig_dir = Path(outdir) / 'figures-lifshitz'
    fig_dir.mkdir(exist_ok=True)
    ttag = f'theta{theta:.2f}'.replace('.', 'p')
    fname = fig_dir / f'dos-vhs-{ttag}.pdf'
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  [plot] {fname}')


def print_summary(results_list):
    print('\n' + '=' * 64)
    print('  Lifshitz transition filling factors')
    print('=' * 64)
    print(f'  {"theta":>7}  {"nu_hole":>10}  {"nu_elec":>10}  '
          f'{"E_h (meV)":>12}  {"E_e (meV)":>12}')
    print('  ' + '-' * 58)
    for r in results_list:
        holes = [v for v in r['vhs'] if v['side'] == 'hole']
        elecs = [v for v in r['vhs'] if v['side'] == 'electron']
        h = holes[0] if holes else {}
        e = elecs[0] if elecs else {}
        print(f'  {r["theta"]:7.2f}  '
              f'{h.get("nu", float("nan")):+.3f}  '
              f'{e.get("nu", float("nan")):+.3f}  '
              f'{h.get("E_rel", float("nan"))*1e3:+.1f}  '
              f'{e.get("E_rel", float("nan"))*1e3:+.1f}')
    print('=' * 64)


def main():
    parser = argparse.ArgumentParser(description='Lifshitz VHS batch scan')
    parser.add_argument('--theta', type=float, nargs='+', required=True)
    parser.add_argument('--nk', type=int, default=24)
    parser.add_argument('--n-shells', type=int, default=4)
    parser.add_argument('--method', choices=['triangle', 'gaussian'], default='triangle')
    parser.add_argument('--sigma', type=float, default=0.2e-3)
    parser.add_argument('--eta', type=float, default=0.05e-3)
    parser.add_argument('--nE', type=int, default=3000)
    parser.add_argument('--cache-dir', default='.')
    parser.add_argument('--save-cache', action='store_true')
    parser.add_argument('--plot', action='store_true', default=True)
    parser.add_argument('--no-plot', dest='plot', action='store_false')
    args = parser.parse_args()

    print('=' * 64)
    print('  Lifshitz Transition Analysis — VHS & Filling Factors')
    print('=' * 64)
    print(f'  theta:  {args.theta}')
    print(f'  method: {args.method},  nk={args.nk},  nE={args.nE}')

    results = []
    for theta in args.theta:
        print(f'\n-- theta = {theta} --')
        t0 = time.time()
        try:
            r = analyze_theta(
                theta, nk=args.nk, n_shells=args.n_shells,
                method=args.method, sigma=args.sigma, eta=args.eta,
                nE=args.nE, cache_dir=args.cache_dir,
                save_cache=args.save_cache, plot=args.plot)
            print(f'  CNP: {r["E_cnp"]*1e3:.3f} meV')
            for v in r['vhs']:
                print(f'  VHS [{v["side"]:7s}]: E={v["E_vhs"]*1e3:+8.3f} meV  '
                      f'nu={v["nu"]:+.4f}')
            print(f'  time: {time.time()-t0:.1f}s')
            results.append(r)
        except Exception as e:
            print(f'  [skip] {e}')

    if results:
        print_summary(results)


if __name__ == '__main__':
    main()
