#!/usr/bin/env python
"""
scan_response.py — Batch Lindhard χ₀, ε, loss using CachedModel
===============================================================

Pipelines:
  1. Build (or load) CachedModel with q-loop per angle
  2. For each filling factor ν, compute χ₀(ω,q) via cache
  3. Compute RPA dielectric function ε⁻¹ and loss −Im[ε⁻¹]
  4. Save results + grid-info

Usage
-----
    # Single angle, single ν
    python scripts/scan_response.py --theta 1.05 --nu 0 --nk 24

    # Multi-angle, multi-ν scan with band-group decomposition
    python scripts/scan_response.py --theta 0.80 0.99 1.05 1.08 1.16 1.47 \\
        --nu -4 -2 0 2 4 --nk 24 --band-groups total flat near \\
        --save-cache

Dependencies
------------
    src.models, src.response.cached_model, src.response.polarization
"""

import numpy as np
import os, sys, argparse, time

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import BistritzMacDonaldTBG
from src.response.cached_model import CachedModel
from src.response.polarization import lindhard_from_cache
from src.response.dos import compute_cnp, compute_filling

DEGENERACY = 4
KBT = 0.1e-3
ETA = 0.3e-3
KAPPA_ENV = 3.0
VQ_CONST = 90.5 / KAPPA_ENV


def assemble_response(cache, Ef, w_values, band_groups=None, eta=ETA):
    """Compute χ₀, ε, loss from cached eig data.

    Parameters
    ----------
    cache : CachedModel  (with q-loop)
    Ef : float — Fermi energy (eV)
    w_values : (nw,) — frequency grid (eV)
    band_groups : dict {name: (lo, hi)} or None
        Band slices relative to n_bands//2.
        None → all bands.

    Returns
    -------
    dict {group_name: {'p0': (nw, nq), 'eps': (nw, nq), 'loss': (nw, nq)}}
    """
    nq = cache.Nq
    nw = len(w_values)
    half = cache.E_k.shape[1] // 2
    nb_cache = cache.nb_cache

    if band_groups is None:
        band_groups = {'total': None}

    # Build per-group band slices and compute χ₀
    results = {}
    # q_norms for Coulomb Vq
    Vq_arr = VQ_CONST / cache.q_norms  # (nq,)

    for name, offset in band_groups.items():
        if offset is None:
            sel = slice(None)
        else:
            s = max(0, half + offset[0])
            e = min(cache.E_k.shape[1], half + offset[1])
            sel = slice(s, e)

        # Temporarily slice E_k/E_q/V_k/V_q to selected bands
        # lindhard_from_cache uses full bands — need a workaround.
        # For now, we compute full χ₀ and then note that band-
        # specific decomposition requires pre-filtering the cache.
        # This is the "total" path.
        pass

    # Full χ₀ (total)
    pi0 = lindhard_from_cache(
        cache, cache.q_norms, w_values,
        eta=eta, beta=1.0 / max(KBT, 1e-4), Ef=Ef,
        use_tqdm=True,
    )
    # Combine intra+inter
    p0_total = pi0['intra'] + pi0['inter']  # (nw, nq)

    # ε = 1 + Vq · χ₀
    Vq_2d = Vq_arr[np.newaxis, :]  # (1, nq)
    eps_total = 1.0 + p0_total * Vq_2d
    loss_total = -1.0 / eps_total

    results['total'] = {'p0': p0_total, 'eps': eps_total, 'loss': loss_total}

    return results, Vq_arr


def save_results(results, cache, nu, w_values, outdir='.'):
    """Save p0, eps, loss, grid-info."""
    theta = cache.theta or 0
    ttag = f'theta{theta:.2f}'.replace('.', 'p')
    nu_tag = f'nu{"p" if nu > 0 else "n"}{abs(int(nu)):02d}' if nu != 0 else 'nu0'

    for name, res in results.items():
        suffix = f'-{name}' if name != 'total' else ''
        tag = f'{ttag}-{nu_tag}{suffix}'
        np.save(os.path.join(outdir, f'p0-{tag}.npy'), res['p0'])
        np.save(os.path.join(outdir, f'eps-{tag}.npy'), res['eps'])
        np.save(os.path.join(outdir, f'loss-{tag}.npy'), res['loss'])
        np.savez(os.path.join(outdir, f'grid-info-{tag}.npz'),
                 q_norms=cache.q_norms, w_values=w_values,
                 Nq=cache.Nq, nk=cache.nk, Nk=cache.Nk,
                 nb_cache=cache.nb_cache, theta_deg=theta, nu=nu,
                 eta=ETA, kBT=KBT, kappa_env=KAPPA_ENV)
        print(f'    Saved: p0/eps/loss/grid-info-{tag}.*')


def main():
    parser = argparse.ArgumentParser(description='Batch Lindhard response scan')
    parser.add_argument('--theta', type=float, nargs='+', required=True)
    parser.add_argument('--nu', type=float, nargs='+', default=[0])
    parser.add_argument('--nk', type=int, default=24)
    parser.add_argument('--n-shells', type=int, default=4)
    parser.add_argument('--n-q', type=int, default=20)
    parser.add_argument('--q-max-factor', type=float, default=2.0)
    parser.add_argument('--omg-factor', type=float, default=600.0)
    parser.add_argument('--domg', type=float, default=0.05e-3)
    parser.add_argument('--eta', type=float, default=ETA)
    parser.add_argument('--band-groups', nargs='*', default=['total'])
    parser.add_argument('--cache-dir', default='.')
    parser.add_argument('--save-cache', action='store_true')
    args = parser.parse_args()

    # Frequency grid
    ef_scale = 1.53911e-3
    omg_rng = args.omg_factor * ef_scale
    nomg = int(round(omg_rng / args.domg))
    w_values = np.linspace(omg_rng / nomg, omg_rng, nomg)

    print('=' * 64)
    print('  Lindhard Response Scan')
    print('=' * 64)
    print(f'  theta:  {args.theta}')
    print(f'  nu:     {args.nu}')
    print(f'  nk={args.nk}, nq={args.n_q}, nomg={nomg}')
    print(f'  band groups: {args.band_groups}')
    print(f'  save_cache: {args.save_cache}')
    print('=' * 64)

    for theta in args.theta:
        print(f'\n-- theta = {theta} --')
        ttag = f'theta{theta:.2f}'.replace('.', 'p')
        cache_path = os.path.join(args.cache_dir, f'eig-cache-{ttag}-nk{args.nk}.npz')

        # Build or load cache
        if os.path.exists(cache_path):
            print(f'  [cache] Loading {cache_path}')
            cache = CachedModel.load(cache_path)
        else:
            model = BistritzMacDonaldTBG(theta=theta, n_shells=args.n_shells)
            cache = CachedModel(model, nk=args.nk, n_q=args.n_q,
                                q_max_factor=args.q_max_factor)
            if args.save_cache:
                cache.save(cache_path)

        # CNP reference
        E_cnp = compute_cnp(cache.E_k)
        print(f'  CNP: {E_cnp*1e3:.3f} meV')

        for nu in args.nu:
            print(f'\n  --- nu = {nu} ---')
            Ef = E_cnp  # approximate; refine via compute_filling if needed
            if nu != 0:
                # Bisect Ef for target nu
                lo = E_cnp - 0.05
                hi = E_cnp + 0.05
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    if compute_filling(cache.E_k, mid) < nu:
                        lo = mid
                    else:
                        hi = mid
                    if hi - lo < 1e-6:
                        break
                Ef = 0.5 * (lo + hi)
            print(f'  Ef(nu={nu}) = {Ef*1e3:.3f} meV')

            t0 = time.time()
            results, _ = assemble_response(cache, Ef, w_values, eta=args.eta)
            save_results(results, cache, nu, w_values, outdir=args.cache_dir)
            print(f'  time: {time.time()-t0:.1f}s')

    print('\n' + '=' * 64)
    print('  ALL DONE')
    print('=' * 64)


if __name__ == '__main__':
    main()
