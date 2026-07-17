#!/usr/bin/env python
"""
scan_response.py — Batch Lindhard chi0, epsilon, loss using CachedModel
======================================================================

Pipeline:
  1. Build (or load) CachedModel with q-loop per angle
  2. For each filling factor nu, compute chi0(omega,q) via cache
  3. Compute RPA dielectric function epsilon^-1 and loss -Im[epsilon^-1]
  4. Save results + grid-info

q-mesh convention (July 2026 revision)
--------------------------------------
Half-step offset grid: q_j = (j + 1/2) * Δq, no strict q=0, no q_eps.
The q_norms saved in grid-info is exactly the q used in computation — no
hidden offset between numerical values and display.

Usage
-----
    # Single angle, nu=-2 (partial filling → Fermi surface → plasmon):
    python scripts/scan_response.py --theta 1.05 --nu -2 --nk 24 --save-cache

    # Batch multi-angle + multi-filling:
    python scripts/scan_response.py --theta 0.80 0.99 1.05 1.08 1.16 1.47 \\
        --nu -2 0 2 --nk 24 --save-cache

    # Form-factor-ones baseline (diagnostic):
    python scripts/scan_response.py --theta 1.05 --nu -2 --nk 24 \\
        --no-form --save-cache
"""

import numpy as np
import os, sys, argparse, time, warnings

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


def assemble_response(cache, Ef, w_values, eta=ETA, form=True):
    """Compute chi0, epsilon, loss from cached eig.

    Parameters
    ----------
    form : bool
        Whether to use form factor |<k+q|k>|^2.  False = all-ones.
    """
    pi0 = lindhard_from_cache(
        cache, cache.q_norms, w_values,
        eta=eta, beta=1.0 / max(KBT, 1e-4), Ef=Ef,
        form=form, use_tqdm=True,
    )
    p0 = pi0['intra'] + pi0['inter']      # (nw, nq)

    Vq_arr = VQ_CONST / cache.q_norms      # (nq,)
    Vq_2d = Vq_arr[np.newaxis, :]          # (1, nq)
    eps = 1.0 + p0 * Vq_2d
    loss = -1.0 / eps

    return {'p0': p0, 'eps': eps, 'loss': loss}, Vq_arr


def save_results(results, cache, nu, w_values, form=True, outdir='.'):
    """Save p0, eps, loss, grid-info."""
    theta = cache.theta or 0
    ttag = f'theta{theta:.2f}'.replace('.', 'p')
    nu_tag = f'nu{"p" if nu>0 else "n"}{abs(int(nu)):02d}' if nu != 0 else 'nu0'
    ff = 'ff' if form else 'noff'

    tag = f'{ttag}-{nu_tag}-{ff}'
    np.save(os.path.join(outdir, f'p0-{tag}.npy'), results['p0'])
    np.save(os.path.join(outdir, f'eps-{tag}.npy'), results['eps'])
    np.save(os.path.join(outdir, f'loss-{tag}.npy'), results['loss'])
    np.savez(os.path.join(outdir, f'grid-info-{tag}.npz'),
             q_norms=cache.q_norms, w_values=w_values,
             Nq=cache.Nq, nk=cache.nk, Nk=cache.Nk,
             nb_cache=cache.nb_cache, theta_deg=theta, nu=nu,
             eta=ETA, kBT=KBT, kappa_env=KAPPA_ENV,
             use_form=form)
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
    parser.add_argument('--cache-dir', default='.')
    parser.add_argument('--nb-cache', type=int, default=32,
                        help='Bands to cache around CNP (default 32)')
    parser.add_argument('--save-cache', action='store_true')
    parser.add_argument('--q-eps', type=float, default=None,
                        help='DEPRECATED: half-step grid removes the need for q_eps. '
                             'Kept for backward compat; ignored if set.')
    parser.add_argument('--form', dest='use_form', action='store_true',
                        default=True,
                        help='Use real form factor |<k+q|k>|^2 (default)')
    parser.add_argument('--no-form', dest='use_form', action='store_false',
                        help='Use all-ones form factor (diagnostic baseline)')
    args = parser.parse_args()

    # -- deprecation warning for q_eps --
    if args.q_eps is not None and args.q_eps != 0.0:
        warnings.warn(
            f"--q-eps {args.q_eps} is deprecated. "
            f"CachedModel now uses half-step grid with no offset.",
            DeprecationWarning, stacklevel=2,
        )

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
    print(f'  form_factor: {args.use_form}')
    print(f'  q_mesh: half-step grid (q_j = (j+1/2)*Δq)')
    print(f'  save_cache: {args.save_cache}')
    print('=' * 64)

    for theta in args.theta:
        print(f'\n-- theta = {theta} --')
        ttag = f'theta{theta:.2f}'.replace('.', 'p')
        cache_path = os.path.join(args.cache_dir,
                                   f'eig-cache-{ttag}-nk{args.nk}.npz')

        if os.path.exists(cache_path):
            print(f'  [cache] Loading {cache_path}')
            cache = CachedModel.load(cache_path)
            print(f'  q_norms = [{cache.q_norms[0]:.5f}, ..., '
                  f'{cache.q_norms[-1]:.5f}] (Nq={cache.Nq})')
        else:
            model = BistritzMacDonaldTBG(theta=theta, n_shells=args.n_shells)
            cache = CachedModel(model, nk=args.nk, n_q=args.n_q,
                                nb_cache=args.nb_cache,
                                q_max_factor=args.q_max_factor)
            if args.save_cache:
                cache.save(cache_path)

        E_cnp = compute_cnp(cache.E_k)
        print(f'  CNP: {E_cnp*1e3:.3f} meV')

        for nu in args.nu:
            print(f'\n  --- nu = {nu} ---')
            # Bisect Ef for target nu
            lo, hi = E_cnp - 0.05, E_cnp + 0.05
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
            results, _ = assemble_response(cache, Ef, w_values,
                                           eta=args.eta,
                                           form=args.use_form)
            save_results(results, cache, nu, w_values,
                         form=args.use_form, outdir=args.cache_dir)
            print(f'  time: {time.time()-t0:.1f}s')

    print('\n' + '=' * 64)
    print('  ALL DONE')
    print('=' * 64)


if __name__ == '__main__':
    main()
