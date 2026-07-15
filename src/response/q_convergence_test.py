# -*- coding: utf-8 -*-
"""q_convergence_test.py — Verify q-offset ε convergence for chi0(q,omega) at small q.
============================================================================

Sweep q_eps over 3-4 orders of magnitude and check if the response
spectrum at the smallest q point converges to a plateau.  If the plateau
is flat, the offset is in the "safe" window — sampling the genuine
q→0+ limit rather than finite-q physics or floating-point noise.

Usage
-----
    # Quick test on a CachedModel
    python -m src.response.q_convergence_test cache.npz

    # Sweep and plot
    from src.response.q_convergence_test import run_convergence_test
    eps_values, spectra = run_convergence_test(cache, w_values)

Reference
---------
    Brillouin zone scan path approximation (2026-07-15)
"""

import numpy as np

def run_convergence_test(cache, w_values, eta=0.3e-3,
                         q_eps_values=None, n_q_eps=8, Ef=0.0,
                         kBT=0.1e-3):
    """Sweep q_eps over 3-4 decades and return chi0(q_first, omega) for each.

    Parameters
    ----------
    cache : CachedModel — must have q-loop (n_q > 0) and a model attribute.
    w_values : (nw,) — frequency grid (eV).
    eta : float — Lindhard broadening.
    q_eps_values : list of float or None
        If None, auto-generated: [step/10^4, step/10^3, step/10^2, step/10, step, step*10]
        where step = cache.q_norms[0] (first q magnitude).
    n_q_eps : int — number of eps values (ignored if q_eps_values given).
    Ef : float — Fermi energy.
    kBT : float — temperature (eV).

    Returns
    -------
    eps_values : (n,)
    spectra : dict
        {'intra': (n, nw,), 'inter': (n, nw), 'total': (n, nw)}
        where total = intra + inter, all complex128.
    """
    from .polarization import lindhard_from_cache
    from .dos import compute_cnp

    step = cache.q_norms[-1]  # smallest q magnitude in the mesh
    if q_eps_values is None:
        q_eps_values = [step / 10**p for p in range(4, 0, -1)] + [step * 10**p for p in range(1, 3)]
        q_eps_values = sorted(q_eps_values)
        # Filter: keep only values < step/2 (physical offset should be tiny)
        q_eps_values = [v for v in q_eps_values if v < step / 2]

    if Ef is None:
        Ef = compute_cnp(cache.E_k) if hasattr(cache, 'E_k') else 0.0

    n = len(q_eps_values)
    nw = len(w_values)
    spectra = {
        'intra': np.zeros((n, nw), dtype=complex),
        'inter': np.zeros((n, nw), dtype=complex),
        'total': np.zeros((n, nw), dtype=complex),
        'eps_values': np.array(q_eps_values),
    }

    for i, eps in enumerate(q_eps_values):
        pi0 = lindhard_from_cache(cache, cache.q_norms, w_values,
                                   eta=eta, beta=1.0/max(kBT, 1e-4),
                                   Ef=Ef, q_eps=eps)
        spectra['intra'][i] = pi0['intra'][:, -1]  # smallest q
        spectra['inter'][i] = pi0['inter'][:, -1]
        spectra['total'][i] = pi0['intra'][:, -1] + pi0['inter'][:, -1]

    return np.array(q_eps_values), spectra


def plot_convergence(eps_values, spectra, w_values, figsize=(10, 6)):
    """Plot convergence diagnostic: Im[chi0(w, q_first)] vs frequency, colored by eps.

    Returns fig, axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None, None

    n = len(eps_values)
    cmap = plt.cm.viridis
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True, sharex=True)

    for title, key, ax in zip(
        ['Im[chi0_intra]', 'Im[chi0_inter]', 'Im[chi0_total]'],
        ['intra', 'inter', 'total'],
        axes,
    ):
        for i, eps in enumerate(eps_values):
            ax.plot(w_values, -spectra[key][i].imag, color=colors[i],
                    lw=1.0, alpha=0.7,
                    label=f'{eps:.2e}' if i in [0, n//2, n-1] else None)
        ax.set_xlabel('omega (eV)')
        ax.set_title(title)
        ax.legend(fontsize=7, title='q_eps')

    axes[0].set_ylabel('-Im[chi]')
    fig.suptitle('q->0 convergence test: sweep epsilon offset over 3-4 decades',
                 fontweight='bold')
    fig.tight_layout()
    return fig, axes


def recommend_offset(eps_values, spectra, tol=1e-3):
    """Recommend a safe q_eps value based on plateau behaviour.

    Looks for the largest eps that gives spectrum within tol
    of the spectrum at the smallest eps tested.

    Returns recommended q_eps (float).
    """
    ref = spectra['total'][0]  # smallest eps
    norm = np.max(np.abs(ref))
    if norm < 1e-30:
        return eps_values[0]

    for i in range(1, len(eps_values)):
        diff = np.max(np.abs(spectra['total'][i] - ref))
        if diff / max(norm, 1e-30) > tol:
            return eps_values[max(0, i - 1)]

    return eps_values[-1]  # all converged — use largest safe value
