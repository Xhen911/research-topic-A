# -*- coding: utf-8 -*-
"""q_convergence_test.py — q->0 convergence test via single-q diagonalisation.
=================================================================================

Instead of building a full q-loop CachedModel, only diagonalises at
k + q_eps for each trial offset, then computes chi(q_eps, omega)
for just that one q point.  This avoids building the entire q-path
and is fast enough for interactive use.

Provides a quantitative convergence metric: relative spectral error
between adjacent epsilon values.

Usage
-----
    from src.response.q_convergence_test import run
    eps, spectra, errors, rec = run(theta=1.05, nk=24, nb_cache=32)

Reference
---------
    Brillouin zone scan path approximation (2026-07-15)
"""

import numpy as np

PI = np.pi


def default_w_values(theta, omg_factor=600.0, domg=0.05e-3):
    """Generate default frequency grid."""
    ef_scale = 1.53911e-3
    omg_rng = omg_factor * ef_scale
    nomg = int(round(omg_rng / domg))
    return np.linspace(omg_rng / nomg, omg_rng, nomg)


def _chi0_single_q(E_k, V_k, E_q, V_q, w_values, Ef, kBT, eta,
                    degeneracy, S_norm, use_form_factor=True):
    """Lindhard chi0 for a single q point, using pre-diagonalised k and k+q.

    Returns intra, inter, total — each (nw,) complex128.
    Uses nb_cache bands for form-factor consistency.
    """
    from .polarization import fermi_dirac
    Nk, nb = V_k.shape[0], V_k.shape[2]  # nb = nb_cache
    nw = len(w_values)

    # Form factor |<m,k+q|n,k>|^2 — optional: disabled at tiny q
    # where BM gauge-fixing is unstable in near-degenerate flat bands.
    if use_form_factor:
        M = np.abs(np.einsum('kbm,kbn->kmn', V_q.conj(), V_k)) ** 2
    else:
        Nk, nb = V_k.shape[0], V_k.shape[2]
        M = np.tile(np.eye(nb), (Nk, 1, 1))  # identity — gauge-free

    # Slice energies to the SAME band range as V_k/V_q (bs_cache)
    half = E_k.shape[1] // 2
    bs = slice(half - nb // 2, half + nb // 2)
    f_k = fermi_dirac(E_k[:, bs], Ef, 1.0 / max(kBT, 1e-4))
    f_q = fermi_dirac(E_q[:, bs], Ef, 1.0 / max(kBT, 1e-4))
    f_diff = f_k[:, :, None] - f_q[:, None, :]
    ediff = E_k[:, bs, None] - E_q[:, None, bs]
    num = -degeneracy / S_norm * f_diff * M

    pi0_intra = np.zeros(nw, dtype=complex)
    pi0_inter = np.zeros(nw, dtype=complex)

    for m in range(nb):
        for n in range(nb):
            denom = -ediff[:, m, n, None] + w_values + 1j * eta
            contrib = np.sum(num[:, m, n, None] / denom, axis=0)
            if m == n:
                pi0_intra += contrib
            else:
                pi0_inter += contrib

    return pi0_intra, pi0_inter, pi0_intra + pi0_inter


def convergence_metric(eps_values, spectra):
    """Pairwise convergence metric, sorted ascending by eps.

    Uses L2-norm of the spectral difference, normalised by
    the geometric mean of the two spectral norms — more robust
    against single-frequency outliers than pointwise-max ratio.

    Returns:
        eps_sorted (n_eps,) — ascending
        errs (n_eps-1,) — in (0, ~1) where 0 = identical, ~1 = fully different
    """
    eps_arr = np.asarray(eps_values)
    order = np.argsort(eps_arr)
    eps_sorted = eps_arr[order]
    chi_total = spectra['total'][order]
    errs = []
    for i in range(len(chi_total) - 1):
        diff = np.linalg.norm(chi_total[i] - chi_total[i + 1])
        n1 = np.linalg.norm(chi_total[i])
        n2 = np.linalg.norm(chi_total[i + 1])
        denom = np.sqrt(max(n1 * n2, 1e-30))
        errs.append(diff / denom)
    return eps_sorted, np.array(errs)



def recommend_offset(eps_values, spectra, tol=1e-3):
    """Recommend a safe q_eps from the convergence plateau.

    Sorts eps ascending, then scans from small to large.
    Returns the largest eps whose spectrum is still within ``tol``
    of the next value, plus a diagnostic boolean ``converged``.

    Returns:
        eps : float — recommended offset
        converged : bool — whether a plateau was found
    """
    eps_sorted, errs = convergence_metric(eps_values, spectra)
    if len(errs) == 0:
        return eps_sorted[0], False
    for i in range(len(errs)):
        if errs[i] > tol:
            if i == 0:
                print(f"  WARNING: no plateau found — all errors > tol ({tol:.0e}).")
                print(f"    Returning smallest eps tested ({eps_sorted[0]:.1e}) as fallback.")
            return eps_sorted[i], (i > 0)
    return eps_sorted[-1], True

def plot_convergence(eps_values, spectra, w_values, errors=None,
                     figsize=(10, 8)):
    """2x2 plot: Im[chi] spectra + convergence error vs epsilon."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None, None

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    n = len(eps_values)
    cmap = plt.cm.viridis
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    for ax, key, title in zip(
        axes[0], ['intra', 'inter'],
        ['Im[chi_intra]', 'Im[chi_inter]'],
    ):
        for i, eps in enumerate(eps_values):
            ax.plot(w_values, -spectra[key][i].imag, color=colors[i],
                    lw=1.0, alpha=0.7,
                    label=f'{eps:.1e}' if i in [0, n // 2, n - 1] else None)
        ax.set_xlabel('omega (eV)')
        ax.set_title(title)
        ax.legend(fontsize=7, title='q_eps')

    ax = axes[1, 0]
    for i, eps in enumerate(eps_values):
        ax.plot(w_values, -spectra['total'][i].imag, color=colors[i],
                lw=1.0, alpha=0.7)
    ax.set_xlabel('omega (eV)')
    ax.set_title('Im[chi_total]')

    ax = axes[1, 1]
    if errors is None:
        errors = convergence_metric(spectra)
    ax.loglog(eps_values[1:], np.maximum(errors, 1e-16), 'ko-', lw=1.5, markersize=6)
    ax.set_xlabel('q_eps (1/A)')
    ax.set_ylabel('rel. error between adjacent eps')
    ax.set_title('Convergence metric')
    ax.grid(True, alpha=0.3)
    ax.axhline(1e-3, color='r', ls='--', lw=0.8, label='1e-3 threshold')
    ax.legend(fontsize=8)

    fig.suptitle('q->0 convergence test (single-q diagonalisation)',
                 fontweight='bold')
    fig.tight_layout()
    return fig, axes


def run(
    theta=1.05,
    nk=24,
    n_shells=4,
    nb_cache=32,
    q_eps_values=None,
    omg_factor=600.0,
    domg=0.05e-3,
    eta=0.3e-3,
    kBT=0.1e-3,
    plot=True,
):
    """Self-contained convergence test — builds k-grid cache only.

    For each trial q_eps, diagonalises model at k + q_eps (single q) and
    computes chi0(omega, q_eps).  No full q-loop needed.

    Parameters
    ----------
    theta : float
    nk : int
    n_shells, nb_cache : model / cache params.
    q_eps_values : list or None
        If None, uses dq_q * [1e-1, 1e-2, 1e-3, 1e-4].
    omg_factor, domg : frequency grid params.
    eta, kBT : physical parameters.
    plot : bool.

    Returns
    -------
    eps_values, spectra, errors, recommended_eps, converged
    """
    from ..models import BistritzMacDonaldTBG
    from .cached_model import CachedModel
    from .dos import compute_cnp

    model = BistritzMacDonaldTBG(theta=theta, n_shells=n_shells)

    # k-grid only, no q-loop
    cache = CachedModel(model, nk=nk, nb_cache=nb_cache, n_q=0)
    w_values = default_w_values(theta, omg_factor=omg_factor, domg=domg)
    Ef = compute_cnp(cache.E_k)
    degeneracy = model.degeneracy_factor()
    S_norm = abs(np.linalg.det(model.reciprocal_vectors))

    hs = model.high_symmetry_points()
    q_dir = hs['K'] / np.linalg.norm(hs['K'])

    # Actual q-spacing in scan_response: q_max / (Nq - 1)
    # (not the local cache.dk estimate, which is ~5x smaller)
    kf = np.linalg.norm(hs['K'])
    q_max = 2.0 * kf
    nq_for_step = 20  # matching scan_response default --n-q
    dq_q = q_max / max(nq_for_step - 1, 1)
    if q_eps_values is None:
        ratios = [1e-1, 1e-2, 1e-3, 1e-4]  # dq_q/10 .. dq_q/10000
        q_eps_values = [dq_q * r for r in ratios]

    n_eps = len(q_eps_values)
    nw = len(w_values)
    spectra = {
        'intra': np.zeros((n_eps, nw), dtype=complex),
        'inter': np.zeros((n_eps, nw), dtype=complex),
        'total': np.zeros((n_eps, nw), dtype=complex),
    }

    print(f'q->0 convergence test: theta={theta}, nk={nk}')
    print(f'  dq_q ~ {dq_q:.4e}')
    print(f'  eps range = [{q_eps_values[0]:.1e}, {q_eps_values[-1]:.1e}]')

    for i, eps in enumerate(q_eps_values):
        q_vec = eps * q_dir
        E_q, V_q = cache.eig_at_q(q_vec)
        intra, inter, total = _chi0_single_q(
            cache.E_k, cache.V_k, E_q, V_q,
            w_values, Ef, kBT, eta, degeneracy, S_norm,
        )
        spectra['intra'][i] = intra
        spectra['inter'][i] = inter
        spectra['total'][i] = total
        print(f'  eps={eps:.1e} done')

    eps_arr = np.array(q_eps_values)
    _, errors = convergence_metric(eps_arr, spectra)
    recommended, converged = recommend_offset(eps_arr, spectra)
    if not converged:
        print(f'  WARNING: no convergence plateau detected above tol')

    print(f'  pairwise errors: {[f"{e:.2e}" for e in errors]}')
    print(f'  recommended q_eps = {recommended:.1e}')

    if plot:
        fig, _ = plot_convergence(eps_arr, spectra, w_values, errors)
        if fig is not None:
            fname = f'q_convergence_theta{theta:.2f}.png'
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            print(f'  plot saved: {fname}')

    return eps_arr, spectra, errors, recommended, converged
