"""
Regression tests for src/validation/convergence.py plot_convergence.

Locks in the two P2 plot fixes from 2026-07-23:
  * errors=None branch used to call convergence_metric(spectra) missing the
    required eps_values argument -> always TypeError.  Now it calls
    convergence_metric(eps_values, spectra).
  * The convergence error is computed with eps sorted ASCENDING inside
    convergence_metric, but the plot used the raw (descending, input-order)
    eps_values[1:] as the x-axis -> every point sat at the wrong abscissa.
    The plot now uses the ascending sort, matching the error ordering.
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, '.')

from src.validation.convergence import convergence_metric, plot_convergence


def _fake_spectra(eps_values, nw=40, seed=0):
    rng = np.random.default_rng(seed)
    n = len(eps_values)
    return {
        'intra': rng.standard_normal((n, nw)) + 0j,
        'inter': rng.standard_normal((n, nw)) + 0j,
        'total': rng.standard_normal((n, nw)) + 0j,
    }


def test_plot_errors_none_no_crash():
    """The latent errors=None branch must not raise TypeError."""
    eps = np.array([1e-1, 1e-2, 1e-3, 1e-4])  # descending input order
    spectra = _fake_spectra(eps)
    fig, _ = plot_convergence(eps, spectra, np.linspace(0, 1, 40), errors=None)
    assert fig is not None


def test_plot_xaxis_ascending_aligned_with_errors():
    """x-axis of the convergence panel must be ascending and match error order."""
    eps = np.array([1e-1, 1e-2, 1e-3, 1e-4])  # descending input order
    spectra = _fake_spectra(eps)
    _, errors = convergence_metric(eps, spectra)  # errors aligned to ascending eps

    # errors passed (run() path)
    _, axes = plot_convergence(eps, spectra, np.linspace(0, 1, 40), errors=errors)
    xe = axes[1, 1].get_lines()[0].get_xdata()
    assert np.all(np.diff(xe) > 0), f"x-axis not ascending: {list(xe)}"
    np.testing.assert_allclose(xe, np.sort(eps)[1:])

    # errors=None path
    _, axes2 = plot_convergence(eps, spectra, np.linspace(0, 1, 40), errors=None)
    xe2 = axes2[1, 1].get_lines()[0].get_xdata()
    assert np.all(np.diff(xe2) > 0)
    np.testing.assert_allclose(xe2, np.sort(eps)[1:])
