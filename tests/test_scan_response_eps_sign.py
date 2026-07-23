"""
Regression test for the P0 dielectric-sign bug in scripts/scan_response.py.

History
-------
scan_response.py migrated the epsilon formula from an OLD inline chi0
implementation that used the convention Pi = -chi_ret (positive static) with
``eps = 1 + V*Pi``.  After the chi0 source was switched to
``lindhard_from_cache`` (STANDARD retarded chi0, NEGATIVE static), the sign was
left flipped, giving ``eps = 1 + V*chi0`` -> anti-screened (negative) static
dielectric function for any new production run.

The physically correct RPA screening is ``eps = 1 - V*chi0`` (the same relation
used in src/observables/dielectric.py).  This test anchors the sign with a
seconds-scale BM instance so the convention cannot silently flip back.

Instance matches the numerical proof in deliverables/repo-review-2026-07-23.md:
theta=1.05, nk=4, nb_cache=8, n_q=3 -> static Re eps = +14.87, +3.44, +2.04.
"""

import os
import sys
import numpy as np

import pytest

# Make the repo root importable when run via `python -m pytest tests/`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bands import BistritzMacDonaldTBG          # noqa: E402
from src.core.cache import CachedModel             # noqa: E402
from src.bands.occupations import compute_cnp      # noqa: E402
from scripts.scan_response import assemble_response  # noqa: E402


# Small BM instance: fast to build, enough q-points to probe the sign.
THETA = 1.05
NK = 4
N_SHELLS = 4
NB_CACHE = 8
N_Q = 3


@pytest.fixture(scope="module")
def cache():
    model = BistritzMacDonaldTBG(theta=THETA, n_shells=N_SHELLS)
    c = CachedModel(model, nk=NK, n_q=N_Q, nb_cache=NB_CACHE, q_max_factor=2.0)
    return c


def test_static_epsilon_is_screening_not_antiscreening(cache):
    """Static limit Re eps(q, omega->0) must be > 1 for every q.

    A negative (anti-screened) static dielectric function is unphysical for
    these gapped/compressible BM systems: it would put the plasmon pole at the
    wrong sign of Re chi0 and corrupt the ELF structure.
    """
    E_cnp = compute_cnp(cache.E_k)
    # Static limit: a single very small frequency along with a couple of larger
    # ones so the test also exercises the dynamic branch.
    w_values = np.array([1e-4, 1e-2, 5e-2])

    results, _ = assemble_response(cache, E_cnp, w_values,
                                   eta=0.3e-3, form=True)
    eps = results['eps']  # (nw, nq)

    # eps[0, :] is the static-limit row (smallest omega).
    static_real = eps[0, :].real
    assert static_real.shape[0] == N_Q
    assert np.all(static_real > 1.0), (
        f"Static Re eps must be > 1 (screening), got {static_real}"
    )


def test_epsilon_sign_matches_dielectric_module(cache):
    """Cross-check: assemble_response's eps must equal 1 - V*chi0 directly.

    Reconstructs epsilon from the raw polarisation and the 2D Coulomb kernel
    and compares to the returned eps, guarding against any future sign drift
    inside assemble_response.
    """
    E_cnp = compute_cnp(cache.E_k)
    w_values = np.array([1e-4, 1e-2, 5e-2])

    results, Vq_arr = assemble_response(cache, E_cnp, w_values,
                                         eta=0.3e-3, form=True)
    p0 = results['p0']                 # (nw, nq)
    Vq_2d = Vq_arr[np.newaxis, :]      # (1, nq)
    eps_rebuilt = 1.0 - p0 * Vq_2d

    assert np.allclose(results['eps'], eps_rebuilt, rtol=1e-12, atol=1e-12)
