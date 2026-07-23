"""
Fast pytest for the TB-hybrid (nested-grid) response.

History
-------
Previously tests/test_graphene_tb_hybrid.py was a verbatim copy of the full
production verify script, which ran a ~9 minute scan on *import* and had no
assertions.  On 2026-07-23 (P1) the heavy scan was moved to
scripts/tb_hybrid_verify.py (run via ``python scripts/tb_hybrid_verify.py``),
and this file was rewritten as a seconds-scale pytest that locks in the two
physics facts the nested-grid tail subtraction is specifically there to guarantee:

  * V0  — Im[chi0] must stay non-positive (Kramers-Kronig / retardation).
          The pre-fix discretization mismatch produced a SPURIOUS POSITIVE
          Im[chi0] of ~ +316 % of the DOS.  A 3 % of DOS threshold catches any
          regression by a factor of ~100.
  * V1b — the static chi0 must be (essentially) real; only a tiny Pauli-blocking
          residual is allowed (< 3 % of DOS).

Light resolution (NK=50, NTH=40) runs in ~13 s per check and is well within the
margins measured during development.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.validation.graphene_tb_hybrid import (      # noqa: E402
    build_model, chi0_tb_hybrid, chi0_to_physical, dos_dirac_model,
)

# Light resolution: nested-grid fix is resolution-independent for V0/V1b.
EF = 2.0
NK = 50
NTH = 40
K_CUT = 2.0
# 3 % of DOS — far above the measured residual (~0.4 %) and far below the
# pre-fix spurious +316 % of DOS, so any regression is caught decisively.
THRESH = 0.03


@pytest.fixture(scope="module")
def model():
    return build_model()


def _phys(model, q, w, eta):
    chi = chi0_tb_hybrid(model, q, w, Ef=EF, Nk=NK, Ntheta=NTH,
                         k_cut_factor=K_CUT, eta=eta, verbose=False)
    return chi0_to_physical(chi)


def test_v0_no_spurious_positive_imag(model):
    """Retardation: Im[chi0] must not become spuriously positive.

    Pre-fix (analytical vs numerical Dirac mismatch) gave +316 % of DOS;
    the all-numerical nested-grid subtraction drives it to <= 0 (measured
    ~ -0.05 % of DOS).  Threshold 3 % of DOS is a decisive guard.
    """
    q = np.linspace(0.05, 0.30, 6)
    w = np.linspace(0.05, 2.0, 40)
    chi = _phys(model, q, w, eta=0.005)
    D = dos_dirac_model(EF) / 2.46 ** 2
    assert np.max(chi.imag) / D < THRESH


def test_v1b_static_imag_vanishes(model):
    """Static chi0 must be real to < 3 % of DOS (Pauli-blocking residual only)."""
    q_st = np.linspace(0.05, 0.60, 12)
    chi_st = chi0_to_physical(
        chi0_tb_hybrid(model, q_st, np.array([1e-4]), Ef=EF,
                       Nk=NK, Ntheta=NTH, k_cut_factor=K_CUT,
                       eta=1e-4, verbose=False)[0]
    )
    D = dos_dirac_model(EF) / 2.46 ** 2
    assert np.max(np.abs(chi_st.imag)) / D < THRESH
