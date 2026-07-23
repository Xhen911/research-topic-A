# -*- coding: utf-8 -*-
"""sum_rules.py — spectral sum-rule checks (skeleton).

Physics targets (implementation deferred to a dedicated validation PR;
not part of the L5 layer refactor):

1. DOS sum rule (triangle-method DOS, propagators/dos.py):

       ∫ DOS(E) dE = g * nb * A_BZ / (2π)²

   with A_BZ = |det(R)| the moiré BZ area.  This is EXACT for the
   η→0 Lehmann–Taut triangle formula (no broadening).  The regression
   baseline (tests/test_regression_baseline.py::test_dos_sum_rule) and
   the implemented ``check_dos_sum_rule`` in ``propagators/dos.py`` already
   enforce rel. err. < 1e-3 — the assertion is REAL, not deferred.  This
   module is a planned higher-level collector for DOS + JDOS + A(k,ω)
   variants.

2. f-sum rule for the optical conductivity (propagators/kubo.py):

       ∫₀^∞ Re σ_xx(ω) dω  =  (π e² / 2ħ²) · ⟨−T_xx⟩

   NOTE: the universal σ₀ = e²/4ħ absolute calibration is currently
   NOT converged (TB nk=60, η=0.01: inter/σ₀ ≈ 0.06–0.27 — open item
   since 2026-07-22).  The check must REPORT the ratio, not assert 1.

Conventions frozen by the refactor (do not change here):
- npz data format, half-step q grid q_j = (j+½)·Δq
- nb_cache=32 / bs_cache semantics, --no-form diagnostic path
"""


def dos_sum_rule(*args, **kwargs):
    """∫DOS dE vs g·nb·A_BZ/(2π)² — relative deviation.

    Planned signature:
        dos_sum_rule(E_grid, dos, g, nb, A_BZ) -> dict
        returns {"integral": float, "expected": float, "rel_err": float}
    """
    raise NotImplementedError(
        "sum-rule checks are implemented in a dedicated validation PR; "
        "see deliverables/repo-restructure-plan.md"
    )


def f_sum_rule(*args, **kwargs):
    """f-sum rule: ∫ Re σ(ω) dω vs band-energy (kinetic) expectation.

    Planned signature:
        f_sum_rule(w_values, sigma_re, model, Ef) -> dict
        returns {"integral": float, "expected": float, "ratio": float}
    """
    raise NotImplementedError(
        "sum-rule checks are implemented in a dedicated validation PR; "
        "see deliverables/repo-restructure-plan.md"
    )
