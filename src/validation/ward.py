# -*- coding: utf-8 -*-
"""ward.py — Ward-identity / charge-conservation checks (skeleton).

Target: verify gauge invariance of the response pipeline, i.e. the
longitudinal Ward identity at operator level

    ω·ρ(q,ω) − q·J(q,ω) = 0

Consequences to be tested numerically (dedicated PR):

1. Static limit: χ₀(q→0, ω=0) reduces to the compressibility
   (DOS at E_F) — the q→0 behaviour of propagators/lindhard.py.
   With the half-step grid, q=0 is never hit directly; extrapolate.
2. Density vertex (bands/vertices.py): M = |⟨m,k+q|n,k⟩|² → δ_mn
   as q→0, i.e. interband weight ∝ q².  Established 2026-07-17;
   this test freezes that behaviour as a Ward-consistency check.
3. RPA preserves the Ward identity by construction (scalar density–
   density interaction).  TDDFT / BSE kernels (interactions/*.py,
   skeletons) must pass this check before any production use.
"""


def ward_check(*args, **kwargs):
    """Ward-identity consistency for density response at small q.

    Planned signature:
        ward_check(cache, q_small, Ef) -> dict
        returns {"compressibility_ratio": float, "vertex_q2_exponent": float}
    """
    raise NotImplementedError(
        "Ward-identity checks are implemented in a dedicated validation PR; "
        "see deliverables/repo-restructure-plan.md"
    )
