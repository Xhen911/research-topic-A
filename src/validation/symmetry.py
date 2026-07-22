# -*- coding: utf-8 -*-
"""symmetry.py — symmetry checks (skeleton).

Targets (implementation deferred to a dedicated validation PR):

1. Particle–hole symmetry of the BM continuum model.  Approximate for
   realistic u ≠ u′ — the check must MEASURE the deviation, not assert 0.
2. ε(q,ω) = ε(−q,ω): inversion / time-reversal symmetry of the
   response.  On the half-step q grid this compares opposite-direction
   points with equal |q| (grid stores direction = K_M).
3. Valley / spin degeneracy: band energies respect degeneracy_factor
   g = 4; flat-pair filling ν ∈ [−4, +4].
"""


def check_particle_hole(*args, **kwargs):
    """Particle–hole deviation of the BM band structure (measure, not assert)."""
    raise NotImplementedError(
        "symmetry checks are implemented in a dedicated validation PR; "
        "see deliverables/repo-restructure-plan.md"
    )


def check_epsilon_q_parity(*args, **kwargs):
    """ε(q,ω) vs ε(−q,ω) on the half-step grid."""
    raise NotImplementedError(
        "symmetry checks are implemented in a dedicated validation PR; "
        "see deliverables/repo-restructure-plan.md"
    )
