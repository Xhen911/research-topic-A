# -*- coding: utf-8 -*-
"""kk.py — Kramers–Kronig self-consistency checks (skeleton).

Target: verify that Re ε(q,ω) and Im ε(q,ω) (observables/dielectric.py)
satisfy the Kramers–Kronig relations via numerical Hilbert transform:

    Re ε(ω) − 1 = (2/π)  P ∫₀^∞ ω′·Im ε(ω′) / (ω′² − ω²) dω′
    Im ε(ω)     = −(2ω/π) P ∫₀^∞ (Re ε(ω′) − 1) / (ω′² − ω²) dω′

Motivation: catches normalisation / sign errors in the Lindhard → RPA
pipeline that the algebraic identity tests cannot see (e.g. the σ₀
absolute-calibration open item documented 2026-07-22).

Implementation notes for the dedicated PR:
- NumPy 2.0 compat: use getattr(np, "trapezoid", np.trapz).
- Production ω grid is finite (~600·E_F scale): the high-ω tail
  Im ε ~ 1/ω must be modelled analytically, not truncated.
- Principal value: exclude the ω′ = ω singular bin explicitly.
"""


def kk_check_epsilon(*args, **kwargs):
    """KK self-consistency of ε(q,ω) on the production frequency grid.

    Planned signature:
        kk_check_epsilon(w_values, eps_qw) -> dict
        returns {"re_err": float, "im_err": float, "max_rel_dev": float}
    """
    raise NotImplementedError(
        "KK checks are implemented in a dedicated validation PR; "
        "see deliverables/repo-restructure-plan.md"
    )
