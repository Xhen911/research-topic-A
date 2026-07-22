"""
Validation layer (L5) — self-consistency and physics sanity checks.

Modules
-------
convergence   : q->0 convergence test (moved from response/q_convergence_test.py, PR-5)
model_checks  : model verification suite (moved from debug/model_checks.py, PR-5)
sum_rules     : DOS / f-sum rule checks (skeleton — implementation deferred)
kk            : Kramers-Kronig self-consistency of eps(q,w) (skeleton)
ward          : Ward identity / charge-conservation checks (skeleton)
symmetry      : particle-hole and eps(q)=eps(-q) symmetry checks (skeleton)

Skeletons raise NotImplementedError by design: physics implementations
land in dedicated validation PRs, not in the layer refactor.
"""

from .convergence import (
    convergence_metric,
    default_w_values,
    recommend_offset,
    run,
)
from .model_checks import ModelDebugSuite, verify_kp_models

__all__ = [
    # convergence (moved from response/)
    "convergence_metric",
    "default_w_values",
    "recommend_offset",
    "run",
    # model_checks (moved from debug/)
    "ModelDebugSuite",
    "verify_kp_models",
]
