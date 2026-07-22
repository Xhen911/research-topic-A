"""
Response package — remaining L3/L4/L5 modules.

After the L2 refactor (PR-3), only dielectric.py (L3/L4) and
q_convergence_test.py (L5) remain here.  They will be moved to
their final layer directories in PR-4 and PR-5 respectively.
"""

from .dielectric import (
    coulomb_2d,
    rpa_response,
    dielectric_function,
    energy_loss_function,
)

__all__ = [
    'coulomb_2d',
    'rpa_response',
    'dielectric_function',
    'energy_loss_function',
]
