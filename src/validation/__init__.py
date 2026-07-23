"""L5 Validation — benchmarks, convergence tests, and model checks."""

from .dirac_benchmarks import (
    dirac_dos_analytical,
    dirac_jdos_analytical,
    dirac_optical_jdos_analytical,
)
from .vhs_analysis import find_vhs_peaks, find_vhs_derivative
from .graphene_dirac_rpa import (
    chi0_doped, chi0_undoped, chi0_static_closedform,
    rpa_dielectric, plasmon_freq, fsum_th, dos_at_EF,
)

__all__ = [
    'dirac_dos_analytical',
    'dirac_jdos_analytical',
    'dirac_optical_jdos_analytical',
    'find_vhs_peaks',
    'find_vhs_derivative',
    'chi0_doped',
    'chi0_undoped',
    'chi0_static_closedform',
    'rpa_dielectric',
    'plasmon_freq',
    'fsum_th',
    'dos_at_EF',
]
