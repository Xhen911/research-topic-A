"""L2 Propagators — independent-particle response functions."""

from .lindhard import (
    fermi_dirac,
    generate_k_mesh,
    wrap_k_plus_q,
    lindhard_polarization,
    lindhard_from_cache,
    generate_dirac_mesh,
)
from .dos import (
    compute_eigenvalues,
    check_dos_sum_rule,
    compute_dos_triangle,
)
from .triangle_core import triangle_spectrum
from .dos_gaussian import (
    compute_dos,
    compute_dos_vectorized,
    compute_jdos_q0,
    compute_jdos,
    compute_optical_jdos,
)
from .jdos import (
    compute_jdos_q_triangle,
    compute_jdos_q0_triangle,
    check_jdos_sum_rule,
)
from ..core.quadrature import integrate_bz
from ..validation.dirac_benchmarks import (
    dirac_dos_analytical,
    dirac_jdos_analytical,
    dirac_optical_jdos_analytical,
)
from ..validation.vhs_analysis import find_vhs_peaks, find_vhs_derivative
from .kubo import (
    SIGMA_0,
    velocity_matrix_elements,
    optical_conductivity_interband,
    optical_conductivity_intraband,
    optical_conductivity,
    optical_conductivity_xx,
)

__all__ = [
    # lindhard
    'fermi_dirac',
    'generate_k_mesh',
    'wrap_k_plus_q',
    'lindhard_polarization',
    'lindhard_from_cache',
    # dos
    'compute_dos',
    'compute_dos_vectorized',
    'compute_eigenvalues',
    'integrate_bz',
    'dirac_dos_analytical',
    'generate_dirac_mesh',
    'compute_jdos',
    'compute_jdos_q0',
    'compute_optical_jdos',
    'dirac_jdos_analytical',
    'dirac_optical_jdos_analytical',
    'check_dos_sum_rule',
    'compute_dos_triangle',
    'triangle_spectrum',
    'compute_jdos_q_triangle',
    'compute_jdos_q0_triangle',
    'check_jdos_sum_rule',
    'find_vhs_peaks',
    'find_vhs_derivative',
    # kubo
    'SIGMA_0',
    'velocity_matrix_elements',
    'optical_conductivity_interband',
    'optical_conductivity_intraband',
    'optical_conductivity',
    'optical_conductivity_xx',
]
