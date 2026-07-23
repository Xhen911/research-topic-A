"""
test_graphene_dirac_rpa.py — Regression tests for graphene Dirac RPA benchmark.
================================================================================

Validates the analytic-benchmark graphene RPA module (src/validation/graphene_dirac_rpa.py)
against known analytic limits.  Also includes a production-data snapshot test.

Run with SKIP_PROD=1 to reuse cached graphene-rpa-map.npz (fast CI mode).
"""
import os
import numpy as np
import pytest

from src.validation.graphene_dirac_rpa import (
    chi0_doped, chi0_undoped, chi0_static_closedform, chi0_undoped_bruteforce,
    rpa_dielectric, plasmon_freq, fsum_th, dos_at_EF, kF,
    VF_DEFAULT, G_DEG, E2_4PIEPS0,
)

EF = 0.5       # eV
VF = VF_DEFAULT
KAPPA = 1.0
KF = kF(EF, VF)
D_EF = dos_at_EF(EF, VF, G_DEG)
ETA = 0.005
ETA_STATIC = 1e-4


# ── Helpers ────────────────────────────────────────────────

def _trapz(y, x, **kw):
    try:
        return np.trapezoid(y, x, **kw)
    except AttributeError:
        return np.trapz(y, x, **kw)


# ── V0: Retardation ────────────────────────────────────────

def test_retardation():
    """Im chi0(q,w) <= 0 for w > 0."""
    q_test = np.array([0.01, 0.02])
    w_test = np.linspace(0.01, 0.5, 20)
    chi0 = chi0_doped(q_test, w_test, EF, eta=ETA, Nk=600, Ntheta=120)
    assert np.all(chi0.imag <= 1e-8), f"max Im chi0 = {chi0.imag.max():.2e}"


# ── V1: Static plateau ─────────────────────────────────────

def test_static_plateau():
    """Re chi0(q,0) = -D(E_F) for q < 2k_F."""
    q_st = np.linspace(0.03, 0.9, 15) * 2 * KF
    chi_st = chi0_doped(q_st, np.array([0.0]), EF, eta=ETA_STATIC,
                        Nk=1200, Ntheta=120)
    plateau = q_st < 0.9 * 2 * KF
    dev = np.max(np.abs(chi_st[0].real[plateau] / (-D_EF) - 1.0))
    assert dev < 0.02, f"static plateau deviation = {dev*100:.2f}%"


# ── V2: Closed form beyond 2k_F ────────────────────────────

def test_closed_form():
    """chi0_static_closedform matches numerical beyond 2k_F."""
    q_st = np.linspace(1.02, 3.0, 20) * 2 * KF
    chi_num = chi0_doped(q_st, np.array([0.0]), EF, eta=ETA_STATIC,
                         Nk=1200, Ntheta=120)
    chi_cf = chi0_static_closedform(q_st, EF)
    beyond = q_st > 2.0 * KF
    assert beyond.sum() > 0
    dev = np.max(np.abs(chi_num[0].real[beyond] / chi_cf[beyond] - 1.0))
    assert dev < 0.001, f"closed form deviation = {dev*100:.2f}%"


# ── V3: Undoped analytic vs brute-force ────────────────────

def test_undoped_bruteforce():
    """chi0_undoped analytic matches brute-force numerical integration."""
    worst = 0.0
    for qq in (0.3, 0.6, 1.0):
        qv = qq * KF
        for wfac in (0.0, 0.5, 1.5):
            wv = wfac * VF * qv
            ana = chi0_undoped(qv, wv, vf=VF, eta=0.005)
            num = chi0_undoped_bruteforce(qv, np.array([wv]), vf=VF,
                                          eta=0.005, K=80*qv)[0]
            rel = abs(num - ana) / max(abs(ana), 1e-30)
            worst = max(worst, rel)
    assert worst < 0.02, f"undoped brute-force worst = {worst*100:.2f}%"


# ── V4: Plasmon sqrt(q) prefactor ──────────────────────────

def test_plasmon_prefactor():
    """Long-wavelength plasmon follows the analytic sqrt(q) dispersion."""
    q_grid = np.linspace(0.002, 0.020, 10)
    w_grid = np.linspace(0.0, 0.8, 400)

    chi0 = chi0_doped(q_grid, w_grid, EF, eta=ETA, Nk=1200, Ntheta=120)
    _, _, elf = rpa_dielectric(chi0, q_grid, kappa=KAPPA)

    # Extract plasmon peak positions
    wpl = np.zeros(len(q_grid))
    for iq in range(len(q_grid)):
        i0 = np.argmax(elf[2:, iq]) + 2
        wpl[iq] = w_grid[i0]

    A_th = plasmon_freq(1.0, EF, kappa=KAPPA)[()]
    A_num = wpl[0] / np.sqrt(q_grid[0])
    assert abs(A_num / A_th - 1) < 0.05, \
        f"plasmon prefactor: {A_num:.3f} vs {A_th:.3f}"


# ── V5: f-sum rule ─────────────────────────────────────────

def test_fsum():
    """Integrated spectral weight matches f-sum rule at small q."""
    q_grid = np.linspace(0.002, 0.012, 6)
    w_grid = np.linspace(0.0, 0.8, 400)

    chi0 = chi0_doped(q_grid, w_grid, EF, eta=ETA, Nk=2400, Ntheta=240)
    _, _, elf = rpa_dielectric(chi0, q_grid, kappa=KAPPA)

    fsum_num = _trapz(w_grid[:, None] * elf, w_grid, axis=0)
    fsum_th_q = fsum_th(q_grid, EF, kappa=KAPPA)
    ratio = fsum_num / fsum_th_q
    assert 0.65 < ratio.min() and ratio.max() < 1.25, \
        f"f-sum ratio range: [{ratio.min():.3f}, {ratio.max():.3f}]"


# ── V6: Production data snapshot ───────────────────────────

def test_production_snapshot():
    """Full (q,w) map matches reference snapshot."""
    data_path = os.path.join(os.path.dirname(__file__),
                             "data", "graphene-rpa-map.npz")
    if not os.path.exists(data_path):
        pytest.skip("graphene-rpa-map.npz not found")

    ref = np.load(data_path)
    q_grid = ref["q"]
    w_grid = ref["w"]

    chi0 = chi0_doped(q_grid, w_grid, EF, eta=ref["eta"].item(),
                      Nk=2400, Ntheta=240)
    _, _, elf = rpa_dielectric(chi0, q_grid, kappa=ref["kappa"].item())

    rel_diff = np.max(np.abs(elf - ref["elf"]) / max(ref["elf"].max(), 1e-30))
    assert rel_diff < 0.05, f"ELF snapshot deviation = {rel_diff*100:.2f}%"
