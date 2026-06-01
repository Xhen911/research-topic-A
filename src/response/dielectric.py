"""
dielectric.py
=============
RPA dielectric response chain built on top of the Lindhard polarisation Π₀.

Functions
---------
- coulomb_2d(q_values, alpha=1.0)         — V(q) = 2π/q × α
- rpa_response(pi0, v_q)                  — Π_RPA = Π₀ / (1 − V_q · Π₀)
- dielectric_function(pi0, v_q)           — ε, ε⁻¹ = 1 − V_q · Π₀
- energy_loss_function(pi0, v_q)          — ELF = −Im[1/ε]

All functions use the *corrected* RPA denominator (no spurious 1/q factor).

Author: Refactored from Sabio(2008) graphene_polarization_swt_tb_v2.py
Date:   2026-05-30
"""

import numpy as np
from typing import Tuple

PI = np.pi


def coulomb_2d(q_values: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """二维库仑相互作用 V(q) = 2π / q × α。

    Parameters
    ----------
    q_values : np.ndarray, shape (nq,)
        动量转移模长。
    alpha : float
        耦合常数。悬浮石墨烯 α ≈ 2.2，简化模型常用 ≈ 3.77。

    Returns
    -------
    v_q : np.ndarray, shape (nq,)
    """
    q_safe = np.where(q_values < 1e-12, 1e-12, q_values)
    return alpha * 2 * PI / q_safe


def rpa_response(pi0: np.ndarray, v_q: np.ndarray) -> np.ndarray:
    """RPA 响应函数: Π_RPA = Π₀ / (1 − V_q · Π₀)。

    Parameters
    ----------
    pi0 : np.ndarray, shape (nw, nq), complex
        Lindhard 极化函数 Π₀(q, ω)。
    v_q : np.ndarray, shape (nq,)
        库仑相互作用 V(q)。

    Returns
    -------
    pi_rpa : np.ndarray, shape (nw, nq), complex
    """
    pi_rpa = np.zeros_like(pi0, dtype=complex)
    for iq in range(len(v_q)):
        pi_rpa[:, iq] = pi0[:, iq] / (1.0 - v_q[iq] * pi0[:, iq])
    return pi_rpa


def dielectric_function(
    pi0: np.ndarray, v_q: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """介电函数: ε(q, ω) = 1 − V_q · Π₀，及其倒数 ε⁻¹。

    Parameters
    ----------
    pi0 : np.ndarray, shape (nw, nq), complex
        Lindhard 极化函数 Π₀(q, ω)。
    v_q : np.ndarray, shape (nq,)
        库仑相互作用 V(q)。

    Returns
    -------
    eps : np.ndarray, shape (nw, nq), complex
    eps_inv : np.ndarray, shape (nw, nq), complex
    """
    eps = np.zeros_like(pi0, dtype=complex)
    eps_inv = np.zeros_like(pi0, dtype=complex)
    for iq in range(len(v_q)):
        eps[:, iq] = 1.0 - v_q[iq] * pi0[:, iq]
        eps_inv[:, iq] = 1.0 / eps[:, iq]
    return eps, eps_inv


def energy_loss_function(pi0: np.ndarray, v_q: np.ndarray) -> np.ndarray:
    """能量损失函数: −Im[1/ε(q, ω)]。

    等离激元频率处出现峰值（ε → 0）。

    Parameters
    ----------
    pi0 : np.ndarray, shape (nw, nq), complex
    v_q : np.ndarray, shape (nq,)

    Returns
    -------
    elf : np.ndarray, shape (nw, nq)
    """
    _, eps_inv = dielectric_function(pi0, v_q)
    return -eps_inv.imag
