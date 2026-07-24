"""Characterization (regression) baseline for the layer refactor.

这些测试在重构**之前**建立，冻结当前已验证的行为。
重构（models/response -> bands/propagators/interactions/observables/validation）
的每个 PR 合并前必须全部通过。

基准来源（见 .workbuddy/memory/checkpoint-2026-07-20.md）:
1. BM 模型本征值/速度算符快照 (theta=1.05, n_shells=3)
2. 三角形 DOS 求和规则: ∫DOS dE = g*nb*A_BZ/(2pi)^2
3. CNP 处填充因子 nu = 0
4. Lindhard 输出契约 (keys/shape/dtype)
5. dielectric/RPA 代数恒等式
6. 光电导数值冻结（当前行为快照）

注意: 通用光电导 sigma0 = e^2/4hbar 的**绝对标定**在当前参数下
并未收敛到 1（TB, nk=60 时 inter/sigma0 ~ 0.06-0.27），
这是已知的物理标定开放项，留给 validation 层（KK/Ward PR）处理。
此处仅做数值冻结，防止重构引入意外变化。
"""

import os

import numpy as np
import pytest

from src.bands.graphene import SingleLayerGrapheneTB
from src.bands.tbg_bm import BistritzMacDonaldTBG
from src.propagators.kubo import optical_conductivity_xx
from src.interactions.rpa import coulomb_2d, rpa_response
from src.observables.dielectric import dielectric_function, energy_loss_function
from src.propagators.dos import (
    check_dos_sum_rule,
    compute_dos_triangle,
    compute_eigenvalues,
)
from src.propagators.triangle_core import triangle_spectrum, _triangles_for_kmesh
from src.bands.occupations import compute_cnp, compute_filling
from src.propagators.lindhard import generate_k_mesh, lindhard_polarization

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


# ── 1. BM 模型本征值/速度算符快照 ─────────────────────────
@pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_DIR, 'bm_eig_snapshot.npz')),
    reason="bm_eig_snapshot.npz fixture not present in this checkout")
def test_bm_eigenvalue_snapshot():
    snap = np.load(os.path.join(DATA_DIR, 'bm_eig_snapshot.npz'))
    model = BistritzMacDonaldTBG(theta=float(snap['theta']),
                                 n_shells=int(snap['n_shells']))
    for i, k in enumerate(snap['kpts']):
        E, _ = model.solve(k)
        np.testing.assert_allclose(
            E, snap['eigenvalues'][i], rtol=1e-12, atol=1e-13,
            err_msg=f'BM eigenvalues drifted at k={k}')
        vx, _ = model.velocity_operator(k)
        np.testing.assert_allclose(
            np.diag(vx).real, snap['vx_diag'][i], rtol=1e-10, atol=1e-11,
            err_msg=f'BM velocity operator drifted at k={k}')


# ── 2. BM 哈密顿量厄米性 ──────────────────────────────────
def test_bm_hamiltonian_hermitian():
    model = BistritzMacDonaldTBG(theta=1.05, n_shells=3)
    for k in ([0.0, 0.0], [0.013, -0.007], [0.031, 0.022]):
        assert model.check_hermitian(np.asarray(k, float)), f'not hermitian at k={k}'


# ── 3. 三角形 DOS 求和规则（石墨烯 TB）────────────────────
def test_dos_sum_rule_graphene_triangle():
    g = SingleLayerGrapheneTB(t=2.78)
    E, dos = compute_dos_triangle(g, nk=24, nE=2000)
    g1, g2 = g.reciprocal_vectors
    area_bz = abs(g1[0] * g2[1] - g1[1] * g2[0])
    ok, integral, expected = check_dos_sum_rule(
        E, dos, g=g.degeneracy_factor(), nb=g.n_bands, area_BZ=area_bz)
    assert ok, f'DOS sum rule: integral={integral:.4f} vs expected={expected:.4f}'


# ── 3b. DOS delegation: compute_dos_triangle == hand-rolled primitive ──
def test_dos_triangle_delegates_to_primitive():
    """compute_dos_triangle must be a thin wrapper over triangle_spectrum —
    no numerical drift introduced by the delegation refactor."""
    g = SingleLayerGrapheneTB(t=2.78)
    nk = 24
    nE = 2000

    E, dos = compute_dos_triangle(g, nk=nk, nE=nE)

    # Replicate the internals of compute_dos_triangle directly.
    _, k_cart = generate_k_mesh(nk, g.reciprocal_vectors)
    E_k, _ = compute_eigenvalues(g, k_cart)
    area_BZ = abs(np.linalg.det(g.reciprocal_vectors))
    prefactor = g.degeneracy_factor() / ((2 * np.pi) ** 2)
    tri_idx = _triangles_for_kmesh(nk)
    spectrum = triangle_spectrum(
        vertex_fields=E_k, E_grid=E, weights=None, triangles=tri_idx,
        area_BZ=area_BZ, prefactor=prefactor, enforce_sum_rule=True)
    dos_ref = spectrum.sum(axis=1)

    np.testing.assert_allclose(dos, dos_ref, rtol=1e-10, atol=1e-12)


# ── 4. CNP 处填充因子为零（BM, 中间两条平带）──────────────
def test_cnp_filling_bm():
    model = BistritzMacDonaldTBG(theta=1.05, n_shells=3)
    _, k_cart = generate_k_mesh(8, model.reciprocal_vectors)
    E_k, _ = compute_eigenvalues(model, k_cart)
    half = E_k.shape[1] // 2
    bs = slice(half - 1, half + 1)
    mu = compute_cnp(E_k, band_slice=bs)
    nu = compute_filling(E_k, mu, band_slice=bs)
    assert abs(nu) < 1e-4, f'filling at CNP should be 0, got nu={nu}'


# ── 5. Lindhard 输出契约 ──────────────────────────────────
def test_lindhard_output_contract():
    g = SingleLayerGrapheneTB(t=2.78)
    q = np.array([0.01, 0.02, 0.03])
    w = np.linspace(0.01, 0.2, 4)
    r = lindhard_polarization(g, q, w, nk=12, eta=0.01, beta=200.0,
                              Ef=0.1, use_tqdm=False)
    assert set(r.keys()) == {'intra', 'inter'}
    for key in ('intra', 'inter'):
        assert r[key].shape == (len(w), len(q))
        assert r[key].dtype == np.complex128
        assert np.all(np.isfinite(r[key]))


# ── 6. dielectric / RPA 代数恒等式 ────────────────────────
def test_dielectric_rpa_identities():
    rng = np.random.default_rng(42)
    nw, nq = 6, 4
    pi0 = (rng.standard_normal((nw, nq)) + 1j * rng.standard_normal((nw, nq))) * 1e-3
    q = np.array([0.005, 0.01, 0.02, 0.04])
    v_q = coulomb_2d(q)

    eps, eps_inv = dielectric_function(pi0, v_q)
    np.testing.assert_allclose(eps, 1.0 - v_q[None, :] * pi0, rtol=1e-14)
    np.testing.assert_allclose(eps_inv * eps, 1.0, rtol=1e-12)

    pi_rpa = rpa_response(pi0, v_q)
    np.testing.assert_allclose(pi_rpa, pi0 / eps, rtol=1e-12)

    elf = energy_loss_function(pi0, v_q)
    np.testing.assert_allclose(elf, -eps_inv.imag, rtol=1e-14)


# ── 7. 光电导数值冻结（当前行为快照，非绝对标定）──────────
def test_optical_conductivity_freeze():
    g = SingleLayerGrapheneTB(t=2.78)
    w = np.array([0.5, 1.0, 1.5])
    r = optical_conductivity_xx(g, w, nk=60, Ef=0.0, eta=0.01)
    ref_inter = np.array([0.05837018, 0.26756604, 0.25503265])  # 2026-07-22 冻结
    np.testing.assert_allclose(r['inter'], ref_inter, rtol=1e-5)
    assert np.all(r['inter'] >= 0.0)


# ── 8. 半步偏移 q 网格（CachedModel 代码级，自包含）────────
def test_halfstep_grid_cached_model():
    """q_j = (j+1/2)*dq — 当前代码约定（2026-07 修订）。

    注意: dump 目录部分早期生产数据 (theta=0.80/0.99, nq=57) 是
    q_j = (j+1)*dq 旧约定，不能用同一断言扫描——故此处直接
    对 CachedModel 建最小实例验证代码行为。
    """
    from src.core.cache import CachedModel
    model = BistritzMacDonaldTBG(theta=1.05, n_shells=3)
    cache = CachedModel(model, nk=4, nb_cache=8, n_q=3, verbose=False)
    q = np.linalg.norm(cache.q_cart, axis=1)
    dq = np.diff(q)
    np.testing.assert_allclose(dq, dq[0], rtol=1e-12,
                               err_msg='q-mesh not uniform')
    assert abs(q[0] - dq[0] / 2) / dq[0] < 1e-10, \
        f'first q point {q[0]} != dq/2 (not half-step grid)'


# ── 9. 生产数据网格均匀性（可选，需 TBG_PROD_DATA_DIR）─────
@pytest.mark.skipif(not os.environ.get('TBG_PROD_DATA_DIR'),
                    reason='set TBG_PROD_DATA_DIR to production data folder')
def test_production_grid_uniform():
    """所有生产 grid-info 的 q 网格必须均匀（不假定 q0，兼容两代约定）。"""
    import glob
    prod = os.environ['TBG_PROD_DATA_DIR']
    files = glob.glob(os.path.join(prod, 'grid-info-*.npz'))
    assert files, f'no grid-info-*.npz under {prod}'
    for f in files:
        q = np.load(f)['q_norms']
        dq = np.diff(q)
        np.testing.assert_allclose(dq, dq[0], rtol=1e-10,
                                   err_msg=f'{f}: q-mesh not uniform')
