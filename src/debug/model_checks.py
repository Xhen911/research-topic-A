"""
Model verification and debugging utilities.

Usage:
    python src/debug/model_checks.py
    # or from parent of src/:
    python -m debug.model_checks
"""

import sys
from pathlib import Path

# Ensure src/ is in path (same convention as notebooks)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from models.base import HamiltonianModel
from models.graphene import (
    SingleLayerGrapheneTB,
    BilayerGrapheneTB,
    SingleLayerGrapheneKP,
    BilayerGrapheneKP,
    BilayerGrapheneKPAA,
)


class ModelDebugSuite:
    """Quick debug suite for a single model."""

    def __init__(self, model: HamiltonianModel):
        self.model = model

    def check_hermitian_random(self, n=20, seed=42, tol=1e-10):
        """Random k-point Hermiticity check."""
        rng = np.random.default_rng(seed)
        for i in range(n):
            k = rng.normal(0, 3, 2)
            if not self.model.check_hermitian(k, tol=tol):
                return False, f"non-Hermitian at k={k}"
        return True, f"All {n} random k-points passed"

    def check_k_point(self, label, k_expected_energies, tol=0.01):
        """High-symmetry point band verification."""
        k = self.model.high_symmetry_points()[label]
        E, _ = self.model.solve(k)
        if len(E) != len(k_expected_energies):
            return False, f"n_bands {len(E)} != expected {len(k_expected_energies)}"
        diff = np.max(np.abs(np.sort(E) - np.sort(k_expected_energies)))
        return diff < tol, f"{label}: E={E} (expected {k_expected_energies}, diff={diff:.6f})"

    def check_solve_shape(self, k=None):
        """Check solve() output shapes."""
        if k is None:
            k = np.array([0.5, 0.3])
        E, V = self.model.solve(k)
        ok_e = E.shape == (self.model.n_bands,)
        ok_v = V.shape == (self.model.n_orbitals, self.model.n_bands)
        return ok_e and ok_v, f"E: {E.shape}, V: {V.shape}"


# ============================================================
#  KP model-specific verification
# ============================================================

def verify_kp_models(verbose=True):
    """
    Full KP model verification:
    - Dirac point energies
    - KP vs TB consistency
    - Bilayer KP (AB/AA/biased)
    - Hermiticity
    """
    results = []

    # --- Single-layer KP ---
    kp1 = SingleLayerGrapheneKP(t=2.78, valley=+1)
    k_K = kp1.K_point
    E_K, _ = kp1.solve(k_K)
    results.append(("SL KP at K == [0,0]", np.allclose(np.sort(E_K), [0.0, 0.0])))

    k_near = k_K + np.array([0.1, 0.0])
    E_near, _ = kp1.solve(k_near)
    expected = np.sqrt(3) / 2 * 2.78 * 0.1  # vF * dk ≈ 0.2408
    results.append(("SL KP near K+dk ~ +/-0.2408", np.allclose(np.sort(np.abs(E_near)), [expected, expected], atol=1e-3)))

    # --- KP vs TB consistency ---
    tb1 = SingleLayerGrapheneTB(t=2.78)
    E_tb_near, _ = tb1.solve(k_near)
    diff_kp_tb = np.max(np.abs(E_near - E_tb_near))
    results.append((f"KP-TB diff near K = {diff_kp_tb:.4f}", diff_kp_tb < 0.01))

    # --- Bilayer KP (AB) ---
    kp_bl = BilayerGrapheneKP(t=2.78, gamma1=0.39, valley=+1)
    E_bl_K, _ = kp_bl.solve(k_K)
    results.append(("BL KP at K ~ [-0.39, 0, 0, 0.39]", np.allclose(np.sort(np.abs(E_bl_K)), [0, 0, 0.39, 0.39], atol=1e-2)))

    # --- Bilayer KP (AB + bias) ---
    kp_biased = BilayerGrapheneKP(t=2.78, gamma1=0.39, u=0.5, valley=+1)
    E_biased_K, _ = kp_biased.solve(k_K)
    results.append(("Biased BL KP n_bands=4", len(E_biased_K) == 4))

    # --- Bilayer KP (AA) ---
    kp_aa = BilayerGrapheneKPAA(t=2.78, gamma_aa=0.2, valley=+1)
    E_aa_K, _ = kp_aa.solve(k_K)
    results.append(("AA KP at K ~ +/-0.2 double-degenerate", np.allclose(np.sort(np.abs(E_aa_K)), [0.2, 0.2, 0.2, 0.2], atol=1e-2)))

    # --- Random k-point Hermiticity ---
    rng = np.random.default_rng(42)
    all_hermitian = True
    for i in range(10):
        k = rng.normal(0, 3, 2)
        for name, model in [("SL KP", kp1), ("BL KP", kp_bl), ("AA KP", kp_aa)]:
            if not model.check_hermitian(k):
                all_hermitian = False
                results.append((f"{name} non-Hermitian at {k}", False))
                break
    if all_hermitian:
        results.append(("30 random k-points Hermitian", True))

    # --- 输出 ---
    if verbose:
        print("=" * 52)
        print("  KP Model Verification")
        print("=" * 52)
        passed = 0
        for desc, ok in results:
            status = "PASS" if ok else "FAIL!"
            print(f"  [{status}] {desc}")
            if ok:
                passed += 1
        print("-" * 52)
        print(f"  Passed: {passed}/{len(results)}")
        print("=" * 52)

    return all(ok for _, ok in results)


# ============================================================
#  Direct run entry point
# ============================================================

if __name__ == "__main__":
    all_ok = verify_kp_models(verbose=True)
    sys.exit(0 if all_ok else 1)
