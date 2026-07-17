
"""
cached_model.py — Eigenvalue cache wrapper for HamiltonianModel
================================================================

Diagonalize once, evaluate many times.  Wraps any HamiltonianModel
and caches the full k-grid (and optionally q-loop) diagonalisation
results.  Ideal for θ/ν batch scans where the Hamiltonian is
θ-dependent but ν-independent.

Usage
-----
    from src.models import BistritzMacDonaldTBG
    from src.response.cached_model import CachedModel

    model = BistritzMacDonaldTBG(theta=1.05, n_shells=4)
    cache = CachedModel(model, nk=24)
    # or with q-loop:
    cache = CachedModel(model, nk=24, n_q=20, q_max_factor=2.0)

    # Read cached data
    E_k  = cache.E_k       # (Nk, n_bands)
    V_k  = cache.V_k       # (Nk, n_orbitals, nb_cache)
    k_cart = cache.k_cart   # (Nk, 2)

    # Save / load from disk
    cache.save('cache_theta1.05.npz')
    cache2 = CachedModel.load('cache_theta1.05.npz', model)
"""

import numpy as np
import os
from .polarization import generate_k_mesh

class CachedModel:
    """Pre-diagonalised eigenvalue cache for any HamiltonianModel.

    Parameters
    ----------
    model : HamiltonianModel
        Any model implementing solve(k).
    nk : int
        k-points per reciprocal direction (total nk²).
    nb_cache : int or None
        Number of bands around CNP to cache (None = all).
    n_q : int or None
        If set, also cache q-loop eigenvalues.
    q_max_factor : float
        q_max = q_max_factor × kf  (kf from high_symmetry_points).
    n_workers : int or None
        Thread pool workers for q-loop (None = cpu_count).
    verbose : bool
    """

    def __init__(self, model, nk=24, nb_cache=None,
                 n_q=None, q_max_factor=2.0, q_eps=1e-6,
                 n_workers=None, verbose=True):
        self.model = model
        self.theta = getattr(model, 'theta', None)
        self.nk = nk
        self.Nk = nk * nk
        self.nb_cache = nb_cache or model.n_bands

        # ── k-grid diagonalisation ──
        _, self.k_cart = generate_k_mesh(nk, model.reciprocal_vectors)
        self.dk = np.sqrt(abs(np.linalg.det(model.reciprocal_vectors)) / self.Nk)

        if verbose:
            print(f'CachedModel: nk={nk}, Nk={self.Nk}, dk={self.dk:.5f} 1/A')

        self.E_k = np.zeros((self.Nk, model.n_bands))
        self.V_k = np.zeros((self.Nk, model.n_orbitals, self.nb_cache), dtype=complex)
        self.bs_cache = slice(model.n_bands//2 - self.nb_cache//2,
                                  model.n_bands//2 + self.nb_cache//2)

        for i in range(self.Nk):
            Ei, Vi = model.solve(self.k_cart[i])
            self.E_k[i] = Ei
            self.V_k[i] = Vi[:, self.bs_cache]

        # ── q-loop (optional) ──
        self.E_q = None
        self.V_q = None
        self.q_cart = None
        self.q_norms = None
        self.Nq = 0

        if n_q is not None and n_q > 0:
            hs = model.high_symmetry_points()
            kf = np.linalg.norm(hs['K'])
            q_max = q_max_factor * kf
            q_spacing = self.dk * (1 + 1/np.sqrt(3)) / 3
            self.Nq = max(2, int(round(q_max / q_spacing)))
            self.Nq = min(self.Nq, n_q)
            dq_q = q_max / self.Nq  # q-step in the mesh

            # -- q_eps sanity bounds --
            if q_eps != 0.0:
                assert q_eps < dq_q / 2, (
                    f"q_eps={q_eps:.2e} >= dq_q/2={dq_q/2:.2e} "
                    f"-- offset too large, would sample finite q")
                _floor = 1e-9
                assert q_eps > _floor, (
                    f"q_eps={q_eps:.2e} <= {_floor:.1e} "
                    f"-- fp cancellation risk in energy denominators")

            q_mag = np.linspace(0, 1, self.Nq) * q_max     # 0 .. q_max
            q_mag = np.maximum(q_mag, q_eps)          # shift q=0 to eps
            q_dir = hs['K'] / kf
            self.q_cart = np.array([qi * q_dir for qi in q_mag])
            self.q_norms = np.linalg.norm(self.q_cart, axis=1)

            self.E_q = np.zeros((self.Nq, self.Nk, model.n_bands))
            self.V_q = np.zeros((self.Nq, self.Nk, model.n_orbitals, self.nb_cache),
                                dtype=complex)

            if verbose:
                print(f'  q-loop: Nq={self.Nq}, q_max={q_max:.4f} 1/A')

            for iq in range(self.Nq):
                kq = self.k_cart + self.q_cart[iq]
                for ik in range(self.Nk):
                    Ei, Vi = model.solve(kq[ik])
                    self.E_q[iq, ik] = Ei
                    self.V_q[iq, ik] = Vi[:, self.bs_cache]

    # ── IO ────────────────────────────────────────────────
    # -- Single-q evaluation (lightweight, no full q-loop) --
    def eig_at_q(self, q_vec):
        """Diagonalise model at k+q for a single q-vector.

        Returns E_q (Nk, n_bands), V_q (Nk, n_orbitals, nb_cache).
        Does NOT require a full q-loop in the cache.
        """
        if self.model is None:
            raise RuntimeError("eig_at_q requires a model reference")
        Nk = self.Nk
        E_q = np.zeros((Nk, self.model.n_bands))
        V_q = np.zeros((Nk, self.model.n_orbitals, self.nb_cache), dtype=complex)
        for ik in range(Nk):
            Ei, Vi = self.model.solve(self.k_cart[ik] + q_vec)
            E_q[ik] = Ei
            V_q[ik] = Vi[:, self.bs_cache]
        return E_q, V_q

    def save(self, path):
        """Save cache to .npz file."""
        d = dict(nk=self.nk, Nk=self.Nk, dk=self.dk,
                 nb_cache=self.nb_cache,
                 E_k=self.E_k, V_k=self.V_k,
                 k_cart=self.k_cart,
                 theta=self.theta)
        if self.E_q is not None:
            d.update(E_q=self.E_q, V_q=self.V_q,
                     q_cart=self.q_cart, q_norms=self.q_norms, Nq=self.Nq)
        np.savez(path, **d)
        print(f'Saved: {path} ({os.path.getsize(path)/1e6:.1f} MB)')

    @staticmethod
    def load(path, model=None):
        """Load cache from .npz file.  model is needed for metadata only."""
        data = np.load(path, allow_pickle=False)
        cache = object.__new__(CachedModel)
        cache.model = model
        cache.theta = float(data.get('theta', 0))
        cache.nk = int(data['nk'])
        cache.Nk = int(data['Nk'])
        cache.dk = float(data['dk'])
        cache.nb_cache = int(data['nb_cache'])
        cache.E_k = data['E_k']
        cache.V_k = data['V_k']
        cache.k_cart = data['k_cart']
        cache.E_q = data.get('E_q')
        cache.V_q = data.get('V_q')
        cache.q_cart = data.get('q_cart')
        cache.q_norms = data.get('q_norms')
        cache.Nq = int(data.get('Nq', 0))
        half = cache.E_k.shape[1] // 2
        cache.bs_cache = slice(half - cache.nb_cache // 2,
                               half + cache.nb_cache // 2)
        return cache

    # ── Convenience ───────────────────────────────────────
    def get_flat_bands(self):
        """Return E_k sliced to flat pair [half-1, half]."""
        half = self.E_k.shape[1] // 2
        return self.E_k[:, half-1: half+1]
