"""
tbg_relaxed.py
==============
Self-consistent lattice relaxation for twisted bilayer graphene.

Computes the displacement field u⁻(r) that minimizes elastic + interlayer
adhesion energy in tBLG, following the formalism of:

    "Maximized electron interactions at the magic angle in twisted bilayer graphene"
    Nature 572, 101–105 (2019)

    "Imaging moire deformation and dynamics in twisted bilayer graphene"
    Nat Commun 13, 70 (2022)

The relaxation yields effective interlayer tunneling rates (u, up) that
can be fed into the Bistritzer–MacDonald continuum model.

Usage
-----
    rel = TBGRelaxation(theta=1.05)        # by angle
    # or
    rel = TBGRelaxation(n_moire=31)        # by commensurate index
    u, up = rel.relax()                    # get effective tunneling rates

    # For full displacement field:
    rel.relax()
    u_field = rel.displacement_field       # real-space u⁻(r) in Å
    pmf = rel.pseudomagnetic_field          # strain-induced B_z(r) in Tesla

    # Feed into BM model:
    model = rel.to_bm_model()
"""

import numpy as np
from typing import Tuple, Optional

# ── Physical constants ──────────────────────────────────────────────────
_A_CC = 1.42               # carbon-carbon bond length in Angstrom
_A_C = _A_CC * np.sqrt(3)  # graphene lattice constant in Angstrom (~2.46)
_C_INTERLAYER = 3.35       # interlayer spacing in Angstrom (1.67 * 2 in notebook)
_DELTA_EPSILON = 0.0189    # eV/atom, adhesion energy scale
_SG = (_A_C) ** 2 * np.sin(np.pi / 3)               # graphene unit cell area (A^2)
_V0_DEFAULT = 4 * _DELTA_EPSILON / 9 / _SG           # ~0.001603 eV/A^2
_U0_DEFAULT = 0.104         # eV, bare interlayer tunneling (AA coupling at theta=0)
_TOLERANCE = 1e-4           # self-consistency convergence criterion

# Default Lame parameters for graphene (eV/A^2 at scale=1)
_LAMBDA_DEFAULT = 3.25
_MU_DEFAULT = 9.57


# ── Module-level helpers ────────────────────────────────────────────────

def _reciprocal_2d(a1: np.ndarray, a2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 2D reciprocal lattice vectors from a real-space pair."""
    a1_3d = np.append(a1, 0.0)
    a2_3d = np.append(a2, 0.0)
    n = np.array([0.0, 0.0, 1.0])
    a1xn = np.cross(a1_3d, n)
    a2xn = np.cross(a2_3d, n)
    B1 = (2 * np.pi * a2xn / (a1_3d @ a2xn))[:2]
    B2 = (2 * np.pi * a1xn / (a2_3d @ a1xn))[:2]
    return B1, B2


def _rotation_matrix(angle: float) -> np.ndarray:
    """2D anti-clockwise rotation matrix for *angle* in radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def _hex_loops(radius: int):
    """Generator for hexagonal-shell iteration indices.

    Yields ``(loop1, loop2)`` tuples for use with :func:`_iter2loops`.
    """
    loop1s = [
        (1, radius + 1),
        (1, radius + 1),
        (-radius, 1),
        (-radius, 0),
        (-radius, 0),
        (radius,),
    ]
    loop2s = [
        lambda m1, r=radius: (m1,),
        lambda m1, r=radius: (m1, r + 1),
        lambda m1, r=radius: (1, m1 + r + 1),
        lambda m1, r=radius: (m1 + 1, 1),
        lambda m1, r=radius: (-r, m1 + 1),
        lambda m1, r=radius: (-r + m1, 0),
    ]
    for l1, l2 in zip(loop1s, loop2s):
        yield l1, l2


# ── Main class ──────────────────────────────────────────────────────────

class TBGRelaxation:
    """Self-consistent lattice relaxation for twisted bilayer graphene.

    Computes the displacement field u⁻(r) that minimises elastic +
    interlayer adhesion energy, yielding effective BM interlayer
    tunneling parameters ``(u, up)``.

    Parameters
    ----------
    theta : float, optional
        Twist angle in degrees.  Overrides *n_moire* if provided.
    n_moire : int
        Commensurate moire index *m* (with *n* = *m* - 1).  Default 31 (~1.05 deg).
    lambda_lame : float
        Lame first parameter lambda (eV/A^2).  Default 3.25.
    mu : float
        Shear modulus mu (eV/A^2).  Default 9.57.
    u0 : float
        Bare AA interlayer tunneling at theta = 0 (eV).  Default 0.104.
    V0 : float, optional
        Adhesion potential scale (eV/A^2).  Auto-computed from delta_epsilon if *None*.

    Attributes
    ----------
    theta : float
        Twist angle in degrees.
    n_moire : int
        Commensurate index *m*.
    moire_period : float
        Moire superlattice period in Angstrom.
    u : float or None
        Relaxed AA interlayer tunneling (eV).  Populated after :meth:`relax`.
    up : float or None
        Relaxed AB interlayer tunneling (eV).  Populated after :meth:`relax`.
    displacement_field : np.ndarray or None
        Real-space displacement vector field ``u⁻(r)``, shape ``(2, N, N)``, in Angstrom.
    pseudomagnetic_field : np.ndarray or None
        Strain-induced pseudomagnetic field ``B_z(r)``, shape ``(N, N)``, in Tesla.
    """

    def __init__(
        self,
        theta: Optional[float] = None,
        n_moire: int = 31,
        lambda_lame: float = _LAMBDA_DEFAULT,
        mu: float = _MU_DEFAULT,
        u0: float = _U0_DEFAULT,
        V0: Optional[float] = None,
    ):
        self.lambda_lame = lambda_lame
        self.mu = mu
        self.u0 = u0
        self.V0 = V0 if V0 is not None else _V0_DEFAULT
        self.lat_a = _A_C
        self.lat_c = _C_INTERLAYER * 2  # unit-cell height

        # Determine twist angle from n_moire or directly
        if theta is not None:
            self.theta_deg = theta
            self.theta_rad = np.radians(theta)
            # Exact commensurate index from cos(theta) = 1 - 1/(6n^2+6n+2)
            s2 = np.sin(self.theta_rad / 2) ** 2
            n_float = (-1 + np.sqrt(max(0, 1 / (3 * s2 + 1e-30) - 1 / 3))) / 2
            self.n_moire = int(round(n_float))
            self.m = self.n_moire + 1
            self.n = self.n_moire
        else:
            self.n_moire = n_moire
            self.m = n_moire + 1
            self.n = n_moire
            # Exact twist angle from commensurability formula
            self.theta_rad = np.arccos(
                (self.m ** 2 + self.n ** 2 + 4 * self.m * self.n)
                / (2 * self.m ** 2 + 2 * self.n ** 2 + 2 * self.m * self.n)
            )
            self.theta_deg = np.degrees(self.theta_rad)

        # Rotation matrices
        self._R_half = _rotation_matrix(self.theta_rad / 2)   # half-twist
        self._R_full = _rotation_matrix(self.theta_rad)        # full twist

        # Layer-resolved real-space basis
        a1 = np.array([1.0, 0.0])
        a2 = np.array([0.5, np.sqrt(3) / 2])
        a0 = np.array([a1, a2]).T            # 2x2, columns are basis vectors

        self._a = self._R_half @ a0           # top layer basis  (R_{theta/2} . a0)
        self._at = self._R_full.T @ self._a   # bottom layer basis (R_{-theta} . a)
        self._ab = (self._a + self._at) / 2   # average basis  a_bar

        # Supercell lattice vectors
        m_mat = np.array([[self.m, self.n], [self.n + self.m, -self.m]])
        mp_mat = np.array([[self.n, self.m], [self.m + self.n, -self.n]])
        self.r_u = self._a @ m_mat.T    # top supercell  (2x2)
        self.r_d = self._a @ mp_mat.T   # bottom supercell (2x2)

        # Number of graphene unit cells per supercell
        self.length = round(np.linalg.norm(self.r_u[:, 1]) ** 2)

        # Reciprocal superlattice vectors
        g1, g2 = _reciprocal_2d(
            self._a[:, 0] * self.lat_a, self._a[:, 1] * self.lat_a
        )
        g3 = -g1 - g2
        self.g = np.array([g1, g2, g3])

        gt1, gt2 = _reciprocal_2d(
            self._at[:, 0] * self.lat_a, self._at[:, 1] * self.lat_a
        )
        gt3 = -gt1 - gt2
        self.gt = np.array([gt1, gt2, gt3])

        self.gb = (self.g + self.gt) / 2    # g_bar: averaged reciprocal vectors

        # Displacement wavevectors GM (moire supercell basis)
        GM = []
        for j in range(3):
            GM.append(
                1 / (self.m - self.n) * (np.eye(2) - self._R_full.T) @ self.g[j]
            )
        self.GM = np.array(GM)  # shape (3, 2)

        # Results (populated by relax())
        self.displacement_field: Optional[np.ndarray] = None   # u⁻(r)  real space (A)
        self.uq: Optional[np.ndarray] = None                   # u⁻(q)  Fourier space
        self.pseudomagnetic_field: Optional[np.ndarray] = None # B_z(r)  Tesla
        self.u: Optional[float] = None
        self.up: Optional[float] = None

        # Internal workspace (set up inside relax())
        self._x_grid = None
        self._y_grid = None
        self._mesh_x = None
        self._mesh_y = None
        self._qx = None
        self._qy = None
        self._Nmesh = None
        self._denom = None
        self._eta = None

    # ── Convenience properties ───────────────────────────────────────

    @property
    def theta(self) -> float:
        """Twist angle in degrees."""
        return self.theta_deg

    @property
    def moire_period(self) -> float:
        """Moire superlattice period in Angstrom."""
        return np.linalg.norm(self.r_u[:, 0]) * self.lat_a

    # ── Core: self-consistent relaxation ─────────────────────────────

    def relax(
        self,
        Nmesh: Optional[int] = None,
        alpha: float = 0.4,
        fft_norm_prefix: str = "forward",
        radius: int = 4,
        idx: int = 1,
        eta: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Run self-consistent relaxation and return relaxed **(u, up)**.

        Parameters
        ----------
        Nmesh : int, optional
            FFT mesh size per side.  Default: ``~1 / [2 sin(theta/2)]``.
        alpha : float
            Mixing parameter for self-consistent iteration.  Default 0.4.
        fft_norm_prefix : str
            NumPy FFT normalisation: ``"forward"``, ``"backward"``, or
            ``"ortho"``.  Default ``"forward"``.
        radius : int
            Hexagonal shell radius for pseudomagnetic field reconstruction.
            Default 4.
        idx : int
            Reciprocal direction index for q-space slices.  Default 1.
        eta : float, optional
            Dimensionless interlayer coupling parameter.
            Auto-computed from *V0* if *None*.

        Returns
        -------
        u : float
            Relaxed AA interlayer tunneling (eV).
        up : float
            Relaxed AB interlayer tunneling (eV).
        """
        # Set up real-space mesh
        lm = self.moire_period  # moire period in Angstrom
        self._Nmesh = (
            Nmesh
            if isinstance(Nmesh, int)
            else round(1 / (2 * np.sin(self.theta_rad / 2)))
        )
        N = self._Nmesh

        # FFT normalisation
        if fft_norm_prefix == "backward":
            self._denom = N ** 2
            fft_norm = "backward"
        elif fft_norm_prefix == "ortho":
            self._denom = N
            fft_norm = "ortho"
        else:  # 'forward'
            self._denom = 1
            fft_norm = "forward"

        # Real-space grid (Angstrom)
        r_mat = self.r_u.T  # 2x2
        mesh = np.meshgrid(
            np.linspace(0, 1, N), np.linspace(0, 1, N), indexing="ij"
        )
        self._x_grid = (
            mesh[0] * r_mat[0, 0] + mesh[1] * r_mat[1, 0]
        ) * self.lat_a
        self._y_grid = (
            mesh[0] * r_mat[0, 1] + mesh[1] * r_mat[1, 1]
        ) * self.lat_a

        # Dimensionless coupling eta
        if eta is not None:
            self._eta = eta
            self.V0 = (self.lambda_lame + self.mu) * (
                self._eta * self.lat_a / lm
            ) ** 2
        else:
            self._eta = (
                np.sqrt(self.V0 / (self.lambda_lame + self.mu))
                * lm
                / self.lat_a
            )
        print(
            f"theta: {self.theta_deg:.3f} deg, "
            f"LM: {lm:.2f} A, "
            f"eta: {self._eta:.3f}, "
            f"V0: {self.V0:.5f} eV/A^2, "
            f"N: {N}"
        )

        # Fourier-space grid
        array_q = np.fft.fftfreq(N, d=1 / N)
        self._mesh_x, self._mesh_y = np.meshgrid(
            array_q, array_q, indexing="ij"
        )
        self._qx = (
            self._mesh_x * self.GM[0, 0] + self._mesh_y * self.GM[idx, 0]
        )
        self._qy = (
            self._mesh_x * self.GM[0, 1] + self._mesh_y * self.GM[idx, 1]
        )

        # Self-consistent loop
        u_minus0 = np.zeros((2, N, N), dtype=complex)
        u_q = np.zeros((2, N, N), dtype=complex)
        u_q0 = np.zeros((2, N, N), dtype=complex)
        f_q = np.ones((3, N, N), dtype=complex)
        f_q0 = np.ones((3, N, N), dtype=complex)

        # Build inverse elastic kernel K^{-1}(q)
        qx_arr, qy_arr = self._qx, self._qy
        qp_x, qp_y = qy_arr, -qx_arr               # q_perp
        q4 = (qx_arr ** 2 + qy_arr ** 2) ** 2
        q4[0, 0] += 1e-10                           # avoid div-by-zero at q=0

        tsum = lambda a, b: np.einsum("ijk,ijk->jk", a, b)
        q_components = np.array([qx_arr, qy_arr])
        qp_components = np.array([qp_x, qp_y])
        invKq = (
            tsum(q_components, q_components) / (self.lambda_lame + 2 * self.mu)
            + tsum(qp_components, qp_components) / self.mu
        ) / q4

        diff = 1e4
        cnt = 0

        while diff >= _TOLERANCE:
            # u⁻(r) = IFFT[u⁻(q)]
            u_minus = np.fft.ifft2(u_q, norm=fft_norm)

            # f^j_q = FFT[sin(G_j . r_bar + g_bar_j . u⁻(r))]
            for j in range(3):
                phase = (
                    self._x_grid * self.GM[j, 0]
                    + self._y_grid * self.GM[j, 1]
                    + u_minus[0] * self.gb[j, 0]
                    + u_minus[1] * self.gb[j, 1]
                )
                f_q[j] = np.fft.fft2(np.sin(phase), norm=fft_norm)

            # u⁻(q) = 4 V0 sum_j K^{-1}(q) . g_bar_j . f^j_q
            u_q = np.sum(
                [
                    4
                    * self.V0
                    * (
                        np.array(
                            [invKq * self.gb[j, 0], invKq * self.gb[j, 1]]
                        )
                        * f_q[j]
                    )
                    for j in range(3)
                ],
                axis=0,
            )

            difference_m = np.linalg.norm(u_minus - u_minus0)
            difference_f = np.linalg.norm(f_q - f_q0)
            difference_u = np.linalg.norm(u_q - u_q0)

            print(
                f"\rdiff-m: {difference_m:.4g}  "
                f"diff-f: {difference_f:.4g}  "
                f"diff-u: {difference_u:.4g}",
                end="",
                flush=True,
            )

            if diff < max(difference_m, difference_f, difference_u) and cnt > 5:
                alpha *= 0.5  # reduce mixing if diverging

            diff = max(difference_m, difference_f, difference_u)

            # Under-relaxation (mixing)
            u_minus = alpha * u_minus + (1 - alpha) * u_minus0
            u_minus0 = u_minus.copy()
            f_q = alpha * f_q + (1 - alpha) * f_q0
            f_q0 = f_q.copy()
            u_q = alpha * u_q + (1 - alpha) * u_q0
            u_q0 = u_q.copy()
            cnt += 1

        print(
            f"\nConverged after {cnt} iterations, "
            f"final diff = {diff:.4e}, "
            f"theta = {self.theta_deg:.3f} deg"
        )

        # Store results
        self.displacement_field = np.real(u_minus)
        self.uq = u_q

        # Effective tunneling rates
        # Average |u_q| over first hexagonal shell
        uq1_vals = self._get_uq(self.uq, radius=1)
        mean_uq1 = np.mean([np.linalg.norm(v) for v in uq1_vals])
        # Compensate FFT normalisation (uq scaled by 1/denom in forward norm)
        mean_uq1 /= self._denom

        alpha_param = 2 * np.pi / np.sqrt(3) * mean_uq1 / self.lat_a
        self.u = self.u0 * (1 - 2 * alpha_param)
        self.up = self.u0 * (1 + alpha_param / 2)

        # Pseudomagnetic field
        qf = lambda x, y: [-2 * x * y, -x ** 2 + y ** 2]
        Vq = np.array(qf(self._qx, self._qy))
        self.pseudomagnetic_field = self._pseudomagnetic_field_calc(
            Vq, self.uq, radius, idx
        )

        return self.u, self.up

    # ── Internal helpers ─────────────────────────────────────────────

    def _get_uq(self, uq, radius=1):
        """Extract Fourier amplitudes on hexagonal shells."""
        uqls = []
        uqfunc = lambda m1, m2: np.abs(uq[:, m1, m2])
        for l1, l2 in _hex_loops(radius):
            uqls = self._iter2loops(uqls, l1, l2, uqfunc)
        return uqls

    def _pseudomagnetic_field_calc(self, Vq, uq, radius=3, idx=1):
        """Compute strain-induced pseudomagnetic field B_z in Tesla."""
        FourierReprLS = []
        frfunc = lambda m1, m2: np.einsum(
            "ijk,i,jk->jk",
            Vq,
            uq[:, m1, m2],
            np.exp(
                1j
                * np.einsum(
                    "i,ijk->jk",
                    m1 * self.GM[0] + m2 * self.GM[idx],
                    np.array([self._x_grid, self._y_grid]),
                )
            ),
        )
        for l1, l2 in _hex_loops(radius):
            FourierReprLS = self._iter2loops(FourierReprLS, l1, l2, frfunc)
        FourierRepr = np.sum(FourierReprLS, axis=0)

        # Conversion factor: B_z = -3/4 * beta/2 * 2*pi/(sqrt(3)*a) * hbar/e * 1e10
        #   beta ~ 3.14,  a = lat_a (A),  hbar = H/(2*pi),  e = E (C)
        E = 1.602176634e-19   # elementary charge (C)
        H = 6.6262e-34        # Planck constant (J.s)
        scale = (
            -3 / 4 * 3.14 * 2 / np.sqrt(3) / self.lat_a / 2 * 1e10 / E * H
        )
        return scale / self._denom * np.real(FourierRepr)

    @staticmethod
    def _iter2loops(mylist, loop1, loop2, myfunc):
        """Iterate over hexagonal loops and apply *myfunc*."""
        for m1 in range(*loop1):
            for m2 in range(*loop2(m1)):
                mylist.append(myfunc(m1, m2))
        return mylist

    # ── Convenience: build a relaxed BM model ────────────────────────

    def to_bm_model(self, **kwargs):
        """Build a :class:`BistritzMacDonaldTBG` with relaxed ``(u, up)``.

        Parameters
        ----------
        **kwargs
            Forwarded to :class:`BistritzMacDonaldTBG` (e.g. *valley*, *n_shells*).

        Returns
        -------
        BistritzMacDonaldTBG
        """
        from .tbg_bm import BistritzMacDonaldTBG

        if self.u is None or self.up is None:
            raise RuntimeError("Call relax() first.")

        return BistritzMacDonaldTBG(

            theta=self.theta_deg,

            u=self.u,

            up=self.up,

            **kwargs,

        )



    # ── Summary ─────────────────────────────────────────────────────



    def summary(self) -> str:

        """Return a human-readable summary of the relaxation result."""

        lines = [

            f"TBGRelaxation @ theta = {self.theta_deg:.4f} deg",

            f"  moire period    = {self.moire_period:.1f} A",

            f"  n_moire         = {self.n_moire}",

            f"  supercell atom N= {self.length}",

        ]

        if self.u is not None:

            lines.append(f"  u  (AA)         = {self.u:.6f} eV")

            lines.append(f"  up (AB)         = {self.up:.6f} eV")

        if self.displacement_field is not None and self._Nmesh > 0:

            max_disp = np.max(

                np.hypot(

                    self.displacement_field[0],

                    self.displacement_field[1],

                )

            )

            lines.append(f"  max |u-|        = {max_disp:.3f} A")

        if self.pseudomagnetic_field is not None and self._Nmesh > 0:

            max_b = np.max(np.abs(self.pseudomagnetic_field))

            lines.append(f"  max |Bz|        = {max_b:.1f} T")

        return "\n".join(lines)
