"""Rigorous Coupled Wave Analysis.

The algorithm, described in Glytsis, "Three-dimensional (vector)
rigorous coupled-wave analysis of anisotropic grating diffraction",
JOSA A, 7(8), 1990, permits to find the diffraction efficiencies
(i.e. the power normalized to unity) reflected and transmitted by a
multilayer.
The multilayer can be made of isotropic or anisotropic layers and
binary gratings.
Two versions of the algorithm are present: an isotropic one and an
anisotropic one, both stabilized via Moharam's Enhanced Transmittance
Matrix recursion for arbitrary diffraction orders and layer thicknesses.

"""

__author__ = "Lorenzo Bolla"

import numpy as np
from numpy import pi
from numpy.linalg import inv, solve as linsolve
from scipy.linalg import eig, toeplitz

from EMpy.constants import eps0
from EMpy.utils import (
    cond,
    warning,
    BinaryGrating,
    SymmetricDoubleGrating,
    AsymmetricDoubleGrating,
)


def dispersion_relation_ordinary(kx, ky, k, nO):
    """Dispersion relation for the ordinary wave.

    NOTE
    See eq. 15 in Glytsis, "Three-dimensional (vector) rigorous
    coupled-wave analysis of anisotropic grating diffraction",
    JOSA A, 7(8), 1990 Always give positive real or negative
    imaginary.
    """

    if kx.shape != ky.shape:
        raise ValueError("kx and ky must have the same length")

    delta = np.asarray((k * nO) ** 2 - (kx**2 + ky**2), dtype=complex)
    kz = np.sqrt(delta)

    # Choose the algebraic root +/-sqrt(delta) (preserving kz**2 == delta) with branch cut
    # along the positive imaginary axis of delta (Re(kz) - Im(kz) >= 0): continuous across
    # both propagating (Re(kz) >= 0, including lossy-superstrate Quadrant-I delta) and
    # evanescent / TIR (Im(kz) < 0) regimes under exp(+i*w*t - i*kz*z).
    flip = (kz.real - kz.imag < 0) | ((kz.real - kz.imag == 0) & (kz.real < 0))
    kz = np.where(flip, -kz, kz)
    if np.all(np.abs(delta.imag) == 0):
        kz = np.where(delta.real >= 0, kz.real + 0.0j, 1j * kz.imag)

    return kz


def dispersion_relation_extraordinary(kx, ky, k, nO, nE, c):
    """Dispersion relation for the extraordinary wave.

    NOTE
    See eq. 16 in Glytsis, "Three-dimensional (vector) rigorous
    coupled-wave analysis of anisotropic grating diffraction",
    JOSA A, 7(8), 1990 Always give positive real or negative
    imaginary.
    """

    if kx.shape != ky.shape or c.size != 3:
        raise ValueError(
            "kx and ky must have the same length and c must have 3 components"
        )

    kx_c = np.asarray(kx, dtype=complex)
    ky_c = np.asarray(ky, dtype=complex)
    alpha = nE**2 - nO**2
    beta = kx_c / k * c[0] + ky_c / k * c[1]

    c2 = nO**2 + c[2] ** 2 * alpha
    c1 = 2.0 * c[2] * beta * alpha
    c0 = nO**2 * (kx_c**2 + ky_c**2) / k**2 + alpha * beta**2 - nO**2 * nE**2

    half_b = 0.5 * c1 / c2
    disc = np.asarray(half_b**2 - c0 / c2, dtype=complex)
    s = np.sqrt(disc)
    flip = (s.real - s.imag < 0) | ((s.real - s.imag == 0) & (s.real < 0))
    s = np.where(flip, -s, s)
    kz = k * (-half_b + s)
    if (
        np.all(np.abs(disc.imag) == 0)
        and np.all(np.abs(np.imag(half_b)) == 0)
        and np.all(disc.real >= 0)
    ):
        kz = kz.real + 0.0j

    return kz


class RCWA:
    """Class to handle the RCWA solvers.

    NOTE
    See Glytsis, "Three-dimensional (vector) rigorous coupled-wave
    analysis of anisotropic grating diffraction", JOSA A, 7(8), 1990

    The following variables, used throughout the code, have the following
    meaning:

    alpha:  float
            angle between wave vector k1 and the z axis (surface normal),
            in radians

    delta:  float
            angle between the x axis and the projection of k1 onto the xy
            plane, in radians

    psi:    angle between the D vector of the plane wave and the plane of
            incidence, in radians, TM: 0, TE: numpy.pi / 2

    phi:    angle between the grating vector K and z axis (in the xz plane),
            in radians, the grating is modulated in the direction of the
            grating vector (phi = pi/2 corresponds to K along x).

    """

    def __init__(self, multilayer, alpha, delta, psi, phi, n):
        """Set the multilayer, the angles of incidence and the diffraction order.

        INPUT
        multilayer = Multilayer obj describing the sequence of layers.
        alpha, delta, psi, phi = angles of the incident wave (in
        radiant).
        n = orders of diffractions to retain in the computation.
        """
        self.setMultilayer(multilayer)
        self.LAMBDA = self.get_pitch()
        self.alpha = alpha
        self.delta = delta
        self.psi = psi
        self.phi = phi
        self.n = n

    def setMultilayer(self, m):
        """Set the multilayer, simplifying it first."""
        self.multilayer = m.simplify()

    def get_pitch(self):
        """Inspect the multilayer to check that all the binary
        gratings present have the same pitch, and return it."""
        idx = np.where(
            [
                (
                    isinstance(m, BinaryGrating)
                    | isinstance(m, SymmetricDoubleGrating)
                    | isinstance(m, AsymmetricDoubleGrating)
                    | hasattr(m, "pitch")
                )
                for m in self.multilayer
            ]
        )[0]
        if idx.size == 0:
            # warning('no BinaryGratings: better use a simple transfer matrix.')
            return 1.0  # return LAMBDA: any value will do
        else:
            # check that all the pitches are the same!
            l = np.asarray([self.multilayer[i].pitch for i in idx])
            if not np.all(l == l[0]):
                raise ValueError("All the BinaryGratings must have the same pitch.")
            else:
                return l[0]


class IsotropicRCWA(RCWA):
    """Isotropic RCWA solver."""

    def solve(self, wls):
        """Isotropic solver.

        INPUT
        wls = wavelengths to scan (any asarray-able object).

        OUTPUT
        self.DE1, self.DE3 = power reflected and transmitted.

        NOTE
        see:
        Moharam, "Formulation for stable and efficient implementation
        of the rigorous coupled-wave analysis of binary gratings",
        JOSA A, 12(5), 1995
        Lalanne, "Highly improved convergence of the coupled-wave
        method for TM polarization", JOSA A, 13(4), 1996
        Moharam, "Stable implementation of the rigorous coupled-wave
        analysis for surface-relief gratings: enhanced trasmittance
        matrix approach", JOSA A, 12(5), 1995
        """

        self.wls = np.atleast_1d(wls)

        LAMBDA = self.LAMBDA
        n = self.n
        multilayer = self.multilayer
        alpha = self.alpha
        delta = self.delta
        psi = self.psi
        phi = self.phi

        if not np.isclose(np.cos(phi), 0.0, atol=1e-10):
            raise ValueError("First Order Helmholtz Operator not implemented, yet!")

        nlayers = len(multilayer)
        i = np.arange(-n, n + 1)
        nood = 2 * n + 1
        hmax = nood - 1

        # grating vector (on the xz plane)
        # grating on the xy plane
        K = 2 * pi / LAMBDA * np.array([np.sin(phi), 0.0, np.cos(phi)], dtype=complex)

        DE1 = np.zeros((nood, self.wls.size))
        DE3 = np.zeros_like(DE1)

        dirk1 = np.array(
            [
                np.sin(alpha) * np.cos(delta),
                np.sin(alpha) * np.sin(delta),
                np.cos(alpha),
            ]
        )

        # usefull matrices
        I = np.eye(i.size, dtype=complex)
        I2 = np.eye(i.size * 2, dtype=complex)
        ZERO = np.zeros_like(I)

        X = np.zeros((2 * nood, 2 * nood, nlayers), dtype=complex)
        MTp1 = np.zeros((2 * nood, 2 * nood, nlayers), dtype=complex)
        MTp2 = np.zeros_like(MTp1)

        EPS2 = np.zeros(2 * hmax + 1, dtype=complex)
        EPS21 = np.zeros_like(EPS2)

        dlt = (i == 0).astype(int)

        for iwl, wl in enumerate(self.wls):
            # free space wavevector
            k = 2 * pi / wl

            n1 = multilayer[0].mat.n(wl).item()
            n3 = multilayer[-1].mat.n(wl).item()

            # incident plane wave wavevector
            k1 = k * n1 * dirk1

            if np.absolute(K[2] / k) > 1e-10:
                raise ValueError("First Order Helmholtz Operator not implemented, yet!")

            # Precompute wavelength-only Toeplitz permittivity blocks per internal layer
            # and apply any Helmholtz singularity shift uniformly to the incident Bloch
            # wavevector BEFORE constructing superstrate/substrate/layer matrices so all
            # layers share one consistent Floquet lattice kx_m = k1x_eff - m * Kx.
            layer_mats = []
            for nlayer in range(nlayers - 2, 0, -1):
                layer = multilayer[nlayer]
                EPS2, EPS21 = layer.getEPSFourierCoeffs(wl, n, anisotropic=False)
                E = toeplitz(EPS2[hmax::-1], EPS2[hmax:])
                E1 = toeplitz(EPS21[hmax::-1], EPS21[hmax:])
                layer_mats.append((nlayer, layer.thickness, E, inv(E1)))

            tmp_x = k1[0] - i * K[0]
            tmp_y = k1[1] * np.ones_like(i)
            if any(
                cond(np.diag((tmp_x / k) ** 2) - E_l) > 1e10
                for _, _, E_l, _ in layer_mats
            ):
                warning("BAD CONDITIONING: randomization of kx")
                rng = np.random.default_rng(0)
                dk_x = 0.0
                for _ in range(100):
                    if all(
                        cond(np.diag((tmp_x / k) ** 2) - E_l) <= 1e10
                        for _, _, E_l, _ in layer_mats
                    ):
                        break
                    dk_x += 1e-9 * k * (0.5 + float(rng.random()))
                    tmp_x = (k1[0] + dk_x) - i * K[0]

            tmp_z = dispersion_relation_ordinary(tmp_x, tmp_y, k, n1)
            k1i = np.r_[[tmp_x], [tmp_y], [tmp_z]]

            # k2i = np.r_[[k1[0] - i*K[0]], [k1[1] - i * K[1]], [-i * K[2]]]

            tmp_z = dispersion_relation_ordinary(tmp_x, tmp_y, k, n3)
            k3i = np.r_[[k1i[0, :]], [k1i[1, :]], [tmp_z]]

            # aliases for constant wavevectors
            kx = k1i[0, :]
            ky = k1[1]

            # Complex plane-of-diffraction direction cosines (fc, fs) = (kx, ky) / kt
            # on the passive branch Re(kt) - Im(kt) >= 0 so k_t . (-fs, fc) == 0 holds
            # identically even for conical diffraction in a lossy superstrate; at normal
            # diffraction (kx == ky == 0), preserve delta for continuity as alpha -> 0+.
            kt = np.sqrt(kx**2 + ky**2 + 0j)
            flip_kt = (kt.real - kt.imag < 0) | (
                (kt.real - kt.imag == 0) & (kt.real < 0)
            )
            kt = np.where(flip_kt, -kt, kt)
            zero_kt = np.abs(kt) <= 1e-13 * k
            kt_safe = np.where(zero_kt, 1.0 + 0j, kt)
            fc = np.where(zero_kt, np.cos(delta) + 0j, kx / kt_safe)
            fs = np.where(zero_kt, np.sin(delta) + 0j, ky / kt_safe)

            Kx = np.diag(kx / k)
            Ky = ky / k * I
            Z1 = np.diag(k1i[2, :] / (k * n1**2))
            Y1 = np.diag(k1i[2, :] / k)
            Z3 = np.diag(k3i[2, :] / (k * n3**2))
            Y3 = np.diag(k3i[2, :] / k)

            MR = np.block([[I, ZERO], [-1j * Y1, ZERO], [ZERO, I], [ZERO, -1j * Z1]])

            MT = np.block([[I, ZERO], [1j * Y3, ZERO], [ZERO, I], [ZERO, 1j * Z3]])

            # internal layers (grating or layer)
            X.fill(0.0)
            MTp1.fill(0.0)
            MTp2.fill(0.0)
            for nlayer, d, E, E11 in layer_mats:
                B = kx[:, np.newaxis] / k * linsolve(E, Kx) - I
                A = np.diag((kx / k) ** 2) - E

                if ky == 0 or np.allclose(np.diag(Ky / ky * k), 1):
                    # lalanne
                    # H_U_reduced = np.dot(Ky, Ky) + A
                    H_U_reduced = (ky / k) ** 2 * I + A
                    # H_S_reduced = np.dot(Ky, Ky) + np.dot(Kx, linsolve(E, np.dot(Kx, E11))) - E11
                    H_S_reduced = (
                        (ky / k) ** 2 * I
                        + kx[:, np.newaxis]
                        / k
                        * linsolve(E, kx[:, np.newaxis] / k * E11)
                        - E11
                    )

                    # Eigenvalues of H_U_reduced / H_S_reduced are q^2 = -k_z^2 / k^2,
                    # with forward modes exp(-i k_z z) = exp(-k q z) => q = +i k_z / k.
                    # Under exp(+iwt - ik.r), the passive branch Re(k_z) - Im(k_z) >= 0
                    # (continuous across propagating, TIR, plasmonic, and lossy-superstrate
                    # regimes, where oblique Im(k_x) < 0 puts k_z in Quadrant I and q in
                    # Quadrant II) maps to Re(q) + Im(q) >= 0; bare np.sqrt returns Re(q) >= 0
                    # and would pick the backward root -q in Quadrant IV.
                    q1, W1 = eig(H_U_reduced)
                    q1 = np.sqrt(q1 + 0j)
                    flip_q1 = (q1.real + q1.imag < 0) | (
                        (q1.real + q1.imag == 0) & (q1.imag < 0)
                    )
                    q1 = np.where(flip_q1, -q1, q1)
                    q2, W2 = eig(H_S_reduced)
                    q2 = np.sqrt(q2 + 0j)
                    flip_q2 = (q2.real + q2.imag < 0) | (
                        (q2.real + q2.imag == 0) & (q2.imag < 0)
                    )
                    q2 = np.where(flip_q2, -q2, q2)

                    # boundary conditions

                    # V11 = np.dot(linsolve(A, W1), np.diag(q1))
                    V11 = linsolve(A, W1) * q1[np.newaxis, :]
                    V12 = (ky / k) * np.dot(linsolve(A, Kx), W2)
                    V21 = (ky / k) * np.dot(linsolve(B, Kx), linsolve(E, W1))
                    # V22 = np.dot(linsolve(B, W2), np.diag(q2))
                    V22 = linsolve(B, W2) * q2[np.newaxis, :]

                    # Vss = np.dot(Fc, V11)
                    Vss = fc[:, np.newaxis] * V11
                    # Wss = np.dot(Fc, W1)  + np.dot(Fs, V21)
                    Wss = fc[:, np.newaxis] * W1 + fs[:, np.newaxis] * V21
                    # Vsp = np.dot(Fc, V12) - np.dot(Fs, W2)
                    Vsp = fc[:, np.newaxis] * V12 - fs[:, np.newaxis] * W2
                    # Wsp = np.dot(Fs, V22)
                    Wsp = fs[:, np.newaxis] * V22
                    # Wpp = np.dot(Fc, V22)
                    Wpp = fc[:, np.newaxis] * V22
                    # Vpp = np.dot(Fc, W2)  + np.dot(Fs, V12)
                    Vpp = fc[:, np.newaxis] * W2 + fs[:, np.newaxis] * V12
                    # Wps = np.dot(Fc, V21) - np.dot(Fs, W1)
                    Wps = fc[:, np.newaxis] * V21 - fs[:, np.newaxis] * W1
                    # Vps = np.dot(Fs, V11)
                    Vps = fs[:, np.newaxis] * V11

                    Mc2bar = np.block(
                        [
                            [Vss, Vsp, Vss, Vsp],
                            [Wss, Wsp, -Wss, -Wsp],
                            [Wps, Wpp, -Wps, -Wpp],
                            [Vps, Vpp, Vps, Vpp],
                        ]
                    )

                    x = np.r_[np.exp(-k * q1 * d), np.exp(-k * q2 * d)]

                    # Mc1 = np.dot(Mc2bar, np.diag(np.r_[np.ones_like(x), x]))
                    xx = np.r_[np.ones_like(x), x]
                    Mc1 = Mc2bar * xx[np.newaxis, :]

                    X[:, :, nlayer] = np.diag(x)

                    MTp = linsolve(Mc2bar, MT)
                    MTp1[:, :, nlayer] = MTp[0 : 2 * nood, :]
                    MTp2 = MTp[2 * nood :, :]

                    MT = np.dot(
                        Mc1,
                        np.r_[
                            I2,
                            np.dot(MTp2, linsolve(MTp1[:, :, nlayer], X[:, :, nlayer])),
                        ],
                    )

                else:
                    raise ValueError(
                        "Second Order Helmholtz Operator not implemented, yet!"
                    )

            # M = np.asarray(np.bmat([-MR, MT]))
            M = np.c_[-MR, MT]
            E_inc = np.array(
                [
                    np.cos(psi) * np.cos(alpha) * np.cos(delta)
                    - np.sin(psi) * np.sin(delta),
                    np.cos(psi) * np.cos(alpha) * np.sin(delta)
                    + np.sin(psi) * np.cos(delta),
                    -np.cos(psi) * np.sin(alpha),
                ],
                dtype=complex,
            )
            H_inc = np.cross(k1 / k, E_inc)
            fc0 = fc[n]
            fs0 = fs[n]
            b = np.r_[
                (-fs0 * E_inc[0] + fc0 * E_inc[1]) * dlt,
                -1j * (fc0 * H_inc[0] + fs0 * H_inc[1]) * dlt,
                -1j * (-fs0 * H_inc[0] + fc0 * H_inc[1]) * dlt,
                (fc0 * E_inc[0] + fs0 * E_inc[1]) * dlt,
            ]

            x = linsolve(M, b)
            R, T = np.split(x, 2)
            Rs, Rp = np.split(R, 2)
            for ii in range(1, nlayers - 1):
                T = np.dot(linsolve(MTp1[:, :, ii], X[:, :, ii]), T)
            Ts, Tp = np.split(T, 2)

            z1_d = k1i[2, :] / (k * n1**2)
            y1_d = k1i[2, :] / k
            z3_d = k3i[2, :] / (k * n3**2)
            y3_d = k3i[2, :] / k

            ERx = -fs * Rs - 1j * fc * z1_d * Rp
            ERy = fc * Rs - 1j * fs * z1_d * Rp
            HRx = fc * y1_d * Rs - 1j * fs * Rp
            HRy = fs * y1_d * Rs + 1j * fc * Rp

            ETx = -fs * Ts + 1j * fc * z3_d * Tp
            ETy = fc * Ts + 1j * fs * z3_d * Tp
            HTx = -fc * y3_d * Ts - 1j * fs * Tp
            HTy = -fs * y3_d * Ts + 1j * fc * Tp

            denom = float(
                (E_inc[0] * np.conj(H_inc[1]) - E_inc[1] * np.conj(H_inc[0])).real
            )
            DE1[:, iwl] = -(ERx * np.conj(HRy) - ERy * np.conj(HRx)).real / denom
            DE3[:, iwl] = (ETx * np.conj(HTy) - ETy * np.conj(HTx)).real / denom

        # save the results
        self.DE1 = DE1
        self.DE3 = DE3

        return self

    # def plot(self):
    #         """Plot the diffraction efficiencies."""
    #         g = Gnuplot.Gnuplot()
    #         g('set xlabel "$\lambda$"')
    #         g('set ylabel "diffraction efficiency"')
    #         g('set yrange [0:1]')
    #         g('set data style linespoints')
    #         g.plot(Gnuplot.Data(self.wls, self.DE1[self.n,:], with_ = 'linespoints', title = 'DE1'), \
    #                Gnuplot.Data(self.wls, self.DE3[self.n,:], with_ = 'linespoints', title = 'DE3'))
    #         raw_input('press enter to close the graph...')

    def __str__(self):
        return (
            "ISOTROPIC RCWA SOLVER\n\n%s\n\nLAMBDA = %g\nalpha = %g\ndelta = %g\npsi = %g\nphi = %g\nn = %d"
            % (
                self.multilayer.__str__(),
                self.LAMBDA,
                self.alpha,
                self.delta,
                self.psi,
                self.phi,
                self.n,
            )
        )


class AnisotropicRCWA(RCWA):
    """Anisotropic RCWA solver."""

    def solve(self, wls):
        """Anisotropic solver.

        INPUT
        wls = wavelengths to scan (any asarray-able object).

        OUTPUT
        self.DEO1, self.DEE1, self.DEO3, self.DEE3 = power reflected
        and transmitted.
        """

        self.wls = np.atleast_1d(wls)

        LAMBDA = self.LAMBDA
        n = self.n
        multilayer = self.multilayer
        alpha = self.alpha
        delta = self.delta
        psi = self.psi
        phi = self.phi

        if not np.isclose(np.cos(phi), 0.0, atol=1e-10):
            raise ValueError("First Order Helmholtz Operator not implemented, yet!")

        nlayers = len(multilayer)
        i = np.arange(-n, n + 1)
        nood = 2 * n + 1
        hmax = nood - 1

        DEO1 = np.zeros((nood, self.wls.size))
        DEO3 = np.zeros_like(DEO1)
        DEE1 = np.zeros_like(DEO1)
        DEE3 = np.zeros_like(DEO1)
        DE1 = np.zeros_like(DEO1)
        DE3 = np.zeros_like(DEO1)

        c1 = np.array([1.0, 0.0, 0.0])
        c3 = np.array([1.0, 0.0, 0.0])
        # grating on the xy plane
        K = 2 * pi / LAMBDA * np.array([np.sin(phi), 0.0, np.cos(phi)], dtype=complex)
        dirk1 = np.array(
            [
                np.sin(alpha) * np.cos(delta),
                np.sin(alpha) * np.sin(delta),
                np.cos(alpha),
            ]
        )

        # D polarization vector
        u = np.array(
            [
                np.cos(psi) * np.cos(alpha) * np.cos(delta)
                - np.sin(psi) * np.sin(delta),
                np.cos(psi) * np.cos(alpha) * np.sin(delta)
                + np.sin(psi) * np.cos(delta),
                -np.cos(psi) * np.sin(alpha),
            ]
        )

        kO1i = np.zeros((3, i.size), dtype=complex)
        kE1i = np.zeros_like(kO1i)
        kO3i = np.zeros_like(kO1i)
        kE3i = np.zeros_like(kO1i)

        step_T = np.zeros((2 * nood, 2 * nood, nlayers), dtype=complex)

        dlt = (i == 0).astype(int)

        def _passive_sqrt(val):
            rt = np.sqrt(np.asarray(val, dtype=complex)).item()
            if rt.real - rt.imag < 0 or (rt.real - rt.imag == 0 and rt.real < 0):
                rt = -rt
            return rt

        def _end_layer_indices(mat, wl):
            if hasattr(mat, "n"):
                n_val = mat.n(wl).item()
                return n_val, n_val
            eps_rel = mat.epsilonTensor(wl)[:, :, 0] / eps0
            if not np.allclose(eps_rel, np.diag(np.diag(eps_rel))) or not np.isclose(
                eps_rel[1, 1], eps_rel[2, 2]
            ):
                raise NotImplementedError(
                    "AnisotropicRCWA end layers only support isotropic or x-axis uniaxial materials."
                )
            return _passive_sqrt(eps_rel[1, 1]), _passive_sqrt(eps_rel[0, 0])

        def _li_transform(eps):
            inv_exx = 1.0 / eps[0, 0]
            L = np.empty((3, 3), dtype=complex)
            L[0, 0] = inv_exx
            L[0, 1] = eps[0, 1] * inv_exx
            L[0, 2] = eps[0, 2] * inv_exx
            L[1, 0] = eps[1, 0] * inv_exx
            L[2, 0] = eps[2, 0] * inv_exx
            L[1, 1] = eps[1, 1] - eps[1, 0] * eps[0, 1] * inv_exx
            L[1, 2] = eps[1, 2] - eps[1, 0] * eps[0, 2] * inv_exx
            L[2, 1] = eps[2, 1] - eps[2, 0] * eps[0, 1] * inv_exx
            L[2, 2] = eps[2, 2] - eps[2, 0] * eps[0, 2] * inv_exx
            return L

        def _toep(arr):
            return toeplitz(arr[hmax::-1], arr[hmax:])

        for iwl, wl in enumerate(self.wls):
            nO1, nE1 = _end_layer_indices(multilayer[0].mat, wl)
            nO3, nE3 = _end_layer_indices(multilayer[-1].mat, wl)
            iso_sup = bool(np.isclose(nO1, nE1, rtol=1e-12, atol=1e-14))
            iso_sub = bool(np.isclose(nO3, nE3, rtol=1e-12, atol=1e-14))

            # wavevectors
            k = 2 * pi / wl

            eps1 = np.diag(np.asarray([nO1 if iso_sup else nE1, nO1, nO1]) ** 2)
            eps3 = np.diag(np.asarray([nO3 if iso_sub else nE3, nO3, nO3]) ** 2)

            # Precompute wavelength-only Fourier permittivity blocks per internal layer
            layer_ops = []
            h_idx = np.arange(-hmax, hmax + 1)
            for nlayer in range(nlayers - 2, 0, -1):
                layer = multilayer[nlayer]
                thickness = layer.thickness
                if isinstance(layer, BinaryGrating):
                    e1 = layer.mat1.epsilonTensor(wl)[:, :, 0] / eps0
                    e2 = layer.mat2.epsilonTensor(wl)[:, :, 0] / eps0
                    L1, L2 = _li_transform(e1), _li_transform(e2)
                    f_dc = layer.dc
                    LF = (L1 - L2)[:, :, None] * (f_dc * np.sinc(h_idx * f_dc))[
                        None, None, :
                    ] + L2[:, :, None] * (h_idx == 0)[None, None, :]
                elif isinstance(layer, SymmetricDoubleGrating):
                    e1 = layer.mat1.epsilonTensor(wl)[:, :, 0] / eps0
                    e2 = layer.mat2.epsilonTensor(wl)[:, :, 0] / eps0
                    e3 = layer.mat3.epsilonTensor(wl)[:, :, 0] / eps0
                    L1, L2, L3 = _li_transform(e1), _li_transform(e2), _li_transform(e3)
                    f1, f2 = layer.dc1, layer.dc2
                    fM = (
                        layer.dcM
                        if isinstance(layer, AsymmetricDoubleGrating)
                        else (1.0 - f1 - f2) / 2.0
                    )
                    nh = len(h_idx)
                    sh_a = -nh * (f1 + fM) / 2.0
                    sh_b = nh * (f2 + fM) / 2.0
                    w1 = f1 * np.sinc(h_idx * f1) * np.exp(2j * pi * h_idx / nh * sh_a)
                    w2 = f2 * np.sinc(h_idx * f2) * np.exp(2j * pi * h_idx / nh * sh_b)
                    LF = (
                        (L1 - L3)[:, :, None] * w1[None, None, :]
                        + (L2 - L3)[:, :, None] * w2[None, None, :]
                        + L3[:, :, None] * (h_idx == 0)[None, None, :]
                    )
                else:
                    EPS2, EPS21 = layer.getEPSFourierCoeffs(wl, n, anisotropic=True)
                    if hmax == 0 or np.all(EPS2[:, :, :hmax] == 0):
                        LF = np.zeros_like(EPS2)
                        LF[:, :, hmax] = _li_transform(EPS2[:, :, hmax])
                    else:
                        LF = None

                if LF is not None:
                    LT = [[_toep(LF[r, c, :]) for c in range(3)] for r in range(3)]
                    Exx_1_1 = inv(LT[0][0])
                    Exy = np.dot(Exx_1_1, LT[0][1])
                    Exz = np.dot(Exx_1_1, LT[0][2])
                    Eyx = np.dot(LT[1][0], Exx_1_1)
                    Ezx = np.dot(LT[2][0], Exx_1_1)
                    Eyy = LT[1][1] + np.dot(Eyx, LT[0][1])
                    Eyz = LT[1][2] + np.dot(Eyx, LT[0][2])
                    Ezy = LT[2][1] + np.dot(Ezx, LT[0][1])
                    Ezz = LT[2][2] + np.dot(Ezx, LT[0][2])
                else:
                    Exy = _toep(EPS2[0, 1, :])
                    Exz = _toep(EPS2[0, 2, :])
                    Eyx = _toep(EPS2[1, 0, :])
                    Eyy = _toep(EPS2[1, 1, :])
                    Eyz = _toep(EPS2[1, 2, :])
                    Ezx = _toep(EPS2[2, 0, :])
                    Ezy = _toep(EPS2[2, 1, :])
                    Ezz = _toep(EPS2[2, 2, :])
                    # EPS21[0, 0] is the Fourier series of (eps^-1)_xx, which equals
                    # 1 / eps_xx only when x has no off-diagonal tensor coupling.
                    # Fall back to Laurent _toep(EPS2[0, 0]) otherwise so the series
                    # converges to the true permittivity tensor as n -> inf.
                    x_decoupled = (
                        np.all(EPS2[0, 1, :] == 0)
                        and np.all(EPS2[0, 2, :] == 0)
                        and np.all(EPS2[1, 0, :] == 0)
                        and np.all(EPS2[2, 0, :] == 0)
                    )
                    if x_decoupled:
                        Exx_1_1 = inv(_toep(EPS21[0, 0, :]))
                    else:
                        Exx_1_1 = _toep(EPS2[0, 0, :])

                Ezz_1 = inv(Ezz)
                layer_ops.append(
                    (
                        nlayer,
                        thickness,
                        Exx_1_1,
                        Exy,
                        Exz,
                        Eyx,
                        Eyy,
                        Eyz,
                        Ezx,
                        Ezy,
                        Ezz_1,
                    )
                )

            def _solve_floquet(k1_vec, E_inc, H_inc):
                kO1i[0, :] = k1_vec[0] - i * K[0]
                kO1i[1, :] = k1_vec[1] * np.ones_like(i)
                kz_o1 = dispersion_relation_ordinary(kO1i[0, :], kO1i[1, :], k, nO1)
                kz_o1 = np.where(np.abs(kz_o1) < 1e-13 * k, -1e-13j * k, kz_o1)
                kO1i[2, :] = -kz_o1

                kE1i[0, :] = kO1i[0, :]
                kE1i[1, :] = kO1i[1, :]
                if iso_sup:
                    kE1i[2, :] = kO1i[2, :]
                else:
                    kz_e1 = dispersion_relation_extraordinary(
                        kE1i[0, :], kE1i[1, :], k, nO1, nE1, c1
                    )
                    kz_e1 = np.where(np.abs(kz_e1) < 1e-13 * k, -1e-13j * k, kz_e1)
                    kE1i[2, :] = -kz_e1

                kO3i[0, :] = kO1i[0, :]
                kO3i[1, :] = kO1i[1, :]
                kz_o3 = dispersion_relation_ordinary(kO3i[0, :], kO3i[1, :], k, nO3)
                kO3i[2, :] = np.where(np.abs(kz_o3) < 1e-13 * k, -1e-13j * k, kz_o3)

                kE3i[0, :] = kO1i[0, :]
                kE3i[1, :] = kO1i[1, :]
                if iso_sub:
                    kE3i[2, :] = kO3i[2, :]
                else:
                    kz_e3 = dispersion_relation_extraordinary(
                        kE3i[0, :], kE3i[1, :], k, nO3, nE3, c3
                    )
                    kE3i[2, :] = np.where(np.abs(kz_e3) < 1e-13 * k, -1e-13j * k, kz_e3)

                k2i = np.r_[[k1_vec[0] - i * K[0]], [k1_vec[1] - i * K[1]], [-i * K[2]]]
                kx = kO1i[0, :]
                ky = k1_vec[1]

                I = np.eye(nood, dtype=complex)
                ZERO = np.zeros((nood, nood), dtype=complex)
                Kx = np.diag(kx / k)
                Ky = ky / k * I
                Kz = np.diag(k2i[2, :] / k)
                KO1z = np.diag(kO1i[2, :] / k)
                KE1z = np.diag(kE1i[2, :] / k)
                KO3z = np.diag(kO3i[2, :] / k)
                KE3z = np.diag(kE3i[2, :] / k)

                ARO = Kx * eps1[0, 0] + Ky * eps1[1, 0] + KO1z * eps1[2, 0]
                BRO = Kx * eps1[0, 1] + Ky * eps1[1, 1] + KO1z * eps1[2, 1]
                CRO_1 = inv(Kx * eps1[0, 2] + Ky * eps1[1, 2] + KO1z * eps1[2, 2])

                ARE = Kx * eps1[0, 0] + Ky * eps1[1, 0] + KE1z * eps1[2, 0]
                BRE = Kx * eps1[0, 1] + Ky * eps1[1, 1] + KE1z * eps1[2, 1]
                CRE_1 = inv(Kx * eps1[0, 2] + Ky * eps1[1, 2] + KE1z * eps1[2, 2])

                ATO = Kx * eps3[0, 0] + Ky * eps3[1, 0] + KO3z * eps3[2, 0]
                BTO = Kx * eps3[0, 1] + Ky * eps3[1, 1] + KO3z * eps3[2, 1]
                CTO_1 = inv(Kx * eps3[0, 2] + Ky * eps3[1, 2] + KO3z * eps3[2, 2])

                ATE = Kx * eps3[0, 0] + Ky * eps3[1, 0] + KE3z * eps3[2, 0]
                BTE = Kx * eps3[0, 1] + Ky * eps3[1, 1] + KE3z * eps3[2, 1]
                CTE_1 = inv(Kx * eps3[0, 2] + Ky * eps3[1, 2] + KE3z * eps3[2, 2])

                DRE = c1[1] * KE1z - c1[2] * Ky
                ERE = c1[2] * Kx - c1[0] * KE1z
                FRE = c1[0] * Ky - c1[1] * Kx

                DTE = c3[1] * KE3z - c3[2] * Ky
                ETE = c3[2] * Kx - c3[0] * KE3z
                FTE = c3[0] * Ky - c3[1] * Kx

                b = np.r_[
                    E_inc[0] * dlt,
                    E_inc[1] * dlt,
                    H_inc[0] * dlt,
                    H_inc[1] * dlt,
                ]

                # Singularity-free 4N x 2N reduced end-layer matrices spanning the exact
                # 2N outgoing eigenmode subspace (c1 = c3 = [1, 0, 0]) with unit-conditioned
                # tangential electric block [[ZERO, I], [I, ZERO]], remaining full-rank both
                # on isotropic end layers (nE == nO) and at the conical evanescent uniaxial
                # exceptional point Kx^2 == nO^2, Ky != 0 (where (KEz - KOz)/(nO^2 - Kx^2)
                # cancels analytically to (nE^2 - nO^2) / (nO^2 * (KEz + KOz))).
                def _build_end_red(kzO_signed, kzE_signed, nO, nE, iso_flag):
                    Kx_d = kx / k
                    Ky_d = ky / k
                    KOz_d = kzO_signed / k
                    KEz_d = KOz_d if iso_flag else kzE_signed / k
                    m31 = -(Ky_d**2) / KOz_d - KOz_d
                    m41 = (Kx_d * Ky_d) / KOz_d
                    if iso_flag:
                        m32 = -(Kx_d * Ky_d) / KOz_d
                        m42 = (Kx_d**2) / KOz_d + KOz_d
                    else:
                        dn2 = nE**2 - nO**2
                        denom_sum = KEz_d + KOz_d
                        m32 = -(Kx_d * Ky_d * nE**2) / (nO**2 * KEz_d) - (
                            Kx_d * Ky_d * dn2
                        ) / (nO**2 * denom_sum) * (
                            (Ky_d**2) / (KEz_d * KOz_d) - 1.0
                        )
                        m42 = (
                            (Kx_d**2 * nE**2) / (nO**2 * KEz_d)
                            + KEz_d
                            + (Kx_d**2 * Ky_d**2 * dn2)
                            / (nO**2 * KEz_d * KOz_d * denom_sum)
                        )
                    return np.block(
                        [
                            [ZERO, I],
                            [I, ZERO],
                            [np.diag(m31), np.diag(m32)],
                            [np.diag(m41), np.diag(m42)],
                        ]
                    )

                MR_red = _build_end_red(kO1i[2, :], kE1i[2, :], nO1, nE1, iso_sup)
                MT_red = _build_end_red(kO3i[2, :], kE3i[2, :], nO3, nE3, iso_sub)
                MT_curr = MT_red.copy()
                step_T.fill(0.0)

                for (
                    nlayer,
                    thickness,
                    Exx_1_1,
                    Exy,
                    Exz,
                    Eyx,
                    Eyy,
                    Eyz,
                    Ezx,
                    Ezy,
                    Ezz_1,
                ) in layer_ops:
                    Ky_Ezz_1 = ky / k * Ezz_1
                    Kx_Ezz_1 = kx[:, np.newaxis] / k * Ezz_1
                    Exz_Ezz_1 = np.dot(Exz, Ezz_1)
                    Eyz_Ezz_1 = np.dot(Eyz, Ezz_1)
                    H11 = 1j * np.dot(Ky_Ezz_1, Ezy)
                    H12 = 1j * np.dot(Ky_Ezz_1, Ezx)
                    H13 = np.dot(Ky_Ezz_1, Kx)
                    H14 = I - np.dot(Ky_Ezz_1, Ky)
                    H21 = 1j * np.dot(Kx_Ezz_1, Ezy)
                    H22 = 1j * np.dot(Kx_Ezz_1, Ezx)
                    H23 = np.dot(Kx_Ezz_1, Kx) - I
                    H24 = -np.dot(Kx_Ezz_1, Ky)
                    H31 = np.dot(Kx, Ky) + Exy - np.dot(Exz_Ezz_1, Ezy)
                    H32 = Exx_1_1 - np.dot(Ky, Ky) - np.dot(Exz_Ezz_1, Ezx)
                    H33 = 1j * np.dot(Exz_Ezz_1, Kx)
                    H34 = -1j * np.dot(Exz_Ezz_1, Ky)
                    H41 = np.dot(Kx, Kx) - Eyy + np.dot(Eyz_Ezz_1, Ezy)
                    H42 = -np.dot(Kx, Ky) - Eyx + np.dot(Eyz_Ezz_1, Ezx)
                    H43 = -1j * np.dot(Eyz_Ezz_1, Kx)
                    H44 = 1j * np.dot(Eyz_Ezz_1, Ky)
                    H = 1j * np.diag(np.repeat(np.diag(Kz), 4)) + np.block(
                        [
                            [H11, H12, H13, H14],
                            [H21, H22, H23, H24],
                            [H31, H32, H33, H34],
                            [H41, H42, H43, H44],
                        ]
                    )

                    q, W = eig(H)
                    W1, W2, W3, W4 = np.split(W, 4)
                    W_full = np.r_[W2, W1, -1j * W4, -1j * W3]

                    re_tol = 1e-10 * max(1.0, float(np.max(np.abs(q))))
                    sz = np.sum(
                        np.real(
                            W_full[0:nood, :] * np.conj(W_full[3 * nood : 4 * nood, :])
                            - W_full[nood : 2 * nood, :]
                            * np.conj(W_full[2 * nood : 3 * nood, :])
                        ),
                        axis=0,
                    )
                    w_norm2 = np.sum(np.abs(W_full) ** 2, axis=0)

                    def _mode_sort_key(idx):
                        qr = float(q[idx].real)
                        qi = float(q[idx].imag)
                        # State-matrix modes evolve as exp(+k q z) = exp(-i k_z z), so
                        # q = -i k_z / k. For lossless propagating modes (|Re(q)| <= re_tol),
                        # normal Poynting flux S_z classifies forward vs. backward waves
                        # (preserving energy-velocity direction in tilted anisotropic media).
                        # Otherwise, classify by the passive branch Re(k_z) - Im(k_z) >= 0,
                        # i.e. -(qr + qi) >= 0, rather than Re(q) < 0 alone: under oblique
                        # incidence from a lossy superstrate (Im(k_x) < 0), forward waves in
                        # a lossless internal layer have Re(k_z) > Im(k_z) > 0, giving
                        # qr = +Im(k_z)/k > 0 and qi = -Re(k_z)/k < 0 (Quadrant IV).
                        if abs(qr) <= re_tol:
                            sz_rel = float(sz[idx]) / max(float(w_norm2[idx]), 1e-30)
                            if sz_rel > 1e-10:
                                cat = 1
                            elif sz_rel < -1e-10:
                                cat = -1
                            else:
                                cat = 0
                        else:
                            passive = -(qr + qi)
                            if passive > re_tol:
                                cat = 2
                            elif passive < -re_tol:
                                cat = -2
                            else:
                                sz_rel = float(sz[idx]) / max(
                                    float(w_norm2[idx]), 1e-30
                                )
                                if sz_rel > 1e-10:
                                    cat = 1
                                elif sz_rel < -1e-10:
                                    cat = -1
                                else:
                                    cat = 0
                        return (cat, -qi, -qr)

                    order = sorted(range(4 * nood), key=_mode_sort_key, reverse=True)
                    q_f = q[order[: 2 * nood]]
                    q_b = q[order[2 * nood :]]
                    W_ord = W_full[:, order]
                    W_f = W_ord[:, : 2 * nood]
                    W_b = W_ord[:, 2 * nood :]

                    x_f = np.exp(k * q_f * thickness)
                    x_b = np.exp(-k * q_b * thickness)
                    D1_inv_4 = np.tile(np.exp(1j * k2i[2, :] * thickness), 4)

                    MTp = linsolve(W_ord, D1_inv_4[:, np.newaxis] * MT_curr)
                    MTp1 = MTp[: 2 * nood, :]
                    MTp2 = MTp[2 * nood :, :]
                    step_T[:, :, nlayer] = linsolve(MTp1, np.diag(x_f))
                    MT_curr = W_f + np.dot(
                        W_b * x_b[np.newaxis, :], np.dot(MTp2, step_T[:, :, nlayer])
                    )

                sol = linsolve(np.c_[-MR_red, MT_curr], b)
                r_red = sol[: 2 * nood]
                t_curr = sol[2 * nood :]
                for nlayer in range(1, nlayers - 1):
                    t_curr = np.dot(step_T[:, :, nlayer], t_curr)

                r_y, r_x = np.split(r_red, 2)
                t_y, t_x = np.split(t_curr, 2)
                HR_tot_x, HR_tot_y = np.split(np.dot(MR_red[2 * nood :, :], r_red), 2)
                HT_tot_x, HT_tot_y = np.split(np.dot(MT_red[2 * nood :, :], t_curr), 2)

                kt = np.sqrt(kx**2 + ky**2 + 0j)
                flip_kt = (kt.real - kt.imag < 0) | (
                    (kt.real - kt.imag == 0) & (kt.real < 0)
                )
                kt = np.where(flip_kt, -kt, kt)
                zero_kt = np.abs(kt) <= 1e-13 * k
                kt_safe = np.where(zero_kt, 1.0 + 0j, kt)
                fc = np.where(zero_kt, np.cos(delta) + 0j, kx / kt_safe)
                fs = np.where(zero_kt, np.sin(delta) + 0j, ky / kt_safe)
                sx_hat = -fs
                sy_hat = fc

                if iso_sup:
                    ROz_tot = -np.dot(CRO_1, (np.dot(ARO, r_x) + np.dot(BRO, r_y)))
                    R_tot_vec = np.vstack([r_x, r_y, ROz_tot])
                    r_s_amp = R_tot_vec[0, :] * sx_hat + R_tot_vec[1, :] * sy_hat
                    RO_vec = np.vstack(
                        [r_s_amp * sx_hat, r_s_amp * sy_hat, 0 * r_s_amp]
                    )
                    RE_vec = R_tot_vec - RO_vec
                    HR_O = np.cross(kO1i / k, RO_vec, axis=0)
                    HR_E = np.cross(kE1i / k, RE_vec, axis=0)
                    HR_tot = HR_O + HR_E
                else:
                    q1_r = np.diag(DRE - np.dot(np.dot(FRE, CRE_1), ARE))
                    p1 = np.diag(ERE - np.dot(np.dot(FRE, CRE_1), BRE))
                    nrm1 = np.hypot(np.abs(p1), np.abs(q1_r))
                    ep1 = np.abs(p1) <= 1e-12 * np.maximum(nrm1, 1e-30)
                    p1_safe = np.where(ep1, 1.0 + 0j, p1)
                    ratio1 = np.where(ep1, 0.0 + 0j, q1_r / p1_safe)
                    ROx = np.where(ep1, r_x, 0.0 + 0j)
                    ROy = np.where(ep1, r_y, r_y + ratio1 * r_x)
                    REx = np.where(ep1, 0.0 + 0j, r_x)
                    REy = np.where(ep1, 0.0 + 0j, -ratio1 * r_x)
                    ROz = -np.dot(CRO_1, (np.dot(ARO, ROx) + np.dot(BRO, ROy)))
                    REz = -np.dot(CRE_1, (np.dot(ARE, REx) + np.dot(BRE, REy)))
                    RO_vec = np.vstack([ROx, ROy, ROz])
                    RE_vec = np.vstack([REx, REy, REz])
                    R_tot_vec = RO_vec + RE_vec
                    HR_O = np.cross(kO1i / k, RO_vec, axis=0)
                    HR_E = np.cross(kE1i / k, RE_vec, axis=0)
                    HR_O[0, :] = np.where(ep1, HR_tot_x, HR_O[0, :])
                    HR_O[1, :] = np.where(ep1, HR_tot_y, HR_O[1, :])
                    HR_E[0, :] = np.where(ep1, 0.0 + 0j, HR_E[0, :])
                    HR_E[1, :] = np.where(ep1, 0.0 + 0j, HR_E[1, :])
                    HR_tot = np.vstack([HR_tot_x, HR_tot_y, HR_O[2, :] + HR_E[2, :]])

                if iso_sub:
                    TOz_tot = -np.dot(CTO_1, (np.dot(ATO, t_x) + np.dot(BTO, t_y)))
                    T_tot_vec = np.vstack([t_x, t_y, TOz_tot])
                    t_s_amp = T_tot_vec[0, :] * sx_hat + T_tot_vec[1, :] * sy_hat
                    TO_vec = np.vstack(
                        [t_s_amp * sx_hat, t_s_amp * sy_hat, 0 * t_s_amp]
                    )
                    TE_vec = T_tot_vec - TO_vec
                    HT_O = np.cross(kO3i / k, TO_vec, axis=0)
                    HT_E = np.cross(kE3i / k, TE_vec, axis=0)
                    HT_tot = HT_O + HT_E
                else:
                    q3_t = np.diag(DTE - np.dot(np.dot(FTE, CTE_1), ATE))
                    p3 = np.diag(ETE - np.dot(np.dot(FTE, CTE_1), BTE))
                    nrm3 = np.hypot(np.abs(p3), np.abs(q3_t))
                    ep3 = np.abs(p3) <= 1e-12 * np.maximum(nrm3, 1e-30)
                    p3_safe = np.where(ep3, 1.0 + 0j, p3)
                    ratio3 = np.where(ep3, 0.0 + 0j, q3_t / p3_safe)
                    TOx = np.where(ep3, t_x, 0.0 + 0j)
                    TOy = np.where(ep3, t_y, t_y + ratio3 * t_x)
                    TEx = np.where(ep3, 0.0 + 0j, t_x)
                    TEy = np.where(ep3, 0.0 + 0j, -ratio3 * t_x)
                    TOz = -np.dot(CTO_1, (np.dot(ATO, TOx) + np.dot(BTO, TOy)))
                    TEz = -np.dot(CTE_1, (np.dot(ATE, TEx) + np.dot(BTE, TEy)))
                    TO_vec = np.vstack([TOx, TOy, TOz])
                    TE_vec = np.vstack([TEx, TEy, TEz])
                    T_tot_vec = TO_vec + TE_vec
                    HT_O = np.cross(kO3i / k, TO_vec, axis=0)
                    HT_E = np.cross(kE3i / k, TE_vec, axis=0)
                    HT_O[0, :] = np.where(ep3, HT_tot_x, HT_O[0, :])
                    HT_O[1, :] = np.where(ep3, HT_tot_y, HT_O[1, :])
                    HT_E[0, :] = np.where(ep3, 0.0 + 0j, HT_E[0, :])
                    HT_E[1, :] = np.where(ep3, 0.0 + 0j, HT_E[1, :])
                    HT_tot = np.vstack([HT_tot_x, HT_tot_y, HT_O[2, :] + HT_E[2, :]])

                denom = float(
                    (E_inc[0] * np.conj(H_inc[1]) - E_inc[1] * np.conj(H_inc[0])).real
                )
                num_deo1 = -(
                    RO_vec[0, :] * np.conj(HR_O[1, :])
                    - RO_vec[1, :] * np.conj(HR_O[0, :])
                ).real
                num_dee1 = -(
                    RE_vec[0, :] * np.conj(HR_E[1, :])
                    - RE_vec[1, :] * np.conj(HR_E[0, :])
                ).real
                num_de1 = -(
                    R_tot_vec[0, :] * np.conj(HR_tot[1, :])
                    - R_tot_vec[1, :] * np.conj(HR_tot[0, :])
                ).real
                num_deo3 = (
                    TO_vec[0, :] * np.conj(HT_O[1, :])
                    - TO_vec[1, :] * np.conj(HT_O[0, :])
                ).real
                num_dee3 = (
                    TE_vec[0, :] * np.conj(HT_E[1, :])
                    - TE_vec[1, :] * np.conj(HT_E[0, :])
                ).real
                num_de3 = (
                    T_tot_vec[0, :] * np.conj(HT_tot[1, :])
                    - T_tot_vec[1, :] * np.conj(HT_tot[0, :])
                ).real

                return denom, num_deo1, num_dee1, num_de1, num_deo3, num_dee3, num_de3

            # ordinary and extraordinary incident wavevector magnitudes in region 1
            abskO1 = k * nO1
            abskE1 = k * _passive_sqrt(
                (nO1**2 * nE1**2)
                / (nO1**2 + (nE1**2 - nO1**2) * np.dot(c1, dirk1) ** 2)
            )

            c_cross_s = np.cross(c1, dirk1)
            norm_cs = float(np.linalg.norm(c_cross_s))
            aniso_superstrate = not iso_sup and norm_cs > 1e-12

            if not aniso_superstrate:
                k1_inc = abskO1 * dirk1
                E_inc = u.astype(complex)
                H_inc = np.cross(k1_inc / k, E_inc)
                (
                    denom,
                    n_deo1,
                    n_dee1,
                    n_de1,
                    n_deo3,
                    n_dee3,
                    n_de3,
                ) = _solve_floquet(k1_inc, E_inc, H_inc)
            else:
                dO_hat = c_cross_s / norm_cs
                dE_hat = np.cross(dirk1, dO_hat)
                aO = np.dot(dO_hat, u)
                aE = np.dot(dE_hat, u)

                kO1_inc = np.asarray(abskO1 * dirk1, dtype=complex)
                kE1_inc = np.asarray(abskE1 * dirk1, dtype=complex)

                EO_inc = aO * dO_hat.astype(complex)
                HO_inc = np.cross(kO1_inc / k, EO_inc)

                EE_inc = aE * np.array(
                    [(nO1 / nE1) ** 2 * dE_hat[0], dE_hat[1], dE_hat[2]],
                    dtype=complex,
                )
                HE_inc = np.cross(kE1_inc / k, EE_inc)

                if np.allclose(kO1_inc[:2], kE1_inc[:2], rtol=0.0, atol=1e-14 * k):
                    # Normal incidence (k_parallel == 0): O and E share one Floquet space
                    (
                        denom,
                        n_deo1,
                        n_dee1,
                        n_de1,
                        n_deo3,
                        n_dee3,
                        n_de3,
                    ) = _solve_floquet(kO1_inc, EO_inc + EE_inc, HO_inc + HE_inc)
                elif abs(aE) <= 1e-15:
                    (
                        denom,
                        n_deo1,
                        n_dee1,
                        n_de1,
                        n_deo3,
                        n_dee3,
                        n_de3,
                    ) = _solve_floquet(kO1_inc, EO_inc, HO_inc)
                elif abs(aO) <= 1e-15:
                    (
                        denom,
                        n_deo1,
                        n_dee1,
                        n_de1,
                        n_deo3,
                        n_dee3,
                        n_de3,
                    ) = _solve_floquet(kE1_inc, EE_inc, HE_inc)
                else:
                    # Oblique mixed O+E incidence in a birefringent superstrate: solve
                    # the two distinct tangential-wavevector Floquet problems independently
                    # and superpose their period-averaged fluxes weighted by incident Sz.
                    res_O = _solve_floquet(kO1_inc, EO_inc, HO_inc)
                    res_E = _solve_floquet(kE1_inc, EE_inc, HE_inc)
                    denom = res_O[0] + res_E[0]
                    n_deo1 = res_O[1] + res_E[1]
                    n_dee1 = res_O[2] + res_E[2]
                    n_de1 = res_O[3] + res_E[3]
                    n_deo3 = res_O[4] + res_E[4]
                    n_dee3 = res_O[5] + res_E[5]
                    n_de3 = res_O[6] + res_E[6]

            DEO1[:, iwl] = n_deo1 / denom
            DEE1[:, iwl] = n_dee1 / denom
            DE1[:, iwl] = n_de1 / denom
            DEO3[:, iwl] = n_deo3 / denom
            DEE3[:, iwl] = n_dee3 / denom
            DE3[:, iwl] = n_de3 / denom

        # save the results
        self.DEO1 = DEO1
        self.DEE1 = DEE1
        self.DE1 = DE1
        self.DEO3 = DEO3
        self.DEE3 = DEE3
        self.DE3 = DE3

        return self

    # def plot(self):
    #         """Plot the diffraction efficiencies."""
    #         g = Gnuplot.Gnuplot()
    #         g('set xlabel "$\lambda$"')
    #         g('set ylabel "diffraction efficiency"')
    #         g('set yrange [0:1]')
    #         g('set data style linespoints')
    #         g.plot(Gnuplot.Data(self.wls, self.DEO1[self.n,:], with_ = 'linespoints', title = 'DEO1'), \
    #                Gnuplot.Data(self.wls, self.DEO3[self.n,:], with_ = 'linespoints', title = 'DEO3'), \
    #                Gnuplot.Data(self.wls, self.DEE1[self.n,:], with_ = 'linespoints', title = 'DEE1'), \
    #                Gnuplot.Data(self.wls, self.DEE3[self.n,:], with_ = 'linespoints', title = 'DEE3'))
    #         raw_input('press enter to close the graph...')

    def __str__(self):
        return (
            "ANISOTROPIC RCWA SOLVER\n\n%s\n\nLAMBDA = %g\nalpha = %g\ndelta = %g\npsi = %g\nphi = %g\nn = %d"
            % (
                self.multilayer.__str__(),
                self.LAMBDA,
                self.alpha,
                self.delta,
                self.psi,
                self.phi,
                self.n,
            )
        )
