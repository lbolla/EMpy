"""Rigorous Coupled Wave Analysis.

The algorithm, described in Glytsis, "Three-dimensional (vector)
rigorous coupled-wave analysis of anisotropic grating diffraction",
JOSA A, 7(8), 1990, permits to find the diffraction efficiencies
(i.e. the power normalized to unity) reflected and transmitted by a
multilayer.
The multilayer can be made of isotropic or anisotropic layers and
binary gratings.
Two versions of the algorithm are present: an isotropic one, stable
for every diffraction order and layer thickness, and an anisotropic
one, only stable for low diffraction orders and thin layers.

"""
__author__ = "Lorenzo Bolla"

import numpy as np
from scipy.linalg import toeplitz, inv, eig, solve as linsolve
from numpy import pi

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

    delta = (k * nO) ** 2 - (kx**2 + ky**2)
    kz = np.sqrt(delta)

    # Adjust sign of real/imag part
    kz.real = abs(kz.real)
    kz.imag = -abs(kz.imag)

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

    kz = np.empty_like(kx)

    for ii in range(0, kx.size):
        alpha = nE**2 - nO**2
        beta = kx[ii] / k * c[0] + ky[ii] / k * c[1]

        # coeffs
        C = np.array(
            [
                nO**2 + c[2] ** 2 * alpha,
                2.0 * c[2] * beta * alpha,
                nO**2 * (kx[ii] ** 2 + ky[ii] ** 2) / k**2
                + alpha * beta**2
                - nO**2 * nE**2,
            ]
        )

        # two solutions of type +x or -x, purely real or purely imag
        tmp_kz = k * np.roots(C)

        # get the negative imaginary part or the positive real one
        if np.any(np.isreal(tmp_kz)):
            kz[ii] = np.absolute(tmp_kz[0])
        else:
            kz[ii] = -1j * np.absolute(tmp_kz[0])

    return kz


class RCWA:

    """Class to handle the RCWA solvers.

    NOTE
    See Glytsis, "Three-dimensional (vector) rigorous coupled-wave
    analysis of anisotropic grating diffraction", JOSA A, 7(8), 1990

    The following variables, used throughout the code, have the following
    meaning:

    alpha:  float
            angle between wave vector k1 and xy plane, in radians

    delta:  float
            angle between the y axis and the projection of k1 onto the xy
            plane, in radians

    psi:    angle between the D vector of the plane wave and the xy plane,
            in radians, TM: 0, TE: numpy.pi / 2

    phi:    angle between the grating vector K and y axis (in the xy plane),
            in radians, the grating is modulated in the direction of the
            grating vector.

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
            [np.sin(alpha) * np.cos(delta), np.sin(alpha) * np.sin(delta), np.cos(alpha)]
        )

        # usefull matrices
        I = np.eye(i.size)
        I2 = np.eye(i.size * 2)
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

            # all the other wavevectors
            tmp_x = k1[0] - i * K[0]
            tmp_y = k1[1] * np.ones_like(i)
            tmp_z = dispersion_relation_ordinary(tmp_x, tmp_y, k, n1)
            k1i = np.r_[[tmp_x], [tmp_y], [tmp_z]]

            # k2i = np.r_[[k1[0] - i*K[0]], [k1[1] - i * K[1]], [-i * K[2]]]

            tmp_z = dispersion_relation_ordinary(tmp_x, tmp_y, k, n3)
            k3i = np.r_[[k1i[0, :]], [k1i[1, :]], [tmp_z]]

            # aliases for constant wavevectors
            kx = k1i[0, :]
            ky = k1[1]

            # angles of reflection
            # phi_i = np.arctan2(ky,kx)
            phi_i = np.arctan2(ky, kx.real)  # OKKIO

            Kx = np.diag(kx / k)
            Ky = ky / k * I
            Z1 = np.diag(k1i[2, :] / (k * n1**2))
            Y1 = np.diag(k1i[2, :] / k)
            Z3 = np.diag(k3i[2, :] / (k * n3**2))
            Y3 = np.diag(k3i[2, :] / k)
            # Fc = np.diag(np.cos(phi_i))
            fc = np.cos(phi_i)
            # Fs = np.diag(np.sin(phi_i))
            fs = np.sin(phi_i)

            MR = np.asarray(
                np.bmat([[I, ZERO], [-1j * Y1, ZERO], [ZERO, I], [ZERO, -1j * Z1]])
            )

            MT = np.asarray(
                np.bmat([[I, ZERO], [1j * Y3, ZERO], [ZERO, I], [ZERO, 1j * Z3]])
            )

            # internal layers (grating or layer)
            X.fill(0.0)
            MTp1.fill(0.0)
            MTp2.fill(0.0)
            for nlayer in range(nlayers - 2, 0, -1):  # internal layers
                layer = multilayer[nlayer]
                d = layer.thickness

                EPS2, EPS21 = layer.getEPSFourierCoeffs(wl, n, anisotropic=False)

                E = toeplitz(EPS2[hmax::-1], EPS2[hmax:])
                E1 = toeplitz(EPS21[hmax::-1], EPS21[hmax:])
                E11 = inv(E1)
                # B = np.dot(Kx, linsolve(E,Kx)) - I
                B = kx[:, np.newaxis] / k * linsolve(E, Kx) - I
                # A = np.dot(Kx, Kx) - E
                A = np.diag((kx / k) ** 2) - E

                # Note: solution bug alfredo
                # randomizzo Kx un po' a caso finche' cond(A) e' piccolo (<1e10)
                # soluzione sporca... :-(
                # per certi kx, l'operatore di helmholtz ha 2 autovalori nulli e A, B
                # non sono invertibili --> cambio leggermente i kx... ma dovrei invece
                # trattare separatamente (analiticamente) questi casi
                if cond(A) > 1e10:
                    warning("BAD CONDITIONING: randomization of kx")
                    while cond(A) > 1e10:
                        Kx = Kx * (1 + 1e-9 * np.rand())
                        B = kx[:, np.newaxis] / k * linsolve(E, Kx) - I
                        A = np.diag((kx / k) ** 2) - E

                if np.absolute(K[2] / k) > 1e-10:
                    raise ValueError(
                        "First Order Helmholtz Operator not implemented, yet!"
                    )

                elif ky == 0 or np.allclose(np.diag(Ky / ky * k), 1):
                    # lalanne
                    # H_U_reduced = np.dot(Ky, Ky) + A
                    H_U_reduced = (ky / k) ** 2 * I + A
                    # H_S_reduced = np.dot(Ky, Ky) + np.dot(Kx, linsolve(E, np.dot(Kx, E11))) - E11
                    H_S_reduced = (
                        (ky / k) ** 2 * I
                        + kx[:, np.newaxis] / k * linsolve(E, kx[:, np.newaxis] / k * E11)
                        - E11
                    )

                    q1, W1 = eig(H_U_reduced)
                    q1 = np.sqrt(q1)
                    q2, W2 = eig(H_S_reduced)
                    q2 = np.sqrt(q2)

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

                    Mc2bar = np.asarray(
                        np.bmat(
                            [
                                [Vss, Vsp, Vss, Vsp],
                                [Wss, Wsp, -Wss, -Wsp],
                                [Wps, Wpp, -Wps, -Wpp],
                                [Vps, Vpp, Vps, Vpp],
                            ]
                        )
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
                    ValueError("Second Order Helmholtz Operator not implemented, yet!")

            # M = np.asarray(np.bmat([-MR, MT]))
            M = np.c_[-MR, MT]
            b = np.r_[
                np.sin(psi) * dlt,
                1j * np.sin(psi) * n1 * np.cos(alpha) * dlt,
                -1j * np.cos(psi) * n1 * dlt,
                np.cos(psi) * np.cos(alpha) * dlt,
            ]

            x = linsolve(M, b)
            R, T = np.split(x, 2)
            Rs, Rp = np.split(R, 2)
            for ii in range(1, nlayers - 1):
                T = np.dot(linsolve(MTp1[:, :, ii], X[:, :, ii]), T)
            Ts, Tp = np.split(T, 2)

            DE1[:, iwl] = (k1i[2, :] / (k1[2])).real * np.absolute(Rs) ** 2 + (
                k1i[2, :] / (k1[2] * n1**2)
            ).real * np.absolute(Rp) ** 2
            DE3[:, iwl] = (k3i[2, :] / (k1[2])).real * np.absolute(Ts) ** 2 + (
                k3i[2, :] / (k1[2] * n3**2)
            ).real * np.absolute(Tp) ** 2

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

        nlayers = len(multilayer)
        i = np.arange(-n, n + 1)
        nood = 2 * n + 1
        hmax = nood - 1

        DEO1 = np.zeros((nood, self.wls.size))
        DEO3 = np.zeros_like(DEO1)
        DEE1 = np.zeros_like(DEO1)
        DEE3 = np.zeros_like(DEO1)

        c1 = np.array([1.0, 0.0, 0.0])
        c3 = np.array([1.0, 0.0, 0.0])
        # grating on the xy plane
        K = 2 * pi / LAMBDA * np.array([np.sin(phi), 0.0, np.cos(phi)], dtype=complex)
        dirk1 = np.array(
            [np.sin(alpha) * np.cos(delta), np.sin(alpha) * np.sin(delta), np.cos(alpha)]
        )

        # D polarization vector
        u = np.array(
            [
                np.cos(psi) * np.cos(alpha) * np.cos(delta) - np.sin(psi) * np.sin(delta),
                np.cos(psi) * np.cos(alpha) * np.sin(delta) + np.sin(psi) * np.cos(delta),
                -np.cos(psi) * np.sin(alpha),
            ]
        )

        kO1i = np.zeros((3, i.size), dtype=complex)
        kE1i = np.zeros_like(kO1i)
        kO3i = np.zeros_like(kO1i)
        kE3i = np.zeros_like(kO1i)

        Mp = np.zeros((4 * nood, 4 * nood, nlayers), dtype=complex)
        M = np.zeros((4 * nood, 4 * nood, nlayers), dtype=complex)

        dlt = (i == 0).astype(int)

        for iwl, wl in enumerate(self.wls):
            nO1 = nE1 = multilayer[0].mat.n(wl).item()
            nO3 = nE3 = multilayer[-1].mat.n(wl).item()

            # wavevectors
            k = 2 * pi / wl

            eps1 = np.diag(np.asarray([nE1, nO1, nO1]) ** 2)
            eps3 = np.diag(np.asarray([nE3, nO3, nO3]) ** 2)

            # ordinary wave
            abskO1 = k * nO1
            # abskO3 = k * nO3
            # extraordinary wave
            # abskE1 = k * nO1 *nE1 / np.sqrt(nO1**2 + (nE1**2 - nO1**2) * np.dot(-c1, dirk1)**2)
            # abskE3 = k * nO3 *nE3 / np.sqrt(nO3**2 + (nE3**2 - nO3**2) * np.dot(-c3, dirk1)**2)

            k1 = abskO1 * dirk1

            kO1i[0, :] = k1[0] - i * K[0]
            kO1i[1, :] = k1[1] * np.ones_like(i)
            kO1i[2, :] = -dispersion_relation_ordinary(kO1i[0, :], kO1i[1, :], k, nO1)

            kE1i[0, :] = kO1i[0, :]
            kE1i[1, :] = kO1i[1, :]
            kE1i[2, :] = -dispersion_relation_extraordinary(
                kE1i[0, :], kE1i[1, :], k, nO1, nE1, c1
            )

            kO3i[0, :] = kO1i[0, :]
            kO3i[1, :] = kO1i[1, :]
            kO3i[2, :] = dispersion_relation_ordinary(kO3i[0, :], kO3i[1, :], k, nO3)

            kE3i[0, :] = kO1i[0, :]
            kE3i[1, :] = kO1i[1, :]
            kE3i[2, :] = dispersion_relation_extraordinary(
                kE3i[0, :], kE3i[1, :], k, nO3, nE3, c3
            )

            # k2i = np.r_[[k1[0] - i * K[0]], [k1[1] - i * K[1]], [k1[2] - i * K[2]]]
            k2i = np.r_[[k1[0] - i * K[0]], [k1[1] - i * K[1]], [-i * K[2]]]

            # aliases for constant wavevectors
            kx = kO1i[0, :]  # o kE1i(1,;), tanto e' lo stesso
            ky = k1[1]

            # matrices
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
                u[0] * dlt,
                u[1] * dlt,
                (k1[1] / k * u[2] - k1[2] / k * u[1]) * dlt,
                (k1[2] / k * u[0] - k1[0] / k * u[2]) * dlt,
            ]
            Ky_CRO_1 = ky / k * CRO_1
            Ky_CRE_1 = ky / k * CRE_1
            Kx_CRO_1 = kx[:, np.newaxis] / k * CRO_1
            Kx_CRE_1 = kx[:, np.newaxis] / k * CRE_1
            MR31 = -np.dot(Ky_CRO_1, ARO)
            MR32 = -np.dot(Ky_CRO_1, BRO) - KO1z
            MR33 = -np.dot(Ky_CRE_1, ARE)
            MR34 = -np.dot(Ky_CRE_1, BRE) - KE1z
            MR41 = np.dot(Kx_CRO_1, ARO) + KO1z
            MR42 = np.dot(Kx_CRO_1, BRO)
            MR43 = np.dot(Kx_CRE_1, ARE) + KE1z
            MR44 = np.dot(Kx_CRE_1, BRE)
            MR = np.asarray(
                np.bmat(
                    [
                        [I, ZERO, I, ZERO],
                        [ZERO, I, ZERO, I],
                        [MR31, MR32, MR33, MR34],
                        [MR41, MR42, MR43, MR44],
                    ]
                )
            )

            Ky_CTO_1 = ky / k * CTO_1
            Ky_CTE_1 = ky / k * CTE_1
            Kx_CTO_1 = kx[:, np.newaxis] / k * CTO_1
            Kx_CTE_1 = kx[:, np.newaxis] / k * CTE_1
            MT31 = -np.dot(Ky_CTO_1, ATO)
            MT32 = -np.dot(Ky_CTO_1, BTO) - KO3z
            MT33 = -np.dot(Ky_CTE_1, ATE)
            MT34 = -np.dot(Ky_CTE_1, BTE) - KE3z
            MT41 = np.dot(Kx_CTO_1, ATO) + KO3z
            MT42 = np.dot(Kx_CTO_1, BTO)
            MT43 = np.dot(Kx_CTE_1, ATE) + KE3z
            MT44 = np.dot(Kx_CTE_1, BTE)
            MT = np.asarray(
                np.bmat(
                    [
                        [I, ZERO, I, ZERO],
                        [ZERO, I, ZERO, I],
                        [MT31, MT32, MT33, MT34],
                        [MT41, MT42, MT43, MT44],
                    ]
                )
            )

            Mp.fill(0.0)
            M.fill(0.0)

            for nlayer in range(nlayers - 2, 0, -1):  # internal layers
                layer = multilayer[nlayer]
                thickness = layer.thickness

                EPS2, EPS21 = layer.getEPSFourierCoeffs(wl, n, anisotropic=True)

                # Exx = np.squeeze(EPS2[0, 0, :])
                # Exx = toeplitz(np.flipud(Exx[0:hmax + 1]), Exx[hmax:])
                Exy = np.squeeze(EPS2[0, 1, :])
                Exy = toeplitz(np.flipud(Exy[0 : hmax + 1]), Exy[hmax:])
                Exz = np.squeeze(EPS2[0, 2, :])
                Exz = toeplitz(np.flipud(Exz[0 : hmax + 1]), Exz[hmax:])

                Eyx = np.squeeze(EPS2[1, 0, :])
                Eyx = toeplitz(np.flipud(Eyx[0 : hmax + 1]), Eyx[hmax:])
                Eyy = np.squeeze(EPS2[1, 1, :])
                Eyy = toeplitz(np.flipud(Eyy[0 : hmax + 1]), Eyy[hmax:])
                Eyz = np.squeeze(EPS2[1, 2, :])
                Eyz = toeplitz(np.flipud(Eyz[0 : hmax + 1]), Eyz[hmax:])

                Ezx = np.squeeze(EPS2[2, 0, :])
                Ezx = toeplitz(np.flipud(Ezx[0 : hmax + 1]), Ezx[hmax:])
                Ezy = np.squeeze(EPS2[2, 1, :])
                Ezy = toeplitz(np.flipud(Ezy[0 : hmax + 1]), Ezy[hmax:])
                Ezz = np.squeeze(EPS2[2, 2, :])
                Ezz = toeplitz(np.flipud(Ezz[0 : hmax + 1]), Ezz[hmax:])

                Exx_1 = np.squeeze(EPS21[0, 0, :])
                Exx_1 = toeplitz(np.flipud(Exx_1[0 : hmax + 1]), Exx_1[hmax:])
                Exx_1_1 = inv(Exx_1)

                # lalanne
                Ezz_1 = inv(Ezz)
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
                H = 1j * np.diag(np.repeat(np.diag(Kz), 4)) + np.asarray(
                    np.bmat(
                        [
                            [H11, H12, H13, H14],
                            [H21, H22, H23, H24],
                            [H31, H32, H33, H34],
                            [H41, H42, H43, H44],
                        ]
                    )
                )

                q, W = eig(H)
                W1, W2, W3, W4 = np.split(W, 4)

                #
                # boundary conditions
                #
                # x = [R T]
                # R = [ROx ROy REx REy]
                # T = [TOx TOy TEx TEy]
                # b + MR.R = M1p.c
                # M1.c = M2p.c
                # ...
                # ML.c = MT.T
                # therefore: b + MR.R = (M1p.M1^-1.M2p.M2^-1. ...).MT.T
                # missing equations from (46)..(49) in glytsis_rigorous
                # [b] = [-MR Mtot.MT] [R]
                # [0]   [...........] [T]

                z = np.zeros_like(q)
                z[np.where(q.real > 0)] = -thickness
                D = np.exp(k * q * z)
                Sy0 = W1 * D[np.newaxis, :]
                Sx0 = W2 * D[np.newaxis, :]
                Uy0 = W3 * D[np.newaxis, :]
                Ux0 = W4 * D[np.newaxis, :]

                z = thickness * np.ones_like(q)
                z[np.where(q.real > 0)] = 0
                D = np.exp(k * q * z)
                D1 = np.exp(-1j * k2i[2, :] * thickness)
                Syd = D1[:, np.newaxis] * W1 * D[np.newaxis, :]
                Sxd = D1[:, np.newaxis] * W2 * D[np.newaxis, :]
                Uyd = D1[:, np.newaxis] * W3 * D[np.newaxis, :]
                Uxd = D1[:, np.newaxis] * W4 * D[np.newaxis, :]

                Mp[:, :, nlayer] = np.r_[Sx0, Sy0, -1j * Ux0, -1j * Uy0]
                M[:, :, nlayer] = np.r_[Sxd, Syd, -1j * Uxd, -1j * Uyd]

            Mtot = np.eye(4 * nood, dtype=complex)
            for nlayer in range(1, nlayers - 1):
                Mtot = np.dot(np.dot(Mtot, Mp[:, :, nlayer]), inv(M[:, :, nlayer]))

            BC_b = np.r_[b, np.zeros_like(b)]
            BC_A1 = np.c_[-MR, np.dot(Mtot, MT)]
            BC_A2 = np.asarray(
                np.bmat(
                    [
                        [
                            (c1[0] * I - c1[2] * np.dot(CRO_1, ARO)),
                            (c1[1] * I - c1[2] * np.dot(CRO_1, BRO)),
                            ZERO,
                            ZERO,
                            ZERO,
                            ZERO,
                            ZERO,
                            ZERO,
                        ],
                        [
                            ZERO,
                            ZERO,
                            (DRE - np.dot(np.dot(FRE, CRE_1), ARE)),
                            (ERE - np.dot(np.dot(FRE, CRE_1), BRE)),
                            ZERO,
                            ZERO,
                            ZERO,
                            ZERO,
                        ],
                        [
                            ZERO,
                            ZERO,
                            ZERO,
                            ZERO,
                            (c3[0] * I - c3[2] * np.dot(CTO_1, ATO)),
                            (c3[1] * I - c3[2] * np.dot(CTO_1, BTO)),
                            ZERO,
                            ZERO,
                        ],
                        [
                            ZERO,
                            ZERO,
                            ZERO,
                            ZERO,
                            ZERO,
                            ZERO,
                            (DTE - np.dot(np.dot(FTE, CTE_1), ATE)),
                            (ETE - np.dot(np.dot(FTE, CTE_1), BTE)),
                        ],
                    ]
                )
            )

            BC_A = np.r_[BC_A1, BC_A2]

            x = linsolve(BC_A, BC_b)

            ROx, ROy, REx, REy, TOx, TOy, TEx, TEy = np.split(x, 8)

            ROz = -np.dot(CRO_1, (np.dot(ARO, ROx) + np.dot(BRO, ROy)))
            REz = -np.dot(CRE_1, (np.dot(ARE, REx) + np.dot(BRE, REy)))
            TOz = -np.dot(CTO_1, (np.dot(ATO, TOx) + np.dot(BTO, TOy)))
            TEz = -np.dot(CTE_1, (np.dot(ATE, TEx) + np.dot(BTE, TEy)))

            denom = (k1[2] - np.dot(u, k1) * u[2]).real
            DEO1[:, iwl] = (
                -(
                    (np.absolute(ROx) ** 2 + np.absolute(ROy) ** 2 + np.absolute(ROz) ** 2)
                    * np.conj(kO1i[2, :])
                    - (ROx * kO1i[0, :] + ROy * kO1i[1, :] + ROz * kO1i[2, :])
                    * np.conj(ROz)
                ).real
                / denom
            )
            DEE1[:, iwl] = (
                -(
                    (np.absolute(REx) ** 2 + np.absolute(REy) ** 2 + np.absolute(REz) ** 2)
                    * np.conj(kE1i[2, :])
                    - (REx * kE1i[0, :] + REy * kE1i[1, :] + REz * kE1i[2, :])
                    * np.conj(REz)
                ).real
                / denom
            )
            DEO3[:, iwl] = (
                (np.absolute(TOx) ** 2 + np.absolute(TOy) ** 2 + np.absolute(TOz) ** 2)
                * np.conj(kO3i[2, :])
                - (TOx * kO3i[0, :] + TOy * kO3i[1, :] + TOz * kO3i[2, :]) * np.conj(TOz)
            ).real / denom
            DEE3[:, iwl] = (
                (np.absolute(TEx) ** 2 + np.absolute(TEy) ** 2 + np.absolute(TEz) ** 2)
                * np.conj(kE3i[2, :])
                - (TEx * kE3i[0, :] + TEy * kE3i[1, :] + TEz * kE3i[2, :]) * np.conj(TEz)
            ).real / denom

        # save the results
        self.DEO1 = DEO1
        self.DEE1 = DEE1
        self.DEO3 = DEO3
        self.DEE3 = DEE3

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
