"""Transfer matrix for isotropic and anisotropic multilayer.

The transfer matrix algorithms is used to compute the power reflected
and transmitted by a multilayer.
The multilayer can be made of isotropic and anisotropic layers, with
any thickness.
Two versions of the algorithm are present: an isotropic one and an
anisotropic one.
"""

__author__ = "Lorenzo Bolla"

import numpy as np
from scipy.linalg import inv
from EMpy.utils import snell, norm
from EMpy.constants import c, mu0

# import Gnuplot


class TransferMatrix:
    """Class to handle the transfer matrix solvers."""

    def __init__(self, multilayer):
        """Set the multilayer.

        INPUT
        multilayer = Multilayer obj describing the sequence of layers.
        """
        self.setMultilayer(multilayer)

    def setMultilayer(self, m):
        self.multilayer = m.simplify()


class IsotropicTransferMatrix(TransferMatrix):
    def __init__(self, multilayer, theta_inc):
        """Set the multilayer and the incident angle.

        INPUT
        multilayer = Multilayer obj describing the sequence of layers.
        theta_inc = angle of the incident wave (in radiant) wrt the
        normal.
        """
        if not multilayer.isIsotropic():
            raise ValueError(
                "Cannot use IsotropicTransferMatrix with anisotropic multilayer"
            )
        TransferMatrix.__init__(self, multilayer)
        self.theta_inc = theta_inc

    def solve(self, wls):
        """Isotropic solver.

        INPUT
        wls = wavelengths to scan (any asarray-able object).

        OUTPUT
        self.Rs, self.Ts, self.Rp, self.Tp = power reflected and
        transmitted on s and p polarizations.
        """

        self.wls = np.asarray(wls)

        multilayer = self.multilayer
        theta_inc = self.theta_inc

        nlayers = len(multilayer)
        d = np.array([l.thickness for l in multilayer]).ravel()

        Rs = np.zeros_like(self.wls, dtype=float)
        Ts = np.zeros_like(self.wls, dtype=float)
        Rp = np.zeros_like(self.wls, dtype=float)
        Tp = np.zeros_like(self.wls, dtype=float)

        Dp = np.zeros((2, 2), dtype=complex)
        Ds = np.zeros((2, 2), dtype=complex)
        P = np.zeros((2, 2), dtype=complex)
        Ms = np.zeros((2, 2), dtype=complex)
        Mp = np.zeros((2, 2), dtype=complex)
        k = np.zeros((nlayers, 2), dtype=complex)

        ntot = np.zeros((self.wls.size, nlayers), dtype=complex)
        for i, l in enumerate(multilayer):
            #            ntot[:,i] = l.mat.n(self.wls,l.mat.T0)
            ntot[:, i] = l.mat.n(self.wls, l.mat.toc.T0)

        for iwl, wl in enumerate(self.wls):
            n = ntot[iwl, :]
            theta = snell(theta_inc, n)

            k[:, 0] = 2 * np.pi * n / wl * np.cos(theta)
            k[:, 1] = 2 * np.pi * n / wl * np.sin(theta)

            Ds = [[1.0, 1.0], [n[0] * np.cos(theta[0]), -n[0] * np.cos(theta[0])]]
            Dp = [[np.cos(theta[0]), np.cos(theta[0])], [n[0], -n[0]]]
            Ms = inv(Ds)
            Mp = inv(Dp)

            for nn, dd, tt, kk in zip(n[1:-1], d[1:-1], theta[1:-1], k[1:-1, 0]):
                Ds = [[1.0, 1.0], [nn * np.cos(tt), -nn * np.cos(tt)]]
                Dp = [[np.cos(tt), np.cos(tt)], [nn, -nn]]
                phi = kk * dd
                P = [[np.exp(1j * phi), 0], [0, np.exp(-1j * phi)]]
                Ms = np.dot(Ms, np.dot(Ds, np.dot(P, inv(Ds))))
                Mp = np.dot(Mp, np.dot(Dp, np.dot(P, inv(Dp))))

            Ds = [[1.0, 1.0], [n[-1] * np.cos(theta[-1]), -n[-1] * np.cos(theta[-1])]]
            Dp = [[np.cos(theta[-1]), np.cos(theta[-1])], [n[-1], -n[-1]]]
            Ms = np.dot(Ms, Ds)
            Mp = np.dot(Mp, Dp)

            rs = Ms[1, 0] / Ms[0, 0]
            ts = 1.0 / Ms[0, 0]

            rp = Mp[1, 0] / Mp[0, 0]
            tp = 1.0 / Mp[0, 0]

            Rs[iwl] = np.absolute(rs) ** 2
            Ts[iwl] = (
                np.absolute((n[-1] * np.cos(theta[-1])) / (n[0] * np.cos(theta[0])))
                * np.absolute(ts) ** 2
            )
            Rp[iwl] = np.absolute(rp) ** 2
            Tp[iwl] = (
                np.absolute((n[-1] * np.cos(theta[-1])) / (n[0] * np.cos(theta[0])))
                * np.absolute(tp) ** 2
            )

        self.Rs = Rs
        self.Ts = Ts
        self.Rp = Rp
        self.Tp = Tp
        return self

    #     def plot(self):
    #         """Plot the solution."""
    #         g = Gnuplot.Gnuplot()
    #         g('set xlabel "$\lambda$"')
    #         g('set ylabel "power"')
    #         g('set yrange [0:1]')
    #         g('set data style linespoints')
    #         g.plot(Gnuplot.Data(self.wls, self.Rs, with_ = 'linespoints', title = 'Rs'), \
    #                Gnuplot.Data(self.wls, self.Ts, with_ = 'linespoints', title = 'Ts'), \
    #                Gnuplot.Data(self.wls, self.Rp, with_ = 'linespoints', title = 'Rp'), \
    #                Gnuplot.Data(self.wls, self.Tp, with_ = 'linespoints', title = 'Tp'))
    #         raw_input('press enter to close the graph...')

    def __str__(self):
        return "ISOTROPIC TRANSFER MATRIX SOLVER\n\n%s\n\ntheta inc = %g" % (
            self.multilayer.__str__(),
            self.theta_inc,
        )


class AnisotropicTransferMatrix(TransferMatrix):
    def __init__(self, multilayer, theta_inc_x, theta_inc_y):
        """Set the multilayer and the incident angle.

        INPUT
        multilayer = Multilayer obj describing the sequence of layers.
        theta_inc_x and theta_inc_y = angles of the incident wave (in
        radiant) wrt the normal.
        """
        TransferMatrix.__init__(self, multilayer)
        self.theta_inc_x = theta_inc_x
        self.theta_inc_y = theta_inc_y

    def solve(self, wls):
        """Anisotropic solver.

        INPUT
        wls = wavelengths to scan (any asarray-able object).

        OUTPUT
        self.R, self.T = power reflected and transmitted.
        """

        self.wls = np.asarray(wls)

        multilayer = self.multilayer
        theta_inc_x = self.theta_inc_x
        theta_inc_y = self.theta_inc_y

        def find_roots(wl, epsilon, alpha, beta):
            """Find roots of characteristic equation.

            Given a wavelength, a 3x3 tensor epsilon and the tangential components
            of the wavevector k = (alpha,beta,gamma_i), returns the 4 possible
            gamma_i, i = 1,2,3,4 that satisfy the boundary conditions.
            """

            omega = 2.0 * np.pi * c / wl
            K = omega**2 * mu0 * epsilon

            k0 = 2.0 * np.pi / wl
            K /= k0**2
            alpha /= k0
            beta /= k0

            alpha2 = alpha**2
            alpha3 = alpha**3
            alpha4 = alpha**4
            beta2 = beta**2
            beta3 = beta**3
            beta4 = beta**4

            coeff = [
                K[2, 2],
                alpha * (K[0, 2] + K[2, 0]) + beta * (K[1, 2] + K[2, 1]),
                alpha2 * (K[0, 0] + K[2, 2])
                + alpha * beta * (K[1, 0] + K[0, 1])
                + beta2 * (K[1, 1] + K[2, 2])
                + (
                    K[0, 2] * K[2, 0]
                    + K[1, 2] * K[2, 1]
                    - K[0, 0] * K[2, 2]
                    - K[1, 1] * K[2, 2]
                ),
                alpha3 * (K[0, 2] + K[2, 0])
                + beta3 * (K[1, 2] + K[2, 1])
                + alpha2 * beta * (K[1, 2] + K[2, 1])
                + alpha * beta2 * (K[0, 2] + K[2, 0])
                + alpha
                * (
                    K[0, 1] * K[1, 2]
                    + K[1, 0] * K[2, 1]
                    - K[0, 2] * K[1, 1]
                    - K[2, 0] * K[1, 1]
                )
                + beta
                * (
                    K[0, 1] * K[2, 0]
                    + K[1, 0] * K[0, 2]
                    - K[0, 0] * K[1, 2]
                    - K[0, 0] * K[2, 1]
                ),
                alpha4 * (K[0, 0])
                + beta4 * (K[1, 1])
                + alpha3 * beta * (K[0, 1] + K[1, 0])
                + alpha * beta3 * (K[0, 1] + K[1, 0])
                + alpha2 * beta2 * (K[0, 0] + K[1, 1])
                + alpha2
                * (
                    K[0, 1] * K[1, 0]
                    + K[0, 2] * K[2, 0]
                    - K[0, 0] * K[2, 2]
                    - K[0, 0] * K[1, 1]
                )
                + beta2
                * (
                    K[0, 1] * K[1, 0]
                    + K[1, 2] * K[2, 1]
                    - K[0, 0] * K[1, 1]
                    - K[1, 1] * K[2, 2]
                )
                + alpha
                * beta
                * (
                    K[0, 2] * K[2, 1]
                    + K[2, 0] * K[1, 2]
                    - K[0, 1] * K[2, 2]
                    - K[1, 0] * K[2, 2]
                )
                + K[0, 0] * K[1, 1] * K[2, 2]
                - K[0, 0] * K[1, 2] * K[2, 1]
                - K[1, 0] * K[0, 1] * K[2, 2]
                + K[1, 0] * K[0, 2] * K[2, 1]
                + K[2, 0] * K[0, 1] * K[1, 2]
                - K[2, 0] * K[0, 2] * K[1, 1],
            ]

            gamma = np.roots(coeff)
            tmp = np.sort_complex(gamma)
            gamma = tmp[[3, 0, 2, 1]]  # convention

            k = (
                k0
                * np.array(
                    [alpha * np.ones(gamma.shape), beta * np.ones(gamma.shape), gamma]
                ).T
            )
            v = np.zeros((4, 3), dtype=complex)

            for i, g in enumerate(gamma):
                # H = K + [
                #     [-beta2 - g ** 2, alpha * beta, alpha * g],
                #     [alpha * beta, -alpha2 - g ** 2, beta * g],
                #     [alpha * g, beta * g, -alpha2 - beta2],
                # ]
                v[i, :] = [
                    (K[1, 1] - alpha2 - g**2) * (K[2, 2] - alpha2 - beta2)
                    - (K[1, 2] + beta * g) ** 2,
                    (K[1, 2] + beta * g) * (K[2, 0] + alpha * g)
                    - (K[0, 1] + alpha * beta) * (K[2, 2] - alpha2 - beta2),
                    (K[0, 1] + alpha * beta) * (K[1, 2] + beta * g)
                    - (K[0, 2] + alpha * g) * (K[1, 1] - alpha2 - g**2),
                ]

            p3 = v[0, :]
            p3 /= norm(p3)
            p4 = v[1, :]
            p4 /= norm(p4)
            p1 = np.cross(p3, k[0, :])
            p1 /= norm(p1)
            p2 = np.cross(p4, k[1, :])
            p2 /= norm(p2)

            p = np.array([p1, p2, p3, p4])
            q = wl / (2.0 * np.pi * mu0 * c) * np.cross(k, p)

            return k, p, q

        nlayers = len(multilayer)
        d = np.asarray([l.thickness for l in multilayer])

        # R and T are real, because they are powers
        # r and t are complex!
        R = np.zeros((2, 2, self.wls.size))
        T = np.zeros((2, 2, self.wls.size))

        epstot = np.zeros((3, 3, self.wls.size, nlayers), dtype=complex)
        for i, l in enumerate(multilayer):
            epstot[:, :, :, i] = l.mat.epsilonTensor(self.wls)

        for iwl, wl in enumerate(self.wls):
            epsilon = epstot[:, :, iwl, :]

            kx = 2 * np.pi / wl * np.sin(theta_inc_x)
            ky = 2 * np.pi / wl * np.sin(theta_inc_y)
            x = np.array([1, 0, 0], dtype=float)
            y = np.array([0, 1, 0], dtype=float)
            # z = np.array([0, 0, 1], dtype=float)
            k = np.zeros((4, 3, nlayers), dtype=complex)
            p = np.zeros((4, 3, nlayers), dtype=complex)
            q = np.zeros((4, 3, nlayers), dtype=complex)
            D = np.zeros((4, 4, nlayers), dtype=complex)
            P = np.zeros((4, 4, nlayers), dtype=complex)

            for i in range(nlayers):
                k[:, :, i], p[:, :, i], q[:, :, i] = find_roots(
                    wl, epsilon[:, :, i], kx, ky
                )
                D[:, :, i] = [
                    [
                        np.dot(x, p[0, :, i]),
                        np.dot(x, p[1, :, i]),
                        np.dot(x, p[2, :, i]),
                        np.dot(x, p[3, :, i]),
                    ],
                    [
                        np.dot(y, q[0, :, i]),
                        np.dot(y, q[1, :, i]),
                        np.dot(y, q[2, :, i]),
                        np.dot(y, q[3, :, i]),
                    ],
                    [
                        np.dot(y, p[0, :, i]),
                        np.dot(y, p[1, :, i]),
                        np.dot(y, p[2, :, i]),
                        np.dot(y, p[3, :, i]),
                    ],
                    [
                        np.dot(x, q[0, :, i]),
                        np.dot(x, q[1, :, i]),
                        np.dot(x, q[2, :, i]),
                        np.dot(x, q[3, :, i]),
                    ],
                ]

            for i in range(1, nlayers - 1):
                P[:, :, i] = np.diag(np.exp(1j * k[:, 2, i] * d[i]))

            M = inv(D[:, :, 0])
            for i in range(1, nlayers - 1):
                M = np.dot(M, np.dot(D[:, :, i], np.dot(P[:, :, i], inv(D[:, :, i]))))
            M = np.dot(M, D[:, :, -1])

            deltaM = M[0, 0] * M[2, 2] - M[0, 2] * M[2, 0]

            # reflectance matrix (from yeh_electromagnetic)
            # r = [rss rsp; rps rpp]
            r = (
                np.array(
                    [
                        [
                            M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0],
                            M[3, 0] * M[2, 2] - M[3, 2] * M[2, 0],
                        ],
                        [
                            M[0, 0] * M[1, 2] - M[1, 0] * M[0, 2],
                            M[0, 0] * M[3, 2] - M[3, 0] * M[0, 2],
                        ],
                    ],
                    dtype=complex,
                )
                / deltaM
            )

            # transmittance matrix (from yeh_electromagnetic)
            # t = [tss tsp; tps tpp]
            t = np.array([[M[2, 2], -M[2, 0]], [-M[0, 2], M[0, 0]]]) / deltaM

            # P_t/P_inc = |E_t|**2/|E_inc|**2 . k_t_z/k_inc_z
            T[:, :, iwl] = (np.absolute(t) ** 2 * k[0, 2, -1] / k[0, 2, 0]).real

            # P_r/P_inc = |E_r|**2/|E_inc|**2
            R[:, :, iwl] = np.absolute(r) ** 2

        self.R = R
        self.T = T
        return self

    def __str__(self):
        return (
            "ANISOTROPIC TRANSFER MATRIX SOLVER\n\n%s\n\ntheta inc x = %g\ntheta inc y = %g"
            % (self.multilayer.__str__(), self.theta_inc_x, self.theta_inc_y)
        )
