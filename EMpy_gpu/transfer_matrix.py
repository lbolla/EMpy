"""Transfer matrix for isotropic and anisotropic multilayer.

The transfer matrix algorithms is used to compute the power reflected
and transmitted by a multilayer.
The multilayer can be made of isotropic and anisotropic layers, with
any thickness.
Two versions of the algorithm are present: an isotropic one and an
anisotropic one.
"""
from builtins import zip
from builtins import range
from builtins import object

__author__ = 'Lorenzo Bolla'

import scipy as S
from scipy.linalg import inv
from EMpy_gpu.utils import snell, norm
from EMpy_gpu.constants import c, mu0
# import Gnuplot


class TransferMatrix(object):

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
                'Cannot use IsotropicTransferMatrix with anisotropic multilayer')
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

        self.wls = S.asarray(wls)

        multilayer = self.multilayer
        theta_inc = self.theta_inc

        nlayers = len(multilayer)
        d = S.array([l.thickness for l in multilayer]).ravel()

        Rs = S.zeros_like(self.wls)
        Ts = S.zeros_like(self.wls)
        Rp = S.zeros_like(self.wls)
        Tp = S.zeros_like(self.wls)

        Dp = S.zeros((2, 2), dtype=complex)
        Ds = S.zeros((2, 2), dtype=complex)
        P = S.zeros((2, 2), dtype=complex)
        Ms = S.zeros((2, 2), dtype=complex)
        Mp = S.zeros((2, 2), dtype=complex)
        k = S.zeros((nlayers, 2), dtype=complex)

        ntot = S.zeros((self.wls.size, nlayers), dtype=complex)
        for i, l in enumerate(multilayer):
            #            ntot[:,i] = l.mat.n(self.wls,l.mat.T0)
            ntot[:, i] = l.mat.n(self.wls, l.mat.toc.T0)

        for iwl, wl in enumerate(self.wls):

            n = ntot[iwl, :]
            theta = snell(theta_inc, n)

            k[:, 0] = 2 * S.pi * n / wl * S.cos(theta)
            k[:, 1] = 2 * S.pi * n / wl * S.sin(theta)

            Ds = [[1., 1.], [n[0] * S.cos(theta[0]), -n[0] * S.cos(theta[0])]]
            Dp = [[S.cos(theta[0]), S.cos(theta[0])], [n[0], -n[0]]]
            Ms = inv(Ds)
            Mp = inv(Dp)

            for nn, dd, tt, kk in zip(
                    n[1:-1], d[1:-1], theta[1:-1], k[1:-1, 0]):

                Ds = [[1., 1.], [nn * S.cos(tt), -nn * S.cos(tt)]]
                Dp = [[S.cos(tt), S.cos(tt)], [nn, -nn]]
                phi = kk * dd
                P = [[S.exp(1j * phi), 0], [0, S.exp(-1j * phi)]]
                Ms = S.dot(Ms, S.dot(Ds, S.dot(P, inv(Ds))))
                Mp = S.dot(Mp, S.dot(Dp, S.dot(P, inv(Dp))))

            Ds = [
                [1., 1.], [n[-1] * S.cos(theta[-1]), -n[-1] * S.cos(theta[-1])]]
            Dp = [[S.cos(theta[-1]), S.cos(theta[-1])], [n[-1], -n[-1]]]
            Ms = S.dot(Ms, Ds)
            Mp = S.dot(Mp, Dp)

            rs = Ms[1, 0] / Ms[0, 0]
            ts = 1. / Ms[0, 0]

            rp = Mp[1, 0] / Mp[0, 0]
            tp = 1. / Mp[0, 0]

            Rs[iwl] = S.absolute(rs) ** 2
            Ts[iwl] = S.absolute(
                (n[-1] * S.cos(theta[-1])) / (n[0] * S.cos(theta[0]))) * S.absolute(ts) ** 2
            Rp[iwl] = S.absolute(rp) ** 2
            Tp[iwl] = S.absolute(
                (n[-1] * S.cos(theta[-1])) / (n[0] * S.cos(theta[0]))) * S.absolute(tp) ** 2

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
        return 'ISOTROPIC TRANSFER MATRIX SOLVER\n\n%s\n\ntheta inc = %g' % \
               (self.multilayer.__str__(), self.theta_inc)


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

        self.wls = S.asarray(wls)

        multilayer = self.multilayer
        theta_inc_x = self.theta_inc_x
        theta_inc_y = self.theta_inc_y

        def find_roots(wl, epsilon, alpha, beta):
            """Find roots of characteristic equation.

            Given a wavelength, a 3x3 tensor epsilon and the tangential components
            of the wavevector k = (alpha,beta,gamma_i), returns the 4 possible
            gamma_i, i = 1,2,3,4 that satisfy the boundary conditions.
            """

            omega = 2. * S.pi * c / wl
            K = omega ** 2 * mu0 * epsilon

            k0 = 2. * S.pi / wl
            K /= k0 ** 2
            alpha /= k0
            beta /= k0

            alpha2 = alpha ** 2
            alpha3 = alpha ** 3
            alpha4 = alpha ** 4
            beta2 = beta ** 2
            beta3 = beta ** 3
            beta4 = beta ** 4

            coeff = [K[2, 2],

                     alpha * (K[0, 2] + K[2, 0]) +
                     beta * (K[1, 2] + K[2, 1]),

                     alpha2 * (K[0, 0] + K[2, 2]) +
                     alpha * beta * (K[1, 0] + K[0, 1]) +
                     beta2 * (K[1, 1] + K[2, 2]) +
                     (K[0, 2] *
                      K[2, 0] +
                      K[1, 2] *
                      K[2, 1] -
                      K[0, 0] *
                      K[2, 2] -
                      K[1, 1] *
                      K[2, 2]),

                     alpha3 * (K[0, 2] + K[2, 0]) +
                     beta3 * (K[1, 2] + K[2, 1]) +
                     alpha2 * beta * (K[1, 2] + K[2, 1]) +
                     alpha * beta2 * (K[0, 2] + K[2, 0]) +
                     alpha * (K[0, 1] * K[1, 2] + K[1, 0] * K[2, 1] - K[0, 2] * K[1, 1] - K[2, 0] * K[1, 1]) +
                     beta * (K[0,
                               1] * K[2,
                                      0] + K[1,
                                             0] * K[0,
                                                    2] - K[0,
                                                           0] * K[1,
                                                                  2] - K[0,
                                                                         0] * K[2,
                                                                                1]),

                     alpha4 * (K[0, 0]) +
                     beta4 * (K[1, 1]) +
                     alpha3 * beta * (K[0, 1] + K[1, 0]) +
                     alpha * beta3 * (K[0, 1] + K[1, 0]) +
                     alpha2 * beta2 * (K[0, 0] + K[1, 1]) +
                     alpha2 * (K[0, 1] * K[1, 0] + K[0, 2] * K[2, 0] - K[0, 0] * K[2, 2] - K[0, 0] * K[1, 1]) +
                     beta2 * (K[0, 1] * K[1, 0] + K[1, 2] * K[2, 1] - K[0, 0] * K[1, 1] - K[1, 1] * K[2, 2]) +
                     alpha * beta * (K[0, 2] * K[2, 1] + K[2, 0] * K[1, 2] - K[0, 1] * K[2, 2] - K[1, 0] * K[2, 2]) +
                     K[0, 0] * K[1, 1] * K[2, 2] -
                     K[0, 0] * K[1, 2] * K[2, 1] -
                     K[1, 0] * K[0, 1] * K[2, 2] +
                     K[1, 0] * K[0, 2] * K[2, 1] +
                     K[2, 0] * K[0, 1] * K[1, 2] -
                     K[2, 0] * K[0, 2] * K[1, 1]]

            gamma = S.roots(coeff)
            tmp = S.sort_complex(gamma)
            gamma = tmp[[3, 0, 2, 1]]  # convention

            k = k0 * \
                S.array([alpha *
                         S.ones(gamma.shape), beta *
                         S.ones(gamma.shape), gamma]).T
            v = S.zeros((4, 3), dtype=complex)

            for i, g in enumerate(gamma):

                H = K + [[-beta2 - g ** 2, alpha * beta, alpha * g],
                         [alpha * beta, -alpha2 - g ** 2, beta * g],
                         [alpha * g, beta * g, -alpha2 - beta2]]
                v[i, :] = [(K[1, 1] - alpha2 - g ** 2) * (K[2, 2] - alpha2 - beta2) - (K[1, 2] + beta * g) ** 2,
                           (K[1,
                              2] + beta * g) * (K[2,
                                                  0] + alpha * g) - (K[0,
                                                                       1] + alpha * beta) * (K[2,
                                                                                               2] - alpha2 - beta2),
                           (K[0, 1] + alpha * beta) * (K[1, 2] + beta * g) - (K[0, 2] + alpha * g) * (K[1, 1] - alpha2 - g ** 2)]

            p3 = v[0, :]
            p3 /= norm(p3)
            p4 = v[1, :]
            p4 /= norm(p4)
            p1 = S.cross(p3, k[0, :])
            p1 /= norm(p1)
            p2 = S.cross(p4, k[1, :])
            p2 /= norm(p2)

            p = S.array([p1, p2, p3, p4])
            q = wl / (2. * S.pi * mu0 * c) * S.cross(k, p)

            return k, p, q

        nlayers = len(multilayer)
        d = S.asarray([l.thickness for l in multilayer])

        # R and T are real, because they are powers
        # r and t are complex!
        R = S.zeros((2, 2, self.wls.size))
        T = S.zeros((2, 2, self.wls.size))

        epstot = S.zeros((3, 3, self.wls.size, nlayers), dtype=complex)
        for i, l in enumerate(multilayer):
            epstot[:, :, :, i] = l.mat.epsilonTensor(self.wls)

        for iwl, wl in enumerate(self.wls):

            epsilon = epstot[:, :, iwl, :]

            kx = 2 * S.pi / wl * S.sin(theta_inc_x)
            ky = 2 * S.pi / wl * S.sin(theta_inc_y)
            x = S.array([1, 0, 0], dtype=float)
            y = S.array([0, 1, 0], dtype=float)
            z = S.array([0, 0, 1], dtype=float)
            k = S.zeros((4, 3, nlayers), dtype=complex)
            p = S.zeros((4, 3, nlayers), dtype=complex)
            q = S.zeros((4, 3, nlayers), dtype=complex)
            D = S.zeros((4, 4, nlayers), dtype=complex)
            P = S.zeros((4, 4, nlayers), dtype=complex)

            for i in range(nlayers):

                k[:, :, i], p[:, :, i], q[:, :, i] = find_roots(
                    wl, epsilon[:, :, i], kx, ky)
                D[:, :, i] = [[S.dot(x, p[0, :, i]), S.dot(x, p[1, :, i]), S.dot(x, p[2, :, i]), S.dot(x, p[3, :, i])],
                              [
                    S.dot(
                        y, q[
                            0, :, i]), S.dot(
                        y, q[
                            1, :, i]), S.dot(
                        y, q[
                            2, :, i]), S.dot(
                                y, q[
                                    3, :, i])],
                    [
                    S.dot(
                        y, p[
                            0, :, i]), S.dot(
                        y, p[
                            1, :, i]), S.dot(
                        y, p[
                            2, :, i]), S.dot(
                                y, p[
                                    3, :, i])],
                    [S.dot(x, q[0, :, i]), S.dot(x, q[1, :, i]), S.dot(x, q[2, :, i]), S.dot(x, q[3, :, i])]]

            for i in range(1, nlayers - 1):
                P[:, :, i] = S.diag(S.exp(1j * k[:, 2, i] * d[i]))

            M = inv(D[:, :, 0])
            for i in range(1, nlayers - 1):
                M = S.dot(
                    M, S.dot(D[:, :, i], S.dot(P[:, :, i], inv(D[:, :, i]))))
            M = S.dot(M, D[:, :, -1])

            deltaM = M[0, 0] * M[2, 2] - M[0, 2] * M[2, 0]

            # reflectance matrix (from yeh_electromagnetic)
            # r = [rss rsp; rps rpp]
            r = S.array([[M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0], M[3, 0] * M[2, 2] - M[3, 2] * M[2, 0]],
                         [M[0, 0] * M[1, 2] - M[1, 0] * M[0, 2], M[0, 0] * M[3, 2] - M[3, 0] * M[0, 2]]], dtype=complex) / deltaM

            # transmittance matrix (from yeh_electromagnetic)
            # t = [tss tsp; tps tpp]
            t = S.array([[M[2, 2], -M[2, 0]], [-M[0, 2], M[0, 0]]]) / deltaM

            # P_t/P_inc = |E_t|**2/|E_inc|**2 . k_t_z/k_inc_z
            T[:, :, iwl] = (S.absolute(t) ** 2 * k[0, 2, -1] / k[0, 2, 0]).real

            # P_r/P_inc = |E_r|**2/|E_inc|**2
            R[:, :, iwl] = S.absolute(r) ** 2

        self.R = R
        self.T = T
        return self

    def __str__(self):
        return 'ANISOTROPIC TRANSFER MATRIX SOLVER\n\n%s\n\ntheta inc x = %g\ntheta inc y = %g' % \
               (self.multilayer.__str__(), self.theta_inc_x, self.theta_inc_y)
