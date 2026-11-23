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
from numpy.linalg import inv
from EMpy.utils import snell
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
        theta_inc = angle of the incident wave (in radians) wrt the
        normal, measured in the first layer (multilayer[0]).
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
        transmitted on s and p polarizations, normalized to incident
        normal Poynting flux at the entry (z = 0-) and exit (z = d+)
        interfaces (under TIR from a lossy incident medium into a
        lossless exit medium, lateral Goos-Hanchen attenuation yields
        local S_z(d+) < 0).
        """

        self.wls = np.atleast_1d(wls)

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

            # Compute kz / k0 = n * cos(theta) directly from the radical on the
            # passive forward (+z) branch (cut along +i*R_+ of qz**2: Re(qz) - Im(qz) >= 0)
            # to preserve continuity both in TIR and for lossy incident media (kappa_0 -> 0+).
            qz = np.sqrt(n**2 - (n[0] * np.sin(theta_inc)) ** 2 + 0j)
            flip = (qz.real - qz.imag < 0) | ((qz.real - qz.imag == 0) & (qz.real < 0))
            qz = np.where(flip, -qz, qz)
            qz[0] = n[0] * np.cos(theta_inc)
            cos_t = qz / n

            k0 = 2 * np.pi / wl
            k[:, 0] = k0 * qz
            k[:, 1] = k0 * n * np.sin(theta)

            Ds = [[1.0, 1.0], [qz[0], -qz[0]]]
            Dp = [[cos_t[0], cos_t[0]], [n[0], -n[0]]]
            Ms = inv(Ds)
            Mp = inv(Dp)
            log_scale = 0.0

            # Use closed-form Abeles layer matrices (avoiding inv(Ds)/inv(Dp),
            # which are singular at exact inner-layer cutoff qz == 0) scaled by
            # exp(-|Im(phi)|) so thick metallic/evanescent layers never overflow.
            for nn, dd, qzz, kk in zip(n[1:-1], d[1:-1], qz[1:-1], k[1:-1, 0]):
                phi = kk * dd
                re_phi = float(np.real(phi))
                im_phi = float(np.imag(phi))
                alpha = abs(im_phi)
                decay = np.exp(-2.0 * alpha)
                if im_phi <= 0.0:
                    e_plus = np.exp(1j * re_phi)
                    e_minus = np.exp(-1j * re_phi) * decay
                else:
                    e_plus = np.exp(1j * re_phi) * decay
                    e_minus = np.exp(-1j * re_phi)
                cos_s = 0.5 * (e_plus + e_minus)
                sin_s = -0.5j * (e_plus - e_minus)
                if abs(phi) < 1e-6:
                    # Smooth sinc limit at qz -> 0: (sin(phi) / qz) * exp(-alpha) = k0 * d * sinc(phi / pi) * exp(-alpha)
                    sin_over_qz = k0 * dd * np.sinc(phi / np.pi) * np.exp(-alpha)
                else:
                    sin_over_qz = sin_s / qzz
                Ls = np.array(
                    [[cos_s, 1j * sin_over_qz], [1j * qzz * sin_s, cos_s]],
                    dtype=complex,
                )
                Lp = np.array(
                    [
                        [cos_s, 1j * qzz * sin_s / (nn**2)],
                        [1j * (nn**2) * sin_over_qz, cos_s],
                    ],
                    dtype=complex,
                )
                Ms = np.dot(Ms, Ls)
                Mp = np.dot(Mp, Lp)
                log_scale += alpha

            Ds = [[1.0, 1.0], [qz[-1], -qz[-1]]]
            Dp = [[cos_t[-1], cos_t[-1]], [n[-1], -n[-1]]]
            Ms = np.dot(Ms, Ds)
            Mp = np.dot(Mp, Dp)

            rs = Ms[1, 0] / Ms[0, 0]
            ts = np.exp(-log_scale) / Ms[0, 0]

            rp = Mp[1, 0] / Mp[0, 0]
            tp = np.exp(-log_scale) / Mp[0, 0]

            Rs[iwl] = np.absolute(rs) ** 2
            Ts[iwl] = np.real(qz[-1]) / np.real(qz[0]) * np.absolute(ts) ** 2
            Rp[iwl] = np.absolute(rp) ** 2
            Tp[iwl] = (
                np.real(np.conj(n[-1]) * cos_t[-1])
                / np.real(np.conj(n[0]) * cos_t[0])
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
        radians) wrt the normal in vacuum, i.e. kx = k0 * sin(theta_inc_x)
        and ky = k0 * sin(theta_inc_y) (unlike IsotropicTransferMatrix,
        where theta_inc is measured in multilayer[0]; for prism-coupled
        incidence with n0 * sin(theta_0) > 1, pass complex
        np.arcsin(n0 * np.sin(theta_0) + 0j)).
        """
        TransferMatrix.__init__(self, multilayer)
        self.theta_inc_x = theta_inc_x
        self.theta_inc_y = theta_inc_y

    def solve(self, wls):
        """Anisotropic solver.

        INPUT
        wls = wavelengths to scan (any asarray-able object).

        OUTPUT
        self.R, self.T = mode-resolved power reflected and transmitted.
        self.R_tot, self.T_tot = total normal Poynting flux reflected at
        z = 0- and transmitted at z = d+, normalized to incident Sz(0-)
        (including cross-mode interference and, under TIR from a lossy
        incident medium into a lossless exit medium, negative local
        S_z(d+) from lateral Goos-Hanchen attenuation).
        """

        self.wls = np.atleast_1d(wls)

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

            if np.allclose(K, K[0, 0] * np.eye(3), rtol=1e-13, atol=1e-13):
                g = np.sqrt(K[0, 0] - alpha2 - beta2 + 0j)
                if g.real - g.imag < 0 or (g.real - g.imag == 0 and g.real < 0):
                    g = -g
                gamma = np.array([g, -g, g, -g], dtype=complex)
                raw_p = None
            else:
                # Solve the first-order 4x4 Berreman eigenproblem (rather than companion
                # roots of the expanded quartic) to avoid O(sqrt(eps)) accuracy loss near
                # double roots (e.g. weak birefringence or optic axes).
                kzz = K[2, 2]
                if abs(kzz) > 1e-14 * np.max(np.abs(K)):
                    ez_row = (
                        np.array([-K[2, 0], -alpha, -K[2, 1], beta], dtype=complex)
                        / kzz
                    )
                    hz_row = np.array([-beta, 0.0, alpha, 0.0], dtype=complex)
                    B_mat = np.zeros((4, 4), dtype=complex)
                    B_mat[0, :] = alpha * ez_row + np.array([0.0, 1.0, 0.0, 0.0])
                    B_mat[1, :] = (
                        beta * hz_row
                        + np.array([K[0, 0], 0.0, K[0, 1], 0.0])
                        + K[0, 2] * ez_row
                    )
                    B_mat[2, :] = beta * ez_row + np.array([0.0, 0.0, 0.0, -1.0])
                    B_mat[3, :] = (
                        alpha * hz_row
                        - np.array([K[1, 0], 0.0, K[1, 1], 0.0])
                        - K[1, 2] * ez_row
                    )
                    raw_gamma = np.linalg.eigvals(B_mat)
                else:
                    raw_gamma = np.roots(coeff)

                def _wave_op(g_val):
                    return K + np.array(
                        [
                            [-beta2 - g_val**2, alpha * beta, alpha * g_val],
                            [alpha * beta, -alpha2 - g_val**2, beta * g_val],
                            [alpha * g_val, beta * g_val, -alpha2 - beta2],
                        ],
                        dtype=complex,
                    )

                raw_k = (
                    k0
                    * np.array(
                        [
                            alpha * np.ones(raw_gamma.shape),
                            beta * np.ones(raw_gamma.shape),
                            raw_gamma,
                        ]
                    ).T
                )
                raw_p = np.zeros((4, 3), dtype=complex)
                for i, g in enumerate(raw_gamma):
                    _, _, vh = np.linalg.svd(_wave_op(g))
                    raw_p[i, :] = vh[-1, :].conj()
                    raw_p[i, :] /= np.linalg.norm(raw_p[i, :])
                raw_q = wl / (2.0 * np.pi * mu0 * c) * np.cross(raw_k, raw_p)
                raw_sz = 0.5 * np.real(
                    raw_p[:, 0] * np.conj(raw_q[:, 1])
                    - raw_p[:, 1] * np.conj(raw_q[:, 0])
                )

                # Classify roots into 2 forward (+z) and 2 backward (-z) modes
                # under exp(+i*w*t - i*k*z):
                # - For real tangential (kx, ky), |Im(gamma)| > im_tol decays in +z iff
                #   Im(gamma) < 0 (including tilted birefringent TIR where Re(gamma) != 0),
                #   and propagating modes carry flux in +z iff Sz > 0.
                # - For complex tangential (kx, ky) (e.g. lossy superstrate), inhomogeneous
                #   propagating modes in a lossless/lower-loss layer have |Re(gamma)| > |Im(gamma)|
                #   and Sz > 0 (even when Im(gamma) > 0), while TIR modes have |Im(gamma)| >= |Re(gamma)|
                #   and decay in +z iff Im(gamma) < 0.
                im_tol = 1e-6 * max(1.0, float(np.max(np.abs(raw_gamma))))
                sz_tol = 1e-9 / (mu0 * c)
                cplx_kt = (abs(np.imag(alpha)) + abs(np.imag(beta))) > 1e-14 * max(
                    1.0, abs(alpha) + abs(beta)
                )

                def _fwd_rank(idx):
                    g = raw_gamma[idx]
                    sz = float(raw_sz[idx])
                    if cplx_kt:
                        if abs(g.imag) > max(im_tol, abs(g.real)):
                            return (
                                1 if g.imag < 0 else -1,
                                -float(g.imag),
                                sz,
                                float(g.real),
                            )
                        if abs(sz) > sz_tol:
                            return (
                                0,
                                1 if sz > 0 else -1,
                                sz,
                                float(g.real - g.imag),
                            )
                        return (
                            0,
                            1 if (g.real - g.imag >= 0) else -1,
                            float(g.real - g.imag),
                            -float(g.imag),
                        )
                    if abs(g.imag) > im_tol:
                        return (
                            1 if g.imag < 0 else -1,
                            -float(g.imag),
                            sz,
                            float(g.real),
                        )
                    if abs(sz) > sz_tol:
                        return (0, 1 if sz > 0 else -1, sz, float(g.real - g.imag))
                    return (0, 0, float(g.real - g.imag), -float(g.imag))

                ranked = sorted(range(4), key=_fwd_rank, reverse=True)
                fwd = sorted(
                    ranked[:2],
                    key=lambda idx: (raw_gamma[idx].real, -raw_gamma[idx].imag),
                    reverse=True,
                )
                bwd = sorted(
                    ranked[2:],
                    key=lambda idx: (raw_gamma[idx].real, -raw_gamma[idx].imag),
                )
                order = [fwd[0], bwd[0], fwd[1], bwd[1]]
                gamma = raw_gamma[order]
                raw_p = raw_p[order, :]

            k = (
                k0
                * np.array(
                    [alpha * np.ones(gamma.shape), beta * np.ones(gamma.shape), gamma]
                ).T
            )
            p = np.zeros((4, 3), dtype=complex)
            if abs(alpha) + abs(beta) > 1e-12:
                p_s_ref = np.array([-beta, alpha, 0.0], dtype=complex)
            else:
                p_s_ref = np.array([0.0, 1.0, 0.0], dtype=complex)
            p_s_ref /= np.linalg.norm(p_s_ref)

            if raw_p is None:
                # Isotropic layer: construct exact orthogonal s and p basis
                p_p_fwd = np.cross(p_s_ref, k[0, :])
                p_p_fwd /= np.linalg.norm(p_p_fwd)
                p_p_bwd = np.cross(p_s_ref, k[1, :])
                p_p_bwd /= np.linalg.norm(p_p_bwd)
                p = np.array([p_s_ref, p_s_ref, p_p_fwd, p_p_bwd], dtype=complex)
            else:
                p = raw_p.copy()
                # Only treat a forward or backward pair as degenerate at near-machine
                # precision (preserving weak birefringence down to ~1e-11), and span
                # the exact 2D SVD nullspace of H(gamma) so biaxial optic axes (where
                # D perp k but E not perp k) satisfy Maxwell's wave equation exactly.
                deg_tol = 1e-11 * max(1.0, float(np.max(np.abs(gamma))))
                for i0, i1 in ((0, 2), (1, 3)):
                    if abs(gamma[i0] - gamma[i1]) < deg_tol:
                        g_avg = 0.5 * (gamma[i0] + gamma[i1])
                        H_deg = K + np.array(
                            [
                                [-beta2 - g_avg**2, alpha * beta, alpha * g_avg],
                                [alpha * beta, -alpha2 - g_avg**2, beta * g_avg],
                                [alpha * g_avg, beta * g_avg, -alpha2 - beta2],
                            ],
                            dtype=complex,
                        )
                        _, _, vh_deg = np.linalg.svd(H_deg)
                        V_null = vh_deg[-2:, :].conj().T  # (3, 2) orthonormal basis
                        c_s = V_null.conj().T @ p_s_ref
                        nrm_cs = float(np.linalg.norm(c_s))
                        if nrm_cs > 1e-14:
                            c_s = c_s / nrm_cs
                        else:
                            c_s = np.array([1.0 + 0j, 0.0 + 0j])
                        c_p = np.array([-np.conj(c_s[1]), np.conj(c_s[0])])
                        v_s = V_null @ c_s
                        v_p = V_null @ c_p
                        p_p_ref = np.cross(p_s_ref, k[i0, :])
                        if np.linalg.norm(p_p_ref) > 0 and (
                            np.real(np.vdot(p_p_ref, v_p)) < 0
                        ):
                            v_p = -v_p
                        p[i0, :] = v_s / np.linalg.norm(v_s)
                        p[i1, :] = v_p / np.linalg.norm(v_p)

            q = wl / (2.0 * np.pi * mu0 * c) * np.cross(k, p)

            return k, p, q

        nlayers = len(multilayer)
        d = np.asarray([l.thickness for l in multilayer])

        # R and T are real, because they are powers
        # r and t are complex!
        R = np.zeros((2, 2, self.wls.size))
        T = np.zeros((2, 2, self.wls.size))
        R_tot = np.zeros((2, self.wls.size))
        T_tot = np.zeros((2, self.wls.size))

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

            # Bounded Enhanced Transmittance Matrix (ETM) cascade:
            # Permute each layer's modes into forward [0, 2] and backward [1, 3]
            # pairs so all internal propagation factors X_f = exp(-i k_z_f d) and
            # X_b = exp(+i k_z_b d) stay bounded (|X| <= 1 in passive layers),
            # avoiding catastrophic cancellation in 4x4 transfer-matrix minors on
            # thick coupled anisotropic layers and overflow on thick metals.
            idx_fb = [0, 2, 1, 3]
            MT = D[:, [0, 2], -1]
            step_T = []
            for i in range(nlayers - 2, 0, -1):
                D_ord = D[:, idx_fb, i]
                x_f = np.exp(-1j * k[[0, 2], 2, i] * d[i])
                x_b = np.exp(1j * k[[1, 3], 2, i] * d[i])
                MTp = np.linalg.solve(D_ord, MT)
                a_f = MTp[:2, :]
                a_b = MTp[2:, :]
                S_i = np.linalg.solve(a_f, np.diag(x_f))
                step_T.append(S_i)
                MT = D[:, [0, 2], i] + (D[:, [1, 3], i] * x_b[None, :]) @ (a_b @ S_i)

            # Match at the superstrate (z = 0): D_{f,0} + D_{b,0} R = MT T_0
            sol = np.linalg.solve(np.c_[-D[:, [1, 3], 0], MT], D[:, [0, 2], 0])
            r = sol[:2, :].T
            t_curr = sol[2:, :]
            for S_i in reversed(step_T):
                t_curr = S_i @ t_curr
            t = t_curr.T

            # Compute power reflectance and transmittance from z-Poynting flux:
            # Sz_j = 0.5 * Re(Ex_j * Hy_j^* - Ey_j * Hx_j^*)
            def _sz(p_vec, q_vec):
                return 0.5 * np.real(
                    p_vec[0] * np.conj(q_vec[1]) - p_vec[1] * np.conj(q_vec[0])
                )

            sz_inc = np.array(
                [_sz(p[0, :, 0], q[0, :, 0]), _sz(p[2, :, 0], q[2, :, 0])]
            )
            sz_ref = np.array(
                [-_sz(p[1, :, 0], q[1, :, 0]), -_sz(p[3, :, 0], q[3, :, 0])]
            )
            sz_tr = np.array(
                [_sz(p[0, :, -1], q[0, :, -1]), _sz(p[2, :, -1], q[2, :, -1])]
            )

            R[:, :, iwl] = np.absolute(r) ** 2 * (sz_ref[None, :] / sz_inc[:, None])
            T[:, :, iwl] = np.absolute(t) ** 2 * (sz_tr[None, :] / sz_inc[:, None])

            # Total reflected and transmitted z-Poynting flux per input polarization
            # (includes cross-mode interference when end media are both lossy and birefringent).
            for m_in in range(2):
                e_ref = r[m_in, 0] * p[1, :, 0] + r[m_in, 1] * p[3, :, 0]
                h_ref = r[m_in, 0] * q[1, :, 0] + r[m_in, 1] * q[3, :, 0]
                e_tr = t[m_in, 0] * p[0, :, -1] + t[m_in, 1] * p[2, :, -1]
                h_tr = t[m_in, 0] * q[0, :, -1] + t[m_in, 1] * q[2, :, -1]
                R_tot[m_in, iwl] = -_sz(e_ref, h_ref) / sz_inc[m_in]
                T_tot[m_in, iwl] = _sz(e_tr, h_tr) / sz_inc[m_in]

        self.R = R
        self.T = T
        self.R_tot = R_tot
        self.T_tot = T_tot
        return self

    def __str__(self):
        return (
            "ANISOTROPIC TRANSFER MATRIX SOLVER\n\n%s\n\ntheta inc x = %g\ntheta inc y = %g"
            % (self.multilayer.__str__(), self.theta_inc_x, self.theta_inc_y)
        )
