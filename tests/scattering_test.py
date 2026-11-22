import unittest

import numpy

import EMpy.scattering as S


class ScatteringTest(unittest.TestCase):
    def test_stack_shapes(self):
        x = numpy.arange(4)
        y = numpy.arange(3)
        z = numpy.arange(2)
        expected = S.stack(x, y, z)
        self.assertEqual(expected.shape, (3, 24))
        numpy.testing.assert_array_equal(
            expected, S.stack(x.reshape(1, -1), y.reshape(-1, 1), list(z))
        )

    def test_kottler_validation_and_dS_shapes(self):
        P = numpy.array([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
        Q = numpy.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        J = numpy.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        M = numpy.zeros_like(J)
        dS_1d = numpy.array([0.1, 0.2])
        dS_2d = dS_1d.reshape(1, -1)

        f1 = S.currentsScatteringKottler(P, J, M, Q, dS_1d, f=1e9)
        f2 = S.currentsScatteringKottler(P, J, M, Q, dS_2d, f=1e9)
        numpy.testing.assert_allclose(f1.E, f2.E)
        numpy.testing.assert_allclose(f1.H, f2.H)
        self.assertTrue(numpy.all(numpy.isfinite(f1.E)))
        self.assertTrue(numpy.all(numpy.isfinite(f1.H)))

        with self.assertRaises(ValueError):
            S.currentsScatteringKottler(numpy.zeros((2, 2)), J, M, Q, dS_1d, f=1e9)
        with self.assertRaises(ValueError):
            S.currentsScatteringKottler(P, J, M, Q, numpy.array([0.1]), f=1e9)

    def test_kottler_self_term_r_zero(self):
        Q = numpy.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        P = Q.copy()
        J = numpy.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        M = numpy.zeros_like(J)
        dS = numpy.array([0.1, 0.1])

        field = S.currentsScatteringKottler(P, J, M, Q, dS, f=1e9)
        self.assertTrue(numpy.all(numpy.isfinite(field.E)))
        self.assertTrue(numpy.all(numpy.isfinite(field.H)))

        field_single = S.currentsScatteringKottler(
            Q[:, :1], J[:, :1], M[:, :1], Q[:, :1], dS[:1], f=1e9
        )
        numpy.testing.assert_array_equal(field_single.E, numpy.zeros((3, 1)))
        numpy.testing.assert_array_equal(field_single.H, numpy.zeros((3, 1)))

    def test_kottler_maxwell_curl_farfield_impedance_and_lossy_limits(self):
        """Verify Kottler scattering obeys Maxwell's curl equations with exact CODATA eta0 = mu0*c and lossy limits."""
        import EMpy.constants as C

        self.assertAlmostEqual(C.eta0, numpy.sqrt(C.mu0 / C.eps0), places=12)
        numpy.testing.assert_allclose(C.eta0, 376.730313668, rtol=1e-9)
        f = 1.0e9
        omega = 2.0 * numpy.pi * f
        epsr = 2.25
        Q = numpy.array([[0.0], [0.0], [0.0]])
        J = numpy.array([[0.3 - 0.1j], [-0.2 + 0.4j], [0.7 + 0.2j]])
        M = numpy.array([[-0.15 + 0.25j], [0.35 - 0.1j], [0.2 - 0.3j]])
        dS = numpy.array([0.05])

        # 1. Central finite-difference check of Faraday (curl E = -j*w*mu0*H) and Ampere (curl H = +j*w*eps0*epsr*E)
        p0 = numpy.array([0.42, -0.31, 0.53])
        h = 1.0e-5
        pts = [p0]
        for ax in range(3):
            dp = numpy.zeros(3)
            dp[ax] = h
            pts.append(p0 + dp)
            pts.append(p0 - dp)
        P_sten = numpy.column_stack(pts)
        fld = S.currentsScatteringKottler(P_sten, J, M, Q, dS, f=f, epsr=epsr)
        E0, H0 = fld.E[:, 0], fld.H[:, 0]
        dE = [
            (fld.E[:, 2 * ax + 1] - fld.E[:, 2 * ax + 2]) / (2.0 * h) for ax in range(3)
        ]
        dH = [
            (fld.H[:, 2 * ax + 1] - fld.H[:, 2 * ax + 2]) / (2.0 * h) for ax in range(3)
        ]
        curl_E = numpy.array(
            [dE[1][2] - dE[2][1], dE[2][0] - dE[0][2], dE[0][1] - dE[1][0]]
        )
        curl_H = numpy.array(
            [dH[1][2] - dH[2][1], dH[2][0] - dH[0][2], dH[0][1] - dH[1][0]]
        )
        numpy.testing.assert_allclose(
            curl_E, -1j * omega * C.mu0 * H0, rtol=1e-8, atol=1e-10
        )
        numpy.testing.assert_allclose(
            curl_H, 1j * omega * C.eps0 * epsr * E0, rtol=1e-8, atol=1e-10
        )

        # 2. Far-field wave impedance |E_far| / |H_far| == eta0 / sqrt(epsr) (not 120*pi / sqrt(epsr))
        P_far = numpy.array([[0.0], [0.0], [1.0e5]])
        J_x = numpy.array([[1.0 + 0j], [0.0], [0.0]])
        M_0 = numpy.zeros_like(J_x)
        f_far = S.currentsScatteringKottler(P_far, J_x, M_0, Q, dS, f=f, epsr=epsr)
        z_meas = numpy.linalg.norm(f_far.E[:, 0]) / numpy.linalg.norm(f_far.H[:, 0])
        self.assertAlmostEqual(
            float(z_meas), float(C.eta0 / numpy.sqrt(epsr)), delta=1e-7
        )

        # 3. Passive lossy dielectric epsr = (n_r - 1j * kap)**2: spatial attenuation and kap -> 0+ continuity
        P_obs = numpy.array([[0.5, 2.5], [0.0, 0.0], [0.0, 0.0]])
        f_ll = S.currentsScatteringKottler(P_obs, J, M, Q, dS, f=f, epsr=1.5**2)
        f_c0 = S.currentsScatteringKottler(
            P_obs, J, M, Q, dS, f=f, epsr=(1.5 - 0.0j) ** 2
        )
        numpy.testing.assert_allclose(f_c0.E, f_ll.E, rtol=1e-14, atol=1e-16)
        numpy.testing.assert_allclose(f_c0.H, f_ll.H, rtol=1e-14, atol=1e-16)

        e_scale = float(numpy.max(numpy.abs(f_ll.E)))
        kap_errs = []
        for kap in (1e-2, 1e-5, 1e-8):
            f_k = S.currentsScatteringKottler(
                P_obs, J, M, Q, dS, f=f, epsr=(1.5 - 1j * kap) ** 2
            )
            # Farther point (r = 2.5 m) must experience stronger attenuation relative to lossless than r = 0.5 m
            ratio_near = numpy.linalg.norm(f_k.E[:, 0]) / numpy.linalg.norm(
                f_ll.E[:, 0]
            )
            ratio_far = numpy.linalg.norm(f_k.E[:, 1]) / numpy.linalg.norm(f_ll.E[:, 1])
            self.assertLess(ratio_far, ratio_near)
            self.assertLess(ratio_near, 1.0)
            kap_errs.append(float(numpy.max(numpy.abs(f_k.E - f_ll.E))) / e_scale)
        self.assertLess(kap_errs[1], kap_errs[0])
        self.assertLess(kap_errs[2], kap_errs[1])
        self.assertLess(kap_errs[2], 1e-6)

        # 4. Negative real permittivity epsr < 0 (lossless plasma/metal): finite evanescent decay, no NaN/blowup
        P_ev = numpy.array([[0.1, 0.3], [0.0, 0.0], [0.0, 0.0]])
        f_ev_r = S.currentsScatteringKottler(P_ev, J, M, Q, dS, f=f, epsr=-4.0)
        f_ev_c = S.currentsScatteringKottler(P_ev, J, M, Q, dS, f=f, epsr=-4.0 - 1e-9j)
        self.assertTrue(numpy.all(numpy.isfinite(f_ev_r.E)))
        numpy.testing.assert_allclose(f_ev_r.E, f_ev_c.E, rtol=1e-7, atol=1e-9)
        self.assertLess(
            numpy.linalg.norm(f_ev_r.E[:, 1]), numpy.linalg.norm(f_ev_r.E[:, 0])
        )

    def test_kottler_exact_hertzian_electric_and_magnetic_dipole(self):
        """Verify currentsScatteringKottler against exact closed-form Hertzian E/M dipole fields."""
        import EMpy.constants as C

        f = 1.0e9
        omega = 2.0 * numpy.pi * f
        Q = numpy.array([[0.0], [0.0], [0.0]])
        dS = numpy.array([0.05])
        # Near-field (~0.1 wl), mid-field (~1 wl), and far-field (~20 wl) points
        P = numpy.array(
            [
                [0.02, 0.21, 3.5],
                [-0.015, 0.18, -2.4],
                [0.025, -0.14, 4.1],
            ]
        )
        p_vec = numpy.array([0.3 - 0.1j, -0.2 + 0.4j, 0.7 + 0.2j])  # J * dS
        m_vec = numpy.array([-0.15 + 0.25j, 0.35 - 0.1j, 0.2 - 0.3j])  # M * dS
        J = (p_vec / dS[0])[:, None]
        M = (m_vec / dS[0])[:, None]
        zeros = numpy.zeros_like(J)

        for epsr in (2.25, 2.25 - 0.4j):
            n = numpy.sqrt(complex(epsr))
            if n.real - n.imag < 0:
                n = -n
            k = omega * n / C.c
            eps = C.eps0 * epsr
            eta = C.eta0 / n

            R = numpy.linalg.norm(P, axis=0)
            rhat = P / R[None, :]
            g = numpy.exp(-1j * k * R) / (4.0 * numpy.pi * R)
            ikR = 1j * k * R

            # Exact Hertzian electric dipole (J * dS = p_vec) under exp(+iwt - ikR):
            # E_J = (1 / (j*w*eps)) * g * [(k^2 - (1 + jkR)/R^2)*p + (-k^2 + 3*(1 + jkR)/R^2)*(rhat.p)*rhat]
            # Wait: check exact sign:
            # E_J = -j*w*mu0 * g * [ (1 - j/(kR) - 1/(kR)^2)*p + (-1 + 3j/(kR) + 3/(kR)^2)*(rhat.p)*rhat ]
            # H_J = (j*k) * (1 + 1/(j*k*R)) * g * (p x rhat)
            p_dot_r = numpy.sum(rhat * p_vec[:, None], axis=0)
            E_J_exact = (
                -1j
                * omega
                * C.mu0
                * g[None, :]
                * (
                    (1.0 + 1.0 / ikR + 1.0 / ikR**2)[None, :] * p_vec[:, None]
                    + (-1.0 - 3.0 / ikR - 3.0 / ikR**2)[None, :]
                    * p_dot_r[None, :]
                    * rhat
                )
            )
            H_J_exact = (
                (1j * k)
                * (1.0 + 1.0 / ikR)[None, :]
                * g[None, :]
                * numpy.cross(p_vec[:, None], rhat, axis=0)
            )

            fld_J = S.currentsScatteringKottler(P, J, zeros, Q, dS, f=f, epsr=epsr)
            numpy.testing.assert_allclose(fld_J.E, E_J_exact, rtol=1e-12, atol=1e-14)
            numpy.testing.assert_allclose(fld_J.H, H_J_exact, rtol=1e-12, atol=1e-14)

            # Exact Hertzian magnetic dipole (M * dS = m_vec) by duality (J -> M, E -> H, H -> -E, mu -> eps):
            m_dot_r = numpy.sum(rhat * m_vec[:, None], axis=0)
            H_M_exact = (
                -1j
                * omega
                * eps
                * g[None, :]
                * (
                    (1.0 + 1.0 / ikR + 1.0 / ikR**2)[None, :] * m_vec[:, None]
                    + (-1.0 - 3.0 / ikR - 3.0 / ikR**2)[None, :]
                    * m_dot_r[None, :]
                    * rhat
                )
            )
            E_M_exact = (
                -(1j * k)
                * (1.0 + 1.0 / ikR)[None, :]
                * g[None, :]
                * numpy.cross(m_vec[:, None], rhat, axis=0)
            )
            fld_M = S.currentsScatteringKottler(P, zeros, M, Q, dS, f=f, epsr=epsr)
            numpy.testing.assert_allclose(fld_M.E, E_M_exact, rtol=1e-12, atol=1e-14)
            numpy.testing.assert_allclose(fld_M.H, H_M_exact, rtol=1e-12, atol=1e-14)
            _ = eta
