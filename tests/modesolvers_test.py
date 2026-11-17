from unittest import TestCase

import matplotlib
import numpy

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

import EMpy.modesolvers  # noqa: E402


class VFDModeSolverTest(TestCase):

    def test_solve(self):

        def epsfunc(x_, y_):
            """Return a matrix describing a 2d material.

            :param x_: x values
            :param y_: y values
            :return: 2d-matrix
            """
            xx, yy = numpy.meshgrid(x_, y_)
            return numpy.where(
                (numpy.abs(xx.T - 1.24e-6) <= 0.24e-6)
                * (numpy.abs(yy.T - 1.11e-6) <= 0.11e-6),
                3.4757**2,
                1.446**2,
            )

        neigs = 2
        tol = 1e-8
        boundary = "0000"
        wl = 1.55e-6
        x = numpy.linspace(0, 2.48e-6, 125)
        y = numpy.linspace(0, 2.22e-6, 112)
        solver = EMpy.modesolvers.FD.VFDModeSolver(wl, x, y, epsfunc, boundary).solve(
            neigs, tol
        )
        self.assertEqual(len(solver.modes), 2)
        self.assertAlmostEqual(solver.modes[0].neff, 2.4179643622942937)

        O11, O22, O12, O21 = EMpy.modesolvers.interface.interface_matrix(solver, solver)
        numpy.testing.assert_allclose(O11, numpy.eye(2), atol=1e-6)
        numpy.testing.assert_allclose(O12, numpy.eye(2), atol=1e-6)

    def test_svfd_symmetry_matches_full_domain(self):
        def epsfunc(x_, y_):
            xx, yy = numpy.meshgrid(x_, y_)
            return numpy.where(
                (numpy.abs(xx.T) <= 0.25e-6) * (numpy.abs(yy.T) <= 0.12e-6),
                3.48**2,
                1.44**2,
            )

        wl = 1.55e-6
        x_full = numpy.linspace(-1.0e-6, 1.0e-6, 41)
        y_full = numpy.linspace(-1.0e-6, 1.0e-6, 41)
        x_q = x_full[20:]
        y_q = y_full[20:]

        # Fundamental Ex mode: Hx antisymmetric, Hy symmetric ('A' on S/W walls)
        sv_ex_full = EMpy.modesolvers.FD.SVFDModeSolver(
            wl, x_full, y_full, epsfunc, "0000", method="Ex"
        ).solve(1, 1e-10)
        sv_ex_q = EMpy.modesolvers.FD.SVFDModeSolver(
            wl, x_q, y_q, epsfunc, "0A0A", method="Ex"
        ).solve(1, 1e-10)
        self.assertAlmostEqual(sv_ex_q.neff[0].real, sv_ex_full.neff[0].real, places=9)

        # Fundamental Ey mode: Hx symmetric, Hy antisymmetric ('S' on S/W walls)
        sv_ey_full = EMpy.modesolvers.FD.SVFDModeSolver(
            wl, x_full, y_full, epsfunc, "0000", method="Ey"
        ).solve(1, 1e-10)
        sv_ey_q = EMpy.modesolvers.FD.SVFDModeSolver(
            wl, x_q, y_q, epsfunc, "0S0S", method="Ey"
        ).solve(1, 1e-10)
        self.assertAlmostEqual(sv_ey_q.neff[0].real, sv_ey_full.neff[0].real, places=9)


class FMMModeSolverTest(TestCase):

    def test_fmm1d_uniform_lossy_tm(self):
        Uy = numpy.array([0.0, 2.0, 2.22, 4.22])
        n1 = 3.4757 - 1e-4j
        ny = numpy.array([n1, n1, n1])
        solver = EMpy.modesolvers.FMM.FMM1d(Uy, ny, "AA").solve(1.55, 2, "TM")
        self.assertEqual(len(solver.modes), 2)
        self.assertTrue(numpy.iscomplexobj(solver.modes[0].sl))
        self.assertTrue(numpy.isfinite(solver.modes[0].keff).all())
        y_grid = numpy.linspace(Uy[0], Uy[-1], 21)
        plt.figure()
        solver.modes[0].plot(y_grid)
        _, line_imag = plt.gca().get_lines()
        numpy.testing.assert_allclose(
            line_imag.get_ydata(), numpy.imag(solver.modes[0].eval(y_grid))
        )
        plt.close("all")

    def test_fmm_shooting_complex_delta(self):
        wl = 1.55
        k0 = 2.0 * numpy.pi / wl
        Uy = numpy.array([0.0, 2.0, 2.22, 4.22])
        n2 = 1.446
        n1 = 3.4757 - 1e-3j
        ny = numpy.array([n2, n1, n2])
        pars = {
            "beta": k0 * ny,
            "epsilon": ny**2,
            "Uy": Uy,
            "boundary": "AA",
        }
        kz_real = numpy.linspace(k0 * 1.5, k0 * 3.4, 5)
        delta_te, _ = EMpy.modesolvers.FMM.FMMshooting(kz_real, pars)
        delta_tm, _ = EMpy.modesolvers.FMM.FMMshootingTM(kz_real, pars)
        self.assertTrue(numpy.iscomplexobj(delta_te))
        self.assertTrue(numpy.iscomplexobj(delta_tm))
        self.assertGreater(numpy.max(numpy.abs(numpy.imag(delta_te))), 0.0)
        self.assertGreater(numpy.max(numpy.abs(numpy.imag(delta_tm))), 0.0)

        for pol, shoot_fn in [
            ("TE", EMpy.modesolvers.FMM.FMMshooting),
            ("TM", EMpy.modesolvers.FMM.FMMshootingTM),
        ]:
            sol = EMpy.modesolvers.FMM.FMM1d(Uy, ny, "AA").solve(wl, 1, pol)
            delta_root, _ = shoot_fn(sol.modes[0].keff, pars)
            self.assertLess(abs(delta_root[0]), 1e-4)

    def test_fmm2d_lossy_channel_waveguide(self):
        wl = 1.55
        Ux = numpy.array([0.0, 2.0, 2.4, 4.4])
        Uy = numpy.array([0.0, 2.0, 2.22, 4.22])
        boundary = EMpy.modesolvers.FMM.Boundary(
            xleft="Electric Wall",
            yleft="Magnetic Wall",
            xright="Electric Wall",
            yright="Magnetic Wall",
        )
        n2 = 1.446
        n1 = 3.4757 - 1e-4j
        refindex = numpy.array([[n2, n2, n2], [n2, n1, n2], [n2, n2, n2]])
        simul = EMpy.modesolvers.FMM.FMM2d(Ux, Uy, refindex, boundary).solve(wl, 2, 2)
        self.assertEqual(len(simul.modes), 2)
        self.assertTrue(numpy.isfinite(simul.modes[0].keff))
        self.assertTrue(numpy.isfinite(simul.modes[0].norm()))

        m0 = simul.modes[0]
        m0.normalize()
        ov_m0 = EMpy.modesolvers.interface.overlap(m0, m0, None, None)
        self.assertAlmostEqual(numpy.real(ov_m0), 1.0, places=10)
        self.assertLess(abs(numpy.imag(ov_m0)), 1e-4)
        O11, _, _, _ = EMpy.modesolvers.interface.interface_matrix(simul, simul)
        self.assertEqual(O11.shape, (2, 2))
        te_frac_old = m0.TEfrac_old(
            numpy.linspace(Ux[0], Ux[-1], 51), numpy.linspace(Uy[0], Uy[-1], 51)
        )
        te_frac = m0.TEfrac()
        self.assertTrue(0.0 <= te_frac <= 1.0)
        self.assertTrue(0.0 <= te_frac_old <= 1.0)
        self.assertAlmostEqual(te_frac, te_frac_old, delta=5e-2)
        m0.plot(numpy.linspace(Ux[0], Ux[-1], 11), numpy.linspace(Uy[0], Uy[-1], 11))
        plt.close("all")

        # get_fields_for_FDTD with non-square grid (len(x) != len(y))
        x_fdtd = numpy.linspace(Ux[0], Ux[-1], 9)
        y_fdtd = numpy.linspace(Uy[0], Uy[-1], 13)
        Ex_f, Ey_f, Ez_f, Hx_f, Hy_f, Hz_f = m0.get_fields_for_FDTD(x_fdtd, y_fdtd)
        self.assertEqual(Ex_f.shape, (len(x_fdtd) - 1, len(y_fdtd) - 2))
        self.assertEqual(Ey_f.shape, (len(x_fdtd) - 2, len(y_fdtd) - 1))
        self.assertEqual(Ez_f.shape, (len(x_fdtd) - 2, len(y_fdtd) - 2))
        self.assertEqual(Hx_f.shape, (len(x_fdtd) - 2, len(y_fdtd) - 1))
        self.assertEqual(Hy_f.shape, (len(x_fdtd) - 1, len(y_fdtd) - 2))
        self.assertEqual(Hz_f.shape, (len(x_fdtd) - 1, len(y_fdtd) - 1))
        # Direct evaluation at Ez grid points must match interpolated Ez_f closely
        _, _, Ez_direct, _, _, _ = m0.fields(x_fdtd[1:-1], y_fdtd[1:-1])
        numpy.testing.assert_allclose(Ez_f, Ez_direct.T, rtol=5e-2, atol=1e-3)

    def test_fmm2d_asymmetric_x_boundaries(self):
        """Mirrored (Electric, Magnetic) vs (Magnetic, Electric) x-walls on a symmetric waveguide must give identical neff."""
        wl = 1.55
        Ux = numpy.array([0.0, 0.1, 0.6, 0.7])
        Uy = numpy.array([0.0, 0.5, 1.0, 1.5])
        n2, n1 = 1.446, 3.4757
        refindex = numpy.array([[n2, n2, n2], [n2, n1, n2], [n2, n2, n2]])

        b_em = EMpy.modesolvers.FMM.Boundary(
            xleft="Electric Wall",
            yleft="Magnetic Wall",
            xright="Magnetic Wall",
            yright="Magnetic Wall",
        )
        b_me = EMpy.modesolvers.FMM.Boundary(
            xleft="Magnetic Wall",
            yleft="Magnetic Wall",
            xright="Electric Wall",
            yright="Magnetic Wall",
        )
        b_ee = EMpy.modesolvers.FMM.Boundary(
            xleft="Electric Wall",
            yleft="Magnetic Wall",
            xright="Electric Wall",
            yright="Magnetic Wall",
        )
        s_em = EMpy.modesolvers.FMM.FMM2d(Ux, Uy, refindex, b_em).solve(wl, 6, 1)
        s_me = EMpy.modesolvers.FMM.FMM2d(Ux, Uy, refindex, b_me).solve(wl, 6, 1)
        s_ee = EMpy.modesolvers.FMM.FMM2d(Ux, Uy, refindex, b_ee).solve(wl, 6, 1)
        numpy.testing.assert_allclose(s_em.modes[0].neff, s_me.modes[0].neff, rtol=1e-3)
        self.assertGreater(
            abs(s_em.modes[0].neff - s_ee.modes[0].neff),
            1e-3,
        )

    def test_fmm2d_overlap_tefrac_cross_waveguide_and_lossless_limit(self):
        """Verify FMMMode2d TEfrac, TM norm, cross-waveguide overlap, and kappa -> 0+ continuity."""
        wl = 1.55
        Ux1 = numpy.array([0.0, 1.0, 1.5, 2.5])
        Uy1 = numpy.array([0.0, 1.0, 1.22, 2.22])
        boundary = EMpy.modesolvers.FMM.Boundary(
            xleft="Electric Wall",
            yleft="Magnetic Wall",
            xright="Electric Wall",
            yright="Magnetic Wall",
        )
        rix_lossless = numpy.array(
            [[1.45, 1.45, 1.45], [1.45, 3.45, 1.45], [1.45, 1.45, 1.45]]
        )
        s0 = EMpy.modesolvers.FMM.FMM2d(Ux1, Uy1, rix_lossless, boundary).solve(
            wl, 2, 2
        )

        # Both quasi-TE (mode 0) and quasi-TM (mode 1) must have real positive norm and TEfrac in [0, 1]
        for m in s0.modes:
            n_val = m.norm()
            self.assertAlmostEqual(numpy.imag(n_val), 0.0, places=12)
            self.assertGreater(numpy.real(n_val), 0.0)
            m.normalize()

        xg = numpy.unique(
            numpy.r_[
                numpy.linspace(0.0, 1.0, 81),
                numpy.linspace(1.0001, 1.4999, 61),
                numpy.linspace(1.5, 2.5, 81),
            ]
        )
        yg = numpy.unique(
            numpy.r_[
                numpy.linspace(0.0, 1.0, 81),
                numpy.linspace(1.0001, 1.2199, 41),
                numpy.linspace(1.22, 2.22, 81),
            ]
        )

        self.assertGreater(s0.modes[0].TEfrac(), 0.9)
        self.assertLess(s0.modes[1].TEfrac(), 0.1)
        self.assertAlmostEqual(
            s0.modes[0].TEfrac(), s0.modes[0].TEfrac_old(xg, yg), delta=5e-3
        )
        self.assertAlmostEqual(
            s0.modes[1].TEfrac(), s0.modes[1].TEfrac_old(xg, yg), delta=5e-3
        )

        # Orthonormality on lossless waveguide: O11 == I_2
        O11, _, _, _ = EMpy.modesolvers.interface.interface_matrix(s0, s0)
        numpy.testing.assert_allclose(O11, numpy.eye(2), atol=1e-10)

        # Cross-waveguide overlap with distinct (Ux, Uy, n_core) must satisfy |O12| < 1 and match quadrature
        Ux2 = numpy.array([0.0, 0.7, 1.8, 2.5])
        Uy2 = numpy.array([0.0, 0.9, 1.32, 2.22])
        rix2 = numpy.array([[1.45, 1.45, 1.45], [1.45, 2.50, 1.45], [1.45, 1.45, 1.45]])
        s2 = EMpy.modesolvers.FMM.FMM2d(Ux2, Uy2, rix2, boundary).solve(wl, 2, 1)
        s2.modes[0].normalize()

        ov_ana = s0.modes[0].overlap(s2.modes[0])
        xg_both = numpy.unique(numpy.r_[xg, Ux2])
        yg_both = numpy.unique(numpy.r_[yg, Uy2])
        ov_quad = s0.modes[0].overlap_old(s2.modes[0], xg_both, yg_both)
        self.assertLess(abs(ov_ana), 1.0)
        self.assertGreater(ov_ana.real, 0.5)
        numpy.testing.assert_allclose(ov_ana, ov_quad, atol=5e-3)

        # Lossy -> non-lossy degenerate limit (kappa = 1e-3, 1e-6, 0.0+0j -> lossless)
        neff_errs = []
        tefrac_errs = []
        s_lossy_ref = None
        for kap in [1e-3, 1e-6, 0.0]:
            nc = complex(3.45, -kap)
            rk = numpy.array([[1.45, 1.45, 1.45], [1.45, nc, 1.45], [1.45, 1.45, 1.45]])
            sk = EMpy.modesolvers.FMM.FMM2d(Ux1, Uy1, rk, boundary).solve(wl, 2, 2)
            if kap > 0:
                self.assertLess(sk.modes[0].neff.imag, 0.0)
                self.assertLess(sk.modes[1].neff.imag, 0.0)
            sk.modes[0].normalize()
            sk.modes[1].normalize()
            if kap == 1e-3:
                s_lossy_ref = sk
            self.assertAlmostEqual(
                sk.modes[0].overlap(sk.modes[0]).real, 1.0, places=10
            )
            self.assertAlmostEqual(
                sk.modes[1].overlap(sk.modes[1]).real, 1.0, places=10
            )
            neff_errs.append(float(abs(sk.modes[0].neff - s0.modes[0].neff)))
            tefrac_errs.append(float(abs(sk.modes[0].TEfrac() - s0.modes[0].TEfrac())))

        self.assertLess(neff_errs[1], neff_errs[0])
        self.assertLess(neff_errs[2], 1e-8)
        self.assertLess(tefrac_errs[1], tefrac_errs[0])
        self.assertLess(tefrac_errs[2], 1e-8)

        # Independent Proof 1: Black-box 2D Gauss-Legendre integration of fields(x, y)
        # on smooth sub-cells shares zero code with __overlap / scalarprod / _refine_mode1d
        # and must match analytical (Sx, Sy) and overlap() to near machine precision (~1e-13).
        assert s_lossy_ref is not None
        xi, wi = numpy.polynomial.legendre.leggauss(18)

        def _gl_poynting(ma, mb):
            ux = numpy.unique(numpy.r_[Ux1, Ux2])
            uy = numpy.unique(numpy.r_[Uy1, Uy2])
            x_pts = numpy.concatenate(
                [
                    0.5 * (ux[i + 1] - ux[i]) * xi + 0.5 * (ux[i + 1] + ux[i])
                    for i in range(len(ux) - 1)
                ]
            )
            w_x = numpy.concatenate(
                [0.5 * (ux[i + 1] - ux[i]) * wi for i in range(len(ux) - 1)]
            )
            y_pts = numpy.concatenate(
                [
                    0.5 * (uy[j + 1] - uy[j]) * xi + 0.5 * (uy[j + 1] + uy[j])
                    for j in range(len(uy) - 1)
                ]
            )
            w_y = numpy.concatenate(
                [0.5 * (uy[j + 1] - uy[j]) * wi for j in range(len(uy) - 1)]
            )
            Ex1, Ey1, _, _, _, _ = ma.fields(x_pts, y_pts)
            _, _, _, cBx2, cBy2, _ = mb.fields(x_pts, y_pts)
            w2d = w_y[:, None] * w_x[None, :]
            return (
                0.5 * numpy.sum(w2d * Ex1 * numpy.conj(cBy2)),
                0.5 * numpy.sum(w2d * (-Ey1) * numpy.conj(cBx2)),
            )

        for ma, mb in [
            (s_lossy_ref.modes[0], s_lossy_ref.modes[0]),
            (s_lossy_ref.modes[1], s_lossy_ref.modes[1]),
            (s_lossy_ref.modes[0], s2.modes[0]),
            (s2.modes[0], s_lossy_ref.modes[1]),
        ]:
            sx_gl, sy_gl = _gl_poynting(ma, mb)
            numpy.testing.assert_allclose(
                ma.overlap(mb), sx_gl + sy_gl, rtol=1e-12, atol=1e-13
            )

        # Independent Proof 2: Pointwise Maxwell curl and divergence equations inside
        # lossy core and cladding cells (verifies exact du/dx consistency in eval()).
        k0 = 2.0 * numpy.pi / wl
        m_chk = s_lossy_ref.modes[0]
        kz = m_chk.keff
        h = 1e-6
        for x0, y0, n_cell in [
            (1.25, 1.11, 3.45 - 1e-3j),
            (0.45, 0.45, 1.45 + 0.0j),
        ]:
            eps_c = n_cell**2

            def _f_at(px, py):
                vals = m_chk.fields(numpy.array([px]), numpy.array([py]))
                return numpy.array([v[0, 0] for v in vals])

            f0 = _f_at(x0, y0)
            dfdx = (_f_at(x0 + h, y0) - _f_at(x0 - h, y0)) / (2.0 * h)
            dfdy = (_f_at(x0, y0 + h) - _f_at(x0, y0 - h)) / (2.0 * h)
            dfdz = -1j * kz * f0
            E_v, cB_v = f0[:3], f0[3:]
            curlE = numpy.array(
                [dfdy[2] - dfdz[1], dfdz[0] - dfdx[2], dfdx[1] - dfdy[0]]
            )
            curlB = numpy.array(
                [dfdy[5] - dfdz[4], dfdz[3] - dfdx[5], dfdx[4] - dfdy[3]]
            )
            divD = eps_c * (dfdx[0] + dfdy[1] + dfdz[2])
            divB = dfdx[3] + dfdy[4] + dfdz[5]
            self.assertLess(
                numpy.linalg.norm(curlE + 1j * k0 * cB_v)
                / numpy.linalg.norm(k0 * cB_v),
                1e-9,
            )
            self.assertLess(
                numpy.linalg.norm(curlB - 1j * k0 * eps_c * E_v)
                / numpy.linalg.norm(k0 * eps_c * E_v),
                1e-9,
            )
            self.assertLess(abs(divD) / numpy.linalg.norm(k0 * eps_c * E_v), 1e-9)
            self.assertLess(abs(divB) / numpy.linalg.norm(k0 * cB_v), 1e-9)

        # Independent Proof 3: Cross-solver agreement against VFDModeSolver on an interface-aligned grid.
        xfd = numpy.unique(
            numpy.r_[
                numpy.linspace(0.0, 1.0, 41),
                numpy.linspace(1.0, 1.5, 26),
                numpy.linspace(1.5, 2.5, 41),
            ]
        )
        yfd = numpy.unique(
            numpy.r_[
                numpy.linspace(0.0, 1.0, 41),
                numpy.linspace(1.0, 1.22, 16),
                numpy.linspace(1.22, 2.22, 41),
            ]
        )

        def _eps_vfd(x_, y_):
            return numpy.where(
                (x_[:, None] > 1.0)
                & (x_[:, None] < 1.5)
                & (y_[None, :] > 1.0)
                & (y_[None, :] < 1.22),
                3.45**2,
                1.45**2,
            )

        vfd_sol = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xfd, yfd, _eps_vfd, "0000"
        ).solve(2, 1e-10)
        self.assertAlmostEqual(
            s0.modes[0].TEfrac(), vfd_sol.modes[0].TEfrac(), delta=3e-2
        )
        self.assertAlmostEqual(
            s0.modes[1].TEfrac(), vfd_sol.modes[1].TEfrac(), delta=3e-2
        )
        xc_fd = EMpy.utils.centered1d(xfd)
        yc_fd = EMpy.utils.centered1d(yfd)
        for im in range(2):
            cross_solver_ov = abs(s0.modes[im].overlap(vfd_sol.modes[im], xc_fd, yc_fd))
            self.assertGreater(cross_solver_ov, 0.95)
            self.assertLessEqual(cross_solver_ov, 1.01)


class GeometriesTest(TestCase):

    def test_s2t_t2s_roundtrip_and_generic_device(self):
        rng = numpy.random.default_rng(0)
        s = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        t = EMpy.modesolvers.geometries.S2T(s)
        s_rt = EMpy.modesolvers.geometries.T2S(t)
        numpy.testing.assert_allclose(s_rt, s)

        # GenericDevice whose first element has no .solver attribute (like SimpleJoint)
        class _DummyDevice:
            def __init__(self, mat):
                self.T = mat
                self.inputLHS = numpy.array([1.0, 0.0])
                self.inputRHS = numpy.array([0.0, 0.0])

        d1 = _DummyDevice(numpy.eye(4, dtype=complex))
        d2 = _DummyDevice(numpy.diag([1.0, 1.0, 1.0, 1.0]).astype(complex))
        gd = EMpy.modesolvers.geometries.GenericDevice([d1, d2])
        numpy.testing.assert_allclose(gd.T, numpy.eye(4, dtype=complex))
        numpy.testing.assert_allclose(gd.outputRHS, numpy.array([1.0, 0.0]))

    def test_simple_joint_exact_fresnel_te10_lossless_and_lossy(self):
        """Verify SimpleJoint reproduces exact complex TE10 reflection/transmission to ~1e-14."""
        wl = 1.55e-6
        x = numpy.linspace(0.0, 2.0e-6, 31)
        y = numpy.linspace(0.0, 1.3e-6, 21)

        for n1, n2 in [(1.5, 2.4), (1.5, 2.4 - 0.5j), (2.1 - 0.1j, 1.4 - 0.3j)]:
            s1 = EMpy.modesolvers.FD.VFDModeSolver(
                wl,
                x,
                y,
                lambda x_, y_, n=n1: numpy.full((len(x_), len(y_)), n**2),
                "SSAA",
            ).solve(1, 1e-12)
            s2 = EMpy.modesolvers.FD.VFDModeSolver(
                wl,
                x,
                y,
                lambda x_, y_, n=n2: numpy.full((len(x_), len(y_)), n**2),
                "SSAA",
            ).solve(1, 1e-12)
            sj = EMpy.modesolvers.geometries.SimpleJoint(s1, s2)
            ne1, ne2 = s1.modes[0].neff, s2.modes[0].neff
            r_exact = (ne1 - ne2) / (ne1 + ne2)
            numpy.testing.assert_allclose(sj.S[0, 0], r_exact, rtol=1e-12, atol=1e-13)
            numpy.testing.assert_allclose(sj.S[1, 1], -r_exact, rtol=1e-12, atol=1e-13)
            if numpy.isreal(n1) and numpy.isreal(n2):
                t_exact = 2.0 * numpy.sqrt(ne1 * ne2) / (ne1 + ne2)
                numpy.testing.assert_allclose(
                    abs(sj.S[1, 0]), abs(t_exact), rtol=1e-12, atol=1e-13
                )
                numpy.testing.assert_allclose(
                    sj.S.conj().T @ sj.S, numpy.eye(2), atol=1e-13
                )

    def test_simple_joint_passivity_reciprocity_and_degenerate_limits(self):
        """Verify SimpleJoint passivity, reciprocity, swap symmetry, and aniso/lossy -> iso/lossless limits."""
        wl = 1.55e-6
        xg = numpy.linspace(0.0, 3.0e-6, 41)
        yg = numpy.linspace(0.0, 2.0e-6, 31)

        def _eps_iso(w, kap=0.0):
            def _f(x_, y_):
                nc = complex(3.45, -kap) if kap > 0 else 3.45
                ncl = complex(1.45, -0.2 * kap) if kap > 0 else 1.45
                return numpy.where(
                    (numpy.abs(x_[:, None] - 1.5e-6) <= w / 2.0)
                    & (numpy.abs(y_[None, :] - 1.0e-6) <= 0.11e-6),
                    nc**2,
                    ncl**2,
                )

            return _f

        def _eps_aniso(w, dn=0.0, phi_deg=33.0):
            ph = EMpy.utils.deg2rad(phi_deg)

            def _f(x_, y_):
                core = (numpy.abs(x_[:, None] - 1.5e-6) <= w / 2.0) & (
                    numpy.abs(y_[None, :] - 1.0e-6) <= 0.11e-6
                )
                nO = numpy.where(core, 3.45, 1.45)
                nE = numpy.where(core, 3.45 + dn, 1.45 + dn)
                exx = nE**2 * numpy.cos(ph) ** 2 + nO**2 * numpy.sin(ph) ** 2
                eyy = nE**2 * numpy.sin(ph) ** 2 + nO**2 * numpy.cos(ph) ** 2
                exy = (nE**2 - nO**2) * numpy.sin(ph) * numpy.cos(ph)
                ezz = nO**2
                return numpy.stack([exx, exy, exy, eyy, ezz], axis=-1)

            return _f

        sn = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xg, yg, _eps_iso(0.4e-6), "0000"
        ).solve(1, 1e-10)
        sw = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xg, yg, _eps_iso(1.6e-6), "0000"
        ).solve(1, 1e-10)

        # Identical waveguide butt-joint must have zero reflection and unit transmission
        sj_same = EMpy.modesolvers.geometries.SimpleJoint(sn, sn)
        numpy.testing.assert_allclose(
            numpy.abs(sj_same.S), numpy.array([[0.0, 1.0], [1.0, 0.0]]), atol=1e-12
        )

        # Step discontinuity: strict passivity (||S||_2 < 1 due to radiation loss), reciprocity, swap symmetry
        sj_nw = EMpy.modesolvers.geometries.SimpleJoint(sn, sw)
        sj_wn = EMpy.modesolvers.geometries.SimpleJoint(sw, sn)
        sv_nw = numpy.linalg.svd(sj_nw.S, compute_uv=False)
        self.assertTrue(numpy.all(sv_nw < 1.0))
        self.assertLess(abs(sj_nw.S[0, 0]) ** 2 + abs(sj_nw.S[1, 0]) ** 2, 1.0)
        self.assertLess(abs(sj_nw.S[1, 1]) ** 2 + abs(sj_nw.S[0, 1]) ** 2, 1.0)
        self.assertAlmostEqual(abs(sj_nw.S[0, 1]), abs(sj_nw.S[1, 0]), places=12)
        numpy.testing.assert_allclose(
            numpy.abs(sj_nw.S),
            numpy.abs(sj_wn.S[[1, 0], :][:, [1, 0]]),
            atol=1e-12,
        )

        # Multi-mode junction with decoupled symmetry modes must remain passive and non-singular
        sn2 = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xg, yg, _eps_iso(0.4e-6), "0000"
        ).solve(2, 1e-10)
        sw2 = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xg, yg, _eps_iso(1.6e-6), "0000"
        ).solve(2, 1e-10)
        sj2 = EMpy.modesolvers.geometries.SimpleJoint(sn2, sw2)
        self.assertTrue(numpy.all(numpy.linalg.svd(sj2.S, compute_uv=False) <= 1.0))

        # Anisotropic 5-component rotated tensor -> isotropic limit (dn = 1e-4, 1e-8, 0)
        aniso_errs = []
        for dn in [1e-4, 1e-8, 0.0]:
            sna = EMpy.modesolvers.FD.VFDModeSolver(
                wl, xg, yg, _eps_aniso(0.4e-6, dn), "0000"
            ).solve(1, 1e-10)
            swa = EMpy.modesolvers.FD.VFDModeSolver(
                wl, xg, yg, _eps_aniso(1.6e-6, dn), "0000"
            ).solve(1, 1e-10)
            sja = EMpy.modesolvers.geometries.SimpleJoint(sna, swa)
            self.assertTrue(numpy.all(numpy.linalg.svd(sja.S, compute_uv=False) <= 1.0))
            self.assertAlmostEqual(abs(sja.S[0, 1]), abs(sja.S[1, 0]), places=11)
            aniso_errs.append(
                float(numpy.max(numpy.abs(numpy.abs(sja.S) - numpy.abs(sj_nw.S))))
            )
        self.assertLess(aniso_errs[1], aniso_errs[0])
        self.assertLess(aniso_errs[2], 1e-10)

        # Lossy -> non-lossy limit (kappa = 1e-2, 1e-5, 0.0+0j -> lossless)
        lossy_errs = []
        for kap in [1e-2, 1e-5, 0.0]:
            snk = EMpy.modesolvers.FD.VFDModeSolver(
                wl, xg, yg, _eps_iso(0.4e-6, kap), "0000"
            ).solve(1, 1e-10)
            swk = EMpy.modesolvers.FD.VFDModeSolver(
                wl, xg, yg, _eps_iso(1.6e-6, kap), "0000"
            ).solve(1, 1e-10)
            sjk = EMpy.modesolvers.geometries.SimpleJoint(snk, swk)
            self.assertTrue(numpy.all(numpy.linalg.svd(sjk.S, compute_uv=False) <= 1.0))
            self.assertAlmostEqual(abs(sjk.S[0, 1]), abs(sjk.S[1, 0]), delta=1e-5)
            lossy_errs.append(
                float(numpy.max(numpy.abs(numpy.abs(sjk.S) - numpy.abs(sj_nw.S))))
            )
        self.assertLess(lossy_errs[1], lossy_errs[0])
        self.assertLess(lossy_errs[2], 1e-10)
