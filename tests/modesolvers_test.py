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
        self.assertAlmostEqual(
            EMpy.modesolvers.interface.overlap(m0, m0, None, None), 1.0, places=5
        )
        O11, _, _, _ = EMpy.modesolvers.interface.interface_matrix(simul, simul)
        self.assertEqual(O11.shape, (2, 2))
        te_frac = m0.TEfrac_old(
            numpy.linspace(Ux[0], Ux[-1], 11), numpy.linspace(Uy[0], Uy[-1], 11)
        )
        self.assertTrue(0.0 <= te_frac <= 1.0)
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
