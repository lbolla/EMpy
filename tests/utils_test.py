from unittest import TestCase

import math
import random

import matplotlib
import numpy

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

import EMpy.materials  # noqa: E402
import EMpy.utils as U  # noqa: E402


class UtilsTest(TestCase):

    def test_deg2rad(self):
        x = -2 * numpy.pi + 4 * numpy.pi * random.random()
        self.assertAlmostEqual(x, U.deg2rad(U.rad2deg(x)))

    def test_norm(self):
        self.assertEqual(U.norm([1, 0, 0]), 1)
        self.assertEqual(U.norm([0, 1, 0]), 1)
        self.assertEqual(U.norm([0, 0, 1]), 1)
        self.assertAlmostEqual(U.norm([1, 0, 1]), math.sqrt(2))

    def test_trapz2(self):
        f = [[i + j for i in range(5)] for j in range(5)]
        res = U.trapz2(f, dx=1.0, dy=1.0)
        # integral of f = x + y over [0,4]x[0,4] = 64
        self.assertAlmostEqual(res, 64.0)

    def test_layer_fourier_coeffs_isotropic(self):
        layer = U.Layer(EMpy.materials.SiO2, 1e-6)
        wl = 1.55e-6
        n = EMpy.materials.SiO2.n(wl).item()
        eps, eps1 = layer.getEPSFourierCoeffs(wl, 2, anisotropic=False)
        self.assertEqual(eps.shape, (9,))
        self.assertAlmostEqual(eps[4], n**2)
        self.assertAlmostEqual(eps1[4], n**-2)

    def test_cross_section_epsfunc_and_plot(self):
        s1 = U.Slice(1.0, [U.Layer(EMpy.materials.SiO2, 1.0)])
        s2 = U.Slice(
            1.0,
            [
                U.Layer(EMpy.materials.SiO2, 0.5),
                U.Layer(EMpy.materials.Si, 0.5),
            ],
        )
        cs = U.CrossSection([s1, s2])
        wl = 1.55e-6
        x = numpy.array([0.5, 1.5])
        y = numpy.array([0.25, 0.75])
        eps = cs.epsfunc(x, y, wl)
        n_sio2 = EMpy.materials.SiO2.n(wl).item()
        n_si = EMpy.materials.Si.n(wl).item()
        numpy.testing.assert_allclose(
            eps,
            numpy.array([[n_sio2**2, n_sio2**2], [n_sio2**2, n_si**2]]),
        )
        plt.figure()
        cs.plot(wl=wl)
        plt.close("all")

    def test_progress_bar(self):
        pb = U.ProgressBar(0, 100, 40)
        self.assertIn("0%", pb.progBar)
        pb.updateAmount(50)
        self.assertIn("50%", pb.progBar)
        pb.updateAmount(100)
        self.assertIn("100%", pb.progBar)

    def test_find_peaks(self):
        x = numpy.linspace(0.0, 10.0, 1001)
        y = 1.0 * numpy.exp(-((x - 3.0) ** 2) / 0.2) + 2.0 * numpy.exp(
            -((x - 7.0) ** 2) / 0.2
        )
        peaks = U.find_peaks(x, y)
        self.assertEqual(len(peaks), 2)
        # Sorted descending by peak height
        self.assertAlmostEqual(peaks[0].x, 7.0, places=4)
        self.assertAlmostEqual(peaks[0].y, 2.0, places=4)
        self.assertAlmostEqual(peaks[1].x, 3.0, places=4)
        self.assertAlmostEqual(peaks[1].y, 1.0, places=4)

        # Monotonic curve falls back to boundary max
        mono_peaks = U.find_peaks(x, x)
        self.assertEqual(len(mono_peaks), 1)
        self.assertAlmostEqual(mono_peaks[0].x, 10.0)

    def test_multilayer_slicing(self):
        l1 = U.Layer(EMpy.materials.Air, 1.0)
        l2 = U.Layer(EMpy.materials.SiO2, 2.0)
        l3 = U.Layer(EMpy.materials.Si, 3.0)
        m = U.Multilayer([l1, l2, l3])

        sub = m[1:3]
        self.assertIsInstance(sub, U.Multilayer)
        self.assertEqual(len(sub), 2)
        self.assertTrue(sub.isIsotropic())
        self.assertIs(m[0], l1)

        m[0:2] = [l3, l1]
        self.assertIs(m[0], l3)
        self.assertIs(m[1], l1)

    def test_liquid_crystal_cell_nlayers_guard(self):
        lc = EMpy.materials.LiquidCrystal()
        with self.assertRaises(ValueError):
            U.LiquidCrystalCell(
                lc, voltage=0.0, t_tot=10e-6, t_anchoring=1e-6, nlayers=2
            )
        cell = U.LiquidCrystalCell(
            lc, voltage=0.0, t_tot=10e-6, t_anchoring=1e-6, nlayers=5
        )
        m = cell.createMultilayer()
        self.assertIsInstance(m, U.Multilayer)
        self.assertEqual(len(m), 5)
        with self.assertRaises(ImportError):
            U.LiquidCrystalCell(
                lc, voltage=1.0, t_tot=10e-6, t_anchoring=1e-6, nlayers=5
            )

    def test_symmetric_double_grating_fourier_coeffs(self):
        import scipy.integrate

        n1, n2, n3 = 1.5, 2.5, 1.0
        m1 = EMpy.materials.IsotropicMaterial(
            "m1", n0=EMpy.materials.RefractiveIndex(n0_const=n1)
        )
        m2 = EMpy.materials.IsotropicMaterial(
            "m2", n0=EMpy.materials.RefractiveIndex(n0_const=n2)
        )
        m3 = EMpy.materials.IsotropicMaterial(
            "m3", n0=EMpy.materials.RefractiveIndex(n0_const=n3)
        )
        dc1, dc2 = 0.2, 0.3
        dcM = (1.0 - dc1 - dc2) / 2.0
        wl = 1.55e-6
        n_orders = 3

        sdg = U.SymmetricDoubleGrating(m1, m2, m3, dc1, dc2, 1.0e-6, 0.2e-6)
        adg = U.AsymmetricDoubleGrating(m1, m2, m3, dc1, dc2, dcM, 1.0e-6, 0.2e-6)

        eps_s, eps1_s = sdg.getEPSFourierCoeffs(wl, n_orders, anisotropic=False)
        eps_a, eps1_a = adg.getEPSFourierCoeffs(wl, n_orders, anisotropic=False)
        numpy.testing.assert_allclose(eps_s, eps_a)
        numpy.testing.assert_allclose(eps1_s, eps1_a)

        # Direct numerical integration of piecewise-constant eps(x) on [-0.5, 0.5]
        def eps_of_x(x):
            if -dc1 - dcM / 2.0 <= x <= -dcM / 2.0:
                return n1**2
            if dcM / 2.0 <= x <= dc2 + dcM / 2.0:
                return n2**2
            return n3**2

        hmax = 2 * n_orders
        h = numpy.arange(-hmax, hmax + 1)
        pts = [-dc1 - dcM / 2.0, -dcM / 2.0, dcM / 2.0, dc2 + dcM / 2.0]
        for idx, hh in enumerate(h):
            re_val, _ = scipy.integrate.quad(
                lambda x: eps_of_x(x) * numpy.cos(2 * numpy.pi * hh * x),
                -0.5,
                0.5,
                points=pts,
            )
            im_val, _ = scipy.integrate.quad(
                lambda x: eps_of_x(x) * numpy.sin(2 * numpy.pi * hh * x),
                -0.5,
                0.5,
                points=pts,
            )
            numpy.testing.assert_allclose(eps_s[idx], re_val + 1j * im_val, atol=1e-10)

        # Identical ridges with symmetric spacers must have zero odd harmonics
        sdg_eq = U.SymmetricDoubleGrating(m1, m1, m3, 0.25, 0.25, 1.0e-6, 0.2e-6)
        eps_eq, _ = sdg_eq.getEPSFourierCoeffs(wl, n_orders, anisotropic=True)
        for idx, hh in enumerate(h):
            if hh % 2 != 0:
                numpy.testing.assert_allclose(eps_eq[:, :, idx], 0.0, atol=1e-14)

    def test_rix_loss_roundtrip(self):
        wl = 1.55e-6
        n_real = 3.48

        n_m = U.loss_m2rix(n_real, 200.0, wl)
        nr, ni, alpha, alpha_cm1, alpha_dBcm1 = U.rix2losses(n_m, wl)
        self.assertAlmostEqual(nr, n_real)
        self.assertLess(ni, 0.0)
        self.assertAlmostEqual(alpha, 200.0)
        self.assertAlmostEqual(alpha_cm1, 2.0)

        n_cm = U.loss_cm2rix(n_real, 10.0, wl)
        _, _, _, alpha_cm1_rt, _ = U.rix2losses(n_cm, wl)
        self.assertAlmostEqual(alpha_cm1_rt, 10.0)

        n_db = U.loss_dBcm2rix(n_real, 3.0, wl)
        _, _, _, _, alpha_dBcm1_rt = U.rix2losses(n_db, wl)
        self.assertAlmostEqual(alpha_dBcm1_rt, 3.0)
