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
