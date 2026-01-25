from unittest import TestCase

import random
import math

from numpy import pi

import EMpy.utils as U


class UtilsTest(TestCase):

    def test_deg2rad(self):
        x = -2 * pi + 4 * pi * random.random()
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