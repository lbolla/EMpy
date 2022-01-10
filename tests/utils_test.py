# pylint: disable=no-self-use
from unittest import TestCase

import random
import math

from numpy import pi
from numpy.testing import assert_almost_equal

import EMpy_gpu.utils as U


class UtilsTest(TestCase):

    def test_deg2rad(self):
        # TODO use hypothesis
        x = -2 * pi + 4 * pi * random.random()
        assert_almost_equal(x, U.deg2rad(U.rad2deg(x)))

    def test_norm(self):
        self.assertEqual(U.norm([1, 0, 0]), 1)
        self.assertEqual(U.norm([0, 1, 0]), 1)
        self.assertEqual(U.norm([0, 0, 1]), 1)
        assert_almost_equal(U.norm([1, 0, 1]), math.sqrt(2))
