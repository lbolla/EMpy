# pylint: disable=no-self-use
from unittest import TestCase

from numpy import array
from numpy.testing import assert_almost_equal, assert_raises

import EMpy_gpu.materials as mat


class RefractiveIndexTest(TestCase):

    def test_all_nones(self):
        with assert_raises(ValueError):
            mat.RefractiveIndex()

    def test_const(self):
        test_rix = 1.50
        a = mat.RefractiveIndex(n0_const=test_rix)
        self.assertEqual(a.get_rix(1.0)[0], array([test_rix]))

    def test_poly(self):
        test_poly = [1, 1]  # n(wl) = 1 * wl + 1
        test_rix = 2.0  # n(1) = 1 * 1 + 1 = 2
        a = mat.RefractiveIndex(n0_poly=test_poly)
        assert_almost_equal(a.get_rix(1.0)[0], array([test_rix]))

    def test_smcoeffs(self):
        test_poly = [1] * 6
        ''' 6-coeffs:
            n(wls) =  1. +
            B1 * wls ** 2 / (wls ** 2 - C1) +
            B2 * wls ** 2 / (wls ** 2 - C2) +
            B3 * wls ** 2 / (wls ** 2 - C3)
        '''
        test_rix = 1.0536712127723509e-08
        a = mat.RefractiveIndex(n0_smcoeffs=test_poly)
        assert_almost_equal(a.get_rix(0.5)[0], array([test_rix]))

    def test_func(self):
        test_rix = 1.50

        def test_func_const(x):
            # returns a const
            return 0.0 * x + test_rix

        a = mat.RefractiveIndex(n0_func=test_func_const)
        assert_almost_equal(a.get_rix([1.0, 1.5]), array([1.5, 1.5]))

        def test_func_var(x):
            # returns a const
            return 1.0 * x + test_rix

        b = mat.RefractiveIndex(n0_func=test_func_var)
        assert_almost_equal(b.get_rix([1.0, 1.5]), array([2.5, 3.0]))

    def test_known(self):
        test_rix = 1.50
        test_wl = 1.0
        n0_known = {
            test_wl: test_rix
        }
        a = mat.RefractiveIndex(n0_known=n0_known)
        self.assertEqual(a.get_rix(test_wl)[0], array([test_rix]))
