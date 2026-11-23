from unittest import TestCase

from numpy import array
from numpy.testing import assert_almost_equal, assert_raises

import EMpy.materials as mat


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
        """ 6-coeffs:
            n(wls) =  1. +
            B1 * wls ** 2 / (wls ** 2 - C1) +
            B2 * wls ** 2 / (wls ** 2 - C2) +
            B3 * wls ** 2 / (wls ** 2 - C3)
        """
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
        with assert_raises(ValueError):
            mat.RefractiveIndex(n0_known={})

        test_rix = 1.50
        test_wl = 1.0
        n0_known = {test_wl: test_rix}
        a = mat.RefractiveIndex(n0_known=n0_known)
        n0_known[2.0] = 9.9
        self.assertEqual(a.get_rix(test_wl)[0], array([test_rix]))
        self.assertEqual(a.get_rix(1.7)[0], array([test_rix]))

        # Linear interpolation + endpoint extrapolation (lossless and lossy -> lossless limit)
        r_ll = mat.RefractiveIndex(n0_known={1.5e-6: 2.0, 1.6e-6: 2.2})
        assert_almost_equal(
            r_ll.get_rix([1.45e-6, 1.55e-6, 1.65e-6]), array([1.9, 2.1, 2.3])
        )
        errs = []
        for kap in (1e-2, 1e-5, 0.0):
            rk = mat.RefractiveIndex(
                n0_known={1.5e-6: complex(2.0, -kap), 1.6e-6: complex(2.2, -2 * kap)}
            )
            vals = rk.get_rix([1.45e-6, 1.55e-6, 1.65e-6])
            if kap > 0:
                self.assertTrue((vals.imag < 0).all())
            errs.append(
                float(abs(vals - r_ll.get_rix([1.45e-6, 1.55e-6, 1.65e-6])).max())
            )
        self.assertLess(errs[1], errs[0])
        self.assertLess(errs[2], 1e-15)

        # Multi-dimensional wls shape preservation (1-point and multi-point n0_known)
        wls_2d = array([[1.45e-6, 1.55e-6], [1.60e-6, 1.65e-6]])
        self.assertEqual(a.get_rix(wls_2d).shape, (2, 2))
        assert_almost_equal(r_ll.get_rix(wls_2d), array([[1.9, 2.1], [2.2, 2.3]]))


class EpsilonTensorTest(TestCase):

    def test_multi_wavelength_known(self):
        import numpy as np

        eps_const = 2.0 * np.eye(3)
        eps_wl1 = (3.0 - 0.1j) * np.eye(3)
        et = mat.EpsilonTensor(
            epsilon_tensor_const=eps_const,
            epsilon_tensor_known={1.55e-6: eps_wl1},
        )
        res = et([1.50e-6, 1.55e-6, 1.60e-6])
        self.assertEqual(res.shape, (3, 3, 3))
        assert_almost_equal(res[:, :, 0], eps_const)
        assert_almost_equal(res[:, :, 1], eps_wl1)
        assert_almost_equal(res[:, :, 2], eps_const)

    def test_no_shared_mutable_defaults(self):
        import EMpy.scattering

        m1 = mat.AnisotropicMaterial("a")
        m2 = mat.AnisotropicMaterial("b")
        m1.epsilonTensor.epsilon_tensor_known[1.55e-6] = 4.0 * array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )
        self.assertEqual(m2.epsilonTensor.epsilon_tensor_known, {})

        f1 = EMpy.scattering.Field()
        f2 = EMpy.scattering.Field()
        f1.E[0, 0] = 5.0
        self.assertEqual(f2.E[0, 0], 0.0)

    def test_sio2_and_si_thermo_optic_coefficients(self):
        wl = 1.55e-6
        t0 = mat.SiO2.toc.T0
        dn_dt_sio2 = float(
            ((mat.SiO2.n(wl, t0 + 10.0) - mat.SiO2.n(wl, t0)) / 10.0).item()
        )
        self.assertAlmostEqual(dn_dt_sio2, 1.0e-5, places=10)
        dn_dt_si = float(((mat.Si.n(wl, t0 + 10.0) - mat.Si.n(wl, t0)) / 10.0).item())
        self.assertAlmostEqual(dn_dt_si, 1.86e-4, delta=1e-5)
