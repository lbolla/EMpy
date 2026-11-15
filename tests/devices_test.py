from unittest import TestCase

import numpy

import EMpy.devices
import EMpy.materials
import EMpy.transfer_matrix
import EMpy.utils


class NRRTest(TestCase):

    def test_solve(self):
        N = 1000
        wls = numpy.linspace(1.53e-6, 1.57e-6, N)

        Ks = [
            EMpy.devices.Coupler(wls, numpy.sqrt(0.08), 1.0),
            EMpy.devices.Coupler(wls, numpy.sqrt(0.008), 1.0),
            EMpy.devices.Coupler(wls, numpy.sqrt(0.006), 1.0),
            EMpy.devices.Coupler(wls, numpy.sqrt(0.09), 1.0),
        ]

        R = 5e-6
        l1s = [numpy.pi * R, numpy.pi * R, numpy.pi * R]
        l2s = [numpy.pi * R, numpy.pi * R, numpy.pi * R]

        SWG = EMpy.devices.SWG(400, 220, 125).solve(wls)
        neffs = [SWG.neff, SWG.neff, SWG.neff]

        NRR = EMpy.devices.NRR(Ks, neffs, l1s, l2s).solve()

        self.assertEqual(NRR.THRU.shape, (N,))
        self.assertAlmostEqual(numpy.absolute(NRR.THRU).min(), 0.3868920546379094)


class SWGTest(TestCase):

    def test_tabulated_points(self):
        wls = numpy.linspace(1.53e-6, 1.57e-6, 5)
        expected = {
            (400, 25): EMpy.devices.SWG.pf400_25,
            (400, 125): EMpy.devices.SWG.pf400_125,
            (400, 225): EMpy.devices.SWG.pf400_225,
            (488, 25): EMpy.devices.SWG.pf488_25,
            (488, 125): EMpy.devices.SWG.pf488_125,
            (488, 225): EMpy.devices.SWG.pf488_225,
        }
        for (w, T), pf in expected.items():
            swg = EMpy.devices.SWG(w, 220, T).solve(wls)
            numpy.testing.assert_allclose(swg.neff, numpy.polyval(pf, wls * 1e9))

    def test_interpolation_and_bounds(self):
        wls = numpy.linspace(1.53e-6, 1.57e-6, 10)
        swg_mid = EMpy.devices.SWG(444, 220, 75).solve(wls)
        pf_mid = 0.25 * (
            numpy.array(EMpy.devices.SWG.pf400_25)
            + numpy.array(EMpy.devices.SWG.pf400_125)
            + numpy.array(EMpy.devices.SWG.pf488_25)
            + numpy.array(EMpy.devices.SWG.pf488_125)
        )
        numpy.testing.assert_allclose(swg_mid.neff, numpy.polyval(pf_mid, wls * 1e9))
        self.assertEqual(swg_mid.disp.shape, wls.shape)
        self.assertEqual(swg_mid.ng.shape, wls.shape)

        with self.assertRaises(ValueError):
            EMpy.devices.SWG(399, 220, 25).solve(wls)
        with self.assertRaises(ValueError):
            EMpy.devices.SWG(440, 220, 230).solve(wls)


class EtalonTest(TestCase):

    def test_solve_and_matches_transfer_matrix(self):
        n_slab = 2.5
        thickness = 10e-6
        mat = EMpy.materials.IsotropicMaterial(
            "slab", n0=EMpy.materials.RefractiveIndex(n0_const=n_slab)
        )
        layer = EMpy.utils.Layer(mat, thickness)
        R = ((n_slab - 1.0) / (n_slab + 1.0)) ** 2
        wls = numpy.linspace(1.54e-6, 1.56e-6, 201)

        etalon = EMpy.devices.Etalon(layer, R, 0.0).solve(wls)
        multilayer = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(EMpy.materials.Air, numpy.inf),
                layer,
                EMpy.utils.Layer(EMpy.materials.Air, numpy.inf),
            ]
        )
        tmm = EMpy.transfer_matrix.IsotropicTransferMatrix(multilayer, 0.0).solve(wls)

        numpy.testing.assert_allclose(etalon.Te, tmm.Ts, rtol=1e-12, atol=1e-12)
        numpy.testing.assert_allclose(etalon.FWHMwl, etalon.FSRwl / etalon.FINESSE)
        numpy.testing.assert_allclose(etalon.FWHM, etalon.FSR / etalon.FINESSE)

        # Scalar wavelength input
        etalon_scalar = EMpy.devices.Etalon(layer, 0.9, 0.0).solve(1.55e-6)
        self.assertEqual(etalon_scalar.Te.shape, (1,))
        self.assertEqual(etalon_scalar.FWHMwl.shape, (1,))
