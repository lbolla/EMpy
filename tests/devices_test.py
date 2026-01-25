from unittest import TestCase

import numpy

import EMpy.devices


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