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
        self.assertEqual(NRR.DROP.shape, (N,))
        self.assertAlmostEqual(numpy.absolute(NRR.THRU).min(), 0.3868920546379094)
        numpy.testing.assert_allclose(
            numpy.abs(NRR.THRU) ** 2 + numpy.abs(NRR.DROP) ** 2, 1.0, atol=1e-12
        )

    def test_nrr_matches_srr_and_obeys_lossy_limits(self):
        """Verify N=1 NRR matches exact SRR (amplitude & phase) and N-ring passivity / kappa -> 0+ limits."""
        wls = numpy.linspace(1.545e-6, 1.555e-6, 51)
        c1 = EMpy.devices.Coupler(
            wls, numpy.full_like(wls, numpy.sqrt(0.3)), numpy.full_like(wls, 0.97)
        )
        c2 = EMpy.devices.Coupler(
            wls, numpy.full_like(wls, numpy.sqrt(0.5)), numpy.full_like(wls, 0.91)
        )
        neff_lossy = (2.45 - 2e-3j) * numpy.ones_like(wls)
        l1, l2 = 11.0e-6, 19.0e-6

        srr = EMpy.devices.SRR(c1, c2, neff_lossy, l1, l2).solve()
        nrr1 = EMpy.devices.NRR([c1, c2], [neff_lossy], [l1], [l2]).solve()
        numpy.testing.assert_allclose(nrr1.THRU, srr.THRU, rtol=1e-13, atol=1e-14)
        numpy.testing.assert_allclose(nrr1.DROP, srr.DROP, rtol=1e-13, atol=1e-14)
        self.assertTrue(
            numpy.all(numpy.abs(nrr1.THRU) ** 2 + numpy.abs(nrr1.DROP) ** 2 < 1.0)
        )

        # Scalar wavelength support in NRR
        c1_s = EMpy.devices.Coupler(1.55e-6, numpy.sqrt(0.3), 0.97)
        c2_s = EMpy.devices.Coupler(1.55e-6, numpy.sqrt(0.5), 0.91)
        srr_s = EMpy.devices.SRR(c1_s, c2_s, 2.45 - 2e-3j, l1, l2).solve()
        nrr_s = EMpy.devices.NRR([c1_s, c2_s], [2.45 - 2e-3j], [l1], [l2]).solve()
        self.assertAlmostEqual(nrr_s.THRU, srr_s.THRU, places=14)
        self.assertAlmostEqual(nrr_s.DROP, srr_s.DROP, places=14)

        # Asymmetric lossy arcs (l1 = 0, l2 = 100 um with strong absorption): DROP must be strongly attenuated
        c_hi1 = EMpy.devices.Coupler(
            wls, numpy.full_like(wls, 0.8), numpy.ones_like(wls)
        )
        c_hi2 = EMpy.devices.Coupler(
            wls, numpy.full_like(wls, 0.9), numpy.ones_like(wls)
        )
        nrr_asym = EMpy.devices.NRR(
            [c_hi1, c_hi2], [(2.45 - 0.05j) * numpy.ones_like(wls)], [0.0], [100.0e-6]
        ).solve()
        self.assertTrue(numpy.all(numpy.abs(nrr_asym.DROP) < 1e-5))
        self.assertTrue(
            numpy.all(
                numpy.abs(nrr_asym.THRU) ** 2 + numpy.abs(nrr_asym.DROP) ** 2 < 1.0
            )
        )

        # 3-ring NRR: lossless unitarity, positive DROP group delay, and passive kappa -> 0+ convergence
        ks3 = [
            EMpy.devices.Coupler(
                wls, numpy.full_like(wls, numpy.sqrt(kv)), numpy.ones_like(wls)
            )
            for kv in (0.25, 0.06, 0.08, 0.30)
        ]
        l1s_3 = [10.0e-6, 13.0e-6, 11.5e-6]
        l2s_3 = [14.0e-6, 11.0e-6, 15.5e-6]

        def _solve_nrr3(kap, force_complex=False):
            ns = [
                (complex(nr, -kap) if (kap > 0 or force_complex) else nr)
                * numpy.ones_like(wls)
                for nr in (2.42, 2.45, 2.43)
            ]
            return EMpy.devices.NRR(ks3, ns, l1s_3, l2s_3).solve()

        nrr3_ll = _solve_nrr3(0.0, force_complex=False)
        nrr3_c0 = _solve_nrr3(0.0, force_complex=True)
        numpy.testing.assert_allclose(
            numpy.abs(nrr3_ll.THRU) ** 2 + numpy.abs(nrr3_ll.DROP) ** 2,
            1.0,
            atol=1e-13,
        )
        numpy.testing.assert_allclose(nrr3_c0.THRU, nrr3_ll.THRU, atol=1e-14)
        numpy.testing.assert_allclose(nrr3_c0.DROP, nrr3_ll.DROP, atol=1e-14)
        _, tau_drop, _ = EMpy.utils.group_delay_and_dispersion(wls, nrr3_ll.DROP)
        self.assertTrue(numpy.all(tau_drop > 0.0))

        errs = []
        for kap in (1e-2, 1e-5):
            nrr3_k = _solve_nrr3(kap)
            pwr_k = numpy.abs(nrr3_k.THRU) ** 2 + numpy.abs(nrr3_k.DROP) ** 2
            self.assertTrue(numpy.all(pwr_k < 1.0))
            errs.append(float(numpy.max(numpy.abs(nrr3_k.DROP - nrr3_ll.DROP))))
        self.assertLess(errs[1], errs[0])
        self.assertLess(errs[1], 1e-2)


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

    def test_dispersive_group_index_oblique_angle_and_lossy_limits(self):
        """Verify Etalon uses group index ng in FSRwl, internal refraction angle theta, and passive lossy limits."""
        # 1. Dispersive Silicon etalon: peak-to-peak FSR of Te must match group-index FSRwl (ng ~ 3.60 != n ~ 3.48)
        wls_si = numpy.linspace(1.54e-6, 1.56e-6, 4001)
        si_layer = EMpy.utils.Layer(EMpy.materials.Si, 50.0e-6)
        et_si = EMpy.devices.Etalon(si_layer, 0.85, 0.0).solve(wls_si)
        peaks = sorted(
            [p.x for p in EMpy.utils.find_peaks(wls_si, et_si.Te) if p.y > 0.99]
        )
        self.assertGreaterEqual(len(peaks), 2)
        p_mid = len(peaks) // 2
        measured_fsr = peaks[p_mid] - peaks[p_mid - 1]
        idx_mid = int(
            numpy.argmin(numpy.abs(wls_si - 0.5 * (peaks[p_mid] + peaks[p_mid - 1])))
        )
        self.assertAlmostEqual(
            et_si.FSRwl[idx_mid], measured_fsr, delta=0.005 * measured_fsr
        )
        # Scalar wavelength on dispersive Si must also use ng (~3.60), not phase index (~3.48)
        et_si_1 = EMpy.devices.Etalon(si_layer, 0.85, 0.0).solve(1.55e-6)
        self.assertAlmostEqual(
            float(et_si_1.FSRwl[0]),
            float(et_si.FSRwl[len(wls_si) // 2]),
            delta=1e-11,
        )

        # 2. Oblique internal refraction angle theta_int vs IsotropicTransferMatrix at external angle theta_ext
        n_slab = 2.4
        theta_ext = EMpy.utils.deg2rad(35.0)
        theta_int = float(numpy.arcsin(numpy.sin(theta_ext) / n_slab))
        r_s = (
            (numpy.cos(theta_ext) - n_slab * numpy.cos(theta_int))
            / (numpy.cos(theta_ext) + n_slab * numpy.cos(theta_int))
        ) ** 2
        wls = numpy.linspace(1.545e-6, 1.555e-6, 101)
        slab = EMpy.utils.Layer(
            EMpy.materials.IsotropicMaterial(
                "s", n0=EMpy.materials.RefractiveIndex(n0_const=n_slab)
            ),
            8.0e-6,
        )
        et_obl = EMpy.devices.Etalon(slab, r_s, theta_int).solve(wls)
        tmm_obl = EMpy.transfer_matrix.IsotropicTransferMatrix(
            EMpy.utils.Multilayer(
                [
                    EMpy.utils.Layer(EMpy.materials.Air, numpy.inf),
                    slab,
                    EMpy.utils.Layer(EMpy.materials.Air, numpy.inf),
                ]
            ),
            theta_ext,
        ).solve(wls)
        numpy.testing.assert_allclose(et_obl.Te, tmm_obl.Ts, rtol=1e-12, atol=1e-13)

        # 3. Passive lossy cavity -> lossless limit (kap = 1e-2, 1e-5, 0.0 + 0j)
        def _solve_lossy_et(kap, force_complex=False):
            n_val = complex(n_slab, -kap) if (kap > 0 or force_complex) else n_slab
            lyr = EMpy.utils.Layer(
                EMpy.materials.IsotropicMaterial(
                    "lk", n0=EMpy.materials.RefractiveIndex(n0_const=n_val)
                ),
                8.0e-6,
            )
            return EMpy.devices.Etalon(lyr, 0.7, theta_int).solve(wls)

        et_ll = _solve_lossy_et(0.0, force_complex=False)
        et_c0 = _solve_lossy_et(0.0, force_complex=True)
        self.assertFalse(numpy.iscomplexobj(et_c0.Te))
        numpy.testing.assert_allclose(et_c0.Te, et_ll.Te, rtol=1e-13, atol=1e-14)

        k_errs = []
        for kap in (1e-2, 1e-5):
            et_k = _solve_lossy_et(kap)
            self.assertFalse(numpy.iscomplexobj(et_k.Te))
            self.assertTrue(numpy.all((et_k.Te >= 0.0) & (et_k.Te < 1.0)))
            k_errs.append(float(numpy.max(numpy.abs(et_k.Te - et_ll.Te))))
        self.assertLess(k_errs[1], k_errs[0])
        self.assertLess(k_errs[1], 5e-3)


class TunableRingTest(TestCase):

    def test_t_tcrr_and_t_ctcrr_do_not_mutate_inputs(self):
        wls = numpy.linspace(1.53e-6, 1.57e-6, 5)
        neff = numpy.full_like(wls, 2.5)
        K_in = numpy.array([0.3, 0.5, 0.7])
        l_in = numpy.array([10e-6, 12e-6, 14e-6, 16e-6])
        K_orig = K_in.copy()
        l_orig = l_in.copy()

        tcrr = EMpy.devices.T_TCRR(neff, K_in, 1.0, l_in, coupling="optimum").solve(wls)
        numpy.testing.assert_array_equal(K_in, K_orig)
        numpy.testing.assert_array_equal(l_in, l_orig)
        numpy.testing.assert_array_equal(tcrr.K, K_orig)
        numpy.testing.assert_array_equal(tcrr.l, l_orig)
        numpy.testing.assert_allclose(
            numpy.abs(tcrr.THRU) ** 2 + numpy.abs(tcrr.DROP) ** 2, 1.0, atol=1e-12
        )

        ctcrr = EMpy.devices.T_CTCRR(neff, K_in, 1.0, l_in, coupling="optimum").solve(
            wls
        )
        numpy.testing.assert_array_equal(K_in, K_orig)
        numpy.testing.assert_array_equal(l_in, l_orig)
        numpy.testing.assert_array_equal(ctcrr.K, K_orig)
        numpy.testing.assert_array_equal(ctcrr.l, l_orig)
        numpy.testing.assert_allclose(
            numpy.abs(ctcrr.THRU) ** 2 + numpy.abs(ctcrr.DROP) ** 2, 1.0, atol=1e-12
        )
