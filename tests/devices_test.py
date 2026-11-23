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
        # Closed-form lossless CROW recursion: r_m = (t_m - r_{m+1} * z) / (1 - t_m * r_{m+1} * z)
        t_vals = [numpy.sqrt(1.0 - k.K**2) for k in Ks]
        z_rt = numpy.exp(-1j * (2.0 * numpy.pi * SWG.neff / wls) * (2.0 * numpy.pi * R))
        thru_analytic = t_vals[-1].astype(complex)
        for t_m in reversed(t_vals[:-1]):
            thru_analytic = (t_m - thru_analytic * z_rt) / (
                1.0 - t_m * thru_analytic * z_rt
            )
        numpy.testing.assert_allclose(NRR.THRU, thru_analytic, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(numpy.absolute(NRR.THRU).min(), 0.10440851213414967)
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
        for kap in (1e-2, 1e-4, 1e-6):
            nrr3_k = _solve_nrr3(kap)
            pwr_k = numpy.abs(nrr3_k.THRU) ** 2 + numpy.abs(nrr3_k.DROP) ** 2
            self.assertTrue(numpy.all(pwr_k < 1.0))
            errs.append(float(numpy.max(numpy.abs(nrr3_k.DROP - nrr3_ll.DROP))))
        self.assertLess(errs[1], errs[0])
        self.assertLess(errs[2], errs[1])
        self.assertLess(errs[2], 1e-2)

    def test_nrr_matches_independent_crow_network(self):
        """Verify N=2 and N=3 NRR match an independent linear-system CROW solver."""

        def _solve_crow_direct(wls, k_vals, q_vals, neffs, l1s, l2s):
            n_rings = len(neffs)
            thru = numpy.zeros(len(wls), dtype=complex)
            drop = numpy.zeros(len(wls), dtype=complex)
            r_vals = [numpy.sqrt(1.0 - k**2) for k in k_vals]
            # Unknowns per ring m: b2_m (index 2*m) and b1_{m+1} (index 2*m + 1)
            for iwl, wl in enumerate(wls):
                p1 = [
                    numpy.exp(-1j * 2.0 * numpy.pi * n[iwl] / wl * l1)
                    for n, l1 in zip(neffs, l1s, strict=True)
                ]
                p2 = [
                    numpy.exp(-1j * 2.0 * numpy.pi * n[iwl] / wl * l2)
                    for n, l2 in zip(neffs, l2s, strict=True)
                ]
                A = numpy.eye(2 * n_rings, dtype=complex)
                rhs = numpy.zeros(2 * n_rings, dtype=complex)
                # Coupler 0: b2_0 = q0 * (j*k0*1 + r0*p1_0*b1_1)
                A[0, 1] = -q_vals[0] * r_vals[0] * p1[0]
                rhs[0] = q_vals[0] * 1j * k_vals[0]
                # Interior couplers m = 1 .. n_rings - 1
                for m in range(1, n_rings):
                    # b1_m = q_m * (r_m * p2_{m-1} * b2_{m-1} + j*k_m * p1_m * b1_{m+1})
                    A[2 * m - 1, 2 * m - 2] = -q_vals[m] * r_vals[m] * p2[m - 1]
                    A[2 * m - 1, 2 * m + 1] = -q_vals[m] * 1j * k_vals[m] * p1[m]
                    # b2_m = q_m * (j*k_m * p2_{m-1} * b2_{m-1} + r_m * p1_m * b1_{m+1})
                    A[2 * m, 2 * m - 2] = -q_vals[m] * 1j * k_vals[m] * p2[m - 1]
                    A[2 * m, 2 * m + 1] = -q_vals[m] * r_vals[m] * p1[m]
                # Drop coupler n_rings: b1_{n_rings} = q_N * r_N * p2_{N-1} * b2_{N-1}
                A[2 * n_rings - 1, 2 * n_rings - 2] = (
                    -q_vals[n_rings] * r_vals[n_rings] * p2[n_rings - 1]
                )
                sol = numpy.linalg.solve(A, rhs)
                thru[iwl] = q_vals[0] * (r_vals[0] + 1j * k_vals[0] * p1[0] * sol[1])
                drop[iwl] = (
                    q_vals[n_rings]
                    * 1j
                    * k_vals[n_rings]
                    * p2[n_rings - 1]
                    * sol[2 * n_rings - 2]
                )
            return thru, drop

        wls = numpy.linspace(1.545e-6, 1.555e-6, 25)
        for k_vals, q_vals, n_vals, l1s, l2s in [
            (
                [numpy.sqrt(0.3), numpy.sqrt(0.05), numpy.sqrt(0.4)],
                [0.98, 0.96, 0.99],
                [2.42 - 1e-3j, 2.47 - 2e-3j],
                [10.0e-6, 13.0e-6],
                [14.0e-6, 11.0e-6],
            ),
            (
                [
                    numpy.sqrt(0.25),
                    numpy.sqrt(0.06),
                    numpy.sqrt(0.08),
                    numpy.sqrt(0.30),
                ],
                [1.0, 0.99, 0.98, 0.97],
                [2.42 - 5e-4j, 2.45 - 1e-3j, 2.43 - 8e-4j],
                [10.0e-6, 13.0e-6, 11.5e-6],
                [14.0e-6, 11.0e-6, 15.5e-6],
            ),
        ]:
            ks = [
                EMpy.devices.Coupler(
                    wls, numpy.full_like(wls, k), numpy.full_like(wls, q)
                )
                for k, q in zip(k_vals, q_vals, strict=True)
            ]
            neffs = [n * numpy.ones_like(wls) for n in n_vals]
            nrr = EMpy.devices.NRR(ks, neffs, l1s, l2s).solve()
            thru_ref, drop_ref = _solve_crow_direct(
                wls, k_vals, q_vals, neffs, l1s, l2s
            )
            numpy.testing.assert_allclose(nrr.THRU, thru_ref, rtol=1e-13, atol=1e-14)
            numpy.testing.assert_allclose(nrr.DROP, drop_ref, rtol=1e-13, atol=1e-14)


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
            et_si.FSRwl[idx_mid], measured_fsr, delta=1e-4 * measured_fsr
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

        # 4. Uncoated low-reflectivity glass (R = 0.04, F < 1): no RuntimeWarning and clean NaN FWHM/FINESSE
        et_low_r = EMpy.devices.Etalon(
            EMpy.utils.Layer(EMpy.materials.SiO2, 10.0e-6), 0.04, 0.0
        ).solve(wls)
        self.assertTrue(numpy.all(numpy.isfinite(et_low_r.Te)))
        self.assertTrue(numpy.isnan(et_low_r.FINESSE))
        self.assertTrue(numpy.all(numpy.isnan(et_low_r.FWHMwl)))

        # 5. Tabulated dispersive RefractiveIndex(n0_known=...) uses group index ng (both array and scalar wl)
        wls_tab = numpy.linspace(1.54e-6, 1.56e-6, 11)
        slope = -1.0e7  # dn/dwl [m^-1] -> strong normal dispersion (ng ~ 19 != n ~ 3.5)
        n_known = {float(w): float(3.5 + slope * (w - 1.55e-6)) for w in wls_tab}
        lyr_tab = EMpy.utils.Layer(
            EMpy.materials.IsotropicMaterial(
                "tab", n0=EMpy.materials.RefractiveIndex(n0_known=n_known)
            ),
            10.0e-6,
        )
        lyr_poly = EMpy.utils.Layer(
            EMpy.materials.IsotropicMaterial(
                "poly",
                n0=EMpy.materials.RefractiveIndex(
                    n0_poly=[slope, 3.5 - slope * 1.55e-6]
                ),
            ),
            10.0e-6,
        )
        et_tab = EMpy.devices.Etalon(lyr_tab, 0.8, 0.0).solve(wls_tab)
        et_poly = EMpy.devices.Etalon(lyr_poly, 0.8, 0.0).solve(wls_tab)
        numpy.testing.assert_allclose(et_tab.FSRwl, et_poly.FSRwl, rtol=1e-8)
        et_tab_1 = EMpy.devices.Etalon(lyr_tab, 0.8, 0.0).solve(1.55e-6)
        self.assertAlmostEqual(
            float(et_tab_1.FSRwl[0]), float(et_poly.FSRwl[5]), delta=1e-14
        )


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

    def test_t_tcrr_and_t_ctcrr_optimum_resonance_validation_and_lossy_limits(self):
        """Verify Barbarossa 1997 optimum critical coupling (|DROP|^2 == 1), coupling validation, and lossy limits."""
        wl0 = 1.55e-6
        neff0 = 2.5
        # Choose l0 = 20 * (wl0 / neff0) so beta * l_eff[0] == 2 * m * pi at wl0
        l_unit = wl0 / neff0
        l_in = numpy.array([20.0, 12.0, 10.0, 10.0]) * l_unit
        K_in = numpy.array([0.3, 0.4, 0.7])
        wls = numpy.array([wl0])
        neff = numpy.array([neff0])

        tcrr_opt = EMpy.devices.T_TCRR(neff, K_in, 1.0, l_in, coupling="optimum").solve(
            wls
        )
        ctcrr_opt = EMpy.devices.T_CTCRR(
            neff, K_in, 1.0, l_in, coupling="optimum"
        ).solve(wls)
        self.assertAlmostEqual(float(numpy.abs(tcrr_opt.DROP[0]) ** 2), 1.0, places=13)
        self.assertAlmostEqual(float(numpy.abs(tcrr_opt.THRU[0]) ** 2), 0.0, places=13)
        self.assertAlmostEqual(float(numpy.abs(ctcrr_opt.DROP[0]) ** 2), 1.0, places=13)
        self.assertAlmostEqual(float(numpy.abs(ctcrr_opt.THRU[0]) ** 2), 0.0, places=13)

        # Invalid coupling strings must raise ValueError instead of silently overriding K, l
        for cls in (EMpy.devices.T_TCRR, EMpy.devices.T_CTCRR):
            with self.assertRaises(ValueError):
                cls(neff, K_in, 1.0, l_in, coupling="invalid")

        # Passive lossy waveguide neff = neff0 - 1j * kap: strict passivity and kap -> 0+ limit
        ctcrr_c0 = EMpy.devices.T_CTCRR(
            numpy.array([neff0 + 0.0j]), K_in, 1.0, l_in, coupling="optimum"
        ).solve(wls)
        numpy.testing.assert_allclose(ctcrr_c0.DROP, ctcrr_opt.DROP, atol=1e-12)
        numpy.testing.assert_allclose(ctcrr_c0.THRU, ctcrr_opt.THRU, atol=1e-12)

        kap_errs = []
        for kap in (1e-3, 1e-6, 1e-9):
            ck = EMpy.devices.T_CTCRR(
                numpy.array([complex(neff0, -kap)]),
                K_in,
                1.0,
                l_in,
                coupling="optimum",
            ).solve(wls)
            pwr = float(numpy.abs(ck.THRU[0]) ** 2 + numpy.abs(ck.DROP[0]) ** 2)
            self.assertLess(pwr, 1.0)
            kap_errs.append(float(numpy.abs(ck.DROP[0] - ctcrr_opt.DROP[0])))
        self.assertLess(kap_errs[1], kap_errs[0])
        self.assertLess(kap_errs[2], kap_errs[1])
        self.assertLess(kap_errs[2], 1e-6)

        # Vernier-tuned rings (l3 != 2*l1): continuous exp(-1.5j*beta*l) phase without pi branch-cut spikes
        # Straddle beta * l1 = 65 * pi (wl = 2 * 2.5 * 20e-6 / 65 = 1.5384615e-6) where |DROP| > 0
        wl_branch = 2.0 * 2.5 * 20e-6 / 65.0
        wls_span = numpy.linspace(wl_branch - 0.5e-9, wl_branch + 0.5e-9, 1001)
        neff_span = numpy.full_like(wls_span, 2.5)
        crt_tuned = EMpy.devices.T_CRTCRR(
            neff_span, [0.3, 0.3, 0.5], 1.0, [20e-6, 20e-6, 45e-6]
        ).solve(wls_span)
        ct_tuned = EMpy.devices.T_CTCRR(
            neff_span, [0.3, 0.3, 0.5], 1.0, [20e-6, 20e-6, 45e-6, 45e-6]
        ).solve(wls_span)
        numpy.testing.assert_allclose(
            numpy.abs(crt_tuned.THRU) ** 2 + numpy.abs(crt_tuned.DROP) ** 2,
            1.0,
            atol=1e-12,
        )
        self.assertGreater(float(numpy.min(numpy.abs(crt_tuned.DROP))), 0.05)
        self.assertLess(float(numpy.max(numpy.abs(numpy.diff(crt_tuned.DROP)))), 5e-3)
        self.assertLess(float(numpy.max(numpy.abs(numpy.diff(ct_tuned.DROP)))), 5e-3)
        _, tau_crt, _ = EMpy.utils.group_delay_and_dispersion(wls_span, crt_tuned.DROP)
        _, tau_ct, _ = EMpy.utils.group_delay_and_dispersion(wls_span, ct_tuned.DROP)
        self.assertTrue(numpy.all(tau_crt > 0.0))
        self.assertTrue(numpy.all(tau_ct > 0.0))

        # Complex amplitude & relative phase of T_CTCRR must match ChainMatrix composition of MZ1, K[2], MZ2
        wls_cm = numpy.linspace(1.545e-6, 1.555e-6, 11)
        neff_cm = numpy.full(wls_cm.shape, 2.5 - 1e-3j, dtype=complex)
        K_cm = numpy.array([0.3, 0.4, 0.5])
        q_cm = 0.97
        l_cm = numpy.array([20e-6, 23e-6, 27e-6, 31e-6])
        ct_cm = EMpy.devices.T_CTCRR(neff_cm, K_cm, q_cm, l_cm).solve(wls_cm)

        c0 = EMpy.devices.Coupler(
            wls_cm, numpy.full_like(wls_cm, K_cm[0]), numpy.full_like(wls_cm, q_cm)
        )
        c1 = EMpy.devices.Coupler(
            wls_cm, numpy.full_like(wls_cm, K_cm[1]), numpy.full_like(wls_cm, q_cm)
        )
        c2 = EMpy.devices.Coupler(
            wls_cm, numpy.full_like(wls_cm, K_cm[2]), numpy.full_like(wls_cm, q_cm)
        )
        mz1 = EMpy.devices.MZ(
            c0, EMpy.devices.Line(wls_cm, neff_cm, l_cm[2], neff_cm, l_cm[0]), c0
        )
        mz2 = EMpy.devices.MZ(
            c1, EMpy.devices.Line(wls_cm, neff_cm, l_cm[1], neff_cm, l_cm[3]), c1
        )
        r1_arc = EMpy.devices.Line(wls_cm, neff_cm, 0.0, neff_cm, 0.5 * l_cm[0])
        r2_arc = EMpy.devices.Line(wls_cm, neff_cm, 0.0, neff_cm, 0.5 * l_cm[1])
        thru_ref = numpy.zeros(len(wls_cm), dtype=complex)
        drop_ref = numpy.zeros(len(wls_cm), dtype=complex)
        for iwl, wl in enumerate(wls_cm):
            cm1 = EMpy.devices.composeTMlist(
                [r1_arc.TM(wl), mz1.TM(wl), r1_arc.TM(wl)]
            ).to_chain()
            cm2 = EMpy.devices.composeTMlist(
                [r2_arc.TM(wl), c2.TM(wl), r2_arc.TM(wl)]
            ).to_chain()
            H = EMpy.devices.composeCMlist([cm1, cm2, mz2.CM(wl)]).to_transfer()
            thru_ref[iwl] = H.data[0, 0]
            drop_ref[iwl] = H.data[1, 0]
        numpy.testing.assert_allclose(ct_cm.THRU, thru_ref, rtol=1e-12, atol=1e-13)
        numpy.testing.assert_allclose(ct_cm.DROP, drop_ref, rtol=1e-12, atol=1e-13)


class AnalyticDevicesTest(TestCase):

    def test_aprr_mz_srr_exact_analytic_relations(self):
        """Verify APRR, MZ, and SRR against exact closed-form Yariv/Schwelb formulas."""
        # 1. APRR: Yariv critical coupling (|THRU|=0 at t = a_rt) & exact resonant group delay
        n_r = 2.4
        l_ring = 50.0e-6
        m_order = 78  # even order so half-ring phase exp(-i*m*pi) = +1
        wl_res = n_r * l_ring / m_order
        a_rt = 0.85
        kappa = -wl_res * numpy.log(a_rt) / (2.0 * numpy.pi * l_ring)
        t_crit = a_rt
        k_crit = numpy.sqrt(1.0 - t_crit**2)
        aprr_crit = EMpy.devices.APRR(
            EMpy.devices.Coupler(wl_res, k_crit, 1.0),
            complex(n_r, -kappa),
            l_ring,
        ).solve()
        self.assertLess(abs(aprr_crit.THRU), 1e-12)

        # Lossless APRR on-resonance group delay: tau_g = ((1 + t) / (1 - t)) * (n_r * L / c)
        t_ll = 0.8
        k_ll = numpy.sqrt(1.0 - t_ll**2)
        wls_gd = numpy.linspace(wl_res - 2e-11, wl_res + 2e-11, 201)
        aprr_ll = EMpy.devices.APRR(
            EMpy.devices.Coupler(wls_gd, k_ll, 1.0),
            numpy.full_like(wls_gd, n_r),
            l_ring,
        ).solve()
        _, tau_ps, _ = EMpy.utils.group_delay_and_dispersion(wls_gd, aprr_ll.THRU)
        tau_rt_ps = (n_r * l_ring / EMpy.constants.c) * 1e12
        tau_exact_ps = ((1.0 + t_ll) / (1.0 - t_ll)) * tau_rt_ps
        self.assertAlmostEqual(float(tau_ps[len(tau_ps) // 2]), tau_exact_ps, places=4)

        # 2. MZ with balanced 3 dB couplers: |THRU|^2 = sin^2(dphi/2), |DROP|^2 = cos^2(dphi/2)
        wls_mz = numpy.linspace(1.53e-6, 1.57e-6, 51)
        c3db = EMpy.devices.Coupler(wls_mz, numpy.sqrt(0.5), 1.0)
        l1_mz, l2_mz = 100.0e-6, 130.0e-6
        neff_mz = numpy.full_like(wls_mz, 2.45)
        line_mz = EMpy.devices.Line(wls_mz, neff_mz, l1_mz, neff_mz, l2_mz)
        mz = EMpy.devices.MZ(c3db, line_mz, c3db).solve()
        dphi = 2.0 * numpy.pi * neff_mz / wls_mz * (l2_mz - l1_mz)
        numpy.testing.assert_allclose(
            numpy.abs(mz.THRU) ** 2, numpy.sin(0.5 * dphi) ** 2, atol=1e-13
        )
        numpy.testing.assert_allclose(
            numpy.abs(mz.DROP) ** 2, numpy.cos(0.5 * dphi) ** 2, atol=1e-13
        )

        # 3. SRR: Yariv critical coupling (|THRU|=0 at t1 = t2 * a_rt) & exact resonant DROP
        k2 = numpy.sqrt(0.1)
        t2 = numpy.sqrt(1.0 - k2**2)
        t1_crit = t2 * a_rt
        k1_crit = numpy.sqrt(1.0 - t1_crit**2)
        srr_crit = EMpy.devices.SRR(
            EMpy.devices.Coupler(wl_res, k1_crit, 1.0),
            EMpy.devices.Coupler(wl_res, k2, 1.0),
            complex(n_r, -kappa),
            0.5 * l_ring,
            0.5 * l_ring,
        ).solve()
        self.assertLess(abs(srr_crit.THRU), 1e-13)
        drop_exact = -k1_crit * k2 * numpy.sqrt(a_rt) / (1.0 - t1_crit * t2 * a_rt)
        self.assertAlmostEqual(abs(srr_crit.DROP - drop_exact), 0.0, places=13)
