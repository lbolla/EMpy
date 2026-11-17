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

    def test_snell_propagating_tir_and_lossy_limit(self):
        # Propagating lossless stack preserves real dtype and Snell's invariant
        theta0 = U.deg2rad(30.0)
        n_prop = numpy.array([1.0, 1.45, 2.4])
        theta_prop = U.snell(theta0, n_prop)
        self.assertFalse(numpy.iscomplexobj(theta_prop))
        numpy.testing.assert_allclose(
            n_prop * numpy.sin(theta_prop), n_prop[0] * numpy.sin(theta0), atol=1e-15
        )
        self.assertTrue(numpy.all((n_prop * numpy.cos(theta_prop)) > 0))

        # Lossless TIR on both real float64 and complex (n + 0j) inputs must select
        # the passive decaying branch Im(n * cos(theta)) < 0 and match kappa -> 0+
        theta_tir_inc = U.deg2rad(60.0)
        n_tir_real = numpy.array([1.5, 1.0])
        n_tir_cplx0 = numpy.array([1.5 + 0.0j, 1.0 + 0.0j])
        th_real = U.snell(theta_tir_inc, n_tir_real)
        th_cplx0 = U.snell(theta_tir_inc, n_tir_cplx0)
        numpy.testing.assert_allclose(th_real, th_cplx0, atol=1e-15)
        numpy.testing.assert_allclose(
            n_tir_real * numpy.sin(th_real),
            n_tir_real[0] * numpy.sin(theta_tir_inc),
            atol=1e-15,
        )
        qz_tir = n_tir_real[1] * numpy.cos(th_real[1])
        self.assertLess(qz_tir.imag, -0.8)

        # Continuous limit kappa -> 0+ in passive TIR medium n = 1.0 - i*kappa
        th_k4 = U.snell(theta_tir_inc, numpy.array([1.5, 1.0 - 1e-4j]))
        th_k8 = U.snell(theta_tir_inc, numpy.array([1.5, 1.0 - 1e-8j]))
        numpy.testing.assert_allclose(th_k8, th_real, atol=1e-7)
        self.assertLess(
            abs(th_k8[1] - th_real[1]),
            abs(th_k4[1] - th_real[1]),
        )

    def test_group_delay_and_dispersion(self):
        import scipy.integrate
        import EMpy.constants as C
        import EMpy.devices

        with self.assertRaises(ValueError):
            U.group_delay_and_dispersion(numpy.ones(5), numpy.ones(6))

        # 1. Dispersionless line (n = 2.5, L = 0.5 mm) on coarse 50-pt grid (|dphi| ~ 1.33 rad in (pi/4, pi))
        #    and on a non-uniform wavelength grid: exact tau = n*L/c, exact D = 0 ps/nm
        n0 = 2.5
        length = 0.5e-3
        tau_exact_ps = n0 * length / C.c * 1e12
        for wls in (
            numpy.linspace(1.54e-6, 1.56e-6, 50),
            1.54e-6 + 2.0e-8 * numpy.linspace(0.0, 1.0, 75) ** 1.7,
        ):
            y = numpy.exp(-1j * 2.0 * numpy.pi * n0 / wls * length)
            phi, tau, dpsnm = U.group_delay_and_dispersion(wls, y)
            self.assertEqual(phi.shape, wls.shape)
            self.assertEqual(tau.shape, (len(wls) - 1,))
            self.assertEqual(dpsnm.shape, (len(wls) - 2,))
            numpy.testing.assert_allclose(tau, tau_exact_ps, rtol=1e-10, atol=1e-10)
            numpy.testing.assert_allclose(dpsnm, 0.0, atol=1e-7)

        # 2. Linear-chirp transfer function tau(wl) = tau0 + D0 * (wl - wl0): verify exact sign & magnitude
        wls_c = numpy.linspace(1.545e-6, 1.555e-6, 1001)
        wl0 = 1.55e-6
        for d0_psnm in (5.0, -3.5):
            tau_s = 10.0e-12 + (d0_psnm * 1e-12 / 1e-9) * (wls_c - wl0)
            dphi_dwl = 2.0 * numpy.pi * C.c / wls_c**2 * tau_s
            phi_in = scipy.integrate.cumulative_trapezoid(dphi_dwl, wls_c, initial=0.0)
            _, tau_c, dpsnm_c = U.group_delay_and_dispersion(
                wls_c, numpy.exp(1j * phi_in)
            )
            wls_mid = 0.5 * (wls_c[:-1] + wls_c[1:])
            tau_expected = 10.0 + d0_psnm * (wls_mid - wl0) * 1e9
            numpy.testing.assert_allclose(tau_c, tau_expected, rtol=1e-5, atol=1e-5)
            numpy.testing.assert_allclose(dpsnm_c, d0_psnm, rtol=1e-4, atol=1e-4)

        # 3. Dispersive Silicon waveguide (1 mm) vs analytical ng = n - wl*dn/dwl and D = -(wl*L/c)*d2n/dwl2
        wls_si = numpy.linspace(1.54e-6, 1.56e-6, 201)
        l_si = 1.0e-3
        n_si = EMpy.materials.Si.n(wls_si)
        h = 1.0e-11
        dn_dwl = (EMpy.materials.Si.n(wls_si + h) - EMpy.materials.Si.n(wls_si - h)) / (
            2.0 * h
        )
        d2n_dwl2 = (
            EMpy.materials.Si.n(wls_si + h)
            - 2.0 * n_si
            + EMpy.materials.Si.n(wls_si - h)
        ) / h**2
        ng_si = n_si - wls_si * dn_dwl
        tau_si_exact = 0.5 * (ng_si[:-1] + ng_si[1:]) * l_si / C.c * 1e12
        d_si_exact = -wls_si[1:-1] * l_si / C.c * d2n_dwl2[1:-1] * 1e3

        y_si = numpy.exp(-1j * 2.0 * numpy.pi * n_si / wls_si * l_si)
        _, tau_si, d_si = U.group_delay_and_dispersion(wls_si, y_si)
        numpy.testing.assert_allclose(tau_si, tau_si_exact, rtol=1e-7, atol=1e-7)
        numpy.testing.assert_allclose(d_si, d_si_exact, rtol=1e-4, atol=1e-6)

        # 4. 2D batched input shape (2, N) and lossy -> non-lossy limit on ring resonator (APRR)
        wls_2d = numpy.vstack([wls_si, wls_si])
        y_2d = numpy.vstack(
            [y_si, numpy.exp(-1j * 2.0 * numpy.pi * (n_si - 1e-4j) / wls_si * l_si)]
        )
        phi_2d, tau_2d, d_2d = U.group_delay_and_dispersion(wls_2d, y_2d)
        self.assertEqual(phi_2d.shape, (2, len(wls_si)))
        self.assertEqual(tau_2d.shape, (2, len(wls_si) - 1))
        self.assertEqual(d_2d.shape, (2, len(wls_si) - 2))
        numpy.testing.assert_allclose(tau_2d[1], tau_2d[0], rtol=1e-10, atol=1e-10)
        numpy.testing.assert_allclose(d_2d[1], d_2d[0], rtol=1e-8, atol=1e-10)

        # Lossy APRR -> lossless APRR limit (kappa -> 0+)
        wls_r = numpy.linspace(1.5495e-6, 1.5505e-6, 401)
        l_ring = 50.0e-6
        coupler = EMpy.devices.Coupler(wls_r, numpy.sqrt(0.2), 1.0)
        aprr_ll = EMpy.devices.APRR(coupler, 2.4 + 0.0 * wls_r, l_ring).solve()
        _, tau_ll, _ = U.group_delay_and_dispersion(wls_r, aprr_ll.THRU)
        self.assertTrue(numpy.all(tau_ll > 0.0))

        aprr_errs = []
        for kap in (1e-3, 1e-6, 0.0):
            aprr_k = EMpy.devices.APRR(
                coupler, (2.4 - 1j * kap) + 0.0 * wls_r, l_ring
            ).solve()
            _, tau_k, _ = U.group_delay_and_dispersion(wls_r, aprr_k.THRU)
            aprr_errs.append(float(numpy.max(numpy.abs(tau_k - tau_ll))))
        self.assertLess(aprr_errs[1], aprr_errs[0])
        self.assertLess(aprr_errs[2], 1e-10)

    def test_liquid_crystal_cell_alignment_twist_bvp_and_degenerate_limits(self):
        from pathlib import Path
        import sys
        import types
        from unittest import mock
        import scipy.integrate
        import EMpy.constants as C
        import EMpy.transfer_matrix as TM

        wl = 1.55e-6
        lc_file_path = str(
            Path(__file__).resolve().parent.parent / "examples" / "lc_data.dat"
        )
        lc = EMpy.materials.LiquidCrystal(
            "E7",
            nO=1.52,
            nE=1.74,
            nO_electrical=1.8,
            nE_electrical=4.2,
            K11=11e-12,
            K22=6.5e-12,
            K33=17e-12,
        )

        # 1. At V = 0, BVP planar alignment must match examples/lc_data.dat (86.5 deg from z for 3.5 deg pretilt)
        c_file = U.LiquidCrystalCell(
            lc,
            voltage=0.0,
            t_tot=10e-6,
            t_anchoring=0.0,
            pretilt=U.deg2rad(3.5),
            nlayers=9,
            data_file=lc_file_path,
        )
        c_bvp = U.LiquidCrystalCell(
            lc,
            voltage=0.0,
            t_tot=10e-6,
            t_anchoring=0.0,
            pretilt=U.deg2rad(3.5),
            nlayers=9,
        )
        self.assertFalse(numpy.iscomplexobj(c_file.angles))
        numpy.testing.assert_allclose(c_bvp.angles, 86.5, atol=1e-14)
        numpy.testing.assert_allclose(c_bvp.angles, c_file.angles, atol=1e-14)
        e_file = c_file.createMultilayer()[0].mat.epsilonTensor(wl)[:, :, 0]
        e_bvp = c_bvp.createMultilayer()[0].mat.epsilonTensor(wl)[:, :, 0]
        numpy.testing.assert_allclose(e_bvp, e_file, rtol=1e-14, atol=1e-15)
        self.assertGreater(e_bvp[0, 0].real, e_bvp[2, 2].real)

        # 2. Gooch-Tarry 90-deg Twisted Nematic first Mauguin minimum (u = sqrt(3): ~100% polarization rotation)
        n_o, n_e = 1.50, 1.55
        d_gt = numpy.sqrt(3.0) * wl / (2.0 * (n_e - n_o))
        lc_tn = EMpy.materials.LiquidCrystal("TN", nO=n_o, nE=n_e)
        cell_tn = U.LiquidCrystalCell(
            lc_tn,
            voltage=0.0,
            t_tot=d_gt,
            t_anchoring=0.0,
            pretilt=0.0,
            totaltwist=numpy.pi / 2.0,
            nlayers=60,
        )
        numpy.testing.assert_allclose(cell_tn.twists[[0, -1]], [0.0, 90.0], atol=1e-13)
        amb = EMpy.materials.IsotropicMaterial(
            "amb", n0=EMpy.materials.RefractiveIndex(n0_const=0.5 * (n_o + n_e))
        )
        ml_tn = U.Multilayer(
            [U.Layer(amb, numpy.inf), cell_tn, U.Layer(amb, numpy.inf)]
        ).simplify()
        tmm_tn = TM.AnisotropicTransferMatrix(ml_tn, 0.0, 0.0).solve([wl])
        self.assertLess(tmm_tn.T[1, 1, 0], 1e-6)
        self.assertGreater(tmm_tn.T[1, 0, 0], 0.9998)

        # 3. Live BVP Freedericksz transition & chiral q0*d scale invariance via scipy.integrate.solve_bvp
        def _colnew_shim(boundary_points, degrees, fsub, gsub, **kwargs):
            z_mesh = numpy.linspace(0.0, 1.0, 41)
            y_init, _ = kwargs["initial_guess"](z_mesh)

            def _fun(z, y):
                ddt, ddp, ddu = fsub(z, y)
                return numpy.vstack([y[1], ddt, y[3], ddp, y[5], ddu])

            def _bc(ya, yb):
                return gsub(numpy.column_stack([ya, ya, ya, yb, yb, yb]))

            res = scipy.integrate.solve_bvp(
                _fun, _bc, z_mesh, y_init, tol=1e-8, max_nodes=2000
            )
            self.assertTrue(res.success, msg=res.message)

            class _Sol:
                mesh = res.x

                def __call__(self, z):
                    return res.sol(z).T

            return _Sol()

        fake_bvp_pkg = types.ModuleType("scikits.bvp1lg")
        fake_colnew = types.ModuleType("scikits.bvp1lg.colnew")
        fake_colnew.solve = _colnew_shim  # type: ignore[attr-defined]
        fake_bvp_pkg.colnew = fake_colnew  # type: ignore[attr-defined]

        with mock.patch.dict(
            sys.modules,
            {
                "scikits": types.ModuleType("scikits"),
                "scikits.bvp1lg": fake_bvp_pkg,
                "scikits.bvp1lg.colnew": fake_colnew,
            },
        ):
            lc_mix = EMpy.materials.get_10400_000_100(50)
            v_th = numpy.pi * numpy.sqrt(lc_mix.K11 / (C.eps0 * lc_mix.deleps))
            c_low = U.LiquidCrystalCell(
                lc_mix,
                voltage=0.6 * v_th,
                t_tot=8e-6,
                t_anchoring=0.0,
                pretilt=0.02,
                nlayers=21,
            )
            c_hi = U.LiquidCrystalCell(
                lc_mix,
                voltage=3.0 * v_th,
                t_tot=8e-6,
                t_anchoring=0.0,
                pretilt=0.02,
                nlayers=21,
            )
            self.assertGreater(c_low.angles[10], 85.0)
            self.assertLess(c_hi.angles[10], 10.0)
            self.assertGreater(c_hi.capacitance(), 1.4 * c_low.capacitance())

            # Chiral q0 * d dimensional scaling invariance: identical q0 * d must produce identical profiles
            lc_q0 = EMpy.materials.LiquidCrystal(
                "c0", 1.5, 1.7, 1.8, 3.5, 11e-12, 6.5e-12, 17e-12, q0=0.0
            )
            lc_qa = EMpy.materials.LiquidCrystal(
                "cA",
                1.5,
                1.7,
                1.8,
                3.5,
                11e-12,
                6.5e-12,
                17e-12,
                q0=numpy.pi / (2.0 * 4e-6),
            )
            lc_qb = EMpy.materials.LiquidCrystal(
                "cB",
                1.5,
                1.7,
                1.8,
                3.5,
                11e-12,
                6.5e-12,
                17e-12,
                q0=numpy.pi / (2.0 * 40e-6),
            )
            c0 = U.LiquidCrystalCell(
                lc_q0,
                voltage=1.5,
                t_tot=6e-6,
                t_anchoring=1e-6,
                pretilt=0.2,
                totaltwist=numpy.pi / 2.0,
                nlayers=15,
            )
            ca = U.LiquidCrystalCell(
                lc_qa,
                voltage=1.5,
                t_tot=6e-6,
                t_anchoring=1e-6,
                pretilt=0.2,
                totaltwist=numpy.pi / 2.0,
                nlayers=15,
            )
            cb = U.LiquidCrystalCell(
                lc_qb,
                voltage=1.5,
                t_tot=42e-6,
                t_anchoring=1e-6,
                pretilt=0.2,
                totaltwist=numpy.pi / 2.0,
                nlayers=15,
            )
            numpy.testing.assert_allclose(ca.angles, cb.angles, atol=1e-12)
            numpy.testing.assert_allclose(ca.twists, cb.twists, atol=1e-12)
            self.assertGreater(float(numpy.max(numpy.abs(ca.angles - c0.angles))), 0.1)

        # 4. Degenerate limits: twisted/pretilted LC -> IsotropicTransferMatrix (dn -> 0) and lossy kappa -> 0+
        air = EMpy.materials.Air
        glass = EMpy.materials.IsotropicMaterial(
            "glass", n0=EMpy.materials.RefractiveIndex(n0_const=1.48)
        )
        lc_iso_mat = EMpy.materials.IsotropicMaterial(
            "lc_iso", n0=EMpy.materials.RefractiveIndex(n0_const=1.52)
        )
        tmm_iso = TM.IsotropicTransferMatrix(
            U.Multilayer(
                [
                    U.Layer(air, numpy.inf),
                    U.Layer(lc_iso_mat, 5.0e-6),
                    U.Layer(glass, numpy.inf),
                ]
            ),
            U.deg2rad(25.0),
        ).solve([wl])

        dn_errs = []
        for dn in (1e-4, 1e-8, 0.0):
            lc_d = EMpy.materials.LiquidCrystal("d", nO=1.52, nE=1.52 + dn)
            cell_d = U.LiquidCrystalCell(
                lc_d,
                voltage=0.0,
                t_tot=5.0e-6,
                t_anchoring=0.5e-6,
                pretilt=U.deg2rad(12.0),
                totaltwist=U.deg2rad(75.0),
                nlayers=12,
            )
            ml_d = U.Multilayer(
                [U.Layer(air, numpy.inf), cell_d, U.Layer(glass, numpy.inf)]
            )
            tmm_d = TM.AnisotropicTransferMatrix(ml_d, U.deg2rad(25.0), 0.0).solve([wl])
            err = (
                abs(float(tmm_d.R[0, 0, 0] - tmm_iso.Rs[0]))
                + abs(float(tmm_d.T[0, 0, 0] - tmm_iso.Ts[0]))
                + abs(float(tmm_d.R[1, 1, 0] - tmm_iso.Rp[0]))
                + abs(float(tmm_d.T[1, 1, 0] - tmm_iso.Tp[0]))
                + float(tmm_d.T[0, 1, 0] + tmm_d.T[1, 0, 0])
            )
            dn_errs.append(err)
        self.assertLess(dn_errs[1], dn_errs[0])
        self.assertLess(dn_errs[2], 1e-13)

        # Lossy -> non-lossy limit on twisted LC cell (kap = 1e-2, 1e-5, 0.0 + 0j)
        def _solve_lc_lossy(kap, force_complex=False):
            no = complex(1.52, -kap) if (kap > 0 or force_complex) else 1.52
            ne = complex(1.65, -1.5 * kap) if (kap > 0 or force_complex) else 1.65
            lc_k = EMpy.materials.LiquidCrystal("lk", nO=no, nE=ne)
            cell_k = U.LiquidCrystalCell(
                lc_k,
                voltage=0.0,
                t_tot=4.0e-6,
                t_anchoring=0.2e-6,
                pretilt=U.deg2rad(5.0),
                totaltwist=numpy.pi / 2.0,
                nlayers=15,
            )
            ml_k = U.Multilayer(
                [U.Layer(air, numpy.inf), cell_k, U.Layer(glass, numpy.inf)]
            )
            return TM.AnisotropicTransferMatrix(ml_k, U.deg2rad(15.0), 0.0).solve([wl])

        t_ll = _solve_lc_lossy(0.0, force_complex=False)
        t_c0 = _solve_lc_lossy(0.0, force_complex=True)
        numpy.testing.assert_allclose(t_c0.R, t_ll.R, atol=1e-14)
        numpy.testing.assert_allclose(t_c0.T, t_ll.T, atol=1e-14)
        numpy.testing.assert_allclose(
            t_ll.R[:, :, 0].sum(axis=1) + t_ll.T[:, :, 0].sum(axis=1), 1.0, atol=1e-13
        )

        kap_errs = []
        for kap in (1e-2, 1e-5):
            tk = _solve_lc_lossy(kap)
            self.assertTrue(
                numpy.all(tk.R[:, :, 0].sum(axis=1) + tk.T[:, :, 0].sum(axis=1) < 1.0)
            )
            kap_errs.append(float(numpy.max(numpy.abs(tk.T - t_ll.T))))
        self.assertLess(kap_errs[1], kap_errs[0])
        self.assertLess(kap_errs[1], 1e-3)
