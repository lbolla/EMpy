from unittest import TestCase

import matplotlib
import numpy

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

import EMpy.modesolvers  # noqa: E402


class VFDModeSolverTest(TestCase):

    def test_solve(self):

        def epsfunc(x_, y_):
            """Return a matrix describing a 2d material.

            :param x_: x values
            :param y_: y values
            :return: 2d-matrix
            """
            xx, yy = numpy.meshgrid(x_, y_)
            return numpy.where(
                (numpy.abs(xx.T - 1.24e-6) <= 0.24e-6)
                * (numpy.abs(yy.T - 1.11e-6) <= 0.11e-6),
                3.4757**2,
                1.446**2,
            )

        neigs = 2
        tol = 1e-8
        boundary = "0000"
        wl = 1.55e-6
        x = numpy.linspace(0, 2.48e-6, 125)
        y = numpy.linspace(0, 2.22e-6, 112)
        solver = EMpy.modesolvers.FD.VFDModeSolver(wl, x, y, epsfunc, boundary).solve(
            neigs, tol
        )
        self.assertEqual(len(solver.modes), 2)
        self.assertAlmostEqual(solver.modes[0].neff, 2.4179643622942937)

        O11, O22, O12, O21 = EMpy.modesolvers.interface.interface_matrix(solver, solver)
        numpy.testing.assert_allclose(O11, numpy.eye(2), atol=1e-6)
        numpy.testing.assert_allclose(O12, numpy.eye(2), atol=1e-6)

    def test_svfd_symmetry_matches_full_domain(self):
        def epsfunc(x_, y_):
            xx, yy = numpy.meshgrid(x_, y_)
            return numpy.where(
                (numpy.abs(xx.T) <= 0.25e-6) * (numpy.abs(yy.T) <= 0.12e-6),
                3.48**2,
                1.44**2,
            )

        wl = 1.55e-6
        x_full = numpy.linspace(-1.0e-6, 1.0e-6, 41)
        y_full = numpy.linspace(-1.0e-6, 1.0e-6, 41)
        x_q = x_full[20:]
        y_q = y_full[20:]

        # Fundamental Ex mode: Hx antisymmetric, Hy symmetric ('A' on S/W walls)
        sv_ex_full = EMpy.modesolvers.FD.SVFDModeSolver(
            wl, x_full, y_full, epsfunc, "0000", method="Ex"
        ).solve(1, 1e-10)
        sv_ex_q = EMpy.modesolvers.FD.SVFDModeSolver(
            wl, x_q, y_q, epsfunc, "0A0A", method="Ex"
        ).solve(1, 1e-10)
        self.assertAlmostEqual(sv_ex_q.neff[0].real, sv_ex_full.neff[0].real, places=9)

        # Fundamental Ey mode: Hx symmetric, Hy antisymmetric ('S' on S/W walls)
        sv_ey_full = EMpy.modesolvers.FD.SVFDModeSolver(
            wl, x_full, y_full, epsfunc, "0000", method="Ey"
        ).solve(1, 1e-10)
        sv_ey_q = EMpy.modesolvers.FD.SVFDModeSolver(
            wl, x_q, y_q, epsfunc, "0S0S", method="Ey"
        ).solve(1, 1e-10)
        self.assertAlmostEqual(sv_ey_q.neff[0].real, sv_ey_full.neff[0].real, places=9)

    def test_stretchmesh_geometry_symmetry_and_pml_signs(self):
        """Verify stretchmesh profiles (U/L/P/G), boundary mirror symmetry, and passive PML Im(dx), Im(dy) <= 0."""
        x = numpy.linspace(0.0, 2.0, 21)
        y = numpy.linspace(-1.0, 1.5, 26)
        n = 5

        # Degenerate unstretched limits: nlayers=0 or factor=1
        xx0, yy0, xc0, yc0, dx0, dy0 = EMpy.modesolvers.FD.stretchmesh(x, y, 0, 1 - 2j)
        numpy.testing.assert_allclose(xx0, x, atol=1e-15)
        numpy.testing.assert_allclose(yy0, y, atol=1e-15)
        xx1, yy1, _, _, _, _ = EMpy.modesolvers.FD.stretchmesh(
            x, y, [n, n, n, n], [1, 1, 1, 1]
        )
        numpy.testing.assert_allclose(xx1, x, atol=1e-15)
        numpy.testing.assert_allclose(yy1, y, atol=1e-15)

        # Real expansion across all four methods ('U', 'L', 'P', 'G') with list / int-array inputs
        nl_list = [n, n, n, n]
        fac_int = numpy.array([3, 3, 3, 3], dtype=int)
        for mch in ("U", "L", "P", "G"):
            xx, yy, _, _, dx, dy = EMpy.modesolvers.FD.stretchmesh(
                x, y, nl_list, fac_int, method=mch * 4
            )
            self.assertEqual(nl_list, [n, n, n, n])
            numpy.testing.assert_array_equal(fac_int, numpy.array([3, 3, 3, 3]))
            numpy.testing.assert_allclose(xx.imag, 0.0, atol=1e-15)
            numpy.testing.assert_allclose(yy.imag, 0.0, atol=1e-15)
            numpy.testing.assert_allclose(dx[:n], dx[-n:][::-1], rtol=1e-13, atol=1e-14)
            numpy.testing.assert_allclose(dy[:n], dy[-n:][::-1], rtol=1e-13, atol=1e-14)
            self.assertAlmostEqual(
                float(xx[-1].real - xx[0].real),
                (2.0 - 2 * 0.5) + 2 * 3.0 * 0.5,
                places=12,
            )

        # Geometric method 'G' across compression (0 < f < 1) and expansion (f > 1):
        # must converge to the unique nonzero root of exp(b) - 1 - f*b = 0 with exact
        # total length f * L_orig, positive cell widths, and constant geometric ratio exp(b / n).
        for f_g in (0.3, 0.5, 0.7, 0.9, 0.999, 1.001, 1.5, 3.0):
            xx_g, yy_g, _, _, dx_g, dy_g = EMpy.modesolvers.FD.stretchmesh(
                x, y, n, f_g, method="GGGG"
            )
            self.assertTrue(numpy.all(dx_g.real > 0.0))
            self.assertTrue(numpy.all(dy_g.real > 0.0))
            numpy.testing.assert_allclose(dx_g.imag, 0.0, atol=1e-15)
            numpy.testing.assert_allclose(
                dx_g[:n], dx_g[-n:][::-1], rtol=1e-12, atol=1e-13
            )
            self.assertAlmostEqual(
                float(xx_g[-1].real - xx_g[0].real),
                (2.0 - 2 * 0.5) + 2 * f_g * 0.5,
                places=12,
            )
            # Constant geometric progression ratio r = dx[i+1] / dx[i] across the East stretched boundary
            ratios = dx_g[-n + 1 :].real / dx_g[-n:-1].real
            numpy.testing.assert_allclose(ratios, ratios[0], rtol=1e-12, atol=1e-13)
            if f_g < 1.0:
                self.assertLess(ratios[0], 1.0)
            else:
                self.assertGreater(ratios[0], 1.0)

        # Complex PML coordinate stretching across 'U', 'L', 'P'
        f_pml = 1.0 - 2.0j
        for mch in ("U", "L", "P"):
            xx, yy, xc, yc, dx, dy = EMpy.modesolvers.FD.stretchmesh(
                x, y, n, f_pml, method=mch * 4
            )
            # Interior mesh must remain strictly real and untouched
            numpy.testing.assert_allclose(
                xx[n : len(x) - n], x[n : len(x) - n], atol=1e-15
            )
            numpy.testing.assert_allclose(
                yy[n : len(y) - n], y[n : len(y) - n], atol=1e-15
            )
            # All four PMLs must have Im(dx) < 0 and Im(dy) < 0 (passive absorption, never gain)
            self.assertTrue(numpy.all(dx[:n].imag < 0.0))
            self.assertTrue(numpy.all(dx[-n:].imag < 0.0))
            self.assertTrue(numpy.all(dy[:n].imag < 0.0))
            self.assertTrue(numpy.all(dy[-n:].imag < 0.0))
            # Exact mirror symmetry between West/East and South/North in both real and imag parts
            numpy.testing.assert_allclose(dx[:n], dx[-n:][::-1], rtol=1e-14, atol=1e-15)
            numpy.testing.assert_allclose(dy[:n], dy[-n:][::-1], rtol=1e-14, atol=1e-15)
            # Total complex domain span includes cumulative PML stretching on both sides
            self.assertAlmostEqual(
                xx[-1] - xx[0], (2.0 - 2 * 0.5) + 2 * f_pml * 0.5, places=13
            )
            # Conjugating factor must conjugate the stretched coordinates
            xx_conj, yy_conj, _, _, _, _ = EMpy.modesolvers.FD.stretchmesh(
                x, y, n, numpy.conj(f_pml), method=mch * 4
            )
            numpy.testing.assert_allclose(xx_conj, numpy.conj(xx), atol=1e-15)
            numpy.testing.assert_allclose(yy_conj, numpy.conj(yy), atol=1e-15)

        with self.assertRaises(ValueError):
            EMpy.modesolvers.FD.stretchmesh(x, y, n, 2.0, method="XXXX")

    def test_stretchmesh_pml_leaky_slab_exact_dispersion_and_invariance(self):
        """Verify VFD/SVFD + stretchmesh PML reproduces exact analytical leaky-waveguide complex neff."""
        import scipy.optimize

        wl = 1.55e-6
        k0 = 2.0 * numpy.pi / wl
        n1 = 1.45
        n2 = 1.80
        w = 2.4e-6

        # Exact fundamental leaky TE0 dispersion root: kx1 * tan(kx1 * w / 2) = i * kx2 (Re(kx2) > 0)
        def _leaky_disp(vec):
            neff = vec[0] + 1j * vec[1]
            kx1 = numpy.sqrt((k0 * n1) ** 2 - (k0 * neff) ** 2)
            kx2 = numpy.sqrt((k0 * n2) ** 2 - (k0 * neff) ** 2)
            if kx2.real < 0:
                kx2 = -kx2
            res = (kx1 * numpy.tan(kx1 * w / 2.0) - 1j * kx2) / k0
            return [float(res.real), float(res.imag)]

        neff0 = float(numpy.sqrt(n1**2 - (wl / (2.0 * w)) ** 2))
        sol = scipy.optimize.root(_leaky_disp, [neff0, -0.01], tol=1e-12)
        self.assertTrue(sol.success)
        neff_exact = sol.x[0] + 1j * sol.x[1]
        self.assertLess(neff_exact.imag, -0.01)

        # 1. East/West PMLs (leaky along x, invariant along y) across multiple PML strengths
        lx = 6.0e-6
        x = numpy.linspace(0.0, lx, 121)
        y = numpy.linspace(0.0, 0.5e-6, 6)

        def _eps_x(xc, yc):
            return numpy.where(
                numpy.abs(xc[:, None] - lx / 2.0) + 0.0 * yc[None, :] <= w / 2.0,
                n1**2,
                n2**2,
            )

        neffs_pml = []
        for f_pml in (1.0 - 1.5j, 1.0 - 2.5j, 1.0 - 3.5j):
            xx, yy, *_ = EMpy.modesolvers.FD.stretchmesh(x, y, [0, 0, 20, 20], f_pml)
            vfd = EMpy.modesolvers.FD.VFDModeSolver(wl, xx, yy, _eps_x, "SS00").solve(
                4, 1e-12, guess=neff_exact.real - 0.02
            )
            cands = numpy.array([m.neff for m in vfd.modes])
            best = cands[numpy.argmin(numpy.abs(cands - neff_exact))]
            self.assertLess(best.imag, 0.0)
            self.assertLess(abs(best - neff_exact), 2.5e-4)
            neffs_pml.append(best)
        # PML-strength invariance across 1-1.5j, 1-2.5j, 1-3.5j
        self.assertLess(abs(neffs_pml[0] - neffs_pml[1]), 5e-5)
        self.assertLess(abs(neffs_pml[1] - neffs_pml[2]), 5e-5)

        # Semi-vectorial SVFDModeSolver (Ey) on the same PML mesh
        xx_sv, yy_sv, *_ = EMpy.modesolvers.FD.stretchmesh(
            x, y, [0, 0, 20, 20], 1.0 - 2.5j
        )
        sv = EMpy.modesolvers.FD.SVFDModeSolver(
            wl, xx_sv, yy_sv, _eps_x, "SS00", method="Ey"
        ).solve(4, 1e-12, guess=neff_exact.real - 0.02)
        best_sv = sv.neff[numpy.argmin(numpy.abs(sv.neff - neff_exact))]
        self.assertLess(abs(best_sv - neff_exact), 2.5e-4)

        # 2. 90-degree rotated North/South PMLs (leaky along y, invariant along x) must match East/West identically
        def _eps_y(xc, yc):
            return numpy.where(
                0.0 * xc[:, None] + numpy.abs(yc[None, :] - lx / 2.0) <= w / 2.0,
                n1**2,
                n2**2,
            )

        xx_ns, yy_ns, *_ = EMpy.modesolvers.FD.stretchmesh(
            y, x, [20, 20, 0, 0], 1.0 - 2.5j
        )
        vfd_ns = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xx_ns, yy_ns, _eps_y, "00AA"
        ).solve(4, 1e-12, guess=neff_exact.real - 0.02)
        cands_ns = numpy.array([m.neff for m in vfd_ns.modes])
        best_ns = cands_ns[numpy.argmin(numpy.abs(cands_ns - neff_exact))]
        self.assertAlmostEqual(best_ns, neffs_pml[1], places=10)

    def test_stretchmesh_pml_anisotropic_and_lossy_degenerate_limits(self):
        """Verify PML + VFDModeSolver/FDMode across PML sigma -> 0+, material kappa -> 0+, and aniso dn -> 0 limits."""
        wl = 1.55e-6
        xg = numpy.linspace(0.0, 3.0e-6, 49)
        yg = numpy.linspace(0.0, 2.2e-6, 37)

        def _eps_iso(kap=0.0):
            def _f(xc, yc):
                nc = complex(3.45, -kap) if kap > 0 else 3.45
                ncl = complex(1.45, -0.2 * kap) if kap > 0 else 1.45
                return numpy.where(
                    (xc[:, None] >= 1.2e-6)
                    & (xc[:, None] <= 1.8e-6)
                    & (yc[None, :] >= 0.99e-6)
                    & (yc[None, :] <= 1.21e-6),
                    nc**2,
                    ncl**2,
                )

            return _f

        def _eps_aniso(dn=0.0, kap=0.0, phi_deg=33.0):
            ph = EMpy.utils.deg2rad(phi_deg)

            def _f(xc, yc):
                core = (
                    (xc[:, None] >= 1.2e-6)
                    & (xc[:, None] <= 1.8e-6)
                    & (yc[None, :] >= 0.99e-6)
                    & (yc[None, :] <= 1.21e-6)
                )
                nc = complex(3.45, -kap) if kap > 0 else 3.45
                ncl = complex(1.45, -0.2 * kap) if kap > 0 else 1.45
                nO = numpy.where(core, nc, ncl)
                nE = numpy.where(core, nc + dn, ncl + dn)
                exx = nE**2 * numpy.cos(ph) ** 2 + nO**2 * numpy.sin(ph) ** 2
                eyy = nE**2 * numpy.sin(ph) ** 2 + nO**2 * numpy.cos(ph) ** 2
                exy = (nE**2 - nO**2) * numpy.sin(ph) * numpy.cos(ph)
                ezz = nO**2
                return numpy.stack([exx, exy, exy, eyy, ezz], axis=-1)

            return _f

        s_real = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xg, yg, _eps_iso(0.0), "0000"
        ).solve(1, 1e-12, guess=2.3)

        # 1. PML sigma -> 0+ limit (sigma = 1.5, 1e-2, 1e-5, 0.0) -> real unstretched mesh
        pml_errs = []
        for sig in (1.5, 1e-2, 0.0):
            xx, yy, *_ = EMpy.modesolvers.FD.stretchmesh(xg, yg, 6, 1.0 - 1j * sig)
            sp = EMpy.modesolvers.FD.VFDModeSolver(
                wl, xx, yy, _eps_iso(0.0), "0000"
            ).solve(1, 1e-12, guess=2.3)
            pml_errs.append(abs(sp.modes[0].neff - s_real.modes[0].neff))
        self.assertLess(pml_errs[1], pml_errs[0])
        self.assertLess(pml_errs[2], 1e-9)

        # Full FDMode post-processing on complex PML mesh (get_field, intensity, TEfrac, overlap, FDTD, plot)
        xx_p, yy_p, *_ = EMpy.modesolvers.FD.stretchmesh(xg, yg, 6, 1.0 - 1.5j)
        s_pml = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xx_p, yy_p, _eps_iso(0.0), "0000"
        ).solve(1, 1e-12, guess=2.3)
        m_pml = s_pml.modes[0]
        self.assertAlmostEqual(m_pml.neff.real, s_real.modes[0].neff.real, places=4)
        self.assertAlmostEqual(float(numpy.real(m_pml.norm())), 1.0, places=10)
        self.assertAlmostEqual(
            m_pml.TEfrac(xg, yg), s_real.modes[0].TEfrac(xg, yg), places=3
        )
        self.assertGreater(abs(m_pml.overlap(s_real.modes[0])), 0.999)
        self.assertEqual(m_pml.get_field("Ex", xg, yg).shape, (len(xg), len(yg)))
        self.assertEqual(len(m_pml.get_fields_for_FDTD()), 6)
        m_pml.plot()
        m_pml.plot_intensity()
        plt.close("all")

        # 2. Material lossy -> lossless limit on complex PML mesh (kap = 1e-2, 1e-5, 0.0)
        kap_errs = []
        for kap in (1e-2, 1e-5, 0.0):
            sk = EMpy.modesolvers.FD.VFDModeSolver(
                wl, xx_p, yy_p, _eps_iso(kap), "0000"
            ).solve(1, 1e-12, guess=2.3)
            if kap > 0:
                self.assertLess(sk.modes[0].neff.imag, 0.0)
            kap_errs.append(abs(sk.modes[0].neff - m_pml.neff))
        self.assertLess(kap_errs[1], kap_errs[0])
        self.assertLess(kap_errs[2], 1e-9)

        # 3. Anisotropic 5-component rotated tensor -> scalar isotropic limit on complex PML mesh
        ani_errs = []
        for dn in (1e-4, 1e-8, 0.0):
            sa = EMpy.modesolvers.FD.VFDModeSolver(
                wl, xx_p, yy_p, _eps_aniso(dn=dn, kap=0.0), "0000"
            ).solve(1, 1e-12, guess=2.3)
            ani_errs.append(abs(sa.modes[0].neff - m_pml.neff))
        self.assertLess(ani_errs[1], ani_errs[0])
        self.assertLess(ani_errs[2], 1e-9)

        # Lossy anisotropic with dn = 0 on PML mesh matches lossy scalar isotropic on PML mesh
        s_iso_l = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xx_p, yy_p, _eps_iso(1e-3), "0000"
        ).solve(1, 1e-12, guess=2.3)
        s_ani_l = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xx_p, yy_p, _eps_aniso(dn=0.0, kap=1e-3), "0000"
        ).solve(1, 1e-12, guess=2.3)
        self.assertAlmostEqual(s_ani_l.modes[0].neff, s_iso_l.modes[0].neff, places=9)

    def test_fdtd_overlap_f_forward_backward_units_and_limits(self):
        """Verify scripts.FDTD.overlap_f extracts exact (A, B, Pm) across SI/micron grids and aniso/lossy limits."""
        import importlib.util
        from pathlib import Path

        fdtd_path = Path(__file__).resolve().parent.parent / "scripts" / "FDTD.py"
        spec = importlib.util.spec_from_file_location("_empy_fdtd_script", fdtd_path)
        assert spec is not None and spec.loader is not None
        FDTD_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(FDTD_mod)

        z0_imp = EMpy.constants.eta0
        sqrt_z0 = numpy.sqrt(z0_imp)

        def _make_sensor(
            mode, dx_um, dy_um, nx_nodes, ny_nodes, a_amp, b_amp, unit_scale
        ):
            x_g = dx_um * unit_scale * numpy.arange(nx_nodes)
            y_g = dy_um * unit_scale * numpy.arange(ny_nodes)
            Ex_f, Ey_f, _, Hx_f, Hy_f, _ = mode.get_fields_for_FDTD(x_g, y_g)
            n1 = nx_nodes - 1
            n2 = ny_nodes - 1
            sens = FDTD_mod.Sensor()
            sens.dx1 = dx_um
            sens.dx2 = dy_um
            sens.E1 = numpy.zeros((n1, n2 + 1, 1), dtype=complex)
            sens.H1 = numpy.zeros((n1, n2 + 1, 1), dtype=complex)
            sens.E2 = numpy.zeros((n1 + 1, n2, 1), dtype=complex)
            sens.H2 = numpy.zeros((n1 + 1, n2, 1), dtype=complex)
            sens.E1[:, 1:-1, 0] = (a_amp + b_amp) * sqrt_z0 * Ex_f
            sens.E2[1:-1, :, 0] = (a_amp + b_amp) * sqrt_z0 * Ey_f
            sens.H1[:, 1:-1, 0] = (a_amp - b_amp) * sqrt_z0 * Hy_f
            sens.H2[1:-1, :, 0] = (a_amp - b_amp) * sqrt_z0 * Hx_f
            return sens

        # 1. SI-meter VFDModeSolver: counter-propagating complex amplitudes (A, B) and Parseval power Pm
        wl = 1.55e-6
        nx_n, ny_n = 61, 41
        dx_um, dy_um = 0.05, 0.05
        xg = dx_um * 1e-6 * numpy.arange(nx_n)
        yg = dy_um * 1e-6 * numpy.arange(ny_n)

        def _eps_si(kap=0.0):
            def _f(xc, yc):
                nc = complex(3.45, -kap) if kap > 0 else 3.45
                return numpy.where(
                    (numpy.abs(xc[:, None] - 1.5e-6) <= 0.25e-6)
                    & (numpy.abs(yc[None, :] - 1.0e-6) <= 0.11e-6),
                    nc**2,
                    1.45**2,
                )

            return _f

        s_si = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xg, yg, _eps_si(0.0), "0000"
        ).solve(1, 1e-12)
        a0 = 1.4 * numpy.exp(0.5j)
        b0 = -0.6 * numpy.exp(-0.3j)
        sens_si = _make_sensor(s_si.modes[0], dx_um, dy_um, nx_n, ny_n, a0, b0, 1e-6)
        vu, ju, A, B, Pm, P = FDTD_mod.overlap_f([sens_si], s_si, 0)
        self.assertAlmostEqual(A[0, 0], a0, places=3)
        self.assertAlmostEqual(B[0, 0], b0, places=3)
        self.assertAlmostEqual(Pm[0, 0], abs(a0) ** 2 - abs(b0) ** 2, places=3)
        self.assertAlmostEqual(P[0], Pm[0, 0], places=12)

        # 2. Micron-unit FMM2d solver (wl = 1.55): consistent micron grid & integration without 1e-12 scaling error
        fmm = EMpy.modesolvers.FMM.FMM2d(
            numpy.array([0.0, 1.25, 1.75, 3.0]),
            numpy.array([0.0, 0.89, 1.11, 2.0]),
            numpy.array([[1.45, 1.45, 1.45], [1.45, 3.45, 1.45], [1.45, 1.45, 1.45]]),
            EMpy.modesolvers.FMM.Boundary(
                xleft="Electric Wall",
                yleft="Magnetic Wall",
                xright="Electric Wall",
                yright="Magnetic Wall",
            ),
        ).solve(1.55, 2, 1)
        fmm.modes[0].normalize()
        sens_fmm = _make_sensor(fmm.modes[0], dx_um, dy_um, nx_n, ny_n, a0, b0, 1.0)
        _, _, A_fmm, B_fmm, Pm_fmm, _ = FDTD_mod.overlap_f([sens_fmm], fmm, 0)
        self.assertAlmostEqual(A_fmm[0, 0], a0, delta=3e-2)
        self.assertAlmostEqual(B_fmm[0, 0], b0, delta=3e-2)
        self.assertAlmostEqual(Pm_fmm[0, 0], abs(a0) ** 2 - abs(b0) ** 2, delta=5e-2)

        # 3. Lossy kappa -> 0+ and 5-component rotated anisotropic dn -> 0 limits in overlap_f
        k_errs = []
        for kap in (1e-2, 1e-5, 0.0):
            sk = EMpy.modesolvers.FD.VFDModeSolver(
                wl, xg, yg, _eps_si(kap), "0000"
            ).solve(1, 1e-12)
            sens_k = _make_sensor(sk.modes[0], dx_um, dy_um, nx_n, ny_n, a0, b0, 1e-6)
            _, _, Ak, Bk, _, _ = FDTD_mod.overlap_f([sens_k], sk, 0)
            k_errs.append(abs(Ak[0, 0] - A[0, 0]) + abs(Bk[0, 0] - B[0, 0]))
        self.assertLess(k_errs[1], k_errs[0])
        self.assertLess(k_errs[2], 1e-9)

        ph = EMpy.utils.deg2rad(31.0)
        dn_errs = []
        for dn in (1e-4, 1e-8, 0.0):

            def _eps_ani(xc, yc, dn_=dn):
                core = (numpy.abs(xc[:, None] - 1.5e-6) <= 0.25e-6) & (
                    numpy.abs(yc[None, :] - 1.0e-6) <= 0.11e-6
                )
                n_o = numpy.where(core, 3.45, 1.45)
                n_e = numpy.where(core, 3.45 + dn_, 1.45 + dn_)
                exx = n_e**2 * numpy.cos(ph) ** 2 + n_o**2 * numpy.sin(ph) ** 2
                eyy = n_e**2 * numpy.sin(ph) ** 2 + n_o**2 * numpy.cos(ph) ** 2
                exy = (n_e**2 - n_o**2) * numpy.sin(ph) * numpy.cos(ph)
                return numpy.stack([exx, exy, exy, eyy, n_o**2], axis=-1)

            sa = EMpy.modesolvers.FD.VFDModeSolver(wl, xg, yg, _eps_ani, "0000").solve(
                1, 1e-12
            )
            sens_a = _make_sensor(sa.modes[0], dx_um, dy_um, nx_n, ny_n, a0, b0, 1e-6)
            _, _, Aa, Ba, _, _ = FDTD_mod.overlap_f([sens_a], sa, 0)
            dn_errs.append(abs(Aa[0, 0] - A[0, 0]) + abs(Ba[0, 0] - B[0, 0]))
        self.assertLess(dn_errs[1], dn_errs[0])
        self.assertLess(dn_errs[2], 1e-9)

    def test_vfd_nonuniform_anisotropic_xy_symmetry(self):
        """Verify exact x <-> y symmetry of VFDModeSolver on non-uniform anisotropic meshes."""
        wl = 1.55
        # Non-uniform asymmetric grids
        x = numpy.array([0.0, 0.12, 0.29, 0.55, 0.90, 1.40, 2.00])
        y = numpy.array([0.0, 0.18, 0.42, 0.72, 1.15, 1.75])
        nx, ny = len(x) - 1, len(y) - 1

        rng = numpy.random.default_rng(42)
        base_r = rng.uniform(2.0, 4.0, size=(nx, ny))
        base_i = -rng.uniform(0.0, 0.15, size=(nx, ny))
        exx = base_r + 1j * base_i
        eyy = rng.uniform(2.0, 4.0, size=(nx, ny)) - 1j * rng.uniform(
            0.0, 0.15, size=(nx, ny)
        )
        exy = rng.uniform(-0.6, 0.6, size=(nx, ny)) - 1j * rng.uniform(
            0.0, 0.05, size=(nx, ny)
        )
        eyx = rng.uniform(-0.6, 0.6, size=(nx, ny)) - 1j * rng.uniform(
            0.0, 0.05, size=(nx, ny)
        )
        ezz = rng.uniform(2.0, 4.0, size=(nx, ny)) - 1j * rng.uniform(
            0.0, 0.15, size=(nx, ny)
        )

        eps1 = numpy.stack([exx, exy, eyx, eyy, ezz], axis=-1)
        # Swap x <-> y: (exx, exy, eyx, eyy, ezz) -> (eyy.T, eyx.T, exy.T, exx.T, ezz.T)
        eps2 = numpy.stack([eyy.T, eyx.T, exy.T, exx.T, ezz.T], axis=-1)

        s1 = EMpy.modesolvers.FD.VFDModeSolver(wl, x, y, lambda xc, yc: eps1, "0000")
        s2 = EMpy.modesolvers.FD.VFDModeSolver(wl, y, x, lambda xc, yc: eps2, "0000")
        A1 = s1.build_matrix().toarray()
        A2 = s2.build_matrix().toarray()

        # Permutation swapping (Hx, Hy) and transposing vertex indices (i, j) -> (j, i)
        mx, my = len(x), len(y)
        npts = mx * my
        perm = numpy.empty(2 * npts, dtype=int)
        for i in range(mx):
            for j in range(my):
                idx1 = i * my + j
                idx2 = j * mx + i
                perm[idx2] = npts + idx1
                perm[npts + idx2] = idx1
        A1_rot = A1[numpy.ix_(perm, perm)]
        numpy.testing.assert_allclose(A2, A1_rot, rtol=1e-12, atol=1e-12)

        # Eigenvalue 90-degree rotational invariance on a non-uniform tilted uniaxial waveguide
        t = numpy.linspace(-1.0, 1.0, 21)
        g = 1.5 * numpy.sinh(1.2 * t) / numpy.sinh(1.2)
        phi = numpy.deg2rad(28.0)
        n_o, n_e = 2.4 - 1e-3j, 2.7 - 2e-3j
        n_cl = 1.45

        def _make_eps(rot):
            def _fn(xc, yc):
                if not rot:
                    core = (numpy.abs(xc[:, None]) <= 0.35) & (
                        numpy.abs(yc[None, :]) <= 0.22
                    )
                    ang = phi
                else:
                    core = (numpy.abs(xc[:, None]) <= 0.22) & (
                        numpy.abs(yc[None, :]) <= 0.35
                    )
                    ang = 0.5 * numpy.pi - phi
                no = numpy.where(core, n_o, n_cl)
                ne = numpy.where(core, n_e, n_cl)
                c, s = numpy.cos(ang), numpy.sin(ang)
                return numpy.stack(
                    [
                        ne**2 * c**2 + no**2 * s**2,
                        (ne**2 - no**2) * s * c,
                        (ne**2 - no**2) * s * c,
                        ne**2 * s**2 + no**2 * c**2,
                        no**2,
                    ],
                    axis=-1,
                )

            return _fn

        m1 = EMpy.modesolvers.FD.VFDModeSolver(
            wl, g, g, _make_eps(False), "0000"
        ).solve(2, 1e-10)
        m2 = EMpy.modesolvers.FD.VFDModeSolver(wl, g, g, _make_eps(True), "0000").solve(
            2, 1e-10
        )
        numpy.testing.assert_allclose(
            [m.neff for m in m1.modes],
            [m.neff for m in m2.modes],
            rtol=1e-10,
            atol=1e-10,
        )

    def test_fd_analytic_pec_waveguide_and_uniaxial_slab(self):
        """Verify VFDModeSolver and SVFDModeSolver against closed-form homogeneous waveguide & uniaxial slab dispersion."""
        wl = 1.55
        a, b = 2.0, 1.4
        x = numpy.linspace(0.0, a, 41)
        y = numpy.linspace(0.0, b, 29)

        # 1. Homogeneous dielectric box with Dirichlet ("0000") BCs:
        #    ghost-cell Dirichlet walls sit one step outside the vertex grid, so
        #    a_eff = a + 2*dx, b_eff = b + 2*dy, with exact neff = sqrt(n^2 - (wl/2a_eff)^2 - (wl/2b_eff)^2).
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        a_eff = a + 2.0 * dx
        b_eff = b + 2.0 * dy
        for n_mat in (1.5, 1.5 - 0.02j):
            neff_exact = numpy.sqrt(
                complex(n_mat) ** 2
                - (wl / (2.0 * a_eff)) ** 2
                - (wl / (2.0 * b_eff)) ** 2
            )
            eps_const = lambda xc, yc, e=n_mat**2: numpy.full(  # noqa: E731
                (len(xc), len(yc)), e
            )
            vfd = EMpy.modesolvers.FD.VFDModeSolver(wl, x, y, eps_const, "0000").solve(
                1, 1e-10
            )
            self.assertAlmostEqual(vfd.modes[0].neff, neff_exact, places=3)
            # FDMode.norm() must be strictly real positive so normalize() yields norm() == 1
            m0 = vfd.modes[0]
            self.assertEqual(numpy.imag(m0.norm()), 0.0)
            m0.normalize()
            self.assertAlmostEqual(float(m0.norm()), 1.0, places=12)
            self.assertAlmostEqual(abs(m0.overlap(m0)), 1.0, places=12)
            # SVFDModeSolver solves on cell centers (len-1), so ghost walls sit at a+dx, b+dy
            neff_exact_sv = numpy.sqrt(
                complex(n_mat) ** 2
                - (wl / (2.0 * (a + dx))) ** 2
                - (wl / (2.0 * (b + dy))) ** 2
            )
            for method in ("Ex", "Ey", "scalar"):
                svfd = EMpy.modesolvers.FD.SVFDModeSolver(
                    wl, x, y, eps_const, "0000", method=method
                ).solve(1, 1e-10)
                self.assertAlmostEqual(svfd.neff[0], neff_exact_sv, places=3)

        # 2. Uniaxial anisotropic dielectric slab (uniform in y, Dirichlet on x, 'A' on N/S so Hy is symmetric, k_y = 0):
        #    Core has diag(nE^2, nO^2, nO^2), cladding has nC^2.
        #    TM0 (Hy-polarized, Ex/Ez electric field) obeys exact uniaxial dispersion:
        #    (kappa_TM / nO^2) * tan(kappa_TM * h_half) == (gamma_TM / nC^2) * coth(gamma_TM * D_eff)
        #    where kappa_TM = sqrt(k0^2 * nO^2 - (nO^2 / nE^2) * beta^2).
        k0 = 2.0 * numpy.pi / wl
        h_half, D = 0.25, 1.25
        nO, nE, nC = 2.2, 2.8, 1.45
        x_slab = numpy.linspace(-(h_half + D), h_half + D, 121)
        y_slab = numpy.linspace(0.0, 0.2, 5)
        D_eff = D + (x_slab[1] - x_slab[0])

        def _eps_uniaxial_slab(xc, yc):
            core = numpy.abs(xc[:, None]) <= h_half
            exx = numpy.where(core, nE**2, nC**2) * numpy.ones((1, len(yc)))
            eyy = numpy.where(core, nO**2, nC**2) * numpy.ones((1, len(yc)))
            ezz = eyy.copy()
            z0 = numpy.zeros_like(exx)
            return numpy.stack([exx, z0, z0, eyy, ezz], axis=-1)

        # "AA00": antisymmetric Hx / symmetric Hy on N/S (allows k_y = 0 for Hy-polarized TM0), Dirichlet on E/W
        vfd_uni = EMpy.modesolvers.FD.VFDModeSolver(
            wl, x_slab, y_slab, _eps_uniaxial_slab, "AA00"
        ).solve(2, 1e-10)
        tm_mode = max(
            vfd_uni.modes,
            key=lambda m: float(numpy.max(numpy.abs(m.Hy)))
            / max(float(numpy.max(numpy.abs(m.Hx))), 1e-30),
        )
        beta = k0 * float(tm_mode.neff.real)
        kappa_tm = numpy.sqrt((k0 * nO) ** 2 - (nO**2 / nE**2) * beta**2)
        gamma_tm = numpy.sqrt(beta**2 - (k0 * nC) ** 2)
        lhs = (kappa_tm / nO**2) * numpy.tan(kappa_tm * h_half)
        rhs = (gamma_tm / nC**2) / numpy.tanh(gamma_tm * D_eff)
        self.assertAlmostEqual(lhs / rhs, 1.0, places=2)


class FMMModeSolverTest(TestCase):

    def test_fmm1d_uniform_lossy_tm(self):
        Uy = numpy.array([0.0, 2.0, 2.22, 4.22])
        n1 = 3.4757 - 1e-4j
        ny = numpy.array([n1, n1, n1])
        solver = EMpy.modesolvers.FMM.FMM1d(Uy, ny, "AA").solve(1.55, 2, "TM")
        self.assertEqual(len(solver.modes), 2)
        self.assertTrue(numpy.iscomplexobj(solver.modes[0].sl))
        self.assertTrue(numpy.isfinite(solver.modes[0].keff).all())
        y_grid = numpy.linspace(Uy[0], Uy[-1], 21)
        plt.figure()
        solver.modes[0].plot(y_grid)
        _, line_imag = plt.gca().get_lines()
        numpy.testing.assert_allclose(
            line_imag.get_ydata(), numpy.imag(solver.modes[0].eval(y_grid))
        )
        plt.close("all")

    def test_fmm_shooting_complex_delta(self):
        wl = 1.55
        k0 = 2.0 * numpy.pi / wl
        Uy = numpy.array([0.0, 2.0, 2.22, 4.22])
        n2 = 1.446
        n1 = 3.4757 - 1e-3j
        ny = numpy.array([n2, n1, n2])
        pars = {
            "beta": k0 * ny,
            "epsilon": ny**2,
            "Uy": Uy,
            "boundary": "AA",
        }
        kz_real = numpy.linspace(k0 * 1.5, k0 * 3.4, 5)
        delta_te, _ = EMpy.modesolvers.FMM.FMMshooting(kz_real, pars)
        delta_tm, _ = EMpy.modesolvers.FMM.FMMshootingTM(kz_real, pars)
        self.assertTrue(numpy.iscomplexobj(delta_te))
        self.assertTrue(numpy.iscomplexobj(delta_tm))
        self.assertGreater(numpy.max(numpy.abs(numpy.imag(delta_te))), 0.0)
        self.assertGreater(numpy.max(numpy.abs(numpy.imag(delta_tm))), 0.0)

        for pol, shoot_fn in [
            ("TE", EMpy.modesolvers.FMM.FMMshooting),
            ("TM", EMpy.modesolvers.FMM.FMMshootingTM),
        ]:
            sol = EMpy.modesolvers.FMM.FMM1d(Uy, ny, "AA").solve(wl, 1, pol)
            delta_root, _ = shoot_fn(sol.modes[0].keff, pars)
            self.assertLess(abs(delta_root[0]), 1e-4)

    def test_fmm2d_lossy_channel_waveguide(self):
        wl = 1.55
        Ux = numpy.array([0.0, 2.0, 2.4, 4.4])
        Uy = numpy.array([0.0, 2.0, 2.22, 4.22])
        boundary = EMpy.modesolvers.FMM.Boundary(
            xleft="Electric Wall",
            yleft="Magnetic Wall",
            xright="Electric Wall",
            yright="Magnetic Wall",
        )
        n2 = 1.446
        n1 = 3.4757 - 1e-4j
        refindex = numpy.array([[n2, n2, n2], [n2, n1, n2], [n2, n2, n2]])
        simul = EMpy.modesolvers.FMM.FMM2d(Ux, Uy, refindex, boundary).solve(wl, 2, 2)
        self.assertEqual(len(simul.modes), 2)
        self.assertTrue(numpy.isfinite(simul.modes[0].keff))
        self.assertTrue(numpy.isfinite(simul.modes[0].norm()))

        m0 = simul.modes[0]
        m0.normalize()
        ov_m0 = EMpy.modesolvers.interface.overlap(m0, m0, None, None)
        self.assertAlmostEqual(numpy.real(ov_m0), 1.0, places=10)
        self.assertLess(abs(numpy.imag(ov_m0)), 1e-4)
        O11, _, _, _ = EMpy.modesolvers.interface.interface_matrix(simul, simul)
        self.assertEqual(O11.shape, (2, 2))
        te_frac_old = m0.TEfrac_old(
            numpy.linspace(Ux[0], Ux[-1], 51), numpy.linspace(Uy[0], Uy[-1], 51)
        )
        te_frac = m0.TEfrac()
        self.assertTrue(0.0 <= te_frac <= 1.0)
        self.assertTrue(0.0 <= te_frac_old <= 1.0)
        self.assertAlmostEqual(te_frac, te_frac_old, delta=5e-2)
        m0.plot(numpy.linspace(Ux[0], Ux[-1], 11), numpy.linspace(Uy[0], Uy[-1], 11))
        plt.close("all")

        # get_fields_for_FDTD with non-square grid (len(x) != len(y))
        x_fdtd = numpy.linspace(Ux[0], Ux[-1], 9)
        y_fdtd = numpy.linspace(Uy[0], Uy[-1], 13)
        Ex_f, Ey_f, Ez_f, Hx_f, Hy_f, Hz_f = m0.get_fields_for_FDTD(x_fdtd, y_fdtd)
        self.assertEqual(Ex_f.shape, (len(x_fdtd) - 1, len(y_fdtd) - 2))
        self.assertEqual(Ey_f.shape, (len(x_fdtd) - 2, len(y_fdtd) - 1))
        self.assertEqual(Ez_f.shape, (len(x_fdtd) - 2, len(y_fdtd) - 2))
        self.assertEqual(Hx_f.shape, (len(x_fdtd) - 2, len(y_fdtd) - 1))
        self.assertEqual(Hy_f.shape, (len(x_fdtd) - 1, len(y_fdtd) - 2))
        self.assertEqual(Hz_f.shape, (len(x_fdtd) - 1, len(y_fdtd) - 1))
        # Direct evaluation at Ez grid points must match interpolated Ez_f closely
        _, _, Ez_direct, _, _, _ = m0.fields(x_fdtd[1:-1], y_fdtd[1:-1])
        numpy.testing.assert_allclose(Ez_f, Ez_direct.T, rtol=5e-2, atol=1e-3)

    def test_fmm2d_asymmetric_x_boundaries(self):
        """Mirrored (Electric, Magnetic) vs (Magnetic, Electric) x-walls on a symmetric waveguide must give identical neff."""
        wl = 1.55
        Ux = numpy.array([0.0, 0.1, 0.6, 0.7])
        Uy = numpy.array([0.0, 0.5, 1.0, 1.5])
        n2, n1 = 1.446, 3.4757
        refindex = numpy.array([[n2, n2, n2], [n2, n1, n2], [n2, n2, n2]])

        b_em = EMpy.modesolvers.FMM.Boundary(
            xleft="Electric Wall",
            yleft="Magnetic Wall",
            xright="Magnetic Wall",
            yright="Magnetic Wall",
        )
        b_me = EMpy.modesolvers.FMM.Boundary(
            xleft="Magnetic Wall",
            yleft="Magnetic Wall",
            xright="Electric Wall",
            yright="Magnetic Wall",
        )
        b_ee = EMpy.modesolvers.FMM.Boundary(
            xleft="Electric Wall",
            yleft="Magnetic Wall",
            xright="Electric Wall",
            yright="Magnetic Wall",
        )
        s_em = EMpy.modesolvers.FMM.FMM2d(Ux, Uy, refindex, b_em).solve(wl, 6, 1)
        s_me = EMpy.modesolvers.FMM.FMM2d(Ux, Uy, refindex, b_me).solve(wl, 6, 1)
        s_ee = EMpy.modesolvers.FMM.FMM2d(Ux, Uy, refindex, b_ee).solve(wl, 6, 1)
        numpy.testing.assert_allclose(s_em.modes[0].neff, s_me.modes[0].neff, rtol=1e-3)
        self.assertGreater(
            abs(s_em.modes[0].neff - s_ee.modes[0].neff),
            1e-3,
        )

    def test_fmm2d_overlap_tefrac_cross_waveguide_and_lossless_limit(self):
        """Verify FMMMode2d TEfrac, TM norm, cross-waveguide overlap, and kappa -> 0+ continuity."""
        wl = 1.55
        Ux1 = numpy.array([0.0, 1.0, 1.5, 2.5])
        Uy1 = numpy.array([0.0, 1.0, 1.22, 2.22])
        boundary = EMpy.modesolvers.FMM.Boundary(
            xleft="Electric Wall",
            yleft="Magnetic Wall",
            xright="Electric Wall",
            yright="Magnetic Wall",
        )
        rix_lossless = numpy.array(
            [[1.45, 1.45, 1.45], [1.45, 3.45, 1.45], [1.45, 1.45, 1.45]]
        )
        s0 = EMpy.modesolvers.FMM.FMM2d(Ux1, Uy1, rix_lossless, boundary).solve(
            wl, 2, 2
        )

        # Both quasi-TE (mode 0) and quasi-TM (mode 1) must have real positive norm and TEfrac in [0, 1]
        for m in s0.modes:
            n_val = m.norm()
            self.assertAlmostEqual(numpy.imag(n_val), 0.0, places=12)
            self.assertGreater(numpy.real(n_val), 0.0)
            m.normalize()

        xg = numpy.unique(
            numpy.r_[
                numpy.linspace(0.0, 1.0, 81),
                numpy.linspace(1.0001, 1.4999, 61),
                numpy.linspace(1.5, 2.5, 81),
            ]
        )
        yg = numpy.unique(
            numpy.r_[
                numpy.linspace(0.0, 1.0, 81),
                numpy.linspace(1.0001, 1.2199, 41),
                numpy.linspace(1.22, 2.22, 81),
            ]
        )

        self.assertGreater(s0.modes[0].TEfrac(), 0.9)
        self.assertLess(s0.modes[1].TEfrac(), 0.1)
        self.assertAlmostEqual(
            s0.modes[0].TEfrac(), s0.modes[0].TEfrac_old(xg, yg), delta=5e-3
        )
        self.assertAlmostEqual(
            s0.modes[1].TEfrac(), s0.modes[1].TEfrac_old(xg, yg), delta=5e-3
        )

        # Orthonormality on lossless waveguide: O11 == I_2
        O11, _, _, _ = EMpy.modesolvers.interface.interface_matrix(s0, s0)
        numpy.testing.assert_allclose(O11, numpy.eye(2), atol=1e-10)

        # Cross-waveguide overlap with distinct (Ux, Uy, n_core) must satisfy |O12| < 1 and match quadrature
        Ux2 = numpy.array([0.0, 0.7, 1.8, 2.5])
        Uy2 = numpy.array([0.0, 0.9, 1.32, 2.22])
        rix2 = numpy.array([[1.45, 1.45, 1.45], [1.45, 2.50, 1.45], [1.45, 1.45, 1.45]])
        s2 = EMpy.modesolvers.FMM.FMM2d(Ux2, Uy2, rix2, boundary).solve(wl, 2, 1)
        s2.modes[0].normalize()

        ov_ana = s0.modes[0].overlap(s2.modes[0])
        xg_both = numpy.unique(numpy.r_[xg, Ux2])
        yg_both = numpy.unique(numpy.r_[yg, Uy2])
        ov_quad = s0.modes[0].overlap_old(s2.modes[0], xg_both, yg_both)
        self.assertLess(abs(ov_ana), 1.0)
        self.assertGreater(ov_ana.real, 0.5)
        numpy.testing.assert_allclose(ov_ana, ov_quad, atol=5e-3)

        # Lossy -> non-lossy degenerate limit (kappa = 1e-3, 1e-6, 0.0+0j -> lossless)
        neff_errs = []
        tefrac_errs = []
        s_lossy_ref = None
        for kap in [1e-3, 1e-6, 0.0]:
            nc = complex(3.45, -kap)
            rk = numpy.array([[1.45, 1.45, 1.45], [1.45, nc, 1.45], [1.45, 1.45, 1.45]])
            sk = EMpy.modesolvers.FMM.FMM2d(Ux1, Uy1, rk, boundary).solve(wl, 2, 2)
            if kap > 0:
                self.assertLess(sk.modes[0].neff.imag, 0.0)
                self.assertLess(sk.modes[1].neff.imag, 0.0)
            sk.modes[0].normalize()
            sk.modes[1].normalize()
            if kap == 1e-3:
                s_lossy_ref = sk
            self.assertAlmostEqual(
                sk.modes[0].overlap(sk.modes[0]).real, 1.0, places=10
            )
            self.assertAlmostEqual(
                sk.modes[1].overlap(sk.modes[1]).real, 1.0, places=10
            )
            neff_errs.append(float(abs(sk.modes[0].neff - s0.modes[0].neff)))
            tefrac_errs.append(float(abs(sk.modes[0].TEfrac() - s0.modes[0].TEfrac())))

        self.assertLess(neff_errs[1], neff_errs[0])
        self.assertLess(neff_errs[2], 1e-8)
        self.assertLess(tefrac_errs[1], tefrac_errs[0])
        self.assertLess(tefrac_errs[2], 1e-8)

        # Independent Proof 1: Black-box 2D Gauss-Legendre integration of fields(x, y)
        # on smooth sub-cells shares zero code with __overlap / scalarprod / _refine_mode1d
        # and must match analytical (Sx, Sy) and overlap() to near machine precision (~1e-13).
        assert s_lossy_ref is not None
        xi, wi = numpy.polynomial.legendre.leggauss(18)

        def _gl_poynting(ma, mb):
            ux = numpy.unique(numpy.r_[Ux1, Ux2])
            uy = numpy.unique(numpy.r_[Uy1, Uy2])
            x_pts = numpy.concatenate(
                [
                    0.5 * (ux[i + 1] - ux[i]) * xi + 0.5 * (ux[i + 1] + ux[i])
                    for i in range(len(ux) - 1)
                ]
            )
            w_x = numpy.concatenate(
                [0.5 * (ux[i + 1] - ux[i]) * wi for i in range(len(ux) - 1)]
            )
            y_pts = numpy.concatenate(
                [
                    0.5 * (uy[j + 1] - uy[j]) * xi + 0.5 * (uy[j + 1] + uy[j])
                    for j in range(len(uy) - 1)
                ]
            )
            w_y = numpy.concatenate(
                [0.5 * (uy[j + 1] - uy[j]) * wi for j in range(len(uy) - 1)]
            )
            Ex1, Ey1, _, _, _, _ = ma.fields(x_pts, y_pts)
            _, _, _, cBx2, cBy2, _ = mb.fields(x_pts, y_pts)
            w2d = w_y[:, None] * w_x[None, :]
            return (
                0.5 * numpy.sum(w2d * Ex1 * numpy.conj(cBy2)),
                0.5 * numpy.sum(w2d * (-Ey1) * numpy.conj(cBx2)),
            )

        for ma, mb in [
            (s_lossy_ref.modes[0], s_lossy_ref.modes[0]),
            (s_lossy_ref.modes[1], s_lossy_ref.modes[1]),
            (s_lossy_ref.modes[0], s2.modes[0]),
            (s2.modes[0], s_lossy_ref.modes[1]),
        ]:
            sx_gl, sy_gl = _gl_poynting(ma, mb)
            numpy.testing.assert_allclose(
                ma.overlap(mb), sx_gl + sy_gl, rtol=1e-12, atol=1e-13
            )

        # Independent Proof 2: Pointwise Maxwell curl and divergence equations inside
        # lossy core and cladding cells (verifies exact du/dx consistency in eval()).
        k0 = 2.0 * numpy.pi / wl
        m_chk = s_lossy_ref.modes[0]
        kz = m_chk.keff
        h = 1e-6
        for x0, y0, n_cell in [
            (1.25, 1.11, 3.45 - 1e-3j),
            (0.45, 0.45, 1.45 + 0.0j),
        ]:
            eps_c = n_cell**2

            def _f_at(px, py):
                vals = m_chk.fields(numpy.array([px]), numpy.array([py]))
                return numpy.array([v[0, 0] for v in vals])

            f0 = _f_at(x0, y0)
            dfdx = (_f_at(x0 + h, y0) - _f_at(x0 - h, y0)) / (2.0 * h)
            dfdy = (_f_at(x0, y0 + h) - _f_at(x0, y0 - h)) / (2.0 * h)
            dfdz = -1j * kz * f0
            E_v, cB_v = f0[:3], f0[3:]
            curlE = numpy.array(
                [dfdy[2] - dfdz[1], dfdz[0] - dfdx[2], dfdx[1] - dfdy[0]]
            )
            curlB = numpy.array(
                [dfdy[5] - dfdz[4], dfdz[3] - dfdx[5], dfdx[4] - dfdy[3]]
            )
            divD = eps_c * (dfdx[0] + dfdy[1] + dfdz[2])
            divB = dfdx[3] + dfdy[4] + dfdz[5]
            self.assertLess(
                numpy.linalg.norm(curlE + 1j * k0 * cB_v)
                / numpy.linalg.norm(k0 * cB_v),
                1e-9,
            )
            self.assertLess(
                numpy.linalg.norm(curlB - 1j * k0 * eps_c * E_v)
                / numpy.linalg.norm(k0 * eps_c * E_v),
                1e-9,
            )
            self.assertLess(abs(divD) / numpy.linalg.norm(k0 * eps_c * E_v), 1e-9)
            self.assertLess(abs(divB) / numpy.linalg.norm(k0 * cB_v), 1e-9)

        # Independent Proof 3: Cross-solver agreement against VFDModeSolver on an interface-aligned grid.
        xfd = numpy.unique(
            numpy.r_[
                numpy.linspace(0.0, 1.0, 41),
                numpy.linspace(1.0, 1.5, 26),
                numpy.linspace(1.5, 2.5, 41),
            ]
        )
        yfd = numpy.unique(
            numpy.r_[
                numpy.linspace(0.0, 1.0, 41),
                numpy.linspace(1.0, 1.22, 16),
                numpy.linspace(1.22, 2.22, 41),
            ]
        )

        def _eps_vfd(x_, y_):
            return numpy.where(
                (x_[:, None] > 1.0)
                & (x_[:, None] < 1.5)
                & (y_[None, :] > 1.0)
                & (y_[None, :] < 1.22),
                3.45**2,
                1.45**2,
            )

        vfd_sol = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xfd, yfd, _eps_vfd, "0000"
        ).solve(2, 1e-10)
        self.assertAlmostEqual(
            s0.modes[0].TEfrac(), vfd_sol.modes[0].TEfrac(), delta=3e-2
        )
        self.assertAlmostEqual(
            s0.modes[1].TEfrac(), vfd_sol.modes[1].TEfrac(), delta=3e-2
        )
        xc_fd = EMpy.utils.centered1d(xfd)
        yc_fd = EMpy.utils.centered1d(yfd)
        for im in range(2):
            cross_solver_ov = abs(s0.modes[im].overlap(vfd_sol.modes[im], xc_fd, yc_fd))
            self.assertGreater(cross_solver_ov, 0.95)
            self.assertLessEqual(cross_solver_ov, 1.01)

    def test_fmm_scale_invariance_and_cross_solver_fd_overlap(self):
        """Verify FMM1d/FMM2d SI-meter scale invariance and bidirectional FDMode <-> FMMMode2d overlaps."""
        # 1. FMM1d exact scale invariance (microns vs SI meters, lossless and lossy)
        uy_um = numpy.array([0.0, 2.0, 2.22, 4.22])
        for kap in (0.0, 1e-3):
            ny = numpy.array([1.446, complex(3.4757, -kap), 1.446])
            for pol in ("TE", "TM"):
                s1_um = EMpy.modesolvers.FMM.FMM1d(uy_um, ny, "SS").solve(1.55, 2, pol)
                s1_m = EMpy.modesolvers.FMM.FMM1d(uy_um * 1e-6, ny, "SS").solve(
                    1.55e-6, 2, pol
                )
                n_um = [m.keff.item() / (2.0 * numpy.pi / 1.55) for m in s1_um.modes]
                n_m = [m.keff.item() / (2.0 * numpy.pi / 1.55e-6) for m in s1_m.modes]
                numpy.testing.assert_allclose(n_m, n_um, rtol=1e-12, atol=1e-13)

        # 2. FMM2d exact scale invariance in SI meters + kappa -> 0+ limit
        bnd = EMpy.modesolvers.FMM.Boundary(
            xleft="Electric Wall",
            yleft="Magnetic Wall",
            xright="Electric Wall",
            yright="Magnetic Wall",
        )
        ux_um = numpy.array([0.0, 1.0, 1.5, 2.5])
        uy2_um = numpy.array([0.0, 1.0, 1.22, 2.22])
        rix_ll = numpy.array(
            [[1.45, 1.45, 1.45], [1.45, 3.45, 1.45], [1.45, 1.45, 1.45]]
        )
        fmm_um = EMpy.modesolvers.FMM.FMM2d(ux_um, uy2_um, rix_ll, bnd).solve(
            1.55, 2, 2
        )
        fmm_m = EMpy.modesolvers.FMM.FMM2d(
            ux_um * 1e-6, uy2_um * 1e-6, rix_ll, bnd
        ).solve(1.55e-6, 2, 2)
        numpy.testing.assert_allclose(
            [m.neff for m in fmm_m.modes],
            [m.neff for m in fmm_um.modes],
            rtol=1e-12,
            atol=1e-13,
        )
        for m in fmm_m.modes:
            m.normalize()

        kap_errs = []
        for kap in (1e-3, 1e-6, 0.0):
            rk = numpy.array(
                [
                    [1.45, 1.45, 1.45],
                    [1.45, complex(3.45, -kap), 1.45],
                    [1.45, 1.45, 1.45],
                ]
            )
            sk = EMpy.modesolvers.FMM.FMM2d(ux_um * 1e-6, uy2_um * 1e-6, rk, bnd).solve(
                1.55e-6, 2, 1
            )
            kap_errs.append(abs(sk.modes[0].neff - fmm_m.modes[0].neff))
        self.assertLess(kap_errs[1], kap_errs[0])
        self.assertLess(kap_errs[2], 1e-12)

        # 3. Bidirectional FDMode <-> FMMMode2d overlap and SimpleJoint in SI meters (+ 5-comp aniso dn -> 0)
        xfd = numpy.unique(
            numpy.r_[
                numpy.linspace(0.0, 1.0e-6, 21),
                numpy.linspace(1.0e-6, 1.5e-6, 16),
                numpy.linspace(1.5e-6, 2.5e-6, 21),
            ]
        )
        yfd = numpy.unique(
            numpy.r_[
                numpy.linspace(0.0, 1.0e-6, 21),
                numpy.linspace(1.0e-6, 1.22e-6, 11),
                numpy.linspace(1.22e-6, 2.22e-6, 21),
            ]
        )
        ph = EMpy.utils.deg2rad(27.0)

        def _eps_vfd_si(dn=0.0):
            def _f(x_, y_):
                core = (
                    (x_[:, None] > 1.0e-6)
                    & (x_[:, None] < 1.5e-6)
                    & (y_[None, :] > 1.0e-6)
                    & (y_[None, :] < 1.22e-6)
                )
                nO = numpy.where(core, 3.45, 1.45)
                nE = numpy.where(core, 3.45 + dn, 1.45 + dn)
                exx = nE**2 * numpy.cos(ph) ** 2 + nO**2 * numpy.sin(ph) ** 2
                eyy = nE**2 * numpy.sin(ph) ** 2 + nO**2 * numpy.cos(ph) ** 2
                exy = (nE**2 - nO**2) * numpy.sin(ph) * numpy.cos(ph)
                return numpy.stack([exx, exy, exy, eyy, nO**2], axis=-1)

            return _f

        vfd_iso = EMpy.modesolvers.FD.VFDModeSolver(
            1.55e-6,
            xfd,
            yfd,
            lambda x_, y_: _eps_vfd_si(0.0)(x_, y_)[:, :, 0],
            "0000",
        ).solve(2, 1e-10)
        ov_fd_to_fmm = abs(vfd_iso.modes[0].overlap(fmm_m.modes[0]))
        ov_fmm_to_fd = abs(fmm_m.modes[0].overlap(vfd_iso.modes[0]))
        self.assertGreater(ov_fd_to_fmm, 0.90)
        self.assertGreater(ov_fmm_to_fd, 0.90)
        sj_cross = EMpy.modesolvers.geometries.SimpleJoint(
            vfd_iso, fmm_m, method="passive"
        )
        self.assertTrue(
            numpy.all(numpy.linalg.svd(sj_cross.S, compute_uv=False) <= 1.0)
        )

        dn_errs = []
        for dn in (1e-4, 1e-8, 0.0):
            vfd_a = EMpy.modesolvers.FD.VFDModeSolver(
                1.55e-6, xfd, yfd, _eps_vfd_si(dn), "0000"
            ).solve(2, 1e-10)
            dn_errs.append(
                abs(abs(vfd_a.modes[0].overlap(fmm_m.modes[0])) - ov_fd_to_fmm)
            )
        self.assertLess(dn_errs[1], dn_errs[0])
        self.assertLess(dn_errs[2], 1e-9)

    def test_fmm_shooting_multilayer_right_branch(self):
        """Verify FMMshooting/FMMshootingTM propagate continuity when >= 2 layers lie right of peak index."""
        wl = 1.55
        # Splitting the right cladding of a 3-layer slab into 2 layers (4 total) must leave all modes invariant
        # and satisfy the exact finite-box transcendental equation k_c * tan(k_c * a) = gamma * coth(gamma * D)
        a, D = 0.15, 1.0
        Uy3 = numpy.array([0.0, D, D + 2.0 * a, 2.0 * D + 2.0 * a])
        ny3 = numpy.array([1.444, 3.476, 1.444])
        Uy4 = numpy.array([0.0, D, D + 2.0 * a, D + 2.0 * a + 0.4, 2.0 * D + 2.0 * a])
        ny4 = numpy.array([1.444, 3.476, 1.444, 1.444])
        k0 = 2.0 * numpy.pi / wl
        for pol in ("TE", "TM"):
            m3 = EMpy.modesolvers.FMM.FMM1d_y(Uy3, ny3, wl, 4, "AA", pol)
            m4 = EMpy.modesolvers.FMM.FMM1d_y(Uy4, ny4, wl, 4, "AA", pol)
            numpy.testing.assert_allclose(
                [m.keff for m in m4], [m.keff for m in m3], rtol=1e-12, atol=1e-12
            )
            kz0 = float(m4[0].keff[0].real)
            kc = numpy.sqrt((k0 * 3.476) ** 2 - kz0**2)
            gam = numpy.sqrt(kz0**2 - (k0 * 1.444) ** 2)
            if pol == "TE":
                lhs = kc * numpy.tan(kc * a)
                rhs = gam / numpy.tanh(gam * D)
            else:
                lhs = (kc / 3.476**2) * numpy.tan(kc * a)
                rhs = (gam / 1.444**2) / numpy.tanh(gam * D)
            self.assertAlmostEqual(lhs / rhs, 1.0, places=12)

        # Genuine 4-layer slab (air / Si / SiN / SiO2) checked against exact 2x2 Abeles transfer matrix
        Uy_4mat = numpy.array([0.0, 0.8, 1.1, 1.5, 2.5])
        ny_4mat = numpy.array([1.0, 3.476, 2.0, 1.444])
        d_4mat = numpy.diff(Uy_4mat)
        for pol in ("TE", "TM"):
            modes_4 = EMpy.modesolvers.FMM.FMM1d_y(Uy_4mat, ny_4mat, wl, 4, "AA", pol)
            for mode in modes_4:
                kz = mode.keff[0]
                M = numpy.eye(2, dtype=complex)
                for dj, nj in zip(d_4mat, ny_4mat, strict=True):
                    kj = numpy.emath.sqrt((k0 * nj) ** 2 - kz**2)
                    c = numpy.cos(kj * dj)
                    s_over_k = numpy.sinc(kj * dj / numpy.pi) * dj
                    s_times_k = numpy.sin(kj * dj) * kj
                    if pol == "TE":
                        Mj = numpy.array(
                            [[c, s_over_k], [-s_times_k, c]], dtype=complex
                        )
                    else:
                        ej = nj**2
                        Mj = numpy.array(
                            [[c, ej * s_over_k], [-s_times_k / ej, c]], dtype=complex
                        )
                    M = Mj @ M
                self.assertLess(abs(M[0, 1]) / numpy.linalg.norm(M), 1e-8)

        # Asymmetric lossy 3-layer slab with peak index at layer 0 must match its mirror image (peak at layer 2)
        Uy_a = numpy.array([0.0, 0.4, 1.0, 2.0])
        ny_a = numpy.array([3.476 - 1e-3j, 2.0 - 5e-4j, 1.444])
        Uy_b = numpy.array([0.0, 1.0, 1.6, 2.0])
        ny_b = ny_a[::-1]
        for pol in ("TE", "TM"):
            ma = EMpy.modesolvers.FMM.FMM1d_y(Uy_a, ny_a, wl, 4, "AS", pol)
            mb = EMpy.modesolvers.FMM.FMM1d_y(Uy_b, ny_b, wl, 4, "SA", pol)
            numpy.testing.assert_allclose(
                [m.keff for m in ma], [m.keff for m in mb], rtol=1e-12, atol=1e-12
            )

    def test_fmm2d_multislice_mvec_and_exact_coupled_slab(self):
        """Verify FMM2d Mvec and check_matching for Nslices >= 4 against exact Abeles dispersion."""
        wl = 1.55
        k0 = 2.0 * numpy.pi / wl
        b_ee = EMpy.modesolvers.FMM.Boundary(
            xleft="Electric Wall",
            xright="Electric Wall",
            yleft="Electric Wall",
            yright="Electric Wall",
        )

        # 1. Genuine 5-slice coupled-slab (clad/core/gap/core/clad, uniform in y):
        #    with Electric Walls on y, k_y = 0 Ey-polarized supermodes match exact 1D TE Abeles dispersion.
        Uy_u = numpy.array([0.0, 1.0])
        Ux_dc = numpy.array([0.0, 0.7, 1.0, 1.2, 1.5, 2.2])
        nx_dc = [1.444, 3.476, 1.444, 3.476, 1.444]
        s_dc = EMpy.modesolvers.FMM.FMM2d(
            Ux_dc, Uy_u, numpy.array([[n] for n in nx_dc]), b_ee
        ).solve(wl, 2, 2)
        d_dc = numpy.diff(Ux_dc)
        for mode in s_dc.modes:
            kz = mode.keff
            M = numpy.eye(2, dtype=complex)
            for dj, nj in zip(d_dc, nx_dc, strict=True):
                kj = numpy.emath.sqrt((k0 * nj) ** 2 - kz**2)
                c = numpy.cos(kj * dj)
                s_over_k = numpy.sinc(kj * dj / numpy.pi) * dj
                s_times_k = numpy.sin(kj * dj) * kj
                M = numpy.array([[c, s_over_k], [-s_times_k, c]], dtype=complex) @ M
            self.assertLess(abs(M[0, 1]) / numpy.linalg.norm(M), 1e-10)

        # 2. 4-slice 2D channel waveguide (split core) must match the 3-slice waveguide identically
        Uy = numpy.array([0.0, 0.8, 1.02, 1.82])
        ny_cl = [1.444, 1.444, 1.444]
        ny_co = [1.444, 3.476, 1.444]
        s3 = EMpy.modesolvers.FMM.FMM2d(
            numpy.array([0.0, 0.8, 1.28, 2.08]),
            Uy,
            numpy.array([ny_cl, ny_co, ny_cl]),
            b_ee,
        ).solve(wl, 3, 1)
        s4 = EMpy.modesolvers.FMM.FMM2d(
            numpy.array([0.0, 0.8, 1.0, 1.28, 2.08]),
            Uy,
            numpy.array([ny_cl, ny_co, ny_co, ny_cl]),
            b_ee,
        ).solve(wl, 3, 1)
        numpy.testing.assert_allclose(
            s4.modes[0].neff, s3.modes[0].neff, rtol=1e-12, atol=1e-12
        )

    def test_fmm_evanescent_wide_cladding_exact_fields_and_overlap(self):
        """Verify two-endpoint evanescent evaluation against exact closed-form slab fields for |Im(k*d)| > 50..100."""
        wl = 1.55
        k0 = 2.0 * numpy.pi / wl

        # 1. 1D wide-cladding slab (|Im(k*D)| ~ 60): exact sinh(gamma*s)/sinh(gamma*D) cladding field
        D, a = 5.0, 0.25
        Uy = numpy.array([0.0, D, D + 2.0 * a, 2.0 * D + 2.0 * a])
        ny = numpy.array([1.444, 3.476, 1.444])
        m1d = EMpy.modesolvers.FMM.FMM1d_y(Uy, ny, wl, 1, "AA", "TE")[0]
        kz = float(m1d.keff[0].real)
        gam = numpy.sqrt(kz**2 - (k0 * 1.444) ** 2)
        kc = numpy.sqrt((k0 * 3.476) ** 2 - kz**2)
        self.assertGreater(gam * D, 55.0)
        self.assertEqual(abs(m1d.eval(0.0)[0]), 0.0)
        self.assertEqual(abs(m1d.eval(Uy[-1])[0]), 0.0)
        y_grid = numpy.linspace(0.0, Uy[-1], 401)
        u_num = m1d.eval(y_grid)
        u_peak = m1d.eval(D + a)[0]
        u_exact = numpy.where(
            y_grid < D,
            u_peak
            * numpy.cos(kc * a)
            * numpy.exp(-gam * (D - y_grid))
            * (1.0 - numpy.exp(-2.0 * gam * y_grid))
            / (1.0 - numpy.exp(-2.0 * gam * D)),
            numpy.where(
                y_grid <= D + 2.0 * a,
                u_peak * numpy.cos(kc * (y_grid - (D + a))),
                u_peak
                * numpy.cos(kc * a)
                * numpy.exp(-gam * (y_grid - (D + 2.0 * a)))
                * (1.0 - numpy.exp(-2.0 * gam * (Uy[-1] - y_grid)))
                / (1.0 - numpy.exp(-2.0 * gam * D)),
            ),
        )
        numpy.testing.assert_allclose(
            u_num / u_peak, u_exact / u_peak, rtol=1e-12, atol=1e-13
        )

        # 2. 2D wide-cladding slab (Ux = [0, 6, 6.4, 12.4], n1d = 12 => max|Im(kx*d)| > 100):
        #    exact closed-form Ey(x) profile and positive unit self-overlap after normalize()
        b_ee = EMpy.modesolvers.FMM.Boundary(
            xleft="Electric Wall",
            xright="Electric Wall",
            yleft="Electric Wall",
            yright="Electric Wall",
        )
        Ux = numpy.array([0.0, 6.0, 6.4, 12.4])
        Uy_u = numpy.array([0.0, 1.0])
        s_slab = EMpy.modesolvers.FMM.FMM2d(
            Ux, Uy_u, numpy.array([[1.444], [3.476], [1.444]]), b_ee
        ).solve(wl, 12, 1)
        ms = s_slab.modes[0]
        ms.normalize()
        self.assertAlmostEqual(float(ms.norm()), 1.0, places=12)
        ov_ana = ms.overlap(ms)
        self.assertAlmostEqual(float(numpy.real(ov_ana)), 1.0, places=11)

        xg = numpy.concatenate(
            [
                numpy.linspace(0.0, 6.0, 101),
                numpy.linspace(6.0, 6.4, 81)[1:],
                numpy.linspace(6.4, 12.4, 101)[1:],
            ]
        )
        yg = numpy.linspace(0.0, 1.0, 11)
        ov_num = ms.overlap(ms, xg, yg)
        self.assertAlmostEqual(float(numpy.real(ov_num)), 1.0, places=1)

        _, Ey, _, _, _, _ = ms.fields(xg, yg)
        Ey_1d = Ey[5, :]
        kc2 = numpy.sqrt((k0 * 3.476) ** 2 - float(ms.keff.real) ** 2)
        gam2 = numpy.sqrt(float(ms.keff.real) ** 2 - (k0 * 1.444) ** 2)
        E0 = Ey_1d[numpy.argmin(numpy.abs(xg - 6.2))]
        Ey_exact = numpy.where(
            xg < 6.0,
            E0
            * numpy.cos(kc2 * 0.2)
            * numpy.exp(-gam2 * (6.0 - xg))
            * (1.0 - numpy.exp(-2.0 * gam2 * xg))
            / (1.0 - numpy.exp(-2.0 * gam2 * 6.0)),
            numpy.where(
                xg <= 6.4,
                E0 * numpy.cos(kc2 * (xg - 6.2)),
                E0
                * numpy.cos(kc2 * 0.2)
                * numpy.exp(-gam2 * (xg - 6.4))
                * (1.0 - numpy.exp(-2.0 * gam2 * (12.4 - xg)))
                / (1.0 - numpy.exp(-2.0 * gam2 * 6.0)),
            ),
        )
        numpy.testing.assert_allclose(Ey_1d / E0, Ey_exact / E0, rtol=1e-12, atol=1e-13)

    def test_fmm_shooting_interface_node_analytic_continuity(self):
        """Verify FMMshooting/FMMshootingTM match derivative continuously when a mode has a node at Uy[n1]."""
        wl = 1.55
        k0 = 2.0 * numpy.pi / wl
        n1, n2 = 2.0, 3.0
        kz = 1.5 * k0
        k1 = numpy.sqrt((k0 * n1) ** 2 - kz**2)
        k2 = numpy.sqrt((k0 * n2) ** 2 - kz**2)
        d1 = numpy.pi / k1
        d2 = numpy.pi / k2
        Uy = numpy.array([0.0, d1, d1 + d2])
        ny = numpy.array([n1, n2])
        pars = {
            "epsilon": ny**2,
            "beta": ny * k0,
            "boundary": "AA",
            "Uy": Uy,
        }

        # TE: exact mode is sin(k1*y)/k1 on [0, d1] and -sin(k2*(y-d1))/k2 on [d1, d1+d2]
        delta_te, m_te = EMpy.modesolvers.FMM.FMMshooting(kz, pars)
        self.assertLess(abs(delta_te[0]), 1e-12)
        self.assertAlmostEqual(m_te.sr[0], m_te.sl[1], places=12)
        self.assertAlmostEqual(m_te.ar[0], m_te.al[1], places=12)
        y_g = numpy.linspace(0.0, d1 + d2, 201)
        u_te_exact = numpy.where(
            y_g <= d1,
            numpy.sin(k1 * y_g) / k1,
            -numpy.sin(k2 * (y_g - d1)) / k2,
        )
        numpy.testing.assert_allclose(
            m_te.eval(y_g), u_te_exact, rtol=1e-12, atol=1e-13
        )

        # TM: continuous u and u'/eps => right branch scales by eps2 / eps1
        delta_tm, m_tm = EMpy.modesolvers.FMM.FMMshootingTM(kz, pars)
        self.assertLess(abs(delta_tm[0]), 1e-12)
        self.assertAlmostEqual(m_tm.sr[0], m_tm.sl[1], places=12)
        self.assertAlmostEqual(m_tm.ar[0] / n1**2, m_tm.al[1] / n2**2, places=12)
        u_tm_exact = numpy.where(
            y_g <= d1,
            numpy.sin(k1 * y_g) / k1,
            -(n2**2 / n1**2) * numpy.sin(k2 * (y_g - d1)) / k2,
        )
        numpy.testing.assert_allclose(
            m_tm.eval(y_g), u_tm_exact, rtol=1e-12, atol=1e-13
        )

    def test_fmm1d_passive_cutoff_branch_and_lossy_limit(self):
        """Verify analyticalsolution and FMM1d_y select passive cutoff branch Re(keff) - Im(keff) >= 0 continuously as kappa -> 0+."""
        wl = 1.55
        k0 = 2.0 * numpy.pi / wl
        L = 1.0
        Uy_u = numpy.array([0.0, L])

        # 1. Uniform slice (analyticalsolution): exact closed-form keff = sqrt((k0*n)^2 - (m*pi/L)^2)
        #    on passive branch Re(keff) - Im(keff) >= 0 with no sign jump at kappa -> 0+.
        keff_ll = numpy.array(
            [
                m.keff[0]
                for m in EMpy.modesolvers.FMM.FMM1d_y(
                    Uy_u, numpy.array([1.5]), wl, 4, "AA", "TE"
                )
            ]
        )
        for m_idx in range(1, 5):
            val = (k0 * 1.5) ** 2 - (m_idx * numpy.pi / L) ** 2
            expected = numpy.sqrt(val) if val >= 0 else -1j * numpy.sqrt(-val)
            self.assertAlmostEqual(keff_ll[m_idx - 1], expected, places=12)

        errs = []
        for kap in (1e-2, 1e-6, 1e-12):
            keff_k = numpy.array(
                [
                    m.keff[0]
                    for m in EMpy.modesolvers.FMM.FMM1d_y(
                        Uy_u, numpy.array([1.5 - 1j * kap]), wl, 4, "AA", "TE"
                    )
                ]
            )
            self.assertTrue(numpy.all(keff_k.real - keff_k.imag >= -1e-12))
            self.assertTrue(numpy.all(keff_k.imag <= 1e-12))
            errs.append(float(numpy.max(numpy.abs(keff_k - keff_ll))))
        self.assertLess(errs[1], errs[0])
        self.assertLess(errs[2], errs[1])
        self.assertLess(errs[2], 1e-9)

        # 2. Non-uniform lossy slice (FMM1d_y): cutoff modes stay in passive Quadrant IV (Re >= 0, Im <= 0)
        Uy_2 = numpy.array([0.0, 0.75, 1.5])
        ny_2 = numpy.array([1.5 - 0.01j, 2.0 - 0.01j])
        modes_2 = EMpy.modesolvers.FMM.FMM1d_y(Uy_2, ny_2, wl, 5, "AA", "TE")
        keff_2 = numpy.array([m.keff[0] for m in modes_2])
        self.assertTrue(numpy.all(keff_2.real >= -1e-12))
        self.assertTrue(numpy.all(keff_2.imag <= 1e-12))
        self.assertTrue(numpy.all(keff_2.real - keff_2.imag >= -1e-12))

    def test_fmm_findzerosnew_single_zero_and_lossy_guided_mode(self):
        """Verify findzerosnew handles single-zero scan windows and broad lossy guided-mode minima."""
        wl = 1.55
        k0 = 2.0 * numpy.pi / wl

        def _abeles_te_res(kz, Uy, ny):
            M = numpy.eye(2, dtype=complex)
            for dj, nj in zip(numpy.diff(Uy), ny, strict=True):
                kj = numpy.emath.sqrt((k0 * nj) ** 2 - kz**2)
                c = numpy.cos(kj * dj)
                s_over_k = numpy.sinc(kj * dj / numpy.pi) * dj
                s_times_k = numpy.sin(kj * dj) * kj
                M = numpy.array([[c, s_over_k], [-s_times_k, c]], dtype=complex) @ M
            return abs(M[0, 1]) / numpy.linalg.norm(M)

        # 1. Single-mode 2-layer slab (len(zeri) == 1 in findzerosnew, previously ValueError)
        Uy1 = numpy.array([0.0, 0.3, 0.6])
        ny1 = numpy.array([1.444, 2.5])
        m1 = EMpy.modesolvers.FMM.FMM1d_y(Uy1, ny1, wl, 2, "AA", "TE")
        self.assertLess(_abeles_te_res(m1[0].keff[0], Uy1, ny1), 1e-12)

        # 2. Moderately lossy 1D slab (kappa = 0.01, previously missed fundamental guided mode)
        Uy2 = numpy.array([0.0, 0.75, 1.5])
        ny2 = numpy.array([1.5 - 0.01j, 2.0 - 0.01j])
        m2 = EMpy.modesolvers.FMM.FMM1d_y(Uy2, ny2, wl, 2, "AA", "TE")
        kz0 = m2[0].keff[0]
        self.assertGreater(kz0.real / k0, 1.8)
        self.assertLess(kz0.imag, 0.0)
        self.assertLess(_abeles_te_res(kz0, Uy2, ny2), 1e-12)


class GeometriesTest(TestCase):

    def test_s2t_t2s_roundtrip_and_generic_device(self):
        rng = numpy.random.default_rng(0)
        s = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        t = EMpy.modesolvers.geometries.S2T(s)
        s_rt = EMpy.modesolvers.geometries.T2S(t)
        numpy.testing.assert_allclose(s_rt, s)

        # GenericDevice whose first element has no .solver attribute (like SimpleJoint)
        class _DummyDevice:
            def __init__(self, mat):
                self.T = mat
                self.inputLHS = numpy.array([1.0, 0.0])
                self.inputRHS = numpy.array([0.0, 0.0])

        d1 = _DummyDevice(numpy.eye(4, dtype=complex))
        d2 = _DummyDevice(numpy.diag([1.0, 1.0, 1.0, 1.0]).astype(complex))
        gd = EMpy.modesolvers.geometries.GenericDevice([d1, d2])
        numpy.testing.assert_allclose(gd.T, numpy.eye(4, dtype=complex))
        numpy.testing.assert_allclose(gd.outputRHS, numpy.array([1.0, 0.0]))

    def test_simple_joint_exact_fresnel_te10_lossless_and_lossy(self):
        """Verify SimpleJoint reproduces exact complex TE10 reflection/transmission to ~1e-14."""
        wl = 1.55e-6
        x = numpy.linspace(0.0, 2.0e-6, 31)
        y = numpy.linspace(0.0, 1.3e-6, 21)

        for n1, n2 in [(1.5, 2.4), (1.5, 2.4 - 0.5j), (2.1 - 0.1j, 1.4 - 0.3j)]:
            s1 = EMpy.modesolvers.FD.VFDModeSolver(
                wl,
                x,
                y,
                lambda x_, y_, n=n1: numpy.full((len(x_), len(y_)), n**2),
                "SSAA",
            ).solve(1, 1e-12)
            s2 = EMpy.modesolvers.FD.VFDModeSolver(
                wl,
                x,
                y,
                lambda x_, y_, n=n2: numpy.full((len(x_), len(y_)), n**2),
                "SSAA",
            ).solve(1, 1e-12)
            sj = EMpy.modesolvers.geometries.SimpleJoint(s1, s2)
            ne1, ne2 = s1.modes[0].neff, s2.modes[0].neff
            r_exact = (ne1 - ne2) / (ne1 + ne2)
            numpy.testing.assert_allclose(sj.S[0, 0], r_exact, rtol=1e-12, atol=1e-13)
            numpy.testing.assert_allclose(sj.S[1, 1], -r_exact, rtol=1e-12, atol=1e-13)
            if numpy.isreal(n1) and numpy.isreal(n2):
                t_exact = 2.0 * numpy.sqrt(ne1 * ne2) / (ne1 + ne2)
                numpy.testing.assert_allclose(
                    abs(sj.S[1, 0]), abs(t_exact), rtol=1e-12, atol=1e-13
                )
                numpy.testing.assert_allclose(
                    sj.S.conj().T @ sj.S, numpy.eye(2), atol=1e-13
                )

    def test_simple_joint_passivity_reciprocity_and_degenerate_limits(self):
        """Verify SimpleJoint passivity, reciprocity, swap symmetry, and aniso/lossy -> iso/lossless limits."""
        wl = 1.55e-6
        xg = numpy.linspace(0.0, 3.0e-6, 41)
        yg = numpy.linspace(0.0, 2.0e-6, 31)

        def _eps_iso(w, kap=0.0):
            def _f(x_, y_):
                nc = complex(3.45, -kap) if kap > 0 else 3.45
                ncl = complex(1.45, -0.2 * kap) if kap > 0 else 1.45
                return numpy.where(
                    (numpy.abs(x_[:, None] - 1.5e-6) <= w / 2.0)
                    & (numpy.abs(y_[None, :] - 1.0e-6) <= 0.11e-6),
                    nc**2,
                    ncl**2,
                )

            return _f

        def _eps_aniso(w, dn=0.0, phi_deg=33.0):
            ph = EMpy.utils.deg2rad(phi_deg)

            def _f(x_, y_):
                core = (numpy.abs(x_[:, None] - 1.5e-6) <= w / 2.0) & (
                    numpy.abs(y_[None, :] - 1.0e-6) <= 0.11e-6
                )
                nO = numpy.where(core, 3.45, 1.45)
                nE = numpy.where(core, 3.45 + dn, 1.45 + dn)
                exx = nE**2 * numpy.cos(ph) ** 2 + nO**2 * numpy.sin(ph) ** 2
                eyy = nE**2 * numpy.sin(ph) ** 2 + nO**2 * numpy.cos(ph) ** 2
                exy = (nE**2 - nO**2) * numpy.sin(ph) * numpy.cos(ph)
                ezz = nO**2
                return numpy.stack([exx, exy, exy, eyy, ezz], axis=-1)

            return _f

        sn = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xg, yg, _eps_iso(0.4e-6), "0000"
        ).solve(1, 1e-10)
        sw = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xg, yg, _eps_iso(1.6e-6), "0000"
        ).solve(1, 1e-10)

        # Identical waveguide butt-joint must have zero reflection and unit transmission
        sj_same = EMpy.modesolvers.geometries.SimpleJoint(sn, sn)
        numpy.testing.assert_allclose(
            numpy.abs(sj_same.S), numpy.array([[0.0, 1.0], [1.0, 0.0]]), atol=1e-12
        )

        # Step discontinuity: strict passivity (||S||_2 < 1 due to radiation loss), reciprocity, swap symmetry
        sj_nw = EMpy.modesolvers.geometries.SimpleJoint(sn, sw)
        sj_wn = EMpy.modesolvers.geometries.SimpleJoint(sw, sn)
        sv_nw = numpy.linalg.svd(sj_nw.S, compute_uv=False)
        self.assertTrue(numpy.all(sv_nw < 1.0))
        self.assertLess(abs(sj_nw.S[0, 0]) ** 2 + abs(sj_nw.S[1, 0]) ** 2, 1.0)
        self.assertLess(abs(sj_nw.S[1, 1]) ** 2 + abs(sj_nw.S[0, 1]) ** 2, 1.0)
        self.assertAlmostEqual(abs(sj_nw.S[0, 1]), abs(sj_nw.S[1, 0]), places=12)
        numpy.testing.assert_allclose(
            numpy.abs(sj_nw.S),
            numpy.abs(sj_wn.S[[1, 0], :][:, [1, 0]]),
            atol=1e-12,
        )

        # Multi-mode junction with decoupled symmetry modes must remain passive and non-singular
        sn2 = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xg, yg, _eps_iso(0.4e-6), "0000"
        ).solve(2, 1e-10)
        sw2 = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xg, yg, _eps_iso(1.6e-6), "0000"
        ).solve(2, 1e-10)
        sj2 = EMpy.modesolvers.geometries.SimpleJoint(sn2, sw2)
        self.assertTrue(numpy.all(numpy.linalg.svd(sj2.S, compute_uv=False) <= 1.0))

        # Anisotropic 5-component rotated tensor -> isotropic limit (dn = 1e-4, 1e-8, 0)
        aniso_errs = []
        for dn in [1e-4, 1e-8, 0.0]:
            sna = EMpy.modesolvers.FD.VFDModeSolver(
                wl, xg, yg, _eps_aniso(0.4e-6, dn), "0000"
            ).solve(1, 1e-10)
            swa = EMpy.modesolvers.FD.VFDModeSolver(
                wl, xg, yg, _eps_aniso(1.6e-6, dn), "0000"
            ).solve(1, 1e-10)
            sja = EMpy.modesolvers.geometries.SimpleJoint(sna, swa)
            self.assertTrue(numpy.all(numpy.linalg.svd(sja.S, compute_uv=False) <= 1.0))
            self.assertAlmostEqual(abs(sja.S[0, 1]), abs(sja.S[1, 0]), places=11)
            aniso_errs.append(
                float(numpy.max(numpy.abs(numpy.abs(sja.S) - numpy.abs(sj_nw.S))))
            )
        self.assertLess(aniso_errs[1], aniso_errs[0])
        self.assertLess(aniso_errs[2], 1e-10)

        # Lossy -> non-lossy limit (kappa = 1e-2, 1e-5, 0.0+0j -> lossless)
        lossy_errs = []
        for kap in [1e-2, 1e-5, 0.0]:
            snk = EMpy.modesolvers.FD.VFDModeSolver(
                wl, xg, yg, _eps_iso(0.4e-6, kap), "0000"
            ).solve(1, 1e-10)
            swk = EMpy.modesolvers.FD.VFDModeSolver(
                wl, xg, yg, _eps_iso(1.6e-6, kap), "0000"
            ).solve(1, 1e-10)
            sjk = EMpy.modesolvers.geometries.SimpleJoint(snk, swk)
            self.assertTrue(numpy.all(numpy.linalg.svd(sjk.S, compute_uv=False) <= 1.0))
            self.assertAlmostEqual(abs(sjk.S[0, 1]), abs(sjk.S[1, 0]), delta=1e-5)
            lossy_errs.append(
                float(numpy.max(numpy.abs(numpy.abs(sjk.S) - numpy.abs(sj_nw.S))))
            )
        self.assertLess(lossy_errs[1], lossy_errs[0])
        self.assertLess(lossy_errs[2], 1e-10)

    def test_swg_fabry_perot_exact_lossless_and_lossy(self):
        """Verify SWG + SimpleJoint + GenericDevice matches exact analytical Fabry-Perot S-matrix."""
        wl = 1.55e-6
        x = numpy.linspace(0.0, 2.0e-6, 31)
        y = numpy.linspace(0.0, 1.3e-6, 21)

        def _pec_solver(n):
            return EMpy.modesolvers.FD.VFDModeSolver(
                wl,
                x,
                y,
                lambda x_, y_, n_=n: numpy.full((len(x_), len(y_)), n_**2),
                "SSAA",
            ).solve(1, 1e-12)

        # Zero-length degenerate limit
        s_ref = _pec_solver(1.5)
        swg_zero = EMpy.modesolvers.geometries.SWG(s_ref, 0.0)
        numpy.testing.assert_allclose(
            swg_zero.T, numpy.eye(2, dtype=complex), atol=1e-15
        )
        numpy.testing.assert_allclose(
            swg_zero.S, numpy.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex), atol=1e-15
        )
        plt.figure()
        swg_zero.plot()
        plt.close("all")

        # Lossless resonant half-wave Fabry-Perot cavity: R = 0, |T| = 1
        s1_ll = _pec_solver(1.5)
        s2_ll = _pec_solver(2.4)
        beta2_ll = 2.0 * numpy.pi / wl * s2_ll.modes[0].neff
        l_half = float(numpy.real(numpy.pi / beta2_ll))
        fp_res = EMpy.modesolvers.geometries.GenericDevice(
            [
                EMpy.modesolvers.geometries.SimpleJoint(s1_ll, s2_ll),
                EMpy.modesolvers.geometries.SWG(s2_ll, l_half),
                EMpy.modesolvers.geometries.SimpleJoint(s2_ll, s1_ll),
            ]
        )
        self.assertAlmostEqual(abs(fp_res.S[0, 0]), 0.0, places=12)
        self.assertAlmostEqual(abs(fp_res.S[1, 0]), 1.0, places=12)

        # General complex Fabry-Perot cavity across lossless and passive lossy media
        length = 1.73e-6
        for n1, n2 in [(1.5, 2.4), (1.5, 2.4 - 0.15j), (2.1 - 0.05j, 1.4 - 0.2j)]:
            s1 = _pec_solver(n1)
            s2 = _pec_solver(n2)
            ne1, ne2 = s1.modes[0].neff, s2.modes[0].neff
            beta2 = 2.0 * numpy.pi / wl * ne2
            r = (ne1 - ne2) / (ne1 + ne2)
            p = numpy.exp(-1j * beta2 * length)
            s11_exact = r * (1.0 - p**2) / (1.0 - (r * p) ** 2)
            s21_exact = (1.0 - r**2) * p / (1.0 - (r * p) ** 2)

            fp = EMpy.modesolvers.geometries.GenericDevice(
                [
                    EMpy.modesolvers.geometries.SimpleJoint(s1, s2),
                    EMpy.modesolvers.geometries.SWG(s2, length),
                    EMpy.modesolvers.geometries.SimpleJoint(s2, s1),
                ]
            )
            s_exact = numpy.array([[s11_exact, s21_exact], [s21_exact, s11_exact]])
            numpy.testing.assert_allclose(fp.S, s_exact, rtol=1e-12, atol=1e-13)
            numpy.testing.assert_allclose(fp.S, fp.S.T, atol=1e-13)
            if numpy.isreal(n1) and numpy.isreal(n2):
                numpy.testing.assert_allclose(
                    fp.S.conj().T @ fp.S, numpy.eye(2), atol=1e-13
                )
            else:
                self.assertTrue(
                    numpy.all(numpy.linalg.svd(fp.S, compute_uv=False) < 1.0)
                )

    def test_swg_passivity_reciprocity_and_degenerate_limits(self):
        """Verify SWG Beer-Lambert attenuation, reciprocity, and aniso/lossy -> iso/lossless limits."""
        wl = 1.55e-6
        xg = numpy.linspace(0.0, 3.0e-6, 41)
        yg = numpy.linspace(0.0, 2.0e-6, 31)
        length = 50.0e-6

        def _eps_iso(kap=0.0, force_complex=False):
            def _f(x_, y_):
                if kap > 0 or force_complex:
                    nc = complex(3.45, -kap)
                    ncl = complex(1.45, -0.2 * kap)
                else:
                    nc = 3.45
                    ncl = 1.45
                return numpy.where(
                    (numpy.abs(x_[:, None] - 1.5e-6) <= 0.6e-6)
                    & (numpy.abs(y_[None, :] - 1.0e-6) <= 0.11e-6),
                    nc**2,
                    ncl**2,
                )

            return _f

        def _eps_aniso(dn=0.0, kap=0.0, phi_deg=37.0):
            ph = EMpy.utils.deg2rad(phi_deg)

            def _f(x_, y_):
                core = (numpy.abs(x_[:, None] - 1.5e-6) <= 0.6e-6) & (
                    numpy.abs(y_[None, :] - 1.0e-6) <= 0.11e-6
                )
                nc = complex(3.45, -kap) if kap > 0 else 3.45
                ncl = complex(1.45, -0.2 * kap) if kap > 0 else 1.45
                nO = numpy.where(core, nc, ncl)
                nE = numpy.where(core, nc + dn, ncl + dn)
                exx = nE**2 * numpy.cos(ph) ** 2 + nO**2 * numpy.sin(ph) ** 2
                eyy = nE**2 * numpy.sin(ph) ** 2 + nO**2 * numpy.cos(ph) ** 2
                exy = (nE**2 - nO**2) * numpy.sin(ph) * numpy.cos(ph)
                ezz = nO**2
                return numpy.stack([exx, exy, exy, eyy, ezz], axis=-1)

            return _f

        s_ref = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xg, yg, _eps_iso(0.0), "0000"
        ).solve(2, 1e-10)
        swg_ref = EMpy.modesolvers.geometries.SWG(s_ref, length)
        numpy.testing.assert_allclose(
            swg_ref.S.conj().T @ swg_ref.S, numpy.eye(4), atol=1e-14
        )
        numpy.testing.assert_allclose(swg_ref.S, swg_ref.S.T, atol=1e-15)

        # Complex zero-loss (n_r + 0.0j) must match strictly real lossless SWG
        s_c0 = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xg, yg, _eps_iso(0.0, force_complex=True), "0000"
        ).solve(2, 1e-10)
        swg_c0 = EMpy.modesolvers.geometries.SWG(s_c0, length)
        numpy.testing.assert_allclose(swg_c0.S, swg_ref.S, rtol=1e-10, atol=1e-10)

        # Continuous lossy -> lossless limit (kap = 1e-2, 1e-4, 1e-8 -> 0)
        loss_errs = []
        for kap in [1e-2, 1e-4, 1e-8]:
            sk = EMpy.modesolvers.FD.VFDModeSolver(
                wl, xg, yg, _eps_iso(kap), "0000"
            ).solve(2, 1e-10)
            swgk = EMpy.modesolvers.geometries.SWG(sk, length)
            betas_k = 2.0 * numpy.pi / wl * numpy.array([m.neff for m in sk.modes])
            self.assertTrue(numpy.all(numpy.imag(betas_k) < 0.0))
            # Forward and backward transmission must both attenuate as exp(-i * beta * L)
            expected_diag = numpy.exp(-1j * betas_k * length)
            numpy.testing.assert_allclose(
                numpy.diag(swgk.S[2:, :2]), expected_diag, rtol=1e-13, atol=1e-14
            )
            numpy.testing.assert_allclose(
                numpy.diag(swgk.S[:2, 2:]), expected_diag, rtol=1e-13, atol=1e-14
            )
            self.assertTrue(numpy.all(numpy.abs(expected_diag) < 1.0))
            self.assertTrue(numpy.all(numpy.linalg.svd(swgk.S, compute_uv=False) < 1.0))
            numpy.testing.assert_allclose(swgk.S, swgk.S.T, atol=1e-15)
            loss_errs.append(float(numpy.max(numpy.abs(swgk.S - swg_ref.S))))
        self.assertLess(loss_errs[1], loss_errs[0])
        self.assertLess(loss_errs[2], loss_errs[1])
        self.assertLess(loss_errs[2], 1e-5)

        # Anisotropic 5-component rotated tensor -> isotropic limit (dn = 1e-4, 1e-8, 0)
        aniso_errs = []
        for dn in [1e-4, 1e-8, 0.0]:
            sa = EMpy.modesolvers.FD.VFDModeSolver(
                wl, xg, yg, _eps_aniso(dn=dn, kap=0.0), "0000"
            ).solve(2, 1e-10)
            swga = EMpy.modesolvers.geometries.SWG(sa, length)
            numpy.testing.assert_allclose(
                swga.S.conj().T @ swga.S, numpy.eye(4), atol=1e-14
            )
            numpy.testing.assert_allclose(swga.S, swga.S.T, atol=1e-15)
            aniso_errs.append(float(numpy.max(numpy.abs(swga.S - swg_ref.S))))
        self.assertLess(aniso_errs[1], aniso_errs[0])
        self.assertLess(aniso_errs[2], 1e-9)

        # Lossy anisotropic with dn = 0 must match lossy scalar isotropic
        s_iso_lossy = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xg, yg, _eps_iso(1e-3), "0000"
        ).solve(2, 1e-10)
        s_ani_lossy = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xg, yg, _eps_aniso(dn=0.0, kap=1e-3), "0000"
        ).solve(2, 1e-10)
        swg_iso_l = EMpy.modesolvers.geometries.SWG(s_iso_lossy, length)
        swg_ani_l = EMpy.modesolvers.geometries.SWG(s_ani_lossy, length)
        numpy.testing.assert_allclose(swg_ani_l.S, swg_iso_l.S, rtol=1e-9, atol=1e-10)

    def test_swg_and_generic_device_plot_coherent_physics(self):
        """Verify SWG.plot uses (x, y) grids and captures Beer-Lambert decay, standing waves, and MMI beating."""
        wl = 1.55e-6
        # Non-square domain (Lx = 3.2 um != Ly = 1.4 um) catches old m.intensity(x, x) coordinate bug
        xg = numpy.linspace(0.0, 3.2e-6, 49)
        yg = numpy.linspace(0.0, 1.4e-6, 29)

        def _eps_mmi(kap=0.0):
            def _f(x_, y_):
                nc = complex(3.45, -kap) if kap > 0 else 3.45
                ncl = 1.45 + 0.0j if kap > 0 else 1.45
                return numpy.where(
                    (numpy.abs(x_[:, None] - 1.6e-6) <= 0.8e-6)
                    & (numpy.abs(y_[None, :] - 0.7e-6) <= 0.11e-6),
                    nc**2,
                    ncl**2,
                )

            return _f

        # 1. Beer-Lambert decay along +z (forward) and -z (backward) on lossy waveguide
        s_lossy = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xg, yg, _eps_mmi(kap=0.02), "0000"
        ).solve(1, 1e-10)
        beta_l = 2.0 * numpy.pi / wl * s_lossy.modes[0].neff
        length_l = 15.0e-6
        swg_fwd = EMpy.modesolvers.geometries.SWG(
            s_lossy, length_l, inputLHS=numpy.array([1.0]), inputRHS=numpy.array([0.0])
        )
        cx, z_grid, f_fwd = swg_fwd._marginal_intensity(sumx=1, nxy=60, nz=31)
        p_fwd = numpy.trapezoid(f_fwd, x=cx, axis=0)
        numpy.testing.assert_allclose(
            p_fwd / p_fwd[0],
            numpy.exp(2.0 * numpy.imag(beta_l) * z_grid),
            rtol=1e-12,
            atol=1e-13,
        )

        swg_bwd = EMpy.modesolvers.geometries.SWG(
            s_lossy, length_l, inputLHS=numpy.array([0.0]), inputRHS=numpy.array([1.0])
        )
        cy, z_grid_y, f_bwd = swg_bwd._marginal_intensity(sumx=None, nxy=60, nz=31)
        p_bwd = numpy.trapezoid(f_bwd, x=cy, axis=0)
        numpy.testing.assert_allclose(
            p_bwd / p_bwd[-1],
            numpy.exp(2.0 * numpy.imag(beta_l) * (length_l - z_grid_y)),
            rtol=1e-12,
            atol=1e-13,
        )

        # 2. Counter-propagating standing wave (L = pi / b0): nodes at z = 0, L and 4x antinode at z = L/2
        s_2m = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xg, yg, _eps_mmi(kap=0.0), "0000"
        ).solve(2, 1e-10)
        b0 = float(numpy.real(2.0 * numpy.pi / wl * s_2m.modes[0].neff))
        l_sw = numpy.pi / b0
        swg_sw = EMpy.modesolvers.geometries.SWG(
            s_2m,
            l_sw,
            inputLHS=numpy.array([1.0, 0.0]),
            inputRHS=numpy.array([1.0, 0.0]),
        )
        cx_sw, _, f_sw = swg_sw._marginal_intensity(sumx=1, nxy=60, nz=5)
        p_sw = numpy.trapezoid(f_sw, x=cx_sw, axis=0)
        self.assertLess(abs(p_sw[0]), 1e-12)
        self.assertLess(abs(p_sw[4]), 1e-12)
        self.assertAlmostEqual(p_sw[2] / p_sw[1], 2.0, places=12)
        self.assertGreater(p_sw[2], 3.8)

        # 3. Two-mode MMI beating: total power conserved while lateral centroid shifts across waveguide center
        # Pick two modes with opposite x-parity (fundamental + first antisymmetric TE-like mode)
        s_3m = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xg, yg, _eps_mmi(kap=0.0), "0000"
        ).solve(3, 1e-10)
        # Find the higher-order mode that produces nonzero lateral centroid shift at z = 0
        mmi_idx = 1
        for cand in (1, 2):
            bc = float(numpy.real(2.0 * numpy.pi / wl * s_3m.modes[cand].neff))
            inp = numpy.zeros(3)
            inp[0] = 1.0 / numpy.sqrt(2.0)
            inp[cand] = 1.0 / numpy.sqrt(2.0)
            swg_try = EMpy.modesolvers.geometries.SWG(
                s_3m, numpy.pi / abs(b0 - bc), inputLHS=inp
            )
            cx_m, _, f_try = swg_try._marginal_intensity(sumx=1, nxy=65, nz=9)
            x_shift = float(numpy.trapezoid((cx_m - 1.6e-6) * f_try[:, 0], x=cx_m))
            if abs(x_shift) > 5e-8:
                mmi_idx = cand
                break
        b_mmi = float(numpy.real(2.0 * numpy.pi / wl * s_3m.modes[mmi_idx].neff))
        inp_mmi = numpy.zeros(3)
        inp_mmi[0] = 1.0 / numpy.sqrt(2.0)
        inp_mmi[mmi_idx] = 1.0 / numpy.sqrt(2.0)
        swg_mmi = EMpy.modesolvers.geometries.SWG(
            s_3m, numpy.pi / abs(b0 - b_mmi), inputLHS=inp_mmi
        )
        cx_mmi, _, f_mmi = swg_mmi._marginal_intensity(sumx=1, nxy=81, nz=17)
        p_mmi = numpy.trapezoid(f_mmi, x=cx_mmi, axis=0)
        numpy.testing.assert_allclose(p_mmi, p_mmi[0], rtol=2e-2)
        centroid_0 = float(numpy.trapezoid((cx_mmi - 1.6e-6) * f_mmi[:, 0], x=cx_mmi))
        centroid_end = float(
            numpy.trapezoid((cx_mmi - 1.6e-6) * f_mmi[:, -1], x=cx_mmi)
        )
        self.assertGreater(abs(centroid_0), 5e-8)
        self.assertAlmostEqual(centroid_0, -centroid_end, delta=1e-8)

        # 4. Matplotlib rendering smoke test for GenericDevice.plot (both sumx=1 and sumx=None) and FMM2d SWG.plot
        gd = EMpy.modesolvers.geometries.GenericDevice(
            [
                EMpy.modesolvers.geometries.SWG(s_lossy, 5.0e-6),
                EMpy.modesolvers.geometries.SimpleJoint(s_lossy, s_lossy),
                EMpy.modesolvers.geometries.SWG(s_lossy, 5.0e-6),
            ]
        )
        plt.figure()
        gd.plot(sumx=1, nxy=25, nz=11)
        self.assertEqual(plt.gca().get_xlabel(), "x")
        self.assertEqual(plt.gca().get_ylabel(), "z")
        plt.close("all")

        plt.figure()
        gd.plot(sumx=None, nxy=25, nz=11)
        self.assertEqual(plt.gca().get_xlabel(), "y")
        self.assertEqual(plt.gca().get_ylabel(), "z")
        plt.close("all")

        # FMM2d mode solver compatibility in SWG.plot (transposed fields(x, y) shape)
        fmm = EMpy.modesolvers.FMM.FMM2d(
            numpy.array([0.0, 1.0, 1.4, 2.8]),
            numpy.array([0.0, 0.5, 0.72, 1.4]),
            numpy.array(
                [
                    [1.45, 1.45, 1.45],
                    [1.45, 3.45, 1.45],
                    [1.45, 1.45, 1.45],
                ]
            ),
            EMpy.modesolvers.FMM.Boundary(
                xleft="Electric Wall",
                yleft="Magnetic Wall",
                xright="Electric Wall",
                yright="Magnetic Wall",
            ),
        ).solve(1.55, 2, 1)
        swg_fmm = EMpy.modesolvers.geometries.SWG(fmm, 4.0)
        plt.figure()
        swg_fmm.plot(sumx=1, nxy=21, nz=9)
        swg_fmm.plot(sumx=None, nxy=21, nz=9)
        plt.close("all")

    def test_fd_shift_invert_guess_above_below_and_subcutoff_passivity(self):
        """Verify FD solvers find true guided modes for guess above/below/None and keep sub-cutoff modes passive."""
        wl = 1.55e-6
        mat1 = EMpy.materials.IsotropicMaterial(
            "SiN", n0=EMpy.materials.RefractiveIndex(n0_const=1.97)
        )
        mat2 = EMpy.materials.IsotropicMaterial(
            "Si", n0=EMpy.materials.RefractiveIndex(n0_const=3.4757)
        )
        l1 = EMpy.utils.Layer(mat1, 4.1e-6)
        l21 = EMpy.utils.Layer(mat1, 2.0e-6)
        l22 = EMpy.utils.Layer(mat2, 0.1e-6)
        cs = EMpy.utils.CrossSection(
            [
                EMpy.utils.Slice(2.0e-6, [l1]),
                EMpy.utils.Slice(0.5e-6, [l21, l22, l21]),
                EMpy.utils.Slice(2.0e-6, [l1]),
            ]
        )
        xg, yg = cs.grid((12, 8, 12), (12, 12, 12))

        def _eps(x_, y_):
            return cs.epsfunc(x_, y_, wl)

        # VFDModeSolver & SVFDModeSolver: guess = n_core (above neff), guess below neff, and guess = None must agree
        v_none = EMpy.modesolvers.FD.VFDModeSolver(wl, xg, yg, _eps, "0000").solve(
            2, 1e-10
        )
        v_hi = EMpy.modesolvers.FD.VFDModeSolver(wl, xg, yg, _eps, "0000").solve(
            2, 1e-10, guess=3.4757
        )
        v_lo = EMpy.modesolvers.FD.VFDModeSolver(wl, xg, yg, _eps, "0000").solve(
            2, 1e-10, guess=2.05
        )
        self.assertGreater(v_none.modes[0].neff.real, 2.0)
        numpy.testing.assert_allclose(
            v_hi.modes[0].neff, v_none.modes[0].neff, rtol=1e-10
        )
        numpy.testing.assert_allclose(
            v_lo.modes[0].neff, v_none.modes[0].neff, rtol=1e-10
        )

        sv_none = EMpy.modesolvers.FD.SVFDModeSolver(
            wl, xg, yg, _eps, "0000", method="Ex"
        ).solve(1, 1e-10)
        sv_hi = EMpy.modesolvers.FD.SVFDModeSolver(
            wl, xg, yg, _eps, "0000", method="Ex"
        ).solve(1, 1e-10, guess=3.4757)
        numpy.testing.assert_allclose(sv_hi.neff[0], sv_none.neff[0], rtol=1e-10)

        # Sub-cutoff PEC air waveguide (0.4 um x 0.3 um at wl = 1.55 um): Im(neff) < 0 and SWG |S21| < 1
        xp = numpy.linspace(0.0, 0.4e-6, 17)
        yp = numpy.linspace(0.0, 0.3e-6, 13)
        s_ev = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xp, yp, lambda x_, y_: numpy.ones((len(x_), len(y_))), "0000"
        ).solve(2, 1e-10)
        sv_ev = EMpy.modesolvers.FD.SVFDModeSolver(
            wl,
            xp,
            yp,
            lambda x_, y_: numpy.ones((len(x_), len(y_))),
            "0000",
            method="Ex",
        ).solve(2, 1e-10)
        self.assertTrue(numpy.all(numpy.imag([m.neff for m in s_ev.modes]) < -1.0))
        self.assertTrue(numpy.all(numpy.imag(sv_ev.neff) < -1.0))
        swg_ev = EMpy.modesolvers.geometries.SWG(s_ev, 0.5e-6)
        # TE10 cutoff mode (neff ~ -1.66j) decays as exp(-k0 * 1.66 * 0.5e-6) ~ 0.034 < 0.05 (not 1e61 gain)
        self.assertTrue(numpy.all(numpy.linalg.svd(swg_ev.S, compute_uv=False) < 0.05))

    def test_simple_joint_multimode_passive_vs_reciprocal_and_fmm1d_slab(self):
        """Verify SimpleJoint method='passive' vs 'reciprocal' on N=3 step junction and exact FMM1d slab dispersion."""
        wl = 1.55e-6
        xg = numpy.linspace(0.0, 3.2e-6, 41)
        yg = numpy.linspace(0.0, 1.8e-6, 25)

        def _eps_step(w, kap=0.0):
            nc = complex(3.45, -kap) if kap > 0 else 3.45
            ncl = complex(1.45, -0.2 * kap) if kap > 0 else 1.45
            return lambda x_, y_: numpy.where(
                (numpy.abs(x_[:, None] - 1.6e-6) <= w / 2.0)
                & (numpy.abs(y_[None, :] - 0.9e-6) <= 0.11e-6),
                nc**2,
                ncl**2,
            )

        for kap in (0.0, 0.02):
            s1 = EMpy.modesolvers.FD.VFDModeSolver(
                wl, xg, yg, _eps_step(0.5e-6, kap), "0000"
            ).solve(3, 1e-10)
            s2 = EMpy.modesolvers.FD.VFDModeSolver(
                wl, xg, yg, _eps_step(1.6e-6, kap), "0000"
            ).solve(3, 1e-10)
            sj_pas = EMpy.modesolvers.geometries.SimpleJoint(s1, s2, method="passive")
            sj_rec = EMpy.modesolvers.geometries.SimpleJoint(
                s1, s2, method="reciprocal"
            )
            # Wide-to-narrow fundamental port (index 3) exhibits strict radiation loss (< 1)
            self.assertLess(numpy.sum(numpy.abs(sj_pas.S[:, 3]) ** 2), 1.0)
            numpy.testing.assert_allclose(sj_rec.S, sj_rec.S.T, atol=1e-12)
            if kap == 0.0:
                numpy.testing.assert_allclose(
                    sj_rec.S.conj().T @ sj_rec.S, numpy.eye(6), atol=1e-12
                )

        with self.assertRaises(ValueError):
            EMpy.modesolvers.geometries.SimpleJoint(s1, s2, method="unknown")

        s1_one = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xg, yg, _eps_step(0.5e-6, 0.0), "0000"
        ).solve(1, 1e-10)
        with self.assertRaises(ValueError):
            EMpy.modesolvers.geometries.SimpleJoint(s1_one, s2, method="passive")
        with self.assertRaises(ValueError):
            EMpy.modesolvers.geometries.SimpleJoint(s1_one, s2, method="reciprocal")
        with self.assertRaises(ValueError):
            EMpy.modesolvers.interface.interface_matrix(s1_one, s2)

        # Exact 1D symmetric slab waveguide analytical dispersion (TE and TM, lossless and passive lossy)
        wl_1d = 1.55
        k0 = 2.0 * numpy.pi / wl_1d
        d_core = 0.30
        Uy = numpy.array([0.0, 4.5, 4.5 + d_core, 9.0 + d_core])
        for kap in (0.0, 0.01):
            n_core = complex(3.45, -kap) if kap > 0 else 3.45
            n_clad = 1.45
            ny = numpy.array([n_clad, n_core, n_clad])
            for pol in ("TE", "TM"):
                f1d = EMpy.modesolvers.FMM.FMM1d(Uy, ny, "AA").solve(wl_1d, 1, pol)
                kz = f1d.modes[0].keff
                kc = numpy.sqrt((k0 * n_core) ** 2 - kz**2 + 0j)
                gc = numpy.sqrt(kz**2 - (k0 * n_clad) ** 2 + 0j)
                if gc.real < 0:
                    gc = -gc
                ratio = (n_core / n_clad) ** 2 if pol == "TM" else 1.0
                # Fundamental symmetric mode satisfies kc * tan(kc * d_core / 2) == ratio * gc
                lhs = kc * numpy.tan(0.5 * kc * d_core)
                rhs = ratio * gc
                numpy.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10)

    def test_simple_joint_high_contrast_passivity_reciprocal_swap_and_limits(self):
        """Verify SimpleJoint passivity on slot-spiked high-contrast junctions and reciprocal swap symmetry."""
        wl = 1.55e-6
        xg = numpy.linspace(0.0, 2.4e-6, 33)
        yg = numpy.linspace(0.0, 1.8e-6, 25)

        def _eps_hc(w, h, kap=0.0, dn=0.0, force_complex=False):
            if kap > 0 or force_complex:
                nc = complex(3.48, -kap)
                ncl = complex(1.44, -0.1 * kap)
            else:
                nc = 3.48
                ncl = 1.44
            ph = 0.5

            def _f(x_, y_):
                core = (numpy.abs(x_[:, None] - 1.2e-6) <= w / 2.0) & (
                    numpy.abs(y_[None, :] - 0.9e-6) <= h / 2.0
                )
                no = numpy.where(core, nc, ncl)
                ne = no + dn
                if dn == 0.0 and not force_complex and kap == 0.0:
                    return no**2
                exx = ne**2 * numpy.cos(ph) ** 2 + no**2 * numpy.sin(ph) ** 2
                eyy = ne**2 * numpy.sin(ph) ** 2 + no**2 * numpy.cos(ph) ** 2
                exy = (ne**2 - no**2) * numpy.sin(ph) * numpy.cos(ph)
                return numpy.stack([exx, exy, exy, eyy, no**2], axis=-1)

            return _f

        # 1. High-contrast 0.35x0.18 -> 1.4x0.5 um junction (where raw O12*O21 > 1):
        # verify ||S||_2 <= 1 for method="passive" and exact swap + reciprocity + unitarity for method="reciprocal"
        for nm in (1, 4):
            for kap in (0.0, 0.05):
                for dn in (0.0, 0.3):
                    s1 = EMpy.modesolvers.FD.VFDModeSolver(
                        wl,
                        xg,
                        yg,
                        _eps_hc(0.35e-6, 0.18e-6, kap, dn, force_complex=(dn > 0)),
                        "0000",
                    ).solve(nm, 1e-10)
                    s2 = EMpy.modesolvers.FD.VFDModeSolver(
                        wl,
                        xg,
                        yg,
                        _eps_hc(1.4e-6, 0.5e-6, kap, dn, force_complex=(dn > 0)),
                        "0000",
                    ).solve(nm, 1e-10)
                    sj_p12 = EMpy.modesolvers.geometries.SimpleJoint(
                        s1, s2, method="passive"
                    )
                    sj_p21 = EMpy.modesolvers.geometries.SimpleJoint(
                        s2, s1, method="passive"
                    )
                    P_swap = numpy.block(
                        [
                            [numpy.zeros((nm, nm)), numpy.eye(nm)],
                            [numpy.eye(nm), numpy.zeros((nm, nm))],
                        ]
                    )
                    self.assertTrue(
                        numpy.all(numpy.linalg.svd(sj_p12.S, compute_uv=False) <= 1.0)
                    )
                    numpy.testing.assert_allclose(
                        numpy.abs(sj_p12.S),
                        numpy.abs(P_swap @ sj_p21.S @ P_swap.T),
                        atol=1e-12,
                    )

                    sj_r12 = EMpy.modesolvers.geometries.SimpleJoint(
                        s1, s2, method="reciprocal"
                    )
                    sj_r21 = EMpy.modesolvers.geometries.SimpleJoint(
                        s2, s1, method="reciprocal"
                    )
                    numpy.testing.assert_allclose(sj_r12.S, sj_r12.S.T, atol=1e-12)
                    numpy.testing.assert_allclose(
                        sj_r12.S, P_swap @ sj_r21.S @ P_swap.T, atol=1e-12
                    )
                    if kap == 0.0:
                        numpy.testing.assert_allclose(
                            sj_r12.S.conj().T @ sj_r12.S,
                            numpy.eye(2 * nm),
                            atol=1e-10,
                        )

        # 2. Smooth anisotropic -> isotropic (dn = 1e-4, 1e-8, 0) and lossy -> lossless (kap = 1e-2, 1e-5, 0+0j) limits
        s1_iso = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xg, yg, _eps_hc(0.35e-6, 0.18e-6, 0.0, 0.0), "0000"
        ).solve(2, 1e-10)
        s2_iso = EMpy.modesolvers.FD.VFDModeSolver(
            wl, xg, yg, _eps_hc(1.4e-6, 0.5e-6, 0.0, 0.0), "0000"
        ).solve(2, 1e-10)
        sj_iso_p = EMpy.modesolvers.geometries.SimpleJoint(
            s1_iso, s2_iso, method="passive"
        )
        sj_iso_r = EMpy.modesolvers.geometries.SimpleJoint(
            s1_iso, s2_iso, method="reciprocal"
        )

        dn_errs_p, dn_errs_r = [], []
        for dn in (1e-4, 1e-8, 0.0):
            s1_d = EMpy.modesolvers.FD.VFDModeSolver(
                wl,
                xg,
                yg,
                _eps_hc(0.35e-6, 0.18e-6, 0.0, dn, force_complex=True),
                "0000",
            ).solve(2, 1e-10)
            s2_d = EMpy.modesolvers.FD.VFDModeSolver(
                wl,
                xg,
                yg,
                _eps_hc(1.4e-6, 0.5e-6, 0.0, dn, force_complex=True),
                "0000",
            ).solve(2, 1e-10)
            sjp = EMpy.modesolvers.geometries.SimpleJoint(s1_d, s2_d, method="passive")
            sjr = EMpy.modesolvers.geometries.SimpleJoint(
                s1_d, s2_d, method="reciprocal"
            )
            self.assertTrue(numpy.all(numpy.linalg.svd(sjp.S, compute_uv=False) <= 1.0))
            dn_errs_p.append(
                float(numpy.max(numpy.abs(numpy.abs(sjp.S) - numpy.abs(sj_iso_p.S))))
            )
            dn_errs_r.append(
                float(numpy.max(numpy.abs(numpy.abs(sjr.S) - numpy.abs(sj_iso_r.S))))
            )
        self.assertLess(dn_errs_p[1], dn_errs_p[0])
        self.assertLess(dn_errs_p[2], 1e-9)
        self.assertLess(dn_errs_r[1], dn_errs_r[0])
        self.assertLess(dn_errs_r[2], 1e-9)

        kap_errs_p, kap_errs_r = [], []
        for kap in (1e-2, 1e-5, 0.0):
            s1_k = EMpy.modesolvers.FD.VFDModeSolver(
                wl,
                xg,
                yg,
                _eps_hc(0.35e-6, 0.18e-6, kap, 0.0, force_complex=True),
                "0000",
            ).solve(2, 1e-10)
            s2_k = EMpy.modesolvers.FD.VFDModeSolver(
                wl,
                xg,
                yg,
                _eps_hc(1.4e-6, 0.5e-6, kap, 0.0, force_complex=True),
                "0000",
            ).solve(2, 1e-10)
            sjp = EMpy.modesolvers.geometries.SimpleJoint(s1_k, s2_k, method="passive")
            sjr = EMpy.modesolvers.geometries.SimpleJoint(
                s1_k, s2_k, method="reciprocal"
            )
            self.assertTrue(numpy.all(numpy.linalg.svd(sjp.S, compute_uv=False) <= 1.0))
            kap_errs_p.append(
                float(numpy.max(numpy.abs(numpy.abs(sjp.S) - numpy.abs(sj_iso_p.S))))
            )
            kap_errs_r.append(
                float(numpy.max(numpy.abs(numpy.abs(sjr.S) - numpy.abs(sj_iso_r.S))))
            )
        self.assertLess(kap_errs_p[1], kap_errs_p[0])
        self.assertLess(kap_errs_p[2], 1e-9)
        self.assertLess(kap_errs_r[1], kap_errs_r[0])
        self.assertLess(kap_errs_r[2], 1e-9)

        # 3. Degenerate square-core self-joint: zero spurious reflection in both passive and reciprocal modes
        xsq = numpy.linspace(0.0, 2.4e-6, 33)
        ysq = numpy.linspace(0.0, 2.4e-6, 33)
        s_sq = EMpy.modesolvers.FD.VFDModeSolver(
            wl,
            xsq,
            ysq,
            lambda x_, y_: numpy.where(
                (numpy.abs(x_[:, None] - 1.2e-6) <= 0.25e-6)
                & (numpy.abs(y_[None, :] - 1.2e-6) <= 0.25e-6),
                3.45**2,
                1.45**2,
            ),
            "0000",
        ).solve(2, 1e-10)
        for meth in ("passive", "reciprocal"):
            sj_sq = EMpy.modesolvers.geometries.SimpleJoint(s_sq, s_sq, method=meth)
            numpy.testing.assert_allclose(sj_sq.S[:2, :2], 0.0, atol=1e-12)
            numpy.testing.assert_allclose(sj_sq.S[2:, 2:], 0.0, atol=1e-12)
            numpy.testing.assert_allclose(sj_sq.S[:2, 2:], numpy.eye(2), atol=1e-12)
            numpy.testing.assert_allclose(sj_sq.S[2:, :2], numpy.eye(2), atol=1e-12)

    def test_generic_device_subcutoff_and_long_lossy_cascade(self):
        """Verify GenericDevice Redheffer star-product stability with sub-cutoff and long lossy waveguides."""
        wl = 1.55e-6
        x = numpy.linspace(0.0, 0.65e-6, 17)
        y = numpy.linspace(0.0, 0.25e-6, 15)
        ph = EMpy.utils.deg2rad(29.0)

        def _hom_solver(n_base, kap=0.0, dn=0.0, anisotropic=False):
            nO = complex(n_base, -kap) if kap > 0 else n_base
            nE = complex(n_base + dn, -kap) if kap > 0 else (n_base + dn)

            def _eps(xx, yy):
                if not anisotropic:
                    return numpy.full((len(xx), len(yy)), nO**2)
                exx = nE**2 * numpy.cos(ph) ** 2 + nO**2 * numpy.sin(ph) ** 2
                eyy = nE**2 * numpy.sin(ph) ** 2 + nO**2 * numpy.cos(ph) ** 2
                exy = (nE**2 - nO**2) * numpy.sin(ph) * numpy.cos(ph)
                shape = (len(xx), len(yy))
                return numpy.stack(
                    [
                        numpy.full(shape, exx),
                        numpy.full(shape, exy),
                        numpy.full(shape, exy),
                        numpy.full(shape, eyy),
                        numpy.full(shape, nO**2),
                    ],
                    axis=-1,
                )

            return EMpy.modesolvers.FD.VFDModeSolver(wl, x, y, _eps, "SSAA").solve(
                2, 1e-12
            )

        s1 = _hom_solver(1.6)
        s2 = _hom_solver(1.9)
        self.assertGreater(s1.modes[0].neff.real, 0.0)
        self.assertLess(s1.modes[1].neff.imag, -1.0)
        self.assertLess(s2.modes[1].neff.imag, -1.0)

        # At L = 200 um, exp(+1j * beta_subcutoff * L) overflows float64, but Redheffer cascade stays exact
        length = 200e-6
        gd = EMpy.modesolvers.geometries.GenericDevice(
            [
                EMpy.modesolvers.geometries.SimpleJoint(s1, s2),
                EMpy.modesolvers.geometries.SWG(s2, length),
                EMpy.modesolvers.geometries.SimpleJoint(s2, s1),
            ],
            inputLHS=numpy.array([1.0, 0.4]),
            inputRHS=numpy.array([-0.25j, 0.1]),
        )
        self.assertTrue(numpy.all(numpy.isfinite(gd.S)))
        self.assertTrue(
            all(
                numpy.all(numpy.isfinite(d.outputLHS))
                and numpy.all(numpy.isfinite(d.outputRHS))
                for d in gd.devlist
            )
        )
        self.assertLessEqual(
            float(numpy.linalg.svd(gd.S, compute_uv=False)[0]), 1.0 + 1e-12
        )

        ne1, ne2 = s1.modes[0].neff, s2.modes[0].neff
        r0 = (ne1 - ne2) / (ne1 + ne2)
        p0 = numpy.exp(-1j * (2.0 * numpy.pi / wl * ne2) * length)
        s21_0_exact = (1.0 - r0**2) * p0 / (1.0 - (r0 * p0) ** 2)
        self.assertAlmostEqual(gd.S[2, 0], s21_0_exact, places=12)

        # Degenerate limits on the L = 200 um sub-cutoff cascade: aniso dn -> 0 and lossy kap -> 0+
        dn_errs = []
        for dn in (1e-4, 1e-8, 0.0):
            s1_a = _hom_solver(1.6, dn=dn, anisotropic=True)
            s2_a = _hom_solver(1.9, dn=dn, anisotropic=True)
            gd_a = EMpy.modesolvers.geometries.GenericDevice(
                [
                    EMpy.modesolvers.geometries.SimpleJoint(s1_a, s2_a),
                    EMpy.modesolvers.geometries.SWG(s2_a, length),
                    EMpy.modesolvers.geometries.SimpleJoint(s2_a, s1_a),
                ]
            )
            dn_errs.append(abs(abs(gd_a.S[2, 0]) - abs(gd.S[2, 0])))
        self.assertLess(dn_errs[1], dn_errs[0])
        self.assertLess(dn_errs[2], 1e-10)

        kap_errs = []
        for kap in (1e-3, 1e-6, 0.0):
            s2_k = _hom_solver(1.9, kap=kap)
            gd_k = EMpy.modesolvers.geometries.GenericDevice(
                [
                    EMpy.modesolvers.geometries.SimpleJoint(s1, s2_k),
                    EMpy.modesolvers.geometries.SWG(s2_k, length),
                    EMpy.modesolvers.geometries.SimpleJoint(s2_k, s1),
                ]
            )
            if kap > 0:
                self.assertLess(abs(gd_k.S[0, 0]) ** 2 + abs(gd_k.S[2, 0]) ** 2, 1.0)
            kap_errs.append(abs(gd_k.S[2, 0] - gd.S[2, 0]))
        self.assertLess(kap_errs[1], kap_errs[0])
        self.assertLess(kap_errs[2], 1e-10)
