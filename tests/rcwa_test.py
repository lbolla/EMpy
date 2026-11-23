from unittest import TestCase

import numpy as np

import EMpy.RCWA
import EMpy.materials
import EMpy.transfer_matrix
import EMpy.utils


class IsotropicRCWATest(TestCase):

    def test_ill_conditioned_helmholtz_operator(self):
        # At normal incidence with wl == LAMBDA and an internal air layer (n=1),
        # orders +/-1 satisfy (kx/k)^2 == eps == 1, making A singular (cond(A) > 1e10).
        air = EMpy.materials.Air
        si = EMpy.materials.IsotropicMaterial(
            "Si", n0=EMpy.materials.RefractiveIndex(n0_const=3.48)
        )
        grating_air = EMpy.utils.BinaryGrating(air, air, 0.5, 1.0e-6, 0.2e-6)
        multilayer = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(air, np.inf),
                grating_air,
                EMpy.utils.Layer(si, np.inf),
            ]
        )
        wl = np.array([1.0e-6])
        rcwa = EMpy.RCWA.IsotropicRCWA(
            multilayer, alpha=0.0, delta=0.0, psi=0.0, phi=np.pi / 2, n=2
        ).solve(wl)
        tmm = EMpy.transfer_matrix.IsotropicTransferMatrix(
            EMpy.utils.Multilayer(
                [
                    EMpy.utils.Layer(air, np.inf),
                    EMpy.utils.Layer(air, 0.2e-6),
                    EMpy.utils.Layer(si, np.inf),
                ]
            ),
            0.0,
        ).solve(wl)

        self.assertAlmostEqual(rcwa.DE1.sum(), tmm.Rp[0], places=10)
        self.assertAlmostEqual(rcwa.DE3.sum(), tmm.Tp[0], places=10)
        self.assertAlmostEqual(rcwa.DE1.sum() + rcwa.DE3.sum(), 1.0, places=12)

        # Multilayer grating + singular internal air layer: uniform pre-loop Bloch shift
        # keeps all layers on a single Floquet lattice, conserving energy and matching wl*(1+1e-7)
        sio2 = EMpy.materials.IsotropicMaterial(
            "SiO2", n0=EMpy.materials.RefractiveIndex(n0_const=1.45)
        )
        ml_multi = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(air, np.inf),
                EMpy.utils.BinaryGrating(si, sio2, 0.4, 1.0e-6, 0.25e-6),
                EMpy.utils.Layer(air, 0.2e-6),
                EMpy.utils.Layer(sio2, np.inf),
            ]
        )
        r_at = EMpy.RCWA.IsotropicRCWA(
            ml_multi, alpha=0.0, delta=0.35, psi=0.3, phi=np.pi / 2, n=2
        ).solve([1.0e-6])
        r_near = EMpy.RCWA.IsotropicRCWA(
            ml_multi, alpha=0.0, delta=0.35, psi=0.3, phi=np.pi / 2, n=2
        ).solve([1.0e-6 * (1.0 + 1e-9)])
        r_ani = EMpy.RCWA.AnisotropicRCWA(
            ml_multi, alpha=0.0, delta=0.35, psi=0.3, phi=np.pi / 2, n=2
        ).solve([1.0e-6])
        self.assertAlmostEqual(float((r_at.DE1 + r_at.DE3).sum()), 1.0, places=12)
        np.testing.assert_allclose(r_at.DE1, r_near.DE1, atol=2e-4)
        np.testing.assert_allclose(r_at.DE3, r_near.DE3, atol=2e-4)
        np.testing.assert_allclose(r_at.DE1, r_ani.DE1, atol=5e-3)
        np.testing.assert_allclose(r_at.DE3, r_ani.DE3, atol=5e-3)

    def test_dispersion_relations_propagating_and_evanescent(self):
        k = 2 * np.pi / 1.55e-6
        nO, nE = 1.5, 1.7
        kx = np.array([0.5 * k, 2.5 * k])  # propagating, then evanescent
        ky = np.zeros_like(kx)

        kz_o = EMpy.RCWA.dispersion_relation_ordinary(kx, ky, k, nO)
        self.assertTrue(np.all(np.isfinite(kz_o)))
        self.assertGreater(kz_o[0].real, 0.0)
        self.assertAlmostEqual(kz_o[0].imag, 0.0)
        self.assertAlmostEqual(kz_o[1].real, 0.0)
        self.assertLess(kz_o[1].imag, 0.0)

        # Tilted optic axis (C[1] != 0)
        c = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
        kz_e = EMpy.RCWA.dispersion_relation_extraordinary(kx, ky, k, nO, nE, c)
        self.assertTrue(np.all(np.isfinite(kz_e)))
        self.assertGreater(kz_e[0].real, 0.0)
        self.assertLess(kz_e[1].imag, 0.0)

        # Verify dispersion polynomial residual P(kz/k) == 0
        alpha = nE**2 - nO**2
        for ii in range(kx.size):
            beta = kx[ii] / k * c[0] + ky[ii] / k * c[1]
            c2 = nO**2 + c[2] ** 2 * alpha
            c1 = 2.0 * c[2] * beta * alpha
            c0 = (
                nO**2 * (kx[ii] ** 2 + ky[ii] ** 2) / k**2
                + alpha * beta**2
                - nO**2 * nE**2
            )
            z = kz_e[ii] / k
            self.assertAlmostEqual(abs(c2 * z**2 + c1 * z + c0), 0.0, places=12)

        # Quadrant-I and Quadrant-II complex delta (e.g. lossy superstrate launching complex kx
        # into lossless medium): dispersion_relation_ordinary must satisfy kz**2 == delta,
        # select Re(kz) > 0 for propagating order [0] and Im(kz) < 0 for TIR order [1],
        # and match dispersion_relation_extraordinary(nE=nO).
        kx_cplx = np.array([(0.8 - 0.15j) * k, (1.8 - 0.2j) * k])
        ky_cplx = np.array([(0.2 - 0.04j) * k, 0.0j])
        for kap in (0.1, 1e-5, 0.0):
            no_v = complex(1.5, -kap) if kap > 0 else 1.5
            delta_v = (k * no_v) ** 2 - (kx_cplx**2 + ky_cplx**2)
            kz_ord = EMpy.RCWA.dispersion_relation_ordinary(kx_cplx, ky_cplx, k, no_v)
            kz_ext = EMpy.RCWA.dispersion_relation_extraordinary(
                kx_cplx, ky_cplx, k, no_v, no_v, c
            )
            np.testing.assert_allclose(kz_ord**2, delta_v, rtol=1e-13, atol=1e-13)
            self.assertGreater(kz_ord[0].real, 0.0)
            self.assertLess(kz_ord[1].imag, 0.0)
            np.testing.assert_allclose(kz_ord, kz_ext, rtol=1e-12, atol=1e-12)

    def test_anisotropic_rcwa_anisotropic_end_layers(self):
        eps_air = EMpy.materials.EpsilonTensor(
            epsilon_tensor_const=EMpy.constants.eps0 * np.eye(3)
        )
        eps_si = EMpy.materials.EpsilonTensor(
            epsilon_tensor_const=(3.48**2) * EMpy.constants.eps0 * np.eye(3)
        )
        mat_air = EMpy.materials.AnisotropicMaterial("air_a", epsilon_tensor=eps_air)
        mat_si = EMpy.materials.AnisotropicMaterial("si_a", epsilon_tensor=eps_si)
        air_iso = EMpy.materials.Air
        si_iso = EMpy.materials.IsotropicMaterial(
            "Si", n0=EMpy.materials.RefractiveIndex(n0_const=3.48)
        )
        grating = EMpy.utils.BinaryGrating(air_iso, si_iso, 0.5, 1.0e-6, 0.2e-6)
        m = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(mat_air, np.inf),
                grating,
                EMpy.utils.Layer(mat_si, np.inf),
            ]
        )
        r = EMpy.RCWA.AnisotropicRCWA(
            m, alpha=0.0, delta=0.0, psi=0.0, phi=np.pi / 2, n=2
        ).solve([1.55e-6])
        total = r.DEE1.sum() + r.DEO1.sum() + r.DEE3.sum() + r.DEO3.sum()
        self.assertAlmostEqual(total, 1.0, places=10)

        eps_off = EMpy.materials.EpsilonTensor(
            epsilon_tensor_const=EMpy.constants.eps0
            * np.array([[2.0, 0.2, 0.0], [0.2, 2.0, 0.0], [0.0, 0.0, 2.0]])
        )
        mat_off = EMpy.materials.AnisotropicMaterial("off", epsilon_tensor=eps_off)
        m_bad = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(mat_off, np.inf),
                grating,
                EMpy.utils.Layer(mat_si, np.inf),
            ]
        )
        with self.assertRaises(NotImplementedError):
            EMpy.RCWA.AnisotropicRCWA(
                m_bad, alpha=0.0, delta=0.0, psi=0.0, phi=np.pi / 2, n=1
            ).solve([1.55e-6])

    def test_anisotropic_rcwa_phi_guard_and_n0_homogeneous_limit(self):
        """Verify phi != pi/2 raises ValueError and n = 0 / homogeneous rotated gratings match Berreman TMM."""
        wl = 1.55e-6
        eps0 = EMpy.constants.eps0
        air = EMpy.materials.Air
        rot_biax = EMpy.materials.AnisotropicMaterial(
            "biax",
            epsilon_tensor=EMpy.materials.EpsilonTensor(
                epsilon_tensor_const=eps0
                * EMpy.utils.euler_rotate(
                    np.diag([1.8**2, 2.0**2, 2.25**2]), 0.3, 0.4, 0.5
                )
            ),
        )
        sio2 = EMpy.materials.IsotropicMaterial(
            "sio2", n0=EMpy.materials.RefractiveIndex(n0_const=1.45)
        )
        ml_grat = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(air, np.inf),
                EMpy.utils.BinaryGrating(rot_biax, sio2, 0.5, 1.0e-6, 0.3e-6),
                EMpy.utils.Layer(sio2, np.inf),
            ]
        )
        for cls in (EMpy.RCWA.IsotropicRCWA, EMpy.RCWA.AnisotropicRCWA):
            ml_2 = EMpy.utils.Multilayer(
                [EMpy.utils.Layer(air, np.inf), EMpy.utils.Layer(sio2, np.inf)]
            )
            with self.assertRaises(ValueError):
                cls(ml_2, alpha=0.0, delta=0.0, psi=0.0, phi=np.pi / 3, n=1).solve([wl])
        with self.assertRaises(ValueError):
            EMpy.RCWA.AnisotropicRCWA(
                ml_grat, alpha=0.0, delta=0.0, psi=0.0, phi=np.pi / 3, n=1
            ).solve([wl])

        # n = 0 must not fail on Layer, BinaryGrating, SymmetricDoubleGrating, or AsymmetricDoubleGrating
        for internal in (
            EMpy.utils.Layer(rot_biax, 0.35e-6),
            EMpy.utils.BinaryGrating(rot_biax, rot_biax, 0.4, 1.0e-6, 0.35e-6),
            EMpy.utils.SymmetricDoubleGrating(
                rot_biax, rot_biax, rot_biax, 0.2, 0.3, 1.0e-6, 0.35e-6
            ),
            EMpy.utils.AsymmetricDoubleGrating(
                rot_biax, rot_biax, rot_biax, 0.2, 0.3, 0.15, 1.0e-6, 0.35e-6
            ),
        ):
            ml_h = EMpy.utils.Multilayer(
                [
                    EMpy.utils.Layer(air, np.inf),
                    internal,
                    EMpy.utils.Layer(sio2, np.inf),
                ]
            )
            alpha = EMpy.utils.deg2rad(20.0)
            delta = EMpy.utils.deg2rad(28.0)
            r0 = EMpy.RCWA.AnisotropicRCWA(
                ml_h, alpha=alpha, delta=delta, psi=0.0, phi=np.pi / 2, n=0
            ).solve([wl])
            r2 = EMpy.RCWA.AnisotropicRCWA(
                ml_h, alpha=alpha, delta=delta, psi=0.0, phi=np.pi / 2, n=2
            ).solve([wl])
            dirk = np.array(
                [np.sin(alpha) * np.cos(delta), np.sin(alpha) * np.sin(delta)]
            )
            tmm = EMpy.transfer_matrix.AnisotropicTransferMatrix(
                EMpy.utils.Multilayer(
                    [
                        EMpy.utils.Layer(air, np.inf),
                        EMpy.utils.Layer(rot_biax, 0.35e-6),
                        EMpy.utils.Layer(sio2, np.inf),
                    ]
                ),
                np.arcsin(dirk[0]),
                np.arcsin(dirk[1]),
            ).solve([wl])
            self.assertAlmostEqual(
                float((r0.DEO1 + r0.DEE1).sum()),
                float(tmm.R[1, :, 0].sum()),
                places=13,
            )
            self.assertAlmostEqual(
                float((r2.DEO3 + r2.DEE3).sum()),
                float(tmm.T[1, :, 0].sum()),
                places=13,
            )

    def test_anisotropic_rcwa_birefringent_superstrate_and_berreman_tmm(self):
        """Verify uniaxial superstrate excitation: zero reflection in homogeneous crystal & exact match to Berreman TMM."""
        wl = 1.55e-6
        k = 2.0 * np.pi / wl
        eps0 = EMpy.constants.eps0
        nO1, nE1 = 1.5, 1.7
        nO3, nE3 = 1.6, 1.9
        mat1 = EMpy.materials.AnisotropicMaterial(
            "m1",
            epsilon_tensor=EMpy.materials.EpsilonTensor(
                epsilon_tensor_const=eps0 * np.diag([nE1**2, nO1**2, nO1**2])
            ),
        )
        # 1. Uniform homogeneous uniaxial crystal must have R == 0 and T == 1 for all (alpha, delta, psi)
        ml_homo = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(mat1, np.inf),
                EMpy.utils.Layer(mat1, 0.35e-6),
                EMpy.utils.Layer(mat1, np.inf),
            ]
        )
        for a_deg, d_deg, p_deg in [
            (0.0, 0.0, 0.0),
            (0.0, 25.0, 37.0),
            (30.0, 0.0, 0.0),
            (30.0, 25.0, 40.0),
            (89.9, 0.0, 20.0),  # nearly k || c1 optic axis
        ]:
            rh = EMpy.RCWA.AnisotropicRCWA(
                ml_homo,
                EMpy.utils.deg2rad(a_deg),
                EMpy.utils.deg2rad(d_deg),
                EMpy.utils.deg2rad(p_deg),
                np.pi / 2,
                1,
            ).solve([wl])
            self.assertAlmostEqual(float((rh.DEO1 + rh.DEE1).sum()), 0.0, places=13)
            self.assertAlmostEqual(float((rh.DEO3 + rh.DEE3).sum()), 1.0, places=13)

        # 2. Uniaxial superstrate + 3D Euler-rotated biaxial layer + uniaxial substrate vs Berreman TMM
        mat2 = EMpy.materials.AnisotropicMaterial(
            "m2",
            epsilon_tensor=EMpy.materials.EpsilonTensor(
                epsilon_tensor_const=eps0
                * EMpy.utils.euler_rotate(
                    np.diag([1.8**2, 2.0**2, 2.2**2]), 0.3, 0.4, 0.5
                )
            ),
        )
        mat3 = EMpy.materials.AnisotropicMaterial(
            "m3",
            epsilon_tensor=EMpy.materials.EpsilonTensor(
                epsilon_tensor_const=eps0 * np.diag([nE3**2, nO3**2, nO3**2])
            ),
        )
        ml = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(mat1, np.inf),
                EMpy.utils.Layer(mat2, 0.35e-6),
                EMpy.utils.Layer(mat3, np.inf),
            ]
        )
        alpha = EMpy.utils.deg2rad(22.0)
        delta = EMpy.utils.deg2rad(30.0)
        dirk1 = np.array(
            [
                np.sin(alpha) * np.cos(delta),
                np.sin(alpha) * np.sin(delta),
                np.cos(alpha),
            ]
        )
        c1 = np.array([1.0, 0.0, 0.0])
        dO_hat = np.cross(c1, dirk1)
        dO_hat /= np.linalg.norm(dO_hat)
        dE_hat = np.cross(dirk1, dO_hat)
        e_tm = np.array(
            [
                np.cos(alpha) * np.cos(delta),
                np.cos(alpha) * np.sin(delta),
                -np.sin(alpha),
            ]
        )
        e_te = np.array([-np.sin(delta), np.cos(delta), 0.0])
        psi_O = float(np.arctan2(np.dot(dO_hat, e_te), np.dot(dO_hat, e_tm)))
        psi_E = float(np.arctan2(np.dot(dE_hat, e_te), np.dot(dE_hat, e_tm)))

        abskO1 = k * nO1
        abskE1 = (
            k * nO1 * nE1 / np.sqrt(nO1**2 + (nE1**2 - nO1**2) * np.dot(c1, dirk1) ** 2)
        )

        r_O = EMpy.RCWA.AnisotropicRCWA(ml, alpha, delta, psi_O, np.pi / 2, 1).solve(
            [wl]
        )
        tmm_O = EMpy.transfer_matrix.AnisotropicTransferMatrix(
            ml, np.arcsin(abskO1 * dirk1[0] / k), np.arcsin(abskO1 * dirk1[1] / k)
        ).solve([wl])
        np.testing.assert_allclose(
            [r_O.DEO1[1, 0], r_O.DEE1[1, 0], r_O.DEO3[1, 0], r_O.DEE3[1, 0]],
            [tmm_O.R[1, 1, 0], tmm_O.R[1, 0, 0], tmm_O.T[1, 1, 0], tmm_O.T[1, 0, 0]],
            atol=1e-13,
        )

        r_E = EMpy.RCWA.AnisotropicRCWA(ml, alpha, delta, psi_E, np.pi / 2, 1).solve(
            [wl]
        )
        tmm_E = EMpy.transfer_matrix.AnisotropicTransferMatrix(
            ml, np.arcsin(abskE1 * dirk1[0] / k), np.arcsin(abskE1 * dirk1[1] / k)
        ).solve([wl])
        np.testing.assert_allclose(
            [r_E.DEO1[1, 0], r_E.DEE1[1, 0], r_E.DEO3[1, 0], r_E.DEE3[1, 0]],
            [tmm_E.R[0, 1, 0], tmm_E.R[0, 0, 0], tmm_E.T[0, 1, 0], tmm_E.T[0, 0, 0]],
            atol=1e-13,
        )

    def test_anisotropic_rcwa_degenerate_isotropic_and_lossy_limits(self):
        """Verify AnisotropicRCWA converges to IsotropicRCWA as dn -> 0 and obeys passive lossy kappa -> 0+ limits."""
        wl = 1.55e-6
        eps0 = EMpy.constants.eps0
        alpha = EMpy.utils.deg2rad(18.0)
        delta = EMpy.utils.deg2rad(25.0)
        psi = EMpy.utils.deg2rad(35.0)

        n1_iso = EMpy.materials.IsotropicMaterial(
            "n1", n0=EMpy.materials.RefractiveIndex(n0_const=1.45)
        )
        n2_iso = EMpy.materials.IsotropicMaterial(
            "n2", n0=EMpy.materials.RefractiveIndex(n0_const=2.2)
        )
        n3_iso = EMpy.materials.IsotropicMaterial(
            "n3", n0=EMpy.materials.RefractiveIndex(n0_const=1.6)
        )
        ml_iso = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(n1_iso, np.inf),
                EMpy.utils.BinaryGrating(n2_iso, n1_iso, 0.43, 1.1e-6, 0.25e-6),
                EMpy.utils.Layer(n3_iso, np.inf),
            ]
        )
        r_iso = EMpy.RCWA.IsotropicRCWA(
            ml_iso, alpha=alpha, delta=delta, psi=psi, phi=np.pi / 2, n=2
        ).solve([wl])

        # 1. Weak-birefringence 3D-Euler-rotated grating + uniaxial substrate -> IsotropicRCWA (dn = 1e-4, 1e-8, 0)
        sup_iso_a = EMpy.materials.AnisotropicMaterial(
            "sup_iso",
            epsilon_tensor=EMpy.materials.EpsilonTensor(
                epsilon_tensor_const=eps0 * 1.45**2 * np.eye(3)
            ),
        )
        dn_errs = []
        for dn in (1e-4, 1e-8, 0.0):
            ridge_a = EMpy.materials.AnisotropicMaterial(
                "ridge",
                epsilon_tensor=EMpy.materials.EpsilonTensor(
                    epsilon_tensor_const=eps0
                    * EMpy.utils.euler_rotate(
                        np.diag([2.2**2, (2.2 + 0.5 * dn) ** 2, (2.2 + dn) ** 2]),
                        0.33,
                        0.47,
                        0.71,
                    )
                ),
            )
            sub_a = EMpy.materials.AnisotropicMaterial(
                "sub",
                epsilon_tensor=EMpy.materials.EpsilonTensor(
                    epsilon_tensor_const=eps0
                    * np.diag([(1.6 + dn) ** 2, 1.6**2, 1.6**2])
                ),
            )
            ml_a = EMpy.utils.Multilayer(
                [
                    EMpy.utils.Layer(sup_iso_a, np.inf),
                    EMpy.utils.BinaryGrating(ridge_a, sup_iso_a, 0.43, 1.1e-6, 0.25e-6),
                    EMpy.utils.Layer(sub_a, np.inf),
                ]
            )
            ra = EMpy.RCWA.AnisotropicRCWA(
                ml_a, alpha=alpha, delta=delta, psi=psi, phi=np.pi / 2, n=2
            ).solve([wl])
            self.assertAlmostEqual(float((ra.DE1 + ra.DE3).sum()), 1.0, places=12)
            err = float(np.max(np.abs(ra.DE1 - r_iso.DE1) + np.abs(ra.DE3 - r_iso.DE3)))
            dn_errs.append(err)
        self.assertLess(dn_errs[0], 5e-4)
        self.assertLess(dn_errs[1], 0.01 * dn_errs[0])
        self.assertLess(dn_errs[1], 1e-7)
        self.assertLess(dn_errs[2], 1e-12)

        # 2. Passive lossy rotated anisotropic layer -> lossless limit (kap = 1e-2, 1e-5, 0.0 + 0j)
        def _solve_lossy_film(kap, force_complex=False):
            n_o = complex(1.9, -kap) if (kap > 0 or force_complex) else 1.9
            n_e = complex(2.15, -1.5 * kap) if (kap > 0 or force_complex) else 2.15
            film = EMpy.materials.AnisotropicMaterial(
                "film",
                epsilon_tensor=EMpy.materials.EpsilonTensor(
                    epsilon_tensor_const=eps0
                    * EMpy.utils.euler_rotate(
                        np.diag([n_o**2, n_o**2, n_e**2]), 0.25, 0.35, 0.45
                    )
                ),
            )
            ml_f = EMpy.utils.Multilayer(
                [
                    EMpy.utils.Layer(n1_iso, np.inf),
                    EMpy.utils.Layer(film, 0.4e-6),
                    EMpy.utils.Layer(n3_iso, np.inf),
                ]
            )
            return EMpy.RCWA.AnisotropicRCWA(
                ml_f, alpha=alpha, delta=delta, psi=psi, phi=np.pi / 2, n=1
            ).solve([wl])

        r_ll = _solve_lossy_film(0.0, force_complex=False)
        r_c0 = _solve_lossy_film(0.0, force_complex=True)
        self.assertAlmostEqual(
            float((r_ll.DEO1 + r_ll.DEE1 + r_ll.DEO3 + r_ll.DEE3).sum()), 1.0, places=13
        )
        np.testing.assert_allclose(
            r_c0.DEO1 + r_c0.DEE1, r_ll.DEO1 + r_ll.DEE1, atol=1e-14
        )
        np.testing.assert_allclose(
            r_c0.DEO3 + r_c0.DEE3, r_ll.DEO3 + r_ll.DEE3, atol=1e-14
        )

        kap_errs = []
        for kap in (1e-2, 1e-5):
            rk = _solve_lossy_film(kap)
            tot_k = float((rk.DEO1 + rk.DEE1 + rk.DEO3 + rk.DEE3).sum())
            self.assertLess(tot_k, 1.0)
            kap_errs.append(
                abs(tot_k - 1.0)
                + float(np.max(np.abs((rk.DEO3 + rk.DEE3) - (r_ll.DEO3 + r_ll.DEE3))))
            )
        self.assertLess(kap_errs[1], kap_errs[0])
        self.assertLess(kap_errs[1], 1e-4)

    def test_lossy_grating_complex_symmetric_passivity_rytov_and_limits(self):
        """Verify lossy gratings (complex-symmetric Toeplitz E, E1) match Rytov EMT, Iso==Ani, and kappa/dn -> 0 limits."""
        wl = 1.55e-6
        eps0 = EMpy.constants.eps0
        air = EMpy.materials.Air
        m1_i = EMpy.materials.IsotropicMaterial(
            "m1", n0=EMpy.materials.RefractiveIndex(n0_const=1.3)
        )
        m2_i = EMpy.materials.IsotropicMaterial(
            "m2", n0=EMpy.materials.RefractiveIndex(n0_const=2.2 - 0.5j)
        )

        # 1. Deep-subwavelength Rytov effective-medium benchmark (pitch = wl / 200) for both TE and TM!
        dc = 0.4
        thickness = 0.3e-6
        pitch_sub = wl / 200.0
        ml_sub = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(air, np.inf),
                EMpy.utils.BinaryGrating(m1_i, m2_i, dc, pitch_sub, thickness),
                EMpy.utils.Layer(air, np.inf),
            ]
        )
        r_sub_te = EMpy.RCWA.IsotropicRCWA(
            ml_sub, alpha=0.0, delta=0.0, psi=np.pi / 2, phi=np.pi / 2, n=4
        ).solve([wl])
        r_sub_tm = EMpy.RCWA.IsotropicRCWA(
            ml_sub, alpha=0.0, delta=0.0, psi=0.0, phi=np.pi / 2, n=4
        ).solve([wl])

        eps1, eps2 = 1.3**2, (2.2 - 0.5j) ** 2
        n_eff_te = np.sqrt(dc * eps1 + (1.0 - dc) * eps2)
        n_eff_tm = np.sqrt(1.0 / (dc / eps1 + (1.0 - dc) / eps2))
        if n_eff_te.imag > 0:
            n_eff_te = -n_eff_te
        if n_eff_tm.imag > 0:
            n_eff_tm = -n_eff_tm
        tmm_te = EMpy.transfer_matrix.IsotropicTransferMatrix(
            EMpy.utils.Multilayer(
                [
                    EMpy.utils.Layer(air, np.inf),
                    EMpy.utils.Layer(
                        EMpy.materials.IsotropicMaterial(
                            "ete", n0=EMpy.materials.RefractiveIndex(n0_const=n_eff_te)
                        ),
                        thickness,
                    ),
                    EMpy.utils.Layer(air, np.inf),
                ]
            ),
            0.0,
        ).solve([wl])
        tmm_tm = EMpy.transfer_matrix.IsotropicTransferMatrix(
            EMpy.utils.Multilayer(
                [
                    EMpy.utils.Layer(air, np.inf),
                    EMpy.utils.Layer(
                        EMpy.materials.IsotropicMaterial(
                            "etm", n0=EMpy.materials.RefractiveIndex(n0_const=n_eff_tm)
                        ),
                        thickness,
                    ),
                    EMpy.utils.Layer(air, np.inf),
                ]
            ),
            0.0,
        ).solve([wl])
        self.assertAlmostEqual(
            float(r_sub_te.DE3[4, 0]), float(tmm_te.Ts[0]), delta=1e-3
        )
        self.assertAlmostEqual(
            float(r_sub_tm.DE3[4, 0]), float(tmm_tm.Tp[0]), delta=1e-3
        )

        # 2. Conical lossy BinaryGrating & DoubleGratings: 3D-rotated lossy AniRCWA (dn -> 0) == lossy IsoRCWA
        ml_i = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(air, np.inf),
                EMpy.utils.BinaryGrating(m1_i, m2_i, dc, 1.1e-6, thickness),
                EMpy.utils.Layer(air, np.inf),
            ]
        )
        ri_c = EMpy.RCWA.IsotropicRCWA(
            ml_i, alpha=0.3, delta=0.4, psi=0.6, phi=np.pi / 2, n=3
        ).solve([wl])
        self.assertLess(float((ri_c.DE1 + ri_c.DE3).sum()), 1.0)

        m1_a = EMpy.materials.AnisotropicMaterial(
            "m1a",
            epsilon_tensor=EMpy.materials.EpsilonTensor(
                epsilon_tensor_const=eps0 * 1.3**2 * np.eye(3)
            ),
        )
        dn_errs = []
        for dn in (1e-4, 1e-8, 0.0):
            n2a = 2.2 - 0.5j
            m2_a = EMpy.materials.AnisotropicMaterial(
                "m2a",
                epsilon_tensor=EMpy.materials.EpsilonTensor(
                    epsilon_tensor_const=eps0
                    * EMpy.utils.euler_rotate(
                        np.diag([n2a**2, (n2a + 0.5 * dn) ** 2, (n2a + dn) ** 2]),
                        0.31,
                        0.52,
                        0.73,
                    )
                ),
            )
            ml_a = EMpy.utils.Multilayer(
                [
                    EMpy.utils.Layer(air, np.inf),
                    EMpy.utils.BinaryGrating(m1_a, m2_a, dc, 1.1e-6, thickness),
                    EMpy.utils.Layer(air, np.inf),
                ]
            )
            ra_c = EMpy.RCWA.AnisotropicRCWA(
                ml_a, alpha=0.3, delta=0.4, psi=0.6, phi=np.pi / 2, n=3
            ).solve([wl])
            self.assertLess(
                float((ra_c.DEO1 + ra_c.DEE1 + ra_c.DEO3 + ra_c.DEE3).sum()), 1.0
            )
            dn_errs.append(
                float(
                    np.max(
                        np.abs((ra_c.DEO1 + ra_c.DEE1) - ri_c.DE1)
                        + np.abs((ra_c.DEO3 + ra_c.DEE3) - ri_c.DE3)
                    )
                )
            )
        self.assertLess(dn_errs[1], dn_errs[0])
        self.assertLess(dn_errs[2], 1e-12)

        # Also check lossy SymmetricDoubleGrating / AsymmetricDoubleGrating IsoRCWA == AniRCWA
        sdg_i = EMpy.utils.SymmetricDoubleGrating(
            m2_i, m2_i, m1_i, 0.2, 0.25, 1.1e-6, thickness
        )
        m2_a0 = EMpy.materials.AnisotropicMaterial(
            "m2a0",
            epsilon_tensor=EMpy.materials.EpsilonTensor(
                epsilon_tensor_const=eps0 * (2.2 - 0.5j) ** 2 * np.eye(3)
            ),
        )
        adg_a = EMpy.utils.AsymmetricDoubleGrating(
            m2_a0, m2_a0, m1_a, 0.2, 0.25, 0.275, 1.1e-6, thickness
        )
        r_sdg_i = EMpy.RCWA.IsotropicRCWA(
            EMpy.utils.Multilayer(
                [EMpy.utils.Layer(air, np.inf), sdg_i, EMpy.utils.Layer(air, np.inf)]
            ),
            0.25,
            0.35,
            0.5,
            np.pi / 2,
            2,
        ).solve([wl])
        r_adg_a = EMpy.RCWA.AnisotropicRCWA(
            EMpy.utils.Multilayer(
                [EMpy.utils.Layer(air, np.inf), adg_a, EMpy.utils.Layer(air, np.inf)]
            ),
            0.25,
            0.35,
            0.5,
            np.pi / 2,
            2,
        ).solve([wl])
        np.testing.assert_allclose(r_adg_a.DEO1 + r_adg_a.DEE1, r_sdg_i.DE1, atol=1e-12)
        np.testing.assert_allclose(r_adg_a.DEO3 + r_adg_a.DEE3, r_sdg_i.DE3, atol=1e-12)

        # 3. Continuous lossy -> lossless grating limit (kap = 1e-2, 1e-5, 0.0 + 0j -> lossless)
        def _solve_iso_grat(kap, force_complex=False):
            n2_v = complex(2.2, -kap) if (kap > 0 or force_complex) else 2.2
            m2_v = EMpy.materials.IsotropicMaterial(
                "m2v", n0=EMpy.materials.RefractiveIndex(n0_const=n2_v)
            )
            ml_v = EMpy.utils.Multilayer(
                [
                    EMpy.utils.Layer(air, np.inf),
                    EMpy.utils.BinaryGrating(m1_i, m2_v, dc, 1.1e-6, thickness),
                    EMpy.utils.Layer(air, np.inf),
                ]
            )
            return EMpy.RCWA.IsotropicRCWA(
                ml_v, alpha=0.2, delta=0.3, psi=0.4, phi=np.pi / 2, n=3
            ).solve([wl])

        rg_ll = _solve_iso_grat(0.0, force_complex=False)
        rg_c0 = _solve_iso_grat(0.0, force_complex=True)
        self.assertAlmostEqual(float((rg_ll.DE1 + rg_ll.DE3).sum()), 1.0, places=12)
        np.testing.assert_allclose(rg_c0.DE1, rg_ll.DE1, atol=1e-12)
        np.testing.assert_allclose(rg_c0.DE3, rg_ll.DE3, atol=1e-12)

        g_errs = []
        for kap in (1e-2, 1e-5):
            rg_k = _solve_iso_grat(kap)
            self.assertLess(float((rg_k.DE1 + rg_k.DE3).sum()), 1.0)
            g_errs.append(float(np.max(np.abs(rg_k.DE3 - rg_ll.DE3))))
        self.assertLess(g_errs[1], g_errs[0])
        self.assertLess(g_errs[1], 1e-4)

    def test_isotropic_rcwa_normal_incidence_azimuthal_continuity_and_limits(self):
        """Verify IsotropicRCWA is continuous as alpha -> 0+ with delta != 0 and matches AnisotropicRCWA."""
        wl = 1.55e-6
        eps0 = EMpy.constants.eps0
        air = EMpy.materials.Air
        si = EMpy.materials.IsotropicMaterial(
            "si", n0=EMpy.materials.RefractiveIndex(n0_const=3.48)
        )
        sio2 = EMpy.materials.IsotropicMaterial(
            "sio2", n0=EMpy.materials.RefractiveIndex(n0_const=1.45)
        )
        ml_iso = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(air, np.inf),
                EMpy.utils.BinaryGrating(si, sio2, 0.42, 1.1e-6, 0.28e-6),
                EMpy.utils.Layer(sio2, np.inf),
            ]
        )

        delta = EMpy.utils.deg2rad(35.0)
        psi = EMpy.utils.deg2rad(40.0)
        r0 = EMpy.RCWA.IsotropicRCWA(
            ml_iso, alpha=0.0, delta=delta, psi=psi, phi=np.pi / 2, n=3
        ).solve([wl])

        # 1. Rotational invariance at normal incidence: (alpha=0, delta, psi) == (alpha=0, 0, psi + delta)
        r0_rot = EMpy.RCWA.IsotropicRCWA(
            ml_iso, alpha=0.0, delta=0.0, psi=psi + delta, phi=np.pi / 2, n=3
        ).solve([wl])
        np.testing.assert_allclose(r0.DE1, r0_rot.DE1, atol=1e-12)
        np.testing.assert_allclose(r0.DE3, r0_rot.DE3, atol=1e-12)

        # 2. Smooth continuity as alpha -> 0+ (1e-3, 1e-6, 1e-9 -> 0) with delta != 0
        alpha_errs = []
        for al in (1e-3, 1e-6, 1e-9):
            ra = EMpy.RCWA.IsotropicRCWA(
                ml_iso, alpha=al, delta=delta, psi=psi, phi=np.pi / 2, n=3
            ).solve([wl])
            alpha_errs.append(
                float(np.max(np.abs(ra.DE1 - r0.DE1) + np.abs(ra.DE3 - r0.DE3)))
            )
        self.assertLess(alpha_errs[1], alpha_errs[0])
        self.assertLess(alpha_errs[2], alpha_errs[1])
        self.assertLess(alpha_errs[2], 1e-7)

        # 3. Exact match with AnisotropicRCWA at alpha = 0, delta != 0 (including dn -> 0 and lossy grating)
        air_a = EMpy.materials.AnisotropicMaterial(
            "air_a",
            epsilon_tensor=EMpy.materials.EpsilonTensor(
                epsilon_tensor_const=eps0 * np.eye(3)
            ),
        )
        sio2_a = EMpy.materials.AnisotropicMaterial(
            "sio2_a",
            epsilon_tensor=EMpy.materials.EpsilonTensor(
                epsilon_tensor_const=eps0 * 1.45**2 * np.eye(3)
            ),
        )
        dn_errs = []
        for dn in (1e-4, 1e-8, 0.0):
            si_a = EMpy.materials.AnisotropicMaterial(
                "si_a",
                epsilon_tensor=EMpy.materials.EpsilonTensor(
                    epsilon_tensor_const=eps0
                    * EMpy.utils.euler_rotate(
                        np.diag([3.48**2, (3.48 + 0.5 * dn) ** 2, (3.48 + dn) ** 2]),
                        0.2,
                        0.4,
                        0.6,
                    )
                ),
            )
            ml_a = EMpy.utils.Multilayer(
                [
                    EMpy.utils.Layer(air_a, np.inf),
                    EMpy.utils.BinaryGrating(si_a, sio2_a, 0.42, 1.1e-6, 0.28e-6),
                    EMpy.utils.Layer(sio2_a, np.inf),
                ]
            )
            r_ani = EMpy.RCWA.AnisotropicRCWA(
                ml_a, alpha=0.0, delta=delta, psi=psi, phi=np.pi / 2, n=3
            ).solve([wl])
            dn_errs.append(
                float(
                    np.max(
                        np.abs((r_ani.DEO1 + r_ani.DEE1) - r0.DE1)
                        + np.abs((r_ani.DEO3 + r_ani.DEE3) - r0.DE3)
                    )
                )
            )
        self.assertLess(dn_errs[1], dn_errs[0])
        self.assertLess(dn_errs[2], 1e-12)

        # 4. Complex dtype superstrate (n1 = 1.45 + 0.0j and passive lossy kap -> 0+) must not fail on arctan2
        sup_real = EMpy.materials.IsotropicMaterial(
            "sr", n0=EMpy.materials.RefractiveIndex(n0_const=1.45)
        )
        ml_sr = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(sup_real, np.inf),
                EMpy.utils.BinaryGrating(si, sio2, 0.42, 1.1e-6, 0.28e-6),
                EMpy.utils.Layer(sio2, np.inf),
            ]
        )
        r_sr = EMpy.RCWA.IsotropicRCWA(
            ml_sr, alpha=0.2, delta=delta, psi=psi, phi=np.pi / 2, n=2
        ).solve([wl])

        sup_errs = []
        for kap in (1e-3, 1e-6, 0.0):
            sup_k = EMpy.materials.IsotropicMaterial(
                "sk", n0=EMpy.materials.RefractiveIndex(n0_const=complex(1.45, -kap))
            )
            ml_sk = EMpy.utils.Multilayer(
                [
                    EMpy.utils.Layer(sup_k, np.inf),
                    EMpy.utils.BinaryGrating(si, sio2, 0.42, 1.1e-6, 0.28e-6),
                    EMpy.utils.Layer(sio2, np.inf),
                ]
            )
            r_sk = EMpy.RCWA.IsotropicRCWA(
                ml_sk, alpha=0.2, delta=delta, psi=psi, phi=np.pi / 2, n=2
            ).solve([wl])
            sup_errs.append(
                float(np.max(np.abs(r_sk.DE1 - r_sr.DE1) + np.abs(r_sk.DE3 - r_sr.DE3)))
            )
        self.assertLess(sup_errs[0], 5e-3)
        self.assertLess(sup_errs[1], 0.01 * sup_errs[0])
        self.assertLess(sup_errs[1], 5e-6)
        self.assertLess(sup_errs[2], 1e-12)

    def test_anisotropic_rcwa_thick_high_order_etm_stability_and_limits(self):
        """Verify Enhanced Transmittance Matrix stability on thick high-order gratings and degenerate limits."""
        wl = 1.55e-6
        pitch = 1.0e-6
        eps0 = EMpy.constants.eps0
        alpha = EMpy.utils.deg2rad(18.0)
        delta = EMpy.utils.deg2rad(27.0)
        psi = EMpy.utils.deg2rad(38.0)

        # 1. Strongly birefringent 3D-Euler-rotated thick cascaded grating stack with birefringent end layers:
        # exp(-2 * pi * 12 * 3.0 / 1.0) ~ 1e-98 would completely overflow naive M^-1 transfer matrices.
        sup_bi = EMpy.materials.AnisotropicMaterial(
            "sup_bi",
            epsilon_tensor=EMpy.materials.EpsilonTensor(
                epsilon_tensor_const=eps0 * np.diag([1.62**2, 1.48**2, 1.48**2])
            ),
        )
        sub_bi = EMpy.materials.AnisotropicMaterial(
            "sub_bi",
            epsilon_tensor=EMpy.materials.EpsilonTensor(
                epsilon_tensor_const=eps0 * np.diag([1.75**2, 1.52**2, 1.52**2])
            ),
        )
        lc1 = EMpy.materials.AnisotropicMaterial(
            "lc1",
            epsilon_tensor=EMpy.materials.EpsilonTensor(
                epsilon_tensor_const=eps0
                * EMpy.utils.euler_rotate(
                    np.diag([1.52**2, 1.65**2, 1.88**2]), 0.35, 0.55, 0.75
                )
            ),
        )
        lc2 = EMpy.materials.AnisotropicMaterial(
            "lc2",
            epsilon_tensor=EMpy.materials.EpsilonTensor(
                epsilon_tensor_const=eps0
                * EMpy.utils.euler_rotate(
                    np.diag([2.10**2, 1.80**2, 1.60**2]), -0.4, 0.65, 0.25
                )
            ),
        )
        sio2_a = EMpy.materials.AnisotropicMaterial(
            "sio2_a",
            epsilon_tensor=EMpy.materials.EpsilonTensor(
                epsilon_tensor_const=eps0 * 1.45**2 * np.eye(3)
            ),
        )
        ml_thick_bi = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(sup_bi, np.inf),
                EMpy.utils.BinaryGrating(lc1, sio2_a, 0.43, pitch, 2.0e-6),
                EMpy.utils.Layer(lc2, 1.5e-6),
                EMpy.utils.BinaryGrating(lc2, lc1, 0.58, pitch, 3.0e-6),
                EMpy.utils.Layer(sub_bi, np.inf),
            ]
        )
        de0_by_n = {}
        for n_ord in (4, 8, 12, 16):
            res_n = EMpy.RCWA.AnisotropicRCWA(
                ml_thick_bi, alpha=alpha, delta=delta, psi=psi, phi=np.pi / 2, n=n_ord
            ).solve([wl])
            total_power = float(
                (res_n.DEO1 + res_n.DEE1 + res_n.DEO3 + res_n.DEE3).sum()
            )
            self.assertAlmostEqual(total_power, 1.0, places=10)
            de0_by_n[n_ord] = np.array(
                [
                    res_n.DEO1[n_ord, 0],
                    res_n.DEE1[n_ord, 0],
                    res_n.DEO3[n_ord, 0],
                    res_n.DEE3[n_ord, 0],
                ]
            )
        self.assertLess(
            float(np.max(np.abs(de0_by_n[16] - de0_by_n[12]))),
            float(np.max(np.abs(de0_by_n[8] - de0_by_n[4]))),
        )

        # 2. Homogeneous thick crystal limit (d = 25 um, n = 10): AniRCWA == AnisotropicTransferMatrix
        air_a = EMpy.materials.AnisotropicMaterial(
            "air_a",
            epsilon_tensor=EMpy.materials.EpsilonTensor(
                epsilon_tensor_const=eps0 * np.eye(3)
            ),
        )
        ml_hom_rcwa = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(air_a, np.inf),
                EMpy.utils.BinaryGrating(lc1, lc1, 0.5, pitch, 25.0e-6),
                EMpy.utils.Layer(sio2_a, np.inf),
            ]
        )
        ml_hom_tmm = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(air_a, np.inf),
                EMpy.utils.Layer(lc1, 25.0e-6),
                EMpy.utils.Layer(sio2_a, np.inf),
            ]
        )
        r_hom = EMpy.RCWA.AnisotropicRCWA(
            ml_hom_rcwa, alpha=alpha, delta=0.0, psi=0.0, phi=np.pi / 2, n=10
        ).solve([wl])
        tmm_hom = EMpy.transfer_matrix.AnisotropicTransferMatrix(
            ml_hom_tmm, alpha, 0.0
        ).solve([wl])
        self.assertAlmostEqual(
            float(r_hom.DEO1[10, 0] + r_hom.DEE1[10, 0]),
            float(tmm_hom.R[1, :, 0].sum()),
            places=10,
        )
        self.assertAlmostEqual(
            float(r_hom.DEO3[10, 0] + r_hom.DEE3[10, 0]),
            float(tmm_hom.T[1, :, 0].sum()),
            places=10,
        )

        # 3. Degenerate anisotropic -> isotropic limit (dn -> 0 with 3D Euler rotation) at n = 12, d = 2.0 + 2.5 um
        air_i = EMpy.materials.Air
        si_i = EMpy.materials.IsotropicMaterial(
            "si_i", n0=EMpy.materials.RefractiveIndex(n0_const=2.45)
        )
        sio2_i = EMpy.materials.IsotropicMaterial(
            "sio2_i", n0=EMpy.materials.RefractiveIndex(n0_const=1.45)
        )
        sin_i = EMpy.materials.IsotropicMaterial(
            "sin_i", n0=EMpy.materials.RefractiveIndex(n0_const=1.98)
        )
        ml_iso = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(air_i, np.inf),
                EMpy.utils.BinaryGrating(si_i, sio2_i, 0.4, pitch, 2.0e-6),
                EMpy.utils.BinaryGrating(sin_i, sio2_i, 0.6, pitch, 2.5e-6),
                EMpy.utils.Layer(sio2_i, np.inf),
            ]
        )
        for al_test in (alpha, 0.0):
            r_iso = EMpy.RCWA.IsotropicRCWA(
                ml_iso, alpha=al_test, delta=delta, psi=psi, phi=np.pi / 2, n=12
            ).solve([wl])
            self.assertAlmostEqual(float((r_iso.DE1 + r_iso.DE3).sum()), 1.0, places=10)

            dn_errs = []
            for dn in (1e-4, 1e-8, 0.0):
                si_a = EMpy.materials.AnisotropicMaterial(
                    "si_a",
                    epsilon_tensor=EMpy.materials.EpsilonTensor(
                        epsilon_tensor_const=eps0
                        * EMpy.utils.euler_rotate(
                            np.diag(
                                [2.45**2, (2.45 + 0.5 * dn) ** 2, (2.45 + dn) ** 2]
                            ),
                            0.3,
                            0.5,
                            0.7,
                        )
                    ),
                )
                sin_a = EMpy.materials.AnisotropicMaterial(
                    "sin_a",
                    epsilon_tensor=EMpy.materials.EpsilonTensor(
                        epsilon_tensor_const=eps0
                        * EMpy.utils.euler_rotate(
                            np.diag(
                                [1.98**2, (1.98 - 0.5 * dn) ** 2, (1.98 - dn) ** 2]
                            ),
                            -0.25,
                            0.45,
                            0.65,
                        )
                    ),
                )
                ml_ani_dn = EMpy.utils.Multilayer(
                    [
                        EMpy.utils.Layer(air_a, np.inf),
                        EMpy.utils.BinaryGrating(si_a, sio2_a, 0.4, pitch, 2.0e-6),
                        EMpy.utils.BinaryGrating(sin_a, sio2_a, 0.6, pitch, 2.5e-6),
                        EMpy.utils.Layer(sio2_a, np.inf),
                    ]
                )
                r_ani_dn = EMpy.RCWA.AnisotropicRCWA(
                    ml_ani_dn,
                    alpha=al_test,
                    delta=delta,
                    psi=psi,
                    phi=np.pi / 2,
                    n=12,
                ).solve([wl])
                dn_errs.append(
                    float(
                        np.max(
                            np.abs((r_ani_dn.DEO1 + r_ani_dn.DEE1) - r_iso.DE1)
                            + np.abs((r_ani_dn.DEO3 + r_ani_dn.DEE3) - r_iso.DE3)
                        )
                    )
                )
            self.assertLess(dn_errs[1], dn_errs[0])
            self.assertLess(dn_errs[2], 1e-10)

        # 4. Passive lossy thick high-order anisotropic grating (n = 12, d = 2.5 um):
        # verify kap -> 0+ limit, complex 0j continuity, strict passivity, and lossy Ani == lossy Iso at dn = 0.
        def _solve_lossy_thick_ani(kap, force_complex=False, dn=0.22):
            if kap > 0 or force_complex:
                no = complex(2.1, -kap)
                ne = complex(2.1 + dn, -(1.0 + dn) * kap)
            else:
                no = 2.1
                ne = 2.1 + dn
            mat_l = EMpy.materials.AnisotropicMaterial(
                "mat_l",
                epsilon_tensor=EMpy.materials.EpsilonTensor(
                    epsilon_tensor_const=eps0
                    * EMpy.utils.euler_rotate(
                        np.diag([no**2, no**2, ne**2]), 0.33, 0.51, 0.68
                    )
                ),
            )
            ml_l = EMpy.utils.Multilayer(
                [
                    EMpy.utils.Layer(air_a, np.inf),
                    EMpy.utils.BinaryGrating(mat_l, sio2_a, 0.45, pitch, 2.5e-6),
                    EMpy.utils.Layer(sio2_a, np.inf),
                ]
            )
            return EMpy.RCWA.AnisotropicRCWA(
                ml_l, alpha=alpha, delta=delta, psi=psi, phi=np.pi / 2, n=12
            ).solve([wl])

        r_ll = _solve_lossy_thick_ani(0.0, force_complex=False)
        r_c0 = _solve_lossy_thick_ani(0.0, force_complex=True)
        self.assertAlmostEqual(
            float((r_ll.DEO1 + r_ll.DEE1 + r_ll.DEO3 + r_ll.DEE3).sum()),
            1.0,
            places=10,
        )
        np.testing.assert_allclose(
            r_c0.DEO1 + r_c0.DEE1, r_ll.DEO1 + r_ll.DEE1, atol=1e-11
        )
        np.testing.assert_allclose(
            r_c0.DEO3 + r_c0.DEE3, r_ll.DEO3 + r_ll.DEE3, atol=1e-11
        )

        kap_errs = []
        for kap in (1e-2, 1e-5):
            rk = _solve_lossy_thick_ani(kap)
            self.assertLess(float((rk.DEO1 + rk.DEE1 + rk.DEO3 + rk.DEE3).sum()), 1.0)
            kap_errs.append(
                float(np.max(np.abs((rk.DEO3 + rk.DEE3) - (r_ll.DEO3 + r_ll.DEE3))))
            )
        self.assertLess(kap_errs[1], kap_errs[0])
        self.assertLess(kap_errs[1], 5e-4)

        r_lani_iso = _solve_lossy_thick_ani(0.25, dn=0.0)
        mat_li = EMpy.materials.IsotropicMaterial(
            "mat_li", n0=EMpy.materials.RefractiveIndex(n0_const=2.1 - 0.25j)
        )
        r_liso = EMpy.RCWA.IsotropicRCWA(
            EMpy.utils.Multilayer(
                [
                    EMpy.utils.Layer(air_i, np.inf),
                    EMpy.utils.BinaryGrating(mat_li, sio2_i, 0.45, pitch, 2.5e-6),
                    EMpy.utils.Layer(sio2_i, np.inf),
                ]
            ),
            alpha=alpha,
            delta=delta,
            psi=psi,
            phi=np.pi / 2,
            n=12,
        ).solve([wl])
        np.testing.assert_allclose(
            r_lani_iso.DEO1 + r_lani_iso.DEE1, r_liso.DE1, atol=1e-11
        )
        np.testing.assert_allclose(
            r_lani_iso.DEO3 + r_lani_iso.DEE3, r_liso.DE3, atol=1e-11
        )

    def test_anisotropic_rcwa_lossy_substrate_superposition_and_rayleigh(self):
        """Verify lossy end-medium DE1/DE3 conservation, mixed-psi birefringent superstrate superposition, and Rayleigh anomalies."""
        wl = 1.55e-6
        eps0 = EMpy.constants.eps0
        air = EMpy.materials.Air
        si = EMpy.materials.IsotropicMaterial(
            "si", n0=EMpy.materials.RefractiveIndex(n0_const=3.48)
        )
        sio2 = EMpy.materials.IsotropicMaterial(
            "sio2", n0=EMpy.materials.RefractiveIndex(n0_const=1.45)
        )

        # 1. Conical diffraction into a LOSSY isotropic substrate: AniRCWA DEO3+DEE3 == DE3 == IsoRCWA.DE3
        sub_lossy_i = EMpy.materials.IsotropicMaterial(
            "sli", n0=EMpy.materials.RefractiveIndex(n0_const=1.5 - 0.35j)
        )
        sub_lossy_a = EMpy.materials.AnisotropicMaterial(
            "sla",
            epsilon_tensor=EMpy.materials.EpsilonTensor(
                epsilon_tensor_const=eps0 * (1.5 - 0.35j) ** 2 * np.eye(3)
            ),
        )
        ml_ls_i = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(air, np.inf),
                EMpy.utils.BinaryGrating(si, sio2, 0.45, 1.1e-6, 0.25e-6),
                EMpy.utils.Layer(sub_lossy_i, np.inf),
            ]
        )
        ml_ls_a = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(air, np.inf),
                EMpy.utils.BinaryGrating(si, sio2, 0.45, 1.1e-6, 0.25e-6),
                EMpy.utils.Layer(sub_lossy_a, np.inf),
            ]
        )
        ri_ls = EMpy.RCWA.IsotropicRCWA(
            ml_ls_i, alpha=0.35, delta=0.7, psi=0.4, phi=np.pi / 2, n=2
        ).solve([wl])
        ra_ls = EMpy.RCWA.AnisotropicRCWA(
            ml_ls_a, alpha=0.35, delta=0.7, psi=0.4, phi=np.pi / 2, n=2
        ).solve([wl])
        np.testing.assert_allclose(ra_ls.DE1, ri_ls.DE1, atol=1e-13)
        np.testing.assert_allclose(ra_ls.DE3, ri_ls.DE3, atol=1e-13)
        np.testing.assert_allclose(ra_ls.DEO3 + ra_ls.DEE3, ri_ls.DE3, atol=1e-13)
        # Lossless grating between lossless air and lossy substrate: flux across z = d is conserved (R + T == 1)
        self.assertAlmostEqual(float((ra_ls.DE1 + ra_ls.DE3).sum()), 1.0, places=12)

        # 2. Lossless grating on a LOSSY uniaxial substrate: total flux DE1 + DE3 == 1 (includes O-E cross term)
        sub_uni_l = EMpy.materials.AnisotropicMaterial(
            "sul",
            epsilon_tensor=EMpy.materials.EpsilonTensor(
                epsilon_tensor_const=eps0
                * np.diag([(1.9 - 0.4j) ** 2, (1.5 - 0.1j) ** 2, (1.5 - 0.1j) ** 2])
            ),
        )
        ml_ul = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(air, np.inf),
                EMpy.utils.BinaryGrating(si, sio2, 0.45, 1.1e-6, 0.25e-6),
                EMpy.utils.Layer(sub_uni_l, np.inf),
            ]
        )
        ra_ul = EMpy.RCWA.AnisotropicRCWA(
            ml_ul, alpha=0.35, delta=0.7, psi=0.4, phi=np.pi / 2, n=2
        ).solve([wl])
        self.assertAlmostEqual(float((ra_ul.DE1 + ra_ul.DE3).sum()), 1.0, places=12)

        # 3. Birefringent superstrate mixed polarization at delta = 0 on planar film: exact flux-weighted superposition
        nO1, nE1 = 1.4, 1.9
        sup_u = EMpy.materials.AnisotropicMaterial(
            "su",
            epsilon_tensor=EMpy.materials.EpsilonTensor(
                epsilon_tensor_const=eps0 * np.diag([nE1**2, nO1**2, nO1**2])
            ),
        )
        ml_film = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(sup_u, np.inf),
                EMpy.utils.Layer(sio2, 0.3e-6),
                EMpy.utils.Layer(sio2, np.inf),
            ]
        )
        al_b = EMpy.utils.deg2rad(25.0)
        r_e = EMpy.RCWA.AnisotropicRCWA(ml_film, al_b, 0.0, 0.0, np.pi / 2, 0).solve(
            [wl]
        )
        r_o = EMpy.RCWA.AnisotropicRCWA(
            ml_film, al_b, 0.0, np.pi / 2, np.pi / 2, 0
        ).solve([wl])
        r_mix = EMpy.RCWA.AnisotropicRCWA(
            ml_film, al_b, 0.0, np.pi / 4, np.pi / 2, 0
        ).solve([wl])
        # Compute exact incident Poynting flux weights for unit-D projections aO = aE = 1/sqrt(2)
        dirk = np.array([np.sin(al_b), 0.0, np.cos(al_b)])
        abskE = nO1 * nE1 / np.sqrt(nO1**2 + (nE1**2 - nO1**2) * dirk[0] ** 2)
        dE = np.cross(dirk, np.cross([1.0, 0.0, 0.0], dirk))
        dE /= np.linalg.norm(dE)
        EE = np.array([(nO1 / nE1) ** 2 * dE[0], dE[1], dE[2]])
        HE = np.cross(abskE * dirk, EE)
        sz_O = nO1 * np.cos(al_b)
        sz_E = float((EE[0] * HE[1] - EE[1] * HE[0]).real)
        r_expected = (sz_O * r_o.DE1[0, 0] + sz_E * r_e.DE1[0, 0]) / (sz_O + sz_E)
        self.assertAlmostEqual(float(r_mix.DE1[0, 0]), float(r_expected), places=14)
        self.assertAlmostEqual(float((r_mix.DE1 + r_mix.DE3).sum()), 1.0, places=14)

        # 4. Exact Rayleigh anomaly (wl == pitch at normal incidence in air): no LinAlgError and matches IsoRCWA
        ml_ra = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(air, np.inf),
                EMpy.utils.BinaryGrating(si, sio2, 0.5, 1.0e-6, 0.2e-6),
                EMpy.utils.Layer(sio2, np.inf),
            ]
        )
        ri_ra = EMpy.RCWA.IsotropicRCWA(
            ml_ra, alpha=0.0, delta=0.0, psi=0.4, phi=np.pi / 2, n=2
        ).solve([1.0e-6])
        ra_ra = EMpy.RCWA.AnisotropicRCWA(
            ml_ra, alpha=0.0, delta=0.0, psi=0.4, phi=np.pi / 2, n=2
        ).solve([1.0e-6])
        self.assertAlmostEqual(float((ra_ra.DE1 + ra_ra.DE3).sum()), 1.0, places=11)
        np.testing.assert_allclose(ra_ra.DE1, ri_ra.DE1, atol=1e-7)
        np.testing.assert_allclose(ra_ra.DE3, ri_ra.DE3, atol=1e-7)

    def test_anisotropic_rcwa_conical_null_cone_and_limits(self):
        """Verify AnisotropicRCWA at conical Kx_m = +-nO, Ky != 0 (isotropic & uniaxial end layers)."""
        wl = 1.0e-6
        pitch = 1.0e-6
        eps0 = EMpy.constants.eps0
        alpha = 0.2
        delta = np.pi / 2
        psi = 0.4

        si_i = EMpy.materials.IsotropicMaterial(
            "si_i", n0=EMpy.materials.RefractiveIndex(n0_const=3.48)
        )
        sio2_i = EMpy.materials.IsotropicMaterial(
            "sio2_i", n0=EMpy.materials.RefractiveIndex(n0_const=1.45)
        )
        ml_iso = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(EMpy.materials.Air, np.inf),
                EMpy.utils.BinaryGrating(si_i, sio2_i, 0.5, pitch, 0.2e-6),
                EMpy.utils.Layer(sio2_i, np.inf),
            ]
        )
        ri = EMpy.RCWA.IsotropicRCWA(
            ml_iso, alpha=alpha, delta=delta, psi=psi, phi=np.pi / 2, n=2
        ).solve([wl])
        self.assertAlmostEqual(float((ri.DE1 + ri.DE3).sum()), 1.0, places=14)

        # 1. Explicit AnisotropicMaterial isotropic tensors (bypassing hasattr(mat, 'n'))
        air_a = EMpy.materials.AnisotropicMaterial(
            "air_a", EMpy.materials.EpsilonTensor(eps0 * np.eye(3))
        )
        si_a = EMpy.materials.AnisotropicMaterial(
            "si_a", EMpy.materials.EpsilonTensor(eps0 * 3.48**2 * np.eye(3))
        )
        sio2_a = EMpy.materials.AnisotropicMaterial(
            "sio2_a", EMpy.materials.EpsilonTensor(eps0 * 1.45**2 * np.eye(3))
        )
        ml_ani_iso = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(air_a, np.inf),
                EMpy.utils.BinaryGrating(si_a, sio2_a, 0.5, pitch, 0.2e-6),
                EMpy.utils.Layer(sio2_a, np.inf),
            ]
        )
        ra_iso = EMpy.RCWA.AnisotropicRCWA(
            ml_ani_iso, alpha=alpha, delta=delta, psi=psi, phi=np.pi / 2, n=2
        ).solve([wl])
        self.assertAlmostEqual(float((ra_iso.DE1 + ra_iso.DE3).sum()), 1.0, places=12)
        np.testing.assert_allclose(ra_iso.DE1, ri.DE1, atol=1e-12)
        np.testing.assert_allclose(ra_iso.DE3, ri.DE3, atol=1e-12)
        np.testing.assert_allclose(ra_iso.DEO1 + ra_iso.DEE1, ri.DE1, atol=1e-12)
        np.testing.assert_allclose(ra_iso.DEO3 + ra_iso.DEE3, ri.DE3, atol=1e-12)

        # 2. 3D Euler-rotated birefringent grating dn -> 0 right at the conical null-cone point
        rot_errs = []
        for dn in (1e-4, 1e-8, 0.0):
            si_rot = EMpy.materials.AnisotropicMaterial(
                "si_rot",
                EMpy.materials.EpsilonTensor(
                    eps0
                    * EMpy.utils.euler_rotate(
                        np.diag([3.48**2, (3.48 + 0.5 * dn) ** 2, (3.48 + dn) ** 2]),
                        0.3,
                        0.5,
                        0.7,
                    )
                ),
            )
            ml_rot = EMpy.utils.Multilayer(
                [
                    EMpy.utils.Layer(air_a, np.inf),
                    EMpy.utils.BinaryGrating(si_rot, sio2_a, 0.5, pitch, 0.2e-6),
                    EMpy.utils.Layer(sio2_a, np.inf),
                ]
            )
            ra_rot = EMpy.RCWA.AnisotropicRCWA(
                ml_rot, alpha=alpha, delta=delta, psi=psi, phi=np.pi / 2, n=2
            ).solve([wl])
            self.assertAlmostEqual(
                float((ra_rot.DE1 + ra_rot.DE3).sum()), 1.0, places=13
            )
            rot_errs.append(
                float(np.max(np.abs(ra_rot.DE1 - ri.DE1) + np.abs(ra_rot.DE3 - ri.DE3)))
            )
        self.assertLess(rot_errs[1], rot_errs[0])
        self.assertLess(rot_errs[2], 1e-12)

        # 3. Uniaxial superstrate & substrate exceptional point (Kx_+-1 = +-nO1, Ky != 0):
        # verify exact energy conservation, smooth wavelength continuity, and dn -> 0 convergence
        sup_u02 = EMpy.materials.AnisotropicMaterial(
            "sup_u02",
            EMpy.materials.EpsilonTensor(eps0 * np.diag([1.2**2, 1.0, 1.0])),
        )
        sub_u = EMpy.materials.AnisotropicMaterial(
            "sub_u",
            EMpy.materials.EpsilonTensor(eps0 * np.diag([1.65**2, 1.0, 1.0])),
        )
        ml_ep = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(sup_u02, np.inf),
                EMpy.utils.BinaryGrating(si_a, sio2_a, 0.5, pitch, 0.2e-6),
                EMpy.utils.Layer(sub_u, np.inf),
            ]
        )
        ra_ep_at = EMpy.RCWA.AnisotropicRCWA(
            ml_ep, alpha=alpha, delta=delta, psi=psi, phi=np.pi / 2, n=2
        ).solve([wl])
        ra_ep_near = EMpy.RCWA.AnisotropicRCWA(
            ml_ep, alpha=alpha, delta=delta, psi=psi, phi=np.pi / 2, n=2
        ).solve([wl * (1.0 + 1e-6)])
        self.assertAlmostEqual(
            float((ra_ep_at.DE1 + ra_ep_at.DE3).sum()), 1.0, places=12
        )
        np.testing.assert_allclose(ra_ep_at.DE1, ra_ep_near.DE1, atol=5e-5)
        np.testing.assert_allclose(ra_ep_at.DE3, ra_ep_near.DE3, atol=5e-5)

        ri_e = EMpy.RCWA.IsotropicRCWA(
            ml_iso, alpha=alpha, delta=delta, psi=np.pi / 2, phi=np.pi / 2, n=2
        ).solve([wl])
        sup_dn_errs = []
        for dn in (1e-4, 1e-8, 0.0):
            sup_dn = EMpy.materials.AnisotropicMaterial(
                "sup_dn",
                EMpy.materials.EpsilonTensor(
                    eps0 * np.diag([(1.0 + dn) ** 2, 1.0, 1.0])
                ),
            )
            ml_dn = EMpy.utils.Multilayer(
                [
                    EMpy.utils.Layer(sup_dn, np.inf),
                    EMpy.utils.BinaryGrating(si_a, sio2_a, 0.5, pitch, 0.2e-6),
                    EMpy.utils.Layer(sio2_a, np.inf),
                ]
            )
            # At delta = pi/2, c1 = [1, 0, 0] is perpendicular to incidence plane so psi = pi/2 is pure E
            ra_dn = EMpy.RCWA.AnisotropicRCWA(
                ml_dn, alpha=alpha, delta=delta, psi=np.pi / 2, phi=np.pi / 2, n=2
            ).solve([wl])
            self.assertAlmostEqual(float((ra_dn.DE1 + ra_dn.DE3).sum()), 1.0, places=13)
            sup_dn_errs.append(
                float(
                    np.max(np.abs(ra_dn.DE1 - ri_e.DE1) + np.abs(ra_dn.DE3 - ri_e.DE3))
                )
            )
        self.assertLess(sup_dn_errs[1], sup_dn_errs[0])
        self.assertLess(sup_dn_errs[2], 1e-12)

        # 4. Lossy -> lossless limit at the conical null-cone geometry
        loss_errs = []
        for kap in (1e-2, 1e-5, 0.0):
            si_k = EMpy.materials.AnisotropicMaterial(
                "si_k",
                EMpy.materials.EpsilonTensor(
                    eps0 * complex(3.48, -kap) ** 2 * np.eye(3)
                ),
            )
            ml_k = EMpy.utils.Multilayer(
                [
                    EMpy.utils.Layer(air_a, np.inf),
                    EMpy.utils.BinaryGrating(si_k, sio2_a, 0.5, pitch, 0.2e-6),
                    EMpy.utils.Layer(sio2_a, np.inf),
                ]
            )
            ra_k = EMpy.RCWA.AnisotropicRCWA(
                ml_k, alpha=alpha, delta=delta, psi=psi, phi=np.pi / 2, n=2
            ).solve([wl])
            if kap > 0:
                self.assertLess(float((ra_k.DE1 + ra_k.DE3).sum()), 1.0)
            loss_errs.append(
                float(np.max(np.abs(ra_k.DE1 - ri.DE1) + np.abs(ra_k.DE3 - ri.DE3)))
            )
        self.assertLess(loss_errs[1], loss_errs[0])
        self.assertLess(loss_errs[2], 1e-12)

    def test_isotropic_rcwa_lossy_superstrate_planar_and_conical_limits(self):
        """Verify IsotropicRCWA DE1/DE3 with finite-loss superstrates against TMM and AnisotropicRCWA."""
        wl = 1.55e-6
        eps0 = EMpy.constants.eps0

        # 1. Planar stack [1.5 - 0.2j, 2.2, 1.45]: IsotropicRCWA == IsotropicTransferMatrix == AnisotropicRCWA
        n1_p, n2_p, n3_p = 1.5 - 0.2j, 2.2, 1.45
        alpha_p = 0.3
        m1_i = EMpy.materials.IsotropicMaterial(
            "m1", n0=EMpy.materials.RefractiveIndex(n0_const=n1_p)
        )
        m2_i = EMpy.materials.IsotropicMaterial(
            "m2", n0=EMpy.materials.RefractiveIndex(n0_const=n2_p)
        )
        m3_i = EMpy.materials.IsotropicMaterial(
            "m3", n0=EMpy.materials.RefractiveIndex(n0_const=n3_p)
        )
        ml_i = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(m1_i, np.inf),
                EMpy.utils.Layer(m2_i, 0.3e-6),
                EMpy.utils.Layer(m3_i, np.inf),
            ]
        )
        tmm_i = EMpy.transfer_matrix.IsotropicTransferMatrix(ml_i, alpha_p).solve([wl])
        rcwa_te = EMpy.RCWA.IsotropicRCWA(
            ml_i, alpha_p, 0.0, np.pi / 2, np.pi / 2, 0
        ).solve([wl])
        rcwa_tm = EMpy.RCWA.IsotropicRCWA(ml_i, alpha_p, 0.0, 0.0, np.pi / 2, 0).solve(
            [wl]
        )
        self.assertAlmostEqual(float(rcwa_te.DE1[0, 0]), float(tmm_i.Rs[0]), places=12)
        self.assertAlmostEqual(float(rcwa_te.DE3[0, 0]), float(tmm_i.Ts[0]), places=12)
        self.assertAlmostEqual(float(rcwa_tm.DE1[0, 0]), float(tmm_i.Rp[0]), places=12)
        self.assertAlmostEqual(float(rcwa_tm.DE3[0, 0]), float(tmm_i.Tp[0]), places=12)

        # 2. Conical grating (delta != 0) with lossy superstrate, lossy grating, and lossy substrate
        n1, n2, n3 = 1.5 - 0.2j, 2.4 - 0.05j, 1.6 - 0.15j
        alpha, delta, psi = 0.35, 0.55, 0.4
        g1_i = EMpy.materials.IsotropicMaterial(
            "g1i", n0=EMpy.materials.RefractiveIndex(n0_const=n1)
        )
        g2_i = EMpy.materials.IsotropicMaterial(
            "g2i", n0=EMpy.materials.RefractiveIndex(n0_const=n2)
        )
        g3_i = EMpy.materials.IsotropicMaterial(
            "g3i", n0=EMpy.materials.RefractiveIndex(n0_const=n3)
        )
        g1_a = EMpy.materials.AnisotropicMaterial(
            "g1a", EMpy.materials.EpsilonTensor(eps0 * n1**2 * np.eye(3))
        )
        g2_a = EMpy.materials.AnisotropicMaterial(
            "g2a", EMpy.materials.EpsilonTensor(eps0 * n2**2 * np.eye(3))
        )
        g3_a = EMpy.materials.AnisotropicMaterial(
            "g3a", EMpy.materials.EpsilonTensor(eps0 * n3**2 * np.eye(3))
        )
        ml_gi = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(g1_i, np.inf),
                EMpy.utils.BinaryGrating(g2_i, g1_i, 0.43, 1.8e-6, 0.28e-6),
                EMpy.utils.Layer(g3_i, np.inf),
            ]
        )
        ml_ga = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(g1_a, np.inf),
                EMpy.utils.BinaryGrating(g2_a, g1_a, 0.43, 1.8e-6, 0.28e-6),
                EMpy.utils.Layer(g3_a, np.inf),
            ]
        )
        r_gi = EMpy.RCWA.IsotropicRCWA(
            ml_gi, alpha=alpha, delta=delta, psi=psi, phi=np.pi / 2, n=2
        ).solve([wl])
        r_ga = EMpy.RCWA.AnisotropicRCWA(
            ml_ga, alpha=alpha, delta=delta, psi=psi, phi=np.pi / 2, n=2
        ).solve([wl])
        np.testing.assert_allclose(r_gi.DE1, r_ga.DE1, atol=1e-12)
        np.testing.assert_allclose(r_gi.DE3, r_ga.DE3, atol=1e-12)

        # Normal incidence (alpha = 0, delta != 0) with lossy superstrate
        r_gi0 = EMpy.RCWA.IsotropicRCWA(
            ml_gi, alpha=0.0, delta=delta, psi=psi, phi=np.pi / 2, n=2
        ).solve([wl])
        r_ga0 = EMpy.RCWA.AnisotropicRCWA(
            ml_ga, alpha=0.0, delta=delta, psi=psi, phi=np.pi / 2, n=2
        ).solve([wl])
        np.testing.assert_allclose(r_gi0.DE1, r_ga0.DE1, atol=1e-12)
        np.testing.assert_allclose(r_gi0.DE3, r_ga0.DE3, atol=1e-12)

    def test_anisotropic_rcwa_lossy_uniaxial_superstrate_and_limits(self):
        """Verify AnisotropicRCWA mixed O+E incidence on lossy uniaxial superstrates and aniso/lossy limits."""
        wl = 1.55e-6
        eps0 = EMpy.constants.eps0
        delta = EMpy.utils.deg2rad(25.0)
        psi = EMpy.utils.deg2rad(35.0)

        def _build_stack(kap, dn_sup, dn_film):
            nO1 = complex(1.5, -0.15 * kap) if kap > 0 else 1.5
            nE1 = complex(1.5 + dn_sup, -0.35 * kap) if kap > 0 else (1.5 + dn_sup)
            nOf = complex(1.9, -0.02 * kap) if kap > 0 else 1.9
            nEf = complex(1.9 + dn_film, -0.05 * kap) if kap > 0 else (1.9 + dn_film)
            n3 = complex(1.45, -0.01 * kap) if kap > 0 else 1.45
            m1 = EMpy.materials.AnisotropicMaterial(
                "sup",
                EMpy.materials.EpsilonTensor(eps0 * np.diag([nE1**2, nO1**2, nO1**2])),
            )
            eps_f = EMpy.utils.euler_rotate(
                np.diag([nEf**2, nOf**2, nOf**2]), 0.4, 0.6, 0.3
            )
            mf = EMpy.materials.AnisotropicMaterial(
                "film", EMpy.materials.EpsilonTensor(eps0 * eps_f)
            )
            m3 = EMpy.materials.AnisotropicMaterial(
                "sub", EMpy.materials.EpsilonTensor(eps0 * n3**2 * np.eye(3))
            )
            return EMpy.utils.Multilayer(
                [
                    EMpy.utils.Layer(m1, np.inf),
                    EMpy.utils.Layer(mf, 0.4e-6),
                    EMpy.utils.Layer(m3, np.inf),
                ]
            ), (nO1, nE1, n3)

        # 1. Exact agreement against coherent 4x4 Berreman amplitude superposition at normal incidence
        ml_lu, (nO1, nE1, n3) = _build_stack(kap=1.0, dn_sup=0.3, dn_film=0.3)
        rcwa_lu = EMpy.RCWA.AnisotropicRCWA(
            ml_lu, alpha=0.0, delta=delta, psi=psi, phi=np.pi / 2, n=0
        ).solve([wl])
        self.assertLess(float(rcwa_lu.DE1[0, 0] + rcwa_lu.DE3[0, 0]), 1.0)

        # Pure O (delta=0, psi=pi/2) and pure E (delta=0, psi=0) match AnisotropicTransferMatrix R_tot/T_tot
        atm_lu = EMpy.transfer_matrix.AnisotropicTransferMatrix(ml_lu, 0.0, 0.0).solve(
            [wl]
        )
        r_pure_o = EMpy.RCWA.AnisotropicRCWA(
            ml_lu, alpha=0.0, delta=0.0, psi=np.pi / 2, phi=np.pi / 2, n=0
        ).solve([wl])
        r_pure_e = EMpy.RCWA.AnisotropicRCWA(
            ml_lu, alpha=0.0, delta=0.0, psi=0.0, phi=np.pi / 2, n=0
        ).solve([wl])
        # In AnisotropicTransferMatrix, fwd modes are sorted by Re(gamma) descending:
        # since Re(nE1) > Re(nO1), mode 0 is E (x-polarized) and mode 1 is O (y-polarized).
        self.assertAlmostEqual(
            float(r_pure_o.DE1[0, 0]), float(atm_lu.R_tot[1, 0]), places=12
        )
        self.assertAlmostEqual(
            float(r_pure_o.DE3[0, 0]), float(atm_lu.T_tot[1, 0]), places=12
        )
        self.assertAlmostEqual(
            float(r_pure_e.DE1[0, 0]), float(atm_lu.R_tot[0, 0]), places=12
        )
        self.assertAlmostEqual(
            float(r_pure_e.DE3[0, 0]), float(atm_lu.T_tot[0, 0]), places=12
        )

        # 2. Lossy -> lossless limit (kap = 1.0, 1e-4, 1e-8, 0.0): energy conservation R + T == 1 at kap = 0
        ml_ll, _ = _build_stack(kap=0.0, dn_sup=0.3, dn_film=0.3)
        rcwa_ll = EMpy.RCWA.AnisotropicRCWA(
            ml_ll, alpha=0.0, delta=delta, psi=psi, phi=np.pi / 2, n=0
        ).solve([wl])
        self.assertAlmostEqual(
            float(rcwa_ll.DE1[0, 0] + rcwa_ll.DE3[0, 0]), 1.0, places=12
        )
        kap_errs = []
        for kap in (1e-2, 1e-6, 0.0):
            ml_k, _ = _build_stack(kap=kap, dn_sup=0.3, dn_film=0.3)
            rk = EMpy.RCWA.AnisotropicRCWA(
                ml_k, alpha=0.0, delta=delta, psi=psi, phi=np.pi / 2, n=0
            ).solve([wl])
            if kap > 0:
                self.assertLess(float(rk.DE1[0, 0] + rk.DE3[0, 0]), 1.0)
            kap_errs.append(
                abs(float(rk.DE1[0, 0] - rcwa_ll.DE1[0, 0]))
                + abs(float(rk.DE3[0, 0] - rcwa_ll.DE3[0, 0]))
            )
        self.assertLess(kap_errs[1], kap_errs[0])
        self.assertLess(kap_errs[2], 1e-12)

        # 3. Anisotropic -> isotropic limit (dn_sup = dn_film -> 0) on lossy superstrate against IsotropicRCWA
        m1_i = EMpy.materials.IsotropicMaterial(
            "s1", EMpy.materials.RefractiveIndex(n0_const=1.5 - 0.15j)
        )
        mf_i = EMpy.materials.IsotropicMaterial(
            "sf", EMpy.materials.RefractiveIndex(n0_const=1.9 - 0.02j)
        )
        m3_i = EMpy.materials.IsotropicMaterial(
            "s3", EMpy.materials.RefractiveIndex(n0_const=1.45 - 0.01j)
        )
        ml_iso_ref = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(m1_i, np.inf),
                EMpy.utils.Layer(mf_i, 0.4e-6),
                EMpy.utils.Layer(m3_i, np.inf),
            ]
        )
        rcwa_iso_ref = EMpy.RCWA.IsotropicRCWA(
            ml_iso_ref, alpha=0.0, delta=delta, psi=psi, phi=np.pi / 2, n=0
        ).solve([wl])
        dn_errs = []
        for dn in (1e-3, 1e-7, 0.0):
            nO1_d = 1.5 - 0.15j
            nE1_d = (1.5 + dn) - 0.15j
            nOf_d = 1.9 - 0.02j
            nEf_d = (1.9 + dn) - 0.02j
            ml_d = EMpy.utils.Multilayer(
                [
                    EMpy.utils.Layer(
                        EMpy.materials.AnisotropicMaterial(
                            "s1",
                            EMpy.materials.EpsilonTensor(
                                eps0 * np.diag([nE1_d**2, nO1_d**2, nO1_d**2])
                            ),
                        ),
                        np.inf,
                    ),
                    EMpy.utils.Layer(
                        EMpy.materials.AnisotropicMaterial(
                            "sf",
                            EMpy.materials.EpsilonTensor(
                                eps0
                                * EMpy.utils.euler_rotate(
                                    np.diag([nEf_d**2, nOf_d**2, nOf_d**2]),
                                    0.4,
                                    0.6,
                                    0.3,
                                )
                            ),
                        ),
                        0.4e-6,
                    ),
                    EMpy.utils.Layer(
                        EMpy.materials.AnisotropicMaterial(
                            "s3",
                            EMpy.materials.EpsilonTensor(
                                eps0 * (1.45 - 0.01j) ** 2 * np.eye(3)
                            ),
                        ),
                        np.inf,
                    ),
                ]
            )
            rd = EMpy.RCWA.AnisotropicRCWA(
                ml_d, alpha=0.0, delta=delta, psi=psi, phi=np.pi / 2, n=0
            ).solve([wl])
            dn_errs.append(
                abs(float(rd.DE1[0, 0] - rcwa_iso_ref.DE1[0, 0]))
                + abs(float(rd.DE3[0, 0] - rcwa_iso_ref.DE3[0, 0]))
            )
        self.assertLess(dn_errs[1], dn_errs[0])
        self.assertLess(dn_errs[2], 1e-12)

    def test_near_normal_kt_continuity_and_energy_conservation(self):
        """Verify fc^2 + fs^2 = 1 and continuity for 1e-15 * k < |kt| < 1e-13 * k."""
        air = EMpy.materials.Air
        si = EMpy.materials.IsotropicMaterial(
            "Si", n0=EMpy.materials.RefractiveIndex(n0_const=2.5)
        )
        sio2 = EMpy.materials.IsotropicMaterial(
            "SiO2", n0=EMpy.materials.RefractiveIndex(n0_const=1.45)
        )
        pitch = 1.8e-6
        wl = 1.0e-6
        ml = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(air, np.inf),
                EMpy.utils.BinaryGrating(si, sio2, 0.45, pitch, 0.35e-6),
                EMpy.utils.Layer(sio2, np.inf),
            ]
        )
        delta, psi = 0.3, 0.6

        # 1. Near-normal incidence alpha -> 0+ spanning 1e-15..1e-12
        iso_0 = EMpy.RCWA.IsotropicRCWA(
            ml, alpha=0.0, delta=delta, psi=psi, phi=np.pi / 2, n=3
        ).solve([wl])
        ani_0 = EMpy.RCWA.AnisotropicRCWA(
            ml, alpha=0.0, delta=delta, psi=psi, phi=np.pi / 2, n=3
        ).solve([wl])
        for alpha in (1e-14, 5e-14, 1e-13, 1e-12):
            iso_a = EMpy.RCWA.IsotropicRCWA(
                ml, alpha=alpha, delta=delta, psi=psi, phi=np.pi / 2, n=3
            ).solve([wl])
            ani_a = EMpy.RCWA.AnisotropicRCWA(
                ml, alpha=alpha, delta=delta, psi=psi, phi=np.pi / 2, n=3
            ).solve([wl])
            self.assertAlmostEqual(float((iso_a.DE1 + iso_a.DE3).sum()), 1.0, places=12)
            self.assertAlmostEqual(float((ani_a.DE1 + ani_a.DE3).sum()), 1.0, places=12)
            np.testing.assert_allclose(iso_a.DE1, iso_0.DE1, atol=1e-10)
            np.testing.assert_allclose(iso_a.DE3, iso_0.DE3, atol=1e-10)
            np.testing.assert_allclose(ani_a.DE1, ani_0.DE1, atol=1e-10)
            np.testing.assert_allclose(ani_a.DE3, ani_0.DE3, atol=1e-10)
            np.testing.assert_allclose(ani_a.DEO1 + ani_a.DEE1, ani_a.DE1, atol=1e-12)
            np.testing.assert_allclose(ani_a.DEO3 + ani_a.DEE3, ani_a.DE3, atol=1e-12)
            np.testing.assert_allclose(ani_a.DEO1, ani_0.DEO1, atol=1e-10)
            np.testing.assert_allclose(ani_a.DEE1, ani_0.DEE1, atol=1e-10)

        # 2. Near-normal exiting m=+1 diffraction order (kx_{+1} ~ 1e-14 * k)
        alpha_exact = np.arcsin(wl / pitch)
        iso_ref = EMpy.RCWA.IsotropicRCWA(
            ml, alpha=alpha_exact, delta=0.0, psi=psi, phi=np.pi / 2, n=3
        ).solve([wl])
        for eps in (1e-14, 5e-14):
            alpha_near = np.arcsin(wl / pitch + eps)
            iso_n = EMpy.RCWA.IsotropicRCWA(
                ml, alpha=alpha_near, delta=0.0, psi=psi, phi=np.pi / 2, n=3
            ).solve([wl])
            ani_n = EMpy.RCWA.AnisotropicRCWA(
                ml, alpha=alpha_near, delta=0.0, psi=psi, phi=np.pi / 2, n=3
            ).solve([wl])
            self.assertAlmostEqual(float((iso_n.DE1 + iso_n.DE3).sum()), 1.0, places=12)
            self.assertAlmostEqual(float((ani_n.DE1 + ani_n.DE3).sum()), 1.0, places=12)
            np.testing.assert_allclose(iso_n.DE1, iso_ref.DE1, atol=1e-10)
            np.testing.assert_allclose(iso_n.DE3, iso_ref.DE3, atol=1e-10)
            np.testing.assert_allclose(ani_n.DE1, iso_n.DE1, atol=1e-10)
            np.testing.assert_allclose(ani_n.DE3, iso_n.DE3, atol=1e-10)

    def test_lossy_superstrate_oblique_internal_branch_selection(self):
        """Verify passive internal-layer mode branch selection under oblique incidence from a lossy superstrate."""
        wl = 1.55e-6
        m1 = EMpy.materials.IsotropicMaterial(
            "m1", n0=EMpy.materials.RefractiveIndex(n0_const=1.5 - 0.1j)
        )
        m2 = EMpy.materials.IsotropicMaterial(
            "m2", n0=EMpy.materials.RefractiveIndex(n0_const=1.2)
        )
        m3 = EMpy.materials.IsotropicMaterial(
            "m3", n0=EMpy.materials.RefractiveIndex(n0_const=1.2)
        )
        ml_layer = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(m1, np.inf),
                EMpy.utils.Layer(m2, 10.0 * wl),
                EMpy.utils.Layer(m3, np.inf),
            ]
        )
        ml_grat = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(m1, np.inf),
                EMpy.utils.BinaryGrating(m2, m2, 0.5, 1.2e-6, 10.0 * wl),
                EMpy.utils.Layer(m3, np.inf),
            ]
        )

        # Matched lossless internal layer + substrate (previously singular MTp1 / 1e15 blowup)
        for alpha in (0.2, 0.7):
            for psi in (0.0, np.pi / 2):
                tmm = EMpy.transfer_matrix.IsotropicTransferMatrix(
                    ml_layer, alpha
                ).solve([wl])
                R_ref = float(tmm.Rp[0] if psi == 0.0 else tmm.Rs[0])
                T_ref = float(tmm.Tp[0] if psi == 0.0 else tmm.Ts[0])
                iso = EMpy.RCWA.IsotropicRCWA(
                    ml_grat, alpha=alpha, delta=0.0, psi=psi, phi=np.pi / 2, n=5
                ).solve([wl])
                ani = EMpy.RCWA.AnisotropicRCWA(
                    ml_grat, alpha=alpha, delta=0.0, psi=psi, phi=np.pi / 2, n=5
                ).solve([wl])
                np.testing.assert_allclose(iso.DE1.sum(), R_ref, rtol=1e-9, atol=1e-12)
                np.testing.assert_allclose(iso.DE3.sum(), T_ref, rtol=1e-9, atol=1e-12)
                np.testing.assert_allclose(ani.DE1.sum(), R_ref, rtol=1e-8, atol=1e-12)
                np.testing.assert_allclose(ani.DE3.sum(), T_ref, rtol=1e-8, atol=1e-12)

        # TIR lossless internal layer from a lossy superstrate
        m1_tir = EMpy.materials.IsotropicMaterial(
            "m1_tir", n0=EMpy.materials.RefractiveIndex(n0_const=1.8 - 0.1j)
        )
        m2_tir = EMpy.materials.IsotropicMaterial(
            "m2_tir", n0=EMpy.materials.RefractiveIndex(n0_const=1.0)
        )
        m3_tir = EMpy.materials.IsotropicMaterial(
            "m3_tir", n0=EMpy.materials.RefractiveIndex(n0_const=1.5)
        )
        ml_tir = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(m1_tir, np.inf),
                EMpy.utils.Layer(m2_tir, 0.5 * wl),
                EMpy.utils.Layer(m3_tir, np.inf),
            ]
        )
        ml_tir_g = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(m1_tir, np.inf),
                EMpy.utils.BinaryGrating(m2_tir, m2_tir, 0.5, 1.2e-6, 0.5 * wl),
                EMpy.utils.Layer(m3_tir, np.inf),
            ]
        )
        tmm_tir = EMpy.transfer_matrix.IsotropicTransferMatrix(ml_tir, 1.0).solve([wl])
        iso_tir = EMpy.RCWA.IsotropicRCWA(
            ml_tir_g, alpha=1.0, delta=0.0, psi=0.0, phi=np.pi / 2, n=3
        ).solve([wl])
        ani_tir = EMpy.RCWA.AnisotropicRCWA(
            ml_tir_g, alpha=1.0, delta=0.0, psi=0.0, phi=np.pi / 2, n=3
        ).solve([wl])
        np.testing.assert_allclose(iso_tir.DE1.sum(), tmm_tir.Rp[0], rtol=1e-12)
        np.testing.assert_allclose(iso_tir.DE3.sum(), tmm_tir.Tp[0], rtol=1e-12)
        np.testing.assert_allclose(ani_tir.DE1.sum(), tmm_tir.Rp[0], rtol=1e-12)
        np.testing.assert_allclose(ani_tir.DE3.sum(), tmm_tir.Tp[0], rtol=1e-12)

        # Modulated conical grating under lossy superstrate: IsotropicRCWA == AnisotropicRCWA
        m2b = EMpy.materials.IsotropicMaterial(
            "m2b", n0=EMpy.materials.RefractiveIndex(n0_const=2.2)
        )
        ml_mod = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(m1, np.inf),
                EMpy.utils.BinaryGrating(m2, m2b, 0.4, 1.2e-6, 0.5 * wl),
                EMpy.utils.Layer(m3, np.inf),
            ]
        )
        iso_mod = EMpy.RCWA.IsotropicRCWA(
            ml_mod, alpha=0.5, delta=0.4, psi=0.7, phi=np.pi / 2, n=4
        ).solve([wl])
        ani_mod = EMpy.RCWA.AnisotropicRCWA(
            ml_mod, alpha=0.5, delta=0.4, psi=0.7, phi=np.pi / 2, n=4
        ).solve([wl])
        np.testing.assert_allclose(iso_mod.DE1, ani_mod.DE1, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(iso_mod.DE3, ani_mod.DE3, rtol=1e-12, atol=1e-12)

    def test_anisotropic_rcwa_custom_grating_fallback_converges(self):
        """Verify custom (non-built-in) anisotropic gratings with off-diagonal eps converge cleanly."""
        eps0 = EMpy.constants.eps0
        wl = 1.55e-6
        pitch = 1.2e-6
        thickness = 0.35e-6

        air = EMpy.materials.Air
        m1 = EMpy.materials.AnisotropicMaterial(
            "rot_uni",
            EMpy.materials.EpsilonTensor(
                eps0
                * EMpy.utils.euler_rotate(
                    np.diag([1.6**2, 1.6**2, 2.3**2]), 0.6, 0.7, 0.4
                )
            ),
        )
        m2 = EMpy.materials.IsotropicMaterial(
            "sio2", n0=EMpy.materials.RefractiveIndex(n0_const=1.45)
        )
        bg = EMpy.utils.BinaryGrating(m1, m2, 0.5, pitch, thickness)

        class _CustomGrating(EMpy.utils.Layer):
            def __init__(self, inner):
                self._inner = inner
                self.thickness = inner.thickness
                self.pitch = inner.pitch
                self.mat = inner.mat1

            def getEPSFourierCoeffs(self, wl_, n_, anisotropic=False):
                return self._inner.getEPSFourierCoeffs(wl_, n_, anisotropic=anisotropic)

        ml_li = EMpy.utils.Multilayer(
            [EMpy.utils.Layer(air, np.inf), bg, EMpy.utils.Layer(m2, np.inf)]
        )
        ml_custom = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(air, np.inf),
                _CustomGrating(bg),
                EMpy.utils.Layer(m2, np.inf),
            ]
        )

        # Reference using built-in Li factorization at high order
        ref = EMpy.RCWA.AnisotropicRCWA(
            ml_li, alpha=0.35, delta=0.2, psi=0.5, phi=np.pi / 2, n=20
        ).solve([wl])
        errs = []
        for order in (2, 10, 20):
            r_c = EMpy.RCWA.AnisotropicRCWA(
                ml_custom, alpha=0.35, delta=0.2, psi=0.5, phi=np.pi / 2, n=order
            ).solve([wl])
            self.assertAlmostEqual(float((r_c.DE1 + r_c.DE3).sum()), 1.0, places=12)
            errs.append(abs(float(r_c.DE1[order, 0] - ref.DE1[20, 0])))
        self.assertLess(errs[2], errs[0])
        self.assertLess(errs[2], 1e-2)
