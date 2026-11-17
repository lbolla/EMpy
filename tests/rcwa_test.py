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

        self.assertAlmostEqual(rcwa.DE1.sum(), tmm.Rp[0], places=5)
        self.assertAlmostEqual(rcwa.DE3.sum(), tmm.Tp[0], places=5)
        self.assertAlmostEqual(rcwa.DE1.sum() + rcwa.DE3.sum(), 1.0, places=5)

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

        # 1. Weak-birefringence 3D-Euler-rotated grating + uniaxial superstrate -> IsotropicRCWA (dn = 1e-4, 1e-8, 0)
        dn_errs = []
        for dn in (1e-4, 1e-8, 0.0):
            sup_a = EMpy.materials.AnisotropicMaterial(
                "sup",
                epsilon_tensor=EMpy.materials.EpsilonTensor(
                    epsilon_tensor_const=eps0
                    * np.diag([(1.45 + dn) ** 2, 1.45**2, 1.45**2])
                ),
            )
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
                    EMpy.utils.Layer(sup_a, np.inf),
                    EMpy.utils.BinaryGrating(ridge_a, sup_a, 0.43, 1.1e-6, 0.25e-6),
                    EMpy.utils.Layer(sub_a, np.inf),
                ]
            )
            ra = EMpy.RCWA.AnisotropicRCWA(
                ml_a, alpha=alpha, delta=delta, psi=psi, phi=np.pi / 2, n=2
            ).solve([wl])
            self.assertAlmostEqual(
                float((ra.DEO1 + ra.DEE1 + ra.DEO3 + ra.DEE3).sum()), 1.0, places=12
            )
            err = float(
                np.max(
                    np.abs((ra.DEO1 + ra.DEE1) - r_iso.DE1)
                    + np.abs((ra.DEO3 + ra.DEE3) - r_iso.DE3)
                )
            )
            dn_errs.append(err)
        self.assertLess(dn_errs[1], dn_errs[0])
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
        self.assertLess(dn_errs[2], 1e-14)

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
        np.testing.assert_allclose(r_adg_a.DEO1 + r_adg_a.DEE1, r_sdg_i.DE1, atol=1e-14)
        np.testing.assert_allclose(r_adg_a.DEO3 + r_adg_a.DEE3, r_sdg_i.DE3, atol=1e-14)

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
        self.assertAlmostEqual(float((rg_ll.DE1 + rg_ll.DE3).sum()), 1.0, places=13)
        np.testing.assert_allclose(rg_c0.DE1, rg_ll.DE1, atol=1e-14)
        np.testing.assert_allclose(rg_c0.DE3, rg_ll.DE3, atol=1e-14)

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
        np.testing.assert_allclose(r0.DE1, r0_rot.DE1, atol=1e-14)
        np.testing.assert_allclose(r0.DE3, r0_rot.DE3, atol=1e-14)

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
        self.assertLess(dn_errs[2], 1e-14)

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
        self.assertLess(sup_errs[1], sup_errs[0])
        self.assertLess(sup_errs[2], 1e-14)
