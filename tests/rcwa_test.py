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
