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
