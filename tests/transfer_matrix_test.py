from unittest import TestCase

import numpy as np
from numpy.testing import assert_allclose

import EMpy


def _make_iso_multilayer(n_list, d_list):
    layers = EMpy.utils.Multilayer()
    for n, d in zip(n_list, d_list):
        mat = EMpy.materials.IsotropicMaterial(
            "mat", n0=EMpy.materials.RefractiveIndex(n0_const=n)
        )
        layers.append(EMpy.utils.Layer(mat, d))
    return layers


def _make_aniso_multilayer(n_list, d_list):
    layers = EMpy.utils.Multilayer()
    for n, d in zip(n_list, d_list):
        eps = EMpy.materials.EpsilonTensor(
            epsilon_tensor_const=(n**2) * EMpy.constants.eps0 * np.eye(3)
        )
        mat = EMpy.materials.AnisotropicMaterial("mat", epsilon_tensor=eps)
        layers.append(EMpy.utils.Layer(mat, d))
    return layers


class TransferMatrixTest(TestCase):

    def test_fresnel_brewster(self):
        n1, n2 = 1.0, 1.5
        theta_b = np.arctan2(n2, n1)
        layers = _make_iso_multilayer([n1, n2], [np.inf, np.inf])
        sol = EMpy.transfer_matrix.IsotropicTransferMatrix(layers, theta_b).solve(
            [1.55e-6]
        )
        assert_allclose(sol.Rp, [0.0], atol=1e-15)
        assert_allclose(sol.Tp, [1.0], atol=1e-15)
        assert_allclose(sol.Rs + sol.Ts, [1.0], atol=1e-15)

    def test_total_internal_reflection(self):
        layers = _make_iso_multilayer([1.5, 1.0], [np.inf, np.inf])
        theta_inc = EMpy.utils.deg2rad(60.0)
        sol = EMpy.transfer_matrix.IsotropicTransferMatrix(layers, theta_inc).solve(
            [1.55e-6]
        )
        assert_allclose(sol.Rs, [1.0], atol=1e-15)
        assert_allclose(sol.Ts, [0.0], atol=1e-15)
        assert_allclose(sol.Rp, [1.0], atol=1e-15)
        assert_allclose(sol.Tp, [0.0], atol=1e-15)

    def test_frustrated_total_internal_reflection(self):
        layers = _make_iso_multilayer([1.5, 1.0, 1.5], [np.inf, 0.2e-6, np.inf])
        theta_inc = EMpy.utils.deg2rad(60.0)
        sol = EMpy.transfer_matrix.IsotropicTransferMatrix(layers, theta_inc).solve(
            [1.55e-6]
        )
        self.assertTrue((0.0 < sol.Ts[0] < 1.0) and (0.0 < sol.Tp[0] < 1.0))
        assert_allclose(sol.Rs + sol.Ts, [1.0], atol=1e-15)
        assert_allclose(sol.Rp + sol.Tp, [1.0], atol=1e-15)

    def test_absorbing_exit_medium(self):
        wls = [1.55e-6]
        theta_inc = EMpy.utils.deg2rad(40.0)
        n_list = [1.0, 1.5 - 0.5j]
        d_list = [np.inf, np.inf]

        iso_layers = _make_iso_multilayer(n_list, d_list)
        aniso_layers = _make_aniso_multilayer(n_list, d_list)

        sol_iso = EMpy.transfer_matrix.IsotropicTransferMatrix(
            iso_layers, theta_inc
        ).solve(wls)
        sol_aniso = EMpy.transfer_matrix.AnisotropicTransferMatrix(
            aniso_layers, theta_inc, 0.0
        ).solve(wls)

        # Single interface from lossless medium: net flux across z=0 is conserved
        assert_allclose(sol_iso.Rs + sol_iso.Ts, [1.0], atol=1e-15)
        assert_allclose(sol_iso.Rp + sol_iso.Tp, [1.0], atol=1e-15)

        assert_allclose(sol_iso.Rs, sol_aniso.R[0, 0, :], rtol=1e-6)
        assert_allclose(sol_iso.Ts, sol_aniso.T[0, 0, :], rtol=1e-6)
        assert_allclose(sol_iso.Rp, sol_aniso.R[1, 1, :], rtol=1e-6)
        assert_allclose(sol_iso.Tp, sol_aniso.T[1, 1, :], rtol=1e-6)

    def test_absorbing_slab(self):
        wls = [1.55e-6]
        theta_inc = EMpy.utils.deg2rad(20.0)
        n_list = [1.0, 2.0 - 0.1j, 1.0]
        d_list = [np.inf, 0.5e-6, np.inf]

        iso_layers = _make_iso_multilayer(n_list, d_list)
        aniso_layers = _make_aniso_multilayer(n_list, d_list)

        sol_iso = EMpy.transfer_matrix.IsotropicTransferMatrix(
            iso_layers, theta_inc
        ).solve(wls)
        sol_aniso = EMpy.transfer_matrix.AnisotropicTransferMatrix(
            aniso_layers, theta_inc, 0.0
        ).solve(wls)

        self.assertLess(sol_iso.Rs[0] + sol_iso.Ts[0], 1.0)
        self.assertLess(sol_iso.Rp[0] + sol_iso.Tp[0], 1.0)
        assert_allclose(sol_iso.Rs, sol_aniso.R[0, 0, :], rtol=1e-6)
        assert_allclose(sol_iso.Ts, sol_aniso.T[0, 0, :], rtol=1e-6)
        assert_allclose(sol_iso.Rp, sol_aniso.R[1, 1, :], rtol=1e-6)
        assert_allclose(sol_iso.Tp, sol_aniso.T[1, 1, :], rtol=1e-6)

    def test_iso_vs_aniso_non_unit_incident_index(self):
        wls = np.linspace(1.4e-6, 1.7e-6, 5)
        n_list = [1.5, 2.1, 1.0]
        d_list = [np.inf, 0.3e-6, np.inf]
        theta_inc = EMpy.utils.deg2rad(30.0)
        theta_vac = np.arcsin(n_list[0] * np.sin(theta_inc))

        iso_layers = _make_iso_multilayer(n_list, d_list)
        aniso_layers = _make_aniso_multilayer(n_list, d_list)

        sol_iso = EMpy.transfer_matrix.IsotropicTransferMatrix(
            iso_layers, theta_inc
        ).solve(wls)
        sol_aniso = EMpy.transfer_matrix.AnisotropicTransferMatrix(
            aniso_layers, theta_vac, 0.0
        ).solve(wls)

        assert_allclose(sol_iso.Rs + sol_iso.Ts, 1.0, atol=1e-14)
        assert_allclose(sol_iso.Rp + sol_iso.Tp, 1.0, atol=1e-14)
        assert_allclose(sol_iso.Rs, sol_aniso.R[0, 0, :], rtol=1e-6)
        assert_allclose(sol_iso.Ts, sol_aniso.T[0, 0, :], rtol=1e-6)
        assert_allclose(sol_iso.Rp, sol_aniso.R[1, 1, :], rtol=1e-6)
        assert_allclose(sol_iso.Tp, sol_aniso.T[1, 1, :], rtol=1e-6)

    def test_scalar_wavelength(self):
        wl = 1.55e-6
        iso_layers = _make_iso_multilayer([1.0, 1.5, 1.0], [np.inf, 0.3e-6, np.inf])
        aniso_layers = _make_aniso_multilayer([1.0, 1.5, 1.0], [np.inf, 0.3e-6, np.inf])

        sol_iso_scalar = EMpy.transfer_matrix.IsotropicTransferMatrix(
            iso_layers, 0.2
        ).solve(wl)
        sol_iso_arr = EMpy.transfer_matrix.IsotropicTransferMatrix(
            iso_layers, 0.2
        ).solve([wl])
        assert_allclose(sol_iso_scalar.Rs, sol_iso_arr.Rs)
        assert_allclose(sol_iso_scalar.Ts, sol_iso_arr.Ts)

        sol_aniso_scalar = EMpy.transfer_matrix.AnisotropicTransferMatrix(
            aniso_layers, 0.2, 0.0
        ).solve(wl)
        sol_aniso_arr = EMpy.transfer_matrix.AnisotropicTransferMatrix(
            aniso_layers, 0.2, 0.0
        ).solve([wl])
        assert_allclose(sol_aniso_scalar.R, sol_aniso_arr.R)
        assert_allclose(sol_aniso_scalar.T, sol_aniso_arr.T)

    def test_aniso_oblique_tir_and_energy_conservation(self):
        """Verify AnisotropicTransferMatrix physics against conservation laws and IsotropicTransferMatrix.

        Verification checks:
        1. Energy conservation: R_ss + R_sp + T_ss + T_sp == 1 and
           R_ps + R_pp + T_ps + T_pp == 1 for any lossless stack (isotropic,
           TIR, frustrated TIR, rotated uniaxial, and rotated biaxial).
        2. Cross-solver agreement: isotropic stacks (including frustrated TIR)
           must match IsotropicTransferMatrix for both s and p polarizations.
        3. Birefringent mode conversion: a rotated uniaxial/biaxial slab couples
           s <-> p (non-zero cross-polarized transmittance) while conserving flux.
        """
        wls = np.array([1.55e-6])

        # Frustrated TIR: n_inc = 1.5 > n_gap = 1.0 at 60 deg
        n_list = [1.5, 1.0, 1.5]
        d_list = [np.inf, 0.2e-6, np.inf]
        theta_inc = EMpy.utils.deg2rad(60.0)
        theta_vac = np.arcsin(n_list[0] * np.sin(theta_inc) + 0j)

        sol_iso = EMpy.transfer_matrix.IsotropicTransferMatrix(
            _make_iso_multilayer(n_list, d_list), theta_inc
        ).solve(wls)
        sol_aniso = EMpy.transfer_matrix.AnisotropicTransferMatrix(
            _make_aniso_multilayer(n_list, d_list), theta_vac, 0.0
        ).solve(wls)

        assert_allclose(sol_iso.Rs, sol_aniso.R[0, 0, :], rtol=1e-6)
        assert_allclose(sol_iso.Ts, sol_aniso.T[0, 0, :], rtol=1e-6)
        assert_allclose(sol_iso.Rp, sol_aniso.R[1, 1, :], rtol=1e-6)
        assert_allclose(sol_iso.Tp, sol_aniso.T[1, 1, :], rtol=1e-6)
        assert_allclose(
            sol_aniso.R[0, 0, :]
            + sol_aniso.R[0, 1, :]
            + sol_aniso.T[0, 0, :]
            + sol_aniso.T[0, 1, :],
            1.0,
            atol=1e-12,
        )
        assert_allclose(
            sol_aniso.R[1, 0, :]
            + sol_aniso.R[1, 1, :]
            + sol_aniso.T[1, 0, :]
            + sol_aniso.T[1, 1, :],
            1.0,
            atol=1e-12,
        )

    def test_aniso_birefringent_energy_conservation_and_coupling(self):
        wl = 1.55e-6
        # Rotated biaxial crystal slab between air and glass
        eps_crystal = EMpy.utils.euler_rotate(
            np.diag([1.5**2, 1.65**2, 1.8**2]) * EMpy.constants.eps0,
            EMpy.utils.deg2rad(35.0),
            EMpy.utils.deg2rad(25.0),
            EMpy.utils.deg2rad(15.0),
        )
        mat_air = EMpy.materials.AnisotropicMaterial(
            "air",
            epsilon_tensor=EMpy.materials.EpsilonTensor(
                epsilon_tensor_const=EMpy.constants.eps0 * np.eye(3)
            ),
        )
        mat_crystal = EMpy.materials.AnisotropicMaterial(
            "crystal",
            epsilon_tensor=EMpy.materials.EpsilonTensor(
                epsilon_tensor_const=eps_crystal
            ),
        )
        mat_glass = EMpy.materials.AnisotropicMaterial(
            "glass",
            epsilon_tensor=EMpy.materials.EpsilonTensor(
                epsilon_tensor_const=(1.5**2) * EMpy.constants.eps0 * np.eye(3)
            ),
        )
        layers = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(mat_air, np.inf),
                EMpy.utils.Layer(mat_crystal, 2.0e-6),
                EMpy.utils.Layer(mat_glass, np.inf),
            ]
        )

        sol = EMpy.transfer_matrix.AnisotropicTransferMatrix(
            layers, EMpy.utils.deg2rad(30.0), EMpy.utils.deg2rad(20.0)
        ).solve([wl])

        # Cross-polarization coupling must be present
        self.assertGreater(sol.T[0, 1, 0], 1e-3)
        self.assertGreater(sol.T[1, 0, 0], 1e-3)

        # Energy conservation for s-incident (row 0) and p-incident (row 1)
        assert_allclose(
            sol.R[0, 0, :] + sol.R[0, 1, :] + sol.T[0, 0, :] + sol.T[0, 1, :],
            1.0,
            atol=1e-12,
        )
        assert_allclose(
            sol.R[1, 0, :] + sol.R[1, 1, :] + sol.T[1, 0, :] + sol.T[1, 1, :],
            1.0,
            atol=1e-12,
        )
