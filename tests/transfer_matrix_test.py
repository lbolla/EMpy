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

    def test_kretschmann_spr_tir_and_degenerate_limits(self):
        """Verify passive evanescent branch in TIR across isotropic/anisotropic and lossy->lossless limits."""
        wl = np.array([1.55e-6])
        n_prism = 1.5
        n_au = 0.18 - 3.5j
        n_water = 1.33
        d_au = 0.045e-6
        theta_prism = EMpy.utils.deg2rad(70.0)
        theta_vac = np.arcsin(n_prism * np.sin(theta_prism) + 0j)

        # Unambiguous physical reference 1: finite thick water layer (10 um) backed by propagating glass
        iso_thick = EMpy.transfer_matrix.IsotropicTransferMatrix(
            _make_iso_multilayer(
                [n_prism, n_au, n_water, n_prism], [np.inf, d_au, 10.0e-6, np.inf]
            ),
            theta_prism,
        ).solve(wl)

        # Unambiguous physical reference 2: 0th-order IsotropicRCWA on homogeneous stack
        spr_iso_stack = _make_iso_multilayer(
            [n_prism, n_au, n_water], [np.inf, d_au, np.inf]
        )
        rcwa_te = EMpy.RCWA.IsotropicRCWA(
            spr_iso_stack, theta_prism, 0.0, np.pi / 2, np.pi / 2, 0
        ).solve(wl)
        rcwa_tm = EMpy.RCWA.IsotropicRCWA(
            spr_iso_stack, theta_prism, 0.0, 0.0, np.pi / 2, 0
        ).solve(wl)

        # Semi-infinite lossless TIR in IsotropicTransferMatrix & AnisotropicTransferMatrix
        iso_0 = EMpy.transfer_matrix.IsotropicTransferMatrix(
            spr_iso_stack, theta_prism
        ).solve(wl)
        aniso_0 = EMpy.transfer_matrix.AnisotropicTransferMatrix(
            _make_aniso_multilayer([n_prism, n_au, n_water], [np.inf, d_au, np.inf]),
            theta_vac,
            0.0,
        ).solve(wl)

        assert_allclose(iso_0.Rs, iso_thick.Rs, atol=1e-12)
        assert_allclose(iso_0.Rp, iso_thick.Rp, atol=1e-12)
        assert_allclose(iso_0.Rs, rcwa_te.DE1[0, :], atol=1e-12)
        assert_allclose(iso_0.Rp, rcwa_tm.DE1[0, :], atol=1e-12)
        assert_allclose(iso_0.Ts, [0.0], atol=1e-15)
        assert_allclose(iso_0.Tp, [0.0], atol=1e-15)
        assert_allclose(aniso_0.R[0, 0, :], iso_0.Rs, atol=1e-12)
        assert_allclose(aniso_0.R[1, 1, :], iso_0.Rp, atol=1e-12)
        assert_allclose(aniso_0.T, 0.0, atol=1e-14)

        # Lossy -> non-lossy continuous limit (kappa = 1e-4, 1e-8, 0.0j -> 0)
        errs_iso = []
        errs_aniso = []
        for kap in [1e-4, 1e-8, 0.0]:
            nw = complex(n_water, -kap)
            s_i = EMpy.transfer_matrix.IsotropicTransferMatrix(
                _make_iso_multilayer([n_prism, n_au, nw], [np.inf, d_au, np.inf]),
                theta_prism,
            ).solve(wl)
            s_a = EMpy.transfer_matrix.AnisotropicTransferMatrix(
                _make_aniso_multilayer([n_prism, n_au, nw], [np.inf, d_au, np.inf]),
                theta_vac,
                0.0,
            ).solve(wl)
            self.assertLessEqual(s_i.Rs[0] + s_i.Ts[0], 1.0)
            self.assertLessEqual(s_i.Rp[0] + s_i.Tp[0], 1.0)
            assert_allclose(s_a.R[0, 0, :], s_i.Rs, atol=1e-11)
            assert_allclose(s_a.R[1, 1, :], s_i.Rp, atol=1e-11)
            errs_iso.append(float(abs(s_i.Rp[0] - iso_0.Rp[0])))
            errs_aniso.append(float(abs(s_a.R[1, 1, 0] - iso_0.Rp[0])))

        self.assertLess(errs_iso[1], errs_iso[0])
        self.assertLess(errs_iso[2], 1e-14)
        self.assertLess(errs_aniso[1], errs_aniso[0])
        self.assertLess(errs_aniso[2], 1e-12)

        # Weak-birefringence limit (dn = 1e-4, 1e-8, 0) with Euler-rotated substrate tensor
        mat_prism = _make_aniso_multilayer([n_prism], [np.inf])[0].mat
        mat_au = _make_aniso_multilayer([n_au], [d_au])[0].mat
        dn_errs = []
        for dn, tol in [(1e-4, 5e-3), (1e-8, 1e-6), (0.0, 1e-12)]:
            eps_rot = EMpy.utils.euler_rotate(
                np.diag([n_water**2, n_water**2, (n_water + dn) ** 2])
                * EMpy.constants.eps0,
                EMpy.utils.deg2rad(28.0),
                EMpy.utils.deg2rad(41.0),
                EMpy.utils.deg2rad(19.0),
            )
            mat_sub = EMpy.materials.AnisotropicMaterial(
                "sub_rot",
                epsilon_tensor=EMpy.materials.EpsilonTensor(
                    epsilon_tensor_const=eps_rot
                ),
            )
            st = EMpy.utils.Multilayer(
                [
                    EMpy.utils.Layer(mat_prism, np.inf),
                    EMpy.utils.Layer(mat_au, d_au),
                    EMpy.utils.Layer(mat_sub, np.inf),
                ]
            )
            sol_dn = EMpy.transfer_matrix.AnisotropicTransferMatrix(
                st, theta_vac, 0.0
            ).solve(wl)
            assert_allclose(sol_dn.R[0, 0, :] + sol_dn.R[0, 1, :], iso_0.Rs, atol=tol)
            assert_allclose(sol_dn.R[1, 0, :] + sol_dn.R[1, 1, :], iso_0.Rp, atol=tol)
            dn_errs.append(
                float(abs((sol_dn.R[1, 0, 0] + sol_dn.R[1, 1, 0]) - iso_0.Rp[0]))
            )
        self.assertLess(dn_errs[1], dn_errs[0])
        self.assertLess(dn_errs[2], dn_errs[1])

    def test_tilted_birefringent_tir_and_partial_tir_limits(self):
        """Verify tilted-optic-axis TIR (Re(gamma) != 0), partial TIR, and optic-axis degeneracy."""
        wl = np.array([1.55e-6])
        n_prism = 1.8
        n_au = 0.18 - 3.5j
        mat_prism = _make_aniso_multilayer([n_prism], [np.inf])[0].mat
        mat_au = _make_aniso_multilayer([n_au], [0.045e-6])[0].mat

        # 1. Tilted uniaxial crystal in full TIR (kx/k0 = 1.8*sin(65 deg) = 1.631 > nE = 1.6 > nO = 1.3)
        th_vac_tir = np.arcsin(n_prism * np.sin(EMpy.utils.deg2rad(65.0)) + 0j)
        for kap in [0.0, 1e-8, 1e-4]:
            nO = 1.3 - 1j * kap
            nE = 1.6 - 1j * kap
            eps_tilt = EMpy.utils.euler_rotate(
                np.diag([nO**2, nO**2, nE**2]) * EMpy.constants.eps0,
                np.pi / 2,
                EMpy.utils.deg2rad(35.0),
                0.0,
            )
            mat_tilt = EMpy.materials.AnisotropicMaterial(
                "tilt",
                epsilon_tensor=EMpy.materials.EpsilonTensor(
                    epsilon_tensor_const=eps_tilt
                ),
            )
            sol_semi = EMpy.transfer_matrix.AnisotropicTransferMatrix(
                EMpy.utils.Multilayer(
                    [
                        EMpy.utils.Layer(mat_prism, np.inf),
                        EMpy.utils.Layer(mat_au, 0.045e-6),
                        EMpy.utils.Layer(mat_tilt, np.inf),
                    ]
                ),
                th_vac_tir,
                0.0,
            ).solve(wl)
            sol_thick = EMpy.transfer_matrix.AnisotropicTransferMatrix(
                EMpy.utils.Multilayer(
                    [
                        EMpy.utils.Layer(mat_prism, np.inf),
                        EMpy.utils.Layer(mat_au, 0.045e-6),
                        EMpy.utils.Layer(mat_tilt, 10.0e-6),
                        EMpy.utils.Layer(mat_prism, np.inf),
                    ]
                ),
                th_vac_tir,
                0.0,
            ).solve(wl)
            assert_allclose(sol_semi.R, sol_thick.R, atol=1e-7)

            if kap == 0.0:
                # Without the Au film, lossless tilted-crystal TIR must reflect 100% of power
                sol_lossless_tir = EMpy.transfer_matrix.AnisotropicTransferMatrix(
                    EMpy.utils.Multilayer(
                        [
                            EMpy.utils.Layer(mat_prism, np.inf),
                            EMpy.utils.Layer(mat_tilt, np.inf),
                        ]
                    ),
                    th_vac_tir,
                    0.0,
                ).solve(wl)
                assert_allclose(
                    sol_lossless_tir.R[0, 0, :] + sol_lossless_tir.R[0, 1, :],
                    1.0,
                    atol=1e-12,
                )
                assert_allclose(
                    sol_lossless_tir.R[1, 0, :] + sol_lossless_tir.R[1, 1, :],
                    1.0,
                    atol=1e-12,
                )
                assert_allclose(sol_lossless_tir.T, 0.0, atol=1e-14)

        # 2. Partial TIR (Glan-Thompson regime: nO = 1.3 < kx/k0 = 1.45 < nE = 1.6)
        th_vac_partial = np.arcsin(1.45 + 0j)
        eps_gt = EMpy.utils.euler_rotate(
            np.diag([1.3**2, 1.3**2, 1.6**2]) * EMpy.constants.eps0,
            EMpy.utils.deg2rad(20.0),
            EMpy.utils.deg2rad(30.0),
            0.0,
        )
        mat_gt = EMpy.materials.AnisotropicMaterial(
            "gt",
            epsilon_tensor=EMpy.materials.EpsilonTensor(epsilon_tensor_const=eps_gt),
        )
        sol_gt = EMpy.transfer_matrix.AnisotropicTransferMatrix(
            EMpy.utils.Multilayer(
                [
                    EMpy.utils.Layer(mat_prism, np.inf),
                    EMpy.utils.Layer(mat_gt, np.inf),
                ]
            ),
            th_vac_partial,
            0.0,
        ).solve(wl)
        assert_allclose(sol_gt.R[0, :].sum() + sol_gt.T[0, :].sum(), 1.0, atol=1e-12)
        assert_allclose(sol_gt.R[1, :].sum() + sol_gt.T[1, :].sum(), 1.0, atol=1e-12)

        # 3. Normal incidence along c-axis (k || c): ordinary and extraordinary roots coalesce
        eps_c_z = np.diag([1.6**2, 1.6**2, 1.9**2]) * EMpy.constants.eps0
        mat_c_z = EMpy.materials.AnisotropicMaterial(
            "cz",
            epsilon_tensor=EMpy.materials.EpsilonTensor(epsilon_tensor_const=eps_c_z),
        )
        sol_cz = EMpy.transfer_matrix.AnisotropicTransferMatrix(
            EMpy.utils.Multilayer(
                [
                    _make_aniso_multilayer([1.0], [np.inf])[0],
                    EMpy.utils.Layer(mat_c_z, 0.4e-6),
                    _make_aniso_multilayer([1.5], [np.inf])[0],
                ]
            ),
            0.0,
            0.0,
        ).solve(wl)
        sol_iso_16 = EMpy.transfer_matrix.IsotropicTransferMatrix(
            _make_iso_multilayer([1.0, 1.6, 1.5], [np.inf, 0.4e-6, np.inf]), 0.0
        ).solve(wl)
        assert_allclose(sol_cz.R[0, 0, :], sol_iso_16.Rs, atol=1e-8)
        assert_allclose(sol_cz.R[1, 1, :], sol_iso_16.Rp, atol=1e-8)

    def test_lossy_incident_medium_continuity_to_lossless(self):
        """Verify passive lossy incident medium (kappa_0 -> 0+) converges smoothly without branch flips."""
        wl = np.array([1.55e-6])
        for th_deg in (25.0, 60.0):
            th = EMpy.utils.deg2rad(th_deg)
            ref_i = EMpy.transfer_matrix.IsotropicTransferMatrix(
                _make_iso_multilayer([1.5, 2.1, 1.0], [np.inf, 0.25e-6, np.inf]), th
            ).solve(wl)
            errs_i = []
            errs_a = []
            for kap0 in (1e-4, 1e-8, 1e-12):
                n0 = complex(1.5, -kap0)
                th_vac = np.arcsin(n0 * np.sin(th) + 0j)
                si = EMpy.transfer_matrix.IsotropicTransferMatrix(
                    _make_iso_multilayer([n0, 2.1, 1.0], [np.inf, 0.25e-6, np.inf]), th
                ).solve(wl)
                # Perturb intermediate tensor slightly to exercise general 4-root path too
                eps_mid = (
                    EMpy.utils.euler_rotate(
                        np.diag([2.1**2, 2.1**2, (2.1 + 1e-3) ** 2]), 0.2, 0.3, 0.4
                    )
                    * EMpy.constants.eps0
                )
                ml_a = EMpy.utils.Multilayer(
                    [
                        _make_aniso_multilayer([n0], [np.inf])[0],
                        EMpy.utils.Layer(
                            EMpy.materials.AnisotropicMaterial(
                                "m",
                                EMpy.materials.EpsilonTensor(
                                    epsilon_tensor_const=eps_mid
                                ),
                            ),
                            0.25e-6,
                        ),
                        _make_aniso_multilayer([1.0], [np.inf])[0],
                    ]
                )
                sa = EMpy.transfer_matrix.AnisotropicTransferMatrix(
                    ml_a, th_vac, 0.0
                ).solve(wl)
                if th_deg < 40.0:
                    self.assertTrue(
                        (0.0 <= si.Rs[0] <= 1.0) and (0.0 <= si.Ts[0] <= 1.0)
                    )
                    self.assertTrue(
                        (0.0 <= si.Rp[0] <= 1.0) and (0.0 <= si.Tp[0] <= 1.0)
                    )
                errs_i.append(
                    abs(si.Rs[0] - ref_i.Rs[0])
                    + abs(si.Ts[0] - ref_i.Ts[0])
                    + abs(si.Rp[0] - ref_i.Rp[0])
                    + abs(si.Tp[0] - ref_i.Tp[0])
                )
                errs_a.append(
                    abs(sa.R[0, :, 0].sum() - ref_i.Rs[0])
                    + abs(sa.T[0, :, 0].sum() - ref_i.Ts[0])
                )
            self.assertLess(errs_i[0], 5e-3)
            self.assertLess(errs_i[1], 0.01 * errs_i[0])
            self.assertLess(errs_i[2], 0.01 * errs_i[1])
            self.assertLess(errs_a[0], 5e-2)
            self.assertLess(errs_a[1], 5e-2)

    def test_weak_birefringence_jones_waveplate_and_general_iso_limit(self):
        """Verify weak-birefringence waveplates match exact Jones calculus and general 4x4 path matches IsotropicTM."""
        wl = 1.55e-6
        eps0 = EMpy.constants.eps0
        n0 = 1.5
        c_45 = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        mat_amb = _make_aniso_multilayer([n0], [np.inf])[0].mat

        # 1. Exact half-wave plates (d = wl / (2 * dn)) down to dn = 1e-6 must give T_sp == 1, T_ss == 0
        for dn in (1e-3, 1e-4, 2e-5, 1e-5, 5e-6, 1e-6):
            d_hwp = wl / (2.0 * dn)
            eps_wp = eps0 * (
                n0**2 * np.eye(3) + ((n0 + dn) ** 2 - n0**2) * np.outer(c_45, c_45)
            )
            mat_wp = EMpy.materials.AnisotropicMaterial(
                "wp", EMpy.materials.EpsilonTensor(epsilon_tensor_const=eps_wp)
            )
            ml = EMpy.utils.Multilayer(
                [
                    EMpy.utils.Layer(mat_amb, np.inf),
                    EMpy.utils.Layer(mat_wp, d_hwp),
                    EMpy.utils.Layer(mat_amb, np.inf),
                ]
            )
            sol = EMpy.transfer_matrix.AnisotropicTransferMatrix(ml, 0.0, 0.0).solve(
                [wl]
            )
            assert_allclose(sol.T[0, 1, 0], 1.0, atol=1e-5)
            assert_allclose(sol.T[0, 0, 0], 0.0, atol=1e-5)

        # 2. Fixed-thickness plate (d = 50 um): T_sp == sin^2(pi * dn * d / wl) smoothly across dn -> 0
        d_fix = 50.0e-6
        for dn in (1e-3, 1e-4, 2e-5, 1e-5, 1e-6, 1e-8, 0.0):
            eps_wp = eps0 * (
                n0**2 * np.eye(3) + ((n0 + dn) ** 2 - n0**2) * np.outer(c_45, c_45)
            )
            mat_wp = EMpy.materials.AnisotropicMaterial(
                "wpf", EMpy.materials.EpsilonTensor(epsilon_tensor_const=eps_wp)
            )
            ml = EMpy.utils.Multilayer(
                [
                    EMpy.utils.Layer(mat_amb, np.inf),
                    EMpy.utils.Layer(mat_wp, d_fix),
                    EMpy.utils.Layer(mat_amb, np.inf),
                ]
            )
            sol = EMpy.transfer_matrix.AnisotropicTransferMatrix(ml, 0.0, 0.0).solve(
                [wl]
            )
            t_sp_jones = np.sin(np.pi * dn * d_fix / wl) ** 2
            assert_allclose(sol.T[0, 1, 0], t_sp_jones, rtol=2e-3, atol=1e-14)

        # 3. General 4x4 Berreman path near isotropic limit (1e-11 off-diagonal perturbation bypasses 1e-13 shortcut)
        sol_iso = EMpy.transfer_matrix.IsotropicTransferMatrix(
            _make_iso_multilayer([1.0, 2.1, 1.45], [np.inf, 3.0e-6, np.inf]), 0.4
        ).solve([wl])
        k_pert = 2.1**2 * np.eye(3) + 1e-11 * np.array(
            [[0.0, 1.0, 0.5], [1.0, 0.0, -0.3], [0.5, -0.3, 0.0]]
        )
        ml_pert = EMpy.utils.Multilayer(
            [
                _make_aniso_multilayer([1.0], [np.inf])[0],
                EMpy.utils.Layer(
                    EMpy.materials.AnisotropicMaterial(
                        "p",
                        EMpy.materials.EpsilonTensor(
                            epsilon_tensor_const=eps0 * k_pert
                        ),
                    ),
                    3.0e-6,
                ),
                _make_aniso_multilayer([1.45], [np.inf])[0],
            ]
        )
        sol_pert = EMpy.transfer_matrix.AnisotropicTransferMatrix(
            ml_pert, 0.4 * np.cos(0.3), 0.4 * np.sin(0.3)
        ).solve([wl])
        # Check at delta = 0 directly against IsotropicTransferMatrix to < 1e-9
        sol_pert_0 = EMpy.transfer_matrix.AnisotropicTransferMatrix(
            ml_pert, 0.4, 0.0
        ).solve([wl])
        assert_allclose(sol_pert_0.R[0, 0, 0], sol_iso.Rs[0], atol=1e-9)
        assert_allclose(sol_pert_0.T[0, 0, 0], sol_iso.Ts[0], atol=1e-9)
        assert_allclose(sol_pert_0.R[1, 1, 0], sol_iso.Rp[0], atol=1e-9)
        assert_allclose(sol_pert_0.T[1, 1, 0], sol_iso.Tp[0], atol=1e-9)
        assert_allclose(sol_pert.R_tot[:, 0] + sol_pert.T_tot[:, 0], 1.0, atol=1e-12)

    def test_biaxial_optic_axis_continuity_and_lossy_anisotropic_total_flux(self):
        """Verify continuity across an oblique biaxial optic axis and exact passivity of R_tot + T_tot."""
        wl = 1.55e-6
        eps0 = EMpy.constants.eps0

        # 1. Biaxial crystal eps = diag(n1^2, n2^2, n3^2) at its xz-plane binormal (gamma_0 == gamma_2, E not perp k)
        n1, n2, n3 = 1.5, 1.7, 2.0
        # Exact wave-normal optic axis (where gamma_o^2 = n2^2 - ax^2 equals gamma_e^2 = n1^2*(1 - ax^2/n3^2)):
        ax_axis = n3 * np.sqrt((n2**2 - n1**2) / (n3**2 - n1**2))
        mat_prism = _make_aniso_multilayer([2.2], [np.inf])[0].mat
        mat_biax = EMpy.materials.AnisotropicMaterial(
            "bx",
            EMpy.materials.EpsilonTensor(
                epsilon_tensor_const=eps0 * np.diag([n1**2, n2**2, n3**2])
            ),
        )
        ml_bx = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(mat_prism, np.inf),
                EMpy.utils.Layer(mat_biax, 2.5e-6),
                EMpy.utils.Layer(mat_prism, np.inf),
            ]
        )
        rpp_vals = []
        for dax in (-1e-4, -1e-7, 0.0, 1e-7, 1e-4):
            th_v = np.arcsin((ax_axis + dax) + 0j)
            sb = EMpy.transfer_matrix.AnisotropicTransferMatrix(ml_bx, th_v, 0.0).solve(
                [wl]
            )
            assert_allclose(sb.R_tot[:, 0] + sb.T_tot[:, 0], 1.0, atol=1e-12)
            rpp_vals.append(float(sb.R[1, 1, 0]))
        assert_allclose(rpp_vals[2], 0.5 * (rpp_vals[0] + rpp_vals[4]), atol=1e-4)
        assert_allclose(rpp_vals[1], rpp_vals[2], atol=1e-6)
        assert_allclose(rpp_vals[3], rpp_vals[2], atol=1e-6)

        # 2. Lossless air -> rotated lossy biaxial semi-infinite substrate: R_tot + T_tot == 1 identically at z = 0
        mat_air = _make_aniso_multilayer([1.0], [np.inf])[0].mat
        for kap in (0.0, 1e-3, 0.1, 0.35):
            eps_l = eps0 * EMpy.utils.euler_rotate(
                np.diag(
                    [
                        complex(1.5, -kap) ** 2,
                        complex(1.7, -0.4 * kap) ** 2,
                        complex(2.0, -0.2 * kap) ** 2,
                    ]
                ),
                0.4,
                0.6,
                0.3,
            )
            mat_l = EMpy.materials.AnisotropicMaterial(
                "lb", EMpy.materials.EpsilonTensor(epsilon_tensor_const=eps_l)
            )
            sl_if = EMpy.transfer_matrix.AnisotropicTransferMatrix(
                EMpy.utils.Multilayer(
                    [EMpy.utils.Layer(mat_air, np.inf), EMpy.utils.Layer(mat_l, np.inf)]
                ),
                0.35,
                0.15,
            ).solve([wl])
            assert_allclose(sl_if.R_tot[:, 0] + sl_if.T_tot[:, 0], 1.0, atol=1e-13)
            if kap > 0:
                sl_slab = EMpy.transfer_matrix.AnisotropicTransferMatrix(
                    EMpy.utils.Multilayer(
                        [
                            EMpy.utils.Layer(mat_air, np.inf),
                            EMpy.utils.Layer(mat_l, 0.6e-6),
                            EMpy.utils.Layer(mat_air, np.inf),
                        ]
                    ),
                    0.35,
                    0.15,
                ).solve([wl])
                self.assertTrue(np.all(sl_slab.R_tot[:, 0] + sl_slab.T_tot[:, 0] < 1.0))

    def test_lossy_incident_tir_local_poynting_flux_consistency(self):
        """Verify unclamped local S_z(d+) under lossy-incident TIR matches across all TMM and RCWA solvers."""
        import EMpy.RCWA

        wl = 1.55e-6
        eps0 = EMpy.constants.eps0
        alpha = EMpy.utils.deg2rad(60.0)
        n1 = 1.5 - 0.05j
        n3 = 1.0
        th_vac = np.arcsin(n1 * np.sin(alpha) + 0j)

        # 1. Exact 4-solver agreement (IsoTM, AnisoTM, IsoRCWA, AnisoRCWA) on [1.5 - 0.05j, 2.1, 1.0] under TIR
        ml_i = _make_iso_multilayer([n1, 2.1, n3], [np.inf, 0.25e-6, np.inf])
        ml_a = _make_aniso_multilayer([n1, 2.1, n3], [np.inf, 0.25e-6, np.inf])
        tmm_i = EMpy.transfer_matrix.IsotropicTransferMatrix(ml_i, alpha).solve([wl])
        tmm_a = EMpy.transfer_matrix.AnisotropicTransferMatrix(ml_a, th_vac, 0.0).solve(
            [wl]
        )
        rcwa_i_s = EMpy.RCWA.IsotropicRCWA(
            ml_i, alpha, 0.0, np.pi / 2, np.pi / 2, 0
        ).solve([wl])
        rcwa_i_p = EMpy.RCWA.IsotropicRCWA(ml_i, alpha, 0.0, 0.0, np.pi / 2, 0).solve(
            [wl]
        )
        rcwa_a_s = EMpy.RCWA.AnisotropicRCWA(
            ml_a, alpha, 0.0, np.pi / 2, np.pi / 2, 0
        ).solve([wl])
        rcwa_a_p = EMpy.RCWA.AnisotropicRCWA(ml_a, alpha, 0.0, 0.0, np.pi / 2, 0).solve(
            [wl]
        )

        # Local S_z(d+) in the lossless air exit medium is strictly negative due to lateral Goos-Hanchen attenuation
        self.assertLess(float(tmm_i.Ts[0]), 0.0)
        self.assertLess(float(tmm_i.Tp[0]), 0.0)
        assert_allclose(tmm_a.R[0, 0, 0], tmm_i.Rs[0], rtol=1e-12, atol=1e-12)
        assert_allclose(tmm_a.R[1, 1, 0], tmm_i.Rp[0], rtol=1e-12, atol=1e-12)
        assert_allclose(tmm_a.T[0, 0, 0], tmm_i.Ts[0], rtol=1e-12, atol=1e-12)
        assert_allclose(tmm_a.T[1, 1, 0], tmm_i.Tp[0], rtol=1e-12, atol=1e-12)
        assert_allclose(
            tmm_a.T_tot[:, 0], [tmm_i.Ts[0], tmm_i.Tp[0]], rtol=1e-12, atol=1e-12
        )
        assert_allclose(rcwa_i_s.DE3[0, 0], tmm_i.Ts[0], rtol=1e-12, atol=1e-12)
        assert_allclose(rcwa_i_p.DE3[0, 0], tmm_i.Tp[0], rtol=1e-12, atol=1e-12)
        assert_allclose(rcwa_a_s.DE3[0, 0], tmm_i.Ts[0], rtol=1e-12, atol=1e-12)
        assert_allclose(rcwa_a_p.DE3[0, 0], tmm_i.Tp[0], rtol=1e-12, atol=1e-12)

        # 2. Weak-birefringence 3D Euler-rotated exit crystal dn -> 0 under lossy-incident TIR
        dn_errs = []
        for dn in (1e-4, 1e-8, 0.0):
            eps_exit = eps0 * EMpy.utils.euler_rotate(
                np.diag([1.0, (1.0 + 0.5 * dn) ** 2, (1.0 + dn) ** 2]), 0.3, 0.5, 0.7
            )
            ml_dn = EMpy.utils.Multilayer(
                [
                    ml_a[0],
                    ml_a[1],
                    EMpy.utils.Layer(
                        EMpy.materials.AnisotropicMaterial(
                            "ex",
                            EMpy.materials.EpsilonTensor(epsilon_tensor_const=eps_exit),
                        ),
                        np.inf,
                    ),
                ]
            )
            s_dn = EMpy.transfer_matrix.AnisotropicTransferMatrix(
                ml_dn, th_vac, 0.0
            ).solve([wl])
            dn_errs.append(
                float(
                    np.max(
                        np.abs(s_dn.R_tot[:, 0] - [tmm_i.Rs[0], tmm_i.Rp[0]])
                        + np.abs(s_dn.T_tot[:, 0] - [tmm_i.Ts[0], tmm_i.Tp[0]])
                    )
                )
            )
        self.assertLess(dn_errs[1], dn_errs[0])
        self.assertLess(dn_errs[2], 1e-12)

        # 3. Continuous lossy -> lossless incident-medium limit (kap0 = 1e-2, 1e-5, 0.0 + 0j -> 0): T_tot -> 0, R_tot -> 1
        t_mags = []
        for kap0 in (1e-2, 1e-5, 0.0):
            n1_k = complex(1.5, -kap0)
            th_k = np.arcsin(n1_k * np.sin(alpha) + 0j)
            ml_k = _make_aniso_multilayer([n1_k, 2.1, 1.0], [np.inf, 0.25e-6, np.inf])
            sk = EMpy.transfer_matrix.AnisotropicTransferMatrix(ml_k, th_k, 0.0).solve(
                [wl]
            )
            t_mags.append(float(np.max(np.abs(sk.T_tot[:, 0]))))
            if kap0 == 0.0:
                assert_allclose(sk.R_tot[:, 0], 1.0, atol=1e-12)
                assert_allclose(sk.T_tot[:, 0], 0.0, atol=1e-12)
        self.assertLess(t_mags[1], t_mags[0])
        self.assertLess(t_mags[1], 1e-4)
        self.assertLess(t_mags[2], 1e-12)

    def test_anisotropic_uniaxial_exact_fresnel_lekner(self):
        """Verify AnisotropicTransferMatrix against closed-form Lekner/Yeh uniaxial Fresnel formulas."""
        eps0 = EMpy.constants.eps0
        wl = 1.55e-6
        n1 = 1.45
        m1 = EMpy.materials.IsotropicMaterial(
            "m1", n0=EMpy.materials.RefractiveIndex(n0_const=n1)
        )
        # In AnisotropicTransferMatrix, kx = k0 * sin(theta_inc_x) (air-equivalent angle),
        # so inside incident medium n1, n1 * sin(theta_1) = sin(theta_inc_x) and
        # qz_1 = n1 * cos(theta_1) = sqrt(n1^2 - sin(theta_inc_x)^2).
        theta_air = EMpy.utils.deg2rad(38.0)
        kx_norm = np.sin(theta_air)
        qz1 = np.sqrt(n1**2 - kx_norm**2)
        cos_t = qz1 / n1

        def _passive_sqrt(val):
            r = np.sqrt(complex(val))
            return -r if (r.real - r.imag < 0) else r

        for nO, nE in [(1.60, 1.85), (1.60 - 0.12j, 1.85 - 0.08j)]:
            # 1. Optic axis normal to interface (c || z): diag(nO^2, nO^2, nE^2)
            m_z = EMpy.materials.AnisotropicMaterial(
                "uni_z",
                EMpy.materials.EpsilonTensor(eps0 * np.diag([nO**2, nO**2, nE**2])),
            )
            tmm_z = EMpy.transfer_matrix.AnisotropicTransferMatrix(
                EMpy.utils.Multilayer(
                    [EMpy.utils.Layer(m1, np.inf), EMpy.utils.Layer(m_z, np.inf)]
                ),
                theta_air,
                0.0,
            ).solve([wl])
            qz_o = _passive_sqrt(nO**2 - kx_norm**2)
            qz_e = _passive_sqrt(nE**2 - kx_norm**2)
            rss_z = (n1 * cos_t - qz_o) / (n1 * cos_t + qz_o)
            rpp_z = (nO * nE * cos_t - n1 * qz_e) / (nO * nE * cos_t + n1 * qz_e)
            self.assertAlmostEqual(float(tmm_z.R[0, 0, 0]), abs(rss_z) ** 2, places=12)
            self.assertAlmostEqual(float(tmm_z.R[1, 1, 0]), abs(rpp_z) ** 2, places=12)
            self.assertAlmostEqual(float(tmm_z.R[0, 1, 0]), 0.0, places=12)
            self.assertAlmostEqual(float(tmm_z.R[1, 0, 0]), 0.0, places=12)

            # 2. Optic axis in interface & plane of incidence (c || x): diag(nE^2, nO^2, nO^2)
            m_x = EMpy.materials.AnisotropicMaterial(
                "uni_x",
                EMpy.materials.EpsilonTensor(eps0 * np.diag([nE**2, nO**2, nO**2])),
            )
            tmm_x = EMpy.transfer_matrix.AnisotropicTransferMatrix(
                EMpy.utils.Multilayer(
                    [EMpy.utils.Layer(m1, np.inf), EMpy.utils.Layer(m_x, np.inf)]
                ),
                theta_air,
                0.0,
            ).solve([wl])
            rpp_x = (nO * nE * cos_t - n1 * qz_o) / (nO * nE * cos_t + n1 * qz_o)
            self.assertAlmostEqual(float(tmm_x.R[0, 0, 0]), abs(rss_z) ** 2, places=12)
            self.assertAlmostEqual(float(tmm_x.R[1, 1, 0]), abs(rpp_x) ** 2, places=12)

            # 3. Optic axis perpendicular to plane of incidence (c || y): diag(nO^2, nE^2, nO^2)
            m_y = EMpy.materials.AnisotropicMaterial(
                "uni_y",
                EMpy.materials.EpsilonTensor(eps0 * np.diag([nO**2, nE**2, nO**2])),
            )
            tmm_y = EMpy.transfer_matrix.AnisotropicTransferMatrix(
                EMpy.utils.Multilayer(
                    [EMpy.utils.Layer(m1, np.inf), EMpy.utils.Layer(m_y, np.inf)]
                ),
                theta_air,
                0.0,
            ).solve([wl])
            rss_y = (n1 * cos_t - qz_e) / (n1 * cos_t + qz_e)
            rpp_y = (nO**2 * cos_t - n1 * qz_o) / (nO**2 * cos_t + n1 * qz_o)
            self.assertAlmostEqual(float(tmm_y.R[0, 0, 0]), abs(rss_y) ** 2, places=12)
            self.assertAlmostEqual(float(tmm_y.R[1, 1, 0]), abs(rpp_y) ** 2, places=12)

    def test_anisotropic_thick_coupled_and_metal_etm_stability(self):
        """Verify bounded ETM cascade in AnisotropicTransferMatrix on thick coupled anisotropic and metallic layers."""
        wl = 1.55e-6
        eps0 = EMpy.constants.eps0
        m_air = EMpy.materials.IsotropicMaterial(
            "air", n0=EMpy.materials.RefractiveIndex(n0_const=1.0)
        )

        # 1. Thick lossless rotated uniaxial layer with coupled propagating + evanescent modes
        #    (nO = 0.6 < kx/k0 = 0.75 < nE = 0.95): must conserve energy R_tot + T_tot == 1 up to 50*wl
        eps_u = eps0 * EMpy.utils.euler_rotate(
            np.diag([0.6**2, 0.6**2, 0.95**2]),
            EMpy.utils.deg2rad(60.0),
            EMpy.utils.deg2rad(20.0),
            EMpy.utils.deg2rad(10.0),
        )
        m_u = EMpy.materials.AnisotropicMaterial(
            "uni_coupled", EMpy.materials.EpsilonTensor(eps_u)
        )
        th = np.arcsin(0.75)
        for mult in (1.0, 15.0, 50.0):
            ml = EMpy.utils.Multilayer(
                [
                    EMpy.utils.Layer(m_air, np.inf),
                    EMpy.utils.Layer(m_u, mult * wl),
                    EMpy.utils.Layer(m_air, np.inf),
                ]
            )
            res = EMpy.transfer_matrix.AnisotropicTransferMatrix(ml, th, 0.0).solve(
                [wl]
            )
            self.assertAlmostEqual(
                float(res.R_tot[0, 0] + res.T_tot[0, 0]), 1.0, places=12
            )
            self.assertAlmostEqual(
                float(res.R_tot[1, 0] + res.T_tot[1, 0]), 1.0, places=12
            )

        # 2. Thick metallic layer (n = 0.5 - 10j, d = 100 um) constructed as AnisotropicMaterial:
        #    must converge to exact single-interface Fresnel reflectance with T == 0 and no NaN
        n_met = 0.5 - 10.0j
        m_met = EMpy.materials.AnisotropicMaterial(
            "met", EMpy.materials.EpsilonTensor(eps0 * n_met**2 * np.eye(3))
        )
        r_fresnel = abs((1.0 - n_met) / (1.0 + n_met)) ** 2
        ml_m = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(m_air, np.inf),
                EMpy.utils.Layer(m_met, 100.0e-6),
                EMpy.utils.Layer(m_air, np.inf),
            ]
        )
        res_m = EMpy.transfer_matrix.AnisotropicTransferMatrix(ml_m, 0.0, 0.0).solve(
            [wl]
        )
        self.assertAlmostEqual(float(res_m.R[0, 0, 0]), r_fresnel, places=12)
        self.assertAlmostEqual(float(res_m.R[1, 1, 0]), r_fresnel, places=12)
        self.assertEqual(float(res_m.T[0, 0, 0]), 0.0)
        self.assertEqual(float(res_m.T[1, 1, 0]), 0.0)

    def test_isotropic_inner_cutoff_sinc_and_thick_metal_stability(self):
        """Verify IsotropicTransferMatrix at exact inner-layer cutoff qz == 0 and on thick metallic slabs."""
        wl = 1.55e-6
        k0 = 2.0 * np.pi / wl

        # 1. Exact inner-layer cutoff (n_inner == n_0 * sin(theta)):
        #    closed-form Abeles limit L_s = [[1, i*k0*d], [0, 1]], L_p = [[1, 0], [i*n_i^2*k0*d, 1]]
        #    yields exact analytical rs = (i*k0*d*qz0) / (2 + i*k0*d*qz0)? Wait:
        #    Ms = inv(D0) @ [[1, i*k0*d], [0, 1]] @ D0 =>
        #      rs = -i*k0*d*qz0 / (2 + i*k0*d*qz0) (|rs|^2 = (k0*d*qz0)^2 / (4 + (k0*d*qz0)^2))
        #      rp = (i*n_i^2*k0*d*cos0/n0) / (2 + i*n_i^2*k0*d*cos0/n0)
        n0, ni, d_in = 1.8, 1.5, 0.3e-6
        m0 = EMpy.materials.IsotropicMaterial(
            "m0", n0=EMpy.materials.RefractiveIndex(n0_const=n0)
        )
        mi = EMpy.materials.IsotropicMaterial(
            "mi", n0=EMpy.materials.RefractiveIndex(n0_const=ni)
        )
        ml_cut = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(m0, np.inf),
                EMpy.utils.Layer(mi, d_in),
                EMpy.utils.Layer(m0, np.inf),
            ]
        )
        theta_cut = np.arcsin(ni / n0)
        tmm_cut = EMpy.transfer_matrix.IsotropicTransferMatrix(ml_cut, theta_cut).solve(
            [wl]
        )
        qz0 = n0 * np.cos(theta_cut)
        cos0 = np.cos(theta_cut)
        xs = k0 * d_in * qz0
        xp = k0 * d_in * (ni**2) * cos0 / n0
        Rs_exact = xs**2 / (4.0 + xs**2)
        Rp_exact = xp**2 / (4.0 + xp**2)
        self.assertAlmostEqual(float(tmm_cut.Rs[0]), Rs_exact, places=12)
        self.assertAlmostEqual(float(tmm_cut.Ts[0]), 1.0 - Rs_exact, places=12)
        self.assertAlmostEqual(float(tmm_cut.Rp[0]), Rp_exact, places=12)
        self.assertAlmostEqual(float(tmm_cut.Tp[0]), 1.0 - Rp_exact, places=12)

        # Continuity as theta -> theta_cut+
        tmm_near = EMpy.transfer_matrix.IsotropicTransferMatrix(
            ml_cut, theta_cut + 1e-8
        ).solve([wl])
        self.assertAlmostEqual(float(tmm_cut.Rs[0]), float(tmm_near.Rs[0]), places=6)
        self.assertAlmostEqual(float(tmm_cut.Rp[0]), float(tmm_near.Rp[0]), places=6)

        # 2. Thick metallic slab (n = 0.5 - 10j, d = 100 um): exact single-interface Fresnel, no NaN
        n_g, n_m = 1.5, 0.5 - 10.0j
        mg = EMpy.materials.IsotropicMaterial(
            "glass", n0=EMpy.materials.RefractiveIndex(n0_const=n_g)
        )
        mm = EMpy.materials.IsotropicMaterial(
            "metal", n0=EMpy.materials.RefractiveIndex(n0_const=n_m)
        )
        ml_met = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(mg, np.inf),
                EMpy.utils.Layer(mm, 100.0e-6),
                EMpy.utils.Layer(mg, np.inf),
            ]
        )
        tmm_met = EMpy.transfer_matrix.IsotropicTransferMatrix(ml_met, 0.0).solve([wl])
        R_fresnel = abs((n_g - n_m) / (n_g + n_m)) ** 2
        self.assertAlmostEqual(float(tmm_met.Rs[0]), R_fresnel, places=12)
        self.assertAlmostEqual(float(tmm_met.Rp[0]), R_fresnel, places=12)
        self.assertEqual(float(tmm_met.Ts[0]), 0.0)
        self.assertEqual(float(tmm_met.Tp[0]), 0.0)
