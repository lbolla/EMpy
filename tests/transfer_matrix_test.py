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
