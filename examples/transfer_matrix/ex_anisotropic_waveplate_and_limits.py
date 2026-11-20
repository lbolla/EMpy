"""Anisotropic 4x4 Yeh–Berreman waveplate and degenerate isotropic limit.

Demonstrates ``EMpy.transfer_matrix.AnisotropicTransferMatrix`` on arbitrarily
Euler-rotated birefringent multilayers and verifies two physical benchmarks:

1. Cross-polarized conversion (T_ps, T_sp) of an anti-reflection-coated quartz
   waveplate oriented at 45 deg azimuth vs. analytical Jones retardance
   sin^2(pi * delta_n * d / wl).
2. Continuous convergence of ``AnisotropicTransferMatrix`` (with 3D
   ``euler_rotate`` off-diagonal permittivity tensors) to
   ``IsotropicTransferMatrix`` as birefringence delta_n -> 0 at oblique
   incidence.

References
----------
- D. W. Berreman, "Optics in stratified and anisotropic media: 4 x 4-matrix
  formulation," *J. Opt. Soc. Am.* **62**(4), 502–510 (1972).
- P. Yeh, "Electromagnetic propagation in birefringent layered media,"
  *J. Opt. Soc. Am.* **69**(5), 742–756 (1979); *Optical Waves in Layered Media*
  (Wiley, 1988).
"""

from matplotlib import pyplot as plt
import numpy as np

import EMpy
from EMpy.constants import eps0


def main():
    wls = np.linspace(1.20e-6, 1.90e-6, 201)

    # --- 1. Index-matched birefringent waveplate at 45 deg optic-axis azimuth ---
    n_o, n_e = 1.544, 1.553  # Quartz-like birefringence (dn = 0.009)
    n_amb = 0.5 * (n_o + n_e)
    wl_hw = 1.55e-6
    d_wp = wl_hw / (2.0 * (n_e - n_o))  # Half-wave thickness at 1.55 um

    eps_wp = eps0 * EMpy.utils.euler_rotate(
        np.diag([n_e**2, n_o**2, n_o**2]),
        EMpy.utils.deg2rad(45.0),
        0.0,
        0.0,
    )
    amb_mat = EMpy.materials.AnisotropicMaterial(
        "ambient", EMpy.materials.EpsilonTensor(eps0 * n_amb**2 * np.eye(3))
    )
    wp_mat = EMpy.materials.AnisotropicMaterial(
        "waveplate", EMpy.materials.EpsilonTensor(eps_wp)
    )
    ml_wp = EMpy.utils.Multilayer(
        [
            EMpy.utils.Layer(amb_mat, np.inf),
            EMpy.utils.Layer(wp_mat, d_wp),
            EMpy.utils.Layer(amb_mat, np.inf),
        ]
    )

    sol_wp = EMpy.transfer_matrix.AnisotropicTransferMatrix(ml_wp, 0.0, 0.0).solve(wls)
    gamma_jones = 2.0 * np.pi * (n_e - n_o) * d_wp / wls
    t_cross_jones = np.sin(0.5 * gamma_jones) ** 2
    t_co_jones = np.cos(0.5 * gamma_jones) ** 2

    # --- 2. 3D Euler-rotated biaxial multilayer: Anisotropic -> Isotropic (dn -> 0) ---
    theta_inc = EMpy.utils.deg2rad(22.0)
    n_base = [1.0, 2.1, 1.46, 3.2, 1.5]
    d_stack = [np.inf, 0.45e-6, 0.85e-6, 0.22e-6, np.inf]

    ml_iso = EMpy.utils.Multilayer(
        [
            EMpy.utils.Layer(
                EMpy.materials.IsotropicMaterial(
                    f"iso_{i}", EMpy.materials.RefractiveIndex(n0_const=n_val)
                ),
                d_val,
            )
            for i, (n_val, d_val) in enumerate(zip(n_base, d_stack))
        ]
    )
    sol_iso = EMpy.transfer_matrix.IsotropicTransferMatrix(ml_iso, theta_inc).solve(wls)

    dn_values = [0.15, 0.02, 1e-3, 1e-6, 0.0]
    max_diffs = []
    aniso_traces = {}
    for dn in dn_values:
        layers_a = []
        for i, (n_val, d_val) in enumerate(zip(n_base, d_stack)):
            if np.isinf(d_val):
                eps_i = eps0 * n_val**2 * np.eye(3)
            else:
                eps_i = eps0 * EMpy.utils.euler_rotate(
                    np.diag([n_val**2, (n_val + 0.5 * dn) ** 2, (n_val + dn) ** 2]),
                    0.35 * (i + 1),
                    0.50,
                    0.65,
                )
            layers_a.append(
                EMpy.utils.Layer(
                    EMpy.materials.AnisotropicMaterial(
                        f"ani_{i}", EMpy.materials.EpsilonTensor(eps_i)
                    ),
                    d_val,
                )
            )
        sol_a = EMpy.transfer_matrix.AnisotropicTransferMatrix(
            EMpy.utils.Multilayer(layers_a), theta_inc, 0.0
        ).solve(wls)
        err = float(
            np.max(
                np.abs(sol_a.R_tot[0, :] - sol_iso.Rs)
                + np.abs(sol_a.R_tot[1, :] - sol_iso.Rp)
            )
        )
        max_diffs.append(err)
        if dn in (0.15, 0.02, 0.0):
            aniso_traces[dn] = sol_a.R_tot[1, :]

    print(
        f"Half-wave plate max |T_ps - Jones sin^2(Gamma/2)| = "
        f"{np.max(np.abs(sol_wp.T[0, 1, :] - t_cross_jones)):.3e}"
    )
    for dn, err in zip(dn_values, max_diffs):
        print(f"  dn = {dn:<8g} -> max |AnisotropicTM - IsotropicTM| = {err:.3e}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax1.plot(wls * 1e6, sol_wp.T[0, 1, :], linewidth=2, label="T_sp (Berreman 4×4)")
    ax1.plot(wls * 1e6, sol_wp.T[0, 0, :], linewidth=2, label="T_ss (Berreman 4×4)")
    ax1.plot(wls * 1e6, t_cross_jones, "k:", label="Jones sin²(Γ/2)")
    ax1.plot(wls * 1e6, t_co_jones, "k--", alpha=0.6, label="Jones cos²(Γ/2)")
    ax1.axvline(wl_hw * 1e6, color="gray", linestyle=":", alpha=0.7)
    ax1.set_title("45° Quartz Half-Wave Plate vs. Jones Calculus")
    ax1.set_xlabel("Wavelength (µm)")
    ax1.set_ylabel("Transmittance")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8.5)

    ax2.plot(wls * 1e6, sol_iso.Rp, "k-", linewidth=2.2, label="IsotropicTM Rp")
    for dn, trace in aniso_traces.items():
        style = "--" if dn == 0.0 else "-"
        ax2.plot(wls * 1e6, trace, style, label=f"AnisotropicTM R_tot,p (Δn={dn:g})")
    ax2.set_title("Rotated Biaxial Stack: Δn → 0 Convergence (θ = 22°)")
    ax2.set_xlabel("Wavelength (µm)")
    ax2.set_ylabel("p-Polarized Reflectance")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8.5)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
