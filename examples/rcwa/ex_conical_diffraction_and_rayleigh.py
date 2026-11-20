"""Conical diffraction Wood–Rayleigh anomalies and Anisotropic-to-Isotropic limits.

Demonstrates ``EMpy.RCWA.IsotropicRCWA``, ``EMpy.RCWA.AnisotropicRCWA``,
``dispersion_relation_ordinary``, and ``dispersion_relation_extraordinary`` in a
3D conical diffraction mount (azimuth delta != 0, mixed polarization psi = 45 deg):

1. Spectral sweep across substrate and superstrate Rayleigh cut-off wavelengths
   where diffracted wavevector normal components k_{z,m} vanish, producing sharp
   Wood–Rayleigh anomalies in the passing-off and remaining orders.
2. Exact numerical equivalence of ``AnisotropicRCWA`` (with 3D Euler-rotated
   birefringent grating tensors as delta_n -> 0) and ``IsotropicRCWA`` across
   both lossless and passive lossy (kappa -> 0+) gratings.

References
----------
- Lord Rayleigh, "On the dynamical theory of gratings," *Proc. R. Soc. London
  Ser. A* **79**(532), 399–416 (1907).
- E. N. Glytsis and T. K. Gaylord, "Three-dimensional (vector) rigorous
  coupled-wave analysis of anisotropic grating diffraction," *J. Opt. Soc. Am. A*
  **7**(8), 1399–1419 (1990).
- M. G. Moharam, E. B. Grann, D. A. Pommet, and T. K. Gaylord, "Formulation for
  stable and efficient implementation of the rigorous coupled-wave analysis of
  binary gratings," *J. Opt. Soc. Am. A* **12**(5), 1068–1076 (1995).
"""

from matplotlib import pyplot as plt
import numpy as np

import EMpy
from EMpy.constants import eps0


def rayleigh_wavelength(pitch, n_medium, alpha, delta, m_order):
    """Solve |(n1*sin(alpha)*cos(delta) - m*wl/pitch, n1*sin(alpha)*sin(delta))| = n_medium."""
    # With n1 = 1 (air): (sin(alpha)*cos(delta) - m*u)^2 + sin^2(alpha)*sin^2(delta) = n_m^2
    # where u = wl / pitch > 0.
    kx0 = np.sin(alpha) * np.cos(delta)
    ky0 = np.sin(alpha) * np.sin(delta)
    rad = n_medium**2 - ky0**2
    if rad <= 0 or m_order == 0:
        return np.nan
    roots = [(kx0 + s * np.sqrt(rad)) / m_order for s in (-1.0, 1.0)]
    pos = [r * pitch for r in roots if r > 0]
    return min(pos) if pos else np.nan


def main():
    pitch = 1.0e-6
    thickness = 0.42e-6
    dc = 0.45
    alpha = EMpy.utils.deg2rad(18.0)
    delta = EMpy.utils.deg2rad(32.0)
    psi = EMpy.utils.deg2rad(45.0)
    phi = np.pi / 2.0
    n_harm = 5

    n_air = 1.0
    n_ridge = 2.05
    n_sub = 1.50
    wls = np.linspace(0.85e-6, 1.85e-6, 141)

    air_iso = EMpy.materials.Air
    sub_iso = EMpy.materials.IsotropicMaterial(
        "Glass", n0=EMpy.materials.RefractiveIndex(n0_const=n_sub)
    )
    ridge_iso = EMpy.materials.IsotropicMaterial(
        "Si3N4", n0=EMpy.materials.RefractiveIndex(n0_const=n_ridge)
    )

    ml_iso = EMpy.utils.Multilayer(
        [
            EMpy.utils.Layer(air_iso, np.inf),
            EMpy.utils.BinaryGrating(air_iso, ridge_iso, dc, pitch, thickness),
            EMpy.utils.Layer(sub_iso, np.inf),
        ]
    )
    sol_iso = EMpy.RCWA.IsotropicRCWA(ml_iso, alpha, delta, psi, phi, n_harm).solve(wls)

    # Compute analytical Rayleigh cut-off wavelengths and verify kz_m -> 0 via dispersion_relation_ordinary
    wl_r_air_m1 = rayleigh_wavelength(pitch, n_air, alpha, delta, +1)
    wl_r_sub_m1 = rayleigh_wavelength(pitch, n_sub, alpha, delta, +1)
    wl_r_sub_p1 = rayleigh_wavelength(pitch, n_sub, alpha, delta, -1)

    k0 = 2.0 * np.pi / wls
    kx_p1 = k0 * np.sin(alpha) * np.cos(delta) - 2.0 * np.pi / pitch
    ky_0 = k0 * np.sin(alpha) * np.sin(delta)
    kz_sub_p1 = EMpy.RCWA.dispersion_relation_ordinary(kx_p1, ky_0, k0, n_sub)
    kz_e_check = EMpy.RCWA.dispersion_relation_extraordinary(
        kx_p1, ky_0, k0, n_sub, n_sub, np.array([0.4, 0.5, np.sqrt(1.0 - 0.41)])
    )

    # --- 2. Conical AnisotropicRCWA -> IsotropicRCWA across dn -> 0 & kappa -> 0+ ---
    wl_test = np.array([1.30e-6, 1.55e-6])
    air_ani = EMpy.materials.AnisotropicMaterial(
        "Air_ani", EMpy.materials.EpsilonTensor(eps0 * np.eye(3))
    )
    sub_ani = EMpy.materials.AnisotropicMaterial(
        "Glass_ani", EMpy.materials.EpsilonTensor(eps0 * n_sub**2 * np.eye(3))
    )

    dn_list = [0.20, 0.05, 1e-3, 1e-6, 0.0]
    kappa_list = [0.08, 1e-3, 0.0]
    diff_grid = np.zeros((len(kappa_list), len(dn_list)))

    for ik, kappa in enumerate(kappa_list):
        n_lossy = n_ridge - 1j * kappa
        ridge_lossy_iso = EMpy.materials.IsotropicMaterial(
            "Lossy_iso", n0=EMpy.materials.RefractiveIndex(n0_const=n_lossy)
        )
        s_ref = EMpy.RCWA.IsotropicRCWA(
            EMpy.utils.Multilayer(
                [
                    EMpy.utils.Layer(air_iso, np.inf),
                    EMpy.utils.BinaryGrating(
                        air_iso, ridge_lossy_iso, dc, pitch, thickness
                    ),
                    EMpy.utils.Layer(sub_iso, np.inf),
                ]
            ),
            alpha,
            delta,
            psi,
            phi,
            n_harm,
        ).solve(wl_test)

        for idn, dn in enumerate(dn_list):
            eps_rot = eps0 * EMpy.utils.euler_rotate(
                np.diag([n_lossy**2, (n_lossy + 0.5 * dn) ** 2, (n_lossy + dn) ** 2]),
                0.4,
                0.6,
                0.8,
            )
            ridge_ani = EMpy.materials.AnisotropicMaterial(
                "Lossy_ani", EMpy.materials.EpsilonTensor(eps_rot)
            )
            s_ani = EMpy.RCWA.AnisotropicRCWA(
                EMpy.utils.Multilayer(
                    [
                        EMpy.utils.Layer(air_ani, np.inf),
                        EMpy.utils.BinaryGrating(
                            air_ani, ridge_ani, dc, pitch, thickness
                        ),
                        EMpy.utils.Layer(sub_ani, np.inf),
                    ]
                ),
                alpha,
                delta,
                psi,
                phi,
                n_harm,
            ).solve(wl_test)
            diff_grid[ik, idn] = float(
                np.max(np.abs(s_ani.DE1 - s_ref.DE1) + np.abs(s_ani.DE3 - s_ref.DE3))
            )

    print(
        f"Rayleigh cut-off wavelengths (m=+1 air, m=+1 sub, m=-1 sub): "
        f"{wl_r_air_m1 * 1e6:.4f} um, {wl_r_sub_m1 * 1e6:.4f} um, {wl_r_sub_p1 * 1e6:.4f} um"
    )
    print(
        f"max |kz_E(nE=nO) - kz_O| = {np.max(np.abs(kz_e_check - kz_sub_p1)):.2e}; "
        f"max |AnisotropicRCWA - IsotropicRCWA| at dn=0 across kappa={kappa_list}: "
        f"{np.max(diff_grid[:, -1]):.2e}"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4))

    # Note: in Glytsis/EMpy convention K_x,i = k_x0 - i * 2*pi/pitch where index = i + n_harm
    ax1.plot(wls * 1e6, sol_iso.DE3[n_harm, :], "k-", linewidth=1.8, label="T_0")
    ax1.plot(wls * 1e6, sol_iso.DE3[n_harm + 1, :], "b-", linewidth=1.6, label="T_{+1}")
    ax1.plot(wls * 1e6, sol_iso.DE3[n_harm - 1, :], "r-", linewidth=1.6, label="T_{-1}")
    ax1.plot(wls * 1e6, sol_iso.DE1[n_harm, :], "g--", linewidth=1.5, label="R_0")
    for wl_r, lbl in [
        (wl_r_air_m1, "Rayleigh air (+1)"),
        (wl_r_sub_m1, "Rayleigh sub (+1)"),
        (wl_r_sub_p1, "Rayleigh sub (-1)"),
    ]:
        if np.isfinite(wl_r) and wls[0] <= wl_r <= wls[-1]:
            ax1.axvline(wl_r * 1e6, color="gray", linestyle=":", linewidth=1.2)
            ax1.text(
                wl_r * 1e6 + 0.01,
                0.82,
                lbl,
                rotation=90,
                fontsize=7.5,
                va="top",
            )
    ax1.set_title("Conical Diffraction & Wood–Rayleigh Anomalies")
    ax1.set_xlabel("Wavelength (µm)")
    ax1.set_ylabel("Diffraction Efficiency")
    ax1.set_ylim(-0.02, 0.92)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8.5)

    dn_plot = np.where(np.array(dn_list) == 0.0, 1e-16, np.array(dn_list))
    markers = ["o-", "s--", "^:"]
    for ik, kappa in enumerate(kappa_list):
        ax2.loglog(
            dn_plot,
            np.maximum(diff_grid[ik, :], 1e-16),
            markers[ik],
            linewidth=1.7,
            label=f"κ = {kappa:g}",
        )
    ax2.set_title("Conical AnisotropicRCWA → IsotropicRCWA (Δn → 0)")
    ax2.set_xlabel("Birefringence Δn (0 plotted at 10⁻¹⁶)")
    ax2.set_ylabel("max |DE_aniso − DE_iso|")
    ax2.invert_xaxis()
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8.5)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
