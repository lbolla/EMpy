"""Kretschmann surface-plasmon resonance (SPR) and lossy-to-lossless TIR limit.

Reproduces the classic Kretschmann–Raether attenuated total reflection (ATR)
geometry—a BK7 glass prism coupled through a 47 nm gold film to air at
lambda = 632.8 nm—and cross-checks ``IsotropicTransferMatrix`` against
``AnisotropicTransferMatrix`` across both:

1. Plasmonic TM (p-polarized) resonance absorption vs. TE (s-polarized) total
   internal reflection above the critical angle theta_c = arcsin(1 / n_prism).
2. Continuous lossy-to-lossless limit (kappa -> 0+), showing exact energy
   conservation R + T = 1 when kappa = 0 and resonant surface-plasmon
   absorption A_p = 1 - R_p - T_p when kappa > 0.

References
----------
- E. Kretschmann and H. Raether, "Radiative decay of non radiative surface
  plasmons excited by light," *Z. Naturforsch. A* **23**(12), 2135–2136 (1968).
- H. Raether, *Surface Plasmons on Smooth and Rough Surfaces and on Gratings*
  (Springer, 1988), Chap. 2.
"""

from matplotlib import pyplot as plt
import numpy as np

import EMpy
from EMpy.constants import eps0


def main():
    wl = np.array([632.8e-9])
    n_prism = 1.515
    n_au = 0.18 - 3.43j  # Passive convention n = n_r - i * kappa (kappa > 0)
    n_air = 1.0
    d_au = 47e-9

    angles_deg = np.linspace(36.0, 65.0, 241)
    theta_c_deg = EMpy.utils.rad2deg(np.arcsin(n_air / n_prism))
    eps_m = (n_au**2).real
    theta_spr_deg = EMpy.utils.rad2deg(
        np.arcsin(np.sqrt(eps_m * n_air**2 / (eps_m + n_air**2)) / n_prism)
    )

    prism_iso = EMpy.materials.IsotropicMaterial(
        "BK7", EMpy.materials.RefractiveIndex(n0_const=n_prism)
    )
    au_iso = EMpy.materials.IsotropicMaterial(
        "Au", EMpy.materials.RefractiveIndex(n0_const=n_au)
    )
    air_iso = EMpy.materials.IsotropicMaterial(
        "Air", EMpy.materials.RefractiveIndex(n0_const=n_air)
    )
    ml_iso = EMpy.utils.Multilayer(
        [
            EMpy.utils.Layer(prism_iso, np.inf),
            EMpy.utils.Layer(au_iso, d_au),
            EMpy.utils.Layer(air_iso, np.inf),
        ]
    )

    # Explicit isotropic permittivity tensors to exercise AnisotropicTransferMatrix
    ml_ani = EMpy.utils.Multilayer(
        [
            EMpy.utils.Layer(
                EMpy.materials.AnisotropicMaterial(
                    "BK7", EMpy.materials.EpsilonTensor(eps0 * n_prism**2 * np.eye(3))
                ),
                np.inf,
            ),
            EMpy.utils.Layer(
                EMpy.materials.AnisotropicMaterial(
                    "Au", EMpy.materials.EpsilonTensor(eps0 * n_au**2 * np.eye(3))
                ),
                d_au,
            ),
            EMpy.utils.Layer(
                EMpy.materials.AnisotropicMaterial(
                    "Air", EMpy.materials.EpsilonTensor(eps0 * n_air**2 * np.eye(3))
                ),
                np.inf,
            ),
        ]
    )

    rs_iso, rp_iso, ap_iso = [], [], []
    rp_ani = []
    for deg in angles_deg:
        th = EMpy.utils.deg2rad(deg)
        s_i = EMpy.transfer_matrix.IsotropicTransferMatrix(ml_iso, th).solve(wl)
        rs_iso.append(s_i.Rs[0])
        rp_iso.append(s_i.Rp[0])
        ap_iso.append(1.0 - s_i.Rp[0] - s_i.Tp[0])

        # AnisotropicTransferMatrix defines theta_inc_x in vacuum: sin(theta_vac) = n_prism * sin(th)
        th_vac = np.arcsin(n_prism * np.sin(th) + 0j)
        s_a = EMpy.transfer_matrix.AnisotropicTransferMatrix(ml_ani, th_vac, 0.0).solve(
            wl
        )
        rp_ani.append(s_a.R_tot[1, 0])

    rs_arr = np.array(rs_iso)
    rp_arr = np.array(rp_iso)
    ap_arr = np.array(ap_iso)
    rp_ani_arr = np.array(rp_ani)

    # Lossy -> lossless dielectric film sweep (n = 1.35 - i * kappa) under FTIR
    kappas = [0.15, 0.03, 0.005, 0.0]
    ftir_curves = {}
    for kap in kappas:
        film_k = EMpy.materials.IsotropicMaterial(
            "film", EMpy.materials.RefractiveIndex(n0_const=complex(1.35, -kap))
        )
        ml_k = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(prism_iso, np.inf),
                EMpy.utils.Layer(film_k, 250e-9),
                EMpy.utils.Layer(air_iso, np.inf),
            ]
        )
        rt_sum = [
            float(
                (
                    s := EMpy.transfer_matrix.IsotropicTransferMatrix(
                        ml_k, EMpy.utils.deg2rad(deg)
                    ).solve(wl)
                ).Rp[0]
                + s.Tp[0]
            )
            for deg in angles_deg
        ]
        ftir_curves[kap] = np.array(rt_sum)

    min_idx = int(np.argmin(rp_arr))
    print(
        f"Critical angle = {theta_c_deg:.2f} deg | Analytical SPR angle = {theta_spr_deg:.2f} deg | "
        f"Simulated SPR dip = {angles_deg[min_idx]:.2f} deg (Rp = {rp_arr[min_idx]:.4f})"
    )
    print(
        f"Max |IsotropicTM.Rp - AnisotropicTM.R_tot[p]| = {np.max(np.abs(rp_arr - rp_ani_arr)):.2e}"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax1.plot(angles_deg, rs_arr, label="Rs (TE, IsotropicTM)")
    ax1.plot(angles_deg, rp_arr, linewidth=2, label="Rp (TM, IsotropicTM)")
    ax1.plot(
        angles_deg[::6], rp_ani_arr[::6], "ko", markersize=3.5, label="AnisotropicTM"
    )
    ax1.plot(angles_deg, ap_arr, "--", label="TM Absorption 1 - Rp - Tp")
    ax1.axvline(
        theta_c_deg, color="gray", linestyle=":", label=f"θc = {theta_c_deg:.1f}°"
    )
    ax1.axvline(
        theta_spr_deg,
        color="crimson",
        linestyle=":",
        label=f"θ_SPR ≈ {theta_spr_deg:.1f}°",
    )
    ax1.set_title("Kretschmann SPR (BK7 / 47 nm Au / Air, 632.8 nm)")
    ax1.set_xlabel("Internal Incidence Angle in Prism (deg)")
    ax1.set_ylabel("Power Fraction")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8.5)

    for kap, curve in ftir_curves.items():
        lbl = "κ = 0 (lossless: Rp + Tp = 1)" if kap == 0.0 else f"κ = {kap}"
        ax2.plot(angles_deg, curve, label=lbl)
    ax2.axvline(theta_c_deg, color="gray", linestyle=":")
    ax2.set_title("Lossy → Lossless Limit: Total Power Rp + Tp")
    ax2.set_xlabel("Internal Incidence Angle in Prism (deg)")
    ax2.set_ylabel("Rp + Tp")
    ax2.set_ylim(0.25, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
