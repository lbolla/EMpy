"""Binary grating groove-depth sweep and TE/TM Fourier-order convergence (Moharam 1995).

Demonstrates ``EMpy.RCWA.IsotropicRCWA`` and ``EMpy.utils.BinaryGrating`` on
two canonical grating-diffraction benchmarks:

1. Groove-depth sweep (0 <= d / wl0 <= 2.5) of a dielectric Littrow binary
   grating (n_g = n_sub = 2.04, duty cycle f = 0.5, pitch = wl0) for TE and TM
   polarizations, illustrating numerical stability of the Enhanced
   Transmittance Matrix (ETM) formulation and exact lossless energy balance
   sum(DE1 + DE3) = 1.
2. Diffraction-order truncation convergence (orders n = 1..21) for both a
   lossless dielectric grating and a lossy gold metallic grating at 1.0 um
   (n_Au = 0.22 - 6.71j), showcasing Lalanne & Morris inverse-rule TM
   convergence and strict passivity sum(DE1 + DE3) < 1.

References
----------
- M. G. Moharam, E. B. Grann, D. A. Pommet, and T. K. Gaylord, "Formulation for
  stable and efficient implementation of the rigorous coupled-wave analysis of
  binary gratings," *J. Opt. Soc. Am. A* **12**(5), 1068–1076 (1995).
- M. G. Moharam, D. A. Pommet, E. B. Grann, and T. K. Gaylord, "Stable
  implementation of the rigorous coupled-wave analysis for surface-relief
  gratings: enhanced transmittance matrix approach," *J. Opt. Soc. Am. A*
  **12**(5), 1077–1086 (1995).
- P. Lalanne and G. M. Morris, "Highly improved convergence of the coupled-wave
  method for TM polarization," *J. Opt. Soc. Am. A* **13**(4), 779–784 (1996).
"""

from matplotlib import pyplot as plt
import numpy as np

import EMpy


def main():
    wl0 = np.array([1.0e-6])
    pitch = 1.0e-6
    n_sub_val = 2.04
    air = EMpy.materials.Air
    dielectric = EMpy.materials.IsotropicMaterial(
        "Dielectric", n0=EMpy.materials.RefractiveIndex(n0_const=n_sub_val)
    )

    # First-order Bragg / Littrow angle in air: sin(alpha) = wl0 / (2 * pitch) = 0.5 -> 30 deg
    alpha = np.arcsin(float(wl0[0]) / (2.0 * pitch))
    delta = 0.0
    phi = np.pi / 2.0  # Grating vector along x

    # --- 1. Groove-depth sweep (Moharam et al. 1995 surface-relief benchmark) ---
    n_harm = 10
    depths = np.linspace(0.0, 2.5e-6, 51)
    de3_te_m1 = np.zeros_like(depths)
    de3_te_0 = np.zeros_like(depths)
    de3_tm_m1 = np.zeros_like(depths)
    de3_tm_0 = np.zeros_like(depths)
    energy_err = np.zeros_like(depths)

    for i, d in enumerate(depths):
        stack = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(air, np.inf),
                EMpy.utils.BinaryGrating(air, dielectric, 0.5, pitch, float(d)),
                EMpy.utils.Layer(dielectric, np.inf),
            ]
        )
        sol_te = EMpy.RCWA.IsotropicRCWA(
            stack, alpha, delta, np.pi / 2.0, phi, n_harm
        ).solve(wl0)
        sol_tm = EMpy.RCWA.IsotropicRCWA(stack, alpha, delta, 0.0, phi, n_harm).solve(
            wl0
        )
        # Orders are [-n, ..., -1, 0, +1, ..., +n]; m = -1 is index n_harm - 1, m = 0 is n_harm
        de3_te_m1[i] = sol_te.DE3[n_harm - 1, 0]
        de3_te_0[i] = sol_te.DE3[n_harm, 0]
        de3_tm_m1[i] = sol_tm.DE3[n_harm - 1, 0]
        de3_tm_0[i] = sol_tm.DE3[n_harm, 0]
        energy_err[i] = abs(np.sum(sol_tm.DE1[:, 0] + sol_tm.DE3[:, 0]) - 1.0)

    # --- 2. Harmonic convergence on lossless dielectric vs. lossy Au grating ---
    gold = EMpy.materials.IsotropicMaterial(
        "Au_1um", n0=EMpy.materials.RefractiveIndex(n0_const=0.22 - 6.71j)
    )
    d_conv = 0.5e-6
    stack_diel = EMpy.utils.Multilayer(
        [
            EMpy.utils.Layer(air, np.inf),
            EMpy.utils.BinaryGrating(air, dielectric, 0.5, pitch, d_conv),
            EMpy.utils.Layer(dielectric, np.inf),
        ]
    )
    stack_au = EMpy.utils.Multilayer(
        [
            EMpy.utils.Layer(air, np.inf),
            EMpy.utils.BinaryGrating(air, gold, 0.5, pitch, 0.25e-6),
            EMpy.utils.Layer(gold, np.inf),
        ]
    )

    orders = np.arange(1, 22, 2)
    conv_diel_tm = np.zeros(len(orders))
    conv_au_te = np.zeros(len(orders))
    conv_au_tm = np.zeros(len(orders))
    au_total_power = np.zeros(len(orders))

    for i, n_ord in enumerate(orders):
        n_i = int(n_ord)
        s_diel = EMpy.RCWA.IsotropicRCWA(stack_diel, alpha, delta, 0.0, phi, n_i).solve(
            wl0
        )
        s_au_te = EMpy.RCWA.IsotropicRCWA(
            stack_au, alpha, delta, np.pi / 2.0, phi, n_i
        ).solve(wl0)
        s_au_tm = EMpy.RCWA.IsotropicRCWA(stack_au, alpha, delta, 0.0, phi, n_i).solve(
            wl0
        )
        # In Glytsis/EMpy convention K_x,m = k_x0 - m * 2*pi/pitch (index m + n_i):
        # m = +1 (index n_i + 1) is the Littrow retro-reflected order in air & substrate!
        conv_diel_tm[i] = s_diel.DE3[n_i + 1, 0]
        conv_au_te[i] = s_au_te.DE1[n_i + 1, 0]
        conv_au_tm[i] = s_au_tm.DE1[n_i + 1, 0]
        au_total_power[i] = float(np.sum(s_au_tm.DE1[:, 0] + s_au_tm.DE3[:, 0]))

    print(
        f"Dielectric Littrow grating max energy error across groove depths: "
        f"{np.max(energy_err):.2e}"
    )
    print(
        f"Lossy Au grating (n={orders[-1]}): R_+1(TE) = {conv_au_te[-1]:.4f}, "
        f"R_+1(TM) = {conv_au_tm[-1]:.4f}, absorptance A(TM) = {1.0 - au_total_power[-1]:.4f}"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax1.plot(depths / wl0[0], de3_te_m1, "b-", linewidth=1.8, label="TE T_{-1}")
    ax1.plot(depths / wl0[0], de3_te_0, "b--", alpha=0.75, label="TE T_0")
    ax1.plot(depths / wl0[0], de3_tm_m1, "r-", linewidth=1.8, label="TM T_{-1}")
    ax1.plot(depths / wl0[0], de3_tm_0, "r--", alpha=0.75, label="TM T_0")
    ax1.set_title("Dielectric Binary Grating vs. Groove Depth (α = 30°)")
    ax1.set_xlabel("Normalized Groove Depth d / λ₀")
    ax1.set_ylabel("Transmitted Diffraction Efficiency")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8.5)

    ax2.plot(orders, conv_diel_tm, "ks-", markersize=4.5, label="Dielectric TM T_{+1}")
    ax2.plot(orders, conv_au_te, "bo-", markersize=4.5, label="Lossy Au TE R_{+1}")
    ax2.plot(
        orders,
        conv_au_tm,
        "ro-",
        markersize=4.5,
        label="Lossy Au TM R_{+1} (Lalanne rule)",
    )
    ax2.plot(
        orders,
        1.0 - au_total_power,
        "r:",
        linewidth=1.6,
        label="Lossy Au TM Absorptance A",
    )
    ax2.set_title("RCWA Harmonic Truncation Convergence")
    ax2.set_xlabel("Retained Harmonic Order n (2n + 1 Harmonics)")
    ax2.set_ylabel("Efficiency / Absorptance")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8.5)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
