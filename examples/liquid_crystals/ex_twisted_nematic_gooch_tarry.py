"""90-degree Twisted Nematic (TN) liquid-crystal cell: Gooch–Tarry minima & Oseen–Frank switching.

Demonstrates ``EMpy.utils.LiquidCrystalCell``, ``EMpy.materials.get_10400_000_100``,
``Multilayer.capacitance()``, and ``AnisotropicTransferMatrix``:

1. Off-state (0 V) 90-deg TN cell transmission vs. Mauguin parameter
   u = 2 * d * delta_n / wl, reproducing Gooch & Tarry's exact analytical
   rotatory-power curve and first/second Mauguin maxima (u = sqrt(3), sqrt(15)).
2. Coupled 1D Oseen–Frank elastic + Laplace electrostatic BVP switching across
   voltage (0 V -> 6 V), showing director tilt/twist profiles, electro-optic
   threshold vs. the analytical Freedericksz voltage, and static cell capacitance.

References
----------
- C. H. Gooch and H. A. Tarry, "The optical properties of twisted nematic
  liquid crystal structures with twist angles <= 90 deg," *J. Phys. D: Appl.
  Phys.* **8**(13), 1575–1584 (1975).
- P. G. de Gennes and J. Prost, *The Physics of Liquid Crystals*, 2nd ed.
  (Oxford University Press, 1993).
- D. W. Berreman, "Optics in stratified and anisotropic media: 4 x 4-matrix
  formulation," *J. Opt. Soc. Am.* **62**(4), 502–510 (1972).
"""

from matplotlib import pyplot as plt
import numpy as np

import EMpy
from EMpy.constants import eps0


def main():
    wl = np.array([550e-9])
    lc = EMpy.materials.get_10400_000_100(50.0)
    dn = lc.nE - lc.nO
    n_avg = 0.5 * (lc.nO + lc.nE)

    # Index-matched isotropic bounding media isolate intrinsic TN polarization rotation
    amb = EMpy.materials.AnisotropicMaterial(
        "glass", EMpy.materials.EpsilonTensor(eps0 * n_avg**2 * np.eye(3))
    )

    # --- 1. Gooch-Tarry curve at V = 0 across Mauguin parameter u = 2 * d * dn / wl ---
    u_vals = np.linspace(0.2, 6.5, 45)
    t_cross_num = []
    for u in u_vals:
        d_cell = u * wl[0] / (2.0 * dn)
        cell = EMpy.utils.LiquidCrystalCell(
            lc,
            voltage=0.0,
            t_tot=d_cell,
            t_anchoring=0.0,
            pretilt=0.0,
            totaltwist=np.pi / 2.0,
            nlayers=45,
        )
        ml = EMpy.utils.Multilayer(
            [EMpy.utils.Layer(amb, np.inf), cell, EMpy.utils.Layer(amb, np.inf)]
        )
        sol = EMpy.transfer_matrix.AnisotropicTransferMatrix(ml, 0.0, 0.0).solve(wl)
        t_cross_num.append(float(sol.T[0, 1, 0]))

    t_cross_gt = 1.0 - np.sin(0.5 * np.pi * np.sqrt(1.0 + u_vals**2)) ** 2 / (
        1.0 + u_vals**2
    )

    # --- 2. Electro-optic Oseen-Frank + Laplace BVP switching vs. voltage ---
    d_tn = np.sqrt(3.0) * wl[0] / (2.0 * dn)  # First Mauguin maximum
    pretilt = EMpy.utils.deg2rad(2.0)
    k_eff = lc.K11 + 0.25 * (lc.K33 - 2.0 * lc.K22)
    v_th_tn = np.pi * np.sqrt(k_eff / (eps0 * lc.deleps))

    voltages = np.linspace(0.0, 5.0, 21)
    t_vs_v = []
    c_vs_v = []
    profiles = {}

    cell_v = EMpy.utils.LiquidCrystalCell(
        lc,
        voltage=0.0,
        t_tot=d_tn,
        t_anchoring=0.0,
        pretilt=pretilt,
        totaltwist=np.pi / 2.0,
        nlayers=41,
    )
    for v in voltages:
        cell_v.voltage = float(v)
        ml_v = EMpy.utils.Multilayer(
            [EMpy.utils.Layer(amb, np.inf), cell_v, EMpy.utils.Layer(amb, np.inf)]
        )
        sol_v = EMpy.transfer_matrix.AnisotropicTransferMatrix(ml_v, 0.0, 0.0).solve(wl)
        t_vs_v.append(float(sol_v.T[0, 1, 0]))
        c_vs_v.append(cell_v.capacitance(area=1e-4, wl=0) * 1e9)  # nF / cm^2
        if np.isclose(v, 0.0) or np.isclose(v, 2.0) or np.isclose(v, 4.0):
            profiles[float(v)] = (
                cell_v.normalized_sample_points.copy(),
                90.0 - cell_v.angles.copy(),
            )

    print(
        f"Gooch–Tarry 90-deg TN max |Berreman - analytical| = "
        f"{np.max(np.abs(np.array(t_cross_num) - t_cross_gt)):.3e}"
    )
    print(
        f"Analytical TN Freedericksz threshold V_th = {v_th_tn:.2f} V | "
        f"Capacitance(0 V -> 5 V) = {c_vs_v[0]:.2f} -> {c_vs_v[-1]:.2f} nF/cm^2"
    )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))

    ax1.plot(u_vals, t_cross_gt, "k-", linewidth=2, label="Gooch–Tarry exact")
    ax1.plot(u_vals, t_cross_num, "ro", markersize=4, label="LiquidCrystalCell + TMM")
    for m_idx, u_m in enumerate((np.sqrt(3.0), np.sqrt(15.0)), start=1):
        ax1.axvline(u_m, color="gray", linestyle=":", label=f"Mauguin #{m_idx}")
    ax1.set_title("90° TN Off-State vs. Mauguin Parameter u")
    ax1.set_xlabel("u = 2 d Δn / λ")
    ax1.set_ylabel("Cross-Polarized Transmittance T_⊥")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.plot(voltages, t_vs_v, "b-o", markersize=4, label="T_⊥(V)")
    ax2.axvline(
        v_th_tn, color="crimson", linestyle="--", label=f"V_th ≈ {v_th_tn:.2f} V"
    )
    ax2_c = ax2.twinx()
    ax2_c.plot(voltages, c_vs_v, "g--s", markersize=3.5, label="C(V) [nF/cm²]")
    ax2.set_title("Electro-Optic Switching & Cell Capacitance")
    ax2.set_xlabel("Applied Voltage (V)")
    ax2.set_ylabel("T_⊥", color="b")
    ax2_c.set_ylabel("Capacitance (nF / cm²)", color="g")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="center right", fontsize=8)

    for v_key, (z_norm, tilt_deg) in profiles.items():
        ax3.plot(z_norm, tilt_deg, label=f"V = {v_key:.1f} V")
    ax3.set_title("Oseen–Frank Director Tilt Profile θ(z)")
    ax3.set_xlabel("Normalized Depth z / d")
    ax3.set_ylabel("Director Tilt Angle (deg)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
