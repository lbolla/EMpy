"""Quarter-wave anti-reflection coating and TiO2/SiO2 Distributed Bragg Reflector.

Demonstrates ``EMpy.transfer_matrix.IsotropicTransferMatrix`` together with
Sellmeier (``EMpy.materials.BK7``), constant, and thermo-optic isotropic
material models (``EMpy.materials.SiO2``):

1. Single-layer MgF2 quarter-wave anti-reflection (AR) coating on dispersive
   BK7 crown glass at oblique incidence (10 deg), compared against the
   analytical normal-incidence quarter-wave minimum.
2. 8-pair TiO2 / SiO2 Distributed Bragg Reflector (DBR) centered at 1.55 um,
   comparing the simulated stopband width against Macleod's analytical
   arcsin formula.

References
----------
- H. A. Macleod, *Thin-Film Optical Filters*, 4th ed. (CRC Press, 2010).
- M. Born and E. Wolf, *Principles of Optics*, 7th ed. (Cambridge University
  Press, 1999), Sec. 1.6.
"""

from matplotlib import pyplot as plt
import numpy as np

import EMpy


def main():
    wls = np.linspace(0.95e-6, 2.15e-6, 401)
    wl0 = 1.55e-6

    # --- 1. Quarter-wave AR coating (Air / MgF2 / dispersive BK7) ---
    n_mgf2 = 1.38
    mgf2 = EMpy.materials.IsotropicMaterial(
        "MgF2", n0=EMpy.materials.RefractiveIndex(n0_const=n_mgf2)
    )
    d_ar = wl0 / (4.0 * n_mgf2)
    ar_stack = EMpy.utils.Multilayer(
        [
            EMpy.utils.Layer(EMpy.materials.Air, np.inf),
            EMpy.utils.Layer(mgf2, d_ar),
            EMpy.utils.Layer(EMpy.materials.BK7, np.inf),
        ]
    )

    theta_ar = EMpy.utils.deg2rad(10.0)
    sol_ar = EMpy.transfer_matrix.IsotropicTransferMatrix(ar_stack, theta_ar).solve(wls)
    n_bk7_0 = float(np.real(EMpy.materials.BK7.n(wl0).item()))
    r0_theory = ((1.0 * n_bk7_0 - n_mgf2**2) / (1.0 * n_bk7_0 + n_mgf2**2)) ** 2

    # --- 2. 8-pair TiO2 / SiO2 quarter-wave DBR on BK7 ---
    n_tio2 = 2.45
    tio2 = EMpy.materials.IsotropicMaterial(
        "TiO2", n0=EMpy.materials.RefractiveIndex(n0_const=n_tio2)
    )
    sio2 = EMpy.materials.SiO2
    n_sio2 = float(np.real(sio2.n(wl0).item()))
    d_h = wl0 / (4.0 * n_tio2)
    d_l = wl0 / (4.0 * n_sio2)

    n_pairs = 8
    dbr_layers = [EMpy.utils.Layer(EMpy.materials.Air, np.inf)]
    for _ in range(n_pairs):
        dbr_layers.append(EMpy.utils.Layer(tio2, d_h))
        dbr_layers.append(EMpy.utils.Layer(sio2, d_l))
    dbr_layers.append(EMpy.utils.Layer(EMpy.materials.BK7, np.inf))
    dbr_stack = EMpy.utils.Multilayer(dbr_layers)

    sol_dbr = EMpy.transfer_matrix.IsotropicTransferMatrix(dbr_stack, 0.0).solve(wls)
    dwl_macleod = wl0 * (4.0 / np.pi) * np.arcsin((n_tio2 - n_sio2) / (n_tio2 + n_sio2))

    idx0 = int(np.argmin(np.abs(wls - wl0)))
    print(
        f"AR coating at {wl0 * 1e6:.2f} um (10 deg): Rs = {sol_ar.Rs[idx0]:.5f}, "
        f"Rp = {sol_ar.Rp[idx0]:.5f} (0-deg theory R = {r0_theory:.5f})"
    )
    print(
        f"8-pair TiO2/SiO2 DBR peak R({wl0 * 1e6:.2f} um) = {sol_dbr.Rs[idx0]:.6f}, "
        f"Macleod stopband width = {dwl_macleod * 1e9:.1f} nm"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    ax1.plot(wls * 1e6, 10.0 * np.log10(sol_ar.Rs), label="Rs (10°)")
    ax1.plot(wls * 1e6, 10.0 * np.log10(sol_ar.Rp), "--", label="Rp (10°)")
    ax1.axhline(
        10.0 * np.log10(r0_theory),
        color="k",
        linestyle=":",
        label="Normal-incidence QW theory",
    )
    ax1.set_title("MgF2 Quarter-Wave AR Coating on BK7")
    ax1.set_xlabel("Wavelength (µm)")
    ax1.set_ylabel("Reflectance (dB)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(wls * 1e6, sol_dbr.Rs, label="Reflectance R")
    ax2.plot(wls * 1e6, sol_dbr.Ts, "--", label="Transmittance T")
    ax2.axvspan(
        (wl0 - 0.5 * dwl_macleod) * 1e6,
        (wl0 + 0.5 * dwl_macleod) * 1e6,
        color="gray",
        alpha=0.2,
        label="Macleod stopband",
    )
    ax2.set_title(f"{n_pairs}-Pair TiO2/SiO2 DBR on BK7")
    ax2.set_xlabel("Wavelength (µm)")
    ax2.set_ylabel("Power")
    ax2.set_ylim(-0.02, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
