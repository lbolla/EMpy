"""Cholesteric chiral Bragg reflection band and tabulated LC data-file cell.

Demonstrates two additional capabilities of ``EMpy.utils.LiquidCrystalCell``:

1. **Chiral cholesteric Bragg reflector:** A planar helical LC stack over 8
   chiral pitches exhibits circularly polarized selective reflection inside the
   de Vries photonic bandgap ``[n_O * P, n_E * P]`` (reflecting ~100% of
   co-handed circular polarization while transmitting ~100% of cross-handed
   circular polarization, yielding R = 0.5 for unpolarized/linearly polarized
   input).
2. **Tabulated voltage-profile interpolation (``data_file``):** Loads director
   tilt profiles from ``examples/data/lc_data.dat`` across applied voltages and
   computes voltage-tuned phase retardation spectra.

References
----------
- H. de Vries, "Rotatory power and other optical properties of certain liquid
  crystals," *Acta Crystallogr.* **4**(3), 219–226 (1951).
- P. Yeh and C. Gu, *Optics of Liquid Crystal Displays*, 2nd ed. (Wiley, 2009),
  Chap. 7.
"""

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

import EMpy
from EMpy.constants import eps0


def main():
    # --- 1. Chiral cholesteric Bragg reflection band ---
    n_o, n_e = 1.50, 1.66
    pitch = 0.98e-6  # Helical pitch -> Bragg center lambda_0 = n_avg * P = 1.548 um
    n_pitches = 6
    d_chol = n_pitches * pitch
    q0 = 2.0 * np.pi / pitch

    lc_chol = EMpy.materials.LiquidCrystal(
        "Cholesteric",
        nO=n_o,
        nE=n_e,
        K11=12e-12,
        K22=7e-12,
        K33=18e-12,
        q0=q0,
    )
    cell_chol = EMpy.utils.LiquidCrystalCell(
        lc_chol,
        voltage=0.0,
        t_tot=d_chol,
        t_anchoring=0.0,
        pretilt=0.0,
        totaltwist=n_pitches * 2.0 * np.pi,
        nlayers=n_pitches * 14,
    )
    n_avg = 0.5 * (n_o + n_e)
    amb = EMpy.materials.AnisotropicMaterial(
        "amb", EMpy.materials.EpsilonTensor(eps0 * n_avg**2 * np.eye(3))
    )
    ml_chol = EMpy.utils.Multilayer(
        [EMpy.utils.Layer(amb, np.inf), cell_chol, EMpy.utils.Layer(amb, np.inf)]
    )

    wls_chol = np.linspace(1.35e-6, 1.75e-6, 81)
    sol_chol = EMpy.transfer_matrix.AnisotropicTransferMatrix(ml_chol, 0.0, 0.0).solve(
        wls_chol
    )
    wl_band_lo, wl_band_hi = n_o * pitch, n_e * pitch

    # --- 2. Tabulated data_file LC cell across voltage ---
    data_path = Path(__file__).resolve().parents[1] / "data" / "lc_data.dat"
    lc_e7 = EMpy.materials.LiquidCrystal("E7", nO=1.52, nE=1.74)
    wls_vis = np.linspace(1.30e-6, 1.70e-6, 121)
    glass = EMpy.materials.AnisotropicMaterial(
        "glass", EMpy.materials.EpsilonTensor(eps0 * 1.52**2 * np.eye(3))
    )

    datafile_spectra = {}
    for v_app in (0.0, 2.0, 4.0, 10.0):
        cell_df = EMpy.utils.LiquidCrystalCell(
            lc_e7,
            voltage=v_app,
            t_tot=5.0e-6,
            t_anchoring=20e-9,
            pretilt=EMpy.utils.deg2rad(3.5),
            totaltwist=np.pi / 2.0,
            nlayers=35,
            data_file=str(data_path),
        )
        ml_df = EMpy.utils.Multilayer(
            [EMpy.utils.Layer(glass, np.inf), cell_df, EMpy.utils.Layer(glass, np.inf)]
        )
        sol_df = EMpy.transfer_matrix.AnisotropicTransferMatrix(ml_df, 0.0, 0.0).solve(
            wls_vis
        )
        datafile_spectra[v_app] = sol_df.T[0, 1, :]

    idx_mid = int(np.argmin(np.abs(wls_chol - n_avg * pitch)))
    print(
        f"Cholesteric Bragg band: [{wl_band_lo * 1e6:.3f}, {wl_band_hi * 1e6:.3f}] um | "
        f"Linear-input R_tot(center) = {sol_chol.R_tot[0, idx_mid]:.4f} (~0.50 selective CP mirror)"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax1.plot(wls_chol * 1e6, sol_chol.R_tot[0, :], label="R_tot (x/y linear input)")
    ax1.plot(
        wls_chol * 1e6, sol_chol.T_tot[0, :], "--", label="T_tot (x/y linear input)"
    )
    ax1.axvspan(
        wl_band_lo * 1e6,
        wl_band_hi * 1e6,
        color="gold",
        alpha=0.25,
        label="de Vries band [n_O P, n_E P]",
    )
    ax1.set_title(f"Chiral Cholesteric Bragg Band ({n_pitches} Pitches)")
    ax1.set_xlabel("Wavelength (µm)")
    ax1.set_ylabel("Reflectance / Transmittance")
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8.5)

    for v_app, spec in datafile_spectra.items():
        ax2.plot(wls_vis * 1e6, spec, label=f"V = {v_app:g} V")
    ax2.set_title("Tabulated LC Cell (lc_data.dat) Cross-Polarized T_sp")
    ax2.set_xlabel("Wavelength (µm)")
    ax2.set_ylabel("Transmittance T_sp")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
