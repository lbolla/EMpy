"""Dispersive and lossy Fabry–Pérot etalon vs. IsotropicTransferMatrix.

Demonstrates ``EMpy.devices.Etalon`` on dispersive (``EMpy.materials.Si`` and
``BK7``) and lossy cavities, cross-checking ``Etalon.Te`` against full
electromagnetic ``EMpy.transfer_matrix.IsotropicTransferMatrix`` Airy fringes:

1. Dispersive silicon wafer etalon at normal incidence: compares ``Etalon.Te``
   (with natural uncoated Si/air Fresnel reflectivity ``R = ((n - 1)/(n + 1))^2``)
   against ``IsotropicTransferMatrix``, and compares group-index-corrected
   ``FSRwl`` (``ng = n - wl * dn/dwl``) against ``find_peaks`` fringe spacing.
2. High-finesse lossy cavity across continuous intra-cavity absorption
   ``alpha_cm1 -> 0+`` (``loss_cm2rix``), verifying peak transmission roll-off
   and broadening.

References
----------
- M. Born and E. Wolf, *Principles of Optics*, 7th ed. (Cambridge University
  Press, 1999), Sec. 7.6.
- C. Fabry and A. Pérot, "Théorie et applications d'une nouvelle méthode de
  spectroscopie interférentielle," *Ann. Chim. Phys.* **16**, 115–144 (1899).
"""

from matplotlib import pyplot as plt
import numpy as np

import EMpy


def main():
    # --- 1. Uncoated dispersive silicon etalon: Etalon vs. IsotropicTransferMatrix ---
    wls = np.linspace(1.545e-6, 1.555e-6, 1001)
    d_si = 80.0e-6
    si_mat = EMpy.materials.Si
    n_si = np.real(si_mat.n(wls))
    r_fresnel = ((n_si - 1.0) / (n_si + 1.0)) ** 2

    si_layer = EMpy.utils.Layer(si_mat, d_si)
    etalon_si = EMpy.devices.Etalon(si_layer, r_fresnel, theta=0.0).solve(wls)

    ml_si = EMpy.utils.Multilayer(
        [
            EMpy.utils.Layer(EMpy.materials.Air, np.inf),
            si_layer,
            EMpy.utils.Layer(EMpy.materials.Air, np.inf),
        ]
    )
    tmm_si = EMpy.transfer_matrix.IsotropicTransferMatrix(ml_si, 0.0).solve(wls)
    max_tmm_diff = float(np.max(np.abs(etalon_si.Te - tmm_si.Ts)))

    peaks_si = sorted(
        EMpy.utils.find_peaks(wls * 1e9, etalon_si.Te, threshold=0.5),
        key=lambda p: p.x,
    )
    measured_fsr_nm = float(np.mean(np.diff([p.x for p in peaks_si])))
    mid_idx = len(wls) // 2
    analytic_fsr_nm = float(etalon_si.FSRwl[mid_idx] * 1e9)

    # --- 2. High-finesse coated BK7 etalon at 12 deg across lossy kappa -> 0+ ---
    wls_hf = np.linspace(1.5490e-6, 1.5510e-6, 1001)
    d_cav = 250.0e-6
    r_coat = 0.90
    theta_ext = EMpy.utils.deg2rad(12.0)
    n_bk7_0 = float(np.real(EMpy.materials.BK7.n(1.55e-6).item()))
    theta_int = float(EMpy.utils.snell(theta_ext, np.array([1.0, n_bk7_0]))[1])

    alphas_cm1 = [0.0, 0.5, 2.0, 6.0]
    hf_traces = {}
    for a_cm in alphas_cm1:
        n_lossy = EMpy.utils.loss_cm2rix(n_bk7_0, a_cm, 1.55e-6)
        mat_c = EMpy.materials.IsotropicMaterial(
            f"Cavity_{a_cm}", n0=EMpy.materials.RefractiveIndex(n0_const=n_lossy)
        )
        et = EMpy.devices.Etalon(
            EMpy.utils.Layer(mat_c, d_cav), r_coat, theta_int
        ).solve(wls_hf)
        hf_traces[a_cm] = et

    et0 = hf_traces[0.0]
    print(
        f"Uncoated dispersive Si etalon: max |Etalon.Te - IsotropicTM.Ts| = "
        f"{max_tmm_diff:.3e}"
    )
    print(
        f"Group-index FSRwl = {analytic_fsr_nm:.4f} nm vs. find_peaks spacing = "
        f"{measured_fsr_nm:.4f} nm"
    )
    print(
        f"High-finesse coated etalon (R={r_coat}): finesse = {float(np.mean(et0.FINESSE)):.2f}, "
        f"FWHM = {float(np.mean(et0.FWHM)) * 1e-9:.2f} GHz"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4))

    ax1.plot(
        wls * 1e9,
        etalon_si.Te,
        "b-",
        linewidth=1.8,
        label="Etalon.Te (group-index dispersive)",
    )
    ax1.plot(
        wls * 1e9,
        tmm_si.Ts,
        "r--",
        linewidth=1.5,
        label="IsotropicTransferMatrix Ts",
    )
    ax1.set_title("80 µm Dispersive Silicon Etalon vs. Transfer Matrix")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Transmittance")
    ax1.set_ylim(0.25, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8.5)

    for a_cm, et in hf_traces.items():
        lbl = "Lossless (α = 0 cm⁻¹)" if a_cm == 0.0 else f"Lossy (α = {a_cm:g} cm⁻¹)"
        ax2.plot(wls_hf * 1e9, et.Te, linewidth=1.7, label=lbl)
    ax2.set_title("High-Finesse Etalon (R = 0.90): Lossy → Lossless (κ → 0⁺)")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Transmittance")
    ax2.set_ylim(-0.02, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8.5)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
