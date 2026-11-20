"""Symmetry-protected BIC and quasi-BIC Fano resonances in dimer double gratings.

Demonstrates ``EMpy.utils.SymmetricDoubleGrating``,
``EMpy.utils.AsymmetricDoubleGrating``, ``EMpy.RCWA.IsotropicRCWA``, and
``EMpy.utils.find_peaks`` on a silicon-nitride dimer waveguide metasurface:

1. When two identical Si3N4 bars of duty cycles dc1 = dc2 = f0 are spaced
   symmetrically inside a unit cell of pitch Lambda (spacer dcM = (1 - 2*f0)/2),
   the effective period halves to Lambda / 2. Odd-harmonic guided resonances
   are dark symmetry-protected bound states in the continuum (BICs) at normal
   incidence.
2. Perturbing the bar widths to dc1 = f0*(1 + alpha), dc2 = f0*(1 - alpha) via
   ``AsymmetricDoubleGrating`` breaks period-halving symmetry and opens a 100%
   reflection quasi-BIC Fano peak whose radiative quality factor scales
   inversely with asymmetry squared: Q ~ alpha^-2.

References
----------
- K. Koshelev, S. Lepeshov, M. Liu, A. Bogdanov, and Y. Kivshar, "Asymmetric
  metasurfaces with high-Q resonances governed by bound states in the
  continuum," *Phys. Rev. Lett.* **121**(19), 193903 (2018).
- C. W. Hsu, B. Zhen, A. D. Stone, J. D. Joannopoulos, and M. Soljačić, "Bound
  states in the continuum," *Nat. Rev. Mater.* **1**, 16048 (2016).
- M. G. Moharam, E. B. Grann, D. A. Pommet, and T. K. Gaylord, "Formulation for
  stable and efficient implementation of the rigorous coupled-wave analysis of
  binary gratings," *J. Opt. Soc. Am. A* **12**(5), 1068–1076 (1995).
"""

from matplotlib import pyplot as plt
import numpy as np

import EMpy


def main():
    pitch = 0.88e-6
    h_grating = 0.18e-6
    h_wg = 0.26e-6
    f0 = 0.22
    dc_spacer = 0.5 * (1.0 - 2.0 * f0)  # 0.28 -> symmetric half-period spacing

    air = EMpy.materials.Air
    sio2 = EMpy.materials.IsotropicMaterial(
        "SiO2", n0=EMpy.materials.RefractiveIndex(n0_const=1.45)
    )
    sin = EMpy.materials.IsotropicMaterial(
        "Si3N4", n0=EMpy.materials.RefractiveIndex(n0_const=2.05)
    )

    n_harm = 4
    alpha_inc = 0.0  # Normal incidence
    delta = 0.0
    psi = np.pi / 2.0  # TE polarization (E along y)
    phi = np.pi / 2.0

    # --- 1. Spectral comparison: SymmetricDoubleGrating vs. AsymmetricDoubleGrating ---
    wls = np.linspace(1.495e-6, 1.555e-6, 301)
    ml_sym = EMpy.utils.Multilayer(
        [
            EMpy.utils.Layer(air, np.inf),
            EMpy.utils.SymmetricDoubleGrating(sin, sin, air, f0, f0, pitch, h_grating),
            EMpy.utils.Layer(sin, h_wg),
            EMpy.utils.Layer(sio2, np.inf),
        ]
    )
    sol_sym = EMpy.RCWA.IsotropicRCWA(ml_sym, alpha_inc, delta, psi, phi, n_harm).solve(
        wls
    )

    # Verify exact equivalence of SymmetricDoubleGrating and AsymmetricDoubleGrating at alpha = 0
    ml_asym0 = EMpy.utils.Multilayer(
        [
            EMpy.utils.Layer(air, np.inf),
            EMpy.utils.AsymmetricDoubleGrating(
                sin, sin, air, f0, f0, dc_spacer, pitch, h_grating
            ),
            EMpy.utils.Layer(sin, h_wg),
            EMpy.utils.Layer(sio2, np.inf),
        ]
    )
    sol_asym0 = EMpy.RCWA.IsotropicRCWA(
        ml_asym0, alpha_inc, delta, psi, phi, n_harm
    ).solve(wls)
    sym_diff = float(np.max(np.abs(sol_sym.DE1 - sol_asym0.DE1)))

    # Sweep asymmetry parameter alpha = (dc1 - dc2) / (dc1 + dc2)
    asym_list = [0.16, 0.22, 0.30, 0.40, 0.50]
    spectra = {}
    q_factors = []
    res_wls_nm = []

    wls_fine = np.linspace(1.500e-6, 1.542e-6, 421)
    for a_val in asym_list:
        dc1 = f0 * (1.0 + a_val)
        dc2 = f0 * (1.0 - a_val)
        ml_a = EMpy.utils.Multilayer(
            [
                EMpy.utils.Layer(air, np.inf),
                EMpy.utils.AsymmetricDoubleGrating(
                    sin, sin, air, dc1, dc2, dc_spacer, pitch, h_grating
                ),
                EMpy.utils.Layer(sin, h_wg),
                EMpy.utils.Layer(sio2, np.inf),
            ]
        )
        sol_a = EMpy.RCWA.IsotropicRCWA(ml_a, alpha_inc, delta, psi, phi, n_harm).solve(
            wls_fine
        )
        r0_fine = sol_a.DE1[n_harm, :]

        if a_val in (0.16, 0.30, 0.50):
            sol_broad = EMpy.RCWA.IsotropicRCWA(
                ml_a, alpha_inc, delta, psi, phi, n_harm
            ).solve(wls)
            spectra[a_val] = sol_broad.DE1[n_harm, :]

        r_excess = np.maximum(r0_fine - np.min(r0_fine), 0.0)
        peaks = EMpy.utils.find_peaks(wls_fine * 1e9, r_excess, threshold=0.2)
        pk = peaks[0]
        q_factors.append(pk.x / pk.FWHM)
        res_wls_nm.append(pk.x)

    print(
        f"SymmetricDoubleGrating vs AsymmetricDoubleGrating(dcM=0.28) max diff = "
        f"{sym_diff:.2e}"
    )
    for a_val, wl_nm, q_val in zip(asym_list, res_wls_nm, q_factors):
        print(
            f"  asymmetry α = {a_val:.2f}: λ_res = {wl_nm:.3f} nm, Q = {q_val:.1f}, "
            f"Q * α² = {q_val * a_val**2:.2f}"
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4))

    ax1.plot(
        wls * 1e9,
        sol_sym.DE1[n_harm, :],
        "k--",
        linewidth=1.9,
        label="Symmetric dimer (α = 0, dark BIC)",
    )
    for a_val, trace in spectra.items():
        ax1.plot(
            wls * 1e9,
            trace,
            linewidth=1.6,
            label=f"Asymmetric dimer (α = {a_val:.2f})",
        )
    ax1.set_title("Si₃N₄ Dimer Grating: Symmetry-Protected BIC → Quasi-BIC")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Normal-Incidence Reflectance R₀")
    ax1.set_ylim(-0.02, 1.04)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8.5)

    asym_arr = np.array(asym_list)
    q_arr = np.array(q_factors)
    c_fit = float(q_arr[0] * asym_arr[0] ** 2)
    asym_dense = np.linspace(asym_arr[0] * 0.85, asym_arr[-1] * 1.15, 100)

    ax2.loglog(asym_arr, q_arr, "bo", markersize=6.5, label="RCWA + find_peaks Q")
    ax2.loglog(
        asym_dense,
        c_fit / asym_dense**2,
        "r--",
        linewidth=1.7,
        label=f"Koshelev Q = {c_fit:.1f} · α⁻² law",
    )
    ax2.set_title("Quasi-BIC Radiative Quality Factor Scaling")
    ax2.set_xlabel("Dimer Asymmetry Parameter α = (dc₁ − dc₂) / (dc₁ + dc₂)")
    ax2.set_ylabel("Quality Factor Q = λ_res / FWHM")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8.5)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
