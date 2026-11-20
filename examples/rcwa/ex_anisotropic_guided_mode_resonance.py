"""Anisotropic guided-mode resonance filter vs. Rytov effective-medium theory.

Demonstrates ``EMpy.RCWA.AnisotropicRCWA`` on a subwavelength 3D Euler-rotated
birefringent waveguide grating and compares the full-wave diffraction spectrum
against Rytov's zero-order effective-medium theory (EMT) evaluated via
``EMpy.transfer_matrix.AnisotropicTransferMatrix``:

- Away from guided-mode phase matching, the subwavelength grating behaves as a
  homogeneous biaxial thin film whose R/T baseline matches Rytov EMT
  (eps_xx,eff = (f / eps_1 + (1 - f) / eps_2)^-1, eps_yy,zz,eff = f*eps_1 + (1-f)*eps_2).
- When +/-1 evanescent grating harmonics phase-match to leaky waveguide modes of
  the high-index SiN core, ``AnisotropicRCWA`` captures sharp 100% guided-mode
  resonance (GMR) reflection peaks and cross-polarized conversion missed by EMT.

References
----------
- S. S. Wang and R. Magnusson, "Theory and applications of guided-mode resonance
  filters," *Appl. Opt.* **32**(14), 2606–2613 (1993).
- E. N. Glytsis and T. K. Gaylord, "Three-dimensional (vector) rigorous
  coupled-wave analysis of anisotropic grating diffraction," *J. Opt. Soc. Am. A*
  **7**(8), 1399–1419 (1990).
- L. Li, "Reformulation of the Fourier modal method for surface-relief gratings
  made with anisotropic materials," *J. Mod. Opt.* **45**(7), 1313–1334 (1998).
- S. M. Rytov, "Electromagnetic properties of a finely stratified medium,"
  *Sov. Phys. JETP* **2**(3), 466–475 (1956).
"""

from matplotlib import pyplot as plt
import numpy as np

import EMpy
from EMpy.constants import eps0


def rytov_tensor(eps1, eps2, f):
    """Zero-order Rytov effective permittivity tensor for a 1D x-periodic grating.

    Enforces continuity of (D_x, E_y, E_z) across x-normal grating interfaces
    for general 3x3 permittivity tensors using Schur-complement averaging.
    """

    # Map (E_x, E_y, E_z) -> (D_x, E_y, E_z) via G(eps), average <G>, and invert
    def to_hybrid(eps):
        exx = eps[0, 0]
        g = np.zeros((3, 3), dtype=complex)
        g[0, 0] = 1.0 / exx
        g[0, 1:] = -eps[0, 1:] / exx
        g[1:, 0] = eps[1:, 0] / exx
        g[1:, 1:] = eps[1:, 1:] - np.outer(eps[1:, 0], eps[0, 1:]) / exx
        return g

    def from_hybrid(g):
        g00 = g[0, 0]
        eps = np.zeros((3, 3), dtype=complex)
        eps[0, 0] = 1.0 / g00
        eps[0, 1:] = -g[0, 1:] / g00
        eps[1:, 0] = g[1:, 0] / g00
        eps[1:, 1:] = g[1:, 1:] - np.outer(g[1:, 0], g[0, 1:]) / g00
        return eps

    g_avg = f * to_hybrid(eps1) + (1.0 - f) * to_hybrid(eps2)
    return from_hybrid(g_avg)


def main():
    wls = np.linspace(1.46e-6, 1.62e-6, 161)
    pitch = 0.92e-6
    dc = 0.50
    h_grating = 0.16e-6
    h_wg = 0.28e-6

    alpha = EMpy.utils.deg2rad(4.0)
    delta = 0.0
    psi = np.pi / 2.0  # TE incident wave (E along y)
    phi = np.pi / 2.0
    n_harm = 5

    n_cladding = 1.45
    n_wg_o, n_wg_e = 2.02, 2.14

    eps_clad = eps0 * n_cladding**2 * np.eye(3)
    # Rotate birefringent waveguide/grating optic axis in 3D so O and E modes hybridize
    eps_core = eps0 * EMpy.utils.euler_rotate(
        np.diag([n_wg_o**2, n_wg_e**2, n_wg_o**2]),
        EMpy.utils.deg2rad(28.0),
        EMpy.utils.deg2rad(20.0),
        0.0,
    )

    clad_mat = EMpy.materials.AnisotropicMaterial(
        "SiO2_clad", EMpy.materials.EpsilonTensor(eps_clad)
    )
    core_mat = EMpy.materials.AnisotropicMaterial(
        "Biref_SiN", EMpy.materials.EpsilonTensor(eps_core)
    )

    ml_rcwa = EMpy.utils.Multilayer(
        [
            EMpy.utils.Layer(clad_mat, np.inf),
            EMpy.utils.BinaryGrating(core_mat, clad_mat, dc, pitch, h_grating),
            EMpy.utils.Layer(core_mat, h_wg),
            EMpy.utils.Layer(clad_mat, np.inf),
        ]
    )
    sol_rcwa = EMpy.RCWA.AnisotropicRCWA(ml_rcwa, alpha, delta, psi, phi, n_harm).solve(
        wls
    )

    # Zero-order Rytov effective-medium multilayer solved with AnisotropicTransferMatrix
    eps_emt = rytov_tensor(eps_core, eps_clad, dc)
    emt_mat = EMpy.materials.AnisotropicMaterial(
        "Rytov_EMT", EMpy.materials.EpsilonTensor(eps_emt)
    )
    ml_emt = EMpy.utils.Multilayer(
        [
            EMpy.utils.Layer(clad_mat, np.inf),
            EMpy.utils.Layer(emt_mat, h_grating),
            EMpy.utils.Layer(core_mat, h_wg),
            EMpy.utils.Layer(clad_mat, np.inf),
        ]
    )
    sol_emt = EMpy.transfer_matrix.AnisotropicTransferMatrix(ml_emt, alpha, 0.0).solve(
        wls
    )

    # Energy conservation check (subwavelength: pitch*n_clad = 1.334 um < 1.46 um at normal,
    # check all retained orders sum(DE1 + DE3) = 1)
    energy_err = float(
        np.max(np.abs(np.sum(sol_rcwa.DE1 + sol_rcwa.DE3, axis=0) - 1.0))
    )
    peaks = EMpy.utils.find_peaks(wls * 1e9, sol_rcwa.DE1[n_harm, :], threshold=0.2)

    print(f"AnisotropicRCWA max |sum(DE1 + DE3) - 1| = {energy_err:.2e}")
    for idx, pk in enumerate(peaks[:3]):
        print(
            f"  GMR peak {idx + 1}: wl = {pk.x:.3f} nm, R_0 = {pk.y:.4f}, "
            f"FWHM = {pk.FWHM:.3f} nm"
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4))

    ax1.plot(
        wls * 1e9,
        sol_rcwa.DE1[n_harm, :],
        "b-",
        linewidth=1.8,
        label="AnisotropicRCWA R₀ (total)",
    )
    ax1.plot(
        wls * 1e9,
        sol_rcwa.DE3[n_harm, :],
        "r-",
        linewidth=1.5,
        alpha=0.8,
        label="AnisotropicRCWA T₀ (total)",
    )
    ax1.plot(
        wls * 1e9,
        sol_emt.R_tot[0, :],
        "k--",
        linewidth=1.8,
        label="Rytov EMT (4×4 TMM R_s)",
    )
    ax1.set_title("Subwavelength Birefringent GMR vs. Rytov EMT")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Zero-Order Power")
    ax1.set_ylim(-0.02, 1.04)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8.5)

    # Polarization decomposition into Ordinary (s-like in isotropic cladding) and Extraordinary (p-like)
    ax2.plot(
        wls * 1e9,
        sol_rcwa.DEO1[n_harm, :],
        "b-",
        linewidth=1.7,
        label="RCWA R₀,ss (co-pol DEO1)",
    )
    ax2.plot(
        wls * 1e9,
        sol_rcwa.DEE1[n_harm, :],
        "m-",
        linewidth=1.7,
        label="RCWA R₀,ps (cross-pol DEE1)",
    )
    ax2.plot(
        wls * 1e9,
        sol_emt.R[1, 0, :],
        "k--",
        linewidth=1.5,
        label="Rytov EMT R_ps baseline",
    )
    ax2.set_title("Resonant Polarization Conversion from 3D Optic Axis")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Polarization-Resolved Reflectance")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8.5)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
