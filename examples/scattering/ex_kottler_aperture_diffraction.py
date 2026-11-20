"""Vector Huygens–Kottler circular aperture diffraction vs. Airy pattern.

Demonstrates ``EMpy.scattering.currentsScatteringKottler``, ``Field``, and
``stack`` on 3D vector electromagnetic radiation from a uniformly illuminated
circular aperture of radius ``a = 2.5 * wl0``:

1. Compares full vector Kottler surface-current integration in the E-plane
   (phi = 0) and H-plane (phi = 90 deg) against the Fraunhofer Huygens–Airy
   analytical radiation pattern ``[(1 + cos(theta))/2 * 2*J1(k*a*sin(theta)) /
   (k*a*sin(theta))]^2``.
2. Evaluates on-axis Fresnel-to-Fraunhofer near-field propagation along +z in
   both lossless vacuum (``epsr = 1``) and a passive lossy dielectric medium
   (``epsr = (1.5 - 0.04j)**2``), comparing against the exact on-axis circular
   aperture formula ``|E(z)|^2 ~ |1 - exp(-i*k*(sqrt(z^2 + a^2) - z))|^2``.

References
----------
- F. Kottler, "Elektromagnetische Theorie der Beugung an schwarzen Schirmen,"
  *Ann. Phys.* **376**(15), 457–508 (1923).
- J. A. Stratton and L. J. Chu, "Diffraction theory of electromagnetic waves,"
  *Phys. Rev.* **56**(1), 99–107 (1939).
- C. A. Balanis, *Advanced Engineering Electromagnetics*, 2nd ed. (Wiley, 2012),
  Sec. 12.3.
"""

from matplotlib import pyplot as plt
import numpy as np
from scipy.special import j1

import EMpy


def build_circular_aperture(radius, n_grid):
    """Discretize a circular aperture in z = 0 using EMpy.scattering.stack."""
    u = np.linspace(-radius, radius, n_grid)
    du = float(u[1] - u[0])
    q_grid = EMpy.scattering.stack(u, u, np.array([0.0]))
    inside = q_grid[0, :] ** 2 + q_grid[1, :] ** 2 <= radius**2
    q_ap = q_grid[:, inside]
    ds_ap = np.full(q_ap.shape[1], du * du)
    return q_ap, ds_ap


def main():
    f0 = 30.0e9  # 30 GHz (wl0 = 10 mm)
    wl0 = EMpy.constants.c / f0
    k0 = 2.0 * np.pi / wl0
    a_rad = 2.5 * wl0

    q_ap, ds_ap = build_circular_aperture(a_rad, 36)
    n_q = q_ap.shape[1]

    # Equivalent Huygens currents for an x-polarized +z plane wave E0 = 1 V/m in free space
    e0 = 1.0
    j_free = np.vstack(
        [np.full(n_q, e0 / EMpy.constants.eta0), np.zeros(n_q), np.zeros(n_q)]
    )
    m_free = np.vstack([np.zeros(n_q), np.full(n_q, e0), np.zeros(n_q)])

    # --- 1. Far-field E-plane (x-z) and H-plane (y-z) radiation patterns ---
    r_ff = 250.0 * wl0
    thetas = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 121)

    p_eplane = np.vstack(
        [r_ff * np.sin(thetas), np.zeros_like(thetas), r_ff * np.cos(thetas)]
    )
    p_hplane = np.vstack(
        [np.zeros_like(thetas), r_ff * np.sin(thetas), r_ff * np.cos(thetas)]
    )

    fld_e = EMpy.scattering.currentsScatteringKottler(
        p_eplane, j_free, m_free, q_ap, ds_ap, f0, epsr=1.0
    )
    fld_h = EMpy.scattering.currentsScatteringKottler(
        p_hplane, j_free, m_free, q_ap, ds_ap, f0, epsr=1.0
    )

    pat_e = np.sum(np.abs(fld_e.E) ** 2, axis=0)
    pat_h = np.sum(np.abs(fld_h.E) ** 2, axis=0)
    peak_0 = pat_e[len(thetas) // 2]
    pat_e_norm = pat_e / peak_0
    pat_h_norm = pat_h / peak_0

    arg = k0 * a_rad * np.sin(thetas)
    arg_safe = np.where(np.abs(arg) < 1e-12, 1.0, arg)
    airy_core = np.where(np.abs(arg) < 1e-12, 1.0, 2.0 * j1(arg_safe) / arg_safe)
    obliquity = 0.5 * (1.0 + np.cos(thetas))
    pat_airy = (obliquity * airy_core) ** 2

    # --- 2. On-axis Fresnel near-field in vacuum vs. lossy dielectric ---
    z_axis = np.linspace(0.6 * wl0, 12.0 * wl0, 90)
    p_axis = np.vstack([np.zeros_like(z_axis), np.zeros_like(z_axis), z_axis])

    fld_ax_vac = EMpy.scattering.currentsScatteringKottler(
        p_axis, j_free, m_free, q_ap, ds_ap, f0, epsr=1.0
    )
    n_lossy = 1.4 - 0.03j
    epsr_lossy = n_lossy**2
    j_lossy = np.vstack(
        [
            np.full(n_q, e0 * n_lossy / EMpy.constants.eta0),
            np.zeros(n_q),
            np.zeros(n_q),
        ]
    )
    fld_ax_lossy = EMpy.scattering.currentsScatteringKottler(
        p_axis, j_lossy, m_free, q_ap, ds_ap, f0, epsr=epsr_lossy
    )

    # Exact Huygens-Rayleigh-Sommerfeld on-axis amplitude for a uniform circular aperture
    r_edge = np.sqrt(z_axis**2 + a_rad**2)
    e_exact_vac = (
        0.5
        * e0
        * (
            np.exp(-1j * k0 * z_axis)
            - 0.5 * (1.0 + z_axis / r_edge) * np.exp(-1j * k0 * r_edge) * 2.0
        )
    )

    mainlobe_mask = np.abs(thetas) <= EMpy.utils.deg2rad(35.0)
    ff_err = float(np.max(np.abs(pat_e_norm[mainlobe_mask] - pat_airy[mainlobe_mask])))
    print(
        f"Circular aperture (2a = {2 * a_rad / wl0:.1f} wl0, {n_q} patches): "
        f"mainlobe/sidelobe max |Kottler - Airy| = {ff_err:.3e}"
    )
    print(
        f"First Airy null angle = {EMpy.utils.rad2deg(np.arcsin(3.8317 / (k0 * a_rad))):.2f} deg; "
        f"on-axis ratio |E_lossy / E_vac| at z=10*wl0 = "
        f"{abs(fld_ax_lossy.E[0, -1]) / abs(fld_ax_vac.E[0, -1]):.3f}"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4))

    deg = EMpy.utils.rad2deg(thetas)
    ax1.plot(
        deg,
        10.0 * np.log10(np.maximum(pat_e_norm, 1e-5)),
        "b-",
        linewidth=1.8,
        label="Kottler E-plane (φ = 0°)",
    )
    ax1.plot(
        deg,
        10.0 * np.log10(np.maximum(pat_h_norm, 1e-5)),
        "r--",
        linewidth=1.6,
        label="Kottler H-plane (φ = 90°)",
    )
    ax1.plot(
        deg,
        10.0 * np.log10(np.maximum(pat_airy, 1e-5)),
        "k:",
        linewidth=1.6,
        label="Analytical Huygens–Airy",
    )
    ax1.set_title("Vector Kottler Circular Aperture Pattern (a = 2.5 λ₀)")
    ax1.set_xlabel("Observation Angle θ (deg)")
    ax1.set_ylabel("Normalized Directivity (dB)")
    ax1.set_ylim(-42, 2)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8.5)

    ax2.plot(
        z_axis / wl0,
        np.abs(fld_ax_vac.E[0, :]) ** 2,
        "b-",
        linewidth=1.8,
        label="Kottler |E_x(0,0,z)|² (vacuum)",
    )
    ax2.plot(
        z_axis / wl0,
        np.abs(e_exact_vac) ** 2,
        "k:",
        linewidth=1.6,
        label="Analytical Fresnel on-axis",
    )
    ax2.plot(
        z_axis / wl0,
        np.abs(fld_ax_lossy.E[0, :]) ** 2,
        "r--",
        linewidth=1.7,
        label="Kottler |E_x|² (lossy n = 1.4 − 0.03j)",
    )
    ax2.set_title("Fresnel Near-Field Axial Oscillations & Lossy Decay")
    ax2.set_xlabel("Axial Distance z / λ₀")
    ax2.set_ylabel("Electric Field Intensity |E_x|² (V²/m²)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8.5)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
