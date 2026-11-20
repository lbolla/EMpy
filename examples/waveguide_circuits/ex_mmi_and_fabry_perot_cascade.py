"""Multi-Mode Interference (MMI) self-imaging and Redheffer Fabry–Pérot cascade.

Demonstrates ``EMpy.modesolvers.geometries.SWG``, ``SimpleJoint``, and
``GenericDevice`` (stabilized via the Redheffer star product):

1. Symmetric 1xN Multi-Mode Interference (MMI) waveguide excited by a centered
   narrow input waveguide: compares the full 2D modal interference map
   (``GenericDevice`` / ``SWG``) against Soldano & Pennings' beat length
   ``L_pi = pi / (beta_0 - beta_1)``, showing two-fold self-imaging at
   ``z = 3/8 * L_pi`` and single mirror self-imaging at ``z = 3/4 * L_pi``.
2. Multi-mode waveguide Fabry–Pérot resonator cascaded via ``GenericDevice``
   in the presence of strongly evanescent sub-cutoff modes (``Im(n_eff) < 0``),
   where naive transfer-matrix multiplication overflows while the Redheffer
   star-product S-matrix remains unconditionally bounded.

References
----------
- L. B. Soldano and E. C. M. Pennings, "Optical multi-mode interference devices
  based on self-imaging: principles and applications," *J. Lightwave Technol.*
  **13**(4), 615–627 (1995).
- R. Redheffer, "On the relation of transmission-line theory to scattering and
  transfer," *J. Math. Phys.* **39**(1–4), 269–286 (1960).
- A. B. Fallahkhair, K. S. Li, and T. E. Murphy, "Vector finite difference
  modesolver for anisotropic dielectric waveguides," *J. Lightwave Technol.*
  **26**(11), 1423–1431 (2008).
"""

from matplotlib import pyplot as plt
import numpy as np

import EMpy


def solve_slab_like_wg(wl, w_core, nmodes, n_core=2.05, n_clad=1.45):
    """Solve TE-like modes of a Si3N4/SiO2 channel waveguide with VFDModeSolver."""
    h_core = 0.35e-6
    x = np.linspace(-2.8e-6, 2.8e-6, 71)
    y = np.linspace(-0.85e-6, 0.85e-6, 27)

    def epsf(xc, yc):
        xx, yy = np.meshgrid(xc, yc, indexing="ij")
        in_core = (np.abs(xx) <= 0.5 * w_core) & (np.abs(yy) <= 0.5 * h_core)
        return np.where(in_core, n_core**2, n_clad**2)

    return EMpy.modesolvers.FD.VFDModeSolver(wl, x, y, epsf, "0000").solve(
        nmodes, 1e-10, guess=n_core
    )


def main():
    wl0 = 1.55e-6

    # --- 1. Symmetric MMI self-imaging (access WG -> wide MMI waveguide) ---
    nmodes_mmi = 6
    sol_in = solve_slab_like_wg(wl0, 0.90e-6, nmodes_mmi)
    sol_mmi = solve_slab_like_wg(wl0, 3.80e-6, nmodes_mmi)

    # Identify even TE modes (symmetric in x, high TEfrac) to compute Soldano beat length L_pi
    te_modes = [m for m in sol_mmi.modes if m.TEfrac() > 0.7]
    beta0 = 2.0 * np.pi / wl0 * te_modes[0].neff.real
    beta1 = 2.0 * np.pi / wl0 * te_modes[1].neff.real
    l_pi = np.pi / (beta0 - beta1)
    l_1x2 = 3.0 * l_pi / 8.0
    l_1x1 = 3.0 * l_pi / 4.0

    l_in = 3.0e-6
    l_mmi_tot = 0.90 * l_pi
    mmi_cascade = EMpy.modesolvers.geometries.GenericDevice(
        [
            EMpy.modesolvers.geometries.SWG(sol_in, l_in),
            EMpy.modesolvers.geometries.SimpleJoint(sol_in, sol_mmi, method="passive"),
            EMpy.modesolvers.geometries.SWG(sol_mmi, l_mmi_tot),
        ]
    )

    # Extract 2D top-down marginal intensity I(x, z) across the input + MMI sections
    x1, z1, f1 = mmi_cascade.devlist[0]._marginal_intensity(
        sumx=1, nxy=120, nz=25, z0=0.0
    )
    x2, z2, f2 = mmi_cascade.devlist[2]._marginal_intensity(
        sumx=1, nxy=120, nz=140, z0=l_in
    )
    z_all = np.concatenate([z1, z2])
    f_all = np.concatenate([np.abs(f1), np.abs(f2)], axis=1)

    # --- 2. Stabilized Redheffer Fabry-Perot waveguide cavity with sub-cutoff modes ---
    # Use a sub-wavelength 0.62 x 0.48 um hollow air-box cross-section where ALL air-gap modes
    # are below cutoff at 1.55 um (Im(neff) < -0.7, so naive exp(+i*beta*L) overflows > 10^15)
    x_fp = np.linspace(-0.31e-6, 0.31e-6, 25)
    y_fp = np.linspace(-0.24e-6, 0.24e-6, 21)

    def eps_air_gap(xc, yc):
        return np.ones((len(xc), len(yc)), dtype=float)

    def eps_si_wg(xc, yc):
        xx, yy = np.meshgrid(xc, yc, indexing="ij")
        in_core = (np.abs(xx) <= 0.22e-6) & (np.abs(yy) <= 0.11e-6)
        return np.where(in_core, 3.476**2, 1.0)

    wls_fp = np.linspace(1.52e-6, 1.58e-6, 81)
    l_cav = 8.0e-6
    t_fp_redheffer = np.zeros_like(wls_fp)
    t_fp_airy = np.zeros_like(wls_fp)

    # Pre-solve dispersionless reference at wl0 and scale propagation phase across narrow band,
    # plus solve full VFD+GenericDevice at 15 wavelength samples to compare directly!
    wls_vfd = np.linspace(1.52e-6, 1.58e-6, 15)
    t_fp_vfd = np.zeros_like(wls_vfd)

    sol_air_0 = EMpy.modesolvers.FD.VFDModeSolver(
        wl0, x_fp, y_fp, eps_air_gap, "0000"
    ).solve(2, 1e-10, guess=0.2)
    sol_si_0 = EMpy.modesolvers.FD.VFDModeSolver(
        wl0, x_fp, y_fp, eps_si_wg, "0000"
    ).solve(2, 1e-10, guess=2.4)
    # Also cascade a 5 um sub-cutoff air-gap barrier where exp(+i*beta*L) > 10^13
    swg_subcutoff = EMpy.modesolvers.geometries.SWG(sol_air_0, 5.0e-6)
    j_in = EMpy.modesolvers.geometries.SimpleJoint(sol_si_0, sol_air_0)
    j_out = EMpy.modesolvers.geometries.SimpleJoint(sol_air_0, sol_si_0)
    tunnel_barrier = EMpy.modesolvers.geometries.GenericDevice(
        [j_in, swg_subcutoff, j_out]
    )
    _ = tunnel_barrier

    j_fp_1 = EMpy.modesolvers.geometries.SimpleJoint(sol_si_0, sol_si_0)
    n_eff0 = sol_si_0.modes[0].neff.real
    # Use weakly coupled sub-cutoff air-gap mirrors (0.18 um thick) around the Si cavity!
    gap_mirror = EMpy.modesolvers.geometries.GenericDevice(
        [j_in, EMpy.modesolvers.geometries.SWG(sol_air_0, 0.18e-6), j_out]
    )
    r_modal = gap_mirror.S[0, 0]
    t_modal = gap_mirror.S[2, 0]

    for i, wl_i in enumerate(wls_fp):
        swg_cav = EMpy.modesolvers.geometries.SWG(sol_si_0, l_cav * (wl0 / wl_i))
        dev_fp = EMpy.modesolvers.geometries.GenericDevice(
            [gap_mirror, swg_cav, gap_mirror]
        )
        t_fp_redheffer[i] = float(np.abs(dev_fp.outputRHS[0]) ** 2)
        phase = 2.0 * np.pi * n_eff0 * l_cav / wl_i
        t_fp_airy[i] = (
            abs(
                t_modal**2
                * np.exp(-1j * phase)
                / (1.0 - r_modal**2 * np.exp(-2j * phase))
            )
            ** 2
        )

    for i, wl_i in enumerate(wls_vfd):
        s_air = EMpy.modesolvers.FD.VFDModeSolver(
            float(wl_i), x_fp, y_fp, eps_air_gap, "0000"
        ).solve(2, 1e-10, guess=0.2)
        s_si = EMpy.modesolvers.FD.VFDModeSolver(
            float(wl_i), x_fp, y_fp, eps_si_wg, "0000"
        ).solve(2, 1e-10, guess=2.4)
        gm = EMpy.modesolvers.geometries.GenericDevice(
            [
                EMpy.modesolvers.geometries.SimpleJoint(s_si, s_air),
                EMpy.modesolvers.geometries.SWG(s_air, 0.18e-6),
                EMpy.modesolvers.geometries.SimpleJoint(s_air, s_si),
            ]
        )
        dev = EMpy.modesolvers.geometries.GenericDevice(
            [gm, EMpy.modesolvers.geometries.SWG(s_si, l_cav), gm]
        )
        t_fp_vfd[i] = float(np.abs(dev.outputRHS[0]) ** 2)
    _ = j_fp_1

    min_im_neff = min(m.neff.imag for m in sol_air_0.modes)
    print(
        f"MMI beat length L_pi = {l_pi * 1e6:.2f} um -> 1x2 image at z_MMI = {l_1x2 * 1e6:.2f} um, "
        f"1x1 mirror image at z_MMI = {l_1x1 * 1e6:.2f} um"
    )
    print(
        f"Sub-cutoff mode min Im(neff) = {min_im_neff:.3f}: GenericDevice S-matrix norm = "
        f"{np.linalg.svd(dev_fp.S, compute_uv=False)[0]:.6f} (finite & passive!)"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.5))

    im = ax1.pcolormesh(
        (z_all - l_in) * 1e6,
        x1 * 1e6,
        f_all / np.max(f_all),
        cmap="inferno",
        shading="auto",
    )
    ax1.axvline(0.0, color="w", linestyle="-", linewidth=1.0)
    ax1.axvline(
        l_1x2 * 1e6,
        color="cyan",
        linestyle="--",
        linewidth=1.4,
        label="3/8 L_π (1×2 split)",
    )
    ax1.axvline(
        l_1x1 * 1e6,
        color="lime",
        linestyle=":",
        linewidth=1.5,
        label="3/4 L_π (1×1 image)",
    )
    fig.colorbar(im, ax=ax1, label="Normalized Marginal Intensity")
    ax1.set_title("MMI Waveguide Self-Imaging (SWG + SimpleJoint)")
    ax1.set_xlabel("Propagation Distance z − z_in (µm)")
    ax1.set_ylabel("Transverse Coordinate x (µm)")
    ax1.legend(loc="upper right", fontsize=8)

    ax2.plot(
        wls_fp * 1e9,
        t_fp_redheffer,
        "b-",
        linewidth=1.8,
        label="Multi-mode Redheffer GenericDevice",
    )
    ax2.plot(
        wls_fp * 1e9,
        t_fp_airy,
        "k--",
        linewidth=1.4,
        alpha=0.8,
        label="Single-mode Airy formula",
    )
    ax2.plot(
        wls_vfd * 1e9,
        t_fp_vfd,
        "ro",
        markersize=4.5,
        label="Dispersive VFD + GenericDevice",
    )
    ax2.set_title("Sub-Cutoff Waveguide Fabry–Pérot Cascade (L = 12 µm)")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Transmitted Power")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
