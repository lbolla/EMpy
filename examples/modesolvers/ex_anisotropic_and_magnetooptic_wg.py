"""Anisotropic liquid-crystal-clad and magneto-optic gyrotropic waveguides.

Demonstrates the full 5-component ``(nx, ny, 5)`` permittivity tensor capability
``[epsxx, epsxy, epsyx, epsyy, epszz]`` of ``EMpy.modesolvers.FD.VFDModeSolver``:

1. Birefringent nematic-liquid-crystal-clad silicon waveguide with an in-plane
   rotated optic axis (symmetric real off-diagonal ``epsxy == epsyx``), showing
   polarization rotation and continuous convergence to scalar isotropic
   ``VFDModeSolver`` as birefringence ``delta_n -> 0``.
2. Non-reciprocal magneto-optic Ce:TbIG garnet-clad silicon waveguide with
   anti-symmetric imaginary off-diagonal gyrotropy ``epsxy = -epsyx = -1j * g``
   (Faraday / longitudinal magnetization geometry), inducing elliptical spin
   polarization quantified by the normalized Stokes parameter
   ``s3 = 2 * Im(Ex * conj(Ey)) / (|Ex|^2 + |Ey|^2)``.

Adapted in part from the anisotropic Stokes analysis contributed by David
Hutchings.

References
----------
- A. B. Fallahkhair, K. S. Li, and T. E. Murphy, "Vector finite difference
  modesolver for anisotropic dielectric waveguides," *J. Lightwave Technol.*
  **26**(11), 1423–1431 (2008).
- L. Bi, J. Hu, P. Jiang, D. H. Kim, G. F. Dionne, L. C. Kimerling, and
  C. A. Ross, "On-chip optical isolation in monolithically integrated
  non-reciprocal optical resonators," *Nat. Photonics* **5**(12), 758–762
  (2011).
- S. Ghosh, S. Keyvaninia, Y. Shoji, W. Van Roy, T. Mizumoto, G. Roelkens, and
  R. Baets, "Compact Mach–Zehnder interferometer Ce:YIG/SOI optical isolators,"
  *IEEE Photon. Technol. Lett.* **24**(18), 1653–1656 (2012).
"""

from matplotlib import pyplot as plt
import numpy as np

import EMpy


def build_waveguide_eps(xc, yc, w_core, h_core, y_box, eps_clad_5, n_core, n_box):
    """Return (nx, ny, 5) relative permittivity for a core on SiO2 with anisotropic top cladding."""
    xx, yy = np.meshgrid(xc, yc, indexing="ij")
    out = np.zeros((len(xc), len(yc), 5), dtype=complex)

    in_box = yy < y_box
    in_core = (yy >= y_box) & (yy <= y_box + h_core) & (np.abs(xx) <= 0.5 * w_core)
    in_clad = ~(in_box | in_core)

    for idx_diag in (0, 3, 4):
        out[in_box, idx_diag] = n_box**2
        out[in_core, idx_diag] = n_core**2
    for c_idx in range(5):
        out[in_clad, c_idx] = eps_clad_5[c_idx]
    return out


def main():
    wl = 1.55e-6
    w_core = 0.42e-6
    h_core = 0.40e-6  # Near-square core enhances TE/TM hybridization sensitivity
    y_box = 0.70e-6
    n_si = 3.476
    n_sio2 = 1.444

    x = np.linspace(-1.1e-6, 1.1e-6, 55)
    y = np.linspace(0.0, 1.8e-6, 55)

    # --- 1. Birefringent LC top cladding rotated by 35 deg in x-y: dn -> 0 limit ---
    n_o = 1.52
    phi_lc = EMpy.utils.deg2rad(35.0)

    def eps_iso_scalar(xc, yc):
        xx, yy = np.meshgrid(xc, yc, indexing="ij")
        eps = np.full((len(xc), len(yc)), n_o**2, dtype=float)
        eps[yy < y_box] = n_sio2**2
        in_core = (yy >= y_box) & (yy <= y_box + h_core) & (np.abs(xx) <= 0.5 * w_core)
        eps[in_core] = n_si**2
        return eps

    sol_iso_ref = EMpy.modesolvers.FD.VFDModeSolver(
        wl, x, y, eps_iso_scalar, "0000"
    ).solve(2, 1e-11)
    neff_iso_ref = np.array([m.neff.real for m in sol_iso_ref.modes])

    dn_vals = [0.20, 0.05, 1e-2, 1e-4, 1e-7, 0.0]
    dn_errs = []
    for dn in dn_vals:
        n_e = n_o + dn
        eps_3 = EMpy.utils.euler_rotate(
            np.diag([n_e**2, n_o**2, n_o**2]), phi_lc, 0.0, 0.0
        )
        eps_5 = [eps_3[0, 0], eps_3[0, 1], eps_3[1, 0], eps_3[1, 1], eps_3[2, 2]]
        sol_a = EMpy.modesolvers.FD.VFDModeSolver(
            wl,
            x,
            y,
            lambda xc, yc, e5=eps_5: build_waveguide_eps(
                xc, yc, w_core, h_core, y_box, e5, n_si, n_sio2
            ),
            "0000",
        ).solve(2, 1e-11)
        neff_a = np.array([m.neff.real for m in sol_a.modes])
        dn_errs.append(float(np.max(np.abs(neff_a - neff_iso_ref))))

    # --- 2. Magneto-optic Ce:TbIG garnet cladding (epsxy = -epsyx = -1j * g) ---
    n_garnet = 2.22
    g_faraday = 0.035  # Enhanced magneto-optic gyrotropy for visualization
    eps_mo_5 = [
        n_garnet**2,
        -1j * g_faraday,
        +1j * g_faraday,
        n_garnet**2,
        n_garnet**2,
    ]
    sol_mo_pos = EMpy.modesolvers.FD.VFDModeSolver(
        wl,
        x,
        y,
        lambda xc, yc: build_waveguide_eps(
            xc, yc, w_core, h_core, y_box, eps_mo_5, n_si, n_sio2
        ),
        "0000",
    ).solve(2, 1e-11)

    # Reversing magnetization (+M -> -M) conjugates epsxy <-> epsyx and flips Stokes s3 -> -s3
    eps_mo_neg = [
        n_garnet**2,
        +1j * g_faraday,
        -1j * g_faraday,
        n_garnet**2,
        n_garnet**2,
    ]
    sol_mo_neg = EMpy.modesolvers.FD.VFDModeSolver(
        wl,
        x,
        y,
        lambda xc, yc: build_waveguide_eps(
            xc, yc, w_core, h_core, y_box, eps_mo_neg, n_si, n_sio2
        ),
        "0000",
    ).solve(2, 1e-11)

    m0_pos = sol_mo_pos.modes[0]
    m0_neg = sol_mo_neg.modes[0]
    xc = EMpy.utils.centered1d(x)
    yc = EMpy.utils.centered1d(y)

    s0_pos = np.abs(m0_pos.Ex) ** 2 + np.abs(m0_pos.Ey) ** 2
    s3_pos = 2.0 * np.imag(m0_pos.Ex * np.conj(m0_pos.Ey))
    s3_neg = 2.0 * np.imag(m0_neg.Ex * np.conj(m0_neg.Ey))
    s3_avg_pos = float(np.sum(s3_pos) / np.sum(s0_pos))
    s3_avg_neg = float(
        np.sum(s3_neg) / (np.sum(np.abs(m0_neg.Ex) ** 2 + np.abs(m0_neg.Ey) ** 2))
    )

    print("LC-clad anisotropic VFD -> scalar isotropic VFD (dn -> 0):")
    for dn, err in zip(dn_vals, dn_errs):
        print(f"  dn = {dn:<8g} -> max |neff_aniso - neff_iso| = {err:.3e}")
    print(
        f"Magneto-optic Ce:TbIG mode 0: neff = {m0_pos.neff.real:.6f}, "
        f"<S3/S0>(+M) = {s3_avg_pos:+.4f}, <S3/S0>(-M) = {s3_avg_neg:+.4f}"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4))

    dn_plot = np.where(np.array(dn_vals) == 0.0, 1e-15, np.array(dn_vals))
    ax1.loglog(
        dn_plot,
        np.maximum(dn_errs, 1e-15),
        "bo-",
        linewidth=1.8,
        markersize=5.5,
        label="5-comp VFD vs. scalar VFD",
    )
    ax1.set_title("Anisotropic VFD → Isotropic Limit (Δn → 0)")
    ax1.set_xlabel("LC Cladding Birefringence Δn (0 plotted at 10⁻¹⁵)")
    ax1.set_ylabel("max |n_eff,aniso − n_eff,iso|")
    ax1.invert_xaxis()
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8.5)

    # Normalized circular polarization spin density s3 = S3 / S0 where S0 > 1% peak
    mask = s0_pos > 0.01 * np.max(s0_pos)
    s3_norm = np.where(mask, s3_pos / np.maximum(s0_pos, 1e-30), 0.0)
    im = ax2.pcolormesh(
        xc * 1e6,
        yc * 1e6,
        s3_norm.T,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        shading="auto",
    )
    ax2.axhline(y_box * 1e6, color="k", linestyle="--", linewidth=1.0)
    ax2.plot(
        [
            -0.5 * w_core * 1e6,
            0.5 * w_core * 1e6,
            0.5 * w_core * 1e6,
            -0.5 * w_core * 1e6,
            -0.5 * w_core * 1e6,
        ],
        [
            y_box * 1e6,
            y_box * 1e6,
            (y_box + h_core) * 1e6,
            (y_box + h_core) * 1e6,
            y_box * 1e6,
        ],
        "k-",
        linewidth=1.6,
    )
    fig.colorbar(im, ax=ax2, label="Normalized Stokes S₃ / S₀")
    ax2.set_aspect("equal")
    ax2.set_title("Ce:TbIG Gyrotropic Waveguide Spin Density S₃/S₀")
    ax2.set_xlabel("x (µm)")
    ax2.set_ylabel("y (µm)")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
