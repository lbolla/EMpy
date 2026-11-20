"""SOI channel waveguide: SVFD vs. VFD vs. FMM2d and CrossSection meshing.

Demonstrates ``EMpy.modesolvers.FD.SVFDModeSolver`` (``scalar``, ``Ex``, ``Ey``),
``EMpy.modesolvers.FD.VFDModeSolver``, ``EMpy.modesolvers.FMM.FMM2d``, and
piecewise-uniform ``EMpy.utils.CrossSection`` / ``Slice`` grid generation on a
high-index-contrast Silicon-on-Insulator (SOI) strip waveguide at 1.55 um:

1. Compares fundamental quasi-TE0 and quasi-TM0 effective indices, TE
   polarization fraction ``TEfrac()``, and direct cross-solver Poynting mode
   overlap ``FDMode.overlap(FMMMode2d)`` on a 450 nm x 220 nm SOI core.
2. Sweeps core width w in [280, 700] nm, showing strong agreement between full
   vectorial solvers (``VFDModeSolver`` and ``FMM2d``) and quantifying the
   breakdown of scalar and semi-vectorial approximations at submicron widths.

References
----------
- A. B. Fallahkhair, K. S. Li, and T. E. Murphy, "Vector finite difference
  modesolver for anisotropic dielectric waveguides," *J. Lightwave Technol.*
  **26**(11), 1423–1431 (2008).
- A. S. Sudbø, "Film mode matching: a versatile numerical method for vector mode
  field calculations in dielectric waveguides," *Pure Appl. Opt.* **2**(3),
  211–233 (1993).
- M. S. Stern, "Semivectorial polarized finite difference method for optical
  waveguides with arbitrary index profiles," *IEE Proc. J* **135**(1), 56–63
  (1988).
"""

from matplotlib import pyplot as plt
import numpy as np

import EMpy


def solve_soi_width(wl, w_core, h_core, n_si, n_sio2, run_fmm=True):
    """Solve TE0 and TM0 modes for a given SOI core width across FD (and optionally FMM2d)."""
    clad_x = 0.85e-6
    clad_y = 0.75e-6

    si_mat = EMpy.materials.IsotropicMaterial(
        "Si", n0=EMpy.materials.RefractiveIndex(n0_const=n_si)
    )
    ox_mat = EMpy.materials.IsotropicMaterial(
        "SiO2", n0=EMpy.materials.RefractiveIndex(n0_const=n_sio2)
    )

    s_side = EMpy.utils.Slice(
        clad_x,
        [EMpy.utils.Layer(ox_mat, clad_y + h_core + clad_y)],
    )
    s_core = EMpy.utils.Slice(
        w_core,
        [
            EMpy.utils.Layer(ox_mat, clad_y),
            EMpy.utils.Layer(si_mat, h_core),
            EMpy.utils.Layer(ox_mat, clad_y),
        ],
    )
    cs = EMpy.utils.CrossSection([s_side, s_core, s_side])
    x, y = cs.grid((14, 22, 14), (14, 16, 14))

    def epsf(xc, yc):
        return cs.epsfunc(xc, yc, wl)

    sv_scalar = EMpy.modesolvers.FD.SVFDModeSolver(
        wl, x, y, epsf, "0000", method="scalar"
    ).solve(2, 1e-10)
    sv_ex = EMpy.modesolvers.FD.SVFDModeSolver(
        wl, x, y, epsf, "0000", method="Ex"
    ).solve(2, 1e-10)
    sv_ey = EMpy.modesolvers.FD.SVFDModeSolver(
        wl, x, y, epsf, "0000", method="Ey"
    ).solve(2, 1e-10)
    vfd = EMpy.modesolvers.FD.VFDModeSolver(wl, x, y, epsf, "0000").solve(2, 1e-10)

    fmm = None
    if run_fmm:
        ux = np.array([0.0, clad_x, clad_x + w_core, 2.0 * clad_x + w_core])
        uy = np.array([0.0, clad_y, clad_y + h_core, 2.0 * clad_y + h_core])
        rix = np.array(
            [
                [n_sio2, n_sio2, n_sio2],
                [n_sio2, n_si, n_sio2],
                [n_sio2, n_sio2, n_sio2],
            ]
        )
        bc = EMpy.modesolvers.FMM.Boundary(
            xleft="Electric Wall",
            xright="Electric Wall",
            yleft="Magnetic Wall",
            yright="Magnetic Wall",
        )
        fmm = EMpy.modesolvers.FMM.FMM2d(ux, uy, rix, bc).solve(wl, 5, 2)
        for m in fmm.modes:
            m.normalize()

    return cs, vfd, sv_scalar, sv_ex, sv_ey, fmm


def main():
    wl = 1.55e-6
    h_core = 0.22e-6
    w0 = 0.45e-6
    n_si = 3.476
    n_sio2 = 1.444

    cs0, vfd0, sv_sc0, sv_ex0, sv_ey0, fmm0 = solve_soi_width(
        wl, w0, h_core, n_si, n_sio2, run_fmm=True
    )
    assert fmm0 is not None

    m_vfd_te = vfd0.modes[0]
    m_vfd_tm = vfd0.modes[1]
    m_fmm_te = fmm0.modes[0]
    m_fmm_tm = fmm0.modes[1]
    ov_te = abs(m_vfd_te.overlap(m_fmm_te))
    ov_tm = abs(m_vfd_tm.overlap(m_fmm_tm))

    print(f"450 nm x 220 nm SOI strip waveguide at {wl * 1e6:.2f} um:")
    print(
        f"  TE0: VFD neff = {m_vfd_te.neff.real:.5f} (TEfrac={m_vfd_te.TEfrac():.3f}), "
        f"FMM2d neff = {m_fmm_te.neff.real:.5f} (TEfrac={m_fmm_te.TEfrac():.3f}), "
        f"|overlap| = {ov_te:.5f}"
    )
    print(
        f"  TM0: VFD neff = {m_vfd_tm.neff.real:.5f} (TEfrac={m_vfd_tm.TEfrac():.3f}), "
        f"FMM2d neff = {m_fmm_tm.neff.real:.5f} (TEfrac={m_fmm_tm.TEfrac():.3f}), "
        f"|overlap| = {ov_tm:.5f}"
    )
    print(
        f"  Semi-vectorial SVFD: Ex(TE0) = {sv_ex0.neff[0].real:.5f}, "
        f"Ey(TM0) = {sv_ey0.neff[0].real:.5f}, Scalar = {sv_sc0.neff[0].real:.5f}"
    )

    # Core-width sweep comparing scalar, semi-vectorial, VFD, and FMM2d
    widths = np.linspace(0.30e-6, 0.60e-6, 7)
    fmm_sample_idx = {0, 3, 6}  # 300 nm, 450 nm (reuse fmm0), 600 nm
    widths_fmm = widths[sorted(fmm_sample_idx)]
    neff_sc = np.zeros_like(widths)
    neff_sv_ex = np.zeros_like(widths)
    neff_sv_ey = np.zeros_like(widths)
    neff_vfd_te = np.zeros_like(widths)
    neff_vfd_tm = np.zeros_like(widths)
    neff_fmm_te = []
    neff_fmm_tm = []

    for i, w in enumerate(widths):
        need_fmm = (i in fmm_sample_idx) and not np.isclose(w, w0)
        _, vfd, sv_sc, sv_ex, sv_ey, fmm = solve_soi_width(
            wl, float(w), h_core, n_si, n_sio2, run_fmm=need_fmm
        )
        neff_sc[i] = sv_sc.neff[0].real
        neff_sv_ex[i] = sv_ex.neff[0].real
        neff_sv_ey[i] = sv_ey.neff[0].real
        v_modes = sorted(vfd.modes, key=lambda m: m.TEfrac(), reverse=True)
        neff_vfd_te[i] = v_modes[0].neff.real
        neff_vfd_tm[i] = v_modes[-1].neff.real
        if i in fmm_sample_idx:
            fmm_use = fmm0 if np.isclose(w, w0) else fmm
            assert fmm_use is not None
            f_modes = sorted(fmm_use.modes, key=lambda m: m.TEfrac(), reverse=True)
            neff_fmm_te.append(f_modes[0].neff.real)
            neff_fmm_tm.append(f_modes[-1].neff.real)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.5))

    x_eval = np.linspace(0.45e-6, 1.70e-6, 110)
    y_eval = np.linspace(0.35e-6, 1.37e-6, 100)
    i_vfd = np.abs(m_vfd_te.intensity(x_eval, y_eval)).T
    i_fmm = np.abs(m_fmm_te.intensity(x_eval, y_eval))
    i_vfd /= np.max(i_vfd)
    i_fmm /= np.max(i_fmm)

    levels = np.linspace(0.1, 0.9, 5)
    c1 = ax1.contour(
        x_eval * 1e6, y_eval * 1e6, i_vfd, levels=levels, colors="tab:blue"
    )
    c2 = ax1.contour(
        x_eval * 1e6,
        y_eval * 1e6,
        i_fmm,
        levels=levels,
        colors="tab:red",
        linestyles="dashed",
    )
    x0_box, y0_box = 0.85, 0.75
    ax1.plot(
        [x0_box, x0_box + w0 * 1e6, x0_box + w0 * 1e6, x0_box, x0_box],
        [y0_box, y0_box, y0_box + h_core * 1e6, y0_box + h_core * 1e6, y0_box],
        "k-",
        linewidth=1.6,
    )
    ax1.plot([], [], color="tab:blue", label="VFDModeSolver Sz")
    ax1.plot([], [], color="tab:red", linestyle="--", label="FMM2d Sz")
    ax1.clabel(c1, fmt="%.1f", fontsize=7.5)
    ax1.set_aspect("equal")
    ax1.set_title("450×220 nm SOI TE₀ Intensity Contours")
    ax1.set_xlabel("x (µm)")
    ax1.set_ylabel("y (µm)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right", fontsize=8.5)

    ax2.plot(widths * 1e9, neff_sc, "k:", linewidth=1.7, label="SVFD (scalar)")
    ax2.plot(widths * 1e9, neff_sv_ex, "b--", linewidth=1.5, label="SVFD (Ex, TE₀)")
    ax2.plot(widths * 1e9, neff_vfd_te, "b-", linewidth=1.9, label="VFD (TE₀)")
    ax2.plot(widths_fmm * 1e9, neff_fmm_te, "bo", markersize=5.5, label="FMM2d (TE₀)")
    ax2.plot(widths * 1e9, neff_sv_ey, "r--", linewidth=1.5, label="SVFD (Ey, TM₀)")
    ax2.plot(widths * 1e9, neff_vfd_tm, "r-", linewidth=1.9, label="VFD (TM₀)")
    ax2.plot(widths_fmm * 1e9, neff_fmm_tm, "rs", markersize=5.0, label="FMM2d (TM₀)")
    ax2.set_title("SOI Effective Index vs. Width: Scalar vs. Vectorial")
    ax2.set_xlabel("Core Width w (nm)")
    ax2.set_ylabel("Effective Index n_eff")
    ax2.grid(True, alpha=0.3)
    ax2.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    plt.show()
    _ = c2


if __name__ == "__main__":
    main()
