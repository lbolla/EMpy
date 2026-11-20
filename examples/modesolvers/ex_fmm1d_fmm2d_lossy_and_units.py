"""Film Mode Matching (FMM1d & FMM2d): lossy attenuation and scale invariance.

Demonstrates ``EMpy.modesolvers.FMM.FMM1d``, ``FMM2d``, ``Boundary``, and the
complex-index / attenuation conversion helpers ``loss_dBcm2rix``,
``loss_cm2rix``, ``loss_m2rix``, and ``rix2losses``:

1. ``FMM1d`` TE and TM modes of an asymmetric SOI slab waveguide cross-checked
   against the exact transcendental 3-layer slab dispersion equation.
2. ``FMM2d`` on a lossy silicon channel waveguide across continuous bulk-core
   absorption ``alpha_core -> 0+`` (0 to 50 dB/cm), showing linear modal
   attenuation scaling ``alpha_mode = Gamma_conf * alpha_core`` and exact
   geometric unit invariance between SI meters (``wl = 1.55e-6``) and microns
   (``wl = 1.55``).

References
----------
- A. S. Sudbø, "Film mode matching: a versatile numerical method for vector mode
  field calculations in dielectric waveguides," *Pure Appl. Opt.* **2**(3),
  211–233 (1993).
- A. S. Sudbø, "Improved formulation of the film mode matching method for mode
  field calculations in dielectric waveguides," *Pure Appl. Opt.* **3**(3),
  381–388 (1994).
- C. Vassallo, *Optical Waveguide Concepts* (Elsevier, Amsterdam, 1991).
"""

from matplotlib import pyplot as plt
import numpy as np
import scipy.optimize

import EMpy


def exact_asymmetric_slab_neffs(wl, h, n_sub, n_core, n_sup, pol="TE"):
    """Analytical guided effective indices of an asymmetric 3-layer planar slab."""
    k0 = 2.0 * np.pi / wl
    n_min = max(n_sub, n_sup)

    def disp(neff):
        u = k0 * np.sqrt(max(n_core**2 - neff**2, 0.0))
        gs = k0 * np.sqrt(max(neff**2 - n_sub**2, 0.0))
        gc = k0 * np.sqrt(max(neff**2 - n_sup**2, 0.0))
        if pol == "TE":
            p_s, p_c = 1.0, 1.0
        else:
            p_s = (n_core / n_sub) ** 2
            p_c = (n_core / n_sup) ** 2
        return u * h - np.arctan2(p_s * gs, u) - np.arctan2(p_c * gc, u)

    grid = np.linspace(n_core - 1e-9, n_min + 1e-9, 2000)
    vals = [disp(v) for v in grid]
    roots = []
    for m in range(10):
        target = m * np.pi
        shifted = np.array(vals) - target
        idx = np.where(np.sign(shifted[:-1]) * np.sign(shifted[1:]) <= 0)[0]
        for j in idx:
            r = scipy.optimize.brentq(lambda v: disp(v) - target, grid[j + 1], grid[j])
            roots.append(r)
    return np.array(sorted(roots, reverse=True))


def main():
    wl_m = 1.55e-6

    # --- 1. FMM1d asymmetric slab TE & TM modes vs. analytical dispersion ---
    h_slab = 0.65e-6
    n_sub, n_core, n_sup = 1.444, 3.476, 1.0
    uy_1d = np.array([0.0, 2.0e-6, 2.0e-6 + h_slab, 4.0e-6 + h_slab])
    ny_1d = np.array([n_sub, n_core, n_sup])

    te_exact = exact_asymmetric_slab_neffs(wl_m, h_slab, n_sub, n_core, n_sup, "TE")
    tm_exact = exact_asymmetric_slab_neffs(wl_m, h_slab, n_sub, n_core, n_sup, "TM")

    fmm1d_te = EMpy.modesolvers.FMM.FMM1d(uy_1d, ny_1d, "AA").solve(
        wl_m, len(te_exact), "TE"
    )
    fmm1d_tm = EMpy.modesolvers.FMM.FMM1d(uy_1d, ny_1d, "SS").solve(
        wl_m, len(tm_exact), "TM"
    )
    neff_1d_te = np.array(
        [m.keff.item().real * wl_m / (2 * np.pi) for m in fmm1d_te.modes]
    )
    neff_1d_tm = np.array(
        [m.keff.item().real * wl_m / (2 * np.pi) for m in fmm1d_tm.modes]
    )

    # --- 2. FMM2d lossy channel waveguide (kappa -> 0+) & SI meters vs. microns ---
    bc = EMpy.modesolvers.FMM.Boundary(
        xleft="Electric Wall",
        xright="Electric Wall",
        yleft="Magnetic Wall",
        yright="Magnetic Wall",
    )
    ux_m = np.array([0.0, 1.2e-6, 1.65e-6, 2.85e-6])
    uy_m = np.array([0.0, 1.0e-6, 1.22e-6, 2.22e-6])

    alpha_core_dbcm = np.array([0.0, 20.0, 50.0])
    alpha_te0_dbcm = np.zeros_like(alpha_core_dbcm)
    alpha_tm0_dbcm = np.zeros_like(alpha_core_dbcm)
    neff_te0_real = np.zeros_like(alpha_core_dbcm)

    # Also cross-check loss_cm2rix and loss_m2rix round-trip consistency
    n_test_cm = EMpy.utils.loss_cm2rix(n_core, 5.0, wl_m)
    n_test_m = EMpy.utils.loss_m2rix(n_core, 500.0, wl_m)
    assert np.allclose(n_test_cm, n_test_m)

    last_sol_m = None
    for i, a_db in enumerate(alpha_core_dbcm):
        n_si_lossy = (
            n_core
            if a_db == 0.0
            else EMpy.utils.loss_dBcm2rix(n_core, float(a_db), wl_m)
        )
        rix = np.array(
            [
                [n_sub, n_sub, n_sub],
                [n_sub, n_si_lossy, n_sub],
                [n_sub, n_sub, n_sub],
            ]
        )
        sol = EMpy.modesolvers.FMM.FMM2d(ux_m, uy_m, rix, bc).solve(wl_m, 3, 2)
        last_sol_m = sol
        modes_sorted = sorted(sol.modes, key=lambda m: m.TEfrac(), reverse=True)
        neff_te0_real[i] = modes_sorted[0].neff.real
        alpha_te0_dbcm[i] = EMpy.utils.rix2losses(modes_sorted[0].neff, wl_m)[4]
        alpha_tm0_dbcm[i] = EMpy.utils.rix2losses(modes_sorted[-1].neff, wl_m)[4]

    # Check exact SI meters (1.55e-6) vs. microns (1.55) scale invariance on lossy FMM2d
    assert last_sol_m is not None
    n_si_check = EMpy.utils.loss_dBcm2rix(n_core, 50.0, wl_m)
    rix_check = np.array(
        [
            [n_sub, n_sub, n_sub],
            [n_sub, n_si_check, n_sub],
            [n_sub, n_sub, n_sub],
        ]
    )
    sol_microns = EMpy.modesolvers.FMM.FMM2d(
        ux_m * 1e6, uy_m * 1e6, rix_check, bc
    ).solve(wl_m * 1e6, 3, 2)
    unit_diff = max(
        abs(m1.neff - m2.neff) for m1, m2 in zip(last_sol_m.modes, sol_microns.modes)
    )

    print(
        f"FMM1d vs exact asymmetric slab max |neff_FMM - neff_exact|: "
        f"TE = {np.max(np.abs(neff_1d_te - te_exact)):.2e}, "
        f"TM = {np.max(np.abs(neff_1d_tm - tm_exact)):.2e}"
    )
    print(
        f"FMM2d SI meters vs. microns max |neff(m) - neff(um)| = {unit_diff:.2e}; "
        f"effective confinement slope d(alpha_mode)/d(alpha_core): "
        f"TE0 = {alpha_te0_dbcm[-1] / alpha_core_dbcm[-1]:.3f}, "
        f"TM0 = {alpha_tm0_dbcm[-1] / alpha_core_dbcm[-1]:.3f}"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4))

    m_idx_te = np.arange(len(te_exact))
    m_idx_tm = np.arange(len(tm_exact))
    ax1.plot(m_idx_te, te_exact, "b-", linewidth=1.8, label="Exact slab TE_m")
    ax1.plot(
        m_idx_te, neff_1d_te, "bo", markersize=6.5, fillstyle="none", label="FMM1d TE_m"
    )
    ax1.plot(m_idx_tm, tm_exact, "r--", linewidth=1.8, label="Exact slab TM_m")
    ax1.plot(
        m_idx_tm, neff_1d_tm, "rs", markersize=6.0, fillstyle="none", label="FMM1d TM_m"
    )
    ax1.axhline(n_sub, color="gray", linestyle=":", label="Substrate cut-off n_SiO2")
    ax1.set_title("FMM1d vs. Exact Asymmetric 650 nm SOI Slab")
    ax1.set_xlabel("Slab Mode Order m")
    ax1.set_ylabel("Effective Index n_eff")
    ax1.set_xticks(m_idx_te)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8.5)

    ax2.plot(
        alpha_core_dbcm,
        alpha_te0_dbcm,
        "bo-",
        linewidth=1.8,
        label="FMM2d TE₀ modal loss",
    )
    ax2.plot(
        alpha_core_dbcm,
        alpha_tm0_dbcm,
        "rs--",
        linewidth=1.8,
        label="FMM2d TM₀ modal loss",
    )
    ax2.plot(
        alpha_core_dbcm,
        alpha_core_dbcm,
        "k:",
        alpha=0.7,
        label="Bulk core loss α_core (1:1)",
    )
    ax2.set_title("FMM2d Lossy Channel Waveguide (κ → 0⁺ Limit)")
    ax2.set_xlabel("Bulk Core Absorption α_core (dB/cm)")
    ax2.set_ylabel("Modal Propagation Loss α_mode (dB/cm)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8.5)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
