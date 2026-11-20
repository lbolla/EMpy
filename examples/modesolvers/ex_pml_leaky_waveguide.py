"""Complex-coordinate-stretching PMLs for radiative leaky waveguides.

Demonstrates ``EMpy.modesolvers.FD.stretchmesh`` Perfectly Matched Layer (PML)
absorbing boundaries with ``EMpy.modesolvers.FD.VFDModeSolver`` on a W-type
depressed-cladding leaky slab/channel waveguide:

- A core of index n_core = 1.46 (width 2a = 1.6 um) is separated from a
  higher-index outer substrate/cladding (n_outer = 1.49) by finite low-index
  trench barriers (n_barrier = 1.40) of width t_b.
- Because n_eff < n_outer, guided core light tunnels through the finite barrier
  and radiates outward, acquiring a passive complex effective index
  n_eff = n_r - 1j * kappa_leak (kappa_leak > 0 under exp(+i*w*t - i*beta*z),
  requiring Im(factor) < 0 in ``stretchmesh``).
- Compares 2D ``VFDModeSolver`` + ``stretchmesh`` radiative attenuation against
  the exact 1D 5-layer analytical leaky-wave dispersion relation as barrier
  thickness t_b increases (lossy leaky -> lossless guided limit).

References
----------
- W. C. Chew and W. H. Weedon, "A 3D perfectly matched medium from modified
  Maxwell's equations with stretched coordinates," *Microw. Opt. Technol. Lett.*
  **7**(13), 599–604 (1994).
- A. K. Ghatak, K. Thyagarajan, and M. R. Shenoy, "Numerical analysis of planar
  optical waveguides using matrix approach," *J. Lightwave Technol.* **5**(5),
  660–667 (1987).
- A. B. Fallahkhair, K. S. Li, and T. E. Murphy, "Vector finite difference
  modesolver for anisotropic dielectric waveguides," *J. Lightwave Technol.*
  **26**(11), 1423–1431 (2008).
"""

from matplotlib import pyplot as plt
import numpy as np
import scipy.optimize

import EMpy


def leaky_w_slab_te0(wl, half_core, t_barrier, n_core, n_barrier, n_outer):
    """Exact complex TE0 root of a symmetric 5-layer W-type leaky slab waveguide.

    Even TE modes satisfy (x = 0 at core center, |x| < a: core, a < |x| < a+t_b:
    barrier, |x| > a+t_b: outgoing radiation exp(-i * k_out * |x|) with Re(k_out) > 0):
        u * tan(u * a) = q * (q + i*k_out * coth(q * t_b)) / (q * coth(q * t_b) + i*k_out)
    Wait: matching E_y and dE_y/dx at x = a and x = a + t_b with E_y(x) ~ cos(u*x)
    and outgoing wave exp(-i * k_out * x) for x > a + t_b gives:
        u * tan(u * a) = q * (q * sinh(q*t_b) + i*k_out * cosh(q*t_b))
                           / (q * cosh(q*t_b) + i*k_out * sinh(q*t_b)).
    """
    k0 = 2.0 * np.pi / wl

    def residual(vec):
        neff = vec[0] + 1j * vec[1]
        u = k0 * np.sqrt(n_core**2 - neff**2 + 0j)
        q = k0 * np.sqrt(neff**2 - n_barrier**2 + 0j)
        if q.real < 0:
            q = -q
        k_out = k0 * np.sqrt(n_outer**2 - neff**2 + 0j)
        if k_out.real - k_out.imag < 0:
            k_out = -k_out
        tanh_qt = np.tanh(q * t_barrier)
        rhs = q * (q * tanh_qt + 1j * k_out) / (q + 1j * k_out * tanh_qt)
        f = u * np.tan(u * half_core) - rhs
        return [f.real, f.imag]

    # Seed from infinitely thick barrier lossless slab root
    def lossless_res(nr):
        u = k0 * np.sqrt(max(n_core**2 - nr**2, 0.0))
        q = k0 * np.sqrt(max(nr**2 - n_barrier**2, 0.0))
        return u * np.tan(u * half_core) - q

    nr0 = scipy.optimize.brentq(lossless_res, n_barrier + 1e-6, n_core - 1e-6)
    sol = scipy.optimize.root(residual, [nr0, -1e-4], tol=1e-13)
    return sol.x[0] + 1j * sol.x[1]


def main():
    wl = 1.55e-6
    a_half = 0.85e-6  # Core width 2a = 1.70 um
    n_core = 1.46
    n_barrier = 1.40
    n_outer = 1.49

    # Quasi-1D slab simulated with VFDModeSolver: x has core/barrier/outer + East/West PMLs,
    # y has Magnetic Walls ("SS") on North/South so y-uniform TE_y (Ey-polarized, Hx-dominant)
    # or Electric/Magnetic walls for exact 1D slab reduction!
    # Wait: in VFDModeSolver, North/South "0" vs "S"/"A":
    # In FD.py: 'S' means Hx is symmetric, Hy is antisymmetric -> dHx/dy = 0, Hy = 0 -> pure TM (Ey, Hx)!
    # 'A' means Hx is antisymmetric, Hy is symmetric -> Hx = 0, dHy/dy = 0 -> pure TE (Ex, Hy)!
    # Wait: for x-stratified slab n(x), TE to x-interfaces has E_y (tangential to x=const) and H_x!
    # Wait: for n = n(x) independent of y and d/dy = 0:
    # Mode with (Ey, Hx, Hz) has E tangential to x-interfaces -> TE slab mode! And Hx is independent of y -> North/South 'S'!
    barrier_widths = np.linspace(0.60e-6, 1.80e-6, 7)
    loss_theory_dbcm = np.zeros_like(barrier_widths)
    loss_vfd_dbcm = np.zeros_like(barrier_widths)
    neff_r_theory = np.zeros_like(barrier_widths)
    neff_r_vfd = np.zeros_like(barrier_widths)

    profile_x = None
    profile_ey = None

    for i, t_b in enumerate(barrier_widths):
        neff_th = leaky_w_slab_te0(wl, a_half, float(t_b), n_core, n_barrier, n_outer)
        neff_r_theory[i] = neff_th.real
        loss_theory_dbcm[i] = EMpy.utils.rix2losses(neff_th, wl)[4]

        x_max = a_half + float(t_b) + 2.2e-6
        x_raw = np.linspace(-x_max, x_max, 221)
        y_raw = np.linspace(0.0, 0.4e-6, 5)
        # Stretch only East and West boundaries (nlayers = [North, South, East, West])
        xx, yy, _, _, _, _ = EMpy.modesolvers.FD.stretchmesh(
            x_raw, y_raw, [0, 0, 28, 28], 1.0 - 1.8j, method="PPPP"
        )

        def eps_w(xc, yc, tb=float(t_b)):
            ax = np.abs(xc)[:, np.newaxis]
            eps_col = np.where(
                ax <= a_half,
                n_core**2,
                np.where(ax <= a_half + tb, n_barrier**2, n_outer**2),
            )
            return np.repeat(eps_col, len(yc), axis=1)

        sol = EMpy.modesolvers.FD.VFDModeSolver(wl, xx, yy, eps_w, "SS00").solve(
            2, 1e-11, guess=neff_th.real
        )
        # Pick the core-guided leaky mode closest to neff_th
        best_m = min(sol.modes, key=lambda m: abs(m.neff - neff_th))
        neff_r_vfd[i] = best_m.neff.real
        loss_vfd_dbcm[i] = EMpy.utils.rix2losses(best_m.neff, wl)[4]

        if i == 1:
            profile_x = EMpy.utils.centered1d(np.real(xx))
            profile_ey = best_m.Ey[:, len(yc_mid := best_m.Ey[0]) // 2]
            _ = yc_mid

    rel_err = np.max(np.abs(loss_vfd_dbcm - loss_theory_dbcm) / loss_theory_dbcm)
    print("W-type leaky waveguide tunneling radiation loss (VFD+PML vs. exact 1D):")
    for tb, l_th, l_fd in zip(barrier_widths, loss_theory_dbcm, loss_vfd_dbcm):
        print(
            f"  barrier t_b = {tb * 1e6:.2f} um: analytical = {l_th:8.3f} dB/cm, "
            f"VFD+stretchmesh PML = {l_fd:8.3f} dB/cm"
        )
    print(f"Max relative leakage-loss discrepancy: {rel_err:.2%}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4))

    ax1.semilogy(
        barrier_widths * 1e6,
        loss_theory_dbcm,
        "k-",
        linewidth=2.0,
        label="Exact 5-layer leaky dispersion",
    )
    ax1.semilogy(
        barrier_widths * 1e6,
        loss_vfd_dbcm,
        "ro",
        markersize=6.0,
        label="VFDModeSolver + stretchmesh PML",
    )
    ax1.set_title("Tunneling Radiation Loss vs. Barrier Thickness (κ → 0⁺)")
    ax1.set_xlabel("Low-Index Trench Width t_b (µm)")
    ax1.set_ylabel("Radiative Leakage Loss (dB/cm)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8.5)

    assert profile_x is not None and profile_ey is not None
    ey_norm = profile_ey / profile_ey[len(profile_ey) // 2]
    tb_plot = float(barrier_widths[1])
    ax2.plot(profile_x * 1e6, np.abs(ey_norm), "b-", linewidth=1.8, label="|E_y(x)|")
    ax2.plot(
        profile_x * 1e6,
        np.real(ey_norm),
        "r--",
        linewidth=1.3,
        alpha=0.85,
        label="Re(E_y): oscillatory leaky tail",
    )
    ax2.axvspan(
        -a_half * 1e6, a_half * 1e6, color="gold", alpha=0.2, label="Core (n=1.46)"
    )
    ax2.axvspan(
        a_half * 1e6,
        (a_half + tb_plot) * 1e6,
        color="skyblue",
        alpha=0.25,
        label="Barrier (n=1.40)",
    )
    ax2.axvspan(-(a_half + tb_plot) * 1e6, -a_half * 1e6, color="skyblue", alpha=0.25)
    ax2.set_title(f"Leaky Mode Profile & PML Absorption (t_b = {tb_plot * 1e6:.2f} µm)")
    ax2.set_xlabel("x (µm)")
    ax2.set_ylabel("Normalized Electric Field")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
