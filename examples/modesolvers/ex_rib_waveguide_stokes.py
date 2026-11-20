"""Asymmetric etched SOI rib waveguide: TE–TM hybridization and Stokes maps.

Demonstrates ``EMpy.modesolvers.FD.VFDModeSolver`` on a laterally asymmetric
half-etched silicon rib waveguide where broken mirror symmetry couples
horizontal (Ex) and vertical (Ey) field components:

1. Sweeps rib width across the TM0–TE1 mode-hybridization anti-crossing,
   tracking ``neff`` and ``TEfrac()`` as the two eigenmodes exchange
   polarization character.
2. Computes spatially resolved Stokes polarization parameters (``S0``, ``S1/S0``,
   ``S2/S0``, ``S3/S0``) at the hybrid anti-crossing point.

Refined from the waveguide Stokes polarization example contributed by David
Hutchings.

References
----------
- A. B. Fallahkhair, K. S. Li, and T. E. Murphy, "Vector finite difference
  modesolver for anisotropic dielectric waveguides," *J. Lightwave Technol.*
  **26**(11), 1423–1431 (2008).
- D. Dai and J. E. Bowers, "Novel concept for ultracompact polarization
  splitter-rotator based on silicon nanowires," *Opt. Express* **19**(11),
  10940–10949 (2011).
- B. M. A. Rahman, S. S. A. Obayya, N. Somasiri, M. Rajarajan, K. T. V. Grattan,
  and H. A. El-Mikati, "Design and characterization of compact single-section
  passive polarization rotator," *J. Lightwave Technol.* **19**(4), 512–519
  (2001).
"""

from matplotlib import pyplot as plt
import numpy as np

import EMpy


def asymmetric_rib_eps(xc, yc, w_rib):
    """Asymmetric L-shaped / one-sided slab SOI rib waveguide cross-section."""
    n_sio2 = 1.444
    n_si = 3.476
    n_air = 1.0
    y_box = 0.60e-6
    h_slab = 0.11e-6
    h_rib = 0.22e-6

    xx, yy = np.meshgrid(xc, yc, indexing="ij")
    eps = np.full((len(xc), len(yc)), n_air**2, dtype=float)
    eps[yy < y_box] = n_sio2**2

    # Full-height core in [-w_rib/2, +w_rib/2], plus one-sided etched slab on x > +w_rib/2
    in_rib = (yy >= y_box) & (yy <= y_box + h_rib) & (np.abs(xx) <= 0.5 * w_rib)
    in_right_slab = (
        (yy >= y_box) & (yy <= y_box + h_slab) & (xx > 0.5 * w_rib) & (xx <= 0.95e-6)
    )
    eps[in_rib | in_right_slab] = n_si**2
    return eps


def main():
    wl = 1.55e-6
    x = np.linspace(-1.05e-6, 1.25e-6, 62)
    y = np.linspace(0.0, 1.45e-6, 54)

    # --- 1. Sweep rib width across the TM0 <-> TE1 hybridization anti-crossing ---
    widths = np.linspace(0.56e-6, 0.84e-6, 11)
    neff_m1 = np.zeros_like(widths)
    neff_m2 = np.zeros_like(widths)
    tefrac_m1 = np.zeros_like(widths)
    tefrac_m2 = np.zeros_like(widths)

    hybrid_solver = None
    best_hybrid_score = 1.0
    w_hybrid = widths[0]

    for i, w in enumerate(widths):
        sol = EMpy.modesolvers.FD.VFDModeSolver(
            wl, x, y, lambda xc, yc, ww=float(w): asymmetric_rib_eps(xc, yc, ww), "0000"
        ).solve(3, 1e-10)
        # Mode 0 is TE0; Modes 1 and 2 undergo TM0 <-> TE1 hybridization
        neff_m1[i] = sol.modes[1].neff.real
        neff_m2[i] = sol.modes[2].neff.real
        tefrac_m1[i] = sol.modes[1].TEfrac()
        tefrac_m2[i] = sol.modes[2].TEfrac()

        score = abs(tefrac_m1[i] - 0.5)
        if score < best_hybrid_score:
            best_hybrid_score = score
            hybrid_solver = sol
            w_hybrid = float(w)

    assert hybrid_solver is not None
    m_hyb = hybrid_solver.modes[1]
    xc = EMpy.utils.centered1d(x)
    yc = EMpy.utils.centered1d(y)

    # Transverse Stokes parameters from electric field (Ex, Ey)
    ex, ey = m_hyb.Ex, m_hyb.Ey
    s0 = np.abs(ex) ** 2 + np.abs(ey) ** 2
    s1 = np.abs(ex) ** 2 - np.abs(ey) ** 2
    s2 = 2.0 * np.real(ex * np.conj(ey))
    mask = s0 > 0.02 * np.max(s0)
    s2_norm = np.where(mask, s2 / np.maximum(s0, 1e-30), 0.0)

    print(
        f"Strongest TM0-TE1 hybridization at w_rib = {w_hybrid * 1e9:.1f} nm: "
        f"Mode 1 neff = {m_hyb.neff.real:.5f} (TEfrac = {m_hyb.TEfrac():.3f}), "
        f"Mode 2 neff = {hybrid_solver.modes[2].neff.real:.5f} "
        f"(TEfrac = {hybrid_solver.modes[2].TEfrac():.3f})"
    )
    print(
        f"Power-weighted <S1/S0> = {np.sum(s1) / np.sum(s0):+.3f}, "
        f"<|S2|/S0> = {np.sum(np.abs(s2)) / np.sum(s0):.3f}"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.5))

    sc1 = ax1.scatter(
        widths * 1e9,
        neff_m1,
        c=tefrac_m1,
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        s=50,
        zorder=3,
        label="Upper branch (Mode 1)",
    )
    ax1.scatter(
        widths * 1e9,
        neff_m2,
        c=tefrac_m2,
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        marker="s",
        s=45,
        zorder=3,
        label="Lower branch (Mode 2)",
    )
    ax1.plot(widths * 1e9, neff_m1, "k-", alpha=0.35)
    ax1.plot(widths * 1e9, neff_m2, "k--", alpha=0.35)
    ax1.axvline(w_hybrid * 1e9, color="gray", linestyle=":", linewidth=1.2)
    fig.colorbar(sc1, ax=ax1, label="TE Polarization Fraction")
    ax1.set_title("Asymmetric SOI Rib: TM₀–TE₁ Avoided Crossing")
    ax1.set_xlabel("Rib Width w_rib (nm)")
    ax1.set_ylabel("Effective Index n_eff")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8.5)

    im = ax2.pcolormesh(
        xc * 1e6,
        yc * 1e6,
        s2_norm.T,
        cmap="PRGn",
        vmin=-1.0,
        vmax=1.0,
        shading="auto",
    )
    ax2.contour(
        xc * 1e6,
        yc * 1e6,
        (s0 / np.max(s0)).T,
        levels=[0.15, 0.45, 0.75],
        colors="k",
        linewidths=0.9,
    )
    # Draw asymmetric L-rib outline
    y0, hr, hs, hw = 0.60, 0.22, 0.11, 0.5 * w_hybrid * 1e6
    ax2.plot(
        [-hw, 0.95, 0.95, hw, hw, -hw, -hw],
        [y0, y0, y0 + hs, y0 + hs, y0 + hr, y0 + hr, y0],
        "k-",
        linewidth=1.6,
    )
    fig.colorbar(im, ax=ax2, label="Diagonal Stokes S₂ / S₀ (±45° Hybridization)")
    ax2.set_aspect("equal")
    ax2.set_title(f"Hybrid Mode Stokes S₂/S₀ (w = {w_hybrid * 1e9:.0f} nm)")
    ax2.set_xlabel("x (µm)")
    ax2.set_ylabel("y (µm)")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
