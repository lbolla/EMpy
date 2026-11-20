"""Waveguide step-discontinuity S-matrix: passive vs. reciprocal mode matching.

Demonstrates ``EMpy.modesolvers.geometries.SimpleJoint`` (``method="passive"``
and ``method="reciprocal"``), ``S2T`` / ``T2S``, and
``EMpy.modesolvers.interface.interface_matrix`` at an abrupt waveguide width
step (w1 = 0.50 um -> w2 = 1.35 um):

1. Sweeps retained mode count N = 1..6, comparing ``method="passive"``
   (unconditional passivity ||S||_2 <= 1 under basis truncation) against
   ``method="reciprocal"`` (exact Lorentz reciprocity S == S^T at every finite
   N) as both converge to the same physical scattering amplitudes.
2. Verifies cross-solver compatibility by matching ``VFDModeSolver`` on the left
   waveguide directly to ``FMM2d`` on the right waveguide, and checks strict
   passivity ||S||_2 < 1 when material absorption is introduced.

References
----------
- C. Vassallo, *Optical Waveguide Concepts* (Elsevier, Amsterdam, 1991), Ch. 4.
- G. V. Eleftheriades, A. S. Omar, L. P. B. Katehi, and G. M. Rebeiz, "Some
  important properties of waveguide junction generalized scattering matrices in
  the context of the mode matching technique," *IEEE Trans. Microw. Theory Tech.*
  **42**(10), 1896–1903 (1994).
- A. S. Sudbø, "Film mode matching: a versatile numerical method for vector mode
  field calculations in dielectric waveguides," *Pure Appl. Opt.* **2**(3),
  211–233 (1993).
"""

from matplotlib import pyplot as plt
import numpy as np

import EMpy


def make_vfd_waveguide(wl, width, nmodes, n_core=3.476 + 0.0j, n_clad=1.444):
    """Solve a symmetric buried channel waveguide of given width with VFDModeSolver."""
    h_core = 0.22e-6
    x = np.linspace(-1.35e-6, 1.35e-6, 55)
    y = np.linspace(-0.75e-6, 0.75e-6, 37)

    def epsf(xc, yc):
        xx, yy = np.meshgrid(xc, yc, indexing="ij")
        in_core = (np.abs(xx) <= 0.5 * width) & (np.abs(yy) <= 0.5 * h_core)
        return np.where(in_core, n_core**2, n_clad**2)

    return EMpy.modesolvers.FD.VFDModeSolver(wl, x, y, epsf, "0000").solve(
        nmodes, 1e-10, guess=n_core.real
    )


def main():
    wl = 1.55e-6
    w_narrow = 0.50e-6
    w_wide = 1.35e-6

    mode_counts = [1, 2, 3, 4, 5, 6]
    t00_passive = np.zeros(len(mode_counts))
    t00_recip = np.zeros(len(mode_counts))
    r00_passive = np.zeros(len(mode_counts))
    r00_recip = np.zeros(len(mode_counts))
    rec_err_passive = np.zeros(len(mode_counts))
    rec_err_recip = np.zeros(len(mode_counts))
    norm_passive = np.zeros(len(mode_counts))
    norm_recip = np.zeros(len(mode_counts))

    for idx, nm in enumerate(mode_counts):
        sol1 = make_vfd_waveguide(wl, w_narrow, nm)
        sol2 = make_vfd_waveguide(wl, w_wide, nm)

        j_pas = EMpy.modesolvers.geometries.SimpleJoint(sol1, sol2, method="passive")
        j_rec = EMpy.modesolvers.geometries.SimpleJoint(sol1, sol2, method="reciprocal")

        # Verify S <-> T round-trip when all retained modes couple (N = 1; for N >= 2,
        # opposite-symmetry/polarization modes have zero cross-overlap so s12 is rank-deficient
        # and only the generalized S-matrix is nonsingular).
        if nm == 1:
            s_rt = EMpy.modesolvers.geometries.T2S(j_pas.T)
            assert np.allclose(s_rt, j_pas.S, atol=1e-9)

        t00_passive[idx] = abs(j_pas.S[nm, 0]) ** 2
        t00_recip[idx] = abs(j_rec.S[nm, 0]) ** 2
        r00_passive[idx] = abs(j_pas.S[0, 0]) ** 2
        r00_recip[idx] = abs(j_rec.S[0, 0]) ** 2
        rec_err_passive[idx] = float(np.linalg.norm(j_pas.S - j_pas.S.T))
        rec_err_recip[idx] = float(np.linalg.norm(j_rec.S - j_rec.S.T))
        norm_passive[idx] = float(np.linalg.svd(j_pas.S, compute_uv=False)[0])
        norm_recip[idx] = float(np.linalg.svd(j_rec.S, compute_uv=False)[0])

    # Cross-solver SimpleJoint: VFDModeSolver (narrow) <-> FMM2d (wide) + lossy check
    sol1_vfd = make_vfd_waveguide(wl, w_narrow, 2)
    ux = np.array([-1.35e-6, -0.5 * w_wide, 0.5 * w_wide, 1.35e-6])
    uy = np.array([-0.75e-6, -0.11e-6, 0.11e-6, 0.75e-6])
    rix = np.array(
        [
            [1.444, 1.444, 1.444],
            [1.444, 3.476, 1.444],
            [1.444, 1.444, 1.444],
        ]
    )
    bc = EMpy.modesolvers.FMM.Boundary(
        xleft="Electric Wall",
        xright="Electric Wall",
        yleft="Magnetic Wall",
        yright="Magnetic Wall",
    )
    sol2_fmm = EMpy.modesolvers.FMM.FMM2d(ux, uy, rix, bc).solve(wl, 6, 2)
    for m in sol2_fmm.modes:
        m.normalize()
    j_cross = EMpy.modesolvers.geometries.SimpleJoint(
        sol1_vfd, sol2_fmm, method="passive"
    )
    o11, _, o12, _ = EMpy.modesolvers.interface.interface_matrix(sol1_vfd, sol2_fmm)

    sol1_lossy = make_vfd_waveguide(wl, w_narrow, 2, n_core=3.476 - 0.02j)
    sol2_lossy = make_vfd_waveguide(wl, w_wide, 2, n_core=3.476 - 0.02j)
    j_lossy = EMpy.modesolvers.geometries.SimpleJoint(
        sol1_lossy, sol2_lossy, method="passive"
    )
    s_norm_lossy = float(np.linalg.svd(j_lossy.S, compute_uv=False)[0])

    print(
        f"N={mode_counts[-1]} step junction (0.50 -> 1.35 um): "
        f"|S21,00|^2 passive = {t00_passive[-1]:.5f}, reciprocal = {t00_recip[-1]:.5f}; "
        f"|S11,00|^2 passive = {r00_passive[-1]:.5f}"
    )
    print(
        f"Max spectral norm ||S||_2 across N: passive = {np.max(norm_passive):.6f}, "
        f"reciprocal = {np.max(norm_recip):.6f}; lossy ||S||_2 = {s_norm_lossy:.6f}"
    )
    print(
        f"Cross-solver VFD <-> FMM2d |O12[0,0]| = {abs(o12[0, 0]):.5f} "
        f"(|O11[0,0]| = {abs(o11[0, 0]):.5f}), |S21,00|^2 = {abs(j_cross.S[2, 0])**2:.5f}"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4))

    ax1.plot(
        mode_counts,
        t00_passive,
        "bo-",
        linewidth=1.8,
        label="Passive |S₂₁,₀₀|² (TE₀ → TE₀)",
    )
    ax1.plot(
        mode_counts,
        t00_recip,
        "rs--",
        linewidth=1.8,
        label="Reciprocal |S₂₁,₀₀|² (TE₀ → TE₀)",
    )
    ax1.plot(
        mode_counts,
        100.0 * r00_passive,
        "b^:",
        linewidth=1.5,
        label="100 × |S₁₁,₀₀|² (passive)",
    )
    ax1.plot(
        mode_counts,
        100.0 * r00_recip,
        "rv:",
        linewidth=1.5,
        label="100 × |S₁₁,₀₀|² (reciprocal)",
    )
    ax1.set_title("Waveguide Step Scattering vs. Truncated Mode Count N")
    ax1.set_xlabel("Retained Modes per Waveguide N")
    ax1.set_ylabel("Power Coupling Coefficient")
    ax1.set_xticks(mode_counts)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8.5)

    ax2.semilogy(
        mode_counts,
        np.maximum(rec_err_passive, 1e-16),
        "bo-",
        linewidth=1.8,
        label="||S − Sᵀ||_F (passive)",
    )
    ax2.semilogy(
        mode_counts,
        np.maximum(rec_err_recip, 1e-16),
        "rs--",
        linewidth=1.8,
        label="||S − Sᵀ||_F (reciprocal)",
    )
    ax2.semilogy(
        mode_counts,
        np.maximum(np.abs(1.0 - norm_passive), 1e-16),
        "k^:",
        linewidth=1.5,
        label="|1 − ||S||₂| (passive)",
    )
    ax2.set_title("Reciprocity & Passivity Invariants vs. Mode Basis Size")
    ax2.set_xlabel("Retained Modes per Waveguide N")
    ax2.set_ylabel("Matrix Norm Residual")
    ax2.set_xticks(mode_counts)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8.5)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
