"""Tunable triple-coupler and coupled-ring flat-top filters (Barbarossa 1995/1997).

Demonstrates ``EMpy.devices.T_TCRR``, ``T_CRTCRR``, and ``T_CTCRR`` together
with ``EMpy.utils.find_peaks`` to reproduce the coupled-ring Vernier / MZI-assisted
resonator topologies of Barbarossa, Matteo, and Armenise:

1. ``T_TCRR`` (Tunable Triple-Coupled Ring Resonator): compares ``coupling="optimum"``
   synthesis (``l2 = 2*l1``, ``K3 = 2*K1*sqrt(1 - K1^2)``) against detuned MZI
   arm coupling for tunable finesse and extinction control.
2. ``T_CRTCRR`` and ``T_CTCRR`` (``coupling="optimum"``): compares maximally
   flat passband synthesis across lossless (``q = 1``) and lossy (``q = 0.992``)
   regimes, extracting 3-dB passband widths via ``EMpy.utils.find_peaks``.

References
----------
- G. Barbarossa, A. M. Matteo, and M. N. Armenise, "Theoretical analysis of
  triple-coupler ring-based optical guided-wave resonator," *J. Lightwave
  Technol.* **13**(2), 148–157 (1995).
- G. Barbarossa, A. M. Matteo, and M. N. Armenise, "Novel configuration for
  fourth-order optical guided-wave resonator," *IEE Proc.-Optoelectron.*
  **144**(4), 203–208 (1997).
"""

from matplotlib import pyplot as plt
import numpy as np

import EMpy


def main():
    wls = np.linspace(1.546e-6, 1.554e-6, 1601)
    swg = EMpy.devices.SWG(460, 220, 25).solve(wls)
    l0 = 25.0e-6  # Base arc length

    # --- 1. T_TCRR: optimum MZI-assisted critical condition vs. detuned coupling ---
    k_tcrr = np.array([0.35, 0.35, 0.40])
    l_tcrr = np.array([0.5 * l0, 1.0 * l0, 0.25 * l0, 0.25 * l0])
    tcrr_opt = EMpy.devices.T_TCRR(
        swg.neff, k_tcrr, 1.0, l_tcrr, coupling="optimum"
    ).solve(wls)
    tcrr_detuned = EMpy.devices.T_TCRR(
        swg.neff, np.array([0.35, 0.20, 0.28]), 1.0, l_tcrr, coupling=None
    ).solve(wls)

    # --- 2. Higher-order coupled-ring filters: T_CRTCRR & T_CTCRR ("optimum") ---
    k_crt = np.array([0.42, 0.14, 0.36])
    l_crt = np.array([l0, l0, 2.0 * l0])
    crt_lossless = EMpy.devices.T_CRTCRR(swg.neff, k_crt, 1.0, l_crt).solve(wls)

    k_ctc = np.array([0.45, 0.45, 0.20])
    l_ctc = np.array([l0, l0, 2.0 * l0, 2.0 * l0])
    ctc_opt_lossless = EMpy.devices.T_CTCRR(
        swg.neff, k_ctc, 1.0, l_ctc, coupling="optimum"
    ).solve(wls)
    ctc_opt_lossy = EMpy.devices.T_CTCRR(
        swg.neff, k_ctc, 0.992, l_ctc, coupling="optimum"
    ).solve(wls)

    # Verify lossless unitarity |THRU|^2 + |DROP|^2 = 1 on all three architectures
    u_err_tcrr = float(
        np.max(np.abs(np.abs(tcrr_opt.THRU) ** 2 + np.abs(tcrr_opt.DROP) ** 2 - 1.0))
    )
    u_err_crt = float(
        np.max(
            np.abs(
                np.abs(crt_lossless.THRU) ** 2 + np.abs(crt_lossless.DROP) ** 2 - 1.0
            )
        )
    )
    u_err_ctc = float(
        np.max(
            np.abs(
                np.abs(ctc_opt_lossless.THRU) ** 2
                + np.abs(ctc_opt_lossless.DROP) ** 2
                - 1.0
            )
        )
    )

    peaks_tcrr = EMpy.utils.find_peaks(
        wls * 1e9, np.abs(tcrr_opt.DROP) ** 2, threshold=0.2
    )
    peaks_ctc = EMpy.utils.find_peaks(
        wls * 1e9, np.abs(ctc_opt_lossless.DROP) ** 2, threshold=0.2
    )

    print(
        f"Lossless unitarity max ||THRU|^2 + |DROP|^2 - 1|: "
        f"T_TCRR={u_err_tcrr:.2e}, T_CRTCRR={u_err_crt:.2e}, T_CTCRR={u_err_ctc:.2e}"
    )
    print(
        f"3-dB passband FWHM (via find_peaks): "
        f"T_TCRR(optimum) = {peaks_tcrr[0].FWHM:.3f} nm, "
        f"T_CTCRR(optimum) = {peaks_ctc[0].FWHM:.3f} nm"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4))

    ax1.plot(
        wls * 1e9,
        10.0 * np.log10(np.abs(tcrr_opt.DROP) ** 2),
        "b-",
        linewidth=1.8,
        label="T_TCRR DROP (optimum)",
    )
    ax1.plot(
        wls * 1e9,
        10.0 * np.log10(np.maximum(np.abs(tcrr_opt.THRU) ** 2, 1e-5)),
        "b:",
        linewidth=1.4,
        label="T_TCRR THRU (optimum)",
    )
    ax1.plot(
        wls * 1e9,
        10.0 * np.log10(np.abs(tcrr_detuned.DROP) ** 2),
        "r--",
        linewidth=1.6,
        label="T_TCRR DROP (detuned)",
    )
    ax1.set_title("Barbarossa T_TCRR: Optimum vs. Detuned MZI Coupling")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Power Transmission (dB)")
    ax1.set_ylim(-40, 2)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8.5)

    ax2.plot(
        wls * 1e9,
        10.0 * np.log10(np.abs(ctc_opt_lossless.DROP) ** 2),
        "b-",
        linewidth=1.9,
        label="T_CTCRR DROP (optimum, q=1)",
    )
    ax2.plot(
        wls * 1e9,
        10.0 * np.log10(np.abs(ctc_opt_lossy.DROP) ** 2),
        "b--",
        linewidth=1.5,
        label="T_CTCRR DROP (optimum, q=0.992)",
    )
    ax2.plot(
        wls * 1e9,
        10.0 * np.log10(np.abs(crt_lossless.DROP) ** 2),
        "g-.",
        linewidth=1.5,
        label="T_CRTCRR DROP (q=1)",
    )
    ax2.set_title("4th-Order Coupled-Ring Flat-Top Synthesis")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("DROP Transmission (dB)")
    ax2.set_ylim(-40, 2)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8.5)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
