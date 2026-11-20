"""All-pass, single-ring (critical coupling), and 3-ring Vernier resonators.

Demonstrates ``EMpy.devices.Coupler``, ``APRR``, ``SRR``, ``NRR``, ``SWG``, and
``EMpy.utils.group_delay_and_dispersion`` on dispersive Silicon-on-Insulator
microring resonators:

1. All-Pass Ring Resonator (``APRR``): compares under-coupled vs. over-coupled
   phase transitions and extracts resonant group delay (ps) and chromatic
   dispersion (ps/nm) via ``group_delay_and_dispersion``.
2. Single-Ring Add–Drop Resonator (``SRR``): compares lossless unitary response
   (|THRU|^2 + |DROP|^2 = 1) against under-, critical-, and over-coupled lossy
   rings, verifying Yariv's critical-coupling condition |THRU(wl_res)| = 0 when
   t_1 = t_2 * a_rt.
3. Coupled-Resonator Optical Waveguide (3-ring ``NRR``): synthesizes a box-like
   flat-top drop passband with steeper roll-off than a single-ring ``SRR``.

References
----------
- O. Schwelb, "Transmission, group delay, and dispersion in single-ring optical
  resonators and add/drop filters—a tutorial overview," *J. Lightwave Technol.*
  **22**(5), 1380–1394 (2004).
- A. Yariv, "Universal relations for coupling of optical power between
  microresonators and dielectric waveguides," *Electron. Lett.* **36**(4),
  321–322 (2000).
- B. E. Little, S. T. Chu, H. A. Haus, J. Foresi, and J.-P. Laine, "Microring
  resonator channel dropping filters," *J. Lightwave Technol.* **15**(6),
  998–1005 (1997).
"""

from matplotlib import pyplot as plt
import numpy as np

import EMpy


def main():
    wls = np.linspace(1.544e-6, 1.556e-6, 1201)
    swg = EMpy.devices.SWG(450, 220, 25).solve(wls)
    radius = 10.0e-6
    l_ring = 2.0 * np.pi * radius

    # Lossy complex effective index corresponding to 18 dB/cm propagation loss in the ring
    alpha_ring_dbcm = 18.0
    dn_loss = EMpy.utils.loss_dBcm2rix(0.0, alpha_ring_dbcm, 1.55e-6)
    neff_lossy = swg.neff + dn_loss
    a_rt = float(np.abs(np.exp(-1j * (2 * np.pi / 1.55e-6) * dn_loss * l_ring)))

    # --- 1. All-Pass Ring Resonator (APRR): group delay & dispersion ---
    k_crit_aprr = np.sqrt(1.0 - a_rt**2)
    aprr_over = EMpy.devices.APRR(
        EMpy.devices.Coupler(wls, 1.65 * k_crit_aprr, 1.0), neff_lossy, l_ring
    ).solve()
    aprr_under = EMpy.devices.APRR(
        EMpy.devices.Coupler(wls, 0.55 * k_crit_aprr, 1.0), neff_lossy, l_ring
    ).solve()

    phi_over, tau_over, disp_over = EMpy.utils.group_delay_and_dispersion(
        wls, aprr_over.THRU
    )
    phi_under, tau_under, _ = EMpy.utils.group_delay_and_dispersion(
        wls, aprr_under.THRU
    )

    # --- 2. Single-Ring Resonator (SRR): lossless vs. critical coupling ---
    k2 = np.sqrt(0.04)  # Weak drop-bus coupling (4% power)
    t2 = np.sqrt(1.0 - k2**2)
    # Critical input coupling on THRU port: K1bar = K2bar * a_rt -> K1_crit = sqrt(1 - (t2 * a_rt)^2)
    k1_crit = np.sqrt(1.0 - (t2 * a_rt) ** 2)

    srr_lossless = EMpy.devices.SRR(
        EMpy.devices.Coupler(wls, k1_crit),
        EMpy.devices.Coupler(wls, k2),
        swg.neff,
        0.5 * l_ring,
        0.5 * l_ring,
    ).solve()
    srr_crit = EMpy.devices.SRR(
        EMpy.devices.Coupler(wls, k1_crit),
        EMpy.devices.Coupler(wls, k2),
        neff_lossy,
        0.5 * l_ring,
        0.5 * l_ring,
    ).solve()
    srr_over = EMpy.devices.SRR(
        EMpy.devices.Coupler(wls, np.sqrt(0.20)),
        EMpy.devices.Coupler(wls, k2),
        neff_lossy,
        0.5 * l_ring,
        0.5 * l_ring,
    ).solve()

    # --- 3. 3-Ring Butterworth-like NRR flat-top filter vs. 1-Ring SRR ---
    wls_nrr = np.linspace(1.5475e-6, 1.5525e-6, 401)
    swg_nrr = EMpy.devices.SWG(450, 220, 25).solve(wls_nrr)
    k_bus = np.sqrt(0.22)
    k_inter = np.sqrt(0.018)
    couplers_3 = [
        EMpy.devices.Coupler(wls_nrr, k_bus),
        EMpy.devices.Coupler(wls_nrr, k_inter),
        EMpy.devices.Coupler(wls_nrr, k_inter),
        EMpy.devices.Coupler(wls_nrr, k_bus),
    ]
    half_arcs = [0.5 * l_ring] * 3
    nrr_3 = EMpy.devices.NRR(
        couplers_3, [swg_nrr.neff] * 3, half_arcs, half_arcs
    ).solve()
    srr_1_ref = EMpy.devices.SRR(
        EMpy.devices.Coupler(wls_nrr, np.sqrt(0.08)),
        EMpy.devices.Coupler(wls_nrr, np.sqrt(0.08)),
        swg_nrr.neff,
        0.5 * l_ring,
        0.5 * l_ring,
    ).solve()

    unitarity_err = float(
        np.max(
            np.abs(
                np.abs(srr_lossless.THRU) ** 2 + np.abs(srr_lossless.DROP) ** 2 - 1.0
            )
        )
    )
    min_thru_crit_db = float(20.0 * np.log10(np.min(np.abs(srr_crit.THRU))))
    print(
        f"Lossless SRR max ||THRU|^2 + |DROP|^2 - 1| = {unitarity_err:.2e}; "
        f"Critical-coupled SRR min THRU = {min_thru_crit_db:.1f} dB"
    )
    print(
        f"APRR peak group delay (over-coupled): {np.max(tau_over):.2f} ps, "
        f"max |dispersion| = {np.max(np.abs(disp_over)):.1f} ps/nm"
    )
    _ = (phi_over, phi_under)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.2, 4.3))

    wls_mid = 0.5 * (wls[:-1] + wls[1:])
    ax1.plot(wls_mid * 1e9, tau_over, "b-", linewidth=1.7, label="Over-coupled τ_g")
    ax1.plot(wls_mid * 1e9, tau_under, "r--", linewidth=1.6, label="Under-coupled τ_g")
    ax1.set_title("APRR Group Delay Spectrum")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Group Delay τ_g (ps)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.plot(
        wls * 1e9,
        20.0 * np.log10(np.abs(srr_lossless.THRU)),
        "k:",
        linewidth=1.5,
        label="Lossless THRU",
    )
    ax2.plot(
        wls * 1e9,
        20.0 * np.log10(np.maximum(np.abs(srr_crit.THRU), 1e-4)),
        "b-",
        linewidth=1.7,
        label="Critical lossy THRU",
    )
    ax2.plot(
        wls * 1e9,
        20.0 * np.log10(np.abs(srr_over.THRU)),
        "r--",
        linewidth=1.5,
        label="Over-coupled lossy THRU",
    )
    ax2.set_title("SRR Critical Coupling vs. Lossless")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("THRU Transmission (dB)")
    ax2.set_ylim(-45, 2)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    ax3.plot(
        wls_nrr * 1e9,
        20.0 * np.log10(np.abs(nrr_3.DROP)),
        "b-",
        linewidth=1.8,
        label="3-Ring NRR DROP",
    )
    ax3.plot(
        wls_nrr * 1e9,
        20.0 * np.log10(np.maximum(np.abs(nrr_3.THRU), 1e-4)),
        "b:",
        linewidth=1.3,
        label="3-Ring NRR THRU",
    )
    ax3.plot(
        wls_nrr * 1e9,
        20.0 * np.log10(np.abs(srr_1_ref.DROP)),
        "r--",
        linewidth=1.6,
        label="1-Ring SRR DROP",
    )
    ax3.set_title("3-Ring CROW Filter vs. Single Ring")
    ax3.set_xlabel("Wavelength (nm)")
    ax3.set_ylabel("Power Transmission (dB)")
    ax3.set_ylim(-38, 2)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
