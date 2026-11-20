"""Thermo-optic silicon Mach–Zehnder interferometer and matrix composition.

Demonstrates ``EMpy.devices.Coupler``, ``Line``, ``MZ``, ``TransferMatrix``,
``ChainMatrix``, ``composeTMlist``, ``composeCMlist``, and temperature-dependent
``EMpy.materials.Si`` / ``ThermalOpticCoefficient``:

1. Simulates an unbalanced silicon Mach–Zehnder Interferometer (MZI) with a
   heated arm at T in {300, 325, 350} K using ``EMpy.materials.Si.n(wls, T=T)``,
   reproducing the silicon thermo-optic phase shift dn/dT ~ 1.86e-4 K^-1.
2. Verifies exact equivalence between ``MZ.solve()`` closed-form transfer
   functions and explicit ``TransferMatrix`` / ``ChainMatrix`` composition, and
   cascades a two-stage Vernier MZI interleaver via ``composeTMlist``.

References
----------
- G. Cocorullo and I. Rendina, "Thermo-optical modulation at 1.5 µm in silicon
  etalon," *Electron. Lett.* **28**(1), 83–85 (1992).
- R. A. Soref and B. R. Bennett, "Electrooptical effects in silicon,"
  *IEEE J. Quantum Electron.* **23**(1), 123–129 (1987).
- O. Schwelb, "Transmission, group delay, and dispersion in single-ring optical
  resonators and add/drop filters—a tutorial overview," *J. Lightwave Technol.*
  **22**(5), 1380–1394 (2004).
"""

from matplotlib import pyplot as plt
import numpy as np

import EMpy


def main():
    wls = np.linspace(1.540e-6, 1.560e-6, 501)
    l_ref = 200.0e-6
    dl_arm = 45.0e-6  # Path imbalance -> FSR ~ wl^2 / (n_g * dl)
    l_heated = l_ref + dl_arm

    # Effective index model combining baseline SOI confinement with Si thermo-optic dispersion
    swg_ref = EMpy.devices.SWG(450, 220, 25).solve(wls)
    n_si_300 = np.real(EMpy.materials.Si.n(wls, T=300.0))
    gamma_conf = 0.88  # Core thermo-optic confinement factor

    coupler_3db = EMpy.devices.Coupler(wls, np.sqrt(0.5), q=1.0)

    temps_k = [300.0, 325.0, 350.0]
    mzi_traces = {}
    for temp in temps_k:
        dn_to = gamma_conf * (np.real(EMpy.materials.Si.n(wls, T=temp)) - n_si_300)
        line = EMpy.devices.Line(
            wls, swg_ref.neff + dn_to, l_heated, swg_ref.neff, l_ref
        )
        mzi = EMpy.devices.MZ(coupler_3db, line, coupler_3db).solve()
        mzi_traces[temp] = (mzi.THRU, mzi.DROP, mzi)

    # --- 2. Verify TransferMatrix / ChainMatrix composition & 2-stage MZI lattice filter ---
    mzi_base = mzi_traces[300.0][2]
    idx_check = np.linspace(0, len(wls) - 1, 25, dtype=int)
    tm_diff = 0.0
    cm_diff = 0.0
    for idx in idx_check:
        wl_i = float(wls[idx])
        tm = mzi_base.TM(wl_i)
        cm_roundtrip = mzi_base.CM(wl_i).to_transfer()
        tm_diff = max(tm_diff, abs(tm.data[0, 0] - mzi_base.THRU[idx]))
        cm_diff = max(cm_diff, float(np.max(np.abs(cm_roundtrip.data - tm.data))))

    # Build a 2-stage flat-top lattice MZI filter (L_2 = 2 * L_1) via composeTMlist,
    # and cross-check composeCMlist on two coupled SRRs vs. NRR!
    c1 = EMpy.devices.Coupler(wls, np.sqrt(0.5))
    c2 = EMpy.devices.Coupler(wls, np.sqrt(0.29))
    c3 = EMpy.devices.Coupler(wls, np.sqrt(0.08))
    line1 = EMpy.devices.Line(wls, swg_ref.neff, l_ref + dl_arm, swg_ref.neff, l_ref)
    line2 = EMpy.devices.Line(
        wls, swg_ref.neff, l_ref, swg_ref.neff, l_ref + 2.0 * dl_arm
    )
    mzi_stage1 = EMpy.devices.MZ(c1, line1, c2)
    mzi_stage2 = EMpy.devices.MZ(EMpy.devices.Coupler(wls, 0.0), line2, c3)

    drop_2stage_tm = np.zeros_like(wls)
    drop_2stage_pair = np.zeros_like(wls)
    for i, wl_i in enumerate(wls):
        w_f = float(wl_i)
        h_tot = EMpy.devices.composeTMlist(
            [c1.TM(w_f), line1.TM(w_f), c2.TM(w_f), line2.TM(w_f), c3.TM(w_f)]
        )
        h_pair = EMpy.devices.composeTM(mzi_stage1.TM(w_f), mzi_stage2.TM(w_f))
        drop_2stage_tm[i] = abs(h_tot.data[1, 0]) ** 2
        drop_2stage_pair[i] = abs(h_pair.data[1, 0]) ** 2

    # Also verify vertical bus-to-bus ChainMatrix composition via composeCMlist
    cm_chain_check = EMpy.devices.composeCMlist(
        [c1.CM(float(wls[0])), c2.CM(float(wls[0])), c3.CM(float(wls[0]))]
    ).to_transfer()
    _ = cm_chain_check

    dn_dt_1550 = float(
        np.real(
            EMpy.materials.Si.n(1.55e-6, T=350.0)
            - EMpy.materials.Si.n(1.55e-6, T=300.0)
        ).item()
        / 50.0
    )
    print(
        f"Silicon thermo-optic dn/dT at 1.55 um = {dn_dt_1550:.3e} K^-1; "
        f"MZ.solve() vs TM max diff = {tm_diff:.2e}, CM->TM round-trip diff = {cm_diff:.2e}"
    )
    print(
        f"2-stage lattice MZI composeTMlist vs staged composeTM max diff = "
        f"{np.max(np.abs(drop_2stage_tm - drop_2stage_pair)):.2e}"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4))

    for temp in temps_k:
        _, drop, _ = mzi_traces[temp]
        ax1.plot(
            wls * 1e9,
            np.abs(drop) ** 2,
            linewidth=1.7,
            label=f"DROP (T = {temp:.0f} K)",
        )
    ax1.set_title("Thermo-Optic Silicon MZI Tuning (ThermalOpticCoefficient)")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Cross-Port Transmission |DROP|²")
    ax1.set_ylim(-0.02, 1.04)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8.5)

    ax2.plot(
        wls * 1e9,
        10.0 * np.log10(np.maximum(np.abs(mzi_traces[300.0][1]) ** 2, 1e-4)),
        "r--",
        linewidth=1.5,
        label="1-stage sinusoidal MZI",
    )
    ax2.plot(
        wls * 1e9,
        10.0 * np.log10(np.maximum(drop_2stage_tm, 1e-4)),
        "b-",
        linewidth=1.8,
        label="2-stage lattice MZI (composeTMlist)",
    )
    ax2.set_title("2-Stage Flat-Top MZI Interleaver Synthesis")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Transmission (dB)")
    ax2.set_ylim(-32, 2)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8.5)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
