---
layout: default
title: Optical Ring Resonators & Photonic Filters in Python
description: Simulate all-pass (APRR), single-ring add-drop (SRR), N-ring CROW (NRR), and tunable coupled-resonator optical filters in Python with EMpy.
permalink: /ring-resonators-photonic-circuits/
---

## Optical Ring Resonators & Photonic Filters in Python

`EMpy.devices` models **optical ring resonators**, coupled-resonator optical waveguides (**CROW**), Mach–Zehnder interferometers (**MZI**), and dispersive **Fabry–Pérot etalons** using analytical transfer-matrix and chain-matrix formulations (`TransferMatrix`, `ChainMatrix`, `composeTMlist`, `composeCMlist`).

---

### Supported Ring Resonator & Device Models

- **`APRR` (All-Pass Ring Resonator):** Single bus-coupled ring for phase response, group delay $$\tau_g = -\frac{d\phi}{d\omega}$$, and chromatic dispersion via `EMpy.utils.group_delay_and_dispersion`.
- **`SRR` (Single-Ring Add–Drop Resonator):** Dual-bus microring filter supporting lossless unitary response ($$\lvert \mathrm{THRU} \rvert^2 + \lvert \mathrm{DROP} \rvert^2 = 1$$) and Yariv critical coupling ($$t_1 = t_2 a_{\mathrm{rt}} \implies \mathrm{THRU}(\lambda_{\mathrm{res}}) = 0$$).
- **`NRR` ($$N$$-Ring Coupled Resonator / CROW):** Arbitrary-order coupled microring cascades for box-like flat-top add–drop filters and Vernier spectra.
- **Barbarossa Tunable Coupled-Ring Filters (`T_TCRR`, `T_CRTCRR`, `T_CTCRR`):** MZI-assisted triple- and quadruple-coupler ring resonators with `coupling="optimum"` flat-top synthesis.
- **Dispersive Silicon Waveguides & Interferometers:** `SWG` (dispersive SOI strip waveguide effective index), `Coupler`, `Line`, thermo-optic `MZ` (`ThermalOpticCoefficient`), and `Etalon`.

---

### Quick Python Example: Critical Coupling & Group Delay in a Silicon Microring

```python
import numpy as np
import EMpy

wls = np.linspace(1.544e-6, 1.556e-6, 1201)
swg = EMpy.devices.SWG(450, 220, 25).solve(wls)  # 450x220 nm SOI waveguide
l_ring = 2.0 * np.pi * 10.0e-6                   # 10 um ring radius

# 18 dB/cm propagation loss -> complex effective index n_eff = n_r - i*kappa
dn_loss = EMpy.utils.loss_dBcm2rix(0.0, 18.0, 1.55e-6)
neff_lossy = swg.neff + dn_loss
a_rt = float(np.abs(np.exp(-1j * (2 * np.pi / 1.55e-6) * dn_loss * l_ring)))

# Single-Ring Add-Drop Resonator (SRR) at Yariv critical coupling
k2 = np.sqrt(0.04)  # 4% drop-bus power coupling
t2 = np.sqrt(1.0 - k2**2)
k1_crit = np.sqrt(1.0 - (t2 * a_rt) ** 2)

srr = EMpy.devices.SRR(
    EMpy.devices.Coupler(wls, k1_crit),
    EMpy.devices.Coupler(wls, k2),
    neff_lossy,
    0.5 * l_ring,
    0.5 * l_ring,
).solve()

# All-Pass Ring Resonator (APRR) group delay & chromatic dispersion
aprr = EMpy.devices.APRR(
    EMpy.devices.Coupler(wls, 1.65 * np.sqrt(1.0 - a_rt**2), 1.0),
    neff_lossy,
    l_ring,
).solve()
phi, tau_ps, disp_ps_nm = EMpy.utils.group_delay_and_dispersion(wls, aprr.THRU)
```

![Optical ring resonators in Python: APRR group delay, SRR critical coupling, and 3-ring NRR CROW filter]({{ '/images/devices_rings.png' | relative_url }})

---

### Benchmark Scripts & References

- [`ex_ring_resonators_aprr_srr_nrr.py`](https://github.com/lbolla/EMpy/blob/master/examples/devices/ex_ring_resonators_aprr_srr_nrr.py) — `APRR` group delay/dispersion, `SRR` critical coupling (`-30.6 dB` notch), and 3-ring `NRR` CROW filter (*Schwelb 2004; Yariv 2000; Little et al. 1997*).
- [`ex_tunable_coupled_ring_filters.py`](https://github.com/lbolla/EMpy/blob/master/examples/devices/ex_tunable_coupled_ring_filters.py) — Barbarossa `T_TCRR`, `T_CRTCRR`, and 4th-order `T_CTCRR` flat-top filter synthesis (*Barbarossa et al. 1995, 1997*).
- [`ex_thermo_optic_mzi.py`](https://github.com/lbolla/EMpy/blob/master/examples/devices/ex_thermo_optic_mzi.py) — Silicon thermo-optic `MZ` and 2-stage lattice MZI.
- [`ex_dispersive_lossy_etalon.py`](https://github.com/lbolla/EMpy/blob/master/examples/devices/ex_dispersive_lossy_etalon.py) — Dispersive Si wafer `Etalon` cross-checked against `IsotropicTransferMatrix`.

**See also:** [Methodology & References]({{ '/methodology/#devices-materials' | relative_url }}) · [All Examples]({{ '/examples/#catalog' | relative_url }})
