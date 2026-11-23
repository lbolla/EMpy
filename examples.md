---
layout: default
title: Examples & Topic Guides
description: Browse EMpy's 7 dedicated Python photonics topic guides and complete index of 22 self-contained benchmark scripts across transfer matrices, RCWA, mode solvers, S-matrix circuits, ring resonators, liquid crystals, and vector diffraction.
permalink: /examples/
---

## Examples & Topic Guides

`EMpy 3.0` includes **22 self-contained benchmark scripts** organized across 7 physical domains in [`examples/`](https://github.com/lbolla/EMpy/tree/master/examples). Every script includes bibliographic literature citations, prints quantitative cross-check metrics, and runs from any working directory:

```bash
uv run examples/devices/ex_ring_resonators_aprr_srr_nrr.py
```

---

## Topic Guides {#topics}

| Domain | Topic Guide | Key `EMpy` Modules & Benchmarks |
| :--- | :--- | :--- |
| **Ring Resonators & Devices** | [Optical Ring Resonators & Photonic Filters]({{ '/ring-resonators-photonic-circuits/' | relative_url }}) | `APRR`, `SRR`, `NRR` CROWs, Barbarossa `T_TCRR` / `T_CRTCRR` / `T_CTCRR`, thermo-optic `MZ`, `Etalon` |
| **Transfer Matrix** | [Transfer Matrix Method (2 × 2 & 4 × 4 Yeh–Berreman)]({{ '/transfer-matrix-multilayers/' | relative_url }}) | `IsotropicTransferMatrix`, `AnisotropicTransferMatrix`, AR/DBRs, Kretschmann SPR, waveplates |
| **RCWA Gratings** | [Rigorous Coupled-Wave Analysis (RCWA)]({{ '/rcwa-gratings/' | relative_url }}) | `IsotropicRCWA`, `AnisotropicRCWA`, conical diffraction, GMR filters, dimer BICs |
| **Waveguide Mode Solvers** | [Optical Waveguide Mode Solvers (FD, PML & FMM)]({{ '/waveguide-mode-solvers/' | relative_url }}) | `SVFDModeSolver`, `VFDModeSolver`, `FMM1d`, `FMM2d`, `stretchmesh` PMLs, Stokes polarization maps |
| **Waveguide Circuits** | [Waveguide S-Matrix Circuits & Mode-Matching]({{ '/waveguide-s-matrix-circuits/' | relative_url }}) | `SimpleJoint` (`passive` / `reciprocal`), `SWG`, `GenericDevice`, Redheffer star-product MMI cascades |
| **Liquid Crystals** | [Liquid-Crystal Electro-Optics & Oseen–Frank BVP]({{ '/liquid-crystal-optics/' | relative_url }}) | `LiquidCrystalCell`, 90° twisted nematic Gooch–Tarry minima, cholesteric Bragg bands |
| **Vector Scattering** | [3D Vector Diffraction & Huygens–Kottler Scattering]({{ '/vector-diffraction-scattering/' | relative_url }}) | `currentsScatteringKottler`, circular aperture Airy pattern & lossy Fresnel propagation |

---

## Complete 22-Script Index {#catalog}

| Domain | Script | Benchmark & Key APIs |
| :--- | :--- | :--- |
| **Transfer Matrix** | [`ex_ar_coating_and_dbr.py`](https://github.com/lbolla/EMpy/blob/master/examples/transfer_matrix/ex_ar_coating_and_dbr.py) | Quarter-wave MgF₂ AR coating & TiO₂/SiO₂ DBR (`IsotropicTransferMatrix`, `BK7`, `SiO2`) |
| | [`ex_kretschmann_spr_lossy_vs_lossless.py`](https://github.com/lbolla/EMpy/blob/master/examples/transfer_matrix/ex_kretschmann_spr_lossy_vs_lossless.py) | Kretschmann Au SPR dip & `κ → 0⁺` TIR limit (`IsotropicTM` vs. `AnisotropicTM`) |
| | [`ex_anisotropic_waveplate_and_limits.py`](https://github.com/lbolla/EMpy/blob/master/examples/transfer_matrix/ex_anisotropic_waveplate_and_limits.py) | 45° quartz waveplate vs. Jones calculus & biaxial `Δn → 0` (`AnisotropicTransferMatrix`, `euler_rotate`) |
| | [`ex_laser_etch_monitor.py`](https://github.com/lbolla/EMpy/blob/master/examples/transfer_matrix/ex_laser_etch_monitor.py) | In-situ 670 nm RIE reflectometry on GaAs/AlGaAs VCSEL DBR (`RefractiveIndex(n0_func=...)`) |
| **Liquid Crystals** | [`ex_twisted_nematic_gooch_tarry.py`](https://github.com/lbolla/EMpy/blob/master/examples/liquid_crystals/ex_twisted_nematic_gooch_tarry.py) | 90° TN Gooch–Tarry minima, Oseen–Frank switching & `capacitance()` (`LiquidCrystalCell`) |
| | [`ex_chiral_lc_bragg_and_datafile.py`](https://github.com/lbolla/EMpy/blob/master/examples/liquid_crystals/ex_chiral_lc_bragg_and_datafile.py) | Cholesteric chiral Bragg reflection band & tabulated `lc_data.dat` cell (`LiquidCrystal`) |
| **RCWA** | [`ex_moharam_binary_grating.py`](https://github.com/lbolla/EMpy/blob/master/examples/rcwa/ex_moharam_binary_grating.py) | Littrow groove-depth ETM sweep & TE/TM order convergence (`IsotropicRCWA`, `BinaryGrating`) |
| | [`ex_conical_diffraction_and_rayleigh.py`](https://github.com/lbolla/EMpy/blob/master/examples/rcwa/ex_conical_diffraction_and_rayleigh.py) | Conical Wood–Rayleigh anomalies & `AnisotropicRCWA → IsotropicRCWA` (`dispersion_relation_*`) |
| | [`ex_anisotropic_guided_mode_resonance.py`](https://github.com/lbolla/EMpy/blob/master/examples/rcwa/ex_anisotropic_guided_mode_resonance.py) | 3D birefringent SiN GMR filter vs. Rytov EMT (`AnisotropicRCWA`, `AnisotropicTransferMatrix`) |
| | [`ex_double_grating_dimer_bic.py`](https://github.com/lbolla/EMpy/blob/master/examples/rcwa/ex_double_grating_dimer_bic.py) | Dimer dark BIC vs. quasi-BIC `Q ∝ α⁻²` (`SymmetricDoubleGrating`, `AsymmetricDoubleGrating`, `find_peaks`) |
| **Mode Solvers** | [`ex_soi_channel_svfd_vfd_fmm.py`](https://github.com/lbolla/EMpy/blob/master/examples/modesolvers/ex_soi_channel_svfd_vfd_fmm.py) | SOI strip `SVFDModeSolver` vs. `VFDModeSolver` vs. `FMM2d` (`CrossSection`, `overlap`) |
| | [`ex_anisotropic_and_magnetooptic_wg.py`](https://github.com/lbolla/EMpy/blob/master/examples/modesolvers/ex_anisotropic_and_magnetooptic_wg.py) | LC-clad `Δn → 0` limit & Ce:TbIG magneto-optic gyrotropic spin `S₃ / S₀` (`VFDModeSolver` 5-comp) |
| | [`ex_rib_waveguide_stokes.py`](https://github.com/lbolla/EMpy/blob/master/examples/modesolvers/ex_rib_waveguide_stokes.py) | Asymmetric etched SOI rib `TM₀–TE₁` anti-crossing & Stokes `S₂ / S₀` (`TEfrac`) |
| | [`ex_pml_leaky_waveguide.py`](https://github.com/lbolla/EMpy/blob/master/examples/modesolvers/ex_pml_leaky_waveguide.py) | W-type leaky waveguide tunneling loss vs. exact 5-layer theory (`stretchmesh`, `rix2losses`) |
| | [`ex_fmm1d_fmm2d_lossy_and_units.py`](https://github.com/lbolla/EMpy/blob/master/examples/modesolvers/ex_fmm1d_fmm2d_lossy_and_units.py) | `FMM1d` vs. exact slab, lossy `FMM2d` attenuation, & m vs. µm invariance (`loss_*2rix`) |
| **Waveguide Circuits** | [`ex_waveguide_step_junction.py`](https://github.com/lbolla/EMpy/blob/master/examples/waveguide_circuits/ex_waveguide_step_junction.py) | Step junction `passive` vs. `reciprocal` S-matrices & `VFD ↔ FMM2d` (`SimpleJoint`, `interface_matrix`) |
| | [`ex_mmi_and_fabry_perot_cascade.py`](https://github.com/lbolla/EMpy/blob/master/examples/waveguide_circuits/ex_mmi_and_fabry_perot_cascade.py) | `1 × 2` MMI self-imaging & sub-cutoff Redheffer Fabry–Pérot cavity (`SWG`, `GenericDevice`) |
| **Devices** | [`ex_ring_resonators_aprr_srr_nrr.py`](https://github.com/lbolla/EMpy/blob/master/examples/devices/ex_ring_resonators_aprr_srr_nrr.py) | `APRR` dispersion, `SRR` critical coupling, & 3-ring `NRR` (`group_delay_and_dispersion`) |
| | [`ex_tunable_coupled_ring_filters.py`](https://github.com/lbolla/EMpy/blob/master/examples/devices/ex_tunable_coupled_ring_filters.py) | Barbarossa `T_TCRR`, `T_CRTCRR`, & `T_CTCRR` flat-top synthesis (`coupling="optimum"`) |
| | [`ex_thermo_optic_mzi.py`](https://github.com/lbolla/EMpy/blob/master/examples/devices/ex_thermo_optic_mzi.py) | Silicon thermo-optic `MZ` & 2-stage lattice MZI (`ThermalOpticCoefficient`, `composeTMlist`) |
| | [`ex_dispersive_lossy_etalon.py`](https://github.com/lbolla/EMpy/blob/master/examples/devices/ex_dispersive_lossy_etalon.py) | Dispersive Si wafer `Etalon` vs. `IsotropicTransferMatrix` & lossy `κ → 0⁺` cavity |
| **Scattering** | [`ex_kottler_aperture_diffraction.py`](https://github.com/lbolla/EMpy/blob/master/examples/scattering/ex_kottler_aperture_diffraction.py) | 3D vector circular aperture Airy pattern & lossy Fresnel axial decay (`currentsScatteringKottler`, `stack`) |
