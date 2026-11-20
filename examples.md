---
layout: default
title: Examples Gallery
permalink: /examples/
---

## Examples Gallery

`EMpy 3.0` includes **22 self-contained benchmark scripts** organized across 7 physical domains in [`examples/`](https://github.com/lbolla/EMpy/tree/master/examples). Every script includes bibliographic literature citations, prints quantitative cross-check metrics, and runs from any working directory:

```bash
uv run examples/transfer_matrix/ex_kretschmann_spr_lossy_vs_lossless.py
```

**Jump to domain:** [1. Transfer Matrix](#transfer-matrix) · [2. Liquid Crystals](#liquid-crystals) · [3. RCWA Gratings](#rcwa) · [4. Mode Solvers](#modesolvers) · [5. Waveguide Circuits](#waveguide-circuits) · [6. Photonic Devices](#devices) · [7. Vector Scattering](#scattering) · [Complete 22-Script Index](#catalog)

---

## 1. Transfer Matrix — Thin Films, DBRs & Plasmonics {#transfer-matrix}

### Quarter-Wave AR Coating & TiO₂/SiO₂ Distributed Bragg Reflector
**Source:** [`examples/transfer_matrix/ex_ar_coating_and_dbr.py`](https://github.com/lbolla/EMpy/blob/master/examples/transfer_matrix/ex_ar_coating_and_dbr.py) · *Macleod (2010); Born & Wolf (1999)*

Simulates a single-layer MgF₂ quarter-wave anti-reflection coating on Sellmeier `BK7` crown glass at `10°` incidence alongside an 8-pair TiO₂ / SiO₂ DBR centered at `1.55 µm`, matching Macleod's analytical stopband width (`Δλ = 514.4 nm`, peak `R = 0.9994`).

```python
ar_stack = EMpy.utils.Multilayer([
    EMpy.utils.Layer(EMpy.materials.Air, np.inf),
    EMpy.utils.Layer(mgf2, wl0 / (4.0 * 1.38)),
    EMpy.utils.Layer(EMpy.materials.BK7, np.inf),
])
sol_ar = EMpy.transfer_matrix.IsotropicTransferMatrix(
    ar_stack, EMpy.utils.deg2rad(10.0)
).solve(wls)
```

![AR coating and DBR reflectance]({{ '/images/transfer_matrix_ar_dbr.png' | relative_url }})

### Kretschmann Surface-Plasmon Resonance & $$\kappa \to 0^+$$ TIR Limit
**Source:** [`examples/transfer_matrix/ex_kretschmann_spr_lossy_vs_lossless.py`](https://github.com/lbolla/EMpy/blob/master/examples/transfer_matrix/ex_kretschmann_spr_lossy_vs_lossless.py) · *Kretschmann & Raether (1968)*

Reproduces attenuated total reflection (ATR) in a `BK7 / 47 nm Au / Air` prism setup at `632.8 nm`, capturing the sharp TM plasmonic dip above the critical angle ($$\theta_c = 41.3^\circ$$, $$\theta_{\mathrm{SPR}} \approx 43.8^\circ$$), exact equivalence between `IsotropicTransferMatrix` and `AnisotropicTransferMatrix` (`max diff = 2.5e-14`), and continuous $$R_p + T_p \to 1$$ restoration as extinction $$\kappa \to 0^+$$.

![Kretschmann SPR and lossy-to-lossless limit]({{ '/images/transfer_matrix_spr.png' | relative_url }})

*Also in `transfer_matrix/`:* [`ex_anisotropic_waveplate_and_limits.py`](https://github.com/lbolla/EMpy/blob/master/examples/transfer_matrix/ex_anisotropic_waveplate_and_limits.py) (45° quartz waveplate vs. Jones calculus & 3D `euler_rotate` $$\Delta n \to 0$$ limit) and [`ex_laser_etch_monitor.py`](https://github.com/lbolla/EMpy/blob/master/examples/transfer_matrix/ex_laser_etch_monitor.py) (in-situ 670 nm RIE endpoint interferometry on an absorbing GaAs/AlGaAs VCSEL stack).

---

## 2. Liquid Crystals — Twisted Nematic & Cholesteric Cells {#liquid-crystals}

### 90° Twisted Nematic Cell: Gooch–Tarry Minima & Oseen–Frank Switching
**Source:** [`examples/liquid_crystals/ex_twisted_nematic_gooch_tarry.py`](https://github.com/lbolla/EMpy/blob/master/examples/liquid_crystals/ex_twisted_nematic_gooch_tarry.py) · *Gooch & Tarry (1975); de Gennes & Prost (1993)*

Solves the coupled 1D Oseen–Frank elastic and Laplace electrostatic BVP in `LiquidCrystalCell` across applied voltages (`0 → 5 V`), reproducing Gooch & Tarry's analytical off-state rotatory power ($$u = \sqrt{3}, \sqrt{15}$$ Mauguin maxima), the electro-optic Freedericksz threshold ($$V_{\mathrm{th}} = 2.67\;\mathrm{V}$$), mid-plane director tilt $$\theta(z)$$, and voltage-dependent static `capacitance()`.

```python
cell = EMpy.utils.LiquidCrystalCell(
    lc, voltage=3.5, t_tot=4.8e-6, t_anchoring=0.0,
    pretilt=EMpy.utils.deg2rad(2.0), totaltwist=np.pi / 2.0, nlayers=48,
)
ml = EMpy.utils.Multilayer([EMpy.utils.Layer(amb, np.inf), cell, EMpy.utils.Layer(amb, np.inf)])
sol = EMpy.transfer_matrix.AnisotropicTransferMatrix(ml, 0.0, 0.0).solve(wl)
```

![Twisted nematic Gooch-Tarry and Oseen-Frank switching]({{ '/images/liquid_crystals_tn.png' | relative_url }})

*Also in `liquid_crystals/`:* [`ex_chiral_lc_bragg_and_datafile.py`](https://github.com/lbolla/EMpy/blob/master/examples/liquid_crystals/ex_chiral_lc_bragg_and_datafile.py) (selective circular-polarization Bragg band $$[n_O P, n_E P]$$ in a chiral cholesteric helix & tabulated `data/lc_data.dat` interpolation).

---

## 3. RCWA — Surface-Relief, Anisotropic & Dimer BIC Gratings {#rcwa}

### Symmetry-Protected BIC & Quasi-BIC Fano Resonances in Dimer Gratings
**Source:** [`examples/rcwa/ex_double_grating_dimer_bic.py`](https://github.com/lbolla/EMpy/blob/master/examples/rcwa/ex_double_grating_dimer_bic.py) · *Koshelev et al., PRL (2018); Moharam et al. (1995)*

Compares a symmetric period-halved Si₃N₄ dimer grating (`SymmetricDoubleGrating`, dark symmetry-protected bound state in the continuum) against width-perturbed dimers (`AsymmetricDoubleGrating`), extracting Fano linewidths via `EMpy.utils.find_peaks` to verify Koshelev's inverse-square radiative quality-factor law `Q ∝ α⁻²`.

```python
ml_asym = EMpy.utils.Multilayer([
    EMpy.utils.Layer(air, np.inf),
    EMpy.utils.AsymmetricDoubleGrating(sin, sin, air, dc1, dc2, dc_spacer, pitch, h_grating),
    EMpy.utils.Layer(sin, h_wg),
    EMpy.utils.Layer(sio2, np.inf),
])
sol = EMpy.RCWA.IsotropicRCWA(ml_asym, 0.0, 0.0, np.pi / 2.0, np.pi / 2.0, n=4).solve(wls)
```

![Dimer metasurface symmetry-protected BIC to quasi-BIC]({{ '/images/rcwa_dimer_bic.png' | relative_url }})

### 3D Anisotropic Guided-Mode Resonance vs. Rytov Effective-Medium Theory
**Source:** [`examples/rcwa/ex_anisotropic_guided_mode_resonance.py`](https://github.com/lbolla/EMpy/blob/master/examples/rcwa/ex_anisotropic_guided_mode_resonance.py) · *Wang & Magnusson (1993); Glytsis & Gaylord (1990); Li (1998); Rytov (1956)*

Solves a subwavelength 3D Euler-rotated birefringent SiN waveguide grating with `AnisotropicRCWA`, matching zero-order Rytov effective-medium `AnisotropicTransferMatrix` away from resonance while capturing the `99.2%` leaky guided-mode resonance peak and resonant cross-polarized conversion.

![Anisotropic guided-mode resonance vs Rytov EMT]({{ '/images/rcwa_anisotropic_gmr.png' | relative_url }})

*Also in `rcwa/`:* [`ex_moharam_binary_grating.py`](https://github.com/lbolla/EMpy/blob/master/examples/rcwa/ex_moharam_binary_grating.py) (Littrow groove-depth sweep & Lalanne TM harmonic convergence on lossless dielectric and lossy Au gratings) and [`ex_conical_diffraction_and_rayleigh.py`](https://github.com/lbolla/EMpy/blob/master/examples/rcwa/ex_conical_diffraction_and_rayleigh.py) (3D conical Wood–Rayleigh cut-off anomalies & `AnisotropicRCWA → IsotropicRCWA` limits).

---

## 4. Waveguide Mode Solvers — FD, Stokes Maps, PMLs & FMM {#modesolvers}

### Complex-Coordinate-Stretching PMLs for Leaky Tunneling Waveguides
**Source:** [`examples/modesolvers/ex_pml_leaky_waveguide.py`](https://github.com/lbolla/EMpy/blob/master/examples/modesolvers/ex_pml_leaky_waveguide.py) · *Chew & Weedon (1994); Ghatak et al. (1987); Fallahkhair et al. (2008)*

Applies `EMpy.modesolvers.FD.stretchmesh` (`factor = 1.0 - 1.8j`, passive $$\operatorname{Im}(\text{factor}) < 0$$) to a depressed-cladding W-type leaky waveguide where core light tunnels across a low-index barrier of width $$t_b$$ into a higher-index outer region, matching the exact 5-layer analytical leaky-wave dispersion across nearly two decades of radiation loss (`85 → 2268 dB/cm`).

```python
xx, yy, _, _, _, _ = EMpy.modesolvers.FD.stretchmesh(
    x_raw, y_raw, [0, 0, 28, 28], 1.0 - 1.8j, method="PPPP"
)
sol = EMpy.modesolvers.FD.VFDModeSolver(wl, xx, yy, eps_w, "SS00").solve(2, 1e-11, guess=1.45)
loss_db_cm = EMpy.utils.rix2losses(sol.modes[0].neff, wl)[4]
```

![Leaky waveguide tunneling loss with stretchmesh PML]({{ '/images/modesolvers_pml_leaky.png' | relative_url }})

### Asymmetric Etched SOI Rib Waveguide: `TM₀–TE₁` Anti-Crossing & Stokes Maps
**Source:** [`examples/modesolvers/ex_rib_waveguide_stokes.py`](https://github.com/lbolla/EMpy/blob/master/examples/modesolvers/ex_rib_waveguide_stokes.py) · *Fallahkhair et al. (2008); Dai & Bowers (2011)*

Tracks polarization hybridization across the `TM₀–TE₁` avoided crossing of a one-sided etched SOI rib waveguide via `FDMode.TEfrac()` and maps the spatially resolved diagonal Stokes parameter `S₂ / S₀`.

![Asymmetric SOI rib avoided crossing and Stokes S2 map]({{ '/images/modesolvers_rib_stokes.png' | relative_url }})

*Also in `modesolvers/`:* [`ex_soi_channel_svfd_vfd_fmm.py`](https://github.com/lbolla/EMpy/blob/master/examples/modesolvers/ex_soi_channel_svfd_vfd_fmm.py) (`SVFDModeSolver` vs. `VFDModeSolver` vs. `FMM2d` + `CrossSection` meshing), [`ex_anisotropic_and_magnetooptic_wg.py`](https://github.com/lbolla/EMpy/blob/master/examples/modesolvers/ex_anisotropic_and_magnetooptic_wg.py) (LC-clad `Δn → 0` limit & Ce:TbIG magneto-optic gyrotropic spin density `S₃ / S₀`), and [`ex_fmm1d_fmm2d_lossy_and_units.py`](https://github.com/lbolla/EMpy/blob/master/examples/modesolvers/ex_fmm1d_fmm2d_lossy_and_units.py) (`FMM1d` vs. exact asymmetric slab, lossy `FMM2d`, & SI-meter vs. micron scale invariance).

---

## 5. Waveguide Circuits — Mode-Matching S-Matrices & MMI Cascades {#waveguide-circuits}

### Waveguide Step Discontinuity (`passive` vs. `reciprocal`) & MMI Self-Imaging
**Sources:** [`ex_waveguide_step_junction.py`](https://github.com/lbolla/EMpy/blob/master/examples/waveguide_circuits/ex_waveguide_step_junction.py) & [`ex_mmi_and_fabry_perot_cascade.py`](https://github.com/lbolla/EMpy/blob/master/examples/waveguide_circuits/ex_mmi_and_fabry_perot_cascade.py) · *Soldano & Pennings (1995); Redheffer (1960); Vassallo (1991)*

Demonstrates `SimpleJoint` (`method="passive"` guaranteeing $$\lVert S \rVert_2 \le 1$$ vs. `method="reciprocal"` enforcing exact $$S = S^T$$ at finite mode count $$N$$), cross-solver `VFD ↔ FMM2d` mode matching, $$1 \times 2$$ Multi-Mode Interference (MMI) self-imaging at $$z = \tfrac{3}{8} L_\pi$$ and $$\tfrac{3}{4} L_\pi$$, and Redheffer star-product `GenericDevice` cascading across sub-cutoff evanescent barriers ($$\operatorname{Im}(n_{\mathrm{eff}}) < -1.5$$).

```python
mmi_cascade = EMpy.modesolvers.geometries.GenericDevice([
    EMpy.modesolvers.geometries.SWG(sol_in, l_in),
    EMpy.modesolvers.geometries.SimpleJoint(sol_in, sol_mmi, method="passive"),
    EMpy.modesolvers.geometries.SWG(sol_mmi, 0.90 * l_pi),
])
```

![Waveguide step junction passive vs reciprocal mode matching]({{ '/images/circuits_step_junction.png' | relative_url }})

![MMI self-imaging and sub-cutoff Redheffer cascade]({{ '/images/circuits_mmi_fp.png' | relative_url }})

---

## 6. Photonic Devices — Ring Resonators, MZIs & Etalons {#devices}

### All-Pass Dispersion, Critical Coupling & 3-Ring CROW Filters
**Source:** [`examples/devices/ex_ring_resonators_aprr_srr_nrr.py`](https://github.com/lbolla/EMpy/blob/master/examples/devices/ex_ring_resonators_aprr_srr_nrr.py) · *Schwelb (2004); Yariv (2000); Little et al. (1997)*

Evaluates `APRR` resonant group delay via `group_delay_and_dispersion`, verifies Yariv's critical-coupling extinction (`-30.6 dB` THRU notch) vs. lossless unitarity (`|THRU|² + |DROP|² = 1`) in `SRR`, and synthesizes a flat-top 3-ring `NRR` add–drop filter on dispersive SOI `SWG` waveguides.

```python
srr_crit = EMpy.devices.SRR(
    EMpy.devices.Coupler(wls, k1_crit),
    EMpy.devices.Coupler(wls, k2),
    neff_lossy, 0.5 * l_ring, 0.5 * l_ring,
).solve()
phi, tau_ps, disp_ps_nm = EMpy.utils.group_delay_and_dispersion(wls, aprr.THRU)
```

![APRR group delay, SRR critical coupling, and 3-ring NRR filter]({{ '/images/devices_rings.png' | relative_url }})

*Also in `devices/`:* [`ex_tunable_coupled_ring_filters.py`](https://github.com/lbolla/EMpy/blob/master/examples/devices/ex_tunable_coupled_ring_filters.py) (Barbarossa `T_TCRR`, `T_CRTCRR`, & `T_CTCRR` `"optimum"` flat-top synthesis), [`ex_thermo_optic_mzi.py`](https://github.com/lbolla/EMpy/blob/master/examples/devices/ex_thermo_optic_mzi.py) (silicon `ThermalOpticCoefficient` MZI tuning & 2-stage lattice MZI via `composeTMlist`), and [`ex_dispersive_lossy_etalon.py`](https://github.com/lbolla/EMpy/blob/master/examples/devices/ex_dispersive_lossy_etalon.py) (dispersive Si wafer `Etalon` matching `IsotropicTransferMatrix` to `8.9e-16`).

---

## 7. Vector Scattering — 3D Huygens–Kottler Diffraction {#scattering}

### Circular Aperture E/H-Plane Airy Pattern & Fresnel Near-Field Decay
**Source:** [`examples/scattering/ex_kottler_aperture_diffraction.py`](https://github.com/lbolla/EMpy/blob/master/examples/scattering/ex_kottler_aperture_diffraction.py) · *Kottler (1923); Stratton & Chu (1939); Balanis (2012)*

Integrates equivalent electric and magnetic Huygens surface currents across a circular aperture (`2a = 5 λ₀`) using `currentsScatteringKottler` and `stack`, matching the Fraunhofer Huygens–Airy radiation pattern in both E- and H-planes and reproducing on-axis Fresnel interference oscillations in vacuum and passive lossy dielectric media (`epsr = (1.4 - 0.03j)²`).

```python
fld_e = EMpy.scattering.currentsScatteringKottler(
    p_eplane, j_free, m_free, q_ap, ds_ap, f0, epsr=1.0
)
```

![Vector Kottler circular aperture diffraction vs Airy pattern]({{ '/images/scattering_kottler_airy.png' | relative_url }})

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
