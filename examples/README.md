# EMpy Examples Catalog

All example scripts are self-contained, resolve data files relative to `__file__` (so they can be run from any working directory), print quantitative physical summaries, and include literature citations in their module docstrings.

Run any script with `uv` from the repository root:

```bash
uv run examples/transfer_matrix/ex_kretschmann_spr_lossy_vs_lossless.py
```

Or run headless (without opening GUI figure windows):

```bash
MPLBACKEND=Agg uv run examples/rcwa/ex_moharam_binary_grating.py
```

---

## Directory Overview (22 Scripts Across 7 Domains)

### 1. `transfer_matrix/` — Isotropic & 4 × 4 Yeh–Berreman Multilayers

| Script | Physics, Literature & Cross-Checks | Key `EMpy` APIs |
| :--- | :--- | :--- |
| `ex_ar_coating_and_dbr.py` | Quarter-wave MgF₂ anti-reflection coating on Sellmeier BK7 & 8-pair TiO₂/SiO₂ DBR vs. Macleod stopband formula (Macleod 2010; Born & Wolf 1999) | `IsotropicTransferMatrix`, `BK7`, `SiO2`, `Multilayer` |
| `ex_kretschmann_spr_lossy_vs_lossless.py` | Kretschmann prism/Au/air surface-plasmon resonance ATR dip & continuous `κ → 0⁺` TIR limit, cross-checking isotropic vs. anisotropic TMM (Kretschmann & Raether 1968) | `IsotropicTransferMatrix`, `AnisotropicTransferMatrix`, `snell` |
| `ex_anisotropic_waveplate_and_limits.py` | 45° quartz half-wave plate vs. Jones calculus & 3D Euler-rotated biaxial multilayer `Δn → 0` convergence (Berreman 1972; Yeh 1979) | `AnisotropicTransferMatrix`, `EpsilonTensor`, `euler_rotate` |
| `ex_laser_etch_monitor.py` | In-situ 670 nm RIE laser reflectometry endpoint detection on an absorbing GaAs/AlGaAs VCSEL DBR stack (Bacher et al. 1992) | `IsotropicTransferMatrix`, `RefractiveIndex(n0_func=...)`, `data/nk_gaas_algaas.py` |

### 2. `liquid_crystals/` — Electro-Optic & Chiral Liquid-Crystal Cells

| Script | Physics, Literature & Cross-Checks | Key `EMpy` APIs |
| :--- | :--- | :--- |
| `ex_twisted_nematic_gooch_tarry.py` | 90° Twisted Nematic (TN) cell Mauguin minima vs. Gooch–Tarry formula, Oseen–Frank voltage switching (`0 → 6 V`) vs. Freedericksz threshold, and cell capacitance (Gooch & Tarry 1975; de Gennes & Prost 1993) | `LiquidCrystalCell`, `get_10400_000_100`, `capacitance()` |
| `ex_chiral_lc_bragg_and_datafile.py` | Cholesteric chiral Bragg selective circular-polarization reflection (`[n_O P, n_E P]`, de Vries 1951) & tabulated voltage-profile interpolation from `data/lc_data.dat` (Yeh & Gu 2009) | `LiquidCrystal`, `LiquidCrystalCell`, `AnisotropicTransferMatrix` |

### 3. `rcwa/` — Planar, Conical & Anisotropic Grating Diffraction

| Script | Physics, Literature & Cross-Checks | Key `EMpy` APIs |
| :--- | :--- | :--- |
| `ex_moharam_binary_grating.py` | Littrow binary grating groove-depth sweep (ETM stability) & TE/TM harmonic truncation convergence on lossless dielectric and lossy Au gratings (Moharam et al. 1995; Lalanne & Morris 1996) | `IsotropicRCWA`, `BinaryGrating` |
| `ex_conical_diffraction_and_rayleigh.py` | Conical diffraction (`δ ≠ 0`) across Wood–Rayleigh passing-off anomalies & `AnisotropicRCWA` → `IsotropicRCWA` (`Δn → 0`, `κ → 0⁺`) (Rayleigh 1907; Glytsis & Gaylord 1990) | `IsotropicRCWA`, `AnisotropicRCWA`, `dispersion_relation_ordinary`, `dispersion_relation_extraordinary` |
| `ex_anisotropic_guided_mode_resonance.py` | 3D Euler-rotated birefringent SiN waveguide grating guided-mode resonance (GMR) vs. Rytov subwavelength effective-medium TMM (Wang & Magnusson 1993; Li 1998; Rytov 1956) | `AnisotropicRCWA`, `AnisotropicTransferMatrix`, `find_peaks` |
| `ex_double_grating_dimer_bic.py` | Silicon dimer metasurface symmetry-protected dark BIC vs. symmetry-broken quasi-BIC Fano resonance & `Q ∝ α⁻²` scaling (Koshelev et al. 2018; Hsu et al. 2016) | `SymmetricDoubleGrating`, `AsymmetricDoubleGrating`, `find_peaks` |

### 4. `modesolvers/` — Finite-Difference, PML & Film Mode Matching Waveguides

| Script | Physics, Literature & Cross-Checks | Key `EMpy` APIs |
| :--- | :--- | :--- |
| `ex_soi_channel_svfd_vfd_fmm.py` | SOI strip waveguide comparing scalar/semi-vectorial `SVFDModeSolver`, full-vectorial `VFDModeSolver` (`CrossSection` grid), and `FMM2d` + cross-solver mode overlaps (Fallahkhair et al. 2008; Sudbø 1993; Stern 1988) | `SVFDModeSolver`, `VFDModeSolver`, `FMM2d`, `CrossSection`, `Slice` |
| `ex_anisotropic_and_magnetooptic_wg.py` | Birefringent LC-clad waveguide (`Δn → 0` limit) & non-reciprocal Ce:TbIG magneto-optic gyrotropic waveguide Stokes `S₃/S₀` spin density (Fallahkhair et al. 2008; Bi et al. 2011; Ghosh et al. 2012) | `VFDModeSolver` (5-component `(nx, ny, 5)` tensor) |
| `ex_rib_waveguide_stokes.py` | Asymmetric etched SOI rib waveguide `TM₀–TE₁` polarization anti-crossing & spatial Stokes `S₂/S₀` hybridization map (Dai & Bowers 2011; Rahman et al. 2001) | `VFDModeSolver`, `FDMode.TEfrac()` |
| `ex_pml_leaky_waveguide.py` | W-type depressed-cladding leaky waveguide tunneling radiation loss with `stretchmesh` complex-coordinate PML vs. exact 5-layer leaky dispersion (Chew & Weedon 1994; Ghatak et al. 1987) | `stretchmesh`, `VFDModeSolver`, `rix2losses` |
| `ex_fmm1d_fmm2d_lossy_and_units.py` | `FMM1d` vs. exact asymmetric slab dispersion, `FMM2d` lossy `κ → 0⁺` modal attenuation, and SI-meter (`1.55e-6`) vs. micron (`1.55`) scale invariance (Sudbø 1993, 1994; Vassallo 1991) | `FMM1d`, `FMM2d`, `Boundary`, `loss_dBcm2rix`, `loss_cm2rix`, `loss_m2rix`, `rix2losses` |

### 5. `waveguide_circuits/` — Mode-Matching S-Matrices & Multi-Element Cascades

| Script | Physics, Literature & Cross-Checks | Key `EMpy` APIs |
| :--- | :--- | :--- |
| `ex_waveguide_step_junction.py` | Abrupt waveguide step junction comparing `SimpleJoint` (`method="passive"` vs. `"reciprocal"`), mode-count convergence, passivity/reciprocity invariants, and cross-solver `VFD ↔ FMM2d` matching (Vassallo 1991; Eleftheriades et al. 1994) | `SimpleJoint`, `interface_matrix`, `S2T`, `T2S` |
| `ex_mmi_and_fabry_perot_cascade.py` | `1 × 2` Multi-Mode Interference (MMI) self-imaging vs. Soldano `L_π` beat length & sub-cutoff waveguide Fabry–Pérot cavity stabilized via Redheffer star-product cascading (Soldano & Pennings 1995; Redheffer 1960) | `SWG`, `SimpleJoint`, `GenericDevice` |

### 6. `devices/` — Ring Resonators, Interferometers & Fabry–Pérot Etalons

| Script | Physics, Literature & Cross-Checks | Key `EMpy` APIs |
| :--- | :--- | :--- |
| `ex_ring_resonators_aprr_srr_nrr.py` | `APRR` group delay/dispersion, `SRR` critical coupling (`|THRU| = 0` vs. lossless), and 3-ring `NRR` flat-top CROW filter (Schwelb 2004; Yariv 2000; Little et al. 1997) | `Coupler`, `APRR`, `SRR`, `NRR`, `SWG`, `group_delay_and_dispersion` |
| `ex_tunable_coupled_ring_filters.py` | MZI-assisted `T_TCRR`, `T_CRTCRR`, and 4th-order `T_CTCRR` (`coupling="optimum"` flat-top synthesis & `find_peaks` FWHM) (Barbarossa et al. 1995, 1997) | `T_TCRR`, `T_CRTCRR`, `T_CTCRR`, `find_peaks` |
| `ex_thermo_optic_mzi.py` | Silicon thermo-optic unbalanced MZI (`ThermalOpticCoefficient`) & 2-stage flat-top lattice MZI via `TransferMatrix` / `ChainMatrix` composition (Cocorullo & Rendina 1992; Soref & Bennett 1987) | `MZ`, `Line`, `TransferMatrix`, `ChainMatrix`, `composeTMlist`, `composeCMlist`, `Si` |
| `ex_dispersive_lossy_etalon.py` | Dispersive uncoated silicon wafer etalon vs. `IsotropicTransferMatrix` & coated high-finesse lossy cavity (`κ → 0⁺`) (Born & Wolf 1999; Fabry & Pérot 1899) | `Etalon`, `IsotropicTransferMatrix`, `snell`, `loss_cm2rix` |

### 7. `scattering/` — 3D Vector Surface-Current Diffraction

| Script | Physics, Literature & Cross-Checks | Key `EMpy` APIs |
| :--- | :--- | :--- |
| `ex_kottler_aperture_diffraction.py` | Vector Huygens–Kottler circular aperture diffraction in E- and H-planes vs. analytical Airy pattern & Fresnel near-field axial propagation in vacuum and lossy dielectric media (Kottler 1923; Stratton & Chu 1939; Balanis 2012) | `currentsScatteringKottler`, `Field`, `stack` |
