# Changelog

All notable changes to `EMpy - ElectroMagnetic Python` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.1] - 2026-11-23

`EMpy 3.0.1` is a physics and numerical bug-fix release resolving 17 issues uncovered by a systematic electrodynamic audit across all modules, and expanding the test suite to 104 tests with closed-form analytical benchmarks covering every solver.

### Fixed

- **Photonic Devices & Materials (`EMpy.devices`, `EMpy.materials`):**
  - Fixed `NRR.CM` for `N ≥ 2` rings to compose each ring section `(L1 · K · L2)` as a `ChainMatrix` via `composeCMlist` instead of cascading with `composeTMlist` (previously modeled a single ring coupled `N` times to the input bus rather than `N` serially coupled CROW rings).
  - Fixed `T_CTCRR.DROP` relative phase sign (`-1j` instead of `+1j`, matching `j³ = -j` across three cross-couplings under `Coupler`'s `[[t, jK], [jK, t]]` convention).
  - Fixed `EMpy.materials.SiO2` thermo-optic coefficient `dn/dT` from `1.1e-4` to `1.0e-5 K⁻¹` (fused silica near 1.55 µm).
- **Transfer Matrix (`EMpy.transfer_matrix`):**
  - Replaced `AnisotropicTransferMatrix.solve`'s raw 4 × 4 `D P D⁻¹` cascade with a bounded passive Enhanced Transmittance Matrix (ETM) backward sweep (`|exp(-i k_z d)| ≤ 1`), eliminating catastrophic cancellation on thick coupled anisotropic layers and `NaN` overflow on thick metals.
  - Replaced `IsotropicTransferMatrix.solve`'s raw cascade with closed-form Abeles layer matrices using `sinc` at `q_z → 0` (eliminating `LinAlgError` at exact inner-layer grazing/cutoff `n_i = n_0 sin θ`) and per-layer `exp(-|Im φ|)` scaling (eliminating thick-metal overflow).
- **Rigorous Coupled-Wave Analysis (`EMpy.RCWA`):**
  - Unified the `zero_kt` threshold (`|kt| ≤ 1e-13 * k`) in `IsotropicRCWA` and `AnisotropicRCWA` so `fc² + fs² = 1` holds identically near normal incidence and for near-normal exiting diffraction orders.
  - Fixed internal-layer mode branch selection in `IsotropicRCWA` (`Re(q) + Im(q) ≥ 0`) and `AnisotropicRCWA` (`-(Re(q) + Im(q)) ≥ 0`) under oblique incidence from a lossy superstrate, eliminating singular/exploding `MTp1`.
  - Fixed `AnisotropicRCWA`'s custom-grating fallback to avoid using `(ε⁻¹)_xx` as `1/ε_xx` when off-diagonal `x`-coupling is present, and updated `get_pitch()` to recognize custom grating classes with a `.pitch` attribute.
- **Waveguide Mode Solvers (`EMpy.modesolvers.FD`, `EMpy.modesolvers.FMM`):**
  - Fixed swapped `n`/`s` mesh weights in `VFDModeSolver.build_matrix` (`ayxn` off-diagonal `ε_xy` terms), restoring exact `x ↔ y` rotational symmetry and second-order consistency on non-uniform anisotropic meshes.
  - Fixed `stretchmesh` geometric method `'G'` for mesh compression (`0 < f < 1`) via exact convex `brentq` bracketing on `(-1/f, ln f)`, preventing convergence to the trivial root `b = 0`.
  - Fixed `FDMode.norm()` to return strictly real `sqrt(|P|)` so `normalize()` yields unit norm without injecting a spurious global phase on lossy/PML modes.
  - Fixed off-by-one right-branch continuity guards (`idx1 > n2 - 1`) in `FMMshooting` and `FMMshootingTM` for slices with `≥ 2` layers to the right of the peak-index layer, and added scale-invariant derivative/flux matching fallback when a mode has a node at the matching interface.
  - Fixed `FMM2d` interior-slice indexing (`i = idx - 1`) in `Mvec` and generalized `check_matching` for `Nslices ≥ 4`.
  - Replaced single-endpoint evanescent evaluation in `FMMMode1dy.eval`, `FMMMode2d.eval`, `_slice_mode1dx`, and `_refine_mode1d` with symmetric two-endpoint decaying exponentials (`_eval_segment`) for `|Im(k·d)| > 1`, and made `FMMMode2d.norm()` strictly real.
  - Enforced the passive cutoff branch `Re(k_eff) - Im(k_eff) ≥ 0` continuously as `κ → 0⁺` in `analyticalsolution` and `FMM1d_y`.
  - Fixed `findzerosnew` scalar indexing on single-zero scan windows (`len(zeri) == 1`) and added local-minimum detection so broad off-axis minima in lossy waveguides are not missed.

### Tests

- Expanded the verification suite from 85 to 104 tests with closed-form analytical benchmarks across every solver: Lekner/Yeh uniaxial oblique Fresnel (`AnisotropicTransferMatrix`), lossless CROW Möbius recursion (`NRR`), Yariv critical coupling and resonant group delay (`APRR`, `SRR`), balanced 3 dB `MZ`, homogeneous PEC waveguide & uniaxial anisotropic slab dispersion (`VFDModeSolver`, `SVFDModeSolver`), finite-box slab & multi-layer/multi-slice Abeles dispersion (`FMM1d`, `FMM2d`), Hertzian electric & magnetic dipole near/mid/far fields (`currentsScatteringKottler`), and Stefan–Boltzmann / Wien blackbody laws.

## [3.0.0] - 2026-11-20

`EMpy 3.0.0` is a major numerical, electrodynamic, and architectural overhaul across every module of the library, resolving dozens of long-standing physical and numerical bugs, stabilizing ill-conditioned multilayer/grating/waveguide cascades, removing legacy broken dependencies, and expanding both the verification test suite and literature-benchmarked examples.

### Breaking / Convention Alignment Notice

- **Unified Harmonic & Passive Loss Convention (`exp(+iωt - ik·r)`):**
  - All solvers and utilities now consistently follow `exp(+iωt - ik·r)` with passive complex refractive index `n = n_r - iκ` (`κ ≥ 0`, `Im(n) ≤ 0`), passive forward/evanescent wavevector branch selection (`Re(k_z) - Im(k_z) ≥ 0`), and passive complex-coordinate PML stretching (`Im(factor) < 0` in `stretchmesh`).
  - `EMpy.utils.rix2losses` now computes positive absorption `α = -4π Im(n) / λ` matching `loss_cm2rix`, `loss_m2rix`, and `loss_dBcm2rix` (previously had an inverted sign).
  - `EMpy.constants.eta0` now uses the exact CODATA relation `mu0 * c` instead of the `120π` approximation.
- **Python & Scientific Stack Modernization:**
  - Requires Python `>= 3.10`, `numpy >= 2.1`, `scipy >= 1.13`, and `matplotlib >= 3.8.4` (`numpy.trapz` updated to `numpy.trapezoid`; Python 3 slicing, sorting, and integer indexing restored throughout).
- **Reorganized `examples/` Directory Layout:**
  - Legacy numbered `examples/ex_*.py` scripts have been replaced by 22 topic-organized scripts across 7 domain subdirectories (`transfer_matrix/`, `liquid_crystals/`, `rcwa/`, `modesolvers/`, `waveguide_circuits/`, `devices/`, `scattering/`) plus `examples/data/` and `examples/README.md`.

### Added

- **Out-of-the-Box Liquid-Crystal BVP Solver (`EMpy.utils.LiquidCrystalCell`):**
  - Solves the coupled 1D Oseen–Frank elastic free-energy and Laplace electrostatic boundary-value problem directly via `scipy.integrate.solve_bvp`, eliminating the unmaintained `scikits.bvp1lg` requirement (while retaining compatibility if explicitly installed).
- **Dual Passive & Lorentz-Reciprocal Mode-Matching (`EMpy.modesolvers.geometries.SimpleJoint`):**
  - Added `method="passive"` (default conjugated-Poynting mode matching guaranteeing `‖S‖₂ ≤ 1` under modal basis truncation) and `method="reciprocal"` (bilateral unconjugated Lorentz-reaction polar-midpoint formulation enforcing exact `S = Sᵀ` and port-swap symmetry at every truncation order).
  - Added direct cross-solver mode-overlap support between `FDMode` (`VFDModeSolver`) and `FMMMode2d` (`FMM2d`).
- **22 Literature-Benchmarked Examples (`examples/`):**
  - Full coverage of every `EMpy` module with bibliographic citations in each docstring, including Kretschmann SPR (`κ → 0⁺`), Gooch–Tarry TN switching, de Vries chiral Bragg reflectors, Moharam ETM gratings, conical Wood–Rayleigh anomalies, Koshelev dimer quasi-BICs, Chew–Weedon PML leaky waveguides, Soldano MMI self-imaging, Barbarossa coupled-ring filters, thermo-optic silicon MZIs, and Kottler vector aperture diffraction.
- **Comprehensive Physics & Degenerate-Limit Test Suite (`tests/`):**
  - Expanded to 85 tests validating lossless energy conservation/unitarity, passive lossy bounds, Lorentz reciprocity, `Δn → 0` anisotropic-to-isotropic limits, `κ → 0⁺` lossy-to-lossless limits, scale invariance, and CI smoke-testing across Python 3.10–3.14 and `lowest-direct` dependencies.

### Fixed

- **Transfer Matrix (`EMpy.transfer_matrix`):**
  - Rewrote `AnisotropicTransferMatrix` around the exact 4 × 4 Berreman eigenproblem with SVD nullspace polarization vectors, biaxial optic-axis / binormal orthogonalization, weak-birefringence (`Δn → 0`) continuity, and full cross-mode-interference Poynting flux `S_z` (including negative local Goos–Hänchen TIR flux from lossy superstrates without artificial clamping).
  - Fixed `snell` and `IsotropicTransferMatrix` evanescent branch selection under total internal reflection (TIR), lossy incident media, and negative-permittivity (`Re(n) = 0, Im(n) < 0`) plasmonic layers, plus scalar wavelength support.
- **Rigorous Coupled-Wave Analysis (`EMpy.RCWA`):**
  - Stabilized `AnisotropicRCWA` with Moharam's Enhanced Transmittance Matrix (ETM) recursion to eliminate exponential overflow on thick or high-order gratings.
  - Implemented Li's inverse-rule Fourier factorization for discontinuous anisotropic permittivity tensors, proper ordinary + extraordinary eigenwave superposition in uniaxial end layers, and phase-preserving unit polarization vectors (`dE_hat`) on lossy uniaxial superstrates.
  - Fixed `IsotropicRCWA` normal-incidence conical azimuth (`α = 0, δ ≠ 0`), complex transverse wavevectors `(kx, ky)` in lossy/negative-`ε` superstrates, ill-conditioned Wood-anomaly `kx` shifts, and `SymmetricDoubleGrating` ridge spacing.
  - Fixed `dispersion_relation_ordinary` and `dispersion_relation_extraordinary` to preserve exact Helmholtz roots (`kz² = delta`) on the passive branch cut.
- **Waveguide Mode Solvers & Circuits (`EMpy.modesolvers`):**
  - Made `FMM1d` and `FMM2d` geometrically scale-invariant across SI meters (`wl ~ 1.55e-6`) and microns (`wl ~ 1.55`) by scaling wavenumber search steps/bounds by `k₀`, normalizing `Mvec / k₀`, removing global `numpy.random.seed()` calls, fixing removable `k_x d → 0` singularities, and eliminating `O(exp(4 |Im(k)| d))` catastrophic cancellation in evanescent `scalarprod` integrals.
  - Fixed `FMMMode2d` analytical Poynting `overlap` conjugation/signs, right-wall boundary condition checks (`xright`), and staggered-grid `get_fields_for_FDTD` transposes, as well as `scripts/FDTD.py` (`overlap_f`) impedance scaling and conjugation.
  - Stabilized `GenericDevice` and `SWG` via direct bounded `exp(-iβL)` S-matrices and Redheffer star-product cascading so sub-cutoff (`Im(n_eff) < 0`) and long lossy waveguides never overflow.
  - Fixed `SVFDModeSolver` polarization-dependent boundary symmetry signs (`Ex` vs. `Ey`), `VFDModeSolver` shift-invert `"LM"` target selection and passive cutoff branch, `FDMode` staggered-grid interpolation, and `stretchmesh` complex PML coordinate differentials.
- **Photonic Devices, Materials, Scattering & Utilities (`EMpy.devices`, `EMpy.materials`, `EMpy.scattering`, `EMpy.utils`):**
  - Fixed `NRR.solve()` reading `H.data[1, 0]` for `DROP`, `T_TCRR` / `T_CTCRR` non-mutating `"optimum"` coupling synthesis and phase factors, and `Etalon` dispersive group-index `FSRwl`, passive lossy Airy transmission, and low-reflectivity `F < 1` finesse handling.
  - Fixed `RefractiveIndex` / `EpsilonTensor` tabulated `n0_known` interpolation preserving input `wls` array shapes (`scalar`, `1D`, `2D`).
  - Fixed `LiquidCrystalCell` zero-voltage planar pretilt alignment, linear twist profile, chiral `q₀ · d` scaling, real static `capacitance()`, and anchoring thickness validation.
  - Fixed `group_delay_and_dispersion` phase unwrapping and wavelength differentiation, and `find_peaks` flat-top plateau / monotonic inflection handling.
  - Fixed `currentsScatteringKottler` and `stack` input shape normalization, `R == 0` self-point guards, and passive complex `epsr` refractive-index branch selection.
