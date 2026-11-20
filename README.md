# EMpy - ElectroMagnetic Python

[![CI](https://github.com/lbolla/EMpy/actions/workflows/python-app.yml/badge.svg)](https://github.com/lbolla/EMpy/actions/workflows/python-app.yml) [![PyPI](https://badge.fury.io/py/ElectroMagneticPython.svg)](https://badge.fury.io/py/ElectroMagneticPython)

`EMpy - ElectroMagnetic Python` is a Python suite of numerical algorithms widely used in electromagnetism, photonics, and integrated optics: transfer matrix methods, rigorous coupled-wave analysis (RCWA), finite-difference and film-mode-matching waveguide mode solvers, coupled-resonator photonic circuits, liquid-crystal optics, and vector diffraction/scattering integrals.

Run the domain-organized scripts in `examples/` (see `examples/README.md` for the full catalog of 22 literature benchmarks and cross-solver comparisons) to see how `EMpy` works in practice, and visit https://lbolla.github.io/EMpy/ for additional documentation.

## Implemented Algorithms

- **Transfer Matrix Method (`EMpy.transfer_matrix`):**
  - `IsotropicTransferMatrix`: 2 × 2 characteristic-matrix solver for s- and p-polarized plane waves in isotropic multilayers, including absorbing media, total internal reflection (TIR), and surface-plasmon resonance (SPR) geometries.
  - `AnisotropicTransferMatrix`: 4 × 4 Yeh–Berreman solver for arbitrarily oriented uniaxial, biaxial, twisted-nematic, and lossy anisotropic multilayers, solving the first-order Berreman eigenproblem with SVD nullspace polarization vectors, biaxial optic-axis / binormal degeneracy orthogonalization, and cross-mode-interference Poynting flux.
- **Rigorous Coupled-Wave Analysis (`EMpy.RCWA`):**
  - `IsotropicRCWA`: Planar and conical diffraction solver for multilayer surface-relief gratings (`BinaryGrating`, `SymmetricDoubleGrating`, `AsymmetricDoubleGrating`) based on Moharam's formulation, Lalanne's TM Fourier factorization, and Enhanced Transmittance Matrix (ETM) stabilization.
  - `AnisotropicRCWA`: 3D vector coupled-wave solver for arbitrarily Euler-rotated anisotropic gratings with isotropic or uniaxial end layers, Li's inverse-rule Fourier factorization for discontinuous permittivity tensors, and ETM recursion for thick high-order stacks.
- **Waveguide Mode Solvers & Junctions (`EMpy.modesolvers`):**
  - `FD` (`SVFDModeSolver`, `VFDModeSolver`): Semivectorial and full-vectorial 2D finite-difference mode solvers on non-uniform staggered grids, supporting symmetry walls, 5-component anisotropic permittivity tensors, complex-coordinate-stretching PML boundaries (`stretchmesh`), and shift-invert ARPACK eigensolvers.
  - `FMM` (`FMM1d`, `FMM2d`): Scale-invariant 1D slab and 2D channel waveguide Film Mode Matching solvers (supporting both SI meters and microns) expanding fields in slice eigenmodes with overflow-free evanescent inner products and analytical sub-cell Poynting overlaps.
  - `geometries` & `interface` (`SWG`, `SimpleJoint`, `GenericDevice`): Mode-matching S-matrix junctions (offering both passive conjugated-Poynting mode matching and Lorentz-reciprocal unconjugated-reaction mode matching, including cross-solver `FDMode` ↔ `FMMMode2d` overlaps), stabilized multi-element Redheffer star-product S-matrix cascades (`GenericDevice`) for sub-cutoff and long lossy waveguides, and staggered-grid FDTD mode-overlap helpers (`scripts/FDTD.py`).
- **Photonic Devices, Scattering, Materials & Liquid Crystals (`EMpy.devices`, `EMpy.scattering`, `EMpy.materials`, `EMpy.utils`):**
  - `devices`: Transfer- and chain-matrix models for directional couplers (`Coupler`), delay lines (`Line`), Mach–Zehnder interferometers (`MZ`), single and N-coupled ring resonators (`APRR`, `SRR`, `NRR`), tunable multi-coupler ring filters (`T_TCRR`, `T_CRTCRR`, `T_CTCRR`), dispersive/lossy Fabry–Pérot cavities (`Etalon`), and silicon strip waveguide dispersion (`SWG`).
  - `scattering`: 3D Kottler–Stratton–Chu vector surface-current radiation and scattering integral (`currentsScatteringKottler`).
  - `materials` & `utils`: Constant, polynomial, Sellmeier, tabulated (interpolated), and thermo-optic `IsotropicMaterial`, `AnisotropicMaterial`, and `LiquidCrystal` models, together with `LiquidCrystalCell` (coupled 1D Oseen–Frank elastic free-energy and Laplace electrostatic BVP solver with pretilt, twist, chiral pitch, and voltage).

## Testing & Verification Strategy

`EMpy`'s test suite (`tests/`) validates numerical implementations against analytical electrodynamics and physical invariants:

1. **Consistent Harmonic & Passive Loss Convention:**
   - All solvers follow `exp(+iωt - ik·r)` with passive complex refractive index `n = n_r - iκ` (`κ ≥ 0`), passive square-root branch selection (`Re(k_z) - Im(k_z) ≥ 0`) across propagating, evanescent/TIR, and negative-`ε_r` regimes, and general complex-symmetric linear algebra (`M = Mᵀ`) without spurious Hermitian conjugation.
2. **Conservation Laws, Passivity & Reciprocity:**
   - *Lossless structures:* Exact energy conservation and unitarity (`R + T = 1`, `sum(DE1 + DE3) = 1`, `Sᴴ S = I`, `|THRU|² + |DROP|² = 1`).
   - *Passive lossy structures (`κ > 0`):* Strict passivity (`R + T ≤ 1`, `‖S‖₂ ≤ 1`, Beer–Lambert modal attenuation `|exp(-iβL)| < 1`) and Lorentz reciprocity (`S = Sᵀ` whenever `ε = εᵀ`).
3. **Degenerate-Limit Continuity (`AGENTS.md`):**
   - *Anisotropic → Isotropic (`Δn → 0`):* Every anisotropic solver (`AnisotropicMaterial`, `LiquidCrystalCell`, `AnisotropicTransferMatrix`, `AnisotropicRCWA`, 5-component `VFDModeSolver`) is tested against the corresponding isotropic solver both via explicit isotropic tensors (`ε = ε₀ n² I₃` bypassing `isIsotropic()` shortcuts) and along weak-birefringence sequences (`Δn ∈ {1e-4, 1e-8, 0}`) with arbitrary 3D Euler rotations, optic-axis / binormal degeneracies, normal-incidence azimuthal limits (`α → 0⁺, δ ≠ 0`), and `n = 0` / homogeneous grating limits.
   - *Lossy → Lossless (`κ → 0⁺`):* Lossy code paths are tested both at complex zero-loss dtype (`n = n_r + 0.0j`) and along continuous `κ → 0⁺` sequences against strictly real lossless paths across propagating, sub-cutoff, PML, and TIR regimes.
4. **Independent Cross-Solver & Analytical Benchmarks:**
   - Verified against Fresnel/Brewster/Airy equations, Kretschmann SPR dips, Jones waveplate calculus, Gooch–Tarry twisted-nematic transmission, Rytov subwavelength effective-medium theory, 1D slab dispersion relations, 2D Gauss–Legendre quadrature, and pointwise Maxwell curl/divergence equations, as well as CI runs across Python 3.10–3.14, `lowest-direct` dependency bounds, and all `examples/**/ex_*.py` scripts.

## Installation

Requires Python `>= 3.10`:

```bash
pip install ElectroMagneticPython
```

> **Note:** `LiquidCrystalCell` solves the 1D Oseen–Frank liquid-crystal boundary-value problem out of the box using `scipy.integrate.solve_bvp` (and will only use the legacy `scikits.bvp1lg` package if explicitly installed).

## Development

Download the source code from https://github.com/lbolla/EMpy.

`EMpy` is managed with `uv` (https://docs.astral.sh/uv/).

Install the development environment:

```bash
make develop
```

Run tests and linters (`black --check`, `flake8`, `pyflakes`, `mypy`):

```bash
make test
make lint
```

Format code with `black`:

```bash
make black
```

Run an example script (see `examples/README.md` for all categories):

```bash
uv run examples/devices/ex_ring_resonators_aprr_srr_nrr.py
```

Upgrade locked dependencies:

```bash
make upgrade
```

## Release process

Package versioning is managed with `bump2version` and `setuptools_scm`:

```bash
make release PART=patch  # PART=[major|minor|patch]
```

Pushing the version tag triggers GitHub Actions to build and publish the package to PyPI.

## References

- D. W. Berreman, "Optics in stratified and anisotropic media: 4 × 4-matrix formulation," *J. Opt. Soc. Am.* **62**(4), 502–510 (1972).
- P. Yeh, "Electromagnetic propagation in birefringent layered media," *J. Opt. Soc. Am.* **69**(5), 742–756 (1979); *Optical Waves in Layered Media* (Wiley, 1988).
- E. N. Glytsis and T. K. Gaylord, "Three-dimensional (vector) rigorous coupled-wave analysis of anisotropic grating diffraction," *J. Opt. Soc. Am. A* **7**(8), 1399–1410 (1990).
- M. G. Moharam, E. B. Grann, D. A. Pommet, and T. K. Gaylord, "Formulation for stable and efficient implementation of the rigorous coupled-wave analysis of binary gratings," *J. Opt. Soc. Am. A* **12**(5), 1068–1076 (1995).
- M. G. Moharam, D. A. Pommet, E. B. Grann, and T. K. Gaylord, "Stable implementation of the rigorous coupled-wave analysis for surface-relief gratings: enhanced transmittance matrix approach," *J. Opt. Soc. Am. A* **12**(5), 1077–1086 (1995).
- P. Lalanne and G. M. Morris, "Highly improved convergence of the coupled-wave method for TM polarization," *J. Opt. Soc. Am. A* **13**(4), 779–784 (1996).
- L. Li, "Use of Fourier series in the analysis of discontinuous periodic structures," *J. Opt. Soc. Am. A* **13**(9), 1870–1876 (1996); "Reformulation of the Fourier modal method for surface-relief gratings made with anisotropic materials," *J. Mod. Opt.* **45**(7), 1313–1334 (1998).
- A. B. Fallahkhair, K. S. Li, and T. E. Murphy, "Vector finite difference modesolver for anisotropic dielectric waveguides," *J. Lightwave Technol.* **26**(11), 1423–1431 (2008).
- A. S. Sudbø, "Film mode matching: a versatile numerical method for vector mode field calculations in dielectric waveguides," *Pure Appl. Opt.* **2**(3), 211–233 (1993); "Improved formulation of the film mode matching method for mode field calculations in dielectric waveguides," *Pure Appl. Opt.* **3**(3), 381–388 (1994).
- O. Schwelb, "Transmission, group delay, and dispersion in single-ring optical resonators and add/drop filters—a tutorial overview," *J. Lightwave Technol.* **22**(5), 1380–1394 (2004).
- G. Barbarossa, A. M. Matteo, and M. N. Armenise, "Theoretical analysis of triple-coupler ring-based optical guided-wave resonator," *J. Lightwave Technol.* **13**(2), 148–157 (1995).
- C. H. Gooch and H. A. Tarry, "The optical properties of twisted nematic liquid crystal structures with twist angles ≤ 90°," *J. Phys. D: Appl. Phys.* **8**(13), 1575–1584 (1975).
- F. Kottler, "Elektromagnetische Theorie der Beugung an schwarzen Schirmen," *Ann. Phys.* **376**(15), 457–508 (1923); J. A. Stratton and L. J. Chu, "Diffraction theory of electromagnetic waves," *Phys. Rev.* **56**(1), 99–107 (1939).
- S. M. Rytov, "Electromagnetic properties of a finely stratified medium," *Sov. Phys. JETP* **2**(3), 466–475 (1956).

## Citation

If you find EMpy useful in your job, please consider adding a citation.

As reference:

```
Bolla, L. (2017). EMpy [Computer software]. https://github.com/lbolla/EMpy/
```

As text:

```
We used EMpy (version x.y.z) to complete our work.
```
