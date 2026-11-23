# EMpy - ElectroMagnetic Python

[![CI](https://github.com/lbolla/EMpy/actions/workflows/python-app.yml/badge.svg)](https://github.com/lbolla/EMpy/actions/workflows/python-app.yml) [![PyPI](https://badge.fury.io/py/ElectroMagneticPython.svg)](https://pypi.org/project/ElectroMagneticPython/)

`EMpy - ElectroMagnetic Python` is a Python suite of numerical algorithms for electromagnetism, photonics, and integrated optics: isotropic and 4 × 4 anisotropic transfer matrix methods, planar/conical/anisotropic rigorous coupled-wave analysis (RCWA), finite-difference and film-mode-matching waveguide mode solvers, passive and Lorentz-reciprocal waveguide S-matrix cascades, optical ring resonators and coupled-resonator photonic circuits, liquid-crystal electro-optics, and 3D vector diffraction integrals.

- **Homepage & Documentation:** https://lbolla.github.io/EMpy/
- **Methodology & Literature References:** https://lbolla.github.io/EMpy/methodology/
- **Examples Gallery (22 literature benchmarks):** https://lbolla.github.io/EMpy/examples/ (also in `examples/`)
- **Release Notes:** [CHANGELOG.md](https://github.com/lbolla/EMpy/blob/master/CHANGELOG.md)

## What's New in 3.0

- **Unified Passive Electrodynamic Conventions:** Consistent `exp(+iωt - ik·r)` harmonic phase, passive complex refractive index `n = n_r - iκ` (`κ ≥ 0`), passive branch-cut selection (`Re(k_z) - Im(k_z) ≥ 0`) across TIR, plasmonic, and lossy media, and complex-symmetric linear algebra throughout.
- **Stabilized Full-Wave & Multilayer Solvers:** ETM-stabilized 4 × 4 Yeh–Berreman `AnisotropicTransferMatrix` and closed-form Abeles `IsotropicTransferMatrix`, ETM-stabilized `AnisotropicRCWA` with Li's inverse-rule factorization, scale-invariant multi-slice `FMM1d`/`FMM2d` (SI meters & microns), and Redheffer star-product S-matrix cascades (`GenericDevice`, passive & reciprocal `SimpleJoint`).
- **Out-of-the-Box Liquid Crystals & Modern Stack:** `LiquidCrystalCell` solves coupled 1D Oseen–Frank elastic + electrostatic BVPs natively via `scipy.integrate.solve_bvp` (no `scikits.bvp1lg` required), targeting Python 3.10–3.14, NumPy 2.x, and SciPy 1.13+.
- **Rigorous Verification:** 104-test physics/invariant suite validating closed-form analytical benchmarks (Fresnel/Lekner, Airy/Abeles, CROW Möbius, Yariv critical coupling, Hertzian dipoles, uniaxial slabs), conservation laws (`R + T = 1`, `Sᴴ S = I`), passivity (`‖S‖₂ ≤ 1`), Lorentz reciprocity (`S = Sᵀ`), and continuous `Δn → 0` and `κ → 0⁺` degenerate limits.

## Implemented Modules

- **`EMpy.transfer_matrix`**: `IsotropicTransferMatrix` (2 × 2) and `AnisotropicTransferMatrix` (4 × 4 Yeh–Berreman) for dispersive, lossy, TIR/SPR, and 3D Euler-rotated biaxial multilayers.
- **`EMpy.RCWA`**: `IsotropicRCWA` and `AnisotropicRCWA` for planar, conical, and 3D birefringent gratings with Lalanne/Li Fourier factorization and Enhanced Transmittance Matrix (ETM) stabilization.
- **`EMpy.modesolvers`**: Semivectorial (`SVFDModeSolver`) and 5-component anisotropic full-vectorial (`VFDModeSolver`) finite-difference solvers with `stretchmesh` PMLs, `FMM1d`/`FMM2d` Film Mode Matching, and S-matrix junctions/cascades (`SimpleJoint`, `SWG`, `GenericDevice`).
- **`EMpy.devices`**: Optical ring resonators (`APRR`, `SRR`, `NRR` microring and CROW filters), tunable multi-coupler ring resonators (`T_TCRR`, `T_CRTCRR`, `T_CTCRR`), directional couplers, delay lines, thermo-optic `MZ` interferometers, and `Etalon`.
- **`EMpy.materials`, `EMpy.scattering`, `EMpy.utils`**: Sellmeier/polynomial/tabulated/thermo-optic materials, `LiquidCrystalCell` electro-optic BVP solver, and 3D Kottler–Stratton–Chu vector surface-current scattering (`currentsScatteringKottler`).

## Installation

Requires Python `>= 3.10`:

```bash
pip install ElectroMagneticPython
```

## Development

Managed with [`uv`](https://docs.astral.sh/uv/):

```bash
make develop   # install dev environment
make test      # run test suite
make lint      # black --check, flake8, pyflakes, mypy
make black     # format code
```

Run any benchmark script from `examples/` (see `examples/README.md`):

```bash
uv run examples/devices/ex_ring_resonators_aprr_srr_nrr.py
```

## Commercial Support, Consulting & Sponsorship

`EMpy` is free and open-source under the MIT License. If your team uses `EMpy` in R&D or production, commercial services are available from the author (`code@lbolla.info`):

- **Custom Modeling & Optimization:** Grating/metasurface design (`AnisotropicRCWA`), liquid-crystal & birefringent stack modeling (`LiquidCrystalCell`, `AnisotropicTransferMatrix`), waveguide mode/circuit simulation (`FD`, `FMM`, `devices`), and fitting models to experimental data.
- **Integration & Feature Development:** Embedding `EMpy` into internal Python/CI workflows, porting legacy MATLAB scripts, or developing custom solver features and material models.
- **Support Retainers & Sponsorships:** Priority bug fixes, upgrade support, and engineering office hours—or sponsor ongoing maintenance via [GitHub Sponsors](https://github.com/sponsors/lbolla).

## Citation

If you find `EMpy` useful in your work, please consider citing it:

```text
Bolla, L. (2017). EMpy [Computer software]. https://github.com/lbolla/EMpy/
```

Or in text:

```text
We used EMpy (version x.y.z) to complete our work.
```
