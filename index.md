---
layout: default
title: Home
---

[![CI](https://github.com/lbolla/EMpy/actions/workflows/python-app.yml/badge.svg)](https://github.com/lbolla/EMpy/actions/workflows/python-app.yml)
[![PyPI](https://badge.fury.io/py/ElectroMagneticPython.svg)](https://pypi.org/project/ElectroMagneticPython/)

**`EMpy` (ElectroMagnetic Python)** is an open-source Python suite of numerical algorithms for electromagnetism, photonics, and integrated optics: isotropic and 4 × 4 anisotropic transfer matrices, planar/conical/anisotropic rigorous coupled-wave analysis (RCWA), finite-difference and film-mode-matching waveguide mode solvers, passive and Lorentz-reciprocal waveguide S-matrix cascades, coupled-ring photonic circuits, liquid-crystal electro-optics, and 3D vector diffraction integrals.

*(Not to be confused with the Python templating system [empy](http://www.alcyone.com/software/empy/).)*

<div class="callout-box" markdown="1">
### What's New in EMpy 3.0.0

`EMpy 3.0.0` is a landmark numerical, electrodynamic, and architectural release across every module of the library:

- **Unified Passive Electrodynamic Conventions:** Consistent $$\exp(+i\omega t - i\mathbf{k}\cdot\mathbf{r})$$ harmonic convention, passive complex refractive index $$n = n_r - i\kappa$$ ($$\kappa \ge 0$$), passive branch selection ($$\operatorname{Re}(k_z) - \operatorname{Im}(k_z) \ge 0$$) across propagating, total-internal-reflection (TIR), and plasmonic media, and complex-symmetric linear algebra throughout.
- **Overhauled Multilayer, Grating & Mode-Matching Solvers:** Rewritten 4 × 4 Yeh–Berreman `AnisotropicTransferMatrix` (SVD nullspace eigenmodes, biaxial binormals, cross-mode Poynting flux), ETM-stabilized `AnisotropicRCWA` with Li's inverse-rule factorization, scale-invariant `FMM1d`/`FMM2d` (SI meters & microns), and overflow-free Redheffer star-product waveguide cascades (`GenericDevice`, passive & reciprocal `SimpleJoint`).
- **Out-of-the-Box Liquid-Crystal BVP Solver:** `LiquidCrystalCell` solves coupled 1D Oseen–Frank elastic and Laplace electrostatic BVPs natively via `scipy.integrate.solve_bvp`—no legacy `scikits.bvp1lg` dependency needed.
- **22 Literature Benchmarks & 85 Physics Tests:** Explore the [Methodology & References]({{ '/methodology/' | relative_url }}), browse the [Examples Gallery]({{ '/examples/' | relative_url }}), or read the full [CHANGELOG.md](https://github.com/lbolla/EMpy/blob/master/CHANGELOG.md).
</div>

---

## Capabilities at a Glance

| Module | Solvers & Capabilities | Explore |
| :--- | :--- | :--- |
| `EMpy.transfer_matrix` | `IsotropicTransferMatrix` (2 × 2) & `AnisotropicTransferMatrix` (4 × 4 Yeh–Berreman) for dispersive, lossy, TIR/SPR, and 3D Euler-rotated biaxial multilayers | [Methodology]({{ '/methodology/#transfer-matrix' | relative_url }}) · [Examples]({{ '/examples/#transfer-matrix' | relative_url }}) |
| `EMpy.materials` & `utils` | Sellmeier, polynomial, tabulated, and thermo-optic materials (`BK7`, `SiO2`, `Si`), `LiquidCrystalCell` Oseen–Frank electro-optic BVP, `CrossSection`, `group_delay_and_dispersion`, `find_peaks` | [Methodology]({{ '/methodology/#devices-materials' | relative_url }}) · [Examples]({{ '/examples/#liquid-crystals' | relative_url }}) |
| `EMpy.RCWA` | `IsotropicRCWA` & `AnisotropicRCWA` for planar, conical, and 3D birefringent gratings (`BinaryGrating`, `SymmetricDoubleGrating`, `AsymmetricDoubleGrating`) with Moharam ETM recursion | [Methodology]({{ '/methodology/#rcwa' | relative_url }}) · [Examples]({{ '/examples/#rcwa' | relative_url }}) |
| `EMpy.modesolvers.FD` & `FMM` | Semivectorial (`SVFDModeSolver`) & 5-component anisotropic/gyrotropic full-vectorial (`VFDModeSolver`) finite-difference solvers with `stretchmesh` PMLs, plus `FMM1d`/`FMM2d` Film Mode Matching | [Methodology]({{ '/methodology/#modesolvers' | relative_url }}) · [Examples]({{ '/examples/#modesolvers' | relative_url }}) |
| `EMpy.modesolvers.geometries` | Waveguide step junctions (`SimpleJoint`: `"passive"` & `"reciprocal"`), cross-solver `FD ↔ FMM` overlaps, and multi-mode Redheffer S-matrix cascades (`SWG`, `GenericDevice`) | [Methodology]({{ '/methodology/#modesolvers' | relative_url }}) · [Examples]({{ '/examples/#waveguide-circuits' | relative_url }}) |
| `EMpy.devices` | Directional couplers, delay lines, thermo-optic `MZ` interferometers, `APRR` / `SRR` / `NRR` microring filters, Barbarossa tunable coupled rings (`T_TCRR`, `T_CRTCRR`, `T_CTCRR`), and `Etalon` | [Methodology]({{ '/methodology/#devices-materials' | relative_url }}) · [Examples]({{ '/examples/#devices' | relative_url }}) |
| `EMpy.scattering` | 3D vector Huygens–Kottler–Stratton–Chu surface-current radiation and aperture diffraction (`currentsScatteringKottler`) in lossless and lossy media | [Methodology]({{ '/methodology/#devices-materials' | relative_url }}) · [Examples]({{ '/examples/#scattering' | relative_url }}) |

---

## Installation

Requires Python `>= 3.10` (`numpy >= 2.1`, `scipy >= 1.13`, `matplotlib >= 3.8.4`):

```bash
pip install ElectroMagneticPython
```

For local development (managed with [uv](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/lbolla/EMpy.git
cd EMpy
make develop
make test
```

---

## Quickstart

Compute the reflectance and transmittance of an 8-pair TiO₂ / SiO₂ Distributed Bragg Reflector on dispersive `BK7` glass around `1.55 µm`:

```python
import numpy as np
import EMpy

wls = np.linspace(0.95e-6, 2.15e-6, 401)
wl0 = 1.55e-6

tio2 = EMpy.materials.IsotropicMaterial(
    "TiO2", n0=EMpy.materials.RefractiveIndex(n0_const=2.45)
)
sio2 = EMpy.materials.SiO2
d_h = wl0 / (4.0 * 2.45)
d_l = wl0 / (4.0 * float(np.real(sio2.n(wl0).item())))

layers = [EMpy.utils.Layer(EMpy.materials.Air, np.inf)]
for _ in range(8):
    layers += [EMpy.utils.Layer(tio2, d_h), EMpy.utils.Layer(sio2, d_l)]
layers.append(EMpy.utils.Layer(EMpy.materials.BK7, np.inf))

sol = EMpy.transfer_matrix.IsotropicTransferMatrix(
    EMpy.utils.Multilayer(layers), theta_inc_x=0.0
).solve(wls)
```

See the [Examples Gallery]({{ '/examples/' | relative_url }}) for 22 complete, self-contained scripts across all modules.

---

## Citation

If you find `EMpy` useful in your research or engineering work, please consider citing it:

```text
Bolla, L. (2017). EMpy — ElectroMagnetic Python [Computer software]. https://github.com/lbolla/EMpy/
```

Or in prose:

```text
Simulations were performed using EMpy (version 3.0.0; https://github.com/lbolla/EMpy/).
```

---

## Contact & Contributing

- **Author & Maintainer:** Lorenzo Bolla ([lbolla.info](https://lbolla.info) · [@lbolla](https://github.com/lbolla))
- **Source Repository:** [github.com/lbolla/EMpy](https://github.com/lbolla/EMpy)
- **Bug Reports, Questions & Pull Requests:** [github.com/lbolla/EMpy/issues](https://github.com/lbolla/EMpy/issues)
- **PyPI Package:** [pypi.org/project/ElectroMagneticPython](https://pypi.org/project/ElectroMagneticPython/)
- **License:** MIT License
