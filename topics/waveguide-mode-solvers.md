---
layout: default
title: Optical Waveguide Mode Solvers (FD, PML & FMM) in Python
description: Solve 1D/2D optical waveguide eigenmodes, birefringent/gyrotropic tensors, polarization anti-crossings, and leaky PML radiation losses in Python with EMpy.modesolvers.
permalink: /waveguide-mode-solvers/
---

## Optical Waveguide Mode Solvers (FD, PML & FMM) in Python

`EMpy.modesolvers` provides both **finite-difference (`FD`)** and semi-analytical **Film Mode Matching (`FMM`)** eigenmode solvers for 1D slab and 2D channel/rib optical waveguides.

---

### Solvers & Capabilities

- **`SVFDModeSolver` (`EMpy.modesolvers.FD`):** Scalar and polarized semivectorial (`Ex`, `Ey`) finite-difference solver on non-uniform grids (`CrossSection` / `Slice`) with polarization-aware symmetry walls.
- **`VFDModeSolver` (`EMpy.modesolvers.FD`):** Full-vectorial staggered-grid finite-difference solver (Fallahkhair, Li & Murphy 2008) supporting arbitrary 5-component `(nx, ny, 5)` $$[\varepsilon_{xx}, \varepsilon_{xy}, \varepsilon_{yx}, \varepsilon_{yy}, \varepsilon_{zz}]$$ anisotropic and non-reciprocal magneto-optic gyrotropic tensors, shift-invert ARPACK eigensolving, and complex-coordinate-stretching **PML absorbing boundaries** (`stretchmesh` with $$\operatorname{Im}(\text{factor}) < 0$$).
- **`FMM1d` & `FMM2d` (`EMpy.modesolvers.FMM`):** Semi-analytical 1D slab and 2D channel waveguide mode solvers (Sudbø 1993, 1994) with $$k_0$$-scaled root search (invariant between SI meters and microns) and cancellation-free evanescent overlap integrals.

---

### Quick Python Example: Leaky Waveguide Tunneling Loss with `stretchmesh` PML

```python
import numpy as np
import EMpy

wl = 1.55e-6
x_raw = np.linspace(-6.0e-6, 6.0e-6, 121)
y_raw = np.linspace(-6.0e-6, 6.0e-6, 121)

# Apply complex-coordinate-stretching PML (passive Im(factor) < 0) on outer boundaries
xx, yy, _, _, _, _ = EMpy.modesolvers.FD.stretchmesh(
    x_raw, y_raw, [0, 0, 28, 28], 1.0 - 1.8j, method="PPPP"
)

def eps_w(x, y):
    # W-type depressed-cladding leaky waveguide permittivity profile
    ...

sol = EMpy.modesolvers.FD.VFDModeSolver(wl, xx, yy, eps_w, "SS00").solve(2, 1e-11, guess=1.45)
loss_db_cm = EMpy.utils.rix2losses(sol.modes[0].neff, wl)[4]
```

![Leaky waveguide tunneling loss with stretchmesh PML compared to exact 5-layer theory]({{ '/images/modesolvers_pml_leaky.png' | relative_url }})

![Asymmetric SOI rib waveguide TM0-TE1 avoided crossing and Stokes S2 map]({{ '/images/modesolvers_rib_stokes.png' | relative_url }})

---

### Benchmark Scripts & References

- [`ex_pml_leaky_waveguide.py`](https://github.com/lbolla/EMpy/blob/master/examples/modesolvers/ex_pml_leaky_waveguide.py) — W-type depressed-cladding leaky waveguide radiation loss with `stretchmesh` PML vs. exact 5-layer dispersion (*Chew & Weedon 1994; Ghatak et al. 1987*).
- [`ex_rib_waveguide_stokes.py`](https://github.com/lbolla/EMpy/blob/master/examples/modesolvers/ex_rib_waveguide_stokes.py) — Asymmetric etched SOI rib `TM₀–TE₁` anti-crossing and Stokes `S₂ / S₀` map (*Dai & Bowers 2011*).
- [`ex_soi_channel_svfd_vfd_fmm.py`](https://github.com/lbolla/EMpy/blob/master/examples/modesolvers/ex_soi_channel_svfd_vfd_fmm.py) — Cross-solver comparison of `SVFDModeSolver`, `VFDModeSolver`, and `FMM2d`.
- [`ex_anisotropic_and_magnetooptic_wg.py`](https://github.com/lbolla/EMpy/blob/master/examples/modesolvers/ex_anisotropic_and_magnetooptic_wg.py) — LC-clad $$\Delta n \to 0$$ limit & Ce:TbIG magneto-optic gyrotropic spin density `S₃ / S₀`.
- [`ex_fmm1d_fmm2d_lossy_and_units.py`](https://github.com/lbolla/EMpy/blob/master/examples/modesolvers/ex_fmm1d_fmm2d_lossy_and_units.py) — `FMM1d` vs. exact slab, lossy `FMM2d`, and SI-meter vs. micron invariance (*Sudbø 1993, 1994*).

**See also:** [Methodology & References]({{ '/methodology/#modesolvers' | relative_url }}) · [All Examples]({{ '/examples/#catalog' | relative_url }})
