---
layout: default
title: Liquid-Crystal Electro-Optics & Oseen–Frank BVP in Python
description: Solve 1D Oseen–Frank elastic and electrostatic director profiles in twisted nematic and cholesteric liquid crystal cells and compute 4x4 Berreman optics in Python with EMpy.
permalink: /liquid-crystal-optics/
---

## Liquid-Crystal Electro-Optics & Oseen–Frank BVP in Python

`EMpy.utils.LiquidCrystalCell` solves the coupled 1D **Oseen–Frank elastic free-energy** ($$K_{11}, K_{22}, K_{33}$$, chiral wavenumber $$q_0$$) and **inhomogeneous Laplace electrostatic** boundary-value problem natively via `scipy.integrate.solve_bvp`, and expands the resulting director profile into a 4 × 4 Berreman `Multilayer` stack for `AnisotropicTransferMatrix`.

---

### Capabilities

- **Self-Consistent Director & Voltage Profiles:** Computes tilt $$\theta(z)$$, twist $$\phi(z)$$, electrostatic potential $$V(z)$$, Freedericksz threshold $$V_{\mathrm{th}}$$, and voltage-dependent static `capacitance()`.
- **Twisted Nematic (TN) & Cholesteric Bragg Optics:** Reproduces Gooch–Tarry Mauguin transmission minima ($$u = \sqrt{3}, \sqrt{15}$$) and de Vries selective circular-polarization Bragg bands $$[n_O P, n_E P]$$.
- **Tabulated Profile Interpolation:** Supports experimental or external director-tilt data files alongside analytical/BVP solutions.

---

### Quick Python Example: 90° Twisted Nematic Cell Electro-Optics

```python
import numpy as np
import EMpy

lc = EMpy.materials.get_10400_000_100()
amb = EMpy.materials.IsotropicMaterial("Glass", n0=1.5)

cell = EMpy.utils.LiquidCrystalCell(
    lc,
    voltage=3.5,
    t_tot=4.8e-6,
    t_anchoring=0.0,
    pretilt=EMpy.utils.deg2rad(2.0),
    totaltwist=np.pi / 2.0,
    nlayers=48,
)

ml = EMpy.utils.Multilayer([
    EMpy.utils.Layer(amb, np.inf),
    cell,
    EMpy.utils.Layer(amb, np.inf),
])

wl = np.array([550e-9])
sol = EMpy.transfer_matrix.AnisotropicTransferMatrix(ml, 0.0, 0.0).solve(wl)
cap = cell.capacitance()
```

![Twisted nematic liquid crystal Gooch-Tarry minima and Oseen-Frank voltage switching]({{ '/images/liquid_crystals_tn.png' | relative_url }})

---

### Benchmark Scripts & References

- [`ex_twisted_nematic_gooch_tarry.py`](https://github.com/lbolla/EMpy/blob/master/examples/liquid_crystals/ex_twisted_nematic_gooch_tarry.py) — 90° Twisted Nematic cell Gooch–Tarry minima, Oseen–Frank voltage switching ($$0 \to 6\;\mathrm{V}$$), Freedericksz threshold ($$V_{\mathrm{th}} = 2.67\;\mathrm{V}$$), and `capacitance()` (*Gooch & Tarry 1975; de Gennes & Prost 1993*).
- [`ex_chiral_lc_bragg_and_datafile.py`](https://github.com/lbolla/EMpy/blob/master/examples/liquid_crystals/ex_chiral_lc_bragg_and_datafile.py) — Cholesteric chiral Bragg selective reflection band ($$[n_O P, n_E P]$$, *de Vries 1951*) and tabulated `lc_data.dat` interpolation (*Yeh & Gu 2009*).

**See also:** [Methodology & References]({{ '/methodology/#devices-materials' | relative_url }}) · [All Examples]({{ '/examples/#catalog' | relative_url }})
