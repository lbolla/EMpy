---
layout: default
title: Waveguide S-Matrix Circuits & Mode-Matching in Python
description: Model waveguide step discontinuities, MMI self-imaging couplers, and Redheffer star-product S-matrix cascades with passive or Lorentz-reciprocal mode matching in Python.
permalink: /waveguide-s-matrix-circuits/
---

## Waveguide S-Matrix Circuits & Mode-Matching in Python

`EMpy.modesolvers.geometries` and `EMpy.modesolvers.interface` compute generalized **$$2N \times 2N$$ mode-matching scattering matrices (S-matrices)** at waveguide discontinuities and cascade multi-section photonic circuits without exponential overflow.

---

### Solvers & Capabilities

- **`SimpleJoint`:** Computes the $$2N \times 2N$$ S-matrix at an abrupt waveguide junction (including cross-solver `FDMode` ↔ `FMMMode2d` matching) with two complementary formulations:
  - `method="passive"` — conjugated-Poynting projection guaranteeing strict passivity ($$\lVert S \rVert_2 \le 1$$) under finite modal truncation $$N$$.
  - `method="reciprocal"` — bilateral unconjugated Lorentz-reaction polar-midpoint S-matrix enforcing exact Lorentz reciprocity ($$S = S^T$$).
- **`SWG` & `GenericDevice`:** Propagate forward and backward modal fields along straight waveguide sections (`SWG`, strictly bounded $$\lvert \exp(-i\beta L) \rvert \le 1$$) and cascade arbitrary sequences via **Redheffer star-product** S-matrix composition—including sub-cutoff evanescent barriers ($$\operatorname{Im}(n_{\mathrm{eff}}) < -1.5$$).

---

### Quick Python Example: $$1 \times 2$$ MMI Cascade via Redheffer Star-Product

```python
import EMpy

# sol_in, sol_mmi: solved 2D modes from VFDModeSolver or FMM2d
mmi_cascade = EMpy.modesolvers.geometries.GenericDevice([
    EMpy.modesolvers.geometries.SWG(sol_in, l_in),
    EMpy.modesolvers.geometries.SimpleJoint(sol_in, sol_mmi, method="passive"),
    EMpy.modesolvers.geometries.SWG(sol_mmi, 0.90 * l_pi),
])

# Evaluate total S-matrix and internal field distributions across the MMI
S_total = mmi_cascade.S
```

![Waveguide step junction passive vs reciprocal mode matching S-parameters]({{ '/images/circuits_step_junction.png' | relative_url }})

![1x2 MMI self-imaging and sub-cutoff Redheffer S-matrix cascade]({{ '/images/circuits_mmi_fp.png' | relative_url }})

---

### Benchmark Scripts & References

- [`ex_waveguide_step_junction.py`](https://github.com/lbolla/EMpy/blob/master/examples/waveguide_circuits/ex_waveguide_step_junction.py) — Waveguide step junction comparing `SimpleJoint` `passive` vs. `reciprocal` S-matrices, mode-count convergence, and `VFD ↔ FMM2d` matching (*Vassallo 1991; Eleftheriades et al. 1994*).
- [`ex_mmi_and_fabry_perot_cascade.py`](https://github.com/lbolla/EMpy/blob/master/examples/waveguide_circuits/ex_mmi_and_fabry_perot_cascade.py) — $$1 \times 2$$ MMI self-imaging ($$z = \tfrac{3}{8} L_\pi, \tfrac{3}{4} L_\pi$$) and sub-cutoff Fabry–Pérot cavity stabilized via `GenericDevice` (*Soldano & Pennings 1995; Redheffer 1960*).

**See also:** [Methodology & References]({{ '/methodology/#modesolvers' | relative_url }}) · [All Examples]({{ '/examples/#catalog' | relative_url }})
