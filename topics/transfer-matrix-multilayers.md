---
layout: default
title: Transfer Matrix Method (2x2 & 4x4 Yeh–Berreman) in Python
description: Simulate isotropic thin-film stacks, DBRs, Kretschmann SPR, and 3D Euler-rotated biaxial anisotropic multilayers in Python using EMpy.transfer_matrix.
permalink: /transfer-matrix-multilayers/
---

## Transfer Matrix Method (2 × 2 & 4 × 4 Yeh–Berreman) in Python

`EMpy.transfer_matrix` provides exact frequency-domain **transfer matrix method (TMM)** solvers for planar stratified media across isotropic, dispersive, lossy, plasmonic, and arbitrarily 3D-rotated anisotropic multilayers.

---

### Solvers & Capabilities

- **`IsotropicTransferMatrix` (2 × 2 Abeles):** Solves $$s$$- (TE) and $$p$$- (TM) polarized plane waves with passive complex refractive index $$n = n_r - i\kappa$$ ($$\kappa \ge 0$$), passive branch selection $$\operatorname{Re}(k_z) - \operatorname{Im}(k_z) \ge 0$$, and exact Poynting-flux transmittance normalization across total internal reflection (TIR) and surface-plasmon resonance (SPR).
- **`AnisotropicTransferMatrix` (4 × 4 Yeh–Berreman):** Solves arbitrarily 3D Euler-rotated uniaxial, biaxial, twisted-nematic, and chiral multilayers:
  - Extracts `(2 forward, 2 backward)` Maxwell eigenwaves via SVD nullspaces.
  - Orthogonalizes degenerate eigenspaces along uniaxial optic axes and biaxial binormals ($$\Delta n \to 0$$ continuous).
  - Evaluates reflected and transmitted power from cross-mode-interference normal Poynting flux $$S_z$$.

---

### Quick Python Example: Quarter-Wave AR Coating on BK7 Glass

```python
import numpy as np
import EMpy

wl0 = 550e-9
wls = np.linspace(400e-9, 800e-9, 401)
mgf2 = EMpy.materials.IsotropicMaterial("MgF2", n0=1.38)

ar_stack = EMpy.utils.Multilayer([
    EMpy.utils.Layer(EMpy.materials.Air, np.inf),
    EMpy.utils.Layer(mgf2, wl0 / (4.0 * 1.38)),
    EMpy.utils.Layer(EMpy.materials.BK7, np.inf),
])

sol = EMpy.transfer_matrix.IsotropicTransferMatrix(
    ar_stack, EMpy.utils.deg2rad(10.0)
).solve(wls)
# sol.Rs, sol.Rp, sol.Ts, sol.Tp give power reflectance & transmittance
```

![Anti-reflection coating and TiO2/SiO2 DBR reflectance simulated with EMpy transfer matrix method]({{ '/images/transfer_matrix_ar_dbr.png' | relative_url }})

![Kretschmann surface plasmon resonance ATR dip and lossy-to-lossless TIR limit]({{ '/images/transfer_matrix_spr.png' | relative_url }})

---

### Benchmark Scripts & References

- [`ex_ar_coating_and_dbr.py`](https://github.com/lbolla/EMpy/blob/master/examples/transfer_matrix/ex_ar_coating_and_dbr.py) — Quarter-wave MgF₂ AR coating and 8-pair TiO₂/SiO₂ DBR vs. Macleod stopband formula (*Macleod 2010; Born & Wolf 1999*).
- [`ex_kretschmann_spr_lossy_vs_lossless.py`](https://github.com/lbolla/EMpy/blob/master/examples/transfer_matrix/ex_kretschmann_spr_lossy_vs_lossless.py) — Kretschmann `BK7 / 47 nm Au / Air` SPR dip and $$\kappa \to 0^+$$ TIR limit (*Kretschmann & Raether 1968*).
- [`ex_anisotropic_waveplate_and_limits.py`](https://github.com/lbolla/EMpy/blob/master/examples/transfer_matrix/ex_anisotropic_waveplate_and_limits.py) — 45° quartz half-wave plate vs. Jones calculus and 3D `euler_rotate` $$\Delta n \to 0$$ convergence (*Berreman 1972; Yeh 1979*).
- [`ex_laser_etch_monitor.py`](https://github.com/lbolla/EMpy/blob/master/examples/transfer_matrix/ex_laser_etch_monitor.py) — In-situ 670 nm RIE reflectometry on GaAs/AlGaAs VCSEL DBR.

**See also:** [Methodology & References]({{ '/methodology/#transfer-matrix' | relative_url }}) · [All Examples]({{ '/examples/#catalog' | relative_url }})
