---
layout: default
title: Rigorous Coupled-Wave Analysis (RCWA) in Python
description: Simulate 1D/2D surface-relief gratings, conical diffraction, guided-mode resonances, dimer BIC metasurfaces, and 3D birefringent gratings in Python with EMpy.RCWA.
permalink: /rcwa-gratings/
---

## Rigorous Coupled-Wave Analysis (RCWA) in Python

`EMpy.RCWA` implements **Rigorous Coupled-Wave Analysis** (also known as the Fourier Modal Method) for periodic multilayer gratings in planar, conical, and 3D anisotropic regimes.

---

### Solvers & Numerical Formulation

- **`IsotropicRCWA`:** Planar and 3D conical diffraction solver for multilayer surface-relief gratings (`BinaryGrating`, `SymmetricDoubleGrating`, `AsymmetricDoubleGrating`) combining Moharam's 1995 eigenmode formulation, Lalanne & Morris's inverse-rule TM Fourier factorization, and Moharam's **Enhanced Transmittance Matrix (ETM)** recursion to eliminate exponential overflow in deep grooves.
- **`AnisotropicRCWA`:** Full 3D vector coupled-wave solver for arbitrarily Euler-rotated birefringent and lossy gratings with isotropic or uniaxial semi-infinite end layers, applying Li's 1998 inverse-rule factorization across discontinuous permittivity walls.

---

### Quick Python Example: Symmetry-Protected BIC & Quasi-BIC in a Si₃N₄ Dimer Grating

```python
import numpy as np
import EMpy

air = EMpy.materials.IsotropicMaterial("Air", n0=1.0)
sin = EMpy.materials.IsotropicMaterial("SiN", n0=2.0)
sio2 = EMpy.materials.IsotropicMaterial("SiO2", n0=1.45)

ml_asym = EMpy.utils.Multilayer([
    EMpy.utils.Layer(air, np.inf),
    EMpy.utils.AsymmetricDoubleGrating(
        sin, sin, air, dc1=0.22, dc2=0.18, dc_spacer=0.30, pitch=1.0e-6, thickness=220e-9
    ),
    EMpy.utils.Layer(sin, 180e-9),
    EMpy.utils.Layer(sio2, np.inf),
])

wls = np.linspace(1.40e-6, 1.65e-6, 501)
sol = EMpy.RCWA.IsotropicRCWA(ml_asym, 0.0, 0.0, np.pi / 2.0, np.pi / 2.0, n=4).solve(wls)
# sol.DE1, sol.DE3 give reflected & transmitted diffraction efficiencies per order
```

![Dimer metasurface symmetry-protected BIC to quasi-BIC Fano resonance simulated with EMpy RCWA]({{ '/images/rcwa_dimer_bic.png' | relative_url }})

![3D anisotropic guided-mode resonance vs Rytov effective-medium theory]({{ '/images/rcwa_anisotropic_gmr.png' | relative_url }})

---

### Benchmark Scripts & References

- [`ex_double_grating_dimer_bic.py`](https://github.com/lbolla/EMpy/blob/master/examples/rcwa/ex_double_grating_dimer_bic.py) — Symmetry-protected dark BIC vs. quasi-BIC $$Q \propto \alpha^{-2}$$ scaling (*Koshelev et al. 2018; Moharam et al. 1995*).
- [`ex_anisotropic_guided_mode_resonance.py`](https://github.com/lbolla/EMpy/blob/master/examples/rcwa/ex_anisotropic_guided_mode_resonance.py) — 3D Euler-rotated birefringent SiN GMR filter vs. Rytov effective-medium theory (*Wang & Magnusson 1993; Li 1998*).
- [`ex_moharam_binary_grating.py`](https://github.com/lbolla/EMpy/blob/master/examples/rcwa/ex_moharam_binary_grating.py) — Littrow groove-depth ETM sweep & Lalanne TM harmonic convergence on dielectric and Au gratings.
- [`ex_conical_diffraction_and_rayleigh.py`](https://github.com/lbolla/EMpy/blob/master/examples/rcwa/ex_conical_diffraction_and_rayleigh.py) — 3D conical Wood–Rayleigh anomalies and `AnisotropicRCWA → IsotropicRCWA` limits.

**See also:** [Methodology & References]({{ '/methodology/#rcwa' | relative_url }}) · [All Examples]({{ '/examples/#catalog' | relative_url }})
