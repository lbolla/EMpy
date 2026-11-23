---
layout: default
title: 3D Vector Diffraction & Huygens–Kottler Scattering in Python
description: Compute 3D vector electromagnetic radiation and aperture diffraction from equivalent electric and magnetic surface currents using Kottler-Stratton-Chu integrals in Python.
permalink: /vector-diffraction-scattering/
---

## 3D Vector Diffraction & Huygens–Kottler Scattering in Python

`EMpy.scattering` evaluates **3D vector Huygens–Kottler–Stratton–Chu diffraction integrals** (`currentsScatteringKottler`) for arbitrary equivalent electric and magnetic surface currents $$(\mathbf{J}, \mathbf{M})$$ radiating into lossless or passive lossy homogeneous dielectric media.

---

### Capabilities

- **Full Vector Near-Field & Far-Field Diffraction:** Evaluates $$\mathbf{E}(\mathbf{r})$$ and $$\mathbf{H}(\mathbf{r})$$ at arbitrary 3D observation points in both Fresnel near-field and Fraunhofer far-field regimes.
- **Passive Lossy Media Support:** Supports complex relative permittivity $$\varepsilon_r = (n_r - i\kappa)^2$$ ($$\kappa \ge 0$$) with passive Green's function decay.
- **Field & Current Utilities:** Provides `Field` and `stack` helpers for composing aperture illuminations and evaluating E-plane and H-plane radiation patterns.

---

### Quick Python Example: Circular Aperture Vector Diffraction

```python
import EMpy

# p_eplane: (N, 3) observation coordinates
# j_free, m_free: equivalent electric & magnetic surface currents on aperture
# q_ap, ds_ap: aperture quadrature points and surface area weights
fld_e = EMpy.scattering.currentsScatteringKottler(
    p_eplane, j_free, m_free, q_ap, ds_ap, f0, epsr=1.0
)
```

![Vector Kottler circular aperture diffraction in E- and H-planes vs analytical Airy pattern]({{ '/images/scattering_kottler_airy.png' | relative_url }})

---

### Benchmark Script & References

- [`ex_kottler_aperture_diffraction.py`](https://github.com/lbolla/EMpy/blob/master/examples/scattering/ex_kottler_aperture_diffraction.py) — Circular aperture ($$2a = 5\lambda_0$$) E- and H-plane diffraction vs. analytical Airy pattern and on-axis Fresnel interference in vacuum and passive lossy media ($$\varepsilon_r = (1.4 - 0.03i)^2$$) (*Kottler 1923; Stratton & Chu 1939; Balanis 2012*).

**See also:** [Methodology & References]({{ '/methodology/#devices-materials' | relative_url }}) · [All Examples]({{ '/examples/#catalog' | relative_url }})
