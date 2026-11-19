---
layout: default
---

## EMpy

`EMpy` (Electromagnetic Python) (not to be confused with [empy](http://www.alcyone.com/software/empy/)) is a suite of numerical algorithms widely used in electromagnetism.

The package contains:

- an isotropic and anisotropic transfer matrix algorithm;
- an isotropic and anisotropic rigorous coupled wave analysis (RCWA) algorithm;
- the numerical model of the frequency response of different well-known devices (Mach-Zehnder, Ring Resonators, etc.);
- a finite-difference fully vectorial modesolver.

Install via PyPI:

```bash
pip install ElectromagneticPython
```

---

## Examples {#examples}

More (and more up-to-date) examples are [available in the source code](https://github.com/lbolla/EMpy/tree/master/examples).

Here are some simple examples of EMpy's functionalities:

- [Single Layer Anti-Reflection Coating](#example_slarc)
- [Anisotropic Multilayer](#example_am)
- [Single Ring Resonator](#example_srr)
- [N-Rings Resonator](#example_nrr)
- [Finite Difference Fully Vectorial Modesolver](#example_modesolver)
- [Multilayer with grating](#example_RCWA)

### Single Layer Anti-Reflection Coating {#example_slarc}

The isotropic transfer matrix algorithm can be used to study [anti-reflection coatings](https://en.wikipedia.org/wiki/Anti-reflective_coating). A very simple single layer anti-reflection coating, described [here](http://hyperphysics.phy-astr.gsu.edu/hbase/phyopt/antiref.html#c1), is computed by the following script ([`examples/ex_transfer_matrix_2.py`](https://github.com/lbolla/EMpy/blob/master/examples/ex_transfer_matrix_2.py)):

```python
from matplotlib import pyplot as plt
import numpy

import EMpy

# define multilayer
n = numpy.array([1.0, 1.38, 1.9044])
d = numpy.array([numpy.inf, 387.5e-9 / 1.38, numpy.inf])
iso_layers = EMpy.utils.Multilayer()
for i in range(n.size):
    n0 = EMpy.materials.RefractiveIndex(n[i])
    iso_layers.append(
        EMpy.utils.Layer(EMpy.materials.IsotropicMaterial("mat", n0=n0), d[i])
    )

# define incident wave plane
theta_inc = EMpy.utils.deg2rad(10.0)
wls = numpy.linspace(0.85e-6, 2.25e-6, 300)

# solve
tm = EMpy.transfer_matrix.IsotropicTransferMatrix(iso_layers, theta_inc)
solution_iso = tm.solve(wls)

# plot
plt.figure()
plt.plot(
    wls,
    10 * numpy.log10(solution_iso.Rs),
    "rx-",
    wls,
    10 * numpy.log10(solution_iso.Rp),
    "g.-",
)
plt.legend(("Rs", "Rp"))
plt.title("Single Layer Anti-Reflection Coating")
plt.xlabel("wavelength /m")
plt.ylabel("Power /dB")
plt.grid()
plt.xlim(wls.min(), wls.max())
plt.show()
```

In lines 7–14, the multilayer is defined. Indices and thicknesses are chosen so that a single layer of refractive index 1.38 and thickness about one quarter of the central wavelength 1.55 µm is positioned between two semi-infinite layers of air and glass. This gives theoretically no reflection for the central wavelength. The following graph is the result, which confirms the theoretical predictions.

![Single Layer Anti-Reflection Coating](images/single_layer_ar_coating.png)

The graph shows two curves, one for each polarization. They differ because the incident plane wave is tilted by 10 degrees (line 17).

[back to examples](#examples)

### Anisotropic Multilayer {#example_am}

The anisotropic transfer-matrix algorithm can be used to study the following example ([`examples/ex_transfer_matrix_3.py`](https://github.com/lbolla/EMpy/blob/master/examples/ex_transfer_matrix_3.py)):

```python
from matplotlib import pyplot as plt
import numpy

import EMpy

# define the multilayer
epsilon = [
    1.0**2 * EMpy.constants.eps0 * numpy.eye(3),
    EMpy.constants.eps0 * numpy.diag([2.1, 2.0, 1.9]),
    2.3**2 * EMpy.constants.eps0 * numpy.eye(3),
    4.3**2 * EMpy.constants.eps0 * numpy.eye(3),
    3.0**2 * EMpy.constants.eps0 * numpy.eye(3),
]

d = numpy.array([numpy.inf, 1e-6, 2.3e-6, 0.1e-6, numpy.inf])

aniso_layers = EMpy.utils.Multilayer()
for i in range(len(epsilon)):
    eps = EMpy.materials.EpsilonTensor(epsilon[i] * numpy.eye(3))
    mat = EMpy.materials.AnisotropicMaterial("layer_%d" % i, eps)
    layer = EMpy.utils.Layer(mat, d[i])
    aniso_layers.append(layer)

# define the planewave
theta_inc_x = EMpy.utils.deg2rad(0.0)
theta_inc_y = 0.0
wls = numpy.linspace(1.4e-6, 1.7e-6, 100)

# solve
tm = EMpy.transfer_matrix.AnisotropicTransferMatrix(
    aniso_layers, theta_inc_x, theta_inc_y
)
solution_aniso = tm.solve(wls)

# plot
plt.figure()
plt.plot(
    wls,
    solution_aniso.R[0, 0, :],
    wls,
    solution_aniso.R[1, 0, :],
    wls,
    solution_aniso.R[0, 1, :],
    wls,
    solution_aniso.R[1, 1, :],
    wls,
    solution_aniso.T[0, 0, :],
    wls,
    solution_aniso.T[1, 0, :],
    wls,
    solution_aniso.T[0, 1, :],
    wls,
    solution_aniso.T[1, 1, :],
)
plt.legend(("Rss", "Rps", "Rsp", "Rpp", "Tss", "Tps", "Tsp", "Tpp"))
plt.title("Anisotropic Multilayer")
plt.xlabel("wavelength /m")
plt.ylabel("Power /dB")
plt.xlim(wls.min(), wls.max())
plt.show()
```

The result is shown in the following graph.

![Anisotropic Multilayer](images/anisotropic_multilayer.png)

Four curves are clearly visible because the two polarizations behave in a different way, even though the incidence is normal (lines 25–26), because of the anisotropic material (line 9).

[back to examples](#examples)

### Single Ring Resonator {#example_srr}

This is the script to study a single [ring resonator](https://en.wikipedia.org/wiki/Optical_ring_resonators) ([`examples/ex_SRR.py`](https://github.com/lbolla/EMpy/blob/master/examples/ex_SRR.py)):

```python
"""Single ring resonator example."""

from matplotlib import pyplot as plt
import numpy

import EMpy

wls = numpy.linspace(1.53e-6, 1.56e-6, 1000)
K1 = EMpy.devices.Coupler(wls, numpy.sqrt(0.08), 1.0)
K2 = EMpy.devices.Coupler(wls, numpy.sqrt(0.08), 1.0)
l1 = numpy.pi * 5e-6
l2 = numpy.pi * 5e-6
SWG = EMpy.devices.SWG(488, 220, 25).solve(wls)
SRR = EMpy.devices.SRR(K1, K2, SWG.neff, l1, l2).solve()

plt.plot(wls, numpy.absolute(SRR.THRU), "r.-", wls, numpy.absolute(SRR.DROP), "g.-")
plt.axis("tight")
plt.ylim([0, 1])
plt.xlabel("wavelength /m")
plt.ylabel("power")
plt.legend(("THRU", "DROP"))
plt.show()
```

In lines 9 and 10, the two coupling coefficients and the two coupler losses are defined. The half-roundtrip lengths of the ring are defined in lines 11 and 12, while its effective index is that of a silicon rib waveguide of 488 nm × 220 nm at room temperature.

This is its spectrum:

![Single Ring Resonator](images/single_ring_resonator.png)

The maximum drop power is 0 dB (1 in linear units) because no losses are considered, and it is periodic with the period being the FSR of the ring.

[back to examples](#examples)

### N-Rings Resonator {#example_nrr}

This is the script to study a ring resonator made of 3 rings ([`examples/ex_NRR.py`](https://github.com/lbolla/EMpy/blob/master/examples/ex_NRR.py)):

```python
"""N-Ring resonators example."""

from matplotlib import pyplot as plt
import numpy

import EMpy

wls = numpy.linspace(1.53e-6, 1.57e-6, 1000)

Ks = [
    EMpy.devices.Coupler(wls, numpy.sqrt(0.08), 1.0),
    EMpy.devices.Coupler(wls, numpy.sqrt(0.008), 1.0),
    EMpy.devices.Coupler(wls, numpy.sqrt(0.006), 1.0),
    EMpy.devices.Coupler(wls, numpy.sqrt(0.09), 1.0),
]

R = 5e-6
l1s = [numpy.pi * R, numpy.pi * R, numpy.pi * R]
l2s = [numpy.pi * R, numpy.pi * R, numpy.pi * R]

SWG = EMpy.devices.SWG(400, 220, 125).solve(wls)
neffs = [SWG.neff, SWG.neff, SWG.neff]

NRR = EMpy.devices.NRR(Ks, neffs, l1s, l2s).solve()

plt.plot(
    wls,
    20 * numpy.log10(numpy.absolute(NRR.THRU)),
    "r.-",
    wls,
    20 * numpy.log10(numpy.absolute(NRR.DROP)),
    "g.-",
)
plt.axis("tight")
plt.ylim([-30, 0])
plt.xlabel("wavelength /m")
plt.ylabel("power /dB")
plt.legend(("THRU", "DROP"))
plt.show()
```

After the imports, a list of couplers is defined: four, in this case, because the resonator is made of 3 rings. Then, the arcs connecting the resonators are defined: the rings are all equal, and the couplers are on the diameters of the rings. Finally, the rings have an effective index equal to a 400 nm × 220 nm straight waveguide at 125 degrees.

This is its spectrum:

![N-Ring Resonator](images/n_ring_resonator.png)

The drop is always lower than 0 dB because the coupling coefficients are not optimized.

[back to examples](#examples)

### Finite Difference Fully Vectorial Modesolver {#example_modesolver}

This is the script to find the first two modes of a channel SOI waveguide ([`examples/ex_modesolver.py`](https://github.com/lbolla/EMpy/blob/master/examples/ex_modesolver.py)):

```python
"""Fully vectorial finite-difference mode solver example."""

from matplotlib import pyplot as plt
import numpy

import EMpy


def epsfunc(x_, y_):
    """Return a matrix describing a 2d material.

    :param x_: x values
    :param y_: y values
    :return: 2d-matrix
    """
    xx, yy = numpy.meshgrid(x_, y_)
    return numpy.where(
        (numpy.abs(xx.T - 1.24e-6) <= 0.24e-6) * (numpy.abs(yy.T - 1.11e-6) <= 0.11e-6),
        3.4757**2,
        1.446**2,
    )


wl = 1.55e-6
x = numpy.linspace(0, 2.48e-6, 125)
y = numpy.linspace(0, 2.22e-6, 112)

neigs = 2
tol = 1e-8
boundary = "0000"

solver = EMpy.modesolvers.FD.VFDModeSolver(wl, x, y, epsfunc, boundary).solve(
    neigs, tol
)

fig = plt.figure()

fig.add_subplot(1, 3, 1)
Ex = numpy.transpose(solver.modes[0].get_field("Ex", x, y))
plt.contourf(x, y, abs(Ex), 50)
plt.title("Ex")

fig.add_subplot(1, 3, 2)
Ey = numpy.transpose(solver.modes[0].get_field("Ey", x, y))
plt.contourf(x, y, abs(Ey), 50)
plt.title("Ey")

fig.add_subplot(1, 3, 3)
Ez = numpy.transpose(solver.modes[0].get_field("Ez", x, y))
plt.contourf(x, y, abs(Ez), 50)
plt.title("Ez")

fig.add_subplot(1, 3, 1)
Hx = numpy.transpose(solver.modes[0].get_field("Hx", x, y))
plt.contourf(x, y, abs(Hx), 50)
plt.title("Hx")

fig.add_subplot(1, 3, 2)
Hy = numpy.transpose(solver.modes[0].get_field("Hy", x, y))
plt.contourf(x, y, abs(Hy), 50)
plt.title("Hy")

fig.add_subplot(1, 3, 3)
Hz = numpy.transpose(solver.modes[0].get_field("Hz", x, y))
plt.contourf(x, y, abs(Hz), 50)
plt.title("Hz")

plt.show()
```

This is a screenshot of the first (TE) mode:

![Mode Channel WG](images/mode_channel.png)

[back to examples](#examples)

### Multilayer with grating {#example_RCWA}

This is the script to study the reflectivity of a grating embedded in an anisotropic multilayer ([`examples/ex_RCWA.py`](https://github.com/lbolla/EMpy/blob/master/examples/ex_RCWA.py)). The multilayer is made of 7 layers, the 4th being a binary grating made with an anisotropic material (SiN, rotated with the optical axis not aligned with the coordinate axis) and an isotropic material (BPTEOS). Incidence is not normal. A comparison of the reflectivity is made with a similar multilayer, where the grating layer has been substituted by an "effective" isotropic layer.

```python
"""Rigorous Coupled Wave Analysis example."""

from matplotlib import pyplot as plt
import numpy

import EMpy
from EMpy.materials import (
    IsotropicMaterial,
    AnisotropicMaterial,
    RefractiveIndex,
    EpsilonTensor,
)

alpha = 0.0
delta = 0.0
psi = EMpy.utils.deg2rad(70.0)  # hybrid
phi = EMpy.utils.deg2rad(90.0)

LAMBDA = 1016e-9  # grating periodicity
n = 2  # orders of diffraction

UV6 = IsotropicMaterial("UV6", n0=RefractiveIndex(n0_const=1.560))
SiN = AnisotropicMaterial(
    "SiN",
    epsilon_tensor=EpsilonTensor(
        epsilon_tensor_const=EMpy.constants.eps0
        * EMpy.utils.euler_rotate(
            numpy.diag(numpy.asarray([1.8550, 1.8750, 1.9130]) ** 2),
            EMpy.utils.deg2rad(14),
            EMpy.utils.deg2rad(25),
            EMpy.utils.deg2rad(32),
        )
    ),
)
BPTEOS = IsotropicMaterial("BPTEOS", n0=RefractiveIndex(n0_const=1.448))
ARC1 = IsotropicMaterial("ARC1", n0=RefractiveIndex(n0_const=1.448))

EFF = IsotropicMaterial("EFF", n0=RefractiveIndex(n0_const=1.6))

multilayer1 = EMpy.utils.Multilayer(
    [
        EMpy.utils.Layer(EMpy.materials.Air, numpy.inf),
        EMpy.utils.Layer(SiN, 226e-9),
        EMpy.utils.Layer(BPTEOS, 226e-9),
        EMpy.utils.BinaryGrating(SiN, BPTEOS, 0.659, LAMBDA, 123e-9),
        EMpy.utils.Layer(SiN, 219e-9),
        EMpy.utils.Layer(EMpy.materials.SiO2, 2188e-9),
        EMpy.utils.Layer(EMpy.materials.Si, numpy.inf),
    ]
)

multilayer2 = EMpy.utils.Multilayer(
    [
        EMpy.utils.Layer(EMpy.materials.Air, numpy.inf),
        EMpy.utils.Layer(SiN, 226e-9),
        EMpy.utils.Layer(BPTEOS, 226e-9),
        EMpy.utils.Layer(IsotropicMaterial(n0=RefractiveIndex(n0_const=1.6)), 123e-9),
        EMpy.utils.Layer(SiN, 219e-9),
        EMpy.utils.Layer(EMpy.materials.SiO2, 2188e-9),
        EMpy.utils.Layer(EMpy.materials.Si, numpy.inf),
    ]
)

wls = numpy.linspace(1.45e-6, 1.75e-6, 301)

solution1 = EMpy.RCWA.AnisotropicRCWA(multilayer1, alpha, delta, psi, phi, n).solve(wls)
solution2 = EMpy.RCWA.AnisotropicRCWA(multilayer2, alpha, delta, psi, phi, n).solve(wls)

um = 1e-6
plt.plot(
    wls / um,
    solution1.DEE1[n, :],
    "b.-",
    wls / um,
    solution1.DEE3[n, :],
    "g.-",
    wls / um,
    solution2.DEE1[n, :],
    "b--",
    wls / um,
    solution2.DEE3[n, :],
    "g--",
)
plt.xlabel("wavelength [um]")
plt.ylabel("diffraction efficiency")
plt.legend(("DEO1", "DEO3", "DEE1", "DEE3"))
plt.axis("tight")
plt.ylim([0, 0.15])
plt.show()
```

The reflectivities are shown in the picture below (in dashed lines is the reflectivity without the grating): the effect of the grating is clearly visible as a peak in the reflectivity.

![RCWA](images/RCWA.png)

[back to examples](#examples)
