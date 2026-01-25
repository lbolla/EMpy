"""All-pass ring resonator example."""

from matplotlib import pyplot as plt
import numpy

import EMpy

wls = numpy.linspace(1.5e-6, 1.6e-6, 1000)
K = EMpy.devices.Coupler(wls, numpy.sqrt(0.08), 1.0)
l = 2 * numpy.pi * 5e-6
SWG = EMpy.devices.SWG(400, 220, 125).solve(wls)
APRR = EMpy.devices.APRR(K, SWG.neff, l).solve()

plt.plot(wls, numpy.unwrap(numpy.angle(APRR.THRU)), "r.-")
plt.axis("tight")
plt.xlabel("wavelength /m")
plt.ylabel("phase")
plt.show()
