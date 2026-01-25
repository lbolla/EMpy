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
