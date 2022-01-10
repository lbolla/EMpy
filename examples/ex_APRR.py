"""All-pass ring resonator example."""

import EMpy_gpu
import numpy
import pylab

wls = numpy.linspace(1.5e-6, 1.6e-6, 1000)
K = EMpy_gpu.devices.Coupler(wls, numpy.sqrt(0.08), 1.)
l = 2 * numpy.pi * 5e-6
SWG = EMpy_gpu.devices.SWG(400, 220, 125).solve(wls)
APRR = EMpy_gpu.devices.APRR(K, SWG.neff, l).solve()

pylab.plot(wls, numpy.unwrap(numpy.angle(APRR.THRU)), 'r.-')
pylab.axis('tight')
pylab.xlabel('wavelength /m')
pylab.ylabel('phase')
pylab.show()
