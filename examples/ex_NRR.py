"""N-Ring resonators example."""

import EMpy
import numpy
import pylab

wls = numpy.linspace(1.53e-6, 1.57e-6, 1000)

Ks = [EMpy.devices.Coupler(wls, numpy.sqrt(0.08), 1.),
      EMpy.devices.Coupler(wls, numpy.sqrt(0.008), 1.),
      EMpy.devices.Coupler(wls, numpy.sqrt(0.006), 1.),
      EMpy.devices.Coupler(wls, numpy.sqrt(0.09), 1.)]

R = 5e-6
l1s = [numpy.pi * R, numpy.pi * R, numpy.pi * R]
l2s = [numpy.pi * R, numpy.pi * R, numpy.pi * R]

SWG = EMpy.devices.SWG(400, 220, 125).solve(wls)
neffs = [SWG.neff, SWG.neff, SWG.neff]

NRR = EMpy.devices.NRR(Ks, neffs, l1s, l2s).solve()

pylab.plot(wls, 20 * numpy.log10(numpy.absolute(NRR.THRU)), 'r.-',
           wls, 20 * numpy.log10(numpy.absolute(NRR.DROP)), 'g.-')
pylab.axis('tight')
pylab.ylim([-30, 0])
pylab.xlabel('wavelength /m')
pylab.ylabel('power /dB')
pylab.legend(('THRU', 'DROP'))
pylab.show()

