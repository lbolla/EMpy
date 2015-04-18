"""Single ring resonator example."""

import EMpy
import numpy
import pylab

wls = numpy.linspace(1.53e-6, 1.56e-6, 1000)
K1 = EMpy.devices.Coupler(wls, numpy.sqrt(0.08), 1.)
K2 = EMpy.devices.Coupler(wls, numpy.sqrt(0.08), 1.)
l1 = numpy.pi * 5e-6
l2 = numpy.pi * 5e-6
SWG = EMpy.devices.SWG(488, 220, 25).solve(wls)
SRR = EMpy.devices.SRR(K1, K2, SWG.neff, l1, l2).solve()

pylab.plot(wls, numpy.absolute(SRR.THRU), 'r.-',
           wls, numpy.absolute(SRR.DROP), 'g.-')
pylab.axis('tight')
pylab.ylim([0, 1])
pylab.xlabel('wavelength /m')
pylab.ylabel('power')
pylab.legend(('THRU', 'DROP'))
pylab.show()

