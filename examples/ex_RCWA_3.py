"""Rigorous Coupled Wave Analysis example."""

import numpy
import pylab

import EMpy_gpu
from EMpy_gpu.materials import IsotropicMaterial, RefractiveIndex


alpha = EMpy_gpu.utils.deg2rad(30.)
delta = EMpy_gpu.utils.deg2rad(45.)
psi = EMpy_gpu.utils.deg2rad(0.)  # TE
phi = EMpy_gpu.utils.deg2rad(90.)

wls = numpy.linspace(1.5495e-6, 1.550e-6, 101)

LAMBDA = 1.e-6  # grating periodicity
n = 3  # orders of diffraction

Top = IsotropicMaterial('Top', n0=RefractiveIndex(n0_const=1.))
Bottom = IsotropicMaterial('Bottom', n0=RefractiveIndex(n0_const=3.47))

multilayer = EMpy_gpu.utils.Multilayer([
    EMpy_gpu.utils.Layer(Top, numpy.inf),
    EMpy_gpu.utils.BinaryGrating(Top, Bottom, .4, LAMBDA, .01),
    EMpy_gpu.utils.Layer(Bottom, numpy.inf),
])

solution = EMpy_gpu.RCWA.IsotropicRCWA(multilayer, alpha, delta, psi, phi, n).solve(wls)

pylab.plot(wls, solution.DE1[n, :], 'ko-',
           wls, solution.DE3[n, :], 'ro-',
           wls, solution.DE1[n - 1, :], 'kx-',
           wls, solution.DE3[n - 1, :], 'rx-',
           wls, solution.DE1[n + 1, :], 'k.-',
           wls, solution.DE3[n + 1, :], 'r.-',
           )
pylab.xlabel('wavelength /m')
pylab.ylabel('diffraction efficiency')
pylab.legend(('DE1:0', 'DE3:0', 'DE1:-1', 'DE3:-1', 'DE1:+1', 'DE3:+1'))
pylab.axis('tight')
pylab.ylim([0, 1])
pylab.show()

