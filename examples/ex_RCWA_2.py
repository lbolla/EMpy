"""Rigorous Coupled Wave Analysis example.

Inspired by Moharam, "Formulation for stable and efficient implementation of the rigorous coupled-wave analysis of
binary gratings", JOSA A, 12(5), 1995
"""

import numpy
import pylab

import EMpy_gpu
from EMpy_gpu.materials import IsotropicMaterial, RefractiveIndex


alpha = EMpy_gpu.utils.deg2rad(10.)
delta = EMpy_gpu.utils.deg2rad(0.)
psi = EMpy_gpu.utils.deg2rad(0.)  # TE
phi = EMpy_gpu.utils.deg2rad(90.)

wl = numpy.array([1.55e-6])
ds = numpy.linspace(0., 5., 100) * wl
LAMBDA = 10 * wl

n = 3  # orders of diffraction

Top = IsotropicMaterial('Top', n0=RefractiveIndex(n0_const=1.))
Bottom = IsotropicMaterial('Bottom', n0=RefractiveIndex(n0_const=2.04))

solutions = []
for d in ds:
    multilayer = EMpy_gpu.utils.Multilayer([
        EMpy_gpu.utils.Layer(Top, numpy.inf),
        EMpy_gpu.utils.BinaryGrating(Top, Bottom, .3, LAMBDA, d),
        EMpy_gpu.utils.Layer(Bottom, numpy.inf),
    ])

    solution = EMpy_gpu.RCWA.IsotropicRCWA(multilayer, alpha, delta, psi, phi, n).solve(wl)
    solutions.append(solution)

DE1 = numpy.zeros(len(solutions))
DE3 = numpy.zeros(len(solutions))
for ss, s in enumerate(solutions):
    DE1[ss] = s.DE1[n, 0]
    DE3[ss] = s.DE3[n, 0]

pylab.plot(ds / wl, DE1[:], 'k.-',
           ds / wl, DE3[:], 'r.-')
pylab.xlabel('normalized groove depth')
pylab.ylabel('diffraction efficiency')
pylab.legend(('DE1', 'DE3'))
pylab.axis('tight')
pylab.ylim([0, 1])
pylab.show()

