"""Rigorous Coupled Wave Analysis example."""

import numpy
import pylab

import EMpy
from EMpy.materials import (
    IsotropicMaterial, AnisotropicMaterial, RefractiveIndex, EpsilonTensor)


alpha = 0.
delta = 0.
# psi = EMpy.utils.deg2rad(0.)  # TM
# psi = EMpy.utils.deg2rad(90.)  # TE
psi = EMpy.utils.deg2rad(70.)  # hybrid
phi = EMpy.utils.deg2rad(90.)

LAMBDA = 1016e-9  # grating periodicity
n = 2  # orders of diffraction

UV6 = IsotropicMaterial(
    'UV6',
    n0=RefractiveIndex(n0_const=1.560))
SiN = AnisotropicMaterial(
    'SiN',
    epsilon_tensor=EpsilonTensor(
        epsilon_tensor_const=EMpy.constants.eps0 * EMpy.utils.euler_rotate(
            numpy.diag(numpy.asarray([1.8550, 1.8750, 1.9130]) ** 2),
            EMpy.utils.deg2rad(14),
            EMpy.utils.deg2rad(25),
            EMpy.utils.deg2rad(32))))
BPTEOS = IsotropicMaterial(
    'BPTEOS',
    n0=RefractiveIndex(n0_const=1.448))
ARC1 = IsotropicMaterial(
    'ARC1', n0=RefractiveIndex(n0_const=1.448))

EFF = IsotropicMaterial(
    'EFF', n0=RefractiveIndex(n0_const=1.6))

multilayer1 = EMpy.utils.Multilayer([
    EMpy.utils.Layer(EMpy.materials.Air, numpy.inf),
    EMpy.utils.Layer(SiN, 226e-9),
    EMpy.utils.Layer(BPTEOS, 226e-9),
    EMpy.utils.BinaryGrating(SiN, BPTEOS, .659, LAMBDA, 123e-9),
    EMpy.utils.Layer(SiN, 219e-9),
    EMpy.utils.Layer(EMpy.materials.SiO2, 2188e-9),
    EMpy.utils.Layer(EMpy.materials.Si, numpy.inf),
])

multilayer2 = EMpy.utils.Multilayer([
    EMpy.utils.Layer(EMpy.materials.Air, numpy.inf),
    EMpy.utils.Layer(SiN, 226e-9),
    EMpy.utils.Layer(BPTEOS, 226e-9),
    EMpy.utils.Layer(
        IsotropicMaterial(n0=RefractiveIndex(n0_const=1.6)), 123e-9),
    EMpy.utils.Layer(SiN, 219e-9),
    EMpy.utils.Layer(EMpy.materials.SiO2, 2188e-9),
    EMpy.utils.Layer(EMpy.materials.Si, numpy.inf),
])

wls = numpy.linspace(1.45e-6, 1.75e-6, 301)

solution1 = EMpy.RCWA.AnisotropicRCWA(
    multilayer1, alpha, delta, psi, phi, n).solve(wls)
solution2 = EMpy.RCWA.AnisotropicRCWA(
    multilayer2, alpha, delta, psi, phi, n).solve(wls)

um = 1e-6
pylab.plot(
    # wls / um, solution1.DEO1[n, :], 'k.-',
    # wls / um, solution1.DEO3[n, :], 'r.-',
    wls / um, solution1.DEE1[n, :], 'b.-',
    wls / um, solution1.DEE3[n, :], 'g.-',
    # wls / um, solution2.DEO1[n, :], 'k--',
    # wls / um, solution2.DEO3[n, :], 'r--',
    wls / um, solution2.DEE1[n, :], 'b--',
    wls / um, solution2.DEE3[n, :], 'g--',
)
pylab.xlabel('wavelength [um]')
pylab.ylabel('diffraction efficiency')
pylab.legend(('DEO1', 'DEO3', 'DEE1', 'DEE3'))
pylab.axis('tight')
pylab.ylim([0, 0.15])
pylab.show()
