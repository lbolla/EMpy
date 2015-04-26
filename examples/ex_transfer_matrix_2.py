# taken from http://hyperphysics.phy-astr.gsu.edu/hbase/phyopt/antiref.html#c1

import numpy
import pylab

import EMpy

# define multilayer
n = numpy.array([1., 1.38, 1.9044])
d = numpy.array([numpy.inf, 387.5e-9 / 1.38, numpy.inf])
iso_layers = EMpy.utils.Multilayer()
for i in xrange(n.size):
    n0 = EMpy.materials.RefractiveIndex(n[i])
    iso_layers.append(
        EMpy.utils.Layer(EMpy.materials.IsotropicMaterial('mat', n0=n0), d[i]))

# define incident wave plane
theta_inc = EMpy.utils.deg2rad(10.)
wls = numpy.linspace(0.85e-6, 2.25e-6, 300)

# solve
tm = EMpy.transfer_matrix.IsotropicTransferMatrix(iso_layers, theta_inc)
solution_iso = tm.solve(wls)

# plot
pylab.figure()
pylab.plot(wls, 10 * numpy.log10(solution_iso.Rs), 'rx-',
           wls, 10*numpy.log10(solution_iso.Rp), 'g.-')
pylab.legend(('Rs', 'Rp'))
pylab.title('Single Layer Anti-Reflection Coating')
pylab.xlabel('wavelength /m')
pylab.ylabel('Power /dB')
pylab.grid()
pylab.xlim(wls.min(), wls.max())
pylab.savefig(__file__ + '.png')
pylab.show()
