import numpy
import EMpy_gpu
import pylab

# define the multilayer
epsilon = [1.0 ** 2 * EMpy_gpu.constants.eps0 * numpy.eye(3),
           EMpy_gpu.constants.eps0 * numpy.diag([2.1, 2.0, 1.9]),
           2.3 ** 2 * EMpy_gpu.constants.eps0 * numpy.eye(3),
           4.3 ** 2 * EMpy_gpu.constants.eps0 * numpy.eye(3),
           3.0 ** 2 * EMpy_gpu.constants.eps0 * numpy.eye(3)]

d = numpy.array([numpy.inf, 1e-6, 2.3e-6, 0.1e-6, numpy.inf])

aniso_layers = EMpy_gpu.utils.Multilayer()
for i in xrange(len(epsilon)):
    eps = EMpy_gpu.materials.EpsilonTensor(epsilon[i] * numpy.eye(3))
    mat = EMpy_gpu.materials.AnisotropicMaterial('layer_%d' % i, eps)
    layer = EMpy_gpu.utils.Layer(mat, d[i])
    aniso_layers.append(layer)

# define the planewave
theta_inc_x = EMpy_gpu.utils.deg2rad(0.)
theta_inc_y = 0.
wls = numpy.linspace(1.4e-6, 1.7e-6, 100)

# solve
tm = EMpy_gpu.transfer_matrix.AnisotropicTransferMatrix(
    aniso_layers,
    theta_inc_x,
    theta_inc_y)
solution_aniso = tm.solve(wls)

# plot
pylab.figure()
pylab.plot(wls, solution_aniso.R[0, 0, :],
           wls, solution_aniso.R[1, 0, :],
           wls, solution_aniso.R[0, 1, :],
           wls, solution_aniso.R[1, 1, :],
           wls, solution_aniso.T[0, 0, :],
           wls, solution_aniso.T[1, 0, :],
           wls, solution_aniso.T[0, 1, :],
           wls, solution_aniso.T[1, 1, :])
pylab.legend(('Rss', 'Rps', 'Rsp', 'Rpp', 'Tss', 'Tps', 'Tsp', 'Tpp'))
pylab.title('Anisotropic Multilayer')
pylab.xlabel('wavelength /m')
pylab.ylabel('Power /dB')
pylab.xlim(wls.min(), wls.max())
pylab.show()
