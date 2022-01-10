"""Semi-vectorial finite-difference mode solver example."""

import numpy
import EMpy_gpu
import pylab


def epsfunc(x_, y_):
    """Return a matrix describing a 2d material.

    :param x_: x values
    :param y_: y values
    :return: 2d-matrix
    """
    xx, yy = numpy.meshgrid(x_, y_)
    return numpy.where((numpy.abs(xx.T - 1.24e-6) <= .24e-6) *
                       (numpy.abs(yy.T - 1.11e-6) <= .11e-6),
                       3.4757**2,
                       1.446**2)

wl = 1.55e-6
x = numpy.linspace(0, 2.48e-6, 125)
y = numpy.linspace(0, 2.22e-6, 112)

neigs = 2
tol = 1e-8
boundary = '0000'

solver = EMpy_gpu.modesolvers.FD.SVFDModeSolver(wl, x, y, epsfunc, boundary).solve(
    neigs, tol)

fig = pylab.figure()
fig.add_subplot(1, 2, 1)
pylab.contourf(abs(solver.Ex[0]), 50)
pylab.title('Ex first mode')
fig.add_subplot(1, 2, 2)
pylab.contourf(abs(solver.Ex[1]), 50)
pylab.title('Ex second mode')
pylab.show()
