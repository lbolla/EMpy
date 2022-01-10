"""Fully vectorial finite-difference mode solver example."""

import numpy
import EMpy_gpu
import pylab


def epsfunc(x_, y_):
    '''Similar to ex_modesolver.py, but using anisotropic eps.'''
    eps = numpy.zeros((len(x_), len(y_), 5))
    for ix, xx in enumerate(x_):
        for iy, yy in enumerate(y_):
            if abs(xx - 1.24e-6) <= .24e-6 and abs(yy - 1.11e-6) <= .11e-6:
                a = 3.4757**2
                b = 1  # some xy value
                # eps_xx, xy, yx, yy, zz
                eps[ix, iy, :] = [a, b, b, a, a]
            else:
                a = 1.446**2
                # isotropic
                eps[ix, iy, :] = [a, 0, 0, a, a]
    return eps


wl = 1.55e-6
x = numpy.linspace(0, 2.48e-6, 125)
y = numpy.linspace(0, 2.22e-6, 112)

neigs = 2
tol = 1e-8
boundary = '0000'

solver = EMpy_gpu.modesolvers.FD.VFDModeSolver(wl, x, y, epsfunc, boundary).solve(
    neigs, tol)

fig = pylab.figure()
fig.add_subplot(1, 3, 1)
pylab.contourf(abs(solver.modes[0].Ex), 50)
pylab.title('Ex')
fig.add_subplot(1, 3, 2)
pylab.contourf(abs(solver.modes[0].Ey), 50)
pylab.title('Ey')
fig.add_subplot(1, 3, 3)
pylab.contourf(abs(solver.modes[0].Ez), 50)
pylab.title('Ez')
pylab.show()
