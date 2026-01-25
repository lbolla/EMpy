"""Fully vectorial finite-difference mode solver example."""

from matplotlib import pyplot as plt
import numpy

import EMpy


def epsfunc(x_, y_):
    """Return a matrix describing a 2d material.

    :param x_: x values
    :param y_: y values
    :return: 2d-matrix
    """
    xx, yy = numpy.meshgrid(x_, y_)
    return numpy.where(
        (numpy.abs(xx.T - 1.24e-6) <= 0.24e-6) * (numpy.abs(yy.T - 1.11e-6) <= 0.11e-6),
        3.4757**2,
        1.446**2,
    )


wl = 1.55e-6
x = numpy.linspace(0, 2.48e-6, 125)
y = numpy.linspace(0, 2.22e-6, 112)

neigs = 2
tol = 1e-8
boundary = "0000"

solver = EMpy.modesolvers.FD.VFDModeSolver(wl, x, y, epsfunc, boundary).solve(
    neigs, tol
)

fig = plt.figure()

fig.add_subplot(1, 3, 1)
Ex = numpy.transpose(solver.modes[0].get_field("Ex", x, y))
plt.contourf(x, y, abs(Ex), 50)
plt.title("Ex")

fig.add_subplot(1, 3, 2)
Ey = numpy.transpose(solver.modes[0].get_field("Ey", x, y))
plt.contourf(x, y, abs(Ey), 50)
plt.title("Ey")

fig.add_subplot(1, 3, 3)
Ez = numpy.transpose(solver.modes[0].get_field("Ez", x, y))
plt.contourf(x, y, abs(Ez), 50)
plt.title("Ez")

fig.add_subplot(1, 3, 1)
Hx = numpy.transpose(solver.modes[0].get_field("Hx", x, y))
plt.contourf(x, y, abs(Hx), 50)
plt.title("Hx")

fig.add_subplot(1, 3, 2)
Hy = numpy.transpose(solver.modes[0].get_field("Hy", x, y))
plt.contourf(x, y, abs(Hy), 50)
plt.title("Hy")

fig.add_subplot(1, 3, 3)
Hz = numpy.transpose(solver.modes[0].get_field("Hz", x, y))
plt.contourf(x, y, abs(Hz), 50)
plt.title("Hz")

plt.show()
