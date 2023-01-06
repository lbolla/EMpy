"""Fully vectorial finite-difference mode solver example.

Flexible mode solver example for fundamental modes
by David Hutchings, School of Engineering, University of Glasgow
David.Hutchings@glasgow.ac.uk
"""

import numpy
import EMpy
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy import signal

"""Define cross-section geometry of simulation
each rectangular element tuple contains
("label",xmin,ymin,xmax,ymax,eps=refractive index squared)
first element tuple defines simulation extent
units should match units of wavelength wl
"""
geom = (
    ("base", -0.8, -0.8, 0.8, 1.0, 1.0**2),
    ("substrate", -0.8, -0.8, 0.8, 0.0, 1.5**2),
    ("core1", -0.30, 0.0, 0.0, 0.34, 3.4**2),
    ("core2", 0.0, 0.0, 0.30, 0.145, 3.4**2),
)

wl = 1.55
nx, ny = 161, 181


def epsfunc(x_, y_):
    """Return a matrix describing a 2d material.

    :param x_: x values
    :param y_: y values
    :return: 2d-matrix
    """
    # xx, yy = numpy.meshgrid(x_, y_)
    working = geom[0][5] * numpy.ones((x_.size, y_.size))
    for i in range(1, len(geom)):
        ixmin = numpy.searchsorted(x_, geom[i][1], side="left")
        iymin = numpy.searchsorted(y_, geom[i][2], side="left")
        ixmax = numpy.searchsorted(x_, geom[i][3], side="right")
        iymax = numpy.searchsorted(y_, geom[i][4], side="right")
        working[ixmin:ixmax, iymin:iymax] = geom[i][5]

    return working


x = numpy.linspace(geom[0][1], geom[0][3], nx)
y = numpy.linspace(geom[0][2], geom[0][4], ny)

neigs = 2
tol = 1e-8
boundary = "0000"

solver = EMpy.modesolvers.FD.VFDModeSolver(wl, x, y, epsfunc, boundary).solve(
    neigs, tol
)

levls = numpy.geomspace(1.0 / 32.0, 1.0, num=11)
levls2 = numpy.geomspace(1.0 / 1024.0, 1.0, num=11)
xe = signal.convolve(x, [0.5, 0.5])[1:-1]
ye = signal.convolve(y, [0.5, 0.5])[1:-1]


def geom_outline():
    ax.set_xlim(geom[0][1], geom[0][3])
    ax.set_ylim(geom[0][2], geom[0][4])
    for i in range(1, len(geom)):
        plt.hlines(geom[i][2], geom[i][1], geom[i][3])
        plt.hlines(geom[i][4], geom[i][1], geom[i][3])
        plt.vlines(geom[i][1], geom[i][2], geom[i][4])
        plt.vlines(geom[i][3], geom[i][2], geom[i][4])
    return


print(solver.modes[0].neff)
fmax = abs(solver.modes[0].Ex).max()
fig = plt.figure()
ax = fig.add_subplot(2, 3, 1)
plt.contour(
    xe,
    ye,
    abs(solver.modes[0].Ex.T),
    fmax * levls,
    cmap="jet",
    locator=ticker.LogLocator(),
)
plt.title("Ex")
geom_outline()
ax = fig.add_subplot(2, 3, 2)
plt.contour(
    xe,
    ye,
    abs(solver.modes[0].Ey.T),
    fmax * levls,
    cmap="jet",
    locator=ticker.LogLocator(),
)
plt.title("Ey")
geom_outline()
ax = fig.add_subplot(2, 3, 3)
plt.contour(
    xe,
    ye,
    abs(solver.modes[0].Ez.T),
    fmax * levls,
    cmap="jet",
    locator=ticker.LogLocator(),
)
plt.title("Ez")
geom_outline()
fmax = abs(solver.modes[0].Hy).max()
ax = fig.add_subplot(2, 3, 4)
plt.contour(
    x,
    y,
    abs(solver.modes[0].Hx.T),
    fmax * levls,
    cmap="jet",
    locator=ticker.LogLocator(),
)
plt.title("Hx")
geom_outline()
ax = fig.add_subplot(2, 3, 5)
plt.contour(
    x,
    y,
    abs(solver.modes[0].Hy.T),
    fmax * levls,
    cmap="jet",
    locator=ticker.LogLocator(),
)
plt.title("Hy")
geom_outline()
ax = fig.add_subplot(2, 3, 6)
plt.contour(
    x,
    y,
    abs(solver.modes[0].Hz.T),
    fmax * levls,
    cmap="jet",
    locator=ticker.LogLocator(),
)
plt.title("Hz")
geom_outline()
plt.show()

ExatH = signal.convolve2d(solver.modes[0].Ex, [[0.25, 0.25], [0.25, 0.25]])
EyatH = signal.convolve2d(solver.modes[0].Ey, [[0.25, 0.25], [0.25, 0.25]])
EzatH = signal.convolve2d(solver.modes[0].Ez, [[0.25, 0.25], [0.25, 0.25]])
# Stokes parameters
q1 = ExatH * numpy.conjugate(solver.modes[0].Hy)
q2 = EyatH * numpy.conjugate(solver.modes[0].Hx)
q3 = EyatH * numpy.conjugate(solver.modes[0].Hy)
q4 = ExatH * numpy.conjugate(solver.modes[0].Hx)

S0 = q1.real - q2.real
S1 = q1.real + q2.real
S2 = q3.real - q4.real
S3 = q3.imag + q4.imag
denom = S0.sum()
print("ave S1=", S1.sum() / denom)
print("ave S2=", S2.sum() / denom)
print("ave S3=", S3.sum() / denom)

fmax = abs(S0).max()
fig = plt.figure()
ax = fig.add_subplot(2, 2, 1)
plt.contour(x, y, abs(S0.T), fmax * levls2, cmap="jet", locator=ticker.LogLocator())
plt.title("S0")
geom_outline()
ax = fig.add_subplot(2, 2, 2)
plt.contour(x, y, abs(S1.T), fmax * levls2, cmap="jet", locator=ticker.LogLocator())
plt.title("S1")
geom_outline()
ax = fig.add_subplot(2, 2, 3)
plt.contour(x, y, abs(S2.T), fmax * levls2, cmap="jet", locator=ticker.LogLocator())
plt.title("S2")
geom_outline()
ax = fig.add_subplot(2, 2, 4)
plt.contour(x, y, abs(S3.T), fmax * levls2, cmap="jet", locator=ticker.LogLocator())
plt.title("S3")
geom_outline()
plt.show()

print(solver.modes[1].neff)
fmax = abs(solver.modes[1].Ey).max()
fig = plt.figure()
ax = fig.add_subplot(2, 3, 1)
plt.contour(
    xe,
    ye,
    abs(solver.modes[1].Ex.T),
    fmax * levls,
    cmap="jet",
    locator=ticker.LogLocator(),
)
plt.title("Ex")
geom_outline()
ax = fig.add_subplot(2, 3, 2)
plt.contour(
    xe,
    ye,
    abs(solver.modes[1].Ey.T),
    fmax * levls,
    cmap="jet",
    locator=ticker.LogLocator(),
)
plt.title("Ey")
geom_outline()
ax = fig.add_subplot(2, 3, 3)
plt.contour(
    xe,
    ye,
    abs(solver.modes[1].Ez.T),
    fmax * levls,
    cmap="jet",
    locator=ticker.LogLocator(),
)
plt.title("Ez")
geom_outline()
fmax = abs(solver.modes[1].Hx).max()
ax = fig.add_subplot(2, 3, 4)
plt.contour(
    x,
    y,
    abs(solver.modes[1].Hx.T),
    fmax * levls,
    cmap="jet",
    locator=ticker.LogLocator(),
)
plt.title("Hx")
geom_outline()
ax = fig.add_subplot(2, 3, 5)
plt.contour(
    x,
    y,
    abs(solver.modes[1].Hy.T),
    fmax * levls,
    cmap="jet",
    locator=ticker.LogLocator(),
)
plt.title("Hy")
geom_outline()
ax = fig.add_subplot(2, 3, 6)
plt.contour(
    x,
    y,
    abs(solver.modes[1].Hz.T),
    fmax * levls,
    cmap="jet",
    locator=ticker.LogLocator(),
)
plt.title("Hz")
geom_outline()
plt.show()

ExatH = signal.convolve2d(solver.modes[1].Ex, [[0.25, 0.25], [0.25, 0.25]])
EyatH = signal.convolve2d(solver.modes[1].Ey, [[0.25, 0.25], [0.25, 0.25]])
EzatH = signal.convolve2d(solver.modes[1].Ez, [[0.25, 0.25], [0.25, 0.25]])
# Stokes parameters
q1 = ExatH * numpy.conjugate(solver.modes[1].Hy)
q2 = EyatH * numpy.conjugate(solver.modes[1].Hx)
q3 = EyatH * numpy.conjugate(solver.modes[1].Hy)
q4 = ExatH * numpy.conjugate(solver.modes[1].Hx)

S0 = q1.real - q2.real
S1 = q1.real + q2.real
S2 = q3.real - q4.real
S3 = q3.imag + q4.imag
denom = S0.sum()
print("ave S1=", S1.sum() / denom)
print("ave S2=", S2.sum() / denom)
print("ave S3=", S3.sum() / denom)

fmax = abs(S0).max()
fig = plt.figure()
ax = fig.add_subplot(2, 2, 1)
plt.contour(x, y, abs(S0.T), fmax * levls2, cmap="jet", locator=ticker.LogLocator())
plt.title("S0")
geom_outline()
ax = fig.add_subplot(2, 2, 2)
plt.contour(x, y, abs(S1.T), fmax * levls2, cmap="jet", locator=ticker.LogLocator())
plt.title("S1")
geom_outline()
ax = fig.add_subplot(2, 2, 3)
plt.contour(x, y, abs(S2.T), fmax * levls2, cmap="jet", locator=ticker.LogLocator())
plt.title("S2")
geom_outline()
ax = fig.add_subplot(2, 2, 4)
plt.contour(x, y, abs(S3.T), fmax * levls2, cmap="jet", locator=ticker.LogLocator())
plt.title("S3")
geom_outline()
plt.show()
