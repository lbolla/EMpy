"""Fully vectorial finite-difference mode solver example.

Flexible mode solver example for fundamental TE & TM modes
by David Hutchings, School of Engineering, University of Glasgow
David.Hutchings@glasgow.ac.uk
"""

import numpy
import EMpy 
import matplotlib.pyplot as plt
from matplotlib import ticker

"""Define cross-section geometry of simulation
each rectangular element tuple contains
("label",refractive index,xmin,ymin,xmax,ymax)
first element tuple defines simulation extent
units should match units of wavelength wl
"""
geom = (("base",1.0,-0.8,-0.8,0.8,1.0),
        ("substrate",1.5,-0.8,-0.8,0.8,0.0),
        ("core",3.4,-0.30,0.0,0.30,0.34))

wl = 1.55
nx, ny = 81, 91

def epsfunc(x_, y_):
    """Return a matrix describing a 2d material.

    :param x_: x values
    :param y_: y values
    :return: 2d-matrix
    """
    #xx, yy = numpy.meshgrid(x_, y_)
    working = geom[0][1]**2*numpy.ones((x_.size,y_.size))
    for i in range(1,len(geom)):
        ixmin = numpy.searchsorted(x_,geom[i][2],side='left')
        iymin = numpy.searchsorted(y_,geom[i][3],side='left')
        ixmax = numpy.searchsorted(x_,geom[i][4],side='right')
        iymax = numpy.searchsorted(y_,geom[i][5],side='right') 
        working[ixmin:ixmax,iymin:iymax] = geom[i][1]**2
    
    return working

x = numpy.linspace(geom[0][2], geom[0][4], nx)
y = numpy.linspace(geom[0][3], geom[0][5], ny)

neigs = 2
tol = 1e-6
boundary = '0000'

solver = EMpy.modesolvers.FD.VFDModeSolver(wl, x, y, epsfunc, boundary).solve(
    neigs, tol)

levls = numpy.geomspace(1./32.,1.,num=11)
xe = x[:-1]+0.5*(x[1]-x[0])
ye = y[:-1]+0.5*(y[1]-y[0])

print(solver.modes[0].neff)
fmax=abs(solver.modes[0].Ex).max()
fig = plt.figure()
ax = fig.add_subplot(2, 3, 1)
plt.contour(xe,ye,abs(solver.modes[0].Ex.T), fmax*levls, 
            cmap='jet', locator=ticker.LogLocator())
plt.title('Ex')
ax.set_xlim(geom[0][2], geom[0][4])
ax.set_ylim(geom[0][3], geom[0][5])
for i in range(1,len(geom)):
    plt.hlines(geom[i][3],geom[i][2],geom[i][4])
    plt.hlines(geom[i][5],geom[i][2],geom[i][4])
    plt.vlines(geom[i][2],geom[i][3],geom[i][5])
    plt.vlines(geom[i][4],geom[i][3],geom[i][5])
ax = fig.add_subplot(2, 3, 2)
plt.contour(xe,ye,abs(solver.modes[0].Ey.T), fmax*levls, 
            cmap='jet', locator=ticker.LogLocator())
plt.title('Ey')
ax.set_xlim(geom[0][2], geom[0][4])
ax.set_ylim(geom[0][3], geom[0][5])
for i in range(1,len(geom)):
    plt.hlines(geom[i][3],geom[i][2],geom[i][4])
    plt.hlines(geom[i][5],geom[i][2],geom[i][4])
    plt.vlines(geom[i][2],geom[i][3],geom[i][5])
    plt.vlines(geom[i][4],geom[i][3],geom[i][5])
ax = fig.add_subplot(2, 3, 3)
plt.contour(xe,ye,abs(solver.modes[0].Ez.T), fmax*levls, 
            cmap='jet', locator=ticker.LogLocator())
plt.title('Ez')
ax.set_xlim(geom[0][2], geom[0][4])
ax.set_ylim(geom[0][3], geom[0][5])
for i in range(1,len(geom)):
    plt.hlines(geom[i][3],geom[i][2],geom[i][4])
    plt.hlines(geom[i][5],geom[i][2],geom[i][4])
    plt.vlines(geom[i][2],geom[i][3],geom[i][5])
    plt.vlines(geom[i][4],geom[i][3],geom[i][5])
fmax=abs(solver.modes[0].Hy).max()
ax = fig.add_subplot(2, 3, 4)
plt.contour(x,y,abs(solver.modes[0].Hx.T), fmax*levls, 
            cmap='jet', locator=ticker.LogLocator())
plt.title('Hx')
ax.set_xlim(geom[0][2], geom[0][4])
ax.set_ylim(geom[0][3], geom[0][5])
for i in range(1,len(geom)):
    plt.hlines(geom[i][3],geom[i][2],geom[i][4])
    plt.hlines(geom[i][5],geom[i][2],geom[i][4])
    plt.vlines(geom[i][2],geom[i][3],geom[i][5])
    plt.vlines(geom[i][4],geom[i][3],geom[i][5])
ax = fig.add_subplot(2, 3, 5)
plt.contour(x,y,abs(solver.modes[0].Hy.T), fmax*levls, 
            cmap='jet', locator=ticker.LogLocator())
plt.title('Hy')
ax.set_xlim(geom[0][2], geom[0][4])
ax.set_ylim(geom[0][3], geom[0][5])
for i in range(1,len(geom)):
    plt.hlines(geom[i][3],geom[i][2],geom[i][4])
    plt.hlines(geom[i][5],geom[i][2],geom[i][4])
    plt.vlines(geom[i][2],geom[i][3],geom[i][5])
    plt.vlines(geom[i][4],geom[i][3],geom[i][5])
ax = fig.add_subplot(2, 3, 6)
plt.contour(x,y,abs(solver.modes[0].Hz.T), fmax*levls, 
            cmap='jet', locator=ticker.LogLocator())
plt.title('Hz')
ax.set_xlim(geom[0][2], geom[0][4])
ax.set_ylim(geom[0][3], geom[0][5])
for i in range(1,len(geom)):
    plt.hlines(geom[i][3],geom[i][2],geom[i][4])
    plt.hlines(geom[i][5],geom[i][2],geom[i][4])
    plt.vlines(geom[i][2],geom[i][3],geom[i][5])
    plt.vlines(geom[i][4],geom[i][3],geom[i][5])
plt.show()

print(solver.modes[1].neff)
fmax=abs(solver.modes[1].Ey).max()
fig = plt.figure()
ax = fig.add_subplot(2, 3, 1)
plt.contour(xe,ye,abs(solver.modes[1].Ex.T), fmax*levls, 
            cmap='jet', locator=ticker.LogLocator())
plt.title('Ex')
ax.set_xlim(geom[0][2], geom[0][4])
ax.set_ylim(geom[0][3], geom[0][5])
for i in range(1,len(geom)):
    plt.hlines(geom[i][3],geom[i][2],geom[i][4])
    plt.hlines(geom[i][5],geom[i][2],geom[i][4])
    plt.vlines(geom[i][2],geom[i][3],geom[i][5])
    plt.vlines(geom[i][4],geom[i][3],geom[i][5])
ax = fig.add_subplot(2, 3, 2)
plt.contour(xe,ye,abs(solver.modes[1].Ey.T), fmax*levls, 
            cmap='jet', locator=ticker.LogLocator())
plt.title('Ey')
ax.set_xlim(geom[0][2], geom[0][4])
ax.set_ylim(geom[0][3], geom[0][5])
for i in range(1,len(geom)):
    plt.hlines(geom[i][3],geom[i][2],geom[i][4])
    plt.hlines(geom[i][5],geom[i][2],geom[i][4])
    plt.vlines(geom[i][2],geom[i][3],geom[i][5])
    plt.vlines(geom[i][4],geom[i][3],geom[i][5])
ax = fig.add_subplot(2, 3, 3)
plt.contour(xe,ye,abs(solver.modes[1].Ez.T), fmax*levls, 
            cmap='jet', locator=ticker.LogLocator())
plt.title('Ez')
ax.set_xlim(geom[0][2], geom[0][4])
ax.set_ylim(geom[0][3], geom[0][5])
for i in range(1,len(geom)):
    plt.hlines(geom[i][3],geom[i][2],geom[i][4])
    plt.hlines(geom[i][5],geom[i][2],geom[i][4])
    plt.vlines(geom[i][2],geom[i][3],geom[i][5])
    plt.vlines(geom[i][4],geom[i][3],geom[i][5])
fmax=abs(solver.modes[1].Hx).max()
ax = fig.add_subplot(2, 3, 4)
plt.contour(x,y,abs(solver.modes[1].Hx.T), fmax*levls, 
            cmap='jet', locator=ticker.LogLocator())
plt.title('Hx')
ax.set_xlim(geom[0][2], geom[0][4])
ax.set_ylim(geom[0][3], geom[0][5])
for i in range(1,len(geom)):
    plt.hlines(geom[i][3],geom[i][2],geom[i][4])
    plt.hlines(geom[i][5],geom[i][2],geom[i][4])
    plt.vlines(geom[i][2],geom[i][3],geom[i][5])
    plt.vlines(geom[i][4],geom[i][3],geom[i][5])
ax = fig.add_subplot(2, 3, 5)
plt.contour(x,y,abs(solver.modes[1].Hy.T), fmax*levls, 
            cmap='jet', locator=ticker.LogLocator())
plt.title('Hy')
ax.set_xlim(geom[0][2], geom[0][4])
ax.set_ylim(geom[0][3], geom[0][5])
for i in range(1,len(geom)):
    plt.hlines(geom[i][3],geom[i][2],geom[i][4])
    plt.hlines(geom[i][5],geom[i][2],geom[i][4])
    plt.vlines(geom[i][2],geom[i][3],geom[i][5])
    plt.vlines(geom[i][4],geom[i][3],geom[i][5])
ax = fig.add_subplot(2, 3, 6)
plt.contour(x,y,abs(solver.modes[1].Hz.T), fmax*levls, 
            cmap='jet', locator=ticker.LogLocator())
plt.title('Hz')
ax.set_xlim(geom[0][2], geom[0][4])
ax.set_ylim(geom[0][3], geom[0][5])
for i in range(1,len(geom)):
    plt.hlines(geom[i][3],geom[i][2],geom[i][4])
    plt.hlines(geom[i][5],geom[i][2],geom[i][4])
    plt.vlines(geom[i][2],geom[i][3],geom[i][5])
    plt.vlines(geom[i][4],geom[i][3],geom[i][5])
plt.show()
