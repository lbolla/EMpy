import numpy
import EMpy
import pylab

def epsfunc(x, y):
    [X, Y] = numpy.meshgrid(x, y)
    return numpy.where((numpy.abs(X.T - 1.24e-6) <= .24e-6) *
                       (numpy.abs(Y.T - 1.11e-6) <= .11e-6),
                       3.4757**2,
                       1.446**2)
    
wl = 1.55e-6
x = numpy.linspace(0, 2.48e-6, 125)
y = numpy.linspace(0, 2.22e-6, 112)

neigs = 2
tol = 1e-8
boundary = '0000' 

solver = EMpy.modesolvers.FD.VFDModeSolver(wl, x, y, epsfunc, boundary).solve(neigs, tol)
