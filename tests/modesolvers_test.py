from unittest import TestCase

import numpy

import EMpy.modesolvers


class VFDModeSolverTest(TestCase):

    def test_solve(self):

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

        neigs = 2
        tol = 1e-8
        boundary = "0000"
        wl = 1.55e-6
        x = numpy.linspace(0, 2.48e-6, 125)
        y = numpy.linspace(0, 2.22e-6, 112)
        solver = EMpy.modesolvers.FD.VFDModeSolver(wl, x, y, epsfunc, boundary).solve(
            neigs, tol
        )
        self.assertEqual(len(solver.modes), 2)
        self.assertAlmostEqual(solver.modes[0].neff, 2.4179643622942937)