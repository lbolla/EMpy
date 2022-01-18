# pylint: disable=line-too-long,too-many-locals,too-many-statements,too-many-branches
# pylint: disable=redefined-builtin,wildcard-import,unused-wildcard-import
# pylint: disable=attribute-defined-outside-init,too-many-instance-attributes
# pylint: disable=arguments-differ,too-many-arguments
"""Finite Difference Modesolver.

@see: Fallahkhair, "Vector Finite Difference Modesolver for Anisotropic Dielectric Waveguides",
@see: JLT 2007 <http://www.photonics.umd.edu/wp-content/uploads/pubs/ja-20/Fallahkhair_JLT_26_1423_2008.pdf>}
@see: DOI of above reference <http://doi.org/10.1109/JLT.2008.923643>
@see: http://www.mathworks.com/matlabcentral/fileexchange/loadFile.do?objectId=12734&objectType=FILE

"""
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import range

import numpy
import scipy
import scipy.optimize
import EMpy_gpu.utils
from EMpy_gpu.modesolvers.interface import *


class SVFDModeSolver(ModeSolver):

    """
    This function calculates the modes of a dielectric waveguide
    using the semivectorial finite difference method.
    It is slightly faster than the full-vectorial VFDModeSolver,
    but it does not accept non-isotropic permittivity. For example,
    birefringent materials, which have
    different refractive indices along different dimensions cannot be used.
    It is adapted from the "svmodes.m" matlab code of Thomas Murphy and co-workers.
    https://www.mathworks.com/matlabcentral/fileexchange/12734-waveguide-mode-solver/content/svmodes.m

    Parameters
    ----------
    wl : float
        optical wavelength
        units are arbitrary, but must be self-consistent.
        I.e., just use micron for everything.
    x : 1D array of floats
        Array of x-values
    y : 1D array of floats
        Array of y-values
    epsfunc : function
        This is a function that provides the relative permittivity matrix
        (square of the refractive index) as a function of its x and y
        numpy.arrays (the function's input parameters). The function must be
        of the form: ``myRelativePermittivity(x,y)``, where x and y are 2D
        numpy "meshgrid" arrays that will be passed by this function.
        The function returns a relative permittivity numpy.array of
        shape( x.shape[0], y.shape[0] ) where each element of the array
        is a single float, corresponding the an isotropic refractive index.
        If an anisotropic refractive index is desired, the full-vectorial
        VFDModeSolver function should be used.
    boundary : str
        This is a string that identifies the type of boundary conditions applied.
        The following options are available:
           'A' - Hx is antisymmetric, Hy is symmetric.
           'S' - Hx is symmetric and, Hy is antisymmetric.
           '0' - Hx and Hy are zero immediately outside of the boundary.
        The string identifies all four boundary conditions, in the order:
            North, south, east, west.
        For example, boundary='000A'

    method : str
        must be 'Ex', 'Ey', or 'scalar'
        this identifies the field that will be calculated.


    Returns
    -------
    self : an instance of the SVFDModeSolver class
        Typically self.solve() will be called in order to actually find the modes.

    """

    def __init__(self, wl, x, y, epsfunc, boundary, method='Ex'):
        self.wl = wl
        self.x = x
        self.y = y
        self.epsfunc = epsfunc
        self.boundary = boundary
        self.method = method

    def _get_eps(self, xc, yc):
        eps = self.epsfunc(xc, yc)
        eps = numpy.c_[eps[:, 0:1], eps, eps[:, -1:]]
        eps = numpy.r_[eps[0:1, :], eps, eps[-1:, :]]
        return eps

    def build_matrix(self):

        from scipy.sparse import coo_matrix

        wl = self.wl
        x = self.x
        y = self.y
        boundary = self.boundary
        method = self.method

        dx = numpy.diff(x)
        dy = numpy.diff(y)

        dx = numpy.r_[dx[0], dx, dx[-1]].reshape(-1, 1)
        dy = numpy.r_[dy[0], dy, dy[-1]].reshape(1, -1)

        xc = (x[:-1] + x[1:]) / 2
        yc = (y[:-1] + y[1:]) / 2
        eps = self._get_eps(xc, yc)

        nx = len(xc)
        ny = len(yc)

        self.nx = nx
        self.ny = ny

        k = 2 * numpy.pi / wl

        ones_nx = numpy.ones((nx, 1))
        ones_ny = numpy.ones((1, ny))

        n = numpy.dot(ones_nx, 0.5 * (dy[:, 2:] + dy[:, 1:-1])).flatten()
        s = numpy.dot(ones_nx, 0.5 * (dy[:, 0:-2] + dy[:, 1:-1])).flatten()
        e = numpy.dot(0.5 * (dx[2:, :] + dx[1:-1, :]), ones_ny).flatten()
        w = numpy.dot(0.5 * (dx[0:-2, :] + dx[1:-1, :]), ones_ny).flatten()
        p = numpy.dot(dx[1:-1, :], ones_ny).flatten()
        q = numpy.dot(ones_nx, dy[:, 1:-1]).flatten()

        en = eps[1:-1, 2:].flatten()
        es = eps[1:-1, 0:-2].flatten()
        ee = eps[2:, 1:-1].flatten()
        ew = eps[0:-2, 1:-1].flatten()
        ep = eps[1:-1, 1:-1].flatten()

        # three methods: Ex, Ey and scalar

        if method == 'Ex':

            # Ex

            An = 2 / n / (n + s)
            As = 2 / s / (n + s)
            Ae = 8 * (p * (ep - ew) + 2 * w * ew) * ee / \
                ((p * (ep - ee) + 2 * e * ee) * (p ** 2 * (ep - ew) + 4 * w ** 2 * ew) +
                 (p * (ep - ew) + 2 * w * ew) * (p ** 2 * (ep - ee) + 4 * e ** 2 * ee))
            Aw = 8 * (p * (ep - ee) + 2 * e * ee) * ew / \
                ((p * (ep - ee) + 2 * e * ee) * (p ** 2 * (ep - ew) + 4 * w ** 2 * ew) +
                 (p * (ep - ew) + 2 * w * ew) * (p ** 2 * (ep - ee) + 4 * e ** 2 * ee))
            Ap = ep * k ** 2 - An - As - Ae * ep / ee - Aw * ep / ew

        elif method == 'Ey':

            # Ey

            An = 8 * (q * (ep - es) + 2 * s * es) * en / \
                ((q * (ep - en) + 2 * n * en) * (q ** 2 * (ep - es) + 4 * s ** 2 * es) +
                 (q * (ep - es) + 2 * s * es) * (q ** 2 * (ep - en) + 4 * n ** 2 * en))
            As = 8 * (q * (ep - en) + 2 * n * en) * es / \
                ((q * (ep - en) + 2 * n * en) * (q ** 2 * (ep - es) + 4 * s ** 2 * es) +
                 (q * (ep - es) + 2 * s * es) * (q ** 2 * (ep - en) + 4 * n ** 2 * en))
            Ae = 2 / e / (e + w)
            Aw = 2 / w / (e + w)
            Ap = ep * k ** 2 - An * ep / en - As * ep / es - Ae - Aw

        elif method == 'scalar':

            # scalar

            An = 2 / n / (n + s)
            As = 2 / s / (n + s)
            Ae = 2 / e / (e + w)
            Aw = 2 / w / (e + w)
            Ap = ep * k ** 2 - An - As - Ae - Aw

        else:

            raise ValueError('unknown method')

        ii = numpy.arange(nx * ny).reshape(nx, ny)

        # north boundary
        ib = ii[:, -1]
        if boundary[0] == 'S':
            Ap[ib] += An[ib]
        elif boundary[0] == 'A':
            Ap[ib] -= An[ib]
        # else:
        #     raise ValueError('unknown boundary')

        # south
        ib = ii[:, 0]
        if boundary[1] == 'S':
            Ap[ib] += As[ib]
        elif boundary[1] == 'A':
            Ap[ib] -= As[ib]
        # else:
        #     raise ValueError('unknown boundary')

        # east
        ib = ii[-1, :]
        if boundary[2] == 'S':
            Ap[ib] += Ae[ib]
        elif boundary[2] == 'A':
            Ap[ib] -= Ae[ib]
        # else:
        #     raise ValueError('unknown boundary')

        # west
        ib = ii[0, :]
        if boundary[3] == 'S':
            Ap[ib] += Aw[ib]
        elif boundary[3] == 'A':
            Ap[ib] -= Aw[ib]
        # else:
        #     raise ValueError('unknown boundary')

        iall = ii.flatten()
        i_n = ii[:, 1:].flatten()
        i_s = ii[:, :-1].flatten()
        i_e = ii[1:, :].flatten()
        i_w = ii[:-1, :].flatten()

        I = numpy.r_[iall, i_w, i_e, i_s, i_n]
        J = numpy.r_[iall, i_e, i_w, i_n, i_s]
        V = numpy.r_[Ap[iall], Ae[i_w], Aw[i_e], An[i_s], As[i_n]]

        A = coo_matrix((V, (I, J))).tocsr()

        return A

    def solve(self, neigs, tol):

        from scipy.sparse.linalg import eigen

        self.nmodes = neigs
        self.tol = tol

        A = self.build_matrix()

        [eigvals, eigvecs] = eigen.eigs(A,
                                        k=neigs,
                                        which='LR',
                                        tol=tol,
                                        ncv=10 * neigs,
                                        return_eigenvectors=True)

        neff = self.wl * scipy.sqrt(eigvals) / (2 * numpy.pi)
        phi = []
        for ieig in range(neigs):
            tmp = eigvecs[:, ieig].reshape(self.nx, self.ny)
            phi.append(tmp)

        # sort and save the modes
        idx = numpy.flipud(numpy.argsort(neff))
        self.neff = neff[idx]
        tmp = []
        for i in idx:
            tmp.append(phi[i])

        if self.method == 'scalar':
            self.phi = tmp
        elif self.method == 'Ex':
            self.Ex = tmp
        if self.method == 'Ey':
            self.Ey = tmp

        return self

    def __str__(self):
        descr = (
            'Semi-Vectorial Finite Difference Modesolver\n\tmethod: %s\n' %
            self.method)
        return descr


class VFDModeSolver(ModeSolver):

    """
    The VFDModeSolver class computes the electric and magnetic fields
    for modes of a dielectric waveguide using the "Vector Finite
    Difference (VFD)" method, as described in A. B. Fallahkhair,
    K. S. Li and T. E. Murphy, "Vector Finite Difference Modesolver
    for Anisotropic Dielectric Waveguides", J. Lightwave
    Technol. 26(11), 1423-1431, (2008).


    Parameters
    ----------
    wl : float
        The wavelength of the optical radiation (units are arbitrary,
        but must be self-consistent between all inputs. It is recommended to
        just use microns for everthing)
    x : 1D array of floats
        Array of x-values
    y : 1D array of floats
        Array of y-values
    epsfunc : function
        This is a function that provides the relative permittivity
        matrix (square of the refractive index) as a function of its x
        and y numpy.arrays (the function's input parameters). The
        function must be of the form: ``myRelativePermittivity(x,y)``
        The function returns a relative permittivity numpy.array of either
        shape( x.shape[0], y.shape[0] ) where each element of the
        array can either be a single float, corresponding the an
        isotropic refractive index, or (x.shape[0], y.shape[0], 5),
        where the last dimension describes the relative permittivity in
        the form (epsxx, epsxy, epsyx, epsyy, epszz).
    boundary : str
        This is a string that identifies the type of boundary
        conditions applied.
        The following options are available:
           'A' - Hx is antisymmetric, Hy is symmetric.
           'S' - Hx is symmetric and, Hy is antisymmetric.
           '0' - Hx and Hy are zero immediately outside of the boundary.
        The string identifies all four boundary conditions, in the
        order: North, south, east, west.  For example, boundary='000A'

    Returns
    -------
    self : an instance of the VFDModeSolver class
        Typically self.solve() will be called in order to actually
        find the modes.

    """

    def __init__(self, wl, x, y, epsfunc, boundary):
        self.wl = wl
        self.x = x
        self.y = y
        self.epsfunc = epsfunc
        self.boundary = boundary

    def _get_eps(self, xc, yc):
        tmp = self.epsfunc(xc, yc)

        def _reshape(tmp):
            """
            pads the array by duplicating edge values
            """
            tmp = numpy.c_[tmp[:, 0:1], tmp, tmp[:, -1:]]
            tmp = numpy.r_[tmp[0:1, :], tmp, tmp[-1:, :]]
            return tmp

        if tmp.ndim == 2: # isotropic refractive index
            tmp = _reshape(tmp)
            epsxx = epsyy = epszz = tmp
            epsxy = epsyx = numpy.zeros_like(epsxx)

        elif tmp.ndim == 3: # anisotropic refractive index
            assert tmp.shape[2] == 5, 'eps must be NxMx5'
            epsxx = _reshape(tmp[:, :, 0])
            epsxy = _reshape(tmp[:, :, 1])
            epsyx = _reshape(tmp[:, :, 2])
            epsyy = _reshape(tmp[:, :, 3])
            epszz = _reshape(tmp[:, :, 4])

        else:
            raise ValueError('Invalid eps')

        return epsxx, epsxy, epsyx, epsyy, epszz

    def build_matrix(self):

        from scipy.sparse import coo_matrix

        wl = self.wl
        x = self.x
        y = self.y
        boundary = self.boundary

        dx = numpy.diff(x)
        dy = numpy.diff(y)

        dx = numpy.r_[dx[0], dx, dx[-1]].reshape(-1, 1)
        dy = numpy.r_[dy[0], dy, dy[-1]].reshape(1, -1)

        # Note: the permittivity is actually defined at the center of each
        # region *between* the mesh points used for the H-field calculation.
        # (See Fig. 1 of Fallahkhair and Murphy)
        # In other words, eps is defined on (xc,yc) which is offset from
        # (x,y), the grid where H is calculated, by
        # "half a pixel" in the positive-x and positive-y directions.
        xc = (x[:-1] + x[1:]) / 2
        yc = (y[:-1] + y[1:]) / 2
        epsxx, epsxy, epsyx, epsyy, epszz = self._get_eps(xc, yc)

        nx = len(x)
        ny = len(y)

        self.nx = nx
        self.ny = ny

        k = 2 * numpy.pi / wl

        ones_nx = numpy.ones((nx, 1))
        ones_ny = numpy.ones((1, ny))

        # distance of mesh points to nearest neighbor mesh point:
        n = numpy.dot(ones_nx, dy[:, 1:]).flatten()
        s = numpy.dot(ones_nx, dy[:, :-1]).flatten()
        e = numpy.dot(dx[1:, :], ones_ny).flatten()
        w = numpy.dot(dx[:-1, :], ones_ny).flatten()

        # These define the permittivity (eps) tensor relative to each mesh point
        # using the following geometry:
        #
        #                 NW------N------NE
        #                 |       |       |
        #                 |   1   n   4   |
        #                 |       |       |
        #                 W---w---P---e---E
        #                 |       |       |
        #                 |   2   s   3   |
        #                 |       |       |
        #                 SW------S------SE

        exx1 = epsxx[:-1, 1:].flatten()
        exx2 = epsxx[:-1, :-1].flatten()
        exx3 = epsxx[1:, :-1].flatten()
        exx4 = epsxx[1:, 1:].flatten()

        eyy1 = epsyy[:-1, 1:].flatten()
        eyy2 = epsyy[:-1, :-1].flatten()
        eyy3 = epsyy[1:, :-1].flatten()
        eyy4 = epsyy[1:, 1:].flatten()

        exy1 = epsxy[:-1, 1:].flatten()
        exy2 = epsxy[:-1, :-1].flatten()
        exy3 = epsxy[1:, :-1].flatten()
        exy4 = epsxy[1:, 1:].flatten()

        eyx1 = epsyx[:-1, 1:].flatten()
        eyx2 = epsyx[:-1, :-1].flatten()
        eyx3 = epsyx[1:, :-1].flatten()
        eyx4 = epsyx[1:, 1:].flatten()

        ezz1 = epszz[:-1, 1:].flatten()
        ezz2 = epszz[:-1, :-1].flatten()
        ezz3 = epszz[1:, :-1].flatten()
        ezz4 = epszz[1:, 1:].flatten()

        ns21 = n * eyy2 + s * eyy1
        ns34 = n * eyy3 + s * eyy4
        ew14 = e * exx1 + w * exx4
        ew23 = e * exx2 + w * exx3

        # calculate the finite difference coefficients following
        # Fallahkhair and Murphy, Appendix Eqs 21 though 37

        axxn = ((2 * eyy4 * e - eyx4 * n) * (eyy3 / ezz4) / ns34 +
                (2 * eyy1 * w + eyx1 * n) * (eyy2 / ezz1) / ns21) / (n * (e + w))
        axxs = ((2 * eyy3 * e + eyx3 * s) * (eyy4 / ezz3) / ns34 +
                (2 * eyy2 * w - eyx2 * s) * (eyy1 / ezz2) / ns21) / (s * (e + w))
        ayye = (2 * n * exx4 - e * exy4) * exx1 / ezz4 / e / ew14 / \
            (n + s) + (2 * s * exx3 + e * exy3) * \
            exx2 / ezz3 / e / ew23 / (n + s)
        ayyw = (2 * exx1 * n + exy1 * w) * exx4 / ezz1 / w / ew14 / \
            (n + s) + (2 * exx2 * s - exy2 * w) * \
            exx3 / ezz2 / w / ew23 / (n + s)
        axxe = 2 / (e * (e + w)) + \
            (eyy4 * eyx3 / ezz3 - eyy3 * eyx4 / ezz4) / (e + w) / ns34
        axxw = 2 / (w * (e + w)) + \
            (eyy2 * eyx1 / ezz1 - eyy1 * eyx2 / ezz2) / (e + w) / ns21
        ayyn = 2 / (n * (n + s)) + \
            (exx4 * exy1 / ezz1 - exx1 * exy4 / ezz4) / (n + s) / ew14
        ayys = 2 / (s * (n + s)) + \
            (exx2 * exy3 / ezz3 - exx3 * exy2 / ezz2) / (n + s) / ew23

        axxne = +eyx4 * eyy3 / ezz4 / (e + w) / ns34
        axxse = -eyx3 * eyy4 / ezz3 / (e + w) / ns34
        axxnw = -eyx1 * eyy2 / ezz1 / (e + w) / ns21
        axxsw = +eyx2 * eyy1 / ezz2 / (e + w) / ns21

        ayyne = +exy4 * exx1 / ezz4 / (n + s) / ew14
        ayyse = -exy3 * exx2 / ezz3 / (n + s) / ew23
        ayynw = -exy1 * exx4 / ezz1 / (n + s) / ew14
        ayysw = +exy2 * exx3 / ezz2 / (n + s) / ew23

        axxp = -axxn - axxs - axxe - axxw - axxne - axxse - axxnw - axxsw + k ** 2 * \
            (n + s) * \
            (eyy4 * eyy3 * e / ns34 + eyy1 * eyy2 * w / ns21) / (e + w)
        ayyp = -ayyn - ayys - ayye - ayyw - ayyne - ayyse - ayynw - ayysw + k ** 2 * \
            (e + w) * \
            (exx1 * exx4 * n / ew14 + exx2 * exx3 * s / ew23) / (n + s)
        axyn = (eyy3 * eyy4 / ezz4 / ns34 - eyy2 * eyy1 / ezz1 /
                ns21 + s * (eyy2 * eyy4 - eyy1 * eyy3) / ns21 / ns34) / (e + w)
        axys = (eyy1 * eyy2 / ezz2 / ns21 - eyy4 * eyy3 / ezz3 /
                ns34 + n * (eyy2 * eyy4 - eyy1 * eyy3) / ns21 / ns34) / (e + w)
        ayxe = (exx1 * exx4 / ezz4 / ew14 - exx2 * exx3 / ezz3 /
                ew23 + w * (exx2 * exx4 - exx1 * exx3) / ew23 / ew14) / (n + s)
        ayxw = (exx3 * exx2 / ezz2 / ew23 - exx4 * exx1 / ezz1 /
                ew14 + e * (exx4 * exx2 - exx1 * exx3) / ew23 / ew14) / (n + s)

        axye = (eyy4 * (1 + eyy3 / ezz4) - eyy3 * (1 + eyy4 / ezz4)) / ns34 / (e + w) - \
               (2 * eyx1 * eyy2 / ezz1 * n * w / ns21 +
                2 * eyx2 * eyy1 / ezz2 * s * w / ns21 +
                2 * eyx4 * eyy3 / ezz4 * n * e / ns34 +
                2 * eyx3 * eyy4 / ezz3 * s * e / ns34 +
                2 * eyy1 * eyy2 * (1. / ezz1 - 1. / ezz2) * w ** 2 / ns21) / e / (e + w) ** 2

        axyw = (eyy2 * (1 + eyy1 / ezz2) - eyy1 * (1 + eyy2 / ezz2)) / ns21 / (e + w) - \
               (2 * eyx1 * eyy2 / ezz1 * n * e / ns21 +
                2 * eyx2 * eyy1 / ezz2 * s * e / ns21 +
                2 * eyx4 * eyy3 / ezz4 * n * w / ns34 +
                2 * eyx3 * eyy4 / ezz3 * s * w / ns34 +
                2 * eyy3 * eyy4 * (1. / ezz3 - 1. / ezz4) * e ** 2 / ns34) / w / (e + w) ** 2

        ayxn = (exx4 * (1 + exx1 / ezz4) - exx1 * (1 + exx4 / ezz4)) / ew14 / (n + s) - \
               (2 * exy3 * exx2 / ezz3 * e * s / ew23 +
                2 * exy2 * exx3 / ezz2 * w * n / ew23 +
                2 * exy4 * exx1 / ezz4 * e * s / ew14 +
                2 * exy1 * exx4 / ezz1 * w * n / ew14 +
                2 * exx3 * exx2 * (1. / ezz3 - 1. / ezz2) * s ** 2 / ew23) / n / (n + s) ** 2

        ayxs = (exx2 * (1 + exx3 / ezz2) - exx3 * (1 + exx2 / ezz2)) / ew23 / (n + s) - \
               (2 * exy3 * exx2 / ezz3 * e * n / ew23 +
                2 * exy2 * exx3 / ezz2 * w * n / ew23 +
                2 * exy4 * exx1 / ezz4 * e * s / ew14 +
                2 * exy1 * exx4 / ezz1 * w * s / ew14 +
                2 * exx1 * exx4 * (1. / ezz1 - 1. / ezz4) * n ** 2 / ew14) / s / (n + s) ** 2

        axyne = +eyy3 * (1 - eyy4 / ezz4) / (e + w) / ns34
        axyse = -eyy4 * (1 - eyy3 / ezz3) / (e + w) / ns34
        axynw = -eyy2 * (1 - eyy1 / ezz1) / (e + w) / ns21
        axysw = +eyy1 * (1 - eyy2 / ezz2) / (e + w) / ns21

        ayxne = +exx1 * (1 - exx4 / ezz4) / (n + s) / ew14
        ayxse = -exx2 * (1 - exx3 / ezz3) / (n + s) / ew23
        ayxnw = -exx4 * (1 - exx1 / ezz1) / (n + s) / ew14
        ayxsw = +exx3 * (1 - exx2 / ezz2) / (n + s) / ew23

        axyp = -(axyn + axys + axye + axyw + axyne + axyse + axynw + axysw) - k ** 2 * (w * (n * eyx1 *
                                                                                             eyy2 + s * eyx2 * eyy1) / ns21 + e * (s * eyx3 * eyy4 + n * eyx4 * eyy3) / ns34) / (e + w)
        ayxp = -(ayxn + ayxs + ayxe + ayxw + ayxne + ayxse + ayxnw + ayxsw) - k ** 2 * (n * (w * exy1 *
                                                                                             exx4 + e * exy4 * exx1) / ew14 + s * (w * exy2 * exx3 + e * exy3 * exx2) / ew23) / (n + s)

        ii = numpy.arange(nx * ny).reshape(nx, ny)

        # NORTH boundary

        ib = ii[:, -1]

        if boundary[0] == 'S':
            sign = 1
        elif boundary[0] == 'A':
            sign = -1
        elif boundary[0] == '0':
            sign = 0
        else:
            raise ValueError('unknown boundary conditions')

        axxs[ib]  += sign * axxn[ib]
        axxse[ib] += sign * axxne[ib]
        axxsw[ib] += sign * axxnw[ib]
        ayxs[ib]  += sign * ayxn[ib]
        ayxse[ib] += sign * ayxne[ib]
        ayxsw[ib] += sign * ayxnw[ib]
        ayys[ib]  -= sign * ayyn[ib]
        ayyse[ib] -= sign * ayyne[ib]
        ayysw[ib] -= sign * ayynw[ib]
        axys[ib]  -= sign * axyn[ib]
        axyse[ib] -= sign * axyne[ib]
        axysw[ib] -= sign * axynw[ib]

        # SOUTH boundary

        ib = ii[:, 0]

        if boundary[1] == 'S':
            sign = 1
        elif boundary[1] == 'A':
            sign = -1
        elif boundary[1] == '0':
            sign = 0
        else:
            raise ValueError('unknown boundary conditions')

        axxn[ib]  += sign * axxs[ib]
        axxne[ib] += sign * axxse[ib]
        axxnw[ib] += sign * axxsw[ib]
        ayxn[ib]  += sign * ayxs[ib]
        ayxne[ib] += sign * ayxse[ib]
        ayxnw[ib] += sign * ayxsw[ib]
        ayyn[ib]  -= sign * ayys[ib]
        ayyne[ib] -= sign * ayyse[ib]
        ayynw[ib] -= sign * ayysw[ib]
        axyn[ib]  -= sign * axys[ib]
        axyne[ib] -= sign * axyse[ib]
        axynw[ib] -= sign * axysw[ib]

        # EAST boundary

        ib = ii[-1, :]

        if boundary[2] == 'S':
            sign = 1
        elif boundary[2] == 'A':
            sign = -1
        elif boundary[2] == '0':
            sign = 0
        else:
            raise ValueError('unknown boundary conditions')

        axxw[ib]  += sign * axxe[ib]
        axxnw[ib] += sign * axxne[ib]
        axxsw[ib] += sign * axxse[ib]
        ayxw[ib]  += sign * ayxe[ib]
        ayxnw[ib] += sign * ayxne[ib]
        ayxsw[ib] += sign * ayxse[ib]
        ayyw[ib]  -= sign * ayye[ib]
        ayynw[ib] -= sign * ayyne[ib]
        ayysw[ib] -= sign * ayyse[ib]
        axyw[ib]  -= sign * axye[ib]
        axynw[ib] -= sign * axyne[ib]
        axysw[ib] -= sign * axyse[ib]

        # WEST boundary

        ib = ii[0, :]

        if boundary[3] == 'S':
            sign = 1
        elif boundary[3] == 'A':
            sign = -1
        elif boundary[3] == '0':
            sign = 0
        else:
            raise ValueError('unknown boundary conditions')

        axxe[ib]  += sign * axxw[ib]
        axxne[ib] += sign * axxnw[ib]
        axxse[ib] += sign * axxsw[ib]
        ayxe[ib]  += sign * ayxw[ib]
        ayxne[ib] += sign * ayxnw[ib]
        ayxse[ib] += sign * ayxsw[ib]
        ayye[ib]  -= sign * ayyw[ib]
        ayyne[ib] -= sign * ayynw[ib]
        ayyse[ib] -= sign * ayysw[ib]
        axye[ib]  -= sign * axyw[ib]
        axyne[ib] -= sign * axynw[ib]
        axyse[ib] -= sign * axysw[ib]

        # Assemble sparse matrix

        iall = ii.flatten()
        i_s = ii[:, :-1].flatten()
        i_n = ii[:, 1:].flatten()
        i_e = ii[1:, :].flatten()
        i_w = ii[:-1, :].flatten()
        i_ne = ii[1:, 1:].flatten()
        i_se = ii[1:, :-1].flatten()
        i_sw = ii[:-1, :-1].flatten()
        i_nw = ii[:-1, 1:].flatten()

        Ixx = numpy.r_[iall, i_w, i_e, i_s, i_n, i_ne, i_se, i_sw, i_nw]
        Jxx = numpy.r_[iall, i_e, i_w, i_n, i_s, i_sw, i_nw, i_ne, i_se]
        Vxx = numpy.r_[axxp[iall], axxe[i_w], axxw[i_e], axxn[i_s], axxs[
            i_n], axxsw[i_ne], axxnw[i_se], axxne[i_sw], axxse[i_nw]]

        Ixy = numpy.r_[iall, i_w, i_e, i_s, i_n, i_ne, i_se, i_sw, i_nw]
        Jxy = numpy.r_[
            iall, i_e, i_w, i_n, i_s, i_sw, i_nw, i_ne, i_se] + nx * ny
        Vxy = numpy.r_[axyp[iall], axye[i_w], axyw[i_e], axyn[i_s], axys[
            i_n], axysw[i_ne], axynw[i_se], axyne[i_sw], axyse[i_nw]]

        Iyx = numpy.r_[
            iall, i_w, i_e, i_s, i_n, i_ne, i_se, i_sw, i_nw] + nx * ny
        Jyx = numpy.r_[iall, i_e, i_w, i_n, i_s, i_sw, i_nw, i_ne, i_se]
        Vyx = numpy.r_[ayxp[iall], ayxe[i_w], ayxw[i_e], ayxn[i_s], ayxs[
            i_n], ayxsw[i_ne], ayxnw[i_se], ayxne[i_sw], ayxse[i_nw]]

        Iyy = numpy.r_[
            iall, i_w, i_e, i_s, i_n, i_ne, i_se, i_sw, i_nw] + nx * ny
        Jyy = numpy.r_[
            iall, i_e, i_w, i_n, i_s, i_sw, i_nw, i_ne, i_se] + nx * ny
        Vyy = numpy.r_[ayyp[iall], ayye[i_w], ayyw[i_e], ayyn[i_s], ayys[
            i_n], ayysw[i_ne], ayynw[i_se], ayyne[i_sw], ayyse[i_nw]]

        I = numpy.r_[Ixx, Ixy, Iyx, Iyy]
        J = numpy.r_[Jxx, Jxy, Jyx, Jyy]
        V = numpy.r_[Vxx, Vxy, Vyx, Vyy]
        A = coo_matrix((V, (I, J))).tocsr()

        return A

    def compute_other_fields(self, neffs, Hxs, Hys):

        from scipy.sparse import coo_matrix

        wl = self.wl
        x = self.x
        y = self.y
        boundary = self.boundary

        Hzs = []
        Exs = []
        Eys = []
        Ezs = []
        for neff, Hx, Hy in zip(neffs, Hxs, Hys):

            dx = numpy.diff(x)
            dy = numpy.diff(y)

            dx = numpy.r_[dx[0], dx, dx[-1]].reshape(-1, 1)
            dy = numpy.r_[dy[0], dy, dy[-1]].reshape(1, -1)

            xc = (x[:-1] + x[1:]) / 2
            yc = (y[:-1] + y[1:]) / 2
            epsxx, epsxy, epsyx, epsyy, epszz = self._get_eps(xc, yc)

            nx = len(x)
            ny = len(y)

            k = 2 * numpy.pi / wl

            ones_nx = numpy.ones((nx, 1))
            ones_ny = numpy.ones((1, ny))

            n = numpy.dot(ones_nx, dy[:, 1:]).flatten()
            s = numpy.dot(ones_nx, dy[:, :-1]).flatten()
            e = numpy.dot(dx[1:, :], ones_ny).flatten()
            w = numpy.dot(dx[:-1, :], ones_ny).flatten()

            exx1 = epsxx[:-1, 1:].flatten()
            exx2 = epsxx[:-1, :-1].flatten()
            exx3 = epsxx[1:, :-1].flatten()
            exx4 = epsxx[1:, 1:].flatten()

            eyy1 = epsyy[:-1, 1:].flatten()
            eyy2 = epsyy[:-1, :-1].flatten()
            eyy3 = epsyy[1:, :-1].flatten()
            eyy4 = epsyy[1:, 1:].flatten()

            exy1 = epsxy[:-1, 1:].flatten()
            exy2 = epsxy[:-1, :-1].flatten()
            exy3 = epsxy[1:, :-1].flatten()
            exy4 = epsxy[1:, 1:].flatten()

            eyx1 = epsyx[:-1, 1:].flatten()
            eyx2 = epsyx[:-1, :-1].flatten()
            eyx3 = epsyx[1:, :-1].flatten()
            eyx4 = epsyx[1:, 1:].flatten()

            ezz1 = epszz[:-1, 1:].flatten()
            ezz2 = epszz[:-1, :-1].flatten()
            ezz3 = epszz[1:, :-1].flatten()
            ezz4 = epszz[1:, 1:].flatten()

            b = neff * k

            bzxne = (0.5 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * eyx4 / ezz4 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy3 * eyy1 * w * eyy2 +
                     0.5 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * (1 - exx4 / ezz4) / ezz3 / ezz2 / (w * exx3 + e * exx2) / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * exx1 * s) / b

            bzxse = (-0.5 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * eyx3 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy1 * w * eyy2 +
                     0.5 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * (1 - exx3 / ezz3) / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * n * exx1 * exx4) / b

            bzxnw = (-0.5 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * eyx1 / ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy2 * e -
                     0.5 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * (1 - exx1 / ezz1) / ezz3 / ezz2 / (w * exx3 + e * exx2) / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * exx4 * s) / b

            bzxsw = (0.5 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * eyx2 / ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * e -
                     0.5 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * (1 - exx2 / ezz2) / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx3 * n * exx1 * exx4) / b

            bzxn = ((0.5 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * n * ezz1 * ezz2 / eyy1 * (2 * eyy1 / ezz1 / n ** 2 + eyx1 / ezz1 / n / w) + 0.5 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * n * ezz4 * ezz3 / eyy4 * (2 * eyy4 / ezz4 / n ** 2 - eyx4 / ezz4 / n / e)) / ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * w * eyy2 * e + ((ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * (0.5 * ezz4 * ((1 - exx1 / ezz1) / n / w - exy1 / ezz1 *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 (2. / n ** 2 - 2 / n ** 2 * s / (n + s))) / exx1 * ezz1 * w + (ezz4 - ezz1) * s / n / (n + s) + 0.5 * ezz1 * (-(1 - exx4 / ezz4) / n / e - exy4 / ezz4 * (2. / n ** 2 - 2 / n ** 2 * s / (n + s))) / exx4 * ezz4 * e) - (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * (-ezz3 * exy2 / n / (n + s) / exx2 * w + (ezz3 - ezz2) * s / n / (n + s) - ezz2 * exy3 / n / (n + s) / exx3 * e)) / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b

            bzxs = ((0.5 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * s * ezz2 * ezz1 / eyy2 * (2 * eyy2 / ezz2 / s ** 2 - eyx2 / ezz2 / s / w) + 0.5 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * s * ezz3 * ezz4 / eyy3 * (2 * eyy3 / ezz3 / s ** 2 + eyx3 / ezz3 / s / e)) / ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * w * eyy2 * e + ((ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * (-ezz4 * exy1 / s / (n + s) / exx1 * w - (ezz4 - ezz1)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   * n / s / (n + s) - ezz1 * exy4 / s / (n + s) / exx4 * e) - (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * (0.5 * ezz3 * (-(1 - exx2 / ezz2) / s / w - exy2 / ezz2 * (2. / s ** 2 - 2 / s ** 2 * n / (n + s))) / exx2 * ezz2 * w - (ezz3 - ezz2) * n / s / (n + s) + 0.5 * ezz2 * ((1 - exx3 / ezz3) / s / e - exy3 / ezz3 * (2. / s ** 2 - 2 / s ** 2 * n / (n + s))) / exx3 * ezz3 * e)) / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b

            bzxe = ((n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * (0.5 * n * ezz4 * ezz3 / eyy4 * (2. / e ** 2 - eyx4 / ezz4 / n / e) + 0.5 * s * ezz3 * ezz4 / eyy3 * (2. / e ** 2 + eyx3 / ezz3 / s / e)) / ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * w * eyy2 * e +
                    (-0.5 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * ezz1 * (1 - exx4 / ezz4) / n / exx4 * ezz4 - 0.5 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * ezz2 * (1 - exx3 / ezz3) / s / exx3 * ezz3) / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b

            bzxw = ((-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * (0.5 * n * ezz1 * ezz2 / eyy1 * (2. / w ** 2 + eyx1 / ezz1 / n / w) + 0.5 * s * ezz2 * ezz1 / eyy2 * (2. / w ** 2 - eyx2 / ezz2 / s / w)) / ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * w * eyy2 * e +
                    (0.5 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * ezz4 * (1 - exx1 / ezz1) / n / exx1 * ezz1 + 0.5 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * ezz3 * (1 - exx2 / ezz2) / s / exx2 * ezz2) / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b

            bzxp = (((-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * (0.5 * n * ezz1 * ezz2 / eyy1 * (-2. / w ** 2 - 2 * eyy1 / ezz1 / n ** 2 + k ** 2 * eyy1 - eyx1 / ezz1 / n / w) + 0.5 * s * ezz2 * ezz1 / eyy2 * (-2. / w ** 2 - 2 * eyy2 / ezz2 / s ** 2 + k ** 2 * eyy2 + eyx2 / ezz2 / s / w)) + (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * (0.5 * n * ezz4 * ezz3 / eyy4 * (-2. / e ** 2 - 2 * eyy4 / ezz4 / n ** 2 + k ** 2 * eyy4 + eyx4 / ezz4 / n / e) + 0.5 * s * ezz3 * ezz4 / eyy3 * (-2. / e ** 2 - 2 * eyy3 / ezz3 / s ** 2 + k ** 2 * eyy3 - eyx3 / ezz3 / s / e))) / ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * w * eyy2 * e + ((ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * (0.5 * ezz4 * (-k **
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     2 * exy1 - (1 - exx1 / ezz1) / n / w - exy1 / ezz1 * (-2. / n ** 2 - 2 / n ** 2 * (n - s) / s)) / exx1 * ezz1 * w + (ezz4 - ezz1) * (n - s) / n / s + 0.5 * ezz1 * (-k ** 2 * exy4 + (1 - exx4 / ezz4) / n / e - exy4 / ezz4 * (-2. / n ** 2 - 2 / n ** 2 * (n - s) / s)) / exx4 * ezz4 * e) - (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * (0.5 * ezz3 * (-k ** 2 * exy2 + (1 - exx2 / ezz2) / s / w - exy2 / ezz2 * (-2. / s ** 2 + 2 / s ** 2 * (n - s) / n)) / exx2 * ezz2 * w + (ezz3 - ezz2) * (n - s) / n / s + 0.5 * ezz2 * (-k ** 2 * exy3 - (1 - exx3 / ezz3) / s / e - exy3 / ezz3 * (-2. / s ** 2 + 2 / s ** 2 * (n - s) / n)) / exx3 * ezz3 * e)) / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b

            bzyne = (0.5 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * (1 - eyy4 / ezz4) / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy3 * eyy1 * w *
                     eyy2 + 0.5 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * exy4 / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * exx1 * s) / b

            bzyse = (-0.5 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * (1 - eyy3 / ezz3) / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy1 * w *
                     eyy2 + 0.5 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * exy3 / ezz3 / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * n * exx1 * exx4) / b

            bzynw = (-0.5 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * (1 - eyy1 / ezz1) / ezz4 / ezz3 / (n * eyy3 + s * eyy4) / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 *
                     eyy2 * e - 0.5 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * exy1 / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * exx4 * s) / b

            bzysw = (0.5 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * (1 - eyy2 / ezz2) / ezz4 / ezz3 / (n * eyy3 + s * eyy4) / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 *
                     e - 0.5 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * exy2 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx3 * n * exx1 * exx4) / b

            bzyn = ((0.5 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * ezz1 * ezz2 / eyy1 * (1 - eyy1 / ezz1) / w - 0.5 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * ezz4 * ezz3 / eyy4 * (1 - eyy4 / ezz4) / e) / ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * w *
                    eyy2 * e + (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * (0.5 * ezz4 * (2. / n ** 2 + exy1 / ezz1 / n / w) / exx1 * ezz1 * w + 0.5 * ezz1 * (2. / n ** 2 - exy4 / ezz4 / n / e) / exx4 * ezz4 * e) / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b

            bzys = ((-0.5 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * ezz2 * ezz1 / eyy2 * (1 - eyy2 / ezz2) / w + 0.5 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * ezz3 * ezz4 / eyy3 * (1 - eyy3 / ezz3) / e) / ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * w *
                    eyy2 * e - (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * (0.5 * ezz3 * (2. / s ** 2 - exy2 / ezz2 / s / w) / exx2 * ezz2 * w + 0.5 * ezz2 * (2. / s ** 2 + exy3 / ezz3 / s / e) / exx3 * ezz3 * e) / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b

            bzye = (((-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * (-n * ezz2 / eyy1 * eyx1 / e / (e + w) + (ezz1 - ezz2) * w / e / (e + w) - s * ezz1 / eyy2 * eyx2 / e / (e + w)) + (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * (0.5 * n * ezz4 * ezz3 / eyy4 * (-(1 - eyy4 / ezz4) / n / e - eyx4 / ezz4 * (2. / e ** 2 - 2 / e ** 2 * w / (e + w))) + 0.5 * s * ezz3 * ezz4 / eyy3 * ((1 - eyy3 / ezz3) / s / e - eyx3 / ezz3 * (2. / e ** 2 - 2 / e ** 2 * w / (e + w))) + (ezz4 - ezz3) * w / e / (e + w))) / ezz4 /
                    ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * w * eyy2 * e + (0.5 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * ezz1 * (2 * exx4 / ezz4 / e ** 2 - exy4 / ezz4 / n / e) / exx4 * ezz4 * e - 0.5 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * ezz2 * (2 * exx3 / ezz3 / e ** 2 + exy3 / ezz3 / s / e) / exx3 * ezz3 * e) / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b

            bzyw = (((-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * (0.5 * n * ezz1 * ezz2 / eyy1 * ((1 - eyy1 / ezz1) / n / w - eyx1 / ezz1 * (2. / w ** 2 - 2 / w ** 2 * e / (e + w))) - (ezz1 - ezz2) * e / w / (e + w) + 0.5 * s * ezz2 * ezz1 / eyy2 * (-(1 - eyy2 / ezz2) / s / w - eyx2 / ezz2 * (2. / w ** 2 - 2 / w ** 2 * e / (e + w)))) + (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * (-n * ezz3 / eyy4 * eyx4 / w / (e + w) - s * ezz4 / eyy3 * eyx3 / w / (e + w) - (ezz4 - ezz3) * e / w / (e + w))) / ezz4 /
                    ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * w * eyy2 * e + (0.5 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * ezz4 * (2 * exx1 / ezz1 / w ** 2 + exy1 / ezz1 / n / w) / exx1 * ezz1 * w - 0.5 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * ezz3 * (2 * exx2 / ezz2 / w ** 2 - exy2 / ezz2 / s / w) / exx2 * ezz2 * w) / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b

            bzyp = (((-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * (0.5 * n * ezz1 * ezz2 / eyy1 * (-k ** 2 * eyx1 - (1 - eyy1 / ezz1) / n / w - eyx1 / ezz1 * (-2. / w ** 2 + 2 / w ** 2 * (e - w) / e)) + (ezz1 - ezz2) * (e - w) / e / w + 0.5 * s * ezz2 * ezz1 / eyy2 * (-k ** 2 * eyx2 + (1 - eyy2 / ezz2) / s / w - eyx2 / ezz2 * (-2. / w ** 2 + 2 / w ** 2 * (e - w) / e))) + (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * (0.5 * n * ezz4 * ezz3 / eyy4 * (-k ** 2 * eyx4 + (1 - eyy4 / ezz4) / n / e - eyx4 / ezz4 * (-2. / e ** 2 - 2 / e ** 2 * (e - w) / w)) + 0.5 * s * ezz3 * ezz4 / eyy3 * (-k ** 2 * eyx3 - (1 - eyy3 / ezz3) / s / e - eyx3 / ezz3 * (-2. / e ** 2 - 2 / e ** 2 * (e - w) / w)) + (ezz4 - ezz3) * (e - w) / e / w)) / ezz4 / ezz3 / (n * eyy3 + s * eyy4) /
                    ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * w * eyy2 * e + ((ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * (0.5 * ezz4 * (-2. / n ** 2 - 2 * exx1 / ezz1 / w ** 2 + k ** 2 * exx1 - exy1 / ezz1 / n / w) / exx1 * ezz1 * w + 0.5 * ezz1 * (-2. / n ** 2 - 2 * exx4 / ezz4 / e ** 2 + k ** 2 * exx4 + exy4 / ezz4 / n / e) / exx4 * ezz4 * e) - (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * (0.5 * ezz3 * (-2. / s ** 2 - 2 * exx2 / ezz2 / w ** 2 + k ** 2 * exx2 + exy2 / ezz2 / s / w) / exx2 * ezz2 * w + 0.5 * ezz2 * (-2. / s ** 2 - 2 * exx3 / ezz3 / e ** 2 + k ** 2 * exx3 - exy3 / ezz3 / s / e) / exx3 * ezz3 * e)) / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b

            ii = numpy.arange(nx * ny).reshape(nx, ny)

            # NORTH boundary

            ib = ii[:, -1]

            if boundary[0] == 'S':
                sign = 1
            elif boundary[0] == 'A':
                sign = -1
            elif boundary[0] == '0':
                sign = 0
            else:
                raise ValueError('unknown boundary conditions')

            bzxs[ib]  += sign * bzxn[ib]
            bzxse[ib] += sign * bzxne[ib]
            bzxsw[ib] += sign * bzxnw[ib]
            bzys[ib]  -= sign * bzyn[ib]
            bzyse[ib] -= sign * bzyne[ib]
            bzysw[ib] -= sign * bzynw[ib]

            # SOUTH boundary

            ib = ii[:, 0]

            if boundary[1] == 'S':
                sign = 1
            elif boundary[1] == 'A':
                sign = -1
            elif boundary[1] == '0':
                sign = 0
            else:
                raise ValueError('unknown boundary conditions')

            bzxn[ib]  += sign * bzxs[ib]
            bzxne[ib] += sign * bzxse[ib]
            bzxnw[ib] += sign * bzxsw[ib]
            bzyn[ib]  -= sign * bzys[ib]
            bzyne[ib] -= sign * bzyse[ib]
            bzynw[ib] -= sign * bzysw[ib]

            # EAST boundary

            ib = ii[-1, :]

            if boundary[2] == 'S':
                sign = 1
            elif boundary[2] == 'A':
                sign = -1
            elif boundary[2] == '0':
                sign = 0
            else:
                raise ValueError('unknown boundary conditions')

            bzxw[ib]  += sign * bzxe[ib]
            bzxnw[ib] += sign * bzxne[ib]
            bzxsw[ib] += sign * bzxse[ib]
            bzyw[ib]  -= sign * bzye[ib]
            bzynw[ib] -= sign * bzyne[ib]
            bzysw[ib] -= sign * bzyse[ib]

            # WEST boundary

            ib = ii[0, :]

            if boundary[3] == 'S':
                sign = 1
            elif boundary[3] == 'A':
                sign = -1
            elif boundary[3] == '0':
                sign = 0
            else:
                raise ValueError('unknown boundary conditions')

            bzxe[ib]  += sign * bzxw[ib]
            bzxne[ib] += sign * bzxnw[ib]
            bzxse[ib] += sign * bzxsw[ib]
            bzye[ib]  -= sign * bzyw[ib]
            bzyne[ib] -= sign * bzynw[ib]
            bzyse[ib] -= sign * bzysw[ib]

            # Assemble sparse matrix

            iall = ii.flatten()
            i_s = ii[:, :-1].flatten()
            i_n = ii[:, 1:].flatten()
            i_e = ii[1:, :].flatten()
            i_w = ii[:-1, :].flatten()
            i_ne = ii[1:, 1:].flatten()
            i_se = ii[1:, :-1].flatten()
            i_sw = ii[:-1, :-1].flatten()
            i_nw = ii[:-1, 1:].flatten()

            Izx = numpy.r_[iall, i_w, i_e, i_s, i_n, i_ne, i_se, i_sw, i_nw]
            Jzx = numpy.r_[iall, i_e, i_w, i_n, i_s, i_sw, i_nw, i_ne, i_se]
            Vzx = numpy.r_[bzxp[iall], bzxe[i_w], bzxw[i_e], bzxn[i_s], bzxs[
                i_n], bzxsw[i_ne], bzxnw[i_se], bzxne[i_sw], bzxse[i_nw]]

            Izy = numpy.r_[iall, i_w, i_e, i_s, i_n, i_ne, i_se, i_sw, i_nw]
            Jzy = numpy.r_[
                iall, i_e, i_w, i_n, i_s, i_sw, i_nw, i_ne, i_se] + nx * ny
            Vzy = numpy.r_[bzyp[iall], bzye[i_w], bzyw[i_e], bzyn[i_s], bzys[
                i_n], bzysw[i_ne], bzynw[i_se], bzyne[i_sw], bzyse[i_nw]]

            I = numpy.r_[Izx, Izy]
            J = numpy.r_[Jzx, Jzy]
            V = numpy.r_[Vzx, Vzy]
            B = coo_matrix((V, (I, J))).tocsr()

            HxHy = numpy.r_[Hx, Hy]
            Hz = B * HxHy.ravel() / 1j
            Hz = Hz.reshape(Hx.shape)

            # in xc e yc
            exx = epsxx[1:-1, 1:-1]
            exy = epsxy[1:-1, 1:-1]
            eyx = epsyx[1:-1, 1:-1]
            eyy = epsyy[1:-1, 1:-1]
            ezz = epszz[1:-1, 1:-1]
            edet = (exx * eyy - exy * eyx)

            h = e.reshape(nx, ny)[:-1, :-1]
            v = n.reshape(nx, ny)[:-1, :-1]

            # in xc e yc
            Dx = neff * EMpy_gpu.utils.centered2d(Hy) + (
                Hz[:-1, 1:] + Hz[1:, 1:] - Hz[:-1, :-1] - Hz[1:, :-1]) / (2j * k * v)
            Dy = -neff * EMpy_gpu.utils.centered2d(Hx) - (
                Hz[1:, :-1] + Hz[1:, 1:] - Hz[:-1, 1:] - Hz[:-1, :-1]) / (2j * k * h)
            Dz = ((Hy[1:, :-1] + Hy[1:, 1:] - Hy[:-1, 1:] - Hy[:-1, :-1]) / (2 * h) -
                  (Hx[:-1, 1:] + Hx[1:, 1:] - Hx[:-1, :-1] - Hx[1:, :-1]) / (2 * v)) / (1j * k)

            Ex = (eyy * Dx - exy * Dy) / edet
            Ey = (exx * Dy - eyx * Dx) / edet
            Ez = Dz / ezz

            Hzs.append(Hz)
            Exs.append(Ex)
            Eys.append(Ey)
            Ezs.append(Ez)

        return (Hzs, Exs, Eys, Ezs)

    def solve(self, neigs=4, tol=0, guess=None):
        """
        This function finds the eigenmodes.

        Parameters
        ----------
        neigs : int
            number of eigenmodes to find
        tol : float
            Relative accuracy for eigenvalues.
            The default value of 0 implies machine precision.
        guess : float
            A guess for the refractive index.
            The modesolver will only finds eigenvectors with an
            effective refrative index higher than this value.

        Returns
        -------
        self : an instance of the VFDModeSolver class
            obtain the fields of interest for specific modes using, for example:
            solver = EMpy.modesolvers.FD.VFDModeSolver(wavelength, x, y, epsf, boundary).solve()
            Ex = solver.modes[0].Ex
            Ey = solver.modes[0].Ey
            Ez = solver.modes[0].Ez
        """

        from scipy.sparse.linalg import eigen 

        self.nmodes = neigs
        self.tol = tol

        A = self.build_matrix()

        if guess is not None:
            # calculate shift for eigs function
            k = 2 * numpy.pi / self.wl
            shift = (guess * k) ** 2
        else:
            shift = None

        # ! Here
        # Here is where the actual mode-solving takes place!
        [eigvals, eigvecs] = eigen.eigs(A,
                                        k=neigs,
                                        which='LR',
                                        tol=tol,
                                        ncv=10*neigs,
                                        return_eigenvectors=True,
                                        sigma=shift)

        neffs = self.wl * scipy.sqrt(eigvals) / (2 * numpy.pi)
        Hxs = []
        Hys = []
        nx = self.nx
        ny = self.ny
        for ieig in range(neigs):
            Hxs.append(eigvecs[:nx * ny, ieig].reshape(nx, ny))
            Hys.append(eigvecs[nx * ny:, ieig].reshape(nx, ny))

        # sort the modes
        idx = numpy.flipud(numpy.argsort(neffs))
        neffs = neffs[idx]
        tmpx = []
        tmpy = []
        for i in idx:
            tmpx.append(Hxs[i])
            tmpy.append(Hys[i])
        Hxs = tmpx
        Hys = tmpy

        [Hzs, Exs, Eys, Ezs] = self.compute_other_fields(neffs, Hxs, Hys)

        self.modes = []
        for (neff, Hx, Hy, Hz, Ex, Ey, Ez) in zip(neffs, Hxs, Hys, Hzs, Exs, Eys, Ezs):
            self.modes.append(
                FDMode(self.wl, self.x, self.y, neff, Ex, Ey, Ez, Hx, Hy, Hz).normalize())

        return self

    def save_modes_for_FDTD(self, x=None, y=None):
        for im, m in enumerate(self.modes):
            m.save_for_FDTD(str(im), x, y)

    def __str__(self):
        descr = 'Vectorial Finite Difference Modesolver\n'
        return descr


class FDMode(Mode):

    def __init__(self, wl, x, y, neff, Ex, Ey, Ez, Hx, Hy, Hz):
        self.wl = wl
        self.x = x
        self.y = y
        self.neff = neff
        self.Ex = Ex
        self.Ey = Ey
        self.Ez = Ez
        self.Hx = Hx
        self.Hy = Hy
        self.Hz = Hz

    def get_x(self, n=None):
        if n is None:
            return self.x
        return numpy.linspace(self.x[0], self.x[-1], n)

    def get_y(self, n=None):
        if n is None:
            return self.y
        return numpy.linspace(self.y[0], self.y[-1], n)

    def get_field(self, fname, x=None, y=None):

        if fname == 'Ex':
            f = self.Ex
            centered = True
        elif fname == 'Ey':
            f = self.Ey
            centered = True
        elif fname == 'Ez':
            f = self.Ez
            centered = True
        elif fname == 'Hx':
            f = self.Hx
            centered = False
        elif fname == 'Hy':
            f = self.Hy
            centered = False
        elif fname == 'Hz':
            f = self.Hz
            centered = False

        if (x is None) and (y is None):
            return f

        if not centered:
            # magnetic fields are not centered
            x0 = self.x
            y0 = self.y
        else:
            # electric fields and intensity are centered
            x0 = EMpy_gpu.utils.centered1d(self.x)
            y0 = EMpy_gpu.utils.centered1d(self.y)

        return EMpy_gpu.utils.interp2(x, y, x0, y0, f)

    def intensityTETM(self, x=None, y=None):
        I_TE = self.Ex * EMpy_gpu.utils.centered2d(numpy.conj(self.Hy)) / 2.
        I_TM = -self.Ey * EMpy_gpu.utils.centered2d(numpy.conj(self.Hx)) / 2.
        if x is None and y is None:
            return (I_TE, I_TM)
        else:
            x0 = EMpy_gpu.utils.centered1d(self.x)
            y0 = EMpy_gpu.utils.centered1d(self.y)
            I_TE_ = EMpy_gpu.utils.interp2(x, y, x0, y0, I_TE)
            I_TM_ = EMpy_gpu.utils.interp2(x, y, x0, y0, I_TM)
            return (I_TE_, I_TM_)

    def intensity(self, x=None, y=None):
        I_TE, I_TM = self.intensityTETM(x, y)
        return I_TE + I_TM

    def TEfrac(self, x_=None, y_=None):
        if x_ is None:
            x = EMpy_gpu.utils.centered1d(self.x)
        else:
            x = x_
        if y_ is None:
            y = EMpy_gpu.utils.centered1d(self.y)
        else:
            y = y_
        STE, STM = self.intensityTETM(x_, y_)
        num = EMpy_gpu.utils.trapz2(numpy.abs(STE), x=x, y=y)
        den = EMpy_gpu.utils.trapz2(numpy.abs(STE) + numpy.abs(STM), x=x, y=y)
        return num / den

    def norm(self):
        x = EMpy_gpu.utils.centered1d(self.x)
        y = EMpy_gpu.utils.centered1d(self.y)
        return scipy.sqrt(EMpy_gpu.utils.trapz2(self.intensity(), x=x, y=y))

    def normalize(self):
        n = self.norm()
        self.Ex /= n
        self.Ey /= n
        self.Ez /= n
        self.Hx /= n
        self.Hy /= n
        self.Hz /= n

        return self

    def overlap(self, m, x=None, y=None):

        x1 = EMpy_gpu.utils.centered1d(self.x)
        y1 = EMpy_gpu.utils.centered1d(self.y)

        x2 = EMpy_gpu.utils.centered1d(m.x)
        y2 = EMpy_gpu.utils.centered1d(m.y)

        if x is None:
            x = x2

        if y is None:
            y = y2

        # Interpolates m1 onto m2 grid:
        Ex1 = EMpy_gpu.utils.interp2(x, y, x1, y1, self.Ex)
        Ey1 = EMpy_gpu.utils.interp2(x, y, x1, y1, self.Ey)
        Hx2 = EMpy_gpu.utils.interp2(x, y, x2, y2, m.Hx)
        Hy2 = EMpy_gpu.utils.interp2(x, y, x2, y2, m.Hy)

        intensity = (Ex1 * EMpy_gpu.utils.centered2d(numpy.conj(Hy2)) -
                     Ey1 * EMpy_gpu.utils.centered2d(numpy.conj(Hx2))) / 2.

        return EMpy_gpu.utils.trapz2(intensity, x=x, y=y)

    def get_fields_for_FDTD(self, x=None, y=None):
        """Get mode's field on a staggered grid.

        Note: ignores some fields on the boudaries.

        """

        if x is None:
            x = self.x
        if y is None:
            y = self.y

        # Ex: ignores y = 0, max
        x_Ex = EMpy_gpu.utils.centered1d(self.x)
        y_Ex = EMpy_gpu.utils.centered1d(self.y)
        x_Ex_FDTD = EMpy_gpu.utils.centered1d(x)
        y_Ex_FDTD = y[1:-1]
        Ex_FDTD = EMpy_gpu.utils.interp2(x_Ex_FDTD, y_Ex_FDTD, x_Ex, y_Ex, self.Ex)
        # Ey: ignores x = 0, max
        x_Ey = EMpy_gpu.utils.centered1d(self.x)
        y_Ey = EMpy_gpu.utils.centered1d(self.y)
        x_Ey_FDTD = x[1:-1]
        y_Ey_FDTD = EMpy_gpu.utils.centered1d(y)
        Ey_FDTD = EMpy_gpu.utils.interp2(x_Ey_FDTD, y_Ey_FDTD, x_Ey, y_Ey, self.Ey)
        # Ez: ignores x, y = 0, max
        x_Ez = EMpy_gpu.utils.centered1d(self.x)
        y_Ez = EMpy_gpu.utils.centered1d(self.y)
        x_Ez_FDTD = x[1:-1]
        y_Ez_FDTD = y[1:-1]
        Ez_FDTD = EMpy_gpu.utils.interp2(x_Ez_FDTD, y_Ez_FDTD, x_Ez, y_Ez, self.Ez)
        # Hx: ignores x = 0, max, /120pi, reverse direction
        x_Hx = self.x
        y_Hx = self.y
        x_Hx_FDTD = x[1:-1]
        y_Hx_FDTD = EMpy_gpu.utils.centered1d(y)
        Hx_FDTD = EMpy_gpu.utils.interp2(
            x_Hx_FDTD, y_Hx_FDTD, x_Hx, y_Hx, self.Hx) / (-120. * numpy.pi)
        # Hy: ignores y = 0, max, /120pi, reverse direction
        x_Hy = self.x
        y_Hy = self.y
        x_Hy_FDTD = EMpy_gpu.utils.centered1d(x)
        y_Hy_FDTD = y[1:-1]
        Hy_FDTD = EMpy_gpu.utils.interp2(
            x_Hy_FDTD, y_Hy_FDTD, x_Hy, y_Hy, self.Hy) / (-120. * numpy.pi)
        # Hz: /120pi, reverse direction
        x_Hz = self.x
        y_Hz = self.y
        x_Hz_FDTD = EMpy_gpu.utils.centered1d(x)
        y_Hz_FDTD = EMpy_gpu.utils.centered1d(y)
        Hz_FDTD = EMpy_gpu.utils.interp2(
            x_Hz_FDTD, y_Hz_FDTD, x_Hz, y_Hz, self.Hz) / (-120. * numpy.pi)

        return (Ex_FDTD, Ey_FDTD, Ez_FDTD, Hx_FDTD, Hy_FDTD, Hz_FDTD)

    @staticmethod
    def plot_field(x, y, field):
        try:
            import pylab
        except ImportError:
            print('no pylab installed')
            return
        pylab.hot()
        pylab.contour(x, y, numpy.abs(field.T), 16)
        pylab.axis('image')

    def plot_Ex(self, x=None, y=None):
        if x is None:
            x = EMpy_gpu.utils.centered1d(self.x)
        if y is None:
            y = EMpy_gpu.utils.centered1d(self.y)
        Ex = self.get_field('Ex', x, y)
        self.plot_field(x, y, Ex)

    def plot_Ey(self, x=None, y=None):
        if x is None:
            x = EMpy_gpu.utils.centered1d(self.x)
        if y is None:
            y = EMpy_gpu.utils.centered1d(self.y)
        Ey = self.get_field('Ey', x, y)
        self.plot_field(x, y, Ey)

    def plot_Ez(self, x=None, y=None):
        if x is None:
            x = EMpy_gpu.utils.centered1d(self.x)
        if y is None:
            y = EMpy_gpu.utils.centered1d(self.y)
        Ez = self.get_field('Ez', x, y)
        self.plot_field(x, y, Ez)

    def plot_Hx(self, x=None, y=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        Hx = self.get_field('Hx', x, y)
        self.plot_field(x, y, Hx)

    def plot_Hy(self, x=None, y=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        Hy = self.get_field('Hy', x, y)
        self.plot_field(x, y, Hy)

    def plot_Hz(self, x=None, y=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        Hz = self.get_field('Hz', x, y)
        self.plot_field(x, y, Hz)

    def plot_intensity(self):
        x = EMpy_gpu.utils.centered1d(self.x)
        y = EMpy_gpu.utils.centered1d(self.y)
        I = self.intensity(x, y)
        self.plot_field(x, y, I)

    def plot(self):
        """Plot the mode's fields."""
        try:
            import pylab
        except ImportError:
            print('no pylab installed')
            return
        pylab.figure()
        pylab.subplot(2, 3, 1)
        self.plot_Ex()
        pylab.title('Ex')
        pylab.subplot(2, 3, 2)
        self.plot_Ey()
        pylab.title('Ey')
        pylab.subplot(2, 3, 3)
        self.plot_Ez()
        pylab.title('Ez')
        pylab.subplot(2, 3, 4)
        self.plot_Hx()
        pylab.title('Hx')
        pylab.subplot(2, 3, 5)
        self.plot_Hy()
        pylab.title('Hy')
        pylab.subplot(2, 3, 6)
        self.plot_Hz()
        pylab.title('Hz')


def stretchmesh(x, y, nlayers, factor, method='PPPP'):

    # OKKIO: check me!

    # This function can be used to continuously stretch the grid
    # spacing at the edges of the computation window for
    # finite-difference calculations.  This is useful when you would
    # like to increase the size of the computation window without
    # increasing the total number of points in the computational
    # domain.  The program implements four different expansion
    # methods: uniform, linear, parabolic (the default) and
    # geometric.  The first three methods also allow for complex
    # coordinate stretching, which is useful for creating
    # perfectly-matched non-reflective boundaries.
    #
    # USAGE:
    #
    # [x,y] = stretchmesh(x,y,nlayers,factor);
    # [x,y] = stretchmesh(x,y,nlayers,factor,method);
    # [x,y,xc,yc] = stretchmesh(x,y,nlayers,factor);
    # [x,y,xc,yc] = stretchmesh(x,y,nlayers,factor,method);
    # [x,y,xc,yc,dx,dy] = stretchmesh(x,y,nlayers,factor);
    # [x,y,xc,yc,dx,dy] = stretchmesh(x,y,nlayers,factor,method);
    #
    # INPUT:
    #
    # x,y - vectors that specify the vertices of the original
    #   grid, which are usually linearly spaced.
    # nlayers - vector that specifies how many layers of the grid
    #   you would like to expand:
    # nlayers(1) = # of layers on the north boundary to stretch
    # nlayers(2) = # of layers on the south boundary to stretch
    # nlayers(3) = # of layers on the east boundary to stretch
    # nlayers(4) = # of layers on the west boundary to stretch
    # factor - cumulative factor by which the layers are to be
    #   expanded.  As with nlayers, this can be a 4-vector.
    # method - 4-letter string specifying the method of
    #   stretching for each of the four boundaries.  Four different
    #   methods are supported: uniform, linear, parabolic (default)
    #   and geometric.  For example, method = 'LLLG' will use linear
    #   expansion for the north, south and east boundaries and
    #   geometric expansion for the west boundary.
    #
    # OUTPUT:
    #
    # x,y - the vertices of the new stretched grid
    # xc,yc (optional) - the center cell coordinates of the
    #   stretched grid
    # dx,dy (optional) - the grid spacing (dx = diff(x))

    xx = x.astype(complex)
    yy = y.astype(complex)

    nlayers *= numpy.ones(4, dtype=int) 
    factor *= numpy.ones(4)

    for idx, (n, f, m) in enumerate(zip(nlayers, factor, method.upper())):

        if n > 0 and f != 1:

            if idx == 0:
                # north boundary
                kv = numpy.arange(len(y) - 1 - n, len(y))
                z = yy
                q1 = z[-1 - n]
                q2 = z[-1]
            elif idx == 1:
                # south boundary
                kv = numpy.arange(0, n)
                z = yy
                q1 = z[n]
                q2 = z[0]
            elif idx == 2:
                # east boundary
                kv = numpy.arange(len(x) - 1 - n, len(x))
                z = xx
                q1 = z[-1 - n]
                q2 = z[-1]
            elif idx == 3:
                # west boundary
                kv = numpy.arange(0, n)
                z = xx
                q1 = z[n]
                q2 = z[0]

            kv = kv.astype(int)

            if m == 'U':
                c = numpy.polyfit([q1, q2], [q1, q1 + f * (q2 - q1)], 1)
                z[kv] = numpy.polyval(c, z[kv])
            elif m == 'L':
                c = (f - 1) / (q2 - q1)
                b = 1 - 2 * c * q1
                a = q1 - b * q1 - c * q1 ** 2
                z[kv] = a + b * z[kv] + c * z[kv] ** 2
            elif m == 'P':
                z[kv] = z[kv] + (f - 1) * (z[kv] - q1) ** 3 / (q2 - q1) ** 2
            elif m == 'G':
                b = scipy.optimize.newton(
                    lambda s: numpy.exp(s) - 1 - f * s, f)
                a = (q2 - q1) / b
                z[kv] = q1 + a * (numpy.exp((z[kv] - q1) / a) - 1)

    xx = xx.real + 1j * numpy.abs(xx.imag)
    yy = yy.real + 1j * numpy.abs(yy.imag)

    xc = (xx[:-1] + xx[1:]) / 2.
    yc = (yy[:-1] + yy[1:]) / 2.

    dx = numpy.diff(xx)
    dy = numpy.diff(yy)

    return (xx, yy, xc, yc, dx, dy)
