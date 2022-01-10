# pylint: disable=R0913,R0914,W0201,W0622,C0302,R0902,R0903,W1001,W0612,W0613

"""Useful functions and objects used more or less everywhere."""

from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import range
from builtins import object

__author__ = 'Lorenzo Bolla'

import numpy
import EMpy_gpu.constants
import EMpy_gpu.materials
import scipy.linalg
import scipy.interpolate
import scipy.optimize
import time
import sys


class Layer(object):

    """A layer is defined by a material (iso or aniso) and a thickness."""

    def __init__(self, mat, thickness):
        """Set the material and the thickness."""

        self.mat = mat
        self.thickness = thickness

    def isIsotropic(self):
        """Return True if the material is isotropic, False if anisotropic."""

        return self.mat.isIsotropic()

    def getEPSFourierCoeffs(self, wl, n, anisotropic=True):
        """Return the Fourier coefficients of eps and eps**-1, orders [-n,n]."""

        nood = 2 * n + 1
        hmax = nood - 1
        if not anisotropic:
            # isotropic
            EPS = numpy.zeros(2 * hmax + 1, dtype=complex)
            EPS1 = numpy.zeros_like(EPS)
            rix = self.mat.n(wl)
            EPS[hmax] = rix ** 2
            EPS1[hmax] = rix ** -2
            return EPS, EPS1
        else:
            # anisotropic
            EPS = numpy.zeros((3, 3, 2 * hmax + 1), dtype=complex)
            EPS1 = numpy.zeros_like(EPS)
            EPS[:, :, hmax] = numpy.squeeze(
                self.mat.epsilonTensor(wl)) / EMpy_gpu.constants.eps0
            EPS1[:, :, hmax] = scipy.linalg.inv(EPS[:, :, hmax])
            return EPS, EPS1

    def capacitance(self, area=1., wl=0):
        """Capacitance = eps0 * eps_r * area / thickness."""

        if self.isIsotropic():
            eps = EMpy_gpu.constants.eps0 * numpy.real(self.mat.n(wl).item() ** 2)
        else:
            # suppose to compute the capacitance along the z-axis
            eps = self.mat.epsilonTensor(wl)[2, 2, 0]

        return eps * area / self.thickness

    def __str__(self):
        """Return the description of a layer."""

        return "%s, thickness: %g" % (self.mat, self.thickness)


class BinaryGrating(object):
    """A Binary Grating is defined by two materials (iso or aniso), a
    duty cycle, a pitch and a thickness."""

    def __init__(self, mat1, mat2, dc, pitch, thickness):
        """Set the materials, the duty cycle and the thickness."""
        self.mat1 = mat1
        self.mat2 = mat2
        self.dc = dc
        self.pitch = pitch
        self.thickness = thickness

    def isIsotropic(self):
        """Return True if both the materials are isotropic, False otherwise."""
        return self.mat1.isIsotropic() and self.mat2.isIsotropic()

    def getEPSFourierCoeffs(self, wl, n, anisotropic=True):
        """Return the Fourier coefficients of eps and eps**-1, orders [-n,n]."""
        nood = 2 * n + 1
        hmax = nood - 1
        if not anisotropic:
            # isotropic
            rix1 = self.mat1.n(wl)
            rix2 = self.mat2.n(wl)
            f = self.dc
            h = numpy.arange(-hmax, hmax + 1)
            EPS = (rix1 ** 2 - rix2 ** 2) * f * \
                numpy.sinc(h * f) + rix2 ** 2 * (h == 0)
            EPS1 = (rix1 ** -2 - rix2 ** -2) * f * \
                numpy.sinc(h * f) + rix2 ** -2 * (h == 0)
            return EPS, EPS1
        else:
            # anisotropic
            EPS = numpy.zeros((3, 3, 2 * hmax + 1), dtype=complex)
            EPS1 = numpy.zeros_like(EPS)
            eps1 = numpy.squeeze(
                self.mat1.epsilonTensor(wl)) / EMpy_gpu.constants.eps0
            eps2 = numpy.squeeze(
                self.mat2.epsilonTensor(wl)) / EMpy_gpu.constants.eps0
            f = self.dc
            h = numpy.arange(-hmax, hmax + 1)
            for ih, hh in enumerate(h):
                EPS[:, :, ih] = (eps1 - eps2) * f * \
                    numpy.sinc(hh * f) + eps2 * (hh == 0)
                EPS1[:, :, ih] = (
                    scipy.linalg.inv(eps1) - scipy.linalg.inv(eps2)
                ) * f * numpy.sinc(hh * f) + scipy.linalg.inv(eps2) * (hh == 0)
            return EPS, EPS1

    def capacitance(self, area=1., wl=0):
        """Capacitance = eps0 * eps_r * area / thickness."""

        if self.isIsotropic():
            eps = EMpy_gpu.constants.eps0 * numpy.real(
                self.mat1.n(wl) ** 2 * self.dc + self.mat2.n(wl) ** 2
                * (1 - self.dc))
        else:
            eps1 = self.mat1.epsilonTensor(wl)[2, 2, 0]
            eps2 = self.mat2.epsilonTensor(wl)[2, 2, 0]
            eps = numpy.real(eps1 * self.dc + eps2 * (1 - self.dc))

        return eps * area / self.thickness

    def __str__(self):
        """Return the description of a binary grating."""
        return "(%s, %s), dc: %g, pitch: %g, thickness: %g" % (
            self.mat1, self.mat2, self.dc, self.pitch, self.thickness)


class SymmetricDoubleGrating(object):
    """A Symmetric Double Grating is defined by three materials (iso
    or aniso), two duty cycles, a pitch and a thickness.

    Inside the pitch there are two rect of width dc1*pitch of mat1 and
    dc2*pitch of mat2, with a spacer of fixed width made of mat3 between them.
    """

    def __init__(self, mat1, mat2, mat3, dc1, dc2, pitch, thickness):
        """Set the materials, the duty cycle and the thickness."""
        self.mat1 = mat1
        self.mat2 = mat2
        self.mat3 = mat3
        self.dc1 = dc1
        self.dc2 = dc2
        self.pitch = pitch
        self.thickness = thickness

    def isIsotropic(self):
        """Return True if all the materials are isotropic, False otherwise."""
        return (self.mat1.isIsotropic() and
                self.mat2.isIsotropic() and
                self.mat3.isIsotropic())

    def getEPSFourierCoeffs(self, wl, n, anisotropic=True):
        """Return the Fourier coefficients of eps and eps**-1, orders [-n,n]."""
        nood = 2 * n + 1
        hmax = nood - 1
        if not anisotropic:
            # isotropic
            rix1 = self.mat1.n(wl)
            rix2 = self.mat2.n(wl)
            rix3 = self.mat3.n(wl)
            f1 = self.dc1
            f2 = self.dc2
            h = numpy.arange(-hmax, hmax + 1)
            N = len(h)
            A = -N*f1 / 2.
            B = N*f2 / 2.
            EPS = (
                rix3 ** 2 * (h == 0) + (rix1 ** 2 - rix3 ** 2) * f1 *
                numpy.sinc(h * f1) * numpy.exp(2j * numpy.pi * h / N * A) +
                (rix2 ** 2 - rix3 ** 2) * f2 * numpy.sinc(h * f2) *
                numpy.exp(2j * numpy.pi * h / N * B)
            )
            EPS1 = (
                rix3 ** -2 * (h == 0) + (rix1 ** -2 - rix3 ** -2) * f1 *
                numpy.sinc(h * f1) * numpy.exp(2j * numpy.pi * h / N * A) +
                (rix2 ** -2 - rix3 ** -2) * f2 * numpy.sinc(h * f2) *
                numpy.exp(2j * numpy.pi * h / N * B)
            )
            return EPS, EPS1
        else:
            # anisotropic
            EPS = numpy.zeros((3, 3, 2 * hmax + 1), dtype=complex)
            EPS1 = numpy.zeros_like(EPS)
            eps1 = numpy.squeeze(
                self.mat1.epsilonTensor(wl)) / EMpy_gpu.constants.eps0
            eps2 = numpy.squeeze(
                self.mat2.epsilonTensor(wl)) / EMpy_gpu.constants.eps0
            eps3 = numpy.squeeze(
                self.mat3.epsilonTensor(wl)) / EMpy_gpu.constants.eps0
            f1 = self.dc1
            f2 = self.dc2
            h = numpy.arange(-hmax, hmax + 1)
            N = len(h)
            A = -N*f1 / 2.
            B = N*f2 / 2.
            for ih, hh in enumerate(h):
                EPS[:, :, ih] = (
                    (eps1 - eps3) * f1 * numpy.sinc(hh * f1) *
                    numpy.exp(2j * numpy.pi * hh / N * A) +
                    (eps2 - eps3) * f2 * numpy.sinc(hh * f2) *
                    numpy.exp(2j * numpy.pi * hh / N * B) +
                    eps3 * (hh == 0)
                )
                EPS1[:, :, ih] = (
                    (scipy.linalg.inv(eps1) - scipy.linalg.inv(eps3)) * f1 *
                    numpy.sinc(hh * f1) *
                    numpy.exp(2j * numpy.pi * hh / N * A) +
                    (scipy.linalg.inv(eps2) - scipy.linalg.inv(eps3)) * f2 *
                    numpy.sinc(hh * f2) *
                    numpy.exp(2j * numpy.pi * hh / N * B) +
                    scipy.linalg.inv(eps3) * (hh == 0)
                )
            return EPS, EPS1

    def capacitance(self, area=1., wl=0):
        """Capacitance = eps0 * eps_r * area / thickness."""

        if self.isIsotropic():
            eps = EMpy_gpu.constants.eps0 * numpy.real(
                self.mat1.n(wl) ** 2 * self.dc1 + self.mat2.n(wl) ** 2
                * self.dc2 + self.mat3.n(wl) ** 2 * (1 - self.dc1 - self.dc2))
        else:
            eps1 = self.mat1.epsilonTensor(wl)[2, 2, 0]
            eps2 = self.mat2.epsilonTensor(wl)[2, 2, 0]
            eps3 = self.mat3.epsilonTensor(wl)[2, 2, 0]
            eps = numpy.real(
                eps1 * self.dc1 + eps2 * self.dc2 +
                eps3 * (1 - self.dc1 - self.dc2))

        return eps * area / self.thickness

    def __str__(self):
        """Return the description of a binary grating."""
        return "(%s, %s, %s), dc1: %g, dc2: %g, pitch: %g, thickness: %g" % (
            self.mat1, self.mat2, self.mat3, self.dc1, self.dc2, self.pitch,
            self.thickness)


class AsymmetricDoubleGrating(SymmetricDoubleGrating):
    """An Asymmetric Double Grating is defined by three materials (iso
    or aniso), three duty cycles, a pitch and a thickness.

    Inside the pitch there are two rect of width dc1*pitch of mat1 and
    dc2*pitch of mat2, separated by dcM*pitch mat3 (between mat1 e
    mat2, not between mat2 and mat1!).
    """

    def __init__(self, mat1, mat2, mat3, dc1, dc2, dcM, pitch, thickness):
        SymmetricDoubleGrating.__init__(
            self, mat1, mat2, mat3, dc1, dc2, pitch, thickness)
        self.dcM = dcM

    def getEPSFourierCoeffs(self, wl, n, anisotropic=True):
        """Return the Fourier coefficients of eps and eps**-1, orders [-n,n]."""
        nood = 2 * n + 1
        hmax = nood - 1
        if not anisotropic:
            # isotropic
            rix1 = self.mat1.n(wl)
            rix2 = self.mat2.n(wl)
            rix3 = self.mat3.n(wl)
            f1 = self.dc1
            f2 = self.dc2
            fM = self.dcM
            h = numpy.arange(-hmax, hmax + 1)
            N = len(h)
            A = -N * (f1 + fM) / 2.
            B = N * (f2 + fM) / 2.
            EPS = (
                rix3 ** 2 * (h == 0) + (rix1 ** 2 - rix3 ** 2) * f1 *
                numpy.sinc(h * f1) * numpy.exp(2j * numpy.pi * h / N * A) +
                (rix2 ** 2 - rix3 ** 2) * f2 *
                numpy.sinc(h * f2) * numpy.exp(2j * numpy.pi * h / N * B)
            )
            EPS1 = (
                rix3 ** -2 * (h == 0) + (rix1 ** -2 - rix3 ** -2) * f1 *
                numpy.sinc(h * f1) * numpy.exp(2j * numpy.pi * h / N * A) +
                (rix2 ** -2 - rix3 ** -2) * f2 * numpy.sinc(h * f2) *
                numpy.exp(2j * numpy.pi * h / N * B)
            )
            return EPS, EPS1
        else:
            # anisotropic
            EPS = numpy.zeros((3, 3, 2 * hmax + 1), dtype=complex)
            EPS1 = numpy.zeros_like(EPS)
            eps1 = numpy.squeeze(
                self.mat1.epsilonTensor(wl)) / EMpy_gpu.constants.eps0
            eps2 = numpy.squeeze(
                self.mat2.epsilonTensor(wl)) / EMpy_gpu.constants.eps0
            eps3 = numpy.squeeze(
                self.mat3.epsilonTensor(wl)) / EMpy_gpu.constants.eps0
            f1 = self.dc1
            f2 = self.dc2
            fM = self.dcM
            h = numpy.arange(-hmax, hmax + 1)
            N = len(h)
            A = -N * (f1 + fM) / 2.
            B = N * (f2 + fM) / 2.
            for ih, hh in enumerate(h):
                EPS[:, :, ih] = (
                    (eps1 - eps3) * f1 * numpy.sinc(hh * f1) *
                    numpy.exp(2j * numpy.pi * hh / N * A) +
                    (eps2 - eps3) * f2 * numpy.sinc(hh * f2) *
                    numpy.exp(2j * numpy.pi * hh / N * B) +
                    eps3 * (hh == 0)
                )
                EPS1[:, :, ih] = (
                    (scipy.linalg.inv(eps1) - scipy.linalg.inv(eps3)) * f1 *
                    numpy.sinc(hh * f1) *
                    numpy.exp(2j * numpy.pi * hh / N * A) +
                    (scipy.linalg.inv(eps2) - scipy.linalg.inv(eps3)) * f2 *
                    numpy.sinc(hh * f2) *
                    numpy.exp(2j * numpy.pi * hh / N * B) +
                    scipy.linalg.inv(eps3) * (hh == 0)
                )
            return EPS, EPS1

    def capacitance(self, area=1., wl=0):
        """Capacitance = eps0 * eps_r * area / thickness."""

        if self.isIsotropic():
            eps = EMpy_gpu.constants.eps0 * numpy.real(
                self.mat1.n(wl) ** 2 * self.dc1 + self.mat2.n(wl) ** 2 *
                self.dc2 + self.mat3.n(wl) ** 2 * (1 - self.dc1 - self.dc2))
        else:
            eps1 = self.mat1.epsilonTensor(wl)[2, 2, 0]
            eps2 = self.mat2.epsilonTensor(wl)[2, 2, 0]
            eps3 = self.mat3.epsilonTensor(wl)[2, 2, 0]
            eps = numpy.real(
                eps1 * self.dc1 + eps2 * self.dc2 +
                eps3 * (1 - self.dc1 - self.dc2))

        return eps * area / self.thickness

    def __str__(self):
        """Return the description of a binary grating."""
        return ("(%s, %s, %s), dc1: %g, dc2: %g, dcM: %g, "
                "pitch: %g, thickness: %g") % (
            self.mat1, self.mat2, self.mat3, self.dc1, self.dc2,
            self.dcM, self.pitch, self.thickness)


class LiquidCrystalCell(object):
    """Liquid Crystal Cell.

    A liquid crystal cell is determined by a liquid crystal, a voltage
    applied to it, a total thickness, an anchoring thickness.  The
    liquid crystal molecules are anchored to the cell with a given
    pretilt angle (that, at zero volts, is constant throughout all the LC cell).
    The cell is decomposed in nlayers homogeneous layers. The LC
    characteristics in each layer is either read from file or deduced
    by the LC physical parameters solving a boundary value problem
    (bvp).

    Inspiration from:
    U{http://www.ee.ucl.ac.uk/~rjames/modelling/constant-order/oned/}.

    @ivar lc: Liquid Crystal.
    @ivar voltage: voltage applied.
    @ivar t_tot: total thickness.
    @ivar t_anchoring: anchoring thickness.
    @ivar pretilt: LC angle pretilt.
    @ivar totaltwist: LC angle total twist between the anchoring layers.
    @ivar nlayers: number of layers to subdived the cell.
    @ivar data_file: file with the angles for voltages applid to the cell.

    """

    def __init__(self, lc, voltage, t_tot, t_anchoring, pretilt=0,
                 totaltwist=0, nlayers=100, data_file=None):

        self.lc = lc
        self.t_tot = t_tot
        self.t_anchoring = t_anchoring
        self.pretilt = pretilt
        self.totaltwist = totaltwist
        self.nlayers = nlayers
        self.data_file = data_file
        # thicknesses of internal layers
        tlc_internal = (self.t_tot - 2. * self.t_anchoring) / \
            (self.nlayers - 2.) * numpy.ones(self.nlayers - 2)
        # thicknesses of layers
        self.tlc = numpy.r_[self.t_anchoring, tlc_internal, self.t_anchoring]
        # internal sample points
        lhs = numpy.r_[0, numpy.cumsum(tlc_internal)]
        # normalized sample points: at the center of internal layers, plus the
        # boundaries (i.e. the anchoring layers)
        self.normalized_sample_points = numpy.r_[
            0, (lhs[1:] + lhs[:-1]) / 2. / (self.t_tot - 2 * self.t_anchoring),
            1]
        tmp = numpy.r_[0, numpy.cumsum(self.tlc)]
        self.sample_points = .5 * (tmp[1:] + tmp[:-1])
        # finally, apply voltage
        self.voltage = voltage

    def getvoltage(self):
        return self.__voltage

    def setvoltage(self, v):
        self.__voltage = v
        if self.data_file is not None:
            self.__angles = self._get_angles_from_file()
        else:
            self.__angles = self._get_angles_from_bvp()

    voltage = property(fget=getvoltage, fset=setvoltage)

    def getangles(self):
        return self.__angles

    angles = property(fget=getangles)

    def __ode_3k(self, z, f):
        """Inspiration from:
        U{http://www.ee.ucl.ac.uk/~rjames/modelling/constant-order/oned/}."""

        # ------------------------------------------------------------
        # minimise Oseen Frank free energy and solve Laplace equation
        # ------------------------------------------------------------
        # [f(1..6)] = [theta theta' phi phi' u u']

        theta2, dtheta2dz, phi2, dphi2dz, u2, du2dz = f
        K11 = self.lc.K11
        K22 = self.lc.K22
        K33 = self.lc.K33
        q0 = self.lc.q0
        epslow = self.lc.epslow
        deleps = self.lc.deleps

        e0 = EMpy_gpu.constants.eps0
        K1122 = K11 - K22
        K3322 = K33 - K22
        costheta1 = numpy.cos(theta2)
        sintheta1 = numpy.sin(theta2)
        ezz = e0 * (epslow + deleps * sintheta1 ** 2)

        # maple generated (see lc3k.mws)
        ddtheta2dz = costheta1 * sintheta1 * (
            K1122 * dtheta2dz ** 2 +
            2 * K3322 * costheta1 ** 2 * dphi2dz ** 2 -
            K3322 * dtheta2dz ** 2 -
            K22 * dphi2dz ** 2 -
            e0 * deleps * du2dz ** 2 +
            2 * q0 * K22 * dphi2dz -
            K3322 * dphi2dz ** 2
        ) / (
            K1122 * costheta1 ** 2 -
            K3322 * costheta1 ** 2 +
            K22 + K3322
        )
        ddphi2dz = 2 * sintheta1 * dtheta2dz * (
            2 * K3322 * costheta1 ** 2 * dphi2dz -
            K22 * dphi2dz +
            q0 * K22 -
            K3322 * dphi2dz
        ) / costheta1 / (K3322 * costheta1 ** 2 - K22 - K3322)

        ddu2dz = -2 * e0 * deleps * sintheta1 * \
            costheta1 * dtheta2dz * du2dz / ezz

        return numpy.array([ddtheta2dz, ddphi2dz, ddu2dz])

    def __bc_nosplay(self, f):
        """Inspiration from:
        U{http://www.ee.ucl.ac.uk/~rjames/modelling/constant-order/oned/}."""

        theta2, dtheta2dz, phi2, dphi2dz, u2, du2dz = f
        return numpy.array([theta2[0] - self.pretilt,
                            phi2[1] - 0,
                            u2[2] - 0,
                            theta2[3] - self.pretilt,
                            phi2[4] - self.totaltwist,
                            u2[5] - self.voltage])

    def __ic_nosplay(self, z):
        """Inspiration from:
        U{http://www.ee.ucl.ac.uk/~rjames/modelling/constant-order/oned/}."""

        self.maxtilt = 90 * numpy.pi / 180 - self.pretilt
        init = numpy.array([self.pretilt + self.maxtilt * 4 * z * (1 - z),
                            self.maxtilt * 4 * (1 - 2 * z),
                            self.totaltwist * z,
                            self.totaltwist * numpy.ones_like(z),
                            self.voltage * z,
                            self.voltage * numpy.ones_like(z)])

        return init, self.__ode_3k(z, init)

    def __apply_tension(self):
        """Inspiration from:
        U{http://www.ee.ucl.ac.uk/~rjames/modelling/constant-order/oned/}."""

        try:
            from scikits.bvp1lg import colnew
        except ImportError:
            warning("bvp module not found.")
            raise

        boundary_points = numpy.array([0, 0, 0, 1, 1, 1])
        tol = 1e-6 * numpy.ones_like(boundary_points)
        degrees = numpy.array([2, 2, 2])

        solution = colnew.solve(
            boundary_points, degrees, self.__ode_3k, self.__bc_nosplay,
            is_linear=False, initial_guess=self.__ic_nosplay,
            tolerances=tol, vectorized=True,
            maximum_mesh_size=1000)

        self.bvp_solution = solution

    def get_parameters(self, z=None):
        """Inspiration from:
        U{http://www.ee.ucl.ac.uk/~rjames/modelling/constant-order/oned/}."""

        if z is None:
            z = self.bvp_solution.mesh

        data = self.bvp_solution(z)
        theta = EMpy_gpu.utils.rad2deg(numpy.pi / 2. - data[:, 0])
        phi = EMpy_gpu.utils.rad2deg(data[:, 2])
        u = data[:, 4]

        return z, theta, phi, u

    def _get_angles_from_file(self):

        # interpolate data file
        data = numpy.loadtxt(self.data_file)
        data_x = numpy.linspace(0, 1, data.shape[0] - 1)
        data_y = data[0, :]
        x = self.normalized_sample_points
        y = [self.voltage]
        angles = interp2(x, y, data_x, data_y, data[1:, :])

        return angles.squeeze()

    def _get_angles_from_bvp(self):

        # solve bvp
        self.__apply_tension()
        z_ = self.normalized_sample_points
        z, theta, phi, u = self.get_parameters(z_)

        return theta

    def createMultilayer(self):
        """Split the cell in nlayers homogeneous layers."""

        m = []
        for a, t in zip(EMpy_gpu.utils.deg2rad(self.angles), self.tlc):
            epsT = EMpy_gpu.materials.EpsilonTensor(
                epsilon_tensor_const=EMpy_gpu.utils.euler_rotate(
                    numpy.diag([self.lc.nE,
                                self.lc.nO,
                                self.lc.nO]) ** 2,
                    0., numpy.pi / 2., numpy.pi / 2. - a) * EMpy_gpu.constants.eps0,
                epsilon_tensor_known={
                    0: EMpy_gpu.utils.euler_rotate(
                        numpy.diag([self.lc.nE_electrical,
                                    self.lc.nO_electrical,
                                    self.lc.nO_electrical]) ** 2,
                        0., numpy.pi / 2.,
                        numpy.pi / 2. - a) * EMpy_gpu.constants.eps0,
                }
            )
            m.append(
                Layer(EMpy_gpu.materials.AnisotropicMaterial(
                    'LC', epsilon_tensor=epsT), t))

        return Multilayer(m)

    def capacitance(self, area=1., wl=0):
        """Capacitance = eps0 * eps_r * area / thickness."""
        return self.createMultilayer().capacitance(area, wl)

    @staticmethod
    def isIsotropic():
        """Return False."""
        return False

    def __str__(self):
        """Return the description of a LiquidCrystal."""
        return ("datafile: %s, voltage: %g, t_tot: %g, "
                "t_anchoring: %g, (nO, nE) = (%g, %g)") % (
                    self.data_file, self.voltage, self.t_tot, self.t_anchoring,
                    self.lc.nO, self.lc.nE)


class Multilayer(object):

    """A Multilayer is a list of layers with some more methods."""

    def __init__(self, data=None):
        """Initialize the data list."""
        if data is None:
            data = []
        self.data = data[:]

    def __delitem__(self, i):
        """Delete an item from list."""
        del self.data[i]

    def __getitem__(self, i):
        """Get an item of the list of layers."""
        return self.data[i]

    def __getslice__(self, i, j):
        """Get a Multilayer from a slice of layers."""
        return Multilayer(self.data[i:j])

    def __len__(self):
        """Return the number of layers."""
        return len(self.data)

    def __setitem__(self, i, item):
        """Set an item of the list of layers."""
        self.data[i] = item

    def __setslice__(self, i, j, other):
        """Set a slice of layers."""
        self.data[i:j] = other

    def append(self, item):
        """Append a layer to the layers list."""
        self.data.append(item)

    def extend(self, other):
        """Extend the layers list with other layers."""
        self.data.extend(other)

    def insert(self, i, item):
        """Insert a new layer in the layers list at the position i."""
        self.data.insert(i, item)

    def remove(self, item):
        """Remove item from layers list."""
        self.data.remove(item)

    def pop(self, i=-1):
        return self.data.pop(i)

    def isIsotropic(self):
        """Return True if all the layers of the multilayers are
        isotropic, False otherwise."""
        return numpy.asarray([m.isIsotropic() for m in self.data]).all()

    def simplify(self):
        """Return a new flatten Multilayer, with expanded LiquidCrystalCells."""
        # make a tmp list, copy of self, to work with
        tmp = self.data[:]
        # expand the liquid crystals
        for il, l in enumerate(tmp):
            if isinstance(l, LiquidCrystalCell):
                tmp[il] = l.createMultilayer()

        # flatten the tmp list
        def helper(multilayer):
            """Recurse to flatten all the nested Multilayers."""
            ret = []
            for layer in multilayer:
                if not isinstance(layer, Multilayer):
                    ret.append(layer)
                else:
                    ret.extend(helper(layer[:]))
            return ret

        return Multilayer(helper(tmp))

    def capacitance(self, area=1., wl=0):
        """Capacitance = eps0 * eps_r * area / thickness."""

        m = self.simplify()
        ctot_1 = 0.
        for l in m:
            if numpy.isfinite(l.thickness):
                ctot_1 += 1. / l.capacitance(area, wl)
        return 1. / ctot_1

    def __str__(self):
        """Return a description of the Multilayer."""
        if self.__len__() == 0:
            list_str = "<emtpy>"
        else:
            list_str = '\n'.join([
                '%d: %s' % (il, l.__str__()) for il, l in enumerate(self.data)
            ])
        return 'Multilayer\n----------\n' + list_str


class Slice(Multilayer):

    def __init__(self, width, *argv):
        Multilayer.__init__(self, *argv)
        self.width = width

    def heights(self):
        return numpy.array([l.thickness for l in self])

    def ys(self):
        return numpy.r_[0., self.heights().cumsum()]

    def height(self):
        return self.heights().sum()

    def find_layer(self, y):
        l = numpy.where(self.ys() <= y)[0]
        if len(l) > 0:
            return self[min(l[-1], len(self) - 1)]
        else:
            return self[0]

    def plot(self, x0, x1, nmin, nmax, wl=1.55e-6):
        try:
            import pylab
        except ImportError:
            warning('no pylab installed')
            return
        y0 = 0
        # ytot = sum([l.thickness for l in self])
        for l in self:
            y1 = y0 + l.thickness
            n = l.mat.n(wl)
            r = 1. - (1. * (n - nmin) / (nmax - nmin))
            pylab.fill(
                [x0, x1, x1, x0], [y0, y0, y1, y1], ec='yellow', fc=(r, r, r),
                alpha=.5)
            y0 = y1
        pylab.axis('image')

    def __str__(self):
        return 'width = %e\n%s' % (self.width, Multilayer.__str__(self))


class CrossSection(list):

    def __str__(self):
        return '\n'.join('%s' % s for s in self)

    def widths(self):
        return numpy.array([s.width for s in self])

    def xs(self):
        return numpy.r_[0., self.widths().cumsum()]

    def ys(self):
        tmp = numpy.concatenate([s.ys() for s in self])
        # get rid of numerical errors
        tmp = numpy.round(tmp * 1e10) * 1e-10
        return numpy.unique(tmp)

    def width(self):
        return self.widths().sum()

    def grid(self, nx_per_region, ny_per_region):

        xs = self.xs()
        ys = self.ys()

        nxregions = len(xs) - 1
        nyregions = len(ys) - 1

        if numpy.isscalar(nx_per_region):
            nx = (nx_per_region,) * nxregions
        elif len(nx_per_region) != nxregions:
            raise ValueError('wrong nx_per_region dim')
        else:
            nx = nx_per_region

        if numpy.isscalar(ny_per_region):
            ny = (ny_per_region,) * nyregions
        elif len(ny_per_region) != nyregions:
            raise ValueError('wrong ny_per_region dim')
        else:
            ny = ny_per_region

        X = []
        x0 = xs[0]
        for x, n in zip(xs[1:], nx):
            X.append(numpy.linspace(x0, x, n + 1)[:-1])
            x0 = x
        X = numpy.concatenate(X)
        X = numpy.r_[X, x0]

        Y = []
        y0 = ys[0]
        for y, n in zip(ys[1:], ny):
            Y.append(numpy.linspace(y0, y, n + 1)[:-1])
            y0 = y
        Y = numpy.concatenate(Y)
        Y = numpy.r_[Y, y0]

        return X, Y

    def find_slice(self, x):
        s = numpy.where(self.xs() <= x)[0]
        if len(s) > 0:
            return self[min(s[-1], len(self) - 1)]
        else:
            return self[0]

    def _epsfunc(self, x, y, wl):
        if numpy.isscalar(x) and numpy.isscalar(y):
            return self.find_slice(x).find_layer(y).mat.n(wl) ** 2
        else:
            raise ValueError('only scalars, please!')

    def epsfunc(self, x, y, wl):
        eps = numpy.ones((len(x), len(y)), dtype=complex)
        for ix, xx in enumerate(x):
            for iy, yy in enumerate(y):
                eps[ix, iy] = self._epsfunc(xx, yy, wl)
        return eps

    def plot(self, wl=1.55e-6):
        try:
            import pylab
        except ImportError:
            warning('no pylab installed')
            return
        x0 = 0
        ns = [[l.mat.n(wl) for l in s] for s in self]
        nmax = max(max(ns))
        nmin = min(min(ns))
        for s in self:
            x1 = x0 + s.width
            s.plot(x0, x1, nmin, nmax, wl=wl)
            x0 = x1
        pylab.axis('image')


class Peak(object):

    def __init__(self, x, y, idx, x0, y0, xFWHM_1, xFWHM_2):
        self.x = x
        self.y = y
        self.idx = idx
        self.x0 = x0
        self.y0 = y0
        self.xFWHM_1 = xFWHM_1
        self.xFWHM_2 = xFWHM_2
        self.FWHM = numpy.abs(xFWHM_2 - xFWHM_1)

    def __str__(self):
        return '(%g, %g) [%d, (%g, %g)] FWHM = %s' % (
            self.x, self.y, self.idx, self.x0, self.y0, self.FWHM)


def deg2rad(x):
    """Convert from deg to rad."""
    return x / 180. * numpy.pi


def rad2deg(x):
    """Convert from rad to deg."""
    return x / numpy.pi * 180.


def norm(x):
    """Return the norm of a 1D vector."""
    return numpy.sqrt(numpy.vdot(x, x))


def normalize(x):
    """Return a normalized 1D vector."""
    return x / norm(x)


def euler_rotate(X, phi, theta, psi):
    """Euler rotate.

    Rotate the matrix X by the angles phi, theta, psi.

    INPUT
    X = 2d numpy.array.
    phi, theta, psi = rotation angles.

    OUTPUT
    Rotated matrix = 2d numpy.array.

    NOTE
    see http://mathworld.wolfram.com/EulerAngles.html
    """

    A = numpy.array([
        [numpy.cos(psi) * numpy.cos(phi) -
         numpy.cos(theta) * numpy.sin(phi) * numpy.sin(psi),
         -numpy.sin(psi) * numpy.cos(phi) -
         numpy.cos(theta) * numpy.sin(phi) * numpy.cos(psi),
         numpy.sin(theta) * numpy.sin(phi)],
        [numpy.cos(psi) * numpy.sin(phi) +
         numpy.cos(theta) * numpy.cos(phi) * numpy.sin(psi),
         -numpy.sin(psi) * numpy.sin(phi) +
         numpy.cos(theta) * numpy.cos(phi) * numpy.cos(psi),
         -numpy.sin(theta) * numpy.cos(phi)],
        [numpy.sin(theta) * numpy.sin(psi),
         numpy.sin(theta) * numpy.cos(psi), numpy.cos(theta)]
    ])
    return numpy.dot(A, numpy.dot(X, scipy.linalg.inv(A)))


def snell(theta_inc, n):
    """Snell law.

    INPUT
    theta_inc = angle of incidence.
    n = 1D numpy.array of refractive indices.

    OUTPUT
    theta = 1D numpy.array.
    """

    theta = numpy.zeros_like(n)
    theta[0] = theta_inc
    for i in range(1, n.size):
        theta[i] = numpy.arcsin(n[i - 1] / n[i] * numpy.sin(theta[i - 1]))
    return theta


def group_delay_and_dispersion(wls, y):
    """Compute group delay and dispersion.

    INPUT
    wls = wavelengths (ndarray).
    y = function (ndarray).

    OUTPUT
    phi = phase of function in rad.
    tau = group delay in ps.
    Dpsnm = dispersion in ps/nm.

    NOTE
    wls and y must have the same shape.
    phi has the same shape as wls.
    tau has wls.shape - (..., 1)
    Dpsnm has wls.shape - (..., 2)
    """

    # transform the input in ndarrays
    wls = numpy.asarray(wls)
    y = numpy.asarray(y)

    # check for good input
    if wls.shape != y.shape:
        raise ValueError('wls and y must have the same shape.')

    f = EMpy_gpu.constants.c / wls

    df = numpy.diff(f)
    toPSNM = 1E12 / 1E9
    cnmps = EMpy_gpu.constants.c / toPSNM

    # phase
    phi = numpy.unwrap(4. * numpy.angle(y)) / 4.

    # group delay
    tau = -.5 / numpy.pi * numpy.diff(phi) / df * 1E12

    # dispersion in ps/nm
    Dpsnm = -.5 / numpy.pi / cnmps * \
        f[1:-1] ** 2 * numpy.diff(phi, 2) / df[0:-1] ** 2

    return phi, tau, Dpsnm


def rix2losses(n, wl):
    """Return real(n), imag(n), alpha, alpha_cm1, alpha_dBcm1, given a
    complex refractive index.  Power goes as: P = P0 exp(-alpha*z)."""
    nr = numpy.real(n)
    ni = numpy.imag(n)
    alpha = 4 * numpy.pi * ni / wl
    alpha_cm1 = alpha / 100.
    alpha_dBcm1 = 10 * numpy.log10(numpy.exp(1)) * alpha_cm1
    return nr, ni, alpha, alpha_cm1, alpha_dBcm1


def loss_cm2rix(n_real, alpha_cm1, wl):
    """Return complex refractive index, given real index (n_real), absorption coefficient (alpha_cm1) in cm^-1, and wavelength (wl) in meters.
    Do not pass more than one argument as array, will return erroneous result."""
    ni = 100 * alpha_cm1 * wl /(numpy.pi * 4)
    return (n_real - 1j*ni)


def loss_m2rix(n_real, alpha_m1, wl):
    """Return complex refractive index, given real index (n_real), absorption coefficient (alpha_m1) in m^-1, and wavelength (wl) in meters.
    Do not pass more than one argument as array, will return erroneous result."""
    ni = alpha_m1 * wl /(numpy.pi * 4)
    return (n_real - 1j*ni)


def loss_dBcm2rix(n_real, alpha_dBcm1, wl):
    """Return complex refractive index, given real index (n_real), absorption coefficient (alpha_dBcm1) in dB/cm, and wavelength (wl) in meters.
    Do not pass more than one argument as array, will return erroneous result."""
    ni = 10 * alpha_dBcm1 * wl / (numpy.log10(numpy.exp(1)) * 4 * numpy.pi)
    return (n_real - 1j*ni)


def wl2f(wl0, dwl):
    """Convert a central wavelength and an interval to frequency."""
    wl1 = wl0 - dwl / 2.
    wl2 = wl0 + dwl / 2.
    f1 = EMpy_gpu.constants.c / wl2
    f2 = EMpy_gpu.constants.c / wl1
    f0 = (f1 + f2) / 2.
    df = (f2 - f1)
    return f0, df


def f2wl(f0, df):
    """Convert a central frequency and an interval to wavelength."""
    return wl2f(f0, df)


def find_peaks(x, y, threshold=1e-6):
    # find peaks' candidates
    dy = numpy.diff(y)
    ddy = numpy.diff(numpy.sign(dy))
    # idxs = numpy.where(ddy < 0)[0] + 1
    idxs = numpy.where(ddy < 0)

    if len(idxs) == 0:
        # there is only 1 min in f, so the max is on either boundary
        # get the max and set FWHM = 0
        idx = numpy.argmax(y)
        p = Peak(x[idx], y[idx], idx, x[idx], y[idx], x[idx], x[idx])
        # return a list of one element
        return [p]

    # refine search with splines
    tck = scipy.interpolate.splrep(x, y)
    # look for zero derivative
    absdy = lambda x_: numpy.abs(scipy.interpolate.splev(x_, tck, der=1))

    peaks = []
    for idx in idxs:

        # look around the candidate
        xtol = (x.max() - x.min()) * 1e-6
        xopt = scipy.optimize.fminbound(
            absdy, x[idx - 1], x[idx + 1], xtol=xtol, disp=False)
        yopt = scipy.interpolate.splev(xopt, tck)

        if yopt > threshold * y.max():

            # FWHM
            tckFWHM = scipy.interpolate.splrep(x, y - 0.5 * yopt)
            roots = scipy.interpolate.sproot(tckFWHM)

            idxFWHM = numpy.searchsorted(roots, xopt)
            if idxFWHM <= 0:
                xFWHM_1 = x[0]
            else:
                xFWHM_1 = roots[idxFWHM - 1]
            if idxFWHM >= len(roots):
                xFWHM_2 = x[-1]
            else:
                xFWHM_2 = roots[idxFWHM]

            p = Peak(xopt, yopt, idx, x[idx], y[idx], xFWHM_1, xFWHM_2)
            peaks.append(p)

    def cmp_y(x_, y_):
        # to sort in descending order
        if x_.y == y_.y:
            return 0
        if x_.y > y_.y:
            return -1
        return 1

    peaks.sort(cmp=cmp_y)

    return peaks


def cond(M):
    """Return the condition number of the 2D array M."""
    svdv = scipy.linalg.svdvals(M)
    return svdv.max() / svdv.min()


def interp2(x, y, xp, yp, fp):
    """Interpolate a 2D complex array.

    :rtype : numpy.array
    """
    f1r = numpy.zeros((len(xp), len(y)))
    f1i = numpy.zeros((len(xp), len(y)))
    for ixp in range(len(xp)):
        f1r[ixp, :] = numpy.interp(y, yp, numpy.real(fp[ixp, :]))
        f1i[ixp, :] = numpy.interp(y, yp, numpy.imag(fp[ixp, :]))
    fr = numpy.zeros((len(x), len(y)))
    fi = numpy.zeros((len(x), len(y)))
    for iy in range(len(y)):
        fr[:, iy] = numpy.interp(x, xp, f1r[:, iy])
        fi[:, iy] = numpy.interp(x, xp, f1i[:, iy])
    return fr + 1j * fi


def trapz2(f, x=None, y=None, dx=1.0, dy=1.0):
    """Double integrate."""
    return numpy.trapz(numpy.trapz(f, x=y, dx=dy), x=x, dx=dx)


def centered1d(x):
    return (x[1:] + x[:-1]) / 2.


def centered2d(x):
    return (x[1:, 1:] + x[1:, :-1] + x[:-1, 1:] + x[:-1, :-1]) / 4.


def blackbody(f, T):
    return 2 * EMpy_gpu.constants.h * f ** 3 / (EMpy_gpu.constants.c ** 2) * 1. / (
        numpy.exp(EMpy_gpu.constants.h * f / (EMpy_gpu.constants.k * T)) - 1)


def warning(s):
    """Print a warning on the stdout.

    :param s: warning message
    :type s: str
    :rtype : str
    """
    print('WARNING --- {}'.format(s))


class ProgressBar(object):

    """ Creates a text-based progress bar. Call the object with the `print'
    command to see the progress bar, which looks something like this:

    [=======>        22%                  ]

    You may specify the progress bar's width, min and max values on init.
    """

    def __init__(self, minValue=0, maxValue=100, totalWidth=80):
        self.progBar = "[]"  # This holds the progress bar string
        self.min = minValue
        self.max = maxValue
        self.span = maxValue - minValue
        self.width = totalWidth
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.amount = 0  # When amount == max, we are 100% done
        self.updateAmount(0)  # Build progress bar string

    def updateAmount(self, newAmount=0):
        """ Update the progress bar with the new amount (with min and max
        values set at initialization; if it is over or under, it takes the
        min or max value as a default. """
        if newAmount < self.min:
            newAmount = self.min
        if newAmount > self.max:
            newAmount = self.max
        self.amount = newAmount

        # Figure out the new percent done, round to an integer
        diffFromMin = float(self.amount - self.min)
        percentDone = (diffFromMin / float(self.span)) * 100.0
        percentDone = int(round(percentDone))

        # Figure out how many hash bars the percentage should be
        allFull = self.width - 2 - 18
        numHashes = (percentDone / 100.0) * allFull
        numHashes = int(round(numHashes))

        # Build a progress bar with an arrow of equal signs; special cases for
        # empty and full
        if numHashes == 0:
            self.progBar = '[>%s]' % (' ' * (allFull - 1))
        elif numHashes == allFull:
            self.progBar = '[%s]' % ('=' * allFull)
        else:
            self.progBar = '[%s>%s]' % ('=' * (numHashes - 1),
                                        ' ' * (allFull - numHashes))

        # figure out where to put the percentage, roughly centered
        percentPlace = (len(self.progBar) / 2) - len(str(percentDone))
        percentString = ' ' + str(percentDone) + '% '

        elapsed_time = time.time() - self.start_time

        # slice the percentage into the bar
        self.progBar = ''.join([self.progBar[0:percentPlace], percentString,
                                self.progBar[
                                    percentPlace + len(percentString):],
                                ])

        if percentDone > 0:
            self.progBar += ' %6ds / %6ds' % (
                int(elapsed_time), int(elapsed_time * (100. / percentDone - 1)))

    def update(self, value, every=1):
        """ Updates the amount, and writes to stdout. Prints a carriage return
        first, so it will overwrite the current line in stdout."""
        if value % every == 0 or value >= self.max:
            print('\r', end=' ')
            self.updateAmount(value)
            sys.stdout.write(self.progBar)
            sys.stdout.flush()
