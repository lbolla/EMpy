# pylint: disable=W0622,R0903,R0902,R0913,W0633

"""Functions and objects to manipulate materials.

A material is an object with a refractive index function.

"""
from builtins import str
from builtins import object
from functools import partial

import numpy
from scipy.integrate import quad
from EMpy_gpu.constants import eps0

__author__ = 'Lorenzo Bolla'


class Material(object):

    """Generic class to handle materials.

    This class is intended to be subclassed to obtain isotropic and
    anisotropic materials.
    """

    def __init__(self, name=''):
        """Set material name."""
        self.name = name


class RefractiveIndex(object):

    """Refractive Index.

    Unaware of temperature.

    Parameters
    ----------
    Provide ONE of the following named-arguments:

    n0_const : float
        A single-value of refractive index, to be used regardless of
        the wavelength requested.
        For example:
            >>> n0_const = 1.448 for SiO2

    n0_poly : list/tuple
        Use a polynomial rix dispersion function: provide the
        polynomial coefficients to be evaluated by numpy.polyval.
         # sets the refractive index function as n = 9(wl**3) +
         # 5(wl**2) + 3(wl) + 1
         For example:
            >>> n0_poly = (9,5,3,1)

    n0_smcoeffs (Sellmeier coefficients): 6-element list/tuple
        Set the rix dispersion function to the 6-parameter Sellmeier
        function as so:
            n(wls) =  sqrt(1. +
            B1 * wls ** 2 / (wls ** 2 - C1) +
            B2 * wls ** 2 / (wls ** 2 - C2) +
            B3 * wls ** 2 / (wls ** 2 - C3))
        For example:
            >>> n0_smcoeffs = [B1, B2, B3, C1, C2, C3]    # six values total

    n0_func : function
        Provide an arbitrary function to return the refractive index
        versus wavelength.
        For example:
            >>> def SiN_func(wl):
            >>>     x = wl * 1e6   # convert to microns
            >>>     return 1.887 + 0.01929/x**2 + 1.6662e-4/x**4  # Cauchy func
            >>> SiN_rix = RefractiveIndex(n0_func=SiN_func)
        or
            >>> SiN_rix = RefractiveIndex(
                    n0_func=lambda wl: 1.887 + 0.01929/(wl*1e6)**2 +
                                       1.6662e-4/(wl*1e6)**4)

        Your function should return a `numpy.array`, since it will be
        passed a `numpy.array` of the wavelengths requested.  This
        conversion to `array` happens automatically if your function
        does math on the wavelength.

    n0_known : dictionary
        Use if RefractiveIndex will only be evaluated at a specific set of `wls`.
        n0_known should be a dictionary of `key:value` pairs
        corresponding to `wavelength:rix`, for example:
            >>> n0_known = { 1500e-9:1.445, 1550e-9:1.446, 1600e-9:1.447 }

    """

    def __init__(self, n0_const=None, n0_poly=None, n0_smcoeffs=None,
                 n0_known=None, n0_func=None):

        if n0_const is not None:
            self.get_rix = partial(self.__from_const, n0_const)

        elif n0_poly is not None:
            self.get_rix = partial(self.__from_poly, n0_poly)

        elif n0_smcoeffs is not None:
            self.get_rix = partial(self.__from_sellmeier, n0_smcoeffs)

        elif n0_func is not None:
            self.get_rix = partial(self.__from_function, n0_func)

        elif n0_known is not None:
            self.get_rix = partial(self.__from_known, n0_known)

        else:
            raise ValueError('Please provide at least one parameter.')

    @staticmethod
    def __from_const(n0, wls):
        wls = numpy.atleast_1d(wls)
        return n0 * numpy.ones_like(wls)

    @staticmethod
    def __from_poly(n0_poly, wls):
        wls = numpy.atleast_1d(wls)
        return numpy.polyval(n0_poly, wls) * numpy.ones_like(wls)

    @staticmethod
    def __from_sellmeier(n0_smcoeffs, wls):
        wls = numpy.atleast_1d(wls)
        B1, B2, B3, C1, C2, C3 = n0_smcoeffs
        return numpy.sqrt(
            1. +
            B1 * wls ** 2 / (wls ** 2 - C1) +
            B2 * wls ** 2 / (wls ** 2 - C2) +
            B3 * wls ** 2 / (wls ** 2 - C3)
        ) * numpy.ones_like(wls)

    @staticmethod
    def __from_function(n0_func, wls):
        # ensure wls is array
        wls = numpy.atleast_1d(wls)
        return n0_func(wls) * numpy.ones_like(wls)

    @staticmethod
    def __from_known(n0_known, wls):
        wls = numpy.atleast_1d(wls)
        # Note: we should interpolate...
        return numpy.array([n0_known.get(wl, 0) for wl in wls])

    def __call__(self, wls):
        return self.get_rix(wls)


class ThermalOpticCoefficient(object):

    """Thermal Optic Coefficient."""

    def __init__(self, data=None, T0=300):
        self.__data = data
        self.T0 = T0

    def TOC(self, T):
        if self.__data is not None:
            return numpy.polyval(self.__data, T)
        else:
            return 0.0

    def __call__(self, T):
        return self.TOC(T)

    def dnT(self, T):
        """Integrate the TOC to get the rix variation."""
        return quad(self.TOC, self.T0, T)[0]


class IsotropicMaterial(Material):

    """Subclasses Material to describe isotropic materials.

    Frequency dispersion and thermic aware.
    In all the member functions, wls must be an ndarray.
    """

    def __init__(self, name='', n0=RefractiveIndex(n0_const=1.),
                 toc=ThermalOpticCoefficient()):
        """Set name, default temperature, refractive index and TOC
        (thermal optic coefficient)."""
        Material.__init__(self, name)
        self.n0 = n0
        self.toc = toc

    def n(self, wls, T=None):
        """Return the refractive index at T as a [1 x wls] array."""
        if T is None:
            T = self.toc.T0
        return self.n0(wls) + self.toc.dnT(T)

    def epsilon(self, wls, T=None):
        """Return the epsilon at T as a [1 x wls] array."""
        if T is None:
            T = self.toc.T0
        return self.n(wls, T) ** 2 * eps0

    def epsilonTensor(self, wls, T=None):
        """Return the epsilon at T as a [3 x 3 x wls] array."""
        if T is None:
            T = self.toc.T0
        tmp = numpy.eye(3)
        return tmp[:, :, numpy.newaxis] * self.epsilon(wls, T)

    @staticmethod
    def isIsotropic():
        """Return True, because the material is isotropic."""
        return True

    def __str__(self):
        """Return material name."""
        return self.name + ', isotropic'


class EpsilonTensor(object):

    def __init__(self, epsilon_tensor_const=eps0 * numpy.eye(3),
                 epsilon_tensor_known=None):
        if epsilon_tensor_known is None:
            epsilon_tensor_known = {}
        self.epsilon_tensor_const = epsilon_tensor_const
        self.epsilon_tensor_known = epsilon_tensor_known

    def __call__(self, wls):
        """Return the epsilon tensor as a [3 x 3 x wls.size] matrix."""
        wls = numpy.atleast_1d(wls)
        if wls.size == 1:
            if wls.item() in self.epsilon_tensor_known:
                return self.epsilon_tensor_known[
                    wls.item()][:, :, numpy.newaxis]
        return self.epsilon_tensor_const[
            :, :, numpy.newaxis] * numpy.ones_like(wls)


class AnisotropicMaterial(Material):

    """Subclass Material to describe anisotropic materials.

    No frequency dispersion nor thermic aware.
    In all the member functions, wls must be an ndarray.

    """

    def __init__(self, name='', epsilon_tensor=EpsilonTensor()):
        """Set name and default epsilon tensor."""
        Material.__init__(self, name)
        self.epsilonTensor = epsilon_tensor

    @staticmethod
    def isIsotropic():
        """Return False, because the material is anisotropic."""
        return False

    def __str__(self):
        """Return material name."""
        return self.name + ', anisotropic'

# Vacuum
Vacuum = IsotropicMaterial(name='Vacuum')

# Air
Air = IsotropicMaterial(name='Air')

# Silicon
Si = IsotropicMaterial(
    name='Silicon',
    n0=RefractiveIndex(
        n0_poly=(0.076006e12, -0.31547e6, 3.783)),
    toc=ThermalOpticCoefficient((-1.49e-10, 3.47e-7, 9.48e-5)))

# SiO2
SiO2 = IsotropicMaterial(
    name='Silica',
    n0=RefractiveIndex(n0_const=1.446),
    toc=ThermalOpticCoefficient((1.1e-4,)))

# BK7 glass (see http://en.wikipedia.org/wiki/Sellmeier_equation)
BK7 = IsotropicMaterial(
    name='Borosilicate crown glass',
    n0=RefractiveIndex(n0_smcoeffs=(
        1.03961212, 2.31792344e-1, 1.01046945, 6.00069867e-15, 2.00179144e-14,
        1.03560653e-10)))


class LiquidCrystal(Material):

    """Liquid Crystal.

    A liquid crystal is determined by it ordinary and extraordinary
    refractive indices, its elastic tensor and its chirality.
    Inspiration here:
    U{http://www.ee.ucl.ac.uk/~rjames/modelling/constant-order/oned/}.

    @ivar name: Liquid Crystal name.
    @ivar nO: Ordinary refractive index.
    @ivar nE: Extraordinary refractive index.
    @ivar K11: Elastic tensor, first component.
    @ivar K22: Elastic tensor, second component.
    @ivar K33: Elastic tensor, third component.
    @ivar q0: Chirality.

    """

    def __init__(self, name='', nO=1., nE=1., nO_electrical=1.,
                 nE_electrical=1., K11=0.0, K22=0.0, K33=0.0, q0=0.0):
        """Set name, the refractive indices, the elastic constants and
        the chirality."""

        Material.__init__(self, name)
        self.nO = nO
        self.nE = nE
        self.nO_electrical = nO_electrical
        self.nE_electrical = nE_electrical
        self.K11 = K11
        self.K22 = K22
        self.K33 = K33
        self.q0 = q0
        self.epslow = self.nO_electrical ** 2
        self.deleps = self.nE_electrical ** 2 - self.epslow


def get_10400_000_100(conc000):
    """Return a LiquidCrystal made of conc% 000 and (100-conc)% 100."""

    conc = [0, 100]
    epsO_electrical = [3.38, 3.28]
    epsE_electrical = [5.567, 5.867]
    epsO = [1.47551 ** 2, 1.46922 ** 2]
    epsE = [1.61300 ** 2, 1.57016 ** 2]

    K11 = 13.5e-12  # elastic constant [N] (splay)
    K22 = 6.5e-12  # elastic constant [N] (twist)
    K33 = 20e-12  # elastic constant [N] (bend)
    q0 = 0  # chirality 2*pi/pitch

    nO_electrical_ = numpy.interp(conc000, conc, epsO_electrical) ** .5
    nE_electrical_ = numpy.interp(conc000, conc, epsE_electrical) ** .5
    nO_ = numpy.interp(conc000, conc, epsO) ** .5
    nE_ = numpy.interp(conc000, conc, epsE) ** .5

    return LiquidCrystal(
        '10400_000_100_' + str(conc000) + '_' + str(100 - conc000),
        nO_, nE_, nO_electrical_, nE_electrical_,
        K11, K22, K33, q0)
