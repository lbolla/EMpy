# pylint: disable=C0302,W0622,W1001,R0902,R0903,W0201,R0913,R0914,C0325,W0221

"""Set of objects and functions to describe the behaviour of some
widely known devices and build new ones.

A set of widely known devices is described, by its transfer function,
transfer matrix and chain matrix:

    - Coupler.
    - Line.
    - Mach-Zehnder Interferometer.
    - All-Pass Ring Resonator.
    - Single Ring Resonator.
    - N-Rings Resonator.
    - Tunable Triple-Coupled Ring Resonator.
    - Tunable Coupled Ring Triple-Coupled Ring Resonator.
    - Tunable Coupled Triple-Coupled Ring Resonator.
    - Straight Waveguide.
    - Etalon.

By combining the transfer matrices and chain matrices of known
devices, new ones can be studied.

@see: U{Schwelb, "Transmission, Group Delay, and Dispersion in
      Single-Ring Optical Resonators and Add/Drop Filters - A Tutorial
      Overview", JLT 22(5), p. 1380, 2004
      <http://jlt.osa.org/abstract.cfm?id=80103>}
@see: U{Barbarossa, "Theoretical Analysis of Triple-Coupler Ring-Based
    Optical Guided-Wave Resonator", JLT 13(2), p. 148, 1995
    <http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?tp=&arnumber=365200&
    isnumber=8367>}

"""
from builtins import zip
from builtins import range
from builtins import object

__author__ = 'Lorenzo Bolla'

import numpy
import EMpy_gpu.utils
from functools import reduce


class DeviceMatrix(object):

    """Device Matrix.

    Base class for TransferMatrix and ChainMatrix.

    Notes
    =====

        - Data is stored as numpy.matrix.
        - It is a virtual class.

    @ivar data: Device matrix data, automatically converted into a
                numpy.matrix on assignment.

    """

    def __init__(self, data):
        self.data = data

    def getdata(self):
        return self.__data

    def setdata(self, value):
        self.__data = numpy.asmatrix(value)

    __data = None
    data = property(fget=getdata, fset=setdata, doc="data")

    def compose(self, A):
        """Compose two device matrices.

        @raise NotImplementedError: always, because DeviceMatrix is a
            virtual class.

        """
        raise NotImplementedError('DeviceMatrix is virtual class.')

    def __str__(self):
        return self.data.__str__()

    __repr__ = __str__


class TransferMatrix(DeviceMatrix):

    """Transfer Matrix.

    Notation::

        b1 = h11 h12 * a1 --> b12 = H a12
        b2   h21 h22   a2

    """

    def __init__(self, data):
        DeviceMatrix.__init__(self, data)

    def to_transfer(self):
        """Convert to transfer matrix (nothing to do).

        @rtype: L{TransferMatrix}

        """

        return self

    def to_chain(self):
        """Convert to chain matrix.

        Notation::

            b1 = h11 a1 + h12 a2
                --> a1 = (b2 - h22 a2) / h21 = 1 / h21 b2 - h22 / h21 a2
            b2 = h21 a1 + h22 a2
                --> b1 = h11 / h21 * (b2 - h22 a2) + h12 a2
                       = h11 / h21 b2 + (h12 * h21 - h11 * h22) / h12 a2

        @rtype: L{ChainMatrix}

        """
        return ChainMatrix([
            [1. / self.data[1, 0],
             -self.data[1, 1] / self.data[1, 0]],
            [self.data[0, 0] / self.data[1, 0],
             (self.data[0, 1] * self.data[1, 0] -
              self.data[0, 0] * self.data[1, 1]) / self.data[1, 0]]
        ])

    def compose(self, M):
        """Compose the transfer matrix with another device matrix.

        Notation::

            b12 = H1 a12 and d12 = H2 c12, but b12 == c12,

        therefore::

            d12 = H2*H1 a12.

        Notes
        =====

        First, convert the second matrix to transfer.

        @param M: DeviceMatrix (will be converted to TransferMatrix).
        @rtype: L{TransferMatrix}

        """

        return TransferMatrix(M.to_transfer().data * self.data)


class ChainMatrix(DeviceMatrix):

    """Chain Matrix.

    Notation::

        a1 = g11 g12 * b2 --> ab1 = G ba2
        b1   g21 g22   a2

    """

    def __init__(self, *args, **kwargs):
        DeviceMatrix.__init__(self, *args, **kwargs)

    def to_transfer(self):
        """Convert to transfer matrix.

        Notation::

            a1 = g11 b2 + g12 a2
            b2 = 1 / g11 a1 - g12 / g11 a2 = 1 / g11 a1 - g12 / g11 a2
            b1 = g21 b2 + g22 a2
            b1 = g21 / g11 a1 + (g22 - g12 * g21 / g11) a2
               = g21 / g11 a1 + (g11 g22 - g12 g21) / g11 a2

        @rtype: L{TransferMatrix}

        """
        return TransferMatrix([
            [self.data[1, 0] / self.data[0, 0],
             (self.data[0, 0] * self.data[1, 1] -
              self.data[0, 1] * self.data[1, 0]) / self.data[0, 0]],
            [1. / self.data[0, 0],
             -self.data[0, 1] / self.data[0, 0]]])

    def to_chain(self):
        """Convert to chain matrix (nothing to do).

        @rtype: L{ChainMatrix}

        """

        return self

    def compose(self, M):
        """Compose the chain matrix with another device matrix.

        Notation::

            ab1 = G1 ba2 and ab3 = G2 ba4, but ba2 == ab3,

        therefore::

            ab1 = G1*G2 ba4.

        Notes
        =====

        First, convert the second matrix to chain.

        @param M: L{DeviceMatrix} (will be converted to L{ChainMatrix}).
        @rtype: L{ChainMatrix}

        """

        return ChainMatrix(self.data * M.to_chain().data)


def composeTM(A, B):
    """Compose two transfer matrices.

    Notes
    =====

    First convert them to TM, to be sure.

    @param A: First L{TransferMatrix}.
    @param B: Second L{TransferMatrix}.
    @rtype: L{TransferMatrix}

    """

    return A.to_transfer().compose(B)


def composeCM(A, B):
    """Compose two chain matrices.

    Notes
    =====

    First convert them to CM, to be sure.

    @param A: First L{ChainMatrix}.
    @param B: Second L{ChainMatrix}.
    @rtype: L{ChainMatrix}

    """

    return A.to_chain().compose(B)


def composeTMlist(TMlist):
    """Compose n transfer matrices in a list.

    @param TMlist: List of L{TransferMatrix}.
    @rtype: L{TransferMatrix}

    """

    return reduce(composeTM, TMlist)


def composeCMlist(CMlist):
    """Compose n chain matrices in a list.

    @param CMlist: List of L{ChainMatrix}.
    @rtype: L{ChainMatrix}

    """

    return reduce(composeCM, CMlist)


class Device(object):

    """Generic device.

    Notes
    =====

    Parent virtual class for all the other devices.

    """

    def sanity_check(self):
        raise NotImplementedError()

    def solve(self, *args, **kwargs):
        raise NotImplementedError()


class Coupler(Device):

    """Coupler.

    Examples
    ========

    Here is how to compute the THRU and DROP of a 3dB coupler, with
    wavelength-dependent losses:

    >>> wls = numpy.linspace(1.52e-6, 1.57e-6, 1000)
    >>> k = .5 ** .5
    >>> q = numpy.linspace(.9, .95, 1000)
    >>> coupler = EMpy.devices.Coupler(wls, k, q).solve()

    @ivar wl: Wavelength.
    @type wl: numpy.ndarray
    @ivar K: Field coupling coefficient.
    @type K: numpy.ndarray
    @ivar q: Field coupling loss.
    @type q: numpy.ndarray
    @ivar THRU: also called "bar".
    @type THRU: numpy.ndarray
    @ivar DROP: also called "cross".
    @type DROP: numpy.ndarray

    """

    def __init__(self, wl, K, q=1.):
        """Set the parameters of the coupler and check them."""

        self.wl = wl
        one = numpy.ones_like(wl)
        # make K,q of the same shape as wl (scalar or ndarray)
        self.K = K * one
        self.q = q * one
        self.sanity_check()

    def sanity_check(self):
        """Check for good input.

        @raise ValueError: if input numpy.ndarrays have wrong shape.

        """

        if (numpy.isscalar(self.q) or
            numpy.isscalar(self.K)) and not numpy.isscalar(self.wl):
            raise ValueError('wl is not a scalar but K and q are.')
        if (not numpy.isscalar(self.q) or
            not numpy.isscalar(self.K)) and numpy.isscalar(self.wl):
            raise ValueError('wl is a scalar but K and q are not.')
        if not (numpy.asarray(self.wl).shape ==
                numpy.asarray(self.K).shape ==
                numpy.asarray(self.q).shape):
            raise ValueError('wl, K and q must have the same shape')

    def solve(self):
        """Compute the THRU and DROP."""

        Kbar = numpy.sqrt(1. - self.K ** 2)
        self.THRU = self.q * Kbar
        self.DROP = self.q * 1j * self.K

        return self

    def TM(self, wl=None):
        """Return the L{TransferMatrix} of the L{Coupler}, for a given wl.

        Notation::

            TM = q * [ t jk] , with t = (1 - k**2)**.5
                     [jk  t]

        @param wl: Wavelength at which computing the TM. If None, use
            the only wl provided at construction time.
        @type wl: scalar

        @return: L{TransferMatrix}.
        @rtype: L{TransferMatrix}

        @raise ValueError: (if wl=None and self.wl is not scalar) or
            wl is not scalar.

        """

        if wl is None:
            if not numpy.isscalar(self.wl):
                raise ValueError('which wl?')
            K = self.K
            q = self.q
        elif numpy.isscalar(wl):
            K = numpy.interp(numpy.atleast_1d(wl), numpy.atleast_1d(
                self.wl), numpy.atleast_1d(self.K)).item()
            q = numpy.interp(numpy.atleast_1d(wl), numpy.atleast_1d(
                self.wl), numpy.atleast_1d(self.q)).item()
        else:
            raise ValueError('wl must be scalar')

        Kbar = numpy.sqrt(1 - K ** 2)
        return TransferMatrix([[q * Kbar, q * 1j * K], [q * 1j * K, q * Kbar]])

    def CM(self, wl=None):
        """Return the L{ChainMatrix} of the coupler.

        Notes
        =====

        First build its L{TransferMatrix}, then convert it to L{ChainMatrix}.

        @param wl: Wavelength at which computing the L{ChainMatrix}.

        @return: L{ChainMatrix}.
        @rtype: L{ChainMatrix}

        """

        return self.TM(wl).to_chain()


class Line(Device):

    """Line.

    Notes
    =====

    The Line device is a four ports device, hence is made of two
    two-ports lines.

    Examples
    ========

    Here is how to create a L{Line}, where the first line is a 2.pi.R1
    long L{SWG} and the second has zero length.

    >>> wls = numpy.linspace(1.52e-6, 1.57e-6, 1000)
    >>> SWG = EMpy.devices.SWG(400, 220, 125).solve(wls)
    >>> R1 = 5.664e-6
    >>> line = EMpy.devices.Line(wls, SWG.neff, 2 * numpy.pi * R1, SWG.neff, 0)

    @ivar wl: Wavelength.
    @type wl: numpy.ndarray
    @ivar neff1: Effective index of the first line.
    @type neff1: numpy.ndarray
    @ivar l1: Length of the first line.
    @type l1: scalar
    @ivar neff2: Effective index of the second line.
    @type neff2: numpy.ndarray
    @ivar l2: Length of the second line.
    @type l2: scalar
    @ivar THRU: output of the first line.
    @type THRU: numpy.ndarray
    @ivar DROP: output of the second line.
    @type DROP: numpy.ndarray

    """

    def __init__(self, wl, neff1, l1, neff2, l2):
        """Set the parameters of the line and check them."""

        self.wl = wl
        one = numpy.ones_like(wl)
        self.neff1 = neff1 * one
        self.l1 = l1
        self.neff2 = neff2 * one
        self.l2 = l2
        self.sanity_check()

    def sanity_check(self):
        """Check for good input.

        @raise ValueError: if l1 and l2 are not scalar.

        """

        if not (numpy.isscalar(self.l1) and numpy.isscalar(self.l2)):
            raise ValueError('lengths must be scalars')

    def solve(self):
        """Compute the THRU and DROP."""

        beta1 = 2 * numpy.pi * self.neff1 / self.wl
        beta2 = 2 * numpy.pi * self.neff2 / self.wl
        self.THRU = numpy.exp(-1j * beta1 * self.l1)
        self.DROP = numpy.exp(-1j * beta2 * self.l2)

        return self

    def TM(self, wl=None):
        """Return the L{TransferMatrix} of the L{Line}, for a given wl.

        Notation::

            TM = [exp(-1j * beta1 * l1) 0                    ]
                 [0                     exp(-1j * beta2 * l2)]

        @param wl: Wavelength at which computing the TM. If None, use
            the only wl provided at construction time.
        @type wl: scalar

        @return: L{TransferMatrix}.
        @rtype: L{TransferMatrix}

        @raise ValueError: (if wl=None and self.wl is not scalar) or
            wl is not scalar.

        """

        if wl is None:
            if not numpy.isscalar(self.wl):
                raise ValueError('which wl?')
            neff1 = self.neff1
            neff2 = self.neff2
            wl = self.wl
        elif numpy.isscalar(wl):
            neff1 = numpy.interp(numpy.atleast_1d(wl), numpy.atleast_1d(
                self.wl), numpy.atleast_1d(self.neff1)).item()
            neff2 = numpy.interp(numpy.atleast_1d(wl), numpy.atleast_1d(
                self.wl), numpy.atleast_1d(self.neff2)).item()
        else:
            raise ValueError('wl must be scalar')

        beta1 = 2 * numpy.pi * neff1 / wl
        beta2 = 2 * numpy.pi * neff2 / wl
        return TransferMatrix([[numpy.exp(-1j * beta1 * self.l1), 0],
                               [0, numpy.exp(-1j * beta2 * self.l2)]])

    def CM(self, wl=None):
        """Return the L{ChainMatrix} of the line.

        Notes
        =====

        First build its L{TransferMatrix}, then convert it to L{ChainMatrix}.

        @return: L{ChainMatrix}.
        @rtype: L{ChainMatrix}

        """

        return self.TM(wl).to_chain()


class MZ(Device):

    """Mach-Zehnder Interferometer (MZI).

    A MZI is made of two L{Coupler}s connected by a L{Line}.
    A different phase shift on either arms of the L{Line} determines
    the amount of power in bar (THRU) or cross (DROP).

    Examples
    ========

    Here is how to build a MZI:

    >>> wls = numpy.linspace(1.52e-6, 1.57e-6, 1000)
    >>> SWG = EMpy.devices.SWG(400, 220, 125).solve(wls)
    >>> R1 = 5.664e-6
    >>> Kbus2ringeq = .5
    >>> coupler = EMpy.devices.Coupler(wls, Kbus2ringeq**.5)
    >>> line = EMpy.devices.Line(wls, SWG.neff, 2 * numpy.pi * R1, SWG.neff, 0)
    >>> mz = EMpy.devices.MZ(coupler, line, coupler).solve()

    @ivar coupler1: First L{Coupler}
    @type coupler1: L{Coupler}
    @ivar line: L{Line} connecting the two L{Coupler}s.
    @type line: L{Line}
    @ivar coupler2: Second L{Coupler}
    @type coupler2: L{Coupler}

    """

    def __init__(self, coupler1, line, coupler2):
        """Set the parameters of the MZ and check them."""

        self.wl = coupler1.wl
        self.coupler1 = coupler1
        self.coupler2 = coupler2
        self.line = line
        self.sanity_check()

    def sanity_check(self):
        """Check for good input.

        @raise ValueError: if coupler1 or coupler2 or line have incompatible wl.

        """

        self.coupler1.sanity_check()
        self.coupler2.sanity_check()
        self.line.sanity_check()
        if not (numpy.alltrue(self.coupler1.wl == self.line.wl) and
                numpy.alltrue(self.coupler1.wl == self.coupler2.wl)):
            raise ValueError('incompatible wl')

    def solve(self):
        """Compute the THRU and DROP."""

        K1 = self.coupler1.K
        K2 = self.coupler2.K
        K1bar = numpy.sqrt(1. - K1 ** 2)
        K2bar = numpy.sqrt(1. - K2 ** 2)
        q1 = self.coupler1.q
        q2 = self.coupler2.q

        beta1 = 2. * numpy.pi * self.line.neff1 / self.wl
        beta2 = 2. * numpy.pi * self.line.neff2 / self.wl

        dephasing1 = numpy.exp(-1j * beta1 * self.line.l1)
        dephasing2 = numpy.exp(-1j * beta2 * self.line.l2)

        self.THRU = q1 * q2 * \
            (K1bar * K2bar * dephasing1 - K1 * K2 * dephasing2)
        self.DROP = q1 * q2 * 1j * \
            (K1 * K2bar * dephasing2 + K2 * K1bar * dephasing1)

        return self

    def TM(self, wl=None):
        """Return the transfer matrix of the MZ.

        Notes
        =====

        Compose the TM of the single devices which compose the MZ.

        @return: L{TransferMatrix}.
        @rtype: L{TransferMatrix}

        """

        return composeTMlist([
            self.coupler1.TM(wl), self.line.TM(wl), self.coupler2.TM(wl)])

    def CM(self, wl=None):
        """Return the chain matrix of the MZ.

        Notes
        =====

        First build its L{TransferMatrix}, then convert it to L{ChainMatrix}.

        @return: L{ChainMatrix}.
        @rtype: L{ChainMatrix}

        """

        return self.TM(wl).to_chain()


class APRR(Device):

    """All-Pass Ring Resonator.

    An APRR is made of a L{Coupler} with a feedback line (the ring).

    Examples
    ========

    >>> wls = numpy.linspace(1.5e-6, 1.6e-6, 1000)
    >>> K = numpy.sqrt(0.08)
    >>> q = 1.
    >>> l = 2*numpy.pi*5e-6
    >>> SWG = EMpy.devices.SWG(488, 220, 25).solve(wls)
    >>> coupler = EMpy.devices.Coupler(wls, K, q)
    >>> APRR = EMpy.devices.APRR(coupler, SWG.neff, l).solve()

    @ivar coupler: L{Coupler}.
    @type coupler: L{Coupler}
    @ivar neff: Effective index of the feedback line.
    @type neff: numpy.ndarray
    @ivar l: Length of the feedback line (perimeter of the ring).
    @type l: scalar
    @ivar THRU: THRU of the all-pass.
    @type THRU: numpy.ndarray

    """

    def __init__(self, coupler, neff, l):
        """Set the parameters of the all-pass and check them."""

        self.wl = coupler.wl
        self.coupler = coupler
        self.neff = neff
        self.l = l
        self.sanity_check()

    def sanity_check(self):
        """Check for good input.

        @raise ValueError: if l is not a scalar or wl and neff have
        different shapes.

        """

        self.coupler.sanity_check()
        if not numpy.isscalar(self.wl):
            if not numpy.isscalar(self.neff):
                if self.wl.shape != self.neff.shape:
                    raise ValueError('wl and neff must have the same shape.')
        if not numpy.isscalar(self.l):
            raise ValueError('l must be scalar.')

    def solve(self):
        """Compute the THRU."""

        K = self.coupler.K
        q = self.coupler.q

        Kbar = numpy.sqrt(1. - K ** 2)
        beta = 2. * numpy.pi * self.neff / self.wl
        t = numpy.exp(-1j * beta * self.l)
        self.THRU = q * (Kbar - q * t) / (1. - q * Kbar * t)

        return self


class SRR(Device):

    """Single Ring Resonator.

    A SRR is made of the parallel of a L{Coupler} a feedback line (the
    ring) and another L{Coupler}.

    Examples
    ========

        >>> wls = numpy.linspace(1.5e-6, 1.6e-6, 1000)
        >>> coupler1 = EMpy.devices.Coupler(wls, 0.08**.5)
        >>> coupler2 = EMpy.devices.Coupler(wls, 0.08**.5)
        >>> R = 5e-6
        >>> l1 = numpy.pi * R
        >>> l2 = numpy.pi * R
        >>> SWG = EMpy.devices.SWG(488, 220, 25).solve(wls)
        >>> SRR = EMpy.devices.SRR(coupler1, coupler2, SWG.neff, l1, l2).solve()

    @ivar coupler1: Bus-to-Ring L{Coupler}.
    @type coupler1: L{Coupler}
    @ivar coupler2: Ring-to-Bus L{Coupler}.
    @type coupler2: L{Coupler}
    @ivar neff: Effective index of the feedback line.
    @type neff: numpy.ndarray
    @ivar l1: Length of the feedback line (perimeter of the ring) from
        coupler1 to coupler2.
    @type l1: scalar
    @ivar l2: Length of the feedback line (perimeter of the ring) from
        coupler2 to coupler1.
    @type l2: scalar
    @ivar THRU: THRU of the filter.
    @type THRU: numpy.ndarray
    @ivar DROP: DROP of the filter.
    @type DROP: numpy.ndarray

    """

    def __init__(self, coupler1, coupler2, neff, l1, l2):
        """Set the parameters of the device and check them."""

        self.wl = coupler1.wl
        self.coupler1 = coupler1
        self.coupler2 = coupler2
        self.neff = neff
        self.l1 = l1
        self.l2 = l2
        self.line1 = Line(self.wl, neff, 0., neff, l1)
        self.line2 = Line(self.wl, neff, 0., neff, l2)
        self.sanity_check()

    def sanity_check(self):
        """Check for good input.

        @raise ValueError: if wl are incompatible.

        """

        self.coupler1.sanity_check()
        self.coupler2.sanity_check()
        self.line1.sanity_check()
        self.line2.sanity_check()
        if not (numpy.alltrue(self.coupler1.wl == self.coupler2.wl) and
                numpy.alltrue(self.coupler1.wl == self.line1.wl) and
                numpy.alltrue(self.coupler1.wl == self.line2.wl)):
            raise ValueError('incompatible wl')

    def solve(self):
        """Compute the THRU and the DROP."""

        K1 = self.coupler1.K
        K2 = self.coupler2.K
        K1bar = numpy.sqrt(1. - K1 ** 2)
        K2bar = numpy.sqrt(1. - K2 ** 2)
        q1 = self.coupler1.q
        q2 = self.coupler2.q
        l1 = self.line1.l2
        l2 = self.line2.l2

        beta = 2. * numpy.pi * self.neff / self.wl

        t1 = numpy.exp(-1j * beta * l1)
        t2 = numpy.exp(-1j * beta * l2)

        denom = 1. - q1 * q2 * K1bar * K2bar * t1 * t2
        self.DROP = -q1 * q2 * K1 * K2 * t2 / denom
        self.THRU = q1 * (K1bar - q1 * q2 * K2bar * t1 * t2) / denom

        return self

    def TM(self, wl=None):
        """Return the L{TransferMatrix} of the SRR.

        Notes
        =====

        First build its L{ChainMatrix}, then convert it to L{TransferMatrix}.

        @return: L{TransferMatrix}.
        @rtype: L{TransferMatrix}

        """

        return self.CM(wl).to_transfer()

    def CM(self, wl=None):
        """Return the L{ChainMatrix} of the SRR.

        Notes
        =====

        Compose the CM of the single devices which compose the SRR.

        @return: L{ChainMatrix}.
        @rtype: L{ChainMatrix}

        """

        return composeCM(
            composeTMlist([
                self.line1.TM(wl), self.coupler1.TM(wl), self.line2.TM(wl)]),
            self.coupler2.CM(wl))


class NRR(Device):

    """N-Rings Resonator.

    A NRR is made of the parallel of two lines connected by N ring resonators.

    Examples
    ========

    >>> wls = numpy.linspace(1.5e-6, 1.6e-6, 1000)
    >>> from EMpy.devices import Coupler
    >>> couplers = [Coupler(wls, 0.08**.5), Coupler(wls, 0.003**.5),
        Coupler(wls, 0.08**.5)]
    >>> q = [1., 1., 1.]
    >>> R = 5e-6
    >>> l1s = [numpy.pi * R, numpy.pi * R]
    >>> l2s = [numpy.pi * R, numpy.pi * R]
    >>> SWG = EMpy.devices.SWG(488, 220, 25).solve(wls)
    >>> neffs = [SWG.neff, SWG.neff]
    >>> NRR = EMpy.devices.NRR(couplers, neffs, l1s, l2s).solve()

    @ivar Ks: List of L{Coupler}s.
    @type Ks: list
    @ivar neffs: List of rings' effective indexes.
    @type neffs: numpy.ndarray
    @ivar l1s: List of rings' arcs connecting one coupler to the next.
    @type l1s: list
    @ivar l2s: List of rings' arcs connecting one coupler to the previous.
    @type l2s: list
    @ivar THRU: THRU of the filter.
    @type THRU: numpy.ndarray
    @ivar DROP: DROP of the filter.
    @type DROP: numpy.ndarray

    """

    def __init__(self, Ks, neffs, l1s, l2s):
        """Set the parameters of the device and check them."""

        self.wl = Ks[0].wl
        self.Ks = Ks
        self.neffs = neffs
        self.l1s = l1s
        self.l2s = l2s
        self.sanity_check()

    def sanity_check(self):
        """Check for good input.

        @raise ValueError: if wl or the number of couplers are incompatible.

        """

        for K in self.Ks:
            K.sanity_check()
            if not (numpy.alltrue(self.wl == K.wl)):
                raise ValueError('incompatible wl')
        if not (len(self.Ks) - 1 ==
                len(self.neffs) ==
                len(self.l1s) ==
                len(self.l2s)):
            raise ValueError(
                'number of couplers and number of rings do not match.')

    def solve(self):
        """Compute the THRU and the DROP.

        Notes
        =====

        No analytic expression is available, therefore build the TM of
        the device for each wl.

        """

        self.THRU = numpy.zeros_like(self.wl).astype('complex')
        self.DROP = numpy.zeros_like(self.wl).astype('complex')
        for iwl, wl in enumerate(self.wl):
            H = self.TM(wl)
            self.THRU[iwl] = H.data[0, 0]
            self.DROP[iwl] = H.data[0, 1]

        return self

    def TM(self, wl=None):
        """Return the L{TransferMatrix} of the NRR.

        Notes
        =====

        Compose the TM of the single devices which compose the NRR.

        @return: L{TransferMatrix}.
        @rtype: L{TransferMatrix}

        """

        return self.CM(wl).to_transfer()

    def CM(self, wl=None):
        """Return the L{TransferMatrix} of the NRR.

        Notes
        =====

        First build its L{TransferMatrix}, then convert it to L{ChainMatrix}.

        @return: L{ChainMatrix}.
        @rtype: L{ChainMatrix}

        """

        if wl is None:
            wl = self.wl
        Hs = []
        for K, neff, l1, l2 in zip(self.Ks, self.neffs, self.l1s, self.l2s):
            Hs.append(composeTMlist([
                Line(self.wl, neff, 0., neff, l1).TM(wl), K.TM(wl),
                Line(self.wl, neff, 0., neff, l2).TM(wl)]))
        return composeCM(composeTMlist(Hs), self.Ks[-1].CM(wl))


class T_TCRR(Device):

    """Tunable Triple-Coupled Ring Resonator.

    A T_TCRR is made of a L{MZ}, with a feedback line.

    """

    def __init__(self, neff, K, q, l, coupling=None):
        """Set the parameters of the device.

        INPUT
        neff = effective index (can be complex).
        K = field coupling coeffs.
        q = coupler losses.
        l = ring arcs.
        coupling = 'optimum' to change K2, K3 and l.
        """

        self.neff = numpy.asarray(neff)
        self.K = numpy.asarray(K)
        self.q = q
        self.l = numpy.asarray(l)
        self.coupling = coupling
        self.sanity_check()

    def sanity_check(self):
        """Check for good input."""
        if self.K.shape != (3,):
            raise ValueError('K must be a 1D-array with 3 elements.')
        if not numpy.isscalar(self.q):
            raise ValueError('q must be a scalar.')
        if self.l.shape != (4,):
            raise ValueError('l must be a 1D-array with 4 elements.')

    def solve(self, wls):
        """Compute the THRU and the DROP."""
        neff, K, q, l, coupling = (
            self.neff, self.K, self.q, self.l, self.coupling)

        # transform the input in ndarrays
        wls = numpy.asarray(wls)
        if wls.shape != neff.shape:
            raise ValueError('wrong wls and neff shape.')

        if coupling == "optimum":
            # optimum: l2 = 2l1, l3 = l4
            circ = l[0] + l[2] + l[3]
            l = circ * numpy.array([.5, 1.0, .25, .25])
            # optimum: K1==K2, K3=2*K1*K1bar
            K[1] = K[0]
            K[2] = 2. * K[0] * numpy.sqrt(1 - K[0] ** 2)

        Kbar = numpy.sqrt(1. - K ** 2)
        beta = 2. * numpy.pi * neff / wls

        t1 = numpy.exp(-1j * beta * l[0])
        t2 = numpy.exp(-1j * beta * l[1])
        t3 = numpy.exp(-1j * beta * l[2])
        t4 = numpy.exp(-1j * beta * l[3])

        denom = q ** 3 * Kbar[2] * t3 * t4 * \
            (Kbar[0] * Kbar[1] * t1 - K[0] * K[1] * t2) - 1.
        self.DROP = q ** 3 * \
            K[2] * t3 * (K[0] * Kbar[1] * t1 + Kbar[0] * K[1] * t2) / denom
        self.THRU = (
            q ** 2 * (K[0] * K[1] * t1 - Kbar[0] * Kbar[1] * t2) +
            q ** 5 * Kbar[2] * t1 * t2 * t3 * t4) / denom

        return self


class T_CRTCRR(Device):

    """Tunable Coupled Ring Triple-Coupled Ring Resonator.

    NOTE
    see 1997_IEEE Proc El_Barbarossa_Novel
    """

    def __init__(self, neff, K, q, l, coupling=None):
        """Set the parameters of the device.

        INPUT
        neff = effective index (can be complex).
        K = field coupling coeffs.
        q = coupler losses.
        l = ring arcs.
        coupling = 'optimum' <not implemented yet>.
        """

        self.neff = numpy.asarray(neff)
        self.K = numpy.asarray(K)
        self.q = q
        self.l = numpy.asarray(l)
        self.coupling = coupling
        self.sanity_check()

    def sanity_check(self):
        """Check for good input."""
        if not (self.K.shape == self.l.shape == (3,)):
            raise ValueError('K and l must be 1D-arrays with 3 elements.')
        if not numpy.isscalar(self.q):
            raise ValueError('q must be a scalar.')

    def solve(self, wls):
        """Compute the THRU and the DROP."""
        neff, K, q, l, _ = (
            self.neff, self.K, self.q, self.l, self.coupling)

        # transform the input in ndarrays
        wls = numpy.asarray(wls)
        if wls.shape != neff.shape:
            raise ValueError('wrong wls and neff shape.')

        Kbar = numpy.sqrt(1. - K ** 2)
        beta = 2. * numpy.pi * neff / wls

        t1 = numpy.exp(-1j * beta * l[0])
        t2 = numpy.exp(-1j * beta * l[1])
        t3 = numpy.exp(-1j * beta * l[2])
        t31 = t3 / t1

        N1 = Kbar[0] ** 2 - K[0] ** 2 * t31
        N1bar = K[0] ** 2 - Kbar[0] ** 2 * t31

        denom = 1 - q ** 3 * t1 ** 2 * N1 * \
            Kbar[2] - q ** 2 * Kbar[1] * t2 ** 2 * \
            (Kbar[2] - q ** 3 * t1 ** 2 * N1)
        self.DROP = 1j * q ** 4 * \
            K[0] * K[1] * K[2] * Kbar[0] * t1 ** 1.5 * t2 * (1 + t31) / denom
        self.THRU = q ** 2 * t1 * (
            N1bar - q ** 2 * Kbar[1] * Kbar[2] * t2 ** 2 * N1bar +
            q ** 3 * Kbar[2] * t1 * t3 -
            q ** 5 * Kbar[1] * t1 * t2 ** 2 * t3) / denom

        return self


class T_CTCRR(Device):

    """Tunable Coupled Triple Coupled Ring Resonator.

    NOTE
    see 1997_IEEE Proc El_Barbarossa_Novel
    """

    def __init__(self, neff, K, q, l, coupling=None):
        """Set the parameters of the device.

        INPUT
        neff = effective index (can be complex).
        K = field coupling coeffs.
        q = coupler losses.
        l = ring arcs.
        coupling = 'optimum'.
        """

        self.neff = numpy.asarray(neff)
        self.K = numpy.asarray(K)
        self.q = q
        self.l = numpy.asarray(l)
        self.coupling = coupling
        self.sanity_check()

    def sanity_check(self):
        """Check for good input."""
        if self.K.shape != (3,):
            raise ValueError('K must be a 1D-array with 3 elements.')
        if self.l.shape != (4,):
            raise ValueError('l must be a 1D-array with 4 elements.')
        if not numpy.isscalar(self.q):
            raise ValueError('q must be a scalar.')

    def solve(self, wls):
        """Compute the THRU and the DROP."""
        neff, K, q, l, coupling = (
            self.neff, self.K, self.q, self.l, self.coupling)

        # transform the input in ndarrays
        wls = numpy.asarray(wls)
        if wls.shape != neff.shape:
            raise ValueError('wrong wls and neff shape.')

        if coupling is not None:
            # optimum: l2 = 2l1, l3 = l4
            l[1] = l[0]  # OKKIO: check me!
            l[2] = 2 * l[0]
            l[3] = 2 * l[1]
            # optimum: see article
            K[2] = numpy.absolute(
                2 * K[0] * K[1] * numpy.sqrt(1 - K[0] ** 2) *
                numpy.sqrt(1 - K[1] ** 2) /
                (K[0] ** 2 + K[1] ** 2 - 2 * K[0] ** 2 * K[1] ** 2 - 1))

        Kbar = numpy.sqrt(1. - K ** 2)
        beta = 2. * numpy.pi * neff / wls

        t1 = numpy.exp(-1j * beta * l[0])
        t2 = numpy.exp(-1j * beta * l[1])
        t3 = numpy.exp(-1j * beta * l[2])
        t4 = numpy.exp(-1j * beta * l[3])
        t31 = t3 / t1
        t42 = t4 / t2

        N1 = Kbar[0] ** 2 - K[0] ** 2 * t31
        N1bar = K[0] ** 2 - Kbar[0] ** 2 * t31
        N2 = Kbar[1] ** 2 - K[1] ** 2 * t42

        denom = 1. - q ** 3 * t1 ** 2 * N1 * \
            Kbar[2] - q ** 3 * t2 ** 2 * N2 * (Kbar[2] - q ** 3 * t1 ** 2 * N1)
        self.DROP = (
            1j * q ** 5 * K[0] * K[1] * K[2] * Kbar[0] * Kbar[1] * t1 ** 1.5 *
            t2 ** 1.5 * (1 + t31) * (1 + t42) / denom)
        self.THRU = (
            q ** 5 * t1 ** 2 * t3 *
            (q ** 3 * t2 ** 2 * N2 - Kbar[2]) + q ** 2 * t1 * N1bar *
            (q ** 3 * t2 ** 2 * N2 * Kbar[2] - 1.)) / denom

        return self


class SWG(Device):
    """SOI rib straight waveguides.

    Compute the effective index, dispersion and group index of a SOI
    rib straight waveguides.

    Available dimensions:

        - width: 400nm..488nm.
        - height: 220nm.
        - temperature: 25..225 centrigrades.

    Notes
    =====

    The effective index is interpolated from values obtained by FimmWAVE.
    Interpolation is done by a polynomial of degree 2.

    @ivar w: Width of the waveguide.
    @type w: scalar
    @ivar h: Height of the waveguide.
    @type h: scalar
    @ivar T: Working temperature.
    @type T: scalar
    @ivar neff: Effective index.
    @type neff: numpy.ndarray
    @ivar disp: Dispersion.
    @type disp: numpy.ndarray
    @ivar ng: Group index.
    @type ng: numpy.ndarray

    """

    pf488_25 = [-0.00000006023385, -0.00096294997155, 4.06422992413444]
    pf488_125 = [-0.00000006391000, -0.00095671206284, 4.08461627147979]
    pf488_225 = [-0.00000006806052, -0.00094901669805, 4.10613946676196]

    pf400_25 = [1.733732171601458e-007, -
                1.911057235401467e-003, 4.774734532616097e+000]
    pf400_125 = [1.477287990868007e-007, -
                 1.842388456386622e-003, 4.750925877259235e+000]
    pf400_225 = [1.255208385485759e-007, -
                 1.784557912270417e-003, 4.737972109281652e+000]

    def __init__(self, w, h, T):

        self.w = w
        self.h = h  # not used!
        self.T = T

    def sanity_check(self):
        pass

    def solve(self, wls):
        """Compute neff, disp and ng for a given wls.

        @param wls: Wavelength.
        @type wls: numpy.ndarray

        @raise ValueError: if w or T are out of bounds (and
            interpolation is not possible).

        """

        if (self.w, self.T) == (488, 25):
            pf = SWG.pf488_25
        elif (self.w, self.T) == (488, 125):
            pf = SWG.pf488_125
        elif (self.w, self.T) == (488, 225):
            pf = SWG.pf488_225
        elif (self.w, self.T) == (400, 225):
            pf = SWG.pf400_25
        elif (self.w, self.T) == (400, 125):
            pf = SWG.pf400_125
        elif (self.w, self.T) == (400, 225):
            pf = SWG.pf400_225
        elif (self.w < 400 or
              self.w > 488 or
              self.T < 25 or
              self.T > 225):
            raise ValueError('input out of bounds')
        else:
            import scipy.interpolate

            pf_ = numpy.zeros((2, 3, 3))
            [w_, T_] = numpy.meshgrid([400, 488], [25, 125, 225])
            pf_[0, 0, :] = SWG.pf400_25
            pf_[0, 1, :] = SWG.pf400_125
            pf_[0, 2, :] = SWG.pf400_225
            pf_[1, 0, :] = SWG.pf488_25
            pf_[1, 1, :] = SWG.pf488_125
            pf_[1, 2, :] = SWG.pf488_225
            pf = [scipy.interpolate.interp2d(w_.T, T_.T, pf_[:, :, i])(
                self.w, self.T)[0] for i in range(3)]

        # transform the input in an ndarray
        wls = numpy.asarray(wls)

        wls_nm = wls * 1e9
        neff = numpy.polyval(pf, wls_nm)
        # disp = dneff/dlambda [nm^-1]
        disp = numpy.polyval([2 * pf[0], pf[1]], wls_nm)
        # ng = neff - lambda * disp
        ng = neff - disp * wls_nm

        self.neff = neff
        self.disp = disp
        self.ng = ng

        return self


class Etalon(Device):

    """Etalon.

    @ivar R: Reflectivity.
    @ivar theta: Angle of incidence.

    @see: U{http://en.wikipedia.org/wiki/Etalon}

    """

    def __init__(self, layer, R, theta):
        self.FWHMwl = self.FSRwl / self.FINESSE
        self.layer = layer
        self.R = R
        self.theta = theta

    def sanity_check(self):
        pass

    def solve(self, wls):
        """Compute the frequency response of the etalon."""

        # transform the input in an ndarray
        wls = numpy.asarray(wls)

        n = self.layer.mat.n(wls)
        l = self.layer.thickness

        # phase difference between any succeeding reflection
        self.delta = 2 * numpy.pi / wls * 2 * n * l * numpy.cos(self.theta)
        # coefficient of finesse
        self.F = 4. * self.R / (1 - self.R) ** 2
        # transmission function
        self.Te = 1. / (1. + self.F * numpy.sin(self.delta / 2.) ** 2)
        self.Rmax = 1. - 1. / (1. + self.F)

        self.wl0 = numpy.mean(wls)
        self.FSRwl = self.wl0 ** 2 / \
            (2 * n * l * numpy.cos(self.theta) + self.wl0)

        (self.f0, self.FSR) = EMpy_gpu.utils.wl2f(self.wl0, self.FSRwl)

        self.FINESSE = numpy.pi / (2 * numpy.arcsin(1. / numpy.sqrt(self.F)))
        # self.FINESSE = numpy.pi * numpy.sqrt(self.F) / 2.
        # self.FINESSE = numpy.pi * numpy.sqrt(self.R) / (1 - self.R)

        self.FWHM = self.FSR / self.FINESSE

        return self
