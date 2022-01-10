"""Film Mode Matching Mode Solver

Implementation of the Film Mode Matching (FMM) algorithm, as described in:

 - Sudbo, "Film mode matching a versatile numerical method for vector mode field calculations in dielectric waveguides", Pure App. Optics, 2 (1993), 211-233
 - Sudbo, "Improved formulation of the film mode matching method for mode field calculations in dielectric waveguides", Pure App. Optics, 3 (1994), 381-388

Examples
========

    See L{FMM1d} and L{FMM2d}.

"""
from __future__ import print_function
from builtins import zip
from builtins import range
from builtins import object
from functools import reduce

__author__ = 'Luca Gamberale & Lorenzo Bolla'

import numpy
import scipy
import scipy.optimize
import copy
import EMpy_gpu.utils
from EMpy_gpu.modesolvers.interface import *
import pylab

class Message(object):
    def __init__(self, msg, verbosity=0):
        self.msg = msg
        self.verbosity = verbosity
        
    def show(self, verbosity=0):
        if self.verbosity <= verbosity:
            print((self.verbosity - 1) * '\t' + self.msg)
            
class Struct(object):
    """Empty class to fill with whatever I want. Maybe a dictionary would do?"""    
    pass

class Boundary(object):
    """Boundary conditions.
    
    Electric and Magnetic boundary conditions are translated to Symmetric 
    and Antisymmetric for each field.
    
    @ivar xleft: Left bc on x.
    @ivar xright: Right bc on x.
    @ivar yleft: Left bc on y.
    @ivar yright: Right bc on y.
    
    """
    
    def __init__(self, xleft='Electric Wall', 
                       yleft='Magnetic Wall', 
                       xright='Electric Wall', 
                       yright='Magnetic Wall'):        
        """Set the boundary conditions, validate and translate."""
        self.xleft = xleft
        self.yleft = yleft
        self.xright = xright
        self.yright = yright
        self.validate()
        self.translate()
    
    def validate(self):
        """Validate the input.
        
        @raise ValueError: Unknown boundary.
        
        """
        
        if not reduce(lambda x, y: x & y, 
                      [(x == 'Electric Wall') | (x == 'Magnetic Wall') for x in [self.xleft, self.yleft, self.xright, self.yright]]):
            raise ValueError('Unknown boundary.')
        
    def translate(self):
        """Translate for each field.
                
        @raise ValueError: Unknown boundary.
        
        """
        
        self.xh = ''
        self.xe = ''
        self.yh = ''
        self.ye = ''
        if self.xleft == 'Electric Wall':
            self.xh += 'A'
            self.xe += 'S'
        elif self.xleft == 'Magnetic Wall':
            self.xh += 'S'
            self.xe += 'A'
        else:
            raise ValueError('Unknown boundary.')
        if self.xright == 'Electric Wall':
            self.xh += 'A'
            self.xe += 'S'
        elif self.xright == 'Magnetic Wall':
            self.xh += 'S'
            self.xe += 'A'
        else:
            raise ValueError('Unknown boundary.')
        if self.yleft == 'Electric Wall':
            self.yh += 'A'
            self.ye += 'S'
        elif self.yleft == 'Magnetic Wall':
            self.yh += 'S'
            self.ye += 'A'
        else:
            raise ValueError('Unknown boundary.')
        if self.yright == 'Electric Wall':
            self.yh += 'A'
            self.ye += 'S'
        elif self.yright == 'Magnetic Wall':
            self.yh += 'S'
            self.ye += 'A'
        else:
            raise ValueError('Unknown boundary.')
    
    def __str__(self):
        return 'xleft = %s, xright = %s, yleft = %s, yright = %s' % (self.xleft, self.xright, self.yleft, self.yright)

class Slice(object):
    """One dimensional arrangement of layers and 1d modes.
    
    A slice is made of a stack of layers, i.e. refractive indeces with a thickness,
    with given boundary conditions.
    It holds 1d modes, both TE and TM.
    
    @ivar x1: start point of the slice in x.
    @ivar x2: end point of the slice in x.
    @ivar Uy: array of points delimiting the layers.
    @ivar boundary: boundary conditions.
    @ivar modie: E modes.
    @ivar modih: H modes.
    @ivar Ux: array of points delimiting the slices in x (internally set).
    @ivar refractiveindex: refractive index of all the slices (internally set).
    @ivar epsilon: epsilon of all the slices (internally set).
    @ivar wl: vacuum wavelength.

    """
    
    def __init__(self, x1, x2, Uy, boundary, modie, modih):        
        self.x1 = x1
        self.x2 = x2
        self.Uy = Uy
        self.boundary = boundary
        self.modie = modie
        self.modih = modih
        
    def __str__(self):
        return 'x1 = %g, x2 = %g\nUy = %s\nboundary = %s' % (self.x1, self.x2, self.Uy, self.boundary)

class FMMMode1d(Mode):
    """One dimensional mode.
    
    Note
    ====

        Virtual class.
        
    """
    
    pass

class FMMMode1dx(FMMMode1d):
    """Matching coefficients in the x-direction.
    
    L{FMMMode1dy}s are weighted by these coefficients to assure continuity.
    
    """
    
    def __str__(self):
        return 'sl = %s\nsr = %s\nal = %s\nar = %s\nk = %s\nU = %s' % \
                (self.sl.__str__(), 
                 self.sr.__str__(), 
                 self.al.__str__(), 
                 self.ar.__str__(),
                 self.k.__str__(),
                 self.U.__str__())

class FMMMode1dy(FMMMode1d):
    """One dimensional mode.
    
    It holds the coefficients that describe the mode in the FMM expansion.
    
    Note
    ====
    
        The mode is suppose one dimensional, in the y direction.
    
    @ivar sl: array of value of the mode at the lhs of each slice.
    @ivar sr: array of value of the mode at the rhs of each slice.
    @ivar al: array of value of the derivative of the mode at the lhs of each slice.
    @ivar ar: array of value of the derivative of the mode at the lhs of each slice.
    @ivar k: wavevector inside each layer.
    @ivar keff: effective wavevector.
    @ivar zero: how good the mode is? it must be as close to zero as possible!
    @ivar Uy: array of points delimiting the layers.
    
    """

    def eval(self, y_):
        """Evaluate the mode at y."""
        y = numpy.atleast_1d(y_)
        ny = len(y)
        f = numpy.zeros(ny, dtype=complex)
        for iU in range(len(self.U) - 1):
            k = self.k[iU]
            sl = self.sl[iU]
            al = self.al[iU]
            Ul = self.U[iU]
            Ur = self.U[iU+1]
            idx = numpy.where((Ul <= y) & (y <= Ur))
            yy = y[idx] - Ul
            f[idx] = sl * numpy.cos(k * yy) + al * sinxsux(k * yy) * yy
        return f
    
    def plot(self, y):
        f = self.eval(y)
        pylab.plot(y, numpy.real(f), y, numpy.imag(y))
        pylab.legend(('real', 'imag'))
        pylab.xlabel('y')
        pylab.ylabel('mode1d')
        pylab.show()
    
    def __str__(self):
        return 'sl = %s\nsr = %s\nal = %s\nar = %s\nk = %s\nkeff = %s\nzero = %s\nU = %s' % \
                (self.sl.__str__(), 
                 self.sr.__str__(), 
                 self.al.__str__(), 
                 self.ar.__str__(),
                 self.k.__str__(), 
                 self.keff.__str__(),
                 self.zero.__str__(),
                 self.U.__str__())

class FMMMode2d(Mode):
    """Two dimensional mode.
    
    It holds the coefficients that describe the mode in the FMM expansion.
    
    """
    
    def get_x(self, n=100):
        return numpy.linspace(self.slicesx[0].Ux[0], self.slicesx[0].Ux[-1], n)
    
    def get_y(self, n=100):
        return numpy.linspace(self.slicesx[0].Uy[0], self.slicesx[0].Uy[-1], n)
    
    def eval(self, x_=None, y_=None):
        """Evaluate the mode at x,y."""
        
        if x_ is None:
            x = self.get_x()
        else:
            x = numpy.atleast_1d(x_)
        if y_ is None:
            y = self.get_y()
        else:
            y = numpy.atleast_1d(y_)
            
        nmodi = len(self.modie)
        lenx = len(x)
        leny = len(y)
        k0 = 2. * numpy.pi / self.slicesx[0].wl
        kz = self.keff

        uh = numpy.zeros((nmodi, lenx), dtype=complex)
        ue = numpy.zeros_like(uh)
        udoth = numpy.zeros_like(uh)
        udote = numpy.zeros_like(uh)        
        Exsh = numpy.zeros((leny, nmodi), dtype=complex)
        Exah = numpy.zeros_like(Exsh)
        Exse = numpy.zeros_like(Exsh)
        Exae = numpy.zeros_like(Exsh)
        Eysh = numpy.zeros_like(Exsh)
        Eyah = numpy.zeros_like(Exsh)
        Eyse = numpy.zeros_like(Exsh)
        Eyae = numpy.zeros_like(Exsh)
        Ezsh = numpy.zeros_like(Exsh)
        Ezah = numpy.zeros_like(Exsh)
        Ezse = numpy.zeros_like(Exsh)
        Ezae = numpy.zeros_like(Exsh)
        cBxsh = numpy.zeros_like(Exsh)
        cBxah = numpy.zeros_like(Exsh)
        cBxse = numpy.zeros_like(Exsh)
        cBxae = numpy.zeros_like(Exsh)
        cBysh = numpy.zeros_like(Exsh)
        cByah = numpy.zeros_like(Exsh)
        cByse = numpy.zeros_like(Exsh)
        cByae = numpy.zeros_like(Exsh)
        cBzsh = numpy.zeros_like(Exsh)
        cBzah = numpy.zeros_like(Exsh)
        cBzse = numpy.zeros_like(Exsh)
        cBzae = numpy.zeros_like(Exsh)
        ExTE = numpy.zeros((leny,lenx), dtype=complex)
        EyTE = numpy.zeros_like(ExTE)
        EzTE = numpy.zeros_like(ExTE)
        ExTM = numpy.zeros_like(ExTE)
        EyTM = numpy.zeros_like(ExTE)
        EzTM = numpy.zeros_like(ExTE)
        cBxTE = numpy.zeros_like(ExTE)
        cByTE = numpy.zeros_like(ExTE)
        cBzTE = numpy.zeros_like(ExTE)
        cBxTM = numpy.zeros_like(ExTE)
        cByTM = numpy.zeros_like(ExTE)
        cBzTM = numpy.zeros_like(ExTE)


        for mx, slice in enumerate(self.slicesx):

            idx = numpy.where((slice.x1 <= x) & (x < slice.x2))
            x2 = x[idx] - slice.x1
            x1 = slice.x2 - x[idx]
            dx = slice.x2 - slice.x1
            
            for n in range(nmodi):

                fi = slice.modih[n].eval(y)
                fidot = dot(slice.modih[n]).eval(y)
                
                psi = slice.modie[n].eval(y)
                psisueps = sueps(slice.modie[n]).eval(y)
                psidotsueps = sueps(dot(slice.modie[n])).eval(y)
          
                kfh = self.modih[n].k[mx]
                kxh = scipy.sqrt(kfh**2 - kz**2)
                sl = self.modih[n].sl[mx] * (k0/kfh)**2
                al = self.modih[n].al[mx]
                sr = self.modih[n].sr[mx] * (k0/kfh)**2
                ar = self.modih[n].ar[mx]

                uh[n,idx]    = (numpy.sin(kxh * x1) * sl + numpy.sin(kxh * x2) * sr) / numpy.sin(kxh * dx)
                udoth[n,idx] = (numpy.sin(kxh * x1) * al + numpy.sin(kxh * x2) * ar) / numpy.sin(kxh * dx)

                kfe = self.modie[n].k[mx]
                kxe = scipy.sqrt(kfe**2 - kz**2)
                sl = self.modie[n].sl[mx] * (k0/kfe)**2
                al = self.modie[n].al[mx]
                sr = self.modie[n].sr[mx] * (k0/kfe)**2
                ar = self.modie[n].ar[mx]

                ue[n,idx]    = (numpy.sin(kxe * x1) * sl + numpy.sin(kxe * x2) * sr) / numpy.sin(kxe * dx)
                udote[n,idx] = (numpy.sin(kxe * x1) * al + numpy.sin(kxe * x2) * ar) / numpy.sin(kxe * dx)

                Exsh[:,n] = (kz/k0) * fi
                Exah[:,n] = 0
                Exse[:,n] = 0
                Exae[:,n] = -psidotsueps / k0**2

                Eysh[:,n] = 0
                Eyah[:,n] = 0
                Eyse[:,n] = -(kfe/k0)**2 * psisueps
                Eyae[:,n] = 0

                Ezsh[:,n] = 0
                Ezah[:,n] = -1j * fi / k0
                Ezse[:,n] = 1j * kz / k0**2 * psidotsueps
                Ezae[:,n] = 0

                cBxsh[:,n] = 0
                cBxah[:,n] = fidot / k0**2
                cBxse[:,n] = kz / k0 * psi
                cBxae[:,n] = 0

                cBysh[:,n] = (kfh/k0)**2 * fi
                cByah[:,n] = 0
                cByse[:,n] = 0
                cByae[:,n] = 0

                cBzsh[:,n] = -1j * kz / k0**2 * fidot
                cBzah[:,n] = 0
                cBzse[:,n] = 0
                cBzae[:,n] = -1j * psi / k0

            ExTE[:,idx] = numpy.tensordot(Exsh, uh[:,idx], axes=1) + numpy.tensordot(Exah, udoth[:,idx], axes=1)
            ExTM[:,idx] = numpy.tensordot(Exse, ue[:,idx], axes=1) + numpy.tensordot(Exae, udote[:,idx], axes=1)

            EyTE[:,idx] = numpy.tensordot(Eysh, uh[:,idx], axes=1) + numpy.tensordot(Eyah, udoth[:,idx], axes=1)
            EyTM[:,idx] = numpy.tensordot(Eyse, ue[:,idx], axes=1) + numpy.tensordot(Eyae, udote[:,idx], axes=1)

            EzTE[:,idx] = numpy.tensordot(Ezsh, uh[:,idx], axes=1) + numpy.tensordot(Ezah, udoth[:,idx], axes=1)
            EzTM[:,idx] = numpy.tensordot(Ezse, ue[:,idx], axes=1) + numpy.tensordot(Ezae, udote[:,idx], axes=1)

            cBxTE[:,idx] = numpy.tensordot(cBxsh, uh[:,idx], axes=1) + numpy.tensordot(cBxah, udoth[:,idx], axes=1)
            cBxTM[:,idx] = numpy.tensordot(cBxse, ue[:,idx], axes=1) + numpy.tensordot(cBxae, udote[:,idx], axes=1) 

            cByTE[:,idx] = numpy.tensordot(cBysh, uh[:,idx], axes=1) + numpy.tensordot(cByah, udoth[:,idx], axes=1)
            cByTM[:,idx] = numpy.tensordot(cByse, ue[:,idx], axes=1) + numpy.tensordot(cByae, udote[:,idx], axes=1)

            cBzTE[:,idx] = numpy.tensordot(cBzsh, uh[:,idx], axes=1) + numpy.tensordot(cBzah, udoth[:,idx], axes=1)
            cBzTM[:,idx] = numpy.tensordot(cBzse, ue[:,idx], axes=1) + numpy.tensordot(cBzae, udote[:,idx], axes=1)

        return (ExTE, ExTM, EyTE, EyTM, EzTE, EzTM, cBxTE, cBxTM, cByTE, cByTM, cBzTE, cBzTM)

    def fields(self, x=None, y=None):
        ExTE, ExTM, EyTE, EyTM, EzTE, EzTM, cBxTE, cBxTM, cByTE, cByTM, cBzTE, cBzTM = self.eval(x, y)
        Ex = ExTE + ExTM
        Ey = EyTE + EyTM
        Ez = EzTE + EzTM
        cBx = cBxTE + cBxTM
        cBy = cByTE + cByTM
        cBz = cBzTE + cBzTM
        return (Ex, Ey, Ez, cBx, cBy, cBz)
    
    def intensity(self, x=None, y=None):
        Ex, Ey, Ez, cBx, cBy, cBz = self.fields(x, y)
        cSz = .5 * (Ex * numpy.conj(cBy) - Ey * numpy.conj(cBx))
        return cSz
        
    def TEfrac_old(self, x_=None, y_=None):
        
        if x_ is None:
            x = self.get_x()
        else:
            x = numpy.atleast_1d(x_)
        if y_ is None:
            y = self.get_y()
        else:
            y = numpy.atleast_1d(y_)
            
        Ex, Ey, Ez, cBx, cBy, cBz, cSz = self.fields(x, y)
        cSTE = .5 * EMpy_gpu.utils.trapz2(Ex * numpy.conj(cBy), y, x)
        cSTM = .5 * EMpy_gpu.utils.trapz2(-Ey * numpy.conj(cBx), y, x)
        return numpy.abs(cSTE) / (numpy.abs(cSTE) + numpy.abs(cSTM))
    
    def TEfrac(self):
        
        Sx, Sy = self.__overlap(self)
        return Sx / (Sx - Sy)

    def overlap_old(self, m, x_=None, y_=None):
                
        if x_ is None:
            x = self.get_x()
        else:
            x = numpy.atleast_1d(x_)
        if y_ is None:
            y = self.get_y()
        else:
            y = numpy.atleast_1d(y_)
            
        Ex, Ey, Ez, cBx, cBy, cBz = self.fields(x, y)
        cSz = self.intensity(x, y)
        norm = scipy.sqrt(EMpy_gpu.utils.trapz2(cSz, y, x))
        Ex1, Ey1, Ez1, cBx1, cBy1, cBz1 = m.fields(x, y)
        cSz1 = m.intensity(x, y)
        norm1 = scipy.sqrt(EMpy_gpu.utils.trapz2(cSz1, y, x))
        return .5 * EMpy_gpu.utils.trapz2(Ex/norm * numpy.conj(cBy1/norm1) - Ey/norm * numpy.conj(cBx1/norm1), y, x)        

    def __overlap_old(self, mode):
            
        nmodi = len(self.modie)
        k0 = 2. * numpy.pi / self.slicesx[0].wl
        kz = self.keff
        
        Sx = 0j
        Sy = 0j

        for mx, slice in enumerate(self.slicesx):

            for n1 in range(nmodi):
                
                phi_n1 = slice.modih[n1]
                phidot_n1 = dot(phi_n1)
                
                psi_n1 = slice.modie[n1]
                psisueps_n1 = sueps(psi_n1)
                psidotsueps_n1 = sueps(dot(psi_n1))
          
                uh_n1 = copy.deepcopy(self.modih[n1])
                # reduce to a single slice
                kfh_n1 = uh_n1.k[mx]
                uh_n1.k = numpy.atleast_1d(scipy.sqrt(kfh_n1**2 - kz**2))
                uh_n1.sl = numpy.atleast_1d(uh_n1.sl[mx] * (k0/kfh_n1)**2)
                uh_n1.al = numpy.atleast_1d(uh_n1.al[mx])
                uh_n1.sr = numpy.atleast_1d(uh_n1.sr[mx] * (k0/kfh_n1)**2)
                uh_n1.ar = numpy.atleast_1d(uh_n1.ar[mx])
                uh_n1.U = numpy.atleast_1d(uh_n1.U[mx:mx+2])
                uhdot_n1 = dot(uh_n1)
                
                ue_n1 = copy.deepcopy(self.modie[n1])
                # reduce to a single slice
                kfe_n1 = ue_n1.k[mx]
                ue_n1.k = numpy.atleast_1d(scipy.sqrt(kfe_n1**2 - kz**2))
                ue_n1.sl = numpy.atleast_1d(ue_n1.sl[mx] * (k0/kfe_n1)**2)
                ue_n1.al = numpy.atleast_1d(ue_n1.al[mx])
                ue_n1.sr = numpy.atleast_1d(ue_n1.sr[mx] * (k0/kfe_n1)**2)
                ue_n1.ar = numpy.atleast_1d(ue_n1.ar[mx])
                ue_n1.U = numpy.atleast_1d(ue_n1.U[mx:mx+2])
                uedot_n1 = dot(ue_n1)
                
                for n2 in range(nmodi):

                    phi_n2 = mode.slicesx[mx].modih[n2]
                    phidot_n2 = dot(phi_n2)
                    
                    psi_n2 = mode.slicesx[mx].modie[n2]
                    psisueps_n2 = sueps(psi_n2)
                    psidotsueps_n2 = sueps(dot(psi_n2))
              
                    uh_n2 = copy.deepcopy(mode.modih[n2])
                    # reduce to a single slice
                    kfh_n2 = uh_n2.k[mx]
                    uh_n2.k = numpy.atleast_1d(scipy.sqrt(kfh_n2**2 - kz**2))
                    uh_n2.sl = numpy.atleast_1d(uh_n2.sl[mx] * (k0/kfh_n2)**2)
                    uh_n2.al = numpy.atleast_1d(uh_n2.al[mx])
                    uh_n2.sr = numpy.atleast_1d(uh_n2.sr[mx] * (k0/kfh_n2)**2)
                    uh_n2.ar = numpy.atleast_1d(uh_n2.ar[mx])
                    uh_n2.U = numpy.atleast_1d(uh_n2.U[mx:mx+2])
                    uhdot_n2 = dot(uh_n2)
                    
                    ue_n2 = copy.deepcopy(mode.modie[n2])
                    # reduce to a single slice
                    kfe_n2 = ue_n2.k[mx]
                    ue_n2.k = numpy.atleast_1d(scipy.sqrt(kfe_n2**2 - kz**2))
                    ue_n2.sl = numpy.atleast_1d(ue_n2.sl[mx] * (k0/kfe_n2)**2)
                    ue_n2.al = numpy.atleast_1d(ue_n2.al[mx])
                    ue_n2.sr = numpy.atleast_1d(ue_n2.sr[mx] * (k0/kfe_n2)**2)
                    ue_n2.ar = numpy.atleast_1d(ue_n2.ar[mx])
                    ue_n2.U = numpy.atleast_1d(ue_n2.U[mx:mx+2])
                    uedot_n2 = dot(ue_n2)
                    
                    Sx += kz * kfh_n2**2 / k0**3 * scalarprod(uh_n1, uh_n2) * scalarprod(phi_n1, phi_n2) \
                        - kfh_n2**2 / k0**4 * scalarprod(uedot_n1, uh_n2) * scalarprod(psidotsueps_n1, phi_n2)
                    Sy += kfe_n1**2 * kz / k0**3 * scalarprod(ue_n1, ue_n2) * scalarprod(psisueps_n1, psi_n2) \
                        + kfe_n1**2 / k0**4 * scalarprod(ue_n1, uhdot_n2) * scalarprod(psisueps_n1, phidot_n2)
                    
        return (Sx, Sy)

    def __overlap(self, mode):
            
        nmodi = len(self.modie)
        k0 = 2. * numpy.pi / self.slicesx[0].wl
        kz = self.keff
        
        Sx = 0j
        Sy = 0j

        for mx, slice in enumerate(self.slicesx):

            phi_n1s = []
            phidot_n1s = []
            psi_n1s = []
            psisueps_n1s = []
            psidotsueps_n1s = []
            uh_n1s = []
            uhdot_n1s = []
            ue_n1s = []
            uedot_n1s = []
            kfe_n1s = []
            kfh_n1s = []
            
            phi_n2s = []
            phidot_n2s = []
            psi_n2s = []
            psisueps_n2s = []
            psidotsueps_n2s = []
            uh_n2s = []
            uhdot_n2s = []
            ue_n2s = []
            uedot_n2s = []
            kfe_n2s = []
            kfh_n2s = []
            
            for n1 in range(nmodi):
                
                phi_n1 = slice.modih[n1]
                phi_n1s.append(phi_n1)
                phidot_n1s.append(dot(phi_n1))
                
                psi_n1 = slice.modie[n1]
                psi_n1s.append(psi_n1)
                psisueps_n1s.append(sueps(psi_n1))
                psidotsueps_n1s.append(sueps(dot(psi_n1)))
          
                uh_n1 = copy.deepcopy(self.modih[n1])
                # reduce to a single slice
                kfh_n1 = uh_n1.k[mx]
                kfh_n1s.append(kfh_n1)
                uh_n1.k = numpy.atleast_1d(scipy.sqrt(kfh_n1**2 - kz**2))
                uh_n1.sl = numpy.atleast_1d(uh_n1.sl[mx] * (k0/kfh_n1)**2)
                uh_n1.al = numpy.atleast_1d(uh_n1.al[mx])
                uh_n1.sr = numpy.atleast_1d(uh_n1.sr[mx] * (k0/kfh_n1)**2)
                uh_n1.ar = numpy.atleast_1d(uh_n1.ar[mx])
                uh_n1.U = numpy.atleast_1d(uh_n1.U[mx:mx+2])
                uh_n1s.append(uh_n1)
                uhdot_n1s.append(dot(uh_n1))
                
                ue_n1 = copy.deepcopy(self.modie[n1])
                # reduce to a single slice
                kfe_n1 = ue_n1.k[mx]
                kfe_n1s.append(kfe_n1)
                ue_n1.k = numpy.atleast_1d(scipy.sqrt(kfe_n1**2 - kz**2))
                ue_n1.sl = numpy.atleast_1d(ue_n1.sl[mx] * (k0/kfe_n1)**2)
                ue_n1.al = numpy.atleast_1d(ue_n1.al[mx])
                ue_n1.sr = numpy.atleast_1d(ue_n1.sr[mx] * (k0/kfe_n1)**2)
                ue_n1.ar = numpy.atleast_1d(ue_n1.ar[mx])
                ue_n1.U = numpy.atleast_1d(ue_n1.U[mx:mx+2])
                ue_n1s.append(ue_n1)
                uedot_n1s.append(dot(ue_n1))
                
                phi_n2 = mode.slicesx[mx].modih[n1]
                phi_n2s.append(phi_n2)
                phidot_n2s.append(dot(phi_n2))
                
                psi_n2 = mode.slicesx[mx].modie[n1]
                psi_n2s.append(psi_n2)
                psisueps_n2s.append(sueps(psi_n2))
                psidotsueps_n2s.append(sueps(dot(psi_n2)))
          
                uh_n2 = copy.deepcopy(mode.modih[n1])
                # reduce to a single slice
                kfh_n2 = uh_n2.k[mx]
                kfh_n2s.append(kfh_n2)
                uh_n2.k = numpy.atleast_1d(scipy.sqrt(kfh_n2**2 - kz**2))
                uh_n2.sl = numpy.atleast_1d(uh_n2.sl[mx] * (k0/kfh_n2)**2)
                uh_n2.al = numpy.atleast_1d(uh_n2.al[mx])
                uh_n2.sr = numpy.atleast_1d(uh_n2.sr[mx] * (k0/kfh_n2)**2)
                uh_n2.ar = numpy.atleast_1d(uh_n2.ar[mx])
                uh_n2.U = numpy.atleast_1d(uh_n2.U[mx:mx+2])
                uh_n2s.append(uh_n2)
                uhdot_n2s.append(dot(uh_n2))
                
                ue_n2 = copy.deepcopy(mode.modie[n1])
                # reduce to a single slice
                kfe_n2 = ue_n2.k[mx]
                kfe_n2s.append(kfe_n2)
                ue_n2.k = numpy.atleast_1d(scipy.sqrt(kfe_n2**2 - kz**2))
                ue_n2.sl = numpy.atleast_1d(ue_n2.sl[mx] * (k0/kfe_n2)**2)
                ue_n2.al = numpy.atleast_1d(ue_n2.al[mx])
                ue_n2.sr = numpy.atleast_1d(ue_n2.sr[mx] * (k0/kfe_n2)**2)
                ue_n2.ar = numpy.atleast_1d(ue_n2.ar[mx])
                ue_n2.U = numpy.atleast_1d(ue_n2.U[mx:mx+2])
                ue_n2s.append(ue_n2)
                uedot_n2.append(dot(ue_n2))

            for n1 in range(nmodi):
                
                uh_n1 = uh_n1s[n1]
                ue_n1 = ue_n1s[n1]
                uedot_n1 = uedot_n1s[n1]
                phi_n1 = phi_n1s[n1]
                psi_n1 = psi_n1s[n1]
                psidotsueps_n1 = psidotsueps_n1s[n1]
                psisueps_n1 = psisueps_n1s[n1]
                kfe_n1 = kfe_n1s[n1]
                
                for n2 in range(nmodi):            

                    uh_n2 = uh_n2s[n2]
                    uhdot_n2 = uhdot_n2s[n2]
                    ue_n2 = ue_n2s[n2]
                    phi_n2 = phi_n2s[n2]
                    phidot_n2 = phidot_n2s[n2]
                    psi_n2 = psi_n2s[n2]
                    kfh_n2 = kfh_n2s[n2]
                
                    Sx += kz * kfh_n2**2 / k0**3 * scalarprod(uh_n1, uh_n2) * scalarprod(phi_n1, phi_n2) \
                        - kfh_n2**2 / k0**4 * scalarprod(uedot_n1, uh_n2) * scalarprod(psidotsueps_n1, phi_n2)
                    Sy += kfe_n1**2 * kz / k0**3 * scalarprod(ue_n1, ue_n2) * scalarprod(psisueps_n1, psi_n2) \
                        + kfe_n1**2 / k0**4 * scalarprod(ue_n1, uhdot_n2) * scalarprod(psisueps_n1, phidot_n2)
                    
        return (Sx, Sy)

    def overlap(self, mode):
        
        Sx, Sy = self.__overlap(mode)
        return Sx - Sy

    def norm(self):
                
        return scipy.sqrt(self.overlap(self))
    
    def normalize(self):
        
        n = self.norm()
        for ue, uh in zip(self.modie, self.modih):
            ue.sl /= n
            ue.al /= n
            ue.sr /= n
            ue.ar /= n
            uh.sl /= n
            uh.al /= n
            uh.sr /= n
            uh.ar /= n

    def get_fields_for_FDTD(self, x, y):
        """Get mode's field on a staggered grid.
        
        Note: ignores some fields on the boudaries.
        
        """
        
        x0 = self.get_x()
        y0 = self.get_y()
        Ex, Ey, Ez, cBx, cBy, cBz = self.fields(x0, y0)
        
        # Ex: ignores y = 0, max
        x_Ex_FDTD = EMpy_gpu.utils.centered1d(x)
        y_Ex_FDTD = y[1:-1]
        Ex_FDTD = EMpy_gpu.utils.interp2(x_Ex_FDTD, y_Ex_FDTD, x0, y0, Ex)
        # Ey: ignores x = 0, max
        x_Ey_FDTD = x[1:-1]
        y_Ey_FDTD = EMpy_gpu.utils.centered1d(y)
        Ey_FDTD = EMpy_gpu.utils.interp2(x_Ey_FDTD, y_Ey_FDTD, x0, y0, Ey)
        # Ez: ignores x, y = 0, max
        x_Ez_FDTD = x[1:-1]
        y_Ez_FDTD = y[1:-1]
        Ez_FDTD = EMpy_gpu.utils.interp2(x_Ez_FDTD, y_Ez_FDTD, x0, y0, Ez)
        # Hx: ignores x = 0, max, /120pi, reverse direction
        x_Hx_FDTD = x[1:-1]
        y_Hx_FDTD = EMpy_gpu.utils.centered1d(y)
        Hx_FDTD = EMpy_gpu.utils.interp2(x_Hx_FDTD, y_Hx_FDTD, x0, y0, cBx) / (-120. * numpy.pi) # OKKIO!
        # Hy: ignores y = 0, max, /120pi, reverse direction
        x_Hy_FDTD = EMpy_gpu.utils.centered1d(x)
        y_Hy_FDTD = y[1:-1]
        Hy_FDTD = EMpy_gpu.utils.interp2(x_Hy_FDTD, y_Hy_FDTD, x0, y0, Hy) / (-120. * numpy.pi)
        # Hz: /120pi, reverse direction
        x_Hz_FDTD = EMpy_gpu.utils.centered1d(x)
        y_Hz_FDTD = EMpy_gpu.utils.centered1d(y)
        Hz_FDTD = EMpy_gpu.utils.interp2(x_Hz_FDTD, y_Hz_FDTD, x0, y0, Hz) / (-120. * numpy.pi)
        
        return (Ex_FDTD, Ey_FDTD, Ez_FDTD, Hx_FDTD, Hy_FDTD, Hz_FDTD)    

    def plot(self, x_=None, y_=None):
        
        if x_ is None:
            x = self.get_x()
        else:
            x = numpy.atleast_1d(x_)
        if y_ is None:
            y = self.get_y()
        else:
            y = numpy.atleast_1d(y_)

        f = self.fields(x, y)

        # fields
        pylab.figure()
        titles = ['Ex', 'Ey', 'Ez', 'cBx', 'cBy', 'cBz']
        for i in range(6):
            subplot_id = 231 + i
            pylab.subplot(subplot_id)
            pylab.contour(x, y, numpy.abs(f[i]))
            pylab.xlabel('x')
            pylab.ylabel('y')
            pylab.title(titles[i])
            pylab.axis('image')
        pylab.show()
        
        # power
        pylab.figure()
        pylab.contour(x, y, numpy.abs(f[-1]))
        pylab.xlabel('x')
        pylab.ylabel('y')
        pylab.title('cSz')
        pylab.axis('image')
        pylab.show()
    
    def __str__(self):
        return 'neff = %s' % (self.keff / (2 * numpy.pi / self.slicesx[0].wl))

class FMM(ModeSolver):
    pass

class FMM1d(FMM):
    """Drive to simulate 1d structures.
    
    Examples
    ========

    Find the first 3 TE modes of two slabs of refractive indeces 1 and 3, 
    of thickness 1um each, for wl = 1, with symmetric boundary conditions:

        >>> import numpy
        >>> import FMM
        >>> Uy = numpy.array([0., 1., 2.])
        >>> ny = numpy.array([1., 3.])
        >>> wl = 1.
        >>> nmodi = 3
        >>> simul = FMM.FMM1d(Uy, ny, 'SS').solve(wl, nmodi, 'TE')
        >>> keff_0_expected = 18.790809413149393
        >>> keff_1_expected = 18.314611633384185
        >>> keff_2_expected = 17.326387847565034
        >>> assert(numpy.allclose(simul.modes[0].keff, keff_0_expected))
        >>> assert(numpy.allclose(simul.modes[1].keff, keff_1_expected))
        >>> assert(numpy.allclose(simul.modes[2].keff, keff_2_expected))
        
    """
    
    def __init__(self, Uy, ny, boundary):
        """Set coordinates of regions, refractive indeces and boundary conditions."""
        self.Uy = Uy
        self.ny = ny
        self.boundary = boundary
        
    def solve(self, wl, nmodes, polarization, verbosity=0):
        """Find nmodes modes at a given wavelength and polarization."""
        Message('Solving 1d modes.', 1).show(verbosity)
        self.wl = wl
        self.nmodes = nmodes
        self.polarization = polarization
        self.modes = FMM1d_y(self.Uy, self.ny, self.wl, self.nmodes, self.boundary, self.polarization, verbosity)
        return self
    
class FMM2d(FMM):
    """Drive to simulate 2d structures.
    
    Examples
    ========

    Find the first 2 modes of a lossy Si channel waveguide in SiO2, using
    only 3 1dmodes and with electric and magnetic bc on x and y, respectively:

        >>> import numpy
        >>> import FMM
        >>> wl = 1.55
        >>> nmodislices = 3
        >>> nmodi2d = 2
        >>> Ux = numpy.array([0, 2, 2.4, 4.4])
        >>> Uy = numpy.array([0, 2, 2.22, 4.22])
        >>> boundary = Boundary(xleft='Electric Wall', 
                                yleft='Magnetic Wall', 
                                xright='Electric Wall', 
                                yright='Magnetic Wall')
        >>> n2 = 1.446
        >>> n1 = 3.4757 - 1e-4j
        >>> refindex = numpy.array([[n2, n2, n2],
                                    [n2, n1, n2],
                                    [n2, n2, n2]])
        >>> simul = FMM.FMM2d(Ux, Uy, refindex, boundary).solve(wl, nmodislices, nmodi2d)
        >>> keff0_expected = 9.666663697969399e+000 -4.028846755836984e-004j
        >>> keff1_expected = 7.210476803133368e+000 -2.605078086535284e-004j
        >>> assert(numpy.allclose(simul.modes[0].keff, keff0_expected))
        >>> assert(numpy.allclose(simul.modes[1].keff, keff1_expected))
        
    """
    
    def __init__(self, Ux, Uy, rix, boundary):
        """Set coordinates of regions, refractive indeces and boundary conditions."""
        self.Ux = Ux
        self.Uy = Uy
        self.rix = rix
        self.boundary = boundary
        
    def solve(self, wl, n1dmodes, nmodes, verbosity=0):
        """Find nmodes modes at a given wavelength using n1dmodes 1d modes in each slice."""
        Message('Solving 2d modes', 1).show(verbosity)
        self.wl = wl
        self.n1dmodes = n1dmodes
        self.nmodes = nmodes
        self.slices = script1d(self.Ux, self.Uy, self.rix, self.wl, self.boundary, self.n1dmodes, verbosity)
        self.modes = FMM1d_x_component(self.slices, nmodes, verbosity)
        return self

def analyticalsolution(nmodi, TETM, FMMpars):
    
    betay = FMMpars['beta']
    epsilon = FMMpars['epsilon']
    Uy = FMMpars['Uy']
    by = FMMpars['boundary']

    Nregions = len(epsilon)
    sl = numpy.zeros((nmodi,Nregions), dtype=complex)
    sr = numpy.zeros_like(sl)
    al = numpy.zeros_like(sl)
    ar = numpy.zeros_like(sl)
    # interval
    D = Uy[-1] - Uy[0]
    if TETM == 'TE':
        N = numpy.sqrt(2. / D)
    else:
        N = numpy.sqrt(2. / D * epsilon[0])

    # boundary condition
    if by == 'AA':
        kn = (numpy.pi * numpy.arange(1, nmodi + 1) / D)
        kn = kn[:, numpy.newaxis]
        sl = numpy.sin(kn * (Uy[:-1] - Uy[0]))
        sr = numpy.sin(kn * (Uy[1:]  - Uy[0]))
        al = numpy.cos(kn * (Uy[:-1] - Uy[0]))
        ar = numpy.cos(kn * (Uy[1:]  - Uy[0]))
        sr[:, -1] = 0.
        sl[:, 0] = 0.
    elif by == 'AS':
        kn = (numpy.pi * (numpy.arange(0, nmodi) + .5) / D)
        kn = kn[:, numpy.newaxis]
        sl = numpy.sin(kn * (Uy[:-1] - Uy[0]))
        sr = numpy.sin(kn * (Uy[1:]  - Uy[0]))
        al = numpy.cos(kn * (Uy[:-1] - Uy[0]))
        ar = numpy.cos(kn * (Uy[1:]  - Uy[0]))
        ar[:, -1] = 0.
        sl[:, 0] = 0.
    elif by == 'SA':
        kn = (numpy.pi * (numpy.arange(0, nmodi) + .5) / D)
        kn = kn[:, numpy.newaxis]
        sl = numpy.cos(kn * (Uy[:-1] - Uy[0]))
        sr = numpy.cos(kn * (Uy[1:]  - Uy[0]))
        al = -numpy.sin(kn * (Uy[:-1] - Uy[0]))
        ar = -numpy.sin(kn * (Uy[1:]  - Uy[0]))
        sr[:, -1] = 0.
        al[:, 0] = 0.
    elif by == 'SS':
        kn = (numpy.pi * numpy.arange(0, nmodi) / D)
        kn = kn[:, numpy.newaxis]
        sl = numpy.cos(kn * (Uy[:-1] - Uy[0]))
        sr = numpy.cos(kn * (Uy[1:]  - Uy[0]))
        al = -numpy.sin(kn * (Uy[:-1] - Uy[0]))
        ar = -numpy.sin(kn * (Uy[1:]  - Uy[0]))
        ar[:, -1] = 0.
        al[:, 0] = 0.

    # normalizzazione
    sl *= N
    sr *= N

    for n in range(0, nmodi):
        al[n,:] *= N * kn[n]
        ar[n,:] *= N * kn[n]

    # caso speciale. se k=0 la funzione e' costante e la normalizzazione e'
    # diversa. capita solo con boundary SS e per il primo modo
    if by == 'SS':
        sqrt2 = numpy.sqrt(2.)
        sl[0,:] /= sqrt2
        sr[0,:] /= sqrt2
        al[0,:] /= sqrt2
        ar[0,:] /= sqrt2

    modi = []
    for mk in range(0, nmodi):
        modo = FMMMode1dy()
        modo.sl = sl[mk,:].astype(complex)
        modo.sr = sr[mk,:].astype(complex)
        modo.al = al[mk,:].astype(complex)
        modo.ar = ar[mk,:].astype(complex)
        modo.k = kn[mk] * numpy.ones(Nregions)
        modo.U = Uy
        modo.keff = scipy.sqrt(betay[0]**2 - kn[mk]**2)
        modo.zero = 0.
        modo.pars = FMMpars
        modi.append(modo)
    return modi

def sinxsux(x):
    return numpy.sinc(x / numpy.pi)
    
def FMMshootingTM(kz_, FMMpars):
    
    betay = FMMpars['beta']
    eps = FMMpars['epsilon']
    Uy = FMMpars['Uy']
    by = FMMpars['boundary']
    
    kz = numpy.atleast_1d(kz_)

    Nregions = len(betay)
    d = numpy.diff(Uy)
    Delta = numpy.zeros_like(kz)

    sl = numpy.zeros(Nregions, dtype=complex)
    sr = numpy.zeros_like(sl)
    al = numpy.zeros_like(sl)
    ar = numpy.zeros_like(sl)
    
    k_ = scipy.sqrt(betay**2 - kz[:,numpy.newaxis]**2)
    kd = k_[:,numpy.newaxis] * d
    sinkdsuk_ = sinxsux(kd) * d
    coskd_ = numpy.cos(kd)
    sinkdk_ = numpy.sin(kd) * k_[:,numpy.newaxis]
    
    # left boundary condition
    if by[0] == 'A':
        al[0] = 1
    elif by[0] == 'S':
        sl[0] = 1
    else:
        raise ValueError('unrecognized left boundary condition')

    # right boundary condition
    if by[1] == 'A':
        ar[-1] = 1
    elif by[1] == 'S':
        sr[-1] = 1
    else:
        raise ValueError('unrecognized right boundary condition')
        
    # ciclo sui layer
    maxbetay = numpy.max(numpy.real(betay))
    n1 = numpy.argmax(numpy.real(betay)) + 1
    if n1 == Nregions:
        n1 = Nregions - 1
    n2 = n1 + 1

    modo = FMMMode1dy()
    
    for m in range(0, len(kz)):

        k = k_[m,:]
        sinkdsuk = sinkdsuk_[m,:][0]
        coskd = coskd_[m,:][0]
        sinkdk = sinkdk_[m,:][0]

        for idx in range(0, n1):
            
            sr[idx] = sl[idx] * coskd[idx] + al[idx] * sinkdsuk[idx]
            ar[idx] = al[idx] * coskd[idx] - sl[idx] * sinkdk[idx]

            #******************* requirement of continuity
            if idx < n1 - 1:
                sl[idx+1] = sr[idx];
                al[idx+1] = ar[idx] / eps[idx] * eps[idx + 1];
            #*******************
            
        for idx1 in range(Nregions - 1, n2 - 2, -1):
            
            sl[idx1] = sr[idx1] * coskd[idx1] - ar[idx1] * sinkdsuk[idx1]
            al[idx1] = ar[idx1] * coskd[idx1] + sr[idx1] * sinkdk[idx1]

            #******************* requirement of continuity
            if idx1 > n2:
                sr[idx1 - 1] = sl[idx1]
                ar[idx1 - 1] = al[idx1] / eps[idx1] * eps[idx1 - 1]
            #*******************

        Delta[m] = (eps[n1-1] * sr[n1-1] * al[n2-1] - eps[n2-1] * ar[n1-1] * sl[n2-1])

        if len(kz) < 2:
            
            # normalize and save only if len(kz) == 1
            # otherwise, modo is ignored and only Delta is useful

            # normalizza la propagazione sinistra e quella destra
            alfa = sr[n1-1] / sl[n2-1]
            sl[n2-1:] *= alfa
            sr[n2-1:] *= alfa
            al[n2-1:] *= alfa
            ar[n2-1:] *= alfa

            modo.sl = sl
            modo.sr = sr
            modo.al = al
            modo.ar = ar
            modo.k = k
            modo.U = Uy
            modo.keff = kz
            modo.zero = Delta
            modo.pars = FMMpars

    return (Delta, modo)

def FMMshooting(kz_, FMMpars):

    betay = FMMpars['beta']
    Uy = FMMpars['Uy']
    by = FMMpars['boundary']

    kz = numpy.atleast_1d(kz_)

    Nregions = len(betay)
    d = numpy.diff(Uy)
    Delta = numpy.zeros_like(kz)

    sl = numpy.zeros(Nregions, dtype=complex)
    sr = numpy.zeros_like(sl)
    al = numpy.zeros_like(sl)
    ar = numpy.zeros_like(sl)
    
    k_ = scipy.sqrt(betay**2 - kz[:,numpy.newaxis]**2)
    kd = k_[:,numpy.newaxis] * d
    sinkdsuk_ = sinxsux(kd) * d
    coskd_ = numpy.cos(kd)
    sinkdk_ = numpy.sin(kd) * k_[:,numpy.newaxis]
    
    # left boundary condition
    if by[0] == 'A':
        al[0] = 1
    elif by[0] == 'S':
        sl[0] = 1
    else:
        raise ValueError('unrecognized left boundary condition')

    # right boundary condition
    if by[1] == 'A':
        ar[-1] = 1
    elif by[1] == 'S':
        sr[-1] = 1
    else:
        raise ValueError('unrecognized right boundary condition')

    # ciclo sui layer
    maxbetay = numpy.max(numpy.real(betay))
    n1 = numpy.argmax(numpy.real(betay)) + 1
    if n1 == Nregions:
        n1 = Nregions - 1
    n2 = n1 + 1

    modo = FMMMode1dy()
    
    for m in range(0, len(kz)):

        k = k_[m,:]
        sinkdsuk = sinkdsuk_[m,:][0]
        coskd = coskd_[m,:][0]
        sinkdk = sinkdk_[m,:][0]
    
        for idx in range(0, n1):
            
            sr[idx] = sl[idx] * coskd[idx] + al[idx] * sinkdsuk[idx]
            ar[idx] = al[idx] * coskd[idx] - sl[idx] * sinkdk[idx]

            #******************* requirement of continuity
            if idx < n1 - 1:
                sl[idx + 1] = sr[idx];
                al[idx + 1] = ar[idx];
            #*******************
            
        for idx1 in range(Nregions - 1, n2 - 2, -1):
            
            sl[idx1] = sr[idx1] * coskd[idx1] - ar[idx1] * sinkdsuk[idx1]
            al[idx1] = ar[idx1] * coskd[idx1] + sr[idx1] * sinkdk[idx1]

            #******************* requirement of continuity
            if idx1 > n2:
                sr[idx1 - 1] = sl[idx1]
                ar[idx1 - 1] = al[idx1]
            #*******************
        Delta[m] = (sr[n1-1] * al[n2-1] - ar[n1-1] * sl[n2-1])

##    len_kz = len(kz)    
##    k = k_[0,:]
##    sinkdsuk = sinkdsuk_[0,:][0]
##    coskd = coskd_[0,:][0]
##    sinkdk = sinkdk_[0,:][0]
##    code = """
##            for (int m = 0; m < len_kz; ++m) {
##                //k = k_(m,:);
##                //sinkdsuk = sinkdsuk_(0,:);
##                //coskd = coskd_(0,:);
##                //sinkdk = sinkdk_(0,:);
##                int nn1 = int(n1);
##                for (int idx = 0; idx < nn1; ++idx) {
##                    sr(idx) = sl(idx) * coskd(idx) + al(idx) * sinkdsuk(idx);
##                    ar(idx) = al(idx) * coskd(idx) - sl(idx) * sinkdk(idx);
##                    if (idx < nn1 - 1) {
##                        sl(idx + 1) = sr(idx);
##                        al(idx + 1) = ar(idx);
##                    }
##                }
##                int nn2 = int(n2);
##                for (int idx1 = Nregions - 1; idx1 > nn2 - 2; --idx1) {
##                    sl(idx1) = sr(idx1) * coskd(idx1) - ar(idx1) * sinkdsuk(idx1);
##                    al(idx1) = ar(idx1) * coskd(idx1) + sr(idx1) * sinkdk(idx1);
##                    if (idx1 > nn2) {
##                        sr(idx1 - 1) = sl(idx1);
##                        ar(idx1 - 1) = al(idx1);
##                    }
##                }
##                //Delta(m) = std::complex<double>(1) * (sr(nn1-1) * al(nn2-1) - ar(nn1-1) * sl(nn2-1));
##            }
##            """
##    
##    from scipy import weave
##    from scipy.weave import converters
##    weave.inline(code,
##                 ['n1', 'n2', 'Nregions', 'sl', 'sr', 'al', 'ar', 'len_kz', 'Delta', 
##                  'k', 'sinkdsuk', 'sinkdk', 'coskd', 
##                  'k_', 'sinkdsuk_', 'sinkdk_', 'coskd_'],
##                 type_converters = converters.blitz,
##                 compiler = 'gcc')
                    
    if len(kz) < 2:
            
        # normalize and save only if len(kz) == 1
        # otherwise, modo is ignored and only Delta is useful

        # normalizza la propagazione sinistra e quella destra
        alfa = sr[n1-1] / sl[n2-1]

        sl[n2-1:] *= alfa
        sr[n2-1:] *= alfa
        al[n2-1:] *= alfa
        ar[n2-1:] *= alfa

        modo.sl = sl
        modo.sr = sr
        modo.al = al
        modo.ar = ar
        modo.k = k
        modo.U = Uy
        modo.keff = kz
        modo.zero = Delta
        modo.pars = FMMpars

    return (Delta, modo)

def remove_consecutives(x, y):

    b = numpy.r_[numpy.diff(x) == 1, 0].astype(int)

    ic = 0
    flag = 0
    l = []
    for ib in range(len(b)):
        if flag == 0:
            c = [x[ib]]
            ic += 1
            if b[ib] == 1:
                flag = 1
            else:
                l.append(c)
        else:
            c.append(x[ib])
            if b[ib] != 1:
                flag = 0
                l.append(c)

    index = []
    for il, ll in enumerate(l):
        newi = ll
        itmp = numpy.argmax(y[newi])
        index.append(newi[0] + itmp)
    
    return index

def findzerosnew(x, y, searchinterval):

    minsi = 2 * numpy.abs(x[1] - x[0])
    if searchinterval < minsi:
        searchinterval = minsi

    dy = numpy.r_[0, numpy.diff(numpy.diff(scipy.log(y))), 0]
    idy = numpy.where(dy > 0.005)[0]
    
    if len(idy) == 0:
        zeri = numpy.array([])
        z1 = numpy.array([])
        z2 = numpy.array([])
    else:
        ind = remove_consecutives(idy, dy)
        zeri = x[ind]
        z1 = numpy.zeros_like(zeri)
        z2 = numpy.zeros_like(zeri)

        dz = numpy.abs(numpy.diff(zeri))
        if len(dz) == 0:
            z1[0] = zeri - searchinterval/2
            z2[0] = zeri + searchinterval/2
        else:            
            delta = numpy.min([dz[0], searchinterval])
            z1[0] = zeri[0] - delta/2
            z2[0] = zeri[0] + delta/2

            for idx in range(1, len(zeri) - 1):
                delta = numpy.min([dz[idx - 1], dz[idx], searchinterval])
                z1[idx] = zeri[idx] - delta/2
                z2[idx] = zeri[idx] + delta/2
                
            delta = numpy.min([dz[-1], searchinterval])
            z1[-1] = zeri[-1] - delta/2
            z2[-1] = zeri[-1] + delta/2
            
    return (zeri, z1, z2)
                
def absfzzero2(t, f, xmin, xmax, ymin, ymax):
    
    xmean = numpy.mean([xmin, xmax])
    ymean = numpy.mean([ymin, ymax])
    xwidth = xmax - xmin
    ywidth = ymax - ymin
    x = xmean + xwidth * t[0] / (1. + numpy.abs(t[0])) / 2.
    y = ymean + ywidth * t[1] / (1. + numpy.abs(t[1])) / 2.
    z = x + 1j * y
    fv = f(z)
    return numpy.abs(fv)**2

def fzzeroabs2(f, zmin, zmax):
    
    xmin = numpy.real(zmin)
    ymin = numpy.imag(zmin)
    xmax = numpy.real(zmax)
    ymax = numpy.imag(zmax)

    tx0 = 0.
    ty0 = 0.

    t0 = scipy.optimize.fmin(lambda t: absfzzero2(t, f, xmin, xmax, ymin, ymax), [tx0, ty0], 
                             maxiter=100000, maxfun=100000, xtol=1e-15, ftol=1e-15, disp=0)

    xmean = numpy.mean([xmin, xmax])
    ymean = numpy.mean([ymin, ymax])
    xwidth = xmax - xmin
    ywidth = ymax - ymin
    x0 = xmean + xwidth * t0[0] / (1 + numpy.abs(t0[0])) / 2
    y0 = ymean + ywidth * t0[1] / (1 + numpy.abs(t0[1])) / 2

    z0 = x0 + 1j * y0
    valf = f(z0)
    
    return (z0, valf)

def scalarprod(modo1, modo2):

    d = numpy.diff(modo1.U)

    ky1 = modo1.k
    al1 = modo1.al
    sl1 = modo1.sl
    ar1 = modo1.ar
    sr1 = modo1.sr

    ky2 = modo2.k
    al2 = modo2.al
    sl2 = modo2.sl
    ar2 = modo2.ar
    sr2 = modo2.sr

    Nlayers = len(modo1.sl)
    scprod = numpy.zeros_like(modo1.sl)

    for idy in range(Nlayers):
        
        if numpy.allclose(ky1[idy], ky2[idy]):
            if numpy.linalg.norm(ky1) < 1e-10:
                scprod[idy] = sl1[idy] * sl2[idy] * (modo1.U[idy+1] - modo1.U[idy])
            else:
                scprod[idy] = (sl1[idy] * al2[idy] - sr1[idy] * ar2[idy]) / ky1[idy] / ky2[idy] / 2. + \
                              d[idy]/2. * (sl1[idy] * sl2[idy] + al1[idy] * al2[idy] / ky1[idy] / ky2[idy])
        else:
            if numpy.linalg.norm(ky1) < 1e-10:
                scprod[idy] = sl1[idy] * (al2[idy] - ar2[idy]) / ky2[idy]**2
            elif numpy.linalg.norm(ky2) < 1e-10:
                scprod[idy] = sl2[idy] * (al1[idy] - ar1[idy]) / ky1[idy]**2
            else:
                scprod[idy] = (sr1[idy] * ar2[idy] - ar1[idy] * sr2[idy] - 
                               sl1[idy] * al2[idy] + al1[idy] * sl2[idy]) / (ky1[idy]**2 - ky2[idy]**2)

    return numpy.sum(scprod)

def sueps(modo):
    
    eps = modo.pars['epsilon']
    modosueps = copy.deepcopy(modo)
    modosueps.sl /= eps
    modosueps.sr /= eps
    modosueps.al /= eps
    modosueps.ar /= eps
    
    return modosueps

def FMM1d_y(Uy, ny, wl, nmodi, boundaryRL, TETM, verbosity=0):
    
    k0 = 2 * numpy.pi / wl
    betay = ny * k0
    Nstepskz = 1543
    searchinterval = max(50. / Nstepskz, numpy.abs(numpy.min(numpy.imag(2. * betay))))
    imsearchinterval = 10 * k0
    ypointsperregion = 5000
    
    FMMpars = {'epsilon': ny**2, 'beta': betay, 'boundary': boundaryRL, 'Uy': Uy}

    # analytical solution
    if numpy.allclose(ny, ny[0]):
        Message('Uniform slice found: using analytical solution.', 2).show(verbosity)
        return analyticalsolution(nmodi, TETM, FMMpars)
    
    ##rekz = numpy.linspace(numpy.max(numpy.real(betay)) + searchinterval, 0., Nstepskz)
    rekz2 = numpy.linspace((numpy.max(numpy.real(betay))+searchinterval)**2, 0., Nstepskz)
    rekz = scipy.sqrt(rekz2)
    if TETM == 'TM':
        Message('Shooting TM.', 3).show(verbosity)
        matchingre, modotmp = FMMshootingTM(rekz, FMMpars)
    else:
        Message('Shooting TE.', 3).show(verbosity)
        matchingre, modotmp = FMMshooting(rekz, FMMpars)
        
    nre = rekz / k0
    nre2 = rekz2 / k0**2
    ##zerire, z1, z2 = findzerosnew(nre, numpy.abs(matchingre), searchinterval / k0)
    zerire2, z12, z22 = findzerosnew(nre2, numpy.abs(matchingre), (searchinterval / k0)**2)
    zerire = scipy.sqrt(zerire2)
    kz1 = zerire * k0 - searchinterval / 2. + 1j * imsearchinterval
    kz2 = zerire * k0 + searchinterval / 2. - 1j * imsearchinterval
    Message('Found %d real zeros.' % len(zerire), 2).show(verbosity)
    if len(zerire) < nmodi:
        Message('Number of real zeros not enough: scan imaginary axis.', 2).show(verbosity)
        imkza = -numpy.max(numpy.real(betay))
        imkzb = 0.
        while len(kz1) < nmodi:
            imkza = imkza + numpy.max(numpy.real(betay))
            imkzb = imkzb + numpy.max(numpy.real(betay))
            ##imkz = numpy.linspace(imkza, imkzb, Nstepskz)
            imkz2 = numpy.linspace(imkza**2, imkzb**2, Nstepskz)
            imkz = scipy.sqrt(imkz2)
            if TETM == 'TM':
                matchingim, modotmp = FMMshootingTM(1j * imkz, FMMpars)
            else:
                matchingim, modotmp = FMMshooting(1j * imkz, FMMpars)
            nim = imkz * wl / 2. / numpy.pi
            nim2 = imkz2 * (wl / 2. / numpy.pi)**2
            ##zeriim, z1, z2 = findzerosnew(nim, numpy.abs(matchingim), searchinterval / k0)
            zeriim2, z12, z22 = findzerosnew(nim2, numpy.abs(matchingim), (searchinterval / k0)**2)
            zeriim = scipy.sqrt(zeriim2)
            Message('Found %d imag zeros.' % len(zeriim), 2).show(verbosity)
            kz1 = numpy.r_[kz1, 1j * (zeriim * k0 - imsearchinterval / 2. + 1j * searchinterval / 2.)]
            kz2 = numpy.r_[kz2, 1j * (zeriim * k0 + imsearchinterval / 2. - 1j * searchinterval / 2.)]

    mk = 0
    modi = []
    # inizia il ciclo sugli intervalli
    Message('Refine zeros.', 2).show(verbosity)
    for m in range(0, len(kz1)):

        if mk == nmodi:
            break
        if TETM == 'TM':
            z0, valf = fzzeroabs2(lambda kz: FMMshootingTM(kz, FMMpars)[0], kz1[m], kz2[m])
            z0 = numpy.atleast_1d(z0)
        else:
            z0, valf = fzzeroabs2(lambda kz: FMMshooting(kz, FMMpars)[0], kz1[m], kz2[m])
            z0 = numpy.atleast_1d(z0)
            
        if len(z0) > 0:
            if TETM == 'TM':
                zero, modo = FMMshootingTM(z0, FMMpars)
            else:
                zero, modo = FMMshooting(z0, FMMpars)

            if TETM == 'TM':
                normalizzazione = 1. / numpy.sqrt(scalarprod(modo, sueps(modo)))
            else:
                normalizzazione = 1. / numpy.sqrt(scalarprod(modo, modo))

            modo.sl *= normalizzazione
            modo.al *= normalizzazione
            modo.sr *= normalizzazione
            modo.ar *= normalizzazione

            mk += 1
            modi.append(modo)
    
    return modi

def script1d(Ux, Uy, refindex, wl, boundary, nmodislices, verbosity=0):
    
    nx = refindex.shape[0]
    slices = []
    for m in range(nx):
        Message('Finding 1dmodes TE.', 1).show(verbosity)
        ymodih = FMM1d_y(Uy, refindex[m,:], wl, nmodislices, boundary.yh, 'TE', verbosity)
        Message('Finding 1dmodes TM.', 1).show(verbosity)
        ymodie = FMM1d_y(Uy, refindex[m,:], wl, nmodislices, boundary.ye, 'TM', verbosity)
        slice = Slice(x1=Ux[m], x2=Ux[m+1], Uy=Uy, boundary=boundary, modie=ymodie, modih=ymodih)
        # OKKIO: do I really need them?
        slice.Ux = Ux
        slice.refractiveindex = refindex
        slice.epsilon = refindex**2
        slice.wl = wl
        slices.append(slice)
    return slices

def dot(modo):

    k = modo.k    
    mododot = copy.deepcopy(modo)
    mododot.sl = modo.al
    mododot.sr = modo.ar
    mododot.al = -k**2 * modo.sl
    mododot.ar = -k**2 * modo.sr    
    return mododot
    
def genera_rotazione(slices):
    
    nmodi = len(slices[0].modih)
    k0 = 2 * numpy.pi / slices[0].wl
    Nslices = len(slices);

    R = Struct()
    # alloc R
    R.Ree = numpy.zeros((nmodi, nmodi, Nslices-1), dtype=complex)
    R.Reem = numpy.zeros_like(R.Ree)
    R.Rhh = numpy.zeros_like(R.Ree)
    R.Rhhm = numpy.zeros_like(R.Ree)
    R.Rhe = numpy.zeros_like(R.Ree)
    R.Rhem = numpy.zeros_like(R.Ree)

    for idx in range(len(slices) - 1):
        slice = slices[idx]
        slicep1 = slices[idx + 1]

        for n in range(nmodi):
            Fhn = slice.modih[n]
            Fp1hn = slicep1.modih[n]
            Fen = slice.modie[n]
            Fp1en = slicep1.modie[n]
            Fhndot = dot(Fhn)
            Fp1hndot = dot(Fp1hn)
            khidx = slice.modih[n].keff
            khidxp1 = slicep1.modih[n].keff

            for m in range(nmodi):
                Fem = slice.modie[m]
                Fhm = slice.modih[m]
                Fp1em = slicep1.modie[m]
                Fp1hm = slicep1.modih[m]
                Femsueps = sueps(Fem)
                Femdotsueps = dot(Femsueps)
                Fp1emsueps = sueps(Fp1em)
                Fp1emdotsueps = dot(Fp1emsueps)
                keidx = slice.modie[m].keff
                keidxp1 = slicep1.modie[m].keff

                R.Ree[n, m, idx] = scalarprod(Fen, Fp1emsueps)
                R.Reem[n, m, idx] = scalarprod(Fp1en, Femsueps)
                R.Rhh[n, m, idx] = scalarprod(Fhn, Fp1hm)
                R.Rhhm[n, m, idx] = scalarprod(Fp1hn, Fhm)
                  
                s1 = k0 * scalarprod(Fhndot, Fp1emsueps) / khidx**2
                s2 = k0 * scalarprod(Fhn, Fp1emdotsueps) / keidxp1**2
                R.Rhe[n, m, idx] = (s1 + s2).item()

                s1 = k0 * scalarprod(Fp1hndot, Femsueps) / khidxp1**2
                s2 = k0 * scalarprod(Fp1hn, Femdotsueps) / keidx**2
                R.Rhem[n, m, idx] = (s1 + s2).item()
                
    return R

def ortonormalita(slices):

    nmodi = len(slices[0].modih)
    k0 = 2 * numpy.pi / slices[0].wl
    Nslices = len(slices);

    neesueps = numpy.zeros(Nslices, dtype=complex)
    nhh = numpy.zeros_like(neesueps)
    nRhe = numpy.zeros_like(neesueps)
    nRee = numpy.zeros_like(neesueps)
    nRhh = numpy.zeros_like(neesueps)
    nAC = numpy.zeros_like(neesueps)

    M = Struct()
    M.ee = numpy.zeros((nmodi, nmodi, Nslices), dtype=complex)
    M.eesueps = numpy.zeros_like(M.ee)
    M.hh = numpy.zeros_like(M.ee)
    M.Rhe = numpy.zeros_like(M.ee)

    for idx, slice in enumerate(slices):
        
        for n in range(nmodi):
            Fhn = slice.modih[n]
            Fen = slice.modie[n]
            khidx = slice.modih[n].keff

            for m in range(nmodi):
                Fem = slice.modie[m]
                Fhm = slice.modih[m]
                keidxp1 = slice.modie[m].keff

                M.ee[n, m, idx] = scalarprod(Fen, Fem)
                M.eesueps[n, m, idx] = scalarprod(Fen, sueps(Fem))
                M.hh[n, m, idx] = scalarprod(Fhn, Fhm)
                
                Fp1em = slice.modie[m]
                s1 = k0 * scalarprod(dot(Fhn), sueps(Fp1em)) / khidx**2
                s2 = k0 * scalarprod(Fhn, sueps(dot(Fp1em))) / keidxp1**2
                M.Rhe[n, m, idx] = (s1 + s2).item()
                
    R = genera_rotazione(slices)
    Ident = numpy.eye(nmodi)
    for idx in range(Nslices):
        neesueps[idx] = numpy.linalg.norm(M.eesueps[:,:,idx] - Ident)
        nhh[idx] = numpy.linalg.norm(M.hh[:,:,idx] - Ident)
        nRhe[idx] = numpy.linalg.norm(M.Rhe[:,:,idx])
        
    for idx in range(Nslices-1):
        nRee[idx] = numpy.linalg.norm(numpy.dot(R.Ree[:,:,idx], R.Reem[:,:,idx]) - Ident)
        nRhh[idx] = numpy.linalg.norm(numpy.dot(R.Rhh[:,:,idx], R.Rhhm[:,:,idx]) - Ident)
        nAC[idx] = numpy.linalg.norm(numpy.dot(R.Rhe[:,:,idx], R.Reem[:,:,idx]) + 
                                     numpy.dot(R.Rhh[:,:,idx], R.Rhem[:,:,idx]))
    
    ns1 = numpy.linalg.norm(numpy.r_[neesueps, nhh, nRhe])
    ns2 = numpy.linalg.norm(numpy.r_[nRee, nRhh, nAC])
    errore = numpy.linalg.norm(numpy.r_[ns1, ns2]) / scipy.sqrt(8 * nmodi)
    return errore

def method_of_component(kz_, slices, Rot, uscelto=None, icomp=None):
    
    kz = numpy.atleast_1d(kz_)
    abscomp = numpy.zeros(len(kz))
##    tmp = 500 # OKKIO: perche' 500?
    tmp = 100 * len(slices[0].modie) * (len(slices) - 1) # OKKIO: dimension of Mvec * 50. enough?
    normu = numpy.zeros(tmp, dtype=complex)
    for m in range(len(kz)):
        M = Mvec(kz[m], slices, Rot)
        urn = numpy.zeros((M.shape[0], tmp), dtype=complex)
        if (uscelto is None) and (icomp is None):
            for k in range(tmp):
                numpy.random.seed()
                ur = numpy.random.rand(M.shape[0])
                urn[:,k] = ur / numpy.linalg.norm(ur)
                normu[k] = numpy.linalg.norm(numpy.dot(M, urn[:,k]))
            iurn = numpy.argmin(normu)
            uscelto = urn[:, iurn]
            icomp = numpy.argmax(uscelto)

        Mmeno1u = numpy.linalg.solve(M, uscelto)
        abscomp[m] = 1. / numpy.linalg.norm(Mmeno1u)
        
    return (abscomp, uscelto, icomp)

def creaTeThSeSh(kz, slices):
    
    Nslices = len(slices)
    nmodi = len(slices[0].modie)
    d = numpy.array([s.x2 - s.x1 for s in slices])
    k0 = 2. * numpy.pi / slices[0].wl

    Th = numpy.zeros((nmodi, Nslices), dtype=complex)
    Sh = numpy.zeros_like(Th)
    Te = numpy.zeros_like(Th)
    Se = numpy.zeros_like(Th)
    Thleft = numpy.zeros_like(Th)
    Teleft = numpy.zeros_like(Th)
    Thright = numpy.zeros_like(Th)
    Teright = numpy.zeros_like(Th)
    
    for idx in range(Nslices):
        ke = numpy.array([m.keff.item() for m in slices[idx].modie])
        kh = numpy.array([m.keff.item() for m in slices[idx].modih])

        kxh = scipy.sqrt(kh**2 - kz**2)
        kxe = scipy.sqrt(ke**2 - kz**2)
        
        Th[:,idx] = (k0/kh)**2 * kxh / numpy.tan(kxh * d[idx])
        Sh[:,idx] = (k0/kh)**2 * kxh / numpy.sin(kxh * d[idx])
        Te[:,idx] = (k0/ke)**2 * kxe / numpy.tan(kxe * d[idx])
        Se[:,idx] = (k0/ke)**2 * kxe / numpy.sin(kxe * d[idx])

    ke = numpy.array([m.keff.item() for m in slices[0].modie])
    kh = numpy.array([m.keff.item() for m in slices[0].modih])
    kxh = scipy.sqrt(kh**2 - kz**2)
    kxe = scipy.sqrt(ke**2 - kz**2)

    if slices[0].boundary.xleft == 'Electric Wall':
        # ah = 0
        Thleft = -(k0/kh)**2 * kxh * numpy.tan(kxh * d[0])
        Teleft = Te[:,0]
    else:
        # ae = 0
        Teleft = -(k0/ke)**2 * kxe * numpy.tan(kxe * d[0])
        Thleft = Th[:,0]

    ke = numpy.array([m.keff.item() for m in slices[-1].modie])
    kh = numpy.array([m.keff.item() for m in slices[-1].modih])
    kxh = scipy.sqrt(kh**2 - kz**2)
    kxe = scipy.sqrt(ke**2 - kz**2)
    
    if slices[-1].boundary.xleft == 'Electric Wall':
        # ah = 0
        Thright = -(k0/kh)**2 * kxh * numpy.tan(kxh * d[-1])
        Teright = Te[:,-1]
    else:
        # ae = 0
        Teright = -(k0/ke)**2 * kxe * numpy.tan(kxe * d[-1])
        Thright = Th[:,-1]

    return (Te, Th, Se, Sh, Teleft, Teright, Thleft, Thright)

def Mvec(kz, slices, R):

    Nslices = len(slices)
    Rhh = R.Rhh
    Ree = R.Ree
    Rhe = R.Rhe
    Rhem = R.Rhem
    Rhhm = R.Rhhm

    Te, Th, Se, Sh, Teleft, Teright, Thleft, Thright = creaTeThSeSh(kz,slices)
    Te[:,0] = Teleft
    Te[:,-1] = Teright
    Th[:,0] = Thleft
    Th[:,-1] = Thright

    # case Nslices=2
    if Nslices==2:
        raise NotImplementedError('2 slices not implemented yet.')

    dim1 = Th.shape[0]
    M = numpy.zeros((2 * dim1 * (Nslices-1), 2 * dim1 * (Nslices-1)), dtype=complex)
    Dim1 = numpy.arange(dim1)
    
    for idx in range(3, Nslices):
        
        idxeJA = (2 * idx - 4) * dim1
        idxeJB = (2 * idx - 6) * dim1
        idxeJC = (2 * idx - 2) * dim1
        idxeJD = (2 * idx - 3) * dim1
        
        idxeIA = (2 * idx - 4) * dim1
        idxeIB = (2 * idx - 4) * dim1
        idxeIC = (2 * idx - 4) * dim1 
        idxeID = (2 * idx - 4) * dim1
        
        idxhJA = (2 * idx - 3) * dim1
        idxhJB = (2 * idx - 5) * dim1
        idxhJC = (2 * idx - 1) * dim1
        idxhJD = (2 * idx - 4) * dim1
        
        idxhIA = (2 * idx - 3) * dim1
        idxhIB = (2 * idx - 3) * dim1
        idxhIC = (2 * idx - 3) * dim1
        idxhID = (2 * idx - 3) * dim1
        
        IAe = Dim1 + idxeIA
        JAe = Dim1 + idxeJA
        IBe = Dim1 + idxeIB
        JBe = Dim1 + idxeJB
        ICe = Dim1 + idxeIC
        JCe = Dim1 + idxeJC
        IDe = Dim1 + idxeID
        JDe = Dim1 + idxeJD
        
        IAh = Dim1 + idxhIA
        JAh = Dim1 + idxhJA
        IBh = Dim1 + idxhIB
        JBh = Dim1 + idxhJB
        ICh = Dim1 + idxhIC
        JCh = Dim1 + idxhJC
        IDh = Dim1 + idxhID
        JDh = Dim1 + idxhJD
        
        M[numpy.ix_(IAe, JAe)] = numpy.dot(Ree[:,:,idx-1].T * Te[:,idx-1], Ree[:,:,idx-1]) + numpy.diag(Te[:,idx])
        M[numpy.ix_(IBe, JBe)] = -Ree[:,:,idx-1].T * Se[:,idx-1]
        M[numpy.ix_(ICe, JCe)] = -Se[:,numpy.newaxis,idx] * Ree[:,:,idx]
        M[numpy.ix_(IDe, JDe)] = -kz * numpy.dot(Ree[:,:,idx-1].T, Rhem[:,:,idx-1].T)
        
        M[numpy.ix_(IAh, JAh)] = numpy.dot(Rhhm[:,:,idx-1] * Th[:,idx-1], Rhh[:,:,idx-1]) + numpy.diag(Th[:,idx])
        M[numpy.ix_(IBh, JBh)] = -Rhhm[:,:,idx-1] * Sh[:,idx-1]
        M[numpy.ix_(ICh, JCh)] = -Sh[:,numpy.newaxis,idx] * Rhh[:,:,idx]
        M[numpy.ix_(IDh, JDh)] = kz * numpy.dot(Rhhm[:,:,idx-1], Rhe[:,:,idx-1])

    idx = 2

    idxeJA = (2 * idx - 4) * dim1
    idxeJC = (2 * idx - 2) * dim1
    idxeJD = (2 * idx - 3) * dim1
    
    idxeIA = (2 * idx - 4) * dim1
    idxeIC = (2 * idx - 4) * dim1 
    idxeID = (2 * idx - 4) * dim1
    
    idxhJA = (2 * idx - 3) * dim1
    idxhJC = (2 * idx - 1) * dim1
    idxhJD = (2 * idx - 4) * dim1
    
    idxhIA = (2 * idx - 3) * dim1
    idxhIC = (2 * idx - 3) * dim1
    idxhID = (2 * idx - 3) * dim1
    
    IAe = Dim1 + idxeIA
    JAe = Dim1 + idxeJA
    ICe = Dim1 + idxeIC
    JCe = Dim1 + idxeJC
    IDe = Dim1 + idxeID
    JDe = Dim1 + idxeJD
    
    IAh = Dim1 + idxhIA
    JAh = Dim1 + idxhJA
    ICh = Dim1 + idxhIC
    JCh = Dim1 + idxhJC
    IDh = Dim1 + idxhID
    JDh = Dim1 + idxhJD
    
    idx -= 1
    
    M[numpy.ix_(IAe, JAe)] = numpy.dot(Ree[:,:,idx-1].T * Te[:,idx-1], Ree[:,:,idx-1]) + numpy.diag(Te[:,idx])
    M[numpy.ix_(ICe, JCe)] = -Se[:,numpy.newaxis,idx] * Ree[:,:,idx]
    M[numpy.ix_(IDe, JDe)] = -kz * numpy.dot(Ree[:,:,idx-1].T, Rhem[:,:,idx-1].T)
    
    M[numpy.ix_(IAh, JAh)] = numpy.dot(Rhhm[:,:,idx-1] * Th[:,idx-1], Rhh[:,:,idx-1]) + numpy.diag(Th[:,idx])
    M[numpy.ix_(ICh, JCh)] = -Sh[:,numpy.newaxis,idx] * Rhh[:,:,idx]
    M[numpy.ix_(IDh, JDh)] = kz * numpy.dot(Rhhm[:,:,idx-1], Rhe[:,:,idx-1])

    idx = Nslices
    
    idxeJA = (2 * idx - 4) * dim1
    idxeJB = (2 * idx - 6) * dim1
    idxeJD = (2 * idx - 3) * dim1
    
    idxeIA = (2 * idx - 4) * dim1
    idxeIB = (2 * idx - 4) * dim1
    idxeID = (2 * idx - 4) * dim1
    
    idxhJA = (2 * idx - 3) * dim1
    idxhJB = (2 * idx - 5) * dim1
    idxhJD = (2 * idx - 4) * dim1
    
    idxhIA = (2 * idx - 3) * dim1
    idxhIB = (2 * idx - 3) * dim1
    idxhID = (2 * idx - 3) * dim1
    
    IAe = Dim1 + idxeIA
    JAe = Dim1 + idxeJA
    IBe = Dim1 + idxeIB
    JBe = Dim1 + idxeJB
    IDe = Dim1 + idxeID
    JDe = Dim1 + idxeJD
    
    IAh = Dim1 + idxhIA
    JAh = Dim1 + idxhJA
    IBh = Dim1 + idxhIB
    JBh = Dim1 + idxhJB
    IDh = Dim1 + idxhID
    JDh = Dim1 + idxhJD
    
    idx -= 1
    
    M[numpy.ix_(IAe, JAe)] = numpy.dot(Ree[:,:,idx-1].T * Te[:,idx-1], Ree[:,:,idx-1]) + numpy.diag(Te[:,idx])
    M[numpy.ix_(IBe, JBe)] = -Ree[:,:,idx-1].T * Se[:,idx-1]
    M[numpy.ix_(IDe, JDe)] = -kz * numpy.dot(Ree[:,:,idx-1].T, Rhem[:,:,idx-1].T)
    
    M[numpy.ix_(IAh, JAh)] = numpy.dot(Rhhm[:,:,idx-1] * Th[:,idx-1], Rhh[:,:,idx-1]) + numpy.diag(Th[:,idx])
    M[numpy.ix_(IBh, JBh)] = -Rhhm[:,:,idx-1] * Sh[:,idx-1]
    M[numpy.ix_(IDh, JDh)] = kz * numpy.dot(Rhhm[:,:,idx-1], Rhe[:,:,idx-1])

    return M

def check_matching(kz, slices, modo, R):

    Te, Th, Se, Sh, Teleft, Teright, Thleft, Thright = creaTeThSeSh(kz, slices)
    Shr = numpy.array([m.sr for m in modo.modih])
    Shl = numpy.array([m.sl for m in modo.modih])
    Ser = numpy.array([m.sr for m in modo.modie])
    Sel = numpy.array([m.sl for m in modo.modie])
    Ahr = numpy.array([m.ar for m in modo.modih])
    Ahl = numpy.array([m.al for m in modo.modih])
    Aer = numpy.array([m.ar for m in modo.modie])
    Ael = numpy.array([m.al for m in modo.modie])
    
    kz = modo.keff
    
    n1 = numpy.linalg.norm(numpy.dot(R.Rhh[:,:,1], Shl[:,2]) - Shr[:,1])
    n2 = numpy.linalg.norm(numpy.dot(R.Rhh[:,:,0], Shl[:,1]) - Shr[:,0])
    n3 = numpy.linalg.norm(numpy.dot(R.Ree[:,:,1], Sel[:,2]) - Ser[:,1])
    n4 = numpy.linalg.norm(numpy.dot(R.Ree[:,:,0], Sel[:,1]) - Ser[:,0])
    n5 = numpy.linalg.norm(numpy.dot(R.Rhh[:,:,1], Ahl[:,2]) - kz * numpy.dot(R.Rhe[:,:,1], Sel[:,2]) - Ahr[:,1])
    n6 = numpy.linalg.norm(numpy.dot(R.Rhh[:,:,0], Ahl[:,1]) - kz * numpy.dot(R.Rhe[:,:,0], Sel[:,1]) - Ahr[:,0])
    n7 = numpy.linalg.norm(numpy.dot(R.Reem[:,:,1].T, Ael[:,2]) + kz * numpy.dot(R.Rhem[:,:,1].T, Shl[:,2]) - Aer[:,1])
    n8 = numpy.linalg.norm(numpy.dot(R.Reem[:,:,0].T, Ael[:,1]) + kz * numpy.dot(R.Rhem[:,:,0].T, Shl[:,1]) - Aer[:,0])
    n9 = numpy.linalg.norm(-Te[:,0] * Sel[:,0] + Se[:,0] * Ser[:,0] - Ael[:,0])
    n10 = numpy.linalg.norm(-Te[:,1] * Sel[:,1] + Se[:,1] * Ser[:,1] - Ael[:,1])
    n11 = numpy.linalg.norm(-Te[:,2] * Sel[:,2] + Se[:,2] * Ser[:,2] - Ael[:,2])
    n12 = numpy.linalg.norm(-Th[:,0] * Shl[:,0] + Sh[:,0] * Shr[:,0] - Ahl[:,0])
    n13 = numpy.linalg.norm(-Th[:,1] * Shl[:,1] + Sh[:,1] * Shr[:,1] - Ahl[:,1])
    n14 = numpy.linalg.norm(-Th[:,2] * Shl[:,2] + Sh[:,2] * Shr[:,2] - Ahl[:,2])
    n15 = numpy.linalg.norm(-Sh[:,0] * Shl[:,0] + Th[:,0] * Shr[:,0] - Ahr[:,0])
    n16 = numpy.linalg.norm(-Sh[:,1] * Shl[:,1] + Th[:,1] * Shr[:,1] - Ahr[:,1])
    n17 = numpy.linalg.norm(-Sh[:,2] * Shl[:,2] + Th[:,2] * Shr[:,2] - Ahr[:,2])
    n18 = numpy.linalg.norm(-Se[:,0] * Sel[:,0] + Te[:,0] * Ser[:,0] - Aer[:,0])
    n19 = numpy.linalg.norm(-Se[:,1] * Sel[:,1] + Te[:,1] * Ser[:,1] - Aer[:,1])
    n20 = numpy.linalg.norm(-Se[:,2] * Sel[:,2] + Te[:,2] * Ser[:,2] - Aer[:,2])

    Nv = numpy.array([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20])
    N = numpy.linalg.norm(Nv);
    return N

def creacoeffx3(kz, solution, slices, R):

    Nslices = len(slices)

    xl = slices[0].boundary.xleft
    xr = slices[0].boundary.xright

    nmodi = len(slices[0].modih)
    Rhh = R.Rhh
    Ree = R.Ree
    Rhe = R.Rhe
    Rhem = R.Rhem
    Reem = R.Reem

    Te, Th, Se, Sh, Teleft, Teright, Thleft, Thright = creaTeThSeSh(kz, slices)

    ##sl2end = numpy.reshape(solution, (nmodi, 2 * (Nslices - 1)))
    sl2end = numpy.reshape(solution, (2 * (Nslices - 1), nmodi)).T
    idxslices = 2 * numpy.arange((Nslices-1))
    sle2end = sl2end[:, idxslices]
    slh2end = sl2end[:, idxslices + 1]

    ale = numpy.zeros((nmodi,Nslices), dtype=complex);
    sre = numpy.zeros_like(ale)
    are = numpy.zeros_like(ale)
    alh = numpy.zeros_like(ale)
    srh = numpy.zeros_like(ale)
    arh = numpy.zeros_like(ale)

    if (xl == 'Electric Wall') & (xr == 'Electric Wall'):
        
        sle = numpy.c_[numpy.zeros((nmodi,1)), sle2end]
        sre[:,-1] = numpy.zeros(nmodi)
        are[:,-1] = -Se[:,-1] * sle[:,-1]
        ale[:,-1] = -Te[:,-1] * sle[:,-1]

        slh = numpy.c_[numpy.zeros((nmodi,1)), slh2end]
        arh[:,-1] = numpy.zeros(nmodi)
        srh[:,-1] = Sh[:,-1] / Th[:,-1] * slh[:,-1]
        alh[:,-1] = -Th[:,-1] * slh[:,-1] + Sh[:,-1] * srh[:,-1]

        for idx in range(1, Nslices):
            sre[:,idx-1] = numpy.dot(Ree[:,:,idx-1], sle[:,idx])
            srh[:,idx-1] = numpy.dot(Rhh[:,:,idx-1], slh[:,idx])
            
        slh[:,0] = Sh[:,0] / Th[:,0] * srh[:,0]

        for idx in range(Nslices - 1, 0, -1):
            are[:,idx-1] = numpy.dot(Reem[:,:,idx-1].T, ale[:,idx]) + kz * numpy.dot(Rhem[:,:,idx-1].T, slh[:,idx])
            arh[:,idx-1] = numpy.dot(Rhh[:,:,idx-1], alh[:,idx]) - kz * numpy.dot(Rhe[:,:,idx-1], sle[:,idx])
            ale[:,idx-1] = -Te[:,idx-1] * sle[:,idx-1] + Se[:,idx-1] * sre[:,idx-1]
            alh[:,idx-1] = -Th[:,idx-1] * slh[:,idx-1] + Sh[:,idx-1] * srh[:,idx-1]

    elif (xl == 'Electric Wall') & (xr == 'Magnetic Wall'):

        sle = numpy.c_[numpy.zeros((nmodi,1)), sle2end]
        are[:,-1] = numpy.zeros(nmodi)
        sre[:,-1] = Se[:,-1] / Te[:,-1] * sle[:,-1]
        ale[:,-1] = -Te[:,-1] * sle[:,-1] + Se[:,-1] * sre[:,-1]

        slh = numpy.c_[numpy.zeros((nmodi,1)), slh2end]
        srh[:,-1] = numpy.zeros(nmodi)
        arh[:,-1] = -Sh[:,-1] * slh[:,-1]
        alh[:,-1] = -Th[:,-1] * slh[:,-1]

        for idx in range(1, Nslices):
            sre[:,idx-1] = numpy.dot(Ree[:,:,idx-1], sle[:,idx])
            srh[:,idx-1] = numpy.dot(Rhh[:,:,idx-1], slh[:,idx])

        slh[:,0] = Sh[:,0] / Th[:,0] * srh[:,0]

        for idx in range(Nslices - 1, 0, -1):
            are[:,idx-1] = numpy.dot(Reem[:,:,idx-1].T, ale[:,idx]) + kz * numpy.dot(Rhem[:,:,idx-1].T, slh[:,idx])
            arh[:,idx-1] = numpy.dot(Rhh[:,:,idx-1], alh[:,idx])- kz * numpy.dot(Rhe[:,:,idx-1], sle[:,idx])
            ale[:,idx-1] = -Te[:,idx-1] * sle[:,idx-1] + Se[:,idx-1] * sre[:,idx-1]
            alh[:,idx-1] = -Th[:,idx-1] * slh[:,idx-1] + Sh[:,idx-1] * srh[:,idx-1]

    elif (xl == 'Magnetic Wall') & (xr == 'Electric Wall'):

        sle = numpy.c_[numpy.zeros((nmodi,1)), sle2end]
        slh = numpy.c_[numpy.zeros((nmodi,1)), slh2end]

        sre[:,-1] = numpy.zeros(nmodi);
        arh[:,-1] = numpy.zeros(nmodi);
        srh[:,-1] = Sh[:,-1] / Th[:,-1] * slh[:,-1]
        are[:,-1] = -Se[:,-1] * sle[:,-1]
        ale[:,-1] = -Te[:,-1] * sle[:,-1]
        alh[:,-1] = -Th[:,-1] * slh[:,-1] + Sh[:,-1] * srh[:,-1]

        for idx in range(1, Nslices):
            sre[:,idx-1] = numpy.dot(Ree[:,:,idx-1], sle[:,idx])
            srh[:,idx-1] = numpy.dot(Rhh[:,:,idx-1], slh[:,idx])

        sle[:,0] = Se[:,0] / Te[:,0] * sre[:,0]

        for idx in range(Nslices - 1, 0, -1):
            are[:,idx-1] = numpy.dot(Reem[:,:,idx-1].T, ale[:,idx]) + kz * numpy.dot(Rhem[:,:,idx-1].T, slh[:,idx])
            arh[:,idx-1] = numpy.dot(Rhh[:,:,idx-1], alh[:,idx]) - kz * numpy.dot(Rhe[:,:,idx-1], sle[:,idx])
            ale[:,idx-1] = -Te[:,idx-1] * sle[:,idx-1] + Se[:,idx-1] * sre[:,idx-1]
            alh[:,idx-1] = -Th[:,idx-1] * slh[:,idx-1] + Sh[:,idx-1] * srh[:,idx-1]

    elif (xl == 'Magnetic Wall') & (xr == 'Magnetic Wall'):

        sle = numpy.c_[numpy.zeros((nmodi,1)), sle2end]
        slh = numpy.c_[numpy.zeros((nmodi,1)), slh2end]

        srh[:,-1] = numpy.zeros(nmodi)
        are[:,-1] = numpy.zeros(nmodi)
        sre[:,-1] = Se[:,-1] / Te[:,-1] * sle[:,-1]
        arh[:,-1] = -Sh[:,-1] * slh[:,-1]
        alh[:,-1] = -Th[:,-1] * slh[:,-1]
        ale[:,-1] = -Te[:,-1] * sle[:,-1] + Se[:,-1] * sre[:,-1]

        for idx in range(1, Nslices):
            sre[:,idx-1] = numpy.dot(Ree[:,:,idx-1], sle[:,idx])
            srh[:,idx-1] = numpy.dot(Rhh[:,:,idx-1], slh[:,idx])
        
        sle[:,0] = Se[:,0] / Te[:,0] * sre[:,0]

        for idx in range(Nslices - 1, 0, -1):
            are[:,idx-1] = numpy.dot(Reem[:,:,idx-1].T, ale[:,idx]) + kz * numpy.dot(Rhem[:,:,idx-1].T, slh[:,idx])
            arh[:,idx-1] = numpy.dot(Rhh[:,:,idx-1], alh[:,idx]) - kz * numpy.dot(Rhe[:,:,idx-1], sle[:,idx])
            ale[:,idx-1] = -Te[:,idx-1] * sle[:,idx-1] + Se[:,idx-1] * sre[:,idx-1]
            alh[:,idx-1] = -Th[:,idx-1] * slh[:,idx-1] + Sh[:,idx-1] * srh[:,idx-1]

    else:
        raise ValueError('Unrecognized boundary condition')

    modie = []
    modih = []
    kh = numpy.zeros((nmodi, Nslices), dtype=complex)
    ke = numpy.zeros_like(kh)
    
    for n in range(nmodi):
        
        for q, slice in enumerate(slices):
            kh[n,q] = slice.modih[n].keff.item()
            ke[n,q] = slice.modie[n].keff.item()
        
        modoe  = FMMMode1dx()
        modoe.sl = sle[n,:]
        modoe.al = ale[n,:]
        modoe.sr = sre[n,:]
        modoe.ar = are[n,:]
        modoe.U = slices[0].Ux
        modoe.k = ke[n,:]
        modie.append(modoe)

        modoh = FMMMode1dx()
        modoh.sl = slh[n,:]
        modoh.al = alh[n,:]
        modoh.sr = srh[n,:]
        modoh.ar = arh[n,:]
        modoh.U = slices[0].Ux
        modoh.k = kh[n,:]
        modih.append(modoh)
        
    return (modie, modih)

def FMM1d_x_component(slices, nmodi, verbosity=0):

    Message('Matching 1dmodes on slices interfaces.', 1).show(verbosity)
    
    TolnullEig = 1e-1
    searchinterval = 6e-3

    Ux = slices[0].Ux
    lambda0 = slices[0].wl
    k0 = 2. * numpy.pi / lambda0

    Rot = genera_rotazione(slices)

    maxrekz = -1e6
    minrekz = 1e6
    
    for slice in slices:
        maxrekz0 = numpy.max(numpy.real([m.keff for m in slice.modih] + [m.keff for m in slice.modie]))
        minrekz0 = numpy.min(numpy.real(k0 * slice.refractiveindex))
        if maxrekz < maxrekz0:
            maxrekz = maxrekz0
        if minrekz > minrekz0:
            minrekz = minrekz0

    # only guided modes
##    rekz = numpy.linspace(maxrekz + searchinterval, minrekz - searchinterval, numpy.floor(3. / searchinterval))
    # also radiative modes
    # increase 10./searchinterval if not enough candidate zeros are found
##    rekz = numpy.linspace(maxrekz + searchinterval, 0., 
##                          numpy.floor(6. / searchinterval * 
##                                      (1 + (minrekz - searchinterval) / (maxrekz - minrekz + 2 * searchinterval))))
    # make rekz[1]-rekz[0] <= searchinterval
    # OKKIO: forse mettere 3x punti??
    rekz = numpy.linspace(maxrekz + searchinterval, 0., 10. * numpy.floor((maxrekz + searchinterval) / searchinterval))
    
    abscomp, uscelto, icomp = method_of_component(maxrekz, slices, Rot)
    matchingre = method_of_component(rekz, slices, Rot, uscelto, icomp)[0]
    nre = rekz / k0
    zerire = findzerosnew(nre, numpy.abs(matchingre), searchinterval)[0]
    kz1 = k0 * zerire - searchinterval / 2. + 1j * searchinterval
    kz2 = k0 * zerire + searchinterval / 2. - 1j * searchinterval
    Message('Found %d possible zeros.' % len(zerire), 1).show(verbosity)
    Message('Possible zeros:\n%s' % zerire, 3).show(verbosity)
    Message('Refining zeros.', 1).show(verbosity)
    mk = 0
    modi = []
    for m in range(len(kz1)):

        if mk == nmodi:
            break
        
        abscomp, uscelto, icomp = method_of_component(numpy.mean([kz1[m], kz2[m]]), slices, Rot)
        z0, valf = fzzeroabs2(lambda kz: method_of_component(kz, slices, Rot, uscelto, icomp)[0], kz1[m], kz2[m])

        if len(valf) > 0:

            eigval, eigvec = numpy.linalg.eig(Mvec(z0, slices, Rot))
            mk += 1
            index0 = numpy.where(numpy.abs(eigval) < TolnullEig)[0]
            if len(index0) > 1:
                Message('Two eigenvectors found with the same eigenvalue. Mixing of modes may result', 1).show(verbosity)
                
            imm = numpy.argmin(numpy.abs(eigval))
            solution = eigvec[:, imm]
            modie, modih = creacoeffx3(z0, solution, slices, Rot)
            modo = FMMMode2d()
            modo.keff = z0
            modo.neff = modo.keff * lambda0 / (2 * numpy.pi)
            modo.modie = modie
            modo.modih = modih
            modo.zero = numpy.linalg.norm(numpy.dot(Mvec(z0, slices, Rot), solution))
            modo.distance_from_zero = check_matching(z0, slices, modo, Rot)
            modo.slicesx = slices
            modi.append(modo)
            
            Message('Distance from zero of the final solution: %f' % modo.distance_from_zero, 2).show(verbosity);
            Message('%d. neff = %s' % (mk, modo.keff / k0), 1).show(verbosity)

    return modi
