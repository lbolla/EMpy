"""Geometries.

"""
from builtins import zip
from builtins import object

import numpy
import EMpy_gpu.utils
import EMpy_gpu.modesolvers
from EMpy_gpu.modesolvers.interface import *
import pylab

def S2T(S):
    dim = S.shape[0] / 2.
    s11 = S[:dim, :dim]
    s22 = S[dim:, dim:]
    s12 = S[:dim, dim:]
    s21 = S[dim:, :dim]
    T = numpy.zeros_like(S)
    s12_1 = numpy.linalg.inv(s12)
    T[:dim, :dim] = s21 - numpy.dot(numpy.dot(s22, s12_1), s11)
    T[dim:, dim:] = s12_1
    T[:dim, dim:] = numpy.dot(s22, s12_1)
    T[dim:, :dim] = -numpy.dot(s12_1, s11)
    return T

def T2S(T):
    dim = T.shape[0] / 2.
    t11 = T[:dim, :dim]
    t22 = T[dim:, dim:]
    t12 = T[:dim, dim:]
    t21 = T[dim:, :dim]
    S = numpy.zeros_like(T)
    t22_1 = numpy.linalg.inv(t22)
    S[:dim, :dim] = -numpy.dot(t22_1, t21)
    S[dim:, dim:] = numpy.dot(t12, t22_1)
    S[:dim, dim:] = t22_1
    S[dim:, :dim] = t11 - numpy.dot(numpy.dot(t12, t22_1), t21)
    return S    

class SWG(object):
    
##    def __init__(self, cs, solver, length, inputLHS=None, inputRHS=None):
    def __init__(self, solver, length, inputLHS=None, inputRHS=None):
##        self.cs = cs
        self.solver = solver
        self.length = length
        self.build_matrix()
        self.compute_output(inputLHS, inputRHS)
    
    def compute_output(self, inputLHS=None, inputRHS=None):
        if inputLHS is None:
            self.inputLHS = numpy.zeros(self.solver.nmodes)
            self.inputLHS[0] = 1.
        else:
            self.inputLHS = inputLHS
        if inputRHS is None:
            self.inputRHS = numpy.zeros(self.solver.nmodes)
        else:
            self.inputRHS = inputRHS
        input = numpy.r_[self.inputLHS, self.inputRHS]
        output = numpy.dot(self.S, input)
        self.outputLHS = output[:self.solver.nmodes]
        self.outputRHS = output[self.solver.nmodes:]
    
    def build_matrix(self):
        neffs = numpy.array([m.neff for m in self.solver.modes])
        betas = 2 * numpy.pi / self.solver.wl * neffs
        neigs = self.solver.nmodes
        T = numpy.zeros((2 * neigs, 2 * neigs), dtype=complex)
        T[:neigs, :neigs] = numpy.diag(numpy.exp( 1j * betas * self.length))
        T[neigs:, neigs:] = numpy.diag(numpy.exp(-1j * numpy.conj(betas) * self.length))
        self.T = T
        self.S = T2S(self.T)
        
    def plot(self, sumx=1, nxy=100, nz=100, z0=0):
        if sumx is None: # sum in y
            x = self.solver.modes[0].get_y(nxy)
            axis = 0
        else: # sum in x
            x = self.solver.modes[0].get_x(nxy)
            axis = 1
        z = numpy.linspace(0, self.length, nz)
        const_z = numpy.ones_like(z)
        f = numpy.zeros((len(x), len(z)), dtype=complex)
        for im, (coeffLHS,coeffRHS) in enumerate(zip(self.inputLHS, self.inputRHS)):
            m = self.solver.modes[im]
            beta = 2 * numpy.pi / self.solver.wl * m.neff
            tmp = numpy.sum(m.intensity(x, x), axis=axis)
            f += numpy.abs(coeffLHS)**2 * tmp[:, numpy.newaxis] * const_z + \
                 numpy.abs(coeffRHS)**2 * tmp[:, numpy.newaxis] * const_z
        pylab.hot()
        pylab.contourf(x, z0 + z, numpy.abs(f).T, 16)
        
class SimpleJoint(object):
    
##    def __init__(self, cs1, cs2, solver1, solver2, inputLHS=None, inputRHS=None):
    def __init__(self, solver1, solver2, inputLHS=None, inputRHS=None):
##        self.cs1 = cs1
        self.solver1 = solver1
##        self.cs2 = cs2
        self.solver2 = solver2
        self.length = 0.
        self.build_matrix()
        self.compute_output(inputLHS, inputRHS)
    
    def compute_output(self, inputLHS=None, inputRHS=None):
        if inputLHS is None:
            self.inputLHS = numpy.zeros(self.solver1.nmodes)
            self.inputLHS[0] = 1.
        else:
            self.inputLHS = inputLHS
        if inputRHS is None:
            self.inputRHS = numpy.zeros(self.solver1.nmodes)
        else:
            self.inputRHS = inputRHS
        input = numpy.r_[self.inputLHS, self.inputRHS]
        output = numpy.dot(self.S, input)
        self.outputLHS = output[:self.solver1.nmodes]
        self.outputRHS = output[self.solver1.nmodes:]
    
    def build_matrix(self):
        O11, O22, O12, O21 = interface_matrix(self.solver1, self.solver2)
        neffA = numpy.array([m.neff for m in self.solver1.modes])
        neffB = numpy.array([m.neff for m in self.solver2.modes])
        betaA = 2 * numpy.pi / self.solver1.wl * neffA
        betaB = 2 * numpy.pi / self.solver1.wl * neffB

        BetaA = numpy.diag(betaA)
        BetaB = numpy.diag(betaB)

##        MRA1 = numpy.dot(O12, BetaA + BetaB)
##        MRA2 = numpy.dot(O12, BetaA - BetaB)
##        MRA = numpy.dot(numpy.linalg.inv(MRA1), MRA2)
##        S11 = MRA
##
##        MRB1 = numpy.dot(O21, BetaA + BetaB)
##        MRB2 = numpy.dot(O21, BetaB - BetaA)
##        MRB = numpy.dot(numpy.linalg.inv(MRB1), MRB2)
##        S22 = MRB
##
##        MTAB1 = numpy.dot(O22, BetaA + BetaB)
##        MTAB2 = numpy.dot(O12, 2. * BetaA)
##        MTAB = numpy.dot(numpy.linalg.inv(MTAB1), MTAB2)
##        S12 = MTAB
##    
##        MTBA1 = numpy.dot(O11, BetaA + BetaB)
##        MTBA2 = numpy.dot(O21, 2. * BetaB)
##        MTBA = numpy.dot(numpy.linalg.inv(MTBA1), MTBA2)
##        S21 = MTBA

        SUM = numpy.dot(BetaA, O21) + numpy.dot(O21, BetaB)
        DIF = numpy.dot(BetaA, O21) - numpy.dot(O21, BetaB)
        A = 0.5 * numpy.dot(numpy.linalg.inv(BetaA), SUM)
        B = 0.5 * numpy.dot(numpy.linalg.inv(BetaA), DIF)
        A_1 = numpy.linalg.inv(A)
        S11 = numpy.dot(B, A_1)
        S12 = A - numpy.dot(numpy.dot(B, A_1), B)
        S21 = A_1
        S22 = -numpy.dot(A_1, B)
        
        dim = O11.shape[0]

        S = numpy.zeros((2 * dim, 2 * dim), dtype=complex)
        S[:dim, :dim] = S11
        S[dim:, dim:] = S22
        S[:dim, dim:] = S12
        S[dim:, :dim] = S21

        self.S = S
        self.T = S2T(self.S)        
        
    def plot(self, sumx=1, nxy=100, nz=100, z0=0):
        pass

class GenericDevice(object):
    
    def __init__(self, devlist, inputLHS=None, inputRHS=None):
        self.devlist = devlist
        self.build_matrix()
        self.compute_output(inputLHS, inputRHS)

    def compute_output(self, inputLHS=None, inputRHS=None):
        if inputLHS is None:
            self.inputLHS = numpy.zeros_like(self.devlist[0].inputLHS)
            self.inputLHS[0] = 1.
        else:
            self.inputLHS = inputLHS
        if inputRHS is None:
            self.inputRHS = numpy.zeros_like(self.devlist[0].inputRHS)
        else:
            self.inputRHS = inputRHS
        input = numpy.r_[self.inputLHS, self.inputRHS]
        output = numpy.dot(self.S, input)
        self.outputLHS = output[:len(self.inputLHS)]
        self.outputRHS = output[len(self.inputLHS):]
        # compute for each device
        inputLHS = self.inputLHS
        outputLHS = self.outputLHS
        dim = len(inputLHS)
        for d in self.devlist:
            d.inputLHS = inputLHS
            d.outputLHS = outputLHS
            LHS = numpy.r_[inputLHS, outputLHS]
            RHS = numpy.dot(d.T, LHS)
            d.outputRHS = RHS[:dim]
            d.inputRHS = RHS[dim:]
            inputLHS = d.outputRHS
            outputLHS = d.inputRHS
    
    def build_matrix(self):
        dim = self.devlist[0].solver.nmodes
        T = numpy.eye(2 * dim, dtype=complex)
        for d in self.devlist:
            T = numpy.dot(d.T, T)
        self.T = T
        self.S = T2S(self.T)
        
    def plot(self, sumx=1, nxy=100, nz=100, z0=0):        
        z = z0
        for d in self.devlist:
            d.plot(sumx, nxy, nz, z0=z)
            z += d.length
        pylab.ylabel('z')
        if sumx is None:
            pylab.xlabel('y')
        else:
            pylab.xlabel('x')
        pylab.axis('tight')
