"""Geometries."""

from matplotlib import pyplot as plt
import numpy

from EMpy.modesolvers.interface import interface_matrix


def S2T(S):
    dim = S.shape[0] // 2
    s11 = S[:dim, :dim]
    s22 = S[dim:, dim:]
    s12 = S[:dim, dim:]
    s21 = S[dim:, :dim]
    T = numpy.zeros_like(S)
    try:
        s12_1 = numpy.linalg.inv(s12)
    except numpy.linalg.LinAlgError:
        s12_1 = numpy.linalg.pinv(s12)
    T[:dim, :dim] = s21 - numpy.dot(numpy.dot(s22, s12_1), s11)
    T[dim:, dim:] = s12_1
    T[:dim, dim:] = numpy.dot(s22, s12_1)
    T[dim:, :dim] = -numpy.dot(s12_1, s11)
    return T


def T2S(T):
    dim = T.shape[0] // 2
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


class SWG:
    #    def __init__(self, cs, solver, length, inputLHS=None, inputRHS=None):
    def __init__(self, solver, length, inputLHS=None, inputRHS=None):
        #        self.cs = cs
        self.solver = solver
        self.length = length
        self.build_matrix()
        self.compute_output(inputLHS, inputRHS)

    def compute_output(self, inputLHS=None, inputRHS=None):
        if inputLHS is None:
            self.inputLHS = numpy.zeros(self.solver.nmodes)
            self.inputLHS[0] = 1.0
        else:
            self.inputLHS = inputLHS
        if inputRHS is None:
            self.inputRHS = numpy.zeros(self.solver.nmodes)
        else:
            self.inputRHS = inputRHS
        input = numpy.r_[self.inputLHS, self.inputRHS]
        output = numpy.dot(self.S, input)
        self.outputLHS = output[: self.solver.nmodes]
        self.outputRHS = output[self.solver.nmodes :]

    def build_matrix(self):
        neffs = numpy.array([m.neff for m in self.solver.modes])
        betas = 2 * numpy.pi / self.solver.wl * neffs
        neigs = self.solver.nmodes
        T = numpy.zeros((2 * neigs, 2 * neigs), dtype=complex)
        T[:neigs, :neigs] = numpy.diag(numpy.exp(1j * betas * self.length))
        T[neigs:, neigs:] = numpy.diag(numpy.exp(-1j * numpy.conj(betas) * self.length))
        self.T = T
        self.S = T2S(self.T)

    def plot(self, sumx=1, nxy=100, nz=100, z0=0):
        if sumx is None:  # sum in y
            x = self.solver.modes[0].get_y(nxy)
            axis = 0
        else:  # sum in x
            x = self.solver.modes[0].get_x(nxy)
            axis = 1
        z = numpy.linspace(0, self.length, nz)
        const_z = numpy.ones_like(z)
        f = numpy.zeros((len(x), len(z)), dtype=complex)
        for im, (coeffLHS, coeffRHS) in enumerate(zip(self.inputLHS, self.inputRHS)):
            m = self.solver.modes[im]
            # beta = 2 * numpy.pi / self.solver.wl * m.neff
            tmp = numpy.sum(m.intensity(x, x), axis=axis)
            f += (
                numpy.abs(coeffLHS) ** 2 * tmp[:, numpy.newaxis] * const_z
                + numpy.abs(coeffRHS) ** 2 * tmp[:, numpy.newaxis] * const_z
            )
        plt.hot()
        plt.contourf(x, z0 + z, numpy.abs(f).T, 16)


class SimpleJoint:
    #    def __init__(self, cs1, cs2, solver1, solver2, inputLHS=None, inputRHS=None):
    def __init__(self, solver1, solver2, inputLHS=None, inputRHS=None):
        #        self.cs1 = cs1
        self.solver1 = solver1
        #        self.cs2 = cs2
        self.solver2 = solver2
        self.length = 0.0
        self.build_matrix()
        self.compute_output(inputLHS, inputRHS)

    def compute_output(self, inputLHS=None, inputRHS=None):
        if inputLHS is None:
            self.inputLHS = numpy.zeros(self.solver1.nmodes)
            self.inputLHS[0] = 1.0
        else:
            self.inputLHS = inputLHS
        if inputRHS is None:
            self.inputRHS = numpy.zeros(self.solver1.nmodes)
        else:
            self.inputRHS = inputRHS
        input = numpy.r_[self.inputLHS, self.inputRHS]
        output = numpy.dot(self.S, input)
        self.outputLHS = output[: self.solver1.nmodes]
        self.outputRHS = output[self.solver1.nmodes :]

    def build_matrix(self):
        """Symmetric vector mode-matching S-matrix at a zero-length waveguide junction.

        Matching transverse E_t = sum (a + b) E_m and H_t = sum (+-a -+ b) H_m across z = 0
        and projecting against the modes of each side using O_pq = 0.5 * int (E_p x H_q^*)_z:
          Side 1: O11^T (a1 + b1) = O21^T (b2 + a2),  O11^* (a1 - b1) = O12^* (b2 - a2)
          Side 2: O22^T (b2 + a2) = O12^T (a1 + b1),  O22^* (b2 - a2) = O21^* (a1 - b1)
        Eliminating incoming/outgoing combinations yields the passive, reciprocal coupled
        system [I, -R12; -R21, I] [b1; b2] = [0, T12; T21, 0] [a1; a2], which reproduces
        exact Fresnel reflection/transmission in the homogeneous limit (including lossy media)
        and accounts for radiation loss under basis truncation without unphysical gain.
        """
        O11, O22, O12, O21 = interface_matrix(self.solver1, self.solver2)
        dim = O11.shape[0]

        E1_from_2 = numpy.linalg.solve(O11.T, O21.T)
        H1_from_2 = numpy.linalg.solve(O11.conj(), O12.conj())
        E2_from_1 = numpy.linalg.solve(O22.T, O12.T)
        H2_from_1 = numpy.linalg.solve(O22.conj(), O21.conj())

        R12 = 0.5 * (E1_from_2 - H1_from_2)
        T12 = 0.5 * (E1_from_2 + H1_from_2)
        T21 = 0.5 * (E2_from_1 + H2_from_1)
        R21 = 0.5 * (E2_from_1 - H2_from_1)

        ident = numpy.eye(dim, dtype=complex)
        zeros = numpy.zeros((dim, dim), dtype=complex)
        lhs = numpy.block([[ident, -R12], [-R21, ident]])
        rhs = numpy.block([[zeros, T12], [T21, zeros]])

        self.S = numpy.linalg.solve(lhs, rhs)
        self.T = S2T(self.S)

    def plot(self, sumx=1, nxy=100, nz=100, z0=0):
        pass


class GenericDevice:
    def __init__(self, devlist, inputLHS=None, inputRHS=None):
        self.devlist = devlist
        self.build_matrix()
        self.compute_output(inputLHS, inputRHS)

    def compute_output(self, inputLHS=None, inputRHS=None):
        if inputLHS is None:
            self.inputLHS = numpy.zeros_like(self.devlist[0].inputLHS)
            self.inputLHS[0] = 1.0
        else:
            self.inputLHS = inputLHS
        if inputRHS is None:
            self.inputRHS = numpy.zeros_like(self.devlist[0].inputRHS)
        else:
            self.inputRHS = inputRHS
        input = numpy.r_[self.inputLHS, self.inputRHS]
        output = numpy.dot(self.S, input)
        self.outputLHS = output[: len(self.inputLHS)]
        self.outputRHS = output[len(self.inputLHS) :]
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
        dim = self.devlist[0].T.shape[0] // 2
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
        plt.ylabel("z")
        if sumx is None:
            plt.xlabel("y")
        else:
            plt.xlabel("x")
        plt.axis("tight")
