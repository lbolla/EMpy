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
        T[:neigs, :neigs] = numpy.diag(numpy.exp(-1j * betas * self.length))
        T[neigs:, neigs:] = numpy.diag(numpy.exp(1j * betas * self.length))
        self.T = T
        self.S = T2S(self.T)

    def _marginal_intensity(self, sumx=1, nxy=100, nz=100, z0=0):
        m0 = self.solver.modes[0]
        x_grid = m0.get_x() if nxy is None else m0.get_x(nxy)
        y_grid = m0.get_y() if nxy is None else m0.get_y(nxy)
        z = numpy.linspace(0, self.length, nz)

        nx, ny = len(x_grid), len(y_grid)
        Ex_tot = numpy.zeros((nx, ny, nz), dtype=complex)
        Ey_tot = numpy.zeros((nx, ny, nz), dtype=complex)
        Hx_tot = numpy.zeros((nx, ny, nz), dtype=complex)
        Hy_tot = numpy.zeros((nx, ny, nz), dtype=complex)
        sz_incoherent = None

        for im, (coeffLHS, coeffRHS) in enumerate(zip(self.inputLHS, self.inputRHS)):
            if coeffLHS == 0 and coeffRHS == 0:
                continue
            m = self.solver.modes[im]
            beta = 2 * numpy.pi / self.solver.wl * m.neff
            env = coeffLHS * numpy.exp(-1j * beta * z) + coeffRHS * numpy.exp(
                1j * beta * (z - self.length)
            )
            if hasattr(m, "fields"):
                Ex, Ey, _, Hx, Hy, _ = m.fields(x_grid, y_grid)
                Ex, Ey, Hx, Hy = Ex.T, Ey.T, Hx.T, Hy.T
            elif hasattr(m, "get_field"):
                Ex = m.get_field("Ex", x_grid, y_grid)
                Ey = m.get_field("Ey", x_grid, y_grid)
                Hx = m.get_field("Hx", x_grid, y_grid)
                Hy = m.get_field("Hy", x_grid, y_grid)
            else:
                if sz_incoherent is None:
                    sz_incoherent = numpy.zeros((nx, ny, nz), dtype=float)
                sz_incoherent += (
                    numpy.real(m.intensity(x_grid, y_grid))[:, :, numpy.newaxis]
                    * numpy.abs(env)[numpy.newaxis, numpy.newaxis, :] ** 2
                )
                continue

            Ex_tot += Ex[:, :, numpy.newaxis] * env[numpy.newaxis, numpy.newaxis, :]
            Ey_tot += Ey[:, :, numpy.newaxis] * env[numpy.newaxis, numpy.newaxis, :]
            Hx_tot += Hx[:, :, numpy.newaxis] * env[numpy.newaxis, numpy.newaxis, :]
            Hy_tot += Hy[:, :, numpy.newaxis] * env[numpy.newaxis, numpy.newaxis, :]

        sz = 0.5 * numpy.real(Ex_tot * numpy.conj(Hy_tot) - Ey_tot * numpy.conj(Hx_tot))
        if sz_incoherent is not None:
            sz = sz + sz_incoherent

        if sumx is None:
            coord = y_grid
            f = numpy.trapezoid(sz, x=x_grid, axis=0)
        else:
            coord = x_grid
            f = numpy.trapezoid(sz, x=y_grid, axis=1)
        return coord, z0 + z, f

    def plot(self, sumx=1, nxy=100, nz=100, z0=0):
        if self.length == 0:
            return
        coord, z_grid, f = self._marginal_intensity(sumx=sumx, nxy=nxy, nz=nz, z0=z0)
        plt.hot()
        plt.contourf(coord, z_grid, numpy.abs(f).T, 16)


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
