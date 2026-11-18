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
        """Compute 1D-integrated bidirectional modal optical intensity for visualization.

        Note: applies the coherent scalar wave envelope a * exp(-i*beta*z) +
        b * exp(+i*beta*(z - L)) to each mode's impedance-weighted profile so
        counter-propagating waves exhibit standing-wave nodes/antinodes (rather
        than net z-Poynting flux |a|^2 - |b|^2, which vanishes in an equal-amplitude
        standing wave).
        """
        m0 = self.solver.modes[0]
        x_grid = m0.get_x() if nxy is None else m0.get_x(nxy)
        y_grid = m0.get_y() if nxy is None else m0.get_y(nxy)
        z = numpy.linspace(0, self.length, nz)

        nx, ny = len(x_grid), len(y_grid)
        Ex_tot = numpy.zeros((nx, ny, nz), dtype=complex)
        Ey_tot = numpy.zeros((nx, ny, nz), dtype=complex)
        Hx_env = numpy.zeros((nx, ny, nz), dtype=complex)
        Hy_env = numpy.zeros((nx, ny, nz), dtype=complex)
        intensity_incoherent = None

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
                if intensity_incoherent is None:
                    intensity_incoherent = numpy.zeros((nx, ny, nz), dtype=float)
                intensity_incoherent += (
                    numpy.real(m.intensity(x_grid, y_grid))[:, :, numpy.newaxis]
                    * numpy.abs(env)[numpy.newaxis, numpy.newaxis, :] ** 2
                )
                continue

            Ex_tot += Ex[:, :, numpy.newaxis] * env[numpy.newaxis, numpy.newaxis, :]
            Ey_tot += Ey[:, :, numpy.newaxis] * env[numpy.newaxis, numpy.newaxis, :]
            Hx_env += Hx[:, :, numpy.newaxis] * env[numpy.newaxis, numpy.newaxis, :]
            Hy_env += Hy[:, :, numpy.newaxis] * env[numpy.newaxis, numpy.newaxis, :]

        intensity_3d = 0.5 * numpy.real(
            Ex_tot * numpy.conj(Hy_env) - Ey_tot * numpy.conj(Hx_env)
        )
        if intensity_incoherent is not None:
            intensity_3d = intensity_3d + intensity_incoherent

        if sumx is None:
            coord = y_grid
            f = numpy.trapezoid(intensity_3d, x=x_grid, axis=0)
        else:
            coord = x_grid
            f = numpy.trapezoid(intensity_3d, x=y_grid, axis=1)
        return coord, z0 + z, f

    def plot(self, sumx=1, nxy=100, nz=100, z0=0):
        if self.length == 0:
            return
        coord, z_grid, f = self._marginal_intensity(sumx=sumx, nxy=nxy, nz=nz, z0=z0)
        plt.hot()
        plt.contourf(coord, z_grid, numpy.abs(f).T, 16)


class SimpleJoint:
    #    def __init__(self, cs1, cs2, solver1, solver2, inputLHS=None, inputRHS=None):
    def __init__(
        self, solver1, solver2, inputLHS=None, inputRHS=None, method="passive"
    ):
        #        self.cs1 = cs1
        if method not in ("passive", "reciprocal"):
            raise ValueError("method must be 'passive' or 'reciprocal'.")
        self.solver1 = solver1
        #        self.cs2 = cs2
        self.solver2 = solver2
        self.method = method
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
        """Vector mode-matching S-matrix at a zero-length waveguide junction.

        With ``method="passive"`` (default), projects transverse E_t = sum (a + b) E_m
        and H_t = sum (+-a -+ b) H_m onto the modes of each side using conjugated power
        overlaps O_pq = 0.5 * int (E_p x H_q^*)_z:
          Side 1: O11^T (a1 + b1) = O21^T (b2 + a2),  O11^* (a1 - b1) = O12^* (b2 - a2)
          Side 2: O22^T (b2 + a2) = O12^T (a1 + b1),  O22^* (b2 - a2) = O21^* (a1 - b1)
        This guarantees unconditional passivity (||S||_2 <= 1) under basis truncation so
        cascaded GenericDevice circuits remain stable, reproduces exact Fresnel reflection
        and transmission in homogeneous/1D limits (lossless and lossy), and converges to
        Lorentz reciprocity as nmodes increases.

        With ``method="reciprocal"``, uses unconjugated Lorentz-reaction overlaps
        R_pq = 0.5 * int (E_p x H_q)_z (normalized to unit self-reaction) to enforce
        exact S == S^T at any finite truncation order.
        """
        if self.method == "reciprocal":
            dim = self.solver1.nmodes

            def _unconj_overlap(m1, m2):
                x = m2.get_x()
                y = m2.get_y()
                if hasattr(m1, "fields"):
                    Ex1, Ey1, _, _, _, _ = m1.fields(x, y)
                    Ex1, Ey1 = Ex1.T, Ey1.T
                else:
                    Ex1 = m1.get_field("Ex", x, y)
                    Ey1 = m1.get_field("Ey", x, y)
                if hasattr(m2, "fields"):
                    _, _, _, Hx2, Hy2, _ = m2.fields(x, y)
                    Hx2, Hy2 = Hx2.T, Hy2.T
                else:
                    Hx2 = m2.get_field("Hx", x, y)
                    Hy2 = m2.get_field("Hy", x, y)
                return 0.5 * numpy.trapezoid(
                    numpy.trapezoid(Ex1 * Hy2 - Ey1 * Hx2, x=y, axis=1), x=x, axis=0
                )

            d1 = numpy.array(
                [_unconj_overlap(m, m) for m in self.solver1.modes], dtype=complex
            )
            d2 = numpy.array(
                [_unconj_overlap(m, m) for m in self.solver2.modes], dtype=complex
            )
            X = numpy.zeros((dim, dim), dtype=complex)
            for i in range(dim):
                for j in range(dim):
                    X[i, j] = _unconj_overlap(
                        self.solver1.modes[i], self.solver2.modes[j]
                    ) / numpy.sqrt(d1[i] * d2[j])
            ident = numpy.eye(dim, dtype=complex)
            s11 = numpy.linalg.solve(ident + X @ X.T, ident - X @ X.T).T
            s21 = 2.0 * numpy.linalg.solve(ident + X.T @ X, X.T)
            s22 = -numpy.linalg.solve(ident + X.T @ X, ident - X.T @ X).T
            s12 = 2.0 * numpy.linalg.solve(ident + X @ X.T, X)
            self.S = numpy.block([[s11, s12], [s21, s22]])
            self.T = S2T(self.S)
            return

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
