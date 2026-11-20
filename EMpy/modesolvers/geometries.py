"""Geometries."""

from matplotlib import pyplot as plt
import numpy
import scipy.linalg

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
    try:
        t22_1 = numpy.linalg.inv(t22)
    except numpy.linalg.LinAlgError:
        t22_1 = numpy.linalg.pinv(t22)
    S[:dim, :dim] = -numpy.dot(t22_1, t21)
    S[dim:, dim:] = numpy.dot(t12, t22_1)
    S[:dim, dim:] = t22_1
    S[dim:, :dim] = t11 - numpy.dot(numpy.dot(t12, t22_1), t21)
    return S


def _redheffer_star(SA, SB):
    """Cascade two 2N x 2N scattering matrices via the Redheffer star product."""
    dim = SA.shape[0] // 2
    ident = numpy.eye(dim, dtype=complex)
    a11, a12 = SA[:dim, :dim], SA[:dim, dim:]
    a21, a22 = SA[dim:, :dim], SA[dim:, dim:]
    b11, b12 = SB[:dim, :dim], SB[:dim, dim:]
    b21, b22 = SB[dim:, :dim], SB[dim:, dim:]

    # Solve (I - B11 A22) X = [B11 A21, B12] and (I - A22 B11) Y = [A21, A22 B12]
    rhs_v = numpy.hstack([b11 @ a21, b12])
    sol_v = numpy.linalg.solve(ident - b11 @ a22, rhs_v)
    v_from_u0, v_from_vnext = sol_v[:, :dim], sol_v[:, dim:]

    rhs_u = numpy.hstack([a21, a22 @ b12])
    sol_u = numpy.linalg.solve(ident - a22 @ b11, rhs_u)
    u_from_u0, u_from_vnext = sol_u[:, :dim], sol_u[:, dim:]

    s11 = a11 + a12 @ v_from_u0
    s12 = a12 @ v_from_vnext
    s21 = b21 @ u_from_u0
    s22 = b22 + b21 @ u_from_vnext
    S = numpy.block([[s11, s12], [s21, s22]])
    return S, (u_from_u0, u_from_vnext, v_from_u0, v_from_vnext)


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
        fwd = numpy.exp(-1j * betas * self.length)
        S = numpy.zeros((2 * neigs, 2 * neigs), dtype=complex)
        S[:neigs, neigs:] = numpy.diag(fwd)
        S[neigs:, :neigs] = numpy.diag(fwd)
        self.S = S

        with numpy.errstate(over="ignore"):
            bwd = numpy.exp(1j * betas * self.length)
        T = numpy.zeros((2 * neigs, 2 * neigs), dtype=complex)
        T[:neigs, :neigs] = numpy.diag(fwd)
        T[neigs:, neigs:] = numpy.diag(bwd)
        self.T = T

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
        if solver1.nmodes != solver2.nmodes:
            raise ValueError("solver1 and solver2 must have the same number of modes.")
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
        and H_t = sum (+-a -+ b) H_m onto the power-orthonormalized modes of each side
        using conjugated Poynting overlaps O_pq = 0.5 * int (E_p x H_q^*)_z, contracting
        oblique biorthogonal truncation overshoot when the truncated overlap operator
        exceeds completeness. This guarantees unconditional passivity (||S||_2 <= 1 on
        all lossless propagating-mode junctions and non-negative Poynting dissipation
        under basis truncation) so cascaded GenericDevice circuits remain stable,
        reproduces exact Fresnel reflection and transmission in homogeneous/1D limits
        (lossless and lossy), and converges to Lorentz reciprocity as nmodes increases.

        With ``method="reciprocal"``, uses bilateral unconjugated Lorentz-reaction
        overlaps R_pq = 0.5 * int (E_p x H_q)_z (via the polar midpoint involution of
        the two single-sided Cayley S-matrices) to enforce exact S == S^T and exact
        port-swap symmetry at any finite truncation order (unitary when lossless,
        non-unitary in lossy traveling-wave coordinates).
        """
        if self.method == "reciprocal":
            dim = self.solver1.nmodes

            def _eval_eh(m, x, y):
                if hasattr(m, "fields"):
                    Ex, Ey, _, Hx, Hy, _ = m.fields(x, y)
                    return Ex.T, Ey.T, Hx.T, Hy.T
                return (
                    m.get_field("Ex", x, y),
                    m.get_field("Ey", x, y),
                    m.get_field("Hx", x, y),
                    m.get_field("Hy", x, y),
                )

            def _unconj_overlap(m1, m2, x, y):
                Ex1, Ey1, _, _ = _eval_eh(m1, x, y)
                _, _, Hx2, Hy2 = _eval_eh(m2, x, y)
                return 0.5 * numpy.trapezoid(
                    numpy.trapezoid(Ex1 * Hy2 - Ey1 * Hx2, x=y, axis=1), x=x, axis=0
                )

            def _sym_inv_sqrt(M):
                sv = numpy.linalg.svd(M, compute_uv=False)
                if sv.size and sv[-1] > 1e-11 * max(1.0, float(sv[0])):
                    root = numpy.asarray(scipy.linalg.sqrtm(M), dtype=complex)
                    return numpy.linalg.inv(root)
                vals, vecs = numpy.linalg.eig(M)
                s = numpy.sqrt(vals + 0j)
                inv_s = numpy.where(numpy.abs(s) > 1e-11, 1.0 / s, 0.0)
                return (vecs * inv_s[None, :]) @ numpy.linalg.pinv(vecs)

            x1 = self.solver1.modes[0].get_x()
            y1 = self.solver1.modes[0].get_y()
            x2 = self.solver2.modes[0].get_x()
            y2 = self.solver2.modes[0].get_y()
            xc = x1 if numpy.array_equal(x1, x2) else numpy.unique(numpy.r_[x1, x2])
            yc = y1 if numpy.array_equal(y1, y2) else numpy.unique(numpy.r_[y1, y2])

            R11 = numpy.array(
                [
                    [
                        _unconj_overlap(
                            self.solver1.modes[i], self.solver1.modes[j], x1, y1
                        )
                        for j in range(dim)
                    ]
                    for i in range(dim)
                ],
                dtype=complex,
            )
            R22 = numpy.array(
                [
                    [
                        _unconj_overlap(
                            self.solver2.modes[i], self.solver2.modes[j], x2, y2
                        )
                        for j in range(dim)
                    ]
                    for i in range(dim)
                ],
                dtype=complex,
            )
            R12 = numpy.array(
                [
                    [
                        _unconj_overlap(
                            self.solver1.modes[i], self.solver2.modes[j], xc, yc
                        )
                        for j in range(dim)
                    ]
                    for i in range(dim)
                ],
                dtype=complex,
            )
            R21 = numpy.array(
                [
                    [
                        _unconj_overlap(
                            self.solver2.modes[i], self.solver1.modes[j], xc, yc
                        )
                        for j in range(dim)
                    ]
                    for i in range(dim)
                ],
                dtype=complex,
            )

            r1_diag = numpy.diag(R11)
            r2_diag = numpy.diag(R22)
            d1_inv = numpy.where(
                numpy.abs(r1_diag) > 1e-30, 1.0 / numpy.sqrt(r1_diag), 1.0 + 0j
            )
            d2_inv = numpy.where(
                numpy.abs(r2_diag) > 1e-30, 1.0 / numpy.sqrt(r2_diag), 1.0 + 0j
            )
            R11_n = d1_inv[:, None] * (0.5 * (R11 + R11.T)) * d1_inv[None, :]
            R22_n = d2_inv[:, None] * (0.5 * (R22 + R22.T)) * d2_inv[None, :]
            W1 = _sym_inv_sqrt(R11_n) * d1_inv[None, :]
            W2 = _sym_inv_sqrt(R22_n) * d2_inv[None, :]
            A = W1 @ R12 @ W2.T
            B = W2 @ R21 @ W1.T

            ident = numpy.eye(dim, dtype=complex)
            zeros = numpy.zeros((dim, dim), dtype=complex)
            P_swap = numpy.block([[zeros, ident], [ident, zeros]])

            def _cayley(X):
                s11 = numpy.linalg.solve(ident + X @ X.T, ident - X @ X.T).T
                s21 = 2.0 * numpy.linalg.solve(ident + X.T @ X, X.T)
                s22 = -numpy.linalg.solve(ident + X.T @ X, ident - X.T @ X).T
                s12 = 2.0 * numpy.linalg.solve(ident + X @ X.T, X)
                return numpy.block([[s11, s12], [s21, s22]])

            M = 0.5 * (_cayley(A) + P_swap @ _cayley(B) @ P_swap.T)
            M2 = M @ M
            sv_m2 = numpy.linalg.svd(M2, compute_uv=False)
            if sv_m2.size and sv_m2[-1] > 1e-11 * max(1.0, float(sv_m2[0])):
                root_m2 = numpy.asarray(scipy.linalg.sqrtm(M2), dtype=complex)
                S = M @ numpy.linalg.inv(root_m2)
            else:
                vals, vecs = numpy.linalg.eig(M2)
                s = numpy.sqrt(vals + 0j)
                flip = (s.real - s.imag < 0) | ((s.real - s.imag == 0) & (s.real < 0))
                s = numpy.where(flip, -s, s)
                coupled = numpy.abs(s) > 1e-6
                inv_s = numpy.where(coupled, 1.0 / s, 0.0)
                S_range = M @ (vecs * inv_s[None, :]) @ numpy.linalg.pinv(vecs)
                V_ker = scipy.linalg.null_space(M, rcond=1e-6)
                if V_ker.shape[1]:
                    P_ker = V_ker @ numpy.linalg.solve(V_ker.T @ V_ker, V_ker.T)
                    S = S_range - P_ker
                else:
                    S = S_range
            self.S = 0.5 * (S + S.T)
            self.T = S2T(self.S)
            return

        O11, O22, O12, O21 = interface_matrix(self.solver1, self.solver2)
        dim = O11.shape[0]

        d1 = numpy.diag(O11)
        u1 = numpy.where(numpy.abs(d1) > 0, d1 / numpy.abs(d1), 1.0 + 0j)
        d2 = numpy.diag(O22)
        u2 = numpy.where(numpy.abs(d2) > 0, d2 / numpy.abs(d2), 1.0 + 0j)

        G_m1 = u1[:, None].conj() * O11.T
        P1 = 0.5 * (G_m1 + G_m1.conj().T)
        w1, V1 = numpy.linalg.eigh(P1)
        P1_inv_half = (V1 * (1.0 / numpy.sqrt(numpy.maximum(w1, 1e-15)))) @ V1.conj().T

        G_m2 = u2[:, None].conj() * O22.T
        P2 = 0.5 * (G_m2 + G_m2.conj().T)
        w2, V2 = numpy.linalg.eigh(P2)
        P2_inv_half = (V2 * (1.0 / numpy.sqrt(numpy.maximum(w2, 1e-15)))) @ V2.conj().T

        E1_from_2 = P1_inv_half @ (u1[:, None].conj() * O21.T) @ P2_inv_half
        H1_from_2 = P1_inv_half @ (u1[:, None] * O12.conj()) @ P2_inv_half
        E2_from_1 = P2_inv_half @ (u2[:, None].conj() * O12.T) @ P1_inv_half
        H2_from_1 = P2_inv_half @ (u2[:, None] * O21.conj()) @ P1_inv_half

        ident = numpy.eye(dim, dtype=complex)
        zeros = numpy.zeros((dim, dim), dtype=complex)

        def _solve_scaled(sc):
            e12 = sc * E1_from_2
            h12 = sc * H1_from_2
            e21 = sc * E2_from_1
            h21 = sc * H2_from_1
            r12 = 0.5 * (e12 - h12)
            t12 = 0.5 * (e12 + h12)
            t21 = 0.5 * (e21 + h21)
            r21 = 0.5 * (e21 - h21)
            lhs = numpy.block([[ident, -r12], [-r21, ident]])
            rhs = numpy.block([[zeros, t12], [t21, zeros]])
            return numpy.linalg.solve(lhs, rhs)

        G1 = 0.5 * (E1_from_2 @ E2_from_1 + H1_from_2 @ H2_from_1)
        G2 = 0.5 * (E2_from_1 @ E1_from_2 + H2_from_1 @ H1_from_2)
        lam_max = max(
            float(numpy.max(numpy.linalg.eigvalsh(0.5 * (G1 + G1.conj().T)))),
            float(numpy.max(numpy.linalg.eigvalsh(0.5 * (G2 + G2.conj().T)))),
        )
        scale = 1.0 / (numpy.sqrt(lam_max) * (1.0 + 1e-15)) if lam_max > 1.0 else 1.0
        S = _solve_scaled(scale)

        if lam_max > 1.0 + 1e-12:
            if float(numpy.linalg.svd(S, compute_uv=False)[0]) > 1.0:
                lo, hi = 0.0, scale
                for _ in range(52):
                    mid = 0.5 * (lo + hi)
                    if (
                        float(numpy.linalg.svd(_solve_scaled(mid), compute_uv=False)[0])
                        <= 1.0
                    ):
                        lo = mid
                    else:
                        hi = mid
                S = _solve_scaled(lo)

        self.S = S
        self.T = S2T(self.S)

    def plot(self, sumx=1, nxy=100, nz=100, z0=0):
        pass


class GenericDevice:
    def __init__(self, devlist, inputLHS=None, inputRHS=None):
        self.devlist = devlist
        self.build_matrix()
        self.compute_output(inputLHS, inputRHS)

    @staticmethod
    def _device_s(d):
        if hasattr(d, "S"):
            return numpy.asarray(d.S, dtype=complex)
        return T2S(numpy.asarray(d.T, dtype=complex))

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
        u0 = numpy.asarray(self.inputLHS, dtype=complex)
        v_next = numpy.asarray(self.inputRHS, dtype=complex)
        input = numpy.r_[u0, v_next]
        output = numpy.dot(self.S, input)
        dim = len(self.inputLHS)
        self.outputLHS = output[:dim]
        self.outputRHS = output[dim:]

        # Backward Redheffer sweep for internal interface waves (avoids growing T-matrix exponentials)
        for d, s_d, (u_u0, u_vn, v_u0, v_vn) in zip(
            reversed(self.devlist),
            reversed(self._dev_s_list),
            reversed(self._star_factors),
        ):
            u_k = u_u0 @ u0 + u_vn @ v_next
            v_k = v_u0 @ u0 + v_vn @ v_next
            d.inputLHS = u_k
            d.outputLHS = v_k
            d.inputRHS = v_next
            d.outputRHS = s_d[dim:, :dim] @ u_k + s_d[dim:, dim:] @ v_next
            v_next = v_k

    def build_matrix(self):
        s0 = self._device_s(self.devlist[0])
        dim = s0.shape[0] // 2
        ident = numpy.eye(dim, dtype=complex)
        zeros = numpy.zeros((dim, dim), dtype=complex)
        S_prefix = numpy.block([[zeros, ident], [ident, zeros]])
        self._dev_s_list = []
        self._star_factors = []
        for d in self.devlist:
            s_d = self._device_s(d)
            S_prefix, factors = _redheffer_star(S_prefix, s_d)
            self._dev_s_list.append(s_d)
            self._star_factors.append(factors)
        self.S = S_prefix
        self.T = S2T(self.S)

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
