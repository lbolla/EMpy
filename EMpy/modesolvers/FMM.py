"""Film Mode Matching Mode Solver

Implementation of the Film Mode Matching (FMM) algorithm, as described in:

 - Sudbo, "Film mode matching a versatile numerical method for vector mode field calculations in dielectric waveguides",
   Pure App. Optics, 2 (1993), 211-233
 - Sudbo, "Improved formulation of the film mode matching method for mode field calculations in dielectric waveguides",
   Pure App. Optics, 3 (1994), 381-388

Examples
========

    See L{FMM1d} and L{FMM2d}.

"""

__author__ = "Luca Gamberale & Lorenzo Bolla"

import copy
from functools import reduce

from matplotlib import pyplot as plt
import numpy
import scipy.optimize

import EMpy.utils
from EMpy.modesolvers.interface import Mode, ModeSolver


class Message:
    def __init__(self, msg, verbosity=0):
        self.msg = msg
        self.verbosity = verbosity

    def show(self, verbosity=0):
        if self.verbosity <= verbosity:
            print((self.verbosity - 1) * "\t" + self.msg)


class Struct:
    """Empty class to fill with whatever I want. Maybe a dictionary would do?"""

    pass


class Boundary:
    """Boundary conditions.

    Electric and Magnetic boundary conditions are translated to Symmetric
    and Antisymmetric for each field.

    @ivar xleft: Left bc on x.
    @ivar xright: Right bc on x.
    @ivar yleft: Left bc on y.
    @ivar yright: Right bc on y.

    """

    def __init__(
        self,
        xleft="Electric Wall",
        yleft="Magnetic Wall",
        xright="Electric Wall",
        yright="Magnetic Wall",
    ):
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

        if not reduce(
            lambda x, y: x & y,
            [
                (x == "Electric Wall") | (x == "Magnetic Wall")
                for x in [self.xleft, self.yleft, self.xright, self.yright]
            ],
        ):
            raise ValueError("Unknown boundary.")

    def translate(self):
        """Translate for each field.

        @raise ValueError: Unknown boundary.

        """

        self.xh = ""
        self.xe = ""
        self.yh = ""
        self.ye = ""
        if self.xleft == "Electric Wall":
            self.xh += "A"
            self.xe += "S"
        elif self.xleft == "Magnetic Wall":
            self.xh += "S"
            self.xe += "A"
        else:
            raise ValueError("Unknown boundary.")
        if self.xright == "Electric Wall":
            self.xh += "A"
            self.xe += "S"
        elif self.xright == "Magnetic Wall":
            self.xh += "S"
            self.xe += "A"
        else:
            raise ValueError("Unknown boundary.")
        if self.yleft == "Electric Wall":
            self.yh += "A"
            self.ye += "S"
        elif self.yleft == "Magnetic Wall":
            self.yh += "S"
            self.ye += "A"
        else:
            raise ValueError("Unknown boundary.")
        if self.yright == "Electric Wall":
            self.yh += "A"
            self.ye += "S"
        elif self.yright == "Magnetic Wall":
            self.yh += "S"
            self.ye += "A"
        else:
            raise ValueError("Unknown boundary.")

    def __str__(self):
        return "xleft = %s, xright = %s, yleft = %s, yright = %s" % (
            self.xleft,
            self.xright,
            self.yleft,
            self.yright,
        )


class Slice:
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
        return "x1 = %g, x2 = %g\nUy = %s\nboundary = %s" % (
            self.x1,
            self.x2,
            self.Uy,
            self.boundary,
        )


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
        return "sl = %s\nsr = %s\nal = %s\nar = %s\nk = %s\nU = %s" % (
            self.sl.__str__(),
            self.sr.__str__(),
            self.al.__str__(),
            self.ar.__str__(),
            self.k.__str__(),
            self.U.__str__(),
        )


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
            sr = self.sr[iU]
            al = self.al[iU]
            Ul = self.U[iU]
            Ur = self.U[iU + 1]
            idx = numpy.where((Ul <= y) & (y <= Ur))
            yy = y[idx] - Ul
            f[idx], _ = _eval_segment(k, sl, sr, al, Ur - Ul, yy)
        return f

    def plot(self, y):
        f = self.eval(y)
        plt.plot(y, numpy.real(f), y, numpy.imag(f))
        plt.legend(("real", "imag"))
        plt.xlabel("y")
        plt.ylabel("mode1d")

    def __str__(self):
        return (
            "sl = %s\nsr = %s\nal = %s\nar = %s\nk = %s\nkeff = %s\nzero = %s\nU = %s"
            % (
                self.sl.__str__(),
                self.sr.__str__(),
                self.al.__str__(),
                self.ar.__str__(),
                self.k.__str__(),
                self.keff.__str__(),
                self.zero.__str__(),
                self.U.__str__(),
            )
        )


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
        k0 = 2.0 * numpy.pi / self.slicesx[0].wl
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
        ExTE = numpy.zeros((leny, lenx), dtype=complex)
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
            if mx == len(self.slicesx) - 1:
                idx = numpy.where((slice.x1 <= x) & (x <= slice.x2))
            else:
                idx = numpy.where((slice.x1 <= x) & (x < slice.x2))
            x2 = x[idx] - slice.x1

            for n in range(nmodi):
                fi = slice.modih[n].eval(y)
                fidot = dot(slice.modih[n]).eval(y)

                psi = slice.modie[n].eval(y)
                psisueps = sueps(slice.modie[n]).eval(y)
                psidotsueps = sueps(dot(slice.modie[n])).eval(y)

                # Evaluate u(x) and du/dx consistently: use left-endpoint (sl, al)
                # for oscillatory/weakly-evanescent slices (|Im(k*dx)| <= 1, avoiding
                # 0/0 when sin(k*dx) == 0), and symmetric two-endpoint (sl, sr) decaying
                # exponentials for evanescent slices (|Im(k*dx)| > 1) to avoid
                # O(exp(|Im(k*dx)|)) catastrophic cancellation.
                dx_slice = slice.x2 - slice.x1
                kfh = self.modih[n].k[mx]
                kxh = numpy.emath.sqrt(kfh**2 - kz**2)
                slh = self.modih[n].sl[mx] * (k0 / kfh) ** 2
                srh = self.modih[n].sr[mx] * (k0 / kfh) ** 2
                alh = self.modih[n].al[mx]
                uh[n, idx], udoth[n, idx] = _eval_segment(
                    kxh, slh, srh, alh, dx_slice, x2
                )

                kfe = self.modie[n].k[mx]
                kxe = numpy.emath.sqrt(kfe**2 - kz**2)
                sle = self.modie[n].sl[mx] * (k0 / kfe) ** 2
                sre = self.modie[n].sr[mx] * (k0 / kfe) ** 2
                ale = self.modie[n].al[mx]
                ue[n, idx], udote[n, idx] = _eval_segment(
                    kxe, sle, sre, ale, dx_slice, x2
                )

                Exsh[:, n] = (kz / k0) * fi
                Exah[:, n] = 0
                Exse[:, n] = 0
                Exae[:, n] = -psidotsueps / k0**2

                Eysh[:, n] = 0
                Eyah[:, n] = 0
                Eyse[:, n] = -((kfe / k0) ** 2) * psisueps
                Eyae[:, n] = 0

                Ezsh[:, n] = 0
                Ezah[:, n] = -1j * fi / k0
                Ezse[:, n] = 1j * kz / k0**2 * psidotsueps
                Ezae[:, n] = 0

                cBxsh[:, n] = 0
                cBxah[:, n] = fidot / k0**2
                cBxse[:, n] = kz / k0 * psi
                cBxae[:, n] = 0

                cBysh[:, n] = (kfh / k0) ** 2 * fi
                cByah[:, n] = 0
                cByse[:, n] = 0
                cByae[:, n] = 0

                cBzsh[:, n] = -1j * kz / k0**2 * fidot
                cBzah[:, n] = 0
                cBzse[:, n] = 0
                cBzae[:, n] = -1j * psi / k0

            ExTE[:, idx] = numpy.tensordot(Exsh, uh[:, idx], axes=1) + numpy.tensordot(
                Exah, udoth[:, idx], axes=1
            )
            ExTM[:, idx] = numpy.tensordot(Exse, ue[:, idx], axes=1) + numpy.tensordot(
                Exae, udote[:, idx], axes=1
            )

            EyTE[:, idx] = numpy.tensordot(Eysh, uh[:, idx], axes=1) + numpy.tensordot(
                Eyah, udoth[:, idx], axes=1
            )
            EyTM[:, idx] = numpy.tensordot(Eyse, ue[:, idx], axes=1) + numpy.tensordot(
                Eyae, udote[:, idx], axes=1
            )

            EzTE[:, idx] = numpy.tensordot(Ezsh, uh[:, idx], axes=1) + numpy.tensordot(
                Ezah, udoth[:, idx], axes=1
            )
            EzTM[:, idx] = numpy.tensordot(Ezse, ue[:, idx], axes=1) + numpy.tensordot(
                Ezae, udote[:, idx], axes=1
            )

            cBxTE[:, idx] = numpy.tensordot(
                cBxsh, uh[:, idx], axes=1
            ) + numpy.tensordot(cBxah, udoth[:, idx], axes=1)
            cBxTM[:, idx] = numpy.tensordot(
                cBxse, ue[:, idx], axes=1
            ) + numpy.tensordot(cBxae, udote[:, idx], axes=1)

            cByTE[:, idx] = numpy.tensordot(
                cBysh, uh[:, idx], axes=1
            ) + numpy.tensordot(cByah, udoth[:, idx], axes=1)
            cByTM[:, idx] = numpy.tensordot(
                cByse, ue[:, idx], axes=1
            ) + numpy.tensordot(cByae, udote[:, idx], axes=1)

            cBzTE[:, idx] = numpy.tensordot(
                cBzsh, uh[:, idx], axes=1
            ) + numpy.tensordot(cBzah, udoth[:, idx], axes=1)
            cBzTM[:, idx] = numpy.tensordot(
                cBzse, ue[:, idx], axes=1
            ) + numpy.tensordot(cBzae, udote[:, idx], axes=1)

        return (
            ExTE,
            ExTM,
            EyTE,
            EyTM,
            EzTE,
            EzTM,
            cBxTE,
            cBxTM,
            cByTE,
            cByTM,
            cBzTE,
            cBzTM,
        )

    def fields(self, x=None, y=None):
        (
            ExTE,
            ExTM,
            EyTE,
            EyTM,
            EzTE,
            EzTM,
            cBxTE,
            cBxTM,
            cByTE,
            cByTM,
            cBzTE,
            cBzTM,
        ) = self.eval(x, y)
        Ex = ExTE + ExTM
        Ey = EyTE + EyTM
        Ez = EzTE + EzTM
        cBx = cBxTE + cBxTM
        cBy = cByTE + cByTM
        cBz = cBzTE + cBzTM
        return (Ex, Ey, Ez, cBx, cBy, cBz)

    def intensity(self, x=None, y=None):
        Ex, Ey, Ez, cBx, cBy, cBz = self.fields(x, y)
        cSz = 0.5 * (Ex * numpy.conj(cBy) - Ey * numpy.conj(cBx))
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

        Ex, Ey, Ez, cBx, cBy, cBz = self.fields(x, y)
        cSTE = 0.5 * EMpy.utils.trapz2(Ex * numpy.conj(cBy), y, x)
        cSTM = 0.5 * EMpy.utils.trapz2(-Ey * numpy.conj(cBx), y, x)
        return numpy.abs(cSTE) / (numpy.abs(cSTE) + numpy.abs(cSTM))

    def TEfrac(self, x_=None, y_=None):
        if x_ is not None or y_ is not None:
            return self.TEfrac_old(x_, y_)
        Sx, Sy = self.__overlap(self)
        return float(numpy.abs(Sx) / (numpy.abs(Sx) + numpy.abs(Sy)))

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
        norm = numpy.emath.sqrt(numpy.real(EMpy.utils.trapz2(cSz, y, x)))
        Ex1, Ey1, Ez1, cBx1, cBy1, cBz1 = m.fields(x, y)
        cSz1 = m.intensity(x, y)
        norm1 = numpy.emath.sqrt(numpy.real(EMpy.utils.trapz2(cSz1, y, x)))
        return 0.5 * EMpy.utils.trapz2(
            Ex / norm * numpy.conj(cBy1 / norm1) - Ey / norm * numpy.conj(cBx1 / norm1),
            y,
            x,
        )

    def __overlap_old(self, mode):
        return self.__overlap(mode)

    def __overlap(self, mode):
        """Analytical z-Poynting components (Sx, Sy) of 0.5 * int (E_1 x cB_2^*)_z dx dy.

        Notes:
        - Ey in eval() already carries a minus sign (-kfe^2/k0^2 * psi/eps * ue),
          so +Sy = 0.5 * int (-Ey_1 * cBx_2^*) dx dy and total overlap is Sx + Sy.
        - Uses self.keff for mode 1 and mode.keff (conjugated where appropriate)
          for mode 2, conjugating mode 2's 1D factors via _conj_mode1d.
        - Refines onto the common (Ux, Uy) mesh so waveguides with distinct slice
          widths or layer thicknesses integrate analytically on smooth sub-cells.
        """
        nmodi1 = len(self.modie)
        nmodi2 = len(mode.modie)
        k0_1 = 2.0 * numpy.pi / self.slicesx[0].wl
        k0_2 = 2.0 * numpy.pi / mode.slicesx[0].wl
        kz1 = self.keff
        kz2 = mode.keff

        Ux1 = numpy.asarray(self.slicesx[0].Ux, dtype=float)
        Ux2 = numpy.asarray(mode.slicesx[0].Ux, dtype=float)
        Uy1 = numpy.asarray(self.slicesx[0].Uy, dtype=float)
        Uy2 = numpy.asarray(mode.slicesx[0].Uy, dtype=float)

        x_min = max(Ux1[0], Ux2[0])
        x_max = min(Ux1[-1], Ux2[-1])
        y_min = max(Uy1[0], Uy2[0])
        y_max = min(Uy1[-1], Uy2[-1])
        if x_max <= x_min or y_max <= y_min:
            return (0j, 0j)

        Ux_common = numpy.unique(numpy.r_[Ux1, Ux2])
        Ux_common = Ux_common[(Ux_common >= x_min) & (Ux_common <= x_max)]
        Uy_common = numpy.unique(numpy.r_[Uy1, Uy2])
        Uy_common = Uy_common[(Uy_common >= y_min) & (Uy_common <= y_max)]

        Sx = 0j
        Sy = 0j

        for ix in range(len(Ux_common) - 1):
            xa = Ux_common[ix]
            xb = Ux_common[ix + 1]
            if xb <= xa:
                continue
            x_mid = 0.5 * (xa + xb)
            mx1 = int(
                numpy.clip(
                    numpy.searchsorted(Ux1, x_mid, side="right") - 1,
                    0,
                    len(self.slicesx) - 1,
                )
            )
            mx2 = int(
                numpy.clip(
                    numpy.searchsorted(Ux2, x_mid, side="right") - 1,
                    0,
                    len(mode.slicesx) - 1,
                )
            )
            slice1 = self.slicesx[mx1]
            slice2 = mode.slicesx[mx2]
            U_sub_x = numpy.array([xa, xb], dtype=float)

            phi_n1s = []
            psidotsueps_n1s = []
            psisueps_n1s = []
            uh_n1s = []
            ue_n1s = []
            uedot_n1s = []
            kfe_n1s = []

            for n1 in range(nmodi1):
                phi_n1 = _refine_mode1d(slice1.modih[n1], Uy_common)
                psi_n1 = slice1.modie[n1]
                psisueps_n1 = _refine_mode1d(sueps(psi_n1), Uy_common)
                psidotsueps_n1 = _refine_mode1d(sueps(dot(psi_n1)), Uy_common)

                uh_n1_full, _ = _slice_mode1dx(self.modih[n1], mx1, k0_1, kz1)
                uh_n1 = _refine_mode1d(uh_n1_full, U_sub_x)

                ue_n1_full, kfe_n1 = _slice_mode1dx(self.modie[n1], mx1, k0_1, kz1)
                ue_n1 = _refine_mode1d(ue_n1_full, U_sub_x)
                uedot_n1 = dot(ue_n1)

                phi_n1s.append(phi_n1)
                psisueps_n1s.append(psisueps_n1)
                psidotsueps_n1s.append(psidotsueps_n1)
                uh_n1s.append(uh_n1)
                ue_n1s.append(ue_n1)
                uedot_n1s.append(uedot_n1)
                kfe_n1s.append(kfe_n1)

            phi_n2s_c = []
            phidot_n2s_c = []
            psi_n2s_c = []
            uh_n2s_c = []
            uhdot_n2s_c = []
            ue_n2s_c = []
            kfh_n2s = []

            for n2 in range(nmodi2):
                phi_n2 = _refine_mode1d(slice2.modih[n2], Uy_common)
                phidot_n2 = dot(phi_n2)
                psi_n2 = _refine_mode1d(slice2.modie[n2], Uy_common)

                uh_n2_full, kfh_n2 = _slice_mode1dx(mode.modih[n2], mx2, k0_2, kz2)
                uh_n2 = _refine_mode1d(uh_n2_full, U_sub_x)
                uhdot_n2 = dot(uh_n2)

                ue_n2_full, _ = _slice_mode1dx(mode.modie[n2], mx2, k0_2, kz2)
                ue_n2 = _refine_mode1d(ue_n2_full, U_sub_x)

                phi_n2s_c.append(_conj_mode1d(phi_n2))
                phidot_n2s_c.append(_conj_mode1d(phidot_n2))
                psi_n2s_c.append(_conj_mode1d(psi_n2))
                uh_n2s_c.append(_conj_mode1d(uh_n2))
                uhdot_n2s_c.append(_conj_mode1d(uhdot_n2))
                ue_n2s_c.append(_conj_mode1d(ue_n2))
                kfh_n2s.append(kfh_n2)

            kz2_c = numpy.conj(kz2)
            for n1 in range(nmodi1):
                uh_n1 = uh_n1s[n1]
                ue_n1 = ue_n1s[n1]
                uedot_n1 = uedot_n1s[n1]
                phi_n1 = phi_n1s[n1]
                psidotsueps_n1 = psidotsueps_n1s[n1]
                psisueps_n1 = psisueps_n1s[n1]
                kfe_n1_sq = kfe_n1s[n1] ** 2

                for n2 in range(nmodi2):
                    uh_n2_c = uh_n2s_c[n2]
                    uhdot_n2_c = uhdot_n2s_c[n2]
                    ue_n2_c = ue_n2s_c[n2]
                    phi_n2_c = phi_n2s_c[n2]
                    phidot_n2_c = phidot_n2s_c[n2]
                    psi_n2_c = psi_n2s_c[n2]
                    kfh_n2_sq_c = numpy.conj(kfh_n2s[n2] ** 2)

                    Sx += 0.5 * (
                        kz1
                        * kfh_n2_sq_c
                        / (k0_1 * k0_2**2)
                        * scalarprod(uh_n1, uh_n2_c)
                        * scalarprod(phi_n1, phi_n2_c)
                        - kfh_n2_sq_c
                        / (k0_1**2 * k0_2**2)
                        * scalarprod(uedot_n1, uh_n2_c)
                        * scalarprod(psidotsueps_n1, phi_n2_c)
                    )
                    Sy += 0.5 * (
                        kfe_n1_sq
                        * kz2_c
                        / (k0_1**2 * k0_2)
                        * scalarprod(ue_n1, ue_n2_c)
                        * scalarprod(psisueps_n1, psi_n2_c)
                        + kfe_n1_sq
                        / (k0_1**2 * k0_2**2)
                        * scalarprod(ue_n1, uhdot_n2_c)
                        * scalarprod(psisueps_n1, phidot_n2_c)
                    )

        return (Sx, Sy)

    def overlap(self, mode, x=None, y=None):
        if isinstance(mode, FMMMode2d) and x is None and y is None:
            Sx, Sy = self.__overlap(mode)
            return Sx + Sy

        if x is None:
            x = self.get_x()
        else:
            x = numpy.atleast_1d(x)
        if y is None:
            y = self.get_y()
        else:
            y = numpy.atleast_1d(y)

        Ex1, Ey1, _, _, _, _ = self.fields(x, y)
        if isinstance(mode, FMMMode2d):
            _, _, _, cBx2, cBy2, _ = mode.fields(x, y)
        else:
            cBx2 = mode.get_field("Hx", x, y).T
            cBy2 = mode.get_field("Hy", x, y).T
        return 0.5 * EMpy.utils.trapz2(
            Ex1 * numpy.conj(cBy2) - Ey1 * numpy.conj(cBx2), y, x
        )

    def norm(self):
        return numpy.sqrt(numpy.abs(numpy.real(self.overlap(self))))

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
        x_Ex_FDTD = EMpy.utils.centered1d(x)
        y_Ex_FDTD = y[1:-1]
        Ex_FDTD = EMpy.utils.interp2(x_Ex_FDTD, y_Ex_FDTD, x0, y0, Ex.T)
        # Ey: ignores x = 0, max
        x_Ey_FDTD = x[1:-1]
        y_Ey_FDTD = EMpy.utils.centered1d(y)
        Ey_FDTD = EMpy.utils.interp2(x_Ey_FDTD, y_Ey_FDTD, x0, y0, Ey.T)
        # Ez: ignores x, y = 0, max
        x_Ez_FDTD = x[1:-1]
        y_Ez_FDTD = y[1:-1]
        Ez_FDTD = EMpy.utils.interp2(x_Ez_FDTD, y_Ez_FDTD, x0, y0, Ez.T)
        # Hx: ignores x = 0, max, /eta0, reverse direction
        x_Hx_FDTD = x[1:-1]
        y_Hx_FDTD = EMpy.utils.centered1d(y)
        Hx_FDTD = EMpy.utils.interp2(x_Hx_FDTD, y_Hx_FDTD, x0, y0, cBx.T) / (
            -EMpy.constants.eta0
        )  # OKKIO!
        # Hy: ignores y = 0, max, /eta0, reverse direction
        x_Hy_FDTD = EMpy.utils.centered1d(x)
        y_Hy_FDTD = y[1:-1]
        Hy_FDTD = EMpy.utils.interp2(x_Hy_FDTD, y_Hy_FDTD, x0, y0, cBy.T) / (
            -EMpy.constants.eta0
        )
        # Hz: /eta0, reverse direction
        x_Hz_FDTD = EMpy.utils.centered1d(x)
        y_Hz_FDTD = EMpy.utils.centered1d(y)
        Hz_FDTD = EMpy.utils.interp2(x_Hz_FDTD, y_Hz_FDTD, x0, y0, cBz.T) / (
            -EMpy.constants.eta0
        )

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
        plt.figure()
        titles = ["Ex", "Ey", "Ez", "cBx", "cBy", "cBz"]
        for i in range(6):
            subplot_id = 231 + i
            plt.subplot(subplot_id)
            plt.contour(x, y, numpy.abs(f[i]))
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(titles[i])
            plt.axis("image")

        # power
        plt.figure()
        plt.contour(x, y, numpy.abs(self.intensity(x, y)))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("cSz")
        plt.axis("image")

    def __str__(self):
        return "neff = %s" % (self.keff / (2 * numpy.pi / self.slicesx[0].wl))


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
        Message("Solving 1d modes.", 1).show(verbosity)
        self.wl = wl
        self.nmodes = nmodes
        self.polarization = polarization
        self.modes = FMM1d_y(
            self.Uy,
            self.ny,
            self.wl,
            self.nmodes,
            self.boundary,
            self.polarization,
            verbosity,
        )
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
        Message("Solving 2d modes", 1).show(verbosity)
        self.wl = wl
        self.n1dmodes = n1dmodes
        self.nmodes = nmodes
        self.slices = script1d(
            self.Ux, self.Uy, self.rix, self.wl, self.boundary, self.n1dmodes, verbosity
        )
        self.modes = FMM1d_x_component(self.slices, nmodes, verbosity)
        return self


def analyticalsolution(nmodi, TETM, FMMpars):
    betay = FMMpars["beta"]
    epsilon = FMMpars["epsilon"]
    Uy = FMMpars["Uy"]
    by = FMMpars["boundary"]

    Nregions = len(epsilon)
    sl = numpy.zeros((nmodi, Nregions), dtype=complex)
    sr = numpy.zeros_like(sl)
    al = numpy.zeros_like(sl)
    ar = numpy.zeros_like(sl)
    # interval
    D = Uy[-1] - Uy[0]
    if TETM == "TE":
        N = numpy.sqrt(2.0 / D)
    else:
        N = numpy.sqrt(2.0 / D * epsilon[0])

    # boundary condition
    if by == "AA":
        kn = numpy.pi * numpy.arange(1, nmodi + 1) / D
        kn = kn[:, numpy.newaxis]
        sl[:] = numpy.sin(kn * (Uy[:-1] - Uy[0]))
        sr[:] = numpy.sin(kn * (Uy[1:] - Uy[0]))
        al[:] = numpy.cos(kn * (Uy[:-1] - Uy[0]))
        ar[:] = numpy.cos(kn * (Uy[1:] - Uy[0]))
        sr[:, -1] = 0.0
        sl[:, 0] = 0.0
    elif by == "AS":
        kn = numpy.pi * (numpy.arange(0, nmodi) + 0.5) / D
        kn = kn[:, numpy.newaxis]
        sl[:] = numpy.sin(kn * (Uy[:-1] - Uy[0]))
        sr[:] = numpy.sin(kn * (Uy[1:] - Uy[0]))
        al[:] = numpy.cos(kn * (Uy[:-1] - Uy[0]))
        ar[:] = numpy.cos(kn * (Uy[1:] - Uy[0]))
        ar[:, -1] = 0.0
        sl[:, 0] = 0.0
    elif by == "SA":
        kn = numpy.pi * (numpy.arange(0, nmodi) + 0.5) / D
        kn = kn[:, numpy.newaxis]
        sl[:] = numpy.cos(kn * (Uy[:-1] - Uy[0]))
        sr[:] = numpy.cos(kn * (Uy[1:] - Uy[0]))
        al[:] = -numpy.sin(kn * (Uy[:-1] - Uy[0]))
        ar[:] = -numpy.sin(kn * (Uy[1:] - Uy[0]))
        sr[:, -1] = 0.0
        al[:, 0] = 0.0
    elif by == "SS":
        kn = numpy.pi * numpy.arange(0, nmodi) / D
        kn = kn[:, numpy.newaxis]
        sl[:] = numpy.cos(kn * (Uy[:-1] - Uy[0]))
        sr[:] = numpy.cos(kn * (Uy[1:] - Uy[0]))
        al[:] = -numpy.sin(kn * (Uy[:-1] - Uy[0]))
        ar[:] = -numpy.sin(kn * (Uy[1:] - Uy[0]))
        ar[:, -1] = 0.0
        al[:, 0] = 0.0

    # normalizzazione
    sl *= N
    sr *= N

    for n in range(0, nmodi):
        al[n, :] *= N * kn[n]
        ar[n, :] *= N * kn[n]

    # caso speciale. se k=0 la funzione e' costante e la normalizzazione e'
    # diversa. capita solo con boundary SS e per il primo modo
    if by == "SS":
        sqrt2 = numpy.sqrt(2.0)
        sl[0, :] /= sqrt2
        sr[0, :] /= sqrt2
        al[0, :] /= sqrt2
        ar[0, :] /= sqrt2

    modi = []
    for mk in range(0, nmodi):
        modo = FMMMode1dy()
        modo.sl = sl[mk, :].astype(complex)
        modo.sr = sr[mk, :].astype(complex)
        modo.al = al[mk, :].astype(complex)
        modo.ar = ar[mk, :].astype(complex)
        modo.k = kn[mk] * numpy.ones(Nregions)
        modo.U = Uy
        # Select passive branch Re(keff) - Im(keff) >= 0 (exp(+iwt - i*keff*z))
        # so lossless cutoff modes lie on the negative imaginary axis (-i|keff|)
        # continuously with the lossy kappa -> 0+ limit.
        keff = numpy.emath.sqrt(betay[0] ** 2 - kn[mk] ** 2)
        flip_keff = (keff.real - keff.imag < 0) | (
            (keff.real - keff.imag == 0) & (keff.real < 0)
        )
        modo.keff = numpy.where(flip_keff, -keff, keff)
        modo.zero = 0.0
        modo.pars = FMMpars
        modi.append(modo)
    return modi


def sinxsux(x):
    return numpy.sinc(x / numpy.pi)


def FMMshootingTM(kz_, FMMpars):
    betay = FMMpars["beta"]
    eps = FMMpars["epsilon"]
    Uy = FMMpars["Uy"]
    by = FMMpars["boundary"]

    kz = numpy.atleast_1d(kz_)

    Nregions = len(betay)
    d = numpy.diff(Uy)
    Delta = numpy.zeros(kz.shape, dtype=complex)

    sl = numpy.zeros(Nregions, dtype=complex)
    sr = numpy.zeros_like(sl)
    al = numpy.zeros_like(sl)
    ar = numpy.zeros_like(sl)

    k_ = numpy.emath.sqrt(betay**2 - kz[:, numpy.newaxis] ** 2)
    kd = k_[:, numpy.newaxis] * d
    sinkdsuk_ = sinxsux(kd) * d
    coskd_ = numpy.cos(kd)
    sinkdk_ = numpy.sin(kd) * k_[:, numpy.newaxis]

    # left boundary condition
    if by[0] == "A":
        al[0] = 1
    elif by[0] == "S":
        sl[0] = 1
    else:
        raise ValueError("unrecognized left boundary condition")

    # right boundary condition
    if by[1] == "A":
        ar[-1] = 1
    elif by[1] == "S":
        sr[-1] = 1
    else:
        raise ValueError("unrecognized right boundary condition")

    # ciclo sui layer
    # maxbetay = numpy.max(numpy.real(betay))
    n1 = numpy.argmax(numpy.real(betay)) + 1
    if n1 == Nregions:
        n1 = Nregions - 1
    n2 = n1 + 1

    modo = FMMMode1dy()

    for m in range(0, len(kz)):
        k = k_[m, :]
        sinkdsuk = sinkdsuk_[m, :][0]
        coskd = coskd_[m, :][0]
        sinkdk = sinkdk_[m, :][0]

        for idx in range(0, n1):
            sr[idx] = sl[idx] * coskd[idx] + al[idx] * sinkdsuk[idx]
            ar[idx] = al[idx] * coskd[idx] - sl[idx] * sinkdk[idx]

            # ******************* requirement of continuity
            if idx < n1 - 1:
                sl[idx + 1] = sr[idx]
                al[idx + 1] = ar[idx] / eps[idx] * eps[idx + 1]
            # *******************

        for idx1 in range(Nregions - 1, n2 - 2, -1):
            sl[idx1] = sr[idx1] * coskd[idx1] - ar[idx1] * sinkdsuk[idx1]
            al[idx1] = ar[idx1] * coskd[idx1] + sr[idx1] * sinkdk[idx1]

            # ******************* requirement of continuity
            # n2 is 1-based, so the last right-to-left layer is idx1 == n2 - 1;
            # propagate across every interface strictly to the right of it.
            if idx1 > n2 - 1:
                sr[idx1 - 1] = sl[idx1]
                ar[idx1 - 1] = al[idx1] / eps[idx1] * eps[idx1 - 1]
            # *******************

        Delta[m] = (
            eps[n1 - 1] * sr[n1 - 1] * al[n2 - 1]
            - eps[n2 - 1] * ar[n1 - 1] * sl[n2 - 1]
        )

        if len(kz) < 2:
            # normalize and save only if len(kz) == 1
            # otherwise, modo is ignored and only Delta is useful

            # Match left and right shooting branches at interface Uy[n1].
            # Use field ratio sr/sl unless the mode has a node (or near-node)
            # at the interface, in which case match the continuous flux a/eps.
            k_scale = max(float(numpy.max(numpy.abs(betay))), 1.0)
            if abs(sl[n2 - 1]) * k_scale >= abs(al[n2 - 1]):
                alfa = sr[n1 - 1] / sl[n2 - 1]
            else:
                alfa = (ar[n1 - 1] / eps[n1 - 1]) / (al[n2 - 1] / eps[n2 - 1])
            sl[n2 - 1 :] *= alfa
            sr[n2 - 1 :] *= alfa
            al[n2 - 1 :] *= alfa
            ar[n2 - 1 :] *= alfa

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
    betay = FMMpars["beta"]
    Uy = FMMpars["Uy"]
    by = FMMpars["boundary"]

    kz = numpy.atleast_1d(kz_)

    Nregions = len(betay)
    d = numpy.diff(Uy)
    Delta = numpy.zeros(kz.shape, dtype=complex)

    sl = numpy.zeros(Nregions, dtype=complex)
    sr = numpy.zeros_like(sl)
    al = numpy.zeros_like(sl)
    ar = numpy.zeros_like(sl)

    k_ = numpy.emath.sqrt(betay**2 - kz[:, numpy.newaxis] ** 2)
    kd = k_[:, numpy.newaxis] * d
    sinkdsuk_ = sinxsux(kd) * d
    coskd_ = numpy.cos(kd)
    sinkdk_ = numpy.sin(kd) * k_[:, numpy.newaxis]

    # left boundary condition
    if by[0] == "A":
        al[0] = 1
    elif by[0] == "S":
        sl[0] = 1
    else:
        raise ValueError("unrecognized left boundary condition")

    # right boundary condition
    if by[1] == "A":
        ar[-1] = 1
    elif by[1] == "S":
        sr[-1] = 1
    else:
        raise ValueError("unrecognized right boundary condition")

    # ciclo sui layer
    # maxbetay = numpy.max(numpy.real(betay))
    n1 = numpy.argmax(numpy.real(betay)) + 1
    if n1 == Nregions:
        n1 = Nregions - 1
    n2 = n1 + 1

    modo = FMMMode1dy()

    for m in range(0, len(kz)):
        k = k_[m, :]
        sinkdsuk = sinkdsuk_[m, :][0]
        coskd = coskd_[m, :][0]
        sinkdk = sinkdk_[m, :][0]

        for idx in range(0, n1):
            sr[idx] = sl[idx] * coskd[idx] + al[idx] * sinkdsuk[idx]
            ar[idx] = al[idx] * coskd[idx] - sl[idx] * sinkdk[idx]

            # ******************* requirement of continuity
            if idx < n1 - 1:
                sl[idx + 1] = sr[idx]
                al[idx + 1] = ar[idx]
            # *******************

        for idx1 in range(Nregions - 1, n2 - 2, -1):
            sl[idx1] = sr[idx1] * coskd[idx1] - ar[idx1] * sinkdsuk[idx1]
            al[idx1] = ar[idx1] * coskd[idx1] + sr[idx1] * sinkdk[idx1]

            # ******************* requirement of continuity
            # n2 is 1-based, so the last right-to-left layer is idx1 == n2 - 1;
            # propagate across every interface strictly to the right of it.
            if idx1 > n2 - 1:
                sr[idx1 - 1] = sl[idx1]
                ar[idx1 - 1] = al[idx1]
            # *******************
        Delta[m] = sr[n1 - 1] * al[n2 - 1] - ar[n1 - 1] * sl[n2 - 1]

    #    len_kz = len(kz)
    #    k = k_[0,:]
    #    sinkdsuk = sinkdsuk_[0,:][0]
    #    coskd = coskd_[0,:][0]
    #    sinkdk = sinkdk_[0,:][0]
    #    code = """
    #            for (int m = 0; m < len_kz; ++m) {
    #                //k = k_(m,:);
    #                //sinkdsuk = sinkdsuk_(0,:);
    #                //coskd = coskd_(0,:);
    #                //sinkdk = sinkdk_(0,:);
    #                int nn1 = int(n1);
    #                for (int idx = 0; idx < nn1; ++idx) {
    #                    sr(idx) = sl(idx) * coskd(idx) + al(idx) * sinkdsuk(idx);
    #                    ar(idx) = al(idx) * coskd(idx) - sl(idx) * sinkdk(idx);
    #                    if (idx < nn1 - 1) {
    #                        sl(idx + 1) = sr(idx);
    #                        al(idx + 1) = ar(idx);
    #                    }
    #                }
    #                int nn2 = int(n2);
    #                for (int idx1 = Nregions - 1; idx1 > nn2 - 2; --idx1) {
    #                    sl(idx1) = sr(idx1) * coskd(idx1) - ar(idx1) * sinkdsuk(idx1);
    #                    al(idx1) = ar(idx1) * coskd(idx1) + sr(idx1) * sinkdk(idx1);
    #                    if (idx1 > nn2) {
    #                        sr(idx1 - 1) = sl(idx1);
    #                        ar(idx1 - 1) = al(idx1);
    #                    }
    #                }
    #                //Delta(m) = std::complex<double>(1) * (sr(nn1-1) * al(nn2-1) - ar(nn1-1) * sl(nn2-1));
    #            }
    #            """
    #
    #    from scipy import weave
    #    from scipy.weave import converters
    #    weave.inline(code,
    #                 ['n1', 'n2', 'Nregions', 'sl', 'sr', 'al', 'ar', 'len_kz', 'Delta',
    #                  'k', 'sinkdsuk', 'sinkdk', 'coskd',
    #                  'k_', 'sinkdsuk_', 'sinkdk_', 'coskd_'],
    #                 type_converters = converters.blitz,
    #                 compiler = 'gcc')

    if len(kz) < 2:
        # normalize and save only if len(kz) == 1
        # otherwise, modo is ignored and only Delta is useful

        # Match left and right shooting branches at interface Uy[n1].
        # Use field ratio sr/sl unless the mode has a node (or near-node)
        # at the interface, in which case match the continuous derivative a.
        k_scale = max(float(numpy.max(numpy.abs(betay))), 1.0)
        if abs(sl[n2 - 1]) * k_scale >= abs(al[n2 - 1]):
            alfa = sr[n1 - 1] / sl[n2 - 1]
        else:
            alfa = ar[n1 - 1] / al[n2 - 1]

        sl[n2 - 1 :] *= alfa
        sr[n2 - 1 :] *= alfa
        al[n2 - 1 :] *= alfa
        ar[n2 - 1 :] *= alfa

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

    # Detect sharp lossless dips (dy > 0.005) as well as broad local minima
    # produced by off-axis complex roots in lossy waveguides, where discrete
    # log-curvature scales as (dx / Im(kz))^2 and can fall below 0.005.
    y_safe = numpy.maximum(numpy.real(y), 1e-300)
    dy = numpy.r_[0.0, numpy.diff(numpy.diff(numpy.log(y_safe))), 0.0]
    is_local_min = numpy.r_[
        False,
        (y_safe[1:-1] < y_safe[:-2]) & (y_safe[1:-1] <= y_safe[2:]),
        False,
    ]
    idy = numpy.where((dy > 0.005) | (is_local_min & (dy > 1e-12)))[0]

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
            z1[0] = zeri[0] - searchinterval / 2
            z2[0] = zeri[0] + searchinterval / 2
        else:
            delta = numpy.min([dz[0], searchinterval])
            z1[0] = zeri[0] - delta / 2
            z2[0] = zeri[0] + delta / 2

            for idx in range(1, len(zeri) - 1):
                delta = numpy.min([dz[idx - 1], dz[idx], searchinterval])
                z1[idx] = zeri[idx] - delta / 2
                z2[idx] = zeri[idx] + delta / 2

            delta = numpy.min([dz[-1], searchinterval])
            z1[-1] = zeri[-1] - delta / 2
            z2[-1] = zeri[-1] + delta / 2

    return (zeri, z1, z2)


def absfzzero2(t, f, xmin, xmax, ymin, ymax):
    xmean = numpy.mean([xmin, xmax])
    ymean = numpy.mean([ymin, ymax])
    xwidth = xmax - xmin
    ywidth = ymax - ymin
    x = xmean + xwidth * t[0] / (1.0 + numpy.abs(t[0])) / 2.0
    y = ymean + ywidth * t[1] / (1.0 + numpy.abs(t[1])) / 2.0
    z = x + 1j * y
    fv = f(z)
    return numpy.abs(fv) ** 2


def fzzeroabs2(f, zmin, zmax):
    xmin = numpy.real(zmin)
    ymin = numpy.imag(zmin)
    xmax = numpy.real(zmax)
    ymax = numpy.imag(zmax)

    tx0 = 0.0
    ty0 = 0.0

    t0 = scipy.optimize.fmin(
        lambda t: absfzzero2(t, f, xmin, xmax, ymin, ymax),
        [tx0, ty0],
        maxiter=100000,
        maxfun=100000,
        xtol=1e-15,
        ftol=1e-15,
        disp=0,
    )

    xmean = numpy.mean([xmin, xmax])
    ymean = numpy.mean([ymin, ymax])
    xwidth = xmax - xmin
    ywidth = ymax - ymin
    x0 = xmean + xwidth * t0[0] / (1 + numpy.abs(t0[0])) / 2
    y0 = ymean + ywidth * t0[1] / (1 + numpy.abs(t0[1])) / 2

    z0 = x0 + 1j * y0
    valf = f(z0)

    return (z0, valf)


def _conj_mode1d(modo):
    """Return an FMMMode1d evaluating to conj(modo(u)) for real u."""
    mc = copy.deepcopy(modo)
    mc.k = numpy.conj(modo.k)
    mc.sl = numpy.conj(modo.sl)
    mc.sr = numpy.conj(modo.sr)
    mc.al = numpy.conj(modo.al)
    mc.ar = numpy.conj(modo.ar)
    return mc


def _eval_segment(k, sl, sr, al, d, s):
    """Evaluate Helmholtz segment u'' + k^2 u = 0 and u' at offset s in [0, d].

    Uses left-endpoint (sl, al) when |Im(k * d)| <= 1 (exact at k = 0 and sin(k*d) = 0),
    and symmetric two-endpoint (sl, sr) decaying exponentials when |Im(k * d)| > 1 so
    all exponentials stay bounded by 1 with no O(exp(|Im(k * d)|)) cancellation.
    """
    if d == 0.0:
        return sl + 0.0 * s, al + 0.0 * s
    if abs(numpy.imag(k * d)) > 1.0:
        # Choose k_d with Im(k_d) < 0 so |exp(-1j * k_d * s)| <= 1 for s >= 0
        k_d = k if numpy.imag(k) < 0 else -k
        p = numpy.exp(-1j * k_d * d)
        denom = 1.0 - p**2
        eL = numpy.exp(-1j * k_d * s)
        eR = numpy.exp(-1j * k_d * (d - s))
        u = (sl * (eL - p * eR) + sr * (eR - p * eL)) / denom
        du = -1j * k_d * (sl * (eL + p * eR) - sr * (eR + p * eL)) / denom
        return u, du
    u = sl * numpy.cos(k * s) + al * s * sinxsux(k * s)
    du = -k * sl * numpy.sin(k * s) + al * numpy.cos(k * s)
    return u, du


def _slice_mode1dx(modo_x, mx, k0, kz):
    """Extract single-slice x-mode on [U[mx], U[mx + 1]] for longitudinal wavevector kz.

    Computes right-endpoint (sr, ar) from (k, sl, al) when |Im(k * dx)| <= 1, and
    preserves bounded endpoints (sl, sr) from creacoeffx3 when |Im(k * dx)| > 1 so the
    Sturm-Liouville Wronskian in scalarprod stays free of evanescent cancellation.
    """
    m = FMMMode1dx()
    kf = modo_x.k[mx]
    k = numpy.emath.sqrt(kf**2 - kz**2)
    sl = modo_x.sl[mx] * (k0 / kf) ** 2
    sr_stored = modo_x.sr[mx] * (k0 / kf) ** 2
    al_stored = modo_x.al[mx]
    U_slice = numpy.asarray(modo_x.U[mx : mx + 2], dtype=float)
    dx = U_slice[1] - U_slice[0]
    if abs(numpy.imag(k * dx)) > 1.0:
        sr = sr_stored
        _, al = _eval_segment(k, sl, sr, al_stored, dx, 0.0)
        _, ar = _eval_segment(k, sl, sr, al_stored, dx, dx)
    else:
        al = al_stored
        sr, ar = _eval_segment(k, sl, sr_stored, al, dx, dx)
    m.k = numpy.atleast_1d(k)
    m.sl = numpy.atleast_1d(sl)
    m.al = numpy.atleast_1d(al)
    m.sr = numpy.atleast_1d(sr)
    m.ar = numpy.atleast_1d(ar)
    m.U = U_slice
    return m, kf


def _refine_mode1d(modo, U_new):
    """Restrict/refine a piecewise-sinusoidal FMMMode1d onto knot vector U_new."""
    U_old = numpy.asarray(modo.U, dtype=float)
    U_new = numpy.asarray(U_new, dtype=float)
    if numpy.array_equal(U_old, U_new):
        return copy.deepcopy(modo)

    n_seg = len(U_new) - 1
    dtype = numpy.result_type(modo.k, modo.sl, modo.sr, modo.al, modo.ar)
    m = copy.copy(modo)
    m.U = U_new.copy()
    m.k = numpy.empty(n_seg, dtype=dtype)
    m.sl = numpy.empty(n_seg, dtype=dtype)
    m.sr = numpy.empty(n_seg, dtype=dtype)
    m.al = numpy.empty(n_seg, dtype=dtype)
    m.ar = numpy.empty(n_seg, dtype=dtype)

    for i in range(n_seg):
        ua = U_new[i]
        ub = U_new[i + 1]
        u_mid = 0.5 * (ua + ub)
        idx = int(
            numpy.clip(
                numpy.searchsorted(U_old, u_mid, side="right") - 1,
                0,
                len(modo.sl) - 1,
            )
        )
        k = modo.k[idx]
        sl = modo.sl[idx]
        sr = modo.sr[idx]
        al = modo.al[idx]
        d_seg = U_old[idx + 1] - U_old[idx]
        da = ua - U_old[idx]
        db = ub - U_old[idx]

        m.k[i] = k
        m.sl[i], m.al[i] = _eval_segment(k, sl, sr, al, d_seg, da)
        m.sr[i], m.ar[i] = _eval_segment(k, sl, sr, al, d_seg, db)

    return m


def scalarprod(modo1, modo2):
    """Exact unconjugated 1D integral int modo1(u) * modo2(u) du across layers.

    Kept unconjugated for Sudbo's bi-orthogonal 1D slice formulation on lossy media;
    pass _conj_mode1d(modo2) when a Hermitian L2 inner product is needed. Uses the
    Sturm-Liouville Wronskian when k1^2 != k2^2 and a smooth sinxsux midpoint form
    near k1 = +-k2 (including k = 0) to avoid removable singularities.
    """
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
    dtype = numpy.result_type(sl1, al1, sr1, ar1, sl2, al2, sr2, ar2, ky1, ky2)
    scprod = numpy.zeros(Nlayers, dtype=dtype)

    for idy in range(Nlayers):
        di = d[idy]
        k1 = ky1[idy]
        k2 = ky2[idy]
        diff_k2 = k1**2 - k2**2
        scale = 1.0 + abs(k1 * di) ** 2 + abs(k2 * di) ** 2
        if abs(diff_k2) * (di**2) > 1e-10 * scale:
            scprod[idy] = (
                sr1[idy] * ar2[idy]
                - ar1[idy] * sr2[idy]
                - sl1[idy] * al2[idy]
                + al1[idy] * sl2[idy]
            ) / diff_k2
        else:
            k = numpy.emath.sqrt(0.5 * (k1**2 + k2**2))
            kd = k * di
            if abs(numpy.imag(kd)) > 1.0:
                if numpy.imag(kd) > 0:
                    k = -k
                    kd = -kd
                p = numpy.exp(-1j * kd)
                p2 = p * p
                one_m_p2_sq = (1.0 - p2) ** 2
                i_diag = -(1j * (1.0 - p2 * p2) + 4.0 * kd * p2) / (
                    2.0 * k * one_m_p2_sq
                )
                i_cross = p * (kd * (1.0 + p2) + 1j * (1.0 - p2)) / (k * one_m_p2_sq)
                res = (sl1[idy] * sl2[idy] + sr1[idy] * sr2[idy]) * i_diag + (
                    sl1[idy] * sr2[idy] + sr1[idy] * sl2[idy]
                ) * i_cross
            else:
                s2kd = sinxsux(2.0 * kd)
                skd = sinxsux(kd)
                icc = 0.5 * di * (1.0 + s2kd)
                isc = 0.5 * (di**2) * (skd**2)
                if abs(kd) < 1e-2:
                    kd2 = kd**2
                    iss = (di**3) * (
                        1.0 / 3.0
                        - kd2 / 15.0
                        + 2.0 * (kd2**2) / 315.0
                        - (kd2**3) / 2835.0
                    )
                else:
                    iss = 0.5 * di * (1.0 - s2kd) / (k**2)
                res = (
                    sl1[idy] * sl2[idy] * icc
                    + (sl1[idy] * al2[idy] + al1[idy] * sl2[idy]) * isc
                    + al1[idy] * al2[idy] * iss
                )
            scprod[idy] = (
                numpy.real(res) if numpy.issubdtype(dtype, numpy.floating) else res
            )

    return numpy.sum(scprod)


def sueps(modo):
    eps = modo.pars["epsilon"]
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
    max_re_betay = float(numpy.max(numpy.real(betay)))
    searchinterval = max(
        max((25.0 / numpy.pi) * k0, 2.0 * max_re_betay) / Nstepskz,
        float(numpy.abs(numpy.min(numpy.imag(2.0 * betay)))),
    )
    imsearchinterval = 10 * k0
    # ypointsperregion = 5000

    FMMpars = {"epsilon": ny**2, "beta": betay, "boundary": boundaryRL, "Uy": Uy}

    # analytical solution
    if numpy.allclose(ny, ny[0]):
        Message("Uniform slice found: using analytical solution.", 2).show(verbosity)
        return analyticalsolution(nmodi, TETM, FMMpars)

    # rekz = numpy.linspace(numpy.max(numpy.real(betay)) + searchinterval, 0., Nstepskz)
    rekz2 = numpy.linspace(
        (numpy.max(numpy.real(betay)) + searchinterval) ** 2, 0.0, Nstepskz
    )
    rekz = numpy.emath.sqrt(rekz2)
    if TETM == "TM":
        Message("Shooting TM.", 3).show(verbosity)
        matchingre, modotmp = FMMshootingTM(rekz, FMMpars)
    else:
        Message("Shooting TE.", 3).show(verbosity)
        matchingre, modotmp = FMMshooting(rekz, FMMpars)

    # nre = rekz / k0
    nre2 = rekz2 / k0**2
    # zerire, z1, z2 = findzerosnew(nre, numpy.abs(matchingre), searchinterval / k0)
    zerire2, z12, z22 = findzerosnew(
        nre2, numpy.abs(matchingre), (searchinterval / k0) ** 2
    )
    zerire = numpy.emath.sqrt(zerire2)
    kz1 = zerire * k0 - searchinterval / 2.0 + 1j * imsearchinterval
    kz2 = zerire * k0 + searchinterval / 2.0 - 1j * imsearchinterval
    Message("Found %d real zeros." % len(zerire), 2).show(verbosity)
    if len(zerire) < nmodi:
        Message("Number of real zeros not enough: scan imaginary axis.", 2).show(
            verbosity
        )
        imkza = -numpy.max(numpy.real(betay))
        imkzb = 0.0
        while len(kz1) < nmodi:
            imkza = imkza + numpy.max(numpy.real(betay))
            imkzb = imkzb + numpy.max(numpy.real(betay))
            # imkz = numpy.linspace(imkza, imkzb, Nstepskz)
            imkz2 = numpy.linspace(imkza**2, imkzb**2, Nstepskz)
            imkz = numpy.emath.sqrt(imkz2)
            if TETM == "TM":
                matchingim, modotmp = FMMshootingTM(-1j * imkz, FMMpars)
            else:
                matchingim, modotmp = FMMshooting(-1j * imkz, FMMpars)
            # nim = imkz * wl / 2.0 / numpy.pi
            nim2 = imkz2 * (wl / 2.0 / numpy.pi) ** 2
            # zeriim, z1, z2 = findzerosnew(nim, numpy.abs(matchingim), searchinterval / k0)
            zeriim2, z12, z22 = findzerosnew(
                nim2, numpy.abs(matchingim), (searchinterval / k0) ** 2
            )
            zeriim = numpy.emath.sqrt(zeriim2)
            Message("Found %d imag zeros." % len(zeriim), 2).show(verbosity)
            # Scan the passive negative imaginary axis (-i * imkz) so lossy
            # cutoff modes converge in Quadrant IV (Re >= 0, Im <= 0).
            kz1 = numpy.r_[
                kz1,
                -1j * (zeriim * k0 - imsearchinterval / 2.0) - searchinterval / 2.0,
            ]
            kz2 = numpy.r_[
                kz2,
                -1j * (zeriim * k0 + imsearchinterval / 2.0) + searchinterval / 2.0,
            ]

    mk = 0
    modi = []
    # inizia il ciclo sugli intervalli
    Message("Refine zeros.", 2).show(verbosity)
    for m in range(0, len(kz1)):
        if mk == nmodi:
            break
        shoot_fn = (
            (lambda kz: FMMshootingTM(kz, FMMpars)[0])
            if TETM == "TM"
            else (lambda kz: FMMshooting(kz, FMMpars)[0])
        )
        scale_0 = max(
            float(numpy.max(numpy.abs(shoot_fn(0.5 * (kz1[m] + kz2[m]))))), 1e-30
        )
        z0, valf = fzzeroabs2(lambda kz: shoot_fn(kz) / scale_0, kz1[m], kz2[m])
        eps_box = 1e-7 * k0
        scale_1 = max(float(numpy.max(numpy.abs(shoot_fn(z0 + eps_box)))), 1e-30)
        z0, valf = fzzeroabs2(
            lambda kz: shoot_fn(kz) / scale_1,
            z0 - eps_box + 1j * eps_box,
            z0 + eps_box - 1j * eps_box,
        )
        z0 = numpy.atleast_1d(z0)
        flip_z0 = (z0.real - z0.imag < 0) | ((z0.real - z0.imag == 0) & (z0.real < 0))
        z0 = numpy.where(flip_z0, -z0, z0)

        if len(z0) > 0:
            if TETM == "TM":
                zero, modo = FMMshootingTM(z0, FMMpars)
            else:
                zero, modo = FMMshooting(z0, FMMpars)

            if TETM == "TM":
                normalizzazione = 1.0 / numpy.sqrt(scalarprod(modo, sueps(modo)))
            else:
                normalizzazione = 1.0 / numpy.sqrt(scalarprod(modo, modo))

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
        Message("Finding 1dmodes TE.", 1).show(verbosity)
        ymodih = FMM1d_y(
            Uy, refindex[m, :], wl, nmodislices, boundary.yh, "TE", verbosity
        )
        Message("Finding 1dmodes TM.", 1).show(verbosity)
        ymodie = FMM1d_y(
            Uy, refindex[m, :], wl, nmodislices, boundary.ye, "TM", verbosity
        )
        slice = Slice(
            x1=Ux[m], x2=Ux[m + 1], Uy=Uy, boundary=boundary, modie=ymodie, modih=ymodih
        )
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
    mododot.al = -(k**2) * modo.sl
    mododot.ar = -(k**2) * modo.sr
    return mododot


def genera_rotazione(slices):
    nmodi = len(slices[0].modih)
    k0 = 2 * numpy.pi / slices[0].wl
    Nslices = len(slices)

    R = Struct()
    # alloc R
    R.Ree = numpy.zeros((nmodi, nmodi, Nslices - 1), dtype=complex)
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
    Nslices = len(slices)

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
        neesueps[idx] = numpy.linalg.norm(M.eesueps[:, :, idx] - Ident)
        nhh[idx] = numpy.linalg.norm(M.hh[:, :, idx] - Ident)
        nRhe[idx] = numpy.linalg.norm(M.Rhe[:, :, idx])

    for idx in range(Nslices - 1):
        nRee[idx] = numpy.linalg.norm(
            numpy.dot(R.Ree[:, :, idx], R.Reem[:, :, idx]) - Ident
        )
        nRhh[idx] = numpy.linalg.norm(
            numpy.dot(R.Rhh[:, :, idx], R.Rhhm[:, :, idx]) - Ident
        )
        nAC[idx] = numpy.linalg.norm(
            numpy.dot(R.Rhe[:, :, idx], R.Reem[:, :, idx])
            + numpy.dot(R.Rhh[:, :, idx], R.Rhem[:, :, idx])
        )

    ns1 = numpy.linalg.norm(numpy.r_[neesueps, nhh, nRhe])
    ns2 = numpy.linalg.norm(numpy.r_[nRee, nRhh, nAC])
    errore = numpy.linalg.norm(numpy.r_[ns1, ns2]) / numpy.emath.sqrt(8 * nmodi)
    return errore


def method_of_component(kz_, slices, Rot, uscelto=None, icomp=None):
    kz = numpy.atleast_1d(kz_)
    k0 = 2.0 * numpy.pi / slices[0].wl
    abscomp = numpy.zeros(len(kz))
    #    tmp = 500 # OKKIO: perche' 500?
    tmp = (
        100 * len(slices[0].modie) * (len(slices) - 1)
    )  # OKKIO: dimension of Mvec * 50. enough?
    rng = numpy.random.default_rng(0)
    for m in range(len(kz)):
        M = Mvec(kz[m], slices, Rot) / k0
        if (uscelto is None) and (icomp is None):
            urn = numpy.zeros((M.shape[0], tmp), dtype=complex)
            normu = numpy.zeros(tmp, dtype=complex)
            for k in range(tmp):
                ur = rng.random(M.shape[0])
                urn[:, k] = ur / numpy.linalg.norm(ur)
                normu[k] = numpy.linalg.norm(numpy.dot(M, urn[:, k]))
            iurn = numpy.argmin(normu)
            uscelto = urn[:, iurn]
            icomp = numpy.argmax(uscelto)

        Mmeno1u = numpy.linalg.solve(M, uscelto)
        abscomp[m] = 1.0 / numpy.linalg.norm(Mmeno1u)

    return (abscomp, uscelto, icomp)


def creaTeThSeSh(kz, slices):
    Nslices = len(slices)
    nmodi = len(slices[0].modie)
    d = numpy.array([s.x2 - s.x1 for s in slices])
    k0 = 2.0 * numpy.pi / slices[0].wl

    Th = numpy.zeros((nmodi, Nslices), dtype=complex)
    Sh = numpy.zeros_like(Th)
    Te = numpy.zeros_like(Th)
    Se = numpy.zeros_like(Th)
    Thleft = numpy.zeros_like(Th)
    Teleft = numpy.zeros_like(Th)
    Thright = numpy.zeros_like(Th)
    Teright = numpy.zeros_like(Th)

    def _kx_cot_csc(kx, width):
        u = numpy.asarray(kx * width, dtype=complex)
        small = numpy.abs(u) < 1e-6
        large_im = numpy.abs(u.imag) > 100.0
        u_safe = numpy.where(small | large_im, 1.0 + 0.0j, u)
        cot_term = kx / numpy.tan(u_safe)
        csc_term = kx / numpy.sin(u_safe)
        s_im = numpy.where(u.imag >= 0, 1.0, -1.0)
        cot_term = numpy.where(
            small,
            (1.0 - u**2 / 3.0) / width,
            numpy.where(large_im, -1j * s_im * kx, cot_term),
        )
        csc_term = numpy.where(
            small,
            (1.0 + u**2 / 6.0) / width,
            numpy.where(large_im, 0.0 + 0.0j, csc_term),
        )
        return cot_term, csc_term

    for idx in range(Nslices):
        ke = numpy.array([m.keff.item() for m in slices[idx].modie])
        kh = numpy.array([m.keff.item() for m in slices[idx].modih])

        kxh = numpy.emath.sqrt(kh**2 - kz**2)
        kxe = numpy.emath.sqrt(ke**2 - kz**2)

        cot_h, csc_h = _kx_cot_csc(kxh, d[idx])
        cot_e, csc_e = _kx_cot_csc(kxe, d[idx])
        Th[:, idx] = (k0 / kh) ** 2 * cot_h
        Sh[:, idx] = (k0 / kh) ** 2 * csc_h
        Te[:, idx] = (k0 / ke) ** 2 * cot_e
        Se[:, idx] = (k0 / ke) ** 2 * csc_e

    ke = numpy.array([m.keff.item() for m in slices[0].modie])
    kh = numpy.array([m.keff.item() for m in slices[0].modih])
    kxh = numpy.emath.sqrt(kh**2 - kz**2)
    kxe = numpy.emath.sqrt(ke**2 - kz**2)

    if slices[0].boundary.xleft == "Electric Wall":
        # ah = 0
        Thleft = -((k0 / kh) ** 2) * kxh * numpy.tan(kxh * d[0])
        Teleft = Te[:, 0]
    else:
        # ae = 0
        Teleft = -((k0 / ke) ** 2) * kxe * numpy.tan(kxe * d[0])
        Thleft = Th[:, 0]

    ke = numpy.array([m.keff.item() for m in slices[-1].modie])
    kh = numpy.array([m.keff.item() for m in slices[-1].modih])
    kxh = numpy.emath.sqrt(kh**2 - kz**2)
    kxe = numpy.emath.sqrt(ke**2 - kz**2)

    if slices[-1].boundary.xright == "Electric Wall":
        # ah = 0
        Thright = -((k0 / kh) ** 2) * kxh * numpy.tan(kxh * d[-1])
        Teright = Te[:, -1]
    else:
        # ae = 0
        Teright = -((k0 / ke) ** 2) * kxe * numpy.tan(kxe * d[-1])
        Thright = Th[:, -1]

    return (Te, Th, Se, Sh, Teleft, Teright, Thleft, Thright)


def Mvec(kz, slices, R):
    Nslices = len(slices)
    Rhh = R.Rhh
    Ree = R.Ree
    Rhe = R.Rhe
    Rhem = R.Rhem
    Rhhm = R.Rhhm

    Te, Th, Se, Sh, Teleft, Teright, Thleft, Thright = creaTeThSeSh(kz, slices)
    Te[:, 0] = Teleft
    Te[:, -1] = Teright
    Th[:, 0] = Thleft
    Th[:, -1] = Thright

    # case Nslices=2
    if Nslices == 2:
        raise NotImplementedError("2 slices not implemented yet.")

    dim1 = Th.shape[0]
    M = numpy.zeros((2 * dim1 * (Nslices - 1), 2 * dim1 * (Nslices - 1)), dtype=complex)
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

        # Convert 1-based slice index idx to 0-based i (matching the idx=2 and
        # idx=Nslices blocks below) before indexing Ree/Rhh/Te/Th/Se/Sh.
        i = idx - 1

        M[numpy.ix_(IAe, JAe)] = numpy.dot(
            Ree[:, :, i - 1].T * Te[:, i - 1], Ree[:, :, i - 1]
        ) + numpy.diag(Te[:, i])
        M[numpy.ix_(IBe, JBe)] = -Ree[:, :, i - 1].T * Se[:, i - 1]
        M[numpy.ix_(ICe, JCe)] = -Se[:, numpy.newaxis, i] * Ree[:, :, i]
        M[numpy.ix_(IDe, JDe)] = -kz * numpy.dot(
            Ree[:, :, i - 1].T, Rhem[:, :, i - 1].T
        )

        M[numpy.ix_(IAh, JAh)] = numpy.dot(
            Rhhm[:, :, i - 1] * Th[:, i - 1], Rhh[:, :, i - 1]
        ) + numpy.diag(Th[:, i])
        M[numpy.ix_(IBh, JBh)] = -Rhhm[:, :, i - 1] * Sh[:, i - 1]
        M[numpy.ix_(ICh, JCh)] = -Sh[:, numpy.newaxis, i] * Rhh[:, :, i]
        M[numpy.ix_(IDh, JDh)] = kz * numpy.dot(Rhhm[:, :, i - 1], Rhe[:, :, i - 1])

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

    M[numpy.ix_(IAe, JAe)] = numpy.dot(
        Ree[:, :, idx - 1].T * Te[:, idx - 1], Ree[:, :, idx - 1]
    ) + numpy.diag(Te[:, idx])
    M[numpy.ix_(ICe, JCe)] = -Se[:, numpy.newaxis, idx] * Ree[:, :, idx]
    M[numpy.ix_(IDe, JDe)] = -kz * numpy.dot(
        Ree[:, :, idx - 1].T, Rhem[:, :, idx - 1].T
    )

    M[numpy.ix_(IAh, JAh)] = numpy.dot(
        Rhhm[:, :, idx - 1] * Th[:, idx - 1], Rhh[:, :, idx - 1]
    ) + numpy.diag(Th[:, idx])
    M[numpy.ix_(ICh, JCh)] = -Sh[:, numpy.newaxis, idx] * Rhh[:, :, idx]
    M[numpy.ix_(IDh, JDh)] = kz * numpy.dot(Rhhm[:, :, idx - 1], Rhe[:, :, idx - 1])

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

    M[numpy.ix_(IAe, JAe)] = numpy.dot(
        Ree[:, :, idx - 1].T * Te[:, idx - 1], Ree[:, :, idx - 1]
    ) + numpy.diag(Te[:, idx])
    M[numpy.ix_(IBe, JBe)] = -Ree[:, :, idx - 1].T * Se[:, idx - 1]
    M[numpy.ix_(IDe, JDe)] = -kz * numpy.dot(
        Ree[:, :, idx - 1].T, Rhem[:, :, idx - 1].T
    )

    M[numpy.ix_(IAh, JAh)] = numpy.dot(
        Rhhm[:, :, idx - 1] * Th[:, idx - 1], Rhh[:, :, idx - 1]
    ) + numpy.diag(Th[:, idx])
    M[numpy.ix_(IBh, JBh)] = -Rhhm[:, :, idx - 1] * Sh[:, idx - 1]
    M[numpy.ix_(IDh, JDh)] = kz * numpy.dot(Rhhm[:, :, idx - 1], Rhe[:, :, idx - 1])

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
    Nslices = len(slices)
    residuals = []
    for i in range(Nslices - 1):
        residuals.append(
            numpy.linalg.norm(numpy.dot(R.Rhh[:, :, i], Shl[:, i + 1]) - Shr[:, i])
        )
        residuals.append(
            numpy.linalg.norm(numpy.dot(R.Ree[:, :, i], Sel[:, i + 1]) - Ser[:, i])
        )
        residuals.append(
            numpy.linalg.norm(
                numpy.dot(R.Rhh[:, :, i], Ahl[:, i + 1])
                - kz * numpy.dot(R.Rhe[:, :, i], Sel[:, i + 1])
                - Ahr[:, i]
            )
        )
        residuals.append(
            numpy.linalg.norm(
                numpy.dot(R.Reem[:, :, i].T, Ael[:, i + 1])
                + kz * numpy.dot(R.Rhem[:, :, i].T, Shl[:, i + 1])
                - Aer[:, i]
            )
        )
    for i in range(Nslices):
        residuals.append(
            numpy.linalg.norm(-Te[:, i] * Sel[:, i] + Se[:, i] * Ser[:, i] - Ael[:, i])
        )
        residuals.append(
            numpy.linalg.norm(-Th[:, i] * Shl[:, i] + Sh[:, i] * Shr[:, i] - Ahl[:, i])
        )
        residuals.append(
            numpy.linalg.norm(-Sh[:, i] * Shl[:, i] + Th[:, i] * Shr[:, i] - Ahr[:, i])
        )
        residuals.append(
            numpy.linalg.norm(-Se[:, i] * Sel[:, i] + Te[:, i] * Ser[:, i] - Aer[:, i])
        )

    N = numpy.linalg.norm(numpy.array(residuals))
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

    # sl2end = numpy.reshape(solution, (nmodi, 2 * (Nslices - 1)))
    sl2end = numpy.reshape(solution, (2 * (Nslices - 1), nmodi)).T
    idxslices = 2 * numpy.arange((Nslices - 1))
    sle2end = sl2end[:, idxslices]
    slh2end = sl2end[:, idxslices + 1]

    ale = numpy.zeros((nmodi, Nslices), dtype=complex)
    sre = numpy.zeros_like(ale)
    are = numpy.zeros_like(ale)
    alh = numpy.zeros_like(ale)
    srh = numpy.zeros_like(ale)
    arh = numpy.zeros_like(ale)

    if (xl == "Electric Wall") & (xr == "Electric Wall"):
        sle = numpy.c_[numpy.zeros((nmodi, 1)), sle2end]
        sre[:, -1] = numpy.zeros(nmodi)
        are[:, -1] = -Se[:, -1] * sle[:, -1]
        ale[:, -1] = -Te[:, -1] * sle[:, -1]

        slh = numpy.c_[numpy.zeros((nmodi, 1)), slh2end]
        arh[:, -1] = numpy.zeros(nmodi)
        srh[:, -1] = Sh[:, -1] / Th[:, -1] * slh[:, -1]
        alh[:, -1] = -Th[:, -1] * slh[:, -1] + Sh[:, -1] * srh[:, -1]

        for idx in range(1, Nslices):
            sre[:, idx - 1] = numpy.dot(Ree[:, :, idx - 1], sle[:, idx])
            srh[:, idx - 1] = numpy.dot(Rhh[:, :, idx - 1], slh[:, idx])

        slh[:, 0] = Sh[:, 0] / Th[:, 0] * srh[:, 0]

        for idx in range(Nslices - 1, 0, -1):
            are[:, idx - 1] = numpy.dot(
                Reem[:, :, idx - 1].T, ale[:, idx]
            ) + kz * numpy.dot(Rhem[:, :, idx - 1].T, slh[:, idx])
            arh[:, idx - 1] = numpy.dot(
                Rhh[:, :, idx - 1], alh[:, idx]
            ) - kz * numpy.dot(Rhe[:, :, idx - 1], sle[:, idx])
            ale[:, idx - 1] = (
                -Te[:, idx - 1] * sle[:, idx - 1] + Se[:, idx - 1] * sre[:, idx - 1]
            )
            alh[:, idx - 1] = (
                -Th[:, idx - 1] * slh[:, idx - 1] + Sh[:, idx - 1] * srh[:, idx - 1]
            )

    elif (xl == "Electric Wall") & (xr == "Magnetic Wall"):
        sle = numpy.c_[numpy.zeros((nmodi, 1)), sle2end]
        are[:, -1] = numpy.zeros(nmodi)
        sre[:, -1] = Se[:, -1] / Te[:, -1] * sle[:, -1]
        ale[:, -1] = -Te[:, -1] * sle[:, -1] + Se[:, -1] * sre[:, -1]

        slh = numpy.c_[numpy.zeros((nmodi, 1)), slh2end]
        srh[:, -1] = numpy.zeros(nmodi)
        arh[:, -1] = -Sh[:, -1] * slh[:, -1]
        alh[:, -1] = -Th[:, -1] * slh[:, -1]

        for idx in range(1, Nslices):
            sre[:, idx - 1] = numpy.dot(Ree[:, :, idx - 1], sle[:, idx])
            srh[:, idx - 1] = numpy.dot(Rhh[:, :, idx - 1], slh[:, idx])

        slh[:, 0] = Sh[:, 0] / Th[:, 0] * srh[:, 0]

        for idx in range(Nslices - 1, 0, -1):
            are[:, idx - 1] = numpy.dot(
                Reem[:, :, idx - 1].T, ale[:, idx]
            ) + kz * numpy.dot(Rhem[:, :, idx - 1].T, slh[:, idx])
            arh[:, idx - 1] = numpy.dot(
                Rhh[:, :, idx - 1], alh[:, idx]
            ) - kz * numpy.dot(Rhe[:, :, idx - 1], sle[:, idx])
            ale[:, idx - 1] = (
                -Te[:, idx - 1] * sle[:, idx - 1] + Se[:, idx - 1] * sre[:, idx - 1]
            )
            alh[:, idx - 1] = (
                -Th[:, idx - 1] * slh[:, idx - 1] + Sh[:, idx - 1] * srh[:, idx - 1]
            )

    elif (xl == "Magnetic Wall") & (xr == "Electric Wall"):
        sle = numpy.c_[numpy.zeros((nmodi, 1)), sle2end]
        slh = numpy.c_[numpy.zeros((nmodi, 1)), slh2end]

        sre[:, -1] = numpy.zeros(nmodi)
        arh[:, -1] = numpy.zeros(nmodi)
        srh[:, -1] = Sh[:, -1] / Th[:, -1] * slh[:, -1]
        are[:, -1] = -Se[:, -1] * sle[:, -1]
        ale[:, -1] = -Te[:, -1] * sle[:, -1]
        alh[:, -1] = -Th[:, -1] * slh[:, -1] + Sh[:, -1] * srh[:, -1]

        for idx in range(1, Nslices):
            sre[:, idx - 1] = numpy.dot(Ree[:, :, idx - 1], sle[:, idx])
            srh[:, idx - 1] = numpy.dot(Rhh[:, :, idx - 1], slh[:, idx])

        sle[:, 0] = Se[:, 0] / Te[:, 0] * sre[:, 0]

        for idx in range(Nslices - 1, 0, -1):
            are[:, idx - 1] = numpy.dot(
                Reem[:, :, idx - 1].T, ale[:, idx]
            ) + kz * numpy.dot(Rhem[:, :, idx - 1].T, slh[:, idx])
            arh[:, idx - 1] = numpy.dot(
                Rhh[:, :, idx - 1], alh[:, idx]
            ) - kz * numpy.dot(Rhe[:, :, idx - 1], sle[:, idx])
            ale[:, idx - 1] = (
                -Te[:, idx - 1] * sle[:, idx - 1] + Se[:, idx - 1] * sre[:, idx - 1]
            )
            alh[:, idx - 1] = (
                -Th[:, idx - 1] * slh[:, idx - 1] + Sh[:, idx - 1] * srh[:, idx - 1]
            )

    elif (xl == "Magnetic Wall") & (xr == "Magnetic Wall"):
        sle = numpy.c_[numpy.zeros((nmodi, 1)), sle2end]
        slh = numpy.c_[numpy.zeros((nmodi, 1)), slh2end]

        srh[:, -1] = numpy.zeros(nmodi)
        are[:, -1] = numpy.zeros(nmodi)
        sre[:, -1] = Se[:, -1] / Te[:, -1] * sle[:, -1]
        arh[:, -1] = -Sh[:, -1] * slh[:, -1]
        alh[:, -1] = -Th[:, -1] * slh[:, -1]
        ale[:, -1] = -Te[:, -1] * sle[:, -1] + Se[:, -1] * sre[:, -1]

        for idx in range(1, Nslices):
            sre[:, idx - 1] = numpy.dot(Ree[:, :, idx - 1], sle[:, idx])
            srh[:, idx - 1] = numpy.dot(Rhh[:, :, idx - 1], slh[:, idx])

        sle[:, 0] = Se[:, 0] / Te[:, 0] * sre[:, 0]

        for idx in range(Nslices - 1, 0, -1):
            are[:, idx - 1] = numpy.dot(
                Reem[:, :, idx - 1].T, ale[:, idx]
            ) + kz * numpy.dot(Rhem[:, :, idx - 1].T, slh[:, idx])
            arh[:, idx - 1] = numpy.dot(
                Rhh[:, :, idx - 1], alh[:, idx]
            ) - kz * numpy.dot(Rhe[:, :, idx - 1], sle[:, idx])
            ale[:, idx - 1] = (
                -Te[:, idx - 1] * sle[:, idx - 1] + Se[:, idx - 1] * sre[:, idx - 1]
            )
            alh[:, idx - 1] = (
                -Th[:, idx - 1] * slh[:, idx - 1] + Sh[:, idx - 1] * srh[:, idx - 1]
            )

    else:
        raise ValueError("Unrecognized boundary condition")

    modie = []
    modih = []
    kh = numpy.zeros((nmodi, Nslices), dtype=complex)
    ke = numpy.zeros_like(kh)

    for n in range(nmodi):
        for q, slice in enumerate(slices):
            kh[n, q] = slice.modih[n].keff.item()
            ke[n, q] = slice.modie[n].keff.item()

        modoe = FMMMode1dx()
        modoe.sl = sle[n, :]
        modoe.al = ale[n, :]
        modoe.sr = sre[n, :]
        modoe.ar = are[n, :]
        modoe.U = slices[0].Ux
        modoe.k = ke[n, :]
        modie.append(modoe)

        modoh = FMMMode1dx()
        modoh.sl = slh[n, :]
        modoh.al = alh[n, :]
        modoh.sr = srh[n, :]
        modoh.ar = arh[n, :]
        modoh.U = slices[0].Ux
        modoh.k = kh[n, :]
        modih.append(modoh)

    return (modie, modih)


def FMM1d_x_component(slices, nmodi, verbosity=0):
    Message("Matching 1dmodes on slices interfaces.", 1).show(verbosity)

    TolnullEig = 1e-1
    dn_search = 1.5e-3

    # Ux = slices[0].Ux
    lambda0 = slices[0].wl
    k0 = 2.0 * numpy.pi / lambda0
    searchinterval = dn_search * k0

    Rot = genera_rotazione(slices)

    maxrekz = -numpy.inf
    minrekz = numpy.inf
    max_im_keff = 0.0

    for slice in slices:
        keffs_slice = [m.keff for m in slice.modih] + [m.keff for m in slice.modie]
        maxrekz0 = float(numpy.max(numpy.real(keffs_slice)))
        minrekz0 = float(numpy.min(numpy.real(k0 * slice.refractiveindex)))
        max_im_keff = max(
            max_im_keff, float(numpy.max(numpy.abs(numpy.imag(keffs_slice))))
        )
        if maxrekz < maxrekz0:
            maxrekz = maxrekz0
        if minrekz > minrekz0:
            minrekz = minrekz0

    # only guided modes
    #    rekz = numpy.linspace(maxrekz + searchinterval, minrekz - searchinterval, numpy.floor(3. / searchinterval))
    # also radiative modes
    # increase 10./searchinterval if not enough candidate zeros are found
    #    rekz = numpy.linspace(maxrekz + searchinterval, 0.,
    #                          numpy.floor(6. / searchinterval *
    #                                      (1 + (minrekz - searchinterval) / (maxrekz - minrekz + 2 * searchinterval))))
    # make rekz[1]-rekz[0] <= searchinterval
    # OKKIO: forse mettere 3x punti??
    rekz = numpy.linspace(
        maxrekz + searchinterval,
        0.0,
        int(10 * numpy.floor((maxrekz + searchinterval) / searchinterval)),
    )

    abscomp, uscelto, icomp = method_of_component(maxrekz, slices, Rot)
    matchingre = method_of_component(rekz, slices, Rot, uscelto, icomp)[0]
    nre = rekz / k0
    zerire = findzerosnew(nre, numpy.abs(matchingre), 4.0 * dn_search)[0]
    kz_im_half = max(searchinterval, 2.0 * max_im_keff)
    kz1 = k0 * zerire - searchinterval / 2.0 + 1j * kz_im_half
    kz2 = k0 * zerire + searchinterval / 2.0 - 1j * kz_im_half
    Message("Found %d possible zeros." % len(zerire), 1).show(verbosity)
    Message("Possible zeros:\n%s" % zerire, 3).show(verbosity)
    Message("Refining zeros.", 1).show(verbosity)
    mk = 0
    modi = []
    for m in range(len(kz1)):
        if mk == nmodi:
            break

        abscomp, uscelto, icomp = method_of_component(
            numpy.mean([kz1[m], kz2[m]]), slices, Rot
        )
        z0, valf = fzzeroabs2(
            lambda kz: method_of_component(kz, slices, Rot, uscelto, icomp)[0],
            kz1[m],
            kz2[m],
        )

        if len(valf) > 0:
            eigval, eigvec = numpy.linalg.eig(Mvec(z0, slices, Rot))
            mk += 1
            index0 = numpy.where(numpy.abs(eigval) < TolnullEig)[0]
            if len(index0) > 1:
                Message(
                    "Two eigenvectors found with the same eigenvalue. Mixing of modes may result",
                    1,
                ).show(verbosity)

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

            Message(
                "Distance from zero of the final solution: %f"
                % modo.distance_from_zero,
                2,
            ).show(verbosity)
            Message("%d. neff = %s" % (mk, modo.keff / k0), 1).show(verbosity)

    return modi
