"""FDTD Postprocessing."""

import os

from matplotlib import pyplot as plt
import numpy

import EMpy.utils

__author__ = "Lorenzo Bolla"


class Input:
    """Data structure to handle input files."""

    def __init__(self, filename):
        """Set the input filename."""
        self.filename = filename

    def __str__(self):
        """Return a representation of the input file."""

        dftmon_str = "%g ! #timemonitors \n" % len(self.dftmonitors)
        if len(self.dftmonitors) > 0:
            dftmon_str += "".join(
                [
                    "%g %g %g %g %g %g\n%g %g\n"
                    % (
                        dm[0][0],
                        dm[0][1],
                        dm[0][2],
                        dm[0][3],
                        dm[0][4],
                        dm[0][5],
                        dm[1][0],
                        dm[1][1],
                    )
                    for dm in self.dftmonitors
                ]
            )

        timemon_str = "%g ! #timemonitors \n" % len(self.dftmonitors)
        if len(timemon_str) > 0:
            timemon_str += "%g %g \n %s" % (
                self.timemonitors_time_interval[0],
                self.timemonitors_time_interval[1],
                "".join(
                    [
                        "%g %g %g ! time_monitor #%d\n" % (s[0], s[1], s[2], iss)
                        for iss, s in enumerate(self.timemonitors)
                    ]
                ),
            )

        return (
            "%g %g %g %g ! dx dy dz cfl \n"
            "%g %g %g %g %g %g %s %g %g ! xmax ymax zmax pmlx pmly pmlz pmltype pmlsmooth pmlref \n"
            "%g %g %g %g ! xmax ymax zmax pmlx pmly pmlz \n"
            "%g ! output3deps? \n"
            "%g ! number diel slices \n"
            "%s \n"
            "%g ! number field slices \n"
            "%s \n"
            "%g %g %g ! #dielobjs, index of bg, conductivity of bg \n"
            "%s"
            "%g ! smoothing method \n"
            "%g ! #sources \n"
            "%s"
            "%g %g %g ! lambdamin, lambdamax, dlambda \n"
            "%s"
            "%s"
            % (
                self.dx,
                self.dy,
                self.dz,
                self.cfl,
                self.xmax,
                self.ymax,
                self.zmax,
                self.pmlx,
                self.pmly,
                self.pmlz,
                self.pmltype,
                self.pmlsmooth,
                self.pmlref,
                self.start,
                self.end,
                self.slides,
                self.snapshot,
                self.output3deps,
                len(self.dielslices),
                "\n".join(
                    [
                        "%g %g %g ! dielslice #%d" % (d[0], d[1], d[2], dd)
                        for (dd, d) in enumerate(self.dielslices)
                    ]
                ),
                len(self.fieldslices),
                "\n".join(
                    [
                        "%g %g %g ! fieldslice #%d" % (f[0], f[1], f[2], ff)
                        for (ff, f) in enumerate(self.fieldslices)
                    ]
                ),
                len(self.dielobjs),
                self.bgrix,
                self.bgsigma,
                "".join(["%s %s\n" % obj for obj in self.dielobjs]),
                self.smoothing_method,
                len(self.sources),
                "".join(["%s\n%s\n%s\n%s\n" % src for src in self.sources]),
                self.lambdamin,
                self.lambdamax,
                self.dlambda,
                dftmon_str,
                timemon_str,
            )
        )

    def tofile(self, filename=None):
        """Save the input data to the input file."""
        if filename is None:
            filename = self.filename
        f = open(filename, "w")
        f.write(self.__str__())
        f.close()


class Param:
    """Data structure to handle the param file."""

    def __str__(self):
        """Return a representation of the input file."""
        return (
            "%g\n"
            "%g\n"
            "%g\n"
            "%g\n"
            "%g\n"
            "%g\n"
            "%g\n"
            "%g\n"
            "%g\n"
            "%g\n"
            "%g\n"
            "%g\n"
            "%g\n"
            "%g\n"
            "%s"
            % (
                self.dx,
                self.dy,
                self.dz,
                self.dt,
                self.mx,
                self.my,
                self.mz,
                self.pmlx,
                self.pmly,
                self.pmlz,
                self.nflux,
                self.ntime,
                self.step1,
                self.step2,
                "\n".join(
                    [
                        "%d\n%d\n%d\n%d\n%d\n%d"
                        % (
                            dm["direction"],
                            dm["nfreq"],
                            dm["flxlim"][0],
                            dm["flxlim"][1],
                            dm["flxlim"][2],
                            dm["flxlim"][3],
                        )
                        for dm in self.dftmonitors
                    ]
                ),
            )
        )


class Sensor:
    """Data structure to handle the FFT sensor's data."""

    def plot(self, n):
        """Plot the sensor's fields."""
        plt.clf()
        plt.hot()
        plt.subplot(2, 2, 1)
        plt.contour(numpy.abs(self.E1[:, :, n].T), 16)
        plt.axis("image")
        plt.title("E1")
        plt.subplot(2, 2, 2)
        plt.contour(numpy.abs(self.H1[:, :, n].T), 16)
        plt.axis("image")
        plt.title("H1")
        plt.subplot(2, 2, 3)
        plt.contour(numpy.abs(self.E2[:, :, n].T), 16)
        plt.axis("image")
        plt.title("E2")
        plt.subplot(2, 2, 4)
        plt.contour(numpy.abs(self.H2[:, :, n].T), 16)
        plt.axis("image")
        plt.title("H2")
        plt.show()

    def __str__(self):
        """Return a representation of the sensor."""
        return "E1\n%s\nH1\n%s\nE2\n%s\nH2\n%s\n" % (self.E1, self.H1, self.E2, self.H2)


class TimeSensor:
    """Data structure to handle the time sensor's data."""

    def plot_Ex(self, logplot=False):
        self.__plot_field(self.Ex, logplot)

    def plot_Ey(self, logplot=False):
        self.__plot_field(self.Ey, logplot)

    def plot_Ez(self, logplot=False):
        self.__plot_field(self.Ez, logplot)

    def plot_Hx(self, logplot=False):
        self.__plot_field(self.Hx, logplot)

    def plot_Hy(self, logplot=False):
        self.__plot_field(self.Hy, logplot)

    def plot_Hz(self, logplot=False):
        self.__plot_field(self.Hz, logplot)

    def __plot_field(self, field, logplot=False):
        if logplot:
            data = 20 * numpy.log10(1e-20 + numpy.abs(field))
            plt.plot(self.t, data)
        else:
            data = field
            plt.plot(self.t, data)
        plt.show()


class FDTD:
    """FDTD.
    Data structure to handle an FDTD simulation. It manages an input file, a param file and the sensors' output.
    It can run a simulation via a system call.
    """

    def __init__(self):
        self.input = None
        self.param = None
        self.sensors = None

    def fetch_data(
        self,
        remote_dir_="./",
        input_file="inp.txt",
        param_file="param",
        directory_="./",
    ):
        remote_dir = fixdir(remote_dir_)
        directory = fixdir(directory_)
        # input file
        os.system(
            "scp -C bollalo001@pico:" + remote_dir + "/" + input_file + " " + directory
        )
        # param file
        os.system(
            "scp -C bollalo001@pico:" + remote_dir + "/" + param_file + " " + directory
        )
        # fieldslices, flux and time sensors
        os.system(
            "scp -C bollalo001@pico:" + remote_dir + "/[EHeh]*_*" + " " + directory
        )
        # dielslices
        os.system("scp -C bollalo001@pico:" + remote_dir + "/diel*" + " " + directory)

    def put_data(self, remote_dir_="./", input_file="inp.txt", directory_="./"):
        remote_dir = fixdir(remote_dir_)
        directory = fixdir(directory_)
        # input file
        os.system("scp -C" + directory + input_file + " bollalo001@pico:" + remote_dir)
        # .dat modesolver's files
        os.system("scp -C" + directory + "*.dat bollalo001@pico:" + remote_dir)

    def load(
        self, directory_="./", input_file="inp.txt", param_file="param", remote_dir_=""
    ):
        """Load input, param and sensors."""
        remote_dir = fixdir(remote_dir_)
        directory = fixdir(directory_)
        if remote_dir != "":
            self.fetch_data(remote_dir, input_file, param_file, directory)
        self.load_input_file(directory, input_file)
        self.load_param(directory, param_file)
        self.load_sensors(directory)
        self.load_time_sensors(directory)

    def load_input_file(self, directory_="./", filename="inp.txt"):
        """Load input file."""
        directory = fixdir(directory_)
        try:
            f = open(directory + filename)
        except Exception:
            print("ERROR: input file")
            return
        inp = Input(filename)

        inp.dx, inp.dy, inp.dz, inp.cfl = numpy.fromstring(
            strip_comment(f.readline()), sep=" "
        )
        tmp = strip_comment(f.readline())
        tmp_idx = tmp.find("P")
        if tmp_idx > 0:
            inp.pmltype = "P"
        else:
            tmp_idx = tmp.find("G")
            if tmp_idx > 0:
                inp.pmltype = "G"
            else:
                raise ValueError("wrong pmltype")
        inp.xmax, inp.ymax, inp.zmax, inp.pmlx, inp.pmly, inp.pmlz = numpy.fromstring(
            tmp[:tmp_idx], sep=" "
        )
        inp.pmlsmooth, inp.pmlref = numpy.fromstring(tmp[tmp_idx + 1 :], sep=" ")
        inp.start, inp.end, inp.slides, inp.snapshot = numpy.fromstring(
            strip_comment(f.readline()), sep=" "
        )
        inp.output3deps = numpy.fromstring(strip_comment(f.readline()), sep=" ")

        # dielslices
        ndielslices = numpy.fromstring(strip_comment(f.readline()), sep=" ")
        inp.dielslices = []
        for i in range(ndielslices):
            inp.dielslices.append(
                numpy.fromstring(strip_comment(f.readline()), sep=" ")
            )

        # fieldslices
        nfieldslices = numpy.fromstring(strip_comment(f.readline()), sep=" ")
        inp.fieldslices = []
        for i in range(nfieldslices):
            inp.fieldslices.append(
                numpy.fromstring(strip_comment(f.readline()), sep=" ")
            )

        # dielobjs
        ndielobjs, inp.bgrix, inp.bgsigma = numpy.fromstring(
            strip_comment(f.readline()), sep=" "
        )
        inp.dielobjs = []
        for i in range(int(ndielobjs)):
            inp.dielobjs.append(
                (strip_comment(f.readline()), strip_comment(f.readline()))
            )
        inp.smoothing_method = numpy.fromstring(strip_comment(f.readline()), sep=" ")

        # sources
        nsources = numpy.fromstring(strip_comment(f.readline()), dtype=int, sep=" ")
        inp.sources = []
        #        (inp.time_dependence, inp.wls, inp.pwidth, inp.shift) = numpy.fromstring(strip_comment(f.readline()), sep = ' ')
        for i in range(nsources):
            inp.sources.append(
                (
                    strip_comment(f.readline()),
                    strip_comment(f.readline()),
                    strip_comment(f.readline()),
                    strip_comment(f.readline()),
                )
            )

        # dft monitors
        inp.lambdamin, inp.lambdamax, inp.dlambda = numpy.fromstring(
            strip_comment(f.readline()), sep=" "
        )
        ndftmonitors = numpy.fromstring(strip_comment(f.readline()), dtype=int, sep=" ")
        inp.dftmonitors = []
        for i in range(ndftmonitors):
            inp.dftmonitors.append(
                (
                    numpy.fromstring(strip_comment(f.readline()), sep=" "),
                    numpy.fromstring(strip_comment(f.readline()), sep=" "),
                )
            )

        # time monitors
        ntimemonitors = numpy.fromstring(strip_comment(f.readline()), sep=" ")
        inp.timemonitors_time_interval = numpy.fromstring(
            strip_comment(f.readline()), sep=" "
        )
        inp.timemonitors = []
        for i in range(ntimemonitors):
            inp.timemonitors.append(
                numpy.fromstring(strip_comment(f.readline()), sep=" ")
            )

        f.close()
        self.input = inp

    def load_param(self, directory_="./", filename="param"):
        """Load param file."""
        directory = fixdir(directory_)
        param = Param()
        try:
            data = numpy.fromfile(directory + filename, sep=" ")
        except Exception:
            print("ERROR: param file")
            return
        param.dx, param.dy, param.dz, param.dt = data[0:4]
        (
            param.mx,
            param.my,
            param.mz,
            param.pmlx,
            param.pmly,
            param.pmlz,
            param.nflux,
            param.ntime,
            param.step1,
            param.step2,
        ) = data[4:14].astype(numpy.int32)
        param.dftmonitors = []
        for iflux in range(int(param.nflux)):
            direction, nfreq = data[14 + iflux * 6 : 16 + iflux * 6]
            flxlim = data[16 + iflux * 6 : 20 + iflux * 6]
            param.dftmonitors.append(
                {"direction": int(direction), "nfreq": int(nfreq), "flxlim": flxlim}
            )
        self.param = param

    def load_time_sensors(self, directory_="./"):
        """Load time sensors."""
        directory = fixdir(directory_)
        time_sensors = []
        if self.param is None:
            self.load_param(directory)
        for itime in range(self.param.ntime):
            tmp = TimeSensor()
            tmp.Ex = load_fortran_unformatted(directory + "Ex_time_%02d" % (itime + 1))
            tmp.Ey = load_fortran_unformatted(directory + "Ey_time_%02d" % (itime + 1))
            tmp.Ez = load_fortran_unformatted(directory + "Ez_time_%02d" % (itime + 1))
            tmp.Hx = load_fortran_unformatted(directory + "Hx_time_%02d" % (itime + 1))
            tmp.Hy = load_fortran_unformatted(directory + "Hy_time_%02d" % (itime + 1))
            tmp.Hz = load_fortran_unformatted(directory + "Hz_time_%02d" % (itime + 1))
            tmp.t = self.param.dt * numpy.arange(len(tmp.Ex))
            time_sensors.append(tmp)

        self.time_sensors = time_sensors

    def load_sensors(self, directory_="./"):
        """Load sensors."""
        directory = fixdir(directory_)
        sensors = []
        if self.param is None:
            self.load_param(directory)
        for iflux in range(self.param.nflux):
            tmp = Sensor()
            dm = self.param.dftmonitors[iflux]
            tmp.E1 = load_fortran_unformatted(directory + "E1_%02d" % (iflux + 1))
            tmp.H1 = load_fortran_unformatted(directory + "H1_%02d" % (iflux + 1))
            tmp.E2 = load_fortran_unformatted(directory + "E2_%02d" % (iflux + 1))
            tmp.H2 = load_fortran_unformatted(directory + "H2_%02d" % (iflux + 1))
            # [tmp.E1, tmp.H1, tmp.E2, tmp.H2] = map(lambda x: x[0::2] + 1j * x[1::2], [tmp.E1, tmp.H1, tmp.E2, tmp.H2])
            # more memory efficient!
            tmp.E1 = tmp.E1[0::2] + 1j * tmp.E1[1::2]
            tmp.H1 = tmp.H1[0::2] + 1j * tmp.H1[1::2]
            tmp.E2 = tmp.E2[0::2] + 1j * tmp.E2[1::2]
            tmp.H2 = tmp.H2[0::2] + 1j * tmp.H2[1::2]

            n1 = dm["flxlim"][1] - dm["flxlim"][0] + 1
            n2 = dm["flxlim"][3] - dm["flxlim"][2] + 1
            tmp.E1 = tmp.E1.reshape((n1, n2 + 1, dm["nfreq"]), order="F")
            tmp.H1 = tmp.H1.reshape((n1, n2 + 1, dm["nfreq"]), order="F")
            tmp.E2 = tmp.E2.reshape((n1 + 1, n2, dm["nfreq"]), order="F")
            tmp.H2 = tmp.H2.reshape((n1 + 1, n2, dm["nfreq"]), order="F")
            if dm["direction"] == 1:
                # sensors in the x-direction
                tmp.dx1 = self.param.dy
                tmp.dx2 = self.param.dz
            elif dm["direction"] == 2:
                # sensors in the y-direction
                tmp.dx1 = self.param.dx
                tmp.dx2 = self.param.dz
            elif dm["direction"] == 3:
                # sensors in the z-direction
                tmp.dx1 = self.param.dx
                tmp.dx2 = self.param.dy
            else:
                raise ValueError("wrong direction")

            sensors.append(tmp)

        self.sensors = sensors

    def viz2D(self, filename, directory_="./", const_dir="z", logplot=False):
        """Visualize a slice."""
        directory = fixdir(directory_)
        data = load_fortran_unformatted(directory + filename)
        if self.param is None:
            self.load_param(directory)
        x = numpy.linspace(
            self.param.dx / 2.0,
            self.param.dx * self.param.mx - self.param.dx / 2.0,
            self.param.mx,
        )
        y = numpy.linspace(
            self.param.dy / 2.0,
            self.param.dy * self.param.my - self.param.dy / 2.0,
            self.param.my,
        )
        z = numpy.linspace(
            self.param.dz / 2.0,
            self.param.dz * self.param.mz - self.param.dz / 2.0,
            self.param.mz,
        )
        if const_dir == "x":
            n1 = self.param.my
            n2 = self.param.mz
            x1 = y
            x2 = z
            x1label = "y"
            x2label = "z"
        elif const_dir == "y":
            n1 = self.param.mx
            n2 = self.param.mz
            x1 = x
            x2 = z
            x1label = "x"
            x2label = "z"
        else:
            n1 = self.param.mx
            n2 = self.param.my
            x1 = x
            x2 = y
            x1label = "x"
            x2label = "y"
        data = data.reshape((n2, n1))
        plt.clf()
        if logplot:
            data = 20 * numpy.log10(numpy.abs(data).clip(1e-30, 1e30))
            plt.jet()
        else:
            plt.hot()
        plt.contour(x1, x2, data, 64)
        plt.colorbar()
        plt.axis("image")
        plt.xlabel(x1label + " /um")
        plt.ylabel(x2label + " /um")
        plt.show()

    def memory(self):
        """Estimate the memory occupation."""
        # size_of_char = 1
        # size_of_int = 4
        size_of_real = 4
        # size_of_complex = 2 * size_of_real
        # size_of_dielobj = size_of_int + 31 * size_of_real + 2 * 16 * size_of_char
        # size_of_source = 9 * size_of_int + 5 * size_of_real + 6 * 16 * size_of_char
        # size_of_monitor = (6 + 2) * 6 * size_of_int

        Gb = 1024**3
        max_available_RAM = 32 * Gb

        dynamic_alloc_memory = 0

        # epsx, epsy, epsz
        dynamic_alloc_memory = (
            dynamic_alloc_memory
            + 3
            * (self.param.mx + 2 * self.input.pmlx)
            * (self.param.my + 2 * self.input.pmly)
            * (self.param.mz + 2 * self.input.pmlz)
            * size_of_real
        )
        # sigma
        dynamic_alloc_memory = (
            dynamic_alloc_memory
            + 2
            * (self.param.mx + 2 * self.input.pmlx)
            * (self.param.my + 2 * self.input.pmly)
            * (self.param.mz + 2 * self.input.pmlz)
            * size_of_real
        )
        # cex, cmx
        dynamic_alloc_memory = (
            dynamic_alloc_memory
            + 2 * 2 * (self.param.mx + 2 * self.input.pmlx) * size_of_real
        )
        # cey, cmy
        dynamic_alloc_memory = (
            dynamic_alloc_memory
            + 2 * 2 * (self.param.my + 2 * self.input.pmly) * size_of_real
        )
        # cez, cmz
        dynamic_alloc_memory = (
            dynamic_alloc_memory
            + 2 * 2 * (self.param.mz + 2 * self.input.pmlz) * size_of_real
        )

        # exy, exz, eyx, eyz, ...
        dynamic_alloc_memory = (
            dynamic_alloc_memory
            + 12
            * (self.param.mx + 2 * self.input.pmlx)
            * (self.param.my + 2 * self.input.pmly)
            * (self.param.mz + 2 * self.input.pmlz)
            * size_of_real
        )

        print(
            "Alloc mem = %g Gb, [%d%%]"
            % (
                1.0 * dynamic_alloc_memory / Gb,
                int(1.0 * dynamic_alloc_memory / max_available_RAM * 100),
            )
        )

    def run(
        self,
        directory_="./",
        exe_file="/xlv1/labsoi_devices/devices/f3d",
        output_file="output",
        ncpu=12,
        bg=False,
        remote=True,
    ):
        """Run the simulation, possibly in remote."""
        directory = fixdir(directory_)
        #        os.environ['OMP_NUM_THREAD'] = str(ncpu)
        #        cmd = 'dplace -x6 ' + exe_file + ' > ' + output_file
        cmd = (
            "cd"
            + directory
            + "; setenv OMP_NUM_THREAD"
            + str(ncpu)
            + "dplace -x6 "
            + exe_file
            + " > "
            + output_file
        )
        if bg:
            cmd += "&"
        if remote:
            cmd = 'ssh pico "' + cmd + '"'
        os.system(cmd)

    def __str__(self):
        """Return a representation of the FDTD data structure."""
        return "INPUT\n%s\nPARAM\n%s\nSENSORS\n%s\n" % (
            self.input,
            self.param,
            self.sensors,
        )


def load_fortran_unformatted(filename):
    """Load data from an unformatted fortran binary file."""
    try:
        f = open(filename, "rb")
    except Exception:
        print("ERROR")
        return
    nbytes = numpy.fromfile(f, dtype=numpy.int32, count=1)
    n = nbytes / numpy.float32().nbytes
    data = numpy.fromfile(f, dtype=numpy.float32, count=n)
    f.close()
    return data


def strip_comment(line):
    """Get rid of fortran comments."""
    idx = line.find("!")
    if idx != -1:
        return line[:idx].strip()
    return line


def fixdir(str, sep="/"):
    tmp = str
    if len(tmp) > 0:
        if tmp[-1] != sep:
            tmp += sep
    return tmp


# def overlap_f(simul, solver, nwl):
#     vu = numpy.zeros((len(simul.sensors), len(solver.modes)), dtype=complex)
#     ju = numpy.zeros((len(simul.sensors), len(solver.modes)), dtype=complex)
#     for isens, sens in enumerate(simul.sensors):
#         for imode, mode in enumerate(solver.modes):
#             Ex, Ey, Ez, Hx, Hy, Hz = mode.get_fields_for_FDTD()
#             vu[isens, imode] = 0.5 * (
#                         numpy.trapz(numpy.trapz(sens.E1[:,1:-1,nwl] * Hy, dx=sens.dx2*1e-6), dx=sens.dx1*1e-6) -
#                         numpy.trapz(numpy.trapz(sens.E2[1:-1,:,nwl] * Hx, dx=sens.dx2*1e-6), dx=sens.dx1*1e-6))
#             ju[isens, imode] = 0.5 * (
#                         numpy.trapz(numpy.trapz(sens.H2[1:-1,:,nwl] * Ey, dx=sens.dx2*1e-6), dx=sens.dx1*1e-6) -
#                         numpy.trapz(numpy.trapz(sens.H1[:,1:-1,nwl] * Ex, dx=sens.dx2*1e-6), dx=sens.dx1*1e-6))
#     A = (vu + ju) / 2.
#     B = (vu - ju) / 2.
#     Pm = numpy.abs(A)**2 - numpy.abs(B)**2
#     P = Pm.sum(axis=1)
#     return (vu, ju, A, B, Pm, P)


def overlap_f(sensors, solver, nwl):
    vu = numpy.zeros((len(sensors), len(solver.modes)), dtype=complex)
    ju = numpy.zeros((len(sensors), len(solver.modes)), dtype=complex)
    for isens, sens in enumerate(sensors):
        x = sens.dx1 * numpy.arange(sens.E2.shape[0])
        y = sens.dx2 * numpy.arange(sens.E1.shape[1])
        for imode, mode in enumerate(solver.modes):
            # resample the mode to the sensor's grid
            Ex, Ey, Ez, Hx, Hy, Hz = mode.get_fields_for_FDTD(x, y)
            #            vu[isens, imode] = 0.5 * (
            #                        numpy.trapz(numpy.trapz(sens.E1[:,1:-1,nwl] * Hy, dx=sens.dx2*1e-6), dx=sens.dx1*1e-6) -
            #                        numpy.trapz(numpy.trapz(sens.E2[1:-1,:,nwl] * Hx, dx=sens.dx2*1e-6), dx=sens.dx1*1e-6))
            #            ju[isens, imode] = 0.5 * (
            #                        numpy.trapz(numpy.trapz(sens.H2[1:-1,:,nwl] * Ey, dx=sens.dx2*1e-6), dx=sens.dx1*1e-6) -
            #                        numpy.trapz(numpy.trapz(sens.H1[:,1:-1,nwl] * Ex, dx=sens.dx2*1e-6), dx=sens.dx1*1e-6))
            vu[isens, imode] = 0.5 * (
                EMpy.utils.trapz2(
                    sens.E1[:, 1:-1, nwl] * Hy, dx=sens.dx1 * 1e-6, dy=sens.dx2 * 1e-6
                )
                - EMpy.utils.trapz2(
                    sens.E2[1:-1, :, nwl] * Hx, dx=sens.dx1 * 1e-6, dy=sens.dx2 * 1e-6
                )
            )
            ju[isens, imode] = 0.5 * (
                EMpy.utils.trapz2(
                    sens.H2[1:-1, :, nwl] * Ey, dx=sens.dx1 * 1e-6, dy=sens.dx1 * 1e-6
                )
                - EMpy.utils.trapz2(
                    sens.H1[:, 1:-1, nwl] * Ex, dx=sens.dx1 * 1e-6, dy=sens.dx1 * 1e-6
                )
            )
    A = (vu + ju) / 2.0
    B = (vu - ju) / 2.0
    Pm = numpy.abs(A) ** 2 - numpy.abs(B) ** 2
    P = Pm.sum(axis=1)
    return (vu, ju, A, B, Pm, P)
