# taken from http://hyperphysics.phy-astr.gsu.edu/hbase/phyopt/antiref.html#c1


from matplotlib import pyplot as plt
import numpy

import EMpy

# define multilayer
n = numpy.array([1.0, 1.38, 1.9044])
d = numpy.array([numpy.inf, 387.5e-9 / 1.38, numpy.inf])
iso_layers = EMpy.utils.Multilayer()
for i in range(n.size):
    n0 = EMpy.materials.RefractiveIndex(n[i])
    iso_layers.append(
        EMpy.utils.Layer(EMpy.materials.IsotropicMaterial("mat", n0=n0), d[i])
    )

# define incident wave plane
theta_inc = EMpy.utils.deg2rad(10.0)
wls = numpy.linspace(0.85e-6, 2.25e-6, 300)

# solve
tm = EMpy.transfer_matrix.IsotropicTransferMatrix(iso_layers, theta_inc)
solution_iso = tm.solve(wls)

# plot
plt.figure()
plt.plot(
    wls,
    10 * numpy.log10(solution_iso.Rs),
    "rx-",
    wls,
    10 * numpy.log10(solution_iso.Rp),
    "g.-",
)
plt.legend(("Rs", "Rp"))
plt.title("Single Layer Anti-Reflection Coating")
plt.xlabel("wavelength /m")
plt.ylabel("Power /dB")
plt.grid()
plt.xlim(wls.min(), wls.max())
plt.savefig(__file__ + ".png")
plt.show()
