"""Fully vectorial finite-difference mode solver example."""

from matplotlib import pyplot as plt

import EMpy

mat1 = EMpy.materials.IsotropicMaterial(
    "SiN", n0=EMpy.materials.RefractiveIndex(n0_const=1.97)
)
mat2 = EMpy.materials.IsotropicMaterial(
    "Si", n0=EMpy.materials.RefractiveIndex(n0_const=3.4757)
)

l1 = EMpy.utils.Layer(mat1, 4.1e-6)
l21 = EMpy.utils.Layer(mat1, 2e-6)
l22 = EMpy.utils.Layer(mat2, 0.1e-6)

w = 0.5e-6

d = EMpy.utils.EMpy.utils.CrossSection(
    [
        EMpy.utils.Slice(2e-6, [l1]),
        EMpy.utils.Slice(w, [l21, l22, l21]),
        EMpy.utils.Slice(2e-6, [l1]),
    ]
)

nx_points_per_region = (20, 10, 20)
ny_points_per_region = (20, 20, 20)

X, Y = d.grid(nx_points_per_region, ny_points_per_region)
eps = d.epsfunc(X, Y, 1.55e-6)


def epsfunc(x, y):
    return d.epsfunc(x, y, wl)


wl = 1.55e-6

neigs = 2
tol = 1e-8
boundary = "0000"

solver = EMpy.modesolvers.FD.VFDModeSolver(wl, X, Y, epsfunc, boundary).solve(
    neigs, tol
)
fig = plt.figure()
fig.add_subplot(1, 3, 1)
plt.contourf(abs(solver.modes[0].Ex), 50)
plt.title("Ex")
fig.add_subplot(1, 3, 2)
plt.contourf(abs(solver.modes[0].Ey), 50)
plt.title("Ey")
fig.add_subplot(1, 3, 3)
plt.contourf(abs(solver.modes[0].Ez), 50)
plt.title("Ez")
plt.show()
