"""Transfer matrix example.

Solve for both an isotropic and anisotropic multilayer.
"""

from matplotlib import pyplot as plt
import numpy as np

import EMpy

n = np.array([1.0, 2.0, 2.3, 4.3, 3.0])
d = np.array([np.inf, 1e-6, 2.3e-6, 0.1e-6, np.inf])

iso_layers = EMpy.utils.Multilayer()
aniso_layers = EMpy.utils.Multilayer()

for i in range(n.size):
    iso_layers.append(
        EMpy.utils.Layer(
            EMpy.materials.IsotropicMaterial(
                "mat", n0=EMpy.materials.RefractiveIndex(n[i])
            ),
            d[i],
        )
    )
    aniso_layers.append(
        EMpy.utils.Layer(
            EMpy.materials.AnisotropicMaterial(
                "Air",
                epsilon_tensor=EMpy.materials.EpsilonTensor(
                    epsilon_tensor_const=n[i] ** 2 * EMpy.constants.eps0 * np.eye(3)
                ),
            ),
            d[i],
        )
    )

theta_inc = EMpy.utils.deg2rad(10.0)
theta_inc_x = theta_inc
theta_inc_y = 0.0
wls = np.linspace(1.4e-6, 1.7e-6, 100)
solution_iso = EMpy.transfer_matrix.IsotropicTransferMatrix(
    iso_layers, theta_inc
).solve(wls)
solution_aniso = EMpy.transfer_matrix.AnisotropicTransferMatrix(
    aniso_layers, theta_inc_x, theta_inc_y
).solve(wls)

plt.figure()
plt.plot(
    wls,
    solution_iso.Rs,
    wls,
    solution_iso.Ts,
    wls,
    solution_iso.Rp,
    wls,
    solution_iso.Tp,
)
plt.title("isotropic")

plt.figure()
plt.plot(
    wls,
    solution_aniso.R[0, 0, :],
    wls,
    solution_aniso.R[1, 0, :],
    wls,
    solution_aniso.R[0, 1, :],
    wls,
    solution_aniso.R[1, 1, :],
    wls,
    solution_aniso.T[0, 0, :],
    wls,
    solution_aniso.T[1, 0, :],
    wls,
    solution_aniso.T[0, 1, :],
    wls,
    solution_aniso.T[1, 1, :],
)
plt.title("anisotropic")
plt.show()
