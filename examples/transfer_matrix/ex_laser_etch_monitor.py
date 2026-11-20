"""In-situ laser etch monitor (endpoint detection) on an absorbing GaAs/AlGaAs VCSEL stack.

Simulates real-time 670 nm laser reflectometry as a Si3N4-capped 5-pair
GaAs / Al0.95Ga0.05As Distributed Bragg Reflector (DBR) cavity is progressively
thinned during reactive-ion etching (RIE), using tabulated complex refractive
indices ``n(wl) - i * k(wl)`` via ``RefractiveIndex(n0_func=...)``.

Adapted from the contributed EMpy example by Demis D. John (March 2017).

References
----------
- K. Bacher, B. Pezeshki, S. M. Lord, and J. S. Harris Jr., "Molecular beam
  epitaxy growth of vertical cavity optical devices with in situ corrections,"
  *Appl. Phys. Lett.* **61**(12), 1387–1389 (1992).
- H. A. Macleod, *Thin-Film Optical Filters*, 4th ed. (CRC Press, 2010).
"""

from copy import copy, deepcopy
import importlib.util
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

import EMpy


def _load_nk_module():
    nk_path = Path(__file__).resolve().parents[1] / "data" / "nk_gaas_algaas.py"
    spec = importlib.util.spec_from_file_location("nk_gaas_algaas", nk_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _count_noninf(multilayer):
    return sum(0 if np.isinf(layer.thickness) else 1 for layer in multilayer)


def main():
    nk = _load_nk_module()

    wl_lasermon = 670e-9
    etch_step = 10e-9
    wls = np.array([wl_lasermon - 1e-9, wl_lasermon, wl_lasermon + 1e-9])
    wl_idx = 1
    theta_inc = 0.0

    mat_air = EMpy.materials.IsotropicMaterial(
        "air", EMpy.materials.RefractiveIndex(n0_const=1.0)
    )
    mat_sin = EMpy.materials.IsotropicMaterial(
        "Si3N4", EMpy.materials.RefractiveIndex(n0_const=1.997)
    )

    def n_gaas(w):
        return nk.GaAs_interp(w * 1e6, k=True)

    def n_algaas95(w):
        return nk.AlGaAs_interp(0.95, w * 1e6, k=True)

    mat_gaas = EMpy.materials.IsotropicMaterial(
        "GaAs", EMpy.materials.RefractiveIndex(n0_func=n_gaas)
    )
    mat_algaas95 = EMpy.materials.IsotropicMaterial(
        "Al95Ga05As", EMpy.materials.RefractiveIndex(n0_func=n_algaas95)
    )

    wl_center = 1100e-9
    d_gaas_dbr = wl_center / float(np.real(n_gaas(wl_center)[0])) / 4.0
    d_algaas_dbr = wl_center / float(np.real(n_algaas95(wl_center)[0])) / 4.0

    Layer = EMpy.utils.Layer
    air = Layer(mat_air, np.inf)
    sin_cap = Layer(mat_sin, 300e-9)
    gaas_dbr = Layer(mat_gaas, d_gaas_dbr)
    algaas_dbr = Layer(mat_algaas95, d_algaas_dbr)
    gaas_core = Layer(mat_gaas, 551.75e-9)
    algaas_spacer = Layer(mat_algaas95, 500e-9)
    gaas_spacer = Layer(mat_gaas, 300e-9)
    gaas_sub = Layer(mat_gaas, np.inf)

    layers_init = (
        [air, sin_cap]
        + 5 * [gaas_dbr, algaas_dbr]
        + [gaas_core]
        + 5 * [algaas_dbr, gaas_dbr]
        + [algaas_spacer, gaas_spacer, gaas_sub]
    )
    layers = EMpy.utils.Multilayer([copy(layer) for layer in layers_init])

    etch_depths = []
    r_laser = []
    refr_idx = []
    step_rem = 0.0
    idx_top = 1  # First finite layer below semi-infinite air

    while _count_noninf(layers) > 0:
        if layers[idx_top].thickness <= step_rem:
            step_rem -= layers[idx_top].thickness
            layers.pop(idx_top)
            continue

        layers[idx_top].thickness -= step_rem
        snap = deepcopy(layers)
        refr_idx.append(float(np.real(snap[idx_top].mat.n(wl_lasermon).item())))
        etch_depths.append(0.0 if not etch_depths else etch_depths[-1] + etch_step)
        step_rem = etch_step

        sol = EMpy.transfer_matrix.IsotropicTransferMatrix(snap, theta_inc).solve(wls)
        r_laser.append(float(sol.Rs[wl_idx]))

    print(
        f"Simulated {len(etch_depths)} RIE steps up to {etch_depths[-1] * 1e9:.0f} nm "
        f"at lambda = {wl_lasermon * 1e9:.1f} nm "
        f"(R range: {min(r_laser) * 100:.1f}% .. {max(r_laser) * 100:.1f}%)"
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 5.5))
    depths_nm = np.array(etch_depths) * 1e9

    ax1.plot(depths_nm, refr_idx, color="seagreen")
    ax1.set_title(f"In-Situ Laser Etch Monitor at λ = {wl_lasermon * 1e9:.1f} nm")
    ax1.set_ylabel("Top-Layer Re(n)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(depths_nm, np.array(r_laser) * 100.0, color="royalblue")
    ax2.set_ylabel("Reflectivity (%)")
    ax2.set_xlabel("Etch Depth (nm)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
