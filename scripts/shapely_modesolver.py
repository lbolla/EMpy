"""Full-vectorial finite-difference mode solver using Shapely polygon geometries.

Demonstrates ``EMpy.modesolvers.FD.VFDModeSolver`` with arbitrary 2D polygon
waveguide cross-sections (isotropic, birefringent, and magneto-optic/gyrotropic),
2x2 permittivity smoothing to reduce staircase artifacts, and Poynting-vector
Stokes parameter maps co-located on the magnetic-field grid.

Requires the optional dependency ``shapely>=2.0``. Run in an isolated
ephemeral environment with ``uv``::

    uv run --with shapely scripts/shapely_modesolver.py --example fibre
    uv run --with shapely scripts/shapely_modesolver.py --example faraday_clad_si

Or run headless (without opening GUI figure windows)::

    MPLBACKEND=Agg uv run --with shapely scripts/shapely_modesolver.py --example fibre

Adapted and merged from the isotropic and anisotropic Shapely mode-solver
scripts contributed by David C. Hutchings (James Watt School of Engineering,
University of Glasgow).

References
----------
- A. B. Fallahkhair, K. S. Li, and T. E. Murphy, "Vector finite difference
  modesolver for anisotropic dielectric waveguides," *J. Lightwave Technol.*
  **26**(11), 1423-1431 (2008).
- D. C. Hutchings, "Theory of Faraday effect and waveguide mode conversion," in
  *Integrated Optical Isolators* (Springer Series in Optical Sciences).
"""

import argparse

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from scipy import signal
import shapely  # type: ignore[import-not-found]
from shapely import plotting  # type: ignore[import-not-found]

import EMpy

# Each geometry is a tuple of regions ordered from background to foreground:
# - Isotropic:   (label, polygon, eps)
# - Anisotropic: (label, polygon, eps_xx, eps_xy, eps_yx, eps_yy, eps_zz)
# Spatial coordinates must match the wavelength unit (here: nm).
EXAMPLES = {
    "fibre": (
        ("base", shapely.box(-960, -960, 960, 960), 1.46**2),
        ("core", shapely.Point(0, 0).buffer(300), 2.2**2),
    ),
    "si_rib_dry": (
        ("base", shapely.box(-640, -480, 640, 800), 1.0**2),
        ("substrate", shapely.box(-640, -480, 640, 0), 1.46**2),
        ("core", shapely.box(-320, 0, 320, 340), 3.4**2),
    ),
    "si_rib_wet": (
        ("base", shapely.box(-640, -480, 640, 800), 1.0**2),
        ("substrate", shapely.box(-640, -480, 640, 0), 1.46**2),
        (
            "core",
            shapely.Polygon(((-400, 0), (240, 0), (0, 340), (-400, 340), (-400, 0))),
            3.4**2,
        ),
    ),
    "gaas_strip_loaded": (
        ("base", shapely.box(-2500, -2500, 2500, 2500), 1.0**2),
        ("substrate", shapely.box(-2500, -2500, 2500, -550), 3.18**2),
        ("core", shapely.box(-2500, -550, 2500, -50), 3.38**2),
        (
            "cladding",
            shapely.Polygon(
                (
                    (-2500, -50),
                    (-2500, 0),
                    (-400, 0),
                    (-400, 450),
                    (400, 450),
                    (400, 0),
                    (2500, 0),
                    (2500, -50),
                    (-2500, -50),
                )
            ),
            3.18**2,
        ),
    ),
    "faraday_fibre": (
        ("base", shapely.box(-960, -960, 960, 960), 1.46**2),
        (
            "core",
            shapely.Point(0, 0).buffer(300),
            2.2**2,
            -0.01j,
            0.01j,
            2.2**2,
            2.2**2,
        ),
    ),
    "faraday_clad_si": (
        ("base", shapely.box(-640, -480, 640, 800), 1.0**2),
        ("substrate", shapely.box(-640, -480, 640, 0), 1.46**2),
        ("core", shapely.box(-320, 0, 320, 340), 3.4**2),
        (
            "cladding",
            shapely.box(-320, 340, 320, 440),
            2.2**2,
            -0.1j,
            0.1j,
            2.2**2,
            2.2**2,
        ),
    ),
    "birefringent_gaas": (
        ("base", shapely.box(-2500, -2500, 2500, 2500), 1.0**2),
        ("substrate", shapely.box(-2500, -2500, 2500, -550), 3.18**2),
        (
            "core",
            shapely.box(-2500, -550, 2500, -50),
            3.38**2,
            0.01,
            0.01,
            3.38**2,
            3.38**2,
        ),
        (
            "cladding",
            shapely.Polygon(
                (
                    (-2500, -50),
                    (-2500, 0),
                    (-400, 0),
                    (-400, 450),
                    (400, 450),
                    (400, 0),
                    (2500, 0),
                    (2500, -50),
                    (-2500, -50),
                )
            ),
            3.18**2,
        ),
    ),
}


def build_eps_from_polygons(geom, x_, y_, smooth=True):
    """Rasterize a tuple of Shapely regions onto cell centers ``(x_, y_)``.

    Returns a 2D array ``(nx, ny)`` if all regions are isotropic, or a 3D array
    ``(nx, ny, 5)`` with ``[epsxx, epsxy, epsyx, epsyy, epszz]`` otherwise.
    Uses real ``float`` dtype when all permittivities are real so
    ``VFDModeSolver`` stays in real arithmetic.
    """
    is_anisotropic = any(len(region) > 3 for region in geom)
    is_complex = any(np.iscomplexobj(val) for region in geom for val in region[2:])
    dtype = complex if is_complex else float

    if smooth:
        xw = np.concatenate(
            (
                [(3.0 * x_[0] - x_[1]) / 2.0],
                0.5 * (x_[:-1] + x_[1:]),
                [(3.0 * x_[-1] - x_[-2]) / 2.0],
            )
        )
        yw = np.concatenate(
            (
                [(3.0 * y_[0] - y_[1]) / 2.0],
                0.5 * (y_[:-1] + y_[1:]),
                [(3.0 * y_[-1] - y_[-2]) / 2.0],
            )
        )
    else:
        xw = x_
        yw = y_

    xx, yy = np.meshgrid(xw, yw, indexing="ij")
    pts = shapely.points(xx, yy)

    def _to_eps5(region):
        if len(region) == 3:
            eps = region[2]
            return np.array([eps, 0.0, 0.0, eps, eps], dtype=dtype)
        return np.array(region[2:], dtype=dtype)

    if is_anisotropic:
        working = np.empty((xw.size, yw.size, 5), dtype=dtype)
        working[:, :, :] = _to_eps5(geom[0])
        for region in geom[1:]:
            mask = shapely.covers(region[1], pts)
            working[mask, :] = _to_eps5(region)
    else:
        working = np.full((xw.size, yw.size), geom[0][2], dtype=dtype)
        for region in geom[1:]:
            mask = shapely.covers(region[1], pts)
            working[mask] = region[2]

    if smooth:
        working = 0.25 * (
            working[:-1, :-1] + working[1:, :-1] + working[:-1, 1:] + working[1:, 1:]
        )

    return working


def compute_poynting_stokes(mode):
    """Compute Poynting-vector Stokes parameters on the ``H``-field grid."""
    kernel = np.full((2, 2), 0.25)
    ex_at_h = signal.convolve2d(mode.Ex, kernel, mode="full")
    ey_at_h = signal.convolve2d(mode.Ey, kernel, mode="full")

    q1 = ex_at_h * np.conj(mode.Hy)
    q2 = ey_at_h * np.conj(mode.Hx)
    q3 = ey_at_h * np.conj(mode.Hy)
    q4 = ex_at_h * np.conj(mode.Hx)

    s0 = q1.real - q2.real
    s1 = q1.real + q2.real
    s2 = q3.real - q4.real
    s3 = q3.imag + q4.imag
    return s0, s1, s2, s3


def draw_geometry(ax, geom, bbox):
    """Draw waveguide polygon outlines on ``ax``."""
    ax.set_box_aspect(1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.margins(0)
    for idx, region in enumerate(geom[1:], start=1):
        shade = "0.8" if idx == 1 else "0.55"
        plotting.plot_polygon(region[1], ax=ax, add_points=False, color=shade)


def plot_mode_fields(mode, geom, bbox, x, y):
    """Plot 6-component modal field contours."""
    xe = EMpy.utils.centered1d(x)
    ye = EMpy.utils.centered1d(y)
    levels = np.geomspace(1.0 / 32.0, 1.0, num=11)
    label_loc = [(3.0 * bbox[2] + bbox[0]) / 4.0, (5.0 * bbox[3] + bbox[1]) / 6.0]

    e_max = max(np.abs(mode.Ex).max(), np.abs(mode.Ey).max())
    h_max = max(np.abs(mode.Hx).max(), np.abs(mode.Hy).max())
    panels = [
        (xe, ye, mode.Ex, e_max, r"$E_x$"),
        (xe, ye, mode.Ey, e_max, r"$E_y$"),
        (xe, ye, mode.Ez, e_max, r"$E_z$"),
        (x, y, mode.Hx, h_max, r"$H_x$"),
        (x, y, mode.Hy, h_max, r"$H_y$"),
        (x, y, mode.Hz, h_max, r"$H_z$"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2))
    for ax, (gx, gy, field, fmax, label) in zip(axes.ravel(), panels):
        draw_geometry(ax, geom, bbox)
        ax.contour(
            gx,
            gy,
            np.abs(field.T),
            fmax * levels,
            cmap="jet",
            locator=ticker.LogLocator(),
        )
        ax.text(label_loc[0], label_loc[1], label)
    fig.tight_layout()


def plot_stokes_maps(s0, s1, s2, s3, geom, bbox, x, y):
    """Plot irradiance S0 and signed normalized Stokes maps S1/S0, S2/S0, S3/S0."""
    levels_s0 = np.geomspace(1.0 / 1024.0, 1.0, num=11)
    s0_max = np.abs(s0).max()
    mask = np.abs(s0) > 0.01 * s0_max

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.2))
    draw_geometry(axes[0, 0], geom, bbox)
    axes[0, 0].contour(
        x,
        y,
        np.abs(s0.T),
        s0_max * levels_s0,
        cmap="jet",
        locator=ticker.LogLocator(),
    )
    axes[0, 0].set_title(r"$S_0$ (-3 dB contours)")

    for ax, stokes, title in (
        (axes[0, 1], s1, r"$S_1 / S_0$"),
        (axes[1, 0], s2, r"$S_2 / S_0$"),
        (axes[1, 1], s3, r"$S_3 / S_0$"),
    ):
        draw_geometry(ax, geom, bbox)
        norm_val = np.where(mask, stokes / np.where(mask, s0, 1.0), 0.0)
        im = ax.pcolormesh(
            x, y, norm_val.T, cmap="RdBu_r", vmin=-1.0, vmax=1.0, shading="auto"
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)

    fig.tight_layout()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--example",
        choices=sorted(EXAMPLES),
        default="faraday_clad_si",
        help="Waveguide geometry preset to solve",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable 2x2 permittivity smoothing",
    )
    parser.add_argument(
        "--neigs", type=int, default=2, help="Number of eigenmodes to compute"
    )
    parser.add_argument(
        "--grid-pow",
        type=int,
        default=7,
        help="Grid resolution exponent k (nx = ny = 2^k + 1)",
    )
    args = parser.parse_args()

    geom = EXAMPLES[args.example]
    smooth = not args.no_smooth
    wl = 1550.0
    n_pts = 2**args.grid_pow + 1
    bbox = geom[0][1].bounds
    x = np.linspace(bbox[0], bbox[2], n_pts)
    y = np.linspace(bbox[1], bbox[3], n_pts)

    solver = EMpy.modesolvers.FD.VFDModeSolver(
        wl,
        x,
        y,
        lambda xc, yc: build_eps_from_polygons(geom, xc, yc, smooth=smooth),
        "0000",
    ).solve(args.neigs, 1e-6)

    for idx, mode in enumerate(solver.modes):
        s0, s1, s2, s3 = compute_poynting_stokes(mode)
        denom = EMpy.utils.trapz2(s0, x=x, y=y)
        avg_s1 = EMpy.utils.trapz2(s1, x=x, y=y) / denom
        avg_s2 = EMpy.utils.trapz2(s2, x=x, y=y) / denom
        avg_s3 = EMpy.utils.trapz2(s3, x=x, y=y) / denom
        print(
            f"Mode {idx}: neff = {mode.neff:.6f}, "
            f"<S1/S0> = {avg_s1:+.4f}, <S2/S0> = {avg_s2:+.4f}, <S3/S0> = {avg_s3:+.4f}"
        )
        plot_mode_fields(mode, geom, bbox, x, y)
        plot_stokes_maps(s0, s1, s2, s3, geom, bbox, x, y)

    plt.show()


if __name__ == "__main__":
    main()
