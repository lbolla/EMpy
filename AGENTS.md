# EMpy Agent Guidelines

## 1. Degenerate-Limit Testing for Anisotropic and Lossy Implementations

Whenever implementing, refactoring, auditing, or testing code that handles **anisotropic** or **lossy** materials/solvers in `EMpy`, always test against the **isotropic** and **non-lossy** cases as degenerate (simplified) limits:

1. **Anisotropic → Isotropic Limit (`Δn → 0`)**
   - Test every anisotropic solver or material representation (`AnisotropicMaterial`, `EpsilonTensor`, `LiquidCrystal`, `LiquidCrystalCell`, `AnisotropicTransferMatrix`, `AnisotropicRCWA`, 5-component `(nx, ny, 5)` `VFDModeSolver`) against the corresponding isotropic solver (`IsotropicMaterial`, `IsotropicTransferMatrix`, `IsotropicRCWA`, scalar `(nx, ny)` `VFDModeSolver`).
   - Construct isotropic media explicitly through the anisotropic data structures (e.g. `AnisotropicMaterial` with `ε = ε₀ n² I₃`, or `(nx, ny, 5)` `[eps, 0, 0, eps, eps]`) so `isIsotropic()` shortcuts are bypassed.
   - Test weak-birefringence sequences (`Δn = n_E - n_O ∈ {1e-4, 1e-8, 0}`) with arbitrary 3D Euler rotations (`euler_rotate`) to exercise off-diagonal tensor terms, eigenproblem root-finding/sorting, and degenerate eigenspace/SVD branches.
   - Include combined coordinate and degeneracy edge cases:
     - Normal incidence (`α = 0`) with nonzero azimuthal angle (`δ ≠ 0`) and `α → 0⁺` continuity.
     - Propagation along uniaxial or biaxial optic axes (`k ∥ c`).
     - Conical diffraction (`δ ≠ 0`) with isotropic and uniaxial end layers, including evanescent orders near `K_{x,m} = ±n_O` (`K_y ≠ 0`) where `(K_{Ez} - K_{Oz}) / (n_O² - K_x²)` has a removable `0/0` singularity.
     - Homogeneous grating limits (`mat1 == mat2` or diffraction order `n = 0`) cross-checked against transfer-matrix solvers.
     - Lossy anisotropic media reduced to lossy isotropic media (including lossy uniaxial superstrates with mixed O+E polarization).

2. **Lossy → Non-Lossy Limit (`κ → 0⁺`)**
   - `EMpy` uses the harmonic convention `exp(+iωt - ik·r)` with passive complex refractive index `n = n_r - iκ` (`κ ≥ 0`), so forward-propagating/decaying waves satisfy `Re(k_z) ≥ 0` and `Im(k_z) ≤ 0`.
   - Select passive square-root branches with the cut along the positive imaginary axis of `k_z²` (`Re(k_z) - Im(k_z) ≥ 0`), which stays continuous across propagating, total-internal-reflection (TIR), lossy-superstrate (`Im(k_x) < 0`), and negative-permittivity / plasmonic (`Re(n) = 0, Im(n) < 0`) regimes.
   - Test lossy code paths both at `n = n_r + 0.0j` (complex dtype with zero loss) and along continuous `κ → 0⁺` sequences (e.g. `κ ∈ {1e-2, 1e-5, 0}`) against strictly real lossless paths. Note that under TIR from a **lossy** incident medium into a **lossless** exit medium, lateral Goos–Hänchen attenuation yields a physically negative local normal Poynting flux `S_z(d⁺) < 0`—do not clamp local interface fluxes to `0`.
   - Verify physical invariants across lossy and lossless regimes:
     - **Lossless (`κ = 0`):** energy conservation and unitarity (`R + T = 1`, `sum(DE1 + DE3) = 1`, `Sᴴ S = I`, `|THRU|² + |DROP|² = 1`).
     - **Passive lossy (`κ > 0`):** strict passivity (`R + T ≤ 1`, `‖S‖₂ ≤ 1`, modal attenuation `|exp(-iβL)| < 1`) and Lorentz reciprocity (`S = Sᵀ` whenever `ε = εᵀ`) plus port-swap symmetry.

3. **Verification Against Known Analytic Solutions**
   - Always try to validate fixes, new features, and regression tests against closed-form or exact analytic benchmarks rather than relying solely on self-consistency, unitarity, or hard-coded numerical snapshots (which can mask topological or branch errors).
   - Examples: Fresnel / Airy / Abeles 2×2 transfer-matrix relations for multilayers and gratings, finite-box transcendental slab dispersion equations (`κ tan(κa) = γ coth(γD)`), lossless Möbius CROW recursions (`r_m = (t_m - r_{m+1} z) / (1 - t_m r_{m+1} z)`), analytical Fréedericksz thresholds, and exact symmetries (`x ↔ y`, mirror, reciprocity).
   - Document non-obvious sign, branch-cut, or algebraic choices with brief inline comments so future audits can trace the physical reasoning.

## 2. Numerical & Electrodynamic Pitfalls to Avoid

1. **Complex-Symmetric vs. Hermitian Algebra & Complex Wavevectors**
   - Reciprocal lossy permittivity tensors and symmetric lossy grating Toeplitz matrices are **complex symmetric** (`M = Mᵀ`, `Im(M) ≠ 0`), **not** Hermitian (`M ≠ Mᴴ`). Use general complex linear solvers (`numpy.linalg.inv`, `numpy.linalg.solve`) and conjugate **only** where required by Poynting flux `0.5 * Re(E × H*)`.
   - Never normalize complex wavevectors or polarization direction vectors with `numpy.linalg.norm(v)` (`sqrt(vᴴ v)`) or `numpy.vdot` when algebraic Maxwell orthogonality (`k · E = 0`, `s_hat · s_hat = 1`) is required: for complex scalar `c` and real unit vector `u`, `(c * u) / norm(c * u) = (c / |c|) * u` injects a spurious phase factor.
   - Never compute diffraction/refraction azimuths via `arctan2(ky.real, kx.real)` when `(kx, ky)` is complex (e.g. conical diffraction in a lossy or negative-`ε` superstrate); use complex direction cosines `(kx, ky) / kt` on the passive `kt` branch.

2. **Exponential Conditioning in Multilayer, Grating & Waveguide Cascades**
   - Never cascade thick layers, high diffraction orders, sub-cutoff evanescent waveguide modes (`Im(n_eff) < 0`), or long lossy waveguides via raw transfer matrices containing growing `exp(+iβL)` or `exp(+k₀ |Im(q)| d)` factors; catastrophic cancellation corrupts propagating modes well before `float64` overflow.
   - Always formulate cascades in bounded passive variables (`|exp(-i k_z d)| ≤ 1`, `|exp(-iβL)| ≤ 1`) using Enhanced Transmittance Matrix (ETM) recursion (`RCWA`) or Redheffer star-product S-matrix cascading (`SWG`, `GenericDevice`).
   - In analytical 1D slice integrals (`FMM.scalarprod`), avoid single-endpoint expansions on evanescent layers (`|Im(k * d)| > 1`) where `O(exp(4 |Im(k)| d))` terms cancel; express integrals symmetrically via both endpoint values `(sl, sr)` and decaying `exp(-|Im(k * d)|)` factors.

3. **Geometric Unit / Scale Invariance (SI Meters vs. Microns)**
   - Most `EMpy` modules (`materials`, `transfer_matrix`, `RCWA`, `devices`, `CrossSection`) use SI meters (`wl ~ 1.55e-6`), whereas some mode-solver examples use microns (`wl ~ 1.55`).
   - Never hardcode dimensional wavenumber steps, bounds, or tolerances that assume `wl ~ 1` (e.g. unscaled `searchinterval` or `maxrekz`). Scale wavenumber quantities by `k0 = 2π / wl`, normalize residuals to be dimensionless, and test solvers in both microns (`1.55`) and SI meters (`1.55e-6`).

4. **Cross-Solver Interfaces, Array Shapes, Determinism & Code Quality**
   - Keep `Mode` interfaces (`FDMode`, `FMMMode2d`) mutually compatible in `overlap()`, `interface_matrix()`, `SimpleJoint`, and `SWG`, and preserve input `wls` array shapes (`1D`, `2D`, scalar via `atleast_1d`) across all `RefractiveIndex` / `Material` dispersion models.
   - Never call global `numpy.random.seed()`; use a local deterministic generator (`numpy.random.default_rng(seed)`).
   - Avoid brittle `1e-14` tolerances on dense/iterative eigensolvers (`scipy.linalg.eig`, `scipy.sparse.linalg.eigs`); use `~1e-12` for cross-platform LAPACK/OpenBLAS reproducibility.
   - Always run `make lint` (`black --check`, `flake8`, `pyflakes`, `mypy`) and `make test` before committing.

## 3. Documentation, Examples & `gh-pages` Maintenance

1. **Keep `README.md`, `examples/`, and `gh-pages` in Sync**
   - Whenever making significant changes to implemented algorithms, public APIs, testing strategies, dependencies, or example scripts/plots, update `README.md`, verify the `examples/**/ex_*.py` scripts (always including literature references in each example's module docstring), and update the `gh-pages` branch (`index.md` and `images/*.png` for https://lbolla.github.io/EMpy/) accordingly.
2. **GitHub & PyPI Markdown Compatibility**
   - Keep Markdown files (`README.md`, `AGENTS.md`) cleanly renderable on both GitHub and PyPI: prefer plain Unicode symbols (`Δn → 0`, `κ → 0⁺`, `Re(k_z) ≥ 0`, `Sᴴ S = I`, `‖S‖₂ ≤ 1`) and inline backticks over `$...$` MathJax blocks, and never use unsupported GitHub MathJax macros (`\operatorname`, `\boldsymbol`) or place `$...$` inside bold/italic/parenthesized spans.
