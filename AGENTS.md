# EMpy Agent Guidelines

## Degenerate-Limit Testing for Anisotropic and Lossy Implementations

Whenever implementing, refactoring, auditing, or testing code that handles **anisotropic** or **lossy** materials/solvers in `EMpy`, always test against the **isotropic** and **non-lossy** cases as degenerate (simplified) limits:

1. **Anisotropic $\to$ Isotropic Limit**
   - Test every anisotropic solver or material representation (`AnisotropicMaterial`, `EpsilonTensor`, `LiquidCrystal`, `AnisotropicTransferMatrix`, `AnisotropicRCWA`, 5-component `(nx, ny, 5)` `VFDModeSolver`) against the corresponding isotropic solver (`IsotropicMaterial`, `IsotropicTransferMatrix`, `IsotropicRCWA`, scalar `(nx, ny)` `VFDModeSolver`).
   - Construct isotropic media explicitly through the anisotropic data structures (e.g., `AnisotropicMaterial` with $\boldsymbol{\varepsilon} = \varepsilon_0 n^2 I_3$, or `(nx, ny, 5)` `[eps, 0, 0, eps, eps]`) so `isIsotropic()` shortcuts are bypassed.
   - Test weak-birefringence limits ($\Delta n = n_E - n_O \to 0$, e.g. $10^{-4}, 10^{-8}, 0$) with arbitrary 3D Euler rotations (`euler_rotate`) to exercise off-diagonal tensor terms, polynomial root-finding/sorting, and degenerate eigenspace/SVD branches.
   - Include coordinate/degeneracy edge cases: normal incidence ($\alpha = 0$) with nonzero azimuthal angle ($\delta \neq 0$), propagation along an optic axis ($\mathbf{k} \parallel \mathbf{c}$), homogeneous grating limits ($n_1 = n_2$ or diffraction order $n = 0$), and lossy anisotropic media reduced to lossy isotropic media.

2. **Lossy $\to$ Non-Lossy Limit ($\kappa \to 0^+$)**
   - `EMpy` uses the harmonic convention $\exp(+i\omega t - i\mathbf{k}\cdot\mathbf{r})$ with passive complex refractive index $n = n_r - i\kappa$ ($\kappa \ge 0$), so forward-propagating/decaying waves satisfy $\operatorname{Re}(k_z) \ge 0$ and $\operatorname{Im}(k_z) \le 0$.
   - Test lossy code paths both at `n = n_r + 0.0j` (complex dtype with zero loss) and in the continuous limit $\kappa \to 0^+$ (e.g. $\kappa = 10^{-4}, 10^{-8} \to 0$) against the strictly real lossless case ($\kappa = 0$), checking for branch-cut discontinuities—especially in evanescent / total-internal-reflection (TIR) regimes.
   - Verify physical invariants across lossy and lossless regimes:
     - **Lossless**: energy conservation / unitarity ($R + T = 1$, $S^H S = I$, $|\text{THRU}|^2 + |\text{DROP}|^2 = 1$).
     - **Passive lossy ($\kappa > 0$)**: strict passivity ($R + T \le 1$, $\|S\|_2 \le 1$, modal attenuation $|e^{-i\beta L}| < 1$) and Lorentz reciprocity ($S = S^T$ when $\boldsymbol{\varepsilon} = \boldsymbol{\varepsilon}^T$).
   - Remember that reciprocal lossy permittivity tensors and symmetric lossy grating Toeplitz matrices are **complex symmetric** ($M = M^T$, $\operatorname{Im} M \neq 0$), not Hermitian ($M \neq M^H$); use general complex linear solvers (`numpy.linalg.inv`, `numpy.linalg.solve`) and conjugate only where required by Poynting flux $\frac{1}{2}\operatorname{Re}(\mathbf{E} \times \mathbf{H}^*)$.
