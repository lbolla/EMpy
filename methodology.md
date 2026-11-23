---
layout: default
title: Methodology & References
description: Electrodynamic conventions, numerical algorithms (4x4 Yeh-Berreman TMM, ETM RCWA, FMM, VFD, Redheffer S-matrix), verification strategy, and literature references for EMpy.
permalink: /methodology/
---

## Methodology, Verification & Literature References

**On this page:** [Electrodynamic Conventions](#conventions) · [Implemented Algorithms](#algorithms) · [Verification Strategy](#verification) · [Literature References](#references)

---

## Electrodynamic & Numerical Conventions {#conventions}

All solvers and helper utilities in `EMpy 3.0` enforce a unified set of electrodynamic conventions:

1. **Harmonic Phase & Passive Loss ($$\exp(+i\omega t - i\mathbf{k}\cdot\mathbf{r})$$):**
   - Plane-wave and modal propagation along $$+z$$ vary as $$\exp(+i\omega t - i k_z z)$$ and $$\exp(+i\omega t - i\beta z)$$.
   - Passive absorbing materials have complex refractive index $$n = n_r - i\kappa$$ ($$\kappa \ge 0$$, $$\operatorname{Im}(n) \le 0$$), so forward-propagating and evanescently decaying waves satisfy $$\operatorname{Re}(k_z) \ge 0$$ and $$\operatorname{Im}(k_z) \le 0$$.
   - `EMpy.utils.rix2losses`, `loss_cm2rix`, `loss_m2rix`, and `loss_dBcm2rix` convert consistently between $$n = n_r - i\kappa$$ and Beer–Lambert power attenuation $$P(z) = P(0)\exp(-\alpha z)$$.
2. **Passive Helmholtz Square-Root Branch Selection:**
   - Across `snell`, `transfer_matrix`, `RCWA`, `FD` (`VFDModeSolver`, `SVFDModeSolver`), and `scattering`, normal wavevector roots $$k_z = \pm\sqrt{k_z^2}$$ use the passive branch cut along the positive imaginary axis ($$\operatorname{Re}(k_z) - \operatorname{Im}(k_z) \ge 0$$), remaining continuous across propagating, total-internal-reflection (TIR), lossy-superstrate ($$\operatorname{Im}(k_x) < 0$$), and plasmonic ($$\operatorname{Re}(n) = 0, \operatorname{Im}(n) < 0$$) regimes.
3. **Complex-Symmetric vs. Hermitian Linear Algebra:**
   - Reciprocal lossy dielectric tensors ($$\boldsymbol{\varepsilon} = \boldsymbol{\varepsilon}^T$$) and symmetric lossy grating Toeplitz matrices are **complex symmetric** ($$M = M^T$$), **not** Hermitian ($$M \ne M^H$$). `EMpy` uses general complex solvers (`numpy.linalg.solve`, `scipy.linalg.eig`) and applies complex conjugation strictly where required by the time-averaged Poynting vector $$\mathbf{S} = \tfrac{1}{2}\operatorname{Re}(\mathbf{E} \times \mathbf{H}^*)$$.
4. **Bounded Passive Cascading:**
   - Thick evanescent layers, high diffraction orders, and sub-cutoff waveguides are cascaded strictly in bounded passive variables ($$\lvert \exp(-i k_z d) \rvert \le 1$$, $$\lvert \exp(-i\beta L) \rvert \le 1$$) via Enhanced Transmittance Matrix (ETM) recursion (`RCWA`) and Redheffer star-product S-matrices (`GenericDevice`), preventing exponential overflow.

---

## Implemented Algorithms {#algorithms}

### 1. Transfer Matrix Method (`EMpy.transfer_matrix`) {#transfer-matrix}

- **`IsotropicTransferMatrix`:** $$2 \times 2$$ characteristic-matrix solver for $$s$$- (TE) and $$p$$- (TM) polarized plane waves in isotropic multilayers, supporting dispersive, absorbing, TIR, and Kretschmann surface-plasmon resonance (SPR) stacks with exact Poynting-flux transmittance normalization.
- **`AnisotropicTransferMatrix`:** $$4 \times 4$$ Yeh–Berreman solver for arbitrarily 3D Euler-rotated uniaxial, biaxial, twisted-nematic, chiral, and lossy anisotropic multilayers:
  - Constructs the first-order Berreman differential operator in each layer and extracts `(2 forward, 2 backward)` eigenwaves.
  - Computes divergence-free Maxwell eigenvectors via SVD nullspaces, orthogonalizes degenerate eigenspaces along uniaxial optic axes and biaxial binormals ($$\Delta n \to 0$$ continuous), and evaluates reflected/transmitted power directly from cross-mode-interference normal Poynting flux $$S_z$$.

### 2. Rigorous Coupled-Wave Analysis (`EMpy.RCWA`) {#rcwa}

- **`IsotropicRCWA`:** Planar and 3D conical diffraction solver for multilayer surface-relief gratings (`BinaryGrating`, `SymmetricDoubleGrating`, `AsymmetricDoubleGrating`), combining Moharam's 1995 eigenmode formulation, Lalanne & Morris's inverse-rule TM Fourier factorization, and Moharam's Enhanced Transmittance Matrix (ETM) recursion.
- **`AnisotropicRCWA`:** Full 3D vector coupled-wave solver for arbitrarily Euler-rotated birefringent/lossy gratings with isotropic or uniaxial semi-infinite end layers:
  - Applies Li's 1998 inverse-rule Fourier factorization to discontinuous permittivity tensors across grating walls.
  - Superposes ordinary and extraordinary eigenwaves in uniaxial superstrates/substrates without complex-norm phase distortion and stabilizes multi-layer cascades via ETM recursion.

### 3. Waveguide Mode Solvers & S-Matrix Circuits (`EMpy.modesolvers`) {#modesolvers}

- **Finite-Difference Mode Solvers (`EMpy.modesolvers.FD`):**
  - `SVFDModeSolver`: Scalar and polarized semivectorial (`Ex`, `Ey`) mode solver on non-uniform grids (`CrossSection` / `Slice`) with polarization-aware symmetry walls.
  - `VFDModeSolver`: Full-vectorial staggered-grid finite-difference solver (Fallahkhair, Li & Murphy 2008) supporting arbitrary 5-component `(nx, ny, 5)` $$[\varepsilon_{xx}, \varepsilon_{xy}, \varepsilon_{yx}, \varepsilon_{yy}, \varepsilon_{zz}]$$ anisotropic and non-reciprocal magneto-optic gyrotropic tensors, shift-invert ARPACK eigensolving, and complex-coordinate-stretching PML absorbing boundaries (`stretchmesh` with $$\operatorname{Im}(\text{factor}) < 0$$).
- **Film Mode Matching (`EMpy.modesolvers.FMM`):**
  - `FMM1d` & `FMM2d`: Semi-analytical 1D slab and 2D channel waveguide solvers (Sudbø 1993, 1994) expanding modes in 1D vertical slice bases with $$k_0$$-scaled root search (invariant between SI meters and microns) and cancellation-free two-endpoint evanescent overlap integrals.
- **Waveguide Junctions & Cascades (`EMpy.modesolvers.geometries`, `interface`):**
  - `SimpleJoint`: Generalized $$2N \times 2N$$ mode-matching S-matrix at waveguide discontinuities supporting both `method="passive"` (conjugated-Poynting projection guaranteeing $$\lVert S \rVert_2 \le 1$$ under basis truncation) and `method="reciprocal"` (bilateral unconjugated Lorentz-reaction polar-midpoint S-matrix enforcing exact $$S = S^T$$), including cross-solver `FDMode` ↔ `FMMMode2d` matching.
  - `SWG` & `GenericDevice`: Bidirectional modal field propagation and Redheffer star-product S-matrix cascading with backward internal-wave sweeps.

### 4. Photonic Devices, Liquid Crystals, Materials & Scattering {#devices-materials}

- **Photonic Integrated Circuits (`EMpy.devices`):** `TransferMatrix` and `ChainMatrix` algebra (`composeTMlist`, `composeCMlist`) and closed-form/cascaded models for `Coupler`, `Line`, `MZ`, `APRR`, `SRR`, `NRR`, Barbarossa tunable coupled-ring filters (`T_TCRR`, `T_CRTCRR`, `T_CTCRR`), dispersive SOI waveguides (`SWG`), and group-index/lossy Fabry–Pérot `Etalon` cavities.
- **Liquid-Crystal Electro-Optics (`EMpy.utils.LiquidCrystalCell`):** Solves the coupled 1D Euler–Lagrange Oseen–Frank elastic free-energy ($$K_{11}, K_{22}, K_{33}$$, intrinsic chiral wavenumber $$q_0$$) and inhomogeneous Laplace electrostatic equations under applied voltage (`scipy.integrate.solve_bvp`) or interpolates experimental tilt profiles, expanding into Berreman `Multilayer` stacks and static `capacitance()`.
- **Vector Diffraction (`EMpy.scattering`):** 3D Kottler–Stratton–Chu surface-current integral (`currentsScatteringKottler`) for arbitrary electric/magnetic equivalent surface currents $$(\mathbf{J}, \mathbf{M})$$ in lossless or passive lossy homogeneous media.

---

## Testing & Verification Strategy {#verification}

`EMpy`'s test suite (`tests/`, 104 tests) continuously verifies physical invariants and analytical limits across Python 3.10–3.14 and `lowest-direct` dependency bounds:

1. **Conservation Laws, Passivity & Reciprocity:**
   - *Lossless structures ($$\kappa = 0$$):* Energy conservation and unitarity ($$R + T = 1$$, $$\sum (\mathrm{DE}_1 + \mathrm{DE}_3) = 1$$, $$S^H S = I$$, $$\lvert \mathrm{THRU} \rvert^2 + \lvert \mathrm{DROP} \rvert^2 = 1$$).
   - *Passive lossy structures ($$\kappa > 0$$):* Strict passivity ($$R + T \le 1$$, $$\lVert S \rVert_2 \le 1$$, $$\lvert \exp(-i\beta L) \rvert < 1$$) and Lorentz reciprocity ($$S = S^T$$ whenever $$\boldsymbol{\varepsilon} = \boldsymbol{\varepsilon}^T$$).
2. **Degenerate-Limit Continuity:**
   - *Anisotropic → Isotropic ($$\Delta n \to 0$$):* Every anisotropic solver (`AnisotropicMaterial`, `LiquidCrystalCell`, `AnisotropicTransferMatrix`, `AnisotropicRCWA`, 5-component `VFDModeSolver`) is verified against its isotropic counterpart both via explicit isotropic tensors ($$\boldsymbol{\varepsilon} = \varepsilon_0 n^2 I_3$$) and along weak-birefringence sequences ($$\Delta n \in \{10^{-4}, 10^{-8}, 0\}$$) with 3D Euler rotations, conical diffraction, and optic-axis degeneracies.
   - *Lossy → Lossless ($$\kappa \to 0^+$$):* Lossy code paths are verified at complex zero-loss dtype (`n = n_r + 0.0j`) and along continuous $$\kappa \to 0^+$$ sequences across propagating, TIR, PML, and sub-cutoff regimes.
3. **Analytical & Cross-Solver Benchmarks:**
   - Cross-checked against Fresnel/Brewster/Airy equations, Kretschmann SPR dips, Jones calculus, Gooch–Tarry twisted-nematic transmission, de Vries cholesteric Bragg bands, Rytov effective-medium theory, 1D guided/leaky slab dispersion relations, 2D Gauss–Legendre quadrature, Fraunhofer Airy diffraction, and pointwise Maxwell curl/divergence residuals.

---

## Literature References {#references}

- D. W. Berreman, "Optics in stratified and anisotropic media: 4 × 4-matrix formulation," *J. Opt. Soc. Am.* **62**(4), 502–510 (1972).
- P. Yeh, "Electromagnetic propagation in birefringent layered media," *J. Opt. Soc. Am.* **69**(5), 742–756 (1979); *Optical Waves in Layered Media* (Wiley, 1988).
- H. A. Macleod, *Thin-Film Optical Filters*, 4th ed. (CRC Press, 2010).
- M. Born and E. Wolf, *Principles of Optics*, 7th ed. (Cambridge University Press, 1999).
- E. Kretschmann and H. Raether, "Radiative decay of non radiative surface plasmons excited by light," *Z. Naturforsch. A* **23**(12), 2135–2136 (1968).
- C. H. Gooch and H. A. Tarry, "The optical properties of twisted nematic liquid crystal structures with twist angles ≤ 90°," *J. Phys. D: Appl. Phys.* **8**(13), 1575–1584 (1975).
- H. de Vries, "Rotatory power and other optical properties of certain liquid crystals," *Acta Crystallogr.* **4**(3), 219–226 (1951).
- P. G. de Gennes and J. Prost, *The Physics of Liquid Crystals*, 2nd ed. (Oxford University Press, 1993).
- Lord Rayleigh, "On the dynamical theory of gratings," *Proc. R. Soc. London Ser. A* **79**(532), 399–416 (1907).
- E. N. Glytsis and T. K. Gaylord, "Three-dimensional (vector) rigorous coupled-wave analysis of anisotropic grating diffraction," *J. Opt. Soc. Am. A* **7**(8), 1399–1419 (1990).
- S. S. Wang and R. Magnusson, "Theory and applications of guided-mode resonance filters," *Appl. Opt.* **32**(14), 2606–2613 (1993).
- M. G. Moharam, E. B. Grann, D. A. Pommet, and T. K. Gaylord, "Formulation for stable and efficient implementation of the rigorous coupled-wave analysis of binary gratings," *J. Opt. Soc. Am. A* **12**(5), 1068–1076 (1995).
- M. G. Moharam, D. A. Pommet, E. B. Grann, and T. K. Gaylord, "Stable implementation of the rigorous coupled-wave analysis for surface-relief gratings: enhanced transmittance matrix approach," *J. Opt. Soc. Am. A* **12**(5), 1077–1086 (1995).
- P. Lalanne and G. M. Morris, "Highly improved convergence of the coupled-wave method for TM polarization," *J. Opt. Soc. Am. A* **13**(4), 779–784 (1996).
- L. Li, "Use of Fourier series in the analysis of discontinuous periodic structures," *J. Opt. Soc. Am. A* **13**(9), 1870–1876 (1996); "Reformulation of the Fourier modal method for surface-relief gratings made with anisotropic materials," *J. Mod. Opt.* **45**(7), 1313–1334 (1998).
- S. M. Rytov, "Electromagnetic properties of a finely stratified medium," *Sov. Phys. JETP* **2**(3), 466–475 (1956).
- K. Koshelev, S. Lepeshov, M. Liu, A. Bogdanov, and Y. Kivshar, "Asymmetric metasurfaces with high-Q resonances governed by bound states in the continuum," *Phys. Rev. Lett.* **121**(19), 193903 (2018).
- A. B. Fallahkhair, K. S. Li, and T. E. Murphy, "Vector finite difference modesolver for anisotropic dielectric waveguides," *J. Lightwave Technol.* **26**(11), 1423–1431 (2008).
- W. C. Chew and W. H. Weedon, "A 3D perfectly matched medium from modified Maxwell's equations with stretched coordinates," *Microw. Opt. Technol. Lett.* **7**(13), 599–604 (1994).
- A. S. Sudbø, "Film mode matching: a versatile numerical method for vector mode field calculations in dielectric waveguides," *Pure Appl. Opt.* **2**(3), 211–233 (1993); "Improved formulation of the film mode matching method for mode field calculations in dielectric waveguides," *Pure Appl. Opt.* **3**(3), 381–388 (1994).
- L. B. Soldano and E. C. M. Pennings, "Optical multi-mode interference devices based on self-imaging: principles and applications," *J. Lightwave Technol.* **13**(4), 615–627 (1995).
- R. Redheffer, "On the relation of transmission-line theory to scattering and transfer," *J. Math. Phys.* **39**(1–4), 269–286 (1960).
- O. Schwelb, "Transmission, group delay, and dispersion in single-ring optical resonators and add/drop filters—a tutorial overview," *J. Lightwave Technol.* **22**(5), 1380–1394 (2004).
- G. Barbarossa, A. M. Matteo, and M. N. Armenise, "Theoretical analysis of triple-coupler ring-based optical guided-wave resonator," *J. Lightwave Technol.* **13**(2), 148–157 (1995); "Novel configuration for fourth-order optical guided-wave resonator," *IEE Proc.-Optoelectron.* **144**(4), 203–208 (1997).
- F. Kottler, "Elektromagnetische Theorie der Beugung an schwarzen Schirmen," *Ann. Phys.* **376**(15), 457–508 (1923); J. A. Stratton and L. J. Chu, "Diffraction theory of electromagnetic waves," *Phys. Rev.* **56**(1), 99–107 (1939).
