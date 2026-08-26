# Summary: Generalized Multilinear Models for Sufficient Dimension Reduction on Tensor-Valued Predictors

- **Authors:** Daniel Kapla and Efstathia Bura
- **Journal:** *The Annals of Statistics* (DOI: [10.1214/25-AOS2598](https://doi.org/10.1214/25-AOS2598)), pp. 981–1005
- **Supplement:** Kapla and Bura (2026), DOI: [10.1214/25-AOS2598SUPP](https://doi.org/10.1214/25-AOS2598SUPP) (Appendices A–F)
- **Local PDF:** [ref/Generalized Multilinear Models for Sufficient Dimension Reduction on Tensor-Valued Predictors Kapla 2025.pdf](file:///Users/manish/gh/research/sdr/ref/Generalized%20Multilinear%20Models%20for%20Sufficient%20Dimension%20Reduction%20on%20Tensor-Valued%20Predictors%20Kapla%202025.pdf)
- **Obsidian Hub Links:** [[pdf_reference_archive]] · [[sdr_papers_summary]] · [[Fusing sufficient dimension reduction with neural networks. 2021]] · [[SDR and Neural Networks]]
- **BibTeX Key:** `kaplaGeneralizedMultilinearModels2025`

---


## 1. Overview and Motivation

The paper tackles the general regression problem of fitting a response **Y** of arbitrary form (univariate, multivariate, or tensor-valued) on a **tensor-valued predictor** 𝒳 (a multiway array of order r, with dimensions p₁ × ⋯ × pᵣ, continuous or discrete entries). The authors work within **sufficient dimension reduction (SDR)**: the goal is to replace the huge predictor 𝒳 with a lower-dimensional tensor ℛ(𝒳) of dimension q₁ × ⋯ × qᵣ (qⱼ ≤ pⱼ) such that

> F(Y | 𝒳) = F(Y | ℛ(𝒳))

i.e., the reduced predictor carries *all* the information 𝒳 contains about Y, achieving the **maximum dimension reduction without information loss**.

Key ingredients of the approach:

- **Inverse regression.** Instead of modeling Y | 𝒳 directly, they model the inverse conditional distribution 𝒳 | Y and exploit Cook's (2007) equivalence: Y | 𝒳 =ᵈ Y | ℛ(𝒳) ⟺ 𝒳 | (Y, ℛ(𝒳)) =ᵈ 𝒳 | ℛ(𝒳). A sufficient statistic for Y in the inverse regression is a sufficient reduction for the forward regression.
- **Quadratic exponential family.** 𝒳 | Y is assumed to follow a full-rank quadratic exponential family whose natural parameter depends on Y and whose first and second moments admit a **separable Kronecker product structure**. Members are fully characterized by their first two moments. This family includes:
  - the **multilinear normal** distribution (continuous tensor data), and
  - the **multilinear Ising** distribution (binary tensor data — a generalization of the inverse Ising model, i.e., a multivariate Bernoulli with up to second-order interactions),
  - and mixtures of the two. Higher moments could be incorporated but make the parameter count prohibitive.
- **Maximum likelihood estimation** of the sufficient reduction, with the classical optimality properties: consistency, efficiency, and asymptotic normality.

### The identifiability problem and the manifold solution

The central estimation challenge in multilinear tensor regression is **parameter non-identifiability**: because β₂ ⊗ β₁ = (aβ₂) ⊗ (a⁻¹β₁) for any a ≠ 0 (and likewise for the covariance components), many parameter values describe the same density, so consistency and asymptotic normality cannot be established in the usual way. Prior work (Zhou, Li and Zhu 2013; Hoff 2015; Sun and Li 2017; Li and Zhang 2017; Lock 2018; Llosa-Vite and Maitra 2023) dealt with this via penalty terms and/or sparsity constraints. This paper instead **models the parameter space as a smooth embedded manifold**, which restores identifiability without any penalization while enabling rich modeling flexibility.

### Main contributions

1. Formulating the dimension reduction problem via the quadratic exponential family (modeling up to two-way interactions), which yields the **minimal sufficient dimension reduction in closed form**.
2. **Modeling flexibility** by defining the parameter space as a smooth embedded manifold.
3. Deriving the **MLE of the sufficient reduction subject to multilinear (Kronecker) constraints**, overcoming parameter non-identifiability.
4. Establishing **consistency and asymptotic normality** of the estimators.
5. Developing **computationally efficient estimation algorithms** — very fast for multilinear normal predictors.

### Motivating example (EEG data)

The electroencephalography (EEG) dataset (Begleiter 1999, UCI KDD archive) contains 77 alcoholic and 45 control subjects (n = 122). Each subject's predictor is a 256 × 64 matrix: voltage from 64 scalp electrodes sampled at 256 time points over 1 second (256 Hz), averaged over 120 trials under a single stimulus condition (S1). The response is binary (Y = 1 alcoholic, Y = 0 control). Here p₁ × p₂ = 256 × 64 = 16,384 ≫ 122 = n — a setting where most earlier methods needed to pre-reduce the data (e.g., via (2D)²PCA) or impose sparsity/regularization, while the proposed method works directly on the raw data.

---

## 2. Notation (Section 2)

- Tensors are denoted in fraktur (e.g., 𝒜); an r-tensor 𝒜 ∈ ℝ^{q₁×⋯×qᵣ} has r modes (axes); a matrix is a 2-tensor.
- **k-mode product** 𝒜 ×ₖ B: multiplies tensor 𝒜 by matrix B along mode k; commutes across different modes; applying it over all modes is the **multilinear multiplication / Tucker operator**. For matrices: A ×₁ B₁ = B₁A, A ×₂ B₂ = AB₂ᵀ.
- **vec(𝒜)**: vectorization in lexicographic index order; **vech**: half-vectorization of a symmetric matrix; **𝒜₍ₖ₎**: mode-k flattening/matricization.
- Inner product ⟨𝒜, ℬ⟩ = (vec 𝒜)ᵀ(vec ℬ); Frobenius norm ‖𝒜‖_F = √⟨𝒜, 𝒜⟩; outer product 𝒜 ∘ ℬ; iterated Kronecker product ⊗ₖ Aₖ.
- The gradient of any-shaped function of any-shaped argument is defined via vectorizations: ∇_𝒳 ℱ = ∂(vec ℱ(𝒳))ᵀ / ∂(vec 𝒳).

---

## 3. Problem Formulation and the GMLM (Sections 3–4)

### Quadratic exponential family (Eq. 3)

The inverse conditional density is

> f(𝒳 | Y = y) = h(𝒳) exp( η_yᵀ t(𝒳) − b(η_y) ),  with  t(𝒳) = (vec 𝒳, vech((vec 𝒳)ᵀ(vec 𝒳)))

- η₁y captures the **first-order** and η₂y the **second-order** relationship between Y and 𝒳.
- h is nonnegative; b is twice continuously differentiable and strictly convex.

### Generalized Multilinear Model (GMLM) (Eqs. 4–8)

Let ℱ_y be a known, zero-mean (E ℱ_Y = 0) **tensor-valued** mapping of y taking values in ℝ^{q₁×⋯×qᵣ}. The GMLM parameterizes:

- **First-order natural parameter:** η₁y = vec( η + ℱ_y ×ⱼ βⱼ ), i.e., a mode-wise multilinear regression on ℱ_y with B = ⊗ⱼ βⱼ, where βⱼ ∈ ℝ^{pⱼ×qⱼ} are the mode-wise reduction matrices.
- **Second-order natural parameter (independent of Y):** η₂ = c D_pᵀ vec Ω with **separable** Ω = ⊗ⱼ Ωⱼ, Ωⱼ ∈ ℝ^{pⱼ×pⱼ} symmetric; c is a known nonzero constant easing modeling (e.g., c = −1/2 for the normal makes Ω the inverse covariance); D_p is the duplication matrix.

The resulting GMLM density (Eq. 8) is

> f(𝒳 | Y = y) = h(𝒳) exp( ⟨𝒳, η + ℱ_y ×ⱼ βⱼ⟩ + c ⟨𝒳, 𝒳 ×ⱼ Ωⱼ⟩ − b(η_y) )

placing the problem in a generalized-linear-model framework with a mode-wise linear predictor–response relation.

### Theorem 4.1 — Sufficient reduction

Under the quadratic exponential family inverse regression model (8),

> **ℛ(𝒳) = (𝒳 − E 𝒳) ×ₖ βₖᵀ**   (vectorized: vec ℛ(𝒳) = Bᵀ vec(𝒳 − E 𝒳))

is a **sufficient reduction** for the forward regression Y | 𝒳; it is **minimal** if all βⱼ are full rank. The reduction acts **linearly along each mode** of the tensor.

Special cases:

- **Vector predictor (r = 1):** R(x) = β₁ᵀ(x − E X) ∈ ℝ^{q₁}.
- **Matrix predictor (r = 2):** R(X) = β₁ᵀ (X − E X) β₂ ∈ ℝ^{q₁×q₂}, with B = β₂ ⊗ β₁.

---

## 4. Manifolds and Parameter Spaces (Section 5)

Because the Kronecker components βⱼ, Ωⱼ are not identifiable (rescaling one by a and another by a⁻¹ leaves the model unchanged), the authors constrain the parameter space Θ to be a **smooth embedded submanifold** of a Euclidean space. Embedded manifolds inherit differentiation (and, from the ambient Euclidean Riemannian structure, higher derivatives) while avoiding abstract manifold machinery, and remain consistent with classical linear-subspace parameter sets.

- **Definition 5.1 (Embedded smooth manifold):** 𝒜 ⊆ ℝ^p is a d-dimensional embedded smooth manifold if every point has a smooth bicontinuous chart into an open subset of ℝ^d.
- **Theorem 5.2 (Kronecker product manifolds):** If 𝒜, 𝔅 are smooth embedded submanifolds (excluding 0), then {A ⊗ B : A ∈ 𝒜, B ∈ 𝔅} is a smooth embedded d-manifold under either:
  - the **sphere condition** — at least one of 𝒜, 𝔅 is spherical (constant Frobenius norm), with d = dim 𝒜 + dim 𝔅; or
  - the **cone condition** — both are cones (scale invariant), with d = dim 𝒜 + dim 𝔅 − 1.
- **Theorem 5.3 (Parameter manifolds):** With mode-wise building blocks 𝔅ₖ ⊆ ℝ^{pₖ×qₖ}∖{0} and 𝔒ₖ ⊆ Sym_{pₖ×pₖ}∖{0} that are spheres or cones, the full constrained parameter space Θ = ℝ^p × 𝔎_B × vech(𝔎_Ω) ⊂ ℝ^{p(p+2q+3)/2} (and each of its components) is a smooth embedded manifold.
- **Modeling flexibility (Section 5.2, Table 1):** many matrix manifolds (cones or spherical) can serve as building blocks — enabling, e.g., **low-rank regression** or **band matrices** that restrict two-way interactions to direct axis neighbors. The Grassmann manifold is excluded since it is not embedded in ℝ^{p×q}.

---

## 5. Maximum Likelihood Estimation (Section 6)

Given i.i.d. (𝒳ᵢ, Yᵢ), i = 1,…,n, the MLE maximizes the empirical log-likelihood (Eq. 10) over Θ (Eq. 11). Unlike classical GLMs, an invertible link may not exist; knowing the conditional first and second moments suffices.

**Theorem 6.1 (Likelihood gradients):** closed-form partial gradients of the log-likelihood with respect to η, each βⱼ, and each Ωⱼ are derived (expressed via the residuals 𝒳ᵢ − E_θ[𝒳 | Y = yᵢ], mode-j unfoldings, and 𝒳 ⊗ 𝒳 moments), enabling gradient-descent fitting of any GMLM.

### 5.1 Multilinear normal predictors (Section 6.1)

- The multilinear normal extends the matrix normal (Dawid 1981; Arnold 1981; Kollo and von Rosen 2005; Hoff 2011; Ohlson, Ahmad and von Rosen 2013) to tensors; its defining feature is the **Kronecker-separable covariance** Σ = ⊗ₖ Σₖ with symmetric positive definite Σₖ.
- With c = −1/2, the GMLM parameters relate as **μ_y = ℱ_y ×ₖ Ωₖ⁻¹ βₖ** and **Ωₖ = Σₖ⁻¹** (Eq. 12); w.l.o.g. E 𝒳 = 0 implies η = 0.
- **Initialization heuristic:** moment-based mode-wise marginal covariances of 𝒳 and ℱ_Y; take leading eigenvectors (Uₖ, Vₖ) and set βₖ⁽⁰⁾ = Uₖ √Dₖ Vₖᵀ, Ωₖ⁽⁰⁾ = I.
- **Cyclic ("flip-flop"-type) updating:** setting the βⱼ-gradient to zero yields a **closed-form update** for βⱼ given the other parameters (Eq. 13). The Ωⱼ score equation is a quadratic matrix equation, so in practice Ωⱼ is updated faster and more stably from **mode-wise residual covariances**, with a scaling factor s̃ chosen so the mean squared error equals the trace of the Kronecker covariance. Ill-conditioned updates are regularized via (s̃Σ̃ⱼ + 0.2 λ₁ I)⁻¹, typically needed only in the first few iterations. Estimates can be projected back onto the manifold after each update.
- The scheme relates to the block coordinate descent / "flip-flop" algorithms of Mardia and Goodall (1993), Dutilleul (1999), Lu and Zimmerman (2005), and Manceur and Dutilleul (2013). Notably, Kronecker-structured MLEs can exist for **very small samples** (as few as 3 observations; Drton, Kuriki and Hoff 2021), so n may be smaller than any mode dimension pⱼ.

### 5.2 Multilinear Ising model (Section 6.2)

- The Ising (Lenz–Ising) model, from statistical physics (ferromagnetism; spins ±1 on a lattice with neighbor interactions), is a discrete quadratic exponential family for multivariate binary outcomes.
- For x ∈ {0,1}^p, the pmf is P_γ(x) = p₀(γ) exp(vech(xxᵀ)ᵀγ), with partition function p₀(γ)⁻¹ = Σ_x exp(vech(xxᵀ)ᵀγ) (Eq. 14). Diagonal parameters γⱼⱼ are conditional log-odds of Xⱼ = 1; off-diagonal γⱼₗ are conditional log-odds of joint occurrence (Eq. 15).
- The paper extends conditional Ising models (Cheng et al. 2014; Bura et al. 2022) to **tensor-valued binary predictors** by imposing Kronecker constraints on the natural parameter, giving the GMLM form (Eq. 16) with η = 0 (the diagonal of Ω partially takes over the role of η) and natural parameter γ_y = D_pᵀ vec(⊗ₖ Ωₖ + diag(vec(ℱ_y ×ₖ βₖ))) (Eq. 17). Here the Ωₖ need only be symmetric (not positive definite).
- No closed-form cyclic scheme exists; estimation uses the Theorem 6.1 gradients within a gradient-descent variant (details in Supplement Appendix E).

---

## 6. Statistical Properties (Section 7)

- Estimators are cast as **M-estimators**; **Definition 7.1** introduces weak and strong M-estimators (an "almost maximizer" up to o_P(1) or o_P(n⁻¹) suffices).
- **Theorem 7.2 (Consistency and asymptotic normality):** under regularity Conditions 1–3 (Supplement Appendix B), a strong M-estimator sequence from the log-likelihood over the manifold Θ exists; every strong M-estimator is **consistent** (θ̂_n →ᵖ θ₀) and **asymptotically normal**:

  > √n (θ̂_n − θ₀) →ᵈ N(0, Σ_{θ₀})

- **Intuition for Σ_{θ₀}:** in the unconstrained setting the asymptotic covariance is a positive definite matrix (a hyperellipsoid); on the manifold, Σ_{θ₀} is positive semidefinite — a degenerate hyperellipsoid obtained by intersecting the unconstrained ellipsoid with the **tangent space of Θ at θ₀**, distorted by the manifold's local curvature.
- Theorem 7.2 is a special case of the more general Theorem B.4 (Supplement), which generalizes Theorem 5.23 of van der Vaart (1998) from open Euclidean sets to embedded manifolds.

---

## 7. Simulations (Section 8)

**Setup:** competitors include Tensor Sliced Inverse Regression (TSIR; Ding and Cook 2015), Multiway Generalized Canonical Correlation Analysis (MGCCA; Chen, Kolar and Tsay 2021; Girka et al. 2024), higher-order PCA via the Tucker decomposition (HOPCA; Kolda and Bader 2009), and plain PCA on vectorized data; for binary data also Logistic PCA (LPCA) and Convex Logistic PCA (CLPCA; Landgraf and Lee 2020). Sample sizes n = 100, 200, 300, 500, 750; 100 replications each. Accuracy is measured by a normalized **subspace distance** d(B, B̂) ∈ [0, 1] (0 = subspace overlap, 1 = orthogonality).

### Multilinear normal experiments

| Setting | Description | Outcome |
|---|---|---|
| (1a) | 2×3×5 array; ℱ_y ≡ y scalar; βₖ = e₁; AR(0.5) Ωₖ | TSIR and GMLM equivalent and best (TSIR is known to give the MLE under matrix-normal settings); MGCCA, HOPCA, PCA far behind |
| (1b) | Cubic polynomial relation via 2×2×2 ℱ_y | GMLM slightly beats TSIR (avoids slicing; estimates an 8-dim subspace) |
| (1c) | Rank-1 βₖ matrices | GMLM clearly best; **TSIR fails** — traced to poor estimation of mode covariance matrices under a high signal-to-noise ratio, not to the classical slicing/multimodality issue |
| (1d) | Tri-diagonal Ωₖ (covariance structure info) | Like (1b), but GMLM gains a statistically significant lead over TSIR |
| (1e) | **Misspecified model**: multivariate (not multilinear) normal, Kronecker assumptions violated, wrong ℱ_y fitted | GMLM, TSIR, MGCCA on par; GMLM modestly better up to n ≈ 300, MGCCA leads after n ≈ 500; PCA and HOPCA outperformed |

Overall: GMLM excels at high signal-to-noise ratios; TSIR can be more robust at low signal-to-noise.

### Ising experiments

Y ~ Uniform[−1, 1]; 𝒳 is 2×3 with conditional matrix Ising distribution; baseline uses ℱ_y built from sin(πy), cos(πy). Settings: (2a) purely linear relation; (2b) baseline; (2c) rank-1 β₁, β₂ (low-rank regression); (2d) "physical" Ising model — tri-diagonal, zero-diagonal Ωₖ with identical neighbor couplings (temperature 1/2), the mean effect acting as an external magnetic field.

**Result:** the Ising GMLM is the best performer in every setting. The performance ordering is roughly: GMLM > TSIR > MGCCA > LPCA ≈ CLPCA > PCA ≈ HOPCA. Methods ignoring the response (PCA, HOPCA, LPCA, CLPCA) perform poorly.

---

## 8. EEG Data Analysis (Section 9)

**Data versions:** the 2D EEG set (S1 stimulus only; 256×64 matrices) and the 3D EEG set (adding the stimulus-condition mode: S1 only, S1 = S2 matched, S1 ≠ S2 non-matched — pictures from the 1980 Snodgrass and Vanderwart set — giving 256×64×3 tensors). Response: alcoholic (1) vs control (0).

**Competing methods:** K-PIR(ls) (Pfeiffer, Kapla and Bura 2021), LSIR (Pfeiffer, Forzani and Bura 2012), TSIR (Ding and Cook 2015), a newly regularized TSIR(reg), GMLM, and GMLM(FFT) — a variant modeling the time-axis reduction as a time-series mixture retaining only the 10 highest-magnitude Fourier frequencies. Most legacy methods require prescreening with (2D)²PCA down to (p₁, p₂) = (3,4), (15,15), or (20,30); GMLM and TSIR apply directly to the raw 16,384-dimensional data. Evaluation: AUC under leave-one-out cross-validation.

**Results (mean AUC, Table 2):**

| Method | 2D (3,4) | 2D (15,15) | 2D (20,30) | 2D full (256,64) | 3D (3,4,3) | 3D (15,15,3) | 3D (20,30,3) | 3D full (256,64,3) |
|---|---|---|---|---|---|---|---|---|
| K-PIR(ls) | 78% | 78% | 78% | — | 74% | 75% | 75% | — |
| LSIR | 85% | 81% | 83% | 39% | 88% | 81% | 71% | 52% |
| TSIR | 85% | 83% | 80% | 69% | 87% | 89% | 88% | 86% |
| TSIR(reg) | 85% | 84% | 84% | 84% | 87% | 84% | 84% | 84% |
| GMLM | 85% | 84% | 84% | 84% | 88% | 87% | 87% | 87% |
| GMLM(FFT) | — | — | — | 85% | — | — | — | 87% |

Key observations:

- K-PIR(ls) is flat at 78% (2D) and worsens in 3D (74–75%) — the only method not helped by the extra mode.
- LSIR is excellent with heavy prescreening (up to 85%/88%) but **fails on full data** (39%/52%) since the time dimension exceeds n.
- TSIR's collapse on the full 2D data (69%) vs. its stable 3D performance was diagnosed (Supplement Appendix D) as caused by **inverting ill-conditioned covariance estimates**; adding regularization (TSIR(reg)) stabilizes it at 84% everywhere.
- **GMLM matches TSIR(reg) in 2D (84% on full data) and is slightly better and uniformly stable in 3D (87%)**, with no prescreening or variable selection. GMLM(FFT) reaches 85% (2D) and 87% (3D) on full data.

A second real-data proof of concept — chess boards as binary 8×8 matrices analyzed with the Ising GMLM — is in Supplement Appendix F.

---

## 9. Discussion and Future Directions (Section 10)

- The GMLM provides a general inverse-regression formulation for tensor-valued predictors and yields a multilinear sufficient reduction for the forward regression/classification problem, with consistent, asymptotically normal estimators. It can also serve as a **stand-alone tensor-on-tensor regression model**, and can be viewed as an extension of reduced-rank regression to tensor-valued responses.
- The **manifold viewpoint** both solves non-identifiability and enables problem-specific parameter spaces that encode domain knowledge (e.g., longitudinal matrices with discrete AR(k) time dependence across columns).
- The multilinear Ising model extends the covariate-adjusted Ising approach of Cheng et al. (2014) while exploiting separable first/second moments — avoiding sparsity assumptions or penalization despite high dimensionality.
- **One-parameter Ising special case:** with a fixed symmetric zero-diagonal matrix A, Θ = {0} × {θA + 0.5 I_p : θ ∈ ℝ} recovers the model of Xu and Mukherjee (2023). Their non-normal limiting results do not contradict this paper's asymptotic normality: they let the dimension p → ∞ (a different limiting experiment), whereas here n → ∞ with fixed p.
- **Future work:** mixtures of continuous and binary tensor-valued predictors; replacing the single Kronecker product (an implicit rank-1 constraint) with **sums of separable Kronecker terms**, provided the sum still forms a parameter manifold.

---

## 10. Paper Structure and Supplement

- **Main paper:** §2 Notation; §3 Problem formulation; §4 GMLM and Theorem 4.1; §5 Manifolds/parameter spaces; §6 MLE (multilinear normal, multilinear Ising); §7 Asymptotics; §8 Simulations; §9 EEG data; §10 Discussion.
- **Supplement (Kapla and Bura 2026):** Appendix A — vectorization/matricization examples; B — proofs and regularity conditions (incl. general Theorem B.4); C — numerical example of tensor vs. vector representation benefits; D — TSIR vs. GMLM on EEG and the TSIR(reg) fix; E — technical estimation details for the multilinear Ising model; F — Ising GMLM proof-of-concept on chess data.
