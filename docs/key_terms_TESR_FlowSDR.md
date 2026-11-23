[[SDR and Neural Networks]] · [[project2_arxiv_literature_review]] · [[sdr_papers_summary]] · [[pdf_reference_archive]]

# Prerequisite Key Terms: TESR (Ge et al. 2025) & FlowSDR (Dong et al. 2026)

This note extracts the foundational mathematical definitions, notation, and structural roles of the prerequisite concepts underlying **Transfer Learning through Enhanced Sufficient Representation (TESR)** \citep{geTransferLearningEnhanced2025} and **FlowSDR / Neural Gaussian SDR** \citep{dongFlowSDRSufficientDimension2026}.

---

## Category 1: Foundational Sufficiency & Transfer Concepts

### 1. Sufficient and Invariant Representation (SIRep)
* **Mathematical Definition**: Given $S$ source domains $\mathcal{D}_s = \{\mathbf{X}_s, Y_s\}_{s=1}^S$ and pooled dataset $\mathcal{D}_{\text{pool}} = \{\mathbf{X}_{\text{pool}}, Y_{\text{pool}}, Z\}$ where $Z \in \{1, \dots, S\}$ indexes the source domain, a measurable mapping $R_c : \mathbb{R}^d \to \mathbb{R}^{r_c}$ is a Sufficient and Invariant Representation if it satisfies \citep{geTransferLearningEnhanced2025}:
  \[
  Y_s \perp\!\!\perp \mathbf{X}_s \mid R_c(\mathbf{X}_s) \quad \text{and} \quad R_c(\mathbf{X}_{\text{pool}}) \perp\!\!\perp Z, \quad \forall s \in \{1, \dots, S\}
  \]
* **Notation in Paper**: $R_c(\mathbf{X})$ (nonlinear encoder) or $\mathbf{B}_c^\top \mathbf{X}$ with $\mathbf{B}_c \in \mathbb{R}^{d \times r_c}$, $\mathbf{B}_c^\top \mathbf{B}_c = \mathbf{I}_{r_c}$ (linear case).
* **Why It Matters**: Enforcing $Y_s \perp\!\!\perp \mathbf{X}_s \mid R_c(\mathbf{X}_s)$ guarantees that no predictive information from the sources is lost, while enforcing $R_c(\mathbf{X}_{\text{pool}}) \perp\!\!\perp Z$ ensures the representation distribution is invariant across heterogeneous source environments \citep{arjovskyInvariantRiskMinimization2019, geTransferLearningEnhanced2025}.

### 2. Orthogonal Target Enhancement Representation
* **Mathematical Definition**: Given a fixed source representation $R_c(\mathbf{X}_0)$ on the target domain $\mathcal{D}_0 = \{\mathbf{X}_0, Y_0\}$, an augmenting representation $R_t : \mathbb{R}^d \to \mathbb{R}^{r_t}$ is a measurable function satisfying \citep{geTransferLearningEnhanced2025}:
  \[
  Y_0 \perp\!\!\perp \mathbf{X}_0 \mid \left[ R_c(\mathbf{X}_0), R_t(\mathbf{X}_0) \right] \quad \text{and} \quad R_c(\mathbf{X}_0) \perp\!\!\perp R_t(\mathbf{X}_0)
  \]
* **Notation in Paper**: $R_t(\mathbf{X}_0)$ (nonlinear) or $\mathbf{B}_t^\top \mathbf{X}_0$ with $\mathbf{B}_c^\top \mathbf{B}_t = \mathbf{0}$ (linear projection decomposition $\mathbf{P}_{[\mathbf{B}_c, \mathbf{B}_t]} = \mathbf{P}_{\mathbf{B}_c} + \mathbf{P}_{\mathbf{B}_t}$).
* **Why It Matters**: Requiring statistical independence $R_c(\mathbf{X}_0) \perp\!\!\perp R_t(\mathbf{X}_0)$ forces the small target sample $n_0$ to be used exclusively for learning residual target-specific directions not already captured by the source representation $R_c$ \citep{geTransferLearningEnhanced2025}.

---

## Category 2: Nonparametric Dependence & Regularization Apparatus

### 3. Squared Distance Covariance ($\mathbb{V}$) and Fourth-Order U-Statistic ($\mathbb{V}_n$)
* **Mathematical Definition**: For random vectors $\mathbf{U} \in \mathbb{R}^d$ and $\mathbf{V} \in \mathbb{R}^m$ with characteristic functions $\psi_{\mathbf{U}}, \psi_{\mathbf{V}}$ and joint characteristic function $\psi_{\mathbf{U},\mathbf{V}}$, squared distance covariance is defined as \citep{szekelyMeasuringTestingDependence2007}:
  \[
  \mathbb{V}[\mathbf{U}, \mathbf{V}] = \int_{\mathbb{R}^{d+m}} \frac{\left|\psi_{\mathbf{U},\mathbf{V}}(\mathbf{s}, \mathbf{t}) - \psi_{\mathbf{U}}(\mathbf{s})\psi_{\mathbf{V}}(\mathbf{t})\right|^2}{c_d c_m \|\mathbf{s}\|^{d+1} \|\mathbf{t}\|^{m+1}} \, d\mathbf{s} \, d\mathbf{t}
  \]
  which satisfies $\mathbb{V}[\mathbf{U}, \mathbf{V}] \ge 0$ with equality if and only if $\mathbf{U} \perp\!\!\perp \mathbf{V}$, and $\mathbb{V}[\mathbf{U}, \mathbf{V}] \ge \mathbb{V}[R(\mathbf{U}), \mathbf{V}]$ with equality if and only if $R(\mathbf{U})$ is sufficient for $\mathbf{V}$ \citep{shengSufficientDimensionReduction2016, geTransferLearningEnhanced2025}.
* **Notation in Paper**: $\mathbb{V}(\mathbf{U}, \mathbf{V})$ (population) and $\mathbb{V}_n[\mathbf{U}, \mathbf{V}] = \binom{n}{4}^{-1} \sum_{1 \le i_1 < i_2 < i_3 < i_4 \le n} h((\mathbf{U}_{i_1}, \mathbf{V}_{i_1}), \dots, (\mathbf{U}_{i_4}, \mathbf{V}_{i_4}))$ (unbiased U-statistic estimator \citep{huoFastComputingDistance2016}).
* **Why It Matters**: Serves a dual role in TESR: maximizing $\mathbb{V}_n(R(\mathbf{X}), Y)$ recovers sufficient representations without parametric link assumptions, while minimizing $\mathbb{V}_n(R_c(\mathbf{X}_{\text{pool}}), Z)$ and $\mathbb{V}_n(R_t(\mathbf{X}_0), \widehat{R}_c(\mathbf{X}_0))$ enforces domain invariance and target-source disentanglement \citep{geTransferLearningEnhanced2025}.

### 4. Energy Distance ($\mathbb{D}$)
* **Mathematical Definition**: For random vectors $\mathbf{U}, \mathbf{V} \in \mathbb{R}^r$ with independent copies $\mathbf{U}', \mathbf{V}'$, the energy distance is \citep{rizzoEnergyDistance2016}:
  \[
  \mathbb{D}(\mathbf{U} \parallel \mathbf{V}) = 2\operatorname{E}\|\mathbf{U} - \mathbf{V}\| - \operatorname{E}\|\mathbf{U} - \mathbf{U}'\| - \operatorname{E}\|\mathbf{V} - \mathbf{V}'\|
  \]
  satisfying $\mathbb{D}(\mathbf{U} \parallel \mathbf{V}) \ge 0$ with $\mathbb{D}(\mathbf{U} \parallel \mathbf{V}) = 0 \iff \mathbf{U} \stackrel{d}{=} \mathbf{V}$.
* **Notation in Paper**: $\mathbb{D}(R(\mathbf{X}_s) \parallel \gamma_{r_c})$ (population) and $\mathbb{D}_n(\mathbf{U} \parallel \mathbf{V}) = \binom{n}{2}^{-1} \sum_{1 \le i < j \le n} h_e(\mathbf{u}_i, \mathbf{u}_j; \mathbf{v}_i, \mathbf{v}_j)$ where $\gamma_{r_c} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{r_c})$.
* **Why It Matters**: Replaces the unstable adversarial GAN discriminator of [[DEEP NONLINEAR SUFFICIENT DIMENSION REDUCTION 2024|DDR]] \citep{huang2024deep} with a closed-form U-statistic regularizer that pushes representations toward a standard Gaussian distribution \citep{geTransferLearningEnhanced2025}.

### 5. Anisotropic Besov Space ($B_{p,q,\tilde{\beta}}^{\boldsymbol{\beta}}(\Omega)$)
* **Mathematical Definition**: A function space characterized by direction-specific smoothness indices $\boldsymbol{\beta} = (\beta_1, \dots, \beta_d)^\top \in \mathbb{R}_{++}^d$ along each coordinate axis, with harmonic-mean effective smoothness \citep{suzukiDeepLearningAdaptive2021}:
  \[
  \tilde{\beta} = \left( \sum_{k=1}^d \frac{1}{\beta_k} \right)^{-1}
  \]
* **Notation in Paper**: $B_{p,q,\tilde{\beta}}$, with source smoothness $\tilde{\beta}_c$, target residual smoothness $\tilde{\beta}_t$, and full target smoothness $\tilde{\beta}_0$.
* **Why It Matters**: Governs the non-asymptotic excess risk bound in Theorem 1 of \citet{geTransferLearningEnhanced2025}, proving that transfer learning via TESR beats target-only learning when the residual target component $R_t$ is smoother than the full target representation $R_0$ ($\tilde{\beta}_t > \tilde{\beta}_0$).

---

## Category 3: Likelihood, Conditional Variance & Conformal SDR Apparatus

### 6. Information Index ($\mathcal{J}(\mathbf{A})$)
* **Mathematical Definition**: For a projection matrix $\mathbf{A} \in \mathbb{R}^{p \times q}$ ($q \le p$), the population information index is \citep{yinSuccessiveDirectionExtraction2008, dongFlowSDRSufficientDimension2026}:
  \[
  \mathcal{J}(\mathbf{A}) = \operatorname{E}\left\{ \log f(Y \mid \mathbf{A}^\top \mathbf{X}) \right\} = \mathcal{I}(Y; \mathbf{A}^\top \mathbf{X}) + \operatorname{E}\left\{ \log f(Y) \right\}
  \]
* **Notation in Paper**: $\mathcal{J}(\mathbf{B})$ (population target) and $M_n(\mathbf{B}) = \sup_{\boldsymbol{\theta} \in \Theta} \frac{1}{n} \sum_{i=1}^n \log f_{\boldsymbol{\theta}}(Y_i \mid \mathbf{B}^\top \mathbf{X}_i)$ (empirical profile log-likelihood over Stiefel manifold $\mathcal{V}_{p,d}$).
* **Why It Matters**: By Lemma A.1 of \citet{dongFlowSDRSufficientDimension2026}, $\mathcal{J}(\mathbf{B})$ is invariant to orthogonal rotations $\mathbf{C} \in \mathcal{O}(q)$ and uniquely attains its global maximum $\mathcal{J}(\mathbf{I}_p)$ if and only if $\operatorname{span}(\mathbf{B}) \supseteq \mathcal{S}_{Y \mid \mathbf{X}}$, establishing Fisher consistency for FlowSDR.

### 7. Monotone Rational-Quadratic Spline (RQS) Conditional Flow
* **Mathematical Definition**: An invertible, strictly monotone univariate transformation $Z = T_{\boldsymbol{\theta}}(Y; \mathbf{w}) \sim f_0 = \mathcal{N}(0,1)$ conditioned on $\mathbf{w} = \mathbf{B}^\top \mathbf{X}$, parameterized over $K$ knots $\{y_k(\mathbf{w}), z_k(\mathbf{w}), s_k(\mathbf{w})\}_{k=0}^K$ output by a neural network $\phi_{\boldsymbol{\theta}}(\mathbf{w})$, inducing conditional density \citep{durkanNeuralSplineFlows2019, dongFlowSDRSufficientDimension2026}:
  \[
  f_{\boldsymbol{\theta}}(y \mid \mathbf{w}) = f_0\left\{ T_{\boldsymbol{\theta}}(y; \mathbf{w}) \right\} \left| \frac{\partial T_{\boldsymbol{\theta}}(y; \mathbf{w})}{\partial y} \right|
  \]
* **Notation in Paper**: $T_{\boldsymbol{\theta}}(y; \mathbf{B}^\top \mathbf{X}_i)$ and $\ell_n(\mathbf{B}, \boldsymbol{\theta})$.
* **Why It Matters**: Unlike affine coupling flows (which collapse to location-scale transforms in 1D), RQS flows approximate multimodal, skewed, and heavy-tailed conditional densities $f(Y \mid \mathbf{B}^\top \mathbf{X})$, enabling FlowSDR to recover sufficient directions entering purely through conditional variance or tail shape \citep{dongFlowSDRSufficientDimension2026}.

### 8. Profile Log-Conditional-Variance Estimand (Neural Gaussian SDR)
* **Mathematical Definition**: Under the working heteroscedastic Gaussian model $Y \mid \mathbf{B}^\top \mathbf{X} = \mathbf{w} \sim \mathcal{N}(\mu_\gamma(\mathbf{w}), \sigma_\gamma^2(\mathbf{w}))$, profiling out $(\mu, \sigma^2)$ yields the population objective (Proposition C.1 of \citet{dongFlowSDRSufficientDimension2026}):
  \[
  \mathbf{B}^{\mathrm{G}} = \operatorname*{argmin}_{\mathbf{B} \in \mathcal{V}_{p,d}} \operatorname{E}\left\{ \log \operatorname{Var}(Y \mid \mathbf{B}^\top \mathbf{X}) \right\}
  \]
* **Notation in Paper**: $\mathbf{B}^{\mathrm{G}}$ (population) and $\widehat{\mathbf{B}}^{\mathrm{G}} = \operatorname*{argmax}_{\mathbf{B} \in \mathcal{V}_{p,d}} M_n^{\mathrm{G}}(\mathbf{B})$ (sample).
* **Why It Matters**: Clarifies why [[MAVE-Gradient-Estimation-2011-12-01|MAVE]] ($\operatorname*{argmin}_{\mathbf{B}} \operatorname{E}\{\operatorname{Var}(Y \mid \mathbf{B}^\top \mathbf{X})\}$) fails to capture the **Central Variance Subspace (CVS)** $\mathcal{S}_{\operatorname{Var}(Y \mid \mathbf{X})}$ \citep{cookDimensionReductionConditional2002}: by Jensen's inequality, expectation of variance only depends on the conditional mean $\operatorname{E}(Y \mid \mathbf{B}^\top \mathbf{X})$, whereas expectation of *log* conditional variance depends directly on the dispersion function $\operatorname{Var}(Y \mid \mathbf{B}^\top \mathbf{X})$ \citep{dongFlowSDRSufficientDimension2026}.

---

## Suggested New Notes
- `Energy Distance in SDR` — comparison of energy distance vs. GAN regularization in neural SDR.
- `Monotone Rational-Quadratic Spline Flows` — density transport mechanisms for conditional density estimation.
