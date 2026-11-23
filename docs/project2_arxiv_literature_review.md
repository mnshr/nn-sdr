[[SDR and Neural Networks]] · [[Oct 09 2026 SDR]] · [[Sep 25 2026 SDR]] · [[key_terms_TESR_FlowSDR]]

# Project 2 Literature Review: Pretrained Sufficiency Auditing, Transfer Learning & Uncertainty Quantification in SDR

This note synthesizes the systematic arXiv literature review conducted for Project 2 following the advisor meetings on [[Sep 25 2026 SDR|September 25, 2026]] and [[Oct 09 2026 SDR|October 9, 2026]]. Raw JSON query outputs and full-text PDFs are archived in `arxiv_search_project2/` and `ref/`.

---

## 1. Screening Matrix of Key Papers Across Themes 1–3

| # | Paper Title & Short Name | Authors & Year | Venue & DOI / Link | Core Target & Mechanism | Datasets & Stack |
|---|---|---|---|---|---|
| 1 | **Transfer Learning through Enhanced Sufficient Representation** (TESR) | Yeheng Ge, Xueyu Zhou, Jian Huang (2025) | [arXiv:2502.20414](https://arxiv.org/abs/2502.20414) | Source invariant sufficient representation $R_c(\mathbf{X})$ + target-specific orthogonal augmentation $R_t(\mathbf{X}_0)$ via Distance Covariance $\mathbb{V}_n$ and Energy Distance $\mathbb{D}_n$ \citep{geTransferLearningEnhanced2025} | **GTEx (13 brain tissues, 1,813 genes, JAM2)**, CIFAR-10 $\to$ PACS; [[PyTorch]] |
| 2 | **FlowSDR: Sufficient Dimension Reduction via Conditional Normalizing Flows** (FlowSDR) | Yuexiao Dong, Kenichiro McAlinn, Edoardo Airoldi, Lei Li (2026) | [arXiv:2606.01346](https://arxiv.org/abs/2606.01346) | [[Central-Subspace-Definition-Properties-2005-06-01|Central subspace]] $\mathcal{S}_{Y \mid \mathbf{X}}$ and full conditional density $f(Y \mid \mathbf{B}^\top \mathbf{X})$ via monotone rational-quadratic spline flows; **Neural Gaussian SDR** minimizing $\operatorname{E}\{\log \operatorname{Var}(Y \mid \mathbf{B}^\top \mathbf{X})\}$ + **Conformal Prediction** \citep{dongFlowSDRSufficientDimension2026} | Heteroscedastic, heavy-tailed ($t_3$), multimodal, and pure-tail simulations; UTKFace ($p=1812$); [[PyTorch]] + R |
| 3 | **SDR via Inverse Conditional Mean or Variance Independence** (ICMI / ICVI) | Jicai Liu, Yu Zhang, Jinhong Li (2026) | *Statistics and Computing* (2026)<br>[DOI: 10.1007/s11222-026-10943-4](https://doi.org/10.1007/s11222-026-10943-4) | Central subspace via projection- and kernel-based inverse conditional mean and variance independence \citep{liuSufficientDimensionReduction2026} | High-dimensional simulations and real benchmarks |
| 4 | **A Reproducing-Kernel-Based Nonparametric Test for Conditional Independence** (CCCO) | Yin Tang, [[SDR Bing Li|Bing Li]] (2026) | *Journal of Multivariate Analysis* (2026)<br>[DOI: 10.1016/j.jmva.2026.105709](https://doi.org/10.1016/j.jmva.2026.105709) | Nonparametric test of $H_0: Y \perp\!\!\perp X \mid Z$ using spectral decomposition of the Conjoined Conditional Covariance Operator in [[RKHS]] \citep{tangReproducingKernelBasedNonparametric2026} | Activity/biometrics and macroeconomic functional data |
| 5 | **Robust SDR via $\alpha$-Distance Covariance** | Hsin-Hsiung Huang, Feng Yu, Teng Zhang (2024) | *Journal of Nonparametric Statistics* (2024)<br>[DOI: 10.1080/10485252.2024.2313137](https://doi.org/10.1080/10485252.2024.2313137) | Robust central subspace $\mathcal{S}_{Y \mid \mathbf{X}}$ via $\alpha$-Distance Covariance optimization over Stiefel manifold | Heavy-tailed / outlier benchmarks |
| 6 | **Conditional Variance Estimator (CVE)** | Lukas Fertl, Efstathia Bura (2022) | *Bernoulli* 28(3): 1957–1983 (2022)<br>[DOI: 10.3150/21-BEJ1402](https://doi.org/10.3150/21-BEJ1402) | Forward exhaustive SDR minimizing conditional variance without [[Linearity-Condition-LCM-Assumption-2024-10-10|linearity]] or constant variance conditions \citep{fertlConditionalVariance2022} | Additive error continuous models; R (`CVarE`) |
| 7 | **Ensemble Conditional Variance Estimator (ECVE)** | Lukas Fertl, Efstathia Bura (2022) | *Electronic Journal of Statistics* 16(1): 2405–2436 (2022)<br>[DOI: 10.1214/22-EJS1994](https://doi.org/10.1214/22-EJS1994) | Forward exhaustive SDR for general non-additive error regressions \citep{fertlEnsembleConditionalVariance2022} | Non-additive error models; R (`CVarE`) |
| 8 | **Principal Asymmetric Least Squares (PALS)** | Abdul-Nasah Soale, Yuexiao Dong (2022) | *Journal of Nonparametric Statistics* 34(3): 586–613 (2022)<br>[DOI: 10.1080/10485252.2021.2025237](https://doi.org/10.1080/10485252.2021.2025237) | Linear and nonlinear SDR under heteroscedasticity by synthesizing expectile levels \citep{soalePrincipalAsymmetricLeastSquares2022} | Heteroscedastic regression simulations |
| 9 | **Expectile-Assisted Inverse Regression** | Abdul-Nasah Soale, Yuexiao Dong (2021) | *Journal of Statistical Planning and Inference* 214: 88–101 (2021)<br>[DOI: 10.1016/j.jspi.2020.12.006](https://doi.org/10.1016/j.jspi.2020.12.006) | Kernel expectile regression + random projection of regression expectiles \citep{soaleExpectileAssistedInverseRegression2021} | Big Mac data, simulations |
| 10 | **Slicing-free Inverse Regression in High-dimensional SDR** | Qing Mai, Xiaofeng Shao, Runmin Wang, Xin Zhang (2025) | *Statistica Sinica* 35(1): 1–23 (2025)<br>[DOI: 10.5705/ss.202022.0112](https://doi.org/10.5705/ss.202022.0112) | Penalized eigen-decomposition of Martingale Difference Divergence Matrix (MDDM) \citep{maiSlicingFreeInverseRegression2025} | High-dimensional multivariate regression |
| 11 | **Tail Inverse Regression (TIREX)** | Anass Aghbalou, François Portier, Anne Sabourin, Chen Zhou (2024) | *Bernoulli* 30(4): 2977–3008 (2024)<br>[DOI: 10.3150/23-BEJ1686](https://doi.org/10.3150/23-BEJ1686) | Extreme SDR subspace under **Tail Conditional Independence** for high-threshold exceedances \citep{aghbalouTailInverseRegression2024} | Extreme value / tail risk benchmarks |
| 12 | **Hellinger Correlation SDR** | Seungbeom Hong, Ilmun Kim, Jun Song (2024) | *ICML 2024* (PMLR 235: 18634–18647)<br>[mlr.press](https://proceedings.mlr.press/v235/hong24a.html) | Single-index central subspace detection via Hellinger correlation \citep{hongEnhancingSufficientDimension2024} | Benchmark synthetic datasets |
| 13 | **On Conditional Stochastic Interpolation for Generative Nonlinear SDR** (GenSDR) | Shuntuo Xu, Zhou Yu, Jian Huang (2025) | [arXiv:2512.18971](https://arxiv.org/abs/2512.18971) (Preprint) | Exhaustive recovery of the nonlinear central $\sigma$-field via conditional stochastic interpolation \citep{xuConditionalStochasticInterpolation2025} | Euclidean and [[non_euclidean_random_objects_sdr|non-Euclidean responses]]; [[PyTorch]] |
| 14 | **On Relative Universality, Regression Operator, and Conditional Independence** | [[SDR Bing Li|Bing Li]], Ben Jones, Andreas Artemiou (2025) | [arXiv:2504.11044](https://arxiv.org/abs/2504.11044) (Preprint) | $\epsilon$-measurable relative universality enabling RKHS regression operators to characterize $Y \perp\!\!\perp \mathbf{X} \mid G(\mathbf{X})$ \citep{liRelativeUniversalityRegression2025} | Theoretical foundation for [[A General Theory for Nonlinear SDR 2013|nonlinear SDR]] and CI testing |
| 15 | **Deep Transfer Learning: Model Framework and Error Analysis** | Yuling Jiao, Huazhen Lin, Yuchen Luo, Jerry Zhijian Yang (2024) | [arXiv:2410.09383](https://arxiv.org/abs/2410.09383) (Preprint) | Automatic disentanglement of shared vs. domain-specific deep representations with minimax error rates \citep{jiaoDeepTransferLearning2024} | Multi-domain regression and image classification |
| 16 | **Deep Sufficient Representation Learning via Mutual Information** (MSRL) | Siming Zheng, Yuanyuan Lin, Jian Huang (2022) | [arXiv:2207.10772](https://arxiv.org/abs/2207.10772) (Preprint) | Variational mutual information maximization with order-two U-process convergence bounds \citep{zhengDeepSufficientRepresentation2022} | Continuous and categorical multivariate responses |
| 17 | **MEASURE: Multi-scale Minimal Sufficient Representation Learning** | Sangmin Jo et al. (2025) | [arXiv:2510.12070](https://arxiv.org/abs/2510.12070) & [OpenReview](https://openreview.net/forum?id=Ww599CnVnU) | Domain-invariant minimal sufficient representation via contrastive alignment \citep{joMEASUREMultiScaleMinimal2025} | SleepEDF-20, MASS |

---

## 2. Technical Analysis of Flagship Mechanisms

### 2.1 TESR: Transfer Learning through Enhanced Sufficient Representation \citep{geTransferLearningEnhanced2025}
TESR directly addresses transferring representations from $S$ source domains $\mathcal{D}_s = \{\mathbf{X}_s, Y_s\}_{s=1}^S$ (pooled size $N = \sum_{s=1}^S n_s$) to a small-sample target domain $\mathcal{D}_0 = \{\mathbf{X}_0, Y_0\}$ ($n_0 \ll N$), and is the exact paper containing the **GTEx gene expression** (Table 2, Figure 6) and **PACS image** (Table 3, Figure 7) benchmarks discussed in the [[Oct 09 2026 SDR|October 9 call]].

\paragraph{Stage I (Source Sufficient & Invariant Representation — SIRep).}
TESR estimates a common neural encoder $R_c : \mathbb{R}^d \to \mathbb{R}^{r_c}$ satisfying both sufficiency across all sources and domain invariance against the domain indicator $Z \in \{1, \dots, S\}$ \citep{geTransferLearningEnhanced2025}:
\[
Y_s \perp\!\!\perp \mathbf{X}_s \mid R_c(\mathbf{X}_s) \quad \text{and} \quad R_c(\mathbf{X}_{\text{pool}}) \perp\!\!\perp Z, \quad s = 1, \dots, S
\]
Rather than relying on adversarial GAN discriminators as in [[DEEP NONLINEAR SUFFICIENT DIMENSION REDUCTION 2024|DDR]] \citep{huang2024deep}, TESR enforces Gaussianity $R_c(\mathbf{X}_s) \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{r_c})$ via the closed-form U-statistic **Energy Distance** $\mathbb{D}_n(\cdot \parallel \gamma_{r_c})$ \citep{rizzoEnergyDistance2016} alongside unbiased fourth-order U-statistic **Distance Covariance** $\mathbb{V}_n(\cdot, \cdot)$ \citep{szekelyMeasuringTestingDependence2007, huoFastComputingDistance2016}:
\[
\mathcal{L}_{S,n}(R) = \sum_{s=1}^S \left\{ -\mathbb{V}_n\left(R(\mathbf{X}_s), Y_s\right) + \lambda_E \mathbb{D}_n\left(R(\mathbf{X}_s) \parallel \gamma_{r_c}\right) \right\} + \lambda_Z \mathbb{V}_n\left(R(\mathbf{X}_{\text{pool}}), Z\right)
\]

\paragraph{Stage II (Target Residual Enhancement).}
Holding the source encoder $\widehat{R}_c$ frozen on $\mathcal{D}_0$, TESR trains an augmenting target network $R_t : \mathbb{R}^d \to \mathbb{R}^{r_t}$ to capture target-specific information absent from $\widehat{R}_c$:
\[
Y_0 \perp\!\!\perp \mathbf{X}_0 \mid \left[ R_c(\mathbf{X}_0), R_t(\mathbf{X}_0) \right] \quad \text{subject to} \quad R_c(\mathbf{X}_0) \perp\!\!\perp R_t(\mathbf{X}_0)
\]
via the target empirical objective \citep{geTransferLearningEnhanced2025}:
\[
\mathcal{L}_{T,n}(R, \widehat{R}_c) = -\mathbb{V}_n\left(\left[R(\mathbf{X}_0), \widehat{R}_c(\mathbf{X}_0)\right], Y_0\right) + \lambda_C \mathbb{V}_n\left(R(\mathbf{X}_0), \widehat{R}_c(\mathbf{X}_0)\right) + \lambda_{E,0} \mathbb{D}_n\left(R(\mathbf{X}_0) \parallel \gamma_{r_t}\right)
\]

---

### 2.2 FlowSDR & Neural Gaussian SDR \citep{dongFlowSDRSufficientDimension2026}
FlowSDR and Neural Gaussian SDR establish a forward likelihood-based framework that connects [[Sufficient Dimension Reduction (SDR)]] directly to conditional variance estimation, tail behavior, and **conformal prediction intervals**.

\paragraph{Neural Gaussian SDR vs. MAVE.}
Neural Gaussian SDR models $Y \mid \mathbf{B}^\top \mathbf{X} = \mathbf{w} \sim \mathcal{N}(\mu_\gamma(\mathbf{w}), \sigma_\gamma^2(\mathbf{w}))$ with a shared neural network. Proposition C.1 of \citet{dongFlowSDRSufficientDimension2026} proves that profiling out $(\mu, \sigma^2)$ yields the population target:
\[
\mathbf{B}^{\mathrm{G}} = \operatorname*{argmin}_{\mathbf{B} \in \mathcal{V}_{p,d}} \operatorname{E}\left\{ \log \operatorname{Var}(Y \mid \mathbf{B}^\top \mathbf{X}) \right\}
\]
Whereas classical [[MAVE-Gradient-Estimation-2011-12-01|MAVE]] minimizes $\operatorname{E}\{\operatorname{Var}(Y \mid \mathbf{B}^\top \mathbf{X})\}$ (targeting only the [[SDR central mean subspace|central mean subspace]]) and fails when a direction $\boldsymbol{\beta}_2$ enters purely through the conditional variance, Neural Gaussian SDR recovers both mean and variance directions \citep{dongFlowSDRSufficientDimension2026}.

\paragraph{FlowSDR for Heavy Tails & Pure Tail Shifts.}
When the conditional distribution $Y \mid \mathbf{X}$ exhibits regimes with identical mean ($0$) and identical variance ($1$) that differ solely in tail heaviness (Model IV of \citet{dongFlowSDRSufficientDimension2026}: $\mathcal{N}(0,1)$ for $\boldsymbol{\beta}_5^\top \mathbf{X} > 0$ vs. $t_3/\sqrt{3}$ for $\boldsymbol{\beta}_5^\top \mathbf{X} \le 0$), both MAVE and Neural Gaussian SDR fail to recover $\boldsymbol{\beta}_5$. By parameterizing $Z = T_{\boldsymbol{\theta}}(Y; \mathbf{B}^\top \mathbf{X}) \sim \mathcal{N}(0,1)$ via monotone rational-quadratic spline normalizing flows \citep{durkanNeuralSplineFlows2019}, FlowSDR maximizes the mutual-information-equivalent index $\mathcal{J}(\mathbf{B}) = \operatorname{E}\{\log f(Y \mid \mathbf{B}^\top \mathbf{X})\}$ and consistently recovers pure tail directions \citep{dongFlowSDRSufficientDimension2026}.

\paragraph{Conformal Prediction Intervals on Reduced Subspaces.}
Using a calibration set $\{(\mathbf{X}_i, Y_i)\}_{i=1}^{n_{\text{cal}}}$, \citet{dongFlowSDRSufficientDimension2026} construct distribution-free conformal prediction intervals over the reduced subspace $\widehat{\mathbf{B}}^\top \mathbf{X}$:
* **Heteroscedastic Gaussian Score**: $R_i = |Y_i - \hat{\mu}(\widehat{\mathbf{B}}^\top \mathbf{X}_i)| / \hat{\sigma}(\widehat{\mathbf{B}}^\top \mathbf{X}_i)$.
* **Flow Quantile Score**: $R_i = \max\{\hat{Q}_{\alpha/2}(\widehat{\mathbf{B}}^\top \mathbf{X}_i) - Y_i, Y_i - \hat{Q}_{1-\alpha/2}(\widehat{\mathbf{B}}^\top \mathbf{X}_i), 0\}$.

---

## 3. Identifying the Exact Research Gap for Project 2

Synthesizing TESR \citep{geTransferLearningEnhanced2025}, FlowSDR \citep{dongFlowSDRSufficientDimension2026}, and RKHS Conditional Independence Testing \citep{tangReproducingKernelBasedNonparametric2026, liRelativeUniversalityRegression2025} isolates two open methodological gaps that directly align with the [[Oct 09 2026 SDR|October 9 call]]:

1. **Formal Sufficiency Auditing & Structural Dimension Selection ($r_t$) in Transfer SDR**:
   * TESR \citep{geTransferLearningEnhanced2025} assumes a fixed, ad-hoc dimension $r_t = 32$ or $64$ for the target augmentation $R_t(\mathbf{X}_0)$ and always trains $R_t$ without testing whether $\widehat{R}_c(\mathbf{X}_0)$ is *already* statistically sufficient ($H_0: Y_0 \perp\!\!\perp \mathbf{X}_0 \mid \widehat{R}_c(\mathbf{X}_0)$). In Section 7, \citet{geTransferLearningEnhanced2025} explicitly identify selecting the dimensions of $R_c$ and $R_t$ and exploring alternative conditional independence measures as open problems.
   * **Project 2 Mechanism**: Develop a formal **Sufficiency Audit Criterion** on $\mathcal{D}_0$ to test $H_0: Y_0 \perp\!\!\perp \mathbf{X}_0 \mid \widehat{R}_c(\mathbf{X}_0)$ and sequentially determine the minimal residual dimension $r_t = \dim(R_t)$ needed to restore sufficiency.
2. **Auditing Variance & Tail Sufficiency for Reliable Uncertainty Quantification**:
   * TESR evaluates transferred representations exclusively via point-prediction MSE or accuracy \citep{geTransferLearningEnhanced2025}. However, a pretrained representation $\widehat{R}_c(\mathbf{X}_0)$ may transfer point-prediction mean structure $\operatorname{E}(Y_0 \mid \mathbf{X}_0)$ while missing target-specific heteroscedastic variance directions $\mathcal{S}_{\operatorname{Var}(Y_0 \mid \mathbf{X}_0)}$ \citep{cookDimensionReductionConditional2002} or tail behavior \citep{aghbalouTailInverseRegression2021, dongFlowSDRSufficientDimension2026}.
   * **Project 2 Mechanism**: Audit and repair pretrained representations with respect to **conditional variance and conditional quantile/flow sufficiency**, yielding calibrated conformal prediction intervals on the target domain without full model retraining.

---

## Suggested New Notes
- `key_terms_TESR_FlowSDR` — Prerequisite mathematical terms for TESR and FlowSDR (created alongside this review).
- `Conjoined Conditional Covariance Operator (CCCO)` — RKHS operator test for conditional independence \citep{tangReproducingKernelBasedNonparametric2026}.
- `Tail Conditional Independence (TIREX)` — Extreme value dimension reduction framework \citep{aghbalouTailInverseRegression2021}.
