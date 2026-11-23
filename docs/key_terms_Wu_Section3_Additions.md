# Prerequisite Key Terms: Dr. Wu's Section 3 (NN-SDR) Additions

This note catalogs the prerequisite mathematical definitions, paper-specific notation, and conceptual motivations underlying the papers added to Section 3 (NN-SDR) of the [[SDR and Neural Networks|SDR-NN Working Paper]]. Cross-reference [[sdr_papers_summary]], [[pdf_reference_archive]], and [[key_terms_TESR_FlowSDR]] (which covers FlowSDR and TESR).

---

## 1. Classification-Ensemble & Spectral Order Determination (`DPSVM`, Chen, Liu, and Qiu, 2025)

### Central Class ($\mathcal{M}_{Y|X}$) & Unbiased Nonlinear SDR Function
* **Mathematical Definition**: Assuming $\{P_{X|Y}(\cdot \mid y) : y \in \Omega\}$ is dominated by a $\sigma$-finite measure, there exists a unique minimal sufficient $\sigma$-field (the [[Central-Sigma-Field-And-Class-2013-01-01|central $\sigma$-field]]) $\mathcal{G}_{Y|X} = \sigma(f_0(X))$ such that $Y \perp\!\!\perp X \mid \mathcal{G}_{Y|X}$ (Chen, Liu, and Qiu, 2025, Sec. 2.1; [[A General Theory for Nonlinear SDR 2013|Lee, Li, and Chiaromonte, 2013]]). The associated central class $\mathcal{M}_{Y|X} \subset L^2(P_X)$ is the space of all square-integrable, $\mathcal{G}_{Y|X}$-measurable functions. Any $f(X) \in \mathcal{M}_{Y|X}$ is called an unbiased nonlinear SDR function (Chen, Liu, and Qiu, 2025, Definition 2.1).
* **Notation Used**: $\mathcal{G}_{Y|X}$, $\mathcal{M}_{Y|X}$, $L^2(P_X)$, true standardized generators $f_0(X)$ satisfying $E[f_0(X)] = 0$ and $\operatorname{Var}(f_0(X)) = I_d$.
* **Why It Matters**: Theorem 4.1 of Chen, Liu, and Qiu (2025) proves that minimizing the population Deep PSVM loss over $L^2(P_X) \times \mathbb{R}$ produces minimizers $f^*(X) \in \mathcal{M}_{Y|X}$ ($\sigma(f^*(X)) \subseteq \mathcal{G}_{Y|X}$), establishing unbiased nonlinear [[Sufficient Dimension Reduction (SDR)|SDR]] without the [[Linearity-Condition-LCM-Assumption-2024-10-10|linearity condition (LCM)]].

### Left-Versus-Right (LVR) and One-Versus-Another (OVA) Slicing
* **Mathematical Definition**: Schemes that discretize $Y$ into $l$ binary response variables $\tilde{Y} \in \{-1, 0, 1\}$ along sample quantiles $q_1 < \cdots < q_{h-1}$ (with $q_0 = \min_i Y_i$, $q_h = \max_i Y_i$): LVR sets $\tilde{Y}_i^r = I(Y_i > q_r) - I(Y_i \le q_r)$ for $1 \le r \le h-1$ ($l = h-1$); OVA pairs distinct slices via $\tilde{Y}_i^{rs} = I(q_{s-1} < Y_i \le q_s) - I(q_{r-1} < Y_i \le q_r)$ for $1 \le r < s \le h$ ($l = h(h-1)/2$) (Chen, Liu, and Qiu, 2025, Sec. 3.1; Li et al., 2011).
* **Notation Used**: $q_r$, $\tilde{Y}_i^r$ (LVR), $\tilde{Y}_i^{rs}$ (OVA), slice count $h$, ensemble Gram matrix $M \in \mathbb{R}^{l \times l}$ with entries $M_{ij} = \sum_{k=1}^n \hat{f}_i(X_k)\hat{f}_j(X_k)$.
* **Why It Matters**: Converts continuous (LVR) or categorical (OVA) regression targets into an ensemble of binary margin problems whose leading eigenfunctions $\sum_j v_{ij}\hat{f}_j$ span the central class (Chen, Liu, and Qiu, 2025, Sec. 3.1).

### Deep Principal Support Vector Machine (DPSVM) Objective
* **Mathematical Definition**: Over a ReLU neural network class $\mathcal{F}_n$ and threshold $t \in \mathbb{R}$, DPSVM solves
  $$L_n(f, t) = \lambda \operatorname{Var}_n(f(X)) + \lambda t^2 + \frac{1}{n}\sum_{i=1}^n \rho\!\left(-\tilde{Y}_i\left(f(X_i) - E_n(f(X)) - t\right)\right),$$
  where $\lambda > 0$, $\operatorname{Var}_n$ and $E_n$ denote empirical variance and mean, and $\rho$ is any nondecreasing convex Lipschitz loss such as hinge loss $\rho(u) = \max\{1+u, 0\}$ (Chen, Liu, and Qiu, 2025, Eq. (6)).
* **Notation Used**: $\mathcal{F}_n$, $L_n(f, t)$, $E_n(\cdot)$, $\operatorname{Var}_n(\cdot)$, hinge/margin loss $\rho$.
* **Why It Matters**: Replaces $O(n^3)$ [[RKHS]] kernel matrix inversion with scalable first-order optimization in [[PyTorch]], accommodates complex non-vector inputs such as images, and gains outlier robustness from Lipschitz margin losses (Chen, Liu, and Qiu, 2025, Sec. 3.1 & 4).

### Ladle Order Determination Estimator
* **Mathematical Definition**: Structural dimension estimator $\hat{d} = \arg\min_{k=0,\dots,r_l} \{f_n(k) + g_n(k)\}$ combining normalized bootstrap eigenvector dispersion
  $$f_n^0(k) = \begin{cases} 0, & k = 0, \\ n^{-1}\sum_{i=1}^n \left\{1 - |\det(B_k^\top B_{k,i}^*)|\right\}, & k \ge 1, \end{cases} \quad f_n(k) = \frac{f_n^0(k)}{1 + \sum_{j=0}^{r_l} f_n^0(j)},$$
  and normalized squared eigenvalue tail $g_n(k) = \lambda_{k+1}^2 / (1 + \sum_{j=0}^{r_l}\lambda_{j+1}^2)$, where $B_k = (v_1,\dots,v_k)$ holds the top $k$ eigenvectors of $M$, $B_{k,i}^*$ are bootstrap replicates, and $r_l = l-1$ ($l \le 10$) or $\lfloor l/\log l \rfloor$ ($l > 10$) (Chen, Liu, and Qiu, 2025, Sec. 3.2; Luo & Li, 2016).
* **Notation Used**: $\hat{d}$, $f_n^0(k)$, $f_n(k)$, $g_n(k)$, $B_k$, $B_{k,i}^*$, $\lambda_k$, cutoff $r_l$.
* **Why It Matters**: Exploits both eigenvalue decay and eigenvector stability so DPSVM estimates the structural dimension $d$ automatically without pre-specification or Gaussian null asymptotics.

---

## 2. Optimal Transport & Generative Distributional SDR (`SDR-COT`, Zeng & Bura, 2026; `Belted Engression`, Tan, Li, & Xue, 2026a)

### Triangular Conditional Optimal Transport (COT) Map
* **Mathematical Definition**: Given covariate marginal $\eta = P_X$ and independent reference response $Y_0 \sim \rho \in \mathcal{P}_2^r(\mathcal{Y})$ on separable Hilbert spaces $\mathcal{X}, \mathcal{Y}$, the static quadratic COT problem seeks a triangular map $T^\star(x, y) = (x, T_Y^\star(x, y))$ pushing $\mu_0 = \eta \otimes \rho$ to the joint law $\mu_1 = P_{(X,Y_1)}$ such that $T_Y^\star(x, \cdot)$ is the $\rho$-a.e. unique quadratic Monge map from $\rho$ to the regular conditional measure $\mu_1(\cdot \mid x)$ for $\eta$-a.e. $x$ (Zeng and Bura, 2026, Sec. 1 & Theorem 3.1).
* **Notation Used**: $T^\star = (\mathrm{id}, T_Y^\star)$, $\mu_0 = \eta \otimes \rho$, $\mu_1 = P_{(X,Y_1)}$, regular conditional measure $\mu_1(\cdot \mid x) = P_{Y_1|X}(\cdot \mid x)$, Wasserstein-2 Monge interpolation $T_t^\star$.
* **Why It Matters**: Zeng and Bura (2026, Proposition 4.4) prove that $Y_1 \perp\!\!\perp X \mid R(X)$ forces both the terminal response transport $T_Y^\star(x, y) = G_Y^\star(R(x), y)$ and the Monge current-state velocity field $v_t^\star$ on $[0, 1-\tau]$ to depend on $x$ exclusively through $R(x)$, reducing COT to a regression-style flow-matching objective.

### Anisotropic Relaxed COT Cost ($c_\varepsilon$)
* **Mathematical Definition**: Because unobserved conditional couplings under continuous covariates $X$ cannot be directly paired in finite samples, SDR-COT solves an ordinary empirical optimal transport problem between $\hat{\mu}_0 = n^{-1}\sum_{i=1}^n \delta_{(X_i, Y_{0,i})}$ and $\hat{\mu}_1 = n^{-1}\sum_{i=1}^n \delta_{(X_i, Y_{1,i})}$ under the anisotropic quadratic cost
  $$c_\varepsilon(x, y, \tilde{x}, \tilde{y}) := \|x - \tilde{x}\|_{\mathcal{X}}^2 + \varepsilon \|y - \tilde{y}\|_{\mathcal{Y}}^2, \qquad \varepsilon > 0,$$
  and trains a conditional velocity field $u(t, R(x), Y_t)$ via flow matching on $t \sim \operatorname{Unif}[0, 1-\tau]$ (Zeng and Bura, 2026, Sec. 5.2).
* **Notation Used**: $c_\varepsilon$, empirical relaxed coupling $\hat{\gamma}_\varepsilon^\star$, truncation $\tau > 0$.
* **Why It Matters**: For small $\varepsilon_n \downarrow 0$, $\hat{\gamma}_{\varepsilon_n}^\star \to \gamma^\star$ in $W_2$ (Proposition 6.1), yielding consistent linear SDR in both Euclidean spaces (via Caffarelli contraction bounds) and infinite-dimensional Hilbert spaces (via Cameron–Martin Sobolev regularity and uniqueness of the Gaussian continuity equation; Zeng and Bura, 2026, Sec. 6).

### Compress-then-Generate Law Factorization & Energy Score (`Belted Engression`)
* **Mathematical Definition**: Under compact bottleneck space $\Omega_S = [0,1]^d$ and auxiliary noise $\eta \sim \operatorname{Unif}([0,1]^q)$ independent of $(X, Y)$, Proposition 1 of Tan, Li, and Xue (2026a, Sec. 3.1) proves that $Y \perp\!\!\perp X \mid s^*(X)$ is equivalent to the existence of $h^* : [0,1]^d \times [0,1]^q \to \Omega_Y$ such that the conditional law of $h^*(s^*(x), \eta)$ equals $P_{Y|X}(\cdot \mid x)$ for $P_X$-a.e. $x$. Belted Engression learns $(s, h)$ jointly end-to-end by minimizing the expected energy score loss of $g(X, \eta) = h(s(X), \eta)$:
  $$L(g) = E\!\left[\|Y - g(X, \eta)\| - \frac{1}{2}\|g(X, \eta) - g(X, \eta')\|\right], \qquad \eta, \eta' \stackrel{\text{i.i.d.}}{\sim} P_\eta$$
  (Tan, Li, and Xue, 2026a, Eq. (2)).
* **Notation Used**: Sufficient predictor "belt" $s : \Omega_X \to [0,1]^d$, reduced generator $h : [0,1]^d \times \Omega_\eta \to \Omega_Y$, conditional generated law $P_{s,h}(\cdot \mid x)$, energy score $\operatorname{ES}(P_g(\cdot \mid x), y)$.
* **Why It Matters**: Unlike [[Belted and Ensembled Neural Network for Linear and Nonlinear SDR|BENN]], which constructs an ensemble of response transformations, Belted Engression couples an SDR belt with a single conditional generative decoder under an energy-score criterion, yielding sharper finite-sample rates and asymptotically fewer network parameters than direct Engression (Tan, Li, and Xue, 2026a, Theorems 2–3).

---

## 3. Mutual Information & Variational Contrastive Bounds (`MSRL`, Zheng, Lin, and Huang, 2022)

### Uniformized Sufficient Representation (USR) & Intrinsic Dimension ($d_0$)
* **Mathematical Definition**: A representation $R_0 : \mathcal{X} \to [0,1]^{d_0}$ satisfying $Y \perp\!\!\perp X \mid R_0(X)$ and $R_0(X) \sim \gamma_U \equiv \operatorname{Uniform}[0,1]^{d_0}$, where $d_0 := \min\{\dim(R) : R \text{ is a USR}\}$ is the intrinsic dimension (Zheng, Lin, and Huang, 2022, Eq. (2.3)).
* **Notation Used**: $R_0$, reference law $\gamma_U = \operatorname{Uniform}[0,1]^{d_0}$, minimal dimension $d_0$.
* **Why It Matters**: By optimal transport onto $[0,1]^{d_0}$ (Lemma C.1), any absolutely continuous sufficient representation can be reparameterized into a canonical USR, making the target distribution identifiable while preserving sufficiency.

### Variational Fenchel-Legendre $f$-Divergence Objective
* **Mathematical Definition**: MSRL estimates $R_0$ by minimizing
  $$L(R; \lambda) := -I_{\operatorname{KL}}(Y; R(X)) + \lambda D_{\operatorname{KL}}(P_R \| \gamma_U), \qquad \lambda \ge 0,$$
  rewritten via Lemma 2.2 (Nguyen, Wainwright, and Jordan, 2010) using the Fenchel conjugate $f^*(t) = e^{t-1}$ of $f(t)=t\log t$:
  $$D_{\operatorname{KL}}(P_1 \| P_2) = \sup_{D : \mathbb{R}^d \to \mathbb{R}} \left\{E_{P_1}[D(Z)] - E_{P_2}[e^{D(Z)}] + 1\right\}$$
  applied jointly to discriminator $D(Y, R(X))$ (for $I_{\operatorname{KL}}(Y; R(X))$) and marginal regularizer $Q(R(X))$ (for $D_{\operatorname{KL}}(P_R \| \gamma_U)$) (Zheng, Lin, and Huang, 2022, Eqs. (2.4)–(2.7)).
* **Notation Used**: $L(R; \lambda)$, $I_{\operatorname{KL}}$, $D_{\operatorname{KL}}$, Fenchel conjugate $f^*$, critics $D : \mathbb{R}^{d_Y + d_0} \to \mathbb{R}$ and $Q : \mathbb{R}^{d_0} \to \mathbb{R}$.
* **Why It Matters**: Avoids response slicing or parametric density specification for multi-dimensional continuous or categorical responses while supporting non-asymptotic estimation error bounds via a generalized Dudley inequality for order-two U-processes indexed by neural networks (Zheng, Lin, and Huang, 2022, Sec. 1–2).

---

## 4. Architecture-Embedded Linear Bottlenecks & Sparse Selection (`GRNN-SDR`, Yang & Yang, 2025; `DeepIn`, Tan, Liu, & Lin, 2026b)

### Barron Class & Golden-Ratio Validation Criterion (`GRNN-SDR`)
* **Mathematical Definition**: For regression targets $f(x) = g(\beta^\top x)$ ($\beta \in \mathbb{R}^{p \times d}$, $\operatorname{rank}(\beta) = d$) where $g$ has finite first absolute moment $C_g < \infty$ of its Fourier magnitude distribution (Barron class), Yang and Yang (2025, Sec. 2.3–2.5) parameterize a two-hidden-layer neural network $f_{k,m}(x, \theta, w_k) = \sum_{i=1}^m \tau_i \phi(u_i^\top I(w_k^\top x) + v_i) + \tau_0$ with a $k$-unit linear-activation first hidden layer $w_k \in \mathbb{R}^{p \times k}$ and $m$ sigmoidal second-layer units. Candidate widths $k \in [1, p]$ are searched via golden-section probes $k_1 = m_0 + \lfloor 0.382(n_0 - m_0)\rfloor$ and $k_2 = m_0 + \lfloor 0.618(n_0 - m_0)\rfloor$ to minimize
  $$CR(k) = \operatorname{MSE}_{\text{va}}(k) + k \cdot \operatorname{pen}(N, n_{\text{va}}),$$
  where $\operatorname{MSE}_{\text{va}}(k)$ is validation MSE under training size $N$ and validation size $n_{\text{va}}$ (Yang and Yang, 2025, Eqs. (27)–(30)).
* **Notation Used**: $w_k \in \mathbb{R}^{p\times k}$, $C_g$, $CR(k)$, penalty $\operatorname{pen}(N, n_{\text{va}})$ satisfying $((N^{-1}\log N)^{1/2} + n_{\text{va}}^{-1/2}) \ll \operatorname{pen}(N, n_{\text{va}}) \ll c_N^2$.
* **Why It Matters**: Jointly estimates the [[SDR central mean subspace|central mean subspace]] and proves consistent structural dimension selection $P(\hat{d}=d)\to 1$ (Theorems 2, 3, and 5) while cutting grid search from $p$ widths down to at most $\lceil 1.44(\log_2 p - 2)\rceil + 3$ trainings.

### RePU Networks & Group-LASSO Representation Matrix (`DeepIn`)
* **Mathematical Definition**: Formulates $Y \sim g_\theta(\mathbf{B}X)$ with a square matrix $\mathbf{B} \in \mathbb{R}^{d \times d}$ (spanning standard DNNs when $\mathbf{B} = \mathbf{I}_d$) and Rectified Power Unit (RePU) activations $\sigma(x) = \{\max(x, 0)\}^p$ ($p \ge 2$), fitted via penalized empirical risk:
  $$R_n(g_\theta, \mathbf{B}) = \frac{1}{n}\sum_{i=1}^n \tilde{L}\!\left(g_\theta(\mathbf{B}X_i), Y_i\right) + \lambda_1 \sum_{k=1}^d \|\mathbf{B}_{[k,]}\|_2 + \lambda_2 \sum_{k=1}^d \|\mathbf{B}_{[,k]}\|_2 + \lambda_3 \|\theta\|_1$$
  (Tan, Liu, and Lin, 2026b, Sec. 2 Eq. (2)).
* **Notation Used**: Representation matrix $\mathbf{B} \in \mathbb{R}^{d\times d}$ whose nonzero rows $\mathbf{B}_{[k,]}$ count the Minimal Representation Dimension (MRD, $d_0$) and whose nonzero columns $\mathbf{B}_{[,k]}$ select the $s_0$ active variables; network parameters $\theta$; minimal total parameter count $T_0 = S_0^{\text{net}} + d_0 s_0$.
* **Why It Matters**: Simultaneously learns minimal sufficient linear features, performs variable selection, and prunes subnetwork complexity with selection consistency and optimal $d_0$-adaptive non-asymptotic rates, while RePU smoothness ($p \ge 2$) enables derivative-based statistical hypothesis testing on both variables and representations (Tan, Liu, and Lin, 2026b, Sec. 1–4).

---

## Suggested New Notes
* `Deep Principal Support Vector Machines (DPSVM)` — Dedicated paper observation note covering LVR/OVA experiments and image applications in Chen, Liu, and Qiu (2025).
* `SDR-COT (Conditional Optimal Transport)` — Dedicated summary of Hilbert-space COT factorization, Caffarelli bounds, and functional regression in Zeng and Bura (2026).
* `Belted Engression` — Note contrasting energy-score generative SDR (Tan, Li, and Xue, 2026a) against indicator-ensemble [[Belted and Ensembled Neural Network for Linear and Nonlinear SDR|BENN]].
* `GRNN-SDR (Golden Ratio SDR)` — Note on Barron-class approximation bounds and golden-section order selection in Yang and Yang (2025).
* `DeepIn (Minimal Sufficient Self-Interpretable DNNs)` — Note on MRD selection consistency and representation hypothesis testing in Tan, Liu, and Lin (2026b).
* `MSRL (Mutual Information SDR)` — Note on uniformized sufficient representations and U-process Dudley bounds in Zheng, Lin, and Huang (2022).
