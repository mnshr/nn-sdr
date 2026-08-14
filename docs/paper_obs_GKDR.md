[[SDR and Neural Networks]] | [[SDR references]] | [[RKHS]] | [[SDR central mean subspace]]

# Paper Observations: Gradient-based Kernel Dimension Reduction for Supervised Learning

**Citation**: Fukumizu & Leng (2014), *Journal of the American Statistical Association*, 109(505), 299–311.
**PDF**: [Gradient-based Kernel Dimension Reduction Fukumizu 2014.pdf](file:///Users/manish/gh/research/sdr/ref/Gradient-based%20Kernel%20Dimension%20Reduction%20Fukumizu%202014.pdf)
**Key Terms**: [[key_terms_GKDR]]
**Code**: `https://www.ism.ac.jp/~fukumizu/software.html` | [arXiv:1109.0455](https://arxiv.org/abs/1109.0455)

---

## 1. Narrative Placement

- **Role in Review**: **Computational Solver / Transitional Bridge** — GKDR sits at the critical junction between classical matrix-based SDR (SIR, SAVE, MAVE) and the modern neural SDR era. It solves the gradient estimation problem through RKHS operators rather than local smoothing, but introduces $O(n^3)$ Gram matrix scaling that neural methods later circumvent.
- **Core Shift**: Classical gradient-based SDR methods (ADE, IADE) estimate $\nabla E[Y \mid \mathbf{X} = \mathbf{x}]$ through local kernel smoothing, which requires careful bandwidth selection and fails under symmetric link functions ($E[\phi'(B^\top \mathbf{X})] = 0$). GKDR replaces local smoothing with RKHS cross-covariance operators, converting the gradient estimation into a stable spectral problem that handles arbitrary response types $Y$ (multivariate, categorical, structured) without distributional assumptions on $\mathbf{X}$.
- **Manuscript Relevance**: In the review manuscript (`sdr_v8.1.tex`), GKDR belongs to the classical-to-kernel transition discussed in Section 2.5 (Kernel SDR). It also motivates why neural SDR methods needed to arise — the $O(n^3)$ computational barrier and the non-convex optimization of KDR that GKDR partially solved but did not eliminate for large-scale data.

---

## 2. Mathematical Core

### Population Target

GKDR targets the **effective dimension reduction (EDR) space**, which under the model
\[
Y \perp\!\!\!\perp \mathbf{X} \mid B^\top \mathbf{X}, \qquad B \in \mathbb{R}^{m \times d},
\]
coincides with the column space $\operatorname{span}(B)$. Under the regression model $E[Y \mid \mathbf{X} = \mathbf{x}] = \phi(B^\top \mathbf{x})$, the EDR space contains the [[SDR central mean subspace|central mean subspace]] $\mathcal{S}_{E(Y \mid \mathbf{X})}$. GKDR recovers the EDR space through the range of a kernel-based operator rather than through inverse regression or moment conditions.

### Defining Operator: The $M(x)$ Matrix

The core mathematical object is the $m \times m$ symmetric matrix $M(\mathbf{x})$ whose eigenvectors for non-zero eigenvalues lie in the EDR space:
\[
M(\mathbf{x}) = \left\langle \frac{\partial k_X(\cdot, \mathbf{x})}{\partial \mathbf{x}},\; C_{XX}^{-1} C_{XY} C_{YX} C_{XX}^{-1}\; \frac{\partial k_X(\cdot, \mathbf{x})}{\partial \mathbf{x}} \right\rangle_{\mathcal{H}_X}.
\]

The population-level estimand is the expectation $E[M(\mathbf{X})]$, and the EDR space is recovered from the top $d$ eigenvectors of this matrix.

### Empirical Estimator

The finite-sample estimator replaces covariance operators with their empirical counterparts and Tikhonov regularization:
\[
\hat{M}_n(\mathbf{x}) = \nabla k_X(\mathbf{x})^\top (G_X + n\varepsilon_n I)^{-1} G_Y (G_X + n\varepsilon_n I)^{-1} \nabla k_X(\mathbf{x}),
\]
where $G_X = (k_X(X_i, X_j))_{ij}$ and $G_Y = (k_Y(Y_i, Y_j))_{ij}$ are Gram matrices, $\nabla k_X(\mathbf{x}) = \left(\frac{\partial k_X(X_1, \mathbf{x})}{\partial \mathbf{x}}, \ldots, \frac{\partial k_X(X_n, \mathbf{x})}{\partial \mathbf{x}}\right)^\top \in \mathbb{R}^{n \times m}$, and $\varepsilon_n \to 0$ is the regularization parameter.

The averaged estimator used in practice is:
\[
\tilde{M}_n = \frac{1}{n} \sum_{i=1}^n \hat{M}_n(X_i).
\]
The projection matrix $\hat{B}$ is given by the top $d$ eigenvectors of $\tilde{M}_n$.

### Domain of Validity & Key Assumptions

1. **RKHS regularity**: $\frac{\partial k_X(\cdot, \mathbf{x})}{\partial x^a} \in \mathcal{R}(C_{XX}^{\beta+1})$ for some $\beta \geq 0$ (a smoothness condition linking the kernel derivative to the range of the covariance operator).
2. **Conditional expectation in RKHS**: $E[k_Y(\tilde{y}, Y) \mid \mathbf{X} = \cdot] \in \mathcal{H}_X$ for every $\tilde{y} \in \mathcal{Y}$ — this ensures the cross-covariance operator inversion in Theorem 1 is well-defined.
3. **Injectivity**: $C_{XX}$ is injective, which holds when $k_X$ is a continuous kernel and the support of $P_X$ has non-empty interior.
4. **Differentiability**: $k_X(\mathbf{x}, \tilde{\mathbf{x}})$ is continuously differentiable w.r.t.\ both arguments.
5. **No distributional assumptions on $\mathbf{X}$** — unlike [[SDR Sliced inverse regression (SIR)|SIR]] and [[SAVE-Asymptotic-Distribution-2005-06-01|SAVE]], GKDR does not require the [[Linearity-Condition-LCM-Assumption-2024-10-10|linearity condition (LCM)]] or elliptical symmetry.

### Consistency (Theorem 2)

Under the above conditions with $\varepsilon_n = n^{-\max\{1/3, 1/(2\beta+2)\}}$:
\[
\hat{M}_n(\mathbf{x}) - M(\mathbf{x}) = O_p\!\left(n^{-\min\{1/3, (2\beta+1)/(4\beta+4)\}}\right).
\]

---

## 3. Mechanistic Insights & Trade-offs

### Why It Works

GKDR exploits a fundamental identity connecting **kernel derivatives** to the **EDR space**. Under the model $Y \perp\!\!\!\perp \mathbf{X} \mid B^\top \mathbf{X}$, the gradient of any conditional expectation $E[g(Y) \mid \mathbf{X} = \mathbf{x}]$ is confined to $\operatorname{span}(B)$ (Eq.\ 6 in the paper). Classical methods (ADE, [[MAVE-Gradient-Estimation-2011-12-01|IADE]]) estimate this gradient through local polynomial smoothing, which:
- requires a careful bandwidth schedule,
- degrades rapidly in high dimensions,
- fails when the link function has a symmetric derivative ($E[\phi'(B^\top \mathbf{X})] = 0$).

GKDR bypasses these difficulties by expressing the gradient through the reproducing property of the RKHS:
\[
\frac{\partial}{\partial \mathbf{x}} E[g(Y) \mid \mathbf{X} = \mathbf{x}] = \left\langle C_{XX}^{-1} C_{XY} g,\; \frac{\partial k_X(\cdot, \mathbf{x})}{\partial \mathbf{x}} \right\rangle_{\mathcal{H}_X}.
\]
This converts a nonparametric gradient estimation problem into a **linear-algebraic operation on Gram matrices**. The kernel $k_Y$ on the response space simultaneously evaluates all "regression functions" $E[k_Y(\tilde{y}, Y) \mid \mathbf{X} = \mathbf{x}]$ indexed by $\tilde{y}$, which avoids the symmetry failure of ADE.

### Hidden Trade-offs & Friction Points

| Dimension | Trade-off |
| :--- | :--- |
| **Computational cost** | $O(n^3)$ for Gram matrix inversion; low-rank Cholesky approximation reduces this to $O(nmr)$ time and $O(nmr)$ space, but introduces approximation error |
| **Kernel selection** | Results depend on kernel choice; cross-validation with $k$-NN is used but adds computational overhead |
| **Rank limitation in classification** | For $L$-class classification, $G_Y$ has rank $\leq L$, limiting the discoverable subspace dimension — particularly restrictive for binary problems ($d \leq 1$). The gKDR-v variant addresses this through projector averaging but adds further cost |
| **Regularization sensitivity** | The Tikhonov parameter $\varepsilon_n$ controls the bias-variance trade-off in operator inversion; the rate $\varepsilon_n = n^{-\max\{1/3, 1/(2\beta+2)\}}$ depends on the unknown smoothness parameter $\beta$ |
| **No optimization** | Unlike [[Central-Sigma-Field-And-Class-2013-01-01|KDR]], GKDR reduces to an eigenproblem — no iterative non-convex optimization. This is both a strength (stability, reproducibility) and a limitation (no ability to incorporate auxiliary constraints or structured penalties) |

### Comparative Positioning

| Dimension | [[SDR Sliced inverse regression (SIR)|SIR]] / [[SAVE-Asymptotic-Distribution-2005-06-01|SAVE]] | [[MAVE-Gradient-Estimation-2011-12-01|MAVE]] / IADE | KDR | **GKDR** |
| :--- | :--- | :--- | :--- | :--- |
| **Estimator class** | Inverse regression (moment-based) | Forward regression (local smoothing) | Conditional independence (kernel) | Gradient (kernel operator) |
| **Distributional assumptions on $\mathbf{X}$** | Linearity condition / elliptical symmetry | Mild (smoothness only) | None | None |
| **Response type $Y$** | Sliceable (ordinal/continuous) | Continuous | Arbitrary | Arbitrary |
| **Computational complexity** | $O(nH)$ slicing | $O(n^2)$ local smoothing | $O(n^3)$ per gradient step (iterative) | $O(n^3)$ one-shot eigensolve |
| **Symmetry failure** | Immune (inverse regression) | Vulnerable | Immune | Immune |
| **Optimization** | Eigendecomposition | Iterative local optimization | Non-convex gradient descent | **Eigendecomposition (no optimization)** |

### Connection to Neural SDR

GKDR is the **last major non-neural, non-parametric method** in the SDR lineage before the neural era (DVSDR 2018, [[Fusing sufficient dimension reduction with neural networks. 2021|NN-SDR 2021]], [[NeurIPS-2022-nonlinear-sufficient-dimension-reduction-with-a-stochastic-neural-network StoNet|StoNet 2022]]). The $O(n^3)$ Gram matrix bottleneck of GKDR is precisely the computational barrier that neural architectures eliminate through minibatch SGD ($O(n)$ per epoch). However, GKDR's population-level operator identity ($M(\mathbf{x})$ contains the EDR space) provides a cleaner theoretical target than the implicit representations learned by bottleneck networks — a gap that [[Neural Networks perform SDR using rank regularization 2024|Xu & Yu (2024)]] and [[DEEP NONLINEAR SUFFICIENT DIMENSION REDUCTION 2024|Chen et al.\ (2024)]] later closed with explicit population-level guarantees for neural estimators.

---

## 4. Manuscript Integration

### LaTeX Prose Block

```latex
\paragraph{Gradient-based Kernel Dimension Reduction.}
gKDR \citep{fukumizuGradientbasedKernelDimension2014} recovers the EDR space through an eigendecomposition of the averaged matrix $\tilde{M}_n = n^{-1}\sum_{i=1}^n \hat{M}_n(X_i)$, where each $\hat{M}_n(\mathbf{x})$ is constructed from kernel gradient evaluations and regularized inversions of the Gram matrices $G_X$ and $G_Y$.
Unlike gradient-based predecessors such as ADE and IADE, which estimate $\nabla E[Y \mid \mathbf{X} = \mathbf{x}]$ through local polynomial smoothing, gKDR expresses the gradient through the reproducing property of the RKHS, yielding an estimator that is stable without bandwidth tuning and immune to symmetric-link failures.
The method imposes no distributional assumptions on $\mathbf{X}$ and handles arbitrary response types, including multivariate and categorical $Y$.
However, the $O(n^3)$ cost of Gram matrix inversion limits scalability to moderate sample sizes, a bottleneck that neural SDR architectures later circumvent through minibatch optimization.
```

### Summary Table Row

```latex
\textbf{Gradient-based Kernel Dimension Reduction (gKDR)} \citep{fukumizuGradientbasedKernelDimension2014}: 
& EDR Space / Central Mean Subspace
& Eigendecomposition of averaged RKHS gradient matrix $\tilde{M}_n = n^{-1}\sum_{i=1}^n \hat{M}_n(X_i)$; no iterative optimization
& $C_{XX}$ injective; kernel derivative in $\mathcal{R}(C_{XX}^{\beta+1})$; 
  $E[k_Y(\tilde{y},Y) \mid X = \cdot] \in \mathcal{H}_X$ for all $\tilde{y} \in \mathcal{Y}$;
  Gaussian kernel with cross-validated bandwidth
& $O(n^3)$ Gram matrix inversion; rank-$L$ limitation for $L$-class classification;
  kernel and regularization parameter selection via CV
\\ \addlinespace
```

### Open Questions for Manuscript Section 4

1. **GKDR → Neural bridge**: Can the GKDR operator $C_{XX}^{-1} C_{XY} C_{YX} C_{XX}^{-1}$ be approximated by a two-layer neural network with random Fourier features, reducing the Gram matrix cost to $O(nD)$ where $D$ is the number of random features?
2. **Consistency rate comparison**: The $O_p(n^{-1/3})$ rate (best case $\beta = 0$) is slower than the minimax rate achieved by [[DEEP NONLINEAR SUFFICIENT DIMENSION REDUCTION 2024|GMDDNet]]. Does the RKHS smoothness condition ($\beta$ parameter) create an inherent rate ceiling for kernel-based SDR, or is the gap an artifact of the analysis?
3. **Classification rank limitation**: The gKDR-v variant (projector averaging) addresses the rank-$L$ ceiling for classification but lacks formal consistency guarantees. Can the BENN ensemble strategy be interpreted as a neural analog of gKDR-v's projector averaging?

---

## Suggested New Notes

- `[[Gradient-based Kernel Dimension Reduction Fukumizu 2014]]` — Obsidian note for this paper (currently does not exist in vault)
- `[[ADE Average Derivative Estimates]]` — for the Hristache et al.\ (2001) predecessor
- `[[Tikhonov Regularization Operator Inversion]]` — for the regularization technique used in GKDR and KDR
