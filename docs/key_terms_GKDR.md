[[paper_obs_GKDR]] | [[RKHS]] | [[SDR central mean subspace]] | [[SDR references]]

# Key Terms: GKDR (Gradient-based Kernel Dimension Reduction)

> [!ABSTRACT] Purpose
> Prerequisite key terms whose definitions must be understood before the core contributions of Fukumizu & Leng (2014) can be grasped. Organized by conceptual category.

---

## Category 1: RKHS Foundations

### Positive Definite Kernel
- **Definition**: A symmetric function $k: \Omega \times \Omega \to \mathbb{R}$ satisfying $\sum_{i,j=1}^n c_i c_j k(x_i, x_j) \geq 0$ for all $n$, all $x_1, \ldots, x_n \in \Omega$, and all $c_1, \ldots, c_n \in \mathbb{R}$.
- **Notation in paper**: $k_X$ (on predictor space $\mathcal{X}$), $k_Y$ (on response space $\mathcal{Y}$).
- **Why it matters**: The entire GKDR framework rests on the reproducing property of the associated RKHS. The choice of kernel (typically Gaussian) determines what functions are representable and what smoothness conditions are implicitly imposed.

### Reproducing Kernel Hilbert Space (RKHS)
- **Definition**: The unique Hilbert space $\mathcal{H}$ of functions on $\Omega$ such that (i) $k(\cdot, x) \in \mathcal{H}$, (ii) $\operatorname{span}\{k(\cdot, x) : x \in \Omega\}$ is dense in $\mathcal{H}$, and (iii) $\langle f, k(\cdot, x) \rangle_{\mathcal{H}} = f(x)$ (reproducing property).
- **Notation in paper**: $\mathcal{H}_X$ (RKHS of $k_X$), $\mathcal{H}_Y$ (RKHS of $k_Y$).
- **Why it matters**: The reproducing property (iii) is the mechanism that converts function evaluation into inner products, enabling gradient computation via Eq. (5): $\frac{\partial f}{\partial x} = \left\langle f, \frac{\partial k(\cdot, x)}{\partial x} \right\rangle$.

---

## Category 2: Cross-Covariance Operators

### Cross-Covariance Operator $C_{YX}$
- **Definition**: The operator $C_{YX}: \mathcal{H}_X \to \mathcal{H}_Y$ defined by $\langle g, C_{YX} f \rangle_{\mathcal{H}_Y} = E[f(X) g(Y)]$ for all $f \in \mathcal{H}_X$, $g \in \mathcal{H}_Y$.
- **Notation in paper**: $C_{YX}$, $C_{XY}$, $C_{XX}$ (self-covariance operator on $\mathcal{H}_X$).
- **Why it matters**: $C_{YX}$ is the infinite-dimensional analog of the cross-covariance matrix $\operatorname{Cov}(\Phi_Y(Y), \Phi_X(X))$. The conditional expectation formula $E[g(Y) \mid X = \cdot] = C_{XX}^{-1} C_{XY} g$ (Theorem 1) is the foundation of GKDR — it replaces nonparametric local smoothing with algebraic operator inversion.

### Empirical Covariance Operator
- **Definition**: $\hat{C}_{YX}^{(n)} f = \frac{1}{n} \sum_{i=1}^n f(X_i) k_Y(\cdot, Y_i)$.
- **Notation in paper**: $\hat{C}_{YX}^{(n)}$, $\hat{C}_{XX}^{(n)}$.
- **Why it matters**: Estimation is "straightforward" (the paper's word) — no bandwidth, no local neighborhoods. The empirical operator is $\sqrt{n}$-consistent in Hilbert-Schmidt norm. Its matrix representation through Gram matrices ($G_X$, $G_Y$) is what makes the algorithm computationally concrete.

---

## Category 3: Gram Matrices & Regularized Inversion

### Gram Matrix
- **Definition**: For kernel $k$ and data $\{x_1, \ldots, x_n\}$, the $n \times n$ matrix $G = (k(x_i, x_j))_{ij}$.
- **Notation in paper**: $G_X = (k_X(X_i, X_j))_{ij}$, $G_Y = (k_Y(Y_i, Y_j))_{ij}$.
- **Why it matters**: All GKDR computation passes through these matrices. The $O(n^3)$ inversion cost of $G_X$ is the primary scalability bottleneck that motivates low-rank Cholesky approximation (§2.3.2) and, more broadly, the neural SDR paradigm.

### Tikhonov Regularization
- **Definition**: Replacing the ill-conditioned inverse $C_{XX}^{-1}$ with $(C_{XX} + \varepsilon_n I)^{-1}$, where $\varepsilon_n \to 0$ as $n \to \infty$.
- **Notation in paper**: $\varepsilon_n$ (regularization coefficient); optimal rate $\varepsilon_n = n^{-\max\{1/3, 1/(2\beta+2)\}}$.
- **Why it matters**: The covariance operator $C_{XX}$ is compact (thus not boundedly invertible) in infinite-dimensional RKHS. Tikhonov regularization controls the bias-variance trade-off in the inversion. The smoothness parameter $\beta$ governs the convergence rate through the range condition $\frac{\partial k_X(\cdot, x)}{\partial x^a} \in \mathcal{R}(C_{XX}^{\beta+1})$.

---

## Category 4: SDR-Specific Concepts

### Effective Dimension Reduction (EDR) Space
- **Definition**: The column space $\operatorname{span}(B)$ under the model $Y \perp\!\!\!\perp \mathbf{X} \mid B^\top \mathbf{X}$, $B \in \mathbb{R}^{m \times d}$.
- **Notation in paper**: "EDR space", $B$ (projection matrix), $d$ (structural dimension).
- **Why it matters**: GKDR targets the EDR space rather than the [[SDR central mean subspace|central mean subspace]] or [[Central-Subspace-Definition-Properties-2005-06-01|central subspace]] directly. Under the regression model, the EDR space contains the CMS — the paper's consistency result (Theorem 2) is stated for the EDR space.

### Gradient Containment Property
- **Definition**: Under $Y \perp\!\!\!\perp \mathbf{X} \mid B^\top \mathbf{X}$, the gradient $\frac{\partial}{\partial \mathbf{x}} E[Y \mid \mathbf{X} = \mathbf{x}]$ lies in $\operatorname{span}(B)$ for all $\mathbf{x}$.
- **Why it matters**: This is the fundamental identity that makes gradient-based SDR possible. ADE, IADE, and GKDR all exploit this property; they differ only in *how* the gradient is estimated. GKDR's innovation is estimating it through the RKHS reproducing property rather than local polynomial fitting.

---

## Category 5: Algorithmic Variants

### gKDR-i (Iterative Dimensionality Reduction)
- **Definition**: Starting from a larger dimension $d_1 > d$, GKDR is applied iteratively: project to $d_1$, then to $d_2 < d_1$, and so on down to $d$. The final estimator is $\hat{B} = B_\ell \cdots B_2 B_1$.
- **Why it matters**: Addresses the difficulty of accurate nonparametric gradient estimation in high dimensions by progressively reducing dimensionality. Analogous to the iterative approach of IADE.

### gKDR-v (Variation-based Estimator)
- **Definition**: Instead of averaging $M(X_i)$ matrices, average the *projectors* $\hat{B}_i \hat{B}_i^\top$ computed from individual $\hat{M}_n(X_i)$: $\hat{P} = \frac{1}{n}\sum_{i=1}^n \hat{B}_i \hat{B}_i^\top$.
- **Why it matters**: Addresses the rank-$L$ limitation in classification. When $Y$ takes $L$ classes, $G_Y$ has rank $\leq L$, limiting the discoverable subspace dimension. gKDR-v overcomes this by exploiting variation across data points rather than the average.

---

## Suggested New Notes

- `[[ADE Average Derivative Estimates]]`
- `[[Tikhonov Regularization Operator Inversion]]`
- `[[Gradient-based Kernel Dimension Reduction Fukumizu 2014]]`
