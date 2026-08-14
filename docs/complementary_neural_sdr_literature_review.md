# Complementary Deep / Neural Sufficient Dimension Reduction Literature Review (2020–2026)

## Overview

This note provides a systematic literature summary of recent research (2020–2026) at the intersection of **Neural Networks / Deep Learning and [[Sufficient Dimension Reduction (SDR)]]**. These papers complement the baseline algorithms tracked in [[sdr_papers_summary|SDR Central Summary]] and [[SDR and Neural Networks|SDR-NN Working Paper]].

The reviewed methods expand neural SDR beyond traditional regression moments to include **Normalizing Flows, Conditional Diffusion / Stochastic Interpolation, Variational Mutual Information, Optimal Transport, and Algorithmic Fairness**.

---

## 📑 Detailed Summaries of Complementary Papers

### 1. FlowSDR: Sufficient Dimension Reduction via Conditional Normalizing Flows
* **ArXiv**: [arXiv:2606.01346](https://arxiv.org/abs/2606.01346v1) (May/June 2026)
* **Authors**: Yuexiao Dong, Kenichiro McAlinn, Edoardo Airoldi, Lei Li
* **Key Innovation**: Replaces indirect moment matching, local kernel regression, and ensemble estimators with an **exact maximum likelihood framework**. Uses monotone rational-quadratic spline flows parameterized by neural networks to directly model the conditional log-likelihood $\log p(Y \mid \mathbf{B}^\top X)$.
* **Methodological Complement**: Unlike [[Neural Networks perform SDR using rank regularization 2024|DRNN]] or [[Fusing sufficient dimension reduction with neural networks. 2021|NN-SDR]] (which target conditional mean regression $\mathbb{E}[Y \mid X]$), FlowSDR models complex non-Gaussian, heavy-tailed, and multimodal conditional distributions.

### 2. GenSDR: Conditional Stochastic Interpolation for Generative Nonlinear SDR
* **ArXiv**: [arXiv:2512.18971](https://arxiv.org/abs/2512.18971v1) (Dec 2025)
* **Authors**: Shuntuo Xu, Zhou Yu, Jian Huang
* **Key Innovation**: Leverages **Conditional Stochastic Interpolation (Diffusion Matching)** to construct velocity field neural networks. Proves population- and sample-level exhaustiveness of the learned representation in recovering the minimal central $\sigma$-field $\mathcal{S}_{Y \mid X}$.
* **Methodological Complement**: Solves the over-sufficiency problem inherent in ensemble or variational methods (such as [[Belted and Ensembled Neural Network for Linear and Nonlinear SDR|BENN]] or [[NeurIPS-2022-nonlinear-sufficient-dimension-reduction-with-a-stochastic-neural-network StoNet|StoNet]]). Connects directly to non-Euclidean extensions in [[non_euclidean_random_objects_sdr]].

### 3. Deep Fair Learning: Unified Fine-tuning with Sufficient Networks
* **ArXiv**: [arXiv:2504.06470](https://arxiv.org/abs/2504.06470v1) (April 2025)
* **Authors**: Enze Shi, Linglong Kong, Bei Jiang
* **Key Innovation**: Integrates nonlinear SDR with deep learning to build **fair, unbiased representations**. Enforces conditional independence between sensitive attributes $A$ and learned features $\mathbf{f}_\theta(X)$ via a fine-tuning penalty, satisfying $Y \perp\!\!\!\perp A \mid \mathbf{B}^\top X$.
* **Methodological Complement**: Demonstrates an applied downstream use case of Neural SDR in AI ethics and algorithmic fairness across continuous, discrete, and multi-group sensitive attributes.

### 4. Deep Sufficient Representation Learning via Mutual Information (MSRL)
* **ArXiv**: [arXiv:2207.10772](https://arxiv.org/abs/2207.10772v1) (July 2022)
* **Authors**: Siming Zheng, Yuanyuan Lin, Jian Huang
* **Key Innovation**: Formulates neural SDR as a **Variational Mutual Information (MI)** optimization problem. Maximizes $I(\mathbf{f}_\theta(X); Y)$ using deep neural networks while proving non-asymptotic error bounds via generalized Dudley's inequality on order-two U-processes.
* **Methodological Complement**: Offers an information-theoretic loss objective complementing [[DVSDR-Deep-Variational-Architecture|DVSDR]] (ELBO-based) and [[DEEP NONLINEAR SUFFICIENT DIMENSION REDUCTION 2024|DDR]] (distance covariance-based).

### 5. Principal Optimal Transport Direction (POTD) for SDR Classification
* **ArXiv / Venue**: [arXiv:2010.09921](https://arxiv.org/abs/2010.09921v4) \| **NeurIPS 2020**
* **Authors**: Cheng Meng, Jun Yu, Jingyi Zhang, Ping Ma, Wenxuan Zhong
* **Key Innovation**: Fuses **Wasserstein Distance / Optimal Transport** with SDR for binary and multi-class classification. Estimates the SDR subspace basis via the principal directions of optimal transport couplings across response categories.
* **Methodological Complement**: Overcomes the failure of classical slice-based inverse regression ([[SDR Sliced inverse regression (SIR)|SIR]]) in categorical/binary response settings.

---

## 📊 Taxonomy & Citation Tables

### Table 1: Complementary Neural SDR Methods

| Method | arXiv / Publication | Venue / Journal | Authors | Optimization Objective | Sufficiency Target | Google Scholar Citations |
|---|---|---|---|---|---|---|
| **POTD** | [arXiv:2010.09921](https://arxiv.org/abs/2010.09921v4) | **NeurIPS 2020** | C. Meng, J. Yu, J. Zhang, P. Ma, W. Zhong | Optimal Transport / Wasserstein Coupling | Categorical SDR Subspace | **31** |
| **MSRL** | [arXiv:2207.10772](https://arxiv.org/abs/2207.10772v1) | **arXiv 2022** | S. Zheng, Y. Lin, J. Huang | Variational Mutual Information Bound | Central Subspace | **1** |
| **GenSDR** | [arXiv:2512.18971](https://arxiv.org/abs/2512.18971v1) | **arXiv 2025** | S. Xu, Z. Yu, J. Huang | Conditional Stochastic Interpolation (Diffusion) | Central $\sigma$-field ($G_{Y \mid X}$) | **1** |
| **Deep Fair SDR** | [arXiv:2504.06470](https://arxiv.org/abs/2504.06470v1) | **arXiv 2025** | E. Shi, L. Kong, B. Jiang | Neural SDR + Fair Representation Penalty | Fair Subspace ($Y \perp\!\!\!\perp A \mid B^\top X$) | **0** *(Preprint)* |
| **FlowSDR** | [arXiv:2606.01346](https://arxiv.org/abs/2606.01346v1) | **arXiv 2026** | Y. Dong, K. McAlinn, E. Airoldi, L. Li | Monotone Rational-Quadratic Spline Flow Likelihood | Full Distribution $p(Y \mid X)$ | **0** *(Preprint)* |

---

### Table 2: Complete Neural SDR Citation Landscape (Primary + Complementary)

| Paper / Method | Authors & Year | Primary Venue | Core Architectural Paradigm | Google Scholar Citations |
|---|---|---|---|---|
| **POTD** | Meng et al. (2020) | **NeurIPS 2020** | Optimal Transport Wasserstein directions for discrete $Y$ | **31** |
| **DDR** | Huang et al. (2024) | **IEEE TIT 2024** | Pairwise distance covariance $\text{dCov}^2(f(X), Y)$ | **22** |
| **NN-SDR** | Kapla et al. (2021) | **CSDA 2021** | 2-Stage OPG-initialized Stiefel bottleneck MLP | **18** |
| **StoNet** | Liang et al. (2022) | **NeurIPS 2022** | Stochastic Markov hidden layers & MCMC | **15** |
| **GMDDNet** | Chen et al. (2024) | **JASA 2024** | Augmented Lagrange + GMDD matrix | **8** |
| **DVSDR** | Banijamali et al. (2018) | **AISTATS 2018** | Variational Autoencoder (ELBO) | **24** |
| **DRNN** | Xu & Yu (2024) | **arXiv 2024** | Rank-regularized first layer least-squares | **2** |
| **MSRL** | Zheng et al. (2022) | **arXiv 2022** | Mutual Information variational lower bound | **1** |
| **BENN** | Tang & Li (2024) | **arXiv 2024** | Belted ensemble probability-determining family | **1** |
| **GenSDR** | Xu, Yu, Huang (2025) | **arXiv 2025** | Diffusion / Stochastic Interpolation velocity net | **1** |
| **Deep Fair SDR** | Shi, Kong, Jiang (2025) | **arXiv 2025** | Algorithmic fairness via conditional independence | **0** |
| **FlowSDR** | Dong et al. (2026) | **arXiv 2026** | Conditional Normalizing Flow log-likelihood | **0** |

---

## Related Vault Notes
- [[SDR and Neural Networks]]
- [[sdr_papers_summary]]
- [[pdf_reference_archive]]
- [[DEEP NONLINEAR SUFFICIENT DIMENSION REDUCTION 2024]]
- [[Belted and Ensembled Neural Network for Linear and Nonlinear SDR]]
- [[Neural Networks perform SDR using rank regularization 2024]]
- [[non_euclidean_random_objects_sdr]]

## Suggested New Notes
- `FlowSDR (Normalizing Flow SDR)` — Normalizing flows for conditional density estimation in SDR.
- `GenSDR (Generative Diffusion SDR)` — Diffusion matching and stochastic interpolation for exhaustive SDR.
- `Principal Optimal Transport Direction (POTD)` — Optimal transport couplings for categorical response SDR.
- `Deep Fair SDR` — Fairness constraints in SDR representations.
