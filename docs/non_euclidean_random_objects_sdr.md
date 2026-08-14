# SDR for Non-Euclidean Data and Random Objects

## Overview

Traditional [[Sufficient Dimension Reduction (SDR)]] assumes that predictors $\mathbf{X} \in \mathbb{R}^p$ and response $Y \in \mathbb{R}$ (or $\mathbb{R}^q$) reside in Euclidean spaces. However, modern complex applications frequently generate non-Euclidean response variables—such as probability distributions, manifold-valued data, functional curves, graph/network objects, or metric space elements.

This document serves as a dedicated repository for research, references, and mathematical formulations targeting **Non-Euclidean SDR and Random Objects**, tracked for **Phase 2** research following the primary Euclidean [[SDR and Neural Networks|SDR-NN working paper]].

---

## Key Papers & References

### 1. Fréchet Cumulative Covariance Net (FCCov-Net)
* **Title**: *Fréchet Cumulative Covariance Net for Deep Nonlinear Sufficient Dimension Reduction with Random Objects*
* **ArXiv**: [arXiv:2502.15374](https://arxiv.org/abs/2502.15374) (Feb 2025)
* **Authors**: Hang Yuan, Christina Dan Wang, Zhou Yu (`zyu@stat.ecnu.edu.cn`)
* **Framework**: PyTorch (Feedforward NNs + ResNet-type CNNs)
* **Key Innovations**:
  - Introduces **Fréchet Cumulative Covariance (FCCov)** to quantify dependence between Euclidean predictors $\mathbf{X}$ and non-Euclidean random object responses $Y \in \Omega$ (where $\Omega$ is a metric space equipped with metric $d_\Omega$).
  - Proves unbiasedness at the $\sigma$-field level under squared Frobenius norm regularization.
  - Establishes non-asymptotic convergence rates matching the minimax rate of non-parametric regression up to logarithmic factors.
  - Demonstrates empirical performance on facial expression recognition datasets and non-Euclidean simulated objects.

### 2. Connection to Generalized Distance Dependence
* Extends concepts from Generalized Mean Distance Dependence (GMDD) and Distance Covariance ($\text{dCov}$) to general metric spaces using Fréchet metrics and Fréchet means.

---

## Core Mathematical Formulations

### Fréchet Mean and Distance in Metric Space $\Omega$
For a random object $Y$ taking values in metric space $(\Omega, d_\Omega)$, the Fréchet mean set is defined by minimizing expected metric distances:
$$ \mathbb{E}\left[ d_\Omega^2(Y, \omega) \right] $$

### Fréchet Cumulative Covariance (FCCov)
Let $(\Omega, d_\Omega)$ be a metric space and $Y \in \Omega$ be a random object. The FCCov metric measures dependence between $f(\mathbf{X})$ and $Y$ by constructing kernel distance operators over centered distance metrics, enabling deep neural networks ($f_\theta$) to learn low-dimensional sufficient representations without requiring Euclidean operations on $Y$.

---

## Phase 2 Future Action Items
- [ ] **Implementation Study**: Code custom PyTorch loss modules for Fréchet Distance and FCCov.
- [ ] **Dataset Procurement**: Identify benchmark non-Euclidean datasets (e.g., manifold image representations, probability density functions, functional curve datasets).
- [ ] **Methodological Paper Draft**: Target a secondary paper on *Deep Non-Euclidean SDR for Metric-Valued Outcomes*.

---

## Related Vault Notes
- [[Sufficient Dimension Reduction (SDR)]]
- [[SDR and Neural Networks]]
- [[DEEP NONLINEAR SUFFICIENT DIMENSION REDUCTION 2024]]
- [[Belted and Ensembled Neural Network for Linear and Nonlinear SDR]]

## Suggested New Notes
- `Fréchet Cumulative Covariance (FCCov)` — non-parametric metric for random object dependence.
- `Object Data Analysis (ODA)` — statistical framework for metric-space valued random variables.
