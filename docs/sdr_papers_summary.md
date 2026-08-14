# SDR + Neural Networks — Paper Resources Summary

## Overview

| #   | Paper                                                                     | Year | Code Repo                                                                                               | Framework        | Datasets                                                                                                                                                     | Authors / Contact                                                                                                      |
| --- | ------------------------------------------------------------------------- | ---- | ------------------------------------------------------------------------------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| 1   | **Fusing Sufficient Dimension Reduction with Neural Networks**            | 2021 | ✅ [NNSDR](https://git.art-ist.cc/daniel/NNSDR)                                                          | R + TensorFlow   | Boston Housing (`mlbench`), kc_house_data (`MAVE`), Beijing Air Quality ([UCI](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data)) | Daniel Kapla et al.                                                                                                    |
| 2   | **Deep Dimension Reduction for Supervised Representation Learning** (DDR) | 2024 | ✅ [Liao-Xu/DDR](https://github.com/Liao-Xu/DDR) ([IEEE](https://ieeexplore.ieee.org/document/10456552)) | Python + PyTorch | MNIST, Toy classification/regression                                                                                                                         | Jian Huang, Yuling Jiao (`yulingjiaomath@whu.edu.cn`), Xu Liao, Jin Liu, Zhou Yu (`zyu@stat.ecnu.edu.cn`)              |
| 3   | **Belted and Ensembled Neural Network for SDR** (BENN)                    | 2024 | ⚠️ [tyy20/BENN-codes](https://github.com/tyy20/BENN-codes) (Partial: requires external `GMDDNet`/`StoNet` files) | PyTorch          | Superconductivity ([UCI, DOI](https://doi.org/10.24432/C53P47)), simulations                                                                                 | Yuying Tang (`tangyuying@pku.edu.cn`), Bing Li (`bxl9@psu.edu`)                                                       |
| 4   | **Supervised Dynamic Dimension Reduction with Deep NN**                   | 2025 | ❌ None                                                                                                  | Not specified    | FinC (Kozak 2019), Jena Climate (Max Planck), Weather (Kaggle), Energy & Light ([UCI](https://doi.org/10.24432/C53P47))                                      | Not extracted                                                                                                          |
| 5   | **Neural Networks Perform SDR** (DRNN)                                    | 2024 | ✅ [DRNN](https://github.com/oaksword/DRNN)                                                              | Python + PyTorch | Seoul Weather (UCI), simulations                                                                                                                             | Zhou Yu — `zyu@stat.ecnu.edu.cn` (Includes SIR, SAVE, PCA, SPCA, Generalized Kernel, NN+LS in `demo`/`util`)           |
| 6   | **Nonlinear SDR with a Stochastic Neural Network** (StoNet)               | 2022 | ✅ [booml247/StoNet_SDR_NeurIPS](https://github.com/booml247/StoNet_SDR_NeurIPS) | Python + PyTorch | Simulations, sub-MNIST, CT slices (UCI)                                                                                                                      | Faming Liang — `fmliang@purdue.edu`                                                                                    |
| 7   | **Fréchet Cumulative Covariance Net for SDR with Random Objects**         | 2025 | ❌ None                                                                                                  | PyTorch          | Facial expression recognition, non-Euclidean random objects                                                                                                  | Hang Yuan, Christina Dan Wang, Zhou Yu (`zyu@stat.ecnu.edu.cn`) — [arXiv:2502.15374](https://arxiv.org/abs/2502.15374) |

---

## Cloned Repos

The following repos have been cloned locally into the [sdr workspace](file:///Users/manish/gh/research/sdr):

| Repo | Local Path | Language |
|------|-----------|----------|
| NNSDR | [nn-sdr/methods/NNSDR/](file:///Users/manish/gh/research/sdr/nn-sdr/methods/NNSDR) | R (with TensorFlow via `reticulate`) |
| DRNN | [nn-sdr/methods/DRNN/](file:///Users/manish/gh/research/sdr/nn-sdr/methods/DRNN) | Python (PyTorch) |
| DDR | [nn-sdr/methods/DDR/](file:///Users/manish/gh/research/sdr/nn-sdr/methods/DDR) | Python (PyTorch) |
| nsdr | [nn-sdr/methods/nsdr/](file:///Users/manish/gh/research/sdr/nn-sdr/methods/nsdr) | R (CRAN: Nonlinear SDR — GSIR, GSAVE, KPCA) |
| BENN | [nn-sdr/methods/BENN/](file:///Users/manish/gh/research/sdr/nn-sdr/methods/BENN) | Python (PyTorch + R for GSIR baseline) |
| StoNet | [nn-sdr/methods/StoNet/](file:///Users/manish/gh/research/sdr/nn-sdr/methods/StoNet) | Python (PyTorch) |

---

## Comparison Method Code Links (from Paper #6 — StoNet)

These are publicly available implementations of comparison/baseline SDR methods referenced across papers:

| Method              | URL                                                                                                                                                                           |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| nsdr (R package)    | `https://github.com/cran/nsdr` (CRAN — `nsdr`; GSIR, GSAVE, KPCA by Bing Li & Kyongwon Kim)                                                                                 |
| NNSDR               | `https://git.art-ist.cc/daniel/NNSDR/src/branch/master`                                                                                                                       |
| RCIT (R package)    | `https://github.com/ericstrobl/RCIT`                                                                                                                                          |
| LSMIE               | `http://www.ms.k.u-tokyo.ac.jp/software.html#LSDR`                                                                                                                            |
| GKDR / KDR          | `https://www.ism.ac.jp/~fukumizu/software.html` ([arXiv:1109.0455](https://arxiv.org/abs/1109.0455); *Gradient-based Kernel Dimension Reduction*, Fukumizu & Leng 2014, JASA) |
| SIR / SAVE (Python) | `https://joshloyal.github.io/sliced/`                                                                                                                                         |
| CVarE (R package)   | CRAN — `CVarE`                                                                                                                                                                |
| MAVE (R package)    | CRAN — `MAVE`                                                                                                                                                                 |

---

## Key R Packages Used Across Papers

| Package | Purpose |
|---------|---------|
| `tensorflow` | R interface to TensorFlow (Paper #1) |
| `MAVE` | MAVE estimation + kc_house_data dataset |
| `CVarE` | CVE method implementation |
| `mlbench` | Boston Housing dataset |
| `sliced` | Python package for SIR/SAVE |
| `scikit-learn` | StandardScaler, PCA |
