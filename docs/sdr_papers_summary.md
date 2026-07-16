# SDR + Neural Networks — Paper Resources Summary

## Overview

| # | Paper | Year | Code Repo | Framework | Datasets | Authors / Contact |
|---|-------|------|-----------|-----------|----------|-------------------|
| 1 | **Fusing Sufficient Dimension Reduction with Neural Networks** | 2021 | ✅ [NNSDR](https://git.art-ist.cc/daniel/NNSDR) | R + TensorFlow | Boston Housing (`mlbench`), kc_house_data (`MAVE`), Beijing Air Quality ([UCI](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data)) | Daniel Kapla et al. |
| 2 | **Deep Nonlinear Sufficient Dimension Reduction** (Chen 2024) | 2024 | ❌ None | Not specified | MNIST, Fashion-MNIST, simulations | Zhou Yu (corresponding) — `zyu@stat.ecnu.edu.cn`, Yuling Jiao — `yulingjiaomath@whu.edu.cn` |
| 3 | **Belted and Ensembled Neural Network for Linear and Nonlinear SDR** | 2024 | ❌ None | PyTorch | Superconductivity ([UCI, DOI](https://doi.org/10.24432/C53P47)), simulations | Authors acknowledge receiving code from Faming Liang & Zhou Yu |
| 4 | **Supervised Dynamic Dimension Reduction with Deep Neural Network** | 2025 | ❌ None | Not specified | FinC (Kozak 2019), Jena Climate (Max Planck), Weather (Kaggle), Energy & Light ([UCI](https://doi.org/10.24432/C53P47)) | Not extracted |
| 5 | **Neural Networks Perform SDR** (DRNN) | 2024 | ✅ [DRNN](https://github.com/oaksword/DRNN) | Python + PyTorch | Seoul Weather (UCI), simulations | Zhou Yu — `zyu@stat.ecnu.edu.cn` |
| 6 | **Nonlinear SDR with a Stochastic Neural Network** (StoNet) | 2022 | ❌ None (but references many comparison repos) | Not specified | Simulations, sub-MNIST, CT slices (UCI) | Faming Liang — `fmliang@purdue.edu` |

---

## Cloned Repos

The following repos have been cloned locally into the [sdr workspace](file:///Users/manish/gh/research/sdr):

| Repo | Local Path | Language |
|------|-----------|----------|
| NNSDR | [NNSDR/](file:///Users/manish/gh/research/sdr/NNSDR) | R (with TensorFlow via `reticulate`) |
| DRNN | [DRNN/](file:///Users/manish/gh/research/sdr/DRNN) | Python (PyTorch) |

---

## Comparison Method Code Links (from Paper #6 — StoNet)

These are publicly available implementations of comparison/baseline SDR methods referenced across papers:

| Method | URL |
|--------|-----|
| NNSDR | `https://git.art-ist.cc/daniel/NNSDR/src/branch/master` |
| RCIT (R package) | `https://github.com/ericstrobl/RCIT` |
| LSMIE | `http://www.ms.k.u-tokyo.ac.jp/software.html#LSDR` |
| KDR | `https://www.ism.ac.jp/~fukumizu/software.html` |
| SIR / SAVE (Python) | `https://joshloyal.github.io/sliced/` |
| CVarE (R package) | CRAN — `CVarE` |
| MAVE (R package) | CRAN — `MAVE` |

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
