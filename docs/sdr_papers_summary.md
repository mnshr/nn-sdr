# SDR + Neural Networks — Paper Resources Summary

> 🌐 **Public Open-Source Repository**: [github.com/mnshr/nn-sdr (main)](https://github.com/mnshr/nn-sdr/tree/main)  
> 📁 **Local Workspace Path**: [nn-sdr/](file:///Users/manish/gh/research/sdr/nn-sdr) | Symlinked Obsidian Vault: `SDR-Research/sdr_papers_summary`

---

## Overview of Reviewed Papers & Vault Notes

| #   | Paper Title                                                               | Short Name | Author | Obsidian Vault Note | Year | Code Repo | Framework | Datasets | Contacts / Notes |
| --- | ------------------------------------------------------------------------- | ---------- | ------ | ------------------- | ---- | --------- | --------- | -------- | ---------------- |
| 1   | **Deep Variational Sufficient Dimensionality Reduction** (DVSDR)          | DVSDR | Banijamali et al. | [[DVSDR-Deep-Variational-Architecture-2018-12-18]] | 2018 | ❌ None | Python | MNIST, USPS | Erfan Banijamali et al. (AISTATS 2018) |
| 2   | **Fusing Sufficient Dimension Reduction with Neural Networks** (NN-SDR)   | DNN-SDR | Kapla et al. | [[Fusing sufficient dimension reduction with neural networks. 2021]] | 2021 | ✅ [NNSDR](https://git.art-ist.cc/daniel/NNSDR) | R + TensorFlow | Boston Housing (`mlbench`), kc_house_data (`MAVE`), Beijing Air Quality ([UCI](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data)) | Daniel Kapla, Lukas Fertl, Efstathia Bura |
| 3   | **Nonlinear SDR with a Stochastic Neural Network** (StoNet)               | StoNet | Liang et al. | [[NeurIPS-2022-nonlinear-sufficient-dimension-reduction-with-a-stochastic-neural-network StoNet]] | 2022 | ✅ [booml247/StoNet_SDR_NeurIPS](https://github.com/booml247/StoNet_SDR_NeurIPS) | Python + PyTorch | Simulations, sub-MNIST, CT slices (UCI) | Faming Liang — `fmliang@purdue.edu` |
| 4   | **Deep Nonlinear Sufficient Dimension Reduction** (GMDDNet) / **DDR**     | GMDD | Chen et al. / Huang et al. | [[DEEP NONLINEAR SUFFICIENT DIMENSION REDUCTION 2024]] | 2024 | ✅ [Liao-Xu/DDR](https://github.com/Liao-Xu/DDR) ([IEEE](https://ieeexplore.ieee.org/document/10456552)) | Python + PyTorch | MNIST, Toy classification/regression | Yinfeng Chen, Jian Huang, Yuling Jiao (`yulingjiaomath@whu.edu.cn`), Xu Liao, Jin Liu, Zhou Yu (`zyu@stat.ecnu.edu.cn`) |
| 5   | **Neural Networks Perform SDR** (DRNN)                                    | SDR-rank-regularized | Xu & Yu | [[Neural Networks perform SDR using rank regularization 2024]] | 2024 | ✅ [DRNN](https://github.com/oaksword/DRNN) | Python + PyTorch | Seoul Weather (UCI), simulations | Shuntuo Xu, Zhou Yu — `zyu@stat.ecnu.edu.cn` (Includes SIR, SAVE, PCA, SPCA, GKDR, NN+LS) |
| 6   | **Belted and Ensembled Neural Network for SDR** (BENN)                    | BENN | Tang & Li | [[Belted and Ensembled Neural Network for Linear and Nonlinear SDR]] | 2024 | ⚠️ [tyy20/BENN-codes](https://github.com/tyy20/BENN-codes) | Python + PyTorch (+ R) | Superconductivity ([UCI, DOI](https://doi.org/10.24432/C53P47)), simulations | Yuying Tang (`tangyuying@pku.edu.cn`), Bing Li (`bxl9@psu.edu`) |
| 7   | **Supervised Dynamic Dimension Reduction with Deep NN**                   | Dynamic SDR | Kapla et al. | [[SDR and Neural Networks]] | 2025 | ❌ None | Not specified | FinC (Kozak 2019), Jena Climate (Max Planck), Weather (Kaggle), Energy & Light ([UCI](https://doi.org/10.24432/C53P47)) | Daniel Kapla et al. (IEEE 2025) |
| 8   | **Fréchet Cumulative Covariance Net for SDR with Random Objects** (FCCov) | FCCov-Net | Yuan, Wang, Yu | [[non_euclidean_random_objects_sdr]] | 2025 | ❌ None | PyTorch | Facial expression recognition, non-Euclidean random objects | Hang Yuan, Christina Dan Wang, Zhou Yu (`zyu@stat.ecnu.edu.cn`) — [arXiv:2502.15374](https://arxiv.org/abs/2502.15374) |

---

## Cloned Repositories

The following implementations are cloned and synced inside the [mnshr/nn-sdr](https://github.com/mnshr/nn-sdr/tree/main) repository:

| Repo / Method | Local Workspace Path | Language & Framework | Description |
|---|---|---|---|
| **NNSDR** | [nn-sdr/methods/NNSDR/](file:///Users/manish/gh/research/sdr/nn-sdr/methods/NNSDR) | R + TensorFlow (`reticulate`) | Two-stage OPG + neural bottleneck refinement (Kapla 2021) |
| **DRNN** | [nn-sdr/methods/DRNN/](file:///Users/manish/gh/research/sdr/nn-sdr/methods/DRNN) | Python (PyTorch) | Rank-regularized first layer for CMS recovery (Xu & Yu 2024) |
| **DDR** | [nn-sdr/methods/DDR/](file:///Users/manish/gh/research/sdr/nn-sdr/methods/DDR) | Python (PyTorch) | Deep Dimension Reduction with distance correlation / GMDD (Huang et al. 2024) |
| **BENN** | [nn-sdr/methods/BENN/](file:///Users/manish/gh/research/sdr/nn-sdr/methods/BENN) | Python (PyTorch) + R | Belted and Ensembled Neural Network (Tang & Li 2024; `tyy20/BENN-codes`) |
| **StoNet** | [nn-sdr/methods/StoNet/](file:///Users/manish/gh/research/sdr/nn-sdr/methods/StoNet) | Python (PyTorch) | Stochastic neural network with Markovian layer structure (Liang 2022; `booml247`) |
| **nsdr** | [nn-sdr/methods/nsdr/](file:///Users/manish/gh/research/sdr/nn-sdr/methods/nsdr) | R (CRAN package) | Classical nonlinear SDR benchmarks: GSIR, GSAVE, KPCA (Bing Li & Kyongwon Kim) |

---

## Comparison Method Code Links

Publicly available implementations of classical / baseline comparison SDR methods:

| Method              | URL / Source                                                                                                                                                                  |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **nsdr (R package)**| `https://github.com/cran/nsdr` (CRAN — `nsdr`; GSIR, GSAVE, KPCA by Bing Li & Kyongwon Kim)                                                                                 |
| **NNSDR**           | `https://git.art-ist.cc/daniel/NNSDR/src/branch/master`                                                                                                                       |
| **RCIT (R package)**| `https://github.com/ericstrobl/RCIT`                                                                                                                                          |
| **LSMIE**           | `http://www.ms.k.u-tokyo.ac.jp/software.html#LSDR`                                                                                                                            |
| **GKDR / KDR**      | `https://www.ism.ac.jp/~fukumizu/software.html` ([arXiv:1109.0455](https://arxiv.org/abs/1109.0455); *Gradient-based Kernel Dimension Reduction*, Fukumizu & Leng 2014, JASA) |
| **SIR / SAVE**      | `https://joshloyal.github.io/sliced/` (Python package `sliced`)                                                                                                               |
| **CVarE**           | CRAN — `CVarE`                                                                                                                                                                |
| **MAVE**            | CRAN — `MAVE`                                                                                                                                                                 |

---

## Key R & Python Packages Used

| Package | Language | Purpose |
|---|---|---|
| `torch` / `torchvision` | Python | Deep learning training for DRNN, DDR, BENN, StoNet |
| `tensorflow` | R / Python | R interface to TensorFlow via `reticulate` for NNSDR (Kapla 2021) |
| `nsdr` | R | Classical nonlinear SDR benchmarks: GSIR, GSAVE, KPCA |
| `MAVE` | R | MAVE estimation + `kc_house_data` dataset |
| `CVarE` | R | Conditional variance estimation (CVE) |
| `mlbench` | R | Boston Housing benchmark dataset |
| `sliced` | Python | Python implementations of SIR and SAVE |
| `scikit-learn` | Python | StandardScaler, PCA, evaluation metrics |
