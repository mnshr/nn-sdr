# Neural Network Enabled Sufficient Dimension Reduction (SDR) Methods

A library and benchmark suite aggregating neural network-based Sufficient Dimension Reduction (SDR) methods in both Python and R.

---

## Directory Structure

- `methods/`
  - `DRNN/`: PyTorch implementation of *Neural Networks Perform SDR*.
  - `NNSDR/`: R/TensorFlow implementation of *Fusing Sufficient Dimension Reduction with Neural Networks*.
- `data/`: Datasets download and preprocessing helper utilities.
- `simulations/`: Scripts for synthetic benchmarks and replication models.
- `docs/`: Summaries of papers and author contacts.

## Getting Started

### 1. Requirements

For Python:
```bash
pip install -r methods/DRNN/requirements.txt
```

For R:
Ensure you have `reticulate` and `tensorflow` installed.

### 2. Download Datasets
You can download the benchmark datasets (Beijing Air Quality, Fashion-MNIST, MNIST, Seoul Weather, Superconductivity, etc.) using:
```bash
python data/download.py
```
This script will download and organize the data in `data/`.

---

## Literature Reference & Summary

A list of paper references, author contacts, and resources is maintained in `docs/sdr_papers_summary.md`.
