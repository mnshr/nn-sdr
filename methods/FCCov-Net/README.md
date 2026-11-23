# Environment

In the codebase, mccov is another name of our fccov method.

```bash
# create conda environment
conda create -n fccov python=3.8.20
conda activate fccov

# install R inside the conda environment
conda install -c conda-forge r-base

# install PyTorch
pip install torch==2.4.1+cu121 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# install Python dependencies
pip install -r requirements.txt
```

Then install the R package from the repository root:

```bash
# GSIR uses `rpy2` to call the R package `nsdr`.
Rscript -e 'install.packages("./nsdr_0.1.1.tar.gz", repos = NULL, type = "source")'
```

## Use of the bundled R package `nsdr`

The bundled file `nsdr_0.1.1.tar.gz` is a source archive of the R package
*nsdr* (Nonlinear Sufficient Dimension Reduction), so its installation does
not depend on its current availability on CRAN. The package provides
kernel-based nonlinear dimension-reduction methods, including kernel PCA,
generalized sliced inverse regression (GSIR), and generalized SAVE.

In this repository, the only function called directly from this R package is
`gsir()`. It is used as the GSIR benchmark in `generate_gsir_res()` in
`Simulations/Euclidean/euclidean_d1.py` and
`Simulations/Euclidean/euclidean_d2.py`, which generate the GSIR results for
the Euclidean simulation tables. The Python scripts pass the training
predictors, test predictors, and scalar training responses to R through
`rpy2` and call

`gsir(x_train, x_test, y_train, "scalar", 0.01, 0.01, 1, 1, r = d)`.

The function constructs centered Gaussian-kernel Gram matrices for the
predictors and response, applies regularized inverses, and uses an
eigendecomposition of the GSIR objective matrix to obtain an `r`-dimensional
representation of the test predictors. Only the returned `suff.pred`
component is converted back to a NumPy array and used in the subsequent
distance-correlation evaluation. The local Python files named `NSDR.py`
implement a separate neural-network model and are not Python wrappers for
the R package.

# Data and Checkpoint Download

## Real-data Assets

The real-data assets and pretrained checkpoints used by the scripts under `Real_data/` are provided as `fccov_data.zip`. Download and extract the archive as follows:

```bash
cd ./Real_data
wget -c "https://drive.google.com/uc?export=download&id=19ixcFhRVDkoHrpFl_S6bYlBEucvm0k--" -O fccov_data.zip
unzip fccov_data.zip -d data
```

After extraction, the expected directory is:

```text
./Real_data/data/
```

If Google Drive blocks command-line download, download `fccov_data.zip` manually from the browser link below, place it under `./Real_data/`, and run the same `unzip` command.

```text
https://drive.google.com/file/d/19ixcFhRVDkoHrpFl_S6bYlBEucvm0k--/view?usp=drive_link
```

## Non-Euclidean Simulation Checkpoints

The pretrained model weights and saved results for the Non-Euclidean simulations (distribution, SPD, and sphere settings) are provided as `non_euclidean_checkpoints.zip`. Download and extract the archive as follows:

```bash
cd ./Simulations/Non-Euclidean
wget -c "https://drive.google.com/uc?export=download&id=1GbnnbvyHfKdbwH8txYpiWRU6vCX-inYO" -O non_euclidean_checkpoints.zip
unzip non_euclidean_checkpoints.zip
```

After extraction, the expected directory is:

```text
./Simulations/Non-Euclidean/revise_result/
├── distribution/   # Model weights and results for the distribution-valued setting
├── spd/            # Model weights and results for the SPD setting
└── sphere/         # Results for the spherical setting
```

If Google Drive blocks command-line download, download `non_euclidean_checkpoints.zip` manually from the browser link below, place it under `./Simulations/Non-Euclidean/`, and run the same `unzip` command.

```text
https://drive.google.com/file/d/1GbnnbvyHfKdbwH8txYpiWRU6vCX-inYO/view?usp=sharing
```

# Project Tree

```text
fccov-code-jasa/
├── Real_data/                  # All real-data-related code and assets
│   ├── data/                       # Datasets and saved model weights
│   │   ├── 2d_ckpt/                    # Checkpoint for reproducing the 2D JAFFE feature plot
│   │   ├── 3d_graph1/                  # Auxiliary saved data and weights for 3D visualization
│   │   ├── 3d_graph2/                  # Data and model weights for 3D visualization
│   │   ├── MNIST_256/                  # Processed MNIST tensors used by the MNIST notebook
│   │   └── jaffe/                      # Preprocessed JAFFE dataset
│   ├── dense_Hellinger/            # Code for experiments under Hellinger distance
│   ├── dense_TV/                   # Code for experiments under total variation distance
│   └── figures/                    # Reproduced figures for the real-data section
└── Simulations/                # All simulation-related code
    ├── Euclidean/                  # Euclidean simulation code, results, and model weights
    │   ├── revise_results_d1/          # Euclidean simulation results for structural dimension d = 1
    │   └── revise_results_d2/          # Euclidean simulation results for structural dimension d = 2
    └── Non-Euclidean/              # Non-Euclidean simulation code, results, and model weights
        └── revise_result/              # Results for all non-Euclidean simulations
            ├── distribution/               # Results and weights for the distribution-valued setting
            ├── spd/                        # Results and weights for the SPD setting
            └── sphere/                     # Results for the spherical setting
```

# File Overview

Euclidean simulations:

- `Simulations/Euclidean/euclidean_d1.py`: Main script for Euclidean simulation experiments with structural dimension `d = 1`. 
- `Simulations/Euclidean/euclidean_d2.py`: Main script for Euclidean simulation experiments with structural dimension `d = 2`.
- `Simulations/Euclidean/NSDR.py`: Core implementation of the NSDR training pipeline and the neural-network-based dimension reduction model used by FCCov-related Euclidean experiments.
- `Simulations/Euclidean/support_function.py`: Utility functions for Euclidean simulations, including data generation, metric calculations, and helper routines used by the main scripts.
- `Simulations/Euclidean/model_reg.py`: Implementations of comparison methods used in the Euclidean experiments.
- `Simulations/Euclidean/dimension_selection.py`: Script for structural dimension selection experiments and related analysis.
- `Simulations/Euclidean/default.txt`: Default log/output text produced during NSDR-related runs.

Non-Euclidean simulations:

- `Simulations/Non-Euclidean/revise_sphere.py`: Main script for non-Euclidean simulations on spherical-valued responses.
- `Simulations/Non-Euclidean/revise_spd.py`: Main script for non-Euclidean simulations on SPD-matrix-valued responses.
- `Simulations/Non-Euclidean/revise_distribution.py`: Main script for non-Euclidean simulations on distribution-valued responses.
- `Simulations/Non-Euclidean/mynet.py`: Neural network architectures and training code used by the non-Euclidean FCCov experiments.
- `Simulations/Non-Euclidean/support_function.py`: Utility functions for non-Euclidean simulations, including geometric operations, data-generation helpers, and baseline-related routines.

Real-data experiments:

- `Real_data/dense_Hellinger/dense_hellinger.py`: Main script for real-data experiments under the Hellinger distance setting on the JAFFE dataset.
- `Real_data/dense_TV/dense_TV.py`: Main script for real-data experiments under the total variation distance setting on the JAFFE dataset.
- `Real_data/draw_figure1.py`: Script for reproducing Figure 1, including the selected JAFFE face images, empirical label distributions, and combined figure panels.
- `Real_data/draw_figure2_2d_feature.py`: Script for reproducing the 2D JAFFE feature scatter plot with embedded face thumbnails from the saved DCNN checkpoint.
- `Real_data/draw_figure3_3d_visualization.py`: Script for rendering the learned 3D JAFFE embeddings from saved visualization data and pretrained DenseNet weights.
- `Real_data/JAFFE_draw_images.ipynb`: Notebook for JAFFE data processing, visualization checks, and qualitative analysis used during figure preparation.
- `Real_data/MNIST.ipynb`: Notebook for the MNIST real-data experiment and related checks.
- `Real_data/NSDR.py`: Core implementation of the NSDR model used in the real-data experiments.
- `Real_data/mynet.py`: Neural network architectures used by the real-data experiments.
- `Real_data/support_function.py`: Shared utility functions for preprocessing, distance computation, evaluation, and training support in the real-data experiments.
- `Real_data/data/jaffe/`: Preprocessed JAFFE data, labels, and empirical label-distribution arrays used by the JAFFE experiments.
- `Real_data/data/2d_ckpt/`: Saved DCNN checkpoint used by `draw_figure2_2d_feature.py` to reproduce the 2D feature visualization.
- `Real_data/data/3d_graph1/`: Auxiliary saved train/test splits, label distributions, and DenseNet checkpoint for 3D visualization.
- `Real_data/data/3d_graph2/`: Saved train/test splits, label distributions, DenseNet checkpoint, and archived 3D view images for the main 3D visualization setting.
- `Real_data/data/MNIST_256/`: Processed MNIST tensors used by `Real_data/MNIST.ipynb`.
- `Real_data/figures/`: Reproduced real-data figures, including `2d_feature.png`, `3d_view_angle1.png`, `3d_view_angle2.png`, `3d_view_angle3.png`, `dis_combination.png`, and `image_combination.png`.


# Reproducibility of Figures and Tables in the Main Text

## For Table 2 in the Main Text
For Model I, simply run:

```bash
cd ./Simulations/Euclidean
python euclidean_d1.py
```

The results will be saved to `./Simulations/Euclidean/revise_results_d1/test`.
The four columns correspond to FCCov-Net, GMDD-Net, GSIR, and DDR, respectively.

For Model II, simply run:

```bash
python euclidean_d2.py
```

The results will be saved to `./Simulations/Euclidean/revise_results_d2/test`.

## For Table 3 in the Main Text, Outlier Case 1
For Model I, simply run:

```bash
cd ./Simulations/Euclidean
python euclidean_d1.py --outlier --case one --result-name revise_results_d1/test_outlier-1_I --n-samples 1000 3000
```

The results will be saved to `./Simulations/Euclidean/revise_results_d1/test_outlier-1_I`.

For Model II, simply run:

```bash
cd ./Simulations/Euclidean
python euclidean_d2.py --outlier --case one --result-name revise_results_d2/test_outlier-1_II --n-samples 1000 3000
```

The results will be saved to `./Simulations/Euclidean/revise_results_d2/test_outlier-1_II`.

## For Table 4 in the Main Text, Outlier Case 2
For Model I, simply run:

```bash
cd ./Simulations/Euclidean
python euclidean_d1.py --outlier --case two --result-name revise_results_d1/test_outlier-2_I --n-samples 1000 3000
```

The results will be saved to `./Simulations/Euclidean/revise_results_d1/test_outlier-2_I`.

For Model II, simply run:

```bash
cd ./Simulations/Euclidean
python euclidean_d2.py --outlier --case two --result-name revise_results_d2/test_outlier-2_II --n-samples 1000 3000
```

The results will be saved to `./Simulations/Euclidean/revise_results_d2/test_outlier-2_II`.

## For Table 5 in the Main Text
Simply run:

```bash
cd ./Simulations/Non-Euclidean
python revise_distribution.py
```

The results will be saved to `./Simulations/Non-Euclidean/revise_result/distribution/test`.

## For Tables 6 and 7 in the Main Text
Simply run:

```bash
cd ./Simulations/Non-Euclidean
python revise_spd.py
```

The results will be saved to `./Simulations/Non-Euclidean/revise_result/spd/test`.
Here, `dist_mode = 2` corresponds to the SPD affine-invariant distance, and `dist_mode = 6` corresponds to the SPD log-Cholesky distance.

## For Table 8 and 9 in the Main Text
For the results in the first two rows of these two tables, i.e., the methods with the `-HD` suffix, run:

```bash
cd ./Real_data/dense_Hellinger
python dense_hellinger.py --d 1 --bl 2e-6
python dense_hellinger.py --d 2 --bl 3e-6
python dense_hellinger.py --d 3 --bl 2e-6
```

These commands reproduce the corresponding results directly. Sample output:

```text
Classification accuracy:
MCCOV[0.27906977 0.37209302 0.37209302 0.46511628];
WIRE[0.11627907 0.30232558 0.39534884 0.02325581]
MSE of KNN (Frechet Regression):
MCCOV[0.10424098 0.04336037 0.0978208  1.7914729 ];
WIRE[0.12263799 0.04527765 0.11589474 1.95464702]
```

For the results in the last two rows of these two tables, i.e., the methods with the `-TV` suffix, run:

```bash
cd ./Real_data/dense_TV
python dense_TV.py --d 1 --bl 4e-6
python dense_TV.py --d 2 --bl 4e-6
python dense_TV.py --d 3 --bl 2e-6
```

These commands reproduce the corresponding results directly.

## Visualization

For Figure 1, run the following code. The results are saved to `./Real_data/figures/dis_combination.png` and `./Real_data/figures/image_combination.png`.

```bash
cd ./Real_data
python draw_figure1.py
```

For Figure 2, run the following code. The result is saved to `./Real_data/figures/2d_feature.png`.

```bash
cd ./Real_data
python draw_figure2_2d_feature.py
```

For Figure 3, run the following code. The results are saved to `./Real_data/figures/3d_view_angle1.png`, `./Real_data/figures/3d_view_angle2.png`, and `./Real_data/figures/3d_view_angle3.png`.

```bash
cd ./Real_data
python draw_figure3_3d_visualization.py
```

# Some Additional Results in Appendix

For MNIST, run block by block in `./Real_data/MNIST.ipynb`

For data processing, run `./Real_data/JAFFE_draw_images.ipynb`
