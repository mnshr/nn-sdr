## Simulations - Models A--C (Section S.6.1)

- Folder: `simulation-model-ABC`

### Methodology implementation

- **BENN**: 
  - Sample size `n=1000`: Run `benn-1000.ipynb` 
    - running time: ~40min for all 9 settings, each with 100 experiments
  - Sample size `n=2000`: Run `benn-2000.ipynb` 
    - running time: ~80min for all 9 settings, each with 100 experiments
  - Sample size `n=5000`: Run `benn-5000.ipynb`, with variables `model1` and `model2` modified for different settings, or run `benn-5000.py` with command-line arguments `model1--` and `model2--`
    - `--model1` takes values `1, 2, 3` for Models A, B, C; `--model2` takes values `1, 2, 3` for Settings I, II, III
    - running time: ~45min for each setting, with 100 experiments
      
- **StoNet**: 
  - Run `my_stonet_all.ipynb` with command-line arguments `--model1`, `--model2`, `--n`, and `--r`
    - `--model1` takes values `1, 2, 3` for Models A, B, C; `--model2` takes values `1, 2, 3` for Settings I, II, III; 
    `--n` takes values `1000, 2000, 5000` for different sample sizes; `--r` takes values `10, 15, 20, 25` for different network scales
    - running time: ~60min for `n=1000`, ~115min for `n=2000`, ~280min for `n=5000`, each with 100 experiments and `r=25`



### Result summarization

- Reproduce Table S.1: Run `results-summary-model123-final.R`


### Remarks

- Some `.py` files are also provided in case it takes too long to run the `.ipynb` file.
- We conduct preliminary analysis for BENN: `benn-prelim-model123.ipynb`
- Directories need to be changed accordingly in each code.
- Running times above are under the JupyterLab with 8 cores and 16GB memory.
- Results from GMDDNet and GSIR come from Chen et al. (2024). https://doi.org/10.1214/24-AOS2390




