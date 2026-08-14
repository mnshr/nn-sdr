## Comparisons to GMDDNet - Model G-IV (Section S.6.6)

- Folder: `gmdd-comp-G-IV`
- **Note**: In all commands, `n` is replaced by sample size `1000, ..., 8000`; `y_mode` is replaced by corresponding `Theta(v)` function `0, 2, 3, 4, 5`; `t` is replaced by the index of the data `1, ..., 100`

### Methodology implementation

- **BENN**:
  - Run `benn-all-ens-m-ker-ep100.ipynb` or `benn-all-ens-m-ker-ep100.py` using the following command:
``` bash
python benn-all-ens-m-ker-ep100.py --model1 4 --model2 4 --n n --m 1000 --d 2 --t t
```

- **GMDDNet**: 
  - Run `my_gmdd_correct-ymode.ipynb` or `my_gmdd_correct-ymode.py` using the following command:
``` bash
python my_gmdd_correct-ymode.py --model1 4 --model2 4 --n n --d 2 --y_mode y_mode --t t
```



### Result summarization

- Reproduce Table S.8: Run `results-summary-gmdd-model4-4-ymode.R`


### Remarks

- Some `.py` files are also provided in case it takes too long to run the `.ipynb` file.
- Directories need to be changed accordingly in each code.
- CPU times for each method under each sample size are similar to Table S.3.
- In the command-line arguments,
  - `--model1` and `--model2` are labels of the settings
  - `--n` is the sample size
  - `--m` is the number of ensemble transformations for BENN
  - `--d` is the width of the belt layer (the dimension of sufficient predictor)
  - `--y_mode` is the index of `Theta(v)` used in Chen et al. (2024) https://doi.org/10.1214/24-AOS2390
  - `--t` is the index of experiment