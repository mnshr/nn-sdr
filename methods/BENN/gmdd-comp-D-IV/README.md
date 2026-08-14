## Comparisons to GMDDNet - Model D-IV (Section S.6.6)

- Folder: `gmdd-comp-D-IV`
- **Note**: In all commands, `n` is replaced by sample size `1000, ..., 8000`; `y_mode` is replaced by corresponding `Theta(v)` function `0, 2, 3, 4, 5`; `t` is replaced by the index of the data `1, ..., 100`

### Methodology implementation

- **GMDDNet**: 
  - Run `my_gmdd_correct-ymode.ipynb` or `my_gmdd_correct-ymode.py` using the following command:
``` bash
python my_gmdd_correct-ymode.py --model1 4 --model2 1 --n n --d 2 --y_mode y_mode --t t
```



### Result summarization

- Reproduce Table S.7: Run `results-summary-gmdd-model4-1-ymode.R`


### Remarks

- Some `.py` files are also provided in case it takes too long to run the `.ipynb` file.
- Directories need to be changed accordingly in each code.
- CPU times for each method under each sample size are similar to Table S.3.
- In the command-line arguments,
  - `--model1` and `--model2` are labels of the settings
  - `--n` is the sample size
  - `--d` is the width of the belt layer (the dimension of sufficient predictor)
  - `--y_mode` is the index of `Theta(v)` used in Chen et al. (2024) https://doi.org/10.1214/24-AOS2390
  - `--t` is the index of experiment








-   Result summarization: 
-   Side note: `results-model4-1.Rdata` is the intermediate file from Model D-IV.