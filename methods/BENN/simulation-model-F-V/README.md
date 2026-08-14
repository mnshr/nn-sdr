## Simulation - Model F-V

- Folder: `simulation-model-F-V`
- **Note**: In all commands, `n` is replaced by sample size `1000, ..., 10000`; `m` is replaced by number of transformations `1, 2, 3`; `t` is replaced by the index of the data `1, ..., 100`.

### Methodology implementation

- **BENN**:
  - Run `benn-prelim-linear.ipynb` for preliminary analysis
  - Run `benn-linear-ens-all.ipynb` or `benn-linear-ens-all.py` using the following command:
```bash
python benn-linear-ens-all.py --model1 5 --model2 3 --n n --m m --d 2 --t t
```
  - Running time: <10s for each experiment

### Result summarization

- Reproduce Table S.6: Run `results-summary-model5-3-correct.R`


### Remarks

- Some `.py` files are also provided in case it takes too long to run the `.ipynb` file.
- Directories need to be changed accordingly in each code.
- Running times above are under the JupyterLab with 8 cores and 16GB memory.
- In the command-line arguments,
  - `--model1` and `--model2` are labels of the settings
  - `--n` is the sample size
  - `--m` is the number of ensemble transformations
  - `--d` is the width of the belt layer (the dimension of sufficient predictor)
  - `--t` is the index of experiment



