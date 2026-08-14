## Simulation - Model E-IV (Sections S.6.2 and S.6.3)

- Folder: `simulation-model-E-IV`
- **Note**: In all commands, `n` is replaced by sample size `1000, ..., 8000`; `t` is replaced by the index of the data `1, ..., 100`

### Methodology implementation

- **BENN-1**: 
  - Run `benn-prelim-model4-2-1.ipynb` for preliminary analysis
  - Run `benn-model4-poly.ipynb` or `benn-model4-poly.py` using the following command:
```bash
python benn-model4-poly.py --model1 4 --model2 2 --n n --m 1 --d 1 --t t --ep 100
```

- **BENN-2**:
  - Run `benn-prelim-model4-2-2.ipynb` for preliminary analysis
  - Run `benn-model4-poly.ipynb` or `benn-model4-poly.py` using the following command:
```bash
python benn-model4-poly.py --model1 4 --model2 2 --n n --m 2 --d 1 --t t --ep 100
```


- **BENN-1000**:
  - Run `benn-prelim-model4-2.ipynb` for preliminary analysis
  - Run `benn-all-ens-m-ker.ipynb` or `benn-all-ens-m-ker.py` using the following command:
```bash
python benn-all-ens-m-ker.py --model1 4 --model2 2 --n n --m 1000 --d 1 --t t
```


- **GMDDNet**
  - Run `my_gmdd_correct.ipynb` or `my_gmdd_correct.py` using the following command:
```bash
python my_gmdd_correct.py --model1 4 --model2 2 --n n --d 1 --t t
```


- **StoNet**
  - Run `my_stonet_all-4.ipynb` or `my_stonet_all-4.py` using the following command:
```bash
python my_stonet_all-4.py --model1 4 --model2 2 --n n --r 25 --d 1 --t t
```

- **GSIR**
  - Run `test-gsir-1.R` using the following command:
```bash
Rscript test-gsir-1.R n t 2
```

### Result summarization

- Reproduce Table S.2: Run `results-summary-model4-2-final.R`
- Reproduce Table S.4: Run `time-summary-model4-2.R`


### Remarks

- Some `.py` files are also provided in case it takes too long to run the `.ipynb` file.
- Directories need to be changed accordingly in each code.
- CPU times for each method under each sample size are summarized in Table S.4.
- In the command-line arguments,
  - `--model1` and `--model2` are labels of the settings
  - `--n` is the sample size
  - `--m` is the number of ensemble transformations
  - `--ep` is the number of epochs
  - `--d` is the width of the belt layer (the dimension of sufficient predictor)
  - `--t` is the index of experiment
  - `--r` is the width of the StoNet (specifically for StoNet)






