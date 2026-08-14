## Simulation - Sensntivity analysis (Section S.6.4)

- Folder: `sensitivity-analysis`
- **Note**: In all commands, `t` is replaced by the index of the data `1, ..., 100`


### Methodology implementation

- Run all codes in `prelim-analysis` folder for preliminary analysis
- Run `benn-all-ens-m-ker-ep.ipynb` or `benn-all-ens-m-ker-ep.py` using the following commands:
  - Commands for Model D-IV: 
```bash
python benn-all-ens-m-ker-ep.py --model1 4 --model2 1 --n 5000 --l1 2 --r1 50 --l2 1 --r2 2000 --m 1000 --ep 150 --caseid 0 --d 2 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 1 --n 5000 --l1 1 --r1 100 --l2 1 --r2 2000 --m 1000 --ep 150 --caseid 1 --d 2 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 1 --n 5000 --l1 3 --r1 100 --l2 1 --r2 2000 --m 1000 --ep 100 --caseid 2 --d 2 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 1 --n 5000 --l1 3 --r1 50 --l2 1 --r2 2000 --m 1000 --ep 150 --caseid 3 --d 2 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 1 --n 5000 --l1 2 --r1 50 --l2 1 --r2 1000 --m 500 --ep 150 --caseid 4 --d 2 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 1 --n 5000 --l1 1 --r1 100 --l2 1 --r2 1000 --m 500 --ep 100 --caseid 5 --d 2 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 1 --n 5000 --l1 3 --r1 100 --l2 1 --r2 1000 --m 500 --ep 100 --caseid 6 --d 2 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 1 --n 5000 --l1 3 --r1 50 --l2 1 --r2 1000 --m 500 --ep 150 --caseid 7 --d 2 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 1 --n 5000 --l1 2 --r1 50 --l2 2 --r2 1000 --m 500 --ep 150 --caseid 8 --d 2 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 1 --n 5000 --l1 1 --r1 100 --l2 2 --r2 1000 --m 500 --ep 100 --caseid 9 --d 2 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 1 --n 5000 --l1 3 --r1 100 --l2 2 --r2 1000 --m 500 --ep 100 --caseid 10 --d 2 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 1 --n 5000 --l1 3 --r1 50 --l2 2 --r2 1000 --m 500 --ep 150 --caseid 11 --d 2 --t t
```
  - Commands for Model E-IV: 
```bash
python benn-all-ens-m-ker-ep.py --model1 4 --model2 2 --n 5000 --l1 2 --r1 50 --l2 1 --r2 2000 --m 1000 --ep 150 --caseid 0 --d 1 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 2 --n 5000 --l1 1 --r1 100 --l2 1 --r2 2000 --m 1000 --ep 100 --caseid 1 --d 1 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 2 --n 5000 --l1 3 --r1 100 --l2 1 --r2 2000 --m 1000 --ep 100 --caseid 2 --d 1 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 2 --n 5000 --l1 3 --r1 50 --l2 1 --r2 2000 --m 1000 --ep 100 --caseid 3 --d 1 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 2 --n 5000 --l1 2 --r1 50 --l2 1 --r2 1000 --m 500 --ep 150 --caseid 4 --d 1 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 2 --n 5000 --l1 1 --r1 100 --l2 1 --r2 1000 --m 500 --ep 100 --caseid 5 --d 1 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 2 --n 5000 --l1 3 --r1 100 --l2 1 --r2 1000 --m 500 --ep 100 --caseid 6 --d 1 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 2 --n 5000 --l1 3 --r1 50 --l2 1 --r2 1000 --m 500 --ep 100 --caseid 7 --d 1 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 2 --n 5000 --l1 2 --r1 50 --l2 2 --r2 1000 --m 500 --ep 100 --caseid 8 --d 1 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 2 --n 5000 --l1 1 --r1 100 --l2 2 --r2 1000 --m 500 --ep 100 --caseid 9 --d 1 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 2 --n 5000 --l1 3 --r1 100 --l2 2 --r2 1000 --m 500 --ep 100 --caseid 10 --d 1 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 2 --n 5000 --l1 3 --r1 50 --l2 2 --r2 1000 --m 500 --ep 100 --caseid 11 --d 1 --t t
```

### Result summarization

- Reproduce Table S.5: Run `results-summary-sensitivity.R`
- Summarize time: Run `time-sensitivity.R`


### Remarks

- Some `.py` files are also provided in case it takes too long to run the `.ipynb` file.
- Directories need to be changed accordingly in each code.
- CPU times for each experiment are ~1-2min.
- In the command-line arguments,
  - `--model1` and `--model2` are labels of the settings
  - `--n` is the sample size
  - `--l1` is the depth of the dimension reduction network
  - `--r1` is the width of the dimension reduction network
  - `--l2` is the depth of the ensemble regression network
  - `--r2` is the width of the ensemble regression network
  - `--m` is the number of ensemble transformations
  - `--p` is the number of epochs
  - `--caseid` is the index of the settings in Table S.5
  - `--d` is the width of the belt layer (the dimension of sufficient predictor)
  - `--t` is the index of experiment





