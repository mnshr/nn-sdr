## Application 1: Superconductivty dataset (Sections 7 and S.7)

- Folder: `application-1`

### Dataset information

- Original data source: <https://archive.ics.uci.edu/dataset/464/superconductivty+data>
- Dataset file: `train.csv` (downloaded from the original source)

### Implementation details

- **BENN (`m=1`)**: Run `benn-all-application-1-1.ipynb` 
  - **Run this file first** since it includes splitting the training and testing data.
  - Run `application_graph_benn-1-reverse.ipynb` to reproduce Figure 4.
- **BENN (`m=2`)**: Run `benn-all-application-1-2.ipynb` 
  - This file also reproduces Figure S.1.
- **BENN (`m=10`)**: Run `benn-all-application-1-10.ipynb` 
  - This file also reproduces Figure S.2.
- **BENN (`m=50`)**: Run `benn-all-application-1-50.ipynb` 
  - This file also reproduces Figure S.3.
- **BENN (`m=100`)**: Run `benn-all-application-1-100.ipynb` 
  - This file also reproduces Figure S.4.
- **BENN (`m=200`)**: Run `benn-all-application-1-200.ipynb` 
  - This file also reproduces Figure S.5.
- **GMDDNet**: Run `gmdd-application-1-correct.ipynb`
  - This file also reproduces Figure S.6.
- **StoNet**: `my_stonet_application-1.ipynb`
  - This file also reproduces Figure S.7.
- **GSIR**: Run `test-gsir-application-1.R` first, and then `application_graph_gsir-1.ipynb`
  - The former performs the algorithm, while the later reproduces Figure S.8.
- **Running time summarization**: Run `application-1-result-summary.R`
  - This file reproduces Table S.9.

### Remarks

- Run `benn-all-application-1-1.ipynb` first to split the training and testing data. All other codes utilize the splitted data directly.
- Some intermediate files are also provided in the folder.