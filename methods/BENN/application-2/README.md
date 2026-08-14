## Application 2: Geographical origin of music dataset (Section S.8)

- Folder: `application-2`

### Dataset information

- Original data source: <https://archive.ics.uci.edu/dataset/315/geographical+original+of+music>
- Dataset file: `default_features_1059_tracks.txt` (downloaded from the original source)


### Implementation details

- **BENN (`m=1`)**: Run `benn-all-application-2-1.ipynb` 
  - **Run this file first** since it includes splitting the training and testing data.
  - This file also reproduces Figure S.9.
- **BENN (`m=2`)**: Run `benn-all-application-2-2.ipynb` 
  - This file also reproduces Figure S.10.
- **GMDDNet**: Run `gmdd-application-2-correct.ipynb`
  - This file also reproduces Figure S.11.
- **StoNet**: `my_stonet_application.ipynb`
  - This file also reproduces Figure S.12.
- **GSIR**: Run `test-gsir-application.R` first, and then `application_graph_gsir.ipynb`
  - The former performs the algorithm, while the later reproduces Figure S.13.


### Remarks

-   Run `benn-all-application-2-1.ipynb` first to split the training and testing data. All other codes utilize the splitted data directly.
-   Some intermediate files are also provided in the folder.
