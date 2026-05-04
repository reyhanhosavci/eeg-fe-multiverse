# Datasets

This project relies on two publicly available EEG datasets. Neither is redistributed in this repository. Users must download the data themselves from the original sources and place the files in the locations described below before running any analysis.

## Bonn University EEG Dataset

The primary dataset is the Bonn University EEG database, originally published by the Department of Epileptology, University of Bonn. The data are distributed by the Nonlinear Time Series Analysis group at Universitat Pompeu Fabra (https://www.upf.edu/web/ntsa/downloads). The five recording sets used in this work (A, B, C, D, E) cover surface and intracranial EEG segments from healthy controls, interictal patients, and ictal recordings. After download, place the unpacked `.txt` files under `data/bonn/raw/`, with one subdirectory per set (`Z`, `O`, `N`, `F`, `S` or equivalently `A`–`E`, depending on the original archive naming).

## CHB-MIT Scalp EEG Database

Cross-dataset validation uses the CHB-MIT Scalp EEG database hosted on PhysioNet (https://physionet.org/content/chbmit/1.0.0/). The full database contains pediatric scalp recordings from 23 patients with intractable seizures. The present study uses a 10-patient subset (chb01, chb02, chb03, chb04, chb05, chb06, chb07, chb08, chb09, chb10, chb23) consistent with the patient selection reported in the manuscript. Place the downloaded `.edf` files and per-patient `chbXX-summary.txt` annotation files under `data/chbmit/raw/`, preserving the original PhysioNet folder layout (one subdirectory per patient).

## Reproducibility note

The `data/` directory is excluded from version control. Reproducing the reported results requires the user to obtain both datasets from the sources above. Segment lengths, sampling rates, and the patient subset are documented in the corresponding loader modules (`src/data_loader.py` for Bonn, `src/chbmit_loader.py` for CHB-MIT) and in the manuscript.
