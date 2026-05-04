# The Feature Extraction Multiverse: How Hyperparameter Choices in Handcrafted EEG Features Shape Classification Outcomes

## Overview

This project quantifies how hyperparameter choices inside handcrafted EEG feature extractors propagate into classification outcomes. We evaluate 188 hyperparameter combinations spanning nine feature families (SampEn, ApEn, PermEn, FuzzyEn, LZC, DWT, PSD, Hjorth, Hurst) on three classification tasks defined over the Bonn University EEG dataset (binary, ternary, and five-class), and replicate the central comparisons on the CHB-MIT Scalp EEG database for cross-dataset validation. The framing is a multiverse / sensitivity analysis (the goal is to map how outcomes vary across defensible hyperparameter choices), not hyperparameter optimization for peak accuracy.

## Repository structure

```
eeg-fe-multiverse/
├── configs/         hyperparameter grid (YAML) consumed by feature extractors
├── data/            local-only datasets (Bonn, CHB-MIT); excluded from version control
├── notebooks/       exploratory and reporting notebooks
├── paper/           LaTeX manuscript sources and bibliography
├── results/         generated outputs
│   ├── figures/     manuscript figures (PNG and PDF pairs)
│   └── tables/      per-feature and aggregated CSV results
├── scripts/         entry points that run feature extraction and classification
└── src/             library code
    └── features/    one module per feature family plus a shared grid runner
```

## Installation

Requires Python 3.10 or newer.

```
git clone https://github.com/reyhanhosavci/eeg-fe-multiverse.git
cd eeg-fe-multiverse
pip install -r requirements.txt
```

## Data

The Bonn University EEG dataset and the CHB-MIT Scalp EEG database are not included in this repository. Download instructions and the expected on-disk layout are documented in [data_links.md](data_links.md). All scripts assume the data have been placed under `data/bonn/raw/` and `data/chbmit/raw/` as described there.

## Reproducing the analysis

The `scripts/` directory contains the entry points used to regenerate every CSV under `results/tables/` and every figure under `results/figures/`. Run them from the repository root in the order below.

1. `run_entropy_full.py`, compute the five entropy feature families (SampEn, ApEn, PermEn, FuzzyEn, LZC) across the full hyperparameter grid on Bonn.
2. `run_remaining_features.py`, compute the four remaining feature families (DWT, PSD, Hjorth, Hurst) on Bonn.
3. `run_classification.py`, classify every (feature, hyperparameter, task) combination with LightGBM and write per-feature accuracy and F1 tables.
4. `run_model_independence.py`, repeat the classification step with XGBoost and Random Forest to assess model dependence.
5. `test_chbmit_segments.py`, sanity-check the CHB-MIT segment extraction pipeline before the full validation run.
6. `run_chbmit_full.py`, extract features on the CHB-MIT subset.
7. `run_chbmit_validation.py`, classify the CHB-MIT features and compute the cross-dataset ranking comparison.

Figures are produced from the resulting tables by the visualization helpers in `src/visualization.py` and `src/fig10_timedomain_heatmap.py`.

## Citation

If you use this code or build on the analysis, please cite:

```bibtex
@article{hosavci2026multiverse,
  title   = {The Feature Extraction Multiverse: How Hyperparameter Choices in Handcrafted {EEG} Features Shape Classification Outcomes},
  author  = {Hosavci, Reyhan and others},
  journal = {Manuscript submitted to Computer Methods and Programs in Biomedicine},
  year    = {2026}
}
```

## License

Released under the MIT License (see [LICENSE](LICENSE)).
