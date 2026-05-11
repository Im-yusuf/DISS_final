# ECG Lead Reduction for Multi-label 12-Lead Classification

This repository studies how far a 12-lead ECG can be reduced before classification performance degrades materially. The code trains and compares two deep learning architectures across multiple lead subsets, evaluates the resulting models on a multi-label classification task, and generates both comparison figures and lead-importance visualisations.

The project is organised as a regular Python codebase with a small `scripts/` layer for command-line entry points and an `ecg_lead_reduction/` package for the implementation.

## What the project does

- Merges three ECG corpora into one training/evaluation pipeline.
- Preprocesses raw WFDB records into a compressed NumPy cache.
- Trains both ResNet and CNN-LSTM models.
- Compares performance across 12-, 6-, 4-, 3-, 2-, and 1-lead input configurations.
- Saves per-run metrics, summary tables, checkpoints, and figures.
- Computes Integrated Gradients lead-importance maps from trained checkpoints.

## Repository layout

```text
.
|- artifacts/
|  |- checkpoints/
|  |- figures/
|  \- results/
|- data/
|  |- processed/
|  \- raw/
|- ecg_lead_reduction/
|  |- analysis/
|  |- core/
|  |- data/
|  |- evaluation/
|  |- experiments/
|  |- models/
|  \- training/
|- scripts/
|- requirements.txt
\- README.md
```

## Datasets and expected raw-data layout

The preprocessing pipeline expects three datasets under `data/raw/`:

- CPSC 2018
- Georgia
- PTB-XL

The current code resolves them from these directories:

```text
data/raw/
|- cpsc_2018/
|  \- cpsc_2018/
|- georgia/
|  \- georgia/
\- ptb-xl/
|  \- ptb-xl/
```

Within each dataset directory, the preprocessing script searches for WFDB header files using the glob pattern `g*/*.hea`.

## Dataset download guide

This repository is built against the PhysioNet Challenge 2020 training-folder layout for these datasets. That matters because the preprocessing code expects files to live inside `g#` subfolders such as `g1/`, `g2/`, and so on.

Recommended source:

- CPSC 2018, Georgia, and PTB-XL should be downloaded from the PhysioNet Challenge 2020 training release, not mixed from different directory formats.
- The Challenge 2020 training index includes exactly the subfolders this repo uses: `cpsc_2018/`, `georgia/`, and `ptb-xl/`.
- The standalone PTB-XL release is useful for reference, but its native `records100/` and `records500/` layout does not match this repository's current preprocessing logic.

Reference pages:

- Challenge 2020 dataset page: https://physionet.org/content/challenge-2020/1.0.2/
- Challenge 2020 files index: https://physionet.org/files/challenge-2020/1.0.2/training/
- Standalone PTB-XL page: https://physionet.org/content/ptb-xl/1.0.3/

### Option 1: Download only the folders this repo uses

Create the raw-data directories first:

```bash
mkdir -p data/raw/cpsc_2018 data/raw/georgia data/raw/ptb-xl
```

Then download each dataset directly from the Challenge 2020 training release:

```bash
wget -r -N -c -np -nH --cut-dirs=4 -P data/raw/cpsc_2018 \
	https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/

wget -r -N -c -np -nH --cut-dirs=4 -P data/raw/georgia \
	https://physionet.org/files/challenge-2020/1.0.2/training/georgia/

wget -r -N -c -np -nH --cut-dirs=4 -P data/raw/ptb-xl \
	https://physionet.org/files/challenge-2020/1.0.2/training/ptb-xl/
```

After these commands, the directory tree should look like this:

```text
data/raw/
|- cpsc_2018/
|  \- cpsc_2018/
|     |- g1/
|     |- g2/
|     \- ...
|- georgia/
|  \- georgia/
|     |- g1/
|     |- g2/
|     \- ...
\- ptb-xl/
	 \- ptb-xl/
			|- g1/
			|- g2/
			\- ...
```

### Option 2: Download the full Challenge 2020 training package and keep the needed folders

If you prefer a single download, PhysioNet also provides a full archive and a full training tree for Challenge 2020. In that release, the training folder contains multiple sources:

- `cpsc_2018/`
- `cpsc_2018_extra/`
- `georgia/`
- `ptb/`
- `ptb-xl/`
- `st_petersburg_incart/`

This repository currently uses only:

- `cpsc_2018/`
- `georgia/`
- `ptb-xl/`

So if you download the full release, keep those three directories and place them under `data/raw/` using the nested layout shown above.

### Browser download route

If you do not want to use `wget`, open the Challenge 2020 files index in a browser and download the three dataset folders manually:

- https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/
- https://physionet.org/files/challenge-2020/1.0.2/training/georgia/
- https://physionet.org/files/challenge-2020/1.0.2/training/ptb-xl/

Then arrange them so the final local paths are:

- `data/raw/cpsc_2018/cpsc_2018/`
- `data/raw/georgia/georgia/`
- `data/raw/ptb-xl/ptb-xl/`

### Important note about PTB-XL

The standalone PTB-XL release on PhysioNet uses a different structure with `records100/` and `records500/` directories plus metadata CSV files. That release is not drop-in compatible with this repository's current preprocessing script because the code searches for `g*/*.hea` inside the Challenge-style folder layout.

If you want to use the standalone PTB-XL release directly, you would need to change:

- the raw-data path configuration in [ecg_lead_reduction/core/config.py](/Users/yusufyusuf/Desktop/DISS_final/ecg_lead_reduction/core/config.py)
- the file discovery logic in [ecg_lead_reduction/data/preprocess.py](/Users/yusufyusuf/Desktop/DISS_final/ecg_lead_reduction/data/preprocess.py)

Important assumptions in the current pipeline:

- Records must contain the standard 12 leads.
- Signals are resampled implicitly by using records that already match the configured sampling setup.
- Each ECG is padded or truncated to 5000 samples.
- Labels are extracted from the `# Dx:` line in the WFDB header.
- SNOMED codes are mapped to short diagnosis abbreviations in [ecg_lead_reduction/core/config.py](/Users/yusufyusuf/Desktop/DISS_final/ecg_lead_reduction/core/config.py).
- Only classes with at least 1800 examples are kept.

## Environment setup

The code uses Python 3.10+ syntax. A local virtual environment is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Core dependencies are:

- PyTorch
- NumPy
- SciPy
- scikit-learn
- pandas
- matplotlib
- WFDB
- tqdm

## Quick start

Run all commands from the repository root.

### 1. Preprocess the raw ECG data

```bash
python scripts/preprocess.py
```

This creates:

- `data/processed/combined.npz`

Useful option:

```bash
python scripts/preprocess.py --force
```

Use `--force` to rebuild the processed cache even if it already exists.

### 2. Train one model configuration

```bash
python scripts/train.py --arch resnet --lead-config 12-lead
python scripts/train.py --arch cnn_lstm --lead-config 1-lead
```

Valid architectures:

- `resnet`
- `cnn_lstm`

Valid lead configurations:

- `12-lead`
- `6-lead`
- `4-lead`
- `3-lead`
- `2-lead`
- `1-lead`

Training outputs are written to:

- `artifacts/checkpoints/best_<arch>_<lead-config>.pt`
- `artifacts/results/results_<arch>_<lead-config>.json`

### 3. Run the full experiment grid

```bash
python scripts/run_experiments.py
```

This runs all architecture x lead-configuration combinations, builds summary tables, and generates comparison figures.

Useful variants:

```bash
python scripts/run_experiments.py --arch resnet
python scripts/run_experiments.py --lead-config 6-lead
python scripts/run_experiments.py --skip-training
python scripts/run_experiments.py --parallel 4
```

Notes:

- `--skip-training` evaluates existing checkpoints without retraining.
- `--parallel N` runs multiple experiments in parallel.
- On Apple Silicon, the code falls back to CPU workers for parallel runs to avoid MPS contention.

### 4. Regenerate XAI outputs only

```bash
python scripts/xai.py
```

This loads saved checkpoints and creates Integrated Gradients lead-importance outputs under `artifacts/figures/xai/`.

## Lead configurations used in the study

The current lead subsets are defined in [ecg_lead_reduction/core/config.py](/Users/yusufyusuf/Desktop/DISS_final/ecg_lead_reduction/core/config.py):

| Configuration | Leads used |
| --- | --- |
| 12-lead | I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6 |
| 6-lead | I, II, III, aVR, aVL, aVF |
| 4-lead | I, II, V1, V5 |
| 3-lead | I, II, III |
| 2-lead | I, II |
| 1-lead | II |

## Preprocessing pipeline

The preprocessing stage in [ecg_lead_reduction/data/preprocess.py](/Users/yusufyusuf/Desktop/DISS_final/ecg_lead_reduction/data/preprocess.py) does the following:

1. Scans all supported datasets for WFDB header files.
2. Reads diagnosis codes from each header.
3. Maps supported SNOMED labels to project-specific abbreviations.
4. Filters out unsupported and underrepresented classes.
5. Loads each signal and reorders channels into standard 12-lead order.
6. Replaces NaNs with zeros when needed.
7. Pads or truncates every recording to 5000 samples.
8. Applies a 0.5 to 47.0 Hz fourth-order band-pass filter.
9. Normalises each lead independently.
10. Saves signals, labels, class names, record IDs, and class count into a compressed `.npz` archive.

## Data loading and split strategy

The dataset utilities live in [ecg_lead_reduction/data/dataset.py](/Users/yusufyusuf/Desktop/DISS_final/ecg_lead_reduction/data/dataset.py).

Key details:

- Splits are stratified using `labels.argmax(axis=1)`.
- Test split: 10% of the full dataset.
- Validation split: 12.5% of the remaining train/validation pool.
- Effective split ratio is approximately 78.75% train, 11.25% validation, 10% test.
- Training data augmentation adds Gaussian noise, amplitude scaling, and temporal shifts.
- Class imbalance is handled through `BCEWithLogitsLoss` with positive-class weights.

## Models

The model definitions are in [ecg_lead_reduction/models/model.py](/Users/yusufyusuf/Desktop/DISS_final/ecg_lead_reduction/models/model.py).

### ResNet

- 1D convolutional residual network.
- Squeeze-and-Excitation blocks enabled by default.
- Base filters: 32.
- Number of residual blocks: 4.
- Kernel size: 15.

### CNN-LSTM

- Four-stage 1D convolutional front-end.
- Filter sizes: 32, 64, 128, 256.
- Bidirectional LSTM back-end.
- Hidden size: 128.
- Number of LSTM layers: 2.

## Training configuration

The default training loop is implemented in [ecg_lead_reduction/training/train.py](/Users/yusufyusuf/Desktop/DISS_final/ecg_lead_reduction/training/train.py).

Current defaults:

- Batch size: 64
- Epochs: 50
- Learning rate: 1e-3
- Weight decay: 1e-4
- Early stopping patience: 10 epochs
- Optimiser: AdamW
- LR scheduler: ReduceLROnPlateau
- Dropout: 0.3
- Random seed: 42
- Prediction threshold: 0.5

The device is selected automatically in this order:

1. CUDA
2. Apple MPS
3. CPU

## Evaluation outputs

The evaluation code is in [ecg_lead_reduction/evaluation/evaluate.py](/Users/yusufyusuf/Desktop/DISS_final/ecg_lead_reduction/evaluation/evaluate.py).

Saved metrics include:

- Macro AUROC
- Macro F1
- Sample-wise F1
- Exact-match accuracy
- Macro specificity
- Per-class AUROC
- Per-class F1, precision, recall, and specificity

The experiment runner also writes summary files:

- `artifacts/results/results_summary.json`
- `artifacts/results/results_summary.csv`

## Figures and XAI outputs

Comparison figures are generated by [ecg_lead_reduction/analysis/visualise.py](/Users/yusufyusuf/Desktop/DISS_final/ecg_lead_reduction/analysis/visualise.py).

Typical outputs include:

- `artifacts/figures/comparison_auroc_macro.png`
- `artifacts/figures/comparison_f1_macro.png`
- `artifacts/figures/per_class_auroc_heatmap.png`
- `artifacts/figures/training_curves.png`
- `artifacts/figures/lead_degradation.png`

XAI outputs are generated by [ecg_lead_reduction/analysis/xai.py](/Users/yusufyusuf/Desktop/DISS_final/ecg_lead_reduction/analysis/xai.py) and include:

- Per-run lead-importance heatmaps
- Cross-configuration comparison heatmaps
- JSON exports of lead-importance scores for each class

## Main code entry points

Use the wrappers in `scripts/` for normal command-line usage:

- [scripts/preprocess.py](/Users/yusufyusuf/Desktop/DISS_final/scripts/preprocess.py)
- [scripts/train.py](/Users/yusufyusuf/Desktop/DISS_final/scripts/train.py)
- [scripts/run_experiments.py](/Users/yusufyusuf/Desktop/DISS_final/scripts/run_experiments.py)
- [scripts/xai.py](/Users/yusufyusuf/Desktop/DISS_final/scripts/xai.py)

The implementation lives under `ecg_lead_reduction/`:

- `core/` for configuration
- `data/` for preprocessing and dataset loading
- `models/` for architectures
- `training/` for optimisation and checkpoints
- `evaluation/` for metrics and comparison tables
- `analysis/` for plots and explainability
- `experiments/` for orchestration

## Changing defaults

Most project-level settings live in [ecg_lead_reduction/core/config.py](/Users/yusufyusuf/Desktop/DISS_final/ecg_lead_reduction/core/config.py), including:

- data paths
- sampling rate and signal length
- label mappings
- minimum class count
- lead subsets
- architecture list
- training hyperparameters
- output locations

If you want to change the study setup, this is the first place to edit.

## Reproducibility notes

- The code sets a fixed random seed of 42 for Python, NumPy, and PyTorch.
- Data splitting is deterministic for a fixed processed dataset.
- Parallel experiment execution changes runtime behaviour but is designed to preserve the same experiment definitions.

## Common issues

### `Preprocessed data not found`

Run:

```bash
python scripts/preprocess.py
```

### Raw dataset directory not found

Check that the folder names under `data/raw/` match the paths configured in [ecg_lead_reduction/core/config.py](/Users/yusufyusuf/Desktop/DISS_final/ecg_lead_reduction/core/config.py).

### Want to rerun evaluation from saved checkpoints only

Use:

```bash
python scripts/run_experiments.py --skip-training
```

### Want to regenerate only the figures/XAI outputs

Run:

```bash
python scripts/xai.py
```

or rerun the experiment summary/plot stage with saved checkpoints:

```bash
python scripts/run_experiments.py --skip-training
```
