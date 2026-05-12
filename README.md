# ECG Lead Reduction for Multi-label ECG Classification

This repository contains the full experiment pipeline for studying how much 12-lead ECG input can be reduced before multi-label classification performance degrades materially. It preprocesses Challenge-style WFDB ECG records, trains two neural architectures across multiple lead subsets, evaluates the saved checkpoints, and generates comparison and explainability artifacts.

The project is structured as an importable Python package. Run commands from the repository root with `python -m ...`.

## Study Snapshot

- Data sources: CPSC 2018, Georgia, and PTB-XL folders from the PhysioNet/CinC Challenge 2020 training release.
- Task: multi-label ECG diagnosis classification from fixed-length 12-lead signal windows.
- Input format: tensors shaped `(batch, leads, samples)` after preprocessing to 5000 samples.
- Lead subsets: 12, 6, 4, 3, 2, and 1 lead.
- Models: compact 1D ResNet with optional Squeeze-and-Excitation blocks, and CNN-LSTM with a bidirectional recurrent back end.
- Metrics: macro AUROC, macro F1, sample-wise F1, exact-match accuracy, specificity, and per-class metrics.
- Explainability: Integrated Gradients lead-importance scores aggregated by class and lead configuration.

## Current Result Snapshot

The saved artifacts in this workspace record the following headline results. Full per-class metrics are available in [artifacts/results/results_summary.csv](artifacts/results/results_summary.csv) and [artifacts/results/results_summary.json](artifacts/results/results_summary.json).

| Architecture | 12-lead AUROC | 12-lead F1 | Best reduced setup | Best reduced AUROC | Best reduced F1 | 1-lead AUROC | 1-lead F1 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| CNN-LSTM | 0.9611 | 0.7030 | 4-lead | 0.9579 | 0.6826 | 0.9399 | 0.6493 |
| ResNet | 0.9525 | 0.6460 | 4-lead | 0.9514 | 0.6451 | 0.9380 | 0.6027 |

In the current run, the CNN-LSTM gives the strongest 12-lead result, while the 4-lead configuration is the closest reduced setting for both architectures. Single-lead Lead II remains informative but shows a clearer F1 drop.

## Repository Layout

```text
.
|- artifacts/
|  |- checkpoints/        # trained model weights
|  |- figures/            # comparison and XAI figures
|  \- results/            # per-run JSON plus summary CSV/JSON
|- data/
|  |- processed/          # generated combined.npz cache
|  \- raw/                # local PhysioNet Challenge 2020 data
|- ecg_lead_reduction/
|  |- analysis/           # plotting and Integrated Gradients
|  |- core/               # paths, labels, lead sets, hyperparameters
|  |- data/               # preprocessing, datasets, dataloaders
|  |- evaluation/         # metrics and summary tables
|  |- experiments/        # full experiment-grid runner
|  |- models/             # ResNet and CNN-LSTM definitions
|  \- training/           # training loop and checkpointing
|- requirements.txt
\- README.md
```

The small `__init__.py` files are intentionally kept so each folder is treated as a normal Python package across Python tools, editors, and `python -m` entry points.

## Environment Setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Core dependencies include PyTorch, WFDB, NumPy, SciPy, scikit-learn, pandas, matplotlib, and tqdm.

## Data Setup

The preprocessing code expects the PhysioNet/CinC Challenge 2020 training-folder layout, not the standalone PTB-XL release layout.

Expected local structure:

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

Download the three folders used by this project:

```bash
mkdir -p data/raw/cpsc_2018 data/raw/georgia data/raw/ptb-xl

wget -r -N -c -np -nH --cut-dirs=4 -P data/raw/cpsc_2018 \
  https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/

wget -r -N -c -np -nH --cut-dirs=4 -P data/raw/georgia \
  https://physionet.org/files/challenge-2020/1.0.2/training/georgia/

wget -r -N -c -np -nH --cut-dirs=4 -P data/raw/ptb-xl \
  https://physionet.org/files/challenge-2020/1.0.2/training/ptb-xl/
```

Reference pages:

- Challenge 2020 dataset: <https://physionet.org/content/challenge-2020/1.0.2/>
- Challenge 2020 training files: <https://physionet.org/files/challenge-2020/1.0.2/training/>
- Standalone PTB-XL reference: <https://physionet.org/content/ptb-xl/1.0.3/>

Important: the standalone PTB-XL release uses `records100/` and `records500/` directories. It is not drop-in compatible with this repository because [ecg_lead_reduction/data/preprocess.py](ecg_lead_reduction/data/preprocess.py) searches for Challenge-style `g*/*.hea` files under the paths configured in [ecg_lead_reduction/core/config.py](ecg_lead_reduction/core/config.py).

## Reproducing the Pipeline

### 1. Preprocess Raw ECG Records

```bash
python -m ecg_lead_reduction.data.preprocess
```

This creates `data/processed/combined.npz`. To rebuild the cache even when it already exists:

```bash
python -m ecg_lead_reduction.data.preprocess --force
```

The preprocessing stage:

- scans the configured raw dataset folders for WFDB headers,
- extracts SNOMED diagnosis codes from `# Dx:` header lines,
- maps supported SNOMED codes to compact diagnosis abbreviations,
- keeps classes with at least 1800 examples,
- loads records that contain the standard 12 leads,
- pads or truncates each ECG to 5000 samples,
- applies a 0.5-47.0 Hz fourth-order Butterworth band-pass filter,
- normalises each lead independently,
- writes signals, labels, record IDs, class names, and class count to a compressed `.npz` file.

The code assumes records are compatible with the configured 500 Hz sampling setup. It does not perform a separate resampling step.

### 2. Train One Configuration

```bash
python -m ecg_lead_reduction.training.train --arch resnet --lead-config 12-lead
python -m ecg_lead_reduction.training.train --arch cnn_lstm --lead-config 1-lead
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

Training saves:

- `artifacts/checkpoints/best_<arch>_<lead-config>.pt`
- `artifacts/results/results_<arch>_<lead-config>.json`

### 3. Run the Full Experiment Grid

```bash
python -m ecg_lead_reduction.experiments.run_experiments
```

Useful variants:

```bash
python -m ecg_lead_reduction.experiments.run_experiments --arch resnet
python -m ecg_lead_reduction.experiments.run_experiments --lead-config 6-lead
python -m ecg_lead_reduction.experiments.run_experiments --skip-training
python -m ecg_lead_reduction.experiments.run_experiments --parallel 4
```

`--skip-training` evaluates existing checkpoints and regenerates summaries/figures without retraining. `--parallel N` runs multiple experiments at once; on Apple Silicon, parallel workers use CPU to avoid MPS contention.

### 4. Regenerate XAI Outputs

```bash
python -m ecg_lead_reduction.analysis.xai
```

This loads saved checkpoints and writes Integrated Gradients outputs under `artifacts/figures/xai/`.

## Lead Configurations

Lead subsets are defined in [ecg_lead_reduction/core/config.py](ecg_lead_reduction/core/config.py).

| Configuration | Leads used |
| --- | --- |
| 12-lead | I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6 |
| 6-lead | I, II, III, aVR, aVL, aVF |
| 4-lead | I, II, V1, V5 |
| 3-lead | I, II, III |
| 2-lead | I, II |
| 1-lead | II |

## Models and Training Defaults

Model definitions live in [ecg_lead_reduction/models/model.py](ecg_lead_reduction/models/model.py).

ResNet:

- 1D convolutional input stem.
- Four pre-activation residual blocks.
- Squeeze-and-Excitation attention enabled by default.
- Global average pooling before the multi-label output layer.

CNN-LSTM:

- Four convolutional blocks with filters `[32, 64, 128, 256]`.
- Max-pooling after each convolutional block to shorten the sequence.
- Two-layer bidirectional LSTM with hidden size 128.
- Dropout before the final classification layer.

Training defaults are configured in [ecg_lead_reduction/core/config.py](ecg_lead_reduction/core/config.py):

| Setting | Value |
| --- | ---: |
| Batch size | 64 |
| Max epochs | 50 |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Early-stopping patience | 10 epochs |
| Dropout | 0.3 |
| Random seed | 42 |
| Prediction threshold | 0.5 |

The training loop uses AdamW, `BCEWithLogitsLoss` with positive-class weights, gradient clipping, `ReduceLROnPlateau`, and early stopping on validation macro AUROC. Device selection prefers CUDA, then Apple MPS, then CPU.

## Data Splits and Labels

[ecg_lead_reduction/data/dataset.py](ecg_lead_reduction/data/dataset.py) builds deterministic train/validation/test splits from the processed cache.

- Test split: 10% of the full processed dataset.
- Validation split: 12.5% of the remaining train/validation pool.
- Effective split: about 78.75% train, 11.25% validation, and 10% test.
- Stratification proxy: `labels.argmax(axis=1)` for each multi-label row.
- Training augmentation: small Gaussian noise, amplitude scaling, and temporal shifts.

The classes retained in a run depend on the processed data and `MIN_CLASS_COUNT`. The current saved result files include AF, IAVB, LAD, LAnFB, NSR, PAC, RBBB, SB, STD, STach, and TAb.

## Outputs

Primary generated files:

- Processed cache: `data/processed/combined.npz`
- Checkpoints: `artifacts/checkpoints/best_<arch>_<lead-config>.pt`
- Per-run metrics: `artifacts/results/results_<arch>_<lead-config>.json`
- Summary table: [artifacts/results/results_summary.csv](artifacts/results/results_summary.csv)
- Summary JSON: [artifacts/results/results_summary.json](artifacts/results/results_summary.json)

Comparison figures generated by [ecg_lead_reduction/analysis/visualise.py](ecg_lead_reduction/analysis/visualise.py):

- [artifacts/figures/comparison_auroc_macro.png](artifacts/figures/comparison_auroc_macro.png)
- [artifacts/figures/comparison_f1_macro.png](artifacts/figures/comparison_f1_macro.png)
- [artifacts/figures/per_class_auroc_heatmap.png](artifacts/figures/per_class_auroc_heatmap.png)
- [artifacts/figures/training_curves.png](artifacts/figures/training_curves.png)
- [artifacts/figures/lead_degradation.png](artifacts/figures/lead_degradation.png)

XAI artifacts generated by [ecg_lead_reduction/analysis/xai.py](ecg_lead_reduction/analysis/xai.py) include per-run lead-importance heatmaps, cross-configuration heatmaps, and JSON score exports under `artifacts/figures/xai/`.

## Code Map

- Configuration, paths, labels, lead subsets, and defaults: [ecg_lead_reduction/core/config.py](ecg_lead_reduction/core/config.py)
- Raw WFDB parsing and signal preprocessing: [ecg_lead_reduction/data/preprocess.py](ecg_lead_reduction/data/preprocess.py)
- Dataset splitting, augmentation, and dataloaders: [ecg_lead_reduction/data/dataset.py](ecg_lead_reduction/data/dataset.py)
- Model architectures: [ecg_lead_reduction/models/model.py](ecg_lead_reduction/models/model.py)
- Training loop, checkpointing, and single-run CLI: [ecg_lead_reduction/training/train.py](ecg_lead_reduction/training/train.py)
- Metric computation and summary tables: [ecg_lead_reduction/evaluation/evaluate.py](ecg_lead_reduction/evaluation/evaluate.py)
- Full experiment orchestration: [ecg_lead_reduction/experiments/run_experiments.py](ecg_lead_reduction/experiments/run_experiments.py)
- Comparison plotting: [ecg_lead_reduction/analysis/visualise.py](ecg_lead_reduction/analysis/visualise.py)
- Integrated Gradients lead-importance analysis: [ecg_lead_reduction/analysis/xai.py](ecg_lead_reduction/analysis/xai.py)

## Common Issues

### `Preprocessed data not found`

Run preprocessing first:

```bash
python -m ecg_lead_reduction.data.preprocess
```

### No `.hea` files are found

Check that the raw data uses the nested Challenge 2020 folder layout shown above. The configured paths are in [ecg_lead_reduction/core/config.py](ecg_lead_reduction/core/config.py).

### Existing checkpoints should be reused

Run the experiment runner in evaluation mode:

```bash
python -m ecg_lead_reduction.experiments.run_experiments --skip-training
```

### Only figures or XAI outputs need to be refreshed

For XAI outputs only:

```bash
python -m ecg_lead_reduction.analysis.xai
```

For summary tables plus comparison figures from saved checkpoints:

```bash
python -m ecg_lead_reduction.experiments.run_experiments --skip-training
```
