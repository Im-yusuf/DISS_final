"""Central configuration for paths, labels, lead subsets, and training defaults.

The constants in this module are shared across preprocessing, model building,
training, evaluation, and plotting. Keeping them in one place makes the
experiment pipeline reproducible and keeps the command-line entry points light.
All paths are resolved from the repository root so `python -m ...` works
consistently from the project root without hard-coding absolute machine paths.
"""

import torch
from pathlib import Path


# Repository-level paths.
# Generated artifacts live outside the package itself so runs do not mix code and outputs.
PROJECT_ROOT       = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR      = PROJECT_ROOT / "artifacts"
RAW_DATA_ROOT      = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR        = ARTIFACTS_DIR / "results"
CHECKPOINTS_DIR    = ARTIFACTS_DIR / "checkpoints"
FIGURES_DIR        = ARTIFACTS_DIR / "figures"


# Challenge 2020 training folders used by this project.
# Each path points at the nested folder structure described in the README.
RAW_DATA_DIRS = {
    'cpsc_2018': RAW_DATA_ROOT / "cpsc_2018" / "cpsc_2018",
    'georgia':   RAW_DATA_ROOT / "georgia"  / "georgia",
    'ptb_xl':    RAW_DATA_ROOT / "ptb-xl"   / "ptb-xl",
}


# Compatibility alias retained for code paths that expect a single raw-data root.
RAW_DATA_DIR = RAW_DATA_DIRS['cpsc_2018']


PROCESSED_NPZ = "combined.npz"


# Signal preprocessing parameters.
# All records are normalised onto the same temporal grid for batching.
SAMPLING_RATE = 500
SIGNAL_LENGTH = 5000


FILTER_LOW_HZ  = 0.5
FILTER_HIGH_HZ = 47.0
FILTER_ORDER   = 4


LEAD_NAMES_12 = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# SNOMED-CT diagnosis codes mapped to compact label names used in outputs.
# Several source codes intentionally collapse to the same clinical abbreviation.
SNOMED_TO_ABBR = {

    '164889003': 'AF',
    '164890007': 'AFL',
    '6374002':   'BBB',
    '426627000': 'Brady',
    '733534002': 'LBBB',
    '713427006': 'RBBB',
    '270492004': 'IAVB',
    '713426002': 'IRBBB',
    '39732003':  'LAD',
    '445118002': 'LAnFB',
    '164909002': 'LBBB',
    '251146004': 'LQRSV',
    '698252002': 'NSIVCB',
    '426783006': 'NSR',
    '284470004': 'PAC',
    '63593006':  'PAC',
    '10370003':  'PR',
    '365413008': 'PRWP',
    '427172004': 'PVC',
    '17338001':  'PVC',
    '164947007': 'LPR',
    '111975006': 'LQT',
    '164917005': 'QAb',
    '47665007':  'RAD',
    '59118001':  'RBBB',
    '427393009': 'SA',
    '426177001': 'SB',
    '427084000': 'STach',
    '164934002': 'TAb',
    '59931005':  'TInv',

    '429622005': 'STD',
    '164931005': 'STE',
}


MIN_CLASS_COUNT = 1800


# Lead subsets compared in the reduction experiments.
# Indices refer to positions in `LEAD_NAMES_12`.
LEAD_CONFIGS = {
    '12-lead': list(range(12)),
    '6-lead':  [0, 1, 2, 3, 4, 5],
    '4-lead':  [0, 1, 6, 10],
    '3-lead':  [0, 1, 2],
    '2-lead':  [0, 1],
    '1-lead':  [1],
}


# Dataset split and reproducibility controls.
TEST_SPLIT  = 0.10
VAL_SPLIT   = 0.125
RANDOM_SEED = 42


ARCHITECTURES = ['resnet', 'cnn_lstm']

# Shared training hyperparameters.
# These defaults are chosen for the experiment grid rather than per-model tuning.
BATCH_SIZE    = 64
NUM_EPOCHS    = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 10
DROPOUT_RATE  = 0.3
NUM_WORKERS   = 0


# ResNet defaults.
RESNET_BASE_FILTERS = 32
RESNET_NUM_BLOCKS   = 4
RESNET_KERNEL_SIZE  = 15
USE_SE_BLOCK        = True
SE_REDUCTION        = 16


# CNN-LSTM defaults.
CNN_LSTM_FILTERS = [32, 64, 128, 256]
CNN_LSTM_KERNEL  = 15
LSTM_HIDDEN      = 128
LSTM_LAYERS      = 2
LSTM_DROPOUT     = 0.3


# Probability threshold used when converting sigmoid outputs to binary predictions.
LABEL_THRESHOLD = 0.5


# Prefer GPU acceleration when available, with Apple Silicon MPS as the fallback.
# The training/evaluation entry points import this once and share the same device policy.
DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps'  if torch.backends.mps.is_available() else
    'cpu'
)
