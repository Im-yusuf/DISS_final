import torch
from pathlib import Path


PROJECT_ROOT       = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR      = PROJECT_ROOT / "artifacts"
RAW_DATA_ROOT      = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR        = ARTIFACTS_DIR / "results"
CHECKPOINTS_DIR    = ARTIFACTS_DIR / "checkpoints"
FIGURES_DIR        = ARTIFACTS_DIR / "figures"


RAW_DATA_DIRS = {
    'cpsc_2018': RAW_DATA_ROOT / "cpsc_2018" / "cpsc_2018",
    'georgia':   RAW_DATA_ROOT / "georgia"  / "georgia",
    'ptb_xl':    RAW_DATA_ROOT / "ptb-xl"   / "ptb-xl",
}


RAW_DATA_DIR = RAW_DATA_DIRS['cpsc_2018']


PROCESSED_NPZ = "combined.npz"


SAMPLING_RATE = 500
SIGNAL_LENGTH = 5000


FILTER_LOW_HZ  = 0.5
FILTER_HIGH_HZ = 47.0
FILTER_ORDER   = 4


LEAD_NAMES_12 = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


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


LEAD_CONFIGS = {
    '12-lead': list(range(12)),
    '6-lead':  [0, 1, 2, 3, 4, 5],
    '4-lead':  [0, 1, 6, 10],
    '3-lead':  [0, 1, 2],
    '2-lead':  [0, 1],
    '1-lead':  [1],
}


TEST_SPLIT  = 0.10
VAL_SPLIT   = 0.125
RANDOM_SEED = 42


ARCHITECTURES = ['resnet', 'cnn_lstm']

BATCH_SIZE    = 64
NUM_EPOCHS    = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 10
DROPOUT_RATE  = 0.3
NUM_WORKERS   = 0


RESNET_BASE_FILTERS = 32
RESNET_NUM_BLOCKS   = 4
RESNET_KERNEL_SIZE  = 15
USE_SE_BLOCK        = True
SE_REDUCTION        = 16


CNN_LSTM_FILTERS = [32, 64, 128, 256]
CNN_LSTM_KERNEL  = 15
LSTM_HIDDEN      = 128
LSTM_LAYERS      = 2
LSTM_DROPOUT     = 0.3


LABEL_THRESHOLD = 0.5


DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps'  if torch.backends.mps.is_available() else
    'cpu'
)
