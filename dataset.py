import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from config import (
    BATCH_SIZE,
    DEVICE,
    LEAD_CONFIGS,
    NUM_WORKERS,
    PROCESSED_DATA_DIR,
    PROCESSED_NPZ,
    RANDOM_SEED,
    SAMPLING_RATE,
    TEST_SPLIT,
    VAL_SPLIT,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_MAX_SHIFT_SAMPLES = SAMPLING_RATE // 2


class ECGDataset(Dataset):

    def __init__(self, signals: np.ndarray, labels: np.ndarray,
                 lead_config: str = "12-lead", augment: bool = False):
        lead_indices = LEAD_CONFIGS[lead_config]

        self.signals = signals[:, lead_indices, :]
        self.labels  = labels
        self.augment = augment

    def __getitem__(self, sample_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ecg_signal = self.signals[sample_index].copy()
        target_vector  = self.labels[sample_index]

        if self.augment:

            noise_sample  = np.random.normal(0.0, 0.02, ecg_signal.shape).astype(np.float32)
            ecg_signal = ecg_signal + noise_sample


            amplitude_scale  = np.random.uniform(0.9, 1.1)
            ecg_signal = ecg_signal * amplitude_scale


            time_shift = np.random.randint(-_MAX_SHIFT_SAMPLES,
                                       _MAX_SHIFT_SAMPLES + 1)
            ecg_signal = np.roll(ecg_signal, time_shift, axis=-1)

        return (torch.tensor(ecg_signal, dtype=torch.float32),
                torch.tensor(target_vector,  dtype=torch.float32))

    def __len__(self) -> int:
        return len(self.labels)


def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    positive_counts = labels.sum(axis=0)
    negative_counts = len(labels) - positive_counts
    positive_weights = np.where(positive_counts > 0, negative_counts / positive_counts, 1.0)
    positive_weights = np.clip(positive_weights, 0.5, 50.0)
    return torch.tensor(positive_weights, dtype=torch.float32)


def load_data(npz_path: Path | None = None):
    if npz_path is None:
        npz_path = PROCESSED_DATA_DIR / PROCESSED_NPZ

    archive_data = np.load(npz_path, allow_pickle=True)
    signals     = archive_data["signals"]
    labels      = archive_data["labels"]
    class_names = [str(class_label) for class_label in archive_data["class_names"]]
    num_classes = int(archive_data["num_classes"])
    record_ids  = archive_data["record_ids"]

    return signals, labels, class_names, num_classes, record_ids


def get_dataloaders(lead_config: str,
                    batch_size: int = BATCH_SIZE,
                    test_split: float = TEST_SPLIT,
                    val_split: float = VAL_SPLIT,
                    seed: int = RANDOM_SEED,
                    npz_path: Path | None = None):
    signals, labels, class_names, num_classes, _ = load_data(npz_path)

    sample_count = len(labels)
    sample_indices = np.arange(sample_count)


    stratification_labels = labels.argmax(axis=1)


    train_val_indices, test_indices = train_test_split(
        sample_indices, test_size=test_split, random_state=seed,
        stratify=stratification_labels,
    )


    train_stratification_labels = stratification_labels[train_val_indices]
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_split, random_state=seed,
        stratify=train_stratification_labels,
    )


    train_dataset = ECGDataset(signals[train_indices], labels[train_indices],
                               lead_config=lead_config, augment=True)
    val_dataset   = ECGDataset(signals[val_indices],   labels[val_indices],
                               lead_config=lead_config, augment=False)
    test_dataset  = ECGDataset(signals[test_indices],  labels[test_indices],
                               lead_config=lead_config, augment=False)


    pin_memory = (DEVICE.type == "cuda")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin_memory, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin_memory,
    )

    print(f"  Data loaders created [{lead_config}]:")
    print(f"    Train : {len(train_dataset):,} records")
    print(f"    Val   : {len(val_dataset):,} records")
    print(f"    Test  : {len(test_dataset):,} records")

    return (train_loader, val_loader, test_loader,
            class_names, num_classes, labels[train_indices])
