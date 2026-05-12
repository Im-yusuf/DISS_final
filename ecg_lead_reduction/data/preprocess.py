"""Preprocess Challenge-style WFDB ECG folders into a single compressed cache.

The CLI performs two passes over the raw headers. The first pass decides which
diagnosis classes have enough examples to keep; the second pass loads matching
signals, standardises them, builds multi-hot labels, and writes `combined.npz`.
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

from ecg_lead_reduction.core.config import (
    RAW_DATA_DIRS, PROCESSED_DATA_DIR, PROCESSED_NPZ,
    SIGNAL_LENGTH, SAMPLING_RATE,
    LEAD_NAMES_12, SNOMED_TO_ABBR, MIN_CLASS_COUNT, RANDOM_SEED,
    FILTER_LOW_HZ, FILTER_HIGH_HZ, FILTER_ORDER,
)


def set_seed(seed: int) -> None:
    """Seed Python and NumPy for deterministic preprocessing choices."""

    random.seed(seed)
    np.random.seed(seed)


def parse_header_dx(header_path: Path) -> list[str]:
    """Extract SNOMED diagnosis codes from the `# Dx:` line of a WFDB header.

    Missing or empty diagnosis lines return an empty list so callers can skip
    unsupported records cleanly.
    """

    diagnosis_codes: list[str] = []
    with open(header_path, "r") as file_handle:
        for line in file_handle:
            if line.startswith("# Dx:"):
                diagnosis_text = line.strip().split("# Dx:")[1].strip()
                diagnosis_codes = [raw_code.strip() for raw_code in diagnosis_text.split(",") if raw_code.strip()]
                break
    return diagnosis_codes


def get_lead_reorder_indices(wfdb_record: wfdb.Record) -> list[int] | None:
    """Return indices that reorder a WFDB record into the standard 12-lead order.

    Some WFDB records list the same leads in different orders. Returning `None`
    marks records that cannot provide the complete standard 12-lead set.
    """

    signal_names = [s.strip() for s in wfdb_record.sig_name]
    lead_order_indices: list[int] = []
    for lead_name in LEAD_NAMES_12:
        if lead_name in signal_names:
            lead_order_indices.append(signal_names.index(lead_name))
        else:
            return None
    return lead_order_indices


def pad_or_truncate(ecg_signal: np.ndarray, target_length: int) -> np.ndarray:
    """Force an ECG signal to the configured sample length along the time axis.

    Signals longer than `target_length` are cropped from the start; shorter
    signals are zero-padded at the end so all model inputs share one shape.
    """

    signal_length = ecg_signal.shape[1]
    if signal_length >= target_length:
        return ecg_signal[:, :target_length]
    padding_width = target_length - signal_length
    return np.pad(ecg_signal, ((0, 0), (0, padding_width)),
                  mode="constant", constant_values=0)


def bandpass_filter(ecg_signal: np.ndarray,
                    sampling_rate_hz: int = SAMPLING_RATE,
                    low_cut_hz: float = FILTER_LOW_HZ,
                    high_cut_hz: float = FILTER_HIGH_HZ,
                    filter_order: int = FILTER_ORDER) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter independently per lead.

    The default passband keeps the ECG morphology used by the classifiers while
    reducing baseline wander and high-frequency noise.
    """

    nyquist_hz = sampling_rate_hz / 2.0
    filter_sos = butter(filter_order, [low_cut_hz / nyquist_hz, high_cut_hz / nyquist_hz],
                 btype="band", output="sos")


    # Filter each lead separately to avoid mixing information across channels.
    filtered_signal = np.empty_like(ecg_signal)
    for channel_index in range(ecg_signal.shape[0]):
        filtered_signal[channel_index] = sosfiltfilt(filter_sos, ecg_signal[channel_index])
    return filtered_signal


def normalize_signal(ecg_signal: np.ndarray) -> np.ndarray:
    """Standardise each lead to zero mean and unit variance when possible.

    Flat leads are set to zero to avoid amplifying numerical noise when the
    standard deviation is effectively zero.
    """

    for channel_index in range(ecg_signal.shape[0]):
        mean_value  = ecg_signal[channel_index].mean()
        std_value = ecg_signal[channel_index].std()
        if std_value > 1e-6:
            ecg_signal[channel_index] = (ecg_signal[channel_index] - mean_value) / std_value
        else:
            ecg_signal[channel_index] = 0.0
    return ecg_signal


def main() -> None:
    """Run the preprocessing CLI and write `combined.npz` to processed data."""

    parser = argparse.ArgumentParser(
        description="Preprocess multi-dataset 12-lead ECG data into .npz cache")
    parser.add_argument("--force", action="store_true",
                        help="Force reprocessing even if .npz already exists")
    cli_args = parser.parse_args()

    set_seed(RANDOM_SEED)

    processed_path = PROCESSED_DATA_DIR / PROCESSED_NPZ

    if processed_path.exists() and not cli_args.force:
        print(f"Processed file already exists: {processed_path}")
        print("Use --force to reprocess.")
        return


    header_paths: list[Path] = []
    for dataset_name, dataset_dir in RAW_DATA_DIRS.items():
        if not dataset_dir.exists():
            print(f"  [{dataset_name}] Directory not found: {dataset_dir} — skipping.")
            continue
        dataset_headers = sorted(dataset_dir.glob("g*/*.hea"))
        print(f"  [{dataset_name}] Found {len(dataset_headers)} records in {dataset_dir}")
        header_paths.extend(dataset_headers)

    print(f"\nTotal records across all datasets: {len(header_paths)}")

    if len(header_paths) == 0:
        print("ERROR: No .hea files found. Check RAW_DATA_DIRS in config.py")
        sys.exit(1)


    print("\nPass 1: Scanning diagnosis codes …")
    diagnosis_records: list[tuple[Path, list[str]]] = []
    diagnosis_counts: Counter = Counter()
    skipped_without_known_dx = 0

    for header_path in tqdm(header_paths, desc="Scanning"):
        snomed_codes = parse_header_dx(header_path)


        # Collapse duplicate SNOMED mappings so each class appears once per record.
        diagnosis_labels: set[str] = set()
        for code in snomed_codes:
            if code in SNOMED_TO_ABBR:
                diagnosis_labels.add(SNOMED_TO_ABBR[code])

        if not diagnosis_labels:
            skipped_without_known_dx += 1
            continue

        diagnosis_records.append((header_path, sorted(diagnosis_labels)))
        for diagnosis_label in diagnosis_labels:
            diagnosis_counts[diagnosis_label] += 1

    print(f"\nRecords with known diagnoses: {len(diagnosis_records)}")
    print(f"Records skipped (no known Dx):  {skipped_without_known_dx}")
    print(f"\nDiagnosis distribution (all known):")
    for abbr, class_count in diagnosis_counts.most_common():
        print(f"  {abbr:>8s}: {class_count:>5d}")


    # Keep only classes with enough support for stable training/evaluation.
    class_names = sorted(
        [abbr for abbr, class_count in diagnosis_counts.items()
         if class_count >= MIN_CLASS_COUNT]
    )
    num_classes = len(class_names)
    class_to_index = {class_name: class_index for class_index, class_name in enumerate(class_names)}

    print(f"\nSelected {num_classes} classes (min count >= {MIN_CLASS_COUNT}):")
    for class_name in class_names:
        print(f"  {class_name}: {diagnosis_counts[class_name]}")


    selected_records: list[tuple[Path, list[str]]] = []
    for header_path, diagnosis_labels in diagnosis_records:
        selected_labels = [diagnosis_label for diagnosis_label in diagnosis_labels if diagnosis_label in class_to_index]
        if selected_labels:
            selected_records.append((header_path, selected_labels))

    print(f"\nRecords after filtering: {len(selected_records)}")


    print("\nPass 2: Loading & processing signals …")
    signal_arrays: list[np.ndarray] = []
    label_arrays:  list[np.ndarray] = []
    record_ids:   list[str]        = []
    signal_lengths:      list[int]        = []
    failed_records:       list[tuple[str, str]] = []

    for header_path, diagnosis_labels in tqdm(selected_records, desc="Loading"):
        record_path = str(header_path.with_suffix(""))


        try:
            wfdb_record = wfdb.rdrecord(record_path)
        except Exception as error:
            failed_records.append((header_path.stem, str(error)))
            continue


        if wfdb_record.n_sig != 12:
            failed_records.append((header_path.stem,
                           f"Expected 12 leads, got {wfdb_record.n_sig}"))
            continue

        lead_indices = get_lead_reorder_indices(wfdb_record)
        if lead_indices is None:
            failed_records.append((header_path.stem, "Missing standard leads"))
            continue


        # WFDB loads samples x leads; models expect leads x samples.
        ecg_signal = wfdb_record.p_signal[:, lead_indices].T.astype(np.float32)


        if np.isnan(ecg_signal).any():
            ecg_signal = np.nan_to_num(ecg_signal, nan=0.0)

        signal_lengths.append(ecg_signal.shape[1])


        ecg_signal = pad_or_truncate(ecg_signal, SIGNAL_LENGTH)


        ecg_signal = bandpass_filter(ecg_signal)


        ecg_signal = normalize_signal(ecg_signal)


        # Multi-label targets are stored as one binary vector per record.
        label_vector = np.zeros(num_classes, dtype=np.float32)
        for abbr in diagnosis_labels:
            if abbr in class_to_index:
                label_vector[class_to_index[abbr]] = 1.0

        signal_arrays.append(ecg_signal)
        label_arrays.append(label_vector)
        record_ids.append(header_path.stem)


    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    signal_array        = np.array(signal_arrays, dtype=np.float32)
    label_array         = np.array(label_arrays,  dtype=np.float32)
    record_ids_array = np.array(record_ids,   dtype="U20")
    class_names_array = np.array(class_names, dtype="U20")

    np.savez_compressed(
        processed_path,
        signals=signal_array,
        labels=label_array,
        record_ids=record_ids_array,
        class_names=class_names_array,
        num_classes=num_classes,
    )


    signal_lengths_array = np.array(signal_lengths)
    print(f"\n{'=' * 60}")
    print(f"  Saved to {processed_path}")
    print(f"  Total records : {len(signal_array)}")
    print(f"  Signal shape  : {signal_array.shape}")
    print(f"  Labels shape  : {label_array.shape}")
    print(f"  Classes ({num_classes}): {class_names}")
    print(f"  Signal lengths: min={signal_lengths_array.min()}, "
          f"max={signal_lengths_array.max()}, "
          f"median={int(np.median(signal_lengths_array))}")
    print(f"\n  Label distribution:")
    for class_index, class_name in enumerate(class_names):
        class_count = int(label_array[:, class_index].sum())
        percentage = 100.0 * class_count / len(label_array)
        print(f"    {class_name:>8s}: {class_count:>5d}  ({percentage:5.1f}%)")

    multi_label_count = int((label_array.sum(axis=1) > 1).sum())
    print(f"\n  Multi-label records: {multi_label_count} "
          f"({100.0 * multi_label_count / len(label_array):.1f}%)")

    if failed_records:
        print(f"\n  Failed records ({len(failed_records)}):")
        for record_id, error_message in failed_records[:10]:
            print(f"    {record_id}: {error_message}")
        if len(failed_records) > 10:
            print(f"    … and {len(failed_records) - 10} more")

    print(f"\n  Done!")


if __name__ == "__main__":
    main()
