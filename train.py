import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (
    BATCH_SIZE, CHECKPOINTS_DIR, DEVICE, LEAD_CONFIGS,
    LEARNING_RATE, NUM_EPOCHS, PATIENCE,
    RANDOM_SEED, RESULTS_DIR, TEST_SPLIT, VAL_SPLIT, WEIGHT_DECAY,
)
from dataset import compute_pos_weight, get_dataloaders, set_seed
from evaluate import compute_metrics
from model import build_model


class EarlyStopping:

    def __init__(self, patience: int, checkpoint_path: str | Path):
        self.patience = patience
        self.checkpoint_path = Path(checkpoint_path)
        self.best_score: float = -1.0
        self.counter: int = 0
        self.best_epoch: int = -1

    def step(self, score: float, model: nn.Module, epoch: int) -> bool:
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.checkpoint_path)
            return False
        self.counter += 1
        return self.counter >= self.patience


def _run_epoch(model: nn.Module,
               data_loader,
               loss_function: nn.Module,
               optimizer=None,
               device: torch.device = DEVICE):
    training_mode = optimizer is not None
    model.train() if training_mode else model.eval()

    total_loss = 0.0
    label_batches: list[np.ndarray] = []
    logit_batches: list[np.ndarray] = []
    batch_count = 0

    grad_context = torch.enable_grad() if training_mode else torch.no_grad()
    with grad_context:
        for signal_batch, label_batch in data_loader:
            signal_batch = signal_batch.to(device)
            label_batch  = label_batch.to(device)

            batch_logits = model(signal_batch)
            batch_loss   = loss_function(batch_logits, label_batch)

            if training_mode:
                optimizer.zero_grad()
                batch_loss.backward()


                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += batch_loss.item()
            batch_count  += 1
            label_batches.append(label_batch.cpu().numpy())
            logit_batches.append(batch_logits.cpu().detach().numpy())

    mean_loss   = total_loss / max(batch_count, 1)
    labels_array = np.concatenate(label_batches)
    logits_array = np.concatenate(logit_batches)
    return mean_loss, labels_array, logits_array


def train_model(arch: str, lead_config: str, tag: str | None = None,
                device: torch.device = DEVICE) -> dict:
    set_seed(RANDOM_SEED)

    run_name = tag or f"{arch}_{lead_config}"
    print(f"\n{'=' * 70}")
    print(f"  Training: {run_name}")
    print(f"  Architecture: {arch} | Lead config: {lead_config} | Device: {device}")
    print(f"{'=' * 70}")


    (train_loader, val_loader, test_loader,
     class_names, num_classes, train_labels) = get_dataloaders(
        lead_config, BATCH_SIZE, TEST_SPLIT, VAL_SPLIT, RANDOM_SEED,
    )


    positive_weights = compute_pos_weight(train_labels).to(device)
    print(f"  Classes ({num_classes}): {class_names}")
    print(f"  Pos weights: {positive_weights.cpu().numpy().round(2)}")


    num_leads    = len(LEAD_CONFIGS[lead_config])
    model        = build_model(arch, num_leads, num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")


    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weights)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE,
                      weight_decay=WEIGHT_DECAY)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                  patience=5)


    checkpoint_path  = CHECKPOINTS_DIR / f"best_{run_name}.pt"
    early_stopper = EarlyStopping(patience=PATIENCE, checkpoint_path=checkpoint_path)


    training_start_time = time.time()
    history: list[dict] = []

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time()


        train_loss, _, _ = _run_epoch(
            model, train_loader, loss_function, optimizer, device)


        val_loss, val_labels, val_logits = _run_epoch(
            model, val_loader, loss_function, device=device)
        validation_metrics = compute_metrics(val_labels, val_logits, class_names)
        validation_auroc   = validation_metrics["auroc_macro"]
        validation_f1      = validation_metrics["f1_macro"]


        lr_scheduler.step(validation_auroc)
        learning_rate = optimizer.param_groups[0]["lr"]

        epoch_elapsed = time.time() - epoch_start_time

        history_entry = {
            "epoch":           epoch,
            "train_loss":      round(train_loss, 4),
            "val_loss":        round(val_loss, 4),
            "val_auroc_macro": round(validation_auroc, 4),
            "val_f1_macro":    round(validation_f1, 4),
            "lr":              learning_rate,
            "epoch_time_s":    round(epoch_elapsed, 1),
        }
        history.append(history_entry)

        should_stop = early_stopper.step(validation_auroc, model, epoch)
        new_best = (early_stopper.counter == 0)
        print(f"  Epoch {epoch:>3d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}  AUROC: {validation_auroc:.4f}  "
              f"F1: {validation_f1:.4f} | "
              f"LR: {learning_rate:.2e} | {epoch_elapsed:.1f}s"
              f"{'  *' if new_best else ''}")

        if should_stop:
            print(f"\n  Early stopping at epoch {epoch}.  "
                  f"Best val AUROC: {early_stopper.best_score:.4f} "
                  f"at epoch {early_stopper.best_epoch}")
            break

    training_duration = time.time() - training_start_time


    print(f"\n  Loading best checkpoint (epoch {early_stopper.best_epoch}) …")
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True))

    _, test_labels, test_logits = _run_epoch(
        model, test_loader, loss_function, device=device)
    test_results = compute_metrics(test_labels, test_logits, class_names)


    run_results = {
        "arch":            arch,
        "lead_config":     lead_config,
        "num_leads":       num_leads,
        "num_classes":     num_classes,
        "class_names":     class_names,
        "total_params":    total_params,
        "training_time_s": round(training_duration, 1),
        "best_epoch":      early_stopper.best_epoch,
        "best_val_auroc":  round(early_stopper.best_score, 4),
        **test_results,
        "history":         history,
    }


    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"results_{run_name}.json"
    with open(output_path, "w") as f:
        json.dump(_make_json_serialisable(run_results), f, indent=2)
    print(f"  Results saved to {output_path}")


    print(f"\n  {'=' * 50}")
    print(f"  TEST RESULTS: {run_name}")
    print(f"  {'=' * 50}")
    print(f"  AUROC Macro  : {test_results['auroc_macro']:.4f}")
    print(f"  F1 Macro     : {test_results['f1_macro']:.4f}")
    print(f"  F1 Samples   : {test_results['f1_samples']:.4f}")
    print(f"  Exact Match  : {test_results['exact_match_accuracy']:.4f}")
    print(f"  Specificity  : {test_results.get('specificity_macro', 0):.4f}")
    print(f"\n  Per-class AUROC:")
    for class_name in class_names:
        print(f"    {class_name:>8s}: {test_results.get(f'auroc_{class_name}', 0):.4f}", end="  ")
    print()

    return run_results


def _make_json_serialisable(value):
    if isinstance(value, dict):
        return {key: _make_json_serialisable(item_value) for key, item_value in value.items()}
    if isinstance(value, list):
        return [_make_json_serialisable(item_value) for item_value in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a single ECG classification model")
    parser.add_argument("--arch", type=str, required=True,
                        choices=["resnet", "cnn_lstm"])
    parser.add_argument("--lead-config", type=str, required=True,
                        choices=list(LEAD_CONFIGS.keys()))
    cli_args = parser.parse_args()
    train_model(cli_args.arch, cli_args.lead_config)
