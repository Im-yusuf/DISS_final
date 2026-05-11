"""Metric computation and tabular comparison helpers for experiment outputs."""

import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ecg_lead_reduction.core.config import LABEL_THRESHOLD


def compute_metrics(labels: np.ndarray,
                    logits: np.ndarray,
                    class_names: list[str] | None = None,
                    threshold: float = LABEL_THRESHOLD) -> dict:
    """Compute multi-label classification metrics from labels and raw logits."""

    probabilities = sigmoid(logits)
    predictions = (probabilities >= threshold).astype(np.float32)

    num_classes = labels.shape[1]
    if class_names is None:
        class_names = [f"class_{class_index}" for class_index in range(num_classes)]

    metrics: dict = {}


    auroc_scores: list[float] = []
    for class_index in range(num_classes):
        positive_count = labels[:, class_index].sum()
        negative_count = len(labels) - positive_count
        if positive_count > 0 and negative_count > 0:
            auroc_score = roc_auc_score(labels[:, class_index], probabilities[:, class_index])
            metrics[f"auroc_{class_names[class_index]}"] = round(float(auroc_score), 4)
            auroc_scores.append(auroc_score)
        else:
            metrics[f"auroc_{class_names[class_index]}"] = 0.0

    metrics["auroc_macro"] = (round(float(np.mean(auroc_scores)), 4)
                              if auroc_scores else 0.0)


    f1_scores:   list[float] = []
    precision_scores: list[float] = []
    recall_scores:  list[float] = []
    specificity_scores: list[float] = []

    for class_index in range(num_classes):
        f1_value   = f1_score(labels[:, class_index], predictions[:, class_index], zero_division=0)
        precision_value = precision_score(labels[:, class_index], predictions[:, class_index], zero_division=0)
        recall_value  = recall_score(labels[:, class_index], predictions[:, class_index], zero_division=0)


        true_negatives = float(((1 - labels[:, class_index]) * (1 - predictions[:, class_index])).sum())
        false_positives = float(((1 - labels[:, class_index]) * predictions[:, class_index]).sum())
        specificity_value = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0

        metrics[f"f1_{class_names[class_index]}"]          = round(float(f1_value), 4)
        metrics[f"precision_{class_names[class_index]}"]   = round(float(precision_value), 4)
        metrics[f"recall_{class_names[class_index]}"]      = round(float(recall_value), 4)
        metrics[f"specificity_{class_names[class_index]}"] = round(float(specificity_value), 4)
        f1_scores.append(f1_value)
        precision_scores.append(precision_value)
        recall_scores.append(recall_value)
        specificity_scores.append(specificity_value)


    metrics["f1_macro"]          = round(float(np.mean(f1_scores)), 4)
    metrics["precision_macro"]   = round(float(np.mean(precision_scores)), 4)
    metrics["recall_macro"]      = round(float(np.mean(recall_scores)), 4)
    metrics["specificity_macro"] = round(float(np.mean(specificity_scores)), 4)


    exact_match_accuracy = np.all(predictions == labels, axis=1).mean()
    metrics["exact_match_accuracy"] = round(float(exact_match_accuracy), 4)


    sample_f1_score = f1_score(labels, predictions, average="samples", zero_division=0)
    metrics["f1_samples"] = round(float(sample_f1_score), 4)

    return metrics


def compare_results(results_dict: dict,
                    class_names: list[str] | None = None) -> pd.DataFrame:
    """Convert per-run result dictionaries into a sorted summary DataFrame."""

    summary_rows: list[dict] = []
    for experiment_name, metrics in results_dict.items():
        summary_row = {
            "run":                experiment_name,
            "arch":               metrics.get("arch", ""),
            "lead_config":        metrics.get("lead_config", ""),
            "num_leads":          metrics.get("num_leads", ""),
            "total_params":       metrics.get("total_params", ""),
            "training_time_s":    metrics.get("training_time_s", ""),
            "best_epoch":         metrics.get("best_epoch", ""),
            "auroc_macro":        metrics.get("auroc_macro", 0),
            "f1_macro":           metrics.get("f1_macro", 0),
            "f1_samples":         metrics.get("f1_samples", 0),
            "exact_match_accuracy": metrics.get("exact_match_accuracy", 0),
            "specificity_macro":  metrics.get("specificity_macro", 0),
        }


        if class_names:
            for class_name in class_names:
                summary_row[f"auroc_{class_name}"] = metrics.get(f"auroc_{class_name}", 0)
                summary_row[f"f1_{class_name}"]    = metrics.get(f"f1_{class_name}", 0)

        summary_rows.append(summary_row)

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame = summary_frame.sort_values(["arch", "lead_config"]).reset_index(drop=True)
    return summary_frame
