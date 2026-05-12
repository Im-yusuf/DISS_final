"""Generate comparison plots from saved experiment result summaries.

The experiment runner writes `results_summary.json`; this module turns that
summary into publication-style PNG figures under `artifacts/figures/` and then
attempts to generate optional XAI plots from saved checkpoints.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ecg_lead_reduction.core.config import FIGURES_DIR, LEAD_CONFIGS


def generate_all_figures(summary_json_path: str | Path) -> None:
    """Generate all standard comparison and explainability figures.

    Missing summaries are treated as a warning so the experiment runner can fail
    gracefully without producing partially initialised plotting errors.
    """

    summary_json_path = Path(summary_json_path)
    if not summary_json_path.exists():
        print(f"  WARNING: Summary not found: {summary_json_path}")
        return

    with open(summary_json_path) as summary_file:
        results_by_run = json.load(summary_file)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Keep the standard report order stable so reruns overwrite a predictable figure set.
    _plot_macro_comparison(results_by_run, "auroc_macro", "Macro AUROC")
    _plot_macro_comparison(results_by_run, "f1_macro", "Macro F1")
    _plot_per_class_heatmap(results_by_run)
    _plot_training_curves(results_by_run)
    _plot_lead_degradation(results_by_run)


    print("\n  Generating Integrated Gradients XAI figures...")
    try:
        from ecg_lead_reduction.analysis.xai import generate_xai_figures
        generate_xai_figures()
    except Exception as error:
        print(f"  WARNING: XAI figure generation failed: {error}")

    print(f"  Figures saved to {FIGURES_DIR}")


def _plot_macro_comparison(results_by_run: dict, metric_key: str,
                           metric_name: str) -> None:
    """Plot one macro metric for each run in the experiment summary.

    Bar colours distinguish architectures while the x-axis labels keep the full
    run names so lead configuration remains visible in exported figures.
    """

    # Sort by run key so figure ordering matches summary-table ordering across reruns.
    run_names   = sorted(results_by_run.keys())
    metric_values = [results_by_run[run_name].get(metric_key, 0) for run_name in run_names]

    figure, axis = plt.subplots(figsize=(max(10, len(run_names) * 1.2), 5))

    bar_colors = ["#2196F3" if "resnet" in run_name else "#FF9800" for run_name in run_names]

    bar_containers = axis.bar(range(len(run_names)), metric_values, color=bar_colors,
                  edgecolor="white", linewidth=0.5)


    for bar_container, metric_value in zip(bar_containers, metric_values):
        axis.text(bar_container.get_x() + bar_container.get_width() / 2,
                bar_container.get_height() + 0.005,
                f"{metric_value:.3f}", ha="center", va="bottom", fontsize=8)

    axis.set_xticks(range(len(run_names)))
    axis.set_xticklabels([run_name.replace("_", "\n") for run_name in run_names],
                       fontsize=8, rotation=45, ha="right")
    axis.set_ylabel(metric_name)
    axis.set_title(f"{metric_name} by Architecture × Lead Configuration")
    axis.set_ylim(0, min(1.0, max(metric_values) + 0.1) if metric_values else 1.0)
    axis.grid(axis="y", alpha=0.3)

    legend_handles = [Patch(facecolor="#2196F3", label="ResNet"),
                       Patch(facecolor="#FF9800", label="CNN-LSTM")]
    axis.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    output_metric_name = metric_key.replace(" ", "_").lower()
    plt.savefig(FIGURES_DIR / f"comparison_{output_metric_name}.png", dpi=150)
    plt.close()


def _plot_per_class_heatmap(results_by_run: dict) -> None:
    """Plot per-class AUROC values as a run-by-class heatmap.

    If stored `class_names` are unavailable, class labels are inferred from
    `auroc_<class>` keys in the first result entry.
    """

    run_names = sorted(results_by_run.keys())
    if not run_names:
        return


    first_result = results_by_run[run_names[0]]
    class_names = first_result.get("class_names", [])
    if not class_names:
        # Older result files may not store `class_names`, so infer them from metric keys.
        class_names = sorted(
            k.replace("auroc_", "") for k in first_result
            if k.startswith("auroc_") and k != "auroc_macro"
        )
    if not class_names:
        return


    heatmap_values = np.zeros((len(run_names), len(class_names)))
    for row_index, run_name in enumerate(run_names):
        for column_index, class_name in enumerate(class_names):
            heatmap_values[row_index, column_index] = results_by_run[run_name].get(f"auroc_{class_name}", 0)

    figure, axis = plt.subplots(
        figsize=(max(8, len(class_names) * 0.8),
                 max(4, len(run_names) * 0.5)))
    heatmap_image = axis.imshow(heatmap_values, aspect="auto", cmap="YlOrRd",
                   vmin=0.5, vmax=1.0)

    axis.set_xticks(range(len(class_names)))
    axis.set_xticklabels(class_names, fontsize=8, rotation=45, ha="right")
    axis.set_yticks(range(len(run_names)))
    axis.set_yticklabels([run_name.replace("_", "\n") for run_name in run_names], fontsize=7)

    for row_index in range(len(run_names)):
        for column_index in range(len(class_names)):
            metric_value = heatmap_values[row_index, column_index]
            text_color = "black" if metric_value < 0.75 else "white"
            axis.text(column_index, row_index, f"{metric_value:.2f}", ha="center", va="center",
                    fontsize=6, color=text_color)

    axis.set_title("Per-Class AUROC Heatmap")
    plt.colorbar(heatmap_image, ax=axis, fraction=0.02, pad=0.04)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "per_class_auroc_heatmap.png", dpi=150)
    plt.close()


def _plot_training_curves(results_by_run: dict) -> None:
    """Plot training loss and validation AUROC curves when history is available.

    Skip-training runs do not include history, so this figure is optional.
    """

    # Skip runs that came from checkpoint-only evaluation because they have no training trace.
    history_by_run = {
        run_name: run_result for run_name, run_result in results_by_run.items()
        if "history" in run_result and run_result["history"]
    }
    if not history_by_run:
        return

    figure, axes_grid = plt.subplots(1, 2, figsize=(14, 5))

    for run_name, run_result in sorted(history_by_run.items()):
        history = run_result["history"]
        epochs     = [history_entry["epoch"] for history_entry in history]
        training_losses = [history_entry["train_loss"] for history_entry in history]
        validation_aurocs  = [history_entry.get("val_auroc_macro", 0) for history_entry in history]

        axes_grid[0].plot(epochs, training_losses, label=run_name, alpha=0.8)
        axes_grid[1].plot(epochs, validation_aurocs,  label=run_name, alpha=0.8)

    axes_grid[0].set_xlabel("Epoch")
    axes_grid[0].set_ylabel("Training Loss")
    axes_grid[0].set_title("Training Loss Curves")
    axes_grid[0].legend(fontsize=6)
    axes_grid[0].grid(alpha=0.3)

    axes_grid[1].set_xlabel("Epoch")
    axes_grid[1].set_ylabel("Validation AUROC (Macro)")
    axes_grid[1].set_title("Validation AUROC Curves")
    axes_grid[1].legend(fontsize=6)
    axes_grid[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "training_curves.png", dpi=150)
    plt.close()


def _plot_lead_degradation(results_by_run: dict) -> None:
    """Plot macro AUROC against the number of input leads for each architecture.

    The x-axis is inverted so the figure reads naturally from full 12-lead input
    to the smallest lead subset.
    """

    ordered_lead_configs = sorted(LEAD_CONFIGS.keys(),
                             key=lambda k: len(LEAD_CONFIGS[k]),
                             reverse=True)


    # Infer the architecture family from the saved run key convention.
    architectures = sorted({run_name.rsplit("_", 1)[0] for run_name in results_by_run
                    if "_" in run_name})
    if not architectures:
        return

    figure, axis = plt.subplots(figsize=(8, 5))
    architecture_styles = {
        "resnet":   {"color": "#2196F3", "marker": "o"},
        "cnn_lstm": {"color": "#FF9800", "marker": "s"},
    }

    for arch in architectures:
        lead_counts: list[int]   = []
        auroc_values: list[float] = []
        point_labels: list[str]   = []

        for lead_config_name in ordered_lead_configs:
            run_key = f"{arch}_{lead_config_name}"
            if run_key not in results_by_run:
                continue
            lead_count = len(LEAD_CONFIGS[lead_config_name])
            auroc_value   = results_by_run[run_key].get("auroc_macro", 0)
            lead_counts.append(lead_count)
            auroc_values.append(auroc_value)
            point_labels.append(lead_config_name)

        if not lead_counts:
            continue

        plot_style = architecture_styles.get(arch, {"color": "gray", "marker": "^"})
        axis.plot(lead_counts, auroc_values, label=arch, linewidth=2,
                markersize=8, **plot_style)


        for lead_count_value, auroc_point_value, point_label in zip(lead_counts, auroc_values, point_labels):
            axis.annotate(f"{auroc_point_value:.3f}", (lead_count_value, auroc_point_value), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=7)

    axis.set_xlabel("Number of Leads")
    axis.set_ylabel("Macro AUROC")
    axis.set_title("Lead Reduction: Macro AUROC vs. Number of Leads")
    axis.set_xticks(sorted({len(v) for v in LEAD_CONFIGS.values()}))
    axis.legend()
    axis.grid(alpha=0.3)
    axis.invert_xaxis()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "lead_degradation.png", dpi=150)
    plt.close()
