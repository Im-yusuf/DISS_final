"""Integrated Gradients utilities for ECG lead-importance analysis."""

import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from ecg_lead_reduction.core.config import (
    FIGURES_DIR, CHECKPOINTS_DIR, PROCESSED_DATA_DIR, PROCESSED_NPZ,
    LEAD_CONFIGS, LEAD_NAMES_12, DEVICE,
    RANDOM_SEED, TEST_SPLIT,
)
from ecg_lead_reduction.models.model import build_model


class IntegratedGradientsECG:
    """Integrated Gradients attribution helper for batched ECG tensors."""

    def __init__(self, model: nn.Module, device: torch.device = DEVICE,
                 n_steps: int = 50):
        """Store the model and attribution resolution for later calls."""

        self.model = model
        self.device = device
        self.n_steps = n_steps
        self.model.eval()

    def attribute(self, ecg_signal: torch.Tensor, target_class_index: int,
                  baseline_signal: torch.Tensor = None) -> np.ndarray:
        """Compute per-sample attributions for one target class."""

        ecg_signal = ecg_signal.to(self.device)
        if baseline_signal is None:
            baseline_signal = torch.zeros_like(ecg_signal)
        else:
            baseline_signal = baseline_signal.to(self.device)

        interpolation_steps = torch.linspace(0, 1, self.n_steps + 1, device=self.device)
        interpolation_steps = interpolation_steps.view(-1, 1, 1, 1)

        signal_delta = ecg_signal - baseline_signal
        interpolated_signals = baseline_signal + interpolation_steps * signal_delta
        interpolated_signals = interpolated_signals.squeeze(1)


        batch_chunk_size = 16
        gradient_batches = []

        for row_index in range(0, len(interpolated_signals), batch_chunk_size):
            input_chunk = interpolated_signals[row_index:row_index + batch_chunk_size].detach().requires_grad_(True)
            batch_logits = self.model(input_chunk)
            target_score_sum = batch_logits[:, target_class_index].sum()
            self.model.zero_grad()
            target_score_sum.backward()
            gradient_batches.append(input_chunk.grad.detach())

        gradient_path = torch.cat(gradient_batches, dim=0)


        average_gradients = (gradient_path[:-1] + gradient_path[1:]).mean(dim=0) / 2


        integrated_gradients = (signal_delta.squeeze(0) * average_gradients).detach().cpu().numpy()

        return integrated_gradients


def compute_lead_importance(model, ecg_signal, target_class_index, device=DEVICE,
                            n_steps=30):
    """Aggregate Integrated Gradients attributions into normalised lead scores."""

    integrated_gradients = IntegratedGradientsECG(model, device, n_steps=n_steps)
    attribution = integrated_gradients.attribute(ecg_signal, target_class_index)

    lead_importance = np.mean(np.abs(attribution), axis=1)

    importance_total = lead_importance.sum()
    if importance_total > 0:
        lead_importance = lead_importance / importance_total

    return lead_importance


def compute_lead_importance_batch(model, signals, label_matrix, target_class_index,
                                  num_samples=50, device=DEVICE):
    """Average lead-importance scores over positive samples for one class."""

    model.eval()
    positive_sample_indices = np.where(label_matrix[:, target_class_index] == 1)[0]
    positive_count = len(positive_sample_indices)

    if positive_count == 0:
        return None, 0

    if positive_count > num_samples:
        rng = np.random.RandomState(RANDOM_SEED)
        selected_indices = rng.choice(positive_sample_indices, num_samples, replace=False)
    else:
        selected_indices = positive_sample_indices

    importance_samples = []
    for sample_index in selected_indices:
        sample_tensor = torch.tensor(signals[sample_index:sample_index+1], dtype=torch.float32).to(device)
        importance_values = compute_lead_importance(model, sample_tensor, target_class_index, device)
        importance_samples.append(importance_values)

    return np.mean(importance_samples, axis=0), positive_count


def plot_lead_importance_heatmap(importance_by_class, selected_lead_names, experiment_name,
                                 output_file):
    """Save a class-by-lead heatmap for one trained experiment."""

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    condition_names = sorted([k for k, v in importance_by_class.items() if v is not None])
    if not condition_names:
        return

    heatmap_values = np.zeros((len(condition_names), len(selected_lead_names)))
    for row_index, condition_name in enumerate(condition_names):
        heatmap_values[row_index] = importance_by_class[condition_name]

    figure_height = max(6, len(condition_names) * 0.45)
    figure_width = max(8, len(selected_lead_names) * 0.8)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))

    heatmap_image = axis.imshow(heatmap_values, cmap='YlOrRd', aspect='auto', vmin=0)

    axis.set_xticks(range(len(selected_lead_names)))
    axis.set_xticklabels(selected_lead_names, rotation=45, ha='right', fontsize=10)
    axis.set_yticks(range(len(condition_names)))
    axis.set_yticklabels(condition_names, fontsize=9)
    axis.set_xlabel('Lead', fontsize=11)
    axis.set_ylabel('Condition', fontsize=11)


    run_parts = experiment_name.split('_')
    if run_parts[0] == 'cnn':
        architecture_label = 'CNN-LSTM'
        lead_config_label = '_'.join(run_parts[2:]).replace('_', '-')
    else:
        architecture_label = run_parts[0].upper()
        lead_config_label = '_'.join(run_parts[1:]).replace('_', '-')
    axis.set_title(
        f'Lead Importance by Condition — {lead_config_label} ({architecture_label})',
        fontsize=12,
    )

    colorbar = plt.colorbar(heatmap_image, ax=axis, shrink=0.8)
    colorbar.set_label('Importance Score')


    for row_index in range(heatmap_values.shape[0]):
        for column_index in range(heatmap_values.shape[1]):
            cell_value = heatmap_values[row_index, column_index]
            text_color = 'white' if cell_value > 0.5 * heatmap_values.max() else 'black'
            axis.text(column_index, row_index, f'{cell_value:.2f}', ha='center', va='center',
                    fontsize=7, color=text_color)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def plot_lead_importance_comparison(importance_by_config, architecture_name,
                                    output_file):
    """Save a cross-configuration heatmap for shared classes in one architecture."""

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    lead_config_order = ['12-lead', '6-lead', '4-lead', '3-lead', '2-lead', '1-lead']
    available_configs = [lead_config_name for lead_config_name in lead_config_order if lead_config_name in importance_by_config]
    if len(available_configs) < 2:
        return


    condition_counts = {}
    for lead_config_name in available_configs:
        for condition_name, importance_values in importance_by_config[lead_config_name].items():
            if importance_values is not None:
                condition_counts[condition_name] = condition_counts.get(condition_name, 0) + 1
    shared_conditions = sorted([class_index for class_index, n in condition_counts.items() if n >= 2])
    if not shared_conditions:
        return

    total_leads = len(LEAD_NAMES_12)
    config_count = len(available_configs)
    condition_count = len(shared_conditions)

    plot_column_count = min(3, condition_count)
    plot_row_count = (condition_count + plot_column_count - 1) // plot_column_count

    figure, axes_grid = plt.subplots(
        plot_row_count, plot_column_count,
        figsize=(6 * plot_column_count + 0.8, max(3, config_count * 0.5) * plot_row_count),
        squeeze=False,
    )

    for sample_index, condition_name in enumerate(shared_conditions):
        subplot_row = sample_index // plot_column_count
        subplot_col = sample_index % plot_column_count
        axis = axes_grid[subplot_row][subplot_col]

        heatmap_values = np.full((config_count, total_leads), np.nan)
        for r, lead_config_name in enumerate(available_configs):
            lead_indices = LEAD_CONFIGS[lead_config_name]
            importance_values = importance_by_config[lead_config_name].get(condition_name)
            if importance_values is not None:
                for column_index, lead_index in enumerate(lead_indices):
                    heatmap_values[r, lead_index] = importance_values[column_index]


        masked_values = np.ma.masked_invalid(heatmap_values)
        color_map = plt.cm.YlOrRd.copy()
        color_map.set_bad(color='#e0e0e0')

        heatmap_image = axis.imshow(masked_values, cmap=color_map, aspect='auto', vmin=0)

        axis.set_xticks(range(total_leads))
        axis.set_xticklabels(LEAD_NAMES_12, rotation=45, ha='right', fontsize=8)
        axis.set_yticks(range(config_count))
        axis.set_yticklabels(available_configs, fontsize=8)
        axis.set_title(condition_name, fontsize=10, fontweight='bold')


        for row_index in range(heatmap_values.shape[0]):
            for column_index in range(heatmap_values.shape[1]):
                cell_value = heatmap_values[row_index, column_index]
                if np.isnan(cell_value):
                    axis.text(column_index, row_index, '—', ha='center', va='center',
                            fontsize=6, color='#999999')
                else:
                    text_color = 'white' if cell_value > 0.12 else 'black'
                    axis.text(column_index, row_index, f'{cell_value:.2f}', ha='center', va='center',
                            fontsize=6, color=text_color)


    for sample_index in range(condition_count, plot_row_count * plot_column_count):
        axes_grid[sample_index // plot_column_count][sample_index % plot_column_count].set_visible(False)

    architecture_title = architecture_name.upper().replace('_', '-')
    figure.suptitle(
        f'Lead Importance Across Configurations — {architecture_title}',
        fontsize=13, fontweight='bold', y=1.01,
    )
    figure.subplots_adjust(left=0.055, right=0.92, bottom=0.09, top=0.92,
                        wspace=0.16, hspace=0.32)
    colorbar_axis = figure.add_axes([0.94, 0.24, 0.015, 0.52])
    colorbar = figure.colorbar(heatmap_image, cax=colorbar_axis)
    colorbar.set_label('Importance Score')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def generate_xai_figures(num_samples: int = 50):
    """Generate all checkpoint-based lead-importance figures and JSON exports."""

    from ecg_lead_reduction.data.dataset import load_data

    processed_archive_path = PROCESSED_DATA_DIR / PROCESSED_NPZ
    if not processed_archive_path.exists():
        print("  XAI: preprocessed data not found — skipping.")
        return

    all_signals, all_labels, class_names, num_classes, _ = load_data(processed_archive_path)


    from sklearn.model_selection import train_test_split
    sample_index = np.arange(len(all_signals))
    _, test_indices = train_test_split(
        sample_index,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=all_labels.argmax(axis=1),
    )
    test_signals = all_signals[test_indices]
    test_labels  = all_labels[test_indices]

    xai_output_dir = FIGURES_DIR / "xai"
    xai_output_dir.mkdir(parents=True, exist_ok=True)


    importance_by_architecture = {}

    for checkpoint_file in sorted(CHECKPOINTS_DIR.glob("best_*.pt")):
        experiment_name = checkpoint_file.stem.replace("best_", "")

        if experiment_name.startswith("cnn_lstm_"):
            architecture        = "cnn_lstm"
            lead_config_name = experiment_name[len("cnn_lstm_"):]
        elif experiment_name.startswith("resnet_"):
            architecture        = "resnet"
            lead_config_name = experiment_name[len("resnet_"):]
        else:
            continue

        if lead_config_name not in LEAD_CONFIGS:
            continue

        print(f"\n  Computing lead importance for {experiment_name}...")

        lead_indices = LEAD_CONFIGS[lead_config_name]
        lead_count    = len(lead_indices)
        selected_lead_names   = [LEAD_NAMES_12[row_index] for row_index in lead_indices]

        model = build_model(architecture, lead_count, num_classes).to(DEVICE)
        model.load_state_dict(
            torch.load(checkpoint_file, map_location=DEVICE, weights_only=True)
        )
        model.eval()

        selected_test_signals = test_signals[:, lead_indices, :]

        importance_by_class = {}

        for class_index in range(num_classes):
            class_name = class_names[class_index]
            importance_values, positive_count = compute_lead_importance_batch(
                model, selected_test_signals, test_labels, class_index,
                num_samples=num_samples, device=DEVICE,
            )
            if importance_values is not None:
                importance_by_class[class_name] = importance_values
                ranked_leads = sorted(zip(selected_lead_names, importance_values), key=lambda sample_tensor: -sample_tensor[1])
                top_leads_text = ', '.join([f"{n}={v:.3f}" for n, v in ranked_leads[:3]])
                print(f"    {class_name:>6} ({positive_count:>3} pos): top -> {top_leads_text}")
            else:
                importance_by_class[class_name] = None


        heatmap_file = xai_output_dir / f"lead_importance_{experiment_name}.png"
        plot_lead_importance_heatmap(importance_by_class, selected_lead_names,
                                     experiment_name, heatmap_file)


        json_payload = {
            'lead_config': lead_config_name,
            'architecture': architecture,
            'lead_names': selected_lead_names,
            'xai_method': 'integrated_gradients',
            'lead_importance_per_class': {},
        }
        for class_name, importance_values in importance_by_class.items():
            if importance_values is not None:
                json_payload['lead_importance_per_class'][class_name] = {
                    'importance': importance_values.tolist(),
                    'lead_names': selected_lead_names,
                }

        json_file = xai_output_dir / f"lead_importance_{experiment_name}.json"
        with open(json_file, 'w') as json_file_handle:
            json.dump(json_payload, json_file_handle, indent=2)
        print(f"  Saved: {json_file}")


        if architecture not in importance_by_architecture:
            importance_by_architecture[architecture] = {}
        importance_by_architecture[architecture][lead_config_name] = importance_by_class


    for architecture, config_importance in importance_by_architecture.items():
        comparison_file = xai_output_dir / f"lead_importance_comparison_{architecture}.png"
        plot_lead_importance_comparison(config_importance, architecture, comparison_file)

    print("\n  All XAI figures generated.")


def main() -> None:
    """Run the XAI figure-generation CLI."""

    generate_xai_figures()


if __name__ == '__main__':
    main()
