import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    ARCHITECTURES, BATCH_SIZE, CHECKPOINTS_DIR, DEVICE,
    LEAD_CONFIGS, PROCESSED_DATA_DIR, PROCESSED_NPZ,
    RANDOM_SEED, RESULTS_DIR,
    TEST_SPLIT, VAL_SPLIT,
)
from dataset import compute_pos_weight, get_dataloaders, set_seed
from evaluate import compare_results, compute_metrics
from model import build_model
from train import _make_json_serialisable, _run_epoch, train_model
from visualise import generate_all_figures


def _train_worker(worker_job: tuple[str, str, str, int]) -> tuple[str, dict | None]:
    import torch
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    arch, lead_config, device_name, thread_count = worker_job
    torch.set_num_threads(thread_count)
    experiment_name = f"{arch}_{lead_config}"
    try:
        worker_result = train_model(arch, lead_config,
                             device=torch.device(device_name))
        return experiment_name, worker_result
    except Exception as error:
        print(f"  ERROR in {experiment_name}: {error}", flush=True)
        return experiment_name, None


def run_skip_training(arch: str, lead_config: str) -> dict | None:
    experiment_name  = f"{arch}_{lead_config}"
    checkpoint_path = CHECKPOINTS_DIR / f"best_{experiment_name}.pt"

    if not checkpoint_path.exists():
        print(f"  WARNING: Checkpoint not found: {checkpoint_path} — skipping")
        return None

    print(f"\n  Evaluating {experiment_name} from checkpoint …")

    set_seed(RANDOM_SEED)
    (train_data_loader, validation_data_loader, test_data_loader,
     class_names, num_classes, train_labels) = get_dataloaders(
        lead_config, BATCH_SIZE, TEST_SPLIT, VAL_SPLIT, RANDOM_SEED,
    )

    num_leads = len(LEAD_CONFIGS[lead_config])
    model = build_model(arch, num_leads, num_classes).to(DEVICE)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))

    positive_weights = compute_pos_weight(train_labels).to(DEVICE)
    loss_function  = nn.BCEWithLogitsLoss(pos_weight=positive_weights)

    _, test_labels, test_logits = _run_epoch(
        model, test_data_loader, loss_function, device=DEVICE)
    test_results = compute_metrics(test_labels, test_logits, class_names)

    run_results = {
        "arch":        arch,
        "lead_config": lead_config,
        "num_leads":   num_leads,
        "num_classes": num_classes,
        "class_names": class_names,
        **test_results,
    }

    print(f"  {experiment_name} → AUROC: {test_results['auroc_macro']:.4f} | "
          f"F1: {test_results['f1_macro']:.4f}")
    return run_results


def _print_delta_table(results_by_run: dict, selected_architectures: list[str]) -> None:
    print(f"\n{'=' * 70}")
    print("  LEAD REDUCTION — DELTA TABLE (relative to 12-lead baseline)")
    print(f"{'=' * 70}")

    for arch in selected_architectures:
        baseline_key = f"{arch}_12-lead"
        baseline_results = results_by_run.get(baseline_key)
        if baseline_results is None:
            continue
        baseline_auroc = baseline_results.get("auroc_macro", 0)
        baseline_f1    = baseline_results.get("f1_macro", 0)

        print(f"\n  {arch}")
        for lead_config_name in LEAD_CONFIGS:
            run_key = f"{arch}_{lead_config_name}"
            lead_result = results_by_run.get(run_key)
            if lead_result is None:
                continue
            auroc_value = lead_result.get("auroc_macro", 0)
            f1_value    = lead_result.get("f1_macro", 0)
            delta_auroc = auroc_value - baseline_auroc
            delta_f1  = f1_value - baseline_f1
            if lead_config_name == "12-lead":
                print(f"    {lead_config_name:>8s}  →  AUROC {auroc_value:.4f}"
                      f"              F1 {f1_value:.4f}")
            else:
                print(f"    {lead_config_name:>8s}  →  AUROC {auroc_value:.4f}  "
                      f"(Δ {delta_auroc:+.4f})  F1 {f1_value:.4f}  (Δ {delta_f1:+.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ECG lead reduction experiments")
    parser.add_argument("--arch", type=str, choices=["resnet", "cnn_lstm"],
                        help="Run only this architecture")
    parser.add_argument("--lead-config", type=str,
                        choices=list(LEAD_CONFIGS.keys()),
                        help="Run only this lead configuration")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training; evaluate from checkpoints")
    parser.add_argument("--parallel", type=int, default=1, metavar="N",
                        help="Number of experiments to run in parallel (default 1).  "
                             "E.g. --parallel 6 runs 6 at once.  "
                             "Each process gets its own MPS context.")
    cli_args = parser.parse_args()


    selected_architectures        = [cli_args.arch] if cli_args.arch else ARCHITECTURES
    selected_lead_configs = ([cli_args.lead_config] if cli_args.lead_config
                    else list(LEAD_CONFIGS.keys()))

    experiment_grid = [(a, lead_config_name) for a in selected_architectures for lead_config_name in selected_lead_configs]
    run_count  = len(experiment_grid)

    print(f"\n{'=' * 70}")
    print(f"  ECG Lead Reduction Experiments — three-database corpus")
    print(f"  Architectures  : {selected_architectures}")
    print(f"  Lead configs   : {selected_lead_configs}")
    print(f"  Total runs     : {run_count}")
    print(f"  Device         : {DEVICE}")
    print(f"  Skip training  : {cli_args.skip_training}")
    print(f"  Parallel workers: {cli_args.parallel}")
    print(f"{'=' * 70}")


    processed_archive_path = PROCESSED_DATA_DIR / PROCESSED_NPZ
    if not processed_archive_path.exists():
        print("\nERROR: Preprocessed data not found.  Run:")
        print("    python preprocess.py")
        sys.exit(1)


    if not cli_args.skip_training:
        all_checkpoints_exist = all(
            (CHECKPOINTS_DIR / f"best_{a}_{lead_config_name}.pt").exists()
            for a, lead_config_name in experiment_grid
        )
        if all_checkpoints_exist:
            print("\n  All checkpoints already exist — auto-skipping training.")
            cli_args.skip_training = True


    results_by_run: dict = {}
    class_names: list[str] | None = None
    experiments_start_time = time.time()

    if cli_args.parallel > 1 and not cli_args.skip_training:


        import torch as _torch
        mps_available = _torch.backends.mps.is_available()
        worker_device_name = "cpu" if mps_available else str(DEVICE)
        cpu_total = multiprocessing.cpu_count()
        thread_count_per_worker = max(1, cpu_total // cli_args.parallel)
        if mps_available:
            print(f"  MPS detected — workers will use CPU "
                  f"({thread_count_per_worker} threads each) to avoid contention.")
        print(f"  Launching {len(experiment_grid)} jobs across "
              f"{cli_args.parallel} parallel workers …\n")
        worker_jobs = [(a, lead_config_name, worker_device_name, thread_count_per_worker)
                for a, lead_config_name in experiment_grid]
        process_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=cli_args.parallel,
                                  mp_context=process_context) as executor:
            future_map = {executor.submit(_train_worker, worker_job): worker_job
                       for worker_job in worker_jobs}
            for completed_future in as_completed(future_map):
                experiment_name, run_results = completed_future.result()
                if run_results is not None:
                    results_by_run[experiment_name] = run_results
                    if class_names is None:
                        class_names = run_results.get("class_names", [])
                    print(f"  ✓ {experiment_name} done — "
                          f"AUROC {run_results.get('auroc_macro', 0):.4f}")
    else:

        for arch, lead_config in experiment_grid:
            experiment_name = f"{arch}_{lead_config}"
            if cli_args.skip_training:
                run_results = run_skip_training(arch, lead_config)
            else:
                run_results = train_model(arch, lead_config)

            if run_results is not None:
                results_by_run[experiment_name] = run_results
                if class_names is None:
                    class_names = run_results.get("class_names", [])

    experiment_duration = time.time() - experiments_start_time

    if not results_by_run:
        print("\nNo results collected.  Check error output above.")
        sys.exit(1)


    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_json_path = RESULTS_DIR / "results_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(_make_json_serialisable(results_by_run), f, indent=2)
    print(f"\nSummary JSON saved to {summary_json_path}")


    summary_frame = compare_results(results_by_run, class_names)
    summary_csv_path = RESULTS_DIR / "results_summary.csv"
    summary_frame.to_csv(summary_csv_path, index=False)
    print(f"Summary CSV  saved to {summary_csv_path}")


    display_columns = ["run", "auroc_macro", "f1_macro", "f1_samples",
                    "exact_match_accuracy", "specificity_macro"]
    if class_names:
        display_columns += [f"auroc_{cn}" for cn in class_names]


    display_columns = [c for c in display_columns if c in summary_frame.columns]

    print(f"\n{'=' * 70}")
    print("  RESULTS COMPARISON")
    print(f"{'=' * 70}")
    print(summary_frame[display_columns].to_string(index=False))


    _print_delta_table(results_by_run, selected_architectures)

    print(f"\n  Total experiment time: {experiment_duration:.1f}s")


    print("\n  Generating comparison figures …")
    generate_all_figures(summary_json_path)

    print(f"\n{'=' * 70}")
    print("  All experiments complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
