import json
from pathlib import Path

import pandas as pd

from fedqtrust.real_results import RealResultsConfig, audit_real_results_outputs, run_real_results_suite


def test_real_results_suite_reads_completed_training_run(tmp_path):
    output_dir = tmp_path / "scientific_results"
    run = output_dir / "raw_training" / "run_20260906_120000"
    run.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "dataset": "pneumoniamnist",
                "display": "PneumoniaMNIST",
                "model": "classical_cnn",
                "rounds": 2,
                "test_accuracy": 0.70,
                "test_weighted_f1": 0.68,
                "total_model_time_s": 1.2,
            },
            {
                "dataset": "pneumoniamnist",
                "display": "PneumoniaMNIST",
                "model": "fedqcnn",
                "rounds": 2,
                "test_accuracy": 0.74,
                "test_weighted_f1": 0.72,
                "total_model_time_s": 1.4,
            },
        ]
    ).to_csv(run / "final_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset": "pneumoniamnist",
                "display": "PneumoniaMNIST",
                "model": "classical_cnn",
                "round": 1,
                "train_loss": 0.9,
                "val_accuracy": 0.60,
                "round_time_s": 0.5,
                "gpu_peak_memory_allocated_bytes": 1024,
            },
            {
                "dataset": "pneumoniamnist",
                "display": "PneumoniaMNIST",
                "model": "fedqcnn",
                "round": 1,
                "train_loss": 0.8,
                "val_accuracy": 0.65,
                "round_time_s": 0.6,
                "gpu_peak_memory_allocated_bytes": 2048,
            },
            {
                "dataset": "pneumoniamnist",
                "display": "PneumoniaMNIST",
                "model": "classical_cnn",
                "round": 2,
                "train_loss": 0.7,
                "val_accuracy": 0.68,
                "round_time_s": 0.5,
                "gpu_peak_memory_allocated_bytes": 1024,
            },
            {
                "dataset": "pneumoniamnist",
                "display": "PneumoniaMNIST",
                "model": "fedqcnn",
                "round": 2,
                "train_loss": 0.6,
                "val_accuracy": 0.71,
                "round_time_s": 0.6,
                "gpu_peak_memory_allocated_bytes": 2048,
            },
        ]
    ).to_csv(run / "round_metrics.csv", index=False)
    (run / "environment.json").write_text(json.dumps({"selected_device_for_run": "cpu"}), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps({"completed_datasets": 1}), encoding="utf-8")
    (run / "fedqcnn_pneumoniamnist_best.pt").write_bytes(b"checkpoint")
    (run / "DONE").write_text("done\n", encoding="utf-8")

    out = run_real_results_suite(
        RealResultsConfig(
            source_dir=str(output_dir / "raw_training"),
            output_dir=str(output_dir),
            device="cpu",
            strict_infra=False,
        )
    )

    ready, issues = audit_real_results_outputs(out, strict_infra=False)
    assert ready, [issue.message for issue in issues]
    assert (out / "raw_metrics" / "final_metrics_all.csv").exists()
    assert (out / "figures" / "Fig02_real_test_accuracy_by_dataset.png").stat().st_size > 0
    assert "does not synthesize" in (out / "summaries" / "final_summary.md").read_text(encoding="utf-8")
