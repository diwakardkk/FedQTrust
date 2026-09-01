"""Command line interface for FedQTrust."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from fedqtrust.config import load_config
from fedqtrust.data.datasets import ensure_all_datasets
from fedqtrust.device import write_environment_report
from fedqtrust.smoke import run_smoke


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--profile", default="smoke", choices=["smoke", "phase1", "phase2", "paper"])


def preflight(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, profile=args.profile, device=args.device, output_dir=args.output_dir)
    env = write_environment_report(Path(args.output_dir) / "environment", args.device)
    print(json.dumps({"config": cfg.to_dict(), "environment": env}, indent=2))
    if args.profile == "paper":
        missing = []
        if not env["hyperledger_fabric_available"]:
            missing.append("Hyperledger Fabric peer CLI")
        if not env["liboqs_available"]:
            missing.append("liboqs")
        if not env["qiskit_machine_learning_version"]:
            missing.append("qiskit-machine-learning")
        if missing:
            raise RuntimeError("paper profile missing required dependencies: " + ", ".join(missing))
    return 0


def download_data(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, profile=args.profile, device=args.device, output_dir=args.output_dir)
    ensure_all_datasets(cfg.data.root, True, "data/metadata")
    return 0


def prepare_data(args: argparse.Namespace) -> int:
    from fedqtrust.data.datasets import DATASET_SPECS, label_array, load_medmnist_dataset
    from fedqtrust.data.partition import dirichlet_partition, save_partitions, write_data_stats

    cfg = load_config(args.config, profile=args.profile, device=args.device, output_dir=args.output_dir)
    splits_by_dataset = {}
    for spec in DATASET_SPECS:
        train = load_medmnist_dataset(spec.medmnist_flag, "train", cfg.data.root, False)
        splits, notes = dirichlet_partition(
            label_array(train),
            cfg.training.clients_total // len(DATASET_SPECS),
            cfg.data.partition_alpha,
            cfg.data.partition_seed,
            cfg.data.validation_fraction,
            cfg.data.min_samples_per_class,
            cfg.data.max_partition_retries,
        )
        for note in notes:
            print(f"[WARN] {spec.display}: {note}")
        save_partitions(spec.key, splits, cfg.data.partitions, cfg.data.partition_seed)
        splits_by_dataset[spec.key] = splits
    write_data_stats(splits_by_dataset, Path(args.output_dir) / "tables" / "data_stats.csv")
    return 0


def run_experiment(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, profile=args.profile, device=args.device, output_dir=args.output_dir)
    manifest_path = Path(args.output_dir) / "experiments" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "run_id": f"{args.experiment}_{args.seed or 'configured'}",
        "experiment": args.experiment,
        "method": "configured",
        "dataset": "configured",
        "attack": "configured",
        "attack_parameter": "configured",
        "seed": args.seed if args.seed is not None else ",".join(map(str, cfg.seeds)),
        "status": "PENDING",
        "output_path": str(Path(args.output_dir) / "experiments" / args.experiment),
    }
    exists = manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"[MANIFEST] Added {args.experiment} to {manifest_path}")
    print("[INFO] Full training orchestration is scaffolded; use smoke-test/preflight before launching GPU jobs.")
    return 0


def generate_report(args: argparse.Namespace) -> int:
    from fedqtrust.reporting.summary import append_note

    out = Path(args.output_dir)
    for sub in ["paper_figures", "paper_tables", "statistics", "summaries"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    append_note(out, "Report generation consumes existing raw experiment outputs and does not retrain models.")
    (out / "summaries" / "final_summary.md").write_text(
        "# FedQTrust Report\n\nRun E1-E9 experiments first; generated reports will aggregate raw CSV outputs without fabricating values.\n",
        encoding="utf-8",
    )
    print(f"[REPORT] Wrote {out / 'summaries' / 'final_summary.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fedqtrust")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke-test")
    _common(smoke)
    smoke.add_argument("--download-data", action="store_true")

    for name in ["download-data", "prepare-data", "preflight", "run-all", "generate-report"]:
        p = sub.add_parser(name)
        _common(p)

    run = sub.add_parser("run")
    _common(run)
    run.add_argument("--experiment", required=True, choices=[f"E{i}" for i in range(1, 10)])

    args = parser.parse_args(argv)
    if args.command == "smoke-test":
        return run_smoke(args.download_data, args.device, args.output_dir, args.config)
    if args.command == "download-data":
        return download_data(args)
    if args.command == "prepare-data":
        return prepare_data(args)
    if args.command == "preflight":
        return preflight(args)
    if args.command == "run":
        return run_experiment(args)
    if args.command == "run-all":
        for exp in [f"E{i}" for i in range(1, 10)]:
            args.experiment = exp
            run_experiment(args)
        return 0
    if args.command == "generate-report":
        return generate_report(args)
    raise ValueError(args.command)

