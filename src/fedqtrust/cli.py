"""Command line interface for FedQTrust."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from fedqtrust.config import load_config
from fedqtrust.data.datasets import ensure_all_datasets
from fedqtrust.device import write_environment_report
from fedqtrust.one_shot import OneShotConfig, run_one_shot
from fedqtrust.publication import audit_publication_outputs, write_publication_audit
from fedqtrust.publication_suite import PublicationSuiteConfig, run_publication_suite
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


def run_gpu_once(args: argparse.Namespace) -> int:
    import os
    import sys
    import traceback

    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    datasets = tuple(item.strip().lower() for item in args.datasets.split(",") if item.strip())
    cfg = OneShotConfig(
        output_dir=args.output_dir,
        device=args.device,
        rounds=args.rounds,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed or 42,
        require_cuda=args.require_cuda,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        models=models,
        datasets=datasets,
    )
    try:
        run_one_shot(cfg, download_data=args.download_data, amp=args.amp)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def audit_publication(args: argparse.Namespace) -> int:
    report = write_publication_audit(args.output_dir, args.strict_infra)
    publishable, issues = audit_publication_outputs(args.output_dir, args.strict_infra)
    print(f"[AUDIT] Wrote {report}")
    print(f"[AUDIT] Status: {'PUBLISHABLE_ARTIFACT_SET_PRESENT' if publishable else 'NOT_PUBLISHABLE'}")
    print(f"[AUDIT] Issues: {len(issues)}")
    return 0 if publishable else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fedqtrust")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke-test")
    _common(smoke)
    smoke.add_argument("--download-data", action="store_true")

    for name in ["download-data", "prepare-data", "preflight", "run-all", "generate-report"]:
        p = sub.add_parser(name)
        _common(p)

    gpu_once = sub.add_parser("run-gpu-once")
    _common(gpu_once)
    gpu_once.add_argument("--download-data", action="store_true")
    gpu_once.add_argument("--rounds", type=int, default=3)
    gpu_once.add_argument("--batch-size", type=int, default=128)
    gpu_once.add_argument("--learning-rate", type=float, default=0.001)
    gpu_once.add_argument("--weight-decay", type=float, default=1e-4)
    gpu_once.add_argument("--num-workers", type=int, default=4)
    gpu_once.add_argument("--max-train-samples", type=int, default=None)
    gpu_once.add_argument("--max-eval-samples", type=int, default=None)
    gpu_once.add_argument("--models", default="classical_cnn,fedqcnn")
    gpu_once.add_argument("--datasets", default="pathmnist,octmnist,pneumoniamnist,retinamnist,breastmnist")
    gpu_once.add_argument("--amp", action="store_true")
    gpu_once.add_argument("--require-cuda", action="store_true")

    suite = sub.add_parser("publication-suite")
    suite.add_argument("--output-dir", default="output/publication_suite")
    suite.add_argument("--mode", default="test", choices=["test", "paper"])
    suite.add_argument("--device", default="cpu")
    suite.add_argument("--rounds", type=int, default=None)
    suite.add_argument("--seeds", default=None)
    suite.add_argument("--download-data", action="store_true")

    audit = sub.add_parser("audit-publication")
    audit.add_argument("--output-dir", default="output")
    audit.add_argument("--strict-infra", action="store_true")

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
        if args.profile == "paper":
            print(
                "[WARN] Full E1-E9 paper experiment engines are not implemented yet. "
                "Recording E1-E9 manifest entries only; use run-gpu-once for real training results.",
                file=sys.stderr,
            )
        for exp in [f"E{i}" for i in range(1, 10)]:
            args.experiment = exp
            run_experiment(args)
        return 0
    if args.command == "run-gpu-once":
        return run_gpu_once(args)
    if args.command == "publication-suite":
        seeds = None
        if args.seeds:
            seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
        cfg = PublicationSuiteConfig(
            output_dir=args.output_dir,
            mode=args.mode,
            device=args.device,
            rounds=args.rounds,
            seeds=seeds,
            download_data=args.download_data,
        )
        run_publication_suite(cfg)
        return 0
    if args.command == "audit-publication":
        return audit_publication(args)
    if args.command == "generate-report":
        return generate_report(args)
    raise ValueError(args.command)
