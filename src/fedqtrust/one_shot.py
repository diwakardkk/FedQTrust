"""One-command GPU run for external result collection.

This module is deliberately labeled as an engineering validation run. It trains
the implemented CNN/FedQCNN paths on real MedMNIST splits and writes traceable
metrics, but it does not claim to replace the full E1-E9 paper experiment grid.
"""

from __future__ import annotations

import json
import time
import zipfile
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Subset

from fedqtrust.config import DATASETS
from fedqtrust.data.datasets import DATASET_SPECS, ensure_all_datasets, label_array, load_medmnist_dataset
from fedqtrust.device import DeviceManager, collect_environment, git_commit, write_environment_report
from fedqtrust.models.classical_cnn import ClassicalCNN, count_parameters
from fedqtrust.models.fedqcnn import FedQCNN
from fedqtrust.quantum.backend import detect_quantum_backend
from fedqtrust.reproducibility import seed_everything


@dataclass
class OneShotConfig:
    output_dir: str = "output/gpu_once"
    device: str = "auto"
    rounds: int = 3
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    num_workers: int = 4
    seed: int = 42
    require_cuda: bool = False
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    models: tuple[str, ...] = ("classical_cnn", "fedqcnn")
    datasets: tuple[str, ...] = tuple(DATASETS)


def _labels_to_long(y: torch.Tensor) -> torch.Tensor:
    if y.ndim > 1:
        y = y[:, 0]
    return y.long()


def _subset(dataset, max_samples: int | None, seed: int):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=max_samples, replace=False)
    return Subset(dataset, indices.tolist())


def _loader(dataset, batch_size: int, shuffle: bool, num_workers: int, device: str) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": max(0, num_workers),
        "pin_memory": device.startswith("cuda"),
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def _build_model(name: str, num_classes: int) -> nn.Module:
    if name == "classical_cnn":
        return ClassicalCNN(num_classes)
    if name == "fedqcnn":
        return FedQCNN(num_classes)
    raise ValueError(f"unknown one-shot model: {name}")


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: str, num_classes: int) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    loss_fn = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = _labels_to_long(y).to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        probs = torch.softmax(logits, dim=1)
        total_loss += float(loss.cpu())
        n_batches += 1
        y_true.extend(y.cpu().numpy().astype(int).tolist())
        y_pred.extend(torch.argmax(probs, dim=1).cpu().numpy().astype(int).tolist())
        if num_classes == 2:
            y_score.extend(probs[:, 1].cpu().numpy().astype(float).tolist())
    out = {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auc_roc": float("nan"),
    }
    if num_classes == 2 and len(set(y_true)) == 2:
        out["auc_roc"] = float(roc_auc_score(y_true, y_score))
    return out


def _train_round(model: nn.Module, loader: DataLoader, device: str, lr: float, weight_decay: float, amp: bool) -> float:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.startswith("cuda"))
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = _labels_to_long(y).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp and device.startswith("cuda")):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    fields = list(rows[0])
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def _latex_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    fields = list(rows[0])
    lines = ["\\begin{tabular}{" + "l" * len(fields) + "}", "\\toprule", " & ".join(fields) + " \\\\", "\\midrule"]
    for row in rows:
        lines.append(" & ".join(str(row.get(field, "")) for field in fields) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _write_summary(run_dir: Path, final_rows: list[dict[str, object]], cfg: OneShotConfig, zip_path: Path) -> None:
    lines = [
        "# FedQTrust GPU One-Shot Results",
        "",
        "This run trains implemented model paths on real MedMNIST splits and records official test metrics.",
        "It is an external GPU validation bundle, not a replacement for the full E1-E9 paper grid.",
        "",
        f"- timestamp UTC: {datetime.now(timezone.utc).isoformat()}",
        f"- git commit: {git_commit()}",
        f"- rounds per dataset/model: {cfg.rounds}",
        f"- models: {', '.join(cfg.models)}",
        f"- datasets: {', '.join(cfg.datasets)}",
        f"- result zip: {zip_path.name}",
        "",
        "## Final Metrics",
        "",
        _markdown_table(final_rows),
        "",
    ]
    (run_dir / "README_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def _zip_run(run_dir: Path) -> Path:
    zip_path = run_dir.parent / f"{run_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(run_dir.parent))
    return zip_path


def run_one_shot(cfg: OneShotConfig, download_data: bool = True, amp: bool = False) -> Path:
    seed_everything(cfg.seed)
    device = DeviceManager(cfg.device).select()
    if cfg.require_cuda and not device.startswith("cuda"):
        raise RuntimeError("CUDA was required for run-gpu-once, but the selected device is not CUDA")
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    env = collect_environment(cfg.device)
    env["selected_device_for_run"] = device
    env["quantum_backend"] = asdict(detect_quantum_backend())
    (run_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    write_environment_report(Path("output") / "environment", cfg.device)

    ensure_all_datasets("data/raw", download_data, "data/metadata")

    round_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    for spec in DATASET_SPECS:
        if spec.key not in cfg.datasets:
            continue
        train_ds = _subset(load_medmnist_dataset(spec.medmnist_flag, "train", "data/raw", False), cfg.max_train_samples, cfg.seed)
        val_ds = _subset(load_medmnist_dataset(spec.medmnist_flag, "val", "data/raw", False), cfg.max_eval_samples, cfg.seed)
        test_ds = _subset(load_medmnist_dataset(spec.medmnist_flag, "test", "data/raw", False), cfg.max_eval_samples, cfg.seed)
        labels = label_array(load_medmnist_dataset(spec.medmnist_flag, "train", "data/raw", False))
        dataset_meta = {
            "dataset": spec.key,
            "display": spec.display,
            "classes": spec.classes,
            "train_samples_used": len(train_ds),
            "val_samples_used": len(val_ds),
            "test_samples_used": len(test_ds),
            "official_train_samples": len(labels),
        }
        train_loader = _loader(train_ds, cfg.batch_size, True, cfg.num_workers, device)
        val_loader = _loader(val_ds, cfg.batch_size, False, cfg.num_workers, device)
        test_loader = _loader(test_ds, cfg.batch_size, False, cfg.num_workers, device)
        for model_name in cfg.models:
            print(f"[RUN] {model_name} on {spec.display}: {cfg.rounds} rounds, device={device}", flush=True)
            model = _build_model(model_name, spec.classes).to(device)
            params = count_parameters(model)
            best_val = -1.0
            start_model = time.perf_counter()
            for round_idx in range(1, cfg.rounds + 1):
                if device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                start_round = time.perf_counter()
                lr = cfg.learning_rate / (round_idx**0.5)
                train_loss = _train_round(model, train_loader, device, lr, cfg.weight_decay, amp)
                val_metrics = _evaluate(model, val_loader, device, spec.classes)
                if val_metrics["accuracy"] > best_val:
                    best_val = val_metrics["accuracy"]
                    torch.save(model.state_dict(), run_dir / f"{model_name}_{spec.key}_best.pt")
                peak_allocated = torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0
                peak_reserved = torch.cuda.max_memory_reserved() if device.startswith("cuda") else 0
                row = {
                    **dataset_meta,
                    **params,
                    "model": model_name,
                    "round": round_idx,
                    "learning_rate": lr,
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_weighted_f1": val_metrics["weighted_f1"],
                    "val_auc_roc": val_metrics["auc_roc"],
                    "round_time_s": time.perf_counter() - start_round,
                    "gpu_peak_memory_allocated_bytes": peak_allocated,
                    "gpu_peak_memory_reserved_bytes": peak_reserved,
                }
                round_rows.append(row)
                print(
                    f"[RUN] {model_name} {spec.display} round {round_idx}/{cfg.rounds}: "
                    f"train_loss={train_loss:.4f} val_acc={val_metrics['accuracy']:.4f}",
                    flush=True,
                )
            test_metrics = _evaluate(model, test_loader, device, spec.classes)
            final_rows.append(
                {
                    **dataset_meta,
                    **params,
                    "model": model_name,
                    "rounds": cfg.rounds,
                    "best_val_accuracy": best_val,
                    "test_loss": test_metrics["loss"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_weighted_f1": test_metrics["weighted_f1"],
                    "test_auc_roc": test_metrics["auc_roc"],
                    "total_model_time_s": time.perf_counter() - start_model,
                }
            )

    _write_csv(run_dir / "round_metrics.csv", round_rows)
    _write_csv(run_dir / "final_metrics.csv", final_rows)
    (run_dir / "final_metrics.md").write_text(_markdown_table(final_rows), encoding="utf-8")
    (run_dir / "final_metrics.tex").write_text(_latex_table(final_rows), encoding="utf-8")
    _save_one_shot_plots(run_dir, round_rows, final_rows)
    (run_dir / "DONE").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    zip_path = _zip_run(run_dir)
    _write_summary(run_dir, final_rows, cfg, zip_path)
    _zip_run(run_dir)
    print(f"[DONE] Results directory: {run_dir}", flush=True)
    print(f"[DONE] Share this zip: {zip_path}", flush=True)
    return run_dir


def _save_one_shot_plots(run_dir: Path, round_rows: list[dict[str, object]], final_rows: list[dict[str, object]]) -> None:
    import os

    Path("/tmp/fedqtrust-matplotlib").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/fedqtrust-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    if round_rows:
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        groups: dict[tuple[str, str], list[dict[str, object]]] = {}
        for row in round_rows:
            groups.setdefault((str(row["display"]), str(row["model"])), []).append(row)
        for (dataset, model), part in groups.items():
            part = sorted(part, key=lambda row: int(row["round"]))
            ax.plot(
                [int(row["round"]) for row in part],
                [float(row["val_accuracy"]) for row in part],
                marker="o",
                linewidth=2,
                label=f"{dataset} {model}",
            )
        ax.set_xlabel("Round")
        ax.set_ylabel("Validation Accuracy")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
        fig.savefig(fig_dir / "validation_accuracy_by_round.pdf", bbox_inches="tight")
        fig.savefig(fig_dir / "validation_accuracy_by_round.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    if final_rows:
        displays = sorted({str(row["display"]) for row in final_rows})
        models = sorted({str(row["model"]) for row in final_rows})
        values = {
            (str(row["display"]), str(row["model"])): float(row["test_accuracy"])
            for row in final_rows
        }
        x = np.arange(len(displays))
        width = 0.8 / max(len(models), 1)
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        for idx, model in enumerate(models):
            heights = [values.get((display, model), float("nan")) for display in displays]
            ax.bar(x + idx * width - 0.4 + width / 2, heights, width=width, label=model)
        ax.set_xticks(x)
        ax.set_xticklabels(displays, rotation=20, ha="right")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Official Test Accuracy")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
        fig.savefig(fig_dir / "test_accuracy_by_dataset.pdf", bbox_inches="tight")
        fig.savefig(fig_dir / "test_accuracy_by_dataset.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
