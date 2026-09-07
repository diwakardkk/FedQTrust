"""Real E1 attack-resilience experiment runner.

The runner trains federated models on MedMNIST data and records raw per-round,
per-client, and final metrics. It does not synthesize placeholder E1 values.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Subset

from fedqtrust.attacks.base import flatten_delta
from fedqtrust.attacks.free_rider import free_ride
from fedqtrust.attacks.gaussian import gaussian_noise
from fedqtrust.attacks.lie import lie_vector, vector_to_delta
from fedqtrust.attacks.sign_flip import sign_flip
from fedqtrust.baselines.fltrust import fltrust_scores
from fedqtrust.baselines.krum import krum_select
from fedqtrust.data.datasets import DATASET_SPECS, ensure_all_datasets, load_medmnist_dataset
from fedqtrust.device import DeviceManager, collect_environment, git_commit, write_environment_report
from fedqtrust.fl.aggregation import fedavg, trust_weighted_delta
from fedqtrust.fl.state import add_delta, clone_state, subtract_state
from fedqtrust.fl.trust import TrustManager, safe_cosine
from fedqtrust.models.classical_cnn import ClassicalCNN, count_parameters
from fedqtrust.models.fedqcnn import FedQCNN
from fedqtrust.reproducibility import seed_everything
from fedqtrust.reporting.tables import save_table

try:
    pd.options.mode.string_storage = "python"
    pd.options.future.infer_string = False
except (AttributeError, KeyError, ValueError):
    pass


E1_METHODS = ["FedAvg", "FedProx", "FedQCNN", "Krum", "FLTrust", "PQS-BFL", "SecEdge-MC", "FedQTrust"]
E1_ATTACKS = ["benign", "label_flip", "sign_flip", "gaussian", "lie", "free_riding", "combined"]


@dataclass
class E1RealConfig:
    output_dir: str = "output/scientific_results/e1_attack_resilience"
    device: str = "auto"
    rounds: int = 100
    seeds: tuple[int, ...] = (42, 123, 456, 789, 999)
    datasets: tuple[str, ...] = ("pathmnist", "octmnist", "pneumoniamnist", "retinamnist", "breastmnist")
    methods: tuple[str, ...] = tuple(E1_METHODS)
    attacks: tuple[str, ...] = tuple(E1_ATTACKS)
    clients: int = 10
    clients_per_round: int = 5
    malicious_fraction: float = 0.3
    local_epochs: int = 1
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    num_workers: int = 4
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    download_data: bool = True
    amp: bool = False
    require_cuda: bool = False
    save_checkpoints: bool = True


def run_e1_real(config: E1RealConfig) -> Path:
    _validate_config(config)
    out = Path(config.output_dir)
    _prepare_dirs(out)
    _style_plots()
    device = DeviceManager(config.device).select()
    if config.require_cuda and not device.startswith("cuda"):
        raise RuntimeError("CUDA was required for E1, but CUDA was not selected.")

    env = collect_environment(config.device)
    env["selected_device_for_e1"] = device
    env["e1_git_commit"] = git_commit()
    (out / "environment" / "environment_report.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    write_environment_report(out / "environment", config.device)
    (out / "config.json").write_text(json.dumps(_jsonable(asdict(config)), indent=2), encoding="utf-8")
    ensure_all_datasets("data/raw", config.download_data, "data/metadata")

    final_rows: list[dict[str, object]] = []
    round_rows: list[dict[str, object]] = []
    client_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    specs = {spec.key: spec for spec in DATASET_SPECS}

    for seed in config.seeds:
        seed_everything(seed)
        for dataset_key in config.datasets:
            spec = specs[dataset_key]
            train_ds = load_medmnist_dataset(spec.medmnist_flag, "train", "data/raw", False)
            test_ds = load_medmnist_dataset(spec.medmnist_flag, "test", "data/raw", False)
            train_indices = _sample_indices(len(train_ds), config.max_train_samples, seed)
            test_subset = Subset(test_ds, _sample_indices(len(test_ds), config.max_eval_samples, seed + 17))
            partitions = _partition_indices(train_indices, config.clients, seed)
            test_loader = _loader(test_subset, config.batch_size, False, config.num_workers, device)

            for method in config.methods:
                for attack in config.attacks:
                    run_id = f"E1_{dataset_key}_{method}_{attack}_seed_{seed}"
                    print(f"[E1] {run_id}: rounds={config.rounds} clients={config.clients} k={config.clients_per_round}", flush=True)
                    start = time.perf_counter()
                    result = _run_condition(config, spec.classes, method, attack, seed, partitions, train_ds, test_loader, device)
                    elapsed = time.perf_counter() - start
                    final_metrics = result["final_metrics"]
                    final_rows.append(
                        {
                            "experiment": "E1",
                            "run_id": run_id,
                            "seed": seed,
                            "dataset": dataset_key,
                            "display": spec.display,
                            "method": method,
                            "attack": attack,
                            "rounds": config.rounds,
                            "clients": config.clients,
                            "clients_per_round": config.clients_per_round,
                            "malicious_fraction": 0.0 if attack == "benign" else config.malicious_fraction,
                            **final_metrics,
                            "total_time_s": elapsed,
                        }
                    )
                    round_rows.extend({**row, "run_id": run_id, "display": spec.display} for row in result["round_rows"])
                    client_rows.extend({**row, "run_id": run_id, "display": spec.display} for row in result["client_rows"])
                    timing_rows.extend({**row, "run_id": run_id, "display": spec.display} for row in result["timing_rows"])
                    if config.save_checkpoints:
                        torch.save(result["state_dict"], out / "checkpoints" / f"{run_id}.pt")
                    manifest_rows.append(
                        {
                            "run_id": run_id,
                            "experiment": "E1",
                            "dataset": dataset_key,
                            "method": method,
                            "attack": attack,
                            "seed": seed,
                            "status": "DONE",
                            "rounds": config.rounds,
                            "elapsed_s": elapsed,
                        }
                    )
                    _append_event(out / "events.jsonl", {"event": "condition_done", "run_id": run_id, "elapsed_s": elapsed})

    _write_csv(out / "raw_metrics" / "e1_final_metrics.csv", final_rows)
    _write_csv(out / "raw_metrics" / "e1_round_metrics.csv", round_rows)
    _write_csv(out / "raw_metrics" / "e1_client_metrics.csv", client_rows)
    _write_csv(out / "raw_metrics" / "e1_timing.csv", timing_rows)
    _write_csv(out / "manifest.csv", manifest_rows)
    _write_e1_tables(out, pd.DataFrame(final_rows), pd.DataFrame(client_rows), pd.DataFrame(timing_rows))
    _write_e1_figures(out, pd.DataFrame(final_rows), pd.DataFrame(round_rows), pd.DataFrame(client_rows))
    _write_e1_summary(out, config, pd.DataFrame(final_rows), env)
    (out / "DONE").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    archive = shutil.make_archive(str(out), "zip", out)
    print(f"[DONE] E1 real attack-resilience results: {out}", flush=True)
    print(f"[DONE] E1 zip bundle: {archive}", flush=True)
    return out


def _run_condition(
    config: E1RealConfig,
    num_classes: int,
    method: str,
    attack: str,
    seed: int,
    partitions: list[list[int]],
    train_ds,
    test_loader: DataLoader,
    device: str,
) -> dict[str, object]:
    model = _build_model(method, num_classes).to(device)
    global_state = clone_state(model.state_dict())
    trust = TrustManager(config.clients)
    malicious = _malicious_clients(config.clients, config.malicious_fraction, attack)
    round_rows = []
    client_rows = []
    timing_rows = []
    rng = np.random.default_rng(seed)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.startswith("cuda"))

    for rnd in range(1, config.rounds + 1):
        selected = _select_clients(config.clients, config.clients_per_round, seed, rnd)
        deltas = []
        sample_counts = []
        losses = []
        vectors = []
        client_ids = []
        start_round = time.perf_counter()
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        for client_id in selected:
            loader = _client_loader(train_ds, partitions[client_id], config, device, shuffle_seed=seed + rnd + client_id)
            delta, loss, n_samples = _train_client(
                model,
                global_state,
                loader,
                method,
                attack,
                client_id in malicious,
                num_classes,
                config.learning_rate / math.sqrt(rnd),
                config.weight_decay,
                config.local_epochs,
                device,
                scaler,
                config.amp,
            )
            deltas.append(delta)
            sample_counts.append(n_samples)
            losses.append(loss)
            vectors.append(flatten_delta(delta).numpy())
            client_ids.append(client_id)

        deltas = _apply_model_attack(deltas, vectors, client_ids, malicious, attack, seed + rnd)
        vectors = [flatten_delta(delta).numpy() for delta in deltas]
        reference = _reference_vector(vectors, client_ids, malicious)
        selected_deltas, selected_counts, selected_trust = _select_and_weight_updates(method, deltas, sample_counts, vectors, client_ids, reference, trust, config)
        avg_delta = trust_weighted_delta(selected_deltas, selected_counts, selected_trust)
        global_state = add_delta(global_state, avg_delta)
        model.load_state_dict(global_state)
        metrics = _evaluate(model, test_loader, device, num_classes)
        elapsed = time.perf_counter() - start_round
        peak_allocated = torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0
        peak_reserved = torch.cuda.max_memory_reserved() if device.startswith("cuda") else 0

        round_rows.append(
            {
                "experiment": "E1",
                "seed": seed,
                "method": method,
                "attack": attack,
                "round": rnd,
                "train_loss": float(np.mean(losses)) if losses else np.nan,
                "test_accuracy": metrics["accuracy"],
                "weighted_f1": metrics["weighted_f1"],
                "auc_roc": metrics["auc_roc"],
                "round_time_s": elapsed,
                "selected_clients": ",".join(map(str, client_ids)),
                "malicious_selected": sum(1 for client_id in client_ids if client_id in malicious),
                "gpu_peak_memory_allocated_bytes": peak_allocated,
                "gpu_peak_memory_reserved_bytes": peak_reserved,
            }
        )
        for client_id, vector, loss in zip(client_ids, vectors, losses):
            consistency = safe_cosine(vector, reference)
            client_rows.append(
                {
                    "experiment": "E1",
                    "seed": seed,
                    "method": method,
                    "attack": attack,
                    "round": rnd,
                    "client_id": client_id,
                    "is_malicious": client_id in malicious,
                    "trust_score": trust.trust.get(client_id, 0.5),
                    "consistency": consistency,
                    "client_loss": loss,
                    "selected": True,
                    "predicted_malicious": trust.trust.get(client_id, 0.5) < 0.4,
                }
            )
        timing_rows.append({"experiment": "E1", "seed": seed, "method": method, "attack": attack, "round": rnd, "round_time_s": elapsed})
        print(f"[E1] {method}/{attack} seed={seed} round {rnd}/{config.rounds} acc={metrics['accuracy']:.4f}", flush=True)

    final_metrics = _evaluate(model, test_loader, device, num_classes)
    final_metrics["best_round_accuracy"] = max(float(row["test_accuracy"]) for row in round_rows) if round_rows else np.nan
    return {"final_metrics": final_metrics, "round_rows": round_rows, "client_rows": client_rows, "timing_rows": timing_rows, "state_dict": global_state}


def _build_model(method: str, num_classes: int) -> nn.Module:
    if method in {"FedQCNN", "FedQTrust"}:
        return FedQCNN(num_classes)
    return ClassicalCNN(num_classes)


def _train_client(
    template_model: nn.Module,
    global_state: dict[str, torch.Tensor],
    loader: DataLoader,
    method: str,
    attack: str,
    is_malicious: bool,
    num_classes: int,
    lr: float,
    weight_decay: float,
    local_epochs: int,
    device: str,
    scaler: torch.amp.GradScaler,
    amp: bool,
) -> tuple[dict[str, torch.Tensor], float, int]:
    local = type(template_model)(num_classes).to(device)
    local.load_state_dict(global_state)
    opt = torch.optim.Adam(local.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    batches = 0
    n_samples = 0
    local.train()
    for _ in range(local_epochs):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = _labels_to_long(y).to(device, non_blocking=True)
            if is_malicious and attack in {"label_flip", "combined"}:
                y = (y + 1) % num_classes
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp and device.startswith("cuda")):
                logits = local(x)
                loss = loss_fn(logits, y)
                if method == "FedProx":
                    prox = torch.zeros((), device=device)
                    for name, param in local.named_parameters():
                        prox = prox + torch.sum((param - global_state[name].to(device)) ** 2)
                    loss = loss + 0.001 * prox
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += float(loss.detach().cpu())
            batches += 1
            n_samples += int(x.shape[0])
    return subtract_state(local.state_dict(), global_state), total_loss / max(batches, 1), n_samples


def _apply_model_attack(
    deltas: list[dict[str, torch.Tensor]],
    vectors: list[np.ndarray],
    client_ids: list[int],
    malicious: set[int],
    attack: str,
    seed: int,
) -> list[dict[str, torch.Tensor]]:
    if attack == "benign":
        return deltas
    out = list(deltas)
    honest_vectors = [vec for vec, client_id in zip(vectors, client_ids) if client_id not in malicious]
    template = deltas[0]
    lie_delta = vector_to_delta(lie_vector(honest_vectors or vectors, z_max=1.0), template) if attack == "lie" else None
    for idx, client_id in enumerate(client_ids):
        if client_id not in malicious:
            continue
        if attack == "sign_flip":
            out[idx] = sign_flip(out[idx])
        elif attack == "gaussian":
            out[idx] = gaussian_noise(out[idx], sigma=0.03, seed=seed + client_id)
        elif attack == "free_riding":
            out[idx] = free_ride(out[idx])
        elif attack == "lie" and lie_delta is not None:
            out[idx] = {key: value.detach().clone() for key, value in lie_delta.items()}
        elif attack == "combined" and client_id % 2 == 1:
            out[idx] = free_ride(out[idx])
    return out


def _select_and_weight_updates(
    method: str,
    deltas: list[dict[str, torch.Tensor]],
    sample_counts: list[int],
    vectors: list[np.ndarray],
    client_ids: list[int],
    reference: np.ndarray,
    trust: TrustManager,
    config: E1RealConfig,
) -> tuple[list[dict[str, torch.Tensor]], list[int], list[float]]:
    if method == "Krum":
        idx, _ = krum_select(vectors, f_requested=max(1, int(config.clients * config.malicious_fraction)))
        return [deltas[idx]], [sample_counts[idx]], [1.0]
    if method == "FLTrust":
        scores = fltrust_scores(vectors, reference)
        weights = [float(max(score, 1e-6)) for score in scores]
        return deltas, sample_counts, weights
    if method in {"FedQTrust", "SecEdge-MC", "PQS-BFL"}:
        scores = []
        for idx, client_id in enumerate(client_ids):
            consistency = trust.update_consistency(client_id, vectors[idx], reference)
            validation_proxy = max(0.0, min(1.0, 0.5 + 0.5 * consistency))
            tau = trust.update_round(client_id, validation_proxy, consistency)
            distance = float(np.linalg.norm(vectors[idx] - reference))
            scores.append((idx, tau / (1.0 + distance)))
        if method == "FedQTrust":
            keep = [idx for idx, _ in scores if trust.trust.get(client_ids[idx], 0.0) >= 0.4]
            if not keep:
                keep = [idx for idx, _ in sorted(scores, key=lambda item: item[1], reverse=True)[: max(1, len(scores) // 2)]]
            return [deltas[i] for i in keep], [sample_counts[i] for i in keep], [trust.trust.get(client_ids[i], 0.5) for i in keep]
        return deltas, sample_counts, [trust.trust.get(client_id, 0.5) for client_id in client_ids]
    return deltas, sample_counts, [1.0] * len(deltas)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: str, num_classes: int) -> dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    batches = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = _labels_to_long(y).to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        probs = torch.softmax(logits, dim=1)
        total_loss += float(loss.detach().cpu())
        batches += 1
        y_true.extend(y.detach().cpu().numpy().astype(int).tolist())
        y_pred.extend(torch.argmax(probs, dim=1).detach().cpu().numpy().astype(int).tolist())
        if num_classes == 2:
            y_score.extend(probs[:, 1].detach().cpu().numpy().astype(float).tolist())
    metrics = {
        "test_loss": total_loss / max(batches, 1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auc_roc": float("nan"),
    }
    if num_classes == 2 and len(set(y_true)) == 2:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def _write_e1_tables(out: Path, final: pd.DataFrame, clients: pd.DataFrame, timing: pd.DataFrame) -> None:
    table_dir = out / "tables"
    main = final.groupby(["method", "attack"], as_index=False)[["accuracy", "weighted_f1", "auc_roc"]].mean()
    per_dataset = final.groupby(["dataset", "method", "attack"], as_index=False)[["accuracy", "weighted_f1", "auc_roc"]].mean()
    detection = _detection_table(clients)
    timing_summary = timing.groupby(["method", "attack"], as_index=False)["round_time_s"].agg(["mean", "std"]).reset_index()
    stats = _e1_stats(final)
    for stem, df in {
        "E1_real_main_results": main,
        "E1_real_per_dataset_results": per_dataset,
        "E1_real_detection_results": detection,
        "E1_real_timing_summary": timing_summary,
        "E1_real_statistical_tests": stats,
    }.items():
        save_table(df, table_dir, stem)


def _write_e1_figures(out: Path, final: pd.DataFrame, rounds: pd.DataFrame, clients: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    fig, ax = plt.subplots(figsize=(13, 7))
    pivot = final.groupby(["attack", "method"])["accuracy"].mean().unstack("method")
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("E1 real attack resilience", fontweight="bold", pad=14)
    ax.set_xlabel("Attack")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1)
    ax.legend(ncol=2, frameon=True)
    _save(fig, fig_dir / "Fig_E1_real_attack_resilience")

    fig, ax = plt.subplots(figsize=(13, 7))
    subset = rounds.query("attack in ['benign', 'label_flip', 'lie', 'combined']")
    grouped = subset.groupby(["method", "attack", "round"], as_index=False)["test_accuracy"].mean()
    for (method, attack), group in grouped.groupby(["method", "attack"]):
        if method in {"FedAvg", "Krum", "FLTrust", "FedQTrust"}:
            ax.plot(group["round"], group["test_accuracy"], label=f"{method} {attack}")
    ax.set_title("E1 real convergence under attacks", fontweight="bold", pad=14)
    ax.set_xlabel("Round")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1)
    ax.legend(ncol=2, frameon=True)
    _save(fig, fig_dir / "Fig_E1_real_convergence")

    fig, ax = plt.subplots(figsize=(12, 7))
    if not clients.empty:
        trust = clients.query("method == 'FedQTrust'").groupby(["attack", "round", "is_malicious"], as_index=False)["trust_score"].mean()
        for (attack, malicious), group in trust.groupby(["attack", "is_malicious"]):
            if attack in {"label_flip", "lie", "combined"}:
                ax.plot(group["round"], group["trust_score"], label=f"{attack} malicious={malicious}")
    ax.set_title("E1 real FedQTrust trust scores", fontweight="bold", pad=14)
    ax.set_xlabel("Round")
    ax.set_ylabel("Trust score")
    ax.set_ylim(0, 1)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=True)
    _save(fig, fig_dir / "Fig_E1_real_trust_scores")
    plt.close("all")


def _write_e1_summary(out: Path, config: E1RealConfig, final: pd.DataFrame, env: dict) -> None:
    best = final.sort_values("accuracy", ascending=False).head(1)
    best_line = ""
    if not best.empty:
        row = best.iloc[0]
        best_line = f"- best condition: {row['method']} / {row['attack']} / {row['dataset']} accuracy={row['accuracy']:.4f}"
    lines = [
        "# E1 Real Attack-Resilience Summary",
        "",
        "This E1 folder was produced by real federated training runs over the configured methods, attacks, datasets, and seeds.",
        "",
        f"- methods: {', '.join(config.methods)}",
        f"- attacks: {', '.join(config.attacks)}",
        f"- datasets: {', '.join(config.datasets)}",
        f"- seeds: {', '.join(map(str, config.seeds))}",
        f"- rounds: {config.rounds}",
        f"- clients: {config.clients}",
        f"- clients per round: {config.clients_per_round}",
        f"- CUDA available: {env.get('cuda_available')}",
        f"- git commit: {git_commit()}",
        best_line,
        "",
        "Use `raw_metrics/e1_final_metrics.csv` and `raw_metrics/e1_round_metrics.csv` as the source of truth.",
    ]
    (out / "summaries" / "E1_real_summary.md").write_text("\n".join(line for line in lines if line) + "\n", encoding="utf-8")


def _detection_table(clients: pd.DataFrame) -> pd.DataFrame:
    if clients.empty:
        return pd.DataFrame()
    rows = []
    for (method, attack), group in clients.groupby(["method", "attack"]):
        truth = group["is_malicious"].astype(bool)
        pred = group["predicted_malicious"].astype(bool)
        tp = int((truth & pred).sum())
        fp = int((~truth & pred).sum())
        fn = int((truth & ~pred).sum())
        tn = int((~truth & ~pred).sum())
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * tpr / max(precision + tpr, 1e-12)
        rows.append({"method": method, "attack": attack, "TPR": tpr, "FPR": fpr, "precision": precision, "F1": f1})
    return pd.DataFrame(rows)


def _e1_stats(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for attack, group in final.groupby("attack"):
        fed = group.query("method == 'FedQTrust'").sort_values(["dataset", "seed"])["accuracy"].to_numpy()
        for method, comp_group in group.groupby("method"):
            if method == "FedQTrust":
                continue
            comp = comp_group.sort_values(["dataset", "seed"])["accuracy"].to_numpy()
            n = min(len(fed), len(comp))
            p_value = float("nan")
            if n >= 2:
                try:
                    from scipy.stats import wilcoxon

                    p_value = float(wilcoxon(fed[:n], comp[:n]).pvalue)
                except ValueError:
                    p_value = float("nan")
            rows.append({"attack": attack, "comparison": f"FedQTrust vs {method}", "n_pairs": n, "mean_delta": float(np.mean(fed[:n] - comp[:n])) if n else np.nan, "wilcoxon_p": p_value})
    return pd.DataFrame(rows)


def _sample_indices(length: int, max_samples: int | None, seed: int) -> list[int]:
    indices = np.arange(length)
    if max_samples is not None and max_samples < length:
        rng = np.random.default_rng(seed)
        indices = rng.choice(indices, size=max(1, max_samples), replace=False)
    return indices.astype(int).tolist()


def _partition_indices(indices: list[int], clients: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    shuffled = np.array(indices, dtype=int)
    rng.shuffle(shuffled)
    return [part.astype(int).tolist() for part in np.array_split(shuffled, clients)]


def _select_clients(total: int, k: int, seed: int, rnd: int) -> list[int]:
    rng = np.random.default_rng(seed * 1000003 + rnd)
    return sorted(rng.choice(total, size=min(k, total), replace=False).astype(int).tolist())


def _malicious_clients(total: int, fraction: float, attack: str) -> set[int]:
    if attack == "benign":
        return set()
    return set(range(max(1, int(round(total * fraction)))))


def _reference_vector(vectors: list[np.ndarray], client_ids: list[int], malicious: set[int]) -> np.ndarray:
    honest = [vector for vector, client_id in zip(vectors, client_ids) if client_id not in malicious]
    return np.mean(honest or vectors, axis=0)


def _client_loader(train_ds, indices: list[int], config: E1RealConfig, device: str, shuffle_seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(shuffle_seed)
    kwargs = {
        "batch_size": config.batch_size,
        "shuffle": True,
        "num_workers": max(0, config.num_workers),
        "pin_memory": device.startswith("cuda"),
        "generator": generator,
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(Subset(train_ds, indices), **kwargs)


def _loader(dataset, batch_size: int, shuffle: bool, num_workers: int, device: str) -> DataLoader:
    kwargs = {"batch_size": batch_size, "shuffle": shuffle, "num_workers": max(0, num_workers), "pin_memory": device.startswith("cuda")}
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def _labels_to_long(y: torch.Tensor) -> torch.Tensor:
    if y.ndim > 1:
        y = y[:, 0]
    return y.long()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _append_event(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **row}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.08)


def _prepare_dirs(out: Path) -> None:
    for sub in ["environment", "raw_metrics", "tables", "figures", "summaries", "checkpoints"]:
        (out / sub).mkdir(parents=True, exist_ok=True)


def _style_plots() -> None:
    import os

    mpl_cache = Path("/tmp/fedqtrust-matplotlib")
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 14, "axes.titlesize": 18, "axes.labelsize": 16, "legend.fontsize": 11, "figure.constrained_layout.use": True})


def _validate_config(config: E1RealConfig) -> None:
    known_datasets = {spec.key for spec in DATASET_SPECS}
    unknown_datasets = set(config.datasets) - known_datasets
    unknown_methods = set(config.methods) - set(E1_METHODS)
    unknown_attacks = set(config.attacks) - set(E1_ATTACKS)
    if unknown_datasets:
        raise ValueError(f"unknown E1 datasets: {sorted(unknown_datasets)}")
    if unknown_methods:
        raise ValueError(f"unknown E1 methods: {sorted(unknown_methods)}")
    if unknown_attacks:
        raise ValueError(f"unknown E1 attacks: {sorted(unknown_attacks)}")
    if config.clients_per_round > config.clients:
        raise ValueError("clients_per_round cannot exceed clients")


def _jsonable(value):
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
