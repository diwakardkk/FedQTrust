"""End-to-end smoke test required by the implementation brief."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from fedqtrust.attacks.combined import combined_attack_sets
from fedqtrust.attacks.free_rider import free_ride
from fedqtrust.attacks.gaussian import gaussian_noise
from fedqtrust.attacks.label_flip import flip_labels
from fedqtrust.attacks.lie import lie_vector
from fedqtrust.attacks.sign_flip import sign_flip
from fedqtrust.blockchain.memory_backend import InMemoryBlockchain
from fedqtrust.config import DATASETS, load_config
from fedqtrust.data.datasets import DATASET_SPECS, ensure_all_datasets, label_array, load_medmnist_dataset
from fedqtrust.data.partition import dirichlet_partition, save_partitions, write_data_stats
from fedqtrust.device import git_commit, write_environment_report
from fedqtrust.fl.aggregation import trust_weighted_delta
from fedqtrust.fl.trainer import tiny_fl_run
from fedqtrust.fl.trust import TrustManager
from fedqtrust.models.classical_cnn import ClassicalCNN, count_parameters
from fedqtrust.models.fedqcnn import FedQCNN
from fedqtrust.quantum.backend import detect_quantum_backend
from fedqtrust.quantum.exact_solver import solve_exact
from fedqtrust.quantum.qaoa import solve_qaoa
from fedqtrust.quantum.qubo import QuboInstance, pairwise_cosine_distances, project_cardinality
from fedqtrust.quantum.simulated_annealing import solve_sa
from fedqtrust.reporting.plots import save_smoke_plot
from fedqtrust.reporting.summary import append_note
from fedqtrust.reporting.tables import save_table
from fedqtrust.security.pqc import enabled_mechanisms, oqs_available


def _status(name: str, ok: bool, detail: str = "") -> dict[str, str]:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}{': ' + detail if detail else ''}", flush=True)
    return {"check": name, "status": mark, "detail": detail}


def _synthetic_delta() -> dict[str, torch.Tensor]:
    return {"w": torch.tensor([1.0, -2.0, 0.5]), "b": torch.tensor([0.2])}


def run_smoke(download_data: bool, device: str, output_dir: str, config_path: str | None = None) -> int:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path, profile="smoke", device=device, output_dir=output_dir)
    env_dir = out / "environment"
    results: list[dict[str, str]] = []
    skipped: list[str] = []
    failed = False

    try:
        env = write_environment_report(env_dir, device)
        write_environment_report("output/environment", device)
        print(json.dumps(env, indent=2), flush=True)
        selected_device = env["selected_device"]
        results.append(_status("environment report", True, selected_device))
    except Exception as exc:
        results.append(_status("environment report", False, str(exc)))
        return 1

    try:
        manifest = ensure_all_datasets(cfg.data.root, download_data, "data/metadata")
        results.append(_status("MedMNIST datasets", True, f"{len(manifest)} checked"))
    except Exception as exc:
        results.append(_status("MedMNIST datasets", False, str(exc)))
        failed = True
        manifest = []

    splits_by_dataset = {}
    if not failed:
        try:
            for spec in DATASET_SPECS:
                train = load_medmnist_dataset(spec.medmnist_flag, "train", cfg.data.root, False)
                labels = label_array(train)
                splits, notes = dirichlet_partition(labels, 2, cfg.data.partition_alpha, cfg.data.partition_seed, max_retries=2)
                save_partitions(spec.key, splits, cfg.data.partitions, cfg.data.partition_seed)
                splits_by_dataset[spec.key] = splits
                for note in notes:
                    append_note("output", f"{spec.display}: {note}")
                subset = Subset(train, splits[0].train_indices[: min(4, len(splits[0].train_indices))])
                batch = next(iter(DataLoader(subset, batch_size=min(2, len(subset)))))
                if batch[0].shape[1:] != (1, 28, 28):
                    raise ValueError(f"{spec.display} batch shape {tuple(batch[0].shape)}")
            write_data_stats(splits_by_dataset, "output/tables/data_stats.csv")
            results.append(_status("partitions and one batch per dataset", True))
        except Exception as exc:
            results.append(_status("partitions and batches", False, str(exc)))
            failed = True

    try:
        x = torch.rand(4, 1, 28, 28)
        y = torch.tensor([0, 1, 0, 1])
        classical = ClassicalCNN(2).to(selected_device)
        opt = torch.optim.Adam(classical.parameters(), lr=0.001)
        logits = classical(x.to(selected_device))
        loss = nn.CrossEntropyLoss()(logits, y.to(selected_device))
        loss.backward()
        opt.step()
        params = count_parameters(classical)
        results.append(_status("classical CNN forward/backward", True, f"{params['total_trainable_parameters']} parameters"))
    except Exception as exc:
        results.append(_status("classical CNN forward/backward", False, str(exc)))
        failed = True

    try:
        fedqcnn = FedQCNN(2).to(selected_device)
        logits = fedqcnn(x[:1].to(selected_device))
        qloss = logits.sum()
        qloss.backward()
        grad_ok = fedqcnn.quantum.quantum_weights.grad is not None
        if not grad_ok:
            raise RuntimeError("quantum gradient is None")
        (out / "quantum_circuit_metadata.json").write_text(json.dumps(fedqcnn.quantum.circuit_metadata(), indent=2), encoding="utf-8")
        results.append(_status("FedQCNN tiny quantum forward/backward", True))
    except Exception as exc:
        results.append(_status("FedQCNN tiny quantum forward/backward", False, str(exc)))
        failed = True

    try:
        tm = TrustManager(4)
        initial = tm.trust[0]
        if abs(initial - 0.65) > 1e-12:
            raise RuntimeError(f"initial trust was {initial}")
        tm.update_round(0, 0.7, 0.8)
        results.append(_status("trust equations", True, "initial trust 0.65"))
    except Exception as exc:
        results.append(_status("trust equations", False, str(exc)))
        failed = True

    try:
        trust = np.array([0.9, 0.8, 0.3, 0.4])
        updates = [np.array([1, 0, 0]), np.array([0.9, 0.1, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0])]
        distances = pairwise_cosine_distances(updates)
        instance = QuboInstance(trust, distances, 1.0, 2.0, 2, [0, 1, 2, 3])
        exact = solve_exact(instance)
        sa = solve_sa(instance, seed=42, steps=100)
        qaoa = solve_qaoa(instance, strict=False)
        projected = project_cardinality(np.array([1, 1, 1, 0]), trust, 2)
        if exact.bitstring.sum() != 2 or sa.bitstring.sum() != 2 or projected.sum() != 2:
            raise RuntimeError("cardinality check failed")
        if qaoa.solver_used.endswith("SKIP"):
            skipped.append("QAOA dependency stack unavailable; exact compatibility check used")
        results.append(_status("QUBO exact/SA/QAOA/cardinality", True, qaoa.solver_used))
    except Exception as exc:
        results.append(_status("QUBO exact/SA/QAOA/cardinality", False, str(exc)))
        failed = True

    try:
        labels = np.array([0, 1, 1, 0, 1])
        flipped = flip_labels(labels, 2, 0.4, 42)
        delta = _synthetic_delta()
        _ = sign_flip(delta)
        _ = gaussian_noise(delta, 0.1, 42)
        _ = free_ride(delta)
        _ = lie_vector([np.array([1.0, 2.0]), np.array([1.5, 1.0])], 1.0)
        _ = combined_attack_sets(40)
        if np.array_equal(labels, flipped):
            raise RuntimeError("label flip did not change sampled labels")
        results.append(_status("attack transformations", True))
    except Exception as exc:
        results.append(_status("attack transformations", False, str(exc)))
        failed = True

    try:
        d1 = {"w": torch.tensor([1.0, 3.0])}
        d2 = {"w": torch.tensor([3.0, 5.0])}
        agg = trust_weighted_delta([d1, d2], [1, 3], [0.5, 1.0])
        expected = (d1["w"] * 0.5 + d2["w"] * 3.0) / 3.5
        if not torch.allclose(agg["w"], expected):
            raise RuntimeError("aggregation mismatch")
        bc = InMemoryBlockchain()
        bc.set_trust(1, 0.7)
        bc.log_selection({"round": 0, "selected": [0, 1]})
        if not bc.ping() or bc.get_trust(1) != 0.7:
            raise RuntimeError("in-memory blockchain failed")
        results.append(_status("aggregation and in-memory blockchain", True))
    except Exception as exc:
        results.append(_status("aggregation and in-memory blockchain", False, str(exc)))
        failed = True

    try:
        fl_rows = tiny_fl_run(selected_device)
        pd.DataFrame(fl_rows).to_csv(out / "tiny_fl_round_metrics.csv", index=False)
        results.append(_status("tiny 2-round FL run", True))
    except Exception as exc:
        results.append(_status("tiny 2-round FL run", False, str(exc)))
        failed = True

    try:
        save_smoke_plot(out)
        save_table(pd.DataFrame(results), out, "smoke_checks")
        results.append(_status("plot and table generation", True))
    except Exception as exc:
        results.append(_status("plot and table generation", False, str(exc)))
        failed = True

    q_backend = detect_quantum_backend()
    if not q_backend.gpu_available:
        skipped.append(f"Qiskit Aer GPU unavailable: {q_backend.fallback_reason}")
    if oqs_available():
        mechs = enabled_mechanisms()
        if not mechs["kem"] or not mechs["sig"]:
            skipped.append("liboqs import worked but mechanisms were not enumerable")
        else:
            results.append(_status("liboqs mechanism enumeration", True, f"{len(mechs['kem'])} KEM, {len(mechs['sig'])} SIG"))
    else:
        skipped.append("liboqs unavailable")
    skipped.append("Fabric external service skipped in smoke; InMemoryBlockchain tested")

    summary = {"checks": results, "skipped": skipped}
    (out / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not failed:
        success = "\n".join(
            [
                f"timestamp: {datetime.now(timezone.utc).isoformat()}",
                f"Git commit: {git_commit()}",
                f"datasets checked: {','.join(row['dataset'] for row in manifest)}",
                f"PyTorch device: {selected_device}",
                f"quantum device: {q_backend.quantum_device}",
                f"tests passed: {sum(1 for row in results if row['status'] == 'PASS')}",
                f"tests skipped: {len(skipped)}",
            ]
        )
        (out / "SMOKE_TEST_PASSED.txt").write_text(success + "\n", encoding="utf-8")
        print("[PASS] SMOKE TEST COMPLETE", flush=True)
        return 0
    print("[FAIL] SMOKE TEST FAILED", file=sys.stderr, flush=True)
    return 1
