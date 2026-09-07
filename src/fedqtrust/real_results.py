"""Build publication-ready artifacts by reading real training outputs.

This module is intentionally different from ``publication_suite``: it does not
create E1-E9 placeholder metrics. It reads completed run directories containing
``final_metrics.csv`` and ``round_metrics.csv`` and then writes plots, tables,
statistics, summaries, and an audit for claims that are directly supported by
those raw metrics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from fedqtrust.device import collect_environment, git_commit, write_environment_report
from fedqtrust.reporting.tables import save_table

try:
    pd.options.mode.string_storage = "python"
    pd.options.future.infer_string = False
except (AttributeError, KeyError, ValueError):
    pass


REAL_FIGURES = [
    "Fig01_real_validation_accuracy_by_round",
    "Fig02_real_test_accuracy_by_dataset",
    "Fig03_real_training_loss_by_round",
    "Fig04_real_training_time_by_dataset",
    "Fig05_real_gpu_memory_by_round",
]
REAL_TABLES = [
    "real_run_manifest",
    "real_final_metrics",
    "real_model_comparison",
    "real_convergence_summary",
    "real_timing_summary",
    "real_checkpoint_manifest",
    "real_statistical_tests",
]


@dataclass
class RealResultsConfig:
    source_dir: str = "output/scientific_results/raw_training"
    output_dir: str = "output/scientific_results"
    device: str = "auto"
    strict_infra: bool = True


@dataclass
class RealResultsIssue:
    severity: str
    item: str
    message: str


def run_real_results_suite(config: RealResultsConfig) -> Path:
    out = Path(config.output_dir)
    source = Path(config.source_dir)
    _prepare_dirs(out)
    _style_plots()

    runs = discover_completed_runs(source)
    if not runs:
        raise RuntimeError(
            f"No completed real training runs found under {source}. "
            "Run `python -m fedqtrust run-gpu-once ...` first."
        )

    final_df, round_df, manifest_df = load_real_metrics(runs)
    if final_df.empty or round_df.empty:
        raise RuntimeError("Completed run directories were found, but metric CSV files were empty.")

    env = collect_environment(config.device)
    env["real_results_source_dir"] = str(source)
    env["real_results_run_count"] = len(runs)
    env["real_results_git_commit"] = git_commit()
    (out / "environment" / "environment_report.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    write_environment_report(out / "environment", config.device)
    (out / "config.json").write_text(json.dumps({**config.__dict__, "runs": [str(run) for run in runs]}, indent=2), encoding="utf-8")

    final_df.to_csv(out / "raw_metrics" / "final_metrics_all.csv", index=False)
    round_df.to_csv(out / "raw_metrics" / "round_metrics_all.csv", index=False)
    manifest_df.to_csv(out / "raw_metrics" / "real_run_manifest.csv", index=False)

    tables = _write_real_tables(out, final_df, round_df, manifest_df)
    _write_real_figures(out, final_df, round_df)
    _write_real_statistics(out, tables["real_statistical_tests"])
    _write_real_summaries(out, config, final_df, round_df, manifest_df)
    write_real_results_audit(out, config.strict_infra)
    archive = shutil.make_archive(str(out), "zip", out)
    print(f"[DONE] Real-results suite artifacts: {out}", flush=True)
    print(f"[DONE] Zip bundle: {archive}", flush=True)
    return out


def discover_completed_runs(source_dir: str | Path) -> list[Path]:
    source = Path(source_dir)
    if not source.exists():
        return []
    candidates = [source] if (source / "DONE").exists() else []
    candidates.extend(path.parent for path in source.rglob("DONE"))
    runs = []
    seen = set()
    for run in candidates:
        resolved = run.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (run / "final_metrics.csv").is_file() and (run / "round_metrics.csv").is_file():
            runs.append(run)
    return sorted(runs)


def load_real_metrics(runs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final_frames = []
    round_frames = []
    manifest_rows = []
    for run in runs:
        final_path = run / "final_metrics.csv"
        round_path = run / "round_metrics.csv"
        final = pd.read_csv(final_path)
        rounds = pd.read_csv(round_path)
        run_id = run.name
        final["source_run_id"] = run_id
        final["source_run_dir"] = str(run)
        rounds["source_run_id"] = run_id
        rounds["source_run_dir"] = str(run)
        final_frames.append(final)
        round_frames.append(rounds)
        manifest_rows.append(_manifest_row(run, final_path, round_path))
    return pd.concat(final_frames, ignore_index=True), pd.concat(round_frames, ignore_index=True), pd.DataFrame(manifest_rows)


def audit_real_results_outputs(output_dir: str | Path, strict_infra: bool = True) -> tuple[bool, list[RealResultsIssue]]:
    out = Path(output_dir)
    issues: list[RealResultsIssue] = []
    env_path = out / "environment" / "environment_report.json"
    env = _read_json(env_path)

    if strict_infra:
        if not env.get("cuda_available"):
            issues.append(RealResultsIssue("BLOCKER", "environment", "CUDA was not available in the recorded environment."))
        if not env.get("liboqs_available"):
            issues.append(RealResultsIssue("BLOCKER", "environment", "liboqs was not available; PQC environment evidence is missing."))
        if not env.get("hyperledger_fabric_available"):
            issues.append(RealResultsIssue("BLOCKER", "environment", "Hyperledger Fabric was not available; blockchain environment evidence is missing."))

    for path in [
        out / "raw_metrics" / "final_metrics_all.csv",
        out / "raw_metrics" / "round_metrics_all.csv",
        out / "raw_metrics" / "real_run_manifest.csv",
        out / "summaries" / "final_summary.md",
        out / "summaries" / "scientific_claims_supported.md",
    ]:
        if not _exists_nonempty(path):
            issues.append(RealResultsIssue("BLOCKER", path.name, f"Missing nonempty file: {path}."))

    for stem in REAL_FIGURES:
        for suffix in ("pdf", "png"):
            path = out / "figures" / f"{stem}.{suffix}"
            if not _exists_nonempty(path):
                issues.append(RealResultsIssue("BLOCKER", stem, f"Missing nonempty figure: {path}."))

    for stem in REAL_TABLES:
        for suffix in ("csv", "md", "tex"):
            path = out / "tables" / f"{stem}.{suffix}"
            if not _exists_nonempty(path):
                issues.append(RealResultsIssue("BLOCKER", stem, f"Missing nonempty table: {path}."))

    final_path = out / "raw_metrics" / "final_metrics_all.csv"
    round_path = out / "raw_metrics" / "round_metrics_all.csv"
    if final_path.exists():
        final = pd.read_csv(final_path)
        if len(final) < 1:
            issues.append(RealResultsIssue("BLOCKER", "final_metrics", "No final metric rows were recorded."))
        if "test_accuracy" not in final.columns:
            issues.append(RealResultsIssue("BLOCKER", "final_metrics", "Missing test_accuracy column."))
    if round_path.exists():
        rounds = pd.read_csv(round_path)
        if len(rounds) < 1:
            issues.append(RealResultsIssue("BLOCKER", "round_metrics", "No round metric rows were recorded."))
        if "val_accuracy" not in rounds.columns:
            issues.append(RealResultsIssue("BLOCKER", "round_metrics", "Missing val_accuracy column."))

    if not list(out.rglob("*.pt")):
        issues.append(RealResultsIssue("BLOCKER", "checkpoints", "No model checkpoints were found inside the result bundle."))

    ready = not any(issue.severity == "BLOCKER" for issue in issues)
    return ready, issues


def write_real_results_audit(output_dir: str | Path, strict_infra: bool = True) -> Path:
    out = Path(output_dir)
    path = out / "summaries" / "real_results_audit.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    ready, issues = audit_real_results_outputs(out, strict_infra)
    lines = [
        "# Real Results Audit",
        "",
        f"Status: {'REAL_TRAINING_ARTIFACTS_PRESENT' if ready else 'NOT_READY'}",
        f"Strict infrastructure checks: {strict_infra}",
        "",
    ]
    if issues:
        lines.append("## Issues")
        lines.append("")
        for issue in issues:
            lines.append(f"- **{issue.severity}** `{issue.item}`: {issue.message}")
    else:
        lines.append("No blocking real-result issues found.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _prepare_dirs(out: Path) -> None:
    for sub in ["environment", "figures", "tables", "statistics", "summaries", "raw_metrics"]:
        (out / sub).mkdir(parents=True, exist_ok=True)


def _style_plots() -> None:
    import os

    mpl_cache = Path("/tmp/fedqtrust-matplotlib")
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 15,
            "axes.titlesize": 19,
            "axes.labelsize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "lines.linewidth": 2.8,
            "lines.markersize": 6,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.constrained_layout.use": True,
        }
    )


def _write_real_tables(out: Path, final: pd.DataFrame, rounds: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    checkpoint_manifest = _checkpoint_manifest(out)
    tables = {
        "real_run_manifest": manifest,
        "real_final_metrics": final,
        "real_model_comparison": _model_comparison(final),
        "real_convergence_summary": _convergence_summary(rounds),
        "real_timing_summary": _timing_summary(final, rounds),
        "real_checkpoint_manifest": checkpoint_manifest,
        "real_statistical_tests": _statistical_tests(final),
    }
    table_dir = out / "tables"
    for stem in REAL_TABLES:
        save_table(tables[stem], table_dir, stem)
    return tables


def _write_real_figures(out: Path, final: pd.DataFrame, rounds: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    _plot_validation_accuracy(fig_dir, rounds)
    _plot_test_accuracy(fig_dir, final)
    _plot_training_loss(fig_dir, rounds)
    _plot_training_time(fig_dir, final)
    _plot_gpu_memory(fig_dir, rounds)
    plt.close("all")


def _write_real_statistics(out: Path, stats: pd.DataFrame) -> None:
    stats_dir = out / "statistics"
    stats.to_csv(stats_dir / "real_statistical_tests.csv", index=False)
    summary = {
        "alpha": 0.05,
        "comparisons": int(len(stats)),
        "significant_unadjusted": int((stats.get("p_value", pd.Series(dtype=float)) < 0.05).sum()) if not stats.empty else 0,
        "note": "Tests are computed only from real metric rows with paired model results.",
    }
    (stats_dir / "statistical_test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _write_real_summaries(out: Path, config: RealResultsConfig, final: pd.DataFrame, rounds: pd.DataFrame, manifest: pd.DataFrame) -> None:
    datasets = sorted(str(item) for item in final.get("dataset", pd.Series(dtype=str)).dropna().unique())
    models = sorted(str(item) for item in final.get("model", pd.Series(dtype=str)).dropna().unique())
    max_round = int(rounds["round"].max()) if "round" in rounds and not rounds.empty else 0
    best = final.sort_values("test_accuracy", ascending=False).head(1)
    best_line = "No final accuracy rows were available."
    if not best.empty:
        row = best.iloc[0]
        best_line = f"Best observed test accuracy: {float(row['test_accuracy']):.4f} for {row.get('model')} on {row.get('display', row.get('dataset'))}."

    summary = [
        "# FedQTrust Real Training Results",
        "",
        "This folder was built by reading completed training output CSV files. It does not synthesize E1-E9 paper metrics.",
        "",
        f"- source directory: {config.source_dir}",
        f"- completed runs read: {len(manifest)}",
        f"- datasets: {', '.join(datasets)}",
        f"- models: {', '.join(models)}",
        f"- max recorded rounds: {max_round}",
        f"- git commit: {git_commit()}",
        f"- generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"- {best_line}",
        "",
        "Use the raw CSV files in `raw_metrics/` as the source of truth for manuscript numbers.",
    ]
    (out / "summaries" / "final_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    e1_present = (out / "e1_attack_resilience" / "DONE").exists()
    supported = [
        "- Real MedMNIST training/test metrics for the models and datasets listed in `raw_metrics/final_metrics_all.csv`.",
        "- Validation convergence over the recorded rounds in `raw_metrics/round_metrics_all.csv`.",
        "- GPU/CPU runtime and peak memory evidence when recorded by the training runner.",
        "- Checkpoint evidence for trained model states included in the bundle.",
    ]
    unsupported = [
        "- Full E2 QAOA-vs-baseline client-selection claims.",
        "- Full E3-E9 paper-grid claims.",
        "- Claims based on missing seeds, attacks, methods, datasets, or rounds.",
    ]
    if e1_present:
        supported.append("- Real E1 attack/baseline evidence from `e1_attack_resilience/raw_metrics/e1_final_metrics.csv`.")
    else:
        unsupported.insert(0, "- Full E1 attack-resilience claims across all attacks and baselines.")

    claims = [
        "# Scientific Claims Supported",
        "",
        "Supported by this bundle:",
        "",
        *supported,
        "",
        "Not supported by this bundle unless separate raw experiment outputs are added:",
        "",
        *unsupported,
        "",
        "This distinction is deliberate so paper text cannot accidentally cite generated placeholder values.",
    ]
    (out / "summaries" / "scientific_claims_supported.md").write_text("\n".join(claims) + "\n", encoding="utf-8")


def _manifest_row(run: Path, final_path: Path, round_path: Path) -> dict[str, object]:
    env = _read_json(run / "environment.json")
    summary = _read_json(run / "summary.json")
    return {
        "run_id": run.name,
        "run_dir": str(run),
        "final_metrics_sha256": _sha256(final_path),
        "round_metrics_sha256": _sha256(round_path),
        "done_file": str(run / "DONE"),
        "selected_device": env.get("selected_device_for_run") or env.get("selected_device"),
        "cuda_available": env.get("cuda_available"),
        "gpu_count": env.get("gpu_count"),
        "git_commit": env.get("git_commit") or git_commit(),
        "completed_datasets": summary.get("completed_datasets", ""),
    }


def _checkpoint_manifest(out: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(out.rglob("*.pt")):
        rows.append({"checkpoint": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return pd.DataFrame(rows, columns=["checkpoint", "bytes", "sha256"])


def _model_comparison(final: pd.DataFrame) -> pd.DataFrame:
    if "model" not in final or "test_accuracy" not in final:
        return pd.DataFrame()
    grouped = final.groupby(["model"], as_index=False).agg(
        test_accuracy_mean=("test_accuracy", "mean"),
        test_accuracy_std=("test_accuracy", "std"),
        weighted_f1_mean=("test_weighted_f1", "mean") if "test_weighted_f1" in final else ("test_accuracy", "mean"),
        total_time_s_mean=("total_model_time_s", "mean") if "total_model_time_s" in final else ("test_accuracy", "count"),
    )
    return grouped


def _convergence_summary(rounds: pd.DataFrame) -> pd.DataFrame:
    if rounds.empty:
        return pd.DataFrame()
    group_cols = [col for col in ["source_run_id", "dataset", "display", "model"] if col in rounds.columns]
    rows = []
    for keys, group in rounds.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = dict(zip(group_cols, keys))
        ordered = group.sort_values("round")
        rows.append(
            {
                **values,
                "rounds": int(ordered["round"].max()),
                "first_val_accuracy": float(ordered["val_accuracy"].iloc[0]) if "val_accuracy" in ordered else np.nan,
                "best_val_accuracy": float(ordered["val_accuracy"].max()) if "val_accuracy" in ordered else np.nan,
                "last_val_accuracy": float(ordered["val_accuracy"].iloc[-1]) if "val_accuracy" in ordered else np.nan,
                "last_train_loss": float(ordered["train_loss"].iloc[-1]) if "train_loss" in ordered else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _timing_summary(final: pd.DataFrame, rounds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if "round_time_s" in rounds:
        for model, group in rounds.groupby("model"):
            rows.append(
                {
                    "metric": "round_time_s",
                    "model": model,
                    "mean": float(group["round_time_s"].mean()),
                    "std": float(group["round_time_s"].std()),
                    "n": int(len(group)),
                }
            )
    if "total_model_time_s" in final:
        for model, group in final.groupby("model"):
            rows.append(
                {
                    "metric": "total_model_time_s",
                    "model": model,
                    "mean": float(group["total_model_time_s"].mean()),
                    "std": float(group["total_model_time_s"].std()),
                    "n": int(len(group)),
                }
            )
    return pd.DataFrame(rows)


def _statistical_tests(final: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "model", "source_run_id", "test_accuracy"}
    if not required.issubset(final.columns):
        return pd.DataFrame(columns=["comparison", "n_pairs", "mean_delta", "p_value", "note"])
    models = sorted(final["model"].dropna().unique())
    if len(models) < 2:
        return pd.DataFrame([{"comparison": "not_available", "n_pairs": 0, "mean_delta": np.nan, "p_value": np.nan, "note": "Need at least two models."}])
    baseline, proposed = models[0], models[-1]
    pivot = final.pivot_table(index=["source_run_id", "dataset"], columns="model", values="test_accuracy", aggfunc="mean").dropna()
    if baseline not in pivot or proposed not in pivot or len(pivot) < 2:
        return pd.DataFrame([{"comparison": f"{proposed} vs {baseline}", "n_pairs": len(pivot), "mean_delta": np.nan, "p_value": np.nan, "note": "Not enough paired rows."}])
    delta = pivot[proposed] - pivot[baseline]
    try:
        p_value = float(wilcoxon(pivot[proposed], pivot[baseline]).pvalue)
    except ValueError:
        p_value = np.nan
    return pd.DataFrame(
        [
            {
                "comparison": f"{proposed} vs {baseline}",
                "n_pairs": int(len(delta)),
                "mean_delta": float(delta.mean()),
                "p_value": p_value,
                "note": "Paired Wilcoxon over source_run_id,dataset test_accuracy rows.",
            }
        ]
    )


def _plot_validation_accuracy(fig_dir: Path, rounds: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))
    grouped = rounds.groupby(["display", "model", "round"], as_index=False)["val_accuracy"].mean()
    for (display, model), group in grouped.groupby(["display", "model"]):
        ax.plot(group["round"], group["val_accuracy"], marker="o", label=f"{display} {model}")
    ax.set_title("Real validation accuracy by round", fontweight="bold", pad=14)
    ax.set_xlabel("Round")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0, 1)
    ax.legend(ncol=2, frameon=True)
    _save(fig, fig_dir / "Fig01_real_validation_accuracy_by_round")


def _plot_test_accuracy(fig_dir: Path, final: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    pivot = final.pivot_table(index="display", columns="model", values="test_accuracy", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(11, 7))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Real official test accuracy", fontweight="bold", pad=14)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Model", frameon=True)
    _save(fig, fig_dir / "Fig02_real_test_accuracy_by_dataset")


def _plot_training_loss(fig_dir: Path, rounds: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    grouped = rounds.groupby(["display", "model", "round"], as_index=False)["train_loss"].mean()
    fig, ax = plt.subplots(figsize=(12, 7))
    for (display, model), group in grouped.groupby(["display", "model"]):
        ax.plot(group["round"], group["train_loss"], label=f"{display} {model}")
    ax.set_title("Real training loss by round", fontweight="bold", pad=14)
    ax.set_xlabel("Round")
    ax.set_ylabel("Training loss")
    ax.legend(ncol=2, frameon=True)
    _save(fig, fig_dir / "Fig03_real_training_loss_by_round")


def _plot_training_time(fig_dir: Path, final: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 7))
    if "total_model_time_s" in final:
        pivot = final.pivot_table(index="display", columns="model", values="total_model_time_s", aggfunc="mean")
        pivot.plot(kind="bar", ax=ax)
    ax.set_title("Real training time by dataset", fontweight="bold", pad=14)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Seconds")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Model", frameon=True)
    _save(fig, fig_dir / "Fig04_real_training_time_by_dataset")


def _plot_gpu_memory(fig_dir: Path, rounds: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))
    col = "gpu_peak_memory_allocated_bytes"
    if col in rounds:
        data = rounds.copy()
        data["gpu_peak_memory_gb"] = data[col] / 1024**3
        grouped = data.groupby(["model", "round"], as_index=False)["gpu_peak_memory_gb"].mean()
        for model, group in grouped.groupby("model"):
            ax.plot(group["round"], group["gpu_peak_memory_gb"], marker="o", label=model)
    ax.set_title("Real GPU memory by round", fontweight="bold", pad=14)
    ax.set_xlabel("Round")
    ax.set_ylabel("Peak allocated memory (GB)")
    ax.legend(title="Model", frameon=True)
    _save(fig, fig_dir / "Fig05_real_gpu_memory_by_round")


def _save(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.08)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exists_nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0
