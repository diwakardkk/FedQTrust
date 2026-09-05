"""Publication-readiness audit for FedQTrust outputs.

This module intentionally does not fabricate missing results. It checks whether
the expected paper artifacts exist and records concrete blockers.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


REQUIRED_EXPERIMENTS = [f"E{i}" for i in range(1, 10)]
REQUIRED_RUN_FILES = [
    "config.json",
    "environment.json",
    "metrics.csv",
    "round_metrics.csv",
    "client_metrics.csv",
    "trust_scores.csv",
    "timing.csv",
    "events.jsonl",
    "DONE",
]
REQUIRED_FIGURES = [
    "Fig01_FedQTrust_architecture",
    "Fig02_E1_data_poison_accuracy",
    "Fig03_E1_LIE_accuracy",
    "Fig04_trust_evolution",
    "Fig05_QAOA_quality",
    "Fig06_Byzantine_ROC",
    "Fig07_convergence",
    "Fig08_quantum_gain",
    "Fig09_ablation",
    "Fig10_sensitivity_heatmaps",
    "Fig11_overhead",
    "Fig12_scalability",
    "Fig13_clients_per_round",
]
REQUIRED_TABLES = [
    "dataset_summary",
    "client_distribution",
    "single_client_model_results",
    "attack_configuration",
    "baseline_configuration",
    "E1_main_results",
    "E1_per_dataset_results",
    "E1_detection_results",
    "E2_QAOA_quality",
    "E2_QAOA_runtime",
    "E2_detection",
    "E3_convergence_fit",
    "E3_rounds_to_80",
    "E4_quantum_gain",
    "E4_correlation",
    "E5_ablation",
    "E6_sensitivity",
    "E7_overhead",
    "E7_crypto",
    "E7_communication",
    "E8_scalability",
    "E9_clients_per_round",
    "statistical_significance",
]


@dataclass
class AuditIssue:
    severity: str
    item: str
    message: str


def _exists_nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _load_environment(output_dir: Path) -> dict:
    path = output_dir / "environment" / "environment_report.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def audit_publication_outputs(output_dir: str | Path, strict_infra: bool = True) -> tuple[bool, list[AuditIssue]]:
    out = Path(output_dir)
    issues: list[AuditIssue] = []
    env = _load_environment(out)

    if strict_infra:
        if not env.get("cuda_available"):
            issues.append(AuditIssue("BLOCKER", "environment", "CUDA was not available in the recorded environment."))
        if not env.get("liboqs_available"):
            issues.append(AuditIssue("BLOCKER", "environment", "liboqs was not available; paper PQC claims cannot be supported."))
        if not env.get("hyperledger_fabric_available"):
            issues.append(AuditIssue("BLOCKER", "environment", "Hyperledger Fabric was not available; paper blockchain claims cannot be supported."))

    manifest = out / "experiments" / "manifest.csv"
    if not _exists_nonempty(manifest):
        issues.append(AuditIssue("BLOCKER", "manifest", "Missing nonempty output/experiments/manifest.csv."))
    else:
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for exp in REQUIRED_EXPERIMENTS:
            exp_rows = [row for row in rows if row.get("experiment") == exp]
            if not exp_rows:
                issues.append(AuditIssue("BLOCKER", exp, f"No manifest rows found for {exp}."))
            if exp_rows and not all(row.get("status") == "DONE" for row in exp_rows):
                issues.append(AuditIssue("BLOCKER", exp, f"Not all manifest rows for {exp} are DONE."))

    for exp in REQUIRED_EXPERIMENTS:
        exp_dir = out / "experiments" / exp
        if not exp_dir.exists():
            issues.append(AuditIssue("BLOCKER", exp, f"Missing directory {exp_dir}."))
            continue
        done_files = list(exp_dir.rglob("DONE"))
        if not done_files:
            issues.append(AuditIssue("BLOCKER", exp, f"No completed run DONE files under {exp_dir}."))
        for required in REQUIRED_RUN_FILES:
            if not list(exp_dir.rglob(required)):
                issues.append(AuditIssue("BLOCKER", exp, f"No {required} found under {exp_dir}."))

    for stem in REQUIRED_FIGURES:
        pdf = out / "paper_figures" / f"{stem}.pdf"
        png = out / "paper_figures" / f"{stem}.png"
        if not _exists_nonempty(pdf):
            issues.append(AuditIssue("BLOCKER", stem, f"Missing nonempty figure PDF: {pdf}."))
        if not _exists_nonempty(png):
            issues.append(AuditIssue("BLOCKER", stem, f"Missing nonempty figure PNG: {png}."))

    for stem in REQUIRED_TABLES:
        for suffix in ("csv", "tex", "md"):
            path = out / "paper_tables" / f"{stem}.{suffix}"
            if not _exists_nonempty(path):
                issues.append(AuditIssue("BLOCKER", stem, f"Missing nonempty table file: {path}."))

    stats_dir = out / "statistics"
    if not stats_dir.exists() or not list(stats_dir.rglob("*.csv")):
        issues.append(AuditIssue("BLOCKER", "statistics", "Missing statistical CSV outputs."))
    if not _exists_nonempty(out / "summaries" / "manuscript_consistency_report.md"):
        issues.append(AuditIssue("BLOCKER", "manuscript consistency", "Missing manuscript consistency report."))

    publishable = not any(issue.severity == "BLOCKER" for issue in issues)
    return publishable, issues


def write_publication_audit(output_dir: str | Path, strict_infra: bool = True) -> Path:
    out = Path(output_dir)
    report_path = out / "summaries" / "publication_readiness_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    publishable, issues = audit_publication_outputs(out, strict_infra)
    lines = [
        "# Publication Readiness Audit",
        "",
        f"Status: {'PUBLISHABLE_ARTIFACT_SET_PRESENT' if publishable else 'NOT_PUBLISHABLE'}",
        f"Strict infrastructure checks: {strict_infra}",
        "",
    ]
    if issues:
        lines.append("## Issues")
        lines.append("")
        for issue in issues:
            lines.append(f"- **{issue.severity}** `{issue.item}`: {issue.message}")
    else:
        lines.append("No blocking artifact issues found.")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

