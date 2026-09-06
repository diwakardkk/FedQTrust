# FedQTrust

FedQTrust is a research-code repository for trustworthy, quantum-resilient federated learning on multi-modal MedMNIST medical imaging. It includes deterministic data setup, CNN/FedQCNN model paths, trust scoring, QUBO client selection, Byzantine attack utilities, crypto wrappers, blockchain abstractions, smoke tests, and experiment/reporting scaffolds.

This public repository intentionally does **not** include `main-v2.tex`, downloaded datasets, checkpoints, or generated results.

## Clone

```bash
git clone https://github.com/diwakardkk/FedQTrust.git
cd FedQTrust
```

## What Git Tracks

Tracked:

```text
src/fedqtrust/
fedqtrust/
configs/
scripts/
tests/
blockchain/
slurm/
requirements*.txt
pyproject.toml
environment.yml
README.md
Makefile
```

Ignored:

```text
main-v2.tex
data/raw/*
data/partitions/*
data/metadata/*
output/*
*.pt
*.pth
.env
.venv/
```

## GPU Server Requirements

Recommended for full runs:

- Linux server with NVIDIA GPU
- recent NVIDIA driver
- CUDA-compatible PyTorch build
- Python 3.10-3.12 recommended for production
- multi-core CPU and sufficient RAM
- optional: Qiskit Aer GPU build
- optional for paper-strict mode: Docker, Hyperledger Fabric, liboqs

Check the server:

```bash
nvidia-smi
python3 --version
```

## GPU Installation

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install a CUDA PyTorch wheel matching your server. Example for CUDA 12.1:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Example for CUDA 12.4:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Install FedQTrust dependencies:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Optional Qiskit Aer GPU:

```bash
python -m pip uninstall -y qiskit-aer
python -m pip install qiskit-aer-gpu
```

If `qiskit-aer-gpu` is not compatible with the machine, keep normal `qiskit-aer`; FedQTrust reports the fallback instead of pretending GPU quantum simulation was used.

The helper setup script uses the safe default install and checks CUDA visibility:

```bash
bash scripts/setup_gpu.sh
```

It does not require Fabric or liboqs.

## Verify CUDA

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

Then run:

```bash
python -m fedqtrust preflight --profile smoke --device auto --output-dir output
```

For strict infrastructure checks:

```bash
python -m fedqtrust preflight --profile paper --device cuda --output-dir output
```

Paper mode requires real external services such as Hyperledger Fabric and liboqs. If they are missing, preflight fails clearly.

## Required Smoke Test

```bash
python -m fedqtrust smoke-test --download-data --device auto --output-dir output/smoke_test
```

Expected final line:

```text
[PASS] SMOKE TEST COMPLETE
```

Success file:

```text
output/smoke_test/SMOKE_TEST_PASSED.txt
```

The smoke test downloads or validates:

- PathMNIST
- OCTMNIST
- PneumoniaMNIST
- RetinaMNIST
- BreastMNIST

All archives stay in `data/raw/` and are ignored by git.

## One-Command GPU Run to Share Results

Ask the GPU-server user to run this after installation:

```bash
bash scripts/run_gpu_once.sh
```

This command:

- downloads or validates all five MedMNIST datasets
- uses `--device auto` and CUDA when available
- trains `classical_cnn` and `fedqcnn` once on each dataset
- evaluates official MedMNIST test splits
- records round-level metrics, final metrics, environment metadata, figures, tables, model checkpoints, and a `DONE` file
- creates a zip inside `output/gpu_once/` that can be sent back

The result bundle is clearly labeled as a GPU one-shot validation run. It is useful for checking that the full installed code runs on the server and for collecting first real metrics. It is not the full paper E1-E9 experiment grid.

Copy-paste instructions for the GPU user:

```bash
git clone https://github.com/diwakardkk/FedQTrust.git
cd FedQTrust
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Choose the CUDA wheel matching the server. CUDA 12.4 example:
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m pytest -q
python -m fedqtrust smoke-test --download-data --device auto --output-dir output/smoke_test
bash scripts/run_gpu_once.sh
```

Default run:

```bash
python -m fedqtrust run-gpu-once \
  --download-data \
  --device auto \
  --output-dir output/gpu_once \
  --rounds 3 \
  --batch-size 128 \
  --num-workers 4 \
  --models classical_cnn,fedqcnn \
  --datasets pathmnist,octmnist,pneumoniamnist,retinamnist,breastmnist \
  --require-cuda \
  --amp
```

Longer run:

```bash
ROUNDS=20 BATCH_SIZE=256 NUM_WORKERS=8 bash scripts/run_gpu_once.sh
```

Quick server check:

```bash
ROUNDS=1 DATASETS=pneumoniamnist MODELS=classical_cnn bash scripts/run_gpu_once.sh
```

After completion, send back:

```text
output/gpu_once/run_YYYYMMDD_HHMMSS.zip
```

## CPU Smoke Test

Use this on a CPU-only machine to verify the installation, datasets, model forward/backward paths, trust/QUBO logic, attacks, aggregation, plotting, and tables:

```bash
bash scripts/run_cpu_smoke.sh
```

Equivalent direct command:

```bash
python -m fedqtrust smoke-test --download-data --device cpu --output-dir output/smoke_test_cpu
```

The success marker is saved at:

```text
output/smoke_test_cpu/SMOKE_TEST_PASSED.txt
```

## Real Scientific Results Suite

Use this fast CPU command to test that the figure/table pipeline works with sample artifacts:

```bash
bash scripts/run_publication_test_sample.sh
```

It writes:

```text
output/publication_test_sample/
output/publication_test_sample.zip
```

For the GPU/server scientific result run, the command now trains implemented models first and then builds plots/tables by reading the saved real CSV metrics:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA=cu124 DEVICE=cuda MODE=paper ROUNDS=100 bash scripts/run_full_experiment_once.sh
```

For a CPU-only real training run:

```bash
PYTORCH_CUDA=cpu DEVICE=cpu MODE=paper ROUNDS=100 STRICT_INFRA=false bash scripts/run_full_experiment_once.sh
```

The command creates:

```text
output/scientific_results/
output/scientific_results.zip
output/scientific_results/raw_training/<run_id>/
```

Important: `publication-suite --mode paper` is disabled because it previously generated paper-looking placeholder metrics. Use the one-command runner above. It runs real training through `run-gpu-once`, then `real-results-suite` reads those saved metrics and creates the final plots, tables, statistics, summaries, checkpoints manifest, and zip bundle.

`MODE=test` remains only for fast pipeline verification and must not be used for manuscript claims.

## Known-Good 100-Round Training Script

The previously validated standalone training workflow is available from the repository root and saves a complete results bundle:

```bash
python run_training_final_correct.py --device cpu --rounds 100 --output-dir output/final_correct
```

For a quick CPU test:

```bash
python run_training_final_correct.py --device cpu --rounds 1 --max-samples 64 --datasets pneumoniamnist --output-dir output/final_correct_quick
```

The script saves `round_metrics.csv`, `final_metrics.csv`, `training_loss.png`, model checkpoints, metadata, a `DONE` file, and a zip bundle under `output/final_correct/`.

## Strict Publishable Artifact Audit

For a paper submission, do not use the one-shot GPU bundle as the final result set. A publishable FedQTrust result set must include completed E1-E9 runs, raw per-round/per-client metrics, all required paper figures and tables, statistics, environment metadata, and strict evidence for paper-only infrastructure.

## One Command for a GPU Collaborator

After cloning the repository, the collaborator can run the real GPU result workflow with one command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA=cu124 DEVICE=cuda MODE=paper ROUNDS=100 bash scripts/run_full_experiment_once.sh
```

Use `PYTORCH_CUDA=cu121` or another supported value if the server needs a different PyTorch CUDA wheel:

```bash
PYTORCH_CUDA=cu121 bash scripts/run_full_experiment_once.sh
```

For a CPU-only machine, use:

```bash
PYTORCH_CUDA=cpu DEVICE=cpu MODE=paper ROUNDS=100 STRICT_INFRA=false bash scripts/run_full_experiment_once.sh
```

This command creates `.venv`, installs dependencies, installs CPU or CUDA PyTorch based on `PYTORCH_CUDA`, checks `nvidia-smi` only for GPU runs, runs tests and smoke checks, trains the implemented models on real MedMNIST data, reads the saved raw metrics, generates figures/tables/statistics, audits the output, and creates:

```text
output/scientific_results/
output/scientific_results.zip
```

The collaborator should send back `output/scientific_results.zip`.

The generated `summaries/scientific_claims_supported.md` file is the paper-safety checklist. It lists exactly which claims are supported by the real output and which E1-E9 claims still require separate raw experiments.

## Results Suite With Readiness Audit

This script runs the implemented real training workflow, then builds the result bundle by reading the saved raw metrics. It does not write E1-E9 placeholder manifest entries.

```bash
bash scripts/run_publishable_suite.sh
```

To run only the strict real-results audit manually:

```bash
python -m fedqtrust audit-real-results --strict-infra --output-dir output/scientific_results
```

The audit writes:

```text
output/scientific_results/summaries/real_results_audit.md
```

The manual audit exits with code `0` only when real metric files, figures, tables, statistics, checkpoints, CUDA evidence, Fabric evidence, and liboqs evidence are present. It exits with code `1` and lists blockers otherwise.

## Commands

Download data:

```bash
python -m fedqtrust download-data --output-dir output
```

Prepare deterministic partitions:

```bash
python -m fedqtrust prepare-data --output-dir output
```

Run tests:

```bash
python -m pytest -q
```

Run one experiment manifest:

```bash
python -m fedqtrust run --experiment E1 --profile phase1 --device cuda --resume --output-dir output
```

Run the one-shot GPU validation:

```bash
python -m fedqtrust run-gpu-once --download-data --device auto --output-dir output/gpu_once --rounds 3 --amp
```

Audit publication readiness:

```bash
python -m fedqtrust audit-publication --strict-infra --output-dir output
```

Run the implemented results suite:

```bash
bash scripts/run_all_experiments.sh
```

Generate reports from existing raw outputs:

```bash
python -m fedqtrust generate-report --output-dir output
```

## Profiles

`smoke`: fast checks using real implementation paths, tiny data batches, in-memory blockchain backend, and clear skips for unavailable external services.

`phase1`: data pipeline, partitions, CNN/FedQCNN paths, trust, QUBO exact/SA, FedAvg/FedProx/Krum/FLTrust foundations, label flip and LIE paths.

`phase2`: expanded experiment and infrastructure paths for QAOA, PQC, Fabric, extra attacks, sensitivity, scalability, and overhead studies.

`paper`: strict mode. Real datasets, configured rounds/seeds, QAOA dependencies where specified, liboqs, and Hyperledger Fabric are required. No fake result generation.

## Outputs

All generated outputs go under `output/`:

```text
output/environment/
output/smoke_test/
output/experiments/E1/
output/experiments/E2/
output/paper_figures/
output/paper_tables/
output/statistics/
output/summaries/
```

Do not commit `output/`.

## Hyperledger Fabric

Paper-strict blockchain experiments require Fabric:

```bash
docker --version
peer version
```

Deploy chaincode named `fedqtrust` with:

- `Ping`
- `SetTrust`
- `GetTrust`
- `LogSelection`
- `GetSelection`

The local smoke test uses `InMemoryBlockchain` and reports Fabric as skipped when Docker/Fabric are unavailable.

## liboqs

For paper-strict PQC runs, install Open Quantum Safe and `liboqs-python` with:

- ML-KEM-768
- ML-DSA-65

Verify:

```bash
python - <<'PY'
import oqs
print("KEM count:", len(oqs.get_enabled_kem_mechanisms()))
print("SIG count:", len(oqs.get_enabled_sig_mechanisms()))
PY
```

## SLURM

Single experiment:

```bash
EXPERIMENT=E1 sbatch slurm/run_experiment.sbatch
```

Seed array:

```bash
EXPERIMENT=E1 sbatch slurm/run_seed_array.sbatch
```

## Upload to GitHub

This repository is intended for:

```text
https://github.com/diwakardkk/FedQTrust.git
```

From this folder:

```bash
git status
git add .
git status
git commit -m "Initial FedQTrust implementation scaffold"
git branch -M main
git remote add origin https://github.com/diwakardkk/FedQTrust.git
git push -u origin main
```

If the remote already exists:

```bash
git remote set-url origin https://github.com/diwakardkk/FedQTrust.git
git push -u origin main
```

Before pushing, confirm these are **not** staged:

```bash
git status --short
```

Do not push:

```text
main-v2.tex
data/raw/
output/
```

## Current Local Validation

Smoke test passed:

```bash
python -m fedqtrust smoke-test --download-data --device auto --output-dir output/smoke_test
```

Unit tests passed:

```text
13 passed
```
