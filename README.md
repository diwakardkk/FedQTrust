# FedQTrust

Trust-aware federated learning for multi-modal medical imaging with a hybrid quantum–classical
classifier (FedQCNN), QUBO/QAOA Byzantine-robust client selection, a trust ledger and NIST
post-quantum cryptography (ML-KEM-768, ML-DSA-65).

This repository contains the exact code that produced the results and the aggregated results of the
complete E1–E9 run (1,071 / 1,071 jobs).

```
FedQTrust/
├── code/                    Code used for the final run
│   ├── fedqtrust/           Package: models, FL engine, trust, QUBO/QAOA, attacks, crypto, report
│   ├── tests/               Unit tests (pytest)
│   ├── blockchain/          Hyperledger Fabric chaincode (not deployed in this run)
│   ├── docs/                Implementation notes and validation
│   ├── run.py               CLI: plan | preflight | data | run | benchmarks | report | status | export
│   ├── server.py            Detached supervisor, one worker per GPU-list entry (repeats allowed)
│   ├── run_complete.sh      Preflight + start
│   ├── fedq_launch.sh       Watchdog (cron every 10 min + @reboot; resumes from checkpoints)
│   ├── gpus.conf            Worker → GPU map used (12 workers on 8 GPUs)
│   ├── settings.json        Configuration of the final run
│   ├── setup_server.sh      Environment setup (Python ≥ 3.11, PyTorch cu126, Qiskit, liboqs)
│   └── requirements*.txt
└── results/
    ├── figures/             Fig01–Fig13 (+ Fig07b), PNG 300 dpi
    ├── paper_tables/        Result tables (CSV)
    ├── tables/              Aggregated tables: final_results, results_mean_std, statistics, detection, …
    └── audit.json           Run audit and documented limitations
```

## Datasets

Five 2D datasets from [MedMNIST v2](https://medmnist.com/) at 28×28 resolution. Each dataset is a different imaging modality.

| Dataset | Modality | Task | Classes | Train | Val | Test |
|---|---|---|---|---|---|---|
| PathMNIST | Colon pathology (H&E) | Multi-class | 9 | 89,996 | 10,004 | 7,180 |
| OCTMNIST | Retinal OCT | Multi-class | 4 | 97,477 | 10,832 | 1,000 |
| PneumoniaMNIST | Chest X-ray | Binary | 2 | 4,708 | 524 | 624 |
| RetinaMNIST | Fundus camera | Ordinal regression (treated as 5 classes) | 5 | 1,080 | 120 | 400 |
| BreastMNIST | Breast ultrasound | Binary | 2 | 546 | 78 | 156 |

**Data use in this work**
- **Download:** `python run.py data` downloads the official `.npz` files (via the `medmnist` package) into `code/data/raw`. The datasets are not included in this repository.
- **Preprocessing:** RGB images (PathMNIST, RetinaMNIST) are converted to grayscale and scaled to [0, 1], so every modality uses a single input channel.
- **Server root set:** 1% of each official training split is held out as the server's clean root dataset, used by FLTrust.
- **Client partition:** the remaining 99% is split across the clients of that dataset with a Dirichlet(α = 0.5) label partition (partition seed 42). The default K = 40 gives 8 clients per dataset. E8 uses K = 20 and K = 80, which gives 4 and 16 clients per dataset.
- **Client split:** each client splits its data 80/20 into local training and validation. The validation part is used for the trust score's accuracy term.
- **Evaluation:** test accuracy, weighted F1 and AUC are measured on the **official test split** of each dataset. The official validation split is not used.
- **Macro-average:** the E1 macro-average is the unweighted mean over the five datasets.

If you use these datasets, cite: J. Yang et al., "MedMNIST v2 – A large-scale lightweight benchmark for 2D and 3D biomedical image classification," *Scientific Data* 10, 41 (2023).

## Setup

| Item | Value |
|---|---|
| Datasets | 5 MedMNIST v2 modalities (see above), 28×28 grayscale |
| Federation | K = 40 clients (8 per dataset), Dirichlet α = 0.5, k = 5 per round, 100 rounds, 3 local epochs |
| Model | Shared CNN backbone + 4-qubit ZZFeatureMap / RealAmplitudes (reps = 2) quantum layer, 5 heads |
| Selection | Trust filter (τ ≥ θ) → QUBO solved with QAOA (p = 3), simulated-annealing fallback |
| Crypto | ML-KEM-768 + ML-DSA-65 via liboqs 0.16.0 (real measurements) |
| Ledger | In-memory ledger (`infrastructure: local`); Hyperledger Fabric **not** deployed |
| Seeds | 42, 123, 456, 789, 999 (E8/E9: 3 seeds; E2/E3/E6: seed 42) |
| Hardware | 8 GPUs (6× RTX 3090, 2× V100S), 36 CPU cores; workload is CPU-bound (quantum simulation) |

## Key results

Test accuracy (%), mean ± std over 5 seeds. Full tables are in [results/paper_tables](results/paper_tables), and figures are in [results/figures](results/figures).

**E1, 20% attack ratio, macro-average over 5 datasets**

| Attack | FedAvg | FedProx | FedQCNN | Krum | FLTrust | FedQTrust |
|---|---|---|---|---|---|---|
| Benign | 53.75 ± 3.17 | 52.69 ± 4.28 | 53.00 ± 1.76 | 41.41 ± 5.84 | 51.00 ± 2.35 | 46.70 ± 2.53 |
| Label flip | 51.69 ± 3.99 | 48.99 ± 3.97 | 51.82 ± 2.91 | 37.34 ± 2.14 | 49.21 ± 2.06 | 44.84 ± 2.61 |
| Sign flip | 50.68 ± 3.44 | 49.37 ± 3.58 | 51.27 ± 4.14 | 42.98 ± 1.21 | 51.66 ± 4.17 | 45.06 ± 1.21 |
| LIE | 51.90 ± 3.50 | 48.57 ± 3.08 | 53.91 ± 1.60 | 40.67 ± 5.36 | 51.89 ± 1.43 | 49.61 ± 2.50 |
| Free-ride 40% | 51.47 ± 2.67 | 46.97 ± 3.26 | 50.72 ± 1.58 | 41.19 ± 3.07 | 47.20 ± 3.32 | 44.61 ± 3.61 |
| Combined 30% | 50.62 ± 2.76 | 47.57 ± 3.41 | 51.28 ± 2.13 | 40.96 ± 6.41 | 47.96 ± 3.85 | 45.00 ± 3.18 |

PQS-BFL and SecEdge-MC have no Byzantine defence, so their accuracy equals FedAvg.

**E2: selection rules under 20% LIE, PathMNIST (seed 42).** QUBO-QAOA 52.20, QUBO + noise 53.23, QUBO-SA 49.57, L2-norm 40.79, Random 36.74, FLTrust 31.67, Krum 18.01.

**E5: ablation, PneumoniaMNIST, 20% attacks**

| Config | Label flip | LIE |
|---|---|---|
| No-QUBO (L2-norm) | 52.28 ± 1.77 | 73.30 ± 6.99 |
| No-Trust | 76.41 ± 4.09 | 79.10 ± 4.61 |
| No-Blockchain | 72.76 ± 4.11 | 73.04 ± 8.00 |
| No-PQC | 72.76 ± 4.11 | 73.04 ± 8.00 |
| No-Quantum | 84.13 ± 2.86 | 81.54 ± 3.55 |
| FedQTrust | 71.70 ± 4.17 | 74.23 ± 5.65 |

**E7: overhead per round.**
- Quantum-layer training: 42.3 s (k = 5).
- QAOA selection: 4.1 s.
- ML-KEM-768: 13.7 ms.
- ML-DSA-65: 33.9 ms.
- Total round time is 1,184% above FedAvg.

### Summary of findings
- **Supported:**
  - The QUBO selection objective is the main robustness component. Under LIE on PathMNIST it beats Krum and FLTrust by more than 20 points.
  - Removing QUBO costs 19–21 points under label flipping.
  - SA reaches similar accuracy to QAOA, so the gain comes from the objective, not from the quantum solver.
  - Post-quantum cryptography overhead is negligible: under 0.3 ms per client and 0.17% of traffic.
- **Not supported:**
  - In the multi-modal federation, FedQTrust is 2–7 points below FedAvg.
  - The simulated 4-qubit layer does not improve accuracy over a classical CNN (E4, E5).
  - The QAOA approximation ratio does not improve with depth.
  - The a/√t + b convergence fit fails (R² < 0.06; clients use Adam).
  - The overhead target of under 30% is not met.

## Deviations from the specification

1. **QAOA size cap.** The trust filter typically leaves about 40 eligible clients, so QAOA runs on the 15 most-trusted (seeded tie-break). Larger sets use SA (for example, K = 80 in E8).
2. **Warm-up.** The first ⌈K/k⌉ rounds select clients round-robin, so every client has an update before QUBO selection starts. Attacks start at round 10.
3. **Shot-based simulation.** Above 12 qubits, Aer's statevector method is used with 1,024 shots.
4. **No Hyperledger Fabric.** Ledger latency and consensus are not measured.
5. **Other limitations.** The Krum f bound, CITADEL not run, about 0.5M classifier parameters, and Wilcoxon tests with 5 seeds are listed in [results/audit.json](results/audit.json).

595 non-trust baseline jobs finished before fixes 1–3 and were carried over unchanged. Their code path is not affected by those fixes.

Raw per-round logs (`client_metrics.csv`, `round_metrics.csv`, `trust_trace.csv`, `timing.csv`, 0.05–1.2 GB each) are not included because of GitHub's size limits. They can be regenerated with the run below.

## Reproduce

```bash
cd code
bash setup_server.sh                                  # creates .venv (liboqs required)
.venv/bin/python run.py data --config settings.json   # downloads MedMNIST into data/raw
.venv/bin/python -m pytest tests -q
bash run_complete.sh 0,1,2,3                          # or: bash fedq_launch.sh (uses gpus.conf)
.venv/bin/python server.py status
.venv/bin/python run.py report --config settings.json
```

Runs resume automatically from per-job checkpoints. A change to the code requires a new `output_dir`, because outputs are tied to a hash of the source code.
