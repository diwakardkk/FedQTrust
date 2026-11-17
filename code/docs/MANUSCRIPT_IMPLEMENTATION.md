# Current compact server package

Use `FedQTrust_GPU_server.zip` and the repository README for the current
settings-file workflow, detached multi-GPU execution and round checkpoints. The
commands below also remain available in the full source checkout.

# Manuscript implementation and experiment execution

This revision replaces the trigonometric quantum surrogate with a real Qiskit
four-qubit QNN and replaces the exact-solver QAOA wrapper with actual QAOA.
The manuscript E1–E9 runner is `python -m fedqtrust manuscript`.
**Old scientific-results bundles remain historical results of the old code. They
must not be relabeled as results of this implementation.**

## Server commands

From the project directory on the GPU server:

```bash
source .venv/bin/activate
python -m pip install -r requirements-manuscript.txt
python -m pytest -q
python -m fedqtrust manuscript plan --device cuda --output-dir output/manuscript_v2
python -m fedqtrust manuscript preflight --device cuda
```

Preflight must pass before paper execution. It checks CUDA when requested,
Nairobi noise-model availability, actual ML-KEM/ML-DSA operations, and a deployed
Fabric chaincode Ping. `liboqs-python` needs a working native liboqs installation.
Fabric needs its normal peer MSP/TLS environment and a deployed `fedqtrust`
chaincode. See `blockchain/README.md` for the new contract interface.

Run only after preflight succeeds:

```bash
mkdir -p output/logs
nohup bash scripts/run_manuscript.sh > output/logs/manuscript_v2.log 2>&1 &
echo $!
```

This is a substantial experiment grid with true simulated quantum circuits.
A CUDA classifier does **not** make Qiskit statevector/parameter-shift simulation
run on the GPU. Benchmark a single job before scheduling the full grid.

The plan enumerates 1,071 training conditions and separate E2/E7 benchmark
trials. It includes additional attack-strength sweeps beyond the primary
comparisons. Jobs can be submitted individually from `plan.json`:

```bash
python -m fedqtrust manuscript run --device cuda --output-dir output/manuscript_v2 --job-id E1_REPLACE_WITH_ID_FROM_PLAN
```

Use one output directory/plan, and never schedule the same job twice concurrently.
A GPU scheduler should restrict each process to its assigned GPU. No implicit
multi-GPU training is claimed. Completed jobs are skipped; interrupted jobs resume after the last completed round,
restoring the model, trust history, cached updates and random states. An unfinished
round is repeated. Component benchmarks restart if interrupted.
Array jobs do not concurrently regenerate the shared report. After jobs finish,
run the serial runner (completed jobs are skipped) to execute missing benchmarks,
then generate the report:

```bash
python -m fedqtrust manuscript report --output-dir output/manuscript_v2
```

For a development-only experiment without Fabric or liboqs:

```bash
python -m fedqtrust manuscript run --experiments E1 --infrastructure development --solver sa --device cpu --output-dir output/development_e1
```

This is still a large training grid. Development runs are labeled as such and
cannot substantiate full-infrastructure or QAOA claims. Use `plan` and `--job-id`
to select a single condition. No smoke dataset is substituted in these commands.

## Implemented protocol

- Four-qubit ZZFeatureMap (one repetition), RealAmplitudes (two repetitions,
  linear entanglement), four Z observables, EstimatorQNN/TorchConnector and
  parameter-shift input/weight gradients. Twelve circuit weights are measured;
  parameter counts are not forced to a target. Patches are 2×2, stride two.
- Dense classifier: 784 → 512 → 256 → dataset-specific classes. One shared
  backbone and modality-specific heads within each federated condition.
- E1 uses 40 clients over five modalities, five selected clients, three local
  epochs, batch 32, learning rate 0.01/sqrt(round), and five seeds.
- Fixed Dirichlet(0.5) partitions use partition seed 42 for all model seeds and
  methods. A 1% root set is withheld from **every** method to match the FLTrust
  comparison; 20% of each remaining client partition is validation.
- Local validation accuracy, sliding update consistency and reputation drive
  trust. Defenses never receive malicious identities. Only the attack generator
  and evaluator use simulator ground truth.
- Selection precedes current local training and uses cached previous updates.
  QUBO inputs, raw solver objective, cardinality projection and actual solver
  choice are logged. Above 12 variables, QAOA uses Aer matrix-product-state
  sampling. K=80 may use the explicitly documented SA fallback above 12 eligible
  clients. Large dense QAOA simulations can still be extremely expensive.
- Aggregation combines shared parameters across selected clients, but each head
  uses only its matching modality. Unrepresented heads remain unchanged.
- Attacks begin at round 10; combined label-flipping and free-riding sets are
  disjoint. LIE uses a normal-quantile bound and a search against actual Krum
  selection, capped by the configured z sweep. Actual z and acceptance are saved.
- Final-round test metrics match the saved final-round checkpoint. No best-test
  checkpoint selection is used. The standalone diagnostic runner now preserves
  Adam state across epochs and also saves its final model.
- ML-KEM establishes a key derived with HKDF for AES-GCM; signatures cover
  context-bound update digests. Per-client and server signing keys are distinct.
  Transported/verified updates are deserialized before aggregation.
- Fabric records selection inputs, verifies registered ML-DSA or ECDSA proofs,
  and rejects trust updates without a verified, unused client proof. Run owner
  identity and signer keys are pinned for each execution attempt. No-PQC uses
  RSA-2048 + AES-GCM and ECDSA; No-Blockchain uses local memory only.
- Clients are simulated sequentially in one process. No distributed network
  throughput or parallel-client speedup is inferred.

## E1–E9 coverage

| Experiment | Implementation |
|---|---|
| E1 | All eight methods, five seeds, seven attack types plus strength sweeps; joint multi-modal FL |
| E2 | Actual QAOA, SA and selection controls; fixed random QUBOs; exact optima for small instances; Nairobi-derived noisy trials and noisy FL condition |
| E3 | Dedicated 200-round jobs, Byzantine fractions and penalty sweep; actual squared model-update norm divided by learning-rate squared; inverse-square-root fit |
| E4 | Separate dataset/method/attack/seed runs; FedQTrust versus PQS-BFL gain and dataset-size correlation |
| E5 | Explicit component-removal conditions, both primary datasets, two attacks and five seeds; no baseline rows relabeled as ablations |
| E6 | Trust weights with gamma fixed at 0.3; threshold, penalties, QAOA depth, reputation decay, consistency window and two 2D grids |
| E7 | Measured FL component times; 1,000 trials for each KEM/signature operation; 50 confirmed Fabric commits; QAOA size/depth timing trials |
| E8 | K=20/40/80, benign/poisoned, three seeds |
| E9 | k=3/5/10, FedQTrust/Krum, LIE, three seeds |

Outputs include raw final, round, client, selection and timing observations,
seed-level means/std, paired statistics, coverage, figures and a LaTeX report.
Missing jobs are listed, not synthesized. A completed job marker is not a
publication-readiness certificate. The aggregate audit deliberately leaves
publication readiness false while the qualifications below remain unresolved.

## Scientific corrections and remaining validation

The explicit model and experiment tables take priority over inconsistent target
numbers. These corrections must be reviewed before making manuscript claims:

1. The specified dense architecture has approximately half a million parameters,
   not 8K. ZZFeatureMap is feature/angle encoding, not amplitude encoding.
2. The seed count is five as requested. An exact two-sided signed-rank test with
   five nonzero pairs cannot attain p<0.05 (its minimum is 0.0625), much less
   0.05/42. More independent seeds or a justified revised analysis is needed.
3. Sparse client classes retain the global class labels and head dimensions;
   silently reducing a client's class count would invalidate cross-client heads.
   A deterministic minimum-size repair moves existing samples from the largest
   client partitions so that every client has train and validation examples.
   This deviation and the moves are logged; no examples are fabricated.
4. The paper's Krum f=floor(0.3K) is incompatible with k=5. The implementation
   records an effective f bounded by the number of available updates. This is
   a documented protocol correction, **not** the original guarantee; k=3 cannot
   support Krum with one Byzantine participant.
5. The manuscript's LIE “Krum norm threshold” is not an actual Krum rule. The
   implementation searches the real Krum selection rule and records the outcome.
6. ML-KEM/AES-GCM protects transport; the server still sees individual decrypted
   updates. This is not secure aggregation that hides updates from the server.
7. Fabric chaincode and adapters are provided, but deployment, endorsement/BFT
   assumptions, ML-DSA interoperability, and end-to-end server execution need
   verification on the target Fabric/liboqs environment. The contract permits the
   authenticated run owner to register simulation client keys; it is not a
   decentralized hospital identity enrollment system.
8. PQS-BFL and SecEdge-MC are documented matched-model protocol adaptations,
   not independently verified reproductions of the cited external implementations.
   CITADEL is omitted because it is absent from the explicit baseline table.
9. Single-dataset experiments use K clients on that dataset; joint E1 uses K/5
   clients per modality. The separate-dataset and joint protocols are not pooled.
10. No accuracy threshold is a correctness test for an attack. Attack transformations
    and activation are checked directly. Expected accuracy/overhead/ranking values
    are hypotheses, never substituted into results.
11. Full distributed deployment and the complete training grid have **not** run
    locally. Unit/integration tests cover circuits, solver execution, aggregation,
    a small FL loop, reporting, resume checks, and contract logic with a test ledger.
12. FedProx selects mu from 0.001/0.01/0.1 using a matched-initialization local
    validation warm-up on the first k clients; this explicit warm-up protocol is
    logged. E7 includes 20 three-epoch local-training trials per modality, plus
    per-client timing during FL.

## API references used

- Qiskit QNN/TorchConnector: https://qiskit-community.github.io/qiskit-machine-learning/tutorials/05_torch_connector.html
- QAOA optimizer: https://qiskit-community.github.io/qiskit-optimization/tutorials/03_minimum_eigen_optimizer.html
- liboqs Python KEM: https://github.com/open-quantum-safe/liboqs-python/blob/main/examples/kem.py
- On-chain verifier library API: https://github.com/paulmillr/noble-post-quantum

13. Nairobi has seven physical qubits while QUBOs can have more than seven
    variables. The noise experiment explicitly extrapolates the first calibrated
    error for each gate type and readout homogeneously to the simulated register.
    It is labeled **Nairobi-derived noise**, not a topology-faithful execution on
    the seven-qubit device. This modeling choice needs review before publication.
