# Local validation — 8 October 2026

- Full test suite: **35 passed**.
- Actual four-qubit QNN expectations matched an independent statevector calculation;
  parameter-shift weight and input gradients were finite and nonzero.
- Actual QAOA executed and its reported objective matched the returned bitstring.
- Small federated runs tested FedAvg, FedProx warm-up, and trust/QUBO selection.
- Modality head aggregation, sparse partitions, resume configuration checks,
  reporting coverage and missing-job handling passed.
- Classical encryption/signatures and Fabric contract logic (test ledger) passed
  identity, invalid-signature and replay checks.
- Interrupted trust-based FL with Gaussian attacks resumed after round two and
  exactly matched uninterrupted final model weights, metrics and selection history.
- Detached supervisor tests covered two workers, successful export and preflight
  failure without false completion.
- Standalone upload package: **17 focused tests passed** from its own directory.
- Package dependency closure, archive integrity, file hashes and exact attachment
  identity passed. The ZIP contains 26 internal implementation modules.
- CPU development preflight passed actual circuit and QAOA probes.
- Python compilation, shell syntax and Git whitespace checks passed.
- One existing sample-suite curve-fit covariance warning remains; it concerns
  synthetic sample data, not a newly generated scientific result.

## Not executed locally

The complete E1–E9 grid, real Fabric transactions, cross-library ML-DSA chaincode
verification, ML-KEM transport and Nairobi-derived noise runs remain unvalidated
on the target server. The real-data smoke command was not run locally because
the datasets are not present. The new generated LaTeX template was inspected but
not compiled because a TeX engine is unavailable. Paper preflight reports missing liboqs, Fabric peer/deployment
and qiskit-ibm-runtime in the local environment. No new paper results are claimed.

## Test environment

Python: 3.14.7

- torch: 2.14.0
- qiskit: 2.5.2
- qiskit-machine-learning: 0.9.1
- qiskit-optimization: 0.7.0
- qiskit-aer: 0.17.2
- pytest: 9.1.1

## Local-ledger run update

The server GPU checks passed under CUDA 12.6 after correcting PCI enumeration.
The external Fabric dependency blocked startup. A new explicit `local` mode
retains application cryptography and all training conditions while using an
in-process ledger. Its integration test executes real classical authenticated
transport and fails immediately if it attempts Fabric access. Reports mark
Fabric network measurements unavailable. ML-KEM/ML-DSA remain checked by server
preflight; they have not been executed in the local test environment.

The run-fix archive was extracted independently; its hashes, complete 1,071-job
plan and status command passed. No full-data training results are claimed.
