"""E2 optimizer/noise trials and E7 measured infrastructure operations."""
import time
import numpy as np
from pathlib import Path
from .engine import write_csv, save_json
from fedqtrust.quantum.qubo import QuboInstance, pairwise_cosine_distances
from fedqtrust.quantum.exact_solver import solve_exact
from fedqtrust.quantum.simulated_annealing import solve_sa
from fedqtrust.quantum.qaoa import solve_qaoa

def noisy_sampler(shots, seed):
    # Never substitute an unrelated fake backend if Nairobi is unavailable.
    from qiskit_ibm_runtime.fake_provider import FakeNairobiV2
    from qiskit_aer.noise import NoiseModel, QuantumError, ReadoutError
    from qiskit_aer.primitives import SamplerV2
    backend = FakeNairobiV2()
    calibrated = NoiseModel.from_backend(backend)
    # do not silently leave variables above qubit six noiseless.
    # Pick the first calibration for each operation and apply it homogeneously;
    # Explicit all-register extrapolation: Nairobi has only seven physical qubits.
    noise = NoiseModel(basis_gates=calibrated.basis_gates)
    seen = set()
    for error in calibrated.to_dict()['errors']:
        if error['type'] == 'qerror':
            for op in error['operations']:
                if op not in seen:
                    noise.add_all_qubit_quantum_error(QuantumError.from_dict(error), op)
                    seen.add(op)
        elif error['type'] == 'roerror' and 'readout' not in seen:
            noise.add_all_qubit_readout_error(ReadoutError(error['probabilities']))
            seen.add('readout')
    sampler = SamplerV2(
        default_shots=shots,
        seed=seed,
        options={'backend_options': {'method': 'statevector', 'noise_model': noise}}
    )
    sampler.fedqtrust_basis_gates = noise.basis_gates
    return sampler

def _instance(n, trial):
    rng = np.random.default_rng(42 + trial + n * 100)
    # Random fixed QUBO benchmark instances, explicitly distinct from FL gradients.
    trust = rng.uniform(0.4, 1, n)
    dist = pairwise_cosine_distances(rng.normal(size=(n, 32)))
    return QuboInstance(trust, dist, 1.0, 2.0, min(5, n - 1), list(range(n)))


def _ratio(value, exact):
    return value / exact.objective if exact is not None and abs(exact.objective) > 1e-12 else None


def qubo_benchmarks(out, quick=False):
    """Task E2 items 2-4 exactly as specified:
    (2) approximation ratio: 5 fixed instances, |E| in {6,8,10,12}, p in {1,2,3,4};
    (3) solve time: |E| in {5,8,10,12,15}, 10 trials, QAOA (p=3) vs SA;
    (4) noise: fake Nairobi noisy QAOA vs noiseless on the item-2 instances (p=3)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    instances = []
    sizes_ratio = (4,) if quick else (6, 8, 10, 12)
    depths = (1,) if quick else (1, 2, 3, 4)
    for n in sizes_ratio:
        for trial in range(1 if quick else 5):
            instance = _instance(n, trial)
            instances.append(dict(purpose='approximation_ratio', n=n, trial=trial, trust=instance.trust.tolist(),
                                  distances=instance.distances.tolist(), k=instance.k))
            exact = solve_exact(instance)
            for p in depths:
                sol = solve_qaoa(instance, reps=p, maxiter=8 if quick else 100, shots=128 if quick else 1024, seed=42 + trial)
                rows.append(dict(
                    item='approximation_ratio', n=n, trial=trial, solver='QAOA', p=p, noise=False,
                    runtime_s=sol.runtime_s, objective=sol.objective, feasible=sol.feasible_cardinality,
                    expected_objective=sol.expected_objective, best_sampled_objective=sol.best_sampled_objective,
                    exact_objective=exact.objective, objective_gap=sol.objective - exact.objective,
                    objective_ratio=_ratio(sol.objective, exact),
                    expected_ratio=_ratio(sol.expected_objective, exact)))
            if not quick:
                sol = solve_qaoa(instance, reps=3, seed=42 + trial, sampler=noisy_sampler(1024, 42 + trial))
                rows.append(dict(
                    item='noise', n=n, trial=trial, solver='QAOA', p=3, noise=True,
                    runtime_s=sol.runtime_s, objective=sol.objective, feasible=sol.feasible_cardinality,
                    expected_objective=sol.expected_objective, best_sampled_objective=sol.best_sampled_objective,
                    exact_objective=exact.objective, objective_gap=sol.objective - exact.objective,
                    objective_ratio=_ratio(sol.objective, exact),
                    expected_ratio=_ratio(sol.expected_objective, exact)))
            write_csv(out / 'optimizer_trials.csv', rows)
    for n in (4,) if quick else (5, 8, 10, 12, 15):
        for trial in range(1 if quick else 10):
            instance = _instance(n, 1000 + trial)
            exact = solve_exact(instance) if n <= 12 else None
            sol = solve_qaoa(instance, reps=1 if quick else 3, maxiter=8 if quick else 100,
                             shots=128 if quick else 1024, seed=42 + trial)
            sa = solve_sa(instance, seed=42 + trial)
            for name, res in (('QAOA', sol), ('SA', sa)):
                rows.append(dict(
                    item='timing', n=n, trial=trial, solver=name, p=(1 if quick else 3) if name == 'QAOA' else 0,
                    noise=False, runtime_s=res.runtime_s, objective=res.objective, feasible=res.feasible_cardinality,
                    exact_objective=exact.objective if exact else None,
                    objective_gap=res.objective - exact.objective if exact else None))
            write_csv(out / 'optimizer_trials.csv', rows)
    save_json(out / 'instances.json', instances)
    save_json(out / 'benchmark_status.json', dict(complete=True, quick=quick))

def infrastructure_benchmarks(out, data_root='data/raw', device='cpu', infrastructure='paper'):
    import oqs
    from fedqtrust.blockchain.fabric_backend import HyperledgerFabricBlockchain
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    def measure(component, fn, trial):
        t = time.perf_counter()
        result = fn()
        rows.append(dict(component=component, trial=trial, seconds=time.perf_counter() - t))
        return result
    for trial in range(1000):
        with oqs.KeyEncapsulation('ML-KEM-768') as recipient, oqs.KeyEncapsulation('ML-KEM-768') as sender, oqs.Signature('ML-DSA-65') as signer, oqs.Signature('ML-DSA-65') as verifier:
            pub = measure('ML-KEM-768_keygen', recipient.generate_keypair, trial)
            ciphertext, secret = measure('ML-KEM-768_encapsulation', lambda: sender.encap_secret(pub), trial)
            recovered = measure('ML-KEM-768_decapsulation', lambda: recipient.decap_secret(ciphertext), trial)
            if recovered != secret:
                raise RuntimeError('KEM benchmark mismatch')
            key = measure('ML-DSA-65_keygen', signer.generate_keypair, trial)
            payload = b'FedQTrust component benchmark'
            signature = measure('ML-DSA-65_sign', lambda: signer.sign(payload), trial)
            if not measure('ML-DSA-65_verify', lambda: verifier.verify(payload, signature, key), trial):
                raise RuntimeError('Signature benchmark mismatch')
    if infrastructure == 'paper':
        chain = HyperledgerFabricBlockchain(namespace='E7_benchmark')
        ledger_operation = 'Fabric_commit'
    else:
        from fedqtrust.blockchain.memory_backend import InMemoryBlockchain
        chain = InMemoryBlockchain()
        ledger_operation = 'Local_memory_ledger_write'
    for trial in range(50):
        measure(ledger_operation, lambda: chain.log_selection({'trial': trial}), trial)
    # 20 trials at each requested QAOA size/depth.
    for n in (5, 8, 10, 12):
        for p in (1, 2, 3):
            for trial in range(20):
                rng = np.random.default_rng(100 * n + trial)
                instance = QuboInstance(
                    rng.random(n),
                    pairwise_cosine_distances(rng.normal(size=(n, 32))),
                    1.0,
                    2.0,
                    min(n, 5),
                    list(range(n))
                )
                measure(f'QAOA_n{n}_p{p}', lambda: solve_qaoa(instance, reps=p, seed=42 + trial), trial)
    write_csv(out / 'component_trials.csv', rows)
    summary = []
    for name in sorted({r['component'] for r in rows}):
        vals = np.array([r['seconds'] for r in rows if r['component'] == name])
        summary.append(dict(
            component=name,
            n=len(vals),
            mean_s=float(vals.mean()),
            std_s=float(vals.std(ddof=1)),
            p50_s=float(np.quantile(vals, 0.5)),
            p99_s=float(np.quantile(vals, 0.99))
        ))
    write_csv(out / 'component_summary.csv', summary)
    local_training_benchmarks(out, device=device, data_root=data_root)
    save_json(out / 'benchmark_status.json', dict(
        complete=True, quick=False, infrastructure=infrastructure,
        fabric_network_measured=infrastructure == 'paper',
        unmeasured=[] if infrastructure == 'paper' else ['Fabric commit latency', 'Network consensus', 'On-chain verification']
    ))

def local_training_benchmarks(out, device='cpu', data_root='data/raw'):
    from .plan import Job, DATASETS
    from .engine import prepare_data, JointModel, train_local, clone_state
    from fedqtrust.reproducibility import seed_everything
    import torch
    rows = []
    for dataset in DATASETS:
        job = Job(experiment='E7', datasets=(dataset,), clients=8, epochs=3, batch=32)
        clients, _, _, _ = prepare_data(job, data_root, Path(out) / 'partitions')
        for trial in range(20):
            seed_everything(42 + trial)
            model = JointModel((dataset,), True).to(device)
            state = clone_state(model.state_dict())
            start = time.perf_counter()
            _, loss, _ = train_local(model, state, clients[0], job, 1, 0, device)
            rows.append(dict(
                dataset=dataset,
                trial=trial,
                local_epochs=3,
                batch_size=32,
                train_samples=clients[0]['n'],
                seconds=time.perf_counter() - start,
                loss=loss,
                device=device
            ))
            write_csv(Path(out) / 'local_training_trials.csv', rows)
