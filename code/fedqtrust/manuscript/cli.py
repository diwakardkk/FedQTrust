"""Configure, check, execute, resume, and report the manuscript experiments."""
import argparse
import collections
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from .artifacts import export_results, save_json, status
from .plan import build_plan

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'action',
        choices=['plan', 'preflight', 'data', 'smoke', 'run', 'benchmarks', 'report', 'status', 'export']
    )
    parser.add_argument('--config', help='JSON settings; command-line options take priority')
    parser.add_argument('--output-dir', default='output/manuscript_v3')
    parser.add_argument('--data-root', default='data/raw')
    parser.add_argument('--experiments', default='E1,E2,E3,E4,E5,E6,E7,E8,E9')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--infrastructure', default='paper', choices=['paper', 'local', 'development'])
    parser.add_argument('--solver', default='qaoa', choices=['qaoa', 'sa'])
    parser.add_argument('--job-id', help='Run one job from plan.json')
    parser.add_argument('--download-data', action='store_true')
    parser.add_argument('--skip-benchmarks', action='store_true')
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--archive', help='Export ZIP destination outside output directory')
    parser.add_argument('--include-checkpoints', action='store_true')
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        settings = json.loads(Path(preliminary.config).read_text())
        allowed = {'output_dir', 'data_root', 'experiments', 'device', 'infrastructure', 'solver'}
        if set(settings) - allowed:
            parser.error('Unknown settings: ' + ', '.join(sorted(set(settings) - allowed)))
        parser.set_defaults(**settings)
    args = parser.parse_args(argv)
    if args.device not in (
        'cpu',
        'cuda'
    ) or args.infrastructure not in ('paper', 'local', 'development') or args.solver not in ('qaoa', 'sa'):
        parser.error('Invalid device, infrastructure, or solver in settings')
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        parser.error('Require shards >= 1 and 0 <= shard-index < shards')
    if args.job_id and args.shards != 1:
        parser.error('Choose either a job ID or sharding')
    return args

def planned_jobs(args):
    experiments = tuple((part.strip() for part in args.experiments.split(',')))
    if not experiments or not set(experiments) <= {f'E{i}' for i in range(1, 10)}:
        raise ValueError('Experiments must be a comma-separated subset of E1 through E9')
    jobs = build_plan(experiments)
    if args.solver == 'sa':
        jobs = list({job.id: job for job in (replace(j, solver='sa') for j in jobs)}.values())
    plan = dict(
        protocol_version=3,
        experiments=experiments,
        infrastructure=args.infrastructure,
        device=args.device,
        data_root=str(Path(args.data_root).resolve()),
        jobs=[dict(id=job.id, **asdict(job)) for job in jobs]
    )
    return (experiments, jobs, plan)

def persist_plan(output, plan):
    path = output / 'plan.json'
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(plan)):
        raise ValueError('Output already has a different plan. Choose a new output directory.')
    save_json(path, plan)

def record_source(output):
    code_root = Path(__file__).parents[1]
    hashes = {str(path.relative_to(code_root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(code_root.rglob('*.py'))}
    path = output / 'source_hashes.json'
    if path.exists() and json.loads(path.read_text()) != hashes:
        raise ValueError('Source changed since the run started. Choose a new output directory.')
    save_json(path, hashes)

def download_data(root):
    from fedqtrust.data.datasets import ensure_all_datasets
    ensure_all_datasets(root, True, str(Path(root).parent / 'metadata'))

def benchmarks(output, experiments, infrastructure, data_root, device):
    from .benchmarks import qubo_benchmarks, infrastructure_benchmarks
    if 'E2' in experiments:
        qubo_benchmarks(output / 'benchmarks/E2')
    if 'E7' in experiments and infrastructure in ('paper', 'local'):
        infrastructure_benchmarks(output / 'benchmarks/E7', data_root=data_root, device=device,
                                  infrastructure=infrastructure)

def smoke(args):
    """Two rounds on a small real-data subset, separate from all E1--E9 results."""
    import torch
    from torch.utils.data import Subset
    from .engine import prepare_data, run_job
    from .plan import Job
    if preflight(args.device, 'development', ()):
        return 1
    if args.download_data:
        download_data(args.data_root)
    output = Path(args.output_dir).parent / 'smoke'
    job = Job(
        experiment='SMOKE',
        datasets=('pneumoniamnist',),
        method='FedAvg',
        clients=3,
        k=2,
        rounds=2,
        epochs=1,
        batch=4,
        solver='sa'
    )
    clients, tests, roots, notes = prepare_data(job, args.data_root, output / 'partitions')
    for client in clients:
        for split in ('train', 'val'):
            client[split] = Subset(client[split], list(range(min(8, len(client[split])))))
        client['n'] = len(client['train'])
    tests = {name: Subset(ds, list(range(min(16, len(ds))))) for name, ds in tests.items()}
    torch.set_num_threads(min(4, torch.get_num_threads()))
    run_job(
        job,
        output / 'jobs' / job.id,
        device=args.device,
        infrastructure='development',
        data=(clients, tests, roots, notes),
        data_root=args.data_root
    )
    save_json(
        output / 'SMOKE_ONLY.json',
        {'purpose': 'Circuit/solver probes and tiny real-data FL execution; not paper results'}
    )
    print('SMOKE passed. Full-data quantum training and Fabric still require target-server validation.')
    return 0

def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir)
    if args.action == 'status':
        print(json.dumps(status(output), indent=2))
        return 0
    if args.action == 'export':
        print(export_results(output, args.archive, args.include_checkpoints))
        return 0
    if args.action == 'report':
        from .report import build_report
        build_report(output)
        return 0
    if args.action == 'data':
        download_data(args.data_root)
        return 0
    if args.action == 'smoke':
        return smoke(args)
    experiments, jobs, plan = planned_jobs(args)
    if args.action == 'preflight':
        return preflight(args.device, args.infrastructure, experiments)
    persist_plan(output, plan)
    if args.action == 'plan':
        print(json.dumps(
            {
                'jobs': len(jobs),
                'by_experiment': dict(collections.Counter((j.experiment for j in jobs))),
                'plan': str(output / 'plan.json')
            },
            indent=2
        ))
        return 0
    if preflight(args.device, args.infrastructure, experiments):
        return 1
    record_source(output)
    if args.download_data:
        download_data(args.data_root)
    if args.action == 'benchmarks':
        benchmarks(output, experiments, args.infrastructure, args.data_root, args.device)
        return 0
    if args.job_id and args.job_id not in {job.id for job in jobs}:
        raise ValueError('Job ID not found in plan')
    from .engine import run_job
    for index, job in enumerate(jobs):
        if args.job_id and job.id != args.job_id:
            continue
        if index % args.shards != args.shard_index:
            continue
        run_job(
            job,
            output / 'jobs' / job.id,
            device=args.device,
            infrastructure=args.infrastructure,
            data_root=args.data_root
        )
    if args.shards == 1 and (not args.job_id):
        if not args.skip_benchmarks:
            benchmarks(output, experiments, args.infrastructure, args.data_root, args.device)
        from .report import build_report
        build_report(output)
    return 0

def preflight(device, infrastructure, experiments):
    import torch
    errors = []
    for module in ['qiskit', 'qiskit_machine_learning', 'qiskit_optimization', 'qiskit_aer']:
        if importlib.util.find_spec(module) is None:
            errors.append('Missing ' + module)
    if device == 'cuda' and (not torch.cuda.is_available()):
        errors.append('CUDA requested but unavailable')
    if not errors:
        try:
            import numpy as np
            from fedqtrust.models.fedqcnn import QuantumPatchLayer
            from fedqtrust.quantum.qaoa import solve_qaoa
            from fedqtrust.quantum.qubo import QuboInstance
            layer = QuantumPatchLayer().to(device)
            features = layer(torch.rand(1, 1, 2, 2, device=device))
            classifier = torch.nn.Linear(4, 2).to(device)
            classifier(features).square().mean().backward()
            if device == 'cuda':
                torch.cuda.synchronize()
            if layer.quantum_weights.grad is None or not torch.isfinite(layer.quantum_weights.grad).all():
                raise ValueError('Missing/nonfinite QNN gradient')
            # The training backend must reproduce Qiskit EstimatorQNN on this device.
            reference = QuantumPatchLayer(backend='qiskit')
            with torch.no_grad():
                layer.quantum_weights.copy_(reference.quantum_weights.to(device))
            probe = torch.rand(2, 1, 4, 4)
            gap = (layer(probe.to(device)).cpu() - reference(probe)).abs().max().item()
            if gap > 1e-4:
                raise ValueError(f'QNN backend differs from Qiskit EstimatorQNN by {gap:.2e}')
            solve_qaoa(
                QuboInstance(np.array([0.9, 0.2]), np.zeros((2, 2)), 1, 2, 1, [0, 1]),
                reps=1,
                shots=64,
                maxiter=5
            )
        except Exception as exc:
            errors.append('Quantum execution probe failed: ' + str(exc))
    if 'E2' in experiments:
        try:
            from .benchmarks import noisy_sampler
            noisy_sampler(32, 42)
        except Exception as exc:
            errors.append('Nairobi noise model unavailable: ' + str(exc))
    if infrastructure in ('paper', 'local'):
        try:
            from fedqtrust.security.transport import Transport
            t = Transport('pqc')
            t.roundtrip(b'preflight', b'preflight')
            t.authenticate(b'preflight')
        except Exception as exc:
            errors.append('ML-KEM/ML-DSA execution unavailable: ' + str(exc))
    if infrastructure == 'paper':
        try:
            from fedqtrust.blockchain.fabric_backend import HyperledgerFabricBlockchain
            chain = HyperledgerFabricBlockchain()
            if not chain.ping():
                errors.append('Fabric Ping failed')
            if chain._call('ProtocolVersion').stdout.strip() != 'fedqtrust-manuscript-v2':
                errors.append('Fabric contract protocol version mismatch')
        except Exception as exc:
            errors.append('Fabric unavailable: ' + str(exc))
    print(json.dumps(
        dict(device=device, infrastructure=infrastructure, ready=not errors, errors=errors,
             ledger_backend='fabric' if infrastructure == 'paper' else 'local_memory',
             fabric_required=infrastructure == 'paper'),
        indent=2
    ))
    return int(bool(errors))
if __name__ == '__main__':
    sys.exit(main())
