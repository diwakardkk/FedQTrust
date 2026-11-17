"""Explicit E1--E9 job grid, including controls and documented interpretations."""
from dataclasses import dataclass, asdict, replace
import hashlib, json, itertools
DATASETS = ('pathmnist', 'octmnist', 'pneumoniamnist', 'retinamnist', 'breastmnist')
SEEDS = (42, 123, 456, 789, 999)
METHODS = ('FedAvg', 'FedProx', 'FedQCNN', 'Krum', 'FLTrust', 'PQS-BFL', 'SecEdge-MC', 'FedQTrust')

@dataclass(frozen=True)
class Job:
    experiment: str = 'E1'
    datasets: tuple = DATASETS
    method: str = 'FedQTrust'
    attack: str = 'benign'
    fraction: float = 0.0
    seed: int = 42
    rounds: int = 100
    clients: int = 40
    k: int = 5
    epochs: int = 3
    batch: int = 32
    lr: float = 0.01
    decay: float = 0.0001
    partition_seed: int = 42
    partition_alpha: float = 0.5
    attack_start: int = 10
    # Malicious label-flipping clients relabel all their samples, so the attack ratio r
    # (clients {1..floor(rK)}) equals the relabelled fraction of training samples (Table 3).
    label_fraction: float = 1.0
    sigma: float = 0.5
    lie_z: float = 1.0
    alpha: float = 0.4
    beta: float = 0.3
    gamma: float = 0.3
    theta: float = 0.4
    reputation_decay: float = 0.9
    window: int = 5
    lambda_: float = 1.0
    mu: float = 2.0
    p: int = 3
    solver: str = 'qaoa'
    shots: int = 1024
    maxiter: int = 100
    proximal_mu: float = 0.01
    noise: bool = False
    # Exact statevector simulation of the specified circuit (validated against EstimatorQNN).
    qnn_backend: str = 'torch_statevector'
    # SA is the 'emergency fallback if QAOA times out' (Sec. 3.4); every fallback is logged.
    qaoa_timeout_s: float = 120.0

    def __post_init__(self):
        if not self.datasets or not set(self.datasets) <= set(DATASETS):
            raise ValueError('Unknown datasets')
        if self.clients < self.k or self.k < 1 or self.clients % len(self.datasets):
            raise ValueError('Invalid client counts')
        if min(self.rounds, self.epochs, self.batch, self.p, self.shots, self.maxiter, self.window) < 1:
            raise ValueError('Iteration counts must be positive')
        if abs(self.alpha + self.beta + self.gamma - 1) > 1e-08 or min(self.alpha, self.beta, self.gamma) < 0:
            raise ValueError('Invalid trust weights')
        if not 0 <= self.fraction < 1 or not 0 <= self.label_fraction <= 1 or (not 0 <= self.theta <= 1):
            raise ValueError('Fractions must be valid probabilities')
        if self.solver not in ('qaoa', 'sa'):
            raise ValueError('Unknown solver')
        if self.qnn_backend not in ('torch_statevector', 'qiskit'):
            raise ValueError('Unknown QNN backend')

    @property
    def id(self):
        return self.experiment + '_' + hashlib.sha256(json.dumps(
            asdict(self),
            sort_keys=True
        ).encode()).hexdigest()[:16]

def build_plan(experiments=tuple((f'E{i}' for i in range(1, 10)))):
    jobs = []

    def add(exp, **kw):
        jobs.append(Job(experiment=exp, **kw))
    if 'E1' in experiments:
        conditions = [('benign', 0.0, {})]
        for attack, ratios in [
            ('label_flip', (0.1, 0.2, 0.3)),
            ('sign_flip', (0.1, 0.2, 0.3)),
            ('free_riding', (0.1, 0.2, 0.4))
        ]:
            for r in ratios:
                conditions.append((attack, r, {}))
        conditions += [('gaussian', 0.2, {'sigma': s}) for s in (0.1, 0.5, 1.0)]
        conditions += [('lie', 0.2, {'lie_z': z}) for z in (0.5, 1.0, 2.0)]
        conditions += [('combined', 0.3, {})]
        for method, seed, (attack, fraction, kw) in itertools.product(METHODS, SEEDS, conditions):
            add('E1', method=method, seed=seed, attack=attack, fraction=fraction, **kw)
    if 'E2' in experiments:
        for method, solver in [
            ('FedQTrust', 'qaoa'),
            ('FedQTrust', 'sa'),
            ('No-QUBO', 'sa'),
            ('Krum', 'sa'),
            ('FLTrust', 'sa'),
            ('Random-Selection', 'sa')
        ]:
            add('E2', datasets=('pathmnist',), method=method, solver=solver, attack='lie', fraction=0.2)
        add('E2', datasets=('pathmnist',), attack='lie', fraction=0.2, noise=True)
    if 'E3' in experiments:
        for frac in (0.0, 0.1, 0.2, 0.3):
            add(
                'E3',
                datasets=('pneumoniamnist',),
                rounds=200,
                attack='benign' if frac == 0 else 'lie',
                fraction=frac
            )
        for lam in (0.1, 5.0):
            add('E3', datasets=('pneumoniamnist',), rounds=200, attack='lie', fraction=0.2, lambda_=lam)
        for method, attack in itertools.product(('FedQCNN', 'FedProx', 'FedAvg'), ('benign', 'lie')):
            add(
                'E3',
                datasets=('pneumoniamnist',),
                method=method,
                rounds=200,
                attack=attack,
                fraction=0.0 if attack == 'benign' else 0.2
            )
    if 'E4' in experiments:
        for dataset, method, attack, seed in itertools.product(
            DATASETS,
            ('FedQTrust', 'PQS-BFL'),
            ('benign', 'label_flip', 'lie'),
            SEEDS
        ):
            add(
                'E4',
                datasets=(dataset,),
                method=method,
                attack=attack,
                fraction=0.0 if attack == 'benign' else 0.2,
                seed=seed
            )
    if 'E5' in experiments:
        for dataset, method, attack, seed in itertools.product(
            ('pneumoniamnist', 'pathmnist'),
            ('FedQTrust', 'No-QUBO', 'No-Trust', 'No-Blockchain', 'No-PQC', 'No-Quantum'),
            ('label_flip', 'lie'),
            SEEDS
        ):
            add('E5', datasets=(dataset,), method=method, attack=attack, fraction=0.2, seed=seed)
    if 'E6' in experiments:
        base = dict(datasets=('pneumoniamnist',), attack='label_flip', fraction=0.2)
        for alpha in (0.2, 0.3, 0.4, 0.5, 0.6):
            add('E6', **base, alpha=alpha, beta=0.7 - alpha, gamma=0.3)
        for key, values in [
            ('theta', (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)),
            ('lambda_', (0.1, 0.5, 1.0, 2.0, 5.0)),
            ('mu', (0.5, 1.0, 2.0, 5.0)),
            ('p', (1, 2, 3, 4)),
            ('reputation_decay', (0.7, 0.8, 0.9, 0.95)),
            ('window', (3, 5, 10, 20))
        ]:
            for value in values:
                add('E6', **base, **{key: value})
        for alpha, theta in itertools.product((0.2, 0.3, 0.4, 0.5, 0.6), (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)):
            add('E6', **base, alpha=alpha, beta=0.7 - alpha, gamma=0.3, theta=theta)
        for lam, mu in itertools.product((0.1, 0.5, 1.0, 2.0, 5.0), (0.5, 1.0, 2.0, 5.0)):
            add('E6', **base, lambda_=lam, mu=mu)
    if 'E7' in experiments:
        for method in ('FedQTrust', 'FedAvg'):
            add('E7', method=method, rounds=20)
    if 'E8' in experiments:
        for clients, attack, seed in itertools.product((20, 40, 80), ('benign', 'label_flip'), SEEDS[:3]):
            add(
                'E8',
                datasets=('pneumoniamnist',),
                clients=clients,
                attack=attack,
                fraction=0.0 if attack == 'benign' else 0.2,
                seed=seed
            )
    if 'E9' in experiments:
        for k, method, seed in itertools.product((3, 5, 10), ('FedQTrust', 'Krum'), SEEDS[:3]):
            add('E9', datasets=('pneumoniamnist',), k=k, method=method, attack='lie', fraction=0.2, seed=seed)
    # Identical default sensitivity conditions run only once, retaining exact identity.
    return list({job.id: job for job in jobs}.values())
