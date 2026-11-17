"""Deterministic cross-silo FL execution with true per-client observations."""
from __future__ import annotations
import copy, csv, hashlib, io, json, math, time, uuid
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from .plan import Job
from fedqtrust.config import DATASETS
from fedqtrust.models.fedqcnn import QuantumPatchLayer
from fedqtrust.data.datasets import load_medmnist_dataset, label_array
from fedqtrust.data.partition import dirichlet_partition
from fedqtrust.fl.trust import TrustManager
from fedqtrust.fl.state import clone_state, subtract_state, flatten_state
from fedqtrust.baselines.krum import krum_select
from fedqtrust.baselines.fltrust import fltrust_scores
from fedqtrust.quantum.qubo import QuboInstance, pairwise_cosine_distances, project_cardinality
from fedqtrust.quantum.qaoa import solve_qaoa
from fedqtrust.quantum.simulated_annealing import solve_sa
from fedqtrust.reproducibility import seed_everything
from fedqtrust.security.transport import Transport
from fedqtrust.blockchain.memory_backend import InMemoryBlockchain
from fedqtrust.blockchain.fabric_backend import HyperledgerFabricBlockchain
QUBO_MAX_ELIGIBLE = 15
TRUST_METHODS = {
    'FedQTrust',
    'No-QUBO',
    'No-Trust',
    'No-Blockchain',
    'No-PQC',
    'No-Quantum',
    'Random-Selection'
}
QUANTUM_METHODS = {
    'FedQTrust',
    'FedQCNN',
    'No-QUBO',
    'No-Trust',
    'No-Blockchain',
    'No-PQC',
    'Random-Selection'
}

class JointModel(nn.Module):

    def __init__(self, datasets, quantum=True, backend='torch_statevector'):
        super().__init__()
        self.features = QuantumPatchLayer(backend=backend) if quantum else nn.Sequential(nn.Conv2d(1, 4, 2, 2), nn.Flatten())
        self.backbone = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.heads = nn.ModuleDict({d: nn.Linear(256, DATASETS[d]['classes']) for d in datasets})

    def forward(self, x, dataset):
        return self.heads[dataset](self.backbone(self.features(x)))

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)

def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys((k for row in rows for k in row)))
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)

def tensorize(ds):
    """Whole MedMNIST split as tensors (same grayscale/[0,1] mapping as ToGrayTensor),
    so training on GPU is not throttled by per-image PIL transforms."""
    imgs = np.asarray(ds.imgs)
    if imgs.ndim == 4:
        imgs = imgs[..., :3] @ np.array([0.299, 0.587, 0.114])
    x = torch.from_numpy((imgs.astype('float32') / 255.0)[:, None])
    y = torch.from_numpy(np.asarray(ds.labels).reshape(len(imgs), -1)[:, :1].astype('int64'))
    return torch.utils.data.TensorDataset(x, y)

def prepare_data(job, root, partition_dir):
    """Common split seed across methods, attacks and model seeds; root held out globally."""
    if job.clients % len(job.datasets):
        raise ValueError('Client count must be divisible by modality count')
    groups = []
    tests = {}
    roots = {}
    notes = []
    for dataset in job.datasets:
        group = []
        raw = load_medmnist_dataset(dataset, 'train', root, False)
        labels = label_array(raw).reshape(-1)
        ds = tensorize(raw)
        test = tensorize(load_medmnist_dataset(dataset, 'test', root, False))
        rng = np.random.default_rng(job.partition_seed)
        perm = rng.permutation(len(ds))
        root_n = max(1, int(0.01 * len(ds)))
        root_idx = perm[:root_n]
        pool = perm[root_n:]
        splits, split_notes = dirichlet_partition(
            labels[pool],
            job.clients // len(job.datasets),
            job.partition_alpha,
            job.partition_seed
        )
        record = dict(
            dataset=dataset,
            partition_seed=job.partition_seed,
            alpha=job.partition_alpha,
            root_indices=root_idx.tolist(),
            clients=[dict(
                train=pool[s.train_indices].tolist(),
                validation=pool[s.validation_indices].tolist(),
                class_counts=s.class_counts
            ) for s in splits],
            notes=split_notes
        )
        path = Path(partition_dir) / f'{dataset}_K{len(splits)}_a{job.partition_alpha}_seed{job.partition_seed}.json'
        if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(record)):
            raise RuntimeError('Persisted partition differs from regenerated partition: ' + str(path))
        save_json(path, record)
        notes += split_notes
        for split in record['clients']:
            if not split['train'] or not split['validation']:
                raise ValueError(f'Empty client split in {dataset}; change partition policy explicitly')
            group.append(dict(
                dataset=dataset,
                train=Subset(ds, split['train']),
                val=Subset(ds, split['validation']),
                n=len(split['train'])
            ))
        groups.append(group)
        roots[dataset] = Subset(ds, root_idx.tolist())
        tests[dataset] = test
    # Client ids interleave modalities (id c -> dataset c mod D), so the malicious set
    # {1..floor(rK)} of Task A2 spans every modality instead of filling PathMNIST first.
    clients = [group[i] for i in range(len(groups[0])) for group in groups]
    return (clients, tests, roots, notes)

def loader(ds, batch, shuffle=False, seed=42):
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed)
    )

@torch.no_grad()
def evaluate(model, ds, dataset, batch, device):
    model.eval()
    truth = []
    pred = []
    scores = []
    total = 0.0
    n = 0
    for x, y in loader(ds, batch):
        x = x.to(device)
        y = y.reshape(-1).long().to(device)
        logits = model(x, dataset)
        total += float(nn.functional.cross_entropy(logits, y, reduction='sum'))
        n += len(y)
        truth.extend(y.cpu().tolist())
        pred.extend(logits.argmax(1).cpu().tolist())
        scores.extend(logits.softmax(1).cpu().tolist())
    if not n:
        raise ValueError('Empty evaluation set')
    auc = float('nan')
    if DATASETS[dataset]['classes'] == 2 and len(set(truth)) == 2:
        auc = float(roc_auc_score(truth, np.array(scores)[:, 1]))
    return dict(
        accuracy=float(accuracy_score(truth, pred)),
        weighted_f1=float(f1_score(truth, pred, average='weighted', zero_division=0)),
        auc_roc=auc,
        loss=total / n,
        n=n
    )

class FlippedLabels(torch.utils.data.Dataset):
    """Label flipping (Table 3): a fixed, seeded label_fraction of this client's samples is
    relabelled to a uniform random class once, and stays relabelled in every round."""

    def __init__(self, base, classes, fraction, seed):
        self.base = base
        rng = np.random.default_rng(seed)
        self.flip = rng.random(len(base)) < fraction
        self.labels = rng.integers(0, classes, len(base))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if self.flip[idx]:
            y = torch.tensor([int(self.labels[idx])])
        return x, y

def train_local(model, state, client, job, rnd, cid, device, label_attack=False):
    local = copy.deepcopy(model).to(device)
    local.load_state_dict(state)
    local.train()
    opt = torch.optim.Adam(local.parameters(), lr=job.lr / math.sqrt(rnd), weight_decay=job.decay)
    train_set = client['train']
    if label_attack:
        train_set = FlippedLabels(train_set, DATASETS[client['dataset']]['classes'], job.label_fraction,
                                  job.seed * 100003 + cid)
    loss_sum = 0.0
    n = 0
    for epoch in range(job.epochs):
        for x, y in loader(train_set, job.batch, True, job.seed + 10000 * rnd + cid + epoch):
            y = y.reshape(-1).long()
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(local(x, client['dataset']), y)
            if job.method == 'FedProx':
                loss = loss + 0.5 * job.proximal_mu * sum((
                    (p - state[name].to(device)).square().sum() for name,
                    p in local.named_parameters()
                ))
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * len(y)
            n += len(y)
    return (subtract_state(local.state_dict(), state), loss_sum / n, local)

def malicious_sets(job):
    if job.attack == 'benign':
        return (set(), set())
    if job.attack == 'combined':
        n = int(0.15 * job.clients)
        return (set(range(n)), set(range(n, 2 * n)))
    ids = set(range(int(job.fraction * job.clients)))
    return (ids, ids if job.attack == 'free_riding' else set())

def aggregate(state, deltas, client_ids, clients, weights):
    """All selected clients update backbone; only matching modalities update heads."""
    result = clone_state(state)
    for key in state:
        eligible = [
            i for i,
            cid in enumerate(client_ids) if not key.startswith('heads.') or key.split('.')[1] == clients[cid]['dataset']
        ]
        denom = sum((weights[i] for i in eligible))
        if denom > 0:
            result[key] = state[key] + sum((deltas[i][key] * float(weights[i] / denom) for i in eligible))
    return result

def select_clients(job, trust, cache, round_idx):
    rng = np.random.default_rng(job.seed + round_idx)
    if job.method not in TRUST_METHODS or job.method == 'Random-Selection':
        return (
            rng.choice(job.clients, job.k, replace=False).tolist(),
            dict(solver='RANDOM', eligible=job.clients)
        )
    # Warm-up: every client contributes one update before trust/QUBO selection starts.
    # Clients without an update have a zero cached vector (cosine distance 1 to all), so
    # QUBO would otherwise never select them and lock onto the first k clients.
    warmup_rounds = -(-job.clients // job.k)
    if round_idx <= warmup_rounds:
        order = np.random.default_rng(job.seed).permutation(job.clients)
        ids = [int(order[(i + (round_idx - 1) * job.k) % job.clients]) for i in range(job.k)]
        return (ids, dict(solver='WARMUP_ROUND_ROBIN', eligible=job.clients, warmup_rounds=warmup_rounds))
    threshold = job.theta
    tau = np.ones(job.clients) if job.method == 'No-Trust' else np.array([trust.trust[i] for i in range(job.clients)])
    eligible = np.flatnonzero(tau >= threshold)
    while len(eligible) < job.k and threshold > 0:
        threshold = max(0.0, threshold - 0.1)
        eligible = np.flatnonzero(tau >= threshold)
    eligible_before_cap = len(eligible)
    if len(eligible) > QUBO_MAX_ELIGIBLE and job.method != 'No-QUBO':
        # Sec. 3.4 Stage 1 assumes the pre-filter leaves 10--15 clients ("K=40 exceeds
        # simulation limits"). Keep the 15 most-trusted eligible clients; ties are broken
        # by a seeded random key, never by client index.
        tie = rng.random(len(eligible))
        order = np.lexsort((tie, -tau[eligible]))
        eligible = np.sort(eligible[order[:QUBO_MAX_ELIGIBLE]])
    if job.method == 'No-QUBO':
        center = np.mean(cache, axis=0)
        dist = np.array([np.linalg.norm(v - center) for v in cache])
        cut = float(dist.mean() + 2 * dist.std())  # outlier rule: d_i > mean + 2 sigma_d
        good = [int(i) for i in eligible if dist[i] <= cut]
        if len(good) < job.k:
            good = list(map(int, eligible[np.argsort(dist[eligible])]))
        ids = good[:job.k]
        return (ids, dict(solver='L2', eligible=len(eligible), threshold=cut))
    if len(eligible) == job.k:
        return (eligible.tolist(), dict(solver='ALL_ELIGIBLE', eligible=len(eligible)))
    distances = pairwise_cosine_distances([cache[i] for i in eligible])
    inst = QuboInstance(tau[eligible], distances, job.lambda_, job.mu, job.k, eligible.tolist())
    # Sec. 6 allowed approximation: K=80 with |E|>12 uses SA. SA is also the
    # 'emergency fallback if QAOA times out'. Both are recorded per round.
    fallback = 'K80_large_E' if job.clients == 80 and len(eligible) > 12 else False
    qaoa_s = None
    if job.solver == 'sa' or fallback:
        sol = solve_sa(inst, seed=job.seed + round_idx)
    else:
        from fedqtrust.quantum.qaoa import QAOATimeout
        sampler = None
        if job.noise:
            from .benchmarks import noisy_sampler
            sampler = noisy_sampler(job.shots, job.seed + round_idx)
        qaoa_start = time.perf_counter()
        try:
            sol = solve_qaoa(
                inst,
                reps=job.p,
                shots=job.shots,
                maxiter=job.maxiter,
                seed=job.seed + round_idx,
                sampler=sampler,
                timeout_s=job.qaoa_timeout_s
            )
        except QAOATimeout:
            qaoa_s = time.perf_counter() - qaoa_start
            fallback = 'QAOA_timeout'
            sol = solve_sa(inst, seed=job.seed + round_idx)
    projected = project_cardinality(sol.bitstring, inst.trust, job.k)
    return (
        eligible[projected.astype(bool)].tolist(),
        dict(
            solver=sol.solver_used,
            eligible=len(eligible),
            eligible_before_cap=eligible_before_cap,
            raw_objective=sol.objective,
            raw_feasible=sol.feasible_cardinality,
            noise_model='Nairobi-derived homogeneous calibration' if job.noise else 'none',
            sa_fallback=fallback,
            qaoa_timeout_elapsed_s=qaoa_s,
            qaoa_expected_objective=getattr(sol, 'expected_objective', None),
            trust=inst.trust.tolist(),
            distances=distances.tolist(),
            eligible_ids=eligible.tolist()
        )
    )

def backbone_vector(state):
    return flatten_state({k: v for k, v in state.items() if not k.startswith('heads.')}).numpy()

def _run_job(
    job: Job,
    out,
    device='cpu',
    infrastructure='paper',
    data=None,
    model_factory=JointModel,
    data_root='data/raw'
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = asdict(job) | dict(
        infrastructure=infrastructure,
        device=device,
        data_root=str(Path(data_root).resolve()),
        engine_version=4
    )
    config_path = out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != json.loads(json.dumps(config)):
        raise ValueError('Refusing to resume a different configuration in the same directory')
    save_json(config_path, config)
    if (out / 'DONE').exists():
        return
    seed_everything(job.seed)
    clients, tests, roots, notes = data or prepare_data(job, data_root, out.parent.parent / 'partitions')
    save_json(out / 'partition_notes.json', notes)
    quantum = job.method in QUANTUM_METHODS
    model = (JointModel(job.datasets, quantum, backend=job.qnn_backend) if model_factory is JointModel
             else model_factory(job.datasets, quantum)).to(device)
    state = clone_state(model.state_dict())
    initial = backbone_vector(state)
    cache = [np.zeros_like(initial) for _ in clients]
    trust = TrustManager(
        job.clients,
        alpha=job.alpha,
        beta=job.beta,
        gamma=job.gamma,
        reputation_decay=job.reputation_decay,
        consistency_window=job.window
    )
    from .checkpoint import restore_round, save_round
    restored = None
    if (out / 'checkpoint.pt').exists():
        restored = restore_round(out / 'checkpoint.pt', trust, device)
        state, cache = (restored['state'], restored['cache'])
        model.load_state_dict(state)
        print(f"[RESUME] {job.id} after round {restored['round']}", flush=True)
    uses_infra = job.method in TRUST_METHODS or job.method in {'PQS-BFL', 'SecEdge-MC'}
    use_chain = uses_infra and job.method != 'No-Blockchain'
    attempt_namespace = job.id + '_' + uuid.uuid4().hex[:12]
    save_json(out / 'attempt.json', dict(namespace=attempt_namespace))
    chain = HyperledgerFabricBlockchain(namespace=attempt_namespace) if infrastructure == 'paper' and use_chain else InMemoryBlockchain()
    if not chain.ping():
        raise RuntimeError('Blockchain unavailable')
    if not (infrastructure in ('paper', 'local') and uses_infra):
        mode = 'none'
    elif job.method == 'No-PQC':
        mode = 'classical'  # RSA-2048 + ECDSA
    elif job.method == 'PQS-BFL':
        mode = 'pqc_sign'  # ML-DSA-65 only (Table 4)
    else:
        mode = 'pqc'  # ML-KEM-768 + ML-DSA-65
    transport = Transport(mode)
    for cid in range(job.clients):
        chain.set_trust(cid, trust.trust[cid])
    effective_job = job
    if restored:
        from dataclasses import replace
        effective_job = replace(job, proximal_mu=restored['proximal_mu'])
    if job.method == 'FedProx' and (not restored):
        from dataclasses import replace
        validation_trials = []
        # Match initialization, client set and seed; tune only on held-out validation.
        for mu in (0.001, 0.01, 0.1):
            candidate = replace(job, proximal_mu=mu)
            seed_everything(job.seed)
            scores = []
            for cid in range(min(job.k, len(clients))):
                _, _, local = train_local(model, state, clients[cid], candidate, 1, cid, device)
                scores.append(evaluate(
                    local,
                    clients[cid]['val'],
                    clients[cid]['dataset'],
                    job.batch,
                    device
                )['accuracy'])
            validation_trials.append(dict(mu=mu, validation_accuracy=float(np.mean(scores))))
        best = max(validation_trials, key=lambda row: row['validation_accuracy'])['mu']
        effective_job = replace(job, proximal_mu=best)
        write_csv(out / 'fedprox_validation_warmup.csv', validation_trials)
    bad, free = malicious_sets(job)
    bad |= free
    rows = restored['rows'] if restored else []
    trust_trace = restored.get('trust_trace', []) if restored else []
    client_rows = restored['clients'] if restored else []
    timings = restored['timing'] if restored else []
    selection_rows = restored['selections'] if restored else []
    elapsed_before = restored['elapsed_s'] if restored else 0.0
    first_round = restored['round'] + 1 if restored else 1
    start = time.perf_counter()
    for rnd in range(first_round, job.rounds + 1):
        tick = time.perf_counter()
        selection_start = tick
        if use_chain:
            for cid in range(job.clients):
                value = chain.get_trust(cid)
                if value is None:
                    raise RuntimeError('Missing committed trust')
                trust.trust[cid] = value
        trust_read_s = time.perf_counter() - selection_start
        selection_start = time.perf_counter()
        selected, selection = select_clients(job, trust, cache, rnd)
        selection_s = time.perf_counter() - selection_start
        selection_rows.append(dict(round=rnd, selected=selected, **selection))
        blockchain_s = trust_read_s if use_chain else 0.0
        t = time.perf_counter()
        if use_chain:
            chain.log_selection(dict(round=rnd, selected=selected, **selection))
        blockchain_s += time.perf_counter() - t
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        deltas = []
        losses = []
        validations = []
        local_times = []
        train_start = time.perf_counter()
        active = rnd >= job.attack_start
        for cid in selected:
            local_start = time.perf_counter()
            delta, loss, local = train_local(
                model,
                state,
                clients[cid],
                effective_job,
                rnd,
                cid,
                device,
                label_attack=active and cid in bad and (cid not in free) and (job.attack in {
                    'label_flip',
                    'combined'
                })
            )
            local_times.append(time.perf_counter() - local_start)
            deltas.append(delta)
            losses.append(loss)
            validations.append(evaluate(
                local,
                clients[cid]['val'],
                clients[cid]['dataset'],
                job.batch,
                device
            )['accuracy'])
        training_s = time.perf_counter() - train_start
        # Only attack construction may use simulator ground-truth identities.
        actual_z = job.lie_z
        lie_accepted = None
        if active:
            honest = [d for cid, d in zip(selected, deltas) if cid not in bad]
            if job.attack == 'lie' and honest and any((cid in bad for cid in selected)):
                from fedqtrust.attacks.lie import adaptive_lie
                _, actual_z, lie_accepted, _ = adaptive_lie(
                    [backbone_vector(d) for d in honest],
                    sum((cid in bad for cid in selected)),
                    job.lie_z
                )
            for i, cid in enumerate(selected):
                if cid not in bad:
                    continue
                if cid in free:
                    deltas[i] = {k: torch.zeros_like(v) for k, v in deltas[i].items()}
                elif job.attack == 'sign_flip':
                    deltas[i] = {k: -v for k, v in deltas[i].items()}
                elif job.attack == 'gaussian':
                    gen = torch.Generator(device=device).manual_seed(job.seed + rnd * 1000 + cid)
                    deltas[i] = {
                        k: v + job.sigma * torch.randn(
                            v.shape,
                            generator=gen,
                            device=v.device,
                            dtype=v.dtype
                        ) if not k.startswith('heads.') or k.split('.')[1] == clients[cid]['dataset'] else v for k,
                        v in deltas[i].items()
                    }
                elif job.attack == 'lie' and honest:
                    # Shared backbone can be poisoned coherently across heterogeneous heads.
                    for key in deltas[i]:
                        if key.startswith('heads.') and key.split('.')[1] != clients[cid]['dataset']:
                            continue
                        h = [
                            d[key] for c,
                            d in zip(
                                selected,
                                deltas
                            ) if c not in bad and (not key.startswith('heads.') or clients[c]['dataset'] == clients[cid]['dataset'])
                        ]
                        if h:
                            stacked = torch.stack(h)
                            deltas[i][key] = stacked.mean(0) - actual_z * stacked.std(0, unbiased=False)
        transfer_s = 0.0
        kem_s = 0.0
        sign_s = 0.0
        wire_bytes = 0
        update_bytes = 0
        signature_bytes = 0
        for i, delta in enumerate(deltas):
            buff = io.BytesIO()
            torch.save(delta, buff)
            payload = buff.getvalue()
            decoded, stats = transport.roundtrip(payload, f'{job.id}:{rnd}:{selected[i]}'.encode())
            signed = transport.authenticate(
                f'{job.id}:{rnd}:{selected[i]}:'.encode() + payload,
                signer_id=str(selected[i])
            )
            if use_chain and infrastructure == 'paper' and ('proof' in signed):
                t = time.perf_counter()
                chain.verify_record(signed['proof'])
                blockchain_s += time.perf_counter() - t
            # Deserialize the verified payload rather than aggregating an untransported copy.
            deltas[i] = torch.load(io.BytesIO(decoded), map_location=device, weights_only=True)
            transfer_s += stats['transport_s'] + signed['sign_verify_s']
            kem_s += stats['transport_s']
            sign_s += signed['sign_verify_s']
            wire_bytes += stats['wire_bytes'] + signed['signature_bytes']
            update_bytes += stats['update_bytes']
            signature_bytes += signed['signature_bytes']
        vectors = [backbone_vector(d) for d in deltas]
        # Observable aggregate reference: never an oracle average of honest clients.
        reference = np.mean(vectors, axis=0)
        scores = np.array([trust.trust[cid] for cid in selected])
        effective_f = None
        krum_idx = None
        krum_scores = None
        if job.method == 'Krum':
            krum_idx, effective_f = krum_select(vectors, int(job.fraction * len(selected)))
            neighbors = max(1, len(selected) - effective_f - 2)
            krum_scores = [
                sum(sorted((
                    float(np.sum((v - u) ** 2)) for j,
                    u in enumerate(vectors) if j != i
                ))[:neighbors]) for i,
                v in enumerate(vectors)
            ]
        divergence = pairwise_cosine_distances(vectors).mean(axis=1)
        l2_scores = np.array([np.linalg.norm(v - reference) for v in vectors])
        if job.method == 'FLTrust':
            root_deltas = []
            for d, root in roots.items():
                root_client = dict(train=root, dataset=d)
                root_deltas.append(train_local(model, state, root_client, job, rnd, job.clients, device)[0])
            root_reference = np.mean([backbone_vector(d) for d in root_deltas], axis=0)
            scores = fltrust_scores(vectors, root_reference)
            norm = np.linalg.norm(root_reference)
            deltas = [
                {k: v * float(norm / max(np.linalg.norm(vec), 1e-12)) for k, v in delta.items()} for vec,
                delta in zip(vectors, deltas)
            ]
        for i, cid in enumerate(selected):
            consistency = trust.update_consistency(cid, vectors[i], reference)
            trust.update_round(cid, validations[i], consistency)
            cache[cid] = vectors[i]
            detection_score = 1 - trust.trust[cid] + float(divergence[i])
            predicted = trust.trust[cid] < job.theta
            decision_rule = 'trust_threshold'
            if job.method == 'FLTrust':
                detection_score = float(1 - scores[i])
                predicted = bool(scores[i] <= 0)
                decision_rule = 'nonpositive_root_cosine'
            elif job.method == 'Krum':
                detection_score = krum_scores[i]
                predicted = i != krum_idx
                decision_rule = 'not_selected_by_krum'
            elif job.method == 'No-QUBO':
                detection_score = float(l2_scores[i])
                predicted = detection_score > float(l2_scores.mean() + 2 * l2_scores.std())
                decision_rule = 'l2_2std'
            client_rows.append(dict(
                round=rnd,
                client=cid,
                dataset=clients[cid]['dataset'],
                is_malicious=cid in bad,
                attack_active=active,
                local_training_s=local_times[i],
                accuracy=validations[i],
                consistency=consistency,
                trust=trust.trust[cid],
                loss=losses[i],
                selected=True,
                detection_score=detection_score,
                decision_rule=decision_rule,
                predicted_malicious=predicted
            ))
            if use_chain:
                t = time.perf_counter()
                chain.set_trust(cid, trust.trust[cid])
                blockchain_s += time.perf_counter() - t
        # Trust of every client after this round (Fig. 4: 3 benign + 3 malicious traces).
        trust_trace.extend(dict(round=rnd, client=cid, dataset=clients[cid]['dataset'],
                                is_malicious=cid in bad, selected=cid in selected,
                                trust=float(trust.trust[cid])) for cid in range(job.clients))
        if job.method == 'Krum':
            weights = [float(i == krum_idx) for i in range(len(selected))]
        elif job.method == 'FLTrust':
            weights = scores.tolist()
        elif job.method in TRUST_METHODS and job.method != 'No-Trust':
            weights = [clients[cid]['n'] * trust.trust[cid] for cid in selected]
        else:
            weights = [clients[cid]['n'] for cid in selected]
        t = time.perf_counter()
        new_state = aggregate(state, deltas, selected, clients, weights)
        gradient_proxy = float(sum(((new_state[k] - state[k]).square().sum().item() for k in state)) / (job.lr / math.sqrt(rnd)) ** 2)
        state = new_state
        model.load_state_dict(state)
        aggregation_s = time.perf_counter() - t
        buff = io.BytesIO()
        torch.save(state, buff)
        global_payload = f'{job.id}:{rnd}:global:'.encode() + buff.getvalue()
        auth = transport.authenticate(global_payload)
        client_verify_start = time.perf_counter()
        if 'proof' in auth:
            for _ in selected:
                transport.verify(global_payload, auth['proof'])
        transfer_s += time.perf_counter() - client_verify_start
        sign_s += time.perf_counter() - client_verify_start
        if use_chain and infrastructure == 'paper' and ('proof' in auth):
            t = time.perf_counter()
            chain.verify_record(auth['proof'])
            blockchain_s += time.perf_counter() - t
        transfer_s += auth['sign_verify_s']
        sign_s += auth['sign_verify_s']
        signature_bytes += auth['signature_bytes']
        wire_bytes += (auth['signature_bytes'] + len(buff.getvalue())) * len(selected)
        metrics = {d: evaluate(model, ds, d, job.batch, device) for d, ds in tests.items()}
        elapsed = time.perf_counter() - tick
        for d, m in metrics.items():
            rows.append(dict(
                round=rnd,
                dataset=d,
                **m,
                train_loss=float(np.mean(losses)),
                gradient_proxy=gradient_proxy,
                lr=job.lr / math.sqrt(rnd),
                round_time_s=elapsed
            ))
        timings.append(dict(
            round=rnd,
            training_s=training_s,
            selection_s=selection_s,
            blockchain_s=blockchain_s,
            transport_sign_s=transfer_s,
            kem_aes_s=kem_s,
            signature_s=sign_s,
            aggregation_s=aggregation_s,
            total_s=elapsed,
            update_bytes=update_bytes,
            wire_bytes=wire_bytes,
            signature_bytes=signature_bytes,
            solver=selection['solver'],
            eligible=selection['eligible'],
            krum_effective_f=effective_f,
            lie_z_used=actual_z,
            lie_accepted_by_krum=lie_accepted,
            malicious_selected=sum((cid in bad for cid in selected)),
            gpu_peak_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0
        ))
        write_csv(out / 'rounds.csv', rows)
        write_csv(out / 'clients.csv', client_rows)
        write_csv(out / 'timing.csv', timings)
        write_csv(out / 'trust_trace.csv', trust_trace)
        save_json(out / 'selections.json', selection_rows)
        save_round(
            out / 'checkpoint.pt',
            round_index=rnd,
            state=state,
            trust=trust,
            cache=cache,
            rows=rows,
            clients=client_rows,
            timing=timings,
            selections=selection_rows,
            proximal_mu=effective_job.proximal_mu,
            elapsed_s=elapsed_before + time.perf_counter() - start,
            trust_trace=trust_trace
        )
        print(
            f"[{job.experiment}] {job.id} {rnd}/{job.rounds} acc={np.mean([m['accuracy'] for m in metrics.values()]):.4f}",
            flush=True
        )
    metrics = {d: evaluate(model, ds, d, job.batch, device) for d, ds in tests.items()}
    final = [
        dict(dataset=d, **m, effective_proximal_mu=effective_job.proximal_mu) for d,
        m in metrics.items()
    ]
    write_csv(out / 'final.csv', final)
    write_csv(out / 'final_trust.csv', [dict(
        client=cid,
        dataset=clients[cid]['dataset'],
        role='free_rider' if cid in free else ('poisoner' if cid in bad else 'honest'),
        is_malicious=cid in bad,
        trust=float(trust.trust[cid]),
        predicted_malicious=bool(trust.trust[cid] < job.theta),
        times_selected=sum(1 for row in selection_rows if cid in row['selected'])
    ) for cid in range(job.clients)])
    torch.save(state, out / 'final_model.pt')
    save_json(
        out / 'metadata.json',
        dict(
            elapsed_s=elapsed_before + time.perf_counter() - start,
            parameters=sum((p.numel() for p in model.parameters())),
            quantum_parameters=int(model.features.quantum_weights.numel()) if hasattr(getattr(model, 'features', None), 'quantum_weights') else 0,
            infrastructure=infrastructure,
            ledger_backend='fabric' if infrastructure == 'paper' and use_chain else 'local_memory',
            cryptography_mode=mode,
            fabric_network_measured=infrastructure == 'paper' and use_chain,
            model_backend=job.qnn_backend if job.method in QUANTUM_METHODS else 'pytorch_cnn',
            partition_notes=notes,
            protocol_notes=[
                'Root dataset: 1% withheld for every method for matched splits.',
                'Client validation uses local held-out examples.',
                'Single-process client simulation; no network latency claims.',
                'Krum effective f bounded by participating updates; theoretical limits reported separately.',
                'LIE uses a normal-quantile bound and actual Krum acceptance search, capped by the configured z sweep.',
                ('Signatures verified in application and by Fabric VerifyRecord.' if infrastructure == 'paper'
                 else 'Local ledger execution: no Fabric network, consensus or on-chain signature measurements.')
            ]
        )
    )
    (out / 'DONE').write_text('MEASURED_JOB_COMPLETE\n')

def run_job(
    job: Job,
    out,
    device='cpu',
    infrastructure='paper',
    data=None,
    model_factory=JointModel,
    data_root='data/raw'
):
    import fcntl
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('This job is already running: ' + job.id) from exc
        return _run_job(job, out, device, infrastructure, data, model_factory, data_root)
