"""Atomic round checkpoints, including randomness and selection history."""
from collections import deque
from pathlib import Path
import random
import uuid
import numpy as np
import torch

def save_round(
    path,
    *,
    round_index,
    state,
    trust,
    cache,
    rows,
    clients,
    timing,
    selections,
    proximal_mu,
    elapsed_s,
    trust_trace=None
):
    numpy_state = np.random.get_state()
    record = {
        'round': round_index,
        'state': {key: value.detach().cpu() for key, value in state.items()},
        'trust': {name: dict(getattr(trust, name)) for name in ('accuracy', 'reputation', 'trust')},
        'consistency': {key: list(value) for key, value in trust.consistency_history.items()},
        'cache': [torch.from_numpy(value.copy()) for value in cache],
        'rows': rows,
        'clients': clients,
        'timing': timing,
        'selections': selections,
        'proximal_mu': proximal_mu,
        'elapsed_s': elapsed_s,
        'trust_trace': trust_trace or [],
        'random_state': random.getstate(),
        'numpy_state': (
            numpy_state[0],
            torch.tensor(numpy_state[1].astype('int64')),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4]
        ),
        'torch_state': torch.get_rng_state(),
        'cuda_states': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    }
    target = Path(path)
    temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
    torch.save(record, temporary)
    temporary.replace(target)

def restore_round(path, trust, device):
    record = torch.load(path, map_location='cpu', weights_only=True)
    for name, values in record['trust'].items():
        setattr(trust, name, values)
    for client, values in record['consistency'].items():
        trust.consistency_history[client] = deque(values)
    record['state'] = {key: value.to(device) for key, value in record['state'].items()}
    record['cache'] = [value.numpy() for value in record['cache']]
    random.setstate(record['random_state'])
    name, keys, position, has_gaussian, gaussian = record['numpy_state']
    np.random.set_state((name, keys.numpy().astype('uint32'), position, has_gaussian, gaussian))
    torch.set_rng_state(record['torch_state'])
    if record['cuda_states'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(record['cuda_states'])
    return record
