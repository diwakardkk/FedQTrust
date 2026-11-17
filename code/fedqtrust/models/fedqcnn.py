"""Four-qubit patch QNN specified by the manuscript (no classical surrogate).

Circuit (manuscript Sec. 6.2): per 2x2 patch, ZZFeatureMap(4, reps=1, full
entanglement) followed by RealAmplitudes(4, reps=2, linear entanglement), read out
as <Z0>,<Z1>,<Z2>,<Z3>.

Two backends compute the *same* circuit:

* ``torch_statevector`` (default): exact 16-amplitude statevector simulation of
  the circuit written as tensor operations. Gradients are exact analytic
  derivatives (identical to parameter-shift values). Batched on CPU or GPU.
* ``qiskit``: qiskit_machine_learning EstimatorQNN + TorchConnector with
  ParamShiftEstimatorGradient. Numerically identical but ~10^4-10^5x slower,
  because each patch is a separate Qiskit circuit evaluation on CPU.

``tests/test_manuscript.py::test_torch_statevector_matches_qiskit_estimatorqnn``
checks that outputs and gradients of both backends agree.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

N_QUBITS = 4
DIM = 2 ** N_QUBITS
# Qiskit little-endian convention: basis index = sum_q b_q 2^q.
_BITS = torch.tensor([[(idx >> q) & 1 for q in range(N_QUBITS)] for idx in range(DIM)], dtype=torch.float64)
_PAIRS = [(i, j) for i in range(N_QUBITS) for j in range(i + 1, N_QUBITS)]  # full entanglement
_LINEAR = [(0, 1), (1, 2), (2, 3)]  # RealAmplitudes linear entanglement


def _cx_permutation(control: int, target: int) -> torch.Tensor:
    perm = torch.arange(DIM)
    for idx in range(DIM):
        if (idx >> control) & 1:
            perm[idx] = idx ^ (1 << target)
    return perm


def _cx_chain_permutation() -> torch.Tensor:
    # Applying CX(0,1), CX(1,2), CX(2,3) in sequence to basis states.
    perm = torch.arange(DIM)
    for control, target in _LINEAR:
        perm = _cx_permutation(control, target)[perm]
    return perm


_CHAIN = _cx_chain_permutation()


def _ry_layer(thetas: torch.Tensor) -> torch.Tensor:
    """16x16 real matrix for RY(theta_q) on every qubit q (qubit 0 = least significant)."""
    mats = []
    for q in range(N_QUBITS):
        c, s = torch.cos(thetas[q] / 2), torch.sin(thetas[q] / 2)
        mats.append(torch.stack([torch.stack([c, -s]), torch.stack([s, c])]))
    # kron order: most significant qubit first.
    out = mats[3]
    for q in (2, 1, 0):
        out = torch.kron(out, mats[q])
    return out


def ansatz_unitary(weights: torch.Tensor, reps: int = 2) -> torch.Tensor:
    """Real unitary of RealAmplitudes(4, reps, linear): RY layer, (CX chain, RY layer) x reps."""
    eye = torch.eye(DIM, dtype=weights.dtype, device=weights.device)
    cx = eye[:, _CHAIN.to(weights.device)]  # column b -> basis vector e_{perm(b)}
    unitary = _ry_layer(weights[0:4])
    for r in range(1, reps + 1):
        unitary = _ry_layer(weights[4 * r:4 * r + 4]) @ cx @ unitary
    return unitary


def zz_feature_state(x: torch.Tensor) -> torch.Tensor:
    """ZZFeatureMap(4, reps=1, full): H^4 then diagonal phases. x: (N,4) -> (N,16) complex."""
    bits = _BITS.to(device=x.device, dtype=x.dtype)
    phase = 2.0 * x @ bits.T  # P(2 x_q) on each qubit
    for i, j in _PAIRS:
        parity = (bits[:, i] + bits[:, j]) % 2  # CX P(2 phi) CX acts on b_i XOR b_j
        phi = (math.pi - x[:, i]) * (math.pi - x[:, j])
        phase = phase + 2.0 * phi[:, None] * parity[None, :]
    amp = 1.0 / math.sqrt(DIM)
    return torch.polar(torch.full_like(phase, amp), phase)


def circuit_expectations(x: torch.Tensor, weights: torch.Tensor, reps: int = 2) -> torch.Tensor:
    """Exact <Z_q> for every row of x (N,4) -> (N,4)."""
    state = zz_feature_state(x)
    unitary = ansatz_unitary(weights, reps).to(state.dtype)
    out = state @ unitary.T
    prob = out.real.square() + out.imag.square()
    z = 1.0 - 2.0 * _BITS.to(device=x.device, dtype=prob.dtype)
    return prob @ z


class QuantumPatchLayer(nn.Module):

    def __init__(self, patch_size=2, stride=2, reps=2, angle_scale=math.pi, chunk_size=64,
                 backend='torch_statevector'):
        super().__init__()
        if patch_size != 2:
            raise ValueError('Only the specified four-qubit 2x2 patch circuit is supported.')
        if backend not in ('torch_statevector', 'qiskit'):
            raise ValueError(f'Unknown QNN backend {backend}')
        self.patch_size, self.stride, self.reps = (patch_size, stride, reps)
        self.angle_scale, self.chunk_size, self.backend = (angle_scale, chunk_size, backend)
        initial = np.random.uniform(-0.05, 0.05, N_QUBITS * (reps + 1))
        if backend == 'torch_statevector':
            self.weight = nn.Parameter(torch.tensor(initial, dtype=torch.float32))
            self._circuit = None
            return
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import zz_feature_map, real_amplitudes
        from qiskit.primitives import StatevectorEstimator
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_machine_learning.neural_networks import EstimatorQNN
        from qiskit_machine_learning.connectors import TorchConnector
        from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
        feature = zz_feature_map(4, reps=1)
        ansatz = real_amplitudes(4, reps=reps, entanglement='linear')
        circuit = QuantumCircuit(4)
        circuit.compose(feature, inplace=True)
        circuit.compose(ansatz, inplace=True)
        estimator = StatevectorEstimator(default_precision=0.0, seed=42)
        qnn = EstimatorQNN(
            circuit=circuit,
            estimator=estimator,
            observables=[SparsePauliOp.from_list([('I' * (3 - i) + 'Z' + 'I' * i, 1)]) for i in range(4)],
            input_params=list(feature.parameters),
            weight_params=list(ansatz.parameters),
            gradient=ParamShiftEstimatorGradient(estimator),
            input_gradients=False,  # inputs are pixels; their gradients are never used
            default_precision=0.0
        )
        self.connector = TorchConnector(qnn, initial_weights=initial)
        self._circuit = circuit

    @property
    def quantum_weights(self):
        return self.weight if self.backend == 'torch_statevector' else self.connector.weight

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError('Expected B x 1 x H x W grayscale images')
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.stride).transpose(1, 2)
        flat = (patches * self.angle_scale).reshape(-1, 4)
        if self.backend == 'torch_statevector':
            out = circuit_expectations(flat, self.weight.to(flat.dtype), self.reps)
            return out.to(x.dtype).reshape(x.shape[0], -1)
        # Qiskit simulates on CPU; TorchConnector preserves autograd across transfers.
        outputs = [self.connector(chunk) for chunk in flat.split(self.chunk_size)]
        return torch.cat(outputs).reshape(x.shape[0], -1)

    def circuit_metadata(self):
        return dict(
            number_of_qubits=4,
            circuit_depth=4 + 3 * self.reps,
            number_of_trainable_quantum_parameters=self.quantum_weights.numel(),
            number_of_observables=4,
            observables='<Z0>,<Z1>,<Z2>,<Z3>',
            backend=self.backend,
            feature_map='ZZFeatureMap(reps=1, full entanglement)',
            ansatz=f'RealAmplitudes(reps={self.reps}, linear entanglement)',
            gradient='exact analytic (autograd)' if self.backend == 'torch_statevector' else 'parameter_shift',
            reps=self.reps
        )


class FedQCNN(nn.Module):

    def __init__(self, num_classes, patch_size=2, stride=2, angle_scale=math.pi, backend='torch_statevector'):
        super().__init__()
        self.quantum = QuantumPatchLayer(patch_size, stride, angle_scale=angle_scale, backend=backend)
        flat = ((28 - patch_size) // stride + 1) ** 2 * 4
        self.backbone = nn.Sequential(
            self.quantum,
            nn.Linear(flat, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))
