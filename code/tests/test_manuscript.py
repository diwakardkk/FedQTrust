import json
from dataclasses import replace,asdict
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset
from fedqtrust.manuscript.plan import Job,build_plan
from fedqtrust.manuscript.engine import aggregate,run_job,save_json,malicious_sets
from fedqtrust.manuscript.report import build_report
from fedqtrust.models.fedqcnn import QuantumPatchLayer
from fedqtrust.quantum.qaoa import solve_qaoa
from fedqtrust.quantum.qubo import QuboInstance,qubo_objective

def test_real_circuit_expectations_and_parameter_shift():
    from qiskit.quantum_info import Statevector,SparsePauliOp
    q=QuantumPatchLayer(backend='qiskit');x=torch.tensor([[[[.1,.2],[.3,.4]]]])
    output=q(x)
    network=q.connector.neural_network
    values=dict(zip(network.input_params,(x.detach().numpy().reshape(-1)*np.pi)))
    values.update(zip(network.weight_params,q.quantum_weights.detach().numpy()))
    sv=Statevector.from_instruction(network.circuit.assign_parameters(values))
    expected=[sv.expectation_value(SparsePauliOp.from_list([('I'*(3-i)+'Z'+'I'*i,1)])).real for i in range(4)]
    assert np.allclose(output.detach().numpy().reshape(-1),expected,atol=1e-5)
    output.sum().backward()
    assert torch.isfinite(q.quantum_weights.grad).all() and q.quantum_weights.grad.norm()>0
    assert q.quantum_weights.numel()==12

def test_qaoa_executes_and_reports_actual_objective():
    instance=QuboInstance(np.array([.9,.2]),np.zeros((2,2)),1,2,1,[0,1])
    s=solve_qaoa(instance,reps=1,shots=128,maxiter=8)
    assert s.solver_used=='QAOA'
    assert s.objective==pytest.approx(qubo_objective(s.bitstring,instance.trust,instance.distances,1,2,1))

def test_full_plan_settings_and_real_sweeps():
    jobs=build_plan();assert len({j.id for j in jobs})==len(jobs)
    assert {j.seed for j in jobs if j.experiment=='E1'}=={42,123,456,789,999}
    assert all(j.clients==40 and j.epochs==3 and j.batch==32 and j.lr==.01 for j in jobs if j.experiment=='E1')
    assert all(j.rounds==200 for j in jobs if j.experiment=='E3')
    assert {j.k for j in jobs if j.experiment=='E9'}=={3,5,10}
    assert {j.method for j in jobs if j.experiment=='E9'}=={'FedQTrust','Krum'}
    assert all(j.gamma==.3 and j.beta==pytest.approx(.7-j.alpha) for j in jobs if j.experiment=='E6')
    assert {j.p for j in jobs if j.experiment=='E6'}=={1,2,3,4}
    assert any(j.noise for j in jobs if j.experiment=='E2')

def test_modality_heads_do_not_dilute_or_cross_aggregate():
    state={'backbone.w':torch.zeros(1),'heads.a.w':torch.zeros(1),'heads.b.w':torch.zeros(1)}
    a={'backbone.w':torch.tensor([2.]),'heads.a.w':torch.tensor([4.]),'heads.b.w':torch.tensor([99.])}
    b={'backbone.w':torch.tensor([6.]),'heads.a.w':torch.tensor([99.]),'heads.b.w':torch.tensor([8.])}
    result=aggregate(state,[a,b],[0,1],[{'dataset':'a'},{'dataset':'b'}],[1,3])
    assert result['backbone.w'].item()==5
    assert result['heads.a.w'].item()==4
    assert result['heads.b.w'].item()==8

def test_combined_attacks_disjoint():
    a,b=malicious_sets(Job(attack='combined',fraction=.3))
    assert not a&b and len(a)==6 and len(b)==6

class Tiny(nn.Module):
    def __init__(self,datasets,quantum):
        super().__init__();self.backbone=nn.Linear(4,3);self.heads=nn.ModuleDict({d:nn.Linear(3,2) for d in datasets})
    def forward(self,x,d):return self.heads[d](self.backbone(x.flatten(1)))

@pytest.mark.parametrize("method", ["FedAvg", "FedProx", "FedQTrust"])
def test_small_federated_execution_resume_and_report(tmp_path, method):
    torch.set_num_threads(1)
    job=Job(experiment='E3',datasets=('pneumoniamnist',),method=method,solver='sa',clients=3,k=2,rounds=3,epochs=1,batch=4)
    ds=TensorDataset(torch.rand(12,1,2,2),torch.tensor([0,1]*6))
    clients=[dict(dataset='pneumoniamnist',train=ds,val=ds,n=12) for _ in range(3)]
    data=(clients,{'pneumoniamnist':ds},{'pneumoniamnist':ds},[])
    out=tmp_path/'jobs'/job.id
    run_job(job,out,infrastructure='development',data=data,model_factory=Tiny)
    import pandas as pd
    rounds=pd.read_csv(out/'rounds.csv')
    assert len(rounds)==3 and (rounds.gradient_proxy>=0).all()
    checkpoint=torch.load(out/'final_model.pt',weights_only=True)
    assert checkpoint
    before=(out/'final.csv').read_bytes()
    run_job(job,out,infrastructure='development',data=data,model_factory=Tiny)
    assert (out/'final.csv').read_bytes()==before
    with pytest.raises(ValueError,match='different configuration'):
        run_job(replace(job,rounds=4),out,infrastructure='development',data=data,model_factory=Tiny)
    save_json(tmp_path/'plan.json',{'experiments':['E3'],'jobs':[dict(id=job.id,**asdict(job)),dict(id='missing',experiment='E5')]})
    build_report(tmp_path)
    audit=json.loads((tmp_path/'audit.json').read_text())
    assert not audit['all_jobs_complete'] and not audit['publication_ready']
    assert (tmp_path/'tables/convergence_fits.csv').exists()
    assert (tmp_path/'main.tex').exists()

def test_classical_transport_and_signature_proofs():
    from fedqtrust.security.transport import Transport
    from cryptography.hazmat.primitives import serialization,hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    t=Transport('classical');payload=b'actual model update'
    received,stats=t.roundtrip(payload,b'run:round:client')
    assert received==payload and stats['wire_bytes']>len(payload)
    proof=t.authenticate(payload,signer_id='1')['proof']
    public=serialization.load_der_public_key(bytes.fromhex(proof['public_key']))
    public.verify(bytes.fromhex(proof['signature']),bytes.fromhex(proof['digest']),ec.ECDSA(hashes.SHA256()))
    assert proof['scheme']=='ECDSA-P256'

def test_chaincode_rejects_invalid_signature_replay_and_owner():
    import shutil,subprocess
    from pathlib import Path
    if not shutil.which('node'):pytest.skip('Node unavailable')
    subprocess.run(['node',str(Path(__file__).with_name('chaincode_test.cjs'))],check=True)

def test_lie_search_records_krum_acceptance():
    from fedqtrust.attacks.lie import adaptive_lie
    honest=[np.array([float(i),.1*i]) for i in range(5)]
    vector,z,accepted,f=adaptive_lie(honest,1,z_cap=1.)
    assert vector.shape==(2,) and 0<=z<=1 and f==1
    assert isinstance(accepted,bool)

def test_partition_sparse_labels_preserved_and_disjoint():
    from fedqtrust.data.partition import dirichlet_partition
    labels=np.array([0]*75+[1]*5)
    splits,notes=dirichlet_partition(labels,20,.1,42)
    seen=[]
    for s in splits:
        assert s.train_indices and s.validation_indices
        assert not set(s.train_indices)&set(s.validation_indices)
        seen.extend(s.train_indices+s.validation_indices)
        assert set(s.class_counts)=={0,1}
    assert sorted(seen)==list(range(len(labels)))
    assert notes


def test_interrupted_round_resume_matches_uninterrupted(tmp_path, monkeypatch):
    import fedqtrust.manuscript.checkpoint as checkpoint
    import pandas as pd
    torch.set_num_threads(1)
    job = Job(experiment='E3', datasets=('pneumoniamnist',), method='FedQTrust',
              solver='sa', clients=3, k=2, rounds=4, epochs=1, batch=4,
              attack='gaussian', fraction=.34, attack_start=2)
    generator = torch.Generator().manual_seed(77)
    ds = TensorDataset(torch.rand(12, 1, 2, 2, generator=generator), torch.tensor([0, 1] * 6))
    data = ([dict(dataset='pneumoniamnist', train=ds, val=ds, n=12) for _ in range(3)],
            {'pneumoniamnist': ds}, {'pneumoniamnist': ds}, [])
    reference, interrupted = tmp_path / 'reference', tmp_path / 'interrupted'
    run_job(job, reference, infrastructure='development', data=data, model_factory=Tiny)
    original = checkpoint.save_round

    def stop_after_checkpoint(path, **record):
        original(path, **record)
        if record['round_index'] == 2:
            raise RuntimeError('Simulated interruption')

    monkeypatch.setattr(checkpoint, 'save_round', stop_after_checkpoint)
    with pytest.raises(RuntimeError, match='Simulated interruption'):
        run_job(job, interrupted, infrastructure='development', data=data, model_factory=Tiny)
    assert not (interrupted / 'DONE').exists()
    monkeypatch.setattr(checkpoint, 'save_round', original)
    run_job(job, interrupted, infrastructure='development', data=data, model_factory=Tiny)
    expected = torch.load(reference / 'final_model.pt', weights_only=True)
    actual = torch.load(interrupted / 'final_model.pt', weights_only=True)
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    columns = ['round', 'accuracy', 'loss', 'gradient_proxy']
    pd.testing.assert_frame_equal(pd.read_csv(reference / 'rounds.csv')[columns],
                                  pd.read_csv(interrupted / 'rounds.csv')[columns])
    assert json.loads((reference / 'selections.json').read_text()) == json.loads((interrupted / 'selections.json').read_text())


def test_settings_sharding_status_and_export(tmp_path):
    from fedqtrust.manuscript.cli import parse_args, planned_jobs, persist_plan
    from fedqtrust.manuscript.artifacts import status, export_results
    from zipfile import ZipFile
    config = tmp_path / 'settings.json'
    config.write_text(json.dumps(dict(device='cuda', experiments='E9', output_dir=str(tmp_path / 'run'))))
    args = parse_args(['plan', '--config', str(config), '--device', 'cpu'])
    assert args.device == 'cpu'
    experiments, jobs, plan = planned_jobs(args)
    shards = [{job.id for i, job in enumerate(jobs) if i % 3 == shard} for shard in range(3)]
    assert set.union(*shards) == {job.id for job in jobs}
    assert sum(len(shard) for shard in shards) == len(jobs)
    root = tmp_path / 'run'
    persist_plan(root, plan)
    job_root = root / 'jobs' / jobs[0].id
    job_root.mkdir(parents=True)
    (job_root / 'rounds.csv').write_text('round,accuracy\n1,0.5\n')
    (job_root / 'checkpoint.pt').write_bytes(b'large checkpoint')
    assert status(root)['partial_jobs'][0]['last_logged_round'] == 1
    assert not status(root)['training_complete']
    archive = export_results(root)
    with ZipFile(archive) as result:
        assert 'export_status.json' in result.namelist()
        assert not any(name.endswith('.pt') for name in result.namelist())
        assert 'plan.json' in result.namelist()


def test_local_crypto_execution_never_contacts_fabric(tmp_path, monkeypatch):
    import fedqtrust.manuscript.engine as engine
    import pandas as pd
    from fedqtrust.manuscript.cli import parse_args
    assert parse_args(['plan', '--infrastructure', 'local']).infrastructure == 'local'

    def forbidden_fabric(*args, **kwargs):
        raise AssertionError('Local mode must not contact Fabric')

    monkeypatch.setattr(engine, 'HyperledgerFabricBlockchain', forbidden_fabric)
    torch.set_num_threads(1)
    # No-PQC exercises the local path with real RSA/AES-GCM/ECDSA on this machine.
    job = Job(experiment='E3', datasets=('pneumoniamnist',), method='No-PQC',
              solver='sa', clients=3, k=2, rounds=3, epochs=1, batch=4)
    ds = TensorDataset(torch.rand(12, 1, 2, 2), torch.tensor([0, 1] * 6))
    data = ([dict(dataset='pneumoniamnist', train=ds, val=ds, n=12) for _ in range(3)],
            {'pneumoniamnist': ds}, {'pneumoniamnist': ds}, [])
    output = tmp_path / 'jobs' / job.id
    run_job(job, output, infrastructure='local', data=data, model_factory=Tiny)
    metadata = json.loads((output / 'metadata.json').read_text())
    assert metadata['cryptography_mode'] == 'classical'
    assert metadata['ledger_backend'] == 'local_memory'
    assert not metadata['fabric_network_measured']
    timing = pd.read_csv(output / 'timing.csv')
    assert (timing.signature_bytes > 0).all()
    save_json(tmp_path / 'plan.json', {'experiments': ['E3'], 'jobs': [dict(id=job.id, **asdict(job))]})
    build_report(tmp_path)
    audit = json.loads((tmp_path / 'audit.json').read_text())
    assert audit['all_jobs_complete']
    assert not audit['fabric_network_measured']
    assert not audit['publication_ready']
    assert 'LOCAL LEDGER RUN' in (tmp_path / 'main.tex').read_text()


def test_torch_statevector_matches_qiskit_estimatorqnn():
    torch.manual_seed(1); np.random.seed(1)
    fast = QuantumPatchLayer(backend='torch_statevector').double()
    ref = QuantumPatchLayer(backend='qiskit')
    w = torch.tensor(np.random.uniform(-2, 2, 12))
    with torch.no_grad():
        fast.weight.copy_(w); ref.connector.weight.copy_(w)
    x = torch.rand(2, 1, 4, 4, dtype=torch.float64)
    a, b = fast(x), ref(x.float())
    assert torch.allclose(a, b.double(), atol=1e-5)
    a.sum().backward(); b.sum().backward()
    assert torch.allclose(fast.weight.grad, ref.connector.weight.grad.double(), atol=1e-4)


def test_lie_is_not_a_noop_with_one_attacker_among_five():
    from fedqtrust.attacks.lie import adaptive_lie
    rng = np.random.default_rng(0)
    honest = [rng.normal(size=50) for _ in range(4)]
    vector, z, _, _ = adaptive_lie(honest, 1, z_cap=1.0)
    assert z > 0 and not np.allclose(vector, np.mean(honest, axis=0))


def test_malicious_clients_span_all_modalities():
    job = Job(attack='lie', fraction=.2)
    bad, _ = malicious_sets(job)
    datasets = job.datasets
    # prepare_data interleaves: client c belongs to datasets[c % D]
    assert {datasets[c % len(datasets)] for c in bad} == set(datasets)


def test_trust_uses_previous_round_accuracy():
    from fedqtrust.fl.trust import TrustManager
    t = TrustManager(1)
    tau = t.update_round(0, 0.0, 1.0)  # Acc^(t-1)=0.5 initial, Rep=0.9*0.5+0.1*0.65
    assert tau == pytest.approx(0.4 * 0.5 + 0.3 * 1.0 + 0.3 * (0.45 + 0.065))
    assert t.accuracy[0] == 0.0


def test_label_flip_is_fixed_across_rounds():
    from fedqtrust.manuscript.engine import FlippedLabels
    ds = TensorDataset(torch.rand(50, 1, 2, 2), torch.zeros(50, 1, dtype=torch.long))
    a, b = FlippedLabels(ds, 5, 1.0, 7), FlippedLabels(ds, 5, 1.0, 7)
    assert [int(a[i][1]) for i in range(50)] == [int(b[i][1]) for i in range(50)]
    assert any(int(a[i][1]) != 0 for i in range(50))


def test_qaoa_returns_most_probable_and_expected_objective():
    instance = QuboInstance(np.array([.9, .2, .5]), np.zeros((3, 3)), 1, 2, 1, [0, 1, 2])
    s = solve_qaoa(instance, reps=1, shots=128, maxiter=8)
    assert np.isfinite(s.expected_objective) and s.best_sampled_objective <= s.objective + 1e-9


def test_warmup_covers_every_client_and_qubo_is_capped():
    import numpy as np
    from fedqtrust.manuscript.engine import select_clients, QUBO_MAX_ELIGIBLE
    from fedqtrust.manuscript.plan import Job
    from fedqtrust.fl.trust import TrustManager
    job = Job(datasets=('pneumoniamnist',), clients=40, k=5, solver='sa')
    trust = TrustManager(40)
    cache = [np.random.default_rng(i).normal(size=8) for i in range(40)]
    seen = set()
    for rnd in range(1, 9):
        ids, info = select_clients(job, trust, cache, rnd)
        assert info['solver'] == 'WARMUP_ROUND_ROBIN' and len(ids) == 5
        seen.update(ids)
    assert seen == set(range(40))
    ids, info = select_clients(job, trust, cache, 9)
    assert info['eligible_before_cap'] == 40 and info['eligible'] == QUBO_MAX_ELIGIBLE
    assert len(ids) == 5 and ids != [0, 1, 2, 3, 4]
