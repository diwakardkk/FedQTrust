"""Exercise detached orchestration with tiny subprocess fixtures, without GPUs."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest


@pytest.mark.parametrize('fail_preflight', [False, True])
def test_supervisor_completion_and_failure(tmp_path, fail_preflight):
    root = Path(__file__).resolve().parents[1]
    shutil.copy(root / 'server.py', tmp_path / 'server.py')
    from fedqtrust.manuscript import artifacts
    package = tmp_path / 'fedqtrust/manuscript'
    package.mkdir(parents=True)
    shutil.copy(artifacts.__file__, package / 'artifacts.py')
    output = tmp_path / 'output'
    settings = {'output_dir': str(output), 'device': 'cuda', 'fail_preflight': fail_preflight}
    (tmp_path / 'settings.json').write_text(json.dumps(settings))
    (tmp_path / 'run.py').write_text('''
import argparse, json, os
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('action')
p.add_argument('--config')
p.add_argument('--shards', type=int, default=1)
p.add_argument('--shard-index', type=int, default=0)
p.add_argument('--skip-benchmarks', action='store_true')
a = p.parse_args()
assert os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
s = json.loads(Path(a.config).read_text())
o = Path(s['output_dir'])
if a.action == 'plan':
    (o/'plan.json').write_text(json.dumps({'jobs': [{'id': str(i), 'rounds': 1} for i in range(2)]}))
if a.action == 'preflight' and s['fail_preflight']:
    raise SystemExit(7)
if a.action == 'run':
    for i in range(2):
        if i % a.shards == a.shard_index:
            d = o/'jobs'/str(i)
            d.mkdir(parents=True)
            (d/'DONE').write_text('fixture only')
if a.action == 'report':
    (o/'audit.json').write_text(json.dumps({'benchmarks_complete': True, 'publication_ready': False}))
if a.action == 'export':
    (o/'export_fixture.txt').write_text('export reached')
''')
    subprocess.run([sys.executable, str(tmp_path / 'server.py'), 'start', '--gpus', '0,1'],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        result = subprocess.run([sys.executable, str(tmp_path / 'server.py'), 'status'],
                                cwd=tmp_path, check=True, capture_output=True, text=True)
        progress = json.loads(result.stdout)
        if not progress['supervisor_active']:
            break
        time.sleep(.1)
    else:
        subprocess.run([sys.executable, str(tmp_path / 'server.py'), 'stop'], cwd=tmp_path)
        pytest.fail('Supervisor did not finish')
    expected = 'FAILED' if fail_preflight else 'COMPLETE'
    assert progress['last_supervisor']['state'] == expected
    assert (output / 'export_fixture.txt').exists() is not fail_preflight
    assert progress['progress']['completed_jobs'] == (0 if fail_preflight else 2)
