#!/usr/bin/env python3
"""Detached Linux/macOS supervisor: one independent experiment worker per GPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src') if (ROOT / 'src').is_dir() else str(ROOT))
from fedqtrust.manuscript.artifacts import save_json, status


def command(action, config, *extra):
    return [sys.executable, '-u', str(ROOT / 'run.py'), action, '--config', str(config), *extra]


def lock_run(output):
    output.mkdir(parents=True, exist_ok=True)
    handle = (output / '.server.lock').open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def supervise(args, output, settings):
    # The inherited descriptor retains the start command's exclusive lock.
    lock = os.fdopen(args.lock_fd, 'a')
    children = []
    handles = []
    state = {'state': 'RUNNING', 'pid': os.getpid(), 'started_unix': time.time(), 'gpus': args.gpus}
    state_path = output / 'server_state.json'
    save_json(state_path, state)
    environment = os.environ.copy()
    # Numeric GPU IDs follow PCI order on the target server.
    environment['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    environment.setdefault('OMP_NUM_THREADS', '4')
    environment.setdefault('MKL_NUM_THREADS', '4')
    devices = args.gpus.split(',')
    if settings['device'] == 'cuda':
        environment['CUDA_VISIBLE_DEVICES'] = devices[0]

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for action in ('plan', 'preflight', 'data'):
            subprocess.run(command(action, args.config), env=environment, check=True)
        with (output / 'environment.txt').open('w') as handle:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=handle, check=True)
        for index, gpu in enumerate(devices):
            worker_env = environment.copy()
            if settings['device'] == 'cuda':
                worker_env['CUDA_VISIBLE_DEVICES'] = gpu
            handle = (output / 'logs' / f'worker_{index}.log').open('a')
            handles.append(handle)
            child = subprocess.Popen(command('run', args.config, '--shards', str(len(devices)),
                                              '--shard-index', str(index), '--skip-benchmarks'),
                                     env=worker_env, stdout=handle, stderr=subprocess.STDOUT)
            children.append(child)
        state['workers'] = [child.pid for child in children]
        save_json(state_path, state)
        while any(child.poll() is None for child in children):
            failures = [child.returncode for child in children if child.returncode not in (None, 0)]
            if failures:
                raise RuntimeError(f'Worker failed with exit code {failures[0]}; inspect worker logs')
            time.sleep(1)
        if any(child.returncode for child in children):
            raise RuntimeError('A worker failed; inspect worker logs')
        for action in ('benchmarks', 'report'):
            subprocess.run(command(action, args.config), env=environment, check=True)
        progress = status(output)
        state['state'] = ('COMPLETE' if progress['training_complete'] and
                          progress['report_audit'].get('benchmarks_complete') else 'INCOMPLETE')
        state['completed_jobs'] = progress['completed_jobs']
        state['infrastructure'] = settings.get('infrastructure', 'paper')
        state['fabric_network_measured'] = progress['report_audit'].get('fabric_network_measured', False)
        save_json(state_path, state)
        subprocess.run(command('export', args.config), env=environment, check=True)
        return 0 if state['state'] == 'COMPLETE' else 2
    except KeyboardInterrupt:
        state.update(state='INTERRUPTED', error='Stopped; restart to resume the last saved round')
        return 130
    except Exception as exc:
        state.update(state='FAILED', error=str(exc))
        print(str(exc), file=sys.stderr, flush=True)
        return 1
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for handle in handles:
            handle.close()
        state['finished_unix'] = time.time()
        save_json(state_path, state)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'status', 'stop', '_supervise'])
    parser.add_argument('--config', default='settings.json')
    parser.add_argument('--gpus', default='0', help='Comma-separated GPU IDs; use cpu for CPU settings')
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    args.config = Path(args.config).resolve()
    settings = json.loads(args.config.read_text())
    output = Path(settings['output_dir']).resolve()
    if args.action == '_supervise':
        return supervise(args, output, settings)
    lock = lock_run(output)
    state_path = output / 'server_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'state': 'NOT_STARTED'}
    if args.action == 'status':
        print(json.dumps({'supervisor_active': lock is None, 'last_supervisor': state,
                          'progress': status(output)}, indent=2))
        if lock:
            lock.close()
        return 0
    if args.action == 'stop':
        if lock:
            lock.close()
            print('No active supervisor.')
        elif state.get('pid'):
            os.killpg(state['pid'], signal.SIGTERM)
            print('Stop requested. Restart with the same settings to resume.')
        else:
            raise RuntimeError('Supervisor is starting; retry status shortly.')
        return 0
    if lock is None:
        raise RuntimeError('A supervisor is already running for this output directory.')
    devices = args.gpus.split(',')
    if any(not value for value in devices):
        raise ValueError('GPU IDs must be nonempty (repeat an ID to run several workers on one GPU)')
    if settings['device'] == 'cpu' and args.gpus != 'cpu':
        raise ValueError('CPU settings require --gpus cpu')
    if settings['device'] == 'cuda' and any(not value.isdigit() for value in devices):
        raise ValueError('CUDA GPU IDs must be integers such as 0,1')
    (output / 'logs').mkdir(parents=True, exist_ok=True)
    with (output / 'logs' / 'server.log').open('a') as log:
        child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '_supervise',
                                  '--config', str(args.config), '--gpus', args.gpus,
                                  '--lock-fd', str(lock.fileno())],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, pass_fds=(lock.fileno(),))
    lock.close()
    print(f'Started supervisor PID {child.pid}. Check python server.py status and {output / "logs/server.log"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
