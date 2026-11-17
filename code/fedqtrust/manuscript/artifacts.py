"""Lightweight progress and result export, usable without scientific dependencies."""
import json
import os
from pathlib import Path
import uuid
from zipfile import ZIP_DEFLATED, ZipFile

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)

def status(output):
    output = Path(output)
    plan_path = output / 'plan.json'
    if not plan_path.exists():
        return {'state': 'NOT_PLANNED', 'output': str(output)}
    plan = json.loads(plan_path.read_text())
    complete = []
    partial = []
    for job in plan['jobs']:
        directory = output / 'jobs' / job['id']
        if (directory / 'DONE').exists():
            complete.append(job['id'])
        elif (directory / 'rounds.csv').exists():
            import csv
            with (directory / 'rounds.csv').open() as handle:
                last = max((int(row['round']) for row in csv.DictReader(handle)), default=0)
            partial.append({'id': job['id'], 'last_logged_round': last, 'rounds': job['rounds']})
    audit_path = output / 'audit.json'
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    return {
        'training_complete': len(complete) == len(plan['jobs']),
        'completed_jobs': len(complete),
        'total_jobs': len(plan['jobs']),
        'partial_jobs': partial,
        'report_audit': audit
    }

def export_results(output, destination=None, include_checkpoints=False):
    output = Path(output).resolve()
    if not (output / 'plan.json').exists():
        raise ValueError('No experiment plan found; nothing to export.')
    destination = Path(destination or output.parent / (output.name + '_results.zip')).resolve()
    if destination.is_relative_to(output):
        raise ValueError('Place the archive outside the result directory.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.tmp')
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if not path.is_file() or path.is_symlink():
                continue
            if path.name.startswith('.') or path.suffix == '.tmp':
                continue
            if path.suffix == '.pt' and (not include_checkpoints):
                continue
            archive.write(path, path.relative_to(output))
        archive.writestr('export_status.json', json.dumps(status(output), indent=2))
    os.replace(temporary, destination)
    return destination
