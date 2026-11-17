#!/usr/bin/env python3
"""Download Fabric files on the Mac for transfer to the Linux GPU server."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent
VERSION = '2.5.16'
SOURCES = {
    'samples.tar.gz': 'https://codeload.github.com/hyperledger/fabric-samples/tar.gz/refs/heads/main',
    'linux-binaries.tar.gz': f'https://github.com/hyperledger/fabric/releases/download/v{VERSION}/hyperledger-fabric-linux-amd64-{VERSION}.tar.gz',
}


def download(url, destination):
    if destination.exists():
        with tarfile.open(destination, 'r:gz') as archive:
            archive.getmembers()
        print('Using cached ' + destination.name, flush=True)
        return
    temporary = destination.with_suffix('.download')
    subprocess.run(['curl', '--fail', '--location', '--retry', '5', '--retry-all-errors',
                    '--retry-delay', '3', '--connect-timeout', '20', '--max-time', '1800',
                    '--output', str(temporary), url], check=True)
    with tarfile.open(temporary, 'r:gz') as archive:
        archive.getmembers()
    temporary.replace(destination)


def assemble(cache, output):
    """Assemble only verified-readable input archives; never execute Linux binaries."""
    with tempfile.TemporaryDirectory(prefix='fabric-stage-', dir=cache) as temporary:
        stage = Path(temporary)
        with tarfile.open(cache / 'samples.tar.gz', 'r:gz') as archive:
            archive.extractall(stage, filter='data')
        candidates = [path for path in stage.iterdir() if path.is_dir()]
        if len(candidates) != 1 or not (candidates[0] / 'test-network/network.sh').is_file():
            raise ValueError('Unexpected fabric-samples archive layout')
        samples = stage / 'fabric-samples'
        if candidates[0] != samples:
            candidates[0].rename(samples)
        with tarfile.open(cache / 'linux-binaries.tar.gz', 'r:gz') as archive:
            archive.extractall(samples, filter='data')
        for name in ('bin/peer', 'bin/cryptogen', 'bin/configtxgen', 'bin/osnadmin', 'config/core.yaml'):
            if not (samples / name).is_file():
                raise ValueError('Missing required Fabric file: ' + name)
        provenance = {'fabric_version': VERSION, 'target': 'linux-amd64',
                      'sources': {name: {'url': url, 'sha256': hashlib.sha256((cache/name).read_bytes()).hexdigest()}
                                  for name, url in SOURCES.items()}}
        (samples / 'SOURCE_MANIFEST.json').write_text(json.dumps(provenance, indent=2))
        pending = output.with_suffix('.download')
        with tarfile.open(pending, 'w:gz') as archive:
            archive.add(samples, arcname='fabric-samples')
        pending.replace(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name + '.sha256').write_text(digest + '  ' + output.name + '\n')
    print(f'Ready: {output} ({output.stat().st_size / 1024**2:.1f} MiB)', flush=True)


def main():
    cache = ROOT / '.fabric-downloads'
    cache.mkdir(exist_ok=True)
    for name, url in SOURCES.items():
        download(url, cache / name)
    assemble(cache, ROOT / 'FedQTrust_Fabric_files.tar.gz')


if __name__ == '__main__':
    main()
