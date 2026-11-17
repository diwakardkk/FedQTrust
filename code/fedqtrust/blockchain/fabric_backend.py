"""Fabric CLI adapter with commit confirmation and namespaced experiment records."""
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass

@dataclass
class FabricResult:
    stdout: str
    stderr: str
    returncode: int
    latency_s: float

class HyperledgerFabricBlockchain:

    def __init__(self, chaincode='fedqtrust', namespace='default', channel=None):
        if shutil.which('peer') is None:
            raise RuntimeError('Hyperledger Fabric peer CLI is not available')
        self.chaincode = chaincode
        self.namespace = namespace
        self.verified = {}
        self.channel = channel or os.environ.get('FEDQTRUST_FABRIC_CHANNEL', 'mychannel')

    def _peer(self, args):
        start = time.perf_counter()
        p = subprocess.run(['peer', *args], capture_output=True, text=True, timeout=120)
        return FabricResult(p.stdout, p.stderr, p.returncode, time.perf_counter() - start)

    def _call(self, fn, *args, write=False):
        cmd = [
            'chaincode',
            'invoke' if write else 'query',
            '-C',
            self.channel,
            '-n',
            self.chaincode,
            '-c',
            json.dumps({'Args': [fn, *map(str, args)]})
        ]
        if write:
            cmd += ['--waitForEvent', '--waitForEventTimeout', '90s']
            cmd += json.loads(os.environ.get('FEDQTRUST_FABRIC_INVOKE_ARGS', '[]'))
        r = self._peer(cmd)
        if r.returncode:
            raise RuntimeError('Fabric transaction failed: ' + r.stderr)
        return r

    def ping(self):
        return self._call('Ping').stdout.strip() == 'pong'

    def set_trust(self, client_id, value):
        self._call(
            'SetTrust',
            self.namespace,
            client_id,
            value,
            self.verified.pop(str(client_id), ''),
            write=True
        )

    def get_trust(self, client_id):
        raw = self._call('GetTrust', self.namespace, client_id).stdout.strip()
        return float(raw) if raw else None

    def log_selection(self, record):
        self._call('LogSelection', self.namespace, json.dumps(record, separators=(',', ':')), write=True)

    def verify_record(self, proof):
        self._call(
            'RegisterSigner',
            self.namespace,
            proof['signer'],
            proof['public_key'],
            proof.get('scheme', 'ML-DSA-65'),
            write=True
        )
        self._call(
            'VerifyRecord',
            self.namespace,
            proof['signer'],
            proof['digest'],
            proof['signature'],
            write=True
        )
        self.verified[proof['signer']] = proof['digest']
