#!/usr/bin/env bash
# Provision a dedicated local Fabric test network for the FedQTrust runner.
# Run as the experiment user; only Docker and missing OS prerequisites use sudo.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FABRIC_WORK="$PROJECT_DIR/.fabric-local"
FABRIC_VERSION=2.5.16
CA_VERSION=1.5.17
PYTHON="$PROJECT_DIR/.venv/bin/python"
[[ -x "$PYTHON" ]] || { echo 'Activate/install the project .venv first.' >&2; exit 1; }
[[ $(uname -s) == Linux ]] || { echo 'Run this script on the Linux server.' >&2; exit 1; }
mkdir -p "$FABRIC_WORK"
exec 9>"$FABRIC_WORK/setup.lock"
flock -n 9 || { echo 'Fabric setup is already running.' >&2; exit 1; }
exec > >(tee -a "$FABRIC_WORK/setup.log") 2>&1
trap 'echo "Setup stopped at line $LINENO. See .fabric-local/setup.log; existing data was retained." >&2' ERR

missing=()
for program in curl git jq; do
  command -v "$program" >/dev/null || missing+=("$program")
done
if (( ${#missing[@]} )); then
  sudo apt-get update
  sudo apt-get install -y "${missing[@]}"
fi
DOCKER_REAL="$(command -v docker)"
mkdir -p "$FABRIC_WORK/wrappers"
if "$DOCKER_REAL" info >/dev/null 2>&1; then
  printf '#!/usr/bin/env bash\nexec %q "$@"\n' "$DOCKER_REAL" > "$FABRIC_WORK/wrappers/docker"
else
  sudo -v
  sudo "$DOCKER_REAL" info >/dev/null
  printf '#!/usr/bin/env bash\nexec sudo %q "$@"\n' "$DOCKER_REAL" > "$FABRIC_WORK/wrappers/docker"
fi
chmod 700 "$FABRIC_WORK/wrappers/docker"
export PATH="$FABRIC_WORK/wrappers:$PATH"
export CONTAINER_CLI=docker
export CONTAINER_CLI_COMPOSE='docker compose'
export COMPOSE_PROJECT_NAME=fedqtrust
export DOCKER_SOCK=/var/run/docker.sock
docker compose version

# Never reuse an unrelated sample network's containers or named resources.
if [[ ! -f "$FABRIC_WORK/ownership.json" ]]; then
  docker ps -a --format '{{.Names}}' > "$FABRIC_WORK/existing-containers.txt"
  docker network ls --format '{{.Name}}' > "$FABRIC_WORK/existing-networks.txt"
  docker volume ls --format '{{.Name}}' > "$FABRIC_WORK/existing-volumes.txt"
  "$PYTHON" - "$FABRIC_WORK" <<'PY'
import json, socket, sys
from pathlib import Path
root = Path(sys.argv[1])
names = set((root/'existing-containers.txt').read_text().splitlines())
reserved = {'orderer.example.com', 'peer0.org1.example.com', 'peer0.org2.example.com'}
if names & reserved:
    raise SystemExit('Existing Fabric containers conflict with this setup: ' + ', '.join(names & reserved))
if 'fedqtrust_test' in (root/'existing-networks.txt').read_text().splitlines():
    raise SystemExit('An unrelated fedqtrust_test network already exists.')
if any(name.startswith('fedqtrust_') for name in (root/'existing-volumes.txt').read_text().splitlines()):
    raise SystemExit('Existing fedqtrust Docker volumes require inspection before setup.')
for port in (7050, 7051, 7053, 9051, 9443, 9444, 9445):
    with socket.socket() as sock:
        try:
            sock.bind(('127.0.0.1', port))
        except OSError:
            raise SystemExit(f'Port {port} is already occupied; no network has been started.')
(root/'ownership.json').write_text(json.dumps({'purpose': 'FedQTrust local test network', 'root': str(root)}))
PY
fi

cd "$FABRIC_WORK"
# A Mac-prepared bundle avoids GitHub access from the GPU server.
if [[ -f "$PROJECT_DIR/FedQTrust_Fabric_files.tar.gz" && ! -f fabric-samples/config/core.yaml ]]; then
  "$PYTHON" - "$PROJECT_DIR/FedQTrust_Fabric_files.tar.gz" "$FABRIC_WORK" <<'PYBUNDLE'
import sys, tarfile
from pathlib import Path
archive, target = map(Path, sys.argv[1:])
if (target/'fabric-samples/test-network/organizations/peerOrganizations').exists():
    raise SystemExit('An existing network needs inspection before replacing missing configuration.')
with tarfile.open(archive, 'r:gz') as bundle:
    if any(not member.name.startswith('fabric-samples/') and member.name != 'fabric-samples'
           for member in bundle.getmembers()):
        raise SystemExit('Unexpected bundle contents')
    bundle.extractall(target, filter='data')
PYBUNDLE
fi
if [[ ! -x fabric-samples/bin/peer || ! -f fabric-samples/config/core.yaml ]]; then
  # Save the installer only after a complete download and syntax check.
  if [[ ! -s install-fabric.sh ]] || ! bash -n install-fabric.sh; then
    curl --ipv4 --http1.1 --fail --location --retry 5 --retry-all-errors \
      --retry-delay 3 --connect-timeout 20 --max-time 180 \
      https://raw.githubusercontent.com/hyperledger/fabric/main/scripts/install-fabric.sh \
      -o install-fabric.sh.download
    [[ -s install-fabric.sh.download ]]
    bash -n install-fabric.sh.download
    mv install-fabric.sh.download install-fabric.sh
  fi
  bash install-fabric.sh --fabric-version "$FABRIC_VERSION" --ca-version "$CA_VERSION" samples binary
fi
SAMPLES="$FABRIC_WORK/fabric-samples"
[[ -x "$SAMPLES/bin/peer" && -f "$SAMPLES/config/core.yaml" ]]
export FABRIC_CFG_PATH="$SAMPLES/config"
export PATH="$SAMPLES/bin:$PATH"
peer version
for component in peer orderer ccenv baseos; do
  image="ghcr.io/hyperledger/fabric-${component}:${FABRIC_VERSION}"
  docker image inspect "$image" >/dev/null 2>&1 || docker pull "$image"
done

# Fabric's Node platform builds with npm and starts in /usr/local/src.
# Node 22 supplies the >=20.19 runtime required by the PQ verifier.
mkdir -p "$FABRIC_WORK/node-runtime"
cat > "$FABRIC_WORK/node-runtime/Dockerfile" <<'DOCKERFILE'
FROM node:22-bookworm-slim
WORKDIR /usr/local/src
DOCKERFILE
if ! docker image inspect fedqtrust-nodeenv:22 >/dev/null 2>&1; then
  docker build -t fedqtrust-nodeenv:22 "$FABRIC_WORK/node-runtime"
fi
docker run --rm fedqtrust-nodeenv:22 node --version

"$PYTHON" - "$SAMPLES" "$FABRIC_VERSION" <<'PY'
from pathlib import Path
import sys, yaml
root, version = Path(sys.argv[1]), sys.argv[2]
network = root/'test-network'
# Pin images in scripts as well as compose files, including prerequisite probes.
for path in network.rglob('*'):
    if path.is_file() and path.suffix in ('.sh', '.yaml', '.yml'):
        text = path.read_text()
        for component in ('peer', 'orderer', 'ccenv', 'baseos'):
            text = text.replace(f'hyperledger/fabric-{component}:latest',
                                f'ghcr.io/hyperledger/fabric-{component}:{version}')
        text = text.replace('fabric_test', 'fedqtrust_test')
        path.write_text(text)
compose = network/'compose/compose-test-net.yaml'
spec = yaml.safe_load(compose.read_text())
for name, service in spec['services'].items():
    service['ports'] = [f'127.0.0.1:{port}' if len(str(port).split(':')) == 2 else port
                        for port in service.get('ports', [])]
    if name.startswith('peer0.'):
        overrides = {'CORE_CHAINCODE_NODE_RUNTIME': 'fedqtrust-nodeenv:22',
                     'CORE_CHAINCODE_INSTALLTIMEOUT': '600s'}
        environment = [entry for entry in service.get('environment', [])
                       if entry.split('=', 1)[0] not in overrides]
        service['environment'] = environment + [f'{key}={value}' for key, value in overrides.items()]
compose.write_text(yaml.safe_dump(spec, sort_keys=False))
PY

# Persist only the connection environment, never private-key contents.
"$PYTHON" - "$PROJECT_DIR" "$SAMPLES" <<'PY'
import json, shlex, sys
from pathlib import Path
project, samples = map(Path, sys.argv[1:])
orgs = samples/'test-network/organizations'
org1 = orgs/'peerOrganizations/org1.example.com'
org2 = orgs/'peerOrganizations/org2.example.com'
ca1 = org1/'peers/peer0.org1.example.com/tls/ca.crt'
ca2 = org2/'peers/peer0.org2.example.com/tls/ca.crt'
orderer_ca = orgs/'ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem'
values = {'FABRIC_CFG_PATH': str(samples/'config'), 'CORE_PEER_TLS_ENABLED': 'true',
          'CORE_PEER_LOCALMSPID': 'Org1MSP', 'CORE_PEER_ADDRESS': 'localhost:7051',
          'CORE_PEER_MSPCONFIGPATH': str(org1/'users/Admin@org1.example.com/msp'),
          'CORE_PEER_TLS_ROOTCERT_FILE': str(ca1), 'FEDQTRUST_FABRIC_CHANNEL': 'mychannel',
          'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
          'FEDQTRUST_FABRIC_INVOKE_ARGS': json.dumps([
              '-o', 'localhost:7050', '--ordererTLSHostnameOverride', 'orderer.example.com',
              '--tls', '--cafile', str(orderer_ca),
              '--peerAddresses', 'localhost:7051', '--tlsRootCertFiles', str(ca1),
              '--peerAddresses', 'localhost:9051', '--tlsRootCertFiles', str(ca2)])}
lines = ['# Generated connection settings for the local FedQTrust test network.']
lines.append('export PATH=' + shlex.quote(str(samples/'bin')) + ':"$PATH"')
lines.extend(f'export {key}={shlex.quote(value)}' for key, value in values.items())
(project/'fabric_env.sh').write_text('\n'.join(lines) + '\n')
PY

cd "$SAMPLES/test-network"
source "$PROJECT_DIR/fabric_env.sh"
if ! peer channel list 2>/dev/null | grep -qx mychannel; then
  bash network.sh up createChannel -c mychannel
fi
# network.sh runs in a child shell; restore the CLI's core.yaml directory explicitly.
source "$PROJECT_DIR/fabric_env.sh"
if ! peer lifecycle chaincode querycommitted -C mychannel -n fedqtrust >/dev/null 2>&1; then
  bash network.sh deployCC -c mychannel -ccn fedqtrust \
    -ccp "$PROJECT_DIR/blockchain/chaincode" -ccl javascript -ccv 1.0 -ccs 1
fi
source "$PROJECT_DIR/fabric_env.sh"

cd "$PROJECT_DIR"
# Validate actual commits, liboqs-to-Node ML-DSA interoperability and replay rejection.
"$PYTHON" - <<'PY'
import json, uuid
from pathlib import Path
from fedqtrust.blockchain.fabric_backend import HyperledgerFabricBlockchain
from fedqtrust.security.transport import Transport
chain = HyperledgerFabricBlockchain(namespace='setup_' + uuid.uuid4().hex)
assert chain.ping(), 'Fabric Ping failed'
assert chain._call('ProtocolVersion').stdout.strip() == 'fedqtrust-manuscript-v2'
chain.set_trust(0, .65)
transport = Transport('pqc')
payload = b'FedQTrust setup interoperability test'
received, _ = transport.roundtrip(payload, b'setup')
assert received == payload
proof = transport.authenticate(payload, signer_id='0')['proof']
chain.verify_record(proof)
chain.set_trust(0, .7)
assert abs(chain.get_trust(0) - .7) < 1e-9
try:
    chain._call('SetTrust', chain.namespace, 0, .8, proof['digest'], write=True)
except RuntimeError as exc:
    if 'replayed update proof' not in str(exc):
        raise
else:
    raise RuntimeError('Fabric accepted a replayed update')
Path('.fabric-local/integration_check.json').write_text(json.dumps({
    'ml_kem_roundtrip': True, 'fabric_ml_dsa_verification': True,
    'trust_commit': True, 'replay_rejected': True,
    'topology': 'two local peers; single Raft orderer; not a distributed BFT evaluation'
}, indent=2))
settings = json.loads(Path('settings.json').read_text())
provenance = Path(settings['output_dir'])/'infrastructure'
provenance.mkdir(parents=True, exist_ok=True)
(provenance/'fabric_deployment.json').write_text(Path('.fabric-local/integration_check.json').read_text())
print('PASS: Fabric ML-DSA verification, trust commit/readback, and replay rejection.')
PY
peer version > "$FABRIC_WORK/fabric-version.txt"
if [[ -d "$SAMPLES/.git" ]]; then
  git -C "$SAMPLES" rev-parse HEAD > "$FABRIC_WORK/samples-commit.txt"
else
  cp "$SAMPLES/SOURCE_MANIFEST.json" "$FABRIC_WORK/samples-source.json"
fi
docker image inspect "ghcr.io/hyperledger/fabric-peer:$FABRIC_VERSION" fedqtrust-nodeenv:22 > "$FABRIC_WORK/images.json"
printf '%s\n' 'SETUP COMPLETE. Next: source fabric_env.sh; run preflight; start server.py.'
