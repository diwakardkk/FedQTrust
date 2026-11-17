# FedQTrust Fabric Notes

Paper-strict experiments require Hyperledger Fabric. Use the official Fabric test-network, deploy chaincode named `fedqtrust`, and expose these operations:

- `Ping`
- `SetTrust(client_id, value)`
- `GetTrust(client_id)`
- `LogSelection(record_json)`
- `GetSelection(round_or_key)`

The Python adapter intentionally checks subprocess return codes and does not treat launched transactions as successful commits.


## Manuscript-v2 contract

The Node chaincode in `chaincode/` implements `Ping`, `SetTrust`, `GetTrust`,
`LogSelection`, `RegisterSigner`, and `VerifyRecord`. Deploy it on your configured
Fabric test network as `fedqtrust` before running manuscript preflight. Node
20.19+ is required by the pinned post-quantum verifier library. Install dependencies
inside `chaincode/` with `npm install` when packaging the chaincode.

Set `FEDQTRUST_FABRIC_CHANNEL` (default `mychannel`), the normal `CORE_PEER_*`
identity/TLS variables, and optionally `FEDQTRUST_FABRIC_INVOKE_ARGS` as a JSON
array of orderer/TLS CLI arguments. Invocations wait for commit events; a query
alone is not counted as commit evidence. The execution namespace isolates each
condition and retry. Run owners and signer keys are pinned, and updating existing
trust requires a verified, unused client signature proof.

The test ledger validates ECDSA signatures, ownership and replay rejection. A
real Fabric deployment and cross-library ML-DSA verification must be tested on
the server. No claim of a BFT network configuration is made by this code.
