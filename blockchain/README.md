# FedQTrust Fabric Notes

Paper-strict experiments require Hyperledger Fabric. Use the official Fabric test-network, deploy chaincode named `fedqtrust`, and expose these operations:

- `Ping`
- `SetTrust(client_id, value)`
- `GetTrust(client_id)`
- `LogSelection(record_json)`
- `GetSelection(round_or_key)`

The Python adapter intentionally checks subprocess return codes and does not treat launched transactions as successful commits.

