# Changelog

## 3.0.0

- Audited 12 canonical DGX Spark controllers.
- Removed Bash pure-numeric underscore literals.
- Standardized `--context`, `--port`, `--bind`, `--advertise-ip`, and `--interface`.
- Added environment overrides for context, port, bind address, advertise address, API key, and client budgets.
- Added route-based advertised IP selection.
- Removed fixed `MASTER_IP` from single-node controllers.
- Kept cluster addresses separate from public API addresses in stacked controllers.
- Added `network-info`.
- Added automatic client input-budget calculation.
- Fixed pipefail-sensitive `grep -q` pipelines.
- Added audit, verification, and safe installation helpers.
- Updated the DGX Spark Model Deployer skill rules and templates.
