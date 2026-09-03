# DGX Spark Model Deployer Skill v3.1.0

Designed by neronain · <https://www.facebook.com/neronain.minidev>

This repository ships controllers that are meant to be **used directly** on a DGX Spark, not generated per model. The skill rules below describe what every controller in the collection must satisfy.

Supported clients:

```text
Claude Code
OpenClaw
Hermes
Portable Agent Skills clients
```

## Mandatory safeguards (from 3.0.0)

```text
no Bash numeric underscore literals
exact GGUF size/header/hash validation
context and port runtime options
bind/advertise/cluster address separation
route-based advertised IP selection
pipefail-safe feature detection
```

## Added in 3.1.0

```text
SCRIPT_VERSION="${SCRIPT_VERSION:-3.1.0}" in every hand-written controller
  (LMDS-generated controllers carry the LMDS version instead — 0.5.1 as of 2026-09-03)
ASCII banner with a designer credit line
info | banner command: model, model ID, runtime, features, context,
  advertised API v1 URL, live state (probes /health) and port
port validation 1..65535 and context validation > 0
no hard-coded Linux username: SSH_USER defaults to ${USER:-$(id -un)}
stacked controllers prompt for head IP, worker IP, and SSH user on
  start/restart only, TTY-gated, Enter keeps the current value
prompt_cluster_config() must be called immediately after the
  MASTER_IP/WORKER_IP/SSH_USER declarations and before any variable
  derived from them
```

## Audit rules enforced

```text
bash-syntax
numeric-separator
pipefail-grep-q
fixed-single-master-ip
non-overridable-api-port
non-overridable-context
missing-context-option
missing-port-option
missing-network-selection
first-hostname-ip
missing-script-version
missing-banner-info
hard-coded-author-username
missing-cluster-prompt
```

Verify a whole directory of controllers:

```bash
python3 audit-controllers.py "$HOME"
./verify-all.sh
```

Expected result for this collection:

```text
Audited 42 scripts: errors=0, warnings=0
All 42 DGX controllers passed static validation.
```

See `skills_Strack.md` for the full stacked-deployment playbook and controller skeleton.
