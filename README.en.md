# DGX Spark Controller Collection v3.3.1

**22 ready-to-run controllers** for serving models on NVIDIA DGX Spark. This is not a script
generator — every file here is a finished controller that was written, hardened and used on real
hardware. Take one and run it.

Designed by **neronain** · <https://www.facebook.com/neronain.minidev>

> 🇹🇭 ภาษาไทย: [README.md](README.md) — the Thai README is the primary document.

Every controller follows the same contract: the same command set, overridable
context/port/bind/IP, an ASCII banner plus an `info` command, and nothing tied to the author's
username or IP addresses.

## Start here

```bash
chmod +x ./*.sh audit-controllers.py

# What model, which port, which features, is it up?
./qwen3-coder-next-single.sh info
```

```text
   ____   ____ __  __    ____                   _
  |  _ \ / ___|\ \/ /   / ___| _ __   __ _ _ __| | __
  | | | | |  _  \  /    \___ \| '_ \ / _` | '__| |/ /
  | |_| | |_| | /  \     ___) | |_) | (_| | |  |   <
  |____/ \____|/_/\_\   |____/| .__/ \__,_|_|  |_|\_\
                              |_|
       =[ DGX Spark Controller · v3.3.1 ]
+ -- --=[ Qwen3-Coder-Next (NVFP4-GB10) ]
+ -- --=[ vLLM (Docker) · code · tools (qwen3_coder) · 256K ctx ]

  Model     : Qwen3-Coder-Next (NVFP4-GB10)
  Runtime   : vLLM (Docker)
  API (v1)  : http://192.168.101.127:8000/v1
  State     : stopped  (port 8000)
```

## Standard commands

```text
info | banner    model / port / features / state — start here
start            start the server
stop | restart   stop / bring back up
status           process or container status plus API health
logs [N]         recent log lines
network-info     bind address and the endpoint advertised to clients
client-config    client-side settings including the token budget
download         fetch model weights (resumable, retries until complete)
help             every option this particular controller supports
```

Many controllers add model-specific commands — `verify-files`, `prepare-runtime`, `props`,
`bench`, `stress`, `test-text`, `test-reasoning`, `test-image`, `test-tools`, `test-tool-loop`,
`doctor`, `sync-worker`. Run `help` on the file to see what it has.

## Context and port

```bash
./controller.sh start --context 65536 --port 8001
```

All options:

```text
--context TOKENS      --bind ADDRESS          --client-input TOKENS|auto
--port PORT           --advertise-ip ADDRESS  --client-output TOKENS
                      --interface NAME
```

Environment variables work too (vLLM uses `MAX_MODEL_LEN`, llama.cpp uses `CTX_SIZE`):

```bash
MAX_MODEL_LEN=65536 API_PORT=8001 ./qwen3-coder-next-single.sh start
CTX_SIZE=65536      API_PORT=8001 ./qwen36-hauhau-q6kp-single.sh start
```

Invalid values are rejected: a port must be 1–65535 and context must be greater than zero.

## Running on your own machines

Nothing here is bound to the author's environment:

- **Username** — `SSH_USER` defaults to the current user (`${USER:-$(id -un)}`); paths come from
  `$HOME` and can be overridden with `USER_HOME`.
- **Cluster IPs** — stacked controllers **ask on `start` / `restart`** when run on a real terminal.
- **Port and context** — overridable everywhere, as above.

### Stacked: head / worker / SSH user

```text
== Cluster configuration (press Enter to keep the current value) ==
  Head (master) node IP [10.100.152.1]: 192.168.101.127
  Worker node IP        [10.100.152.2]: 192.168.101.128
  SSH user for nodes    [dgxuser]:
```

Only `start` and `restart` ask, and only when stdin is a TTY, so cron and automation never hang.
For unattended runs, pass values as environment variables and close stdin:

```bash
MASTER_IP=192.168.101.127 WORKER_IP=192.168.101.128 SSH_USER=dgxuser \
  ./deepseek-v4-flash-nvfp4-stacked.sh start </dev/null
```

## Fabric discovery on stacked controllers (v3.3.0)

`deepseek-v4-flash-nvfp4-stacked.sh` no longer needs the interface name typed in — it derives it
from the transport IP, and asks the worker for its own name over SSH:

```text
[09:45:02] Fabric interface for 10.100.152.1: enp1s0f1np1
[09:45:02] RoCE HCA for enp1s0f1np1: rocep1s0f1
```

This matters more than it looks. Port names on DGX Spark are long and differ per fabric —
`enp1s0f1np1` and `enP2p1s0f1np1` are two separate 200G fabrics on the same machine — and naming
the wrong one does not raise an error: NCCL quietly falls back to the management NIC. Likewise,
leaving `NCCL_IB_HCA` unset makes NCCL run over TCP, so a 200G fabric performs like ordinary
ethernet. Both failures still "work", which is exactly why they are hard to find.

An explicit setting always wins:

```bash
NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1 ./deepseek-v4-flash-nvfp4-stacked.sh start
```

The controller also refuses to start on a machine that does not own `MASTER_IP`, rather than
failing later inside NCCL initialisation.

## Bind versus advertise

`--bind` is the address the server actually listens on. `--advertise-ip` / `--interface` control
the URL handed to Hermes, OpenClaw, Cline and other clients. The usual choice:

```bash
./controller.sh start --bind 0.0.0.0 --interface enp1s0 --port 8000
```

Advertised-IP selection order: `--advertise-ip` → `--interface` → source address from
`ip route get 1.1.1.1` → a global IPv4 that is not a Docker/CNI/cluster interface → `hostname -I`
as a last resort.

## Client token budget

With `--client-input auto`:

```text
client input = server context − max output − 8192 overhead
```

A request where input + output exceeds the server context is rejected.

## Verifying the collection

```bash
./verify-all.sh
```

```text
Audited 22 scripts: errors=0, warnings=0
All 22 DGX controllers passed static validation.
```

`verify-all.sh` runs each controller: `bash -n`, `help`, `info` (banner plus
Model/Runtime/Features/State), `network-info`, `client-config`, rejection of port 70000 and of
context 0, and — for stacked ones — that `prompt_cluster_config` exists, does not fire on `info`,
and that no developer username is left hard-coded.

Audit any controllers already on a machine:

```bash
python3 audit-controllers.py "$HOME"
python3 audit-controllers.py "$HOME" --json
```

The audit covers: bash syntax · numeric separators · `pipefail` with `grep -q` · hard-coded
single-node `MASTER_IP` · non-overridable context/port · missing `--context` / `--port` · missing
`network-info` · picking an IP straight from `hostname -I` · missing `SCRIPT_VERSION` · missing
`banner()`/`info()` or the `info|banner)` dispatch entry · hard-coded developer usernames ·
stacked controllers without `prompt_cluster_config()`.

> **Numeric separators**: bash arithmetic cannot parse `25_000_000_000` — it fails with
> "value too great for base". No controller here has one left.

## Installing onto a target machine

```bash
./install-canonical.sh "$HOME"
```

Existing files of the same name are backed up to `<target>/controller-backups/<timestamp>/` first,
and running models are never stopped or restarted.

## Validation scope

Everything above is **static** validation performed on a workstation, plus `verify-all.sh` on a
real DGX Spark. **Hardware-tested** status applies only to the models the author has actually run
and confirmed on DGX Spark hardware — a release does not re-open every model.

## Other documents

```text
README.md                        Thai documentation (primary)
ADDING-GENERATED-CONTROLLERS.md  adding an LMDS-generated controller to this collection
AUDIT_REPORT.md                  audit results and what this release fixed
CHANGELOG.md                     release history
step.md                          DeepSeek-V4-Flash two-node runbook (hardware-validated)
skills_Strack.md                 playbook for building stacked controllers
MANIFEST.txt                     file listing
PACKAGE_SHA256SUMS               SHA-256 of every file
```
