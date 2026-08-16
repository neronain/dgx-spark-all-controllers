#!/usr/bin/env bash
# DeepSeek-V4-Flash-NVFP4 — stacked 2×DGX Spark controller (reviewed v8.2 — permission-safe, test heredoc fixed)
# Runtime: ghcr.io/anemll/dspark-vllm-gx10:0.1.1
# Distributed: vLLM mp backend, TP=2, nnodes=2 (worker-first startup)
# Requires 200 Gbps RoCEv2 fabric between nodes.
set -Eeuo pipefail

SCRIPT_VERSION="${SCRIPT_VERSION:-3.1.0}"
MODEL_LABEL="${MODEL_LABEL:-DeepSeek-V4-Flash (NVFP4) · 2-node}"
RUNTIME_LABEL="${RUNTIME_LABEL:-vLLM (Docker, stacked)}"
MODEL_FEATURES="${MODEL_FEATURES:-reasoning · tools · tool-loop · 1M ctx}"

# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_ID="${MODEL_ID:-nvidia/DeepSeek-V4-Flash-NVFP4}"
MODEL_REVISION="${MODEL_REVISION:-main}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4-flash}"

# ─── Runtime ──────────────────────────────────────────────────────────────────
VLLM_IMAGE="${VLLM_IMAGE:-ghcr.io/anemll/dspark-vllm-gx10:0.1.1}"
MASTER_CONTAINER="${MASTER_CONTAINER:-vllm-ds4flash-head}"
WORKER_CONTAINER="${WORKER_CONTAINER:-vllm-ds4flash-worker}"

# ─── Cluster nodes ────────────────────────────────────────────────────────────
MASTER_IP="${MASTER_IP:-10.100.152.1}"
WORKER_IP="${WORKER_IP:-10.100.152.2}"
SSH_USER="${SSH_USER:-${USER:-$(id -un)}}"

# ── Interactive cluster config ────────────────────────────────────────────────
# Lets teammates/customers whose cluster IPs or Linux username differ from the
# author's supply their own values. Non-interactive shells keep the current
# values, so env overrides (MASTER_IP=… WORKER_IP=… SSH_USER=…) still work.
prompt_cluster_config() {
  [[ -t 0 ]] || return 0
  local ans
  printf '\n== Cluster configuration (press Enter to keep the current value) ==\n'
  read -rp "  Head (master) node IP [${MASTER_IP}]: " ans || true; [[ -z "$ans" ]] || MASTER_IP="$ans"
  read -rp "  Worker node IP        [${WORKER_IP}]: " ans || true; [[ -z "$ans" ]] || WORKER_IP="$ans"
  read -rp "  SSH user for nodes    [${SSH_USER}]: " ans || true; [[ -z "$ans" ]] || SSH_USER="$ans"
  printf '\n'
}
case "${1:-}" in start|restart) prompt_cluster_config ;; esac


# Fabric/transport IPs for NCCL (often a separate QSFP interface — override if needed)
TRANSPORT_IP_MASTER="${TRANSPORT_IP_MASTER:-$MASTER_IP}"
TRANSPORT_IP_WORKER="${TRANSPORT_IP_WORKER:-$WORKER_IP}"
MASTER_PORT="${MASTER_PORT:-25000}"

# NCCL/RoCE interface — MUST be set to the QSFP interface name on your hardware.
# Find it with: ip link show | grep -E '^[0-9]+:' and look for the 200G interface.
# Common names: enp1s0f0np0, rocep1s0f0, bond0, enp194s0np0
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"   # e.g. enp1s0f0np0
NCCL_IB_HCA="${NCCL_IB_HCA:-}"                 # e.g. rocep1s0f0
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"    # 3 for RoCEv2

# ─── Paths ────────────────────────────────────────────────────────────────────
USER_HOME="${USER_HOME:-$HOME}"
HF_HOME="${HF_HOME:-${USER_HOME}/.cache/huggingface}"
VLLM_CACHE="${VLLM_CACHE:-${USER_HOME}/.cache/vllm}"
FLASHINFER_CACHE="${FLASHINFER_CACHE:-${USER_HOME}/.cache/flashinfer}"
WORKER_HF_HOME="${WORKER_HF_HOME:-$HF_HOME}"
WORKER_VLLM_CACHE="${WORKER_VLLM_CACHE:-${VLLM_CACHE}}"
WORKER_FLASHINFER_CACHE="${WORKER_FLASHINFER_CACHE:-${FLASHINFER_CACHE}}"

# ─── Serving ──────────────────────────────────────────────────────────────────
API_PORT="${API_PORT:-8000}"
API_HOST="${API_HOST:-0.0.0.0}"
ADVERTISE_IP="${ADVERTISE_IP:-}"
ADVERTISE_INTERFACE="${ADVERTISE_INTERFACE:-}"
ROUTE_PROBE_IP="${ROUTE_PROBE_IP:-1.1.1.1}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-24}"
# DGX Spark (SM121): Marlin is the conservative backend. FlashInfer CUTLASS can
# be re-enabled explicitly after verifying the image/package/JIT-cache versions.
MOE_BACKEND="${MOE_BACKEND:-marlin}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-PIECEWISE}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-2400}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"
# Keep DSpark speculative decoding off by default. The observed 2026-07-18
# snapshot failed startup because config.json did not provide a parallel
# drafting token id required by DSpark/DFlash speculators.
ENABLE_DSPARK_SPECULATION="${ENABLE_DSPARK_SPECULATION:-false}"
SPECULATIVE_TOKENS="${SPECULATIVE_TOKENS:-0}"

# Model shard verification
SHARD_COUNT=46
TOTAL_SIZE_APPROX_GB=168

# ─── Helpers ──────────────────────────────────────────────────────────────────
die()      { echo "ERROR: $*" >&2; exit 1; }
log()      { echo "[$(date '+%H:%M:%S')] $*"; }
ssh_worker() { ssh -o BatchMode=yes -o ConnectTimeout=10 "${SSH_USER}@${WORKER_IP}" "$@"; }

_require_non_root() {
  if (( EUID == 0 )); then
    die "Do not run this controller with sudo/root. Run it as the normal user '${SUDO_USER:-$SSH_USER}'. Using sudo changes HOME and SSH_USER and can make the worker image check target root@${WORKER_IP}."
  fi
}

_ensure_local_owned_dir() {
  local dir="$1" parent uid gid
  uid=$(id -u); gid=$(id -g); parent=$(dirname "$dir")
  mkdir -p "$parent" || die "Cannot create parent directory: $parent"

  if [[ -e "$dir" && ! -w "$dir" ]]; then
    log "Repairing ownership of local cache: $dir"
    docker run --rm \
      -v "$dir:/repair" \
      --entrypoint sh "$VLLM_IMAGE" \
      -c "chown -R ${uid}:${gid} /repair" \
      || die "Unable to repair $dir. Run: sudo chown -R ${uid}:${gid} '$dir'"
  fi

  mkdir -p "$dir" || die "Cannot create directory: $dir"
  [[ -w "$dir" ]] || die "Directory is not writable: $dir. Run: sudo chown -R ${uid}:${gid} '$dir'"
}

_ensure_worker_owned_dir() {
  local dir="$1" parent remote_uid remote_gid
  parent=$(dirname "$dir")
  remote_uid=$(ssh_worker "id -u") || die "Cannot read worker UID via SSH"
  remote_gid=$(ssh_worker "id -g") || die "Cannot read worker GID via SSH"

  ssh_worker "mkdir -p '$parent'" || die "Cannot create worker parent directory: $parent"
  if ! ssh_worker "test -w '$dir'" 2>/dev/null; then
    if ssh_worker "test -e '$dir'" 2>/dev/null; then
      log "Repairing ownership of worker cache: $dir"
      ssh_worker "docker run --rm -v '$dir:/repair' --entrypoint sh '$VLLM_IMAGE' -c 'chown -R ${remote_uid}:${remote_gid} /repair'" \
        || die "Unable to repair worker directory $dir. On worker run: sudo chown -R ${remote_uid}:${remote_gid} '$dir'"
    fi
  fi
  ssh_worker "mkdir -p '$dir' && test -w '$dir'" \
    || die "Worker directory is not writable: $dir"
}

# ── Branding / info ───────────────────────────────────────────────────────────
banner() {
  cat <<'ART'

   ____   ____ __  __    ____                   _
  |  _ \ / ___|\ \/ /   / ___| _ __   __ _ _ __| | __
  | | | | |  _  \  /    \___ \| '_ \ / _` | '__| |/ /
  | |_| | |_| | /  \     ___) | |_) | (_| | |  |   <
  |____/ \____|/_/\_\   |____/| .__/ \__,_|_|  |_|\_\
                              |_|
ART
  printf '       =[ DGX Spark Controller · v%s ]\n' "${SCRIPT_VERSION}"
  printf '+ -- --=[ %s ]\n'   "${MODEL_LABEL}"
  printf '+ -- --=[ %s · %s ]\n' "${RUNTIME_LABEL}" "${MODEL_FEATURES}"
  printf '+ -- --=[ Designed by neronain · fb.com/neronain.minidev ]\n\n'
}

# Show which model this controller serves, its port, features, and whether it is up.
info() {
  banner
  local ip url state
  ip="$(detect_advertise_ip 2>/dev/null || true)"; [[ -n "$ip" ]] || ip="${API_HOST}"
  url="http://${ip}:${API_PORT}/v1"
  state="stopped"
  if curl -fsS -m 2 "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
    state="RUNNING"
  fi
  printf '  Model     : %s\n'            "${MODEL_LABEL}"
  printf '  Model ID  : %s\n'            "${MODEL_ID:-${HF_REPO:-${REPO_ID:-n/a}}}"
  printf '  Runtime   : %s\n'            "${RUNTIME_LABEL}"
  printf '  Features  : %s\n'            "${MODEL_FEATURES}"
  printf '  Context   : %s tokens\n'     "${MAX_MODEL_LEN:-${CTX_SIZE:-n/a}}"
  printf '  API (v1)  : %s\n'            "${url}"
  printf '  State     : %s  (port %s)\n\n' "${state}" "${API_PORT}"
}

detect_advertise_ip() {
  [[ -z "$ADVERTISE_IP" ]] || { printf '%s' "$ADVERTISE_IP"; return; }
  if [[ -n "$ADVERTISE_INTERFACE" ]]; then
    ip -4 -o addr show dev "$ADVERTISE_INTERFACE" scope global \
      | awk 'NR==1 { split($4,a,"/"); print a[1] }'
    return
  fi
  local src
  src=$(ip -4 route get "$ROUTE_PROBE_IP" 2>/dev/null \
    | awk '{ for(i=1;i<=NF;i++) if($i=="src"){ print $(i+1); exit } }')
  if [[ -n "$src" ]]; then
    printf '%s' "$src"; return
  fi
  ip -4 addr show scope global \
    | awk '/inet /{split($2,a,"/"); print a[1]}' \
    | grep -v '^127\.' | head -1
}

parse_options() {
  REMAINING_ARGS=()
  while (( $# )); do
    case "$1" in
      --context)       MAX_MODEL_LEN="$2";              shift 2 ;;
      --context=*)     MAX_MODEL_LEN="${1#*=}";          shift ;;
      --port)          API_PORT="$2";                   shift 2 ;;
      --port=*)        API_PORT="${1#*=}";               shift ;;
      --bind)          API_HOST="$2";                   shift 2 ;;
      --bind=*)        API_HOST="${1#*=}";               shift ;;
      --advertise-ip)  ADVERTISE_IP="$2";               shift 2 ;;
      --advertise-ip=*)ADVERTISE_IP="${1#*=}";           shift ;;
      --interface)     ADVERTISE_INTERFACE="$2";        shift 2 ;;
      --interface=*)   ADVERTISE_INTERFACE="${1#*=}";    shift ;;
      *)               REMAINING_ARGS+=("$1");           shift ;;
    esac
  done
  [[ "$MAX_MODEL_LEN" =~ ^[0-9]+$ ]] && (( MAX_MODEL_LEN > 0 )) || die "Invalid --context: ${MAX_MODEL_LEN}"
  [[ "$API_PORT" =~ ^[0-9]+$ ]] && (( API_PORT >= 1 && API_PORT <= 65535 )) || die "Invalid --port: ${API_PORT} (use 1..65535)"
  export MAX_MODEL_LEN API_HOST API_PORT ADVERTISE_IP ADVERTISE_INTERFACE
}

# ─── prepare-runtime ──────────────────────────────────────────────────────────
prepare_runtime() {
  _require_non_root
  mkdir -p "$HF_HOME"
  log "Pulling runtime image on head node …"
  docker pull "$VLLM_IMAGE"
  local head_id
  head_id=$(docker inspect "$VLLM_IMAGE" --format '{{.Id}}' 2>/dev/null)
  echo "$head_id" > "${HF_HOME}/.vllm-stacked-image-id"
  log "Image ID locked: $head_id"

  log "Pulling runtime image on worker node …"
  ssh_worker "docker pull '$VLLM_IMAGE'"
  local worker_id
  worker_id=$(ssh_worker "docker inspect '$VLLM_IMAGE' --format '{{.Id}}'")
  if [[ "$head_id" != "$worker_id" ]]; then
    die "Image digest mismatch between head ($head_id) and worker ($worker_id). Re-pull on both nodes."
  fi
  log "Runtime images match on both nodes. Lock ID: ${head_id:0:32}…"
}

runtime_info() {
  log "=== Runtime info ==="
  echo "Image     : $VLLM_IMAGE"
  echo "Backend   : mp (vLLM multiprocessing, TP=2, nnodes=2)"
  echo "Arch      : SM121 (GB10 Blackwell) — precompiled in Anemll image"
  local locked_id=""
  [[ -f "${HF_HOME}/.vllm-stacked-image-id" ]] && locked_id=$(cat "${HF_HOME}/.vllm-stacked-image-id")
  echo "Locked ID : ${locked_id:-not yet locked (run prepare-runtime)}"
  docker inspect "$VLLM_IMAGE" --format 'Current ID : {{.Id}}' 2>/dev/null || true
}

# ─── download ─────────────────────────────────────────────────────────────────
download() {
  log "Downloading ${MODEL_ID} revision=${MODEL_REVISION} …"
  log "Expected size ~${TOTAL_SIZE_APPROX_GB} GB (46 shards). Resume-safe via snapshot_download."
  mkdir -p "$HF_HOME"

  docker run --rm -i \
    --gpus all \
    --network host \
    --user "$(id -u):$(id -g)" \
    --entrypoint python3 \
    -v "${HF_HOME}:/cache" \
    -e HF_HOME=/cache \
    -e HOME=/tmp \
    -e "MODEL_ID=${MODEL_ID}" \
    -e "MODEL_REVISION=${MODEL_REVISION}" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    "$VLLM_IMAGE" \
    - <<'PYEOF'
from huggingface_hub import snapshot_download
import os

model_id = os.environ["MODEL_ID"]
revision = os.environ["MODEL_REVISION"]
hf_home = os.environ["HF_HOME"]

print(f"Downloading {model_id} @ {revision} to {hf_home}", flush=True)
path = snapshot_download(
    repo_id=model_id,
    revision=revision,
    cache_dir=hf_home,
    ignore_patterns=[
        "*.bak", "*.ckpt", "*.pt", "*.pth",
        "optimizer*", "scheduler*", "*calibration*",
    ],
)
print(f"Done: {path}", flush=True)
PYEOF

  log "Download complete."
}

# ─── verify helpers ───────────────────────────────────────────────────────────
_model_slug() {
  # Hugging Face cache layout uses double dashes between namespace and repo:
  # nvidia/DeepSeek-V4-Flash-NVFP4 -> models--nvidia--DeepSeek-V4-Flash-NVFP4
  printf '%s' "${MODEL_ID/\//--}"
}

_model_cache_dir() {
  local hf_root="${1:-$HF_HOME}"
  local slug; slug=$(_model_slug)
  local candidate
  for candidate in \
    "${hf_root}/models--${slug}" \
    "${hf_root}/hub/models--${slug}"; do
    if [[ -d "${candidate}/snapshots" ]]; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  # Prefer the modern layout for diagnostics when nothing exists yet.
  printf '%s/models--%s' "$hf_root" "$slug"
}

_snapshot_path() {
  local node="${1:-local}"   # "local" or "remote" retained for call compatibility
  local hf_root="${2:-$HF_HOME}"
  local model_dir ref_file commit snap
  model_dir=$(_model_cache_dir "$hf_root")
  if [[ -d "${model_dir}/snapshots/${MODEL_REVISION}" ]]; then
    printf '%s' "${model_dir}/snapshots/${MODEL_REVISION}"
    return 0
  fi
  ref_file="${model_dir}/refs/${MODEL_REVISION}"
  if [[ -f "$ref_file" ]]; then
    commit=$(tr -d '[:space:]' < "$ref_file")
    if [[ -n "$commit" && -d "${model_dir}/snapshots/${commit}" ]]; then
      printf '%s' "${model_dir}/snapshots/${commit}"
      return 0
    fi
  fi
  snap=$(find "${model_dir}/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null || true)
  if [[ -n "$snap" ]]; then
    printf '%s' "$snap"
    return 0
  fi
  printf '%s' "${model_dir}/snapshots/${MODEL_REVISION}"
}

_verify_small_files() {
  local snap="$1"
  log "Checking required config files …"
  for f in config.json generation_config.json hf_quant_config.json \
            tokenizer.json tokenizer_config.json; do
    [[ -f "${snap}/${f}" ]] || die "Missing required file: ${snap}/${f}"
  done
  log "Small files present."
}

_verify_shards_local() {
  local snap="$1"
  log "Verifying ${SHARD_COUNT} shards (size check, following symlinks) …"
  local n=0 fail=0
  while IFS= read -r -d '' st; do
    local fname; fname=$(basename "$st")
    # Check file is non-zero
    local sz; sz=$(stat -Lc '%s' "$st" 2>/dev/null || stat -f '%z' "$st")
    if (( sz < 100000000 )); then
      log "WARN: ${fname} looks too small (${sz} bytes) — may be incomplete"
      (( fail++ ))
    fi
    (( n++ ))
  done < <(find "$snap" -maxdepth 1 \( -name 'model-*.safetensors' -o -name 'model_mtp.safetensors' \) -print0 \
              2>/dev/null | sort -z)
  if (( n == 0 )); then
    die "No safetensors shards found under ${snap}"
  fi
  if (( n != SHARD_COUNT )); then
    die "Expected ${SHARD_COUNT} safetensors shards, found ${n} under ${snap}"
  fi
  if (( fail > 0 )); then
    die "${fail} shard(s) failed size check."
  fi
  log "Local shard check: ${n} shards present, all non-trivial."
}

_verify_dspark_speculation_config() {
  local snap="$1"
  [[ "$ENABLE_DSPARK_SPECULATION" == "true" && "${SPECULATIVE_TOKENS}" != "0" ]] || return 0

  log "Checking DSpark speculative decoding token id in model config ..."
  python3 - "$snap" <<'PYEOF'
import json, pathlib, sys

cfg_path = pathlib.Path(sys.argv[1]) / "config.json"
cfg = json.loads(cfg_path.read_text())

def find_key(obj, keys, path="config"):
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_path = f"{path}.{key}"
            if key in keys:
                return next_path, value
            found = find_key(value, keys, next_path)
            if found:
                return found
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            found = find_key(value, keys, f"{path}[{i}]")
            if found:
                return found
    return None

keys = {
    "mask_token_id",
    "dspark_noise_token_id",
    "pard_token",
    "ptd_token_id",
}
found = find_key(cfg, keys)
if not found:
    sys.exit(
        "ENABLE_DSPARK_SPECULATION=true is unsafe for this snapshot: "
        "config.json has no mask_token_id/dspark_noise_token_id/pard_token/ptd_token_id"
    )
print(f"DSpark speculative token marker found: {found[0]}={found[1]}")
PYEOF
}

verify_files() {
  local snap; snap=$(_snapshot_path local "$HF_HOME")
  [[ -d "$snap" ]] || die "Model snapshot not found at ${snap}. Run: $0 download"
  _verify_small_files "$snap"
  _verify_shards_local "$snap"
  _verify_dspark_speculation_config "$snap"
  log "verify-files: PASS"
}

# ─── sync-worker ──────────────────────────────────────────────────────────────
sync_worker() {
  local snap; snap=$(_snapshot_path local "$HF_HOME")
  [[ -d "$snap" ]] || die "Local snapshot not found. Run download first."

  local worker_hf="${WORKER_HF_HOME:-$HF_HOME}"
  local worker_snap; worker_snap=$(_snapshot_path remote "$worker_hf")
  local model_cache; model_cache=$(_model_cache_dir "$HF_HOME")
  local model_cache_parent; model_cache_parent=$(dirname "$model_cache")
  local model_cache_name; model_cache_name=$(basename "$model_cache")

  log "Syncing model cache to worker (${WORKER_IP}) …"
  log "Source : ${model_cache}/"
  log "Target : ${SSH_USER}@${WORKER_IP}:${worker_hf}/${model_cache_name}/"
  log "Estimated transfer: ~${TOTAL_SIZE_APPROX_GB} GB on first run. Progress shown below."

  ssh_worker "mkdir -p '${worker_hf}' '${WORKER_VLLM_CACHE}' '${WORKER_FLASHINFER_CACHE}'"
  rsync -avz --progress \
    --exclude='*.lock' --exclude='tmp_*' --exclude='trees/' \
    "${model_cache_parent}/${model_cache_name}/" \
    "${SSH_USER}@${WORKER_IP}:${worker_hf}/${model_cache_name}/"
  log "sync-worker: DONE"
}

# ─── verify-worker ────────────────────────────────────────────────────────────
verify_worker() {
  log "Verifying worker (${WORKER_IP}) shard integrity (full_hash_for_stacked=true) …"
  local worker_hf="${WORKER_HF_HOME:-$HF_HOME}"
  local slug; slug=$(_model_slug)

  # Run sha256 check inside worker via docker (avoids host python dependency)
  ssh_worker "docker run --rm \
    -v '${worker_hf}:/cache' \
    -e HF_HOME=/cache \
    -e SHARD_COUNT='${SHARD_COUNT}' \
    --entrypoint python3 \
    '$VLLM_IMAGE' \
    -c \"
import os, hashlib, pathlib, sys

model_dirs = [
    pathlib.Path('/cache/models--${slug}'),
    pathlib.Path('/cache/hub/models--${slug}'),
]
model_dir = next((p for p in model_dirs if (p / 'snapshots').exists()), model_dirs[0])
snap = model_dir / 'snapshots' / '${MODEL_REVISION}'
ref = model_dir / 'refs' / '${MODEL_REVISION}'
if not snap.exists() and ref.exists():
    commit = ref.read_text().strip()
    snap = model_dir / 'snapshots' / commit
if not snap.exists():
    snaps = sorted((model_dir / 'snapshots').glob('*')) if (model_dir / 'snapshots').exists() else []
    if snaps:
        snap = snaps[-1]
if not snap.exists():
    sys.exit(f'Snapshot missing: {snap}')

shards = sorted(snap.glob('model-*.safetensors')) + sorted(snap.glob('model_mtp.safetensors'))
if not shards:
    sys.exit('No shards found on worker')

print(f'Found {len(shards)} shards')
expected = int(os.environ.get('SHARD_COUNT', '${SHARD_COUNT}'))
if len(shards) != expected:
    sys.exit(f'Expected {expected} shards, found {len(shards)}')
fails = []
for s in shards:
    sz = s.stat().st_size
    if sz < 100000000:
        fails.append(f'{s.name}: size {sz} too small')
        continue
    h = hashlib.sha256()
    with open(s,'rb') as f:
        while chunk := f.read(8388608):
            h.update(chunk)
    print(f'OK {s.name}  {sz}  {h.hexdigest()}')

if fails:
    for e in fails: print('FAIL', e)
    sys.exit(f'{len(fails)} shard(s) failed')
print('verify-worker: PASS')
\"
"
  log "verify-worker: PASS"
}

# ─── _nccl_env_args ───────────────────────────────────────────────────────────
# ─── fabric interface discovery ───────────────────────────────
# Interface names on DGX Spark are long and differ per port (enp1s0f1np1 and
# enP2p1s0f1np1 are two separate fabrics on the same box). Typing the wrong one
# does not fail loudly: NCCL quietly falls back to the management NIC. Derive the
# name from the transport IP instead, and let an explicit setting win.
detect_interface_for_ip() {
  local target="$1"
  ip -o -4 addr show 2>/dev/null |
    awk -v want="$target" '$4 ~ ("^" want "/") {print $2; exit}'
}

detect_worker_interface() {
  ssh_worker "ip -o -4 addr show 2>/dev/null | awk -v want='${TRANSPORT_IP_WORKER}' \
    '\$4 ~ (\"^\" want \"/\") {print \$2; exit}'" 2>/dev/null || true
}

# The RoCE HCA that belongs to a NIC is discoverable: each InfiniBand device
# lists the netdev it is bound to. Without NCCL_IB_HCA, NCCL falls back to TCP
# and the 200G fabric performs like ordinary ethernet.
detect_hca_for_interface() {
  local want="$1" dev
  [[ -n "$want" ]] || return 0
  for dev in /sys/class/infiniband/*; do
    [[ -e "$dev/device/net/$want" ]] || continue
    basename "$dev"
    return 0
  done
  return 0
}

_resolve_nccl_hca() {
  [[ -n "$NCCL_IB_HCA" ]] && return 0
  NCCL_IB_HCA="$(detect_hca_for_interface "$NCCL_SOCKET_IFNAME")"
  [[ -n "$NCCL_IB_HCA" ]] &&
    log "RoCE HCA for ${NCCL_SOCKET_IFNAME}: ${NCCL_IB_HCA}"
  return 0
}

_resolve_nccl_ifname() {
  [[ -n "$NCCL_SOCKET_IFNAME" ]] && return 0
  NCCL_SOCKET_IFNAME="$(detect_interface_for_ip "$TRANSPORT_IP_MASTER")"
  [[ -n "$NCCL_SOCKET_IFNAME" ]] &&
    log "Fabric interface for ${TRANSPORT_IP_MASTER}: ${NCCL_SOCKET_IFNAME}"
  return 0
}

# Fail on the wrong host instead of dying inside NCCL init with an opaque error.
check_running_on_master() {
  local interface
  interface="$(detect_interface_for_ip "$TRANSPORT_IP_MASTER")"
  [[ -n "$interface" ]] ||
    die "This host does not own MASTER_IP=${TRANSPORT_IP_MASTER}. Run this controller on the head node (swap MASTER_IP/WORKER_IP if the roles are reversed)."
}

_nccl_env_args() {
  # Emit -e flags for docker run. Warns if interface not configured.
  local args=()
  args+=(-e "NCCL_IGNORE_CPU_AFFINITY=1")
  args+=(-e "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}")

  if [[ -n "$NCCL_SOCKET_IFNAME" ]]; then
    args+=(-e "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}")
    args+=(-e "GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}")
    args+=(-e "TP_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}")
    # UCX and OMPI choose their own transport; without these they can still land
    # on the management NIC while NCCL is correctly on the fabric.
    args+=(-e "UCX_NET_DEVICES=${NCCL_SOCKET_IFNAME}")
    args+=(-e "OMPI_MCA_btl_tcp_if_include=${NCCL_SOCKET_IFNAME}")
  else
    echo "WARN: no interface owns ${TRANSPORT_IP_MASTER}. NCCL may select the wrong one." >&2
    echo "WARN: Set NCCL_SOCKET_IFNAME to your QSFP/RoCE interface (e.g. enp1s0f0np0)" >&2
  fi

  if [[ -n "$NCCL_IB_HCA" ]]; then
    args+=(-e "NCCL_IB_HCA=${NCCL_IB_HCA}")
    args+=(-e "NCCL_IB_DISABLE=0")
  else
    echo "WARN: NCCL_IB_HCA not set. Falling back to TCP (slower). Set to your RoCE HCA." >&2
    args+=(-e "NCCL_IB_DISABLE=1")
  fi

  printf '%s\n' "${args[@]}"
}

# ─── runtime/cache safety helpers ─────────────────────────────────────────────
_current_image_id() {
  docker inspect "$VLLM_IMAGE" --format '{{.Id}}' 2>/dev/null || true
}

_assert_runtime_images() {
  local head_id worker_id locked_id=""
  head_id=$(_current_image_id)
  [[ -n "$head_id" ]] || die "Runtime image is not present locally: ${VLLM_IMAGE}. Run prepare-runtime."
  worker_id=$(ssh_worker "docker inspect '$VLLM_IMAGE' --format '{{.Id}}'" 2>/dev/null || true)
  [[ -n "$worker_id" ]] || die "Runtime image is not present on worker. Run prepare-runtime."
  [[ "$head_id" == "$worker_id" ]] || die "Runtime image mismatch: head=${head_id}, worker=${worker_id}"
  [[ -f "${HF_HOME}/.vllm-stacked-image-id" ]] && locked_id=$(cat "${HF_HOME}/.vllm-stacked-image-id")
  [[ -z "$locked_id" || "$locked_id" == "$head_id" ]] || \
    die "Current image differs from locked image. Run prepare-runtime again before start."
  printf '%s' "$head_id"
}

clear_flashinfer_cache() {
  _require_non_root
  log "Stopping deployment before clearing FlashInfer caches ..."
  stop
  _ensure_local_owned_dir "$FLASHINFER_CACHE"
  log "Clearing head FlashInfer cache: ${FLASHINFER_CACHE}"
  find "$FLASHINFER_CACHE" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +

  _ensure_worker_owned_dir "$WORKER_FLASHINFER_CACHE"
  log "Clearing worker FlashInfer cache: ${WORKER_FLASHINFER_CACHE}"
  ssh_worker "find '${WORKER_FLASHINFER_CACHE}' -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +"
  log "FlashInfer caches cleared on both nodes."
}

doctor() {
  _resolve_nccl_ifname
  _resolve_nccl_hca
  _require_non_root
  local image_id
  image_id=$(_assert_runtime_images)
  echo "=== Runtime compatibility ==="
  echo "Image: ${VLLM_IMAGE}"
  echo "ID   : ${image_id}"
  docker run --rm --gpus all --entrypoint python3 "$VLLM_IMAGE" - <<'PYEOF'
import inspect, os, platform
import torch, vllm, flashinfer
print("python     :", platform.python_version())
print("torch      :", torch.__version__)
print("cuda       :", torch.version.cuda)
print("gpu        :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable")
print("capability :", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "unavailable")
print("vllm       :", getattr(vllm, "__version__", "unknown"))
print("flashinfer :", getattr(flashinfer, "__version__", "unknown"))
try:
    from flashinfer.jit.env import FLASHINFER_CACHE_DIR
    print("fi cache   :", FLASHINFER_CACHE_DIR)
except Exception as e:
    print("fi cache   : unavailable", repr(e))
try:
    from flashinfer.fused_moe.core import MoERunner
    print("MoERunner  :", inspect.getsourcefile(MoERunner))
except Exception as e:
    print("MoERunner  : unavailable", repr(e))
PYEOF
}

# ─── start ────────────────────────────────────────────────────────────────────
start() {
  _require_non_root
  check_running_on_master
  _resolve_nccl_ifname
  _resolve_nccl_hca
  # The worker may name the same fabric differently, so ask that host directly.
  local WORKER_IFNAME="$NCCL_SOCKET_IFNAME"
  local detected_worker
  detected_worker="$(detect_worker_interface)"
  if [[ -n "$detected_worker" ]]; then
    WORKER_IFNAME="$detected_worker"
    [[ "$WORKER_IFNAME" == "$NCCL_SOCKET_IFNAME" ]] ||
      log "Worker fabric interface differs from head: ${WORKER_IFNAME}"
  fi
  # Validate that both nodes use exactly the same locked image.
  local image_id cache_key head_flashinfer_cache worker_flashinfer_cache
  image_id=$(_assert_runtime_images)
  cache_key="${image_id#sha256:}"
  cache_key="${cache_key:0:16}"
  # Isolate generated FlashInfer artifacts by immutable image ID. This prevents
  # Python wrappers from loading stale TVM-FFI modules with an older signature.
  head_flashinfer_cache="${FLASHINFER_CACHE}/${cache_key}"
  worker_flashinfer_cache="${WORKER_FLASHINFER_CACHE}/${cache_key}"
  _ensure_local_owned_dir "$FLASHINFER_CACHE"
  _ensure_local_owned_dir "$head_flashinfer_cache"

  # Check port not in use locally
  local ss_out; ss_out=$(ss -tlnp 2>/dev/null || true)
  if [[ "$ss_out" == *":${API_PORT} "* ]]; then
    die "Port ${API_PORT} is already in use on head node. Stop conflicting service first."
  fi

  # Ensure no stale containers
  docker rm -f "$MASTER_CONTAINER" 2>/dev/null || true
  ssh_worker "docker rm -f '${WORKER_CONTAINER}' 2>/dev/null || true"

  local worker_hf="${WORKER_HF_HOME:-$HF_HOME}"
  ssh_worker "mkdir -p '${worker_hf}' '${WORKER_VLLM_CACHE}'"
  _ensure_worker_owned_dir "$WORKER_FLASHINFER_CACHE"
  _ensure_worker_owned_dir "$worker_flashinfer_cache"
  local host_snap container_model_path
  host_snap=$(_snapshot_path local "$HF_HOME")
  [[ -d "$host_snap" ]] || die "Local snapshot not found. Run verify-files first."
  case "$host_snap" in
    "$HF_HOME"/*) container_model_path="/cache/${host_snap#"$HF_HOME"/}" ;;
    *) die "Resolved snapshot is outside HF_HOME: ${host_snap}" ;;
  esac
  log "Using local snapshot path inside containers: ${container_model_path}"
  # Build NCCL env args
  mapfile -t nccl_env < <(_nccl_env_args)

  # Common vLLM serve flags (written to a shell script to avoid JSON quoting through SSH)
  local vllm_flags=(
    "$container_model_path"
    "--served-model-name" "$SERVED_MODEL_NAME"
    "--tensor-parallel-size" "2"
    "--distributed-executor-backend" "mp"
    "--nnodes" "2"
    "--master-addr" "$TRANSPORT_IP_MASTER"
    "--master-port" "$MASTER_PORT"
    "--kv-cache-dtype" "nvfp4_ds_mla"
    "--block-size" "256"
    "--max-model-len" "$MAX_MODEL_LEN"
    "--max-num-seqs" "$MAX_NUM_SEQS"
    "--max-num-batched-tokens" "$MAX_NUM_BATCHED_TOKENS"
    "--max-cudagraph-capture-size" "$MAX_CUDAGRAPH_CAPTURE_SIZE"
    "--compilation-config" "{\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\"}"
    "--gpu-memory-utilization" "$GPU_MEMORY_UTILIZATION"
    "--moe-backend" "$MOE_BACKEND"
    "--async-scheduling"
    "--enable-chunked-prefill"
    "--enable-prefix-caching"
    "--generation-config" "vllm"
    "--tool-call-parser" "deepseek_v4"
    "--enable-auto-tool-choice"
    "--reasoning-parser" "deepseek_v4"
  )

  if [[ "$ENABLE_THINKING" == "true" ]]; then
    vllm_flags+=("--default-chat-template-kwargs" '{"thinking":true}')
  else
    vllm_flags+=("--default-chat-template-kwargs" '{"thinking":false}')
  fi

  if [[ "$ENABLE_DSPARK_SPECULATION" == "true" && "${SPECULATIVE_TOKENS}" != "0" ]]; then
    local spec_cfg='{"method":"dspark","num_speculative_tokens":'"${SPECULATIVE_TOKENS}"',"draft_sample_method":"probabilistic"}'
    vllm_flags+=("--speculative-config" "$spec_cfg")
  else
    log "DSpark speculative decoding disabled (ENABLE_DSPARK_SPECULATION=${ENABLE_DSPARK_SPECULATION}, SPECULATIVE_TOKENS=${SPECULATIVE_TOKENS})."
  fi

  local quoted_flags
  quoted_flags=$(printf ' %q' "${vllm_flags[@]}")

  # ── Step 1: start worker (rank 1) FIRST ─────────────────────────────────────
  log "Starting worker container (rank 1, headless) on ${WORKER_IP} …"

  local worker_script
  worker_script=$(cat <<WSCRIPT
#!/bin/bash
set -e
export VLLM_HOST_IP=${TRANSPORT_IP_WORKER}
export HF_HOME=/cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}
$([ -n "$WORKER_IFNAME" ] && echo "export NCCL_SOCKET_IFNAME=${WORKER_IFNAME}")
$([ -n "$WORKER_IFNAME" ] && echo "export GLOO_SOCKET_IFNAME=${WORKER_IFNAME}")
$([ -n "$WORKER_IFNAME" ] && echo "export TP_SOCKET_IFNAME=${WORKER_IFNAME}")
$([ -n "$WORKER_IFNAME" ] && echo "export UCX_NET_DEVICES=${WORKER_IFNAME}")
$([ -n "$WORKER_IFNAME" ] && echo "export OMPI_MCA_btl_tcp_if_include=${WORKER_IFNAME}")
$([ -n "$NCCL_IB_HCA" ] && echo "export NCCL_IB_HCA=${NCCL_IB_HCA}")
$([ -n "$NCCL_IB_HCA" ] && echo "export NCCL_IB_DISABLE=0" || echo "export NCCL_IB_DISABLE=1")
exec vllm serve ${quoted_flags} --node-rank 1 --host 127.0.0.1 --port 18000 --headless
WSCRIPT
)

  # Write worker script to temp file, copy to worker, run in container
  ssh_worker "mkdir -p /tmp/ds4flash && cat > /tmp/ds4flash/worker.sh" <<< "$worker_script"
  ssh_worker "chmod +x /tmp/ds4flash/worker.sh"
  ssh_worker "docker run -d --name '${WORKER_CONTAINER}' \
    --gpus all --network host --ipc host \
    -v '${worker_hf}:/cache' \
    -v '${WORKER_VLLM_CACHE}:/root/.cache/vllm' \
    -v '${worker_flashinfer_cache}:/root/.cache/flashinfer' \
    -v '/tmp/ds4flash:/tmp/ds4flash' \
    --entrypoint bash \
    '${VLLM_IMAGE}' /tmp/ds4flash/worker.sh"

  log "Worker container started. Waiting 12 s for rank-1 init …"
  sleep 12
  if [[ "$(ssh_worker "docker inspect -f '{{.State.Running}}' '${WORKER_CONTAINER}' 2>/dev/null" || true)" != "true" ]]; then
    ssh_worker "docker logs --tail 200 '${WORKER_CONTAINER}' 2>&1" || true
    die "Worker container exited before the head node started."
  fi

  # ── Step 2: start head (rank 0) ────────────────────────────────────────────
  log "Starting head container (rank 0, serving API) locally …"

  docker run -d --name "$MASTER_CONTAINER" \
    --gpus all --network host --ipc host \
    -v "${HF_HOME}:/cache" \
    -v "${VLLM_CACHE}:/root/.cache/vllm" \
    -v "${head_flashinfer_cache}:/root/.cache/flashinfer" \
    -e VLLM_HOST_IP="$TRANSPORT_IP_MASTER" \
    -e HF_HOME=/cache \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_USE_FLASHINFER_SAMPLER=1 \
    -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
    -e NCCL_IGNORE_CPU_AFFINITY=1 \
    -e "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}" \
    "${nccl_env[@]}" \
    --entrypoint bash \
    "$VLLM_IMAGE" \
    -lc "$(printf 'exec vllm serve %s --node-rank 0 --host %q --port %q' \
        "$quoted_flags" "$API_HOST" "$API_PORT")"

  # ── Step 3: health poll ────────────────────────────────────────────────────
  log "Polling /v1/models (timeout ${STARTUP_TIMEOUT}s) …"
  local deadline=$(( $(date +%s) + STARTUP_TIMEOUT ))
  while (( $(date +%s) < deadline )); do
    if curl -fsS "http://127.0.0.1:${API_PORT}/v1/models" >/dev/null 2>&1; then
      log "Server is ready."
      local adv; adv=$(detect_advertise_ip)
      echo ""
      echo "  Model   : ${SERVED_MODEL_NAME}"
      echo "  API     : http://${adv}:${API_PORT}/v1"
      echo "  Context : ${MAX_MODEL_LEN} tokens"
      echo "  Topology: 2×DGX Spark, TP=2, mp backend"
      return
    fi
    sleep 10
  done
  die "Server did not become ready within ${STARTUP_TIMEOUT}s. Containers were left running; check: $0 logs head 200"
}

# ─── stop ─────────────────────────────────────────────────────────────────────
stop() {
  log "Stopping head container …"
  docker rm -f "$MASTER_CONTAINER" 2>/dev/null || true
  log "Stopping worker container on ${WORKER_IP} …"
  ssh_worker "docker rm -f '${WORKER_CONTAINER}' 2>/dev/null || true"
  # Release shared memory that mp backend may have created
  sudo -n sh -c 'rm -f /dev/shm/psm_* /dev/shm/sem.mp-*' 2>/dev/null || true
  ssh_worker "sudo -n sh -c 'rm -f /dev/shm/psm_* /dev/shm/sem.mp-*' 2>/dev/null || true"
  log "Stopped."
}

# ─── status ───────────────────────────────────────────────────────────────────
status() {
  echo "=== Configuration ==="
  echo "  Model   : ${MODEL_ID} @ ${MODEL_REVISION}"
  echo "  Image   : ${VLLM_IMAGE}"
  echo "  Master  : ${MASTER_IP}  (transport: ${TRANSPORT_IP_MASTER})"
  echo "  Worker  : ${WORKER_IP}  (transport: ${TRANSPORT_IP_WORKER})"
  echo "  Context : ${MAX_MODEL_LEN}"
  echo "  Port    : ${API_PORT}"

  echo ""
  echo "=== Containers ==="
  echo "--- Head (local) ---"
  docker ps --filter "name=${MASTER_CONTAINER}" --format \
    'table {{.Names}}\t{{.Status}}\t{{.RunningFor}}' 2>/dev/null || echo "(docker not accessible)"

  echo "--- Worker (${WORKER_IP}) ---"
  ssh_worker "docker ps --filter 'name=${WORKER_CONTAINER}' --format \
    'table {{.Names}}\t{{.Status}}\t{{.RunningFor}}'" 2>/dev/null || echo "(SSH failed)"

  echo ""
  echo "=== GPU (head) ==="
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null || echo "(nvidia-smi not available)"

  echo ""
  echo "=== GPU (worker) ==="
  ssh_worker "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader" 2>/dev/null || echo "(SSH or nvidia-smi failed)"

  echo ""
  echo "=== API health ==="
  if curl -fsS "http://127.0.0.1:${API_PORT}/v1/models" 2>/dev/null; then
    echo ""
  else
    echo "NOT READY (http://127.0.0.1:${API_PORT}/v1/models)"
  fi
}

# ─── logs ─────────────────────────────────────────────────────────────────────
logs() {
  local target="${1:-head}"
  local n="${2:-200}"
  case "$target" in
    head|master) docker logs --tail "$n" "$MASTER_CONTAINER" 2>&1 ;;
    worker)      ssh_worker "docker logs --tail '$n' '${WORKER_CONTAINER}' 2>&1" ;;
    *)           die "Usage: $0 logs [head|worker] [N]" ;;
  esac
}

# ─── network-info ─────────────────────────────────────────────────────────────
network_info() {
  _resolve_nccl_ifname
  _resolve_nccl_hca
  local adv; adv=$(detect_advertise_ip)
  echo "=== Network ==="
  echo "  Head management  : ${MASTER_IP}"
  echo "  Worker management: ${WORKER_IP}"
  echo "  Head transport   : ${TRANSPORT_IP_MASTER}"
  echo "  Worker transport : ${TRANSPORT_IP_WORKER}"
  echo "  NCCL interface   : ${NCCL_SOCKET_IFNAME:-<no interface owns ${TRANSPORT_IP_MASTER}>}"
  echo "  NCCL HCA         : ${NCCL_IB_HCA:-<NOT SET — use TCP fallback>}"
  echo "  Master port      : ${MASTER_PORT}"
  echo "  API bind         : ${API_HOST}:${API_PORT}"
  echo "  Public API       : http://${adv}:${API_PORT}/v1"
  echo ""
  echo "  Discover fabric interface: ip link show | grep -E '200|QSFP|roce|mlx'"
}

# ─── client-config ────────────────────────────────────────────────────────────
client_config() {
  local adv; adv=$(detect_advertise_ip)
  cat <<CONF
# OpenAI Python SDK — DeepSeek-V4-Flash-NVFP4 (stacked)
from openai import OpenAI
client = OpenAI(base_url="http://${adv}:${API_PORT}/v1", api_key="none")

# Text (thinking OFF)
resp = client.chat.completions.create(
    model="${SERVED_MODEL_NAME}",
    messages=[{"role":"user","content":"Hello!"}],
    temperature=1.0,
    extra_body={"chat_template_kwargs": {"thinking": False}},
)
print(resp.choices[0].message.content)

# Reasoning (thinking ON)
resp = client.chat.completions.create(
    model="${SERVED_MODEL_NAME}",
    messages=[{"role":"user","content":"What is 37 * 43?"}],
    temperature=1.0,
    extra_body={"chat_template_kwargs": {"thinking": True}},
)
print("Think:", resp.choices[0].message.reasoning_content)
print("Answer:", resp.choices[0].message.content)

# Tool calling
tools = [{"type":"function","function":{"name":"get_weather",
  "description":"Get weather","parameters":{"type":"object",
  "properties":{"location":{"type":"string"}},"required":["location"]}}}]
resp = client.chat.completions.create(
    model="${SERVED_MODEL_NAME}",
    messages=[{"role":"user","content":"Weather in Bangkok?"}],
    tools=tools, tool_choice="required",
    extra_body={"chat_template_kwargs": {"thinking": False}},
)
print(resp.choices[0].message.tool_calls)
CONF
}

# ─── bench ────────────────────────────────────────────────────────────────────
bench() {
  log "Benchmark: single-stream 500-token decode …"
  python3 - <<PYEOF
import urllib.request, json, time, sys

url   = "http://127.0.0.1:${API_PORT}/v1/chat/completions"
model = "${SERVED_MODEL_NAME}"
payload = json.dumps({
    "model": model,
    "messages": [{"role":"user","content":"Write a 500-word essay on large language models."}],
    "max_tokens": 512,
    "temperature": 1.0,
    "stream": False,
    "chat_template_kwargs": {"thinking": False},
}).encode()

req = urllib.request.Request(url, data=payload,
    headers={"Content-Type":"application/json"}, method="POST")
t0 = time.time()
with urllib.request.urlopen(req, timeout=300) as r:
    data = json.loads(r.read())
elapsed = time.time() - t0

usage = data.get("usage", {})
toks  = usage.get("completion_tokens", 0)
ttft  = data.get("time_to_first_token", elapsed)
print(f"Tokens     : {toks}")
print(f"Elapsed    : {elapsed:.1f}s")
print(f"Throughput : {toks/elapsed:.1f} tok/s")
print(f"Expected   : ~41-55 tok/s (community: dual DGX Spark)")
PYEOF
}

# ─── stress ───────────────────────────────────────────────────────────────────
stress() {
  local n="${1:-4}"
  log "Stress test: ${n} concurrent requests …"
  python3 - "$API_PORT" "$SERVED_MODEL_NAME" "$n" <<'PYEOF'
import sys, urllib.request, json, threading, time

port  = int(sys.argv[1])
model = sys.argv[2]
n     = int(sys.argv[3])
url   = f"http://127.0.0.1:{port}/v1/chat/completions"
ok, fail = 0, 0
lock = threading.Lock()

def req(_):
    global ok, fail
    payload = json.dumps({
        "model": model,
        "messages": [{"role":"user","content":"Describe the sky in 3 sentences."}],
        "max_tokens": 64, "temperature": 1.0,
        "chat_template_kwargs": {"thinking": False},
    }).encode()
    try:
        r = urllib.request.urlopen(
            urllib.request.Request(url, data=payload,
                headers={"Content-Type":"application/json"}, method="POST"),
            timeout=120)
        d = json.loads(r.read())
        with lock: ok += 1
    except Exception as e:
        with lock: fail += 1
        print(f"FAIL: {e}")

threads = [threading.Thread(target=req, args=(i,)) for i in range(n)]
t0 = time.time()
for t in threads: t.start()
for t in threads: t.join()
print(f"{ok} OK, {fail} FAIL  ({time.time()-t0:.1f}s)")
PYEOF
}

# ─── test-text ────────────────────────────────────────────────────────────────
test_text() {
  log "test-text …"
  python3 - <<PYEOF
import urllib.request, json, sys

url     = "http://127.0.0.1:${API_PORT}/v1/chat/completions"
model   = "${SERVED_MODEL_NAME}"
payload = json.dumps({
    "model": model,
    "messages": [{"role":"user","content":"Reply with one sentence about Paris."}],
    "max_tokens": 64,
    "chat_template_kwargs": {"thinking": False},
}).encode()
req = urllib.request.Request(url, data=payload,
    headers={"Content-Type":"application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=120) as r:
    d = json.loads(r.read())
msg = d["choices"][0]["message"]["content"].strip()
if not msg:
    print("FAIL: empty response"); sys.exit(1)
print(f"PASS: {msg[:120]}")
PYEOF
}

# ─── test-reasoning ───────────────────────────────────────────────────────────
test_reasoning() {
  log "test-reasoning (thinking=ON, expect 37×43=1591) …"
  python3 - <<PYEOF
import urllib.request, json, sys

url     = "http://127.0.0.1:${API_PORT}/v1/chat/completions"
model   = "${SERVED_MODEL_NAME}"
payload = json.dumps({
    "model": model,
    "messages": [{"role":"user","content":"What is 37 multiplied by 43? Show your work."}],
    "max_tokens": 512,
    "chat_template_kwargs": {"thinking": True},
}).encode()
req = urllib.request.Request(url, data=payload,
    headers={"Content-Type":"application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=120) as r:
    d = json.loads(r.read())
msg  = d["choices"][0]["message"]
think= msg.get("reasoning_content","")
ans  = msg.get("content","")
full = (think + ans).lower()
if "1591" in full:
    print(f"PASS: 1591 found. thinking_len={len(think)} answer_len={len(ans)}")
else:
    print(f"FAIL: 1591 not found.\nThink: {think[:300]}\nAnswer: {ans[:200]}")
    sys.exit(1)
PYEOF
}

# ─── test-tools ───────────────────────────────────────────────────────────────
test_tools() {
  local mode="${1:-required}"
  log "test-tools ${mode} …"
  python3 - "$API_PORT" "$SERVED_MODEL_NAME" "$mode" <<'PYEOF'
import urllib.request, json, sys

port  = int(sys.argv[1])
model = sys.argv[2]
mode  = sys.argv[3]
url   = f"http://127.0.0.1:{port}/v1/chat/completions"

tools = [{"type":"function","function":{
  "name":"get_weather",
  "description":"Get current weather for a location",
  "parameters":{"type":"object",
    "properties":{"location":{"type":"string","description":"City name"}},
    "required":["location"]}}}]

payload = json.dumps({
    "model": model,
    "messages": [{"role":"user","content":"What is the weather in Bangkok?"}],
    "tools": tools,
    "tool_choice": mode,
    "max_tokens": 256,
    "chat_template_kwargs": {"thinking": False},
}).encode()
req = urllib.request.Request(url, data=payload,
    headers={"Content-Type":"application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=120) as r:
    d = json.loads(r.read())

choice = d["choices"][0]
tc = choice["message"].get("tool_calls") or []
if tc:
    fn = tc[0]["function"]
    print(f"PASS: {fn['name']}({fn['arguments'][:80]})")
elif mode == "required":
    print(f"FAIL: no tool_calls in required mode.\n{json.dumps(choice,indent=2)[:400]}")
    sys.exit(1)
else:
    print(f"WARN(auto): model answered in text: {choice['message'].get('content','')[:120]}")
PYEOF
}

# ─── test-tool-loop ───────────────────────────────────────────────────────────
test_tool_loop() {
  log "test-tool-loop (2-turn) …"
  python3 - "$API_PORT" "$SERVED_MODEL_NAME" <<'PYEOF'
import urllib.request, json, sys

port  = int(sys.argv[1])
model = sys.argv[2]
url   = f"http://127.0.0.1:{port}/v1/chat/completions"
tools = [{"type":"function","function":{
  "name":"get_weather","description":"Get weather",
  "parameters":{"type":"object",
    "properties":{"location":{"type":"string"}},"required":["location"]}}}]

def call(messages, tool_choice="auto"):
    p = json.dumps({"model":model,"messages":messages,"tools":tools,
        "tool_choice":tool_choice,"max_tokens":256,
        "chat_template_kwargs":{"thinking":False}}).encode()
    r = urllib.request.urlopen(urllib.request.Request(url,data=p,
        headers={"Content-Type":"application/json"},method="POST"),timeout=120)
    return json.loads(r.read())["choices"][0]["message"]

msgs = [{"role":"user","content":"What is the weather in Chiang Mai?"}]
turn1 = call(msgs, "required")
tc = (turn1.get("tool_calls") or [])
if not tc:
    print("FAIL: no tool call on turn 1"); sys.exit(1)

fn_name = tc[0]["function"]["name"]
call_id = tc[0]["id"]
msgs += [dict(turn1), {"role":"tool","tool_call_id":call_id,
    "name":fn_name,"content":'{"temperature":"32C","condition":"Sunny"}'}]
turn2 = call(msgs)
ans = turn2.get("content","")
if ans:
    print(f"PASS (turn 2): {ans[:120]}")
else:
    print("FAIL: empty turn-2 response"); sys.exit(1)
PYEOF
}

# ─── props ────────────────────────────────────────────────────────────────────
props() {
  curl -fsS "http://127.0.0.1:${API_PORT}/v1/models" | python3 -m json.tool
}

# ─── prepare-runtime ─────────────────────────────────────────────────────────
# (declared above — exposed as command alias)
runtime_prepare() { prepare_runtime; }

# ─── dispatch ─────────────────────────────────────────────────────────────────
parse_options "$@"
set -- "${REMAINING_ARGS[@]:-}"

case "${1:-help}" in
  prepare-runtime)  prepare_runtime ;;
  runtime-info)     runtime_info ;;
  download)         download ;;
  verify-files)     verify_files ;;
  sync-worker)      sync_worker ;;
  verify-worker)    verify_worker ;;
  doctor)           doctor ;;
  clear-fi-cache)   clear_flashinfer_cache ;;
  start)            start ;;
  stop)             stop ;;
  restart)          stop; sleep 3; start ;;
  status)           status ;;
  logs)             logs "${2:-head}" "${3:-200}" ;;
  props)            props ;;
  bench)            bench ;;
  stress)           stress "${2:-4}" ;;
  test-text)        test_text ;;
  test-reasoning)   test_reasoning ;;
  test-tools)       test_tools "${2:-required}" ;;
  test-tool-loop)   test_tool_loop ;;
  info|banner)     info ;;
  network-info)     network_info ;;
  client-config)    client_config ;;
  help|--help|-h)
    cat <<HELP
DeepSeek-V4-Flash-NVFP4 stacked controller (2×DGX Spark)

Usage: $0 [OPTIONS] COMMAND [args]

Options:
  --context TOKENS    Override max_model_len (default: ${MAX_MODEL_LEN})
  --port PORT         Override API port (default: ${API_PORT})
  --bind ADDR         Override API bind address (default: ${API_HOST})
  --advertise-ip IP   Override public IP in client-config / network-info
  --interface NAME    Interface for advertise-ip autodetection

Commands:
  prepare-runtime     Pull and lock image on both nodes
  runtime-info        Show image and runtime details
  download            Download model to head node (snapshot_download, resumable)
  verify-files        Verify local model shards
  sync-worker         Rsync model cache to worker node
  verify-worker       Verify worker shard count and SHA-256 output
  doctor              Print runtime/GPU/vLLM/FlashInfer compatibility details
  clear-fi-cache       Stop containers and clear FlashInfer caches on both nodes
  start               Start 2-node deployment (worker-first)
  stop                Remove containers on both nodes
  restart             stop + start
  status              Show container state, GPU, and API health
  logs [head|worker] [N]  Tail container logs
  props               List models endpoint
  bench               Single-stream throughput benchmark
  stress [N]          N concurrent requests (default 4)
  test-text           Basic text generation
  test-reasoning      Thinking=ON, verify 37×43=1591
  test-tools required Tool call parse (required mode)
  test-tools auto     Tool call parse (auto mode)
  test-tool-loop      Two-turn tool result continuation
  network-info        Show cluster IPs and API endpoint
  client-config       Print OpenAI SDK usage examples

Environment overrides:
  MASTER_IP, WORKER_IP, SSH_USER
  TRANSPORT_IP_MASTER, TRANSPORT_IP_WORKER  (fabric/RoCE IPs if different)
  NCCL_SOCKET_IFNAME, NCCL_IB_HCA, NCCL_IB_GID_INDEX
  VLLM_IMAGE, MODEL_ID, MODEL_REVISION
  HF_HOME, VLLM_CACHE, FLASHINFER_CACHE
  MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION, MAX_NUM_SEQS, STARTUP_TIMEOUT
  CUDAGRAPH_MODE (default: PIECEWISE for SM121 stability)
  MOE_BACKEND (default: marlin; set flashinfer_cutlass only after doctor/cache validation)
  ENABLE_DSPARK_SPECULATION (default: false; requires config token marker)
  SPECULATIVE_TOKENS (default: 0; use only with ENABLE_DSPARK_SPECULATION=true)
  ENABLE_THINKING (default: true)
  HF_TOKEN (for download only; never embedded)
HELP
    ;;
  *)
    die "Unknown command '${1}'. Run: $0 help"
    ;;
esac
