#!/usr/bin/env bash
#
# gemma4-31b-stacked.sh
#
# One-file Master controller for two DGX Spark systems:
#   Master 10.100.152.1
#   Worker 10.100.152.2
#
# Before starting Ray/vLLM, this controller can:
#   - download the model on Master
#   - rsync the complete Hugging Face model cache to Worker
#   - create full SHA-256 manifests on both systems
#   - refuse to start if any file, symlink, size, or hash differs
#
set -Eeuo pipefail

SCRIPT_VERSION="${SCRIPT_VERSION:-3.0.0}"

# ==============================================================================
# USER CONFIGURATION
# ==============================================================================

MASTER_IP="${MASTER_IP:-10.100.152.1}"
WORKER_IP="${WORKER_IP:-10.100.152.2}"
SSH_USER="${SSH_USER:-neronain}"

MODEL_ID="nvidia/Gemma-4-31B-IT-NVFP4"
MODEL_REVISION="main"
SERVED_MODEL_NAME="gemma4-31b-nvfp4"

VLLM_IMAGE="vllm/vllm-openai:gemma4-cu130"

TENSOR_PARALLEL_SIZE="2"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="0.85"
MAX_NUM_SEQS="2"
KV_CACHE_DTYPE="auto"

ENABLE_PREFIX_CACHING="1"
ENABLE_CHUNKED_PREFILL="1"

ENABLE_AUTO_TOOL_CHOICE="1"
TOOL_CALL_PARSER="gemma4"
REASONING_PARSER="gemma4"
DEFAULT_ENABLE_THINKING="false"

MAX_IMAGES_PER_PROMPT="4"
MAX_VIDEO_PER_PROMPT="1"

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
ADVERTISE_IP="${ADVERTISE_IP:-}"
ADVERTISE_INTERFACE="${ADVERTISE_INTERFACE:-}"
ROUTE_PROBE_IP="${ROUTE_PROBE_IP:-1.1.1.1}"
CLIENT_OVERHEAD_TOKENS="${CLIENT_OVERHEAD_TOKENS:-8192}"
API_KEY="${API_KEY:-}"

HF_TOKEN=""
HF_HUB_OFFLINE="1"

CLIENT_CONTEXT_TOKENS="${CLIENT_CONTEXT_TOKENS:-auto}"
CLIENT_MAX_OUTPUT_TOKENS="${CLIENT_MAX_OUTPUT_TOKENS:-8192}"

# 1 = hash the entire cache on both nodes before every start.
# This is slower, but proves both copies are byte-identical.
VERIFY_FULL_HASH_BEFORE_START="1"

HEAD_WAIT_SECONDS="600"
CLUSTER_WAIT_SECONDS="900"
API_WAIT_SECONDS="1800"

# ==============================================================================
# PINNED CLUSTER ASSETS
# ==============================================================================

VLLM_COMMIT="51c1ee9b7c8acbba4899a8ebffd390685d171946"
RUN_CLUSTER_URL="https://raw.githubusercontent.com/vllm-project/vllm/${VLLM_COMMIT}/examples/ray_serving/run_cluster.sh"
RAY_PACKAGE="ray[default]>=2.9"

# ==============================================================================
# PATHS / SESSIONS
# ==============================================================================

MASTER_HOME="/home/${SSH_USER}"
BASE_DIR="${MASTER_HOME}/gemma4-31b-stacked"
HF_HOME="${MASTER_HOME}/.cache/huggingface"

MODEL_CACHE_NAME="models--${MODEL_ID//\//--}"
MODEL_CACHE_PATH="${HF_HOME}/hub/${MODEL_CACHE_NAME}"

RUN_CLUSTER="${BASE_DIR}/run_cluster.sh"
REMOTE_SCRIPT="${BASE_DIR}/.gemma4-31b-stacked.remote.sh"
MANIFEST_HELPER="${BASE_DIR}/cache_manifest.py"
REMOTE_MANIFEST_HELPER="${BASE_DIR}/cache_manifest.py"

SPECIAL_DIR="${BASE_DIR}/special-files"
CHAT_TEMPLATE_HOST="${HF_HOME}/gemma4-31b-tool-chat-template.jinja"
CHAT_TEMPLATE_CONTAINER="/root/.cache/huggingface/gemma4-31b-tool-chat-template.jinja"

HEAD_SESSION="gemma4-head"
WORKER_SESSION="gemma4-worker"
API_SESSION="gemma4-api"

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SSH_TARGET="${SSH_USER}@${WORKER_IP}"

# ==============================================================================
# HELPERS
# ==============================================================================

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
warn() { printf '\nWARNING: %s\n' "$*" >&2; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"
}

# ------------------------------------------------------------------------------
# Unified context, port, and advertised-address overrides
# ------------------------------------------------------------------------------

validate_positive_integer() {
  local name="$1"
  local value="$2"

  [[ "$value" =~ ^[0-9]+$ ]] && (( value > 0 )) ||
    die "${name} must be a positive integer: ${value}"
}

validate_port() {
  local value="$1"

  validate_positive_integer "API_PORT" "$value"
  (( value <= 65535 )) ||
    die "API_PORT must be between 1 and 65535: $value"
}

interface_ipv4() {
  local interface="$1"

  command -v ip >/dev/null 2>&1 || return 1

  ip -4 -o addr show dev "$interface" scope global 2>/dev/null |
    awk 'NR == 1 { split($4, a, "/"); print a[1] }'
}

route_source_ipv4() {
  command -v ip >/dev/null 2>&1 || return 1

  ip -4 route get "$ROUTE_PROBE_IP" 2>/dev/null |
    awk '{ for (i = 1; i <= NF; i++) if ($i == "src") { print $(i + 1); exit } }'
}

route_interface() {
  command -v ip >/dev/null 2>&1 || return 1

  ip -4 route get "$ROUTE_PROBE_IP" 2>/dev/null |
    awk '{ for (i = 1; i <= NF; i++) if ($i == "dev") { print $(i + 1); exit } }'
}

detect_advertise_ip() {
  if [[ -n "$ADVERTISE_IP" ]]; then
    printf '%s' "$ADVERTISE_IP"
    return
  fi

  if [[ -n "$ADVERTISE_INTERFACE" ]]; then
    local explicit_ip
    explicit_ip="$(interface_ipv4 "$ADVERTISE_INTERFACE" || true)"

    [[ -n "$explicit_ip" ]] ||
      die "No global IPv4 address found on interface: $ADVERTISE_INTERFACE"

    printf '%s' "$explicit_ip"
    return
  fi

  local route_ip
  route_ip="$(route_source_ipv4 || true)"

  if [[ -n "$route_ip" ]]; then
    printf '%s' "$route_ip"
    return
  fi

  if command -v ip >/dev/null 2>&1; then
    local fallback_ip
    fallback_ip="$(
      ip -4 -o addr show scope global 2>/dev/null |
        awk '
          $2 !~ /^(lo|docker|br-|veth|virbr|cni|flannel|kube|tailscale|zt|wg|ray)/ {
            split($4, a, "/")
            print a[1]
            exit
          }
        '
    )"

    if [[ -n "$fallback_ip" ]]; then
      printf '%s' "$fallback_ip"
      return
    fi
  fi

  hostname -I 2>/dev/null | awk '{ print $1 }'
}

detect_advertise_interface() {
  if [[ -n "$ADVERTISE_INTERFACE" ]]; then
    printf '%s' "$ADVERTISE_INTERFACE"
    return
  fi

  route_interface || true
}

public_base_url() {
  printf 'http://%s:%s/v1' "$(detect_advertise_ip)" "$API_PORT"
}

resolve_client_context_tokens() {
  if [[ "${CLIENT_CONTEXT_TOKENS}" == "auto" ]]; then
    local computed
    computed=$(( MAX_MODEL_LEN - CLIENT_MAX_OUTPUT_TOKENS - CLIENT_OVERHEAD_TOKENS ))

    (( computed >= 1024 )) ||
      die "Context ${MAX_MODEL_LEN} is too small for output and overhead budgets."

    CLIENT_CONTEXT_TOKENS="$computed"
  fi
}

validate_common_config() {
  validate_positive_integer "MAX_MODEL_LEN" "${MAX_MODEL_LEN}"
  validate_port "$API_PORT"
  validate_positive_integer "CLIENT_MAX_OUTPUT_TOKENS" "${CLIENT_MAX_OUTPUT_TOKENS}"
  validate_positive_integer "CLIENT_OVERHEAD_TOKENS" "$CLIENT_OVERHEAD_TOKENS"

  if [[ "${CLIENT_CONTEXT_TOKENS}" != "auto" ]]; then
    validate_positive_integer "CLIENT_CONTEXT_TOKENS" "${CLIENT_CONTEXT_TOKENS}"
  fi

  resolve_client_context_tokens

  (( CLIENT_CONTEXT_TOKENS + CLIENT_MAX_OUTPUT_TOKENS <= MAX_MODEL_LEN )) ||
    die "Client input + output budgets exceed server context."
}

parse_common_options() {
  REMAINING_ARGS=()

  while (( $# )); do
    case "$1" in
      --context)
        [[ $# -ge 2 ]] || die "--context requires a value"
        MAX_MODEL_LEN="$2"
        shift 2
        ;;
      --context=*)
        MAX_MODEL_LEN="${1#*=}"
        shift
        ;;
      --port)
        [[ $# -ge 2 ]] || die "--port requires a value"
        API_PORT="$2"
        shift 2
        ;;
      --port=*)
        API_PORT="${1#*=}"
        shift
        ;;
      --bind)
        [[ $# -ge 2 ]] || die "--bind requires an address"
        API_HOST="$2"
        shift 2
        ;;
      --bind=*)
        API_HOST="${1#*=}"
        shift
        ;;
      --advertise-ip)
        [[ $# -ge 2 ]] || die "--advertise-ip requires an address"
        ADVERTISE_IP="$2"
        shift 2
        ;;
      --advertise-ip=*)
        ADVERTISE_IP="${1#*=}"
        shift
        ;;
      --interface)
        [[ $# -ge 2 ]] || die "--interface requires a name"
        ADVERTISE_INTERFACE="$2"
        shift 2
        ;;
      --interface=*)
        ADVERTISE_INTERFACE="${1#*=}"
        shift
        ;;
      --client-input)
        [[ $# -ge 2 ]] || die "--client-input requires a value"
        CLIENT_CONTEXT_TOKENS="$2"
        shift 2
        ;;
      --client-input=*)
        CLIENT_CONTEXT_TOKENS="${1#*=}"
        shift
        ;;
      --client-output)
        [[ $# -ge 2 ]] || die "--client-output requires a value"
        CLIENT_MAX_OUTPUT_TOKENS="$2"
        shift 2
        ;;
      --client-output=*)
        CLIENT_MAX_OUTPUT_TOKENS="${1#*=}"
        shift
        ;;
      --)
        shift
        REMAINING_ARGS+=("$@")
        break
        ;;
      *)
        REMAINING_ARGS+=("$1")
        shift
        ;;
    esac
  done

  validate_common_config

  export API_HOST API_PORT ADVERTISE_IP ADVERTISE_INTERFACE ROUTE_PROBE_IP
  export MAX_MODEL_LEN CLIENT_CONTEXT_TOKENS CLIENT_MAX_OUTPUT_TOKENS
}

network_info() {
  local advertise_ip
  local advertise_interface

  advertise_ip="$(detect_advertise_ip)"
  advertise_interface="$(detect_advertise_interface)"

  echo "===== STACKED CONTROLLER NETWORK ====="
  echo "Bind address:        $API_HOST"
  echo "API port:            $API_PORT"
  echo "Advertise IP:        $advertise_ip"
  echo "Advertise interface:${advertise_interface:+ $advertise_interface}"
  echo "Selected endpoint:   http://${advertise_ip}:${API_PORT}/v1"
  echo "Server context:      ${MAX_MODEL_LEN}"
  echo

  if command -v ip >/dev/null 2>&1; then
    echo "===== GLOBAL IPv4 ADDRESSES ====="
    ip -4 -o addr show scope global |
      awk '{ print $2, $4 }'
    echo
    echo "===== ROUTE USED FOR AUTO-SELECTION ====="
    ip -4 route get "$ROUTE_PROBE_IP" 2>/dev/null || true
  fi
}


ssh_worker() {
  ssh \
    -o BatchMode=yes \
    -o ConnectTimeout=10 \
    -o ServerAliveInterval=10 \
    -o ServerAliveCountMax=3 \
    "$SSH_TARGET" "$@"
}

detect_interface() {
  local address="$1"
  ip -o -4 addr show |
    awk -v ip="$address" '$4 ~ ("^" ip "/") {print $2; exit}'
}

check_master() {
  local interface
  interface="$(detect_interface "$MASTER_IP")"
  [[ -n "$interface" ]] ||
    die "Run this controller on Master ${MASTER_IP}."
}

ensure_image_local() {
  if ! docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1; then
    docker pull "$VLLM_IMAGE"
  fi
}

ensure_image_worker() {
  if ! ssh_worker "docker image inspect '$VLLM_IMAGE' >/dev/null 2>&1"; then
    ssh_worker "docker pull '$VLLM_IMAGE'"
  fi
}

model_cache_complete_local() {
  [[ -d "${MODEL_CACHE_PATH}/snapshots" ]] || return 1

  local snapshot
  snapshot="$(
    find "${MODEL_CACHE_PATH}/snapshots" \
      -mindepth 1 \
      -maxdepth 1 \
      -type d \
      -print -quit
  )"

  [[ -n "$snapshot" ]]
}

model_cache_complete_worker() {
  ssh_worker \
    "test -d '${MODEL_CACHE_PATH}/snapshots' &&
     test -n \"\$(find '${MODEL_CACHE_PATH}/snapshots' \
       -mindepth 1 -maxdepth 1 -type d -print -quit)\""
}

resolve_snapshot_dir() {
  local snapshot
  snapshot="$(
    find "${MODEL_CACHE_PATH}/snapshots" \
      -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' |
      sort -nr |
      awk 'NR==1 {$1=""; sub(/^ /,""); print; exit}'
  )"
  [[ -n "$snapshot" ]] || die "No model snapshot found."
  readlink -f "$snapshot"
}

head_container() {
  docker ps \
    --filter "ancestor=${VLLM_IMAGE}" \
    --format '{{.Names}}' |
    grep -E '^node-[0-9]+$' |
    head -n 1 || true
}

api_auth_args() {
  API_AUTH_ARGS=()
  if [[ -n "$API_KEY" ]]; then
    API_AUTH_ARGS=(-H "Authorization: Bearer ${API_KEY}")
  fi
}

# ==============================================================================
# ASSETS
# ==============================================================================

patch_run_cluster() {
  python3 - "$RUN_CLUSTER" "$RAY_PACKAGE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
package = sys.argv[2]
text = path.read_text()

if "pip install -q --root-user-action=ignore" in text:
    raise SystemExit(0)

needle = 'RAY_START_CMD="ray start --block"'
replacement = (
    'RAY_START_CMD="pip install -q --root-user-action=ignore '
    f"'{package}'"
    ' && ray start --block"'
)

if needle not in text:
    raise SystemExit("Expected RAY_START_CMD line not found")

path.write_text(text.replace(needle, replacement, 1))
PY
}

write_manifest_helper() {
  mkdir -p "$BASE_DIR"

  cat > "$MANIFEST_HELPER" <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()

root = args.root.resolve()
if not root.is_dir():
    raise SystemExit(f"Missing directory: {root}")

records = []

for path in sorted(root.rglob("*"), key=lambda p: str(p.relative_to(root))):
    rel = str(path.relative_to(root))

    if path.is_symlink():
        records.append(("L", rel, os.readlink(path)))
        continue

    if path.is_dir():
        records.append(("D", rel, ""))
        continue

    if path.is_file():
        size = path.stat().st_size
        if args.quick:
            digest = "-"
        else:
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
        records.append(("F", rel, f"{size}\t{digest}"))

for kind, rel, value in records:
    print(f"{kind}\t{rel}\t{value}")
PY

  chmod 0755 "$MANIFEST_HELPER"
}

ensure_assets() {
  require_command curl
  require_command python3

  mkdir -p "$BASE_DIR" "$SPECIAL_DIR" "$HF_HOME"

  if [[ ! -s "$RUN_CLUSTER" ]]; then
    curl -fsSL "$RUN_CLUSTER_URL" -o "$RUN_CLUSTER"
  fi

  patch_run_cluster
  chmod 0755 "$RUN_CLUSTER"
  write_manifest_helper
}

copy_assets_worker() {
  ssh_worker "mkdir -p '$BASE_DIR' '$HF_HOME' '$SPECIAL_DIR'"

  scp -q "$RUN_CLUSTER" "${SSH_TARGET}:${RUN_CLUSTER}"
  scp -q "$MANIFEST_HELPER" "${SSH_TARGET}:${REMOTE_MANIFEST_HELPER}"
  scp -q "$SCRIPT_PATH" "${SSH_TARGET}:${REMOTE_SCRIPT}"

  ssh_worker \
    "chmod 0755 '$RUN_CLUSTER' '$REMOTE_MANIFEST_HELPER' '$REMOTE_SCRIPT'"
}

# ==============================================================================
# MODEL DOWNLOAD / SPECIAL FILES
# ==============================================================================

download_model() {
  check_master
  ensure_assets
  ensure_image_local

  local -a token_env=()
  if [[ -n "$HF_TOKEN" ]]; then
    token_env=(-e "HF_TOKEN=${HF_TOKEN}")
  fi

  log "Downloading model on Master"

  docker run --rm \
    --user "$(id -u "$SSH_USER"):$(id -g "$SSH_USER")" \
    -e HOME=/tmp \
    -e HF_HOME=/cache \
    -e MODEL_ID="$MODEL_ID" \
    -e MODEL_REVISION="$MODEL_REVISION" \
    "${token_env[@]}" \
    -v "${HF_HOME}:/cache" \
    --entrypoint python3 \
    "$VLLM_IMAGE" \
    -c '
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    revision=os.environ["MODEL_REVISION"],
)
'

  prepare_special_files
  verify_files_local

  log "Master download complete"
  du -sh "$MODEL_CACHE_PATH"
}

prepare_special_files() {
  local snapshot
  snapshot="$(resolve_snapshot_dir)"

  mkdir -p "$SPECIAL_DIR"

  local file
  local -a files=(
    chat_template.jinja
    config.json
    generation_config.json
    hf_quant_config.json
    model.safetensors.index.json
    processor_config.json
    tokenizer.json
    tokenizer_config.json
  )

  for file in "${files[@]}"; do
    [[ -e "${snapshot}/${file}" ]] ||
      die "Required file missing: ${snapshot}/${file}"
    cp -L "${snapshot}/${file}" "${SPECIAL_DIR}/${file}"
  done

  cp -L "${snapshot}/chat_template.jinja" "$CHAT_TEMPLATE_HOST"

  (
    cd "$SPECIAL_DIR"
    sha256sum "${files[@]}" > SHA256SUMS
  )
}

verify_files_local() {
  require_command python3
  require_command sha256sum

  model_cache_complete_local ||
    die "Model missing on Master. Run: $0 download"

  local snapshot
  snapshot="$(resolve_snapshot_dir)"

  [[ -f "${SPECIAL_DIR}/SHA256SUMS" ]] || prepare_special_files

  python3 - "$snapshot" "$SPECIAL_DIR" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

snapshot = Path(sys.argv[1])
special = Path(sys.argv[2])

required = [
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "hf_quant_config.json",
    "model.safetensors.index.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]

missing = [name for name in required if not (snapshot / name).exists()]
if missing:
    raise SystemExit("Missing files: " + ", ".join(missing))

index = json.loads((snapshot / "model.safetensors.index.json").read_text())
weight_map = index.get("weight_map", {})
shards = sorted(set(weight_map.values()))

if not weight_map:
    raise SystemExit("Empty weight_map")

missing_shards = [name for name in shards if not (snapshot / name).exists()]
if missing_shards:
    raise SystemExit("Missing shards: " + ", ".join(missing_shards))

broken = [p for p in snapshot.rglob("*") if p.is_symlink() and not p.exists()]
if broken:
    raise SystemExit(
        "Broken symlinks:\n" + "\n".join(str(p) for p in broken[:50])
    )

template = (special / "chat_template.jinja").read_text()
for marker in ("enable_thinking", "tools", "<|tool_call>", "<|image|>", "<|video|>"):
    if marker not in template:
        raise SystemExit(f"Template missing marker: {marker}")

json.loads((special / "hf_quant_config.json").read_text())
json.loads((special / "processor_config.json").read_text())

print(f"Snapshot:          {snapshot.name}")
print(f"Indexed tensors:   {len(weight_map)}")
print(f"Referenced shards: {len(shards)}")
print("Special files:     verified")
PY

  (
    cd "$SPECIAL_DIR"
    sha256sum -c SHA256SUMS
  )

  [[ -s "$CHAT_TEMPLATE_HOST" ]] ||
    die "Runtime chat template is missing."

  log "Master files verified"
}

# ==============================================================================
# SYNC / BYTE-FOR-BYTE COMPARISON
# ==============================================================================

sync_worker() {
  check_master
  ensure_assets

  require_command rsync
  require_command scp
  require_command ssh

  model_cache_complete_local ||
    die "Download the model on Master first."

  prepare_special_files
  verify_files_local
  copy_assets_worker
  ensure_image_worker

  log "Preparing Worker directories"

  ssh_worker \
    "mkdir -p '${MODEL_CACHE_PATH}' '${HF_HOME}' '${SPECIAL_DIR}'"

  log "Copying complete Hugging Face model cache to Worker"

  rsync -aH \
    --delete \
    --partial \
    --info=progress2 \
    "${MODEL_CACHE_PATH}/" \
    "${SSH_TARGET}:${MODEL_CACHE_PATH}/"

  log "Copying runtime chat template and special files"

  rsync -aH --delete \
    "${SPECIAL_DIR}/" \
    "${SSH_TARGET}:${SPECIAL_DIR}/"

  scp -q \
    "$CHAT_TEMPLATE_HOST" \
    "${SSH_TARGET}:${CHAT_TEMPLATE_HOST}"

  verify_worker_files

  log "Master and Worker model copies are identical"
}

verify_worker_files() {
  check_master
  ensure_assets
  copy_assets_worker

  model_cache_complete_local ||
    die "Model missing on Master."

  model_cache_complete_worker ||
    die "Model missing on Worker. Run: $0 sync-worker"

  local mode_arg=""
  if [[ "$VERIFY_FULL_HASH_BEFORE_START" != "1" ]]; then
    mode_arg="--quick"
  fi

  local temp_dir
  temp_dir="$(mktemp -d)"
  local local_manifest="${temp_dir}/master.manifest"
  local worker_manifest="${temp_dir}/worker.manifest"
  local remote_manifest="${BASE_DIR}/worker.manifest"

  log "Building Master and Worker manifests"
  if [[ "$VERIFY_FULL_HASH_BEFORE_START" == "1" ]]; then
    log "Full SHA-256 mode: all cache blobs will be hashed"
  else
    log "Quick mode: path, symlink, and size comparison only"
  fi

  ssh_worker \
    "python3 '$REMOTE_MANIFEST_HELPER' '$MODEL_CACHE_PATH' $mode_arg \
       > '$remote_manifest'" &
  local remote_pid=$!

  python3 "$MANIFEST_HELPER" "$MODEL_CACHE_PATH" $mode_arg \
    > "$local_manifest"

  wait "$remote_pid"

  scp -q "${SSH_TARGET}:${remote_manifest}" "$worker_manifest"

  if ! diff -u "$local_manifest" "$worker_manifest"; then
    rm -rf "$temp_dir"
    die "Master and Worker model caches differ. Run sync-worker again."
  fi

  local local_template_hash
  local remote_template_hash

  local_template_hash="$(sha256sum "$CHAT_TEMPLATE_HOST" | awk '{print $1}')"
  remote_template_hash="$(
    ssh_worker "sha256sum '$CHAT_TEMPLATE_HOST' | awk '{print \$1}'"
  )"

  [[ "$local_template_hash" == "$remote_template_hash" ]] ||
    die "Runtime chat templates differ."

  rm -rf "$temp_dir"

  log "Master and Worker caches are byte-identical"
}

# ==============================================================================
# PREFLIGHT
# ==============================================================================

preflight_master() {
  require_command docker
  require_command tmux
  require_command curl
  require_command python3
  require_command nvidia-smi

  docker info >/dev/null 2>&1 ||
    die "Docker unavailable on Master."

  nvidia-smi >/dev/null 2>&1 ||
    die "Host GPU check failed on Master."

  ensure_image_local

  docker run --rm \
    --gpus all \
    --entrypoint nvidia-smi \
    "$VLLM_IMAGE" >/dev/null 2>&1 ||
    die "Fresh container GPU check failed on Master."
}

preflight_worker() {
  ssh_worker \
    "command -v docker >/dev/null &&
     command -v tmux >/dev/null &&
     command -v nvidia-smi >/dev/null" ||
    die "Worker requires docker, tmux, and nvidia-smi."

  ssh_worker "docker info >/dev/null 2>&1" ||
    die "Docker unavailable on Worker."

  ssh_worker "nvidia-smi >/dev/null 2>&1" ||
    die "Host GPU check failed on Worker."

  ensure_image_worker

  ssh_worker \
    "docker run --rm --gpus all \
       --entrypoint nvidia-smi '$VLLM_IMAGE' >/dev/null 2>&1" ||
    die "Fresh container GPU check failed on Worker."
}

# ==============================================================================
# RAY CLUSTER
# ==============================================================================

node_env_args() {
  local node_ip="$1"
  local interface="$2"
  local -n output="$3"

  output=(
    -e "VLLM_HOST_IP=${node_ip}"
    -e "UCX_NET_DEVICES=${interface}"
    -e "NCCL_SOCKET_IFNAME=${interface}"
    -e "OMPI_MCA_btl_tcp_if_include=${interface}"
    -e "GLOO_SOCKET_IFNAME=${interface}"
    -e "TP_SOCKET_IFNAME=${interface}"
    -e "RAY_memory_monitor_refresh_ms=0"
    -e "MASTER_ADDR=${MASTER_IP}"
  )

  if [[ "$HF_HUB_OFFLINE" == "1" ]]; then
    output+=(
      -e HF_HUB_OFFLINE=1
      -e TRANSFORMERS_OFFLINE=1
    )
  fi

  if [[ -n "$HF_TOKEN" ]]; then
    output+=(-e "HF_TOKEN=${HF_TOKEN}")
  fi
}

head_foreground() {
  local interface
  local -a env_args

  interface="$(detect_interface "$MASTER_IP")"
  [[ -n "$interface" ]] ||
    die "Cannot detect Master high-speed interface."

  node_env_args "$MASTER_IP" "$interface" env_args

  exec bash "$RUN_CLUSTER" \
    "$VLLM_IMAGE" \
    "$MASTER_IP" \
    --head \
    "$HF_HOME" \
    "${env_args[@]}"
}

worker_foreground() {
  local interface
  local -a env_args

  interface="$(detect_interface "$WORKER_IP")"
  [[ -n "$interface" ]] ||
    die "Cannot detect Worker high-speed interface."

  node_env_args "$WORKER_IP" "$interface" env_args

  exec bash "$RUN_CLUSTER" \
    "$VLLM_IMAGE" \
    "$MASTER_IP" \
    --worker \
    "$HF_HOME" \
    "${env_args[@]}"
}

worker_start_internal() {
  tmux kill-session -t "$WORKER_SESSION" 2>/dev/null || true

  docker ps -aq \
    --filter "ancestor=${VLLM_IMAGE}" \
    --filter "name=node-" |
    xargs -r docker rm -f >/dev/null 2>&1 || true

  local command
  printf -v command 'exec bash %q _worker-foreground' "$SCRIPT_PATH"

  tmux new-session -d -s "$WORKER_SESSION" "$command"
}

ray_ready() {
  local container="$1"

  docker exec "$container" python3 -c '
import ray
import sys

ray.init(
    address="auto",
    ignore_reinit_error=True,
    logging_level="ERROR",
    log_to_driver=False,
)

nodes = sum(1 for node in ray.nodes() if node.get("Alive"))
gpus = float(ray.cluster_resources().get("GPU", 0))

sys.exit(0 if nodes >= 2 and gpus >= 2 else 1)
' >/dev/null 2>&1
}

wait_for_head() {
  local deadline=$((SECONDS + HEAD_WAIT_SECONDS))

  while (( SECONDS < deadline )); do
    local container
    container="$(head_container)"

    if [[ -n "$container" ]] &&
       docker exec "$container" ray status >/dev/null 2>&1 &&
       docker exec "$container" nvidia-smi >/dev/null 2>&1; then
      printf '%s\n' "$container"
      return
    fi

    sleep 3
  done

  tmux capture-pane -p -t "$HEAD_SESSION" -S -150 2>/dev/null || true
  die "Ray Head did not become ready."
}

wait_for_cluster() {
  local container="$1"
  local deadline=$((SECONDS + CLUSTER_WAIT_SECONDS))

  while (( SECONDS < deadline )); do
    if ray_ready "$container"; then
      docker exec "$container" ray status
      return
    fi
    sleep 4
  done

  docker exec "$container" ray status 2>/dev/null || true
  die "Ray cluster did not reach 2 nodes / 2 GPUs."
}

# ==============================================================================
# API
# ==============================================================================

api_foreground() {
  local container
  container="$(head_container)"
  [[ -n "$container" ]] || die "Head container not found."

  until ray_ready "$container"; do
    sleep 4
  done

  local -a args=(
    serve "$MODEL_ID"
    --served-model-name "$SERVED_MODEL_NAME"
    --quantization modelopt
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --distributed-executor-backend ray
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$MAX_NUM_SEQS"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --host "$API_HOST"
    --port "$API_PORT"
    --chat-template "$CHAT_TEMPLATE_CONTAINER"
    --default-chat-template-kwargs
      "{\"enable_thinking\":${DEFAULT_ENABLE_THINKING}}"
    --reasoning-parser "$REASONING_PARSER"
    --limit-mm-per-prompt
      "{\"image\":${MAX_IMAGES_PER_PROMPT},\"video\":${MAX_VIDEO_PER_PROMPT}}"
  )

  if [[ "$ENABLE_AUTO_TOOL_CHOICE" == "1" ]]; then
    args+=(
      --enable-auto-tool-choice
      --tool-call-parser "$TOOL_CALL_PARSER"
    )
  fi

  if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
    args+=(--enable-prefix-caching)
  fi

  if [[ "$ENABLE_CHUNKED_PREFILL" == "1" ]]; then
    args+=(--enable-chunked-prefill)
  fi

  if [[ -n "$API_KEY" ]]; then
    args+=(--api-key "$API_KEY")
  fi

  exec docker exec "$container" vllm "${args[@]}"
}

wait_for_api() {
  local deadline=$((SECONDS + API_WAIT_SECONDS))

  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 5 \
      "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
      return
    fi
    sleep 5
  done

  tmux capture-pane -p -t "$API_SESSION" -S -250 2>/dev/null || true
  die "API did not become healthy."
}

# ==============================================================================
# START / STOP
# ==============================================================================

stop_all() {
  local container
  container="$(head_container)"

  if [[ -n "$container" ]]; then
    docker exec "$container" \
      bash -lc "pkill -TERM -f '[v]llm serve' || true" \
      >/dev/null 2>&1 || true
  fi

  tmux kill-session -t "$API_SESSION" 2>/dev/null || true

  ssh_worker \
    "tmux kill-session -t '$WORKER_SESSION' 2>/dev/null || true;
     docker ps -aq \
       --filter 'ancestor=${VLLM_IMAGE}' \
       --filter 'name=node-' |
       xargs -r docker rm -f >/dev/null 2>&1 || true" ||
    true

  tmux kill-session -t "$HEAD_SESSION" 2>/dev/null || true

  docker ps -aq \
    --filter "ancestor=${VLLM_IMAGE}" \
    --filter "name=node-" |
    xargs -r docker rm -f >/dev/null 2>&1 || true

  log "Stack stopped"
}

start_all() {
  check_master
  ensure_assets

  preflight_master
  preflight_worker
  verify_files_local
  verify_worker_files
  copy_assets_worker

  stop_all

  log "Starting Ray Head"

  local command
  printf -v command 'exec bash %q _head-foreground' "$SCRIPT_PATH"
  tmux new-session -d -s "$HEAD_SESSION" "$command"

  local container
  container="$(wait_for_head)"
  log "Head container: $container"

  log "Starting Ray Worker from Master"
  ssh_worker "bash '$REMOTE_SCRIPT' _worker-start"

  wait_for_cluster "$container"

  log "Starting distributed Gemma 4 API"
  printf -v command 'exec bash %q _api-foreground' "$SCRIPT_PATH"
  tmux new-session -d -s "$API_SESSION" "$command"

  wait_for_api

  log "READY"
  echo "Base URL: $(public_base_url)"
  echo "Model:    ${SERVED_MODEL_NAME}"
  echo "Context:  ${MAX_MODEL_LEN}"
  echo "TP:       ${TENSOR_PARALLEL_SIZE}"
}

# ==============================================================================
# STATUS / LOGS
# ==============================================================================

show_status() {
  local container
  container="$(head_container)"

  echo "===== MASTER TMUX ====="
  tmux list-sessions 2>/dev/null || true

  echo
  echo "===== HEAD / RAY ====="
  if [[ -n "$container" ]]; then
    echo "Container: $container"
    docker exec "$container" ray status 2>/dev/null || true
    echo
    docker exec "$container" \
      bash -lc "pgrep -af '[v]llm serve' || true" \
      2>/dev/null || true
  else
    echo "Head container not running"
  fi

  echo
  echo "===== WORKER ====="
  ssh_worker \
    "tmux list-sessions 2>/dev/null || true;
     docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'" \
    2>/dev/null || echo "Worker unreachable"

  echo
  echo "===== API ====="
  if curl -fsS --max-time 5 \
    "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
    echo "API: HEALTHY"
    api_auth_args
    curl -sS "${API_AUTH_ARGS[@]}" \
      "http://127.0.0.1:${API_PORT}/v1/models"
    echo
  else
    echo "API: NOT READY"
  fi
}

show_logs() {
  local target="${1:-all}"

  case "$target" in
    head)
      tmux capture-pane -p -t "$HEAD_SESSION" -S -300
      ;;
    worker)
      ssh_worker "tmux capture-pane -p -t '$WORKER_SESSION' -S -300"
      ;;
    api)
      tmux capture-pane -p -t "$API_SESSION" -S -400
      ;;
    all)
      echo "===== HEAD ====="
      tmux capture-pane -p -t "$HEAD_SESSION" -S -120 2>/dev/null || true
      echo
      echo "===== WORKER ====="
      ssh_worker \
        "tmux capture-pane -p -t '$WORKER_SESSION' -S -120 2>/dev/null || true" ||
        true
      echo
      echo "===== API ====="
      tmux capture-pane -p -t "$API_SESSION" -S -200 2>/dev/null || true
      ;;
    *)
      die "logs target must be head, worker, api, or all"
      ;;
  esac
}

# ==============================================================================
# TESTS
# ==============================================================================

test_text() {
  api_auth_args
  curl -sS "${API_AUTH_ARGS[@]}" \
    "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data-binary @- <<EOF
{
  "model": "${SERVED_MODEL_NAME}",
  "messages": [
    {"role": "system", "content": "You are a precise software engineering assistant."},
    {"role": "user", "content": "Write a Python IPv4 validator with type hints and pytest tests."}
  ],
  "chat_template_kwargs": {"enable_thinking": false},
  "temperature": 0.2,
  "max_tokens": 2048
}
EOF
  echo
}

test_reasoning() {
  api_auth_args
  curl -sS "${API_AUTH_ARGS[@]}" \
    "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data-binary @- <<EOF
{
  "model": "${SERVED_MODEL_NAME}",
  "messages": [
    {"role": "user", "content": "Design a secure local coding-agent deployment and explain the tradeoffs."}
  ],
  "chat_template_kwargs": {"enable_thinking": true},
  "temperature": 1.0,
  "top_p": 0.95,
  "max_tokens": 8192
}
EOF
  echo
}

test_tools() {
  local choice="${1:-required}"
  [[ "$choice" == "required" || "$choice" == "auto" ]] ||
    die "Tool choice must be required or auto."

  api_auth_args
  curl -sS "${API_AUTH_ARGS[@]}" \
    "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data-binary @- <<EOF
{
  "model": "${SERVED_MODEL_NAME}",
  "messages": [
    {"role": "user", "content": "Inspect src/app.py with read_file before suggesting changes."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read a UTF-8 file from the workspace.",
        "strict": true,
        "parameters": {
          "type": "object",
          "properties": {"path": {"type": "string"}},
          "required": ["path"],
          "additionalProperties": false
        }
      }
    }
  ],
  "tool_choice": "${choice}",
  "parallel_tool_calls": false,
  "chat_template_kwargs": {"enable_thinking": false},
  "temperature": 0,
  "max_tokens": 1024
}
EOF
  echo
  echo "Expected: choices[0].message.tool_calls"
}

test_image_url() {
  local url="${1:-https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Christ_the_Redeemer_-_Rio_de_Janeiro%2C_Brazil.jpg/800px-Christ_the_Redeemer_-_Rio_de_Janeiro%2C_Brazil.jpg}"

  api_auth_args
  curl -sS "${API_AUTH_ARGS[@]}" \
    "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data-binary @- <<EOF
{
  "model": "${SERVED_MODEL_NAME}",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "${url}"}},
        {"type": "text", "text": "Describe this image and identify the city."}
      ]
    }
  ],
  "chat_template_kwargs": {"enable_thinking": false},
  "max_tokens": 2048
}
EOF
  echo
}

print_client_config() {
  local client_key="${API_KEY:-vllm-local}"

  cat <<EOF
Provider:          OpenAI Compatible
Base URL:          $(public_base_url)
API key:           ${client_key}
Model ID:          ${SERVED_MODEL_NAME}
Context window:    ${MAX_MODEL_LEN}
Max input tokens:  ${CLIENT_CONTEXT_TOKENS}
Max output tokens: ${CLIENT_MAX_OUTPUT_TOKENS}
Tensor parallel:   ${TENSOR_PARALLEL_SIZE}
Supports tools:    yes
Supports reasoning: yes
Supports images:   yes
Supports video:    yes
Supports audio:    no
EOF
}

show_help() {
  cat <<EOF
Usage:
Common options:
  --context TOKENS
  --port PORT
  --bind ADDRESS
  --advertise-ip ADDRESS
  --interface NAME
  --client-input TOKENS|auto
  --client-output TOKENS

  $(basename "$0") download
  $(basename "$0") verify-files
  $(basename "$0") sync-worker
  $(basename "$0") verify-worker
  $(basename "$0") start
  $(basename "$0") stop
  $(basename "$0") restart
  $(basename "$0") status
  $(basename "$0") logs [head|worker|api|all]
  $(basename "$0") test-text
  $(basename "$0") test-reasoning
  $(basename "$0") test-tools [required|auto]
  $(basename "$0") test-image-url [URL]
  $(basename "$0") client-config
EOF
}

parse_common_options "$@"
set -- "${REMAINING_ARGS[@]}"

case "${1:-help}" in
  download) download_model ;;
  verify-files) prepare_special_files; verify_files_local ;;
  sync-worker) sync_worker ;;
  verify-worker) verify_worker_files ;;
  start) start_all ;;
  stop) check_master; stop_all ;;
  restart) check_master; stop_all; start_all ;;
  status) check_master; show_status ;;
  logs) show_logs "${2:-all}" ;;
  test-text) test_text ;;
  test-reasoning) test_reasoning ;;
  test-tools) test_tools "${2:-required}" ;;
  test-image-url) test_image_url "${2:-}" ;;
  client-config) print_client_config ;;
  _head-foreground) head_foreground ;;
  _worker-start) worker_start_internal ;;
  _worker-foreground) worker_foreground ;;
  _api-foreground) api_foreground ;;
  network-info)
    network_info
    ;;
  help|-h|--help) show_help ;;
  *) show_help; exit 1 ;;
esac
