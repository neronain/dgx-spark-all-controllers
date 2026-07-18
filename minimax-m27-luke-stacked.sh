#!/usr/bin/env bash
#
# minimax-m27-luke-stacked.sh
#
# One-file Master controller for:
#   lukealonso/MiniMax-M2.7-NVFP4
#   2x NVIDIA DGX Spark, TP=2 over ConnectX/RoCE
#
# Runtime orchestration:
#   eugr/spark-vllm-docker, pinned to a tested repository commit.
#
# Commands:
#   ./minimax-m27-luke-stacked.sh prepare-runtime
#   ./minimax-m27-luke-stacked.sh runtime-info
#   ./minimax-m27-luke-stacked.sh download
#   ./minimax-m27-luke-stacked.sh verify-files
#   ./minimax-m27-luke-stacked.sh sync-worker
#   ./minimax-m27-luke-stacked.sh verify-worker
#   ./minimax-m27-luke-stacked.sh start
#   ./minimax-m27-luke-stacked.sh stop
#   ./minimax-m27-luke-stacked.sh restart
#   ./minimax-m27-luke-stacked.sh status
#   ./minimax-m27-luke-stacked.sh logs [api|head|worker|watchdog|all]
#   ./minimax-m27-luke-stacked.sh test-text
#   ./minimax-m27-luke-stacked.sh test-reasoning
#   ./minimax-m27-luke-stacked.sh test-tools [required|auto]
#   ./minimax-m27-luke-stacked.sh test-tool-loop
#   ./minimax-m27-luke-stacked.sh stress [minutes]
#   ./minimax-m27-luke-stacked.sh watchdog-start
#   ./minimax-m27-luke-stacked.sh watchdog-stop
#   ./minimax-m27-luke-stacked.sh client-config
#
set -Eeuo pipefail

SCRIPT_VERSION="${SCRIPT_VERSION:-3.0.0}"

# ==============================================================================
# USER CONFIGURATION
# ==============================================================================

MASTER_IP="${MASTER_IP:-10.100.152.1}"
WORKER_IP="${WORKER_IP:-10.100.152.2}"
SSH_USER="${SSH_USER:-neronain}"

MODEL_ID="lukealonso/MiniMax-M2.7-NVFP4"

# Pinned after the repository's tokenizer/weight-scale fix and calibration update.
MODEL_REVISION="db821d7a3ce29ee96d80a1cae88d878d8586b54e"

SERVED_MODEL_NAME="minimax-m2.7-luke-nvfp4"

# Current stable DGX Spark orchestration/build repository at bundle creation.
RUNTIME_REPO_URL="https://github.com/eugr/spark-vllm-docker.git"
RUNTIME_REPO_REF="8b00816d64890e25beec3bff0053f50d85075ce0"

RUNTIME_IMAGE_TAG="vllm-node"
CONTAINER_NAME="minimax_m27_luke"

# Conservative first-run settings for 2x 128 GB unified-memory nodes.
TENSOR_PARALLEL_SIZE="2"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="0.78"
MAX_NUM_SEQS="1"
MAX_NUM_BATCHED_TOKENS="8192"

KV_CACHE_DTYPE="fp8"
ATTENTION_BACKEND="flashinfer"
MOE_BACKEND="flashinfer_cutlass"
MAMBA_SSM_CACHE_DTYPE="float32"
QUANTIZATION="modelopt_fp4"

# Start without CUDA graphs and prefix caching. Enable only after stress tests.
ENFORCE_EAGER="1"
ENABLE_PREFIX_CACHING="0"
ENABLE_CHUNKED_PREFILL="1"

ENABLE_AUTO_TOOL_CHOICE="1"
TOOL_CALL_PARSER="minimax_m2"

# `minimax_m2` separates reasoning from final content and supports interleaved
# thinking between tool calls. Fallback: minimax_m2_append_think.
REASONING_PARSER="minimax_m2"

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
ADVERTISE_IP="${ADVERTISE_IP:-}"
ADVERTISE_INTERFACE="${ADVERTISE_INTERFACE:-}"
ROUTE_PROBE_IP="${ROUTE_PROBE_IP:-1.1.1.1}"
CLIENT_OVERHEAD_TOKENS="${CLIENT_OVERHEAD_TOKENS:-8192}"
API_KEY="${API_KEY:-}"

HF_TOKEN=""
HF_HUB_OFFLINE="1"

# Full byte-for-byte comparison before every start.
VERIFY_FULL_HASH_BEFORE_START="1"

# Dedicated Hugging Face downloader container.
HF_DOWNLOADER_IMAGE="python:3.12-slim"

# Runtime build/copy behavior.
BUILD_JOBS="12"
RUNTIME_BUILD_MODE="prebuilt"
# Values: prebuilt | wheels | source
# prebuilt: pull the runtime selected by the pinned orchestration repository.
# wheels:   build image from DGX Spark prebuilt wheels.
# source:   compile current pinned sources; slowest but most reproducible.

# Manual network overrides. Empty values are auto-detected.
ETH_IF=""
IB_IF=""

# Early OOM monitor inside cluster containers.
ENABLE_EARLYOOM="1"
EARLYOOM_ARGS="-M 1048576,262144 -s 70,50 -r 30"

# Health watchdog.
WATCHDOG_ENABLED="1"
WATCHDOG_INTERVAL_SECONDS="60"
WATCHDOG_FAILURE_THRESHOLD="5"
WATCHDOG_AUTO_RESTART="1"
WATCHDOG_MAX_RESTARTS="3"
WATCHDOG_RESTART_WINDOW_SECONDS="21600"

# Client token budget.
CLIENT_CONTEXT_TOKENS="${CLIENT_CONTEXT_TOKENS:-auto}"
CLIENT_MAX_OUTPUT_TOKENS="${CLIENT_MAX_OUTPUT_TOKENS:-8192}"

API_WAIT_SECONDS="2400"

# ==============================================================================
# PATHS / NAMES
# ==============================================================================

MASTER_HOME="/home/${SSH_USER}"
BASE_DIR="${MASTER_HOME}/minimax-m27-luke"
RUNTIME_DIR="${BASE_DIR}/spark-vllm-docker"
RUNTIME_LOCK="${BASE_DIR}/RUNTIME_LOCK.txt"
CLUSTER_CONFIG="${BASE_DIR}/cluster.env"
SPECIAL_DIR="${BASE_DIR}/special-files"
HF_HOME="${MASTER_HOME}/.cache/huggingface"

MODEL_CACHE_NAME="models--${MODEL_ID//\//--}"
MODEL_CACHE_PATH="${HF_HOME}/hub/${MODEL_CACHE_NAME}"
MODEL_SNAPSHOT_PATH="${MODEL_CACHE_PATH}/snapshots/${MODEL_REVISION}"

CHAT_TEMPLATE_HOST="${HF_HOME}/minimax-m27-luke-chat-template.jinja"
CHAT_TEMPLATE_CONTAINER="/root/.cache/huggingface/minimax-m27-luke-chat-template.jinja"

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"

API_SESSION="minimax-m27-api"
WATCHDOG_SESSION="minimax-m27-watchdog"

LOG_DIR="${BASE_DIR}/logs"
API_LOG="${LOG_DIR}/api.log"
WATCHDOG_LOG="${LOG_DIR}/watchdog.log"
WATCHDOG_STATE="${BASE_DIR}/watchdog-state.tsv"
RESTART_LOCK="${BASE_DIR}/restart.lock"

SSH_TARGET="${SSH_USER}@${WORKER_IP}"

# ==============================================================================
# LOGGING / HELPERS
# ==============================================================================

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

warn() {
  printf '\nWARNING: %s\n' "$*" >&2
}

die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 ||
    die "Missing command: $1"
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

detect_interface_for_ip() {
  local address="$1"

  ip -o -4 addr show |
    awk -v ip="$address" '$4 ~ ("^" ip "/") {print $2; exit}'
}

detect_ib_interfaces() {
  if command -v ibdev2netdev >/dev/null 2>&1; then
    ibdev2netdev |
      awk 'tolower($NF) == "up" {print $1}' |
      paste -sd, -
    return
  fi

  if command -v rdma >/dev/null 2>&1; then
    rdma link show |
      awk -F'[: /]+' '/state ACTIVE/ {print $3}' |
      sort -u |
      paste -sd, -
    return
  fi

  printf ''
}

check_running_on_master() {
  local interface
  interface="$(detect_interface_for_ip "$MASTER_IP")"

  [[ -n "$interface" ]] ||
    die "This host does not own MASTER_IP=${MASTER_IP}. Run on Master."
}

ensure_passwordless_ssh() {
  ssh_worker "true" ||
    die "Passwordless SSH to ${SSH_TARGET} is required."
}

api_auth_args() {
  API_AUTH_ARGS=()

  if [[ -n "$API_KEY" ]]; then
    API_AUTH_ARGS=(-H "Authorization: Bearer ${API_KEY}")
  fi
}

port_in_use() {
  timeout 1 bash -c \
    "</dev/tcp/127.0.0.1/${API_PORT}" 2>/dev/null
}

api_healthy() {
  curl -fsS \
    --max-time 8 \
    "http://127.0.0.1:${API_PORT}/health" \
    >/dev/null 2>&1
}

wait_for_api() {
  local deadline=$((SECONDS + API_WAIT_SECONDS))

  log "Waiting for API health endpoint"

  while (( SECONDS < deadline )); do
    if api_healthy; then
      return
    fi

    if ! tmux has-session -t "$API_SESSION" 2>/dev/null; then
      tail -n 350 "$API_LOG" 2>/dev/null || true
      die "API launcher exited before the server became healthy."
    fi

    sleep 5
  done

  tail -n 350 "$API_LOG" 2>/dev/null || true
  die "API did not become healthy within ${API_WAIT_SECONDS}s."
}

resolve_network_config() {
  local detected_eth
  local detected_ib

  detected_eth="$ETH_IF"
  detected_ib="$IB_IF"

  if [[ -z "$detected_eth" ]]; then
    detected_eth="$(detect_interface_for_ip "$MASTER_IP")"
  fi

  [[ -n "$detected_eth" ]] ||
    die "Cannot detect Ethernet/ConnectX interface for ${MASTER_IP}."

  if [[ -z "$detected_ib" ]]; then
    detected_ib="$(detect_ib_interfaces)"
  fi

  mkdir -p "$BASE_DIR"

  cat > "$CLUSTER_CONFIG" <<EOF
CLUSTER_NODES=${MASTER_IP},${WORKER_IP}
COPY_HOSTS=${WORKER_IP}
LOCAL_IP=${MASTER_IP}
ETH_IF=${detected_eth}
EOF

  if [[ -n "$detected_ib" ]]; then
    printf 'IB_IF=%s\n' "$detected_ib" >> "$CLUSTER_CONFIG"
  fi

  log "Cluster config written: $CLUSTER_CONFIG"
  cat "$CLUSTER_CONFIG"

  ssh_worker \
    "ip -o -4 addr show |
       awk -v target='${WORKER_IP}' '\''$4 ~ ("^" target "/") { found=1 } END { exit(found ? 0 : 1) }'\''" ||
    die "Worker does not own ${WORKER_IP}."

  ssh_worker \
    "ip link show '${detected_eth}' >/dev/null 2>&1" ||
    die "Worker does not have interface ${detected_eth}."

  if [[ -n "$detected_ib" ]]; then
    warn "IB_IF=${detected_ib} was detected on Master. Verify device names match Worker."
  fi
}

# ==============================================================================
# RUNTIME PREPARATION / VERIFICATION
# ==============================================================================

checkout_runtime_repo() {
  require_command git

  mkdir -p "$BASE_DIR"

  if [[ ! -d "${RUNTIME_DIR}/.git" ]]; then
    log "Cloning DGX Spark vLLM runtime repository"
    git clone "$RUNTIME_REPO_URL" "$RUNTIME_DIR"
  fi

  log "Checking out pinned runtime repository revision"

  git -C "$RUNTIME_DIR" fetch origin --tags --prune
  git -C "$RUNTIME_DIR" checkout --detach "$RUNTIME_REPO_REF"
  git -C "$RUNTIME_DIR" reset --hard "$RUNTIME_REPO_REF"
  git -C "$RUNTIME_DIR" clean -fd

  local actual
  actual="$(git -C "$RUNTIME_DIR" rev-parse HEAD)"

  [[ "$actual" == "$RUNTIME_REPO_REF" ]] ||
    die "Runtime repository revision mismatch: ${actual}"
}

write_runtime_lock() {
  local local_id
  local worker_id
  local local_digest
  local worker_digest
  local local_versions

  local_id="$(docker image inspect "$RUNTIME_IMAGE_TAG" --format '{{.Id}}')"
  worker_id="$(
    ssh_worker \
      "docker image inspect '$RUNTIME_IMAGE_TAG' --format '{{.Id}}'"
  )"

  local_digest="$(
    docker image inspect "$RUNTIME_IMAGE_TAG" \
      --format '{{join .RepoDigests ","}}' 2>/dev/null || true
  )"

  worker_digest="$(
    ssh_worker \
      "docker image inspect '$RUNTIME_IMAGE_TAG' \
         --format '{{join .RepoDigests \",\"}}' 2>/dev/null || true"
  )"

  local_versions="$(
    docker run --rm \
      --entrypoint python3 \
      "$RUNTIME_IMAGE_TAG" \
      -c '
import importlib.metadata as m
for package in ("vllm", "flashinfer-python", "torch", "transformers"):
    try:
        print(f"{package}={m.version(package)}")
    except Exception:
        print(f"{package}=unknown")
'
  )"

  cat > "$RUNTIME_LOCK" <<EOF
runtime_repo=${RUNTIME_REPO_URL}
runtime_repo_commit=${RUNTIME_REPO_REF}
runtime_image_tag=${RUNTIME_IMAGE_TAG}
master_image_id=${local_id}
worker_image_id=${worker_id}
master_repo_digests=${local_digest}
worker_repo_digests=${worker_digest}
model_id=${MODEL_ID}
model_revision=${MODEL_REVISION}
generated_at=$(date --iso-8601=seconds)
${local_versions}
EOF

  log "Runtime lock written: $RUNTIME_LOCK"
  cat "$RUNTIME_LOCK"
}

verify_runtime() {
  require_command docker
  require_command git

  [[ -d "${RUNTIME_DIR}/.git" ]] ||
    die "Runtime repository is missing. Run: $0 prepare-runtime"

  local actual
  actual="$(git -C "$RUNTIME_DIR" rev-parse HEAD)"

  [[ "$actual" == "$RUNTIME_REPO_REF" ]] ||
    die "Runtime repository is not pinned to ${RUNTIME_REPO_REF}."

  docker image inspect "$RUNTIME_IMAGE_TAG" >/dev/null 2>&1 ||
    die "Runtime image missing on Master."

  ssh_worker \
    "docker image inspect '$RUNTIME_IMAGE_TAG' >/dev/null 2>&1" ||
    die "Runtime image missing on Worker."

  local local_id
  local worker_id

  local_id="$(docker image inspect "$RUNTIME_IMAGE_TAG" --format '{{.Id}}')"
  worker_id="$(
    ssh_worker \
      "docker image inspect '$RUNTIME_IMAGE_TAG' --format '{{.Id}}'"
  )"

  [[ "$local_id" == "$worker_id" ]] ||
    die "Master/Worker runtime image IDs differ."

  docker run --rm \
    --gpus all \
    --entrypoint nvidia-smi \
    "$RUNTIME_IMAGE_TAG" >/dev/null 2>&1 ||
    die "Fresh runtime container cannot access Master GPU."

  ssh_worker \
    "docker run --rm --gpus all \
       --entrypoint nvidia-smi '$RUNTIME_IMAGE_TAG' >/dev/null 2>&1" ||
    die "Fresh runtime container cannot access Worker GPU."

  docker run --rm \
    --entrypoint python3 \
    "$RUNTIME_IMAGE_TAG" \
    -c '
from vllm.tool_parsers import ToolParserManager
from vllm.reasoning import ReasoningParserManager

tool_names = set(ToolParserManager.tool_parsers)
reason_names = set(ReasoningParserManager.reasoning_parsers)

required_tools = {"minimax_m2"}
required_reasoning = {"minimax_m2", "minimax_m2_append_think"}

missing = (
    [f"tool:{x}" for x in required_tools - tool_names]
    + [f"reasoning:{x}" for x in required_reasoning - reason_names]
)
if missing:
    raise SystemExit("Missing parsers: " + ", ".join(missing))

print("MiniMax tool/reasoning parsers found")
' ||
    die "Runtime parser verification failed."

  log "Runtime is identical and GPU-capable on both nodes"
}

prepare_runtime() {
  check_running_on_master
  ensure_passwordless_ssh

  require_command docker
  require_command git
  require_command ssh

  docker info >/dev/null 2>&1 ||
    die "Docker unavailable on Master."

  ssh_worker "docker info >/dev/null 2>&1" ||
    die "Docker unavailable on Worker."

  checkout_runtime_repo
  resolve_network_config

  log "Preparing and copying DGX Spark runtime image"

  case "$RUNTIME_BUILD_MODE" in
    prebuilt)
      (
        cd "$RUNTIME_DIR"
        ./build-and-copy.sh \
          --config "$CLUSTER_CONFIG" \
          -c "$WORKER_IP" \
          -u "$SSH_USER"
      )
      ;;
    wheels)
      (
        cd "$RUNTIME_DIR"
        ./build-and-copy.sh \
          --config "$CLUSTER_CONFIG" \
          --use-wheels \
          -j "$BUILD_JOBS" \
          -c "$WORKER_IP" \
          -u "$SSH_USER"
      )
      ;;
    source)
      (
        cd "$RUNTIME_DIR"
        ./build-and-copy.sh \
          --config "$CLUSTER_CONFIG" \
          --rebuild-vllm \
          --rebuild-flashinfer \
          -j "$BUILD_JOBS" \
          -c "$WORKER_IP" \
          -u "$SSH_USER"
      )
      ;;
    *)
      die "RUNTIME_BUILD_MODE must be prebuilt, wheels, or source."
      ;;
  esac

  verify_runtime
  write_runtime_lock
}

runtime_info() {
  check_running_on_master
  ensure_passwordless_ssh

  echo "===== RUNTIME REPOSITORY ====="
  if [[ -d "${RUNTIME_DIR}/.git" ]]; then
    git -C "$RUNTIME_DIR" log -1 --oneline
    git -C "$RUNTIME_DIR" status --short --branch
  else
    echo "Not prepared"
  fi

  echo
  echo "===== MASTER IMAGE ====="
  docker image inspect "$RUNTIME_IMAGE_TAG" \
    --format 'ID={{.Id}} Digests={{join .RepoDigests ","}}' \
    2>/dev/null || true

  echo
  echo "===== WORKER IMAGE ====="
  ssh_worker \
    "docker image inspect '$RUNTIME_IMAGE_TAG' \
      --format 'ID={{.Id}} Digests={{join .RepoDigests \",\"}}'" \
    2>/dev/null || true

  echo
  echo "===== PACKAGE VERSIONS ====="
  docker run --rm \
    --entrypoint python3 \
    "$RUNTIME_IMAGE_TAG" \
    -c '
import importlib.metadata as m
for package in ("vllm", "flashinfer-python", "torch", "transformers"):
    try:
        print(f"{package}={m.version(package)}")
    except Exception:
        print(f"{package}=unknown")
' 2>/dev/null || true

  echo
  echo "===== LOCK ====="
  cat "$RUNTIME_LOCK" 2>/dev/null || echo "No runtime lock"
}

# ==============================================================================
# SELECTIVE MODEL DOWNLOAD
# ==============================================================================

download_model() {
  check_running_on_master
  require_command docker

  docker image inspect "$RUNTIME_IMAGE_TAG" >/dev/null 2>&1 ||
    die "Prepare the runtime before downloading: $0 prepare-runtime"

  mkdir -p "$HF_HOME" "$BASE_DIR" "$SPECIAL_DIR"

  log "Selective download of ${MODEL_ID}@${MODEL_REVISION}"
  log "Only indexed weight shards and required runtime/tokenizer files are fetched."

  local -a token_env=()
  if [[ -n "$HF_TOKEN" ]]; then
    token_env=(-e "HF_TOKEN=${HF_TOKEN}")
  fi

  docker run --rm \
    --user "$(id -u "$SSH_USER"):$(id -g "$SSH_USER")" \
    -e HOME=/tmp \
    -e HF_HOME=/cache \
    -e MODEL_ID="$MODEL_ID" \
    -e MODEL_REVISION="$MODEL_REVISION" \
    "${token_env[@]}" \
    -v "${HF_HOME}:/cache" \
    --entrypoint /bin/bash \
    "$RUNTIME_IMAGE_TAG" \
    -lc '
set -Eeuo pipefail

python3 - <<'"'"'PY'"'"'
from __future__ import annotations

import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

repo_id = os.environ["MODEL_ID"]
revision = os.environ["MODEL_REVISION"]

index_path = Path(
    hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors.index.json",
        revision=revision,
        cache_dir="/cache",
    )
)

index = json.loads(index_path.read_text())
weight_map = index.get("weight_map", {})
if not weight_map:
    raise SystemExit("model.safetensors.index.json has no weight_map")

shards = sorted(set(weight_map.values()))
repo_files = HfApi().list_repo_files(repo_id=repo_id, revision=revision)

small_suffixes = {
    ".json",
    ".py",
    ".jinja",
    ".txt",
    ".model",
    ".tiktoken",
    ".md",
}

excluded_prefixes = (
    "amax",
    "model-inputscales",
    "model-codex",
)

allow = set(shards)

for filename in repo_files:
    name = Path(filename).name
    suffix = Path(filename).suffix.lower()

    if filename.endswith(".bak"):
        continue

    if name.startswith(excluded_prefixes):
        continue

    if suffix in small_suffixes:
        allow.add(filename)

required = {
    "config.json",
    "configuration_minimax_m2.py",
    "generation_config.json",
    "hf_quant_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
}

missing_repo = sorted(required - set(repo_files))
if missing_repo:
    raise SystemExit("Repository is missing required files: " + ", ".join(missing_repo))

print(f"Downloading {len(shards)} indexed shards")
print(f"Downloading {len(allow) - len(shards)} small runtime/tokenizer files")
print("Excluded *.bak and calibration/input-scale artifacts")

snapshot = snapshot_download(
    repo_id=repo_id,
    revision=revision,
    allow_patterns=sorted(allow),
    cache_dir="/cache",
)

print(f"Snapshot: {snapshot}")
PY
'

  prepare_special_files
  verify_files_local

  log "Selective model download completed"
  du -sh "$MODEL_CACHE_PATH"
}

prepare_special_files() {
  [[ -d "$MODEL_SNAPSHOT_PATH" ]] ||
    die "Pinned model snapshot is missing: $MODEL_SNAPSHOT_PATH"

  mkdir -p "$SPECIAL_DIR"

  local -a files=(
    chat_template.jinja
    config.json
    configuration_minimax_m2.py
    generation_config.json
    hf_quant_config.json
    model.safetensors.index.json
    tokenizer.json
    tokenizer_config.json
  )

  local optional
  local -a optional_files=(
    modeling_minimax_m2.py
    added_tokens.json
    special_tokens_map.json
    merges.txt
    vocab.json
  )

  local file
  for file in "${files[@]}"; do
    [[ -e "${MODEL_SNAPSHOT_PATH}/${file}" ]] ||
      die "Required model file missing: $file"
    cp -L "${MODEL_SNAPSHOT_PATH}/${file}" "${SPECIAL_DIR}/${file}"
  done

  for optional in "${optional_files[@]}"; do
    if [[ -e "${MODEL_SNAPSHOT_PATH}/${optional}" ]]; then
      cp -L \
        "${MODEL_SNAPSHOT_PATH}/${optional}" \
        "${SPECIAL_DIR}/${optional}"
    fi
  done

  cp -L \
    "${MODEL_SNAPSHOT_PATH}/chat_template.jinja" \
    "$CHAT_TEMPLATE_HOST"

  (
    cd "$SPECIAL_DIR"
    find . -maxdepth 1 -type f \
      ! -name SHA256SUMS \
      -printf '%P\n' |
      sort |
      xargs sha256sum > SHA256SUMS
  )
}

verify_files_local() {
  require_command python3
  require_command sha256sum

  [[ -d "$MODEL_SNAPSHOT_PATH" ]] ||
    die "Pinned model snapshot is missing. Run: $0 download"

  [[ -f "${SPECIAL_DIR}/SHA256SUMS" ]] ||
    prepare_special_files

  python3 - "$MODEL_SNAPSHOT_PATH" "$SPECIAL_DIR" "$MODEL_REVISION" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

snapshot = Path(sys.argv[1])
special = Path(sys.argv[2])
expected_revision = sys.argv[3]

if snapshot.name != expected_revision:
    raise SystemExit(
        f"Snapshot revision mismatch: {snapshot.name} != {expected_revision}"
    )

required = [
    "chat_template.jinja",
    "config.json",
    "configuration_minimax_m2.py",
    "generation_config.json",
    "hf_quant_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
]

missing = [name for name in required if not (snapshot / name).exists()]
if missing:
    raise SystemExit("Missing required files: " + ", ".join(missing))

index = json.loads((snapshot / "model.safetensors.index.json").read_text())
weight_map = index.get("weight_map", {})
if not weight_map:
    raise SystemExit("Safetensors index has no weight_map")

shards = sorted(set(weight_map.values()))
missing_shards = [name for name in shards if not (snapshot / name).exists()]
if missing_shards:
    raise SystemExit("Missing shards: " + ", ".join(missing_shards))

broken = [
    path
    for path in snapshot.rglob("*")
    if path.is_symlink() and not path.exists()
]
if broken:
    raise SystemExit(
        "Broken symlinks:\n" + "\n".join(str(path) for path in broken[:100])
    )

config = json.loads((special / "config.json").read_text())

expected_config = {
    "model_type": "minimax_m2",
    "max_position_embeddings": 196608,
    "num_hidden_layers": 62,
    "num_local_experts": 256,
    "num_experts_per_tok": 8,
}

for key, expected in expected_config.items():
    actual = config.get(key)
    if actual != expected:
        raise SystemExit(
            f"Unexpected config {key}: expected {expected!r}, got {actual!r}"
        )

auto_map = config.get("auto_map", {})
if not auto_map:
    raise SystemExit("config.json has no auto_map; remote code may be incomplete")

template = (special / "chat_template.jinja").read_text()
for marker in (
    "<minimax:tool_call>",
    "</minimax:tool_call>",
    "<invoke name=",
    "<parameter name=",
    "<think>",
    "</think>",
    "message.tool_calls",
    "message.role == 'tool'",
):
    if marker not in template:
        raise SystemExit(f"Chat template missing marker: {marker}")

generation = json.loads((special / "generation_config.json").read_text())
expected_generation = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 40,
}

for key, expected in expected_generation.items():
    actual = generation.get(key)
    if actual != expected:
        raise SystemExit(
            f"Unexpected generation config {key}: {actual!r}"
        )

quant = json.loads((special / "hf_quant_config.json").read_text())
if not quant:
    raise SystemExit("hf_quant_config.json is empty")

forbidden = sorted(
    path.name
    for path in snapshot.iterdir()
    if path.name.endswith(".bak")
    or path.name.startswith(("amax", "model-inputscales", "model-codex"))
)

print(f"Revision:          {snapshot.name}")
print(f"Indexed tensors:   {len(weight_map)}")
print(f"Referenced shards: {len(shards)}")
print("Architecture:      MiniMax M2, 62 layers, 256 experts, top-8")
print("Context:           196608")
print("Tool template:     MiniMax XML markers found")
print("Reasoning:         think markers found")
print("Sampling:          temperature=1.0 top_p=0.95 top_k=40")
print("Quant config:      present")

if forbidden:
    print("WARNING: unused artifacts already exist in this snapshot:")
    for name in forbidden:
        print(f"  {name}")
    print("They are not required or synchronized by this controller.")
PY

  (
    cd "$SPECIAL_DIR"
    sha256sum -c SHA256SUMS
  )

  [[ -s "$CHAT_TEMPLATE_HOST" ]] ||
    die "Runtime chat template is missing."

  log "Pinned model files and special files verified on Master"
}

# ==============================================================================
# MASTER -> WORKER SYNC AND FULL COMPARISON
# ==============================================================================

write_manifest_helper() {
  mkdir -p "$BASE_DIR"

  cat > "${BASE_DIR}/cache_manifest.py" <<'PY'
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

for path in sorted(root.rglob("*"), key=lambda p: str(p.relative_to(root))):
    rel = str(path.relative_to(root))

    if path.is_symlink():
        print(f"L\t{rel}\t{os.readlink(path)}")
        continue

    if path.is_dir():
        print(f"D\t{rel}\t")
        continue

    if path.is_file():
        size = path.stat().st_size

        if args.quick:
            digest = "-"
        else:
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(
                    lambda: handle.read(16 * 1024 * 1024),
                    b"",
                ):
                    hasher.update(chunk)
            digest = hasher.hexdigest()

        print(f"F\t{rel}\t{size}\t{digest}")
PY

  chmod 0755 "${BASE_DIR}/cache_manifest.py"
}

sync_worker() {
  check_running_on_master
  ensure_passwordless_ssh

  require_command rsync
  require_command scp

  verify_files_local
  write_manifest_helper

  ssh_worker \
    "mkdir -p '${MODEL_CACHE_PATH}' '${HF_HOME}' '${BASE_DIR}'"

  log "Synchronizing only this model's Hugging Face cache to Worker"

  rsync -aH \
    --delete \
    --partial \
    --info=progress2 \
    "${MODEL_CACHE_PATH}/" \
    "${SSH_TARGET}:${MODEL_CACHE_PATH}/"

  log "Synchronizing runtime chat template and manifest helper"

  scp -q \
    "$CHAT_TEMPLATE_HOST" \
    "${SSH_TARGET}:${CHAT_TEMPLATE_HOST}"

  scp -q \
    "${BASE_DIR}/cache_manifest.py" \
    "${SSH_TARGET}:${BASE_DIR}/cache_manifest.py"

  ssh_worker \
    "chmod 0755 '${BASE_DIR}/cache_manifest.py'"

  verify_worker_files

  log "Worker synchronization completed"
}

verify_worker_files() {
  check_running_on_master
  ensure_passwordless_ssh

  verify_files_local
  write_manifest_helper

  ssh_worker \
    "test -d '${MODEL_SNAPSHOT_PATH}'" ||
    die "Pinned snapshot is missing on Worker. Run: $0 sync-worker"

  scp -q \
    "${BASE_DIR}/cache_manifest.py" \
    "${SSH_TARGET}:${BASE_DIR}/cache_manifest.py"

  local mode_arg=""
  if [[ "$VERIFY_FULL_HASH_BEFORE_START" != "1" ]]; then
    mode_arg="--quick"
  fi

  local temp_dir
  temp_dir="$(mktemp -d)"

  local master_manifest="${temp_dir}/master.manifest"
  local worker_manifest="${temp_dir}/worker.manifest"
  local remote_manifest="${BASE_DIR}/worker.manifest"

  if [[ "$VERIFY_FULL_HASH_BEFORE_START" == "1" ]]; then
    log "Building full SHA-256 manifests on Master and Worker"
  else
    log "Building quick path/symlink/size manifests on both nodes"
  fi

  ssh_worker \
    "python3 '${BASE_DIR}/cache_manifest.py' \
       '${MODEL_CACHE_PATH}' ${mode_arg} > '${remote_manifest}'" &
  local worker_pid=$!

  python3 \
    "${BASE_DIR}/cache_manifest.py" \
    "$MODEL_CACHE_PATH" \
    $mode_arg \
    > "$master_manifest"

  wait "$worker_pid"

  scp -q \
    "${SSH_TARGET}:${remote_manifest}" \
    "$worker_manifest"

  if ! diff -u "$master_manifest" "$worker_manifest"; then
    rm -rf "$temp_dir"
    die "Master and Worker model caches differ. Run: $0 sync-worker"
  fi

  local master_template_hash
  local worker_template_hash

  master_template_hash="$(
    sha256sum "$CHAT_TEMPLATE_HOST" |
      awk '{print $1}'
  )"

  worker_template_hash="$(
    ssh_worker \
      "sha256sum '$CHAT_TEMPLATE_HOST' | awk '{print \$1}'"
  )"

  [[ "$master_template_hash" == "$worker_template_hash" ]] ||
    die "Runtime chat templates differ between nodes."

  rm -rf "$temp_dir"

  log "Master and Worker model caches are identical"
}

# ==============================================================================
# PREFLIGHT / CLUSTER LAUNCH
# ==============================================================================

preflight() {
  check_running_on_master
  ensure_passwordless_ssh

  require_command docker
  require_command tmux
  require_command curl
  require_command timeout
  require_command nvidia-smi
  require_command flock

  docker info >/dev/null 2>&1 ||
    die "Docker unavailable on Master."

  ssh_worker "docker info >/dev/null 2>&1" ||
    die "Docker unavailable on Worker."

  nvidia-smi >/dev/null 2>&1 ||
    die "Host GPU check failed on Master."

  ssh_worker "nvidia-smi >/dev/null 2>&1" ||
    die "Host GPU check failed on Worker."

  verify_runtime
  verify_files_local
  verify_worker_files
  resolve_network_config
}

launcher_base_args() {
  local -n output="$1"

  output=(
    --config "$CLUSTER_CONFIG"
    --nodes "${MASTER_IP},${WORKER_IP}"
    --eth-if "$(awk -F= '/^ETH_IF=/{print $2}' "$CLUSTER_CONFIG")"
    --ray
    -t "$RUNTIME_IMAGE_TAG"
    --name "$CONTAINER_NAME"
  )

  local configured_ib
  configured_ib="$(awk -F= '/^IB_IF=/{print $2}' "$CLUSTER_CONFIG" || true)"

  if [[ -n "$configured_ib" ]]; then
    output+=(--ib-if "$configured_ib")
  fi

  if [[ "$ENABLE_EARLYOOM" == "1" ]]; then
    output+=(
      --earlyoom
      --earlyoom-args "$EARLYOOM_ARGS"
    )
  fi

  output+=(
    -e VLLM_USE_FLASHINFER_MOE_FP4=1
    -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    -e VLLM_FLASHINFER_MOE_BACKEND=throughput
    -e VLLM_FLOAT32_MATMUL_PRECISION=high
    -e OMP_NUM_THREADS=8
    -e SAFETENSORS_FAST_GPU=1
    -e TOKENIZERS_PARALLELISM=false
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

api_foreground() {
  local -a launcher
  local -a serve_args

  launcher_base_args launcher

  serve_args=(
    vllm serve "$MODEL_ID"
    --revision "$MODEL_REVISION"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$API_HOST"
    --port "$API_PORT"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --distributed-executor-backend ray
    --quantization "$QUANTIZATION"
    --dtype auto
    --trust-remote-code
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --attention-backend "$ATTENTION_BACKEND"
    --moe-backend "$MOE_BACKEND"
    --mamba-ssm-cache-dtype "$MAMBA_SSM_CACHE_DTYPE"
    --disable-custom-all-reduce
    --chat-template "$CHAT_TEMPLATE_CONTAINER"
    --enable-auto-tool-choice
    --tool-call-parser "$TOOL_CALL_PARSER"
    --reasoning-parser "$REASONING_PARSER"
  )

  if [[ "$ENFORCE_EAGER" == "1" ]]; then
    serve_args+=(--enforce-eager)
  fi

  if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
    serve_args+=(--enable-prefix-caching)
  fi

  if [[ "$ENABLE_CHUNKED_PREFILL" == "1" ]]; then
    serve_args+=(--enable-chunked-prefill)
  fi

  if [[ -n "$API_KEY" ]]; then
    serve_args+=(--api-key "$API_KEY")
  fi

  cd "$RUNTIME_DIR"

  exec ./launch-cluster.sh \
    "${launcher[@]}" \
    exec \
    "${serve_args[@]}"
}

stop_cluster_only() {
  if [[ ! -x "${RUNTIME_DIR}/launch-cluster.sh" ]]; then
    return
  fi

  local -a launcher
  launcher_base_args launcher

  (
    cd "$RUNTIME_DIR"
    ./launch-cluster.sh \
      "${launcher[@]}" \
      stop
  ) || true

  tmux kill-session -t "$API_SESSION" 2>/dev/null || true
}

start_cluster_only() {
  mkdir -p "$LOG_DIR"
  : > "$API_LOG"

  if tmux has-session -t "$API_SESSION" 2>/dev/null; then
    tmux kill-session -t "$API_SESSION"
  fi

  if port_in_use; then
    die "Port ${API_PORT} is already in use."
  fi

  local command
  printf -v command \
    'exec bash %q _api-foreground >> %q 2>&1' \
    "$SCRIPT_PATH" \
    "$API_LOG"

  tmux new-session \
    -d \
    -s "$API_SESSION" \
    "$command"

  wait_for_api

  log "MiniMax M2.7 API is ready"
  echo "Base URL: $(public_base_url)"
  echo "Model:    ${SERVED_MODEL_NAME}"
  echo "Context:  ${MAX_MODEL_LEN}"
  echo "TP:       ${TENSOR_PARALLEL_SIZE}"
}

start_all() {
  preflight
  stop_cluster_only
  start_cluster_only

  if [[ "$WATCHDOG_ENABLED" == "1" ]]; then
    start_watchdog
  fi
}

stop_all() {
  stop_watchdog
  stop_cluster_only
  log "Stopped"
}

restart_all() {
  stop_all
  start_all
}

# ==============================================================================
# WATCHDOG
# ==============================================================================

watchdog_record_restart() {
  local now
  now="$(date +%s)"

  touch "$WATCHDOG_STATE"

  awk -v now="$now" -v window="$WATCHDOG_RESTART_WINDOW_SECONDS" '
    NF >= 1 && now - $1 <= window {print}
  ' "$WATCHDOG_STATE" > "${WATCHDOG_STATE}.tmp"

  mv "${WATCHDOG_STATE}.tmp" "$WATCHDOG_STATE"

  printf '%s\t%s\n' "$now" "$(date --iso-8601=seconds)" >> "$WATCHDOG_STATE"
}

watchdog_restart_count() {
  local now
  now="$(date +%s)"

  if [[ ! -f "$WATCHDOG_STATE" ]]; then
    echo 0
    return
  fi

  awk -v now="$now" -v window="$WATCHDOG_RESTART_WINDOW_SECONDS" '
    NF >= 1 && now - $1 <= window {count++}
    END {print count + 0}
  ' "$WATCHDOG_STATE"
}

restart_from_watchdog() {
  require_command flock

  exec 9>"$RESTART_LOCK"

  if ! flock -n 9; then
    warn "Another restart operation already owns the lock."
    return
  fi

  local count
  count="$(watchdog_restart_count)"

  if (( count >= WATCHDOG_MAX_RESTARTS )); then
    warn "Watchdog restart limit reached: ${count}/${WATCHDOG_MAX_RESTARTS}"
    return 1
  fi

  watchdog_record_restart

  log "Watchdog is restarting the cluster"

  stop_cluster_only
  preflight
  start_cluster_only
}

watchdog_foreground() {
  mkdir -p "$LOG_DIR"

  local failures=0

  log "Watchdog started"

  while true; do
    if api_healthy; then
      if (( failures > 0 )); then
        log "API recovered after ${failures} failed health checks"
      fi
      failures=0
    else
      failures=$((failures + 1))
      warn "API health check failed (${failures}/${WATCHDOG_FAILURE_THRESHOLD})"

      if (( failures >= WATCHDOG_FAILURE_THRESHOLD )); then
        if [[ "$WATCHDOG_AUTO_RESTART" == "1" ]]; then
          if "$SCRIPT_PATH" _restart-from-watchdog; then
            failures=0
          else
            warn "Automatic restart failed or restart limit was reached"
            failures=0
          fi
        else
          warn "Automatic restart is disabled"
          failures=0
        fi
      fi
    fi

    sleep "$WATCHDOG_INTERVAL_SECONDS"
  done
}

start_watchdog() {
  require_command tmux

  mkdir -p "$LOG_DIR"
  touch "$WATCHDOG_LOG"

  if tmux has-session -t "$WATCHDOG_SESSION" 2>/dev/null; then
    return
  fi

  local command
  printf -v command \
    'exec bash %q _watchdog >> %q 2>&1' \
    "$SCRIPT_PATH" \
    "$WATCHDOG_LOG"

  tmux new-session \
    -d \
    -s "$WATCHDOG_SESSION" \
    "$command"

  log "Watchdog session started"
}

stop_watchdog() {
  tmux kill-session -t "$WATCHDOG_SESSION" 2>/dev/null || true
}

watchdog_status() {
  echo "Enabled:             $WATCHDOG_ENABLED"
  echo "Auto restart:        $WATCHDOG_AUTO_RESTART"
  echo "Interval:            ${WATCHDOG_INTERVAL_SECONDS}s"
  echo "Failure threshold:   $WATCHDOG_FAILURE_THRESHOLD"
  echo "Maximum restarts:    $WATCHDOG_MAX_RESTARTS"
  echo "Restart window:      ${WATCHDOG_RESTART_WINDOW_SECONDS}s"

  if tmux has-session -t "$WATCHDOG_SESSION" 2>/dev/null; then
    echo "Session:             RUNNING"
  else
    echo "Session:             STOPPED"
  fi

  echo "Recent restarts:     $(watchdog_restart_count)"
  echo "Log:                 $WATCHDOG_LOG"
}

# ==============================================================================
# STATUS / LOGS
# ==============================================================================

show_status() {
  check_running_on_master
  ensure_passwordless_ssh

  echo "===== CONFIG ====="
  echo "Model:             $MODEL_ID"
  echo "Revision:          $MODEL_REVISION"
  echo "Served name:       $SERVED_MODEL_NAME"
  echo "Context:           $MAX_MODEL_LEN"
  echo "Tensor parallel:   $TENSOR_PARALLEL_SIZE"
  echo "GPU utilization:   $GPU_MEMORY_UTILIZATION"
  echo "KV cache:          $KV_CACHE_DTYPE"
  echo "Reasoning parser:  $REASONING_PARSER"
  echo "Tool parser:       $TOOL_CALL_PARSER"
  echo "Eager mode:        $ENFORCE_EAGER"
  echo "Prefix caching:    $ENABLE_PREFIX_CACHING"
  echo

  echo "===== TMUX ====="
  tmux list-sessions 2>/dev/null || true

  echo
  echo "===== CLUSTER ====="
  if [[ -x "${RUNTIME_DIR}/launch-cluster.sh" ]]; then
    local -a launcher
    launcher_base_args launcher

    (
      cd "$RUNTIME_DIR"
      ./launch-cluster.sh \
        "${launcher[@]}" \
        status
    ) || true
  else
    echo "Runtime not prepared"
  fi

  echo
  echo "===== MASTER GPU ====="
  nvidia-smi \
    --query-gpu=name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null || true

  echo
  echo "===== WORKER GPU ====="
  ssh_worker \
    "nvidia-smi \
       --query-gpu=name,memory.used,memory.total,utilization.gpu \
       --format=csv,noheader" \
    2>/dev/null || true

  echo
  echo "===== MASTER VLLM PROCESS ====="
  docker exec "$CONTAINER_NAME" \
    bash -lc "pgrep -af '[v]llm serve' || true" \
    2>/dev/null || true

  echo
  echo "===== WORKER VLLM/RAY PROCESS ====="
  ssh_worker \
    "docker exec '$CONTAINER_NAME' \
       bash -lc \"pgrep -af '[r]ay|[v]llm' || true\"" \
    2>/dev/null || true

  echo
  echo "===== API ====="
  if api_healthy; then
    echo "API: HEALTHY"

    api_auth_args
    curl -sS \
      "${API_AUTH_ARGS[@]}" \
      "http://127.0.0.1:${API_PORT}/v1/models"
    echo
  else
    echo "API: NOT READY"
  fi

  echo
  echo "===== WATCHDOG ====="
  watchdog_status
}

show_logs() {
  local target="${1:-all}"

  case "$target" in
    api)
      tail -n 500 "$API_LOG"
      ;;
    watchdog)
      tail -n 300 "$WATCHDOG_LOG"
      ;;
    head)
      docker logs --tail 300 "$CONTAINER_NAME"
      ;;
    worker)
      ssh_worker \
        "docker logs --tail 300 '$CONTAINER_NAME'"
      ;;
    all)
      echo "===== API LAUNCHER ====="
      tail -n 250 "$API_LOG" 2>/dev/null || true

      echo
      echo "===== HEAD CONTAINER ====="
      docker logs --tail 150 "$CONTAINER_NAME" 2>/dev/null || true

      echo
      echo "===== WORKER CONTAINER ====="
      ssh_worker \
        "docker logs --tail 150 '$CONTAINER_NAME' 2>/dev/null || true" ||
        true

      echo
      echo "===== WATCHDOG ====="
      tail -n 100 "$WATCHDOG_LOG" 2>/dev/null || true
      ;;
    *)
      die "logs target must be api, head, worker, watchdog, or all."
      ;;
  esac
}

# ==============================================================================
# API TESTS
# ==============================================================================

test_text() {
  api_auth_args

  curl -sS \
    "${API_AUTH_ARGS[@]}" \
    "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data-binary @- <<EOF
{
  "model": "${SERVED_MODEL_NAME}",
  "messages": [
    {
      "role": "system",
      "content": "You are a precise software engineering assistant."
    },
    {
      "role": "user",
      "content": "Write a Python IPv4 validator with type hints and pytest tests."
    }
  ],
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 40,
  "max_tokens": 2048
}
EOF

  echo
}

test_reasoning() {
  api_auth_args

  curl -sS \
    "${API_AUTH_ARGS[@]}" \
    "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data-binary @- <<EOF
{
  "model": "${SERVED_MODEL_NAME}",
  "messages": [
    {
      "role": "user",
      "content": "Design a resilient local coding-agent platform. Compare failure modes, tool security boundaries, observability, and recovery steps."
    }
  ],
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 40,
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

  curl -sS \
    "${API_AUTH_ARGS[@]}" \
    "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data-binary @- <<EOF
{
  "model": "${SERVED_MODEL_NAME}",
  "messages": [
    {
      "role": "user",
      "content": "Inspect src/app.py with the read_file tool before suggesting any change."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the workspace.",
        "strict": true,
        "parameters": {
          "type": "object",
          "properties": {
            "path": {
              "type": "string",
              "description": "Path relative to the workspace root"
            }
          },
          "required": ["path"],
          "additionalProperties": false
        }
      }
    }
  ],
  "tool_choice": "${choice}",
  "parallel_tool_calls": false,
  "temperature": 0,
  "max_tokens": 2048
}
EOF

  echo
  echo
  echo "Expected: choices[0].message.tool_calls"
}

test_tool_loop() {
  require_command python3

  local api_key="${API_KEY:-vllm-local}"

  OPENAI_BASE_URL="http://127.0.0.1:${API_PORT}/v1" \
  OPENAI_API_KEY="$api_key" \
  OPENAI_MODEL="$SERVED_MODEL_NAME" \
  python3 - <<'PY'
from __future__ import annotations

import json
import os
import urllib.request
import uuid

base_url = os.environ["OPENAI_BASE_URL"].rstrip("/")
api_key = os.environ["OPENAI_API_KEY"]
model = os.environ["OPENAI_MODEL"]


def request(payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.loads(response.read())


tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 file from the workspace.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    }
]

messages = [
    {
        "role": "user",
        "content": (
            "Read src/app.py, identify the bug, then explain the smallest fix."
        ),
    }
]

first = request(
    {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "temperature": 0,
        "max_tokens": 2048,
    }
)

message = first["choices"][0]["message"]
tool_calls = message.get("tool_calls") or []

if not tool_calls:
    raise SystemExit(
        "FAIL: first response did not contain structured tool_calls\n"
        + json.dumps(first, indent=2, ensure_ascii=False)
    )

tool_call = tool_calls[0]
arguments = json.loads(tool_call["function"]["arguments"])

if tool_call["function"]["name"] != "read_file":
    raise SystemExit(f"FAIL: unexpected tool: {tool_call}")

if arguments.get("path") != "src/app.py":
    raise SystemExit(f"FAIL: unexpected arguments: {arguments}")

messages.append(message)
messages.append(
    {
        "role": "tool",
        "tool_call_id": tool_call.get("id") or str(uuid.uuid4()),
        "name": "read_file",
        "content": (
            "def divide(a: float, b: float) -> float:\n"
            "    return a / 0\n"
        ),
    }
)

second = request(
    {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "temperature": 0,
        "max_tokens": 2048,
    }
)

second_message = second["choices"][0]["message"]
content = second_message.get("content") or ""

if not content.strip():
    raise SystemExit(
        "FAIL: final response content is empty\n"
        + json.dumps(second, indent=2, ensure_ascii=False)
    )

print("PASS: structured tool call and tool-result continuation")
print(content)
PY
}

stress_test() {
  local minutes="${1:-60}"

  [[ "$minutes" =~ ^[0-9]+$ ]] ||
    die "Stress duration must be an integer number of minutes."

  require_command python3

  local api_key="${API_KEY:-vllm-local}"

  OPENAI_BASE_URL="http://127.0.0.1:${API_PORT}/v1" \
  OPENAI_API_KEY="$api_key" \
  OPENAI_MODEL="$SERVED_MODEL_NAME" \
  STRESS_MINUTES="$minutes" \
  python3 - <<'PY'
from __future__ import annotations

import json
import os
import time
import urllib.request

base_url = os.environ["OPENAI_BASE_URL"].rstrip("/")
api_key = os.environ["OPENAI_API_KEY"]
model = os.environ["OPENAI_MODEL"]
duration = int(os.environ["STRESS_MINUTES"]) * 60
deadline = time.monotonic() + duration

prompts = [
    "Write a safe atomic-file replacement function in Python.",
    "Explain a deadlock involving two locks and propose a fix.",
    "Review a hypothetical REST retry policy and identify failure modes.",
    "Design unit tests for a rate limiter.",
]

successes = 0
failures = 0
index = 0

while time.monotonic() < deadline:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompts[index % len(prompts)]}],
        "temperature": 0.2,
        "max_tokens": 512,
    }

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    started = time.monotonic()

    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            result = json.loads(response.read())
        content = result["choices"][0]["message"].get("content") or ""
        if not content.strip():
            raise RuntimeError("empty content")
        successes += 1
        elapsed = time.monotonic() - started
        print(
            f"PASS request={successes + failures} elapsed={elapsed:.1f}s "
            f"successes={successes} failures={failures}",
            flush=True,
        )
    except Exception as exc:
        failures += 1
        elapsed = time.monotonic() - started
        print(
            f"FAIL request={successes + failures} elapsed={elapsed:.1f}s "
            f"error={exc!r}",
            flush=True,
        )

    index += 1

print(
    f"RESULT successes={successes} failures={failures} "
    f"total={successes + failures}"
)

if failures:
    raise SystemExit(1)
PY
}

# ==============================================================================
# CLIENT CONFIG / HELP
# ==============================================================================

print_client_config() {
  local client_key="${API_KEY:-vllm-local}"

  cat <<EOF
==============================================================================
OpenAI-compatible endpoint
==============================================================================

Provider:          OpenAI Compatible
Base URL:          $(public_base_url)
API key:           ${client_key}
Model ID:          ${SERVED_MODEL_NAME}
Context window:    ${MAX_MODEL_LEN}
Max input tokens:  ${CLIENT_CONTEXT_TOKENS}
Max output tokens: ${CLIENT_MAX_OUTPUT_TOKENS}
Tensor parallel:   ${TENSOR_PARALLEL_SIZE}

export OPENAI_BASE_URL="$(public_base_url)"
export OPENAI_API_KEY="${client_key}"
export OPENAI_MODEL="${SERVED_MODEL_NAME}"

Supports tools:              yes
Supports reasoning:          yes
Interleaved tool reasoning:  yes
Parallel tools default:      false
Input modalities:            text
Recommended temperature:     1.0
Recommended top_p:           0.95
Recommended top_k:           40

Notes:
- Reasoning is returned in the response's `reasoning` field.
- The agent executes tools; vLLM only returns structured tool_calls.
- Validate all arguments and use a workspace sandbox.
- Use tool_choice=required or a named function for schema/pipeline testing.
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

  $(basename "$0") prepare-runtime
  $(basename "$0") runtime-info
  $(basename "$0") download
  $(basename "$0") verify-files
  $(basename "$0") sync-worker
  $(basename "$0") verify-worker
  $(basename "$0") start
  $(basename "$0") stop
  $(basename "$0") restart
  $(basename "$0") status
  $(basename "$0") logs [api|head|worker|watchdog|all]
  $(basename "$0") test-text
  $(basename "$0") test-reasoning
  $(basename "$0") test-tools [required|auto]
  $(basename "$0") test-tool-loop
  $(basename "$0") stress [minutes]
  $(basename "$0") watchdog-start
  $(basename "$0") watchdog-stop
  $(basename "$0") watchdog-status
  $(basename "$0") client-config

First setup:
  1. Configure passwordless SSH to ${SSH_TARGET}.
  2. ./$(basename "$0") prepare-runtime
  3. ./$(basename "$0") download
  4. ./$(basename "$0") sync-worker
  5. ./$(basename "$0") verify-worker
  6. Stop another model using GPU/port ${API_PORT}.
  7. ./$(basename "$0") start
  8. ./$(basename "$0") test-tool-loop
  9. ./$(basename "$0") stress 60

After reboot:
  ./$(basename "$0") start
EOF
}

# ==============================================================================
# ENTRY POINT
# ==============================================================================

parse_common_options "$@"
set -- "${REMAINING_ARGS[@]}"

case "${1:-help}" in
  prepare-runtime)
    prepare_runtime
    ;;
  runtime-info)
    runtime_info
    ;;
  download)
    download_model
    ;;
  verify-files)
    prepare_special_files
    verify_files_local
    ;;
  sync-worker)
    sync_worker
    ;;
  verify-worker)
    verify_worker_files
    ;;
  start)
    start_all
    ;;
  stop)
    check_running_on_master
    stop_all
    ;;
  restart)
    restart_all
    ;;
  status)
    show_status
    ;;
  logs)
    show_logs "${2:-all}"
    ;;
  test-text)
    test_text
    ;;
  test-reasoning)
    test_reasoning
    ;;
  test-tools)
    test_tools "${2:-required}"
    ;;
  test-tool-loop)
    test_tool_loop
    ;;
  stress)
    stress_test "${2:-60}"
    ;;
  watchdog-start)
    start_watchdog
    ;;
  watchdog-stop)
    stop_watchdog
    ;;
  watchdog-status)
    watchdog_status
    ;;
  client-config)
    print_client_config
    ;;
  _api-foreground)
    api_foreground
    ;;
  _watchdog)
    watchdog_foreground
    ;;
  _restart-from-watchdog)
    restart_from_watchdog
    ;;
  network-info)
    network_info
    ;;
  help|-h|--help)
    show_help
    ;;
  *)
    show_help
    exit 1
    ;;
esac
