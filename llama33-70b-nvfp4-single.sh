#!/usr/bin/env bash
#
# llama33-70b-nvfp4-single.sh
#
# Single-DGX-Spark controller for:
#   nvidia/Llama-3.3-70B-Instruct-NVFP4
#
# Runtime:
#   NVIDIA NGC vLLM 26.06-py3 (ARM64 / Blackwell)
#
# Commands:
#   prepare-runtime, update-runtime, runtime-info
#   download, seal-files, verify-files
#   start, stop, restart, status, logs
#   test-text, test-thai, test-tools, test-tool-loop
#   bench, stress, client-config
#
set -Eeuo pipefail

SCRIPT_VERSION="${SCRIPT_VERSION:-3.0.0}"

# ==============================================================================
# USER CONFIGURATION
# ==============================================================================


MODEL_ID="nvidia/Llama-3.3-70B-Instruct-NVFP4"
MODEL_REVISION="ec8feebb970caeb572f375eb1767b92475e99d84"
SERVED_MODEL_NAME="llama-3.3-70b-instruct-nvfp4"

# NVIDIA's current multi-arch vLLM container at bundle creation.
VLLM_IMAGE="nvcr.io/nvidia/vllm:26.06-py3"
CONTAINER_NAME="llama33-70b-nvfp4-single"

# Conservative first-run settings. Native maximum is 131072.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="0.85"
MAX_NUM_SEQS="2"
MAX_NUM_BATCHED_TOKENS="8192"
KV_CACHE_DTYPE="fp8"

ENABLE_PREFIX_CACHING="1"
ENABLE_CHUNKED_PREFILL="1"
ENFORCE_EAGER="0"

ENABLE_AUTO_TOOL_CHOICE="1"
TOOL_CALL_PARSER="llama3_json"

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

API_WAIT_SECONDS="1800"

# ==============================================================================
# PATHS
# ==============================================================================

CURRENT_USER="${SUDO_USER:-${USER:-$(id -un)}}"
USER_HOME="$(getent passwd "$CURRENT_USER" | cut -d: -f6)"
[[ -n "$USER_HOME" ]] || {
  echo "ERROR: Cannot resolve home directory." >&2
  exit 1
}

BASE_DIR="${USER_HOME}/llama33-70b-nvfp4"
HF_HOME="${USER_HOME}/.cache/huggingface"

MODEL_CACHE_NAME="models--${MODEL_ID//\//--}"
MODEL_CACHE_PATH="${HF_HOME}/hub/${MODEL_CACHE_NAME}"
MODEL_SNAPSHOT_PATH="${MODEL_CACHE_PATH}/snapshots/${MODEL_REVISION}"

SPECIAL_DIR="${BASE_DIR}/special-files"
TOOL_TEMPLATE_HOST="${SPECIAL_DIR}/tool_chat_template_llama3.1_json.jinja"
TOOL_TEMPLATE_CONTAINER="/app/tool_chat_template_llama3.1_json.jinja"

MODEL_MANIFEST="${SPECIAL_DIR}/MODEL_SHA256SUMS"
SPECIAL_MANIFEST="${SPECIAL_DIR}/SPECIAL_SHA256SUMS"
RUNTIME_LOCK="${BASE_DIR}/RUNTIME_LOCK.txt"

LOG_DIR="${BASE_DIR}/logs"
LOG_FILE="${LOG_DIR}/vllm.log"

# ==============================================================================
# HELPERS
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

  echo "===== SINGLE-NODE NETWORK ====="
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


ensure_docker_access() {
  docker info >/dev/null 2>&1 ||
    die "Docker is unavailable to ${CURRENT_USER}."
}

container_exists() {
  docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1
}

container_running() {
  [[ "$(
    docker container inspect \
      --format '{{.State.Running}}' \
      "$CONTAINER_NAME" 2>/dev/null || true
  )" == "true" ]]
}

port_in_use() {
  timeout 1 bash -c \
    "</dev/tcp/127.0.0.1/${API_PORT}" 2>/dev/null
}

api_auth_args() {
  API_AUTH_ARGS=()
  if [[ -n "$API_KEY" ]]; then
    API_AUTH_ARGS=(-H "Authorization: Bearer ${API_KEY}")
  fi
}

api_healthy() {
  curl -fsS --max-time 8 \
    "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1
}

wait_for_api() {
  local deadline=$((SECONDS + API_WAIT_SECONDS))

  log "Waiting for vLLM API"

  while (( SECONDS < deadline )); do
    if api_healthy; then
      return
    fi

    if ! container_running; then
      docker logs --tail 400 "$CONTAINER_NAME" 2>/dev/null || true
      die "Container exited before the API became healthy."
    fi

    sleep 5
  done

  docker logs --tail 400 "$CONTAINER_NAME" 2>/dev/null || true
  die "API did not become healthy within ${API_WAIT_SECONDS}s."
}

runtime_image_id() {
  docker image inspect "$VLLM_IMAGE" --format '{{.Id}}'
}

verify_runtime_lock() {
  [[ -f "$RUNTIME_LOCK" ]] ||
    die "Runtime lock is missing. Run: $0 prepare-runtime"

  local expected_id
  expected_id="$(awk -F= '/^image_id=/{print $2}' "$RUNTIME_LOCK")"

  local current_id
  current_id="$(runtime_image_id)"

  [[ -n "$expected_id" && "$expected_id" == "$current_id" ]] ||
    die "Runtime image differs from RUNTIME_LOCK.txt. Run update-runtime and repeat all tests."
}

# ==============================================================================
# RUNTIME PREPARATION
# ==============================================================================

extract_tool_template() {
  mkdir -p "$SPECIAL_DIR"

  log "Extracting vLLM's official Llama 3.1 JSON tool template"

  docker run --rm \
    --entrypoint bash \
    "$VLLM_IMAGE" \
    -lc '
set -Eeuo pipefail
candidates=(
  /workspace/examples/tool_chat_template_llama3.1_json.jinja
  /vllm-workspace/examples/tool_chat_template_llama3.1_json.jinja
  /opt/vllm/examples/tool_chat_template_llama3.1_json.jinja
)
for path in "${candidates[@]}"; do
  if [[ -f "$path" ]]; then
    cat "$path"
    exit 0
  fi
done
path="$(find /workspace /vllm-workspace /opt \
  -name tool_chat_template_llama3.1_json.jinja \
  -type f 2>/dev/null | head -n 1 || true)"
[[ -n "$path" ]] || {
  echo "Official Llama tool template not found in container." >&2
  exit 1
}
cat "$path"
' > "$TOOL_TEMPLATE_HOST"

  [[ -s "$TOOL_TEMPLATE_HOST" ]] ||
    die "Extracted tool template is empty."

  python3 - "$TOOL_TEMPLATE_HOST" <<'PY'
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text()
required = [
    "tools_in_user_message",
    "message.tool_calls",
    'message.role == "tool"',
    '"name"',
    '"parameters"',
    "<|start_header_id|>",
]
missing = [marker for marker in required if marker not in text]
if missing:
    raise SystemExit("Tool template missing markers: " + ", ".join(missing))
print("Tool template markers verified")
PY
}

write_runtime_lock() {
  local image_id
  local repo_digests
  local versions

  image_id="$(runtime_image_id)"
  repo_digests="$(
    docker image inspect "$VLLM_IMAGE" \
      --format '{{join .RepoDigests ","}}' 2>/dev/null || true
  )"

  versions="$(
    docker run --rm \
      --entrypoint python3 \
      "$VLLM_IMAGE" \
      -c '
import importlib.metadata as m
for package in ("vllm", "torch", "transformers", "flashinfer-python"):
    try:
        print(f"{package}={m.version(package)}")
    except Exception:
        print(f"{package}=unknown")
'
  )"

  {
    echo "image=${VLLM_IMAGE}"
    echo "image_id=${image_id}"
    echo "repo_digests=${repo_digests}"
    echo "prepared_at=$(date --iso-8601=seconds)"
    echo "$versions"
  } > "$RUNTIME_LOCK"

  log "Runtime lock written"
  cat "$RUNTIME_LOCK"
}

verify_runtime_capabilities() {
  log "Checking GPU and required vLLM parser"

  docker run --rm \
    --gpus all \
    --entrypoint nvidia-smi \
    "$VLLM_IMAGE" >/dev/null 2>&1 ||
    die "A fresh vLLM container cannot access the GPU."

  docker run --rm \
    --entrypoint python3 \
    "$VLLM_IMAGE" \
    -c '
from vllm.tool_parsers import ToolParserManager
names = set(ToolParserManager.tool_parsers)
if "llama3_json" not in names:
    raise SystemExit("llama3_json tool parser is missing")
print("llama3_json parser found")
'
}

prepare_runtime() {
  require_command docker
  require_command python3

  ensure_docker_access

  log "Pulling runtime: $VLLM_IMAGE"
  docker pull "$VLLM_IMAGE"

  verify_runtime_capabilities
  extract_tool_template
  write_runtime_lock
}

update_runtime() {
  warn "This intentionally refreshes the container image and replaces the runtime lock."
  prepare_runtime
}

runtime_info() {
  echo "===== LOCK ====="
  cat "$RUNTIME_LOCK" 2>/dev/null || echo "No runtime lock"

  echo
  echo "===== LOCAL IMAGE ====="
  docker image inspect "$VLLM_IMAGE" \
    --format 'ID={{.Id}} Digests={{join .RepoDigests ","}}' \
    2>/dev/null || true

  echo
  echo "===== TOOL TEMPLATE ====="
  if [[ -f "$TOOL_TEMPLATE_HOST" ]]; then
    sha256sum "$TOOL_TEMPLATE_HOST"
  else
    echo "Missing"
  fi
}

# ==============================================================================
# DOWNLOAD / SPECIAL FILES / INTEGRITY
# ==============================================================================

download_model() {
  require_command docker
  ensure_docker_access
  verify_runtime_lock

  mkdir -p "$HF_HOME" "$SPECIAL_DIR"

  local -a token_env=()
  if [[ -n "$HF_TOKEN" ]]; then
    token_env=(-e "HF_TOKEN=${HF_TOKEN}")
  fi

  log "Downloading ${MODEL_ID}@${MODEL_REVISION}"

  docker run --rm \
    --user "$(id -u "$CURRENT_USER"):$(id -g "$CURRENT_USER")" \
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

path = snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    revision=os.environ["MODEL_REVISION"],
    cache_dir="/cache",
)
print(path)
'

  prepare_special_files
  seal_files
  verify_files

  log "Download completed"
  du -sh "$MODEL_CACHE_PATH"
}

prepare_special_files() {
  [[ -d "$MODEL_SNAPSHOT_PATH" ]] ||
    die "Pinned snapshot is missing: $MODEL_SNAPSHOT_PATH"

  mkdir -p "$SPECIAL_DIR"

  local -a required=(
    config.json
    generation_config.json
    hf_quant_config.json
    model.safetensors.index.json
    special_tokens_map.json
    tokenizer.json
    tokenizer_config.json
  )

  local file
  for file in "${required[@]}"; do
    [[ -e "${MODEL_SNAPSHOT_PATH}/${file}" ]] ||
      die "Required file missing: $file"
    cp -L "${MODEL_SNAPSHOT_PATH}/${file}" "${SPECIAL_DIR}/${file}"
  done

  [[ -s "$TOOL_TEMPLATE_HOST" ]] ||
    die "Tool template is missing. Run: $0 prepare-runtime"

  (
    cd "$SPECIAL_DIR"
    sha256sum \
      config.json \
      generation_config.json \
      hf_quant_config.json \
      model.safetensors.index.json \
      special_tokens_map.json \
      tokenizer.json \
      tokenizer_config.json \
      tool_chat_template_llama3.1_json.jinja \
      > SPECIAL_SHA256SUMS
  )
}

seal_files() {
  require_command sha256sum
  [[ -d "$MODEL_SNAPSHOT_PATH" ]] ||
    die "Pinned snapshot is missing."

  log "Creating trusted SHA-256 manifest for the pinned snapshot"
  log "This hashes approximately 42.7 GB once."

  local temp="${MODEL_MANIFEST}.tmp"
  : > "$temp"

  while IFS= read -r -d '' path; do
    local relative
    relative="${path#${MODEL_SNAPSHOT_PATH}/}"
    local digest
    digest="$(sha256sum "$path" | awk '{print $1}')"
    printf '%s  %s\n' "$digest" "$relative" >> "$temp"
  done < <(
    find "$MODEL_SNAPSHOT_PATH" \
      -maxdepth 1 \
      \( -type f -o -type l \) \
      -print0 |
      sort -z
  )

  mv "$temp" "$MODEL_MANIFEST"
  log "Model manifest written: $MODEL_MANIFEST"
}

verify_files() {
  require_command python3
  require_command sha256sum

  [[ -d "$MODEL_SNAPSHOT_PATH" ]] ||
    die "Pinned snapshot is missing. Run: $0 download"

  [[ -f "$MODEL_MANIFEST" ]] ||
    die "Model SHA manifest is missing. Run: $0 seal-files"

  [[ -f "$SPECIAL_MANIFEST" ]] ||
    prepare_special_files

  python3 - "$MODEL_SNAPSHOT_PATH" "$SPECIAL_DIR" "$MODEL_REVISION" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

snapshot = Path(sys.argv[1])
special = Path(sys.argv[2])
revision = sys.argv[3]

if snapshot.name != revision:
    raise SystemExit(f"Revision mismatch: {snapshot.name} != {revision}")

config = json.loads((snapshot / "config.json").read_text())
expected = {
    "model_type": "llama",
    "hidden_size": 8192,
    "num_hidden_layers": 80,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "max_position_embeddings": 131072,
    "vocab_size": 128256,
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(
            f"Unexpected config {key}: {config.get(key)!r} != {value!r}"
        )

quant = json.loads((snapshot / "hf_quant_config.json").read_text())
q = quant.get("quantization", {})
expected_quant = {
    "quant_algo": "NVFP4",
    "kv_cache_quant_algo": "FP8",
    "group_size": 16,
}
for key, value in expected_quant.items():
    if q.get(key) != value:
        raise SystemExit(
            f"Unexpected quant config {key}: {q.get(key)!r} != {value!r}"
        )
if "lm_head" not in q.get("exclude_modules", []):
    raise SystemExit("lm_head is not excluded from NVFP4 quantization")

index = json.loads((snapshot / "model.safetensors.index.json").read_text())
total_size = index.get("metadata", {}).get("total_size")
if total_size != 42709045632:
    raise SystemExit(f"Unexpected indexed total_size: {total_size}")

weight_map = index.get("weight_map", {})
shards = sorted(set(weight_map.values()))
if len(shards) != 9:
    raise SystemExit(f"Expected 9 shards, found {len(shards)}")

missing = [name for name in shards if not (snapshot / name).exists()]
if missing:
    raise SystemExit("Missing shards: " + ", ".join(missing))

broken = [
    path for path in snapshot.iterdir()
    if path.is_symlink() and not path.exists()
]
if broken:
    raise SystemExit(
        "Broken symlinks:\n" + "\n".join(str(path) for path in broken)
    )

generation = json.loads((snapshot / "generation_config.json").read_text())
if generation.get("temperature") != 0.6 or generation.get("top_p") != 0.9:
    raise SystemExit("Unexpected generation defaults")

template = (special / "tool_chat_template_llama3.1_json.jinja").read_text()
for marker in (
    "tools_in_user_message",
    "message.tool_calls",
    '"parameters"',
    "<|start_header_id|>",
):
    if marker not in template:
        raise SystemExit(f"Tool template missing marker: {marker}")

print(f"Revision:          {snapshot.name}")
print(f"Indexed size:      {total_size} bytes")
print(f"Indexed tensors:   {len(weight_map)}")
print(f"Weight shards:     {len(shards)}")
print("Architecture:      Llama 3.3 70B, 80 layers")
print("Native context:    131072")
print("Quantization:      NVFP4, group size 16")
print("KV quant metadata: FP8")
print("Tool template:     Llama 3 JSON")
PY

  (
    cd "$SPECIAL_DIR"
    sha256sum -c SPECIAL_SHA256SUMS
  )

  log "Verifying complete pinned snapshot against local seal"

  (
    cd "$MODEL_SNAPSHOT_PATH"
    sha256sum -c "$MODEL_MANIFEST"
  )

  log "All model, tokenizer, quantization, and tool-template files verified"
}

# ==============================================================================
# START / STOP
# ==============================================================================

stop_server() {
  require_command docker
  ensure_docker_access

  if container_exists; then
    log "Stopping $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  else
    log "Server is already stopped"
  fi
}

start_server() {
  local -a args
  local -a auth_args
  local -a offline_env
  local -a token_env

  require_command docker
  require_command curl
  require_command timeout
  require_command nvidia-smi

  ensure_docker_access
  verify_runtime_lock
  verify_files

  nvidia-smi >/dev/null 2>&1 ||
    die "Host nvidia-smi failed."

  docker run --rm \
    --gpus all \
    --entrypoint nvidia-smi \
    "$VLLM_IMAGE" >/dev/null 2>&1 ||
    die "A fresh runtime container cannot access the GPU."

  if container_exists; then
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi

  if port_in_use; then
    die "Port ${API_PORT} is already in use."
  fi

  args=(
    serve "$MODEL_ID"
    --revision "$MODEL_REVISION"
    --served-model-name "$SERVED_MODEL_NAME"
    --dtype auto
    --quantization modelopt
    --tensor-parallel-size 1
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --host "$API_HOST"
    --port "$API_PORT"
    --chat-template "$TOOL_TEMPLATE_CONTAINER"
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

  if [[ "$ENFORCE_EAGER" == "1" ]]; then
    args+=(--enforce-eager)
  fi

  auth_args=()
  if [[ -n "$API_KEY" ]]; then
    auth_args=(--api-key "$API_KEY")
  fi

  offline_env=()
  if [[ "$HF_HUB_OFFLINE" == "1" ]]; then
    offline_env=(
      -e HF_HUB_OFFLINE=1
      -e TRANSFORMERS_OFFLINE=1
    )
  fi

  token_env=()
  if [[ -n "$HF_TOKEN" ]]; then
    token_env=(-e "HF_TOKEN=${HF_TOKEN}")
  fi

  mkdir -p "$LOG_DIR"
  : > "$LOG_FILE"

  log "Starting Llama 3.3 70B NVFP4 on one DGX Spark"
  log "Context:   $MAX_MODEL_LEN"
  log "KV cache:  $KV_CACHE_DTYPE"
  log "Tools:     $TOOL_CALL_PARSER"
  log "API:       $(public_base_url)"

  docker run -d \
    --name "$CONTAINER_NAME" \
    --network host \
    --ipc host \
    --shm-size 32g \
    --gpus all \
    -v "${HF_HOME}:/root/.cache/huggingface" \
    -v "${TOOL_TEMPLATE_HOST}:${TOOL_TEMPLATE_CONTAINER}:ro" \
    "${offline_env[@]}" \
    "${token_env[@]}" \
    --entrypoint vllm \
    "$VLLM_IMAGE" \
    "${args[@]}" \
    "${auth_args[@]}" \
    >/dev/null

  wait_for_api

  docker logs "$CONTAINER_NAME" > "$LOG_FILE" 2>&1 || true

  log "READY"
  echo "Base URL: $(public_base_url)"
  echo "Model:    ${SERVED_MODEL_NAME}"
  echo "Context:  ${MAX_MODEL_LEN}"
}

# ==============================================================================
# STATUS / LOGS
# ==============================================================================

show_status() {
  require_command docker
  require_command curl

  echo "===== CONFIG ====="
  echo "Model:             $MODEL_ID"
  echo "Revision:          $MODEL_REVISION"
  echo "Served name:       $SERVED_MODEL_NAME"
  echo "Runtime:           $VLLM_IMAGE"
  echo "Context:           $MAX_MODEL_LEN"
  echo "KV cache:          $KV_CACHE_DTYPE"
  echo "Max sequences:     $MAX_NUM_SEQS"
  echo "Tool parser:       $TOOL_CALL_PARSER"
  echo

  echo "===== RUNTIME LOCK ====="
  cat "$RUNTIME_LOCK" 2>/dev/null || true

  echo
  echo "===== CONTAINER ====="
  docker ps -a \
    --filter "name=^/${CONTAINER_NAME}$" \
    --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'

  if container_running; then
    echo
    echo "===== GPU ====="
    docker exec "$CONTAINER_NAME" \
      nvidia-smi \
      --query-gpu=name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader 2>/dev/null || true

    echo
    echo "===== PROCESS ====="
    docker exec "$CONTAINER_NAME" \
      bash -lc "pgrep -af '[v]llm serve' || true" \
      2>/dev/null || true
  fi

  echo
  echo "===== API ====="
  if api_healthy; then
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
  require_command docker
  ensure_docker_access

  docker logs --tail "${1:-400}" "$CONTAINER_NAME"
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
  "top_p": 0.9,
  "max_tokens": 2048
}
EOF
  echo
}

test_thai() {
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
      "content": "อธิบายหลักการ zero-trust สำหรับระบบ AI agent แบบกระชับ พร้อมยกตัวอย่างมาตรการ 5 ข้อ"
    }
  ],
  "temperature": 0.6,
  "top_p": 0.9,
  "max_tokens": 2048
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
    {
      "role": "user",
      "content": "Inspect src/app.py with read_file before proposing changes."
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
  "max_tokens": 1024
}
EOF
  echo
  echo
  echo "Expected: choices[0].message.tool_calls"
}

test_tool_loop() {
  require_command python3

  local key="${API_KEY:-vllm-local}"

  OPENAI_BASE_URL="http://127.0.0.1:${API_PORT}/v1" \
  OPENAI_API_KEY="$key" \
  OPENAI_MODEL="$SERVED_MODEL_NAME" \
  python3 - <<'PY'
from __future__ import annotations

import json
import os
import urllib.request
import uuid

base = os.environ["OPENAI_BASE_URL"].rstrip("/")
key = os.environ["OPENAI_API_KEY"]
model = os.environ["OPENAI_MODEL"]


def call(payload: dict) -> dict:
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.loads(response.read())


tools = [{
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
}]

messages = [{
    "role": "user",
    "content": "Read src/app.py first, then identify the smallest bug fix.",
}]

first = call({
    "model": model,
    "messages": messages,
    "tools": tools,
    "tool_choice": "required",
    "parallel_tool_calls": False,
    "temperature": 0,
    "max_tokens": 1024,
})

message = first["choices"][0]["message"]
tool_calls = message.get("tool_calls") or []
if not tool_calls:
    raise SystemExit(
        "FAIL: no structured tool_calls\n"
        + json.dumps(first, indent=2, ensure_ascii=False)
    )

tool_call = tool_calls[0]
function = tool_call.get("function") or {}
arguments = json.loads(function.get("arguments") or "{}")

if function.get("name") != "read_file":
    raise SystemExit(f"FAIL: unexpected tool: {tool_call}")
if arguments.get("path") != "src/app.py":
    raise SystemExit(f"FAIL: unexpected arguments: {arguments}")

messages.append(message)
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.get("id") or str(uuid.uuid4()),
    "name": "read_file",
    "content": (
        "def divide(a: float, b: float) -> float:\n"
        "    return a / 0\n"
    ),
})

second = call({
    "model": model,
    "messages": messages,
    "tools": tools,
    "tool_choice": "auto",
    "parallel_tool_calls": False,
    "temperature": 0,
    "max_tokens": 2048,
})

final = second["choices"][0]["message"].get("content") or ""
if not final.strip():
    raise SystemExit(
        "FAIL: final answer is empty\n"
        + json.dumps(second, indent=2, ensure_ascii=False)
    )

print("PASS: structured tool call and role=tool continuation")
print(final)
PY
}

bench_api() {
  require_command python3

  local key="${API_KEY:-vllm-local}"

  OPENAI_BASE_URL="http://127.0.0.1:${API_PORT}/v1" \
  OPENAI_API_KEY="$key" \
  OPENAI_MODEL="$SERVED_MODEL_NAME" \
  python3 - <<'PY'
from __future__ import annotations

import json
import os
import time
import urllib.request

base = os.environ["OPENAI_BASE_URL"].rstrip("/")
key = os.environ["OPENAI_API_KEY"]
model = os.environ["OPENAI_MODEL"]

for tokens in (128, 256, 512):
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": "Explain reliable distributed-system retries with examples.",
        }],
        "temperature": 0.6,
        "top_p": 0.9,
        "max_tokens": tokens,
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as response:
        result = json.loads(response.read())
    elapsed = time.monotonic() - started
    usage = result.get("usage", {})
    completion = usage.get("completion_tokens")
    rate = completion / elapsed if completion else None
    print({
        "max_tokens": tokens,
        "elapsed_seconds": round(elapsed, 3),
        "completion_tokens": completion,
        "completion_tokens_per_second": round(rate, 3) if rate else None,
    })
PY
}

stress_test() {
  local minutes="${1:-30}"
  [[ "$minutes" =~ ^[0-9]+$ ]] ||
    die "Stress duration must be an integer number of minutes."

  require_command python3

  local key="${API_KEY:-vllm-local}"

  OPENAI_BASE_URL="http://127.0.0.1:${API_PORT}/v1" \
  OPENAI_API_KEY="$key" \
  OPENAI_MODEL="$SERVED_MODEL_NAME" \
  STRESS_MINUTES="$minutes" \
  python3 - <<'PY'
from __future__ import annotations

import json
import os
import time
import urllib.request

base = os.environ["OPENAI_BASE_URL"].rstrip("/")
key = os.environ["OPENAI_API_KEY"]
model = os.environ["OPENAI_MODEL"]
deadline = time.monotonic() + int(os.environ["STRESS_MINUTES"]) * 60

prompts = [
    "Write unit tests for a token bucket.",
    "Explain a two-lock deadlock and its fix.",
    "Review an HTTP retry policy for failure modes.",
    "Design a safe atomic file replacement function.",
]

successes = 0
failures = 0
index = 0

while time.monotonic() < deadline:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompts[index % len(prompts)]}],
        "temperature": 0.6,
        "top_p": 0.9,
        "max_tokens": 768,
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=900) as response:
            result = json.loads(response.read())
        content = result["choices"][0]["message"].get("content") or ""
        if not content.strip():
            raise RuntimeError("empty content")
        successes += 1
        print(
            f"PASS request={successes + failures} "
            f"elapsed={time.monotonic() - started:.1f}s "
            f"successes={successes} failures={failures}",
            flush=True,
        )
    except Exception as exc:
        failures += 1
        print(
            f"FAIL request={successes + failures} error={exc!r}",
            flush=True,
        )
    index += 1

print(f"RESULT successes={successes} failures={failures}")
if failures:
    raise SystemExit(1)
PY
}

# ==============================================================================
# CLIENT CONFIG / HELP
# ==============================================================================

print_client_config() {
  local key="${API_KEY:-vllm-local}"

  cat <<EOF
Provider:          OpenAI Compatible
Base URL:          $(public_base_url)
API key:           ${key}
Model ID:          ${SERVED_MODEL_NAME}
Context window:    ${MAX_MODEL_LEN}
Max input tokens:  ${CLIENT_CONTEXT_TOKENS}
Max output tokens: ${CLIENT_MAX_OUTPUT_TOKENS}

Text input/output: yes
Thai:              supported by base model
Native reasoning:  no dedicated reasoning channel
Tools:             llama3_json
Parallel tools:    false
Vision/audio/video: no

Recommended sampling:
  temperature: 0.6
  top_p:       0.9

Agent notes:
  - Send native OpenAI tools.
  - Llama 3 supports one tool call at a time with this template.
  - Validate arguments because arrays may occasionally be serialized as strings.
  - Return tool results with role=tool and tool_call_id.
  - Use workspace, command, path, network, and credential sandboxes.
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
  $(basename "$0") update-runtime
  $(basename "$0") runtime-info
  $(basename "$0") download
  $(basename "$0") seal-files
  $(basename "$0") verify-files
  $(basename "$0") start
  $(basename "$0") stop
  $(basename "$0") restart
  $(basename "$0") status
  $(basename "$0") logs [lines]
  $(basename "$0") test-text
  $(basename "$0") test-thai
  $(basename "$0") test-tools [required|auto]
  $(basename "$0") test-tool-loop
  $(basename "$0") bench
  $(basename "$0") stress [minutes]
  $(basename "$0") client-config

First setup:
  1. ./$(basename "$0") prepare-runtime
  2. ./$(basename "$0") download
  3. Stop another model using GPU/port ${API_PORT}.
  4. ./$(basename "$0") start
  5. ./$(basename "$0") test-text
  6. ./$(basename "$0") test-thai
  7. ./$(basename "$0") test-tools required
  8. ./$(basename "$0") test-tool-loop
  9. ./$(basename "$0") stress 30

After reboot:
  ./$(basename "$0") start
EOF
}

parse_common_options "$@"
set -- "${REMAINING_ARGS[@]}"

case "${1:-help}" in
  prepare-runtime)
    prepare_runtime
    ;;
  update-runtime)
    update_runtime
    ;;
  runtime-info)
    runtime_info
    ;;
  download)
    download_model
    ;;
  seal-files)
    prepare_special_files
    seal_files
    ;;
  verify-files)
    verify_files
    ;;
  start)
    start_server
    ;;
  stop)
    stop_server
    ;;
  restart)
    stop_server
    start_server
    ;;
  status)
    show_status
    ;;
  logs)
    show_logs "${2:-400}"
    ;;
  test-text)
    test_text
    ;;
  test-thai)
    test_thai
    ;;
  test-tools)
    test_tools "${2:-required}"
    ;;
  test-tool-loop)
    test_tool_loop
    ;;
  bench)
    bench_api
    ;;
  stress)
    stress_test "${2:-30}"
    ;;
  client-config)
    print_client_config
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
