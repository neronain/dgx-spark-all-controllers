#!/usr/bin/env bash
#
# nemotron-3-super-single.sh
#
# Single-DGX-Spark controller for:
#   ucbye/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
#
# The script also downloads, verifies, and mounts the model-specific files:
#   - super_v3_reasoning_parser.py (current official NVIDIA copy)
#   - chat_template.jinja
#   - hf_quant_config.json
#   - generation_config.json
#   - config.json
#   - tokenizer files
#   - remote-code model/configuration files
#   - safetensors index and every referenced shard
#
# Commands:
#   ./nemotron-3-super-single.sh download
#   ./nemotron-3-super-single.sh verify-files
#   ./nemotron-3-super-single.sh start
#   ./nemotron-3-super-single.sh stop
#   ./nemotron-3-super-single.sh restart
#   ./nemotron-3-super-single.sh status
#   ./nemotron-3-super-single.sh logs
#   ./nemotron-3-super-single.sh test-text
#   ./nemotron-3-super-single.sh test-reasoning
#   ./nemotron-3-super-single.sh test-tools required
#   ./nemotron-3-super-single.sh client-config
#
set -Eeuo pipefail

SCRIPT_VERSION="${SCRIPT_VERSION:-3.0.0}"

# ==============================================================================
# USER CONFIGURATION — edit values here
# ==============================================================================

MODEL_ID="ucbye/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
MODEL_REVISION="main"

# Keep the parser current even if the duplicated ucbye repository is older.
PARSER_REPO_ID="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
PARSER_REVISION="main"

SERVED_MODEL_NAME="nemotron-3-super"

# Current NVIDIA DGX Spark guidance for this model.
VLLM_IMAGE="vllm/vllm-openai:v0.20.0"
CONTAINER_NAME="nemotron-3-super-vllm"

# The model supports up to 1M tokens. Start at 256K for IDE/agent use.
# After validation, 394000 or 1000000 may be tested.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="0.90"
MAX_NUM_SEQS="4"

KV_CACHE_DTYPE="fp8"
MAMBA_SSM_CACHE_DTYPE="float16"
MOE_BACKEND="marlin"
QUANTIZATION="fp4"

# MTP speculative decoding is part of NVIDIA's current Spark recipe.
ENABLE_MTP_SPECULATION="1"
MTP_SPECULATIVE_TOKENS="3"
MTP_MOE_BACKEND="triton"

# Chunked prefill is recommended for long context.
ENABLE_CHUNKED_PREFILL="1"

# Prefix caching for hybrid Mamba models is still experimental in vLLM.
# Keep disabled until the base deployment is stable.
ENABLE_PREFIX_CACHING="0"

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
ADVERTISE_IP="${ADVERTISE_IP:-}"
ADVERTISE_INTERFACE="${ADVERTISE_INTERFACE:-}"
ROUTE_PROBE_IP="${ROUTE_PROBE_IP:-1.1.1.1}"
CLIENT_OVERHEAD_TOKENS="${CLIENT_OVERHEAD_TOKENS:-8192}"

# Empty = no authentication. Use a long random value for team/LAN access.
API_KEY="${API_KEY:-}"

# Required parsers.
REASONING_PARSER="super_v3"
TOOL_CALL_PARSER="qwen3_coder"
ENABLE_AUTO_TOOL_CHOICE="1"

# Coding/agent-friendly server defaults. Requests can override these values.
DEFAULT_ENABLE_THINKING="false"
DEFAULT_FORCE_NONEMPTY_CONTENT="true"
DEFAULT_LOW_EFFORT="false"

# The model card recommends these sampling values for all tasks.
DEFAULT_TEMPERATURE="1.0"
DEFAULT_TOP_P="0.95"

# Set to 1 once model files are downloaded. Use 0 while updating.
HF_HUB_OFFLINE="1"
HF_TOKEN=""

# Client-side token budget.
CLIENT_CONTEXT_TOKENS="${CLIENT_CONTEXT_TOKENS:-auto}"
CLIENT_MAX_OUTPUT_TOKENS="${CLIENT_MAX_OUTPUT_TOKENS:-16384}"

# Large model startup can take several minutes.
API_WAIT_SECONDS="2400"

# ==============================================================================
# PATHS
# ==============================================================================

CURRENT_USER="${SUDO_USER:-${USER:-$(id -un)}}"
USER_HOME="$(getent passwd "$CURRENT_USER" | cut -d: -f6)"

[[ -n "$USER_HOME" ]] || {
  echo "ERROR: Cannot resolve home directory for ${CURRENT_USER}." >&2
  exit 1
}

BASE_DIR="${USER_HOME}/nemotron-3-super"
HF_HOME="${USER_HOME}/.cache/huggingface"
SPECIAL_DIR="${BASE_DIR}/special-files"

MODEL_CACHE_NAME="models--${MODEL_ID//\//--}"
MODEL_CACHE_PATH="${HF_HOME}/hub/${MODEL_CACHE_NAME}"

RUNTIME_PARSER_HOST="${SPECIAL_DIR}/super_v3_reasoning_parser.py"
RUNTIME_CHAT_TEMPLATE_HOST="${SPECIAL_DIR}/chat_template.jinja"

RUNTIME_PARSER_CONTAINER="/app/super_v3_reasoning_parser.py"
RUNTIME_CHAT_TEMPLATE_CONTAINER="/app/chat_template.jinja"

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


json_bool() {
  case "$1" in
    true|false) printf '%s' "$1" ;;
    *) die "Expected true or false, got: $1" ;;
  esac
}

ensure_docker_access() {
  docker info >/dev/null 2>&1 ||
    die "Docker is not available to ${CURRENT_USER}. Check Docker service/group."
}

ensure_host_gpu() {
  nvidia-smi >/dev/null 2>&1 ||
    die "nvidia-smi failed on the host."
}

ensure_image() {
  if ! docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1; then
    log "Pulling image: $VLLM_IMAGE"
    docker pull "$VLLM_IMAGE"
  fi
}

ensure_fresh_container_gpu() {
  log "Testing GPU access in a fresh container"

  docker run --rm \
    --gpus all \
    --entrypoint nvidia-smi \
    "$VLLM_IMAGE" >/dev/null 2>&1 ||
    die "A fresh Docker container cannot access the GPU."
}

model_cache_complete() {
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

resolve_snapshot_dir() {
  local snapshot

  snapshot="$(
    find "${MODEL_CACHE_PATH}/snapshots" \
      -mindepth 1 \
      -maxdepth 1 \
      -type d \
      -printf '%T@ %p\n' 2>/dev/null |
    sort -nr |
    awk 'NR == 1 {$1=""; sub(/^ /, ""); print; exit}'
  )"

  [[ -n "$snapshot" ]] ||
    die "No model snapshot found under ${MODEL_CACHE_PATH}/snapshots."

  readlink -f "$snapshot"
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

wait_for_api() {
  local deadline=$((SECONDS + API_WAIT_SECONDS))

  log "Waiting for API health endpoint"

  while (( SECONDS < deadline )); do
    if curl -fsS \
      --max-time 5 \
      "http://127.0.0.1:${API_PORT}/health" \
      >/dev/null 2>&1; then
      return 0
    fi

    if ! container_running; then
      docker logs --tail 350 "$CONTAINER_NAME" 2>/dev/null || true
      die "The vLLM container stopped before the API became ready."
    fi

    sleep 5
  done

  docker logs --tail 350 "$CONTAINER_NAME" 2>/dev/null || true
  die "API did not become ready within ${API_WAIT_SECONDS} seconds."
}

# ==============================================================================
# DOWNLOAD AND SPECIAL FILES
# ==============================================================================

download_model() {
  require_command docker
  require_command nvidia-smi
  require_command python3
  require_command sha256sum

  ensure_docker_access
  ensure_host_gpu
  ensure_image

  mkdir -p "$HF_HOME" "$SPECIAL_DIR"

  log "Downloading model repository"
  log "Model:       $MODEL_ID"
  log "Revision:    $MODEL_REVISION"
  log "Destination: $HF_HOME"

  local -a token_env=()
  if [[ -n "$HF_TOKEN" ]]; then
    token_env=(-e "HF_TOKEN=${HF_TOKEN}")
  fi

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

snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    revision=os.environ["MODEL_REVISION"],
)
'

  model_cache_complete ||
    die "Download finished, but the model cache looks incomplete."

  log "Downloading current official reasoning parser"

  docker run --rm \
    --user "$(id -u "$CURRENT_USER"):$(id -g "$CURRENT_USER")" \
    -e HOME=/tmp \
    -e HF_HOME=/cache \
    -e PARSER_REPO_ID="$PARSER_REPO_ID" \
    -e PARSER_REVISION="$PARSER_REVISION" \
    "${token_env[@]}" \
    -v "${HF_HOME}:/cache" \
    -v "${SPECIAL_DIR}:/special" \
    --entrypoint python3 \
    "$VLLM_IMAGE" \
    -c '
import os
import shutil
from huggingface_hub import hf_hub_download

source = hf_hub_download(
    repo_id=os.environ["PARSER_REPO_ID"],
    filename="super_v3_reasoning_parser.py",
    revision=os.environ["PARSER_REVISION"],
)
shutil.copy2(source, "/special/super_v3_reasoning_parser.py")
'

  sync_special_files
  verify_files

  log "Download and special-file preparation completed"
  du -sh "$MODEL_CACHE_PATH"
}

sync_special_files() {
  local snapshot
  snapshot="$(resolve_snapshot_dir)"

  mkdir -p "$SPECIAL_DIR"

  log "Copying important model files into ${SPECIAL_DIR}"

  local file
  local -a files=(
    "__init__.py"
    "chat_template.jinja"
    "config.json"
    "configuration_nemotron_h.py"
    "generation_config.json"
    "hf_quant_config.json"
    "model.safetensors.index.json"
    "modeling_nemotron_h.py"
    "special_tokens_map.json"
    "tokenizer.json"
    "tokenizer_config.json"
  )

  for file in "${files[@]}"; do
    [[ -e "${snapshot}/${file}" ]] ||
      die "Required repository file is missing: ${file}"

    cp -L "${snapshot}/${file}" "${SPECIAL_DIR}/${file}"
  done

  if [[ -e "${snapshot}/super_v3_reasoning_parser.py" ]]; then
    cp -L \
      "${snapshot}/super_v3_reasoning_parser.py" \
      "${SPECIAL_DIR}/super_v3_reasoning_parser.repo.py"
  fi

  [[ -s "$RUNTIME_PARSER_HOST" ]] ||
    die "Current official parser is missing: $RUNTIME_PARSER_HOST"

  (
    cd "$SPECIAL_DIR"
    sha256sum \
      __init__.py \
      chat_template.jinja \
      config.json \
      configuration_nemotron_h.py \
      generation_config.json \
      hf_quant_config.json \
      model.safetensors.index.json \
      modeling_nemotron_h.py \
      special_tokens_map.json \
      super_v3_reasoning_parser.py \
      tokenizer.json \
      tokenizer_config.json \
      > SHA256SUMS
  )
}

verify_files() {
  require_command python3
  require_command sha256sum

  model_cache_complete ||
    die "Model cache is missing. Run: $0 download"

  local snapshot
  snapshot="$(resolve_snapshot_dir)"

  [[ -d "$SPECIAL_DIR" ]] ||
    die "Special-files directory is missing. Run: $0 download"

  log "Verifying repository, shards, parser, template, and quantization files"

  python3 - "$snapshot" "$SPECIAL_DIR" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

snapshot = Path(sys.argv[1])
special = Path(sys.argv[2])

required_snapshot = [
    "__init__.py",
    "chat_template.jinja",
    "config.json",
    "configuration_nemotron_h.py",
    "generation_config.json",
    "hf_quant_config.json",
    "model.safetensors.index.json",
    "modeling_nemotron_h.py",
    "special_tokens_map.json",
    "super_v3_reasoning_parser.py",
    "tokenizer.json",
    "tokenizer_config.json",
]

missing = [name for name in required_snapshot if not (snapshot / name).exists()]
if missing:
    raise SystemExit("Missing repository files: " + ", ".join(missing))

index = json.loads((snapshot / "model.safetensors.index.json").read_text())
weight_map = index.get("weight_map", {})
if not weight_map:
    raise SystemExit("model.safetensors.index.json contains no weight_map")

shards = sorted(set(weight_map.values()))
missing_shards = [name for name in shards if not (snapshot / name).exists()]
if missing_shards:
    raise SystemExit("Missing model shards: " + ", ".join(missing_shards))

broken = [p for p in snapshot.rglob("*") if p.is_symlink() and not p.exists()]
if broken:
    raise SystemExit(
        "Broken symbolic links:\n" + "\n".join(str(p) for p in broken[:50])
    )

parser_path = special / "super_v3_reasoning_parser.py"
parser_text = parser_path.read_text()
if 'register_module("super_v3")' not in parser_text:
    raise SystemExit("Runtime parser does not register super_v3")

template_text = (special / "chat_template.jinja").read_text()
for marker in ("enable_thinking", "low_effort", "tools"):
    if marker not in template_text:
        raise SystemExit(f"chat_template.jinja is missing marker: {marker}")

generation = json.loads((special / "generation_config.json").read_text())
if float(generation.get("temperature", -1)) != 1.0:
    raise SystemExit("generation_config.json does not specify temperature=1.0")
if float(generation.get("top_p", -1)) != 0.95:
    raise SystemExit("generation_config.json does not specify top_p=0.95")

quant_path = special / "hf_quant_config.json"
if quant_path.stat().st_size < 1000:
    raise SystemExit("hf_quant_config.json is unexpectedly small")

print(f"Snapshot:          {snapshot}")
print(f"Referenced shards: {len(shards)}")
print(f"Model tensors:     {len(weight_map)}")
print(f"Parser:            {parser_path}")
print("Chat template:     enable_thinking / low_effort / tools found")
print("Generation config: temperature=1.0, top_p=0.95")
print("Quant config:      present")
PY

  (
    cd "$SPECIAL_DIR"
    sha256sum -c SHA256SUMS
  )

  log "All required files passed verification"
}

# ==============================================================================
# START / STOP
# ==============================================================================

stop_server() {
  require_command docker
  ensure_docker_access

  if container_exists; then
    log "Stopping container: $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  else
    log "Container is already stopped"
  fi
}

start_server() {
  local -a vllm_args
  local -a offline_env
  local -a token_env
  local -a auth_args
  local default_thinking
  local default_force_content
  local default_low_effort
  local default_template_kwargs

  require_command docker
  require_command curl
  require_command nvidia-smi
  require_command timeout
  require_command python3
  require_command sha256sum

  ensure_docker_access
  ensure_host_gpu
  ensure_image
  ensure_fresh_container_gpu

  model_cache_complete ||
    die "Model is not downloaded. Run: $0 download"

  verify_files

  if container_exists; then
    log "Removing previous ${CONTAINER_NAME} container"
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi

  if port_in_use; then
    die "Port ${API_PORT} is already in use. Stop Qwen/Llama/Nemotron or change API_PORT."
  fi

  default_thinking="$(json_bool "$DEFAULT_ENABLE_THINKING")"
  default_force_content="$(json_bool "$DEFAULT_FORCE_NONEMPTY_CONTENT")"
  default_low_effort="$(json_bool "$DEFAULT_LOW_EFFORT")"

  default_template_kwargs="$(
    printf \
      '{"enable_thinking":%s,"force_nonempty_content":%s,"low_effort":%s}' \
      "$default_thinking" \
      "$default_force_content" \
      "$default_low_effort"
  )"

  vllm_args=(
    serve "$MODEL_ID"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$API_HOST"
    --port "$API_PORT"
    --async-scheduling
    --dtype auto
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --tensor-parallel-size 1
    --pipeline-parallel-size 1
    --data-parallel-size 1
    --trust-remote-code
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-model-len "$MAX_MODEL_LEN"
    --moe-backend "$MOE_BACKEND"
    --mamba-ssm-cache-dtype "$MAMBA_SSM_CACHE_DTYPE"
    --quantization "$QUANTIZATION"
    --swap-space 0
    --chat-template "$RUNTIME_CHAT_TEMPLATE_CONTAINER"
    --default-chat-template-kwargs "$default_template_kwargs"
    --reasoning-parser-plugin "$RUNTIME_PARSER_CONTAINER"
    --reasoning-parser "$REASONING_PARSER"
  )

  if [[ "$ENABLE_CHUNKED_PREFILL" == "1" ]]; then
    vllm_args+=(--enable-chunked-prefill)
  fi

  if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
    vllm_args+=(--enable-prefix-caching)
  fi

  if [[ "$ENABLE_MTP_SPECULATION" == "1" ]]; then
    vllm_args+=(
      --speculative-config
      "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_SPECULATIVE_TOKENS},\"moe_backend\":\"${MTP_MOE_BACKEND}\"}"
    )
  fi

  if [[ "$ENABLE_AUTO_TOOL_CHOICE" == "1" ]]; then
    vllm_args+=(
      --enable-auto-tool-choice
      --tool-call-parser "$TOOL_CALL_PARSER"
    )
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

  log "Starting NVIDIA Nemotron 3 Super on one DGX Spark"
  log "Model:          $MODEL_ID"
  log "Served as:      $SERVED_MODEL_NAME"
  log "Context:        $MAX_MODEL_LEN"
  log "MTP:            $ENABLE_MTP_SPECULATION"
  log "Default think:  $DEFAULT_ENABLE_THINKING"
  log "Reason parser:  $REASONING_PARSER"
  log "Tool parser:    $TOOL_CALL_PARSER"
  log "API:            http://127.0.0.1:${API_PORT}/v1"

  docker run -d \
    --name "$CONTAINER_NAME" \
    --network host \
    --ipc host \
    --shm-size 32g \
    --gpus all \
    -v "${HF_HOME}:/root/.cache/huggingface" \
    -v "${RUNTIME_PARSER_HOST}:${RUNTIME_PARSER_CONTAINER}:ro" \
    -v "${RUNTIME_CHAT_TEMPLATE_HOST}:${RUNTIME_CHAT_TEMPLATE_CONTAINER}:ro" \
    -e VLLM_NVFP4_GEMM_BACKEND=marlin \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \
    -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
    "${offline_env[@]}" \
    "${token_env[@]}" \
    --entrypoint vllm \
    "$VLLM_IMAGE" \
    "${vllm_args[@]}" \
    "${auth_args[@]}" \
    >/dev/null

  wait_for_api

  log "READY"
  printf 'Base URL: %s\n' "$(public_base_url)"
  printf 'Model:    %s\n' "$SERVED_MODEL_NAME"

  if [[ -n "$API_KEY" ]]; then
    printf 'Auth:     Authorization: Bearer <API_KEY>\n'
  else
    printf 'Auth:     disabled\n'
  fi
}

# ==============================================================================
# STATUS / LOGS
# ==============================================================================

show_status() {
  require_command docker
  require_command curl

  ensure_docker_access

  echo "===== CONFIG ====="
  echo "Model ID:         $MODEL_ID"
  echo "Served name:      $SERVED_MODEL_NAME"
  echo "Image:            $VLLM_IMAGE"
  echo "Context:          $MAX_MODEL_LEN"
  echo "Default thinking: $DEFAULT_ENABLE_THINKING"
  echo "Reasoning parser: $REASONING_PARSER"
  echo "Tool parser:      $TOOL_CALL_PARSER"
  echo "MTP speculative:  $ENABLE_MTP_SPECULATION"
  echo "API port:         $API_PORT"
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
    echo "===== VLLM PROCESS ====="
    docker exec "$CONTAINER_NAME" \
      bash -lc "pgrep -af '[v]llm' || true" \
      2>/dev/null || true
  fi

  echo
  echo "===== SPECIAL FILES ====="
  if [[ -d "$SPECIAL_DIR" ]]; then
    ls -lh "$SPECIAL_DIR"
  else
    echo "Not prepared"
  fi

  echo
  echo "===== API ====="

  if curl -fsS \
    --max-time 5 \
    "http://127.0.0.1:${API_PORT}/health" \
    >/dev/null 2>&1; then
    echo "API: HEALTHY"

    api_auth_args
    curl -sS \
      "${API_AUTH_ARGS[@]}" \
      "http://127.0.0.1:${API_PORT}/v1/models"
    echo
  else
    echo "API: NOT READY"
  fi
}

show_logs() {
  require_command docker
  ensure_docker_access

  docker logs \
    --tail "${1:-350}" \
    "$CONTAINER_NAME"
}

# ==============================================================================
# TESTS
# ==============================================================================

test_text() {
  require_command curl
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
      "content": "You are a precise technical assistant."
    },
    {
      "role": "user",
      "content": "Explain three practical uses of hybrid Mamba and attention architectures."
    }
  ],
  "chat_template_kwargs": {
    "enable_thinking": false,
    "force_nonempty_content": true
  },
  "temperature": ${DEFAULT_TEMPERATURE},
  "top_p": ${DEFAULT_TOP_P},
  "max_tokens": 2048
}
EOF

  echo
}

test_reasoning() {
  require_command curl
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
      "content": "Design a resilient deployment plan for an internal coding-agent API. Compare failure modes and tradeoffs."
    }
  ],
  "chat_template_kwargs": {
    "enable_thinking": true,
    "low_effort": true,
    "force_nonempty_content": true
  },
  "temperature": ${DEFAULT_TEMPERATURE},
  "top_p": ${DEFAULT_TOP_P},
  "max_tokens": 8192
}
EOF

  echo
}

test_tools() {
  local choice="${1:-required}"

  [[ "$choice" == "required" || "$choice" == "auto" ]] ||
    die "Tool choice must be required or auto."

  require_command curl
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
      "content": "Inspect src/app.py with the read_file tool before proposing any change."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the current workspace.",
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
  "chat_template_kwargs": {
    "enable_thinking": false,
    "force_nonempty_content": true
  },
  "temperature": ${DEFAULT_TEMPERATURE},
  "top_p": ${DEFAULT_TOP_P},
  "max_tokens": 2048
}
EOF

  echo
  echo
  echo "Expected field: choices[0].message.tool_calls"
  echo "The IDE/agent must execute the tool and return a role=tool message."
}

# ==============================================================================
# CLIENT CONFIG
# ==============================================================================

print_client_config() {
  local detected_ip
  local client_key

  detected_ip="$(detect_advertise_ip)"
  client_key="${API_KEY:-vllm-local}"

  cat <<EOF
==============================================================================
OpenAI-compatible endpoint
==============================================================================

Base URL:       http://${detected_ip}:${API_PORT}/v1
API key:        ${client_key}
Model:          ${SERVED_MODEL_NAME}
Server context: ${MAX_MODEL_LEN}
Input budget:   ${CLIENT_CONTEXT_TOKENS}
Max output:     ${CLIENT_MAX_OUTPUT_TOKENS}

export OPENAI_BASE_URL="http://${detected_ip}:${API_PORT}/v1"
export OPENAI_API_KEY="${client_key}"
export OPENAI_MODEL="${SERVED_MODEL_NAME}"

==============================================================================
Recommended VS Code / coding-agent settings
==============================================================================

Provider:          OpenAI Compatible
Base URL:          http://${detected_ip}:${API_PORT}/v1
Model ID:          ${SERVED_MODEL_NAME}
API Key:           ${client_key}
Context window:    ${MAX_MODEL_LEN}
Max input tokens:  ${CLIENT_CONTEXT_TOKENS}
Max output tokens: ${CLIENT_MAX_OUTPUT_TOKENS}
Supports tools:    yes
Reasoning model:   yes
Temperature:       ${DEFAULT_TEMPERATURE}
Top P:             ${DEFAULT_TOP_P}

Coding agents should send:

{
  "chat_template_kwargs": {
    "enable_thinking": false,
    "force_nonempty_content": true
  }
}

Reasoning requests can override with:

{
  "chat_template_kwargs": {
    "enable_thinking": true,
    "low_effort": true,
    "force_nonempty_content": true
  }
}

Important:
- vLLM returns structured tool_calls.
- The agent executes filesystem, shell, browser, Git, and network tools.
- Use workspace sandboxing, allowlists, timeouts, and confirmations.
- Keep temperature=1.0 and top_p=0.95 as recommended by the model card.
EOF
}

# ==============================================================================
# HELP
# ==============================================================================

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
  $(basename "$0") start
  $(basename "$0") stop
  $(basename "$0") restart
  $(basename "$0") status
  $(basename "$0") logs [number_of_lines]
  $(basename "$0") test-text
  $(basename "$0") test-reasoning
  $(basename "$0") test-tools [required|auto]
  $(basename "$0") client-config

First setup:
  1. Edit USER CONFIGURATION near the top.
  2. Run: ./$(basename "$0") download
  3. Run: ./$(basename "$0") verify-files
  4. Stop any other model using this GPU or port 8000.
  5. Run: ./$(basename "$0") start
  6. Run: ./$(basename "$0") test-tools required

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
  download)
    download_model
    ;;
  verify-files)
    sync_special_files
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
    show_logs "${2:-350}"
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
