#!/usr/bin/env bash
#
# gemma4-31b-single.sh
#
# Single-DGX-Spark controller for:
#   nvidia/Gemma-4-31B-IT-NVFP4
#
# Features:
#   - ModelOpt NVFP4 loading through vLLM
#   - Reasoning (gemma4 parser)
#   - OpenAI-compatible tool calling (gemma4 parser)
#   - Text, image, and video input
#   - Structured output support through the OpenAI-compatible API
#   - Model/special-file integrity checks
#
set -Eeuo pipefail

SCRIPT_VERSION="${SCRIPT_VERSION:-3.0.0}"

# ==============================================================================
# USER CONFIGURATION
# ==============================================================================


MODEL_ID="nvidia/Gemma-4-31B-IT-NVFP4"
MODEL_REVISION="main"
SERVED_MODEL_NAME="gemma4-31b-nvfp4"

VLLM_IMAGE="vllm/vllm-openai:gemma4-cu130"
CONTAINER_NAME="gemma4-31b-nvfp4-single"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="0.82"
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

MEDIA_DIR="${HOME}/gemma4-media"
API_WAIT_SECONDS="1800"

# ==============================================================================
# PATHS
# ==============================================================================

CURRENT_USER="${SUDO_USER:-${USER:-$(id -un)}}"
USER_HOME="$(getent passwd "$CURRENT_USER" | cut -d: -f6)"
[[ -n "$USER_HOME" ]] || { echo "Cannot resolve home directory." >&2; exit 1; }

BASE_DIR="${USER_HOME}/gemma4-31b-single"
HF_HOME="${USER_HOME}/.cache/huggingface"
MODEL_CACHE_NAME="models--${MODEL_ID//\//--}"
MODEL_CACHE_PATH="${HF_HOME}/hub/${MODEL_CACHE_NAME}"

SPECIAL_DIR="${BASE_DIR}/special-files"
CHAT_TEMPLATE_HOST="${SPECIAL_DIR}/chat_template.jinja"
CHAT_TEMPLATE_CONTAINER="/app/chat_template.jinja"
CHECKSUM_FILE="${SPECIAL_DIR}/SHA256SUMS"

MEDIA_DIR="${MEDIA_DIR/#\~/$USER_HOME}"

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

ensure_image() {
  if ! docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1; then
    log "Pulling $VLLM_IMAGE"
    docker pull "$VLLM_IMAGE"
  fi
}

ensure_gpu() {
  nvidia-smi >/dev/null 2>&1 || die "Host nvidia-smi failed."

  docker run --rm \
    --gpus all \
    --entrypoint nvidia-smi \
    "$VLLM_IMAGE" >/dev/null 2>&1 ||
    die "A fresh container cannot access the GPU."
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
  timeout 1 bash -c "</dev/tcp/127.0.0.1/${API_PORT}" 2>/dev/null
}

api_auth_args() {
  API_AUTH_ARGS=()
  if [[ -n "$API_KEY" ]]; then
    API_AUTH_ARGS=(-H "Authorization: Bearer ${API_KEY}")
  fi
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
      -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' |
      sort -nr |
      awk 'NR==1 {$1=""; sub(/^ /,""); print; exit}'
  )"
  [[ -n "$snapshot" ]] || die "No model snapshot found."
  readlink -f "$snapshot"
}

wait_for_api() {
  local deadline=$((SECONDS + API_WAIT_SECONDS))

  log "Waiting for vLLM API"

  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 5 \
      "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
      return
    fi

    if ! container_running; then
      docker logs --tail 300 "$CONTAINER_NAME" 2>/dev/null || true
      die "Container stopped before API became ready."
    fi

    sleep 5
  done

  docker logs --tail 300 "$CONTAINER_NAME" 2>/dev/null || true
  die "API did not become healthy within ${API_WAIT_SECONDS}s."
}

prepare_media_file() {
  local source="$1"
  local clean_name
  local destination

  [[ -f "$source" ]] || die "Media file not found: $source"
  mkdir -p "$MEDIA_DIR"

  clean_name="$(basename "$source" | tr -cs 'A-Za-z0-9._-' '_')"
  destination="${MEDIA_DIR}/${clean_name}"

  if [[ "$(readlink -f "$source")" != \
        "$(readlink -f "$destination" 2>/dev/null || true)" ]]; then
    cp -f "$source" "$destination"
  fi

  chmod 0644 "$destination"
  printf '/media/%s' "$clean_name"
}

# ==============================================================================
# DOWNLOAD / VERIFY
# ==============================================================================

download_model() {
  require_command docker
  ensure_docker_access
  ensure_image

  mkdir -p "$HF_HOME" "$SPECIAL_DIR"

  local -a token_env=()
  if [[ -n "$HF_TOKEN" ]]; then
    token_env=(-e "HF_TOKEN=${HF_TOKEN}")
  fi

  log "Downloading $MODEL_ID"

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

  prepare_special_files
  verify_files

  log "Download complete"
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

  (
    cd "$SPECIAL_DIR"
    sha256sum \
      chat_template.jinja \
      config.json \
      generation_config.json \
      hf_quant_config.json \
      model.safetensors.index.json \
      processor_config.json \
      tokenizer.json \
      tokenizer_config.json \
      > SHA256SUMS
  )
}

verify_files() {
  require_command python3
  require_command sha256sum

  model_cache_complete ||
    die "Model is not downloaded. Run: $0 download"

  local snapshot
  snapshot="$(resolve_snapshot_dir)"

  if [[ ! -f "$CHECKSUM_FILE" ]]; then
    prepare_special_files
  fi

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
if not weight_map:
    raise SystemExit("Safetensors index contains no weight_map")

shards = sorted(set(weight_map.values()))
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
        raise SystemExit(f"chat_template.jinja missing marker: {marker}")

quant = json.loads((special / "hf_quant_config.json").read_text())
if not quant:
    raise SystemExit("hf_quant_config.json is empty")

processor = json.loads((special / "processor_config.json").read_text())
if not processor:
    raise SystemExit("processor_config.json is empty")

print(f"Snapshot:          {snapshot.name}")
print(f"Indexed tensors:   {len(weight_map)}")
print(f"Referenced shards: {len(shards)}")
print("Template:          thinking/tools/image/video markers found")
print("Quant config:      present")
print("Processor config:  present")
PY

  (
    cd "$SPECIAL_DIR"
    sha256sum -c SHA256SUMS
  )

  log "Model and special files verified"
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
    log "Server already stopped"
  fi
}

start_server() {
  local -a args
  local -a offline_env
  local -a token_env
  local -a auth_args

  require_command docker
  require_command curl
  require_command timeout
  require_command nvidia-smi

  ensure_docker_access
  ensure_image
  ensure_gpu
  verify_files

  mkdir -p "$MEDIA_DIR"

  if container_exists; then
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi

  if port_in_use; then
    die "Port ${API_PORT} is already in use."
  fi

  args=(
    serve "$MODEL_ID"
    --served-model-name "$SERVED_MODEL_NAME"
    --quantization modelopt
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
    --allowed-local-media-path /media
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

  log "Starting Gemma 4 31B NVFP4 on one DGX Spark"

  docker run -d \
    --name "$CONTAINER_NAME" \
    --network host \
    --ipc host \
    --shm-size 32g \
    --gpus all \
    -v "${HF_HOME}:/root/.cache/huggingface" \
    -v "${CHAT_TEMPLATE_HOST}:${CHAT_TEMPLATE_CONTAINER}:ro" \
    -v "${MEDIA_DIR}:/media:ro" \
    "${offline_env[@]}" \
    "${token_env[@]}" \
    --entrypoint vllm \
    "$VLLM_IMAGE" \
    "${args[@]}" \
    "${auth_args[@]}" \
    >/dev/null

  wait_for_api

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
  echo "Model:       $MODEL_ID"
  echo "Served name: $SERVED_MODEL_NAME"
  echo "Image:       $VLLM_IMAGE"
  echo "Context:     $MAX_MODEL_LEN"
  echo "Tools:       $TOOL_CALL_PARSER"
  echo "Reasoning:   $REASONING_PARSER"
  echo

  echo "===== CONTAINER ====="
  docker ps -a \
    --filter "name=^/${CONTAINER_NAME}$" \
    --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'

  if container_running; then
    echo
    docker exec "$CONTAINER_NAME" nvidia-smi 2>/dev/null || true
    echo
    docker exec "$CONTAINER_NAME" \
      bash -lc "pgrep -af '[v]llm serve' || true" \
      2>/dev/null || true
  fi

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
  docker logs --tail "${1:-350}" "$CONTAINER_NAME"
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
          "properties": {
            "path": {"type": "string"}
          },
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

test_image() {
  local source="${1:-}"
  [[ -n "$source" ]] || die "Usage: $0 test-image /path/to/image"
  local media_path
  media_path="$(prepare_media_file "$source")"

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
        {"type": "image_url", "image_url": {"url": "file://${media_path}"}},
        {"type": "text", "text": "Describe this image accurately and identify visible text."}
      ]
    }
  ],
  "chat_template_kwargs": {"enable_thinking": false},
  "max_tokens": 2048
}
EOF
  echo
}

test_video() {
  local source="${1:-}"
  [[ -n "$source" ]] || die "Usage: $0 test-video /path/to/video"
  local media_path
  media_path="$(prepare_media_file "$source")"

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
        {"type": "video_url", "video_url": {"url": "file://${media_path}"}},
        {"type": "text", "text": "Summarize this video in chronological order."}
      ]
    }
  ],
  "chat_template_kwargs": {"enable_thinking": false},
  "max_tokens": 4096
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
Supports tools:    yes
Supports reasoning: yes
Supports images:   yes
Supports video:    yes
Supports audio:    no

Coding/tools:
  chat_template_kwargs.enable_thinking = false

Reasoning:
  chat_template_kwargs.enable_thinking = true
  temperature = 1.0
  top_p = 0.95
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
  $(basename "$0") start
  $(basename "$0") stop
  $(basename "$0") restart
  $(basename "$0") status
  $(basename "$0") logs [lines]
  $(basename "$0") test-text
  $(basename "$0") test-reasoning
  $(basename "$0") test-tools [required|auto]
  $(basename "$0") test-image /path/to/image
  $(basename "$0") test-video /path/to/video
  $(basename "$0") client-config
EOF
}

parse_common_options "$@"
set -- "${REMAINING_ARGS[@]}"

case "${1:-help}" in
  download) download_model ;;
  verify-files) prepare_special_files; verify_files ;;
  start) start_server ;;
  stop) stop_server ;;
  restart) stop_server; start_server ;;
  status) show_status ;;
  logs) show_logs "${2:-350}" ;;
  test-text) test_text ;;
  test-reasoning) test_reasoning ;;
  test-tools) test_tools "${2:-required}" ;;
  test-image) test_image "${2:-}" ;;
  test-video) test_video "${2:-}" ;;
  client-config) print_client_config ;;
  network-info)
    network_info
    ;;
  help|-h|--help) show_help ;;
  *) show_help; exit 1 ;;
esac
