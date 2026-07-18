#!/usr/bin/env bash
#
# qwen3-coder-next-single.sh
#
# Single-DGX-Spark controller for:
#   ucbye/Qwen3-Coder-Next-NVFP4-GB10
#
# Commands:
#   ./qwen3-coder-next-single.sh download
#   ./qwen3-coder-next-single.sh start
#   ./qwen3-coder-next-single.sh stop
#   ./qwen3-coder-next-single.sh restart
#   ./qwen3-coder-next-single.sh status
#   ./qwen3-coder-next-single.sh logs
#   ./qwen3-coder-next-single.sh test
#   ./qwen3-coder-next-single.sh test-tools required
#   ./qwen3-coder-next-single.sh client-config
#
set -Eeuo pipefail

SCRIPT_VERSION="${SCRIPT_VERSION:-3.0.0}"

# ==============================================================================
# USER CONFIGURATION — edit values here
# ==============================================================================

MODEL_ID="ucbye/Qwen3-Coder-Next-NVFP4-GB10"
SERVED_MODEL_NAME="qwen3-coder-next"

# Image verified by the model deployment card for GB10 / CUDA 13.
# Pin a tested digest before long-term production use.
VLLM_IMAGE="vllm/vllm-openai:cu130-nightly"

CONTAINER_NAME="qwen3-coder-next-vllm"

# The model supports 262,144 tokens natively.
# 131,072 is a balanced starting point for IDE and agent workloads.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="0.86"

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
ADVERTISE_IP="${ADVERTISE_IP:-}"
ADVERTISE_INTERFACE="${ADVERTISE_INTERFACE:-}"
ROUTE_PROBE_IP="${ROUTE_PROBE_IP:-1.1.1.1}"
CLIENT_OVERHEAD_TOKENS="${CLIENT_OVERHEAD_TOKENS:-8192}"

# Leave empty for no API authentication.
# Set a long random value when sharing the endpoint with a team.
API_KEY="${API_KEY:-}"

# Tool calling:
# The exact model card currently recommends qwen3_coder.
TOOL_CALL_PARSER="qwen3_coder"
ENABLE_AUTO_TOOL_CHOICE="1"

# Agent/IDE optimizations.
KV_CACHE_DTYPE="fp8"
ATTENTION_BACKEND="flashinfer"
ENABLE_CHUNKED_PREFILL="1"
ENABLE_PREFIX_CACHING="1"

# Set to 1 after the model is fully cached locally.
# Change to 0 when downloading or updating model files.
HF_HUB_OFFLINE="1"

# Client-side budget. Input + output + tool/chat overhead must remain below
# MAX_MODEL_LEN.
CLIENT_CONTEXT_TOKENS="${CLIENT_CONTEXT_TOKENS:-auto}"
CLIENT_MAX_OUTPUT_TOKENS="${CLIENT_MAX_OUTPUT_TOKENS:-8192}"

# Startup can take several minutes.
API_WAIT_SECONDS="1800"

# ==============================================================================
# PATHS
# ==============================================================================

CURRENT_USER="${SUDO_USER:-${USER:-$(id -un)}}"
USER_HOME="$(getent passwd "$CURRENT_USER" | cut -d: -f6)"

[[ -n "$USER_HOME" ]] || {
  echo "ERROR: Cannot resolve home directory for ${CURRENT_USER}." >&2
  exit 1
}

HF_HOME="${USER_HOME}/.cache/huggingface"
MODEL_CACHE_NAME="models--${MODEL_ID//\//--}"
MODEL_CACHE_PATH="${HF_HOME}/hub/${MODEL_CACHE_NAME}"

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
# Single-node network and runtime overrides
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
    ip -4 -o addr show scope global 2>/dev/null |
      awk '
        $2 !~ /^(lo|docker|br-|veth|virbr|cni|flannel|tailscale|zt|wg)/ {
          split($4, a, "/")
          print a[1]
          exit
        }
      '
    return
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
  if [[ "$CLIENT_CONTEXT_TOKENS" == "auto" ]]; then
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
  validate_positive_integer "CLIENT_MAX_OUTPUT_TOKENS" "$CLIENT_MAX_OUTPUT_TOKENS"
  validate_positive_integer "CLIENT_OVERHEAD_TOKENS" "$CLIENT_OVERHEAD_TOKENS"

  if [[ "$CLIENT_CONTEXT_TOKENS" != "auto" ]]; then
    validate_positive_integer "CLIENT_CONTEXT_TOKENS" "$CLIENT_CONTEXT_TOKENS"
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
  echo "Bind address:       $API_HOST"
  echo "API port:           $API_PORT"
  echo "Advertise IP:       $advertise_ip"
  echo "Advertise interface:${advertise_interface:+ $advertise_interface}"
  echo "Selected endpoint:  http://${advertise_ip}:${API_PORT}/v1"
  echo
  echo "Binding the same port on different DGX hosts is safe because each host"
  echo "has a different IP address. A conflict occurs only when two processes"
  echo "on the same host try to bind the same IP:port."
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
  # /dev/tcp avoids requiring ss/lsof.
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
      docker logs --tail 250 "$CONTAINER_NAME" 2>/dev/null || true
      die "The vLLM container stopped before the API became ready."
    fi

    sleep 5
  done

  docker logs --tail 250 "$CONTAINER_NAME" 2>/dev/null || true
  die "API did not become ready within ${API_WAIT_SECONDS} seconds."
}

# ==============================================================================
# DOWNLOAD
# ==============================================================================

download_model() {
  require_command docker
  require_command nvidia-smi

  ensure_docker_access
  ensure_host_gpu
  ensure_image

  mkdir -p "$HF_HOME"

  log "Downloading ${MODEL_ID}"
  log "Destination: ${HF_HOME}"

  docker run --rm \
    --user "$(id -u "$CURRENT_USER"):$(id -g "$CURRENT_USER")" \
    -e HOME=/tmp \
    -e HF_HOME=/cache \
    -e MODEL_ID="$MODEL_ID" \
    -v "${HF_HOME}:/cache" \
    --entrypoint python3 \
    "$VLLM_IMAGE" \
    -c '
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["MODEL_ID"],
)
'

  if ! model_cache_complete; then
    die "Download command finished, but the model cache looks incomplete."
  fi

  log "Model download is complete"
  du -sh "$MODEL_CACHE_PATH"
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
  local -a auth_args

  require_command docker
  require_command curl
  require_command nvidia-smi
  require_command timeout

  ensure_docker_access
  ensure_host_gpu
  ensure_image
  ensure_fresh_container_gpu

  model_cache_complete ||
    die "Model is not downloaded. Run: $0 download"

  if container_exists; then
    log "Removing previous ${CONTAINER_NAME} container"
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi

  if port_in_use; then
    die "Port ${API_PORT} is already in use. Stop the previous Llama/vLLM server or change API_PORT."
  fi

  vllm_args=(
    serve "$MODEL_ID"
    --served-model-name "$SERVED_MODEL_NAME"
    --dtype auto
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --attention-backend "$ATTENTION_BACKEND"
    --host "$API_HOST"
    --port "$API_PORT"
  )

  if [[ "$ENABLE_CHUNKED_PREFILL" == "1" ]]; then
    vllm_args+=(--enable-chunked-prefill)
  fi

  if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
    vllm_args+=(--enable-prefix-caching)
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

  log "Starting Qwen3-Coder-Next on one DGX Spark"
  log "Model:      $MODEL_ID"
  log "Served as:  $SERVED_MODEL_NAME"
  log "Context:    $MAX_MODEL_LEN"
  log "API:        http://127.0.0.1:${API_PORT}/v1"
  log "Tool parser: $TOOL_CALL_PARSER"

  docker run -d \
    --name "$CONTAINER_NAME" \
    --network host \
    --ipc host \
    --shm-size 32g \
    --gpus all \
    -v "${HF_HOME}:/root/.cache/huggingface" \
    -e VLLM_NVFP4_GEMM_BACKEND=marlin \
    -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
    -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
    -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    "${offline_env[@]}" \
    --entrypoint vllm \
    "$VLLM_IMAGE" \
    "${vllm_args[@]}" \
    "${auth_args[@]}" \
    >/dev/null

  wait_for_api

  log "READY"
  printf 'Base URL: http://%s:%s/v1\n' \
    "$(detect_advertise_ip)" \
    "$API_PORT"
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
  local advertise_ip
  advertise_ip="$(detect_advertise_ip)"

  require_command docker
  require_command curl

  ensure_docker_access

  echo "===== CONFIG ====="
  echo "Model ID:       $MODEL_ID"
  echo "Served name:    $SERVED_MODEL_NAME"
  echo "Image:          $VLLM_IMAGE"
  echo "Context:        $MAX_MODEL_LEN"
  echo "Tool parser:    $TOOL_CALL_PARSER"
  echo "API port:       $API_PORT"
  echo "Advertise endpoint: http://${advertise_ip}:${API_PORT}/v1"
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
    --tail "${1:-300}" \
    "$CONTAINER_NAME"
}

# ==============================================================================
# TESTS
# ==============================================================================

test_chat() {
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
      "content": "You are a precise software engineering assistant."
    },
    {
      "role": "user",
      "content": "Write a Python function with type hints that validates an IPv4 address. Include pytest tests."
    }
  ],
  "temperature": 0.2,
  "max_tokens": 1024
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
      "content": "Inspect app.py with the read_file tool before suggesting a fix."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the current workspace.",
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
  "temperature": 0,
  "max_tokens": 512
}
EOF

  echo
  echo
  echo "Expected field: choices[0].message.tool_calls"
  echo "The IDE/agent must execute the tool and send its result back."
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

Environment variables:

export OPENAI_BASE_URL="http://${detected_ip}:${API_PORT}/v1"
export OPENAI_API_KEY="${client_key}"
export OPENAI_MODEL="${SERVED_MODEL_NAME}"

==============================================================================
Recommended VS Code / Agent settings
==============================================================================

Provider:          OpenAI Compatible
Base URL:          http://${detected_ip}:${API_PORT}/v1
Model ID:          ${SERVED_MODEL_NAME}
API Key:           ${client_key}
Context window:    ${MAX_MODEL_LEN}
Max input tokens:  ${CLIENT_CONTEXT_TOKENS}
Max output tokens: ${CLIENT_MAX_OUTPUT_TOKENS}
Supports tools:    yes
Reasoning model:   no

Use these values in Cline, Roo Code, Continue, Kilo Code, Qwen Code,
Hermes Agent, OpenClaw, or another OpenAI-compatible client.

Important:
- vLLM returns structured tool_calls.
- The client/agent executes the filesystem, shell, browser or network tool.
- Validate tool arguments before execution.
- Prefer tool_choice="required" or a named function when schema validity matters.
- tool_choice="auto" is not schema-constrained in vLLM.
EOF
}

# ==============================================================================
# HELP
# ==============================================================================

show_help() {
  cat <<EOF
Usage:
  $(basename "$0") <command> [command-argument] [common-options]

Common options:
  --context TOKENS
  --port PORT
  --bind ADDRESS
  --advertise-ip ADDRESS
  --interface NAME
  --client-input TOKENS|auto
  --client-output TOKENS

Examples:
  $(basename "$0") start --context 65536 --port 8001
  $(basename "$0") start --interface enp1s0 --port 8000
  $(basename "$0") client-config --advertise-ip 192.168.101.127
  $(basename "$0") network-info

  $(basename "$0") download
  $(basename "$0") start
  $(basename "$0") stop
  $(basename "$0") restart
  $(basename "$0") status
  $(basename "$0") logs [number_of_lines]
  $(basename "$0") test
  $(basename "$0") test-tools [required|auto]
  $(basename "$0") client-config

First setup:
  1. Edit USER CONFIGURATION near the top of this file.
  2. Run: ./$(basename "$0") download
  3. Stop the existing Llama/vLLM stack if it uses the same GPU or port.
  4. Run: ./$(basename "$0") start
  5. Run: ./$(basename "$0") test-tools required

After reboot:
  ./$(basename "$0") start
EOF
}

# ==============================================================================
# ENTRY POINT
# ==============================================================================

COMMAND="${1:-help}"
if (( $# )); then
  shift
fi

parse_common_options "$@"
set -- "${REMAINING_ARGS[@]}"

case "$COMMAND" in
  download)
    download_model
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
    show_logs "${1:-300}"
    ;;
  test)
    test_chat
    ;;
  test-tools)
    test_tools "${1:-required}"
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
