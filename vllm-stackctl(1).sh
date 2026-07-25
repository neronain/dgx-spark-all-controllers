#!/usr/bin/env bash
#
# vllm-stackctl.sh
# Single-file controller for:
#   - Ray Head on Master
#   - Ray Worker over SSH
#   - vLLM OpenAI-compatible API
#   - Llama 3.3 JSON tool calling
#
# Run this file ONLY on the Master:
#   chmod +x vllm-stackctl.sh
#   ./vllm-stackctl.sh start
#
set -Eeuo pipefail

# ==============================================================================
# USER CONFIGURATION — edit values in this section
# ==============================================================================

MASTER_IP="10.100.152.1"
WORKER_IP="10.100.152.2"
SSH_USER="neronain"

VLLM_IMAGE="nvcr.io/nvidia/vllm:26.05-py3"
MODEL_ID="meta-llama/Llama-3.3-70B-Instruct"

# Keep the existing full model ID so current VS Code clients continue to work.
SERVED_MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"

TENSOR_PARALLEL_SIZE="2"
MAX_MODEL_LEN="16384"
GPU_MEMORY_UTILIZATION="0.80"

API_HOST="0.0.0.0"
API_PORT="8000"

# Empty means no API authentication.
# For team/LAN usage, set a strong value and use:
# Authorization: Bearer <API_KEY>
API_KEY=""

# Tool calling for Meta Llama 3.x.
ENABLE_TOOL_CALLING="1"
TOOL_CALL_PARSER="llama3_json"

# The official Llama 3.1 JSON tool template is also used for Llama 3.3.
USE_CUSTOM_CHAT_TEMPLATE="1"

# Repeated agent system prompts and tool schemas benefit from prefix caching.
ENABLE_PREFIX_CACHING="1"

# The model is already fully cached on both nodes.
# Set to 0 when downloading/updating model files.
HF_HUB_OFFLINE="1"

# Agent-side recommendations printed by the "agent-config" command.
AGENT_CONTEXT_TOKENS="14336"
AGENT_MAX_OUTPUT_TOKENS="1024"

# Increase if model initialization takes longer.
HEAD_WAIT_SECONDS="600"
CLUSTER_WAIT_SECONDS="900"
API_WAIT_SECONDS="1800"

# ==============================================================================
# PATHS AND PINNED ASSETS
# ==============================================================================

MASTER_HOME="/home/${SSH_USER}"
BASE_DIR="${MASTER_HOME}/vllm-stack"
HF_HOME="${MASTER_HOME}/.cache/huggingface"

RUN_CLUSTER="${BASE_DIR}/run_cluster.sh"
REMOTE_SCRIPT="${BASE_DIR}/.vllm-stackctl.remote.sh"

HEAD_SESSION="vllm-head"
WORKER_SESSION="vllm-worker"
API_SESSION="vllm-api"

VLLM_COMMIT="51c1ee9b7c8acbba4899a8ebffd390685d171946"
RUN_CLUSTER_URL="https://raw.githubusercontent.com/vllm-project/vllm/${VLLM_COMMIT}/examples/ray_serving/run_cluster.sh"

CHAT_TEMPLATE_NAME="tool_chat_template_llama3.1_json.jinja"
CHAT_TEMPLATE_HOST="${HF_HOME}/${CHAT_TEMPLATE_NAME}"
CHAT_TEMPLATE_CONTAINER="/root/.cache/huggingface/${CHAT_TEMPLATE_NAME}"
CHAT_TEMPLATE_URL="https://raw.githubusercontent.com/vllm-project/vllm/${VLLM_COMMIT}/examples/${CHAT_TEMPLATE_NAME}"

# NVIDIA's Stacked Sparks recipe installs Ray when each container starts.
# For reproducibility, replace this with an exact tested version later, e.g.
# RAY_PACKAGE="ray[default]==2.x.y"
RAY_PACKAGE="ray[default]>=2.9"

NCCL_DEBUG="WARN"

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SSH_TARGET="${SSH_USER}@${WORKER_IP}"
MODEL_CACHE_NAME="models--${MODEL_ID//\//--}"
MODEL_CACHE_PATH="${HF_HOME}/hub/${MODEL_CACHE_NAME}"

# ==============================================================================
# HELPERS
# ==============================================================================

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 ||
    die "Missing command: $1"
}

ssh_worker() {
  ssh \
    -o BatchMode=yes \
    -o ConnectTimeout=8 \
    -o ServerAliveInterval=10 \
    -o ServerAliveCountMax=3 \
    "$SSH_TARGET" "$@"
}

local_node_container() {
  docker ps \
    --filter "ancestor=${VLLM_IMAGE}" \
    --format '{{.Names}}' |
    grep -E '^node-[0-9]+$' |
    head -n 1 || true
}

detect_interface() {
  local ip="$1"
  ip -o -4 addr show |
    awk -v target="$ip" '$4 ~ ("^" target "/") {print $2; exit}'
}

api_curl_args() {
  API_CURL_ARGS=(-sS)
  if [[ -n "$API_KEY" ]]; then
    API_CURL_ARGS+=(-H "Authorization: Bearer ${API_KEY}")
  fi
}

model_cache_exists() {
  [[ -d "${MODEL_CACHE_PATH}/snapshots" ]] &&
    find "${MODEL_CACHE_PATH}/snapshots" -mindepth 1 -maxdepth 1 \
      -type d -print -quit | grep -q .
}

check_running_on_master() {
  local interface
  interface="$(detect_interface "$MASTER_IP")"
  [[ -n "$interface" ]] ||
    die "This machine does not own MASTER_IP=${MASTER_IP}. Run this controller on the Master."
}

patch_run_cluster_for_ray() {
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
    raise SystemExit(
        f"Could not patch {path}: expected RAY_START_CMD line was not found."
    )

path.write_text(text.replace(needle, replacement, 1))
PY
}

ensure_local_assets() {
  require_command curl
  require_command python3

  mkdir -p "$BASE_DIR" "$HF_HOME"

  if [[ ! -s "$RUN_CLUSTER" ]]; then
    log "Downloading pinned run_cluster.sh"
    curl -fsSL "$RUN_CLUSTER_URL" -o "$RUN_CLUSTER"
  fi

  patch_run_cluster_for_ray
  chmod 0755 "$RUN_CLUSTER"

  if [[ "$ENABLE_TOOL_CALLING" == "1" &&
        "$USE_CUSTOM_CHAT_TEMPLATE" == "1" &&
        ! -s "$CHAT_TEMPLATE_HOST" ]]; then
    log "Downloading Llama JSON tool-call chat template"
    curl -fsSL "$CHAT_TEMPLATE_URL" -o "$CHAT_TEMPLATE_HOST"
    chmod 0644 "$CHAT_TEMPLATE_HOST"
  fi
}

ensure_image_local() {
  if ! docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1; then
    log "Pulling image on Master: $VLLM_IMAGE"
    docker pull "$VLLM_IMAGE"
  fi
}

ensure_worker_assets() {
  log "Preparing Worker assets"

  ssh_worker \
    "mkdir -p '$BASE_DIR' '$HF_HOME' &&
     chmod 0755 '$BASE_DIR'"

  scp -q \
    -o BatchMode=yes \
    -o ConnectTimeout=8 \
    "$RUN_CLUSTER" \
    "${SSH_TARGET}:${RUN_CLUSTER}"

  scp -q \
    -o BatchMode=yes \
    -o ConnectTimeout=8 \
    "$SCRIPT_PATH" \
    "${SSH_TARGET}:${REMOTE_SCRIPT}"

  ssh_worker \
    "chmod 0755 '$RUN_CLUSTER' '$REMOTE_SCRIPT'"
}

preflight_master() {
  log "Master preflight"

  require_command docker
  require_command tmux
  require_command curl
  require_command ssh
  require_command scp
  require_command nvidia-smi
  require_command python3

  docker info >/dev/null 2>&1 ||
    die "Docker is not available to user ${USER} on Master."

  nvidia-smi >/dev/null 2>&1 ||
    die "nvidia-smi failed on Master host."

  ensure_image_local

  docker run --rm \
    --gpus all \
    --entrypoint nvidia-smi \
    "$VLLM_IMAGE" >/dev/null 2>&1 ||
    die "A fresh Docker container cannot access the Master GPU."

  model_cache_exists ||
    die "Model cache is missing or incomplete on Master: ${MODEL_CACHE_PATH}"
}

preflight_worker() {
  log "Worker preflight"

  ssh_worker "command -v docker >/dev/null &&
              command -v tmux >/dev/null &&
              command -v nvidia-smi >/dev/null" ||
    die "Worker requires docker, tmux and nvidia-smi."

  ssh_worker "docker info >/dev/null 2>&1" ||
    die "Docker is not available to ${SSH_USER} on Worker."

  ssh_worker "nvidia-smi >/dev/null 2>&1" ||
    die "nvidia-smi failed on Worker host."

  if ! ssh_worker "docker image inspect '$VLLM_IMAGE' >/dev/null 2>&1"; then
    log "Pulling image on Worker: $VLLM_IMAGE"
    ssh_worker "docker pull '$VLLM_IMAGE'"
  fi

  ssh_worker \
    "docker run --rm --gpus all \
       --entrypoint nvidia-smi '$VLLM_IMAGE' >/dev/null 2>&1" ||
    die "A fresh Docker container cannot access the Worker GPU."

  ssh_worker \
    "test -d '${MODEL_CACHE_PATH}/snapshots' &&
     find '${MODEL_CACHE_PATH}/snapshots' \
       -mindepth 1 -maxdepth 1 -type d -print -quit |
       grep -q ." ||
    die "Model cache is missing or incomplete on Worker: ${MODEL_CACHE_PATH}"
}

remove_local_node_containers() {
  docker ps -aq \
    --filter "ancestor=${VLLM_IMAGE}" \
    --filter "name=node-" |
    xargs -r docker rm -f >/dev/null 2>&1 || true
}

remove_worker_node_containers() {
  ssh_worker \
    "docker ps -aq \
       --filter 'ancestor=${VLLM_IMAGE}' \
       --filter 'name=node-' |
     xargs -r docker rm -f >/dev/null 2>&1 || true" ||
    true
}

wait_for_head() {
  local deadline=$((SECONDS + HEAD_WAIT_SECONDS))
  local container=""

  log "Waiting for Ray Head" >&2

  while (( SECONDS < deadline )); do
    container="$(local_node_container)"

    if [[ -n "$container" ]] &&
       docker exec "$container" nvidia-smi >/dev/null 2>&1 &&
       docker exec "$container" ray status >/dev/null 2>&1; then
      printf '%s\n' "$container"
      return 0
    fi

    sleep 3
  done

  tmux capture-pane -p -t "$HEAD_SESSION" -S -120 2>/dev/null || true
  die "Ray Head did not become ready within ${HEAD_WAIT_SECONDS}s."
}

ray_cluster_ready() {
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

alive = sum(1 for node in ray.nodes() if node.get("Alive"))
gpus = float(ray.cluster_resources().get("GPU", 0))

sys.exit(0 if alive >= 2 and gpus >= 2 else 1)
' >/dev/null 2>&1
}

wait_for_cluster() {
  local container="$1"
  local deadline=$((SECONDS + CLUSTER_WAIT_SECONDS))

  log "Waiting for 2 Ray nodes and 2 GPUs"

  while (( SECONDS < deadline )); do
    if ray_cluster_ready "$container"; then
      docker exec "$container" ray status
      return 0
    fi

    sleep 4
  done

  docker exec "$container" ray status 2>/dev/null || true
  ssh_worker \
    "tmux capture-pane -p -t '$WORKER_SESSION' -S -120 2>/dev/null || true" ||
    true

  die "Worker did not join with 2 nodes / 2 GPUs within ${CLUSTER_WAIT_SECONDS}s."
}

wait_for_api() {
  local deadline=$((SECONDS + API_WAIT_SECONDS))

  log "Waiting for vLLM API"

  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 5 \
      "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
      return 0
    fi

    sleep 5
  done

  tmux capture-pane -p -t "$API_SESSION" -S -160 2>/dev/null || true
  die "vLLM API did not become healthy within ${API_WAIT_SECONDS}s."
}

# ==============================================================================
# INTERNAL FOREGROUND COMMANDS
# ==============================================================================

head_foreground() {
  local interface
  local -a docker_args

  interface="$(detect_interface "$MASTER_IP")"
  [[ -n "$interface" ]] ||
    die "Cannot find the Master interface for ${MASTER_IP}."

  docker_args=(
    -e "VLLM_HOST_IP=${MASTER_IP}"
    -e "UCX_NET_DEVICES=${interface}"
    -e "NCCL_SOCKET_IFNAME=${interface}"
    -e "OMPI_MCA_btl_tcp_if_include=${interface}"
    -e "GLOO_SOCKET_IFNAME=${interface}"
    -e "TP_SOCKET_IFNAME=${interface}"
    -e "RAY_memory_monitor_refresh_ms=0"
    -e "MASTER_ADDR=${MASTER_IP}"
    -e "NCCL_DEBUG=${NCCL_DEBUG}"
  )

  if [[ "$HF_HUB_OFFLINE" == "1" ]]; then
    docker_args+=(
      -e "HF_HUB_OFFLINE=1"
      -e "TRANSFORMERS_OFFLINE=1"
    )
  fi

  log "Ray Head foreground"
  log "Interface: ${interface}"

  exec bash "$RUN_CLUSTER" \
    "$VLLM_IMAGE" \
    "$MASTER_IP" \
    --head \
    "$HF_HOME" \
    "${docker_args[@]}"
}

worker_foreground() {
  local interface
  local -a docker_args

  interface="$(detect_interface "$WORKER_IP")"
  [[ -n "$interface" ]] ||
    die "Cannot find the Worker interface for ${WORKER_IP}."

  docker_args=(
    -e "VLLM_HOST_IP=${WORKER_IP}"
    -e "UCX_NET_DEVICES=${interface}"
    -e "NCCL_SOCKET_IFNAME=${interface}"
    -e "OMPI_MCA_btl_tcp_if_include=${interface}"
    -e "GLOO_SOCKET_IFNAME=${interface}"
    -e "TP_SOCKET_IFNAME=${interface}"
    -e "RAY_memory_monitor_refresh_ms=0"
    -e "MASTER_ADDR=${MASTER_IP}"
    -e "NCCL_DEBUG=${NCCL_DEBUG}"
  )

  if [[ "$HF_HUB_OFFLINE" == "1" ]]; then
    docker_args+=(
      -e "HF_HUB_OFFLINE=1"
      -e "TRANSFORMERS_OFFLINE=1"
    )
  fi

  log "Ray Worker foreground"
  log "Interface: ${interface}"
  log "Head: ${MASTER_IP}:6379"

  exec bash "$RUN_CLUSTER" \
    "$VLLM_IMAGE" \
    "$MASTER_IP" \
    --worker \
    "$HF_HOME" \
    "${docker_args[@]}"
}

worker_start_internal() {
  local tmux_command

  tmux kill-session -t "$WORKER_SESSION" 2>/dev/null || true

  docker ps -aq \
    --filter "ancestor=${VLLM_IMAGE}" \
    --filter "name=node-" |
    xargs -r docker rm -f >/dev/null 2>&1 || true

  printf -v tmux_command \
    'exec bash %q _worker-foreground' \
    "$SCRIPT_PATH"

  tmux new-session \
    -d \
    -s "$WORKER_SESSION" \
    "$tmux_command"

  log "Worker session started: ${WORKER_SESSION}"
}

api_foreground() {
  local container
  local -a vllm_args

  container="$(local_node_container)"
  [[ -n "$container" ]] ||
    die "No Head container found."

  while ! ray_cluster_ready "$container"; do
    sleep 4
  done

  vllm_args=(
    vllm serve "$MODEL_ID"
    --served-model-name "$SERVED_MODEL_NAME"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --distributed-executor-backend ray
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --host "$API_HOST"
    --port "$API_PORT"
  )

  if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
    vllm_args+=(--enable-prefix-caching)
  fi

  if [[ "$ENABLE_TOOL_CALLING" == "1" ]]; then
    vllm_args+=(
      --enable-auto-tool-choice
      --tool-call-parser "$TOOL_CALL_PARSER"
    )

    if [[ "$USE_CUSTOM_CHAT_TEMPLATE" == "1" ]]; then
      vllm_args+=(
        --chat-template "$CHAT_TEMPLATE_CONTAINER"
      )
    fi
  fi

  if [[ -n "$API_KEY" ]]; then
    vllm_args+=(--api-key "$API_KEY")
  fi

  log "vLLM API foreground"
  log "Container: ${container}"
  log "Model: ${SERVED_MODEL_NAME}"
  log "Context: ${MAX_MODEL_LEN}"
  log "Tool parser: ${TOOL_CALL_PARSER}"

  exec docker exec \
    -e "HF_HUB_OFFLINE=${HF_HUB_OFFLINE}" \
    "$container" \
    "${vllm_args[@]}"
}

# ==============================================================================
# PUBLIC COMMANDS
# ==============================================================================

stop_all() {
  local container

  log "Stopping vLLM API"

  container="$(local_node_container)"
  if [[ -n "$container" ]]; then
    docker exec "$container" \
      bash -lc "pkill -TERM -f '[v]llm serve' || true" \
      >/dev/null 2>&1 || true
  fi

  tmux kill-session -t "$API_SESSION" 2>/dev/null || true
  sleep 3

  log "Stopping Ray Worker"

  ssh_worker \
    "tmux kill-session -t '$WORKER_SESSION' 2>/dev/null || true" ||
    true

  sleep 3
  remove_worker_node_containers

  log "Stopping Ray Head"

  tmux kill-session -t "$HEAD_SESSION" 2>/dev/null || true
  sleep 3
  remove_local_node_containers

  log "Stack stopped"
}

start_all() {
  local head_container
  local tmux_command

  check_running_on_master
  ensure_local_assets

  stop_all

  preflight_master
  preflight_worker
  ensure_worker_assets

  log "Starting Ray Head"

  printf -v tmux_command \
    'exec bash %q _head-foreground' \
    "$SCRIPT_PATH"

  tmux new-session \
    -d \
    -s "$HEAD_SESSION" \
    "$tmux_command"

  head_container="$(wait_for_head)"
  log "Head container: ${head_container}"

  log "Starting Ray Worker from Master"

  ssh_worker \
    "bash '$REMOTE_SCRIPT' _worker-start"

  wait_for_cluster "$head_container"

  log "Starting vLLM API"

  printf -v tmux_command \
    'exec bash %q _api-foreground' \
    "$SCRIPT_PATH"

  tmux new-session \
    -d \
    -s "$API_SESSION" \
    "$tmux_command"

  wait_for_api

  log "READY"
  printf 'API:        http://%s:%s/v1\n' "$MASTER_IP" "$API_PORT"
  printf 'Model:      %s\n' "$SERVED_MODEL_NAME"
  printf 'Context:    %s\n' "$MAX_MODEL_LEN"
  printf 'Tool calls: %s (%s)\n' "$ENABLE_TOOL_CALLING" "$TOOL_CALL_PARSER"

  if [[ -n "$API_KEY" ]]; then
    printf 'Auth:       Authorization: Bearer <API_KEY>\n'
  else
    printf 'Auth:       disabled\n'
  fi

  printf '\nNext checks:\n'
  printf '  %q status\n' "$SCRIPT_PATH"
  printf '  %q test-tools\n' "$SCRIPT_PATH"
  printf '  %q agent-config\n' "$SCRIPT_PATH"
}

status_all() {
  local container

  check_running_on_master

  echo "===== MASTER TMUX ====="
  tmux list-sessions 2>/dev/null || echo "No Master tmux sessions"

  echo
  echo "===== MASTER CONTAINER / RAY ====="

  container="$(local_node_container)"

  if [[ -z "$container" ]]; then
    echo "No Head container"
  else
    echo "Head container: $container"
    docker exec "$container" nvidia-smi \
      --query-gpu=name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader 2>/dev/null || true
    echo
    docker exec "$container" ray status 2>/dev/null || true
    echo
    docker exec "$container" \
      bash -lc "pgrep -af '[v]llm serve' || true"
  fi

  echo
  echo "===== WORKER ====="

  ssh_worker "
    tmux list-sessions 2>/dev/null || echo 'No Worker tmux sessions'
    echo
    docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
  " 2>/dev/null || echo "Worker unreachable"

  echo
  echo "===== API ====="

  if curl -fsS --max-time 5 \
    "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
    echo "API: HEALTHY"

    api_curl_args
    curl "${API_CURL_ARGS[@]}" \
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
      tmux capture-pane -p -t "$HEAD_SESSION" -S -250
      ;;
    worker)
      ssh_worker \
        "tmux capture-pane -p -t '$WORKER_SESSION' -S -250"
      ;;
    api)
      tmux capture-pane -p -t "$API_SESSION" -S -250
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
      tmux capture-pane -p -t "$API_SESSION" -S -160 2>/dev/null || true
      ;;
    *)
      die "logs target must be: head, worker, api or all"
      ;;
  esac
}

test_tools() {
  local tool_choice="${1:-required}"
  local payload
  local -a curl_args

  [[ "$tool_choice" == "required" || "$tool_choice" == "auto" ]] ||
    die "test-tools mode must be: required or auto"

  payload="$(
    python3 - "$SERVED_MODEL_NAME" "$tool_choice" <<'PY'
import json
import sys

model = sys.argv[1]
tool_choice = sys.argv[2]

print(json.dumps({
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": (
                "Use the get_weather tool to check the weather in Bangkok. "
                "Do not answer from memory."
            ),
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country",
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": False,
                },
            },
        }
    ],
    "tool_choice": tool_choice,
    "max_tokens": 256,
    "temperature": 0,
}))
PY
  )"

  curl_args=(
    -sS
    "http://127.0.0.1:${API_PORT}/v1/chat/completions"
    -H "Content-Type: application/json"
  )

  if [[ -n "$API_KEY" ]]; then
    curl_args+=(-H "Authorization: Bearer ${API_KEY}")
  fi

  log "Tool-call test: tool_choice=${tool_choice}"

  curl "${curl_args[@]}" \
    --data-binary "$payload"

  echo
  echo
  echo "Look for: choices[0].message.tool_calls"
}

print_agent_config() {
  local client_key="${API_KEY:-vllm-local}"

  cat <<EOF
==============================================================================
Hermes Agent: ~/.hermes/config.yaml
==============================================================================

model:
  default: "${SERVED_MODEL_NAME}"
  provider: custom
  base_url: "http://${MASTER_IP}:${API_PORT}/v1"
  api_key: "${client_key}"
  context_length: ${MAX_MODEL_LEN}
  max_tokens: ${AGENT_MAX_OUTPUT_TOKENS}

compression:
  enabled: true
  threshold: 0.50

Or run:
  hermes model
  # Custom endpoint
  # Base URL: http://${MASTER_IP}:${API_PORT}/v1
  # API key: ${client_key}
  # Model: ${SERVED_MODEL_NAME}
  # Context length: ${MAX_MODEL_LEN}

==============================================================================
OpenClaw: ~/.openclaw/openclaw.json (JSON5)
==============================================================================

{
  env: {
    VLLM_API_KEY: "${client_key}",
  },

  models: {
    mode: "merge",
    providers: {
      vllm: {
        baseUrl: "http://${MASTER_IP}:${API_PORT}/v1",
        apiKey: "\${VLLM_API_KEY}",
        api: "openai-completions",
        timeoutSeconds: 1800,
        models: [
          {
            id: "${SERVED_MODEL_NAME}",
            name: "DGX Spark Llama 3.3 70B",
            reasoning: false,
            input: ["text"],
            cost: {
              input: 0,
              output: 0,
              cacheRead: 0,
              cacheWrite: 0,
            },
            contextWindow: ${MAX_MODEL_LEN},
            contextTokens: ${AGENT_CONTEXT_TOKENS},
            maxTokens: ${AGENT_MAX_OUTPUT_TOKENS},
          },
        ],
      },
    },
  },

  agents: {
    defaults: {
      model: {
        primary: "vllm/${SERVED_MODEL_NAME}",
      },
    },
  },
}

Verify:
  export VLLM_API_KEY="${client_key}"
  openclaw models list --provider vllm

==============================================================================
Generic OpenAI-compatible client
==============================================================================

Base URL: http://${MASTER_IP}:${API_PORT}/v1
Model:    ${SERVED_MODEL_NAME}
API key:  ${client_key}

Server tool calling:
  --enable-auto-tool-choice
  --tool-call-parser ${TOOL_CALL_PARSER}
EOF
}

show_help() {
  cat <<EOF
Usage:
  $(basename "$0") start
  $(basename "$0") stop
  $(basename "$0") restart
  $(basename "$0") status
  $(basename "$0") logs [head|worker|api|all]
  $(basename "$0") test-tools [required|auto]
  $(basename "$0") agent-config

Commands:
  start         Start Head, Worker and vLLM from this Master file.
  stop          Stop API, Worker and Head. Model files are preserved.
  restart       Stop and start the complete stack.
  status        Show tmux, containers, Ray resources and API health.
  logs          Print recent logs from tmux sessions.
  test-tools    Test OpenAI tool calling. "required" is the reliable pipeline test.
  agent-config  Print Hermes Agent and OpenClaw configuration examples.

Edit the USER CONFIGURATION section near the top of this file to change:
  model, context length, API port, API key, tool parser and memory utilization.
EOF
}

# ==============================================================================
# ENTRY POINT
# ==============================================================================

case "${1:-help}" in
  start)
    start_all
    ;;
  stop)
    check_running_on_master
    stop_all
    ;;
  restart)
    check_running_on_master
    stop_all
    start_all
    ;;
  status)
    status_all
    ;;
  logs)
    show_logs "${2:-all}"
    ;;
  test-tools)
    test_tools "${2:-required}"
    ;;
  agent-config)
    print_agent_config
    ;;
  _head-foreground)
    head_foreground
    ;;
  _worker-start)
    worker_start_internal
    ;;
  _worker-foreground)
    worker_foreground
    ;;
  _api-foreground)
    api_foreground
    ;;
  help|-h|--help)
    show_help
    ;;
  *)
    show_help
    exit 1
    ;;
esac
