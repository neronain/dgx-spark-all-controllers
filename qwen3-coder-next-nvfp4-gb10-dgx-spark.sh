#!/usr/bin/env bash
# Qwen3-Coder-Next-NVFP4-GB10 — NVIDIA DGX Spark controller
# Runtime: vLLM Docker, OpenAI-compatible API, native qwen3_coder tool calls.
#
# IMPORTANT:
# - Never embed an HF token in this file.
# - Accept the gated repository terms first.
# - Bind defaults to localhost. Use 0.0.0.0 only on a trusted LAN/VPN.

set -Eeuo pipefail

HF_REPO="${HF_REPO:-saricles/Qwen3-Coder-Next-NVFP4-GB10}"
MODEL_REVISION="${MODEL_REVISION:-main}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-coder-next-nvfp4-gb10}"

VLLM_IMAGE="${VLLM_IMAGE:-avarok/dgx-vllm-nvfp4-kernel:v23}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen3-coder-next-nvfp4-gb10}"

USER_HOME="${USER_HOME:-$HOME}"
MODEL_DIR="${MODEL_DIR:-${USER_HOME}/models/qwen3-coder-next-nvfp4-gb10}"
HF_HOME="${HF_HOME:-${USER_HOME}/.cache/huggingface}"
VLLM_CACHE="${VLLM_CACHE:-${USER_HOME}/.cache/vllm}"
HF_VENV="${HF_VENV:-${USER_HOME}/.venvs/hf}"

API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
API_KEY="${API_KEY:-}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
SHM_SIZE="${SHM_SIZE:-32g}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"

HF_TOKEN="${HF_TOKEN:-}"

log() { printf '[%(%H:%M:%S)T] %s\n' -1 "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

detect_advertise_ip() {
  if [[ "$API_HOST" != "0.0.0.0" && "$API_HOST" != "::" ]]; then
    printf '%s' "$API_HOST"
    return
  fi
  local ip_addr
  ip_addr="$(
    ip -4 route get 1.1.1.1 2>/dev/null |
      awk '{for (i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}'
  )"
  [[ -n "$ip_addr" ]] || ip_addr="127.0.0.1"
  printf '%s' "$ip_addr"
}

model_ready() {
  [[ -f "${MODEL_DIR}/config.json" ]] &&
  find "$MODEL_DIR" -maxdepth 1 -type f -name '*.safetensors' \
    -print -quit | grep -q .
}

stored_token_exists() {
  [[ -s "${HF_HOME}/token" ]] || [[ -s "${USER_HOME}/.huggingface/token" ]]
}

ensure_hf_python() {
  if [[ -x "${HF_VENV}/bin/python" ]] &&
     "${HF_VENV}/bin/python" -c 'import huggingface_hub' >/dev/null 2>&1; then
    printf '%s' "${HF_VENV}/bin/python"
    return
  fi

  command -v python3 >/dev/null 2>&1 || die "python3 is required"
  python3 -m venv "$HF_VENV" ||
    die "Install python3-venv, then rerun."
  "${HF_VENV}/bin/python" -m pip install -U pip
  "${HF_VENV}/bin/python" -m pip install -U "huggingface_hub[hf_xet]"
  printf '%s' "${HF_VENV}/bin/python"
}

require_hf_access() {
  if [[ -n "$HF_TOKEN" ]] || stored_token_exists; then
    return
  fi
  cat >&2 <<MSG
The repository is gated.

1. Accept its conditions at:
   https://huggingface.co/${HF_REPO}
2. Then run:
   $0 auth

Alternatively set a NEW read-only token for this shell:
   read -rsp "HF token: " HF_TOKEN; echo; export HF_TOKEN

Never place a token in source code, Git, shell history, or screenshots.
MSG
  exit 1
}

validate_options() {
  [[ "$API_PORT" =~ ^[0-9]+$ ]] || die "Invalid port: ${API_PORT}"
  [[ "$MAX_MODEL_LEN" =~ ^[0-9]+$ ]] || die "Invalid context: ${MAX_MODEL_LEN}"
  [[ "$MAX_NUM_BATCHED_TOKENS" =~ ^[0-9]+$ ]] ||
    die "Invalid max batched tokens"
  [[ "$MAX_NUM_SEQS" =~ ^[0-9]+$ ]] || die "Invalid max seqs"
}

parse_options() {
  REMAINING=()
  while (($#)); do
    case "$1" in
      --bind) API_HOST="$2"; shift 2 ;;
      --bind=*) API_HOST="${1#*=}"; shift ;;
      --port) API_PORT="$2"; shift 2 ;;
      --port=*) API_PORT="${1#*=}"; shift ;;
      --api-key) API_KEY="$2"; shift 2 ;;
      --api-key=*) API_KEY="${1#*=}"; shift ;;
      --context) MAX_MODEL_LEN="$2"; shift 2 ;;
      --context=*) MAX_MODEL_LEN="${1#*=}"; shift ;;
      --gpu-memory-util) GPU_MEMORY_UTIL="$2"; shift 2 ;;
      --gpu-memory-util=*) GPU_MEMORY_UTIL="${1#*=}"; shift ;;
      --max-batched-tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
      --max-batched-tokens=*) MAX_NUM_BATCHED_TOKENS="${1#*=}"; shift ;;
      --max-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
      --max-seqs=*) MAX_NUM_SEQS="${1#*=}"; shift ;;
      --image) VLLM_IMAGE="$2"; shift 2 ;;
      --image=*) VLLM_IMAGE="${1#*=}"; shift ;;
      --model-dir) MODEL_DIR="$2"; shift 2 ;;
      --model-dir=*) MODEL_DIR="${1#*=}"; shift ;;
      *) REMAINING+=("$1"); shift ;;
    esac
  done
  validate_options
}

preflight() {
  command -v docker >/dev/null 2>&1 || die "Docker is not installed"
  docker info >/dev/null 2>&1 ||
    die "Docker daemon is unavailable or this user lacks permission"

  echo "=== DGX Spark preflight ==="
  echo "Docker          : $(docker --version)"
  echo "Architecture    : $(uname -m)"
  echo -n "GPU             : "
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null ||
    echo "nvidia-smi unavailable"

  if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
    echo "NVIDIA runtime  : available"
  else
    echo "NVIDIA runtime  : NOT DETECTED"
  fi

  echo "Docker image    : ${VLLM_IMAGE}"
  echo "Model directory : ${MODEL_DIR}"
  echo "Context         : ${MAX_MODEL_LEN}"
  echo "Bind            : ${API_HOST}:${API_PORT}"

  local free_bytes
  free_bytes=$(df -PB1 "$(dirname "$MODEL_DIR")" | awk 'NR==2 {print $4}')
  echo "Disk free       : $((free_bytes / 1000 / 1000 / 1000)) GB"
  if ((free_bytes < 55000000000)); then
    echo "WARNING: less than 55 GB free; model size is about 45.9 GB." >&2
  fi
}

auth() {
  local py token
  py=$(ensure_hf_python)

  echo "Accept the gated repository conditions before continuing:"
  echo "https://huggingface.co/${HF_REPO}"
  echo ""
  read -rsp "Enter a NEW Hugging Face read-only token: " token
  echo
  [[ -n "$token" ]] || die "No token entered"

  HF_HOME="$HF_HOME" "${HF_VENV}/bin/hf" auth login --token "$token"
  unset token
  echo "Token stored in the local Hugging Face cache."
}

download_model() {
  local mode="${1:-}"
  if [[ "$mode" != "--force" ]] && model_ready; then
    log "Model already exists in ${MODEL_DIR}; skipping."
    echo "Use '$0 download --force' for a fresh snapshot."
    return
  fi

  require_hf_access
  local py
  py=$(ensure_hf_python)

  mkdir -p "$MODEL_DIR" "$HF_HOME"
  export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

  log "Downloading ${HF_REPO} @ ${MODEL_REVISION}"
  HF_HOME="$HF_HOME" \
  HF_TOKEN="$HF_TOKEN" \
  HF_XET_HIGH_PERFORMANCE="$HF_XET_HIGH_PERFORMANCE" \
  MODEL_DIR="$MODEL_DIR" \
  HF_REPO="$HF_REPO" \
  MODEL_REVISION="$MODEL_REVISION" \
  "$py" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["HF_REPO"],
    revision=os.environ["MODEL_REVISION"],
    local_dir=os.environ["MODEL_DIR"],
    token=os.environ.get("HF_TOKEN") or None,
)
PY

  model_ready || die "Required files were not found after download"
  log "Download complete."
}

verify_files() {
  [[ -f "${MODEL_DIR}/config.json" ]] || die "Missing config.json"
  [[ -f "${MODEL_DIR}/tokenizer_config.json" ]] ||
    die "Missing tokenizer_config.json"

  local count total
  count=$(find "$MODEL_DIR" -maxdepth 1 -type f -name '*.safetensors' | wc -l)
  total=$(
    find "$MODEL_DIR" -maxdepth 1 -type f -name '*.safetensors' \
      -printf '%s\n' | awk '{s+=$1} END {printf "%.0f", s+0}'
  )

  ((count >= 1)) || die "No safetensors files found"
  ((total >= 40000000000)) ||
    die "Safetensors total is unexpectedly small: ${total}"

  echo "verify-files: PASS"
  echo "  Safetensors files : ${count}"
  echo "  Safetensors bytes : ${total}"

  python3 - "${MODEL_DIR}/config.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    cfg = json.load(f)
print("  architectures     :", cfg.get("architectures"))
print("  model_type        :", cfg.get("model_type"))
q = cfg.get("quantization_config", {})
print("  quantization      :", q.get("quant_method") or q.get("format") or q)
PY
}

pull() {
  log "Pulling ${VLLM_IMAGE}"
  docker pull "$VLLM_IMAGE"
}

start() {
  model_ready || die "Model not found. Run: $0 download"
  command -v docker >/dev/null 2>&1 || die "Docker is required"

  if docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    die "Container ${CONTAINER_NAME} is already running"
  fi
  if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    docker rm "$CONTAINER_NAME" >/dev/null
  fi

  if ss -tln 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${API_PORT}$"; then
    die "Port ${API_PORT} is already in use"
  fi

  mkdir -p "$HF_HOME" "$VLLM_CACHE"

  local extra_args
  extra_args="--served-model-name ${SERVED_MODEL_NAME}"
  extra_args+=" --dtype auto"
  extra_args+=" --kv-cache-dtype fp8"
  extra_args+=" --attention-backend flashinfer"
  extra_args+=" --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS}"
  extra_args+=" --max-num-seqs ${MAX_NUM_SEQS}"
  extra_args+=" --enable-prefix-caching"
  extra_args+=" --enable-chunked-prefill"
  extra_args+=" --enable-auto-tool-choice"
  extra_args+=" --tool-call-parser qwen3_coder"
  [[ -z "$API_KEY" ]] || extra_args+=" --api-key ${API_KEY}"

  log "Starting ${SERVED_MODEL_NAME}"
  log "  Image       : ${VLLM_IMAGE}"
  log "  Context     : ${MAX_MODEL_LEN}"
  log "  GPU util    : ${GPU_MEMORY_UTIL}"
  log "  Tool parser : qwen3_coder"
  log "  Bind        : ${API_HOST}:${API_PORT}"
  log "  API key     : $([[ -n "$API_KEY" ]] && echo enabled || echo disabled)"

  docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    --gpus all \
    --ipc=host \
    --shm-size "$SHM_SIZE" \
    -p "${API_HOST}:${API_PORT}:8000" \
    -v "${MODEL_DIR}:/models/qwen3-coder-next:ro" \
    -v "${HF_HOME}:/root/.cache/huggingface" \
    -v "${VLLM_CACHE}:/root/.cache/vllm" \
    -e VLLM_NVFP4_GEMM_BACKEND=marlin \
    -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
    -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
    -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    -e MODEL=/models/qwen3-coder-next \
    -e PORT=8000 \
    -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    -e GPU_MEMORY_UTIL="$GPU_MEMORY_UTIL" \
    -e VLLM_EXTRA_ARGS="$extra_args" \
    "$VLLM_IMAGE" >/dev/null

  log "Waiting for /health; first startup may take several minutes"
  local deadline=$(( $(date +%s) + STARTUP_TIMEOUT ))
  while (( $(date +%s) < deadline )); do
    if ! docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
      docker logs --tail 200 "$CONTAINER_NAME" >&2 || true
      die "Container exited before becoming healthy"
    fi
    if curl -fsS "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
      local adv
      adv=$(detect_advertise_ip)
      echo ""
      echo "READY"
      echo "  Model : ${SERVED_MODEL_NAME}"
      echo "  API   : http://${adv}:${API_PORT}/v1"
      echo "  Ctx   : ${MAX_MODEL_LEN}"
      return
    fi
    sleep 5
  done

  docker logs --tail 200 "$CONTAINER_NAME" >&2 || true
  die "Server did not become healthy within ${STARTUP_TIMEOUT}s"
}

stop() {
  if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    log "Stopping ${CONTAINER_NAME}"
    docker rm -f "$CONTAINER_NAME" >/dev/null
    log "Stopped."
  else
    log "Container not found."
  fi
}

restart() { stop; start; }

auth_headers() {
  if [[ -n "$API_KEY" ]]; then
    printf '%s\n' "-H" "Authorization: Bearer ${API_KEY}"
  fi
}

status() {
  echo "=== Configuration ==="
  echo "Model    : ${HF_REPO}"
  echo "Alias    : ${SERVED_MODEL_NAME}"
  echo "Image    : ${VLLM_IMAGE}"
  echo "Context  : ${MAX_MODEL_LEN}"
  echo "GPU util : ${GPU_MEMORY_UTIL}"
  echo "Bind     : ${API_HOST}:${API_PORT}"
  echo "API key  : $([[ -n "$API_KEY" ]] && echo enabled || echo disabled)"
  echo ""

  docker ps -a --filter "name=^/${CONTAINER_NAME}$" \
    --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' || true
  echo ""

  echo "Health:"
  curl -fsS "http://127.0.0.1:${API_PORT}/health" 2>/dev/null ||
    echo "NOT READY"
  echo ""

  echo "Models:"
  local -a h=()
  [[ -z "$API_KEY" ]] || h=(-H "Authorization: Bearer ${API_KEY}")
  curl -fsS "${h[@]}" "http://127.0.0.1:${API_PORT}/v1/models" 2>/dev/null |
    python3 -m json.tool 2>/dev/null || echo "(unavailable)"
}

logs() { docker logs --tail "${1:-200}" "$CONTAINER_NAME"; }
follow_logs() { docker logs -f --tail 100 "$CONTAINER_NAME"; }

post_json() {
  local payload="$1"
  local -a h=(-H "Content-Type: application/json")
  [[ -z "$API_KEY" ]] || h+=(-H "Authorization: Bearer ${API_KEY}")
  curl -fsS "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
    "${h[@]}" -d "$payload" | python3 -m json.tool
}

test_text() {
  post_json "{
    \"model\":\"${SERVED_MODEL_NAME}\",
    \"messages\":[{\"role\":\"user\",\"content\":\"Reply exactly: DGX SPARK CODER READY\"}],
    \"max_tokens\":128,
    \"temperature\":0.0,
    \"chat_template_kwargs\":{\"enable_thinking\":false}
  }"
}

test_code() {
  post_json "{
    \"model\":\"${SERVED_MODEL_NAME}\",
    \"messages\":[{\"role\":\"user\",\"content\":\"Write a robust Python binary_search function with type hints and tests.\"}],
    \"max_tokens\":2048,
    \"temperature\":0.2,
    \"chat_template_kwargs\":{\"enable_thinking\":true}
  }"
}

test_tools() {
  post_json "{
    \"model\":\"${SERVED_MODEL_NAME}\",
    \"messages\":[{\"role\":\"user\",\"content\":\"Read README.md from the workspace.\"}],
    \"tools\":[{
      \"type\":\"function\",
      \"function\":{
        \"name\":\"read_file\",
        \"description\":\"Read a UTF-8 text file\",
        \"parameters\":{
          \"type\":\"object\",
          \"properties\":{\"path\":{\"type\":\"string\"}},
          \"required\":[\"path\"]
        }
      }
    }],
    \"tool_choice\":\"required\",
    \"max_tokens\":512,
    \"chat_template_kwargs\":{\"enable_thinking\":false}
  }"
}

network_info() {
  local adv
  adv=$(detect_advertise_ip)
  echo "Bind   : ${API_HOST}:${API_PORT}"
  echo "API    : http://${adv}:${API_PORT}/v1"
  echo "Health : http://${adv}:${API_PORT}/health"
}

cline_config() {
  local adv
  adv=$(detect_advertise_ip)
  cat <<CFG
Cline:
  Provider       : OpenAI Compatible
  Base URL       : http://${adv}:${API_PORT}/v1
  API Key        : ${API_KEY:-dummy}
  Model ID       : ${SERVED_MODEL_NAME}
  Context Window : ${MAX_MODEL_LEN}
  Max Tokens     : 16384
  Image Support  : Off

Native tool calling is enabled with qwen3_coder.
When the server has no API key, use any non-empty placeholder in Cline.
CFG
}

client_config() {
  local adv
  adv=$(detect_advertise_ip)
  cat <<PY
from openai import OpenAI

client = OpenAI(
    base_url="http://${adv}:${API_PORT}/v1",
    api_key="${API_KEY:-dummy}",
)

response = client.chat.completions.create(
    model="${SERVED_MODEL_NAME}",
    messages=[{"role": "user", "content": "Write a Python LRU cache."}],
    max_tokens=2048,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)

print(response.choices[0].message.content)
PY
}

usage() {
  cat <<HELP
Qwen3-Coder-Next-NVFP4-GB10 — DGX Spark vLLM controller

Usage:
  $0 [OPTIONS] COMMAND [args]

Options:
  --bind ADDR
  --port PORT
  --api-key KEY
  --context TOKENS
  --gpu-memory-util VALUE
  --max-batched-tokens N
  --max-seqs N
  --image IMAGE
  --model-dir PATH

Commands:
  preflight
  auth
  download [--force]
  verify-files
  pull
  start
  stop
  restart
  status
  logs [N]
  follow-logs
  test-text
  test-code
  test-tools
  network-info
  cline-config
  client-config
  help

First use:
  $0 preflight
  $0 auth
  $0 download
  $0 verify-files
  $0 pull
  $0 start
  $0 test-tools

Cline on another trusted LAN computer:
  $0 --bind 0.0.0.0 start
  $0 --bind 0.0.0.0 cline-config
HELP
}

parse_options "$@"
set -- "${REMAINING[@]:-}"

case "${1:-help}" in
  preflight) preflight ;;
  auth) auth ;;
  download) download_model "${2:-}" ;;
  verify-files) verify_files ;;
  pull) pull ;;
  start) start ;;
  stop) stop ;;
  restart) restart ;;
  status) status ;;
  logs) logs "${2:-200}" ;;
  follow-logs) follow_logs ;;
  test-text) test_text ;;
  test-code) test_code ;;
  test-tools) test_tools ;;
  network-info) network_info ;;
  cline-config) cline_config ;;
  client-config) client_config ;;
  help|-h|--help) usage ;;
  *) die "Unknown command: ${1}. Run: $0 help" ;;
esac
