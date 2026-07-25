#!/usr/bin/env bash
# Ornith-1.0-35B Uncensored Heretic NVFP4/FP8Dense — DGX Spark GB10
#
# Runtime:
#   vLLM 0.22.1 Docker + GB10 CUTLASS DSL / FlashInfer SM12x patches
#
# Model:
#   thanet-s/Ornith-1.0-35B-uncensored-heretic-nvfp4-fp8dense-gb10
#
# Features:
#   - Mixed precision: NVFP4 MoE experts, FP8 dense path, BF16 critical tensors
#   - Vision preserved
#   - Qwen3 XML tool calls and Qwen3 reasoning parser
#   - OpenAI-compatible API
#   - Docker restart policy for reboot persistence
#
# Security:
#   - No Hugging Face token is embedded.
#   - Bind defaults to localhost.
#   - Use --bind 0.0.0.0 only on a trusted LAN/VPN.
#
# Generated for NVIDIA DGX Spark / GB10 / SM121.

set -Eeuo pipefail

# ─── Model ─────────────────────────────────────────────────────────────────────
HF_REPO="${HF_REPO:-thanet-s/Ornith-1.0-35B-uncensored-heretic-nvfp4-fp8dense-gb10}"
MODEL_REVISION="${MODEL_REVISION:-main}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Ornith-1.0-35B-uncensored-heretic}"

# Reference repository that carries the two GB10 patch scripts.
PATCH_REPO="${PATCH_REPO:-demon-zombie/Qwen3.5-122B-A10B-NVFP4-FP8Dense-GB10}"
PATCH_REVISION="${PATCH_REVISION:-main}"

# Runtime version used by the model publisher's DGX Spark benchmark.
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.22.1}"
CONTAINER_NAME="${CONTAINER_NAME:-ornith-35b-heretic-nvfp4-fp8dense-gb10}"

# ─── Paths ─────────────────────────────────────────────────────────────────────
USER_HOME="${USER_HOME:-$HOME}"
MODEL_DIR="${MODEL_DIR:-${USER_HOME}/models/ornith-35b-heretic-nvfp4-fp8dense-gb10}"
HF_HOME="${HF_HOME:-${USER_HOME}/.cache/huggingface}"
VLLM_CACHE="${VLLM_CACHE:-${USER_HOME}/.cache/vllm}"
HF_VENV="${HF_VENV:-${USER_HOME}/.venvs/hf}"

RUNTIME_DIR="${RUNTIME_DIR:-${USER_HOME}/.local/share/dgx-vllm-gb10/v0.22.1}"
PATCH_SOURCE_DIR="${PATCH_SOURCE_DIR:-${RUNTIME_DIR}/patch-source}"
PATCHED_DIR="${PATCHED_DIR:-${RUNTIME_DIR}/patched}"

PATCHED_CUTLASS="${PATCHED_DIR}/tvm_ffi_provider.py"
PATCHED_FLASHINFER="${PATCHED_DIR}/moe_dynamic_kernel.py"
PATCH_MANIFEST="${PATCHED_DIR}/PATCH_MANIFEST.txt"

CUTLASS_CONTAINER_PATH="/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cutlass_dsl/tvm_ffi_provider.py"
FLASHINFER_CONTAINER_PATH="/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py"

# ─── Server defaults ───────────────────────────────────────────────────────────
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
API_KEY="${API_KEY:-}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.50}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4176}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
MOE_BACKEND="${MOE_BACKEND:-flashinfer_b12x}"
LOAD_FORMAT="${LOAD_FORMAT:-runai_streamer}"

RUNAI_STREAMER_MEMORY_LIMIT="${RUNAI_STREAMER_MEMORY_LIMIT:-4294967296}"
CUBLASLT_WORKSPACE_SIZE="${CUBLASLT_WORKSPACE_SIZE:-33554432}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"

SHM_SIZE="${SHM_SIZE:-32g}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
RESTART_POLICY="${RESTART_POLICY:-unless-stopped}"

HF_TOKEN="${HF_TOKEN:-}"

# ─── Helpers ───────────────────────────────────────────────────────────────────
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
  [[ -f "${MODEL_DIR}/model.safetensors.index.json" ]] &&
  [[ -f "${MODEL_DIR}/processor_config.json" ]] &&
  [[ -f "${MODEL_DIR}/chat_template.jinja" ]] &&
  [[ "$(find "$MODEL_DIR" -maxdepth 1 -type f -name 'model-*-of-00005.safetensors' | wc -l)" -eq 5 ]]
}

patches_ready() {
  [[ -s "$PATCHED_CUTLASS" ]] &&
  [[ -s "$PATCHED_FLASHINFER" ]] &&
  [[ -s "$PATCH_MANIFEST" ]]
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
    die "Could not create ${HF_VENV}. Install python3-venv first."

  "${HF_VENV}/bin/python" -m pip install -U pip
  "${HF_VENV}/bin/python" -m pip install -U "huggingface_hub[hf_xet]"
  printf '%s' "${HF_VENV}/bin/python"
}

validate_options() {
  [[ "$API_PORT" =~ ^[0-9]+$ ]] || die "Invalid port: ${API_PORT}"
  [[ "$MAX_MODEL_LEN" =~ ^[0-9]+$ ]] || die "Invalid context: ${MAX_MODEL_LEN}"
  [[ "$MAX_NUM_BATCHED_TOKENS" =~ ^[0-9]+$ ]] ||
    die "Invalid max batched tokens: ${MAX_NUM_BATCHED_TOKENS}"
  [[ "$MAX_NUM_SEQS" =~ ^[0-9]+$ ]] || die "Invalid max sequences: ${MAX_NUM_SEQS}"

  case "$KV_CACHE_DTYPE" in
    auto|fp8|fp8_e4m3|fp8_e5m2|bf16) ;;
    *) die "Unsupported --kv-cache-dtype: ${KV_CACHE_DTYPE}" ;;
  esac

  case "$RESTART_POLICY" in
    no|always|unless-stopped|on-failure) ;;
    *) die "Invalid --restart-policy: ${RESTART_POLICY}" ;;
  esac
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

      --kv-cache-dtype) KV_CACHE_DTYPE="$2"; shift 2 ;;
      --kv-cache-dtype=*) KV_CACHE_DTYPE="${1#*=}"; shift ;;

      --max-batched-tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
      --max-batched-tokens=*) MAX_NUM_BATCHED_TOKENS="${1#*=}"; shift ;;

      --max-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
      --max-seqs=*) MAX_NUM_SEQS="${1#*=}"; shift ;;

      --tool-parser) TOOL_CALL_PARSER="$2"; shift 2 ;;
      --tool-parser=*) TOOL_CALL_PARSER="${1#*=}"; shift ;;

      --reasoning-parser) REASONING_PARSER="$2"; shift 2 ;;
      --reasoning-parser=*) REASONING_PARSER="${1#*=}"; shift ;;

      --image) VLLM_IMAGE="$2"; shift 2 ;;
      --image=*) VLLM_IMAGE="${1#*=}"; shift ;;

      --model-dir) MODEL_DIR="$2"; shift 2 ;;
      --model-dir=*) MODEL_DIR="${1#*=}"; shift ;;

      --restart-policy) RESTART_POLICY="$2"; shift 2 ;;
      --restart-policy=*) RESTART_POLICY="${1#*=}"; shift ;;

      *) REMAINING+=("$1"); shift ;;
    esac
  done

  validate_options
}

require_docker() {
  command -v docker >/dev/null 2>&1 || die "Docker is not installed"
  docker info >/dev/null 2>&1 ||
    die "Docker daemon is unavailable or this user lacks permission"
}

# ─── Preflight ─────────────────────────────────────────────────────────────────
preflight() {
  require_docker

  echo "=== Ornith NVFP4/FP8Dense DGX Spark preflight ==="
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

  echo "Model           : ${HF_REPO}"
  echo "Model directory : ${MODEL_DIR}"
  echo "Runtime image   : ${VLLM_IMAGE}"
  echo "Patch source    : ${PATCH_REPO}"
  echo "Patches ready   : $([[ $(patches_ready; echo $?) -eq 0 ]] && echo yes || echo no)"
  echo "Context         : ${MAX_MODEL_LEN}"
  echo "GPU utilization : ${GPU_MEMORY_UTIL}"
  echo "KV cache        : ${KV_CACHE_DTYPE}"
  echo "Tool parser     : ${TOOL_CALL_PARSER}"
  echo "Reason parser   : ${REASONING_PARSER}"
  echo "Bind            : ${API_HOST}:${API_PORT}"
  echo "Restart policy  : ${RESTART_POLICY}"

  local free_bytes
  free_bytes=$(df -PB1 "$(dirname "$MODEL_DIR")" | awk 'NR==2 {print $4}')
  echo "Disk free       : $((free_bytes / 1000 / 1000 / 1000)) GB"

  if ((free_bytes < 30000000000)); then
    echo "WARNING: less than 30 GB free; repository is approximately 22.5 GB." >&2
  fi
}

# ─── Authentication ────────────────────────────────────────────────────────────
auth() {
  local py token
  py=$(ensure_hf_python)

  read -rsp "Enter a NEW Hugging Face read token: " token
  echo
  [[ -n "$token" ]] || die "No token entered"

  HF_HOME="$HF_HOME" "${HF_VENV}/bin/hf" auth login --token "$token"
  unset token

  echo "Token stored in the local Hugging Face cache."
}

# ─── Download ──────────────────────────────────────────────────────────────────
download_model() {
  local mode="${1:-}"

  if [[ "$mode" != "--force" ]] && model_ready; then
    log "Model already exists in ${MODEL_DIR}; skipping download."
    echo "Use '$0 download --force' to refresh the snapshot."
    return
  fi

  local py
  py=$(ensure_hf_python)

  mkdir -p "$MODEL_DIR" "$HF_HOME"
  export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

  log "Downloading ${HF_REPO} @ ${MODEL_REVISION}"
  log "Destination: ${MODEL_DIR}"

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

  model_ready || die "Download completed but the expected 5-shard snapshot is incomplete"
  log "Download complete."
}

# ─── File validation ───────────────────────────────────────────────────────────
verify_files() {
  [[ -d "$MODEL_DIR" ]] || die "Model directory not found: ${MODEL_DIR}"

  local required
  for required in \
    config.json \
    generation_config.json \
    chat_template.jinja \
    model.safetensors.index.json \
    processor_config.json \
    tokenizer.json \
    tokenizer_config.json; do
    [[ -s "${MODEL_DIR}/${required}" ]] || die "Missing or empty: ${required}"
  done

  local count total
  count=$(find "$MODEL_DIR" -maxdepth 1 -type f \
    -name 'model-*-of-00005.safetensors' | wc -l)
  total=$(
    find "$MODEL_DIR" -maxdepth 1 -type f \
      -name 'model-*-of-00005.safetensors' \
      -printf '%s\n' |
      awk '{sum += $1} END {printf "%.0f", sum + 0}'
  )

  ((count == 5)) || die "Expected 5 safetensors shards; found ${count}"
  ((total >= 22000000000)) ||
    die "Safetensors total is unexpectedly small: ${total} bytes"

  MODEL_DIR="$MODEL_DIR" python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["MODEL_DIR"])
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
index = json.loads(
    (root / "model.safetensors.index.json").read_text(encoding="utf-8")
)

quant = config.get("quantization_config", {})
groups = quant.get("config_groups", {})

print("verify-files: PASS")
print("  architectures       :", config.get("architectures"))
print("  model_type          :", config.get("model_type"))
print("  max positions       :", config.get("max_position_embeddings"))
print("  quant method        :", quant.get("quant_method"))
print("  quant format        :", quant.get("format"))
print("  config groups       :", sorted(groups) if isinstance(groups, dict) else groups)
print("  indexed tensors     :", len(index.get("weight_map", {})))
print("  safetensors shards  : 5")
PY

  echo "  safetensors bytes   : ${total}"
}

# ─── GB10 runtime patches ──────────────────────────────────────────────────────
pull() {
  require_docker
  log "Pulling ${VLLM_IMAGE}"
  docker pull "$VLLM_IMAGE"
}

download_patch_source() {
  local py
  py=$(ensure_hf_python)

  mkdir -p "$PATCH_SOURCE_DIR"

  log "Downloading GB10 patch scripts from ${PATCH_REPO}"
  HF_HOME="$HF_HOME" \
  HF_TOKEN="$HF_TOKEN" \
  PATCH_SOURCE_DIR="$PATCH_SOURCE_DIR" \
  PATCH_REPO="$PATCH_REPO" \
  PATCH_REVISION="$PATCH_REVISION" \
  "$py" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["PATCH_REPO"],
    revision=os.environ["PATCH_REVISION"],
    local_dir=os.environ["PATCH_SOURCE_DIR"],
    allow_patterns=["gb10-patches/*"],
    token=os.environ.get("HF_TOKEN") or None,
)
PY
}

prepare_runtime() {
  require_docker
  pull
  download_patch_source

  local cutlass_patch
  local flash_patch
  cutlass_patch="$(
    find "$PATCH_SOURCE_DIR" -type f \
      -name 'patch_cutlass_dsl_global_dtors.py' -print -quit
  )"
  flash_patch="$(
    find "$PATCH_SOURCE_DIR" -type f \
      -name 'patch_b12x_gated_smem.py' -print -quit
  )"

  [[ -s "$cutlass_patch" ]] || die "CUTLASS patch script was not found"
  [[ -s "$flash_patch" ]] || die "FlashInfer patch script was not found"

  mkdir -p "$PATCHED_DIR"
  rm -f "$PATCHED_CUTLASS" "$PATCHED_FLASHINFER" "$PATCH_MANIFEST"

  local temp_container
  temp_container="ornith-gb10-patch-$RANDOM-$RANDOM"

  cleanup_temp() {
    docker rm -f "$temp_container" >/dev/null 2>&1 || true
  }
  trap cleanup_temp RETURN

  docker create --name "$temp_container" "$VLLM_IMAGE" >/dev/null

  docker cp \
    "${temp_container}:${CUTLASS_CONTAINER_PATH}" \
    "$PATCHED_CUTLASS"

  docker cp \
    "${temp_container}:${FLASHINFER_CONTAINER_PATH}" \
    "$PATCHED_FLASHINFER"

  local cutlass_before flash_before cutlass_after flash_after
  cutlass_before=$(sha256sum "$PATCHED_CUTLASS" | awk '{print $1}')
  flash_before=$(sha256sum "$PATCHED_FLASHINFER" | awk '{print $1}')

  python3 "$cutlass_patch" "$PATCHED_CUTLASS"
  python3 "$flash_patch" "$PATCHED_FLASHINFER"

  cutlass_after=$(sha256sum "$PATCHED_CUTLASS" | awk '{print $1}')
  flash_after=$(sha256sum "$PATCHED_FLASHINFER" | awk '{print $1}')

  [[ "$cutlass_before" != "$cutlass_after" ]] ||
    die "CUTLASS patch did not modify the target file"

  [[ "$flash_before" != "$flash_after" ]] ||
    die "FlashInfer patch did not modify the target file"

  cat >"$PATCH_MANIFEST" <<EOF
Created: $(date --iso-8601=seconds)
Runtime image: ${VLLM_IMAGE}
Patch repository: ${PATCH_REPO}
Patch revision: ${PATCH_REVISION}

CUTLASS target:
  container path: ${CUTLASS_CONTAINER_PATH}
  before: ${cutlass_before}
  after:  ${cutlass_after}

FlashInfer target:
  container path: ${FLASHINFER_CONTAINER_PATH}
  before: ${flash_before}
  after:  ${flash_after}
EOF

  cleanup_temp
  trap - RETURN

  log "GB10 runtime patches prepared."
  echo "Manifest: ${PATCH_MANIFEST}"
}

runtime_info() {
  echo "=== Runtime information ==="
  echo "Image            : ${VLLM_IMAGE}"
  echo "Container        : ${CONTAINER_NAME}"
  echo "Patch repository : ${PATCH_REPO}"
  echo "Patched CUTLASS  : ${PATCHED_CUTLASS}"
  echo "Patched FlashInf.: ${PATCHED_FLASHINFER}"
  echo "Patches ready    : $([[ $(patches_ready; echo $?) -eq 0 ]] && echo yes || echo no)"
  if [[ -f "$PATCH_MANIFEST" ]]; then
    echo ""
    cat "$PATCH_MANIFEST"
  fi
}

# ─── Start / stop ──────────────────────────────────────────────────────────────
start() {
  require_docker
  model_ready || die "Model not found. Run: $0 download"
  patches_ready || die "GB10 patches are not prepared. Run: $0 prepare-runtime"

  if docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    die "Container ${CONTAINER_NAME} is already running"
  fi

  if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    docker rm "$CONTAINER_NAME" >/dev/null
  fi

  if ss -tln 2>/dev/null | awk '{print $4}' |
     grep -Eq "[:.]${API_PORT}$"; then
    die "Port ${API_PORT} is already in use"
  fi

  mkdir -p "$HF_HOME" "$VLLM_CACHE"

  local -a args=(
    --model /models/ornith
    --served-model-name "$SERVED_MODEL_NAME"
    --kernel-config "{\"moe_backend\":\"${MOE_BACKEND}\"}"
    --load-format "$LOAD_FORMAT"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTIL"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --enable-prefix-caching
    --enable-chunked-prefill
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --max-num-seqs "$MAX_NUM_SEQS"
    --enable-auto-tool-choice
    --tool-call-parser "$TOOL_CALL_PARSER"
    --reasoning-parser "$REASONING_PARSER"
    --host 0.0.0.0
    --port 8000
  )

  [[ -z "$API_KEY" ]] || args+=(--api-key "$API_KEY")

  log "Starting ${SERVED_MODEL_NAME}"
  log "  Model       : ${MODEL_DIR}"
  log "  Image       : ${VLLM_IMAGE}"
  log "  Context     : ${MAX_MODEL_LEN}"
  log "  GPU util    : ${GPU_MEMORY_UTIL}"
  log "  KV cache    : ${KV_CACHE_DTYPE}"
  log "  MoE backend : ${MOE_BACKEND}"
  log "  Tool parser : ${TOOL_CALL_PARSER}"
  log "  Reasoning   : ${REASONING_PARSER}"
  log "  Bind        : ${API_HOST}:${API_PORT}"
  log "  API key     : $([[ -n "$API_KEY" ]] && echo enabled || echo disabled)"
  log "  Restart     : ${RESTART_POLICY}"

  docker run -d \
    --name "$CONTAINER_NAME" \
    --restart "$RESTART_POLICY" \
    --gpus all \
    --ipc=host \
    --shm-size "$SHM_SIZE" \
    -p "${API_HOST}:${API_PORT}:8000" \
    -v "${MODEL_DIR}:/models/ornith:ro" \
    -v "${HF_HOME}:/root/.cache/huggingface" \
    -v "${VLLM_CACHE}:/root/.cache/vllm" \
    -v "${PATCHED_CUTLASS}:${CUTLASS_CONTAINER_PATH}:ro" \
    -v "${PATCHED_FLASHINFER}:${FLASHINFER_CONTAINER_PATH}:ro" \
    -e CUBLASLT_WORKSPACE_SIZE="$CUBLASLT_WORKSPACE_SIZE" \
    -e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    -e RUNAI_STREAMER_MEMORY_LIMIT="$RUNAI_STREAMER_MEMORY_LIMIT" \
    "$VLLM_IMAGE" \
    "${args[@]}" >/dev/null

  log "Waiting for /health; first startup and kernel compilation may take several minutes."

  local deadline
  deadline=$(( $(date +%s) + STARTUP_TIMEOUT ))

  while (( $(date +%s) < deadline )); do
    if ! docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
      docker logs --tail 250 "$CONTAINER_NAME" >&2 || true
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
      echo "  Tools : ${TOOL_CALL_PARSER}"
      echo "  Think : ${REASONING_PARSER}"
      echo "  Vision: enabled"
      return
    fi

    sleep 5
  done

  docker logs --tail 250 "$CONTAINER_NAME" >&2 || true
  die "Server did not become healthy within ${STARTUP_TIMEOUT} seconds"
}

stop() {
  require_docker

  if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    log "Stopping ${CONTAINER_NAME}"
    docker rm -f "$CONTAINER_NAME" >/dev/null
    log "Stopped."
  else
    log "Container not found."
  fi
}

restart() {
  stop
  start
}

# ─── Status / logs ─────────────────────────────────────────────────────────────
status() {
  require_docker

  echo "=== Configuration ==="
  echo "Model       : ${HF_REPO}"
  echo "Alias       : ${SERVED_MODEL_NAME}"
  echo "Image       : ${VLLM_IMAGE}"
  echo "Context     : ${MAX_MODEL_LEN}"
  echo "GPU util    : ${GPU_MEMORY_UTIL}"
  echo "KV cache    : ${KV_CACHE_DTYPE}"
  echo "MoE backend : ${MOE_BACKEND}"
  echo "Tool parser : ${TOOL_CALL_PARSER}"
  echo "Reasoning   : ${REASONING_PARSER}"
  echo "Bind        : ${API_HOST}:${API_PORT}"
  echo "API key     : $([[ -n "$API_KEY" ]] && echo enabled || echo disabled)"
  echo "Restart     : ${RESTART_POLICY}"
  echo ""

  echo "=== Container ==="
  docker ps -a --filter "name=^/${CONTAINER_NAME}$" \
    --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' || true
  echo ""

  echo "=== Restart policy ==="
  docker inspect \
    -f '{{.HostConfig.RestartPolicy.Name}}' \
    "$CONTAINER_NAME" 2>/dev/null || echo "(container unavailable)"
  echo ""

  echo "=== Health ==="
  curl -fsS "http://127.0.0.1:${API_PORT}/health" 2>/dev/null ||
    echo "NOT READY"
  echo ""

  echo "=== Models ==="
  local -a headers=()
  [[ -z "$API_KEY" ]] ||
    headers=(-H "Authorization: Bearer ${API_KEY}")

  curl -fsS "${headers[@]}" \
    "http://127.0.0.1:${API_PORT}/v1/models" 2>/dev/null |
    python3 -m json.tool 2>/dev/null || echo "(unavailable)"
}

logs() {
  require_docker
  docker logs --tail "${1:-200}" "$CONTAINER_NAME"
}

follow_logs() {
  require_docker
  docker logs -f --tail 100 "$CONTAINER_NAME"
}

# ─── API tests ─────────────────────────────────────────────────────────────────
post_json() {
  local payload="$1"
  local -a headers=(-H "Content-Type: application/json")

  [[ -z "$API_KEY" ]] ||
    headers+=(-H "Authorization: Bearer ${API_KEY}")

  curl -fsS \
    "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
    "${headers[@]}" \
    -d "$payload" |
    python3 -m json.tool
}

test_text() {
  post_json "{
    \"model\":\"${SERVED_MODEL_NAME}\",
    \"messages\":[
      {
        \"role\":\"user\",
        \"content\":\"Reply exactly: ORNITH DGX SPARK READY\"
      }
    ],
    \"max_tokens\":128,
    \"temperature\":0.0
  }"
}

test_reasoning() {
  post_json "{
    \"model\":\"${SERVED_MODEL_NAME}\",
    \"messages\":[
      {
        \"role\":\"user\",
        \"content\":\"A repository has 37 modules and each needs 43 tests. Calculate the total and explain briefly.\"
      }
    ],
    \"max_tokens\":1024,
    \"temperature\":0.2
  }"
}

test_tools() {
  post_json "{
    \"model\":\"${SERVED_MODEL_NAME}\",
    \"messages\":[
      {
        \"role\":\"user\",
        \"content\":\"Read README.md from the current workspace.\"
      }
    ],
    \"tools\":[
      {
        \"type\":\"function\",
        \"function\":{
          \"name\":\"read_file\",
          \"description\":\"Read a UTF-8 text file from the workspace\",
          \"parameters\":{
            \"type\":\"object\",
            \"properties\":{
              \"path\":{\"type\":\"string\"}
            },
            \"required\":[\"path\"]
          }
        }
      }
    ],
    \"tool_choice\":\"required\",
    \"max_tokens\":512,
    \"temperature\":0.0
  }"
}

test_image() {
  local -a headers=(-H "Content-Type: application/json")
  [[ -z "$API_KEY" ]] ||
    headers+=(-H "Authorization: Bearer ${API_KEY}")

  python3 - "$SERVED_MODEL_NAME" <<'PY' |
import base64
import json
import sys

# 1x1 PNG is sufficient to validate the request path, not visual quality.
png = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAAC0lEQVQI12NgAAIABQAABjE+ibYAAAAASUVORK5CYII="
)

print(json.dumps({
    "model": sys.argv[1],
    "messages": [{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64," + png
                }
            },
            {
                "type": "text",
                "text": "Describe the dominant color briefly."
            }
        ]
    }],
    "max_tokens": 256,
    "temperature": 0.0
}))
PY
  curl -fsS \
    "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
    "${headers[@]}" \
    --data-binary @- |
    python3 -m json.tool
}

bench() {
  local -a headers=(-H "Content-Type: application/json")
  [[ -z "$API_KEY" ]] ||
    headers+=(-H "Authorization: Bearer ${API_KEY}")

  python3 - "$API_PORT" "$SERVED_MODEL_NAME" "$API_KEY" <<'PY'
import json
import sys
import time
import urllib.request

port, model, api_key = sys.argv[1:]
url = f"http://127.0.0.1:{port}/v1/chat/completions"

payload = json.dumps({
    "model": model,
    "messages": [{
        "role": "user",
        "content": "Write a technical explanation of prefix caching in about 800 words."
    }],
    "max_tokens": 1024,
    "temperature": 0.2,
    "stream": False,
}).encode()

headers = {"Content-Type": "application/json"}
if api_key:
    headers["Authorization"] = f"Bearer {api_key}"

request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
started = time.time()
with urllib.request.urlopen(request, timeout=900) as response:
    result = json.loads(response.read())
elapsed = time.time() - started

usage = result.get("usage", {})
tokens = usage.get("completion_tokens", 0)

print(f"Completion tokens : {tokens}")
print(f"Elapsed           : {elapsed:.3f} s")
print(f"Throughput        : {tokens / elapsed:.2f} tok/s" if elapsed else "n/a")
PY
}

# ─── Client configuration ──────────────────────────────────────────────────────
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
Cline configuration:
  Provider       : OpenAI Compatible
  Base URL       : http://${adv}:${API_PORT}/v1
  API Key        : ${API_KEY:-dummy}
  Model ID       : ${SERVED_MODEL_NAME}
  Context Window : ${MAX_MODEL_LEN}
  Max Tokens     : 16384
  Image Support  : On

Runtime behavior:
  Tool parser     : ${TOOL_CALL_PARSER}
  Reasoning parser: ${REASONING_PARSER}

When the server was started without --api-key, put any non-empty placeholder
such as "dummy" in Cline's API Key field.
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
    messages=[
        {
            "role": "user",
            "content": "Create a robust Python LRU cache with tests."
        }
    ],
    max_tokens=4096,
)

message = response.choices[0].message
reasoning = getattr(message, "reasoning_content", None)

if reasoning:
    print("Reasoning:")
    print(reasoning)

print("Answer:")
print(message.content)
PY
}

# ─── Help ──────────────────────────────────────────────────────────────────────
usage() {
  cat <<HELP
Ornith-1.0-35B Uncensored Heretic NVFP4/FP8Dense — DGX Spark

Usage:
  $0 [OPTIONS] COMMAND [args]

Options:
  --bind ADDR
      Bind address. Default: ${API_HOST}

  --port PORT
      OpenAI-compatible API port. Default: ${API_PORT}

  --api-key KEY
      Optional API key. Disabled by default.

  --context TOKENS
      Maximum model length. Default: ${MAX_MODEL_LEN}

  --gpu-memory-util VALUE
      vLLM memory utilization. Default: ${GPU_MEMORY_UTIL}

  --kv-cache-dtype TYPE
      auto, fp8, fp8_e4m3, fp8_e5m2, or bf16.
      Default: ${KV_CACHE_DTYPE}

  --max-batched-tokens N
      Default: ${MAX_NUM_BATCHED_TOKENS}

  --max-seqs N
      Default: ${MAX_NUM_SEQS}

  --tool-parser NAME
      Default: ${TOOL_CALL_PARSER}

  --reasoning-parser NAME
      Default: ${REASONING_PARSER}

  --image IMAGE
      Runtime image. Default: ${VLLM_IMAGE}

  --model-dir PATH
      Local model directory.

  --restart-policy POLICY
      no, always, unless-stopped, or on-failure.
      Default: ${RESTART_POLICY}

Commands:
  preflight
      Check Docker, NVIDIA runtime, GPU, disk, and current settings.

  auth
      Store a Hugging Face token securely in the local cache.

  download [--force]
      Download the 5-shard model snapshot.

  verify-files
      Validate required files, shard count, size, and quantization metadata.

  pull
      Pull the pinned vLLM image.

  prepare-runtime
      Download and apply both GB10 runtime patches.

  runtime-info
      Show image and patch manifest.

  start
      Start the vLLM OpenAI-compatible server.

  stop
      Stop and remove the container.

  restart
      Stop and start the server.

  status
      Show container, restart policy, health, and model endpoint.

  logs [N]
      Show recent container logs.

  follow-logs
      Follow logs continuously.

  test-text
  test-reasoning
  test-tools
  test-image
  bench
  network-info
  cline-config
  client-config
  help

First installation:
  $0 preflight
  $0 download
  $0 verify-files
  $0 prepare-runtime
  $0 start
  $0 test-text
  $0 test-tools
  $0 bench

Cline from another trusted LAN computer:
  $0 --bind 0.0.0.0 start
  $0 --bind 0.0.0.0 cline-config

Autostart:
  Docker restart policy is '${RESTART_POLICY}' by default.
  After a successful start, Docker will restart this container after reboot.
  Ensure Docker itself is enabled:

    sudo systemctl enable --now docker
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
  prepare-runtime) prepare_runtime ;;
  runtime-info) runtime_info ;;
  start) start ;;
  stop) stop ;;
  restart) restart ;;
  status) status ;;
  logs) logs "${2:-200}" ;;
  follow-logs) follow_logs ;;
  test-text) test_text ;;
  test-reasoning) test_reasoning ;;
  test-tools) test_tools ;;
  test-image) test_image ;;
  bench) bench ;;
  network-info) network_info ;;
  cline-config) cline_config ;;
  client-config) client_config ;;
  help|-h|--help) usage ;;
  *) die "Unknown command: ${1}. Run: $0 help" ;;
esac
