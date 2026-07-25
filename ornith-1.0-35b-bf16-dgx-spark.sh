#!/usr/bin/env bash
# Ornith-1.0-35B BF16 — single DGX Spark controller (llama.cpp)
# Runtime: llama.cpp built from source, SM121a-real (GB10 Blackwell)
# Model: 2 BF16 GGUF shards; pass shard 1 and llama.cpp auto-loads shard 2
# Vision: mmproj-BF16.gguf is required for image/video-frame input
# Reasoning + agentic coding: embedded Jinja template, <think> parser, and tool calls
set -Eeuo pipefail

SCRIPT_VERSION="${SCRIPT_VERSION:-3.1.0}"
MODEL_LABEL="${MODEL_LABEL:-Ornith-1.0-35B (BF16 GGUF)}"
RUNTIME_LABEL="${RUNTIME_LABEL:-llama.cpp}"
MODEL_FEATURES="${MODEL_FEATURES:-vision · reasoning · tools}"

# ─── Model ────────────────────────────────────────────────────────────────────
HF_REPO="${HF_REPO:-unsloth/Ornith-1.0-35B-GGUF}"
MODEL_REVISION="${MODEL_REVISION:-main}"

MODEL_QUANT="BF16"
MODEL_SHARD_1_REL="${MODEL_SHARD_1_REL:-BF16/Ornith-1.0-35B-BF16-00001-of-00002.gguf}"
MODEL_SHARD_2_REL="${MODEL_SHARD_2_REL:-BF16/Ornith-1.0-35B-BF16-00002-of-00002.gguf}"
MMPROJ_FILE="${MMPROJ_FILE:-mmproj-BF16.gguf}"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-ornith-1.0-35b-bf16}"

# ─── Runtime ──────────────────────────────────────────────────────────────────
LLAMA_CPP_REF="${LLAMA_CPP_REF:-b10054}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-121a-real}"

# ─── Paths ────────────────────────────────────────────────────────────────────
# Portable defaults; override by environment when needed.
# Works for any Linux user, e.g. /home/admin, /home/dgx, /root, etc.
USER_HOME="${USER_HOME:-$HOME}"
HF_HOME="${HF_HOME:-${USER_HOME}/.cache/huggingface}"
HF_VENV="${HF_VENV:-${USER_HOME}/.venvs/hf}"
AUTO_INSTALL_HF="${AUTO_INSTALL_HF:-1}"
HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
MIN_FREE_DOWNLOAD_BYTES="${MIN_FREE_DOWNLOAD_BYTES:-80000000000}"
MODEL_DIR="${MODEL_DIR:-${USER_HOME}/models/ornith-1.0-35b-bf16}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${USER_HOME}/src/llama.cpp}"

# ─── Serving ──────────────────────────────────────────────────────────────────
API_PORT="${API_PORT:-8000}"
API_HOST="${API_HOST:-127.0.0.1}"
ADVERTISE_IP="${ADVERTISE_IP:-}"
ADVERTISE_INTERFACE="${ADVERTISE_INTERFACE:-}"
ROUTE_PROBE_IP="${ROUTE_PROBE_IP:-1.1.1.1}"

CTX_SIZE="${CTX_SIZE:-131072}"
BATCH_SIZE="${BATCH_SIZE:-512}"
UBATCH_SIZE="${UBATCH_SIZE:-256}"
N_THREADS="${N_THREADS:-20}"
PARALLEL_SLOTS="${PARALLEL_SLOTS:-1}"
FLASH_ATTN="${FLASH_ATTN:-on}"
SPLIT_MODE="${SPLIT_MODE:-none}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
REASONING_FORMAT="${REASONING_FORMAT:-deepseek}"
VISION_MODE="${VISION_MODE:-on}"
API_KEY="${API_KEY:-}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"

# ─── Exact file metadata for verification ────────────────────────────────────
SHARD_1_SIZE_BYTES="49781577216"
SHARD_2_SIZE_BYTES="19595060416"
MMPROJ_SIZE_BYTES="902822624"

SHARD_1_SHA256="${SHARD_1_SHA256:-6d4f8b08fbee72eff293b2741d9b1460070e1ec4597d859b0d9999fca388265e}"
SHARD_2_SHA256="${SHARD_2_SHA256:-390fdcd2b4bc18d0ce8c430fc8612654f1f0ca36e3898a03d691ed0bd5b0aa86}"
MMPROJ_SHA256="${MMPROJ_SHA256:-8c10da8c1e4b860fa09d1539d40b7ae22b194706046d9d25d390e0d10c680748}"

# ─── Server PID file ──────────────────────────────────────────────────────────
PID_FILE="/tmp/llama-ornith-1.0-35b-bf16.pid"
LOG_FILE="${LOG_FILE:-${MODEL_DIR}/llama-server.log}"

# ─── Helpers ──────────────────────────────────────────────────────────────────
die()  { echo "ERROR: $*" >&2; exit 1; }
log()  { echo "[$(date '+%H:%M:%S')] $*"; }


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
  if [[ -n "$src" ]]; then printf '%s' "$src"; return; fi
  ip -4 addr show scope global \
    | awk '/inet /{split($2,a,"/"); print a[1]}' \
    | grep -v '^127\.' | head -1
}

parse_options() {
  REMAINING_ARGS=()
  while (( $# )); do
    case "$1" in
      --context)        CTX_SIZE="$2";            shift 2 ;;
      --context=*)      CTX_SIZE="${1#*=}";        shift ;;
      --port)           API_PORT="$2";             shift 2 ;;
      --port=*)         API_PORT="${1#*=}";        shift ;;
      --bind)           API_HOST="$2";             shift 2 ;;
      --bind=*)         API_HOST="${1#*=}";        shift ;;
      --advertise-ip)   ADVERTISE_IP="$2";         shift 2 ;;
      --advertise-ip=*) ADVERTISE_IP="${1#*=}";    shift ;;
      --interface)      ADVERTISE_INTERFACE="$2";  shift 2 ;;
      --interface=*)    ADVERTISE_INTERFACE="${1#*=}"; shift ;;
      --parallel)       PARALLEL_SLOTS="$2";       shift 2 ;;
      --parallel=*)     PARALLEL_SLOTS="${1#*=}";  shift ;;
      --flash-attn)     FLASH_ATTN="$2";          shift 2 ;;
      --flash-attn=*)   FLASH_ATTN="${1#*=}";      shift ;;
      --split-mode)     SPLIT_MODE="$2";           shift 2 ;;
      --split-mode=*)   SPLIT_MODE="${1#*=}";      shift ;;
      --cache-k)        CACHE_TYPE_K="$2";          shift 2 ;;
      --cache-k=*)      CACHE_TYPE_K="${1#*=}";     shift ;;
      --cache-v)        CACHE_TYPE_V="$2";          shift 2 ;;
      --cache-v=*)      CACHE_TYPE_V="${1#*=}";     shift ;;
      --reasoning-format) REASONING_FORMAT="$2";    shift 2 ;;
      --reasoning-format=*) REASONING_FORMAT="${1#*=}"; shift ;;
      --vision)         VISION_MODE="$2";           shift 2 ;;
      --vision=*)       VISION_MODE="${1#*=}";      shift ;;
      --api-key)        API_KEY="$2";               shift 2 ;;
      --api-key=*)      API_KEY="${1#*=}";          shift ;;
      *)                REMAINING_ARGS+=("$1");     shift ;;
    esac
  done
  case "$FLASH_ATTN" in on|off|auto) ;; *) die "Invalid --flash-attn value: ${FLASH_ATTN} (use on, off, or auto)" ;; esac
  case "$SPLIT_MODE" in none|layer|row|tensor) ;; *) die "Invalid --split-mode value: ${SPLIT_MODE} (use none, layer, row, or tensor)" ;; esac
  case "$CACHE_TYPE_K" in f32|f16|bf16|q8_0|q4_0|q4_1|iq4_nl|q5_0|q5_1) ;; *) die "Invalid --cache-k value: ${CACHE_TYPE_K}" ;; esac
  case "$CACHE_TYPE_V" in f32|f16|bf16|q8_0|q4_0|q4_1|iq4_nl|q5_0|q5_1) ;; *) die "Invalid --cache-v value: ${CACHE_TYPE_V}" ;; esac
  case "$REASONING_FORMAT" in deepseek|none) ;; *) die "Invalid --reasoning-format: ${REASONING_FORMAT} (use deepseek or none)" ;; esac
  case "$VISION_MODE" in on|off) ;; *) die "Invalid --vision: ${VISION_MODE} (use on or off)" ;; esac
  [[ "$CTX_SIZE" =~ ^[0-9]+$ ]] && (( CTX_SIZE >= 4096 && CTX_SIZE <= 1048576 )) || die "Invalid --context: ${CTX_SIZE} (use 4096..1048576)"
  [[ "$API_PORT" =~ ^[0-9]+$ ]] && (( API_PORT >= 1 && API_PORT <= 65535 )) || die "Invalid --port: ${API_PORT} (use 1..65535)"
  export CTX_SIZE API_HOST API_PORT ADVERTISE_IP ADVERTISE_INTERFACE PARALLEL_SLOTS FLASH_ATTN SPLIT_MODE CACHE_TYPE_K CACHE_TYPE_V REASONING_FORMAT VISION_MODE API_KEY STARTUP_TIMEOUT
}

# ─── prepare-runtime ──────────────────────────────────────────────────────────
prepare_runtime() {
  log "Building llama.cpp @ ${LLAMA_CPP_REF} with SM${CUDA_ARCHITECTURES} CUDA support …"
  mkdir -p "$(dirname "$LLAMA_CPP_DIR")"

  if [[ -d "$LLAMA_CPP_DIR" ]]; then
    log "Updating existing checkout …"
    git -C "$LLAMA_CPP_DIR" fetch origin
  else
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
  fi

  git -C "$LLAMA_CPP_DIR" checkout "$LLAMA_CPP_REF"
  local actual_ref; actual_ref=$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)
  log "Pinned to: ${actual_ref}"
  echo "$actual_ref" > "${LLAMA_CPP_DIR}/.build-ref"

  cmake -B "${LLAMA_CPP_DIR}/build" \
    -S "$LLAMA_CPP_DIR" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DLLAMA_OPENSSL=ON \
    -DCMAKE_BUILD_TYPE=Release

  cmake --build "${LLAMA_CPP_DIR}/build" --config Release -j"$(nproc)"
  log "Build complete. Binary: ${LLAMA_CPP_DIR}/build/bin/llama-server"
}

runtime_info() {
  echo "=== Runtime info ==="
  echo "llama.cpp dir : ${LLAMA_CPP_DIR}"
  local ref=""
  [[ -f "${LLAMA_CPP_DIR}/.build-ref" ]] && ref=$(cat "${LLAMA_CPP_DIR}/.build-ref")
  echo "Build ref     : ${ref:-unknown (run prepare-runtime)}"
  echo "Target ref    : ${LLAMA_CPP_REF}"
  echo "CUDA arch     : ${CUDA_ARCHITECTURES}"
  local srv="${LLAMA_CPP_DIR}/build/bin/llama-server"
  if [[ -x "$srv" ]]; then
    echo "Binary        : ${srv}"
    "$srv" --version 2>/dev/null | head -1 || true
  else
    echo "Binary        : NOT BUILT (run prepare-runtime)"
  fi
}

# ─── download ─────────────────────────────────────────────────────────────────
download() {
  local subcmd="${1:-}"
  local force_download=0

  case "$subcmd" in
    --list)
      cat <<LIST
Files downloaded from ${HF_REPO}:
  ${MODEL_SHARD_1_REL}  49,781,577,216 bytes
  ${MODEL_SHARD_2_REL}  19,595,060,416 bytes
  ${MMPROJ_FILE}          902,822,624 bytes

Total download: 70,279,460,256 bytes (~70.28 GB decimal)
Recommended free disk space: at least 82 GB
LIST
      return
      ;;
    --force) force_download=1 ;;
    "") ;;
    *) die "Unknown download option '${subcmd}'. Use: download, download --force, or download --list" ;;
  esac

  mkdir -p "$MODEL_DIR" "$HF_HOME"

  local shard1_path="${MODEL_DIR}/${MODEL_SHARD_1_REL}"
  local shard2_path="${MODEL_DIR}/${MODEL_SHARD_2_REL}"
  local mmproj_path="${MODEL_DIR}/${MMPROJ_FILE}"

  gguf_file_ok() {
    local path="$1" min_bytes="$2"
    [[ -f "$path" ]] || return 1
    local size magic
    size=$(stat -Lc '%s' "$path" 2>/dev/null || stat -f '%z' "$path" 2>/dev/null) || return 1
    (( size == min_bytes )) || return 1
    magic=$(LC_ALL=C head -c 4 "$path" | od -A n -t x1 | tr -d ' \n')
    [[ "$magic" == "47475546" ]]
  }

  if (( force_download == 0 )) \
      && gguf_file_ok "$shard1_path" "$SHARD_1_SIZE_BYTES" \
      && gguf_file_ok "$shard2_path" "$SHARD_2_SIZE_BYTES" \
      && gguf_file_ok "$mmproj_path" "$MMPROJ_SIZE_BYTES"; then
    log "Both BF16 shards and mmproj already exist and pass size/header checks."
    log "Nothing to download. Use 'download --force' only for a fresh copy."
    return
  fi

  local existing=0 required free_bytes p
  for p in "$shard1_path" "$shard2_path" "$mmproj_path"; do
    [[ -f "$p" ]] && existing=$(( existing + $(stat -Lc '%s' "$p" 2>/dev/null || echo 0) ))
  done
  required=$(( MIN_FREE_DOWNLOAD_BYTES - existing ))
  (( required < 5000000000 )) && required=5000000000
  free_bytes=$(df -PB1 "$MODEL_DIR" | awk 'NR==2 {print $4}')
  if [[ "$free_bytes" =~ ^[0-9]+$ ]] && (( free_bytes < required )); then
    die "Insufficient disk space: ${free_bytes} bytes free; estimated ${required} bytes still required"
  fi
  log "Disk preflight: ${free_bytes} bytes free; estimated remaining requirement ${required} bytes"

  local files=("$MODEL_SHARD_1_REL" "$MODEL_SHARD_2_REL" "$MMPROJ_FILE")
  log "Downloading/checking 2 BF16 shards + mmproj from ${HF_REPO} …"

  if command -v hf >/dev/null 2>&1; then
    log "Using Hugging Face CLI: $(command -v hf)"
    local args=(download "$HF_REPO" "${files[@]}" --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR")
    (( force_download == 0 )) || args+=(--force-download)
    HF_HOME="$HF_HOME" HF_TOKEN="${HF_TOKEN:-}" HF_XET_HIGH_PERFORMANCE="$HF_XET_HIGH_PERFORMANCE" hf "${args[@]}"
    log "Download complete: ${MODEL_DIR}/"
    return
  fi

  if command -v huggingface-cli >/dev/null 2>&1; then
    log "Using legacy Hugging Face CLI: $(command -v huggingface-cli)"
    local args=(download "$HF_REPO" "${files[@]}" --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR")
    (( force_download == 0 )) || args+=(--force-download)
    HF_HOME="$HF_HOME" HF_TOKEN="${HF_TOKEN:-}" HF_XET_HIGH_PERFORMANCE="$HF_XET_HIGH_PERFORMANCE" huggingface-cli "${args[@]}"
    log "Download complete: ${MODEL_DIR}/"
    return
  fi

  local hf_python=""
  if python3 -c 'import huggingface_hub' >/dev/null 2>&1; then
    hf_python=$(command -v python3)
  elif [[ -x "${HF_VENV}/bin/python" ]] && "${HF_VENV}/bin/python" -c 'import huggingface_hub' >/dev/null 2>&1; then
    hf_python="${HF_VENV}/bin/python"
  fi

  if [[ -z "$hf_python" ]]; then
    [[ "$AUTO_INSTALL_HF" == "1" ]] || die "Hugging Face downloader not found; set AUTO_INSTALL_HF=1 or install huggingface_hub"
    log "Preparing private downloader venv: ${HF_VENV}"
    if [[ ! -x "${HF_VENV}/bin/python" ]]; then
      python3 -m venv "$HF_VENV" || die "Could not create venv. Install once: sudo apt-get install -y python3-venv"
    fi
    "${HF_VENV}/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
    "${HF_VENV}/bin/python" -m pip install --disable-pip-version-check -U pip "huggingface_hub[hf_xet]"
    hf_python="${HF_VENV}/bin/python"
  fi

  log "Using Python downloader: ${hf_python}"
  HF_HOME="$HF_HOME" HF_XET_HIGH_PERFORMANCE="$HF_XET_HIGH_PERFORMANCE" HF_TOKEN="${HF_TOKEN:-}" \
  HF_REPO="$HF_REPO" MODEL_REVISION="$MODEL_REVISION" MODEL_DIR="$MODEL_DIR" \
  MODEL_SHARD_1_REL="$MODEL_SHARD_1_REL" MODEL_SHARD_2_REL="$MODEL_SHARD_2_REL" \
  MMPROJ_FILE="$MMPROJ_FILE" FORCE_DOWNLOAD="$force_download" \
  "$hf_python" - <<'PYEOF'
from huggingface_hub import hf_hub_download
import os

files = (
    os.environ["MODEL_SHARD_1_REL"],
    os.environ["MODEL_SHARD_2_REL"],
    os.environ["MMPROJ_FILE"],
)
for filename in files:
    print(f"Downloading/checking {filename} ...", flush=True)
    path = hf_hub_download(
        repo_id=os.environ["HF_REPO"],
        filename=filename,
        revision=os.environ["MODEL_REVISION"],
        local_dir=os.environ["MODEL_DIR"],
        token=os.environ.get("HF_TOKEN") or None,
        force_download=os.environ.get("FORCE_DOWNLOAD") == "1",
    )
    print(f"Ready: {path}", flush=True)
PYEOF
  log "Download complete: ${MODEL_DIR}/"
}

# ─── verify-files ─────────────────────────────────────────────────────────────
verify_files() {
  log "Verifying both BF16 model shards and mmproj …"
  local shard1_path="${MODEL_DIR}/${MODEL_SHARD_1_REL}"
  local shard2_path="${MODEL_DIR}/${MODEL_SHARD_2_REL}"
  local mmproj_path="${MODEL_DIR}/${MMPROJ_FILE}"

  verify_one() {
    local label="$1" path="$2" expected_bytes="$3" sha="$4"
    [[ -f "$path" ]] || die "${label} not found: ${path}"
    local size magic
    size=$(stat -Lc '%s' "$path" 2>/dev/null || stat -f '%z' "$path")
    (( size == expected_bytes )) || die "${label} size mismatch: ${size} bytes; expected ${expected_bytes}"
    magic=$(LC_ALL=C head -c 4 "$path" | od -A n -t x1 | tr -d ' \n')
    [[ "$magic" == "47475546" ]] || die "${label} is not GGUF (bad magic ${magic}): ${path}"
    if [[ -n "$sha" ]]; then
      log "Verifying ${label} SHA-256 …" >&2
      echo "${sha}  ${path}" | sha256sum --check --quiet || die "${label} SHA-256 mismatch"
    fi
    printf '%s' "$size"
  }

  local s1 s2 sp
  s1=$(verify_one "model shard 1" "$shard1_path" "$SHARD_1_SIZE_BYTES" "$SHARD_1_SHA256")
  s2=$(verify_one "model shard 2" "$shard2_path" "$SHARD_2_SIZE_BYTES" "$SHARD_2_SHA256")
  sp=$(verify_one "mmproj" "$mmproj_path" "$MMPROJ_SIZE_BYTES" "$MMPROJ_SHA256")
  log "verify-files: PASS (shard1 ${s1} bytes; shard2 ${s2} bytes; mmproj ${sp} bytes)"
}

# ─── start ────────────────────────────────────────────────────────────────────
start() {
  local srv="${LLAMA_CPP_DIR}/build/bin/llama-server"
  [[ -x "$srv" ]] || die "llama-server not found: ${srv}. Run: $0 prepare-runtime"

  local model_path="${MODEL_DIR}/${MODEL_SHARD_1_REL}"
  local shard2_path="${MODEL_DIR}/${MODEL_SHARD_2_REL}"
  local mmproj_path="${MODEL_DIR}/${MMPROJ_FILE}"
  [[ -f "$model_path" ]]  || die "Model shard 1 not found: ${model_path}. Run: $0 download"
  [[ -f "$shard2_path" ]] || die "Model shard 2 not found: ${shard2_path}. Run: $0 download"
  if [[ "$VISION_MODE" == "on" ]]; then
    [[ -f "$mmproj_path" ]] || die "mmproj not found: ${mmproj_path}. Run: $0 download"
  fi

  if (( CTX_SIZE > 131072 )); then
    log "WARNING: BF16 + vision + context above 128K can place heavy pressure on DGX Spark unified memory."
  fi

  local ss_out
  ss_out=$(ss -tlnp 2>/dev/null || true)
  if [[ "$ss_out" == *":${API_PORT} "* ]]; then
    die "Port ${API_PORT} already in use. Stop the conflicting service first."
  fi

  if [[ -f "$PID_FILE" ]]; then
    local old_pid
    old_pid=$(cat "$PID_FILE")
    if kill -0 "$old_pid" 2>/dev/null; then
      die "llama-server already running (PID ${old_pid}). Run: $0 stop"
    fi
    rm -f "$PID_FILE"
  fi

  local -a args=(
    -m "$model_path"
    -ngl 999
    --ctx-size "$CTX_SIZE"
    --batch-size "$BATCH_SIZE"
    --ubatch-size "$UBATCH_SIZE"
    --flash-attn "$FLASH_ATTN"
    --no-mmap
    --cache-type-k "$CACHE_TYPE_K"
    --cache-type-v "$CACHE_TYPE_V"
    --fit off
    --threads "$N_THREADS"
    --parallel "$PARALLEL_SLOTS"
    --jinja
    --reasoning-format "$REASONING_FORMAT"
    --split-mode "$SPLIT_MODE"
    --main-gpu 0
    --temp 0.6
    --top-k 20
    --top-p 0.95
    --min-p 0.0
    --host "$API_HOST"
    --port "$API_PORT"
    --alias "$SERVED_MODEL_NAME"
  )
  [[ "$VISION_MODE" == "on" ]] && args+=(--mmproj "$mmproj_path")
  [[ -z "$API_KEY" ]] || args+=(--api-key "$API_KEY")

  log "Starting llama-server …"
  log "  Model     : ${model_path} (+ shard 2 auto-discovered)"
  log "  Vision    : ${VISION_MODE}"
  [[ "$VISION_MODE" == "on" ]] && log "  mmproj    : ${mmproj_path}"
  log "  Context   : ${CTX_SIZE} tokens"
  log "  Reasoning : ${REASONING_FORMAT}"
  log "  Split     : ${SPLIT_MODE}"
  log "  KV        : K=${CACHE_TYPE_K}, V=${CACHE_TYPE_V}"
  log "  Bind      : ${API_HOST}:${API_PORT}"
  log "  API key   : $([[ -n "$API_KEY" ]] && echo enabled || echo disabled)"

  mkdir -p "$MODEL_DIR"
  : > "$LOG_FILE"
  nohup "$srv" "${args[@]}" > "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  log "Server PID: $(cat "$PID_FILE")"

  log "Waiting for /health (timeout ${STARTUP_TIMEOUT} s) …"
  local deadline=$(( $(date +%s) + STARTUP_TIMEOUT ))
  while (( $(date +%s) < deadline )); do
    local server_pid
    server_pid=$(cat "$PID_FILE")
    if ! kill -0 "$server_pid" 2>/dev/null; then
      rm -f "$PID_FILE"
      echo "" >&2
      echo "llama-server exited during startup. Last 100 log lines:" >&2
      tail -n 100 "$LOG_FILE" >&2 || true
      die "Server exited before becoming healthy"
    fi
    if curl -fsS "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
      local adv
      adv=$(detect_advertise_ip)
      echo ""
      echo "READY"
      echo "  Model     : ${SERVED_MODEL_NAME} (BF16)"
      echo "  Vision    : ${VISION_MODE}"
      echo "  API       : http://${adv}:${API_PORT}/v1"
      echo "  Context   : ${CTX_SIZE}"
      echo "  API key   : $([[ -n "$API_KEY" ]] && echo enabled || echo disabled)"
      return
    fi
    sleep 3
  done

  echo "Last 100 log lines:" >&2
  tail -n 100 "$LOG_FILE" >&2 || true
  die "Server did not start within ${STARTUP_TIMEOUT} seconds"
}

# ─── stop ─────────────────────────────────────────────────────────────────────
stop() {
  if [[ -f "$PID_FILE" ]]; then
    local pid; pid=$(cat "$PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
      log "Stopping PID ${pid} …"
      kill "$pid"
      local t=0
      while kill -0 "$pid" 2>/dev/null && (( t < 15 )); do sleep 1; (( t++ )); done
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
    log "Stopped."
  else
    log "No PID file found. Checking by port …"
    fuser -k "${API_PORT}/tcp" 2>/dev/null || log "No process on port ${API_PORT}."
  fi
}

# ─── status ───────────────────────────────────────────────────────────────────
status() {
  echo "=== Configuration ==="
  echo "  Model  : ${HF_REPO}  BF16 (2 shards)"
  echo "  mmproj : ${MMPROJ_FILE}"
  echo "  Context: ${CTX_SIZE}"
  echo "  KV     : K=${CACHE_TYPE_K}, V=${CACHE_TYPE_V}"
  echo "  Reason : ${REASONING_FORMAT}"
  echo "  Vision : ${VISION_MODE}"
  echo "  Bind   : ${API_HOST}:${API_PORT}"
  echo "  API key: $([[ -n "$API_KEY" ]] && echo enabled || echo disabled)"

  echo ""
  echo "=== Process ==="
  if [[ -f "$PID_FILE" ]]; then
    local pid; pid=$(cat "$PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
      echo "  Running (PID ${pid})"
    else
      echo "  PID file exists but process ${pid} is dead"
    fi
  else
    echo "  Not running (no PID file)"
  fi

  echo ""
  echo "=== GPU ==="
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"

  echo ""
  echo "=== API health ==="
  if curl -fsS "http://127.0.0.1:${API_PORT}/health" 2>/dev/null; then
    echo ""
  else
    echo "  NOT READY (http://127.0.0.1:${API_PORT}/health)"
  fi

  echo ""
  echo "=== Models ==="
  if [[ -n "$API_KEY" ]]; then
    curl -fsS -H "Authorization: Bearer ${API_KEY}" "http://127.0.0.1:${API_PORT}/v1/models" 2>/dev/null | python3 -m json.tool
  else
    curl -fsS "http://127.0.0.1:${API_PORT}/v1/models" 2>/dev/null | python3 -m json.tool
  fi 2>/dev/null || echo "  (server not ready)"
}

# ─── logs ─────────────────────────────────────────────────────────────────────
logs() {
  local n="${1:-200}"
  [[ -f "$LOG_FILE" ]] || die "Log file not found: ${LOG_FILE}"
  tail -n "$n" "$LOG_FILE"
}

# ─── network-info ─────────────────────────────────────────────────────────────
network_info() {
  local adv; adv=$(detect_advertise_ip)
  echo "=== Network ==="
  echo "  Bind    : ${API_HOST}:${API_PORT}"
  echo "  API URL : http://${adv}:${API_PORT}/v1"
  echo "  Health  : http://${adv}:${API_PORT}/health"
  echo "  WebUI   : http://${adv}:${API_PORT}/"
}

# ─── client-config ────────────────────────────────────────────────────────────
client_config() {
  local adv; adv=$(detect_advertise_ip)
  cat <<CONF
# OpenAI Python SDK — Ornith-1.0-35B BF16 (single DGX Spark)
from openai import OpenAI
import base64, pathlib

client = OpenAI(base_url="http://${adv}:${API_PORT}/v1", api_key="${API_KEY:-EMPTY}")

# ── Text ──────────────────────────────────────────────────────────────────────
resp = client.chat.completions.create(
    model="${SERVED_MODEL_NAME}",
    messages=[{"role": "user", "content": "What is 17 * 83?"}],
    max_tokens=4096,
    temperature=1.0, top_p=0.95,
)
print(resp.choices[0].message.content)

# ── Image (base64) ────────────────────────────────────────────────────────────
img_b64 = base64.b64encode(pathlib.Path("photo.jpg").read_bytes()).decode()
resp = client.chat.completions.create(
    model="${SERVED_MODEL_NAME}",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": "Describe what you see in detail."},
        ],
    }],
    max_tokens=2048,
    temperature=1.0, top_p=0.95,
)
print(resp.choices[0].message.content)

# ── Image (URL) ───────────────────────────────────────────────────────────────
resp = client.chat.completions.create(
    model="${SERVED_MODEL_NAME}",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"}},
            {"type": "text", "text": "What is this image?"},
        ],
    }],
    max_tokens=512,
    temperature=1.0, top_p=0.95,
)
print(resp.choices[0].message.content)
CONF
}

# ─── props ────────────────────────────────────────────────────────────────────
props() {
  if [[ -n "$API_KEY" ]]; then
    curl -fsS -H "Authorization: Bearer ${API_KEY}"       "http://127.0.0.1:${API_PORT}/v1/models" | python3 -m json.tool
  else
    curl -fsS "http://127.0.0.1:${API_PORT}/v1/models" | python3 -m json.tool
  fi
}

# ─── bench ────────────────────────────────────────────────────────────────────
bench() {
  log "Benchmark: text-only single-stream decode …"
  python3 - <<PYEOF
import urllib.request, json, time, sys

url   = "http://127.0.0.1:${API_PORT}/v1/chat/completions"
model = "${SERVED_MODEL_NAME}"
payload = json.dumps({
    "model": model,
    "messages": [{"role":"user","content":"Write a 300-word essay on neural networks."}],
    "max_tokens": 512,
    "temperature": 0.6, "top_p": 0.95,
    "stream": False,
}).encode()
req = urllib.request.Request(url, data=payload,
    headers={"Content-Type":"application/json"}, method="POST")
t0 = time.time()
with urllib.request.urlopen(req, timeout=300) as r:
    data = json.loads(r.read())
elapsed = time.time() - t0
usage = data.get("usage", {})
toks  = usage.get("completion_tokens", 0)
print(f"Tokens     : {toks}")
print(f"Elapsed    : {elapsed:.1f}s")
print(f"Throughput : {toks/elapsed:.1f} tok/s")
print(f"Expected   : hardware/build dependent (BF16 on DGX Spark)")
PYEOF
}

# ─── stress ───────────────────────────────────────────────────────────────────
stress() {
  local n="${1:-4}"
  log "Stress test: ${n} concurrent text requests …"
  API_PORT="$API_PORT" SERVED_MODEL_NAME="$SERVED_MODEL_NAME" python3 - "$n" <<'PYEOF'
import os, sys, urllib.request, json, threading, time

url   = f"http://127.0.0.1:{os.environ['API_PORT']}/v1/chat/completions"
model = os.environ["SERVED_MODEL_NAME"]
n     = int(sys.argv[1])
ok, fail = 0, 0
lock = threading.Lock()

def req(_):
    global ok, fail
    payload = json.dumps({
        "model": model,
        "messages": [{"role":"user","content":"Count from 1 to 5."}],
        "max_tokens": 64, "temperature": 0.6,
    }).encode()
    try:
        r = urllib.request.urlopen(
            urllib.request.Request(url, data=payload,
                headers={"Content-Type":"application/json"}, method="POST"),
            timeout=120)
        json.loads(r.read())
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
    "messages": [{"role":"user","content":"Reply with exactly one sentence about Thailand."}],
    "max_tokens": 512,
    "temperature": 0.6, "top_p": 0.95,
}).encode()
req = urllib.request.Request(url, data=payload,
    headers={"Content-Type":"application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=120) as r:
    d = json.loads(r.read())
m = d["choices"][0]["message"]
msg = (m.get("content") or "").strip()
full = ((m.get("reasoning_content") or "") + msg).strip()
if not full:
    print("FAIL: empty response"); sys.exit(1)
print(f"PASS: {(msg or full)[:120]}")
PYEOF
}

# ─── test-reasoning ───────────────────────────────────────────────────────────
test_reasoning() {
  log "test-reasoning arithmetic sanity check (expect 37×43=1591) …"
  python3 - <<PYEOF
import urllib.request, json, sys

url     = "http://127.0.0.1:${API_PORT}/v1/chat/completions"
model   = "${SERVED_MODEL_NAME}"
payload = json.dumps({
    "model": model,
    "messages": [{"role":"user","content":"What is 37 multiplied by 43? Show your work."}],
    "max_tokens": 2048,
    "temperature": 0.6, "top_p": 0.95,
}).encode()
req = urllib.request.Request(url, data=payload,
    headers={"Content-Type":"application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=180) as r:
    d = json.loads(r.read())
msg  = d["choices"][0]["message"]
full = (msg.get("reasoning_content","") + msg.get("content","")).lower()
if "1591" in full:
    think_len = len(msg.get("reasoning_content",""))
    ans_len   = len(msg.get("content",""))
    print(f"PASS: 1591 found. thinking_len={think_len} answer_len={ans_len}")
else:
    print(f"FAIL: 1591 not found.\nFull: {full[:400]}")
    sys.exit(1)
PYEOF
}

# ─── test-image ───────────────────────────────────────────────────────────────
test_image() {
  log "test-image (64×64 red PNG via data URI) …"
  API_PORT="$API_PORT" SERVED_MODEL_NAME="$SERVED_MODEL_NAME" python3 - <<'PYEOF'
import os, urllib.request, json, sys, base64, struct, zlib

url   = f"http://127.0.0.1:{os.environ['API_PORT']}/v1/chat/completions"
model = os.environ["SERVED_MODEL_NAME"]

def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)

def solid_red_png(size=64):
    raw = b"".join(b"\x00" + bytes([255, 0, 0]) * size for _ in range(size))
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    return base64.b64encode(png).decode()

PNG_B64 = solid_red_png()

payload = json.dumps({
    "model": model,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}},
            {"type": "text", "text": "What color is this image? Answer in one word."},
        ],
    }],
    "max_tokens": 512,
    "temperature": 0.6, "top_p": 0.95,
}).encode()

req = urllib.request.Request(url, data=payload,
    headers={"Content-Type":"application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=120) as r:
    d = json.loads(r.read())
m = d["choices"][0]["message"]
msg = (m.get("content") or "").strip()
full = ((m.get("reasoning_content") or "") + msg).strip()
if not full:
    print("FAIL: empty image response"); sys.exit(1)
print(f"PASS: {(msg or full)[:80]}")
PYEOF
}

# ─── test-image-turn2 ─────────────────────────────────────────────────────────
test_image_turn2() {
  log "test-image-turn2 (KV cache continuity on second VL request) …"
  API_PORT="$API_PORT" SERVED_MODEL_NAME="$SERVED_MODEL_NAME" python3 - <<'PYEOF'
import os, urllib.request, json, sys, base64, struct, zlib

url   = f"http://127.0.0.1:{os.environ['API_PORT']}/v1/chat/completions"
model = os.environ["SERVED_MODEL_NAME"]

def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)

def solid_red_png(size=64):
    raw = b"".join(b"\x00" + bytes([255, 0, 0]) * size for _ in range(size))
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    return base64.b64encode(png).decode()

PNG_B64 = solid_red_png()

for i in range(2):
    payload = json.dumps({
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}},
                {"type": "text", "text": f"Turn {i+1}: Describe color of this image briefly."},
            ],
        }],
        "max_tokens": 512,
        "temperature": 0.6,
    }).encode()
    req = urllib.request.Request(url, data=payload,
        headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    m = d["choices"][0]["message"]
    msg = (m.get("content") or "").strip()
    full = ((m.get("reasoning_content") or "") + msg).strip()
    shown = msg or full
    if shown and not all(c == '?' for c in shown if c.isalpha()):
        print(f"PASS turn {i+1}: {shown[:60]}")
    else:
        print(f"FAIL turn {i+1}: garbled or empty response: '{shown[:80]}'")
        print("WARN: Known KV cache bug on 2nd VL request (see SPECIAL_FILES.md sec 6)")
        sys.exit(1)
PYEOF
}

# ─── test-tools ───────────────────────────────────────────────────────────────
test_tools() {
  local mode="${1:-required}"
  log "test-tools ${mode} (Ornith native tool calling) …"
  API_PORT="$API_PORT" SERVED_MODEL_NAME="$SERVED_MODEL_NAME" python3 - "$mode" <<'PYEOF'
import os, urllib.request, json, sys

url   = f"http://127.0.0.1:{os.environ['API_PORT']}/v1/chat/completions"
model = os.environ["SERVED_MODEL_NAME"]
mode  = sys.argv[1]
tools = [{"type":"function","function":{
  "name":"get_weather",
  "description":"Get current weather for a location",
  "parameters":{"type":"object",
    "properties":{"location":{"type":"string"}},"required":["location"]}}}]
payload = json.dumps({
    "model": model,
    "messages": [{"role":"user","content":"What is the weather in Bangkok?"}],
    "tools": tools,
    "tool_choice": mode,
    "max_tokens": 256,
    "temperature": 0.6,
}).encode()
req = urllib.request.Request(url, data=payload,
    headers={"Content-Type":"application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=120) as r:
    d = json.loads(r.read())
tc = (d["choices"][0]["message"].get("tool_calls") or [])
if tc:
    fn = tc[0]["function"]
    print(f"PASS: {fn['name']}({fn['arguments'][:80]})")
elif mode == "required":
    print(f"FAIL: no tool_calls in required mode")
    print("WARN: Inspect llama.cpp logs and embedded Jinja tool parsing")
    sys.exit(1)
else:
    print(f"WARN(auto): text response: {d['choices'][0]['message'].get('content','')[:100]}")
PYEOF
}


# ─── cline-config ─────────────────────────────────────────────────────────────
cline_config() {
  local adv
  adv=$(detect_advertise_ip)
  cat <<CONF
Cline / OpenAI Compatible:
  Provider       : OpenAI Compatible
  Base URL       : http://${adv}:${API_PORT}/v1
  API Key        : ${API_KEY:-EMPTY}
  Model ID       : ${SERVED_MODEL_NAME}
  Context Window : ${CTX_SIZE}
  Max Tokens     : 16384
  Image Support  : $([[ "$VISION_MODE" == "on" ]] && echo On || echo Off)

For Cline on another computer:
  $0 --bind 0.0.0.0 start

No API key is enabled unless --api-key is supplied.
Restrict port ${API_PORT} to a trusted LAN or VPN when binding to 0.0.0.0.
CONF
}

# ─── dispatch ─────────────────────────────────────────────────────────────────
parse_options "$@"
set -- "${REMAINING_ARGS[@]:-}"

case "${1:-help}" in
  prepare-runtime)    prepare_runtime ;;
  runtime-info)       runtime_info ;;
  download)           download "${2:-}" ;;
  verify-files)       verify_files ;;
  start)              start ;;
  stop)               stop ;;
  restart)            stop; sleep 2; start ;;
  status)             status ;;
  logs)               logs "${2:-200}" ;;
  props)              props ;;
  bench)              bench ;;
  stress)             stress "${2:-4}" ;;
  test|test-text|test-text01) test_text ;;
  test-reasoning)     test_reasoning ;;
  test-image)         test_image ;;
  test-image-turn2)   test_image_turn2 ;;
  test-tools)         test_tools "${2:-required}" ;;
  info|banner)       info ;;
  network-info)       network_info ;;
  client-config)      client_config ;;
  cline-config)       cline_config ;;
  help|--help|-h)
    cat <<HELP
Ornith-1.0-35B BF16 controller for DGX Spark (llama.cpp)

Usage: $0 [OPTIONS] COMMAND [args]

Options:
  --context TOKENS    Override ctx size (default: ${CTX_SIZE})
  --port PORT         Override API port (default: ${API_PORT})
  --bind ADDR         Override bind address (default: ${API_HOST})
  --advertise-ip IP   Override public IP for client-config / network-info
  --interface NAME    Interface for advertise-ip autodetection
  --parallel N        Parallel request slots (default: ${PARALLEL_SLOTS})
  --flash-attn MODE  Flash Attention: on, off, or auto (default: ${FLASH_ATTN})
  --split-mode MODE  GPU split: none, layer, row, or tensor (default: ${SPLIT_MODE})
  --cache-k TYPE     KV cache K type (default: ${CACHE_TYPE_K})
  --cache-v TYPE     KV cache V type (default: ${CACHE_TYPE_V})
  --reasoning-format FORMAT  deepseek or none (default: ${REASONING_FORMAT})
  --vision MODE      on or off (default: ${VISION_MODE})
  --api-key KEY      Optional llama-server API key (default: disabled)

Commands:
  prepare-runtime     Build llama.cpp from source (SM121a-real)
  runtime-info        Show binary and build ref
  download            Download missing model files; auto-creates HF venv if needed
  download --force    Re-download model and mmproj
  download --list     Show the two BF16 shards and mmproj
  verify-files        Check exact sizes, GGUF headers, and SHA-256
  start               Start llama-server (background)
  stop                Stop server
  restart             stop + start
  status              Process state, GPU, and API health
  logs [N]            Tail server log (default 200 lines)
  props               GET /v1/models
  bench               Single-stream text throughput
  stress [N]          N concurrent text requests (default 4)
  test                Alias for test-text
  test-text           Basic text generation
  test-reasoning      Arithmetic sanity test: verify 37×43=1591
  test-image          Vision: 64×64 PNG identification
  test-image-turn2    Vision: two consecutive VL requests (KV cache continuity)
  test-tools required Native tool-call test (required mode)
  test-tools auto     Native tool-call test (auto mode)
  network-info        Show bind address and public API URL
  client-config       Print OpenAI SDK usage examples
  cline-config        Print Cline/OpenAI-compatible settings

Environment overrides:
  MODEL_SHARD_1_REL First BF16 shard path inside repo
  MODEL_SHARD_2_REL Second BF16 shard path inside repo
  MMPROJ_FILE       mmproj filename override (default: mmproj-BF16.gguf)
  HF_REPO           Hugging Face repo (default: unsloth/Ornith-1.0-35B-GGUF)
  HF_VENV          Private downloader venv (default: ${HF_VENV})
  AUTO_INSTALL_HF  Auto-create downloader venv when missing: 1 or 0 (default: ${AUTO_INSTALL_HF})
  LLAMA_CPP_REF     Git ref to build (default: ${LLAMA_CPP_REF})
  USER_HOME         Base home override (default: \$HOME)
  LLAMA_CPP_DIR     Build directory (default: ${LLAMA_CPP_DIR})
  MODEL_DIR         Model storage directory (default: ${MODEL_DIR})
  CTX_SIZE          Context tokens (default: ${CTX_SIZE}; BF16 above 128K may pressure Spark memory)
  API_PORT          Server port (default: ${API_PORT})
  FLASH_ATTN        Flash Attention mode: on, off, or auto (default: ${FLASH_ATTN})
  SPLIT_MODE       GPU split mode (default: ${SPLIT_MODE}; use none on one DGX Spark)
  CACHE_TYPE_K      KV cache K type (default: ${CACHE_TYPE_K})
  CACHE_TYPE_V      KV cache V type (default: ${CACHE_TYPE_V})
  REASONING_FORMAT  deepseek or none (default: ${REASONING_FORMAT})
  VISION_MODE      on or off (default: ${VISION_MODE})
  API_KEY          Optional server API key (default: disabled)
  STARTUP_TIMEOUT  Startup timeout seconds (default: ${STARTUP_TIMEOUT})
  HF_TOKEN          HuggingFace token for gated models (not embedded)
  HF_XET_HIGH_PERFORMANCE  Use accelerated Xet download mode (default: 1)
  SHARD_1_SHA256    Expected SHA-256 for BF16 shard 1
  SHARD_2_SHA256    Expected SHA-256 for BF16 shard 2
  MMPROJ_SHA256     Expected SHA-256 for mmproj
HELP
    ;;
  *)
    die "Unknown command '${1}'. Run: $0 help"
    ;;
esac
