#!/usr/bin/env bash
#
# gemma4-26b-a4b-q8xl-single.sh
#
# Single-DGX-Spark controller for:
#   unsloth/gemma-4-26B-A4B-it-GGUF
#   gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf
#   mmproj-F16.gguf
#
# Runtime:
#   llama.cpp CUDA, built locally for NVIDIA GB10 / SM121
#
# Commands:
#   build-runtime
#   adopt-runtime
#   update-runtime
#   runtime-info
#   download
#   verify-files
#   start
#   start-thinking
#   stop
#   restart
#   status
#   logs [lines]
#   props
#   bench
#   test-text
#   test-thai
#   test-reasoning
#   test-tools [required|auto]
#   test-tool-loop
#   test-image /path/to/image
#   test-format
#   stress [minutes]
#   client-config
#
set -Eeuo pipefail

# ==============================================================================
# USER CONFIGURATION
# ==============================================================================

MASTER_IP="10.100.152.1"
SCRIPT_VERSION="1.0.2"

MODEL_ID="unsloth/gemma-4-26B-A4B-it-GGUF"

# This revision contains the updated Google Gemma 4 chat template embedded in
# the selected model and projector artifacts.
MODEL_REVISION="b19ae878a3d8d38352d69c370321f94ff98023a0"

MODEL_FILE="gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf"
MODEL_SIZE_BYTES="27636230944"
MODEL_SHA256="50e180d69641e017d7e08a6f602988effde8232ff6bc0231e839636fdcc03d8f"

MMPROJ_FILE="mmproj-F16.gguf"
MMPROJ_SIZE_BYTES="1193058784"
MMPROJ_SHA256="418a6d8723067cd712235facbbc5cba6c8fbbd413fc1292d2aace5a027d5a42f"

SERVED_MODEL_NAME="gemma-4-26b-a4b-it-q8xl"

# llama.cpp support for Gemma 4 is moving quickly. The first build checks out
# LLAMA_CPP_REF, records the exact commit, and locks later builds/startups to it.
LLAMA_CPP_REPO="https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_REF="master"
CUDA_ARCHITECTURES="121a-real"
INSTALL_BUILD_DEPENDENCIES="1"

# Conservative baseline. The model's native context is 262144.
CTX_SIZE="65536"
N_PARALLEL="1"

GPU_LAYERS="all"
FLASH_ATTN="on"
CACHE_TYPE_K="q8_0"
CACHE_TYPE_V="q8_0"

BATCH_SIZE="2048"
UBATCH_SIZE="512"

# Default server mode is fast/non-thinking. `start-thinking` uses the budget
# below and stores the active mode for restart.
DEFAULT_REASONING_MODE="off"
REASONING_BUDGET="4096"

ENABLE_VISION="1"

API_HOST="0.0.0.0"
API_PORT="8000"
BACKEND_API_PORT="8001"
ENABLE_OPENCLAW_SCHEMA_PROXY="1"
API_KEY=""

CLIENT_CONTEXT_TOKENS="49152"
CLIENT_MAX_OUTPUT_TOKENS="8192"

API_WAIT_SECONDS="1800"

# Full hashing is intentionally performed by download/verify-files. Startup
# uses exact size + GGUF header unless enabled here.
VERIFY_SHA_ON_START="0"

# Dedicated read-only media root used by test-image and the server.
MEDIA_DIR="${HOME}/gemma4-media"

# MTP is deliberately disabled in this baseline. Recent llama.cpp reports show
# that Gemma 4 MTP can reduce throughput or expose compatibility issues.
ENABLE_MTP="0"

# ==============================================================================
# PATHS / SESSIONS
# ==============================================================================

CURRENT_USER="${SUDO_USER:-$USER}"
USER_HOME="$(getent passwd "$CURRENT_USER" | cut -d: -f6)"

[[ -n "$USER_HOME" ]] || {
  echo "ERROR: Cannot resolve home directory for ${CURRENT_USER}." >&2
  exit 1
}

BASE_DIR="${USER_HOME}/gemma4-26b-a4b-q8xl"
MODEL_DIR="${USER_HOME}/models/gemma4-26b-a4b-q8xl"
LLAMA_CPP_DIR="${USER_HOME}/src/llama.cpp"

MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
MMPROJ_PATH="${MODEL_DIR}/${MMPROJ_FILE}"

LLAMA_SERVER="${LLAMA_CPP_DIR}/build/bin/llama-server"
LLAMA_BENCH="${LLAMA_CPP_DIR}/build/bin/llama-bench"

RUNTIME_LOCK="${BASE_DIR}/llama-cpp-commit.txt"
RUNTIME_INFO="${BASE_DIR}/RUNTIME_INFO.txt"
ACTIVE_MODE_FILE="${BASE_DIR}/active-mode"

BACKEND_SESSION_NAME="gemma4-26b-a4b-q8xl-backend"
PROXY_SESSION_NAME="gemma4-26b-a4b-q8xl-proxy"
LOG_DIR="${BASE_DIR}/logs"
LOG_FILE="${LOG_DIR}/llama-server.log"
PROXY_LOG_FILE="${LOG_DIR}/openclaw-schema-proxy.log"
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
PROXY_SCRIPT="${SCRIPT_DIR}/openclaw_schema_proxy.py"

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"

MEDIA_DIR="${MEDIA_DIR/#\~/$USER_HOME}"

MODEL_URL="https://huggingface.co/${MODEL_ID}/resolve/${MODEL_REVISION}/${MODEL_FILE}?download=true"
MMPROJ_URL="https://huggingface.co/${MODEL_ID}/resolve/${MODEL_REVISION}/${MMPROJ_FILE}?download=true"

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

validate_reasoning_mode() {
  case "$1" in
    off|on) ;;
    *) die "Reasoning mode must be on or off." ;;
  esac
}

active_mode() {
  cat "$ACTIVE_MODE_FILE" 2>/dev/null || printf '%s' "$DEFAULT_REASONING_MODE"
}

backend_running() {
  tmux has-session -t "$BACKEND_SESSION_NAME" 2>/dev/null
}

proxy_running() {
  tmux has-session -t "$PROXY_SESSION_NAME" 2>/dev/null
}

server_running() {
  if [[ "$ENABLE_OPENCLAW_SCHEMA_PROXY" == "1" ]]; then
    backend_running && proxy_running
  else
    backend_running
  fi
}

port_in_use_number() {
  local port="$1"
  timeout 1 bash -c \
    "</dev/tcp/127.0.0.1/${port}" 2>/dev/null
}

port_in_use() {
  port_in_use_number "$API_PORT"
}

api_auth_args() {
  API_AUTH_ARGS=()

  if [[ -n "$API_KEY" ]]; then
    API_AUTH_ARGS=(-H "Authorization: Bearer ${API_KEY}")
  fi
}

api_healthy_port() {
  local port="$1"

  curl -fsS \
    --max-time 8 \
    "http://127.0.0.1:${port}/health" \
    >/dev/null 2>&1
}

api_healthy() {
  api_healthy_port "$API_PORT"
}

wait_for_backend() {
  local deadline=$((SECONDS + API_WAIT_SECONDS))

  log "Waiting for llama.cpp backend on port ${BACKEND_API_PORT}"

  while (( SECONDS < deadline )); do
    if api_healthy_port "$BACKEND_API_PORT"; then
      return
    fi

    if ! backend_running; then
      tail -n 400 "$LOG_FILE" 2>/dev/null || true
      die "llama-server backend stopped before becoming healthy."
    fi

    sleep 5
  done

  tail -n 400 "$LOG_FILE" 2>/dev/null || true
  die "Backend did not become healthy within ${API_WAIT_SECONDS} seconds."
}

wait_for_api() {
  local deadline=$((SECONDS + API_WAIT_SECONDS))

  log "Waiting for public API on port ${API_PORT}"

  while (( SECONDS < deadline )); do
    if api_healthy; then
      return
    fi

    if ! server_running; then
      tail -n 200 "$PROXY_LOG_FILE" 2>/dev/null || true
      tail -n 400 "$LOG_FILE" 2>/dev/null || true
      die "Public API components stopped before becoming healthy."
    fi

    sleep 2
  done

  tail -n 200 "$PROXY_LOG_FILE" 2>/dev/null || true
  tail -n 400 "$LOG_FILE" 2>/dev/null || true
  die "Public API did not become healthy within ${API_WAIT_SECONDS} seconds."
}

runtime_commit() {
  [[ -d "${LLAMA_CPP_DIR}/.git" ]] || return 1
  git -C "$LLAMA_CPP_DIR" rev-parse HEAD
}

SERVER_HELP_CACHE=""

load_server_help() {
  if [[ -z "$SERVER_HELP_CACHE" ]]; then
    SERVER_HELP_CACHE="$("$LLAMA_SERVER" --help 2>&1 || true)"
  fi

  [[ -n "$SERVER_HELP_CACHE" ]] ||
    die "Unable to read llama-server --help output."
}

server_supports() {
  local flag="$1"

  load_server_help

  # Do not use `producer | grep -q` here. With `set -o pipefail`, grep may
  # exit as soon as it finds a match, the producer receives SIGPIPE, and Bash
  # reports a false failure even though the flag exists.
  grep -F -- "$flag" >/dev/null <<< "$SERVER_HELP_CACHE"
}

file_size() {
  stat -c '%s' "$1"
}

gguf_magic_ok() {
  [[ "$(LC_ALL=C head -c 4 "$1" 2>/dev/null || true)" == "GGUF" ]]
}

quick_file_check() {
  local path="$1"
  local expected_size="$2"
  local label="$3"

  [[ -f "$path" ]] ||
    die "${label} is missing: ${path}"

  local actual_size
  actual_size="$(file_size "$path")"

  [[ "$actual_size" == "$expected_size" ]] ||
    die "${label} size mismatch: expected ${expected_size}, got ${actual_size}."

  gguf_magic_ok "$path" ||
    die "${label} does not start with the GGUF magic header."
}

full_file_check() {
  local path="$1"
  local expected_size="$2"
  local expected_sha="$3"
  local label="$4"

  quick_file_check "$path" "$expected_size" "$label"

  log "Verifying ${label} SHA-256"

  printf '%s  %s\n' "$expected_sha" "$path" |
    sha256sum --check -
}

preserve_corrupt_file() {
  local path="$1"

  if [[ -e "$path" ]]; then
    local backup="${path}.corrupt.$(date +%s)"
    warn "Preserving invalid file as: $backup"
    mv -f "$path" "$backup"
  fi
}

download_and_verify_file() {
  local url="$1"
  local path="$2"
  local expected_size="$3"
  local expected_sha="$4"
  local label="$5"

  mkdir -p "$(dirname "$path")"

  if [[ -f "$path" ]]; then
    local current_size
    current_size="$(file_size "$path")"

    if [[ "$current_size" == "$expected_size" ]]; then
      if gguf_magic_ok "$path" &&
         printf '%s  %s\n' "$expected_sha" "$path" |
           sha256sum --check - >/dev/null 2>&1; then
        log "${label} is already complete and verified; skipping download"
        return
      fi

      preserve_corrupt_file "$path"
    elif (( current_size > expected_size )); then
      preserve_corrupt_file "$path"
    else
      log "Resuming ${label} from byte ${current_size}"
    fi
  fi

  log "Downloading ${label}"

  if ! curl \
    --fail \
    --location \
    --retry 10 \
    --retry-delay 5 \
    --retry-all-errors \
    --continue-at - \
    --output "$path" \
    "$url"; then
    preserve_corrupt_file "$path"
    die "${label} download failed."
  fi

  local downloaded_size
  downloaded_size="$(file_size "$path")"

  if [[ "$downloaded_size" != "$expected_size" ]] ||
     ! gguf_magic_ok "$path" ||
     ! printf '%s  %s\n' "$expected_sha" "$path" |
       sha256sum --check - >/dev/null 2>&1; then

    warn "${label} did not verify after resume; retrying once from byte zero"
    preserve_corrupt_file "$path"

    curl \
      --fail \
      --location \
      --retry 10 \
      --retry-delay 5 \
      --retry-all-errors \
      --output "$path" \
      "$url"

    full_file_check "$path" "$expected_size" "$expected_sha" "$label"
  else
    log "${label} downloaded and verified"
  fi
}

prepare_media_file() {
  local source="$1"

  [[ -f "$source" ]] ||
    die "Media file does not exist: $source"

  mkdir -p "$MEDIA_DIR"

  local clean_name
  clean_name="$(basename "$source" | tr -cs 'A-Za-z0-9._-' '_')"

  local destination="${MEDIA_DIR}/${clean_name}"

  if [[ "$(readlink -f "$source")" != \
        "$(readlink -f "$destination" 2>/dev/null || true)" ]]; then
    cp -f "$source" "$destination"
  fi

  chmod 0644 "$destination"
  printf '%s' "$clean_name"
}

# ==============================================================================
# BUILD LLAMA.CPP FOR DGX SPARK
# ==============================================================================

install_build_dependencies() {
  if [[ "$INSTALL_BUILD_DEPENDENCIES" != "1" ]]; then
    return
  fi

  log "Installing llama.cpp build dependencies"

  sudo apt-get update
  sudo apt-get install -y \
    ca-certificates \
    git \
    clang \
    cmake \
    ninja-build \
    curl \
    jq \
    python3 \
    tmux \
    libcurl4-openssl-dev \
    libssl-dev
}

checkout_locked_or_requested_runtime() {
  local update_requested="$1"
  local target_ref="$LLAMA_CPP_REF"

  if [[ "$update_requested" != "1" && -s "$RUNTIME_LOCK" ]]; then
    target_ref="$(tr -d '[:space:]' < "$RUNTIME_LOCK")"
    log "Using locked llama.cpp commit: $target_ref"
  else
    log "Using requested llama.cpp ref: $target_ref"
  fi

  git -C "$LLAMA_CPP_DIR" fetch --all --tags --prune
  git -C "$LLAMA_CPP_DIR" checkout --detach "$target_ref"

  if [[ "$update_requested" == "1" || ! -s "$RUNTIME_LOCK" ]]; then
    if [[ "$target_ref" == "master" || "$target_ref" == "main" ]]; then
      git -C "$LLAMA_CPP_DIR" checkout "$target_ref"
      git -C "$LLAMA_CPP_DIR" pull --ff-only
    fi
  fi
}

verify_runtime_features() {
  local -a required_flags=(
    "--mmproj"
    "--jinja"
    "--reasoning"
    "--reasoning-budget"
    "--flash-attn"
    "--cache-type-k"
    "--cache-type-v"
  )

  local flag
  for flag in "${required_flags[@]}"; do
    server_supports "$flag" ||
      die "Built llama-server is missing required flag: $flag"
  done

  "$LLAMA_SERVER" --version >/dev/null 2>&1 ||
    die "llama-server version check failed."
}

build_runtime_internal() {
  local update_requested="$1"

  install_build_dependencies

  require_command git
  require_command cmake
  require_command nvidia-smi

  mkdir -p "$(dirname "$LLAMA_CPP_DIR")" "$BASE_DIR"

  if [[ ! -d "${LLAMA_CPP_DIR}/.git" ]]; then
    log "Cloning llama.cpp"
    git clone "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR"
  fi

  checkout_locked_or_requested_runtime "$update_requested"

  local commit
  commit="$(runtime_commit)"

  log "Configuring CUDA build for DGX Spark GB10 / SM121"

  cmake \
    -S "$LLAMA_CPP_DIR" \
    -B "${LLAMA_CPP_DIR}/build" \
    -G Ninja \
    -DGGML_NATIVE=ON \
    -DGGML_CUDA=ON \
    -DGGML_CURL=ON \
    -DGGML_RPC=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES"

  log "Building llama-server and llama-bench"

  cmake \
    --build "${LLAMA_CPP_DIR}/build" \
    --config Release \
    --target llama-server llama-bench \
    -j "$(nproc)"

  [[ -x "$LLAMA_SERVER" ]] ||
    die "llama-server was not produced."

  [[ -x "$LLAMA_BENCH" ]] ||
    die "llama-bench was not produced."

  verify_runtime_features

  write_runtime_lock "$commit"

  log "Runtime build complete and locked"
  cat "$RUNTIME_INFO"
}

write_runtime_lock() {
  local commit="$1"

  mkdir -p "$BASE_DIR"

  printf '%s\n' "$commit" > "$RUNTIME_LOCK"

  {
    echo "llama_cpp_repository=${LLAMA_CPP_REPO}"
    echo "llama_cpp_commit=${commit}"
    echo "cuda_architectures=${CUDA_ARCHITECTURES}"
    echo "script_version=${SCRIPT_VERSION}"
    echo "built_or_adopted_at=$(date --iso-8601=seconds)"
    "$LLAMA_SERVER" --version 2>&1 || true
  } > "$RUNTIME_INFO"
}

adopt_runtime() {
  require_command git

  [[ -x "$LLAMA_SERVER" ]] ||
    die "Existing llama-server binary not found: $LLAMA_SERVER"

  [[ -d "${LLAMA_CPP_DIR}/.git" ]] ||
    die "Existing llama.cpp Git checkout not found: $LLAMA_CPP_DIR"

  SERVER_HELP_CACHE=""
  verify_runtime_features

  local commit
  commit="$(runtime_commit)"

  write_runtime_lock "$commit"

  log "Existing successful build adopted without recompiling"
  cat "$RUNTIME_INFO"
}

build_runtime() {
  build_runtime_internal 0
}

update_runtime() {
  warn "This intentionally advances llama.cpp and replaces the runtime lock."
  rm -f "$RUNTIME_LOCK"
  build_runtime_internal 1
}

runtime_info() {
  echo "===== LOCK ====="
  cat "$RUNTIME_LOCK" 2>/dev/null || echo "No runtime lock"

  echo
  echo "===== BUILD INFO ====="
  cat "$RUNTIME_INFO" 2>/dev/null || echo "No runtime info"

  echo
  echo "===== GIT ====="
  if [[ -d "${LLAMA_CPP_DIR}/.git" ]]; then
    git -C "$LLAMA_CPP_DIR" log -1 --oneline
    git -C "$LLAMA_CPP_DIR" status --short --branch
  fi

  echo
  echo "===== REQUIRED FLAGS ====="
  if [[ -x "$LLAMA_SERVER" ]]; then
    for flag in \
      --mmproj \
      --media-path \
      --jinja \
      --tools \
      --reasoning \
      --reasoning-budget \
      --spec-type; do
      if server_supports "$flag"; then
        echo "$flag: yes"
      else
        echo "$flag: no"
      fi
    done
  else
    echo "llama-server is not built"
  fi

  echo
  echo "NOTE: --tools is a built-in local file/shell feature in current llama.cpp."
  echo "This controller intentionally does not enable it."
  echo "External OpenAI-compatible tool calls remain handled by the client/agent."
}

# ==============================================================================
# DOWNLOAD / VERIFY
# ==============================================================================

download_model() {
  require_command curl
  require_command sha256sum
  require_command df

  mkdir -p "$MODEL_DIR"

  log "Disk space before download"
  df -h "$MODEL_DIR"

  log "Expected downloads"
  echo "Model:  ${MODEL_SIZE_BYTES} bytes"
  if [[ "$ENABLE_VISION" == "1" ]]; then
    echo "mmproj: ${MMPROJ_SIZE_BYTES} bytes"
  fi

  download_and_verify_file \
    "$MODEL_URL" \
    "$MODEL_PATH" \
    "$MODEL_SIZE_BYTES" \
    "$MODEL_SHA256" \
    "$MODEL_FILE"

  if [[ "$ENABLE_VISION" == "1" ]]; then
    download_and_verify_file \
      "$MMPROJ_URL" \
      "$MMPROJ_PATH" \
      "$MMPROJ_SIZE_BYTES" \
      "$MMPROJ_SHA256" \
      "$MMPROJ_FILE"
  fi

  verify_files

  log "Download completed"
  du -sh "$MODEL_DIR"
}

verify_files() {
  require_command sha256sum

  full_file_check \
    "$MODEL_PATH" \
    "$MODEL_SIZE_BYTES" \
    "$MODEL_SHA256" \
    "$MODEL_FILE"

  if [[ "$ENABLE_VISION" == "1" ]]; then
    full_file_check \
      "$MMPROJ_PATH" \
      "$MMPROJ_SIZE_BYTES" \
      "$MMPROJ_SHA256" \
      "$MMPROJ_FILE"
  fi

  log "All selected GGUF artifacts verified"
}

# ==============================================================================
# START / STOP
# ==============================================================================

serve_foreground() {
  local mode="${2:-$DEFAULT_REASONING_MODE}"
  validate_reasoning_mode "$mode"

  local -a args
  local -a auth_args

  [[ -x "$LLAMA_SERVER" ]] ||
    die "llama-server is not built. Run: $0 build-runtime"

  quick_file_check \
    "$MODEL_PATH" \
    "$MODEL_SIZE_BYTES" \
    "$MODEL_FILE"

  if [[ "$ENABLE_VISION" == "1" ]]; then
    quick_file_check \
      "$MMPROJ_PATH" \
      "$MMPROJ_SIZE_BYTES" \
      "$MMPROJ_FILE"
  fi

  mkdir -p "$LOG_DIR" "$MEDIA_DIR"

  args=(
    --model "$MODEL_PATH"
    --alias "$SERVED_MODEL_NAME"
    --host "$API_HOST"
    --port "$BACKEND_API_PORT"
    --ctx-size "$CTX_SIZE"
    --parallel "$N_PARALLEL"
    --n-gpu-layers "$GPU_LAYERS"
    --flash-attn "$FLASH_ATTN"
    --cache-type-k "$CACHE_TYPE_K"
    --cache-type-v "$CACHE_TYPE_V"
    --batch-size "$BATCH_SIZE"
    --ubatch-size "$UBATCH_SIZE"
    --cont-batching
    --jinja
    --reasoning "$mode"
    --temp 1.0
    --top-k 64
    --top-p 0.95
    --min-p 0.0
    --repeat-penalty 1.0
    --metrics
    --timeout 3600
    --log-timestamps
  )

  if [[ "$mode" == "on" ]]; then
    args+=(--reasoning-budget "$REASONING_BUDGET")
  fi

  if [[ "$ENABLE_VISION" == "1" ]]; then
    args+=(--mmproj "$MMPROJ_PATH")

    if server_supports --media-path; then
      args+=(--media-path "$MEDIA_DIR")
    fi
  fi

  if [[ "$ENABLE_MTP" == "1" ]]; then
    die "MTP is intentionally not configured in this baseline package."
  fi

  auth_args=()
  if [[ -n "$API_KEY" ]]; then
    auth_args=(--api-key "$API_KEY")
  fi

  exec "$LLAMA_SERVER" \
    "${args[@]}" \
    "${auth_args[@]}"
}

stop_server() {
  require_command tmux

  if proxy_running; then
    log "Stopping schema proxy session: $PROXY_SESSION_NAME"
    tmux kill-session -t "$PROXY_SESSION_NAME"
  fi

  if backend_running; then
    log "Stopping llama.cpp backend session: $BACKEND_SESSION_NAME"
    tmux kill-session -t "$BACKEND_SESSION_NAME"
  fi

  if ! proxy_running && ! backend_running; then
    log "Server sessions are stopped"
  fi
}

serve_proxy_foreground() {
  [[ -f "$PROXY_SCRIPT" ]] ||
    die "Schema proxy is missing: $PROXY_SCRIPT"

  exec python3 "$PROXY_SCRIPT" \
    --listen-host "$API_HOST" \
    --listen-port "$API_PORT" \
    --upstream-host 127.0.0.1 \
    --upstream-port "$BACKEND_API_PORT"
}

start_server_mode() {
  local mode="$1"
  validate_reasoning_mode "$mode"

  require_command tmux
  require_command curl
  require_command timeout
  require_command nvidia-smi
  require_command git

  nvidia-smi >/dev/null 2>&1 ||
    die "Host nvidia-smi failed."

  [[ -x "$LLAMA_SERVER" ]] ||
    die "llama-server is not built. Run: $0 build-runtime"

  [[ -s "$RUNTIME_LOCK" ]] ||
    die "Runtime lock is missing. Run: $0 build-runtime"

  local actual_commit
  actual_commit="$(runtime_commit)"

  local locked_commit
  locked_commit="$(tr -d '[:space:]' < "$RUNTIME_LOCK")"

  [[ "$actual_commit" == "$locked_commit" ]] ||
    die "llama.cpp source differs from the locked runtime commit."

  quick_file_check \
    "$MODEL_PATH" \
    "$MODEL_SIZE_BYTES" \
    "$MODEL_FILE"

  if [[ "$ENABLE_VISION" == "1" ]]; then
    quick_file_check \
      "$MMPROJ_PATH" \
      "$MMPROJ_SIZE_BYTES" \
      "$MMPROJ_FILE"
  fi

  if [[ "$VERIFY_SHA_ON_START" == "1" ]]; then
    verify_files
  fi

  mkdir -p "$LOG_DIR" "$MEDIA_DIR"
  : > "$LOG_FILE"
  : > "$PROXY_LOG_FILE"

  if backend_running || proxy_running; then
    stop_server
    sleep 2
  fi

  if port_in_use_number "$BACKEND_API_PORT"; then
    die "Backend port ${BACKEND_API_PORT} is already in use."
  fi

  if port_in_use; then
    die "Public port ${API_PORT} is already in use."
  fi

  if [[ "$ENABLE_OPENCLAW_SCHEMA_PROXY" == "1" ]]; then
    require_command python3
    [[ -f "$PROXY_SCRIPT" ]] ||
      die "Missing OpenClaw schema proxy: $PROXY_SCRIPT"
    python3 "$PROXY_SCRIPT" --self-test
  fi

  local backend_command
  printf -v backend_command \
    'exec bash %q _serve %q >> %q 2>&1' \
    "$SCRIPT_PATH" \
    "$mode" \
    "$LOG_FILE"

  log "Starting Gemma 4 26B-A4B Q8_K_XL backend"
  log "Runtime commit: $locked_commit"
  log "Context:        $CTX_SIZE"
  log "KV cache:       K=${CACHE_TYPE_K} V=${CACHE_TYPE_V}"
  log "Reasoning:      $mode"
  log "Vision:         $ENABLE_VISION"
  log "Backend:        http://127.0.0.1:${BACKEND_API_PORT}/v1"

  tmux new-session \
    -d \
    -s "$BACKEND_SESSION_NAME" \
    "$backend_command"

  wait_for_backend

  if [[ "$ENABLE_OPENCLAW_SCHEMA_PROXY" == "1" ]]; then
    local proxy_command
    printf -v proxy_command \
      'exec bash %q _serve_proxy >> %q 2>&1' \
      "$SCRIPT_PATH" \
      "$PROXY_LOG_FILE"

    log "Starting OpenClaw JSON-schema compatibility proxy"
    tmux new-session \
      -d \
      -s "$PROXY_SESSION_NAME" \
      "$proxy_command"
  fi

  printf '%s\n' "$mode" > "$ACTIVE_MODE_FILE"

  wait_for_api

  log "READY"
  echo "Base URL: http://${MASTER_IP}:${API_PORT}/v1"
  echo "Model:    ${SERVED_MODEL_NAME}"
  echo "Context:  ${CTX_SIZE}"
  echo "Reasoning:${mode}"
  echo "Backend:  http://127.0.0.1:${BACKEND_API_PORT}/v1"
  echo "Proxy:    ${ENABLE_OPENCLAW_SCHEMA_PROXY}"
}

start_server() {
  start_server_mode "$DEFAULT_REASONING_MODE"
}

start_thinking() {
  start_server_mode "on"
}

restart_server() {
  local mode
  mode="$(active_mode)"
  stop_server
  start_server_mode "$mode"
}

# ==============================================================================
# STATUS / LOGS / PROPS / BENCH
# ==============================================================================

show_status() {
  require_command curl
  require_command tmux

  echo "===== CONFIG ====="
  echo "Script version:    $SCRIPT_VERSION"
  echo "Repository:        $MODEL_ID"
  echo "Model revision:    $MODEL_REVISION"
  echo "Model file:        $MODEL_FILE"
  echo "Served name:       $SERVED_MODEL_NAME"
  echo "Context:           $CTX_SIZE"
  echo "KV cache:          K=$CACHE_TYPE_K V=$CACHE_TYPE_V"
  echo "Reasoning mode:    $(active_mode)"
  echo "Vision:            $ENABLE_VISION"
  echo "MTP:               $ENABLE_MTP"
  echo "Public API port:   $API_PORT"
  echo "Backend API port:  $BACKEND_API_PORT"
  echo "Schema proxy:      $ENABLE_OPENCLAW_SCHEMA_PROXY"
  echo

  echo "===== RUNTIME ====="
  if [[ -x "$LLAMA_SERVER" ]]; then
    "$LLAMA_SERVER" --version || true
    echo "Locked commit: $(cat "$RUNTIME_LOCK" 2>/dev/null || echo missing)"
  else
    echo "Not built"
  fi

  echo
  echo "===== SESSIONS ====="
  if backend_running; then
    tmux list-sessions | grep "^${BACKEND_SESSION_NAME}:" || true
  else
    echo "BACKEND: STOPPED"
  fi
  if [[ "$ENABLE_OPENCLAW_SCHEMA_PROXY" == "1" ]]; then
    if proxy_running; then
      tmux list-sessions | grep "^${PROXY_SESSION_NAME}:" || true
    else
      echo "PROXY: STOPPED"
    fi
  fi

  echo
  echo "===== GPU ====="
  nvidia-smi \
    --query-gpu=name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null || true

  echo
  echo "===== PROCESS ====="
  pgrep -af "$LLAMA_SERVER" || true

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
}

show_logs() {
  local lines="${1:-400}"

  echo "===== LLAMA.CPP BACKEND ====="
  if [[ -f "$LOG_FILE" ]]; then
    tail -n "$lines" "$LOG_FILE"
  else
    echo "No backend log yet: $LOG_FILE"
  fi

  if [[ "$ENABLE_OPENCLAW_SCHEMA_PROXY" == "1" ]]; then
    echo
    echo "===== OPENCLAW SCHEMA PROXY ====="
    if [[ -f "$PROXY_LOG_FILE" ]]; then
      tail -n "$lines" "$PROXY_LOG_FILE"
    else
      echo "No proxy log yet: $PROXY_LOG_FILE"
    fi
  fi
}

show_props() {
  api_auth_args

  curl -sS \
    "${API_AUTH_ARGS[@]}" \
    "http://127.0.0.1:${API_PORT}/props"

  echo
}

run_bench() {
  [[ -x "$LLAMA_BENCH" ]] ||
    die "llama-bench is not built. Run: $0 build-runtime"

  quick_file_check \
    "$MODEL_PATH" \
    "$MODEL_SIZE_BYTES" \
    "$MODEL_FILE"

  "$LLAMA_BENCH" \
    --model "$MODEL_PATH" \
    --n-gpu-layers "$GPU_LAYERS" \
    --flash-attn "$FLASH_ATTN" \
    --batch-size "$BATCH_SIZE" \
    --ubatch-size "$UBATCH_SIZE" \
    --prompt 2048 \
    --n-gen 256 \
    --repetitions 3
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
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 64,
  "repeat_penalty": 1.0,
  "max_tokens": 4096
}
EOF

  echo
}

test_thai() {
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
      "content": "อธิบายหลักการ Zero Trust สำหรับ AI Agent แบบกระชับ พร้อมมาตรการที่นำไปใช้ได้จริง 5 ข้อ"
    }
  ],
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 64,
  "max_tokens": 2048
}
EOF

  echo
}

test_reasoning() {
  if [[ "$(active_mode)" != "on" ]]; then
    warn "Server reasoning mode is off. Run: $0 start-thinking"
  fi

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
      "content": "Design a resilient local coding-agent platform. Compare failure modes, security boundaries, observability, and recovery procedures."
    }
  ],
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 64,
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
  "parallel_tool_calls": false,
  "temperature": 0,
  "max_tokens": 2048
}
EOF

  echo
  echo
  echo "Expected: choices[0].message.tool_calls"
  echo "No tool is executed by this command."
}

test_tool_loop() {
  require_command python3

  local key="${API_KEY:-llama-local}"

  OPENAI_BASE_URL="http://127.0.0.1:${API_PORT}/v1" \
  OPENAI_API_KEY="$key" \
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


def call(payload: dict) -> dict:
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(request, timeout=900) as response:
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
            "Read src/app.py before identifying the bug. "
            "Use the read_file tool first."
        ),
    }
]

first = call(
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
        "FAIL: no structured tool_calls\n"
        + json.dumps(first, indent=2, ensure_ascii=False)
    )

tool_call = tool_calls[0]
function = tool_call.get("function") or {}
arguments = json.loads(function.get("arguments") or "{}")

if function.get("name") != "read_file":
    raise SystemExit(f"FAIL: unexpected tool call: {tool_call}")

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

second = call(
    {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "temperature": 0,
        "max_tokens": 4096,
    }
)

second_message = second["choices"][0]["message"]
content = second_message.get("content") or ""

if not content.strip():
    raise SystemExit(
        "FAIL: final content is empty\n"
        + json.dumps(second, indent=2, ensure_ascii=False)
    )

print("PASS: structured tool call and tool-result continuation")
print(content)
PY
}

test_image() {
  [[ "$ENABLE_VISION" == "1" ]] ||
    die "Vision is disabled in the controller."

  local source="${1:-}"

  [[ -n "$source" ]] ||
    die "Usage: $0 test-image /path/to/image"

  local media_name
  media_name="$(prepare_media_file "$source")"

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
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "file://${media_name}"
          }
        },
        {
          "type": "text",
          "text": "Describe the image accurately. If text is present, transcribe only what is clearly visible."
        }
      ]
    }
  ],
  "temperature": 0.2,
  "max_tokens": 2048
}
EOF

  echo
}

test_format() {
  require_command python3

  local key="${API_KEY:-llama-local}"

  OPENAI_BASE_URL="http://127.0.0.1:${API_PORT}/v1" \
  OPENAI_API_KEY="$key" \
  OPENAI_MODEL="$SERVED_MODEL_NAME" \
  python3 - <<'PY'
from __future__ import annotations

import json
import os
import urllib.request

base_url = os.environ["OPENAI_BASE_URL"].rstrip("/")
api_key = os.environ["OPENAI_API_KEY"]
model = os.environ["OPENAI_MODEL"]

payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": "State one benefit of unit tests in one sentence.",
        }
    ],
    "temperature": 0.2,
    "max_tokens": 512,
}

request = urllib.request.Request(
    f"{base_url}/chat/completions",
    data=json.dumps(payload).encode(),
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    },
)

with urllib.request.urlopen(request, timeout=600) as response:
    result = json.loads(response.read())

message = result["choices"][0]["message"]
content = message.get("content") or ""

forbidden = (
    "<unused",
    "<|tool_call>",
    "<tool_call|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
)

leaked = [marker for marker in forbidden if marker in content]
if leaked:
    raise SystemExit(
        "FAIL: raw control markers leaked: "
        + ", ".join(leaked)
        + "\n"
        + json.dumps(result, indent=2, ensure_ascii=False)
    )

if not content.strip():
    raise SystemExit(
        "FAIL: final content is empty\n"
        + json.dumps(result, indent=2, ensure_ascii=False)
    )

print("PASS: no known raw Gemma/control markers in final content")
print("Message keys:", sorted(message))
print(content)
PY
}


test_schema_proxy() {
  [[ "$ENABLE_OPENCLAW_SCHEMA_PROXY" == "1" ]] ||
    die "Schema proxy is disabled."

  require_command python3

  local key="${API_KEY:-llama-local}"

  OPENAI_BASE_URL="http://127.0.0.1:${API_PORT}/v1" \
  OPENAI_API_KEY="$key" \
  OPENAI_MODEL="$SERVED_MODEL_NAME" \
  python3 - <<'PY'
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

base = os.environ["OPENAI_BASE_URL"].rstrip("/")
key = os.environ["OPENAI_API_KEY"]
model = os.environ["OPENAI_MODEL"]

payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": "Call read_file for policy/demo.txt.",
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a synthetic policy file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "pattern": "policy/[A-Za-z0-9._/-]+"
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": False
                }
            }
        }
    ],
    "tool_choice": "required",
    "parallel_tool_calls": False,
    "temperature": 0,
    "max_tokens": 1024
}

request = urllib.request.Request(
    f"{base}/chat/completions",
    data=json.dumps(payload).encode(),
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    },
)

try:
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.loads(response.read())
except urllib.error.HTTPError as exc:
    body = exc.read().decode(errors="replace")
    raise SystemExit(f"FAIL HTTP {exc.code}: {body}")

message = result["choices"][0]["message"]
print("PASS: unanchored JSON-schema pattern no longer caused HTTP 400")
print(json.dumps(message, indent=2, ensure_ascii=False))
PY
}

stress_test() {
  local minutes="${1:-30}"

  [[ "$minutes" =~ ^[0-9]+$ ]] ||
    die "Stress duration must be an integer number of minutes."

  require_command python3

  local key="${API_KEY:-llama-local}"

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

base_url = os.environ["OPENAI_BASE_URL"].rstrip("/")
api_key = os.environ["OPENAI_API_KEY"]
model = os.environ["OPENAI_MODEL"]
deadline = time.monotonic() + int(os.environ["STRESS_MINUTES"]) * 60

prompts = [
    "Write a safe atomic file replacement function in Python.",
    "Explain a two-lock deadlock and the smallest reliable fix.",
    "Design unit tests for a token-bucket rate limiter.",
    "Review an HTTP retry policy and identify unsafe edge cases.",
]

successes = 0
failures = 0
index = 0

while time.monotonic() < deadline:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompts[index % len(prompts)],
            }
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "repeat_penalty": 1.0,
        "max_tokens": 1024,
    }

    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    started = time.monotonic()

    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            result = json.loads(response.read())

        content = result["choices"][0]["message"].get("content") or ""
        if not content.strip():
            raise RuntimeError("empty content")

        if "<unused" in content:
            raise RuntimeError("raw <unused...> token leaked")

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
            f"FAIL request={successes + failures} "
            f"elapsed={time.monotonic() - started:.1f}s "
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
  local key="${API_KEY:-llama-local}"

  cat <<EOF
==============================================================================
OpenAI-compatible endpoint
==============================================================================

Provider:          OpenAI Compatible
Base URL:          http://${MASTER_IP}:${API_PORT}/v1
Backend URL:       http://127.0.0.1:${BACKEND_API_PORT}/v1
API key:           ${key}
Model ID:          ${SERVED_MODEL_NAME}
Context window:    ${CTX_SIZE}
Max input tokens:  ${CLIENT_CONTEXT_TOKENS}
Max output tokens: ${CLIENT_MAX_OUTPUT_TOKENS}

export OPENAI_BASE_URL="http://${MASTER_IP}:${API_PORT}/v1"
export OPENAI_API_KEY="${key}"
export OPENAI_MODEL="${SERVED_MODEL_NAME}"

Capabilities:
  text input/output:       yes
  image input:             ${ENABLE_VISION}
  audio input:             no for 26B-A4B
  native reasoning:        yes; server mode on/off
  native function tools:   yes; must pass acceptance tests
  parallel tools:          start with false

Recommended sampling:
  temperature:       1.0
  top_p:             0.95
  top_k:             64
  min_p:             0
  repeat_penalty:    1.0

Agent notes:
  - Start with `start`, not `start-thinking`, for tool workflows.
  - Send native OpenAI `tools`.
  - Set parallel_tool_calls=false during validation.
  - Validate every JSON argument before execution.
  - Return tool results with role=tool and tool_call_id.
  - Use workspace, command, path, network, and credential sandboxes.
  - Gemma 4 tool parsing has had runtime edge cases; run test-tool-loop.
EOF
}

show_help() {
  cat <<EOF
Usage:
  $(basename "$0") build-runtime
  $(basename "$0") adopt-runtime
  $(basename "$0") update-runtime
  $(basename "$0") runtime-info
  $(basename "$0") download
  $(basename "$0") verify-files
  $(basename "$0") start
  $(basename "$0") start-thinking
  $(basename "$0") stop
  $(basename "$0") restart
  $(basename "$0") status
  $(basename "$0") logs [number_of_lines]
  $(basename "$0") props
  $(basename "$0") bench
  $(basename "$0") test-text
  $(basename "$0") test-thai
  $(basename "$0") test-reasoning
  $(basename "$0") test-tools [required|auto]
  $(basename "$0") test-tool-loop
  $(basename "$0") test-image /path/to/image
  $(basename "$0") test-format
  $(basename "$0") test-schema-proxy
  $(basename "$0") stress [minutes]
  $(basename "$0") client-config

First setup:
  1. ./$(basename "$0") build-runtime

Recovery after a successful compile followed by a feature-check error:
  ./$(basename "$0") adopt-runtime

  2. ./$(basename "$0") download
  3. Stop another server using GPU or port ${API_PORT}.
  4. ./$(basename "$0") start
  5. ./$(basename "$0") props
  6. ./$(basename "$0") test-format
  7. ./$(basename "$0") test-schema-proxy
  7. ./$(basename "$0") test-text
  8. ./$(basename "$0") test-tools required
  9. ./$(basename "$0") test-tool-loop
 10. ./$(basename "$0") stress 30

Thinking mode:
  ./$(basename "$0") start-thinking
  ./$(basename "$0") test-reasoning

After reboot:
  ./$(basename "$0") start
EOF
}

case "${1:-help}" in
  build-runtime)
    build_runtime
    ;;
  adopt-runtime)
    adopt_runtime
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
  verify-files)
    verify_files
    ;;
  start)
    start_server
    ;;
  start-thinking)
    start_thinking
    ;;
  stop)
    stop_server
    ;;
  restart)
    restart_server
    ;;
  status)
    show_status
    ;;
  logs)
    show_logs "${2:-400}"
    ;;
  props)
    show_props
    ;;
  bench)
    run_bench
    ;;
  test-text)
    test_text
    ;;
  test-thai)
    test_thai
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
  test-image)
    test_image "${2:-}"
    ;;
  test-format)
    test_format
    ;;
  test-schema-proxy)
    test_schema_proxy
    ;;
  stress)
    stress_test "${2:-30}"
    ;;
  client-config)
    print_client_config
    ;;
  _serve)
    serve_foreground "$@"
    ;;
  _serve_proxy)
    serve_proxy_foreground
    ;;
  help|-h|--help)
    show_help
    ;;
  *)
    show_help
    exit 1
    ;;
esac
