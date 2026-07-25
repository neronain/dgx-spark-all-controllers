#!/usr/bin/env bash
#
# qwen36-hauhau-q6kp-single.sh
#
# Single-DGX-Spark controller for:
#   HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
#   Quant: Q6_K_P (GGUF)
#
# Runtime:
#   llama.cpp built locally with CUDA for DGX Spark GB10 (sm_121a)
#
# Commands:
#   ./qwen36-hauhau-q6kp-single.sh build-runtime
#   ./qwen36-hauhau-q6kp-single.sh download
#   ./qwen36-hauhau-q6kp-single.sh verify-files
#   ./qwen36-hauhau-q6kp-single.sh start
#   ./qwen36-hauhau-q6kp-single.sh stop
#   ./qwen36-hauhau-q6kp-single.sh restart
#   ./qwen36-hauhau-q6kp-single.sh status
#   ./qwen36-hauhau-q6kp-single.sh logs [lines]
#   ./qwen36-hauhau-q6kp-single.sh props
#   ./qwen36-hauhau-q6kp-single.sh test-text
#   ./qwen36-hauhau-q6kp-single.sh test-reasoning
#   ./qwen36-hauhau-q6kp-single.sh test-tools [required|auto]
#   ./qwen36-hauhau-q6kp-single.sh test-image /path/to/image.png
#   ./qwen36-hauhau-q6kp-single.sh test-video /path/to/video.mp4
#   ./qwen36-hauhau-q6kp-single.sh client-config
#
set -Eeuo pipefail

SCRIPT_VERSION="${SCRIPT_VERSION:-3.1.0}"
MODEL_LABEL="${MODEL_LABEL:-Qwen3.6-35B-A3B Uncensored HauhauCS (Q6_K_XL GGUF)}"
RUNTIME_LABEL="${RUNTIME_LABEL:-llama.cpp}"
MODEL_FEATURES="${MODEL_FEATURES:-tools · uncensored · MoE}"

# ==============================================================================
# USER CONFIGURATION — edit values here
# ==============================================================================


REPO_ID="HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive"
REPO_REVISION="main"

MODEL_FILE="Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf"
MMPROJ_FILE="mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf"

# Published SHA-256 values from the Hugging Face file pages.
MODEL_SHA256="90281d33e0790d6da2f125aa3f4352f429d8cc2ce2b32caafd78896397756fc3"
MMPROJ_SHA256="c8e702344a81f8c226a914aa980ed6e1f604bce9374f1fed8e65c896908af414"

# Exact remote sizes. Bash arithmetic does not allow underscore separators.
MODEL_SIZE_BYTES="30649317504"
MMPROJ_SIZE_BYTES="899283072"

SERVED_MODEL_NAME="qwen36-hauhau-q6kp"

# New Qwen3.6 support changes quickly. Start from current upstream master,
# then run `runtime-info` and pin LLAMA_CPP_REF to that commit after validation.
LLAMA_CPP_REPO="https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_REF="master"

# NVIDIA DGX Spark / GB10 build settings.
CUDA_ARCHITECTURES="121a-real"
INSTALL_BUILD_DEPENDENCIES="1"

# Qwen recommends at least 128K to preserve thinking quality.
CTX_SIZE="${CTX_SIZE:-131072}"

# One slot is the safest starting point for maximum context and agent stability.
N_PARALLEL="1"

GPU_LAYERS="all"
FLASH_ATTN="on"

# q8_0 reduces KV memory versus f16 while remaining conservative.
CACHE_TYPE_K="q8_0"
CACHE_TYPE_V="q8_0"

BATCH_SIZE="2048"
UBATCH_SIZE="512"

# Coding agents are normally faster and more reliable with thinking disabled.
DEFAULT_ENABLE_THINKING="false"

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
ADVERTISE_IP="${ADVERTISE_IP:-}"
ADVERTISE_INTERFACE="${ADVERTISE_INTERFACE:-}"
ROUTE_PROBE_IP="${ROUTE_PROBE_IP:-1.1.1.1}"
CLIENT_OVERHEAD_TOKENS="${CLIENT_OVERHEAD_TOKENS:-8192}"

# Empty = no authentication. Set a strong value for shared/LAN use.
API_KEY="${API_KEY:-}"

# Full hashes are always checked by `verify-files`.
# Enable only if you want to re-hash ~31.5 GB before every start.
VERIFY_SHA_ON_START="0"

# Dedicated local media directory exposed read-only to llama-server.
MEDIA_DIR="${HOME}/qwen36-media"

# Client token budget. Input + output + template/tool overhead must fit CTX_SIZE.
CLIENT_CONTEXT_TOKENS="${CLIENT_CONTEXT_TOKENS:-auto}"
CLIENT_MAX_OUTPUT_TOKENS="${CLIENT_MAX_OUTPUT_TOKENS:-8192}"

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

BASE_DIR="${USER_HOME}/qwen36-hauhau-q6kp"
MODEL_DIR="${USER_HOME}/models/qwen36-hauhau-q6kp"
LLAMA_CPP_DIR="${USER_HOME}/src/llama.cpp"

MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
MMPROJ_PATH="${MODEL_DIR}/${MMPROJ_FILE}"
LLAMA_SERVER="${LLAMA_CPP_DIR}/build/bin/llama-server"

MEDIA_DIR="${MEDIA_DIR/#\~/$USER_HOME}"

SESSION_NAME="qwen36-hauhau-q6kp"
LOG_DIR="${BASE_DIR}/logs"
LOG_FILE="${LOG_DIR}/llama-server.log"
RUNTIME_COMMIT_FILE="${BASE_DIR}/llama-cpp-commit.txt"

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"

MODEL_URL="https://huggingface.co/${REPO_ID}/resolve/${REPO_REVISION}/${MODEL_FILE}?download=true"
MMPROJ_URL="https://huggingface.co/${REPO_ID}/resolve/${REPO_REVISION}/${MMPROJ_FILE}?download=true"

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
    computed=$(( CTX_SIZE - CLIENT_MAX_OUTPUT_TOKENS - CLIENT_OVERHEAD_TOKENS ))

    (( computed >= 1024 )) ||
      die "Context ${CTX_SIZE} is too small for output and overhead budgets."

    CLIENT_CONTEXT_TOKENS="$computed"
  fi
}

validate_common_config() {
  validate_positive_integer "CTX_SIZE" "${CTX_SIZE}"
  validate_port "$API_PORT"
  validate_positive_integer "CLIENT_MAX_OUTPUT_TOKENS" "${CLIENT_MAX_OUTPUT_TOKENS}"
  validate_positive_integer "CLIENT_OVERHEAD_TOKENS" "$CLIENT_OVERHEAD_TOKENS"

  if [[ "${CLIENT_CONTEXT_TOKENS}" != "auto" ]]; then
    validate_positive_integer "CLIENT_CONTEXT_TOKENS" "${CLIENT_CONTEXT_TOKENS}"
  fi

  resolve_client_context_tokens

  (( CLIENT_CONTEXT_TOKENS + CLIENT_MAX_OUTPUT_TOKENS <= CTX_SIZE )) ||
    die "Client input + output budgets exceed server context."
}

parse_common_options() {
  REMAINING_ARGS=()

  while (( $# )); do
    case "$1" in
      --context)
        [[ $# -ge 2 ]] || die "--context requires a value"
        CTX_SIZE="$2"
        shift 2
        ;;
      --context=*)
        CTX_SIZE="${1#*=}"
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
  export CTX_SIZE CLIENT_CONTEXT_TOKENS CLIENT_MAX_OUTPUT_TOKENS
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
  echo "Server context:      ${CTX_SIZE}"
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

api_auth_args() {
  API_AUTH_ARGS=()

  if [[ -n "$API_KEY" ]]; then
    API_AUTH_ARGS=(-H "Authorization: Bearer ${API_KEY}")
  fi
}

server_running() {
  tmux has-session -t "$SESSION_NAME" 2>/dev/null
}

port_in_use() {
  timeout 1 bash -c \
    "</dev/tcp/127.0.0.1/${API_PORT}" 2>/dev/null
}

wait_for_api() {
  local deadline=$((SECONDS + API_WAIT_SECONDS))

  log "Waiting for llama.cpp API"

  while (( SECONDS < deadline )); do
    if curl -fsS \
      --max-time 5 \
      "http://127.0.0.1:${API_PORT}/health" \
      >/dev/null 2>&1; then
      return 0
    fi

    if ! server_running; then
      tail -n 350 "$LOG_FILE" 2>/dev/null || true
      die "llama-server stopped before the API became ready."
    fi

    sleep 5
  done

  tail -n 350 "$LOG_FILE" 2>/dev/null || true
  die "API did not become ready within ${API_WAIT_SECONDS} seconds."
}

quick_file_checks() {
  [[ -f "$MODEL_PATH" ]] ||
    die "Model file is missing: $MODEL_PATH"

  [[ -f "$MMPROJ_PATH" ]] ||
    die "Multimodal projector is missing: $MMPROJ_PATH"

  local model_size
  local mmproj_size

  model_size="$(stat -c '%s' "$MODEL_PATH")"
  mmproj_size="$(stat -c '%s' "$MMPROJ_PATH")"

  [[ "$model_size" == "$MODEL_SIZE_BYTES" ]] ||
    die "Model size mismatch: expected ${MODEL_SIZE_BYTES}, got ${model_size} bytes."

  [[ "$mmproj_size" == "$MMPROJ_SIZE_BYTES" ]] ||
    die "mmproj size mismatch: expected ${MMPROJ_SIZE_BYTES}, got ${mmproj_size} bytes."
}

prepare_media_file() {
  local source_path="$1"
  local clean_name
  local destination

  [[ -f "$source_path" ]] ||
    die "Media file does not exist: $source_path"

  mkdir -p "$MEDIA_DIR"

  clean_name="$(
    basename "$source_path" |
    tr -cs 'A-Za-z0-9._-' '_'
  )"

  destination="${MEDIA_DIR}/${clean_name}"

  if [[ "$(readlink -f "$source_path")" != \
        "$(readlink -f "$destination" 2>/dev/null || true)" ]]; then
    cp -f "$source_path" "$destination"
  fi

  chmod 0644 "$destination"

  # llama-server resolves local file:// paths relative to --media-path.
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
    git \
    clang \
    cmake \
    ninja-build \
    libcurl4-openssl-dev \
    libssl-dev \
    ffmpeg
}

build_runtime() {
  require_command git
  require_command cmake
  require_command nvidia-smi

  install_build_dependencies

  mkdir -p "$(dirname "$LLAMA_CPP_DIR")" "$BASE_DIR"

  if [[ ! -d "${LLAMA_CPP_DIR}/.git" ]]; then
    log "Cloning llama.cpp"
    git clone "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR"
  fi

  log "Updating llama.cpp"
  git -C "$LLAMA_CPP_DIR" fetch --all --tags --prune
  git -C "$LLAMA_CPP_DIR" checkout "$LLAMA_CPP_REF"

  if [[ "$LLAMA_CPP_REF" == "master" || "$LLAMA_CPP_REF" == "main" ]]; then
    git -C "$LLAMA_CPP_DIR" pull --ff-only
  fi

  local commit
  commit="$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)"
  printf '%s\n' "$commit" > "$RUNTIME_COMMIT_FILE"

  log "Configuring CUDA build for GB10 sm_121a"

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

  log "Building llama-server"

  cmake \
    --build "${LLAMA_CPP_DIR}/build" \
    --config Release \
    --target llama-server \
    -j "$(nproc)"

  [[ -x "$LLAMA_SERVER" ]] ||
    die "Build completed but llama-server was not found: $LLAMA_SERVER"

  log "Runtime build completed"
  "$LLAMA_SERVER" --version || true
  echo "Commit: $commit"
}

runtime_info() {
  [[ -x "$LLAMA_SERVER" ]] ||
    die "llama-server is not built. Run: $0 build-runtime"

  echo "Binary: $LLAMA_SERVER"
  "$LLAMA_SERVER" --version || true

  if [[ -f "$RUNTIME_COMMIT_FILE" ]]; then
    echo "Recorded commit: $(cat "$RUNTIME_COMMIT_FILE")"
  fi

  if [[ -d "${LLAMA_CPP_DIR}/.git" ]]; then
    echo "Current commit:  $(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)"
    git -C "$LLAMA_CPP_DIR" status --short --branch
  fi
}

# ==============================================================================
# DOWNLOAD / VERIFY
# ==============================================================================

download_file() {
  local url="$1"
  local destination="$2"
  local expected_size="$3"
  local expected_sha256="$4"
  local current_size="0"

  if [[ -f "$destination" ]]; then
    current_size="$(stat -c '%s' "$destination")"

    if [[ "$current_size" == "$expected_size" ]]; then
      log "File already has the exact remote size: $(basename "$destination")"
      log "Verifying existing SHA-256 instead of issuing an EOF range request"

      if printf '%s  %s\n' "$expected_sha256" "$destination" |
         sha256sum --check -; then
        log "Existing file is complete and verified"
        return
      fi

      warn "Existing full-size file has the wrong SHA-256."
      mv -f "$destination" "${destination}.corrupt.$(date +%s)"
      current_size="0"
    elif (( current_size > expected_size )); then
      warn "Existing file is larger than expected; preserving it as corrupt."
      mv -f "$destination" "${destination}.corrupt.$(date +%s)"
      current_size="0"
    else
      log "Resuming $(basename "$destination") from byte ${current_size}"
    fi
  fi

  if [[ ! -f "$destination" ]]; then
    log "Downloading from byte 0: $(basename "$destination")"
  fi

  curl \
    --fail \
    --location \
    --retry 10 \
    --retry-delay 5 \
    --retry-all-errors \
    --continue-at - \
    --output "$destination" \
    "$url"

  current_size="$(stat -c '%s' "$destination")"
  [[ "$current_size" == "$expected_size" ]] ||
    die "Downloaded size mismatch for $(basename "$destination"): expected ${expected_size}, got ${current_size}."

  printf '%s  %s\n' "$expected_sha256" "$destination" |
    sha256sum --check -
}

download_model() {
  require_command curl
  require_command sha256sum
  require_command df

  mkdir -p "$MODEL_DIR"

  log "Disk space before download"
  df -h "$MODEL_DIR"

  log "Expected model bytes:  ${MODEL_SIZE_BYTES}"
  log "Expected mmproj bytes: ${MMPROJ_SIZE_BYTES}"
  log "Model directory: $MODEL_DIR"

  download_file \
    "$MODEL_URL" \
    "$MODEL_PATH" \
    "$MODEL_SIZE_BYTES" \
    "$MODEL_SHA256"

  download_file \
    "$MMPROJ_URL" \
    "$MMPROJ_PATH" \
    "$MMPROJ_SIZE_BYTES" \
    "$MMPROJ_SHA256"

  verify_files

  log "Download completed"
  du -sh "$MODEL_DIR"
}

verify_files() {
  require_command sha256sum

  quick_file_checks

  log "Verifying Q6_K_P SHA-256"
  printf '%s  %s\n' "$MODEL_SHA256" "$MODEL_PATH" |
    sha256sum --check -

  log "Verifying mmproj SHA-256"
  printf '%s  %s\n' "$MMPROJ_SHA256" "$MMPROJ_PATH" |
    sha256sum --check -

  log "Both GGUF files passed SHA-256 verification"
}

# ==============================================================================
# START / STOP
# ==============================================================================

serve_foreground() {
  local -a args
  local -a auth_args
  local thinking_json

  [[ -x "$LLAMA_SERVER" ]] ||
    die "llama-server is not built. Run: $0 build-runtime"

  quick_file_checks
  mkdir -p "$MEDIA_DIR" "$LOG_DIR"

  thinking_json="$(json_bool "$DEFAULT_ENABLE_THINKING")"

  args=(
    --model "$MODEL_PATH"
    --mmproj "$MMPROJ_PATH"
    --alias "$SERVED_MODEL_NAME"
    --host "$API_HOST"
    --port "$API_PORT"
    --ctx-size "$CTX_SIZE"
    --parallel "$N_PARALLEL"
    --n-gpu-layers "$GPU_LAYERS"
    --flash-attn "$FLASH_ATTN"
    --cache-type-k "$CACHE_TYPE_K"
    --cache-type-v "$CACHE_TYPE_V"
    --batch-size "$BATCH_SIZE"
    --ubatch-size "$UBATCH_SIZE"
    --cont-batching
    --cache-prompt
    --jinja
    --reasoning auto
    --reasoning-format deepseek
    --chat-template-kwargs
      "{\"enable_thinking\":${thinking_json}}"
    --media-path "$MEDIA_DIR"
    --metrics
    --timeout 3600
    --log-timestamps
  )

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

  if server_running; then
    log "Stopping tmux session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME"
  else
    log "Server session is already stopped"
  fi
}

start_server() {
  local tmux_command

  require_command tmux
  require_command curl
  require_command timeout
  require_command nvidia-smi

  nvidia-smi >/dev/null 2>&1 ||
    die "nvidia-smi failed on the host."

  [[ -x "$LLAMA_SERVER" ]] ||
    die "llama-server is not built. Run: $0 build-runtime"

  quick_file_checks

  if [[ "$VERIFY_SHA_ON_START" == "1" ]]; then
    verify_files
  fi

  mkdir -p "$LOG_DIR" "$MEDIA_DIR"

  if server_running; then
    log "Stopping previous controller-owned server"
    tmux kill-session -t "$SESSION_NAME"
    sleep 2
  fi

  if port_in_use; then
    die "Port ${API_PORT} is already in use. Stop the previous model or change API_PORT."
  fi

  : > "$LOG_FILE"

  printf -v tmux_command \
    'exec bash %q _serve >> %q 2>&1' \
    "$SCRIPT_PATH" \
    "$LOG_FILE"

  log "Starting Qwen3.6 Q6_K_P with locally built llama.cpp"
  log "Model:       $MODEL_FILE"
  log "mmproj:      $MMPROJ_FILE"
  log "Context:     $CTX_SIZE"
  log "Thinking:    $DEFAULT_ENABLE_THINKING"
  log "API:         $(public_base_url)"

  tmux new-session \
    -d \
    -s "$SESSION_NAME" \
    "$tmux_command"

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
# STATUS / PROPERTIES / LOGS
# ==============================================================================

show_props() {
  require_command curl
  api_auth_args

  curl -sS \
    "${API_AUTH_ARGS[@]}" \
    "http://127.0.0.1:${API_PORT}/props"

  echo
}

show_status() {
  require_command curl
  require_command tmux

  echo "===== CONFIG ====="
  echo "Repository:       $REPO_ID"
  echo "Quant:            Q6_K_P"
  echo "Served name:      $SERVED_MODEL_NAME"
  echo "Runtime:          $LLAMA_SERVER"
  echo "Context:          $CTX_SIZE"
  echo "KV cache:         K=$CACHE_TYPE_K V=$CACHE_TYPE_V"
  echo "Default thinking: $DEFAULT_ENABLE_THINKING"
  echo "API port:         $API_PORT"
  echo

  echo "===== RUNTIME ====="
  if [[ -x "$LLAMA_SERVER" ]]; then
    "$LLAMA_SERVER" --version || true
    if [[ -f "$RUNTIME_COMMIT_FILE" ]]; then
      echo "Commit: $(cat "$RUNTIME_COMMIT_FILE")"
    fi
  else
    echo "Not built"
  fi

  echo
  echo "===== SESSION ====="
  if server_running; then
    tmux list-sessions |
      grep "^${SESSION_NAME}:" || true
  else
    echo "STOPPED"
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
  local lines="${1:-350}"

  [[ -f "$LOG_FILE" ]] ||
    die "Log file does not exist yet: $LOG_FILE"

  tail -n "$lines" "$LOG_FILE"
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
      "content": "You are a precise software engineering assistant."
    },
    {
      "role": "user",
      "content": "Write a Python function with type hints that validates an IPv4 address. Include pytest tests."
    }
  ],
  "chat_template_kwargs": {
    "enable_thinking": false
  },
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "min_p": 0,
  "presence_penalty": 1.5,
  "repeat_penalty": 1.0,
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
      "content": "Design a resilient local coding-agent deployment. Compare failure modes, security boundaries, and recovery steps."
    }
  ],
  "chat_template_kwargs": {
    "enable_thinking": true
  },
  "reasoning_format": "deepseek",
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 20,
  "min_p": 0,
  "presence_penalty": 1.5,
  "repeat_penalty": 1.0,
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
  "parse_tool_calls": true,
  "parallel_tool_calls": false,
  "chat_template_kwargs": {
    "enable_thinking": false
  },
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20,
  "min_p": 0,
  "presence_penalty": 0,
  "repeat_penalty": 1.0,
  "max_tokens": 2048
}
EOF

  echo
  echo
  echo "Expected: choices[0].message.tool_calls"
  echo "The IDE/agent must execute the tool and return a role=tool message."
}

test_image() {
  local source_path="${1:-}"
  local media_name
  local payload

  [[ -n "$source_path" ]] ||
    die "Usage: $0 test-image /path/to/image.png"

  require_command curl
  require_command python3

  media_name="$(prepare_media_file "$source_path")"
  api_auth_args

  payload="$(
    python3 - "$SERVED_MODEL_NAME" "$media_name" <<'PY'
import json
import sys

model, filename = sys.argv[1], sys.argv[2]

print(json.dumps({
    "model": model,
    "messages": [{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"file://{filename}"}
            },
            {
                "type": "text",
                "text": "Describe this image accurately and identify visible text."
            }
        ]
    }],
    "chat_template_kwargs": {"enable_thinking": False},
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0,
    "presence_penalty": 1.5,
    "repeat_penalty": 1.0,
    "max_tokens": 2048
}))
PY
  )"

  curl -sS \
    "${API_AUTH_ARGS[@]}" \
    "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data-binary "$payload"

  echo
}

test_video() {
  local source_path="${1:-}"
  local media_name
  local payload

  [[ -n "$source_path" ]] ||
    die "Usage: $0 test-video /path/to/video.mp4"

  require_command curl
  require_command python3

  media_name="$(prepare_media_file "$source_path")"
  api_auth_args

  payload="$(
    python3 - "$SERVED_MODEL_NAME" "$media_name" <<'PY'
import json
import sys

model, filename = sys.argv[1], sys.argv[2]

print(json.dumps({
    "model": model,
    "messages": [{
        "role": "user",
        "content": [
            {
                "type": "input_video",
                "input_video": {"url": f"file://{filename}"}
            },
            {
                "type": "text",
                "text": "Summarize this video and list key events chronologically."
            }
        ]
    }],
    "chat_template_kwargs": {"enable_thinking": False},
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0,
    "presence_penalty": 1.5,
    "repeat_penalty": 1.0,
    "max_tokens": 4096
}))
PY
  )"

  curl -sS \
    "${API_AUTH_ARGS[@]}" \
    "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data-binary "$payload"

  echo
}

# ==============================================================================
# CLIENT CONFIG
# ==============================================================================

print_client_config() {
  local client_key="${API_KEY:-llama-local}"

  cat <<EOF
==============================================================================
OpenAI-compatible endpoint
==============================================================================

Provider:        OpenAI Compatible
Base URL:        $(public_base_url)
API key:         ${client_key}
Model ID:        ${SERVED_MODEL_NAME}
Server context:  ${CTX_SIZE}
Input budget:    ${CLIENT_CONTEXT_TOKENS}
Max output:      ${CLIENT_MAX_OUTPUT_TOKENS}

export OPENAI_BASE_URL="$(public_base_url)"
export OPENAI_API_KEY="${client_key}"
export OPENAI_MODEL="${SERVED_MODEL_NAME}"

==============================================================================
Recommended VS Code / Agent settings
==============================================================================

Supports tools:      yes, after test-tools validation
Supports reasoning:  yes
Supports images:     yes
Supports video:      yes
Supports audio:      no
Parallel tools:      false
Thinking default:    ${DEFAULT_ENABLE_THINKING}

Coding/tool requests:
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  presence_penalty: 0
  repeat_penalty: 1.0
  chat_template_kwargs.enable_thinking: false

General thinking:
  temperature: 1.0
  top_p: 0.95
  top_k: 20
  presence_penalty: 1.5
  repeat_penalty: 1.0
  chat_template_kwargs.enable_thinking: true

Important:
- llama-server returns structured tool_calls.
- The IDE/agent executes filesystem, shell, Git, browser, and network tools.
- Validate arguments and use a workspace sandbox.
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

  $(basename "$0") build-runtime
  $(basename "$0") runtime-info
  $(basename "$0") download
  $(basename "$0") verify-files
  $(basename "$0") start
  $(basename "$0") stop
  $(basename "$0") restart
  $(basename "$0") status
  $(basename "$0") logs [number_of_lines]
  $(basename "$0") props
  $(basename "$0") test-text
  $(basename "$0") test-reasoning
  $(basename "$0") test-tools [required|auto]
  $(basename "$0") test-image /path/to/image
  $(basename "$0") test-video /path/to/video
  $(basename "$0") client-config

First setup:
  1. Edit USER CONFIGURATION near the top.
  2. Run: ./$(basename "$0") build-runtime
  3. Run: ./$(basename "$0") download
  4. Stop another model using GPU/port ${API_PORT}.
  5. Run: ./$(basename "$0") start
  6. Run: ./$(basename "$0") props
  7. Run: ./$(basename "$0") test-tools required

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
  build-runtime)
    build_runtime
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
  props)
    show_props
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
  test-image)
    test_image "${2:-}"
    ;;
  test-video)
    test_video "${2:-}"
    ;;
  client-config)
    print_client_config
    ;;
  _serve)
    serve_foreground
    ;;
  info|banner)
    info
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
