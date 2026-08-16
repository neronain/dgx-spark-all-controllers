#!/usr/bin/env bash
# Qwen3-VL-32B-Thinking — single DGX Spark controller v5 (llama.cpp)
# Runtime: llama.cpp built from source, SM121a-real (GB10 Blackwell)
# Vision: mmproj-BF16.gguf required for image/video input
# Thinking: embedded Jinja2 chat template (--jinja --reasoning-format deepseek)
set -Eeuo pipefail

SCRIPT_VERSION="${SCRIPT_VERSION:-3.1.0}"
MODEL_LABEL="${MODEL_LABEL:-Qwen3-VL-32B-Thinking (GGUF)}"
RUNTIME_LABEL="${RUNTIME_LABEL:-llama.cpp}"
MODEL_FEATURES="${MODEL_FEATURES:-vision · thinking · tools}"

# ─── Model ────────────────────────────────────────────────────────────────────
HF_REPO="${HF_REPO:-unsloth/Qwen3-VL-32B-Thinking-GGUF}"
MODEL_REVISION="${MODEL_REVISION:-main}"

# Quantization variant — change to any available quant (see download --list)
MODEL_QUANT="${MODEL_QUANT:-Q4_K_M}"
MODEL_FILE="${MODEL_FILE:-Qwen3-VL-32B-Thinking-${MODEL_QUANT}.gguf}"

# mmproj (vision encoder projector) — required for image/video inputs
MMPROJ_FILE="${MMPROJ_FILE:-mmproj-BF16.gguf}"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-32b-thinking}"

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
MODEL_DIR="${MODEL_DIR:-${USER_HOME}/models/qwen3-vl-32b-thinking}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${USER_HOME}/src/llama.cpp}"

# ─── Serving ──────────────────────────────────────────────────────────────────
API_PORT="${API_PORT:-8000}"
API_HOST="${API_HOST:-0.0.0.0}"
ADVERTISE_IP="${ADVERTISE_IP:-}"
ADVERTISE_INTERFACE="${ADVERTISE_INTERFACE:-}"
ROUTE_PROBE_IP="${ROUTE_PROBE_IP:-1.1.1.1}"

CTX_SIZE="${CTX_SIZE:-131072}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-2048}"
N_THREADS="${N_THREADS:-20}"
PARALLEL_SLOTS="${PARALLEL_SLOTS:-1}"
FLASH_ATTN="${FLASH_ATTN:-on}"
SPLIT_MODE="${SPLIT_MODE:-none}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"

# ─── Approximate file sizes for verification (bytes; update after first download)
MODEL_SIZE_MIN_BYTES="18000000000"   # Q4_K_M ≈ 19.8 GB; floor at 18 GB to allow any quant
MMPROJ_SIZE_MIN_BYTES="1000000000"  # mmproj-BF16 ≈ 1.2 GB; floor at 1 GB

# Exact SHA-256 values — record after first verified download
MODEL_SHA256="${MODEL_SHA256:-}"
MMPROJ_SHA256="${MMPROJ_SHA256:-}"

# ─── Server PID file ──────────────────────────────────────────────────────────
PID_FILE="/tmp/llama-qwen3vl-32b.pid"
LOG_FILE="${LOG_FILE:-${MODEL_DIR}/llama-server.log}"

# ─── Helpers ──────────────────────────────────────────────────────────────────
die()  { echo "ERROR: $*" >&2; exit 1; }
log()  { echo "[$(date '+%H:%M:%S')] $*"; }

model_size_min_bytes() {
  case "$MODEL_QUANT" in
    Q2_K)       echo 11000000000 ;;
    Q3_K_M)     echo 14000000000 ;;
    Q4_K_S)     echo 17000000000 ;;
    Q4_K_M)     echo 18000000000 ;;
    UD-Q4_K_XL) echo 18000000000 ;;
    Q5_K_M)     echo 21000000000 ;;
    Q6_K)       echo 25000000000 ;;
    Q8_0)       echo 33000000000 ;;
    *)          echo "$MODEL_SIZE_MIN_BYTES" ;;
  esac
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
      --quant)          MODEL_QUANT="$2"; MODEL_FILE="Qwen3-VL-32B-Thinking-${MODEL_QUANT}.gguf"; shift 2 ;;
      --quant=*)        MODEL_QUANT="${1#*=}"; MODEL_FILE="Qwen3-VL-32B-Thinking-${MODEL_QUANT}.gguf"; shift ;;
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
      *)                REMAINING_ARGS+=("$1");     shift ;;
    esac
  done
  case "$FLASH_ATTN" in on|off|auto) ;; *) die "Invalid --flash-attn value: ${FLASH_ATTN} (use on, off, or auto)" ;; esac
  case "$SPLIT_MODE" in none|layer|row|tensor) ;; *) die "Invalid --split-mode value: ${SPLIT_MODE} (use none, layer, row, or tensor)" ;; esac
  case "$CACHE_TYPE_K" in f32|f16|bf16|q8_0|q4_0|q4_1|iq4_nl|q5_0|q5_1) ;; *) die "Invalid --cache-k value: ${CACHE_TYPE_K}" ;; esac
  case "$CACHE_TYPE_V" in f32|f16|bf16|q8_0|q4_0|q4_1|iq4_nl|q5_0|q5_1) ;; *) die "Invalid --cache-v value: ${CACHE_TYPE_V}" ;; esac
  [[ "$CTX_SIZE" =~ ^[0-9]+$ ]] && (( CTX_SIZE >= 4096 && CTX_SIZE <= 262144 )) || die "Invalid --context: ${CTX_SIZE} (use 4096..262144 for this model)"
  [[ "$API_PORT" =~ ^[0-9]+$ ]] && (( API_PORT >= 1 && API_PORT <= 65535 )) || die "Invalid --port: ${API_PORT} (use 1..65535)"
  export CTX_SIZE API_HOST API_PORT ADVERTISE_IP ADVERTISE_INTERFACE MODEL_QUANT MODEL_FILE PARALLEL_SLOTS FLASH_ATTN SPLIT_MODE CACHE_TYPE_K CACHE_TYPE_V
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
      echo "Available quantizations for ${HF_REPO}:"
      echo ""
      echo "  Quant        Size    Quality"
      echo "  Q2_K         12.3 GB  Low — testing only"
      echo "  Q3_K_M       16.0 GB  Low"
      echo "  Q4_K_S       18.8 GB  Balanced"
      echo "  Q4_K_M       19.8 GB  Balanced (DEFAULT)"
      echo "  UD-Q4_K_XL   20.1 GB  Balanced+ (Unsloth Ultra Discrete)"
      echo "  Q5_K_M       23.2 GB  Good"
      echo "  Q6_K         26.9 GB  Very good"
      echo "  Q8_0         34.8 GB  Near-lossless"
      echo ""
      echo "  mmproj options:"
      echo "  mmproj-BF16.gguf  1.2 GB  DEFAULT — best vision quality"
      echo "  mmproj-F16.gguf   1.2 GB  Same size, float16"
      echo "  mmproj-F32.gguf   2.38 GB Full precision"
      echo ""
      echo "Memory guide (all fit in 128 GB DGX Spark):"
      echo "  Q4_K_M + mmproj-BF16 + 128K ctx + q8 KV ≈ 40-50 GB"
      echo "  Q8_0   + mmproj-BF16 + 128K ctx + q8 KV ≈ 55-65 GB"
      return
      ;;
    --force)
      force_download=1
      ;;
    "")
      ;;
    *)
      die "Unknown download option '${subcmd}'. Use: download, download --force, or download --list"
      ;;
  esac

  mkdir -p "$MODEL_DIR" "$HF_HOME"

  local model_path="${MODEL_DIR}/${MODEL_FILE}"
  local mmproj_path="${MODEL_DIR}/${MMPROJ_FILE}"
  local min_model
  min_model=$(model_size_min_bytes)

  gguf_file_ok() {
    local path="$1"
    local min_bytes="$2"
    [[ -f "$path" ]] || return 1

    local size magic
    size=$(stat -Lc '%s' "$path" 2>/dev/null || stat -f '%z' "$path" 2>/dev/null) || return 1
    (( size >= min_bytes )) || return 1

    magic=$(LC_ALL=C head -c 4 "$path" | od -A n -t x1 | tr -d ' \n')
    [[ "$magic" == "47475546" ]]
  }

  if (( force_download == 0 )) \
      && gguf_file_ok "$model_path" "$min_model" \
      && gguf_file_ok "$mmproj_path" "$MMPROJ_SIZE_MIN_BYTES"; then
    log "Model and mmproj already exist and pass size/header checks."
    log "Nothing to download. Use 'download --force' only when you want a fresh copy."
    return
  fi

  log "Downloading ${MODEL_FILE} and ${MMPROJ_FILE} from ${HF_REPO} …"

  # Use an already available downloader first. No shell activation is required.
  if command -v hf >/dev/null 2>&1; then
    log "Using Hugging Face CLI: $(command -v hf)"
    local hf_args=(download "$HF_REPO" "$MODEL_FILE" "$MMPROJ_FILE"
      --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR")
    (( force_download == 0 )) || hf_args+=(--force-download)
    HF_HOME="$HF_HOME" HF_TOKEN="${HF_TOKEN:-}" hf "${hf_args[@]}"
    log "Download complete: ${MODEL_DIR}/"
    return
  fi

  if command -v huggingface-cli >/dev/null 2>&1; then
    log "Using legacy Hugging Face CLI: $(command -v huggingface-cli)"
    local legacy_args=(download "$HF_REPO" "$MODEL_FILE" "$MMPROJ_FILE"
      --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR")
    (( force_download == 0 )) || legacy_args+=(--force-download)
    HF_HOME="$HF_HOME" HF_TOKEN="${HF_TOKEN:-}" huggingface-cli "${legacy_args[@]}"
    log "Download complete: ${MODEL_DIR}/"
    return
  fi

  local hf_python=""
  if python3 -c 'import huggingface_hub' >/dev/null 2>&1; then
    hf_python=$(command -v python3)
  elif [[ -x "${HF_VENV}/bin/python" ]] \
      && "${HF_VENV}/bin/python" -c 'import huggingface_hub' >/dev/null 2>&1; then
    hf_python="${HF_VENV}/bin/python"
  fi

  # Bootstrap a private downloader venv only if no downloader exists.
  if [[ -z "$hf_python" ]]; then
    [[ "$AUTO_INSTALL_HF" == "1" ]] || die "Hugging Face downloader not found and AUTO_INSTALL_HF=${AUTO_INSTALL_HF}. Set AUTO_INSTALL_HF=1 or install huggingface_hub."

    log "Hugging Face downloader not found; preparing private venv: ${HF_VENV}"
    if [[ ! -x "${HF_VENV}/bin/python" ]]; then
      if ! python3 -m venv "$HF_VENV"; then
        die "Could not create ${HF_VENV}. Install venv support once, then rerun:\n  sudo apt-get update && sudo apt-get install -y python3-venv"
      fi
    fi

    "${HF_VENV}/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
    "${HF_VENV}/bin/python" -m pip install --disable-pip-version-check --upgrade pip huggingface_hub
    hf_python="${HF_VENV}/bin/python"
  fi

  log "Using Python downloader: ${hf_python}"
  HF_HOME="$HF_HOME" \
  HF_TOKEN="${HF_TOKEN:-}" \
  HF_REPO="$HF_REPO" \
  MODEL_REVISION="$MODEL_REVISION" \
  MODEL_DIR="$MODEL_DIR" \
  MODEL_FILE="$MODEL_FILE" \
  MMPROJ_FILE="$MMPROJ_FILE" \
  FORCE_DOWNLOAD="$force_download" \
  "$hf_python" - <<'PYEOF'
from huggingface_hub import hf_hub_download
import os

repo = os.environ["HF_REPO"]
rev = os.environ["MODEL_REVISION"]
dest = os.environ["MODEL_DIR"]
token = os.environ.get("HF_TOKEN") or None
force = os.environ.get("FORCE_DOWNLOAD") == "1"

for filename in (os.environ["MODEL_FILE"], os.environ["MMPROJ_FILE"]):
    print(f"Downloading/checking {filename} ...", flush=True)
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        revision=rev,
        local_dir=dest,
        token=token,
        force_download=force,
    )
    print(f"Ready: {path}", flush=True)
PYEOF

  log "Download complete: ${MODEL_DIR}/"
}

# ─── verify-files ─────────────────────────────────────────────────────────────
verify_files() {
  log "Verifying model and mmproj files …"
  local model_path="${MODEL_DIR}/${MODEL_FILE}"
  local mmproj_path="${MODEL_DIR}/${MMPROJ_FILE}"

  # Model GGUF
  [[ -f "$model_path" ]] || die "Model file not found: ${model_path}"
  local msz min_size; msz=$(stat -Lc '%s' "$model_path" 2>/dev/null || stat -f '%z' "$model_path")
  min_size=$(model_size_min_bytes)
  if (( msz < min_size )); then
    die "Model file too small: ${msz} bytes (expected >${min_size} for ${MODEL_QUANT}). Incomplete download?"
  fi

  # GGUF magic header check
  local magic; magic=$(LC_ALL=C head -c 4 "$model_path" | od -A n -t x1 | tr -d ' \n')
  if [[ "$magic" != "47475546" ]]; then
    die "Not a GGUF file (bad magic: ${magic}): ${model_path}"
  fi

  # mmproj GGUF
  [[ -f "$mmproj_path" ]] || die "mmproj file not found: ${mmproj_path}"
  local psz; psz=$(stat -Lc '%s' "$mmproj_path" 2>/dev/null || stat -f '%z' "$mmproj_path")
  if (( psz < MMPROJ_SIZE_MIN_BYTES )); then
    die "mmproj file too small: ${psz} bytes. Incomplete download?"
  fi
  local pmagic; pmagic=$(LC_ALL=C head -c 4 "$mmproj_path" | od -A n -t x1 | tr -d ' \n')
  if [[ "$pmagic" != "47475546" ]]; then
    die "mmproj is not a GGUF file (bad magic: ${pmagic}): ${mmproj_path}"
  fi

  # SHA-256 check (if recorded)
  if [[ -n "$MODEL_SHA256" ]]; then
    log "Verifying model SHA-256 …"
    echo "${MODEL_SHA256}  ${model_path}" | sha256sum --check --quiet \
      || die "Model SHA-256 mismatch"
    log "Model SHA-256 OK"
  else
    log "MODEL_SHA256 not set — record after first download:"
    echo "  sha256sum '${model_path}'"
  fi

  if [[ -n "$MMPROJ_SHA256" ]]; then
    log "Verifying mmproj SHA-256 …"
    echo "${MMPROJ_SHA256}  ${mmproj_path}" | sha256sum --check --quiet \
      || die "mmproj SHA-256 mismatch"
    log "mmproj SHA-256 OK"
  else
    log "MMPROJ_SHA256 not set — record after first download:"
    echo "  sha256sum '${mmproj_path}'"
  fi

  log "verify-files: PASS (${MODEL_QUANT}, ${msz} bytes; mmproj ${psz} bytes)"
}

# ─── start ────────────────────────────────────────────────────────────────────
start() {
  local srv="${LLAMA_CPP_DIR}/build/bin/llama-server"
  [[ -x "$srv" ]] || die "llama-server not found. Run: $0 prepare-runtime"

  local model_path="${MODEL_DIR}/${MODEL_FILE}"
  local mmproj_path="${MODEL_DIR}/${MMPROJ_FILE}"
  [[ -f "$model_path" ]]  || die "Model not found: ${model_path}. Run: $0 download"
  [[ -f "$mmproj_path" ]] || die "mmproj not found: ${mmproj_path}. Run: $0 download"

  # Port conflict check (no grep -q with pipefail)
  local ss_out; ss_out=$(ss -tlnp 2>/dev/null || true)
  if [[ "$ss_out" == *":${API_PORT} "* ]]; then
    die "Port ${API_PORT} already in use. Stop conflicting service first."
  fi

  # Stale PID cleanup
  if [[ -f "$PID_FILE" ]]; then
    local old_pid; old_pid=$(cat "$PID_FILE")
    if kill -0 "$old_pid" 2>/dev/null; then
      die "llama-server already running (PID ${old_pid}). Run: $0 stop"
    fi
    rm -f "$PID_FILE"
  fi

  log "Starting llama-server …"
  log "  Model  : ${model_path}"
  log "  mmproj : ${mmproj_path}"
  log "  Quant  : ${MODEL_QUANT}"
  log "  Context: ${CTX_SIZE} tokens"
  log "  Split  : ${SPLIT_MODE}"
  log "  KV     : K=${CACHE_TYPE_K}, V=${CACHE_TYPE_V}"
  log "  Port   : ${API_PORT}"

  mkdir -p "$MODEL_DIR"
  nohup "$srv" \
    -m "$model_path" \
    --mmproj "$mmproj_path" \
    -ngl 999 \
    --ctx-size "$CTX_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --ubatch-size "$UBATCH_SIZE" \
    --flash-attn "$FLASH_ATTN" \
    --no-mmap \
    --cache-type-k "$CACHE_TYPE_K" \
    --cache-type-v "$CACHE_TYPE_V" \
    --fit off \
    --threads "$N_THREADS" \
    --parallel "$PARALLEL_SLOTS" \
    --jinja \
    --reasoning-format deepseek \
    --split-mode "$SPLIT_MODE" \
    --main-gpu 0 \
    --temp 1.0 \
    --top-k 20 \
    --top-p 0.95 \
    --min-p 0.0 \
    --host "$API_HOST" \
    --port "$API_PORT" \
    --alias "$SERVED_MODEL_NAME" \
    > "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  log "Server PID: $(cat "$PID_FILE")"

  log "Waiting for /health (timeout 120 s) …"
  local deadline=$(( $(date +%s) + 120 ))
  while (( $(date +%s) < deadline )); do
    local server_pid; server_pid=$(cat "$PID_FILE")
    if ! kill -0 "$server_pid" 2>/dev/null; then
      rm -f "$PID_FILE"
      echo "" >&2
      echo "llama-server exited during startup. Last 80 log lines:" >&2
      tail -n 80 "$LOG_FILE" >&2 || true
      die "Server exited before becoming healthy"
    fi
    if curl -fsS "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
      local adv; adv=$(detect_advertise_ip)
      echo ""
      echo "  Model   : ${SERVED_MODEL_NAME} (${MODEL_QUANT})"
      echo "  Vision  : ${MMPROJ_FILE}"
      echo "  API     : http://${adv}:${API_PORT}/v1"
      echo "  Context : ${CTX_SIZE} tokens"
      return
    fi
    sleep 3
  done
  die "Server did not start within 120 s. Check logs: $0 logs"
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
  echo "  Model  : ${HF_REPO}  quant=${MODEL_QUANT}"
  echo "  mmproj : ${MMPROJ_FILE}"
  echo "  Context: ${CTX_SIZE}"
  echo "  KV     : K=${CACHE_TYPE_K}, V=${CACHE_TYPE_V}"
  echo "  Port   : ${API_PORT}"

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
  curl -fsS "http://127.0.0.1:${API_PORT}/v1/models" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "  (server not ready)"
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
# OpenAI Python SDK — Qwen3-VL-32B-Thinking (single DGX Spark)
from openai import OpenAI
import base64, pathlib

client = OpenAI(base_url="http://${adv}:${API_PORT}/v1", api_key="none")

# ── Text with thinking ────────────────────────────────────────────────────────
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
  curl -fsS "http://127.0.0.1:${API_PORT}/v1/models" | python3 -m json.tool
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
    "temperature": 1.0, "top_p": 0.95,
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
print(f"Expected   : ~40-55 tok/s (Q4_K_M on DGX Spark)")
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
        "max_tokens": 64, "temperature": 1.0,
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
    "temperature": 1.0, "top_p": 0.95,
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
  log "test-reasoning (expect 37×43=1591 in output) …"
  python3 - <<PYEOF
import urllib.request, json, sys

url     = "http://127.0.0.1:${API_PORT}/v1/chat/completions"
model   = "${SERVED_MODEL_NAME}"
payload = json.dumps({
    "model": model,
    "messages": [{"role":"user","content":"What is 37 multiplied by 43? Show your work."}],
    "max_tokens": 2048,
    "temperature": 1.0, "top_p": 0.95,
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
    "temperature": 1.0, "top_p": 0.95,
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
        "temperature": 1.0,
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
  log "test-tools ${mode} (Qwen3-VL tool calling — unverified on hardware) …"
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
    "temperature": 1.0,
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
    print("WARN: Tool calling support unverified for VL-Thinking variant")
    sys.exit(1)
else:
    print(f"WARN(auto): text response: {d['choices'][0]['message'].get('content','')[:100]}")
PYEOF
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
  help|--help|-h)
    cat <<HELP
Qwen3-VL-32B-Thinking single-node controller (llama.cpp)

Usage: $0 [OPTIONS] COMMAND [args]

Options:
  --context TOKENS    Override ctx size (default: ${CTX_SIZE})
  --port PORT         Override API port (default: ${API_PORT})
  --bind ADDR         Override bind address (default: ${API_HOST})
  --advertise-ip IP   Override public IP for client-config / network-info
  --interface NAME    Interface for advertise-ip autodetection
  --quant QUANT       Override quantization (default: ${MODEL_QUANT})
  --parallel N        Parallel request slots (default: ${PARALLEL_SLOTS})
  --flash-attn MODE  Flash Attention: on, off, or auto (default: ${FLASH_ATTN})
  --split-mode MODE  GPU split: none, layer, row, or tensor (default: ${SPLIT_MODE})
  --cache-k TYPE     KV cache K type (default: ${CACHE_TYPE_K})
  --cache-v TYPE     KV cache V type (default: ${CACHE_TYPE_V})

Commands:
  prepare-runtime     Build llama.cpp from source (SM121a-real)
  runtime-info        Show binary and build ref
  download            Download missing model files; auto-creates HF venv if needed
  download --force    Re-download model and mmproj
  download --list     Show quantization options
  verify-files        Check GGUF magic header and file sizes
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
  test-reasoning      Thinking mode: verify 37×43=1591
  test-image          Vision: 1×1 PNG identification
  test-image-turn2    Vision: two consecutive VL requests (KV cache continuity)
  test-tools required Tool calling (required mode, unverified)
  test-tools auto     Tool calling (auto mode, unverified)
  network-info        Show bind address and public API URL
  client-config       Print OpenAI SDK usage examples

Environment overrides:
  MODEL_QUANT       Quantization variant (default: Q4_K_M)
  MODEL_FILE        Full filename override
  MMPROJ_FILE       mmproj filename override (default: mmproj-BF16.gguf)
  HF_REPO           HuggingFace repo (default: unsloth/Qwen3-VL-32B-Thinking-GGUF)
  HF_VENV          Private downloader venv (default: ${HF_VENV})
  AUTO_INSTALL_HF  Auto-create downloader venv when missing: 1 or 0 (default: ${AUTO_INSTALL_HF})
  LLAMA_CPP_REF     Git ref to build (default: ${LLAMA_CPP_REF})
  USER_HOME         Base home override (default: \$HOME)
  LLAMA_CPP_DIR     Build directory (default: ${LLAMA_CPP_DIR})
  MODEL_DIR         Model storage directory (default: ${MODEL_DIR})
  CTX_SIZE          Context window tokens (default: ${CTX_SIZE})
  API_PORT          Server port (default: ${API_PORT})
  FLASH_ATTN        Flash Attention mode: on, off, or auto (default: ${FLASH_ATTN})
  SPLIT_MODE       GPU split mode (default: ${SPLIT_MODE}; use none on one DGX Spark)
  CACHE_TYPE_K      KV cache K type (default: ${CACHE_TYPE_K})
  CACHE_TYPE_V      KV cache V type (default: ${CACHE_TYPE_V})
  HF_TOKEN          HuggingFace token for gated models (not embedded)
  MODEL_SHA256      Model file SHA-256 (optional; record after first download)
  MMPROJ_SHA256     mmproj file SHA-256 (optional)
HELP
    ;;
  *)
    die "Unknown command '${1}'. Run: $0 help"
    ;;
esac
