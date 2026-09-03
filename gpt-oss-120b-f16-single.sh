#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_VERSION="${SCRIPT_VERSION:-3.1.0}"
MODEL_LABEL="${MODEL_LABEL:-GPT-OSS-120B (F16 GGUF)}"
RUNTIME_LABEL="${RUNTIME_LABEL:-llama.cpp}"
MODEL_FEATURES="${MODEL_FEATURES:-reasoning · tools · MoE-120B}"

MODEL_ID="unsloth/gpt-oss-120b-GGUF"
MODEL_REVISION="91daeef64d6b1e1078ad1d007f9efa98526d7bf1"
MODEL_FILE="gpt-oss-120b-F16.gguf"
MODEL_SIZE_BYTES="65369017728"
MODEL_SHA256="2d1f0298ae4b6c874d5a468598c5ce17c1763b3fea99de10b1a07df93cef014f"
SERVED_MODEL_NAME="gpt-oss-120b-f16"
LLAMA_CPP_REPO="https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_REF="master"
CUDA_ARCHITECTURES="121a-real"
INSTALL_BUILD_DEPENDENCIES="1"
CTX_SIZE="${CTX_SIZE:-131072}"
N_PARALLEL="1"
GPU_LAYERS="all"
FLASH_ATTN="on"
CACHE_TYPE_K="q8_0"
CACHE_TYPE_V="q8_0"
BATCH_SIZE="2048"
UBATCH_SIZE="512"
DEFAULT_REASONING_EFFORT="high"
REASONING_FORMAT="none"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
ADVERTISE_IP="${ADVERTISE_IP:-}"
ADVERTISE_INTERFACE="${ADVERTISE_INTERFACE:-}"
ROUTE_PROBE_IP="${ROUTE_PROBE_IP:-1.1.1.1}"
CLIENT_OVERHEAD_TOKENS="${CLIENT_OVERHEAD_TOKENS:-8192}"
API_KEY="${API_KEY:-}"
VERIFY_SHA_ON_START="0"
CLIENT_CONTEXT_TOKENS="${CLIENT_CONTEXT_TOKENS:-auto}"
CLIENT_MAX_OUTPUT_TOKENS="${CLIENT_MAX_OUTPUT_TOKENS:-16384}"
API_WAIT_SECONDS="1800"

CURRENT_USER="${SUDO_USER:-${USER:-$(id -un)}}"
USER_HOME="$(getent passwd "$CURRENT_USER" | cut -d: -f6)"
[[ -n "$USER_HOME" ]] || { echo "ERROR: cannot resolve home directory" >&2; exit 1; }
BASE_DIR="${USER_HOME}/gpt-oss-120b-f16"
MODEL_DIR="${USER_HOME}/models/gpt-oss-120b-f16"
LLAMA_CPP_DIR="${USER_HOME}/src/llama.cpp"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
LLAMA_SERVER="${LLAMA_CPP_DIR}/build/bin/llama-server"
LLAMA_BENCH="${LLAMA_CPP_DIR}/build/bin/llama-bench"
RUNTIME_LOCK="${BASE_DIR}/llama-cpp-commit.txt"
RUNTIME_INFO="${BASE_DIR}/RUNTIME_INFO.txt"
SESSION_NAME="gpt-oss-120b-f16"
LOG_DIR="${BASE_DIR}/logs"
LOG_FILE="${LOG_DIR}/llama-server.log"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
MODEL_URL="https://huggingface.co/${MODEL_ID}/resolve/${MODEL_REVISION}/${MODEL_FILE}?download=true"

log()  { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
warn() { printf '\nWARNING: %s\n' "$*" >&2; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
require_command() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }
validate_effort() { case "$1" in low|medium|high) ;; *) die "reasoning effort must be low, medium, or high" ;; esac; }
api_auth_args() { API_AUTH_ARGS=(); [[ -z "$API_KEY" ]] || API_AUTH_ARGS=(-H "Authorization: Bearer ${API_KEY}"); }
server_running() { tmux has-session -t "$SESSION_NAME" 2>/dev/null; }
port_in_use() { timeout 1 bash -c "</dev/tcp/127.0.0.1/${API_PORT}" 2>/dev/null; }
runtime_commit() { git -C "$LLAMA_CPP_DIR" rev-parse HEAD; }


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
    computed=$(( CTX_SIZE - CLIENT_MAX_OUTPUT_TOKENS - CLIENT_OVERHEAD_TOKENS ))

    (( computed >= 1024 )) ||
      die "Context ${CTX_SIZE} is too small for output and overhead budgets."

    CLIENT_CONTEXT_TOKENS="$computed"
  fi
}

validate_common_config() {
  validate_positive_integer "CTX_SIZE" "${CTX_SIZE}"
  validate_port "$API_PORT"
  validate_positive_integer "CLIENT_MAX_OUTPUT_TOKENS" "$CLIENT_MAX_OUTPUT_TOKENS"
  validate_positive_integer "CLIENT_OVERHEAD_TOKENS" "$CLIENT_OVERHEAD_TOKENS"

  if [[ "$CLIENT_CONTEXT_TOKENS" != "auto" ]]; then
    validate_positive_integer "CLIENT_CONTEXT_TOKENS" "$CLIENT_CONTEXT_TOKENS"
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



quick_file_check() {
  [[ -f "$MODEL_PATH" ]] || die "model is missing: $MODEL_PATH"
  local size
  size="$(stat -c '%s' "$MODEL_PATH")"
  [[ "$size" == "$MODEL_SIZE_BYTES" ]] || die "model size mismatch: expected $MODEL_SIZE_BYTES, got $size"
}

wait_for_api() {
  local deadline=$((SECONDS + API_WAIT_SECONDS))
  log "Waiting for llama.cpp API"
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 5 "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then return; fi
    if ! server_running; then tail -n 350 "$LOG_FILE" 2>/dev/null || true; die "llama-server stopped before API became ready"; fi
    sleep 5
  done
  tail -n 350 "$LOG_FILE" 2>/dev/null || true
  die "API did not become ready within ${API_WAIT_SECONDS}s"
}

install_build_dependencies() {
  [[ "$INSTALL_BUILD_DEPENDENCIES" == "1" ]] || return
  sudo apt-get update
  sudo apt-get install -y git clang cmake ninja-build curl jq tmux libcurl4-openssl-dev libssl-dev
}

checkout_runtime() {
  local update="$1" target="$LLAMA_CPP_REF"
  if [[ "$update" != "1" && -s "$RUNTIME_LOCK" ]]; then target="$(tr -d '[:space:]' < "$RUNTIME_LOCK")"; log "Using locked llama.cpp commit: $target"; else log "Using requested llama.cpp ref: $target"; fi
  git -C "$LLAMA_CPP_DIR" fetch --all --tags --prune
  git -C "$LLAMA_CPP_DIR" checkout --detach "$target"
  if [[ "$update" == "1" || ! -s "$RUNTIME_LOCK" ]]; then
    if [[ "$target" == "master" || "$target" == "main" ]]; then git -C "$LLAMA_CPP_DIR" checkout "$target"; git -C "$LLAMA_CPP_DIR" pull --ff-only; fi
  fi
}

build_runtime_internal() {
  local update="$1"
  install_build_dependencies
  require_command git; require_command cmake; require_command nvidia-smi
  mkdir -p "$(dirname "$LLAMA_CPP_DIR")" "$BASE_DIR"
  [[ -d "${LLAMA_CPP_DIR}/.git" ]] || git clone "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR"
  checkout_runtime "$update"
  local commit
  commit="$(runtime_commit)"
  cmake -S "$LLAMA_CPP_DIR" -B "${LLAMA_CPP_DIR}/build" -G Ninja \
    -DGGML_NATIVE=ON -DGGML_CUDA=ON -DGGML_CURL=ON -DGGML_RPC=ON \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES"
  cmake --build "${LLAMA_CPP_DIR}/build" --config Release --target llama-server llama-bench -j "$(nproc)"
  [[ -x "$LLAMA_SERVER" ]] || die "llama-server was not built"
  [[ -x "$LLAMA_BENCH" ]] || die "llama-bench was not built"
  printf '%s\n' "$commit" > "$RUNTIME_LOCK"
  { echo "llama_cpp_repository=${LLAMA_CPP_REPO}"; echo "llama_cpp_commit=${commit}"; echo "cuda_architectures=${CUDA_ARCHITECTURES}"; echo "built_at=$(date --iso-8601=seconds)"; "$LLAMA_SERVER" --version 2>&1 || true; } > "$RUNTIME_INFO"
  log "Runtime built and locked"; cat "$RUNTIME_INFO"
}

build_runtime() { build_runtime_internal 0; }
update_runtime() { warn "Intentionally updating llama.cpp and replacing the runtime lock"; rm -f "$RUNTIME_LOCK"; build_runtime_internal 1; }
runtime_info() {
  echo "===== LOCK ====="; cat "$RUNTIME_LOCK" 2>/dev/null || echo "No lock"
  echo; echo "===== BUILD INFO ====="; cat "$RUNTIME_INFO" 2>/dev/null || echo "No build info"
  echo; echo "===== BINARY ====="; [[ -x "$LLAMA_SERVER" ]] && "$LLAMA_SERVER" --version || true
  echo; echo "===== GIT ====="; [[ -d "${LLAMA_CPP_DIR}/.git" ]] && { git -C "$LLAMA_CPP_DIR" log -1 --oneline; git -C "$LLAMA_CPP_DIR" status --short --branch; }
}

download_model() {
  require_command curl; require_command sha256sum; require_command df
  mkdir -p "$MODEL_DIR"; df -h "$MODEL_DIR"
  if [[ -f "$MODEL_PATH" ]]; then
    local current_size
    current_size="$(stat -c '%s' "$MODEL_PATH")"
    if [[ "$current_size" == "$MODEL_SIZE_BYTES" ]]; then
      if printf '%s  %s\n' "$MODEL_SHA256" "$MODEL_PATH" | sha256sum --check -; then log "Model already exists and is verified"; return; fi
      mv -f "$MODEL_PATH" "${MODEL_PATH}.corrupt.$(date +%s)"
    elif (( current_size > MODEL_SIZE_BYTES )); then mv -f "$MODEL_PATH" "${MODEL_PATH}.corrupt.$(date +%s)"; else log "Resuming partial download at ${current_size} bytes"; fi
  fi
  curl --fail --location --retry 10 --retry-delay 5 --retry-all-errors --continue-at - --output "$MODEL_PATH" "$MODEL_URL"
  verify_files; du -sh "$MODEL_DIR"
}

verify_files() {
  require_command sha256sum; quick_file_check
  log "Verifying SHA-256 for the 65.4 GB GGUF"
  printf '%s  %s\n' "$MODEL_SHA256" "$MODEL_PATH" | sha256sum --check -
  log "Model size and SHA-256 are correct"
}

serve_foreground() {
  [[ -x "$LLAMA_SERVER" ]] || die "run build-runtime first"
  quick_file_check; validate_effort "$DEFAULT_REASONING_EFFORT"; mkdir -p "$LOG_DIR"
  local -a args=(
    --model "$MODEL_PATH" --alias "$SERVED_MODEL_NAME" --host "$API_HOST" --port "$API_PORT"
    --ctx-size "$CTX_SIZE" --parallel "$N_PARALLEL" --n-gpu-layers "$GPU_LAYERS"
    --flash-attn "$FLASH_ATTN" --cache-type-k "$CACHE_TYPE_K" --cache-type-v "$CACHE_TYPE_V"
    --batch-size "$BATCH_SIZE" --ubatch-size "$UBATCH_SIZE" --cont-batching --cache-prompt --jinja
    --reasoning-format "$REASONING_FORMAT" --chat-template-kwargs "{\"reasoning_effort\":\"${DEFAULT_REASONING_EFFORT}\"}"
    --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 --repeat-penalty 1.0 --metrics --timeout 3600 --log-timestamps
  )
  [[ -z "$API_KEY" ]] || args+=(--api-key "$API_KEY")
  exec "$LLAMA_SERVER" "${args[@]}"
}

start_server() {
  local advertise_ip
  advertise_ip="$(detect_advertise_ip)"

  require_command tmux; require_command curl; require_command timeout; require_command nvidia-smi
  nvidia-smi >/dev/null 2>&1 || die "host nvidia-smi failed"
  [[ -x "$LLAMA_SERVER" ]] || die "run build-runtime first"
  [[ -s "$RUNTIME_LOCK" ]] || die "runtime lock missing; run build-runtime"
  [[ "$(runtime_commit)" == "$(tr -d '[:space:]' < "$RUNTIME_LOCK")" ]] || die "llama.cpp source differs from locked commit"
  quick_file_check; [[ "$VERIFY_SHA_ON_START" != "1" ]] || verify_files
  mkdir -p "$LOG_DIR"; : > "$LOG_FILE"
  server_running && tmux kill-session -t "$SESSION_NAME" && sleep 2 || true
  port_in_use && die "port ${API_PORT} is already in use"
  local command
    printf -v command \
    'API_HOST=%q API_PORT=%q CTX_SIZE=%q API_KEY=%q exec bash %q _serve >> %q 2>&1' \
    "$API_HOST" "$API_PORT" "$CTX_SIZE" "$API_KEY" "$SCRIPT_PATH" "$LOG_FILE"
  tmux new-session -d -s "$SESSION_NAME" "$command"
  wait_for_api
  log "READY"; echo "Base URL: http://${advertise_ip}:${API_PORT}/v1"; echo "Model: ${SERVED_MODEL_NAME}"; echo "Context: ${CTX_SIZE}"
}

stop_server() { require_command tmux; if server_running; then tmux kill-session -t "$SESSION_NAME"; log "Stopped"; else log "Already stopped"; fi; }

show_status() {
  local advertise_ip
  advertise_ip="$(detect_advertise_ip)"

  require_command curl; require_command tmux
  echo "===== CONFIG ====="; echo "Advertise endpoint: http://${advertise_ip}:${API_PORT}/v1"; echo "Model: $MODEL_ID@$MODEL_REVISION"; echo "File: $MODEL_FILE"; echo "Served name: $SERVED_MODEL_NAME"; echo "Context: $CTX_SIZE"; echo "KV cache: K=$CACHE_TYPE_K V=$CACHE_TYPE_V"; echo "Reasoning: $DEFAULT_REASONING_EFFORT"
  echo; echo "===== RUNTIME ====="; [[ -x "$LLAMA_SERVER" ]] && "$LLAMA_SERVER" --version || echo "Not built"; echo "Locked commit: $(cat "$RUNTIME_LOCK" 2>/dev/null || echo missing)"
  echo; echo "===== SESSION ====="; server_running && tmux list-sessions | grep "^${SESSION_NAME}:" || echo "STOPPED"
  echo; echo "===== GPU ====="; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
  echo; echo "===== PROCESS ====="; pgrep -af "$LLAMA_SERVER" || true
  echo; echo "===== API ====="
  if curl -fsS --max-time 5 "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then echo "API: HEALTHY"; api_auth_args; curl -sS "${API_AUTH_ARGS[@]}" "http://127.0.0.1:${API_PORT}/v1/models"; echo; else echo "API: NOT READY"; fi
}

show_logs() { [[ -f "$LOG_FILE" ]] || die "no log file yet"; tail -n "${1:-350}" "$LOG_FILE"; }
show_props() { api_auth_args; curl -sS "${API_AUTH_ARGS[@]}" "http://127.0.0.1:${API_PORT}/props"; echo; }
run_bench() { [[ -x "$LLAMA_BENCH" ]] || die "run build-runtime first"; quick_file_check; "$LLAMA_BENCH" --model "$MODEL_PATH" --n-gpu-layers "$GPU_LAYERS" --flash-attn "$FLASH_ATTN" --batch-size "$BATCH_SIZE" --ubatch-size "$UBATCH_SIZE" --prompt 2048 --n-gen 256 --repetitions 3; }

post_json() { api_auth_args; curl -sS "${API_AUTH_ARGS[@]}" "http://127.0.0.1:${API_PORT}/v1/chat/completions" -H "Content-Type: application/json" --data-binary @-; }

test_text() {
  post_json <<EOF
{"model":"${SERVED_MODEL_NAME}","messages":[{"role":"system","content":"You are a precise software engineering assistant."},{"role":"user","content":"Write a Python IPv4 validator with type hints and pytest tests."}],"chat_template_kwargs":{"reasoning_effort":"medium"},"temperature":1.0,"top_p":1.0,"top_k":0,"min_p":0,"repeat_penalty":1.0,"max_tokens":4096}
EOF
  echo
}

test_reasoning() {
  local effort="${1:-$DEFAULT_REASONING_EFFORT}"; validate_effort "$effort"
  post_json <<EOF
{"model":"${SERVED_MODEL_NAME}","messages":[{"role":"user","content":"Design a resilient local coding-agent platform. Compare failure modes, security boundaries, observability, and recovery procedures."}],"chat_template_kwargs":{"reasoning_effort":"${effort}"},"temperature":1.0,"top_p":1.0,"top_k":0,"min_p":0,"repeat_penalty":1.0,"max_tokens":16384}
EOF
  echo
}

test_tools() {
  local choice="${1:-required}"; [[ "$choice" == "required" || "$choice" == "auto" ]] || die "tool choice must be required or auto"
  post_json <<EOF
{"model":"${SERVED_MODEL_NAME}","messages":[{"role":"user","content":"Inspect src/app.py with the read_file tool before proposing any change."}],"tools":[{"type":"function","function":{"name":"read_file","description":"Read a UTF-8 text file from the current workspace.","strict":true,"parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}}}],"tool_choice":"${choice}","parallel_tool_calls":false,"chat_template_kwargs":{"reasoning_effort":"medium"},"temperature":1.0,"top_p":1.0,"top_k":0,"repeat_penalty":1.0,"max_tokens":4096}
EOF
  echo; echo "Expected: choices[0].message.tool_calls"
}

test_tool_loop() {
  require_command python3
  local key="${API_KEY:-llama-local}"
  OPENAI_BASE_URL="http://127.0.0.1:${API_PORT}/v1" OPENAI_API_KEY="$key" OPENAI_MODEL="$SERVED_MODEL_NAME" python3 - <<'PY'
import json, os, urllib.request, uuid
base=os.environ['OPENAI_BASE_URL'].rstrip('/'); key=os.environ['OPENAI_API_KEY']; model=os.environ['OPENAI_MODEL']
def call(payload):
    req=urllib.request.Request(base+'/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':f'Bearer {key}'})
    with urllib.request.urlopen(req,timeout=900) as r: return json.loads(r.read())
tools=[{'type':'function','function':{'name':'read_file','description':'Read a UTF-8 file from the workspace.','strict':True,'parameters':{'type':'object','properties':{'path':{'type':'string'}},'required':['path'],'additionalProperties':False}}}]
messages=[{'role':'user','content':'Read src/app.py before identifying the bug. Use read_file first.'}]
first=call({'model':model,'messages':messages,'tools':tools,'tool_choice':'required','parallel_tool_calls':False,'chat_template_kwargs':{'reasoning_effort':'medium'},'temperature':1.0,'top_p':1.0,'top_k':0,'repeat_penalty':1.0,'max_tokens':4096})
msg=first['choices'][0]['message']; calls=msg.get('tool_calls') or []
if not calls: raise SystemExit('FAIL: no structured tool_calls\n'+json.dumps(first,indent=2,ensure_ascii=False))
tc=calls[0]; fn=tc.get('function') or {}; args=json.loads(fn.get('arguments') or '{}')
if fn.get('name')!='read_file' or args.get('path')!='src/app.py': raise SystemExit(f'FAIL: unexpected tool call: {tc}')
messages += [msg,{'role':'tool','tool_call_id':tc.get('id') or str(uuid.uuid4()),'name':'read_file','content':'def divide(a: float, b: float) -> float:\n    return a / 0\n'}]
second=call({'model':model,'messages':messages,'tools':tools,'tool_choice':'auto','parallel_tool_calls':False,'chat_template_kwargs':{'reasoning_effort':'medium'},'temperature':1.0,'top_p':1.0,'top_k':0,'repeat_penalty':1.0,'max_tokens':4096})
content=second['choices'][0]['message'].get('content') or ''
if not content.strip(): raise SystemExit('FAIL: final content empty\n'+json.dumps(second,indent=2,ensure_ascii=False))
print('PASS: native Harmony tool call and role=tool continuation'); print(content)
PY
}

test_harmony() {
  require_command python3
  local key="${API_KEY:-llama-local}"
  OPENAI_BASE_URL="http://127.0.0.1:${API_PORT}/v1" OPENAI_API_KEY="$key" OPENAI_MODEL="$SERVED_MODEL_NAME" python3 - <<'PY'
import json, os, urllib.request
base=os.environ['OPENAI_BASE_URL'].rstrip('/'); key=os.environ['OPENAI_API_KEY']; model=os.environ['OPENAI_MODEL']
payload={'model':model,'messages':[{'role':'user','content':'State one benefit of unit tests in one sentence.'}],'chat_template_kwargs':{'reasoning_effort':'low'},'temperature':1.0,'top_p':1.0,'top_k':0,'repeat_penalty':1.0,'max_tokens':1024}
req=urllib.request.Request(base+'/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':f'Bearer {key}'})
with urllib.request.urlopen(req,timeout=600) as r: result=json.loads(r.read())
msg=result['choices'][0]['message']; content=msg.get('content') or ''
markers=('<|start|>','<|channel|>','<|message|>','<|constrain|>'); leaked=[x for x in markers if x in content]
if leaked: raise SystemExit('FAIL: raw Harmony markers leaked: '+', '.join(leaked)+'\n'+json.dumps(result,indent=2,ensure_ascii=False))
if not content.strip(): raise SystemExit('FAIL: final content empty\n'+json.dumps(result,indent=2,ensure_ascii=False))
print('PASS: no raw Harmony control markers in final content'); print('Message keys:',sorted(msg)); print(content)
PY
}

stress_test() {
  local minutes="${1:-30}"; [[ "$minutes" =~ ^[0-9]+$ ]] || die "minutes must be an integer"
  require_command python3
  local key="${API_KEY:-llama-local}"
  OPENAI_BASE_URL="http://127.0.0.1:${API_PORT}/v1" OPENAI_API_KEY="$key" OPENAI_MODEL="$SERVED_MODEL_NAME" STRESS_MINUTES="$minutes" python3 - <<'PY'
import json, os, time, urllib.request
base=os.environ['OPENAI_BASE_URL'].rstrip('/'); key=os.environ['OPENAI_API_KEY']; model=os.environ['OPENAI_MODEL']; deadline=time.monotonic()+int(os.environ['STRESS_MINUTES'])*60
prompts=['Write a safe atomic file replacement function in Python.','Explain a two-lock deadlock and the smallest reliable fix.','Design unit tests for a token-bucket rate limiter.','Review an HTTP retry policy and identify unsafe edge cases.']
ok=bad=i=0
while time.monotonic()<deadline:
    payload={'model':model,'messages':[{'role':'user','content':prompts[i%len(prompts)]}],'chat_template_kwargs':{'reasoning_effort':'low'},'temperature':1.0,'top_p':1.0,'top_k':0,'repeat_penalty':1.0,'max_tokens':1024}
    req=urllib.request.Request(base+'/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':f'Bearer {key}'})
    start=time.monotonic()
    try:
        with urllib.request.urlopen(req,timeout=900) as r: result=json.loads(r.read())
        content=result['choices'][0]['message'].get('content') or ''
        if not content.strip(): raise RuntimeError('empty content')
        ok+=1; print(f'PASS request={ok+bad} elapsed={time.monotonic()-start:.1f}s successes={ok} failures={bad}',flush=True)
    except Exception as exc:
        bad+=1; print(f'FAIL request={ok+bad} elapsed={time.monotonic()-start:.1f}s error={exc!r}',flush=True)
    i+=1
print(f'RESULT successes={ok} failures={bad} total={ok+bad}')
if bad: raise SystemExit(1)
PY
}

client_config() {
  local advertise_ip
  advertise_ip="$(detect_advertise_ip)"
  local key="${API_KEY:-llama-local}"
  cat <<EOF
Provider:          OpenAI Compatible
Base URL:          http://${advertise_ip}:${API_PORT}/v1
API key:           ${key}
Model ID:          ${SERVED_MODEL_NAME}
Context window:    ${CTX_SIZE}
Max input tokens:  ${CLIENT_CONTEXT_TOKENS}
Max output tokens: ${CLIENT_MAX_OUTPUT_TOKENS}
Text input:        yes
Multimodal:        no
Reasoning:         low / medium / high
Native tools:      yes, Harmony parsed by llama.cpp
Parallel tools:    start with false

Recommended sampling:
  temperature=1.0
  top_p=1.0
  top_k=0
  min_p=0
  repeat_penalty=1.0
EOF
}

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
  $(basename "$0") client-config --advertise-ip 192.0.2.10
  $(basename "$0") network-info

  $(basename "$0") build-runtime|update-runtime|runtime-info
  $(basename "$0") download|verify-files
  $(basename "$0") start|stop|restart|status
  $(basename "$0") logs [lines]|props|bench
  $(basename "$0") test-text
  $(basename "$0") test-reasoning [low|medium|high]
  $(basename "$0") test-tools [required|auto]
  $(basename "$0") test-tool-loop|test-harmony
  $(basename "$0") stress [minutes]
  $(basename "$0") client-config
EOF
}

COMMAND="${1:-help}"
if (( $# )); then
  shift
fi

parse_common_options "$@"
set -- "${REMAINING_ARGS[@]}"

case "$COMMAND" in
  build-runtime) build_runtime ;;
  update-runtime) update_runtime ;;
  runtime-info) runtime_info ;;
  download) download_model ;;
  verify-files) verify_files ;;
  start) start_server ;;
  stop) stop_server ;;
  restart) stop_server; start_server ;;
  status) show_status ;;
  logs) show_logs "${1:-350}" ;;
  props) show_props ;;
  bench) run_bench ;;
  test-text) test_text ;;
  test-reasoning) test_reasoning "${1:-$DEFAULT_REASONING_EFFORT}" ;;
  test-tools) test_tools "${1:-required}" ;;
  test-tool-loop) test_tool_loop ;;
  test-harmony) test_harmony ;;
  stress) stress_test "${1:-30}" ;;
  client-config) client_config ;;
  _serve) serve_foreground ;;
  info|banner)
    info
    ;;
  network-info)
    network_info
    ;;
  help|-h|--help) show_help ;;
  *) show_help; exit 1 ;;
esac
