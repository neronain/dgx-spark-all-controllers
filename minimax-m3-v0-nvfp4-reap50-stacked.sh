#!/usr/bin/env bash
# MiniMax-M3-v0-NVFP4-REAP50 — stacked 2×DGX Spark controller v1.0
# Runtime and tuning are based on the Spark Arena experimental recipe:
#   model:     sparkarena/Minimax-M3-v0-NVFP4-REAP50
#   runtime:   SGLang custom W1/W3 NVFP4 normalization build
#   container: scitrera/dgx-spark-sglang-mm:v0
#   topology:  2 nodes, TP=2, SGLang native distributed backend
#
# IMPORTANT
# - This is an experimental REAP50 + NVFP4 checkpoint.
# - Do not replace the runtime with a generic SGLang/vLLM image unless its
#   W1/W3 NVFP4 scale-normalization support has been independently verified.
# - Run this controller as the normal user, not with sudo/root.
set -Eeuo pipefail

SCRIPT_VERSION="${SCRIPT_VERSION:-3.1.0}"
MODEL_LABEL="${MODEL_LABEL:-MiniMax-M3-v0 (NVFP4-REAP50) · 2-node}"
RUNTIME_LABEL="${RUNTIME_LABEL:-SGLang (Docker, stacked)}"
MODEL_FEATURES="${MODEL_FEATURES:-reasoning · tools · vision · experimental}"

# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_ID="${MODEL_ID:-sparkarena/Minimax-M3-v0-NVFP4-REAP50}"
MODEL_REVISION="${MODEL_REVISION:-main}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-minimax-m3-reap50}"

# ─── Runtime ──────────────────────────────────────────────────────────────────
RECOMMENDED_SGLANG_IMAGE="scitrera/dgx-spark-sglang-mm:v0"
SGLANG_IMAGE="${SGLANG_IMAGE:-$RECOMMENDED_SGLANG_IMAGE}"
ALLOW_UNVERIFIED_IMAGE="${ALLOW_UNVERIFIED_IMAGE:-false}"
HEAD_CONTAINER="${HEAD_CONTAINER:-sglang-minimax-m3-reap50-head}"
WORKER_CONTAINER="${WORKER_CONTAINER:-sglang-minimax-m3-reap50-worker}"

# ─── Cluster ──────────────────────────────────────────────────────────────────
MASTER_IP="${MASTER_IP:-10.100.152.1}"
WORKER_IP="${WORKER_IP:-10.100.152.2}"

# Ask for cluster IPs on start/restart so teammates whose addresses differ from
# the author's do not have to edit the file. Non-interactive shells keep the
# current values, so env overrides still work for automation.
prompt_cluster_config() {
  [[ -t 0 ]] || return 0
  local ans
  printf '\n== Cluster configuration (press Enter to keep the current value) ==\n'
  read -rp "  Head (master) node IP [${MASTER_IP}]: " ans || true; [[ -z "$ans" ]] || MASTER_IP="$ans"
  read -rp "  Worker node IP        [${WORKER_IP}]: " ans || true; [[ -z "$ans" ]] || WORKER_IP="$ans"
  read -rp "  SSH user for nodes    [${SSH_USER}]: " ans || true; [[ -z "$ans" ]] || SSH_USER="$ans"
  printf '\n'
}
SSH_USER="${SSH_USER:-${USER:-$(id -un)}}"
TRANSPORT_IP_MASTER="${TRANSPORT_IP_MASTER:-$MASTER_IP}"
TRANSPORT_IP_WORKER="${TRANSPORT_IP_WORKER:-$WORKER_IP}"
DIST_INIT_PORT="${DIST_INIT_PORT:-25000}"

# RoCE/NCCL. Set these to the actual 200G fabric interface/HCA when available.
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
NCCL_IB_HCA="${NCCL_IB_HCA:-}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# ─── Paths ────────────────────────────────────────────────────────────────────
USER_HOME="${USER_HOME:-$HOME}"
HF_HOME="${HF_HOME:-${USER_HOME}/.cache/huggingface}"
SGLANG_CACHE_ROOT="${SGLANG_CACHE_ROOT:-${USER_HOME}/.cache/sglang-minimax-m3}"
WORKER_HF_HOME="${WORKER_HF_HOME:-$HF_HOME}"
WORKER_SGLANG_CACHE_ROOT="${WORKER_SGLANG_CACHE_ROOT:-${SGLANG_CACHE_ROOT}}"
IMAGE_LOCK_FILE="${IMAGE_LOCK_FILE:-${HF_HOME}/.minimax-m3-reap50-sglang-image-id}"

# ─── Serving defaults from the tested Spark Arena recipe ─────────────────────
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
ADVERTISE_IP="${ADVERTISE_IP:-}"
ADVERTISE_INTERFACE="${ADVERTISE_INTERFACE:-}"
ROUTE_PROBE_IP="${ROUTE_PROBE_IP:-1.1.1.1}"

TP_SIZE="${TP_SIZE:-2}"
NNODES="${NNODES:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.81}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
REASONING_PARSER="${REASONING_PARSER:-minimax-m3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-minimax-m3}"
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-triton}"
FP4_GEMM_BACKEND="${FP4_GEMM_BACKEND:-cutlass}"
LOAD_FORMAT="${LOAD_FORMAT:-instanttensor}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-16}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-1 2 4 8 12 16}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-3600}"
HEAD_BEFORE_WORKER_DELAY="${HEAD_BEFORE_WORKER_DELAY:-5}"

# Optional SGLang limits. Empty means use runtime defaults.
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-}"

# Model integrity expectations.
SHARD_COUNT="${SHARD_COUNT:-24}"
MIN_TOTAL_BYTES="${MIN_TOTAL_BYTES:-115000000000}" # conservative floor; repo ~129 GB

# ─── Helpers ──────────────────────────────────────────────────────────────────
die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(date '+%H:%M:%S')] $*"; }
warn() { echo "WARN: $*" >&2; }

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
  ip="$(_detect_advertise_ip 2>/dev/null || true)"; [[ -n "$ip" ]] || ip="${API_HOST}"
  url="http://${ip}:${API_PORT}/v1"
  state="stopped"
  if curl -fsS -m 2 "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
    state="RUNNING"
  fi
  printf '  Model     : %s\n'            "${MODEL_LABEL}"
  printf '  Model ID  : %s\n'            "${MODEL_ID}"
  printf '  Runtime   : %s\n'            "${RUNTIME_LABEL}"
  printf '  Features  : %s\n'            "${MODEL_FEATURES}"
  printf '  Context   : %s tokens\n'     "${MAX_MODEL_LEN:-n/a}"
  printf '  Topology  : 2 nodes · TP=%s · head %s · worker %s\n' \
    "${TP_SIZE}" "${MASTER_IP}" "${WORKER_IP}"
  printf '  API (v1)  : %s\n'            "${url}"
  printf '  State     : %s  (port %s)\n\n' "${state}" "${API_PORT}"
}
ssh_worker() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "${SSH_USER}@${WORKER_IP}" "$@"
}

_require_non_root() {
  if (( EUID == 0 )); then
    die "Do not run this controller with sudo/root. Run as '${SUDO_USER:-$SSH_USER}'. sudo changes HOME, SSH identity, cache paths, and file ownership."
  fi
}

_require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

_require_supported_topology() {
  [[ "$TP_SIZE" == "2" && "$NNODES" == "2" ]] || \
    die "This controller is intentionally validated for TP_SIZE=2 and NNODES=2. Current: TP_SIZE=$TP_SIZE NNODES=$NNODES"
}

_require_recommended_image_or_override() {
  if [[ "$SGLANG_IMAGE" != "$RECOMMENDED_SGLANG_IMAGE" && "$ALLOW_UNVERIFIED_IMAGE" != "true" ]]; then
    die "Unverified runtime image '$SGLANG_IMAGE'. This checkpoint requires special W1/W3 NVFP4 scale normalization. Use '$RECOMMENDED_SGLANG_IMAGE', or set ALLOW_UNVERIFIED_IMAGE=true only after validating another build."
  fi
}

_ensure_local_dir() {
  local dir="$1" uid gid
  uid=$(id -u); gid=$(id -g)
  mkdir -p "$(dirname "$dir")" || die "Cannot create parent for $dir"
  if [[ -e "$dir" && ! -w "$dir" ]]; then
    log "Repairing local cache ownership: $dir"
    docker run --rm -v "$dir:/repair" --entrypoint sh "$SGLANG_IMAGE" \
      -c "chown -R ${uid}:${gid} /repair" \
      || die "Cannot repair $dir. Run: sudo chown -R ${uid}:${gid} '$dir'"
  fi
  mkdir -p "$dir" || die "Cannot create directory: $dir"
  [[ -w "$dir" ]] || die "Directory is not writable: $dir"
}

_ensure_worker_dir() {
  local dir="$1" uid gid
  uid=$(ssh_worker 'id -u') || die "Cannot get worker UID"
  gid=$(ssh_worker 'id -g') || die "Cannot get worker GID"
  ssh_worker "mkdir -p '$(dirname "$dir")'" || die "Cannot create worker cache parent"
  if ssh_worker "test -e '$dir'" && ! ssh_worker "test -w '$dir'"; then
    log "Repairing worker cache ownership: $dir"
    ssh_worker "docker run --rm -v '$dir:/repair' --entrypoint sh '$SGLANG_IMAGE' -c 'chown -R ${uid}:${gid} /repair'" \
      || die "Cannot repair worker directory $dir"
  fi
  ssh_worker "mkdir -p '$dir' && test -w '$dir'" || die "Worker directory is not writable: $dir"
}

_model_slug() {
  printf '%s' "${MODEL_ID/\//--}"
}


_prepare_hf_download_cache() {
  # HF Hub uses repository-specific lock directories below HF_HOME/.locks.
  # HF_HOME itself can be writable while a stale root-owned .locks directory
  # still causes snapshot_download() to fail. Repair only this repository's
  # lock/cache paths so other large model caches are not recursively chowned.
  local uid gid slug model_dir model_rel lock_dir
  uid=$(id -u)
  gid=$(id -g)
  slug=$(_model_slug)
  model_dir=$(_model_cache_dir "$HF_HOME")
  case "$model_dir" in
    "$HF_HOME"/*) model_rel="${model_dir#"$HF_HOME"/}" ;;
    *) die "Resolved model cache is outside HF_HOME: $model_dir" ;;
  esac
  lock_dir="${HF_HOME}/.locks/models--${slug}"

  _ensure_local_dir "$HF_HOME"
  log "Preparing Hugging Face lock/cache paths for ${MODEL_ID}"

  docker run --rm \
    -v "$HF_HOME:/cache" \
    --entrypoint sh \
    "$SGLANG_IMAGE" \
    -c "set -eu
      mkdir -p '/cache/.locks/models--${slug}' '/cache/${model_rel}'
      chown ${uid}:${gid} /cache /cache/.locks
      chown -R ${uid}:${gid} '/cache/.locks/models--${slug}' '/cache/${model_rel}'" \
    || die "Cannot repair Hugging Face cache paths under $HF_HOME"

  # Verify the exact paths that huggingface_hub will write to.
  : > "${lock_dir}/.write-test" \
    || die "Hugging Face lock directory is still not writable: ${lock_dir}"
  rm -f "${lock_dir}/.write-test"
  : > "${model_dir}/.write-test" \
    || die "Hugging Face model cache is still not writable: ${model_dir}"
  rm -f "${model_dir}/.write-test"
}

repair_download_cache() {
  _require_non_root
  _assert_runtime_images >/dev/null
  _prepare_hf_download_cache
  log "Hugging Face download cache: PASS"
}

_model_cache_dir() {
  local root="${1:-$HF_HOME}" slug candidate
  slug=$(_model_slug)
  for candidate in "${root}/models--${slug}" "${root}/hub/models--${slug}"; do
    if [[ -d "${candidate}/snapshots" ]]; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  printf '%s/models--%s' "$root" "$slug"
}

_snapshot_path() {
  local root="${1:-$HF_HOME}" model_dir ref commit snap
  model_dir=$(_model_cache_dir "$root")
  if [[ -d "${model_dir}/snapshots/${MODEL_REVISION}" ]]; then
    printf '%s' "${model_dir}/snapshots/${MODEL_REVISION}"
    return 0
  fi
  ref="${model_dir}/refs/${MODEL_REVISION}"
  if [[ -f "$ref" ]]; then
    commit=$(tr -d '[:space:]' < "$ref")
    if [[ -n "$commit" && -d "${model_dir}/snapshots/${commit}" ]]; then
      printf '%s' "${model_dir}/snapshots/${commit}"
      return 0
    fi
  fi
  snap=$(find "${model_dir}/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null || true)
  if [[ -n "$snap" ]]; then
    printf '%s' "$snap"
    return 0
  fi
  printf '%s' "${model_dir}/snapshots/${MODEL_REVISION}"
}

_snapshot_relative_path() {
  local snap="$1"
  case "$snap" in
    "$HF_HOME"/*) printf '%s' "${snap#"$HF_HOME"/}" ;;
    *) die "Snapshot is outside HF_HOME: $snap" ;;
  esac
}

_container_model_path() {
  local snap="$1" rel
  rel=$(_snapshot_relative_path "$snap")
  printf '/cache/%s' "$rel"
}

_image_id_local() {
  docker image inspect "$SGLANG_IMAGE" --format '{{.Id}}' 2>/dev/null
}

_image_id_worker() {
  ssh_worker "docker image inspect '$SGLANG_IMAGE' --format '{{.Id}}'" 2>/dev/null
}

_assert_runtime_images() {
  local head_id worker_id locked=""
  head_id=$(_image_id_local) || die "Runtime image not present on head. Run: $0 prepare-runtime"
  worker_id=$(_image_id_worker) || die "Runtime image not present on worker for user ${SSH_USER}. Run: $0 prepare-runtime"
  [[ "$head_id" == "$worker_id" ]] || die "Runtime image mismatch: head=$head_id worker=$worker_id"
  [[ -f "$IMAGE_LOCK_FILE" ]] && locked=$(tr -d '[:space:]' < "$IMAGE_LOCK_FILE")
  [[ -n "$locked" ]] || die "Runtime image is not locked. Run: $0 prepare-runtime"
  [[ "$locked" == "$head_id" ]] || die "Current image differs from locked image. locked=$locked current=$head_id. Re-run prepare-runtime after reviewing the image change."
  printf '%s' "$head_id"
}

_image_cache_key() {
  local id="$1"
  id="${id#sha256:}"
  printf '%s' "${id:0:16}"
}

_detect_advertise_ip() {
  [[ -z "$ADVERTISE_IP" ]] || { printf '%s' "$ADVERTISE_IP"; return; }
  if [[ -n "$ADVERTISE_INTERFACE" ]]; then
    ip -4 -o addr show dev "$ADVERTISE_INTERFACE" scope global | awk 'NR==1{split($4,a,"/");print a[1]}'
    return
  fi
  local src
  src=$(ip -4 route get "$ROUTE_PROBE_IP" 2>/dev/null | awk '{for(i=1;i<=NF;i++)if($i=="src"){print $(i+1);exit}}')
  if [[ -n "$src" ]]; then printf '%s' "$src"; return; fi
  ip -4 addr show scope global | awk '/inet /{split($2,a,"/");print a[1]}' | grep -v '^127\.' | head -1
}

_parse_options() {
  REMAINING_ARGS=()
  while (( $# )); do
    case "$1" in
      --context) MAX_MODEL_LEN="$2"; shift 2 ;;
      --context=*) MAX_MODEL_LEN="${1#*=}"; shift ;;
      --port) API_PORT="$2"; shift 2 ;;
      --port=*) API_PORT="${1#*=}"; shift ;;
      --bind) API_HOST="$2"; shift 2 ;;
      --bind=*) API_HOST="${1#*=}"; shift ;;
      --advertise-ip) ADVERTISE_IP="$2"; shift 2 ;;
      --advertise-ip=*) ADVERTISE_IP="${1#*=}"; shift ;;
      *) REMAINING_ARGS+=("$1"); shift ;;
    esac
  done
  # Reject impossible values here rather than letting SGLang fail minutes into startup.
  [[ "$MAX_MODEL_LEN" =~ ^[0-9]+$ ]] && (( MAX_MODEL_LEN > 0 )) ||
    die "Invalid --context: ${MAX_MODEL_LEN}"
  [[ "$API_PORT" =~ ^[0-9]+$ ]] && (( API_PORT >= 1 && API_PORT <= 65535 )) ||
    die "Invalid --port: ${API_PORT} (use 1..65535)"
  export MAX_MODEL_LEN API_PORT API_HOST ADVERTISE_IP
}

_nccl_env_lines() {
  cat <<EOF
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=${NCCL_DEBUG}
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}
EOF
  if [[ -n "$NCCL_SOCKET_IFNAME" ]]; then
    printf 'export NCCL_SOCKET_IFNAME=%q\n' "$NCCL_SOCKET_IFNAME"
    printf 'export GLOO_SOCKET_IFNAME=%q\n' "$NCCL_SOCKET_IFNAME"
  fi
  if [[ -n "$NCCL_IB_HCA" ]]; then
    printf 'export NCCL_IB_HCA=%q\n' "$NCCL_IB_HCA"
    echo 'export NCCL_IB_DISABLE=0'
  else
    echo 'export NCCL_IB_DISABLE=1'
  fi
}

_common_docker_env_args() {
  local args=(
    -e HF_HOME=/cache
    -e HF_HUB_OFFLINE=1
    -e TRANSFORMERS_OFFLINE=1
    -e HF_MODULES_CACHE=/root/.cache/huggingface_modules
    -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    -e NCCL_DEBUG="$NCCL_DEBUG"
    -e NCCL_IGNORE_CPU_AFFINITY=1
    -e NCCL_IB_GID_INDEX="$NCCL_IB_GID_INDEX"
  )
  if [[ -n "$NCCL_SOCKET_IFNAME" ]]; then
    args+=( -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" -e GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" )
  fi
  if [[ -n "$NCCL_IB_HCA" ]]; then
    args+=( -e NCCL_IB_HCA="$NCCL_IB_HCA" -e NCCL_IB_DISABLE=0 )
  else
    args+=( -e NCCL_IB_DISABLE=1 )
  fi
  printf '%s\n' "${args[@]}"
}

_port_in_use_local() {
  local listening
  listening="$(ss -H -ltn 2>/dev/null | awk '{print $4}' || true)"
  grep -Eq "(^|:)$1$" <<< "$listening"
}

_container_running_local() {
  [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null || true)" == "true" ]]
}

_container_running_worker() {
  [[ "$(ssh_worker "docker inspect -f '{{.State.Running}}' '$1' 2>/dev/null" || true)" == "true" ]]
}

# ─── Runtime lifecycle ────────────────────────────────────────────────────────
prepare_runtime() {
  _require_non_root
  _require_recommended_image_or_override
  _require_command docker
  _require_command ssh

  log "Pulling SGLang runtime on head: $SGLANG_IMAGE"
  docker pull "$SGLANG_IMAGE"
  _ensure_local_dir "$HF_HOME"
  local head_id worker_id
  head_id=$(_image_id_local) || die "Cannot inspect head image"

  log "Pulling the same runtime on worker: ${SSH_USER}@${WORKER_IP}"
  ssh_worker "docker pull '$SGLANG_IMAGE'"
  worker_id=$(_image_id_worker) || die "Cannot inspect worker image"
  [[ "$head_id" == "$worker_id" ]] || die "Image IDs differ after pull: head=$head_id worker=$worker_id"

  printf '%s\n' "$head_id" > "$IMAGE_LOCK_FILE"
  log "Runtime locked to immutable image ID: $head_id"
}

runtime_info() {
  echo "Image name : $SGLANG_IMAGE"
  echo "Locked ID  : $(cat "$IMAGE_LOCK_FILE" 2>/dev/null || echo '<not locked>')"
  echo "Head ID    : $(_image_id_local 2>/dev/null || echo '<missing>')"
  echo "Worker ID  : $(_image_id_worker 2>/dev/null || echo '<missing/unreachable>')"
  echo "Runtime    : SGLang native distributed, TP=2, nnodes=2"
  echo "Model      : $MODEL_ID @ $MODEL_REVISION"
}

clear_runtime_cache() {
  _require_non_root
  local image_id key head_cache worker_cache
  image_id=$(_assert_runtime_images)
  key=$(_image_cache_key "$image_id")
  head_cache="${SGLANG_CACHE_ROOT}/${key}"
  worker_cache="${WORKER_SGLANG_CACHE_ROOT}/${key}"
  stop || true
  log "Removing image-scoped runtime cache on head: $head_cache"
  if [[ -e "$head_cache" ]]; then
    docker run --rm -v "$head_cache:/wipe" --entrypoint sh "$SGLANG_IMAGE" -c 'rm -rf /wipe/* /wipe/.[!.]* /wipe/..?* 2>/dev/null || true'
  fi
  log "Removing image-scoped runtime cache on worker: $worker_cache"
  ssh_worker "if [ -e '$worker_cache' ]; then docker run --rm -v '$worker_cache:/wipe' --entrypoint sh '$SGLANG_IMAGE' -c 'rm -rf /wipe/* /wipe/.[!.]* /wipe/..?* 2>/dev/null || true'; fi"
  log "Runtime caches cleared. Model weights were not deleted."
}

# ─── Model lifecycle ──────────────────────────────────────────────────────────
download() {
  _require_non_root
  _assert_runtime_images >/dev/null
  _prepare_hf_download_cache
  log "Downloading $MODEL_ID @ $MODEL_REVISION (~129 GB, resume-safe)"
  docker run --rm -i \
    --user "$(id -u):$(id -g)" \
    -v "$HF_HOME:/cache" \
    -e HF_HOME=/cache \
    -e HOME=/tmp \
    -e MODEL_ID="$MODEL_ID" \
    -e MODEL_REVISION="$MODEL_REVISION" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    --entrypoint python3 \
    "$SGLANG_IMAGE" - <<'PY'
from huggingface_hub import snapshot_download
import os
path = snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    revision=os.environ["MODEL_REVISION"],
    cache_dir=os.environ["HF_HOME"],
    ignore_patterns=["*.ckpt", "*.pt", "*.pth", "optimizer*", "scheduler*"],
)
print(path)
PY
  log "Download complete"
}

_verify_snapshot_python() {
  local snap="$1"
  python3 - "$snap" "$SHARD_COUNT" "$MIN_TOTAL_BYTES" <<'PY'
import json, pathlib, sys
snap = pathlib.Path(sys.argv[1])
expected_shards = int(sys.argv[2])
min_total = int(sys.argv[3])
required = [
    "config.json", "generation_config.json", "hf_quant_config.json",
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
    "model.safetensors.index.json",
]
missing = [name for name in required if not (snap / name).is_file()]
if missing:
    raise SystemExit("Missing required files: " + ", ".join(missing))

cfg = json.loads((snap / "config.json").read_text())
arch = cfg.get("architectures", [])
if "MiniMaxM3SparseForConditionalGeneration" not in arch:
    raise SystemExit(f"Unexpected architecture: {arch}")
if cfg.get("model_type") != "minimax_m3_vl":
    raise SystemExit(f"Unexpected model_type: {cfg.get('model_type')}")
text = cfg.get("text_config", {})
if text.get("num_hidden_layers") != 60:
    raise SystemExit(f"Unexpected layer count: {text.get('num_hidden_layers')}")
if text.get("num_local_experts") != 64 or text.get("num_experts_per_tok") != 4:
    raise SystemExit("Unexpected MoE expert topology")
if text.get("max_position_embeddings") != 524288:
    raise SystemExit(f"Unexpected checkpoint max positions: {text.get('max_position_embeddings')}")
q = cfg.get("quantization_config", {})
if q.get("quant_algo") != "NVFP4" or q.get("group_size") != 16:
    raise SystemExit(f"Unexpected quantization config: {q}")
vision = cfg.get("vision_config", {})
if vision.get("num_hidden_layers") != 32:
    raise SystemExit("Vision tower config is missing or unexpected")
gen = json.loads((snap / "generation_config.json").read_text())
if gen.get("reasoning_parser") != "minimax-m3" or gen.get("tool_call_parser") != "minimax-m3":
    raise SystemExit("MiniMax parser defaults are missing")

shards = sorted(snap.glob("model-*.safetensors"))
if len(shards) != expected_shards:
    raise SystemExit(f"Expected {expected_shards} shards, found {len(shards)}")
small = [(p.name, p.stat().st_size) for p in shards if p.stat().st_size < 100 * 1000 * 1000]
if small:
    raise SystemExit(f"Suspiciously small shards: {small}")
total = sum(p.stat().st_size for p in shards)
if total < min_total:
    raise SystemExit(f"Shard total too small: {total:,} bytes < {min_total:,}")
index = json.loads((snap / "model.safetensors.index.json").read_text())
referenced = set(index.get("weight_map", {}).values())
missing_refs = sorted(name for name in referenced if not (snap / name).exists())
if missing_refs:
    raise SystemExit(f"Index references missing shards: {missing_refs[:5]}")
print(f"PASS snapshot={snap}")
print(f"architecture={arch[0]} model_type={cfg['model_type']}")
print(f"layers={text['num_hidden_layers']} experts={text['num_local_experts']} topk={text['num_experts_per_tok']}")
print(f"quant={q['quant_algo']} group_size={q['group_size']} shards={len(shards)} total={total/1e9:.2f} GB")
print(f"checkpoint_max_position_embeddings={text['max_position_embeddings']}")
PY
}

verify_files() {
  local snap
  snap=$(_snapshot_path "$HF_HOME")
  [[ -d "$snap" ]] || die "Model snapshot not found: $snap. Run: $0 download"
  log "Validating local MiniMax-M3 REAP50 snapshot"
  _verify_snapshot_python "$snap"
}

sync_worker() {
  _require_non_root
  _require_command rsync
  local model_dir parent name worker_root
  verify_files
  model_dir=$(_model_cache_dir "$HF_HOME")
  parent=$(dirname "$model_dir")
  name=$(basename "$model_dir")
  worker_root="$WORKER_HF_HOME"
  _ensure_worker_dir "$worker_root"
  log "Syncing model cache to worker without compression (better for 200G LAN)"
  log "Source: $model_dir/"
  log "Target: ${SSH_USER}@${WORKER_IP}:${worker_root}/${name}/"
  rsync -aH --info=progress2 --partial --inplace \
    --exclude='*.lock' --exclude='tmp_*' \
    "${parent}/${name}/" "${SSH_USER}@${WORKER_IP}:${worker_root}/${name}/"
  log "Worker sync complete"
}

_snapshot_manifest() {
  local snap="$1"
  python3 - "$snap" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
for f in sorted(p.iterdir(), key=lambda x: x.name):
    if f.is_file():
        try: size = f.stat().st_size
        except FileNotFoundError: continue
        print(f"{f.name}\t{size}")
PY
}

verify_worker() {
  _require_non_root
  local local_snap rel worker_snap local_manifest remote_manifest
  verify_worker
  local_snap=$(_snapshot_path "$HF_HOME")
  rel=$(_snapshot_relative_path "$local_snap")
  worker_snap="${WORKER_HF_HOME}/${rel}"
  ssh_worker "test -d '$worker_snap'" || die "Worker snapshot missing: $worker_snap. Run sync-worker"

  local_manifest=$(mktemp)
  remote_manifest=$(mktemp)
  _snapshot_manifest "$local_snap" > "$local_manifest"
  ssh_worker "python3 - '$worker_snap'" > "$remote_manifest" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
for f in sorted(p.iterdir(), key=lambda x: x.name):
    if f.is_file():
        try: size = f.stat().st_size
        except FileNotFoundError: continue
        print(f"{f.name}\t{size}")
PY
  if ! diff -u "$local_manifest" "$remote_manifest"; then
    rm -f "$local_manifest" "$remote_manifest"
    die "Worker snapshot file/size manifest differs from head"
  fi
  rm -f "$local_manifest" "$remote_manifest"
  local shard_n
  shard_n=$(ssh_worker "find -L '$worker_snap' -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l")
  [[ "$shard_n" == "$SHARD_COUNT" ]] || die "Worker has $shard_n shards; expected $SHARD_COUNT"
  log "verify-worker: PASS — file names and sizes match head"
}

verify_worker_full_hash() {
  _require_non_root
  local local_snap rel worker_snap
  verify_worker
  local_snap=$(_snapshot_path "$HF_HOME")
  rel=$(_snapshot_relative_path "$local_snap")
  worker_snap="${WORKER_HF_HOME}/${rel}"
  log "Computing full SHA-256 of all shards on both nodes; this is I/O intensive"
  local local_file remote_file
  local_file=$(mktemp)
  remote_file=$(mktemp)
  (cd "$local_snap" && sha256sum model-*.safetensors) > "$local_file"
  ssh_worker "cd '$worker_snap' && sha256sum model-*.safetensors" > "$remote_file"
  if ! diff -u "$local_file" "$remote_file"; then
    rm -f "$local_file" "$remote_file"
    die "Full shard hashes differ"
  fi
  rm -f "$local_file" "$remote_file"
  log "verify-worker-full-hash: PASS"
}

# ─── Diagnostics ──────────────────────────────────────────────────────────────
doctor() {
  _require_non_root
  _require_recommended_image_or_override
  local id help
  id=$(_assert_runtime_images)
  echo "=== Cluster ==="
  echo "Head        : $MASTER_IP (transport $TRANSPORT_IP_MASTER)"
  echo "Worker      : $WORKER_IP (transport $TRANSPORT_IP_WORKER)"
  echo "SSH user    : $SSH_USER"
  echo "Dist init   : ${TRANSPORT_IP_MASTER}:${DIST_INIT_PORT}"
  echo "Image       : $SGLANG_IMAGE"
  echo "Image ID    : $id"
  echo
  echo "=== Head GPU ==="
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  echo "=== Worker GPU ==="
  ssh_worker "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader" || true
  echo
  echo "=== Runtime packages ==="
  docker run --rm --gpus all --entrypoint python3 "$SGLANG_IMAGE" - <<'PY'
import platform
print("python:", platform.python_version())
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable")
    print("capability:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "unavailable")
except Exception as e: print("torch error:", repr(e))
try:
    import sglang
    print("sglang:", getattr(sglang, "__version__", "unknown"))
except Exception as e: print("sglang error:", repr(e))
try:
    import transformers
    print("transformers:", transformers.__version__)
except Exception as e: print("transformers error:", repr(e))
PY
  help=$(docker run --rm --entrypoint bash "$SGLANG_IMAGE" -lc 'sglang serve --help' 2>&1) || die "sglang serve --help failed"
  local flag
  for flag in --dist-init-addr --nnodes --node-rank --fp4-gemm-backend --load-format --moe-runner-backend --cuda-graph-bs --served-model-name; do
    grep -q -- "$flag" <<< "$help" || die "Required SGLang flag missing in image: $flag"
  done
  echo "Required SGLang CLI flags: PASS"
  [[ "$NCCL_IB_HCA" != "" ]] || warn "NCCL_IB_HCA is not set; the script will use TCP instead of RoCE/RDMA."
  [[ "$NCCL_SOCKET_IFNAME" != "" ]] || warn "NCCL_SOCKET_IFNAME is not set; NCCL may select the wrong interface."
  echo
  verify_files
  verify_worker
  log "doctor: PASS"
}

network_info() {
  local adv
  adv=$(_detect_advertise_ip)
  cat <<EOF
Head management   : $MASTER_IP
Worker management : $WORKER_IP
Head transport    : $TRANSPORT_IP_MASTER
Worker transport  : $TRANSPORT_IP_WORKER
Dist init          : ${TRANSPORT_IP_MASTER}:${DIST_INIT_PORT}
NCCL interface    : ${NCCL_SOCKET_IFNAME:-<not set>}
NCCL HCA          : ${NCCL_IB_HCA:-<not set; TCP fallback>}
RoCE GID index    : $NCCL_IB_GID_INDEX
API bind          : ${API_HOST}:${API_PORT}
Advertised API    : http://${adv}:${API_PORT}/v1
EOF
}

props() {
  cat <<EOF
MODEL_ID=$MODEL_ID
MODEL_REVISION=$MODEL_REVISION
SERVED_MODEL_NAME=$SERVED_MODEL_NAME
SGLANG_IMAGE=$SGLANG_IMAGE
MASTER_IP=$MASTER_IP
WORKER_IP=$WORKER_IP
TRANSPORT_IP_MASTER=$TRANSPORT_IP_MASTER
TRANSPORT_IP_WORKER=$TRANSPORT_IP_WORKER
DIST_INIT_PORT=$DIST_INIT_PORT
HF_HOME=$HF_HOME
WORKER_HF_HOME=$WORKER_HF_HOME
API_HOST=$API_HOST
API_PORT=$API_PORT
TP_SIZE=$TP_SIZE
NNODES=$NNODES
GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
MAX_MODEL_LEN=$MAX_MODEL_LEN
MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS
KV_CACHE_DTYPE=$KV_CACHE_DTYPE
MOE_RUNNER_BACKEND=$MOE_RUNNER_BACKEND
FP4_GEMM_BACKEND=$FP4_GEMM_BACKEND
LOAD_FORMAT=$LOAD_FORMAT
CUDA_GRAPH_MAX_BS=$CUDA_GRAPH_MAX_BS
CUDA_GRAPH_BS=$CUDA_GRAPH_BS
STARTUP_TIMEOUT=$STARTUP_TIMEOUT
EOF
}

# ─── Serving lifecycle ────────────────────────────────────────────────────────
_build_common_flags() {
  local model_path="$1"
  COMMON_FLAGS=(
    --model-path "$model_path"
    --served-model-name "$SERVED_MODEL_NAME"
    --mem-fraction-static "$GPU_MEMORY_UTILIZATION"
    --context-length "$MAX_MODEL_LEN"
    --tp "$TP_SIZE"
    --chunked-prefill-size "$MAX_NUM_BATCHED_TOKENS"
    --fp4-gemm-backend "$FP4_GEMM_BACKEND"
    --load-format "$LOAD_FORMAT"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --reasoning-parser "$REASONING_PARSER"
    --tool-call-parser "$TOOL_CALL_PARSER"
    --moe-runner-backend "$MOE_RUNNER_BACKEND"
    --trust-remote-code
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
  )
  local bs
  COMMON_FLAGS+=(--cuda-graph-bs)
  for bs in $CUDA_GRAPH_BS; do COMMON_FLAGS+=("$bs"); done
  COMMON_FLAGS+=(--host "$API_HOST" --port "$API_PORT")
  [[ -z "$MAX_RUNNING_REQUESTS" ]] || COMMON_FLAGS+=(--max-running-requests "$MAX_RUNNING_REQUESTS")
  [[ -z "$MAX_TOTAL_TOKENS" ]] || COMMON_FLAGS+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
  [[ -z "$MAX_PREFILL_TOKENS" ]] || COMMON_FLAGS+=(--max-prefill-tokens "$MAX_PREFILL_TOKENS")
}

start() {
  _require_non_root
  _require_supported_topology
  _require_recommended_image_or_override
  _require_command docker
  _require_command ssh
  _require_command curl
  _require_command ss

  local image_id key head_cache worker_cache local_snap rel worker_snap container_model quoted
  image_id=$(_assert_runtime_images)
  key=$(_image_cache_key "$image_id")
  head_cache="${SGLANG_CACHE_ROOT}/${key}"
  worker_cache="${WORKER_SGLANG_CACHE_ROOT}/${key}"
  _ensure_local_dir "$head_cache"
  _ensure_worker_dir "$worker_cache"

  verify_worker
  local_snap=$(_snapshot_path "$HF_HOME")
  rel=$(_snapshot_relative_path "$local_snap")
  worker_snap="${WORKER_HF_HOME}/${rel}"
  container_model=$(_container_model_path "$local_snap")

  if _port_in_use_local "$API_PORT"; then
    die "API port $API_PORT is already in use on head"
  fi
  if _port_in_use_local "$DIST_INIT_PORT"; then
    die "Distributed init port $DIST_INIT_PORT is already in use on head"
  fi

  docker rm -f "$HEAD_CONTAINER" >/dev/null 2>&1 || true
  ssh_worker "docker rm -f '$WORKER_CONTAINER' >/dev/null 2>&1 || true"

  _build_common_flags "$container_model"
  quoted=$(printf ' %q' "${COMMON_FLAGS[@]}")
  mapfile -t docker_env < <(_common_docker_env_args)

  log "Starting SGLang head first (node-rank 0)"
  docker run -d --name "$HEAD_CONTAINER" \
    --gpus all --network host --ipc host \
    -v "$HF_HOME:/cache:ro" \
    -v "$head_cache:/root/.cache" \
    "${docker_env[@]}" \
    --entrypoint bash "$SGLANG_IMAGE" \
    -lc "exec sglang serve${quoted} --dist-init-addr $(printf %q "${TRANSPORT_IP_MASTER}:${DIST_INIT_PORT}") --nnodes 2 --node-rank 0" \
    >/dev/null

  sleep "$HEAD_BEFORE_WORKER_DELAY"
  if ! _container_running_local "$HEAD_CONTAINER"; then
    docker logs --tail 250 "$HEAD_CONTAINER" 2>&1 || true
    die "Head container exited before worker startup"
  fi

  log "Starting SGLang worker (node-rank 1) on $WORKER_IP"
  local worker_script
  worker_script=$(cat <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
$(_nccl_env_lines)
export HF_HOME=/cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_MODULES_CACHE=/root/.cache/huggingface_modules
exec sglang serve${quoted} --dist-init-addr $(printf %q "${TRANSPORT_IP_MASTER}:${DIST_INIT_PORT}") --nnodes 2 --node-rank 1
EOF
)
  ssh_worker "mkdir -p /tmp/minimax-m3-reap50 && cat > /tmp/minimax-m3-reap50/worker.sh" <<< "$worker_script"
  ssh_worker "chmod +x /tmp/minimax-m3-reap50/worker.sh"
  ssh_worker "docker run -d --name '$WORKER_CONTAINER' \
    --gpus all --network host --ipc host \
    -v '$WORKER_HF_HOME:/cache:ro' \
    -v '$worker_cache:/root/.cache' \
    -v '/tmp/minimax-m3-reap50:/tmp/minimax-m3-reap50:ro' \
    --entrypoint bash '$SGLANG_IMAGE' /tmp/minimax-m3-reap50/worker.sh" >/dev/null

  sleep 8
  if ! _container_running_worker "$WORKER_CONTAINER"; then
    ssh_worker "docker logs --tail 250 '$WORKER_CONTAINER' 2>&1" || true
    docker logs --tail 100 "$HEAD_CONTAINER" 2>&1 || true
    die "Worker container exited during distributed initialization"
  fi

  log "Polling /v1/models for up to ${STARTUP_TIMEOUT}s"
  local deadline now adv
  deadline=$(( $(date +%s) + STARTUP_TIMEOUT ))
  while :; do
    if curl -fsS "http://127.0.0.1:${API_PORT}/v1/models" >/dev/null 2>&1; then
      adv=$(_detect_advertise_ip)
      log "MiniMax-M3 server is ready"
      cat <<EOF

Model    : $SERVED_MODEL_NAME
API      : http://${adv}:${API_PORT}/v1
Runtime  : SGLang / $SGLANG_IMAGE
Topology : 2×DGX Spark, TP=2, nnodes=2
Context  : $MAX_MODEL_LEN tokens
Mode     : multimodal + reasoning + tool calling
EOF
      return 0
    fi
    if ! _container_running_local "$HEAD_CONTAINER"; then
      docker logs --tail 300 "$HEAD_CONTAINER" 2>&1 || true
      ssh_worker "docker logs --tail 150 '$WORKER_CONTAINER' 2>&1" || true
      die "Head container exited before API became ready"
    fi
    if ! _container_running_worker "$WORKER_CONTAINER"; then
      ssh_worker "docker logs --tail 300 '$WORKER_CONTAINER' 2>&1" || true
      docker logs --tail 150 "$HEAD_CONTAINER" 2>&1 || true
      die "Worker container exited before API became ready"
    fi
    now=$(date +%s)
    (( now < deadline )) || break
    sleep 10
  done
  warn "Containers were left running for inspection"
  die "API did not become ready within ${STARTUP_TIMEOUT}s. Check: $0 logs head 300 and $0 logs worker 300"
}

stop() {
  _require_non_root
  log "Stopping head container"
  docker rm -f "$HEAD_CONTAINER" >/dev/null 2>&1 || true
  log "Stopping worker container"
  ssh_worker "docker rm -f '$WORKER_CONTAINER' >/dev/null 2>&1 || true" || true
  log "Stopped"
}

restart() {
  stop
  start
}

status() {
  echo "=== Configuration ==="
  echo "Model   : $MODEL_ID"
  echo "Image   : $SGLANG_IMAGE"
  echo "Context : $MAX_MODEL_LEN"
  echo "API     : $API_HOST:$API_PORT"
  echo "Dist    : ${TRANSPORT_IP_MASTER}:${DIST_INIT_PORT}"
  echo
  echo "=== Head container ==="
  docker ps -a --filter "name=^/${HEAD_CONTAINER}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' || true
  echo "=== Worker container ==="
  ssh_worker "docker ps -a --filter 'name=^/${WORKER_CONTAINER}$' --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'" || true
  echo
  echo "=== GPUs ==="
  echo "Head:"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
  echo "Worker:"
  ssh_worker "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader" 2>/dev/null || true
  echo
  echo "=== API ==="
  curl -fsS "http://127.0.0.1:${API_PORT}/v1/models" 2>/dev/null || echo "NOT READY"
  echo
}

logs() {
  local target="${1:-head}" n="${2:-200}"
  case "$target" in
    head|master) docker logs --tail "$n" "$HEAD_CONTAINER" 2>&1 ;;
    worker) ssh_worker "docker logs --tail '$n' '$WORKER_CONTAINER' 2>&1" ;;
    follow-head) docker logs -f --tail "$n" "$HEAD_CONTAINER" 2>&1 ;;
    follow-worker) ssh_worker "docker logs -f --tail '$n' '$WORKER_CONTAINER' 2>&1" ;;
    *) die "Usage: $0 logs [head|worker|follow-head|follow-worker] [N]" ;;
  esac
}

# ─── API clients and tests ────────────────────────────────────────────────────
client_config() {
  local adv
  adv=$(_detect_advertise_ip)
  cat <<PY
from openai import OpenAI

client = OpenAI(base_url="http://${adv}:${API_PORT}/v1", api_key="none")

# Non-thinking text
response = client.chat.completions.create(
    model="${SERVED_MODEL_NAME}",
    messages=[{"role": "user", "content": "Explain Tensor Parallel briefly."}],
    temperature=1.0,
    top_p=0.95,
    extra_body={
        "top_k": 40,
        "chat_template_kwargs": {"thinking_mode": "disabled"},
    },
)
print(response.choices[0].message.content)

# Thinking mode
response = client.chat.completions.create(
    model="${SERVED_MODEL_NAME}",
    messages=[{"role": "user", "content": "Solve 137 * 249 and check the result."}],
    temperature=1.0,
    top_p=0.95,
    extra_body={
        "top_k": 40,
        "chat_template_kwargs": {"thinking_mode": "enabled"},
    },
)
print(response.choices[0].message)
PY
}

_api_post_python() {
  python3 - "$API_PORT" <<'PY'
# Placeholder helper intentionally unused; tests below embed their payloads.
PY
}

test_text() {
  python3 - "$API_PORT" "$SERVED_MODEL_NAME" <<'PY'
import json, sys, urllib.request
port, model = sys.argv[1:]
payload = {
  "model": model,
  "messages": [{"role":"user","content":"Reply with exactly one concise sentence about Bangkok."}],
  "max_tokens": 96,
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 40,
  "chat_template_kwargs": {"thinking_mode":"disabled"},
}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
with urllib.request.urlopen(req, timeout=300) as r:
    data = json.loads(r.read())
print(data["choices"][0]["message"].get("content"))
print("usage:", data.get("usage"))
PY
}

test_reasoning() {
  python3 - "$API_PORT" "$SERVED_MODEL_NAME" <<'PY'
import json, sys, urllib.request
port, model = sys.argv[1:]
payload = {
  "model": model,
  "messages": [{"role":"user","content":"Calculate 137 × 249, verify it using a second method, and give the final answer."}],
  "max_tokens": 512,
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 40,
  "chat_template_kwargs": {"thinking_mode":"enabled"},
}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
with urllib.request.urlopen(req, timeout=600) as r:
    data = json.loads(r.read())
msg = data["choices"][0]["message"]
print("reasoning:", msg.get("reasoning_content"))
print("answer:", msg.get("content"))
print("usage:", data.get("usage"))
PY
}

test_tools() {
  python3 - "$API_PORT" "$SERVED_MODEL_NAME" <<'PY'
import json, sys, urllib.request
port, model = sys.argv[1:]
payload = {
  "model": model,
  "messages": [{"role":"user","content":"What is the weather in Bangkok? Use the tool."}],
  "tools": [{"type":"function","function":{
    "name":"get_weather","description":"Get current weather for a location",
    "parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}
  }}],
  "tool_choice":"required",
  "max_tokens":256,
  "temperature":1.0,
  "top_p":0.95,
  "chat_template_kwargs":{"thinking_mode":"disabled"},
}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
with urllib.request.urlopen(req, timeout=300) as r:
    data = json.loads(r.read())
msg = data["choices"][0]["message"]
print(json.dumps(msg.get("tool_calls"), ensure_ascii=False, indent=2))
if not msg.get("tool_calls"):
    raise SystemExit("Tool parser did not return a tool call")
PY
}

test_image() {
  local image_url="${1:-https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG}"
  python3 - "$API_PORT" "$SERVED_MODEL_NAME" "$image_url" <<'PY'
import json, sys, urllib.request
port, model, image_url = sys.argv[1:]
payload = {
  "model": model,
  "messages": [{"role":"user","content":[
      {"type":"image_url","image_url":{"url":image_url}},
      {"type":"text","text":"Describe the image accurately in one sentence."}
  ]}],
  "max_tokens":128,
  "temperature":1.0,
  "top_p":0.95,
  "chat_template_kwargs":{"thinking_mode":"disabled"},
}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
with urllib.request.urlopen(req, timeout=600) as r:
    data = json.loads(r.read())
print(data["choices"][0]["message"].get("content"))
print("usage:", data.get("usage"))
PY
}

stress() {
  local n="${1:-4}"
  python3 - "$API_PORT" "$SERVED_MODEL_NAME" "$n" <<'PY'
import concurrent.futures, json, sys, time, urllib.request
port, model, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
url = f"http://127.0.0.1:{port}/v1/chat/completions"
def run(i):
    payload = {"model":model,"messages":[{"role":"user","content":f"Request {i}: describe a blue sky in two sentences."}],
               "max_tokens":96,"temperature":1.0,"top_p":0.95,
               "chat_template_kwargs":{"thinking_mode":"disabled"}}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
    t=time.time()
    with urllib.request.urlopen(req, timeout=600) as r: data=json.loads(r.read())
    return i, time.time()-t, data.get("usage",{}).get("completion_tokens",0)
t0=time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
    results=list(ex.map(run, range(n)))
for i, sec, tok in results: print(f"request={i} seconds={sec:.2f} completion_tokens={tok}")
print(f"PASS {len(results)}/{n}; wall={time.time()-t0:.2f}s")
PY
}

bench() {
  python3 - "$API_PORT" "$SERVED_MODEL_NAME" <<'PY'
import json, sys, time, urllib.request
port, model = sys.argv[1:]
payload={"model":model,"messages":[{"role":"user","content":"Write a technically accurate 500-word explanation of mixture-of-experts inference."}],
         "max_tokens":700,"temperature":1.0,"top_p":0.95,
         "chat_template_kwargs":{"thinking_mode":"disabled"}}
req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
t0=time.time()
with urllib.request.urlopen(req, timeout=900) as r: data=json.loads(r.read())
elapsed=time.time()-t0
usage=data.get("usage",{}); tokens=usage.get("completion_tokens",0)
print(f"completion_tokens={tokens}")
print(f"elapsed={elapsed:.2f}s")
print(f"end_to_end_throughput={(tokens/elapsed if elapsed else 0):.2f} tok/s")
print("Note: this is end-to-end non-streaming throughput, not pure decode throughput.")
PY
}

official_recipe() {
  cat <<'EOF'
Official Spark Arena path (alternative to this controller):

  sparkrun run @experimental/minimax-m3-v0-nvfp4-2x-reap50

The controller script preserves the recipe's core runtime/tuning while adding:
image locking, local model verification, worker synchronization checks,
permission-safe caches, health polling, logs, tests, and benchmarks.
EOF
}

usage() {
  cat <<EOF
Usage: $0 <command> [options]

Preparation
  prepare-runtime          Pull and lock the exact SGLang image on both nodes
  runtime-info             Show image IDs and lock status
  repair-download-cache    Repair HF lock/cache ownership for this model
  download                 Download the ~129 GB model to the head
  verify-files             Validate config, quantization and all 24 shards
  sync-worker              Rsync the model cache to the worker
  verify-worker            Compare worker file names and sizes with head
  verify-worker-full-hash  SHA-256 all shards on both nodes (slow)
  doctor                   Full preflight: image, flags, GPUs, model and worker

Serving
  start                    Start 2-node SGLang TP=2 service
  stop                     Stop both containers
  restart                  Stop and start
  status                   Show containers, GPUs and API status
  logs [head|worker] [N]   Show logs; follow-head/follow-worker also supported
  clear-runtime-cache      Clear image-scoped compiled/runtime cache only

Overview
  info | banner            Model, port, features, topology and current state

Testing
  test-text                Non-thinking text generation
  test-reasoning           Thinking-mode generation
  test-tools               Function/tool call parser
  test-image [URL]         Native multimodal image test
  stress [N]               N concurrent requests (default 4)
  bench                    Simple end-to-end throughput test
  client-config            Print OpenAI Python SDK example

Information
  props                    Print effective settings
  network-info             Print cluster networking
  official-recipe          Print the official sparkrun command
  help                     Show this help

Common overrides
  --context N              Override context length for this invocation
  --port N                 Override API port
  --bind IP                Override API bind address

Examples
  $0 prepare-runtime
  $0 repair-download-cache
  $0 download
  $0 verify-files
  $0 sync-worker
  $0 doctor
  $0 start
  $0 test-text
  $0 test-image

Environment tuning example
  NCCL_SOCKET_IFNAME=enp1s0f0np0 NCCL_IB_HCA=rocep1s0f0 \
  MAX_MODEL_LEN=131072 GPU_MEMORY_UTILIZATION=0.81 $0 start
EOF
}

main() {
  case "${1:-}" in start|restart) prompt_cluster_config ;; esac
  _parse_options "$@"
  set -- "${REMAINING_ARGS[@]}"
  local cmd="${1:-help}"
  shift || true
  case "$cmd" in
    prepare-runtime) prepare_runtime "$@" ;;
    runtime-info) runtime_info "$@" ;;
    clear-runtime-cache) clear_runtime_cache "$@" ;;
    repair-download-cache) repair_download_cache "$@" ;;
    download) download "$@" ;;
    verify-files) verify_files "$@" ;;
    sync-worker) sync_worker "$@" ;;
    verify-worker) verify_worker "$@" ;;
    verify-worker-full-hash) verify_worker_full_hash "$@" ;;
    doctor) doctor "$@" ;;
    network-info) network_info "$@" ;;
    props) props "$@" ;;
    start) start "$@" ;;
    stop) stop "$@" ;;
    restart) restart "$@" ;;
    status) status "$@" ;;
    logs) logs "$@" ;;
    client-config) client_config "$@" ;;
    test-text) test_text "$@" ;;
    test-reasoning) test_reasoning "$@" ;;
    test-tools) test_tools "$@" ;;
    test-image) test_image "$@" ;;
    stress) stress "$@" ;;
    bench) bench "$@" ;;
    official-recipe) official_recipe "$@" ;;
    info|banner) info ;;
    help|-h|--help) usage ;;
    *) usage; die "Unknown command: $cmd" ;;
  esac
}

main "$@"
