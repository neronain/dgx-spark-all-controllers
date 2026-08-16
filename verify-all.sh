#!/usr/bin/env bash
# Static validation for every DGX Spark controller in this repository.
# Designed by neronain · https://www.facebook.com/neronain.minidev
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_IP="${TEST_IP:-192.0.2.10}"

# Single-node controllers
singles=(
  gemma-4-26b-a4b-it-gguf-single.sh
  gemma-4-31b-it-uncensored-heretic-q8_0-dgx-spark.sh
  gemma-4-31b-it-uncensored-single.sh
  gemma4-26-a4b-q8xl-single.sh
  gemma4-31b-single.sh
  gpt-oss-120b-f16-single.sh
  llama33-70b-nvfp4-single.sh
  nemotron-3-super-single.sh
  nemotron-omni-aeon-single.sh
  ornith-1.0-35b-bf16-dgx-spark.sh
  ornith-1.0-35b-uncensored-heretic-nvfp4-fp8dense-gb10-dgx-spark.sh
  qwen3-coder-next-nvfp4-gb10-dgx-spark.sh
  qwen3-coder-next-single.sh
  qwen3-vl-32b-instruct-1m-bf16-dgx-spark.sh
  qwen3-vl-32b-thinking-single.sh
  qwen36-hauhau-q6kp-single.sh
  redteam-modelctl.sh
)

# Stacked / multi-node controllers. `start` and `restart` ask for the cluster
# addresses on a TTY, so every check below runs with stdin redirected.
stacked=(
  deepseek-v4-flash-nvfp4-stacked.sh
  gemma4-31b-stacked.sh
  minimax-m3-v0-nvfp4-reap50-stacked.sh
  minimax-m27-luke-stacked.sh
  vllm-stackctl.sh
)

controllers=("${singles[@]}" "${stacked[@]}")

for script in "${controllers[@]}"; do
  path="${ROOT}/${script}"
  echo "Checking ${script}"

  [[ -f "$path" ]] || { echo "ERROR: missing ${script}" >&2; exit 1; }

  bash -n "$path"

  # Read-only commands must not leak shell errors. Unescaped backticks or
  # `$(…)` inside an unquoted heredoc surface here as "command not found".
  for read_only in help info network-info client-config; do
    case "${script}:${read_only}" in
      redteam-modelctl.sh:client-config) continue ;;
    esac
    stderr_out="$(bash "$path" "$read_only" </dev/null 2>&1 >/dev/null || true)"
    if grep -qE 'command not found|unbound variable|syntax error' <<<"$stderr_out"; then
      echo "ERROR: ${script} ${read_only} emitted a shell error:" >&2
      echo "$stderr_out" >&2
      exit 1
    fi
  done

  bash "$path" help </dev/null >/dev/null

  # Branding + runtime facts: model, port, features, live state.
  info_out="$(bash "$path" info </dev/null 2>&1)"
  grep -q 'DGX Spark Controller' <<<"$info_out" ||
    { echo "ERROR: ${script} info is missing the banner" >&2; exit 1; }
  for field in 'Model     :' 'Runtime   :' 'Features  :' 'State     :'; do
    grep -q "$field" <<<"$info_out" ||
      { echo "ERROR: ${script} info is missing '${field}'" >&2; exit 1; }
  done

  bash "$path" network-info \
    --context 65536 \
    --port 18100 \
    --advertise-ip "$TEST_IP" \
    </dev/null >/dev/null

  case "$script" in
    redteam-modelctl.sh)
      bash "$path" client-config glm \
        --context 65536 \
        --port 18101 \
        --advertise-ip "$TEST_IP" \
        </dev/null >/dev/null
      ;;
    *)
      bash "$path" client-config \
        --context 65536 \
        --port 18101 \
        --advertise-ip "$TEST_IP" \
        </dev/null >/dev/null
      ;;
  esac

  if bash "$path" network-info \
    --port 70000 \
    --advertise-ip "$TEST_IP" \
    </dev/null >/dev/null 2>&1; then
    echo "ERROR: ${script} accepted invalid port 70000" >&2
    exit 1
  fi

  if bash "$path" network-info \
    --context 0 \
    --advertise-ip "$TEST_IP" \
    </dev/null >/dev/null 2>&1; then
    echo "ERROR: ${script} accepted zero context" >&2
    exit 1
  fi

  echo "OK: ${script}"
done

# Stacked controllers must not hard-code the author's username or ask for the
# cluster configuration on commands other than start/restart.
for script in "${stacked[@]}"; do
  path="${ROOT}/${script}"
  grep -q 'prompt_cluster_config' "$path" ||
    { echo "ERROR: ${script} has no cluster prompt" >&2; exit 1; }
  if bash "$path" info </dev/null 2>&1 | grep -q 'Cluster configuration'; then
    echo "ERROR: ${script} prompts for cluster config outside start/restart" >&2
    exit 1
  fi
done

# The bracketed letter keeps these patterns from matching this file itself.
# The banner credit carries the ".minidev" marker and is the only allowed hit.
if grep -rn 'nero[n]ain' --include='*.sh' "$ROOT" | grep -v 'nero[n]ain\.minidev'; then
  echo "ERROR: a controller still hard-codes the author's username" >&2
  exit 1
fi

python3 "${ROOT}/audit-controllers.py" "$ROOT"

echo "All ${#controllers[@]} DGX controllers passed static validation."
