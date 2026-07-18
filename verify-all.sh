#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_IP="${TEST_IP:-192.0.2.10}"

controllers=(
  gemma4-26-a4b-q8xl-single.sh
  gemma4-31b-single.sh
  gemma4-31b-stacked.sh
  gpt-oss-120b-f16-single.sh
  llama33-70b-nvfp4-single.sh
  minimax-m27-luke-stacked.sh
  nemotron-3-super-single.sh
  nemotron-omni-aeon-single.sh
  qwen3-coder-next-single.sh
  qwen36-hauhau-q6kp-single.sh
  redteam-modelctl.sh
  vllm-stackctl.sh
)

for script in "${controllers[@]}"; do
  path="${ROOT}/${script}"
  echo "Checking ${script}"

  bash -n "$path"
  bash "$path" help >/dev/null

  bash "$path" network-info \
    --context 65536 \
    --port 18100 \
    --advertise-ip "$TEST_IP" \
    >/dev/null

  case "$script" in
    redteam-modelctl.sh)
      bash "$path" client-config glm \
        --context 65536 \
        --port 18101 \
        --advertise-ip "$TEST_IP" \
        >/dev/null
      ;;
    *)
      bash "$path" client-config \
        --context 65536 \
        --port 18101 \
        --advertise-ip "$TEST_IP" \
        >/dev/null
      ;;
  esac

  if bash "$path" network-info \
    --port 70000 \
    --advertise-ip "$TEST_IP" \
    >/dev/null 2>&1; then
    echo "ERROR: ${script} accepted invalid port 70000" >&2
    exit 1
  fi

  if bash "$path" network-info \
    --context 0 \
    --advertise-ip "$TEST_IP" \
    >/dev/null 2>&1; then
    echo "ERROR: ${script} accepted zero context" >&2
    exit 1
  fi

  echo "OK: ${script}"
done

python3 "${ROOT}/audit-controllers.py" "$ROOT"

echo "All canonical DGX controllers passed static validation."
