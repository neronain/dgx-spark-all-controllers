#!/usr/bin/env bash
# Install every controller into a target directory, backing up existing copies.
# Designed by neronain · https://www.facebook.com/neronain.minidev
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-${HOME}}"
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="${TARGET}/controller-backups/${STAMP}"

controllers=(
  deepseek-v4-flash-nvfp4-stacked.sh
  gemma-4-26b-a4b-it-gguf-single.sh
  gemma-4-31b-it-uncensored-heretic-q8_0-dgx-spark.sh
  gemma-4-31b-it-uncensored-single.sh
  gemma4-26-a4b-q8xl-single.sh
  gemma4-31b-single.sh
  gemma4-31b-stacked.sh
  gpt-oss-120b-f16-single.sh
  llama33-70b-nvfp4-single.sh
  minimax-m27-luke-stacked.sh
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
  vllm-stackctl.sh
)

mkdir -p "$TARGET" "$BACKUP"

for name in "${controllers[@]}"; do
  source_path="${ROOT}/${name}"
  target_path="${TARGET}/${name}"

  [[ -f "$source_path" ]] || {
    echo "ERROR: missing package file ${source_path}" >&2
    exit 1
  }

  if [[ -e "$target_path" ]]; then
    cp -a "$target_path" "${BACKUP}/${name}"
  fi

  install -m 0755 "$source_path" "$target_path"
  bash -n "$target_path"

  echo "Installed: ${target_path}"
done

echo
echo "Installed ${#controllers[@]} controllers."
echo "Backup directory: ${BACKUP}"
echo "No running model was stopped or restarted."
echo
echo "Next:"
echo "  ${TARGET}/qwen3-coder-next-single.sh info    # model, port, features, state"
echo "  ${ROOT}/verify-all.sh"
