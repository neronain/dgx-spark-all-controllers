#!/usr/bin/env bash
# Install every controller in this directory into a target directory, backing up existing copies.
# The set is discovered from *.sh (minus this repository's own tools) — nothing to keep in sync.
# Designed by neronain · https://www.facebook.com/neronain.minidev
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-${HOME}}"
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="${TARGET}/controller-backups/${STAMP}"

# Controllers are discovered, not listed.
#
# The hand-written list below fell behind twice: on 2026-08-28 it named 21 files while the
# directory held 35, so `install-canonical.sh` silently installed 60% of the collection and
# reported success. verify-all.sh already discovers controllers the same way; the two must agree.
tools=("verify-all.sh" "install-canonical.sh")
controllers=()
for path in "$ROOT"/*.sh; do
  name="$(basename "$path")"
  skip=""
  for tool in "${tools[@]}"; do [[ "$name" == "$tool" ]] && skip=1; done
  [[ -n "$skip" ]] && continue
  controllers+=("$name")
done
(( ${#controllers[@]} > 0 )) || { echo "ERROR: no controllers found in ${ROOT}" >&2; exit 1; }

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
