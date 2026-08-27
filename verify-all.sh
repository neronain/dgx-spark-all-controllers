#!/usr/bin/env bash
# Static validation for every DGX Spark controller in this repository.
# Designed by neronain · https://www.facebook.com/neronain.minidev
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_IP="${TEST_IP:-192.0.2.10}"

# Controllers are *discovered*, not listed.
#
# The list used to be written out by hand, and it fell behind: on 2026-08-28 the
# directory held 35 controllers while this file named 22, so 13 of them — every
# one added from an LMDS bundle — was never verified by the script whose whole
# job is to verify them. A list that has to be updated by hand is a list that
# eventually disagrees with the directory, and it fails silently in the
# direction that matters: new files go unchecked.
#
# Anything ending in .sh that is not one of this repository's own tools is a
# controller and gets the full treatment.
tools=("verify-all.sh" "install-canonical.sh")

singles=()
stacked=()
for path in "$ROOT"/*.sh; do
  name="$(basename "$path")"
  skip=""
  for tool in "${tools[@]}"; do [[ "$name" == "$tool" ]] && skip=1; done
  [[ -n "$skip" ]] && continue
  # Stacked controllers are the ones that ask for cluster addresses. Detect that
  # property directly rather than trusting the filename — a bundle generated for
  # two nodes may not be named "-stacked".
  if grep -q 'prompt_cluster_config' "$path"; then
    stacked+=("$name")
  else
    singles+=("$name")
  fi
done

controllers=("${singles[@]}" "${stacked[@]}")

if (( ${#controllers[@]} == 0 )); then
  echo "ERROR: no controllers found in ${ROOT}" >&2
  exit 1
fi
echo "Found ${#controllers[@]} controllers (${#singles[@]} single-node, ${#stacked[@]} stacked)"
echo

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
  # Either banner is valid. The hand-written controllers print "DGX Spark
  # Controller · vX.Y.Z"; the ones LMDS generates print "LMDS controller · vX.Y".
  # Both carry the same four lines a student actually reads — version, model,
  # runtime + features, author — and this repository accepts generated
  # controllers on purpose (see ADDING-GENERATED-CONTROLLERS.md). Demanding one
  # product string meant every generated controller failed here, which is why
  # they were quietly left out of the hand-written list instead.
  grep -qE 'DGX Spark Controller|LMDS controller' <<<"$info_out" ||
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
