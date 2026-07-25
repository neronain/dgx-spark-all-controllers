# Changelog

## 3.1.0 — 2026-07-25

Repository-wide cleanup and standardization. Designed by neronain · <https://www.facebook.com/neronain.minidev>

### Added

- ASCII banner (Metasploit style) in all 21 controllers, with a designer credit line.
- `info` command (alias `banner`) in all 21 controllers: model label, model ID, runtime, supported features, context, advertised API v1 URL, and live state with the port. State comes from a real `/health` probe.
- Interactive cluster configuration for all 4 stacked controllers: `start` and `restart` ask for the head node IP, worker node IP, and SSH user, each defaulting to the current value. Only prompts on a TTY, so automation and cron stay unattended.
- `--advertise-ip` and `--interface` options for the two controllers that lacked them.
- Audit rules: `missing-script-version`, `missing-banner-info`, `hard-coded-author-username`, `missing-cluster-prompt`.
- `verify-all.sh` now covers all 21 controllers, exercises `info`, asserts stacked controllers do not prompt outside `start`/`restart`, and fails if any controller hard-codes the author's username.

### Changed

- `SCRIPT_VERSION` is now 3.1.0 across every controller.
- `SSH_USER` defaults to `${USER:-$(id -un)}` everywhere; no controller hard-codes a Linux username.
- Port validation (1..65535) and context validation (> 0) added wherever it was missing or weaker.
- Container, runtime, port, and model-file checks rewritten to store command output before matching, replacing `producer | grep -q` pipelines.
- `install-canonical.sh` installs all 21 controllers.

### Fixed

- `gemma-4-31b-it-uncensored-heretic-q8_0-dgx-spark.sh`: `local rc=$?` read the status of the enclosing `if` construct instead of `wait_for_health`, leaving the startup-timeout branch unreachable and misreporting a timeout as a crash.

### Removed

- `gemma4-26b-a4b-q8xl-single.sh` — superseded 1.0.2 duplicate of `gemma4-26-a4b-q8xl-single.sh`.
- `vllm-stackctl(1).sh` — duplicate download of `vllm-stackctl.sh`, confirmed a strict subset.

### Renamed

- `deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh` → `deepseek-v4-flash-nvfp4-stacked.sh`
- `qwen3-vl-32b-thinking-single-fixed-v5.sh` → `qwen3-vl-32b-thinking-single.sh`

### Documentation

- `README.md`, `AUDIT_REPORT.md`, `MANIFEST.txt`, `SKILL_UPDATE.md`, `step.md`, and `skills_Strack.md` rewritten for the 21-controller v3.1.0 layout and the new commands.
- `PACKAGE_SHA256SUMS` regenerated.

## 3.0.0

- Audited 12 canonical DGX Spark controllers.
- Removed Bash pure-numeric underscore literals.
- Standardized `--context`, `--port`, `--bind`, `--advertise-ip`, and `--interface`.
- Added environment overrides for context, port, bind address, advertise address, API key, and client budgets.
- Added route-based advertised IP selection.
- Removed fixed `MASTER_IP` from single-node controllers.
- Kept cluster addresses separate from public API addresses in stacked controllers.
- Added `network-info`.
- Added automatic client input-budget calculation.
- Fixed pipefail-sensitive `grep -q` pipelines.
- Added audit, verification, and safe installation helpers.
- Updated the DGX Spark Model Deployer skill rules and templates.
