#!/usr/bin/env python3
"""Audit DGX Spark controller shell scripts for portability and Bash hazards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


SCRIPT_VERSION_DECL = re.compile(r'(?m)^SCRIPT_VERSION="\$\{SCRIPT_VERSION:-[0-9]+\.[0-9]+\.[0-9]+\}"')
BANNER_DEF = re.compile(r"(?m)^banner\(\) \{")
INFO_DEF = re.compile(r"(?m)^info\(\) \{")
INFO_DISPATCH = re.compile(r"(?m)^\s*info\|banner\)")
AUTHOR_USERNAME = re.compile(r"neronain")
# The banner credit legitimately names the author; it carries this marker.
AUTHOR_CREDIT_MARKER = "neronain.minidev"
CLUSTER_PROMPT = re.compile(r"(?m)^prompt_cluster_config\(\) \{")
PURE_NUMERIC_SEPARATOR = re.compile(r"\b\d+(?:_\d+)+\b")
PIPEFAIL_GREP_Q = re.compile(r"\|[^\n]*\bgrep\b[^\n]*-[A-Za-z]*q[A-Za-z]*")
FIXED_API_PORT = re.compile(r'(?m)^API_PORT="(?!\$\{API_PORT:-)')
FIXED_CONTEXT = re.compile(r'(?m)^(?:CTX_SIZE|MAX_MODEL_LEN)="(?!\$\{(?:CTX_SIZE|MAX_MODEL_LEN):-)')
FIXED_SINGLE_MASTER = re.compile(r'(?m)^MASTER_IP="[^"]+"')
DIRECT_FIRST_IP = re.compile(r"hostname -I[^\n]*awk[^\n]*\{print \$1\}")


def line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def audit_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    findings: list[dict[str, Any]] = []

    proc = subprocess.run(
        ["bash", "-n", str(path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode:
        findings.append({
            "severity": "error",
            "kind": "bash-syntax",
            "line": None,
            "detail": proc.stderr.strip(),
        })

    for match in PURE_NUMERIC_SEPARATOR.finditer(text):
        findings.append({
            "severity": "error",
            "kind": "numeric-separator",
            "line": line_of(text, match.start()),
            "detail": match.group(0),
        })

    if "set -Eeuo pipefail" in text or "set -o pipefail" in text:
        for match in PIPEFAIL_GREP_Q.finditer(text):
            findings.append({
                "severity": "warning",
                "kind": "pipefail-grep-q",
                "line": line_of(text, match.start()),
                "detail": match.group(0).strip(),
            })

    is_stacked = (
        "stacked" in path.name.lower()
        or "WORKER_IP=" in text
        or "TENSOR_PARALLEL_SIZE=\"2\"" in text
    )

    if not is_stacked:
        match = FIXED_SINGLE_MASTER.search(text)
        if match:
            findings.append({
                "severity": "warning",
                "kind": "fixed-single-master-ip",
                "line": line_of(text, match.start()),
                "detail": match.group(0),
            })

    match = FIXED_API_PORT.search(text)
    if match:
        findings.append({
            "severity": "warning",
            "kind": "non-overridable-api-port",
            "line": line_of(text, match.start()),
            "detail": match.group(0),
        })

    match = FIXED_CONTEXT.search(text)
    if match:
        findings.append({
            "severity": "warning",
            "kind": "non-overridable-context",
            "line": line_of(text, match.start()),
            "detail": match.group(0),
        })

    if "--context" not in text:
        findings.append({
            "severity": "warning",
            "kind": "missing-context-option",
            "line": None,
            "detail": "No --context option",
        })

    if "--port" not in text:
        findings.append({
            "severity": "warning",
            "kind": "missing-port-option",
            "line": None,
            "detail": "No --port option",
        })

    if "network-info" not in text or "detect_advertise_ip" not in text:
        findings.append({
            "severity": "warning",
            "kind": "missing-network-selection",
            "line": None,
            "detail": "Missing network-info or route-based advertised IP selection",
        })

    if not SCRIPT_VERSION_DECL.search(text):
        findings.append({
            "severity": "warning",
            "kind": "missing-script-version",
            "line": None,
            "detail": 'No overridable SCRIPT_VERSION="${SCRIPT_VERSION:-X.Y.Z}"',
        })

    if not (BANNER_DEF.search(text) and INFO_DEF.search(text)
            and INFO_DISPATCH.search(text)):
        findings.append({
            "severity": "warning",
            "kind": "missing-banner-info",
            "line": None,
            "detail": "Missing banner()/info() or the info|banner) dispatch entry",
        })

    for number, line in enumerate(text.splitlines(), start=1):
        if AUTHOR_CREDIT_MARKER in line:
            continue
        if AUTHOR_USERNAME.search(line):
            findings.append({
                "severity": "warning",
                "kind": "hard-coded-author-username",
                "line": number,
                "detail": "Hard-coded username; default to ${USER:-$(id -un)}",
            })

    if is_stacked and not CLUSTER_PROMPT.search(text):
        findings.append({
            "severity": "warning",
            "kind": "missing-cluster-prompt",
            "line": None,
            "detail": "Stacked controller has no prompt_cluster_config()",
        })

    for match in DIRECT_FIRST_IP.finditer(text):
        context = text[max(0, match.start() - 1200):match.start()]
        if "detect_advertise_ip()" not in context[-1200:]:
            findings.append({
                "severity": "warning",
                "kind": "first-hostname-ip",
                "line": line_of(text, match.start()),
                "detail": match.group(0),
            })

    return {
        "file": path.name,
        "path": str(path),
        "status": "ok" if not findings else "findings",
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default=".")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = Path(args.directory).resolve()
    excluded = {"verify-all.sh", "install-canonical.sh"}
    paths = sorted(
        path for path in root.rglob("*.sh")
        if path.name not in excluded
    )
    if not paths:
        raise SystemExit(f"No .sh files found under {root}")

    reports = [audit_file(path) for path in paths]
    errors = sum(
        finding["severity"] == "error"
        for report in reports
        for finding in report["findings"]
    )
    warnings = sum(
        finding["severity"] == "warning"
        for report in reports
        for finding in report["findings"]
    )

    if args.json:
        print(json.dumps({
            "root": str(root),
            "files": reports,
            "error_count": errors,
            "warning_count": warnings,
        }, indent=2))
    else:
        for report in reports:
            if not report["findings"]:
                print(f"OK       {report['file']}")
                continue
            print(f"FINDINGS {report['file']}")
            for item in report["findings"]:
                location = f":{item['line']}" if item["line"] else ""
                print(
                    f"  {item['severity'].upper():7} "
                    f"{item['kind']}{location}: {item['detail']}"
                )
        print()
        print(
            f"Audited {len(reports)} scripts: "
            f"errors={errors}, warnings={warnings}"
        )

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
