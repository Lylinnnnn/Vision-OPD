#!/usr/bin/env python3

import json
import subprocess
import sys
from pathlib import Path


BASELINES_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASELINES_DIR.parents[1]
REGISTRY_PATH = BASELINES_DIR / "registry.json"
GITMODULES_PATH = REPO_ROOT / ".gitmodules"


def run_git(*args, cwd=None):
    result = subprocess.run(
        ["git", *args],
        cwd=cwd or REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def main():
    registry = json.loads(REGISTRY_PATH.read_text())
    failures = []

    for key, entry in registry["baselines"].items():
        source = REPO_ROOT / entry["source_path"]
        if not source.is_dir():
            failures.append(f"{key}: source is not initialized: {source}")
            continue

        actual_commit = run_git("rev-parse", "HEAD", cwd=source)
        if actual_commit != entry["commit"]:
            failures.append(
                f"{key}: commit mismatch: expected {entry['commit']}, got {actual_commit}"
            )

        section = f"submodule.{entry['source_path']}.url"
        configured_url = run_git("config", "-f", str(GITMODULES_PATH), "--get", section)
        if configured_url != entry["repository"]:
            failures.append(
                f"{key}: submodule URL mismatch: expected {entry['repository']}, "
                f"got {configured_url}"
            )

        for relative_license in entry["license_files"]:
            if not (source / relative_license).is_file():
                failures.append(f"{key}: missing license file: {relative_license}")

        if actual_commit == entry["commit"]:
            print(f"[PASS] {key}: {actual_commit}")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}", file=sys.stderr)
        return 1

    print("EXTERNAL_BASELINE_SOURCES=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
