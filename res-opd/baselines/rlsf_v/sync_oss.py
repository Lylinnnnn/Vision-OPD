#!/usr/bin/env python3
"""Upload an RLSF-V artifact tree to OSS and verify each object's byte size."""

import argparse
import re
import shutil
import subprocess
from pathlib import Path


def content_length(output):
    match = re.search(r"Content-Length\s*:\s*(\d+)", output)
    if not match:
        raise ValueError("ossutil stat did not report Content-Length")
    return int(match.group(1))


def sync_tree(source, destination):
    if not source.is_dir():
        raise FileNotFoundError(source)
    if not destination.startswith("oss://"):
        raise ValueError("Destination must be an oss:// URL")
    if shutil.which("ossutil") is None:
        raise RuntimeError("ossutil is not on PATH")
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"No files to upload under {source}")
    for path in files:
        if path.is_symlink():
            raise ValueError(f"Refusing to upload symlink: {path}")
        target = destination.rstrip("/") + "/" + path.relative_to(source).as_posix()
        size = path.stat().st_size
        for attempt in range(2):
            subprocess.run(["ossutil", "cp", str(path), target, "-f"], check=True)
            result = subprocess.run(
                ["ossutil", "stat", target], check=True, capture_output=True, text=True
            )
            if content_length(result.stdout + result.stderr) == size:
                break
            if attempt == 1:
                raise RuntimeError(f"OSS size mismatch after retry: {target}")
        print(f"VERIFIED {path.relative_to(source)} {size} bytes", flush=True)
    print(f"OSS_SYNC=PASS files={len(files)} destination={destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    sync_tree(args.source, args.destination)


if __name__ == "__main__":
    main()
