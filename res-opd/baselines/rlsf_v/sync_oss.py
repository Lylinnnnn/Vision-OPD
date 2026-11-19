#!/usr/bin/env python3
"""Upload, restore, and safely clean RLSF-V artifacts."""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath


MANIFEST = ".rlsf_v_complete.json"
MODELS = {"qwen2b", "qwen4b", "qwen8b", "internvl2b"}
WORK_ROOTS = {"rlsf_v_coco5k", "rlsf_v_coco5k_e1", "rlsf_v_smoke"}


def content_length(output):
    match = re.search(r"Content-Length\s*:\s*(\d+)", output)
    if not match:
        raise ValueError("ossutil stat did not report Content-Length")
    return int(match.group(1))


def require_ossutil():
    if shutil.which("ossutil") is None:
        raise RuntimeError("ossutil is not on PATH")


def oss_copy(source, destination):
    subprocess.run(["ossutil", "cp", str(source), str(destination), "-f"], check=True)


def oss_size(uri):
    result = subprocess.run(["ossutil", "stat", uri], check=True, capture_output=True, text=True)
    return content_length(result.stdout + result.stderr)


def upload_verified(path, uri):
    size = path.stat().st_size
    for attempt in range(2):
        oss_copy(path, uri)
        if oss_size(uri) == size:
            return
        if attempt == 1:
            raise RuntimeError(f"OSS size mismatch after retry: {uri}")


def valid_manifest(data, artifact):
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Invalid RLSF-V OSS manifest schema")
    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("RLSF-V OSS manifest has no files")
    seen = set()
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("Invalid RLSF-V OSS manifest entry")
        path = PurePosixPath(item["path"])
        if (path.is_absolute() or ".." in path.parts or path.as_posix() != item["path"]
                or item["path"] in ("", ".", MANIFEST)):
            raise ValueError(f"Unsafe RLSF-V OSS manifest path: {item['path']}")
        if path.as_posix() in seen or type(item.get("size")) is not int or item["size"] < 0:
            raise ValueError(f"Invalid RLSF-V OSS manifest entry: {item}")
        seen.add(path.as_posix())
    required = "adapter_config.json" if artifact == "adapter" else "config.json"
    weights = ("adapter_model.safetensors", "adapter_model.bin") if artifact == "adapter" else ()
    has_weights = bool(seen.intersection(weights)) if weights else any(
        re.fullmatch(r"(?:model(?:-\d+-of-\d+)?\.safetensors|pytorch_model(?:-\d+-of-\d+)?\.bin)", name)
        for name in seen
    )
    if required not in seen or not has_weights:
        raise ValueError(f"RLSF-V OSS manifest is missing {artifact} weights or {required}")
    return files


def owned_path(path, work_root, model, artifact):
    if model not in MODELS or artifact not in {"adapter", "merged"}:
        raise ValueError("Unknown RLSF-V model or artifact")
    root = Path(work_root).absolute()
    target = Path(path).absolute()
    if root.name not in WORK_ROOTS or root.is_symlink() or (root / model).is_symlink():
        raise ValueError(f"Refusing to touch unexpected RLSF-V work root: {root}")
    expected = root / model / artifact
    staged = (artifact == "merged" and target.parent == root / model
              and re.fullmatch(r"\.merged_stage_[A-Za-z0-9_-]+", target.name))
    if target != expected and not staged:
        raise ValueError(f"Refusing to touch artifact outside {expected}: {target}")
    if target.is_symlink():
        raise ValueError(f"Refusing to touch symlink: {target}")
    return target


def sync_tree(source, destination, artifact=None):
    source = Path(source)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if not destination.startswith("oss://"):
        raise ValueError("Destination must be an oss:// URL")
    require_ossutil()
    all_files = sorted(path for path in source.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in all_files):
        raise ValueError(f"Refusing to upload a symlink under {source}")
    files = [path for path in all_files if path.relative_to(source).as_posix() != MANIFEST]
    if not files:
        raise ValueError(f"No files to upload under {source}")
    manifest = {"schema_version": 1, "files": [
        {"path": path.relative_to(source).as_posix(), "size": path.stat().st_size}
        for path in files
    ]}
    if artifact:
        valid_manifest(manifest, artifact)
    for path in files:
        relative = path.relative_to(source).as_posix()
        size = path.stat().st_size
        upload_verified(path, destination.rstrip("/") + "/" + relative)
        print(f"VERIFIED {relative} {size} bytes", flush=True)
    with tempfile.TemporaryDirectory() as directory:
        marker = Path(directory) / MANIFEST
        marker.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        upload_verified(marker, destination.rstrip("/") + "/" + MANIFEST)
    print(f"OSS_SYNC=PASS files={len(files)} destination={destination}", flush=True)


def restore_tree(source, destination, artifact):
    if not source.startswith("oss://"):
        raise ValueError("Source must be an oss:// URL")
    require_ossutil()
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.restore-", dir=destination.parent))
    try:
        marker = staging / MANIFEST
        oss_copy(source.rstrip("/") + "/" + MANIFEST, marker)
        files = valid_manifest(json.loads(marker.read_text(encoding="utf-8")), artifact)
        if artifact == "adapter":
            files = [item for item in files if not re.fullmatch(
                r"checkpoint-\d+", PurePosixPath(item["path"]).parts[0]
            )]
        for item in files:
            target = staging / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            oss_copy(source.rstrip("/") + "/" + item["path"], target)
            if target.stat().st_size != item["size"]:
                raise RuntimeError(f"Downloaded OSS size mismatch: {item['path']}")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"OSS_RESTORE=PASS files={len(files)} destination={destination}", flush=True)


def remove_owned_tree(path, work_root, model, artifact):
    target = owned_path(path, work_root, model, artifact)
    if not target.is_dir():
        raise FileNotFoundError(target)
    shutil.rmtree(target)
    print(f"LOCAL_CLEANUP=PASS path={target}", flush=True)


def prune_training_checkpoints(source):
    if not (source / "adapter_config.json").is_file() or not any(
        (source / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise RuntimeError(f"Refusing to prune without a final adapter in {source}")
    checkpoints = sorted(path for path in source.iterdir() if re.fullmatch(r"checkpoint-\d+", path.name))
    for path in checkpoints:
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError(f"Refusing to prune unexpected checkpoint path: {path}")
        shutil.rmtree(path)
        print(f"PRUNED {path}", flush=True)
    print(f"CHECKPOINT_PRUNE=PASS count={len(checkpoints)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("upload", "restore", "cleanup"), default="upload")
    parser.add_argument("--source")
    parser.add_argument("--destination")
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--model", choices=sorted(MODELS))
    parser.add_argument("--artifact", choices=("adapter", "merged"))
    parser.add_argument("--delete-source-after-sync", action="store_true")
    parser.add_argument("--prune-checkpoints", action="store_true")
    args = parser.parse_args()
    if args.action == "upload":
        if not args.source or not args.destination:
            parser.error("upload requires --source and --destination")
        if args.delete_source_after_sync:
            if not all((args.work_root, args.model, args.artifact)):
                parser.error("deletion requires --work-root, --model, and --artifact")
            owned_path(args.source, args.work_root, args.model, args.artifact)
        sync_tree(Path(args.source), args.destination, args.artifact)
        if args.delete_source_after_sync:
            remove_owned_tree(args.source, args.work_root, args.model, args.artifact)
        elif args.prune_checkpoints:
            prune_training_checkpoints(Path(args.source))
    elif args.action == "restore":
        if not all((args.source, args.destination, args.work_root, args.model, args.artifact)):
            parser.error("restore requires source, destination, work-root, model, and artifact")
        owned_path(args.destination, args.work_root, args.model, args.artifact)
        restore_tree(args.source, args.destination, args.artifact)
    else:
        if not all((args.destination, args.work_root, args.model, args.artifact)):
            parser.error("cleanup requires destination, work-root, model, and artifact")
        remove_owned_tree(args.destination, args.work_root, args.model, args.artifact)


if __name__ == "__main__":
    main()
