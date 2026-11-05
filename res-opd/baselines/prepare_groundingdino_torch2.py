#!/usr/bin/env python3
"""Apply the GroundingDINO PyTorch dispatch compatibility fix to the local submodule."""

from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "third_party/baselines/HALC/decoder_zoo/GroundingDINO/groundingdino"
    / "models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu"
)
DISPATCH_NAMES = ("ms_deform_attn_forward_cuda", "ms_deform_attn_backward_cuda")


def patch_source(source):
    updated = source
    changed = 0
    for name in DISPATCH_NAMES:
        old = f'AT_DISPATCH_FLOATING_TYPES(value.type(), "{name}"'
        new = f'AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "{name}"'
        if source.count(old) == 1 and source.count(new) == 0:
            updated = updated.replace(old, new)
            changed += 1
        elif source.count(old) == 0 and source.count(new) == 1:
            continue
        else:
            raise RuntimeError(f"Unexpected GroundingDINO dispatch call: {name}")
    return updated, changed


def main():
    if not SOURCE.is_file():
        raise SystemExit(f"GroundingDINO source missing: {SOURCE}")
    original = SOURCE.read_text()
    updated, changed = patch_source(original)
    if changed:
        SOURCE.write_text(updated)
    print(f"GroundingDINO dispatch fix: {'patched' if changed else 'already applied'} ({changed}/2)")


if __name__ == "__main__":
    main()
