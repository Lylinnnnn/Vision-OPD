# External Baselines

This directory contains repository-owned adapters and evaluation contracts for
external hallucination-mitigation baselines. The upstream implementations are
kept as read-only, commit-pinned Git submodules under
`res-opd/third_party/baselines/`.

## Layout

- `registry.json`: authoritative source revisions and integration status.
- `protocols/`: shared evaluation contracts consumed by baseline adapters.
- `check_sources.py`: validates submodule revisions, URLs, and licenses.
- `../third_party/baselines/`: unmodified upstream source checkouts.

Initialize the sources with:

```bash
git submodule update --init --recursive
python3 res-opd/baselines/check_sources.py
```

Do not install an upstream repository's requirements into the main project
environment. OPERA and HALC vendor old Transformers implementations, while VCD
patches the generation stack. Each method therefore needs a Qwen3VL-compatible
adapter or an isolated environment before it can be marked runnable.

## Integration Status

| Method | Upstream source | Current status | Qwen3VL support upstream |
| --- | --- | --- | --- |
| VCD | `third_party/baselines/VCD` | adapter required | No |
| OPERA | `third_party/baselines/OPERA` | source only | No |
| HALC | `third_party/baselines/HALC` | source only | No |

`source only` means that the official implementation is available for audit,
but must not be invoked from the main environment. It does not mean that an
experimental result has been reproduced.

## Evaluation Boundary

Adapters must write the standard CHAIR `eval_results.jsonl` schema documented
in `protocols/chair_official.json`. Scoring is then performed by the repository
implementation in `res-opd/eval/eval_chair.py`. External repositories' local
CHAIR variants are not used. This keeps the image set, prompt, generation
accounting, and official CHAIR metric implementation identical across methods.

Baseline outputs must use a dedicated result suffix such as
`chair_external_vcd_official`. They must not write into an existing base-model,
training, or paper-result directory.
