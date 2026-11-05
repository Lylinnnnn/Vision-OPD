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
| VCD | `third_party/baselines/VCD` | Qwen3VL CHAIR adapter | No |
| OPERA | `third_party/baselines/OPERA` | Qwen3VL CHAIR adapter | No |
| HALC | `third_party/baselines/HALC` | Qwen3VL single-GPU smoke adapter, unvalidated | No |

The HALC Qwen3VL adapter is an experimental translation of the focal-view
contrast and visual beam-selection procedure. It is not a line-for-line port
of the upstream Transformers 4.36.2 generation patch: its object-word trigger
operates on Qwen tokenizer tokens, and it does not include the upstream
HALC/DoLa coupling. Thus the output must not be reported as an official HALC
result. A GPU smoke must first confirm nonzero focal triggers and changed tokens;
algorithmic equivalence needs separate validation. The evaluator deliberately
uses the isolated `chair_external_halc_smoke` result suffix.

## Evaluation Boundary

Adapters must write the standard CHAIR `eval_results.jsonl` schema documented
in `protocols/chair_official.json`. Scoring is then performed by the repository
implementation in `res-opd/eval/eval_chair.py`. External repositories' local
CHAIR variants are not used. This keeps the image set, prompt, generation
accounting, and official CHAIR metric implementation identical across methods.

Baseline outputs must use a dedicated result suffix such as
`chair_external_vcd_official`. They must not write into an existing base-model,
training, or paper-result directory.

The Qwen3VL VCD adapter can be launched with:

```bash
bash res-opd/baselines/run_vcd_chair_sharded_8gpu.sh /path/to/Qwen3VL-2B-Instruct
```

It uses one Hugging Face process per GPU because VCD requires original/noisy
visual logits at every decoding step and cannot be represented by the current
vLLM server interface. Its defaults match the upstream Qwen VCD example:
`alpha=1`, `beta=0.1`, `noise_step=500`, and seeded sampling at temperature 1.

The Qwen3VL OPERA adapter can be launched with:

```bash
bash res-opd/baselines/run_opera_chair_sharded_8gpu.sh /path/to/Qwen3VL-2B-Instruct
```

It ports the pinned upstream OPERA beam-search penalty and rollback procedure
to the Transformers 5.5 custom-generation interface. The default settings are
the upstream values: five beams, `scale_factor=50`, `threshold=15`, five
attention candidates, `penalty_weights=1`, and `window_size=512`. It uses one
eager-attention Hugging Face process per GPU because OPERA needs language-model
self-attention and candidate forwards during decoding.
