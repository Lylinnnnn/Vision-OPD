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
| HALC | `third_party/baselines/HALC` | Qwen3VL smoke and sharded adapter, unvalidated | No |

InternVL3-2B-hf adapters now share the same official CHAIR prompt/scorer and
use separate `chair_external_*_internvl3_adapted_official` result directories.
They are not upstream-supported implementations. VCD uses identical HF
InternVL3 prompts with original/noised visual patches; OPERA uses the
language-model attention and contiguous image-token span; HALC reuses the
experimental focal-crop/BLIP correction with an InternVL3 input bridge.
Only code-level checks have run locally. Verify each on 1-5 images with a
GPU before any full evaluation, and check VCD's paired branch alignment,
OPERA's available attention and recorded rollback count, and HALC's triggered/changed
token counts. A passing CHAIR score alone does not establish algorithmic
equivalence to the upstream baseline.

For a single-image smoke on the server, use the same installed dependencies
and model path as the InternVL3 base evaluation:

```bash
MODEL=/path/to/InternVL3-2B-hf
PYTHON_BIN="$CONDA_PREFIX/bin/python3"
$PYTHON_BIN res-opd/baselines/eval_vcd_chair.py --model-family internvl3 \
  --model-path "$MODEL" --test-json res-opd/data/test_1000.json \
  --output-dir res-opd/eval_results/external_baselines/smoke/internvl3_vcd \
  --max-samples 1 --max-new-tokens 128
$PYTHON_BIN res-opd/baselines/eval_opera_chair.py --model-family internvl3 \
  --model-path "$MODEL" --test-json res-opd/data/test_1000.json \
  --output-dir res-opd/eval_results/external_baselines/smoke/internvl3_opera \
  --max-samples 1 --max-new-tokens 128
$PYTHON_BIN res-opd/baselines/eval_halc_chair.py --model-family internvl3 \
  --model-path "$MODEL" --detector-config /path/to/GroundingDINO_config.py \
  --detector-weights /path/to/groundingdino.pth \
  --blip-model-path /path/to/blip-itm-base-coco \
  --output-dir res-opd/eval_results/external_baselines/smoke/internvl3_halc \
  --max-samples 5 --max-new-tokens 128 --require-trigger --require-change
```

After smoke validation, set `MODEL_FAMILY=internvl3` for the existing
eight-GPU runners. VCD and OPERA additionally require `VCD_ENABLE_FULL_RUN=1`
and `OPERA_ENABLE_FULL_RUN=1`, respectively; HALC still requires
`HALC_ENABLE_FULL_RUN=1`.

The HALC Qwen3VL adapter is an experimental translation of the focal-view
contrast and visual beam-selection procedure. It is not a line-for-line port
of the upstream Transformers 4.36.2 generation patch: its object-word trigger
operates on Qwen tokenizer tokens, and it does not include the upstream
HALC/DoLa coupling. Thus the output must not be reported as an official HALC
result. A GPU smoke must first confirm nonzero focal triggers and changed tokens;
algorithmic equivalence needs separate validation. In particular, its BLIP
image-text matching score is not yet verified to match the upstream scoring
implementation. Full evaluation uses the isolated
`chair_external_halc_qwen3vl_adapted_official` result suffix, where `official`
refers only to the CHAIR scorer, not to the HALC implementation.

Before installing the pinned GroundingDINO source with a recent PyTorch, run
`python res-opd/baselines/prepare_groundingdino_torch2.py`. This applies the
two dispatch-call compatibility edits from GroundingDINO PR #415 to the local
HALC submodule; the upstream commit remains pinned.

After single-image GPU smoke, the sharded runner is
`run_halc_chair_sharded_8gpu.sh <model> <DINO config> <DINO weights> <BLIP ITM model>`.
It requires `HALC_ENABLE_FULL_RUN=1` because the adaptation has not been
validated against upstream HALC. It keeps per-shard logs and generations for
inspection. `HALC_MAX_SAMPLES=0` evaluates the complete test set;
`HALC_MAX_NEW_TOKENS=8192` follows the shared CHAIR generation cap.

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
