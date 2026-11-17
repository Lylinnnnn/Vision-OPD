# RLSF-V LLaVA Series

This directory contains the LLaVA-specific data generation, DPO data conversion, and LoRA-DPO training pipeline for RLSF-V.

For environment setup, see [`../README.md`](../README.md).

The training wrapper activates `rlsf-v-train` by default. Set `SKIP_CONDA_ACTIVATE=1` after activating the environment yourself if needed.

## Directory Layout

```text
llava_series/
  RLSF-V-llava.sh              # LLaVA LoRA-DPO training entry point
  generate/
    generate_llava.sh          # Multi-GPU answer/logits generation
    convert_llava.py           # Convert generated answers/logits to DPO JSON
    run_llava_dpo_pipeline.sh  # Generate answers/logits and then convert to DPO data
    llava-v1.5-7b/             # Default 7B generated outputs
    llava-v1.5-13b/            # Default 13B generated outputs
  checkpoints/                 # Default training output root
```

## Required Inputs

By default, the scripts expect:

- Base models:
  - `liuhaotian/llava-v1.5-7b`
  - `liuhaotian/llava-v1.5-13b`
- Question JSONL:
  - `../RLAIF_dataset/rlaif10k.jsonl`
- Image folder:
  - `../RLAIF_dataset`

If your paths are different, override them with environment variables:

```bash
MODEL_PATH=liuhaotian/llava-v1.5-7b \
QUESTION_FILE=/path/to/rlaif10k.jsonl \
bash generate/generate_llava.sh 7b
```

For training, override the base model with `INPUT_MODEL`:

```bash
INPUT_MODEL=liuhaotian/llava-v1.5-7b \
CONDA_ENV=rlsf-v-train \
bash RLSF-V-llava.sh 7b
```

## Step 1: Generate Answers and Logits

Generate multi-sample answers and top-k logits for LLaVA-7B:

```bash
bash generate/generate_llava.sh 7b
```

Generate for LLaVA-13B:

```bash
bash generate/generate_llava.sh 13b
```

Defaults:

- GPUs: `CUDA_VISIBLE_DEVICES=0,1,2,3`
- Number of workers/chunks: one per visible GPU
- Samples per question: `NUM_SAMPLES=10`
- Top-k logits per generated token: `TOP_K_LOGITS=50`
- Temperature: `TEMPERATURE=1.0`
- Batch size:
  - 7B: `BATCH_SIZE=10`
  - 13B: `BATCH_SIZE=5`

Default outputs:

```text
generate/llava-v1.5-7b/answers.jsonl
generate/llava-v1.5-7b/logits_merged.pt

generate/llava-v1.5-13b/answers.jsonl
generate/llava-v1.5-13b/logits_merged.pt
```

Use `DRY_RUN=1` to print the resolved configuration without starting inference:

```bash
DRY_RUN=1 bash generate/generate_llava.sh 7b
```

## Step 2: Convert to DPO Data

Convert generated answers and logits into DPO training data:

```bash
python generate/convert_llava.py \
  --name llava-v1.5-7b \
  --model-path /path/to/llava-v1.5-7b \
  --input-jsonl generate/llava-v1.5-7b/answers.jsonl \
  --logits-path generate/llava-v1.5-7b/logits_merged.pt
```

The converter uses the fuzzy semantic entropy score by default and writes:

```text
generate/llava-v1.5-7b/llava-v1.5-7b.jsonl
generate/llava-v1.5-7b/llava-v1.5-7b_dpo_data.json
```

Each DPO example is a JSON object with:

```json
{
  "img_path": "...",
  "prompt": "...",
  "chosen": "...",
  "rejected": "..."
}
```

## One-Command Generation and Conversion

Run generation and conversion together:

```bash
bash generate/run_llava_dpo_pipeline.sh 7b
bash generate/run_llava_dpo_pipeline.sh 13b
```

Only run conversion from existing generated files:

```bash
RUN_GENERATE=0 bash generate/run_llava_dpo_pipeline.sh 7b
```

Only run generation:

```bash
RUN_CONVERT=0 bash generate/run_llava_dpo_pipeline.sh 7b
```

## Step 3: Train LLaVA with RLSF-V DPO Data

Train LLaVA-7B:

```bash
CONDA_ENV=rlsf-v-train bash RLSF-V-llava.sh 7b
```

Train LLaVA-13B:

```bash
CONDA_ENV=rlsf-v-train bash RLSF-V-llava.sh 13b
```

The training script defaults to:

- GPUs: `CUDA_VISIBLE_DEVICES=0,1,2,3`
- Global batch size: `GLOBAL_BATCH_SIZE=32`
- Per-device batch size: `PER_DEVICE_BATCH_SIZE=8`
- Gradient accumulation: computed from global batch size, per-device batch size, and GPU count
- Epochs: `NUM_TRAIN_EPOCHS=2`
- Learning rate: `LEARNING_RATE=4e-6`
- DPO beta: `BETA=0.1`
- LoRA: `LORA_R=256`, `LORA_ALPHA=512`
- DeepSpeed config: `accelerate_configs/zero2.json`

Default training data:

```text
generate/llava-v1.5-7b/llava-v1.5-7b_dpo_data.json
generate/llava-v1.5-13b/llava-v1.5-13b_dpo_data.json
```

Default checkpoints:

```text
checkpoints/llava-v1.5-7b/
checkpoints/llava-v1.5-13b/
```

The final LoRA adapter is saved as `adapter_model.safetensors`. To avoid slow network-filesystem writes during DeepSpeed shutdown, the script saves to `/tmp/rlsfv_llava_outputs` first and then copies the completed adapter to `checkpoints/`.

Use a custom DPO file:

```bash
TRAINING_DATA_PATH=/path/to/custom_dpo_data.json \
RUN_NAME=my-llava-run \
CONDA_ENV=rlsf-v-train \
bash RLSF-V-llava.sh 7b
```

Train every DPO JSON in a directory:

```bash
DATA_DIR=/path/to/dpo_json_dir \
CONDA_ENV=rlsf-v-train \
bash RLSF-V-llava.sh 7b
```

Print the resolved DeepSpeed command without training:

```bash
DRY_RUN=1 CONDA_ENV=rlsf-v-train bash RLSF-V-llava.sh 7b
```

## Useful Overrides

Common environment variables:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
MODEL_PATH=/path/to/model              # generation
INPUT_MODEL=/path/to/model             # training
QUESTION_FILE=/path/to/questions.jsonl
OUTPUT_DIR=/path/to/generated_outputs
TRAIN_RESULT_DIR=/path/to/checkpoints
LOCAL_OUTPUT_ROOT=/tmp/rlsfv_llava_outputs
SAVE_THROUGH_LOCAL_TMP=1
```

Use `SKIP_CONDA_ACTIVATE=1` if the correct conda environment is already active:

```bash
conda activate rlsf-v-train
SKIP_CONDA_ACTIVATE=1 bash RLSF-V-llava.sh 7b
```
