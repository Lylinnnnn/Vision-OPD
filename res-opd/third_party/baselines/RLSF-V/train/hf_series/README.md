# HF Series

This directory contains the RLSF-V workflow for Hugging Face based vision-language models:

- Qwen2.5-VL-3B
- Qwen3-VL-8B
- InternVL3.5-8B-HF

The workflow is:

1. Generate sampled answers and top-k logits.
2. Convert sampled answers into DPO preference data with the fuzzy semantic entropy score.
3. Register the converted DPO data in LlamaFactory.
4. Train LoRA adapters with LlamaFactory.

For environment setup, see [`../README.md`](../README.md).

## Directory Layout

```text
hf_series/
├── generate/
│   ├── generate_hf.sh
│   ├── run_hf_dpo_pipeline.sh
│   ├── qwen25_infer.py
│   ├── qwen3_infer.py
│   ├── internvl3_5_infer.py
│   ├── convert_qwen.py
│   └── convert_intern.py
├── LlamaFactory/
│   ├── data/
│   │   ├── convert_dpo_dataset.py
│   │   └── dataset_info.json
│   ├── qwen25.yaml
│   ├── qwen3.yaml
│   └── intern3_5.yaml
```

## Step 1: Generate Answers And Logits

The default input file is ...

```text
RLSF-V/train/RLAIF_dataset/rlaif10k.jsonl
```

Generation uses GPUs `0,1,2,3` by default. Override `CUDA_VISIBLE_DEVICES` if you need a different GPU list.

Run generation for one model.

```bash
bash generate/generate_hf.sh qwen25-3b
bash generate/generate_hf.sh qwen3-8b
bash generate/generate_hf.sh internvl35-8b
```

Outputs include ...

```text
generate/<model_name>/answers.jsonl
generate/<model_name>/logits_merged.pt
```

You can set an explicit output directory.

```bash
OUTPUT_DIR=generate/qwen3-vl-8b \
bash generate/generate_hf.sh qwen3-8b
```

## Step 2: Convert Answers To DPO Data

The conversion step computes the fuzzy semantic entropy score from the generated logits, ranks sampled answers for each prompt, and builds chosen/rejected pairs.

The combined pipeline runs generation and conversion together.

```bash
bash generate/run_hf_dpo_pipeline.sh qwen25-3b
bash generate/run_hf_dpo_pipeline.sh qwen3-8b
bash generate/run_hf_dpo_pipeline.sh internvl35-8b
```

To convert an existing generation directory.

```bash
RUN_GENERATE=0 \
RUN_CONVERT=1 \
OUTPUT_DIR=generate/qwen3-vl-8b \
bash generate/run_hf_dpo_pipeline.sh qwen3-8b
```

Outputs include ...

```text
generate/<model_name>/<model_name>.jsonl
generate/<model_name>/<model_name>_dpo_data.json
```

DPO records contain `img_path`, `prompt`, `chosen`, and `rejected`.

## Step 3: Register LlamaFactory Data

Convert the generated DPO file into the LlamaFactory multimodal DPO format with `LlamaFactory/data/convert_dpo_dataset.py`. Edit the `input_files` and `output_files` lists in that script for the model output you want to register.

```bash
cd RLSF-V/train/hf_series
python LlamaFactory/data/convert_dpo_dataset.py
```

Register the converted file in `LlamaFactory/data/dataset_info.json`.

```json
"qwen3_rlaif10k": {
  "file_name": "qwen3_rlaif10k.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "chosen": "chosen",
    "rejected": "rejected",
    "images": "images"
  },
  "ranking": true
}
```

LlamaFactory records contain `instruction`, `input`, `chosen`, `rejected`, and `images`.

## Step 4: Train LoRA Adapters

Install LlamaFactory with its default setup as described in [`../README.md`](../README.md), then run training from the `LlamaFactory/` directory.

```bash
cd RLSF-V/train/hf_series/LlamaFactory
```

Training configs ...

```text
qwen25.yaml
qwen3.yaml
intern3_5.yaml
```

Pass the dataset key registered in `data/dataset_info.json`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
llamafactory-cli train qwen25.yaml \
  dataset=qwen25_rlaif10k \
  output_dir=saves/qwen25_rlaif10k
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
llamafactory-cli train qwen3.yaml \
  dataset=qwen3_rlaif10k \
  output_dir=saves/qwen3_rlaif10k
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
llamafactory-cli train intern3_5.yaml \
  dataset=internvl35_rlaif10k \
  output_dir=saves/internvl35_rlaif10k
```

The trained adapter is saved under the configured `output_dir`.
