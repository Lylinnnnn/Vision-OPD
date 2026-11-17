# RLSF-V Training

This directory contains all released training and data-generation code for RLSF-V. Evaluation code follows [pspdada/SENTINEL](https://github.com/pspdada/SENTINEL) and will be released soon under `../eval/`.

```text
train/
├── requirements-rlsf-v-train.txt
├── RLAIF_dataset/
├── llava_series/
└── hf_series/
```

## Shared Environment

Use one shared environment for both `llava_series` and `hf_series`.

```bash
git clone https://github.com/XLearning-SCU/RLSF-V
cd RLSF-V/train

conda create -n rlsf-v-train python=3.10 -y
conda activate rlsf-v-train
pip install -r requirements-rlsf-v-train.txt
pip install flash-attn==2.7.3 --no-build-isolation
```

The two series use the same environment. Switch only the `transformers` version before running a specific series.

For `hf_series`:

```bash
conda activate rlsf-v-train
pip install transformers==4.57.3
cd RLSF-V/train/hf_series
```

For `llava_series`:

```bash
conda activate rlsf-v-train
pip install transformers==4.51.3
cd RLSF-V/train/llava_series
```

## LLaVA Series

Use `transformers==4.51.3` before running LLaVA commands.

```bash
cd RLSF-V/train/llava_series
bash generate/run_llava_dpo_pipeline.sh 7b
bash generate/run_llava_dpo_pipeline.sh 13b
```

```bash
bash RLSF-V-llava.sh 7b
bash RLSF-V-llava.sh 13b
```

See [`llava_series/README.md`](llava_series/README.md) for the LLaVA workflow.

## HF Series

Use `transformers==4.57.3` before running HF-series generation and conversion commands.

```bash
cd RLSF-V/train/hf_series
bash generate/run_hf_dpo_pipeline.sh qwen25-3b
bash generate/run_hf_dpo_pipeline.sh qwen3-8b
bash generate/run_hf_dpo_pipeline.sh internvl35-8b
```

See [`hf_series/README.md`](hf_series/README.md) for the HF-series workflow.

## LlamaFactory Training

LlamaFactory training is a separate special case. Create a new `lf` environment for it, then install it from its own directory with the default editable install.

```bash
conda create -n lf python=3.11 -y
conda activate lf
cd RLSF-V/train/hf_series/LlamaFactory
pip install -e .
```

Then train with a dataset key registered in `data/dataset_info.json`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train qwen25.yaml dataset=qwen25_rlaif10k output_dir=saves/qwen25_rlaif10k
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train qwen3.yaml dataset=qwen3_rlaif10k output_dir=saves/qwen3_rlaif10k
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train intern3_5.yaml dataset=internvl35_rlaif10k output_dir=saves/internvl35_rlaif10k
```
