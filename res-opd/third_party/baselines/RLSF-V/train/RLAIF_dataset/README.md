# RLAIF Dataset

This directory contains the prompt file used by the RLSF-V LLaVA pipeline.

```text
RLAIF_dataset/
  rlaif10k.jsonl
  images/
```

Please place the image files from the Hugging Face dataset [openbmb/RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset) into:

```text
RLAIF_dataset/images/
```

The generation scripts use `../RLAIF_dataset/rlaif10k.jsonl` by default. Before running generation, make sure each `image_path` in the JSONL points to the downloaded images on your machine.
