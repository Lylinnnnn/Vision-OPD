# RLSF-V

[ICML 2026] **RLSF-V: Mitigating Hallucinations in MLLMs via Fuzzy Semantic Self-Feedback**

> Official repository for **RLSF-V**.  
> Model weights are available on Hugging Face. All training and data-generation code is now released under `train/`; evaluation code follows [pspdada/SENTINEL](https://github.com/pspdada/SENTINEL) and will be released soon under `eval/`.

---

## 🔥 News

- **All training code is available** under [`train/`](train/), including LLaVA-series and HF-series pipelines.
- **Evaluation code follows [pspdada/SENTINEL](https://github.com/pspdada/SENTINEL)** and will be released soon under [`eval/`](eval/).
- **Model weights are available** at our Hugging Face Collection:  
  👉 [🤗 RLSF-V Model Collection](https://huggingface.co/collections/XLearning-SCU/rlsf-v)

---

## 📦 Model Weights

We provide the model weights mentioned in our paper. All released weights are available in our Hugging Face Collection:

👉 **[🤗 RLSF-V Model Collection](https://huggingface.co/collections/XLearning-SCU/rlsf-v)**

These weights can be plugged into the corresponding base models for inference or further fine-tuning. Please refer to the table below for details.

| Base Model | RLSF-V Model Weights | Status | Code |
|---|---|---|---|
| [🤗 LLaVA-v1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b) | [🤗 XLearning-SCU/LLaVA-v1.5-7B-RLSF-V](https://huggingface.co/XLearning-SCU/LLaVA-v1.5-7B-RLSF-V) | ✅ Released | ✅ [`train/llava_series/`](train/llava_series/) |
| [🤗 LLaVA-v1.5-13B](https://huggingface.co/liuhaotian/llava-v1.5-13b) | [🤗 XLearning-SCU/LLaVA-v1.5-13B-RLSF-V](https://huggingface.co/XLearning-SCU/LLaVA-v1.5-13B-RLSF-V) | ✅ Released | ✅ [`train/llava_series/`](train/llava_series/) |
| [🤗 Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | [🤗 XLearning-SCU/Qwen2.5-VL-3B-Instruct-RLSF-V](https://huggingface.co/XLearning-SCU/Qwen2.5-VL-3B-Instruct-RLSF-V) | ✅ Released | ✅ [`train/hf_series/`](train/hf_series/) |
| [🤗 Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | [🤗 XLearning-SCU/Qwen3-VL-8B-Instruct-RLSF-V](https://huggingface.co/XLearning-SCU/Qwen3-VL-8B-Instruct-RLSF-V) | ✅ Released | ✅ [`train/hf_series/`](train/hf_series/) |
| [🤗 InternVL3.5-8B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF) | [🤗 XLearning-SCU/InternVL3_5-8B-HF-RLSF-V](https://huggingface.co/XLearning-SCU/InternVL3_5-8B-HF-RLSF-V) | ✅ Released | ✅ [`train/hf_series/`](train/hf_series/) |

---

## 🚀 Code

Repository layout:

```text
RLSF-V/
├── train/
│   ├── README.md
│   ├── RLAIF_dataset/
│   ├── llava_series/
│   └── hf_series/
└── eval/                       # Coming soon, following SENTINEL
```

All training and data-generation code used by the released model families is available under `train/`. The evaluation code follows [pspdada/SENTINEL](https://github.com/pspdada/SENTINEL), is being organized, and will be released under `eval/` soon.

`train/llava_series/` contains the LLaVA-v1.5-7B and LLaVA-v1.5-13B RLSF-V pipeline.

```text
train/llava_series/
  RLSF-V-llava.sh
  generate/
    generate_llava.sh
    convert_llava.py
    run_llava_dpo_pipeline.sh
```

`train/hf_series/` contains the Hugging Face model-family pipeline.

```text
train/hf_series/
  generate/
    generate_hf.sh
    run_hf_dpo_pipeline.sh
    convert_qwen.py
    convert_intern.py
  LlamaFactory/
    qwen25.yaml
    qwen3.yaml
    intern3_5.yaml
```

Both pipelines include multi-GPU answer/logits generation, fuzzy semantic entropy based DPO data construction, and LoRA-DPO training.

See [`train/README.md`](train/README.md) for environment setup, and see [`train/llava_series/README.md`](train/llava_series/README.md) and [`train/hf_series/README.md`](train/hf_series/README.md) for detailed usage.

---

## LLaVA Quick Start

Prepare the shared training environment following [`train/README.md`](train/README.md), then run:

```bash
cd RLSF-V/train/llava_series
bash generate/run_llava_dpo_pipeline.sh 7b
bash generate/run_llava_dpo_pipeline.sh 13b
```

```bash
bash RLSF-V-llava.sh 7b
bash RLSF-V-llava.sh 13b
```

---

## HF-Series Quick Start

Prepare the shared training environment following [`train/README.md`](train/README.md), then run:

```bash
cd RLSF-V/train/hf_series
bash generate/run_hf_dpo_pipeline.sh qwen25-3b
bash generate/run_hf_dpo_pipeline.sh qwen3-8b
bash generate/run_hf_dpo_pipeline.sh internvl35-8b
```

For LlamaFactory training, use its default installation described in [`train/README.md`](train/README.md), then train with a dataset key registered in `LlamaFactory/data/dataset_info.json`:

```bash
cd RLSF-V/train/hf_series/LlamaFactory
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train qwen25.yaml dataset=qwen25_rlaif10k output_dir=saves/qwen25_rlaif10k
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train qwen3.yaml dataset=qwen3_rlaif10k output_dir=saves/qwen3_rlaif10k
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train intern3_5.yaml dataset=internvl35_rlaif10k output_dir=saves/internvl35_rlaif10k
```

---

## 📚 Dataset

The prompt file is provided under [`train/RLAIF_dataset/`](train/RLAIF_dataset/):

```text
train/RLAIF_dataset/
  rlaif10k.jsonl
  images/
```

Please place the image files from the Hugging Face dataset [openbmb/RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset) into `train/RLAIF_dataset/images/`.
Before running generation, make sure the `image_path` fields in `rlaif10k.jsonl` point to the downloaded images on your machine.

See [`train/RLAIF_dataset/README.md`](train/RLAIF_dataset/README.md) for details.

---

## 📌 Citation

If you find our work useful, please consider citing our paper.

```bibtex
@inproceedings{rlsfv2026,
  title={RLSF-V: Mitigating Hallucinations in MLLMs via Fuzzy Semantic Self-Feedback},
  author={He, Changhao and Yan, Shuhao and Li, Shuxian and Peng, Xi and Hu, Peng},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026}
}
```

---

## 🙏 Acknowledgement

Our implementation is based on [pspdada/SENTINEL](https://github.com/pspdada/SENTINEL). We sincerely thank the authors for their excellent work.
