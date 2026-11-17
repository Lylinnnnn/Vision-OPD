import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LLAMAFACTORY_DIR = SCRIPT_DIR.parent
HF_SERIES_DIR = LLAMAFACTORY_DIR.parent

input_files = [
    HF_SERIES_DIR / "generate/qwen3-vl-8b/qwen3-vl-8b_dpo_data.json",
]
output_files = [
    SCRIPT_DIR / "qwen3.json",
]


for input_file, output_file in zip(input_files, output_files):
    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_data = []
    for item in data:
        prompt_with_image_tag = "<image>" + item["prompt"]

        formatted_data.append(
            {
                "instruction": prompt_with_image_tag,
                "input": "",
                "chosen": item["chosen"],
                "rejected": item["rejected"],
                "images": [item["img_path"]],
            }
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=2)

    print(f"Converted DPO data saved to {output_file}")
