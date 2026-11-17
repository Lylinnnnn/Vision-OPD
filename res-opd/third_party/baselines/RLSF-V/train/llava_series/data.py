import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.utils.rnn as rnn_utils
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor, PreTrainedTokenizer

from llava.constants import IGNORE_INDEX
from llava.train.train import preprocess, preprocess_multimodal
from arguments import LlavaDataArguments


class LazySupervisedDataset(Dataset):
    """
    DPO preference dataset:
    each sample contains one image + one prompt + (chosen answer, rejected answer).
    JSON format: List[{"img_path","prompt","chosen","rejected"}]
    """

    def __init__(
        self,
        train_data_path: str,
        tokenizer: PreTrainedTokenizer,
        data_args: LlavaDataArguments,
        seed: int = 42,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args

        with open(train_data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            raise ValueError(f"Expected a JSON list at {train_data_path}, got {type(raw)}")

        # Normalize into internal format
        self.data: List[dict] = []
        for ex in raw:
            if not isinstance(ex, dict):
                continue

            img_path = ex.get("img_path")
            prompt = ex.get("prompt")
            chosen = ex.get("chosen")
            rejected = ex.get("rejected")

            if not img_path or not prompt or chosen is None or rejected is None:
                # skip malformed samples
                continue

            question = "<image>\n" + prompt.strip()

            self.data.append(
                {
                    "img_path": img_path,
                    "chosen_conversations": [
                        {"from": "human", "value": question},
                        {"from": "gpt", "value": str(chosen).strip()},
                    ],
                    "reject_conversations": [
                        {"from": "human", "value": question},
                        {"from": "gpt", "value": str(rejected).strip()},
                    ],
                }
            )

        # optional shuffle for reproducibility
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(len(self.data), generator=g).tolist()
        self.data = [self.data[i] for i in perm]

    def __len__(self) -> int:
        return len(self.data)

    def _resolve_image_path(self, img_path: str) -> str:
        # 1) absolute path or relative path that exists
        if os.path.isabs(img_path) and os.path.exists(img_path):
            return img_path
        if os.path.exists(img_path):
            return img_path

        # 2) try using data_args.image_folder as a base (if provided)
        base = getattr(self.data_args, "image_folder", None)
        if base:
            candidate = os.path.join(base, img_path)
            if os.path.exists(candidate):
                return candidate

        # 3) give a clear error
        raise FileNotFoundError(f"Image not found: {img_path} (also tried base folder: {base})")

    def _preprocess_image(self, image: Image.Image, processor: CLIPImageProcessor) -> torch.Tensor:
        if self.data_args.image_aspect_ratio == "pad":

            def expand2square(pil_img: Image.Image, background_color):
                w, h = pil_img.size
                if w == h:
                    return pil_img
                if w > h:
                    result = Image.new(pil_img.mode, (w, w), background_color)
                    result.paste(pil_img, (0, (w - h) // 2))
                    return result
                result = Image.new(pil_img.mode, (h, h), background_color)
                result.paste(pil_img, ((h - w) // 2, 0))
                return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))

        return processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

    def _handle_text(self, conversations: list, has_image: bool) -> Dict[str, torch.Tensor]:
        sources = preprocess_multimodal(copy.deepcopy([conversations]), self.data_args)
        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)
        return {
            "input_ids": data_dict["input_ids"][0],
            "labels": data_dict["labels"][0],
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.data[idx]

        # image
        processor: CLIPImageProcessor = self.data_args.image_processor
        img_path = self._resolve_image_path(ex["img_path"])
        image = Image.open(img_path).convert("RGB")
        image_tensor = self._preprocess_image(image, processor)

        # text (chosen / rejected)
        has_image = True
        chosen = self._handle_text(ex["chosen_conversations"], has_image=has_image)
        rejected = self._handle_text(ex["reject_conversations"], has_image=has_image)

        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_labels": chosen["labels"],
            "reject_input_ids": rejected["input_ids"],
            "reject_labels": rejected["labels"],
            "images": image_tensor,
        }


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: PreTrainedTokenizer

    def __post_init__(self):
        self.pad_token_id: int = self.tokenizer.pad_token_id
        self.max_length: int = self.tokenizer.model_max_length

    def pad_and_truncate(self, sequences: List[torch.Tensor], padding_value: int) -> torch.Tensor:
        padded = rnn_utils.pad_sequence(sequences, batch_first=True, padding_value=padding_value)
        return padded[:, : self.max_length]

    def __call__(self, instances: List[dict]) -> Dict[str, torch.Tensor]:
        chosen_input_ids = [i["chosen_input_ids"] for i in instances]
        chosen_labels = [i["chosen_labels"] for i in instances]
        reject_input_ids = [i["reject_input_ids"] for i in instances]
        reject_labels = [i["reject_labels"] for i in instances]

        chosen_input_ids = self.pad_and_truncate(chosen_input_ids, padding_value=self.pad_token_id)
        chosen_labels = self.pad_and_truncate(chosen_labels, padding_value=IGNORE_INDEX)
        reject_input_ids = self.pad_and_truncate(reject_input_ids, padding_value=self.pad_token_id)
        reject_labels = self.pad_and_truncate(reject_labels, padding_value=IGNORE_INDEX)

        batch = dict(
            chosen_input_ids=chosen_input_ids,
            chosen_labels=chosen_labels,
            reject_input_ids=reject_input_ids,
            reject_labels=reject_labels,
            chosen_attention_mask=chosen_input_ids.ne(self.pad_token_id),
            reject_attention_mask=reject_input_ids.ne(self.pad_token_id),
        )

        images = [i["images"] for i in instances]
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch["images"] = torch.stack(images)
        else:
            batch["images"] = images

        return batch


