"""Qwen3VL adaptation of HALC's focal contrast and visual beam selection."""

import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image


EXEMPT_OBJECT_WORDS = frozenset({
    "image", "side", "background", "feature", "features", "center", "left",
    "right", "scene", "view", "detail", "group", "manner",
})


@dataclass(frozen=True)
class HALCConfig:
    beam_size: int = 3
    context_window: int = 4
    candidate_count: int = 4
    expand_ratio: float = 0.6
    box_threshold: float = 0.4
    contrast_weight: float = 0.05
    skip_rate: float = 0.5
    max_new_tokens: int = 128

    def validate(self):
        if self.beam_size < 1 or self.context_window < 2:
            raise ValueError("HALC needs at least one beam and two visual contexts")
        if self.candidate_count < 2 or self.candidate_count % 2:
            raise ValueError("candidate_count must be positive and even")
        if self.candidate_count // 2 > self.context_window * (self.context_window - 1) // 2:
            raise ValueError("not enough context pairs for candidate_count")
        if not 0 <= self.skip_rate < 1 or not 0 <= self.box_threshold <= 1:
            raise ValueError("skip_rate or box_threshold is out of range")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")


def expand_box(box, factor):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    return (
        max(0.0, left - width * factor / 2),
        max(0.0, top - height * factor / 2),
        min(1.0, right + width * factor / 2),
        min(1.0, bottom + height * factor / 2),
    )


def focal_crops(image, box, count, expand_ratio):
    boxes = [expand_box(box, -expand_ratio), tuple(box)]
    while len(boxes) < count:
        boxes.append(expand_box(boxes[-1], expand_ratio))
    width, height = image.size
    crops = []
    for left, top, right, bottom in boxes:
        x0, y0 = int(left * width), int(top * height)
        x1, y1 = int(right * width), int(bottom * height)
        crops.append(image.crop((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1))))
    return crops


def focal_contrast_candidates(logits_by_view, candidate_count, weight):
    """Top-JSD view pairs, both contrast directions, as in upstream HALC."""
    distributions = [torch.softmax(logits.float(), dim=-1) for logits in logits_by_view]
    pairs = []
    for i, p in enumerate(distributions):
        for j in range(i + 1, len(distributions)):
            q = distributions[j]
            midpoint = (p + q) / 2
            jsd = (
                torch.sum(torch.xlogy(p, p) - torch.xlogy(p, midpoint))
                + torch.sum(torch.xlogy(q, q) - torch.xlogy(q, midpoint))
            ) / 2
            pairs.append((float(jsd), i, j))
    pairs.sort(reverse=True)
    candidates = []
    for _, i, j in pairs[: candidate_count // 2]:
        for base, final in ((i, j), (j, i)):
            contrast = logits_by_view[final].float() - weight * logits_by_view[base].float()
            candidates.append(int(torch.argmax(contrast).item()))
    return candidates


def build_qwen_inputs(processor, image, prompt, device):
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    kwargs = {"text": [text], "images": images, "padding": True, "return_tensors": "pt"}
    if videos:
        kwargs["videos"] = videos
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in processor(**kwargs).items()}


def next_token_logits(model, inputs, generated):
    kwargs = dict(inputs)
    if "mm_token_type_ids" not in kwargs:
        raise ValueError("Qwen3VL processor did not return mm_token_type_ids")
    if generated:
        continuation = torch.tensor([generated], dtype=kwargs["input_ids"].dtype,
                                    device=kwargs["input_ids"].device)
        kwargs["input_ids"] = torch.cat((kwargs["input_ids"], continuation), dim=1)
        kwargs["attention_mask"] = torch.cat(
            (kwargs["attention_mask"], torch.ones_like(continuation)), dim=1
        )
        kwargs["mm_token_type_ids"] = torch.cat(
            (kwargs["mm_token_type_ids"],
             torch.zeros_like(continuation, dtype=kwargs["mm_token_type_ids"].dtype)), dim=1
        )
    if not (kwargs["input_ids"].shape == kwargs["attention_mask"].shape ==
            kwargs["mm_token_type_ids"].shape):
        raise ValueError("Qwen3VL input_ids, attention_mask, and mm_token_type_ids must align")
    output = model.generate(
        **kwargs, max_new_tokens=1, do_sample=False, num_beams=1,
        return_dict_in_generate=True, output_scores=True, use_cache=False,
    )
    if len(output.scores) != 1:
        raise RuntimeError("Qwen3VL did not return one next-token score tensor")
    return output.scores[0][0]


def install_groundingdino_bert_compatibility():
    from groundingdino.models.GroundingDINO import groundingdino

    if getattr(groundingdino.BertModelWarper, "_res_opd_compatible", False):
        return
    original_warper = groundingdino.BertModelWarper

    def compatible_warper(bert_model):
        if hasattr(bert_model, "get_head_mask"):
            return original_warper(bert_model)

        def expand_attention_mask(module, args, kwargs):
            mask = kwargs.get("attention_mask")
            if mask is not None and mask.ndim == 3:
                kwargs["attention_mask"] = module.get_extended_attention_mask(
                    mask, mask.shape[:2]
                )
            return args, kwargs

        bert_model.register_forward_pre_hook(expand_attention_mask, with_kwargs=True)
        return bert_model

    compatible_warper._res_opd_compatible = True
    groundingdino.BertModelWarper = compatible_warper


class HALCDecoder:
    def __init__(self, model, processor, detector_config, detector_weights,
                 blip_model_path, config, device):
        import spacy
        from groundingdino.util.inference import load_model
        from transformers import BlipForImageTextRetrieval, BlipProcessor

        config.validate()
        for path in (detector_config, detector_weights, blip_model_path):
            if not Path(path).exists():
                raise FileNotFoundError(path)
        self.model = model
        self.processor = processor
        self.config = config
        self.device = device
        self.nlp = spacy.load("en_core_web_sm")
        install_groundingdino_bert_compatibility()
        self.detector = load_model(detector_config, detector_weights, device=str(device))
        self.blip_processor = BlipProcessor.from_pretrained(blip_model_path)
        self.blip, loading = BlipForImageTextRetrieval.from_pretrained(
            blip_model_path, output_loading_info=True, local_files_only=True
        )
        missing = [key for key in loading["missing_keys"] if not key.endswith("position_ids")]
        if missing:
            raise RuntimeError(f"BLIP retrieval checkpoint has uninitialized weights: {missing[:10]}")
        self.blip = self.blip.to(device).eval()
        self.eos_ids = {int(x) for x in (model.generation_config.eos_token_id
                       if isinstance(model.generation_config.eos_token_id, (list, tuple))
                       else [model.generation_config.eos_token_id]) if x is not None}

    def _is_object_token(self, token_id):
        token = self.processor.tokenizer.decode([token_id], skip_special_tokens=True).strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z'-]*", token):
            return None
        if token.lower() in EXEMPT_OBJECT_WORDS:
            return None
        parsed = self.nlp(token)
        return token.lower() if parsed and parsed[0].pos_ in ("NOUN", "PROPN") else None

    def _ground_box(self, image_path, word, cache):
        if word in cache:
            return cache[word]
        from groundingdino.util.inference import load_image, predict

        source, transformed = load_image(str(image_path))
        boxes, _, _ = predict(
            model=self.detector, image=transformed, caption=word,
            box_threshold=self.config.box_threshold, text_threshold=0.4,
        )
        candidates = []
        for cx, cy, width, height in boxes.detach().cpu().tolist():
            box = (max(0.0, cx - width / 2), max(0.0, cy - height / 2),
                   min(1.0, cx + width / 2), min(1.0, cy + height / 2))
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area >= 0.01:
                candidates.append((area, box))
        if len(candidates) > 3:
            cache[word] = None
        else:
            cache[word] = min(candidates)[1] if candidates else (0.2, 0.2, 0.8, 0.8)
        return cache[word]

    def _image_text_scores(self, image, token_sequences):
        texts = [self.processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
                 or "image" for tokens in token_sequences]
        inputs = self.blip_processor(
            text=texts, images=[image] * len(texts), return_tensors="pt",
            padding=True, truncation=True
        )
        inputs = {key: value.to(self.device) if torch.is_tensor(value) else value
                  for key, value in inputs.items()}
        outputs = self.blip(**inputs)
        scores = outputs.itm_score
        if scores is None or scores.shape != (len(texts), 2):
            raise RuntimeError("BLIP retrieval model did not return paired ITM logits")
        return scores.float().softmax(dim=-1)[:, 1].tolist()

    @torch.inference_mode()
    def generate(self, image_path, prompt="Describe the image.", seed=42):
        image = Image.open(image_path).convert("RGB")
        original_inputs = build_qwen_inputs(self.processor, image, prompt, self.device)
        rng = random.Random(seed)
        beams = [([], 0.0, False, 0)]
        grounding_cache = {}
        triggered = 0
        for _ in range(self.config.max_new_tokens):
            proposals = []
            corrected = False
            for tokens, score, finished, changed in beams:
                if finished:
                    proposals.append((tokens, score, finished, changed))
                    continue
                base_logits = next_token_logits(self.model, original_inputs, tokens)
                base_token = int(torch.argmax(base_logits).item())
                word = self._is_object_token(base_token)
                options = [base_token]
                if word and rng.random() >= self.config.skip_rate:
                    box = self._ground_box(image_path, word, grounding_cache)
                    if box is not None:
                        crops = focal_crops(image, box, self.config.context_window,
                                            self.config.expand_ratio)
                        view_logits = []
                        for crop in crops:
                            inputs = build_qwen_inputs(self.processor, crop, prompt, self.device)
                            view_logits.append(next_token_logits(self.model, inputs, tokens))
                        options = focal_contrast_candidates(
                            view_logits, self.config.candidate_count,
                            self.config.contrast_weight,
                        )
                        corrected = True
                        triggered += 1
                for token in options:
                    proposals.append((tokens + [token], score, token in self.eos_ids,
                                      changed + int(token != base_token)))
            if corrected:
                visual_scores = self._image_text_scores(
                    image, [tokens for tokens, _, _, _ in proposals]
                )
                proposals = [(tokens, visual_score, finished, changed)
                             for (tokens, _, finished, changed), visual_score
                             in zip(proposals, visual_scores)]
            proposals.sort(key=lambda entry: entry[1], reverse=True)
            deduplicated = {}
            for proposal in proposals:
                deduplicated.setdefault(tuple(proposal[0]), proposal)
            beams = list(deduplicated.values())[: self.config.beam_size]
            if all(finished for _, _, finished, _ in beams):
                break
        best_tokens = beams[0][0]
        return {
            "generated_caption": self.processor.tokenizer.decode(best_tokens, skip_special_tokens=True).strip(),
            "generated_token_ids": best_tokens,
            "generated_token_count": len(best_tokens),
            "halc_triggered_count": triggered,
            "halc_changed_token_count": beams[0][3],
            "halc_unique_grounded_terms": len(grounding_cache),
        }
