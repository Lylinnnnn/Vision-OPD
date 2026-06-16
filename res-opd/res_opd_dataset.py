"""
ResOPDDataset: Custom RLHFDataset with online resolution degradation.

Inherits from verl's RLHFDataset and overrides _build_messages to apply
online image degradation for the student model. Teacher images from the
``hires_images`` field are also optionally degraded when teacher_px is set
below target_px, enabling full decoupling of student and teacher resolutions.

Degradation pipeline: original → resize to {student,teacher}_px → resize
back to target_px (Lanczos interpolation, matching pilot1).

Configuration (passed via data config in yaml or CLI):
    data.custom_cls.path:  res-opd/res_opd_dataset.py
    data.custom_cls.name:  ResOPDDataset
    data.student_px:       224    (student resolution, 0 = no degradation)
    data.teacher_px:       0      (teacher resolution, 0 = use target_px as-is)
    data.target_px:        448    (target spatial dimension after up-scaling)
"""

import re
from io import BytesIO
from typing import Optional

from omegaconf import DictConfig
from PIL import Image
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset


class ResOPDDataset(RLHFDataset):
    """RLHFDataset with online resolution degradation.

    Both student images (``image_key``) and teacher images
    (``hires_images``) are degraded independently according to
    ``student_px`` and ``teacher_px``.  When a *_px value is 0 or
    >= target_px the corresponding image is simply resized to target_px
    without quality loss.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        super().__init__(data_files, tokenizer, config, processor, max_samples)

        self.student_px = config.get("student_px", 0)
        self.teacher_px = config.get("teacher_px", 0)
        self.target_px = config.get("target_px", 448)

        print(
            f"[ResOPDDataset] student_px={self.student_px}, "
            f"teacher_px={self.teacher_px}, target_px={self.target_px}"
        )
        if self.student_px > 0:
            print(
                f"  Student degradation: {self.student_px} → {self.target_px}"
            )
        if self.teacher_px == 0:
            print(
                f"  Teacher: gray/blank image (no visual information)"
            )
        elif self.teacher_px < self.target_px:
            print(
                f"  Teacher degradation: {self.teacher_px} → {self.target_px}"
            )
        else:
            print(
                f"  Teacher: original image at {self.target_px}px (no degradation)"
            )

    @staticmethod
    def _degrade(
        pil_image: Image.Image,
        degrade_px: int,
        target_px: int,
    ) -> Image.Image:
        """Resize down to ``degrade_px`` then back up to ``target_px``."""
        if degrade_px <= 0 or degrade_px >= target_px:
            return pil_image.resize((target_px, target_px), Image.LANCZOS)
        small = pil_image.resize((degrade_px, degrade_px), Image.LANCZOS)
        return small.resize((target_px, target_px), Image.LANCZOS)

    def _load_image(self, image_entry) -> Image.Image:
        """Load a PIL image from various storage formats."""
        if isinstance(image_entry, Image.Image):
            return image_entry.convert("RGB")
        if isinstance(image_entry, dict):
            raw_bytes = image_entry.get("bytes")
            if raw_bytes is not None:
                return Image.open(BytesIO(raw_bytes)).convert("RGB")
            inner = image_entry.get("image")
            if inner is not None:
                if isinstance(inner, Image.Image):
                    return inner.convert("RGB")
                if isinstance(inner, str):
                    return Image.open(inner).convert("RGB")
                raise TypeError(
                    f"Unsupported image type in dict: {type(inner)}"
                )
            path = image_entry.get("path")
            if path is not None:
                return Image.open(path).convert("RGB")
            raise ValueError(
                f"Image dict has no 'bytes', 'image', or 'path': "
                f"{list(image_entry.keys())}"
            )
        if isinstance(image_entry, str):
            return Image.open(image_entry).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image_entry)}")

    def _build_messages(self, example: dict):
        """Override: apply online degradation to student images.

        Teacher images (hires_images) are degraded in-place when teacher_px
        is set, so that ray_trainer's teacher reprompt sees the correct
        resolution.
        """
        messages: list = example[self.prompt_key]
        images = example.pop(self.image_key, None) or []
        videos = example.pop(self.video_key, None) or []

        # Degrade teacher images in-place (they stay in the row_dict and flow
        # into non_tensor_batch → teacher reprompt in ray_trainer)
        # When teacher_px == 0: replace with gray image (no visual information)
        # When 0 < teacher_px < target_px: degrade to teacher_px then upscale
        # When teacher_px >= target_px: use original image at target_px
        teacher_key = "hires_images"
        teacher_images = example.get(teacher_key) or []
        if teacher_images:
            processed_teachers = []
            for entry in teacher_images:
                pil = self._load_image(entry)
                if self.teacher_px == 0:
                    # Blank/gray image: no visual information for teacher
                    gray = Image.new("RGB", (self.target_px, self.target_px), color=(128, 128, 128))
                    processed_teachers.append(gray)
                elif self.teacher_px < self.target_px:
                    degraded = self._degrade(pil, self.teacher_px, self.target_px)
                    processed_teachers.append(degraded)
                else:
                    # teacher_px >= target_px: use original at target resolution
                    processed_teachers.append(pil.resize((self.target_px, self.target_px), Image.LANCZOS))
            example[teacher_key] = processed_teachers

        image_offset, video_offset = 0, 0
        for message in messages:
            if not images and not videos:
                continue
            assert self.processor is not None, (
                "processor is needed to process image and video"
            )

            content = message["content"]
            if not isinstance(content, str):
                continue

            content_list = []
            segments = re.split("(<image>|<video>)", content)
            segments = [seg for seg in segments if seg]
            for segment in segments:
                if segment == "<image>":
                    assert image_offset < len(images), (
                        f"image_offset {image_offset} >= len(images) "
                        f"{len(images)}"
                    )
                    pil = self._load_image(images[image_offset])
                    degraded = self._degrade(
                        pil, self.student_px, self.target_px
                    )
                    content_list.append(
                        {"type": "image", "image": degraded}
                    )
                    image_offset += 1
                elif segment == "<video>":
                    assert video_offset < len(videos), (
                        f"video_offset {video_offset} >= len(videos) "
                        f"{len(videos)}"
                    )
                    content_list.append(
                        {"type": "video", **videos[video_offset]}
                    )
                    video_offset += 1
                else:
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list

        assert image_offset == len(images), (
            f"image_offset {image_offset} != len(images) {len(images)}"
        )
        assert video_offset == len(videos), (
            f"video_offset {video_offset} != len(videos) {len(videos)}"
        )
        return messages