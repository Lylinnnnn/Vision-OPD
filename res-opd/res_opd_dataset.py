"""
ResOPDDataset: Custom RLHFDataset with online resolution degradation.

Inherits from verl's RLHFDataset and overrides _build_messages to apply
online image degradation for the student model. Teacher images from the
``hires_images`` field are also optionally degraded when teacher_px or
teacher_ratio is set, enabling full decoupling of student and teacher
resolutions.

Default square degradation pipeline: original → resize to
{student,teacher}_px → resize back to target_px.

Original-ratio degradation pipeline: original → resize by
{student,teacher}_ratio → resize back to the original width/height.

Configuration (passed via data config in yaml or CLI):
    data.custom_cls.path:  res-opd/res_opd_dataset.py
    data.custom_cls.name:  ResOPDDataset
    data.student_px:       224    (student resolution, 0 = no degradation)
    data.teacher_px:       0      (teacher resolution, 0 = use target_px as-is)
    data.target_px:        448    (target spatial dimension after up-scaling)
    data.degradation_mode: square / original
    data.student_ratio:    1.0    (only used when degradation_mode=original)
    data.teacher_ratio:    1.0    (only used when degradation_mode=original)
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
    (``hires_images``) are degraded independently. The legacy ``square``
    mode uses ``student_px``/``teacher_px`` and returns target_px square
    images. The ``original`` mode uses ``student_ratio``/``teacher_ratio``
    and returns images at their original width/height.
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
        self.degradation_mode = config.get("degradation_mode", "square")
        self.student_ratio = float(config.get("student_ratio", 1.0))
        self.teacher_ratio = float(config.get("teacher_ratio", 1.0))
        if self.degradation_mode not in ("square", "original"):
            raise ValueError(
                "data.degradation_mode must be 'square' or 'original', "
                f"got {self.degradation_mode!r}"
            )

        print(
            f"[ResOPDDataset] student_px={self.student_px}, "
            f"teacher_px={self.teacher_px}, target_px={self.target_px}, "
            f"degradation_mode={self.degradation_mode}, "
            f"student_ratio={self.student_ratio}, "
            f"teacher_ratio={self.teacher_ratio}"
        )
        if self.degradation_mode == "original":
            print(
                "  Student degradation: original-size ratio "
                f"{self.student_ratio}"
            )
            if self.teacher_ratio <= 0:
                print("  Teacher: gray/blank image (no visual information)")
            else:
                print(
                    "  Teacher degradation: original-size ratio "
                    f"{self.teacher_ratio}"
                )
        else:
            if self.student_px > 0:
                print(
                    f"  Student degradation: {self.student_px} → "
                    f"{self.target_px}"
                )
            else:
                print(
                    f"  Student: original image at {self.target_px}px "
                    "(square resize, no downsample)"
                )
            if self.teacher_px == 0:
                print("  Teacher: gray/blank image (no visual information)")
            elif self.teacher_px < self.target_px:
                print(
                    f"  Teacher degradation: {self.teacher_px} → "
                    f"{self.target_px}"
                )
            else:
                print(
                    f"  Teacher: original image at {self.target_px}px "
                    "(no degradation)"
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

    @staticmethod
    def _degrade_by_ratio(
        pil_image: Image.Image,
        ratio: float,
    ) -> Image.Image:
        """Resize by ``ratio`` then back to original width/height."""
        width, height = pil_image.size
        if ratio <= 0:
            return Image.new("RGB", (width, height), color=(128, 128, 128))
        if ratio >= 1.0:
            return pil_image.copy()
        small_size = (
            max(1, int(round(width * ratio))),
            max(1, int(round(height * ratio))),
        )
        small = pil_image.resize(small_size, Image.LANCZOS)
        return small.resize((width, height), Image.LANCZOS)

    def _degrade_student(self, pil_image: Image.Image) -> Image.Image:
        if self.degradation_mode == "original":
            return self._degrade_by_ratio(pil_image, self.student_ratio)
        return self._degrade(pil_image, self.student_px, self.target_px)

    def _degrade_teacher(self, pil_image: Image.Image) -> Image.Image:
        if self.degradation_mode == "original":
            return self._degrade_by_ratio(pil_image, self.teacher_ratio)
        if self.teacher_px == 0:
            return Image.new(
                "RGB",
                (self.target_px, self.target_px),
                color=(128, 128, 128),
            )
        if self.teacher_px < self.target_px:
            return self._degrade(pil_image, self.teacher_px, self.target_px)
        return pil_image.resize(
            (self.target_px, self.target_px),
            Image.LANCZOS,
        )

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
        teacher_key = "hires_images"
        teacher_images = example.get(teacher_key) or []
        if teacher_images:
            processed_teachers = []
            for entry in teacher_images:
                pil = self._load_image(entry)
                processed_teachers.append(self._degrade_teacher(pil))
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
                    degraded = self._degrade_student(pil)
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
