"""
ResOPDDataset: Custom RLHFDataset with online original-ratio degradation.

Inherits from verl's RLHFDataset and overrides _build_messages to apply
online image degradation for the student model. Teacher images from the
``hires_images`` field are also degraded independently, enabling full
decoupling of student and teacher visual inputs.

Original-ratio degradation pipeline: original → resize by
{student,teacher}_ratio → resize back to the original width/height.

Configuration (passed via data config in yaml or CLI):
    data.custom_cls.path:  res-opd/res_opd_dataset.py
    data.custom_cls.name:  ResOPDDataset
    data.degradation_mode: original
    data.student_ratio:    1.0    (1.0 = original; <1.0 = down/up sample)
    data.teacher_ratio:    1.0    (0 = gray/blank; <1.0 = down/up sample)
    data.student_px / teacher_px / target_px may still exist in old configs
    for compatibility, but are not used for image degradation.
"""

import re
import os
from io import BytesIO
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig
from PIL import Image
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset


class ResOPDDataset(RLHFDataset):
    """RLHFDataset with online resolution degradation.

    Both student images (``image_key``) and teacher images
    (``hires_images``) are degraded independently using
    ``student_ratio``/``teacher_ratio`` and returned at their original
    width/height.
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
        self.degradation_mode = config.get("degradation_mode", "original")
        self.student_ratio = float(config.get("student_ratio", 1.0))
        self.teacher_ratio = float(config.get("teacher_ratio", 1.0))
        self._path_remap_logged: set[tuple[str, str]] = set()
        if self.degradation_mode != "original":
            raise ValueError(
                "data.degradation_mode must be 'original'; "
                f"got {self.degradation_mode!r}"
            )

        print(
            "[ResOPDDataset] degradation_mode=original, "
            f"student_ratio={self.student_ratio}, "
            f"teacher_ratio={self.teacher_ratio}; "
            f"legacy_px=student:{self.student_px},teacher:{self.teacher_px},target:{self.target_px}"
        )
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

    @staticmethod
    def _data_root_candidates() -> list[Path]:
        candidates = [
            os.environ.get("BENCHMARK_DATA_DIR", ""),
            os.environ.get("VISION_BENCHMARK_DATA_DIR", ""),
            os.environ.get("RES_OPD_DATA_ROOT", ""),
            str(Path.home() / "notebook" / "data"),
            str(Path.home() / "notebook" / "yanlin" / "data"),
            "/home/zhengyanzhao.zyz/notebook/yanlin/data",
            "/home/liuyanlin.lyl/notebook/data",
        ]
        deduped: list[Path] = []
        seen = set()
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate)
            key = str(path)
            if key not in seen:
                deduped.append(path)
                seen.add(key)
        return deduped

    def _resolve_image_path(self, path: str | os.PathLike) -> str:
        original = Path(path)
        if original.exists():
            return str(original)

        original_text = str(original)
        old_roots = [
            "/home/liuyanlin.lyl/notebook/data",
            "/home/zhengyanzhao.zyz/notebook/yanlin/data",
        ]
        attempted = []
        for old_root in old_roots:
            if not original_text.startswith(old_root + "/"):
                continue
            rel = original_text[len(old_root) + 1 :]
            for data_root in self._data_root_candidates():
                candidate = data_root / rel
                attempted.append(str(candidate))
                if candidate.exists():
                    key = (old_root, str(data_root))
                    if key not in self._path_remap_logged:
                        print(
                            "[ResOPDDataset] Remapping image paths: "
                            f"{old_root} -> {data_root}"
                        )
                        self._path_remap_logged.add(key)
                    return str(candidate)

        attempted_text = "\n  ".join(attempted[:8])
        if attempted_text:
            attempted_text = "\nTried remapped candidates:\n  " + attempted_text
        raise FileNotFoundError(
            f"Image path does not exist: {original_text}{attempted_text}"
        )

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
        return self._degrade_by_ratio(pil_image, self.student_ratio)

    def _degrade_teacher(self, pil_image: Image.Image) -> Image.Image:
        return self._degrade_by_ratio(pil_image, self.teacher_ratio)

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
                    return Image.open(self._resolve_image_path(inner)).convert("RGB")
                raise TypeError(
                    f"Unsupported image type in dict: {type(inner)}"
                )
            path = image_entry.get("path")
            if path is not None:
                return Image.open(self._resolve_image_path(path)).convert("RGB")
            raise ValueError(
                f"Image dict has no 'bytes', 'image', or 'path': "
                f"{list(image_entry.keys())}"
            )
        if isinstance(image_entry, str):
            return Image.open(self._resolve_image_path(image_entry)).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image_entry)}")

    def _build_messages(self, example: dict):
        """Override: apply online degradation to student images.

        Teacher images (hires_images) are degraded in-place so that
        ray_trainer's teacher reprompt sees the correct view.
        """
        messages: list = example[self.prompt_key]
        images = example.pop(self.image_key, None) or []
        videos = example.pop(self.video_key, None) or []

        # Load teacher images from hires_images (now points to original
        # COCO paths in parquet) and apply online degradation via
        # _degrade_teacher().  This avoids pre-storing degraded teacher
        # images on disk.
        teacher_key = "hires_images"
        hires_images = example.pop(teacher_key, None) or []
        if hires_images:
            processed_teachers = []
            for img_entry in hires_images:
                pil = self._load_image(img_entry)
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
