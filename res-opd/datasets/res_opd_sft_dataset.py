"""Res-OPD multimodal SFT dataset helpers.

This keeps verl's official MultiTurnSFTDataset behavior, but resolves image
paths across the two server layouts used by the Res-OPD experiments.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.dataset.vision_utils import process_image, process_video


def _default_data_roots() -> list[Path]:
    home = Path.home()
    candidates = [
        os.environ.get("RES_OPD_DATA_ROOT", ""),
        os.environ.get("BENCHMARK_DATA_DIR", ""),
        os.environ.get("VISION_BENCHMARK_DATA_DIR", ""),
        home / "notebook" / "data",
        home / "notebook" / "yanlin" / "data",
        Path("/home/zhengyanzhao.zyz/notebook/yanlin/data"),
        Path("/home/liuyanlin.lyl/notebook/data"),
    ]
    roots: list[Path] = []
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists() and path not in roots:
            roots.append(path)
    return roots


class ResOPDSFTDataset(MultiTurnSFTDataset):
    """Multi-turn SFT dataset with Res-OPD server path remapping."""

    def _resolve_path_text(self, path_text: str) -> str:
        path = Path(path_text).expanduser()
        if path.exists():
            return str(path)

        text = str(path)
        marker_candidates = [
            "/COCO/",
            "/AMBER/",
            "/MMStar_images/",
            "/CVBench_2d_images/",
            "/CVBench_3d_images/",
        ]
        for marker in marker_candidates:
            if marker not in text:
                continue
            suffix = text.split(marker, 1)[1]
            for root in _default_data_roots():
                candidate = root / marker.strip("/") / suffix
                if candidate.exists():
                    return str(candidate)

        return str(path)

    def _resolve_image_entry(self, image):
        if isinstance(image, str):
            return self._resolve_path_text(image)
        if isinstance(image, dict):
            resolved = dict(image)
            if isinstance(resolved.get("path"), str):
                resolved["path"] = self._resolve_path_text(resolved["path"])
            elif isinstance(resolved.get("image"), str):
                resolved["image"] = self._resolve_path_text(resolved["image"])
            return resolved
        return image

    def _build_messages(self, example: dict):
        """Resolve image paths before delegating to the processor."""
        messages: list = example[self.messages_key]
        images = example[self.image_key] if self.image_key in example else []
        videos = example[self.video_key] if self.video_key in example else []
        images = [self._resolve_image_entry(image) for image in images]

        image_offset, video_offset = 0, 0
        for message in messages:
            if self.image_key not in example and self.video_key not in example:
                continue
            assert self.processor is not None, "processor is needed to process image and video"

            content = message["content"]
            if not isinstance(content, str):
                continue

            content_list = []
            segments = re.split("(<image>|<video>)", content)
            segments = [item for item in segments if item != ""]
            for segment in segments:
                if segment == "<image>":
                    image = process_image(images[image_offset], image_patch_size=self.image_patch_size)
                    content_list.append({"type": "image", "image": image})
                    image_offset += 1
                elif segment == "<video>":
                    video = process_video(videos[video_offset], image_patch_size=self.image_patch_size)
                    content_list.append({"type": "video", "video": video})
                    video_offset += 1
                else:
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list

        assert image_offset == len(images), f"image_offset {image_offset} != len(images) {len(images)}"
        assert video_offset == len(videos), f"video_offset {video_offset} != len(videos) {len(videos)}"
        return messages
