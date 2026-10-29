from pathlib import Path


BENCHMARK_IMAGE_DIRS = {
    "mmstar": ("MMStar_images",),
    "cv-bench": ("CVBench_2d_images", "CVBench_3d_images"),
}


def _existing_unique(paths):
    resolved = []
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            resolved.append(path)
    return resolved


def resolve_benchmark_image_path(path_value, benchmark, benchmark_data_dir):
    path = Path(str(path_value)).expanduser()
    if path.is_file():
        return path

    image_dirs = BENCHMARK_IMAGE_DIRS.get(benchmark, ())
    original_parts = path.parts
    marked_candidates = []
    for image_dir in image_dirs:
        if image_dir not in original_parts:
            continue
        marker_index = original_parts.index(image_dir)
        relative_path = Path(*original_parts[marker_index + 1 :])
        marked_candidates.append(benchmark_data_dir / image_dir / relative_path)

    candidates = marked_candidates or [benchmark_data_dir / image_dir / path.name for image_dir in image_dirs]
    resolved = _existing_unique(candidates)
    if len(resolved) == 1:
        return resolved[0]
    if len(resolved) > 1:
        raise RuntimeError(
            f"Ambiguous image path for benchmark={benchmark}: {path_value}; "
            f"candidates={[str(item) for item in resolved]}"
        )
    raise FileNotFoundError(
        f"Cannot resolve image for benchmark={benchmark}: {path_value}; "
        f"searched={[str(item) for item in candidates]}"
    )


def resolve_benchmark_images(records, benchmark, benchmark_json):
    if benchmark not in BENCHMARK_IMAGE_DIRS:
        return 0

    benchmark_data_dir = Path(benchmark_json).parent
    remapped = 0
    for row_index, item in enumerate(records):
        images = item.get("images") or []
        if not isinstance(images, list) or not images:
            raise ValueError(f"Missing images at benchmark={benchmark}, row={row_index}")
        resolved_images = []
        for path_value in images:
            resolved = resolve_benchmark_image_path(path_value, benchmark, benchmark_data_dir)
            resolved_images.append(str(resolved))
            if str(resolved) != str(path_value):
                remapped += 1
        item["images"] = resolved_images
    return remapped
