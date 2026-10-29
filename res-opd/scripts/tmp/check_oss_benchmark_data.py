#!/usr/bin/env python3

import argparse
import json
import random
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mount-root", type=Path, default=Path("/data/oss_bucket_0/yanlin"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--benchmarks",
        default="chair,pope,amber,mmstar,cv-bench",
    )
    parser.add_argument("--sample-images", type=int, default=8)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


class Report:
    def __init__(self):
        self.rows = []

    def add(self, benchmark, status, message, path=None, details=None):
        row = {
            "benchmark": benchmark,
            "status": status,
            "message": message,
        }
        if path is not None:
            row["path"] = str(path)
        if details is not None:
            row["details"] = details
        self.rows.append(row)
        print(f"[{status}] {benchmark}: {message}" + (f" [{path}]" if path else ""))

    def failed(self):
        return any(row["status"] != "PASS" for row in self.rows)


def first_dir(candidates):
    return next((path for path in candidates if path.is_dir()), None)


def first_file(candidates):
    return next((path for path in candidates if path.is_file() and path.stat().st_size > 0), None)


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def image_files(path):
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in extensions)


def verify_images(report, benchmark, paths, sample_count):
    if not paths:
        report.add(benchmark, "FAIL", "image directory contains no supported images")
        return
    try:
        from PIL import Image
    except ImportError:
        report.add(benchmark, "FAIL", "Pillow is unavailable; cannot verify image readability")
        return
    rng = random.Random(42)
    selected = paths if len(paths) <= sample_count else rng.sample(paths, sample_count)
    failures = []
    for path in selected:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:
            failures.append(f"{path}: {error}")
    if failures:
        report.add(benchmark, "FAIL", "sample image verification failed", details=failures)
    else:
        report.add(benchmark, "PASS", f"verified {len(selected)} sampled images; total={len(paths)}")


def data_roots(mount_root):
    return [mount_root / "data", mount_root]


def check_coco(report, mount_root, repo_root, sample_count, benchmarks):
    coco_root = first_dir(
        root / "COCO" / "coco2017val" for root in data_roots(mount_root)
    )
    for benchmark in benchmarks:
        if coco_root is None:
            report.add(benchmark, "FAIL", "COCO/coco2017val directory not found")
            continue
        report.add(benchmark, "PASS", "resolved COCO root", coco_root)

    if coco_root is None:
        return None
    image_root = coco_root / "val2017"
    instances_path = coco_root / "annotations" / "instances_val2017.json"
    captions_path = coco_root / "annotations" / "captions_val2017.json"
    required = [image_root, instances_path, captions_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        for benchmark in benchmarks:
            report.add(benchmark, "FAIL", "missing required COCO paths", details=missing)
        return None

    images = image_files(image_root)
    if len(images) != 5000:
        for benchmark in benchmarks:
            report.add(benchmark, "FAIL", f"expected 5000 COCO val images, found {len(images)}", image_root)
    else:
        verify_images(report, "chair/pope", images, sample_count)

    try:
        instances = load_json(instances_path)
        captions = load_json(captions_path)
        instance_images = len(instances.get("images", []))
        categories = len(instances.get("categories", []))
        caption_images = len(captions.get("images", []))
        caption_annotations = len(captions.get("annotations", []))
        if instance_images != 5000 or categories != 80 or caption_images != 5000 or caption_annotations == 0:
            raise ValueError(
                f"instances_images={instance_images}, categories={categories}, "
                f"caption_images={caption_images}, caption_annotations={caption_annotations}"
            )
        report.add("chair", "PASS", "official COCO instance/caption annotations are structurally valid")
    except Exception as error:
        report.add("chair", "FAIL", f"invalid COCO annotations: {error}")

    test_path = first_file(
        [repo_root / "res-opd" / "data" / "test_1000.json", repo_root / "res-opd" / "data" / "test.json"]
    )
    if test_path is None:
        for benchmark in benchmarks:
            report.add(benchmark, "FAIL", "Res-OPD test JSON is missing from res-opd/data")
        return coco_root
    try:
        samples = load_json(test_path)
        if not isinstance(samples, list) or not samples:
            raise ValueError("test JSON is not a non-empty list")
        unresolved = []
        for sample in samples:
            current = Path(str(sample.get("image_path", "")))
            file_name = sample.get("file_name") or current.name
            if not current.is_file() and not (image_root / str(file_name)).is_file():
                unresolved.append(str(file_name))
        if unresolved:
            raise ValueError(f"unresolved_images={len(unresolved)}, examples={unresolved[:5]}")
        report.add("chair/pope", "PASS", f"test JSON rows={len(samples)}; all image paths are direct or remappable", test_path)
    except Exception as error:
        report.add("chair/pope", "FAIL", f"invalid Res-OPD test JSON: {error}", test_path)
    return coco_root


def check_amber(report, mount_root, sample_count):
    root = first_dir(base / "AMBER" for base in data_roots(mount_root))
    if root is None:
        report.add("amber", "FAIL", "AMBER directory not found")
        return
    required = [
        root / "inference.py",
        root / "data" / "query" / "query_all.json",
        root / "data" / "relation.json",
        root / "data" / "safe_words.txt",
        root / "data" / "annotations.json",
        root / "data" / "metrics.txt",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        report.add("amber", "FAIL", "missing official AMBER files", root, missing)
        return
    image_root = first_dir([root / "images", root / "image", root / "data" / "images"])
    if image_root is None:
        report.add("amber", "FAIL", "AMBER image directory not found", root)
        return
    try:
        queries = load_json(root / "data" / "query" / "query_all.json")
        if not isinstance(queries, list) or not queries:
            raise ValueError("query_all.json is empty or not a list")
        unresolved = []
        names = sorted({str(item.get("image", "")) for item in queries if item.get("image")})
        for name in names:
            if not (image_root / name).is_file():
                unresolved.append(name)
        if unresolved:
            raise ValueError(f"unresolved_images={len(unresolved)}, examples={unresolved[:5]}")
        report.add("amber", "PASS", f"official files valid; queries={len(queries)}, unique_images={len(names)}", root)
        verify_images(report, "amber", [image_root / name for name in names], sample_count)
    except Exception as error:
        report.add("amber", "FAIL", f"invalid AMBER data: {error}", root)


def check_json_image_benchmark(report, mount_root, benchmark, json_name, image_dirs, expected_rows, sample_count):
    roots = data_roots(mount_root)
    json_path = first_file(root / json_name for root in roots)
    if json_path is None:
        report.add(benchmark, "FAIL", f"{json_name} not found")
        return
    root = json_path.parent
    directories = [root / name for name in image_dirs]
    missing_dirs = [str(path) for path in directories if not path.is_dir()]
    if missing_dirs:
        report.add(benchmark, "FAIL", "missing converted image directories", root, missing_dirs)
        return
    try:
        rows = load_json(json_path)
        if not isinstance(rows, list) or len(rows) != expected_rows:
            raise ValueError(f"expected_rows={expected_rows}, actual_rows={len(rows) if isinstance(rows, list) else 'not-list'}")
        direct = 0
        remappable = 0
        unresolved = []
        resolved_paths = []
        for row in rows:
            values = row.get("images") or []
            if not values:
                unresolved.append("<missing images field>")
                continue
            current = Path(str(values[0]))
            if current.is_file():
                direct += 1
                resolved_paths.append(current)
                continue
            candidates = [directory / current.name for directory in directories]
            candidate = next((path for path in candidates if path.is_file()), None)
            if candidate is None:
                unresolved.append(str(values[0]))
            else:
                remappable += 1
                resolved_paths.append(candidate)
        if unresolved:
            raise ValueError(f"unresolved_images={len(unresolved)}, examples={unresolved[:5]}")
        status = "PASS" if remappable == 0 else "WARN"
        message = f"rows={len(rows)}, direct_paths={direct}, basename_remappable={remappable}"
        if remappable:
            message += "; runtime JSON path remapping is required"
        report.add(benchmark, status, message, json_path)
        verify_images(report, benchmark, resolved_paths, sample_count)
    except Exception as error:
        report.add(benchmark, "FAIL", f"invalid benchmark JSON/images: {error}", json_path)


def main():
    args = parse_args()
    requested = {value.strip().lower() for value in args.benchmarks.split(",") if value.strip()}
    supported = {"chair", "pope", "amber", "mmstar", "cv-bench"}
    unknown = requested - supported
    if unknown:
        raise ValueError(f"unsupported benchmarks: {sorted(unknown)}")
    if args.sample_images < 1:
        raise ValueError("--sample-images must be positive")
    if not args.mount_root.is_dir():
        raise FileNotFoundError(f"OSS mount root does not exist: {args.mount_root}")

    report = Report()
    coco_benchmarks = requested & {"chair", "pope"}
    if coco_benchmarks:
        check_coco(report, args.mount_root, args.repo_root, args.sample_images, sorted(coco_benchmarks))
    if "amber" in requested:
        check_amber(report, args.mount_root, args.sample_images)
    if "mmstar" in requested:
        check_json_image_benchmark(
            report,
            args.mount_root,
            "mmstar",
            "mmstar.json",
            ["MMStar_images"],
            1500,
            args.sample_images,
        )
    if "cv-bench" in requested:
        check_json_image_benchmark(
            report,
            args.mount_root,
            "cv-bench",
            "cv_bench.json",
            ["CVBench_2d_images", "CVBench_3d_images"],
            2638,
            args.sample_images,
        )

    summary = {
        "mount_root": str(args.mount_root),
        "benchmarks": sorted(requested),
        "rows": report.rows,
        "failed": report.failed(),
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved JSON report: {args.json_output}")
    print("BENCHMARK_DATA_PREFLIGHT=" + ("FAIL" if report.failed() else "PASS"))
    return 1 if report.failed() else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
