import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def is_correct(item):
    return str(item.get("judge", "")).strip().lower() == "yes"


def acc_text(correct, total):
    pct = (100.0 * correct / total) if total else 0.0
    return f"{correct}/{total} = {pct:.2f}%"


def calc_vstar(judge_json, benchmark):
    with open(judge_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    overall_correct = 0

    for item in data:
        category = str(item.get("category", "unknown") or "unknown")
        category_stats[category]["total"] += 1
        if is_correct(item):
            overall_correct += 1
            category_stats[category]["correct"] += 1

    for category, stats in category_stats.items():
        total = stats["total"]
        acc = (100.0 * stats["correct"] / total) if total else 0.0
        print(f"{category}: {acc:.2f}% (n={total})")

    overall_acc = (100.0 * overall_correct / len(data)) if data else 0.0
    print(f"{benchmark} Acc: {overall_acc:.2f}% (n={len(data)})")


def calc_hrbench(judge_json, benchmark, benchmark_json):
    with open(judge_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    with open(benchmark_json, "r", encoding="utf-8") as f:
        benchmark_records = json.load(f)

    def image_key(item):
        images = item.get("images") or []
        return str(images[0]) if isinstance(images, list) and images else ""

    def text_key(item):
        return (str(item.get("query", "") or ""), str(item.get("response", "") or ""))

    category_by_image = {}
    category_by_text = {}
    for item in benchmark_records:
        category = str(item.get("category", "unknown") or "unknown")
        img = image_key(item)
        txt = text_key(item)
        if img:
            category_by_image[img] = category
        category_by_text[txt] = category

    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    overall_correct = 0

    for item in records:
        correct = is_correct(item)
        category = str(item.get("category", "") or "").strip()
        if not category:
            category = category_by_image.get(image_key(item)) or category_by_text.get(text_key(item)) or "unknown"
        category_stats[category]["total"] += 1
        if correct:
            overall_correct += 1
            category_stats[category]["correct"] += 1

    print(f"{benchmark} Acc: {acc_text(overall_correct, len(records))}")


def calc_mme_realworld(judge_json, benchmark, benchmark_json):
    with open(judge_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    with open(benchmark_json, "r", encoding="utf-8") as f:
        benchmark_records = json.load(f)

    def image_key(item):
        images = item.get("images") or []
        return str(images[0]) if isinstance(images, list) and images else ""

    def text_key(item):
        return (str(item.get("query", "") or ""), str(item.get("response", "") or ""))

    category_by_image = {}
    category_by_text = {}
    for item in benchmark_records:
        category = str(item.get("category", "unknown") or "unknown")
        img = image_key(item)
        txt = text_key(item)
        if img:
            category_by_image[img] = category
        category_by_text[txt] = category

    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    supercategory_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    overall_correct = 0

    for item in records:
        correct = is_correct(item)
        category = str(item.get("category", "") or "").strip()
        if not category:
            category = category_by_image.get(image_key(item)) or category_by_text.get(text_key(item)) or "unknown"
        supercategory = category.split("/", 1)[0].strip() if "/" in category else category.strip()
        if not supercategory:
            supercategory = "unknown"
        category_stats[category]["total"] += 1
        supercategory_stats[supercategory]["total"] += 1
        if correct:
            overall_correct += 1
            category_stats[category]["correct"] += 1
            supercategory_stats[supercategory]["correct"] += 1

    print(f"{benchmark} Acc: {acc_text(overall_correct, len(records))}")


def _pope_metrics(items):
    tp = fp = fn = tn = 0
    for item in items:
        gt = str(item.get("response", "")).strip().lower()
        correct = is_correct(item)
        if gt == "yes" and correct:
            tp += 1
        elif gt == "no" and not correct:
            fp += 1
        elif gt == "yes" and not correct:
            fn += 1
        else:
            tn += 1
    total = len(items)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    yes_count = sum(1 for x in items if str(x.get("response", "")).strip().lower() == "yes")
    yes_ratio = yes_count / total if total else 0.0
    return accuracy, precision, recall, f1, yes_ratio, total


def _pope_fmt(label, accuracy, precision, recall, f1, yes_ratio, total):
    return (
        f"{label}: accuracy={100*accuracy:.2f}% precision={100*precision:.2f}% "
        f"recall={100*recall:.2f}% f1={100*f1:.2f}% yes_ratio={100*yes_ratio:.2f}% (n={total})"
    )


def calc_pope(judge_json, benchmark):
    with open(judge_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    a, p, r, f1, yr, n = _pope_metrics(data)
    print(_pope_fmt(benchmark, a, p, r, f1, yr, n))


def calc_cvbench(judge_json, benchmark):
    with open(judge_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for item in data:
        dim = str(item.get("type", "unknown") or "unknown")
        type_stats[dim]["total"] += 1
        if is_correct(item):
            type_stats[dim]["correct"] += 1

    acc_2d = type_stats["2D"]["correct"] / type_stats["2D"]["total"] if type_stats["2D"]["total"] else 0.0
    acc_3d = type_stats["3D"]["correct"] / type_stats["3D"]["total"] if type_stats["3D"]["total"] else 0.0
    acc_avg = (acc_2d + acc_3d) / 2.0

    print(f"{benchmark}: {100*acc_avg:.2f}%")


def _item_key(item):
    for key in ("sample_uid", "index", "question_id", "id"):
        value = item.get(key)
        if value is not None and str(value) != "":
            return f"{key}:{value}"
    images = item.get("images") or []
    image0 = images[0] if isinstance(images, list) and images else ""
    return json.dumps({"image": image0, "query": item.get("query", "")}, ensure_ascii=False, sort_keys=True)


def _load_benchmark_meta(benchmark_json):
    if not benchmark_json:
        return {}
    path = Path(benchmark_json)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        return {}
    return {_item_key(item): item for item in rows if isinstance(item, dict)}


def _safe_group(item, meta, key):
    value = item.get(key)
    if value is None or str(value).strip() == "":
        value = meta.get(key) if isinstance(meta, dict) else None
    if value is None or str(value).strip() == "":
        return "unknown"
    return str(value).strip()


def _write_rows_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_breakdown_md(path, title, rows):
    lines = [
        f"# {title}",
        "",
        "| Group | N | Correct | Accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['n']} | {row['correct']} | {100.0 * row['accuracy']:.2f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summarize_groups(records, meta_by_key, group_key):
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for item in records:
        meta = meta_by_key.get(_item_key(item), {})
        group = _safe_group(item, meta, group_key)
        stats[group]["total"] += 1
        if is_correct(item):
            stats[group]["correct"] += 1
    rows = []
    for group, stat in sorted(stats.items(), key=lambda kv: (-kv[1]["total"], kv[0])):
        total = stat["total"]
        correct = stat["correct"]
        rows.append(
            {
                "group_key": group_key,
                "group": group,
                "n": total,
                "correct": correct,
                "accuracy": correct / total if total else 0.0,
            }
        )
    return rows


def calc_mmstar(judge_json, benchmark, benchmark_json):
    with open(judge_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    correct = sum(1 for item in data if is_correct(item))
    acc = correct / total if total else 0.0
    print(f"{benchmark} Acc: {correct}/{total} = {100 * acc:.2f}%")

    meta_by_key = _load_benchmark_meta(benchmark_json)
    category_rows = _summarize_groups(data, meta_by_key, "category")
    l2_rows = _summarize_groups(data, meta_by_key, "l2_category")

    output_dir = Path(judge_json).parent.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "benchmark": benchmark,
        "accuracy": acc,
        "correct": correct,
        "num_samples": total,
        "source_path": str(judge_json),
        "category": category_rows,
        "l2_category": l2_rows,
    }
    summary_path = output_dir / "mmstar_breakdown.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _write_rows_csv(output_dir / "mmstar_category_breakdown.csv", category_rows)
    _write_rows_csv(output_dir / "mmstar_l2_category_breakdown.csv", l2_rows)
    _write_breakdown_md(output_dir / "mmstar_category_breakdown.md", "MMStar Category Breakdown", category_rows)
    _write_breakdown_md(output_dir / "mmstar_l2_category_breakdown.md", "MMStar L2 Category Breakdown", l2_rows)
    print(f"Saved MMStar breakdown: {summary_path}")


def calc_visualprobe(judge_json, benchmark):
    with open(judge_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for item in data:
        category = str(item.get("category", "unknown") or "unknown")
        cat_stats[category]["total"] += 1
        if is_correct(item):
            cat_stats[category]["correct"] += 1

    accs = []
    for cat in ["Easy", "Medium", "Hard"]:
        stats = cat_stats[cat]
        acc = (100.0 * stats["correct"] / stats["total"]) if stats["total"] else 0.0
        accs.append(acc)
        print(f"  {cat}: {acc:.2f}% (n={stats['total']})")

    avg_acc = sum(accs) / len(accs) if accs else 0.0
    print(f"{benchmark} average: {avg_acc:.2f}%")


def calc_generic(judge_json, benchmark):
    with open(judge_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    n = len(data)
    c = sum(1 for x in data if is_correct(x))
    print(f"{benchmark} Acc: {c}/{n} = {100 * c / n:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Calculate accuracy from judge results")
    parser.add_argument("--benchmark", required=True, type=str)
    parser.add_argument("--judge_json", required=True, type=str, help="Path to judge output JSON")
    parser.add_argument("--benchmark_json", default=None, type=str, help="Path to original benchmark JSON (for category breakdown)")
    args = parser.parse_args()

    if args.benchmark == "visualprobe":
        calc_visualprobe(args.judge_json, args.benchmark)
    elif args.benchmark == "mmstar":
        calc_mmstar(args.judge_json, args.benchmark, args.benchmark_json)
    elif args.benchmark == "cv-bench":
        calc_cvbench(args.judge_json, args.benchmark)
    elif args.benchmark in ("pope", "pope_adv", "pope_pop", "pope_random"):
        calc_pope(args.judge_json, args.benchmark)
    elif args.benchmark == "vstar":
        calc_vstar(args.judge_json, args.benchmark)
    elif args.benchmark in ("hrbench-4k", "hrbench-8k"):
        if not args.benchmark_json:
            print("WARNING: --benchmark_json not provided, falling back to generic accuracy.", file=sys.stderr)
            calc_generic(args.judge_json, args.benchmark)
        else:
            calc_hrbench(args.judge_json, args.benchmark, args.benchmark_json)
    elif args.benchmark in ("mme-realworld", "mme-realworld-cn", "mme-realworld-lite"):
        if not args.benchmark_json:
            print("WARNING: --benchmark_json not provided, falling back to generic accuracy.", file=sys.stderr)
            calc_generic(args.judge_json, args.benchmark)
        else:
            calc_mme_realworld(args.judge_json, args.benchmark, args.benchmark_json)
    else:
        calc_generic(args.judge_json, args.benchmark)


if __name__ == "__main__":
    main()
