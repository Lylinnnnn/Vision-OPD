#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path


OFFICIAL_SCHEMA = "official_chair_caption_and_mention_v1"
OFFICIAL_PROMPT = "Describe the image."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="res-opd/eval/manifests/paper_chair_describe45_sources.txt",
    )
    parser.add_argument("--indices", default="1-45")
    parser.add_argument("--output-suffix", default="chair_official_v2")
    parser.add_argument(
        "--fallback-output-suffix",
        action="append",
        default=[],
        help=(
            "Fallback result suffix to try when --output-suffix is absent. "
            "May be specified more than once; every selected result is still "
            "validated against the requested prompt and official metric schema."
        ),
    )
    parser.add_argument(
        "--eval-root",
        default="res-opd/eval_results/instruct/full",
    )
    parser.add_argument(
        "--model-root",
        default=os.environ.get("MODEL_ROOT", "/home/liuyanlin.lyl/notebook/model/qwen"),
    )
    parser.add_argument(
        "--output-prefix",
        default="res-opd/paper/tmp/paper_chair_official_v2_accounting",
    )
    parser.add_argument(
        "--append-md",
        help="Append the generated tables as a new section to an existing Markdown file.",
    )
    parser.add_argument(
        "--append-title",
        default="Additional CHAIR accounting results",
    )
    parser.add_argument("--expected-prompt", default=OFFICIAL_PROMPT)
    parser.add_argument("--expected-samples", type=int, default=1000)
    parser.add_argument(
        "--length-mode",
        choices=("tokenizer", "record", "words"),
        default="tokenizer",
    )
    parser.add_argument(
        "--group-by-scale",
        action="store_true",
        help="Group Markdown tables by model scale and show compact method names.",
    )
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def repo_root():
    return Path(__file__).resolve().parents[3]


def resolve_path(path, root):
    path = Path(path)
    return path if path.is_absolute() else root / path


def parse_indices(spec):
    values = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = (int(value) for value in item.split("-", 1))
            if start < 1 or end < start:
                raise ValueError(f"Invalid index range: {item}")
            values.update(range(start, end + 1))
        else:
            value = int(item)
            if value < 1:
                raise ValueError(f"Invalid index: {item}")
            values.add(value)
    if not values:
        raise ValueError("--indices selected no entries")
    return values


def load_manifest(path):
    entries = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split("|")
        if len(fields) != 7:
            raise ValueError(f"Malformed manifest row: {raw}")
        index, label, source_type, primary, oss_name, step, student_ratio = fields
        entries.append(
            {
                "index": int(index),
                "label": label,
                "source_type": source_type,
                "primary": primary,
                "oss_name": oss_name,
                "step": step,
                "student_ratio": student_ratio,
            }
        )
    indices = [entry["index"] for entry in entries]
    if len(indices) != len(set(indices)):
        raise ValueError("Manifest contains duplicate indices")
    return entries


def ratio_tag(value):
    return str(value).replace(".", "p")


def entry_output_dir(entry, eval_root, suffix):
    if entry["source_type"] == "local_base":
        experiment = Path(entry["primary"]).name
    else:
        experiment = f"{entry['primary']}_{entry['step']}"
    dataset = f"train5000_test1000_original_sr{ratio_tag(entry['student_ratio'])}"
    return eval_root / experiment / dataset / suffix


def model_scale(label):
    match = re.match(r"^(2b|4b|8b)/", label.lower())
    if not match:
        raise ValueError(f"Cannot infer model scale from label: {label}")
    return match.group(1)


def tokenizer_dir(scale, model_root):
    names = {
        "2b": "Qwen3VL-2B-Instruct",
        "4b": "Qwen3VL-4B-Instruct",
        "8b": "Qwen3VL-8B-Instruct",
    }
    return model_root / names[scale]


def method_name(label):
    patterns = (
        ("/base/", "Base"),
        ("/lowres_sft/", "Low-res SFT"),
        ("/frozen_rkl/", "Frozen RKL"),
        ("/randommask/", "RandomMask"),
        ("/entropymask/", "EntropyMask"),
        ("/riskmask/", "RiskMask"),
    )
    lowered = label.lower()
    for marker, name in patterns:
        if marker in lowered:
            return name
    return label


def checkpoint_step(step):
    if not step:
        return "--"
    match = re.fullmatch(r"global_step_(\d+)", step)
    return match.group(1) if match else step


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
    return rows


def final_response_text(row):
    text = str(row.get("generated_caption") or "")
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def load_tokenizer(path):
    if not path.is_dir():
        raise FileNotFoundError(f"Tokenizer model directory does not exist: {path}")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(path),
        trust_remote_code=True,
        local_files_only=True,
    )


def response_length(row, text, mode, tokenizer):
    if mode == "record":
        token_ids = row.get("generated_token_ids")
        if isinstance(token_ids, list):
            return len(token_ids), "generated_token_ids"
        tokens = row.get("generated_tokens")
        if isinstance(tokens, list):
            return len(tokens), "generated_tokens"
        raise ValueError("Record has no generated_token_ids/generated_tokens")
    if mode == "words":
        return len(re.findall(r"\b\w+(?:[-']\w+)*\b", text)), "word_regex"
    return len(tokenizer.encode(text, add_special_tokens=False)), "tokenizer"


def finite_float(value, field):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric field {field}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric field {field}: {value!r}")
    return result


def validate_metrics(metrics, rows, args, metrics_path):
    if metrics.get("metric_schema") != OFFICIAL_SCHEMA:
        raise ValueError(
            f"{metrics_path}: expected metric_schema={OFFICIAL_SCHEMA}, "
            f"got {metrics.get('metric_schema')!r}"
        )
    if len(rows) != args.expected_samples:
        raise ValueError(
            f"{metrics_path}: expected {args.expected_samples} rows, got {len(rows)}"
        )
    scored = int(metrics.get("total_captions", metrics.get("num_samples", 0)))
    if scored != args.expected_samples:
        raise ValueError(
            f"{metrics_path}: expected {args.expected_samples} scored captions, got {scored}"
        )
    if int(metrics.get("invalid_final_answer", 0)) != 0:
        raise ValueError(f"{metrics_path}: invalid_final_answer is nonzero")
    if args.expected_prompt and metrics.get("eval_prompt") != args.expected_prompt:
        raise ValueError(
            f"{metrics_path}: expected prompt {args.expected_prompt!r}, "
            f"got {metrics.get('eval_prompt')!r}"
        )
    total_mentions = int(metrics["total_object_mentions"])
    correct_mentions = int(metrics["total_correct_mentions"])
    hallucinated_mentions = int(metrics["total_hallucinated_mentions"])
    if correct_mentions + hallucinated_mentions != total_mentions:
        raise ValueError(f"{metrics_path}: official mention totals do not add up")
    chair_i = hallucinated_mentions / max(total_mentions, 1)
    chair_s = int(metrics["hallucinated_captions"]) / max(int(metrics["total_captions"]), 1)
    if abs(chair_i - finite_float(metrics["CHAIRi"], "CHAIRi")) > 1e-12:
        raise ValueError(f"{metrics_path}: CHAIRi is inconsistent with mention totals")
    if abs(chair_s - finite_float(metrics["CHAIRs"], "CHAIRs")) > 1e-12:
        raise ValueError(f"{metrics_path}: CHAIRs is inconsistent with caption totals")


def summarize_entry(entry, eval_root, suffixes, args, tokenizer_cache):
    attempted = []
    output_dir = None
    metrics_path = None
    results_path = None
    for suffix in suffixes:
        candidate = entry_output_dir(entry, eval_root, suffix)
        attempted.append(candidate)
        candidate_metrics = candidate / "chair_metrics.json"
        candidate_results = candidate / "eval_results.jsonl"
        if candidate_metrics.is_file() and candidate_results.is_file():
            output_dir = candidate
            metrics_path = candidate_metrics
            results_path = candidate_results
            break
    if output_dir is None:
        attempted_text = "\n  ".join(str(path) for path in attempted)
        raise FileNotFoundError(f"Missing complete CHAIR output; tried:\n  {attempted_text}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = load_jsonl(results_path)
    validate_metrics(metrics, rows, args, metrics_path)

    scale = model_scale(entry["label"])
    tokenizer = None
    if args.length_mode == "tokenizer":
        if scale not in tokenizer_cache:
            tokenizer_cache[scale] = load_tokenizer(tokenizer_dir(scale, Path(args.model_root)))
        tokenizer = tokenizer_cache[scale]

    lengths = []
    words = []
    length_sources = set()
    for row in rows:
        text = final_response_text(row)
        length, source = response_length(row, text, args.length_mode, tokenizer)
        lengths.append(length)
        length_sources.add(source)
        words.append(len(re.findall(r"\b\w+(?:[-']\w+)*\b", text)))

    n = len(rows)
    token_total = sum(lengths)
    mention_total = int(metrics["total_object_mentions"])
    correct_total = int(metrics["total_correct_mentions"])
    hall_total = int(metrics["total_hallucinated_mentions"])
    unique_total = int(metrics["total_mentioned"])
    unique_correct = int(metrics["total_correct"])
    unique_hall = int(metrics["total_hallucinated"])
    obj_recall = finite_float(metrics["ObjRecall"], "ObjRecall")
    unique_gt = round(unique_correct / obj_recall) if obj_recall > 0 else 0
    if unique_gt and abs(unique_correct / unique_gt - obj_recall) > 1e-9:
        raise ValueError(f"{metrics_path}: cannot recover exact unique GT count from ObjRecall")

    return {
        "index": entry["index"],
        "label": entry["label"],
        "method": method_name(entry["label"]),
        "checkpoint_step": checkpoint_step(entry["step"]),
        "scale": scale,
        "output_dir": str(output_dir),
        "n": n,
        "length_mode": args.length_mode,
        "length_source": ",".join(sorted(length_sources)),
        "response_tokens_total": token_total,
        "response_tokens_mean": token_total / n,
        "response_tokens_median": statistics.median(lengths),
        "response_tokens_p90": sorted(lengths)[math.ceil(0.9 * n) - 1],
        "response_words_mean": sum(words) / n,
        "official_pred_mentions": mention_total,
        "official_correct_mentions": correct_total,
        "official_hallucinated_mentions": hall_total,
        "official_pred_mentions_per_caption": mention_total / n,
        "official_correct_mentions_per_caption": correct_total / n,
        "official_hallucinated_mentions_per_caption": hall_total / n,
        "official_pred_mentions_per_100_tokens": mention_total / max(token_total, 1) * 100,
        "official_correct_mentions_per_100_tokens": correct_total / max(token_total, 1) * 100,
        "official_hallucinated_mentions_per_100_tokens": hall_total / max(token_total, 1) * 100,
        "hallucinated_captions": int(metrics["hallucinated_captions"]),
        "CHAIRi": finite_float(metrics["CHAIRi"], "CHAIRi"),
        "CHAIRs": finite_float(metrics["CHAIRs"], "CHAIRs"),
        "unique_pred_categories": unique_total,
        "unique_gt_categories": unique_gt,
        "unique_correct_categories": unique_correct,
        "unique_hallucinated_categories": unique_hall,
        "unique_correct_categories_per_caption": unique_correct / n,
        "unique_hallucinated_categories_per_caption": unique_hall / n,
        "ObjPrec": finite_float(metrics["ObjPrec"], "ObjPrec"),
        "ObjRecall": obj_recall,
        "ObjF1": finite_float(metrics["ObjF1"], "ObjF1"),
    }


def add_base_deltas(rows):
    bases = {
        row["scale"]: row
        for row in rows
        if row["label"] == f"{row['scale']}/base/original"
    }
    delta_fields = {
        "response_tokens_mean": "delta_response_tokens_mean",
        "official_correct_mentions": "delta_official_correct_mentions",
        "official_hallucinated_mentions": "delta_official_hallucinated_mentions",
        "official_correct_mentions_per_caption": "delta_correct_mentions_per_caption",
        "official_hallucinated_mentions_per_caption": "delta_hallucinated_mentions_per_caption",
        "CHAIRi": "delta_CHAIRi",
        "CHAIRs": "delta_CHAIRs",
        "ObjRecall": "delta_ObjRecall",
        "ObjF1": "delta_ObjF1",
    }
    for row in rows:
        base = bases.get(row["scale"])
        row["base_label"] = base["label"] if base else None
        for source, target in delta_fields.items():
            row[target] = row[source] - base[source] if base else None


def fmt(value, digits=4, signed=False):
    if value is None:
        return "--"
    prefix = "+" if signed else ""
    return f"{value:{prefix}.{digits}f}"


def markdown_table(rows, experiment_header="Experiment", compact_names=False):
    lines = [
        f"| {experiment_header} | Step | Tok/img | Delta tok | Correct mentions | Delta correct | Hall mentions | Delta hall | Correct/img | Hall/img | Hall/100 tok | CHAIRi | CHAIRs | ObjRecall | ObjF1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        name = row["method"] if compact_names else row["label"]
        lines.append(
            "| {name} | {step} | {tok} | {dtok} | {correct} | {dcorrect} | "
            "{hall} | {dhall} | {correct_img} | {hall_img} | {hall_100} | "
            "{chair_i} | {chair_s} | {recall} | {f1} |".format(
                name=name,
                step=row["checkpoint_step"],
                tok=fmt(row["response_tokens_mean"], 2),
                dtok=fmt(row["delta_response_tokens_mean"], 2, True),
                correct=row["official_correct_mentions"],
                dcorrect=fmt(row["delta_official_correct_mentions"], 0, True),
                hall=row["official_hallucinated_mentions"],
                dhall=fmt(row["delta_official_hallucinated_mentions"], 0, True),
                correct_img=fmt(row["official_correct_mentions_per_caption"], 3),
                hall_img=fmt(row["official_hallucinated_mentions_per_caption"], 3),
                hall_100=fmt(row["official_hallucinated_mentions_per_100_tokens"], 3),
                chair_i=fmt(row["CHAIRi"], 4),
                chair_s=fmt(row["CHAIRs"], 4),
                recall=fmt(row["ObjRecall"], 4),
                f1=fmt(row["ObjF1"], 4),
            )
        )
    return lines


def unique_markdown_table(rows, experiment_header="Experiment", compact_names=False):
    lines = [
        f"| {experiment_header} | Step | Unique GT | Unique pred | Unique correct | Unique hall | Correct/img | Hall/img | ObjPrec | ObjRecall | ObjF1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        name = row["method"] if compact_names else row["label"]
        lines.append(
            "| {name} | {step} | {gt} | {pred} | {correct} | {hall} | {correct_img} | "
            "{hall_img} | {precision} | {recall} | {f1} |".format(
                name=name,
                step=row["checkpoint_step"],
                gt=row["unique_gt_categories"],
                pred=row["unique_pred_categories"],
                correct=row["unique_correct_categories"],
                hall=row["unique_hallucinated_categories"],
                correct_img=fmt(row["unique_correct_categories_per_caption"], 3),
                hall_img=fmt(row["unique_hallucinated_categories_per_caption"], 3),
                precision=fmt(row["ObjPrec"], 4),
                recall=fmt(row["ObjRecall"], 4),
                f1=fmt(row["ObjF1"], 4),
            )
        )
    return lines


def markdown(rows, group_by_scale=False):
    lines = [
        "# Official CHAIR Response and Mention Accounting",
        "",
        "Official mention counts include repeated object mentions. Unique-category counts deduplicate each canonical object category within a caption and are the basis of ObjPrec/ObjRecall/ObjF1.",
        "",
    ]
    if group_by_scale:
        scale_titles = {
            "2b": "Qwen3VL-2B-Instruct",
            "4b": "Qwen3VL-4B-Instruct",
            "8b": "Qwen3VL-8B-Instruct",
        }
        for scale in ("2b", "4b", "8b"):
            scale_rows = [row for row in rows if row["scale"] == scale]
            if not scale_rows:
                continue
            lines.extend([f"## {scale_titles[scale]}", ""])
            lines.extend(markdown_table(scale_rows, "Method", compact_names=True))
            lines.extend(["", "### Unique-category accounting", ""])
            lines.extend(unique_markdown_table(scale_rows, "Method", compact_names=True))
            lines.append("")
    else:
        lines.extend(markdown_table(rows))
        lines.extend(["", "## Unique-category accounting", ""])
        lines.extend(unique_markdown_table(rows))
    return "\n".join(lines) + "\n"


def append_markdown(rows, path, title, group_by_scale=False):
    if not path.is_file():
        raise FileNotFoundError(f"Append target does not exist: {path}")
    marker_id = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    begin_marker = f"<!-- BEGIN chair-accounting-{marker_id} -->"
    end_marker = f"<!-- END chair-accounting-{marker_id} -->"
    existing = path.read_text(encoding="utf-8")
    if begin_marker in existing or end_marker in existing:
        raise ValueError(f"Append section already exists in {path}: {title}")

    section = markdown(rows, group_by_scale=group_by_scale)
    section = section.replace(
        "# Official CHAIR Response and Mention Accounting",
        f"## {title}",
        1,
    ).replace("## Unique-category accounting", "### Unique-category accounting", 1)
    with path.open("a", encoding="utf-8") as handle:
        if existing and not existing.endswith("\n"):
            handle.write("\n")
        handle.write(f"\n{begin_marker}\n{section}{end_marker}\n")


def write_outputs(
    rows,
    output_prefix,
    metadata,
    append_md=None,
    append_title=None,
    group_by_scale=False,
):
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    if append_md is not None and append_md.resolve() == md_path.resolve():
        raise ValueError("--append-md must differ from the generated --output-prefix Markdown path")

    json_path.write_text(
        json.dumps({"metadata": metadata, "experiments": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(markdown(rows, group_by_scale=group_by_scale), encoding="utf-8")
    outputs = [json_path, csv_path, md_path]
    if append_md is not None:
        append_markdown(rows, append_md, append_title, group_by_scale=group_by_scale)
        outputs.append(append_md)
    return outputs


def main():
    args = parse_args()
    root = repo_root()
    manifest_path = resolve_path(args.manifest, root)
    eval_root = resolve_path(args.eval_root, root)
    output_prefix = resolve_path(args.output_prefix, root)
    append_md = resolve_path(args.append_md, root) if args.append_md else None
    requested = parse_indices(args.indices)
    entries = [entry for entry in load_manifest(manifest_path) if entry["index"] in requested]
    missing_indices = requested - {entry["index"] for entry in entries}
    if missing_indices:
        raise ValueError(f"Indices absent from manifest: {sorted(missing_indices)}")

    rows = []
    skipped = []
    tokenizer_cache = {}
    suffixes = list(dict.fromkeys([args.output_suffix, *args.fallback_output_suffix]))
    for entry in entries:
        try:
            row = summarize_entry(entry, eval_root, suffixes, args, tokenizer_cache)
        except FileNotFoundError as error:
            if not args.allow_missing:
                raise
            skipped.append({"index": entry["index"], "label": entry["label"], "reason": str(error)})
            print(f"WARNING: {error}", file=sys.stderr)
            continue
        rows.append(row)
        print(
            f"[{entry['index']:02d}] {entry['label']}: "
            f"tokens={row['response_tokens_mean']:.2f} "
            f"correct={row['official_correct_mentions']} "
            f"hall={row['official_hallucinated_mentions']}",
            flush=True,
        )
    if not rows:
        raise RuntimeError("No complete experiment outputs were summarized")

    add_base_deltas(rows)
    metadata = {
        "manifest": str(manifest_path),
        "indices": sorted(requested),
        "output_suffix": args.output_suffix,
        "fallback_output_suffixes": args.fallback_output_suffix,
        "expected_prompt": args.expected_prompt,
        "expected_samples": args.expected_samples,
        "length_mode": args.length_mode,
        "model_root": args.model_root,
        "group_by_scale": args.group_by_scale,
        "skipped": skipped,
    }
    outputs = write_outputs(
        rows,
        output_prefix,
        metadata,
        append_md=append_md,
        append_title=args.append_title,
        group_by_scale=args.group_by_scale,
    )
    print("\nSaved:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
