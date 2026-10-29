#!/usr/bin/env python3

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path


OFFICIAL_SCHEMA = "official_chair_caption_and_mention_v1"
OFFICIAL_PROMPT = "Describe the image."
BEGIN_MARKER = "<!-- BEGIN external-vcd-chair-official -->"
END_MARKER = "<!-- END external-vcd-chair-official -->"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate a complete VCD CHAIR run and append its audited summary"
    )
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--append-md", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--expected-model-name", default="Qwen3VL-2B-Instruct")
    parser.add_argument("--expected-samples", type=int, default=1000)
    parser.add_argument("--expected-shards", type=int, default=8)
    parser.add_argument("--expected-prompt", default=OFFICIAL_PROMPT)
    parser.add_argument("--expected-alpha", type=float, default=1.0)
    parser.add_argument("--expected-beta", type=float, default=0.1)
    parser.add_argument("--expected-noise-step", type=int, default=500)
    parser.add_argument("--expected-decode-mode", default="sample")
    parser.add_argument(
        "--length-mode",
        choices=("tokenizer", "record"),
        default="tokenizer",
    )
    return parser.parse_args()


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON file {path}: {error}") from error


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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_number(actual, expected, name, tolerance=1e-12):
    try:
        actual_value = float(actual)
        expected_value = float(expected)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric value for {name}: {actual!r}") from error
    require(math.isfinite(actual_value), f"Non-finite value for {name}: {actual!r}")
    require(
        abs(actual_value - expected_value) <= tolerance,
        f"Unexpected {name}: expected {expected_value}, got {actual_value}",
    )


def validate_metrics(metrics, args):
    require(
        metrics.get("metric_schema") == OFFICIAL_SCHEMA,
        f"Expected metric_schema={OFFICIAL_SCHEMA}, got {metrics.get('metric_schema')!r}",
    )
    for field in ("total_results", "num_samples", "total_captions"):
        require(
            int(metrics.get(field, -1)) == args.expected_samples,
            f"Expected {field}={args.expected_samples}, got {metrics.get(field)!r}",
        )
    require(int(metrics.get("invalid_final_answer", -1)) == 0, "invalid_final_answer is nonzero")
    require(int(metrics.get("unclosed_thinking", -1)) == 0, "unclosed_thinking is nonzero")
    same_number(metrics.get("final_answer_valid_rate"), 1.0, "final_answer_valid_rate")
    require(metrics.get("merged_from_shards") is True, "merged_from_shards is not true")
    require(
        int(metrics.get("shard_count", -1)) == args.expected_shards,
        f"Expected shard_count={args.expected_shards}, got {metrics.get('shard_count')!r}",
    )
    require(
        metrics.get("eval_prompt") == args.expected_prompt,
        f"Expected eval_prompt={args.expected_prompt!r}, got {metrics.get('eval_prompt')!r}",
    )

    total_mentions = int(metrics["total_object_mentions"])
    correct_mentions = int(metrics["total_correct_mentions"])
    hallucinated_mentions = int(metrics["total_hallucinated_mentions"])
    total_captions = int(metrics["total_captions"])
    hallucinated_captions = int(metrics["hallucinated_captions"])
    require(
        correct_mentions + hallucinated_mentions == total_mentions,
        "Official object mention totals do not add up",
    )
    same_number(
        metrics.get("CHAIRi"),
        hallucinated_mentions / max(total_mentions, 1),
        "CHAIRi",
    )
    same_number(
        metrics.get("CHAIRs"),
        hallucinated_captions / max(total_captions, 1),
        "CHAIRs",
    )


def validate_config(config, args, expected_source_commit):
    expected = {
        "method": "vcd",
        "model_name": args.expected_model_name,
        "prompt": args.expected_prompt,
        "alpha": args.expected_alpha,
        "beta": args.expected_beta,
        "noise_step": args.expected_noise_step,
        "decode_mode": args.expected_decode_mode,
        "degradation_mode": "original",
        "student_ratio": 1.0,
    }
    for key, value in expected.items():
        actual = config.get(key)
        if isinstance(value, float):
            same_number(actual, value, f"vcd_config.{key}")
        else:
            require(actual == value, f"Expected vcd_config.{key}={value!r}, got {actual!r}")
    require(
        config.get("source_commit") == expected_source_commit,
        "VCD source commit does not match the pinned baseline registry",
    )


def validate_rows(rows, config, args, tokenizer):
    require(
        len(rows) == args.expected_samples,
        f"Expected {args.expected_samples} JSONL rows, got {len(rows)}",
    )
    image_ids = [row.get("image_id") for row in rows]
    require(all(image_id is not None for image_id in image_ids), "Some rows have no image_id")
    require(len(set(image_ids)) == len(image_ids), "Duplicate image_id values found")

    total_tokens = 0
    for row_index, row in enumerate(rows, 1):
        prefix = f"row {row_index} image_id={row.get('image_id')}"
        caption = row.get("generated_caption")
        require(isinstance(caption, str) and caption.strip(), f"{prefix}: empty generated_caption")
        require(not caption.startswith("[ERROR]"), f"{prefix}: generation error")
        require(row.get("eval_prompt") == args.expected_prompt, f"{prefix}: prompt mismatch")
        require(row.get("decoding_method") == "vcd", f"{prefix}: decoding_method is not vcd")
        require(row.get("eval_degradation_mode") == "original", f"{prefix}: input is not original")
        same_number(row.get("eval_student_ratio"), 1.0, f"{prefix}.eval_student_ratio")
        require(
            row.get("vcd_source_commit") == config["source_commit"],
            f"{prefix}: VCD source commit mismatch",
        )
        same_number(row.get("vcd_alpha"), args.expected_alpha, f"{prefix}.vcd_alpha")
        same_number(row.get("vcd_beta"), args.expected_beta, f"{prefix}.vcd_beta")
        require(
            int(row.get("vcd_noise_step", -1)) == args.expected_noise_step,
            f"{prefix}: noise step mismatch",
        )
        require(
            row.get("vcd_decode_mode") == args.expected_decode_mode,
            f"{prefix}: decode mode mismatch",
        )
        token_ids = row.get("generated_token_ids")
        token_count = row.get("generated_token_count")
        require(isinstance(token_ids, list), f"{prefix}: generated_token_ids is missing")
        require(isinstance(token_count, int), f"{prefix}: generated_token_count is missing")
        require(token_count == len(token_ids), f"{prefix}: generated token count mismatch")
        if args.length_mode == "tokenizer":
            total_tokens += len(tokenizer.encode(caption.strip(), add_special_tokens=False))
        else:
            total_tokens += token_count
    return total_tokens


def build_summary(result_dir, rows, metrics, config, total_tokens, length_mode):
    n = len(rows)
    total_mentions = int(metrics["total_object_mentions"])
    correct_mentions = int(metrics["total_correct_mentions"])
    hall_mentions = int(metrics["total_hallucinated_mentions"])
    return {
        "validation_status": "PASS",
        "result_dir": str(result_dir.resolve()),
        "method": "VCD",
        "method_type": "training_free_decoding",
        "model_name": config["model_name"],
        "n": n,
        "response_tokens_total": total_tokens,
        "response_tokens_mean": total_tokens / n,
        "response_length_mode": length_mode,
        "official_pred_mentions": total_mentions,
        "official_correct_mentions": correct_mentions,
        "official_hallucinated_mentions": hall_mentions,
        "official_correct_mentions_per_caption": correct_mentions / n,
        "official_hallucinated_mentions_per_caption": hall_mentions / n,
        "official_hallucinated_mentions_per_100_tokens": hall_mentions / max(total_tokens, 1) * 100,
        "CHAIRi": float(metrics["CHAIRi"]),
        "CHAIRs": float(metrics["CHAIRs"]),
        "ObjPrec": float(metrics["ObjPrec"]),
        "ObjRecall": float(metrics["ObjRecall"]),
        "ObjF1": float(metrics["ObjF1"]),
        "prompt": config["prompt"],
        "metric_schema": metrics["metric_schema"],
        "merged_from_shards": metrics["merged_from_shards"],
        "shard_count": int(metrics["shard_count"]),
        "vcd_config": config,
    }


def markdown_section(summary):
    config = summary["vcd_config"]
    lines = [
        BEGIN_MARKER,
        "## External inference-time decoding baseline",
        "",
        (
            "VCD is a training-free decoding baseline rather than a selected training "
            "checkpoint. The row below uses the same original-image input, `Describe the "
            "image.` prompt, 1,000-caption split, and official CHAIR caption/mention schema."
        ),
        "",
        (
            "| Method | Decode | Tok/img | Correct mentions | Hall mentions | Hall/100 tok | "
            "CHAIRi | CHAIRs | ObjRecall | ObjF1 |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| VCD | $\\alpha={config['alpha']:g}$, $\\beta={config['beta']:g}$, "
            f"noise={config['noise_step']} | {summary['response_tokens_mean']:.2f} | "
            f"{summary['official_correct_mentions']} | "
            f"{summary['official_hallucinated_mentions']} | "
            f"{summary['official_hallucinated_mentions_per_100_tokens']:.3f} | "
            f"{summary['CHAIRi']:.4f} | {summary['CHAIRs']:.4f} | "
            f"{summary['ObjRecall']:.4f} | {summary['ObjF1']:.4f} |"
        ),
        "",
        END_MARKER,
        "",
    ]
    return "\n".join(lines)


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def upsert_section(document, section):
    has_begin = BEGIN_MARKER in document
    has_end = END_MARKER in document
    require(has_begin == has_end, "Target Markdown has only one VCD boundary marker")
    if has_begin:
        pattern = re.compile(
            re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER) + r"\n?",
            re.DOTALL,
        )
        require(len(pattern.findall(document)) == 1, "Target Markdown has duplicate VCD sections")
        return pattern.sub(section, document, count=1)
    separator = "" if not document or document.endswith("\n\n") else "\n"
    return document + separator + section


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    append_md = Path(args.append_md)
    summary_json = Path(args.summary_json)
    model_path = Path(args.model_path)
    require(result_dir.is_dir(), f"Result directory does not exist: {result_dir}")
    require(append_md.is_file(), f"Append target does not exist: {append_md}")
    require(model_path.is_dir(), f"Model path does not exist: {model_path}")
    require(
        model_path.name == args.expected_model_name,
        f"Expected model path basename {args.expected_model_name!r}, got {model_path.name!r}",
    )

    metrics_path = result_dir / "chair_metrics.json"
    results_path = result_dir / "eval_results.jsonl"
    config_path = result_dir / "vcd_config.json"
    for path in (metrics_path, results_path, config_path):
        require(path.is_file(), f"Required output is missing: {path}")

    metrics = load_json(metrics_path)
    config = load_json(config_path)
    rows = load_jsonl(results_path)
    registry_path = Path(__file__).resolve().parent / "registry.json"
    registry = load_json(registry_path)
    expected_source_commit = registry["baselines"]["vcd"]["commit"]
    validate_metrics(metrics, args)
    validate_config(config, args, expected_source_commit)
    tokenizer = None
    if args.length_mode == "tokenizer":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=True,
        )
    total_tokens = validate_rows(rows, config, args, tokenizer)
    summary = build_summary(
        result_dir,
        rows,
        metrics,
        config,
        total_tokens,
        args.length_mode,
    )
    section = markdown_section(summary)
    updated_document = upsert_section(append_md.read_text(encoding="utf-8"), section)

    atomic_write(summary_json, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    atomic_write(append_md, updated_document)
    print("VCD_CHAIR_VALIDATION=PASS")
    print(f"Rows: {summary['n']}")
    print(f"CHAIRi: {summary['CHAIRi']:.6f}")
    print(f"CHAIRs: {summary['CHAIRs']:.6f}")
    print(f"ObjF1: {summary['ObjF1']:.6f}")
    print(f"Updated: {append_md}")
    print(f"Audit JSON: {summary_json}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"ERROR: {error}")
