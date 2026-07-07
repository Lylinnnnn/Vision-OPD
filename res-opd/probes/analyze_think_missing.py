import json
import statistics

models = {
    "baseline": "res-opd/eval_results/thinking/full/Qwen3-VL-2B-Thinking/train5000_test1000_original_sr1p0/eval_results.jsonl",
    "rkl_s100": "res-opd/eval_results/thinking/full/Res-OPD-Qwen3-VL-2B-Thinking-orig-sr1.0-tr0.75-a1.0-frozen-rkl-b32-rn4-full5k-e1_global_step_100/train5000_test1000_original_sr1p0/eval_results.jsonl",
    "rkl_s150": "res-opd/eval_results/thinking/full/Res-OPD-Qwen3-VL-2B-Thinking-orig-sr1.0-tr0.75-a1.0-frozen-rkl-b32-rn4-full5k-e1_global_step_150/train5000_test1000_original_sr1p0/eval_results.jsonl",
    "risk_s100": "res-opd/eval_results/thinking/full/Res-OPD-Qwen3-VL-2B-Thinking-orig-sr1.0-tr0.75-a1.0-frozen-rkl-riskmask-nll-p30-b32-rn4-full5k-e1_global_step_100/train5000_test1000_original_sr1p0/eval_results.jsonl",
    "risk_s150": "res-opd/eval_results/thinking/full/Res-OPD-Qwen3-VL-2B-Thinking-orig-sr1.0-tr0.75-a1.0-frozen-rkl-riskmask-nll-p30-b32-rn4-full5k-e1_global_step_150/train5000_test1000_original_sr1p0/eval_results.jsonl",
}

for label, path in models.items():
    total = 0
    has_raw = 0
    no_raw = 0
    raw_has_think_close = 0
    raw_has_think_open_no_close = 0
    raw_no_think_at_all = 0
    raw_lengths_with_close = []
    raw_lengths_no_close = []
    gen_lengths = []
    missing_samples = []

    with open(path) as f:
        for line in f:
            row = json.loads(line)
            total += 1
            raw = str(row.get("raw_generated_caption", "") or "")
            gen = str(row.get("generated_caption", "") or "")
            gen_lengths.append(len(gen))

            if not raw:
                no_raw += 1
                continue

            has_raw += 1

            if "</think>" in raw:
                raw_has_think_close += 1
                raw_lengths_with_close.append(len(raw))
            elif ": {raw_has_think_close} ({raw_has_think_close/total*100:.1f}%)")
    print(f"Has  (truncated?): {raw_has_think_open_no_close} ({raw_has_think_open_no_close/total*100:.1f}%)")
    print(f"No thinking tags at all: {raw_no_think_at_all} ({raw_no_think_at_all/total*100:.1f}%)")

    if raw_lengths_with_close:
        print(f"Raw length WITH </think>: median={statistics.median(raw_lengths_with_close):.0f}, mean={statistics.mean(raw_lengths_with_close):.0f}, min={min(raw_lengths_with_close)}, max={max(raw_lengths_with_close)}")
    if raw_lengths_no_close:
        print(f"Raw length WITHOUT </think>: median={statistics.median(raw_lengths_no_close):.0f}, mean={statistics.mean(raw_lengths_no_close):.0f}, min={min(raw_lengths_no_close)}, max={max(raw_lengths_no_close)}")
    if gen_lengths:
        print(f"generated_caption length: median={statistics.median(gen_lengths):.0f}, mean={statistics.mean(gen_lengths):.0f}")
    print()

    if missing_samples:
        print("Sample missing </think> cases:")
        for reason, text in missing_samples:
            print(f"  [{reason}] {repr(text)}")
    print()
SCRIPT_EOF; __aone_exit=$?; pwd -P > '/tmp/aone-copilot-cwd-1783334041170-8pu5ypgfx2m.txt' 2>/dev/null; exit $__aone_exit
