import json
import os
import random
import argparse
from pathlib import Path
import torch
import spacy
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from transformers import AutoTokenizer
from collections import defaultdict


LLAVA_SERIES_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = LLAVA_SERIES_DIR / "generate" / "llava-v1.5-7b"

name = os.environ.get("LLAVA_CONVERT_NAME", "llava-v1.5-7b")

BASE_DIR = os.environ.get("LLAVA_BASE_DIR", str(DEFAULT_BASE_DIR))
MODEL_PATH = os.environ.get("LLAVA_MODEL_PATH", os.environ.get("MODEL_PATH", "liuhaotian/llava-v1.5-7b"))
MAX_WORKERS = int(os.environ.get("LLAVA_MAX_WORKERS", os.environ.get("MAX_WORKERS", "64")))
STEP1_INPUT_JSONL = os.environ.get("LLAVA_STEP1_INPUT_JSONL", str(DEFAULT_BASE_DIR / "answers.jsonl"))
MERGED_LOGITS_PATH = os.environ.get("LLAVA_MERGED_LOGITS_PATH", str(DEFAULT_BASE_DIR / "logits_merged.pt"))
STEP1_OUTPUT_JSONL = os.environ.get("LLAVA_STEP1_OUTPUT_JSONL", str(DEFAULT_BASE_DIR / f"{name}.jsonl"))
STEP2_OUTPUT_DPO = os.environ.get("LLAVA_STEP2_OUTPUT_DPO", str(DEFAULT_BASE_DIR / f"{name}_dpo_data.json"))
TOP_R = int(os.environ.get("LLAVA_TOP_R", "30"))
TARGET_POS = {"NOUN", "PROPN", "NUM", "ADJ", "VERB", "ADV"}

FILTER_EQUAL_ENTROPY = True
GLOBAL_LOGITS_DATA = None 
worker_tokenizer = None
worker_nlp = None

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate LLaVA answer uncertainty from saved logits and build DPO data."
    )
    parser.add_argument("--name", default=name, help="Output name prefix for entropy JSONL and DPO JSON.")
    parser.add_argument("--base-dir", default=BASE_DIR, help="Base output directory kept for compatibility.")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Tokenizer/model path used for generated token decoding.")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS, help="Number of CPU worker processes.")
    parser.add_argument("--input-jsonl", default=STEP1_INPUT_JSONL, help="Generated answers JSONL.")
    parser.add_argument("--logits-path", default=MERGED_LOGITS_PATH, help="Merged logits .pt produced by generate_llava.sh.")
    parser.add_argument("--entropy-output-jsonl", default=None, help="Optional Step 1 output JSONL path.")
    parser.add_argument("--dpo-output-json", default=None, help="Optional Step 2 DPO JSON output path.")
    parser.add_argument("--top-r", type=int, default=TOP_R, help="Keep top-R key tokens by uncertainty.")
    parser.add_argument(
        "--no-filter-equal-entropy",
        action="store_true",
        help="Do not skip question groups whose max/min entropy are equal.",
    )
    return parser.parse_args()

def configure_from_args(args):
    global name, BASE_DIR, MODEL_PATH, MAX_WORKERS
    global STEP1_INPUT_JSONL, MERGED_LOGITS_PATH, STEP1_OUTPUT_JSONL, STEP2_OUTPUT_DPO
    global TOP_R, FILTER_EQUAL_ENTROPY

    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")
    if args.top_r < 0:
        raise ValueError("--top-r must be >= 0")

    name = args.name
    BASE_DIR = args.base_dir
    MODEL_PATH = args.model_path
    MAX_WORKERS = args.max_workers
    STEP1_INPUT_JSONL = args.input_jsonl
    MERGED_LOGITS_PATH = args.logits_path

    output_dir = os.path.dirname(STEP1_INPUT_JSONL) or "."
    STEP1_OUTPUT_JSONL = args.entropy_output_jsonl or os.path.join(output_dir, f"{name}.jsonl")
    STEP2_OUTPUT_DPO = args.dpo_output_json or os.path.join(output_dir, f"{name}_dpo_data.json")
    TOP_R = args.top_r
    FILTER_EQUAL_ENTROPY = not args.no_filter_equal_entropy

def worker_initializer():
    """Initialize tokenizer and spaCy model once per worker process."""
    global worker_tokenizer, worker_nlp
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        worker_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        worker_nlp = spacy.load("en_core_web_sm")
        worker_nlp.disable_pipes(["ner", "parser", "lemmatizer"])
    except Exception as e:
        print(f"Worker initialization failed: {e}")

def get_key_token_mask(raw_tokens):
    token_strs = []
    SPIECE_UNDERLINE = '▁' 
    for t in raw_tokens:
        s = t.replace(SPIECE_UNDERLINE, ' ') if SPIECE_UNDERLINE in t else t
        if '<0x0A>' in s:
            s = s.replace('<0x0A>', '\n')
        if s.startswith('<') and s.endswith('>'):
            s = '' 
        token_strs.append(s)
    
    full_text = "".join(token_strs)
    if not full_text.strip():
        return [False] * len(raw_tokens), full_text

    doc = worker_nlp(full_text) 
    char_mask = [False] * len(full_text)
    for word in doc:
        if word.pos_ in TARGET_POS and not word.is_punct:
            for i in range(word.idx, word.idx + len(word.text)):
                if i < len(char_mask):
                    char_mask[i] = True

    token_mask = []
    current_char_idx = 0
    for s in token_strs:
        s_len = len(s)
        if s_len == 0:
            token_mask.append(False)
            continue
        span_mask = char_mask[current_char_idx : current_char_idx + s_len]
        token_mask.append(any(span_mask))
        current_char_idx += s_len
    return token_mask, full_text

def fuzzy_minmax_entropy_from_logits(logits):
    T, C = logits.shape
    valid = torch.isfinite(logits)
    K = valid.sum(dim=1)
    
    row_max = torch.where(valid, logits, torch.tensor(float("-inf"))).max(dim=1).values
    row_min = torch.where(valid, logits, torch.tensor(float("inf"))).min(dim=1).values
    denom = row_max - row_min
    
    mu = torch.zeros_like(logits)
    mask = valid & (K > 1).unsqueeze(1)
    denom = torch.where(denom == 0, torch.ones_like(denom), denom) 
    
    mu[mask] = ((logits - row_min.unsqueeze(1)) / denom.unsqueeze(1))[mask]
    
    term = torch.zeros_like(mu)
    mid = mask & (mu > 0.0) & (mu < 1.0)
    pm = mu[mid]
    term[mid] = pm * torch.log(pm) + (1.0 - pm) * torch.log(1.0 - pm)
    
    log2 = torch.log(torch.tensor(2.0))
    fuzzy_entropy = - term.sum(dim=1) / (K.clamp(min=1) * log2)
    return torch.nan_to_num(fuzzy_entropy, nan=0.0)

def process_single_line(line):
    item = json.loads(line)
    ans_id = item.get('answer_id')
    
    result_main = item.copy()
    result_main['key_token_entropies'] = None
    result_main['avg_key_entropy'] = None
    result_main['key_token_ratio'] = 0.0
    
    # Worker processes share the preloaded logits through Linux copy-on-write.
    if ans_id and GLOBAL_LOGITS_DATA and ans_id in GLOBAL_LOGITS_DATA:
        try:
            data = GLOBAL_LOGITS_DATA[ans_id]
            
            logits = data['topk_values']
            
            token_ids = data['generated_tokens']
            if token_ids.dim() > 1:
                token_ids = token_ids.squeeze(0)
            
            min_len = min(logits.shape[0], token_ids.shape[0])
            logits = logits[:min_len]
            token_ids = token_ids[:min_len]

            raw_tokens = worker_tokenizer.convert_ids_to_tokens(token_ids.tolist())
            is_key_token, full_text = get_key_token_mask(raw_tokens)
            
            all_entropies = fuzzy_minmax_entropy_from_logits(logits)

            candidate_tokens = []
            SPIECE_UNDERLINE = '▁' 
            for i, is_key in enumerate(is_key_token):
                if is_key:
                    candidate_tokens.append({
                        "token": raw_tokens[i].replace(SPIECE_UNDERLINE, ' '),
                        "entropy": all_entropies[i].item()
                    })
            
            candidate_tokens.sort(key=lambda x: x['entropy'], reverse=True)
            if TOP_R:
                final_tokens = candidate_tokens[:TOP_R] if TOP_R else candidate_tokens
            else:   
                final_tokens = candidate_tokens
            vals = [x['entropy'] for x in final_tokens]
            
            result_main['key_token_entropies'] = vals
            result_main['avg_key_entropy'] = sum(vals)/len(vals) if vals else 0.0
            result_main['key_token_ratio'] = len(vals) / min_len if min_len > 0 else 0.0
            
            return result_main
            
        except Exception:
            return None
    return None


def run_entropy_calculation():
    print(f"\n{'='*20} Step 1: Running Entropy Calculation {'='*20}")
    print(f"Reading input JSONL: {STEP1_INPUT_JSONL}")
    
    if not os.path.exists(STEP1_INPUT_JSONL):
        print(f"Input file not found: {STEP1_INPUT_JSONL}")
        return False

    with open(STEP1_INPUT_JSONL, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"Processing with {MAX_WORKERS} workers -> {STEP1_OUTPUT_JSONL}")

    success_count = 0
    with open(STEP1_OUTPUT_JSONL, 'w', encoding='utf-8') as f_out:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=worker_initializer) as executor:
            chunk_size = 100 if total_lines >= 1000 else 10
            results = list(tqdm(executor.map(process_single_line, lines, chunksize=chunk_size), total=total_lines))
            
            for res in results:
                if res:
                    f_out.write(json.dumps(res) + "\n")
                    if res.get('avg_key_entropy') is not None:
                        success_count += 1

    print(f"Step 1 finished. Processed: {success_count}/{total_lines}")
    return True


def run_dpo_generation():
    print(f"\n{'='*20} Step 2: Generating DPO Data {'='*20}")
    print(f"Reading Step 1 output: {STEP1_OUTPUT_JSONL}")
    
    groups = defaultdict(list)
    with open(STEP1_OUTPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                groups[item['question_id']].append(item)
            except json.JSONDecodeError:
                continue

    dpo_dataset = []
    skipped_count = 0
    skipped_equal_count = 0

    print(f"Selecting chosen/rejected responses (equal-entropy filter: {FILTER_EQUAL_ENTROPY})...")

    for q_id, items in groups.items():
        if len(items) < 2:
            skipped_count += 1
            continue

        min_entropy = min(items, key=lambda x: x['avg_key_entropy'])['avg_key_entropy']
        max_entropy = max(items, key=lambda x: x['avg_key_entropy'])['avg_key_entropy']

        if FILTER_EQUAL_ENTROPY and (max_entropy == min_entropy):
            skipped_equal_count += 1
            continue

        chosen_candidates = [x for x in items if x['avg_key_entropy'] == min_entropy]
        rejected_candidates = [x for x in items if x['avg_key_entropy'] == max_entropy]

        chosen_item = random.choice(chosen_candidates)
        rejected_item = random.choice(rejected_candidates)

        if chosen_item is rejected_item:
            remaining = [x for x in items if x is not chosen_item]
            if remaining:
                rejected_item = random.choice(remaining)
            else:
                skipped_count += 1
                continue

        clean_prompt = chosen_item['prompt'].replace("<image>\n", "")

        entry = {
            "img_path": chosen_item['image_path'],
            "prompt": clean_prompt,
            "chosen": chosen_item['text'],
            "rejected": rejected_item['text']
        }
        dpo_dataset.append(entry)

    print("DPO summary:")
    print(f"  - question groups: {len(groups)}")
    print(f"  - DPO examples: {len(dpo_dataset)}")
    if FILTER_EQUAL_ENTROPY:
        print(f"  - skipped equal-entropy groups: {skipped_equal_count}")
    if skipped_count > 0:
        print(f"  - skipped insufficient groups: {skipped_count}")

    with open(STEP2_OUTPUT_DPO, 'w', encoding='utf-8') as f:
        json.dump(dpo_dataset, f, indent=4, ensure_ascii=False)
    print(f"Step 2 finished. Saved to: {STEP2_OUTPUT_DPO}")


def main():
    global GLOBAL_LOGITS_DATA

    configure_from_args(parse_args())
    
    if os.path.exists(MERGED_LOGITS_PATH):
        print(f"Loading merged logits into memory: {MERGED_LOGITS_PATH}")
        GLOBAL_LOGITS_DATA = torch.load(MERGED_LOGITS_PATH, map_location="cpu")
        print(f"Loaded {len(GLOBAL_LOGITS_DATA)} logits entries.")
    else:
        print(f"Merged logits file not found: {MERGED_LOGITS_PATH}")
        return

    step1_success = run_entropy_calculation()
    if not step1_success:
        print("Step 1 failed. Stopping.")
        return

    print("Releasing logits memory...")
    del GLOBAL_LOGITS_DATA
    import gc
    gc.collect()

    run_dpo_generation()

    print("\nAll tasks finished.")

if __name__ == "__main__":
    main()
