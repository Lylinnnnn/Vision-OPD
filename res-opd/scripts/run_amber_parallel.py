#!/usr/bin/env python3
"""Parallel wrapper for AMBER official inference.py evaluation.

Splits inference data into chunks, runs inference.py logic in parallel
via multiprocessing, then merges partial metrics and prints final results.
"""

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

# Add AMBER root to path so we can import from inference.py
AMBER_ROOT = Path("/home/liuyanlin.lyl/notebook/data/AMBER")
sys.path.insert(0, str(AMBER_ROOT))

import nltk
from nltk.stem import WordNetLemmatizer
import spacy
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def load_shared_data(amber_root, word_association, safe_words, annotation, metrics_file):
    """Load all shared data needed by workers."""
    association = json.load(open(word_association, "r", encoding="utf-8"))
    hallucination_words = []
    for word1 in association.keys():
        hallucination_words.append(word1)
        for word2 in association[word1]:
            hallucination_words.append(word2)

    global_safe_words = []
    with open(safe_words, "r", encoding="utf-8") as f:
        for line in f:
            global_safe_words.append(line.strip())

    ground_truth = json.load(open(annotation, "r", encoding="utf-8"))

    metrics_template = {}
    with open(metrics_file, "r") as f:
        for line in f:
            parts = line.strip().split("=")
            if len(parts) == 2:
                metrics_template[parts[0].strip()] = eval(parts[1].strip())

    return {
        "association": association,
        "hallucination_words": hallucination_words,
        "global_safe_words": global_safe_words,
        "ground_truth": ground_truth,
        "metrics_template": metrics_template,
    }


def process_chunk(args):
    """Process a chunk of inference data and return partial metrics."""
    chunk, shared, similarity_score = args

    nlp = spacy.load("en_core_web_lg")
    lemmatizer = WordNetLemmatizer()

    metrics = {k: 0 for k in shared["metrics_template"]}
    association = shared["association"]
    hallucination_words = shared["hallucination_words"]
    global_safe_words = shared["global_safe_words"]
    ground_truth = shared["ground_truth"]

    def check_synonyms_word(word1, word2):
        token1 = nlp(word1)
        token2 = nlp(word2)
        return token1.similarity(token2) > similarity_score

    def extract_nouns(text):
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        return [lemmatizer.lemmatize(word) for word, pos in tagged if pos.startswith("NN")]

    for item in chunk:
        item_id = item["id"]
        gt = ground_truth[item_id - 1]

        if gt["type"] == "generative":
            nouns = extract_nouns(item["response"])
            after_process_nouns = [n for n in nouns if n in hallucination_words]

            safe_words = []
            safe_list = []
            for idx, word in enumerate(gt["truth"]):
                safe_words += association[word]
                safe_list += [idx] * len(association[word])

            ha_words = []
            ha_list = []
            for idx, word in enumerate(gt["hallu"]):
                ha_words += association[word]
                ha_list += [idx] * len(association[word])

            safe_words += gt["truth"]
            safe_len = len(gt["truth"])
            safe_list += [0] * safe_len
            safe_flag_list = [0] * len(after_process_nouns)

            ha_words += gt["hallu"]
            ha_len = len(gt["hallu"])
            ha_list += [0] * ha_len

            for idx, noun in enumerate(after_process_nouns):
                if noun in global_safe_words:
                    continue
                if noun in safe_words:
                    for j in range(len(safe_words)):
                        if noun == safe_words[j]:
                            if j < (len(safe_list) - safe_len):
                                safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                            else:
                                safe_list[j] = 1
                            break
                    continue
                if noun in ha_words:
                    for j in range(len(ha_words)):
                        if noun == ha_words[j]:
                            if j < (len(ha_list) - ha_len):
                                ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                            else:
                                ha_list[j] = 1
                            break
                    continue

                found_ha = False
                for j, check_word in enumerate(ha_words):
                    if check_synonyms_word(noun, check_word):
                        if j < (len(ha_list) - ha_len):
                            ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                        else:
                            ha_list[j] = 1
                        found_ha = True
                        break
                if found_ha:
                    continue

                flag = False
                for j, check_word in enumerate(safe_words):
                    if check_synonyms_word(noun, check_word):
                        flag = True
                        if j < (len(safe_list) - safe_len):
                            safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                        else:
                            safe_list[j] = 1
                        break
                if not flag:
                    safe_flag_list[idx] = 1

            metrics["chair_score"] += sum(safe_flag_list)
            metrics["chair_num"] += len(safe_flag_list)
            metrics["safe_cover_score"] += sum(safe_list[-safe_len:])
            metrics["safe_cover_num"] += len(safe_list[-safe_len:])
            metrics["hallu_cover_score"] += sum(ha_list[-ha_len:])
            metrics["hallu_cover_num"] += len(ha_list[-ha_len:])
            if sum(safe_flag_list) == 0:
                metrics["non_hallu_score"] += 1
            metrics["non_hallu_num"] += 1

        else:
            metrics["qa_correct_num"] += 1
            gt_type = gt["type"]
            type_prefix_map = {
                "discriminative-attribute-state": "as",
                "discriminative-attribute-number": "an",
                "discriminative-attribute-action": "aa",
                "discriminative-hallucination": "ha",
            }
            prefix = type_prefix_map.get(gt_type, "asso")
            metrics[f"{prefix}_qa_correct_num"] += 1

            truth = gt["truth"]
            response = item["response"]

            if truth == "yes":
                if response == "Yes":
                    metrics["qa_correct_score"] += 1
                    metrics[f"{prefix}_qa_correct_score"] += 1
            else:
                metrics["qa_no_num"] += 1
                metrics[f"{prefix}_qa_no_num"] += 1
                if response == "No":
                    metrics["qa_correct_score"] += 1
                    metrics["qa_no_score"] += 1
                    metrics[f"{prefix}_qa_correct_score"] += 1
                    metrics[f"{prefix}_qa_no_score"] += 1

            if response == "No":
                metrics["qa_ans_no_num"] += 1
                metrics[f"{prefix}_qa_ans_no_num"] += 1
                if truth == "no":
                    metrics["qa_ans_no_score"] += 1
                    metrics[f"{prefix}_qa_ans_no_score"] += 1

    return metrics


def merge_metrics(partial_list):
    """Merge partial metrics from all workers."""
    merged = {}
    for partial in partial_list:
        for key, value in partial.items():
            merged[key] = merged.get(key, 0) + value
    return merged


def print_results(metrics, eval_type):
    """Print final results matching inference.py output format."""
    dimension = {"g": False, "de": False, "da": False, "dr": False}
    if eval_type == "a":
        for key in dimension:
            dimension[key] = True
    elif eval_type == "g":
        dimension["g"] = True
    elif eval_type == "d":
        dimension["de"] = True
        dimension["da"] = True
        dimension["dr"] = True
    else:
        dimension[eval_type] = True

    if dimension["g"]:
        chair = round(metrics["chair_score"] / metrics["chair_num"] * 100, 1)
        cover = round(metrics["safe_cover_score"] / metrics["safe_cover_num"] * 100, 1)
        ha = round(metrics["hallu_cover_score"] / metrics["hallu_cover_num"] * 100, 1)
        ha_p = round(100 - metrics["non_hallu_score"] / metrics["non_hallu_num"] * 100, 1)
        print("Generative Task:")
        print(f"CHAIR:\t\t {chair}")
        print(f"Cover:\t\t {cover}")
        print(f"Hal:\t\t {ha_p}")
        print(f"Cog:\t\t {ha}\n")

    if dimension["de"] and dimension["da"] and dimension["dr"]:
        accuracy = round(metrics["qa_correct_score"] / metrics["qa_correct_num"] * 100, 1)
        precision = round(metrics["qa_ans_no_score"] / metrics["qa_ans_no_num"] * 100, 1)
        recall = round(metrics["qa_no_score"] / metrics["qa_no_num"] * 100, 1)
        f1 = round(2 * (precision / 100) * (recall / 100) / ((precision / 100) + (recall / 100) + 0.0001) * 100, 1)
        print("Descriminative Task:")
        print(f"Accuracy:\t {accuracy}")
        print(f"Precision:\t {precision}")
        print(f"Recall:\t\t {recall}")
        print(f"F1:\t\t {f1}\n")

    if dimension["de"]:
        h_acc = round(metrics["ha_qa_correct_score"] / metrics["ha_qa_correct_num"] * 100, 1)
        h_prec = round(metrics["ha_qa_ans_no_score"] / metrics["ha_qa_ans_no_num"] * 100, 1)
        h_rec = round(metrics["ha_qa_no_score"] / metrics["ha_qa_no_num"] * 100, 1)
        h_f1 = round(2 * (h_prec / 100) * (h_rec / 100) / ((h_prec / 100) + (h_rec / 100) + 0.001) * 100, 1)
        print("Exsitence:")
        print(f"Accuracy:\t {h_acc}")
        print(f"Precision:\t {h_prec}")
        print(f"Recall:\t\t {h_rec}")
        print(f"F1:\t\t {h_f1}\n")

    if dimension["da"]:
        attr_acc = round((metrics["as_qa_correct_score"] + metrics["an_qa_correct_score"] + metrics["aa_qa_correct_score"]) / (metrics["as_qa_correct_num"] + metrics["an_qa_correct_num"] + metrics["aa_qa_correct_num"]) * 100, 1)
        attr_prec = round((metrics["as_qa_ans_no_score"] + metrics["an_qa_ans_no_score"] + metrics["aa_qa_ans_no_score"]) / (metrics["as_qa_ans_no_num"] + metrics["an_qa_ans_no_num"] + metrics["aa_qa_ans_no_num"]) * 100, 1)
        attr_rec = round((metrics["as_qa_no_score"] + metrics["an_qa_no_score"] + metrics["aa_qa_no_score"]) / (metrics["as_qa_no_num"] + metrics["an_qa_no_num"] + metrics["aa_qa_no_num"]) * 100, 1)
        attr_f1 = round(2 * (attr_prec / 100) * (attr_rec / 100) / ((attr_prec / 100) + (attr_rec / 100) + 0.0001) * 100, 1)
        print("Attribute:")
        print(f"Accuracy:\t {attr_acc}")
        print(f"Precision:\t {attr_prec}")
        print(f"Recall:\t\t {attr_rec}")
        print(f"F1:\t\t {attr_f1}\n")

        for label, pfx in [("State", "as"), ("Number", "an"), ("Action", "aa")]:
            acc = round(metrics[f"{pfx}_qa_correct_score"] / metrics[f"{pfx}_qa_correct_num"] * 100, 1)
            prec = round(metrics[f"{pfx}_qa_ans_no_score"] / metrics[f"{pfx}_qa_ans_no_num"] * 100, 1)
            rec = round(metrics[f"{pfx}_qa_no_score"] / metrics[f"{pfx}_qa_no_num"] * 100, 1)
            f1 = round(2 * (prec / 100) * (rec / 100) / ((prec / 100) + (rec / 100) + 0.0001) * 100, 1)
            print(f"{label}:")
            print(f"Accuracy:\t {acc}")
            print(f"Precision:\t {prec}")
            print(f"Recall:\t\t {rec}")
            print(f"F1:\t\t {f1}\n")

    if dimension["dr"]:
        r_acc = round(metrics["asso_qa_correct_score"] / metrics["asso_qa_correct_num"] * 100, 1)
        r_prec = round(metrics["asso_qa_ans_no_score"] / metrics["asso_qa_ans_no_num"] * 100, 1)
        r_rec = round(metrics["asso_qa_no_score"] / metrics["asso_qa_no_num"] * 100, 1)
        r_f1 = round(2 * (r_prec / 100) * (r_rec / 100) / ((r_prec / 100) + (r_rec / 100) + 0.0001) * 100, 1)
        print("Relation:")
        print(f"Accuracy:\t {r_acc}")
        print(f"Precision:\t {r_prec}")
        print(f"Recall:\t\t {r_rec}")
        print(f"F1:\t\t {r_f1}")


def main():
    parser = argparse.ArgumentParser(description="Parallel AMBER evaluation")
    parser.add_argument("--inference_data", required=True, help="Path to response JSON")
    parser.add_argument("--evaluation_type", default="a", choices=["a", "g", "d", "de", "da", "dr"])
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--similarity_score", type=float, default=0.8)
    parser.add_argument("--amber_root", type=str, default=str(AMBER_ROOT))
    args = parser.parse_args()

    amber_root = Path(args.amber_root)
    inference_data = json.load(open(args.inference_data, "r", encoding="utf-8"))

    print(f"Loading {len(inference_data)} samples, using {args.workers} workers...")

    shared = load_shared_data(
        amber_root,
        amber_root / "data" / "relation.json",
        amber_root / "data" / "safe_words.txt",
        amber_root / "data" / "annotations.json",
        amber_root / "data" / "metrics.txt",
    )

    # Split data into chunks
    chunk_size = max(1, len(inference_data) // args.workers)
    chunks = []
    for i in range(0, len(inference_data), chunk_size):
        chunks.append(inference_data[i : i + chunk_size])

    print(f"Split into {len(chunks)} chunks, starting parallel evaluation...")

    task_args = [(chunk, shared, args.similarity_score) for chunk in chunks]

    with mp.Pool(processes=args.workers) as pool:
        partial_results = list(pool.imap_unordered(process_chunk, task_args))

    print("Merging results...")
    merged = merge_metrics(partial_results)
    print_results(merged, args.evaluation_type)


if __name__ == "__main__":
    main()
