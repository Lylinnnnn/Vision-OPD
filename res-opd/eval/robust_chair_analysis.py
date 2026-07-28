# -*- coding: utf-8 -*-
"""
Robust CHAIR supplementary analysis using official Rohrbach CHAIR synonyms.

Does NOT modify chair_fullval_recompute.py or its outputs.

Uses the official synonyms.txt from LisaAnne/Hallucination (Rohrbach et al. 2018)
plus official double-word and qualifier rules from chair.py.

Produces:
  Table A: per-config metrics under full-synonym (original script) and official CHAIR modes.
  Table B: paired bootstrap confidence intervals for key comparisons.

Usage:
    cd /home/liuyanlin.lyl/notebook/lyl/opd/ms-swift/project/pilot0/runs
    /opt/conda/envs/python3.10/bin/python robust_chair_analysis.py
"""

import json
import os
import re
import sys

import numpy as np

# ============================================================
# Official Rohrbach CHAIR synonyms (from data/synonyms.txt)
# Each line: canonical_name, syn1, syn2, ...
# First entry on each line is the COCO category name.
# ============================================================
OFFICIAL_SYNONYMS_RAW = """
person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, person, woman, guy, foreigner, child, gentleman, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager
bicycle, bike, bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter, motor bike, motor cycle, motorbike, scooter, moped
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, jetliner, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, lamb, goat, ewe
cow, cattle, oxen, ox, calf, cattle, holstein, heifer, buffalo, bull, zebu, bison
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bow tie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake, cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield
potted plant, houseplant
bed
dining table, table, desk
toilet, urinal, commode, toilet, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iPhone
microwave
oven, stovetop, stove, stove top oven
toaster
sink
refrigerator, fridge, fridge, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush
""".strip()


def parse_official_synonyms():
    """Parse official synonyms.txt into (mscoco_objects set, inverse_synonym_dict).

    Returns:
        mscoco_objects: set of ALL known terms (categories + all synonyms)
        inverse_synonym_dict: maps any term -> canonical category name
    """
    mscoco_objects = set()
    inverse_synonym_dict = {}
    for line in OFFICIAL_SYNONYMS_RAW.strip().split("\n"):
        parts = [s.strip() for s in line.split(",") if s.strip()]
        canonical = parts[0]
        for syn in parts:
            mscoco_objects.add(syn)
            inverse_synonym_dict[syn] = canonical
    return mscoco_objects, inverse_synonym_dict


# Official CHAIR double-word rules
COCO_DOUBLE_WORDS = [
    "motor bike", "motor cycle", "air plane", "traffic light", "street light",
    "traffic signal", "stop light", "fire hydrant", "stop sign", "parking meter",
    "suit case", "sports ball", "baseball bat", "baseball glove", "tennis racket",
    "wine glass", "hot dog", "cell phone", "mobile phone", "teddy bear",
    "hair drier", "potted plant", "bow tie", "laptop computer", "stove top oven",
    "hot dog", "teddy bear", "home plate", "train track",
]

ANIMAL_WORDS = [
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "animal", "cub",
]
VEHICLE_WORDS = ["jet", "train"]


def build_double_word_dict():
    """Build the official CHAIR double-word dictionary."""
    double_word_dict = {}
    for dw in COCO_DOUBLE_WORDS:
        double_word_dict[dw] = dw
    for animal in ANIMAL_WORDS:
        double_word_dict["baby %s" % animal] = animal
        double_word_dict["adult %s" % animal] = animal
    for vehicle in VEHICLE_WORDS:
        double_word_dict["passenger %s" % vehicle] = vehicle
    double_word_dict["bow tie"] = "tie"
    double_word_dict["toilet seat"] = "toilet"
    double_word_dict["wine glas"] = "wine glass"
    return double_word_dict


def try_singularize(word):
    """Simple rule-based singularization (avoids pattern.en dependency).

    Handles common English plural forms. Falls back to the original word.
    """
    if len(word) <= 2:
        return word
    # Irregular
    irregulars = {
        "people": "person", "men": "man", "women": "woman", "children": "child",
        "mice": "mouse", "geese": "goose", "teeth": "tooth", "feet": "foot",
        "oxen": "ox", "dice": "die", "knives": "knife", "lives": "life",
        "wives": "wife", "shelves": "shelf", "leaves": "leaf",
        "calves": "calf", "halves": "half", "wolves": "wolf",
    }
    if word in irregulars:
        return irregulars[word]
    # -ies -> -y (but not if preceded by a vowel: "keys" -> "key")
    if word.endswith("ies") and len(word) > 3 and word[-4] not in "aeiou":
        return word[:-3] + "y"
    # -ves -> -f/fe
    if word.endswith("ves"):
        return word[:-3] + "f"
    # -ses, -xes, -zes, -ches, -shes -> remove "es"
    if word.endswith("ses") or word.endswith("xes") or word.endswith("zes"):
        return word[:-2]
    if word.endswith("ches") or word.endswith("shes"):
        return word[:-2]
    # -s (but not -ss)
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def caption_to_words(caption, mscoco_objects, inverse_synonym_dict, double_word_dict):
    """Official CHAIR caption-to-words extraction.

    Input: caption string
    Output: (words, node_words) where:
        words = list of matched COCO synonym terms found in caption
        node_words = list of canonical COCO category names
    """
    # Standard preprocessing: tokenize and singularize
    # ``str.split`` leaves punctuation attached (for example ``"dog,"``),
    # which silently drops valid object mentions.  The official evaluator uses
    # nltk.word_tokenize; this dependency-free tokenization has the same
    # relevant behavior for COCO object words and compounds.
    raw_words = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", caption.lower())
    words = [try_singularize(w) for w in raw_words]

    # Replace double words
    i = 0
    processed = []
    while i < len(words):
        double_word = " ".join(words[i:i + 2])
        if double_word in double_word_dict:
            processed.append(double_word_dict[double_word])
            i += 2
        else:
            processed.append(words[i])
            i += 1
    words = processed

    # "toilet seat is not chair" rule
    if "toilet" in words and "seat" in words:
        words = [w for w in words if w != "seat"]

    # Filter to only COCO-related words
    matched_words = [w for w in words if w in mscoco_objects]
    node_words = [inverse_synonym_dict[w] for w in matched_words]

    return matched_words, node_words


def repetition_rate(text):
    """Compute 4-gram repetition rate."""
    words = text.split()
    if len(words) < 20:
        return 0.0
    ngrams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def compute_per_sample(records, mscoco_objects, inverse_synonym_dict,
                       double_word_dict, hit_max_threshold=750):
    """Compute per-caption CHAIR values and custom unique-object metrics.

    Official CHAIR uses every generated object mention for CHAIRi and marks a
    whole generated caption positive for CHAIRs if it contains at least one
    hallucinated object.  ObjPrec/ObjRecall/ObjF1 remain the repository's
    category-level metrics and therefore use unique canonical categories.

    Ground-truth objects are the union of instance categories and object words
    found in the reference captions, matching the official CHAIR protocol.

    Returns dict keyed by image_id.
    """
    per_sample = {}
    for record in records:
        image_id = record.get("image_id", record.get("sample_id"))
        gt_categories = {
            inverse_synonym_dict.get(str(obj).lower(), str(obj).lower())
            for obj in record.get("gt_objects", [])
        }
        for gt_caption in record.get("gt_captions", []) or []:
            _, gt_node_words = caption_to_words(
                gt_caption, mscoco_objects, inverse_synonym_dict, double_word_dict
            )
            gt_categories.update(gt_node_words)
        generated_text = record["generated_text"]

        # Extract objects using official method
        matched_words, node_words = caption_to_words(
            generated_text, mscoco_objects, inverse_synonym_dict, double_word_dict
        )

        # Official CHAIRi counts all mentions, including repeated categories.
        mention_hallucinated = sum(obj not in gt_categories for obj in node_words)
        mention_correct = len(node_words) - mention_hallucinated
        has_hallucination = int(mention_hallucinated > 0)

        # The repository's object precision/recall/F1 use unique categories.
        mentioned_set = set(node_words)
        hallucinated = 0
        correct = 0
        for obj in mentioned_set:
            if obj in gt_categories:
                correct += 1
            else:
                hallucinated += 1

        # Retain the old clause/sentence statistic only as an explicitly named
        # diagnostic.  It is not official CHAIRs.
        sentences = re.split(r"[.!?\n]+", generated_text)
        n_sent = 0
        n_halluc_sent = 0
        for sent in sentences:
            if len(sent.strip()) < 5:
                continue
            n_sent += 1
            _, sent_node_words = caption_to_words(
                sent, mscoco_objects, inverse_synonym_dict, double_word_dict
            )
            sent_mentioned = set(sent_node_words)
            if any(obj not in gt_categories for obj in sent_mentioned):
                n_halluc_sent += 1

        token_len = record.get("generated_ids_len", 0)

        per_sample[image_id] = {
            "chair_object_mentions": len(node_words),
            "chair_hallucinated_mentions": mention_hallucinated,
            "chair_correct_mentions": mention_correct,
            "chair_has_hallucination": has_hallucination,
            "mentioned": len(mentioned_set),
            "hallucinated": hallucinated,
            "correct": correct,
            "gt_count": len(gt_categories),
            "n_sent": n_sent,
            "n_halluc_sent": n_halluc_sent,
            "rep_rate": repetition_rate(generated_text),
            "token_len": token_len,
        }
    return per_sample


def aggregate_from_arrays(sample_dicts, indices):
    """Compute aggregate CHAIR metrics from per-sample dicts at given indices.

    Uses ratio-of-sums (not mean-of-ratios).
    """
    total_mentioned = 0
    total_hallucinated = 0
    total_correct = 0
    total_gt = 0
    total_sent = 0
    total_halluc_sent = 0
    total_object_mentions = 0
    total_hallucinated_mentions = 0
    total_correct_mentions = 0
    total_captions = 0
    hallucinated_captions = 0
    rep_rates = []

    for idx in indices:
        sample = sample_dicts[idx]
        total_mentioned += sample["mentioned"]
        total_hallucinated += sample["hallucinated"]
        total_correct += sample["correct"]
        total_gt += sample["gt_count"]
        total_sent += sample["n_sent"]
        total_halluc_sent += sample["n_halluc_sent"]
        total_object_mentions += sample["chair_object_mentions"]
        total_hallucinated_mentions += sample["chair_hallucinated_mentions"]
        total_correct_mentions += sample["chair_correct_mentions"]
        total_captions += 1
        hallucinated_captions += sample["chair_has_hallucination"]
        rep_rates.append(sample["rep_rate"])

    chair_i = total_hallucinated_mentions / max(total_object_mentions, 1)
    chair_s = hallucinated_captions / max(total_captions, 1)
    unique_category_chair_i_legacy = total_hallucinated / max(total_mentioned, 1)
    sentence_chair_s_proxy = total_halluc_sent / max(total_sent, 1)
    obj_prec = total_correct / max(total_mentioned, 1)
    obj_recall = total_correct / max(total_gt, 1)
    obj_f1 = (2 * obj_prec * obj_recall / (obj_prec + obj_recall)
              if (obj_prec + obj_recall) > 0 else 0.0)
    mean_rep_rate = float(np.mean(rep_rates)) if rep_rates else 0.0

    return {
        "CHAIRi": chair_i,
        "CHAIRs": chair_s,
        "ObjPrec": obj_prec,
        "ObjRecall": obj_recall,
        "ObjF1": obj_f1,
        "RepRate": mean_rep_rate,
        "CHAIRi_unique_category_legacy": unique_category_chair_i_legacy,
        "CHAIRs_sentence_proxy_legacy": sentence_chair_s_proxy,
        "metric_schema": "official_chair_caption_and_mention_v1",
        "total_captions": total_captions,
        "hallucinated_captions": hallucinated_captions,
        "total_object_mentions": total_object_mentions,
        "total_hallucinated_mentions": total_hallucinated_mentions,
        "total_correct_mentions": total_correct_mentions,
        # Backward-compatible unique-category totals used by ObjPrec/Recall/F1.
        "total_mentioned": total_mentioned,
        "total_hallucinated": total_hallucinated,
        "total_correct": total_correct,
    }


def paired_bootstrap_ci(samples_a, samples_b, n_common, n_bootstrap=10000,
                         seed=42, alpha=0.05):
    """Paired bootstrap CI for metric differences (A minus B).

    samples_a, samples_b: list of per-sample dicts, aligned by common image_ids.
    Returns dict with delta point estimate, CI, p-value for each metric.
    """
    rng = np.random.RandomState(seed)
    all_indices = np.arange(n_common)

    # Point estimates on full data
    agg_a = aggregate_from_arrays(samples_a, all_indices)
    agg_b = aggregate_from_arrays(samples_b, all_indices)

    metrics_to_test = ["CHAIRi", "CHAIRs", "ObjF1", "RepRate"]
    deltas_point = {m: agg_a[m] - agg_b[m] for m in metrics_to_test}

    # Bootstrap
    boot_deltas = {m: [] for m in metrics_to_test}
    for _ in range(n_bootstrap):
        boot_idx = rng.choice(n_common, size=n_common, replace=True)
        boot_agg_a = aggregate_from_arrays(samples_a, boot_idx)
        boot_agg_b = aggregate_from_arrays(samples_b, boot_idx)
        for m in metrics_to_test:
            boot_deltas[m].append(boot_agg_a[m] - boot_agg_b[m])

    results = {}
    for m in metrics_to_test:
        deltas_arr = np.array(boot_deltas[m])
        ci_lo = float(np.percentile(deltas_arr, 100 * alpha / 2))
        ci_hi = float(np.percentile(deltas_arr, 100 * (1 - alpha / 2)))
        if deltas_point[m] >= 0:
            p_one_sided = float(np.mean(deltas_arr < 0))
        else:
            p_one_sided = float(np.mean(deltas_arr > 0))
        p_two_sided = min(2 * p_one_sided, 1.0)

        results[m] = {
            "delta": deltas_point[m],
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p": p_two_sided,
            "sig": "***" if p_two_sided < 0.001 else
                   "**" if p_two_sided < 0.01 else
                   "*" if p_two_sided < 0.05 else "ns",
        }
    return results


def load_records(path):
    """Load eval_results.jsonl into list of dicts."""
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def build_gt_objects_from_coco(instances_path, captions_path,
                               mscoco_objects, inverse_synonym_dict,
                               double_word_dict):
    """Build per-image GT object sets from COCO instance + caption annotations.

    This replicates the official CHAIR GT construction:
      1) Instance segmentation categories -> canonical names
      2) GT captions -> extract mentioned COCO objects via caption_to_words

    Returns:
        imid_to_objects: dict mapping image_id -> set of canonical object names
    """
    imid_to_objects = {}

    with open(instances_path) as fh:
        instances = json.load(fh)
    id_to_name = {cat["id"]: cat["name"] for cat in instances["categories"]}

    for annotation in instances["annotations"]:
        imid = annotation["image_id"]
        category_name = id_to_name[annotation["category_id"]]
        node_word = inverse_synonym_dict.get(category_name, category_name)
        if imid not in imid_to_objects:
            imid_to_objects[imid] = set()
        imid_to_objects[imid].add(node_word)

    with open(captions_path) as fh:
        captions_data = json.load(fh)

    for annotation in captions_data["annotations"]:
        imid = annotation["image_id"]
        if imid not in imid_to_objects:
            imid_to_objects[imid] = set()
        _, node_words = caption_to_words(
            annotation["caption"], mscoco_objects, inverse_synonym_dict,
            double_word_dict)
        imid_to_objects[imid].update(node_words)

    return imid_to_objects


class CHAIREvaluator:
    """Convenience wrapper exposing the same API as the legacy evaluator.

    Used by discrimination_gap.py and other scripts that need:
      - caption_to_words(text) -> (words, node_words, idxs)
      - get_gt_objects(image_id) -> set of canonical names
      - inverse_synonym_dict  (for synonym lookup)
    """

    def __init__(self, synonyms_path=None, instances_path=None,
                 captions_path=None):
        self.mscoco_objects, self.inverse_synonym_dict = parse_official_synonyms()
        self.double_word_dict = build_double_word_dict()

        if instances_path and captions_path:
            self.imid_to_objects = build_gt_objects_from_coco(
                instances_path, captions_path,
                self.mscoco_objects, self.inverse_synonym_dict,
                self.double_word_dict)
        else:
            self.imid_to_objects = {}

    def caption_to_words(self, text):
        """Extract COCO objects from caption text.

        Returns (words, node_words, idxs) for backward compat.
        idxs is a placeholder (list of sequential indices).
        """
        words, node_words = caption_to_words(
            text, self.mscoco_objects, self.inverse_synonym_dict,
            self.double_word_dict)
        idxs = list(range(len(words)))
        return words, node_words, idxs

    def get_gt_objects(self, image_id):
        """Get GT objects for an image (canonical forms)."""
        return self.imid_to_objects.get(image_id, set())


# ============================================================
# Main
# ============================================================

def main():
    sweep_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "e3_sweep",
    )

    config_paths = {
        "D":          os.path.join(sweep_base, "ema0.0_p90/lr1e-06/coco_fullval_338/group_D/eval_results.jsonl"),
        "C(JSD)":     os.path.join(sweep_base, "ema0.0_p90/lr1e-06/coco_fullval_338/group_C/eval_results.jsonl"),
        "C(RKL)":     os.path.join(sweep_base, "ema0.0_p90/lr1e-06_kdrkl/coco_fullval_338/group_C/eval_results.jsonl"),
        "F(p85)":     os.path.join(sweep_base, "ema0.0_p85/lr1e-06/coco_fullval_338/group_F/eval_results.jsonl"),
        "F(p90)":     os.path.join(sweep_base, "ema0.0_p90/lr1e-06/coco_fullval_338/group_F/eval_results.jsonl"),
        "F(p90)RKL":  os.path.join(sweep_base, "ema0.0_p90/lr1e-06_kdrkl/coco_fullval_338/group_F/eval_results.jsonl"),
        "F(p95)":     os.path.join(sweep_base, "ema0.0_p95/lr1e-06/coco_fullval_338/group_F/eval_results.jsonl"),
    }
    display_order = ["D", "C(JSD)", "C(RKL)", "F(p85)", "F(p90)", "F(p90)RKL", "F(p95)"]

    # Build official synonym lookup
    mscoco_objects, inverse_synonym_dict = parse_official_synonyms()
    double_word_dict = build_double_word_dict()

    print("=" * 110)
    print("Robust CHAIR Analysis — Official Rohrbach CHAIR synonyms.txt")
    print("=" * 110)
    print()
    print("Synonym source: LisaAnne/Hallucination (Rohrbach et al. EMNLP 2018)")
    print("Total known terms: {} (across {} COCO categories)".format(
        len(mscoco_objects), len(set(inverse_synonym_dict.values()))))
    print("Double-word rules: {} entries (incl. animal/vehicle qualifiers)".format(
        len(double_word_dict)))
    print()

    # Load records and compute per-sample arrays
    all_records = {}
    per_sample_official = {}

    for label in display_order:
        path = config_paths[label]
        if not os.path.exists(path):
            print("WARNING: {} not found, skipping: {}".format(label, path))
            continue
        records = load_records(path)
        all_records[label] = records
        per_sample_official[label] = compute_per_sample(
            records, mscoco_objects, inverse_synonym_dict, double_word_dict
        )
        print("Loaded {}: {} samples".format(label, len(records)))

    print()

    # ================================================================
    # TABLE A: per-config metrics under official CHAIR synonyms
    # ================================================================
    print("=" * 120)
    print("TABLE A — Official CHAIR synonym metrics")
    print("=" * 120)
    header = "{:<12} {:>5}  {:>8} {:>8}  {:>8} {:>8}  {:>8} {:>8}  {:>7}  {:>6}".format(
        "Config", "N",
        "CHAIRi", "CHAIRs",
        "ObjPrec", "ObjRec",
        "ObjF1", "AvgObj",
        "RepRate", "HitMax",
    )
    print(header)
    print("-" * len(header))

    table_a_data = {}
    for label in display_order:
        if label not in per_sample_official:
            continue
        ps = per_sample_official[label]
        ids = sorted(ps.keys())
        idx_all = list(range(len(ids)))
        ps_list = [ps[i] for i in ids]

        agg = aggregate_from_arrays(ps_list, idx_all)
        hit_max = sum(1 for i in ids if ps[i]["token_len"] >= 750)
        avg_mentioned = agg["total_mentioned"] / len(ids) if ids else 0

        table_a_data[label] = {
            "metrics": agg, "n_samples": len(ids), "hit_max": hit_max,
        }

        print("{:<12} {:>5}  {:>8.4f} {:>8.4f}  {:>8.4f} {:>8.4f}  {:>8.4f} {:>8.1f}  {:>7.3f}  {:>3}/{}".format(
            label, len(ids),
            agg["CHAIRi"], agg["CHAIRs"],
            agg["ObjPrec"], agg["ObjRecall"],
            agg["ObjF1"], avg_mentioned,
            agg["RepRate"], hit_max, len(ids),
        ))

    print()
    print("Raw counts:")
    for label in display_order:
        if label not in table_a_data:
            continue
        agg = table_a_data[label]["metrics"]
        print("  {:<12} mentioned={} halluc={} correct={}".format(
            label, agg["total_mentioned"], agg["total_hallucinated"], agg["total_correct"]))

    # ================================================================
    # TABLE B: paired bootstrap CI
    # ================================================================
    comparisons = [
        ("C(JSD)", "D", "KD vs baseline (sanity)"),
        ("F(p85)", "D", "KD vs baseline (sanity)"),
        ("F(p90)", "D", "KD vs baseline (sanity)"),
        ("F(p95)", "D", "KD vs baseline (sanity)"),
        ("F(p95)", "C(JSD)", "selective vs uniform (core)"),
        ("F(p85)", "C(JSD)", "selective vs uniform"),
        ("F(p90)", "C(JSD)", "selective vs uniform"),
        ("C(JSD)", "C(RKL)", "JSD vs RKL (uniform)"),
        ("F(p90)", "F(p90)RKL", "JSD vs RKL (selective)"),
        ("F(p85)", "F(p90)", "percentile comparison"),
        ("F(p85)", "F(p95)", "percentile comparison"),
        ("F(p90)", "F(p95)", "percentile comparison"),
    ]

    print()
    print("=" * 140)
    print("TABLE B — Paired Bootstrap CI (official CHAIR synonyms, N=10000, seed=42)")
    print("=" * 140)
    print()
    print("Sign convention: delta = A - B. Negative delta_CHAIRi means A has LESS hallucination.")
    print()

    table_b_header = "{:<26} {:>8}  {:>8}  {:>22}  {:>6}  {:>4}  {}".format(
        "Pair (A vs B)", "Metric", "Delta", "95% CI", "p", "Sig", "Category",
    )
    print(table_b_header)
    print("-" * len(table_b_header))

    table_b_data = []

    for config_a, config_b, category in comparisons:
        if config_a not in per_sample_official or config_b not in per_sample_official:
            print("{:<26} SKIPPED (missing data)".format("{} vs {}".format(config_a, config_b)))
            continue

        ps_a = per_sample_official[config_a]
        ps_b = per_sample_official[config_b]

        common_ids = sorted(set(ps_a.keys()) & set(ps_b.keys()))
        only_a = set(ps_a.keys()) - set(ps_b.keys())
        only_b = set(ps_b.keys()) - set(ps_a.keys())
        if only_a or only_b:
            print("  WARNING: {} vs {} — {} common, {} only-A, {} only-B".format(
                config_a, config_b, len(common_ids), len(only_a), len(only_b)))

        if not common_ids:
            print("{:<26} SKIPPED (no common samples)".format("{} vs {}".format(config_a, config_b)))
            continue

        aligned_a = [ps_a[img_id] for img_id in common_ids]
        aligned_b = [ps_b[img_id] for img_id in common_ids]

        ci_results = paired_bootstrap_ci(aligned_a, aligned_b, len(common_ids))

        pair_label = "{} vs {}".format(config_a, config_b)
        for metric_name in ["CHAIRi", "CHAIRs", "ObjF1", "RepRate"]:
            result = ci_results[metric_name]
            ci_str = "[{:+.4f}, {:+.4f}]".format(result["ci_lo"], result["ci_hi"])
            print("{:<26} {:>8}  {:>+8.4f}  {:>22}  {:>6.4f}  {:>4}  {}".format(
                pair_label if metric_name == "CHAIRi" else "",
                metric_name, result["delta"], ci_str,
                result["p"], result["sig"],
                category if metric_name == "CHAIRi" else "",
            ))
            table_b_data.append({
                "pair": pair_label,
                "metric": metric_name,
                "delta": result["delta"],
                "ci_lo": result["ci_lo"],
                "ci_hi": result["ci_hi"],
                "p": result["p"],
                "sig": result["sig"],
                "category": category,
            })
        print()

    # ================================================================
    # Summary
    # ================================================================
    print("=" * 100)
    print("INTERPRETATION GUIDE")
    print("=" * 100)
    print("- Sanity (KD vs D): expect significant. If ns, check extraction pipeline.")
    print("- Selective vs uniform (F vs C): core claim under test.")
    print("- JSD vs RKL: if ns, divergence form is interchangeable.")
    print("- Percentile pairs: if all ns, percentile choice is noise.")
    print()

    # Save JSON
    output_path = os.path.join(sweep_base, "robust_chair_official.json")
    save_data = {
        "synonym_source": "LisaAnne/Hallucination synonyms.txt (Rohrbach et al. 2018)",
        "table_a": {},
        "table_b": table_b_data,
        "bootstrap_config": {"n_bootstrap": 10000, "seed": 42, "alpha": 0.05},
    }
    for label, data in table_a_data.items():
        save_data["table_a"][label] = {
            "n_samples": data["n_samples"],
            "hit_max": data["hit_max"],
            **data["metrics"],
        }

    with open(output_path, "w") as fh:
        json.dump(save_data, fh, indent=2, default=float)
    print("Results saved to: {}".format(output_path))


if __name__ == "__main__":
    main()
