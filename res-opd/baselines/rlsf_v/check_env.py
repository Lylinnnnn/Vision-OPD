#!/usr/bin/env python3
"""Check RLSF-V phase dependencies before starting workers or training."""

import argparse
import importlib
import shutil
import sys


MODULES = {
    "prepare": ("pandas", "pyarrow"),
    "generate": ("numpy", "shortuuid", "torch", "PIL", "tqdm", "transformers"),
    "convert": ("spacy", "torch", "tqdm", "transformers"),
    "train": ("yaml", "llamafactory"),
    "export": ("yaml", "llamafactory"),
}
MODULES["pipeline"] = tuple(dict.fromkeys(
    MODULES["prepare"] + MODULES["generate"] + MODULES["convert"]
))


def check(phase, model):
    errors = []
    loaded = {}
    for name in MODULES[phase]:
        try:
            loaded[name] = importlib.import_module(name)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    if phase in {"generate", "pipeline"} and "transformers" in loaded:
        name = "AutoModelForImageTextToText" if model == "internvl2b" else "Qwen3VLForConditionalGeneration"
        try:
            getattr(loaded["transformers"], name)
        except Exception as exc:
            errors.append(f"transformers {name}: {type(exc).__name__}: {exc}")
    if phase in {"generate", "pipeline"} and "torch" in loaded:
        torch = loaded["torch"]
        if not torch.cuda.is_available():
            errors.append("torch.cuda.is_available() is false")
        else:
            print(f"CUDA devices visible: {torch.cuda.device_count()}")
    if phase in {"convert", "pipeline"} and "spacy" in loaded:
        try:
            loaded["spacy"].load("en_core_web_sm")
        except Exception as exc:
            errors.append(f"spaCy en_core_web_sm: {type(exc).__name__}: {exc}")
    if phase in {"train", "export"} and shutil.which("llamafactory-cli") is None:
        errors.append("llamafactory-cli is not on PATH")

    print(f"Python: {sys.executable}")
    for name in ("torch", "transformers", "spacy"):
        if name in loaded:
            print(f"{name}: {getattr(loaded[name], '__version__', 'unknown')}")
    if phase in {"generate", "convert", "pipeline"} and "transformers" in loaded:
        if loaded["transformers"].__version__ != "4.57.3":
            print("WARNING: upstream RLSF-V documents transformers==4.57.3 for HF generation/conversion")
    if errors:
        raise SystemExit("RLSF-V environment FAIL:\n- " + "\n- ".join(errors))
    print(f"RLSF-V environment PASS: phase={phase} model={model}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=MODULES)
    parser.add_argument("--model", choices=("qwen2b", "qwen4b", "qwen8b", "internvl2b"), required=True)
    args = parser.parse_args()
    check(args.phase, args.model)


if __name__ == "__main__":
    main()
