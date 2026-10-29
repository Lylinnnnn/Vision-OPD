import importlib.util
import sys
from pathlib import Path


def load_chair_module():
    module_path = Path(__file__).resolve().parents[1] / "eval" / "robust_chair_analysis.py"
    eval_dir = str(module_path.parent)
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    spec = importlib.util.spec_from_file_location("res_opd_official_chair", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
