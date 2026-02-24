import os
import sys
import random
from typing import Dict, Any

_this_dir = os.path.dirname(os.path.abspath(__file__))
_utils_dir = os.path.normpath(os.path.join(_this_dir, "..", "utils"))
if _utils_dir not in sys.path:
    sys.path.insert(0, _utils_dir)

from math_checker import check_answer, extract_boxed_answer


def compute_score(
    data_source: str = None,
    solution_str: str = None,
    ground_truth: str = None,
    extra_info: Dict[str, Any] = None,
    **kwargs
) -> float:
    if ground_truth is None or solution_str is None:
        print(f"[compute_score] WARNING: ground_truth or solution_str is None! Returning 0.0")
        return 0.0

    is_correct = check_answer(solution_str, ground_truth)
    boxed_answer = extract_boxed_answer(solution_str)

    if random.random() < 0.02:
        print(f"[compute_score] ground_truth={ground_truth[:80]}... "
              f"boxed={boxed_answer[:80] if boxed_answer else None}... "
              f"correct={is_correct}")

    return 1.0 if is_correct else 0.0
