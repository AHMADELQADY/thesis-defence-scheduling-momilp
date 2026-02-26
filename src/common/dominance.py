# src/common/dominance.py
from __future__ import annotations
from typing import Sequence


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """
    Maximization dominance:
      a dominates b if a_i >= b_i for all i AND exists i with a_i > b_i
    """
    ge_all = True
    gt_some = False
    for ai, bi in zip(a, b):
        if ai < bi:
            ge_all = False
            break
        if ai > bi:
            gt_some = True
    return ge_all and gt_some