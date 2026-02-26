# src/algorithms/stage1_g.py
from __future__ import annotations
from gurobipy import GRB

from src.common.symbols import Indices
from src.common.parameters import Parameters
from src.common.solve_tracker import SolveTracker
from src.model.build import build_stage1_g


def solve_g_star(
    idx: Indices,
    par: Parameters,
    *,
    time_limit: float | None = None,
    tracker: SolveTracker | None = None,
) -> int:
    """
    Stage 1 (paper): maximize g (scheduled defences)

    IMPORTANT FIX:
    - run/main.py passes 'tracker', so we must accept it here.
    - prints "=== SOLVE #k ===" so you know where you are.
    """
    if tracker is None:
        tracker = SolveTracker()

    bm = build_stage1_g(idx, par)

    try:
        if time_limit is not None:
            bm.m.Params.TimeLimit = float(time_limit)

        tracker.tick("STAGE 1: maximize g")
        bm.m.optimize()

        # allow TIME_LIMIT: take best bound solution if exists
        if bm.m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            if bm.m.SolCount <= 0:
                raise RuntimeError(f"Stage1 g has no solution. Status={bm.m.Status}")
            g_star = int(round(bm.m.ObjVal))
            return g_star

        raise RuntimeError(f"Stage1 g failed. Status={bm.m.Status}")

    finally:
        # IMPORTANT FIX (memory/stability):
        # Stage 1 builds a model; dispose it to avoid memory buildup in scalability runs.
        bm.m.dispose()