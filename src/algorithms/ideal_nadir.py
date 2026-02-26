# src/algorithms/ideal_nadir.py

"""
This is where you use 10^{-E} with safe computed E.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

from gurobipy import GRB

from src.common.symbols import Indices
from src.common.parameters import Parameters
from src.common.bounds import compute_safe_E
from src.common.solve_tracker import SolveTracker
from src.model.build import build_stage2_base
from src.model.zexpr import build_z_defs


@dataclass(frozen=True)
class IdealNadir:
    z_ideal: List[float]   # length n_z, index 0..n_z-1
    z_nadir: List[float]   # length n_z


def _eval_z_vector(z_defs: Dict[int, object], n_z: int) -> List[float]:
    """
    Evaluate z_1..z_n at the current model solution.

    IMPORTANT FIX:
    In gurobipy, you must evaluate a LinExpr using:
        expr.getValue()
    NOT:
        model.getValue(expr)
    """
    out = []
    for i in range(1, n_z + 1):
        out.append(float(z_defs[i].expr.getValue()))
    return out


def compute_ideal_and_approx_nadir(
    idx: Indices,
    par: Parameters,
    g_value: int,
    n_z: int = 7,
    time_limit: float | None = None,
    tracker: SolveTracker | None = None,
) -> IdealNadir:
    """
    Implements Eq.(36)-(40) / Algorithm 1.

    For each objective i:
        maximize z_i + 10^{-E} * sum_{j!=i} z_j

    Then:
        - z_ideal[i]  = z_i at that solution
        - z_nadir[i]  = min over all rows (approximate nadir)

    IMPORTANT FIX:
    - run/main.py passes 'tracker', so we must accept it here.
    - prints "=== SOLVE #k ===" for each ideal solve.
    """

    if tracker is None:
        tracker = SolveTracker()

    idx.validate()
    par.validate(idx)

    # Compute safe E analytically (Section 3)
    E = compute_safe_E(idx, par, n_z=n_z)
    weight = 10.0 ** (-E)

    # z_i^* values (ideal components) and table z_j^{i*} for nadir computation
    z_star: List[float] = [0.0] * n_z
    z_table: List[List[float]] = [[0.0] * n_z for _ in range(n_z)]  # row i, col j

    for i in range(1, n_z + 1):

        # Build feasible region with fixed g (Stage 2 base model)
        bm = build_stage2_base(idx, par, g_value, name=f"Pz{i}")

        try:
            if time_limit is not None:
                bm.m.Params.TimeLimit = float(time_limit)

            z_defs = build_z_defs(idx, par, bm.var)

            # Eq.(36): maximize z_i + 10^{-E} * sum_{j!=i} z_j
            primary = z_defs[i].expr
            perturb = sum(z_defs[j].expr for j in range(1, n_z + 1) if j != i)

            bm.m.setObjective(primary + weight * perturb, GRB.MAXIMIZE)

            tracker.tick(f"STAGE 2 / Alg.1: IDEAL solve i={i} (E={E}, 10^(-E)={weight:g})")
            bm.m.optimize()

            # IMPORTANT:
            # - OPTIMAL is obviously fine.
            # - TIME_LIMIT is also fine ONLY if Gurobi found at least one feasible solution (SolCount>0).
            # - If SolCount==0, then the solve produced no incumbent -> we cannot evaluate z-values safely.
            if bm.m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or bm.m.SolCount <= 0:
                raise RuntimeError(
                    f"Ideal/Nadir subproblem i={i} not solved properly. "
                    f"Status={bm.m.Status}, SolCount={bm.m.SolCount}"
                )

            # Evaluate all z at this solution
            zv = _eval_z_vector(z_defs, n_z)

            for j in range(n_z):
                z_table[i - 1][j] = zv[j]

            # Eq.(37)-(38): z_i^* = z_i^{rho*} - rho
            # We safely take realized z_i value (perturbation only breaks ties)
            z_star[i - 1] = zv[i - 1]

        finally:
            # IMPORTANT FIX (memory/stability):
            # We create a fresh model per i; dispose it to avoid memory buildup in scalability runs.
            bm.m.dispose()

    # Approx nadir: z_i^nad = min_j z_i^{j*}
    z_nad = [min(z_table[row][col] for row in range(n_z)) for col in range(n_z)]

    return IdealNadir(z_ideal=z_star, z_nadir=z_nad)