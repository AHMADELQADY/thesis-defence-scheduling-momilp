# src/algorithms/augmented_epsilon.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Sequence, Dict, Optional

from time import perf_counter
from pathlib import Path
from gurobipy import GRB

from src.common.solve_tracker import SolveTracker
from src.model.build import build_stage2_base
from src.model.zexpr import build_z_defs


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class SolutionPoint:
    z: List[float]                 # full objective vector in maximize-form (z1..z_n)
    eps: List[float]               # eps used (bounded objectives order)
    z_bounded: List[float]         # achieved bounded objectives (same order as eps)
    status: int                    # GRB status code
    proven: bool                   # True if OPTIMAL; False if TIME_LIMIT incumbent accepted


@dataclass(frozen=True)
class AugEpsMetrics:
    N_count: int
    I_count: int
    skipN: int
    skipI: int
    timeN: float
    timeI: float


# =============================================================================
# Helpers
# =============================================================================

def update_v(v: List[int], steps: List[int]) -> Tuple[List[int], bool]:
    """
    Paper Algorithm 2 style counter.
    Increments the first component that can still increase, then resets lower-order ones.
    """
    v = v[:]
    if all(v[i] == steps[i] for i in range(len(v))):
        return v, True
    i_star = next(i for i in range(len(v)) if v[i] < steps[i])
    v[i_star] += 1
    for i in range(i_star):
        v[i] = 0
    return v, False


def compute_eps_for_bounded(
    z_nad: List[float],
    z_star: List[float],
    v: List[int],
    steps: List[int],
    bounded_objectives: Sequence[int],
) -> List[float]:
    """
    Paper Eq.(44): epsilon grid for bounded objectives only.
    Guard steps==0 to avoid division by zero.
    """
    eps: List[float] = []
    for local, obj_id in enumerate(bounded_objectives):
        if steps[local] <= 0:
            eps_i = z_nad[obj_id - 1]
        else:
            p = 1.0 / steps[local]
            eps_i = z_nad[obj_id - 1] + v[local] * p * (z_star[obj_id - 1] - z_nad[obj_id - 1])
        eps.append(float(eps_i))
    return eps


def vec_dominates(a: List[float], b: List[float]) -> bool:
    """
    Dominance for '>= all and > at least one' (maximize form).
    """
    if len(a) != len(b):
        raise ValueError("dominance vectors must have same length")
    ge_all = True
    gt_any = False
    for i in range(len(a)):
        if a[i] < b[i]:
            ge_all = False
            break
        if a[i] > b[i]:
            gt_any = True
    return ge_all and gt_any


def add_to_N_keep_nondominated(N: List[SolutionPoint], cand: SolutionPoint) -> None:
    """
    Keep N as a NON-DOMINATED set w.r.t. z_bounded (the bounded objectives).
    This makes skip_solutions() correct.
    """
    # If some existing solution dominates candidate -> drop candidate
    for sol in N:
        if vec_dominates(sol.z_bounded, cand.z_bounded):
            return

    # Otherwise remove solutions dominated by candidate
    N[:] = [sol for sol in N if not vec_dominates(cand.z_bounded, sol.z_bounded)]
    N.append(cand)


def skip_solutions(N: List[SolutionPoint], eps: List[float]) -> bool:
    """
    Skip eps if an existing NON-DOMINATED solution already satisfies eps constraints:
        z_bounded >= eps component-wise
    """
    for sol in N:
        if all(sol.z_bounded[i] >= eps[i] for i in range(len(eps))):
            return True
    return False


def skip_infeasible(I: List[List[float]], eps: List[float]) -> bool:
    """
    If eps is harder (>=) than an already-proven infeasible epsbar, then skip.
    """
    for epsbar in I:
        if all(eps[i] >= epsbar[i] for i in range(len(eps))):
            return True
    return False


# =============================================================================
# Main algorithm
# =============================================================================

def solve_augmented_epsilon(
    idx, par, g_value: int,
    z_star: List[float], z_nad: List[float],
    steps: List[int],
    *,
    n_z: int = 7,
    bounded_objectives: Sequence[int] = (3, 4),
    fully_considered_objective: int = 1,

    # Paper-style total budget mode:
    total_time_budget: Optional[float] = None,   # seconds for WHOLE Alg.5 loop

    # Fixed per-iteration time limit (legacy mode)
    time_limit_per_iter: Optional[float] = None, # seconds per P^eps solve

    return_metrics: bool = False,
    tracker: Optional[SolveTracker] = None,

    accept_time_limit_incumbent: bool = False,
    debug_iis: bool = False,
    iis_dir: str = "data/debug/iis",
):
    """
    Augmented ε-constraint method (paper Algorithm 5).

    ✅ Fix A (screenshot):
      - Build P_eps ONCE outside the loop
      - Add ε-constraints ONCE (store them)
      - Each iteration: update constr.RHS = eps_value + update() + optimize()

    Also included because you reuse the model:
      - bm.m.reset() before each optimize
      - keep N non-dominated so skip_solutions is safe
      - TIME_LIMIT without incumbent is NOT infeasible -> do not add to I
    """

    if tracker is None:
        tracker = SolveTracker()

    if len(z_star) != n_z or len(z_nad) != n_z:
        raise ValueError("z_star and z_nad must have length n_z")

    bounded_objectives = list(bounded_objectives)
    if len(steps) != len(bounded_objectives):
        raise ValueError("steps must have same length as bounded_objectives")

    if fully_considered_objective < 1 or fully_considered_objective > n_z:
        raise ValueError("fully_considered_objective must be in 1..n_z")

    for obj_id in bounded_objectives:
        if obj_id < 1 or obj_id > n_z:
            raise ValueError("bounded objective id out of range 1..n_z")
        if obj_id == fully_considered_objective:
            raise ValueError("fully_considered objective cannot be bounded too")

    # if total budget exists, ignore fixed per-iter tl
    if total_time_budget is not None:
        time_limit_per_iter = None

    v = [0] * len(bounded_objectives)
    stop = False

    N: List[SolutionPoint] = []
    I: List[List[float]] = []   # proven infeasible eps vectors only

    skipN = 0
    skipI = 0
    timeN = 0.0
    timeI = 0.0

    iis_path = Path(iis_dir)
    alg5_start = perf_counter()

    # total epsilon iterations = product (steps[d]+1)
    total_iters = 1
    for s in steps:
        total_iters *= (max(s, 0) + 1)

    iter_idx = 0

    # ======================================================================
    # FIX A: build model once, add ε-constraints once, reuse
    # ======================================================================
    bm = build_stage2_base(idx, par, g_value, name="P_eps")
    z_defs = build_z_defs(idx, par, bm.var)

    # ε-constraints (create once; store handles)
    eps_constrs = []
    for local, obj_id in enumerate(bounded_objectives):
        c = bm.m.addConstr(
            z_defs[obj_id].expr >= float(z_nad[obj_id - 1]),
            name=f"eps_obj{obj_id}",
        )
        eps_constrs.append(c)

    # surplus vars Eq.(41) (create once; does not depend on eps value)
    s_vars: Dict[int, object] = {}
    for obj_id in bounded_objectives:
        denom = (z_star[obj_id - 1] - z_nad[obj_id - 1])
        if abs(denom) < 1e-9:
            s_vars[obj_id] = bm.m.addVar(lb=0.0, ub=0.0, vtype=GRB.CONTINUOUS, name=f"surplus_{obj_id}")
        else:
            s_vars[obj_id] = bm.m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"surplus_{obj_id}")
            bm.m.addConstr(
                z_defs[obj_id].expr - z_nad[obj_id - 1] == denom * s_vars[obj_id],
                name=f"surplus_def_{obj_id}",
            )

    # Eq.(42): maximize z_fully + (n_z - 0.9)^(-1) * sum surplus
    phi = (1.0 / (n_z - 0.9)) * sum(s_vars[obj_id] for obj_id in bounded_objectives)
    bm.m.setObjective(z_defs[fully_considered_objective].expr + phi, GRB.MAXIMIZE)

    bm.m.update()

    # ======================================================================
    # Main enumeration loop
    # ======================================================================
    while not stop:
        iter_idx += 1
        eps = compute_eps_for_bounded(z_nad, z_star, v, steps, bounded_objectives)

        if skip_solutions(N, eps):
            skipN += 1

        elif skip_infeasible(I, eps):
            skipI += 1

        else:
            # --------------------------------------------------------------
            # Fix A core: update RHS only + update() + optimize()
            # --------------------------------------------------------------
            for local, c in enumerate(eps_constrs):
                c.RHS = float(eps[local])
            bm.m.update()

            # time limits
            if total_time_budget is not None:
                elapsed = perf_counter() - alg5_start
                remaining = max(0.0, float(total_time_budget) - elapsed)
                remaining_iters = max(1, total_iters - (iter_idx - 1))
                bm.m.Params.TimeLimit = float(remaining / remaining_iters)
            elif time_limit_per_iter is not None:
                bm.m.Params.TimeLimit = float(time_limit_per_iter)
            else:
                bm.m.Params.TimeLimit = 0.0

            tracker.tick(f"STAGE 2 / Alg.5: P^eps solve v={v} eps={eps}")

            t0 = perf_counter()
            bm.m.reset()
            bm.m.optimize()
            dt = perf_counter() - t0

            st = bm.m.Status

            if st == GRB.INFEASIBLE:
                I.append(eps)
                timeI += dt

                if debug_iis:
                    try:
                        iis_path.mkdir(parents=True, exist_ok=True)
                        bm.m.computeIIS()
                        fname = f"iis_v_{'_'.join(map(str, v))}.ilp"
                        bm.m.write(str(iis_path / fname))
                    except Exception as ex:
                        print(f"[debug_iis] IIS export failed: {ex}")

            elif st == GRB.OPTIMAL:
                z_vec = [float(z_defs[i_obj].expr.getValue()) for i_obj in range(1, n_z + 1)]
                z_bounded = [z_vec[obj_id - 1] for obj_id in bounded_objectives]

                add_to_N_keep_nondominated(
                    N,
                    SolutionPoint(
                        z=z_vec,
                        eps=eps,
                        z_bounded=z_bounded,
                        status=st,
                        proven=True,
                    ),
                )
                timeN += dt

            elif st == GRB.TIME_LIMIT:
                # TIME_LIMIT != infeasible
                if accept_time_limit_incumbent and bm.m.SolCount > 0:
                    z_vec = [float(z_defs[i_obj].expr.getValue()) for i_obj in range(1, n_z + 1)]
                    z_bounded = [z_vec[obj_id - 1] for obj_id in bounded_objectives]

                    add_to_N_keep_nondominated(
                        N,
                        SolutionPoint(
                            z=z_vec,
                            eps=eps,
                            z_bounded=z_bounded,
                            status=st,
                            proven=False,
                        ),
                    )
                    timeN += dt
                else:
                    # unknown -> do nothing
                    pass

            else:
                # unknown -> do nothing
                pass

        v, stop = update_v(v, steps)

    if return_metrics:
        metrics = AugEpsMetrics(
            N_count=len(N),
            I_count=len(I),
            skipN=skipN,
            skipI=skipI,
            timeN=float(timeN),
            timeI=float(timeI),
        )
        return N, I, metrics

    return N, I