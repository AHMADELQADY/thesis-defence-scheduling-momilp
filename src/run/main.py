# src/run/main.py
from __future__ import annotations

from src.common.symbols import Indices
from src.common.parameters import Parameters
from src.common.solve_tracker import SolveTracker

from src.algorithms.stage1_g import solve_g_star
from src.algorithms.ideal_nadir import compute_ideal_and_approx_nadir
from src.algorithms.augmented_epsilon import solve_augmented_epsilon


def run_two_stage(
    idx: Indices,
    par: Parameters,
    *,
    steps_per_obj: int = 5,
    time_limit_stage1: float | None = None,
    time_limit_ideal: float | None = None,

    # ------------------------------------------------------------
    # IMPORTANT: support BOTH modes (so test.py and scalability.py work)
    # ------------------------------------------------------------
    time_limit_eps: float | None = None,           # fixed per-iteration TL (legacy / fast testing)
    total_time_budget_eps: float | None = None,    # paper-style total budget for whole Alg.5

    n_z: int = 7,
    return_metrics: bool = True,

    # OPTIONAL (robustness / debug knobs)
    accept_time_limit_incumbent: bool = False,
    debug_iis: bool = False,
    iis_dir: str = "data/debug/iis",

    # PAPER settings (Section 6.1.3)
    bounded_objectives: tuple[int, ...] = (3, 4),
    fully_considered_objective: int = 1,
):
    """
    Runs the full paper pipeline:

      Stage 1: maximize g (scheduled defences)
      Stage 2 / Algorithm 1: compute ideal z* and approximate nadir z^nad using 10^{-E}
      Stage 2 / Algorithm 5: augmented ε-constraint enumeration

    Time policies:
      - Stage 1: can use time_limit_stage1
      - Algorithm 1: can use time_limit_ideal
      - Algorithm 5: choose ONE:
          (A) total_time_budget_eps (paper-style)
          (B) time_limit_eps (fixed per-iteration, good for testing)

    If BOTH are provided:
      - we prioritize paper budget mode (total_time_budget_eps).
    """

    idx.validate()
    par.validate(idx)

    tracker = SolveTracker()

    # ================================================================
    # Stage 1 — Maximize g
    # ================================================================
    g_star = solve_g_star(
        idx,
        par,
        time_limit=time_limit_stage1,
        tracker=tracker,
    )

    # ================================================================
    # Stage 2 — Algorithm 1 (Ideal + Approximate Nadir)
    # ================================================================
    in_res = compute_ideal_and_approx_nadir(
        idx,
        par,
        g_star,
        n_z=n_z,
        time_limit=time_limit_ideal,
        tracker=tracker,
    )

    # ================================================================
    # Stage 2 — Algorithm 5 (Augmented ε-constraint)
    # ================================================================
    steps = [steps_per_obj] * len(bounded_objectives)

    # choose mode
    total_budget = total_time_budget_eps if total_time_budget_eps is not None else None
    per_iter_tl = None if total_budget is not None else time_limit_eps

    if return_metrics:
        N, I, metrics = solve_augmented_epsilon(
            idx,
            par,
            g_star,
            in_res.z_ideal,
            in_res.z_nadir,
            steps,
            n_z=n_z,
            bounded_objectives=bounded_objectives,
            fully_considered_objective=fully_considered_objective,

            # Alg.5 time control:
            total_time_budget=total_budget,
            time_limit_per_iter=per_iter_tl,

            return_metrics=True,
            tracker=tracker,

            accept_time_limit_incumbent=accept_time_limit_incumbent,
            debug_iis=debug_iis,
            iis_dir=iis_dir,
        )
        return g_star, in_res, N, I, metrics

    N, I = solve_augmented_epsilon(
        idx,
        par,
        g_star,
        in_res.z_ideal,
        in_res.z_nadir,
        steps,
        n_z=n_z,
        bounded_objectives=bounded_objectives,
        fully_considered_objective=fully_considered_objective,

        total_time_budget=total_budget,
        time_limit_per_iter=per_iter_tl,

        return_metrics=False,
        tracker=tracker,
        accept_time_limit_incumbent=accept_time_limit_incumbent,
        debug_iis=debug_iis,
        iis_dir=iis_dir,
    )

    return g_star, in_res, N, I