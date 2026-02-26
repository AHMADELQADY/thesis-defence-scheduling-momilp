# scalability.py
from __future__ import annotations

import argparse
from time import perf_counter
from pathlib import Path
from typing import Dict, List, Tuple

from src.instance_generator.presets import SMALL, MED, LARGE, GRID_FIXED2, GRID_FIXED1
from src.instance_generator.generator import generate_instance
from src.instance_generator.io import save_instance
from src.run.main import run_two_stage

# CSV helper you added
from src.experiments.csv_export import write_csv


# =============================================================================
# Formatting helpers (match paper-style "Data" columns)
# =============================================================================

def _fmt_probs_lik(p0: float) -> str:
    """
    Paper tables show lik distribution as [p(lik=0), p(lik=1), p(lik=2)]
    with p(lik=1)=p(lik=2)=(1-p0)/2.
    """
    p1 = (1.0 - p0) / 2.0
    return f"[{p0:.2f}, {p1:.2f}, {p1:.2f}]"


def _fmt_probs_mkp(p0: float) -> str:
    """
    Paper tables show mkp distribution as [p(mkp=0), p(mkp=1)].
    """
    return f"[{p0:.2f}, {1.0 - p0:.2f}]"


def _fmt_vh(p21: float) -> str:
    """
    For v_i and h_i the tables show probabilities of choosing:
      - [1]   with prob (1 - p21)
      - [2,1] with prob p21
    We print as [p([1]), p([2,1])] = [1-p21, p21]
    """
    return f"[{1.0 - p21:.1f}, {p21:.1f}]"


# =============================================================================
# Table runner
# =============================================================================

def run_table(
    name: str,
    size,
    seeds_start: int,
    tl_stage1: float,
    tl_ideal: float,
    tl_eps: float,
    steps: int,
    save_instances: bool = False,
    instances_dir: Path | None = None,
    tag: str = "",
    # ------------------------------------------------------------------
    # NEW (paper-faithful):
    # Algorithm 5 has a TOTAL CPU budget (12h) that must be divided by the
    # remaining iterations, after subtracting time already spent in previous steps.
    # We pass it as a single budget (seconds).
    # ------------------------------------------------------------------
    budget_eps: float | None = None,
) -> Tuple[List[str], List[Dict]]:
    """
    Runs one of the paper tables (C.1 / C.2 / C.3) and returns:
      - fieldnames: CSV column names
      - rows: list of dict rows (paper columns)

    Notes:
    - We keep printing the same console lines as before.
    - Additionally we store each row into `rows` for CSV export.
    - Optionally we save every generated instance into a separate folder.

    Paper time-limit policy (Section 6.1.2):
      (1) Finding g: 30 minutes.
      (2) Algorithm 1: 2 hours, equally divided between the seven objectives.
      (3) Algorithm 5: For each iteration, use:
            (12 hours - time used by previous steps and iterations so far)
            / (remaining number of iterations)
          i.e., a dynamic per-iteration TimeLimit.

    This file implements (3) when you pass `budget_eps`.
    """

    print(f"\n{name}")
    if name.endswith("C.1"):
        print("Computational experiments - small instances (n_i = 25, n_j = 20).")
    elif name.endswith("C.2"):
        print("Computational experiments - medium instances (n_i = 38, n_j = 30).")
    else:
        print("Computational experiments - large instances (n_i = 50, n_j = 40).")

    header = (
        "N  p(n_i.n_j.n_t.n_k.n_ell.n_p.n_q)  d  u_i  e_ijt  c_i  lik  mkp  v_i  h_i  r_iq  t_iq  "
        "|N|  |I|  skip^N  skip^I  time^N  time^I  g  CPU(seconds)"
    )
    print(header)

    # CSV columns in the same order as the printed header / paper layout
    fieldnames: List[str] = [
        "N",
        "p(n_i.n_j.n_t.n_k.n_ell.n_p.n_q)",
        "d",
        "u_i",
        "e_ijt",
        "c_i",
        "lik",
        "mkp",
        "v_i",
        "h_i",
        "r_iq",
        "t_iq",
        "|N|",
        "|I|",
        "skip^N",
        "skip^I",
        "time^N",
        "time^I",
        "g",
        "CPU(seconds)",
    ]

    rows: List[Dict] = []

    # Paper layout: for each table, 32 instances:
    # - first 16 rows fixed_roles=2
    # - next 16 rows fixed_roles=1
    # We achieve 16 per block by doing:
    #   8 configurations * reps(=2) = 16
    reps = 2

    # Instance id ranges as in the paper
    inst_id = 1 if size.n_i == 25 else (33 if size.n_i == 38 else 65)
    seed = seeds_start

    # Total number of Algorithm 5 iterations:
    # steps_per_obj = steps, for objectives 2..7 => 6 dims
    # v_i runs 0..steps inclusive => (steps+1) per dim
    # total = (steps+1)^(n_z-1) = (steps+1)^6
    n_z = 7
    n_dims_eps = n_z - 1
    #total_eps_iters = (steps + 1) ** n_dims_eps
    total_eps_iters = (steps + 1) ** 2


    # Block 0 => fixed_roles=2 configs, Block 1 => fixed_roles=1 configs
    for block in range(2):
        configs = GRID_FIXED2 if block == 0 else GRID_FIXED1

        for cfg in configs:
            for _ in range(reps):
                # -------------------------
                # Generate instance
                # -------------------------
                idx, par = generate_instance(size, cfg, seed=seed)

                # -------------------------
                # Save generated instance (optional)
                # -------------------------
                if save_instances:
                    assert instances_dir is not None
                    instances_dir.mkdir(parents=True, exist_ok=True)

                    # Make filename unique + informative
                    inst_name = (
                        f"inst_{tag}_N{inst_id}_seed{seed}_"
                        f"ni{size.n_i}_nj{size.n_j}_"
                        f"roles{cfg.fixed_roles}_"
                        f"plik{cfg.p_lik0}_pmkp{cfg.p_mkp0}_"
                        f"pv{cfg.p_v_21}_ph{cfg.p_h_21}.json"
                    )
                    save_instance(instances_dir / inst_name, idx, par)

                # -------------------------
                # Run full pipeline (Stage1 + Ideal/Nadir + Augmented eps)
                # -------------------------
                #
                # IMPORTANT FIX (scalability robustness):
                # Some instances may hit TimeLimit inside the IDEAL/NADIR phase
                # with SolCount==0 (no feasible incumbent), which raises RuntimeError.
                # In paper-style scalability tables, you do NOT crash; you record the row
                # as "failed under time limits" and continue with the next instance.
                #
                # IMPORTANT (paper-faithful Algorithm 5):
                # If budget_eps is provided, we do NOT pass a fixed tl_eps.
                # Instead, we pass the total budget to run_two_stage, which will
                # divide it per iteration (dynamic TimeLimit) inside Algorithm 5.
                #
                t0 = perf_counter()
                try:
                    if budget_eps is None:
                        # Old behavior: fixed per-iteration time limit for P^eps
                        g_star, in_res, N, I, metrics = run_two_stage(
                            idx,
                            par,
                            steps_per_obj=steps,
                            time_limit_stage1=tl_stage1,
                            time_limit_ideal=tl_ideal,
                            time_limit_eps=tl_eps,
                        )
                    else:
                        # New behavior (paper-style): pass TOTAL budget for Algorithm 5.
                        # Alg.5 will compute per-iteration TimeLimit dynamically from:
                        #   remaining_budget / remaining_iterations
                        # while also subtracting time used by previous steps and past iters.
                        #
                        # NOTE:
                        # Your pipeline already supports this via `total_time_budget_eps`.
                        # Do NOT pass custom kwargs like eps_budget_total/eps_total_iters.
                        g_star, in_res, N, I, metrics = run_two_stage(
                            idx,
                            par,
                            steps_per_obj=steps,
                            time_limit_stage1=tl_stage1,
                            time_limit_ideal=tl_ideal,
                            total_time_budget_eps=budget_eps,
                        )
                    failed = False
                except RuntimeError as e:
                    # We keep going: record a "failed" row with sentinel values.
                    # (This matches the idea of reporting TL/failed cases in scalability.)
                    failed = True
                    g_star = -1  # sentinel: could not complete stage2 pipeline
                    metrics = type(
                        "MetricsFallback",
                        (),
                        dict(N_count=0, I_count=0, skipN=0, skipI=0, timeN=0.0, timeI=0.0),
                    )()
                    # Optional: show the reason once, but do not stop the table
                    print(f"WARNING: instance {inst_id} failed pipeline: {e}")

                cpu = perf_counter() - t0

                # -------------------------
                # Format paper-style "Data" columns
                # -------------------------
                p_str = f"p({size.n_i}.{size.n_j}.{size.n_t}.{size.n_k}.{size.n_ell}.{size.n_p}.{size.n_q})"
                c_i_val = int((size.n_i + 1) // 2)  # matches your print (13/19/25)

                # -------------------------
                # Console print (unchanged style)
                # -------------------------
                # If failed, counts/times are 0 and g=-1, CPU still printed.
                line = (
                    f"{inst_id:<2d} {p_str:<28} {size.d:<2d} "
                    f"[0.7, 0.3] {cfg.fixed_roles:<5d} {c_i_val:<3d} "
                    f"{_fmt_probs_lik(cfg.p_lik0):<18} {_fmt_probs_mkp(cfg.p_mkp0):<12} "
                    f"{_fmt_vh(cfg.p_v_21):<10} {_fmt_vh(cfg.p_h_21):<10} "
                    f"3    3    "
                    f"{metrics.N_count:<3d} {metrics.I_count:<3d} "
                    f"{metrics.skipN:<6d} {metrics.skipI:<6d} "
                    f"{metrics.timeN:<6.0f} {metrics.timeI:<6.0f} "
                    f"{g_star:<2d} {cpu:>8.0f}"
                )
                print(line)

                # -------------------------
                # CSV row (same info, structured)
                # IMPORTANT:
                # - time^N/time^I are durations; paper prints integers in tables,
                #   so we round to int for CSV (you can keep float if you prefer).
                # - CPU(seconds) rounded to int to mimic paper table look.
                # -------------------------
                row = {
                    "N": inst_id,
                    "p(n_i.n_j.n_t.n_k.n_ell.n_p.n_q)": p_str,
                    "d": size.d,
                    "u_i": "[0.7, 0.3]",
                    "e_ijt": cfg.fixed_roles,
                    "c_i": c_i_val,
                    "lik": _fmt_probs_lik(cfg.p_lik0),
                    "mkp": _fmt_probs_mkp(cfg.p_mkp0),
                    "v_i": _fmt_vh(cfg.p_v_21),
                    "h_i": _fmt_vh(cfg.p_h_21),
                    "r_iq": 3,
                    "t_iq": 3,
                    "|N|": int(metrics.N_count),
                    "|I|": int(metrics.I_count),
                    "skip^N": int(metrics.skipN),
                    "skip^I": int(metrics.skipI),
                    "time^N": int(round(float(metrics.timeN))),
                    "time^I": int(round(float(metrics.timeI))),
                    "g": int(g_star),
                    "CPU(seconds)": int(round(cpu)),
                }
                rows.append(row)

                # next instance
                inst_id += 1
                seed += 1

    return fieldnames, rows


# =============================================================================
# CLI entrypoint
# =============================================================================

def main():
    ap = argparse.ArgumentParser()

    # Same knobs as before
    ap.add_argument("--steps", type=int, default=5)

    # Paper time limits (Section 6.1.2):
    #   - g: 30 minutes
    #   - Algorithm 1: 2 hours, equally divided by 7 objectives
    #   - Algorithm 5: 12 hours TOTAL (dynamic per-iteration)
    ap.add_argument("--tl_stage1", type=float, default=60.0)
    ap.add_argument("--tl_ideal", type=float, default=60.0)

    # Backward-compatible name (old): interpreted as per-iteration TL if --budget_eps is not given
    ap.add_argument("--tl_eps", type=float, default=60.0)

    # NEW (recommended): total CPU budget for Algorithm 5 (seconds)
    ap.add_argument("--budget_eps", type=float, default=None)

    ap.add_argument("--seed_start", type=int, default=1)

    # NEW: output folder for CSVs
    ap.add_argument("--out_dir", type=str, default="data/results")

    # -------------------------
    # Instance saving (separate from test.py)
    # -------------------------
    ap.add_argument("--save_instances", action="store_true")
    ap.add_argument("--instances_dir", type=str, default="data/generated/instances_scalability")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    instances_dir = Path(args.instances_dir)

    # ------------------------------------------------------------------
    # Alias behavior:
    # - If user supplies --budget_eps, we run paper-style (dynamic per-iteration)
    # - Else we run old fixed-per-iteration tl_eps mode.
    # ------------------------------------------------------------------
    if args.budget_eps is None:
        budget_eps = None
        tl_eps = args.tl_eps
    else:
        budget_eps = float(args.budget_eps)
        tl_eps = args.tl_eps  # unused in paper-budget mode

    # # Table C.1 (small)
    # f1, rows1 = run_table(
    #     "Table C.1",
    #     SMALL,
    #     args.seed_start,
    #     args.tl_stage1,
    #     args.tl_ideal,
    #     tl_eps,
    #     args.steps,
    #     save_instances=args.save_instances,
    #     instances_dir=instances_dir,
    #     tag="C1",
    #     budget_eps=budget_eps,
    # )
    # write_csv(out_dir / "table_C1.csv", f1, rows1)

    # # Table C.2 (medium)
    # f2, rows2 = run_table(
    #     "Table C.2",
    #     MED,
    #     args.seed_start + 1000,
    #     args.tl_stage1,
    #     args.tl_ideal,
    #     tl_eps,
    #     args.steps,
    #     save_instances=args.save_instances,
    #     instances_dir=instances_dir,
    #     tag="C2",
    #     budget_eps=budget_eps,
    # )
    # write_csv(out_dir / "table_C2.csv", f2, rows2)
    #
    # Table C.3 (large)
    f3, rows3 = run_table(
        "Table C.3",
        LARGE,
        args.seed_start + 2000,
        args.tl_stage1,
        args.tl_ideal,
        tl_eps,
        args.steps,
        save_instances=args.save_instances,
        instances_dir=instances_dir,
        tag="C3",
        budget_eps=budget_eps,
    )
    write_csv(out_dir / "table_C3.csv", f3, rows3)

    print(f"\nCSV written to: {out_dir.resolve()}")
    print(" - table_C1.csv")
    print(" - table_C2.csv")
    print(" - table_C3.csv")

    if args.save_instances:
        print(f"Instances written to: {instances_dir.resolve()}")

    # Helpful hint to reproduce the paper’s exact limits
    # (g=30min, Alg1=2h/7, Alg5=12h total)
    if args.budget_eps is not None:
        print("\nPaper-style Algorithm 5 budget mode is ON.")
    else:
        print("\nFixed per-iteration tl_eps mode is ON (NOT paper-style).")


if __name__ == "__main__":
    main()