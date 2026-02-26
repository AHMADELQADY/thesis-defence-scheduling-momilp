# test.py (FAST sanity)
from __future__ import annotations

import argparse
import os
import csv
from datetime import datetime
from time import perf_counter
from pathlib import Path

from src.instance_generator.config import InstanceSize, PaperKnobs
from src.instance_generator.generator import generate_instance
from src.instance_generator.io import save_instance
from src.run.main import run_two_stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)

    # keep CLI knobs, but default them to FAST values
    ap.add_argument("--steps", type=int, default=1)          # only 1 eps step
    ap.add_argument("--tl_stage1", type=float, default=5.0)  # seconds
    ap.add_argument("--tl_ideal", type=float, default=5.0)
    ap.add_argument("--tl_eps", type=float, default=5.0)

    # -------------------------
    # Instance saving (separate from scalability)
    # -------------------------
    ap.add_argument("--save_instance", action="store_true")
    ap.add_argument("--instance_dir", type=str, default="data/generated/instances_test")
    ap.add_argument("--instance_name", type=str, default=None)  # optional override

    args = ap.parse_args()

    # SUPER small instance: just to verify the pipeline runs end-to-end
    size = InstanceSize(n_i=8, n_j=6)

    knobs = PaperKnobs(
        fixed_roles=2,
        p_lik0=0.82,
        p_mkp0=0.86,
        p_v_21=0.2,
        p_h_21=0.2,
    )

    # -------------------------
    # Generate instance
    # -------------------------
    idx, par = generate_instance(size, knobs, seed=args.seed)

    # -------------------------
    # Save the generated instance (optional)
    # -------------------------
    if args.save_instance:
        inst_dir = Path(args.instance_dir)
        inst_dir.mkdir(parents=True, exist_ok=True)

        name = args.instance_name or f"inst_test_seed{args.seed}_ni{size.n_i}_nj{size.n_j}.json"
        inst_path = inst_dir / name

        save_instance(inst_path, idx, par)
        print(f"Instance written to: {inst_path.resolve()}")

    # -------------------------
    # Run the pipeline
    # -------------------------
    t0 = perf_counter()
    g_star, in_res, N, I, metrics = run_two_stage(
        idx, par,
        steps_per_obj=args.steps,
        time_limit_stage1=args.tl_stage1,
        time_limit_ideal=args.tl_ideal,
        time_limit_eps=args.tl_eps,
    )
    cpu = perf_counter() - t0

    print("\n=== FAST TEST RUN OUTPUT ===")
    print(f"g*={g_star}  CPU={cpu:.2f}s")
    print(f"|N|={metrics.N_count} |I|={metrics.I_count} "
          f"skipN={metrics.skipN} skipI={metrics.skipI} "
          f"timeN={metrics.timeN:.2f} timeI={metrics.timeI:.2f}")

    print("\n=== IDEAL VECTOR (z*) ===")
    print(in_res.z_ideal)

    print("\n=== NADIR VECTOR (approx) ===")
    print(in_res.z_nadir)

    print("\n=== NON-DOMINATED SOLUTIONS (N) ===")
    for k, sol in enumerate(N, 1):
        print(f"Solution {k}: z={sol.z} eps={sol.eps}")

    # -------------------------
    # CSV export (like scalability)
    # -------------------------
    out_dir = os.path.join("data", "generated")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "test_run.csv")

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "steps_per_obj": args.steps,
        "tl_stage1": args.tl_stage1,
        "tl_ideal": args.tl_ideal,
        "tl_eps": args.tl_eps,
        "n_i": size.n_i,
        "n_j": size.n_j,
        "g_star": g_star,
        "cpu_sec": round(cpu, 4),
        "N_count": metrics.N_count,
        "I_count": metrics.I_count,
        "skipN": metrics.skipN,
        "skipI": metrics.skipI,
        "timeN": round(metrics.timeN, 4),
        "timeI": round(metrics.timeI, 4),
        "z_ideal": str(in_res.z_ideal),
        "z_nadir": str(in_res.z_nadir),
        "N_solutions": str([sol.z for sol in N]),
        "N_eps": str([sol.eps for sol in N]),
    }

    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    print(f"\nCSV written to: {out_csv}\n")


if __name__ == "__main__":
    main()