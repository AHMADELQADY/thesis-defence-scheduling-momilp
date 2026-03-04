# src/experiments/schedule_export.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional


def extract_schedule_rows(idx, var, *, tol: float = 0.5) -> List[Dict[str, Any]]:
    """
    Extract a readable schedule from a solved model.

    Uses:
      - y_def[j,k,ell,p] == 1  -> defence j scheduled day k start slot ell in room p
      - x[i,j,t,k,ell,p] == 1  -> member i assigned to role t for that defence

    Returns list of rows:
      {
        "j": int, "k": int, "ell": int, "p": int,
        "roles": {t: i, ...}
      }
    """
    rows: List[Dict[str, Any]] = []

    # 1) find all scheduled defence starts
    starts: List[Tuple[int, int, int, int]] = []
    for j in idx.J:
        for k in idx.K:
            for ell in idx.L:
                for p in idx.P:
                    if var.y_def[j, k, ell, p].X > tol:
                        starts.append((j, k, ell, p))

    # Sort by (day, slot, room, defence) for nice reading
    starts.sort(key=lambda x: (x[1], x[2], x[3], x[0]))

    # 2) recover role assignments for each scheduled defence start
    for (j, k, ell, p) in starts:
        roles: Dict[int, int] = {}
        for t in idx.T:
            assigned_i = None
            for i in idx.I:
                if var.x[i, j, t, k, ell, p].X > tol:
                    assigned_i = i
                    break
            if assigned_i is not None:
                roles[t] = assigned_i

        rows.append({"j": int(j), "k": int(k), "ell": int(ell), "p": int(p), "roles": roles})

    return rows


def pretty_print_schedule(rows: List[Dict[str, Any]]) -> None:
    """
    Old flat print (kept for debugging).
    """
    if not rows:
        print("(no scheduled defences)")
        return

    for r in rows:
        j, k, ell, p = r["j"], r["k"], r["ell"], r["p"]
        roles = r.get("roles", {}) or {}
        roles_str = ", ".join([f"t{t}->i{roles[t]}" for t in sorted(roles)]) if roles else "(no roles)"
        print(f"j={j:>2}  day k={k:>2}  start ell={ell:>2}  room p={p:>2}   |   {roles_str}")


def pretty_print_timetable(rows: List[Dict[str, Any]]) -> None:
    """
    Pretty timetable view grouped like:

      Day 1:
        slot  3 room 2: defence j=... | t1->i..., t2->i..., ...
        slot  7 room 1: defence j=... | ...

    Notes:
    - Assumes y_def marks the START slot of each defence.
    - Sorting is by day, then slot, then room.
    """
    if not rows:
        print("(no scheduled defences)")
        return

    # Group by day
    by_day: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        by_day.setdefault(int(r["k"]), []).append(r)

    for k in sorted(by_day):
        print(f"Day {k}:")
        day_rows = sorted(by_day[k], key=lambda r: (int(r["ell"]), int(r["p"]), int(r["j"])))
        for r in day_rows:
            j = int(r["j"])
            ell = int(r["ell"])
            p = int(r["p"])
            roles = r.get("roles", {}) or {}
            if roles:
                roles_str = ", ".join([f"t{t}->i{roles[t]}" for t in sorted(roles)])
            else:
                roles_str = "(no roles)"
            print(f"  slot {ell:>2}  room {p}: defence j={j:<2} | {roles_str}")
        print()


def pretty_print_defence_blocks(
    rows: List[Dict[str, Any]],
    *,
    role_names: Optional[Dict[int, str]] = None,
) -> None:
    """
    Prints one nice "block" per defence, like:

      Defence: 6
      Day: 10
      Start time: slot 16
      Room: 2

      Committee:
      Role 1 → member 5
      Role 2 → member 1
      Role 3 → member 7
    """
    if not rows:
        print("(no scheduled defences)")
        return

    # sort by day, slot, room, defence
    rows_sorted = sorted(rows, key=lambda r: (int(r["k"]), int(r["ell"]), int(r["p"]), int(r["j"])))

    for r in rows_sorted:
        j = int(r["j"])
        k = int(r["k"])
        ell = int(r["ell"])
        p = int(r["p"])
        roles = r.get("roles", {}) or {}

        print(f"Defence: {j}")
        print(f"Day: {k}")
        print(f"Start time: slot {ell}")
        print(f"Room: {p}")
        print()
        print("Committee:")

        if not roles:
            print("(no roles)")
        else:
            for t in sorted(roles):
                label = role_names.get(t, f"Role {t}") if role_names else f"Role {t}"
                print(f"{label} \u2192 member {roles[t]}")

        print()  # blank line between defences