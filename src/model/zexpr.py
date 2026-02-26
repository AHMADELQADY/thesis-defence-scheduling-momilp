# src/model/zexpr.py
"""
3.1 src/model/zexpr.py (ONE place that defines z1..z7 consistently)

Important: Algorithm section assumes maximization of all objectives.
But your paper has mixed min/max (27,30,31,32,33 are mins).

So we define a maximization form vector:
    • if paper says min, we use - (that expression) as the “z” inside Algorithm 1/5.

NOTE (bug fix):
    Do NOT write "+expr" for gurobipy LinExpr (unary + is not supported and crashes).
    Just use "expr".
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

from gurobipy import LinExpr, quicksum

from src.common.symbols import Indices
from src.common.parameters import Parameters
from src.model.variables import Vars


@dataclass(frozen=True)
class ZDef:
    name: str
    expr: LinExpr          # ALWAYS "maximize-form"
    paper_sense: str       # "min" or "max" (just for reporting)


def build_z_defs(idx: Indices, par: Parameters, var: Vars) -> Dict[int, ZDef]:
    """
    Returns z_1..z_7 as in paper Eq.(27)-(33),
    but converted to a UNIFORM 'maximize-form' vector for Section 3 algorithms.

    maximize-form rule:
      - if paper objective is MAX: z = expr
      - if paper objective is MIN: z = -expr
    """
    idx.validate()
    par.validate(idx)

    z: Dict[int, ZDef] = {}

    # ------------------------------------------------------------------
    # (27) min workloads
    # ------------------------------------------------------------------
    expr27 = quicksum(
        par.u[i] * (j * j) * var.w[i, j]
        for i in idx.I
        for j in idx.J0
    )
    z[1] = ZDef("z1_workload_fairness", -expr27, "min")

    # ------------------------------------------------------------------
    # (28) max subject coverage
    # ------------------------------------------------------------------
    denom = sum(par.tbar[j][q] for j in idx.J for q in idx.Q)
    if denom <= 0:
        expr28 = LinExpr(0.0)
    else:
        expr28 = (1.0 / denom) * quicksum(
            var.s[i, j, q]
            for i in idx.I
            for j in idx.J
            for q in idx.Q
        )

    # IMPORTANT: NO unary plus on LinExpr
    z[2] = ZDef("z2_subject_coverage", expr28, "max")

    # ------------------------------------------------------------------
    # (29) max suitability
    # ------------------------------------------------------------------
    expr29 = quicksum(
        par.r[i][q] * par.tbar[j][q] * var.x[i, j, t, k, ell, p]
        for i in idx.I
        for q in idx.Q
        for j in idx.J
        for t in idx.T
        for k in idx.K
        for ell in idx.L
        for p in idx.P
    )

    # IMPORTANT: NO unary plus on LinExpr
    z[3] = ZDef("z3_suitability", expr29, "max")

    # ------------------------------------------------------------------
    # (30) min non-consecutive assignments
    # ------------------------------------------------------------------
    nvi = {
        i: sum(par.v[i][lbar] for lbar in range(0, par.b[i] + 1))
        for i in idx.I
    }

    term_max_potential = quicksum(
        par.u[i] * nvi[i] * (j - 1) * var.w[i, j]
        for i in idx.I
        for j in idx.J  # start at 1 to avoid (j-1) negative term for j=0
    )

    term_achieved = quicksum(
        par.u[i] * var.sbar_comp[i, k, ell]
        for i in idx.I
        for k in idx.K
        for ell in idx.L
    )

    expr30 = term_max_potential - term_achieved
    z[4] = ZDef("z4_non_consecutive", -expr30, "min")

    # ------------------------------------------------------------------
    # (31) min time-slot preference non-satisfaction
    # ------------------------------------------------------------------
    expr31 = quicksum(
        par.u[i] * (par.l[i][k][ell] - 1) * var.x[i, j, t, k, ell, p]
        for i in idx.I
        for k in idx.K
        for ell in idx.L
        for j in idx.J
        for t in idx.T
        for p in idx.P
    )
    z[5] = ZDef("z5_timeslot_preference", -expr31, "min")

    # ------------------------------------------------------------------
    # (32) min committee days
    # ------------------------------------------------------------------
    expr32 = quicksum(
        par.u[i] * (k * k) * var.wbar[i, k]
        for i in idx.I
        for k in idx.K0
    )
    z[6] = ZDef("z6_committee_days", -expr32, "min")

    # ------------------------------------------------------------------
    # (33) min room changes
    # ------------------------------------------------------------------
    expr33 = quicksum(
        par.u[i] * var.shat_roomchg[i, k, ell, p]
        for i in idx.I
        for k in idx.K
        for ell in idx.L
        for p in idx.P
    )
    z[7] = ZDef("z7_room_changes", -expr33, "min")

    return z