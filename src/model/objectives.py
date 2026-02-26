# src/model/objectives.py
from __future__ import annotations

from gurobipy import Model, quicksum, GRB

from src.common.symbols import Indices
from src.common.parameters import Parameters
from src.model.variables import Vars


def set_objective_27_minimise_workloads(m: Model, idx: Indices, par: Parameters, var: Vars) -> None:
    """
    Paper Eq. (27): Minimise workloads

        min z1(w) = sum_{i=1..n_i} sum_{j=1..n_j} u_i * j^2 * w[i,j]

    Interpretation:
    - w[i,j] is a one-hot selector saying: "member i is assigned exactly j defences".
    - Penalising j^2 promotes fairness (large workloads get punished more).

    REQUIRED CONSTRAINTS (must already be added):
    - (A.17) ties w[i,j] to the number of x assignments for member i
    - (A.18) enforces w one-hot per member i

    Indexing sync:
    - Our code defines w over J0 = {0..n_j}. Including j=0 is harmless because 0^2 = 0.
    """
    idx.validate()
    par.validate(idx)

    obj = quicksum(
        par.u[i] * (j * j) * var.w[i, j]
        for i in idx.I
        for j in idx.J0   # include 0 safely; matches variable domain
    )

    m.setObjective(obj, GRB.MINIMIZE)


def set_objective_28_maximise_research_subject_coverage(m: Model, idx: Indices, par: Parameters, var: Vars) -> None:
    """
    Paper Eq. (28): Maximise research subject coverage

        max z2(s) = ( sum_{j,q} tbar[j,q] )^{-1} * sum_{i=1..n_i} sum_{j=1..n_j} sum_{q=1..n_q} s[i,j,q]

    Interpretation:
    - s is a one-hot encoding of how many committee members cover subject q for defence j.
    - Summing s over i=1..n_i counts whether coverage is >=1 (do NOT include i=0).

    REQUIRED CONSTRAINTS:
    - (A.10) subject coverage count
    - (A.11) uniqueness of s selector

    Notes:
	•	I used Python sum() for denom because it’s a pure constant (faster than building a Gurobi expression).
	•	I handled the edge case denom=0 (no subjects in the instance) to avoid division-by-zero.
    """
    idx.validate()
    par.validate(idx)

    # Denominator is a positive constant (data only). Including it is optional (doesn't change argmax).
    denom = sum(par.tbar[j][q] for j in idx.J for q in idx.Q)
    if denom <= 0:
        # No subjects at all => objective is meaningless; safest is maximize 0.
        m.setObjective(0.0, GRB.MAXIMIZE)
        return

    numer = quicksum(var.s[i, j, q] for i in idx.I for j in idx.J for q in idx.Q)

    #m.setObjective((1.0 / denom) * numer, GRB.MAXIMIZE)
    m.setObjective(numer, GRB.MAXIMIZE)


def set_objective_29_maximise_committee_member_suitability(m: Model, idx: Indices, par: Parameters, var: Vars) -> None:
    """
    Paper Eq. (29): Maximise committee member suitability

        max z3(x) =
            sum_{i=1..n_i} sum_{q=1..n_q} sum_{j=1..n_j} sum_{t=1..n_t}
            sum_{k=1..n_k} sum_{ell=1..n_ell} sum_{p=1..n_p}
                r[i,q] * tbar[j,q] * x[i,j,t,k,ell,p]

    Interpretation:
    - Rewards assigning members to defences where their expertise (r) matches the defence subjects (tbar).
    - Summing over q means an assignment can earn multiple points if the member covers multiple subjects of that defence.

    REQUIRED CONSTRAINTS:
    - No special constraints for linearity (r and tbar are constants).
    - Feasibility constraints (A.1)-(A.9) should already exist so x represents valid assignments.
    """
    idx.validate()
    par.validate(idx)

    obj = quicksum(
        par.r[i][q] * par.tbar[j][q] * var.x[i, j, t, k, ell, p]
        for i in idx.I
        for q in idx.Q
        for j in idx.J
        for t in idx.T
        for k in idx.K
        for ell in idx.L
        for p in idx.P
    )

    m.setObjective(obj, GRB.MAXIMIZE)


def set_objective_30_minimise_non_consecutive_assignments(m: Model, idx: Indices, par: Parameters, var: Vars) -> None:
    """
    Paper Eq. (30): Minimise non-consecutive assignments

        min z4(w, sbar) =
            sum_{i=1..n_i} sum_{j=1..n_j} u[i] * n_vi * (j-1) * w[i,j]
            - sum_{i=1..n_i} sum_{k=1..n_k} sum_{ell=1..n_ell} u[i] * sbar_comp[i,k,ell]

    Where we compute n_vi as:
        n_vi = sum_{lbar=0..b[i]} v[i][lbar]
    """
    idx.validate()
    par.validate(idx)

    # n_vi per i (max possible compactness contribution per "consecutive opportunity")
    nvi = {
        i: sum(par.v[i][lbar] for lbar in range(0, par.b[i] + 1))
        for i in idx.I
    }

    term_max_potential = quicksum(
        par.u[i] * nvi[i] * (j - 1) * var.w[i, j]
        for i in idx.I
        for j in idx.J  # MUST start at 1 (avoid j=0 because (j-1) becomes negative)
    )

    term_achieved = quicksum(
        par.u[i] * var.sbar_comp[i, k, ell]
        for i in idx.I
        for k in idx.K
        for ell in idx.L
    )

    m.setObjective(term_max_potential - term_achieved, GRB.MINIMIZE)


def set_objective_31_minimise_time_slot_preference_nonsatisfaction(
    m: Model, idx: Indices, par: Parameters, var: Vars
) -> None:
    """
    Paper Eq. (31): Minimise the non-satisfaction of time slot preferences

        min z5(x) =
            sum_{i=1..n_i} sum_{k=1..n_k} sum_{ell=1..n_ell}
            sum_{j=1..n_j} sum_{t=1..n_t} sum_{p=1..n_p}
                u[i] * ( l[i][k][ell] - 1 ) * x[i,j,t,k,ell,p]

    What it penalizes:
    - l=1  => 0 penalty (best slot)
    - l>1  => increasing penalty
    - l=0  => coefficient becomes negative, so you MUST enforce infeasibility with (A.6),
             otherwise the solver would exploit it.

    REQUIRED CONSTRAINTS for correctness:
    - (A.6) Member time slot availability (prevents l=0 assignments)
    - plus usual feasibility constraints (A.1)-(A.9) so x represents a valid schedule.
    """
    idx.validate()
    par.validate(idx)

    obj = quicksum(
        par.u[i] * (par.l[i][k][ell] - 1) * var.x[i, j, t, k, ell, p]
        for i in idx.I
        for k in idx.K
        for ell in idx.L
        for j in idx.J
        for t in idx.T
        for p in idx.P
    )

    m.setObjective(obj, GRB.MINIMIZE)

def set_objective_32_minimise_committee_days(m: Model, idx: Indices, par: Parameters, var: Vars) -> None:
    """
    Paper Eq. (32): Minimise committee days

        min z6(wbar) = sum_{i=1..n_i} sum_{k=1..n_k} u[i] * k^2 * wbar[i,k]

    Interpretation:
    - wbar[i,k] is one-hot: 1 if member i is assigned on exactly k days.
    - k^2 penalizes larger numbers of days more strongly (fairness).
    - u[i] is the member weight.

    REQUIRED CONSTRAINTS:
    - (A.19) and (A.20): define yhat per day (one-hot daily committee count)
    - (A.21) and (A.22): define + enforce one-hot wbar (days-used selector)

    Indexing sync:
    - Our wbar is defined on K0 = {0..n_k}. Including k=0 is safe because 0^2 = 0.
    """
    idx.validate()
    par.validate(idx)

    obj = quicksum(
        par.u[i] * (k * k) * var.wbar[i, k]
        for i in idx.I
        for k in idx.K0
    )

    m.setObjective(obj, GRB.MINIMIZE)


def set_objective_33_minimise_room_changes(m: Model, idx: Indices, par: Parameters, var: Vars) -> None:
    """
    Paper Eq. (33): Minimise room changes

    Paper text defines the penalty variable as:
        shat_roomchg[i,k,ell,p] = \\hat{s}_{ik\\ell p}

    The paper's written sum uses \\hat{s}_{ij}, but this is best interpreted as
    an aggregation of room-change penalties across time/rooms.

    We therefore minimize the total weighted room-change penalty:
        min z7 = sum_{i in I} sum_{k in K} sum_{ell in L} sum_{p in P} u[i] * shat_roomchg[i,k,ell,p]

    REQUIRED CONSTRAINTS:
    - (A.12): definition of y_mem from x
    - (A.23)-(A.26): linearization that defines shat_roomchg from y_mem and parameter a/h
    - plus feasibility constraints (A.1)-(A.9) so x is a valid schedule.
    """
    idx.validate()
    par.validate(idx)

    obj = quicksum(
        par.u[i] * var.shat_roomchg[i, k, ell, p]
        for i in idx.I
        for k in idx.K
        for ell in idx.L
        for p in idx.P
    )

    m.setObjective(obj, GRB.MINIMIZE)