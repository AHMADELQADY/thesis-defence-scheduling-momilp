# src/model/constraints.py
from __future__ import annotations

from gurobipy import Model, quicksum

from src.common.symbols import Indices
from src.common.parameters import Parameters
from src.model.variables import Vars


# =============================================================================
# A.1 - A.11 (Scheduling complete committees)
# =============================================================================

def add_A1_complete_committee_definition(m: Model, idx: Indices, var: Vars) -> None:
    """
    (A.1) Complete committee definition:
        sum_{i=1..n_i} x[i,j,t,k,ell,p] = y_def[j,k,ell,p]
    for all j,t,k,ell,p.
    """
    idx.validate()
    m.addConstrs(
        (
            quicksum(var.x[i, j, t, k, ell, p] for i in idx.I)
            == var.y_def[j, k, ell, p]
            for j in idx.J for t in idx.T for k in idx.K for ell in idx.L for p in idx.P
        ),
        name="A1_complete_committee_definition",
    )


def add_A2_single_committee_assignment(m: Model, idx: Indices, var: Vars) -> None:
    """
    (A.2) Single committee assignment:
        sum_{k,ell,p} y_def[j,k,ell,p] <= 1
    for all j.
    """
    idx.validate()
    m.addConstrs(
        (
            quicksum(var.y_def[j, k, ell, p] for k in idx.K for ell in idx.L for p in idx.P) <= 1
            for j in idx.J
        ),
        name="A2_single_committee_assignment",
    )


def add_A3_total_scheduled_equals_g_var(m: Model, idx: Indices, var: Vars, g_var) -> None:
    """
    (A.3) Total scheduled defences equals stage-1 variable g:
        sum_{j,k,ell,p} y_def[j,k,ell,p] = g
    """
    idx.validate()
    m.addConstr(
        quicksum(var.y_def[j, k, ell, p] for j in idx.J for k in idx.K for ell in idx.L for p in idx.P) == g_var,
        name="A3_total_scheduled_equals_g",
    )


def add_A3_total_scheduled_equals_g_value(m: Model, idx: Indices, var: Vars, g_value: int) -> None:
    """
    (A.3) Fixed-g variant:
        sum_{j,k,ell,p} y_def[j,k,ell,p] = g_value
    """
    idx.validate()
    if not isinstance(g_value, int):
        raise TypeError(f"g_value must be int, got {type(g_value)}")
    if g_value < 0 or g_value > idx.n_j:
        raise ValueError(f"g_value must be in [0, n_j], got {g_value}")

    m.addConstr(
        quicksum(var.y_def[j, k, ell, p] for j in idx.J for k in idx.K for ell in idx.L for p in idx.P) == g_value,
        name="A3_total_scheduled_equals_g_fixed",
    )


def add_A4_committee_member_eligibility(m: Model, idx: Indices, par: Parameters, var: Vars) -> None:
    """
    (A.4) Member eligibility:
        sum_{k,ell,p} x[i,j,t,k,ell,p] <= e[i][j][t]
    for all i,j,t.
    """
    idx.validate(); par.validate(idx)
    m.addConstrs(
        (
            quicksum(var.x[i, j, t, k, ell, p] for k in idx.K for ell in idx.L for p in idx.P)
            <= par.e[i][j][t]
            for i in idx.I for j in idx.J for t in idx.T
        ),
        name="A4_committee_member_eligibility",
    )


def add_A5_member_max_committees(m: Model, idx: Indices, par: Parameters, var: Vars) -> None:
    """
    (A.5) Max number of committees per member:
        sum_{j,t,k,ell,p} x[i,j,t,k,ell,p] <= c[i]
    for all i.
    """
    idx.validate(); par.validate(idx)
    m.addConstrs(
        (
            quicksum(
                var.x[i, j, t, k, ell, p]
                for j in idx.J for t in idx.T for k in idx.K for ell in idx.L for p in idx.P
            ) <= par.c[i]
            for i in idx.I
        ),
        name="A5_member_max_committees",
    )


def add_A6_member_time_slot_availability(m: Model, idx: Indices, par: Parameters, var: Vars) -> None:
    """
    (A.6) Member availability per slot:
        sum_{j,t,p} x[i,j,t,k,ell,p] <= 1{ l[i][k][ell] >= 1 }
    for all i,k,ell.
    """
    idx.validate(); par.validate(idx)
    m.addConstrs(
        (
            quicksum(var.x[i, j, t, k, ell, p] for j in idx.J for t in idx.T for p in idx.P)
            <= (1 if par.l[i][k][ell] >= 1 else 0)
            for i in idx.I for k in idx.K for ell in idx.L
        ),
        name="A6_member_time_slot_availability",
    )


def add_A7_member_no_overlap_duration(m: Model, idx: Indices, par: Parameters, var: Vars) -> None:
    """
    (A.7) Member non-overlap for defence duration d:
    For each i,k and each start slot s, at most one assignment in [s, s+d-1].
    """
    idx.validate(); par.validate(idx)
    d = par.d
    if d > idx.n_ell:
        return

    start_ells = range(1, idx.n_ell - d + 2)
    m.addConstrs(
        (
            quicksum(
                var.x[i, j, t, k, ell, p]
                for j in idx.J
                for t in idx.T
                for ell in range(start_ell, start_ell + d)
                for p in idx.P
            ) <= 1
            for i in idx.I for k in idx.K for start_ell in start_ells
        ),
        name="A7_member_no_overlap_duration",
    )


def add_A8_room_time_slot_availability(m: Model, idx: Indices, par: Parameters, var: Vars) -> None:
    """
    (A.8) Room availability:
        sum_{j} y_def[j,k,ell,p] <= m[k][ell][p]
    for all k,ell,p.
    """
    idx.validate(); par.validate(idx)
    m.addConstrs(
        (
            quicksum(var.y_def[j, k, ell, p] for j in idx.J) <= par.m[k][ell][p]
            for k in idx.K for ell in idx.L for p in idx.P
        ),
        name="A8_room_time_slot_availability",
    )


def add_A9_room_no_overlap_duration(m: Model, idx: Indices, par: Parameters, var: Vars) -> None:
    """
    (A.9) Room non-overlap for defence duration d:
    For each k,p and each start slot s, at most one defence in [s, s+d-1].
    """
    idx.validate(); par.validate(idx)
    d = par.d
    if d > idx.n_ell:
        return

    start_ells = range(1, idx.n_ell - d + 2)
    m.addConstrs(
        (
            quicksum(var.y_def[j, k, ell, p] for j in idx.J for ell in range(start_ell, start_ell + d)) <= 1
            for k in idx.K for p in idx.P for start_ell in start_ells
        ),
        name="A9_room_no_overlap_duration",
    )


def add_A10_subject_coverage_count(m: Model, idx: Indices, par: Parameters, var: Vars) -> None:
    """
    (A.10) Subject coverage count (paper's selector trick):
        sum_{i0 in I0} i0 * s[i0,j,q]
          =
        sum_{i,t,k,ell,p} r[i,q] * tbar[j,q] * x[i,j,t,k,ell,p]
    for all j,q.
    """
    idx.validate(); par.validate(idx)
    m.addConstrs(
        (
            quicksum(i0 * var.s[i0, j, q] for i0 in idx.I0)
            ==
            quicksum(
                par.r[i][q] * par.tbar[j][q] * var.x[i, j, t, k, ell, p]
                for i in idx.I for t in idx.T for k in idx.K for ell in idx.L for p in idx.P
            )
            for j in idx.J for q in idx.Q
        ),
        name="A10_subject_coverage_count",
    )


def add_A11_subject_coverage_uniqueness(m: Model, idx: Indices, var: Vars) -> None:
    """
    (A.11) Uniqueness of s selector:
        sum_{i0 in I0} s[i0,j,q] = 1
    for all j,q.
    """
    idx.validate()
    m.addConstrs(
        (quicksum(var.s[i0, j, q] for i0 in idx.I0) == 1 for j in idx.J for q in idx.Q),
        name="A11_subject_coverage_uniqueness",
    )


# =============================================================================
# A.12 - A.16 Compactness
# =============================================================================

def add_A12_define_y_mem(m: Model, idx: Indices, var: Vars) -> None:
    """
    (A.12) Definition of y_mem (paper: \bar{y}_{ikℓp})
        y_mem[i,k,ell,p] = sum_{j,t} x[i,j,t,k,ell,p]

    Important: y_mem is a PRESENCE indicator (member i is assigned at slot (k,ell,p)).
    The RHS can never exceed 1 in a correct model, because a member cannot be in 2 roles/defences
    in the same slot (k,ell,p).
    """
    idx.validate()
    m.addConstrs(
        (
            var.y_mem[i, k, ell, p]
            == quicksum(var.x[i, j, t, k, ell, p] for j in idx.J for t in idx.T)
            for i in idx.I for k in idx.K for ell in idx.L for p in idx.P
        ),
        name="A12_define_y_mem",
    )


def add_A13_A16_compactness(m: Model, idx: Indices, par: Parameters, var: Vars) -> None:
    """
    (A.13)-(A.16) Compactness big-M formulation for the product:
        sbar_comp[i,k,ell] = A * B

    where (paper):
        A = sum_{p=1..n_p} y_mem[i,k,ell,p]     (binary in practice: presence at time ell in any room)
        B = sum_{lbar=0..b[i]} sum_{p=1..n_p} v[i][lbar] * y_mem[i,k,ell-d-lbar,p]

    REAL FIX #1: Big-M must bound the MAXIMUM possible value of B.
      - B is a SUM over lbar (and rooms).
      - Since for each previous time slot (k, ell-d-lbar) the member can be present in at most ONE room,
        we have:
            sum_p y_mem[i,k,ell-d-lbar,p] <= 1
        so:
            B <= sum_{lbar=0..b[i]} v[i][lbar]
      - Therefore use:
            M_vi = sum(v[i][0..b_i])
        NOT max(v[i]) (max is too small and can break the linearization).

    Paper index note:
      - A.15 and A.16 are stated only for ell = d..n_ell.
      - A.13 and A.14 apply for ell = 1..n_ell.
    """
    idx.validate(); par.validate(idx)

    d = par.d

    for i in idx.I:
        b_i = par.b[i]  # 0..d-1

        # Big-M for compactness: sum_{lbar=0..b_i} v[i][lbar]
        M_vi = 0
        if par.v[i]:
            M_vi = sum(par.v[i][lbar] for lbar in range(0, b_i + 1))

        for k in idx.K:
            for ell in idx.L:
                A = quicksum(var.y_mem[i, k, ell, p] for p in idx.P)

                # (A.13) sbar >= 0  (usually already by variable lower bound, but we keep it explicit we remove it to make the model faster)
                #m.addConstr(
                   # var.sbar_comp[i, k, ell] >= 0,
                   # name=f"A13_comp_nonneg_i{i}_k{k}_ell{ell}",
                #)

                # (A.14) sbar <= M_vi * A   (A.14 applies for ALL ell)
                m.addConstr(
                    var.sbar_comp[i, k, ell] <= M_vi * A,
                    name=f"A14_comp_ub1_i{i}_k{k}_ell{ell}",
                )

                # A.15 and A.16: only for ell = d..n_ell (paper)
                if ell < d:
                    # Paper does not explicitly add A.15/A.16 here.
                    # We keep it faithful: just don't add them for ell < d.
                    continue

                # Build B only with valid previous slots (ell - d - lbar >= 1)
                B = quicksum(
                    par.v[i][lbar] * quicksum(var.y_mem[i, k, ell - d - lbar, p] for p in idx.P)
                    for lbar in range(0, b_i + 1)
                    if (ell - d - lbar) >= 1
                )

                # (A.15) sbar <= B
                m.addConstr(
                    var.sbar_comp[i, k, ell] <= B,
                    name=f"A15_comp_ub2_i{i}_k{k}_ell{ell}",
                )

                # (A.16) sbar >= B - M_vi * (1 - A)
                m.addConstr(
                    var.sbar_comp[i, k, ell] >= B - M_vi * (1 - A),
                    name=f"A16_comp_lb_i{i}_k{k}_ell{ell}",
                )


# =============================================================================
# A.17 - A.22 Workload / Days
# =============================================================================

def add_A17_workload_definition(m: Model, idx: Indices, par: Parameters, var: Vars) -> None:
    """
    (A.17) Workload definition (one-hot selector w):
        sum_{jc=0..c[i]} jc * w[i,jc]
          =
        sum_{j,t,k,ell,p} x[i,j,t,k,ell,p]
    """
    idx.validate(); par.validate(idx)

    for i in idx.I:
        jcounts = range(0, par.c[i] + 1)

        lhs = quicksum(jc * var.w[i, jc] for jc in jcounts)
        rhs = quicksum(
            var.x[i, j, t, k, ell, p]
            for j in idx.J for t in idx.T for k in idx.K for ell in idx.L for p in idx.P
        )

        m.addConstr(lhs == rhs, name=f"A17_workload_def_i{i}")

        # Optional: forbid selecting jc > c[i] (since w may be defined over a larger range)
        for jc in range(par.c[i] + 1, idx.n_j + 1):
            m.addConstr(var.w[i, jc] == 0, name=f"A17_forbid_w_i{i}_jc{jc}")


def add_A18_workload_uniqueness(m: Model, idx: Indices, par: Parameters, var: Vars) -> None:
    """
    (A.18) Uniqueness of workload selector:
        sum_{jc=0..c[i]} w[i,jc] = 1
    """
    idx.validate(); par.validate(idx)

    for i in idx.I:
        jcounts = range(0, par.c[i] + 1)
        m.addConstr(quicksum(var.w[i, jc] for jc in jcounts) == 1, name=f"A18_workload_unique_i{i}")


def add_A19_committee_days_definition(m: Model, idx: Indices, par: Parameters, var: Vars) -> None:
    """
    (A.19) Definition of yhat (one-hot per day):
        sum_{jc=0..c[i]} jc * yhat[i,jc,k]
          =
        sum_{j,t,ell,p} x[i,j,t,k,ell,p]
    for all i,k.
    """
    idx.validate(); par.validate(idx)

    for i in idx.I:
        jcounts = range(0, par.c[i] + 1)

        for k in idx.K:
            lhs = quicksum(jc * var.yhat[i, jc, k] for jc in jcounts)
            rhs = quicksum(
                var.x[i, j, t, k, ell, p]
                for j in idx.J for t in idx.T for ell in idx.L for p in idx.P
            )

            m.addConstr(lhs == rhs, name=f"A19_yhat_def_i{i}_k{k}")

            # Optional: forbid selecting jc > c[i]
            for jc in range(par.c[i] + 1, idx.n_j + 1):
                m.addConstr(var.yhat[i, jc, k] == 0, name=f"A19_forbid_yhat_i{i}_k{k}_jc{jc}")


def add_A20_committee_days_uniqueness(m: Model, idx: Indices, par: Parameters, var: Vars) -> None:
    """
    (A.20) Uniqueness of yhat selector for each (i,k):
        sum_{jc=0..c[i]} yhat[i,jc,k] = 1
    """
    idx.validate(); par.validate(idx)

    for i in idx.I:
        jcounts = range(0, par.c[i] + 1)
        for k in idx.K:
            m.addConstr(quicksum(var.yhat[i, jc, k] for jc in jcounts) == 1,
                        name=f"A20_yhat_unique_i{i}_k{k}")


def add_A21_days_count_definition(m: Model, idx: Indices, par: Parameters, var: Vars) -> None:
    """
    (A.21) Number of days with >=1 committees (one-hot selector wbar):
        sum_{kc=0..n_k} kc * wbar[i,kc]
          =
        sum_{k in K} sum_{jc=1..c[i]} yhat[i,jc,k]
    for all i.
    """
    idx.validate(); par.validate(idx)

    for i in idx.I:
        lhs = quicksum(kc * var.wbar[i, kc] for kc in idx.K0)
        rhs = quicksum(var.yhat[i, jc, k] for jc in range(1, par.c[i] + 1) for k in idx.K)

        m.addConstr(lhs == rhs, name=f"A21_wbar_def_i{i}")


def add_A22_days_count_uniqueness(m: Model, idx: Indices, var: Vars) -> None:
    """
    (A.22) Uniqueness of wbar selector:
        sum_{kc=0..n_k} wbar[i,kc] = 1
    """
    idx.validate()
    m.addConstrs(
        (quicksum(var.wbar[i, kc] for kc in idx.K0) == 1 for i in idx.I),
        name="A22_wbar_unique",
    )


# =============================================================================
# A.23 - A.26 Room-change penalty
# =============================================================================

def add_A23_A26_room_change_penalty(m: Model, idx: Indices, par: Parameters, var: Vars) -> None:
    """
    (A.23)-(A.26) Room-change penalty big-M formulation for the product:
        shat_roomchg[i,k,ell,p] = y_now * prev_sum

    where (paper):
        y_now    = y_mem[i,k,ell,p]  (binary, member present now in room p)
        prev_sum = sum_{lbar=0..a[i]} sum_{pbar != p} h[i][lbar] * y_mem[i,k,ell-d-lbar,pbar]
        (paper notation uses \\hat{ell} = ell - d - lbar)

    REAL FIX #2: Big-M must bound prev_sum.
      - prev_sum is a SUM over lbar.
      - For each previous time slot (k, ell-d-lbar), member can be in at most ONE room:
            sum_{pbar != p} y_mem[...] <= 1
        so:
            prev_sum <= sum_{lbar=0..a[i]} h[i][lbar]
      - Therefore use:
            M_hi = sum(h[i][0..a_i])
        NOT max(h[i]) (max can be too small and break the linearization).

    REAL FIX #3 (paper-faithfulness):
      - A.24 applies for ell = 1..n_ell
      - A.25 and A.26 apply only for ell = d..n_ell
        (so for ell < d, do NOT force shat=0; just skip A25/A26).

    Also: A.26 must be a LOWER bound (>=) for the correct product linearization.
          The paper has a direction typo there.
    """
    idx.validate(); par.validate(idx)

    d = par.d

    for i in idx.I:
        a_i = par.a[i]  # 0..d-1

        # Big-M for room-change: sum_{lbar=0..a_i} h[i][lbar]
        M_hi = 0
        if par.h[i]:
            M_hi = sum(par.h[i][lbar] for lbar in range(0, a_i + 1))

        for k in idx.K:
            for ell in idx.L:
                for p in idx.P:
                    y_now = var.y_mem[i, k, ell, p]

                    # (A.23) shat >= 0  (usually already by variable lower bound, but we keep it explicit we remove it to make it faster)
                    #m.addConstr(
                       # var.shat_roomchg[i, k, ell, p] >= 0,
                        #name=f"A23_roomchg_nonneg_i{i}_k{k}_ell{ell}_p{p}",
                   # )

                    # (A.24) shat <= n_hi * y_now   (A.24 applies for ALL ell)
                    m.addConstr(
                        var.shat_roomchg[i, k, ell, p] <= M_hi * y_now,
                        name=f"A24_roomchg_ub1_i{i}_k{k}_ell{ell}_p{p}",
                    )

                    # A.25 and A.26: only for ell = d..n_ell (paper)
                    if ell < d:
                        continue

                    # Build prev_sum with valid previous slots (ell - d - lbar >= 1)
                    prev_sum = quicksum(
                        par.h[i][lbar] * quicksum(
                            var.y_mem[i, k, ell - d - lbar, pbar]
                            for pbar in idx.P if pbar != p
                        )
                        for lbar in range(0, a_i + 1)
                        if (ell - d - lbar) >= 1
                    )

                    # (A.25) shat <= prev_sum
                    m.addConstr(
                        var.shat_roomchg[i, k, ell, p] <= prev_sum,
                        name=f"A25_roomchg_ub2_i{i}_k{k}_ell{ell}_p{p}",
                    )

                    # (A.26) shat >= prev_sum - M_hi*(1 - y_now)   ✅ LOWER bound
                    m.addConstr(
                        var.shat_roomchg[i, k, ell, p] >= prev_sum - M_hi * (1 - y_now),
                        name=f"A26_roomchg_lb_i{i}_k{k}_ell{ell}_p{p}",
                    )

