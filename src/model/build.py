# src/model/build.py

"""
We need two builders:
	•	Stage 1 (Eq.45): maximize total scheduled defences (sum of y_def)
	•	Stage 2 base model P: same constraints, but with A.3 fixed to g
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from gurobipy import Model, GRB, quicksum

from src.common.symbols import Indices
from src.common.parameters import Parameters
from src.model.variables import Vars, build_variables
from src.model import constraints as C
from src.model.zexpr import build_z_defs


@dataclass
class BuiltModel:
    m: Model
    var: Vars


def _add_all_constraints_with_fixed_g(m: Model, idx: Indices, par: Parameters, var: Vars, g_value: int) -> None:
    # A.1-A.2
    C.add_A1_complete_committee_definition(m, idx, var)
    C.add_A2_single_committee_assignment(m, idx, var)

    # A.3 fixed to g
    C.add_A3_total_scheduled_equals_g_value(m, idx, var, g_value)

    # A.4-A.11
    C.add_A4_committee_member_eligibility(m, idx, par, var)
    C.add_A5_member_max_committees(m, idx, par, var)
    C.add_A6_member_time_slot_availability(m, idx, par, var)
    C.add_A7_member_no_overlap_duration(m, idx, par, var)
    C.add_A8_room_time_slot_availability(m, idx, par, var)
    C.add_A9_room_no_overlap_duration(m, idx, par, var)
    C.add_A10_subject_coverage_count(m, idx, par, var)
    C.add_A11_subject_coverage_uniqueness(m, idx, var)

    # A.12-A.16
    C.add_A12_define_y_mem(m, idx, var)
    C.add_A13_A16_compactness(m, idx, par, var)

    # A.17-A.22
    C.add_A17_workload_definition(m, idx, par, var)
    C.add_A18_workload_uniqueness(m, idx, par, var)
    C.add_A19_committee_days_definition(m, idx, par, var)
    C.add_A20_committee_days_uniqueness(m, idx, par, var)
    C.add_A21_days_count_definition(m, idx, par, var)
    C.add_A22_days_count_uniqueness(m, idx, var)

    # A.23-A.26
    C.add_A23_A26_room_change_penalty(m, idx, par, var)


def build_stage2_base(idx: Indices, par: Parameters, g_value: int, name: str = "P") -> BuiltModel:
    """
    Builds the feasible region X with fixed g (Stage 2 problems).
    No objective set here.
    """
    idx.validate(); par.validate(idx)
    m = Model(name)
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    try:
        m.Params.ConcurrentMIP = 1
    except Exception:
        pass

    var = build_variables(m, idx)
    _add_all_constraints_with_fixed_g(m, idx, par, var, g_value)

    return BuiltModel(m=m, var=var)


def build_stage1_g(idx: Indices, par: Parameters, name: str = "Stage1_g") -> BuiltModel:
    idx.validate(); par.validate(idx)
    m = Model(name)
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    try:
        m.Params.ConcurrentMIP = 1
    except Exception:
        pass

    var = build_variables(m, idx)

    # Keep only constraints needed for feasibility of scheduling g*
    C.add_A1_complete_committee_definition(m, idx, var)
    C.add_A2_single_committee_assignment(m, idx, var)

    C.add_A4_committee_member_eligibility(m, idx, par, var)
    C.add_A5_member_max_committees(m, idx, par, var)
    C.add_A6_member_time_slot_availability(m, idx, par, var)
    C.add_A7_member_no_overlap_duration(m, idx, par, var)
    C.add_A8_room_time_slot_availability(m, idx, par, var)
    C.add_A9_room_no_overlap_duration(m, idx, par, var)

    # Objective: maximize total scheduled defenses
    m.setObjective(
        quicksum(var.y_def[j, k, ell, p] for j in idx.J for k in idx.K for ell in idx.L for p in idx.P),
        GRB.MAXIMIZE
    )

    return BuiltModel(m=m, var=var)