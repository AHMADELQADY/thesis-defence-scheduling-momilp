
#src/instance_generator/config.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class InstanceSize:
    n_i: int
    n_j: int
    n_t: int = 3
    n_k: int = 15
    n_ell: int = 16
    n_p: int = 3
    n_q: int = 15
    d: int = 2

@dataclass(frozen=True)
class PaperKnobs:
    # matches the “Data” columns in Tables C.1–C.3
    fixed_roles: int                 # e_ijt column (1 or 2)
    p_lik0: float                    # e.g., 0.78 / 0.82 / 0.86  (lik unavailability)
    p_mkp0: float                    # e.g., 0.8 / 0.86          (room unavailability)
    p_v_21: float                    # P(v=[2,1])   -> 0.2 or 0.3
    p_h_21: float                    # P(h=[2,1])   -> 0.2 or 0.3
    riq_per_member: int = 3          # “riq” column in tables
    tiq_per_defence: int = 3         # “tiq” column in tables

@dataclass(frozen=True)
class RunConfig:
    seed: int
    steps_per_obj: int = 5
    time_limit_stage1: float | None = None
    time_limit_ideal: float | None = None
    time_limit_eps: float | None = None