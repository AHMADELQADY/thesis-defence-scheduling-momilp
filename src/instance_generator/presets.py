#src/instance_generator/presets.py
"""
These presets mirror the “Data” columns (lik/mkp/vi/hi/fixed_roles) in Tables C.1–C.3.  ￼
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

from src.instance_generator.config import InstanceSize, PaperKnobs

SMALL = InstanceSize(n_i=25, n_j=20)
MED   = InstanceSize(n_i=38, n_j=30)
LARGE = InstanceSize(n_i=50, n_j=40)

# helper: turn (p(v=[1]),p(v=[2,1])) into p_v_21 etc.
def knobs(*, fixed_roles: int, p_lik0: float, p_mkp0: float, p_v1: float, p_h1: float) -> PaperKnobs:
    return PaperKnobs(
        fixed_roles=fixed_roles,
        p_lik0=p_lik0,
        p_mkp0=p_mkp0,
        p_v_21=1.0 - p_v1,
        p_h_21=1.0 - p_h1,
    )

# These are exactly the combinations that appear in the C tables:
# - fixed_roles: 2 for instances 1..16, 1 for 17..32 in Table C.1  [oai_citation:23‡1-s2.0-S0377221723005003-main.pdf](sediment://file_0000000068007246b2dc30a22acae3d6)
# - p(lik=0): {0.78,0.82} for fixed_roles=2; {0.82,0.86} for fixed_roles=1  [oai_citation:24‡1-s2.0-S0377221723005003-main.pdf](sediment://file_00000000ce34722fb3cf1b7105bd8f6d)
# - p(mkp=0): {0.8,0.86}  [oai_citation:25‡1-s2.0-S0377221723005003-main.pdf](sediment://file_00000000ce34722fb3cf1b7105bd8f6d)
# - p(v=[1]) and p(h=[1]) are either 0.7 or 0.8 (with complement 0.3 / 0.2)  [oai_citation:26‡1-s2.0-S0377221723005003-main.pdf](sediment://file_00000000ce34722fb3cf1b7105bd8f6d)
GRID_FIXED2 = [
    knobs(fixed_roles=2, p_lik0=0.82, p_mkp0=0.86, p_v1=0.8, p_h1=0.8),
    knobs(fixed_roles=2, p_lik0=0.82, p_mkp0=0.86, p_v1=0.7, p_h1=0.7),
    knobs(fixed_roles=2, p_lik0=0.82, p_mkp0=0.8,  p_v1=0.8, p_h1=0.8),
    knobs(fixed_roles=2, p_lik0=0.82, p_mkp0=0.8,  p_v1=0.7, p_h1=0.7),
    knobs(fixed_roles=2, p_lik0=0.78, p_mkp0=0.86, p_v1=0.8, p_h1=0.8),
    knobs(fixed_roles=2, p_lik0=0.78, p_mkp0=0.86, p_v1=0.7, p_h1=0.7),
    knobs(fixed_roles=2, p_lik0=0.78, p_mkp0=0.8,  p_v1=0.8, p_h1=0.8),
    knobs(fixed_roles=2, p_lik0=0.78, p_mkp0=0.8,  p_v1=0.7, p_h1=0.7),
]

GRID_FIXED1 = [
    knobs(fixed_roles=1, p_lik0=0.82, p_mkp0=0.86, p_v1=0.8, p_h1=0.8),
    knobs(fixed_roles=1, p_lik0=0.82, p_mkp0=0.86, p_v1=0.7, p_h1=0.7),
    knobs(fixed_roles=1, p_lik0=0.82, p_mkp0=0.8,  p_v1=0.8, p_h1=0.8),
    knobs(fixed_roles=1, p_lik0=0.82, p_mkp0=0.8,  p_v1=0.7, p_h1=0.7),
    knobs(fixed_roles=1, p_lik0=0.86, p_mkp0=0.86, p_v1=0.8, p_h1=0.8),
    knobs(fixed_roles=1, p_lik0=0.86, p_mkp0=0.86, p_v1=0.7, p_h1=0.7),
    knobs(fixed_roles=1, p_lik0=0.86, p_mkp0=0.8,  p_v1=0.8, p_h1=0.8),
    knobs(fixed_roles=1, p_lik0=0.86, p_mkp0=0.8,  p_v1=0.7, p_h1=0.7),
]