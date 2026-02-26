"""

Goal: pick E such that in Eq.(36) the perturbation
10^{-E}\sum_{j\ne i} z_j(x) < 1
for all feasible solutions. That guarantees it never changes the integer part.

We do this with a safe upper bound on \sum_{j\ne i} |z_j|.
"""


# src/common/bounds.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Tuple

from src.common.symbols import Indices
from src.common.parameters import Parameters


@dataclass(frozen=True)
class ObjBounds:
    # bounds in MAXIMIZE-FORM coordinates (the same as build_z_defs)
    lb: float
    ub: float


def objective_bounds_maxform(idx: Indices, par: Parameters) -> Dict[int, ObjBounds]:
    """
    Analytical SAFE bounds for z1..z7 in maximize-form.

    These are intentionally conservative (safe) so E is safe.
    """
    idx.validate(); par.validate(idx)

    B: Dict[int, ObjBounds] = {}

    # z1 = - (sum u_i * j^2 * w_ij), min in paper
    # Since w is one-hot over j in [0..c[i]]: expr27 in [0 .. sum u_i*c[i]^2]
    ub_expr27 = sum(par.u[i] * (par.c[i] ** 2) for i in idx.I)
    B[1] = ObjBounds(lb=-ub_expr27, ub=0.0)

    # z2 = +coverage normalized in [0,1] (or 0 if denom=0)
    B[2] = ObjBounds(lb=0.0, ub=1.0)

    # z3 = +suitability: each x can contribute at most sum_q tbar[j,q] (since r<=1)
    # Safe bound: sum_{i,j,t,k,ell,p,q} r[i,q]*tbar[j,q]*x <= sum_{i,j,t,k,ell,p,q} tbar[j,q]
    # (very loose but safe)
    ub_z3 = (
        len(list(idx.I)) * len(list(idx.T)) * len(list(idx.K)) * len(list(idx.L)) * len(list(idx.P))
        * sum(par.tbar[j][q] for j in idx.J for q in idx.Q)
    )
    B[3] = ObjBounds(lb=0.0, ub=float(ub_z3))

    # z4 = -(expr30). expr30 is nonnegative-ish but safe bound:
    # term_max_potential <= sum_i u[i]*nvi[i]*(c[i]-1)
    # term_achieved >= 0 so expr30 <= that
    nvi = {i: sum(par.v[i][lbar] for lbar in range(0, par.b[i] + 1)) for i in idx.I}
    ub_expr30 = sum(par.u[i] * nvi[i] * max(par.c[i] - 1, 0) for i in idx.I)
    B[4] = ObjBounds(lb=-float(ub_expr30), ub=0.0)

    # z5 = -(expr31). expr31 <= sum u[i]*(max(l)-1)* (#assignments)
    max_l = max(par.l[i][k][ell] for i in idx.I0 for k in idx.K0 for ell in idx.L0)
    ub_expr31 = sum(par.u[i] * max(max_l - 1, 0) * par.c[i] for i in idx.I)
    B[5] = ObjBounds(lb=-float(ub_expr31), ub=0.0)

    # z6 = -(expr32). expr32 in [0, sum u[i]*n_k^2]
    ub_expr32 = sum(par.u[i] * (idx.n_k ** 2) for i in idx.I)
    B[6] = ObjBounds(lb=-float(ub_expr32), ub=0.0)

    # z7 = -(expr33). expr33 >= 0, safe upper bound:
    # each (i,k,ell,p) shat <= sum_{lbar<=a_i} h[i][lbar]
    ub_expr33 = 0.0
    for i in idx.I:
        M_hi = sum(par.h[i][lbar] for lbar in range(0, par.a[i] + 1))
        ub_expr33 += M_hi * len(list(idx.K)) * len(list(idx.L)) * len(list(idx.P)) * par.u[i]
    B[7] = ObjBounds(lb=-float(ub_expr33), ub=0.0)

    return B


def compute_safe_E(idx: Indices, par: Parameters, n_z: int = 7) -> int:
    """
    Choose E so that: 10^{-E} * (max possible sum of absolute other-objectives) < 1
    """
    bounds = objective_bounds_maxform(idx, par)

    # Safe bound on sum of absolute values across all objectives:
    M = 0.0
    for i in range(1, n_z + 1):
        b = bounds[i]
        M += max(abs(b.lb), abs(b.ub))

    # Need 10^{-E} * M < 1  =>  E > log10(M)
    if M <= 0:
        return 1
    return int(math.floor(math.log10(M)) + 2)  # +2 gives strict safety