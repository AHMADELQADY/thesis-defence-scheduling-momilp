# src/model/variables.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from gurobipy import GRB, Model
from src.common.symbols import Indices


@dataclass(frozen=True)
class Vars:
    # Decision variable (paper 2.3.1a)
    x: Any  # x[i,j,t,k,ell,p] binary

    # Auxiliary variables (paper 2.3.2)
    y_def: Any        # y[j,k,ell,p]      defence scheduled indicator
    y_mem: Any        # y_mem[i,k,ell,p]  member presence at (k,ell,p); paper's \bar{y}_{ikℓp}

    yhat: Any         # yhat[i,j,k]   (paper: \hat{y}_{ijk}, j includes 0)
    w: Any            # w[i,j]        (paper: w_ij, j includes 0)
    wbar: Any         # wbar[i,k]     (paper: \bar{w}_{ik}, k includes 0)

    s: Any            # s[i,j,q]      (paper: s_ijq, i includes 0)

    # Objective-measure variables:
    sbar_comp: Any    # \bar{s}_{ikℓ}    compactness value (NO room index p)
    shat_roomchg: Any # \hat{s}_{ikℓp}   room-change value (HAS room index p)


def build_variables(m: Model, idx: Indices) -> Vars:
    """
    Create all variables from paper Section 2.3.

    LOCKING RULES:
    - Decision/scheduling vars NEVER use dummy index 0.
    - Only the specific paper vars that explicitly allow 0 (J0/K0/I0) will use them.

    OPTIONAL PERFORMANCE KNOB:
    - sbar_comp and shat_roomchg are *penalty measures* used in objectives.
      They do NOT need to be integer to keep correctness of the model.
      Making them CONTINUOUS often speeds up Gurobi (fewer integer vars).
    """
    idx.validate()

    # Decision: only real objects
    x = m.addVars(idx.I, idx.J, idx.T, idx.K, idx.L, idx.P,
                  vtype=GRB.BINARY, name="x")

    # Aux scheduling indicators
    y_def = m.addVars(idx.J, idx.K, idx.L, idx.P,
                      vtype=GRB.BINARY, name="y_def")

    # paper's \bar{y}_{ikℓp}
    y_mem = m.addVars(idx.I, idx.K, idx.L, idx.P,
                      vtype=GRB.BINARY, name="y_mem")

    # j includes dummy 0
    yhat = m.addVars(idx.I, idx.J0, idx.K,
                     vtype=GRB.BINARY, name="yhat")

    # j includes dummy 0
    w = m.addVars(idx.I, idx.J0,
                  vtype=GRB.BINARY, name="w")

    # k includes dummy 0
    wbar = m.addVars(idx.I, idx.K0,
                     vtype=GRB.BINARY, name="wbar")

    # i includes dummy 0
    s = m.addVars(idx.I0, idx.J, idx.Q,
                  vtype=GRB.BINARY, name="s")

    # ----------------------------------------------------------------------
    # Penalty variables (objective measures)
    #
    # Optional improvement: CONTINUOUS instead of INTEGER => usually faster.
    # They are still forced to nonnegative by lb=0, so you don't need extra
    # constraints like ">= 0" later.
    # ----------------------------------------------------------------------

    # Compactness penalty: NO room dimension
    sbar_comp = m.addVars(idx.I, idx.K, idx.L,
                          vtype=GRB.CONTINUOUS, lb=0.0, name="sbar_comp")

    # Room-change penalty: HAS room dimension
    shat_roomchg = m.addVars(idx.I, idx.K, idx.L, idx.P,
                             vtype=GRB.CONTINUOUS, lb=0.0, name="shat_roomchg")

    return Vars(
        x=x,
        y_def=y_def,
        y_mem=y_mem,
        yhat=yhat,
        w=w,
        wbar=wbar,
        s=s,
        sbar_comp=sbar_comp,
        shat_roomchg=shat_roomchg,
    )