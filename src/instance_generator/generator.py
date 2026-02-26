#src/instance_generator/generator.py

from __future__ import annotations
import math
import random
from typing import List

from src.common.symbols import Indices
from src.common.parameters import Parameters
from src.instance_generator.config import InstanceSize, PaperKnobs
from src.instance_generator.availability import (
    generate_availability_chain, diag_probs_lik, diag_probs_mkp
)

def _zeros_3d(a: int, b: int, c: int) -> List[List[List[int]]]:
    return [[[0 for _ in range(c)] for _ in range(b)] for _ in range(a)]

def _zeros_2d(a: int, b: int) -> List[List[int]]:
    return [[0 for _ in range(b)] for _ in range(a)]

def _zeros_1d(a: int) -> List[int]:
    return [0 for _ in range(a)]

def _choose_subset(rng: random.Random, universe: List[int], k: int) -> List[int]:
    if k > len(universe):
        raise ValueError("k > universe")
    arr = universe[:]
    rng.shuffle(arr)
    return arr[:k]

def generate_instance(size: InstanceSize, knobs: PaperKnobs, *, seed: int) -> tuple[Indices, Parameters]:
    rng = random.Random(seed)

    idx = Indices(
        n_i=size.n_i, n_j=size.n_j, n_t=size.n_t, n_k=size.n_k,
        n_ell=size.n_ell, n_p=size.n_p, n_q=size.n_q
    )
    idx.validate()

    d = size.d

    # -------------------------
    # c_i = ceil(0.5 * n_i)  -> 13 / 19 / 25 as in Tables C.*  [oai_citation:15‡1-s2.0-S0377221723005003-main.pdf](sediment://file_0000000068007246b2dc30a22acae3d6)
    # -------------------------
    c_val = int(math.ceil(0.5 * size.n_i))
    c = _zeros_1d(size.n_i + 1)
    for i in idx.I:
        c[i] = c_val

    # -------------------------
    # u_i: paper only states [0.7,0.3] in the tables (not the exact sampling rule)
    # We implement a reproducible split: half 0.7, half 0.3, shuffled.
    # -------------------------
    u = _zeros_1d(size.n_i + 1)
    vals = [0.7] * (size.n_i // 2) + [0.3] * (size.n_i - size.n_i // 2)
    rng.shuffle(vals)
    for i, v in zip(list(idx.I), vals):
        u[i] = int(round(v * 10))  # store as int weights (7 or 3) to keep everything integer-safe
    # NOTE: if you want exact floats in objectives, change to float list and adjust Parameters typing.

    # -------------------------
    # e_ijt: “fixed roles” control complexity (1 or 2 fixed roles)  [oai_citation:16‡1-s2.0-S0377221723005003-main.pdf](sediment://file_00000000ce34722fb3cf1b7105bd8f6d)
    # Implement: choose 'fixed_roles' roles that are fixed per defence (only a subset eligible),
    # remaining roles have global eligible set.
    # -------------------------
    e = _zeros_3d(size.n_i + 1, size.n_j + 1, size.n_t + 1)
    # global eligible set per role
    role_global = {}
    for t in idx.T:
        # allow ~70% eligible globally
        eligible = set(_choose_subset(rng, list(idx.I), max(1, int(0.7 * size.n_i))))
        role_global[t] = eligible

    fixed = set(_choose_subset(rng, list(idx.T), knobs.fixed_roles))
    for i in idx.I:
        for j in idx.J:
            for t in idx.T:
                if t in fixed:
                    # fixed: per defence choose subset from global
                    per_def = set(_choose_subset(rng, list(role_global[t]), max(1, int(0.4 * size.n_i))))
                    e[i][j][t] = 1 if i in per_def else 0
                else:
                    e[i][j][t] = 1 if i in role_global[t] else 0

    # -------------------------
    # r_iq: each member covers exactly riq_per_member subjects (tables show 3)  [oai_citation:17‡1-s2.0-S0377221723005003-main.pdf](sediment://file_0000000068007246b2dc30a22acae3d6)
    # tbar_jq: each defence has exactly tiq_per_defence subjects (tables show 3)  [oai_citation:18‡1-s2.0-S0377221723005003-main.pdf](sediment://file_0000000068007246b2dc30a22acae3d6)
    # -------------------------
    r = _zeros_2d(size.n_i + 1, size.n_q + 1)
    for i in idx.I:
        qs = _choose_subset(rng, list(idx.Q), knobs.riq_per_member)
        for q in qs:
            r[i][q] = 1

    tbar = _zeros_2d(size.n_j + 1, size.n_q + 1)
    for j in idx.J:
        qs = _choose_subset(rng, list(idx.Q), knobs.tiq_per_defence)
        for q in qs:
            tbar[j][q] = 1

    # -------------------------
    # v_i and h_i: either [1] or [2,1] with probabilities shown in the paper  [oai_citation:19‡1-s2.0-S0377221723005003-main.pdf](sediment://file_00000000ce34722fb3cf1b7105bd8f6d)
    # We set b_i=a_i=d-1 (so lbar in {0,1} because d=2).
    # Mapping:
    #   [1]   -> [1, 0]
    #   [2,1] -> [2, 1]
    # -------------------------
    b = _zeros_1d(size.n_i + 1)
    a = _zeros_1d(size.n_i + 1)
    v = [[0 for _ in range(d)] for _ in range(size.n_i + 1)]
    h = [[0 for _ in range(d)] for _ in range(size.n_i + 1)]
    for i in idx.I:
        b[i] = d - 1
        a[i] = d - 1

        if rng.random() < knobs.p_v_21:
            v[i] = [2, 1]
        else:
            v[i] = [1, 0]

        if rng.random() < knobs.p_h_21:
            h[i] = [2, 1]
        else:
            h[i] = [1, 0]

    # -------------------------
    # lik: Algorithm 6 Markov availability, warm-up=40, with forced d-1 zeros rule  [oai_citation:20‡1-s2.0-S0377221723005003-main.pdf](sediment://file_00000000a254720a9b4556c0e2d569da)
    # Values are {0,1,2}
    # -------------------------
    l = _zeros_3d(size.n_i + 1, size.n_k + 1, size.n_ell + 1)
    for i in idx.I:
        mat = generate_availability_chain(
            rng=rng, n_k=size.n_k, n_ell=size.n_ell, d=d, warmup=40,
            diag_probs=diag_probs_lik(knobs.p_lik0),
            has_two_available_states=True,
        )
        for k in idx.K:
            for ell in idx.L:
                l[i][k][ell] = mat[k][ell]

    # -------------------------
    # mkp: Algorithm 6 Markov availability for rooms, values {0,1}  [oai_citation:21‡1-s2.0-S0377221723005003-main.pdf](sediment://file_00000000a254720a9b4556c0e2d569da)
    # store in m[k][ell][p]
    # -------------------------
    m = _zeros_3d(size.n_k + 1, size.n_ell + 1, size.n_p + 1)
    for p in idx.P:
        mat = generate_availability_chain(
            rng=rng, n_k=size.n_k, n_ell=size.n_ell, d=d, warmup=40,
            diag_probs=diag_probs_mkp(knobs.p_mkp0),
            has_two_available_states=False,
        )
        for k in idx.K:
            for ell in idx.L:
                m[k][ell][p] = 1 if mat[k][ell] == 1 else 0

    par = Parameters(
        d=d,
        e=e, c=c, u=u,
        l=l, r=r,
        b=b, v=v,
        a=a, h=h,
        m=m,
        tbar=tbar,
    )
    par.validate(idx)
    return idx, par