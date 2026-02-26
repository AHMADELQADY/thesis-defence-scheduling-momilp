#src/instance_generator/availability.py
from __future__ import annotations
from dataclasses import dataclass
import random
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class SelfProbs:
    # conditional probs p(alpha|alpha) for alpha in {0,1,2,...}
    # we only need the diagonals; off-diagonals come from Eq.(46) in the appendix text
    p00: float
    p11: float
    p22: float | None = None  # only for lik (has states 1 and 2)

def _compute_offdiag_by_eq46(diag: Dict[int, float]) -> Dict[Tuple[int, int], float]:
    """
    Implements Eq.(46) idea: distribute (1 - p(alpha|alpha)) proportionally
    to the other diagonal probabilities.
    (The paper describes this proportional construction in the appendix around Algorithm 6.)  [oai_citation:10‡1-s2.0-S0377221723005003-main.pdf](sediment://file_00000000a254720a9b4556c0e2d569da)
    """
    states = sorted(diag.keys())
    out: Dict[Tuple[int, int], float] = {}
    denom_sum = sum(diag[s] for s in states)
    if denom_sum <= 0:
        raise ValueError("Invalid diagonal probabilities sum <= 0")

    for a in states:
        stay = diag[a]
        mass = 1.0 - stay
        for b in states:
            if b == a:
                continue
            out[(a, b)] = (diag[b] / denom_sum) * mass
    return out

def _sample_next(rng: random.Random, probs: List[Tuple[int, float]]) -> int:
    u = rng.random()
    acc = 0.0
    for state, p in probs:
        acc += p
        if u <= acc:
            return state
    return probs[-1][0]

def generate_availability_chain(
    *,
    rng: random.Random,
    n_k: int,
    n_ell: int,
    d: int,
    warmup: int,
    diag_probs: Dict[int, float],
    has_two_available_states: bool,
) -> List[List[int]]:
    """
    Returns matrix [k][ell] (1-indexed in your Parameters tensor; here we return python 1..n)
    Values:
      - 0 = unavailable
      - 1,2 = available “levels” (lik)
      - 1 = available (mkp)
    """
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    # states for Markov: base states are {0,1} or {0,1,2}
    base_states = sorted(diag_probs.keys())
    off = _compute_offdiag_by_eq46(diag_probs)

    # We model the “d−1 forced zeros” using transient states 0e1..0e(d-1).
    # When an availability state goes to 0, we actually go to 0e1, then deterministically
    # proceed to 0e2 ... then to 0. (For d=2, there is only 0e1).  [oai_citation:11‡1-s2.0-S0377221723005003-main.pdf](sediment://file_00000000a254720a9b4556c0e2d569da)
    forced_len = max(d - 1, 0)

    def step_transition(curr: int) -> int:
        # transient forced-zero states are encoded as negative integers: -1,-2,...,-(d-1)
        if curr < 0:
            nxt = curr - 1
            if abs(nxt) > forced_len:
                return 0
            return nxt

        # normal base state
        probs: List[Tuple[int, float]] = []

        # stay
        probs.append((curr, diag_probs[curr]))

        # move to other base states
        for b in base_states:
            if b == curr:
                continue
            p = off[(curr, b)]
            # if moving into 0, go into forced block start (if forced_len>0)
            if b == 0 and forced_len > 0:
                probs.append((-1, p))
            else:
                probs.append((b, p))

        # numeric stability
        s = sum(p for _, p in probs)
        if abs(s - 1.0) > 1e-6:
            # renormalize safely
            probs = [(st, p / s) for st, p in probs]

        return _sample_next(rng, probs)

    out: List[List[int]] = [[0] * (n_ell + 1) for _ in range(n_k + 1)]  # 1-indexed

    # paper initializes ell=1 “as if ell=0 was 0” and uses warm-up per day and per τ.  [oai_citation:12‡1-s2.0-S0377221723005003-main.pdf](sediment://file_00000000a254720a9b4556c0e2d569da)
    for k in range(1, n_k + 1):
        # start from base state 0
        state = 0

        # warm-up
        for _ in range(warmup):
            state = step_transition(state)

        # actual sequence for ell=1..n_ell
        for ell in range(1, n_ell + 1):
            state = step_transition(state)

            if state < 0:
                out[k][ell] = 0
            else:
                out[k][ell] = state  # 0/1/2

    return out

def diag_probs_lik(p_lik0: float) -> Dict[int, float]:
    # Table 3 in the paper (p(0|0)=0.95, p(1|1)=p(2|2) depending on p(lik=0)).  [oai_citation:13‡1-s2.0-S0377221723005003-main.pdf](sediment://file_00000000a254720a9b4556c0e2d569da)
    if abs(p_lik0 - 0.78) < 1e-9:
        p11 = p22 = 0.7
    elif abs(p_lik0 - 0.82) < 1e-9:
        p11 = p22 = 0.63
    elif abs(p_lik0 - 0.86) < 1e-9:
        p11 = p22 = 0.55
    else:
        raise ValueError("p_lik0 must be one of {0.78,0.82,0.86}")
    return {0: 0.95, 1: p11, 2: p22}

def diag_probs_mkp(p_mkp0: float) -> Dict[int, float]:
    # Table 4 in the paper.  [oai_citation:14‡1-s2.0-S0377221723005003-main.pdf](sediment://file_00000000a254720a9b4556c0e2d569da)
    if abs(p_mkp0 - 0.8) < 1e-9:
        p11 = 0.7
    elif abs(p_mkp0 - 0.86) < 1e-9:
        p11 = 0.8
    else:
        raise ValueError("p_mkp0 must be one of {0.8,0.86}")
    return {0: 0.95, 1: p11}