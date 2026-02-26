# src/common/symbols.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Indices:
    """
    Paper Section 2.1 — Indices.

    IMPORTANT convention (paper):
    - All indices include 0..n
    - Index 0 is NOT a real object, but is used to represent 'absence' in
      some constraints/objectives.

    We therefore expose both:
    - *_0 : 0..n  (includes dummy 0)
    - *   : 1..n  (real objects only)
    """

    n_i: int  # committee members
    n_j: int  # defences
    n_t: int  # roles
    n_k: int  # days
    n_ell: int  # hour slots per day
    n_p: int  # rooms
    n_q: int  # research subjects

    # ---------- include dummy 0 ----------
    @property
    def I0(self): return range(0, self.n_i + 1)

    @property
    def J0(self): return range(0, self.n_j + 1)

    @property
    def T0(self): return range(0, self.n_t + 1)

    @property
    def K0(self): return range(0, self.n_k + 1)

    @property
    def L0(self): return range(0, self.n_ell + 1)

    @property
    def P0(self): return range(0, self.n_p + 1)

    @property
    def Q0(self): return range(0, self.n_q + 1)

    # ---------- real objects only ----------
    @property
    def I(self): return range(1, self.n_i + 1)

    @property
    def J(self): return range(1, self.n_j + 1)

    @property
    def T(self): return range(1, self.n_t + 1)

    @property
    def K(self): return range(1, self.n_k + 1)

    @property
    def L(self): return range(1, self.n_ell + 1)

    @property
    def P(self): return range(1, self.n_p + 1)

    @property
    def Q(self): return range(1, self.n_q + 1)

    # ---------- guardrails ----------
    def validate(self) -> None:
        # validate counts
        for name, v in vars(self).items():
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"{name} must be a non-negative int, got {v!r}")

        # validate that real sets start at 1 when non-empty
        if self.n_i > 0 and min(self.I) != 1: raise ValueError("I must start at 1")
        if self.n_j > 0 and min(self.J) != 1: raise ValueError("J must start at 1")
        if self.n_t > 0 and min(self.T) != 1: raise ValueError("T must start at 1")
        if self.n_k > 0 and min(self.K) != 1: raise ValueError("K must start at 1")
        if self.n_ell > 0 and min(self.L) != 1: raise ValueError("L must start at 1")
        if self.n_p > 0 and min(self.P) != 1: raise ValueError("P must start at 1")
        if self.n_q > 0 and min(self.Q) != 1: raise ValueError("Q must start at 1")

    def is_dummy(self, *, i=None, j=None, t=None, k=None, ell=None, p=None, q=None) -> bool:
        """Utility for debugging: true if any provided index equals 0."""
        return any(v == 0 for v in [i, j, t, k, ell, p, q] if v is not None)