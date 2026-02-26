# src/common/parameters.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List

from src.common.symbols import Indices


@dataclass(frozen=True)
class Parameters:
    """
    Paper Section 2.2 — Parameters.

    All tensors are stored INCLUDING dummy index 0,
    i.e., sizes are (n+1) in every dimension.

    Guardrails:
    - b[i], a[i] must be in 0..d-1 for real members
    - dummy index slices (i=0 or j=0 or t=0 etc.) must be 0 where appropriate
    - l[i][k][ell] must be in N0 (non-negative integer)
    - (recommended) r and tbar dummy slices must be 0; r/tbar should be binary
    """

    # 3) Related to thesis defences
    d: int  # duration in hour slots

    # 1) Related to committee members
    e: List[List[List[int]]]      # e[i][j][t] ∈ {0,1}
    c: List[int]                  # c[i] ∈ N
    u: List[int]                  # u[i] ∈ N
    l: List[List[List[int]]]      # l[i][k][ell] ∈ N0 (0 means unavailable)
    r: List[List[int]]            # r[i][q] ∈ {0,1}

    # 2) Related to committee members and rooms
    b: List[int]                  # b[i] ∈ {0..d-1}
    v: List[List[int]]            # v[i][delta] for delta=0..d-1
    a: List[int]                  # a[i] ∈ {0..d-1}
    h: List[List[int]]            # h[i][delta] for delta=0..d-1
    m: List[List[List[int]]]      # m[k][ell][p] ∈ {0,1}

    # defence subject indicator (paper uses \bar{t}_{jq})
    tbar: List[List[int]]         # tbar[j][q] ∈ {0,1}

    def validate(self, idx: Indices) -> None:
        idx.validate()

        if self.d <= 0:
            raise ValueError("d must be positive")

        # ------------------------------------------------------------
        # Helper: enforce correct (n+1) tensor sizes
        # ------------------------------------------------------------
        def expect_len(name: str, arr, n: int):
            if len(arr) != n + 1:
                raise ValueError(f"{name} must have length {n+1}, got {len(arr)}")

        # ------------------------------------------------------------
        # 1D vectors
        # ------------------------------------------------------------
        expect_len("c", self.c, idx.n_i)
        expect_len("u", self.u, idx.n_i)
        expect_len("b", self.b, idx.n_i)
        expect_len("a", self.a, idx.n_i)

        # ------------------------------------------------------------
        # 2D matrices
        # ------------------------------------------------------------
        expect_len("r", self.r, idx.n_i)
        if any(len(self.r[i]) != idx.n_q + 1 for i in idx.I0):
            raise ValueError("r[i] must have length n_q+1 for all i")

        expect_len("tbar", self.tbar, idx.n_j)
        if any(len(self.tbar[j]) != idx.n_q + 1 for j in idx.J0):
            raise ValueError("tbar[j] must have length n_q+1 for all j")

        # ------------------------------------------------------------
        # Guardrails (recommended): r and tbar dummy slices + binarity
        # ------------------------------------------------------------
        # Dummy member i=0 must have no expertise
        for q in idx.Q0:
            if self.r[0][q] != 0:
                raise ValueError("r[0][*] must be 0 (dummy member)")

        # Dummy subject q=0 must be 0 for all members
        for i in idx.I0:
            if self.r[i][0] != 0:
                raise ValueError("r[*][0] must be 0 (dummy subject)")

        # Dummy defence j=0 must have no subjects
        for q in idx.Q0:
            if self.tbar[0][q] != 0:
                raise ValueError("tbar[0][*] must be 0 (dummy defence)")

        # Dummy subject q=0 must be 0 for all defences
        for j in idx.J0:
            if self.tbar[j][0] != 0:
                raise ValueError("tbar[*][0] must be 0 (dummy subject)")

        # Enforce binary values for r and tbar (prevents silent data bugs)
        for i in idx.I0:
            for q in idx.Q0:
                if self.r[i][q] not in (0, 1):
                    raise ValueError(f"r[{i}][{q}] must be 0/1, got {self.r[i][q]!r}")

        for j in idx.J0:
            for q in idx.Q0:
                if self.tbar[j][q] not in (0, 1):
                    raise ValueError(f"tbar[{j}][{q}] must be 0/1, got {self.tbar[j][q]!r}")

        # ------------------------------------------------------------
        # 3D tensor: e
        # ------------------------------------------------------------
        expect_len("e", self.e, idx.n_i)
        if any(len(self.e[i]) != idx.n_j + 1 for i in idx.I0):
            raise ValueError("e[i] must have length n_j+1 for all i")
        if any(len(self.e[i][j]) != idx.n_t + 1 for i in idx.I0 for j in idx.J0):
            raise ValueError("e[i][j] must have length n_t+1 for all i,j")

        # ------------------------------------------------------------
        # 3D tensor: l (time-slot preference / availability)
        # ------------------------------------------------------------
        expect_len("l", self.l, idx.n_i)
        if any(len(self.l[i]) != idx.n_k + 1 for i in idx.I0):
            raise ValueError("l[i] must have length n_k+1 for all i")
        if any(len(self.l[i][k]) != idx.n_ell + 1 for i in idx.I0 for k in idx.K0):
            raise ValueError("l[i][k] must have length n_ell+1 for all i,k")

        # ------------------------------------------------------------
        # 3D tensor: m (room availability)
        # ------------------------------------------------------------
        expect_len("m", self.m, idx.n_k)
        if any(len(self.m[k]) != idx.n_ell + 1 for k in idx.K0):
            raise ValueError("m[k] must have length n_ell+1 for all k")
        if any(len(self.m[k][ell]) != idx.n_p + 1 for k in idx.K0 for ell in idx.L0):
            raise ValueError("m[k][ell] must have length n_p+1 for all k,ell")

        # ------------------------------------------------------------
        # v / h vectors (store full length d)
        # ------------------------------------------------------------
        expect_len("v", self.v, idx.n_i)
        if any(len(self.v[i]) != self.d for i in idx.I0):
            raise ValueError("v[i] must have length d for all i (delta=0..d-1)")

        expect_len("h", self.h, idx.n_i)
        if any(len(self.h[i]) != self.d for i in idx.I0):
            raise ValueError("h[i] must have length d for all i (delta=0..d-1)")

        # ------------------------------------------------------------
        # Guardrails / value locking
        # ------------------------------------------------------------

        # b[i], a[i] bounds (real members only)
        for i in idx.I:
            bi = self.b[i]
            ai = self.a[i]
            if not (0 <= bi <= self.d - 1):
                raise ValueError(f"b[{i}] must be in 0..d-1, got {bi}")
            if not (0 <= ai <= self.d - 1):
                raise ValueError(f"a[{i}] must be in 0..d-1, got {ai}")

        # ------------------------------------------------------------
        # e dummy slices must be 0
        # ------------------------------------------------------------
        for j in idx.J0:
            for t in idx.T0:
                if self.e[0][j][t] != 0:
                    raise ValueError("e[0][*][*] must be 0 (dummy member)")

        for i in idx.I0:
            for t in idx.T0:
                if self.e[i][0][t] != 0:
                    raise ValueError("e[*][0][*] must be 0 (dummy defence)")

        for i in idx.I0:
            for j in idx.J0:
                if self.e[i][j][0] != 0:
                    raise ValueError("e[*][*][0] must be 0 (dummy role)")

        # ------------------------------------------------------------
        # Room availability dummy room p=0 must be 0
        # ------------------------------------------------------------
        for k in idx.K0:
            for ell in idx.L0:
                if self.m[k][ell][0] != 0:
                    raise ValueError("m[*][*][0] must be 0 (dummy room)")

        # ------------------------------------------------------------
        # Availability dummy slices for l must be 0
        # ------------------------------------------------------------
        for i in idx.I0:
            for k in idx.K0:
                if self.l[i][k][0] != 0:
                    raise ValueError("l[*][*][0] must be 0 (dummy hour-slot)")
            for ell in idx.L0:
                if self.l[i][0][ell] != 0:
                    raise ValueError("l[*][0][*] must be 0 (dummy day)")

        # ------------------------------------------------------------
        # l[i][k][ell] must be in N0 (non-negative integers)
        # ------------------------------------------------------------
        # Paper meaning:
        #   l = 0  -> unavailable (blocked by A.6)
        #   l = 1  -> best preference (0 penalty)
        #   l > 1  -> worse preference (positive penalty in objective 31)
        #
        # IMPORTANT:
        #   If l were negative, objective (31) would become invalid.
        #   We therefore enforce:
        #       l[i][k][ell] ∈ {0,1,2,...}
        #
        for i in idx.I0:
            for k in idx.K0:
                for ell in idx.L0:
                    val = self.l[i][k][ell]

                    if not isinstance(val, int) or val < 0:
                        raise ValueError(
                            f"l[{i}][{k}][{ell}] must be a non-negative integer (N0), got {val!r}"
                        )

                    # Dummy member slice must be 0 (extra safety; also implied by checks above)
                    if i == 0 and val != 0:
                        raise ValueError("l[0][*][*] must be 0 (dummy member)")