# src/common/solve_tracker.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class SolveTracker:
    solve_id: int = 0

    def tick(self, label: str) -> int:
        self.solve_id += 1
        print(f"\n=== SOLVE #{self.solve_id}: {label} ===\n")
        return self.solve_id