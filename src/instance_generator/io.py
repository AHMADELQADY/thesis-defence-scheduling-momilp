#src/instance_generator/io.py

from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path

from src.common.symbols import Indices
from src.common.parameters import Parameters

def save_instance(path: str | Path, idx: Indices, par: Parameters) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "idx": asdict(idx),
        "par": asdict(par),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

def load_instance(path: str | Path) -> tuple[Indices, Parameters]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    idx = Indices(**payload["idx"])
    par = Parameters(**payload["par"])
    par.validate(idx)
    return idx, par