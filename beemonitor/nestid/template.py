from __future__ import annotations
import json, os
from dataclasses import dataclass
from typing import List, Tuple, Dict

Point = Tuple[float, float]

@dataclass
class NestTemplate:
    rows: int
    cols: int
    ids: List[str]            # length rows*cols (row-major)
    centers: List[Point]      # canonical centers ordered to match ids

    def to_dict(self) -> Dict:
        return {"rows": self.rows, "cols": self.cols, "ids": self.ids, "centers": self.centers}

    @classmethod
    def from_dict(cls, d: Dict) -> "NestTemplate":
        return cls(rows=int(d["rows"]), cols=int(d["cols"]),
                   ids=list(d["ids"]), centers=[tuple(map(float, c)) for c in d["centers"]])

def save_template(tpl: NestTemplate, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(tpl.to_dict(), f, indent=2)

def load_template(path: str) -> NestTemplate:
    with open(path, "r") as f:
        return NestTemplate.from_dict(json.load(f))
