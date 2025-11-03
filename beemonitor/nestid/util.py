from __future__ import annotations
from typing import List, Tuple
from beemonitor.types import Tube

def bbox_centers(tubes: List[Tube]) -> List[Tuple[float, float]]:
    out = []
    for t in tubes:
        if not t.poly:
            out.append((float("nan"), float("nan")))
            continue
        (x1, y1), (x2, _), (_, y2), _ = t.poly
        out.append(((x1+x2)/2.0, (y1+y2)/2.0))
    return out
