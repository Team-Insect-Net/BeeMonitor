from __future__ import annotations
from typing import List, Tuple
import numpy as np

Point = Tuple[float, float]
Poly = List[Point]
BBox = Tuple[float, float, float, float]

def poly_to_bbox(poly: Poly) -> BBox:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))

def inflate_bbox(b: BBox, px: float) -> BBox:
    x1, y1, x2, y2 = b
    return (x1 - px, y1 - px, x2 + px, y2 + px)

def point_in_bbox(pt: Point, b: BBox) -> bool:
    x, y = pt
    x1, y1, x2, y2 = b
    return (x1 <= x <= x2) and (y1 <= y <= y2)

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t
