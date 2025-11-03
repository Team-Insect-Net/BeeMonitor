from __future__ import annotations
from typing import List, Tuple
import numpy as np, cv2

try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

from beemonitor.types import Tube, Detection
from .template import NestTemplate

BBox = Tuple[float, float, float, float]

# ------------------ small geometry helpers ------------------

def _bbox_center(b: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (0.5*(x1+x2), 0.5*(y1+y2))

def _centers(dets: List[Detection]) -> np.ndarray:
    return np.array([_bbox_center(d.bbox) for d in dets], dtype=np.float32)

def _apply_affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return pts.copy()
    ones = np.ones((pts.shape[0],1), dtype=np.float32)
    return (M @ np.hstack([pts, ones]).T).T

def _apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return pts.copy()
    ones = np.ones((pts.shape[0],1), dtype=np.float32)
    Y = (H @ np.hstack([pts, ones]).T).T
    return Y[:, :2] / np.clip(Y[:, 2:3], 1e-6, None)

def _affine(src: np.ndarray, dst: np.ndarray, thr=4.0, iters=5000):
    # src, dst must be Nx2 and same length
    if src.shape[0] >= 3 and src.shape[0] == dst.shape[0]:
        M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                              ransacReprojThreshold=thr, maxIters=iters)
    else:
        M, mask = None, None
    if M is None:
        M = np.array([[1,0,0],[0,1,0]], dtype=np.float32)
        mask = np.ones((src.shape[0],1), dtype=np.uint8) if src.size else np.zeros((0,1), dtype=np.uint8)
    return M, mask

def _homography(src: np.ndarray, dst: np.ndarray, thr=3.0, iters=5000):
    # src, dst must be Nx2 and same length
    if src.shape[0] >= 4 and src.shape[0] == dst.shape[0]:
        H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC,
                                     ransacReprojThreshold=thr, maxIters=iters)
    else:
        H, mask = None, None
    if H is None:
        H = np.eye(3, dtype=np.float32)
        mask = np.ones((src.shape[0],1), dtype=np.uint8) if src.size else np.zeros((0,1), dtype=np.uint8)
    return H, mask

def _residual(mapped: np.ndarray, tpl: np.ndarray) -> float:
    if mapped.size == 0 or tpl.size == 0:
        return float("inf")
    D = np.linalg.norm(mapped[:, None, :] - tpl[None, :, :], axis=2)
    if _HAVE_SCIPY:
        r, c = linear_sum_assignment(D)
        return float(D[r, c].mean())
    r = np.arange(D.shape[0])
    c = np.argmin(D, axis=1)
    return float(D[r, c].mean())

# ------------------ matching helpers ------------------

def _hungarian(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if _HAVE_SCIPY:
        return linear_sum_assignment(cost)
    # greedy fallback
    n_rows, n_cols = cost.shape
    chosen_cols, rows, cols = set(), [], []
    for i in range(n_rows):
        order = np.argsort(cost[i])
        for j in order:
            j = int(j)
            if j not in chosen_cols:
                rows.append(i); cols.append(j); chosen_cols.add(j); break
    return np.asarray(rows), np.asarray(cols)

def _pair_subsets(det_pts: np.ndarray, tpl_pts: np.ndarray, max_pairs: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce equal-length subsets (src_idx, dst_idx) using nearest-neighbor assignment
    without any transform yet. This gives tentative correspondences so we can
    estimate affine/homography even when counts differ.
    """
    if det_pts.size == 0 or tpl_pts.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    D = np.linalg.norm(det_pts[:, None, :] - tpl_pts[None, :, :], axis=2)  # (K,N)
    r, c = _hungarian(D)  # indices into det_pts and tpl_pts
    if max_pairs is not None:
        k = min(len(r), max_pairs)
        return r[:k], c[:k]
    return r, c

# ------------------ transform picker using paired subsets ------------------

class GeoMap:
    """Stores the chosen transform and provides forward/inverse mapping."""
    def __init__(self, kind: str, M: np.ndarray):
        self.kind = kind
        self.M = M

    def forward(self, pts: np.ndarray) -> np.ndarray:
        return _apply_affine(self.M, pts) if self.kind == "affine" else _apply_homography(self.M, pts)

    def inverse(self, pts: np.ndarray) -> np.ndarray:
        if self.kind == "affine":
            A, t = self.M[:, :2], self.M[:, 2:]
            Ainv = np.linalg.inv(A)
            Minv = np.concatenate([Ainv, -Ainv @ t], axis=1)
            return _apply_affine(Minv, pts)
        Hinv = np.linalg.inv(self.M)
        return _apply_homography(Hinv, pts)

def _pick_transform(det_pts: np.ndarray, tpl_pts: np.ndarray,
                    method_order=("affine","homography"),
                    aff_thr=4.0, hom_thr=3.0, residual_ok=8.0) -> GeoMap:
    """
    1) Build tentative correspondences between det_pts and tpl_pts (Hungarian/greedy).
    2) Fit affine on matched pairs; if residual is high and allowed, fit homography.
    Works when counts differ.
    """
    # build pairs
    r, c = _pair_subsets(det_pts, tpl_pts)
    src = det_pts[r]  # Nx2
    dst = tpl_pts[c]  # Nx2

    # Try affine first (needs >=3 pairs)
    if "affine" in method_order and src.shape[0] >= 3:
        M, _ = _affine(src, dst, thr=aff_thr)
        m = _apply_affine(M, det_pts)
        if _residual(m, tpl_pts) <= residual_ok or "homography" not in method_order:
            return GeoMap("affine", M)

    # Fallback to homography (needs >=4 pairs)
    if "homography" in method_order and src.shape[0] >= 4:
        H, _ = _homography(src, dst, thr=hom_thr)
        return GeoMap("homography", H)

    # Last resort: identity affine
    I = np.array([[1,0,0],[0,1,0]], dtype=np.float32)
    return GeoMap("affine", I)

# ------------------ fill-missing helpers ------------------

def _estimate_box_sizes(dets: List[Detection], k: int = 6) -> Tuple[float, float]:
    """Median w,h from k detections closest to the global median center."""
    if not dets:
        return 16.0, 16.0
    ws, hs, ctrs = [], [], []
    for d in dets:
        x1,y1,x2,y2 = d.bbox
        ws.append(x2-x1); hs.append(y2-y1)
        ctrs.append(((x1+x2)/2.0, (y1+y2)/2.0))
    ws = np.asarray(ws); hs = np.asarray(hs); ctrs = np.asarray(ctrs, dtype=np.float32)
    med = np.median(ctrs, axis=0, keepdims=True)
    order = np.argsort(np.linalg.norm(ctrs - med, axis=1))[:max(1, min(k, len(dets)))]
    return float(np.median(ws[order])), float(np.median(hs[order]))

def _box_from_center(cx: float, cy: float, w: float, h: float):
    x1, y1 = cx - w/2.0, cy - h/2.0
    x2, y2 = cx + w/2.0, cy + h/2.0
    return [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]

# ------------------ main assignment ------------------

def assign_ids(
    dets: List[Detection], template: NestTemplate,
    allow_partial: bool = True, min_visible_fraction: float = 0.85,
    method_order=("affine","homography"),
    aff_thr=4.0, hom_thr=3.0, residual_ok=8.0,
    fill_missing: bool = True
) -> List[Tube]:
    """
    Align detection centers to canonical template and assign IDs consistently.
    If some detections are missing and `fill_missing=True`, synthesize boxes at
    projected template centers (size from neighbors).
    """
    N = template.rows * template.cols
    tpl_pts = np.array(template.centers, dtype=np.float32)

    if not dets:
        return [Tube(tube_id=tid, poly=[]) for tid in template.ids]

    det_pts = _centers(dets)

    # Pick a robust transform using matched subsets (works when counts differ)
    geom = _pick_transform(det_pts, tpl_pts, method_order, aff_thr, hom_thr, residual_ok)

    # Map detections into canonical space and compute cost to every template point
    det_in_canon = geom.forward(det_pts)
    D = np.linalg.norm(det_in_canon[:, None, :] - tpl_pts[None, :, :], axis=2)  # (K,N)

    # Optimal assignment for whatever we do have
    r, c = _hungarian(D)

    # Build polygons for matched detections
    id_to_poly = {}
    for i_det, j_tpl in zip(r, c):
        x1,y1,x2,y2 = dets[int(i_det)].bbox
        id_to_poly[ template.ids[int(j_tpl)] ] = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]

    visible = len(id_to_poly)
    if visible / N < min_visible_fraction:
        # Too few to trust transform → abort to let caller handle
        raise RuntimeError(f"Only {visible}/{N} tubes visible (<{min_visible_fraction*100:.0f}%).")

    # Fill missing using template centers projected back to image space
    if fill_missing and visible < N:
        w_est, h_est = _estimate_box_sizes(dets, k=6)
        missing_ids = [tid for tid in template.ids if tid not in id_to_poly]
        if missing_ids:
            canon_missing = np.array([template.centers[template.ids.index(tid)] for tid in missing_ids], dtype=np.float32)
            img_centers = geom.inverse(canon_missing)
            for tid, (cx,cy) in zip(missing_ids, img_centers):
                id_to_poly[tid] = _box_from_center(float(cx), float(cy), w_est, h_est)

    # Emit in canonical (row-major) order → IDs never shift
    return [Tube(tube_id=tid, poly=id_to_poly.get(tid, [])) for tid in template.ids]
