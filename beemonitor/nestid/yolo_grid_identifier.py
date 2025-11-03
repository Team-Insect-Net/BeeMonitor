from __future__ import annotations
from typing import List, Optional
import os, json, cv2, numpy as np

from beemonitor.types import Tube, Detection
from beemonitor.detect.factory import build_detector
from .template import save_template, load_template, NestTemplate
from .assign import assign_ids

class YoloGridNestIdentifier:
    """
    YOLO-based tube detection + canonical-grid ID assignment with caching and revalidation.
    """

    def __init__(self, cfg_root: dict, rows: int, cols: int,
                 template_path: str, cache_dir: str = "outputs/nest_cache"):
        self.cfg_root = cfg_root
        self.rows = rows
        self.cols = cols
        self.template_path = template_path
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.detector = build_detector(cfg_root["detectors"]["nest"])
        a = (cfg_root.get("nestid") or {}).get("alignment", {})
        self.method_order = tuple(a.get("method_order", ["affine", "homography"]))
        self.aff_thr = float(a.get("ransac_reproj_px", 4.0))
        self.hom_thr = float(a.get("homography_reproj_px", 3.0))
        self.residual_ok = float(a.get("residual_px_ok", 8.0))
        self.allow_partial = bool(a.get("allow_partial", True))
        self.min_visible_fraction = float(a.get("min_visible_fraction", 0.85))
        rv = (cfg_root.get("nestid") or {}).get("revalidate", {})
        self.revalidate_every = int(rv.get("every_n_frames", 300))
        self.jitter_tol = float(rv.get("jitter_px_tolerance", 5.0))
        self.fill_missing = bool((cfg_root.get("nestid") or {}).get("fill_missing", True))

    # ---------- public API ----------

    def detect_tubes(self, frame_bgr: np.ndarray) -> List[Tube]:
        dets = self._detect(frame_bgr)
        tpl = load_template(self.template_path)
        return assign_ids(
            dets, tpl,
            allow_partial=self.allow_partial,
            min_visible_fraction=self.min_visible_fraction,
            method_order=self.method_order,
            aff_thr=self.aff_thr,
            hom_thr=self.hom_thr,
            residual_ok=self.residual_ok,
            fill_missing=self.fill_missing
        )

    def detect_or_load(self, video_path: str) -> List[Tube]:
        name = os.path.splitext(os.path.basename(video_path))[0]
        cache_path = os.path.join(self.cache_dir, f"{name}_tubes.json")
        if os.path.exists(cache_path):
            data = json.load(open(cache_path))
            return [Tube(**t) for t in data["tubes"]]

        tpl = load_template(self.template_path)
        frame = self._first_frame(video_path)
        dets = self._detect(frame)
        tubes = assign_ids(
            dets, tpl,
            allow_partial=self.allow_partial,
            min_visible_fraction=self.min_visible_fraction,
            method_order=self.method_order,
            aff_thr=self.aff_thr,
            hom_thr=self.hom_thr,
            residual_ok=self.residual_ok,
            fill_missing=self.fill_missing
        )
        json.dump({"tubes": [t.__dict__ for t in tubes]}, open(cache_path, "w"), indent=2)
        return tubes

    # ---------- template creation (run once on a clean video) ----------

    def make_template_from_video(self, video_path: str, id_prefix: Optional[str] = None) -> NestTemplate:
        from .assign import _bbox_center
        frame = self._first_frame(video_path)
        dets = self._detect(frame)
        if len(dets) != self.rows * self.cols:
            raise RuntimeError(f"Template requires {self.rows*self.cols} detections; got {len(dets)}")
        centers = np.array([_bbox_center(d.bbox) for d in dets], dtype=np.float32)
        # sort row-major (by y, then x within each row)
        order_y = centers[:,1].argsort()
        dets = [dets[i] for i in order_y]
        centers = centers[order_y]
        chunks = np.array_split(list(zip(dets, centers)), self.rows)
        canonical, ids, n = [], [], 1
        for chunk in chunks:
            r = list(chunk)
            r.sort(key=lambda p: p[1][0])  # by x
            for _, ctr in r:
                canonical.append((float(ctr[0]), float(ctr[1])))
                ids.append((f"{id_prefix}{n}" if id_prefix else str(n)))
                n += 1
        tpl = NestTemplate(rows=self.rows, cols=self.cols, ids=ids, centers=canonical)
        save_template(tpl, self.template_path)
        return tpl

    # ---------- internal helpers ----------

    def _detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        return self.detector.predict(frame_bgr)

    def _first_frame(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Cannot read first frame: {video_path}")
        return frame

    # Optional: periodic revalidation across a long clip (can be called by pipeline)
    def revalidate_over_time(self, video_path: str, tubes: List[Tube]) -> List[Tube]:
        from .util import bbox_centers
        tpl = load_template(self.template_path)
        cap = cv2.VideoCapture(video_path)
        fidx, last_centers = 0, None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if self.revalidate_every > 0 and (fidx % self.revalidate_every == 0):
                dets = self._detect(frame)
                new_tubes = assign_ids(
                    dets, tpl,
                    allow_partial=self.allow_partial,
                    min_visible_fraction=self.min_visible_fraction,
                    method_order=self.method_order,
                    aff_thr=self.aff_thr,
                    hom_thr=self.hom_thr,
                    residual_ok=self.residual_ok,
                    fill_missing=self.fill_missing
                )
                curr = np.array(bbox_centers(new_tubes))
                if last_centers is not None and np.isfinite(curr).all():
                    d = np.linalg.norm(curr - np.array(last_centers), axis=1)
                    if np.nanmean(d) > self.jitter_tol:
                        tubes = new_tubes
                last_centers = curr
            fidx += 1
        cap.release()
        return tubes
