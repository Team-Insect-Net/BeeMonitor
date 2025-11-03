from __future__ import annotations
from typing import List, Optional, Iterable, Dict, Any
import numpy as np

from beemonitor.types import Detection
from .base import Detector

def _as_int_list(v) -> Optional[List[int]]:
    if v is None:
        return None
    return [int(x) for x in v]

class YoloDetector(Detector):
    """
    Ultralytics YOLOv8/v5 wrapper with robust parsing + optional post-class filtering.
    """

    def __init__(
        self,
        weights: str,
        conf: float = 0.25,
        iou: float = 0.45,
        classes: Optional[List[int]] = None,
        device: Optional[str] = None,
        imgsz: Optional[int] = None,
        verbose: bool = False,
        apply_classes_filter_post: bool = True,  # filter AFTER parsing (safer)
    ):
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Ultralytics is required for YoloDetector. Install with `pip install ultralytics`."
            ) from e

        self._model = YOLO(weights)
        self._conf = float(conf)
        self._iou = float(iou)
        self._device = device or "cpu"
        self._imgsz = int(imgsz) if imgsz else None
        self._verbose = bool(verbose)

        # Keep the original classes for optional post-filtering
        self._classes_cfg = _as_int_list(classes)
        self._apply_classes_filter_post = bool(apply_classes_filter_post)

    def _run_raw(self, frame_bgr: np.ndarray):
        # Use BGR as-is (Ultralytics handles numpy BGR/RGB internally). If you prefer RGB:
        # frame_rgb = frame_bgr[..., ::-1]
        kwargs: Dict[str, Any] = {
            "conf": self._conf,
            "iou": self._iou,
            "device": self._device,
            "verbose": False,
        }
        if self._imgsz:
            kwargs["imgsz"] = self._imgsz

        # IMPORTANT: do NOT pass classes here if we're debugging/uncertain.
        # We'll filter after parsing so we can see what YOLO actually produced.
        return self._model.predict(frame_bgr, **kwargs)

    def predict(self, frame_bgr: np.ndarray) -> List[Detection]:
        if frame_bgr is None or frame_bgr.size == 0:
            if self._verbose:
                print("[yolo] empty frame")
            return []

        H, W = frame_bgr.shape[:2]
        results = self._run_raw(frame_bgr)

        if self._verbose:
            print(f"[yolo] frame size={W}x{H} results_len={len(results)}")

        dets: List[Detection] = []
        if not results:
            if self._verbose:
                print("[yolo] no results list")
            return dets

        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            if self._verbose:
                print("[yolo] 0 boxes in r.boxes")
            return dets

        # Tensors → numpy
        xyxy = boxes.xyxy
        clsv = boxes.cls
        confv = boxes.conf

        if hasattr(xyxy, "cpu"):
            xyxy = xyxy.cpu().numpy()
            clsv = clsv.cpu().numpy().astype(int)
            confv = confv.cpu().numpy().astype(float)
        else:
            xyxy = np.asarray(xyxy)
            clsv = np.asarray(clsv, dtype=int)
            confv = np.asarray(confv, dtype=float)

        n = xyxy.shape[0]
        if self._verbose:
            uniq, cnt = np.unique(clsv, return_counts=True)
            hist = {int(u): int(c) for u, c in zip(uniq.tolist(), cnt.tolist())}
            print(f"[yolo] parsed boxes={n} by_class={hist}")

        # Optional post-class filtering (safer for debugging)
        keep_classes = set(self._classes_cfg) if (self._apply_classes_filter_post and self._classes_cfg) else None

        for i in range(n):
            cls_id = int(clsv[i])
            if keep_classes is not None and cls_id not in keep_classes:
                continue
            x1, y1, x2, y2 = map(float, xyxy[i].tolist())
            conf = float(confv[i])
            dets.append(Detection(
                frame=-1,                 # caller can overwrite with real frame index
                bbox=(x1, y1, x2, y2),
                conf=conf,
                cls=cls_id                # IMPORTANT: keep as int
            ))

        if self._verbose:
            print(f"[yolo] converted {len(dets)} detections (after post-filter)")
        return dets
