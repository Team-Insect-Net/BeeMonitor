from __future__ import annotations
import os
from typing import List, Tuple
from .schema import DatasetIndex, FrameAnn

def _norm_box(b, w, h):
    x1,y1,x2,y2 = b
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh

def export_bee_yolo(idx: DatasetIndex, out_dir: str, split: str = "train", bee_label="bee"):
    """
    Export frame-level bee boxes to YOLO format:
      out_dir/
        images/{split}/{video_id}_{frame:06d}.jpg
        labels/{split}/{video_id}_{frame:06d}.txt
    """
    import cv2, json
    os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)
    for vid in idx.splits.get(split, []):
        rec = idx.videos[vid]
        cap = cv2.VideoCapture(rec.meta.path)
        if not cap.isOpened():
            print(f"[WARN] cannot open {rec.meta.path}")
            continue
        # optional: if you stored per-frame anns in a separate json
        has_frame_anns = len(rec.frame_anns) > 0
        frame_map = {fa.frame: fa for fa in rec.frame_anns}
        fidx = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            img_name = f"{vid}_{fidx:06d}.jpg"
            img_path = os.path.join(out_dir, "images", split, img_name)
            lbl_path = os.path.join(out_dir, "labels", split, img_name.replace(".jpg",".txt"))
            cv2.imwrite(img_path, frame)

            H, W = frame.shape[:2]
            lines: List[str] = []
            if has_frame_anns and fidx in frame_map:
                ann = frame_map[fidx]
                for b, lab in zip(ann.bboxes, ann.labels):
                    if lab != bee_label: continue
                    cx, cy, bw, bh = _norm_box(b, W, H)
                    # class id 0 = bee
                    lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            with open(lbl_path, "w") as f:
                f.write("\n".join(lines))
            fidx += 1
        cap.release()

def write_yolo_data_yaml(out_dir: str, dataset_root: str, nc=1, names=None):
    """
    Generates a simple dataset YAML for ultralytics (bee-only).
    """
    import yaml, os
    names = names or ["bee"]
    data = {
        "path": os.path.abspath(dataset_root),
        "train": os.path.join(out_dir, "images", "train"),
        "val":   os.path.join(out_dir, "images", "val"),
        "test":  os.path.join(out_dir, "images", "test"),
        "nc": nc,
        "names": names
    }
    with open(os.path.join(out_dir, "bee.data.yaml"), "w") as f:
        yaml.safe_dump(data, f)
