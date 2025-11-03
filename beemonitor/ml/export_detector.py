from __future__ import annotations
import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--fmt", default="onnx", choices=["onnx","torchscript","engine","openvino","coreml"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true")
    args = ap.parse_args()

    model = YOLO(args.weights)
    path = model.export(format=args.fmt, imgsz=args.imgsz, half=args.half)
    print(f"[OK] Exported to: {path}")

if __name__ == "__main__":
    main()
