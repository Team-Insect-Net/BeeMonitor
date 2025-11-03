from __future__ import annotations
import argparse, os
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data yaml (e.g., bee.data.yaml)")
    ap.add_argument("--model", default="yolov8n.pt", help="base model to finetune")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help='"0", "cuda:0", "mps", or "cpu"')
    ap.add_argument("--project", default="runs/detect", help="Ultralytics output root")
    ap.add_argument("--name", default="beemonitor-train")
    ap.add_argument("--patience", type=int, default=20)
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience
    )

if __name__ == "__main__":
    main()
