"""Fine-tune YOLOv8n on the arena dataset (run prepare.py first)."""
import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../dataset/yolo/data.yaml")
ap.add_argument("--model", default="yolov8n.pt", help="start weights (pretrained COCO, or a previous best.pt)")
ap.add_argument("--epochs", type=int, default=150)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--name", default="arena")
args = ap.parse_args()

YOLO(args.model).train(
    data=args.data,
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=16,
    patience=40,
    project="runs",
    name=args.name,
    exist_ok=True,
    # The arena is never mirrored vertically (enemy is always on top); left/right is symmetric.
    flipud=0.0,
    fliplr=0.5,
    # Team is encoded by color (red/blue health bars), so keep hue shifts small.
    hsv_h=0.005,
    mosaic=1.0,
    device=0,
)
