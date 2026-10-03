"""Export trained weights to ONNX for the Rust runtime (ort)."""
import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="runs/arena/weights/best.pt")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--out", default="../assets/models/arena.onnx")
args = ap.parse_args()

# Fixed input (1x3x640x640), no NMS in-graph: Rust does letterbox + NMS.
path = YOLO(args.weights).export(format="onnx", imgsz=args.imgsz, opset=17, dynamic=False, simplify=True)
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(path, out)
print(f"wrote {out}")
