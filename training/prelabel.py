"""Write model predictions as pre-labels for images that have no labels yet.

dataset/raw/*.jpg (unlabeled) -> dataset/predictions/*.txt; the labeler loads these so
labeling becomes correcting.
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="runs/arena/weights/best.pt")
ap.add_argument("--dataset", default="../dataset")
ap.add_argument("--conf", type=float, default=0.35)
args = ap.parse_args()

root = Path(args.dataset)
out = root / "predictions"
out.mkdir(exist_ok=True)
todo = [p for p in sorted((root / "raw").iterdir())
        if p.suffix in (".jpg", ".png") and not (root / "labels" / f"{p.stem}.txt").exists()]
model = YOLO(args.weights)
for i in range(0, len(todo), 32):
    for img, r in zip(todo[i:i + 32], model.predict(todo[i:i + 32], conf=args.conf, verbose=False)):
        lines = [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                 for c, (x, y, w, h) in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist())]
        (out / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
print(f"pre-labeled {len(todo)} images -> {out}")
