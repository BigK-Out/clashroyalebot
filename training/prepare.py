"""Build the YOLO dataset from labeled images.

dataset/raw/*.jpg + dataset/labels/*.txt (only images that have a label file, i.e. reviewed)
-> dataset/yolo/{images,labels}/{train,val} (symlinks) + dataset/yolo/data.yaml

Split is by recording time, not random: consecutive frames are near-duplicates, so a random
split would leak training images into validation. The newest ~15% go to val.
"""
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="../dataset")
ap.add_argument("--val", type=float, default=0.15)
args = ap.parse_args()

root = Path(args.dataset).resolve()
classes = [c.strip() for c in (root / "classes.txt").read_text().splitlines() if c.strip()]
pairs = sorted(
    (img, root / "labels" / f"{img.stem}.txt")
    for img in (root / "raw").iterdir()
    if img.suffix in (".jpg", ".png") and (root / "labels" / f"{img.stem}.txt").exists()
)
if len(pairs) < 10:
    raise SystemExit(f"only {len(pairs)} labeled images; label more first")

n_val = max(2, int(len(pairs) * args.val))
splits = {"train": pairs[:-n_val], "val": pairs[-n_val:]}
out = root / "yolo"
for split, items in splits.items():
    for kind in ("images", "labels"):
        d = out / kind / split
        d.mkdir(parents=True, exist_ok=True)
        for old in d.iterdir():
            old.unlink()
    for img, lbl in items:
        (out / "images" / split / img.name).symlink_to(img)
        (out / "labels" / split / lbl.name).symlink_to(lbl)

(out / "data.yaml").write_text(
    f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n"
    + "".join(f"  {i}: {c}\n" for i, c in enumerate(classes))
)
counts = [0] * len(classes)
for _, lbl in pairs:
    for line in lbl.read_text().splitlines():
        if line.strip():
            counts[int(line.split()[0])] += 1
print(f"{len(splits['train'])} train / {len(splits['val'])} val images")
for c, n in zip(classes, counts):
    print(f"  {c:16s} {n}")
