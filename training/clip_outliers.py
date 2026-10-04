"""Clean the tag crops from clip_crops.py for training: per card, drop crops that don't look
like the rest of that card's crops (other allied troops in the preview, e.g. the Goblins in
Heal Spirit's clip).

dataset/clip_units/index.csv (how == "tag") -> dataset/clip_units/clean.csv (file, card)
plus dataset/clip_units/outliers.png (dropped crops, one row per card) for review.
A crop is dropped when its ImageNet ResNet-18 feature is less similar to the card's mean
feature than MIN_SIM, or than the card's median similarity minus MAD_K robust deviations.
"""
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from clip_filter import embed

OUT = Path("../dataset/clip_units")
MIN_SIM = 0.55
MAD_K = 3.0
# Previews where the card's own unit is not what the blue tags mostly mark: Heal Spirit's
# clip is a Goblin push (the spirit itself is tiny and short-lived).
SKIP = {"heal_spirit"}


def main():
    rows = [r for r in csv.DictReader(open(OUT / "index.csv")) if r["how"] == "tag"]
    per = {}
    for r in rows:
        if r["card"] in SKIP:
            continue
        per.setdefault(r["card"], []).append(r)
    keep, dropped = [], {}
    for card, rs in sorted(per.items()):
        imgs = [cv2.imread(str(OUT / r["file"])) for r in rs]
        if len(rs) < 4:  # too few to tell what the card looks like
            keep += [(r["file"], card) for r in rs]
            continue
        f = embed(imgs)
        mean = torch.nn.functional.normalize(f.mean(0), dim=0)
        sim = (f @ mean).numpy()
        med = np.median(sim)
        mad = np.median(np.abs(sim - med)) * 1.4826 + 1e-6
        ok = (sim >= MIN_SIM) & (sim >= med - MAD_K * mad)
        keep += [(r["file"], card) for r, k in zip(rs, ok) if k]
        if (~ok).any():
            dropped[card] = [im for im, k in zip(imgs, ok) if not k]
    with open(OUT / "clean.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "card"])
        w.writerows(keep)
    sheet = []
    for card, ims in dropped.items():
        label = np.zeros((64, 150, 3), np.uint8)
        cv2.putText(label, f"{card[:14]} -{len(ims)}", (2, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        tiles = (ims + [np.zeros((64, 64, 3), np.uint8)] * 12)[:12]
        sheet.append(np.hstack([label] + tiles))
    if sheet:
        cv2.imwrite(str(OUT / "outliers.png"), np.vstack(sheet))
    n_drop = sum(len(v) for v in dropped.values())
    print(f"kept {len(keep)}/{len(rows)} tag crops ({n_drop} outliers from {len(dropped)} cards) -> {OUT / 'clean.csv'}")


if __name__ == "__main__":
    main()
