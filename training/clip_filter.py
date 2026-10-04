"""Drop scenery from the frame-difference clip crops (clip_crops.py).

The previews all play in the same arena, so cliffs, rocks, bridges and towers show up in every
card's clips while a card's own unit only shows up in its own. A diff crop is scenery when it
closely matches some random crop from *another* card's clips (ImageNet ResNet-18 features,
cosine). Tag crops are kept as they are.

dataset/clip_units/index.csv -> adds a "keep" column (1/0) and "bg" (best background match);
writes dataset/clip_units/filtered_*.png contact sheets (kept crops only).
"""
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

from clip_crops import SIZE, TILE_H, TILE_W, cut

ROOT = Path("../dataset")
OUT = ROOT / "clip_units"
BG_PER_DIR = 40
THRESH = 0.80

dev = "cuda" if torch.cuda.is_available() else "cpu"
net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
net.fc = torch.nn.Identity()
net.eval().to(dev)
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


@torch.no_grad()
def embed(imgs):
    out = []
    for i in range(0, len(imgs), 256):
        x = torch.from_numpy(np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs[i:i + 256]]))
        x = x.permute(0, 3, 1, 2).float().to(dev) / 255
        x = torch.nn.functional.interpolate(x, size=112, mode="bilinear", align_corners=False)
        f = net((x - mean) / std)
        out.append(torch.nn.functional.normalize(f, dim=1).cpu())
    return torch.cat(out)


def main():
    rows = list(csv.DictReader(open(OUT / "index.csv")))
    rng = random.Random(0)
    # Background bank: random unit-sized boxes from every harvested clip.
    bank, bank_dir = [], []
    w, h = int(2.6 * TILE_W), int(2.4 * TILE_H)
    for d in sorted(p for p in (ROOT / "cards").iterdir() if p.is_dir()):
        clips = sorted(d.glob("clip_*.jpg"))
        for _ in range(BG_PER_DIR if clips else 0):
            img = cv2.imread(str(rng.choice(clips)))
            x, y = rng.randint(28, 892 - w), rng.randint(16, 500 - h)
            bank.append(cut(img, (x, y, x + w, y + h)))
            bank_dir.append(d.name)
    fb = embed(bank)
    crops = [cv2.imread(str(OUT / r["file"])) for r in rows]
    fc = embed(crops)
    sim = fc @ fb.T                                       # crops x bank
    dir_of_card = {}
    for r in rows:
        dir_of_card.setdefault(r["card"], set()).add(r["dir"])
    bank_dir = np.array(bank_dir)
    for i, r in enumerate(rows):
        own = np.isin(bank_dir, list(dir_of_card[r["card"]]))  # same card (also its duplicate dirs)
        s = sim[i].clone()
        s[torch.from_numpy(own)] = -1
        r["bg"] = f"{s.max().item():.3f}"
        r["keep"] = int(r["how"] == "tag" or s.max().item() < THRESH)
    with open(OUT / "index.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    per = {}
    for r, im in zip(rows, crops):
        per.setdefault(r["card"], []).append((r, im))
    sheet = []
    for card, items in sorted(per.items()):
        kept = [im for r, im in items if r["keep"]]
        dropped = [im for r, im in items if not r["keep"]]
        tiles = (kept[:12] + [np.zeros((SIZE, SIZE, 3), np.uint8)] * 12)[:12]
        tiles += [np.full((SIZE, 4, 3), 255, np.uint8)]
        tiles += [cv2.addWeighted(im, 0.5, np.zeros_like(im), 0.5, 0) for im in dropped[:4]]
        tiles += [np.zeros((SIZE, SIZE, 3), np.uint8)] * (4 - min(4, len(dropped)))
        label = np.zeros((SIZE, 150, 3), np.uint8)
        cv2.putText(label, card[:18], (2, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(label, f"{len(kept)} kept/{len(items)}", (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        sheet.append(np.hstack([label] + tiles))
    for s in range(0, len(sheet), 20):
        cv2.imwrite(str(OUT / f"filtered_{s // 20}.png"), np.vstack(sheet[s:s + 20]))
    k = sum(int(r["keep"]) for r in rows)
    print(f"kept {k}/{len(rows)} crops (all tag crops + diff crops with background match < {THRESH})")


if __name__ == "__main__":
    main()
