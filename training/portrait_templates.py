"""Hand-slot templates for every card, from the harvested Info portraits.

dataset/cards/<dir>/info.png (+ assets/cards.json for slug/dir) -> assets/cards_all/<slug>.png
The Info popup shows the card with the same art as the hand slot; we cut the same art
window vision::hand uses (ART fractions of the card rect) and scale it to the size of the
existing assets/cards templates.
"""
import json
from pathlib import Path

import cv2

PORTRAIT = {"card": (75, 860, 325, 1165), "champion": (75, 400, 325, 705)}  # x0, y0, x1, y1 on 1080x2400
ART = (0.06, 0.05, 0.94, 0.72)
ref = cv2.imread("../assets/cards/hog_rider.png")
out = Path("../assets/cards_all")
out.mkdir(exist_ok=True)
for c in json.load(open("../assets/cards.json")):
    d = Path("../dataset/cards") / c["dir"]
    layout = (d / "layout.txt").read_text().strip() if (d / "layout.txt").exists() else "card"
    if layout not in PORTRAIT:
        continue  # tower troops are not in the hand
    x0, y0, x1, y1 = PORTRAIT[layout]
    card = cv2.imread(str(d / "info.png"))[y0:y1, x0:x1]
    h, w = card.shape[:2]
    art = card[int(ART[1] * h):int(ART[3] * h), int(ART[0] * w):int(ART[2] * w)]
    cv2.imwrite(str(out / f"{c['slug']}.png"), cv2.resize(art, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA))
print(len(list(out.glob("*.png"))), "templates ->", out)
