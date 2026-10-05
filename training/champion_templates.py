"""Hand templates for champions, cut from recorded hands.

Champions sit in the hand in a hexagonal frame that their Info portraits do not match, so
sparring never recognized (and never played) them. In a deck with one champion, the hand
slot sparring logged as unknown (null) is almost always that champion: every other card is
recognized when it is affordable. For each such slot just before a logged play, cut the art
from the aligned video, keep colored (affordable) crops, and save the medoid (the crop most
similar to the others) as assets/cards_all/<slug>@hand.png, plus a contact sheet to check.

  cd training && .venv/bin/python champion_templates.py
"""
import json
from pathlib import Path

import cv2
import numpy as np

from selfplay.video import PHONES, load_calib

ROOT = Path(__file__).parent.parent
ART = (0.06, 0.05, 0.94, 0.72)  # vision::hand art window of a card slot
SIZE = (88, 83)  # (w, h) of the existing hand templates
cards = json.load(open(ROOT / "assets/cards.json"))
champs = {c["slug"] for c in cards if c.get("rarity") == "Champion"}


def slot_art(frame, s):
    h, w = frame.shape[:2]
    x0, y0, sw, sh = s["x"] * w, s["y"] * h, s["w"] * w, s["h"] * h
    art = frame[int(y0 + ART[1] * sh):int(y0 + ART[3] * sh), int(x0 + ART[0] * sw):int(x0 + ART[2] * sw)]
    return cv2.resize(art, SIZE, interpolation=cv2.INTER_AREA)


def colored(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean() > 60


def ncc(a, b):
    a, b = a.astype(np.float32).ravel(), b.astype(np.float32).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


crops = {c: [] for c in champs}
for meta_path in sorted((ROOT / "dataset/selfplay").glob("17*/meta.json")):
    m = meta_path.parent
    meta = json.load(open(meta_path))
    if not (m / "align.json").exists():
        continue
    align = json.load(open(m / "align.json"))
    start = int((m / "recording_started_ms").read_text())
    for phone, deck in (("note9", meta.get("sparring_deck", [])), ("note14", meta.get("observer_deck", []))):
        champ = [c for c in deck if c in champs]
        if len(champ) != 1 or not align.get(phone) or len(crops[champ[0]]) >= 40:
            continue
        cal, h = PHONES[phone]
        slots = load_calib(cal)["card_slots"]
        cap = cv2.VideoCapture(str(m / f"{phone}.mkv"))
        for line in open(m / f"plays_{phone}.jsonl"):
            p = json.loads(line)
            for i, name in enumerate(p["hand_before"]):
                if name is not None:
                    continue
                cap.set(cv2.CAP_PROP_POS_MSEC, p["host_ms"] - start + align[phone]["offset_ms"] - 400)
                ok, f = cap.read()
                if not ok:
                    continue
                art = slot_art(cv2.resize(f, (576, h), interpolation=cv2.INTER_AREA), slots[i])
                if colored(art):
                    crops[champ[0]].append(art)

sheet = []
for c in sorted(champs):
    cs = crops[c]
    if len(cs) < 3:
        print(f"{c}: only {len(cs)} crops, skipped")
        continue
    sim = np.array([[ncc(a, b) for b in cs] for a in cs])
    med = int(np.argmax(np.median(sim, 1)))
    agree = float(np.mean(sim[med] > 0.6))
    cv2.imwrite(str(ROOT / f"assets/cards_all/{c}@hand.png"), cs[med])
    print(f"{c}: {len(cs)} crops, medoid agrees with {agree:.0%}")
    row = [cs[med]] + [cs[k] for k in np.argsort(-sim[med])[1:6]]
    row += [np.zeros_like(cs[med])] * (6 - len(row))
    sheet.append(np.concatenate(row, 1))
if sheet:
    cv2.imwrite(str(Path(__file__).parent / "runs/champion_sheet.png"), np.concatenate(sheet, 0))
