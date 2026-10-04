"""Cut unit crops of each card's own unit out of the harvested Info gameplay previews.

dataset/cards/NNN/clip_XX.jpg + assets/cards.json -> dataset/clip_units/<slug>/<NNN>_<XX>_<k>.png
(64x64, same framing and resolution as vision::units::body_rect crops from the 576-px feed),
plus dataset/clip_units/index.csv and a contact sheet per card for review.

In the preview the card's unit is the ally (blue tag); enemies wear red tags and are skipped.
The preview is ~1.11x the arena scale (level-tag digits are 23 px tall there vs ~20.8 px in
game, both on the 1080-px screen), so body_rect's tile-based box is scaled by that.

Frames where the own unit has no tag (just deployed, or the preview hides it) fall back to
frame differencing: moving pixels against the clip's per-pixel median, minus everything near
a tag, the emote and the page dots; the largest remaining blob is taken if it is unit-sized.
"""
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("../dataset")
OUT = ROOT / "clip_units"
SIZE = 64
PREVIEW_SCALE = 23.2 / 20.8
TILE_W = 1032 / 18 * PREVIEW_SCALE          # arena 24..1056 px wide, 18 columns (1080 screen)
TILE_H = (1083.5 - 303.5) / 17 * PREVIEW_SCALE
FEED_SCALE = 576 / 1080 / PREVIEW_SCALE     # preview px -> 576-px feed px
EMOTE = (680, 0, 920, 260)                  # x0, y0, x1, y1: King emote, top right
DOTS = (330, 480, 580, 565)                 # page indicator, bottom center
INNER = (28, 16, 892, 500)                  # inside the preview's rounded frame


def masks(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL).astype(np.float32)
    h, s, v = hsv[..., 0] * 360 / 256, hsv[..., 1] / 255, hsv[..., 2] / 255
    white = (s < 0.3) & (v > 0.8)
    ally = (h >= 185) & (h <= 235) & (s > 0.4) & (v > 0.5)
    enemy = ((h >= 290) | (h <= 8)) & (s > 0.55) & (v > 0.3)
    yellow = (h >= 38) & (h <= 60) & (s > 0.5) & (v > 0.7)
    return white, ally, enemy, yellow


def tags(img):
    """Level tags as (team, x0, y0, x1, y1), like vision::units::detect_units at preview scale."""
    white, ally, enemy, yellow = masks(img)
    n, _, st, _ = cv2.connectedComponentsWithStats(white.astype(np.uint8), connectivity=8)
    digits = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (17 <= h <= 30 and w <= 22 and a >= 16):
            continue
        for team, m in (("A", ally), ("E", enemy)):
            top, bot = m[max(0, y - 6):y, max(0, x - 1):x + w + 1], m[y + h:y + h + 6, max(0, x - 1):x + w + 1]
            if top.size and bot.size and top.mean() >= 0.35 and bot.mean() >= 0.35:
                digits.append([team, x, y, x + w, y + h, 1])
                break
    out = []
    for d in sorted(digits, key=lambda d: (d[0], d[2], d[1])):
        t = next((t for t in out if t[0] == d[0] and abs(t[2] - d[2]) <= 8 and d[1] <= t[3] + 15 and d[3] + 15 >= t[1]), None)
        if t:
            t[1:5] = [min(t[1], d[1]), min(t[2], d[2]), max(t[3], d[3]), max(t[4], d[4])]
            t[5] += 1
        else:
            out.append(list(d))
    keep = []
    for team, x0, y0, x1, y1, k in out:
        crown = yellow[max(0, y0 - 8):y1 + 8, max(0, x0 - 45):x0]
        if k <= 2 and (crown.size == 0 or crown.mean() < 0.12):  # towers: 3-4 digit HP + crown
            keep.append((team, x0, y0, x1, y1))
    return keep


def body_box(tag):
    """vision::units::body_rect for a tag, in preview pixels."""
    _, x0, y0, x1, y1 = tag
    fx = x1 + 0.02 * 1080 * PREVIEW_SCALE
    return int(fx - 1.3 * TILE_W), int(y1), int(fx + 1.3 * TILE_W), int(y1 + 2.4 * TILE_H)


def from_feet(cx, top):
    """Body box for an untagged unit whose sprite starts at `top` (the tag would sit just above)."""
    fx = cx
    return int(fx - 1.3 * TILE_W), int(top - 0.2 * TILE_H), int(fx + 1.3 * TILE_W), int(top + 2.2 * TILE_H)


def cut(img, box):
    """Crop (padded with edge pixels), down to feed resolution, then to SIZE like the runtime."""
    x0, y0, x1, y1 = box
    H, W = img.shape[:2]
    pad = 200
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    c = big[y0 + pad:y1 + pad, x0 + pad:x1 + pad]
    fw, fh = max(8, round(c.shape[1] * FEED_SCALE)), max(8, round(c.shape[0] * FEED_SCALE))
    c = cv2.resize(c, (fw, fh), interpolation=cv2.INTER_AREA)
    return cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)


def aligned_diff(a, b):
    """Pixels that moved between frames a and b after undoing the camera pan (b -> a)."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), resp = cv2.phaseCorrelate(ga, gb)
    if resp < 0.2:
        return None  # scene cut: nothing to compare against
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    bw = cv2.warpAffine(b, M, (b.shape[1], b.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    d = np.abs(cv2.GaussianBlur(a, (5, 5), 0).astype(np.int16) - cv2.GaussianBlur(bw, (5, 5), 0).astype(np.int16)).max(axis=2) > 45
    m = int(np.ceil(max(abs(dx), abs(dy)))) + 4  # edges warped in from outside
    d[:m], d[-m:], d[:, :m], d[:, -m:] = False, False, False, False
    return d


def moving_blobs(frames, i, tag_boxes):
    """Moving blobs in frame i that no tag accounts for (own untagged units): [(cx, top)]."""
    ds = [aligned_diff(frames[i], frames[j]) for j in (i - 1, i + 1) if 0 <= j < len(frames)]
    ds = [d for d in ds if d is not None]
    if not ds:
        return []
    diff = np.logical_and.reduce(ds) if len(ds) > 1 else ds[0]  # in both: where the unit is now
    for x0, y0, x1, y1 in [EMOTE, DOTS] + tag_boxes:
        diff[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = False
    inner = np.zeros_like(diff)
    inner[INNER[1]:INNER[3], INNER[0]:INNER[2]] = True
    diff &= inner
    diff = cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(diff, connectivity=8)
    blobs = []
    for j in range(1, n):
        x, y, w, h, a = st[j]
        if 0.25 * TILE_W <= w <= 2.5 * TILE_W and 0.3 * TILE_H <= h <= 3 * TILE_H and a > 250:
            blobs.append((a, x + w / 2, y))
    return [(cx, top) for _, cx, top in sorted(blobs, reverse=True)[:3]]


def main():
    cards = json.load(open("../assets/cards.json"))
    by_name = {c["name"]: c for c in cards}
    OUT.mkdir(parents=True, exist_ok=True)
    rows, per_card = [], {}
    for d in sorted(p for p in (ROOT / "cards").iterdir() if p.is_dir()):
        nf = d / "name.txt"
        card = by_name.get(nf.read_text().strip()) if nf.exists() else None
        if card is None or card.get("type") not in ("Troop", "Building"):
            continue
        clips = sorted(d.glob("clip_*.jpg"))
        frames = [cv2.imread(str(p)) for p in clips]
        out = OUT / card["slug"]
        out.mkdir(exist_ok=True)
        for i, (p, img) in enumerate(zip(clips, frames)):
            ts = tags(img)
            # Mask a whole body box around every tag: those units are handled (or skipped) by tag.
            boxes = [body_box(t) for t in ts]
            tag_boxes = [(b[0], t[2] - 20, b[2], b[3]) for t, b in zip(ts, boxes)]
            got = [(b, "tag") for t, b in zip(ts, boxes) if t[0] == "A"]
            if not got:
                got = [(from_feet(*m), "diff") for m in moving_blobs(frames, i, tag_boxes)]
            for k, (box, how) in enumerate(got):
                name = f"{d.name}_{p.stem[-2:]}_{k}.png"
                cv2.imwrite(str(out / name), cut(img, box))
                rows.append({"file": f"{card['slug']}/{name}", "card": card["slug"], "dir": d.name, "how": how})
                per_card.setdefault(card["slug"], []).append((out / name, how))
    with open(OUT / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "card", "dir", "how"])
        w.writeheader()
        w.writerows(rows)
    # Contact sheets: one row per card, tag crops then diff crops (diff crops get a red corner).
    sheet_rows = []
    for slug, items in sorted(per_card.items()):
        tiles = []
        for path, how in sorted(items, key=lambda x: x[1] != "tag")[:16]:
            t = cv2.imread(str(path))
            if how == "diff":
                t[:6, :6] = (0, 0, 255)
            tiles.append(t)
        tiles += [np.zeros((SIZE, SIZE, 3), np.uint8)] * (16 - len(tiles))
        label = np.zeros((SIZE, 150, 3), np.uint8)
        cv2.putText(label, slug[:18], (2, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        sheet_rows.append(np.hstack([label] + tiles))
    for s in range(0, len(sheet_rows), 20):
        cv2.imwrite(str(OUT / f"sheet_{s // 20}.png"), np.vstack(sheet_rows[s:s + 20]))
    n_tag = sum(r["how"] == "tag" for r in rows)
    print(f"{len(per_card)} cards, {len(rows)} crops ({n_tag} by tag, {len(rows) - n_tag} by frame diff) -> {OUT}")


if __name__ == "__main__":
    main()
