"""Harvest one tower troop's Info screen (Decks tab -> tower troop slot -> tower collection).

There are only four tower troops and their menu positions move with the screen, so this is
driven by hand: open the troop's menu yourself (or pass --tap), then give the Info button.

  .venv/bin/python harvest_towers.py --info 146,1936 [--tap 146,1700]

-> dataset/cards/<NNN>/ like harvest_cards.py, with layout.txt = "tower":
  info.png (page 1: rarity, type), clip_XX.jpg (page 2 gameplay preview), stats.png (page 3).
Never taps "Use" (that would change the deck's tower troop); closes the popup with its X.
"""
import argparse
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--serial", default="4xwskfkr7xp7w4xo")
ap.add_argument("--adb", default=str(Path.home() / "Android/Sdk/platform-tools/adb"))
ap.add_argument("--out", default="../dataset/cards")
ap.add_argument("--clip-frames", type=int, default=12)
ap.add_argument("--tap", help="x,y of the troop card, to open its menu first")
ap.add_argument("--info", required=True, help="x,y of the menu's Info button")
args = ap.parse_args()

CLOSE_X = (978, 414)
PREVIEW = (80, 1265, 1000, 1830)
TITLE = (480, 580, 300, 1000)


def adb(*a):
    return subprocess.run([args.adb, "-s", args.serial, *a], capture_output=True, check=True).stdout


def shot():
    return cv2.imdecode(np.frombuffer(adb("exec-out", "screencap", "-p"), np.uint8), cv2.IMREAD_COLOR)


def tap(x, y):
    adb("shell", "input", "tap", str(x), str(y))


def popup_open(img):
    x, y = CLOSE_X
    box = img[y - 30:y + 30, x - 30:x + 30].astype(int)
    return ((box[..., 2] > 200) & (box[..., 1] < 110) & (box[..., 0] < 120)).sum() >= 40


def next_page():
    adb("shell", "input", "swipe", "850", "1550", "250", "1550", "250")
    time.sleep(1.2)
    img = shot()
    if not popup_open(img):
        raise SystemExit("popup closed by page swipe; stopping")
    return img


def main():
    import easyocr
    if args.tap:
        tap(*map(int, args.tap.split(",")))
        time.sleep(1.0)
    tap(*map(int, args.info.split(",")))
    time.sleep(1.5)
    img = shot()
    if not popup_open(img):
        raise SystemExit("tower troop Info popup did not open")
    y0, y1, x0, x1 = TITLE
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    name = " ".join(t for t in reader.readtext(img[y0:y1, x0:x1], detail=0) if not t.lower().startswith("level")).title()
    out = Path(args.out)
    for d in out.iterdir():
        if (d / "name.txt").exists() and (d / "name.txt").read_text().strip() == name:
            tap(*CLOSE_X)
            raise SystemExit(f"{name} already harvested in {d}")
    d = out / f"{len([p for p in out.iterdir() if p.is_dir()]):03d}"
    d.mkdir(parents=True)
    cv2.imwrite(str(d / "info.png"), img)
    (d / "name.txt").write_text(name)
    (d / "layout.txt").write_text("tower")
    next_page()
    x0, y0, x1, y1 = PREVIEW
    for k in range(args.clip_frames):
        cv2.imwrite(str(d / f"clip_{k:02d}.jpg"), shot()[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(d / "stats.png"), next_page())
    tap(*CLOSE_X)
    time.sleep(1.0)
    print(f"{name} -> {d}")


main()
