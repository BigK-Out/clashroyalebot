"""Harvest every card's Info screen from the in-game Collection (phone must be on the
Collection tab, scrolled to the top, sorted By Elixir).

Per card -> dataset/cards/<NNN>/
  clip_XX.jpg   page 1: gameplay preview (unit sprite walking in the arena), ~12 frames
  info.png      page 1 full screenshot (name, rarity, type, elixir badge)
  stats.png     page 2 full screenshot (damage, hitpoints, speed, ...)
Page 3 (flavor text) is skipped.

Never sends BACK (that opens "Exit Clash Royale?"); the Info popup is closed with its X.
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
ap.add_argument("--max-cards", type=int, default=1000)
ap.add_argument("--clip-frames", type=int, default=12)
args = ap.parse_args()

COLS = [146, 408, 670, 931]          # card centers (1080x2400 screen)
BAND_DX = -46                         # sample the purple "Max" band left of its text
CARD_ABOVE_BAND = 190                 # card art center is this far above the band center
VISIBLE = (800, 1550)                 # card centers whose menu opens below the card
CLOSE_X = (966, 840)                  # Info popup close button
PREVIEW = (80, 1265, 1000, 1830)      # page-1 gameplay preview box
PAGE_SWIPE_Y = 1550


def adb(*a):
    return subprocess.run([args.adb, "-s", args.serial, *a], capture_output=True, check=True).stdout


def shot():
    return cv2.imdecode(np.frombuffer(adb("exec-out", "screencap", "-p"), np.uint8), cv2.IMREAD_COLOR)


def tap(x, y):
    adb("shell", "input", "tap", str(x), str(y))


def swipe(x0, y0, x1, y1, ms):
    adb("shell", "input", "swipe", str(x0), str(y0), str(x1), str(y1), str(ms))


def runs(mask, min_len):
    out, s = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and s is None:
            s = i
        if not m and s is not None:
            if i - s >= min_len:
                out.append((s, i))
            s = None
    return out


def card_centers_any(img):
    """Card bands anywhere on screen (sanity check that we're on the Collection grid)."""
    out = []
    for x in COLS:
        col = img[:, x + BAND_DX].astype(int)
        b, g, r = col[:, 0], col[:, 1], col[:, 2]
        purple = (r > 120) & (r < 215) & (g > 70) & (g < 150) & (b > 175) & (b > r + 15)
        out += [(x, (s + e) // 2) for s, e in runs(purple, 18)]
    return out


def card_centers(img):
    """(x, y) of every card whose purple Max band is visible, by column."""
    found = []
    for x in COLS:
        col = img[:, x + BAND_DX].astype(int)
        b, g, r = col[:, 0], col[:, 1], col[:, 2]
        purple = (r > 120) & (r < 215) & (g > 70) & (g < 150) & (b > 175) & (b > r + 15)
        for s, e in runs(purple, 18):
            y = (s + e) // 2 - CARD_ABOVE_BAND
            if VISIBLE[0] <= y <= VISIBLE[1]:
                found.append((x, y))
    return found


def find_info_button(img, x, y):
    """Center y of the light-blue Info button below a card at (x, y), or None."""
    col = img[:, x - 70].astype(int)  # left of the white "Info" text
    b, g, r = col[:, 0], col[:, 1], col[:, 2]
    blue = (b > 235) & (g > 140) & (g < 205) & (r > 40) & (r < 130)
    lo, hi = y + 100, min(len(col), y + 400)
    rs = [(s + lo, e + lo) for s, e in runs(blue[lo:hi], 2)]
    merged = []
    for s, e in rs:
        if merged and s - merged[-1][1] < 15:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    cands = [(s, e) for s, e in merged if 50 <= e - s <= 160]
    return (cands[0][0] + cands[0][1]) // 2 if cands else None


def popup_open(img):
    """Info popup is up when its red close X is there."""
    x, y = CLOSE_X
    box = img[y - 30:y + 30, x - 30:x + 30].astype(int)
    red = (box[..., 2] > 200) & (box[..., 1] < 110) & (box[..., 0] < 120)
    return red.sum() >= 40


def on_collection(img):
    return len(card_centers_any(img)) > 0


def signature(img, x, y):
    """Card art in the grid, small + normalized: identifies a card wherever it is on screen."""
    art = cv2.cvtColor(img[y - 140:y + 60, x - 95:x + 95], cv2.COLOR_BGR2GRAY)
    v = cv2.resize(art, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    return (v - v.mean()) / (v.std() + 1e-6)


def seen_before(sig, seen):
    return any(float(sig @ s) / len(sig) > 0.9 for s in seen)


def scroll_shift(a, b):
    """Content moved up by this many px between screenshots a and b (phase correlation)."""
    ga = cv2.cvtColor(a[700:2150], cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b[700:2150], cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(ga, gb)
    return dy


def harvest(idx, x, y, out):
    tap(x, y)
    time.sleep(0.8)
    iy = find_info_button(shot(), x, y)
    if iy is None:
        print(f"  card {idx}: no Info button found at ({x},{y})")
        tap(x, y)  # close the menu
        time.sleep(0.5)
        return False
    tap(x, iy)
    time.sleep(1.2)
    img = shot()
    if not popup_open(img):
        raise SystemExit(f"card {idx}: Info popup did not open; stopping (no blind swipes)")
    d = out / f"{idx:03d}"
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "info.png"), img)
    x0, y0, x1, y1 = PREVIEW
    for k in range(args.clip_frames):
        cv2.imwrite(str(d / f"clip_{k:02d}.jpg"), shot()[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    swipe(850, PAGE_SWIPE_Y, 250, PAGE_SWIPE_Y, 250)
    time.sleep(1.0)
    img = shot()
    if not popup_open(img):
        raise SystemExit(f"card {idx}: popup closed by page swipe; stopping")
    cv2.imwrite(str(d / "stats.png"), img)
    tap(*CLOSE_X)
    time.sleep(0.8)
    if not on_collection(shot()):
        raise SystemExit(f"card {idx}: not back on the Collection grid; stopping")
    return True


def main():
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen = []           # grid-art signatures of cards already handled
    idx = len([p for p in out.iterdir() if p.is_dir()])
    t0 = time.time()
    while idx < args.max_cards:
        # Re-scan until this screen is exhausted: a just-closed card can be highlighted
        # (band color off) for a moment, so one pass can miss it.
        for _ in range(3):
            img = shot()
            if not on_collection(img):
                raise SystemExit("not on the Collection grid; open Collection, scroll to top, rerun")
            todo = []
            for x, y in sorted(card_centers(img), key=lambda c: (c[1], c[0])):
                sig = signature(img, x, y)
                if not seen_before(sig, seen):
                    todo.append((x, y, sig))
            if not todo:
                break
            for x, y, sig in todo:
                if idx >= args.max_cards:
                    break
                ok = harvest(idx, x, y, out)
                seen.append(sig)
                if ok:
                    print(f"card {idx:3d} at ({x},{y})  [{time.time() - t0:.0f}s]")
                    idx += 1
            time.sleep(0.5)
        before = shot()
        swipe(540, 1650, 540, 1250, 1500)  # slow drag (no fling), well under the visible band
        time.sleep(1.0)
        after = shot()
        if np.abs(before[700:2150].astype(int) - after[700:2150].astype(int)).mean() < 2:
            print("end of list")
            break
    print(f"harvested {idx} cards in {time.time() - t0:.0f}s -> {out}")

main()
