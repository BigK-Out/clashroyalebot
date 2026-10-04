"""Harvest every card's Info screen from the in-game Collection (phone must be on the
Collection tab, scrolled to the top, sorted By Elixir).

Per card -> dataset/cards/<NNN>/
  clip_XX.jpg   page 1: gameplay preview (unit sprite walking in the arena), ~12 frames
  info.png      page 1 full screenshot (name, rarity, type, elixir badge)
  stats.png     page 2 full screenshot (damage, hitpoints, speed, ...)
  name.txt      card name (OCR'd title), grid.png  the card's art in the Collection grid
  layout.txt    "card" or "champion" (champions: page 1 info, page 2 preview, ability.png, stats)
Page 3 (flavor text) is skipped.

Resumable: cards already in --out (by name.txt, or OCR'd from info.png) are not harvested
again. A card whose grid art matches a harvested grid.png is skipped without opening it;
otherwise its Info is opened and the title checked. If the game drops back to another
screen (reconnect), it navigates back to the Collection and continues.

Never sends BACK (that opens "Exit Clash Royale?"); the Info popup is closed with its X.
"""
import argparse
import difflib
import re
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
ap.add_argument("--recoveries", type=int, default=5, help="times to navigate back to the Collection")
args = ap.parse_args()

COLS = [146, 408, 670, 931]          # card centers (1080x2400 screen)
BAND_DX = -46                         # sample the purple "Max" band left of its text
CARD_ABOVE_BAND = 190                 # card art center is this far above the band center
VISIBLE = (800, 1550)                 # card centers whose menu opens below the card
CLOSE_X = (966, 840)                  # Info popup close button
PREVIEW = (80, 1265, 1000, 1830)      # page-1 gameplay preview box
PAGE_SWIPE_Y = 1550
TITLE = (830, 1000, 300, 1000)        # y0, y1, x0, x1 of the card name on the Info popup
# Champions have a taller 5-page popup: 1 info, 2 gameplay preview, 3 ability, 4 stats, 5 flavor.
CHAMP_CLOSE_X = (967, 394)
CHAMP_TITLE = (395, 540, 300, 1000)
CARDS_TAB = (270, 2310)               # bottom nav "Cards" button
COLLECTION_TAB = (766, 304)           # "Collection" tab at the top of the Cards screen
KNOWN = [l.strip() for l in open(Path(__file__).with_name("card_names.txt")) if l.strip()]

_reader = None


class Lost(Exception):
    """Not where we expected to be (popup didn't open, dropped off the Collection)."""


def card_name(img, kind="card"):
    """OCR the Info popup title, snapped to a known card name."""
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    y0, y1, x0, x1 = CHAMP_TITLE if kind == "champion" else TITLE
    words = [t for t in _reader.readtext(img[y0:y1, x0:x1], detail=0) if not t.lower().startswith("level") and not t.strip().isdigit()]  # stray elixir digits
    raw = re.sub(r"^\d+\s+", "", " ".join(words).strip()).title()
    # Same length +-1 only: OCR swaps letters ("Kwight"), it doesn't drop syllables, and a
    # name that isn't in the list must not snap to a longer one (Prince -> Princess).
    m = difflib.get_close_matches(raw.lower(), [k.lower() for k in KNOWN if abs(len(k) - len(raw)) <= 1], n=1, cutoff=0.82)
    return next(k for k in KNOWN if k.lower() == m[0]) if m else raw


def adb(*a):
    return subprocess.run([args.adb, "-s", args.serial, *a], capture_output=True, check=True).stdout


def shot():
    return cv2.imdecode(np.frombuffer(adb("exec-out", "screencap", "-p"), np.uint8), cv2.IMREAD_COLOR)


def still_shot():
    """Screenshot once the grid has stopped moving (scroll inertia, highlight animation)."""
    prev = shot()
    for _ in range(10):
        time.sleep(0.25)
        cur = shot()
        if np.abs(prev[700:2150].astype(int) - cur[700:2150].astype(int)).mean() < 1:
            return cur
        prev = cur
    return prev


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
    """Which Info popup is up, by where its red close X is: "card", "champion" or None."""
    for kind, (x, y) in (("card", CLOSE_X), ("champion", CHAMP_CLOSE_X)):
        box = img[y - 30:y + 30, x - 30:x + 30].astype(int)
        red = (box[..., 2] > 200) & (box[..., 1] < 110) & (box[..., 0] < 120)
        if red.sum() >= 40:
            return kind
    return None


def on_collection(img):
    return len(card_centers_any(img)) > 0


def grid_art(img, x, y):
    return img[y - 140:y + 60, x - 95:x + 95]


def signature(art):
    """Card art in the grid, small + normalized: identifies a card wherever it is on screen.
    The top 70 px are left out: the "New!" ribbon there disappears once a card is opened."""
    g = cv2.cvtColor(art[70:], cv2.COLOR_BGR2GRAY)
    v = cv2.resize(g, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    return (v - v.mean()) / (v.std() + 1e-6)


def seen_before(sig, seen, thr=0.9):
    return any(float(sig @ s) / len(sig) > thr for s in seen)


def load_done(out):
    """Names and grid signatures of cards already harvested in out/."""
    names, sigs, dirs = set(), [], {}
    for d in sorted(p for p in out.iterdir() if p.is_dir()):
        nf = d / "name.txt"
        if not nf.exists():
            info = cv2.imread(str(d / "info.png"))
            if info is None:
                continue
            nf.write_text(card_name(info))
        names.add(nf.read_text().strip())
        dirs.setdefault(nf.read_text().strip(), d)
        if (d / "grid.png").exists():
            g = cv2.imread(str(d / "grid.png"))
            sigs.append(signature(g))
    return names, sigs, dirs


def goto_collection(drags):
    """Back to the Collection grid from the main screens (no BACK key), at the top, then
    `drags` scroll steps down: roughly where the scan was, so done cards aren't reopened."""
    tap(*CARDS_TAB)
    time.sleep(2.0)
    tap(*COLLECTION_TAB)
    time.sleep(2.0)
    for _ in range(30):
        before = shot()
        swipe(540, 900, 540, 2000, 200)  # fling down = scroll up
        time.sleep(0.8)
        if np.abs(before[700:2150].astype(int) - shot()[700:2150].astype(int)).mean() < 2:
            break
    for _ in range(max(0, drags - 1)):
        swipe(540, 1650, 540, 1250, 1500)
    time.sleep(1.0)
    return on_collection(shot())


def scroll_shift(a, b):
    """Content moved up by this many px between screenshots a and b (phase correlation)."""
    ga = cv2.cvtColor(a[700:2150], cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b[700:2150], cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(ga, gb)
    return dy


def harvest(idx, x, y, out, art, done):
    """Open the card's Info; save it unless its name is already in done. Returns the name
    (or None if the menu had no Info button)."""
    tap(x, y)
    time.sleep(0.8)
    iy = find_info_button(shot(), x, y)
    if iy is None:
        print(f"  card {idx}: no Info button found at ({x},{y})")
        tap(x, y)  # close the menu
        time.sleep(0.5)
        return None
    tap(x, iy)
    time.sleep(1.2)
    img = shot()
    kind = popup_open(img)
    if kind is None:
        raise Lost(f"card {idx}: Info popup did not open")
    close = CHAMP_CLOSE_X if kind == "champion" else CLOSE_X
    name = card_name(img, kind)
    if name in done:
        tap(*close)
        time.sleep(0.8)
        if not on_collection(shot()):
            raise Lost(f"{name}: not back on the Collection grid")
        return name

    def next_page(what):
        swipe(850, PAGE_SWIPE_Y, 250, PAGE_SWIPE_Y, 250)
        time.sleep(1.0)
        page = shot()
        if popup_open(page) != kind:
            raise Lost(f"card {idx}: popup closed by page swipe ({what})")
        return page

    d = out / f"{idx:03d}"
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "info.png"), img)
    cv2.imwrite(str(d / "grid.png"), art)
    (d / "name.txt").write_text(name)
    (d / "layout.txt").write_text(kind)
    if kind == "champion":
        next_page("preview")
    x0, y0, x1, y1 = PREVIEW
    for k in range(args.clip_frames):
        cv2.imwrite(str(d / f"clip_{k:02d}.jpg"), shot()[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    if kind == "champion":
        cv2.imwrite(str(d / "ability.png"), next_page("ability"))
    cv2.imwrite(str(d / "stats.png"), next_page("stats"))
    tap(*close)
    time.sleep(0.8)
    if not on_collection(shot()):
        raise Lost(f"card {idx}: not back on the Collection grid")
    return name


def scan(out, done, sigs, state):
    """Harvest the Collection from the current scroll position down to the end."""
    while state["idx"] < args.max_cards:
        # Re-scan until this screen is exhausted: a just-closed card can be highlighted
        # (band color off, art scaled) for a moment, so one pass can miss it or see it anew.
        for _ in range(3):
            img = still_shot()
            if not on_collection(img):
                raise Lost("not on the Collection grid")
            todo = []
            for x, y in sorted(card_centers(img), key=lambda c: (c[1], c[0])):
                art = grid_art(img, x, y).copy()
                sig = signature(art)
                if not seen_before(sig, sigs):
                    todo.append((x, y, art, sig))
            if not todo:
                break
            for x, y, art, sig in todo:
                if state["idx"] >= args.max_cards:
                    break
                name = harvest(state["idx"], x, y, out, art, done)
                if name is None:
                    continue  # retried on the next scan
                # Remember the card both as it was and as it looks now: a just-closed card
                # stays highlighted for a while and would not match its first signature.
                sigs += [sig, signature(grid_art(shot(), x, y))]
                if name in done:
                    print(f"  skip {name} (already harvested)")
                    g = state["dirs"].get(name)
                    if g is not None and not (g / "grid.png").exists():
                        cv2.imwrite(str(g / "grid.png"), art)  # next run skips it unopened
                    continue
                done.add(name)
                print(f"card {state['idx']:3d} {name:22s} at ({x},{y})  [{time.time() - state['t0']:.0f}s]")
                state["idx"] += 1
            time.sleep(0.5)
        before = shot()
        swipe(540, 1650, 540, 1250, 1500)  # slow drag (no fling), well under the visible band
        state["drags"] += 1
        time.sleep(1.0)
        after = shot()
        if np.abs(before[700:2150].astype(int) - after[700:2150].astype(int)).mean() < 2:
            print("end of list")
            return


def main():
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done, sigs, dirs = load_done(out)
    print(f"{len(done)} cards already harvested")
    state = {"idx": len([p for p in out.iterdir() if p.is_dir()]), "t0": time.time(), "dirs": dirs, "drags": 0}
    for attempt in range(args.recoveries + 1):
        try:
            scan(out, done, sigs, state)
            break
        except Lost as e:
            print(f"lost: {e}")
            cv2.imwrite(str(out.parent / "harvest_lost.png"), shot())
            if attempt == args.recoveries or not goto_collection(state["drags"]):
                raise SystemExit("could not get back to the Collection; stopping")
            print(f"back on the Collection, {state['drags']} drags down")
    missing = [k for k in dict.fromkeys(KNOWN) if k not in done]
    print(f"harvested {len(done)} cards in {time.time() - state['t0']:.0f}s -> {out}")
    print(f"not harvested ({len(missing)}): {', '.join(missing)}")


main()
