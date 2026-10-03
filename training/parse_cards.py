"""OCR the harvested Info screens into a card table.

dataset/cards/NNN/{info,stats}.png -> ../assets/cards.json
  [{"name", "slug", "elixir", "rarity", "type", "stats": {label: value}, "dir"}], one per card
  (duplicates from the harvest are dropped by name; the first copy wins).
OCR is imperfect: names/type/elixir are reliable, stat values are best-effort strings.
"""
import json
import re
from pathlib import Path

import difflib

import cv2
import easyocr

ROOT = Path("../dataset/cards")
TITLE = (830, 1000, 300, 1000)      # y0, y1, x0, x1 on 1080x2400 screens
TAGS = (1080, 1240, 340, 880)
BADGE = (835, 900, 55, 120)         # elixir drop on the card portrait
STATS = (1250, 1700, 80, 1000)

reader = easyocr.Reader(["en"], gpu=True, verbose=False)
KNOWN = [l.strip() for l in open("card_names.txt") if l.strip()]


def canonical(raw):
    """Snap OCR'd names to known card names ("Fire Spinit" -> "Fire Spirit"); keep unknowns."""
    m = difflib.get_close_matches(raw.lower(), [k.lower() for k in KNOWN], n=1, cutoff=0.75)
    return next(k for k in KNOWN if k.lower() == m[0]) if m else raw


def crop(img, box):
    y0, y1, x0, x1 = box
    return img[y0:y1, x0:x1]


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def elixir(img):
    badge = cv2.resize(crop(img, BADGE), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    txt = "".join(reader.readtext(badge, detail=0, allowlist="0123456789?"))
    if "?" in txt:
        return None  # Mirror: costs 1 more than the mirrored card
    # OCR reliably reads 2-9 but misses the thin "1": nothing read means 1.
    return int(txt) if txt.isdigit() and 1 <= int(txt) <= 10 else 1


STAT_LABELS = ["Damage", "Damage/Sec", "Hitpoints", "Hit Speed", "Targets", "Speed", "Range", "Count",
               "Area Damage", "Ranged Damage", "Crown Tower Damage", "Radius", "Duration", "Stun Duration",
               "Freeze Duration", "Slowdown Duration", "Healing", "Chained Attacks", "Shield Hitpoints",
               "Spawn Speed", "Lifetime", "Deploy Time", "Death Damage", "Charge Damage", "Dash Damage",
               "Spawn Damage", "Elixir Production", "Production Speed", "Troop Spawned", "Skeletons Spawned"]


def clean_label(raw):
    m = difflib.get_close_matches(raw, STAT_LABELS, n=1, cutoff=0.7)
    return m[0] if m else raw


def clean_value(raw):
    """'#l1sec' -> '1.1sec', 'Ai & Ground' -> 'Air & Ground'; digits kept, OCR noise removed."""
    v = raw.replace("Ai &", "Air &").replace("Groud", "Ground")
    m = re.search(r"(\d[\d.,]*)\s*(sec|x)?", v)
    if m and not re.search(r"[A-Za-z]{4,}", v.replace("sec", "")):
        num = m.group(1).replace(",", "")
        return num + (m.group(2) or "")
    return v


def stats(img):
    """Pair each label with the value printed right below it (same column)."""
    boxes = [(b[0][0][0], b[0][0][1], b[1]) for b in reader.readtext(crop(img, STATS))]
    labels = [b for b in boxes if re.search(r"[A-Za-z]{3,}", b[2]) and not re.fullmatch(r"[\d.,x+%]+\w{0,4}", b[2])]
    out = {}
    for lx, ly, lab in labels:
        below = [b for b in boxes if abs(b[0] - lx) < 40 and 15 < b[1] - ly < 60 and b not in labels]
        val = below[0][2] if below else next((b[2] for b in boxes if abs(b[0] - lx) < 40 and 15 < b[1] - ly < 60), None)
        if val is not None:
            out[clean_label(lab)] = clean_value(val)
    return out


cards, seen = [], set()
for d in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    info, st = cv2.imread(str(d / "info.png")), cv2.imread(str(d / "stats.png"))
    if info is None:
        continue
    title = [t for t in reader.readtext(crop(info, TITLE), detail=0) if not t.lower().startswith("level")]
    name = canonical(" ".join(title).strip().title())
    if not name or name in seen:
        continue
    seen.add(name)
    tags = reader.readtext(crop(info, TAGS), detail=0)
    vals = [t for t in tags if t.upper() not in ("RARITY", "TYPE")]
    cards.append({
        "name": name,
        "slug": slug(name),
        "elixir": None if name == "Mirror" else elixir(info),  # Mirror: mirrored card + 1
        "rarity": vals[0].title() if vals else None,
        "type": vals[1].title() if len(vals) > 1 else None,
        "stats": stats(st) if st is not None else {},
        "dir": d.name,
    })
    print(f"{d.name}  {name:22s} {cards[-1]['elixir']}  {cards[-1]['type']}")

Path("../assets/cards.json").write_text(json.dumps(cards, indent=1))
print(f"{len(cards)} unique cards -> ../assets/cards.json")
