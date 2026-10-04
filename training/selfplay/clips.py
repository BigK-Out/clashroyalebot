"""Labeled 8-frame clips around every proposal and every logged enemy play."""
import json
from pathlib import Path

import cv2
import numpy as np

from conftest import ROOT
from selfplay.proposals import enemy_plays, match_props, video_proposals
from selfplay.video import PHONES, load_calib, read_frames

CLASSES = sorted(c["slug"] for c in json.load(open(ROOT / "assets/cards.json")) if c["type"] != "Tower Troop") + ["no_play"]
FRAME_OFFSETS = [i * 150 for i in range(8)]
CROP_TILES = 6
MISSED_DELAY = 300


def tile_center_px(calib, h, col, row):
    a = calib["arena"]
    x0, y0 = a["top_left"]["x"] * 576, a["top_left"]["y"] * h
    x1, y1 = a["bottom_right"]["x"] * 576, a["bottom_right"]["y"] * h
    return x0 + (col + 0.5) / 18 * (x1 - x0), y0 + (row + 0.5) / 32 * (y1 - y0)


def crop(frame, cx, cy, size_px, out=128):
    h, w = frame.shape[:2]
    half = int(size_px / 2)
    x0, y0 = int(cx) - half, int(cy) - half
    canvas = np.zeros((2 * half, 2 * half, 3), np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x0 + 2 * half), min(h, y0 + 2 * half)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return cv2.resize(canvas, (out, out), interpolation=cv2.INTER_AREA)


def label_events(props, plays, video_ms):
    pairs, unmatched = match_props(props, plays)
    by_prop = {j: i for i, j in pairs.items()}
    ev = []
    for j, p in enumerate(props):
        label = plays[by_prop[j]]["card"] if j in by_prop else "no_play"
        ev.append({"t_ms": p.t_ms, "col": p.col, "row": p.row, "label": label, "proposed": True})
    for i, pl in enumerate(plays):
        if i not in pairs:
            ev.append({"t_ms": pl["t_view"] + MISSED_DELAY, "col": pl["col"], "row": pl["row"], "label": pl["card"], "proposed": False})
    end = video_ms - FRAME_OFFSETS[-1]
    return sorted((e for e in ev if 0 <= e["t_ms"] <= end), key=lambda e: e["t_ms"])


def video_length_ms(path):
    cap = cv2.VideoCapture(str(path))
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 30
    return n / fps * 1000


def build_match(match_dir, out_dir):
    match_dir, out_dir = Path(match_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    align = json.load(open(match_dir / "align.json"))
    for viewer, (cal, h) in PHONES.items():
        if not align.get(viewer):
            continue
        calib = load_calib(cal)
        tile_px = (calib["arena"]["bottom_right"]["x"] - calib["arena"]["top_left"]["x"]) * 576 / 18
        video = match_dir / f"{viewer}.mkv"
        events = label_events(video_proposals(match_dir, viewer), enemy_plays(match_dir, viewer), video_length_ms(video))
        wanted = sorted({(k, e["t_ms"] + off) for k, e in enumerate(events) for off in FRAME_OFFSETS}, key=lambda x: x[1])
        X = np.zeros((len(events), len(FRAME_OFFSETS), 128, 128, 3), np.uint8)
        filled = {}
        i = 0
        for t, f in read_frames(video, h):
            while i < len(wanted) and wanted[i][1] <= t:
                k, tw = wanted[i]
                e = events[k]
                slot = FRAME_OFFSETS.index(round(tw - e["t_ms"]))
                cx, cy = tile_center_px(calib, h, e["col"], e["row"])
                X[k, slot] = cv2.cvtColor(crop(f, cx, cy, CROP_TILES * tile_px), cv2.COLOR_BGR2RGB)
                filled[(k, slot)] = True
                i += 1
            if i >= len(wanted):
                break
        y = np.array([CLASSES.index(e["label"]) for e in events], np.int16)
        np.savez_compressed(out_dir / f"{match_dir.name}_{viewer}.npz", X=X, y=y,
                            t=np.array([e["t_ms"] for e in events], np.float32),
                            proposed=np.array([e["proposed"] for e in events], bool))
        print(match_dir.name, viewer, len(events), "events", int((y < len(CLASSES) - 1).sum()), "plays", flush=True)
