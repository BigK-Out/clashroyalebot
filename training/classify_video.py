"""Run the play classifier over a recorded match: proposals -> 8-frame clips -> plays.onnx
-> deck inference. Prints a timeline of recognized enemy plays and writes timeline.png
(one clip strip per play) next to the video.

  .venv/bin/python classify_video.py ../dataset/real/<dir> [--viewer note14] [--min-p 0.5]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from selfplay.clips import CLASSES, CROP_TILES, FRAME_OFFSETS, crop, tile_center_px
from selfplay.deck import refine
from selfplay.proposals import video_proposals
from selfplay.video import PHONES, load_calib, read_frames

ap = argparse.ArgumentParser()
ap.add_argument("match_dir")
ap.add_argument("--viewer", default="note14")
ap.add_argument("--min-p", type=float, default=0.5)
a = ap.parse_args()
m = Path(a.match_dir)
cal, h = PHONES[a.viewer]
calib = load_calib(cal)
tile_px = (calib["arena"]["bottom_right"]["x"] - calib["arena"]["top_left"]["x"]) * 576 / 18

# Enemy plays land on the enemy half or as spells anywhere; proposals cover both.
props = video_proposals(m, a.viewer)
print(f"{len(props)} proposals", flush=True)
wanted = sorted((p.t_ms + off, k, i) for k, p in enumerate(props) for i, off in enumerate(FRAME_OFFSETS))
X = np.zeros((len(props), len(FRAME_OFFSETS), 128, 128, 3), np.uint8)
j = 0
for t, f in read_frames(m / f"{a.viewer}.mkv", h):
    while j < len(wanted) and wanted[j][0] <= t:
        _, k, i = wanted[j]
        cx, cy = tile_center_px(calib, h, props[k].col, props[k].row)
        X[k, i] = cv2.cvtColor(crop(f, cx, cy, CROP_TILES * tile_px), cv2.COLOR_BGR2RGB)
        j += 1
    if j >= len(wanted):
        break

sess = ort.InferenceSession(str(Path(__file__).parent.parent / "assets/models/plays.onnx"))
probs = []
for x in X:
    logits = sess.run(None, {"clip": (x.transpose(0, 3, 1, 2)[None].astype(np.float32) / 255)})[0][0]
    e = np.exp(logits - logits.max())
    probs.append(e / e.sum())
probs = np.array(probs)
NP = len(CLASSES) - 1
plays = [k for k in range(len(props)) if probs[k].argmax() != NP and probs[k].max() >= a.min_p]
# One play shows up as several proposals (tag + motion, a few frames apart): keep the most
# confident of same-card proposals within 2 s and 4 tiles.
kept = []
for k in sorted(plays, key=lambda k: -probs[k].max()):
    c = probs[k].argmax()
    if not any(probs[q].argmax() == c and abs(props[q].t_ms - props[k].t_ms) < 2000
               and max(abs(props[q].col - props[k].col), abs(props[q].row - props[k].row)) <= 4 for q in kept):
        kept.append(k)
kept.sort(key=lambda k: props[k].t_ms)
deck = refine(probs[kept], NP) if kept else []
print(f"{len(kept)} enemy plays recognized:")
for k, d in zip(kept, deck):
    raw = CLASSES[probs[k].argmax()]
    note = "" if d == probs[k].argmax() else f"  (deck inference: {CLASSES[d]})"
    print(f"  {props[k].t_ms / 1000:6.1f}s  tile ({props[k].col:2},{props[k].row:2})  {raw:<18} p={probs[k].max():.2f}{note}")
print("enemy deck seen:", sorted({CLASSES[d] for d in deck}))
rows = []
for k, d in zip(kept, deck):
    strip = np.concatenate(list(X[k][::2]), 1).copy()
    cv2.putText(strip, f"{props[k].t_ms / 1000:.1f}s {CLASSES[d]}", (4, 16), 0, 0.5, (255, 255, 255), 2)
    rows.append(strip)
if rows:
    cv2.imwrite(str(m / "timeline.png"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
