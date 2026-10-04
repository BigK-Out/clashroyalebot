"""Cache per-video enemy tags and motion grids (runs/cache/<match>_<viewer>.pkl) for tuning."""
import json
import pickle
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from conftest import ROOT
from selfplay.proposals import arena_box, enemy_plays
from selfplay.video import PHONES, load_calib, read_frames

out = Path("runs/cache")
out.mkdir(parents=True, exist_ok=True)
for m in sorted(Path("../dataset/selfplay").iterdir()):
    if not (m / "align.json").exists():
        continue
    align = json.load(open(m / "align.json"))
    for viewer, (cal, h) in PHONES.items():
        f_out = out / f"{m.name}_{viewer}.pkl"
        if f_out.exists() or not align.get(viewer):
            continue
        calib = load_calib(cal)
        x0, y0, x1, y1 = arena_box(calib, h)
        proc = subprocess.Popen([str(ROOT / "target/release/units_stream"), "--calibration", str(ROOT / cal), "--height", str(h)],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        tags, grays = [], []
        for t, f in read_frames(m / f"{viewer}.mkv", h, every_ms=100):
            proc.stdin.write(cv2.cvtColor(f, cv2.COLOR_BGR2RGB).tobytes())
            proc.stdin.flush()
            tags.append((t, json.loads(proc.stdout.readline())))
            g = cv2.GaussianBlur(cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (5, 5), 0)
            grays.append((t, cv2.resize(g, (144, 256), interpolation=cv2.INTER_AREA)))
        proc.stdin.close()
        proc.wait()
        pickle.dump({"tags": tags, "grays": grays, "plays": enemy_plays(m, viewer)}, open(f_out, "wb"))
        print(f_out.name, len(tags), flush=True)
