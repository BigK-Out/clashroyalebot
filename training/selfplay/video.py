"""Match videos: frames, the elixir bar, and each video's clock offset.

A phone's own plays make its own elixir drop by the card's cost. Matching those drops
to the phone's play log gives the offset between host time and video time (scrcpy
starts recording ~1.5 s after it is spawned), to within a frame or two.
"""
import json
import statistics
import tomllib
from pathlib import Path

import cv2
import numpy as np

from conftest import ROOT

PHONES = {"note9": ("calibration_note9.toml", 1248), "note14": ("calibration.toml", 1280)}
SEARCH = (-2_500, 1_000)  # where to look for the drop, ms around the uncorrected guess
DROP_WITHIN = 350  # a deploy's elixir drop completes within this many ms
REFINE = 400  # second-pass window around the first median, ms


def load_calib(name):
    return tomllib.load(open(ROOT / name, "rb"))


def read_frames(path, height, every_ms=0):
    """(t_ms, BGR frame at 576 x height); with every_ms, at most one frame per interval."""
    cap = cv2.VideoCapture(str(path))
    last = None
    while True:
        if not cap.grab():
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC)
        if every_ms and last is not None and t - last < every_ms:
            continue
        ok, f = cap.retrieve()
        if not ok:
            break
        last = t
        yield t, cv2.resize(f, (576, height), interpolation=cv2.INTER_AREA)


def elixir_fill(frame, calib):
    """Purple fraction of the elixir bar's middle rows, scaled to 0..10."""
    e = calib["elixir_bar"]
    h, w = frame.shape[:2]
    y = int((e["y"] + e["h"] / 2) * h)
    x0, x1 = int(e["x"] * w), int((e["x"] + e["w"]) * w)
    row = frame[y - 1:y + 2, x0:x1].astype(int).mean(0)
    purple = (row[:, 2] > 150) & (row[:, 0] > 150) & (row[:, 1] < 110)
    return float(purple.mean() * 10)


def _deltas(t, v, plays, start_ms, cost, window):
    deltas = []
    for p in plays:
        c = cost.get(p["card"])
        if not p.get("verified") or not c:
            continue
        guess = p["host_ms"] - start_ms
        for i in np.where((t > guess + window[0]) & (t < guess + window[1]))[0]:
            j = np.searchsorted(t, t[i] + DROP_WITHIN)
            if j < len(v) and v[i] - v[j] >= c - 0.7:
                # The first frame at which the drop has happened, not the window start.
                k = i + int(np.argmax(v[i] - v[i:j + 1] >= c - 0.7))
                deltas.append(t[k] - guess)
                break
    return deltas


def match_offset(series, plays, start_ms, cost):
    t, v = series[:, 0], series[:, 1]
    first = _deltas(t, v, plays, start_ms, cost, SEARCH)
    if len(first) < 3:
        return None
    # Second pass close to the first estimate: unrelated earlier drops fall outside it.
    med = statistics.median(first)
    deltas = _deltas(t, v, plays, start_ms, cost, (med - REFINE - DROP_WITHIN, med + REFINE))
    if len(deltas) < 3:
        return None
    q = np.percentile(deltas, [10, 90])
    return float(statistics.median(deltas)), len(deltas), float(q[1] - q[0])


def align_match(match_dir):
    match_dir = Path(match_dir)
    start = int((match_dir / "recording_started_ms").read_text())
    cost = {c["slug"]: c["elixir"] for c in json.load(open(ROOT / "assets/cards.json")) if c.get("elixir")}
    out = {}
    for phone, (cal, height) in PHONES.items():
        calib = load_calib(cal)
        series = np.array([(t, elixir_fill(f, calib)) for t, f in read_frames(match_dir / f"{phone}.mkv", height)])
        plays = [json.loads(l) for l in open(match_dir / f"plays_{phone}.jsonl")]
        r = match_offset(series, plays, start, cost) if len(series) else None
        out[phone] = None if r is None else {"offset_ms": r[0], "n": r[1], "spread_ms": r[2]}
    (match_dir / "align.json").write_text(json.dumps(out, indent=1))
    return out
