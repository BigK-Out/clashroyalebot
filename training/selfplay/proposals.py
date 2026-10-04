"""Where and when the opponent may have played a card (spec section 2).

Two sources, merged: a new enemy level tag that no tracked tag explains (troops and
buildings), and a sudden, localized burst of change in the arena (spells, late tags).
False proposals are cheap: the classifier's "no_play" class absorbs them.
"""
import json
import subprocess
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np

from conftest import ROOT
from selfplay.video import PHONES, load_calib, read_frames

Proposal = namedtuple("Proposal", "t_ms col row kind")

TRACK_DIST = 0.06  # normalized screen distance a tag may move between samples (10 fps)
TRACK_TTL = 400  # ms a lost tag is remembered (detection flicker)
MOTION_ON = 0.5  # changed-pixel fraction of a tile that counts as burst
MOTION_QUIET = 0.15  # the same tiles must have been below this in the previous 300 ms
MOTION_MIN_TILES = 4
MERGE_MS, MERGE_TILES = 600, 3
MATCH_WINDOW = (-300, 1500)
MATCH_TILES = 3


def tag_proposals(frames_enemies):
    tracks, props = [], []  # track: [x, y, last_seen_ms]
    for t, enemies in frames_enemies:
        tracks = [tr for tr in tracks if t - tr[2] <= TRACK_TTL]
        for x, y, c, r in enemies:
            near = [tr for tr in tracks if abs(tr[0] - x) + abs(tr[1] - y) <= TRACK_DIST]
            if near:
                tr = min(near, key=lambda tr: abs(tr[0] - x) + abs(tr[1] - y))
                tr[0], tr[1], tr[2] = x, y, t
            else:
                tracks.append([x, y, t])
                if c >= 0:
                    props.append(Proposal(t, int(c), int(r), "tag"))
    return props


def motion_proposals(diffs):
    props, history = [], []
    for t, grid in diffs:
        hot = grid >= MOTION_ON
        recent = [g for tt, g in history if t - tt <= 300]
        if recent:
            hot &= np.max(recent, axis=0) < MOTION_QUIET
        n, labels, stats, cents = cv2.connectedComponentsWithStats(hot.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= MOTION_MIN_TILES:
                cx, cy = cents[i]
                props.append(Proposal(t, int(round(cx)), int(round(cy)), "motion"))
        history.append((t, grid))
        history = [(tt, g) for tt, g in history if t - tt <= 300]
    return props


def merge(props):
    out = []
    for p in sorted(props, key=lambda p: p.t_ms):
        if any(p.t_ms - q.t_ms <= MERGE_MS and max(abs(p.col - q.col), abs(p.row - q.row)) <= MERGE_TILES for q in out):
            continue
        out.append(p)
    return out


def match_props(props, plays):
    pairs, used = {}, set()
    for i, pl in enumerate(plays):
        cands = [
            j for j, p in enumerate(props)
            if j not in used
            and MATCH_WINDOW[0] <= p.t_ms - pl["t_view"] <= MATCH_WINDOW[1]
            and max(abs(p.col - pl["col"]), abs(p.row - pl["row"])) <= MATCH_TILES
        ]
        if cands:
            j = min(cands, key=lambda j: props[j].t_ms)
            pairs[i] = j
            used.add(j)
    return pairs, [j for j in range(len(props)) if j not in used]


def other(viewer):
    return "note14" if viewer == "note9" else "note9"


def enemy_plays(match_dir, viewer):
    match_dir = Path(match_dir)
    align = json.load(open(match_dir / "align.json"))
    start = int((match_dir / "recording_started_ms").read_text())
    off = align[viewer]["offset_ms"]
    out = []
    for line in open(match_dir / f"plays_{other(viewer)}.jsonl"):
        p = json.loads(line)
        if not p["verified"]:
            continue
        c, r = p["tile"]
        out.append({"card": p["card"], "t_view": p["host_ms"] - start + off, "col": 17 - c, "row": 31 - r})
    return out


def arena_box(calib, h):
    a = calib["arena"]
    return int(a["top_left"]["x"] * 576), int(a["top_left"]["y"] * h), int(a["bottom_right"]["x"] * 576), int(a["bottom_right"]["y"] * h)


def video_proposals(match_dir, viewer):
    cal, h = PHONES[viewer]
    calib = load_calib(cal)
    x0, y0, x1, y1 = arena_box(calib, h)
    proc = subprocess.Popen([str(ROOT / "target/release/units_stream"), "--calibration", str(ROOT / cal), "--height", str(h)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    tags, diffs, prev = [], [], {}
    for t, f in read_frames(Path(match_dir) / f"{viewer}.mkv", h, every_ms=100):
        proc.stdin.write(cv2.cvtColor(f, cv2.COLOR_BGR2RGB).tobytes())
        proc.stdin.flush()
        tags.append((t, json.loads(proc.stdout.readline())))
        g = cv2.GaussianBlur(cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (5, 5), 0)
        ref = prev.get("g")
        if ref is not None:
            changed = (cv2.absdiff(g, ref) > 40).astype(np.float32)
            diffs.append((t, cv2.resize(changed, (18, 32), interpolation=cv2.INTER_AREA)))
        # Compare against the frame 200 ms back (two samples), not the previous one.
        prev["g"], prev["g1"] = prev.get("g1", g), g
    proc.stdin.close()
    proc.wait()
    return merge(tag_proposals(tags) + motion_proposals(diffs))
