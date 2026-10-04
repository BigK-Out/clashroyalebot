# Enemy Play Classifier (training) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** From the recorded self-play matches, build labeled play clips, train the event classifier (122 cards + "no play"), add deck inference, and report accuracy against the spec's ≥99% target.

**Architecture:** Everything here is offline Python in `training/selfplay/` (a package) plus one small Rust tool. Each match folder has two videos (`note9.mkv`, `note14.mkv`) and two play logs (`plays_note9.jsonl`, `plays_note14.jsonl`); every play by one phone is an enemy play in the other phone's video. Per video, the clock offset comes from that phone's own elixir drops matched to its own logged plays. Enemy tags per frame come from the existing Rust detector (`vision::units::detect_units`) through a stdin/stdout tool. Proposals (new enemy tags + motion bursts) are matched to the other phone's plays to produce labeled clips and "no play" negatives. A per-frame ResNet-18 with temporal pooling is trained on 8-frame clips and exported to ONNX. Deck inference re-ranks per-play predictions with the 8-card deck and cycle constraints.

**Tech Stack:** Python 3.12 venv `training/.venv` (torch 2.14 + CUDA, torchvision, opencv, numpy), pytest (added), Rust workspace crate `vision` (new bin `units_stream`).

**Spec:** `docs/superpowers/specs/2026-10-04-enemy-card-id-design.md` sections 2 (event detection), 3 (classifier and deck inference), 4 (evaluation). Bot integration (`detect::plays`, section 4 "Integration") gets its own plan after this one reports.

## Global Constraints

- Match folder layout (produced by `target/release/selfplay --both`): `dataset/selfplay/<match_id>/{note9.mkv, note14.mkv, plays_note9.jsonl, plays_note14.jsonl, recording_started_ms, meta.json}`; only folders whose `meta.json` has `"complete": true` and both mkv files are used.
- Videos: Note 9 590x1280 (resize to 576x1248, calibration `calibration_note9.toml`), Note 14 576x1280 (calibration `calibration.toml`), ~30 fps, frame timestamps from `cv2.CAP_PROP_POS_MSEC`.
- Video time of a host time: `video_ms = host_ms - recording_started_ms + offset_ms` with `offset_ms` per video (measured about -1460).
- A play in the player's own view at tile (c, r) is at tile (17 - c, 31 - r) in the opponent's view. Arena is 18 cols x 32 rows (`calib::ARENA_COLS/ROWS`).
- Classes: the 122 playable slugs from `assets/cards.json` (type != "Tower Troop"), sorted, then `"no_play"` last: 123 outputs.
- Clip: 8 frames at +0, 150, ..., 1050 ms after the proposal time; crop 6x6 tiles centered on the proposal tile, resized to 128x128 RGB. Early readout uses the first 4 frames (0-450 ms).
- Split by match, never by clip: `match_id` hash mod 10 == 0 is validation.
- Only `verified: true` plays are labels.
- Long jobs go to the background with a log file; never `pkill -f`; stop by PID.
- Commit after every task with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- **Video shorter than the play log** (recording died early or started late): plays outside the video are skipped, not clipped from wrong frames (Task 4 test `plays_outside_video_are_skipped`).
- **A match where alignment fails** (fewer than 3 elixir drops matched): the video is excluded, not aligned with a default offset (Task 1 test `too_few_matches_gives_none`).
- **Crops at the arena edge** (plays on col 0 or row 0): the crop is padded, never shifted onto a different area or crashed (Task 4 test `crop_at_corner_is_padded`).
- **Same card played twice in a row by the opponent** (not allowed by the cycle unless Mirror): deck inference must not force a different card when the classifier is very confident (Task 6 test `confident_prediction_wins_over_cycle`).
- **Spawned units** (Goblin Hut, Tombstone, Witch skeletons): proposals with no logged play become `no_play` clips (Task 4 test `unmatched_proposal_is_no_play`).

---

### Task 1: Video reading and per-video clock alignment

**Files:**
- Create: `training/selfplay/__init__.py` (empty), `training/selfplay/video.py`, `training/tests/test_video.py`, `training/conftest.py`
- Modify: `training/requirements.txt` (add `pytest`, `torchvision`)

**Interfaces:**
- Produces: `PHONES = {"note9": ("calibration_note9.toml", 1248), "note14": ("calibration.toml", 1280)}`; `load_calib(name) -> dict`; `read_frames(path, height, every_ms=0) -> Iterator[(t_ms: float, frame_bgr_576)]`; `elixir_fill(frame, calib) -> float` (0..10); `match_offset(series: np.ndarray[N,2], plays: list[dict], start_ms: int, cost: dict[str,int]) -> Optional[tuple[float, int, float]]` (offset, n matched, p90-p10 spread); `align_match(match_dir) -> dict` writing `align.json` `{"note9": {"offset_ms", "n", "spread_ms"} | null, "note14": ...}`.

- [ ] **Step 1: Test setup.** `cd training && .venv/bin/pip install pytest` and add `pytest` and `torchvision` to `requirements.txt`. Create `training/conftest.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).parent.parent
```

- [ ] **Step 2: Write the failing tests**

```python
# training/tests/test_video.py
import numpy as np

from selfplay.video import match_offset

COST = {"knight": 3, "zap": 2}


def series_with_drops(drops, length_ms=60_000, start=7.0):
    """Elixir at 30 fps: regenerates slowly, drops by `cost` at each (t, cost)."""
    t = np.arange(0, length_ms, 33.0)
    v = np.full_like(t, start)
    for td, cost in drops:
        v[t >= td] -= cost
    return np.stack([t, v], 1)


def test_constant_offset_found():
    start = 1_000_000
    plays = [{"host_ms": start + 5_000 * i + 2_000, "card": "knight", "verified": True} for i in range(6)]
    drops = [(p["host_ms"] - start - 1_460, 3) for p in plays]
    off = match_offset(series_with_drops(drops), plays, start, COST)
    assert off is not None
    offset, n, spread = off
    assert abs(offset - (-1_460)) <= 40 and n == 6 and spread <= 80


def test_too_few_matches_gives_none():
    start = 0
    plays = [{"host_ms": 10_000, "card": "knight", "verified": True}]
    assert match_offset(series_with_drops([(8_540, 3)]), plays, start, COST) is None


def test_drop_smaller_than_cost_is_not_a_match():
    start = 0
    plays = [{"host_ms": 10_000 + 5_000 * i, "card": "knight", "verified": True} for i in range(4)]
    drops = [(p["host_ms"] - 1_460, 1) for p in plays]  # only 1 elixir drops: not this card
    assert match_offset(series_with_drops(drops), plays, start, COST) is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd training && .venv/bin/python -m pytest tests/test_video.py -q`
Expected: FAIL (`ModuleNotFoundError: selfplay.video`).

- [ ] **Step 4: Write the implementation**

```python
# training/selfplay/video.py
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


def match_offset(series, plays, start_ms, cost):
    t, v = series[:, 0], series[:, 1]
    deltas = []
    for p in plays:
        c = cost.get(p["card"])
        if not p.get("verified") or not c:
            continue
        guess = p["host_ms"] - start_ms
        for i in np.where((t > guess + SEARCH[0]) & (t < guess + SEARCH[1]))[0]:
            j = np.searchsorted(t, t[i] + DROP_WITHIN)
            if j < len(v) and v[i] - v[j] >= c - 0.7:
                deltas.append(t[i] - guess)
                break
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
```

Note `SEARCH` starts at -2500 ms because the offset is about -1460 ms; the uncorrected guess is late by that much.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd training && .venv/bin/python -m pytest tests/test_video.py -q`
Expected: PASS (3 tests).

- [ ] **Step 6: Align the recorded matches.** Add `training/selfplay_align.py`:

```python
"""Write align.json for every complete --both match that has none yet."""
import json
from pathlib import Path

from selfplay.video import align_match

for m in sorted(Path("../dataset/selfplay").iterdir()):
    meta = m / "meta.json"
    if not (m / "note9.mkv").exists() or not meta.exists() or not json.load(open(meta)).get("complete"):
        continue
    if (m / "align.json").exists():
        continue
    print(m.name, align_match(m), flush=True)
```

Run: `cd training && .venv/bin/python selfplay_align.py`
Expected: every match prints both offsets between -2000 and -1000 ms with spread under 150 ms.

- [ ] **Step 7: Commit**

```bash
git add training/selfplay training/tests training/conftest.py training/selfplay_align.py training/requirements.txt
git commit -m "training: per-video clock offset from own elixir drops"
```

---

### Task 2: `units_stream`: enemy tags for piped frames

**Files:**
- Create: `crates/vision/src/bin/units_stream.rs`
- Modify: `crates/vision/src/units.rs` (add `pub fn enemy_json(units: &[Unit]) -> String`)

**Interfaces:**
- Produces: CLI `units_stream --calibration <toml> --height <H>` reading raw RGB frames of 576 x H from stdin, one line of JSON per frame on stdout: `[[x, y, col, row], ...]` for enemy units (feet, normalized; tile or -1,-1). `pub fn enemy_json(units: &[Unit]) -> String`.

- [ ] **Step 1: Write the failing test** (in `crates/vision/src/units.rs` tests module; create `#[cfg(test)] mod tests` if missing)

```rust
#[cfg(test)]
mod json_tests {
    use super::*;

    #[test]
    fn enemy_json_keeps_only_enemies() {
        let u = |team, x, y, tile| Unit { team, tag: [0.0; 4], feet: calib::NPoint { x, y }, tile };
        let units = [u(Team::Enemy, 0.25, 0.5, Some((4, 15))), u(Team::Mine, 0.5, 0.7, Some((9, 22))), u(Team::Enemy, 0.1, 0.05, None)];
        assert_eq!(enemy_json(&units), "[[0.2500,0.5000,4,15],[0.1000,0.0500,-1,-1]]");
    }
}
```

Check the names first: `Team::Enemy`/`Team::Mine` and `calib::NPoint` as used in `units.rs`; use the real variant names.

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p vision json_tests`
Expected: FAIL to compile (`enemy_json` not found).

- [ ] **Step 3: Implement**

```rust
// crates/vision/src/units.rs
/// Enemy units as compact JSON for offline tools: [[x, y, col, row], ...] (feet, normalized).
pub fn enemy_json(units: &[Unit]) -> String {
    let items: Vec<String> = units
        .iter()
        .filter(|u| u.team == Team::Enemy)
        .map(|u| {
            let (c, r) = u.tile.map_or((-1, -1), |(c, r)| (c as i64, r as i64));
            format!("[{:.4},{:.4},{c},{r}]", u.feet.x, u.feet.y)
        })
        .collect();
    format!("[{}]", items.join(","))
}
```

```rust
// crates/vision/src/bin/units_stream.rs
//! Enemy unit tags for frames piped in as raw RGB (576 x height each), one JSON line out
//! per frame. Lets the Python dataset tools use the bot's tag detector on video frames.
//!
//!   units_stream --calibration calibration.toml --height 1280 < frames.rgb

use std::io::{BufWriter, Read, Write};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let calib = calib::Calibration::load(get("--calibration").unwrap_or("calibration.toml".into()))?;
    let height: u32 = get("--height").unwrap_or("1280".into()).parse()?;
    let mapping = calib.arena_mapping()?;
    let mut stdin = std::io::stdin().lock();
    let mut out = BufWriter::new(std::io::stdout().lock());
    let mut buf = vec![0u8; 576 * height as usize * 3];
    while stdin.read_exact(&mut buf).is_ok() {
        let frame = capture::Frame { seq: 0, width: 576, height, rgb: buf.clone(), captured_at: std::time::Instant::now() };
        let units = vision::units::detect_units(&frame, &calib, &mapping);
        writeln!(out, "{}", vision::units::enemy_json(&units))?;
        out.flush()?;
    }
    Ok(())
}
```

- [ ] **Step 4: Run tests and build**

Run: `cargo test -p vision json_tests && cargo build --release -p vision --bin units_stream`
Expected: PASS; binary at `target/release/units_stream`.

- [ ] **Step 5: Smoke check.** `training/.venv/bin/python -c "import cv2,subprocess; f=cv2.cvtColor(cv2.resize(cv2.imread('fixtures/frames/live_hog_e10.jpg'),(576,1280)),cv2.COLOR_BGR2RGB); print(subprocess.run(['target/release/units_stream','--height','1280'],input=f.tobytes(),capture_output=True).stdout)"`
Expected: one JSON line (possibly `[]`).

- [ ] **Step 6: Commit**

```bash
git add crates/vision
git commit -m "vision: units_stream, enemy tags for piped frames"
```

---

### Task 3: Proposals (new enemy tags + motion bursts) and their recall

**Files:**
- Create: `training/selfplay/proposals.py`, `training/tests/test_proposals.py`

**Interfaces:**
- Consumes: `read_frames`, `load_calib`, `PHONES` (Task 1); `target/release/units_stream` (Task 2).
- Produces: `Proposal = namedtuple("Proposal", "t_ms col row kind")` (`kind`: "tag" | "motion"); `tag_proposals(frames_enemies: list[tuple[float, list[list[float]]]]) -> list[Proposal]`; `motion_proposals(diffs: list[tuple[float, np.ndarray[32,18]]]) -> list[Proposal]`; `merge(props) -> list[Proposal]`; `enemy_plays(match_dir, viewer) -> list[dict]` (the other phone's verified plays with `t_view` in the viewer's video ms and `col,row` mirrored); `match_props(props, plays) -> (pairs: dict[play_idx -> prop_idx], unmatched_props: list[int])`; `video_proposals(match_dir, viewer) -> list[Proposal]` (runs the video through both detectors at 10 fps).

Matching rule: a proposal matches a play when `-300 <= prop.t - play.t_view <= 1500` ms and Chebyshev tile distance <= 3; each play takes the earliest such proposal.

- [ ] **Step 1: Write the failing tests**

```python
# training/tests/test_proposals.py
import numpy as np

from selfplay.proposals import Proposal, match_props, merge, motion_proposals, tag_proposals


def test_new_tag_is_a_proposal_and_a_tracked_one_is_not():
    frames = [(0, []), (100, [[0.5, 0.3, 9, 8]]), (200, [[0.5, 0.31, 9, 8]]), (300, [[0.5, 0.32, 9, 9]])]
    props = tag_proposals(frames)
    assert [(p.t_ms, p.col, p.row) for p in props] == [(100, 9, 8)]


def test_flicker_does_not_make_a_new_proposal():
    frames = [(0, [[0.5, 0.3, 9, 8]]), (100, []), (200, [[0.5, 0.3, 9, 8]])]
    assert len(tag_proposals(frames)) == 1


def test_motion_burst_in_quiet_area():
    quiet = np.zeros((32, 18))
    burst = quiet.copy()
    burst[5:8, 4:7] = 0.9
    diffs = [(0, quiet), (100, quiet), (200, quiet), (300, burst)]
    props = motion_proposals(diffs)
    assert len(props) == 1 and props[0].kind == "motion" and (props[0].col, props[0].row) == (5, 6)


def test_merge_keeps_earliest_of_close_proposals():
    p = [Proposal(1000, 5, 5, "motion"), Proposal(1200, 6, 5, "tag"), Proposal(5000, 5, 5, "tag")]
    assert merge(p) == [Proposal(1000, 5, 5, "motion"), Proposal(5000, 5, 5, "tag")]


def test_match_props_by_time_and_place():
    props = [Proposal(1300, 4, 10, "tag"), Proposal(9000, 4, 10, "tag")]
    plays = [{"t_view": 1000, "col": 4, "row": 11}, {"t_view": 20000, "col": 4, "row": 11}]
    pairs, unmatched = match_props(props, plays)
    assert pairs == {0: 0} and unmatched == [1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd training && .venv/bin/python -m pytest tests/test_proposals.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write the implementation**

```python
# training/selfplay/proposals.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd training && .venv/bin/python -m pytest tests/test_proposals.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Measure recall on recorded matches.** Add `training/selfplay_recall.py`:

```python
"""Recall of logged enemy plays by proposals, per card kind, on aligned matches."""
import collections
import json
from pathlib import Path

from selfplay.proposals import enemy_plays, match_props, video_proposals

kind = {c["slug"]: c["type"] for c in json.load(open("../assets/cards.json"))}
hit, total, phantom, nprops = collections.Counter(), collections.Counter(), 0, 0
for m in sorted(Path("../dataset/selfplay").iterdir()):
    a = m / "align.json"
    if not a.exists():
        continue
    align = json.load(open(a))
    for viewer in ("note9", "note14"):
        if not align.get(viewer):
            continue
        props = video_proposals(m, viewer)
        plays = enemy_plays(m, viewer)
        pairs, unmatched = match_props(props, plays)
        for i, p in enumerate(plays):
            total[kind[p["card"]]] += 1
            hit[kind[p["card"]]] += i in pairs
        phantom += len(unmatched)
        nprops += len(props)
    print(m.name, dict(total), dict(hit), flush=True)
print("recall by kind:", {k: f"{hit[k]}/{total[k]} = {hit[k] / total[k]:.3f}" for k in total})
print("overall recall:", sum(hit.values()) / max(1, sum(total.values())), "| proposals:", nprops, "phantom:", phantom)
```

Run (background, log to `training/runs/recall.log`): `cd training && .venv/bin/python selfplay_recall.py > runs/recall.log 2>&1`
Expected: overall recall ≥ 0.995. If it is lower, tune in this order and rerun: `MATCH_WINDOW` upper bound (late tags), `MOTION_ON`/`MOTION_QUIET` (spells), `TRACK_DIST` (fast troops hiding new tags). Record the final constants and recall in the commit message.

- [ ] **Step 6: Commit**

```bash
git add training/selfplay/proposals.py training/tests/test_proposals.py training/selfplay_recall.py
git commit -m "training: play proposals from new enemy tags and motion bursts (recall <N>)"
```

---

### Task 4: Clip dataset

**Files:**
- Create: `training/selfplay/clips.py`, `training/tests/test_clips.py`, `training/selfplay_clips.py`

**Interfaces:**
- Consumes: Task 1 `read_frames`, `load_calib`, `PHONES`; Task 3 `video_proposals`, `enemy_plays`, `match_props`.
- Produces: `CLASSES: list[str]` (122 sorted playable slugs + "no_play"); `FRAME_OFFSETS = [0, 150, ..., 1050]`; `tile_center_px(calib, h, col, row) -> (x, y)`; `crop(frame, cx, cy, size_px, out=128) -> np.ndarray[128,128,3]` (zero-padded at edges); `label_events(props, plays, video_ms) -> list[dict]` with `{t_ms, col, row, label, proposed}`; `build_match(match_dir, out_dir)` writing `out_dir/<match_id>_<viewer>.npz` with `X uint8 [N,8,128,128,3]` (RGB), `y int16 [N]`, `t float32 [N]`, `proposed bool [N]`.

Labeling rule: every proposal matched to a play gets the play's card; every unmatched proposal is `no_play`; every play without a proposal is added as an event at `t_view + 300` ms at its own tile with `proposed=False`, so the classifier still sees every logged play.

- [ ] **Step 1: Write the failing tests**

```python
# training/tests/test_clips.py
import numpy as np

from selfplay.clips import CLASSES, crop, label_events
from selfplay.proposals import Proposal


def test_classes_are_122_cards_plus_no_play():
    assert len(CLASSES) == 123 and CLASSES[-1] == "no_play" and "hog_rider" in CLASSES and "tower_princess" not in CLASSES


def test_crop_at_corner_is_padded():
    f = np.full((1280, 576, 3), 200, np.uint8)
    c = crop(f, 0, 0, 180)
    assert c.shape == (128, 128, 3)
    assert c[:60, :60].max() == 0 and c[-10:, -10:].min() == 200


def test_unmatched_proposal_is_no_play():
    props = [Proposal(1300, 4, 10, "tag"), Proposal(9000, 8, 20, "tag")]
    plays = [{"card": "knight", "t_view": 1000, "col": 4, "row": 11}]
    ev = label_events(props, plays, video_ms=60_000)
    assert [(e["label"], e["proposed"]) for e in ev] == [("knight", True), ("no_play", True)]


def test_missed_play_is_added_unproposed():
    plays = [{"card": "zap", "t_view": 5000, "col": 9, "row": 8}]
    ev = label_events([], plays, video_ms=60_000)
    assert ev == [{"t_ms": 5300, "col": 9, "row": 8, "label": "zap", "proposed": False}]


def test_plays_outside_video_are_skipped():
    plays = [{"card": "zap", "t_view": -2000, "col": 9, "row": 8}, {"card": "zap", "t_view": 59_500, "col": 9, "row": 8}]
    assert label_events([], plays, video_ms=60_000) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd training && .venv/bin/python -m pytest tests/test_clips.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write the implementation**

```python
# training/selfplay/clips.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd training && .venv/bin/python -m pytest tests/test_clips.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Build clips for all aligned matches.** `training/selfplay_clips.py`:

```python
"""Clip dataset for every aligned match that has none yet -> ../dataset/selfplay_clips/."""
from pathlib import Path

from selfplay.clips import build_match

out = Path("../dataset/selfplay_clips")
for m in sorted(Path("../dataset/selfplay").iterdir()):
    if (m / "align.json").exists() and not (out / f"{m.name}_note14.npz").exists():
        build_match(m, out)
```

Run in the background: `cd training && .venv/bin/python selfplay_clips.py > runs/clips.log 2>&1`. Expected: one line per match and viewer; open one npz and view 3 clips of different classes as image strips (the card must be visible in its clip).

- [ ] **Step 6: Commit**

```bash
git add training/selfplay/clips.py training/tests/test_clips.py training/selfplay_clips.py
git commit -m "training: labeled 8-frame clip dataset from self-play videos"
```

---

### Task 5: Event classifier: train, evaluate, export

**Files:**
- Create: `training/selfplay/model.py`, `training/tests/test_model.py`, `training/train_plays.py`

**Interfaces:**
- Consumes: Task 4 npz files and `CLASSES`.
- Produces: `PlayNet(n_classes=123)`: input `[B, T, 3, 128, 128]` float (0..1, ImageNet-normalized inside), output logits `[B, 123]` for any T >= 1. Training writes `training/runs/plays/best.pt`, `training/runs/plays/val_preds.npz` (`p2s [N,123]`, `p05 [N,123]`, `y`, `match`, `t`) and exports `assets/models/plays.onnx` (input `clip` `[1, T, 3, 128, 128]`, dynamic T) and `assets/models/plays.txt` (class names, one per line).

- [ ] **Step 1: Write the failing tests**

```python
# training/tests/test_model.py
import torch

from selfplay.model import PlayNet


def test_shapes_for_full_and_early_clips():
    net = PlayNet(123, pretrained=False).eval()
    with torch.no_grad():
        assert net(torch.rand(2, 8, 3, 128, 128)).shape == (2, 123)
        assert net(torch.rand(2, 4, 3, 128, 128)).shape == (2, 123)


def test_learns_a_trivial_task():
    torch.manual_seed(0)
    net = PlayNet(3, pretrained=False)
    x = torch.zeros(6, 4, 3, 128, 128)
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    for i in range(6):
        x[i, :, y[i]] = 1.0  # class k = channel k lit
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(60):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(net(x), y)
        loss.backward()
        opt.step()
    net.eval()
    assert (net(x).argmax(1) == y).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd training && .venv/bin/python -m pytest tests/test_model.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write the model**

```python
# training/selfplay/model.py
"""Per-frame ResNet-18 features, pooled over time (mean + attention), one linear head."""
import torch
import torch.nn as nn
import torchvision

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


class PlayNet(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        r.fc = nn.Identity()
        self.backbone = r
        self.attn = nn.Linear(512, 1)
        self.head = nn.Linear(1024, n_classes)
        self.register_buffer("mean", MEAN)
        self.register_buffer("std", STD)

    def forward(self, clip):  # [B, T, 3, H, W], 0..1
        b, t = clip.shape[:2]
        x = ((clip - self.mean) / self.std).flatten(0, 1)
        f = self.backbone(x).view(b, t, 512)
        w = torch.softmax(self.attn(f), dim=1)
        return self.head(torch.cat([f.mean(1), (w * f).sum(1)], 1))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd training && .venv/bin/python -m pytest tests/test_model.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Write the training script**

```python
# training/train_plays.py
"""Train PlayNet on dataset/selfplay_clips, validate on held-out matches, export ONNX.

  .venv/bin/python train_plays.py [--epochs 12]
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from selfplay.clips import CLASSES
from selfplay.model import PlayNet

CLIPS = Path("../dataset/selfplay_clips")
OUT = Path("runs/plays")


def is_val(match_id):
    return int(hashlib.md5(match_id.encode()).hexdigest(), 16) % 10 == 0


class Clips(Dataset):
    def __init__(self, files, augment):
        self.items, self.augment = [], augment
        self.data = {}
        for f in files:
            d = np.load(f)
            self.data[f] = (d["X"], d["y"], d["t"])
            self.items += [(f, i) for i in range(len(d["y"]))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        f, i = self.items[k]
        X, y, t = self.data[f]
        x = torch.from_numpy(X[i]).permute(0, 3, 1, 2).float() / 255  # [8,3,128,128]
        if self.augment:
            x = x * (0.8 + 0.4 * torch.rand(1)) + 0.1 * (torch.rand(1) - 0.5)
            dx, dy = np.random.randint(-8, 9, 2)
            x = torch.roll(x, (int(dy), int(dx)), (2, 3))
            x = x.clamp(0, 1)
        return x, int(y[i]), f.name.rsplit("_", 1)[0], float(t[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()
    files = sorted(CLIPS.glob("*.npz"))
    train = Clips([f for f in files if not is_val(f.name.rsplit("_", 1)[0])], True)
    val = Clips([f for f in files if is_val(f.name.rsplit("_", 1)[0])], False)
    print(f"train {len(train)} clips, val {len(val)} clips, {len(files)} files", flush=True)
    counts = np.bincount([train.data[f][1][i] for f, i in train.items], minlength=len(CLASSES))
    weights = torch.tensor(1.0 / np.sqrt(np.maximum(counts, 1)), dtype=torch.float32).cuda()
    net = PlayNet(len(CLASSES)).cuda()
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=a.epochs * (len(train) // 32 + 1))
    dl = DataLoader(train, 32, shuffle=True, num_workers=4, drop_last=True)
    best = 0.0
    OUT.mkdir(parents=True, exist_ok=True)
    for ep in range(a.epochs):
        net.train()
        for x, y, _, _ in dl:
            x, y = x.cuda(), y.cuda()
            # Full clip and the early (first 4 frames) readout share the weights.
            loss = F.cross_entropy(net(x), y, weight=weights) + 0.5 * F.cross_entropy(net(x[:, :4]), y, weight=weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        acc2, acc05, preds = evaluate(net, val)
        print(f"epoch {ep}: val top-1 @2s {acc2:.4f} @0.5s {acc05:.4f}", flush=True)
        if acc2 >= best:
            best = acc2
            torch.save(net.state_dict(), OUT / "best.pt")
            np.savez(OUT / "val_preds.npz", **preds)
    net.load_state_dict(torch.load(OUT / "best.pt"))
    export(net.cpu().eval())


@torch.no_grad()
def evaluate(net, val):
    net.eval()
    p2, p05, ys, ms, ts = [], [], [], [], []
    for x, y, m, t in DataLoader(val, 64, num_workers=4):
        x = x.cuda()
        p2.append(torch.softmax(net(x), 1).cpu())
        p05.append(torch.softmax(net(x[:, :4]), 1).cpu())
        ys.append(y)
        ms += list(m)
        ts.append(t)
    p2, p05, ys = torch.cat(p2).numpy(), torch.cat(p05).numpy(), torch.cat(ys).numpy()
    plays = ys != len(CLASSES) - 1
    acc = lambda p: float((p.argmax(1)[plays] == ys[plays]).mean()) if plays.any() else 0.0
    return acc(p2), acc(p05), {"p2s": p2, "p05": p05, "y": ys, "match": np.array(ms), "t": torch.cat(ts).numpy()}


def export(net):
    Path("../assets/models").mkdir(exist_ok=True)
    torch.onnx.export(net, torch.rand(1, 8, 3, 128, 128), "../assets/models/plays.onnx", input_names=["clip"],
                      output_names=["logits"], dynamic_axes={"clip": {1: "frames"}}, opset_version=17)
    Path("../assets/models/plays.txt").write_text("\n".join(CLASSES) + "\n")
    print("exported ../assets/models/plays.onnx")


if __name__ == "__main__":
    main()
```

Accuracy is over clips whose label is a card (logged plays), per spec metric "top-1 per logged play"; `no_play` accuracy is reported in Task 7.

- [ ] **Step 6: Smoke-train.** Run: `cd training && .venv/bin/python train_plays.py --epochs 1 > runs/plays_smoke.log 2>&1` (background). Expected: one epoch line, `best.pt`, `val_preds.npz`, and `assets/models/plays.onnx` exist. If there is no validation match yet (fewer than ~10 matches), the val line reads 0 and that is fine for the smoke run.

- [ ] **Step 7: Commit** (the model files are not committed until Task 7 decides to ship them)

```bash
git add training/selfplay/model.py training/tests/test_model.py training/train_plays.py
git commit -m "training: PlayNet event classifier, training and ONNX export"
```

---

### Task 6: Deck inference

**Files:**
- Create: `training/selfplay/deck.py`, `training/tests/test_deck.py`

**Interfaces:**
- Consumes: per-play probability vectors over `CLASSES` (Task 5 `val_preds.npz`), in time order per match and viewer.
- Produces: `refine(probs: np.ndarray[N, C], no_play: int) -> np.ndarray[N]` returning one class index per play using: at most 8 distinct cards per opponent; a card just played cannot come back within the next 3 plays (4-card hand cycle); a prediction with probability ≥ 0.98 is always kept.

Algorithm (beam search over the play sequence, beam 64): each state is (cards seen so far as a tuple, last 3 plays, log-prob). For play i, extend each state by the top 10 classes of `probs[i]` (excluding `no_play`); drop extensions that make more than 8 distinct cards or repeat a card in the last 3 plays, unless that class has p ≥ 0.98. Keep the 64 best by log-prob; return the best final path.

- [ ] **Step 1: Write the failing tests**

```python
# training/tests/test_deck.py
import numpy as np

from selfplay.deck import refine


def onehot_mix(n_classes, best, second, p_best):
    p = np.full(n_classes, 1e-4)
    p[best], p[second] = p_best, 1 - p_best - 1e-4 * (n_classes - 2)
    return p


def test_cycle_flips_an_unsure_repeat():
    # Plays: 0, 1, then 0 again at 55% (impossible right after) vs 2 at 45%: pick 2.
    c = 10
    probs = np.stack([onehot_mix(c, 0, 1, 0.99), onehot_mix(c, 1, 0, 0.99), onehot_mix(c, 0, 2, 0.55)])
    assert list(refine(probs, no_play=c - 1)) == [0, 1, 2]


def test_confident_prediction_wins_over_cycle():
    c = 10
    probs = np.stack([onehot_mix(c, 0, 1, 0.99), onehot_mix(c, 0, 2, 0.99)])
    assert list(refine(probs, no_play=c - 1)) == [0, 0]


def test_ninth_distinct_card_is_replaced_by_a_known_one():
    c = 20
    rows = [onehot_mix(c, k, (k + 1) % 8, 0.99) for k in range(8)] * 2  # 8 cards, cycled
    rows.append(onehot_mix(c, 15, 3, 0.6))  # unsure 9th distinct card vs known card 3
    out = refine(np.stack(rows), no_play=c - 1)
    assert out[-1] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd training && .venv/bin/python -m pytest tests/test_deck.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write the implementation**

```python
# training/selfplay/deck.py
"""Deck inference (spec section 3): the opponent has 8 cards and a 4-card hand cycle."""
import math

import numpy as np

BEAM, TOP_K, SURE, DECK, CYCLE = 64, 10, 0.98, 8, 3


def refine(probs, no_play):
    beams = [((), (), 0.0, [])]  # (cards seen, last plays, logp, path)
    for p in probs:
        order = [k for k in np.argsort(-p) if k != no_play][:TOP_K]
        nxt = []
        for seen, last, lp, path in beams:
            for k in order:
                k = int(k)
                sure = p[k] >= SURE
                new_seen = seen if k in seen else seen + (k,)
                if not sure and (len(new_seen) > DECK or k in last):
                    continue
                nxt.append((new_seen, (last + (k,))[-CYCLE:], lp + math.log(max(p[k], 1e-12)), path + [k]))
        if not nxt:  # every option broke a rule: fall back to the plain argmax
            k = int(order[0])
            nxt = [(s, (l + (k,))[-CYCLE:], lp, path + [k]) for s, l, lp, path in beams]
        beams = sorted(nxt, key=lambda b: -b[2])[:BEAM]
    return np.array(beams[0][3])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd training && .venv/bin/python -m pytest tests/test_deck.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add training/selfplay/deck.py training/tests/test_deck.py
git commit -m "training: deck inference (8 cards, hand cycle) over per-play predictions"
```

---

### Task 7: Full training and the evaluation report

**Files:**
- Create: `training/plays_report.py`

**Interfaces:**
- Consumes: `runs/plays/val_preds.npz` (Task 5), `refine` (Task 6), `runs/recall.log` (Task 3).
- Produces: `training/runs/plays/report.md` with: top-1 per logged play at 0.5 s and 2 s, each with and without deck inference; `no_play` precision/recall; the 15 most common confusion pairs; per-card accuracy for the 20 worst cards; proposal recall (copied from `runs/recall.log`).

- [ ] **Step 1: Write the report script**

```python
# training/plays_report.py
"""Spec section 4 metrics on held-out matches -> runs/plays/report.md."""
import collections
from pathlib import Path

import numpy as np

from selfplay.clips import CLASSES
from selfplay.deck import refine

NP = len(CLASSES) - 1
d = np.load("runs/plays/val_preds.npz")
y, match, t = d["y"], d["match"], d["t"]
lines = ["# Play classifier report", ""]
for name, p in (("0.5 s", d["p05"]), ("2 s", d["p2s"])):
    plays = y != NP
    raw = p.argmax(1)
    deck = raw.copy()
    for m in np.unique(match):
        idx = np.where((match == m) & plays)[0]
        idx = idx[np.argsort(t[idx])]
        if len(idx):
            deck[idx] = refine(p[idx], NP)
    lines.append(f"- top-1 @ {name}: {np.mean(raw[plays] == y[plays]):.4f} raw, {np.mean(deck[plays] == y[plays]):.4f} with deck inference ({plays.sum()} plays)")
p2 = d["p2s"].argmax(1)
tp = np.sum((p2 == NP) & (y == NP))
lines.append(f"- no_play precision {tp / max(1, np.sum(p2 == NP)):.3f}, recall {tp / max(1, np.sum(y == NP)):.3f}")
conf = collections.Counter((CLASSES[a], CLASSES[b]) for a, b in zip(y, p2) if a != b)
lines += ["", "## Top confusions (true -> predicted)", ""] + [f"- {a} -> {b}: {n}" for (a, b), n in conf.most_common(15)]
per = {CLASSES[k]: np.mean(p2[y == k] == k) for k in np.unique(y) if k != NP}
lines += ["", "## Worst cards", ""] + [f"- {c}: {a:.3f}" for c, a in sorted(per.items(), key=lambda x: x[1])[:20]]
rec = Path("runs/recall.log")
if rec.exists():
    lines += ["", "## Proposal recall", ""] + [l for l in rec.read_text().splitlines() if l.startswith(("recall", "overall"))]
text = "\n".join(lines) + "\n"
Path("runs/plays/report.md").write_text(text)
print(text)
```

- [ ] **Step 2: Full pipeline once the collection run has finished** (in the background, one log per stage): `selfplay_align.py`, `selfplay_recall.py`, `selfplay_clips.py`, `train_plays.py --epochs 12`, then `plays_report.py`.
Expected: report written; the spec target is ≥ 0.99 top-1 @ 2 s with deck inference. If below, the report's worst cards and confusions say which decks to oversample in the next collection round (spec "Risks": rare and confusable cards).

- [ ] **Step 3: Commit** the script and the report; commit `assets/models/plays.onnx` and `plays.txt` only if the 2 s accuracy with deck inference is ≥ 0.95 (else leave them uncommitted and note it).

```bash
git add training/plays_report.py
git commit -m "training: play classifier evaluation report"
```
