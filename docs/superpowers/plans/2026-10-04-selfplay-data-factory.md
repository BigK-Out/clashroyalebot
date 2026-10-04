# Self-play Data Factory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unattended Friendly Battles between the main phone (observer, runs our bot and records) and a second phone (scripted sparring opponent that logs every card it plays), producing `dataset/selfplay/<match_id>/` folders with frames and exact, time-aligned enemy-play labels for all 122 cards.

**Architecture:** A new `selfplay` crate holds the pure logic: play records and logs, the sparring policy, clock alignment, deck rotation and screen-state checks. A `sparring` binary drives the Note 9 from adb screenshots using the existing `vision` hand and elixir readers plus `input::Deployer`. The bot gets a `--selfplay-dir` recording mode. An orchestrator binary, `selfplay`, starts Friendly Battles on both phones, runs observer and sparring for each match, and rotates the sparring deck.

**Tech Stack:** Rust 2024 workspace (existing crates `capture`, `calib`, `vision`, `input`, `state`, `brain`), `serde`/`serde_json`, `fastrand` (already in Cargo.lock), adb (`~/Android/Sdk/platform-tools/adb`), scrcpy → v4l2loopback `/dev/video10` for the observer.

**Spec:** `docs/superpowers/specs/2026-10-04-enemy-card-id-design.md` (section 1, build order steps 1-3). Event detection, the classifier, deck inference and bot integration (build order 4-7) get their own plan once data from this one exists.

## Global Constraints

- Observer phone: Redmi Note 14, serial `4xwskfkr7xp7w4xo`, 1080x2400, main account (Khazar).
- Sparring phone: Redmi Note 9, serial `d8c5ce8a0406`, 1080x2340, second account. Deck editing and sparring play only ever target this serial.
- Game package on both: `nullsroyale.rel.free`.
- Never send `KEYCODE_BACK` (it opens "Exit Clash Royale?"). Close popups with their own buttons.
- Never `pkill -f` with a pattern that could match the agent's own shell. Stop processes by PID.
- Long runs go to the background with a log file; no long foreground sleeps.
- `target/release/bot` keeps its existing command line and behaviour when the new flag is absent.
- All timestamps in logs are host wall-clock milliseconds since the Unix epoch (`host_ms`), taken on the PC.
- Observer frames: 576x1280 (scrcpy `--max-size=1280`), saved as JPEG quality 92.
- Output layout per match: `dataset/selfplay/<match_id>/{frames/<host_ms>.jpg, plays.jsonl, observer.jsonl, meta.json}`.
- At most one champion per sparring deck. A deck is exactly 8 distinct cards.
- Commit after every task with the attribution line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- **Sparring hand misread** (a slot read as the wrong card): the play must not be logged as that card. `sparring` re-reads the hand after the tap and logs `verified: false` when the played slot's card did not leave the hand (Task 5 test `unverified_when_slot_unchanged`).
- **Note 9 adb disconnect mid-match** (seen during setup): `sparring` and `selfplay` must stop the match cleanly with an error in the log, and the match folder is marked `"complete": false` in `meta.json` (Task 8 test `incomplete_match_marked`).
- **Not on the expected screen** (popup, reconnect, chest screen): orchestration must not tap blindly; each flow step checks the screen state first and stops after bounded retries (Task 8 test `flow_refuses_unknown_screen`).
- **Elixir bar unreadable** (menus, transitions, end screen): `sparring` must not play; the policy returns `None` without elixir (Task 4 test `no_elixir_no_play`).
- **Clock offset outliers** (an enemy play the observer's tag detector never saw): alignment uses the median of matched deltas and ignores unmatched plays (Task 7 test `median_ignores_outliers`).

---

### Task 1: Screenshots from adb as frames

**Files:**
- Create: `crates/capture/src/adb_screencap.rs`
- Modify: `crates/capture/src/lib.rs` (add `mod adb_screencap; pub use adb_screencap::{decode_png_scaled, screencap};`)
- Test: in `crates/capture/src/adb_screencap.rs`

**Interfaces:**
- Produces: `pub fn screencap(adb: &str, serial: &str, width: u32) -> anyhow::Result<Frame>` and `pub fn decode_png_scaled(png: &[u8], width: u32) -> anyhow::Result<Frame>`. Frames are scaled to `width` with the aspect kept (Note 9: 576x1248). `seq` is 0 and `captured_at` is the time the screenshot returned.

- [ ] **Step 1: Write the failing test**

```rust
// crates/capture/src/adb_screencap.rs
#[cfg(test)]
mod tests {
    use super::decode_png_scaled;

    fn png(w: u32, h: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(w, h, |x, y| image::Rgb([(x % 256) as u8, (y % 256) as u8, 7]));
        let mut out = std::io::Cursor::new(Vec::new());
        img.write_to(&mut out, image::ImageFormat::Png).unwrap();
        out.into_inner()
    }

    #[test]
    fn scales_to_width_keeping_aspect() {
        let f = decode_png_scaled(&png(1080, 2340), 576).unwrap();
        assert_eq!((f.width, f.height), (576, 1248));
        assert_eq!(f.rgb.len(), 576 * 1248 * 3);
    }

    #[test]
    fn rejects_garbage() {
        assert!(decode_png_scaled(b"error: device not found", 576).is_err());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p capture adb_screencap`
Expected: FAIL to compile, `decode_png_scaled` not found.

- [ ] **Step 3: Write minimal implementation**

```rust
// crates/capture/src/adb_screencap.rs
//! Screenshots over adb (`exec-out screencap -p`) as frames, for phones without a
//! scrcpy/v4l2 feed (the sparring phone). About 300 ms per call; fine for decisions
//! made every second or two, not for recording.

use std::process::Command;
use std::time::Instant;

use anyhow::{Context, bail};

use crate::Frame;

/// Decodes a PNG screenshot and scales it to `width` (aspect kept), like the 576-px feed.
pub fn decode_png_scaled(png: &[u8], width: u32) -> anyhow::Result<Frame> {
    let img = image::load_from_memory_with_format(png, image::ImageFormat::Png)
        .context("screenshot is not a PNG")?
        .into_rgb8();
    let height = (img.height() as f64 * width as f64 / img.width() as f64).round() as u32;
    let img = image::imageops::resize(&img, width, height, image::imageops::FilterType::Triangle);
    Ok(Frame { seq: 0, width, height, rgb: img.into_raw(), captured_at: Instant::now() })
}

/// Takes a screenshot of device `serial`.
pub fn screencap(adb: &str, serial: &str, width: u32) -> anyhow::Result<Frame> {
    let out = Command::new(adb)
        .args(["-s", serial, "exec-out", "screencap", "-p"])
        .output()
        .with_context(|| format!("run {adb} screencap"))?;
    if !out.status.success() {
        bail!("screencap on {serial} failed: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    decode_png_scaled(&out.stdout, width)
}
```

Add `image` features if missing: `crates/capture/Cargo.toml` already has `image` with `png`; no change needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p capture adb_screencap`
Expected: PASS (2 tests).

- [ ] **Step 5: Live check on the Note 9**

Run: `cargo run --release -p capture --example screencap_once -- d8c5ce8a0406 /tmp/n9.png` after adding this example:

```rust
// crates/capture/examples/screencap_once.rs
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let adb = format!("{}/Android/Sdk/platform-tools/adb", std::env::var("HOME")?);
    let f = capture::screencap(&adb, &a[1], 576)?;
    image::RgbImage::from_raw(f.width, f.height, f.rgb).unwrap().save(&a[2])?;
    println!("{}x{}", f.width, f.height);
    Ok(())
}
```

Expected: prints `576x1248`, and `/tmp/n9.png` shows the Note 9 screen.

- [ ] **Step 6: Commit**

```bash
git add crates/capture
git commit -m "capture: screenshots over adb as scaled frames

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Note 9 calibration

**Files:**
- Create: `calibration_note9.toml` (made with the existing `calibrate` tool)
- Create: `fixtures/note9/` (3-5 battle screenshots from the Note 9, 576x1248 PNG)
- Modify: `crates/calib/src/lib.rs` (extend the `repo_calibration` test module)

**Interfaces:**
- Consumes: `capture::screencap` (Task 1).
- Produces: `calibration_note9.toml`, a `calib::Calibration` for 576x1248 frames of the Note 9 (elixir bar, 4 card slots, next card, arena corners).

- [ ] **Step 1: Capture battle screenshots.** Start a Friendly Battle by hand between the two accounts (Note 9 → Friends → Khazar → Friendly Battle, accept on the Note 14). During the battle, run the Task 1 example 5 times, about 10 s apart, saving `fixtures/note9/battle_1.png` ... `battle_5.png`. Include a moment where a card is raised (tap a card on the Note 9 by hand just before one screenshot).

- [ ] **Step 2: Calibrate.** Run `cargo run --release -p calibrate -- fixtures/note9 -o calibration_note9.toml`. Mark the elixir bar (0 end to 10 end), the 4 card slots, the next-card preview and the 4 arena corners (the red no-deploy outline corners, as in `calibration.toml`). Check the tile grid overlay lines up with the bridges and towers on every fixture, then save.

- [ ] **Step 3: Write the failing test**

```rust
// crates/calib/src/lib.rs, inside mod repo_calibration
#[test]
fn note9_calibration_loads_and_maps() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let c = Calibration::load(root.join("calibration_note9.toml")).unwrap();
    let m = c.arena_mapping().unwrap();
    // Bridges are at rows 15-16; the left bridge tile must be in the upper-middle of the screen.
    let p = m.tile_center(3, 15);
    assert!((0.1..0.35).contains(&p.x) && (0.35..0.6).contains(&p.y), "{p:?}");
    assert!(c.card_slots.iter().all(|s| s.y > 0.8), "card slots at the bottom");
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p calib repo_calibration`
Expected: PASS (fails only if the file is missing or the corners were marked wrongly).

- [ ] **Step 5: Check perception on the fixtures**

Add this test to `crates/vision/tests/fixtures.rs` (the existing fixture test file):

```rust
#[test]
fn note9_elixir_and_hand_read() {
    let calib = calib::Calibration::load(root().join("calibration_note9.toml")).unwrap();
    let cards = vision::CardLibrary::load(root().join("assets/cards")).unwrap();
    for i in 1..=5 {
        let f = capture::load_rgb(root().join(format!("fixtures/note9/battle_{i}.png"))).unwrap();
        assert!(vision::read_elixir(&f, &calib).is_some(), "elixir on battle_{i}");
        let hand = cards.read_hand(&f, &calib);
        let known = hand.slots.iter().filter(|s| s.name().is_some()).count();
        assert!(known >= 3, "battle_{i}: {:?}", hand.slots);
    }
}
```

Run: `cargo test -p vision --test fixtures note9`
Expected: PASS. The Hog 2.6 templates in `assets/cards` were cut from the Note 14 feed. If hands do not read, re-cut templates from these fixtures with `cargo run -p vision --bin extract_card -- fixtures/note9/battle_1.png <slot> <name> --calib calibration_note9.toml --out assets/cards` using `@n9` variants (for example `hog_rider@n9.png`), then rerun.

- [ ] **Step 6: Commit**

```bash
git add calibration_note9.toml fixtures/note9 crates/calib crates/vision/tests assets/cards
git commit -m "Note 9 calibration and battle fixtures

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `selfplay` crate: play records, logs, host clock

**Files:**
- Create: `crates/selfplay/Cargo.toml`, `crates/selfplay/src/lib.rs`, `crates/selfplay/src/record.rs`
- Test: in `crates/selfplay/src/record.rs`

**Interfaces:**
- Produces:
  - `pub fn host_ms() -> u64` (wall-clock ms since the Unix epoch)
  - `#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)] pub struct PlayRecord { pub host_ms: u64, pub card: String, pub slot: usize, pub tile: (u32, u32), pub hand_before: [Option<String>; 4], pub elixir_read: Option<u8>, pub verified: bool }`
  - `#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)] pub struct ObserverRecord { pub host_ms: u64, pub kind: String, pub card: Option<String>, pub tile: Option<(u32, u32)>, pub enemies: Vec<(u32, u32)> }` (`kind`: `"deploy"`, `"battle_start"`, `"battle_end"`, `"new_enemy"`)
  - `pub struct JsonlLog<T>` with `pub fn create(path: &Path) -> anyhow::Result<Self>`, `pub fn append(&mut self, rec: &T) -> anyhow::Result<()>` (flushes each line), and `pub fn read_jsonl<T: DeserializeOwned>(path: &Path) -> anyhow::Result<Vec<T>>`
  - `#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)] pub struct MatchMeta { pub match_id: String, pub observer_serial: String, pub sparring_serial: String, pub observer_deck: Vec<String>, pub sparring_deck: Vec<String>, pub clock_offset_ms: Option<i64>, pub complete: bool, pub error: Option<String> }` with `pub fn save(&self, dir: &Path) -> anyhow::Result<()>` (writes `meta.json`)

- [ ] **Step 1: Create the crate**

```toml
# crates/selfplay/Cargo.toml
[package]
name = "selfplay"
edition.workspace = true
version.workspace = true
publish.workspace = true

[dependencies]
anyhow = "1.0.104"
serde = { version = "1.0.229", features = ["derive"] }
serde_json = "1"
fastrand = "2"
brain = { version = "0.1.0", path = "../brain" }

[dev-dependencies]
tempfile = "3"
```

```rust
// crates/selfplay/src/lib.rs
//! Self-play data factory: sparring policy, play logs, clock alignment, deck rotation.

pub mod record;

pub use record::{JsonlLog, MatchMeta, ObserverRecord, PlayRecord, host_ms, read_jsonl};
```

- [ ] **Step 2: Write the failing test**

```rust
// crates/selfplay/src/record.rs
#[cfg(test)]
mod tests {
    use super::*;

    fn play(ms: u64, card: &str) -> PlayRecord {
        PlayRecord {
            host_ms: ms,
            card: card.into(),
            slot: 2,
            tile: (3, 17),
            hand_before: [Some("hog_rider".into()), None, Some(card.into()), Some("the_log".into())],
            elixir_read: Some(7),
            verified: true,
        }
    }

    #[test]
    fn jsonl_roundtrip_and_append() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("plays.jsonl");
        let mut log = JsonlLog::create(&path).unwrap();
        log.append(&play(1, "musketeer")).unwrap();
        log.append(&play(2, "cannon")).unwrap();
        let back: Vec<PlayRecord> = read_jsonl(&path).unwrap();
        assert_eq!(back, vec![play(1, "musketeer"), play(2, "cannon")]);
        // One JSON object per line, readable by Python's json.loads per line.
        assert_eq!(std::fs::read_to_string(&path).unwrap().lines().count(), 2);
    }

    #[test]
    fn meta_saves_json() {
        let dir = tempfile::tempdir().unwrap();
        let m = MatchMeta { match_id: "m1".into(), complete: false, error: Some("adb gone".into()), ..Default::default() };
        m.save(dir.path()).unwrap();
        let back: MatchMeta = serde_json::from_str(&std::fs::read_to_string(dir.path().join("meta.json")).unwrap()).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn host_clock_is_epoch_ms() {
        assert!(host_ms() > 1_700_000_000_000);
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p selfplay record`
Expected: FAIL to compile (types not defined).

- [ ] **Step 4: Write the implementation**

```rust
// crates/selfplay/src/record.rs
//! Labels and logs: one JSON object per line, written as they happen (crash-safe).

use std::io::Write;
use std::marker::PhantomData;
use std::path::Path;

use anyhow::Context;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// Host wall-clock time in ms since the Unix epoch: the one clock both phones' logs share.
pub fn host_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// One card the sparring phone played (the label for an enemy play on the observer).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PlayRecord {
    /// When the deploy taps finished on the sparring phone.
    pub host_ms: u64,
    pub card: String,
    pub slot: usize,
    /// Tile in the sparring phone's own view (its half is rows 17..31).
    pub tile: (u32, u32),
    pub hand_before: [Option<String>; 4],
    pub elixir_read: Option<u8>,
    /// The played slot's card left the hand afterwards (the tap really played this card).
    pub verified: bool,
}

/// Something the observer bot did or saw.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ObserverRecord {
    pub host_ms: u64,
    /// "deploy", "battle_start", "battle_end", "new_enemy".
    pub kind: String,
    pub card: Option<String>,
    pub tile: Option<(u32, u32)>,
    pub enemies: Vec<(u32, u32)>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct MatchMeta {
    pub match_id: String,
    pub observer_serial: String,
    pub sparring_serial: String,
    pub observer_deck: Vec<String>,
    pub sparring_deck: Vec<String>,
    /// Observer time minus sparring tap time for the same play (Task 7).
    pub clock_offset_ms: Option<i64>,
    pub complete: bool,
    pub error: Option<String>,
}

impl MatchMeta {
    pub fn save(&self, dir: &Path) -> anyhow::Result<()> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("meta.json"), serde_json::to_string_pretty(self)?)?;
        Ok(())
    }
}

pub struct JsonlLog<T> {
    file: std::fs::File,
    _t: PhantomData<T>,
}

impl<T: Serialize> JsonlLog<T> {
    /// Creates (or appends to) a JSONL file; parent directories are created.
    pub fn create(path: &Path) -> anyhow::Result<Self> {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("open {}", path.display()))?;
        Ok(Self { file, _t: PhantomData })
    }

    pub fn append(&mut self, rec: &T) -> anyhow::Result<()> {
        let mut line = serde_json::to_string(rec)?;
        line.push('\n');
        self.file.write_all(line.as_bytes())?;
        self.file.flush()?;
        Ok(())
    }
}

pub fn read_jsonl<T: DeserializeOwned>(path: &Path) -> anyhow::Result<Vec<T>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    text.lines().filter(|l| !l.trim().is_empty()).map(|l| Ok(serde_json::from_str(l)?)).collect()
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p selfplay`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add crates/selfplay Cargo.lock
git commit -m "selfplay: play/observer records, JSONL logs, match meta

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Sparring policy (pure)

**Files:**
- Create: `crates/selfplay/src/policy.rs`
- Modify: `crates/selfplay/src/lib.rs` (add `pub mod policy; pub use policy::{SparringPolicy, SparPlay};`)
- Test: in `crates/selfplay/src/policy.rs`

**Interfaces:**
- Consumes: `brain::cards()` (card table: `elixir: Option<u8>`, `kind: Option<String>` = "Troop" | "Spell" | "Building" | "Tower Troop").
- Produces:
  - `#[derive(Debug, Clone, PartialEq)] pub struct SparPlay { pub slot: usize, pub card: String, pub tile: (u32, u32), pub enemy_half: bool }`
  - `pub struct SparringPolicy { rng: fastrand::Rng, last_play_ms: u64, wait_ms: u64 }` with `pub fn new(seed: u64) -> Self` and `pub fn decide(&mut self, now_ms: u64, elixir: Option<u8>, hand: &[Option<String>; 4]) -> Option<SparPlay>`

Rules (from the spec): random affordable card; troops at the bridge (rows 17-18 at col 3 or 14), a back corner (rows 28-30) or in front of its own towers (rows 20-24); buildings at the centre (cols 7-10, rows 20-23); spells on the observer's half (rows 2-14, `enemy_half: true`); varied timing: after each play wait a random 1-6 s; at 10 elixir play immediately.

- [ ] **Step 1: Write the failing tests**

```rust
// crates/selfplay/src/policy.rs
#[cfg(test)]
mod tests {
    use super::*;

    fn hand(c: [&str; 4]) -> [Option<String>; 4] {
        c.map(|s| (!s.is_empty()).then(|| s.to_string()))
    }

    #[test]
    fn no_elixir_no_play() {
        let mut p = SparringPolicy::new(1);
        assert_eq!(p.decide(100_000, None, &hand(["hog_rider", "cannon", "the_log", "skeletons"])), None);
    }

    #[test]
    fn only_affordable_cards() {
        for seed in 0..200 {
            let mut p = SparringPolicy::new(seed);
            let play = p.decide(100_000, Some(2), &hand(["hog_rider", "musketeer", "the_log", "skeletons"]));
            if let Some(pl) = play {
                assert!(pl.card == "the_log" || pl.card == "skeletons", "{pl:?}");
            }
        }
    }

    #[test]
    fn placement_by_card_type() {
        for seed in 0..300 {
            let mut p = SparringPolicy::new(seed);
            let Some(pl) = p.decide(100_000, Some(10), &hand(["hog_rider", "cannon", "fireball", "skeletons"])) else {
                panic!("full elixir must play");
            };
            match pl.card.as_str() {
                "fireball" => assert!(pl.enemy_half && (2..=14).contains(&pl.tile.1), "{pl:?}"),
                "cannon" => assert!(!pl.enemy_half && (7..=10).contains(&pl.tile.0) && (20..=23).contains(&pl.tile.1), "{pl:?}"),
                _ => assert!(!pl.enemy_half && (17..=31).contains(&pl.tile.1) && pl.tile.0 < 18, "{pl:?}"),
            }
        }
    }

    #[test]
    fn waits_between_plays_unless_full() {
        let mut p = SparringPolicy::new(7);
        let h = hand(["hog_rider", "cannon", "the_log", "skeletons"]);
        assert!(p.decide(100_000, Some(10), &h).is_some());
        // Right after a play, with elixir but not full: wait.
        assert_eq!(p.decide(100_200, Some(8), &h), None);
        // Full elixir overrides the wait.
        assert!(p.decide(100_300, Some(10), &h).is_some());
    }

    #[test]
    fn unknown_slots_are_skipped() {
        let mut p = SparringPolicy::new(3);
        let pl = p.decide(100_000, Some(10), &hand(["", "", "skeletons", ""])).unwrap();
        assert_eq!((pl.slot, pl.card.as_str()), (2, "skeletons"));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p selfplay policy`
Expected: FAIL to compile.

- [ ] **Step 3: Write the implementation**

```rust
// crates/selfplay/src/policy.rs
//! The sparring phone's play policy: random but legal, varied placements and timing, so the
//! observer sees every card in many situations. Not trying to win.

#[derive(Debug, Clone, PartialEq)]
pub struct SparPlay {
    pub slot: usize,
    pub card: String,
    /// Tile in the sparring phone's own view.
    pub tile: (u32, u32),
    /// Spells on the observer's half.
    pub enemy_half: bool,
}

pub struct SparringPolicy {
    rng: fastrand::Rng,
    last_play_ms: u64,
    wait_ms: u64,
}

impl SparringPolicy {
    pub fn new(seed: u64) -> Self {
        Self { rng: fastrand::Rng::with_seed(seed), last_play_ms: 0, wait_ms: 0 }
    }

    fn tile_for(&mut self, kind: &str) -> ((u32, u32), bool) {
        let r = &mut self.rng;
        match kind {
            "Spell" => ((r.u32(2..=15), r.u32(2..=14)), true),
            "Building" => ((r.u32(7..=10), r.u32(20..=23)), false),
            _ => {
                let col = if r.bool() { r.u32(2..=4) } else { r.u32(13..=15) };
                let row = match r.u32(0..3) {
                    0 => r.u32(17..=18), // bridge
                    1 => r.u32(28..=30), // back
                    _ => r.u32(20..=24), // in front of the tower
                };
                ((col, row), false)
            }
        }
    }

    pub fn decide(&mut self, now_ms: u64, elixir: Option<u8>, hand: &[Option<String>; 4]) -> Option<SparPlay> {
        let elixir = elixir?;
        if elixir < 10 && now_ms < self.last_play_ms + self.wait_ms {
            return None;
        }
        let cards = brain::cards();
        let options: Vec<(usize, &str, String)> = hand
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let c = c.as_deref()?;
                let card = cards.get(c)?;
                (card.elixir? <= elixir).then(|| (i, c, card.kind.clone().unwrap_or_default()))
            })
            .collect();
        if options.is_empty() {
            return None;
        }
        let (slot, card, kind) = options[self.rng.usize(..options.len())].clone();
        let (tile, enemy_half) = self.tile_for(&kind);
        self.last_play_ms = now_ms;
        self.wait_ms = self.rng.u64(1_000..=6_000);
        Some(SparPlay { slot, card: card.to_string(), tile, enemy_half })
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p selfplay policy`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/selfplay
git commit -m "selfplay: random legal sparring policy

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `sparring` binary

**Files:**
- Create: `bins/sparring/Cargo.toml`, `bins/sparring/src/main.rs`, `bins/sparring/src/lib.rs`
- Test: `bins/sparring/src/lib.rs` (unit tests with a fake tap backend)

**Interfaces:**
- Consumes: `capture::screencap` (Task 1), `calibration_note9.toml` (Task 2), `selfplay::{SparringPolicy, PlayRecord, JsonlLog, host_ms}` (Tasks 3-4), `vision::{read_elixir, CardLibrary}`, `input::{AdbShell, Deployer, TapBackend}`.
- Produces: CLI `sparring --serial d8c5ce8a0406 --calibration calibration_note9.toml --cards assets/cards --out <match_dir>/plays.jsonl [--seed N] [--dry-run]`. It plays until the elixir bar has been unreadable for 15 s after a battle was seen (match over), then exits 0. On adb errors it exits non-zero with the error message.
- Produces (lib): `pub fn step<B: TapBackend>(frame: &Frame, ctx: &mut Ctx<B>) -> anyhow::Result<Option<PlayRecord>>` and `pub fn verify(before: &[Option<String>; 4], after: &[Option<String>; 4], slot: usize, card: &str) -> bool`.

- [ ] **Step 1: Write the failing tests**

```rust
// bins/sparring/src/lib.rs
#[cfg(test)]
mod tests {
    use super::verify;

    fn h(c: [&str; 4]) -> [Option<String>; 4] {
        c.map(|s| (!s.is_empty()).then(|| s.to_string()))
    }

    #[test]
    fn verified_when_card_left_slot() {
        assert!(verify(&h(["hog_rider", "cannon", "the_log", "skeletons"]), &h(["hog_rider", "", "the_log", "skeletons"]), 1, "cannon"));
        assert!(verify(&h(["hog_rider", "cannon", "the_log", "skeletons"]), &h(["hog_rider", "fireball", "the_log", "skeletons"]), 1, "cannon"));
    }

    #[test]
    fn unverified_when_slot_unchanged() {
        assert!(!verify(&h(["hog_rider", "cannon", "the_log", "skeletons"]), &h(["hog_rider", "cannon", "the_log", "skeletons"]), 1, "cannon"));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p sparring`
Expected: FAIL to compile.

- [ ] **Step 3: Write the implementation**

```toml
# bins/sparring/Cargo.toml
[package]
name = "sparring"
edition.workspace = true
version.workspace = true
publish.workspace = true

[dependencies]
anyhow = "1.0.104"
calib = { path = "../../crates/calib" }
capture = { path = "../../crates/capture" }
clap = { version = "4.6.7", features = ["derive"] }
input = { path = "../../crates/input" }
selfplay = { path = "../../crates/selfplay" }
tracing = "0.1.44"
tracing-subscriber = { version = "0.3.23", features = ["env-filter"] }
vision = { path = "../../crates/vision" }
```

```rust
// bins/sparring/src/lib.rs
//! One sparring step: read elixir and hand from a screenshot, maybe play, log the play.

use calib::Calibration;
use capture::Frame;
use input::{Deployer, TapBackend};
use selfplay::{PlayRecord, SparringPolicy, host_ms};
use vision::CardLibrary;

pub struct Ctx<B: TapBackend> {
    pub calib: Calibration,
    pub cards: CardLibrary,
    pub policy: SparringPolicy,
    pub deployer: Deployer<B>,
    pub dry_run: bool,
}

pub fn hand_names(frame: &Frame, ctx: &Ctx<impl TapBackend>) -> [Option<String>; 4] {
    let hand = ctx.cards.read_hand(frame, &ctx.calib);
    std::array::from_fn(|i| hand.slots[i].name().map(str::to_string))
}

/// The played card left its slot (slot now empty or holding a different card).
pub fn verify(before: &[Option<String>; 4], after: &[Option<String>; 4], slot: usize, card: &str) -> bool {
    before[slot].as_deref() == Some(card) && after[slot].as_deref() != Some(card)
}

/// Decides on `frame`; if a card is played, returns its record with `verified: false`
/// (the caller re-reads the hand and sets it).
pub fn step<B: TapBackend>(frame: &Frame, ctx: &mut Ctx<B>) -> anyhow::Result<Option<PlayRecord>> {
    let elixir = vision::read_elixir(frame, &ctx.calib).map(|e| e.value);
    let hand = hand_names(frame, ctx);
    let Some(play) = ctx.policy.decide(host_ms(), elixir, &hand) else { return Ok(None) };
    if !ctx.dry_run {
        ctx.deployer.deploy(play.slot, play.tile.0, play.tile.1, play.enemy_half)?;
    }
    Ok(Some(PlayRecord {
        host_ms: host_ms(),
        card: play.card,
        slot: play.slot,
        tile: play.tile,
        hand_before: hand,
        elixir_read: elixir,
        verified: false,
    }))
}
```

```rust
// bins/sparring/src/main.rs
//! Scripted opponent for self-play on the second phone: plays random legal cards and logs
//! every play to plays.jsonl (the labels for enemy plays seen by the observer).

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Context;
use clap::Parser;
use input::{AdbShell, Deployer};
use selfplay::{JsonlLog, PlayRecord, SparringPolicy};
use sparring::{Ctx, hand_names, step, verify};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "d8c5ce8a0406")]
    serial: String,
    #[arg(long, default_value = "calibration_note9.toml")]
    calibration: PathBuf,
    #[arg(long, default_value = "assets/cards")]
    cards: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long)]
    dry_run: bool,
}

fn adb_path() -> String {
    format!("{}/Android/Sdk/platform-tools/adb", std::env::var("HOME").unwrap_or_default())
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let a = Args::parse();
    let adb = adb_path();
    let calib = calib::Calibration::load(&a.calibration)?;
    let shell = AdbShell::open(&adb, Some(&a.serial)).context("adb shell to sparring phone")?;
    let mut ctx = Ctx {
        deployer: Deployer::new(shell, calib.clone())?,
        calib,
        cards: vision::CardLibrary::load(&a.cards)?,
        policy: SparringPolicy::new(a.seed),
        dry_run: a.dry_run,
    };
    let mut log: JsonlLog<PlayRecord> = JsonlLog::create(&a.out)?;
    let (mut seen_battle, mut last_bar) = (false, Instant::now());
    loop {
        let frame = capture::screencap(&adb, &a.serial, 576).context("screencap")?;
        if vision::read_elixir(&frame, &ctx.calib).is_some() {
            seen_battle = true;
            last_bar = Instant::now();
        } else if seen_battle && last_bar.elapsed() > Duration::from_secs(15) {
            tracing::info!("no elixir bar for 15 s: match over");
            return Ok(());
        }
        if let Some(mut rec) = step(&frame, &mut ctx)? {
            std::thread::sleep(Duration::from_millis(600));
            let after = capture::screencap(&adb, &a.serial, 576).context("screencap after play")?;
            rec.verified = verify(&rec.hand_before, &hand_names(&after, &ctx), rec.slot, &rec.card);
            tracing::info!("played {} slot {} -> {:?} verified {}", rec.card, rec.slot + 1, rec.tile, rec.verified);
            log.append(&rec)?;
        }
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p sparring`
Expected: PASS (2 tests).

- [ ] **Step 5: Dry run against the fixtures**

Add to `bins/sparring/src/lib.rs` tests a dry-run step over `fixtures/note9/battle_*.png` with a fake backend:

```rust
    struct FakeTaps(Vec<(u32, u32)>);
    impl input::TapBackend for FakeTaps {
        fn screen_size(&self) -> (u32, u32) { (1080, 2340) }
        fn taps(&mut self, p: &[(u32, u32)]) -> anyhow::Result<()> { self.0.extend_from_slice(p); Ok(()) }
    }

    #[test]
    fn plays_on_fixture_frames() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let calib = calib::Calibration::load(root.join("calibration_note9.toml")).unwrap();
        let mut ctx = super::Ctx {
            deployer: input::Deployer::new(FakeTaps(Vec::new()), calib.clone()).unwrap(),
            calib,
            cards: vision::CardLibrary::load(root.join("assets/cards")).unwrap(),
            policy: selfplay::SparringPolicy::new(1),
            dry_run: false,
        };
        let mut played = 0;
        for i in 1..=5 {
            let f = capture::load_rgb(root.join(format!("fixtures/note9/battle_{i}.png"))).unwrap();
            // Fresh policy each frame so the inter-play wait never blocks the test.
            ctx.policy = selfplay::SparringPolicy::new(i);
            if let Some(r) = super::step(&f, &mut ctx).unwrap() {
                assert!(r.hand_before[r.slot].as_deref() == Some(r.card.as_str()));
                played += 1;
            }
        }
        assert!(played >= 3, "played {played}/5");
        assert_eq!(ctx.deployer.backend().0.len(), played * 2, "two taps per play");
    }
```

Run: `cargo test -p sparring`
Expected: PASS (3 tests).

- [ ] **Step 6: Live smoke test (one hand-started Friendly Battle)**

Start a Friendly Battle by hand. In the background run `target/release/sparring --out dataset/selfplay/smoke/plays.jsonl > dataset/selfplay/smoke/sparring.log 2>&1`. Watch the Note 9 play. Expected: the process exits 0 about 15 s after the match ends; `plays.jsonl` has 20+ lines; at least 90% have `"verified":true`.

- [ ] **Step 7: Commit**

```bash
git add bins/sparring Cargo.lock
git commit -m "sparring: scripted opponent on the second phone with play log

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Observer recording mode in the bot

**Files:**
- Create: `crates/selfplay/src/frames.rs`
- Modify: `crates/selfplay/src/lib.rs` (add `pub mod frames; pub use frames::FrameSchedule;`)
- Modify: `bins/bot/Cargo.toml` (add `selfplay = { path = "../../crates/selfplay" }`)
- Modify: `bins/bot/src/main.rs` (new `--selfplay-dir` flag; frame saving and `observer.jsonl` records)
- Test: in `crates/selfplay/src/frames.rs`

**Interfaces:**
- Consumes: `selfplay::{JsonlLog, ObserverRecord, host_ms}` (Task 3).
- Produces: `pub struct FrameSchedule` with `pub fn new() -> Self`, `pub fn burst(&mut self, now_ms: u64)` (10 fps for the next 3 s) and `pub fn should_save(&mut self, now_ms: u64) -> bool` (2 fps baseline, 10 fps during a burst). Bot flag: `--selfplay-dir <match_dir>` saves full 576x1280 frames as `<match_dir>/frames/<host_ms>.jpg` while in battle and appends `ObserverRecord`s to `<match_dir>/observer.jsonl`.

- [ ] **Step 1: Write the failing test**

```rust
// crates/selfplay/src/frames.rs
#[cfg(test)]
mod tests {
    use super::FrameSchedule;

    fn saved(s: &mut FrameSchedule, from: u64, to: u64) -> usize {
        (from..to).step_by(10).filter(|&t| s.should_save(t)).count()
    }

    #[test]
    fn two_fps_baseline() {
        let mut s = FrameSchedule::new();
        assert_eq!(saved(&mut s, 0, 10_000), 20);
    }

    #[test]
    fn ten_fps_burst_for_three_seconds() {
        let mut s = FrameSchedule::new();
        s.burst(1_000);
        assert_eq!(saved(&mut s, 1_000, 4_000), 30);
        assert_eq!(saved(&mut s, 4_000, 6_000), 4);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p selfplay frames`
Expected: FAIL to compile.

- [ ] **Step 3: Write the implementation**

```rust
// crates/selfplay/src/frames.rs
//! Which observer frames to keep: 2 fps normally, 10 fps for 3 s after something new
//! (a sparring play or new enemy units), where the classifier needs dense frames.

const BASE_MS: u64 = 500;
const BURST_MS: u64 = 100;
const BURST_LEN_MS: u64 = 3_000;

pub struct FrameSchedule {
    last: Option<u64>,
    burst_until: u64,
}

impl FrameSchedule {
    pub fn new() -> Self {
        Self { last: None, burst_until: 0 }
    }

    pub fn burst(&mut self, now_ms: u64) {
        self.burst_until = now_ms + BURST_LEN_MS;
    }

    pub fn should_save(&mut self, now_ms: u64) -> bool {
        let every = if now_ms < self.burst_until { BURST_MS } else { BASE_MS };
        if self.last.is_some_and(|l| now_ms < l + every) {
            return false;
        }
        self.last = Some(now_ms);
        true
    }
}

impl Default for FrameSchedule {
    fn default() -> Self {
        Self::new()
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p selfplay frames`
Expected: PASS (2 tests).

- [ ] **Step 5: Wire it into the bot**

In `bins/bot/src/main.rs`:

1. Add to `Args`:

```rust
    /// Self-play observer mode: save frames (2 fps, 10 fps after new enemies) and
    /// observer.jsonl into this match directory.
    #[arg(long)]
    selfplay_dir: Option<PathBuf>,
```

2. After `let mut recorder = ...;` add:

```rust
    let mut schedule = selfplay::FrameSchedule::new();
    let mut observer_log = args
        .selfplay_dir
        .as_ref()
        .map(|d| selfplay::JsonlLog::<selfplay::ObserverRecord>::create(&d.join("observer.jsonl")))
        .transpose()?;
    let mut prev_enemies = 0usize;
    let obs = |kind: &str, card: Option<String>, tile: Option<(u32, u32)>, enemies: &[(u32, u32)]| selfplay::ObserverRecord {
        host_ms: selfplay::host_ms(),
        kind: kind.into(),
        card,
        tile,
        enemies: enemies.to_vec(),
    };
```

3. After `let state = tracker.state().clone();` add:

```rust
        if let Some(dir) = args.selfplay_dir.as_ref().filter(|_| state.in_battle) {
            let now = selfplay::host_ms();
            if state.enemies.len() > prev_enemies {
                schedule.burst(now);
                if let Some(log) = observer_log.as_mut() {
                    log.append(&obs("new_enemy", None, None, &state.enemies))?;
                }
            }
            prev_enemies = state.enemies.len();
            if schedule.should_save(now) {
                let path = dir.join("frames").join(format!("{now}.jpg"));
                std::fs::create_dir_all(path.parent().unwrap())?;
                let img = image::RgbImage::from_raw(frame.width, frame.height, frame.rgb.clone()).context("frame size")?;
                let mut out = std::io::BufWriter::new(std::fs::File::create(&path)?);
                image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 92).encode_image(&img)?;
            }
        }
```

4. In the `if state.in_battle != was_in_battle` block, append `battle_start` / `battle_end` records: `if let Some(log) = observer_log.as_mut() { log.append(&obs(if state.in_battle { "battle_start" } else { "battle_end" }, None, None, &state.enemies))?; }`.

5. After a successful deploy (next to `tracker.mark_played(slot);`): `if let Some(log) = observer_log.as_mut() { log.append(&obs("deploy", Some(card.clone()), Some((col, row)), &state.enemies))?; }`.

Add `features = ["png", "jpeg"]` to the bot's `image` dependency in `bins/bot/Cargo.toml`.

- [ ] **Step 6: Build and run the bot tests**

Run: `cargo build --release -p bot && cargo test --workspace`
Expected: build OK, all tests pass. Running `target/release/bot` without `--selfplay-dir` behaves exactly as before.

- [ ] **Step 7: Commit**

```bash
git add crates/selfplay bins/bot Cargo.lock
git commit -m "bot: --selfplay-dir observer recording (frames + observer.jsonl)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Clock alignment

**Files:**
- Create: `crates/selfplay/src/align.rs`
- Modify: `crates/selfplay/src/lib.rs` (add `pub mod align; pub use align::estimate_offset;`)
- Test: in `crates/selfplay/src/align.rs`

**Interfaces:**
- Consumes: `PlayRecord`, `ObserverRecord` (Task 3).
- Produces: `pub fn estimate_offset(plays: &[PlayRecord], observed: &[ObserverRecord]) -> Option<i64>`. For each verified play it takes the first `new_enemy` record within [0, 3000] ms after the play and returns the median of those deltas; `None` with fewer than 3 matches.

- [ ] **Step 1: Write the failing tests**

```rust
// crates/selfplay/src/align.rs
#[cfg(test)]
mod tests {
    use super::estimate_offset;
    use crate::{ObserverRecord, PlayRecord};

    fn play(ms: u64) -> PlayRecord {
        PlayRecord { host_ms: ms, card: "knight".into(), slot: 0, tile: (3, 17), hand_before: Default::default(), elixir_read: Some(5), verified: true }
    }
    fn seen(ms: u64) -> ObserverRecord {
        ObserverRecord { host_ms: ms, kind: "new_enemy".into(), card: None, tile: None, enemies: vec![] }
    }

    #[test]
    fn constant_delay() {
        let plays: Vec<_> = (0..5).map(|i| play(10_000 * i + 1_000)).collect();
        let obs: Vec<_> = (0..5).map(|i| seen(10_000 * i + 1_350)).collect();
        assert_eq!(estimate_offset(&plays, &obs), Some(350));
    }

    #[test]
    fn median_ignores_outliers() {
        let plays: Vec<_> = (0..5).map(|i| play(10_000 * i + 1_000)).collect();
        let mut obs: Vec<_> = (0..4).map(|i| seen(10_000 * i + 1_300)).collect();
        obs.push(seen(40_000 + 1_000 + 2_900)); // the 5th play was only noticed late
        assert_eq!(estimate_offset(&plays, &obs), Some(300));
    }

    #[test]
    fn too_few_matches() {
        assert_eq!(estimate_offset(&[play(1_000)], &[seen(1_300)]), None);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p selfplay align`
Expected: FAIL to compile.

- [ ] **Step 3: Write the implementation**

```rust
// crates/selfplay/src/align.rs
//! Delay between a sparring tap and the observer seeing it (network + scrcpy), per match.

use crate::{ObserverRecord, PlayRecord};

const MAX_DELAY_MS: u64 = 3_000;

pub fn estimate_offset(plays: &[PlayRecord], observed: &[ObserverRecord]) -> Option<i64> {
    let mut deltas: Vec<i64> = plays
        .iter()
        .filter(|p| p.verified)
        .filter_map(|p| {
            observed
                .iter()
                .filter(|o| o.kind == "new_enemy" && o.host_ms >= p.host_ms && o.host_ms <= p.host_ms + MAX_DELAY_MS)
                .map(|o| (o.host_ms - p.host_ms) as i64)
                .min()
        })
        .collect();
    if deltas.len() < 3 {
        return None;
    }
    deltas.sort_unstable();
    Some(deltas[deltas.len() / 2])
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p selfplay align`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/selfplay
git commit -m "selfplay: per-match clock offset from plays vs new enemies

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Friendly Battle orchestration (`selfplay` binary)

**Files:**
- Create: `crates/selfplay/src/screens.rs` (screen-state checks)
- Create: `selfplay_flows.toml` (tap points and check boxes per screen, measured from screenshots)
- Create: `fixtures/screens/{note14,note9}/*.png` (one screenshot per state)
- Create: `bins/selfplay/Cargo.toml`, `bins/selfplay/src/main.rs`
- Modify: `crates/selfplay/src/lib.rs` (add `pub mod screens;`)

**Interfaces:**
- Consumes: Tasks 1, 3, 6, 7; `target/release/bot`, `target/release/sparring`.
- Produces:
  - `#[derive(Deserialize)] pub struct ScreenCheck { pub name: String, pub x: f64, pub y: f64, pub w: f64, pub h: f64, pub rgb: [u8; 3], pub tol: u8, pub min_frac: f32 }` (a normalized box whose pixels must mostly be near a colour) and `pub fn matches(frame: &Frame, check: &ScreenCheck) -> bool`
  - `#[derive(Deserialize)] pub struct Flows { pub note14: PhoneFlows, pub note9: PhoneFlows }`, `#[derive(Deserialize)] pub struct PhoneFlows { pub screens: Vec<ScreenCheck>, pub taps: std::collections::HashMap<String, (f64, f64)> }`
  - `pub fn classify(frame: &Frame, screens: &[ScreenCheck]) -> Option<String>`: the name of the first screen whose check matches.
  - CLI: `selfplay --matches N --out dataset/selfplay`. Per match:
    1. Note 9: main screen → Friends → Khazar → Friendly Battle.
    2. Note 14: accept the invite.
    3. Start `bot --serial 4xwskfkr7xp7w4xo --matches 1 --selfplay-dir <dir>` and `sparring --out <dir>/plays.jsonl` as child processes.
    4. Wait for both to exit, run `estimate_offset`, write `meta.json`, then tap OK on both result screens.

- [ ] **Step 1: Capture the flow screenshots.** Do one Friendly Battle start by hand, taking a screenshot on both phones (Task 1 example; the Note 14 at width 576 too) at each state: `main`, `friends_list`, `friend_menu` (Khazar's row expanded), `invite_sent`, `invite_popup` (Note 14), `battle`, `result`. Save them as `fixtures/screens/<phone>/<state>.png`.

- [ ] **Step 2: Measure the checks and taps.** For each state pick a box that only that screen has (for example the yellow "Friendly Battle" button, the green accept button, the blue OK on the result screen) and its dominant colour, and write `selfplay_flows.toml`:

```toml
# Normalized coordinates (0..1) of 576-px-wide screenshots. One [[noteX.screens]] per state.
[[note9.screens]]
name = "friend_menu"
x = 0.0   # fill in from the screenshot: the yellow "Friendly Battle" button
y = 0.0
w = 0.0
h = 0.0
rgb = [0, 0, 0]
tol = 40
min_frac = 0.6

[note9.taps]
friends_tab = [0.0, 0.0]       # fill in: social / friends button on the main screen
khazar_row = [0.0, 0.0]
friendly_battle = [0.0, 0.0]
result_ok = [0.5, 0.847]

[note14.taps]
accept_invite = [0.0, 0.0]
result_ok = [0.5, 0.847]
```

Every `0.0` above must be replaced with the value measured from the fixture screenshots; none may remain.

- [ ] **Step 3: Write the failing tests**

```rust
// crates/selfplay/src/screens.rs
#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn flows() -> Flows {
        toml::from_str(&std::fs::read_to_string(root().join("selfplay_flows.toml")).unwrap()).unwrap()
    }

    #[test]
    fn every_fixture_is_recognized_as_itself() {
        let f = flows();
        for (phone, screens) in [("note9", &f.note9.screens), ("note14", &f.note14.screens)] {
            for s in screens.iter() {
                let frame = capture::load_rgb(root().join(format!("fixtures/screens/{phone}/{}.png", s.name))).unwrap();
                assert_eq!(classify(&frame, screens).as_deref(), Some(s.name.as_str()), "{phone}/{}", s.name);
            }
        }
    }

    #[test]
    fn flow_refuses_unknown_screen() {
        // A battle frame is none of the menu screens the flow taps on.
        let f = flows();
        let battle = capture::load_rgb(root().join("fixtures/note9/battle_1.png")).unwrap();
        let menus: Vec<_> = f.note9.screens.into_iter().filter(|s| s.name != "battle").collect();
        assert_eq!(classify(&battle, &menus), None);
    }

    #[test]
    fn no_unmeasured_taps() {
        let f = flows();
        for (k, v) in f.note9.taps.iter().chain(f.note14.taps.iter()) {
            assert!(*v != (0.0, 0.0), "tap {k} not measured");
        }
    }
}
```

Add `toml = "1.1.6"` and `capture = { path = "../capture" }` to `crates/selfplay/Cargo.toml` `[dependencies]`.

- [ ] **Step 4: Run tests to verify they fail**

Run: `cargo test -p selfplay screens`
Expected: FAIL to compile.

- [ ] **Step 5: Write `screens.rs`**

```rust
// crates/selfplay/src/screens.rs
//! Which screen a phone shows, from a few colour boxes measured once per screen, so the
//! orchestrator only taps when it knows where it is.

use std::collections::HashMap;

use capture::Frame;
use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
pub struct ScreenCheck {
    pub name: String,
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
    pub rgb: [u8; 3],
    pub tol: u8,
    pub min_frac: f32,
}

#[derive(Deserialize, Debug, Clone)]
pub struct PhoneFlows {
    pub screens: Vec<ScreenCheck>,
    pub taps: HashMap<String, (f64, f64)>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Flows {
    pub note14: PhoneFlows,
    pub note9: PhoneFlows,
}

pub fn matches(frame: &Frame, c: &ScreenCheck) -> bool {
    let (fw, fh) = (frame.width as f64, frame.height as f64);
    let (x0, y0) = ((c.x * fw) as u32, (c.y * fh) as u32);
    let (x1, y1) = (((c.x + c.w) * fw) as u32, ((c.y + c.h) * fh) as u32);
    let (mut hit, mut total) = (0u32, 0u32);
    for y in y0..y1.min(frame.height) {
        for x in x0..x1.min(frame.width) {
            let p = frame.pixel(x, y);
            total += 1;
            if p.iter().zip(c.rgb).all(|(a, b)| a.abs_diff(b) <= c.tol) {
                hit += 1;
            }
        }
    }
    total > 0 && hit as f32 / total as f32 >= c.min_frac
}

pub fn classify(frame: &Frame, screens: &[ScreenCheck]) -> Option<String> {
    screens.iter().find(|s| matches(frame, s)).map(|s| s.name.clone())
}
```

- [ ] **Step 6: Run tests**

Run: `cargo test -p selfplay screens`
Expected: PASS (3 tests). If a fixture is recognized as the wrong screen, move its check box to a more distinctive spot and rerun.

- [ ] **Step 7: Write the orchestrator**

```toml
# bins/selfplay/Cargo.toml
[package]
name = "selfplay-run"
edition.workspace = true
version.workspace = true
publish.workspace = true

[[bin]]
name = "selfplay"
path = "src/main.rs"

[dependencies]
anyhow = "1.0.104"
capture = { path = "../../crates/capture" }
clap = { version = "4.6.7", features = ["derive"] }
input = { path = "../../crates/input" }
selfplay = { path = "../../crates/selfplay" }
toml = "1.1.6"
tracing = "0.1.44"
tracing-subscriber = { version = "0.3.23", features = ["env-filter"] }
```

```rust
// bins/selfplay/src/main.rs
//! Runs Friendly Battles between the observer (Note 14, our bot recording) and the sparring
//! phone (Note 9), one match directory per battle. Never sends BACK; taps only on recognized
//! screens.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use clap::Parser;
use input::{AdbShell, TapBackend};
use selfplay::screens::{Flows, PhoneFlows, classify};
use selfplay::{MatchMeta, ObserverRecord, PlayRecord, estimate_offset, read_jsonl};

const OBSERVER: &str = "4xwskfkr7xp7w4xo";
const SPARRING: &str = "d8c5ce8a0406";

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 1)]
    matches: u32,
    #[arg(long, default_value = "dataset/selfplay")]
    out: PathBuf,
    #[arg(long, default_value = "selfplay_flows.toml")]
    flows: PathBuf,
}

fn adb() -> String {
    format!("{}/Android/Sdk/platform-tools/adb", std::env::var("HOME").unwrap_or_default())
}

/// Waits up to `timeout` for `serial` to show screen `want`.
fn wait_for(serial: &str, flows: &PhoneFlows, want: &str, timeout: Duration) -> anyhow::Result<()> {
    let t0 = Instant::now();
    loop {
        let f = capture::screencap(&adb(), serial, 576)?;
        let got = classify(&f, &flows.screens);
        if got.as_deref() == Some(want) {
            return Ok(());
        }
        if t0.elapsed() > timeout {
            let shot = format!("dataset/selfplay/lost_{serial}.png");
            if let Some(img) = image::RgbImage::from_raw(f.width, f.height, f.rgb) {
                img.save(&shot).ok();
            }
            bail!("{serial}: expected screen {want}, saw {got:?} for {timeout:?} (screenshot {shot})");
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

/// Taps the named point on `serial`, but only when the phone shows screen `on`.
fn tap_on(shell: &mut AdbShell, serial: &str, flows: &PhoneFlows, on: &str, tap: &str) -> anyhow::Result<()> {
    wait_for(serial, flows, on, Duration::from_secs(15))?;
    let (x, y) = *flows.taps.get(tap).with_context(|| format!("no tap {tap} in flows"))?;
    let (w, h) = shell.screen_size();
    shell.taps(&[((x * w as f64) as u32, (y * h as f64) as u32)])
}

fn run_match(dir: &Path, flows: &Flows, n9: &mut AdbShell, n14: &mut AdbShell) -> anyhow::Result<()> {
    tap_on(n9, SPARRING, &flows.note9, "main", "friends_tab")?;
    tap_on(n9, SPARRING, &flows.note9, "friends_list", "khazar_row")?;
    tap_on(n9, SPARRING, &flows.note9, "friend_menu", "friendly_battle")?;
    tap_on(n14, OBSERVER, &flows.note14, "invite_popup", "accept_invite")?;
    let dir_s = dir.to_string_lossy().to_string();
    let mut bot = Command::new("target/release/bot")
        .args(["--serial", OBSERVER, "--matches", "1", "--selfplay-dir", &dir_s])
        .stdout(std::fs::File::create(dir.join("bot.log"))?)
        .stderr(std::fs::File::create(dir.join("bot.err"))?)
        .spawn()?;
    let mut spar = Command::new("target/release/sparring")
        .args(["--serial", SPARRING, "--out", &format!("{dir_s}/plays.jsonl")])
        .stdout(std::fs::File::create(dir.join("sparring.log"))?)
        .stderr(std::fs::File::create(dir.join("sparring.err"))?)
        .spawn()?;
    let spar_status = spar.wait()?;
    let bot_status = bot.wait()?;
    if !spar_status.success() || !bot_status.success() {
        bail!("sparring {spar_status}, bot {bot_status}");
    }
    tap_on(n9, SPARRING, &flows.note9, "result", "result_ok")?;
    tap_on(n14, OBSERVER, &flows.note14, "result", "result_ok")?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let a = Args::parse();
    let flows: Flows = toml::from_str(&std::fs::read_to_string(&a.flows)?)?;
    let mut n9 = AdbShell::open(&adb(), Some(SPARRING))?;
    let mut n14 = AdbShell::open(&adb(), Some(OBSERVER))?;
    for i in 0..a.matches {
        let id = format!("{}_{i:03}", selfplay::host_ms());
        let dir = a.out.join(&id);
        std::fs::create_dir_all(&dir)?;
        let mut meta = MatchMeta { match_id: id.clone(), observer_serial: OBSERVER.into(), sparring_serial: SPARRING.into(), ..Default::default() };
        match run_match(&dir, &flows, &mut n9, &mut n14) {
            Ok(()) => {
                let plays: Vec<PlayRecord> = read_jsonl(&dir.join("plays.jsonl")).unwrap_or_default();
                let obs: Vec<ObserverRecord> = read_jsonl(&dir.join("observer.jsonl")).unwrap_or_default();
                meta.clock_offset_ms = estimate_offset(&plays, &obs);
                meta.complete = true;
                tracing::info!("match {id}: {} plays, offset {:?} ms", plays.len(), meta.clock_offset_ms);
            }
            Err(e) => {
                meta.error = Some(format!("{e:#}"));
                meta.save(&dir)?;
                bail!("match {id} failed: {e:#}");
            }
        }
        meta.save(&dir)?;
    }
    Ok(())
}
```

Add `image = { version = "0.25.10", default-features = false, features = ["png"] }` to `bins/selfplay/Cargo.toml` `[dependencies]` (used by `wait_for` to save the screenshot of an unexpected screen).

- [ ] **Step 8: Add the incomplete-match test**

```rust
// crates/selfplay/src/record.rs, in mod tests
    #[test]
    fn incomplete_match_marked() {
        let dir = tempfile::tempdir().unwrap();
        MatchMeta { match_id: "x".into(), error: Some("sparring exit status: 1".into()), ..Default::default() }.save(dir.path()).unwrap();
        let m: MatchMeta = serde_json::from_str(&std::fs::read_to_string(dir.path().join("meta.json")).unwrap()).unwrap();
        assert!(!m.complete && m.error.is_some());
    }
```

Run: `cargo test -p selfplay`
Expected: PASS.

- [ ] **Step 9: One end-to-end self-play match**

Both phones on the main screen, scrcpy running for the Note 14 (`scrcpy -s 4xwskfkr7xp7w4xo --v4l2-sink=/dev/video10 --no-window --no-audio --max-fps=60 --max-size=1280`). In the background: `cargo build --release && target/release/selfplay --matches 1 > dataset/selfplay/run.log 2>&1`. Expected:
- exit 0
- one match directory containing `frames/` (300+ JPEGs), `plays.jsonl` (20+ plays), `observer.jsonl`, and a `meta.json` with `"complete": true` and `clock_offset_ms` between 100 and 1500.

- [ ] **Step 10: Commit**

```bash
git add crates/selfplay bins/selfplay selfplay_flows.toml fixtures/screens Cargo.lock
git commit -m "selfplay: Friendly Battle orchestration with screen checks, one match end to end

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Hand templates for any card (from harvested portraits)

**Files:**
- Modify: `crates/vision/src/hand.rs` (add `CardLibrary::from_images`)
- Create: `training/portrait_templates.py` (cuts the hand-slot-like art from each `dataset/cards/*/info.png` into `assets/cards_all/<slug>.png`)
- Test: `crates/vision/tests/fixtures.rs`

**Interfaces:**
- Consumes: `assets/cards.json` (`slug`, `dir`), `dataset/cards/<dir>/info.png`.
- Produces: `assets/cards_all/<slug>.png` for all 122 cards (art only, sized like the existing hand templates), and `pub fn load_subset(dir: impl AsRef<Path>, names: &[&str]) -> anyhow::Result<CardLibrary>` (loads only the given card names plus `_empty`). `sparring` gets `--deck a,b,c,...` and uses `load_subset(assets/cards_all, deck)`.

- [ ] **Step 1: Cut the portraits.** Write `training/portrait_templates.py`:

```python
"""Hand-slot templates for every card, from the harvested Info portraits.

dataset/cards/<dir>/info.png (+ assets/cards.json for slug/dir) -> assets/cards_all/<slug>.png
The Info popup shows the card with the same art as the hand slot; we cut the same art
window vision::hand uses (ART fractions of the card rect) and scale it to the size of the
existing assets/cards templates.
"""
import json
from pathlib import Path

import cv2

PORTRAIT = {"card": (75, 860, 325, 1165), "champion": (75, 400, 325, 705)}  # x0, y0, x1, y1 on 1080x2400
ART = (0.06, 0.05, 0.94, 0.72)
ref = cv2.imread("../assets/cards/hog_rider.png")
out = Path("../assets/cards_all")
out.mkdir(exist_ok=True)
for c in json.load(open("../assets/cards.json")):
    d = Path("../dataset/cards") / c["dir"]
    layout = (d / "layout.txt").read_text().strip() if (d / "layout.txt").exists() else "card"
    if layout not in PORTRAIT:
        continue  # tower troops are not in the hand
    x0, y0, x1, y1 = PORTRAIT[layout]
    card = cv2.imread(str(d / "info.png"))[y0:y1, x0:x1]
    h, w = card.shape[:2]
    art = card[int(ART[1] * h):int(ART[3] * h), int(ART[0] * w):int(ART[2] * w)]
    cv2.imwrite(str(out / f"{c['slug']}.png"), cv2.resize(art, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA))
print(len(list(out.glob("*.png"))), "templates ->", out)
```

Before running it, check `PORTRAIT` against one `info.png` and one champion `info.png` by viewing the crop, and adjust the box until it covers exactly the card (frame included). Run: `cd training && .venv/bin/python portrait_templates.py`. Expected: `122 templates -> ../assets/cards_all`. Copy `assets/cards/_empty.png` into `assets/cards_all/`.

- [ ] **Step 2: Write the failing test**

```rust
// crates/vision/tests/fixtures.rs
#[test]
fn portrait_templates_read_note9_hands() {
    let calib = calib::Calibration::load(root().join("calibration_note9.toml")).unwrap();
    let deck = ["hog_rider", "musketeer", "cannon", "ice_golem", "skeletons", "ice_spirit", "the_log", "fireball"];
    let lib = vision::CardLibrary::load_subset(root().join("assets/cards_all"), &deck).unwrap();
    let reference = vision::CardLibrary::load(root().join("assets/cards")).unwrap();
    for i in 1..=5 {
        let f = capture::load_rgb(root().join(format!("fixtures/note9/battle_{i}.png"))).unwrap();
        let want = reference.read_hand(&f, &calib);
        let got = lib.read_hand(&f, &calib);
        for s in 0..4 {
            if let Some(w) = want.slots[s].name() {
                assert_eq!(got.slots[s].name(), Some(w), "battle_{i} slot {s}");
            }
        }
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p vision --test fixtures portrait`
Expected: FAIL to compile (`load_subset` not found).

- [ ] **Step 4: Implement `load_subset`**

In `crates/vision/src/hand.rs`, refactor `load` into a helper that takes a name filter:

```rust
impl CardLibrary {
    pub fn load(dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        Self::load_filtered(dir.as_ref(), |_| true)
    }

    /// Only the given cards (plus the empty-slot template): the sparring deck's 8 cards.
    pub fn load_subset(dir: impl AsRef<Path>, names: &[&str]) -> anyhow::Result<Self> {
        let lib = Self::load_filtered(dir.as_ref(), |n| n == EMPTY_TEMPLATE || names.contains(&n))?;
        let missing: Vec<_> = names.iter().filter(|n| !lib.templates.iter().any(|t| t.name == **n)).collect();
        anyhow::ensure!(missing.is_empty(), "no templates for {missing:?} in {}", dir.as_ref().display());
        Ok(lib)
    }

    fn load_filtered(dir: &Path, keep: impl Fn(&str) -> bool) -> anyhow::Result<Self> {
        let mut templates = Vec::new();
        for entry in std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("png") {
                continue;
            }
            let stem = path.file_stem().and_then(|s| s.to_str()).context("bad file name")?;
            let name = stem.split('@').next().unwrap_or(stem).to_string();
            if !keep(&name) {
                continue;
            }
            let frame = capture::load_rgb(&path)?;
            let patch = Patch::from_frame(&frame).with_context(|| format!("empty template {}", path.display()))?;
            templates.push(Template { name, patch });
        }
        templates.sort_by(|a, b| a.name.cmp(&b.name));
        anyhow::ensure!(!templates.is_empty(), "no card templates in {}", dir.display());
        Ok(Self { templates })
    }
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p vision`
Expected: PASS. If `portrait_templates_read_note9_hands` fails on some slots, the portrait art differs from the hand art (frame, scale). Fix the crop in `portrait_templates.py` (tighter `ART` window) rather than lowering `MIN_SCORE`. Since only 8 candidates compete, the best match decides, so the slot read needs to be correct, not just high-scoring.

- [ ] **Step 6: Use it in `sparring`**

Add to `sparring`'s `Args`: `#[arg(long, value_delimiter = ',')] deck: Vec<String>,`. When it is non-empty, load `vision::CardLibrary::load_subset("assets/cards_all", &deck)` instead of `--cards`. Run `cargo test --workspace`; expected PASS.

- [ ] **Step 7: Commit**

```bash
git add crates/vision training/portrait_templates.py assets/cards_all bins/sparring
git commit -m "Hand templates for all cards from harvested portraits; sparring --deck

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Deck rotation (planner + deck editing on the Note 9)

**Files:**
- Create: `crates/selfplay/src/decks.rs`
- Modify: `crates/selfplay/src/lib.rs` (add `pub mod decks; pub use decks::plan_decks;`)
- Modify: `selfplay_flows.toml` (deck-edit screens and taps), `fixtures/screens/note9/` (deck-edit screenshots)
- Modify: `bins/selfplay/src/main.rs` (`--rotate` flag)

**Interfaces:**
- Consumes: `brain::cards()` (`slug`, `rarity`, `kind`).
- Produces: `pub fn plan_decks(seed: u64, rounds: usize) -> Vec<Vec<String>>`. Each round covers every playable card (not Tower Troop) exactly once in decks of 8, with at most one Champion per deck. Leftover slots in the last deck of a round are filled with random cards from that round's other decks. Orchestrator flag `--rotate <seed>`: before match `i`, set the Note 9 deck to `plan_decks(seed, rounds)[i % len]`, record it in `meta.json` (`sparring_deck`) and pass `--deck` to `sparring`.

- [ ] **Step 1: Write the failing tests**

```rust
// crates/selfplay/src/decks.rs
#[cfg(test)]
mod tests {
    use super::plan_decks;
    use std::collections::HashSet;

    fn champion(slug: &str) -> bool {
        brain::cards().get(slug).and_then(|c| c.rarity.clone()).as_deref() == Some("Champion")
    }

    #[test]
    fn one_round_covers_every_card_in_valid_decks() {
        let decks = plan_decks(1, 1);
        let playable: HashSet<String> = brain::cards()
            .iter()
            .filter(|c| c.kind.as_deref() != Some("Tower Troop"))
            .map(|c| c.slug.clone())
            .collect();
        assert_eq!(playable.len(), 122);
        let seen: HashSet<String> = decks.iter().flatten().cloned().collect();
        assert_eq!(seen, playable);
        assert_eq!(decks.len(), 16); // ceil(122 / 8)
        for d in &decks {
            assert_eq!(d.len(), 8);
            assert_eq!(d.iter().collect::<HashSet<_>>().len(), 8, "distinct: {d:?}");
            assert!(d.iter().filter(|c| champion(c)).count() <= 1, "{d:?}");
        }
    }

    #[test]
    fn rounds_differ_and_are_reproducible() {
        assert_eq!(plan_decks(5, 2), plan_decks(5, 2));
        let d = plan_decks(5, 2);
        assert_eq!(d.len(), 32);
        assert_ne!(d[..16], d[16..]);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p selfplay decks`
Expected: FAIL to compile.

- [ ] **Step 3: Write the implementation**

```rust
// crates/selfplay/src/decks.rs
//! Sparring decks: every round shows every card once; champions spread one per deck.

pub fn plan_decks(seed: u64, rounds: usize) -> Vec<Vec<String>> {
    let cards = brain::cards();
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut out = Vec::new();
    for _ in 0..rounds {
        let mut champs: Vec<String> = Vec::new();
        let mut rest: Vec<String> = Vec::new();
        for c in cards.iter().filter(|c| c.kind.as_deref() != Some("Tower Troop")) {
            if c.rarity.as_deref() == Some("Champion") { champs.push(c.slug.clone()) } else { rest.push(c.slug.clone()) }
        }
        champs.sort();
        rest.sort();
        rng.shuffle(&mut champs);
        rng.shuffle(&mut rest);
        let n = (champs.len() + rest.len()).div_ceil(8);
        assert!(champs.len() <= n, "more champions than decks");
        let mut decks: Vec<Vec<String>> = (0..n).map(|i| champs.get(i).cloned().into_iter().collect()).collect();
        for c in rest {
            let d = decks.iter_mut().filter(|d| d.len() < 8).min_by_key(|d| d.len()).unwrap();
            d.push(c);
        }
        // Fill short decks with non-champions from other decks of this round.
        let pool: Vec<String> = decks.iter().flatten().filter(|c| cards.get(c).and_then(|x| x.rarity.as_deref()) != Some("Champion")).cloned().collect();
        for d in decks.iter_mut() {
            while d.len() < 8 {
                let c = &pool[rng.usize(..pool.len())];
                if !d.contains(c) {
                    d.push(c.clone());
                }
            }
        }
        out.extend(decks);
    }
    out
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p selfplay decks`
Expected: PASS (2 tests).

- [ ] **Step 5: Measure the deck-edit flow.** On the Note 9, by hand, capture screenshots for:
  - `deck_view`: Decks tab, deck 2 shown.
  - `card_menu`: a card tapped in the Collection below, with Info and Use showing.
  - `swap_target`: after Use, waiting for the deck slot to replace.

Add those screens and their taps to `selfplay_flows.toml`: `cards_tab`, `decks_tab`, `deck_slot_1`...`deck_slot_8` and `use_button`. Also add a `collection_search` route: Collection sorted By Elixir, scrolling with slow drags as in `training/harvest_cards.py`, and finding a card by matching its grid art against `dataset/cards/<dir>/grid.png` with `signature()` (port the function into Rust in `screens.rs` as `pub fn grid_signature(frame: &Frame, x: u32, y: u32) -> Vec<f32>`, using the same 24x24 lower-art crop).

- [ ] **Step 6: Write `set_deck`** in `bins/selfplay/src/main.rs`:
  - For each of the 8 deck positions whose current card differs from the target, find the target card in the Collection (scroll and match `grid_signature` against its `grid.png`, threshold 0.9).
  - Tap it, then tap Use, then tap the deck slot.
  - Verify each step with `classify` and stop on any unknown screen.
  - Read the deck back by matching the 8 deck-slot arts against `assets/cards_all`.
  - Fail unless all 8 match the target.
  - Only ever call it on the `SPARRING` serial: put `assert_eq!(serial, SPARRING)` at the top.

- [ ] **Step 7: Run one rotated match.** In the background: `target/release/selfplay --matches 1 --rotate 1 > dataset/selfplay/run.log 2>&1`. Expected:
  - the Note 9 deck becomes `plan_decks(1, 1)[0]`
  - `meta.json` lists it
  - `plays.jsonl` only contains cards from it, and 90%+ of them are verified

- [ ] **Step 8: Commit**

```bash
git add crates/selfplay bins/selfplay selfplay_flows.toml fixtures/screens
git commit -m "selfplay: deck rotation planner and deck editing on the sparring phone

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Unattended runs and dataset report

**Files:**
- Create: `training/selfplay_report.py`
- Modify: `bins/selfplay/src/main.rs` (continue after a failed match: recover both phones to the main screen up to 3 times, else stop)

**Interfaces:**
- Consumes: `dataset/selfplay/*/{meta.json,plays.jsonl,observer.jsonl,frames/}`.
- Produces: a report printed to stdout and written to `dataset/selfplay/report.md`:
  - matches complete/total, plays per card (min, median, the cards under 50), verified rate
  - clock offset median and spread
  - frames per match

- [ ] **Step 1: Write the report script**

```python
"""Summarize the self-play dataset: coverage per card, label quality, timing."""
import collections
import json
import statistics
from pathlib import Path

root = Path("../dataset/selfplay")
cards = {c["slug"] for c in json.load(open("../assets/cards.json")) if c["type"] != "Tower Troop"}
plays, offsets, frames, complete, total, verified = collections.Counter(), [], [], 0, 0, [0, 0]
for m in sorted(p for p in root.iterdir() if (p / "meta.json").exists()):
    meta = json.load(open(m / "meta.json"))
    total += 1
    if not meta.get("complete"):
        continue
    complete += 1
    if meta.get("clock_offset_ms") is not None:
        offsets.append(meta["clock_offset_ms"])
    frames.append(len(list((m / "frames").glob("*.jpg"))))
    for line in open(m / "plays.jsonl"):
        p = json.loads(line)
        verified[0] += p["verified"]
        verified[1] += 1
        if p["verified"]:
            plays[p["card"]] += 1
low = sorted((plays[c], c) for c in cards if plays[c] < 50)
lines = [
    f"matches: {complete}/{total} complete",
    f"plays: {verified[1]} ({verified[0] / max(1, verified[1]):.1%} verified)",
    f"plays per card: min {min((plays[c] for c in cards), default=0)}, median {statistics.median([plays[c] for c in cards])}",
    f"cards under 50 plays ({len(low)}): " + ", ".join(f"{c} {n}" for n, c in low),
    f"clock offset ms: median {statistics.median(offsets) if offsets else None}, "
    f"range {min(offsets, default=None)}..{max(offsets, default=None)}",
    f"frames per match: median {statistics.median(frames) if frames else 0}",
]
text = "\n".join(lines)
print(text)
(root / "report.md").write_text("# Self-play dataset\n\n" + "\n".join(f"- {l}" for l in lines) + "\n")
```

- [ ] **Step 2: Recovery between matches.** In `main`, on a match error, take screenshots of both phones (saved as `lost_*.png`). Then, on each phone, tap only through recognized screens toward `main`: the result screen's `result_ok`, or the Battle tab when the screen is `cards_tab`/`deck_view`. Retry the next match. After 3 consecutive failures, stop with exit code 1.

- [ ] **Step 3: Short unattended run.** In the background: `target/release/selfplay --matches 5 --rotate 1 > dataset/selfplay/run.log 2>&1`, then `cd training && .venv/bin/python selfplay_report.py`. Expected:
  - 5/5 matches complete
  - verified at least 90%
  - clock offset spread under 500 ms
  - 5 different sparring decks in the `meta.json` files

- [ ] **Step 4: Commit**

```bash
git add training/selfplay_report.py bins/selfplay
git commit -m "selfplay: recovery between matches and dataset report

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Full collection (hand-off).** Start `target/release/selfplay --matches 170 --rotate 1` in the background (about 9 hours) and check `selfplay_report.py` periodically. Done when every card has 50+ verified plays. That data is the input for the next plan (event detection, classifier, deck inference).
