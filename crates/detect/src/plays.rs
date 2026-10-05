//! Enemy card plays, live (spec section 2-3): proposals from new enemy level tags and motion
//! bursts, an 8-frame clip around each, classified by `plays.onnx` (122 cards + no_play).
//! Mirrors the training pipeline in `training/selfplay/` (crops, timing, proposal rules).

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use calib::Calibration;
use capture::Frame;
use image::{GrayImage, RgbImage};
use ort::session::Session;
use ort::value::Tensor;

/// Frames per clip, this far apart, starting at the proposal.
pub const CLIP_FRAMES: usize = 8;
pub const FRAME_STEP: Duration = Duration::from_millis(150);
/// Clip crop side (pixels, model input) and size in tiles around the proposal tile.
pub const CROP: u32 = 128;
const CROP_TILES: f64 = 6.0;
/// Motion grid: one cell per arena tile.
pub const GRID_W: usize = 18;
pub const GRID_H: usize = 32;
const MOTION_SIZE: (u32, u32) = (144, 256);
const MOTION_EVERY: Duration = Duration::from_millis(100);
const MOTION_BACK: Duration = Duration::from_millis(180);
const MOTION_DIFF: u8 = 25;
const MOTION_ON: f32 = 0.25;
const MOTION_QUIET: f32 = 0.2;
const MOTION_HISTORY: Duration = Duration::from_millis(300);
const TRACK_TILES: u32 = 2;
const TRACK_TTL: Duration = Duration::from_millis(400);
const MERGE_WINDOW: Duration = Duration::from_millis(600);
const MERGE_TILES: u32 = 3;
const BUFFER_EVERY: Duration = Duration::from_millis(45);
const BUFFER_KEEP: Duration = Duration::from_millis(2600);
/// Own deploys: proposals this close in space and time are our own cards.
const OWN_TILES: u32 = 3;
const OWN_BEFORE: Duration = Duration::from_millis(500);
const OWN_AFTER: Duration = Duration::from_millis(3000);

fn cheb(a: (u32, u32), b: (u32, u32)) -> u32 {
    a.0.abs_diff(b.0).max(a.1.abs_diff(b.1))
}

/// Pixel arena box (x0, y0, x1, y1) of a frame.
fn arena_px(calib: &Calibration, w: u32, h: u32) -> (f64, f64, f64, f64) {
    let a = &calib.arena;
    (a.top_left.x * w as f64, a.top_left.y * h as f64, a.bottom_right.x * w as f64, a.bottom_right.y * h as f64)
}

/// The clip crop for one tile: 6x6 tiles centered on it (zero padded off-screen), 128x128.
pub fn crop_tile(frame: &Frame, calib: &Calibration, col: u32, row: u32) -> RgbImage {
    let (x0, y0, x1, y1) = arena_px(calib, frame.width, frame.height);
    let tile_px = (x1 - x0) / 18.0;
    let (cx, cy) = (x0 + (col as f64 + 0.5) / 18.0 * (x1 - x0), y0 + (row as f64 + 0.5) / 32.0 * (y1 - y0));
    let half = (CROP_TILES * tile_px / 2.0) as i64;
    let (ox, oy) = (cx as i64 - half, cy as i64 - half);
    let side = (2 * half) as u32;
    let mut canvas = RgbImage::new(side, side);
    for y in 0..side as i64 {
        let sy = oy + y;
        if sy < 0 || sy >= frame.height as i64 {
            continue;
        }
        for x in 0..side as i64 {
            let sx = ox + x;
            if sx < 0 || sx >= frame.width as i64 {
                continue;
            }
            canvas.put_pixel(x as u32, y as u32, image::Rgb(frame.pixel(sx as u32, sy as u32)));
        }
    }
    image::imageops::resize(&canvas, CROP, CROP, image::imageops::FilterType::Triangle)
}

pub struct PlayClassifier {
    session: Session,
    pub classes: Vec<String>,
}

impl PlayClassifier {
    /// Loads `plays.onnx` and its class list `plays.txt` (one per line, logit order).
    pub fn load(model: impl AsRef<Path>, classes: impl AsRef<Path>) -> anyhow::Result<Self> {
        let classes = std::fs::read_to_string(classes.as_ref())
            .with_context(|| format!("read {}", classes.as_ref().display()))?
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(String::from)
            .collect();
        let session =
            Session::builder()?.commit_from_file(model.as_ref()).with_context(|| format!("load {}", model.as_ref().display()))?;
        Ok(Self { session, classes })
    }

    /// Class probabilities for one clip (any number of frames; 8 for the full readout).
    pub fn probs(&mut self, clip: &[RgbImage]) -> anyhow::Result<Vec<f32>> {
        let plane = (CROP * CROP) as usize;
        let mut input = vec![0f32; clip.len() * 3 * plane];
        for (t, img) in clip.iter().enumerate() {
            for (i, p) in img.pixels().enumerate() {
                for ch in 0..3 {
                    input[(t * 3 + ch) * plane + i] = p[ch] as f32 / 255.0;
                }
            }
        }
        let tensor = Tensor::from_array(([1usize, clip.len(), 3, CROP as usize, CROP as usize], input))?;
        let outputs = self.session.run(ort::inputs![tensor])?;
        let (_, logits) = outputs[0].try_extract_tensor::<f32>()?;
        let m = logits.iter().copied().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
        let sum: f32 = exps.iter().sum();
        Ok(exps.into_iter().map(|e| e / sum).collect())
    }
}

/// New enemy level tags that no tracked tag explains (troops and buildings).
#[derive(Default)]
pub struct TagProposer {
    tracks: Vec<((u32, u32), Instant)>,
}

impl TagProposer {
    pub fn update(&mut self, at: Instant, enemies: &[(u32, u32)]) -> Vec<(u32, u32)> {
        self.tracks.retain(|(_, seen)| at.saturating_duration_since(*seen) <= TRACK_TTL);
        let mut new = Vec::new();
        for &e in enemies {
            if let Some(tr) = self.tracks.iter_mut().filter(|(t, _)| cheb(*t, e) <= TRACK_TILES).min_by_key(|(t, _)| cheb(*t, e)) {
                *tr = (e, at);
            } else {
                self.tracks.push((e, at));
                new.push(e);
            }
        }
        new
    }
}

fn blur3(g: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; g.len()];
    for r in 0..GRID_H {
        for c in 0..GRID_W {
            let mut s = 0.0;
            for dr in -1i64..=1 {
                for dc in -1i64..=1 {
                    // Mirror at the border (like cv2.blur's default).
                    let rr = (r as i64 + dr).clamp(0, GRID_H as i64 - 1) as usize;
                    let cc = (c as i64 + dc).clamp(0, GRID_W as i64 - 1) as usize;
                    s += g[rr * GRID_W + cc];
                }
            }
            out[r * GRID_W + c] = s / 9.0;
        }
    }
    out
}

/// Tiles where a 3x3 block turned busy after being quiet (`prev_max`: the busiest each cell
/// was over the last 300 ms), one centroid per connected area.
pub fn burst_tiles(grid: &[f32], prev_max: &[f32]) -> Vec<(u32, u32)> {
    let (blk, prev) = (blur3(grid), blur3(prev_max));
    let hot: Vec<bool> = blk.iter().zip(&prev).map(|(b, p)| *b >= MOTION_ON && *p < MOTION_QUIET).collect();
    let mut seen = vec![false; hot.len()];
    let mut out = Vec::new();
    for start in 0..hot.len() {
        if !hot[start] || seen[start] {
            continue;
        }
        let (mut stack, mut cells) = (vec![start], Vec::new());
        seen[start] = true;
        while let Some(i) = stack.pop() {
            cells.push(i);
            let (r, c) = ((i / GRID_W) as i64, (i % GRID_W) as i64);
            for dr in -1..=1 {
                for dc in -1..=1 {
                    let (rr, cc) = (r + dr, c + dc);
                    if rr < 0 || cc < 0 || rr >= GRID_H as i64 || cc >= GRID_W as i64 {
                        continue;
                    }
                    let j = rr as usize * GRID_W + cc as usize;
                    if hot[j] && !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        let n = cells.len() as f32;
        let col = cells.iter().map(|i| (i % GRID_W) as f32).sum::<f32>() / n;
        let row = cells.iter().map(|i| (i / GRID_W) as f32).sum::<f32>() / n;
        out.push((col.round() as u32, row.round() as u32));
    }
    out
}

/// Arena grayscale at 144x256 for motion.
fn motion_gray(frame: &Frame, calib: &Calibration) -> GrayImage {
    let (x0, y0, x1, y1) = arena_px(calib, frame.width, frame.height);
    let (x0, y0) = (x0.max(0.0) as u32, y0.max(0.0) as u32);
    let (w, h) = ((x1 as u32).min(frame.width) - x0, (y1 as u32).min(frame.height) - y0);
    let gray = GrayImage::from_fn(w, h, |x, y| {
        let [r, g, b] = frame.pixel(x0 + x, y0 + y);
        image::Luma([(0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) as u8])
    });
    image::imageops::resize(&gray, MOTION_SIZE.0, MOTION_SIZE.1, image::imageops::FilterType::Triangle)
}

/// Changed-pixel fraction per tile between two motion images.
fn motion_grid(a: &GrayImage, b: &GrayImage) -> Vec<f32> {
    let (cw, ch) = (MOTION_SIZE.0 as usize / GRID_W, MOTION_SIZE.1 as usize / GRID_H);
    let mut g = vec![0.0; GRID_W * GRID_H];
    for (x, y, p) in a.enumerate_pixels() {
        if p[0].abs_diff(b.get_pixel(x, y)[0]) > MOTION_DIFF {
            g[(y as usize / ch).min(GRID_H - 1) * GRID_W + (x as usize / cw).min(GRID_W - 1)] += 1.0;
        }
    }
    let cell = (cw * ch) as f32;
    g.iter_mut().for_each(|v| *v /= cell);
    g
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Proposal {
    pub at: Instant,
    pub tile: (u32, u32),
}

/// Buffers frames, turns enemy tags and motion into proposals, and hands out complete clips.
#[derive(Default)]
pub struct PlayDetector {
    buffer: VecDeque<Arc<Frame>>,
    grays: VecDeque<(Instant, GrayImage)>,
    grids: VecDeque<(Instant, Vec<f32>)>,
    tags: TagProposer,
    pending: Vec<Proposal>,
    recent: VecDeque<Proposal>,
    own: VecDeque<(Instant, (u32, u32))>,
}

impl PlayDetector {
    /// Feeds one frame (any rate) and the enemy tag tiles detected on it.
    pub fn push(&mut self, frame: &Frame, enemies: &[(u32, u32)], calib: &Calibration) {
        let at = frame.captured_at;
        if self.buffer.back().is_none_or(|f| at.saturating_duration_since(f.captured_at) >= BUFFER_EVERY) {
            self.buffer.push_back(Arc::new(frame.clone()));
            while self.buffer.front().is_some_and(|f| at.saturating_duration_since(f.captured_at) > BUFFER_KEEP) {
                self.buffer.pop_front();
            }
        }
        for tile in self.tags.update(at, enemies) {
            self.propose(at, tile);
        }
        if self.grays.back().is_none_or(|(t, _)| at.saturating_duration_since(*t) >= MOTION_EVERY) {
            let g = motion_gray(frame, calib);
            if let Some((_, back)) = self.grays.iter().rev().find(|(t, _)| at.saturating_duration_since(*t) >= MOTION_BACK) {
                let grid = motion_grid(&g, back);
                let mut prev_max = vec![0.0f32; grid.len()];
                for (_, old) in self.grids.iter().filter(|(t, _)| at.saturating_duration_since(*t) <= MOTION_HISTORY) {
                    prev_max.iter_mut().zip(old).for_each(|(m, o)| *m = m.max(*o));
                }
                for tile in burst_tiles(&grid, &prev_max) {
                    self.propose(at, tile);
                }
                self.grids.push_back((at, grid));
                while self.grids.front().is_some_and(|(t, _)| at.saturating_duration_since(*t) > MOTION_HISTORY) {
                    self.grids.pop_front();
                }
            }
            self.grays.push_back((at, g));
            while self.grays.len() > 4 {
                self.grays.pop_front();
            }
        }
    }

    /// One of our own cards landed at `tile`: proposals there are not enemy plays.
    pub fn own_deploy(&mut self, at: Instant, tile: (u32, u32)) {
        self.own.push_back((at, tile));
        while self.own.len() > 16 {
            self.own.pop_front();
        }
    }

    /// Adds a proposal unless it repeats a recent one (600 ms, 3 tiles).
    pub fn propose(&mut self, at: Instant, tile: (u32, u32)) {
        self.recent.retain(|p| at.saturating_duration_since(p.at) <= MERGE_WINDOW);
        if self.recent.iter().any(|p| cheb(p.tile, tile) <= MERGE_TILES) {
            return;
        }
        let p = Proposal { at, tile };
        self.recent.push_back(p);
        self.pending.push(p);
    }

    fn is_own(&self, p: &Proposal) -> bool {
        self.own.iter().any(|&(t, tile)| {
            cheb(tile, p.tile) <= OWN_TILES
                && p.at + OWN_BEFORE >= t
                && p.at.saturating_duration_since(t) <= OWN_AFTER
        })
    }

    /// Proposals whose 8 frames are all buffered by `now`, with their clips (own plays dropped).
    pub fn ready(&mut self, now: Instant, calib: &Calibration) -> Vec<(Proposal, Vec<RgbImage>)> {
        let span = FRAME_STEP * (CLIP_FRAMES as u32 - 1);
        let (done, wait): (Vec<Proposal>, Vec<Proposal>) = self.pending.drain(..).partition(|p| now >= p.at + span + BUFFER_EVERY);
        self.pending = wait;
        done.into_iter()
            .filter(|p| !self.is_own(p))
            .filter_map(|p| {
                let clip: Option<Vec<RgbImage>> = (0..CLIP_FRAMES as u32)
                    .map(|k| {
                        let target = p.at + FRAME_STEP * k;
                        let f = self.buffer.iter().min_by_key(|f| {
                            if f.captured_at > target { f.captured_at - target } else { target - f.captured_at }
                        })?;
                        Some(crop_tile(f, calib, p.tile.0, p.tile.1))
                    })
                    .collect();
                clip.map(|c| (p, c))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    fn root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn calib14() -> Calibration {
        Calibration::load(root().join("calibration.toml")).unwrap()
    }

    fn mean_abs_diff(a: &image::RgbImage, b: &image::RgbImage) -> f64 {
        a.as_raw().iter().zip(b.as_raw()).map(|(x, y)| (*x as f64 - *y as f64).abs()).sum::<f64>() / a.as_raw().len() as f64
    }

    #[test]
    fn crop_matches_the_training_crop() {
        let f = capture::load_rgb(root().join("fixtures/screens/note14/battle.png")).unwrap();
        for (c, r) in [(5, 10), (0, 0)] {
            let want = image::open(root().join(format!("fixtures/clips/crop_note14_{c}_{r}.png"))).unwrap().into_rgb8();
            let got = crop_tile(&f, &calib14(), c, r);
            assert_eq!(got.dimensions(), (CROP, CROP));
            let d = mean_abs_diff(&got, &want);
            assert!(d < 8.0, "tile ({c},{r}): mean abs diff {d}");
        }
    }

    #[test]
    fn classifier_agrees_with_training_on_real_clips() {
        let mut clf = PlayClassifier::load(root().join("assets/models/plays.onnx"), root().join("assets/models/plays.txt")).unwrap();
        for card in ["hog_rider", "fireball", "minions", "no_play"] {
            let strip = image::open(root().join(format!("fixtures/clips/{card}.png"))).unwrap().into_rgb8();
            let clip: Vec<image::RgbImage> =
                (0..CLIP_FRAMES as u32).map(|i| image::imageops::crop_imm(&strip, i * CROP, 0, CROP, CROP).to_image()).collect();
            let probs = clf.probs(&clip).unwrap();
            let best = probs.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
            assert_eq!(clf.classes[best], card);
        }
    }

    #[test]
    fn new_enemy_tag_is_proposed_once() {
        let t0 = Instant::now();
        let ms = |m| t0 + Duration::from_millis(m);
        let mut tags = TagProposer::default();
        assert_eq!(tags.update(ms(0), &[]), vec![]);
        assert_eq!(tags.update(ms(100), &[(9, 8)]), vec![(9, 8)]);
        assert_eq!(tags.update(ms(200), &[(9, 9)]), vec![], "the same unit walking");
        assert_eq!(tags.update(ms(300), &[]), vec![]);
        assert_eq!(tags.update(ms(400), &[(9, 9)]), vec![], "flicker");
        assert_eq!(tags.update(ms(500), &[(9, 10), (3, 5)]), vec![(3, 5)], "a second unit elsewhere");
    }

    #[test]
    fn motion_burst_in_a_quiet_area() {
        let quiet = vec![0.0f32; GRID_W * GRID_H];
        let mut burst = quiet.clone();
        for r in 5..8 {
            for c in 4..7 {
                burst[r * GRID_W + c] = 0.9;
            }
        }
        assert_eq!(burst_tiles(&quiet, &quiet), vec![]);
        assert_eq!(burst_tiles(&burst, &quiet), vec![(5, 6)]);
        assert_eq!(burst_tiles(&burst, &burst), vec![], "already busy: not new");
    }

    fn frame_at(at: Instant) -> Frame {
        Frame { seq: 0, width: 576, height: 1280, rgb: vec![0; 576 * 1280 * 3], captured_at: at }
    }

    #[test]
    fn clip_is_ready_after_the_last_frame_and_own_plays_are_ignored() {
        let calib = calib14();
        let t0 = Instant::now();
        let mut det = PlayDetector::default();
        det.own_deploy(t0, (3, 20));
        for i in 0..40u64 {
            let at = t0 + Duration::from_millis(50 * i);
            // An enemy appears at (10, 8) at 100 ms; something of ours lands at (3, 20).
            let enemies: Vec<(u32, u32)> = if i >= 2 { vec![(10, 8)] } else { vec![] };
            det.push(&frame_at(at), &enemies, &calib);
            if i == 2 {
                det.propose(at, (3, 20)); // our own deploy, seen as motion
            }
        }
        // 1150 ms after the tag appeared: the 8-frame clip (0..1050 ms) is complete.
        let ready = det.ready(t0 + Duration::from_millis(1250), &calib);
        assert_eq!(ready.len(), 1, "own play filtered, enemy play kept");
        assert_eq!(ready[0].0.tile, (10, 8));
        assert_eq!(ready[0].1.len(), CLIP_FRAMES);
        assert!(det.ready(t0 + Duration::from_millis(1300), &calib).is_empty(), "each proposal once");
    }
}
