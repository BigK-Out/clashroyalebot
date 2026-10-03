//! Hand reading: template-match each card slot (and the next-card preview) against a card library.

use std::path::Path;

use anyhow::Context;
use calib::{Calibration, NRect};
use capture::Frame;

use crate::patch::Patch;

/// Card art inside a slot, as fractions of the slot rect: skips the border and the
/// elixir-cost badge at the bottom (it overlaps the art and is shared by many cards).
const ART: (f64, f64, f64, f64) = (0.06, 0.05, 0.94, 0.72);

/// A selected card is drawn raised by this fraction of the slot height (66 px of 230).
const RAISED_DY: f64 = -0.287;

/// Below this score a slot is reported as Unknown.
const MIN_SCORE: f32 = 0.6;

/// Below this texture a slot is a flat empty placeholder.
const EMPTY_TEXTURE: f32 = 0.35;

#[derive(Debug, Clone, PartialEq)]
pub struct CardMatch {
    pub name: String,
    pub score: f32,
    /// The card is lifted (selected, about to be placed).
    pub raised: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Slot {
    Card(CardMatch),
    /// Slot is empty (card just played, replacement animating in).
    Empty,
    /// Something is there but it matches no known card well enough.
    Unknown { best: Option<CardMatch> },
}

impl Slot {
    pub fn name(&self) -> Option<&str> {
        match self {
            Slot::Card(m) => Some(&m.name),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hand {
    pub slots: [Slot; 4],
    pub next: Slot,
}

struct Template {
    name: String,
    patch: Patch,
}

/// Template name for an empty slot (blue placeholder with a faint crown).
pub const EMPTY_TEMPLATE: &str = "_empty";

/// Card templates: PNGs in a directory, file stem = card name (`name@variant` for extra looks).
pub struct CardLibrary {
    templates: Vec<Template>,
}

/// Pixel rect of the card art for a slot rect, optionally shifted to the raised position.
fn art_region(slot: NRect, frame: &Frame, dy: f64) -> (f64, f64, f64, f64) {
    let (fw, fh) = (frame.width as f64, frame.height as f64);
    let (sx, sy, sw, sh) = (slot.x * fw, (slot.y + dy * slot.h) * fh, slot.w * fw, slot.h * fh);
    (sx + ART.0 * sw, sy + ART.1 * sh, sx + ART.2 * sw, sy + ART.3 * sh)
}

impl CardLibrary {
    pub fn load(dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let dir = dir.as_ref();
        let mut templates = Vec::new();
        for entry in std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("png") {
                continue;
            }
            // `name@variant.png` adds another look of the same card (raised, next-preview, ...).
            let stem = path.file_stem().and_then(|s| s.to_str()).context("bad file name")?;
            let name = stem.split('@').next().unwrap_or(stem).to_string();
            let frame = capture::load_rgb(&path)?;
            let patch = Patch::from_frame(&frame).with_context(|| format!("empty template {}", path.display()))?;
            templates.push(Template { name, patch });
        }
        templates.sort_by(|a, b| a.name.cmp(&b.name));
        anyhow::ensure!(!templates.is_empty(), "no card templates in {}", dir.display());
        Ok(Self { templates })
    }

    /// Distinct card names (variants collapsed).
    pub fn names(&self) -> Vec<&str> {
        let mut n: Vec<&str> = self.templates.iter().map(|t| t.name.as_str()).collect();
        n.dedup();
        n
    }

    fn best(&self, patch: &Patch) -> Option<(usize, f32)> {
        self.templates.iter().enumerate().map(|(i, t)| (i, t.patch.ncc(patch))).max_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// Classifies one slot, trying the normal and the raised position.
    fn read_slot(&self, frame: &Frame, slot: NRect, try_raised: bool) -> Slot {
        let offsets: &[(f64, bool)] = if try_raised { &[(0.0, false), (RAISED_DY, true)] } else { &[(0.0, false)] };
        let mut best: Option<CardMatch> = None;
        let mut flat = true;
        for &(dy, raised) in offsets {
            let (x0, y0, x1, y1) = art_region(slot, frame, dy);
            let Some(patch) = Patch::from_region(frame, x0, y0, x1, y1) else { continue };
            if !raised {
                flat = patch.texture() < EMPTY_TEXTURE;
            }
            if let Some((i, score)) = self.best(&patch)
                && best.as_ref().is_none_or(|b| score > b.score)
            {
                best = Some(CardMatch { name: self.templates[i].name.clone(), score, raised });
            }
        }
        match best {
            Some(m) if m.score >= MIN_SCORE && m.name == EMPTY_TEMPLATE => Slot::Empty,
            Some(m) if m.score >= MIN_SCORE => Slot::Card(m),
            _ if flat => Slot::Empty,
            best => Slot::Unknown { best },
        }
    }

    pub fn read_hand(&self, frame: &Frame, calib: &Calibration) -> Hand {
        Hand {
            slots: std::array::from_fn(|i| self.read_slot(frame, calib.card_slots[i], true)),
            next: self.read_slot(frame, calib.next_card, false),
        }
    }
}

/// Saves the art region of a slot as a template PNG (for building the card library).
pub fn save_template(frame: &Frame, slot: NRect, raised: bool, path: impl AsRef<Path>) -> anyhow::Result<()> {
    let (x0, y0, x1, y1) = art_region(slot, frame, if raised { RAISED_DY } else { 0.0 });
    let (x0, y0, x1, y1) = (x0.round() as u32, y0.round() as u32, x1.round() as u32, y1.round() as u32);
    let mut img = image::RgbImage::new(x1 - x0, y1 - y0);
    for (x, y, p) in img.enumerate_pixels_mut() {
        *p = image::Rgb(frame.pixel(x0 + x, y0 + y));
    }
    img.save(path.as_ref()).with_context(|| format!("write {}", path.as_ref().display()))
}
