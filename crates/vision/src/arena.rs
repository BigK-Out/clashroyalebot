//! The arena crop: the region object detection works on (dataset frames, labeling, inference).

use calib::{Calibration, NRect, PxRect};
use capture::Frame;

/// Extra margin around the 18x32 tile grid, in tiles: units stand on edge tiles and
/// sprites/health bars stick out above them.
const MARGIN_TILES: f64 = 1.5;

/// Normalized arena crop: full screen width, tile grid rows plus margin.
pub fn crop_rect(calib: &Calibration) -> NRect {
    let a = &calib.arena;
    let tile_h = (a.bottom_left.y - a.top_left.y) / calib::ARENA_ROWS as f64;
    let y0 = (a.top_left.y.min(a.top_right.y) - MARGIN_TILES * tile_h).max(0.0);
    let y1 = (a.bottom_left.y.max(a.bottom_right.y) + MARGIN_TILES * tile_h).min(1.0);
    NRect { x: 0.0, y: y0, w: 1.0, h: y1 - y0 }
}

/// Copies the arena crop out of a frame (RGB8).
pub fn crop(frame: &Frame, calib: &Calibration) -> (PxRect, Vec<u8>) {
    let r = crop_rect(calib).to_px(frame.width, frame.height);
    let mut rgb = Vec::with_capacity((r.w * r.h * 3) as usize);
    for y in r.y..r.y + r.h {
        let row = ((y * frame.width + r.x) * 3) as usize;
        rgb.extend_from_slice(&frame.rgb[row..row + (r.w * 3) as usize]);
    }
    (r, rgb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_crop_covers_arena_with_margin() {
        let c = Calibration::load(concat!(env!("CARGO_MANIFEST_DIR"), "/../../calibration.toml")).unwrap();
        let r = crop_rect(&c);
        assert!(r.y < c.arena.top_left.y && r.y + r.h > c.arena.bottom_left.y);
        let px = r.to_px(576, 1280);
        // ~0.10..0.77 of the height on the Redmi Note 14.
        assert!(px.y > 100 && px.y < 160 && px.h > 800 && px.h < 900, "{px:?}");
    }
}

/// Saves arena crops at a fixed interval while in battle (dataset collection).
pub struct Recorder {
    dir: std::path::PathBuf,
    every: std::time::Duration,
    last: Option<std::time::Instant>,
    pub saved: u64,
}

impl Recorder {
    pub fn new(dir: impl Into<std::path::PathBuf>, every: std::time::Duration) -> Self {
        Self { dir: dir.into(), every, last: None, saved: 0 }
    }

    /// Saves the frame's arena crop if in battle and the interval has passed.
    pub fn maybe_save(&mut self, frame: &Frame, calib: &Calibration, in_battle: bool) -> anyhow::Result<()> {
        if !in_battle || self.last.is_some_and(|t| frame.captured_at.duration_since(t) < self.every) {
            return Ok(());
        }
        self.last = Some(frame.captured_at);
        std::fs::create_dir_all(&self.dir)?;
        let (r, rgb) = crop(frame, calib);
        let ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis();
        let img = image::RgbImage::from_raw(r.w, r.h, rgb).ok_or_else(|| anyhow::anyhow!("crop size mismatch"))?;
        let path = self.dir.join(format!("{ms}.jpg"));
        let mut out = std::io::BufWriter::new(std::fs::File::create(&path)?);
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 92).encode_image(&img)?;
        self.saved += 1;
        Ok(())
    }
}
