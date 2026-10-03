//! Small image patches for template matching: crop → box-resample → per-channel normalize.

use capture::Frame;

/// Fixed feature size: card art is ~165x150 px on a 1080-wide screen; this keeps enough
/// detail to tell cards apart while being cheap to compare.
pub const PATCH_W: usize = 24;
pub const PATCH_H: usize = 24;

/// Zero-mean, unit-variance patch: RGB (PATCH_W x PATCH_H x 3) plus a grayscale copy.
#[derive(Clone, Debug)]
pub struct Patch {
    rgb: Vec<f32>,
    gray: Vec<f32>,
    /// Mean HSV-style saturation of the raw patch, 0..1. Unaffordable cards are drawn ~grey.
    saturation: f32,
}

impl Patch {
    /// Crops [x0,x1) x [y0,y1) (pixels, clamped to the frame) and box-resamples it.
    /// Returns None if the region is empty after clamping.
    pub fn from_region(frame: &Frame, x0: f64, y0: f64, x1: f64, y1: f64) -> Option<Self> {
        let (fw, fh) = (frame.width as f64, frame.height as f64);
        let (x0, x1) = (x0.clamp(0.0, fw), x1.clamp(0.0, fw));
        let (y0, y1) = (y0.clamp(0.0, fh), y1.clamp(0.0, fh));
        if x1 - x0 < 1.0 || y1 - y0 < 1.0 {
            return None;
        }
        let mut out = Vec::with_capacity(PATCH_W * PATCH_H * 3);
        let (bw, bh) = ((x1 - x0) / PATCH_W as f64, (y1 - y0) / PATCH_H as f64);
        for py in 0..PATCH_H {
            let sy0 = (y0 + py as f64 * bh) as usize;
            let sy1 = ((y0 + (py + 1) as f64 * bh) as usize).max(sy0 + 1).min(frame.height as usize);
            for px in 0..PATCH_W {
                let sx0 = (x0 + px as f64 * bw) as usize;
                let sx1 = ((x0 + (px + 1) as f64 * bw) as usize).max(sx0 + 1).min(frame.width as usize);
                let mut acc = [0u32; 3];
                for sy in sy0..sy1 {
                    let row = &frame.rgb[(sy * frame.width as usize + sx0) * 3..(sy * frame.width as usize + sx1) * 3];
                    for p in row.chunks_exact(3) {
                        acc[0] += p[0] as u32;
                        acc[1] += p[1] as u32;
                        acc[2] += p[2] as u32;
                    }
                }
                let n = ((sy1 - sy0) * (sx1 - sx0)) as f32;
                out.extend(acc.iter().map(|&a| a as f32 / n));
            }
        }
        Some(Self::new(out))
    }

    /// Whole frame as a patch (used for template images, which are pre-cropped).
    pub fn from_frame(frame: &Frame) -> Option<Self> {
        Self::from_region(frame, 0.0, 0.0, frame.width as f64, frame.height as f64)
    }

    fn new(raw: Vec<f32>) -> Self {
        let saturation = raw
            .chunks_exact(3)
            .map(|p| {
                let max = p[0].max(p[1]).max(p[2]);
                let min = p[0].min(p[1]).min(p[2]);
                if max > 0.0 { (max - min) / max } else { 0.0 }
            })
            .sum::<f32>()
            / (raw.len() / 3) as f32;
        let mut gray: Vec<f32> = raw.chunks_exact(3).map(|p| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]).collect();
        let mut rgb = raw;
        // Per channel, so a uniform brightness/tint shift (selection glow, compression) cancels out.
        for c in 0..3 {
            normalize_strided(&mut rgb, c, 3);
        }
        normalize_strided(&mut gray, 0, 1);
        Self { rgb, gray, saturation }
    }

    /// Normalized cross-correlation in [-1, 1]; 1 = identical up to per-channel gain/offset.
    pub fn ncc(&self, other: &Patch) -> f32 {
        dot_mean(&self.rgb, &other.rgb)
    }

    /// Grayscale NCC: for greyed-out (unaffordable) cards, where color is gone.
    pub fn ncc_gray(&self, other: &Patch) -> f32 {
        dot_mean(&self.gray, &other.gray)
    }

    /// True when the patch has almost no color (greyed-out card).
    pub fn is_desaturated(&self) -> bool {
        self.saturation < 0.12
    }

    /// Mean absolute normalized value: ~0 for a flat patch, ~0.8 for textured art.
    pub fn texture(&self) -> f32 {
        self.rgb.iter().map(|x| x.abs()).sum::<f32>() / self.rgb.len() as f32
    }
}

fn normalize_strided(v: &mut [f32], offset: usize, stride: usize) {
    let n = (v.len() / stride) as f32;
    let mean = v.iter().skip(offset).step_by(stride).sum::<f32>() / n;
    let var = v.iter().skip(offset).step_by(stride).map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt().max(1.0); // flat patches (empty slot) stay flat instead of exploding
    for x in v.iter_mut().skip(offset).step_by(stride) {
        *x = (*x - mean) / std;
    }
}

fn dot_mean(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>() / a.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn frame(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> Frame {
        let mut rgb = Vec::new();
        for y in 0..h {
            for x in 0..w {
                rgb.extend(f(x, y));
            }
        }
        Frame { seq: 0, width: w, height: h, rgb, captured_at: Instant::now() }
    }

    #[test]
    fn identical_patterns_score_one_and_brightness_cancels() {
        let a = frame(48, 48, |x, y| [((x * 5) % 255) as u8, ((y * 3) % 255) as u8, ((x + y) % 255) as u8]);
        let b = frame(48, 48, |x, y| [((x * 5) % 255 / 2 + 20) as u8, ((y * 3) % 255 / 2) as u8, ((x + y) % 255 / 2) as u8]);
        let pa = Patch::from_frame(&a).unwrap();
        let pb = Patch::from_frame(&b).unwrap();
        assert!((pa.ncc(&pa) - 1.0).abs() < 1e-3);
        assert!(pa.ncc(&pb) > 0.99, "{}", pa.ncc(&pb));
    }

    #[test]
    fn different_patterns_score_low() {
        let a = frame(48, 48, |x, _| if x < 24 { [255; 3] } else { [0; 3] });
        let b = frame(48, 48, |_, y| if y < 24 { [255; 3] } else { [0; 3] });
        let s = Patch::from_frame(&a).unwrap().ncc(&Patch::from_frame(&b).unwrap());
        assert!(s.abs() < 0.1, "{s}");
    }

    #[test]
    fn greyed_copy_matches_in_gray() {
        let color = frame(48, 48, |x, y| [(x * 5) as u8, (y * 4) as u8, ((x * y) % 200) as u8]);
        let grey = frame(48, 48, |x, y| {
            let [r, g, b] = [(x * 5) as f32, (y * 4) as f32, ((x * y) % 200) as f32];
            let l = (0.299 * r + 0.587 * g + 0.114 * b) as u8 / 2 + 100; // washed out
            [l, l, l]
        });
        let (pc, pg) = (Patch::from_frame(&color).unwrap(), Patch::from_frame(&grey).unwrap());
        assert!(pg.is_desaturated() && !pc.is_desaturated());
        assert!(pg.ncc_gray(&pc) > 0.98, "{}", pg.ncc_gray(&pc));
    }

    #[test]
    fn flat_patch_has_no_texture() {
        let flat = frame(30, 30, |_, _| [10, 60, 140]);
        assert!(Patch::from_frame(&flat).unwrap().texture() < 1e-3);
    }

    #[test]
    fn out_of_bounds_region_is_clamped_or_none() {
        let f = frame(10, 10, |_, _| [1, 2, 3]);
        assert!(Patch::from_region(&f, -5.0, -5.0, 5.0, 5.0).is_some());
        assert!(Patch::from_region(&f, 20.0, 20.0, 30.0, 30.0).is_none());
    }
}
