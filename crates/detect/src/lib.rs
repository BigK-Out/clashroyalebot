//! Learned detectors via ONNX Runtime: YOLOv8 on the arena crop, unit-type classifier.

pub mod plays;
pub mod units;

use std::path::Path;

use anyhow::Context;
use calib::{ArenaMapping, Calibration, NPoint};
use capture::Frame;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

/// Model input side (square, letterboxed).
pub const INPUT: usize = 640;
/// YOLO letterbox padding gray.
const PAD: f32 = 114.0 / 255.0;

#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub class: usize,
    pub score: f32,
    /// Normalized screen box (0..1 of the full frame): x0, y0, x1, y1.
    pub bbox: [f32; 4],
    /// Tile under the unit's feet (bottom-center of the box), if inside the arena.
    pub tile: Option<(u32, u32)>,
}

/// Letterbox geometry: crop pixels → model pixels is `p * scale + pad`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
}

impl Letterbox {
    pub fn new(w: usize, h: usize) -> Self {
        let scale = (INPUT as f32 / w as f32).min(INPUT as f32 / h as f32);
        let (nw, nh) = (w as f32 * scale, h as f32 * scale);
        Self { scale, pad_x: ((INPUT as f32 - nw) / 2.0).floor(), pad_y: ((INPUT as f32 - nh) / 2.0).floor() }
    }

    fn to_crop(self, x: f32, y: f32) -> (f32, f32) {
        ((x - self.pad_x) / self.scale, (y - self.pad_y) / self.scale)
    }
}

/// RGB8 image → NCHW float tensor data (bilinear resize into the letterbox).
pub fn preprocess(rgb: &[u8], w: usize, h: usize) -> (Vec<f32>, Letterbox) {
    let lb = Letterbox::new(w, h);
    let plane = INPUT * INPUT;
    let mut out = vec![PAD; 3 * plane];
    let (nw, nh) = ((w as f32 * lb.scale).round() as usize, (h as f32 * lb.scale).round() as usize);
    for oy in 0..nh {
        let sy = ((oy as f32 + 0.5) / lb.scale - 0.5).clamp(0.0, (h - 1) as f32);
        let (y0, fy) = (sy as usize, sy.fract());
        let y1 = (y0 + 1).min(h - 1);
        for ox in 0..nw {
            let sx = ((ox as f32 + 0.5) / lb.scale - 0.5).clamp(0.0, (w - 1) as f32);
            let (x0, fx) = (sx as usize, sx.fract());
            let x1 = (x0 + 1).min(w - 1);
            let at = |x: usize, y: usize, c: usize| rgb[(y * w + x) * 3 + c] as f32;
            let dst = (oy + lb.pad_y as usize) * INPUT + ox + lb.pad_x as usize;
            for c in 0..3 {
                let top = at(x0, y0, c) * (1.0 - fx) + at(x1, y0, c) * fx;
                let bot = at(x0, y1, c) * (1.0 - fx) + at(x1, y1, c) * fx;
                out[c * plane + dst] = (top * (1.0 - fy) + bot * fy) / 255.0;
            }
        }
    }
    (out, lb)
}

/// Raw box in model pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawBox {
    pub class: usize,
    pub score: f32,
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

fn iou(a: &RawBox, b: &RawBox) -> f32 {
    let iw = (a.x1.min(b.x1) - a.x0.max(b.x0)).max(0.0);
    let ih = (a.y1.min(b.y1) - a.y0.max(b.y0)).max(0.0);
    let inter = iw * ih;
    let union = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

/// Decodes YOLOv8 output `[1, 4 + classes, anchors]` (cx, cy, w, h, class scores) and applies
/// per-class non-maximum suppression.
pub fn decode(out: &[f32], classes: usize, anchors: usize, conf: f32, iou_thresh: f32) -> Vec<RawBox> {
    let mut boxes: Vec<RawBox> = (0..anchors)
        .filter_map(|a| {
            let (class, score) = (0..classes)
                .map(|c| (c, out[(4 + c) * anchors + a]))
                .max_by(|x, y| x.1.total_cmp(&y.1))?;
            if score < conf {
                return None;
            }
            let (cx, cy, w, h) = (out[a], out[anchors + a], out[2 * anchors + a], out[3 * anchors + a]);
            Some(RawBox { class, score, x0: cx - w / 2.0, y0: cy - h / 2.0, x1: cx + w / 2.0, y1: cy + h / 2.0 })
        })
        .collect();
    boxes.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut keep: Vec<RawBox> = Vec::new();
    for b in boxes {
        if keep.iter().all(|k| k.class != b.class || iou(k, &b) < iou_thresh) {
            keep.push(b);
        }
    }
    keep
}

pub struct Detector {
    session: Session,
    pub classes: Vec<String>,
    pub conf: f32,
    pub iou: f32,
}

impl Detector {
    /// Loads `model.onnx` with class names from `classes.txt` (one per line).
    pub fn load(model: impl AsRef<Path>, classes: impl AsRef<Path>) -> anyhow::Result<Self> {
        let classes: Vec<String> = std::fs::read_to_string(classes.as_ref())
            .with_context(|| format!("read {}", classes.as_ref().display()))?
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(String::from)
            .collect();
        // Builder errors carry the (non-Send) builder; keep only the message.
        let msg = |e: ort::Error<ort::session::builder::SessionBuilder>| anyhow::anyhow!("{e}");
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(msg)?
            .with_intra_threads(4)
            .map_err(msg)?
            .commit_from_file(model.as_ref())
            .with_context(|| format!("load {}", model.as_ref().display()))?;
        Ok(Self { session, classes, conf: 0.4, iou: 0.5 })
    }

    pub fn detect(&mut self, frame: &Frame, calib: &Calibration, mapping: &ArenaMapping) -> anyhow::Result<Vec<Detection>> {
        let (crop_px, rgb) = vision::arena::crop(frame, calib);
        let (input, lb) = preprocess(&rgb, crop_px.w as usize, crop_px.h as usize);
        let tensor = Tensor::from_array(([1usize, 3, INPUT, INPUT], input))?;
        let outputs = self.session.run(ort::inputs![tensor])?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        anyhow::ensure!(shape.len() == 3 && shape[1] as usize == 4 + self.classes.len(), "unexpected output shape {shape:?}");
        let anchors = shape[2] as usize;
        let (fw, fh) = (frame.width as f32, frame.height as f32);
        Ok(decode(data, self.classes.len(), anchors, self.conf, self.iou)
            .into_iter()
            .map(|b| {
                let (cx0, cy0) = lb.to_crop(b.x0, b.y0);
                let (cx1, cy1) = lb.to_crop(b.x1, b.y1);
                let bbox = [
                    ((cx0 + crop_px.x as f32) / fw).clamp(0.0, 1.0),
                    ((cy0 + crop_px.y as f32) / fh).clamp(0.0, 1.0),
                    ((cx1 + crop_px.x as f32) / fw).clamp(0.0, 1.0),
                    ((cy1 + crop_px.y as f32) / fh).clamp(0.0, 1.0),
                ];
                let feet = NPoint { x: ((bbox[0] + bbox[2]) / 2.0) as f64, y: bbox[3] as f64 };
                Detection { class: b.class, score: b.score, bbox, tile: mapping.screen_to_tile_index(feet) }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_tall_crop_pads_sides() {
        // Arena crop on the phone feed is ~576x845: height-limited.
        let lb = Letterbox::new(576, 845);
        assert!((lb.scale - 640.0 / 845.0).abs() < 1e-6);
        assert_eq!(lb.pad_y, 0.0);
        assert!(lb.pad_x > 100.0 && lb.pad_x < 103.0, "{lb:?}");
        let (x, y) = lb.to_crop(lb.pad_x, 640.0);
        assert!(x.abs() < 1e-3 && (y - 845.0).abs() < 1e-2);
    }

    #[test]
    fn preprocess_fills_padding_and_scales_values() {
        let (w, h) = (100, 200);
        let rgb = vec![255u8; w * h * 3];
        let (t, lb) = preprocess(&rgb, w, h);
        assert_eq!(t.len(), 3 * INPUT * INPUT);
        assert!((t[0] - PAD).abs() < 1e-6, "left padding");
        let center = INPUT / 2 * INPUT + INPUT / 2;
        assert!((t[center] - 1.0).abs() < 1e-6 && (t[2 * INPUT * INPUT + center] - 1.0).abs() < 1e-6);
        assert!(lb.pad_x > 0.0);
    }

    /// Builds a [4 + classes, anchors] output from (cx, cy, w, h, class, score) rows.
    fn output(rows: &[(f32, f32, f32, f32, usize, f32)], classes: usize) -> Vec<f32> {
        let n = rows.len();
        let mut out = vec![0.0; (4 + classes) * n];
        for (a, &(cx, cy, w, h, c, s)) in rows.iter().enumerate() {
            out[a] = cx;
            out[n + a] = cy;
            out[2 * n + a] = w;
            out[3 * n + a] = h;
            out[(4 + c) * n + a] = s;
        }
        out
    }

    #[test]
    fn decode_thresholds_and_suppresses_duplicates_per_class() {
        let rows = [
            (100.0, 100.0, 20.0, 30.0, 0, 0.9),
            (102.0, 101.0, 20.0, 30.0, 0, 0.8), // duplicate of the first
            (102.0, 101.0, 20.0, 30.0, 1, 0.7), // same place, other class: kept
            (300.0, 300.0, 20.0, 30.0, 0, 0.2), // below threshold
        ];
        let boxes = decode(&output(&rows, 2), 2, rows.len(), 0.4, 0.5);
        assert_eq!(boxes.len(), 2);
        assert_eq!((boxes[0].class, boxes[0].score), (0, 0.9));
        assert_eq!(boxes[1].class, 1);
        assert!((boxes[0].x0 - 90.0).abs() < 1e-6 && (boxes[0].y1 - 115.0).abs() < 1e-6);
    }
}
