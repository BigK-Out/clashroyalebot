//! Unit-type classifier (MobileNetV3-small, ONNX) on the body crop below each level tag.

use std::path::Path;

use anyhow::Context;
use calib::Calibration;
use capture::Frame;
use ort::session::Session;
use ort::value::Tensor;
use vision::units::{Unit, body_rect};

/// Classifier input side; crops are resized to SIZE x SIZE like the training data.
pub const SIZE: u32 = 64;
/// Class used for "not a unit" (timers, popups, tower labels).
pub const JUNK: &str = "junk";

#[derive(Debug, Clone, PartialEq)]
pub struct UnitType {
    pub name: String,
    pub prob: f32,
}

pub struct UnitClassifier {
    session: Session,
    pub classes: Vec<String>,
}

fn softmax_max(logits: &[f32]) -> (usize, f32) {
    let m = logits.iter().copied().fold(f32::MIN, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let (i, e) = exps.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
    (i, e / sum)
}

impl UnitClassifier {
    /// Loads `units.onnx` and the class list `units.txt` (one name per line, logit order).
    pub fn load(model: impl AsRef<Path>, classes: impl AsRef<Path>) -> anyhow::Result<Self> {
        let classes = std::fs::read_to_string(classes.as_ref())
            .with_context(|| format!("read {}", classes.as_ref().display()))?
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(String::from)
            .collect();
        let session = Session::builder()?
            .commit_from_file(model.as_ref())
            .with_context(|| format!("load {}", model.as_ref().display()))?;
        Ok(Self { session, classes })
    }

    /// Classifies all units in one batch. `None` where the crop doesn't fit in the frame.
    pub fn classify(&mut self, frame: &Frame, calib: &Calibration, units: &[Unit]) -> anyhow::Result<Vec<Option<UnitType>>> {
        let img = image::RgbImage::from_raw(frame.width, frame.height, frame.rgb.clone()).context("frame size")?;
        let rects: Vec<_> = units.iter().map(|u| body_rect(u, calib, frame.width, frame.height)).collect();
        let crops: Vec<image::RgbImage> = rects
            .iter()
            .flatten()
            .map(|r| {
                let c = image::imageops::crop_imm(&img, r.x, r.y, r.w, r.h).to_image();
                image::imageops::resize(&c, SIZE, SIZE, image::imageops::FilterType::Triangle)
            })
            .collect();
        if crops.is_empty() {
            return Ok(vec![None; units.len()]);
        }
        let plane = (SIZE * SIZE) as usize;
        let mut input = vec![0f32; crops.len() * 3 * plane];
        for (n, c) in crops.iter().enumerate() {
            for (i, p) in c.pixels().enumerate() {
                for ch in 0..3 {
                    input[n * 3 * plane + ch * plane + i] = p[ch] as f32 / 255.0;
                }
            }
        }
        let tensor = Tensor::from_array(([crops.len(), 3, SIZE as usize, SIZE as usize], input))?;
        let outputs = self.session.run(ort::inputs![tensor])?;
        let (_, logits) = outputs[0].try_extract_tensor::<f32>()?;
        let k = self.classes.len();
        let mut preds = logits.chunks_exact(k).map(|l| {
            let (i, prob) = softmax_max(l);
            UnitType { name: self.classes[i].clone(), prob }
        });
        Ok(rects.iter().map(|r| r.and_then(|_| preds.next())).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_picks_max_with_probability() {
        let (i, p) = softmax_max(&[0.0, 2.0, 0.0]);
        assert_eq!(i, 1);
        assert!((p - 0.787).abs() < 0.01, "{p}");
    }
}
