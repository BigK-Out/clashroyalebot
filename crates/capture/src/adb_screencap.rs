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
