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
