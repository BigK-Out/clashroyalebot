//! Cut unit images out of recorded arena crops (dataset for the unit-type classifier).
//!
//!   unit_crops dataset/raw dataset/units [--every 2]
//!
//! For each recording, finds unit level tags and saves the body region below each tag as a
//! 64x64 PNG: <frame>_<n>_<A|E>.png (A = mine, E = enemy), plus index.csv.

use std::io::Write;
use std::path::PathBuf;

use calib::Calibration;
use capture::Frame;
use vision::units::{Team, detect_units};

/// Recordings come from the 576x1280 scrcpy feed.
const FEED: (u32, u32) = (576, 1280);
const SIZE: u32 = 64;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(args.len() >= 2, "usage: unit_crops <raw_dir> <out_dir> [--every N]");
    let every: usize = args.iter().position(|a| a == "--every").and_then(|i| args.get(i + 1)).map_or(Ok(1), |v| v.parse())?;
    let (raw, out) = (PathBuf::from(&args[0]), PathBuf::from(&args[1]));
    std::fs::create_dir_all(&out)?;
    let calib = Calibration::load("calibration.toml")?;
    let mapping = calib.arena_mapping()?;
    let crop_px = vision::arena::crop_rect(&calib).to_px(FEED.0, FEED.1);

    let mut files: Vec<PathBuf> = std::fs::read_dir(&raw)?.filter_map(|e| e.ok().map(|e| e.path())).collect();
    files.retain(|p| p.extension().is_some_and(|e| e == "jpg" || e == "png"));
    files.sort();
    let mut index = std::fs::File::create(out.join("index.csv"))?;
    writeln!(index, "file,frame,team,col,row")?;
    let (mut frames, mut total) = (0, 0);
    for path in files.iter().step_by(every.max(1)) {
        let img = image::open(path)?.into_rgb8();
        if img.width() != crop_px.w || img.height() != crop_px.h {
            continue;
        }
        // Paste the arena crop back into a full feed-sized frame.
        let mut rgb = vec![0u8; (FEED.0 * FEED.1 * 3) as usize];
        for y in 0..crop_px.h {
            let dst = (((crop_px.y + y) * FEED.0 + crop_px.x) * 3) as usize;
            let src = (y * crop_px.w * 3) as usize;
            rgb[dst..dst + (crop_px.w * 3) as usize].copy_from_slice(&img.as_raw()[src..src + (crop_px.w * 3) as usize]);
        }
        let frame = Frame { seq: 0, width: FEED.0, height: FEED.1, rgb, captured_at: std::time::Instant::now() };
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        frames += 1;
        for (n, u) in detect_units(&frame, &calib, &mapping).iter().enumerate() {
            let Some(r) = vision::units::body_rect(u, &calib, FEED.0, FEED.1) else { continue };
            let full = image::RgbImage::from_raw(FEED.0, FEED.1, frame.rgb.clone()).unwrap();
            let body = image::imageops::crop_imm(&full, r.x, r.y, r.w, r.h).to_image();
            let body = image::imageops::resize(&body, SIZE, SIZE, image::imageops::FilterType::Triangle);
            let team = if u.team == Team::Ally { "A" } else { "E" };
            let name = format!("{stem}_{n}_{team}.png");
            body.save(out.join(&name))?;
            let (c, r) = u.tile.map_or((-1, -1), |(c, r)| (c as i32, r as i32));
            writeln!(index, "{name},{stem},{team},{c},{r}")?;
            total += 1;
        }
    }
    println!("{frames} frames -> {total} unit crops in {}", out.display());
    Ok(())
}
