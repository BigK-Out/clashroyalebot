//! Draw unit-tag detections onto frames: units_debug <out_dir> <frames...>
//! Blue box = ally tag, red = enemy tag, filled square = estimated feet.

use calib::Calibration;
use vision::units::{Team, detect_units};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let out = std::path::PathBuf::from(args.next().ok_or_else(|| anyhow::anyhow!("usage: units_debug <out_dir> <frames...>"))?);
    std::fs::create_dir_all(&out)?;
    let calib = Calibration::load("calibration.toml")?;
    let mapping = calib.arena_mapping()?;
    for path in args {
        let frame = capture::load_rgb(&path)?;
        let t = std::time::Instant::now();
        let units = detect_units(&frame, &calib, &mapping);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let mut img = image::RgbImage::from_raw(frame.width, frame.height, frame.rgb.clone()).unwrap();
        let (fw, fh) = (frame.width as f32, frame.height as f32);
        for u in &units {
            let c = if u.team == Team::Ally { [40, 120, 255] } else { [255, 30, 30] };
            let [x0, y0, x1, y1] = [u.tag[0] * fw, u.tag[1] * fh, u.tag[2] * fw, u.tag[3] * fh].map(|v| v as i32);
            for k in 0..3 {
                let (a0, b0, a1, b1) = (x0 - 2 - k, y0 - 2 - k, x1 + 2 + k, y1 + 2 + k);
                for x in a0..=a1 {
                    for y in [b0, b1] {
                        if x >= 0 && y >= 0 && (x as u32) < frame.width && (y as u32) < frame.height {
                            img.put_pixel(x as u32, y as u32, image::Rgb(c));
                        }
                    }
                }
                for y in b0..=b1 {
                    for x in [a0, a1] {
                        if x >= 0 && y >= 0 && (x as u32) < frame.width && (y as u32) < frame.height {
                            img.put_pixel(x as u32, y as u32, image::Rgb(c));
                        }
                    }
                }
            }
            let (fx, fy) = ((u.feet.x as f32 * fw) as i32, (u.feet.y as f32 * fh) as i32);
            for x in fx - 4..=fx + 4 {
                for y in fy - 4..=fy + 4 {
                    if x >= 0 && y >= 0 && (x as u32) < frame.width && (y as u32) < frame.height {
                        img.put_pixel(x as u32, y as u32, image::Rgb(c));
                    }
                }
            }
        }
        let name = std::path::Path::new(&path).file_stem().unwrap().to_string_lossy().to_string();
        img.save(out.join(format!("{name}.png")))?;
        let ally = units.iter().filter(|u| u.team == Team::Ally).count();
        println!("{name}: {ally} ally, {} enemy, {ms:.1} ms", units.len() - ally);
    }
    Ok(())
}
