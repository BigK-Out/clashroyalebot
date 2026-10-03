//! Print unit tags with predicted types: classify_debug <frames...>

use calib::Calibration;
use detect::units::UnitClassifier;
use vision::units::detect_units;

fn main() -> anyhow::Result<()> {
    let calib = Calibration::load("calibration.toml")?;
    let mapping = calib.arena_mapping()?;
    let mut clf = UnitClassifier::load("assets/models/units.onnx", "assets/models/units.txt")?;
    for path in std::env::args().skip(1) {
        let frame = capture::load_rgb(&path)?;
        let t0 = std::time::Instant::now();
        let units = detect_units(&frame, &calib, &mapping);
        let t1 = std::time::Instant::now();
        let types = clf.classify(&frame, &calib, &units)?;
        let t2 = std::time::Instant::now();
        let desc: Vec<String> = units
            .iter()
            .zip(&types)
            .map(|(u, t)| {
                let team = if u.team == vision::units::Team::Ally { "A" } else { "E" };
                let tile = u.tile.map_or("-".into(), |(c, r)| format!("{c},{r}"));
                match t {
                    Some(t) => format!("{team}:{}@{tile}({:.2})", t.name, t.prob),
                    None => format!("{team}:?@{tile}"),
                }
            })
            .collect();
        println!(
            "{}: tags {:.1} ms, classify {:.1} ms | {}",
            std::path::Path::new(&path).file_stem().unwrap().to_string_lossy(),
            (t1 - t0).as_secs_f64() * 1e3,
            (t2 - t1).as_secs_f64() * 1e3,
            desc.join("  ")
        );
    }
    Ok(())
}
