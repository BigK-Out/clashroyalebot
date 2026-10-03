//! Run perception on image files and print one tab-separated line per file:
//! file, elixir, slot1..4, next (each "name score", "?best score", "(empty)").
//!
//!   perceive frames/unsure/            # all images in a directory
//!   perceive a.png b.png

use std::path::PathBuf;

use calib::Calibration;
use vision::{CardLibrary, Slot, read_elixir};

fn slot(s: &Slot) -> String {
    match s {
        Slot::Card(m) => format!("{}{} {:.2}", m.name, if m.raised { "^" } else { "" }, m.score),
        Slot::Empty => "(empty)".into(),
        Slot::Unknown { best: Some(b) } => format!("?{} {:.2}", b.name, b.score),
        Slot::Unknown { best: None } => "?".into(),
    }
}

fn main() -> anyhow::Result<()> {
    let calib = Calibration::load("calibration.toml")?;
    let cards = CardLibrary::load("assets/cards")?;
    let mut files: Vec<PathBuf> = Vec::new();
    for arg in std::env::args().skip(1) {
        let p = PathBuf::from(arg);
        if p.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(&p)?.filter_map(|e| e.ok().map(|e| e.path())).collect();
            v.retain(|p| p.extension().is_some_and(|e| e == "png" || e == "jpg"));
            v.sort();
            files.extend(v);
        } else {
            files.push(p);
        }
    }
    for f in files {
        let frame = capture::load_rgb(&f)?;
        match read_elixir(&frame, &calib) {
            None => println!("{}\t-", f.display()),
            Some(e) => {
                let h = cards.read_hand(&frame, &calib);
                let slots: Vec<String> = h.slots.iter().map(slot).collect();
                println!("{}\t{}\t{}\t{}", f.display(), e.value, slots.join("\t"), slot(&h.next));
            }
        }
    }
    Ok(())
}
