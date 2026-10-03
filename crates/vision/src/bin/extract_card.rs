//! Cut a card's art out of a frame into the card library.
//!
//!   extract_card frames/x.png 3 mini_pekka --raised
//!   extract_card frames/x.png next minions

use std::path::PathBuf;

use anyhow::{Context, bail};
use calib::Calibration;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    if positional.len() < 3 {
        bail!("usage: extract_card <frame> <1-4|next> <name> [--raised] [--calib calibration.toml] [--out assets/cards]");
    }
    let frame = capture::load_rgb(positional[0])?;
    let calib = Calibration::load(flag("--calib").unwrap_or_else(|| "calibration.toml".into()))?;
    let slot = match positional[1].as_str() {
        "next" => calib.next_card,
        n => {
            let i: usize = n.parse().context("slot must be 1-4 or next")?;
            anyhow::ensure!((1..=4).contains(&i), "slot must be 1-4 or next");
            calib.card_slots[i - 1]
        }
    };
    let out = PathBuf::from(flag("--out").unwrap_or_else(|| "assets/cards".into()));
    std::fs::create_dir_all(&out)?;
    let path = out.join(format!("{}.png", positional[2]));
    vision::hand::save_template(&frame, slot, args.iter().any(|a| a == "--raised"), &path)?;
    println!("wrote {}", path.display());
    Ok(())
}
