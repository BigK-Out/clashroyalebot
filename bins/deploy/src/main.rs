//! Place one card on a tile and measure latency (M4 test tool).
//!
//!   deploy --slot 1 --tile 3,20                 # blind: tap slot 1, then tile (3,20)
//!   deploy --card hog_rider --tile 14,17        # find the card in hand via the live feed
//!   deploy --card hog_rider --tile 14,17 --verify   # + wait until elixir drops on screen
//!   deploy --bench 10 [--bench-at 144,792]      # time 10 taps on a harmless spot

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use calib::Calibration;
use capture::{Frame, FrameSource, V4lSource};
use clap::Parser;
use input::{AdbShell, Deployer, TapBackend};
use vision::{CardLibrary, Slot};

#[derive(Parser)]
struct Args {
    /// Card slot 1..4 (counted from the left).
    #[arg(long, conflicts_with = "card")]
    slot: Option<usize>,
    /// Card name to find in hand (needs the live feed).
    #[arg(long)]
    card: Option<String>,
    /// Target tile "col,row" (cols 0..17 left→right, rows 0..31 top→bottom; mine are 17..31).
    #[arg(long, value_parser = parse_tile)]
    tile: Option<(u32, u32)>,
    /// Allow tiles on the enemy half (spells).
    #[arg(long)]
    enemy_half: bool,
    /// Watch the live feed until the deployment shows (elixir drop) and report latency.
    #[arg(long)]
    verify: bool,
    /// Instead of deploying, time N taps on a harmless spot.
    #[arg(long)]
    bench: Option<u32>,
    /// Device pixel for --bench taps (default: top-left, over the opponent's name in battle).
    #[arg(long, value_parser = parse_tile, default_value = "20,60")]
    bench_at: (u32, u32),
    #[arg(long)]
    serial: Option<String>,
    #[arg(long, default_value = "/dev/video10")]
    device: PathBuf,
    #[arg(long, default_value = "calibration.toml")]
    calibration: PathBuf,
    #[arg(long, default_value = "assets/cards")]
    cards: PathBuf,
}

fn parse_tile(s: &str) -> Result<(u32, u32), String> {
    let (c, r) = s.split_once(',').ok_or("expected col,row")?;
    Ok((c.trim().parse().map_err(|e| format!("{e}"))?, r.trim().parse().map_err(|e| format!("{e}"))?))
}

fn adb_path() -> String {
    let home = std::env::var("ANDROID_HOME")
        .unwrap_or_else(|_| format!("{}/Android/Sdk", std::env::var("HOME").unwrap_or_default()));
    let p = format!("{home}/platform-tools/adb");
    if std::path::Path::new(&p).exists() { p } else { "adb".into() }
}

/// Latest frame: scrcpy only sends frames on change, so read a few and keep the last.
fn fresh_frame(src: &mut V4lSource) -> anyhow::Result<Frame> {
    let mut f = src.next_frame()?;
    let t = Instant::now();
    while t.elapsed() < Duration::from_millis(60) {
        f = src.next_frame()?;
    }
    Ok(f)
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    let args = Args::parse();
    let calib = Calibration::load(&args.calibration)?;
    let t_open = Instant::now();
    let shell = AdbShell::open(&adb_path(), args.serial.as_deref())?;
    println!("adb shell ready in {:?}, screen {:?}", t_open.elapsed(), shell.screen_size());
    let mut deployer = Deployer::new(shell, calib.clone())?;

    if let Some(n) = args.bench {
        let mut times = Vec::new();
        for _ in 0..n {
            let t = Instant::now();
            deployer.backend().taps(&[args.bench_at])?;
            times.push(t.elapsed());
        }
        times.sort();
        println!(
            "{n} taps: min {:?}  median {:?}  max {:?}",
            times[0],
            times[times.len() / 2],
            times[times.len() - 1]
        );
        return Ok(());
    }

    let (col, row) = args.tile.context("--tile col,row is required")?;
    let needs_feed = args.card.is_some() || args.verify;
    let mut feed = if needs_feed {
        Some(V4lSource::open(&args.device).context("live feed (is scrcpy running and the viewer closed?)")?)
    } else {
        None
    };
    let cards = if needs_feed { Some(CardLibrary::load(&args.cards)?) } else { None };

    // Which slot, and elixir before.
    let mut elixir_before = None;
    let slot = if let (Some(src), Some(lib)) = (feed.as_mut(), cards.as_ref()) {
        let f = fresh_frame(src)?;
        let e = vision::read_elixir(&f, &calib).context("not in battle (no elixir bar)")?;
        elixir_before = Some(e);
        let hand = lib.read_hand(&f, &calib);
        let names: Vec<String> = hand.slots.iter().map(|s| s.name().unwrap_or("?").to_string()).collect();
        println!("elixir {} ({:.1}), hand {:?}", e.value, e.fill, names);
        match (&args.card, args.slot) {
            (Some(name), _) => hand
                .slots
                .iter()
                .position(|s| matches!(s, Slot::Card(m) if &m.name == name))
                .with_context(|| format!("{name} is not in hand {names:?}"))?,
            (None, Some(s)) => s.checked_sub(1).context("slots are 1..4")?,
            (None, None) => bail!("--slot or --card is required"),
        }
    } else {
        args.slot.context("--slot or --card is required")?.checked_sub(1).context("slots are 1..4")?
    };

    let t0 = Instant::now();
    let timing = deployer.deploy(slot, col, row, args.enemy_half)?;
    println!(
        "deployed slot {} -> tile ({col},{row}) at device {:?}: taps done in {:?}",
        slot + 1,
        deployer.tile_point(col, row),
        timing.taps
    );

    if let (true, Some(src), Some(before)) = (args.verify, feed.as_mut(), elixir_before) {
        // Visible when the fill drops by at least half an elixir (cards cost >= 1).
        let deadline = t0 + Duration::from_secs(3);
        loop {
            let f = src.next_frame()?;
            if f.captured_at < t0 {
                continue;
            }
            if let Some(e) = vision::read_elixir(&f, &calib)
                && e.fill < before.fill - 0.5
            {
                println!(
                    "visible: elixir {:.1} -> {:.1}, {:?} after start (frame captured {:?} after start)",
                    before.fill,
                    e.fill,
                    t0.elapsed(),
                    f.captured_at - t0
                );
                break;
            }
            if Instant::now() > deadline {
                println!("not visible within 3 s (not enough elixir, invalid tile, or tap missed)");
                break;
            }
        }
    }
    Ok(())
}
