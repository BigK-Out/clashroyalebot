//! Scripted opponent for self-play on the second phone: plays random legal cards and logs
//! every play to plays.jsonl (the labels for enemy plays seen by the observer).

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Context;
use clap::Parser;
use input::{AdbShell, Deployer};
use selfplay::{JsonlLog, PlayRecord, SparringPolicy};
use sparring::{Ctx, hand_names, step, verify};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "d8c5ce8a0406")]
    serial: String,
    #[arg(long, default_value = "calibration_note9.toml")]
    calibration: PathBuf,
    #[arg(long, default_value = "assets/cards")]
    cards: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long)]
    dry_run: bool,
    /// The sparring deck (8 slugs): hand templates come from assets/cards_all, any card.
    #[arg(long, value_delimiter = ',')]
    deck: Vec<String>,
}

fn adb_path() -> String {
    format!("{}/Android/Sdk/platform-tools/adb", std::env::var("HOME").unwrap_or_default())
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let a = Args::parse();
    let adb = adb_path();
    let calib = calib::Calibration::load(&a.calibration)?;
    let shell = AdbShell::open(&adb, Some(&a.serial)).context("adb shell to sparring phone")?;
    let mut ctx = Ctx {
        deployer: Deployer::new(shell, calib.clone())?,
        calib,
        cards: if a.deck.is_empty() {
            vision::CardLibrary::load(&a.cards)?
        } else {
            vision::CardLibrary::load_subset("assets/cards_all", &a.deck.iter().map(String::as_str).collect::<Vec<_>>())?
        },
        policy: SparringPolicy::new(a.seed),
        dry_run: a.dry_run,
    };
    let mut log: JsonlLog<PlayRecord> = JsonlLog::create(&a.out)?;
    let (mut seen_battle, mut last_bar) = (false, Instant::now());
    loop {
        let frame = capture::screencap(&adb, &a.serial, 576).context("screencap")?;
        if vision::read_elixir(&frame, &ctx.calib).is_some() {
            seen_battle = true;
            last_bar = Instant::now();
        } else if seen_battle && last_bar.elapsed() > Duration::from_secs(15) {
            tracing::info!("no elixir bar for 15 s: match over");
            return Ok(());
        }
        if let Some(mut rec) = step(&frame, &mut ctx)? {
            std::thread::sleep(Duration::from_millis(600));
            let after = capture::screencap(&adb, &a.serial, 576).context("screencap after play")?;
            rec.verified = verify(&rec.hand_before, &hand_names(&after, &ctx), rec.slot, &rec.card);
            tracing::info!("played {} slot {} -> {:?} verified {}", rec.card, rec.slot + 1, rec.tile, rec.verified);
            log.append(&rec)?;
        }
    }
}
