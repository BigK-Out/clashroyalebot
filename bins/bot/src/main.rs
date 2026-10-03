//! The bot: capture → perceive → state → policy → tap, with per-action latency logging.
//!
//!   bot                    # play battles as they come (start them yourself), Ctrl+C to stop
//!   bot --start --matches 3    # tap "Battle", play 3 matches (OK + Battle between), exit
//!   bot --dry-run          # decide and log, never tap

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Context;
use brain::{Action, HogCycle, Policy};
use calib::Calibration;
use capture::{Frame, FrameSource, V4lSource};
use clap::Parser;
use input::{AdbShell, Deployer, TapBackend};
use state::Tracker;
use vision::CardLibrary;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    serial: Option<String>,
    #[arg(long, default_value = "/dev/video10")]
    device: PathBuf,
    #[arg(long, default_value = "calibration.toml")]
    calibration: PathBuf,
    #[arg(long, default_value = "assets/cards")]
    cards: PathBuf,
    /// Log decisions without tapping.
    #[arg(long)]
    dry_run: bool,
    /// Tap the main-menu Battle button first (normalized position in calibration space).
    #[arg(long)]
    start: bool,
    /// Exit after this many battles have ended.
    #[arg(long)]
    matches: Option<u32>,
    /// Record arena crops (2 per second, battle only) into this directory for the detector dataset.
    #[arg(long)]
    record: Option<PathBuf>,
    /// Debug frames: saved on battle start/end and every 10 s.
    #[arg(long, default_value = "frames/bot")]
    debug_dir: PathBuf,
}

fn save_debug(f: &Frame, dir: &std::path::Path, tag: &str) {
    let ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
    let path = dir.join(format!("{ms}_{tag}.png"));
    let res = std::fs::create_dir_all(dir).map_err(anyhow::Error::from).and_then(|_| {
        image::RgbImage::from_raw(f.width, f.height, f.rgb.clone())
            .context("bad frame")?
            .save(&path)
            .map_err(anyhow::Error::from)
    });
    if let Err(e) = res {
        tracing::warn!("debug frame: {e:#}");
    }
}

/// Main-menu "Battle" button (normalized; measured on the 1080x2400 menu).
const BATTLE_BUTTON: (f64, f64) = (0.5, 0.822);
/// "OK" on the result screen.
const RESULT_OK: (f64, f64) = (0.5, 0.847);

/// After a deploy, ignore frames until the game has shown it (elixir drop takes ~160 ms).
const SETTLE: Duration = Duration::from_millis(450);

fn adb_path() -> String {
    let home = std::env::var("ANDROID_HOME")
        .unwrap_or_else(|_| format!("{}/Android/Sdk", std::env::var("HOME").unwrap_or_default()));
    let p = format!("{home}/platform-tools/adb");
    if std::path::Path::new(&p).exists() { p } else { "adb".into() }
}

fn spawn_capture(device: PathBuf) -> Arc<Mutex<Option<Frame>>> {
    let latest = Arc::new(Mutex::new(None));
    let slot = latest.clone();
    std::thread::spawn(move || {
        loop {
            match V4lSource::open(&device) {
                Ok(mut src) => {
                    tracing::info!("capturing {}", src.describe());
                    while let Ok(f) = src.next_frame() {
                        *slot.lock().unwrap() = Some(f);
                    }
                }
                Err(e) => tracing::warn!("capture: {e:#} (scrcpy running? viewer closed?)"),
            }
            std::thread::sleep(Duration::from_secs(1));
        }
    });
    latest
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    let args = Args::parse();
    let calib = Calibration::load(&args.calibration)?;
    let cards = CardLibrary::load(&args.cards)?;
    let shell = AdbShell::open(&adb_path(), args.serial.as_deref()).context("adb")?;
    let (w, h) = shell.screen_size();
    let mut deployer = Deployer::new(shell, calib.clone())?;
    let mapping = calib.arena_mapping()?;
    // Unit types are optional: without the model the bot still sees positions and teams.
    let mut classifier = detect::units::UnitClassifier::load("assets/models/units.onnx", "assets/models/units.txt")
        .inspect_err(|e| tracing::warn!("unit classifier disabled: {e:#}"))
        .ok();
    let latest = spawn_capture(args.device.clone());
    let mut tracker = Tracker::default();
    let mut policy = HogCycle::default();

    let to_px = |(x, y): (f64, f64)| ((x * w as f64) as u32, (y * h as f64) as u32);

    // Never start a match blind: require live frames first.
    let t_wait = Instant::now();
    while latest.lock().unwrap().is_none() {
        anyhow::ensure!(
            t_wait.elapsed() < Duration::from_secs(10),
            "no frames from {} after 10 s: start scrcpy --v4l2-sink first",
            args.device.display()
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    if args.start {
        tracing::info!("tapping Battle");
        deployer.backend().taps(&[to_px(BATTLE_BUTTON)])?;
    }

    let mut last_seq = 0;
    let mut last_frame_at = Instant::now();
    let mut stall_warned = false;
    let mut settle_until = Instant::now();
    let mut was_in_battle = false;
    let mut battles_done = 0;
    let mut actions = 0u32;
    let mut last_status = Instant::now();
    let mut recorder = args.record.clone().map(|d| vision::arena::Recorder::new(d, Duration::from_millis(500)));
    let mut last_debug = Instant::now();
    loop {
        let Some(frame) = latest.lock().unwrap().take_if(|f| f.seq != last_seq) else {
            // In battle the screen always animates; seconds without frames = capture is dead.
            if was_in_battle && !stall_warned && last_frame_at.elapsed() > Duration::from_secs(3) {
                tracing::error!("no frames for 3 s during battle: capture stalled (scrcpy died?)");
                stall_warned = true;
            }
            std::thread::sleep(Duration::from_millis(2));
            continue;
        };
        last_frame_at = Instant::now();
        stall_warned = false;
        last_seq = frame.seq;
        let t_perceive = Instant::now();
        let elixir = vision::read_elixir(&frame, &calib);
        let hand = elixir.is_some().then(|| cards.read_hand(&frame, &calib));
        tracker.update(frame.captured_at, elixir, hand.as_ref());
        if elixir.is_some() {
            let units = vision::units::detect_units(&frame, &calib, &mapping);
            let kinds: Vec<Option<String>> = match classifier.as_mut().map(|c| c.classify(&frame, &calib, &units)) {
                Some(Ok(types)) => types.into_iter().map(|t| t.map(|t| t.name)).collect(),
                Some(Err(e)) => {
                    tracing::warn!("classify: {e:#}");
                    Vec::new()
                }
                None => Vec::new(),
            };
            tracker.update_units(frame.captured_at, &units, &kinds);
        }
        let state = tracker.state().clone();
        let perceive_ms = t_perceive.elapsed().as_secs_f64() * 1e3;
        if let Some(rec) = recorder.as_mut()
            && let Err(e) = rec.maybe_save(&frame, &calib, state.in_battle)
        {
            tracing::warn!("record: {e:#}");
        }

        if last_status.elapsed() >= Duration::from_secs(5) {
            last_status = Instant::now();
            tracing::info!(
                "status: in_battle {} t={:.0}s | read elixir {:?} | hand {:?} next {:?} | enemies {:?}",
                state.in_battle,
                state.battle_time.as_secs_f64(),
                elixir.map(|e| e.value),
                state.hand.iter().map(|c| c.as_deref().unwrap_or("-")).collect::<Vec<_>>(),
                state.next.as_deref().unwrap_or("-"),
                state.enemies.iter().zip(&state.enemy_kinds).map(|(t, k)| format!("{}@{},{}", k.as_deref().unwrap_or("?"), t.0, t.1)).collect::<Vec<_>>(),
            );
        }
        if last_debug.elapsed() >= Duration::from_secs(10) {
            last_debug = Instant::now();
            save_debug(&frame, &args.debug_dir, if elixir.is_some() { "bar" } else { "nobar" });
        }

        if state.in_battle != was_in_battle {
            was_in_battle = state.in_battle;
            save_debug(&frame, &args.debug_dir, if state.in_battle { "start" } else { "end" });
            if state.in_battle {
                tracing::info!("battle started");
                actions = 0;
            } else {
                battles_done += 1;
                tracing::info!(
                    "battle over after {:.0} s, {actions} actions, {} frames recorded",
                    state.battle_time.as_secs_f64(),
                    recorder.as_ref().map_or(0, |r| r.saved)
                );
                if args.matches.is_some_and(|m| battles_done >= m) {
                    return Ok(());
                }
                if args.start {
                    // Result screen → OK → main menu → Battle. The end grace already waited 10 s.
                    tracing::info!("requeue: OK, then Battle");
                    deployer.backend().taps(&[to_px(RESULT_OK)])?;
                    std::thread::sleep(Duration::from_secs(5));
                    deployer.backend().taps(&[to_px(BATTLE_BUTTON)])?;
                }
            }
        }
        if frame.captured_at < settle_until {
            continue;
        }
        let Some(action) = policy.decide(&state) else { continue };
        let t_decided = Instant::now();
        let Action::Deploy { slot, ref card, col, row, enemy_half, why } = action;
        let taps_ms = if args.dry_run {
            0.0
        } else {
            match deployer.deploy(slot, col, row, enemy_half) {
                Ok(t) => t.taps.as_secs_f64() * 1e3,
                Err(e) => {
                    tracing::warn!("deploy failed: {e:#}");
                    continue;
                }
            }
        };
        actions += 1;
        tracker.mark_played(slot);
        policy.on_action(&action, &state);
        settle_until = Instant::now() + SETTLE;
        tracing::info!(
            "t={:5.1}s {:<10} slot {} -> ({col},{row}) {why:?} | elixir {} | enemies {:?} | capture->decide {:.0} ms (perceive {:.1}) | taps {:.0} ms | capture->tapped {:.0} ms | hand {:?}",
            state.battle_time.as_secs_f64(),
            card,
            slot + 1,
            state.elixir,
            state.enemies,
            (t_decided - frame.captured_at).as_secs_f64() * 1e3,
            perceive_ms,
            taps_ms,
            frame.captured_at.elapsed().as_secs_f64() * 1e3,
            state.hand.iter().map(|c| c.as_deref().unwrap_or("-")).collect::<Vec<_>>(),
        );
    }
}
