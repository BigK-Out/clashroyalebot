//! Runs Friendly Battles between the observer (Note 14, our bot recording) and the sparring
//! phone (Note 9), one match directory per battle. Never sends BACK; taps only on recognized
//! screens.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use clap::Parser;
use input::{AdbShell, TapBackend};
use selfplay::screens::{Flows, PhoneFlows, classify};
use selfplay::{MatchMeta, ObserverRecord, PlayRecord, estimate_offset, read_jsonl};

const OBSERVER: &str = "4xwskfkr7xp7w4xo";
const SPARRING: &str = "d8c5ce8a0406";

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 1)]
    matches: u32,
    #[arg(long, default_value = "dataset/selfplay")]
    out: PathBuf,
    #[arg(long, default_value = "selfplay_flows.toml")]
    flows: PathBuf,
    /// Rotate the sparring deck before every match through plan_decks(seed, rounds).
    #[arg(long)]
    rotate: Option<u64>,
    #[arg(long, default_value_t = 1)]
    rounds: usize,
}

fn adb() -> String {
    format!("{}/Android/Sdk/platform-tools/adb", std::env::var("HOME").unwrap_or_default())
}

/// Waits up to `timeout` for `serial` to show screen `want`.
fn wait_for(serial: &str, flows: &PhoneFlows, want: &str, timeout: Duration) -> anyhow::Result<()> {
    let t0 = Instant::now();
    loop {
        let f = capture::screencap(&adb(), serial, 576)?;
        let got = classify(&f, &flows.screens);
        if got.as_deref() == Some(want) {
            return Ok(());
        }
        if t0.elapsed() > timeout {
            let shot = format!("dataset/selfplay/lost_{serial}.png");
            if let Some(img) = image::RgbImage::from_raw(f.width, f.height, f.rgb) {
                img.save(&shot).ok();
            }
            bail!("{serial}: expected screen {want}, saw {got:?} for {timeout:?} (screenshot {shot})");
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

/// Taps the named point on `serial`, but only when the phone shows screen `on`.
fn tap_on(shell: &mut AdbShell, serial: &str, flows: &PhoneFlows, on: &str, tap: &str) -> anyhow::Result<()> {
    wait_for(serial, flows, on, Duration::from_secs(15))?;
    let (x, y) = *flows.taps.get(tap).with_context(|| format!("no tap {tap} in flows"))?;
    let (w, h) = shell.screen_size();
    shell.taps(&[((x * w as f64) as u32, (y * h as f64) as u32)])
}

/// Sets the sparring phone's current deck through the game's copy-deck link, then reads the
/// Decks screen back and fails unless it holds exactly `deck`. Main screen to main screen.
fn set_deck(shell: &mut AdbShell, serial: &str, flows: &PhoneFlows, deck: &[String]) -> anyhow::Result<()> {
    assert_eq!(serial, SPARRING, "deck editing only ever runs on the sparring phone");
    let link = selfplay::copy_deck_link(deck)?;
    wait_for(serial, flows, "main", Duration::from_secs(15))?;
    let out = Command::new(adb())
        .args(["-s", serial, "shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", &format!("\"{link}\""), "nullsroyale.rel.free"])
        .output()?;
    anyhow::ensure!(out.status.success(), "am start: {}", String::from_utf8_lossy(&out.stderr));
    tap_on(shell, serial, flows, "copy_deck", "copy_button")?;
    wait_for(serial, flows, "deck_view", Duration::from_secs(15))?;
    std::thread::sleep(Duration::from_millis(800));
    let names: Vec<&str> = deck.iter().map(String::as_str).collect();
    let lib = vision::CardLibrary::load_subset("assets/cards_all", &names)?;
    let got = selfplay::read_deck(&capture::screencap(&adb(), serial, 576)?, &lib);
    if !selfplay::deck_matches(&got, deck) {
        bail!("deck read back as {got:?}, wanted {deck:?}");
    }
    tap_on(shell, serial, flows, "deck_view", "battle_tab")?;
    wait_for(serial, flows, "main", Duration::from_secs(15))
}

fn run_match(dir: &Path, flows: &Flows, n9: &mut AdbShell, n14: &mut AdbShell, deck: &[String]) -> anyhow::Result<()> {
    // Inviting a friend who is in another battle opens spectating instead.
    wait_for(OBSERVER, &flows.note14, "main", Duration::from_secs(15))?;
    tap_on(n9, SPARRING, &flows.note9, "main", "friends_tab")?;
    tap_on(n9, SPARRING, &flows.note9, "friends_list", "khazar_row")?;
    tap_on(n9, SPARRING, &flows.note9, "friend_menu", "friendly_battle")?;
    tap_on(n9, SPARRING, &flows.note9, "mode_picker", "one_v_one")?;
    wait_for(SPARRING, &flows.note9, "invite_sent", Duration::from_secs(15))?;
    // The invite shows on the observer's Social tab, not as a popup over the main screen.
    tap_on(n14, OBSERVER, &flows.note14, "main", "social_tab")?;
    tap_on(n14, OBSERVER, &flows.note14, "invite_popup", "accept_invite")?;
    let dir_s = dir.to_string_lossy().to_string();
    let mut bot = Command::new("target/release/bot")
        .args(["--serial", OBSERVER, "--matches", "1", "--selfplay-dir", &dir_s])
        .stdout(std::fs::File::create(dir.join("bot.log"))?)
        .stderr(std::fs::File::create(dir.join("bot.err"))?)
        .spawn()?;
    let mut spar_args = vec!["--serial".to_string(), SPARRING.into(), "--out".into(), format!("{dir_s}/plays.jsonl")];
    if !deck.is_empty() {
        spar_args.extend(["--deck".into(), deck.join(",")]);
    }
    let mut spar = Command::new("target/release/sparring")
        .args(&spar_args)
        .stdout(std::fs::File::create(dir.join("sparring.log"))?)
        .stderr(std::fs::File::create(dir.join("sparring.err"))?)
        .spawn()?;
    let spar_status = spar.wait()?;
    let bot_status = bot.wait()?;
    if !spar_status.success() || !bot_status.success() {
        bail!("sparring {spar_status}, bot {bot_status}");
    }
    tap_on(n9, SPARRING, &flows.note9, "result", "result_ok")?;
    tap_on(n14, OBSERVER, &flows.note14, "result", "result_ok")?;
    Ok(())
}

/// After a failed match: tap each phone toward the main screen through recognized screens
/// only (result OK, Battle tab), waiting out battles. Up to about 5 minutes per phone.
fn recover(shell: &mut AdbShell, serial: &str, flows: &PhoneFlows) -> anyhow::Result<()> {
    for i in 0..150 {
        let f = capture::screencap(&adb(), serial, 576)?;
        let screen = classify(&f, &flows.screens);
        if i == 0
            && let Some(img) = image::RgbImage::from_raw(f.width, f.height, f.rgb.clone())
        {
            img.save(format!("dataset/selfplay/lost_{serial}_{}.png", selfplay::host_ms())).ok();
        }
        if screen.as_deref() == Some("main") {
            return Ok(());
        }
        if let Some(tap) = selfplay::screens::recovery_tap(screen.as_deref()) {
            let (x, y) = *flows.taps.get(tap).with_context(|| format!("no tap {tap} in flows"))?;
            let (w, h) = shell.screen_size();
            shell.taps(&[((x * w as f64) as u32, (y * h as f64) as u32)])?;
        }
        std::thread::sleep(Duration::from_secs(2));
    }
    bail!("{serial}: could not get back to the main screen")
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let a = Args::parse();
    let flows: Flows = toml::from_str(&std::fs::read_to_string(&a.flows)?)?;
    let mut n9 = AdbShell::open(&adb(), Some(SPARRING))?;
    let mut n14 = AdbShell::open(&adb(), Some(OBSERVER))?;
    let mut failures = 0;
    let decks = a.rotate.map(|seed| selfplay::plan_decks(seed, a.rounds)).unwrap_or_default();
    for i in 0..a.matches {
        let id = format!("{}_{i:03}", selfplay::host_ms());
        let dir = a.out.join(&id);
        std::fs::create_dir_all(&dir)?;
        let mut meta = MatchMeta { match_id: id.clone(), observer_serial: OBSERVER.into(), sparring_serial: SPARRING.into(), ..Default::default() };
        let deck: Vec<String> = if decks.is_empty() { Vec::new() } else { decks[i as usize % decks.len()].clone() };
        meta.sparring_deck = deck.clone();
        let result = if deck.is_empty() { Ok(()) } else { set_deck(&mut n9, SPARRING, &flows.note9, &deck) };
        match result.and_then(|()| run_match(&dir, &flows, &mut n9, &mut n14, &deck)) {
            Ok(()) => {
                let plays: Vec<PlayRecord> = read_jsonl(&dir.join("plays.jsonl")).unwrap_or_default();
                let obs: Vec<ObserverRecord> = read_jsonl(&dir.join("observer.jsonl")).unwrap_or_default();
                meta.clock_offset_ms = estimate_offset(&plays, &obs);
                meta.complete = true;
                failures = 0;
                tracing::info!("match {id}: {} plays, offset {:?} ms", plays.len(), meta.clock_offset_ms);
            }
            Err(e) => {
                meta.error = Some(format!("{e:#}"));
                meta.save(&dir)?;
                failures += 1;
                tracing::error!("match {id} failed ({failures} in a row): {e:#}");
                if failures >= 3 {
                    bail!("3 failed matches in a row, stopping; last: {e:#}");
                }
                recover(&mut n9, SPARRING, &flows.note9).context("recover sparring phone")?;
                recover(&mut n14, OBSERVER, &flows.note14).context("recover observer phone")?;
                continue;
            }
        }
        meta.save(&dir)?;
    }
    Ok(())
}
