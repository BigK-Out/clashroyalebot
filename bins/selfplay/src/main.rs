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
    /// Both phones play scripted (sparring) and both are recorded to video; with --rotate
    /// both decks rotate (the Note 14 only ever in its spare deck 4).
    #[arg(long)]
    both: bool,
    /// With --rotate: focus decks (one per champion + Mirror, Three Musketeers, fast spells).
    #[arg(long)]
    focus: bool,
    /// Observer bot binary (bot evaluation: run an older build against the same opponents).
    #[arg(long, default_value = "target/release/bot")]
    bot: String,
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
    let link = selfplay::copy_deck_link(deck)?;
    wait_for(serial, flows, "main", Duration::from_secs(15))?;
    let out = Command::new(adb())
        .args(["-s", serial, "shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", &format!("\"{link}\""), "nullsroyale.rel.free"])
        .output()?;
    anyhow::ensure!(out.status.success(), "am start: {}", String::from_utf8_lossy(&out.stderr));
    wait_for(serial, flows, "copy_deck", Duration::from_secs(15))?;
    if serial != SPARRING {
        // The observer is the main account: only ever copy into its spare deck.
        let guard = flows.guards.iter().find(|g| g.name == "spare_deck_selected").context("no spare_deck_selected guard")?;
        // The popup slides in over the tabs: look a few times before refusing.
        let mut selected = false;
        for _ in 0..6 {
            if selfplay::screens::matches(&capture::screencap(&adb(), serial, 576)?, guard) {
                selected = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        if !selected {
            bail!("{serial}: the spare deck 4 is not selected, refusing to copy a deck");
        }
    }
    tap_on(shell, serial, flows, "copy_deck", "copy_button")?;
    wait_for(serial, flows, "deck_view", Duration::from_secs(15))?;
    let names: Vec<&str> = deck.iter().map(String::as_str).collect();
    let lib = vision::CardLibrary::load_subset("assets/cards_all", &names)?;
    // The new cards animate in: read until they settle.
    let mut got = Vec::new();
    for _ in 0..8 {
        std::thread::sleep(Duration::from_millis(500));
        got = selfplay::read_deck(&capture::screencap(&adb(), serial, 576)?, &lib);
        if selfplay::deck_matches(&got, deck) {
            break;
        }
    }
    if !selfplay::deck_matches(&got, deck) {
        bail!("deck read back as {got:?}, wanted {deck:?}");
    }
    tap_on(shell, serial, flows, "deck_view", "battle_tab")?;
    wait_for(serial, flows, "main", Duration::from_secs(15))
}

/// Note 9 invites, Note 14 accepts: both end up in the battle.
fn start_battle(flows: &Flows, n9: &mut AdbShell, n14: &mut AdbShell) -> anyhow::Result<()> {
    // Inviting a friend who is in another battle opens spectating instead.
    wait_for(OBSERVER, &flows.note14, "main", Duration::from_secs(15))?;
    tap_on(n9, SPARRING, &flows.note9, "main", "friends_tab")?;
    tap_on(n9, SPARRING, &flows.note9, "friends_list", "khazar_row")?;
    tap_on(n9, SPARRING, &flows.note9, "friend_menu", "friendly_battle")?;
    tap_on(n9, SPARRING, &flows.note9, "mode_picker", "one_v_one")?;
    wait_for(SPARRING, &flows.note9, "invite_sent", Duration::from_secs(15))?;
    // The invite shows on the observer's Social tab, not as a popup over the main screen.
    tap_on(n14, OBSERVER, &flows.note14, "main", "social_tab")?;
    tap_on(n14, OBSERVER, &flows.note14, "invite_popup", "accept_invite")
}

/// Records `serial`'s screen to an mkv (survives being stopped at any point) at 2 Mbit/s.
fn start_recording(serial: &str, path: &Path) -> anyhow::Result<std::process::Child> {
    Ok(Command::new("scrcpy")
        .args(["-s", serial, "--no-window", "--no-audio", "--no-control", "--max-size=1280", "--max-fps=30", "--video-bit-rate=2M"])
        .arg(format!("--record={}", path.display()))
        .stdout(std::process::Stdio::null())
        .stderr(std::fs::File::create(path.with_extension("log"))?)
        .spawn()?)
}

fn stop_recording(mut rec: std::process::Child) -> anyhow::Result<()> {
    // SIGINT lets scrcpy finish the file; a plain kill would be SIGKILL.
    Command::new("kill").args(["-INT", &rec.id().to_string()]).status()?;
    rec.wait()?;
    Ok(())
}

fn spawn_sparring(dir: &Path, serial: &str, calibration: &str, name: &str, deck: &[String], seed: u64) -> anyhow::Result<std::process::Child> {
    let mut args = vec!["--serial".to_string(), serial.into(), "--calibration".into(), calibration.into()];
    args.extend(["--out".into(), dir.join(format!("plays_{name}.jsonl")).to_string_lossy().to_string(), "--seed".into(), seed.to_string()]);
    if !deck.is_empty() {
        args.extend(["--deck".into(), deck.join(",")]);
    }
    Ok(Command::new("target/release/sparring")
        .args(&args)
        .stdout(std::fs::File::create(dir.join(format!("sparring_{name}.log")))?)
        .stderr(std::fs::File::create(dir.join(format!("sparring_{name}.err")))?)
        .spawn()?)
}

/// Both phones play scripted and both are recorded; labels for each side's plays.
fn run_match_both(dir: &Path, flows: &Flows, n9: &mut AdbShell, n14: &mut AdbShell, decks: (&[String], &[String]), seed: u64) -> anyhow::Result<()> {
    let rec9 = start_recording(SPARRING, &dir.join("note9.mkv"))?;
    let rec14 = start_recording(OBSERVER, &dir.join("note14.mkv"))?;
    std::fs::write(dir.join("recording_started_ms"), selfplay::host_ms().to_string())?;
    let result = (|| {
        start_battle(flows, n9, n14)?;
        let mut s9 = spawn_sparring(dir, SPARRING, "calibration_note9.toml", "note9", decks.0, seed)?;
        let mut s14 = spawn_sparring(dir, OBSERVER, "calibration.toml", "note14", decks.1, seed + 1)?;
        let (st9, st14) = (s9.wait()?, s14.wait()?);
        if !st9.success() || !st14.success() {
            bail!("sparring note9 {st9}, note14 {st14}");
        }
        tap_on(n9, SPARRING, &flows.note9, "result", "result_ok")?;
        tap_on(n14, OBSERVER, &flows.note14, "result", "result_ok")
    })();
    stop_recording(rec9)?;
    stop_recording(rec14)?;
    result
}

fn run_match(dir: &Path, flows: &Flows, n9: &mut AdbShell, n14: &mut AdbShell, deck: &[String], bot_bin: &str) -> anyhow::Result<vision::result::Crowns> {
    start_battle(flows, n9, n14)?;
    let dir_s = dir.to_string_lossy().to_string();
    let mut bot = Command::new(bot_bin)
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
    wait_for(OBSERVER, &flows.note14, "result", Duration::from_secs(20))?;
    std::thread::sleep(Duration::from_millis(1500)); // crowns animate in
    let crowns = vision::result::read_crowns(&capture::screencap(&adb(), OBSERVER, 576)?);
    tap_on(n9, SPARRING, &flows.note9, "result", "result_ok")?;
    tap_on(n14, OBSERVER, &flows.note14, "result", "result_ok")?;
    Ok(crowns)
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
    let plan = |seed| if a.focus { selfplay::plan_focus_decks(seed, a.rounds) } else { selfplay::plan_decks(seed, a.rounds) };
    let decks = a.rotate.map(plan).unwrap_or_default();
    // A different shuffle for the Note 14, so decks meet different opponents every round.
    let decks14 = a.rotate.filter(|_| a.both).map(|seed| plan(seed + 1_000)).unwrap_or_default();
    for i in 0..a.matches {
        let id = format!("{}_{i:03}", selfplay::host_ms());
        let dir = a.out.join(&id);
        std::fs::create_dir_all(&dir)?;
        let mut meta = MatchMeta { match_id: id.clone(), observer_serial: OBSERVER.into(), sparring_serial: SPARRING.into(), ..Default::default() };
        let deck: Vec<String> = if decks.is_empty() { Vec::new() } else { decks[i as usize % decks.len()].clone() };
        let deck14: Vec<String> = if decks14.is_empty() { Vec::new() } else { decks14[i as usize % decks14.len()].clone() };
        meta.sparring_deck = deck.clone();
        meta.observer_deck = deck14.clone();
        let mut result = if deck.is_empty() { Ok(()) } else { set_deck(&mut n9, SPARRING, &flows.note9, &deck) };
        if result.is_ok() && !deck14.is_empty() {
            result = set_deck(&mut n14, OBSERVER, &flows.note14, &deck14);
        }
        let result = result.and_then(|()| {
            if a.both {
                run_match_both(&dir, &flows, &mut n9, &mut n14, (&deck, &deck14), i as u64 * 2)
            } else {
                run_match(&dir, &flows, &mut n9, &mut n14, &deck, &a.bot).map(|c| {
                    meta.observer_crowns = Some(c.mine);
                    meta.sparring_crowns = Some(c.theirs);
                    meta.bot = Some(a.bot.clone());
                    tracing::info!("crowns: bot {} - sparring {}", c.mine, c.theirs);
                })
            }
        });
        match result {
            Ok(()) if a.both => {
                let n9p: Vec<PlayRecord> = read_jsonl(&dir.join("plays_note9.jsonl")).unwrap_or_default();
                let n14p: Vec<PlayRecord> = read_jsonl(&dir.join("plays_note14.jsonl")).unwrap_or_default();
                meta.complete = true;
                failures = 0;
                tracing::info!("match {id}: note9 {} plays, note14 {} plays", n9p.len(), n14p.len());
            }
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
