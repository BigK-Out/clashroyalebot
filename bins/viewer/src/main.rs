//! Debug viewer: live frames, capture stats, perception overlay (elixir, hand).
//!
//!   viewer                          # /dev/video10 (scrcpy --v4l2-sink)
//!   viewer --dir frames/ --fps 30   # replay saved frames
//!   viewer --headless               # no window, log fps once per second

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use calib::Calibration;
use capture::{DirSource, Frame, FrameSource, V4lSource};
use vision::{CardLibrary, Elixir, Hand, Slot};
use clap::Parser;
use eframe::egui;

#[derive(Parser)]
struct Args {
    /// v4l2 capture device written by scrcpy.
    #[arg(long, default_value = "/dev/video10")]
    device: PathBuf,
    /// Replay PNG/JPEG frames from this directory instead of the device.
    #[arg(long)]
    dir: Option<PathBuf>,
    /// Replay rate for --dir.
    #[arg(long, default_value_t = 30.0)]
    fps: f64,
    /// Log stats to stdout instead of opening a window.
    #[arg(long)]
    headless: bool,
    /// Where the S key saves frames.
    #[arg(long, default_value = "frames")]
    save_dir: PathBuf,
    #[arg(long, default_value = "calibration.toml")]
    calibration: PathBuf,
    /// Card template directory.
    #[arg(long, default_value = "assets/cards")]
    cards: PathBuf,
    /// Disable auto-saving frames where a card slot matches no known card.
    /// (By default, in battle, at most one such frame per second goes to <save-dir>/unsure/.)
    #[arg(long)]
    no_auto_save: bool,
    /// Record arena crops (2 per second, battle only) into this directory for the detector dataset.
    #[arg(long)]
    record: Option<PathBuf>,
}

/// Auto-save trigger: some slot matched no card. (Weak matches are mostly greyed-out cards,
/// which are read correctly, so they don't count.)
fn is_unsure(p: &Perceived) -> bool {
    let Some(h) = &p.hand else { return false };
    h.slots.iter().chain([&h.next]).any(|s| matches!(s, Slot::Unknown { .. }))
}

/// Calibration + card library; perception is skipped if either is missing.
struct Perceiver {
    calib: Calibration,
    mapping: calib::ArenaMapping,
    cards: CardLibrary,
}

/// Perception result for one frame.
#[derive(Clone)]
struct Perceived {
    elixir: Option<Elixir>,
    hand: Option<Hand>,
    units: Vec<vision::units::Unit>,
    ms: f64,
}

impl Perceiver {
    fn load(args: &Args) -> Option<Self> {
        let load = || -> anyhow::Result<Self> {
            let calib = Calibration::load(&args.calibration)?;
            let mapping = calib.arena_mapping()?;
            Ok(Self { calib, mapping, cards: CardLibrary::load(&args.cards)? })
        };
        load().inspect_err(|e| tracing::warn!("perception disabled: {e:#}")).ok()
    }

    fn perceive(&self, f: &Frame) -> Perceived {
        let t0 = Instant::now();
        let elixir = vision::read_elixir(f, &self.calib);
        // Hand only means something in battle; the elixir bar is the battle signal.
        let hand = elixir.is_some().then(|| self.cards.read_hand(f, &self.calib));
        let units =
            if elixir.is_some() { vision::units::detect_units(f, &self.calib, &self.mapping) } else { Vec::new() };
        Perceived { elixir, hand, units, ms: t0.elapsed().as_secs_f64() * 1e3 }
    }
}

fn slot_text(s: &Slot) -> String {
    match s {
        Slot::Card(m) => format!("{}{} {:.2}", m.name, if m.raised { "^" } else { "" }, m.score),
        Slot::Empty => "(empty)".into(),
        Slot::Unknown { best: Some(b) } => format!("?{} {:.2}", b.name, b.score),
        Slot::Unknown { best: None } => "?".into(),
    }
}

fn perceived_text(p: &Perceived) -> String {
    match (&p.elixir, &p.hand) {
        (Some(e), Some(h)) => {
            let enemy = p.units.iter().filter(|u| u.team == vision::units::Team::Enemy).count();
            format!(
                "elixir {} ({:.1}) | {} | next {} | units {} ally {} enemy | {:.1} ms",
                e.value,
                e.fill,
                h.slots.iter().map(slot_text).collect::<Vec<_>>().join(", "),
                slot_text(&h.next),
                p.units.len() - enemy,
                enemy,
                p.ms
            )
        }
        _ => format!("not in battle | {:.1} ms", p.ms),
    }
}

/// Frames per second over a sliding one-second window.
struct FpsCounter {
    window_start: Instant,
    count: u32,
    fps: f64,
}

impl FpsCounter {
    fn new() -> Self {
        Self { window_start: Instant::now(), count: 0, fps: 0.0 }
    }

    /// Returns true when a new one-second reading is available.
    fn tick(&mut self) -> bool {
        self.count += 1;
        let elapsed = self.window_start.elapsed();
        if elapsed >= Duration::from_secs(1) {
            self.fps = self.count as f64 / elapsed.as_secs_f64();
            self.count = 0;
            self.window_start = Instant::now();
            return true;
        }
        false
    }
}

#[derive(Default)]
struct Shared {
    latest: Option<Frame>,
    source_desc: String,
    capture_fps: f64,
    /// Dequeue → RGB ready (YUV conversion cost), last frame.
    convert_ms: f64,
    perceived: Option<Perceived>,
    error: Option<String>,
}

fn open_source(args: &Args) -> anyhow::Result<Box<dyn FrameSource>> {
    Ok(match &args.dir {
        Some(dir) => Box::new(DirSource::open(dir, Some(args.fps))?),
        None => Box::new(V4lSource::open(&args.device)?),
    })
}

/// Capture loop on its own thread; reopens the source if it fails (e.g. scrcpy restarted).
fn spawn_capture(args: Args, shared: Arc<Mutex<Shared>>, on_frame: impl Fn() + Send + 'static) {
    let perceiver = Perceiver::load(&args);
    let unsure_dir = (!args.no_auto_save).then(|| args.save_dir.join("unsure"));
    let mut last_auto_save: Option<Instant> = None;
    let mut recorder = args.record.clone().map(|d| vision::arena::Recorder::new(d, Duration::from_millis(500)));
    std::thread::spawn(move || {
        loop {
            let mut src = match open_source(&args) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("open source: {e:#}; retrying in 1s");
                    shared.lock().unwrap().error = Some(format!("{e:#}"));
                    std::thread::sleep(Duration::from_secs(1));
                    continue;
                }
            };
            tracing::info!("capturing from {}", src.describe());
            {
                let mut s = shared.lock().unwrap();
                s.source_desc = src.describe();
                s.error = None;
            }
            let mut fps = FpsCounter::new();
            loop {
                match src.next_frame() {
                    Ok(frame) => {
                        let convert_ms = frame.captured_at.elapsed().as_secs_f64() * 1e3;
                        fps.tick();
                        let perceived = perceiver.as_ref().map(|p| p.perceive(&frame));
                        if let (Some(rec), Some(p), Some(per)) = (recorder.as_mut(), &perceived, &perceiver)
                            && let Err(e) = rec.maybe_save(&frame, &per.calib, p.elixir.is_some())
                        {
                            tracing::warn!("record: {e:#}");
                        }
                        if let (Some(dir), Some(p)) = (&unsure_dir, &perceived)
                            && is_unsure(p)
                            && last_auto_save.is_none_or(|t| t.elapsed() >= Duration::from_secs(1))
                        {
                            last_auto_save = Some(Instant::now());
                            match save_frame(&frame, dir) {
                                Ok(path) => tracing::info!("unsure, saved {} ({})", path.display(), perceived_text(p)),
                                Err(e) => tracing::warn!("auto-save: {e:#}"),
                            }
                        }
                        let mut s = shared.lock().unwrap();
                        if perceived.is_some() {
                            s.perceived = perceived;
                        }
                        s.latest = Some(frame);
                        s.capture_fps = fps.fps;
                        s.convert_ms = convert_ms;
                    }
                    Err(e) => {
                        tracing::warn!("capture: {e:#}; reopening");
                        shared.lock().unwrap().error = Some(format!("{e:#}"));
                        break;
                    }
                }
                on_frame();
            }
        }
    });
}

struct ViewerApp {
    shared: Arc<Mutex<Shared>>,
    texture: Option<egui::TextureHandle>,
    shown_seq: u64,
    /// Capture → texture upload, for the frame on screen.
    frame_age_ms: f64,
    display_fps: FpsCounter,
    /// Frame on screen, kept for saving with S.
    last_frame: Option<Frame>,
    save_dir: PathBuf,
    status: String,
}

/// Saves a frame as PNG named by wall-clock millis so dumps sort chronologically.
fn save_frame(f: &Frame, dir: &std::path::Path) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis();
    let path = dir.join(format!("{ms}.png"));
    image::RgbImage::from_raw(f.width, f.height, f.rgb.clone())
        .ok_or_else(|| anyhow::anyhow!("frame buffer size mismatch"))?
        .save(&path)?;
    Ok(path)
}

impl eframe::App for ViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let (desc, capture_fps, convert_ms, error, perceived, new_frame) = {
            let mut s = self.shared.lock().unwrap();
            let new_frame = s.latest.take_if(|f| f.seq != self.shown_seq);
            (s.source_desc.clone(), s.capture_fps, s.convert_ms, s.error.clone(), s.perceived.clone(), new_frame)
        };

        if let Some(f) = new_frame {
            let img = egui::ColorImage::from_rgb([f.width as usize, f.height as usize], &f.rgb);
            match &mut self.texture {
                Some(t) => t.set(img, egui::TextureOptions::LINEAR),
                None => {
                    self.texture = Some(ui.ctx().load_texture("frame", img, egui::TextureOptions::LINEAR))
                }
            }
            self.frame_age_ms = f.captured_at.elapsed().as_secs_f64() * 1e3;
            self.shown_seq = f.seq;
            self.display_fps.tick();
            self.last_frame = Some(f);
        }

        if ui.input(|i| i.key_pressed(egui::Key::S))
            && let Some(f) = &self.last_frame
        {
            self.status = match save_frame(f, &self.save_dir) {
                Ok(p) => format!("saved {}", p.display()),
                Err(e) => format!("save failed: {e:#}"),
            };
            tracing::info!("{}", self.status);
        }

        egui::Panel::top("stats").show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.monospace(format!(
                    "capture {capture_fps:5.1} fps | display {:5.1} fps | convert {convert_ms:4.1} ms | age {:4.1} ms | #{}",
                    self.display_fps.fps, self.frame_age_ms, self.shown_seq
                ));
            });
            if let Some(p) = &perceived {
                let color = if p.elixir.is_some() { egui::Color32::LIGHT_GREEN } else { egui::Color32::GRAY };
                ui.add(egui::Label::new(egui::RichText::new(perceived_text(p)).monospace().color(color)).wrap());
            }
            ui.small(format!(
                "{desc}   [S] save frame to {}/   auto-save of unsure frames: on   {}",
                self.save_dir.display(),
                self.status
            ));
            if let Some(e) = error {
                ui.colored_label(egui::Color32::LIGHT_RED, e);
            }
        });

        egui::CentralPanel::default().show(ui, |ui| {
            if let Some(tex) = &self.texture {
                // Fit to the panel, keep aspect ratio.
                let avail = ui.available_size();
                let size = tex.size_vec2();
                let scale = (avail.x / size.x).min(avail.y / size.y);
                let resp = ui
                    .centered_and_justified(|ui| {
                        ui.add(egui::Image::from_texture(egui::load::SizedTexture::new(tex.id(), size * scale)))
                    })
                    .inner;
                // Unit overlay: tag box + feet dot (blue = mine, red = enemy).
                if let Some(p) = &perceived {
                    let r = resp.rect;
                    let painter = ui.painter_at(r);
                    let at = |x: f32, y: f32| egui::pos2(r.min.x + x * r.width(), r.min.y + y * r.height());
                    for u in &p.units {
                        let c = match u.team {
                            vision::units::Team::Ally => egui::Color32::from_rgb(60, 160, 255),
                            vision::units::Team::Enemy => egui::Color32::from_rgb(255, 50, 50),
                        };
                        let tag = egui::Rect::from_min_max(at(u.tag[0], u.tag[1]), at(u.tag[2], u.tag[3])).expand(3.0);
                        painter.rect_stroke(tag, 1.0, egui::Stroke::new(2.0, c), egui::StrokeKind::Outside);
                        painter.circle_filled(at(u.feet.x as f32, u.feet.y as f32), 4.0, c);
                    }
                }
            } else {
                ui.centered_and_justified(|ui| ui.label("waiting for frames…"));
            }
        });
    }
}

fn run_headless(args: Args) -> anyhow::Result<()> {
    let shared = Arc::new(Mutex::new(Shared::default()));
    spawn_capture(args, shared.clone(), || {});
    let mut last_seq = 0;
    loop {
        std::thread::sleep(Duration::from_secs(1));
        let s = shared.lock().unwrap();
        match (&s.latest, &s.error) {
            (Some(f), _) if f.seq != last_seq => {
                println!(
                    "capture {:5.1} fps | convert {:4.1} ms | {}x{} | #{} | {}",
                    s.capture_fps, s.convert_ms, f.width, f.height, f.seq, s.source_desc
                );
                if let Some(p) = &s.perceived {
                    println!("  {}", perceived_text(p));
                }
                last_seq = f.seq;
            }
            (_, Some(e)) => println!("error: {e}"),
            _ => println!("no new frames"),
        }
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    let args = Args::parse();
    if args.headless {
        return run_headless(args);
    }

    let shared = Arc::new(Mutex::new(Shared::default()));
    let save_dir = args.save_dir.clone();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([480.0, 900.0]).with_title("CR viewer"),
        ..Default::default()
    };
    eframe::run_native(
        "cr-viewer",
        options,
        Box::new(move |cc| {
            let ctx = cc.egui_ctx.clone();
            // Repaint exactly when a frame arrives: no busy loop, no added latency.
            spawn_capture(args, shared.clone(), move || ctx.request_repaint());
            Ok(Box::new(ViewerApp {
                shared,
                texture: None,
                shown_seq: 0,
                frame_age_ms: 0.0,
                display_fps: FpsCounter::new(),
                last_frame: None,
                save_dir,
                status: String::new(),
            }))
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe: {e}"))
}
