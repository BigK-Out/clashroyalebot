//! Calibration tool: mark the elixir bar, card slots and arena corners on a frame,
//! check the tile grid overlay, save `calibration.toml`.
//!
//!   calibrate frames/                 # saved screenshots, ←/→ to switch
//!   calibrate --live                  # /dev/video10, Space to freeze
//!   calibrate frames/ -o other.toml   # custom output path

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use calib::{ARENA_COLS, ARENA_ROWS, Calibration, NPoint, NRect};
use capture::{Frame, FrameSource, V4lSource};
use clap::Parser;
use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};

#[derive(Parser)]
struct Args {
    /// Image file or directory of PNG/JPEG frames.
    input: Option<PathBuf>,
    /// Use the live capture device instead of files.
    #[arg(long)]
    live: bool,
    #[arg(long, default_value = "/dev/video10")]
    device: PathBuf,
    /// Calibration file to load (if present) and save.
    #[arg(short, long, default_value = "calibration.toml")]
    output: PathBuf,
}

#[derive(Clone, Copy, PartialEq)]
enum Target {
    ElixirBar,
    Card(usize),
    NextCard,
    TopLeft,
    TopRight,
    BottomRight,
    BottomLeft,
}

const TARGETS: [Target; 10] = [
    Target::ElixirBar,
    Target::Card(0),
    Target::Card(1),
    Target::Card(2),
    Target::Card(3),
    Target::NextCard,
    Target::TopLeft,
    Target::TopRight,
    Target::BottomRight,
    Target::BottomLeft,
];

impl Target {
    fn label(self) -> String {
        match self {
            Target::ElixirBar => "Elixir bar".into(),
            Target::Card(i) => format!("Card slot {}", i + 1),
            Target::NextCard => "Next card".into(),
            Target::TopLeft => "Arena top-left".into(),
            Target::TopRight => "Arena top-right".into(),
            Target::BottomRight => "Arena bottom-right".into(),
            Target::BottomLeft => "Arena bottom-left".into(),
        }
    }

    fn hint(self) -> &'static str {
        match self {
            Target::ElixirBar => "Drag over the purple bar, from the 0 end (left edge of the bar) to the 10 end. \
                                  Exclude the drop icon and number.",
            Target::Card(_) => "Drag tightly around the card art (including its elixir-cost badge).",
            Target::NextCard => "Drag around the small 'Next' card preview.",
            Target::TopLeft | Target::TopRight | Target::BottomRight | Target::BottomLeft => {
                "Click the outer corner of the tile grid (the checkered field). \
                 Top = enemy side, bottom = your side. The grid overlay should line up with the checkerboard."
            }
        }
    }

    fn is_point(self) -> bool {
        matches!(self, Target::TopLeft | Target::TopRight | Target::BottomRight | Target::BottomLeft)
    }

    fn set_rect(self, c: &mut Calibration, r: NRect) {
        match self {
            Target::ElixirBar => c.elixir_bar = r,
            Target::Card(i) => c.card_slots[i] = r,
            Target::NextCard => c.next_card = r,
            _ => {}
        }
    }

    fn set_point(self, c: &mut Calibration, p: NPoint) {
        match self {
            Target::TopLeft => c.arena.top_left = p,
            Target::TopRight => c.arena.top_right = p,
            Target::BottomRight => c.arena.bottom_right = p,
            Target::BottomLeft => c.arena.bottom_left = p,
            _ => {}
        }
    }
}

enum Source {
    Files { files: Vec<PathBuf>, idx: usize },
    Live { latest: Arc<Mutex<Option<Frame>>>, frozen: bool },
}

struct App {
    source: Source,
    frame: Option<Frame>,
    texture: Option<egui::TextureHandle>,
    calib: Calibration,
    done: [bool; TARGETS.len()],
    target: usize,
    drag_start: Option<NPoint>,
    output: PathBuf,
    status: String,
}

fn list_images(input: &Path) -> anyhow::Result<Vec<PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(input)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg"))
        })
        .collect();
    files.sort();
    anyhow::ensure!(!files.is_empty(), "no images in {}", input.display());
    Ok(files)
}

impl App {
    fn load_current_file(&mut self) {
        if let Source::Files { files, idx } = &self.source {
            match capture::load_rgb(&files[*idx]) {
                Ok(f) => {
                    self.status = format!("{} ({}/{})", files[*idx].display(), idx + 1, files.len());
                    self.frame = Some(f);
                    self.texture = None;
                }
                Err(e) => self.status = format!("load failed: {e:#}"),
            }
        }
    }

    fn save(&mut self) {
        if let Some(f) = &self.frame {
            self.calib.screen_width = f.width;
            self.calib.screen_height = f.height;
        }
        let missing: Vec<String> =
            TARGETS.iter().zip(self.done).filter(|(_, d)| !d).map(|(t, _)| t.label()).collect();
        self.status = match self.calib.save(&self.output) {
            Ok(()) if missing.is_empty() => format!("saved {}", self.output.display()),
            Ok(()) => format!("saved {} (still unset: {})", self.output.display(), missing.join(", ")),
            Err(e) => format!("save failed: {e:#}"),
        };
    }

    fn advance(&mut self) {
        self.done[self.target] = true;
        // Next unset target, else stay.
        if let Some(next) = (1..=TARGETS.len()).map(|k| (self.target + k) % TARGETS.len()).find(|&i| !self.done[i]) {
            self.target = next;
        }
    }

    fn side_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Calibrate");
        ui.label(format!("→ {}", self.output.display()));
        ui.separator();
        for (i, t) in TARGETS.iter().enumerate() {
            let mark = if self.done[i] { "✔" } else { "  " };
            if ui.selectable_label(self.target == i, format!("{mark} {}", t.label())).clicked() {
                self.target = i;
            }
        }
        ui.separator();
        ui.label(egui::RichText::new(TARGETS[self.target].hint()).italics());
        ui.separator();
        if ui.button("Save  (Ctrl+S)").clicked() {
            self.save();
        }
        ui.separator();
        match &self.source {
            Source::Files { .. } => ui.label("←/→ previous/next frame"),
            Source::Live { frozen, .. } => ui.label(if *frozen { "FROZEN — Space to resume" } else { "live — Space to freeze" }),
        };
        ui.label("Mouse over the arena shows the tile.");
        ui.separator();
        ui.small(&self.status);
    }

    fn handle_keys(&mut self, ctx: &egui::Context) {
        let (left, right, space, save) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::ArrowLeft),
                i.key_pressed(egui::Key::ArrowRight),
                i.key_pressed(egui::Key::Space),
                i.modifiers.command && i.key_pressed(egui::Key::S),
            )
        });
        if save {
            self.save();
        }
        match &mut self.source {
            Source::Files { files, idx } => {
                let n = files.len();
                let new = if left { (*idx + n - 1) % n } else if right { (*idx + 1) % n } else { *idx };
                if new != *idx {
                    *idx = new;
                    self.load_current_file();
                }
            }
            Source::Live { latest, frozen } => {
                if space {
                    *frozen = !*frozen;
                }
                if !*frozen && let Some(f) = latest.lock().unwrap().take() {
                    self.frame = Some(f);
                    self.texture = None;
                }
            }
        }
    }

    fn image_panel(&mut self, ui: &mut egui::Ui) {
        let Some(frame) = &self.frame else {
            ui.centered_and_justified(|ui| ui.label("waiting for a frame…"));
            return;
        };
        let (fw, fh) = (frame.width, frame.height);
        let tex = self.texture.get_or_insert_with(|| {
            let img = egui::ColorImage::from_rgb([fw as usize, fh as usize], &frame.rgb);
            ui.ctx().load_texture("frame", img, egui::TextureOptions::LINEAR)
        });
        let (tex_id, size) = (tex.id(), tex.size_vec2());

        // Fit image to the panel.
        let avail = ui.available_rect_before_wrap();
        let scale = (avail.width() / size.x).min(avail.height() / size.y);
        let img_rect = Rect::from_min_size(avail.min, size * scale);
        let resp = ui.allocate_rect(img_rect, Sense::click_and_drag());
        let painter = ui.painter_at(img_rect);
        let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
        painter.image(tex_id, img_rect, uv, Color32::WHITE);

        let to_screen = |p: NPoint| img_rect.min + Vec2::new(p.x as f32 * img_rect.width(), p.y as f32 * img_rect.height());
        let to_norm = |p: Pos2| NPoint {
            x: ((p.x - img_rect.min.x) / img_rect.width()).clamp(0.0, 1.0) as f64,
            y: ((p.y - img_rect.min.y) / img_rect.height()).clamp(0.0, 1.0) as f64,
        };
        let nrect_screen = |r: NRect| {
            Rect::from_min_max(to_screen(NPoint { x: r.x, y: r.y }), to_screen(NPoint { x: r.x + r.w, y: r.y + r.h }))
        };

        // Interaction.
        let target = TARGETS[self.target];
        if target.is_point() {
            if resp.clicked() && let Some(p) = resp.interact_pointer_pos() {
                target.set_point(&mut self.calib, to_norm(p));
                self.advance();
            }
        } else {
            if resp.drag_started() {
                self.drag_start = resp.interact_pointer_pos().map(to_norm);
            }
            if let (Some(start), Some(now)) = (self.drag_start, resp.interact_pointer_pos()) {
                let r = NRect::from_corners(start, to_norm(now));
                painter.rect_stroke(nrect_screen(r), 0.0, Stroke::new(2.0, Color32::WHITE), StrokeKind::Inside);
                if resp.drag_stopped() {
                    if r.w > 0.002 && r.h > 0.002 {
                        target.set_rect(&mut self.calib, r);
                        self.advance();
                    }
                    self.drag_start = None;
                }
            }
        }

        // Overlays: regions.
        let font = egui::FontId::proportional(13.0);
        let draw_rect = |i: usize, r: NRect, color: Color32| {
            if !self.done[i] {
                return;
            }
            let width = if self.target == i { 3.0 } else { 1.5 };
            let sr = nrect_screen(r);
            painter.rect_stroke(sr, 2.0, Stroke::new(width, color), StrokeKind::Outside);
            painter.text(sr.left_top() - Vec2::new(0.0, 2.0), egui::Align2::LEFT_BOTTOM, TARGETS[i].label(), font.clone(), color);
        };
        draw_rect(0, self.calib.elixir_bar, Color32::from_rgb(230, 80, 255));
        for k in 0..4 {
            draw_rect(1 + k, self.calib.card_slots[k], Color32::from_rgb(80, 220, 255));
        }
        draw_rect(5, self.calib.next_card, Color32::from_rgb(255, 210, 80));

        // Overlays: arena grid once all 4 corners exist.
        let corners_done = self.done[6..10].iter().all(|d| *d);
        let a = &self.calib.arena;
        for (i, p) in [(6, a.top_left), (7, a.top_right), (8, a.bottom_right), (9, a.bottom_left)] {
            if self.done[i] {
                painter.circle_filled(to_screen(p), if self.target == i { 6.0 } else { 4.0 }, Color32::YELLOW);
            }
        }
        let mapping = corners_done.then(|| self.calib.arena_mapping().ok()).flatten();
        if let Some(m) = &mapping {
            let grid = Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 0, 110));
            let border = Stroke::new(2.0, Color32::YELLOW);
            let river = Stroke::new(2.0, Color32::from_rgb(0, 200, 255));
            for col in 0..=ARENA_COLS {
                let s = if col == 0 || col == ARENA_COLS { border } else { grid };
                painter.line_segment(
                    [to_screen(m.tile_to_screen(col as f64, 0.0)), to_screen(m.tile_to_screen(col as f64, ARENA_ROWS as f64))],
                    s,
                );
            }
            for row in 0..=ARENA_ROWS {
                let s = match row {
                    0 | ARENA_ROWS => border,
                    15 | 17 => river, // river occupies rows 15–16
                    _ => grid,
                };
                painter.line_segment(
                    [to_screen(m.tile_to_screen(0.0, row as f64)), to_screen(m.tile_to_screen(ARENA_COLS as f64, row as f64))],
                    s,
                );
            }
        } else if corners_done {
            self.status = "arena corners are degenerate".into();
        }

        // Hover: tile readout + magnifier for precise clicks.
        if let Some(hp) = resp.hover_pos() {
            let n = to_norm(hp);
            let tile = mapping.as_ref().and_then(|m| m.screen_to_tile_index(n));
            let px = (n.x * fw as f64, n.y * fh as f64);
            let text = match tile {
                Some((c, r)) => format!("px ({:.0},{:.0})  tile ({c},{r})", px.0, px.1),
                None => format!("px ({:.0},{:.0})", px.0, px.1),
            };
            painter.text(img_rect.left_bottom() + Vec2::new(6.0, -6.0), egui::Align2::LEFT_BOTTOM, text, font.clone(), Color32::WHITE);

            const ZOOM: f32 = 4.0;
            const LOUPE: f32 = 160.0;
            let half_uv = Vec2::new(LOUPE / ZOOM / 2.0 / img_rect.width(), LOUPE / ZOOM / 2.0 / img_rect.height());
            let c = Pos2::new(n.x as f32, n.y as f32);
            let loupe_uv = Rect::from_min_max(c - half_uv, c + half_uv);
            // Put the loupe in the corner away from the cursor.
            let right = hp.x < img_rect.center().x;
            let origin = if right {
                img_rect.right_top() + Vec2::new(-LOUPE - 8.0, 8.0)
            } else {
                img_rect.left_top() + Vec2::new(8.0, 8.0)
            };
            let loupe = Rect::from_min_size(origin, Vec2::splat(LOUPE));
            painter.image(tex_id, loupe, loupe_uv, Color32::WHITE);
            painter.rect_stroke(loupe, 0.0, Stroke::new(2.0, Color32::WHITE), StrokeKind::Outside);
            let lc = loupe.center();
            painter.line_segment([lc - Vec2::new(10.0, 0.0), lc + Vec2::new(10.0, 0.0)], Stroke::new(1.0, Color32::RED));
            painter.line_segment([lc - Vec2::new(0.0, 10.0), lc + Vec2::new(0.0, 10.0)], Stroke::new(1.0, Color32::RED));
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.handle_keys(ui.ctx());
        egui::Panel::left("targets").exact_size(230.0).show(ui, |ui| self.side_panel(ui));
        egui::CentralPanel::default().show(ui, |ui| self.image_panel(ui));
        if matches!(self.source, Source::Live { frozen: false, .. }) {
            ui.ctx().request_repaint();
        }
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    let args = Args::parse();

    let source = if args.live {
        let latest = Arc::new(Mutex::new(None));
        let slot = latest.clone();
        let device = args.device.clone();
        std::thread::spawn(move || {
            loop {
                match V4lSource::open(&device) {
                    Ok(mut src) => {
                        while let Ok(f) = src.next_frame() {
                            *slot.lock().unwrap() = Some(f);
                        }
                    }
                    Err(e) => tracing::warn!("{e:#}"),
                }
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        });
        Source::Live { latest, frozen: false }
    } else {
        let input = args.input.clone().unwrap_or_else(|| PathBuf::from("frames"));
        Source::Files { files: list_images(&input)?, idx: 0 }
    };

    // Resume from an existing calibration: every target counts as set.
    let (calib, done) = match Calibration::load(&args.output) {
        Ok(c) => (c, [true; TARGETS.len()]),
        Err(_) => (Calibration::default(), [false; TARGETS.len()]),
    };

    let mut app = App {
        source,
        frame: None,
        texture: None,
        calib,
        done,
        target: 0,
        drag_start: None,
        output: args.output,
        status: String::new(),
    };
    app.load_current_file();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([760.0, 1000.0]).with_title("CR calibrate"),
        ..Default::default()
    };
    eframe::run_native("cr-calibrate", options, Box::new(|_| Ok(Box::new(app)))).map_err(|e| anyhow::anyhow!("eframe: {e}"))
}
