//! Bounding-box labeler for the detector dataset (YOLO format).
//!
//!   label                       # dataset/raw images, labels in dataset/labels
//!   label --stride 5            # every 5th frame (recordings are near-duplicates at 2/s)
//!   label --dataset other/
//!
//! Drag = new box (current class) · click = select · 1-9 = class (selected box, or for new
//! boxes) · Delete/Backspace = remove selected · ←/→ = prev/next (autosave) · N = next
//! unlabeled · Ctrl+S = save · Esc = deselect.
//! A saved empty label file means "checked, no objects". Pre-labels from
//! <dataset>/predictions/<stem>.txt are loaded for images without labels.

use std::path::{Path, PathBuf};

use clap::Parser;
use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "dataset")]
    dataset: PathBuf,
    /// Only show every Nth image (consecutive recordings are 0.5 s apart, near-duplicates).
    #[arg(long, default_value_t = 1)]
    stride: usize,
}

const DEFAULT_CLASSES: [&str; 6] =
    ["enemy_troop", "ally_troop", "enemy_building", "ally_building", "enemy_tower", "ally_tower"];

/// Box in normalized image coordinates (YOLO: center + size).
#[derive(Clone, Copy, Debug, PartialEq)]
struct BBox {
    class: usize,
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
}

impl BBox {
    fn from_corners(class: usize, a: Pos2, b: Pos2) -> Self {
        let (x0, x1) = (a.x.min(b.x), a.x.max(b.x));
        let (y0, y1) = (a.y.min(b.y), a.y.max(b.y));
        Self { class, cx: (x0 + x1) / 2.0, cy: (y0 + y1) / 2.0, w: x1 - x0, h: y1 - y0 }
    }

    fn rect(&self) -> Rect {
        Rect::from_center_size(Pos2::new(self.cx, self.cy), Vec2::new(self.w, self.h))
    }
}

fn parse_yolo(text: &str) -> Vec<BBox> {
    text.lines()
        .filter_map(|l| {
            let v: Vec<&str> = l.split_whitespace().collect();
            if v.len() < 5 {
                return None;
            }
            Some(BBox {
                class: v[0].parse().ok()?,
                cx: v[1].parse().ok()?,
                cy: v[2].parse().ok()?,
                w: v[3].parse().ok()?,
                h: v[4].parse().ok()?,
            })
        })
        .collect()
}

fn format_yolo(boxes: &[BBox]) -> String {
    boxes.iter().map(|b| format!("{} {:.6} {:.6} {:.6} {:.6}\n", b.class, b.cx, b.cy, b.w, b.h)).collect()
}

fn class_color(class: usize, name: &str) -> Color32 {
    let base = if name.starts_with("enemy") {
        [Color32::from_rgb(255, 70, 70), Color32::from_rgb(255, 150, 40), Color32::from_rgb(255, 60, 200)]
    } else {
        [Color32::from_rgb(60, 170, 255), Color32::from_rgb(60, 230, 160), Color32::from_rgb(160, 120, 255)]
    };
    base[(class / 2) % 3]
}

struct App {
    dataset: PathBuf,
    classes: Vec<String>,
    images: Vec<PathBuf>,
    idx: usize,
    boxes: Vec<BBox>,
    /// Loaded from predictions, not yet saved.
    from_predictions: bool,
    dirty: bool,
    selected: Option<usize>,
    current_class: usize,
    drag_start: Option<Pos2>,
    texture: Option<egui::TextureHandle>,
    image_size: Vec2,
    status: String,
}

impl App {
    fn stem(&self, i: usize) -> String {
        self.images[i].file_stem().and_then(|s| s.to_str()).unwrap_or_default().to_string()
    }

    fn label_path(&self, i: usize) -> PathBuf {
        self.dataset.join("labels").join(format!("{}.txt", self.stem(i)))
    }

    fn labeled_count(&self) -> usize {
        (0..self.images.len()).filter(|&i| self.label_path(i).exists()).count()
    }

    fn load(&mut self, i: usize) {
        self.idx = i;
        self.selected = None;
        self.drag_start = None;
        self.texture = None;
        self.dirty = false;
        let label = self.label_path(i);
        let pred = self.dataset.join("predictions").join(format!("{}.txt", self.stem(i)));
        (self.boxes, self.from_predictions) = if let Ok(t) = std::fs::read_to_string(&label) {
            (parse_yolo(&t), false)
        } else if let Ok(t) = std::fs::read_to_string(&pred) {
            (parse_yolo(&t), true)
        } else {
            (Vec::new(), false)
        };
    }

    fn save(&mut self) {
        let path = self.label_path(self.idx);
        let res = std::fs::create_dir_all(path.parent().unwrap()).and_then(|_| std::fs::write(&path, format_yolo(&self.boxes)));
        self.status = match res {
            Ok(()) => format!("saved {} boxes", self.boxes.len()),
            Err(e) => format!("save failed: {e}"),
        };
        self.dirty = false;
        self.from_predictions = false;
    }

    fn go(&mut self, i: usize) {
        // Moving on counts as "reviewed": save even unchanged or empty images.
        self.save();
        self.load(i);
    }

    fn next_unlabeled(&self) -> Option<usize> {
        (1..=self.images.len()).map(|k| (self.idx + k) % self.images.len()).find(|&i| !self.label_path(i).exists())
    }

    fn keys(&mut self, ctx: &egui::Context) {
        let input = ctx.input(|i| i.clone());
        let n = self.images.len();
        if input.key_pressed(egui::Key::ArrowRight) || input.key_pressed(egui::Key::D) {
            self.go((self.idx + 1) % n);
        }
        if input.key_pressed(egui::Key::ArrowLeft) || input.key_pressed(egui::Key::A) {
            self.go((self.idx + n - 1) % n);
        }
        if input.key_pressed(egui::Key::N) {
            match self.next_unlabeled() {
                Some(i) => self.go(i),
                None => self.status = "all images labeled".into(),
            }
        }
        if input.modifiers.command && input.key_pressed(egui::Key::S) {
            self.save();
        }
        if input.key_pressed(egui::Key::Escape) {
            self.selected = None;
        }
        if (input.key_pressed(egui::Key::Delete) || input.key_pressed(egui::Key::Backspace))
            && let Some(s) = self.selected.take()
        {
            self.boxes.remove(s);
            self.dirty = true;
        }
        let digits = [
            egui::Key::Num1,
            egui::Key::Num2,
            egui::Key::Num3,
            egui::Key::Num4,
            egui::Key::Num5,
            egui::Key::Num6,
            egui::Key::Num7,
            egui::Key::Num8,
            egui::Key::Num9,
        ];
        for (k, key) in digits.iter().enumerate().take(self.classes.len()) {
            if input.key_pressed(*key) {
                self.current_class = k;
                if let Some(s) = self.selected {
                    self.boxes[s].class = k;
                    self.dirty = true;
                }
            }
        }
    }

    fn side_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Label");
        ui.label(format!("{} / {}  ({} labeled)", self.idx + 1, self.images.len(), self.labeled_count()));
        ui.small(self.images[self.idx].file_name().and_then(|s| s.to_str()).unwrap_or_default());
        if self.from_predictions {
            ui.colored_label(Color32::YELLOW, "pre-labeled by model: fix, then move on");
        } else if self.label_path(self.idx).exists() {
            ui.colored_label(Color32::LIGHT_GREEN, "labeled");
        }
        ui.separator();
        for (i, c) in self.classes.clone().iter().enumerate() {
            let text = egui::RichText::new(format!("{}  {c}", i + 1)).color(class_color(i, c));
            if ui.selectable_label(self.current_class == i, text).clicked() {
                self.current_class = i;
                if let Some(s) = self.selected {
                    self.boxes[s].class = i;
                    self.dirty = true;
                }
            }
        }
        ui.separator();
        ui.small(
            "drag: new box\nclick: select\n1-9: class\nDel: delete box\nLeft/Right or A/D: prev/next (saves)\n\
             N: next unlabeled\nEsc: deselect\nCtrl+S: save",
        );
        ui.separator();
        ui.label(format!("{} boxes{}", self.boxes.len(), if self.dirty { " (unsaved)" } else { "" }));
        ui.small(&self.status);
    }

    fn image_panel(&mut self, ui: &mut egui::Ui) {
        if self.texture.is_none() {
            match capture::load_rgb(&self.images[self.idx]) {
                Ok(f) => {
                    let img = egui::ColorImage::from_rgb([f.width as usize, f.height as usize], &f.rgb);
                    self.image_size = Vec2::new(f.width as f32, f.height as f32);
                    self.texture = Some(ui.ctx().load_texture("img", img, egui::TextureOptions::LINEAR));
                }
                Err(e) => {
                    self.status = format!("load failed: {e:#}");
                    return;
                }
            }
        }
        let tex_id = self.texture.as_ref().unwrap().id();
        let avail = ui.available_rect_before_wrap();
        let scale = (avail.width() / self.image_size.x).min(avail.height() / self.image_size.y);
        let img_rect = Rect::from_min_size(avail.min, self.image_size * scale);
        let resp = ui.allocate_rect(img_rect, Sense::click_and_drag());
        let painter = ui.painter_at(img_rect);
        painter.image(tex_id, img_rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);

        let to_norm = |p: Pos2| {
            Pos2::new(
                ((p.x - img_rect.min.x) / img_rect.width()).clamp(0.0, 1.0),
                ((p.y - img_rect.min.y) / img_rect.height()).clamp(0.0, 1.0),
            )
        };
        let to_screen = |r: Rect| {
            Rect::from_min_max(
                img_rect.min + r.min.to_vec2() * img_rect.size(),
                img_rect.min + r.max.to_vec2() * img_rect.size(),
            )
        };

        if resp.clicked()
            && let Some(p) = resp.interact_pointer_pos().map(to_norm)
        {
            // Smallest box under the cursor wins (nested boxes).
            self.selected = self
                .boxes
                .iter()
                .enumerate()
                .filter(|(_, b)| b.rect().contains(p))
                .min_by(|a, b| (a.1.w * a.1.h).total_cmp(&(b.1.w * b.1.h)))
                .map(|(i, _)| i);
        }
        if resp.drag_started() {
            self.drag_start = resp.interact_pointer_pos().map(to_norm);
        }
        if let (Some(start), Some(now)) = (self.drag_start, resp.interact_pointer_pos().map(to_norm)) {
            let b = BBox::from_corners(self.current_class, start, now);
            painter.rect_stroke(to_screen(b.rect()), 0.0, Stroke::new(2.0, Color32::WHITE), StrokeKind::Inside);
            if resp.drag_stopped() {
                // Ignore accidental micro-drags (< ~4 px on screen).
                if b.w * img_rect.width() > 4.0 && b.h * img_rect.height() > 4.0 {
                    self.boxes.push(b);
                    self.selected = Some(self.boxes.len() - 1);
                    self.dirty = true;
                }
                self.drag_start = None;
            }
        }

        let font = egui::FontId::proportional(11.0);
        for (i, b) in self.boxes.iter().enumerate() {
            let name = self.classes.get(b.class).map(String::as_str).unwrap_or("?");
            let color = class_color(b.class, name);
            let width = if self.selected == Some(i) { 3.0 } else { 1.5 };
            let r = to_screen(b.rect());
            painter.rect_stroke(r, 0.0, Stroke::new(width, color), StrokeKind::Outside);
            painter.text(r.left_top(), egui::Align2::LEFT_BOTTOM, name, font.clone(), color);
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.keys(ui.ctx());
        egui::Panel::left("side").exact_size(220.0).show(ui, |ui| self.side_panel(ui));
        egui::CentralPanel::default().show(ui, |ui| self.image_panel(ui));
    }
}

fn load_classes(dataset: &Path) -> anyhow::Result<Vec<String>> {
    let path = dataset.join("classes.txt");
    if !path.exists() {
        std::fs::create_dir_all(dataset)?;
        std::fs::write(&path, DEFAULT_CLASSES.join("\n") + "\n")?;
    }
    Ok(std::fs::read_to_string(path)?.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from).collect())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let classes = load_classes(&args.dataset)?;
    let raw = args.dataset.join("raw");
    let mut images: Vec<PathBuf> = std::fs::read_dir(&raw)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", raw.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "jpg" || e == "png"))
        .collect();
    images.sort();
    let images: Vec<PathBuf> = images.into_iter().step_by(args.stride.max(1)).collect();
    anyhow::ensure!(!images.is_empty(), "no images in {}", raw.display());

    let mut app = App {
        dataset: args.dataset,
        classes,
        images,
        idx: 0,
        boxes: Vec::new(),
        from_predictions: false,
        dirty: false,
        selected: None,
        current_class: 0,
        drag_start: None,
        texture: None,
        image_size: Vec2::ONE,
        status: String::new(),
    };
    let start = (0..app.images.len()).find(|&i| !app.label_path(i).exists()).unwrap_or(0);
    app.load(start);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([900.0, 1000.0]).with_title("CR label"),
        ..Default::default()
    };
    eframe::run_native("cr-label", options, Box::new(|_| Ok(Box::new(app)))).map_err(|e| anyhow::anyhow!("eframe: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yolo_round_trip() {
        let b = vec![BBox { class: 2, cx: 0.5, cy: 0.25, w: 0.1, h: 0.2 }];
        assert_eq!(parse_yolo(&format_yolo(&b)), b);
        assert_eq!(parse_yolo("garbage\n1 0.1\n"), vec![]);
    }

    #[test]
    fn corners_in_any_order() {
        let b = BBox::from_corners(0, Pos2::new(0.6, 0.4), Pos2::new(0.2, 0.8));
        assert!((b.cx - 0.4).abs() < 1e-6 && (b.cy - 0.6).abs() < 1e-6 && (b.w - 0.4).abs() < 1e-6);
    }
}
