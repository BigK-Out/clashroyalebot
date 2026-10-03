//! Screen calibration: where things are on the device screen.
//!
//! All coordinates are stored normalized (0..1 of screen width/height) so a calibration
//! survives resolution changes. Convert to pixels with the `*_px` helpers.

mod homography;

use std::path::Path;

use anyhow::Context;
use serde::{Deserialize, Serialize};

pub use homography::Homography;

/// Arena size in tiles (Clash Royale grid).
pub const ARENA_COLS: u32 = 18;
pub const ARENA_ROWS: u32 = 32;

/// Normalized point (0..1).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct NPoint {
    pub x: f64,
    pub y: f64,
}

/// Normalized axis-aligned rectangle (0..1).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct NRect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

/// Pixel rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PxRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl NPoint {
    pub fn to_px(self, width: u32, height: u32) -> (f64, f64) {
        (self.x * width as f64, self.y * height as f64)
    }

    pub fn from_px(x: f64, y: f64, width: u32, height: u32) -> Self {
        Self { x: x / width as f64, y: y / height as f64 }
    }
}

impl NRect {
    /// Rectangle spanning two normalized corners in any order.
    pub fn from_corners(a: NPoint, b: NPoint) -> Self {
        Self { x: a.x.min(b.x), y: a.y.min(b.y), w: (a.x - b.x).abs(), h: (a.y - b.y).abs() }
    }

    /// Pixel rect, clamped to the screen.
    pub fn to_px(self, width: u32, height: u32) -> PxRect {
        let x0 = (self.x * width as f64).round().clamp(0.0, width as f64) as u32;
        let y0 = (self.y * height as f64).round().clamp(0.0, height as f64) as u32;
        let x1 = ((self.x + self.w) * width as f64).round().clamp(0.0, width as f64) as u32;
        let y1 = ((self.y + self.h) * height as f64).round().clamp(0.0, height as f64) as u32;
        PxRect { x: x0, y: y0, w: x1 - x0, h: y1 - y0 }
    }

    pub fn center(self) -> NPoint {
        NPoint { x: self.x + self.w / 2.0, y: self.y + self.h / 2.0 }
    }
}

/// Outer corners of the playable 18x32 tile grid, as seen on screen.
/// The arena is drawn with slight perspective, so these form a trapezoid, not a rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct ArenaCorners {
    pub top_left: NPoint,
    pub top_right: NPoint,
    pub bottom_right: NPoint,
    pub bottom_left: NPoint,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Calibration {
    /// Screen size the calibration was made on (informational; coordinates are normalized).
    pub screen_width: u32,
    pub screen_height: u32,
    /// The purple elixir bar, from the 0 end to the 10 end.
    pub elixir_bar: NRect,
    /// The 4 playable card slots, left to right.
    pub card_slots: [NRect; 4],
    /// The small "next card" preview.
    pub next_card: NRect,
    pub arena: ArenaCorners,
}

impl Calibration {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parse {}", path.display()))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let path = path.as_ref();
        std::fs::write(path, toml::to_string_pretty(self)?).with_context(|| format!("write {}", path.display()))
    }

    /// Tile-space (cols, rows) → normalized screen mapping for the current arena corners.
    pub fn arena_mapping(&self) -> anyhow::Result<ArenaMapping> {
        ArenaMapping::new(&self.arena)
    }
}

/// Maps between arena tile coordinates and normalized screen coordinates.
///
/// Tile space: x in 0..18 (left→right), y in 0..32 (top = enemy side → bottom = my side).
/// Tile (tx, ty) covers [tx, tx+1) x [ty, ty+1); its center is (tx+0.5, ty+0.5).
#[derive(Debug, Clone)]
pub struct ArenaMapping {
    to_screen: Homography,
    to_tile: Homography,
}

impl ArenaMapping {
    pub fn new(c: &ArenaCorners) -> anyhow::Result<Self> {
        let tile = [
            (0.0, 0.0),
            (ARENA_COLS as f64, 0.0),
            (ARENA_COLS as f64, ARENA_ROWS as f64),
            (0.0, ARENA_ROWS as f64),
        ];
        let screen = [c.top_left, c.top_right, c.bottom_right, c.bottom_left].map(|p| (p.x, p.y));
        let to_screen = Homography::from_points(tile, screen).context("arena corners are degenerate")?;
        let to_tile = Homography::from_points(screen, tile).context("arena corners are degenerate")?;
        Ok(Self { to_screen, to_tile })
    }

    /// Continuous tile coordinates → normalized screen point.
    pub fn tile_to_screen(&self, tx: f64, ty: f64) -> NPoint {
        let (x, y) = self.to_screen.apply(tx, ty);
        NPoint { x, y }
    }

    /// Center of integer tile (col, row) → normalized screen point. Use this for taps.
    pub fn tile_center(&self, col: u32, row: u32) -> NPoint {
        self.tile_to_screen(col as f64 + 0.5, row as f64 + 0.5)
    }

    /// Normalized screen point → continuous tile coordinates (may be outside the arena).
    pub fn screen_to_tile(&self, p: NPoint) -> (f64, f64) {
        self.to_tile.apply(p.x, p.y)
    }

    /// Normalized screen point → integer tile, or None if outside the arena.
    pub fn screen_to_tile_index(&self, p: NPoint) -> Option<(u32, u32)> {
        let (tx, ty) = self.screen_to_tile(p);
        let inside = (0.0..ARENA_COLS as f64).contains(&tx) && (0.0..ARENA_ROWS as f64).contains(&ty);
        inside.then(|| (tx.floor() as u32, ty.floor() as u32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trapezoid() -> ArenaCorners {
        // Narrower at the top, like the in-game perspective.
        ArenaCorners {
            top_left: NPoint { x: 0.08, y: 0.10 },
            top_right: NPoint { x: 0.92, y: 0.10 },
            bottom_right: NPoint { x: 0.98, y: 0.78 },
            bottom_left: NPoint { x: 0.02, y: 0.78 },
        }
    }

    #[test]
    fn corners_map_exactly() {
        let m = ArenaMapping::new(&trapezoid()).unwrap();
        let c = trapezoid();
        for ((tx, ty), want) in [
            ((0.0, 0.0), c.top_left),
            ((18.0, 0.0), c.top_right),
            ((18.0, 32.0), c.bottom_right),
            ((0.0, 32.0), c.bottom_left),
        ] {
            let p = m.tile_to_screen(tx, ty);
            assert!((p.x - want.x).abs() < 1e-9 && (p.y - want.y).abs() < 1e-9, "{tx},{ty} -> {p:?}");
        }
    }

    #[test]
    fn every_tile_center_round_trips() {
        let m = ArenaMapping::new(&trapezoid()).unwrap();
        for row in 0..ARENA_ROWS {
            for col in 0..ARENA_COLS {
                let p = m.tile_center(col, row);
                assert_eq!(m.screen_to_tile_index(p), Some((col, row)), "tile ({col},{row}) via {p:?}");
            }
        }
    }

    #[test]
    fn outside_arena_is_none() {
        let m = ArenaMapping::new(&trapezoid()).unwrap();
        assert_eq!(m.screen_to_tile_index(NPoint { x: 0.5, y: 0.05 }), None); // above
        assert_eq!(m.screen_to_tile_index(NPoint { x: 0.5, y: 0.90 }), None); // card bar
        assert_eq!(m.screen_to_tile_index(NPoint { x: 0.03, y: 0.11 }), None); // left of the narrow top edge
    }

    #[test]
    fn rectangle_arena_is_linear() {
        let m = ArenaMapping::new(&ArenaCorners {
            top_left: NPoint { x: 0.0, y: 0.0 },
            top_right: NPoint { x: 0.9, y: 0.0 },
            bottom_right: NPoint { x: 0.9, y: 0.8 },
            bottom_left: NPoint { x: 0.0, y: 0.8 },
        })
        .unwrap();
        let p = m.tile_center(0, 0);
        assert!((p.x - 0.025).abs() < 1e-9 && (p.y - 0.0125).abs() < 1e-9);
        let p = m.tile_to_screen(9.0, 16.0);
        assert!((p.x - 0.45).abs() < 1e-9 && (p.y - 0.4).abs() < 1e-9);
    }

    #[test]
    fn degenerate_corners_error() {
        let p = NPoint { x: 0.5, y: 0.5 };
        let c = ArenaCorners { top_left: p, top_right: p, bottom_right: p, bottom_left: p };
        assert!(ArenaMapping::new(&c).is_err());
    }

    #[test]
    fn rect_to_px_and_from_corners() {
        let r = NRect::from_corners(NPoint { x: 0.5, y: 0.75 }, NPoint { x: 0.25, y: 0.5 });
        assert_eq!(r, NRect { x: 0.25, y: 0.5, w: 0.25, h: 0.25 });
        assert_eq!(r.to_px(720, 1280), PxRect { x: 180, y: 640, w: 180, h: 320 });
        let wide = NRect { x: 0.9, y: 0.0, w: 0.5, h: 1.0 };
        assert_eq!(wide.to_px(100, 100), PxRect { x: 90, y: 0, w: 10, h: 100 });
    }

    #[test]
    fn toml_round_trip() {
        let mut c = Calibration { screen_width: 720, screen_height: 1280, arena: trapezoid(), ..Default::default() };
        c.card_slots[2] = NRect { x: 0.5, y: 0.85, w: 0.18, h: 0.12 };
        let path = std::env::temp_dir().join(format!("calib-{}.toml", std::process::id()));
        c.save(&path).unwrap();
        assert_eq!(Calibration::load(&path).unwrap(), c);
        std::fs::remove_file(path).unwrap();
    }
}

#[cfg(test)]
mod repo_calibration {
    /// The checked-in calibration must parse and give a usable arena mapping.
    #[test]
    fn repo_calibration_loads() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../calibration.toml");
        let c = super::Calibration::load(path).unwrap();
        let m = c.arena_mapping().unwrap();
        let p = m.tile_center(0, 0);
        assert!(p.x > 0.0 && p.x < 0.1 && p.y > 0.1 && p.y < 0.2, "{p:?}");
    }
}
