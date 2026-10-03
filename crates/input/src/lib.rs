//! Touch input: tap backends and card deployment (slot tap → tile tap).

mod adb;

use std::time::{Duration, Instant};

use anyhow::ensure;
use calib::{ARENA_COLS, ARENA_ROWS, ArenaMapping, Calibration, NPoint};

pub use adb::AdbShell;

/// Something that can tap the device screen in device pixels.
pub trait TapBackend {
    /// Device screen size in pixels (portrait).
    fn screen_size(&self) -> (u32, u32);

    /// Taps each point in order; returns when the device has executed all of them.
    fn taps(&mut self, points: &[(u32, u32)]) -> anyhow::Result<()>;
}

/// First row of my half (rows 17..32 are mine; 15–16 are the river).
pub const MY_FIRST_ROW: u32 = 17;

#[derive(Debug, Clone, Copy)]
pub struct DeployTiming {
    /// Time from the call until the device finished both taps.
    pub taps: Duration,
}

/// Places cards: tap the card slot, then the arena tile.
pub struct Deployer<B: TapBackend> {
    backend: B,
    calib: Calibration,
    mapping: ArenaMapping,
}

impl<B: TapBackend> Deployer<B> {
    pub fn new(backend: B, calib: Calibration) -> anyhow::Result<Self> {
        let mapping = calib.arena_mapping()?;
        Ok(Self { backend, calib, mapping })
    }

    pub fn backend(&mut self) -> &mut B {
        &mut self.backend
    }

    fn to_device(&self, p: NPoint) -> (u32, u32) {
        let (w, h) = self.backend.screen_size();
        ((p.x * w as f64).round() as u32, (p.y * h as f64).round() as u32)
    }

    /// Device pixel of the center of card slot `slot` (0..4).
    pub fn slot_point(&self, slot: usize) -> (u32, u32) {
        self.to_device(self.calib.card_slots[slot].center())
    }

    /// Device pixel of the center of tile (col, row).
    pub fn tile_point(&self, col: u32, row: u32) -> (u32, u32) {
        self.to_device(self.mapping.tile_center(col, row))
    }

    /// Deploys the card in `slot` (0..4) onto tile (col, row). Only my half is allowed unless
    /// `allow_enemy_half` (spells, or after a princess tower falls).
    pub fn deploy(&mut self, slot: usize, col: u32, row: u32, allow_enemy_half: bool) -> anyhow::Result<DeployTiming> {
        ensure!(slot < 4, "slot must be 0..4, got {slot}");
        ensure!(col < ARENA_COLS && row < ARENA_ROWS, "tile ({col},{row}) is outside the 18x32 arena");
        ensure!(allow_enemy_half || row >= MY_FIRST_ROW, "tile ({col},{row}) is not on my half (rows 17..31)");
        let t0 = Instant::now();
        let points = [self.slot_point(slot), self.tile_point(col, row)];
        self.backend.taps(&points)?;
        Ok(DeployTiming { taps: t0.elapsed() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records taps instead of sending them.
    struct Recorder {
        size: (u32, u32),
        taps: Vec<(u32, u32)>,
    }

    impl TapBackend for Recorder {
        fn screen_size(&self) -> (u32, u32) {
            self.size
        }
        fn taps(&mut self, points: &[(u32, u32)]) -> anyhow::Result<()> {
            self.taps.extend_from_slice(points);
            Ok(())
        }
    }

    fn repo_calib() -> Calibration {
        Calibration::load(concat!(env!("CARGO_MANIFEST_DIR"), "/../../calibration.toml")).unwrap()
    }

    #[test]
    fn deploy_taps_slot_then_tile_in_device_pixels() {
        let mut d = Deployer::new(Recorder { size: (1080, 2400), taps: vec![] }, repo_calib()).unwrap();
        d.deploy(2, 3, 17, false).unwrap();
        let taps = &d.backend().taps;
        assert_eq!(taps.len(), 2);
        // Slot 3 is centered around x≈742, y≈2191 on the 1080x2400 phone.
        assert!((taps[0].0 as i32 - 742).abs() <= 2 && (taps[0].1 as i32 - 2191).abs() <= 2, "{taps:?}");
        // Tile (3,17): x = 24 + 3.5*57.33 ≈ 225, y = 303.5 + 17.5*45.88 ≈ 1106.
        assert!((taps[1].0 as i32 - 225).abs() <= 2 && (taps[1].1 as i32 - 1106).abs() <= 2, "{taps:?}");
    }

    #[test]
    fn scales_to_other_resolutions() {
        let mut d = Deployer::new(Recorder { size: (540, 1200), taps: vec![] }, repo_calib()).unwrap();
        d.deploy(2, 3, 17, false).unwrap();
        let t = d.backend().taps[1];
        assert!((t.0 as i32 - 112).abs() <= 2 && (t.1 as i32 - 553).abs() <= 2, "{t:?}");
    }

    #[test]
    fn rejects_bad_targets() {
        let mut d = Deployer::new(Recorder { size: (1080, 2400), taps: vec![] }, repo_calib()).unwrap();
        assert!(d.deploy(4, 3, 20, false).is_err());
        assert!(d.deploy(0, 18, 20, false).is_err());
        assert!(d.deploy(0, 3, 32, false).is_err());
        assert!(d.deploy(0, 3, 10, false).is_err(), "enemy half needs allow_enemy_half");
        assert!(d.deploy(0, 3, 10, true).is_ok());
        assert_eq!(d.backend().taps.len(), 2);
    }
}
