//! Princess tower HP from the HP bars (red under enemy towers, blue under mine): the filled
//! fraction of the bar. A destroyed tower shows no bar. King towers are not read (their bar
//! only appears once damaged).

use calib::Calibration;
use capture::Frame;

/// Bar geometry relative to the arena box (measured on 576x1280 Note 14 frames): left bar
/// starts at x 94, right at 434, 70 px long; enemy bars at y 249-251, mine at 774-776.
const BAR_X: [f64; 2] = [0.1475, 0.7653];
const BAR_W: f64 = 0.1272;
const ENEMY_Y: f64 = 0.1118;
const MINE_Y: f64 = 0.7838;
/// Fewer filled pixels than this (fraction of the bar): no bar there.
const MIN_FILL: f32 = 0.03;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TowerHp {
    /// Left, right princess tower HP as a fraction of full; None = no bar (destroyed).
    pub enemy: [Option<f32>; 2],
    pub mine: [Option<f32>; 2],
}

fn is_red(p: [u8; 3]) -> bool {
    p[0] > 200 && p[1] < 90 && p[2] < 120
}

fn is_blue(p: [u8; 3]) -> bool {
    p[2] > 180 && p[0] < 120 && p[1] > 100
}

fn fill(frame: &Frame, calib: &Calibration, side: usize, y_rel: f64, color: fn([u8; 3]) -> bool) -> Option<f32> {
    let a = &calib.arena;
    let (ax, ay) = (a.top_left.x * frame.width as f64, a.top_left.y * frame.height as f64);
    let (aw, ah) = ((a.bottom_right.x - a.top_left.x) * frame.width as f64, (a.bottom_right.y - a.top_left.y) * frame.height as f64);
    let (x0, w) = ((ax + BAR_X[side] * aw) as u32, (BAR_W * aw) as u32);
    let y = (ay + y_rel * ah) as u32;
    // The fill grows from the left: the rightmost colored pixel marks the HP, on the best of 3 rows.
    let best = (y.saturating_sub(1)..=y + 1)
        .filter(|&yy| yy < frame.height)
        .map(|yy| {
            let hits: Vec<u32> = (0..w).filter(|&dx| x0 + dx < frame.width && color(frame.pixel(x0 + dx, yy))).collect();
            // Fewer than 3 colored pixels is noise, not a bar.
            if hits.len() < 3 { 0 } else { hits.last().map_or(0, |&r| r as usize + 1) }
        })
        .max()
        .unwrap_or(0);
    let f = best as f32 / w as f32;
    (f >= MIN_FILL).then_some(f.min(1.0))
}

/// Median of the last 7 readings per tower: a unit walking over a bar, or a frame without
/// one, does not change the HP. A tower reads as destroyed once most readings lack a bar.
#[derive(Debug, Default)]
pub struct TowerSmoother {
    last: std::collections::VecDeque<TowerHp>,
}

impl TowerSmoother {
    const N: usize = 7;

    pub fn push(&mut self, hp: TowerHp) {
        self.last.push_back(hp);
        while self.last.len() > Self::N {
            self.last.pop_front();
        }
    }

    pub fn get(&self) -> TowerHp {
        let pick = |f: &dyn Fn(&TowerHp) -> Option<f32>| {
            let mut v: Vec<f32> = self.last.iter().filter_map(f).collect();
            if v.len() * 2 <= self.last.len() {
                return None;
            }
            v.sort_by(f32::total_cmp);
            Some(v[v.len() / 2])
        };
        TowerHp {
            enemy: [pick(&|t| t.enemy[0]), pick(&|t| t.enemy[1])],
            mine: [pick(&|t| t.mine[0]), pick(&|t| t.mine[1])],
        }
    }
}

pub fn read_towers(frame: &Frame, calib: &Calibration) -> TowerHp {
    TowerHp {
        enemy: [0, 1].map(|s| fill(frame, calib, s, ENEMY_Y, is_red)),
        mine: [0, 1].map(|s| fill(frame, calib, s, MINE_Y, is_blue)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(name: &str) -> TowerHp {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let calib = calib::Calibration::load(root.join("calibration.toml")).unwrap();
        read_towers(&capture::load_rgb(root.join(format!("fixtures/towers/{name}.png"))).unwrap(), &calib)
    }

    fn near(a: Option<f32>, b: f32) -> bool {
        a.is_some_and(|a| (a - b).abs() < 0.04)
    }

    #[test]
    fn smoother_ignores_a_unit_walking_over_the_bar() {
        let hp = |e: Option<f32>| TowerHp { enemy: [e, Some(1.0)], mine: [Some(1.0), Some(1.0)] };
        let mut s = TowerSmoother::default();
        for r in [Some(0.8), Some(0.8), Some(0.3), Some(0.8), None, Some(0.79), Some(0.8)] {
            s.push(hp(r));
        }
        assert!((s.get().enemy[0].unwrap() - 0.8).abs() < 0.02, "{:?}", s.get());
        for _ in 0..7 {
            s.push(hp(None));
        }
        assert_eq!(s.get().enemy[0], None, "really destroyed");
    }

    #[test]
    fn full_towers_at_the_start() {
        let t = read("note14_30s");
        assert!(near(t.mine[0], 1.0) && near(t.mine[1], 1.0), "{t:?}");
    }

    #[test]
    fn damaged_enemy_towers_match_their_hp_numbers() {
        // On screen: 4456 and 2774 of 4858.
        let t = read("note14_170s");
        assert!(near(t.enemy[0], 4456.0 / 4858.0) && near(t.enemy[1], 2774.0 / 4858.0), "{t:?}");
    }

    #[test]
    fn destroyed_tower_has_no_bar() {
        let t = read("note14_100s");
        assert_eq!(t.mine[0], None, "{t:?}");
        assert!(t.mine[1].is_some_and(|h| h > 0.3 && h < 0.9), "{t:?}");
    }
}
