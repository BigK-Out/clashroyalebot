//! Unit detection without a model: every troop/building shows a level tag above it
//! (blue plate = mine, red/magenta plate = enemy) with white digits.
//!
//! Gives team + position, not unit type.

use calib::{ArenaMapping, Calibration, NPoint};
use capture::Frame;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Team {
    Ally,
    Enemy,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Unit {
    pub team: Team,
    /// Level tag box, normalized screen coordinates (x0, y0, x1, y1).
    pub tag: [f32; 4],
    /// Estimated feet position (below the tag), normalized.
    pub feet: NPoint,
    /// Tile under the feet, if inside the arena.
    pub tile: Option<(u32, u32)>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Px {
    Ally,
    Enemy,
    White,
    Yellow,
    Other,
}

fn hsv([r, g, b]: [u8; 3]) -> (f32, f32, f32) {
    let (r, g, b) = (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let h = if d == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / d) % 6.0)
    } else if max == g {
        60.0 * ((b - r) / d + 2.0)
    } else {
        60.0 * ((r - g) / d + 4.0)
    };
    (if h < 0.0 { h + 360.0 } else { h }, if max > 0.0 { d / max } else { 0.0 }, max)
}

fn classify(p: [u8; 3]) -> Px {
    let (h, s, v) = hsv(p);
    if s < 0.3 && v > 0.8 {
        Px::White
    } else if (185.0..=235.0).contains(&h) && s > 0.4 && v > 0.5 {
        Px::Ally
    } else if (h >= 290.0 || h <= 8.0) && s > 0.55 && v > 0.3 {
        Px::Enemy
    } else if (38.0..=60.0).contains(&h) && s > 0.5 && v > 0.7 {
        Px::Yellow
    } else {
        Px::Other
    }
}

/// Connected components (8-connectivity: thin digit strokes touch diagonally)
/// → bounding boxes (x0, y0, x1 exclusive, y1 exclusive, area).
fn components(mask: &[bool], w: usize, h: usize) -> Vec<(usize, usize, usize, usize, usize)> {
    let mut seen = vec![false; w * h];
    let mut out = Vec::new();
    let mut stack = Vec::new();
    for start in 0..w * h {
        if !mask[start] || seen[start] {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        let (mut x0, mut y0, mut x1, mut y1, mut area) = (w, h, 0, 0, 0);
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
            area += 1;
            let mut push = |j: usize| {
                if mask[j] && !seen[j] {
                    seen[j] = true;
                    stack.push(j);
                }
            };
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let (nx, ny) = (x as isize + dx, y as isize + dy);
                    if (dx, dy) != (0, 0) && nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                        push(ny as usize * w + nx as usize);
                    }
                }
            }
        }
        out.push((x0, y0, x1, y1, area));
    }
    out
}

/// Detects unit level tags in the arena crop.
///
/// Tags are white digits (the level, 1–2 digits) on a team-colored plate. Red units are the
/// same color as red plates, so plates can't be segmented by color alone; instead find white
/// digit blobs with plate color directly above and below, then group neighbouring digits.
/// Tower labels show 3–4 digit HP numbers and are rejected by digit count.
pub fn detect_units(frame: &Frame, calib: &Calibration, mapping: &ArenaMapping) -> Vec<Unit> {
    let r = crate::arena::crop_rect(calib).to_px(frame.width, frame.height);
    let (w, h) = (r.w as usize, r.h as usize);
    let classes: Vec<Px> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .map(|(x, y)| classify(frame.pixel(r.x + x as u32, r.y + y as u32)))
        .collect();
    let at = |x: usize, y: usize| classes[y * w + x];
    // Thresholds were tuned on the 576-px-wide feed; scale with the frame.
    let s = frame.width as f32 / 576.0;
    let px = |v: f32| ((v * s).round() as usize).max(1);

    // Fraction of `team` pixels in rows y0..y1 (clamped), columns x0..x1.
    let band = |x0: usize, x1: usize, y0: isize, y1: isize, team: Px| {
        let (y0, y1) = (y0.clamp(0, h as isize) as usize, y1.clamp(0, h as isize) as usize);
        let (x0, x1) = (x0.min(w), x1.min(w));
        let total = (y1.saturating_sub(y0)) * (x1.saturating_sub(x0));
        if total == 0 {
            return 0.0;
        }
        (y0..y1).flat_map(|y| (x0..x1).map(move |x| (x, y))).filter(|&(x, y)| at(x, y) == team).count() as f32
            / total as f32
    };

    // 1. Digit candidates: small white blobs framed by plate color above and below.
    let white: Vec<bool> = classes.iter().map(|&c| c == Px::White).collect();
    let mut digits: Vec<(Team, usize, usize, usize, usize)> = Vec::new(); // team, x0, y0, x1, y1
    for (x0, y0, x1, y1, area) in components(&white, w, h) {
        let (bw, bh) = (x1 - x0, y1 - y0);
        if bh < px(7.0) || bh > px(12.0) || bw > px(10.0) || area < px(2.0).pow(2) {
            continue;
        }
        let (fx0, fx1) = (x0.saturating_sub(1), x1 + 1);
        let pad = px(3.0) as isize;
        for (team, color) in [(Team::Ally, Px::Ally), (Team::Enemy, Px::Enemy)] {
            if band(fx0, fx1, y0 as isize - pad, y0 as isize, color) >= 0.35
                && band(fx0, fx1, y1 as isize, y1 as isize + pad, color) >= 0.35
            {
                digits.push((team, x0, y0, x1, y1));
                break;
            }
        }
    }

    // 2. Group digits on the same line into tags.
    digits.sort_by_key(|d| (d.0 == Team::Enemy, d.2, d.1));
    let mut tags: Vec<(Team, usize, usize, usize, usize, usize)> = Vec::new(); // + digit count
    for (team, x0, y0, x1, y1) in digits {
        let near = px(7.0);
        match tags.iter_mut().find(|t| {
            t.0 == team && (t.2 as isize - y0 as isize).abs() <= px(4.0) as isize && x0 <= t.3 + near && x1 + near >= t.1
        }) {
            Some(t) => {
                (t.1, t.2, t.3, t.4) = (t.1.min(x0), t.2.min(y0), t.3.max(x1), t.4.max(y1));
                t.5 += 1;
            }
            None => tags.push((team, x0, y0, x1, y1, 1)),
        }
    }

    let (fw, fh) = (frame.width as f32, frame.height as f32);
    let tile_h = (calib.arena.bottom_left.y - calib.arena.top_left.y) / calib::ARENA_ROWS as f64;
    tags.into_iter()
        // Tower HP labels (3–4 digits) carry a yellow crown badge just left of the plate.
        // Units don't; their tags can pick up stray white flecks, so digit count alone isn't enough.
        .filter(|&(_, x0, y0, _, y1, n)| {
            let crown = band(x0.saturating_sub(px(22.0)), x0, y0 as isize - px(4.0) as isize, y1 as isize + px(4.0) as isize, Px::Yellow);
            n <= 4 && crown < 0.12
        })
        .map(|(team, x0, y0, x1, y1, _)| {
            let tag = [
                (r.x as f32 + x0 as f32) / fw,
                (r.y as f32 + y0 as f32) / fh,
                (r.x as f32 + x1 as f32) / fw,
                (r.y as f32 + y1 as f32) / fh,
            ];
            // The plate sits at the left end of the health bar; the unit stands about one
            // tile below it, slightly right of the plate.
            let feet = NPoint { x: tag[2] as f64 + 0.02, y: tag[3] as f64 + 1.2 * tile_h };
            Unit { team, tag, feet, tile: mapping.screen_to_tile_index(feet) }
        })
        .collect()
}

/// Pixel region of a unit's body (below its tag), as cut for the unit-type classifier.
/// Shared by dataset extraction and runtime classification so both see the same crop.
pub fn body_rect(u: &Unit, calib: &Calibration, width: u32, height: u32) -> Option<calib::PxRect> {
    let a = &calib.arena;
    let (tile_w, tile_h) = ((a.top_right.x - a.top_left.x) / 18.0, (a.bottom_left.y - a.top_left.y) / 32.0);
    let x0 = ((u.feet.x - 1.3 * tile_w) * width as f64).max(0.0) as u32;
    let x1 = (((u.feet.x + 1.3 * tile_w) * width as f64) as u32).min(width);
    let y0 = (u.tag[3] as f64 * height as f64).max(0.0) as u32;
    let y1 = (((u.tag[3] as f64 + 2.4 * tile_h) * height as f64) as u32).min(height);
    (x1 > x0 + 8 && y1 > y0 + 8).then_some(calib::PxRect { x: x0, y: y0, w: x1 - x0, h: y1 - y0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_measured_colors() {
        assert_eq!(classify([110, 200, 255]), Px::Ally); // light-blue tag (h~200, s~0.57)
        assert_eq!(classify([60, 110, 220]), Px::Ally);
        assert_eq!(classify([200, 30, 70]), Px::Enemy);
        assert_eq!(classify([250, 250, 250]), Px::White);
        assert_eq!(classify([250, 200, 40]), Px::Yellow);
        assert_eq!(classify([60, 220, 210]), Px::Other); // river turquoise (h~177)
        assert_eq!(classify([140, 190, 70]), Px::Other); // grass
    }

    #[test]
    fn components_join_diagonals_and_split_gaps() {
        let (w, h) = (6, 3);
        #[rustfmt::skip]
        let m = [
            true,  false, false, false, false, true,
            false, true,  false, false, false, true,
            false, false, true,  false, false, false,
        ];
        let mut comps = components(&m, w, h);
        comps.sort();
        assert_eq!(comps, vec![(0, 0, 3, 3, 3), (5, 0, 6, 2, 2)]);
    }

}
