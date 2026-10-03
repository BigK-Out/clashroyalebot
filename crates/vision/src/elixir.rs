//! Elixir from the purple bar: find where the magenta fill ends.

use calib::Calibration;
use capture::Frame;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Elixir {
    /// What the game displays (whole elixir), 0..=10.
    pub value: u8,
    /// Continuous fill, 0.0..=10.0 (includes progress toward the next point).
    pub fill: f32,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Px {
    Fill,
    Empty,
    Other,
}

fn classify([r, g, b]: [u8; 3]) -> Px {
    let (r, g, b) = (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let sat = if max > 0.0 { (max - min) / max } else { 0.0 };
    // Fill is magenta (R≈B ≫ G); the unfilled bar is dark navy (B ≫ R).
    if sat > 0.45 && max > 0.45 && g < r && g < b && (r - b).abs() < 0.25 {
        Px::Fill
    } else if sat > 0.6 && max < 0.7 && b > r + 0.2 && b > g {
        Px::Empty
    } else {
        Px::Other
    }
}

/// Reads elixir, or None when the bar isn't visible (menus, end screen, transitions).
pub fn read_elixir(frame: &Frame, calib: &Calibration) -> Option<Elixir> {
    let r = calib.elixir_bar.to_px(frame.width, frame.height);
    if r.w < 20 || r.h < 4 {
        return None;
    }
    // Three rows through the middle band of the bar; a column's class is the majority vote.
    let rows = [r.y + r.h * 2 / 5, r.y + r.h / 2, r.y + r.h * 3 / 5];
    let classes: Vec<Px> = (r.x..r.x + r.w)
        .map(|x| {
            let votes = rows.map(|y| classify(frame.pixel(x, y)));
            let fill = votes.iter().filter(|&&c| c == Px::Fill).count();
            let empty = votes.iter().filter(|&&c| c == Px::Empty).count();
            if fill >= 2 {
                Px::Fill
            } else if empty >= 2 {
                Px::Empty
            } else {
                Px::Other
            }
        })
        .collect();

    // The first segment is covered by the drop icon and number; judge validity on the rest.
    let seg = classes.len() / 10;
    let visible = &classes[seg..];
    let known = visible.iter().filter(|&&c| c != Px::Other).count();
    if known * 10 < visible.len() * 8 {
        return None;
    }

    // Fill end: rightmost Fill column with (almost) no Fill after it.
    let total = classes.len();
    let mut end = 0;
    let mut fill_after = 0;
    for (i, c) in classes.iter().enumerate().rev() {
        if *c == Px::Fill {
            if fill_after * 20 <= total - i {
                end = i + 1;
                break;
            }
            fill_after += 1;
        }
    }
    if end <= seg {
        // Fill hidden behind the number: 0, or partway to 1.
        end = 0;
    }
    let fill = (end as f32 / total as f32 * 10.0).clamp(0.0, 10.0);
    // Snap slightly up: the fill's anti-aliased edge sits a few px before the segment boundary.
    let value = ((fill + 0.08).floor() as u8).min(10);
    Some(Elixir { value, fill })
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib::NRect;
    use std::time::Instant;

    const FILL: [u8; 3] = [235, 53, 246];
    const EMPTY: [u8; 3] = [5, 54, 123];
    const GRASS: [u8; 3] = [90, 160, 60];

    /// 200x20 frame whose bar (the whole frame) is filled up to `fill` of 10.
    fn bar(fill: f32) -> (Frame, Calibration) {
        let (w, h) = (200u32, 20u32);
        let cut = (fill / 10.0 * w as f32) as u32;
        let mut rgb = Vec::new();
        for _ in 0..h {
            for x in 0..w {
                rgb.extend(if x < cut { FILL } else { EMPTY });
            }
        }
        let calib = Calibration { elixir_bar: NRect { x: 0.0, y: 0.0, w: 1.0, h: 1.0 }, ..Default::default() };
        (Frame { seq: 0, width: w, height: h, rgb, captured_at: Instant::now() }, calib)
    }

    #[test]
    fn reads_whole_and_partial_values() {
        for (fill, want) in [(0.0, 0), (3.0, 3), (6.4, 6), (7.95, 7), (10.0, 10)] {
            let (f, c) = bar(fill);
            let e = read_elixir(&f, &c).unwrap();
            assert_eq!(e.value, want, "fill {fill} -> {e:?}");
            assert!((e.fill - fill).abs() < 0.15, "fill {fill} -> {e:?}");
        }
    }

    #[test]
    fn no_bar_is_none() {
        let (mut f, c) = bar(5.0);
        for p in f.rgb.chunks_exact_mut(3) {
            p.copy_from_slice(&GRASS);
        }
        assert_eq!(read_elixir(&f, &c), None);
    }

    #[test]
    fn classify_samples_from_real_frames() {
        assert_eq!(classify([210, 34, 216]), Px::Fill); // full bar
        assert_eq!(classify([255, 74, 255]), Px::Fill); // cost-preview highlight
        assert_eq!(classify([5, 55, 124]), Px::Empty);
        assert_eq!(classify([255, 255, 255]), Px::Other); // number text
        assert_eq!(classify(GRASS), Px::Other);
    }
}
