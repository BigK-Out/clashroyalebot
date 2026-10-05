//! Crowns on the result screen: gold crowns fill the opponent's banner (above "VS") and mine
//! (below it). Same layout on both phones (576 px wide frames).

use capture::Frame;

/// Crown slot centers (x, 576-px frames) and half width.
const SLOTS: [u32; 3] = [125, 258, 392];
const HALF: u32 = 40;
/// The "VS" line between the two banners, as a fraction of the height.
const SPLIT: f64 = 0.4;
/// Gold pixels in a slot (both banners together) for it to count as a crown.
const MIN_GOLD: u32 = 300;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Crowns {
    pub mine: u8,
    pub theirs: u8,
}

fn is_gold(p: [u8; 3]) -> bool {
    p[0] > 220 && p[1] > 140 && p[1] < 215 && p[2] < 90
}

/// Counts crowns on a result screen (call only when the screen is the result screen).
pub fn read_crowns(frame: &Frame) -> Crowns {
    let split = (frame.height as f64 * SPLIT) as u32;
    let count = |y0: u32, y1: u32| {
        SLOTS
            .iter()
            .filter(|&&cx| {
                let mut n = 0;
                for y in y0..y1 {
                    for x in cx.saturating_sub(HALF)..(cx + HALF).min(frame.width) {
                        n += u32::from(is_gold(frame.pixel(x, y)));
                    }
                }
                n >= MIN_GOLD
            })
            .count() as u8
    };
    Crowns { theirs: count(frame.height / 6, split), mine: count(split, frame.height * 3 / 5) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(name: &str) -> Crowns {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        read_crowns(&capture::load_rgb(root.join(format!("fixtures/screens/{name}.png"))).unwrap())
    }

    #[test]
    fn crowns_on_the_result_screen() {
        assert_eq!(read("note14/result_loss"), Crowns { mine: 0, theirs: 3 });
        assert_eq!(read("note9/result_win"), Crowns { mine: 3, theirs: 0 });
        assert_eq!(read("note14/result"), Crowns { mine: 0, theirs: 0 }, "draw");
        assert_eq!(read("note9/result"), Crowns { mine: 0, theirs: 0 }, "draw");
    }
}
