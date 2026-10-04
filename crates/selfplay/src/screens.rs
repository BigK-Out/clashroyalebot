//! Which screen a phone shows, from a few colour boxes measured once per screen, so the
//! orchestrator only taps when it knows where it is.

use std::collections::HashMap;

use capture::Frame;
use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
pub struct ScreenCheck {
    pub name: String,
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
    pub rgb: [u8; 3],
    pub tol: u8,
    pub min_frac: f32,
}

#[derive(Deserialize, Debug, Clone)]
pub struct PhoneFlows {
    pub screens: Vec<ScreenCheck>,
    pub taps: HashMap<String, (f64, f64)>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Flows {
    pub note14: PhoneFlows,
    pub note9: PhoneFlows,
}

pub fn matches(frame: &Frame, c: &ScreenCheck) -> bool {
    let (fw, fh) = (frame.width as f64, frame.height as f64);
    let (x0, y0) = ((c.x * fw) as u32, (c.y * fh) as u32);
    let (x1, y1) = (((c.x + c.w) * fw) as u32, ((c.y + c.h) * fh) as u32);
    let (mut hit, mut total) = (0u32, 0u32);
    for y in y0..y1.min(frame.height) {
        for x in x0..x1.min(frame.width) {
            let p = frame.pixel(x, y);
            total += 1;
            if p.iter().zip(c.rgb).all(|(a, b)| a.abs_diff(b) <= c.tol) {
                hit += 1;
            }
        }
    }
    total > 0 && hit as f32 / total as f32 >= c.min_frac
}

pub fn classify(frame: &Frame, screens: &[ScreenCheck]) -> Option<String> {
    screens.iter().find(|s| matches(frame, s)).map(|s| s.name.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn flows() -> Flows {
        toml::from_str(&std::fs::read_to_string(root().join("selfplay_flows.toml")).unwrap()).unwrap()
    }

    #[test]
    fn every_fixture_is_recognized_as_itself() {
        let f = flows();
        for (phone, screens) in [("note9", &f.note9.screens), ("note14", &f.note14.screens)] {
            for s in screens.iter() {
                let frame = capture::load_rgb(root().join(format!("fixtures/screens/{phone}/{}.png", s.name))).unwrap();
                assert_eq!(classify(&frame, screens).as_deref(), Some(s.name.as_str()), "{phone}/{}", s.name);
            }
        }
    }

    #[test]
    fn flow_refuses_unknown_screen() {
        // A battle frame is none of the menu screens the flow taps on.
        let f = flows();
        let battle = capture::load_rgb(root().join("fixtures/note9/battle_1.png")).unwrap();
        let menus: Vec<_> = f.note9.screens.into_iter().filter(|s| s.name != "battle").collect();
        assert_eq!(classify(&battle, &menus), None);
    }

    #[test]
    fn result_variants_recognized() {
        // The fixtures named "result" are draws; wins and losses have different banners.
        let f = flows();
        for (phone, file, screens) in [("note9", "result_win", &f.note9.screens), ("note14", "result_loss", &f.note14.screens)] {
            let frame = capture::load_rgb(root().join(format!("fixtures/screens/{phone}/{file}.png"))).unwrap();
            assert_eq!(classify(&frame, screens).as_deref(), Some("result"), "{phone}/{file}");
        }
    }

    #[test]
    fn no_unmeasured_taps() {
        let f = flows();
        for (k, v) in f.note9.taps.iter().chain(f.note14.taps.iter()) {
            assert!(*v != (0.0, 0.0), "tap {k} not measured");
        }
    }
}
