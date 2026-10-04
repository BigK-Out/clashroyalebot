//! Perception on real labeled frames (fixtures/frames/truth.toml).

use std::path::PathBuf;

use calib::Calibration;
use serde::Deserialize;
use vision::{CardLibrary, Slot, read_elixir};

#[derive(Deserialize)]
struct Truth {
    frame: Vec<Labeled>,
}

#[derive(Deserialize)]
struct Labeled {
    file: String,
    elixir: Option<u8>,
    hand: Option<[String; 4]>,
    next: Option<String>,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn slot_label(s: &Slot) -> String {
    match s {
        Slot::Card(m) => m.name.clone(),
        Slot::Empty => vision::hand::EMPTY_TEMPLATE.to_string(),
        Slot::Unknown { best } => format!("?{}", best.as_ref().map_or("", |b| b.name.as_str())),
    }
}

#[test]
fn reads_elixir_and_hand_on_labeled_frames() {
    let calib = Calibration::load(root().join("calibration.toml")).unwrap();
    let cards = CardLibrary::load(root().join("assets/cards")).unwrap();
    let truth: Truth =
        toml::from_str(&std::fs::read_to_string(root().join("fixtures/frames/truth.toml")).unwrap()).unwrap();

    let mut failures = Vec::new();
    for t in &truth.frame {
        let frame = capture::load_rgb(root().join("fixtures/frames").join(&t.file)).unwrap();
        let elixir = read_elixir(&frame, &calib);
        if elixir.map(|e| e.value) != t.elixir {
            failures.push(format!("{}: elixir {elixir:?}, want {:?}", t.file, t.elixir));
        }
        if let (Some(want_hand), Some(want_next)) = (&t.hand, &t.next) {
            let hand = cards.read_hand(&frame, &calib);
            let got: Vec<String> = hand.slots.iter().map(slot_label).collect();
            if got != want_hand.to_vec() {
                failures.push(format!("{}: hand {got:?}, want {want_hand:?} ({:?})", t.file, hand.slots));
            }
            if slot_label(&hand.next) != *want_next {
                failures.push(format!("{}: next {:?}, want {want_next}", t.file, hand.next));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn note9_elixir_and_hand_read() {
    let calib = Calibration::load(root().join("calibration_note9.toml")).unwrap();
    let cards = CardLibrary::load(root().join("assets/cards")).unwrap();
    for i in 1..=5 {
        let f = capture::load_rgb(root().join(format!("fixtures/note9/battle_{i}.png"))).unwrap();
        assert!(read_elixir(&f, &calib).is_some(), "elixir on battle_{i}");
        let hand = cards.read_hand(&f, &calib);
        let known = hand.slots.iter().filter(|s| s.name().is_some()).count();
        assert!(known >= 3, "battle_{i}: {:?}", hand.slots);
    }
}

#[test]
fn portrait_templates_read_note9_hands() {
    let calib = calib::Calibration::load(root().join("calibration_note9.toml")).unwrap();
    let deck = ["hog_rider", "musketeer", "cannon", "ice_golem", "skeletons", "ice_spirit", "the_log", "fireball"];
    let lib = vision::CardLibrary::load_subset(root().join("assets/cards_all"), &deck).unwrap();
    let reference = vision::CardLibrary::load(root().join("assets/cards")).unwrap();
    for i in 1..=5 {
        let f = capture::load_rgb(root().join(format!("fixtures/note9/battle_{i}.png"))).unwrap();
        let want = reference.read_hand(&f, &calib);
        let got = lib.read_hand(&f, &calib);
        for s in 0..4 {
            if let Some(w) = want.slots[s].name() {
                assert_eq!(got.slots[s].name(), Some(w), "battle_{i} slot {s}");
            }
        }
    }
}
