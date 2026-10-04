//! One sparring step: read elixir and hand from a screenshot, maybe play, log the play.

use calib::Calibration;
use capture::Frame;
use input::{Deployer, TapBackend};
use selfplay::{PlayRecord, SparringPolicy, host_ms};
use vision::CardLibrary;

pub struct Ctx<B: TapBackend> {
    pub calib: Calibration,
    pub cards: CardLibrary,
    pub policy: SparringPolicy,
    pub deployer: Deployer<B>,
    pub dry_run: bool,
}

pub fn hand_names(frame: &Frame, ctx: &Ctx<impl TapBackend>) -> [Option<String>; 4] {
    let hand = ctx.cards.read_hand(frame, &ctx.calib);
    std::array::from_fn(|i| hand.slots[i].name().map(str::to_string))
}

/// The played card left its slot (slot now empty or holding a different card).
pub fn verify(before: &[Option<String>; 4], after: &[Option<String>; 4], slot: usize, card: &str) -> bool {
    before[slot].as_deref() == Some(card) && after[slot].as_deref() != Some(card)
}

/// Decides on `frame`; if a card is played, returns its record with `verified: false`
/// (the caller re-reads the hand and sets it).
pub fn step<B: TapBackend>(frame: &Frame, ctx: &mut Ctx<B>) -> anyhow::Result<Option<PlayRecord>> {
    let elixir = vision::read_elixir(frame, &ctx.calib).map(|e| e.value);
    let hand = hand_names(frame, ctx);
    let Some(play) = ctx.policy.decide(host_ms(), elixir, &hand) else { return Ok(None) };
    if !ctx.dry_run {
        ctx.deployer.deploy(play.slot, play.tile.0, play.tile.1, play.enemy_half)?;
    }
    Ok(Some(PlayRecord {
        host_ms: host_ms(),
        card: play.card,
        slot: play.slot,
        tile: play.tile,
        hand_before: hand,
        elixir_read: elixir,
        verified: false,
    }))
}

#[cfg(test)]
mod tests {
    use super::verify;

    fn h(c: [&str; 4]) -> [Option<String>; 4] {
        c.map(|s| (!s.is_empty()).then(|| s.to_string()))
    }

    #[test]
    fn verified_when_card_left_slot() {
        assert!(verify(&h(["hog_rider", "cannon", "the_log", "skeletons"]), &h(["hog_rider", "", "the_log", "skeletons"]), 1, "cannon"));
        assert!(verify(&h(["hog_rider", "cannon", "the_log", "skeletons"]), &h(["hog_rider", "fireball", "the_log", "skeletons"]), 1, "cannon"));
    }

    #[test]
    fn unverified_when_slot_unchanged() {
        assert!(!verify(&h(["hog_rider", "cannon", "the_log", "skeletons"]), &h(["hog_rider", "cannon", "the_log", "skeletons"]), 1, "cannon"));
    }

    struct FakeTaps(Vec<(u32, u32)>);
    impl input::TapBackend for FakeTaps {
        fn screen_size(&self) -> (u32, u32) { (1080, 2340) }
        fn taps(&mut self, p: &[(u32, u32)]) -> anyhow::Result<()> { self.0.extend_from_slice(p); Ok(()) }
    }

    #[test]
    fn plays_on_fixture_frames() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let calib = calib::Calibration::load(root.join("calibration_note9.toml")).unwrap();
        let mut ctx = super::Ctx {
            deployer: input::Deployer::new(FakeTaps(Vec::new()), calib.clone()).unwrap(),
            calib,
            cards: vision::CardLibrary::load(root.join("assets/cards")).unwrap(),
            policy: selfplay::SparringPolicy::new(1),
            dry_run: false,
        };
        let mut played = 0;
        for i in 1..=5 {
            let f = capture::load_rgb(root.join(format!("fixtures/note9/battle_{i}.png"))).unwrap();
            // Fresh policy each frame so the inter-play wait never blocks the test.
            ctx.policy = selfplay::SparringPolicy::new(i);
            if let Some(r) = super::step(&f, &mut ctx).unwrap() {
                assert!(r.hand_before[r.slot].as_deref() == Some(r.card.as_str()));
                played += 1;
            }
        }
        assert!(played >= 3, "played {played}/5");
        assert_eq!(ctx.deployer.backend().0.len(), played * 2, "two taps per play");
    }
}
