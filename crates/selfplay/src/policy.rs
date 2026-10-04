//! The sparring phone's play policy: random but legal, varied placements and timing, so the
//! observer sees every card in many situations. Not trying to win.

#[derive(Debug, Clone, PartialEq)]
pub struct SparPlay {
    pub slot: usize,
    pub card: String,
    /// Tile in the sparring phone's own view.
    pub tile: (u32, u32),
    /// Spells on the observer's half.
    pub enemy_half: bool,
}

pub struct SparringPolicy {
    rng: fastrand::Rng,
    last_play_ms: u64,
    wait_ms: u64,
}

impl SparringPolicy {
    pub fn new(seed: u64) -> Self {
        Self { rng: fastrand::Rng::with_seed(seed), last_play_ms: 0, wait_ms: 0 }
    }

    fn tile_for(&mut self, kind: &str) -> ((u32, u32), bool) {
        let r = &mut self.rng;
        match kind {
            "Spell" => ((r.u32(2..=15), r.u32(2..=14)), true),
            "Building" => ((r.u32(7..=10), r.u32(20..=23)), false),
            _ => {
                let col = if r.bool() { r.u32(2..=4) } else { r.u32(13..=15) };
                let row = match r.u32(0..3) {
                    0 => r.u32(17..=18), // bridge
                    1 => r.u32(28..=30), // back
                    _ => r.u32(20..=24), // in front of the tower
                };
                ((col, row), false)
            }
        }
    }

    pub fn decide(&mut self, now_ms: u64, elixir: Option<u8>, hand: &[Option<String>; 4]) -> Option<SparPlay> {
        let elixir = elixir?;
        if elixir < 10 && now_ms < self.last_play_ms + self.wait_ms {
            return None;
        }
        let cards = brain::cards();
        let options: Vec<(usize, &str, String)> = hand
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let c = c.as_deref()?;
                let card = cards.get(c)?;
                (card.elixir? <= elixir).then(|| (i, c, card.kind.clone().unwrap_or_default()))
            })
            .collect();
        if options.is_empty() {
            return None;
        }
        let (slot, card, kind) = options[self.rng.usize(..options.len())].clone();
        let (tile, enemy_half) = self.tile_for(&kind);
        self.last_play_ms = now_ms;
        self.wait_ms = self.rng.u64(1_000..=6_000);
        Some(SparPlay { slot, card: card.to_string(), tile, enemy_half })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hand(c: [&str; 4]) -> [Option<String>; 4] {
        c.map(|s| (!s.is_empty()).then(|| s.to_string()))
    }

    #[test]
    fn no_elixir_no_play() {
        let mut p = SparringPolicy::new(1);
        assert_eq!(p.decide(100_000, None, &hand(["hog_rider", "cannon", "the_log", "skeletons"])), None);
    }

    #[test]
    fn only_affordable_cards() {
        for seed in 0..200 {
            let mut p = SparringPolicy::new(seed);
            let play = p.decide(100_000, Some(2), &hand(["hog_rider", "musketeer", "the_log", "skeletons"]));
            if let Some(pl) = play {
                assert!(pl.card == "the_log" || pl.card == "skeletons", "{pl:?}");
            }
        }
    }

    #[test]
    fn placement_by_card_type() {
        for seed in 0..300 {
            let mut p = SparringPolicy::new(seed);
            let Some(pl) = p.decide(100_000, Some(10), &hand(["hog_rider", "cannon", "fireball", "skeletons"])) else {
                panic!("full elixir must play");
            };
            match pl.card.as_str() {
                "fireball" => assert!(pl.enemy_half && (2..=14).contains(&pl.tile.1), "{pl:?}"),
                "cannon" => assert!(!pl.enemy_half && (7..=10).contains(&pl.tile.0) && (20..=23).contains(&pl.tile.1), "{pl:?}"),
                _ => assert!(!pl.enemy_half && (17..=31).contains(&pl.tile.1) && pl.tile.0 < 18, "{pl:?}"),
            }
        }
    }

    #[test]
    fn waits_between_plays_unless_full() {
        let mut p = SparringPolicy::new(7);
        let h = hand(["hog_rider", "cannon", "the_log", "skeletons"]);
        assert!(p.decide(100_000, Some(10), &h).is_some());
        // Right after a play, with elixir but not full: wait.
        assert_eq!(p.decide(100_200, Some(8), &h), None);
        // Full elixir overrides the wait.
        assert!(p.decide(100_300, Some(10), &h).is_some());
    }

    #[test]
    fn unknown_slots_are_skipped() {
        let mut p = SparringPolicy::new(3);
        let pl = p.decide(100_000, Some(10), &hand(["", "", "skeletons", ""])).unwrap();
        assert_eq!((pl.slot, pl.card.as_str()), (2, "skeletons"));
    }
}
