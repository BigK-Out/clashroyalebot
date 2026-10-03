//! Game state built from per-frame perception, smoothed over time.

use std::time::{Duration, Instant};

use vision::{Elixir, Hand, Slot};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GameState {
    pub in_battle: bool,
    /// Whole elixir as displayed, 0..=10.
    pub elixir: u8,
    /// Continuous elixir, 0.0..=10.0.
    pub elixir_fill: f32,
    /// Card in each slot (None = empty or never seen).
    pub hand: [Option<String>; 4],
    pub next: Option<String>,
    /// Time since the battle started (first frame with an elixir bar).
    pub battle_time: Duration,
}

impl GameState {
    /// Slot index holding `card`, if any.
    pub fn slot_of(&self, card: &str) -> Option<usize> {
        self.hand.iter().position(|c| c.as_deref() == Some(card))
    }
}

/// Folds per-frame perception into `GameState`.
///
/// Per-frame reads are noisy (greyed cards, animations, popups); a slot only changes when the
/// frame shows a confidently matched card or a clear empty slot. Unknown keeps the last value.
/// Battle ends after `BATTLE_END_GRACE` without an elixir bar (popups flicker it briefly).
#[derive(Default)]
pub struct Tracker {
    state: GameState,
    battle_start: Option<Instant>,
    last_bar: Option<Instant>,
}

pub const BATTLE_END_GRACE: Duration = Duration::from_secs(10);

impl Tracker {
    pub fn update(&mut self, now: Instant, elixir: Option<Elixir>, hand: Option<&Hand>) -> &GameState {
        match elixir {
            Some(e) => {
                if !self.state.in_battle {
                    self.state = GameState { in_battle: true, ..Default::default() };
                    self.battle_start = Some(now);
                }
                self.last_bar = Some(now);
                self.state.elixir = e.value;
                self.state.elixir_fill = e.fill;
            }
            None => {
                if self.state.in_battle && self.last_bar.is_none_or(|t| now.duration_since(t) > BATTLE_END_GRACE) {
                    self.state.in_battle = false;
                }
            }
        }
        if let (Some(h), Some(_)) = (hand, elixir) {
            for (slot, read) in self.state.hand.iter_mut().zip(&h.slots) {
                apply(slot, read);
            }
            apply(&mut self.state.next, &h.next);
        }
        if let Some(start) = self.battle_start {
            self.state.battle_time = now.duration_since(start);
        }
        &self.state
    }

    pub fn state(&self) -> &GameState {
        &self.state
    }

    /// Forget a slot right after playing it, so the old card isn't played twice
    /// before the frame catches up.
    pub fn mark_played(&mut self, slot: usize) {
        self.state.hand[slot] = None;
    }
}

fn apply(slot: &mut Option<String>, read: &Slot) {
    match read {
        Slot::Card(m) => *slot = Some(m.name.clone()),
        Slot::Empty => *slot = None,
        Slot::Unknown { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision::CardMatch;

    fn card(n: &str) -> Slot {
        Slot::Card(CardMatch { name: n.into(), score: 0.9, raised: false })
    }

    fn hand(slots: [Slot; 4], next: Slot) -> Hand {
        Hand { slots, next }
    }

    fn el(v: u8) -> Option<Elixir> {
        Some(Elixir { value: v, fill: v as f32 })
    }

    #[test]
    fn unknown_keeps_last_card_and_empty_clears() {
        let t0 = Instant::now();
        let mut t = Tracker::default();
        t.update(t0, el(5), Some(&hand([card("a"), card("b"), card("c"), card("d")], card("e"))));
        let unk = || Slot::Unknown { best: None };
        let s = t.update(t0, el(5), Some(&hand([unk(), Slot::Empty, card("x"), unk()], unk())));
        assert_eq!(s.hand, [Some("a".into()), None, Some("x".into()), Some("d".into())]);
        assert_eq!(s.next.as_deref(), Some("e"));
        assert_eq!(s.slot_of("x"), Some(2));
    }

    #[test]
    fn battle_starts_on_bar_and_ends_after_grace() {
        let t0 = Instant::now();
        let mut t = Tracker::default();
        assert!(!t.update(t0, None, None).in_battle);
        assert!(t.update(t0 + Duration::from_secs(1), el(5), None).in_battle);
        // A popup hides the bar briefly: still in battle.
        assert!(t.update(t0 + Duration::from_secs(2), None, None).in_battle);
        let s = t.update(t0 + Duration::from_secs(15), None, None);
        assert!(!s.in_battle);
        assert_eq!(s.battle_time, Duration::from_secs(14));
    }

    #[test]
    fn new_battle_resets_hand() {
        let t0 = Instant::now();
        let mut t = Tracker::default();
        t.update(t0, el(5), Some(&hand([card("a"), card("b"), card("c"), card("d")], card("e"))));
        t.update(t0 + Duration::from_secs(20), None, None);
        let s = t.update(t0 + Duration::from_secs(30), el(5), None);
        assert!(s.in_battle && s.hand.iter().all(Option::is_none) && s.battle_time.is_zero());
    }

    #[test]
    fn mark_played_clears_slot() {
        let mut t = Tracker::default();
        t.update(Instant::now(), el(5), Some(&hand([card("a"), card("b"), card("c"), card("d")], card("e"))));
        t.mark_played(1);
        assert_eq!(t.state().hand[1], None);
    }
}
