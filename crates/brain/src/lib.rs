//! Decision making: the `Policy` trait and a first rule-based Hog 2.6 bot.

use std::time::Duration;

use state::GameState;

#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    Deploy {
        slot: usize,
        card: String,
        col: u32,
        row: u32,
        /// Target is on the enemy half (spells).
        enemy_half: bool,
    },
}

pub trait Policy {
    fn decide(&mut self, state: &GameState) -> Option<Action>;

    /// Called after an action was executed, so the policy can track timers.
    fn on_action(&mut self, _action: &Action, _state: &GameState) {}
}

/// Elixir cost of known cards.
pub fn cost(card: &str) -> Option<u8> {
    Some(match card {
        "ice_spirit" | "skeletons" => 1,
        "the_log" | "ice_golem" => 2,
        "cannon" | "archers" | "arrows" | "minions" | "knight" => 3,
        "hog_rider" | "musketeer" | "fireball" | "mini_pekka" => 4,
        "giant" => 5,
        _ => return None,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Lane {
    Left,
    Right,
}

impl Lane {
    /// Bridge tile on my side.
    fn bridge(self) -> (u32, u32) {
        match self {
            Lane::Left => (3, 17),
            Lane::Right => (14, 17),
        }
    }

    /// Enemy princess tower center (spell target).
    fn enemy_tower(self) -> (u32, u32) {
        match self {
            Lane::Left => (3, 6),
            Lane::Right => (14, 6),
        }
    }

    /// Behind my princess tower (ranged support).
    fn behind_tower(self) -> (u32, u32) {
        match self {
            Lane::Left => (3, 26),
            Lane::Right => (14, 26),
        }
    }
}

/// Hog 2.6 without vision of enemy troops (v1, before YOLO):
/// - push: Hog Rider at the bridge when elixir allows, Ice Spirit right behind it;
/// - don't leak: near full elixir, play a defensive card in the center or a cycle card;
/// - at 10 elixir with Fireball in hand, Fireball the enemy tower on the push lane.
pub struct HogCycle {
    pub lane: Lane,
    /// Elixir at which Hog Rider goes in.
    pub hog_at: u8,
    /// Elixir at which a defensive / cycle card is played to avoid leaking.
    pub leak_at: u8,
    /// When the last Hog Rider was played (battle time), for the Ice Spirit follow-up.
    last_hog: Option<Duration>,
    /// Elixir is free for the first seconds; don't spend it all before anything happens.
    pub opening_wait: Duration,
}

impl Default for HogCycle {
    fn default() -> Self {
        Self { lane: Lane::Right, hog_at: 7, leak_at: 9, last_hog: None, opening_wait: Duration::from_secs(8) }
    }
}

const FOLLOW_UP: Duration = Duration::from_millis(1500);

/// Defensive / cycle cards in preference order, with their tile.
fn defense_tile(card: &str, lane: Lane) -> Option<(u32, u32)> {
    Some(match card {
        // Center, pulls Hog Riders and other building-targeters.
        "cannon" => (8, 21),
        // Cycle cards near the king, safe and cheap.
        "skeletons" => (8, 24),
        "ice_golem" => (9, 24),
        "musketeer" => lane.behind_tower(),
        _ => return None,
    })
}

impl HogCycle {
    fn deploy(state: &GameState, card: &str, (col, row): (u32, u32), enemy_half: bool) -> Option<Action> {
        let slot = state.slot_of(card)?;
        (state.elixir >= cost(card)?).then(|| Action::Deploy { slot, card: card.into(), col, row, enemy_half })
    }
}

impl Policy for HogCycle {
    fn decide(&mut self, s: &GameState) -> Option<Action> {
        if !s.in_battle || s.battle_time < self.opening_wait.min(Duration::from_secs(2)) {
            return None;
        }
        // Ice Spirit right behind a fresh Hog Rider.
        if let Some(t) = self.last_hog
            && s.battle_time.saturating_sub(t) < FOLLOW_UP
            && let Some(a) = Self::deploy(s, "ice_spirit", self.lane.bridge(), false)
        {
            return Some(a);
        }
        if s.battle_time < self.opening_wait && s.elixir < 10 {
            return None;
        }
        if s.elixir >= self.hog_at
            && let Some(a) = Self::deploy(s, "hog_rider", self.lane.bridge(), false)
        {
            return Some(a);
        }
        if s.elixir >= 10
            && let Some(a) = Self::deploy(s, "fireball", self.lane.enemy_tower(), true)
        {
            return Some(a);
        }
        if s.elixir >= self.leak_at {
            for card in ["cannon", "musketeer", "ice_golem", "skeletons"] {
                if let Some(tile) = defense_tile(card, self.lane)
                    && let Some(a) = Self::deploy(s, card, tile, false)
                {
                    return Some(a);
                }
            }
            // Last resort to not leak: cycle the cheapest card at the king.
            if let Some(a) = Self::deploy(s, "ice_spirit", (9, 24), false) {
                return Some(a);
            }
        }
        None
    }

    fn on_action(&mut self, action: &Action, s: &GameState) {
        let Action::Deploy { card, .. } = action;
        if card == "hog_rider" {
            self.last_hog = Some(s.battle_time);
        } else if card == "ice_spirit" {
            self.last_hog = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(elixir: u8, hand: [&str; 4], secs: u64) -> GameState {
        GameState {
            in_battle: true,
            elixir,
            elixir_fill: elixir as f32,
            hand: hand.map(|c| (!c.is_empty()).then(|| c.to_string())),
            next: None,
            battle_time: Duration::from_secs(secs),
        }
    }

    fn card_of(a: Option<Action>) -> Option<String> {
        a.map(|Action::Deploy { card, .. }| card)
    }

    #[test]
    fn waits_during_opening_unless_full() {
        let mut p = HogCycle::default();
        assert_eq!(p.decide(&state(8, ["hog_rider", "cannon", "", ""], 3)), None);
        assert_eq!(card_of(p.decide(&state(10, ["hog_rider", "cannon", "", ""], 3))).as_deref(), Some("hog_rider"));
    }

    #[test]
    fn hog_then_ice_spirit_follow_up() {
        let mut p = HogCycle::default();
        let s = state(7, ["ice_spirit", "hog_rider", "cannon", "the_log"], 20);
        let a = p.decide(&s).unwrap();
        assert_eq!(a, Action::Deploy { slot: 1, card: "hog_rider".into(), col: 14, row: 17, enemy_half: false });
        p.on_action(&a, &s);
        let s2 = state(3, ["ice_spirit", "", "cannon", "the_log"], 21);
        assert_eq!(card_of(p.decide(&s2)).as_deref(), Some("ice_spirit"));
        // Too late for the follow-up: nothing at 3 elixir.
        let s3 = state(3, ["ice_spirit", "", "cannon", "the_log"], 25);
        assert_eq!(p.decide(&s3), None);
    }

    #[test]
    fn avoids_leaking_with_defense_in_center() {
        let mut p = HogCycle::default();
        let a = p.decide(&state(9, ["skeletons", "cannon", "the_log", "fireball"], 30)).unwrap();
        assert_eq!(a, Action::Deploy { slot: 1, card: "cannon".into(), col: 8, row: 21, enemy_half: false });
    }

    #[test]
    fn fireball_tower_at_full_elixir() {
        let mut p = HogCycle::default();
        let a = p.decide(&state(10, ["the_log", "fireball", "", ""], 30)).unwrap();
        assert_eq!(a, Action::Deploy { slot: 1, card: "fireball".into(), col: 14, row: 6, enemy_half: true });
    }

    #[test]
    fn nothing_when_not_in_battle_or_unaffordable() {
        let mut p = HogCycle::default();
        let mut s = state(10, ["hog_rider", "", "", ""], 30);
        s.in_battle = false;
        assert_eq!(p.decide(&s), None);
        assert_eq!(p.decide(&state(5, ["hog_rider", "musketeer", "", ""], 30)), None);
    }
}
