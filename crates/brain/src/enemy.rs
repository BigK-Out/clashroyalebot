//! What we know about the opponent from recognized plays: an elixir estimate and air units.

use std::time::Duration;

use crate::cards;

/// Seconds per elixir point (single elixir); double after 2:00, triple after 4:00 (overtime).
const SECS_PER_ELIXIR: f32 = 2.8;
const DOUBLE_AT: f32 = 120.0;
const TRIPLE_AT: f32 = 240.0;

/// Flying troops: only air-targeting cards (Musketeer, Ice Spirit, spells) can stop them.
const AIR: [&str; 13] = [
    "minions", "minion_horde", "mega_minion", "baby_dragon", "inferno_dragon", "electro_dragon",
    "skeleton_dragons", "balloon", "lava_hound", "phoenix", "bats", "flying_machine", "skeleton_barrel",
];

pub fn is_air(card: &str) -> bool {
    AIR.contains(&card)
}

/// Elixir regenerated between two battle times.
fn regen(from: f32, to: f32) -> f32 {
    let rate = |t: f32| if t >= TRIPLE_AT { 3.0 } else if t >= DOUBLE_AT { 2.0 } else { 1.0 };
    let mut t = from;
    let mut gained = 0.0;
    for edge in [DOUBLE_AT, TRIPLE_AT, f32::MAX] {
        if t >= to {
            break;
        }
        if t < edge {
            let end = to.min(edge);
            gained += (end - t) * rate(t) / SECS_PER_ELIXIR;
            t = end;
        }
    }
    gained
}

/// Enemy elixir estimate: 5 at the start, regenerates like ours, recognized plays subtract
/// their cost. Clamped to 0..=10 (a missed or misread play cannot push it out of range).
#[derive(Debug, Clone)]
pub struct EnemyElixir {
    value: f32,
    at: f32,
}

impl Default for EnemyElixir {
    fn default() -> Self {
        Self { value: 5.0, at: 0.0 }
    }
}

impl EnemyElixir {
    /// The estimate at battle time `t` (advances the internal clock).
    pub fn at(&mut self, t: Duration) -> f32 {
        let t = t.as_secs_f32();
        if t > self.at {
            self.value = (self.value + regen(self.at, t)).min(10.0);
            self.at = t;
        }
        self.value
    }

    pub fn on_play(&mut self, card: &str, t: Duration) {
        self.at(t);
        let cost = cards().get(card).and_then(|c| c.elixir).unwrap_or(0) as f32;
        self.value = (self.value - cost).max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn secs(s: f32) -> Duration {
        Duration::from_secs_f32(s)
    }

    #[test]
    fn starts_at_five_and_regenerates() {
        let mut e = EnemyElixir::default();
        assert!((e.at(secs(0.0)) - 5.0).abs() < 1e-3);
        assert!((e.at(secs(2.8)) - 6.0).abs() < 0.01);
        assert!((e.at(secs(60.0)) - 10.0).abs() < 1e-3, "capped at 10");
    }

    #[test]
    fn plays_cost_elixir_and_it_cannot_go_negative() {
        let mut e = EnemyElixir::default();
        e.on_play("hog_rider", secs(0.0));
        assert!((e.at(secs(0.0)) - 1.0).abs() < 1e-3);
        e.on_play("pekka", secs(0.0)); // misread or not seen regenerating: clamp at 0
        assert_eq!(e.at(secs(0.0)), 0.0);
    }

    #[test]
    fn double_elixir_after_two_minutes() {
        let mut e = EnemyElixir::default();
        e.on_play("golem", secs(119.0)); // 8 elixir: down to 2
        let before = e.at(secs(119.0));
        let after = e.at(secs(121.8)); // 1 s single + 1.8 s double
        assert!((after - before - (1.0 / 2.8 + 1.8 * 2.0 / 2.8)).abs() < 0.05, "{before} {after}");
    }

    #[test]
    fn air_units_are_known() {
        assert!(is_air("balloon") && is_air("minions") && is_air("lava_hound"));
        assert!(!is_air("hog_rider") && !is_air("fireball") && !is_air("giant"));
    }
}
