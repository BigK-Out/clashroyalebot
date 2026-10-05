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

/// Our Hog Rider's counters: buildings it gets pulled into, tank killers, and Tornado.
fn is_hog_counter(card: &str) -> bool {
    card == "tornado"
        || cards().get(card).is_some_and(|c| ["building_defense", "building_siege", "tank_killer"].iter().any(|r| c.has_role(r)))
}

#[derive(serde::Deserialize)]
struct Archetype {
    cards: Vec<String>,
    #[serde(default = "one")]
    weight: u32,
}

fn one() -> u32 {
    1
}

fn archetypes() -> &'static std::collections::BTreeMap<String, Archetype> {
    static A: std::sync::OnceLock<std::collections::BTreeMap<String, Archetype>> = std::sync::OnceLock::new();
    A.get_or_init(|| toml::from_str(include_str!("../../../assets/archetypes.toml")).expect("assets/archetypes.toml"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InHand {
    /// Played within the last 4 plays: back in their queue.
    No,
    /// Seen before and cycled back.
    Likely,
    /// Never seen.
    Unknown,
}

/// The opponent's deck as far as we have seen it, in play order. A played card goes to the
/// back of their 4-card queue: it is out of hand until 4 other cards have been played.
#[derive(Debug, Clone, Default)]
pub struct EnemyDeck {
    plays: Vec<String>,
}

impl EnemyDeck {
    pub fn on_play(&mut self, card: &str) {
        self.plays.push(card.to_string());
    }

    /// Distinct cards seen, sorted.
    pub fn known(&self) -> Vec<&str> {
        let mut k: Vec<&str> = self.plays.iter().map(String::as_str).collect();
        k.sort();
        k.dedup();
        k
    }

    pub fn in_hand(&self, card: &str) -> InHand {
        match self.plays.iter().rposition(|p| p == card) {
            None => InHand::Unknown,
            Some(i) if self.plays.len() - 1 - i < 4 => InHand::No,
            Some(_) => InHand::Likely,
        }
    }

    /// Best-matching archetype (assets/archetypes.toml), once its score reaches 2.
    pub fn archetype(&self) -> Option<&'static str> {
        let known = self.known();
        archetypes()
            .iter()
            .map(|(name, a)| (name.as_str(), known.iter().filter(|k| a.cards.iter().any(|c| c == *k)).count() as u32 * a.weight))
            .filter(|(_, score)| *score >= 2)
            .max_by_key(|(_, score)| *score)
            .map(|(name, _)| name)
    }

    /// Known cards of theirs that stop a Hog Rider.
    pub fn hog_counters(&self) -> Vec<&str> {
        self.known().into_iter().filter(|c| is_hog_counter(c)).collect()
    }

    /// A known Hog counter is (likely) in their hand right now.
    pub fn hog_counter_in_hand(&self) -> bool {
        self.hog_counters().iter().any(|c| self.in_hand(c) == InHand::Likely)
    }
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

    fn deck(plays: &[&str]) -> EnemyDeck {
        let mut d = EnemyDeck::default();
        for p in plays {
            d.on_play(p);
        }
        d
    }

    #[test]
    fn played_card_is_out_of_hand_until_four_other_plays() {
        let d = deck(&["tesla", "knight", "fireball"]);
        assert_eq!(d.in_hand("tesla"), InHand::No);
        let d = deck(&["tesla", "knight", "fireball", "zap", "archers"]);
        assert_eq!(d.in_hand("tesla"), InHand::Likely, "back in hand after 4 other plays");
        assert_eq!(d.in_hand("golem"), InHand::Unknown, "never seen");
    }

    #[test]
    fn mirror_does_not_count_as_a_deck_card() {
        let d = deck(&["hog_rider", "mirror", "cannon"]);
        assert_eq!(d.known(), vec!["cannon", "hog_rider", "mirror"]);
    }

    #[test]
    fn archetype_from_signature_cards() {
        assert_eq!(deck(&["golem", "baby_dragon", "elixir_collector"]).archetype(), Some("beatdown"));
        assert_eq!(deck(&["goblin_barrel", "princess", "knight"]).archetype(), Some("bait"));
        assert_eq!(deck(&["x_bow", "tesla", "archers"]).archetype(), Some("siege"));
        assert_eq!(deck(&["knight"]).archetype(), None, "one ordinary card says nothing");
    }

    #[test]
    fn hog_counters_in_hand() {
        // Their Tesla was played 1 play ago: out of cycle. Mini PEKKA never seen: unknown.
        let d = deck(&["knight", "tesla"]);
        assert_eq!(d.hog_counters(), vec!["tesla"]);
        assert!(!d.hog_counter_in_hand(), "the only known counter is out of cycle");
        let d = deck(&["tesla", "knight", "archers", "zap", "fireball"]);
        assert!(d.hog_counter_in_hand());
    }

    #[test]
    fn air_units_are_known() {
        assert!(is_air("balloon") && is_air("minions") && is_air("lava_hound"));
        assert!(!is_air("hog_rider") && !is_air("fireball") && !is_air("giant"));
    }
}
