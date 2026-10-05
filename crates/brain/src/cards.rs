//! Card table: elixir cost and swarm flag per card, from `assets/cards.json` (OCR'd from the
//! in-game Info screens by `training/parse_cards.py`). Embedded at compile time, so the bot
//! needs no extra file at runtime; rebuild after regenerating the JSON.

use std::collections::HashMap;
use std::sync::OnceLock;

use serde::Deserialize;

const JSON: &str = include_str!("../../../assets/cards.json");

#[derive(Debug, Clone, Deserialize)]
pub struct Card {
    pub name: String,
    /// Card id used everywhere else in the bot ("hog_rider", "mini_pekka").
    pub slug: String,
    /// `None` for Mirror (costs one more than the mirrored card).
    pub elixir: Option<u8>,
    #[serde(default)]
    pub rarity: Option<String>,
    /// "Troop", "Spell", "Building", "Tower Troop".
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    /// Units put down by one card (Skeleton Army 15, Goblins 4); 1 if unknown.
    #[serde(default = "one")]
    pub count: u32,
    /// Several units per card: a group of them counts as one card.
    #[serde(default)]
    pub swarm: bool,
    /// What the card does in a deck (see `ROLES`); several per card.
    #[serde(default)]
    pub roles: Vec<String>,
    /// The game's card id (26000021 = Hog Rider), for deck links. `None` for tower troops.
    #[serde(default)]
    pub id: Option<u32>,
}

/// Card roles (from the card-role guide): win conditions fast/slow, tanks, tank killers,
/// support, swarms, cycle cards, spells by size, buildings by job.
pub const ROLES: [&str; 15] = [
    "win_fast", "win_slow", "tank", "mini_tank", "tank_killer", "support", "swarm", "cycle",
    "spell_small", "spell_medium", "spell_big", "building_defense", "building_spawner", "building_siege", "pump",
];

impl Card {
    pub fn has_role(&self, role: &str) -> bool {
        self.roles.iter().any(|r| r == role)
    }
}

fn one() -> u32 {
    1
}

#[derive(Debug)]
pub struct Cards {
    by_slug: HashMap<String, Card>,
}

impl Cards {
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let list: Vec<Card> = serde_json::from_str(json)?;
        Ok(Self { by_slug: list.into_iter().map(|c| (c.slug.clone(), c)).collect() })
    }

    pub fn get(&self, slug: &str) -> Option<&Card> {
        self.by_slug.get(slug)
    }

    pub fn len(&self) -> usize {
        self.by_slug.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_slug.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Card> {
        self.by_slug.values()
    }
}

/// The embedded card table.
pub fn cards() -> &'static Cards {
    static CARDS: OnceLock<Cards> = OnceLock::new();
    CARDS.get_or_init(|| Cards::from_json(JSON).expect("assets/cards.json is valid"))
}

#[cfg(test)]
mod role_tests {
    use super::*;

    #[test]
    fn every_playable_card_has_known_roles() {
        // Mirror has no role of its own: it copies the last card played.
        for c in cards().iter().filter(|c| c.kind.as_deref() != Some("Tower Troop") && c.slug != "mirror") {
            assert!(!c.roles.is_empty(), "{} has no role", c.slug);
            for r in &c.roles {
                assert!(ROLES.contains(&r.as_str()), "{}: unknown role {r}", c.slug);
            }
        }
    }

    #[test]
    fn roles_of_our_deck() {
        let has = |s: &str, r: &str| cards().get(s).unwrap().has_role(r);
        assert!(has("hog_rider", "win_fast") && has("golem", "win_slow") && has("golem", "tank"));
        assert!(has("cannon", "building_defense") && has("the_log", "spell_small") && has("fireball", "spell_medium"));
        assert!(has("ice_golem", "mini_tank") && has("skeletons", "cycle") && has("musketeer", "support"));
        assert!(has("inferno_tower", "tank_killer") && has("x_bow", "building_siege") && has("elixir_collector", "pump"));
    }
}
