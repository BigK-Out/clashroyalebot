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
