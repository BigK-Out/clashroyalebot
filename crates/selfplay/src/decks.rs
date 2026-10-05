//! Sparring decks: every round shows every card once; champions spread one per deck.

pub fn plan_decks(seed: u64, rounds: usize) -> Vec<Vec<String>> {
    let cards = brain::cards();
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut out = Vec::new();
    for _ in 0..rounds {
        let mut champs: Vec<String> = Vec::new();
        let mut rest: Vec<String> = Vec::new();
        for c in cards.iter().filter(|c| c.kind.as_deref() != Some("Tower Troop")) {
            if c.rarity.as_deref() == Some("Champion") { champs.push(c.slug.clone()) } else { rest.push(c.slug.clone()) }
        }
        champs.sort();
        rest.sort();
        rng.shuffle(&mut champs);
        rng.shuffle(&mut rest);
        let n = (champs.len() + rest.len()).div_ceil(8);
        assert!(champs.len() <= n, "more champions than decks");
        let mut decks: Vec<Vec<String>> = (0..n).map(|i| champs.get(i).cloned().into_iter().collect()).collect();
        for c in rest {
            let d = decks.iter_mut().filter(|d| d.len() < 8).min_by_key(|d| d.len()).unwrap();
            d.push(c);
        }
        // Fill short decks with non-champions from other decks of this round.
        let pool: Vec<String> = decks.iter().flatten().filter(|c| cards.get(c).and_then(|x| x.rarity.as_deref()) != Some("Champion")).cloned().collect();
        for d in decks.iter_mut() {
            while d.len() < 8 {
                let c = &pool[rng.usize(..pool.len())];
                if !d.contains(c) {
                    d.push(c.clone());
                }
            }
        }
        out.extend(decks);
    }
    out
}

/// Cards the first big collection missed or that the classifier reads worst: Mirror (never
/// played before its rule), Three Musketeers (9 elixir) and the fast spells.
pub const FOCUS: [&str; 8] = ["mirror", "three_musketeers", "barbarian_barrel", "rocket", "goblin_barrel", "the_log", "fireball", "giant_snowball"];

/// Decks for a focus round: one per champion, each with 4 of the FOCUS cards (every focus
/// card in half the decks) and 3 random other cards.
pub fn plan_focus_decks(seed: u64, rounds: usize) -> Vec<Vec<String>> {
    let cards = brain::cards();
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut champs: Vec<String> = cards.iter().filter(|c| c.rarity.as_deref() == Some("Champion")).map(|c| c.slug.clone()).collect();
    let mut rest: Vec<String> = cards
        .iter()
        .filter(|c| c.kind.as_deref() != Some("Tower Troop") && c.rarity.as_deref() != Some("Champion") && !FOCUS.contains(&c.slug.as_str()))
        .map(|c| c.slug.clone())
        .collect();
    champs.sort();
    rest.sort();
    let mut out = Vec::new();
    for _ in 0..rounds {
        rng.shuffle(&mut champs);
        for (i, champ) in champs.iter().enumerate() {
            let mut d = vec![champ.clone()];
            d.extend((0..4).map(|k| FOCUS[(i + 2 * k) % FOCUS.len()].to_string()));
            while d.len() < 8 {
                let c = &rest[rng.usize(..rest.len())];
                if !d.contains(c) {
                    d.push(c.clone());
                }
            }
            out.push(d);
        }
    }
    out
}

/// The game's copy-deck link for `deck`: opening it on the phone shows "Copy to Deck N".
pub fn copy_deck_link(deck: &[String]) -> anyhow::Result<String> {
    anyhow::ensure!(deck.len() == 8, "a deck has 8 cards, got {}", deck.len());
    let cards = brain::cards();
    // A champion in slot 1 (an evolution slot) makes the server drop the connection; slot 2
    // is the hero slot.
    let mut deck = deck.to_vec();
    if let Some(i) = deck.iter().position(|s| cards.get(s).and_then(|c| c.rarity.as_deref()) == Some("Champion")) {
        let champ = deck.remove(i);
        deck.insert(1, champ);
    }
    let ids = deck
        .iter()
        .map(|s| cards.get(s).and_then(|c| c.id).map(|id| id.to_string()).ok_or_else(|| anyhow::anyhow!("no card id for {s}")))
        .collect::<anyhow::Result<Vec<_>>>()?;
    // Without the tower troop (tt, Tower Princess) the game closes the popup and copies nothing.
    Ok(format!("nullsroyale://copyDeck?deck={}&l=Royals&tt=159000000", ids.join(";")))
}

/// The 8 cards on the Decks screen (scrolled to the top), row by row: card rects measured on
/// 576-px-wide screenshots (card frame without the "Max" band, like the Info portraits).
pub fn read_deck(frame: &capture::Frame, lib: &vision::CardLibrary) -> Vec<Option<String>> {
    const COLS: [f64; 4] = [20.0, 159.0, 298.0, 437.0];
    const ROWS: [f64; 2] = [318.0, 543.0];
    const SIZE: (f64, f64) = (113.0, 138.0);
    ROWS.iter()
        .flat_map(|&y| COLS.iter().map(move |&x| (x, y)))
        .map(|(x, y)| {
            // The Decks screen is laid out from the top: same pixels on 576x1248 and 576x1280.
            let fh = frame.height as f64;
            let rect = calib::NRect { x: x / 576.0, y: y / fh, w: SIZE.0 / 576.0, h: SIZE.1 / fh };
            lib.read_card(frame, rect).name().map(str::to_string)
        })
        .collect()
}

/// The deck read back from the Decks screen is `want`: every recognized card belongs to it,
/// none twice, and at least 5 are recognized. A copy is all or nothing, so 5 cards of the new
/// deck prove it; the gold hero slot and evolution frames often do not match the portraits.
pub fn deck_matches(read: &[Option<String>], want: &[String]) -> bool {
    let known: Vec<&String> = read.iter().flatten().collect();
    let distinct: std::collections::HashSet<&String> = known.iter().copied().collect();
    known.len() >= 5 && distinct.len() == known.len() && known.iter().all(|c| want.contains(c))
}

#[cfg(test)]
mod tests {
    use super::plan_decks;
    use std::collections::HashSet;

    fn champion(slug: &str) -> bool {
        brain::cards().get(slug).and_then(|c| c.rarity.clone()).as_deref() == Some("Champion")
    }

    #[test]
    fn one_round_covers_every_card_in_valid_decks() {
        let decks = plan_decks(1, 1);
        let playable: HashSet<String> = brain::cards()
            .iter()
            .filter(|c| c.kind.as_deref() != Some("Tower Troop"))
            .map(|c| c.slug.clone())
            .collect();
        assert_eq!(playable.len(), 122);
        let seen: HashSet<String> = decks.iter().flatten().cloned().collect();
        assert_eq!(seen, playable);
        assert_eq!(decks.len(), 16); // ceil(122 / 8)
        for d in &decks {
            assert_eq!(d.len(), 8);
            assert_eq!(d.iter().collect::<HashSet<_>>().len(), 8, "distinct: {d:?}");
            assert!(d.iter().filter(|c| champion(c)).count() <= 1, "{d:?}");
        }
    }

    #[test]
    fn copy_deck_link_uses_card_ids_in_order() {
        let deck: Vec<String> = ["hog_rider", "musketeer", "cannon", "ice_golem", "skeletons", "ice_spirit", "the_log", "fireball"]
            .map(String::from)
            .to_vec();
        assert_eq!(
            super::copy_deck_link(&deck).unwrap(),
            "nullsroyale://copyDeck?deck=26000021;26000014;27000000;26000038;26000010;26000030;28000011;28000000&l=Royals&tt=159000000"
        );
    }

    #[test]
    fn copy_deck_link_rejects_bad_decks() {
        let seven: Vec<String> = plan_decks(1, 1)[0][..7].to_vec();
        assert!(super::copy_deck_link(&seven).is_err());
        let mut unknown = plan_decks(1, 1)[0].clone();
        unknown[0] = "not_a_card".into();
        assert!(super::copy_deck_link(&unknown).is_err());
    }

    #[test]
    fn every_planned_card_has_an_id() {
        for d in plan_decks(3, 1) {
            super::copy_deck_link(&d).unwrap();
        }
    }

    #[test]
    fn reads_the_deck_back_from_the_decks_screen() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let deck = ["hog_rider", "musketeer", "cannon", "ice_golem", "skeletons", "ice_spirit", "the_log", "fireball"];
        let lib = vision::CardLibrary::load_subset(root.join("assets/cards_all"), &deck).unwrap();
        let f = capture::load_rgb(root.join("fixtures/screens/note9/deck_view.png")).unwrap();
        let got = super::read_deck(&f, &lib);
        let want = ["skeletons", "hog_rider", "ice_golem", "ice_spirit", "fireball", "cannon", "musketeer", "the_log"];
        assert_eq!(got, want.map(|s| Some(s.to_string())).to_vec());
    }

    #[test]
    fn champion_goes_to_the_second_slot() {
        // In slot 1 (an evolution slot) the server drops the connection; slot 2 is the hero slot.
        let deck: Vec<String> = ["mighty_miner", "void", "spear_goblins", "poison", "royal_giant", "ice_wizard", "mortar", "bomb_tower"]
            .map(String::from)
            .to_vec();
        let link = super::copy_deck_link(&deck).unwrap();
        let ids: Vec<&str> = link.split("deck=").nth(1).unwrap().split('&').next().unwrap().split(';').collect();
        let mm = brain::cards().get("mighty_miner").unwrap().id.unwrap().to_string();
        assert_eq!(ids[1], mm, "{link}");
        assert_eq!(ids.len(), 8);
    }

    #[test]
    fn reads_new_cards_with_ribbons() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let deck = ["void", "mighty_miner", "spear_goblins", "poison", "royal_giant", "ice_wizard", "mortar", "bomb_tower"];
        let lib = vision::CardLibrary::load_subset(root.join("assets/cards_all"), &deck).unwrap();
        let f = capture::load_rgb(root.join("fixtures/screens/note9/deck_view_new.png")).unwrap();
        let want: Vec<String> = deck.map(String::from).to_vec();
        assert!(super::deck_matches(&super::read_deck(&f, &lib), &want));
        // The old Hog deck on screen must not pass as this one.
        let hog = capture::load_rgb(root.join("fixtures/screens/note9/deck_view.png")).unwrap();
        assert!(!super::deck_matches(&super::read_deck(&hog, &lib), &want));
    }

    #[test]
    fn reads_the_deck_on_the_note14_too() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let deck = ["hog_rider", "musketeer", "cannon", "ice_golem", "skeletons", "ice_spirit", "the_log", "fireball"];
        let lib = vision::CardLibrary::load_subset(root.join("assets/cards_all"), &deck).unwrap();
        let f = capture::load_rgb(root.join("fixtures/screens/note14/deck_view.png")).unwrap();
        let read = super::read_deck(&f, &lib);
        assert!(read.iter().flatten().count() >= 7, "{read:?}");
        assert!(super::deck_matches(&read, &deck.map(String::from)));
    }

    #[test]
    fn hero_and_evolution_frames_still_verify() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let deck = ["executioner", "goblinstein", "goblins", "suspicious_bush", "mortar", "rascals", "rocket", "pekka"];
        let lib = vision::CardLibrary::load_subset(root.join("assets/cards_all"), &deck).unwrap();
        let want: Vec<String> = deck.map(String::from).to_vec();
        let f = capture::load_rgb(root.join("fixtures/screens/note9/deck_view_goblinstein.png")).unwrap();
        assert!(super::deck_matches(&super::read_deck(&f, &lib), &want));
        let f14 = capture::load_rgb(root.join("fixtures/screens/note14/deck_view_goblinstein.png")).unwrap();
        assert!(super::deck_matches(&super::read_deck(&f14, &lib), &want), "{:?}", super::read_deck(&f14, &lib));
        let hog = capture::load_rgb(root.join("fixtures/screens/note9/deck_view.png")).unwrap();
        assert!(!super::deck_matches(&super::read_deck(&hog, &lib), &want), "{:?}", super::read_deck(&hog, &lib));
    }

    #[test]
    fn focus_decks_cover_champions_and_weak_cards() {
        let decks = super::plan_focus_decks(2, 1);
        assert_eq!(decks.len(), 8);
        let champs: HashSet<String> = decks.iter().flatten().filter(|c| champion(c)).cloned().collect();
        assert_eq!(champs.len(), 8, "every champion once per round");
        for d in &decks {
            assert_eq!(d.iter().collect::<HashSet<_>>().len(), 8, "{d:?}");
            assert_eq!(d.iter().filter(|c| champion(c)).count(), 1, "{d:?}");
            super::copy_deck_link(d).unwrap();
        }
        for f in super::FOCUS {
            assert!(decks.iter().filter(|d| d.contains(&f.to_string())).count() >= 4, "{f}");
        }
    }

    #[test]
    fn rounds_differ_and_are_reproducible() {
        assert_eq!(plan_decks(5, 2), plan_decks(5, 2));
        let d = plan_decks(5, 2);
        assert_eq!(d.len(), 32);
        assert_ne!(d[..16], d[16..]);
    }
}
