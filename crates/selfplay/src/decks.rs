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
    fn rounds_differ_and_are_reproducible() {
        assert_eq!(plan_decks(5, 2), plan_decks(5, 2));
        let d = plan_decks(5, 2);
        assert_eq!(d.len(), 32);
        assert_ne!(d[..16], d[16..]);
    }
}
