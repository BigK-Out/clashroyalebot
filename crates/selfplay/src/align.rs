//! Delay between a sparring tap and the observer seeing it (network + scrcpy), per match.

use crate::{ObserverRecord, PlayRecord};

const MAX_DELAY_MS: u64 = 3_000;

pub fn estimate_offset(plays: &[PlayRecord], observed: &[ObserverRecord]) -> Option<i64> {
    let mut deltas: Vec<i64> = plays
        .iter()
        .filter(|p| p.verified)
        .filter_map(|p| {
            observed
                .iter()
                .filter(|o| o.kind == "new_enemy" && o.host_ms >= p.host_ms && o.host_ms <= p.host_ms + MAX_DELAY_MS)
                .map(|o| (o.host_ms - p.host_ms) as i64)
                .min()
        })
        .collect();
    if deltas.len() < 3 {
        return None;
    }
    deltas.sort_unstable();
    Some(deltas[deltas.len() / 2])
}

#[cfg(test)]
mod tests {
    use super::estimate_offset;
    use crate::{ObserverRecord, PlayRecord};

    fn play(ms: u64) -> PlayRecord {
        PlayRecord { host_ms: ms, card: "knight".into(), slot: 0, tile: (3, 17), hand_before: Default::default(), elixir_read: Some(5), verified: true }
    }
    fn seen(ms: u64) -> ObserverRecord {
        ObserverRecord { host_ms: ms, kind: "new_enemy".into(), card: None, tile: None, enemies: vec![] }
    }

    #[test]
    fn constant_delay() {
        let plays: Vec<_> = (0..5).map(|i| play(10_000 * i + 1_000)).collect();
        let obs: Vec<_> = (0..5).map(|i| seen(10_000 * i + 1_350)).collect();
        assert_eq!(estimate_offset(&plays, &obs), Some(350));
    }

    #[test]
    fn median_ignores_outliers() {
        let plays: Vec<_> = (0..5).map(|i| play(10_000 * i + 1_000)).collect();
        let mut obs: Vec<_> = (0..4).map(|i| seen(10_000 * i + 1_300)).collect();
        obs.push(seen(40_000 + 1_000 + 2_900)); // the 5th play was only noticed late
        assert_eq!(estimate_offset(&plays, &obs), Some(300));
    }

    #[test]
    fn too_few_matches() {
        assert_eq!(estimate_offset(&[play(1_000)], &[seen(1_300)]), None);
    }
}
