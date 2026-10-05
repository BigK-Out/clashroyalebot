//! Labels and logs: one JSON object per line, written as they happen (crash-safe).

use std::io::Write;
use std::marker::PhantomData;
use std::path::Path;

use anyhow::Context;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// Host wall-clock time in ms since the Unix epoch: the one clock both phones' logs share.
pub fn host_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// One card the sparring phone played (the label for an enemy play on the observer).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PlayRecord {
    /// When the deploy taps finished on the sparring phone.
    pub host_ms: u64,
    pub card: String,
    pub slot: usize,
    /// Tile in the sparring phone's own view (its half is rows 17..31).
    pub tile: (u32, u32),
    pub hand_before: [Option<String>; 4],
    pub elixir_read: Option<u8>,
    /// The played slot's card left the hand afterwards (the tap really played this card).
    pub verified: bool,
}

/// Something the observer bot did or saw.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ObserverRecord {
    pub host_ms: u64,
    /// "deploy", "battle_start", "battle_end", "new_enemy".
    pub kind: String,
    pub card: Option<String>,
    pub tile: Option<(u32, u32)>,
    pub enemies: Vec<(u32, u32)>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct MatchMeta {
    pub match_id: String,
    pub observer_serial: String,
    pub sparring_serial: String,
    pub observer_deck: Vec<String>,
    pub sparring_deck: Vec<String>,
    /// Observer time minus sparring tap time for the same play (Task 7).
    pub clock_offset_ms: Option<i64>,
    pub complete: bool,
    pub error: Option<String>,
    /// Crowns at the end, read from the observer's result screen (bot evaluation runs).
    #[serde(default)]
    pub observer_crowns: Option<u8>,
    #[serde(default)]
    pub sparring_crowns: Option<u8>,
    /// The bot binary the observer ran.
    #[serde(default)]
    pub bot: Option<String>,
}

impl MatchMeta {
    pub fn save(&self, dir: &Path) -> anyhow::Result<()> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("meta.json"), serde_json::to_string_pretty(self)?)?;
        Ok(())
    }
}

pub struct JsonlLog<T> {
    file: std::fs::File,
    _t: PhantomData<T>,
}

impl<T: Serialize> JsonlLog<T> {
    /// Creates (or appends to) a JSONL file; parent directories are created.
    pub fn create(path: &Path) -> anyhow::Result<Self> {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("open {}", path.display()))?;
        Ok(Self { file, _t: PhantomData })
    }

    pub fn append(&mut self, rec: &T) -> anyhow::Result<()> {
        let mut line = serde_json::to_string(rec)?;
        line.push('\n');
        self.file.write_all(line.as_bytes())?;
        self.file.flush()?;
        Ok(())
    }
}

pub fn read_jsonl<T: DeserializeOwned>(path: &Path) -> anyhow::Result<Vec<T>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    text.lines().filter(|l| !l.trim().is_empty()).map(|l| Ok(serde_json::from_str(l)?)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn play(ms: u64, card: &str) -> PlayRecord {
        PlayRecord {
            host_ms: ms,
            card: card.into(),
            slot: 2,
            tile: (3, 17),
            hand_before: [Some("hog_rider".into()), None, Some(card.into()), Some("the_log".into())],
            elixir_read: Some(7),
            verified: true,
        }
    }

    #[test]
    fn jsonl_roundtrip_and_append() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("plays.jsonl");
        let mut log = JsonlLog::create(&path).unwrap();
        log.append(&play(1, "musketeer")).unwrap();
        log.append(&play(2, "cannon")).unwrap();
        let back: Vec<PlayRecord> = read_jsonl(&path).unwrap();
        assert_eq!(back, vec![play(1, "musketeer"), play(2, "cannon")]);
        // One JSON object per line, readable by Python's json.loads per line.
        assert_eq!(std::fs::read_to_string(&path).unwrap().lines().count(), 2);
    }

    #[test]
    fn meta_saves_json() {
        let dir = tempfile::tempdir().unwrap();
        let m = MatchMeta { match_id: "m1".into(), complete: false, error: Some("adb gone".into()), ..Default::default() };
        m.save(dir.path()).unwrap();
        let back: MatchMeta = serde_json::from_str(&std::fs::read_to_string(dir.path().join("meta.json")).unwrap()).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn incomplete_match_marked() {
        let dir = tempfile::tempdir().unwrap();
        MatchMeta { match_id: "x".into(), error: Some("sparring exit status: 1".into()), ..Default::default() }.save(dir.path()).unwrap();
        let m: MatchMeta = serde_json::from_str(&std::fs::read_to_string(dir.path().join("meta.json")).unwrap()).unwrap();
        assert!(!m.complete && m.error.is_some());
    }

    #[test]
    fn host_clock_is_epoch_ms() {
        assert!(host_ms() > 1_700_000_000_000);
    }
}

