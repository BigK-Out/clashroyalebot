//! Self-play data factory: sparring policy, play logs, clock alignment, deck rotation.

pub mod record;

pub use record::{JsonlLog, MatchMeta, ObserverRecord, PlayRecord, host_ms, read_jsonl};
