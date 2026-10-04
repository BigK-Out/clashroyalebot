//! Self-play data factory: sparring policy, play logs, clock alignment, deck rotation.

pub mod policy;
pub mod record;

pub use policy::{SparPlay, SparringPolicy};
pub use record::{JsonlLog, MatchMeta, ObserverRecord, PlayRecord, host_ms, read_jsonl};
