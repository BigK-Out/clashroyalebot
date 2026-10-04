//! Self-play data factory: sparring policy, play logs, clock alignment, deck rotation.

pub mod frames;
pub mod policy;
pub mod record;

pub use frames::FrameSchedule;
pub use policy::{SparPlay, SparringPolicy};
pub use record::{JsonlLog, MatchMeta, ObserverRecord, PlayRecord, host_ms, read_jsonl};
