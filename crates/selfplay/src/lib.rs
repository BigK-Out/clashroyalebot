//! Self-play data factory: sparring policy, play logs, clock alignment, deck rotation.

pub mod align;
pub mod decks;
pub mod frames;
pub mod policy;
pub mod record;
pub mod screens;

pub use align::estimate_offset;
pub use decks::{copy_deck_link, deck_matches, plan_decks, read_deck};
pub use frames::{FrameSchedule, NewEnemyGate};
pub use policy::{SparPlay, SparringPolicy};
pub use record::{JsonlLog, MatchMeta, ObserverRecord, PlayRecord, host_ms, read_jsonl};
