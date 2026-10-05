//! Perception: turn a frame into game facts (elixir, hand, units).

pub mod arena;
pub mod elixir;
pub mod hand;
mod patch;
pub mod result;
pub mod towers;
pub mod units;

pub use elixir::{Elixir, read_elixir};
pub use hand::{CardLibrary, CardMatch, Hand, Slot};
