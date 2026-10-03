//! Perception: turn a frame into game facts (elixir, hand).

pub mod arena;
pub mod elixir;
pub mod hand;
mod patch;

pub use elixir::{Elixir, read_elixir};
pub use hand::{CardLibrary, CardMatch, Hand, Slot};
