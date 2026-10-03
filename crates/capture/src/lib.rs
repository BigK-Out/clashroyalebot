//! Frame sources: live v4l2 (scrcpy → v4l2loopback) and image directories for offline work.

mod dir_source;
mod v4l_source;
pub mod yuv;

use std::time::Instant;

pub use dir_source::DirSource;
pub use v4l_source::V4lSource;

/// One RGB8 frame, row-major, no padding (`rgb.len() == width * height * 3`).
#[derive(Clone)]
pub struct Frame {
    /// Monotonic counter assigned by the source.
    pub seq: u64,
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
    /// When the frame was dequeued from the source. Start of the latency chain.
    pub captured_at: Instant,
}

impl Frame {
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 3] {
        let i = ((y * self.width + x) * 3) as usize;
        [self.rgb[i], self.rgb[i + 1], self.rgb[i + 2]]
    }
}

pub trait FrameSource: Send {
    /// Blocks until the next frame is available.
    fn next_frame(&mut self) -> anyhow::Result<Frame>;

    /// Human-readable description for logs/overlay (device, format).
    fn describe(&self) -> String;
}
