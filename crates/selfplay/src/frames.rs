//! Which observer frames to keep: 2 fps normally, 10 fps for 3 s after something new
//! (a sparring play or new enemy units), where the classifier needs dense frames.

const BASE_MS: u64 = 500;
const BURST_MS: u64 = 100;
const BURST_LEN_MS: u64 = 3_000;

pub struct FrameSchedule {
    last: Option<u64>,
    burst_until: u64,
}

impl FrameSchedule {
    pub fn new() -> Self {
        Self { last: None, burst_until: 0 }
    }

    pub fn burst(&mut self, now_ms: u64) {
        self.burst_until = now_ms + BURST_LEN_MS;
    }

    pub fn should_save(&mut self, now_ms: u64) -> bool {
        let every = if now_ms < self.burst_until { BURST_MS } else { BASE_MS };
        if self.last.is_some_and(|l| now_ms < l + every) {
            return false;
        }
        self.last = Some(now_ms);
        true
    }
}

impl Default for FrameSchedule {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::FrameSchedule;

    fn saved(s: &mut FrameSchedule, from: u64, to: u64) -> usize {
        (from..to).step_by(10).filter(|&t| s.should_save(t)).count()
    }

    #[test]
    fn two_fps_baseline() {
        let mut s = FrameSchedule::new();
        assert_eq!(saved(&mut s, 0, 10_000), 20);
    }

    #[test]
    fn ten_fps_burst_for_three_seconds() {
        let mut s = FrameSchedule::new();
        s.burst(1_000);
        assert_eq!(saved(&mut s, 1_000, 4_000), 30);
        assert_eq!(saved(&mut s, 4_000, 6_000), 4);
    }
}
