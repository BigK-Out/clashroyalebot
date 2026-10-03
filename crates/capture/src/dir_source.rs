use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, ensure};

use crate::{Frame, FrameSource};

/// Replays PNG/JPEG files from a directory in filename order, looping forever.
/// Offline stand-in for the live device: tests, recorded matches, frame dumps.
pub struct DirSource {
    dir: PathBuf,
    files: Vec<PathBuf>,
    idx: usize,
    period: Option<Duration>,
    next_due: Instant,
    seq: u64,
}

impl DirSource {
    /// `fps = None` returns frames as fast as they decode.
    pub fn open(dir: impl AsRef<Path>, fps: Option<f64>) -> anyhow::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .with_context(|| format!("read {}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg"))
            })
            .collect();
        files.sort();
        ensure!(!files.is_empty(), "no png/jpg files in {}", dir.display());
        Ok(Self {
            dir,
            files,
            idx: 0,
            period: fps.map(|f| Duration::from_secs_f64(1.0 / f)),
            next_due: Instant::now(),
            seq: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }
}

impl FrameSource for DirSource {
    fn next_frame(&mut self) -> anyhow::Result<Frame> {
        if let Some(period) = self.period {
            let now = Instant::now();
            if self.next_due > now {
                std::thread::sleep(self.next_due - now);
            }
            self.next_due = self.next_due.max(now) + period;
        }
        let path = &self.files[self.idx];
        self.idx = (self.idx + 1) % self.files.len();
        let mut frame = crate::load_rgb(path)?;
        self.seq += 1;
        frame.seq = self.seq;
        Ok(frame)
    }

    fn describe(&self) -> String {
        format!("{} ({} images)", self.dir.display(), self.files.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replays_in_order_and_loops() {
        let dir = std::env::temp_dir().join(format!("capture-dirsource-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (name, v) in [("b.png", 200u8), ("a.png", 100), ("ignore.txt", 0)] {
            if name.ends_with(".png") {
                image::RgbImage::from_pixel(4, 2, image::Rgb([v, v, v])).save(dir.join(name)).unwrap();
            } else {
                std::fs::write(dir.join(name), "x").unwrap();
            }
        }
        let mut src = DirSource::open(&dir, None).unwrap();
        assert_eq!(src.len(), 2);
        let firsts: Vec<u8> = (0..3).map(|_| src.next_frame().unwrap().pixel(0, 0)[0]).collect();
        assert_eq!(firsts, [100, 200, 100]);
        let f = src.next_frame().unwrap();
        assert_eq!((f.width, f.height, f.rgb.len(), f.seq), (4, 2, 24, 4));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn empty_dir_errors() {
        let dir = std::env::temp_dir().join(format!("capture-empty-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(DirSource::open(&dir, None).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
