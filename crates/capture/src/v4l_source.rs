use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, bail};
use v4l::buffer::Type;
use v4l::io::mmap::Stream;
use v4l::io::traits::CaptureStream;
use v4l::video::Capture;
use v4l::{Device, FourCC};

use crate::{Frame, FrameSource, yuv};

/// Live frames from a v4l2 capture device (scrcpy `--v4l2-sink` → v4l2loopback).
///
/// The pixel format is whatever the producer set; scrcpy writes `YU12`.
/// With `exclusive_caps=1` the device only advertises capture while scrcpy is running,
/// so `open` fails until then.
pub struct V4lSource {
    path: PathBuf,
    stream: Stream<'static>,
    width: usize,
    height: usize,
    fourcc: FourCC,
    seq: u64,
}

const YU12: [u8; 4] = *b"YU12";
const YUYV: [u8; 4] = *b"YUYV";
const RGB3: [u8; 4] = *b"RGB3";

impl V4lSource {
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let dev = Device::with_path(&path).with_context(|| format!("open {}", path.display()))?;
        let caps = dev.query_caps()?;
        if !caps.capabilities.contains(v4l::capability::Flags::VIDEO_CAPTURE) {
            bail!(
                "{} has no capture capability yet (is scrcpy running with --v4l2-sink={}?)",
                path.display(),
                path.display()
            );
        }
        let fmt = dev.format()?;
        if ![YU12, YUYV, RGB3].contains(&fmt.fourcc.repr) {
            bail!("unsupported pixel format {} on {}", fmt.fourcc, path.display());
        }
        // 4 buffers: enough to never stall the producer, few enough to keep latency low.
        let stream = Stream::with_buffers(&dev, Type::VideoCapture, 4)?;
        Ok(Self {
            path,
            stream,
            width: fmt.width as usize,
            height: fmt.height as usize,
            fourcc: fmt.fourcc,
            seq: 0,
        })
    }
}

impl FrameSource for V4lSource {
    fn next_frame(&mut self) -> anyhow::Result<Frame> {
        let (buf, meta) = self.stream.next()?;
        let captured_at = Instant::now();
        let data = &buf[..meta.bytesused as usize];
        let mut rgb = Vec::new();
        match self.fourcc.repr {
            YU12 => yuv::i420_to_rgb(data, self.width, self.height, &mut rgb)?,
            YUYV => yuv::yuyv_to_rgb(data, self.width, self.height, &mut rgb)?,
            RGB3 => rgb.extend_from_slice(&data[..self.width * self.height * 3]),
            _ => bail!("unsupported pixel format {}", self.fourcc),
        }
        self.seq += 1;
        Ok(Frame {
            seq: self.seq,
            width: self.width as u32,
            height: self.height as u32,
            rgb,
            captured_at,
        })
    }

    fn describe(&self) -> String {
        format!("{} {}x{} {}", self.path.display(), self.width, self.height, self.fourcc)
    }
}
