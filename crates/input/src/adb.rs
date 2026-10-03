use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use anyhow::{Context, bail};

use crate::TapBackend;

/// One long-lived `adb shell` with commands written to its stdin, so each tap costs only
/// the on-device `input` command, not a new adb connection.
///
/// Every batch ends with an `echo` marker; reading up to the marker tells us the device
/// has finished, which makes tap latency measurable.
pub struct AdbShell {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    seq: u64,
    size: (u32, u32),
}

impl AdbShell {
    /// `adb` binary and optional device serial (needed when several devices are attached).
    pub fn open(adb: &str, serial: Option<&str>) -> anyhow::Result<Self> {
        let mut cmd = Command::new(adb);
        if let Some(s) = serial {
            cmd.args(["-s", s]);
        }
        let mut child = cmd
            .args(["shell", "sh"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("spawn {adb} shell"))?;
        let stdin = child.stdin.take().context("no stdin")?;
        let stdout = BufReader::new(child.stdout.take().context("no stdout")?);
        let mut shell = Self { child, stdin, stdout, seq: 0, size: (0, 0) };
        let out = shell.run("wm size")?;
        shell.size = parse_wm_size(&out).with_context(|| format!("parse `wm size`: {out:?}"))?;
        Ok(shell)
    }

    /// Runs a shell command line and returns its stdout (up to the completion marker).
    pub fn run(&mut self, command: &str) -> anyhow::Result<String> {
        self.seq += 1;
        let marker = format!("__cr_done_{}", self.seq);
        writeln!(self.stdin, "{command}; echo {marker}")?;
        self.stdin.flush()?;
        let mut out = String::new();
        loop {
            let mut line = String::new();
            if self.stdout.read_line(&mut line)? == 0 {
                bail!("adb shell exited");
            }
            if line.trim_end() == marker {
                return Ok(out);
            }
            out.push_str(&line);
        }
    }
}

/// Uses the override size if set (`wm size 720x1600`), else the physical size.
fn parse_wm_size(out: &str) -> Option<(u32, u32)> {
    let find = |prefix: &str| {
        out.lines().find_map(|l| {
            let (w, h) = l.trim().strip_prefix(prefix)?.trim().split_once('x')?;
            Some((w.parse().ok()?, h.parse().ok()?))
        })
    };
    find("Override size:").or_else(|| find("Physical size:"))
}

impl TapBackend for AdbShell {
    fn screen_size(&self) -> (u32, u32) {
        self.size
    }

    fn taps(&mut self, points: &[(u32, u32)]) -> anyhow::Result<()> {
        let line = points.iter().map(|(x, y)| format!("input tap {x} {y}")).collect::<Vec<_>>().join("; ");
        let out = self.run(&line)?;
        // `input` prints nothing on success; anything else is an error (e.g. missing INJECT_EVENTS).
        if !out.trim().is_empty() {
            bail!("input tap failed: {}", out.trim());
        }
        Ok(())
    }
}

impl Drop for AdbShell {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "exit");
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::parse_wm_size;

    #[test]
    fn wm_size_physical_and_override() {
        assert_eq!(parse_wm_size("Physical size: 1080x2400\n"), Some((1080, 2400)));
        assert_eq!(parse_wm_size("Physical size: 1080x2400\nOverride size: 720x1600\n"), Some((720, 1600)));
        assert_eq!(parse_wm_size("garbage"), None);
    }
}
