//! YUV → RGB8 conversion (BT.601, limited range: what scrcpy's H.264 decode produces).
//! Integer math; good enough for CV, ~2 ms for 720x1280 in release.

#[inline]
fn yuv_to_rgb(y: u8, u: u8, v: u8) -> [u8; 3] {
    let c = (y as i32 - 16) * 298;
    let d = u as i32 - 128;
    let e = v as i32 - 128;
    let clamp = |x: i32| ((x + 128) >> 8).clamp(0, 255) as u8;
    [
        clamp(c + 409 * e),
        clamp(c - 100 * d - 208 * e),
        clamp(c + 516 * d),
    ]
}

/// Planar 4:2:0 (V4L2 `YU12` / I420): Y plane, then U (w/2 × h/2), then V.
pub fn i420_to_rgb(src: &[u8], width: usize, height: usize, dst: &mut Vec<u8>) -> anyhow::Result<()> {
    let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
    let need = width * height + 2 * cw * ch;
    anyhow::ensure!(src.len() >= need, "I420 buffer too small: {} < {need}", src.len());
    let (y_plane, rest) = src.split_at(width * height);
    let (u_plane, v_plane) = rest.split_at(cw * ch);

    dst.resize(width * height * 3, 0);
    for row in 0..height {
        let y_row = &y_plane[row * width..][..width];
        let c_off = (row / 2) * cw;
        let out = &mut dst[row * width * 3..][..width * 3];
        for col in 0..width {
            let ci = c_off + col / 2;
            out[col * 3..col * 3 + 3].copy_from_slice(&yuv_to_rgb(y_row[col], u_plane[ci], v_plane[ci]));
        }
    }
    Ok(())
}

/// Packed 4:2:2 (V4L2 `YUYV`): Y0 U Y1 V per pixel pair.
pub fn yuyv_to_rgb(src: &[u8], width: usize, height: usize, dst: &mut Vec<u8>) -> anyhow::Result<()> {
    anyhow::ensure!(width % 2 == 0, "YUYV width must be even, got {width}");
    let need = width * height * 2;
    anyhow::ensure!(src.len() >= need, "YUYV buffer too small: {} < {need}", src.len());

    dst.resize(width * height * 3, 0);
    for (quad, out) in src[..need].chunks_exact(4).zip(dst.chunks_exact_mut(6)) {
        let [y0, u, y1, v] = [quad[0], quad[1], quad[2], quad[3]];
        out[..3].copy_from_slice(&yuv_to_rgb(y0, u, v));
        out[3..].copy_from_slice(&yuv_to_rgb(y1, u, v));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(got: [u8; 3], want: [u8; 3]) {
        for i in 0..3 {
            assert!((got[i] as i32 - want[i] as i32).abs() <= 2, "got {got:?}, want {want:?}");
        }
    }

    #[test]
    fn limited_range_extremes() {
        assert_close(yuv_to_rgb(16, 128, 128), [0, 0, 0]);
        assert_close(yuv_to_rgb(235, 128, 128), [255, 255, 255]);
        assert_close(yuv_to_rgb(126, 128, 128), [128, 128, 128]);
    }

    #[test]
    fn primary_colors() {
        // BT.601 limited-range encodings of pure red / green / blue.
        assert_close(yuv_to_rgb(81, 90, 240), [255, 0, 0]);
        assert_close(yuv_to_rgb(145, 54, 34), [0, 255, 1]);
        assert_close(yuv_to_rgb(41, 240, 110), [0, 0, 255]);
    }

    #[test]
    fn i420_layout() {
        // 2x2 image: one chroma sample shared by all four pixels; left column black, right white.
        let src = [16, 235, 16, 235, 128, 128];
        let mut out = Vec::new();
        i420_to_rgb(&src, 2, 2, &mut out).unwrap();
        assert_eq!(out.len(), 12);
        assert_close([out[0], out[1], out[2]], [0, 0, 0]);
        assert_close([out[3], out[4], out[5]], [255, 255, 255]);
        assert_close([out[6], out[7], out[8]], [0, 0, 0]);
        assert_close([out[9], out[10], out[11]], [255, 255, 255]);
    }

    #[test]
    fn i420_odd_dimensions() {
        // 3x1: chroma is ceil(3/2) x ceil(1/2) = 2x1.
        let src = [16, 126, 235, 128, 128, 128, 128];
        let mut out = Vec::new();
        i420_to_rgb(&src, 3, 1, &mut out).unwrap();
        assert_close([out[6], out[7], out[8]], [255, 255, 255]);
    }

    #[test]
    fn i420_rejects_short_buffer() {
        assert!(i420_to_rgb(&[0; 5], 2, 2, &mut Vec::new()).is_err());
    }

    #[test]
    fn yuyv_pair() {
        let src = [16, 128, 235, 128];
        let mut out = Vec::new();
        yuyv_to_rgb(&src, 2, 1, &mut out).unwrap();
        assert_close([out[0], out[1], out[2]], [0, 0, 0]);
        assert_close([out[3], out[4], out[5]], [255, 255, 255]);
    }
}
