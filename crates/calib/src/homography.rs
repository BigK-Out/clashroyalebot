/// 3x3 projective transform, normalized so h[2][2] = 1.
#[derive(Debug, Clone, Copy)]
pub struct Homography {
    h: [[f64; 3]; 3],
}

impl Homography {
    /// Solves the transform mapping each `src[i]` to `dst[i]` (4-point DLT).
    /// Returns None when the points are degenerate (three collinear, duplicates).
    pub fn from_points(src: [(f64, f64); 4], dst: [(f64, f64); 4]) -> Option<Self> {
        // Unknowns h00 h01 h02 h10 h11 h12 h20 h21 (h22 = 1); two equations per point.
        let mut a = [[0.0f64; 9]; 8];
        for i in 0..4 {
            let (x, y) = src[i];
            let (u, v) = dst[i];
            a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
            a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
        }
        let s = solve8(a)?;
        Some(Self { h: [[s[0], s[1], s[2]], [s[3], s[4], s[5]], [s[6], s[7], 1.0]] })
    }

    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let h = &self.h;
        let w = h[2][0] * x + h[2][1] * y + h[2][2];
        ((h[0][0] * x + h[0][1] * y + h[0][2]) / w, (h[1][0] * x + h[1][1] * y + h[1][2]) / w)
    }
}

/// Gaussian elimination with partial pivoting on an 8x9 augmented matrix.
fn solve8(mut a: [[f64; 9]; 8]) -> Option<[f64; 8]> {
    for col in 0..8 {
        let pivot = (col..8).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        for row in 0..8 {
            if row != col {
                let f = a[row][col] / a[col][col];
                for k in col..9 {
                    a[row][k] -= f * a[col][k];
                }
            }
        }
    }
    let mut x = [0.0; 8];
    for i in 0..8 {
        x[i] = a[i][8] / a[i][i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity() {
        let pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let h = Homography::from_points(pts, pts).unwrap();
        let (x, y) = h.apply(0.3, 0.7);
        assert!((x - 0.3).abs() < 1e-12 && (y - 0.7).abs() < 1e-12);
    }

    #[test]
    fn inverse_composes_to_identity() {
        let a = [(0.0, 0.0), (18.0, 0.0), (18.0, 32.0), (0.0, 32.0)];
        let b = [(57.0, 120.0), (663.0, 118.0), (705.0, 1000.0), (15.0, 1002.0)];
        let fwd = Homography::from_points(a, b).unwrap();
        let inv = Homography::from_points(b, a).unwrap();
        for (x, y) in [(3.2, 7.9), (17.5, 0.5), (9.0, 16.0)] {
            let (u, v) = fwd.apply(x, y);
            let (x2, y2) = inv.apply(u, v);
            assert!((x - x2).abs() < 1e-9 && (y - y2).abs() < 1e-9);
        }
    }

    #[test]
    fn collinear_is_degenerate() {
        let src = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)];
        let dst = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!(Homography::from_points(src, dst).is_none());
    }
}
