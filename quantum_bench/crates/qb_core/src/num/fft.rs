//! FFT helpers with cached plans (rustfft). Unnormalised transforms:
//!
//! ```text
//! forward: X_k = Σ_j x_j e^{−2πi jk/n}      inverse: x_j = Σ_k X_k e^{+2πi jk/n}
//! ```
//!
//! so `inverse(forward(x)) = n·x`. 2D arrays are row-major, `index = iy·nx + ix`.

use std::cell::RefCell;
use std::sync::Arc;

use rustfft::{Fft, FftPlanner};

use super::C64;

thread_local! {
    static PLANNER: RefCell<FftPlanner<f64>> = RefCell::new(FftPlanner::new());
}

/// The (cached) plan for a transform of length `n`.
pub fn plan(n: usize, inverse: bool) -> Arc<dyn Fft<f64>> {
    PLANNER.with(|p| {
        let mut p = p.borrow_mut();
        if inverse { p.plan_fft_inverse(n) } else { p.plan_fft_forward(n) }
    })
}

/// In-place forward FFT (e^{−2πi jk/n}, unnormalised).
pub fn forward(buf: &mut [C64]) {
    if !buf.is_empty() {
        plan(buf.len(), false).process(buf);
    }
}

/// In-place inverse FFT (e^{+2πi jk/n}, unnormalised).
pub fn inverse(buf: &mut [C64]) {
    if !buf.is_empty() {
        plan(buf.len(), true).process(buf);
    }
}

/// In-place 2D FFT of a row-major `nx × ny` array (unnormalised).
/// Rows are transformed in parallel on the desktop.
pub fn fft2(data: &mut [C64], nx: usize, ny: usize, inverse: bool) {
    assert_eq!(data.len(), nx * ny, "fft2: data has the wrong size");
    let row_plan = plan(nx, inverse);
    crate::par::chunks_mut(data, nx, |_, row| row_plan.process(row));
    let col_plan = plan(ny, inverse);
    let mut t = vec![C64::new(0.0, 0.0); nx * ny];
    transpose(data, &mut t, nx, ny);
    crate::par::chunks_mut(&mut t, ny, |_, col| col_plan.process(col));
    transpose(&t, data, ny, nx);
}

/// `dst[ix·ny + iy] = src[iy·nx + ix]`.
fn transpose(src: &[C64], dst: &mut [C64], nx: usize, ny: usize) {
    const B: usize = 32;
    for by in (0..ny).step_by(B) {
        for bx in (0..nx).step_by(B) {
            for iy in by..(by + B).min(ny) {
                for ix in bx..(bx + B).min(nx) {
                    dst[ix * ny + iy] = src[iy * nx + ix];
                }
            }
        }
    }
}
