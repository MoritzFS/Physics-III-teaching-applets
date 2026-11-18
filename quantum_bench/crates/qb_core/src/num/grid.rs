//! Uniform periodic 1D grid in position, with its momentum grid.

use std::f64::consts::{PI, TAU};

/// `n` points `x_i = x_min + i·dx`, `dx = (x_max − x_min)/n`.
///
/// The grid is periodic (as the FFT assumes): `x_max` itself is not a grid
/// point, it is the image of `x_min`. `n` is always even.
///
/// The momentum grid (ħ = 1) has spacing `dp = 2π/(n·dx)` and runs in
/// ascending order from `−n/2·dp` to `(n/2 − 1)·dp`, i.e. up to ±π/dx.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid1D {
    pub n: usize,
    pub x_min: f64,
    pub dx: f64,
}

impl Grid1D {
    /// `n` (rounded up to even) points covering [x_min, x_max).
    pub fn new(x_min: f64, x_max: f64, n: usize) -> Self {
        assert!(x_max > x_min, "Grid1D: x_max must exceed x_min");
        let n = (n.max(2) + 1) & !1;
        Grid1D {
            n,
            x_min,
            dx: (x_max - x_min) / n as f64,
        }
    }

    /// Position of point `i`.
    #[inline]
    pub fn x(&self, i: usize) -> f64 {
        self.x_min + i as f64 * self.dx
    }

    /// All positions.
    pub fn xs(&self) -> Vec<f64> {
        (0..self.n).map(|i| self.x(i)).collect()
    }

    /// End of the periodic cell (not a grid point).
    pub fn x_max(&self) -> f64 {
        self.x_min + self.n as f64 * self.dx
    }

    /// Length of the periodic cell, n·dx.
    pub fn length(&self) -> f64 {
        self.n as f64 * self.dx
    }

    /// Fractional index of position `x` (may be outside 0..n).
    pub fn index_f(&self, x: f64) -> f64 {
        (x - self.x_min) / self.dx
    }

    /// Index of the grid point nearest to `x`, clamped to the grid.
    pub fn nearest(&self, x: f64) -> usize {
        (self.index_f(x).round().max(0.0) as usize).min(self.n - 1)
    }

    /// Samples `f` at every grid point.
    pub fn sample<T>(&self, f: impl Fn(f64) -> T) -> Vec<T> {
        (0..self.n).map(|i| f(self.x(i))).collect()
    }

    /// Momentum spacing 2π/(n·dx).
    pub fn dp(&self) -> f64 {
        TAU / self.length()
    }

    /// Largest momentum the grid resolves, π/dx.
    pub fn p_max(&self) -> f64 {
        PI / self.dx
    }

    /// Momentum of point `k` of the ascending momentum grid.
    #[inline]
    pub fn p(&self, k: usize) -> f64 {
        (k as f64 - (self.n / 2) as f64) * self.dp()
    }

    /// The ascending momentum grid (the order of [`crate::num::Momentum::phi`]).
    pub fn ps(&self) -> Vec<f64> {
        (0..self.n).map(|k| self.p(k)).collect()
    }

    /// Index in FFT order (0, 1, …, n/2−1, −n/2, …, −1) of ascending momentum index `k`.
    #[inline]
    pub fn fft_index(&self, k: usize) -> usize {
        (k + self.n / 2) % self.n
    }

    /// Fractional index of momentum `p` on the ascending momentum grid.
    pub fn p_index_f(&self, p: f64) -> f64 {
        p / self.dp() + (self.n / 2) as f64
    }
}
