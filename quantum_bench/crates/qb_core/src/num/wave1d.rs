//! A complex wavefunction ψ(x) on a [`Grid1D`] and its momentum amplitude Φ(p).
//!
//! Units ħ = m = 1. Conventions:
//!
//! ```text
//! ∫|ψ|² dx ≈ Σ_i |ψ_i|² dx
//! Φ(p) = (2π)^(−1/2) ∫ ψ(x) e^(−ipx) dx        (absolute x, so a packet at x0
//! ψ(x) = (2π)^(−1/2) ∫ Φ(p) e^(+ipx) dp         carries the phase e^(−ipx0))
//! Σ_k |Φ_k|² dp = Σ_i |ψ_i|² dx                 (Parseval, exact on the grid)
//! ```

use std::f64::consts::TAU;

use super::{C64, Grid1D, fft};

/// Complex wavefunction samples ψ_i = ψ(x_i).
#[derive(Clone, Debug)]
pub struct Wave1D {
    pub grid: Grid1D,
    pub psi: Vec<C64>,
}

/// Momentum-space amplitude Φ(p) on the ascending momentum grid of `grid`.
#[derive(Clone, Debug)]
pub struct Momentum {
    /// the position grid this came from (defines p and dp)
    pub grid: Grid1D,
    /// momenta, ascending: p_k = (k − n/2)·dp
    pub p: Vec<f64>,
    /// Φ(p_k)
    pub phi: Vec<C64>,
    pub dp: f64,
}

/// Expectation values and spreads of a state (from one FFT).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Moments {
    /// ∫|ψ|² dx
    pub norm: f64,
    pub x: f64,
    pub x2: f64,
    pub sigma_x: f64,
    pub p: f64,
    pub p2: f64,
    pub sigma_p: f64,
    /// covariance ⟨(xp + px)/2⟩ − ⟨x⟩⟨p⟩ (0 for an unchirped Gaussian)
    pub cov_xp: f64,
}

impl Moments {
    /// σ_x·σ_p (≥ ½ for ħ = 1).
    pub fn uncertainty_product(&self) -> f64 {
        self.sigma_x * self.sigma_p
    }

    /// σ_x after a further time `dt` of **free** evolution (m = 1), exact for any state:
    /// σ_x²(t + dt) = σ_x² + 2·cov_xp·dt + σ_p²·dt².
    pub fn free_sigma_x(&self, dt: f64) -> f64 {
        (self.sigma_x * self.sigma_x + 2.0 * self.cov_xp * dt + self.sigma_p * self.sigma_p * dt * dt)
            .max(0.0)
            .sqrt()
    }
}

const ZERO: C64 = C64::new(0.0, 0.0);

impl Wave1D {
    /// ψ = 0.
    pub fn zeros(grid: Grid1D) -> Self {
        Wave1D {
            grid,
            psi: vec![ZERO; grid.n],
        }
    }

    /// ψ_i = f(x_i) (not normalised).
    pub fn from_fn(grid: Grid1D, f: impl Fn(f64) -> C64) -> Self {
        Wave1D { grid, psi: grid.sample(f) }
    }

    /// ψ_i = f(x_i) for a real function (not normalised).
    pub fn from_real_fn(grid: Grid1D, f: impl Fn(f64) -> f64) -> Self {
        Wave1D {
            grid,
            psi: grid.sample(|x| C64::new(f(x), 0.0)),
        }
    }

    /// Wraps given samples (length must be `grid.n`).
    pub fn from_samples(grid: Grid1D, psi: Vec<C64>) -> Self {
        assert_eq!(psi.len(), grid.n, "Wave1D::from_samples: wrong length");
        Wave1D { grid, psi }
    }

    /// Normalised Gaussian packet with ⟨x⟩ = x0, σ_x = sigma, ⟨p⟩ = p0:
    /// ψ = (2πσ²)^(−1/4) exp(−(x−x0)²/(4σ²) + i p0 x).
    ///
    /// (PS10 38 writes A e^(−a x²): that is sigma = 1/(2√a).)
    pub fn gaussian(grid: Grid1D, x0: f64, sigma: f64, p0: f64) -> Self {
        let a = (TAU * sigma * sigma).powf(-0.25);
        let mut w = Self::from_fn(grid, |x| {
            let d = x - x0;
            C64::from_polar(a * (-d * d / (4.0 * sigma * sigma)).exp(), p0 * x)
        });
        w.normalize();
        w
    }

    /// Truncated plane wave (PS09 37): ψ = e^(i p0 x)/√(2·half_width) for
    /// |x − x0| < half_width, 0 outside. Normalised on the grid.
    pub fn plane_wave_truncated(grid: Grid1D, x0: f64, half_width: f64, p0: f64) -> Self {
        let a = 1.0 / (2.0 * half_width).sqrt();
        let mut w = Self::from_fn(grid, |x| {
            if (x - x0).abs() < half_width {
                C64::from_polar(a, p0 * x)
            } else {
                ZERO
            }
        });
        w.normalize();
        w
    }

    /// Real eigenstate samples (e.g. from an [`crate::num::EigenBasis`]) as a wave.
    pub fn from_real(grid: Grid1D, values: &[f64]) -> Self {
        assert_eq!(values.len(), grid.n);
        Wave1D {
            grid,
            psi: values.iter().map(|&v| C64::new(v, 0.0)).collect(),
        }
    }

    /// Positions of the grid points.
    pub fn xs(&self) -> Vec<f64> {
        self.grid.xs()
    }

    /// ∫|ψ|² dx.
    pub fn norm_sqr(&self) -> f64 {
        self.psi.iter().map(|c| c.norm_sqr()).sum::<f64>() * self.grid.dx
    }

    /// √∫|ψ|² dx.
    pub fn norm(&self) -> f64 {
        self.norm_sqr().sqrt()
    }

    /// Scales ψ to ∫|ψ|² dx = 1 and returns the previous ∫|ψ|² dx.
    /// A zero wave is left unchanged.
    pub fn normalize(&mut self) -> f64 {
        let n2 = self.norm_sqr();
        if n2 > 0.0 && n2.is_finite() {
            let s = 1.0 / n2.sqrt();
            self.psi.iter_mut().for_each(|c| *c *= s);
        }
        n2
    }

    /// |ψ_i|².
    pub fn density(&self) -> Vec<f64> {
        self.psi.iter().map(|c| c.norm_sqr()).collect()
    }

    /// ∫_a^b |ψ|² dx (cells partly inside count with their overlap).
    pub fn prob_between(&self, a: f64, b: f64) -> f64 {
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        let g = self.grid;
        let mut s = 0.0;
        for (i, c) in self.psi.iter().enumerate() {
            let lo = g.x(i) - 0.5 * g.dx;
            let overlap = (b.min(lo + g.dx) - a.max(lo)).max(0.0);
            s += c.norm_sqr() * overlap;
        }
        s
    }

    /// ⟨x⟩ (divided by the norm).
    pub fn mean_x(&self) -> f64 {
        self.x_moments().1
    }

    /// σ_x = √(⟨x²⟩ − ⟨x⟩²).
    pub fn sigma_x(&self) -> f64 {
        let (_, x, x2) = self.x_moments();
        (x2 - x * x).max(0.0).sqrt()
    }

    /// (∫|ψ|², ⟨x⟩, ⟨x²⟩)
    fn x_moments(&self) -> (f64, f64, f64) {
        let g = self.grid;
        let (mut s0, mut s1, mut s2) = (0.0, 0.0, 0.0);
        for (i, c) in self.psi.iter().enumerate() {
            let w = c.norm_sqr();
            let x = g.x(i);
            s0 += w;
            s1 += w * x;
            s2 += w * x * x;
        }
        if s0 <= 0.0 {
            return (0.0, 0.0, 0.0);
        }
        (s0 * g.dx, s1 / s0, s2 / s0)
    }

    /// Φ(p) on the ascending momentum grid (one FFT).
    pub fn momentum(&self) -> Momentum {
        let g = self.grid;
        let mut buf = self.psi.clone();
        fft::forward(&mut buf);
        let scale = g.dx / TAU.sqrt();
        let p: Vec<f64> = g.ps();
        let phi = (0..g.n)
            .map(|k| buf[g.fft_index(k)] * C64::from_polar(scale, -p[k] * g.x_min))
            .collect();
        Momentum {
            grid: g,
            p,
            phi,
            dp: g.dp(),
        }
    }

    /// The wave whose momentum amplitude is `m` (inverse of [`Self::momentum`]).
    pub fn from_momentum(m: &Momentum) -> Self {
        let g = m.grid;
        let mut buf = vec![ZERO; g.n];
        for k in 0..g.n {
            buf[g.fft_index(k)] = m.phi[k] * C64::from_polar(1.0, m.p[k] * g.x_min);
        }
        fft::inverse(&mut buf);
        let scale = m.dp / TAU.sqrt();
        buf.iter_mut().for_each(|c| *c *= scale);
        Wave1D { grid: g, psi: buf }
    }

    /// ⟨x⟩, σ_x, ⟨p⟩, σ_p, the x–p covariance and the norm (two FFTs).
    pub fn moments(&self) -> Moments {
        let (norm, x, x2) = self.x_moments();
        let m = self.momentum();
        let (_, p, p2) = m.moments();
        // ⟨ψ| x p |ψ⟩ with pψ from the spectrum; its real part is ⟨(xp + px)/2⟩
        let dpsi = self.p_applied(&m);
        let g = self.grid;
        let n2: f64 = self.psi.iter().map(|c| c.norm_sqr()).sum();
        let xp: f64 = if n2 > 0.0 {
            self.psi
                .iter()
                .zip(&dpsi)
                .enumerate()
                .map(|(i, (a, b))| (a.conj() * b).re * g.x(i))
                .sum::<f64>()
                / n2
        } else {
            0.0
        };
        Moments {
            norm,
            x,
            x2,
            sigma_x: (x2 - x * x).max(0.0).sqrt(),
            p,
            p2,
            sigma_p: (p2 - p * p).max(0.0).sqrt(),
            cov_xp: xp - x * p,
        }
    }

    /// p̂ψ = −i ∂ψ/∂x, computed spectrally (exact for band-limited ψ).
    pub fn p_psi(&self) -> Vec<C64> {
        self.p_applied(&self.momentum())
    }

    fn p_applied(&self, m: &Momentum) -> Vec<C64> {
        let mut pm = m.clone();
        pm.phi.iter_mut().zip(&m.p).for_each(|(c, p)| *c *= *p);
        Wave1D::from_momentum(&pm).psi
    }

    /// ⟨p⟩ (one FFT).
    pub fn mean_p(&self) -> f64 {
        self.momentum().mean()
    }

    /// σ_p (one FFT).
    pub fn sigma_p(&self) -> f64 {
        self.momentum().sigma()
    }

    /// ⟨H⟩ = ⟨p²⟩/2 + ⟨V⟩ for a real potential sampled on the grid (per unit norm).
    pub fn energy(&self, v: &[f64]) -> f64 {
        let (n, _, p2) = self.momentum().moments();
        if n <= 0.0 {
            return 0.0;
        }
        let pot: f64 = self.psi.iter().zip(v).map(|(c, v)| c.norm_sqr() * v).sum::<f64>() * self.grid.dx / n;
        0.5 * p2 + pot
    }

    /// ⟨self|other⟩ = ∫ ψ* φ dx.
    pub fn inner(&self, other: &Wave1D) -> C64 {
        self.psi.iter().zip(&other.psi).map(|(a, b)| a.conj() * b).sum::<C64>() * self.grid.dx
    }

    /// ψ ← c·ψ.
    pub fn scale(&mut self, c: C64) {
        self.psi.iter_mut().for_each(|z| *z *= c);
    }

    /// ψ ← ψ + c·other.
    pub fn add_scaled(&mut self, c: C64, other: &Wave1D) {
        for (a, b) in self.psi.iter_mut().zip(&other.psi) {
            *a += c * b;
        }
    }

    /// Probability current j = Im(ψ* ∂ψ/∂x) (central differences, periodic).
    pub fn current(&self) -> Vec<f64> {
        let n = self.grid.n;
        let inv = 1.0 / (2.0 * self.grid.dx);
        (0..n)
            .map(|i| {
                let d = (self.psi[(i + 1) % n] - self.psi[(i + n - 1) % n]) * inv;
                (self.psi[i].conj() * d).im
            })
            .collect()
    }

    /// Linear interpolation of ψ at an arbitrary x (0 outside the grid).
    pub fn at(&self, x: f64) -> C64 {
        let f = self.grid.index_f(x);
        if f < 0.0 || f > (self.grid.n - 1) as f64 {
            return ZERO;
        }
        let i = f.floor() as usize;
        let t = f - i as f64;
        if i + 1 >= self.grid.n {
            return self.psi[i];
        }
        self.psi[i] * (1.0 - t) + self.psi[i + 1] * t
    }
}

impl Momentum {
    /// |Φ(p_k)|².
    pub fn density(&self) -> Vec<f64> {
        self.phi.iter().map(|c| c.norm_sqr()).collect()
    }

    /// ∫|Φ|² dp.
    pub fn norm_sqr(&self) -> f64 {
        self.phi.iter().map(|c| c.norm_sqr()).sum::<f64>() * self.dp
    }

    /// (∫|Φ|² dp, ⟨p⟩, ⟨p²⟩)
    pub fn moments(&self) -> (f64, f64, f64) {
        let (mut s0, mut s1, mut s2) = (0.0, 0.0, 0.0);
        for (c, p) in self.phi.iter().zip(&self.p) {
            let w = c.norm_sqr();
            s0 += w;
            s1 += w * p;
            s2 += w * p * p;
        }
        if s0 <= 0.0 {
            return (0.0, 0.0, 0.0);
        }
        (s0 * self.dp, s1 / s0, s2 / s0)
    }

    /// ⟨p⟩.
    pub fn mean(&self) -> f64 {
        self.moments().1
    }

    /// σ_p.
    pub fn sigma(&self) -> f64 {
        let (_, p, p2) = self.moments();
        (p2 - p * p).max(0.0).sqrt()
    }

    /// ∫_a^b |Φ|² dp.
    pub fn prob_between(&self, a: f64, b: f64) -> f64 {
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        let mut s = 0.0;
        for (c, p) in self.phi.iter().zip(&self.p) {
            let lo = p - 0.5 * self.dp;
            s += c.norm_sqr() * (b.min(lo + self.dp) - a.max(lo)).max(0.0);
        }
        s
    }
}

/// Analytic width of a free Gaussian packet (ħ = m = 1): σ(t) = σ0 √(1 + (t/(2σ0²))²).
pub fn free_gaussian_sigma(sigma0: f64, t: f64) -> f64 {
    sigma0 * (1.0 + (t / (2.0 * sigma0 * sigma0)).powi(2)).sqrt()
}

/// Analytic free Gaussian ψ(x, t) that starts as [`Wave1D::gaussian`]`(x0, σ0, p0)` at t = 0.
pub fn free_gaussian_psi(x: f64, t: f64, x0: f64, sigma0: f64, p0: f64) -> C64 {
    // ψ = (2πσ0²)^(−1/4) (1 + i t/(2σ0²))^(−1/2) exp(−(x−x0−p0 t)²/(4σ0²(1 + i t/(2σ0²))) + i p0 (x − p0 t/2))
    let s2 = sigma0 * sigma0;
    let z = C64::new(1.0, t / (2.0 * s2));
    let d = x - x0 - p0 * t;
    let pre = (TAU * s2).powf(-0.25) / z.sqrt();
    pre * (C64::new(-d * d / (4.0 * s2), 0.0) / z + C64::new(0.0, p0 * (x - 0.5 * p0 * t))).exp()
}
