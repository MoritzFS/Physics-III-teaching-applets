//! Measurements: Born-rule sampling, collapse, and a histogram of outcomes.
//!
//! Measurement UX used in every tab:
//! * **single shot** ("Measure", key M): one outcome, the displayed state
//!   collapses and keeps evolving from there;
//! * **repeat N**: N fresh copies of the *current* state are measured (the
//!   displayed state is untouched) and the outcomes fill a [`ShotHistogram`],
//!   drawn against the Born-rule prediction with
//!   [`crate::widgets::HistogramPlot`].
//!
//! Position/momentum measurements have a finite detector resolution: the
//! outcome is drawn from |ψ|² (|Φ|²), then ψ (Φ) is multiplied by a Gaussian
//! window exp(−(x − x0)²/(4w²)) and renormalised, so the collapsed |ψ|² has a
//! Gaussian envelope of standard deviation w. (Drawing x0 from |ψ|² is the
//! ideal-pointer approximation; it is exact as w → 0.)

use super::{C64, EigenExpansion, Momentum, Wave1D};
use crate::QbRng;

/// Draws continuous values from a density given on a uniform grid
/// (piecewise constant: point i stands for the cell [x_i − dx/2, x_i + dx/2]).
#[derive(Clone, Debug)]
pub struct Sampler {
    x0: f64,
    dx: f64,
    cdf: Vec<f64>,
}

impl Sampler {
    /// `xs` must be uniform. Returns `None` if the density is zero everywhere.
    pub fn new(xs: &[f64], density: &[f64]) -> Option<Self> {
        if xs.len() < 2 || xs.len() != density.len() {
            return None;
        }
        let mut cdf = Vec::with_capacity(density.len());
        let mut s = 0.0;
        for d in density {
            s += d.max(0.0);
            cdf.push(s);
        }
        if s <= 0.0 || !s.is_finite() {
            return None;
        }
        cdf.iter_mut().for_each(|c| *c /= s);
        Some(Sampler {
            x0: xs[0],
            dx: xs[1] - xs[0],
            cdf,
        })
    }

    /// One sample.
    pub fn sample(&self, rng: &mut QbRng) -> f64 {
        let u = rng.uniform();
        let i = self.cdf.partition_point(|&c| c <= u).min(self.cdf.len() - 1);
        self.x0 + (i as f64 + rng.uniform() - 0.5) * self.dx
    }

    /// `n` samples.
    pub fn samples(&self, n: usize, rng: &mut QbRng) -> Vec<f64> {
        (0..n).map(|_| self.sample(rng)).collect()
    }
}

/// One position outcome drawn from |ψ|² (no collapse). `None` for ψ = 0.
pub fn born_sample_x(wave: &Wave1D, rng: &mut QbRng) -> Option<f64> {
    Sampler::new(&wave.grid.xs(), &wave.density()).map(|s| s.sample(rng))
}

/// `n` position outcomes on fresh copies of `wave` ("repeat N").
pub fn born_samples_x(wave: &Wave1D, n: usize, rng: &mut QbRng) -> Vec<f64> {
    Sampler::new(&wave.grid.xs(), &wave.density())
        .map(|s| s.samples(n, rng))
        .unwrap_or_default()
}

/// `n` momentum outcomes on fresh copies of `wave` ("repeat N").
pub fn born_samples_p(wave: &Wave1D, n: usize, rng: &mut QbRng) -> Vec<f64> {
    let m = wave.momentum();
    Sampler::new(&m.p, &m.density()).map(|s| s.samples(n, rng)).unwrap_or_default()
}

/// Multiplies ψ by exp(−(x − x0)²/(4w²)) and renormalises (w ≥ dx is enforced).
pub fn collapse_position(wave: &mut Wave1D, x0: f64, width: f64) {
    let w = width.max(wave.grid.dx);
    let g = wave.grid;
    for (i, c) in wave.psi.iter_mut().enumerate() {
        let d = g.x(i) - x0;
        *c *= (-d * d / (4.0 * w * w)).exp();
    }
    wave.normalize();
}

/// Multiplies Φ by exp(−(p − p0)²/(4w²)) and renormalises (w ≥ dp is enforced).
pub fn collapse_momentum(wave: &mut Wave1D, p0: f64, width: f64) {
    let mut m: Momentum = wave.momentum();
    let w = width.max(m.dp);
    for (c, p) in m.phi.iter_mut().zip(&m.p) {
        let d = p - p0;
        *c *= (-d * d / (4.0 * w * w)).exp();
    }
    *wave = Wave1D::from_momentum(&m);
    wave.normalize();
}

/// Position measurement with a detector of resolution `width`: returns the
/// outcome x0 and collapses ψ around it. For ψ = 0 nothing happens and the
/// result is NaN.
pub fn measure_position(wave: &mut Wave1D, width: f64, rng: &mut QbRng) -> f64 {
    let Some(x0) = born_sample_x(wave, rng) else { return f64::NAN };
    collapse_position(wave, x0, width);
    x0
}

/// Momentum measurement with resolution `width` (in p): returns p0 and collapses Φ around it.
pub fn measure_momentum(wave: &mut Wave1D, width: f64, rng: &mut QbRng) -> f64 {
    let m = wave.momentum();
    let Some(s) = Sampler::new(&m.p, &m.density()) else {
        return f64::NAN;
    };
    let p0 = s.sample(rng);
    collapse_momentum(wave, p0, width);
    p0
}

/// Energy measurement: draws n with probability |c_n|²/Σ|c|² and collapses
/// onto ψ_n. Returns `None` if all coefficients vanish.
pub fn measure_energy(expansion: &mut EigenExpansion, rng: &mut QbRng) -> Option<usize> {
    let n = rng.weighted(&expansion.probabilities())?;
    expansion.collapse_to(n);
    Some(n)
}

/// "Is the particle in [a, b]?" Answers yes with probability ∫_a^b |ψ|²
/// (per unit norm), then projects ψ onto the region (yes) or its complement
/// (no) and renormalises. The projection is sharp, so the collapsed ψ has
/// steps at a and b (and hence high momenta).
pub fn measure_region(wave: &mut Wave1D, a: f64, b: f64, rng: &mut QbRng) -> bool {
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    let total = wave.norm_sqr();
    let p_in = if total > 0.0 { wave.prob_between(a, b) / total } else { 0.0 };
    let yes = rng.bool(p_in);
    let g = wave.grid;
    for (i, c) in wave.psi.iter_mut().enumerate() {
        let inside = (a..=b).contains(&g.x(i));
        if inside != yes {
            *c = C64::new(0.0, 0.0);
        }
    }
    wave.normalize();
    yes
}

/// Bins of a [`ShotHistogram`].
#[derive(Clone, Debug, PartialEq)]
pub enum Bins {
    /// `count` equal bins on [lo, hi)
    Range { lo: f64, hi: f64, count: usize },
    /// outcomes 0 … count−1 (energy levels, spin up/down, yes/no, …)
    Discrete { count: usize },
}

/// Counts of repeated measurement outcomes.
#[derive(Clone, Debug)]
pub struct ShotHistogram {
    pub bins: Bins,
    pub counts: Vec<u64>,
    /// outcomes outside the range (binned histograms only)
    pub outside: u64,
    sum: f64,
    sum2: f64,
    /// the most recent outcome
    pub last: Option<f64>,
}

impl ShotHistogram {
    /// `count` equal bins on [lo, hi) for continuous outcomes.
    pub fn binned(lo: f64, hi: f64, count: usize) -> Self {
        let count = count.max(1);
        Self::with_bins(Bins::Range {
            lo,
            hi: hi.max(lo + 1e-12),
            count,
        })
    }

    /// One bin per discrete outcome 0 … count−1.
    pub fn discrete(count: usize) -> Self {
        Self::with_bins(Bins::Discrete { count: count.max(1) })
    }

    fn with_bins(bins: Bins) -> Self {
        let n = match bins {
            Bins::Range { count, .. } | Bins::Discrete { count } => count,
        };
        ShotHistogram {
            bins,
            counts: vec![0; n],
            outside: 0,
            sum: 0.0,
            sum2: 0.0,
            last: None,
        }
    }

    /// Number of bins.
    pub fn len(&self) -> usize {
        self.counts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.total() == 0
    }

    /// Records a continuous outcome (binned) or an index (discrete, `value as usize`).
    pub fn add(&mut self, value: f64) {
        self.last = Some(value);
        self.sum += value;
        self.sum2 += value * value;
        match self.bins {
            Bins::Range { lo, hi, count } => {
                let f = (value - lo) / (hi - lo) * count as f64;
                if f >= 0.0 && f < count as f64 {
                    self.counts[f as usize] += 1;
                } else {
                    self.outside += 1;
                }
            }
            Bins::Discrete { count } => {
                let i = value.round();
                if i >= 0.0 && (i as usize) < count {
                    self.counts[i as usize] += 1;
                } else {
                    self.outside += 1;
                }
            }
        }
    }

    /// Records a discrete outcome.
    pub fn add_index(&mut self, i: usize) {
        self.add(i as f64);
    }

    pub fn add_all(&mut self, values: &[f64]) {
        values.iter().for_each(|&v| self.add(v));
    }

    pub fn clear(&mut self) {
        self.counts.iter_mut().for_each(|c| *c = 0);
        self.outside = 0;
        self.sum = 0.0;
        self.sum2 = 0.0;
        self.last = None;
    }

    /// All recorded outcomes, including those outside the range.
    pub fn total(&self) -> u64 {
        self.counts.iter().sum::<u64>() + self.outside
    }

    /// Mean of all outcomes (NaN when empty).
    pub fn mean(&self) -> f64 {
        self.sum / self.total() as f64
    }

    /// Standard deviation of all outcomes (NaN when empty).
    pub fn std(&self) -> f64 {
        let n = self.total() as f64;
        let m = self.sum / n;
        (self.sum2 / n - m * m).max(0.0).sqrt()
    }

    /// counts / total for every bin.
    pub fn fractions(&self) -> Vec<f64> {
        let t = self.total().max(1) as f64;
        self.counts.iter().map(|&c| c as f64 / t).collect()
    }

    /// Bin edges (count + 1 values; for discrete bins i − ½ … count − ½).
    pub fn edges(&self) -> Vec<f64> {
        match self.bins {
            Bins::Range { lo, hi, count } => (0..=count).map(|i| lo + (hi - lo) * i as f64 / count as f64).collect(),
            Bins::Discrete { count } => (0..=count).map(|i| i as f64 - 0.5).collect(),
        }
    }

    /// Width of one bin (1 for discrete bins).
    pub fn bin_width(&self) -> f64 {
        match self.bins {
            Bins::Range { lo, hi, count } => (hi - lo) / count as f64,
            Bins::Discrete { .. } => 1.0,
        }
    }

    /// Predicted probability of each bin from a density sampled on a uniform
    /// grid `xs` (cells of width xs[1] − xs[0]), divided by the density's total.
    /// For a position measurement: `h.predict_from_density(&wave.xs(), &wave.density())`.
    pub fn predict_from_density(&self, xs: &[f64], density: &[f64]) -> Vec<f64> {
        let edges = self.edges();
        let mut out = vec![0.0; self.len()];
        if xs.len() < 2 {
            return out;
        }
        let dx = xs[1] - xs[0];
        let total: f64 = density.iter().map(|d| d.max(0.0)).sum();
        if total <= 0.0 {
            return out;
        }
        for (&x, &d) in xs.iter().zip(density) {
            let (lo, hi) = (x - 0.5 * dx, x + 0.5 * dx);
            let first = edges.partition_point(|&e| e <= lo).saturating_sub(1);
            for b in first..self.len() {
                if edges[b] >= hi {
                    break;
                }
                let overlap = (hi.min(edges[b + 1]) - lo.max(edges[b])).max(0.0);
                out[b] += d.max(0.0) * overlap / dx;
            }
        }
        out.iter_mut().for_each(|p| *p /= total);
        out
    }
}
