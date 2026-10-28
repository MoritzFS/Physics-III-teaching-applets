//! Statistics of the simulations: histograms and moments that forget old
//! samples (an exponential moving average over time), and time traces.

use std::collections::VecDeque;

/// How long the histograms remember.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Avg {
    /// only the particles as they are now
    Now,
    /// about the last 5 seconds
    Short,
    /// about the last minute
    Long,
}

impl Avg {
    pub const ALL: [Avg; 3] = [Avg::Now, Avg::Short, Avg::Long];

    /// the memory time in seconds (0: none)
    pub fn tau(self) -> f64 {
        match self {
            Avg::Now => 0.0,
            Avg::Short => 5.0,
            Avg::Long => 60.0,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Avg::Now => "now",
            Avg::Short => "~5 s",
            Avg::Long => "~1 min",
        }
    }
}

/// the factor by which old samples count after dt (tau = 0: forget everything)
fn keep(dt: f64, tau: f64) -> f64 {
    if tau <= 0.0 { 0.0 } else { (-dt / tau).exp() }
}

/// A histogram on [lo, hi) with equal bins. Samples outside count in the total.
#[derive(Clone, Debug)]
pub struct Hist {
    pub lo: f64,
    pub hi: f64,
    pub bins: Vec<f64>,
    pub total: f64,
}

impl Hist {
    pub fn new(lo: f64, hi: f64, n: usize) -> Self {
        Hist { lo, hi, bins: vec![0.0; n], total: 0.0 }
    }

    pub fn width(&self) -> f64 {
        (self.hi - self.lo) / self.bins.len() as f64
    }

    /// the centre of bin k
    pub fn centre(&self, k: usize) -> f64 {
        self.lo + (k as f64 + 0.5) * self.width()
    }

    /// Lets old samples fade before the next snapshot is added.
    pub fn fade(&mut self, dt: f64, tau: f64) {
        let f = keep(dt, tau);
        self.bins.iter_mut().for_each(|b| *b *= f);
        self.total *= f;
    }

    pub fn add(&mut self, x: f64) {
        self.total += 1.0;
        if x >= self.lo && x < self.hi {
            let last = self.bins.len() - 1;
            let k = ((x - self.lo) / self.width()) as usize;
            self.bins[k.min(last)] += 1.0;
        }
    }

    pub fn clear(&mut self) {
        self.bins.iter_mut().for_each(|b| *b = 0.0);
        self.total = 0.0;
    }

    /// probability density in bin k (normalised over all samples)
    pub fn density(&self, k: usize) -> f64 {
        if self.total > 0.0 { self.bins[k] / (self.total * self.width()) } else { 0.0 }
    }

    /// the fraction of samples in bin k
    pub fn fraction(&self, k: usize) -> f64 {
        if self.total > 0.0 { self.bins[k] / self.total } else { 0.0 }
    }
}

/// Mean and variance of samples, with the same fading as `Hist`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Moments {
    w: f64,
    s1: f64,
    s2: f64,
}

impl Moments {
    pub fn fade(&mut self, dt: f64, tau: f64) {
        let f = keep(dt, tau);
        self.w *= f;
        self.s1 *= f;
        self.s2 *= f;
    }

    pub fn add(&mut self, x: f64) {
        self.w += 1.0;
        self.s1 += x;
        self.s2 += x * x;
    }

    pub fn clear(&mut self) {
        *self = Moments::default();
    }

    pub fn mean(&self) -> Option<f64> {
        (self.w > 0.0).then(|| self.s1 / self.w)
    }

    pub fn var(&self) -> Option<f64> {
        let m = self.mean()?;
        Some((self.s2 / self.w - m * m).max(0.0))
    }
}

/// A value that follows its input with a time constant, for calm readouts.
#[derive(Clone, Copy, Debug, Default)]
pub struct Smooth(Option<f64>);

impl Smooth {
    pub fn update(&mut self, x: f64, dt: f64, tau: f64) -> f64 {
        let f = keep(dt, tau);
        let v = match self.0 {
            Some(old) if x.is_finite() => f * old + (1.0 - f) * x,
            _ => x,
        };
        self.0 = Some(v);
        v
    }

    pub fn get(&self) -> Option<f64> {
        self.0
    }

    pub fn clear(&mut self) {
        self.0 = None;
    }
}

/// The last `window` seconds of up to three values (NaN: not shown).
#[derive(Clone, Debug)]
pub struct Trace {
    pub window: f64,
    pub points: VecDeque<(f64, [f64; 3])>,
}

impl Trace {
    pub fn new(window: f64) -> Self {
        Trace { window, points: VecDeque::new() }
    }

    pub fn push(&mut self, t: f64, v: [f64; 3]) {
        self.points.push_back((t, v));
        while self.points.front().is_some_and(|p| p.0 < t - self.window) {
            self.points.pop_front();
        }
    }

    pub fn clear(&mut self) {
        self.points.clear();
    }

    /// the largest value of line k
    pub fn max(&self, k: usize) -> f64 {
        self.points.iter().map(|p| p.1[k]).filter(|v| v.is_finite()).fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_and_moments() {
        let mut h = Hist::new(0.0, 1.0, 4);
        for x in [0.1, 0.3, 0.35, 0.9, 1.5] {
            h.add(x);
        }
        assert_eq!(h.bins, vec![1.0, 2.0, 0.0, 1.0]);
        assert!((h.density(1) - 2.0 / (5.0 * 0.25)).abs() < 1e-12);
        h.fade(1.0, 0.0);
        assert_eq!(h.total, 0.0);
        let mut m = Moments::default();
        for x in [1.0, 2.0, 3.0] {
            m.add(x);
        }
        assert_eq!(m.mean(), Some(2.0));
        assert!((m.var().unwrap() - 2.0 / 3.0).abs() < 1e-12);
    }
}
