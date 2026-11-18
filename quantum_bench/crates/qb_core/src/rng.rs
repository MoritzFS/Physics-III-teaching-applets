//! A small, self-contained, seedable random number generator.
//!
//! `QbRng` is xoshiro256++ seeded through SplitMix64. It works the same on the
//! desktop and in the browser (no `getrandom`), and a given seed always gives
//! the same sequence, so screenshots with `QB_SEED` are reproducible.
//!
//! The shell owns one `QbRng` per tab and hands it to the tab as `ctx.rng`.

use std::f64::consts::TAU;

/// Seedable PRNG (xoshiro256++). Not cryptographic.
#[derive(Clone, Debug)]
pub struct QbRng {
    s: [u64; 4],
}

impl Default for QbRng {
    fn default() -> Self {
        Self::seed(0x5EED_2026_0F9B)
    }
}

impl QbRng {
    /// A generator whose sequence is fixed by `seed`.
    pub fn seed(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        QbRng {
            s: [next(), next(), next(), next()],
        }
    }

    /// A generator seeded from the clock (for interactive use when no seed is wanted).
    pub fn from_time() -> Self {
        let t = web_time::SystemTime::now()
            .duration_since(web_time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(1);
        Self::seed(t)
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = (s[0].wrapping_add(s[3])).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in [a, b).
    pub fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.uniform()
    }

    /// Uniform integer in 0..n (n > 0).
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "QbRng::below(0)");
        // Lemire's multiply-shift; the tiny bias for huge n does not matter here
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    /// true with probability p.
    pub fn bool(&mut self, p: f64) -> bool {
        self.uniform() < p
    }

    /// Standard normal N(0, 1) (Box–Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.uniform(); // (0, 1]
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }

    /// Normal with mean `mu` and standard deviation `sigma`.
    pub fn gauss(&mut self, mu: f64, sigma: f64) -> f64 {
        mu + sigma * self.normal()
    }

    /// Exponential waiting time with the given rate (mean 1/rate).
    pub fn exponential(&mut self, rate: f64) -> f64 {
        -(1.0 - self.uniform()).ln() / rate
    }

    /// Poisson-distributed count with mean `lambda`.
    pub fn poisson(&mut self, lambda: f64) -> u64 {
        if lambda <= 0.0 {
            return 0;
        }
        if lambda < 30.0 {
            // Knuth
            let l = (-lambda).exp();
            let mut k = 0;
            let mut p = 1.0;
            loop {
                p *= self.uniform();
                if p <= l {
                    return k;
                }
                k += 1;
            }
        }
        // normal approximation with continuity correction
        (lambda + lambda.sqrt() * self.normal() + 0.5).floor().max(0.0) as u64
    }

    /// Number of successes in `n` trials with probability `p` (exact, O(n)).
    pub fn binomial(&mut self, n: u64, p: f64) -> u64 {
        (0..n).filter(|_| self.uniform() < p).count() as u64
    }

    /// Index drawn with probability proportional to `weights` (need not be normalised).
    /// Returns `None` if all weights are zero (or the slice is empty).
    pub fn weighted(&mut self, weights: &[f64]) -> Option<usize> {
        let total: f64 = weights.iter().map(|w| w.max(0.0)).sum();
        if total <= 0.0 || !total.is_finite() {
            return None;
        }
        let mut r = self.uniform() * total;
        for (i, w) in weights.iter().enumerate() {
            let w = w.max(0.0);
            if r < w {
                return Some(i);
            }
            r -= w;
        }
        weights.iter().rposition(|w| *w > 0.0)
    }

    /// Fisher–Yates shuffle.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            items.swap(i, self.below(i + 1));
        }
    }

    /// An independent generator derived from this one (e.g. one per worker or per copy).
    pub fn fork(&mut self) -> QbRng {
        QbRng::seed(self.next_u64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproducible_and_reasonable() {
        let mut a = QbRng::seed(42);
        let mut b = QbRng::seed(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let n = 200_000;
        let (mut s, mut s2) = (0.0, 0.0);
        for _ in 0..n {
            let x = a.normal();
            s += x;
            s2 += x * x;
        }
        let mean = s / n as f64;
        let var = s2 / n as f64 - mean * mean;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.01, "var {var}");
        let u: f64 = (0..n).map(|_| a.uniform()).sum::<f64>() / n as f64;
        assert!((u - 0.5).abs() < 0.005);
        let mut counts = [0usize; 3];
        for _ in 0..n {
            counts[a.weighted(&[1.0, 2.0, 1.0]).unwrap()] += 1;
        }
        assert!((counts[1] as f64 / n as f64 - 0.5).abs() < 0.01);
    }
}
