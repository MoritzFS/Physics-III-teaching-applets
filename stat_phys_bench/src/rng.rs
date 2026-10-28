//! A small random number generator (xoshiro256++). Each simulation starts
//! from a fixed seed, so a run can be repeated and tested.

pub struct Rng {
    s: [u64; 4],
    spare: Option<f64>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // splitmix64 spreads the seed over the state
        let mut x = seed;
        let mut next = || {
            x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        Rng { s: [next(), next(), next(), next()], spare: None }
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// uniform in [0, 1)
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// uniform in (0, 1], safe to take the logarithm of
    pub fn uniform_open(&mut self) -> f64 {
        1.0 - self.uniform()
    }

    /// an integer in 0..n (n < 2³²)
    pub fn below(&mut self, n: usize) -> usize {
        (((self.next_u64() >> 32) * n as u64) >> 32) as usize
    }

    /// standard normal (Box–Muller)
    pub fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let r = (-2.0 * self.uniform_open().ln()).sqrt();
        let (s, c) = (std::f64::consts::TAU * self.uniform()).sin_cos();
        self.spare = Some(r * s);
        r * c
    }

    /// geometric: P(k) = (1 − b) bᵏ for k = 0, 1, 2, …
    pub fn geometric(&mut self, b: f64) -> u32 {
        self.geometric_ln(b.ln())
    }

    /// the same, given ln b (for many draws with one b)
    pub fn geometric_ln(&mut self, ln_b: f64) -> u32 {
        if ln_b == f64::NEG_INFINITY || ln_b.is_nan() {
            return 0;
        }
        (self.uniform_open().ln() / ln_b).floor().min(1e9) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moments() {
        let mut rng = Rng::new(7);
        let n = 200_000;
        let (mut s1, mut s2) = (0.0, 0.0);
        for _ in 0..n {
            let z = rng.normal();
            s1 += z;
            s2 += z * z;
        }
        assert!((s1 / n as f64).abs() < 0.01);
        assert!((s2 / n as f64 - 1.0).abs() < 0.01);
        // geometric with b = 0.6: mean b/(1 − b) = 1.5
        let mean = (0..n).map(|_| rng.geometric(0.6) as f64).sum::<f64>() / n as f64;
        assert!((mean - 1.5).abs() < 0.02, "{mean}");
        let below = (0..n).map(|_| rng.below(10) as f64).sum::<f64>() / n as f64;
        assert!((below - 4.5).abs() < 0.03, "{below}");
    }
}
