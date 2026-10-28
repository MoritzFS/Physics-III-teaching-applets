//! Example 1: particles in boxes of fixed energy.
//!
//! Level i has the energy E_i and g_i boxes (its degeneracy). A heat bath at
//! temperature T moves the particles between boxes with the Metropolis rule:
//! a particle tries a random box, and a hop that costs ΔE > 0 is accepted
//! with probability e^{−ΔE/k_BT}. The occupations then settle to
//!
//! ```text
//! n_i / N = g_i e^{−E_i/k_BT} / Z₁,   Z₁ = Σ_i g_i e^{−E_i/k_BT}
//! ```

use crate::rng::Rng;

pub const MAX_LEVELS: usize = 20;
pub const MAX_PARTICLES: usize = 1000;
pub const MAX_G: u32 = 20;
/// the top of the energy axis
pub const E_TOP: f64 = 10.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Level {
    pub e: f64,
    pub g: u32,
}

/// Z₁, ⟨E⟩ and C/k_B of one particle.
#[derive(Clone, Copy, Debug)]
pub struct Thermo {
    pub z: f64,
    pub mean: f64,
    pub c: f64,
}

/// the Boltzmann weights g_i e^{−(E_i − E_min)/k_BT}, their sum and E_min (so nothing overflows)
fn weights(levels: &[Level], kt: f64) -> (Vec<f64>, f64, f64) {
    let e0 = levels.iter().map(|l| l.e).fold(f64::INFINITY, f64::min);
    let w: Vec<f64> = levels.iter().map(|l| l.g as f64 * (-(l.e - e0) / kt).exp()).collect();
    let sum = w.iter().sum();
    (w, sum, e0)
}

pub fn thermo(levels: &[Level], kt: f64) -> Thermo {
    let (w, sum, e0) = weights(levels, kt);
    let mean = w.iter().zip(levels).map(|(w, l)| w * (l.e - e0)).sum::<f64>() / sum;
    let sq = w.iter().zip(levels).map(|(w, l)| w * (l.e - e0 - mean).powi(2)).sum::<f64>() / sum;
    Thermo { z: sum * (-e0 / kt).exp(), mean: mean + e0, c: sq / (kt * kt) }
}

/// the fraction of the particles on each level, g_i e^{−E_i/k_BT} / Z₁
pub fn populations(levels: &[Level], kt: f64) -> Vec<f64> {
    let (w, sum, _) = weights(levels, kt);
    w.iter().map(|w| w / sum).collect()
}

/// the particles and the bath that moves them
pub struct Particles {
    /// the level of each particle
    pub level: Vec<usize>,
    /// the box within its level
    pub slot: Vec<u32>,
    /// where in its box each particle is drawn (fixed random numbers in [0, 1))
    pub spot: Vec<[f32; 2]>,
    rng: Rng,
    carry: f64,
}

impl Particles {
    pub fn new(n: usize, levels: &[Level], kt: f64) -> Self {
        let mut p = Particles { level: vec![], slot: vec![], spot: vec![], rng: Rng::new(1), carry: 0.0 };
        p.resize(n, levels, kt);
        p
    }

    /// Removes the last particles, or adds new ones drawn from equilibrium at kt.
    pub fn resize(&mut self, n: usize, levels: &[Level], kt: f64) {
        self.level.truncate(n);
        self.slot.truncate(n);
        self.spot.truncate(n);
        let pop = populations(levels, kt);
        while self.level.len() < n {
            let u = self.rng.uniform();
            let mut acc = 0.0;
            let i = pop.iter().position(|p| {
                acc += p;
                u < acc
            });
            let i = i.unwrap_or(pop.len() - 1);
            let slot = self.rng.below(levels[i].g as usize) as u32;
            self.level.push(i);
            self.slot.push(slot);
            self.spot.push([self.rng.uniform() as f32, self.rng.uniform() as f32]);
        }
    }

    /// Redraws every particle from equilibrium at kt.
    pub fn equilibrate(&mut self, levels: &[Level], kt: f64) {
        let n = self.level.len();
        self.resize(0, levels, kt);
        self.resize(n, levels, kt);
    }

    /// Keeps every particle in an existing box after levels were removed or boxes taken away.
    pub fn fit(&mut self, levels: &[Level]) {
        let top = levels.len() - 1;
        for k in 0..self.level.len() {
            self.level[k] = self.level[k].min(top);
            self.slot[k] %= levels[self.level[k]].g;
        }
    }

    /// Lets the bath act: on average `hops` tries per particle.
    pub fn step(&mut self, levels: &[Level], kt: f64, hops: f64) {
        let n = self.level.len();
        if n == 0 {
            return;
        }
        let total = hops * n as f64 + self.carry;
        let tries = total.floor().min(1e7);
        self.carry = (total - tries).min(1.0);
        // the first box of each level, counted over all levels
        let mut first = Vec::with_capacity(levels.len() + 1);
        let mut boxes = 0;
        for l in levels {
            first.push(boxes);
            boxes += l.g;
        }
        for _ in 0..tries as usize {
            let k = self.rng.below(n);
            // a random box: every box is proposed equally often, so the proposal is symmetric
            let b = self.rng.below(boxes as usize) as u32;
            let i = first.partition_point(|&f| f <= b) - 1;
            let de = levels[i].e - levels[self.level[k]].e;
            if de <= 0.0 || self.rng.uniform() < (-de / kt).exp() {
                self.level[k] = i;
                self.slot[k] = b - first[i];
            }
        }
    }

    /// the number of particles on each level
    pub fn counts(&self, n_levels: usize) -> Vec<usize> {
        let mut c = vec![0; n_levels];
        for &i in &self.level {
            c[i] += 1;
        }
        c
    }

    /// mean and variance of the particles' energies
    pub fn energy_stats(&self, levels: &[Level]) -> (f64, f64) {
        let n = self.level.len().max(1) as f64;
        let mean = self.level.iter().map(|&i| levels[i].e).sum::<f64>() / n;
        let var = self.level.iter().map(|&i| (levels[i].e - mean).powi(2)).sum::<f64>() / n;
        (mean, var)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LevelPreset {
    TwoLevels,
    TwoLevelsG3,
    ManyBoxesUp,
    Ladder,
    Rotor,
    Random,
}

/// what an example sets up
pub struct LevelSetup {
    pub levels: Vec<Level>,
    pub kt: f64,
    pub n: usize,
    pub per_box: bool,
    pub notes: &'static str,
}

impl LevelPreset {
    pub const ALL: [LevelPreset; 6] =
        [LevelPreset::TwoLevels, LevelPreset::TwoLevelsG3, LevelPreset::ManyBoxesUp, LevelPreset::Ladder, LevelPreset::Rotor, LevelPreset::Random];

    pub fn label(self) -> &'static str {
        match self {
            LevelPreset::TwoLevels => "Two levels",
            LevelPreset::TwoLevelsG3 => "Two levels, upper one with 3 boxes",
            LevelPreset::ManyBoxesUp => "Many boxes higher up",
            LevelPreset::Ladder => "Equally spaced ladder",
            LevelPreset::Rotor => "A rotating molecule",
            LevelPreset::Random => "Random levels",
        }
    }

    pub fn setup(self) -> LevelSetup {
        let lv = |e: f64, g: u32| Level { e, g };
        match self {
            LevelPreset::TwoLevels => LevelSetup {
                levels: vec![lv(0.0, 1), lv(1.0, 1)],
                kt: 0.42,
                n: 1000,
                per_box: false,
                notes: "Two levels, E = 0 and E = Δ = 1, one box each: Z₁ = 1 + e^(−Δ/k_BT).\n\n\
                    • Cold (k_BT much smaller than Δ): every particle sits in the ground level and ⟨E⟩ → 0.\n\
                    • Hot: both boxes are equally full and ⟨E⟩ → Δ/2. The bath never puts more than half of the particles up.\n\
                    • In between, C has a peak, C ≈ 0.44 k_B at k_BT ≈ 0.42 Δ (the Schottky anomaly). That is where this example starts.\n\n\
                    Compare with the table in the notes: at k_BT = Δ, Z₁ = 1.37 and ⟨E⟩ = 0.27 Δ; at k_BT = Δ/3, ⟨E⟩ = 0.047 Δ.\n\n\
                    Drag the upper level and watch the particles follow.",
            },
            LevelPreset::TwoLevelsG3 => LevelSetup {
                levels: vec![lv(0.0, 1), lv(1.0, 3)],
                kt: 1.0,
                n: 1000,
                per_box: true,
                notes: "The upper level now has g = 3 boxes: Z₁ = 1 + 3 e^(−Δ/k_BT).\n\n\
                    The bars show the particles per box: every box, in either level, ends on the same curve e^(−E/k_BT)/Z₁. \
                    A box does not know how many other boxes share its energy.\n\n\
                    Per level, the upper level wins when it is hot: 3/4 of the particles are up and ⟨E⟩ → 3Δ/4 instead of Δ/2.\n\n\
                    Degeneracy is entropy: g e^(−E/k_BT) = e^(−(E − TS)/k_BT) with S = k_B ln g.",
            },
            LevelPreset::ManyBoxesUp => LevelSetup {
                levels: vec![lv(0.0, 1), lv(2.0, 12)],
                kt: 1.0,
                n: 1000,
                per_box: false,
                notes: "One box at E = 0 and twelve boxes at E = 2.\n\n\
                    At k_BT = 1 more particles are up (62 %) than down, but each upper box holds fewer particles than the ground box: \
                    per box the bath always favours the lower energy. This is not a population inversion.\n\n\
                    Switch the bars between 'per level' and 'per box' to see both views.\n\n\
                    Cold, all particles go down anyway: the e^(−E/k_BT) wins over any finite g.",
            },
            LevelPreset::Ladder => LevelSetup {
                levels: (0..20).map(|i| lv(0.5 * i as f64, 1)).collect(),
                kt: 1.0,
                n: 1000,
                per_box: false,
                notes: "Twenty equally spaced levels, E_n = n ε with ε = 0.5: the quantum oscillator of example 5, cut off at n = 19.\n\n\
                    The occupation falls off as e^(−E/k_BT): the same factor e^(−ε/k_BT) from each level to the next.\n\n\
                    For k_BT between about ε and a few ε, C ≈ k_B as for an oscillator. At high k_BT the cut-off shows: \
                    a finite set of levels can only hold so much energy, so ⟨E⟩ levels off and C drops again.",
            },
            LevelPreset::Rotor => LevelSetup {
                levels: (0..10).map(|l| lv(0.1 * (l * (l + 1)) as f64, 2 * l as u32 + 1)).collect(),
                kt: 0.5,
                n: 1000,
                per_box: false,
                notes: "A rotating molecule (rigid rotor): E_l = l(l + 1) B with 2l + 1 states each, here B = 0.1 and l = 0 … 9.\n\n\
                    The most occupied level is not the lowest one: g grows while e^(−E/k_BT) falls.\n\n\
                    For k_BT well above B, Z₁ ≈ k_BT/B ∝ β^(−1): two rotation axes, two quadratic terms, so C ≈ k_B and ⟨E⟩ is close to k_BT. \
                    This is the rotational step of the H₂ heat capacity in example 3 of the notes.\n\n\
                    Cold (k_BT well below the first gap, 2B), the rotation freezes out. Only ten levels are kept here, so above k_BT ≈ 2 the cut-off makes C fall again; \
                    a real molecule has no top level.",
            },
            LevelPreset::Random => {
                let mut rng = Rng::new(2026);
                let mut levels = vec![lv(0.0, 1)];
                for _ in 0..7 {
                    levels.push(lv((0.3 + 6.0 * rng.uniform() * 100.0).round() / 100.0, 1 + rng.below(4) as u32));
                }
                LevelSetup {
                    levels,
                    kt: 1.5,
                    n: 1000,
                    per_box: false,
                    notes: "Any set of levels. Drag them in the diagram, change their boxes in the list, and watch the bath sort the particles.\n\n\
                        Whatever the levels, the limits are the same:\n\
                        • cold: everything in the lowest level, Z₁ → g₀ e^(−E₀/k_BT), C → 0;\n\
                        • hot: every box equally full, Z₁ → Σ g_i (the number of boxes), C → 0 again.\n\n\
                        So C always has at least one bump in between.",
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two(g: u32) -> Vec<Level> {
        vec![Level { e: 0.0, g: 1 }, Level { e: 1.0, g }]
    }

    #[test]
    fn two_level_table() {
        // the table in the notes (g = 1): βΔ, Z₁, ⟨E⟩/Δ
        for (b, z, e) in [(1.0, 1.3679, 0.26894), (3.0, 1.04979, 0.047426), (10.0, 1.0000454, 0.0000454)] {
            let t = thermo(&two(1), 1.0 / b);
            assert!((t.z - z).abs() < 1e-4, "Z {b}: {}", t.z);
            assert!((t.mean - e).abs() < 1e-5, "E {b}: {}", t.mean);
            // C/k_B = (βΔ)² e^{−βΔ} / (1 + e^{−βΔ})²
            let c = b * b * (-b).exp() / (1.0 + (-b).exp()).powi(2);
            assert!((t.c - c).abs() < 1e-12);
        }
        assert!((thermo(&two(1), 1e6).mean - 0.5).abs() < 1e-5);
        assert!((thermo(&two(3), 1e6).mean - 0.75).abs() < 1e-5);
    }

    #[test]
    fn schottky_peak() {
        let (c, kt) = (100..1000).map(|k| (thermo(&two(1), k as f64 / 1000.0).c, k as f64 / 1000.0)).fold((0.0, 0.0), |a, b| if b.0 > a.0 { b } else { a });
        assert!((kt - 0.42).abs() < 0.005, "{kt}");
        assert!((c - 0.44).abs() < 0.005, "{c}");
    }

    #[test]
    fn rotor_and_cutoff() {
        let rotor = LevelPreset::Rotor.setup().levels;
        let t = thermo(&rotor, 0.5);
        assert!((t.c - 1.0).abs() < 0.01, "{}", t.c);
        assert!(thermo(&rotor, 10.0).c < 0.1);
        let up = populations(&LevelPreset::ManyBoxesUp.setup().levels, 1.0)[1];
        assert!((up - 0.62).abs() < 0.005, "{up}");
    }

    #[test]
    fn bath_reaches_boltzmann() {
        let levels = vec![Level { e: 0.0, g: 1 }, Level { e: 0.7, g: 3 }, Level { e: 2.0, g: 2 }, Level { e: 3.5, g: 1 }];
        let kt = 0.9;
        // start far from equilibrium: all on the top level
        let mut p = Particles::new(1000, &levels, 0.01);
        p.level.iter_mut().for_each(|l| *l = 3);
        p.slot.iter_mut().for_each(|s| *s = 0);
        for _ in 0..50 {
            p.step(&levels, kt, 1.0);
        }
        let mut sum = vec![0.0; 4];
        let rounds = 400;
        for _ in 0..rounds {
            p.step(&levels, kt, 0.5);
            for (s, c) in sum.iter_mut().zip(p.counts(4)) {
                *s += c as f64 / (1000.0 * rounds as f64);
            }
        }
        for (s, want) in sum.iter().zip(populations(&levels, kt)) {
            assert!((s - want).abs() < 0.01, "{sum:?}");
        }
        // boxes within a level are filled equally
        let in_box0 = (0..1000).filter(|&k| p.level[k] == 1 && p.slot[k] == 0).count() as f64;
        let in_level = p.counts(4)[1] as f64;
        assert!((in_box0 / in_level - 1.0 / 3.0).abs() < 0.08);
    }
}
