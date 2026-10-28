//! Examples 4 and 5: modes of fixed frequency in contact with a heat bath.
//!
//! Classical (example 4): a mode's energy is E = q² + p², where q² and p² are
//! its potential and kinetic energy (q = √(k/2) x, p = p_x/√(2m)). In the
//! frame that turns with the oscillation, (q, p) only change through the
//! bath: a weak friction γ/2 plus random kicks (an Ornstein–Uhlenbeck
//! process), stepped exactly for any dt:
//!
//! ```text
//! q → f q + s ξ,   f = e^{−γ dt/2},   s = √(k_BT (1 − f²)/2)
//! ```
//!
//! so q and p become Gaussian with variance k_BT/2 and E exponential with
//! mean k_BT. The frequency only sets how fast the drawing swings.
//!
//! Quantum (example 5): E_n = n ε. The bath adds a quantum at the rate
//! γ n̄ (n + 1) and takes one away at γ (n̄ + 1) n, with n̄ = 1/(e^{ε/k_BT} − 1).
//! The ratio of the two rates for n ↔ n + 1 is e^{−ε/k_BT} (detailed
//! balance), and ⟨n⟩ relaxes like the classical energy:
//! d⟨n⟩/dt = γ (n̄ − ⟨n⟩). Over a time t this birth–death process is sampled
//! exactly (Kendall 1948): each quantum present leaves no descendant with
//! probability α, otherwise 1 + Geom(β) of them, and new ones arrive as
//! Geom(β), with
//!
//! ```text
//! α = μ (1 − e^{−γt}) / (μ − λ e^{−γt}),   β = λ (1 − e^{−γt}) / (μ − λ e^{−γt}),   λ = n̄, μ = n̄ + 1
//! ```

use crate::rng::Rng;

pub const MAX_MODES: usize = 10_000;
/// how fast the bath changes a mode's energy (per unit of simulation time)
pub const GAMMA: f64 = 0.5;
/// the open strings of a guitar (E A D G B E), as frequencies relative to the lowest
pub const GUITAR: [f64; 6] = [1.0, 1.3348, 1.7818, 2.3784, 2.9966, 4.0];

/// ⟨n⟩ = 1/(e^{ε/k_BT} − 1)
pub fn mean_quanta(kt: f64, eps: f64) -> f64 {
    1.0 / (eps / kt).exp_m1()
}

/// C/k_B of a quantum mode, (ε/k_BT)² e^{ε/k_BT} / (e^{ε/k_BT} − 1)²
pub fn quantum_heat_capacity(kt: f64, eps: f64) -> f64 {
    let x = eps / kt;
    if x > 600.0 {
        return 0.0;
    }
    let em1 = x.exp_m1();
    x * x * (em1 + 1.0) / (em1 * em1)
}

/// P(n) = (1 − e^{−ε/k_BT}) e^{−nε/k_BT}
pub fn quantum_probability(n: u32, kt: f64, eps: f64) -> f64 {
    let x = eps / kt;
    -(-x).exp_m1() * (-(n as f64) * x).exp()
}

/// The number of quanta after a time t (γt = `gt`), sampled exactly.
pub fn evolve_quanta(n: u32, nbar: f64, gt: f64, rng: &mut Rng) -> u32 {
    let e = (-gt).exp();
    let (lambda, mu) = (nbar, nbar + 1.0);
    let den = mu - lambda * e;
    let alpha = mu * (1.0 - e) / den;
    let ln_beta = (lambda * (1.0 - e) / den).ln();
    let mut m = rng.geometric_ln(ln_beta);
    for _ in 0..n {
        if rng.uniform() >= alpha {
            m += 1 + rng.geometric_ln(ln_beta);
        }
    }
    m
}

pub struct Modes {
    pub quantum: bool,
    /// classical: the amplitude in the turning frame, E = q² + p²
    pub q: Vec<f64>,
    pub p: Vec<f64>,
    /// quantum: the number of quanta
    pub n: Vec<u32>,
    /// quantum: the phase of the drawn vibration
    pub phase: Vec<f64>,
    rng: Rng,
}

impl Modes {
    pub fn new(quantum: bool, m: usize, kt: f64, eps: f64) -> Self {
        let mut modes = Modes { quantum, q: vec![], p: vec![], n: vec![], phase: vec![], rng: Rng::new(if quantum { 5 } else { 4 }) };
        modes.resize(m, kt, eps);
        modes
    }

    pub fn len(&self) -> usize {
        self.phase.len()
    }

    /// Removes the last modes, or adds new ones drawn from equilibrium.
    pub fn resize(&mut self, m: usize, kt: f64, eps: f64) {
        self.q.truncate(m);
        self.p.truncate(m);
        self.n.truncate(m);
        self.phase.truncate(m);
        let s = (0.5 * kt).sqrt();
        let beta = (-eps / kt).exp();
        while self.phase.len() < m {
            self.q.push(s * self.rng.normal());
            self.p.push(s * self.rng.normal());
            self.n.push(self.rng.geometric(beta));
            self.phase.push(std::f64::consts::TAU * self.rng.uniform());
        }
    }

    /// Redraws every mode from equilibrium.
    pub fn equilibrate(&mut self, kt: f64, eps: f64) {
        let m = self.len();
        self.resize(0, kt, eps);
        self.resize(m, kt, eps);
    }

    /// Takes all energy out (every mode at rest, or in n = 0).
    pub fn freeze(&mut self) {
        self.q.iter_mut().for_each(|x| *x = 0.0);
        self.p.iter_mut().for_each(|x| *x = 0.0);
        self.n.iter_mut().for_each(|x| *x = 0);
    }

    /// Lets the bath act for the time dt.
    pub fn step(&mut self, kt: f64, eps: f64, dt: f64) {
        if self.quantum {
            let nbar = mean_quanta(kt, eps);
            for k in 0..self.n.len() {
                self.n[k] = evolve_quanta(self.n[k], nbar, GAMMA * dt, &mut self.rng);
            }
        } else {
            let f = (-0.5 * GAMMA * dt).exp();
            let s = (0.5 * kt * (1.0 - f * f)).sqrt();
            for k in 0..self.q.len() {
                self.q[k] = f * self.q[k] + s * self.rng.normal();
                self.p[k] = f * self.p[k] + s * self.rng.normal();
            }
        }
    }

    pub fn energy(&self, k: usize, eps: f64) -> f64 {
        if self.quantum { self.n[k] as f64 * eps } else { self.q[k] * self.q[k] + self.p[k] * self.p[k] }
    }

    /// The displacement of mode k at the phase θ, in units where the amplitude is √E.
    pub fn displacement(&self, k: usize, theta: f64, eps: f64) -> f64 {
        if self.quantum { (self.n[k] as f64 * eps).sqrt() * (theta + self.phase[k]).cos() } else { self.q[k] * theta.cos() + self.p[k] * theta.sin() }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OscPreset {
    Masses,
    One,
    TenThousand,
    Guitar,
    Pitches,
    QuantumEqual,
    QuantumFrozen,
    QuantumHot,
    QuantumStrings,
    QuantumOne,
}

/// what an example sets up
pub struct OscSetup {
    pub m: usize,
    pub kt: f64,
    pub eps: f64,
    pub strings: bool,
    pub guitar: bool,
    pub long_average: bool,
    pub notes: &'static str,
}

impl OscPreset {
    pub const CLASSICAL: [OscPreset; 5] = [OscPreset::Masses, OscPreset::One, OscPreset::TenThousand, OscPreset::Guitar, OscPreset::Pitches];
    pub const QUANTUM: [OscPreset; 5] =
        [OscPreset::QuantumEqual, OscPreset::QuantumFrozen, OscPreset::QuantumHot, OscPreset::QuantumStrings, OscPreset::QuantumOne];

    pub fn label(self) -> &'static str {
        match self {
            OscPreset::Masses => "Masses on springs",
            OscPreset::One => "One mode, watched for a minute",
            OscPreset::TenThousand => "Ten thousand modes",
            OscPreset::Guitar => "Guitar strings",
            OscPreset::Pitches => "Different masses, same springs",
            OscPreset::QuantumEqual => "k_BT = ε",
            OscPreset::QuantumFrozen => "Cold: frozen out",
            OscPreset::QuantumHot => "Hot: almost classical",
            OscPreset::QuantumStrings => "Quantised strings",
            OscPreset::QuantumOne => "One mode, watched for a minute",
        }
    }

    pub fn setup(self) -> OscSetup {
        let base = OscSetup { m: 200, kt: 1.0, eps: 1.0, strings: false, guitar: false, long_average: false, notes: "" };
        match self {
            OscPreset::Masses => OscSetup {
                notes: "Each mass on a spring is one mode of fixed frequency: H = p²/2m + ½ mω² x². \
                    Two quadratic terms, so Z₁ ∝ β^(−1), ⟨E⟩ = k_BT and C = k_B.\n\n\
                    A heat bath (a weak friction plus random kicks) keeps changing the amplitudes. The colour shows each mode's energy.\n\n\
                    The energy of one mode is spread as P(E) = e^(−E/k_BT)/k_BT: zero is the most likely energy, but any amplitude is allowed. \
                    The mean over all modes sits at k_BT.\n\n\
                    A crystal of N atoms has 3N such modes, so C = 3N k_B (Dulong–Petit).",
                ..base
            },
            OscPreset::One => OscSetup {
                m: 1,
                long_average: true,
                notes: "A single mode. Its energy wanders all the time: the bath gives and takes.\n\n\
                    Watched for a minute, the histogram becomes the exponential e^(−E/k_BT)/k_BT. \
                    The mean is k_BT, and the spread is just as large: the standard deviation is also k_BT.\n\n\
                    Now raise the number of modes and watch the mean settle (middle plot).",
                ..base
            },
            OscPreset::TenThousand => OscSetup {
                m: 10_000,
                notes: "The mean energy of M modes fluctuates by k_BT/√M (the band in the middle plot): as much as k_BT itself for one mode, \
                    1 % of it for ten thousand.\n\n\
                    A single mode (the thin line) still jumps around as much as ever. Only the average becomes sharp.\n\n\
                    For a mole of modes the mean is k_BT to about twelve digits. This is why thermodynamics can work with averages.",
                ..base
            },
            OscPreset::Guitar => OscSetup {
                m: 6,
                strings: true,
                guitar: true,
                long_average: true,
                notes: "Six strings vibrating in their fundamental mode, tuned like a guitar (E A D G B E). The energy of the mode is set by the \
                    amplitude A: E = π²τA²/4L for a string of tension τ and length L. It does not depend on the pitch.\n\n\
                    Every string has ⟨E⟩ = k_BT, whatever its frequency: the frequency drops out of ⟨E⟩ and C.\n\n\
                    For a real guitar string at room temperature (τ ≈ 70 N, L = 0.65 m, k_BT = 4 × 10⁻²¹ J) this is an rms amplitude \
                    of about 4 × 10⁻¹² m. Thermal motion is far too small to hear.",
                ..base
            },
            OscPreset::Pitches => OscSetup {
                m: 60,
                guitar: true,
                notes: "The same springs with different masses: heavy masses (large circles) swing slowly, light ones fast.\n\n\
                    All modes have the same mean energy k_BT, and with equal springs even the same amplitudes. \
                    Equipartition does not care about the frequency.\n\n\
                    A cavity has infinitely many modes of light, and if each took k_BT the energy would be infinite: \
                    the Rayleigh–Jeans law and its ultraviolet catastrophe. Example 5 shows the way out.",
                ..base
            },
            OscPreset::QuantumEqual => OscSetup {
                notes: "The same modes, but the energy now comes in steps: E_n = n ε with n = 0, 1, 2, … \
                    The bath adds or takes away one quantum ε at a time.\n\n\
                    Z₁ = Σ e^(−nε/k_BT) = 1/(1 − e^(−ε/k_BT)), a geometric series. So P(n) = (1 − e^(−ε/k_BT)) e^(−nε/k_BT) \
                    and the mean number of quanta is ⟨n⟩ = 1/(e^(ε/k_BT) − 1).\n\n\
                    At k_BT = ε: ⟨n⟩ = 0.58 and C = 0.92 k_B, as in the table of the notes. The grey lines are the classical modes of example 4.\n\n\
                    The drawing shows a mode with n quanta as a classical vibration of energy n ε, so only certain amplitudes appear (tick marks). \
                    A real quantum state has no definite amplitude; the picture is only for comparing with example 4.\n\n\
                    The notes add a zero-point energy ε/2 to every level, E_n = (n + ½) ε. That shifts ⟨E⟩ by ε/2 and changes neither P(n) nor C.",
                ..base
            },
            OscPreset::QuantumFrozen => OscSetup {
                kt: 0.2,
                notes: "k_BT = 0.2 ε: almost every mode sits in n = 0 (⟨n⟩ = 0.007), with a rare jump to n = 1.\n\n\
                    The bath can rarely pay a whole quantum, so the mode hardly takes up heat: C = 0.17 k_B, and it falls fast \
                    as k_BT drops. The mode is frozen out.\n\n\
                    This is why the high-frequency modes of a cavity stay empty (Planck's law instead of Rayleigh–Jeans), \
                    and why the heat capacity of a crystal drops at low temperature (Einstein model).",
                ..base
            },
            OscPreset::QuantumHot => OscSetup {
                kt: 5.0,
                notes: "k_BT = 5 ε: many quanta per mode. The steps are small compared with k_BT, so the bars lie close together and \
                    look like the continuous exponential of example 4. ⟨E⟩ and C are close to the classical lines (grey): C → k_B.\n\n\
                    Here ⟨E⟩ ≈ k_BT − ε/2: counting from n = 0 loses half a step on average. With the zero-point energy ε/2 of the notes, ⟨E⟩ ≈ k_BT.\n\n\
                    Every quantum system becomes classical once k_BT is large compared with its level spacing.",
                ..base
            },
            OscPreset::QuantumStrings => OscSetup {
                m: 12,
                strings: true,
                long_average: true,
                notes: "Twelve strings whose fundamental mode is quantised, all with the same step ε, at k_BT = ε.\n\n\
                    Each string jumps between allowed amplitudes, A ∝ √n (the faint envelopes). \
                    Lower k_BT and most strings fall silent in n = 0.\n\n\
                    The drawing is a classical vibration with the energy n ε; it is only meant for comparing with example 4.",
                ..base
            },
            OscPreset::QuantumOne => OscSetup {
                m: 1,
                long_average: true,
                notes: "One quantum mode at k_BT = ε, watched for a minute.\n\n\
                    The fraction of time it spends with n quanta falls off by the same factor e^(−ε/k_BT) = 0.37 from each n to the next.",
                ..base
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantum_table() {
        // the table in the notes: k_BT/ε, ⟨n⟩, C/k_B
        for (t, n, c) in [(0.1, 0.0000454, 0.00454), (0.2, 0.00678, 0.1707), (0.5, 0.1565, 0.7241), (1.0, 0.5820, 0.9207), (2.0, 1.5415, 0.9794)] {
            assert!((mean_quanta(t, 1.0) - n).abs() < 2e-4 * n.max(1.0), "{t}: {}", mean_quanta(t, 1.0));
            assert!((quantum_heat_capacity(t, 1.0) - c).abs() < 2e-4, "{t}: {}", quantum_heat_capacity(t, 1.0));
        }
        // hot: classical, ⟨E⟩ ≈ k_BT − ε/2 without the zero-point energy
        assert!((mean_quanta(50.0, 1.0) - 49.5).abs() < 0.01);
        assert!((quantum_heat_capacity(50.0, 1.0) - 1.0).abs() < 1e-3);
        let total: f64 = (0..200).map(|n| quantum_probability(n, 2.0, 1.0)).sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn exact_quantum_relaxation() {
        // ⟨n(t)⟩ = n₀ e^{−γt} + n̄ (1 − e^{−γt}) from any start
        let mut rng = Rng::new(9);
        let (n0, nbar, gt) = (12, 1.5, 0.7);
        let runs = 100_000;
        let mean = (0..runs).map(|_| evolve_quanta(n0, nbar, gt, &mut rng) as f64).sum::<f64>() / runs as f64;
        let want = n0 as f64 * (-gt).exp() + nbar * (1.0 - (-gt).exp());
        assert!((mean - want).abs() < 0.03, "{mean} vs {want}");
    }

    #[test]
    fn bath_reaches_equilibrium() {
        for quantum in [false, true] {
            let (kt, eps) = (0.8, 0.5);
            let mut modes = Modes::new(quantum, 10_000, kt, eps);
            modes.freeze();
            for _ in 0..300 {
                modes.step(kt, eps, 0.1);
            }
            let m = modes.len() as f64;
            let mean = (0..modes.len()).map(|k| modes.energy(k, eps)).sum::<f64>() / m;
            let var = (0..modes.len()).map(|k| (modes.energy(k, eps) - mean).powi(2)).sum::<f64>() / m;
            let (want, c) = if quantum { (eps * mean_quanta(kt, eps), quantum_heat_capacity(kt, eps)) } else { (kt, 1.0) };
            assert!((mean / want - 1.0).abs() < 0.03, "{quantum}: {mean} vs {want}");
            // C = Var(E)/k_BT²
            assert!((var / (kt * kt) / c - 1.0).abs() < 0.06, "{quantum}: {}", var / (kt * kt));
            if quantum {
                let p0 = modes.n.iter().filter(|&&n| n == 0).count() as f64 / m;
                assert!((p0 - quantum_probability(0, kt, eps)).abs() < 0.015, "{p0}");
            }
        }
    }
}
