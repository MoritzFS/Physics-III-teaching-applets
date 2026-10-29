//! Example 6: light in a box. The electromagnetic modes of a cube of side L
//! with conducting walls are oscillators, as in examples 4 and 5.
//!
//! A mode has k = (π/L)(n_x, n_y, n_z) with n_i = 0, 1, 2, … and the field
//! E_x ∝ cos(k_x x) sin(k_y y) sin(k_z z) (and cyclic), with E ⟂ k. With all
//! n_i ≥ 1 there are two polarisations; with one n_i = 0 only one (the field
//! points along that axis); with two zeros the field vanishes. A quantum of the
//! mode has the energy ε = hν = ε₁|n| with ε₁ = hc/2L, the unit of energy here.
//! Counting the points of the positive octant gives (lecture notes, 4.77)
//!
//! ```text
//! g(ε) dε = π ε²/ε₁³ dε    (= 8πν²V/c³ dν)
//! ```

use crate::oscillators::{GAMMA, evolve_quanta};
use crate::rng::Rng;

/// the largest |n| simulated
pub const MAX_R: f64 = 40.0;

/// x = ε/k_BT at the peak of the Planck spectrum: 3(1 − e^{−x}) = x
pub const WIEN_X: f64 = 2.821_439_372_122_078_7;

/// the number of polarisations of the mode n
pub fn polarisations(n: [u16; 3]) -> u8 {
    match n.iter().filter(|&&c| c == 0).count() {
        0 => 2,
        1 => 1,
        _ => 0,
    }
}

/// g(ε): modes per unit energy (units of ε₁)
pub fn density_of_states(eps: f64) -> f64 {
    std::f64::consts::PI * eps * eps
}

/// the number of modes below ε, in the continuum limit
pub fn modes_below(eps: f64) -> f64 {
    std::f64::consts::PI * eps.powi(3) / 3.0
}

/// Planck: ⟨E⟩ = ε/(e^{ε/k_BT} − 1) per mode (no zero-point energy, as in the notes)
pub fn planck_mean(eps: f64, kt: f64) -> f64 {
    if eps <= 0.0 { kt } else { eps / (eps / kt).exp_m1() }
}

/// the Planck energy of all modes below ε_c (continuum)
pub fn planck_total_below(eps_c: f64, kt: f64) -> f64 {
    // Simpson on 400 intervals
    let n = 400;
    let h = eps_c / n as f64;
    let f = |e: f64| density_of_states(e) * planck_mean(e, kt);
    let mut s = f(0.0) + f(eps_c);
    for i in 1..n {
        s += f(i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    s * h / 3.0
}

/// Stefan–Boltzmann: the energy of all modes, π⁵ (k_BT)⁴/15ε₁³
pub fn stefan_boltzmann(kt: f64) -> f64 {
    std::f64::consts::PI.powi(5) * kt.powi(4) / 15.0
}

/// One oscillator per mode and polarisation, sorted by ε, so that a cutoff is a prefix.
pub struct Cavity {
    pub quantum: bool,
    pub n: Vec<[u16; 3]>,
    pub eps: Vec<f64>,
    /// how many modes take part (|n| ≤ the cutoff)
    pub active: usize,
    /// classical: the amplitude in the turning frame, E = q² + p²
    q: Vec<f64>,
    p: Vec<f64>,
    /// quantum: photons in the mode
    pub photons: Vec<u32>,
    rng: Rng,
}

impl Cavity {
    pub fn new(quantum: bool, cutoff: f64, kt: f64) -> Self {
        let r = MAX_R as u16;
        let mut modes = vec![];
        for a in 0..=r {
            for b in 0..=r {
                for c in 0..=r {
                    let n = [a, b, c];
                    let len = ((a as f64).powi(2) + (b as f64).powi(2) + (c as f64).powi(2)).sqrt();
                    if len <= MAX_R {
                        for _ in 0..polarisations(n) {
                            modes.push((len, n));
                        }
                    }
                }
            }
        }
        modes.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
        let m = modes.len();
        let mut cav = Cavity {
            quantum,
            n: modes.iter().map(|x| x.1).collect(),
            eps: modes.iter().map(|x| x.0).collect(),
            active: 0,
            q: vec![0.0; m],
            p: vec![0.0; m],
            photons: vec![0; m],
            rng: Rng::new(6),
        };
        cav.set_cutoff(cutoff, kt);
        cav
    }

    /// Takes part the modes with |n| ≤ cutoff; newly added ones start in equilibrium.
    pub fn set_cutoff(&mut self, cutoff: f64, kt: f64) {
        let active = self.eps.partition_point(|&e| e <= cutoff);
        for k in self.active..active {
            self.draw(k, kt);
        }
        self.active = active;
    }

    fn draw(&mut self, k: usize, kt: f64) {
        let s = (0.5 * kt).sqrt();
        self.q[k] = s * self.rng.normal();
        self.p[k] = s * self.rng.normal();
        self.photons[k] = self.rng.geometric((-self.eps[k] / kt).exp());
    }

    /// Redraws every mode from equilibrium.
    pub fn equilibrate(&mut self, kt: f64) {
        for k in 0..self.active {
            self.draw(k, kt);
        }
    }

    /// Takes all energy out.
    pub fn freeze(&mut self) {
        self.q.iter_mut().for_each(|x| *x = 0.0);
        self.p.iter_mut().for_each(|x| *x = 0.0);
        self.photons.iter_mut().for_each(|x| *x = 0);
    }

    /// Lets the walls (the bath) act for the time dt, mode by mode as in examples 4 and 5.
    pub fn step(&mut self, kt: f64, dt: f64) {
        if self.quantum {
            for k in 0..self.active {
                let nbar = 1.0 / (self.eps[k] / kt).exp_m1();
                self.photons[k] = evolve_quanta(self.photons[k], nbar, GAMMA * dt, &mut self.rng);
            }
        } else {
            let f = (-0.5 * GAMMA * dt).exp();
            let s = (0.5 * kt * (1.0 - f * f)).sqrt();
            for k in 0..self.active {
                self.q[k] = f * self.q[k] + s * self.rng.normal();
                self.p[k] = f * self.p[k] + s * self.rng.normal();
            }
        }
    }

    pub fn energy(&self, k: usize) -> f64 {
        if self.quantum { self.photons[k] as f64 * self.eps[k] } else { self.q[k] * self.q[k] + self.p[k] * self.p[k] }
    }

    /// the formula's total energy of the active modes: k_BT each, or Planck's
    pub fn expected_total(&self, kt: f64) -> f64 {
        if self.quantum { self.eps[..self.active].iter().map(|&e| planck_mean(e, kt)).sum() } else { kt * self.active as f64 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CavityPreset {
    Counting,
    RayleighJeans,
    Planck,
    Cold,
    Hot,
}

pub struct CavitySetup {
    pub quantum: bool,
    pub kt: f64,
    pub cutoff: f64,
    pub notes: &'static str,
}

impl CavityPreset {
    pub const ALL: [CavityPreset; 5] = [CavityPreset::Counting, CavityPreset::RayleighJeans, CavityPreset::Planck, CavityPreset::Cold, CavityPreset::Hot];

    pub fn label(self) -> &'static str {
        match self {
            CavityPreset::Counting => "Counting the modes",
            CavityPreset::RayleighJeans => "Rayleigh–Jeans: k_BT for every mode",
            CavityPreset::Planck => "Planck: high modes freeze out",
            CavityPreset::Cold => "Cold: only the lowest modes",
            CavityPreset::Hot => "Hotter: Wien and Stefan–Boltzmann",
        }
    }

    pub fn setup(self) -> CavitySetup {
        match self {
            CavityPreset::Counting => CavitySetup {
                quantum: true,
                kt: 3.0,
                cutoff: 16.0,
                notes: "Light in a cube of side L with conducting walls (lecture notes 4.6.1). Each standing wave is a mode, and each mode is an oscillator \
                    like those of examples 4 and 5.\n\n\
                    A mode has k = (π/L)(n_x, n_y, n_z): one point of a grid in the positive octant of k-space (left). The field must be ⟂ k, so each point \
                    has two polarisations. (Points with one n_i = 0 have only one, along that axis; the notes' count, which is right for large k, \
                    counts all points twice.)\n\n\
                    The modes between ε and ε + dε lie in a shell of the octant (outlined): their number is ⅛ 4πk² dk × 2 / (π³/V). With the energy of a \
                    quantum, ε = hν = ε₁|n| and ε₁ = hc/2L:\n    g(ε) dε = π ε²/ε₁³ dε   (= 8πν²V/c³ dν).\n\n\
                    Energies are in units of ε₁. For a real box, L = 1 cm gives ε₁ = 6 × 10⁻⁵ eV, so room temperature is k_BT ≈ 400 ε₁ and g(ε) is smooth. \
                    Here k_BT is a few ε₁, so you can see single modes and the steps in the count.",
            },
            CavityPreset::RayleighJeans => CavitySetup {
                quantum: false,
                kt: 4.0,
                cutoff: 40.0,
                notes: "Classical modes: each has a continuous amplitude, so each holds ⟨E⟩ = k_BT on average (equipartition, example 4), whatever its frequency.\n\n\
                    The spectrum is then g(ε) k_BT ∝ ε², the Rayleigh–Jeans law (notes 4.78): it rises without end. The total energy is k_BT times the number \
                    of modes, and there are infinitely many. Here only modes up to |n| = 40 are kept: move the cutoff and watch the total energy grow like \
                    the cutoff cubed (right plot). This is the ultraviolet catastrophe.\n\n\
                    At low ε the classical result is fine: there it agrees with Planck's.",
            },
            CavityPreset::Planck => CavitySetup {
                quantum: true,
                kt: 4.0,
                cutoff: 40.0,
                notes: "Quantised modes: a mode of frequency ν holds energy only in steps ε = hν (example 5), so ⟨E⟩ = hν/(e^(hν/k_BT) − 1) (notes 4.80).\n\n\
                    Modes with hν well above k_BT are frozen out: the walls can rarely pay a whole quantum. Multiplied by g(ε) ∝ ε², this gives \
                    Planck's spectrum (notes 4.81): it rises like Rayleigh–Jeans at low ε and falls off exponentially at high ε.\n\n\
                    The total energy stays finite: moving the cutoff beyond a few k_BT adds nothing. All modes together hold\n\
                    U = π⁵ (k_BT)⁴/15ε₁³, the Stefan–Boltzmann law U ∝ T⁴ (notes 4.88).",
            },
            CavityPreset::Cold => CavitySetup {
                quantum: true,
                kt: 0.8,
                cutoff: 12.0,
                notes: "k_BT = 0.8 ε₁: only the lowest modes hold photons. The lowest mode is (1, 1, 0) with ε = √2 ε₁: no mode has a smaller step.\n\n\
                    At such low energies the continuum g(ε) = πε²/ε₁³ is only a rough guide: the real count comes in steps, mode by mode. \
                    The size of the box matters here, while for k_BT ≫ ε₁ only its volume does.",
            },
            CavityPreset::Hot => CavitySetup {
                quantum: true,
                kt: 7.0,
                cutoff: 40.0,
                notes: "Raise k_BT and watch the peak of the spectrum move: it sits at ε = 2.82 k_BT, from 3 = (3 − x)eˣ (Wien's displacement law, notes 4.85).\n\n\
                    The total energy grows like T⁴ (Stefan–Boltzmann). Doubling k_BT gives 16 times the energy: twice the energy per mode, \
                    in 2³ times as many modes below the peak.\n\n\
                    At k_BT = 7 ε₁ the spectrum reaches the cutoff |n| = 40, so the top of the tail is missing.",
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_count_approaches_the_continuum() {
        let cav = Cavity::new(true, MAX_R, 1.0);
        for r in [10.0, 20.0, 40.0] {
            let exact = cav.eps.partition_point(|&e| e <= r) as f64;
            let ratio = exact / modes_below(r);
            // the boundary of the octant costs a fraction ∝ 1/r
            assert!((ratio - 1.0).abs() < 2.5 / r, "{r}: {ratio}");
        }
        // the lowest mode is (1, 1, 0) and the like, one polarisation each
        assert!((cav.eps[0] - 2f64.sqrt()).abs() < 1e-12);
        assert_eq!(polarisations([1, 1, 0]), 1);
        assert_eq!(polarisations([1, 0, 0]), 0);
        assert_eq!(polarisations([2, 1, 3]), 2);
    }

    #[test]
    fn planck_wien_stefan_boltzmann() {
        assert!((3.0 * (1.0 - (-WIEN_X).exp()) - WIEN_X).abs() < 1e-12);
        let kt = 2.0;
        let peak = (1..4000).map(|i| i as f64 * 0.005).max_by(|a, b| (a.powi(3) / a.exp_m1()).total_cmp(&(b.powi(3) / b.exp_m1()))).unwrap();
        assert!((peak - WIEN_X).abs() < 0.005);
        assert!((planck_total_below(40.0 * kt, kt) / stefan_boltzmann(kt) - 1.0).abs() < 1e-6);
        // the classical limit
        assert!((planck_mean(0.01, 5.0) / 5.0 - 1.0).abs() < 1e-3);
    }

    #[test]
    fn walls_bring_both_kinds_of_modes_to_their_formula() {
        for quantum in [false, true] {
            let kt = 3.0;
            let mut cav = Cavity::new(quantum, 20.0, kt);
            cav.freeze();
            for _ in 0..300 {
                cav.step(kt, 0.1);
            }
            let total: f64 = (0..cav.active).map(|k| cav.energy(k)).sum();
            let want = cav.expected_total(kt);
            assert!((total / want - 1.0).abs() < 0.03, "{quantum}: {total} vs {want}");
        }
    }
}
