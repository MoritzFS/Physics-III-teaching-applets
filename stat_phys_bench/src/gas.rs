//! Examples 2 and 3: a gas in a box, with or without internal levels.
//!
//! Hard spheres of mass m move in a unit box in d = 1, 2 or 3 dimensions
//! (in 1D each particle has its own lane). They collide elastically, and the
//! walls are either a heat bath or simply reflect.
//!
//! - A bath wall sends a particle back with a new velocity drawn at the wall's
//!   temperature: the normal component from the flux distribution
//!   ∝ v_n e^{−m v_n²/2k_BT} and the others from the Maxwell distribution. It
//!   also redraws the internal state from its Boltzmann distribution. This is
//!   an exact heat bath for the gas.
//! - In example 3 each atom has a ground level and an excited level Δ with g
//!   states. In a collision one of the two atoms (chosen at random) may change
//!   its level, paid for by the kinetic energy along the line of centres,
//!   K_n = μ w²/2 with μ = m/2 and w the normal relative speed. Excitation
//!   needs K_n ≥ Δ and happens with probability 1, de-excitation with 1/g.
//!   Because w dw is unchanged by w'² = w² ∓ 4Δ/m and the collision rate is
//!   ∝ w, this obeys detailed balance with the weights g e^{−E/k_BT}: an
//!   isolated gas settles to the Boltzmann distribution at its own kinetic
//!   temperature.

use crate::rng::Rng;

pub const MAX_ATOMS: usize = 50;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Walls {
    /// a heat bath at k_BT
    Bath,
    /// elastic: the gas keeps its energy
    Insulating,
}

/// the internal levels of example 3: ground at 0, Δ with g states
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TwoLevel {
    pub delta: f64,
    pub g: u32,
}

impl TwoLevel {
    pub fn excited_fraction(&self, kt: f64) -> f64 {
        let w = self.g as f64 * (-self.delta / kt).exp();
        w / (1.0 + w)
    }

    /// C_int / k_B = (βΔ)² g e^{−βΔ} / (1 + g e^{−βΔ})²
    pub fn heat_capacity(&self, kt: f64) -> f64 {
        let x = self.delta / kt;
        let w = self.g as f64 * (-x).exp();
        x * x * w / (1.0 + w).powi(2)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GasParams {
    pub n: usize,
    pub dim: usize,
    pub mass: f64,
    pub kt: f64,
    /// the particle radius (box side = 1); 0: no collisions
    pub radius: f64,
    pub walls: Walls,
    pub internal: Option<TwoLevel>,
}

impl GasParams {
    /// the spread of one velocity component, √(k_BT/m)
    pub fn sigma(&self) -> f64 {
        (self.kt / self.mass).sqrt()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Atom {
    pub r: [f64; 3],
    pub v: [f64; 3],
    pub excited: bool,
}

pub struct Gas {
    pub atoms: Vec<Atom>,
    rng: Rng,
    /// collisions between particles so far
    pub collisions: u64,
}

impl Gas {
    pub fn new(p: &GasParams, seed: u64) -> Self {
        let mut gas = Gas { atoms: vec![], rng: Rng::new(seed), collisions: 0 };
        gas.resize(p);
        gas
    }

    fn maxwell(&mut self, p: &GasParams) -> [f64; 3] {
        let s = p.sigma();
        let mut v = [0.0; 3];
        for c in v.iter_mut().take(p.dim) {
            *c = s * self.rng.normal();
        }
        v
    }

    fn thermal_state(&mut self, p: &GasParams, kt: f64) -> bool {
        p.internal.is_some_and(|tl| self.rng.uniform() < tl.excited_fraction(kt))
    }

    /// Removes the last atoms, or adds new ones in equilibrium where there is room.
    pub fn resize(&mut self, p: &GasParams) {
        self.atoms.truncate(p.n);
        let lo = p.radius;
        let hi = 1.0 - p.radius;
        while self.atoms.len() < p.n {
            let mut r = [0.5; 3];
            // a free spot (in 1D every atom has its own lane)
            for _attempt in 0..200 {
                for c in r.iter_mut().take(p.dim) {
                    *c = lo + (hi - lo) * self.rng.uniform();
                }
                if p.dim == 1 || self.atoms.iter().all(|a| dist2(&a.r, &r, p.dim) > (2.0 * p.radius).powi(2)) {
                    break;
                }
            }
            let v = self.maxwell(p);
            let excited = self.thermal_state(p, p.kt);
            self.atoms.push(Atom { r, v, excited });
        }
    }

    /// Fresh Maxwell velocities and internal states at k_BT.
    pub fn thermalise(&mut self, p: &GasParams) {
        for k in 0..self.atoms.len() {
            self.atoms[k].v = self.maxwell(p);
            self.atoms[k].excited = self.thermal_state(p, p.kt);
        }
    }

    /// Every atom gets the same speed (that of ⟨E⟩ = d k_BT/2) in a random direction, and the ground state.
    pub fn same_speed(&mut self, p: &GasParams) {
        let speed = (p.dim as f64 * p.kt / p.mass).sqrt();
        for k in 0..self.atoms.len() {
            let mut v = [0.0; 3];
            for c in v.iter_mut().take(p.dim) {
                *c = self.rng.normal();
            }
            let len = v.iter().map(|c| c * c).sum::<f64>().sqrt().max(1e-12);
            self.atoms[k].v = v.map(|c| c * speed / len);
            self.atoms[k].excited = false;
        }
    }

    /// Multiplies all velocities by `f` (energies by f²).
    pub fn scale_velocities(&mut self, f: f64) {
        for a in &mut self.atoms {
            a.v = a.v.map(|c| c * f);
        }
    }

    /// Puts the unused coordinates back to rest, e.g. after the dimension was changed.
    pub fn fit_dimension(&mut self, p: &GasParams) {
        let (lo, hi) = (p.radius, 1.0 - p.radius);
        for a in &mut self.atoms {
            for c in 0..3 {
                if c >= p.dim {
                    a.r[c] = 0.5;
                    a.v[c] = 0.0;
                } else {
                    a.r[c] = a.r[c].clamp(lo, hi);
                }
            }
        }
    }

    pub fn kinetic(&self, a: &Atom, mass: f64) -> f64 {
        0.5 * mass * a.v.iter().map(|c| c * c).sum::<f64>()
    }

    /// Advances the gas by the time dt.
    pub fn step(&mut self, p: &GasParams, dt: f64) {
        if self.atoms.is_empty() {
            return;
        }
        let collide = p.radius > 0.0 && p.dim >= 2 && self.atoms.len() >= 2;
        let vmax = self.atoms.iter().map(|a| a.v.iter().map(|c| c * c).sum::<f64>()).fold(0.0, f64::max).sqrt().max(1e-9);
        // a fraction of a radius per substep, so that no collision is missed (and walls are met in time)
        let h_max = if collide { 0.25 * p.radius / vmax } else { 0.05 / vmax };
        let steps = (dt / h_max).ceil().clamp(1.0, 4000.0) as usize;
        let h = dt / steps as f64;
        for _ in 0..steps {
            for a in &mut self.atoms {
                for c in 0..p.dim {
                    a.r[c] += a.v[c] * h;
                }
            }
            self.walls(p);
            if collide {
                self.collisions(p);
            }
        }
    }

    fn walls(&mut self, p: &GasParams) {
        let (lo, hi) = (p.radius, 1.0 - p.radius);
        let s = p.sigma();
        for k in 0..self.atoms.len() {
            for c in 0..p.dim {
                let x = self.atoms[k].r[c];
                let inward = if x < lo {
                    1.0
                } else if x > hi {
                    -1.0
                } else {
                    continue;
                };
                // mirror back into the box
                self.atoms[k].r[c] = if x < lo { 2.0 * lo - x } else { 2.0 * hi - x }.clamp(lo, hi);
                match p.walls {
                    Walls::Insulating => self.atoms[k].v[c] = inward * self.atoms[k].v[c].abs(),
                    Walls::Bath => {
                        // flux-weighted normal component (Rayleigh), Maxwell for the others
                        let vn = s * (-2.0 * self.rng.uniform_open().ln()).sqrt();
                        for t in 0..p.dim {
                            self.atoms[k].v[t] = if t == c { inward * vn } else { s * self.rng.normal() };
                        }
                        self.atoms[k].excited = self.thermal_state(p, p.kt);
                    }
                }
            }
        }
    }

    fn collisions(&mut self, p: &GasParams) {
        let d = p.dim;
        let contact2 = (2.0 * p.radius).powi(2);
        let n = self.atoms.len();
        for i in 0..n {
            for j in i + 1..n {
                let (ai, aj) = (self.atoms[i], self.atoms[j]);
                let dist2 = dist2(&ai.r, &aj.r, d);
                if dist2 >= contact2 || dist2 == 0.0 {
                    continue;
                }
                let dist = dist2.sqrt();
                let mut nrm = [0.0; 3];
                for (c, x) in nrm.iter_mut().enumerate().take(d) {
                    *x = (aj.r[c] - ai.r[c]) / dist;
                }
                let vin: f64 = (0..d).map(|c| ai.v[c] * nrm[c]).sum();
                let vjn: f64 = (0..d).map(|c| aj.v[c] * nrm[c]).sum();
                let w = vjn - vin;
                if w >= 0.0 {
                    // already moving apart
                    continue;
                }
                self.collisions += 1;
                // elastic: the normal relative velocity is reversed
                let mut w_after = -w;
                if let Some(tl) = p.internal {
                    let k_n = 0.25 * p.mass * w * w;
                    let which = if self.rng.uniform() < 0.5 { i } else { j };
                    if self.atoms[which].excited {
                        if self.rng.uniform() * (tl.g as f64) < 1.0 {
                            self.atoms[which].excited = false;
                            w_after = (4.0 * (k_n + tl.delta) / p.mass).sqrt();
                        }
                    } else if k_n >= tl.delta {
                        self.atoms[which].excited = true;
                        w_after = (4.0 * (k_n - tl.delta) / p.mass).sqrt();
                    }
                }
                // the centre of mass keeps its velocity; the relative velocity along n becomes w_after
                let cm = 0.5 * (vin + vjn);
                let (dvi, dvj) = (cm - 0.5 * w_after - vin, cm + 0.5 * w_after - vjn);
                for (c, x) in nrm.iter().enumerate().take(d) {
                    self.atoms[i].v[c] += dvi * x;
                    self.atoms[j].v[c] += dvj * x;
                }
            }
        }
    }
}

fn dist2(a: &[f64; 3], b: &[f64; 3], d: usize) -> f64 {
    (0..d).map(|c| (a[c] - b[c]).powi(2)).sum()
}

/// Γ(d/2) for d = 1, 2, 3
fn gamma_half(d: usize) -> f64 {
    match d {
        1 => std::f64::consts::PI.sqrt(),
        2 => 1.0,
        _ => 0.5 * std::f64::consts::PI.sqrt(),
    }
}

/// the speed distribution in d dimensions, f(v) ∝ v^{d−1} e^{−mv²/2k_BT}
pub fn speed_density(v: f64, d: usize, sigma: f64) -> f64 {
    let a = 2.0 * sigma * sigma;
    2.0 * v.powi(d as i32 - 1) / (a.powf(0.5 * d as f64) * gamma_half(d)) * (-v * v / a).exp()
}

/// one velocity component, a Gaussian of width σ = √(k_BT/m)
pub fn component_density(v: f64, sigma: f64) -> f64 {
    (-0.5 * v * v / (sigma * sigma)).exp() / (std::f64::consts::TAU.sqrt() * sigma)
}

/// the distribution of the kinetic energy, P(E) ∝ E^{d/2 − 1} e^{−E/k_BT}
pub fn energy_density(e: f64, d: usize, kt: f64) -> f64 {
    if e <= 0.0 {
        return if d == 2 {
            1.0 / kt
        } else if d == 1 {
            f64::INFINITY
        } else {
            0.0
        };
    }
    let h = 0.5 * d as f64;
    e.powf(h - 1.0) * (-e / kt).exp() / (gamma_half(d) * kt.powf(h))
}

/// most probable, mean and rms speed in d dimensions
pub fn speeds(d: usize, sigma: f64) -> [f64; 3] {
    let mean = match d {
        1 => (2.0 / std::f64::consts::PI).sqrt(),
        2 => (std::f64::consts::PI / 2.0).sqrt(),
        _ => (8.0 / std::f64::consts::PI).sqrt(),
    };
    [sigma * (d as f64 - 1.0).sqrt(), sigma * mean, sigma * (d as f64).sqrt()]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GasPreset {
    Maxwell,
    SameSpeed,
    Heavy,
    OneD,
    TwoD,
}

impl GasPreset {
    pub const ALL: [GasPreset; 5] = [GasPreset::Maxwell, GasPreset::SameSpeed, GasPreset::Heavy, GasPreset::OneD, GasPreset::TwoD];

    pub fn label(self) -> &'static str {
        match self {
            GasPreset::Maxwell => "Maxwell–Boltzmann in 3D",
            GasPreset::SameSpeed => "All start with the same speed",
            GasPreset::Heavy => "Heavier particles",
            GasPreset::OneD => "One dimension",
            GasPreset::TwoD => "Two dimensions",
        }
    }

    /// parameters, whether all atoms start with one speed, and notes
    pub fn setup(self) -> (GasParams, bool, &'static str) {
        let base = GasParams { n: 50, dim: 3, mass: 1.0, kt: 1.0, radius: 0.03, walls: Walls::Bath, internal: None };
        match self {
            GasPreset::Maxwell => (
                base,
                false,
                "Fifty particles in a box whose walls are a heat bath: a particle that hits a wall leaves with a new random velocity at the wall's temperature.\n\n\
                Z₁ = V/λ³ ∝ β^(−3/2): three quadratic terms (p_x², p_y², p_z²), so ⟨E⟩ = 3/2 k_BT and C = 3/2 k_B at every temperature.\n\n\
                The speeds follow Maxwell–Boltzmann, f(v) = 4π v² (m/2πk_BT)^(3/2) e^(−mv²/2k_BT), with\n\
                most probable speed √(2k_BT/m), mean √(8k_BT/πm) and rms √(3k_BT/m).\n\n\
                The v² comes from counting directions: there are more ways to move fast. Switch the left plot to one component, v_x: that is just a Gaussian.\n\n\
                With only 50 particles a snapshot is noisy, so the histograms average over about a minute.",
            ),
            GasPreset::SameSpeed => (
                GasParams { dim: 2, radius: 0.04, walls: Walls::Insulating, ..base },
                true,
                "Now the walls only reflect: the gas keeps its energy. All particles start with the same speed, in random directions.\n\n\
                Each collision shares energy between two particles. After a few collisions per particle the speeds are spread out as \
                Maxwell–Boltzmann, at the temperature set by the energy, k_BT = 2⟨E⟩/d. Nothing else is needed: almost all ways of \
                sharing the energy look like this.\n\n\
                Press 'same speed' to start again, and 'average: now' to watch the spreading. \
                (With 50 particles and a fixed total energy, small differences from the Boltzmann form remain.)",
            ),
            GasPreset::Heavy => (
                GasParams { mass: 4.0, ..base },
                false,
                "Four times the mass at the same temperature: all speeds are halved, v ∝ √(k_BT/m).\n\n\
                The energy distribution does not change: P(E) and ⟨E⟩ = 3/2 k_BT do not depend on m. \
                In a mixture of gases, light and heavy molecules have the same mean energy but different speeds.",
            ),
            GasPreset::OneD => (
                GasParams { dim: 1, ..base },
                false,
                "One dimension: each particle moves along x in its own lane. (Equal masses colliding head-on in 1D would only swap their \
                velocities, so collisions would change nothing here.)\n\n\
                Z₁ ∝ β^(−1/2): one quadratic term, so ⟨E⟩ = k_BT/2 and C = k_B/2. \
                The speed distribution is half a Gaussian, and P(E) ∝ E^(−1/2) e^(−E/k_BT): slow particles are the most common.",
            ),
            GasPreset::TwoD => (
                GasParams { dim: 2, ..base },
                false,
                "Two dimensions: Z₁ ∝ β^(−1), so ⟨E⟩ = k_BT and C = k_B.\n\n\
                f(v) ∝ v e^(−mv²/2k_BT) and the energy distribution is a plain exponential, P(E) = e^(−E/k_BT)/k_BT: \
                the same as one oscillator in example 4, which also has two quadratic terms.",
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtomPreset {
    Schottky,
    Isolated,
    Cold,
    Degenerate,
}

impl AtomPreset {
    pub const ALL: [AtomPreset; 4] = [AtomPreset::Schottky, AtomPreset::Isolated, AtomPreset::Cold, AtomPreset::Degenerate];

    pub fn label(self) -> &'static str {
        match self {
            AtomPreset::Schottky => "Two-level atoms near the bump",
            AtomPreset::Isolated => "Hot gas, no bath: the atoms get excited",
            AtomPreset::Cold => "Cold: never excited",
            AtomPreset::Degenerate => "Upper level with 3 states",
        }
    }

    pub fn setup(self) -> (GasParams, bool, &'static str) {
        let tl = TwoLevel { delta: 1.0, g: 1 };
        let base = GasParams { n: 50, dim: 3, mass: 1.0, kt: 0.42, radius: 0.04, walls: Walls::Bath, internal: Some(tl) };
        match self {
            AtomPreset::Schottky => (
                base,
                false,
                "Atoms that fly around and also have a ground level and an excited level Δ = 1 (orange). The energy is a sum, \
                E = p²/2m + ε_i, so Z factorises: Z₁ = (V/λ³) × (1 + g e^(−Δ/k_BT)).\n\n\
                ln Z adds, so ⟨E⟩ and C add: C = 3/2 k_B + C_int. The translation gives 3/2 at every temperature; \
                the internal levels add the Schottky bump of example 1, here at its peak (k_BT ≈ 0.42 Δ, C ≈ 3/2 + 0.44).\n\n\
                Factorising also means the motion does not care about the internal state: ground and excited atoms have the same speed distribution.\n\n\
                The walls are a heat bath for both the motion and the internal state. Atoms also excite and de-excite each other in collisions.",
            ),
            AtomPreset::Isolated => (
                GasParams { dim: 2, kt: 2.0, walls: Walls::Insulating, ..base },
                true,
                "No bath: the walls only reflect, and the total energy (kinetic + internal) is fixed. \
                The gas starts hot (k_BT = 2), with every atom in the ground state.\n\n\
                In a collision an atom can be excited if the pair brings at least Δ in its motion along the line of centres, \
                and an excited atom can give Δ back. Watch the energy flow from motion into the internal levels: \
                the kinetic temperature drops until the excited fraction matches it.\n\n\
                Part of the heat you put into a gas of such atoms goes into the internal levels instead of the motion: \
                that is the extra heat capacity C_int.\n\n\
                Press 'same speed' to start again.",
            ),
            AtomPreset::Cold => (
                GasParams { kt: 0.12, ..base },
                false,
                "Cold, k_BT = 0.12 Δ: the atoms move, but almost none is ever excited (g e^(−Δ/k_BT) ≈ 2 × 10⁻⁴).\n\n\
                Z₁ ≈ V/λ³ and C = 3/2 k_B: the internal levels are frozen out. Raise k_BT: excited atoms start to appear from about k_BT ≈ 0.3 Δ.",
            ),
            AtomPreset::Degenerate => (
                GasParams { kt: 3.0, internal: Some(TwoLevel { delta: 1.0, g: 3 }), ..base },
                false,
                "The excited level now has g = 3 states. Hot, the atoms spend up to g/(1 + g) = 3/4 of their time excited \
                (at k_BT = 3 Δ already 0.68).\n\n\
                Z_int → 1 + g = 4: it stops depending on β, so C_int → 0 and C returns to 3/2 k_B. \
                The factor 4 still adds k_B ln 4 to the entropy per atom.",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// kinetic plus internal energy of the whole gas
    fn total_energy(gas: &Gas, p: &GasParams) -> f64 {
        let delta = p.internal.map_or(0.0, |tl| tl.delta);
        gas.atoms.iter().map(|a| gas.kinetic(a, p.mass) + if a.excited { delta } else { 0.0 }).sum()
    }

    fn mean_kinetic(gas: &Gas, p: &GasParams) -> f64 {
        gas.atoms.iter().map(|a| gas.kinetic(a, p.mass)).sum::<f64>() / gas.atoms.len() as f64
    }

    #[test]
    fn densities_are_normalised() {
        for d in 1..=3 {
            let (s, kt) = (0.7, 1.3);
            let dv = 1e-3;
            let f: f64 = (0..20000).map(|k| speed_density((k as f64 + 0.5) * dv, d, s) * dv).sum();
            assert!((f - 1.0).abs() < 1e-3, "f(v), d = {d}: {f}");
            // in u = √E (dE = 2u du), so that E^{−1/2} in 1D is no problem
            let du = 1e-4;
            let e: f64 = (0..60_000).map(|k| (k as f64 + 0.5) * du).map(|u| energy_density(u * u, d, kt) * 2.0 * u * du).sum();
            assert!((e - 1.0).abs() < 2e-3, "P(E), d = {d}: {e}");
            // the rms speed squared is d k_BT/m
            let v2: f64 = (0..20000).map(|k| ((k as f64 + 0.5) * dv).powi(2) * speed_density((k as f64 + 0.5) * dv, d, s) * dv).sum();
            assert!((v2 - d as f64 * s * s).abs() < 1e-3);
            let m: f64 = (0..20000).map(|k| (k as f64 + 0.5) * dv * speed_density((k as f64 + 0.5) * dv, d, s) * dv).sum();
            assert!((m - speeds(d, s)[1]).abs() < 1e-3);
        }
    }

    #[test]
    fn bath_gives_equipartition() {
        for dim in 1..=3 {
            let p = GasParams { n: 50, dim, mass: 2.0, kt: 1.5, radius: 0.0, walls: Walls::Bath, internal: None };
            let mut gas = Gas::new(&p, 3);
            gas.same_speed(&GasParams { kt: 0.1, ..p });
            for _ in 0..400 {
                gas.step(&p, 0.05);
            }
            let mut sum = 0.0;
            let rounds = 2000;
            for _ in 0..rounds {
                gas.step(&p, 0.05);
                sum += mean_kinetic(&gas, &p);
            }
            let e = sum / rounds as f64;
            let want = 0.5 * dim as f64 * p.kt;
            assert!((e / want - 1.0).abs() < 0.04, "d = {dim}: {e} vs {want}");
        }
    }

    #[test]
    fn collisions_conserve_energy_and_spread_the_speeds() {
        let p = GasParams { n: 50, dim: 2, mass: 1.0, kt: 1.0, radius: 0.04, walls: Walls::Insulating, internal: None };
        let mut gas = Gas::new(&p, 4);
        gas.same_speed(&p);
        let e0 = total_energy(&gas, &p);
        for _ in 0..400 {
            gas.step(&p, 0.05);
        }
        assert!(gas.collisions > 1000, "{}", gas.collisions);
        // in 2D, v² is exponential at equilibrium: ⟨v⁴⟩/⟨v²⟩² = 2 (it starts at 1)
        let (mut s2, mut s4) = (0.0, 0.0);
        for _ in 0..1000 {
            gas.step(&p, 0.05);
            for a in &gas.atoms {
                let v2 = a.v[0] * a.v[0] + a.v[1] * a.v[1];
                s2 += v2;
                s4 += v2 * v2;
            }
        }
        let n = 1000.0 * 50.0;
        let ratio = (s4 / n) / (s2 / n).powi(2);
        assert!((ratio - 2.0).abs() < 0.15, "{ratio}");
        assert!((total_energy(&gas, &p) / e0 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn isolated_atoms_reach_boltzmann_at_their_kinetic_temperature() {
        for g in [1, 3] {
            let tl = TwoLevel { delta: 1.0, g };
            let p = GasParams { n: 50, dim: 2, mass: 1.0, kt: 1.5, radius: 0.04, walls: Walls::Insulating, internal: Some(tl) };
            let mut gas = Gas::new(&p, 5);
            gas.same_speed(&p);
            let e0 = total_energy(&gas, &p);
            for _ in 0..400 {
                gas.step(&p, 0.05);
            }
            let (mut exc, mut kin) = (0.0, 0.0);
            let rounds = 4000;
            for _ in 0..rounds {
                gas.step(&p, 0.05);
                exc += gas.atoms.iter().filter(|a| a.excited).count() as f64 / 50.0;
                kin += mean_kinetic(&gas, &p);
            }
            let (exc, kin) = (exc / rounds as f64, kin / rounds as f64);
            // k_BT = ⟨E_kin⟩ per atom in 2D
            let want = tl.excited_fraction(kin);
            assert!((exc - want).abs() < 0.02, "g = {g}: {exc} vs {want} at k_BT = {kin}");
            assert!((total_energy(&gas, &p) / e0 - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn bath_sets_the_excited_fraction() {
        let tl = TwoLevel { delta: 1.0, g: 2 };
        let p = GasParams { n: 50, dim: 3, mass: 1.0, kt: 0.6, radius: 0.04, walls: Walls::Bath, internal: Some(tl) };
        let mut gas = Gas::new(&p, 6);
        for _ in 0..200 {
            gas.step(&p, 0.05);
        }
        let mut exc = 0.0;
        let rounds = 3000;
        for _ in 0..rounds {
            gas.step(&p, 0.05);
            exc += gas.atoms.iter().filter(|a| a.excited).count() as f64 / 50.0;
        }
        let exc = exc / rounds as f64;
        assert!((exc - tl.excited_fraction(p.kt)).abs() < 0.02, "{exc}");
        // C peaks at about 0.44 k_B for g = 1 (example 1)
        let c = (100..1000).map(|k| TwoLevel { delta: 1.0, g: 1 }.heat_capacity(k as f64 / 1000.0)).fold(0.0, f64::max);
        assert!((c - 0.44).abs() < 0.005);
    }
}
