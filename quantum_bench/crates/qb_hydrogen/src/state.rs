//! The atom's state: ψ(t) = Σ c_nlm e^(−iE_nm (t − t₀)) ψ_nlm (atomic units, ħ = 1),
//! and the projective measurements of E, l², l_z and l_y on it.

use qb_core::{C64, QbRng, c64};

use crate::orbital::{BASIS, Coeffs, N_BASIS, Nlm, active, energy_ev, ly_eigen_all};

/// A superposition of the 30 eigenstates with n ≤ 4.
#[derive(Clone, Debug)]
pub struct AtomState {
    /// coefficients at time `t_ref`
    c_ref: Coeffs,
    t_ref: f64,
    /// magnetic field along z in atomic units (normal Zeeman effect: E = E_n + μ_B B m, μ_B = ½)
    pub b_au: f64,
}

/// Normalises c in place (no-op for the zero vector).
pub fn normalize(c: &mut Coeffs) {
    let n: f64 = c.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    if n > 0.0 {
        c.iter_mut().for_each(|z| *z /= n);
    }
}

/// An empty coefficient vector.
pub fn zero() -> Coeffs {
    [c64(0.0, 0.0); N_BASIS]
}

impl AtomState {
    /// The state Σ c ψ at t (c is normalised).
    pub fn new(mut c: Coeffs, t: f64) -> Self {
        normalize(&mut c);
        AtomState { c_ref: c, t_ref: t, b_au: 0.0 }
    }

    pub fn eigen(q: Nlm, t: f64) -> Self {
        let mut c = zero();
        c[q.index()] = c64(1.0, 0.0);
        Self::new(c, t)
    }

    /// Energy of basis state i in Hartree, including the normal Zeeman shift μ_B B m.
    pub fn energy(&self, i: usize) -> f64 {
        let q = BASIS[i];
        q.energy() + 0.5 * self.b_au * q.m as f64
    }

    pub fn energies(&self) -> [f64; N_BASIS] {
        std::array::from_fn(|i| self.energy(i))
    }

    /// Coefficients at time t.
    pub fn coeffs(&self, t: f64) -> Coeffs {
        let dt = t - self.t_ref;
        std::array::from_fn(|i| {
            let c = self.c_ref[i];
            if c.norm_sqr() == 0.0 { c } else { c * C64::from_polar(1.0, -self.energy(i) * dt) }
        })
    }

    /// Replace the state at time t (after a collapse); c is normalised.
    pub fn set(&mut self, mut c: Coeffs, t: f64) {
        normalize(&mut c);
        self.c_ref = c;
        self.t_ref = t;
    }

    pub fn active(&self) -> Vec<usize> {
        active(&self.c_ref)
    }

    /// The single eigenstate, if the state is one.
    pub fn single(&self) -> Option<Nlm> {
        match self.active().as_slice() {
            [k] => Some(BASIS[*k]),
            _ => None,
        }
    }

    /// The (n, l) shared by all components, if any.
    pub fn single_nl(&self) -> Option<(u32, u32)> {
        let act = self.active();
        let q0 = BASIS[*act.first()?];
        act.iter().all(|&k| BASIS[k].n == q0.n && BASIS[k].l == q0.l).then_some((q0.n, q0.l))
    }

    /// All components have the same energy (the density does not change in time).
    pub fn stationary(&self) -> bool {
        let act = self.active();
        act.iter().all(|&k| (self.energy(k) - self.energy(act[0])).abs() < 1e-12)
    }

    /// Largest Bohr angular frequency |E_a − E_b| among the components (0 if stationary).
    pub fn max_bohr_frequency(&self) -> f64 {
        let act = self.active();
        let mut w: f64 = 0.0;
        for &a in &act {
            for &b in &act {
                w = w.max((self.energy(a) - self.energy(b)).abs());
            }
        }
        w
    }
}

/// What can be measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Obs {
    Energy,
    L2,
    Lz,
    Ly,
    Position,
}

impl Obs {
    pub fn symbol(self) -> &'static str {
        match self {
            Obs::Energy => "E",
            Obs::L2 => "l²",
            Obs::Lz => "l_z",
            Obs::Ly => "l_y",
            Obs::Position => "position",
        }
    }

    pub fn parse(s: &str) -> Option<Obs> {
        Some(match s {
            "e" | "energy" => Obs::Energy,
            "l2" => Obs::L2,
            "lz" => Obs::Lz,
            "ly" => Obs::Ly,
            "pos" | "x" | "position" => Obs::Position,
            _ => return None,
        })
    }
}

/// One possible outcome of a discrete measurement.
#[derive(Clone, Debug)]
pub struct Outcome {
    /// eigenvalue (Hartree for E, l(l+1) for l², m for l_z and l_y; ħ = 1)
    pub value: f64,
    pub prob: f64,
    pub label: String,
}

fn outcome_label(obs: Obs, value: f64) -> String {
    match obs {
        Obs::Energy => {
            let ev = value * crate::orbital::HARTREE_EV;
            // name the level n if it is a field-free level
            match (1..=4u32).find(|&n| (energy_ev(n) - ev).abs() < 1e-3) {
                Some(n) => format!("E{} = {ev:.2} eV", crate::orbital::sub(n as i64)),
                None => format!("E = {ev:.4} eV"),
            }
        }
        Obs::L2 => {
            let l = ((-1.0 + (1.0 + 4.0 * value).sqrt()) / 2.0).round() as i64;
            format!("l = {l}  ({}ħ²)", value.round())
        }
        Obs::Lz => format!("m = {}  ({}ħ)", crate::orbital::signed(value.round() as i32), crate::orbital::signed(value.round() as i32)),
        Obs::Ly => format!("m_y = {}  ({}ħ)", crate::orbital::signed(value.round() as i32), crate::orbital::signed(value.round() as i32)),
        Obs::Position => String::new(),
    }
}

/// The eigenvalue of basis state i for the diagonal observables.
fn diag_value(obs: Obs, i: usize, energies: &[f64; N_BASIS]) -> f64 {
    let q = BASIS[i];
    match obs {
        Obs::Energy => energies[i],
        Obs::L2 => (q.l * (q.l + 1)) as f64,
        Obs::Lz => q.m as f64,
        _ => f64::NAN,
    }
}

/// Amplitudes ⟨n l, m_y| ψ⟩ of the l_y eigenvectors in each (n, l) block:
/// (n, l, index of the eigenvector, m_y, amplitude).
fn ly_amplitudes(c: &Coeffs) -> Vec<(u32, u32, usize, f64, C64)> {
    let eig = ly_eigen_all();
    let mut out = vec![];
    for n in 1..=4u32 {
        for l in 0..n {
            let any = (-(l as i32)..=l as i32).any(|m| c[Nlm::new(n, l, m).index()].norm_sqr() > 0.0);
            if !any {
                continue;
            }
            for (k, (val, vec)) in eig[l as usize].iter().enumerate() {
                let amp: C64 = vec
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v.conj() * c[Nlm::new(n, l, i as i32 - l as i32).index()])
                    .sum();
                out.push((n, l, k, *val, amp));
            }
        }
    }
    out
}

/// Possible outcomes (distinct eigenvalues, ascending) with their Born probabilities.
pub fn outcomes(c: &Coeffs, obs: Obs, energies: &[f64; N_BASIS]) -> Vec<Outcome> {
    let norm: f64 = c.iter().map(|z| z.norm_sqr()).sum::<f64>().max(1e-300);
    let mut list: Vec<(f64, f64)> = vec![];
    let mut add = |value: f64, p: f64| match list.iter_mut().find(|(v, _)| (v - value).abs() < 1e-9) {
        Some(e) => e.1 += p,
        None => list.push((value, p)),
    };
    match obs {
        Obs::Ly => {
            for (_, _, _, val, amp) in ly_amplitudes(c) {
                add(val, amp.norm_sqr() / norm);
            }
        }
        Obs::Position => {}
        _ => {
            for (i, ci) in c.iter().enumerate() {
                if ci.norm_sqr() > 0.0 {
                    add(diag_value(obs, i, energies), ci.norm_sqr() / norm);
                }
            }
        }
    }
    list.sort_by(|a, b| a.0.total_cmp(&b.0));
    list.into_iter()
        .map(|(value, prob)| Outcome {
            value,
            prob,
            label: outcome_label(obs, value),
        })
        .collect()
}

/// Projects c onto the eigenspace of `obs` with eigenvalue `value` (normalised).
pub fn project(c: &Coeffs, obs: Obs, value: f64, energies: &[f64; N_BASIS]) -> Coeffs {
    let mut out = zero();
    match obs {
        Obs::Ly => {
            let eig = ly_eigen_all();
            for (n, l, k, val, amp) in ly_amplitudes(c) {
                if (val - value).abs() > 1e-9 {
                    continue;
                }
                for (i, v) in eig[l as usize][k].1.iter().enumerate() {
                    out[Nlm::new(n, l, i as i32 - l as i32).index()] += amp * v;
                }
            }
        }
        Obs::Position => return *c,
        _ => {
            for i in 0..N_BASIS {
                if (diag_value(obs, i, energies) - value).abs() < 1e-9 {
                    out[i] = c[i];
                }
            }
        }
    }
    // drop numerical dust
    for z in out.iter_mut() {
        if z.norm_sqr() < 1e-24 {
            *z = c64(0.0, 0.0);
        }
    }
    normalize(&mut out);
    out
}

/// One measurement: draws an outcome with the Born rule and returns it with the collapsed state.
pub fn measure(c: &Coeffs, obs: Obs, energies: &[f64; N_BASIS], rng: &mut QbRng) -> Option<(Outcome, Coeffs)> {
    let outs = outcomes(c, obs, energies);
    let probs: Vec<f64> = outs.iter().map(|o| o.prob).collect();
    let k = rng.weighted(&probs)?;
    let o = outs[k].clone();
    let after = project(c, obs, o.value, energies);
    Some((o, after))
}

/// ⟨O⟩ and the spread ΔO = √(⟨O²⟩ − ⟨O⟩²) of a discrete observable.
pub fn mean_and_spread(c: &Coeffs, obs: Obs, energies: &[f64; N_BASIS]) -> (f64, f64) {
    let outs = outcomes(c, obs, energies);
    let m: f64 = outs.iter().map(|o| o.prob * o.value).sum();
    let m2: f64 = outs.iter().map(|o| o.prob * o.value * o.value).sum();
    (m, (m2 - m * m).max(0.0).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbital::Nlm;

    fn ps11_45() -> Coeffs {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let mut c = zero();
        c[Nlm::new(2, 1, 1).index()] = c64(s, 0.0);
        c[Nlm::new(2, 1, -1).index()] = c64(s, 0.0);
        c
    }

    #[test]
    fn lz_and_ly_on_the_ps11_45_state() {
        let st = AtomState::new(ps11_45(), 0.0);
        let c = st.coeffs(0.0);
        let e = st.energies();
        // l_z: ±1 with probability ½ each
        let lz = outcomes(&c, Obs::Lz, &e);
        assert_eq!(lz.len(), 2);
        assert!((lz[0].value + 1.0).abs() < 1e-12 && (lz[0].prob - 0.5).abs() < 1e-12);
        assert!((lz[1].value - 1.0).abs() < 1e-12 && (lz[1].prob - 0.5).abs() < 1e-12);
        // after l_z = +1 the state is ψ₂₁₁
        let after = project(&c, Obs::Lz, 1.0, &e);
        assert!((after[Nlm::new(2, 1, 1).index()].norm() - 1.0).abs() < 1e-12);
        // l_y: certain outcome 0, state unchanged
        let ly = outcomes(&c, Obs::Ly, &e);
        let p0: f64 = ly.iter().filter(|o| o.value.abs() < 1e-9).map(|o| o.prob).sum();
        assert!((p0 - 1.0).abs() < 1e-12, "{ly:?}");
        let after = project(&c, Obs::Ly, 0.0, &e);
        for i in 0..N_BASIS {
            assert!((after[i] - c[i]).norm() < 1e-12);
        }
        // E: certain, E₂
        let en = outcomes(&c, Obs::Energy, &e);
        assert_eq!(en.len(), 1);
        assert!((en[0].value + 0.125).abs() < 1e-12);
        // part c: the density is stationary, only a global phase turns
        assert!(st.stationary());
        let ct = st.coeffs(7.3);
        let ph = ct[Nlm::new(2, 1, 1).index()] / c[Nlm::new(2, 1, 1).index()];
        assert!((ct[Nlm::new(2, 1, -1).index()] - c[Nlm::new(2, 1, -1).index()] * ph).norm() < 1e-12);
        assert!((ph - C64::from_polar(1.0, 0.125 * 7.3)).norm() < 1e-12);
    }

    #[test]
    fn energy_measurement_projects_onto_the_n_subspace() {
        let mut c = zero();
        c[Nlm::new(2, 0, 0).index()] = c64(0.6, 0.0);
        c[Nlm::new(2, 1, 1).index()] = c64(0.0, 0.48);
        c[Nlm::new(1, 0, 0).index()] = c64(0.64, 0.0);
        let st = AtomState::new(c, 0.0);
        let e = st.energies();
        let c = st.coeffs(0.0);
        let outs = outcomes(&c, Obs::Energy, &e);
        assert_eq!(outs.len(), 2);
        let after = project(&c, Obs::Energy, energy_n(2), &e);
        // both n = 2 components survive with their relative amplitude and phase
        let a = after[Nlm::new(2, 0, 0).index()];
        let b = after[Nlm::new(2, 1, 1).index()];
        assert!((b / a - c64(0.0, 0.8)).norm() < 1e-12);
        assert!(after[0].norm() < 1e-15);
    }

    fn energy_n(n: u32) -> f64 {
        crate::orbital::energy(n)
    }

    #[test]
    fn measurement_statistics() {
        let mut rng = QbRng::seed(9);
        let st = AtomState::new(ps11_45(), 0.0);
        let c = st.coeffs(0.0);
        let e = st.energies();
        let n = 20_000;
        let ups = (0..n).filter(|_| measure(&c, Obs::Lz, &e, &mut rng).unwrap().0.value > 0.0).count();
        assert!((ups as f64 / n as f64 - 0.5).abs() < 0.015);
    }

    #[test]
    fn zeeman_energies_and_selection_rules() {
        // PS12 47: l = 2 (upper, n = 3) and l = 1 (lower, n = 2): 9 dipole-allowed transitions
        let mut st = AtomState::eigen(Nlm::new(2, 1, 0), 0.0);
        st.b_au = 1e-3;
        let base = st.energy(Nlm::new(2, 1, 0).index());
        assert!((st.energy(Nlm::new(2, 1, 1).index()) - base - 0.5e-3).abs() < 1e-15);
        let mut count = 0;
        let mut by_dm = [0; 3];
        for m1 in -1..=1 {
            for m2 in -2..=2 {
                let mut c = zero();
                c[Nlm::new(2, 1, m1).index()] = c64(1.0, 0.0);
                c[Nlm::new(3, 2, m2).index()] = c64(1.0, 0.0);
                let x = crate::orbital::expect(&c, 1, crate::orbital::AngOp::X).norm();
                let y = crate::orbital::expect(&c, 1, crate::orbital::AngOp::Y).norm();
                let z = crate::orbital::expect(&c, 1, crate::orbital::AngOp::Z).norm();
                let allowed = x + y + z > 1e-8;
                assert_eq!(allowed, (m2 - m1).abs() <= 1, "m1 = {m1}, m2 = {m2}");
                if allowed {
                    count += 1;
                    by_dm[(m2 - m1 + 1) as usize] += 1;
                    // π: dipole along z only; σ±: in the x–y plane only
                    if m2 == m1 {
                        assert!(x < 1e-10 && y < 1e-10 && z > 1e-3);
                    } else {
                        assert!(z < 1e-10 && x + y > 1e-3);
                    }
                }
            }
        }
        assert_eq!(count, 9);
        assert_eq!(by_dm, [3, 3, 3]);
        // the relative strengths: each Δm group carries the same total |⟨r_q⟩|²
        let strength = |dm: i32| -> f64 {
            let mut s = 0.0;
            for m1 in -1..=1i32 {
                let m2 = m1 + dm;
                if m2.abs() > 2 {
                    continue;
                }
                let mut c = zero();
                c[Nlm::new(2, 1, m1).index()] = c64(1.0, 0.0);
                c[Nlm::new(3, 2, m2).index()] = c64(1.0, 0.0);
                // ⟨r⟩ of (a + b)/√2 is Re⟨a|r|b⟩ per component; use x, y, z magnitudes
                let x = crate::orbital::expect(&c, 1, crate::orbital::AngOp::X).re;
                let y = crate::orbital::expect(&c, 1, crate::orbital::AngOp::Y).re;
                let z = crate::orbital::expect(&c, 1, crate::orbital::AngOp::Z).re;
                s += x * x + y * y + z * z;
            }
            s
        };
        // sum rule Σ|⟨r_q⟩|² equal for q = 0, ±1; |⟨r⟩|² is |r₀|² for π but |r₊₁|²/2 for σ
        // (rotation radius), so the linear amplitude² is twice the rotating one
        let (sm, s0, sp) = (strength(-1), strength(0), strength(1));
        assert!((sm - sp).abs() < 1e-10);
        assert!((s0 - 2.0 * sp).abs() / s0 < 1e-9, "Δm = 0: {s0}, ±1: {sp}");
    }
}
