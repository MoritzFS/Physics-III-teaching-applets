//! Stationary scattering from the left at energy E (ħ = m = 1):
//!
//! ```text
//! ψ = e^(ik_L x) + r e^(−ik_L x)   far left      k = √(2(E − V))
//! ψ = t e^(ik_R x)                 far right
//! T = (k_R/k_L)|t|²,  R = |r|²                    (T + R = 1 for real V)
//! ```
//!
//! Two methods:
//! * [`Piecewise::scatter`]: exact transfer matrices for a piecewise-constant
//!   potential (barriers, wells, steps);
//! * [`numerov_scatter`]: Numerov integration from right to left for an
//!   arbitrary V(x), as in PS10 41 (start with C = 1 at the right, integrate
//!   backwards, match to A e^(ikx) + B e^(−ikx) at the left).
//!
//! The incoming amplitude is normalised to 1 in both.

use super::C64;

const ZERO: C64 = C64::new(0.0, 0.0);
const I: C64 = C64::new(0.0, 1.0);

/// √(2(E − V)) on the principal branch (iκ in forbidden regions), never exactly 0.
fn wavenumber(e: f64, v: f64) -> C64 {
    let k = C64::new(2.0 * (e - v), 0.0).sqrt();
    if k.norm() < 1e-9 { C64::new(1e-9, 0.0) } else { k }
}

/// Piecewise-constant potential: `v[0]` left of `edges[0]`, `v[j]` between
/// `edges[j−1]` and `edges[j]`, `v[m]` right of the last edge.
#[derive(Clone, Debug, PartialEq)]
pub struct Piecewise {
    /// ascending, m entries
    pub edges: Vec<f64>,
    /// m + 1 values
    pub v: Vec<f64>,
}

/// Result of [`Piecewise::scatter`]: T, R and the stationary wavefunction.
#[derive(Clone, Debug)]
pub struct PiecewiseScattering {
    pub energy: f64,
    pub transmission: f64,
    pub reflection: f64,
    /// incoming amplitude (1), reflected amplitude r, …, per region (A_j, B_j);
    /// in region j, ψ = A_j e^(ik_j(x − x_j)) + B_j e^(−ik_j(x − x_j)) with x_j = `refs[j]`
    pub a: Vec<C64>,
    pub b: Vec<C64>,
    pub k: Vec<C64>,
    pub refs: Vec<f64>,
    pub edges: Vec<f64>,
}

impl Piecewise {
    /// General constructor (`v.len() == edges.len() + 1`, edges ascending).
    pub fn new(edges: Vec<f64>, v: Vec<f64>) -> Self {
        assert_eq!(v.len(), edges.len() + 1, "Piecewise: need one more value than edges");
        assert!(edges.windows(2).all(|w| w[0] <= w[1]), "Piecewise: edges must ascend");
        Piecewise { edges, v }
    }

    /// Rectangular barrier (or well, if v0 < 0) of height v0 on [x0, x1], 0 elsewhere.
    pub fn barrier(x0: f64, x1: f64, v0: f64) -> Self {
        Self::new(vec![x0, x1], vec![0.0, v0, 0.0])
    }

    /// Step from 0 to v0 at x0.
    pub fn step(x0: f64, v0: f64) -> Self {
        Self::new(vec![x0], vec![0.0, v0])
    }

    /// V(x).
    pub fn value(&self, x: f64) -> f64 {
        self.v[self.edges.partition_point(|&e| e <= x)]
    }

    /// V at the given positions.
    pub fn sample(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().map(|&x| self.value(x)).collect()
    }

    /// Scattering from the left at energy `energy` (> V far left).
    pub fn scatter(&self, energy: f64) -> PiecewiseScattering {
        let m = self.edges.len();
        let k: Vec<C64> = self.v.iter().map(|&v| wavenumber(energy, v)).collect();
        // reference point of region j: its left edge (region 0: the first edge)
        let refs: Vec<f64> = (0..=m)
            .map(|j| {
                if j == 0 {
                    self.edges.first().copied().unwrap_or(0.0)
                } else {
                    self.edges[j - 1]
                }
            })
            .collect();
        let mut a = vec![ZERO; m + 1];
        let mut b = vec![ZERO; m + 1];
        a[m] = C64::new(1.0, 0.0);
        for j in (0..m).rev() {
            let e = self.edges[j];
            // ψ and ψ' of region j+1 at its left edge e (= its reference point)
            let psi = a[j + 1] + b[j + 1];
            let dpsi = I * k[j + 1] * (a[j + 1] - b[j + 1]);
            let s = e - refs[j];
            let q = dpsi / (I * k[j]);
            a[j] = (-I * k[j] * s).exp() * (psi + q) * 0.5;
            b[j] = (I * k[j] * s).exp() * (psi - q) * 0.5;
        }
        let norm = a[0];
        for j in 0..=m {
            a[j] /= norm;
            b[j] /= norm;
        }
        let propagating_right = energy > self.v[m];
        let transmission = if propagating_right {
            k[m].re / k[0].re * a[m].norm_sqr()
        } else {
            0.0
        };
        let reflection = b[0].norm_sqr();
        PiecewiseScattering {
            energy,
            transmission,
            reflection,
            a,
            b,
            k,
            refs,
            edges: self.edges.clone(),
        }
    }
}

impl PiecewiseScattering {
    /// ψ(x) of the stationary state.
    pub fn psi(&self, x: f64) -> C64 {
        let j = self.edges.partition_point(|&e| e <= x);
        let d = x - self.refs[j];
        self.a[j] * (I * self.k[j] * d).exp() + self.b[j] * (-I * self.k[j] * d).exp()
    }

    /// ψ at the given positions.
    pub fn sample(&self, xs: &[f64]) -> Vec<C64> {
        xs.iter().map(|&x| self.psi(x)).collect()
    }
}

/// Analytic T(E) of a rectangular barrier of height v0 > 0 and width a (PS10 39):
/// E < V: 1/(1 + V² sinh²(κa)/(4E(V−E))); E > V: 1/(1 + V² sin²(ka)/(4E(E−V))).
pub fn rect_barrier_transmission(e: f64, v0: f64, a: f64) -> f64 {
    if e <= 0.0 {
        return 0.0;
    }
    let d = e - v0;
    if d.abs() < 1e-12 {
        // limit E → V: 1/(1 + m V a²/(2ħ²)) = 1/(1 + V a²/2)
        return 1.0 / (1.0 + v0 * a * a / 2.0);
    }
    if d < 0.0 {
        let kappa = (-2.0 * d).sqrt();
        1.0 / (1.0 + v0 * v0 * (kappa * a).sinh().powi(2) / (4.0 * e * -d))
    } else {
        let k = (2.0 * d).sqrt();
        1.0 / (1.0 + v0 * v0 * (k * a).sin().powi(2) / (4.0 * e * d))
    }
}

/// Result of [`numerov_scatter`].
#[derive(Clone, Debug)]
pub struct NumerovScattering {
    pub energy: f64,
    pub transmission: f64,
    pub reflection: f64,
    /// incoming (A) and reflected (B) amplitudes before normalising A to 1
    pub a_raw: C64,
    pub b_raw: C64,
    /// grid x_0 … x_N (uniform, step dx)
    pub xs: Vec<f64>,
    /// ψ on that grid, normalised so that the incoming amplitude is 1
    pub psi: Vec<C64>,
}

/// Numerov scattering for an arbitrary V(x) on [x_left, x_right] with step dx.
///
/// V should be constant over the first two and the last two grid points
/// (V_L and V_R, e.g. 0); E must exceed V_L. For E ≤ V_R the transmission is 0.
pub fn numerov_scatter(x_left: f64, x_right: f64, dx: f64, v: impl Fn(f64) -> f64, energy: f64) -> NumerovScattering {
    let n = ((x_right - x_left) / dx).round().max(3.0) as usize;
    let xs: Vec<f64> = (0..=n).map(|i| x_left + i as f64 * dx).collect();
    let kk: Vec<f64> = xs.iter().map(|&x| 2.0 * (energy - v(x))).collect();
    let k_l = wavenumber(energy, v(x_left));
    let k_r = wavenumber(energy, v(x_right));
    let mut psi = vec![ZERO; n + 1];
    // C = 1: transmitted wave (or decaying for E < V_R) at the right end
    psi[n] = (I * k_r * xs[n]).exp();
    psi[n - 1] = (I * k_r * xs[n - 1]).exp();
    let h2 = dx * dx / 12.0;
    for i in (1..n).rev() {
        let num = psi[i] * (2.0 * (1.0 - 5.0 * h2 * kk[i])) - psi[i + 1] * (1.0 + h2 * kk[i + 1]);
        psi[i - 1] = num / (1.0 + h2 * kk[i - 1]);
    }
    // match ψ_0, ψ_1 to A e^(ikx) + B e^(−ikx)
    let (x0, x1) = (xs[0], xs[1]);
    let (e0p, e0m) = ((I * k_l * x0).exp(), (-I * k_l * x0).exp());
    let (e1p, e1m) = ((I * k_l * x1).exp(), (-I * k_l * x1).exp());
    let det = e0p * e1m - e0m * e1p;
    let a = (psi[0] * e1m - psi[1] * e0m) / det;
    let b = (e0p * psi[1] - e1p * psi[0]) / det;
    let transmission = if energy > v(x_right) {
        (k_r.re / k_l.re) / a.norm_sqr()
    } else {
        0.0
    };
    let reflection = (b / a).norm_sqr();
    psi.iter_mut().for_each(|p| *p /= a);
    NumerovScattering {
        energy,
        transmission,
        reflection,
        a_raw: a,
        b_raw: b,
        xs,
        psi,
    }
}
