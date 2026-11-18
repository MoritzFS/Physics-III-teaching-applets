//! Bound states: a finite-difference eigen-solver for an arbitrary real V(x),
//! and the analytic harmonic-oscillator and infinite-well states (ħ = m = 1).
//!
//! The numerical Hamiltonian is the 3-point finite difference
//!
//! ```text
//! H_ii = 1/dx² + V_i,   H_i,i±1 = −1/(2dx²)       (ψ = 0 beyond the grid ends)
//! ```
//!
//! a symmetric tridiagonal matrix. Its lowest eigenvalues come from Sturm-count
//! bisection, the vectors from inverse iteration (re-orthogonalised inside
//! clusters of close eigenvalues, e.g. a deep double well). Errors are O(dx²).
//!
//! Conventions for [`EigenBasis`]:
//! * states are real and normalised on the grid, Σ ψ_i² dx = 1;
//! * index 0 is the ground state (for the infinite well that is n = 1);
//! * numerical states have their **first lobe positive** (the first point,
//!   from the left, where |ψ| exceeds 10⁻³ of its maximum is > 0);
//! * the analytic oscillator states use the textbook Hermite convention
//!   ψ_n(x → +∞) > 0, i.e. (−1)ⁿ times the numerical convention; the analytic
//!   well states are √(2/L) sin(nπ(x − a)/L), which agrees with "first lobe positive".

use std::f64::consts::PI;

use super::{Grid1D, Wave1D};

/// Energies and real, normalised eigenstates on a grid.
#[derive(Clone, Debug)]
pub struct EigenBasis {
    pub grid: Grid1D,
    /// ascending
    pub energies: Vec<f64>,
    /// `states[k][i]` = ψ_k(x_i)
    pub states: Vec<Vec<f64>>,
}

impl EigenBasis {
    /// The lowest `count` bound states of the real potential `v` (sampled on the grid).
    pub fn numeric(grid: Grid1D, v: &[f64], count: usize) -> Self {
        assert_eq!(v.len(), grid.n, "EigenBasis::numeric: potential has the wrong length");
        let h2 = 1.0 / (grid.dx * grid.dx);
        let diag: Vec<f64> = v.iter().map(|v| h2 + v).collect();
        let off = vec![-0.5 * h2; grid.n - 1];
        let (energies, mut states) = tridiagonal_lowest(&diag, &off, count.min(grid.n));
        for s in &mut states {
            let norm: f64 = s.iter().map(|x| x * x).sum::<f64>() * grid.dx;
            let scale = 1.0 / norm.sqrt();
            s.iter_mut().for_each(|x| *x *= scale);
            fix_sign_first_lobe(s);
        }
        EigenBasis { grid, energies, states }
    }

    /// Analytic oscillator states n = 0 … count−1 for V = ω²(x − x0)²/2 (textbook sign convention).
    pub fn harmonic(grid: Grid1D, count: usize, omega: f64, x0: f64) -> Self {
        let mut states = vec![vec![0.0; grid.n]; count];
        for i in 0..grid.n {
            let vals = ho_psi_all(count, grid.x(i) - x0, omega);
            for (state, v) in states.iter_mut().zip(vals) {
                state[i] = v;
            }
        }
        EigenBasis {
            grid,
            energies: (0..count).map(|n| ho_energy(n, omega)).collect(),
            states,
        }
    }

    /// Analytic infinite-well states n = 1 … count for walls at a and b (index k ↔ n = k + 1).
    pub fn infinite_well(grid: Grid1D, count: usize, a: f64, b: f64) -> Self {
        let l = b - a;
        let states = (1..=count).map(|n| grid.sample(|x| well_psi(n, x, a, b))).collect();
        EigenBasis {
            grid,
            energies: (1..=count).map(|n| well_energy(n, l)).collect(),
            states,
        }
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// State `k` as a [`Wave1D`].
    pub fn wave(&self, k: usize) -> Wave1D {
        Wave1D::from_real(self.grid, &self.states[k])
    }
}

/// E_n = ω(n + ½).
pub fn ho_energy(n: usize, omega: f64) -> f64 {
    omega * (n as f64 + 0.5)
}

/// Oscillator eigenfunction ψ_n(x) for V = ω²x²/2 (Hermite function,
/// ψ_n(+∞) > 0), by the stable three-term recurrence.
pub fn ho_psi(n: usize, x: f64, omega: f64) -> f64 {
    ho_psi_all(n + 1, x, omega)[n]
}

/// ψ_0(x) … ψ_{count−1}(x) of the oscillator.
pub fn ho_psi_all(count: usize, x: f64, omega: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(count);
    if count == 0 {
        return out;
    }
    let xi = omega.sqrt() * x;
    let p0 = (omega / PI).powf(0.25) * (-0.5 * xi * xi).exp();
    out.push(p0);
    if count > 1 {
        out.push(2f64.sqrt() * xi * p0);
    }
    for n in 1..count.saturating_sub(1) {
        let nf = n as f64;
        let next = (2.0 / (nf + 1.0)).sqrt() * xi * out[n] - (nf / (nf + 1.0)).sqrt() * out[n - 1];
        out.push(next);
    }
    out
}

/// E_n = n²π²/(2L²), n = 1, 2, …
pub fn well_energy(n: usize, l: f64) -> f64 {
    let nf = n as f64;
    nf * nf * PI * PI / (2.0 * l * l)
}

/// √(2/L) sin(nπ(x − a)/L) inside [a, b], 0 outside; n = 1, 2, …
pub fn well_psi(n: usize, x: f64, a: f64, b: f64) -> f64 {
    if x <= a || x >= b {
        return 0.0;
    }
    let l = b - a;
    (2.0 / l).sqrt() * (n as f64 * PI * (x - a) / l).sin()
}

/// Flips `v` so that its first significant value (from the left) is positive.
pub fn fix_sign_first_lobe(v: &mut [f64]) {
    let max = v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    if let Some(first) = v.iter().find(|x| x.abs() > 1e-3 * max)
        && *first < 0.0
    {
        v.iter_mut().for_each(|x| *x = -*x);
    }
}

/// Lowest `count` eigenpairs of the symmetric tridiagonal matrix with diagonal
/// `diag` (length n) and off-diagonal `off` (length n − 1).
/// Vectors are normalised to Σ v² = 1 (not yet sign-fixed).
pub fn tridiagonal_lowest(diag: &[f64], off: &[f64], count: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = diag.len();
    assert!(off.len() + 1 == n, "tridiagonal_lowest: off must have n − 1 entries");
    let count = count.min(n);
    // Gershgorin bounds
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for i in 0..n {
        let r = if i > 0 { off[i - 1].abs() } else { 0.0 } + if i + 1 < n { off[i].abs() } else { 0.0 };
        lo = lo.min(diag[i] - r);
        hi = hi.max(diag[i] + r);
    }
    let scale = hi.abs().max(lo.abs()).max(1e-300);
    let off2: Vec<f64> = off.iter().map(|e| e * e).collect();
    // number of eigenvalues < x
    let count_below = |x: f64| -> usize {
        let mut c = 0;
        let mut q = diag[0] - x;
        if q < 0.0 {
            c += 1;
        }
        for i in 1..n {
            let qq = if q == 0.0 { f64::EPSILON * scale } else { q };
            q = diag[i] - x - off2[i - 1] / qq;
            if q < 0.0 {
                c += 1;
            }
        }
        c
    };
    let mut values = Vec::with_capacity(count);
    let mut start = lo;
    for k in 0..count {
        let (mut a, mut b) = (start, hi);
        for _ in 0..200 {
            let mid = 0.5 * (a + b);
            if mid <= a || mid >= b || b - a <= 4.0 * f64::EPSILON * scale {
                break;
            }
            if count_below(mid) > k {
                b = mid;
            } else {
                a = mid;
            }
        }
        let lambda = 0.5 * (a + b);
        values.push(lambda);
        start = a;
    }

    // inverse iteration
    let cluster_tol = 1e-7 * scale;
    let mut vectors: Vec<Vec<f64>> = Vec::with_capacity(count);
    for (k, &lambda) in values.iter().enumerate() {
        // shift a hair off the eigenvalue so that the factorisation is not exactly singular
        let shift = lambda + 1e-12 * scale;
        let mut v: Vec<f64> = (0..n)
            .map(|i| 1.0 + 0.1 * ((i * 7919 + k * 104729) % 1000) as f64 / 1000.0)
            .collect();
        let cluster: Vec<usize> = (0..k).filter(|&j| (values[j] - lambda).abs() < cluster_tol).collect();
        for _ in 0..3 {
            v = solve_shifted_tridiagonal(diag, off, shift, &v);
            for &j in &cluster {
                let d: f64 = v.iter().zip(&vectors[j]).map(|(a, b)| a * b).sum();
                v.iter_mut().zip(&vectors[j]).for_each(|(a, b)| *a -= d * b);
            }
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm == 0.0 || !norm.is_finite() {
                break;
            }
            v.iter_mut().for_each(|x| *x /= norm);
        }
        vectors.push(v);
    }
    (values, vectors)
}

/// Solves (T − shift·I) x = b for the symmetric tridiagonal T (diag, off),
/// Gaussian elimination with partial pivoting.
fn solve_shifted_tridiagonal(diag: &[f64], off: &[f64], shift: f64, b: &[f64]) -> Vec<f64> {
    let n = diag.len();
    // rows: (l, d, u, u2) after pivoting; standard dgttrf-like scheme
    let mut dl: Vec<f64> = off.to_vec(); // sub-diagonal
    let mut d: Vec<f64> = diag.iter().map(|x| x - shift).collect();
    let mut du: Vec<f64> = off.to_vec(); // super-diagonal
    let mut du2 = vec![0.0; n.saturating_sub(2)];
    let mut ipiv = vec![false; n]; // true: rows i and i+1 swapped
    let tiny = f64::EPSILON * diag.iter().fold(1e-300f64, |m, x| m.max(x.abs()));
    for i in 0..n.saturating_sub(1) {
        if d[i].abs() >= dl[i].abs() {
            // no interchange
            if d[i] == 0.0 {
                d[i] = tiny;
            }
            let f = dl[i] / d[i];
            dl[i] = f;
            d[i + 1] -= f * du[i];
        } else {
            // interchange rows i and i+1
            let f = d[i] / dl[i];
            d[i] = dl[i];
            dl[i] = f;
            let temp = du[i];
            du[i] = d[i + 1];
            d[i + 1] = temp - f * d[i + 1];
            if i + 2 < n {
                du2[i] = du[i + 1];
                du[i + 1] *= -f;
            }
            ipiv[i] = true;
        }
    }
    if d[n - 1] == 0.0 {
        d[n - 1] = tiny;
    }
    // forward: L y = P b
    let mut x = b.to_vec();
    for i in 0..n.saturating_sub(1) {
        if ipiv[i] {
            let temp = x[i];
            x[i] = x[i + 1];
            x[i + 1] = temp - dl[i] * x[i];
        } else {
            x[i + 1] -= dl[i] * x[i];
        }
    }
    // back: U x = y
    x[n - 1] /= d[n - 1];
    if n > 1 {
        x[n - 2] = (x[n - 2] - du[n - 2] * x[n - 1]) / d[n - 2];
    }
    for i in (0..n.saturating_sub(2)).rev() {
        x[i] = (x[i] - du[i] * x[i + 1] - du2[i] * x[i + 2]) / d[i];
    }
    x
}
