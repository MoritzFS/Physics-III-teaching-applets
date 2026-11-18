//! Hydrogen eigenfunctions in atomic units (a₀ = 1, energies in Hartree E_h):
//! ψ_nlm(r, θ, φ) = R_nl(r) Y_lm(θ, φ) for n ≤ 4 (all l, m: 30 states).
//!
//! * R_nl(r) = N_nl e^(−r/n) (2r/n)^l L_(n−l−1)^(2l+1)(2r/n), with the associated
//!   Laguerre polynomials L_k^α(x) = Σ_j (−1)^j C(k+α, k−j) x^j / j!, so that
//!   R_nl(r) > 0 near r = 0 (R₁₀ = 2e^(−r), R₂₁ = r e^(−r/2)/(2√6), …).
//! * Y_lm(θ, φ) = Θ_lm(cos θ) e^(imφ) with the Condon–Shortley phase, as in the
//!   problem sets (PS11 Eq. 44.4: Y₁,±₁ = ∓√(3/8π) sin θ e^(±iφ)), i.e.
//!   Y_l,−m = (−1)^m Y_lm*.
//! * E_n = −1/(2n²) E_h = −13.6 eV/n² (infinite nuclear mass, no spin).
//!
//! Every ψ_nlm is stored as a polynomial times exponentials,
//! ψ = norm · e^(−r/n) r^l P_rad(r) · sin^|m|θ P_ang(cos θ) · e^(imφ),
//! which is what the GPU ray marcher evaluates (same coefficients).
//!
//! The module also holds the tables of radial and angular integrals (matrix
//! elements of r^p, x, y, z, x², …), the angular momentum operators acting on
//! coefficient vectors, the l_y eigenvectors, and exact Born-rule sampling of
//! positions for any superposition.

use std::f64::consts::{PI, TAU};
use std::sync::OnceLock;

use qb_core::{C64, QbRng, c64};

/// Largest principal quantum number.
pub const N_MAX: u32 = 4;
/// Number of states with n ≤ 4.
pub const N_BASIS: usize = 30;
/// Number of (n, l) pairs with n ≤ 4.
pub const N_NL: usize = 10;
/// Number of (l, m) pairs with l ≤ 3.
pub const N_LM: usize = 16;
/// Rydberg energy as used in the problem sets: E_n = −13.6 eV/n² (1 E_h = 27.2 eV).
pub const RY_EV: f64 = 13.6;
/// Hartree in eV (consistent with RY_EV)
pub const HARTREE_EV: f64 = 2.0 * RY_EV;
/// hc in eV·nm
pub const HC_EV_NM: f64 = 1239.842;
/// atomic unit of time ħ/E_h in femtoseconds
pub const AU_TIME_FS: f64 = 0.024_188_843;
/// Bohr magneton in eV/T
pub const MU_B_EV_PER_T: f64 = 5.788_381_8e-5;
/// μ_B B/h in GHz per tesla
pub const MU_B_GHZ_PER_T: f64 = 13.996_245;
/// the atomic unit of magnetic field in tesla (ħ/(e a₀²))
pub const AU_FIELD_T: f64 = 2.350_517_57e5;

/// Coefficients c_nlm of a state over the 30 basis states (index [`Nlm::index`]).
pub type Coeffs = [C64; N_BASIS];

/// Quantum numbers (n, l, m).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Nlm {
    pub n: u32,
    pub l: u32,
    pub m: i32,
}

/// All states with n ≤ 4 in the order n, then l, then m = −l … l.
pub const BASIS: [Nlm; N_BASIS] = build_basis();

const fn build_basis() -> [Nlm; N_BASIS] {
    let mut out = [Nlm { n: 0, l: 0, m: 0 }; N_BASIS];
    let mut i = 0;
    let mut n = 1;
    while n <= N_MAX {
        let mut l = 0;
        while l < n {
            let mut m = -(l as i32);
            while m <= l as i32 {
                out[i] = Nlm { n, l, m };
                i += 1;
                m += 1;
            }
            l += 1;
        }
        n += 1;
    }
    out
}

impl Nlm {
    pub const fn new(n: u32, l: u32, m: i32) -> Self {
        Nlm { n, l, m }
    }

    pub fn is_valid(&self) -> bool {
        self.n >= 1 && self.n <= N_MAX && self.l < self.n && self.m.unsigned_abs() <= self.l
    }

    /// Position in [`BASIS`].
    pub fn index(&self) -> usize {
        let n = self.n as usize;
        (n - 1) * n * (2 * n - 1) / 6 + (self.l * self.l) as usize + (self.m + self.l as i32) as usize
    }

    /// Index of (n, l) among the 10 radial functions.
    pub fn nl(&self) -> usize {
        nl_index(self.n, self.l)
    }

    /// Index of (l, m) among the 16 spherical harmonics.
    pub fn lm(&self) -> usize {
        ((self.l * self.l + self.l) as i32 + self.m) as usize
    }

    /// E_n in Hartree (no field).
    pub fn energy(&self) -> f64 {
        energy(self.n)
    }

    /// "ψ₂₁₋₁"
    pub fn name(&self) -> String {
        format!("ψ{}{}{}", sub(self.n as i64), sub(self.l as i64), sub(self.m as i64))
    }

    /// "2p" (spectroscopic name of n, l)
    pub fn shell(&self) -> String {
        format!("{}{}", self.n, l_letter(self.l))
    }
}

/// Index of (n, l): n(n−1)/2 + l.
pub fn nl_index(n: u32, l: u32) -> usize {
    (n * (n - 1) / 2 + l) as usize
}

/// E_n = −1/(2n²) Hartree.
pub fn energy(n: u32) -> f64 {
    -0.5 / (n * n) as f64
}

/// E_n = −13.6 eV / n².
pub fn energy_ev(n: u32) -> f64 {
    -RY_EV / (n * n) as f64
}

/// "s", "p", "d", "f"
pub fn l_letter(l: u32) -> &'static str {
    ["s", "p", "d", "f", "g"][l.min(4) as usize]
}

/// Integer as subscript digits ("₋₁").
pub fn sub(v: i64) -> String {
    v.to_string()
        .chars()
        .map(|c| match c {
            '-' => '₋',
            d => char::from_u32(0x2080 + d.to_digit(10).unwrap_or(0)).unwrap_or(d),
        })
        .collect()
}

/// "+1", "0", "−2" with a real minus sign.
pub fn signed(v: i32) -> String {
    match v.cmp(&0) {
        std::cmp::Ordering::Greater => format!("+{v}"),
        std::cmp::Ordering::Equal => "0".into(),
        std::cmp::Ordering::Less => format!("−{}", -v),
    }
}

// ------------------------------------------------------------------ special functions

fn factorial(n: u32) -> f64 {
    (1..=n).map(|k| k as f64).product()
}

fn binom(n: u32, k: u32) -> f64 {
    if k > n {
        return 0.0;
    }
    factorial(n) / (factorial(k) * factorial(n - k))
}

/// Power-series coefficients of the associated Laguerre polynomial L_k^α(x).
pub fn laguerre_coeffs(k: u32, alpha: u32) -> Vec<f64> {
    (0..=k)
        .map(|j| {
            let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
            sign * binom(k + alpha, k - j) / factorial(j)
        })
        .collect()
}

/// L_k^α(x) by the three-term recurrence (independent of [`laguerre_coeffs`], used in tests).
pub fn laguerre_rec(k: u32, alpha: u32, x: f64) -> f64 {
    let a = alpha as f64;
    let mut l0 = 1.0;
    if k == 0 {
        return l0;
    }
    let mut l1 = 1.0 + a - x;
    for j in 1..k {
        let j = j as f64;
        let l2 = ((2.0 * j + 1.0 + a - x) * l1 - (j + a) * l0) / (j + 1.0);
        l0 = l1;
        l1 = l2;
    }
    l1
}

/// Associated Legendre function P_l^m(x) for m ≥ 0, **without** the Condon–Shortley
/// phase: P_l^m = (1 − x²)^(m/2) dᵐP_l/dxᵐ (recurrence; used in tests and for Y_lm).
pub fn legendre_rec(l: u32, m: u32, x: f64) -> f64 {
    if m > l {
        return 0.0;
    }
    let s = (1.0 - x * x).max(0.0).sqrt();
    // P_m^m = (2m − 1)!! s^m
    let mut pmm = 1.0;
    for i in 0..m {
        pmm *= (2 * i + 1) as f64 * s;
    }
    if l == m {
        return pmm;
    }
    let mut pm1 = x * (2 * m + 1) as f64 * pmm;
    if l == m + 1 {
        return pm1;
    }
    let mut p = 0.0;
    for ll in (m + 2)..=l {
        p = ((2 * ll - 1) as f64 * x * pm1 - (ll + m - 1) as f64 * pmm) / (ll - m) as f64;
        pmm = pm1;
        pm1 = p;
    }
    p
}

/// Power-series coefficients of dᵐP_l/dxᵐ (Legendre polynomial differentiated m times).
pub fn legendre_deriv_coeffs(l: u32, m: u32) -> Vec<f64> {
    let mut c = vec![0.0; l as usize + 1];
    for k in 0..=l / 2 {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        c[(l - 2 * k) as usize] = sign * binom(l, k) * binom(2 * l - 2 * k, l) / 2f64.powi(l as i32);
    }
    for _ in 0..m {
        c = c.iter().enumerate().skip(1).map(|(j, v)| v * j as f64).collect();
        if c.is_empty() {
            c.push(0.0);
        }
    }
    c
}

/// Normalisation of the radial function: N_nl = √((2/n)³ (n−l−1)! / (2n (n+l)!)).
pub fn radial_norm(n: u32, l: u32) -> f64 {
    let nf = n as f64;
    ((2.0 / nf).powi(3) * factorial(n - l - 1) / (2.0 * nf * factorial(n + l))).sqrt()
}

/// R_nl(r) from the Laguerre recurrence (reference implementation).
pub fn radial_ref(n: u32, l: u32, r: f64) -> f64 {
    let x = 2.0 * r / n as f64;
    radial_norm(n, l) * (-r / n as f64).exp() * x.powi(l as i32) * laguerre_rec(n - l - 1, 2 * l + 1, x)
}

/// Normalisation of the θ part: √((2l+1)/(4π) (l−|m|)!/(l+|m|)!).
pub fn angular_norm(l: u32, m: i32) -> f64 {
    let am = m.unsigned_abs();
    ((2 * l + 1) as f64 / (4.0 * PI) * factorial(l - am) / factorial(l + am)).sqrt()
}

/// Condon–Shortley sign of Θ_lm: (−1)^m for m > 0, +1 for m ≤ 0.
pub fn cs_sign(m: i32) -> f64 {
    if m > 0 && m % 2 == 1 { -1.0 } else { 1.0 }
}

/// Y_lm(θ, φ) from the Legendre recurrence (reference implementation).
pub fn ylm_ref(l: u32, m: i32, theta: f64, phi: f64) -> C64 {
    let th = cs_sign(m) * angular_norm(l, m) * legendre_rec(l, m.unsigned_abs(), theta.cos());
    C64::from_polar(th, m as f64 * phi)
}

// ------------------------------------------------------------------ polynomial form

/// One ψ_nlm in the polynomial form used on the CPU and on the GPU:
/// ψ = rnorm·anorm · e^(−r/n) r^l Σ rad_j r^j · sin^|m|θ Σ ang_j cos^jθ · e^(imφ).
#[derive(Clone, Copy, Debug)]
pub struct OrbitalPoly {
    pub q: Nlm,
    /// N_nl (2/n)^l
    pub rnorm: f64,
    /// Condon–Shortley sign · N_lm
    pub anorm: f64,
    /// 1/n
    pub inv_n: f64,
    /// L_(n−l−1)^(2l+1)(2r/n) as a polynomial in r (degree ≤ 3)
    pub rad: [f64; 4],
    /// dᵐP_l/du^m (u = cos θ, m = |m|) (degree ≤ 3)
    pub ang: [f64; 4],
}

/// Shared geometry of a point (to evaluate several orbitals there).
#[derive(Clone, Copy, Debug)]
pub struct Geo {
    pub r: f64,
    /// cos θ
    pub u: f64,
    /// sin θ
    pub s: f64,
    /// e^(iφ)
    pub eip: C64,
}

impl Geo {
    pub fn new(p: [f64; 3]) -> Self {
        let [x, y, z] = p;
        let rho = (x * x + y * y).sqrt();
        let r = (rho * rho + z * z).sqrt();
        let (u, s) = if r > 1e-300 { (z / r, rho / r) } else { (1.0, 0.0) };
        let eip = if rho > 1e-300 { c64(x / rho, y / rho) } else { c64(1.0, 0.0) };
        Geo { r, u, s, eip }
    }
}

fn horner(c: &[f64; 4], x: f64) -> f64 {
    ((c[3] * x + c[2]) * x + c[1]) * x + c[0]
}

impl OrbitalPoly {
    pub fn new(q: Nlm) -> Self {
        assert!(q.is_valid(), "invalid quantum numbers {q:?}");
        let (n, l) = (q.n, q.l);
        let k = n - l - 1;
        let scale = 2.0 / n as f64;
        let mut rad = [0.0; 4];
        for (j, a) in laguerre_coeffs(k, 2 * l + 1).into_iter().enumerate() {
            rad[j] = a * scale.powi(j as i32);
        }
        let mut ang = [0.0; 4];
        for (j, b) in legendre_deriv_coeffs(l, q.m.unsigned_abs()).into_iter().enumerate() {
            ang[j] = b;
        }
        OrbitalPoly {
            q,
            rnorm: radial_norm(n, l) * scale.powi(l as i32),
            anorm: cs_sign(q.m) * angular_norm(l, q.m),
            inv_n: 1.0 / n as f64,
            rad,
            ang,
        }
    }

    /// R_nl(r)
    pub fn radial(&self, r: f64) -> f64 {
        self.rnorm * (-r * self.inv_n).exp() * r.powi(self.q.l as i32) * horner(&self.rad, r)
    }

    /// Θ_lm(cos θ), so that Y_lm = Θ_lm e^(imφ)
    pub fn theta(&self, u: f64) -> f64 {
        let s = (1.0 - u * u).max(0.0).sqrt();
        self.anorm * s.powi(self.q.m.abs()) * horner(&self.ang, u)
    }

    /// ψ_nlm at a point with precomputed geometry.
    pub fn eval_geo(&self, g: &Geo) -> C64 {
        let radial = (-g.r * self.inv_n).exp() * g.r.powi(self.q.l as i32) * horner(&self.rad, g.r);
        let ang = g.s.powi(self.q.m.abs()) * horner(&self.ang, g.u);
        let mut ph = c64(1.0, 0.0);
        for _ in 0..self.q.m.abs() {
            ph *= g.eip;
        }
        if self.q.m < 0 {
            ph = ph.conj();
        }
        ph * (self.rnorm * self.anorm * radial * ang)
    }

    /// ψ_nlm(x, y, z)
    pub fn eval(&self, p: [f64; 3]) -> C64 {
        self.eval_geo(&Geo::new(p))
    }
}

/// The 30 orbitals in [`BASIS`] order.
pub fn polys() -> &'static [OrbitalPoly; N_BASIS] {
    static P: OnceLock<[OrbitalPoly; N_BASIS]> = OnceLock::new();
    P.get_or_init(|| BASIS.map(OrbitalPoly::new))
}

/// R_nl(r) (polynomial form).
pub fn radial(n: u32, l: u32, r: f64) -> f64 {
    polys()[Nlm::new(n, l, 0).index()].radial(r)
}

/// ψ = Σ c_k ψ_k at a point.
pub fn psi(c: &Coeffs, p: [f64; 3]) -> C64 {
    let g = Geo::new(p);
    let ps = polys();
    c.iter()
        .zip(ps.iter())
        .filter(|(ck, _)| ck.norm_sqr() > 0.0)
        .map(|(ck, o)| ck * o.eval_geo(&g))
        .sum()
}

/// Indices of the nonzero coefficients (|c|² > 1e-14).
pub fn active(c: &Coeffs) -> Vec<usize> {
    (0..N_BASIS).filter(|&i| c[i].norm_sqr() > 1e-14).collect()
}

/// ψ = Σ c_k ψ_k over a precomputed list of active indices.
pub fn psi_active(c: &Coeffs, act: &[usize], p: [f64; 3]) -> C64 {
    let g = Geo::new(p);
    let ps = polys();
    act.iter().map(|&k| c[k] * ps[k].eval_geo(&g)).sum()
}

/// Zeros of R_nl (radial nodes), r > 0.
pub fn radial_nodes(n: u32, l: u32) -> Vec<f64> {
    let o = &polys()[Nlm::new(n, l, 0).index()];
    let f = |r: f64| horner(&o.rad, r);
    let mut out = vec![];
    let r_end = 6.0 * (n * n) as f64 + 10.0;
    let steps = 4000;
    let h = r_end / steps as f64;
    let mut prev = f(1e-9);
    for i in 1..=steps {
        // (an offset that is not a round number, so that no node falls on a grid point)
        let r = (i as f64 + 0.123_456) * h;
        let v = f(r);
        if prev * v < 0.0 {
            // bisection
            let (mut a, mut b) = (r - h, r);
            for _ in 0..60 {
                let c = 0.5 * (a + b);
                if f(a) * f(c) <= 0.0 { b = c } else { a = c }
            }
            out.push(0.5 * (a + b));
        }
        prev = v;
    }
    out
}

/// Closed forms (atomic units) for a single (n, l).
pub mod closed {
    /// ⟨r⟩ = (3n² − l(l+1))/2
    pub fn r1(n: u32, l: u32) -> f64 {
        (3.0 * (n * n) as f64 - (l * (l + 1)) as f64) / 2.0
    }
    /// ⟨r²⟩ = n²(5n² + 1 − 3l(l+1))/2
    pub fn r2(n: u32, l: u32) -> f64 {
        let nn = (n * n) as f64;
        nn * (5.0 * nn + 1.0 - 3.0 * (l * (l + 1)) as f64) / 2.0
    }
    /// ⟨1/r⟩ = 1/n²
    pub fn rinv(n: u32) -> f64 {
        1.0 / (n * n) as f64
    }
    /// ⟨1/r²⟩ = 1/((l + ½) n³)
    pub fn rinv2(n: u32, l: u32) -> f64 {
        1.0 / ((l as f64 + 0.5) * (n * n * n) as f64)
    }
    /// most probable radius of r²R_nl² for l = n − 1: r = n²
    pub fn r_mp_circular(n: u32) -> f64 {
        (n * n) as f64
    }
}

// ------------------------------------------------------------------ quadrature

/// Gauss–Legendre nodes and weights on [−1, 1].
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut xs = vec![0.0; n];
    let mut ws = vec![0.0; n];
    for i in 0..n {
        let mut x = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        for _ in 0..100 {
            // P_n(x) and P_n'(x)
            let (mut p0, mut p1) = (1.0, x);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            let dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        let (mut p0, mut p1) = (1.0, x);
        for k in 2..=n {
            let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
            p0 = p1;
            p1 = p2;
        }
        let dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
        xs[i] = x;
        ws[i] = 2.0 / ((1.0 - x * x) * dp * dp);
    }
    (xs, ws)
}

/// Nodes and weights of composite Gauss–Legendre quadrature on [a, b]
/// (`panels` equal panels with 16 nodes each).
pub fn gl_panels(a: f64, b: f64, panels: usize) -> Vec<(f64, f64)> {
    static GL16: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    let (xs, ws) = GL16.get_or_init(|| gauss_legendre(16));
    let w = (b - a) / panels as f64;
    let mut out = Vec::with_capacity(panels * xs.len());
    for p in 0..panels {
        let c = a + (p as f64 + 0.5) * w;
        for (x, wt) in xs.iter().zip(ws) {
            out.push((c + 0.5 * w * x, 0.5 * w * wt));
        }
    }
    out
}

/// ∫_a^b f by composite Gauss–Legendre quadrature.
pub fn integrate(a: f64, b: f64, panels: usize, f: impl Fn(f64) -> f64) -> f64 {
    gl_panels(a, b, panels).into_iter().map(|(x, w)| w * f(x)).sum()
}

/// Angular functions g(θ, φ) whose matrix elements ⟨Y_l'm'| g |Y_lm⟩ are tabulated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AngOp {
    One = 0,
    /// cos θ  (z/r)
    Z = 1,
    /// sin θ cos φ  (x/r)
    X = 2,
    /// sin θ sin φ  (y/r)
    Y = 3,
    /// sin²θ cos²φ  (x²/r²)
    XX = 4,
    /// sin²θ sin²φ
    YY = 5,
    /// cos²θ
    ZZ = 6,
}

const N_ANG: usize = 7;

fn ang_fn(op: AngOp, u: f64, phi: f64) -> f64 {
    let s2 = (1.0 - u * u).max(0.0);
    let s = s2.sqrt();
    match op {
        AngOp::One => 1.0,
        AngOp::Z => u,
        AngOp::X => s * phi.cos(),
        AngOp::Y => s * phi.sin(),
        AngOp::XX => s2 * phi.cos().powi(2),
        AngOp::YY => s2 * phi.sin().powi(2),
        AngOp::ZZ => u * u,
    }
}

/// Inverse-CDF sampler on a grid.
#[derive(Clone, Debug)]
pub struct Cdf {
    xs: Vec<f64>,
    cdf: Vec<f64>,
}

impl Cdf {
    fn new(xs: Vec<f64>, density: &[f64]) -> Self {
        let mut cdf = vec![0.0; xs.len()];
        for i in 1..xs.len() {
            cdf[i] = cdf[i - 1] + 0.5 * (density[i] + density[i - 1]).max(0.0) * (xs[i] - xs[i - 1]);
        }
        let total = cdf.last().copied().unwrap_or(1.0).max(1e-300);
        cdf.iter_mut().for_each(|c| *c /= total);
        Cdf { xs, cdf }
    }

    /// x with CDF(x) = v (v in [0, 1)).
    pub fn sample(&self, v: f64) -> f64 {
        let i = self.cdf.partition_point(|&c| c < v).clamp(1, self.cdf.len() - 1);
        let (c0, c1) = (self.cdf[i - 1], self.cdf[i]);
        let f = if c1 > c0 { (v - c0) / (c1 - c0) } else { 0.5 };
        self.xs[i - 1] + f * (self.xs[i] - self.xs[i - 1])
    }

    /// x below which a fraction `q` lies.
    pub fn quantile(&self, q: f64) -> f64 {
        self.sample(q)
    }
}

/// Precomputed integrals and sampling tables.
pub struct Tables {
    /// ∫ R_a R_b r^(2+p) dr for p = −2 … 3 (index p + 2), a, b = (n, l) indices
    pub rad: Vec<[[f64; N_NL]; N_NL]>,
    /// ⟨Y_a| g |Y_b⟩ for g = [`AngOp`] (index as usize), a, b = (l, m) indices
    pub ang: Vec<[[C64; N_LM]; N_LM]>,
    /// radial CDF of r²R_nl² per (n, l)
    pub radial_cdf: Vec<Cdf>,
    /// CDF in u = cos θ of 2π Θ_lm(u)² per (l, |m|) (index l(l+1)/2 + |m|)
    pub theta_cdf: Vec<Cdf>,
    /// radius containing 99.9 % of the probability, per (n, l)
    pub r999: [f64; N_NL],
}

/// Quantum numbers (n, l) of radial index a.
pub fn nl_of(a: usize) -> (u32, u32) {
    for n in 1..=N_MAX {
        for l in 0..n {
            if nl_index(n, l) == a {
                return (n, l);
            }
        }
    }
    (1, 0)
}

/// Quantum numbers (l, m) of angular index a.
pub fn lm_of(a: usize) -> (u32, i32) {
    let l = (a as f64).sqrt().floor() as u32;
    (l, a as i32 - (l * l + l) as i32)
}

/// The tables (computed once).
pub fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(build_tables)
}

fn build_tables() -> Tables {
    // radial integrals: Simpson in the scaled variable βr, β = 1/n_a + 1/n_b
    let mut rad = vec![[[0.0; N_NL]; N_NL]; 6];
    for a in 0..N_NL {
        for b in a..N_NL {
            let (na, la) = nl_of(a);
            let (nb, lb) = nl_of(b);
            let beta = 1.0 / na as f64 + 1.0 / nb as f64;
            let r_end = 75.0 / beta;
            for (pi, table) in rad.iter_mut().enumerate() {
                let p = pi as i32 - 2;
                let v = integrate(0.0, r_end, 75, |r| radial(na, la, r) * radial(nb, lb, r) * r.powi(2 + p));
                table[a][b] = v;
                table[b][a] = v;
            }
        }
    }
    // angular integrals: Gauss–Legendre in u, uniform in φ (exact for these trigonometric polynomials)
    let (us, ws) = gauss_legendre(24);
    let nphi = 48;
    let ops = [AngOp::One, AngOp::Z, AngOp::X, AngOp::Y, AngOp::XX, AngOp::YY, AngOp::ZZ];
    let mut ang = vec![[[c64(0.0, 0.0); N_LM]; N_LM]; N_ANG];
    let thetas: Vec<OrbitalPoly> = (0..N_LM)
        .map(|i| {
            let (l, m) = lm_of(i);
            OrbitalPoly::new(Nlm::new(l + 1, l, m))
        })
        .collect();
    for op in ops {
        for a in 0..N_LM {
            for b in 0..N_LM {
                let (ma, mb) = (thetas[a].q.m, thetas[b].q.m);
                let mut acc = c64(0.0, 0.0);
                for (u, w) in us.iter().zip(&ws) {
                    let th = thetas[a].theta(*u) * thetas[b].theta(*u);
                    if th == 0.0 {
                        continue;
                    }
                    for k in 0..nphi {
                        let phi = TAU * k as f64 / nphi as f64;
                        let e = C64::from_polar(1.0, (mb - ma) as f64 * phi);
                        acc += e * (w * th * ang_fn(op, *u, phi) * TAU / nphi as f64);
                    }
                }
                ang[op as usize][a][b] = acc;
            }
        }
    }
    // sampling tables
    let mut radial_cdf = vec![];
    let mut r999 = [0.0; N_NL];
    for (a, slot) in r999.iter_mut().enumerate() {
        let (n, l) = nl_of(a);
        let r_end = 20.0 + 7.0 * (n * n) as f64;
        let m = 6000;
        let xs: Vec<f64> = (0..=m).map(|i| r_end * i as f64 / m as f64).collect();
        let dens: Vec<f64> = xs.iter().map(|&r| (r * radial(n, l, r)).powi(2)).collect();
        let cdf = Cdf::new(xs, &dens);
        *slot = cdf.quantile(0.999);
        radial_cdf.push(cdf);
    }
    let mut theta_cdf = vec![];
    for l in 0..N_MAX {
        for am in 0..=l {
            let o = OrbitalPoly::new(Nlm::new(l + 1, l, am as i32));
            let m = 4000;
            let xs: Vec<f64> = (0..=m).map(|i| -1.0 + 2.0 * i as f64 / m as f64).collect();
            let dens: Vec<f64> = xs.iter().map(|&u| o.theta(u).powi(2)).collect();
            theta_cdf.push(Cdf::new(xs, &dens));
        }
    }
    Tables {
        rad,
        ang,
        radial_cdf,
        theta_cdf,
        r999,
    }
}

/// ⟨ψ| r^p g(θ, φ) |ψ⟩ for p in −2 … 3 (c need not be normalised; divides by Σ|c|²).
pub fn expect(c: &Coeffs, p: i32, op: AngOp) -> C64 {
    let t = tables();
    let rad = &t.rad[(p + 2) as usize];
    let ang = &t.ang[op as usize];
    let act = active(c);
    let mut acc = c64(0.0, 0.0);
    for &a in &act {
        let qa = BASIS[a];
        for &b in &act {
            let qb = BASIS[b];
            let g = ang[qa.lm()][qb.lm()];
            if g.norm_sqr() < 1e-28 {
                continue;
            }
            acc += c[a].conj() * c[b] * g * rad[qa.nl()][qb.nl()];
        }
    }
    let norm: f64 = c.iter().map(|z| z.norm_sqr()).sum();
    acc / norm.max(1e-300)
}

/// ⟨r^p⟩
pub fn expect_r(c: &Coeffs, p: i32) -> f64 {
    expect(c, p, AngOp::One).re
}

/// (⟨x⟩, ⟨y⟩, ⟨z⟩)
pub fn expect_xyz(c: &Coeffs) -> [f64; 3] {
    [expect(c, 1, AngOp::X).re, expect(c, 1, AngOp::Y).re, expect(c, 1, AngOp::Z).re]
}

/// Radial probability density P(r) = r² Σ_lm |Σ_n c_nlm R_nl(r)|² of a state (normalised c).
pub fn radial_density(c: &Coeffs, r: f64) -> f64 {
    let mut by_lm = [c64(0.0, 0.0); N_LM];
    for (i, q) in BASIS.iter().enumerate() {
        if c[i].norm_sqr() > 0.0 {
            by_lm[q.lm()] += c[i] * radial(q.n, q.l, r);
        }
    }
    r * r * by_lm.iter().map(|z| z.norm_sqr()).sum::<f64>()
}

// ------------------------------------------------------------------ angular momentum (ħ = 1)

/// √(l(l+1) − m(m ± 1)): l_± ψ_lm = ħ √(…) ψ_l,m±1 (PS11 43 b).
pub fn ladder_factor(l: u32, m: i32, raise: bool) -> f64 {
    let (l, m) = (l as f64, m as f64);
    let v = l * (l + 1.0) - m * (m + if raise { 1.0 } else { -1.0 });
    v.max(0.0).sqrt()
}

/// l₊ (raise = true) or l₋ applied to a coefficient vector.
pub fn apply_ladder(c: &Coeffs, raise: bool) -> Coeffs {
    let mut out = [c64(0.0, 0.0); N_BASIS];
    for (i, q) in BASIS.iter().enumerate() {
        let m2 = q.m + if raise { 1 } else { -1 };
        if m2.unsigned_abs() > q.l {
            continue;
        }
        out[Nlm::new(q.n, q.l, m2).index()] += c[i] * ladder_factor(q.l, q.m, raise);
    }
    out
}

/// l_z applied to c.
pub fn apply_lz(c: &Coeffs) -> Coeffs {
    let mut out = *c;
    for (i, q) in BASIS.iter().enumerate() {
        out[i] *= q.m as f64;
    }
    out
}

/// l² applied to c.
pub fn apply_l2(c: &Coeffs) -> Coeffs {
    let mut out = *c;
    for (i, q) in BASIS.iter().enumerate() {
        out[i] *= (q.l * (q.l + 1)) as f64;
    }
    out
}

/// l_y = (l₊ − l₋)/(2i) applied to c.
pub fn apply_ly(c: &Coeffs) -> Coeffs {
    let up = apply_ladder(c, true);
    let down = apply_ladder(c, false);
    let mut out = [c64(0.0, 0.0); N_BASIS];
    for i in 0..N_BASIS {
        out[i] = (up[i] - down[i]) / c64(0.0, 2.0);
    }
    out
}

/// l_x = (l₊ + l₋)/2 applied to c.
pub fn apply_lx(c: &Coeffs) -> Coeffs {
    let up = apply_ladder(c, true);
    let down = apply_ladder(c, false);
    let mut out = [c64(0.0, 0.0); N_BASIS];
    for i in 0..N_BASIS {
        out[i] = (up[i] + down[i]) * 0.5;
    }
    out
}

/// ⟨a|b⟩
pub fn inner(a: &Coeffs, b: &Coeffs) -> C64 {
    a.iter().zip(b).map(|(x, y)| x.conj() * y).sum()
}

/// Eigenvalues and eigenvectors of a real symmetric matrix (cyclic Jacobi).
/// Returns (values, vectors) with vectors[k] the k-th eigenvector.
pub fn jacobi_eigen(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v = vec![vec![0.0; n]; n];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _sweep in 0..100 {
        let off: f64 = (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| a[i][j] * a[i][j]).sum();
        if off < 1e-28 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in a.iter_mut() {
                    let (akp, akq) = (row[p], row[q]);
                    row[p] = c * akp - s * akq;
                    row[q] = s * akp + c * akq;
                }
                let (rp, rq) = (a[p].clone(), a[q].clone());
                for k in 0..n {
                    a[p][k] = c * rp[k] - s * rq[k];
                    a[q][k] = s * rp[k] + c * rq[k];
                }
                for row in v.iter_mut() {
                    let (vkp, vkq) = (row[p], row[q]);
                    row[p] = c * vkp - s * vkq;
                    row[q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let values: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    let vectors: Vec<Vec<f64>> = (0..n).map(|k| (0..n).map(|i| v[i][k]).collect()).collect();
    (values, vectors)
}

/// Eigenvalues m_y (= −l … l, in units of ħ) and eigenvectors of l_y in the
/// (2l+1)-dimensional space of one l, in the basis m = −l … l (Condon–Shortley).
///
/// Trick: with U = diag(i^m), U† l_y U = −l_x is real symmetric; its
/// eigenvectors v give the l_y eigenvectors U v.
pub fn ly_eigen(l: u32) -> Vec<(f64, Vec<C64>)> {
    let d = (2 * l + 1) as usize;
    let mut a = vec![vec![0.0; d]; d];
    for i in 0..d - 1 {
        let m = i as i32 - l as i32;
        let v = -0.5 * ladder_factor(l, m, true);
        a[i + 1][i] = v;
        a[i][i + 1] = v;
    }
    let (vals, vecs) = jacobi_eigen(a);
    let mut out: Vec<(f64, Vec<C64>)> = vals
        .into_iter()
        .zip(vecs)
        .map(|(val, vec)| {
            let w: Vec<C64> = vec
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    let m = i as i32 - l as i32;
                    C64::from_polar(*x, m as f64 * std::f64::consts::FRAC_PI_2)
                })
                .collect();
            (val.round(), w)
        })
        .collect();
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out
}

/// The l_y eigenvectors for l = 0 … 3 (cached).
/// (eigenvalue m_y, eigenvector over m = −l … l) for one l.
pub type LyEigen = Vec<(f64, Vec<C64>)>;

pub fn ly_eigen_all() -> &'static [LyEigen; 4] {
    static E: OnceLock<[LyEigen; 4]> = OnceLock::new();
    E.get_or_init(|| [ly_eigen(0), ly_eigen(1), ly_eigen(2), ly_eigen(3)])
}

// ------------------------------------------------------------------ Born sampling

/// One position drawn from |ψ|² (exact): a component k is drawn with weight
/// |c_k|², a point from |ψ_k|² (tabulated r and cos θ, uniform φ), and the
/// point is accepted with probability |ψ|² / (K Σ_k |c_k|²|ψ_k|²) ≤ 1
/// (Cauchy–Schwarz; K = number of components).
pub fn born_sample(c: &Coeffs, act: &[usize], rng: &mut QbRng) -> Option<[f64; 3]> {
    if act.is_empty() {
        return None;
    }
    let t = tables();
    let ps = polys();
    let weights: Vec<f64> = act.iter().map(|&k| c[k].norm_sqr()).collect();
    let kk = act.len() as f64;
    for _ in 0..100_000 {
        let j = rng.weighted(&weights)?;
        let q = BASIS[act[j]];
        let r = t.radial_cdf[q.nl()].sample(rng.uniform());
        let am = q.m.unsigned_abs();
        let u = t.theta_cdf[(q.l * (q.l + 1) / 2 + am) as usize].sample(rng.uniform());
        let phi = TAU * rng.uniform();
        let s = (1.0 - u * u).max(0.0).sqrt();
        let p = [r * s * phi.cos(), r * s * phi.sin(), r * u];
        if act.len() == 1 {
            return Some(p);
        }
        let g = Geo::new(p);
        let mut psi = c64(0.0, 0.0);
        let mut qsum = 0.0;
        for &k in act {
            let v = ps[k].eval_geo(&g);
            psi += c[k] * v;
            qsum += c[k].norm_sqr() * v.norm_sqr();
        }
        if rng.uniform() * kk * qsum <= psi.norm_sqr() {
            return Some(p);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basis_indexing() {
        for (i, q) in BASIS.iter().enumerate() {
            assert_eq!(q.index(), i);
            assert!(q.is_valid());
        }
        assert_eq!(BASIS.len(), 30);
        assert_eq!(Nlm::new(4, 3, 3).index(), 29);
        assert_eq!(Nlm::new(2, 1, -1).name(), "ψ₂₁₋₁");
    }

    #[test]
    fn polynomial_form_matches_recurrences() {
        for q in BASIS {
            let o = OrbitalPoly::new(q);
            for &r in &[0.05, 0.7, 1.9, 4.3, 9.0, 17.5, 30.0] {
                let a = o.radial(r);
                let b = radial_ref(q.n, q.l, r);
                assert!((a - b).abs() < 1e-12 * (1.0 + b.abs()), "R {q:?} r={r}: {a} vs {b}");
            }
            for &th in &[0.1f64, 0.8, 1.4, 2.2, 3.0] {
                for &ph in &[0.0, 0.9, 2.5, 4.4] {
                    let r = 2.3;
                    let p = [r * th.sin() * f64::cos(ph), r * th.sin() * f64::sin(ph), r * th.cos()];
                    let a = o.eval(p);
                    let b = radial_ref(q.n, q.l, r) * ylm_ref(q.l, q.m, th, ph);
                    assert!((a - b).norm() < 1e-12, "ψ {q:?}: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn problem_set_convention_eq_44_4() {
        // R10 = 2 e^(−r), R20 = (1/√2)(1 − r/2) e^(−r/2), R21 = r e^(−r/2)/(2√6)
        let r = 1.7;
        assert!((radial(1, 0, r) - 2.0 * (-r).exp()).abs() < 1e-14);
        assert!((radial(2, 0, r) - 2.0 * 0.5f64.powf(1.5) * (1.0 - r / 2.0) * (-r / 2.0).exp()).abs() < 1e-14);
        assert!((radial(2, 1, r) - 2.0 / 3f64.sqrt() * 0.5f64.powf(1.5) * (r / 2.0) * (-r / 2.0).exp()).abs() < 1e-14);
        // Y00 = √(1/4π), Y10 = √(3/4π) cos θ, Y1±1 = ∓√(3/8π) sin θ e^(±iφ)
        let (th, ph) = (0.7, 1.2);
        let y = |l, m| OrbitalPoly::new(Nlm::new(2, l, m)).theta(f64::cos(th)) * C64::from_polar(1.0, m as f64 * ph);
        assert!((y(0, 0) - c64((1.0 / (4.0 * PI)).sqrt(), 0.0)).norm() < 1e-14);
        assert!((y(1, 0) - c64((3.0 / (4.0 * PI)).sqrt() * f64::cos(th), 0.0)).norm() < 1e-14);
        let a = (3.0 / (8.0 * PI)).sqrt() * f64::sin(th);
        assert!((y(1, 1) - C64::from_polar(-a, ph)).norm() < 1e-14);
        assert!((y(1, -1) - C64::from_polar(a, -ph)).norm() < 1e-14);
    }

    /// ∫ψ_a* ψ_b d³r = δ_ab for all 30 states, by direct quadrature of the full
    /// ψ(x, y, z) (Simpson in r, Gauss–Legendre in cos θ, uniform in φ).
    #[test]
    fn orthonormality_by_numerical_integration() {
        let (us, ws) = gauss_legendre(12);
        let nphi = 16;
        let ps = polys();
        let mut gram = vec![[c64(0.0, 0.0); N_BASIS]; N_BASIS];
        let mut vals = [c64(0.0, 0.0); N_BASIS];
        for (r, wr) in gl_panels(0.0, 130.0, 65) {
            let wr = wr * r * r;
            for (u, wu) in us.iter().zip(&ws) {
                let s = (1.0 - u * u).sqrt();
                for k in 0..nphi {
                    let phi = TAU * k as f64 / nphi as f64;
                    let w = wr * wu * TAU / nphi as f64;
                    let g = Geo::new([r * s * phi.cos(), r * s * phi.sin(), r * u]);
                    for (i, o) in ps.iter().enumerate() {
                        vals[i] = o.eval_geo(&g);
                    }
                    for a in 0..N_BASIS {
                        for b in a..N_BASIS {
                            gram[a][b] += vals[a].conj() * vals[b] * w;
                        }
                    }
                }
            }
        }
        for a in 0..N_BASIS {
            for b in a..N_BASIS {
                let want = if a == b { 1.0 } else { 0.0 };
                assert!(
                    (gram[a][b] - c64(want, 0.0)).norm() < 1e-8,
                    "<{:?}|{:?}> = {}",
                    BASIS[a],
                    BASIS[b],
                    gram[a][b]
                );
            }
        }
    }

    #[test]
    fn radial_expectation_values_match_closed_forms() {
        let t = tables();
        for n in 1..=N_MAX {
            for l in 0..n {
                let a = nl_index(n, l);
                let norm = t.rad[2][a][a];
                assert!((norm - 1.0).abs() < 1e-10, "norm {n}{l}");
                assert!((t.rad[3][a][a] - closed::r1(n, l)).abs() < 1e-9, "<r> {n}{l}");
                assert!((t.rad[4][a][a] - closed::r2(n, l)).abs() < 1e-8, "<r²> {n}{l}");
                assert!((t.rad[1][a][a] - closed::rinv(n)).abs() < 1e-10, "<1/r> {n}{l}");
                assert!((t.rad[0][a][a] - closed::rinv2(n, l)).abs() < 1e-9, "<1/r²> {n}{l}: {}", t.rad[0][a][a]);
                // the same through a state vector
                let mut c = [c64(0.0, 0.0); N_BASIS];
                c[Nlm::new(n, l, 0).index()] = c64(0.0, 1.0);
                assert!((expect_r(&c, 1) - closed::r1(n, l)).abs() < 1e-9);
                assert!((expect_r(&c, -1) - closed::rinv(n)).abs() < 1e-10);
                assert!((expect_r(&c, -2) - closed::rinv2(n, l)).abs() < 1e-9);
            }
        }
        // radial functions with the same l and different n are orthogonal
        for l in 0..3 {
            for n1 in (l + 1)..=N_MAX {
                for n2 in (n1 + 1)..=N_MAX {
                    assert!(t.rad[2][nl_index(n1, l)][nl_index(n2, l)].abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn ground_state_ps11_44_cd() {
        let mut c = [c64(0.0, 0.0); N_BASIS];
        c[0] = c64(1.0, 0.0);
        assert!((expect_r(&c, 1) - 1.5).abs() < 1e-10); // ⟨r⟩ = 3a₀/2
        assert!((expect_r(&c, 2) - 3.0).abs() < 1e-9); // ⟨r²⟩ = 3a₀²
        let [x, y, z] = expect_xyz(&c);
        assert!(x.abs() < 1e-12 && y.abs() < 1e-12 && z.abs() < 1e-12); // ⟨x⟩ = 0
        let xx = expect(&c, 2, AngOp::XX).re;
        assert!((xx - 1.0).abs() < 1e-9, "<x²> = {xx}"); // ⟨x²⟩ = ⟨r²⟩/3 = a₀²
        // ⟨Ê⟩ = −½⟨1/r⟩ = E_n for (1,0,0) and (2,1,0) (PS11 44 b)
        let mut c2 = [c64(0.0, 0.0); N_BASIS];
        c2[Nlm::new(2, 1, 0).index()] = c64(1.0, 0.0);
        assert!((-0.5 * expect_r(&c, -1) - energy(1)).abs() < 1e-10);
        assert!((-0.5 * expect_r(&c2, -1) - energy(2)).abs() < 1e-10);
        // Lyman α: 121.6 nm with E_n = −13.6 eV/n²
        let lambda = HC_EV_NM / (energy_ev(2) - energy_ev(1));
        assert!((lambda - 121.55).abs() < 0.01, "{lambda}");
    }

    #[test]
    fn dipole_matrix_element_1s_2p() {
        // ⟨100|z|210⟩ = 128√2/243 a₀ = 0.7449
        let mut c = [c64(0.0, 0.0); N_BASIS];
        c[0] = c64(1.0, 0.0);
        c[Nlm::new(2, 1, 0).index()] = c64(1.0, 0.0);
        let z = expect_xyz(&c)[2];
        // ⟨z⟩ = 2 Re(c₁* c₂ ⟨1s|z|2p₀⟩)/(|c₁|² + |c₂|²) = ⟨1s|z|2p₀⟩
        assert!((z - 128.0 * 2f64.sqrt() / 243.0).abs() < 1e-10, "{z}");
    }

    #[test]
    fn ladder_operators_ps11_43b() {
        for l in 0..4u32 {
            for m in -(l as i32)..=(l as i32) {
                let mut c = [c64(0.0, 0.0); N_BASIS];
                c[Nlm::new(4, l, m).index()] = c64(1.0, 0.0);
                let up = apply_ladder(&c, true);
                let norm_up: f64 = up.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
                assert!((norm_up - ladder_factor(l, m, true)).abs() < 1e-12);
                if m == l as i32 {
                    assert!(norm_up < 1e-12, "l₊ ψ_ll = 0");
                }
                let down = apply_ladder(&c, false);
                if m == -(l as i32) {
                    assert!(down.iter().all(|z| z.norm() < 1e-12), "l₋ ψ_l,−l = 0");
                }
                // l₋l₊ = l² − l_z² − ħl_z
                let lhs = apply_ladder(&up, false);
                let want = (l * (l + 1)) as f64 - (m * m) as f64 - m as f64;
                assert!((lhs[Nlm::new(4, l, m).index()].re - want).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn ly_eigenvectors() {
        for l in 0..4u32 {
            let e = ly_eigen(l);
            assert_eq!(e.len(), (2 * l + 1) as usize);
            for (k, (val, vec)) in e.iter().enumerate() {
                assert!((val - (k as f64 - l as f64)).abs() < 1e-9);
                let mut c = [c64(0.0, 0.0); N_BASIS];
                for (i, a) in vec.iter().enumerate() {
                    c[Nlm::new(4, l, i as i32 - l as i32).index()] = *a;
                }
                let ly = apply_ly(&c);
                for i in 0..N_BASIS {
                    assert!((ly[i] - c[i] * val).norm() < 1e-9, "l={l} m_y={val}");
                }
            }
        }
    }

    /// PS11 45: Ψ = (ψ₂₁₁ + ψ₂₁₋₁)/√2 is an l_y eigenstate with eigenvalue 0, not an l_z eigenstate.
    #[test]
    fn ps11_45_ly_eigenstate() {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let mut c = [c64(0.0, 0.0); N_BASIS];
        c[Nlm::new(2, 1, 1).index()] = c64(s, 0.0);
        c[Nlm::new(2, 1, -1).index()] = c64(s, 0.0);
        // in the m basis
        let ly = apply_ly(&c);
        assert!(ly.iter().all(|z| z.norm() < 1e-14), "l_y Ψ = 0");
        let lz = apply_lz(&c);
        let overlap = inner(&c, &lz);
        let norm_lz: f64 = lz.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        assert!(overlap.norm() < 1e-14 && (norm_lz - 1.0).abs() < 1e-14, "not an l_z eigenstate");
        // Ψ = −i √(3/4π) R₂₁ sin θ sin φ (lobes along y)
        let p = [0.4, 1.3, -0.7];
        let g = Geo::new(p);
        let want = c64(0.0, -(3.0 / (4.0 * PI)).sqrt() * radial(2, 1, g.r) * g.s * g.eip.im);
        assert!((psi(&c, p) - want).norm() < 1e-14);
        // the differential operator l_y = −i(cos φ ∂_θ − cot θ sin φ ∂_φ) applied numerically
        let f = |th: f64, ph: f64| {
            let r = 2.0;
            psi(&c, [r * th.sin() * ph.cos(), r * th.sin() * ph.sin(), r * th.cos()])
        };
        let h = 1e-5;
        for &(th, ph) in &[(0.7, 0.3), (1.2, 2.0), (2.4, 4.0)] {
            let dth = (f(th + h, ph) - f(th - h, ph)) / (2.0 * h);
            let dph = (f(th, ph + h) - f(th, ph - h)) / (2.0 * h);
            let ly = c64(0.0, -1.0) * (dth * f64::cos(ph) - dph * (f64::sin(ph) / f64::tan(th)));
            assert!(ly.norm() < 1e-8 * f(th, ph).norm().max(1e-3), "l_y Ψ = {ly}");
            // and l_z Ψ = −i ∂_φ Ψ is not proportional to Ψ (it is ∝ sin θ cos φ)
            let lz = c64(0.0, -1.0) * dph;
            assert!((lz - f(th, ph) * (lz / f(th, ph))).norm() < 1e-12);
        }
        // ψ₂₁₋₁ − ψ₂₁₁ ∝ sin θ cos φ: the lobe along x (an l_x eigenstate, m_x = 0)
        let mut cx = [c64(0.0, 0.0); N_BASIS];
        cx[Nlm::new(2, 1, 1).index()] = c64(-s, 0.0);
        cx[Nlm::new(2, 1, -1).index()] = c64(s, 0.0);
        assert!(apply_lx(&cx).iter().all(|z| z.norm() < 1e-14));
    }

    #[test]
    fn born_sampling_reproduces_moments() {
        let mut rng = QbRng::seed(3);
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let mut c = [c64(0.0, 0.0); N_BASIS];
        c[Nlm::new(1, 0, 0).index()] = c64(s, 0.0);
        c[Nlm::new(2, 1, 0).index()] = c64(0.0, s);
        // a real combination so that ⟨z⟩ ≠ 0
        c[Nlm::new(2, 1, 0).index()] = c64(s, 0.0);
        let act = active(&c);
        let n = 60_000;
        let (mut sr, mut sz) = (0.0, 0.0);
        for _ in 0..n {
            let p = born_sample(&c, &act, &mut rng).unwrap();
            sr += (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            sz += p[2];
        }
        let (mr, mz) = (sr / n as f64, sz / n as f64);
        let want_r = expect_r(&c, 1);
        let want_z = expect_xyz(&c)[2];
        assert!((mr - want_r).abs() < 0.05, "{mr} vs {want_r}");
        assert!((mz - want_z).abs() < 0.05, "{mz} vs {want_z}");
    }

    #[test]
    fn radial_nodes_count() {
        for n in 1..=N_MAX {
            for l in 0..n {
                assert_eq!(radial_nodes(n, l).len() as u32, n - l - 1);
            }
        }
        // 2s node at r = 2 a₀
        assert!((radial_nodes(2, 0)[0] - 2.0).abs() < 1e-9);
    }
}
