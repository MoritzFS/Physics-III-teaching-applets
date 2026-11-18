//! Numerics of the box (ħ = m = 1, lengths in units of a, so a = 1).
//!
//! The 3D box 0 ≤ x, y ≤ a, 0 ≤ z ≤ b of PS09 38 is shown through a 2D
//! cross-section: the x–y plane (a × a) or the x–z plane (a × b). In either
//! plane a state is an expansion in the separable 2D modes
//!
//!   φ_ij(u, v) = X_i(u) Y_j(v),   X_i(u) = √(2/L₁) sin(iπu/L₁),   Y_j(v) = √(2/L₂) sin(jπv/L₂),
//!
//! with i, j = 1 … [`M`]. The quantum number of the third direction is fixed
//! (it only adds a constant to every energy, i.e. a global phase).
//!
//! Fields are sampled on a midpoint grid u_k = (k + ½)·L/[`N`]. On this grid
//! the sine modes 1 … N − 1 are *exactly* orthonormal (discrete sine
//! transform), so projecting a sampled field onto the modes and evaluating
//! the expansion are exact inverses of each other for band-limited fields.

use std::f64::consts::PI;

use qb_core::{C64, QbRng, c64, par};

/// The box edge a (the length unit).
pub const A: f64 = 1.0;
/// Energy unit of the ladder: ε = π²ħ²/(2ma²) (= π²/2 here).
pub const EPS: f64 = PI * PI / 2.0;
/// Modes per axis: states are expanded in M × M modes.
pub const M: usize = 40;
/// Grid points per axis (midpoint grid; needs N > M for exact orthonormality).
pub const N: usize = 192;
/// Thumbnails and the composer cover n = 1 … THUMBS in each direction.
pub const THUMBS: usize = 5;

/// Two energies (in units of ε) closer than this belong to one level.
pub const DEGEN_TOL: f64 = 1e-7;

#[inline]
pub fn zero() -> C64 {
    c64(0.0, 0.0)
}

/// Which cross-section of the 3D box is shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Plane {
    /// x–y plane: a × a (square), the third quantum number is n₃
    XY,
    /// x–z plane: a × b, the third quantum number is n₂
    XZ,
}

/// The box and the fixed quantum number of the hidden direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    pub plane: Plane,
    /// b/a
    pub ratio: f64,
    /// the quantum number of the direction perpendicular to the plane
    /// (n₃ in the x–y plane, n₂ in the x–z plane)
    pub hidden: usize,
}

impl Default for Geometry {
    fn default() -> Self {
        Geometry {
            plane: Plane::XY,
            ratio: 1.0,
            hidden: 1,
        }
    }
}

impl Geometry {
    /// Edge along the first (horizontal) axis: always x, length a.
    pub fn lx(&self) -> f64 {
        A
    }

    /// Edge along the second (vertical) axis: y (a) or z (b).
    pub fn ly(&self) -> f64 {
        match self.plane {
            Plane::XY => A,
            Plane::XZ => self.ratio * A,
        }
    }

    /// Edge b of the box.
    pub fn b(&self) -> f64 {
        self.ratio * A
    }

    /// Name of the vertical axis: "y" or "z".
    pub fn v_axis(&self) -> &'static str {
        match self.plane {
            Plane::XY => "y",
            Plane::XZ => "z",
        }
    }

    /// Name of the second displayed quantum number: "n₂" or "n₃".
    pub fn v_index(&self) -> &'static str {
        match self.plane {
            Plane::XY => "n₂",
            Plane::XZ => "n₃",
        }
    }

    /// Name of the hidden quantum number: "n₃" or "n₂".
    pub fn hidden_index(&self) -> &'static str {
        match self.plane {
            Plane::XY => "n₃",
            Plane::XZ => "n₂",
        }
    }

    /// The hidden axis: "z" or "y".
    pub fn hidden_axis(&self) -> &'static str {
        match self.plane {
            Plane::XY => "z",
            Plane::XZ => "y",
        }
    }

    /// (n₁, n₂, n₃) of the displayed mode (i, j).
    pub fn triple(&self, i: usize, j: usize) -> (usize, usize, usize) {
        match self.plane {
            Plane::XY => (i, j, self.hidden),
            Plane::XZ => (i, self.hidden, j),
        }
    }

    /// Full 3D energy E(n₁, n₂, n₃)/ε (Eq. 38.2) of the displayed mode (i, j).
    pub fn e_eps(&self, i: usize, j: usize) -> f64 {
        let (n1, n2, n3) = self.triple(i, j);
        e3_eps(n1, n2, n3, self.ratio)
    }

    /// Energy of the displayed mode (i, j) in units ħ²/(ma²).
    pub fn energy(&self, i: usize, j: usize) -> f64 {
        EPS * self.e_eps(i, j)
    }

    /// The constant the hidden direction adds to every energy, in units of ε.
    pub fn hidden_e_eps(&self) -> f64 {
        let h = (self.hidden * self.hidden) as f64;
        match self.plane {
            Plane::XY => h / (self.ratio * self.ratio),
            Plane::XZ => h,
        }
    }

    /// Revival time of the 2D motion: every phase e^(−iE t) returns to the
    /// same global phase. Some(4a²/π) for a square cross-section; for a × b
    /// only when b²/a² = p/q is rational (small p, q are recognised), then p·4a²/π.
    pub fn revival_time(&self) -> Option<f64> {
        let t1 = 4.0 * A * A / PI;
        if self.plane == Plane::XY {
            return Some(t1);
        }
        let r2 = self.ratio * self.ratio;
        for q in 1..=12u32 {
            let p = r2 * q as f64;
            if (p - p.round()).abs() < 1e-9 && p.round() >= 1.0 {
                return Some(t1 * p.round());
            }
        }
        None
    }
}

/// Eq. 38.2 divided by ε = π²ħ²/(2ma²): n₁² + n₂² + n₃²·(a/b)².
pub fn e3_eps(n1: usize, n2: usize, n3: usize, ratio: f64) -> f64 {
    (n1 * n1 + n2 * n2) as f64 + (n3 * n3) as f64 / (ratio * ratio)
}

/// All (n₁, n₂, n₃) with E/ε = e (each n ≤ nmax), i.e. the degenerate 3D level.
pub fn degenerate_3d(e: f64, ratio: f64, nmax: usize) -> Vec<(usize, usize, usize)> {
    let mut out = vec![];
    for n1 in 1..=nmax {
        for n2 in 1..=nmax {
            let rest = e - (n1 * n1 + n2 * n2) as f64;
            if rest <= 0.0 {
                break;
            }
            // n₃² (a/b)² = rest
            let n3 = (rest * ratio * ratio).sqrt().round() as usize;
            if n3 >= 1 && n3 <= nmax && (e3_eps(n1, n2, n3, ratio) - e).abs() < DEGEN_TOL * e.max(1.0) {
                out.push((n1, n2, n3));
            }
        }
    }
    out
}

/// Index of the mode (i, j) (1-based quantum numbers) in a coefficient vector.
#[inline]
pub fn idx(i: usize, j: usize) -> usize {
    (i - 1) * M + (j - 1)
}

/// (i, j) of a coefficient index.
#[inline]
pub fn ij(k: usize) -> (usize, usize) {
    (k / M + 1, k % M + 1)
}

/// ⟨n|x|m⟩ of the 1D well of length l (x measured from the wall).
pub fn x_element(n: usize, m: usize, l: f64) -> f64 {
    if n == m {
        0.5 * l
    } else if (n + m) % 2 == 1 {
        let (nf, mf) = (n as f64, m as f64);
        -8.0 * l * nf * mf / (PI * PI * (nf * nf - mf * mf).powi(2))
    } else {
        0.0
    }
}

/// ⟨n|d/dx|m⟩ of the 1D well of length l (antisymmetric).
pub fn d_element(n: usize, m: usize, l: f64) -> f64 {
    if n != m && (n + m) % 2 == 1 {
        let (nf, mf) = (n as f64, m as f64);
        4.0 * nf * mf / (l * (nf * nf - mf * mf))
    } else {
        0.0
    }
}

/// One energy level of the plane: E/ε and its states (i, j).
#[derive(Clone, Debug, PartialEq)]
pub struct Level {
    pub e: f64,
    pub states: Vec<(usize, usize)>,
}

/// All levels with E/ε ≤ e_max among the modes i, j ≤ nmax, grouped by
/// energy (degenerate states share a level), sorted by energy.
pub fn levels(geom: &Geometry, nmax: usize, e_max: f64) -> Vec<Level> {
    let mut all: Vec<(f64, (usize, usize))> = vec![];
    for i in 1..=nmax {
        for j in 1..=nmax {
            let e = geom.e_eps(i, j);
            if e <= e_max {
                all.push((e, (i, j)));
            }
        }
    }
    group(all)
}

/// Groups (E/ε, state) pairs into levels.
pub fn group(mut all: Vec<(f64, (usize, usize))>) -> Vec<Level> {
    all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut out: Vec<Level> = vec![];
    for (e, s) in all {
        match out.last_mut() {
            Some(l) if (e - l.e).abs() < DEGEN_TOL * e.max(1.0) => l.states.push(s),
            _ => out.push(Level { e, states: vec![s] }),
        }
    }
    out
}

/// Degeneracy of the level E/ε = e within the plane (all modes up to n = 200,
/// not only those in the expansion).
pub fn plane_degeneracy(geom: &Geometry, e: f64) -> Vec<(usize, usize)> {
    let mut out = vec![];
    for i in 1..=200 {
        if geom.e_eps(i, 1) > e + 1e-9 {
            break;
        }
        for j in 1..=200 {
            let ej = geom.e_eps(i, j);
            if ej > e + 1e-6 * e.max(1.0) {
                break;
            }
            if (ej - e).abs() < DEGEN_TOL * e.max(1.0) {
                out.push((i, j));
            }
        }
    }
    out
}

/// Probability per energy level of a state (all M × M modes): (E/ε, P, indices).
pub fn level_probabilities(geom: &Geometry, c: &[C64]) -> Vec<(f64, f64, Vec<usize>)> {
    let norm: f64 = c.iter().map(|z| z.norm_sqr()).sum::<f64>().max(1e-300);
    let mut all: Vec<(f64, usize)> = c
        .iter()
        .enumerate()
        .filter(|(_, z)| z.norm_sqr() > 0.0)
        .map(|(k, _)| {
            let (i, j) = ij(k);
            (geom.e_eps(i, j), k)
        })
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out: Vec<(f64, f64, Vec<usize>)> = vec![];
    for (e, k) in all {
        let p = c[k].norm_sqr() / norm;
        match out.last_mut() {
            Some(l) if (e - l.0).abs() < DEGEN_TOL * e.max(1.0) => {
                l.1 += p;
                l.2.push(k);
            }
            _ => out.push((e, p, vec![k])),
        }
    }
    out
}

/// Projects onto the eigenspace E/ε = e: every coefficient of another energy
/// is set to zero, the rest keeps its amplitude and phase; then renormalised.
/// Returns the probability of the level (before renormalisation).
pub fn project_level(geom: &Geometry, c: &mut [C64], e: f64) -> f64 {
    let norm: f64 = c.iter().map(|z| z.norm_sqr()).sum();
    let mut kept = 0.0;
    for (k, z) in c.iter_mut().enumerate() {
        let (i, j) = ij(k);
        if (geom.e_eps(i, j) - e).abs() < DEGEN_TOL * e.max(1.0) {
            kept += z.norm_sqr();
        } else {
            *z = zero();
        }
    }
    if kept > 0.0 {
        let s = 1.0 / kept.sqrt();
        c.iter_mut().for_each(|z| *z *= s);
    }
    if norm > 0.0 { kept / norm } else { 0.0 }
}

/// Normalises a coefficient vector (Σ|c|² = 1); returns the norm² before.
pub fn normalize(c: &mut [C64]) -> f64 {
    let n: f64 = c.iter().map(|z| z.norm_sqr()).sum();
    if n > 0.0 {
        let s = 1.0 / n.sqrt();
        c.iter_mut().for_each(|z| *z *= s);
    }
    n
}

/// Expectation values from the coefficients (exact, no grid).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Moments2 {
    pub norm: f64,
    /// ⟨x⟩ and ⟨v⟩ (v = y or z), measured from the corner
    pub x: f64,
    pub y: f64,
    /// ⟨E⟩/ε and ΔE/ε (full 3D energy)
    pub e: f64,
    pub de: f64,
}

/// 1D sine tables, exact orthonormal on the midpoint grid.
#[derive(Clone, Debug)]
pub struct Axis {
    pub l: f64,
    pub n: usize,
    pub dx: f64,
    pub xs: Vec<f64>,
    /// s[(m − 1)·n + k] = √(2/l) sin(mπx_k/l), m = 1 … M
    pub s: Vec<f64>,
    /// ⟨m|x|m'⟩ (M × M)
    pub xm: Vec<f64>,
}

impl Axis {
    pub fn new(l: f64, n: usize) -> Self {
        let dx = l / n as f64;
        let xs: Vec<f64> = (0..n).map(|k| (k as f64 + 0.5) * dx).collect();
        let norm = (2.0 / l).sqrt();
        let mut s = vec![0.0; M * n];
        for m in 1..=M {
            for (k, &x) in xs.iter().enumerate() {
                s[(m - 1) * n + k] = norm * (m as f64 * PI * x / l).sin();
            }
        }
        let mut xm = vec![0.0; M * M];
        for a in 1..=M {
            for b in 1..=M {
                xm[(a - 1) * M + (b - 1)] = x_element(a, b, l);
            }
        }
        Axis { l, n, dx, xs, s, xm }
    }

    /// Row of mode m (1-based).
    #[inline]
    pub fn row(&self, m: usize) -> &[f64] {
        &self.s[(m - 1) * self.n..m * self.n]
    }

    /// √(2/l) sin(mπx/l) and its derivative at any x, for m = 1 … M.
    pub fn at(&self, x: f64) -> ([f64; M], [f64; M]) {
        let norm = (2.0 / self.l).sqrt();
        let mut f = [0.0; M];
        let mut d = [0.0; M];
        let th = PI * x / self.l;
        // sin(mθ), cos(mθ) by the angle-addition recurrence
        let (s1, c1) = th.sin_cos();
        let (mut s, mut c) = (s1, c1);
        for m in 0..M {
            let k = (m + 1) as f64 * PI / self.l;
            f[m] = norm * s;
            d[m] = norm * k * c;
            let (sn, cn) = (s * c1 + c * s1, c * c1 - s * s1);
            s = sn;
            c = cn;
        }
        (f, d)
    }
}

/// The modes of one geometry on the display grid.
#[derive(Clone, Debug)]
pub struct Basis {
    pub geom: Geometry,
    pub ax: Axis,
    pub ay: Axis,
    /// E/ε of every mode (index [`idx`])
    pub e_eps: Vec<f64>,
}

impl Basis {
    pub fn new(geom: Geometry) -> Self {
        let mut e_eps = vec![0.0; M * M];
        for i in 1..=M {
            for j in 1..=M {
                e_eps[idx(i, j)] = geom.e_eps(i, j);
            }
        }
        Basis {
            geom,
            ax: Axis::new(geom.lx(), N),
            ay: Axis::new(geom.ly(), N),
            e_eps,
        }
    }

    pub fn nx(&self) -> usize {
        self.ax.n
    }

    pub fn ny(&self) -> usize {
        self.ay.n
    }

    /// c(t) = c(t₀) e^(−iE (t − t₀)), E the full 3D energy.
    pub fn evolve(&self, c0: &[C64], dt: f64) -> Vec<C64> {
        c0.iter()
            .zip(&self.e_eps)
            .map(|(z, e)| {
                if z.norm_sqr() == 0.0 {
                    zero()
                } else {
                    z * C64::from_polar(1.0, -EPS * e * dt)
                }
            })
            .collect()
    }

    /// ψ on the grid (row-major, row = v index) from the coefficients.
    pub fn eval(&self, c: &[C64], out: &mut Vec<C64>) {
        let (nx, ny) = (self.nx(), self.ny());
        // T_i(v) = Σ_j c_ij Y_j(v) for the rows i that carry weight
        let rows: Vec<(usize, Vec<C64>)> = (1..=M)
            .filter_map(|i| {
                let ci = &c[idx(i, 1)..idx(i, 1) + M];
                if ci.iter().all(|z| z.norm_sqr() == 0.0) {
                    return None;
                }
                let mut t = vec![zero(); ny];
                for (j0, &z) in ci.iter().enumerate() {
                    if z.norm_sqr() == 0.0 {
                        continue;
                    }
                    for (tv, &y) in t.iter_mut().zip(self.ay.row(j0 + 1)) {
                        *tv += z * y;
                    }
                }
                Some((i, t))
            })
            .collect();
        out.clear();
        out.resize(nx * ny, zero());
        let ax = &self.ax;
        par::chunks_mut(out, nx, |iv, row| {
            for (i, t) in &rows {
                let tz = t[iv];
                for (o, &x) in row.iter_mut().zip(ax.row(*i)) {
                    *o += tz * x;
                }
            }
        });
    }

    /// Coefficients c_ij = Σ φ_ij ψ du dv of a field on the grid.
    pub fn project(&self, field: &[C64]) -> Vec<C64> {
        let (nx, ny) = (self.nx(), self.ny());
        let (dx, dy) = (self.ax.dx, self.ay.dx);
        // U_i(v) = Σ_u X_i(u) ψ(u, v) du
        let u: Vec<Vec<C64>> = par::map_range(M, |i0| {
            let xr = self.ax.row(i0 + 1);
            (0..ny)
                .map(|iv| {
                    let row = &field[iv * nx..(iv + 1) * nx];
                    let mut s = zero();
                    for (z, &x) in row.iter().zip(xr) {
                        s += z * x;
                    }
                    s * dx
                })
                .collect()
        });
        let mut c = vec![zero(); M * M];
        for i in 1..=M {
            for j in 1..=M {
                let mut s = zero();
                for (z, &y) in u[i - 1].iter().zip(self.ay.row(j)) {
                    s += z * y;
                }
                c[idx(i, j)] = s * dy;
            }
        }
        c
    }

    /// Exact moments from the coefficients.
    pub fn moments(&self, c: &[C64]) -> Moments2 {
        let norm: f64 = c.iter().map(|z| z.norm_sqr()).sum();
        if norm <= 0.0 {
            return Moments2::default();
        }
        let (mut e, mut e2) = (0.0, 0.0);
        for (z, &en) in c.iter().zip(&self.e_eps) {
            let p = z.norm_sqr();
            e += p * en;
            e2 += p * en * en;
        }
        e /= norm;
        e2 /= norm;
        // ⟨x⟩ = Σ_j Σ_{i,i'} c*_ij c_i'j X_ii'
        let mut x = 0.0;
        for j in 1..=M {
            for i in 1..=M {
                let a = c[idx(i, j)];
                if a.norm_sqr() == 0.0 {
                    continue;
                }
                for i2 in 1..=M {
                    let xe = self.ax.xm[(i - 1) * M + (i2 - 1)];
                    if xe != 0.0 {
                        x += xe * (a.conj() * c[idx(i2, j)]).re;
                    }
                }
            }
        }
        // ⟨v⟩ = Σ_i Σ_{j,j'} c*_ij c_ij' Y_jj'
        let mut y = 0.0;
        for i in 1..=M {
            let ci = &c[idx(i, 1)..idx(i, 1) + M];
            for (j0, a) in ci.iter().enumerate() {
                if a.norm_sqr() == 0.0 {
                    continue;
                }
                for (j20, b) in ci.iter().enumerate() {
                    let ye = self.ay.xm[j0 * M + j20];
                    if ye != 0.0 {
                        y += ye * (a.conj() * b).re;
                    }
                }
            }
        }
        Moments2 {
            norm,
            x: x / norm,
            y: y / norm,
            e,
            de: (e2 - e * e).max(0.0).sqrt(),
        }
    }

    /// ⟨L⟩ about the centre of the plane, L = (u − u_c) p_v − (v − v_c) p_u
    /// (the angular-momentum component perpendicular to the plane), from the
    /// coefficients. Only for sparse states (cost ∝ (number of modes)²).
    pub fn angular_momentum(&self, c: &[C64]) -> Option<f64> {
        let act: Vec<(usize, usize, C64)> = c
            .iter()
            .enumerate()
            .filter(|(_, z)| z.norm_sqr() > 1e-14)
            .map(|(k, z)| {
                let (i, j) = ij(k);
                (i, j, *z)
            })
            .collect();
        if act.len() > 200 {
            return None;
        }
        let norm: f64 = act.iter().map(|a| a.2.norm_sqr()).sum();
        let (lx, ly) = (self.ax.l, self.ay.l);
        let xc = |a: usize, b: usize| x_element(a, b, lx) - if a == b { 0.5 * lx } else { 0.0 };
        let yc = |a: usize, b: usize| x_element(a, b, ly) - if a == b { 0.5 * ly } else { 0.0 };
        // ⟨(u−u_c) ∂_v⟩ − ⟨(v−v_c) ∂_u⟩, then L = −i(…)
        let mut s = zero();
        for &(i, j, a) in &act {
            for &(i2, j2, b) in &act {
                let m = xc(i, i2) * d_element(j, j2, ly) - d_element(i, i2, lx) * yc(j, j2);
                if m != 0.0 {
                    s += a.conj() * b * m;
                }
            }
        }
        Some((c64(0.0, -1.0) * s).re / norm)
    }

    /// ψ, ∂ψ/∂u, ∂ψ/∂v at one point.
    pub fn eval_point(&self, c: &[C64], active: &[usize], u: f64, v: f64) -> (C64, C64, C64) {
        let (fx, dfx) = self.ax.at(u);
        let (fy, dfy) = self.ay.at(v);
        let (mut p, mut pu, mut pv) = (zero(), zero(), zero());
        for &k in active {
            let (i, j) = ij(k);
            let z = c[k];
            p += z * (fx[i - 1] * fy[j - 1]);
            pu += z * (dfx[i - 1] * fy[j - 1]);
            pv += z * (fx[i - 1] * dfy[j - 1]);
        }
        (p, pu, pv)
    }

    /// Probability current j = Im(ψ* ∇ψ) (ħ = m = 1) at a point.
    pub fn current_at(&self, c: &[C64], active: &[usize], u: f64, v: f64) -> (f64, f64, f64) {
        let (p, pu, pv) = self.eval_point(c, active, u, v);
        ((p.conj() * pu).im, (p.conj() * pv).im, p.norm_sqr())
    }
}

/// Indices of the non-zero coefficients.
pub fn active(c: &[C64]) -> Vec<usize> {
    c.iter().enumerate().filter(|(_, z)| z.norm_sqr() > 1e-16).map(|(k, _)| k).collect()
}

/// Marginal densities ∫|ψ|² dv (along u) and ∫|ψ|² du (along v).
pub fn marginals(field: &[C64], nx: usize, ny: usize, dx: f64, dy: f64) -> (Vec<f64>, Vec<f64>) {
    let mut px = vec![0.0; nx];
    let mut py = vec![0.0; ny];
    for iv in 0..ny {
        let row = &field[iv * nx..(iv + 1) * nx];
        let mut s = 0.0;
        for (k, z) in row.iter().enumerate() {
            let d = z.norm_sqr();
            px[k] += d * dy;
            s += d;
        }
        py[iv] = s * dx;
    }
    (px, py)
}

/// 1D coefficients ⟨X_m|g⟩, m = 1 … M, of a Gaussian
/// g(u) = (2πσ²)^(−1/4) exp(−(u − u₀)²/(4σ²) + i p u) in a well of length l
/// (Simpson quadrature on 4000 intervals, independent of the display grid).
pub fn gaussian_1d(l: f64, u0: f64, sigma: f64, p: f64) -> Vec<C64> {
    let n = 4000;
    let h = l / n as f64;
    let amp = (2.0 * PI * sigma * sigma).powf(-0.25);
    let norm = (2.0 / l).sqrt();
    let mut out = vec![zero(); M];
    for k in 0..=n {
        let u = k as f64 * h;
        let w = if k == 0 || k == n {
            1.0
        } else if k % 2 == 1 {
            4.0
        } else {
            2.0
        } * h
            / 3.0;
        let d = u - u0;
        let g = C64::from_polar(amp * (-d * d / (4.0 * sigma * sigma)).exp(), p * u) * w;
        for (m, o) in out.iter_mut().enumerate() {
            *o += g * (norm * ((m + 1) as f64 * PI * u / l).sin());
        }
    }
    out
}

/// Coefficients of the separable Gaussian packet g(u)·h(v) (σ the same in
/// both directions), centred at (u₀, v₀) with momentum (p_u, p_v). Not
/// renormalised: Σ|c|² is the fraction captured by the M × M modes.
pub fn gaussian_2d(basis: &Basis, u0: f64, v0: f64, sigma: f64, pu: f64, pv: f64) -> Vec<C64> {
    let a = gaussian_1d(basis.ax.l, u0, sigma, pu);
    let b = gaussian_1d(basis.ay.l, v0, sigma, pv);
    let mut c = vec![zero(); M * M];
    for i in 1..=M {
        for j in 1..=M {
            c[idx(i, j)] = a[i - 1] * b[j - 1];
        }
    }
    c
}

/// Born sampling of positions from |ψ|² on the grid (cell, then uniform in the cell).
pub struct Sampler2D {
    cdf: Vec<f64>,
    nx: usize,
    dx: f64,
    dy: f64,
}

impl Sampler2D {
    pub fn new(field: &[C64], nx: usize, dx: f64, dy: f64) -> Option<Self> {
        let mut cdf = Vec::with_capacity(field.len());
        let mut s = 0.0;
        for z in field {
            s += z.norm_sqr();
            cdf.push(s);
        }
        if s <= 0.0 || !s.is_finite() {
            return None;
        }
        cdf.iter_mut().for_each(|c| *c /= s);
        Some(Sampler2D { cdf, nx, dx, dy })
    }

    pub fn sample(&self, rng: &mut QbRng) -> [f64; 2] {
        let r = rng.uniform();
        let k = self.cdf.partition_point(|&c| c <= r).min(self.cdf.len() - 1);
        let (iu, iv) = (k % self.nx, k / self.nx);
        [(iu as f64 + rng.uniform()) * self.dx, (iv as f64 + rng.uniform()) * self.dy]
    }
}

/// Multiplies ψ by the detector window exp(−|r − r₀|²/(4w²)) (so |ψ|² gets a
/// Gaussian envelope of standard deviation w around r₀).
pub fn apply_window(field: &mut [C64], basis: &Basis, r0: [f64; 2], w: f64) {
    let nx = basis.nx();
    let (xs, ys) = (&basis.ax.xs, &basis.ay.xs);
    let wx: Vec<f64> = xs.iter().map(|x| (-(x - r0[0]).powi(2) / (4.0 * w * w)).exp()).collect();
    for (iv, row) in field.chunks_mut(nx).enumerate() {
        let wy = (-(ys[iv] - r0[1]).powi(2) / (4.0 * w * w)).exp();
        for (z, &a) in row.iter_mut().zip(&wx) {
            *z *= a * wy;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> Basis {
        Basis::new(Geometry::default())
    }

    #[test]
    fn modes_are_orthonormal_on_the_grid() {
        for geom in [
            Geometry::default(),
            Geometry {
                plane: Plane::XZ,
                ratio: 1.7,
                hidden: 2,
            },
        ] {
            let b = Basis::new(geom);
            for ax in [&b.ax, &b.ay] {
                for m in 1..=M {
                    for m2 in 1..=M {
                        let s: f64 = ax.row(m).iter().zip(ax.row(m2)).map(|(a, c)| a * c).sum::<f64>() * ax.dx;
                        let want = if m == m2 { 1.0 } else { 0.0 };
                        assert!((s - want).abs() < 1e-12, "⟨{m}|{m2}⟩ = {s}");
                    }
                }
            }
        }
    }

    #[test]
    fn eval_and_project_are_inverse() {
        let b = square();
        let mut rng = QbRng::seed(3);
        let mut c: Vec<C64> = (0..M * M).map(|_| c64(rng.normal(), rng.normal())).collect();
        normalize(&mut c);
        let mut f = vec![];
        b.eval(&c, &mut f);
        // grid norm = Σ|c|² (Parseval on the grid)
        let n: f64 = f.iter().map(|z| z.norm_sqr()).sum::<f64>() * b.ax.dx * b.ay.dx;
        assert!((n - 1.0).abs() < 1e-11, "grid norm {n}");
        let c2 = b.project(&f);
        let err: f64 = c.iter().zip(&c2).map(|(a, z)| (a - z).norm_sqr()).sum::<f64>().sqrt();
        assert!(err < 1e-11, "round trip error {err}");
    }

    #[test]
    fn energy_formula_eq_38_2() {
        // E(n1, n2, n3) = π²ħ²/(2m) [(n1² + n2²)/a² + n3²/b²] with ħ = m = a = 1
        let g = Geometry {
            plane: Plane::XY,
            ratio: 1.5,
            hidden: 2,
        };
        let b = g.b();
        let want = PI * PI / 2.0 * ((9.0 + 4.0) / (A * A) + 4.0 / (b * b));
        assert!((g.energy(3, 2) - want).abs() < 1e-12);
        let gz = Geometry { plane: Plane::XZ, ..g };
        // x–z plane: displayed (n1, n3), hidden n2 = 2
        let want = PI * PI / 2.0 * ((9.0 + 4.0) / (A * A) + 25.0 / (b * b));
        assert!((gz.energy(3, 5) - want).abs() < 1e-12);
        assert_eq!(gz.triple(3, 5), (3, 2, 5));
        // the grid Hamiltonian check: −½∇²φ = Eφ, by finite differences on a fine line
        let (i, j) = (3, 2);
        let (x, y, h) = (0.31, 0.47, 1e-4);
        let phi = |x: f64, y: f64| (i as f64 * PI * x).sin() * (j as f64 * PI * y).sin();
        let lap = (phi(x + h, y) + phi(x - h, y) + phi(x, y + h) + phi(x, y - h) - 4.0 * phi(x, y)) / (h * h);
        let e2d = -0.5 * lap / phi(x, y);
        assert!((e2d - EPS * 13.0).abs() < 1e-4, "{e2d}");
    }

    #[test]
    fn degeneracies() {
        let sq = Geometry::default();
        let lv = levels(&sq, 5, 1e9);
        let l5 = lv.iter().find(|l| (l.e - 5.0 - 1.0).abs() < 1e-9).unwrap();
        assert_eq!(l5.states, vec![(1, 2), (2, 1)]);
        assert_eq!(plane_degeneracy(&sq, 50.0 + 1.0).len(), 3); // (1,7) (5,5) (7,1)
        // b = 2a in the x–z plane: (1,4) and (2,2) are accidentally degenerate
        let xz = Geometry {
            plane: Plane::XZ,
            ratio: 2.0,
            hidden: 1,
        };
        assert!((xz.e_eps(1, 4) - xz.e_eps(2, 2)).abs() < 1e-12);
        let xz2 = Geometry { ratio: 2.05, ..xz };
        assert!((xz2.e_eps(1, 4) - xz2.e_eps(2, 2)).abs() > 0.1);
        // cube: E(1,1,2) = E(1,2,1) = E(2,1,1)
        assert_eq!(degenerate_3d(6.0, 1.0, 10).len(), 3);
    }

    #[test]
    fn norm_conserved_by_exact_evolution() {
        let b = square();
        let mut c = gaussian_2d(&b, 0.3, 0.4, 0.06, 20.0, -10.0);
        normalize(&mut c);
        let mut f = vec![];
        for t in [0.0, 0.1, 0.77, 4.0 / PI, 13.3] {
            let ct = b.evolve(&c, t);
            let n: f64 = ct.iter().map(|z| z.norm_sqr()).sum();
            assert!((n - 1.0).abs() < 1e-12);
            b.eval(&ct, &mut f);
            let ng: f64 = f.iter().map(|z| z.norm_sqr()).sum::<f64>() * b.ax.dx * b.ay.dx;
            assert!((ng - 1.0).abs() < 1e-10, "t = {t}: {ng}");
        }
        // revival: ψ(T_rev) = e^(iθ) ψ(0) (square box, any state)
        let tr = sq_rev();
        let ct = b.evolve(&c, tr);
        let overlap: C64 = c.iter().zip(&ct).map(|(a, z)| a.conj() * z).sum();
        assert!((overlap.norm() - 1.0).abs() < 1e-9, "|⟨ψ(0)|ψ(T)⟩| = {}", overlap.norm());
        // half revival: mirror image through the centre
        let ch = b.evolve(&c, 0.5 * tr);
        let mut f0 = vec![];
        b.eval(&c, &mut f0);
        b.eval(&ch, &mut f);
        let n = b.nx();
        let mut ov = zero();
        for iv in 0..n {
            for iu in 0..n {
                ov += f0[(n - 1 - iv) * n + (n - 1 - iu)].conj() * f[iv * n + iu];
            }
        }
        ov *= b.ax.dx * b.ay.dx;
        assert!((ov.norm() - 1.0).abs() < 1e-9, "mirror overlap {}", ov.norm());
    }

    fn sq_rev() -> f64 {
        Geometry::default().revival_time().unwrap()
    }

    #[test]
    fn degenerate_projection_keeps_the_subspace() {
        let g = Geometry::default();
        let mut c = vec![zero(); M * M];
        c[idx(1, 1)] = c64(0.6, 0.0);
        c[idx(1, 2)] = c64(0.0, 0.48);
        c[idx(2, 1)] = c64(0.64, 0.0);
        c[idx(2, 2)] = c64(0.0, 0.0);
        normalize(&mut c);
        let probs = level_probabilities(&g, &c);
        assert_eq!(probs.len(), 2);
        let p5: f64 = c[idx(1, 2)].norm_sqr() + c[idx(2, 1)].norm_sqr();
        assert!((probs[1].1 - p5).abs() < 1e-12);
        assert_eq!(probs[1].2.len(), 2);
        let before = (c[idx(1, 2)], c[idx(2, 1)]);
        let p = project_level(&g, &mut c, 5.0 + 1.0);
        assert!((p - p5).abs() < 1e-12);
        // both degenerate states survive, with their relative amplitude and phase
        assert_eq!(c[idx(1, 1)], zero());
        let ratio_before = before.0 / before.1;
        let ratio_after = c[idx(1, 2)] / c[idx(2, 1)];
        assert!((ratio_before - ratio_after).norm() < 1e-12);
        let n: f64 = c.iter().map(|z| z.norm_sqr()).sum();
        assert!((n - 1.0).abs() < 1e-12);
    }

    #[test]
    fn gaussian_expansion_is_complete() {
        let b = square();
        // at rest, σ = 0.05 a, well inside
        let c = gaussian_2d(&b, 0.32, 0.38, 0.05, 0.0, 0.0);
        let cap: f64 = c.iter().map(|z| z.norm_sqr()).sum();
        assert!((1.0 - cap) < 1e-6, "captured {cap}");
        // with momentum |p| ≈ 58 (n̄ ≈ 19): still captured by 40 modes
        let (u0, v0, s, pu, pv) = (0.35, 0.4, 0.07, 50.0, 30.0);
        let c = gaussian_2d(&b, u0, v0, s, pu, pv);
        let cap: f64 = c.iter().map(|z| z.norm_sqr()).sum();
        assert!((1.0 - cap) < 1e-5, "captured {cap}");
        // the reconstruction matches the Gaussian (2πσ²)^(−1/2) e^(−r²/(4σ²) + ip·r) on the grid
        let mut f = vec![];
        b.eval(&c, &mut f);
        let amp = 1.0 / (2.0 * PI * s * s).sqrt();
        let mut err = 0.0f64;
        for iv in 0..N {
            for iu in 0..N {
                let (u, v) = (b.ax.xs[iu], b.ay.xs[iv]);
                let g = C64::from_polar(
                    amp * (-((u - u0).powi(2) + (v - v0).powi(2)) / (4.0 * s * s)).exp(),
                    pu * u + pv * v,
                );
                err = err.max((f[iv * N + iu] - g).norm());
            }
        }
        assert!(err < 5e-3 * amp, "max deviation {err} (peak {amp})");
        // moments of the expansion: ⟨x⟩ = u0
        let mut cn = c.clone();
        normalize(&mut cn);
        let m = b.moments(&cn);
        assert!((m.x - u0).abs() < 1e-3 && (m.y - v0).abs() < 1e-3, "{m:?}");
    }

    #[test]
    fn moments_and_current() {
        let b = square();
        // (1,1) + (2,1): ⟨x⟩ oscillates around a/2 with amplitude |X_12| = 16a/(9π²)
        let mut c = vec![zero(); M * M];
        c[idx(1, 1)] = c64(1.0, 0.0);
        c[idx(2, 1)] = c64(1.0, 0.0);
        normalize(&mut c);
        let m = b.moments(&c);
        let amp = 16.0 / (9.0 * PI * PI);
        assert!((m.x - (0.5 - amp)).abs() < 1e-12, "{}", m.x);
        assert!((m.y - 0.5).abs() < 1e-12);
        // compare with the grid
        let mut f = vec![];
        b.eval(&c, &mut f);
        let (px, _) = marginals(&f, N, N, b.ax.dx, b.ay.dx);
        let xg: f64 = px.iter().zip(&b.ax.xs).map(|(p, x)| p * x).sum::<f64>() * b.ax.dx;
        assert!((xg - m.x).abs() < 1e-4);
        // (1,2) + i(2,1): stationary, circulating current, ⟨L⟩ ≠ 0; (1,2) + (2,1): j = 0
        let mut cv = vec![zero(); M * M];
        cv[idx(1, 2)] = c64(1.0, 0.0);
        cv[idx(2, 1)] = c64(0.0, 1.0);
        normalize(&mut cv);
        let act = active(&cv);
        let (jx, jy, _) = b.current_at(&cv, &act, 0.5, 0.25);
        let (jx2, jy2, _) = b.current_at(&cv, &act, 0.5, 0.75);
        assert!(jx.abs() > 1e-3 && jx * jx2 < 0.0, "circulation {jx} {jx2}");
        assert!(jy.abs() < 1e-9 && jy2.abs() < 1e-9);
        let l = b.angular_momentum(&cv).unwrap();
        assert!(l.abs() > 0.1, "L = {l}");
        let mut cr = cv.clone();
        cr[idx(2, 1)] = c64(cv[idx(1, 2)].re, 0.0);
        let act = active(&cr);
        let (jx, jy, _) = b.current_at(&cr, &act, 0.3, 0.6);
        assert!(jx.abs() < 1e-12 && jy.abs() < 1e-12);
        assert!(b.angular_momentum(&cr).unwrap().abs() < 1e-12);
        // energy of the 3D state
        assert!((b.moments(&cv).e - 6.0).abs() < 1e-12);
    }

    #[test]
    fn revival_times() {
        assert!((Geometry::default().revival_time().unwrap() - 4.0 / PI).abs() < 1e-12);
        let g = Geometry {
            plane: Plane::XZ,
            ratio: 2.0,
            hidden: 1,
        };
        assert!((g.revival_time().unwrap() - 16.0 / PI).abs() < 1e-12);
        let g = Geometry {
            plane: Plane::XZ,
            ratio: 2f64.sqrt(),
            hidden: 1,
        };
        // b²/a² = 2 (to rounding): T = 2·4a²/π
        assert!((g.revival_time().unwrap() - 8.0 / PI).abs() < 1e-9);
        let b = Basis::new(g);
        let mut c = gaussian_2d(&b, 0.3, 0.6, 0.06, 0.0, 0.0);
        normalize(&mut c);
        let ct = b.evolve(&c, g.revival_time().unwrap());
        let ov: C64 = c.iter().zip(&ct).map(|(a, z)| a.conj() * z).sum();
        assert!((ov.norm() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn position_collapse_is_captured() {
        let b = square();
        let mut c = vec![zero(); M * M];
        c[idx(1, 1)] = c64(1.0, 0.0);
        let mut f = vec![];
        b.eval(&c, &mut f);
        apply_window(&mut f, &b, [0.4, 0.6], 0.04);
        let cc = b.project(&f);
        let n_grid: f64 = f.iter().map(|z| z.norm_sqr()).sum::<f64>() * b.ax.dx * b.ay.dx;
        let n_modes: f64 = cc.iter().map(|z| z.norm_sqr()).sum();
        assert!(n_modes / n_grid > 0.999, "captured {}", n_modes / n_grid);
        let mut rng = QbRng::seed(1);
        let s = Sampler2D::new(&f, N, b.ax.dx, b.ay.dx).unwrap();
        let mut mx = 0.0;
        for _ in 0..4000 {
            mx += s.sample(&mut rng)[0];
        }
        assert!((mx / 4000.0 - 0.4).abs() < 0.01);
    }

    #[test]
    fn matrix_elements_match_quadrature() {
        let l = 1.7;
        let n = 20000;
        let h = l / n as f64;
        for a in 1..=6 {
            for b in 1..=6 {
                let (mut xs, mut ds) = (0.0, 0.0);
                for k in 0..n {
                    let x = (k as f64 + 0.5) * h;
                    let fa = (2.0 / l).sqrt() * (a as f64 * PI * x / l).sin();
                    let fb = (2.0 / l).sqrt() * (b as f64 * PI * x / l).sin();
                    let dfb = (2.0 / l).sqrt() * (b as f64 * PI / l) * (b as f64 * PI * x / l).cos();
                    xs += fa * x * fb * h;
                    ds += fa * dfb * h;
                }
                assert!((xs - x_element(a, b, l)).abs() < 1e-6, "x {a}{b}: {xs} vs {}", x_element(a, b, l));
                assert!((ds - d_element(a, b, l)).abs() < 1e-5, "d {a}{b}: {ds} vs {}", d_element(a, b, l));
            }
        }
    }

    #[test]
    fn degenerate_combination_is_stationary_and_slosh_has_its_period() {
        let b = square();
        let mut c = vec![zero(); M * M];
        c[idx(1, 2)] = c64(1.0, 0.0);
        c[idx(2, 1)] = c64(0.0, 1.0);
        normalize(&mut c);
        let (mut f0, mut f1) = (vec![], vec![]);
        b.eval(&c, &mut f0);
        b.eval(&b.evolve(&c, 0.377), &mut f1);
        let d = f0
            .iter()
            .zip(&f1)
            .map(|(a, z)| (a.norm_sqr() - z.norm_sqr()).abs())
            .fold(0.0, f64::max);
        assert!(d < 1e-12, "density changed by {d}");
        // (1,1) + (2,1): ⟨x⟩(t) = a/2 − (16a/9π²) cos(3εt)
        let mut c = vec![zero(); M * M];
        c[idx(1, 1)] = c64(1.0, 0.0);
        c[idx(2, 1)] = c64(1.0, 0.0);
        normalize(&mut c);
        for t in [0.0, 0.1, 0.2, 4.0 / (3.0 * PI)] {
            let m = b.moments(&b.evolve(&c, t));
            let want = 0.5 - 16.0 / (9.0 * PI * PI) * (3.0 * EPS * t).cos();
            assert!((m.x - want).abs() < 1e-12, "t = {t}: {} vs {want}", m.x);
        }
    }

    #[test]
    fn energy_sampling_follows_level_probabilities() {
        // a packet: sampling levels with rng.weighted reproduces Σ|c|² per level
        let b = square();
        let mut c = gaussian_2d(&b, 0.3, 0.4, 0.08, 0.0, 0.0);
        normalize(&mut c);
        let probs = level_probabilities(&b.geom, &c);
        let total: f64 = probs.iter().map(|l| l.1).sum();
        assert!((total - 1.0).abs() < 1e-12);
        let w: Vec<f64> = probs.iter().map(|l| l.1).collect();
        let mut rng = QbRng::seed(5);
        let n = 200_000;
        let mut counts = vec![0usize; w.len()];
        for _ in 0..n {
            counts[rng.weighted(&w).unwrap()] += 1;
        }
        let k = (0..w.len()).max_by(|&a, &b| w[a].total_cmp(&w[b])).unwrap();
        let f = counts[k] as f64 / n as f64;
        let sd = (w[k] * (1.0 - w[k]) / n as f64).sqrt();
        assert!((f - w[k]).abs() < 5.0 * sd, "{f} vs {}", w[k]);
        // ⟨E⟩ from the levels = ⟨E⟩ from the moments
        let e: f64 = probs.iter().map(|l| l.0 * l.1).sum();
        assert!((e - b.moments(&c).e).abs() < 1e-9);
    }

    #[test]
    fn eval_speed() {
        // the per-frame cost: a full 40 × 40 expansion on the 160 × 160 grid
        let b = square();
        let c = gaussian_2d(&b, 0.3, 0.4, 0.05, 30.0, 10.0);
        let mut f = vec![];
        let t0 = std::time::Instant::now();
        for k in 0..20 {
            let ct = b.evolve(&c, 0.01 * k as f64);
            b.eval(&ct, &mut f);
            let _ = b.moments(&ct);
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / 20.0;
        eprintln!("eval + moments of 40×40 modes on {N}×{N}: {ms:.2} ms per frame (native, release, parallel rows)");
    }
}
