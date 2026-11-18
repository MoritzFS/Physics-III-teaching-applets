//! Two-level (spin-½) state algebra in the z basis, with the conventions of
//! the problem sets:
//!
//! * column vectors in the z basis, |↑z⟩ = (1, 0)ᵀ, |↓z⟩ = (0, 1)ᵀ (Eq. 50.1);
//! * eigenstates of u·σ for u = (sin θ cos φ, sin θ sin φ, cos θ) (Eq. 50.3):
//!   |↑u⟩ = cos(θ/2) e^(−iφ/2) |↑z⟩ + sin(θ/2) e^(iφ/2) |↓z⟩,
//!   |↓u⟩ = −sin(θ/2) e^(−iφ/2) |↑z⟩ + cos(θ/2) e^(iφ/2) |↓z⟩;
//! * rotations R_u(α) = e^(−iα u·σ/2) = cos(α/2) I − i sin(α/2) u·σ (Eq. 51.2, 51.3),
//!   which turn the Bloch vector by +α about u (right-handed);
//! * H₀ = μ_B B σ_z = (ħω/2) σ_z with ω = 2μ_B B/ħ (Eq. 51.1), so
//!   e^(−iH₀t/ħ) = R_z(ωt) = diag(e^(−iωt/2), e^(iωt/2)) (Eq. 51.9).
//!
//! The unit tests check these statements, the probabilities of PS13 50, the
//! states of PS13 51 (Eq. 51.5, 51.8, 51.12, 51.13) and the Ramsey fringe
//! (Eq. 51.14) against explicit matrix products.

use std::f64::consts::PI;
use std::ops::Mul;

use qb_core::{C64, QbRng, c64};

/// A direction or Bloch vector (x, y, z).
pub type Vec3 = [f64; 3];

pub const EX: Vec3 = [1.0, 0.0, 0.0];
pub const EY: Vec3 = [0.0, 1.0, 0.0];
pub const EZ: Vec3 = [0.0, 0.0, 1.0];

pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

pub fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}

pub fn scale(a: Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// a / |a| (z for the zero vector).
pub fn normalize(a: Vec3) -> Vec3 {
    let n = norm(a);
    if n > 1e-300 { scale(a, 1.0 / n) } else { EZ }
}

/// The unit vector u(θ, φ) = (sin θ cos φ, sin θ sin φ, cos θ) of Eq. 50.2.
pub fn dir(theta: f64, phi: f64) -> Vec3 {
    [theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()]
}

/// (θ, φ) of a direction (φ = 0 on the z axis).
pub fn angles(n: Vec3) -> (f64, f64) {
    let r = norm(n);
    if r < 1e-300 {
        return (0.0, 0.0);
    }
    let theta = (n[2] / r).clamp(-1.0, 1.0).acos();
    let rho = (n[0] * n[0] + n[1] * n[1]).sqrt();
    let phi = if rho < 1e-12 * r { 0.0 } else { n[1].atan2(n[0]) };
    (theta, phi)
}

/// v turned by +α about the unit vector u (Rodrigues, right-handed).
pub fn rotate_vec(v: Vec3, u: Vec3, alpha: f64) -> Vec3 {
    let u = normalize(u);
    let (s, c) = alpha.sin_cos();
    let uxv = cross(u, v);
    let udv = dot(u, v);
    [
        v[0] * c + uxv[0] * s + u[0] * udv * (1.0 - c),
        v[1] * c + uxv[1] * s + u[1] * udv * (1.0 - c),
        v[2] * c + uxv[2] * s + u[2] * udv * (1.0 - c),
    ]
}

/// A spin-½ state a|↑z⟩ + b|↓z⟩ (amplitudes in the z basis, global phase kept).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spinor {
    pub up: C64,
    pub down: C64,
}

impl Spinor {
    /// |↑z⟩ = (1, 0)ᵀ (Eq. 50.1)
    pub const UP_Z: Spinor = Spinor {
        up: C64::new(1.0, 0.0),
        down: C64::new(0.0, 0.0),
    };
    /// |↓z⟩ = (0, 1)ᵀ (Eq. 50.1)
    pub const DOWN_Z: Spinor = Spinor {
        up: C64::new(0.0, 0.0),
        down: C64::new(1.0, 0.0),
    };

    pub fn new(up: C64, down: C64) -> Self {
        Spinor { up, down }
    }

    /// Real amplitudes (a, b).
    pub fn real(a: f64, b: f64) -> Self {
        Spinor::new(c64(a, 0.0), c64(b, 0.0))
    }

    pub fn norm_sqr(&self) -> f64 {
        self.up.norm_sqr() + self.down.norm_sqr()
    }

    pub fn normalized(self) -> Self {
        let n = self.norm_sqr().sqrt();
        if n > 1e-300 { self.scaled(c64(1.0 / n, 0.0)) } else { Spinor::UP_Z }
    }

    pub fn scaled(self, c: C64) -> Self {
        Spinor::new(self.up * c, self.down * c)
    }

    /// ⟨self|other⟩
    pub fn inner(&self, other: &Spinor) -> C64 {
        self.up.conj() * other.up + self.down.conj() * other.down
    }

    /// P(φ|ψ) = |⟨φ|ψ⟩|² (Eq. 50.5) for ψ = self (both normalised).
    pub fn prob(&self, phi: &Spinor) -> f64 {
        phi.inner(self).norm_sqr()
    }

    /// |↑u⟩ of Eq. 50.3a for u = u(θ, φ).
    pub fn up_along(theta: f64, phi: f64) -> Self {
        let (c, s) = ((0.5 * theta).cos(), (0.5 * theta).sin());
        Spinor::new(C64::from_polar(c, -0.5 * phi), C64::from_polar(s, 0.5 * phi))
    }

    /// |↓u⟩ of Eq. 50.3b for u = u(θ, φ).
    pub fn down_along(theta: f64, phi: f64) -> Self {
        let (c, s) = ((0.5 * theta).cos(), (0.5 * theta).sin());
        Spinor::new(C64::from_polar(-s, -0.5 * phi), C64::from_polar(c, 0.5 * phi))
    }

    /// |↑u⟩ (Eq. 50.3a) for a direction vector.
    pub fn up_dir(n: Vec3) -> Self {
        let (t, p) = angles(n);
        Self::up_along(t, p)
    }

    /// |↓u⟩ (Eq. 50.3b) for a direction vector.
    pub fn down_dir(n: Vec3) -> Self {
        let (t, p) = angles(n);
        Self::down_along(t, p)
    }

    /// Bloch vector (⟨σx⟩, ⟨σy⟩, ⟨σz⟩) = (2 Re a*b, 2 Im a*b, |a|² − |b|²).
    pub fn bloch(&self) -> Vec3 {
        let ab = self.up.conj() * self.down;
        [2.0 * ab.re, 2.0 * ab.im, self.up.norm_sqr() - self.down.norm_sqr()]
    }

    /// P(↑u|ψ) = |⟨↑u|ψ⟩|² for the axis n.
    pub fn p_up(&self, n: Vec3) -> f64 {
        self.prob(&Spinor::up_dir(n))
    }

    /// ⟨ψ|M|ψ⟩
    pub fn expect(&self, m: &Mat2) -> C64 {
        self.inner(&(*m * *self))
    }
}

/// A 2 × 2 complex matrix (operator in the z basis), row-major.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat2(pub [[C64; 2]; 2]);

impl Mat2 {
    pub fn new(a: C64, b: C64, c: C64, d: C64) -> Self {
        Mat2([[a, b], [c, d]])
    }

    pub fn identity() -> Self {
        let (o, z) = (c64(1.0, 0.0), c64(0.0, 0.0));
        Mat2::new(o, z, z, o)
    }

    /// σx (Eq. 49.1)
    pub fn sigma_x() -> Self {
        let (o, z) = (c64(1.0, 0.0), c64(0.0, 0.0));
        Mat2::new(z, o, o, z)
    }

    /// σy (Eq. 49.1)
    pub fn sigma_y() -> Self {
        let (i, z) = (c64(0.0, 1.0), c64(0.0, 0.0));
        Mat2::new(z, -i, i, z)
    }

    /// σz (Eq. 49.1)
    pub fn sigma_z() -> Self {
        let (o, z) = (c64(1.0, 0.0), c64(0.0, 0.0));
        Mat2::new(o, z, z, -o)
    }

    /// u·σ (Eq. 50.2)
    pub fn sigma_n(n: Vec3) -> Self {
        Mat2::sigma_x()
            .scale(c64(n[0], 0.0))
            .add(&Mat2::sigma_y().scale(c64(n[1], 0.0)))
            .add(&Mat2::sigma_z().scale(c64(n[2], 0.0)))
    }

    /// R_u(α) = e^(−iα u·σ/2) = cos(α/2) I − i sin(α/2) u·σ (Eq. 51.2, 51.3); u is normalised here.
    pub fn rotation(n: Vec3, alpha: f64) -> Self {
        let n = normalize(n);
        let (s, c) = (0.5 * alpha).sin_cos();
        Mat2::identity().scale(c64(c, 0.0)).add(&Mat2::sigma_n(n).scale(c64(0.0, -s)))
    }

    pub fn rx(alpha: f64) -> Self {
        Self::rotation(EX, alpha)
    }

    pub fn ry(alpha: f64) -> Self {
        Self::rotation(EY, alpha)
    }

    pub fn rz(alpha: f64) -> Self {
        Self::rotation(EZ, alpha)
    }

    /// e^(−iH₀t/ħ) for H₀ = μ_B B σ_z = (ħω/2) σ_z: R_z(ωt) = diag(e^(−iωt/2), e^(iωt/2)) (Eq. 51.9).
    pub fn free_evolution(omega: f64, t: f64) -> Self {
        Self::rz(omega * t)
    }

    pub fn add(&self, o: &Mat2) -> Self {
        let mut m = self.0;
        for (i, row) in m.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v += o.0[i][j];
            }
        }
        Mat2(m)
    }

    pub fn scale(&self, c: C64) -> Self {
        let mut m = self.0;
        m.iter_mut().flatten().for_each(|v| *v *= c);
        Mat2(m)
    }

    pub fn adjoint(&self) -> Self {
        let m = self.0;
        Mat2::new(m[0][0].conj(), m[1][0].conj(), m[0][1].conj(), m[1][1].conj())
    }

    /// Largest |entry| of self − o.
    pub fn max_diff(&self, o: &Mat2) -> f64 {
        let mut d: f64 = 0.0;
        for i in 0..2 {
            for j in 0..2 {
                d = d.max((self.0[i][j] - o.0[i][j]).norm());
            }
        }
        d
    }

    /// e^M by its Taylor series (for the tests: Eq. 51.3 "by expanding the exponential").
    pub fn exp_series(&self) -> Self {
        let mut term = Mat2::identity();
        let mut sum = Mat2::identity();
        for k in 1..40 {
            term = (term * *self).scale(c64(1.0 / k as f64, 0.0));
            sum = sum.add(&term);
        }
        sum
    }
}

impl Mul for Mat2 {
    type Output = Mat2;
    fn mul(self, o: Mat2) -> Mat2 {
        let (a, b) = (self.0, o.0);
        let e = |i: usize, j: usize| a[i][0] * b[0][j] + a[i][1] * b[1][j];
        Mat2::new(e(0, 0), e(0, 1), e(1, 0), e(1, 1))
    }
}

impl Mul<Spinor> for Mat2 {
    type Output = Spinor;
    fn mul(self, s: Spinor) -> Spinor {
        let m = self.0;
        Spinor::new(m[0][0] * s.up + m[0][1] * s.down, m[1][0] * s.up + m[1][1] * s.down)
    }
}

/// Projective measurement of u·σ (Born rule): returns (outcome ↑?, the state after it).
/// The state after the measurement is the eigenvector |↑u⟩ or |↓u⟩ of Eq. 50.3
/// (the projection P̂|ψ⟩/‖P̂|ψ⟩‖ differs from it only by an unobservable global phase).
pub fn measure(state: &Spinor, n: Vec3, rng: &mut QbRng) -> (bool, Spinor) {
    let up = rng.bool(state.p_up(n));
    (up, if up { Spinor::up_dir(n) } else { Spinor::down_dir(n) })
}

/// A random pure state, uniform on the Bloch sphere (an atom from an unpolarised oven).
pub fn random_state(rng: &mut QbRng) -> Spinor {
    let z = rng.range(-1.0, 1.0);
    let phi = rng.range(0.0, 2.0 * PI);
    Spinor::up_along(z.acos(), phi)
}

/// Physical constants (CODATA 2018, SI) and the Larmor frequency of PS13 51.
pub mod units {
    use std::f64::consts::PI;

    /// Bohr magneton μ_B in J/T
    pub const MU_B: f64 = 9.274_010_078_3e-24;
    /// reduced Planck constant ħ in J s
    pub const HBAR: f64 = 1.054_571_817e-34;
    /// Planck constant h in J s
    pub const H: f64 = 6.626_070_15e-34;
    /// elementary charge in C (J per eV)
    pub const EV: f64 = 1.602_176_634e-19;

    /// Larmor angular frequency ω = 2μ_B B/ħ in rad/s (g_s ≈ 2 as in Eq. 51.1), B in tesla.
    pub fn larmor_omega(b_tesla: f64) -> f64 {
        2.0 * MU_B * b_tesla / HBAR
    }

    /// Larmor frequency f = ω/2π in Hz, B in tesla.
    pub fn larmor_freq(b_tesla: f64) -> f64 {
        larmor_omega(b_tesla) / (2.0 * PI)
    }

    /// ω in rad/ns for B in mT (the units of the Ramsey scene).
    pub fn omega_per_ns(b_mt: f64) -> f64 {
        larmor_omega(b_mt * 1e-3) * 1e-9
    }

    /// B in mT from ω in rad/ns: B = ħω/(2μ_B).
    pub fn b_mt_from_omega(omega_per_ns: f64) -> f64 {
        HBAR * omega_per_ns * 1e9 / (2.0 * MU_B) * 1e3
    }

    /// Zeeman splitting ħω = 2μ_B B in eV, B in tesla.
    pub fn splitting_ev(b_tesla: f64) -> f64 {
        2.0 * MU_B * b_tesla / EV
    }
}

/// Axis of the second Ramsey pulse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PulseAxis {
    X,
    Y,
}

impl PulseAxis {
    pub fn dir(self) -> Vec3 {
        match self {
            PulseAxis::X => EX,
            PulseAxis::Y => EY,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            PulseAxis::X => "x",
            PulseAxis::Y => "y",
        }
    }
}

/// The Ramsey sequence of PS13 51 by explicit matrix products:
/// |ϕ(τ)⟩ = R_second(π/2) e^(−iH₀τ/ħ) R_y(π/2) |↑z⟩.
pub fn ramsey_final_state(omega: f64, tau: f64, second: PulseAxis) -> Spinor {
    Mat2::rotation(second.dir(), 0.5 * PI) * Mat2::free_evolution(omega, tau) * Mat2::ry(0.5 * PI) * Spinor::UP_Z
}

/// The fringe P(↑z|ϕ(τ)) derived with the problem set's conventions:
/// second pulse about x: (1 + sin ωτ)/2 (Eq. 51.14); about y: (1 − cos ωτ)/2 = sin²(ωτ/2).
pub fn ramsey_p_up(omega: f64, tau: f64, second: PulseAxis) -> f64 {
    match second {
        PulseAxis::X => 0.5 * (1.0 + (omega * tau).sin()),
        PulseAxis::Y => 0.5 * (1.0 - (omega * tau).cos()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    fn close(a: C64, b: C64) -> bool {
        (a - b).norm() < 1e-12
    }

    fn same(a: &Spinor, b: &Spinor) -> bool {
        close(a.up, b.up) && close(a.down, b.down)
    }

    fn grid() -> Vec<(f64, f64)> {
        let mut v = vec![];
        for i in 0..=12 {
            for j in 0..12 {
                v.push((PI * i as f64 / 12.0, 2.0 * PI * j as f64 / 12.0 - PI));
            }
        }
        v
    }

    #[test]
    fn pauli_algebra() {
        let (x, y, z, id) = (Mat2::sigma_x(), Mat2::sigma_y(), Mat2::sigma_z(), Mat2::identity());
        for s in [x, y, z] {
            assert!((s * s).max_diff(&id) < TOL);
            assert!(s.adjoint().max_diff(&s) < TOL);
        }
        assert!((x * y).max_diff(&z.scale(c64(0.0, 1.0))) < TOL);
        assert!((y * z).max_diff(&x.scale(c64(0.0, 1.0))) < TOL);
        assert!((z * x).max_diff(&y.scale(c64(0.0, 1.0))) < TOL);
    }

    /// Eq. 50.3: |↑u⟩, |↓u⟩ are orthonormal eigenvectors of u·σ with ±1.
    #[test]
    fn eq_50_3_eigenvectors() {
        for (theta, phi) in grid() {
            let u = dir(theta, phi);
            let s = Mat2::sigma_n(u);
            let up = Spinor::up_along(theta, phi);
            let dn = Spinor::down_along(theta, phi);
            assert!(same(&(s * up), &up), "σ·u |↑u⟩ ≠ |↑u⟩ at θ={theta}, φ={phi}");
            assert!(same(&(s * dn), &dn.scaled(c64(-1.0, 0.0))));
            assert!((up.norm_sqr() - 1.0).abs() < TOL && (dn.norm_sqr() - 1.0).abs() < TOL);
            assert!(up.inner(&dn).norm() < TOL);
            // Bloch vector of |↑u⟩ is u, of |↓u⟩ is −u
            let b = up.bloch();
            let d = dn.bloch();
            for k in 0..3 {
                assert!((b[k] - u[k]).abs() < 1e-12 && (d[k] + u[k]).abs() < 1e-12);
            }
        }
    }

    /// PS13 50 a): |↑x⟩ = (|↑z⟩ + |↓z⟩)/√2, |↓x⟩ = −(|↑z⟩ − |↓z⟩)/√2 (Eq. 50.6), eigenvectors of σx.
    #[test]
    fn up_down_x() {
        let r = std::f64::consts::FRAC_1_SQRT_2;
        let upx = Spinor::up_along(0.5 * PI, 0.0);
        let dnx = Spinor::down_along(0.5 * PI, 0.0);
        assert!(same(&upx, &Spinor::real(r, r)));
        assert!(same(&dnx, &Spinor::real(-r, r)));
        assert!(same(&(Mat2::sigma_x() * upx), &upx));
        assert!(same(&(Mat2::sigma_x() * dnx), &dnx.scaled(c64(-1.0, 0.0))));
        // the direction helpers agree
        assert!(same(&Spinor::up_dir(EX), &upx));
        assert!(same(&Spinor::down_dir(EX), &dnx));
        assert!(same(&Spinor::up_dir(EZ), &Spinor::UP_Z));
        assert!(same(&Spinor::down_dir(EZ), &Spinor::DOWN_Z));
    }

    /// PS13 50 b, c) and the general rule P(↑θ|↑z) = cos²(θ/2) = (1 + n·u)/2.
    #[test]
    fn stern_gerlach_probabilities() {
        let up_z = Spinor::UP_Z;
        assert!((up_z.p_up(EX) - 0.5).abs() < TOL); // 50 b)
        let up_x = Spinor::up_dir(EX);
        assert!((up_x.p_up(EZ) - 0.5).abs() < TOL); // 50 c)
        assert!((up_z.p_up(EZ) - 1.0).abs() < TOL); // z–z
        for (theta, phi) in grid() {
            let p = up_z.p_up(dir(theta, phi));
            assert!((p - (0.5 * theta).cos().powi(2)).abs() < 1e-12);
            // any pure state n: P(↑u) = (1 + n·u)/2
            let n = dir(0.7, 1.9);
            let s = Spinor::up_dir(n);
            let u = dir(theta, phi);
            assert!((s.p_up(u) - 0.5 * (1.0 + dot(n, u))).abs() < 1e-12);
        }
    }

    /// Born sampling of a z–x–z chain: the last box gives ↑z half of the time.
    #[test]
    fn measurement_statistics() {
        let mut rng = QbRng::seed(7);
        let n = 40_000;
        let mut up_last = 0;
        let mut passed = 0;
        for _ in 0..n {
            let (up1, s1) = measure(&random_state(&mut rng), EZ, &mut rng);
            if !up1 {
                continue;
            }
            let (up2, s2) = measure(&s1, EX, &mut rng);
            if !up2 {
                continue;
            }
            passed += 1;
            let (up3, _) = measure(&s2, EZ, &mut rng);
            up_last += up3 as usize;
        }
        let f = up_last as f64 / passed as f64;
        let sd = (0.25 / passed as f64).sqrt();
        assert!((f - 0.5).abs() < 5.0 * sd, "z–x–z: {f} from {passed} atoms");
        // about a quarter of the oven's atoms pass the first two boxes
        assert!((passed as f64 / n as f64 - 0.25).abs() < 0.01);
        // z–z: always ↑z again
        for _ in 0..1000 {
            let (up, s) = measure(&Spinor::UP_Z, EZ, &mut rng);
            assert!(up && same(&s, &Spinor::UP_Z));
        }
    }

    /// Eq. 51.3: e^(−iασx/2) = cos(α/2) I − i sin(α/2) σx (and for y, z, any u), by the series.
    #[test]
    fn eq_51_3_rotation_series() {
        for k in 0..16 {
            let alpha = -2.0 * PI + k as f64 * 0.37;
            for n in [EX, EY, EZ, normalize([0.3, -0.5, 0.8])] {
                let series = Mat2::sigma_n(n).scale(c64(0.0, -0.5 * alpha)).exp_series();
                assert!(series.max_diff(&Mat2::rotation(n, alpha)) < 1e-12);
            }
        }
    }

    /// R_u(α) turns the Bloch vector by +α about u (right-handed), and is unitary.
    #[test]
    fn rotation_is_bloch_rotation() {
        let s = Spinor::up_along(1.1, -0.4);
        for (theta, phi) in grid() {
            let u = dir(theta, phi);
            let alpha = 0.3 + theta * 1.7;
            let r = Mat2::rotation(u, alpha);
            assert!((r * r.adjoint()).max_diff(&Mat2::identity()) < 1e-12);
            let b = (r * s).bloch();
            let want = rotate_vec(s.bloch(), u, alpha);
            for k in 0..3 {
                assert!((b[k] - want[k]).abs() < 1e-12);
            }
        }
        // R_y(π/2) takes +z to +x, R_x(π/2) takes +y to +z
        let b = (Mat2::ry(0.5 * PI) * Spinor::UP_Z).bloch();
        assert!((b[0] - 1.0).abs() < TOL);
        let b = (Mat2::rx(0.5 * PI) * Spinor::up_dir(EY)).bloch();
        assert!((b[2] - 1.0).abs() < TOL);
    }

    /// Eq. 51.8: R_y(π/2)|↑z⟩ = (1, 1)ᵀ/√2 = |↑x⟩ (no extra phase).
    #[test]
    fn eq_51_8_first_pulse() {
        let r = std::f64::consts::FRAC_1_SQRT_2;
        let ry = Mat2::ry(0.5 * PI);
        assert!(ry.max_diff(&Mat2::new(c64(r, 0.0), c64(-r, 0.0), c64(r, 0.0), c64(r, 0.0))) < TOL);
        assert!(same(&(ry * Spinor::UP_Z), &Spinor::real(r, r)));
    }

    /// Eq. 51.9 and 51.5: e^(−iH₀t/ħ) = diag(e^(−iωt/2), e^(iωt/2)), |φ(t)⟩ = (e^(−iωt/2), e^(iωt/2))ᵀ/√2.
    #[test]
    fn eq_51_5_free_precession() {
        let r = std::f64::consts::FRAC_1_SQRT_2;
        let omega = 0.1759;
        for k in 0..50 {
            let t = k as f64 * 1.3;
            let u = Mat2::free_evolution(omega, t);
            let want = Mat2::new(
                C64::from_polar(1.0, -0.5 * omega * t),
                c64(0.0, 0.0),
                c64(0.0, 0.0),
                C64::from_polar(1.0, 0.5 * omega * t),
            );
            assert!(u.max_diff(&want) < 1e-12);
            // H₀ = (ħω/2)σz: the same as the series of e^(−iωtσz/2)
            let series = Mat2::sigma_z().scale(c64(0.0, -0.5 * omega * t)).exp_series();
            assert!(series.max_diff(&want) < 1e-10);
            let phi = u * (Mat2::ry(0.5 * PI) * Spinor::UP_Z);
            let want = Spinor::new(C64::from_polar(r, -0.5 * omega * t), C64::from_polar(r, 0.5 * omega * t));
            assert!(same(&phi, &want));
            // the Bloch vector precesses counter-clockwise about z: azimuth ωt
            let b = phi.bloch();
            assert!((b[0] - (omega * t).cos()).abs() < 1e-12 && (b[1] - (omega * t).sin()).abs() < 1e-12);
        }
    }

    /// Eq. 51.12: P(↑u|φ(t)) = cos²(φ/2 − ωt/2) for u in the x–y plane; = 1 if φ = ωt.
    #[test]
    fn eq_51_12_rotating_axis() {
        let omega = 0.42;
        for k in 0..40 {
            let t = k as f64 * 0.77;
            let state = Mat2::free_evolution(omega, t) * Mat2::ry(0.5 * PI) * Spinor::UP_Z;
            for j in 0..12 {
                let phi = j as f64 * 0.55;
                let p = state.prob(&Spinor::up_along(0.5 * PI, phi));
                assert!((p - (0.5 * phi - 0.5 * omega * t).cos().powi(2)).abs() < 1e-12);
            }
            assert!((state.prob(&Spinor::up_along(0.5 * PI, omega * t)) - 1.0).abs() < 1e-12);
        }
    }

    /// Eq. 51.13: R_x(π/2) = (1/√2)[[1, −i], [−i, 1]] and
    /// |ϕ(τ)⟩ = ½(e^(−iωτ/2) − i e^(iωτ/2), −i e^(−iωτ/2) + e^(iωτ/2))ᵀ.
    #[test]
    fn eq_51_13_second_pulse() {
        let r = std::f64::consts::FRAC_1_SQRT_2;
        let rx = Mat2::rx(0.5 * PI);
        assert!(rx.max_diff(&Mat2::new(c64(r, 0.0), c64(0.0, -r), c64(0.0, -r), c64(r, 0.0))) < TOL);
        let omega = 0.31;
        let i = c64(0.0, 1.0);
        for k in 0..60 {
            let tau = k as f64 * 0.91;
            let s = ramsey_final_state(omega, tau, PulseAxis::X);
            let (em, ep) = (C64::from_polar(1.0, -0.5 * omega * tau), C64::from_polar(1.0, 0.5 * omega * tau));
            let want = Spinor::new(0.5 * (em - i * ep), 0.5 * (-i * em + ep));
            assert!(same(&s, &want));
        }
    }

    /// Eq. 51.14 (and the y variant) against the explicit matrix product.
    #[test]
    fn ramsey_fringe_formula() {
        for omega in [0.05, 0.1759, 0.9] {
            for k in 0..400 {
                let tau = k as f64 * 0.37;
                for axis in [PulseAxis::X, PulseAxis::Y] {
                    let p = ramsey_final_state(omega, tau, axis).prob(&Spinor::UP_Z);
                    assert!((p - ramsey_p_up(omega, tau, axis)).abs() < 1e-12, "{axis:?} ωτ={}", omega * tau);
                }
            }
        }
        // second pulse about y: |ϕ⟩ = (−i sin(ωτ/2), cos(ωτ/2))ᵀ
        let (omega, tau) = (0.4, 2.3);
        let s = ramsey_final_state(omega, tau, PulseAxis::Y);
        assert!(close(s.up, c64(0.0, -(0.5 * omega * tau).sin())));
        assert!(close(s.down, c64((0.5 * omega * tau).cos(), 0.0)));
        // at τ = 0: x pulse gives P = 1/2, y pulse (R_y(π)) flips to ↓z
        assert!((ramsey_p_up(omega, 0.0, PulseAxis::X) - 0.5).abs() < TOL);
        assert!(ramsey_p_up(omega, 0.0, PulseAxis::Y).abs() < TOL);
    }

    /// ω = 2μ_B B/ħ: about 28 MHz at 1 mT (g_s ≈ 2), splitting ħω ≈ 0.116 μeV.
    #[test]
    fn larmor_frequency() {
        let f = units::larmor_freq(1e-3);
        assert!((f / 1e6 - 27.9925).abs() < 0.001, "f = {f}");
        let w = units::omega_per_ns(1.0);
        assert!((w - 0.175_882).abs() < 1e-5, "ω = {w} rad/ns");
        assert!((units::b_mt_from_omega(w) - 1.0).abs() < 1e-12);
        assert!((units::splitting_ev(1e-3) / 1e-6 - 0.115_768).abs() < 1e-5);
    }

    #[test]
    fn angles_roundtrip() {
        for (theta, phi) in grid() {
            let (t, p) = angles(dir(theta, phi));
            let back = dir(t, p);
            let want = dir(theta, phi);
            for k in 0..3 {
                assert!((back[k] - want[k]).abs() < 1e-12);
            }
        }
    }
}
