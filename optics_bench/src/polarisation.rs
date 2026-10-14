//! Polarisation of a monochromatic plane wave (Saleh & Teich, Fundamentals of
//! Photonics, Sec. 6.1) and how polarisers, waveplates and rotators change it
//! (Jones calculus).
//!
//! The field is E(z, t) = Re[J e^{i(kz − ωt)}] with the Jones vector
//! J = (A_x, A_y), the convention of the problem sets (PS03, exercise 13).
//! Saleh & Teich write e^{j(ωt − kz)}, so their Jones vectors are the complex
//! conjugates of the ones here: their right circular (1, j)/√2 is (1, −i)/√2
//! here. Both call light right-handed when, at a fixed place, the field turns
//! clockwise as seen looking into the oncoming beam (x to the right, y up).
//!
//! The polarisation ellipse has the orientation ψ (the angle of its major axis
//! from x) and the ellipticity angle χ (tan|χ| = minor/major axis), positive
//! for right-handed light. On the Poincaré sphere the state is the point
//!
//! ```text
//! (u1, u2, u3) = (cos 2χ cos 2ψ, cos 2χ sin 2ψ, sin 2χ),
//! ```
//!
//! the Stokes parameters (S1, S2, S3)/S0 with S1 = |A_x|² − |A_y|²,
//! S2 = 2 Re(A_x* A_y) and, in this convention, S3 = 2 Im(A_x A_y*), so that
//! right-handed light is on the upper half as in Saleh & Teich's Fig. 6.1-5.

use std::f64::consts::PI;

use rustfft::num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

pub type Jones = [C64; 2];
pub type Mat = [[C64; 2]; 2];

const ZERO: C64 = C64::new(0.0, 0.0);
const ONE: C64 = C64::new(1.0, 0.0);

pub const DEG: f64 = PI / 180.0;

pub fn mat_mul(a: &Mat, b: &Mat) -> Mat {
    let mut m = [[ZERO; 2]; 2];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
    }
    m
}

pub fn apply(m: &Mat, v: &Jones) -> Jones {
    [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]]
}

pub fn identity() -> Mat {
    [[ONE, ZERO], [ZERO, ONE]]
}

/// rotation by `a` (radians) from x towards y
fn rot(a: f64) -> Mat {
    let (s, c) = a.sin_cos();
    [[C64::new(c, 0.0), C64::new(-s, 0.0)], [C64::new(s, 0.0), C64::new(c, 0.0)]]
}

/// R(a) diag(p, q) R(−a): acts as p along the axis at `a` and as q across it
fn along(a: f64, p: C64, q: C64) -> Mat {
    mat_mul(&mat_mul(&rot(a), &[[p, ZERO], [ZERO, q]]), &rot(-a))
}

pub fn intensity(j: &Jones) -> f64 {
    j[0].norm_sqr() + j[1].norm_sqr()
}

/// The Jones vector of unit intensity with the ellipse angles ψ and χ (radians).
pub fn from_angles(psi: f64, chi: f64) -> Jones {
    // along the major axis cos χ, across it −i sin χ (right-handed for χ > 0), turned by ψ
    let v = [C64::new(chi.cos(), 0.0), C64::new(0.0, -chi.sin())];
    apply(&rot(psi), &v)
}

/// Stokes parameters (S0, S1, S2, S3).
pub fn stokes(j: &Jones) -> [f64; 4] {
    let s0 = intensity(j);
    let s1 = j[0].norm_sqr() - j[1].norm_sqr();
    let s2 = 2.0 * (j[0].conj() * j[1]).re;
    let s3 = 2.0 * (j[0] * j[1].conj()).im;
    [s0, s1, s2, s3]
}

/// The point on the Poincaré sphere (u1, u2, u3).
pub fn sphere(j: &Jones) -> [f64; 3] {
    let s = stokes(j);
    if s[0] <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    [s[1] / s[0], s[2] / s[0], s[3] / s[0]]
}

/// The ellipse angles (ψ in (−π/2, π/2], χ in [−π/4, π/4]) of a state.
pub fn angles(j: &Jones) -> (f64, f64) {
    let u = sphere(j);
    let chi = 0.5 * u[2].clamp(-1.0, 1.0).asin();
    let mut psi = 0.5 * u[1].atan2(u[0]);
    if psi <= -0.5 * PI + 1e-12 {
        psi += PI;
    }
    (psi, chi)
}

/// the state of a point on the sphere (unit intensity)
pub fn from_sphere(u: [f64; 3]) -> Jones {
    let n = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt().max(1e-12);
    let chi = 0.5 * (u[2] / n).clamp(-1.0, 1.0).asin();
    let psi = 0.5 * u[1].atan2(u[0]);
    from_angles(psi, chi)
}

/// the field (E_x, E_y) at kz and ωt
pub fn field(j: &Jones, kz: f64, wt: f64) -> (f64, f64) {
    let e = C64::from_polar(1.0, kz - wt);
    ((j[0] * e).re, (j[1] * e).re)
}

/// amplitudes and the phase difference φ = φ_y − φ_x (radians)
pub fn amplitudes(j: &Jones) -> (f64, f64, f64) {
    let phi = if j[0].norm() > 1e-9 && j[1].norm() > 1e-9 { (j[1] * j[0].conj()).arg() } else { 0.0 };
    (j[0].norm(), j[1].norm(), phi)
}

/// the state with amplitudes a_x, a_y and phase difference φ = φ_y − φ_x
pub fn from_amplitudes(ax: f64, ay: f64, phi: f64) -> Jones {
    [C64::new(ax, 0.0), C64::from_polar(ay, phi)]
}

/// what kind of polarisation this is, in words
pub fn describe(j: &Jones) -> String {
    if intensity(j) < 1e-12 {
        return "no light".into();
    }
    let (psi, chi) = angles(j);
    let (pd, cd) = (psi / DEG, chi / DEG);
    if cd > 44.5 {
        "right circular ↻".into()
    } else if cd < -44.5 {
        "left circular ↺".into()
    } else if cd.abs() < 0.5 {
        if pd.abs() < 0.5 {
            "linear, horizontal (x)".into()
        } else if (pd.abs() - 90.0).abs() < 0.5 {
            "linear, vertical (y)".into()
        } else {
            format!("linear at {pd:.1}°")
        }
    } else {
        format!("{} elliptical, ψ = {pd:.1}°, χ = {cd:.1}°", if cd > 0.0 { "right ↻" } else { "left ↺" })
    }
}

// ---------------------------------------------------------------- elements

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Polariser,
    HalfWave,
    QuarterWave,
    /// a waveplate of any retardance
    Retarder,
    /// turns the polarisation by an angle (optical activity, Faraday rotator)
    Rotator,
}

impl Kind {
    pub const ALL: [Kind; 5] = [Kind::Polariser, Kind::HalfWave, Kind::QuarterWave, Kind::Retarder, Kind::Rotator];

    pub fn label(self) -> &'static str {
        match self {
            Kind::Polariser => "linear polariser",
            Kind::HalfWave => "half-wave plate (λ/2)",
            Kind::QuarterWave => "quarter-wave plate (λ/4)",
            Kind::Retarder => "waveplate (any retardance)",
            Kind::Rotator => "polarisation rotator",
        }
    }

    pub fn short(self) -> &'static str {
        match self {
            Kind::Polariser => "LP",
            Kind::HalfWave => "HWP",
            Kind::QuarterWave => "QWP",
            Kind::Retarder => "WP",
            Kind::Rotator => "rot",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Element {
    pub kind: Kind,
    /// transmission axis (polariser), fast axis (waveplates) or rotation angle (rotator), degrees from x
    pub angle_deg: f64,
    /// retardance Γ of a general waveplate (degrees)
    pub retard_deg: f64,
    pub on: bool,
}

impl Default for Element {
    fn default() -> Self {
        Element { kind: Kind::Polariser, angle_deg: 0.0, retard_deg: 90.0, on: true }
    }
}

impl Element {
    pub fn new(kind: Kind, angle_deg: f64) -> Element {
        Element { kind, angle_deg, ..Element::default() }
    }

    /// retardance Γ (radians) of a waveplate
    pub fn retardance(&self) -> Option<f64> {
        match self.kind {
            Kind::HalfWave => Some(PI),
            Kind::QuarterWave => Some(0.5 * PI),
            Kind::Retarder => Some(self.retard_deg * DEG),
            _ => None,
        }
    }

    /// The Jones matrix. A waveplate advances the light along its fast axis by
    /// Γ/2 and delays the light across it by Γ/2: R(θ) diag(e^{−iΓ/2}, e^{iΓ/2}) R(−θ).
    /// With the fast axis vertical, a λ/4 plate is e^{iπ/4} diag(1, −i), M_QWP of PS03.
    pub fn matrix(&self) -> Mat {
        if !self.on {
            return identity();
        }
        let a = self.angle_deg * DEG;
        match self.kind {
            Kind::Polariser => along(a, ONE, ZERO),
            Kind::Rotator => rot(a),
            _ => {
                let g = self.retardance().unwrap_or(0.0);
                along(a, C64::from_polar(1.0, -0.5 * g), C64::from_polar(1.0, 0.5 * g))
            }
        }
    }

    /// the same element doing only a fraction `f` of its work (for the path on the sphere)
    fn partial(&self, f: f64) -> Mat {
        match self.kind {
            Kind::Rotator => rot(f * self.angle_deg * DEG),
            Kind::Polariser => self.matrix(),
            _ => Element { kind: Kind::Retarder, retard_deg: f * self.retardance().unwrap_or(0.0) / DEG, ..self.clone() }.matrix(),
        }
    }

    pub fn label(&self) -> String {
        let a = fmt_angle(self.angle_deg);
        match self.kind {
            Kind::Polariser => format!("polariser, axis {a}"),
            Kind::HalfWave => format!("λ/2 plate, fast axis {a}"),
            Kind::QuarterWave => format!("λ/4 plate, fast axis {a}"),
            Kind::Retarder => format!("waveplate Γ = {}, fast axis {a}", fmt_angle(self.retard_deg)),
            Kind::Rotator => format!("rotator by {a}"),
        }
    }

    /// the way the state moves on the Poincaré sphere through this element
    pub fn sphere_path(&self, j: &Jones) -> Vec<[f64; 3]> {
        if !self.on {
            return vec![];
        }
        match self.kind {
            // a polariser projects: the state jumps to its axis (drawn straight)
            Kind::Polariser => vec![sphere(j), sphere(&apply(&self.matrix(), j))],
            // a waveplate turns the sphere about its fast axis by Γ, a rotator about u3 by 2ρ
            _ => (0..=48).map(|k| sphere(&apply(&self.partial(k as f64 / 48.0), j))).collect(),
        }
    }
}

pub fn fmt_angle(deg: f64) -> String {
    if (deg - deg.round()).abs() < 1e-6 { format!("{deg:.0}°") } else { format!("{deg:.1}°") }
}

// ---------------------------------------------------------------- numbers as in the notes

/// A real number as it would be written by hand: 0, ½, 1/√2, √3/2, … or decimals.
pub fn fmt_real(v: f64) -> String {
    if v.abs() < 5e-4 {
        return "0".into();
    }
    let sign = if v < 0.0 { "−" } else { "" };
    let a = v.abs();
    let named = [
        (1.0, "1"),
        (0.5, "½"),
        (std::f64::consts::FRAC_1_SQRT_2, "1/√2"),
        (0.75f64.sqrt(), "√3/2"),
        (0.75f64.sqrt() / 2.0, "√3/4"),
        (0.25, "¼"),
        (0.5 * std::f64::consts::FRAC_1_SQRT_2, "1/(2√2)"),
        (0.75, "¾"),
    ];
    for (x, s) in named {
        if (a - x).abs() < 5e-4 {
            return format!("{sign}{s}");
        }
    }
    format!("{sign}{a:.3}")
}

/// A complex number: "1/√2", "−i/2", "0.383 + 0.924i", …
pub fn fmt_complex(c: C64) -> String {
    let (re, im) = (c.re, c.im);
    let small = |x: f64| x.abs() < 5e-4;
    if small(im) {
        return fmt_real(re);
    }
    let imag = |x: f64| {
        let s = fmt_real(x.abs());
        match s.as_str() {
            "1" => "i".to_string(),
            "½" => "i/2".to_string(),
            "¼" => "i/4".to_string(),
            // 1/√2 → i/√2, √3/2 → i√3/2
            _ if s.starts_with("1/") => format!("i{}", &s[1..]),
            _ if s.starts_with('√') => format!("i{s}"),
            _ => format!("{s}i"),
        }
    };
    if small(re) {
        return format!("{}{}", if im < 0.0 { "−" } else { "" }, imag(im));
    }
    format!("{} {} {}", fmt_real(re), if im < 0.0 { "−" } else { "+" }, imag(im))
}

/// a phase factor e^{iα} as text ("" for α = 0)
pub fn fmt_phase(alpha: f64) -> String {
    let mut a = alpha.rem_euclid(2.0 * PI);
    if a > PI {
        a -= 2.0 * PI;
    }
    if a.abs() < 1e-6 {
        return String::new();
    }
    // multiples of π/8
    let q = a / (PI / 8.0);
    if (q - q.round()).abs() < 1e-6 {
        let n = q.round() as i64;
        let (num, den) = reduce(n, 8);
        let sign = if num < 0 { "−" } else { "" };
        let num = num.abs();
        let pi = if num == 1 { "π".to_string() } else { format!("{num}π") };
        return if den == 1 { format!("e^{{{sign}i{pi}}}") } else { format!("e^{{{sign}i{pi}/{den}}}") };
    }
    format!("e^{{{:.3}i}}", a)
}

fn reduce(n: i64, d: i64) -> (i64, i64) {
    let mut g = n.abs().max(1);
    let mut b = d;
    while b != 0 {
        let t = g % b;
        g = b;
        b = t;
    }
    (n / g, d / g)
}

/// The common phase α to take out of some entries so that e^{iα}(…) is shortest
/// to write, counting an i in an entry as three more characters: the phase of the
/// first sizeable entry, a multiple of π/8, or the phase of another entry (in
/// that order on ties).
fn best_phase(entries: &[C64]) -> f64 {
    let sizeable: Vec<C64> = entries.iter().copied().filter(|c| c.norm() > 1e-6).collect();
    let first = sizeable.first().map_or(0.0, |c| c.arg());
    let mut candidates = vec![first];
    candidates.extend((-7..=8).map(|k| k as f64 * PI / 8.0));
    candidates.extend(sizeable.iter().map(|c| c.arg()));
    let length = |alpha: f64| {
        let f = C64::from_polar(1.0, -alpha);
        let written = |c: C64| {
            let s = fmt_complex(c * f);
            s.chars().count() + if s.contains('i') { 3 } else { 0 }
        };
        fmt_phase(alpha).chars().count() + entries.iter().map(|&c| written(c)).sum::<usize>()
    };
    let mut best = (first, length(first));
    for a in candidates {
        let l = length(a);
        if l < best.1 {
            best = (a, l);
        }
    }
    best.0
}

/// A matrix written as e^{iα} M′, as simple as possible (see `best_phase`).
pub fn factor_phase(m: &Mat) -> (f64, Mat) {
    let alpha = best_phase(&[m[0][0], m[0][1], m[1][0], m[1][1]]);
    let f = C64::from_polar(1.0, -alpha);
    (alpha, [[m[0][0] * f, m[0][1] * f], [m[1][0] * f, m[1][1] * f]])
}

/// A vector written as e^{iα} J′, as simple as possible (see `best_phase`).
pub fn factor_phase_vec(j: &Jones) -> (f64, Jones) {
    let alpha = best_phase(j);
    let f = C64::from_polar(1.0, -alpha);
    (alpha, [j[0] * f, j[1] * f])
}

/// A vector with its global phase removed (first sizeable component real and positive).
pub fn normal_form(j: &Jones) -> Jones {
    let first = j.iter().find(|c| c.norm() > 1e-6).copied().unwrap_or(ONE);
    let f = C64::from_polar(1.0, -first.arg());
    [j[0] * f, j[1] * f]
}

// ---------------------------------------------------------------- parameters and examples

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PolParams {
    /// the light going in: the ellipse angles (degrees)
    pub psi_deg: f64,
    pub chi_deg: f64,
    pub elements: Vec<Element>,
    /// which state the x–y plane and the angles show: 0 = the input, k = after element k
    pub stage: usize,
    /// field components E_x and E_y in the 3D view
    pub components: bool,
    /// how many wavelengths the 3D view shows
    pub wavelengths: f64,
}

impl Default for PolParams {
    fn default() -> Self {
        PolPreset::Elliptical.setup().0
    }
}

impl PolParams {
    /// the input with J_x real and positive (or J_y if J_x = 0), as written by hand
    pub fn input(&self) -> Jones {
        normal_form(&from_angles(self.psi_deg * DEG, self.chi_deg * DEG))
    }

    pub fn set_input(&mut self, j: &Jones) {
        if intensity(j) < 1e-12 {
            return;
        }
        let (psi, chi) = angles(j);
        self.psi_deg = psi / DEG;
        self.chi_deg = chi / DEG;
    }

    /// the state before the first element and after each element
    pub fn states(&self) -> Vec<Jones> {
        let mut v = vec![self.input()];
        for e in &self.elements {
            let last = *v.last().expect("states");
            v.push(apply(&e.matrix(), &last));
        }
        v
    }

    /// the product of all matrices, last element first
    pub fn total(&self) -> Mat {
        self.elements.iter().fold(identity(), |m, e| mat_mul(&e.matrix(), &m))
    }

    pub fn shown_stage(&self) -> usize {
        self.stage.min(self.elements.len())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolPreset {
    Linear,
    RightCircular,
    LeftCircular,
    Elliptical,
    Polariser,
    QuarterWave,
    HalfWave,
    TwoQuarterWaves,
    ThreePolarisers,
    Exercise13,
    Exercise13Reversed,
    Rotator,
}

impl PolPreset {
    pub const ALL: [PolPreset; 12] = [
        PolPreset::Linear,
        PolPreset::RightCircular,
        PolPreset::LeftCircular,
        PolPreset::Elliptical,
        PolPreset::Polariser,
        PolPreset::QuarterWave,
        PolPreset::HalfWave,
        PolPreset::TwoQuarterWaves,
        PolPreset::ThreePolarisers,
        PolPreset::Exercise13,
        PolPreset::Exercise13Reversed,
        PolPreset::Rotator,
    ];

    pub fn label(self) -> &'static str {
        match self {
            PolPreset::Linear => "Linear polarisation",
            PolPreset::RightCircular => "Right circular polarisation",
            PolPreset::LeftCircular => "Left circular polarisation",
            PolPreset::Elliptical => "Elliptical polarisation: ψ, χ and the Poincaré sphere",
            PolPreset::Polariser => "Polariser: Malus's law",
            PolPreset::QuarterWave => "λ/4 plate: linear to circular",
            PolPreset::HalfWave => "λ/2 plate: turning linear polarisation",
            PolPreset::TwoQuarterWaves => "Two λ/4 plates make a λ/2 plate",
            PolPreset::ThreePolarisers => "Three polarisers: light through crossed polarisers",
            PolPreset::Exercise13 => "PS03 exercise 13 b: polariser, then λ/4 plate",
            PolPreset::Exercise13Reversed => "PS03 exercise 13 b: λ/4 plate, then polariser",
            PolPreset::Rotator => "Polarisation rotator",
        }
    }

    pub fn setup(self) -> (PolParams, String) {
        let mut p = PolParams { psi_deg: 0.0, chi_deg: 0.0, elements: vec![], stage: 99, components: false, wavelengths: 2.5 };
        let notes = match self {
            PolPreset::Linear => {
                p.psi_deg = 30.0;
                p.components = true;
                "Linear polarisation (Saleh & Teich 6.1): the two components E_x and E_y oscillate in phase \
                 (φ = φ_y − φ_x = 0 or π), so the field always points along one line, here at ψ = 30° to x. In \
                 the 3D view the field stays in one plane; in the x–y plane it swings back and forth along the \
                 line. The amplitudes are a_x = cos ψ and a_y = sin ψ. On the Poincaré sphere all linear \
                 polarisations lie on the equator (χ = 0), at the longitude 2ψ: horizontal and vertical are \
                 opposite points, because they are orthogonal."
            }
            PolPreset::RightCircular | PolPreset::LeftCircular => {
                let right = self == PolPreset::RightCircular;
                p.chi_deg = if right { 45.0 } else { -45.0 };
                p.components = true;
                if right {
                    "Right circular: equal amplitudes, E_y reaching its maximum a quarter period before E_x, \
                     J = (1, −i)/√2 (in the \
                     problem sets' convention e^{i(kz−ωt)}; Saleh & Teich write it (1, j)/√2). At a fixed place \
                     (the x–y plane, looking into the oncoming beam) the field turns clockwise. At a fixed time \
                     (the 3D view, paused) the tip of the field traces a right-handed helix along z, like an \
                     ordinary screw. As time goes on the helix slides forward without turning, which is why the two \
                     pictures turn opposite ways. On the Poincaré sphere it is the north pole (χ = +45°)."
                } else {
                    "Left circular: J = (1, i)/√2 (PS03 13 a 3; Saleh & Teich write (1, −j)/√2). Looking into the \
                     oncoming beam, the field turns counter-clockwise; at a fixed time it forms a left-handed \
                     helix. On the Poincaré sphere it is the south pole (χ = −45°). Right and left circular light \
                     are orthogonal: opposite points on the sphere, like H and V."
                }
            }
            PolPreset::Elliptical => {
                p.psi_deg = 30.0;
                p.chi_deg = 20.0;
                "The general case: an ellipse (Saleh & Teich Figs. 6.1-1 and 6.1-5). Its major axis makes the \
                 angle ψ with x, and tan χ = minor/major axis; the sign of χ says which way the field turns \
                 (here χ > 0: right-handed, clockwise looking into the beam). The same light can be described by \
                 the amplitudes a_x, a_y and the phase difference φ = φ_y − φ_x of the two components; the \
                 ellipse fits into the rectangle 2a_x × 2a_y. On the Poincaré sphere the state is the point at \
                 longitude 2ψ and latitude 2χ: the equator holds the linear states, the poles the circular ones. \
                 Drag the point on the sphere, or the ψ and χ knobs."
            }
            PolPreset::Polariser => {
                p.psi_deg = 0.0;
                p.elements = vec![Element::new(Kind::Polariser, 60.0)];
                "A linear polariser passes the component along its axis and absorbs the other: M = [[cos²θ, \
                 cos θ sin θ], [cos θ sin θ, sin²θ]] for the axis at θ. Horizontal light through a polariser at \
                 60° keeps cos²60° = 25 % of its power (Malus's law), and leaves polarised along 60°. On the \
                 Poincaré sphere the state jumps to the polariser's point on the equator, and the transmitted \
                 fraction is (1 + cos α)/2, where α is the angle between the two points on the sphere: it works \
                 for any input, also circular (50 %). Turn the polariser."
            }
            PolPreset::QuarterWave => {
                p.psi_deg = 45.0;
                p.elements = vec![Element::new(Kind::QuarterWave, 0.0)];
                "A λ/4 plate delays the light polarised across its fast axis by a quarter wavelength relative to \
                 the light along it. Linear light at 45° to the axes has equal parts of both; after the plate one \
                 is a quarter period behind: circular light. With the fast axis horizontal it is left circular, \
                 with the fast axis vertical right circular (try 90°). On the Poincaré sphere a waveplate turns \
                 the sphere about the point of its fast axis on the equator, by the retardance Γ = 90°: from the \
                 equator to a pole. Linear light along the axis (ψ = 0 or 90°) is not changed."
            }
            PolPreset::HalfWave => {
                p.psi_deg = 10.0;
                p.elements = vec![Element::new(Kind::HalfWave, 30.0)];
                "A λ/2 plate delays one component by half a wavelength: it mirrors linear polarisation at its \
                 fast axis. Light at 10° through a plate with its axis at 30° comes out at 50°: turning the plate \
                 by θ turns the polarisation by 2θ, which is how the polarisation is set in a lab. On the sphere \
                 the plate turns everything by 180° about its axis. Circular light changes its handedness (set \
                 χ = 45°)."
            }
            PolPreset::TwoQuarterWaves => {
                p.psi_deg = 10.0;
                p.elements = vec![Element::new(Kind::QuarterWave, 30.0), Element::new(Kind::QuarterWave, 30.0)];
                "Two λ/4 plates with parallel fast axes delay by λ/4 + λ/4 = λ/2: together they act as one λ/2 \
                 plate (compare the total matrix with the example 'λ/2 plate'). On the sphere: two quarter turns \
                 about the same axis. Turn the second plate by 90°: the delays cancel and the light comes out as it \
                 went in."
            }
            PolPreset::ThreePolarisers => {
                p.psi_deg = 0.0;
                p.elements = vec![Element::new(Kind::Polariser, 0.0), Element::new(Kind::Polariser, 45.0), Element::new(Kind::Polariser, 90.0)];
                "Crossed polarisers (0° and 90°) block everything. Put a third one at 45° between them and a \
                 quarter of the light gets through: cos²45° · cos²45° = ¼. The middle polariser does not add \
                 light; it changes the polarisation, so the last polariser finds a component along its axis. Switch \
                 the middle one off and on."
            }
            PolPreset::Exercise13 | PolPreset::Exercise13Reversed => {
                p.psi_deg = 90.0;
                let lp = Element::new(Kind::Polariser, 45.0);
                let qwp = Element::new(Kind::QuarterWave, 90.0);
                if self == PolPreset::Exercise13 {
                    p.elements = vec![lp, qwp];
                    "Problem set 3, exercise 13 b: vertical light (0, 1) through the polariser M_LP = ½[[1, 1], \
                     [1, 1]] (axis at 45°) and then the quarter-wave plate M_QWP = e^{iπ/4}[[1, 0], [0, −i]], whose \
                     fast axis is vertical in the convention e^{i(kz−ωt)}. The polariser gives (1, 1)/2, linear at \
                     45° with half the power; the plate turns it into e^{iπ/4}(1, −i)/2: right circular. Compare with \
                     the other order (next example)."
                } else {
                    p.elements = vec![qwp, lp];
                    "Problem set 3, exercise 13 b, the other order: the quarter-wave plate first. Vertical light \
                     lies along the plate's (fast) axis, so the plate only changes its phase. The polariser then \
                     gives linear light at 45° with half the power, e^{−iπ/4}(1, 1)/2. The order matters: the \
                     matrices do not commute (compare the total matrices)."
                }
            }
            PolPreset::Rotator => {
                p.psi_deg = 0.0;
                p.chi_deg = 20.0;
                p.elements = vec![Element::new(Kind::Rotator, 30.0)];
                "A polarisation rotator (optically active material such as a sugar solution or quartz along its \
                 axis, or a Faraday rotator) turns any polarisation ellipse by the same angle without changing its \
                 shape: M = [[cos ρ, −sin ρ], [sin ρ, cos ρ]]. On the Poincaré sphere it turns about the u3 axis \
                 by 2ρ, keeping the latitude 2χ."
            }
        };
        (p, notes.into())
    }
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &Jones, b: &Jones) -> bool {
        // equal up to a global phase
        let n = (a[0].conj() * b[0] + a[1].conj() * b[1]).norm();
        (n - intensity(a).sqrt() * intensity(b).sqrt()).abs() < 1e-9 && (intensity(a) - intensity(b)).abs() < 1e-9
    }

    #[test]
    fn angles_and_sphere_round_trip() {
        for psi in [-80.0, -45.0, 0.0, 30.0, 60.0, 90.0] {
            for chi in [-45.0, -20.0, 0.0, 10.0, 44.0] {
                let j = from_angles(psi * DEG, chi * DEG);
                assert!((intensity(&j) - 1.0).abs() < 1e-12);
                let u = sphere(&j);
                let want = [(2.0 * chi * DEG).cos() * (2.0 * psi * DEG).cos(), (2.0 * chi * DEG).cos() * (2.0 * psi * DEG).sin(), (2.0 * chi * DEG).sin()];
                for k in 0..3 {
                    assert!((u[k] - want[k]).abs() < 1e-12, "{psi} {chi}");
                }
                assert!(close(&from_sphere(u), &j));
                if chi.abs() < 44.9 {
                    let (p2, c2) = angles(&j);
                    let dpsi = (p2 / DEG - psi).rem_euclid(180.0);
                    assert!(!(1e-6..=180.0 - 1e-6).contains(&dpsi), "{psi} {p2}");
                    assert!((c2 / DEG - chi).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn handedness_in_time_and_space() {
        // right circular (1, −i)/√2: clockwise in time looking into the beam (x right, y up, z towards you)
        let r = from_angles(0.0, 45.0 * DEG);
        assert!(close(&r, &[C64::new(1.0, 0.0) / 2f64.sqrt(), C64::new(0.0, -1.0) / 2f64.sqrt()]));
        let turn = |a: (f64, f64), b: (f64, f64)| a.0 * b.1 - a.1 * b.0;
        let t0 = field(&r, 0.0, 0.0);
        let t1 = field(&r, 0.0, 0.1);
        assert!(turn(t0, t1) < 0.0, "clockwise in time");
        // at a fixed time the tip turns counter-clockwise going along +z: a right-handed helix
        let z1 = field(&r, 0.1, 0.0);
        assert!(turn(t0, z1) > 0.0, "right-handed helix in space");
        // E_x peaks at ωt = 0, E_y a quarter period earlier
        assert!((field(&r, 0.0, -0.5 * PI).1 * 2f64.sqrt() - 1.0).abs() < 1e-12);
        assert_eq!(describe(&r), "right circular ↻");
        // left circular is (1, i)/√2 as in PS03, with φ = φ_y − φ_x = +90° here
        let l = from_amplitudes(1.0 / 2f64.sqrt(), 1.0 / 2f64.sqrt(), 0.5 * PI);
        assert_eq!(describe(&l), "left circular ↺");
        assert!(sphere(&l)[2] < -0.999);
    }

    #[test]
    fn matrices_of_the_problem_set() {
        // M_LP at 45° = ½[[1, 1], [1, 1]]
        let lp = Element::new(Kind::Polariser, 45.0).matrix();
        for row in lp {
            for v in row {
                assert!((v - C64::new(0.5, 0.0)).norm() < 1e-12);
            }
        }
        // M_QWP = e^{iπ/4} diag(1, −i): a quarter-wave plate with its fast axis vertical
        let (alpha, m) = factor_phase(&Element::new(Kind::QuarterWave, 90.0).matrix());
        assert!((alpha - PI / 4.0).abs() < 1e-12);
        assert!((m[0][0] - ONE).norm() < 1e-12 && (m[1][1] - C64::new(0.0, -1.0)).norm() < 1e-12);
        assert!(m[0][1].norm() < 1e-12 && m[1][0].norm() < 1e-12);
        assert_eq!(fmt_phase(alpha), "e^{iπ/4}");
        assert_eq!(fmt_complex(m[1][1]), "−i");
    }

    #[test]
    fn phases_are_taken_out_where_they_simplify() {
        // λ/4 at 30°: not e^{−0.464i}[[0.791, …], …]
        let (alpha, m) = factor_phase(&Element::new(Kind::QuarterWave, 30.0).matrix());
        assert!(!fmt_phase(alpha).contains('.'), "{}", fmt_phase(alpha));
        assert!(fmt_complex(m[0][0]).contains("√2"));
        // λ/2 at 30°: e^{−iπ/2}[[½, √3/2], [√3/2, −½]] rather than [[−i/2, −i√3/2], …]
        let (alpha, m) = factor_phase(&Element::new(Kind::HalfWave, 30.0).matrix());
        assert_eq!((fmt_phase(alpha), fmt_complex(m[0][1])), ("e^{−iπ/2}".to_string(), "√3/2".to_string()));
        // two λ/4 plates at 30° turn linear 10° into e^{−iπ/2}(0.643, 0.766), not (−0.643i, −0.766i)
        let (p, _) = PolPreset::TwoQuarterWaves.setup();
        let (alpha, j) = factor_phase_vec(p.states().last().unwrap());
        assert_eq!((fmt_phase(alpha), fmt_complex(j[0])), ("e^{−iπ/2}".to_string(), "0.643".to_string()));
        // a polariser at 60°: [[¼, √3/4], [√3/4, ¾]]
        let (alpha, m) = factor_phase(&Element::new(Kind::Polariser, 60.0).matrix());
        assert_eq!((fmt_phase(alpha), fmt_complex(m[0][1])), (String::new(), "√3/4".to_string()));
        // λ/4 with the fast axis vertical: e^{iπ/4}[[1, 0], [0, −i]] (M_QWP of PS03)
        let (alpha, m) = factor_phase(&Element::new(Kind::QuarterWave, 90.0).matrix());
        assert_eq!((fmt_phase(alpha), fmt_complex(m[1][1])), ("e^{iπ/4}".to_string(), "−i".to_string()));
        // the output of exercise 13 b as in the solution: e^{iπ/4}(½, −i/2)
        let (p, _) = PolPreset::Exercise13.setup();
        let (alpha, j) = factor_phase_vec(p.states().last().unwrap());
        assert_eq!(fmt_phase(alpha), "e^{iπ/4}");
        assert_eq!((fmt_complex(j[0]), fmt_complex(j[1])), ("½".to_string(), "−i/2".to_string()));
    }

    #[test]
    fn exercise_13_b() {
        let (p, _) = PolPreset::Exercise13.setup();
        let out = *p.states().last().unwrap();
        assert_eq!(describe(&out), "right circular ↻");
        assert!((intensity(&out) - 0.5).abs() < 1e-12);
        // e^{iπ/4}(1, −i)/2
        let want = [C64::from_polar(0.5, PI / 4.0), C64::from_polar(0.5, PI / 4.0) * C64::new(0.0, -1.0)];
        assert!((out[0] - want[0]).norm() < 1e-12 && (out[1] - want[1]).norm() < 1e-12);
        let (q, _) = PolPreset::Exercise13Reversed.setup();
        let out = *q.states().last().unwrap();
        assert_eq!(describe(&out), "linear at 45.0°");
        let want = C64::from_polar(0.5, -PI / 4.0);
        assert!((out[0] - want).norm() < 1e-12 && (out[1] - want).norm() < 1e-12);
    }

    #[test]
    fn waveplates() {
        // λ/4 with the fast axis horizontal makes linear 45° left circular, vertical right circular
        let d = from_angles(45.0 * DEG, 0.0);
        assert_eq!(describe(&apply(&Element::new(Kind::QuarterWave, 0.0).matrix(), &d)), "left circular ↺");
        assert_eq!(describe(&apply(&Element::new(Kind::QuarterWave, 90.0).matrix(), &d)), "right circular ↻");
        // λ/2 mirrors linear light at its axis: 10° through 30° gives 50°
        let out = apply(&Element::new(Kind::HalfWave, 30.0).matrix(), &from_angles(10.0 * DEG, 0.0));
        assert_eq!(describe(&out), "linear at 50.0°");
        // and flips circular handedness
        let out = apply(&Element::new(Kind::HalfWave, 17.0).matrix(), &from_angles(0.0, 45.0 * DEG));
        assert_eq!(describe(&out), "left circular ↺");
        // two λ/4 plates = one λ/2 plate
        let (p, _) = PolPreset::TwoQuarterWaves.setup();
        let hw = Element::new(Kind::HalfWave, 30.0).matrix();
        let (a1, m1) = factor_phase(&p.total());
        let (a2, m2) = factor_phase(&hw);
        let _ = (a1, a2);
        for i in 0..2 {
            for k in 0..2 {
                assert!((m1[i][k] - m2[i][k]).norm() < 1e-12);
            }
        }
        // a waveplate turns the sphere about its fast axis by Γ
        for (theta, gamma) in [(0.0, 90.0), (30.0, 60.0), (-20.0, 180.0)] {
            let e = Element { kind: Kind::Retarder, angle_deg: theta, retard_deg: gamma, on: true };
            let j = from_angles(70.0 * DEG, 10.0 * DEG);
            let (a, b) = (sphere(&j), sphere(&apply(&e.matrix(), &j)));
            let axis = [(2.0 * theta * DEG).cos(), (2.0 * theta * DEG).sin(), 0.0];
            let dot = |u: [f64; 3], v: [f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
            // the same latitude about the axis, and the angle between the projections is Γ
            assert!((dot(a, axis) - dot(b, axis)).abs() < 1e-12);
            let pa = [a[0] - dot(a, axis) * axis[0], a[1] - dot(a, axis) * axis[1], a[2]];
            let pb = [b[0] - dot(b, axis) * axis[0], b[1] - dot(b, axis) * axis[1], b[2]];
            let ang = (dot(pa, pb) / (dot(pa, pa) * dot(pb, pb)).sqrt()).clamp(-1.0, 1.0).acos() / DEG;
            assert!((ang - gamma).abs() < 1e-6, "{theta} {gamma}: {ang}");
        }
    }

    #[test]
    fn malus_on_the_sphere() {
        for (psi, chi) in [(0.0, 0.0), (20.0, 10.0), (0.0, 45.0), (80.0, -30.0)] {
            let j = from_angles(psi * DEG, chi * DEG);
            for theta in [0.0, 33.0, 60.0, 90.0] {
                let out = apply(&Element::new(Kind::Polariser, theta).matrix(), &j);
                let u = sphere(&j);
                let p = [(2.0 * theta * DEG).cos(), (2.0 * theta * DEG).sin(), 0.0];
                let want = 0.5 * (1.0 + u[0] * p[0] + u[1] * p[1]);
                assert!((intensity(&out) - want).abs() < 1e-12);
            }
        }
        let (p, _) = PolPreset::ThreePolarisers.setup();
        assert!((intensity(p.states().last().unwrap()) - 0.25).abs() < 1e-12);
        let (q, _) = PolPreset::Polariser.setup();
        assert!((intensity(q.states().last().unwrap()) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn rotator_keeps_the_shape() {
        let (p, _) = PolPreset::Rotator.setup();
        let s = p.states();
        let (psi0, chi0) = angles(&s[0]);
        let (psi1, chi1) = angles(&s[1]);
        assert!((chi0 - chi1).abs() < 1e-12);
        assert!(((psi1 - psi0) / DEG - 30.0).abs() < 1e-9);
    }

    #[test]
    fn numbers_as_written_by_hand() {
        assert_eq!(fmt_real(std::f64::consts::FRAC_1_SQRT_2), "1/√2");
        assert_eq!(fmt_real(-0.5), "−½");
        assert_eq!(fmt_complex(C64::new(0.0, -std::f64::consts::FRAC_1_SQRT_2)), "−i/√2");
        assert_eq!(fmt_complex(C64::new(0.5, 0.5)), "½ + i/2");
        assert_eq!(fmt_complex(C64::new(0.3827, -0.9239)), "0.383 − 0.924i");
        assert_eq!(fmt_phase(-PI / 4.0), "e^{−iπ/4}");
        assert_eq!(fmt_phase(PI / 2.0), "e^{iπ/2}");
        assert_eq!(fmt_phase(0.0), "");
    }

    #[test]
    fn presets_survive_json() {
        for pr in PolPreset::ALL {
            let (p, notes) = pr.setup();
            assert!(!notes.is_empty());
            let back: PolParams = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
            assert_eq!(back.elements, p.elements, "{}", pr.label());
            assert!((back.psi_deg - p.psi_deg).abs() < 1e-12 && (back.chi_deg - p.chi_deg).abs() < 1e-12);
        }
    }
}
