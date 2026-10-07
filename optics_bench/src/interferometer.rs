//! Interferometers and cavities on an optical table: a square grid.
//!
//! Every part stands on a square and light runs along the grid lines, so
//! mirrors stand at 0° or 45° to the beam. At the reference frequency ν₀ every
//! square is a whole number of wavelengths long. The laser is detuned from ν₀
//! by Δν, so a beam picks up the phase
//!
//! ```text
//! φ = 2π Δν L / c
//! ```
//!
//! over a length L of table (the whole wavelengths drop out). Smaller path
//! changes come from phase shifters and from moving mirrors (a piezo).
//!
//! The field of a beam is a Jones vector (H, V): H lies in the plane of the
//! table, V is vertical, in the frame (h, v, k) with h = v × k. Mirrors reflect
//! H with +r and V with −r, like an ideal metal mirror (circular light changes
//! its handedness), and transmit with i·t. With real r and t this makes every
//! mirror and beam splitter lossless: the 2×2 matrix of each polarisation is
//! unitary when r² + t² = 1.
//!
//! **Steady state.** Each part maps the fields arriving on its four sides to
//! the fields leaving them. With the propagation phases this is a linear
//! system a = K a + s for all outgoing fields a, solved exactly: all round
//! trips of every cavity at once.
//!
//! **Switch-on.** The same parts, but the light moves one square per time
//! step τ = (square size)/c, so one sees the field fill the setup and build up
//! in a cavity round trip by round trip. In the frame rotating at ν₀ a square
//! is a pure delay, and the laser field turns as exp(−2πi Δν t).

use std::collections::HashMap;

use rustfft::num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

pub const C_LIGHT: f64 = 299_792_458.0;

/// a Jones vector (H, V); |H|² + |V|² is the power in mW
pub type Jones = [C64; 2];
pub type Mat = [[C64; 2]; 2];

const ZERO: C64 = C64::new(0.0, 0.0);
const ONE: C64 = C64::new(1.0, 0.0);
const I: C64 = C64::new(0.0, 1.0);

pub fn diag(a: C64, b: C64) -> Mat {
    [[a, ZERO], [ZERO, b]]
}

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

fn scale(m: &Mat, s: C64) -> Mat {
    [[m[0][0] * s, m[0][1] * s], [m[1][0] * s, m[1][1] * s]]
}

/// rotation by `a` (radians) from H towards V
fn rot(a: f64) -> Mat {
    let (s, c) = a.sin_cos();
    [[C64::new(c, 0.0), C64::new(-s, 0.0)], [C64::new(s, 0.0), C64::new(c, 0.0)]]
}

/// a matrix that acts along an axis at `a` (radians from H): R(a) diag(p, q) R(−a)
fn along(a: f64, p: C64, q: C64) -> Mat {
    mat_mul(&mat_mul(&rot(a), &diag(p, q)), &rot(-a))
}

pub fn power(v: &Jones) -> f64 {
    v[0].norm_sqr() + v[1].norm_sqr()
}

fn is_zero(m: &Mat) -> bool {
    m.iter().flatten().all(|v| v.norm_sqr() < 1e-30)
}

fn mixes(m: &Mat) -> bool {
    m[0][1].norm_sqr() > 1e-30 || m[1][0].norm_sqr() > 1e-30
}

/// the eigenvalues of a 2×2 matrix
fn eigenvalues(m: &Mat) -> [C64; 2] {
    let tr = m[0][0] + m[1][1];
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let root = (tr * tr - det * 4.0).sqrt();
    [(tr + root) * 0.5, (tr - root) * 0.5]
}

/// Jones vector of light polarised at `angle` (degrees from horizontal) with
/// ellipticity angle `ellip` (degrees; ±45 = circular), unit power
pub fn polarisation(angle_deg: f64, ellip_deg: f64) -> Jones {
    let (st, ct) = angle_deg.to_radians().sin_cos();
    let (sx, cx) = ellip_deg.to_radians().sin_cos();
    [C64::new(ct * cx, -st * sx), C64::new(st * cx, ct * sx)]
}

/// Orientation and shape of the polarisation ellipse of `v`: (angle of the
/// major axis from H in degrees, ellipticity angle in degrees; positive turns
/// counter-clockwise seen looking into the beam, with H to the right).
pub fn ellipse(v: &Jones) -> (f64, f64) {
    let p = power(v);
    if p <= 0.0 {
        return (0.0, 0.0);
    }
    let s1 = v[0].norm_sqr() - v[1].norm_sqr();
    let s2 = 2.0 * (v[0].conj() * v[1]).re;
    let s3 = 2.0 * (v[0].conj() * v[1]).im;
    let angle = 0.5 * s2.atan2(s1);
    let ellip = 0.5 * (s3 / p).clamp(-1.0, 1.0).asin();
    (angle.to_degrees(), ellip.to_degrees())
}

/// A short name for a polarisation state. Left and right as in PS03,
/// exercise 13: (1, i)/√2 is left circular, turning counter-clockwise when
/// one looks into the beam (↺).
pub fn describe_polarisation(v: &Jones) -> String {
    let (a, e) = ellipse(v);
    if e.abs() > 44.0 {
        return if e > 0.0 { "left circular ↺".into() } else { "right circular ↻".into() };
    }
    let a = if a < -0.5 { a + 180.0 } else { a };
    if e.abs() >= 1.0 {
        format!("{} elliptical, axis {a:.0}°", if e > 0.0 { "left ↺" } else { "right ↻" })
    } else if a.abs() < 1.0 || (a - 180.0).abs() < 1.0 {
        "linear H".to_string()
    } else if (a - 90.0).abs() < 1.0 {
        "linear V".to_string()
    } else {
        format!("linear {a:.0}°")
    }
}

// ---------------------------------------------------------------- the table

/// Directions on the table, counter-clockwise seen from above. On the grid,
/// x runs to the right and y down the screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dir {
    E,
    N,
    W,
    S,
}

impl Dir {
    pub const ALL: [Dir; 4] = [Dir::E, Dir::N, Dir::W, Dir::S];

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn from_index(i: usize) -> Dir {
        Dir::ALL[i % 4]
    }

    pub fn back(self) -> Dir {
        Dir::from_index(self.index() + 2)
    }

    /// one square on, in grid coordinates
    pub fn step(self) -> (i32, i32) {
        match self {
            Dir::E => (1, 0),
            Dir::N => (0, -1),
            Dir::W => (-1, 0),
            Dir::S => (0, 1),
        }
    }

    /// E and N are "forward" for parts that act on both ways alike
    pub fn forward(self) -> bool {
        matches!(self, Dir::E | Dir::N)
    }

    /// an arrow (only the monospace font has arrows)
    pub fn arrow(self) -> &'static str {
        match self {
            Dir::E => "→",
            Dir::N => "↑",
            Dir::W => "←",
            Dir::S => "↓",
        }
    }

    /// on the screen
    pub fn word(self) -> &'static str {
        match self {
            Dir::E => "right",
            Dir::N => "up",
            Dir::W => "left",
            Dir::S => "down",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Laser,
    /// mirrors, beam splitters and cavity mirrors: reflectivity R
    Mirror,
    Pbs,
    Waveplate,
    Polarizer,
    Phase,
    Faraday,
    Detector,
    Block,
}

/// Mirror surfaces in 45° steps (`Part::turn` mod 4), counter-clockwise.
pub const BAR: u8 = 0; // |
pub const BACK: u8 = 1; // \
pub const DASH: u8 = 2; // —
pub const SLASH: u8 = 3; // /

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Part {
    pub id: u32,
    pub kind: Kind,
    pub x: i32,
    pub y: i32,
    /// mirrors and PBS: the surface (BAR, BACK, DASH, SLASH);
    /// laser and Faraday rotator: the direction (Dir index)
    pub turn: u8,
    /// power reflectivity R (mirrors)
    pub reflect: f64,
    /// power absorbed or scattered per pass (mirrors)
    pub loss: f64,
    /// mirror moved along its normal (nm), e.g. by a piezo or a translation stage
    pub shift_nm: f64,
    /// retardance of a waveplate in waves (0.5 = λ/2, 0.25 = λ/4)
    pub retard: f64,
    /// fast axis (waveplate) or transmission axis (polariser), degrees from
    /// horizontal; rotation angle of a Faraday rotator
    pub angle_deg: f64,
    /// phase shifter: added phase (degrees)
    pub phase_deg: f64,
    /// shown next to the part
    pub name: String,
}

impl Default for Part {
    fn default() -> Self {
        Part {
            id: 0,
            kind: Kind::Mirror,
            x: 0,
            y: 0,
            turn: SLASH,
            reflect: 1.0,
            loss: 0.0,
            shift_nm: 0.0,
            retard: 0.5,
            angle_deg: 0.0,
            phase_deg: 0.0,
            name: String::new(),
        }
    }
}

/// which side of a mirror surface a beam travelling in `d` comes from (+1:
/// from the side its normal points to), and cos of the angle of incidence;
/// None if the beam runs along the surface
fn mirror_side(surface: u8, d: Dir) -> Option<(f64, f64)> {
    let h = std::f64::consts::FRAC_1_SQRT_2;
    // normals in table coordinates (x right, y up): | +x, \ (1,1), — +y, / (−1,1)
    let (nx, ny) = match surface % 4 {
        BAR => (1.0, 0.0),
        BACK => (h, h),
        DASH => (0.0, 1.0),
        _ => (-h, h),
    };
    let (kx, ky) = match d {
        Dir::E => (1.0, 0.0),
        Dir::N => (0.0, 1.0),
        Dir::W => (-1.0, 0.0),
        Dir::S => (0.0, -1.0),
    };
    let dot: f64 = kx * nx + ky * ny;
    if dot.abs() < 1e-9 {
        return None;
    }
    // a beam moving along +n arrives from the −n side
    Some((-dot.signum(), dot.abs()))
}

/// where a mirror surface sends a beam travelling in `d`
pub fn reflected(surface: u8, d: Dir) -> Option<Dir> {
    use Dir::*;
    Some(match (surface % 4, d) {
        (BAR, E) => W,
        (BAR, W) => E,
        (DASH, N) => S,
        (DASH, S) => N,
        (SLASH, E) => N,
        (SLASH, N) => E,
        (SLASH, W) => S,
        (SLASH, S) => W,
        (BACK, E) => S,
        (BACK, S) => E,
        (BACK, W) => N,
        (BACK, N) => W,
        _ => return None,
    })
}

/// One outgoing beam of a part: direction, Jones matrix, and the extra path
/// (m) from a moved mirror, whose phase 2πνΔ/c depends on the frequency.
#[derive(Clone, Copy, Debug)]
pub struct Out {
    pub dir: Dir,
    pub m: Mat,
    pub path: f64,
}

impl Part {
    pub fn new(kind: Kind, x: i32, y: i32, turn: u8) -> Part {
        let mut p = Part { kind, x, y, turn, ..Part::default() };
        match kind {
            Kind::Pbs => p.turn = if turn.is_multiple_of(2) { SLASH } else { turn % 4 },
            Kind::Waveplate => p.retard = 0.5,
            Kind::Faraday => p.angle_deg = 45.0,
            _ => {}
        }
        p
    }

    pub fn dir(&self) -> Dir {
        Dir::from_index(self.turn as usize)
    }

    pub fn transmit(&self) -> f64 {
        (1.0 - self.reflect - self.loss).max(0.0)
    }

    /// what the part does to light that arrives travelling in `d`; light that
    /// is not sent on is absorbed
    pub fn scatter(&self, d: Dir) -> Vec<Out> {
        let straight = |m: Mat| Out { dir: d, m, path: 0.0 };
        let backward = !d.forward();
        let axis = |deg: f64| if backward { -deg.to_radians() } else { deg.to_radians() };
        match self.kind {
            Kind::Laser | Kind::Detector | Kind::Block => vec![],
            Kind::Mirror => {
                let Some((side, cos)) = mirror_side(self.turn, d) else { return vec![] };
                let mut v = vec![];
                let r = self.reflect.clamp(0.0, 1.0).sqrt();
                let t = self.transmit().sqrt();
                if r > 0.0 {
                    let to = reflected(self.turn, d).expect("reflection");
                    // moving the mirror by s along its normal shortens the path on that side by 2 s cos θ
                    let path = -side * 2.0 * cos * self.shift_nm * 1e-9;
                    v.push(Out { dir: to, m: diag(C64::new(r, 0.0), C64::new(-r, 0.0)), path });
                }
                if t > 0.0 {
                    v.push(straight(diag(I * t, I * t)));
                }
                v
            }
            Kind::Pbs => {
                let Some(to) = reflected(self.turn, d) else { return vec![] };
                vec![Out { dir: to, m: diag(ZERO, -ONE), path: 0.0 }, straight(diag(I, ZERO))]
            }
            Kind::Waveplate => {
                let g = std::f64::consts::PI * self.retard;
                vec![straight(along(axis(self.angle_deg), C64::from_polar(1.0, -g), C64::from_polar(1.0, g)))]
            }
            Kind::Polarizer => vec![straight(along(axis(self.angle_deg), ONE, ZERO))],
            Kind::Phase => vec![straight(diag(C64::from_polar(1.0, self.phase_deg.to_radians()), C64::from_polar(1.0, self.phase_deg.to_radians())))],
            Kind::Faraday => {
                // rotation about the magnetic field: the same sense in the lab both ways
                let b = self.dir();
                let a = self.angle_deg.to_radians();
                if d == b {
                    vec![straight(rot(a))]
                } else if d == b.back() {
                    vec![straight(rot(-a))]
                } else {
                    vec![]
                }
            }
        }
    }

    /// the part's name, or what it is
    pub fn title(&self) -> String {
        if !self.name.is_empty() {
            return self.name.clone();
        }
        match self.kind {
            Kind::Laser => "laser",
            Kind::Mirror if self.reflect >= 0.9999 => "mirror",
            Kind::Mirror if self.turn % 2 == 1 => "beam splitter",
            Kind::Mirror => "mirror",
            Kind::Pbs => "PBS",
            Kind::Waveplate => "waveplate",
            Kind::Polarizer => "polariser",
            Kind::Phase => "phase shifter",
            Kind::Faraday => "Faraday rotator",
            Kind::Detector => "detector",
            Kind::Block => "beam block",
        }
        .into()
    }

    /// what the part is set to (shown on the table)
    pub fn label(&self) -> String {
        match self.kind {
            Kind::Laser => "laser".into(),
            Kind::Mirror => {
                if self.reflect >= 0.9999 {
                    "mirror".into()
                } else if self.turn % 2 == 1 {
                    format!("BS {:.0}:{:.0}", 100.0 * self.reflect, 100.0 * self.transmit())
                } else {
                    format!("R {}", fmt_percent(self.reflect))
                }
            }
            Kind::Pbs => "PBS".into(),
            Kind::Waveplate => format!("{} {:.0}°", retard_name(self.retard), self.angle_deg),
            Kind::Polarizer => format!("pol {:.0}°", self.angle_deg),
            Kind::Phase => format!("φ {:.0}°", self.phase_deg),
            Kind::Faraday => format!("Faraday {:.0}°", self.angle_deg),
            Kind::Detector => "detector".into(),
            Kind::Block => "block".into(),
        }
    }

    /// name of the swept setting, e.g. "M1 position"
    pub fn sweep_name(&self) -> String {
        let param = self.sweep_param().map_or("", |s| s.0);
        if self.name.is_empty() { format!("{}: {param}", self.title()) } else { format!("{} {param}", self.name) }
    }

    /// the parameter a sweep can scan: (name, unit, value)
    pub fn sweep_param(&self) -> Option<(&'static str, &'static str, f64)> {
        Some(match self.kind {
            Kind::Mirror => ("position", "nm", self.shift_nm),
            Kind::Phase => ("phase", "°", self.phase_deg),
            Kind::Waveplate | Kind::Polarizer => ("angle", "°", self.angle_deg),
            Kind::Faraday => ("rotation", "°", self.angle_deg),
            _ => return None,
        })
    }

    pub fn set_sweep_param(&mut self, v: f64) {
        match self.kind {
            Kind::Mirror => self.shift_nm = v,
            Kind::Phase => self.phase_deg = v,
            Kind::Waveplate | Kind::Polarizer | Kind::Faraday => self.angle_deg = v,
            _ => {}
        }
    }
}

pub fn retard_name(r: f64) -> String {
    for (v, s) in [(0.5, "λ/2"), (0.25, "λ/4"), (0.125, "λ/8"), (1.0, "λ")] {
        if (r - v).abs() < 1e-6 {
            return s.into();
        }
    }
    format!("{r:.3} λ")
}

pub fn fmt_percent(r: f64) -> String {
    let p = 100.0 * r;
    if (99.95..100.0).contains(&p) {
        format!("{p:.2} %")
    } else if (99.0..100.0).contains(&p) {
        format!("{p:.1} %")
    } else {
        format!("{p:.0} %")
    }
}

// ---------------------------------------------------------------- parameters

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Spectrum {
    /// one frequency (a single-mode laser)
    Single,
    /// two lines of equal strength, `line_sep_ghz` apart
    Two,
    /// a Gaussian band of full width `width_ghz` (at half maximum)
    Broad,
}

impl Spectrum {
    pub const ALL: [Spectrum; 3] = [Spectrum::Single, Spectrum::Two, Spectrum::Broad];
    pub fn label(self) -> &'static str {
        match self {
            Spectrum::Single => "single frequency",
            Spectrum::Two => "two lines",
            Spectrum::Broad => "broad band",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Laser {
    pub wavelength_nm: f64,
    pub power_mw: f64,
    /// from the reference frequency ν₀ at which every square holds a whole number of wavelengths
    pub detuning_mhz: f64,
    /// polarisation: axis (degrees from horizontal) and ellipticity (±45° = circular)
    pub pol_deg: f64,
    pub ellip_deg: f64,
    pub spectrum: Spectrum,
    pub line_sep_ghz: f64,
    pub width_ghz: f64,
}

impl Default for Laser {
    fn default() -> Self {
        Laser {
            wavelength_nm: 633.0,
            power_mw: 1.0,
            detuning_mhz: 0.0,
            pol_deg: 0.0,
            ellip_deg: 0.0,
            spectrum: Spectrum::Single,
            line_sep_ghz: 500.0,
            width_ghz: 100.0,
        }
    }
}

impl Laser {
    pub fn jones(&self) -> Jones {
        let a = self.power_mw.max(0.0).sqrt();
        let p = polarisation(self.pol_deg, self.ellip_deg);
        [p[0] * a, p[1] * a]
    }

    pub fn nu0(&self) -> f64 {
        C_LIGHT / (self.wavelength_nm * 1e-9)
    }

    /// the spectral lines (offset from the laser frequency in Hz, share of the power)
    pub fn lines(&self) -> Vec<(f64, f64)> {
        match self.spectrum {
            Spectrum::Single => vec![(0.0, 1.0)],
            Spectrum::Two => {
                let h = 0.5 * self.line_sep_ghz * 1e9;
                vec![(-h, 0.5), (h, 0.5)]
            }
            Spectrum::Broad => {
                let n = 41;
                let sigma = self.width_ghz * 1e9 / 2.3548;
                let w: Vec<(f64, f64)> = (0..n)
                    .map(|i| {
                        let x = (i as f64 / (n - 1) as f64 - 0.5) * 6.0 * sigma;
                        (x, (-0.5 * (x / sigma).powi(2)).exp())
                    })
                    .collect();
                let sum: f64 = w.iter().map(|p| p.1).sum();
                w.into_iter().map(|(x, p)| (x, p / sum)).collect()
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SweepVar {
    /// the laser frequency
    Frequency,
    /// the main parameter of the part with this id (mirror position, phase, angle)
    Part(u32),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Sweep {
    pub var: SweepVar,
    /// frequency sweep: from −span/2 to +span/2
    pub span_mhz: f64,
    /// sweep of a part's parameter: from `from` to `to` (its unit)
    pub from: f64,
    pub to: f64,
    /// the light that comes back into the laser as a trace
    pub show_back: bool,
    /// frequency axis in units of the free spectral range
    pub in_fsr: bool,
    pub log: bool,
}

impl Default for Sweep {
    fn default() -> Self {
        Sweep { var: SweepVar::Frequency, span_mhz: 3000.0, from: 0.0, to: 1000.0, show_back: true, in_fsr: false, log: false }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct IfoParams {
    pub parts: Vec<Part>,
    pub cols: i32,
    pub rows: i32,
    /// size of a square (mm)
    pub cell_mm: f64,
    pub laser: Laser,
    pub sweep: Sweep,
    /// draw the polarisation of the beams
    pub show_pol: bool,
    /// write the power next to the beams
    pub show_power: bool,
}

impl Default for IfoParams {
    fn default() -> Self {
        IfoPreset::FabryPerot.setup().0
    }
}

impl IfoParams {
    pub fn empty() -> Self {
        IfoParams {
            parts: vec![],
            cols: 28,
            rows: 16,
            cell_mm: 25.0,
            laser: Laser::default(),
            sweep: Sweep::default(),
            show_pol: false,
            show_power: false,
        }
    }

    pub fn cell_m(&self) -> f64 {
        self.cell_mm * 1e-3
    }

    pub fn at(&self, x: i32, y: i32) -> Option<usize> {
        self.parts.iter().position(|p| p.x == x && p.y == y)
    }

    pub fn by_id(&self, id: u32) -> Option<&Part> {
        self.parts.iter().find(|p| p.id == id)
    }

    pub fn next_id(&self) -> u32 {
        self.parts.iter().map(|p| p.id + 1).max().unwrap_or(1)
    }

    pub fn add(&mut self, mut p: Part) -> u32 {
        p.id = self.next_id();
        let id = p.id;
        self.parts.retain(|q| !(q.x == p.x && q.y == p.y));
        self.parts.push(p);
        id
    }

    pub fn inside(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && x < self.cols && y < self.rows
    }

    /// detectors in the order they were placed, numbered D1, D2, …
    pub fn detectors(&self) -> Vec<usize> {
        let mut d: Vec<usize> = (0..self.parts.len()).filter(|&i| self.parts[i].kind == Kind::Detector).collect();
        d.sort_by_key(|&i| self.parts[i].id);
        d
    }

    pub fn detector_name(&self, i: usize) -> String {
        let n = self.detectors().iter().position(|&j| j == i).map_or(0, |k| k + 1);
        let p = &self.parts[i];
        if p.name.is_empty() { format!("D{n}") } else { format!("D{n} {}", p.name) }
    }

    /// the sweep's parameter range and unit
    pub fn sweep_axis(&self) -> (f64, f64, &'static str) {
        match self.sweep.var {
            SweepVar::Frequency => (-0.5 * self.sweep.span_mhz, 0.5 * self.sweep.span_mhz, "MHz"),
            SweepVar::Part(id) => {
                let unit = self.by_id(id).and_then(|p| p.sweep_param()).map_or("", |s| s.1);
                (self.sweep.from, self.sweep.to, unit)
            }
        }
    }
}

// ---------------------------------------------------------------- network

/// A beam from one part to the next.
#[derive(Clone, Copy, Debug)]
pub struct Link {
    pub to: Option<usize>,
    /// squares from the part to the next one (to the edge of the table if None)
    pub cells: f64,
}

/// The table as a network of parts and beams.
pub struct Network {
    pub parts: Vec<Part>,
    /// per part and side: the beam leaving that side
    pub links: Vec<[Link; 4]>,
    pub laser: Option<usize>,
    pub cell_m: f64,
}

impl Network {
    pub fn new(p: &IfoParams) -> Network {
        let parts: Vec<Part> = p.parts.iter().filter(|q| p.inside(q.x, q.y)).cloned().collect();
        let mut at: HashMap<(i32, i32), usize> = HashMap::new();
        for (i, q) in parts.iter().enumerate() {
            at.insert((q.x, q.y), i);
        }
        let links = parts
            .iter()
            .map(|q| {
                Dir::ALL.map(|d| {
                    let (dx, dy) = d.step();
                    let (mut x, mut y) = (q.x, q.y);
                    let mut n = 0;
                    loop {
                        x += dx;
                        y += dy;
                        n += 1;
                        if !p.inside(x, y) {
                            break Link { to: None, cells: n as f64 - 0.5 };
                        }
                        if let Some(&j) = at.get(&(x, y)) {
                            break Link { to: Some(j), cells: n as f64 };
                        }
                    }
                })
            })
            .collect();
        let laser = parts.iter().position(|q| q.kind == Kind::Laser);
        Network { parts, links, laser, cell_m: p.cell_m() }
    }
}

/// One term of K: the field leaving `row` gets `m` times the field that left `col`
/// (and travelled `cells` squares), times exp(2πi ν path/c).
#[derive(Clone, Copy, Debug)]
struct Term {
    row: usize,
    col: usize,
    m: Mat,
    cells: f64,
    path: f64,
    /// part whose sweep parameter changes this term (a swept mirror or element)
    part: usize,
}

/// The linear system of the steady state, for the beams light can reach.
pub struct System {
    pub net: Network,
    /// (part, direction) of every beam that can carry light
    pub nodes: Vec<(usize, Dir)>,
    node_of: Vec<[Option<usize>; 4]>,
    terms: Vec<Term>,
    source: Option<(usize, Jones)>,
    /// no part mixes H and V, and the laser is H or V: two separate systems
    decoupled: bool,
    nu0: f64,
}

/// The steady state at one frequency.
pub struct Solution {
    /// field leaving every node
    pub a: Vec<Jones>,
}

impl System {
    pub fn new(p: &IfoParams) -> System {
        let net = Network::new(p);
        let mut node_of = vec![[None; 4]; net.parts.len()];
        let mut nodes = vec![];
        let mut source = None;
        let jones = p.laser.jones();
        if let Some(l) = net.laser {
            let d = net.parts[l].dir();
            node_of[l][d.index()] = Some(0);
            nodes.push((l, d));
            source = Some((0, jones));
        }
        // breadth first through the beams light can reach
        let mut terms = vec![];
        let mut k = 0;
        while k < nodes.len() {
            let (q, d) = nodes[k];
            if let Some(to) = net.links[q][d.index()].to {
                for out in net.parts[to].scatter(d) {
                    if is_zero(&out.m) {
                        continue;
                    }
                    let row = match node_of[to][out.dir.index()] {
                        Some(r) => r,
                        None => {
                            nodes.push((to, out.dir));
                            node_of[to][out.dir.index()] = Some(nodes.len() - 1);
                            nodes.len() - 1
                        }
                    };
                    terms.push(Term { row, col: k, m: out.m, cells: net.links[q][d.index()].cells, path: out.path, part: to });
                }
            }
            k += 1;
        }
        let pure = jones[0].norm_sqr() < 1e-30 || jones[1].norm_sqr() < 1e-30;
        let decoupled = pure && !terms.iter().any(|t| mixes(&t.m));
        System { net, nodes, node_of, terms, source, decoupled, nu0: p.laser.nu0() }
    }

    pub fn node(&self, part: usize, d: Dir) -> Option<usize> {
        self.node_of[part][d.index()]
    }

    /// Replaces the terms of part `i` after its parameters changed (sweeps).
    /// Keeps the structure: only valid if no beam appears or disappears.
    fn retune(&mut self, i: usize, part: &Part) {
        self.net.parts[i] = part.clone();
        let mut fresh: HashMap<(usize, usize), Out> = HashMap::new();
        for t in self.terms.iter().filter(|t| t.part == i) {
            let (_, d) = self.nodes[t.col];
            for out in part.scatter(d) {
                if let Some(r) = self.node_of[i][out.dir.index()] {
                    fresh.insert((t.col, r), out);
                }
            }
        }
        for t in self.terms.iter_mut().filter(|t| t.part == i) {
            match fresh.get(&(t.col, t.row)) {
                Some(out) => {
                    t.m = out.m;
                    t.path = out.path;
                    if mixes(&out.m) {
                        self.decoupled = false;
                    }
                }
                None => t.m = [[ZERO; 2]; 2],
            }
        }
    }

    /// the steady state for the laser at ν₀ + `detune` (Hz) with the field `jones`
    pub fn solve(&self, detune: f64, nu_abs: f64, jones: Jones) -> Solution {
        let n = self.nodes.len();
        let mut a = vec![[ZERO; 2]; n];
        let Some((src, _)) = self.source else { return Solution { a } };
        let k_cell = 2.0 * std::f64::consts::PI * detune * self.net.cell_m / C_LIGHT;
        let k_path = 2.0 * std::f64::consts::PI * nu_abs / C_LIGHT;
        // a tiny loss keeps a lossless trap (light that can never leave) finite on resonance
        let damp = 1.0 - 1e-12;
        let phase = |t: &Term| C64::from_polar(damp, k_cell * t.cells + k_path * t.path);
        let pols: &[usize] = if self.decoupled { &[0, 1] } else { &[2] };
        for &pol in pols {
            // pol 0 or 1: that component alone (n unknowns); 2: both (2n unknowns)
            let (m, comps): (usize, &[usize]) = match pol {
                0 => (n, &[0]),
                1 => (n, &[1]),
                _ => (2 * n, &[0, 1]),
            };
            if pol < 2 && jones[pol].norm_sqr() < 1e-30 {
                continue;
            }
            let idx = |node: usize, c: usize| if pol < 2 { node } else { 2 * node + c };
            let mut mat = vec![ZERO; m * m];
            let mut rhs = vec![ZERO; m];
            for i in 0..m {
                mat[i * m + i] = ONE;
            }
            for t in &self.terms {
                let f = phase(t);
                for &ci in comps {
                    for &cj in comps {
                        let v = t.m[ci][cj];
                        if v.norm_sqr() > 0.0 {
                            mat[idx(t.row, ci) * m + idx(t.col, cj)] -= v * f;
                        }
                    }
                }
            }
            for &c in comps {
                rhs[idx(src, c)] = jones[c];
            }
            solve_dense(&mut mat, &mut rhs, m);
            for (node, v) in a.iter_mut().enumerate() {
                for &c in comps {
                    v[c] = rhs[idx(node, c)];
                }
            }
        }
        Solution { a }
    }

    /// the field arriving at the end of beam `node` (it travelled `cells` squares)
    pub fn arriving(&self, s: &Solution, node: usize, detune: f64) -> Jones {
        let (q, d) = self.nodes[node];
        let cells = self.net.links[q][d.index()].cells;
        let f = C64::from_polar(1.0, 2.0 * std::f64::consts::PI * detune * self.net.cell_m * cells / C_LIGHT);
        [s.a[node][0] * f, s.a[node][1] * f]
    }

    /// power arriving at every part, summed over the sides
    pub fn arrived(&self, s: &Solution) -> Vec<f64> {
        let mut p = vec![0.0; self.net.parts.len()];
        for (k, &(q, d)) in self.nodes.iter().enumerate() {
            if let Some(to) = self.net.links[q][d.index()].to {
                p[to] += power(&s.a[k]);
            }
        }
        p
    }

    /// power that leaves the table
    pub fn escaped(&self, s: &Solution) -> f64 {
        self.nodes.iter().enumerate().filter(|(_, (q, d))| self.net.links[*q][d.index()].to.is_none()).map(|(k, _)| power(&s.a[k])).sum()
    }

    /// Closed loops of beams (cavities): the round trip with the highest gain.
    pub fn cavity(&self) -> Option<Cavity> {
        let n = self.nodes.len();
        let mut next: Vec<Vec<(usize, Mat, f64)>> = vec![vec![]; n];
        let k_path = 2.0 * std::f64::consts::PI * self.nu0 / C_LIGHT;
        for t in &self.terms {
            next[t.col].push((t.row, scale(&t.m, C64::from_polar(1.0, k_path * t.path)), t.cells));
        }
        let mut best: Option<Cavity> = None;
        let mut found = 0;
        // depth-first search for simple cycles that start at their lowest node
        for start in 0..n {
            let mut stack: Vec<(usize, usize)> = vec![(start, 0)];
            let mut on_path = vec![false; n];
            on_path[start] = true;
            let mut mats: Vec<(Mat, f64)> = vec![];
            while let Some(&(node, k)) = stack.last() {
                if found > 4000 {
                    break;
                }
                if k >= next[node].len() || stack.len() > 40 {
                    stack.pop();
                    on_path[node] = false;
                    mats.pop();
                    continue;
                }
                stack.last_mut().unwrap().1 += 1;
                let (to, m, cells) = next[node][k];
                if to == start {
                    found += 1;
                    let mut total = m;
                    let mut len = cells;
                    for (mm, c) in mats.iter().rev() {
                        total = mat_mul(&total, mm);
                        len += c;
                    }
                    let ev = eigenvalues(&total);
                    let gain = ev[0].norm_sqr().max(ev[1].norm_sqr());
                    let better = match &best {
                        None => true,
                        Some(b) => gain > b.gain + 1e-9 || ((gain - b.gain).abs() <= 1e-9 && len < b.cells),
                    };
                    if gain > 1e-6 && better {
                        best = Some(Cavity::new(ev, len, self.net.cell_m, gain));
                    }
                } else if to > start && !on_path[to] {
                    on_path[to] = true;
                    // the matrix of the step into `to`
                    mats.push((m, cells));
                    stack.push((to, 0));
                }
            }
        }
        best
    }
}

/// A closed loop of beams.
#[derive(Clone, Debug, PartialEq)]
pub struct Cavity {
    /// round trip in squares
    pub cells: f64,
    pub round_trip_m: f64,
    pub fsr_hz: f64,
    /// the round-trip gain of the field of each polarisation eigenmode
    pub modes: Vec<CavityMode>,
    gain: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CavityMode {
    /// power left after one round trip
    pub gain: f64,
    pub finesse: f64,
    /// detuning of a resonance in units of the FSR (0 … 1)
    pub offset_fsr: f64,
}

impl Cavity {
    fn new(ev: [C64; 2], cells: f64, cell_m: f64, gain: f64) -> Cavity {
        let round_trip_m = cells * cell_m;
        let mut modes: Vec<CavityMode> = vec![];
        for e in ev {
            let g = e.norm_sqr();
            if g < 1e-6 {
                continue;
            }
            let r = g.sqrt();
            let finesse = if r < 1.0 { std::f64::consts::PI * r.sqrt() / (1.0 - r) } else { f64::INFINITY };
            // resonance where the round-trip phase arg(e) + 2πΔν L/c is a multiple of 2π
            let offset_fsr = (-e.arg() / (2.0 * std::f64::consts::PI)).rem_euclid(1.0);
            let offset_fsr = if offset_fsr > 0.9999 { 0.0 } else { offset_fsr };
            if modes.iter().any(|m| (m.offset_fsr - offset_fsr).abs() < 1e-6 && (m.gain - g).abs() < 1e-9) {
                continue;
            }
            modes.push(CavityMode { gain: g, finesse, offset_fsr });
        }
        Cavity { cells, round_trip_m, fsr_hz: C_LIGHT / round_trip_m, modes, gain }
    }
}

/// complex Gaussian elimination with partial pivoting; the solution replaces `b`
fn solve_dense(a: &mut [C64], b: &mut [C64], n: usize) {
    for col in 0..n {
        let mut piv = col;
        let mut best = a[col * n + col].norm_sqr();
        for r in col + 1..n {
            let v = a[r * n + col].norm_sqr();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-300 {
            continue;
        }
        if piv != col {
            for c in 0..n {
                a.swap(col * n + c, piv * n + c);
            }
            b.swap(col, piv);
        }
        let inv = ONE / a[col * n + col];
        for r in col + 1..n {
            let f = a[r * n + col] * inv;
            if f.norm_sqr() == 0.0 {
                continue;
            }
            for c in col..n {
                let v = a[col * n + c];
                a[r * n + c] -= f * v;
            }
            let v = b[col];
            b[r] -= f * v;
        }
    }
    for col in (0..n).rev() {
        let mut s = b[col];
        for c in col + 1..n {
            s -= a[col * n + c] * b[c];
        }
        let d = a[col * n + col];
        b[col] = if d.norm_sqr() < 1e-300 { ZERO } else { s / d };
    }
}

// ---------------------------------------------------------------- steady state

/// The steady state shown on the table: every beam's field, summed over the
/// lines of the spectrum (powers add, fields are those of the main line).
pub struct Steady {
    /// per part and side: field and power leaving it
    pub field: Vec<[Jones; 4]>,
    pub power: Vec<[f64; 4]>,
    /// power arriving at every part (detectors, blocks; the laser: back-reflection)
    pub arrived: Vec<f64>,
    /// power that leaves the table at its edge
    pub escaped: f64,
    pub cavity: Option<Cavity>,
    /// field arriving at every part (main line), summed over its sides
    pub at_part: Vec<Jones>,
}

pub fn steady(p: &IfoParams) -> Steady {
    let sys = System::new(p);
    let np = sys.net.parts.len();
    let mut field = vec![[[ZERO; 2]; 4]; np];
    let mut pw = vec![[0.0; 4]; np];
    let mut arrived = vec![0.0; np];
    let mut escaped = 0.0;
    let mut at = vec![[ZERO; 2]; np];
    let nu0 = p.laser.nu0();
    let det = p.laser.detuning_mhz * 1e6;
    let jones = p.laser.jones();
    for (k, (off, share)) in p.laser.lines().into_iter().enumerate() {
        let s = share.sqrt();
        let sol = sys.solve(det + off, nu0 + det + off, [jones[0] * s, jones[1] * s]);
        for (n, &(q, d)) in sys.nodes.iter().enumerate() {
            pw[q][d.index()] += power(&sol.a[n]);
            if k == 0 {
                field[q][d.index()] = sol.a[n];
                if let Some(to) = sys.net.links[q][d.index()].to {
                    let v = sys.arriving(&sol, n, det + off);
                    at[to][0] += v[0];
                    at[to][1] += v[1];
                }
            }
        }
        for (a, b) in arrived.iter_mut().zip(sys.arrived(&sol)) {
            *a += b;
        }
        escaped += sys.escaped(&sol);
    }
    // parts not on the table are left out of the network: map back to the parameter indices
    let mut map_field = vec![[[ZERO; 2]; 4]; p.parts.len()];
    let mut map_power = vec![[0.0; 4]; p.parts.len()];
    let mut map_arrived = vec![0.0; p.parts.len()];
    let mut map_at = vec![[ZERO; 2]; p.parts.len()];
    for (i, q) in sys.net.parts.iter().enumerate() {
        if let Some(j) = p.parts.iter().position(|r| r.id == q.id) {
            map_field[j] = field[i];
            map_power[j] = pw[i];
            map_arrived[j] = arrived[i];
            map_at[j] = at[i];
        }
    }
    Steady {
        field: map_field,
        power: map_power,
        arrived: map_arrived,
        escaped,
        cavity: sys.cavity(),
        at_part: map_at,
    }
}

// ---------------------------------------------------------------- sweep

/// Detector powers (and the light back into the laser) as the laser frequency
/// or a part's parameter is swept.
#[derive(Clone, Debug, Default)]
pub struct SweepResult {
    pub xs: Vec<f64>,
    /// one trace per detector (in the order of `IfoParams::detectors`), then the back-reflection
    pub traces: Vec<Vec<f64>>,
    pub names: Vec<String>,
    pub millis: f32,
}

/// powers at the detectors and back into the laser for parameter value `x`
fn sweep_point(sys: &mut System, p: &IfoParams, swept: Option<usize>, x: f64, out: &mut [f64]) {
    let nu0 = p.laser.nu0();
    let mut det = p.laser.detuning_mhz * 1e6;
    match (p.sweep.var, swept) {
        (SweepVar::Frequency, _) => det = x * 1e6,
        (SweepVar::Part(_), Some(i)) => {
            let mut part = sys.net.parts[i].clone();
            part.set_sweep_param(x);
            sys.retune(i, &part);
        }
        _ => {}
    }
    out.iter_mut().for_each(|v| *v = 0.0);
    let jones = p.laser.jones();
    let dets: Vec<usize> = detector_order(&sys.net);
    for (off, share) in p.laser.lines() {
        let s = share.sqrt();
        let sol = sys.solve(det + off, nu0 + det + off, [jones[0] * s, jones[1] * s]);
        let arr = sys.arrived(&sol);
        for (k, &d) in dets.iter().enumerate() {
            out[k] += arr[d];
        }
        if let Some(l) = sys.net.laser {
            out[dets.len()] += arr[l];
        }
    }
}

fn detector_order(net: &Network) -> Vec<usize> {
    let mut d: Vec<usize> = (0..net.parts.len()).filter(|&i| net.parts[i].kind == Kind::Detector).collect();
    d.sort_by_key(|&i| net.parts[i].id);
    d
}

/// `budget_ms`: about how long it may take (big setups get fewer points)
pub fn sweep(p: &IfoParams, budget_ms: f64) -> SweepResult {
    #[cfg(not(target_arch = "wasm32"))]
    let t0 = std::time::Instant::now();
    #[cfg(target_arch = "wasm32")]
    let t0 = web_time::Instant::now();
    let mut sys = System::new(p);
    let dets = detector_order(&sys.net);
    let mut names: Vec<String> = dets
        .iter()
        .map(|&i| {
            let id = sys.net.parts[i].id;
            p.parts.iter().position(|q| q.id == id).map_or_else(|| "D".into(), |j| p.detector_name(j))
        })
        .collect();
    names.push("back into the laser".into());
    let nt = names.len();
    let swept = match p.sweep.var {
        SweepVar::Part(id) => sys.net.parts.iter().position(|q| q.id == id),
        SweepVar::Frequency => None,
    };
    let (a, b, _) = p.sweep_axis();
    if sys.net.laser.is_none() || (b - a).abs() < 1e-12 {
        return SweepResult { xs: vec![], traces: vec![vec![]; nt], names, millis: 0.0 };
    }
    // enough points for the fringes of a moved mirror (λ/2 per fringe), within a cost budget
    let mut n0 = 1601usize;
    if let (SweepVar::Part(_), Some(i)) = (p.sweep.var, swept)
        && sys.net.parts[i].kind == Kind::Mirror
    {
        let fringes = (b - a).abs() / (0.5 * p.laser.wavelength_nm);
        n0 = n0.max((fringes * 12.0) as usize);
    }
    // time a few points to see how many fit into the time budget
    let probe = t0.elapsed().as_secs_f64();
    for k in 0..8 {
        sweep_point(&mut sys, p, swept, a + (b - a) * k as f64 / 7.0, &mut vec![0.0; nt]);
    }
    let per_point = ((t0.elapsed().as_secs_f64() - probe) / 8.0).max(1e-7);
    let max_points = ((budget_ms * 1e-3 / per_point) as usize).clamp(201, 60_000);
    n0 = n0.min(max_points);
    let mut pts: Vec<(f64, Vec<f64>)> = Vec::with_capacity(n0);
    let mut buf = vec![0.0; nt];
    for i in 0..n0 {
        let x = a + (b - a) * i as f64 / (n0 - 1) as f64;
        sweep_point(&mut sys, p, swept, x, &mut buf);
        pts.push((x, buf.clone()));
    }
    // narrow peaks and dips: add points around every sharp extremum, twice
    let refine_budget = max_points.max(400);
    let mut added = 0;
    for _ in 0..2 {
        let mut extra: Vec<f64> = vec![];
        for t in 0..nt {
            let range = pts.iter().map(|q| q.1[t]).fold(0.0f64, f64::max);
            // a dark output (zero up to rounding) has no peaks
            if range <= 1e-9 * p.laser.power_mw {
                continue;
            }
            for i in 1..pts.len().saturating_sub(1) {
                let (y0, y1, y2) = (pts[i - 1].1[t], pts[i].1[t], pts[i + 1].1[t]);
                let peak = y1 > y0 && y1 >= y2;
                let dip = y1 < y0 && y1 <= y2;
                let sharp = (y1 - y0).abs().max((y1 - y2).abs()) > 0.03 * range;
                if (peak || dip) && sharp {
                    // the whole peak, down to its half width
                    let (xa, xb) = (pts[i.saturating_sub(4)].0, pts[(i + 4).min(pts.len() - 1)].0);
                    for k in 1..48 {
                        extra.push(xa + (xb - xa) * k as f64 / 48.0);
                    }
                }
            }
        }
        extra.sort_by(f64::total_cmp);
        extra.dedup_by(|x, y| (*x - *y).abs() < 1e-12 * (b - a).abs());
        if extra.is_empty() || added + extra.len() > refine_budget {
            break;
        }
        added += extra.len();
        for x in extra {
            sweep_point(&mut sys, p, swept, x, &mut buf);
            pts.push((x, buf.clone()));
        }
        pts.sort_by(|u, v| u.0.total_cmp(&v.0));
        pts.dedup_by(|u, v| (u.0 - v.0).abs() < 1e-12 * (b - a).abs());
    }
    let xs = pts.iter().map(|q| q.0).collect();
    let traces = (0..nt).map(|t| pts.iter().map(|q| q.1[t]).collect()).collect();
    SweepResult { xs, traces, names, millis: t0.elapsed().as_secs_f32() * 1000.0 }
}

/// Peaks of a trace: (x, height, full width at half maximum).
/// `floor`: traces below this (mW) are taken as dark.
pub fn peaks(xs: &[f64], ys: &[f64], floor: f64) -> Vec<(f64, f64, f64)> {
    let n = xs.len();
    if n < 3 {
        return vec![];
    }
    let max = ys.iter().cloned().fold(0.0f64, f64::max);
    let min = ys.iter().cloned().fold(f64::INFINITY, f64::min);
    if max <= floor || max - min < 0.05 * max {
        return vec![];
    }
    let mut out = vec![];
    for i in 1..n - 1 {
        if ys[i] > ys[i - 1] && ys[i] >= ys[i + 1] && ys[i] > min + 0.5 * (max - min) {
            let half = 0.5 * (ys[i] + min);
            // the half-maximum points on both sides, interpolated
            let mut l = i;
            while l > 0 && ys[l] > half {
                l -= 1;
            }
            let mut r = i;
            while r < n - 1 && ys[r] > half {
                r += 1;
            }
            if ys[l] > half || ys[r] > half {
                continue;
            }
            let xl = xs[l] + (half - ys[l]) / (ys[l + 1] - ys[l]) * (xs[l + 1] - xs[l]);
            let xr = xs[r - 1] + (ys[r - 1] - half) / (ys[r - 1] - ys[r]) * (xs[r] - xs[r - 1]);
            // the top of a parabola through the three highest points
            let (x0, x1, x2) = (xs[i - 1], xs[i], xs[i + 1]);
            let (y0, y1, y2) = (ys[i - 1], ys[i], ys[i + 1]);
            let den = (x0 - x1) * (x0 - x2) * (x1 - x2);
            let a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den;
            let b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / den;
            let top = if a < 0.0 && den.abs() > 0.0 { (-b / (2.0 * a)).clamp(x0, x2) } else { x1 };
            out.push((top, ys[i], xr - xl));
        }
    }
    out
}

// ---------------------------------------------------------------- switch-on in time

/// Light moving one square per step τ = (square size)/c, in the frame rotating at ν₀.
pub struct TimeSim {
    sys: System,
    /// per node: the fields that left it during the last `cells` steps (a ring buffer)
    buffers: Vec<Vec<Jones>>,
    lens: Vec<usize>,
    /// static phase of every term (moved mirrors), at ν₀
    term_phase: Vec<C64>,
    pub step: u64,
    pub laser_on: bool,
    /// when the laser was switched on or off last
    pub switched_at: u64,
    detune: f64,
    jones: Jones,
    pub tau: f64,
    /// per step: power at each detector, then back into the laser
    pub history: Vec<Vec<f32>>,
    /// per step: field arriving at the first detector, in the frame of the laser
    pub phasor: Vec<Jones>,
    pub phasor_part: Option<usize>,
    /// the parameters this was built for (parts and laser, not the view)
    pub built_for: IfoParams,
}

pub const HISTORY_MAX: usize = 200_000;

impl TimeSim {
    pub fn new(p: &IfoParams) -> TimeSim {
        let sys = System::new(p);
        let lens: Vec<usize> = sys
            .nodes
            .iter()
            .map(|&(q, d)| {
                let l = sys.net.links[q][d.index()];
                if l.to.is_some() { l.cells.round().max(1.0) as usize } else { l.cells.ceil().max(1.0) as usize }
            })
            .collect();
        let buffers = lens.iter().map(|&n| vec![[ZERO; 2]; n]).collect();
        let k_path = 2.0 * std::f64::consts::PI * p.laser.nu0() / C_LIGHT;
        let term_phase = sys.terms.iter().map(|t| C64::from_polar(1.0, k_path * t.path)).collect();
        let dets = detector_order(&sys.net);
        let phasor_part = dets.first().copied();
        TimeSim {
            buffers,
            lens,
            term_phase,
            step: 0,
            laser_on: true,
            switched_at: 0,
            detune: p.laser.detuning_mhz * 1e6,
            jones: p.laser.jones(),
            tau: p.cell_m() / C_LIGHT,
            history: vec![],
            phasor: vec![],
            phasor_part,
            sys,
            built_for: p.clone(),
        }
    }

    pub fn set_laser(&mut self, on: bool) {
        if on != self.laser_on {
            self.laser_on = on;
            self.switched_at = self.step;
        }
    }

    pub fn advance(&mut self, steps: usize) {
        let n = self.sys.nodes.len();
        let np = self.sys.net.parts.len();
        let dets = detector_order(&self.sys.net);
        let mut arriving = vec![[ZERO; 2]; n];
        let mut out = vec![[ZERO; 2]; n];
        for _ in 0..steps {
            let t = self.step;
            for (k, a) in arriving.iter_mut().enumerate() {
                *a = self.buffers[k][(t % self.lens[k] as u64) as usize];
            }
            out.iter_mut().for_each(|v| *v = [ZERO; 2]);
            for (term, ph) in self.sys.terms.iter().zip(&self.term_phase) {
                let v = apply(&term.m, &arriving[term.col]);
                out[term.row][0] += v[0] * ph;
                out[term.row][1] += v[1] * ph;
            }
            if self.laser_on
                && let Some((src, _)) = self.sys.source
            {
                let f = C64::from_polar(1.0, -2.0 * std::f64::consts::PI * self.detune * t as f64 * self.tau);
                out[src][0] += self.jones[0] * f;
                out[src][1] += self.jones[1] * f;
            }
            // powers arriving at the parts in this step
            let mut arr = vec![0.0f64; np];
            let mut at_probe = [ZERO; 2];
            for (k, &(q, d)) in self.sys.nodes.iter().enumerate() {
                if let Some(to) = self.sys.net.links[q][d.index()].to {
                    arr[to] += power(&arriving[k]);
                    if Some(to) == self.phasor_part {
                        at_probe[0] += arriving[k][0];
                        at_probe[1] += arriving[k][1];
                    }
                }
            }
            let mut row: Vec<f32> = dets.iter().map(|&d| arr[d] as f32).collect();
            row.push(self.sys.net.laser.map_or(0.0, |l| arr[l]) as f32);
            if self.history.len() < HISTORY_MAX {
                self.history.push(row);
                // undo the laser's own rotation, so the steady state stands still
                let f = C64::from_polar(1.0, 2.0 * std::f64::consts::PI * self.detune * t as f64 * self.tau);
                self.phasor.push([at_probe[0] * f, at_probe[1] * f]);
            }
            for ((buf, &len), v) in self.buffers.iter_mut().zip(&self.lens).zip(&out) {
                buf[(t % len as u64) as usize] = *v;
            }
            self.step += 1;
        }
    }

    /// The field along the beam leaving `part` (index in the network) towards `d`:
    /// one value per square, nearest first.
    pub fn beam(&self, part: usize, d: Dir) -> Option<Vec<Jones>> {
        let k = self.sys.node(part, d)?;
        let len = self.lens[k];
        // the value written j steps ago is j squares out (the newest was written in step `step − 1`)
        Some(
            (0..len)
                .map(|j| {
                    if self.step < j as u64 + 1 {
                        return [ZERO; 2];
                    }
                    let t = self.step - 1 - j as u64;
                    self.buffers[k][(t % len as u64) as usize]
                })
                .collect(),
        )
    }

    /// network index of a part with this id
    pub fn part_index(&self, id: u32) -> Option<usize> {
        self.sys.net.parts.iter().position(|q| q.id == id)
    }
}

// ---------------------------------------------------------------- examples

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IfoPreset {
    FabryPerot,
    HighFinesse,
    SwitchOn,
    Matching,
    RingCavity,
    PbsReflection,
    Michelson,
    MichelsonFrequency,
    TwoLines,
    MachZehnder,
    Sagnac,
    HalfWavePbs,
    Exercise13,
    Isolator,
    Birefringent,
    Coupled,
}

/// builds a table part by part
struct Table(IfoParams);

impl Table {
    fn new() -> Table {
        Table(IfoParams::empty())
    }
    fn put(&mut self, p: Part) -> u32 {
        self.0.add(p)
    }
    fn laser(&mut self, x: i32, y: i32, d: Dir) -> u32 {
        self.put(Part::new(Kind::Laser, x, y, d.index() as u8))
    }
    fn mirror(&mut self, x: i32, y: i32, surface: u8, r: f64) -> u32 {
        self.put(Part { reflect: r, ..Part::new(Kind::Mirror, x, y, surface) })
    }
    fn pbs(&mut self, x: i32, y: i32, surface: u8) -> u32 {
        self.put(Part::new(Kind::Pbs, x, y, surface))
    }
    fn plate(&mut self, x: i32, y: i32, retard: f64, angle: f64) -> u32 {
        self.put(Part { retard, angle_deg: angle, ..Part::new(Kind::Waveplate, x, y, 0) })
    }
    fn polarizer(&mut self, x: i32, y: i32, angle: f64) -> u32 {
        self.put(Part { angle_deg: angle, ..Part::new(Kind::Polarizer, x, y, 0) })
    }
    fn phase(&mut self, x: i32, y: i32, deg: f64) -> u32 {
        self.put(Part { phase_deg: deg, ..Part::new(Kind::Phase, x, y, 0) })
    }
    fn faraday(&mut self, x: i32, y: i32, d: Dir, deg: f64) -> u32 {
        self.put(Part { angle_deg: deg, ..Part::new(Kind::Faraday, x, y, d.index() as u8) })
    }
    fn detector(&mut self, x: i32, y: i32, name: &str) -> u32 {
        self.put(Part { name: name.into(), ..Part::new(Kind::Detector, x, y, 0) })
    }
    fn name(&mut self, id: u32, name: &str) -> u32 {
        if let Some(p) = self.0.parts.iter_mut().find(|p| p.id == id) {
            p.name = name.into();
        }
        id
    }
    fn sweep_part(&mut self, id: u32, from: f64, to: f64) {
        self.0.sweep.var = SweepVar::Part(id);
        self.0.sweep.from = from;
        self.0.sweep.to = to;
    }
}

impl IfoPreset {
    pub const ALL: [IfoPreset; 16] = [
        IfoPreset::FabryPerot,
        IfoPreset::HighFinesse,
        IfoPreset::SwitchOn,
        IfoPreset::Matching,
        IfoPreset::RingCavity,
        IfoPreset::PbsReflection,
        IfoPreset::Coupled,
        IfoPreset::Birefringent,
        IfoPreset::Michelson,
        IfoPreset::MichelsonFrequency,
        IfoPreset::TwoLines,
        IfoPreset::MachZehnder,
        IfoPreset::Sagnac,
        IfoPreset::HalfWavePbs,
        IfoPreset::Exercise13,
        IfoPreset::Isolator,
    ];

    pub fn label(self) -> &'static str {
        match self {
            IfoPreset::FabryPerot => "Fabry–Pérot cavity: resonances and FSR",
            IfoPreset::HighFinesse => "Better mirrors: higher finesse, more build-up",
            IfoPreset::SwitchOn => "Switch on: the field builds up round trip by round trip",
            IfoPreset::Matching => "Unequal mirrors: impedance matching",
            IfoPreset::RingCavity => "Ring cavity: FSR = c/L",
            IfoPreset::PbsReflection => "Measuring the reflection: PBS and λ/4",
            IfoPreset::Coupled => "Two coupled cavities: the resonances split",
            IfoPreset::Birefringent => "A waveplate in the cavity: two polarisation modes",
            IfoPreset::Michelson => "Michelson interferometer: λ/2 per fringe",
            IfoPreset::MichelsonFrequency => "Michelson with unequal arms: a frequency ruler",
            IfoPreset::TwoLines => "Michelson with two lines: resolving them",
            IfoPreset::MachZehnder => "Mach–Zehnder: a phase shifter, two outputs",
            IfoPreset::Sagnac => "Sagnac: the dark port stays dark",
            IfoPreset::HalfWavePbs => "λ/2 plate and PBS: an adjustable beam splitter",
            IfoPreset::Exercise13 => "PS03 exercise 13: polariser and λ/4 plate, both orders",
            IfoPreset::Isolator => "Optical isolator: Faraday rotator",
        }
    }

    pub fn setup(self) -> (IfoParams, String) {
        let mut t = Table::new();
        let notes = match self {
            IfoPreset::FabryPerot | IfoPreset::HighFinesse | IfoPreset::SwitchOn => {
                let r = match self {
                    IfoPreset::FabryPerot => 0.9,
                    IfoPreset::HighFinesse => 0.99,
                    _ => 0.95,
                };
                t.laser(2, 8, Dir::E);
                let m = t.mirror(9, 8, BAR, r);
                t.name(m, "M1");
                let m = t.mirror(17, 8, BAR, r);
                t.name(m, "M2");
                t.detector(23, 8, "transmitted");
                t.0.sweep.span_mhz = 3000.0;
                match self {
                    IfoPreset::FabryPerot => {
                        "Two mirrors with R = 90 %, 8 squares = 20 cm apart. Sweep the laser frequency: the \
                         transmission (D1) peaks whenever a whole number of half wavelengths fits between the mirrors, \
                         L = qλ/2. The peaks are c/2L = 750 MHz apart: the free spectral range (FSR). In between, \
                         almost all light goes back into the laser. On resonance the light leaking in through the first \
                         mirror adds up in phase with the light already inside, so the field builds up: the beam inside \
                         carries T/(1 − R)² = 10 times the laser power (drawn thicker). Drag the yellow frequency line \
                         in the plot through a peak and watch the cavity light up."
                    }
                    IfoPreset::HighFinesse => {
                        "The same cavity with R = 99 %. The peaks are still c/2L apart, but 10 times narrower: the \
                         finesse is F = FSR/linewidth = π√R/(1 − R) ≈ 310, and the light makes about F/2π ≈ 50 round \
                         trips before it is lost. On resonance the field inside builds up to T/(1 − R)² = 100 times the \
                         laser power, and the transmission is still 100 % (lossless, equal mirrors). Off resonance hardly \
                         anything gets in. Try 99.9 %, or give a mirror 0.5 % loss: the transmission peak drops."
                    }
                    _ => {
                        t.0.laser.detuning_mhz = 0.0;
                        "Press ▶ switch on: the laser starts at t = 0 and the light moves one square (83 ps) per \
                         step. Part of it leaks into the cavity and goes back and forth; on resonance each round trip \
                         adds in phase, so the field inside grows until the losses per round trip (the light leaking out \
                         through both mirrors) equal what comes in. That takes a few times the storage time \
                         τ = 2L/(c(1 − R²)) ≈ 13 ns here. The plot shows the transmitted power against time, and the \
                         field at D1 as a phasor: on resonance the round trips add along a line. Detune the laser by \
                         half a linewidth and switch on again: the phasors curl up and the field settles lower. \
                         Switch the laser off to see the ring-down."
                    }
                }
            }
            IfoPreset::Matching => {
                t.laser(2, 8, Dir::E);
                let m = t.mirror(9, 8, BAR, 0.95);
                t.name(m, "M1");
                let m = t.mirror(17, 8, BAR, 0.99);
                t.name(m, "M2");
                t.detector(23, 8, "transmitted");
                t.0.sweep.span_mhz = 2000.0;
                "Input mirror R₁ = 95 %, end mirror R₂ = 99 %. On resonance the reflection does not go to zero: \
                 the light reflected straight off the first mirror and the light leaking back out of the cavity have \
                 opposite phase, but different amplitudes. They cancel completely only when the input transmission \
                 equals all other losses of a round trip (impedance matching): here for R₂ = R₁. Set R₂ = 95 %. \
                 Then make 4 % of M2 loss (R₂ = 95 %, T₂ = 1 %): the dip still goes to zero, but now most of the \
                 light is absorbed in M2 and only 20 % reaches D1. With R₂ = 100 % all light comes back, but its \
                 phase still turns by 2π across the resonance."
            }
            IfoPreset::RingCavity => {
                t.laser(2, 4, Dir::E);
                for (x, y, surf, r, n) in [(8, 4, SLASH, 0.95, "M1"), (16, 4, BACK, 0.95, "M2"), (16, 10, SLASH, 1.0, "M3"), (8, 10, BACK, 1.0, "M4")] {
                    let m = t.mirror(x, y, surf, r);
                    t.name(m, n);
                }
                t.detector(23, 4, "transmitted");
                t.detector(8, 0, "reflected");
                t.0.sweep.span_mhz = 1500.0;
                "Four mirrors around a rectangle: 28 squares = 70 cm round trip. The light goes round one way only \
                 (a travelling wave), so a resonance needs the whole perimeter to hold a whole number of \
                 wavelengths: FSR = c/L = 428 MHz, without the factor 2 of a linear cavity. The reflected light \
                 leaves in a new direction (D2) instead of going back into the laser, which is why ring cavities are \
                 used to clean up laser beams. Two coupling mirrors with R = 95 %: equal, so the transmission \
                 reaches 100 % on resonance."
            }
            IfoPreset::PbsReflection => {
                t.0.show_pol = true;
                t.laser(2, 8, Dir::E);
                t.pbs(6, 8, SLASH);
                t.plate(9, 8, 0.25, 45.0);
                let m = t.mirror(13, 8, BAR, 0.95);
                t.name(m, "M1");
                let m = t.mirror(21, 8, BAR, 0.95);
                t.name(m, "M2");
                t.detector(25, 8, "transmitted");
                t.detector(6, 13, "reflected");
                t.0.sweep.span_mhz = 2000.0;
                "To measure what a cavity reflects without losing light at a 50:50 beam splitter: the laser is H \
                 polarised and passes the PBS. The λ/4 plate at 45° makes it circular. Reflection turns the \
                 handedness around, so on the way back the plate makes it V, and the PBS sends it to D2. The light \
                 never goes back into the laser. D2 shows the reflection dips of the cavity; D1 + D2 = laser power. \
                 Turn the plate away from 45° and some light leaks back to the laser."
            }
            IfoPreset::Coupled => {
                t.laser(2, 8, Dir::E);
                for (x, n) in [(7, "M1"), (14, "M2"), (21, "M3")] {
                    let m = t.mirror(x, 8, BAR, 0.9);
                    t.name(m, n);
                }
                t.detector(25, 8, "transmitted");
                t.0.sweep.span_mhz = 2500.0;
                "Three mirrors make two cavities of the same length, coupled through the middle mirror. Like two \
                 coupled pendulums, they have no common resonance at the frequency of each alone: every resonance \
                 splits into two, a symmetric and an antisymmetric mode. The weaker the coupling (the higher the \
                 middle mirror's R), the smaller the splitting. Set the middle mirror to 99 %, or to 0 % (one long \
                 cavity: twice the length, half the FSR)."
            }
            IfoPreset::Birefringent => {
                t.0.show_pol = true;
                t.laser(2, 8, Dir::E);
                let m = t.mirror(9, 8, BAR, 0.97);
                t.name(m, "M1");
                t.plate(13, 8, 0.125, 0.0);
                let m = t.mirror(17, 8, BAR, 0.97);
                t.name(m, "M2");
                t.pbs(21, 8, SLASH);
                t.detector(25, 8, "H");
                t.detector(21, 3, "V");
                t.0.laser.pol_deg = 45.0;
                t.0.sweep.span_mhz = 2000.0;
                "A λ/8 plate with its axis horizontal sits in the cavity. In a round trip the light passes it twice: \
                 V is delayed by λ/4 more than H. So H and V resonate at different frequencies, a quarter of an FSR \
                 apart: the cavity has two polarisation modes. The laser is polarised at 45°, so it excites both; the \
                 PBS behind the cavity separates them (D1: H, D2: V). Small birefringence in mirror coatings does \
                 the same in real cavities. Turn the plate to 45° and watch the modes."
            }
            IfoPreset::Michelson => {
                t.laser(2, 8, Dir::E);
                let bs = t.mirror(10, 8, SLASH, 0.5);
                t.name(bs, "BS");
                let m1 = t.mirror(18, 8, BAR, 1.0);
                t.name(m1, "M1");
                let m2 = t.mirror(10, 0, DASH, 1.0);
                t.name(m2, "M2");
                t.detector(10, 14, "output");
                t.sweep_part(m1, 0.0, 1300.0);
                "A Michelson interferometer with equal arms (8 squares). The sweep moves the end mirror M1 by up \
                 to 1.3 µm: when it moves by λ/2 = 316 nm, the light going there travels λ further, and the output \
                 goes once through dark and bright: one fringe. The light that does not reach the detector goes \
                 back into the laser: the two outputs add up to the laser power. Counting fringes measures \
                 distances in units of λ/2 (gravitational wave detectors are Michelsons with 4 km arms)."
            }
            IfoPreset::MichelsonFrequency => {
                t.laser(2, 8, Dir::E);
                let bs = t.mirror(10, 8, SLASH, 0.5);
                t.name(bs, "BS");
                let m1 = t.mirror(18, 8, BAR, 1.0);
                t.name(m1, "M1");
                let m2 = t.mirror(10, 4, DASH, 1.0);
                t.name(m2, "M2");
                t.detector(10, 14, "output");
                t.0.sweep.span_mhz = 6000.0;
                "Now the arms differ: 8 and 4 squares, so the light in one arm travels Δ = 2 × 10 cm = 20 cm \
                 further. Sweeping the laser frequency changes the phase difference 2πνΔ/c: one fringe every \
                 c/Δ = 1.5 GHz. A Michelson with unequal arms turns a frequency change into a power change. Move \
                 M2 closer to equal arms and the fringes get wider; at equal arms the output does not depend on \
                 the frequency at all, which is why white light only gives fringes near equal arms."
            }
            IfoPreset::TwoLines => {
                t.laser(2, 8, Dir::E);
                let bs = t.mirror(10, 8, SLASH, 0.5);
                t.name(bs, "BS");
                let m1 = t.mirror(18, 8, BAR, 1.0);
                t.name(m1, "M1");
                let m2 = t.mirror(10, 4, DASH, 1.0);
                t.name(m2, "M2");
                t.detector(10, 14, "output");
                t.0.laser.spectrum = Spectrum::Two;
                // c/2Δ for the path difference Δ = 20 cm: the two fringe patterns in antiphase
                t.0.laser.line_sep_ghz = C_LIGHT / 0.4 * 1e-9;
                t.sweep_part(m1, 0.0, 1300.0);
                "The source now has two lines 750 MHz apart, e.g. two modes of a laser. Each makes its own fringes \
                 and their powers add. With the path difference Δ = 20 cm of these arms, the phase difference of the \
                 two lines differs by 2π·Δ·δν/c = π: where one is bright the other is dark, and the fringes vanish. \
                 Change the arm length (move M2) and the fringes come back. The fringe contrast goes through zero \
                 every time Δ grows by c/δν = 40 cm: to tell two lines δν apart, the path difference must reach \
                 about c/δν. The resolution is the inverse of the largest path difference, just as for a grating \
                 (Nd) or a Fabry–Pérot."
            }
            IfoPreset::MachZehnder => {
                t.laser(2, 8, Dir::E);
                let b = t.mirror(6, 8, SLASH, 0.5);
                t.name(b, "BS1");
                t.mirror(6, 3, SLASH, 1.0);
                t.mirror(14, 8, SLASH, 1.0);
                let b = t.mirror(14, 3, SLASH, 0.5);
                t.name(b, "BS2");
                let ph = t.phase(10, 3, 0.0);
                t.detector(20, 3, "out 1");
                t.detector(14, 0, "out 2");
                t.sweep_part(ph, 0.0, 720.0);
                "Two beam splitters, two paths of equal length. The light splits at the first and meets again at the \
                 second, where the two paths interfere. The sweep turns the phase shifter in the upper arm from 0° \
                 to 720°: the light moves from one output to the other and back. Both outputs always add up to the \
                 laser power: what is missing at one appears at the other. The Mach–Zehnder is how phase is \
                 measured, e.g. of a gas cell in one arm, or in fibre-optic modulators."
            }
            IfoPreset::Sagnac => {
                t.0.show_pol = true;
                t.laser(2, 8, Dir::E);
                t.mirror(6, 8, SLASH, 0.5);
                t.mirror(14, 8, SLASH, 1.0);
                t.mirror(14, 3, BACK, 1.0);
                t.mirror(6, 3, SLASH, 1.0);
                let ph = t.phase(10, 3, 0.0);
                t.faraday(10, 8, Dir::E, 0.0);
                t.detector(6, 13, "output");
                t.sweep_part(ph, 0.0, 720.0);
                "A Sagnac interferometer: the beam splitter sends the light around a loop both ways, and the two \
                 beams meet again at the beam splitter. They travel the same path, so the output stays dark \
                 whatever the phase shifter in the loop does (the sweep turns it) and whatever the laser frequency: \
                 all light goes back into the laser. Only something that treats the two directions differently \
                 breaks this: turn up the Faraday rotator (rotation is in the same sense in the lab both ways), or \
                 rotate the whole loop (the Sagnac effect of fibre gyroscopes, not simulated)."
            }
            IfoPreset::HalfWavePbs => {
                t.0.show_pol = true;
                t.laser(2, 8, Dir::E);
                let hw = t.plate(6, 8, 0.5, 0.0);
                t.pbs(11, 8, SLASH);
                t.detector(17, 8, "transmitted (H)");
                t.detector(11, 3, "reflected (V)");
                t.sweep_part(hw, 0.0, 90.0);
                "A λ/2 plate turns linear polarisation by twice the angle of its axis. The PBS passes H and reflects \
                 V, so the transmitted power is cos²(2θ) of the laser power: the sweep turns the plate from 0° to \
                 90°. At 22.5° the PBS is a 50:50 beam splitter. This is how the power in the arms of a setup is \
                 set in the lab. Replace the λ/2 plate by a polariser and you get Malus's law, cos²θ, but the light \
                 that does not pass is lost."
            }
            IfoPreset::Exercise13 => {
                t.0.show_pol = true;
                t.0.laser.pol_deg = 90.0;
                t.laser(2, 8, Dir::E);
                t.mirror(6, 8, SLASH, 0.5);
                t.mirror(6, 3, SLASH, 1.0);
                t.polarizer(10, 8, 45.0);
                t.plate(14, 8, 0.25, 90.0);
                t.detector(20, 8, "LP, then QWP");
                t.plate(10, 3, 0.25, 90.0);
                t.polarizer(14, 3, 45.0);
                t.detector(20, 3, "QWP, then LP");
                "Problem set 3, exercise 13 b): vertically polarised light (laser V) through a linear polariser \
                 at 45° (M_LP) and a quarter-wave plate (M_QWP = e^{iπ/4} diag(1, −i): its fast axis is vertical), \
                 and the other way round. The beam splitter sends half of the light each way (V stays V). Bottom \
                 row: the polariser makes (1, 1)/√2, the plate turns it into (1, −i)/√2, right circular. Top row: \
                 the plate first leaves V unchanged (V lies along its axis), and the polariser then gives linear \
                 45°. Hover the beams for their polarisation; the circles show it looking into the beam. Turn the \
                 plate's axis to 0°: the light becomes left circular, (1, i)/√2. At 45° (along the polariser) it \
                 stays linear."
            }
            IfoPreset::Isolator => {
                t.0.show_pol = true;
                t.laser(2, 8, Dir::E);
                t.polarizer(6, 8, 0.0);
                t.faraday(9, 8, Dir::E, 45.0);
                t.polarizer(12, 8, 45.0);
                t.mirror(18, 8, BAR, 0.3);
                t.detector(23, 8, "out");
                "Light reflected back into a laser disturbs it. An isolator lets light through one way only. The \
                 Faraday rotator turns the polarisation by 45° in the same sense (seen in the lab) both ways, \
                 because the rotation follows the magnetic field, not the beam. Going out it turns from 0° to 45° and passes the \
                 second polariser. Coming back it turns on from 45° to 90° and the first polariser blocks it. Nothing gets back into \
                 the laser even though a mirror reflects 30 %. Set the rotation to 0° and compare. A λ/4 plate cannot \
                 do this here: it is reciprocal."
            }
        };
        if t.0.sweep.var == SweepVar::Frequency {
            t.0.sweep.span_mhz = t.0.sweep.span_mhz.max(500.0);
        }
        (t.0, notes.into())
    }
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    fn steady_at(p: &IfoParams, mhz: f64) -> Steady {
        let mut q = p.clone();
        q.laser.detuning_mhz = mhz;
        steady(&q)
    }

    fn det(p: &IfoParams, s: &Steady, k: usize) -> f64 {
        s.arrived[p.detectors()[k]]
    }

    fn back(p: &IfoParams, s: &Steady) -> f64 {
        s.arrived[p.parts.iter().position(|q| q.kind == Kind::Laser).unwrap()]
    }

    /// everything the laser sends out ends up somewhere
    fn energy_balance(p: &IfoParams, s: &Steady) -> f64 {
        let sinks: f64 = p
            .parts
            .iter()
            .enumerate()
            .filter(|(_, q)| matches!(q.kind, Kind::Detector | Kind::Block | Kind::Laser))
            .map(|(i, _)| s.arrived[i])
            .sum();
        sinks + s.escaped
    }

    fn fp(r1: f64, r2: f64) -> IfoParams {
        let mut t = Table::new();
        t.laser(2, 8, Dir::E);
        t.mirror(9, 8, BAR, r1);
        t.mirror(17, 8, BAR, r2);
        t.detector(23, 8, "T");
        t.0
    }

    #[test]
    fn fabry_perot_matches_airy() {
        for (r1, r2) in [(0.9, 0.9), (0.95, 0.99), (0.5, 0.8)] {
            let p = fp(r1, r2);
            let l = 8.0 * p.cell_m();
            let fsr = C_LIGHT / (2.0 * l);
            for f in [0.0, 0.01, 0.1, 0.37, 0.5] {
                let s = steady_at(&p, f * fsr * 1e-6);
                let rr = (r1 * r2).sqrt();
                let delta = 2.0 * std::f64::consts::PI * f;
                let t_airy = (1.0 - r1) * (1.0 - r2) / ((1.0 - rr).powi(2) + 4.0 * rr * (delta / 2.0).sin().powi(2));
                let t = det(&p, &s, 0);
                assert!((t - t_airy).abs() < 1e-9, "R={r1},{r2} f={f}: {t} vs {t_airy}");
                // lossless: what is not transmitted comes back
                assert!((t + back(&p, &s) - 1.0).abs() < 1e-9);
                assert!((energy_balance(&p, &s) - 1.0).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn build_up_inside_the_cavity() {
        let p = fp(0.99, 0.99);
        let s = steady(&p);
        let m1 = 1;
        // forward beam inside: T/(1 − R)² = 100 times the laser power
        let inside = s.power[m1][Dir::E.index()];
        assert!((inside - 100.0).abs() < 1e-6, "{inside}");
    }

    #[test]
    fn cavity_is_found_with_fsr_and_finesse() {
        let p = fp(0.99, 0.99);
        let c = steady(&p).cavity.unwrap();
        assert!((c.round_trip_m - 0.4).abs() < 1e-12);
        assert!((c.fsr_hz - C_LIGHT / 0.4).abs() < 1.0);
        let f = c.modes[0].finesse;
        assert!((f - std::f64::consts::PI * 0.99f64.sqrt() / 0.01).abs() < 1e-6, "{f}");
        assert!(c.modes.iter().all(|m| m.offset_fsr.abs() < 1e-9));
    }

    #[test]
    fn sweep_finds_fsr_and_linewidth() {
        let (mut p, _) = IfoPreset::FabryPerot.setup();
        p.sweep.span_mhz = 2000.0;
        let r = sweep(&p, 100.0);
        let pk = peaks(&r.xs, &r.traces[0], 1e-9);
        assert!(pk.len() >= 2, "{pk:?}");
        let fsr = (pk[1].0 - pk[0].0) * 1e6;
        assert!((fsr - C_LIGHT / 0.4).abs() / fsr < 1e-3, "{fsr}");
        let r_ = 0.9f64;
        let finesse = std::f64::consts::PI * r_.sqrt() / (1.0 - r_);
        let fwhm = pk[0].2 * 1e6;
        // the Airy peak is close to a Lorentzian of width FSR/F
        assert!((fsr / fwhm - finesse).abs() / finesse < 0.02, "{} vs {finesse}", fsr / fwhm);
    }

    #[test]
    fn ring_cavity_has_fsr_c_over_l_and_no_back_reflection() {
        let (p, _) = IfoPreset::RingCavity.setup();
        let c = steady(&p).cavity.unwrap();
        assert!((c.round_trip_m - 28.0 * 0.025).abs() < 1e-12);
        for f in [0.0, 0.3] {
            let s = steady_at(&p, f * c.fsr_hz * 1e-6);
            assert!(back(&p, &s) < 1e-12);
            assert!((energy_balance(&p, &s) - 1.0).abs() < 1e-9);
        }
        // impedance matched: full transmission on resonance
        let s = steady(&p);
        assert!((det(&p, &s, 0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn michelson_outputs_are_complementary() {
        let (p, _) = IfoPreset::Michelson.setup();
        let lambda = p.laser.wavelength_nm;
        let mut seen = vec![];
        for k in 0..8 {
            let mut q = p.clone();
            q.parts[2].shift_nm = k as f64 * lambda / 16.0;
            let s = steady(&q);
            let d = det(&q, &s, 0);
            assert!((d + back(&q, &s) - 1.0).abs() < 1e-9);
            seen.push(d);
        }
        // a quarter wave of mirror travel (half a fringe) takes bright to dark
        assert!((seen[0] - (1.0 - seen[4])).abs() < 1e-9, "{seen:?}");
        let max = seen.iter().cloned().fold(0.0, f64::max);
        assert!(max > 0.999);
    }

    #[test]
    fn michelson_fringe_period_in_frequency() {
        let (p, _) = IfoPreset::MichelsonFrequency.setup();
        // path difference 20 cm: c/Δ = 1.499 GHz
        let period = C_LIGHT / 0.2 * 1e-6;
        let a = det(&p, &steady_at(&p, 100.0), 0);
        let b = det(&p, &steady_at(&p, 100.0 + period), 0);
        let c = det(&p, &steady_at(&p, 100.0 + 0.5 * period), 0);
        assert!((a - b).abs() < 1e-9);
        assert!((a - (1.0 - c)).abs() < 1e-9);
    }

    #[test]
    fn two_lines_wash_out_the_fringes() {
        let (p, _) = IfoPreset::TwoLines.setup();
        // 750 MHz apart, 20 cm path difference: the two fringe patterns are in antiphase
        for k in 0..5 {
            let mut q = p.clone();
            q.parts[2].shift_nm = k as f64 * 40.0;
            let d = det(&q, &steady(&q), 0);
            assert!((d - 0.5).abs() < 1e-4, "{d}");
        }
    }

    #[test]
    fn mach_zehnder_and_sagnac() {
        let (p, _) = IfoPreset::MachZehnder.setup();
        for ph in [0.0, 45.0, 90.0, 180.0] {
            let mut q = p.clone();
            q.parts.iter_mut().find(|x| x.kind == Kind::Phase).unwrap().phase_deg = ph;
            let s = steady(&q);
            let (a, b) = (det(&q, &s, 0), det(&q, &s, 1));
            assert!((a + b - 1.0).abs() < 1e-9);
            let expect = (ph.to_radians() / 2.0).cos().powi(2);
            assert!((a - expect).abs() < 1e-9 || (b - expect).abs() < 1e-9, "{ph}: {a} {b}");
        }
        let (p, _) = IfoPreset::Sagnac.setup();
        for ph in [0.0, 70.0, 180.0] {
            for f in [0.0, 333.0] {
                let mut q = p.clone();
                q.parts.iter_mut().find(|x| x.kind == Kind::Phase).unwrap().phase_deg = ph;
                let s = steady_at(&q, f);
                assert!(det(&q, &s, 0) < 1e-12, "{ph} {f}");
                assert!((back(&q, &s) - 1.0).abs() < 1e-9);
            }
        }
        // a 45° Faraday rotation makes the two beams orthogonal: no interference, half the light
        let mut q = p.clone();
        q.parts.iter_mut().find(|x| x.kind == Kind::Faraday).unwrap().angle_deg = 45.0;
        let s = steady(&q);
        assert!((det(&q, &s, 0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn pbs_and_waveplates() {
        let (p, _) = IfoPreset::HalfWavePbs.setup();
        for th in [0.0, 10.0, 22.5, 45.0] {
            let mut q = p.clone();
            q.parts[1].angle_deg = th;
            let s = steady(&q);
            let expect = (2.0 * th.to_radians()).cos().powi(2);
            assert!((det(&q, &s, 0) - expect).abs() < 1e-9);
            assert!((det(&q, &s, 1) - (1.0 - expect)).abs() < 1e-9);
        }
        // λ/4 at 45° double pass: all reflected light ends up at the PBS's other port
        let (p, _) = IfoPreset::PbsReflection.setup();
        for f in [0.0, 100.0, 777.0] {
            let s = steady_at(&p, f);
            assert!(back(&p, &s) < 1e-12, "{f}");
            assert!((det(&p, &s, 0) + det(&p, &s, 1) - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn isolator_blocks_the_way_back() {
        let (p, _) = IfoPreset::Isolator.setup();
        let s = steady(&p);
        assert!(back(&p, &s) < 1e-12);
        assert!((det(&p, &s, 0) - 0.7).abs() < 1e-9);
        let mut q = p.clone();
        q.parts.iter_mut().find(|x| x.kind == Kind::Faraday).unwrap().angle_deg = 0.0;
        let s = steady(&q);
        // 0° then 45° polariser: half passes, the mirror sends 30 % back, half of that passes the first
        assert!((back(&q, &s) - 0.5 * 0.3 * 0.5).abs() < 1e-9, "{}", back(&q, &s));
    }

    #[test]
    fn exercise_13_polariser_and_quarter_wave_plate() {
        let (p, _) = IfoPreset::Exercise13.setup();
        let s = steady(&p);
        let d = p.detectors();
        // half of the laser goes each way, half of that passes the polariser
        for &k in &d {
            assert!((s.arrived[k] - 0.25).abs() < 1e-9);
        }
        // LP then QWP: right circular, (1, −i); QWP then LP: linear 45°
        let lp_qwp = s.at_part[d[0]];
        let qwp_lp = s.at_part[d[1]];
        assert_eq!(describe_polarisation(&lp_qwp), "right circular ↻");
        assert_eq!(describe_polarisation(&qwp_lp), "linear 45°");
        // the convention of the exercise: (1, i)/√2 is left circular
        assert_eq!(describe_polarisation(&polarisation(0.0, 45.0)), "left circular ↺");
        let v = polarisation(0.0, 45.0);
        assert!((v[1] / v[0] - I).norm() < 1e-12);
    }

    #[test]
    fn circular_light_flips_handedness_on_reflection() {
        let v = polarisation(0.0, 45.0);
        let m = diag(ONE, -ONE);
        let r = apply(&m, &v);
        assert!(ellipse(&v).1 > 44.9);
        assert!(ellipse(&r).1 < -44.9);
    }

    #[test]
    fn birefringent_cavity_splits_by_a_quarter_fsr() {
        let (p, _) = IfoPreset::Birefringent.setup();
        let c = steady(&p).cavity.unwrap();
        assert_eq!(c.modes.len(), 2, "{c:?}");
        let mut o: Vec<f64> = c.modes.iter().map(|m| m.offset_fsr).collect();
        o.sort_by(f64::total_cmp);
        let d = (o[1] - o[0]).min(1.0 - (o[1] - o[0]));
        assert!((d - 0.25).abs() < 1e-9, "{o:?}");
    }

    #[test]
    fn coupled_cavities_split() {
        let (p, _) = IfoPreset::Coupled.setup();
        // a single cavity of 7 squares would resonate at 0; the coupled pair does not
        let s = steady(&p);
        let r = sweep(&p, 100.0);
        let pk = peaks(&r.xs, &r.traces[0], 1e-9);
        assert!(det(&p, &s, 0) < 0.5);
        assert!(pk.len() >= 4, "{pk:?}");
    }

    #[test]
    fn switch_on_reaches_the_steady_state() {
        for detune in [0.0, 2.0] {
            let mut p = fp(0.9, 0.9);
            p.laser.detuning_mhz = detune;
            let s = steady(&p);
            let mut sim = TimeSim::new(&p);
            sim.advance(4000);
            let last = sim.history.last().unwrap();
            let want = det(&p, &s, 0) as f32;
            assert!((last[0] - want).abs() < 1e-4, "{detune}: {} vs {want}", last[0]);
            let want_back = back(&p, &s) as f32;
            assert!((last[1] - want_back).abs() < 1e-4);
            // light needs 21 steps from the laser through the cavity to the detector
            assert!(sim.history[19][0] == 0.0 && sim.history[21][0] > 0.0);
        }
    }

    #[test]
    fn swept_part_matches_direct_solution() {
        let (p, _) = IfoPreset::MachZehnder.setup();
        let r = sweep(&p, 100.0);
        for (i, &x) in r.xs.iter().enumerate().step_by(97) {
            let mut q = p.clone();
            q.parts.iter_mut().find(|x| x.kind == Kind::Phase).unwrap().phase_deg = x;
            let s = steady(&q);
            assert!((r.traces[0][i] - det(&q, &s, 0)).abs() < 1e-9);
        }
    }

    #[test]
    fn presets_are_lossless_and_survive_json() {
        for pr in IfoPreset::ALL {
            let (p, notes) = pr.setup();
            assert!(!notes.is_empty());
            let s = steady(&p);
            let lost = 1.0 - energy_balance(&p, &s);
            // only polarisers absorb
            let has_absorber = p.parts.iter().any(|q| q.kind == Kind::Polarizer || q.loss > 0.0);
            assert!(has_absorber || lost.abs() < 1e-9, "{}: {lost}", pr.label());
            let back: IfoParams = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
            assert_eq!(back, p, "{}", pr.label());
        }
    }
}
