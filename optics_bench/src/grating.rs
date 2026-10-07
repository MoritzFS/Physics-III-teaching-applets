//! A grating spectrometer, laid out as in the lecture notes: a source and a
//! reference lamp are combined at a beam splitter and focused onto the
//! entrance slit. A lens (f₁) makes the light from the slit parallel and sends
//! it onto a reflection grating; a second lens (f₂) focuses every direction
//! leaving the grating onto a point of a line camera.
//!
//! The grating has the groove spacing d = 1/(lines per mm). The parallel beam
//! lights a width W of it (set by an iris), that is N = W/d grooves. With the
//! angles θ_i (in) and θ_m (out) measured from the grating normal as in the
//! notes, neighbouring grooves differ in path by d (sin θ_m − sin θ_i), and
//! the orders are where this is a whole number of wavelengths:
//!
//! ```text
//! d (sin θ_m − sin θ_i) = m λ.
//! ```
//!
//! Across the lit part of the grating the path difference is
//! Δ = N d (sin θ_m − sin θ_i) = m N λ. A second wavelength λ + δλ is
//! resolved (Rayleigh) once Δ holds one wavelength more of the one than of the
//! other: δλ/λ = 1/(mN) = λ/Δ ≥ λ/(2Nd), since |sin θ_m − sin θ_i| ≤ 2.
//!
//! The light of one line is the N-slit pattern sin²(Nφ/2)/sin²(φ/2) with
//! φ = 2π d (sin θ_m − sin θ_i)/λ, times the pattern of a single groove, a
//! tilted facet (blaze; scalar theory, approximate). On the camera it is
//! smeared by the image of the slit and summed over each pixel.
//!
//! Inside, the arms are described by the angles α (towards the collimator)
//! and β (towards the camera) from the grating normal, positive on the same
//! side, so θ_i = −α and θ_m = β, and the grating equation reads
//! d (sin α + sin β) = mλ. The two arms are a fixed angle A apart; turning the
//! grating by ψ gives α = ψ + A/2 and β = ψ − A/2 at the centre of the camera.

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

pub const DEG: f64 = PI / 180.0;

// ---------------------------------------------------------------- lamps

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Lamp {
    Mercury,
    Neon,
    Sodium,
    Hydrogen,
    Helium,
    Cadmium,
    HeNe,
    Green,
    White,
    /// two lines of equal strength (`doublet_nm` ± `doublet_sep_nm`/2)
    Doublet,
}

/// Lines in air (nm) with rough relative strengths (lamps differ).
const MERCURY: &[(f64, f64)] =
    &[(404.656, 0.5), (407.783, 0.06), (435.833, 0.9), (491.607, 0.03), (546.074, 1.0), (576.960, 0.45), (579.066, 0.5)];
const NEON: &[(f64, f64)] = &[
    (540.056, 0.08),
    (585.249, 1.0),
    (588.190, 0.5),
    (594.483, 0.5),
    (597.553, 0.25),
    (603.000, 0.3),
    (607.434, 0.4),
    (609.616, 0.45),
    (614.306, 0.6),
    (616.359, 0.35),
    (621.728, 0.3),
    (626.650, 0.45),
    (630.479, 0.3),
    (633.443, 0.45),
    (638.299, 0.55),
    (640.225, 0.9),
    (650.653, 0.6),
    (653.288, 0.3),
    (659.895, 0.4),
    (667.828, 0.35),
    (671.704, 0.3),
    (692.947, 0.4),
    (703.241, 0.5),
];
const SODIUM: &[(f64, f64)] = &[(588.995, 1.0), (589.592, 0.5)];
const HYDROGEN: &[(f64, f64)] = &[(410.174, 0.08), (434.047, 0.15), (486.135, 0.35), (656.279, 1.0)];
const HELIUM: &[(f64, f64)] =
    &[(388.865, 0.3), (447.148, 0.4), (471.314, 0.1), (492.193, 0.1), (501.568, 0.25), (587.562, 1.0), (667.815, 0.35), (706.519, 0.3)];
const CADMIUM: &[(f64, f64)] = &[(467.815, 0.3), (479.992, 0.5), (508.582, 0.8), (643.847, 1.0)];

/// the incandescent lamp emits from here to there (nm)
pub const WHITE_RANGE: (f64, f64) = (360.0, 1000.0);

impl Lamp {
    pub const ALL: [Lamp; 10] = [
        Lamp::Mercury,
        Lamp::Neon,
        Lamp::Sodium,
        Lamp::Hydrogen,
        Lamp::Helium,
        Lamp::Cadmium,
        Lamp::HeNe,
        Lamp::Green,
        Lamp::White,
        Lamp::Doublet,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Lamp::Mercury => "mercury (Hg)",
            Lamp::Neon => "neon (Ne)",
            Lamp::Sodium => "sodium (Na)",
            Lamp::Hydrogen => "hydrogen (H, Balmer)",
            Lamp::Helium => "helium (He)",
            Lamp::Cadmium => "cadmium (Cd)",
            Lamp::HeNe => "HeNe laser 632.8 nm",
            Lamp::Green => "green laser 532 nm",
            Lamp::White => "white lamp (incandescent)",
            Lamp::Doublet => "test: two close lines",
        }
    }

    pub fn short(self) -> &'static str {
        match self {
            Lamp::Mercury => "Hg",
            Lamp::Neon => "Ne",
            Lamp::Sodium => "Na",
            Lamp::Hydrogen => "H",
            Lamp::Helium => "He",
            Lamp::Cadmium => "Cd",
            Lamp::HeNe => "HeNe",
            Lamp::Green => "laser",
            Lamp::White => "white",
            Lamp::Doublet => "test",
        }
    }

    /// (wavelength in nm, relative strength); none for the white lamp
    pub fn lines(self, p: &GratingParams) -> Vec<(f64, f64)> {
        match self {
            Lamp::Mercury => MERCURY.to_vec(),
            Lamp::Neon => NEON.to_vec(),
            Lamp::Sodium => SODIUM.to_vec(),
            Lamp::Hydrogen => HYDROGEN.to_vec(),
            Lamp::Helium => HELIUM.to_vec(),
            Lamp::Cadmium => CADMIUM.to_vec(),
            Lamp::HeNe => vec![(632.816, 1.0)],
            Lamp::Green => vec![(532.0, 1.0)],
            Lamp::White => vec![],
            Lamp::Doublet => {
                let h = 0.5 * p.doublet_sep_nm;
                vec![(p.doublet_nm - h, 1.0), (p.doublet_nm + h, 1.0)]
            }
        }
    }
}

/// incandescent lamp at 2900 K: Planck's law, 1 at 1000 nm, per nm
pub fn white_spectrum(l_nm: f64) -> f64 {
    if !(WHITE_RANGE.0..=WHITE_RANGE.1).contains(&l_nm) {
        return 0.0;
    }
    let planck = |l: f64| l.powi(-5) / ((1.4388e7 / (l * 2900.0)).exp() - 1.0);
    planck(l_nm) / planck(1000.0)
}

// ---------------------------------------------------------------- parameters

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Groove {
    /// sawtooth: facets tilted by the blaze angle
    Blazed,
    /// flat reflecting strips, a fraction `fill` of the period wide
    Strips,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Axis {
    /// camera pixels
    Pixels,
    /// wavelength from the fit through the reference lines
    Calibrated,
    /// wavelength from the grating equation (as if the geometry were known exactly)
    Exact,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GratingParams {
    /// the lamps at the source and at the reference input
    pub sources: Vec<Lamp>,
    pub references: Vec<Lamp>,
    pub doublet_nm: f64,
    pub doublet_sep_nm: f64,
    pub lines_per_mm: f64,
    pub groove: Groove,
    /// facet angle of a blazed grating
    pub blaze_deg: f64,
    /// width of the strips of a strip grating, as a fraction of d
    pub fill: f64,
    /// lit width of the grating (iris), mm
    pub width_mm: f64,
    /// grating turned by ψ from the bisector of the two arms
    pub angle_deg: f64,
    /// angle between the collimator and the camera arm
    pub arms_deg: f64,
    pub slit_um: f64,
    pub f1_mm: f64,
    pub f2_mm: f64,
    pub pixels: usize,
    pub pixel_um: f64,
    pub axis: Axis,
    /// degree of the calibration polynomial λ(pixel)
    pub fit_degree: usize,
    pub log: bool,
    /// shown part of the camera, as fractions of its length
    pub view: (f64, f64),
    /// all directions instead of the camera
    pub far_field: bool,
    /// far field: shown range of the path difference between neighbouring grooves (nm)
    pub far_view: (f64, f64),
    /// phasor picture: second wavelength, in units of the resolution limit λ/(mN)
    pub phasor_frac: f64,
}

impl Default for GratingParams {
    fn default() -> Self {
        GratingPreset::HowItWorks.setup().0
    }
}

/// The resolution of the current setting.
#[derive(Clone, Debug)]
pub struct Resolution {
    pub m: i32,
    pub lambda: f64,
    /// grooves lit
    pub n: f64,
    pub theta_i: f64,
    pub theta_m: f64,
    /// path difference across the lit grating, mm
    pub path_mm: f64,
    /// diffraction limit λ/(mN), slit and pixel (2 pixels) limits, nm
    pub dl_diff: f64,
    pub dl_slit: f64,
    pub dl_pixel: f64,
    /// the largest resolving power any order and angle could give: 2W/λ
    pub r_max: f64,
}

impl GratingParams {
    fn base() -> Self {
        GratingParams {
            sources: vec![Lamp::Sodium],
            references: vec![Lamp::Mercury],
            doublet_nm: 600.0,
            doublet_sep_nm: 1.0,
            lines_per_mm: 600.0,
            groove: Groove::Blazed,
            blaze_deg: 8.6,
            fill: 0.3,
            width_mm: 25.0,
            angle_deg: 0.0,
            arms_deg: 30.0,
            slit_um: 20.0,
            f1_mm: 200.0,
            f2_mm: 200.0,
            pixels: 2048,
            pixel_um: 14.0,
            axis: Axis::Calibrated,
            fit_degree: 3,
            log: false,
            view: (0.0, 1.0),
            far_field: false,
            far_view: (-2000.0, 2000.0),
            phasor_frac: 1.0,
        }
    }

    pub fn d_mm(&self) -> f64 {
        1.0 / self.lines_per_mm.max(1e-6)
    }

    /// grooves lit (at least one)
    pub fn grooves(&self) -> f64 {
        (self.width_mm * self.lines_per_mm).round().max(1.0)
    }

    pub fn alpha(&self) -> f64 {
        (self.angle_deg + 0.5 * self.arms_deg) * DEG
    }

    pub fn beta_c(&self) -> f64 {
        (self.angle_deg - 0.5 * self.arms_deg) * DEG
    }

    pub fn camera_mm(&self) -> f64 {
        self.pixels as f64 * self.pixel_um * 1e-3
    }

    pub fn beta_at(&self, x_mm: f64) -> f64 {
        self.beta_c() + (x_mm / self.f2_mm).atan()
    }

    /// position on the camera (mm from its centre) of light leaving at β
    pub fn x_of_beta(&self, beta: f64) -> Option<f64> {
        let d = beta - self.beta_c();
        (d.abs() < 0.49 * PI).then(|| self.f2_mm * d.tan())
    }

    /// the direction β of order m of λ
    pub fn beta_of(&self, lambda_nm: f64, m: i32, alpha: f64) -> Option<f64> {
        let s = m as f64 * lambda_nm * 1e-6 / self.d_mm() - alpha.sin();
        (s.abs() <= 1.0).then(|| s.asin())
    }

    /// where order m of λ lands on the camera (mm from its centre)
    pub fn x_of(&self, lambda_nm: f64, m: i32) -> Option<f64> {
        self.x_of_beta(self.beta_of(lambda_nm, m, self.alpha())?)
    }

    /// the wavelength that order m sends to the camera position x
    pub fn lambda_at(&self, x_mm: f64, m: i32) -> f64 {
        self.d_mm() * (self.alpha().sin() + self.beta_at(x_mm).sin()) / m as f64 * 1e6
    }

    pub fn pixel_of_x(&self, x_mm: f64) -> f64 {
        x_mm / (self.pixel_um * 1e-3) + 0.5 * self.pixels as f64
    }

    pub fn x_of_pixel(&self, px: f64) -> f64 {
        (px - 0.5 * self.pixels as f64) * self.pixel_um * 1e-3
    }

    /// d (sin θ_m − sin θ_i) at the centre of the camera, nm
    pub fn center_path_nm(&self) -> f64 {
        self.d_mm() * (self.alpha().sin() + self.beta_c().sin()) * 1e6
    }

    /// The order at the centre of the camera: the lowest whose wavelength
    /// there lies between 380 and 1000 nm (0: the mirror reflection).
    pub fn main_order(&self) -> i32 {
        let path = self.center_path_nm();
        if path.abs() < 300.0 {
            return 0;
        }
        for m in 1..=40 {
            let l = path.abs() / m as f64;
            if (380.0..=1000.0).contains(&l) {
                return m * path.signum() as i32;
            }
        }
        path.signum() as i32
    }

    /// turns the grating so that order m of λ lands at the centre of the camera
    pub fn set_center(&mut self, lambda_nm: f64, m: i32) -> bool {
        let s = m as f64 * lambda_nm * 1e-6 / (2.0 * self.d_mm() * (0.5 * self.arms_deg * DEG).cos());
        if s.abs() > 1.0 {
            return false;
        }
        self.angle_deg = s.asin() / DEG;
        true
    }

    /// shows the wavelengths lo … hi (main order) on the camera
    pub fn zoom_nm(&mut self, lo: f64, hi: f64) {
        let m = self.main_order();
        if m == 0 {
            return;
        }
        let l = self.camera_mm();
        let frac = |nm: f64| self.x_of(nm, m).map(|x| (x / l + 0.5).clamp(0.0, 1.0));
        if let (Some(a), Some(b)) = (frac(lo), frac(hi)) {
            self.view = (a.min(b), a.max(b));
        }
    }

    /// The pattern of a single groove, 1 at its maximum: a facet of width b
    /// tilted by the blaze angle (or a flat strip) diffracts like a slit,
    /// sinc²(πb(sin α′ + sin β′)/λ) with the angles from its own normal.
    pub fn efficiency(&self, lambda_nm: f64, alpha: f64, beta: f64) -> f64 {
        let (tb, b) = match self.groove {
            Groove::Blazed => (self.blaze_deg * DEG, self.d_mm() * (self.blaze_deg * DEG).cos()),
            Groove::Strips => (0.0, self.d_mm() * self.fill.clamp(0.01, 1.0)),
        };
        let u = PI * b * ((alpha - tb).sin() + (beta - tb).sin()) / (lambda_nm * 1e-6);
        if u.abs() < 1e-9 { 1.0 } else { (u.sin() / u).powi(2) }
    }

    pub fn resolution(&self, lambda_nm: f64) -> Resolution {
        let m = self.main_order();
        let n = self.grooves();
        let w = n * self.d_mm();
        let (a, b) = (self.alpha(), self.beta_c());
        let mm = (m.abs() as f64).max(1.0);
        let d = self.d_mm();
        Resolution {
            m,
            lambda: lambda_nm,
            n,
            theta_i: -a,
            theta_m: b,
            path_mm: w * (a.sin() + b.sin()),
            dl_diff: lambda_nm / (mm * n),
            dl_slit: d * a.cos() * (self.slit_um * 1e-3 / self.f1_mm) / mm * 1e6,
            dl_pixel: d * b.cos() * (2.0 * self.pixel_um * 1e-3 / self.f2_mm) / mm * 1e6,
            r_max: 2.0 * w / (lambda_nm * 1e-6),
        }
    }

    /// how wide a line is on the camera (pixels): the widest of its diffraction pattern
    /// (out to the first zero), the slit image and one pixel
    pub fn line_width_px(&self) -> f64 {
        let m = self.main_order();
        let lambda = if m == 0 { 550.0 } else { self.lambda_at(0.0, m) };
        let beta = self.beta_c();
        let pix = self.pixel_um * 1e-3;
        let diff = self.f2_mm * lambda * 1e-6 / (self.grooves() * self.d_mm() * beta.cos().abs().max(1e-6));
        let slit = self.slit_um * 1e-3 * (self.f2_mm / self.f1_mm) * self.alpha().cos().abs() / beta.cos().abs().max(1e-6);
        diff.max(slit).max(pix) / pix
    }

    /// all lines of the enabled lamps: (nm, strength, lamp)
    pub fn lines(&self, lamps: &[Lamp]) -> Vec<(f64, f64, Lamp)> {
        lamps.iter().flat_map(|&l| l.lines(self).into_iter().map(move |(nm, s)| (nm, s, l))).collect()
    }
}

/// sin²(Nφ/2)/sin²(φ/2): N² on the orders
pub fn n_slit(n: f64, phi: f64) -> f64 {
    let s = (0.5 * phi).sin();
    if s.abs() < 1e-7 {
        // near an order the ratio tends to N² cos²…: use the limit
        let k = (phi / (2.0 * PI)).round();
        let e = phi - 2.0 * PI * k;
        let num = (0.5 * n * e).sin();
        let den = 0.5 * e;
        if den.abs() < 1e-12 { n * n } else { (num / den).powi(2) }
    } else {
        ((0.5 * n * phi).sin() / s).powi(2)
    }
}

// ---------------------------------------------------------------- the camera

/// What the camera records, per pixel: with the source lamps and with the reference lamps.
#[derive(Clone, Debug, Default)]
pub struct Recording {
    pub source: Vec<f32>,
    pub reference: Vec<f32>,
    /// colour of the light on each pixel (linear RGB, weighted by power)
    pub rgb: Vec<[f32; 3]>,
    pub millis: f32,
}

/// cumulative integral of samples spaced h (trapezoid), with C(0) = 0
fn cumulative(f: &[f64], h: f64) -> Vec<f64> {
    let mut c = Vec::with_capacity(f.len());
    let mut acc = 0.0;
    c.push(0.0);
    for k in 1..f.len() {
        acc += 0.5 * (f[k - 1] + f[k]) * h;
        c.push(acc);
    }
    c
}

/// linear interpolation of samples spaced h starting at x0; constant beyond the ends
fn interp(c: &[f64], x0: f64, h: f64, x: f64) -> f64 {
    let t = (x - x0) / h;
    if t <= 0.0 {
        return c[0];
    }
    let n = c.len() - 1;
    if t >= n as f64 {
        return c[n];
    }
    let i = t.floor() as usize;
    let f = t - i as f64;
    c[i] * (1.0 - f) + c[i + 1] * f
}

/// Adds one line in order m to the pixels: the N-slit pattern, smeared by the
/// slit image, summed over each pixel. `power` is what the order carries.
fn add_line(p: &GratingParams, lambda_nm: f64, m: i32, power: f64, rgb: [f64; 3], out: &mut [f64], col: &mut [[f64; 3]]) {
    let alpha = p.alpha();
    let Some(beta) = p.beta_of(lambda_nm, m, alpha) else { return };
    let Some(xm) = p.x_of_beta(beta) else { return };
    let half = 0.5 * p.camera_mm();
    let pix = p.pixel_um * 1e-3;
    let n = p.grooves();
    let d = p.d_mm();
    let lam = lambda_nm * 1e-6;
    let f2 = p.f2_mm;
    let sec2 = 1.0 + (xm / f2).powi(2);
    // first zero, next order, slit image (all in mm on the camera)
    let w_diff = f2 * sec2 * lam / (n * d * beta.cos().abs().max(1e-6));
    let period = f2 * sec2 * lam / (d * beta.cos().abs().max(1e-6));
    let w_img = p.slit_um * 1e-3 * (f2 / p.f1_mm) * alpha.cos().abs() / beta.cos().abs().max(1e-6) * sec2;
    let window = (0.5 * w_img + pix + 40.0 * w_diff).min(0.5 * period).max(pix);
    if xm - window > half || xm + window < -half {
        return;
    }
    let smallest = w_img.max(pix);
    // a line much narrower than slit and pixel is a delta
    let delta = w_diff < smallest / 20.0;
    let mut h = if delta { smallest / 8.0 } else { (w_diff / 6.0).min(smallest / 4.0) };
    let mut count = (2.0 * window / h).ceil() as usize + 1;
    if count > 60_000 {
        h = 2.0 * window / 60_000.0;
        count = 60_001;
    }
    let x0 = xm - window;
    let mut f = vec![0.0; count];
    if delta {
        let t = (xm - x0) / h;
        let i = (t.floor() as usize).min(count - 2);
        let fr = t - i as f64;
        f[i] += (1.0 - fr) / h;
        f[i + 1] += fr / h;
    } else {
        // the N-slit pattern, normalised to 1 over an order: D(φ)/(2πN) dφ/dx
        let sa = alpha.sin();
        for (k, v) in f.iter_mut().enumerate() {
            let x = x0 + k as f64 * h;
            let b = p.beta_at(x);
            let phi = 2.0 * PI * d * (sa + b.sin()) / lam;
            let dphi = 2.0 * PI * d * b.cos() / lam / (f2 * (1.0 + (x / f2).powi(2)));
            *v = n_slit(n, phi) / (2.0 * PI * n) * dphi.abs();
        }
    }
    // the slit image: a box of width w_img
    let c1 = cumulative(&f, h);
    let g: Vec<f64> = if w_img > 0.5 * h {
        (0..count)
            .map(|k| {
                let x = x0 + k as f64 * h;
                (interp(&c1, x0, h, x + 0.5 * w_img) - interp(&c1, x0, h, x - 0.5 * w_img)) / w_img
            })
            .collect()
    } else {
        f
    };
    let c2 = cumulative(&g, h);
    let first = (((x0 + half) / pix).floor().max(0.0)) as usize;
    let last = (((x0 + 2.0 * window + half) / pix).ceil() as usize).min(out.len());
    for k in first..last {
        let xl = -half + k as f64 * pix;
        let v = (interp(&c2, x0, h, xl + pix) - interp(&c2, x0, h, xl)) * power;
        if v > 0.0 {
            out[k] += v;
            for c in 0..3 {
                col[k][c] += v * rgb[c];
            }
        }
    }
}

/// the orders of λ that can reach the camera
fn orders_on_camera(p: &GratingParams, lambda_nm: f64) -> std::ops::RangeInclusive<i32> {
    let half = 0.5 * p.camera_mm();
    let (b0, b1) = (p.beta_at(-half), p.beta_at(half));
    let sa = p.alpha().sin();
    let d = p.d_mm() * 1e6;
    // one more order on each side for the wings
    let lo = (d * (sa + b0.sin()) / lambda_nm).floor() as i32 - 1;
    let hi = (d * (sa + b1.sin()) / lambda_nm).ceil() as i32 + 1;
    lo.min(hi)..=hi.max(lo)
}

fn lamp_rgb(lambda_nm: f64) -> [f64; 3] {
    let c = crate::fourier::wavelength_rgb(lambda_nm.clamp(380.0, 780.0) as f32);
    // near infrared: show it dark red
    let ir = if lambda_nm > 780.0 { 0.35 } else { 1.0 };
    [c[0] as f64 * ir, c[1] as f64 * ir, c[2] as f64 * ir]
}

/// relative brightness of the white lamp per nm against a line
const WHITE_SCALE: f64 = 0.02;

fn record_lamps(p: &GratingParams, lamps: &[Lamp], out: &mut [f64], col: &mut [[f64; 3]]) {
    let w = p.grooves() * p.d_mm();
    let alpha = p.alpha();
    for (nm, s, _) in p.lines(lamps) {
        for m in orders_on_camera(p, nm) {
            if let Some(beta) = p.beta_of(nm, m, alpha) {
                let power = s * w * p.efficiency(nm, alpha, beta);
                add_line(p, nm, m, power, lamp_rgb(nm), out, col);
            }
        }
    }
    if lamps.contains(&Lamp::White) {
        // a smooth spectrum: each pixel gets the wavelengths that fall on it, in every order
        let half = 0.5 * p.camera_mm();
        let pix = p.pixel_um * 1e-3;
        let d = p.d_mm() * 1e6;
        let sa = alpha.sin();
        for (k, v) in out.iter_mut().enumerate() {
            let (xl, xr) = (-half + k as f64 * pix, -half + (k + 1) as f64 * pix);
            let (pl, pr) = (d * (sa + p.beta_at(xl).sin()), d * (sa + p.beta_at(xr).sin()));
            let mmax = (pl.abs().max(pr.abs()) / WHITE_RANGE.0).ceil() as i32;
            for m in (-mmax..=mmax).filter(|&m| m != 0) {
                let (l0, l1) = (pl / m as f64, pr / m as f64);
                let lm = 0.5 * (l0 + l1);
                let s = white_spectrum(lm);
                if s <= 0.0 {
                    continue;
                }
                let e = p.efficiency(lm, alpha, p.beta_at(0.5 * (xl + xr)));
                let pw = s * e * (l1 - l0).abs() * w * WHITE_SCALE;
                *v += pw;
                let c = lamp_rgb(lm);
                for i in 0..3 {
                    col[k][i] += pw * c[i];
                }
            }
        }
        // the mirror reflection (order 0) of all colours: one white line
        if let Some(beta0) = p.beta_of(550.0, 0, alpha) {
            let mut total = 0.0;
            let mut l = WHITE_RANGE.0;
            while l < WHITE_RANGE.1 {
                total += white_spectrum(l) * p.efficiency(l, alpha, beta0) * 5.0;
                l += 5.0;
            }
            add_line(p, 550.0, 0, total * w * WHITE_SCALE, [0.9, 0.85, 0.75], out, col);
        }
    }
}

pub fn record(p: &GratingParams) -> Recording {
    #[cfg(not(target_arch = "wasm32"))]
    let t0 = std::time::Instant::now();
    #[cfg(target_arch = "wasm32")]
    let t0 = web_time::Instant::now();
    let n = p.pixels.clamp(16, 16384);
    let mut src = vec![0.0; n];
    let mut refl = vec![0.0; n];
    let mut col = vec![[0.0; 3]; n];
    record_lamps(p, &p.sources, &mut src, &mut col);
    record_lamps(p, &p.references, &mut refl, &mut col);
    let rgb = col
        .iter()
        .zip(src.iter().zip(&refl))
        .map(|(c, (a, b))| {
            let t = (a + b).max(1e-300);
            [(c[0] / t) as f32, (c[1] / t) as f32, (c[2] / t) as f32]
        })
        .collect();
    Recording {
        source: src.iter().map(|&v| v as f32).collect(),
        reference: refl.iter().map(|&v| v as f32).collect(),
        rgb,
        millis: t0.elapsed().as_secs_f32() * 1000.0,
    }
}

// ---------------------------------------------------------------- far field

/// The light in all directions, against the path difference between
/// neighbouring grooves x = d (sin θ_m − sin θ_i) (nm): the orders are at x = mλ.
#[derive(Clone, Debug, Default)]
pub struct FarField {
    pub x: Vec<f64>,
    /// intensity relative to a line's order with full efficiency (peak 1)
    pub inten: Vec<f32>,
    pub rgb: Vec<[f32; 3]>,
}

pub fn far_field(p: &GratingParams, cols: usize) -> FarField {
    let (a, b) = p.far_view;
    let cols = cols.max(16);
    let alpha = p.alpha();
    let sa = alpha.sin();
    let d = p.d_mm() * 1e6;
    let n = p.grooves();
    let lamps: Vec<Lamp> = p.sources.iter().chain(&p.references).copied().collect();
    let lines = p.lines(&lamps);
    let white = lamps.contains(&Lamp::White);
    let step = (b - a) / cols as f64;
    let mut x = Vec::with_capacity(cols);
    let mut inten = vec![0.0f64; cols];
    let mut col = vec![[0.0f64; 3]; cols];
    let ov = 6;
    for k in 0..cols {
        let xc = a + (k as f64 + 0.5) * step;
        x.push(xc);
        for j in 0..ov {
            let xs = a + (k as f64 + (j as f64 + 0.5) / ov as f64) * step;
            let sb = xs / d - sa;
            if sb.abs() > 1.0 {
                continue;
            }
            let beta = sb.asin();
            let mut v = 0.0;
            let mut c = [0.0; 3];
            for &(nm, s, _) in &lines {
                let e = s * p.efficiency(nm, alpha, beta) * n_slit(n, 2.0 * PI * xs / nm) / (n * n);
                v += e;
                let rgb = lamp_rgb(nm);
                for i in 0..3 {
                    c[i] += e * rgb[i];
                }
            }
            if white {
                // every order m ≠ 0 sends the wavelength x/m this way; order 0 all of them
                let mmax = (xs.abs() / WHITE_RANGE.0).ceil() as i32;
                for m in (-mmax..=mmax).filter(|&m| m != 0) {
                    let l = xs / m as f64;
                    let s = white_spectrum(l) * 0.5;
                    if s > 0.0 {
                        let e = s * p.efficiency(l, alpha, beta);
                        v += e;
                        let rgb = lamp_rgb(l);
                        for i in 0..3 {
                            c[i] += e * rgb[i];
                        }
                    }
                }
                // the 0th order only (all colours at x = 0), not the orders of 550 nm
                let e0 = if xs.abs() < 275.0 { 0.5 * p.efficiency(550.0, alpha, beta) * n_slit(n, 2.0 * PI * xs / 550.0) / (n * n) } else { 0.0 };
                v += e0;
                for ci in c.iter_mut() {
                    *ci += e0 * 0.85;
                }
            }
            if v > inten[k] {
                inten[k] = v;
                col[k] = [c[0] / v.max(1e-300), c[1] / v.max(1e-300), c[2] / v.max(1e-300)];
            }
        }
    }
    // orders narrower than a column: make sure their peaks show
    for &(nm, s, _) in &lines {
        let m0 = (a / nm).ceil() as i64;
        let m1 = (b / nm).floor() as i64;
        for m in m0..=m1.min(m0 + 5000) {
            let xm = m as f64 * nm;
            let sb = xm / d - sa;
            if sb.abs() > 1.0 {
                continue;
            }
            let k = (((xm - a) / step) as usize).min(cols - 1);
            let v = s * p.efficiency(nm, alpha, sb.asin());
            if v > inten[k] {
                inten[k] = v;
                let rgb = lamp_rgb(nm);
                col[k] = rgb;
            }
        }
    }
    FarField {
        x,
        inten: inten.iter().map(|&v| v as f32).collect(),
        rgb: col.iter().map(|c| [c[0] as f32, c[1] as f32, c[2] as f32]).collect(),
    }
}

// ---------------------------------------------------------------- peaks and calibration

/// A peak in a recording: centre (pixels, from the centroid of its top) and height.
#[derive(Clone, Copy, Debug)]
pub struct Peak {
    pub px: f64,
    pub height: f64,
}

/// Local maxima above `frac` of the highest, each separated from the next by a
/// dip below 90 % of the lower peak (otherwise they count as one).
pub fn find_peaks(signal: &[f32], frac: f64) -> Vec<Peak> {
    let n = signal.len();
    let max = signal.iter().cloned().fold(0.0f32, f32::max) as f64;
    if max <= 0.0 || n < 3 {
        return vec![];
    }
    let mut cands: Vec<usize> = (1..n - 1)
        .filter(|&i| signal[i] as f64 >= frac * max && signal[i] > signal[i - 1] && signal[i] >= signal[i + 1])
        .collect();
    // merge maxima that are not separated by a real dip
    let mut merged: Vec<usize> = vec![];
    for i in cands.drain(..) {
        if let Some(&j) = merged.last() {
            let dip = signal[j..=i].iter().cloned().fold(f32::INFINITY, f32::min);
            if dip as f64 > 0.9 * (signal[i].min(signal[j]) as f64) {
                if signal[i] > signal[j] {
                    *merged.last_mut().expect("last") = i;
                }
                continue;
            }
        }
        merged.push(i);
    }
    merged
        .into_iter()
        .map(|i| {
            // centroid of the whole line: out to where it stops falling (at most 8 pixels)
            let mut l = i;
            while l > 0 && i - l < 8 && signal[l - 1] < signal[l] {
                l -= 1;
            }
            let mut r = i;
            while r + 1 < n && r - i < 8 && signal[r + 1] < signal[r] {
                r += 1;
            }
            let base = signal[l].min(signal[r]) as f64;
            let (mut sw, mut sx) = (0.0, 0.0);
            for (k, &v) in signal.iter().enumerate().take(r + 1).skip(l) {
                let w = (v as f64 - base).max(0.0);
                sw += w;
                sx += w * (k as f64 + 0.5);
            }
            let px = if sw > 0.0 { sx / sw } else { i as f64 + 0.5 };
            Peak { px, height: signal[i] as f64 }
        })
        .collect()
}

/// Leaves out the side lobes of strong lines: weaker peaks (less than 1/8)
/// within three line widths of a stronger one.
pub fn drop_side_lobes(peaks: Vec<Peak>, width_px: f64) -> Vec<Peak> {
    peaks
        .iter()
        .filter(|q| !peaks.iter().any(|r| r.height > 8.0 * q.height && (r.px - q.px).abs() < 3.0 * width_px))
        .copied()
        .collect()
}

/// A reference line found on the camera.
#[derive(Clone, Debug)]
pub struct Match {
    pub px: f64,
    pub lambda: f64,
    pub lamp: Lamp,
    /// the order it was seen in
    pub m: i32,
    /// fitted wavelength minus the true one (main order only), nm
    pub residual: Option<f64>,
}

/// λ(pixel) as a polynomial in u = (pixel − centre)/(half the pixels).
#[derive(Clone, Debug)]
pub struct Calibration {
    pub coeffs: Vec<f64>,
    pub matches: Vec<Match>,
    pub rms_nm: f64,
    pub m: i32,
    half: f64,
}

impl Calibration {
    pub fn lambda(&self, px: f64) -> f64 {
        let u = (px - self.half) / self.half;
        self.coeffs.iter().rev().fold(0.0, |acc, c| acc * u + c)
    }

    /// the pixel of λ (Newton's method from the linear guess)
    pub fn pixel(&self, lambda: f64) -> f64 {
        let mut px = self.half;
        for _ in 0..30 {
            let f = self.lambda(px) - lambda;
            let df = (self.lambda(px + 0.5) - self.lambda(px - 0.5)).max(1e-12);
            px -= f / df;
        }
        px
    }
}

/// least squares through (u, y) with a polynomial of the given degree
fn polyfit(u: &[f64], y: &[f64], degree: usize) -> Option<Vec<f64>> {
    let k = degree + 1;
    let mut a = vec![vec![0.0; k + 1]; k];
    for (&ui, &yi) in u.iter().zip(y) {
        let pw: Vec<f64> = (0..2 * k).map(|j| ui.powi(j as i32)).collect();
        for r in 0..k {
            for c in 0..k {
                a[r][c] += pw[r + c];
            }
            a[r][k] += pw[r] * yi;
        }
    }
    for col in 0..k {
        let piv = (col..k).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, piv);
        let pivot = a[col].clone();
        for (r, row) in a.iter_mut().enumerate() {
            if r != col {
                let f = row[col] / pivot[col];
                for (v, pv) in row.iter_mut().zip(&pivot).skip(col) {
                    *v -= f * pv;
                }
            }
        }
    }
    Some((0..k).map(|r| a[r][k] / a[r][r]).collect())
}

/// Finds the reference lines in the reference recording and fits λ(pixel)
/// through those in the main order.
pub fn calibrate(p: &GratingParams, rec: &Recording) -> Option<Calibration> {
    let m0 = p.main_order();
    if m0 == 0 || p.references.is_empty() {
        return None;
    }
    let peaks = find_peaks(&rec.reference, 0.01);
    let res = p.resolution(p.lambda_at(0.0, m0));
    // how far a found peak may be from where a line is expected (pixels)
    let nm_per_px = p.d_mm() * 1e6 * p.beta_c().cos() / (m0.abs() as f64 * p.f2_mm) * p.pixel_um * 1e-3;
    let width_px = res.dl_diff.max(res.dl_slit).max(res.dl_pixel) / nm_per_px.max(1e-12);
    let tol = (2.0 * width_px).max(3.0);
    let mut predicted: Vec<(f64, f64, Lamp, i32)> = vec![];
    for (nm, _, lamp) in p.lines(&p.references) {
        for m in orders_on_camera(p, nm).filter(|&m| m != 0) {
            if let Some(x) = p.x_of(nm, m) {
                let px = p.pixel_of_x(x);
                if (0.0..p.pixels as f64).contains(&px) {
                    predicted.push((px, nm, lamp, m));
                }
            }
        }
    }
    let mut matches: Vec<Match> = vec![];
    for pk in &peaks {
        let near: Vec<&(f64, f64, Lamp, i32)> = predicted.iter().filter(|q| (q.0 - pk.px).abs() < tol).collect();
        // two lines in one peak cannot be told apart: skip it
        if near.len() == 1 {
            let q = near[0];
            matches.push(Match { px: pk.px, lambda: q.1, lamp: q.2, m: q.3, residual: None });
        }
    }
    let main: Vec<usize> = (0..matches.len()).filter(|&i| matches[i].m == m0).collect();
    if main.len() < 2 {
        return Some(Calibration { coeffs: vec![], matches, rms_nm: f64::NAN, m: m0, half: 0.5 * p.pixels as f64 });
    }
    let half = 0.5 * p.pixels as f64;
    let degree = p.fit_degree.clamp(1, 3).min(main.len() - 1);
    let u: Vec<f64> = main.iter().map(|&i| (matches[i].px - half) / half).collect();
    let y: Vec<f64> = main.iter().map(|&i| matches[i].lambda).collect();
    let coeffs = polyfit(&u, &y, degree)?;
    let mut cal = Calibration { coeffs, matches, rms_nm: 0.0, m: m0, half };
    let mut sum = 0.0;
    for &i in &main {
        let r = cal.lambda(cal.matches[i].px) - cal.matches[i].lambda;
        cal.matches[i].residual = Some(r);
        sum += r * r;
    }
    cal.rms_nm = (sum / main.len() as f64).sqrt();
    Some(cal)
}

// ---------------------------------------------------------------- examples

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GratingPreset {
    HowItWorks,
    Calibration,
    SodiumDoublet,
    SecondOrder,
    Rayleigh,
    SlitWidth,
    Limit,
    FewGrooves,
    WhiteBlaze,
    OrderOverlap,
}

impl GratingPreset {
    pub const ALL: [GratingPreset; 10] = [
        GratingPreset::HowItWorks,
        GratingPreset::Calibration,
        GratingPreset::SodiumDoublet,
        GratingPreset::SecondOrder,
        GratingPreset::Rayleigh,
        GratingPreset::SlitWidth,
        GratingPreset::Limit,
        GratingPreset::FewGrooves,
        GratingPreset::WhiteBlaze,
        GratingPreset::OrderOverlap,
    ];

    pub fn label(self) -> &'static str {
        match self {
            GratingPreset::HowItWorks => "Grating spectrometer: how it works",
            GratingPreset::Calibration => "Calibrating with reference lamps",
            GratingPreset::SodiumDoublet => "The sodium doublet: how many grooves?",
            GratingPreset::SecondOrder => "Second order: twice the resolution",
            GratingPreset::Rayleigh => "Rayleigh criterion and the phasors",
            GratingPreset::SlitWidth => "Slit and pixels limit real spectrometers",
            GratingPreset::Limit => "The limit: δλ/λ ≥ λ/2Nd",
            GratingPreset::FewGrooves => "A few grooves: the N-slit pattern",
            GratingPreset::WhiteBlaze => "White light, orders and the blaze",
            GratingPreset::OrderOverlap => "Ghost lines: overlapping orders",
        }
    }

    pub fn setup(self) -> (GratingParams, String) {
        let mut p = GratingParams::base();
        p.set_center(550.0, 1);
        let notes = match self {
            GratingPreset::HowItWorks => {
                "The light of the source (sodium) and of the reference lamp (mercury) is focused onto the \
                 entrance slit. The first lens makes it parallel; the grating sends each wavelength into its own \
                 direction, d (sin θ_m − sin θ_i) = mλ; the second lens focuses each direction onto a point of the \
                 camera. So every line of a lamp becomes an image of the slit at its own place. The mercury lines \
                 have known wavelengths: they calibrate the pixel scale (the fit is shown below the spectrum), and \
                 with it the source's lines are measured: the two yellow sodium lines, 589.0 and 589.6 nm. Turn the \
                 grating (drag it, or the angle slider) to move the spectrum across the camera."
            }
            GratingPreset::Calibration => {
                p.sources = vec![Lamp::Helium];
                p.references = vec![Lamp::Mercury, Lamp::Neon];
                p.fit_degree = 1;
                "Which gas is in the source? First calibrate: the mercury and neon lines are at known wavelengths. \
                 The program finds them on the camera and fits λ(pixel) through them; the table lists each line \
                 and how far the fit misses it. A straight line misses by up to 0.35 nm (3 pixels), because the \
                 grating equation is not linear in the camera position: sin θ_m, and the lens maps angles with tan. \
                 Degree 2 misses by 0.1 nm, degree 3 by a few thousandths of a nanometre. Then read off the \
                 source's lines and compare with tables: 447.1, 501.6, 587.6, 667.8 nm is helium."
            }
            GratingPreset::SodiumDoublet => {
                p.references = vec![];
                p.width_mm = 1.0;
                p.slit_um = 5.0;
                p.axis = Axis::Exact;
                p.zoom_nm(584.0, 595.0);
                p.phasor_frac = 0.597 / (589.3 / 600.0);
                "The iris lets only 1 mm of the grating take part: N = 600 grooves. The two sodium lines \
                 (588.995 and 589.592 nm, δλ = 0.6 nm) melt into one, because each line is as wide as λ/(mN) = \
                 1 nm. Open the iris: from W ≈ 1.6 mm (N ≈ 990 ≥ λ/δλ) the dip between them appears (Rayleigh), \
                 and at 3 mm they are clearly apart. The lines get narrower because more grooves take part: the \
                 path difference across the lit grating, Δ = Nd(sin θ_m − sin θ_i), grows. The phasor picture shows \
                 why: for the second line the N phasors turn by 2π·Δ·δλ/λ² in total."
            }
            GratingPreset::SecondOrder => {
                p.references = vec![];
                p.width_mm = 1.0;
                p.slit_um = 5.0;
                p.axis = Axis::Exact;
                p.set_center(589.3, 2);
                p.zoom_nm(584.0, 595.0);
                "The same 600 grooves, but now the camera looks at the second order (the grating is turned \
                 further). The neighbouring grooves now differ by 2λ in path, and the whole lit grating by Δ = 2Nλ: \
                 the resolving power λ/δλ = mN = 1200 is enough for the sodium doublet. The lines are also twice as \
                 far apart on the camera (twice the dispersion), but the gain in resolution comes from the larger \
                 path difference, not from the camera."
            }
            GratingPreset::Rayleigh => {
                p.sources = vec![Lamp::Doublet];
                p.references = vec![];
                p.doublet_nm = 600.0;
                p.width_mm = 0.6;
                p.slit_um = 2.0;
                p.pixel_um = 5.0;
                p.pixels = 4096;
                p.axis = Axis::Exact;
                p.doublet_sep_nm = 600.0 / 360.0;
                p.zoom_nm(592.0, 608.0);
                p.phasor_frac = 1.0;
                "Two lines of equal strength, separated by exactly λ/(mN) = 1.67 nm (N = 360 grooves, first \
                 order). Each line's pattern has its first zero where the other has its maximum: the Rayleigh \
                 criterion. The dip between them is 81 % of the peaks. The phasor picture: at the direction where \
                 the first wavelength has its maximum, its N phasors (one per groove) all point the same way. For \
                 the second wavelength each groove adds a little more phase, 2π·mδλ/λ, and after N grooves it adds \
                 up to 2π: the chain closes into a circle and the sum is zero. Make the lines closer (test doublet \
                 separation) and they merge."
            }
            GratingPreset::SlitWidth => {
                p.references = vec![];
                p.slit_um = 200.0;
                p.axis = Axis::Exact;
                p.zoom_nm(584.0, 595.0);
                "The full grating takes part (25 mm, N = 15000): diffraction alone would separate lines 0.04 nm \
                 apart. But the slit is 200 µm wide and its image on the camera is wider than the line patterns: \
                 the sodium doublet is a single blob. Close the slit to 20 µm and the lines separate; the \
                 instrument is then limited by the pixels (two pixels per line). A narrower slit means less light: \
                 in practice the slit width is a compromise between resolution and signal. See the limits in the \
                 RESOLUTION panel."
            }
            GratingPreset::Limit => {
                p.references = vec![];
                p.lines_per_mm = 1200.0;
                p.blaze_deg = 26.0;
                p.width_mm = 1.0;
                p.arms_deg = 10.0;
                p.slit_um = 2.0;
                p.pixel_um = 5.0;
                p.pixels = 4096;
                p.axis = Axis::Exact;
                p.set_center(589.3, 2);
                p.zoom_nm(586.0, 593.0);
                "How far can a 1 mm wide grating go? The resolving power is λ/δλ = mN = Δ/λ, the path difference \
                 across the grating in wavelengths, and Δ = W(sin θ_m − sin θ_i) can never exceed 2W: so \
                 λ/δλ ≤ 2W/λ = 3400 here. A finer grating (1200 lines/mm) in the second order reaches mN = 2400, \
                 with θ_i and θ_m already at about ±50°. The third order of 589 nm does not exist at all with this \
                 grating (try the angle): it would need sin θ_m − sin θ_i > 2. The only way to resolve finer is a \
                 longer path difference: a wider grating, or an interferometer with long arms (Michelson) or many \
                 round trips (Fabry–Pérot)."
            }
            GratingPreset::FewGrooves => {
                p.sources = vec![Lamp::HeNe];
                p.references = vec![];
                p.lines_per_mm = 100.0;
                p.groove = Groove::Strips;
                p.fill = 0.3;
                p.width_mm = 0.05;
                p.far_field = true;
                p.far_view = (-3.6 * 632.8, 3.6 * 632.8);
                p.set_center(632.8, 1);
                "Only N = 5 grooves (100 lines/mm, 0.05 mm lit) and a laser. The plot shows the light in all \
                 directions against the path difference between neighbouring grooves, d(sin θ_m − sin θ_i): \
                 wherever it is a whole number of wavelengths, all N waves add up (principal maxima, height N²). In \
                 between there are N − 2 = 3 weak maxima. Increase the lit width: the principal maxima get narrower \
                 (width proportional to 1/N) while staying at the same places. The grooves are narrow reflecting strips (30 % \
                 of d), like the slits of a multi-slit: the single strip's pattern makes the higher orders weaker. \
                 Make the strips half as wide as d and every second order disappears."
            }
            GratingPreset::WhiteBlaze => {
                p.sources = vec![Lamp::White];
                p.references = vec![];
                p.far_field = true;
                p.far_view = (-2600.0, 2600.0);
                p.width_mm = 2.0;
                "White light: every order (except the 0th, the mirror reflection) is a spectrum, from violet \
                 (closer to the 0th order) to red. The higher orders are wider and soon overlap: the red of the \
                 second order lies on top of the violet of the third. The grooves are tilted by the blaze angle \
                 (8.6°), so most light goes into the first order on one side. Switch the grooves to flat strips: \
                 the orders on both sides become equal, and most light stays in the 0th order, useless for \
                 spectroscopy."
            }
            GratingPreset::OrderOverlap => {
                p.sources = vec![Lamp::Mercury];
                p.references = vec![];
                p.axis = Axis::Exact;
                p.set_center(830.0, 1);
                "The camera looks at 710–950 nm in the first order, and the mercury lamp seems to have lines at \
                 809.3, 815.6 and 871.7 nm. It has no strong lines there: these are 404.7, 407.8 and 435.8 nm in the \
                 second order, which leaves the grating in the same direction (2 × 404.7 = 809.3); the colour strip \
                 shows them violet and blue. The grating equation only fixes mλ. Spectrometers need an order-sorting filter (here: a long-pass filter \
                 blocking everything below about 600 nm) or a second dispersing element (echelle spectrographs \
                 use a prism across the grating)."
            }
        };
        (p, notes.into())
    }
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    fn quiet(p: &mut GratingParams) {
        p.sources.clear();
        p.references.clear();
    }

    #[test]
    fn grating_equation_round_trip() {
        let (p, _) = GratingPreset::HowItWorks.setup();
        assert_eq!(p.main_order(), 1);
        assert!(p.x_of(550.0, 1).unwrap().abs() < 1e-9);
        for nm in [450.0, 589.0, 650.0] {
            let x = p.x_of(nm, 1).unwrap();
            assert!((p.lambda_at(x, 1) - nm).abs() < 1e-9);
            // the notes' convention: d (sin θ_m − sin θ_i) = mλ
            let (ti, tm) = (-p.alpha(), p.beta_at(x));
            assert!((p.d_mm() * (tm.sin() - ti.sin()) * 1e6 - nm).abs() < 1e-9);
        }
        // order 0 is the mirror reflection: θ_m = θ_i
        let b0 = p.beta_of(500.0, 0, p.alpha()).unwrap();
        assert!((b0 + p.alpha()).abs() < 1e-12);
    }

    #[test]
    fn a_line_keeps_its_power() {
        let (mut p, _) = GratingPreset::HowItWorks.setup();
        quiet(&mut p);
        p.sources = vec![Lamp::HeNe];
        for (w, slit) in [(25.0, 20.0), (1.0, 5.0), (0.2, 50.0)] {
            p.width_mm = w;
            p.slit_um = slit;
            let r = record(&p);
            let total: f64 = r.source.iter().map(|&v| v as f64).sum();
            let alpha = p.alpha();
            let beta = p.beta_of(632.816, 1, alpha).unwrap();
            let expect = p.grooves() * p.d_mm() * p.efficiency(632.816, alpha, beta);
            // the first order (the 0th and 2nd do not reach the camera)
            assert!((total / expect - 1.0).abs() < 0.03, "W={w}: {total} vs {expect}");
        }
    }

    #[test]
    fn sodium_doublet_needs_about_990_grooves() {
        let (mut p, _) = GratingPreset::SodiumDoublet.setup();
        let count = |p: &GratingParams| {
            let r = record(p);
            let lo = p.pixel_of_x(p.x_of(586.0, 1).unwrap()) as usize;
            let hi = p.pixel_of_x(p.x_of(592.0, 1).unwrap()) as usize;
            find_peaks(&r.source[lo..hi], 0.1).len()
        };
        p.width_mm = 1.0;
        assert_eq!(count(&p), 1);
        p.width_mm = 3.0;
        assert_eq!(count(&p), 2);
        let (mut q, _) = GratingPreset::SecondOrder.setup();
        q.width_mm = 1.0;
        let r = record(&q);
        let lo = q.pixel_of_x(q.x_of(586.0, 2).unwrap()) as usize;
        let hi = q.pixel_of_x(q.x_of(592.0, 2).unwrap()) as usize;
        assert_eq!(find_peaks(&r.source[lo..hi], 0.1).len(), 2);
    }

    #[test]
    fn rayleigh_dip_is_81_percent() {
        let (p, _) = GratingPreset::Rayleigh.setup();
        let r = record(&p);
        let pk = find_peaks(&r.source, 0.2);
        assert_eq!(pk.len(), 2, "{pk:?}");
        let (a, b) = (pk[0].px as usize, pk[1].px as usize);
        let dip = r.source[a..b].iter().cloned().fold(f32::INFINITY, f32::min) as f64;
        let ratio = dip / pk[0].height.min(pk[1].height);
        // 8/π² for two sinc² patterns at the Rayleigh distance
        assert!((ratio - 8.0 / (PI * PI)).abs() < 0.03, "{ratio}");
    }

    #[test]
    fn calibration_finds_the_reference_lines() {
        let (mut p, _) = GratingPreset::Calibration.setup();
        p.fit_degree = 1;
        let r = record(&p);
        let lin = calibrate(&p, &r).unwrap();
        p.fit_degree = 3;
        let quad = calibrate(&p, &r).unwrap();
        let used = quad.matches.iter().filter(|m| m.residual.is_some()).count();
        assert!(used >= 20, "{:?}", quad.matches);
        // the numbers in the notes
        let worst = |c: &Calibration| c.matches.iter().filter_map(|m| m.residual).fold(0.0f64, |a, b| a.max(b.abs()));
        assert!((0.25..0.45).contains(&worst(&lin)), "{}", worst(&lin));
        assert!(worst(&quad) < 0.01, "{}", worst(&quad));
        // helium 587.562 nm measured through the fit
        let x = p.x_of(587.562, 1).unwrap();
        let src = find_peaks(&r.source, 0.05);
        let near = src.iter().min_by(|a, b| (a.px - p.pixel_of_x(x)).abs().total_cmp(&(b.px - p.pixel_of_x(x)).abs())).unwrap();
        assert!((quad.lambda(near.px) - 587.562).abs() < 0.01, "{}", quad.lambda(near.px));
    }

    #[test]
    fn second_order_ghosts() {
        let (p, _) = GratingPreset::OrderOverlap.setup();
        let r = record(&p);
        let x = p.x_of(809.312, 1).unwrap();
        let px = p.pixel_of_x(x);
        assert!(find_peaks(&r.source, 0.02).iter().any(|q| (q.px - px).abs() < 2.0));
    }

    #[test]
    fn few_grooves_have_n_minus_2_side_maxima() {
        let (mut p, _) = GratingPreset::FewGrooves.setup();
        // between the 0th and the 1st order (whose top the strips' pattern shifts a little)
        p.far_view = (10.0, 600.0);
        let f = far_field(&p, 2000);
        let n = f.inten.len();
        let maxima = (1..n - 1).filter(|&i| f.inten[i] > f.inten[i - 1] && f.inten[i] >= f.inten[i + 1]).count();
        assert_eq!(maxima, 3);
        // principal maximum: all N waves in phase
        let mut q = p.clone();
        q.far_view = (600.0, 660.0);
        let g = far_field(&q, 2000);
        let top = g.inten.iter().cloned().fold(0.0, f32::max);
        let e = p.efficiency(632.816, p.alpha(), (632.816 / (p.d_mm() * 1e6) - p.alpha().sin()).asin());
        assert!((top as f64 - e).abs() < 1e-3, "{top} {e}");
    }

    #[test]
    fn limit_and_resolution() {
        let (p, _) = GratingPreset::Limit.setup();
        let r = p.resolution(589.3);
        assert_eq!(r.m, 2);
        assert!((r.path_mm * 1e6 / 589.3 - (r.m as f64) * r.n).abs() < 1e-6);
        assert!((r.r_max - 2.0 / 589.3e-6).abs() < 1e-6);
        let mut q = p.clone();
        assert!(!q.set_center(589.3, 3));
        // blaze maximum is 1 at Littrow
        let tb = p.blaze_deg * DEG;
        let l = 2.0 * p.d_mm() * tb.sin() * 1e6;
        assert!((p.efficiency(l, tb, tb) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn recordings_are_fast_enough() {
        for pr in GratingPreset::ALL {
            let (p, _) = pr.setup();
            let t0 = std::time::Instant::now();
            let r = record(&p);
            let f = far_field(&p, 1600);
            eprintln!("{}: camera {:.1} ms, far field {:.1} ms", pr.label(), r.millis, t0.elapsed().as_secs_f32() * 1000.0 - r.millis);
            assert!(r.source.iter().chain(&r.reference).all(|v| v.is_finite()));
            assert!(f.inten.iter().all(|v| v.is_finite()));
        }
    }

    /// equal up to the last bits of floats (serde_json parses floats to within an ulp)
    fn close(a: &serde_json::Value, b: &serde_json::Value) -> bool {
        use serde_json::Value::*;
        match (a, b) {
            (Number(x), Number(y)) => {
                let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
                (x - y).abs() <= 1e-12 * x.abs().max(1.0)
            }
            (Array(x), Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(u, v)| close(u, v)),
            (Object(x), Object(y)) => x.len() == y.len() && x.iter().all(|(k, u)| y.get(k).is_some_and(|v| close(u, v))),
            _ => a == b,
        }
    }

    #[test]
    fn presets_survive_json() {
        for pr in GratingPreset::ALL {
            let (p, notes) = pr.setup();
            assert!(!notes.is_empty());
            let back: GratingParams = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
            assert!(close(&serde_json::to_value(&back).unwrap(), &serde_json::to_value(&p).unwrap()), "{}", pr.label());
        }
    }
}
