//! Scalar wave optics of a 4f system (Fourier optics).
//!
//!   object --f1-- L1 --f1-- Fourier plane (filter) --f2-- L2 --f2-- image
//!
//! The three planes are computed exactly (within the paraxial scalar theory)
//! with 2D FFTs: the field in the back focal plane of L1 is the Fourier
//! transform of the input field, the image is the (inverted) transform of the
//! filtered spectrum. The side view is a 1D angular-spectrum propagation of the
//! central line of the object through the whole system.

use std::f32::consts::PI;
use std::sync::{mpsc, Arc};
use std::time::Instant;

use eframe::egui::{Color32, ColorImage};
use rayon::prelude::*;
use rustfft::num_complex::Complex32 as C;
use rustfft::{Fft, FftPlanner};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Beam {
    /// plane wave filling the whole window
    Plane,
    Gaussian,
    /// uniform, circular
    TopHat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Object {
    Open,
    SingleSlit,
    DoubleSlit,
    Circle,
    Square,
    Grating,
    SineGrating,
    Mesh,
    LetterF,
    MeshF,
    Checker,
    PhaseGrating,
    PhaseF,
    Vortex,
}

impl Object {
    pub const ALL: [Object; 14] = [
        Object::Open,
        Object::SingleSlit,
        Object::DoubleSlit,
        Object::Circle,
        Object::Square,
        Object::Grating,
        Object::SineGrating,
        Object::Mesh,
        Object::LetterF,
        Object::MeshF,
        Object::Checker,
        Object::PhaseGrating,
        Object::PhaseF,
        Object::Vortex,
    ];
    pub fn label(self) -> &'static str {
        match self {
            Object::Open => "nothing (open)",
            Object::SingleSlit => "single slit",
            Object::DoubleSlit => "double slit",
            Object::Circle => "circular hole",
            Object::Square => "square hole",
            Object::Grating => "grating (binary)",
            Object::SineGrating => "grating (sinusoidal)",
            Object::Mesh => "wire mesh",
            Object::LetterF => "letter F",
            Object::MeshF => "letter F behind a mesh",
            Object::Checker => "checkerboard",
            Object::PhaseGrating => "phase grating",
            Object::PhaseF => "transparent F (phase only)",
            Object::Vortex => "spiral phase plate (vortex)",
        }
    }
    /// which of the parameters matter: (size, period, duty, angle, phase)
    pub fn uses(self) -> (bool, bool, bool, bool, bool) {
        match self {
            Object::Open => (false, false, false, false, false),
            Object::SingleSlit => (true, false, false, true, false),
            Object::DoubleSlit => (true, true, false, true, false),
            Object::Circle => (true, false, false, false, false),
            Object::Square => (true, false, false, true, false),
            Object::Grating | Object::Mesh | Object::Checker => (true, true, true, true, false),
            Object::SineGrating => (true, true, false, true, false),
            Object::LetterF => (true, false, false, false, false),
            Object::MeshF => (true, true, true, true, false),
            Object::PhaseGrating => (true, true, true, true, true),
            Object::PhaseF => (true, false, false, false, true),
            Object::Vortex => (true, false, false, false, true),
        }
    }
    fn is_binary(self) -> bool {
        !matches!(self, Object::SineGrating | Object::Open | Object::Vortex)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Filter {
    None,
    LowPass,
    HighPass,
    BandPass,
    Slit,
    KnifeEdge,
    Zernike,
    Spiral,
}

impl Filter {
    pub const ALL: [Filter; 8] = [
        Filter::None,
        Filter::LowPass,
        Filter::HighPass,
        Filter::BandPass,
        Filter::Slit,
        Filter::KnifeEdge,
        Filter::Zernike,
        Filter::Spiral,
    ];
    pub fn label(self) -> &'static str {
        match self {
            Filter::None => "none",
            Filter::LowPass => "low pass (pinhole)",
            Filter::HighPass => "high pass (central stop)",
            Filter::BandPass => "band pass (ring)",
            Filter::Slit => "slit",
            Filter::KnifeEdge => "knife edge (schlieren)",
            Filter::Zernike => "phase dot (Zernike phase contrast)",
            Filter::Spiral => "spiral phase (vortex filter)",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InputView {
    /// hue = phase, brightness = amplitude
    Complex,
    Amplitude,
    Phase,
    Intensity,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FourierParams {
    /// grid size (N x N)
    pub n: usize,
    /// width of the input window (mm)
    pub window_mm: f32,
    pub wavelength_nm: f32,
    pub f1_mm: f32,
    pub f2_mm: f32,
    // illumination
    pub beam: Beam,
    /// Gaussian waist (1/e² radius) or top-hat diameter
    pub beam_mm: f32,
    pub tilt_x_mrad: f32,
    pub tilt_y_mrad: f32,
    /// distance of a point source in front of the object (mm); 0 = plane wave
    pub source_mm: f32,
    // object
    pub object: Object,
    pub size_mm: f32,
    /// length of slits (mm)
    pub length_mm: f32,
    pub period_mm: f32,
    pub duty: f32,
    pub angle_deg: f32,
    /// phase depth (rad) or topological charge
    pub phase: f32,
    // Fourier filter
    pub filter: Filter,
    pub filter_mm: f32,
    pub filter2_mm: f32,
    pub filter_angle_deg: f32,
    pub filter_phase: f32,
    // display
    pub input_view: InputView,
    pub log_fourier: bool,
    pub log_image: bool,
    pub log_side: bool,
    /// display zoom of the input, Fourier and image panels
    pub zoom: [f32; 3],
}

impl Default for FourierParams {
    fn default() -> Self {
        FourierParams {
            n: 512,
            window_mm: 8.0,
            wavelength_nm: 633.0,
            f1_mm: 200.0,
            f2_mm: 200.0,
            beam: Beam::Plane,
            beam_mm: 3.0,
            tilt_x_mrad: 0.0,
            tilt_y_mrad: 0.0,
            source_mm: 0.0,
            object: Object::DoubleSlit,
            size_mm: 0.1,
            length_mm: 1.0,
            period_mm: 0.5,
            duty: 0.5,
            angle_deg: 0.0,
            phase: PI,
            filter: Filter::None,
            filter_mm: 0.2,
            filter2_mm: 0.5,
            filter_angle_deg: 0.0,
            filter_phase: PI / 2.0,
            input_view: InputView::Complex,
            log_fourier: true,
            log_image: false,
            log_side: false,
            zoom: [1.0; 3],
        }
    }
}

impl FourierParams {
    pub fn dx(&self) -> f32 {
        self.window_mm / self.n as f32
    }
    pub fn lambda_mm(&self) -> f32 {
        self.wavelength_nm * 1e-6
    }
    /// pixel size in the Fourier plane (mm)
    pub fn du(&self) -> f32 {
        self.lambda_mm() * self.f1_mm / self.window_mm
    }
    /// width of the Fourier plane window (mm)
    pub fn fourier_window(&self) -> f32 {
        self.du() * self.n as f32
    }
    pub fn magnification(&self) -> f32 {
        self.f2_mm / self.f1_mm
    }
    /// largest tilt that the grid can represent (mrad)
    pub fn max_tilt_mrad(&self) -> f32 {
        self.lambda_mm() / (2.0 * self.dx()) * 1000.0
    }
}

// ---------------------------------------------------------------- field model

fn in_letter_f(x: f32, y: f32) -> bool {
    // unit letter, height 1, centred; y up
    let stem = (-0.35..=-0.13).contains(&x) && (-0.5..=0.5).contains(&y);
    let top = (-0.35..=0.33).contains(&x) && (0.3..=0.5).contains(&y);
    let mid = (-0.35..=0.18).contains(&x) && (-0.06..=0.13).contains(&y);
    stem || top || mid
}

fn frac(v: f32) -> f32 {
    v - v.floor()
}

/// complex transmission of the object at (x, y) in mm (y up)
fn transmission(p: &FourierParams, x: f32, y: f32) -> C {
    let a = p.angle_deg.to_radians();
    let (sa, ca) = a.sin_cos();
    let xr = x * ca + y * sa;
    let yr = -x * sa + y * ca;
    let inside = |r: f32| x * x + y * y <= r * r;
    let amp = |b: bool| if b { C::new(1.0, 0.0) } else { C::new(0.0, 0.0) };
    let s = p.size_mm;
    let per = p.period_mm.max(1e-4);
    let holes = |u: f32, v: f32| frac(u / per) < p.duty && frac(v / per) < p.duty;
    match p.object {
        Object::Open => C::new(1.0, 0.0),
        Object::SingleSlit => amp(xr.abs() <= s * 0.5 && yr.abs() <= p.length_mm * 0.5),
        Object::DoubleSlit => amp(
            ((xr - per * 0.5).abs() <= s * 0.5 || (xr + per * 0.5).abs() <= s * 0.5) && yr.abs() <= p.length_mm * 0.5,
        ),
        Object::Circle => amp(inside(s * 0.5)),
        Object::Square => amp(xr.abs() <= s * 0.5 && yr.abs() <= s * 0.5),
        Object::Grating => amp(inside(s * 0.5) && frac(xr / per + 0.5 * p.duty) < p.duty),
        Object::SineGrating => {
            if inside(s * 0.5) {
                C::new(0.5 + 0.5 * (2.0 * PI * xr / per).cos(), 0.0)
            } else {
                C::new(0.0, 0.0)
            }
        }
        Object::Mesh => amp(inside(s * 0.5) && holes(xr, yr)),
        Object::LetterF => amp(in_letter_f(x / s, y / s)),
        Object::MeshF => amp(in_letter_f(x / s, y / s) && holes(xr, yr)),
        Object::Checker => {
            let c = (xr / per).floor() as i64 + (yr / per).floor() as i64;
            amp(inside(s * 0.5) && c.rem_euclid(2) == 0)
        }
        Object::PhaseGrating => {
            if inside(s * 0.5) {
                let ph = if frac(xr / per) < p.duty { p.phase } else { 0.0 };
                C::from_polar(1.0, ph)
            } else {
                C::new(0.0, 0.0)
            }
        }
        Object::PhaseF => {
            let ph = if in_letter_f(x / s, y / s) { p.phase } else { 0.0 };
            C::from_polar(1.0, ph)
        }
        Object::Vortex => {
            if inside(s * 0.5) {
                C::from_polar(1.0, p.phase.round() * y.atan2(x))
            } else {
                C::new(0.0, 0.0)
            }
        }
    }
}

fn illumination(p: &FourierParams, x: f32, y: f32) -> C {
    let r2 = x * x + y * y;
    let a = match p.beam {
        Beam::Plane => 1.0,
        Beam::Gaussian => (-r2 / (p.beam_mm * p.beam_mm)).exp(),
        Beam::TopHat => {
            if r2 <= p.beam_mm * p.beam_mm * 0.25 {
                1.0
            } else {
                0.0
            }
        }
    };
    let k = 2.0 * PI / p.lambda_mm();
    let mut ph = k * (x * (p.tilt_x_mrad * 1e-3).sin() + y * (p.tilt_y_mrad * 1e-3).sin());
    if p.source_mm.abs() > 1e-3 {
        ph += k * r2 / (2.0 * p.source_mm);
    }
    C::from_polar(a, ph)
}

/// field right behind the object, with 3x3 supersampling of sharp masks
fn input_at(p: &FourierParams, x: f32, y: f32, dx: f32) -> C {
    let t = if p.object.is_binary() {
        let mut acc = C::new(0.0, 0.0);
        for sy in [-1.0f32, 0.0, 1.0] {
            for sx in [-1.0f32, 0.0, 1.0] {
                acc += transmission(p, x + sx * dx / 3.0, y + sy * dx / 3.0);
            }
        }
        acc / 9.0
    } else {
        transmission(p, x, y)
    };
    illumination(p, x, y) * t
}

fn filter_at(p: &FourierParams, u: f32, v: f32) -> C {
    let r = (u * u + v * v).sqrt();
    let a = p.filter_angle_deg.to_radians();
    let ur = u * a.cos() + v * a.sin();
    let pass = |b: bool| if b { C::new(1.0, 0.0) } else { C::new(0.0, 0.0) };
    match p.filter {
        Filter::None => C::new(1.0, 0.0),
        Filter::LowPass => pass(r <= p.filter_mm),
        Filter::HighPass => pass(r >= p.filter_mm),
        Filter::BandPass => pass(r >= p.filter_mm && r <= p.filter2_mm),
        Filter::Slit => pass(ur.abs() <= p.filter_mm * 0.5),
        Filter::KnifeEdge => pass(ur >= p.filter_mm),
        Filter::Zernike => {
            if r <= p.filter_mm {
                C::from_polar(1.0, p.filter_phase)
            } else {
                C::new(1.0, 0.0)
            }
        }
        Filter::Spiral => C::from_polar(1.0, p.filter_phase.round() * v.atan2(u)),
    }
}

// ---------------------------------------------------------------- FFT helpers

fn fft_rows(data: &mut [C], n: usize, fft: &Arc<dyn Fft<f32>>) {
    data.par_chunks_mut(n * 8).for_each(|chunk| {
        let mut scratch = vec![C::new(0.0, 0.0); fft.get_inplace_scratch_len()];
        fft.process_with_scratch(chunk, &mut scratch);
    });
}

fn fftshift2(data: &mut [C], n: usize) {
    let h = n / 2;
    for j in 0..h {
        for i in 0..n {
            data.swap(j * n + i, (j + h) * n + (i + h) % n);
        }
    }
}

/// centred 2D DFT, normalised by 1/n
fn fft2_centered(data: &mut [C], n: usize, fft: &Arc<dyn Fft<f32>>) {
    fftshift2(data, n);
    fft_rows(data, n, fft);
    let mut t = vec![C::new(0.0, 0.0); n * n];
    transpose::transpose(data, &mut t, n, n);
    fft_rows(&mut t, n, fft);
    transpose::transpose(&t, data, n, n);
    fftshift2(data, n);
    let s = 1.0 / n as f32;
    data.par_iter_mut().for_each(|v| *v *= s);
}

// ---------------------------------------------------------------- colours

pub fn wavelength_rgb(l: f32) -> [f32; 3] {
    let (r, g, b) = if l < 440.0 {
        (-(l - 440.0) / 60.0, 0.0, 1.0)
    } else if l < 490.0 {
        (0.0, (l - 440.0) / 50.0, 1.0)
    } else if l < 510.0 {
        (0.0, 1.0, -(l - 510.0) / 20.0)
    } else if l < 580.0 {
        ((l - 510.0) / 70.0, 1.0, 0.0)
    } else if l < 645.0 {
        (1.0, -(l - 645.0) / 65.0, 0.0)
    } else {
        (1.0, 0.0, 0.0)
    };
    let f = if l < 420.0 {
        0.3 + 0.7 * (l - 380.0) / 40.0
    } else if l > 700.0 {
        0.3 + 0.7 * (780.0 - l) / 80.0
    } else {
        1.0
    };
    [r * f, g * f, b * f]
}

fn to8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// intensity (already normalised to 0..1, or log-mapped) in the colour of the light
fn tinted(s: f32, tint: [f32; 3]) -> Color32 {
    let s = s.clamp(0.0, 1.0);
    let white = ((s - 0.8) / 0.2).clamp(0.0, 1.0) * 0.55;
    let c = |t: f32| to8((t * s * 1.1) * (1.0 - white) + white);
    Color32::from_rgb(c(tint[0]), c(tint[1]), c(tint[2]))
}

fn hsv(h: f32, s: f32, v: f32) -> Color32 {
    let h = frac(h) * 6.0;
    let i = h.floor() as i32;
    let f = h - i as f32;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f)));
    let (r, g, b) = match i {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    Color32::from_rgb(to8(r), to8(g), to8(b))
}

fn intensity_image(field: &[C], n: usize, log: bool, tint: [f32; 3]) -> (ColorImage, f32) {
    let inten: Vec<f32> = field.par_iter().map(|c| c.norm_sqr()).collect();
    let max = inten.iter().cloned().fold(0.0f32, f32::max).max(1e-30);
    let pixels: Vec<Color32> = inten
        .par_iter()
        .map(|&v| {
            let r = v / max;
            let s = if log { ((r.max(1e-12).log10() + 5.0) / 5.0).clamp(0.0, 1.0) } else { r.powf(1.0 / 2.2) };
            tinted(s, tint)
        })
        .collect();
    (ColorImage::new([n, n], pixels), max)
}

// ---------------------------------------------------------------- result

pub struct FourierResult {
    pub params: FourierParams,
    pub input: ColorImage,
    pub fourier: ColorImage,
    pub image: ColorImage,
    pub side: ColorImage,
    /// intensity of the side view, column-major (nz columns of `side_m` values, top = +x)
    pub side_intensity: Vec<f32>,
    pub side_nz: usize,
    pub side_m: usize,
    /// half height of the side view (mm)
    pub side_half_mm: f32,
    /// wavefront along the centre line of the input: phase in waves (NaN where dark) and amplitude
    pub row_phase: Vec<f32>,
    pub row_amp: Vec<f32>,
    /// intensity along the centre line of the Fourier plane and of the image (normalised)
    pub fourier_row: Vec<f32>,
    pub image_row: Vec<f32>,
    /// fraction of the light that passes the filter
    pub passed: f32,
    pub millis: f32,
    /// the side view is not sampled finely enough for this setup
    pub side_aliased: bool,
}

/// input field, Fourier-plane field and filtered image field (all n x n, rows top to bottom)
pub fn raw_fields(p: &FourierParams, planner: &mut FftPlanner<f32>) -> (Vec<C>, Vec<C>, Vec<C>) {
    let n = p.n.clamp(64, 2048);
    let dx = p.window_mm / n as f32;
    let du = p.lambda_mm() * p.f1_mm / p.window_mm;
    let h = n as f32 / 2.0;
    let input: Vec<C> = (0..n * n)
        .into_par_iter()
        .map(|idx| {
            let (i, j) = (idx % n, idx / n);
            input_at(p, (i as f32 - h) * dx, (h - j as f32) * dx, dx)
        })
        .collect();
    let fft = planner.plan_fft_forward(n);
    let mut spec = input.clone();
    fft2_centered(&mut spec, n, &fft);
    let mut image: Vec<C> = spec
        .par_iter()
        .enumerate()
        .map(|(idx, c)| {
            let (i, j) = (idx % n, idx / n);
            c * filter_at(p, (i as f32 - h) * du, (h - j as f32) * du)
        })
        .collect();
    fft2_centered(&mut image, n, &fft);
    (input, spec, image)
}

pub fn compute(p: &FourierParams, planner: &mut FftPlanner<f32>) -> FourierResult {
    let t0 = Instant::now();
    let n = p.n.clamp(64, 2048);
    let tint = wavelength_rgb(p.wavelength_nm);
    let (input, spec, image) = raw_fields(p, planner);

    // ---- input field
    let amax = input.iter().map(|c| c.norm()).fold(0.0f32, f32::max).max(1e-20);
    let input_img = ColorImage::new(
        [n, n],
        input
            .par_iter()
            .map(|c| {
                let a = c.norm() / amax;
                let ph = c.arg() / (2.0 * PI);
                match p.input_view {
                    InputView::Complex => hsv(ph, 0.85, a.powf(1.0 / 1.5)),
                    InputView::Amplitude => {
                        let v = to8(a.powf(1.0 / 2.2));
                        Color32::from_rgb(v, v, v)
                    }
                    InputView::Phase => {
                        if a > 0.01 {
                            hsv(ph, 0.85, 1.0)
                        } else {
                            Color32::from_gray(25)
                        }
                    }
                    InputView::Intensity => tinted((a * a).powf(1.0 / 2.2), tint),
                }
            })
            .collect(),
    );

    // wavefront along the centre line
    let row = &input[(n / 2) * n..(n / 2 + 1) * n];
    let mut ph: Vec<f32> = row.iter().map(|c| c.arg()).collect();
    for i in 1..n {
        let d = ph[i] - ph[i - 1];
        ph[i] -= (d / (2.0 * PI)).round() * 2.0 * PI;
    }
    let row_max = row.iter().map(|c| c.norm()).fold(0.0f32, f32::max).max(1e-20);
    let valid: Vec<bool> = row.iter().map(|c| c.norm() > 0.02 * row_max).collect();
    let ref_i = if valid[n / 2] { n / 2 } else { valid.iter().position(|v| *v).unwrap_or(n / 2) };
    let ph0 = ph[ref_i];
    let row_phase: Vec<f32> =
        ph.iter().zip(&valid).map(|(v, ok)| if *ok { (v - ph0) / (2.0 * PI) } else { f32::NAN }).collect();
    let row_amp: Vec<f32> = row.iter().map(|c| c.norm() / row_max).collect();

    // ---- Fourier plane and image (the FFTs are normalised, so power is conserved)
    let (fourier_img, _) = intensity_image(&spec, n, p.log_fourier, tint);
    let (image_img, _) = intensity_image(&image, n, p.log_image, tint);
    let total: f32 = input.par_iter().map(|c| c.norm_sqr()).sum();
    let centre_row = |f: &[C]| {
        let row: Vec<f32> = f[(n / 2) * n..(n / 2 + 1) * n].iter().map(|c| c.norm_sqr()).collect();
        let m = row.iter().cloned().fold(0.0f32, f32::max).max(1e-30);
        row.into_iter().map(|v| v / m).collect::<Vec<f32>>()
    };
    let fourier_row = centre_row(&spec);
    let image_row = centre_row(&image);
    let passed_power: f32 = image.par_iter().map(|c| c.norm_sqr()).sum();

    // ---- side view: 1D angular spectrum along the centre line
    let side = side_view(p, planner, tint);

    FourierResult {
        params: p.clone(),
        input: input_img,
        fourier: fourier_img,
        image: image_img,
        side: side.image,
        side_intensity: side.intensity,
        side_nz: side.nz,
        side_m: side.m,
        side_half_mm: side.half,
        row_phase,
        row_amp,
        fourier_row,
        image_row,
        passed: if total > 0.0 { passed_power / total } else { 0.0 },
        millis: t0.elapsed().as_secs_f32() * 1000.0,
        side_aliased: side.aliased,
    }
}

struct Side {
    image: ColorImage,
    intensity: Vec<f32>,
    nz: usize,
    m: usize,
    half: f32,
    aliased: bool,
}

fn side_view(p: &FourierParams, planner: &mut FftPlanner<f32>, tint: [f32; 3]) -> Side {
    let n = p.n.clamp(64, 2048);
    let dx = p.window_mm / n as f32;
    let grid = n * 8;
    let dxs = dx / 2.0; // the 1D grid is twice as fine and four times as wide
    let lam = p.lambda_mm();
    let k = 2.0 * PI / lam;
    let (f1, f2) = (p.f1_mm, p.f2_mm);
    let xs = |i: usize| (i as f32 - grid as f32 / 2.0) * dxs;
    let fwd = planner.plan_fft_forward(grid);
    let inv = planner.plan_fft_inverse(grid);
    let kx = |j: usize| {
        let jj = if j < grid / 2 { j as f32 } else { j as f32 - grid as f32 };
        2.0 * PI * jj / (grid as f32 * dxs)
    };
    let spectrum = |field: &[C]| {
        let mut b = field.to_vec();
        fwd.process(&mut b);
        b
    };
    let propagate = |spec: &[C], dz: f32| -> Vec<C> {
        let mut b: Vec<C> = spec
            .iter()
            .enumerate()
            .map(|(j, s)| {
                let q = kx(j);
                let kz2 = k * k - q * q;
                if kz2 <= 0.0 { C::new(0.0, 0.0) } else { s * C::from_polar(1.0, (kz2.sqrt() - k) * dz) }
            })
            .collect();
        inv.process(&mut b);
        let s = 1.0 / grid as f32;
        b.iter_mut().for_each(|v| *v *= s);
        b
    };
    let lens = |field: &mut [C], f: f32| {
        for (i, v) in field.iter_mut().enumerate() {
            let x = xs(i);
            *v *= C::from_polar(1.0, -k * x * x / (2.0 * f));
        }
    };
    let u0: Vec<C> = (0..grid)
        .map(|i| {
            let x = xs(i);
            if x.abs() > p.window_mm * 0.5 { C::new(0.0, 0.0) } else { input_at(p, x, 0.0, dxs) }
        })
        .collect();
    let spec_a = spectrum(&u0);
    let mut at_l1 = propagate(&spec_a, f1);
    lens(&mut at_l1, f1);
    let spec_b = spectrum(&at_l1);
    let mut at_f = propagate(&spec_b, f1);
    for (i, v) in at_f.iter_mut().enumerate() {
        *v *= filter_at(p, xs(i), 0.0);
    }
    let spec_c = spectrum(&at_f);
    let mut at_l2 = propagate(&spec_c, f2);
    lens(&mut at_l2, f2);
    let spec_d = spectrum(&at_l2);

    // the lens phase must be sampled well enough across the beam
    let beam_half = 0.5 * p.window_mm.max(p.window_mm * f2 / f1);
    let aliased = k * beam_half / f1.min(f2) > PI / dxs;

    let total = 2.0 * f1 + 2.0 * f2;
    let nz = 480;
    let wf = lam * f1 / dx;
    let half = (0.55 * p.window_mm.max(wf).max(p.window_mm * f2 / f1)).min(grid as f32 * dxs * 0.49);
    let m = 360;
    let columns: Vec<Vec<f32>> = (0..nz)
        .into_par_iter()
        .map(|c| {
            let z = (c as f32 + 0.5) / nz as f32 * total;
            let field = if z < f1 {
                propagate(&spec_a, z)
            } else if z < 2.0 * f1 {
                propagate(&spec_b, z - f1)
            } else if z < 2.0 * f1 + f2 {
                propagate(&spec_c, z - 2.0 * f1)
            } else {
                propagate(&spec_d, z - 2.0 * f1 - f2)
            };
            // bin into m rows, top = +x; keep the maximum so narrow foci stay visible
            let mut col = vec![0.0f32; m];
            for (i, v) in field.iter().enumerate() {
                let x = xs(i);
                if x.abs() >= half {
                    continue;
                }
                let r = (((half - x) / (2.0 * half)) * m as f32) as usize;
                let r = r.min(m - 1);
                col[r] = col[r].max(v.norm_sqr());
            }
            col
        })
        .collect();
    let gmax = columns.iter().flatten().cloned().fold(0.0f32, f32::max).max(1e-30);
    let mut pixels = vec![Color32::BLACK; nz * m];
    for (c, col) in columns.iter().enumerate() {
        let cmax = col.iter().cloned().fold(0.0f32, f32::max).max(1e-30);
        for (r, v) in col.iter().enumerate() {
            let s = if p.log_side {
                (((v / gmax).max(1e-12).log10() + 5.0) / 5.0).clamp(0.0, 1.0)
            } else {
                (v / cmax).powf(1.0 / 2.2)
            };
            pixels[r * nz + c] = tinted(s, tint);
        }
    }
    Side {
        image: ColorImage::new([nz, m], pixels),
        intensity: columns.into_iter().flatten().collect(),
        nz,
        m,
        half,
        aliased,
    }
}

// ---------------------------------------------------------------- worker thread

/// Computes on a background thread; only the newest request is worked on.
pub struct Engine {
    to_worker: mpsc::Sender<FourierParams>,
    from_worker: mpsc::Receiver<FourierResult>,
    busy: bool,
    queued: Option<FourierParams>,
    last_sent: Option<FourierParams>,
}

impl Engine {
    pub fn new(repaint: impl Fn() + Send + 'static) -> Self {
        let (to_worker, rx) = mpsc::channel::<FourierParams>();
        let (tx, from_worker) = mpsc::channel();
        std::thread::Builder::new()
            .name("fourier".into())
            .spawn(move || {
                let mut planner = FftPlanner::new();
                while let Ok(mut p) = rx.recv() {
                    while let Ok(newer) = rx.try_recv() {
                        p = newer;
                    }
                    let r = compute(&p, &mut planner);
                    if tx.send(r).is_err() {
                        break;
                    }
                    repaint();
                }
            })
            .expect("fourier worker");
        Engine { to_worker, from_worker, busy: false, queued: None, last_sent: None }
    }

    /// ask for a new computation if the parameters changed
    pub fn request(&mut self, p: &FourierParams) {
        if self.last_sent.as_ref() == Some(p) || self.queued.as_ref() == Some(p) {
            return;
        }
        if self.busy {
            self.queued = Some(p.clone());
        } else {
            let _ = self.to_worker.send(p.clone());
            self.last_sent = Some(p.clone());
            self.busy = true;
        }
    }

    pub fn poll(&mut self) -> Option<FourierResult> {
        let r = self.from_worker.try_recv().ok()?;
        self.busy = false;
        if let Some(q) = self.queued.take() {
            let _ = self.to_worker.send(q.clone());
            self.last_sent = Some(q);
            self.busy = true;
        }
        Some(r)
    }

    pub fn busy(&self) -> bool {
        self.busy
    }
}

// ---------------------------------------------------------------- examples

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FourierPreset {
    DoubleSlit,
    Airy,
    Talbot,
    AbbePorter,
    LowPass,
    HighPass,
    PhaseContrast,
    Schlieren,
    SpiralPhase,
    TiltAndCurvature,
    Vortex,
}

impl FourierPreset {
    pub const ALL: [FourierPreset; 11] = [
        FourierPreset::DoubleSlit,
        FourierPreset::Airy,
        FourierPreset::Talbot,
        FourierPreset::AbbePorter,
        FourierPreset::LowPass,
        FourierPreset::HighPass,
        FourierPreset::PhaseContrast,
        FourierPreset::Schlieren,
        FourierPreset::SpiralPhase,
        FourierPreset::TiltAndCurvature,
        FourierPreset::Vortex,
    ];

    pub fn label(self) -> &'static str {
        match self {
            FourierPreset::DoubleSlit => "Double slit: Fraunhofer pattern",
            FourierPreset::Airy => "Circular hole: Airy disk",
            FourierPreset::Talbot => "Grating: diffraction orders + Talbot carpet",
            FourierPreset::AbbePorter => "Abbe–Porter: filtering a mesh",
            FourierPreset::LowPass => "Low pass: blurring",
            FourierPreset::HighPass => "High pass: edge enhancement",
            FourierPreset::PhaseContrast => "Zernike phase contrast",
            FourierPreset::Schlieren => "Schlieren (knife edge)",
            FourierPreset::SpiralPhase => "Spiral phase filter: isotropic edges",
            FourierPreset::TiltAndCurvature => "Tilted and curved wavefronts",
            FourierPreset::Vortex => "Optical vortex (doughnut beam)",
        }
    }

    pub fn setup(self) -> (FourierParams, String) {
        let mut p = FourierParams::default();
        let notes = match self {
            FourierPreset::DoubleSlit => {
                p.object = Object::DoubleSlit;
                p.size_mm = 0.1;
                p.period_mm = 0.5;
                p.length_mm = 1.0;
                "Two slits (width a = 0.1 mm, distance d = 0.5 mm). The Fourier plane shows the Fraunhofer \
                 pattern: fringes with spacing λf/d = 0.25 mm under a sinc² envelope of width 2λf/a. \
                 Change d and a, or the wavelength."
            }
            FourierPreset::Airy => {
                p.object = Object::Circle;
                p.size_mm = 0.5;
                p.zoom = [1.0, 3.0, 1.0];
                "A circular hole of diameter D = 0.5 mm: the Fourier plane shows the Airy pattern, first dark \
                 ring at r = 1.22 λf/D ≈ 0.31 mm. Hover over the pattern to measure it."
            }
            FourierPreset::Talbot => {
                p.object = Object::Grating;
                p.size_mm = 6.0;
                p.period_mm = 0.2;
                p.duty = 0.5;
                p.log_side = true;
                "A grating (period 0.2 mm) makes discrete diffraction orders in the Fourier plane, spaced \
                 λf/p. In the propagation view, the grating's image repeats itself behind the object every \
                 Talbot length z_T = 2p²/λ ≈ 126 mm (the 'Talbot carpet')."
            }
            FourierPreset::AbbePorter => {
                p.object = Object::MeshF;
                p.size_mm = 5.0;
                p.period_mm = 0.25;
                p.duty = 0.7;
                p.filter = Filter::Slit;
                p.filter_mm = 0.3;
                p.filter_angle_deg = 0.0;
                "Abbe–Porter experiment: an F seen through a wire mesh. The mesh makes a 2D grid of orders. \
                 The slit in the Fourier plane passes only one column of them: the vertical wires disappear \
                 from the image, the horizontal ones stay. Rotate the slit by 90°, or use a small pinhole \
                 (low pass) to remove the mesh completely."
            }
            FourierPreset::LowPass => {
                p.object = Object::LetterF;
                p.size_mm = 4.0;
                p.filter = Filter::LowPass;
                p.filter_mm = 0.15;
                "A pinhole in the Fourier plane removes the high spatial frequencies: the image of the F is \
                 blurred, with ringing at the edges. The smaller the hole, the coarser the image."
            }
            FourierPreset::HighPass => {
                p.object = Object::LetterF;
                p.size_mm = 4.0;
                p.filter = Filter::HighPass;
                p.filter_mm = 0.06;
                p.log_image = false;
                "A central stop in the Fourier plane removes the low spatial frequencies (the 'DC' light): \
                 only the edges of the F remain. This is the wave-optics version of the anti-aperture."
            }
            FourierPreset::PhaseContrast => {
                p.object = Object::PhaseF;
                p.size_mm = 4.0;
                p.phase = 0.4;
                p.beam = Beam::Gaussian;
                p.beam_mm = 3.5;
                p.filter = Filter::Zernike;
                p.filter_mm = 0.03;
                p.filter_phase = PI / 2.0;
                p.zoom = [1.0, 6.0, 1.0];
                "The F is completely transparent: it only delays the light by 0.4 rad (see the input field \
                 and the wavefront). Without filter it is invisible in the image. The Zernike phase dot shifts \
                 the undiffracted light by π/2, which turns the phase into brightness. Set the filter to \
                 'none' to see the F disappear."
            }
            FourierPreset::Schlieren => {
                p.object = Object::PhaseF;
                p.size_mm = 4.0;
                p.phase = 1.5;
                p.beam = Beam::Gaussian;
                p.beam_mm = 3.5;
                p.filter = Filter::KnifeEdge;
                p.filter_mm = 0.0;
                "Schlieren method: a knife edge blocks half of the Fourier plane (including half of the \
                 focused direct light). Phase gradients in the object bend light to one side, so the edges of \
                 the invisible F light up on one side only. Rotate the knife edge."
            }
            FourierPreset::SpiralPhase => {
                p.object = Object::LetterF;
                p.size_mm = 4.0;
                p.filter = Filter::Spiral;
                p.filter_phase = 1.0;
                "A spiral phase plate (phase = θ) in the Fourier plane is a radial Hilbert transform: it \
                 highlights all edges equally, whatever their direction, without blocking any light."
            }
            FourierPreset::TiltAndCurvature => {
                p.object = Object::Grating;
                p.size_mm = 5.0;
                p.period_mm = 0.25;
                p.beam = Beam::Gaussian;
                p.beam_mm = 2.5;
                p.tilt_x_mrad = 1.5;
                p.source_mm = 2000.0;
                "The grating is lit by a tilted, slightly diverging wave (a point source 2 m away). The \
                 wavefront plot shows a linear ramp (tilt) plus a parabola (curvature). The tilt shifts the \
                 whole Fourier pattern by f·θ, the curvature moves the focus away from the Fourier plane, so \
                 the orders are blurred. Set the source distance to 0 (plane wave) and the tilt to 0."
            }
            FourierPreset::Vortex => {
                p.object = Object::Vortex;
                p.size_mm = 7.0;
                p.phase = 1.0;
                p.beam = Beam::Gaussian;
                p.beam_mm = 0.6;
                p.input_view = InputView::Complex;
                p.log_fourier = false;
                p.zoom = [3.0, 10.0, 3.0];
                "A Gaussian beam through a spiral phase plate: the phase winds once around the centre \
                 (all colours meet in one point, a phase singularity). The intensity right behind the plate \
                 and in the image is still Gaussian, but in the focus (Fourier plane, far field) the light \
                 cannot be in the middle: a doughnut. Try charge 2 or 3. (Scroll in a panel to zoom.)"
            }
        };
        (p, notes.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(object: Object) -> FourierParams {
        FourierParams { object, ..Default::default() }
    }

    #[test]
    fn image_is_the_inverted_object() {
        let mut p = params(Object::LetterF);
        p.size_mm = 4.0;
        let n = p.n;
        let (input, _, image) = raw_fields(&p, &mut FftPlanner::new());
        // image(i, j) = input(-i, -j) (index n - i, with the centre at n/2)
        let mut err = 0.0;
        let mut tot = 0.0;
        for j in 1..n {
            for i in 1..n {
                let a = input[j * n + i].norm_sqr();
                let b = image[(n - j) * n + (n - i)].norm_sqr();
                err += (a - b).abs();
                tot += a;
            }
        }
        assert!(err / tot < 1e-3, "relative error {}", err / tot);
    }

    #[test]
    fn double_slit_fringe_spacing_is_lambda_f_over_d() {
        let p = params(Object::DoubleSlit);
        let n = p.n;
        let (_, spec, _) = raw_fields(&p, &mut FftPlanner::new());
        let row: Vec<f32> = spec[(n / 2) * n..(n / 2 + 1) * n].iter().map(|c| c.norm_sqr()).collect();
        // positions of local maxima near the centre
        let du = p.du();
        let peaks: Vec<f32> = (n / 2 - 40..n / 2 + 40)
            .filter(|&i| row[i] > row[i - 1] && row[i] >= row[i + 1] && row[i] > 0.2 * row[n / 2])
            .map(|i| (i as f32 - n as f32 / 2.0) * du)
            .collect();
        let spacing = (peaks.last().unwrap() - peaks[0]) / (peaks.len() - 1) as f32;
        let expected = p.lambda_mm() * p.f1_mm / p.period_mm;
        assert!((spacing - expected).abs() / expected < 0.03, "spacing {spacing} vs {expected}");
    }

    #[test]
    fn airy_first_zero() {
        let mut p = params(Object::Circle);
        p.size_mm = 0.5;
        p.n = 1024;
        p.window_mm = 16.0;
        let n = p.n;
        let (_, spec, _) = raw_fields(&p, &mut FftPlanner::new());
        let row: Vec<f32> = spec[(n / 2) * n..(n / 2 + 1) * n].iter().map(|c| c.norm_sqr()).collect();
        let first_min = (n / 2 + 1..n - 1).find(|&i| row[i] < row[i - 1] && row[i] <= row[i + 1]).unwrap();
        let r = (first_min as f32 - n as f32 / 2.0) * p.du();
        let expected = 1.22 * p.lambda_mm() * p.f1_mm / p.size_mm;
        assert!((r - expected).abs() / expected < 0.05, "first zero {r} vs {expected}");
    }

    #[test]
    fn talbot_self_image() {
        let mut p = params(Object::Grating);
        p.size_mm = 6.0;
        p.period_mm = 0.2;
        let r = compute(&p, &mut FftPlanner::new());
        // contrast of the side view column at the Talbot distance vs. half of it
        let zt = 2.0 * p.period_mm * p.period_mm / p.lambda_mm();
        let total = 2.0 * p.f1_mm + 2.0 * p.f2_mm;
        let col = |z: f32| {
            let c = ((z / total) * r.side_nz as f32) as usize;
            let v = &r.side_intensity[c * r.side_m..(c + 1) * r.side_m];
            let mid = &v[r.side_m / 2 - 20..r.side_m / 2 + 20];
            let mx = mid.iter().cloned().fold(0.0f32, f32::max);
            let mn = mid.iter().cloned().fold(f32::INFINITY, f32::min);
            (mx - mn) / (mx + mn)
        };
        assert!(col(zt) > 0.6, "contrast at z_T: {}", col(zt));
        assert!(r.millis < 2000.0);
        eprintln!("compute took {:.0} ms", r.millis);
    }
}

#[cfg(test)]
mod timing {
    use super::*;

    #[test]
    fn timing() {
        let mut planner = FftPlanner::new();
        for n in [512, 1024] {
            let p = FourierParams { n, object: Object::MeshF, size_mm: 5.0, period_mm: 0.25, duty: 0.7, ..Default::default() };
            compute(&p, &mut planner); // warm up the planner
            let r = compute(&p, &mut planner);
            eprintln!("N = {n}: {:.0} ms", r.millis);
        }
    }
}
