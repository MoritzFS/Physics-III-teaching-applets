//! Rainbows: sunlight in spherical drops (PS02, exercise 8).
//!
//! A ray hits a drop (radius 1) at the height b, so its angle of incidence is
//! θ = arcsin b, and inside sin θ = n sin φ. After k internal reflections it
//! leaves the drop turned by the total deviation
//!
//! ```text
//! D_k(θ) = 2(θ − φ) + k(π − 2φ).
//! ```
//!
//! With the sun behind, the observer sees that light at the angle δ = π − Θ
//! from the antisolar point, where Θ ∈ [0, π] is D folded into one turn. For
//! k = 1 this is δ = 4φ − 2θ, as in the exercise. Where D is stationary (the
//! Descartes ray, cos²θ = (n² − 1)/((k+1)² − 1)) the rays pile up: the bow.
//!
//! Sunlight fills the cross-section of the drop evenly, so the rays are spread
//! evenly over the disk (weight 2b db). Every path carries the Fresnel factors
//! (1 − R)² Rᵏ, separately for s and p. The radiance seen at δ is power per
//! solid angle, ∝ Σ 2b |db/dδ| / sin δ, convolved with the disk of the sun.

use std::f64::consts::PI;
use std::sync::OnceLock;

use eframe::egui::{Color32, ColorImage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::fourier::wavelength_rgb;

pub const DEG: f64 = PI / 180.0;
/// sodium D line (µm), where n_d is given
pub const LAMBDA_D: f64 = 0.5893;
/// bin width of the brightness tables (degrees)
pub const BIN: f64 = 0.05;
/// number of bins, 0 … 180°
pub const NB: usize = 3600;
/// segments across the drop per order and wavelength
const SEGMENTS: usize = 6000;
/// the highest number of internal reflections
pub const K_MAX: usize = 4;

/// sky behind the rain and the ground (linear RGB)
pub const SKY: [f32; 3] = [0.05, 0.06, 0.08];
pub const GROUND: [f32; 3] = [0.012, 0.017, 0.01];

// ---------------------------------------------------------------- refractive index

/// Cauchy n(λ) = A + B/λ² through two points, as (n_d, B) with λ in µm
pub fn cauchy_through(l1_nm: f64, n1: f64, l2_nm: f64, n2: f64) -> (f64, f64) {
    let inv2 = |l: f64| 1.0 / (l * 1e-3 * l * 1e-3);
    let b = (n2 - n1) / (inv2(l2_nm) - inv2(l1_nm));
    (n1 + b * (1.0 / (LAMBDA_D * LAMBDA_D) - inv2(l1_nm)), b)
}

/// water at 10 °C through the values of the exercise: n(707 nm) = 1.331, n(405 nm) = 1.344
pub fn water() -> (f64, f64) {
    cauchy_through(707.0, 1.331, 405.0, 1.344)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Medium {
    Water,
    SeaWater,
    Glass,
    HighIndex,
    Custom,
}

impl Medium {
    pub const ALL: [Medium; 5] = [Medium::Water, Medium::SeaWater, Medium::Glass, Medium::HighIndex, Medium::Custom];

    pub fn label(self) -> &'static str {
        match self {
            Medium::Water => "water, 10 °C (exercise)",
            Medium::SeaWater => "sea water (35 g/kg salt)",
            Medium::Glass => "glass beads, n = 1.52",
            Medium::HighIndex => "high-index beads, n = 1.90",
            Medium::Custom => "custom",
        }
    }

    /// (n_d, Cauchy B in µm²); the glasses have Abbe numbers of about 64 and 30
    pub fn index(self) -> Option<(f64, f64)> {
        let (nw, bw) = water();
        match self {
            Medium::Water => Some((nw, bw)),
            Medium::SeaWater => Some((nw + 0.0061, bw * 1.03)),
            Medium::Glass => Some((1.52, 0.0042)),
            Medium::HighIndex => Some((1.90, 0.0157)),
            Medium::Custom => None,
        }
    }
}

// ---------------------------------------------------------------- rays

/// one path through the drop
#[derive(Clone, Copy, Debug)]
pub struct Path {
    /// angle of incidence and of refraction (rad)
    pub theta: f64,
    pub phi: f64,
    /// total deviation D (rad)
    pub dev: f64,
    /// angle from the antisolar point at which the light is seen (rad)
    pub delta: f64,
}

/// the ray at height b with k internal reflections (k = 0: straight through)
pub fn path(b: f64, n: f64, k: u32) -> Path {
    let theta = b.clamp(0.0, 1.0).asin();
    let phi = (b / n).clamp(-1.0, 1.0).asin();
    let dev = 2.0 * (theta - phi) + k as f64 * (PI - 2.0 * phi);
    Path { theta, phi, dev, delta: seen_at(dev) }
}

/// angle from the antisolar point of light that was turned by D
pub fn seen_at(dev: f64) -> f64 {
    let mut t = dev.rem_euclid(2.0 * PI);
    if t > PI {
        t = 2.0 * PI - t;
    }
    PI - t
}

/// the Descartes ray of order k ≥ 1, where the deviation is stationary; None if n > k + 1
pub fn descartes(n: f64, k: u32) -> Option<Path> {
    if k == 0 {
        return None;
    }
    let c2 = (n * n - 1.0) / (((k + 1) * (k + 1) - 1) as f64);
    if !(0.0..=1.0).contains(&c2) {
        return None;
    }
    Some(path((1.0 - c2).sqrt(), n, k))
}

/// heights b at which light of order k leaves at the angle δ (rad) from the antisolar point
pub fn rays_at(n: f64, k: u32, delta: f64) -> Vec<f64> {
    const N: usize = 600;
    let f = |b: f64| path(b, n, k).delta - delta;
    let mut out = vec![];
    let (mut b0, mut f0) = (0.0, f(0.0));
    for i in 1..=N {
        let b1 = (i as f64 / N as f64).min(1.0 - 1e-9);
        let f1 = f(b1);
        if f0 == 0.0 {
            out.push(b0);
        } else if f0 * f1 < 0.0 {
            let (mut lo, mut hi, flo) = (b0, b1, f0);
            for _ in 0..50 {
                let m = 0.5 * (lo + hi);
                if (f(m) < 0.0) == (flo < 0.0) {
                    lo = m;
                } else {
                    hi = m;
                }
            }
            out.push(0.5 * (lo + hi));
        }
        b0 = b1;
        f0 = f1;
    }
    out
}

/// Fresnel reflectance (R_s, R_p) at the angle of incidence a, going from n1 into n2
pub fn fresnel(a: f64, n1: f64, n2: f64) -> (f64, f64) {
    let st = n1 / n2 * a.sin();
    if st >= 1.0 {
        return (1.0, 1.0);
    }
    let (c, ct) = (a.cos(), (1.0 - st * st).sqrt());
    let rs = (n1 * c - n2 * ct) / (n1 * c + n2 * ct);
    let rp = (n2 * c - n1 * ct) / (n2 * c + n1 * ct);
    (rs * rs, rp * rp)
}

/// power (s, p) that follows the path of order k, out of ½ + ½ for the incoming ray
pub fn path_power(b: f64, n: f64, k: u32) -> (f64, f64) {
    let (theta, phi) = (b.clamp(0.0, 1.0).asin(), (b / n).clamp(-1.0, 1.0).asin());
    let (rs, rp) = fresnel(theta, 1.0, n);
    let (ris, rip) = fresnel(phi, n, 1.0);
    (0.5 * (1.0 - rs).powi(2) * ris.powi(k as i32), 0.5 * (1.0 - rp).powi(2) * rip.powi(k as i32))
}

/// power (s, p) reflected off the surface of the drop
pub fn reflect_power(b: f64, n: f64) -> (f64, f64) {
    let (rs, rp) = fresnel(b.clamp(0.0, 1.0).asin(), 1.0, n);
    (0.5 * rs, 0.5 * rp)
}

/// fraction of the light hitting the drop that goes into order k (None: reflected off the surface)
pub fn order_power(n: f64, k: Option<u32>) -> f64 {
    const N: usize = 4000;
    (0..N)
        .map(|i| {
            let (b0, b1) = (i as f64 / N as f64, (i + 1) as f64 / N as f64);
            let bm = 0.5 * (b0 + b1);
            let (s, p) = match k {
                Some(k) => path_power(bm, n, k),
                None => reflect_power(bm, n),
            };
            (s + p) * (b1 * b1 - b0 * b0)
        })
        .sum()
}

/// degree of polarisation (R_s-weighted) of the light along the Descartes ray of order k
pub fn polarisation(n: f64, k: u32) -> Option<f64> {
    let d = descartes(n, k)?;
    let (s, p) = path_power(d.theta.sin(), n, k);
    Some((s - p) / (s + p))
}

// ---------------------------------------------------------------- settings

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Spectrum {
    Exercise,
    Single,
    Lines,
    White,
}

impl Spectrum {
    pub const ALL: [Spectrum; 4] = [Spectrum::Exercise, Spectrum::Single, Spectrum::Lines, Spectrum::White];
    pub fn label(self) -> &'static str {
        match self {
            Spectrum::Exercise => "405 + 707 nm (exercise)",
            Spectrum::Single => "one wavelength",
            Spectrum::Lines => "8 lines, 410 … 690 nm",
            Spectrum::White => "white sunlight",
        }
    }
}

/// which rays are drawn in the drop
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DropRays {
    /// the ray at the picked height b
    Picked,
    /// the rays that leave at the picked angle, i.e. reach the eye from the picked drop
    ToEye,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RainbowParams {
    pub spectrum: Spectrum,
    /// for Spectrum::Single
    pub wavelength_nm: f32,
    /// angular diameter of the sun (0 = point source, as in the exercise)
    pub sun_diameter_deg: f32,
    pub sun_elevation_deg: f32,
    pub polarizer: bool,
    /// transmission axis, 0° = horizontal
    pub polarizer_deg: f32,
    pub medium: Medium,
    /// index at 589 nm and Cauchy B (µm²)
    pub n_d: f32,
    pub cauchy_b: f32,
    /// scales the dispersion (0 = none)
    pub dispersion: f32,
    /// light reflected off the surface
    pub reflection: bool,
    /// k = 0 … 4 internal reflections
    pub orders: [bool; K_MAX + 1],
    /// the order drawn in the drop
    pub drop_k: u32,
    pub drop_rays: DropRays,
    /// height of the picked ray
    pub b: f32,
    /// number of rays of the fan in the drop
    pub rays: u32,
    pub fresnel: bool,
    /// picked angle from the antisolar point (δ(θ) plot, sky, observer)
    pub pick_deg: f32,
    pub toward_sun: bool,
    pub brightness: f32,
    /// δ(θ) plot against the height b instead of θ
    pub plot_b: bool,
}

impl Default for RainbowParams {
    fn default() -> Self {
        RainbowPreset::Exercise.setup().0
    }
}

impl RainbowParams {
    fn base() -> Self {
        let (n_d, b) = water();
        let mut p = RainbowParams {
            spectrum: Spectrum::Exercise,
            wavelength_nm: 589.0,
            sun_diameter_deg: 0.0,
            sun_elevation_deg: 0.0,
            polarizer: false,
            polarizer_deg: 90.0,
            medium: Medium::Water,
            n_d: n_d as f32,
            cauchy_b: b as f32,
            dispersion: 1.0,
            reflection: false,
            orders: [false, true, false, false, false],
            drop_k: 1,
            drop_rays: DropRays::Picked,
            b: 0.0,
            rays: 16,
            fresnel: true,
            pick_deg: 41.5,
            toward_sun: false,
            brightness: 1.0,
            plot_b: false,
        };
        p.snap_to_descartes();
        p
    }

    pub fn n(&self, lambda_nm: f64) -> f64 {
        let l = lambda_nm * 1e-3;
        self.n_d as f64 + (self.dispersion * self.cauchy_b) as f64 * (1.0 / (l * l) - 1.0 / (LAMBDA_D * LAMBDA_D))
    }

    pub fn set_medium(&mut self, m: Medium) {
        self.medium = m;
        if let Some((n, b)) = m.index() {
            self.n_d = n as f32;
            self.cauchy_b = b as f32;
        }
    }

    /// the wavelengths that make up the light (nm)
    pub fn wavelengths(&self) -> Vec<f64> {
        match self.spectrum {
            Spectrum::Exercise => vec![405.0, 707.0],
            Spectrum::Single => vec![self.wavelength_nm as f64],
            Spectrum::Lines => (0..8).map(|i| 410.0 + 40.0 * i as f64).collect(),
            // the browser computes on one thread: coarser steps there
            Spectrum::White if cfg!(target_arch = "wasm32") => (0..=30).map(|i| 400.0 + 10.0 * i as f64).collect(),
            Spectrum::White => (0..=60).map(|i| 400.0 + 5.0 * i as f64).collect(),
        }
    }

    /// the wavelengths drawn as separate rays and curves
    pub fn shown(&self) -> Vec<f64> {
        if self.spectrum == Spectrum::White { (0..8).map(|i| 410.0 + 40.0 * i as f64).collect() } else { self.wavelengths() }
    }

    /// the order the drop and the readouts refer to: the one drawn in the drop
    pub fn main_order(&self) -> u32 {
        self.drop_k.min(K_MAX as u32)
    }

    pub fn snap_to_descartes(&mut self) {
        let k = self.main_order().max(1);
        if let Some(d) = descartes(self.n(589.0), k) {
            self.b = (d.theta.sin() as f32).min(0.999);
        }
    }

    pub fn table_key(&self) -> TableKey {
        let lambdas = self.wavelengths();
        TableKey {
            index: lambdas.iter().map(|&l| self.n(l)).collect(),
            lambdas,
            white: self.spectrum == Spectrum::White,
            sun_deg: self.sun_diameter_deg,
            reflection: self.reflection,
            orders: self.orders,
            fresnel: self.fresnel,
        }
    }

    pub fn exposure(&self, t: &Tables) -> f32 {
        self.brightness * 0.3 / t.reference.max(1e-12)
    }

    /// the colour of the sky at δ from the antisolar point (degrees), without the polariser
    pub fn color_at(&self, t: &Tables, delta_deg: f64) -> Color32 {
        let e = self.exposure(t);
        let v = t.at(delta_deg, 1.0, 1.0);
        shade(SKY, [v[0] * e, v[1] * e, v[2] * e])
    }

    /// where the bow of order k is for the wavelength λ (degrees from the antisolar point)
    pub fn bow_deg(&self, lambda_nm: f64, k: u32) -> Option<f64> {
        descartes(self.n(lambda_nm), k).map(|d| d.delta / DEG)
    }
}

// ---------------------------------------------------------------- colour

fn gauss2(l: f64, mu: f64, s1: f64, s2: f64) -> f64 {
    let t = (l - mu) / if l < mu { s1 } else { s2 };
    (-0.5 * t * t).exp()
}

/// CIE 1931 colour matching (Wyman et al. 2013 fit) as linear sRGB, as in the shader
pub fn cie_rgb(l: f64) -> [f64; 3] {
    let x = 1.056 * gauss2(l, 599.8, 37.9, 31.0) + 0.362 * gauss2(l, 442.0, 16.0, 26.7) - 0.065 * gauss2(l, 501.1, 20.4, 26.2);
    let y = 0.821 * gauss2(l, 568.8, 46.9, 40.5) + 0.286 * gauss2(l, 530.9, 16.3, 31.1);
    let z = 1.217 * gauss2(l, 437.0, 11.8, 36.0) + 0.681 * gauss2(l, 459.0, 26.0, 13.8);
    [
        3.2406 * x - 1.5372 * y - 0.4986 * z,
        -0.9689 * x + 1.8758 * y + 0.0415 * z,
        0.0557 * x - 0.2040 * y + 1.0570 * z,
    ]
}

/// display colour of a spectral line
pub fn line_color(lambda_nm: f64) -> Color32 {
    let c = wavelength_rgb(lambda_nm as f32);
    Color32::from_rgb((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8)
}

/// the colour of each wavelength in the tables, normalised so that the whole
/// spectrum adds up to white. Sunlight uses the CIE colour matching functions,
/// separate lines their display colour.
fn line_weights(lambdas: &[f64], white: bool) -> Vec<[f64; 3]> {
    let raw: Vec<[f64; 3]> = lambdas
        .iter()
        .map(|&l| {
            if white {
                cie_rgb(l)
            } else {
                let c = wavelength_rgb(l as f32);
                [c[0] as f64, c[1] as f64, c[2] as f64]
            }
        })
        .collect();
    let mut sum = [0.0; 3];
    for c in &raw {
        for q in 0..3 {
            sum[q] += c[q];
        }
    }
    let floor = 0.3 * sum.iter().cloned().fold(1e-9, f64::max);
    raw.iter().map(|c| [c[0] / sum[0].max(floor), c[1] / sum[1].max(floor), c[2] / sum[2].max(floor)]).collect()
}

fn srgb_lut() -> &'static [u8; 4096] {
    static LUT: OnceLock<[u8; 4096]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0u8; 4096];
        for (i, v) in t.iter_mut().enumerate() {
            let x = i as f32 / 4095.0;
            let s = if x <= 0.003_130_8 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
            *v = (s * 255.0 + 0.5) as u8;
        }
        t
    })
}

/// background plus tone-mapped light (both linear) as an sRGB colour
pub fn shade(bg: [f32; 3], light: [f32; 3]) -> Color32 {
    let lut = srgb_lut();
    let c = |q: usize| {
        let v = bg[q] + 1.0 - (-light[q].max(0.0)).exp();
        lut[(v.clamp(0.0, 1.0) * 4095.0) as usize]
    };
    Color32::from_rgb(c(0), c(1), c(2))
}

// ---------------------------------------------------------------- brightness tables

/// everything the tables depend on
#[derive(Clone, Debug, PartialEq)]
pub struct TableKey {
    pub lambdas: Vec<f64>,
    /// refractive index for each wavelength
    pub index: Vec<f64>,
    pub white: bool,
    pub sun_deg: f32,
    pub reflection: bool,
    pub orders: [bool; K_MAX + 1],
    pub fresnel: bool,
}

/// radiance against δ, as linear RGB, split into s (perpendicular to the
/// scattering plane, i.e. along the bow) and p
pub struct Tables {
    pub key: TableKey,
    pub s: Vec<[f32; 3]>,
    pub p: Vec<[f32; 3]>,
    /// radiance inside the primary bow of water (δ = 25 … 35°) for the exposure
    pub reference: f32,
}

fn bin_of(delta_deg: f64) -> usize {
    ((delta_deg / BIN) as isize).clamp(0, NB as isize - 1) as usize
}

/// spreads the weight w evenly over δ = d0 … d1 (degrees), i.e. δ(b) is linear within a segment
fn deposit(h: &mut [[f64; 2]], d0: f64, d1: f64, w: [f64; 2]) {
    let top = NB as f64 - 1e-9;
    let (a, b) = if d0 <= d1 { (d0 / BIN, d1 / BIN) } else { (d1 / BIN, d0 / BIN) };
    let (a, b) = (a.clamp(0.0, top), b.clamp(0.0, top));
    if b - a < 1e-9 {
        let i = a as usize;
        h[i][0] += w[0];
        h[i][1] += w[1];
        return;
    }
    let mut i = a.floor() as usize;
    while (i as f64) < b {
        let f = (((i + 1) as f64).min(b) - (i as f64).max(a)) / (b - a);
        h[i][0] += w[0] * f;
        h[i][1] += w[1] * f;
        i += 1;
    }
}

/// power per bin of δ, (s, p), for one refractive index
fn histogram(n: f64, orders: &[bool; K_MAX + 1], reflection: bool, fresnel: bool) -> Vec<[f64; 2]> {
    let mut h = vec![[0.0; 2]; NB];
    let bs = |i: usize| (i as f64 / SEGMENTS as f64).min(1.0 - 1e-12);
    let mut run = |delta: &dyn Fn(f64) -> f64, power: &dyn Fn(f64) -> (f64, f64)| {
        let mut d0 = delta(bs(0));
        for i in 0..SEGMENTS {
            let (b0, b1) = (bs(i), bs(i + 1));
            let d1 = delta(b1);
            let (s, p) = if fresnel { power(0.5 * (b0 + b1)) } else { (0.5, 0.5) };
            let area = b1 * b1 - b0 * b0;
            deposit(&mut h, d0 / DEG, d1 / DEG, [s * area, p * area]);
            d0 = d1;
        }
    };
    for k in (0..=K_MAX).filter(|&k| orders[k]) {
        run(&|b| path(b, n, k as u32).delta, &|b| path_power(b, n, k as u32));
    }
    if reflection {
        // reflected at the angle θ: turned by π − 2θ, seen at δ = 2θ
        run(&|b| 2.0 * b.asin(), &|b| reflect_power(b, n));
    }
    h
}

/// power per bin → power per solid angle
fn to_radiance(t: &mut [[f64; 3]]) {
    let floor = (0.25 * DEG).sin();
    for (i, v) in t.iter_mut().enumerate() {
        let omega = 2.0 * PI * (((i as f64 + 0.5) * BIN * DEG).sin()).max(floor) * BIN * DEG;
        for q in v.iter_mut() {
            *q /= omega;
        }
    }
}

/// the sun's disk seen across its diameter (at least a little smoothing)
fn sun_kernel(diameter_deg: f32) -> Vec<(isize, f64)> {
    let r = (diameter_deg as f64 * 0.5).max(0.075) / BIN;
    let m = r.ceil() as isize;
    let raw: Vec<(isize, f64)> = (-m..=m).map(|x| (x, (1.0 - (x as f64 / (r + 0.5)).powi(2)).max(0.0).sqrt())).collect();
    let sum: f64 = raw.iter().map(|(_, v)| v).sum();
    raw.into_iter().map(|(x, v)| (x, v / sum)).collect()
}

fn convolve(t: &[[f64; 3]], kernel: &[(isize, f64)]) -> Vec<[f32; 3]> {
    (0..NB)
        .map(|i| {
            let mut o = [0.0f64; 3];
            for &(x, w) in kernel {
                let j = i as isize + x;
                if j >= 0 && (j as usize) < NB {
                    for q in 0..3 {
                        o[q] += t[j as usize][q] * w;
                    }
                }
            }
            [o[0] as f32, o[1] as f32, o[2] as f32]
        })
        .collect()
}

/// radiance of the primary bow of water at 589 nm, averaged over δ = 25 … 35°
fn reference(fresnel: bool) -> f32 {
    let mut orders = [false; K_MAX + 1];
    orders[1] = true;
    let h = histogram(water().0, &orders, false, fresnel);
    let mut t: Vec<[f64; 3]> = h.iter().map(|v| [v[0] + v[1]; 3]).collect();
    to_radiance(&mut t);
    let (a, b) = (bin_of(25.0), bin_of(35.0));
    (t[a..b].iter().map(|v| v[0]).sum::<f64>() / (b - a) as f64) as f32
}

impl Tables {
    pub fn new(key: TableKey) -> Tables {
        let weights = line_weights(&key.lambdas, key.white);
        let hist: Vec<Vec<[f64; 2]>> = key.index.par_iter().map(|&n| histogram(n, &key.orders, key.reflection, key.fresnel)).collect();
        let mut s = vec![[0.0f64; 3]; NB];
        let mut p = vec![[0.0f64; 3]; NB];
        for (h, w) in hist.iter().zip(&weights) {
            for i in 0..NB {
                for q in 0..3 {
                    s[i][q] += h[i][0] * w[q];
                    p[i][q] += h[i][1] * w[q];
                }
            }
        }
        to_radiance(&mut s);
        to_radiance(&mut p);
        let kernel = sun_kernel(key.sun_deg);
        let reference = reference(key.fresnel);
        Tables {
            s: convolve(&s, &kernel),
            p: convolve(&p, &kernel),
            reference,
            key,
        }
    }

    /// linear RGB at δ (degrees), weighting s and p (1, 1 = unpolarised)
    pub fn at(&self, delta_deg: f64, ws: f32, wp: f32) -> [f32; 3] {
        let i = bin_of(delta_deg);
        let (s, p) = (self.s[i], self.p[i]);
        [s[0] * ws + p[0] * wp, s[1] * ws + p[1] * wp, s[2] * ws + p[2] * wp]
    }

    /// luminance at δ (degrees)
    pub fn luminance(&self, delta_deg: f64) -> f32 {
        let v = self.at(delta_deg, 1.0, 1.0);
        0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]
    }
}

// ---------------------------------------------------------------- sky

/// unit vector to the antisolar point. World frame: x towards the antisolar
/// azimuth, y up, z to the right when facing the antisolar point.
pub fn antisolar_dir(sun_el_deg: f64) -> [f64; 3] {
    let e = sun_el_deg * DEG;
    [e.cos(), -e.sin(), 0.0]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// a panorama: azimuth and elevation are linear in x and y
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyGeom {
    pub w: usize,
    pub h: usize,
    /// degrees per pixel
    pub scale: f64,
    pub el_bottom: f64,
    pub toward_sun: bool,
}

impl SkyGeom {
    pub fn new(w: usize, h: usize, p: &RainbowParams) -> SkyGeom {
        let el_bottom = -8.0;
        let top = if p.toward_sun { (p.sun_elevation_deg as f64 + 56.0).clamp(62.0, 90.0) } else { 62.0 };
        let scale = ((top - el_bottom) / h.max(1) as f64).max(200.0 / w.max(1) as f64);
        SkyGeom { w: w.max(1), h: h.max(1), scale, el_bottom, toward_sun: p.toward_sun }
    }

    /// azimuth (from the view direction, positive to the right) and elevation of a pixel position, in degrees
    pub fn az_el(&self, x: f64, y: f64) -> (f64, f64) {
        ((x - 0.5 * self.w as f64) * self.scale, self.el_bottom + (self.h as f64 - y) * self.scale)
    }

    pub fn xy(&self, az: f64, el: f64) -> (f64, f64) {
        (0.5 * self.w as f64 + az / self.scale, self.h as f64 - (el - self.el_bottom) / self.scale)
    }

    fn offset(&self) -> f64 {
        if self.toward_sun { 180.0 } else { 0.0 }
    }

    pub fn dir(&self, az: f64, el: f64) -> [f64; 3] {
        let a = (az + self.offset()) * DEG;
        let e = el.min(90.0) * DEG;
        [e.cos() * a.cos(), e.sin(), e.cos() * a.sin()]
    }

    pub fn az_el_of(&self, v: [f64; 3]) -> (f64, f64) {
        let el = v[1].clamp(-1.0, 1.0).asin() / DEG;
        let az = (v[2].atan2(v[0]) / DEG - self.offset() + 540.0).rem_euclid(360.0) - 180.0;
        (az, el)
    }
}

/// points of the circle at δ (degrees) around the antisolar point, as (az, el), in one piece per visible arc
pub fn circle(g: &SkyGeom, sun_el_deg: f64, delta_deg: f64) -> Vec<Vec<(f64, f64)>> {
    let a = antisolar_dir(sun_el_deg);
    let u1 = [0.0, 0.0, 1.0];
    let u2 = cross(a, u1);
    let (cd, sd) = ((delta_deg * DEG).cos(), (delta_deg * DEG).sin());
    let mut arcs: Vec<Vec<(f64, f64)>> = vec![vec![]];
    for i in 0..=720 {
        let psi = i as f64 / 720.0 * 2.0 * PI;
        let v: [f64; 3] = std::array::from_fn(|q| cd * a[q] + sd * (psi.cos() * u1[q] + psi.sin() * u2[q]));
        let (az, el) = g.az_el_of(v);
        let last = arcs.last_mut().unwrap();
        let jump = last.last().is_some_and(|&(az0, _)| (az - az0).abs() > 90.0);
        if jump {
            arcs.push(vec![]);
        }
        arcs.last_mut().unwrap().push((az, el));
    }
    arcs.retain(|a| a.len() > 1);
    arcs
}

pub fn sky_image(t: &Tables, p: &RainbowParams, g: &SkyGeom) -> ColorImage {
    let a = antisolar_dir(p.sun_elevation_deg as f64);
    let e = p.exposure(t);
    let pol = p.polarizer.then_some(p.polarizer_deg as f64 * DEG);
    let dim = if pol.is_some() { 0.5 } else { 1.0 };
    let sky = SKY.map(|v| v * dim);
    let ground = shade(GROUND.map(|v| v * dim), [0.0; 3]);
    // without a polariser every pixel at the same δ has the same colour
    let plain: Vec<Color32> = if pol.is_none() {
        (0..NB).map(|i| p.color_at(t, (i as f64 + 0.5) * BIN)).collect()
    } else {
        vec![]
    };
    let cols: Vec<(f64, f64)> = (0..g.w)
        .map(|x| {
            let az = (g.az_el(x as f64 + 0.5, 0.0).0 + g.offset()) * DEG;
            (az.cos(), az.sin())
        })
        .collect();
    let mut pixels = vec![Color32::BLACK; g.w * g.h];
    pixels.par_chunks_mut(g.w).enumerate().for_each(|(y, row)| {
        let el = g.az_el(0.0, y as f64 + 0.5).1;
        if el < 0.0 {
            row.fill(ground);
            return;
        }
        let (se, ce) = (el.min(90.0) * DEG).sin_cos();
        for (x, px) in row.iter_mut().enumerate() {
            let (ca, sa) = cols[x];
            let v = [ce * ca, se, ce * sa];
            let d = dot(v, a).clamp(-1.0, 1.0).acos() / DEG;
            *px = match pol {
                None => plain[bin_of(d)],
                Some(chi) => {
                    // s light oscillates perpendicular to the plane through the sun, the drop and the eye
                    let n = cross(v, a);
                    let (ws, wp) = if dot(n, n) < 1e-18 {
                        (0.5, 0.5)
                    } else {
                        let e_az = [-sa, 0.0, ca];
                        let e_el = [-se * ca, ce, -se * sa];
                        let c = (dot(n, e_el).atan2(dot(n, e_az)) - chi).cos().powi(2) as f32;
                        (c, 1.0 - c)
                    };
                    let l = t.at(d, ws, wp);
                    shade(sky, [l[0] * e, l[1] * e, l[2] * e])
                }
            };
        }
    });
    ColorImage::new([g.w, g.h], pixels)
}

// ---------------------------------------------------------------- examples

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RainbowPreset {
    Exercise,
    Caustic,
    DispersionAngle,
    WhichDrop,
    Secondary,
    RealSun,
    Polarised,
    SunHigh,
    SeaSpray,
    GlassBeads,
    HigherOrders,
    TwoRays,
}

impl RainbowPreset {
    pub const ALL: [RainbowPreset; 12] = [
        RainbowPreset::Exercise,
        RainbowPreset::Caustic,
        RainbowPreset::DispersionAngle,
        RainbowPreset::WhichDrop,
        RainbowPreset::Secondary,
        RainbowPreset::RealSun,
        RainbowPreset::Polarised,
        RainbowPreset::SunHigh,
        RainbowPreset::SeaSpray,
        RainbowPreset::GlassBeads,
        RainbowPreset::HigherOrders,
        RainbowPreset::TwoRays,
    ];

    pub fn label(self) -> &'static str {
        match self {
            RainbowPreset::Exercise => "Exercise 8: red and blue (405 / 707 nm)",
            RainbowPreset::Caustic => "Why a bow? The rays pile up",
            RainbowPreset::DispersionAngle => "Dispersion as a function of the angle",
            RainbowPreset::WhichDrop => "Which drop sends which colour?",
            RainbowPreset::Secondary => "Secondary bow and Alexander's dark band",
            RainbowPreset::RealSun => "The real sun: a 0.53° disk",
            RainbowPreset::Polarised => "The rainbow is polarised",
            RainbowPreset::SunHigh => "Sun too high: no rainbow at noon",
            RainbowPreset::SeaSpray => "Sea spray: a smaller bow",
            RainbowPreset::GlassBeads => "Glass beads and retroreflectors",
            RainbowPreset::HigherOrders => "Towards the sun: 3rd and 4th order",
            RainbowPreset::TwoRays => "Two rays, one direction (outlook: interference)",
        }
    }

    pub fn setup(self) -> (RainbowParams, String) {
        let mut p = RainbowParams::base();
        let white = |p: &mut RainbowParams| {
            p.spectrum = Spectrum::White;
            p.sun_diameter_deg = 0.53;
        };
        let notes = match self {
            RainbowPreset::Exercise => {
                "Exercise 8: a ray enters the drop at the angle θ, is refracted to φ (sin θ = n sin φ), reflected \
                 once and refracted out. It comes back at δ = 4φ − 2θ from the direction opposite to the sun. δ has \
                 a maximum where sin²θ = (4 − n²)/3: 42.4° for red (707 nm, n = 1.331) and 40.5° for violet \
                 (405 nm, n = 1.344), the dots in the δ(θ) plot. Red comes back at larger angles, so it is on the \
                 outside of the bow. Press ▶ sweep: the ray moves from the centre of the drop to its edge and \
                 back. δ grows, stalls near the maximum and falls again; it never goes into the red wedge, and \
                 'largest so far' never passes 42.37°. Drag in the drop to move the ray yourself; double-click \
                 for the ray of maximum δ (the Descartes ray)."
            }
            RainbowPreset::Caustic => {
                p.spectrum = Spectrum::Single;
                p.rays = 60;
                p.b = 0.5;
                p.pick_deg = 30.0;
                p.plot_b = true;
                "One wavelength and 60 rays. Sunlight fills the drop evenly, so the rays are evenly spaced in the \
                 height b (not in θ). Many of them leave close to the maximum of δ, where dδ/db = 0: the rays pile \
                 up there (a caustic), and that is the bright bow. No ray comes back at more than 42°, so the sky \
                 outside the bow is dark, while every angle below 42° gets some light: it is brighter inside. The \
                 yellow line in the δ(θ) plot is the picked angle; drag it. Below the bow it crosses the curve \
                 twice (two rays from each drop reach you), above the bow not at all."
            }
            RainbowPreset::DispersionAngle => {
                p.spectrum = Spectrum::Lines;
                p.dispersion = 5.0;
                p.rays = 0;
                p.b = 0.5;
                p.pick_deg = 36.0;
                "How much does dispersion separate the colours? It depends on the ray. The central ray (θ = 0) \
                 comes straight back in every colour; in real water the Descartes ray splits red and violet by \
                 1.9° and grazing rays by 2.5° (the shaded band in the δ(θ) plot; the dispersion is exaggerated \
                 ×5 here, set it back to 1). But spreading alone does not make colours: at any angle inside the bow \
                 all colours arrive, each from its own θ, and they mix to white (the strip on the right of the \
                 plot). Only between the maxima of violet and red are some colours missing, so the colours appear \
                 at the edge only, which is where the rays pile up."
            }
            RainbowPreset::WhichDrop => {
                white(&mut p);
                p.drop_rays = DropRays::ToEye;
                p.rays = 0;
                p.pick_deg = 41.8;
                "The bow is not a thing at a place. Every drop sends light back into a cone around the antisolar \
                 direction: red up to 42.4°, violet up to 40.5°. Which colour reaches your eye depends only on the \
                 angle between the drop and the antisolar point. Drops 42° away send you red, drops 40.5° away \
                 violet: the red light comes from drops higher up, so red is on the outside. Drag the drop in the \
                 side view, or click in the sky; all drops on the dashed circle look the same. Your neighbour sees \
                 a bow made of other drops."
            }
            RainbowPreset::Secondary => {
                white(&mut p);
                p.orders = [false, true, true, false, false];
                p.reflection = true;
                p.drop_k = 2;
                p.snap_to_descartes();
                p.rays = 10;
                p.pick_deg = 46.0;
                "With two internal reflections the light leaves at δ ≥ 50.4° (red) … 53.7° (violet). Now δ has a \
                 minimum, so the colours are reversed and the sky outside the secondary bow is the brighter side. \
                 Between the two bows no light of either order arrives: Alexander's dark band (Alexander of \
                 Aphrodisias, about 200 AD). It is not black because some light is reflected off the surface of \
                 the drops (switch 'reflected off the surface' off to compare). The secondary bow gets 0.6 % of the \
                 light that hits a drop, the primary 4.1 %."
            }
            RainbowPreset::RealSun => {
                white(&mut p);
                p.rays = 8;
                "The sun is not a point but a disk 0.53° across, and every point of it makes its own bow. The \
                 whole spectrum of the primary bow spans only 1.9°, so neighbouring colours overlap: the colours \
                 of a rainbow are not pure spectral colours. Compare with 'sun Ø' = 0 (a point source, as in the \
                 exercise), and try 2–3°: the colours wash out."
            }
            RainbowPreset::Polarised => {
                white(&mut p);
                p.orders = [false, true, true, false, false];
                p.reflection = true;
                p.polarizer = true;
                p.polarizer_deg = 90.0;
                "The internal reflection happens at φ ≈ 40°, close to the Brewster angle inside water (36.9°). \
                 There, light oscillating in the plane of reflection is hardly reflected, so the bow is about 92 % \
                 polarised along its arc (measured: up to 94 %). With a vertical polariser the top of the bow \
                 disappears and the sides stay; turn the polariser to move the gap. Try it on a real rainbow with \
                 polarising sunglasses."
            }
            RainbowPreset::SunHigh => {
                white(&mut p);
                p.orders = [false, true, true, false, false];
                p.reflection = true;
                p.sun_elevation_deg = 45.0;
                p.pick_deg = 50.0;
                "The bow is a circle of 42° around the antisolar point, which is as far below the horizon as the \
                 sun is above it. So the top of the bow is at 42° minus the sun's elevation, and with the sun higher \
                 than 42° there is no primary bow (the secondary survives up to 54°). In Zurich the sun reaches 66° \
                 at midsummer: in summer, rainbows are a morning and evening thing. From a plane, or with a garden \
                 hose, there are drops below the horizon too, and you can see the full circle."
            }
            RainbowPreset::SeaSpray => {
                white(&mut p);
                p.set_medium(Medium::SeaWater);
                "Sea water has a refractive index about 0.006 higher than rain water. A higher n bends the light \
                 more and the bow gets smaller: 41.2° instead of 42.0°. When rain and sea spray are both around, \
                 you can see two bows next to each other that separate towards the horizon. Switch the medium \
                 between water and sea water and watch the curve in the δ(θ) plot."
            }
            RainbowPreset::GlassBeads => {
                white(&mut p);
                p.set_medium(Medium::Glass);
                p.orders = [false, true, true, false, false];
                p.reflection = true;
                p.pick_deg = 21.0;
                p.snap_to_descartes();
                "Glass beads (n = 1.52) have a much smaller bow: 21°. You can see one on freshly painted road \
                 markings that are sprinkled with glass beads. The higher n, the smaller the bow: drag n towards 2 \
                 and the bow shrinks to the antisolar point, where the light goes straight back to where it came \
                 from. That is why high-index beads (n ≈ 1.9) are used for reflective road markings and signs. For \
                 n > 2 there is no primary bow at all."
            }
            RainbowPreset::HigherOrders => {
                white(&mut p);
                p.orders = [false, false, false, true, true];
                p.reflection = true;
                p.toward_sun = true;
                p.sun_elevation_deg = 10.0;
                p.drop_k = 3;
                p.snap_to_descartes();
                p.pick_deg = 140.0;
                p.brightness = 4.0;
                "Looking towards the sun. With three and four internal reflections the light comes out forwards: \
                 the 3rd-order bow is about 40° from the sun and the 4th about 45°, on the sun's side of the sky. \
                 They are faint: 0.2 % and 0.1 % of the light (the brightness is turned up ×4 here). Now switch on \
                 k = 0, the light that went straight through the drops: 88 % of all light, spread over up to 83° \
                 around the sun. It drowns them. That is why they were only photographed for the first time in \
                 2011 (M. Grossmann; M. Theusner)."
            }
            RainbowPreset::TwoRays => {
                p.spectrum = Spectrum::Single;
                p.drop_rays = DropRays::ToEye;
                p.rays = 0;
                p.pick_deg = 38.0;
                "Below the bow, two rays of the same colour leave a drop in the same direction: one entering closer \
                 to the centre, one closer to the edge (shown for the picked angle; drag the yellow line). Ray optics \
                 simply adds their intensities. But they are waves with different path lengths inside the drop, so \
                 they interfere. This gives faint extra bands just inside the primary bow (supernumerary bows), \
                 which Young explained in 1804 as evidence that light is a wave. The smaller the drops, the wider \
                 the bands; in fog they turn the bow white (a fogbow). That needs wave optics (Airy's theory) and \
                 is not simulated here."
            }
        };
        (p, notes.into())
    }
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn max_delta(n: f64, k: u32) -> (f64, f64) {
        // brute force over θ: the extreme of δ (maximum for k = 1, minimum for k = 2)
        let mut best = (0.0, if k == 2 { f64::MAX } else { f64::MIN });
        for i in 0..=200_000 {
            let b = (i as f64 / 200_000.0).min(1.0 - 1e-12);
            let d = path(b, n, k).delta / DEG;
            if (k == 2 && d < best.1) || (k != 2 && d > best.1) {
                best = (b.asin() / DEG, d);
            }
        }
        best
    }

    #[test]
    fn exercise_8_numbers() {
        // solution: 42.4° for red, 40.5° for blue; θ_max from sin²θ = 4/3 − n²/3
        let (th_r, d_r) = max_delta(1.331, 1);
        let (th_b, d_b) = max_delta(1.344, 1);
        assert!((d_r - 42.37).abs() < 0.01, "{d_r}");
        assert!((d_b - 40.51).abs() < 0.01, "{d_b}");
        for (n, th) in [(1.331, th_r), (1.344, th_b)] {
            let formula = ((4.0 / 3.0 - n * n / 3.0) as f64).sqrt().asin() / DEG;
            assert!((formula - th).abs() < 0.01, "{formula} {th}");
        }
        // δ = 4φ − 2θ for k = 1
        let p = path(0.7, 1.331, 1);
        assert!((p.delta - (4.0 * p.phi - 2.0 * p.theta)).abs() < 1e-12);
    }

    #[test]
    fn descartes_is_the_extreme() {
        for k in 1..=2 {
            for n in [1.331, 1.344, 1.5] {
                let d = descartes(n, k).unwrap().delta / DEG;
                let (_, brute) = max_delta(n, k);
                assert!((d - brute).abs() < 0.01, "k={k} n={n}: {d} vs {brute}");
            }
        }
        // secondary bow of the exercise's red
        assert!((descartes(1.331, 2).unwrap().delta / DEG - 50.37).abs() < 0.02);
        // n = 2: the primary bow closes to the antisolar point; above 2 it is gone
        assert!(descartes(2.0, 1).unwrap().delta.abs() < 1e-6);
        assert!(descartes(2.2, 1).is_none());
    }

    #[test]
    fn water_goes_through_the_exercise_values() {
        let mut p = RainbowParams::default();
        p.set_medium(Medium::Water);
        assert!((p.n(707.0) - 1.331).abs() < 1e-5);
        assert!((p.n(405.0) - 1.344).abs() < 1e-5);
        assert!((p.bow_deg(707.0, 1).unwrap() - 42.37).abs() < 0.01);
        // sea water: the bow is about 0.8° smaller
        let shift = p.bow_deg(589.0, 1).unwrap();
        p.set_medium(Medium::SeaWater);
        let d = shift - p.bow_deg(589.0, 1).unwrap();
        assert!((0.7..0.9).contains(&d), "{d}");
    }

    #[test]
    fn energy_is_conserved() {
        let n = 1.333;
        let mut total = order_power(n, None);
        for k in 0..60 {
            total += order_power(n, Some(k));
        }
        assert!((total - 1.0).abs() < 1e-3, "{total}");
        // the numbers quoted in the notes
        let pct = |k: Option<u32>| 100.0 * order_power(n, k);
        assert!((pct(Some(0)) - 88.4).abs() < 0.3);
        assert!((pct(Some(1)) - 4.1).abs() < 0.1);
        assert!((pct(Some(2)) - 0.6).abs() < 0.05);
        assert!((pct(Some(3)) - 0.17).abs() < 0.03);
        assert!((pct(None) - 6.6).abs() < 0.2);
    }

    #[test]
    fn the_primary_bow_is_polarised() {
        let p = polarisation(1.333, 1).unwrap();
        assert!((0.9..0.95).contains(&p), "{p}");
    }

    #[test]
    fn rays_at_an_angle() {
        let n = 1.333;
        let bow = descartes(n, 1).unwrap().delta;
        // below the bow two rays, above none
        assert_eq!(rays_at(n, 1, bow - 2.0 * DEG).len(), 2);
        assert!(rays_at(n, 1, bow + 2.0 * DEG).is_empty());
        for b in rays_at(n, 1, 30.0 * DEG) {
            assert!((path(b, n, 1).delta / DEG - 30.0).abs() < 1e-6);
        }
    }

    #[test]
    fn brighter_inside_dark_band_between() {
        let (mut p, _) = RainbowPreset::Secondary.setup();
        p.reflection = false;
        let t = Tables::new(p.table_key());
        let inside = t.luminance(35.0);
        let band = t.luminance(46.0);
        let outside = t.luminance(60.0);
        assert!(inside > 3.0 * band, "{inside} {band}");
        assert!(outside > 3.0 * band, "{outside} {band}");
        assert!(band < 1e-3 * inside);
        // the brightest part of the primary is at the bow
        let peak = (bin_of(30.0)..bin_of(45.0)).max_by(|&a, &b| t.s[a][0].total_cmp(&t.s[b][0])).unwrap();
        let peak_deg = peak as f64 * BIN;
        assert!((41.0..42.6).contains(&peak_deg), "{peak_deg}");
    }

    #[test]
    fn white_sun_tables_are_fast_enough() {
        let (p, _) = RainbowPreset::Polarised.setup();
        let t0 = Instant::now();
        let t = Tables::new(p.table_key());
        eprintln!("tables: {:.1} ms", t0.elapsed().as_secs_f32() * 1000.0);
        let g = SkyGeom::new(1400, 420, &p);
        let t0 = Instant::now();
        let img = sky_image(&t, &p, &g);
        eprintln!("sky {}×{}: {:.1} ms", img.size[0], img.size[1], t0.elapsed().as_secs_f32() * 1000.0);
    }

    #[test]
    fn sky_geometry_round_trip() {
        for toward in [false, true] {
            let mut p = RainbowParams::default();
            p.toward_sun = toward;
            let g = SkyGeom::new(800, 300, &p);
            for (az, el) in [(10.0, 20.0), (-40.0, 5.0), (70.0, 50.0)] {
                let (az2, el2) = g.az_el_of(g.dir(az, el));
                assert!((az - az2).abs() < 1e-9 && (el - el2).abs() < 1e-9);
                let (x, y) = g.xy(az, el);
                let (az3, el3) = g.az_el(x, y);
                assert!((az - az3).abs() < 1e-9 && (el - el3).abs() < 1e-9);
            }
        }
        // the top of the bow: straight up from the antisolar point
        let p = RainbowParams::default();
        let g = SkyGeom::new(800, 300, &p);
        let a = antisolar_dir(0.0);
        assert!((dot(g.dir(0.0, 42.0), a).acos() / DEG - 42.0).abs() < 1e-9);
    }

    #[test]
    fn presets_survive_json() {
        for pr in RainbowPreset::ALL {
            let (p, notes) = pr.setup();
            assert!(!notes.is_empty());
            let back: RainbowParams = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
            assert_eq!(back, p, "{}", pr.label());
        }
    }
}
