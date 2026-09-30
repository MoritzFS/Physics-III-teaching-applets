//! Pulse propagation through a dispersive medium (1D, scalar, linear).
//!
//! The pulse is the signal s(t) that reaches the entrance of the medium at
//! x = 0. Every frequency travels with its own wavenumber k(ω) = n(ω) ω / c:
//!
//!     ψ(x, t) = Σ_ω S(ω) exp(i (k(ω) x − ω t)),        field u = Re ψ
//!
//! In front of the medium (x < 0) there is vacuum, k = ω / c. Reflections at
//! the entrance are left out. A complex n(ω) describes absorption too.
//!
//! Units: c = 1 and the reference frequency ω₀ = 2π, so the period T₀ = 1 and
//! the vacuum wavelength λ₀ = 1. The ω grid is uniform (ω_j = j Δω), which
//! makes ψ(x, ·) at a fixed x a single FFT. ψ is periodic in t; the period is
//! chosen long enough that nothing wraps around into the time shown.

use std::f64::consts::{PI, TAU};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use eframe::egui::{Color32, ColorImage};
use rayon::prelude::*;
use rustfft::num_complex::{Complex32 as C32, Complex64 as C64};
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};

use crate::sound::{SoundExample, SoundParams};

/// reference angular frequency ω₀ (period T₀ = 1)
pub const W0: f64 = 2.0 * PI;
/// the free-form control points sit at ω = i · FREE_STEP · ω₀
pub const FREE_STEP: f64 = 0.25;
pub const FREE_POINTS: usize = 13;
/// width of the smooth edges of the rectangular pulse (T₀)
const EDGE: f64 = 1.0;

// ---------------------------------------------------------------- medium

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Model {
    Linear,
    Taylor,
    Glass,
    Resonance,
    Cutoff,
    PowerLaw,
    Free,
}

impl Model {
    pub const ALL: [Model; 7] =
        [Model::Linear, Model::Taylor, Model::Glass, Model::Resonance, Model::Cutoff, Model::PowerLaw, Model::Free];
    pub fn label(self) -> &'static str {
        match self {
            Model::Linear => "no dispersion (n constant)",
            Model::Taylor => "Taylor: phase index, group index, GVD",
            Model::Glass => "glass (Cauchy, n = A + Bω²)",
            Model::Resonance => "resonance (Lorentz oscillator)",
            Model::Cutoff => "cutoff (plasma, waveguide)",
            Model::PowerLaw => "power law ω ~ k^m (water, matter waves)",
            Model::Free => "free form: drag n(ω)",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Medium {
    pub model: Model,
    /// Linear: refractive index
    pub n0: f32,
    /// Taylor expansion of k(ω) around ω₀: phase index n = ck/ω, group index
    /// n_g = c dk/dω, GVD = c ω₀ d²k/dω², TOD = c ω₀² d³k/dω³
    pub n_p: f32,
    pub n_g: f32,
    pub gvd: f32,
    pub tod: f32,
    /// Glass: n(ω₀) and B in n = A + B (ω/ω₀)²
    pub glass_n: f32,
    pub glass_b: f32,
    /// Resonance: ε = 1 + f ω_r² / (ω_r² − ω² − iγω); ω_r and γ in units of ω₀
    pub res_w: f32,
    pub res_f: f32,
    pub res_gamma: f32,
    /// Cutoff: n² = 1 − ω_c²/ω², ω_c in units of ω₀
    pub cutoff: f32,
    /// Power law: ω ~ k^m, with phase index `power_n` at ω₀
    pub power_m: f32,
    pub power_n: f32,
    /// Free form: n at the control points
    pub free: Vec<f32>,
}

impl Default for Medium {
    fn default() -> Self {
        Medium {
            model: Model::Taylor,
            n0: 1.5,
            n_p: 1.3,
            n_g: 1.4,
            gvd: 0.5,
            tod: 0.0,
            glass_n: 1.5,
            glass_b: 0.04,
            res_w: 1.15,
            res_f: 0.08,
            res_gamma: 0.03,
            cutoff: 0.8,
            power_m: 0.5,
            power_n: 1.0,
            free: default_free(),
        }
    }
}

pub fn default_free() -> Vec<f32> {
    (0..FREE_POINTS).map(|i| 1.4 + 0.03 * (i as f32 * FREE_STEP as f32).powi(2)).collect()
}

/// Catmull–Rom spline through the control points, constant beyond the ends
pub fn free_n(pts: &[f32], u: f64) -> f64 {
    let len = pts.len();
    if len == 0 {
        return 1.0;
    }
    if len == 1 {
        return pts[0] as f64;
    }
    let s = (u / FREE_STEP).clamp(0.0, (len - 1) as f64);
    let i = (s.floor() as usize).min(len - 2);
    let t = s - i as f64;
    let p = |k: isize| pts[k.clamp(0, len as isize - 1) as usize] as f64;
    let i = i as isize;
    let (p0, p1, p2, p3) = (p(i - 1), p(i), p(i + 1), p(i + 2));
    0.5 * (2.0 * p1 + (-p0 + p2) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t)
}

impl Medium {
    /// complex wavenumber (λ₀⁻¹, c = 1) at angular frequency w > 0; Im k > 0 is absorption
    pub fn k(&self, w: f64) -> C64 {
        let u = w / W0;
        let re = |v: f64| C64::new(v, 0.0);
        match self.model {
            Model::Linear => re(self.n0 as f64 * w),
            Model::Taylor => {
                let d = u - 1.0;
                re(W0 * (self.n_p as f64 + self.n_g as f64 * d + self.gvd as f64 * d * d / 2.0 + self.tod as f64 * d * d * d / 6.0))
            }
            Model::Glass => {
                let b = self.glass_b as f64;
                re((self.glass_n as f64 - b + b * u * u) * w)
            }
            Model::Resonance => {
                let wr = self.res_w as f64;
                let chi = C64::new(self.res_f as f64 * wr * wr, 0.0) / C64::new(wr * wr - u * u, -(self.res_gamma as f64) * u);
                let n = (C64::new(1.0, 0.0) + chi).sqrt();
                // passive medium: the root with Im n ≥ 0
                let n = if n.im < 0.0 { -n } else { n };
                n * w
            }
            Model::Cutoff => {
                let e = 1.0 - (self.cutoff as f64 / u.max(1e-9)).powi(2);
                // below the cutoff k is imaginary: the wave is evanescent
                if e >= 0.0 { re(e.sqrt() * w) } else { C64::new(0.0, (-e).sqrt() * w) }
            }
            Model::PowerLaw => re(W0 * self.power_n as f64 * u.max(0.0).powf(1.0 / self.power_m.max(0.05) as f64)),
            Model::Free => re(free_n(&self.free, u) * w),
        }
    }

    /// complex refractive index c k / ω
    pub fn n(&self, w: f64) -> C64 {
        self.k(w) / w.max(1e-9)
    }

    /// group slowness Re dk/dω = 1 / v_g (in units of 1/c)
    pub fn slowness(&self, w: f64) -> f64 {
        let h = 1e-4 * W0;
        if w > 2.0 * h {
            (self.k(w + h) - self.k(w - h)).re / (2.0 * h)
        } else {
            (self.k(w + h) - self.k(w)).re / h
        }
    }

    /// Re d²k/dω² (units T₀²/λ₀)
    pub fn k2(&self, w: f64) -> f64 {
        let h = 2e-3 * W0;
        let w = w.max(2.0 * h);
        (self.k(w + h) - self.k(w) * 2.0 + self.k(w - h)).re / (h * h)
    }

    pub fn is_absorbing(&self) -> bool {
        matches!(self.model, Model::Resonance | Model::Cutoff)
    }
}

// ---------------------------------------------------------------- pulse

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Shape {
    Gaussian,
    Delta,
    Rect,
    /// only the picked frequencies
    Lines,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Pulse {
    pub shape: Shape,
    /// carrier frequency ω_c / ω₀
    pub carrier: f32,
    /// Gaussian: rms duration σ of the amplitude envelope exp(−t²/2σ²) (T₀)
    pub sigma: f32,
    /// Gaussian: chirp parameter C in exp(−(1 + iC) t²/2σ²)
    pub chirp: f32,
    /// Rect: duration (T₀)
    pub length: f32,
    /// Delta: highest frequency (ω₀)
    pub bandwidth: f32,
}

impl Default for Pulse {
    fn default() -> Self {
        Pulse { shape: Shape::Gaussian, carrier: 1.0, sigma: 2.5, chirp: 0.0, length: 12.0, bandwidth: 3.0 }
    }
}

impl Pulse {
    /// half of the duration of the pulse (T₀)
    fn half(&self) -> f64 {
        match self.shape {
            Shape::Gaussian => 4.0 * self.sigma as f64,
            Shape::Rect => 0.5 * self.length as f64 + EDGE,
            Shape::Delta => 1.5,
            Shape::Lines => 6.0,
        }
    }

    /// the frequencies that carry the pulse (rad/T₀)
    fn band(&self, picks: &[f32]) -> (f64, f64) {
        let wc = self.carrier as f64 * W0;
        let (lo, hi) = match self.shape {
            Shape::Gaussian => {
                let hw = 4.5 * (1.0 + (self.chirp as f64).powi(2)).sqrt() / self.sigma as f64;
                (wc - hw, wc + hw)
            }
            Shape::Rect => (wc - 2.5 * W0, wc + 2.5 * W0),
            Shape::Delta => (0.0, self.bandwidth as f64 * W0),
            Shape::Lines => {
                let lo = picks.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = picks.iter().cloned().fold(0.0f32, f32::max);
                if lo.is_finite() { (lo as f64 * W0 - 0.1, hi as f64 * W0 + 0.1) } else { (wc - 0.1, wc + 0.1) }
            }
        };
        (lo.max(0.01 * W0), hi.clamp(0.02 * W0, 6.0 * W0))
    }

    /// rough spectral amplitude at w (1 at the peak)
    fn weight(&self, w: f64, picks: &[f32]) -> f64 {
        let wc = self.carrier as f64 * W0;
        let inside = |b: bool| if b { 1.0 } else { 0.0 };
        match self.shape {
            Shape::Gaussian => {
                let s2 = (self.sigma as f64).powi(2) / (1.0 + (self.chirp as f64).powi(2));
                (-(w - wc).powi(2) * s2 / 2.0).exp()
            }
            Shape::Rect => inside((w - wc).abs() < 12.0 * PI / self.length.max(1.0) as f64 + 0.6 * W0),
            Shape::Delta => inside(w <= self.bandwidth as f64 * W0),
            Shape::Lines => inside(picks.iter().any(|p| (*p as f64 * W0 - w).abs() < 0.02)),
        }
    }

    /// the complex signal at the entrance, τ = t − t_c
    fn value(&self, tau: f64) -> C32 {
        let wc = self.carrier as f64 * W0;
        match self.shape {
            Shape::Gaussian => {
                let a = tau * tau / (2.0 * (self.sigma as f64).powi(2));
                C32::from_polar((-a).exp() as f32, (-(self.chirp as f64) * a - wc * tau).rem_euclid(TAU) as f32)
            }
            Shape::Rect => {
                let h = 0.5 * self.length as f64;
                let d = tau.abs() - (h - 0.5 * EDGE);
                let env = if d <= 0.0 {
                    1.0
                } else if d >= EDGE {
                    0.0
                } else {
                    0.5 * (1.0 + (PI * d / EDGE).cos())
                };
                C32::from_polar(env as f32, (-wc * tau).rem_euclid(TAU) as f32)
            }
            _ => C32::new(0.0, 0.0),
        }
    }
}

// ---------------------------------------------------------------- parameters

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tab {
    #[default]
    Waves,
    Sound,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlotView {
    /// refractive index n(ω), group index, absorption
    #[default]
    Index,
    /// ω(k)
    OmegaK,
    /// phase and group velocity
    Velocity,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DispParams {
    pub tab: Tab,
    pub pulse: Pulse,
    pub medium: Medium,
    /// length of the medium (λ₀)
    pub length: f32,
    /// vacuum in front of the medium
    pub lead_in: bool,
    /// picked frequency components (ω/ω₀)
    pub picks: Vec<f32>,
    // display
    pub plot: PlotView,
    pub show_envelope: bool,
    pub show_local_freq: bool,
    pub show_guides: bool,
    pub show_picks: bool,
    pub xt_field: bool,
    pub xt_log: bool,
    /// observer position (λ₀)
    pub observer: f32,
    /// visible part of the x range (fractions)
    pub zoom: [f32; 2],
    /// animation speed factor
    pub speed: f32,
    pub looping: bool,
    pub sound: SoundParams,
}

impl Default for DispParams {
    fn default() -> Self {
        DispParams {
            tab: Tab::Waves,
            pulse: Pulse::default(),
            medium: Medium::default(),
            length: 120.0,
            lead_in: true,
            picks: vec![],
            plot: PlotView::Index,
            show_envelope: true,
            show_local_freq: false,
            show_guides: true,
            show_picks: true,
            xt_field: false,
            xt_log: false,
            observer: 80.0,
            zoom: [0.0, 1.0],
            speed: 1.0,
            looping: false,
            sound: SoundParams::default(),
        }
    }
}

/// what the computation depends on
#[derive(Clone, Debug, PartialEq)]
pub struct PhysKey {
    pub pulse: Pulse,
    pub medium: Medium,
    pub length: f32,
    pub lead_in: bool,
    pub picks: Vec<f32>,
}

impl DispParams {
    pub fn phys(&self) -> PhysKey {
        PhysKey {
            pulse: self.pulse.clone(),
            medium: self.medium.clone(),
            length: self.length,
            lead_in: self.lead_in,
            picks: if self.pulse.shape == Shape::Lines { self.picks.clone() } else { vec![] },
        }
    }
}

// ---------------------------------------------------------------- the computation

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// exp(i z) for complex z
fn cis(z: C64) -> C32 {
    let m = (-z.im).exp();
    let a = z.re.rem_euclid(TAU);
    C32::new((m * a.cos()) as f32, (m * a.sin()) as f32)
}

pub struct Setup {
    pub id: u64,
    pub key: PhysKey,
    /// FFT size; ω_j = j Δω for j < n/2
    pub n: usize,
    pub dw: f64,
    pub dt: f64,
    /// the pulse centre passes the entrance at t_c
    pub t_c: f64,
    /// end of the animation
    pub t_end: f64,
    /// left end of the view (< 0 with vacuum in front)
    pub x_min: f64,
    pub x_max: f64,
    pub spec: Vec<C32>,
    /// k(ω_j) in the medium
    pub k: Vec<C64>,
    /// bins that matter
    pub sig: Vec<usize>,
    pub max_spec: f32,
    /// max |ψ| anywhere (the input pulse has max 1)
    pub peak: f32,
    /// carrier (rad/T₀)
    pub wc: f64,
    pub band: (f64, f64),
}

impl Setup {
    pub fn new(key: &PhysKey, planner: &mut FftPlanner<f32>) -> Setup {
        let m = &key.medium;
        let p = &key.pulse;
        let wc = p.carrier as f64 * W0;
        let half = p.half();
        let l = key.length as f64;
        let (x_min, t_c) = if key.lead_in {
            let lin = (2.0 * half + 4.0).clamp(8.0, (0.5 * l).max(8.0));
            (-lin, 0.5 * lin)
        } else {
            (0.0, half + 1.0)
        };
        let band = p.band(&key.picks);
        // the slowest group that still gets through (with a noticeable amplitude) decides how long to run
        let mut s_max = f64::MIN;
        for i in 0..=400 {
            let w = band.0 + (band.1 - band.0) * i as f64 / 400.0;
            if p.weight(w, &key.picks) * (-m.k(w).im * l).exp() < 2e-3 {
                continue;
            }
            s_max = s_max.max(m.slowness(w));
        }
        if key.pulse.shape == Shape::Lines {
            for &pk in &key.picks {
                let w = pk as f64 * W0;
                if m.k(w).im * l < 7.0 {
                    s_max = s_max.max(m.slowness(w));
                }
            }
        }
        if s_max == f64::MIN {
            s_max = 1.0;
        }
        let t_end = t_c + l * s_max.clamp(0.25, 8.0) + half + 2.0;
        let period = 2.0 * (t_end + half) + 20.0;
        let dw = TAU / period;
        let n = ((2.0 * band.1 / dw) as usize + 16).next_power_of_two().clamp(1024, 1 << 16);
        let dt = period / n as f64;

        // spectrum of the pulse at the entrance
        let mut spec = vec![C32::new(0.0, 0.0); n];
        match p.shape {
            Shape::Gaussian | Shape::Rect => {
                for (i, v) in spec.iter_mut().enumerate() {
                    let tau = (i as f64 * dt - t_c + 0.5 * period).rem_euclid(period) - 0.5 * period;
                    *v = p.value(tau);
                }
                planner.plan_fft_inverse(n).process(&mut spec);
                let s = 1.0 / n as f32;
                spec.iter_mut().for_each(|v| *v *= s);
                spec[0] = C32::new(0.0, 0.0);
            }
            Shape::Delta => {
                let wmax = p.bandwidth as f64 * W0;
                let mut sum = 0.0;
                for (j, v) in spec.iter_mut().enumerate().take(n / 2).skip(1) {
                    // smooth at both ends, so that the analytic signal has no long tails
                    let u = j as f64 * dw / wmax;
                    let low = (0.5 * PI * (j as f64 * dw / (0.2 * W0)).min(1.0)).sin().powi(2);
                    let taper = low
                        * if u < 0.6 {
                            1.0
                        } else if u < 1.0 {
                            (0.5 * PI * (u - 0.6) / 0.4).cos().powi(2)
                        } else {
                            0.0
                        };
                    sum += taper;
                    *v = C32::from_polar(taper as f32, (j as f64 * dw * t_c).rem_euclid(TAU) as f32);
                }
                let s = 1.0 / sum.max(1e-9) as f32;
                spec.iter_mut().for_each(|v| *v *= s);
            }
            Shape::Lines => {
                let picks: Vec<usize> = key.picks.iter().map(|&pk| ((pk as f64 * W0 / dw).round() as usize).clamp(1, n / 2 - 1)).collect();
                let a = 1.0 / picks.len().max(1) as f32;
                for j in picks {
                    spec[j] = C32::from_polar(a, (j as f64 * dw * t_c).rem_euclid(TAU) as f32);
                }
            }
        }
        spec.truncate(n / 2);
        let max_spec = spec.iter().map(|v| v.norm()).fold(0.0f32, f32::max);
        let sig: Vec<usize> = (1..n / 2).filter(|&j| spec[j].norm() > 1e-5 * max_spec).collect();
        let k: Vec<C64> = (0..n / 2).map(|j| if j == 0 { C64::new(0.0, 0.0) } else { m.k(j as f64 * dw) }).collect();

        let mut s = Setup {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            key: key.clone(),
            n,
            dw,
            dt,
            t_c,
            t_end,
            x_min,
            x_max: l,
            spec,
            k,
            sig,
            max_spec,
            peak: 1.0,
            wc,
            band,
        };
        // the largest amplitude anywhere, for a steady vertical scale
        let fft = planner.plan_fft_forward(n);
        let xs: Vec<f64> = (0..48).map(|i| x_min + (l - x_min) * i as f64 / 47.0).collect();
        let n_end = ((t_end / dt).ceil() as usize).min(n);
        s.peak = xs
            .par_iter()
            .map(|&x| {
                let mut buf = s.spectrum_at(x);
                fft.process(&mut buf);
                buf[..n_end].iter().map(|v| v.norm()).fold(0.0f32, f32::max)
            })
            .reduce(|| 0.0, f32::max)
            .max(1e-6);
        s
    }

    pub fn omega(&self, j: usize) -> f64 {
        j as f64 * self.dw
    }

    /// wavenumber of bin j at position x (vacuum in front of the medium)
    pub fn k_at(&self, j: usize, x: f64) -> C64 {
        if x < 0.0 { C64::new(self.omega(j), 0.0) } else { self.k[j] }
    }

    pub fn bin(&self, w: f64) -> usize {
        ((w / self.dw).round() as usize).clamp(1, self.n / 2 - 1)
    }

    /// S(ω) exp(i k x) on the full FFT grid
    pub fn spectrum_at(&self, x: f64) -> Vec<C32> {
        let mut buf = vec![C32::new(0.0, 0.0); self.n];
        for &j in &self.sig {
            buf[j] = self.spec[j] * cis(self.k_at(j, x) * x);
        }
        buf
    }

    /// ψ(x, t) at n equally spaced points from x0 to x1
    pub fn field(&self, t: f64, x0: f64, x1: f64, out: &mut [C32]) {
        let npts = out.len();
        let dx = if npts > 1 { (x1 - x0) / (npts - 1) as f64 } else { 0.0 };
        let base: Vec<(usize, C32)> =
            self.sig.iter().map(|&j| (j, self.spec[j] * cis(C64::new(-self.omega(j) * t, 0.0)))).collect();
        const CH: usize = 256;
        out.par_chunks_mut(CH).enumerate().for_each(|(ci, chunk)| {
            chunk.fill(C32::new(0.0, 0.0));
            let i0 = ci * CH;
            let xa = x0 + i0 as f64 * dx;
            // first sample inside the medium
            let split = (0..chunk.len()).find(|&i| xa + i as f64 * dx >= 0.0).unwrap_or(chunk.len());
            let (vac, med) = chunk.split_at_mut(split);
            let xm = xa + split as f64 * dx;
            for &(j, a) in &base {
                if !vac.is_empty() {
                    accumulate(vac, a, C64::new(self.omega(j), 0.0), xa, dx);
                }
                if !med.is_empty() {
                    accumulate(med, a, self.k[j], xm, dx);
                }
            }
        });
    }

    /// one frequency component S_j exp(i (k x − ω t))
    pub fn component(&self, j: usize, x: f64, t: f64) -> C32 {
        self.spec[j] * cis(self.k_at(j, x) * x - C64::new(self.omega(j) * t, 0.0))
    }

    /// phase velocity of bin j in the medium
    pub fn v_phase(&self, w: f64) -> f64 {
        w / self.key.medium.k(w).re
    }

    pub fn v_group(&self, w: f64) -> f64 {
        1.0 / self.key.medium.slowness(w)
    }

    /// speed of the envelope: the group velocity of the carrier, or Δω/Δk for a few frequencies
    pub fn group_speed(&self) -> Option<f64> {
        match self.key.pulse.shape {
            Shape::Gaussian | Shape::Rect => Some(self.v_group(self.wc)),
            Shape::Lines if self.key.picks.len() >= 2 => {
                let lo = self.key.picks.iter().cloned().fold(f32::INFINITY, f32::min) as f64 * W0;
                let hi = self.key.picks.iter().cloned().fold(0.0f32, f32::max) as f64 * W0;
                let (wa, wb) = (self.omega(self.bin(lo)), self.omega(self.bin(hi)));
                let dk = self.key.medium.k(wb).re - self.key.medium.k(wa).re;
                (dk.abs() > 1e-12).then(|| (wb - wa) / dk)
            }
            _ => None,
        }
    }

    /// where the centre of the pulse would be if it travelled with the group speed
    pub fn group_position(&self, t: f64) -> f64 {
        if t < self.t_c { t - self.t_c } else { (t - self.t_c) * self.group_speed().unwrap_or(1.0) }
    }

    /// position of the crest of frequency w with phase 2πm + phase0 at time t
    pub fn crest(&self, w: f64, phase0: f64, m: f64, t: f64) -> f64 {
        let ph = TAU * m + w * t - phase0;
        let xv = ph / w;
        if xv < 0.0 { xv } else { ph / self.key.medium.k(w).re }
    }
}

fn accumulate(out: &mut [C32], a: C32, k: C64, x: f64, dx: f64) {
    let mut p = a * cis(k * x);
    let step = cis(k * dx);
    for v in out {
        *v += p;
        p *= step;
    }
}

// ---------------------------------------------------------------- space–time diagram

#[derive(Clone)]
pub struct XtRequest {
    pub setup: Arc<Setup>,
    pub x0: f64,
    pub x1: f64,
    pub w: usize,
    pub h: usize,
    pub field: bool,
    pub log: bool,
}

impl PartialEq for XtRequest {
    fn eq(&self, o: &Self) -> bool {
        self.setup.id == o.setup.id
            && self.x0 == o.x0
            && self.x1 == o.x1
            && self.w == o.w
            && self.h == o.h
            && self.field == o.field
            && self.log == o.log
    }
}

pub struct XtResult {
    pub setup_id: u64,
    pub x0: f64,
    pub x1: f64,
    pub t_end: f64,
    pub image: ColorImage,
    pub millis: f32,
}

/// |ψ| (or Re ψ) in the x–t plane: x to the right, t downwards
pub fn space_time(r: &XtRequest, planner: &mut FftPlanner<f32>) -> XtResult {
    let start = Instant::now();
    let s = &r.setup;
    let (w, h) = (r.w.max(2), r.h.max(2));
    let fft = planner.plan_fft_forward(s.n);
    let cols: Vec<Vec<f32>> = (0..w)
        .into_par_iter()
        .map_init(
            || vec![C32::new(0.0, 0.0); fft.get_inplace_scratch_len()],
            |scratch, c| {
                let x = r.x0 + (c as f64 + 0.5) / w as f64 * (r.x1 - r.x0);
                let mut buf = s.spectrum_at(x);
                fft.process_with_scratch(&mut buf, scratch);
                let at = |t: f64| {
                    let f = t / s.dt;
                    let i = (f.floor() as usize).min(s.n - 2);
                    let a = (f - i as f64) as f32;
                    buf[i] * (1.0 - a) + buf[i + 1] * a
                };
                (0..h)
                    .map(|row| {
                        let ta = row as f64 / h as f64 * s.t_end;
                        let tb = (row + 1) as f64 / h as f64 * s.t_end;
                        let mid = 0.5 * (ta + tb);
                        if r.field {
                            at(mid).re
                        } else {
                            let (ia, ib) = ((ta / s.dt).ceil() as usize, ((tb / s.dt).floor() as usize).min(s.n - 1));
                            let mut v = at(mid).norm();
                            for b in buf.iter().take(ib + 1).skip(ia) {
                                v = v.max(b.norm());
                            }
                            v
                        }
                    })
                    .collect()
            },
        )
        .collect();
    let max = cols.iter().flatten().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-9);
    let mut image = ColorImage::new([w, h], vec![Color32::BLACK; w * h]);
    for (c, col) in cols.iter().enumerate() {
        for (row, v) in col.iter().enumerate() {
            let a = v.abs() / max;
            let b = if r.log { ((20.0 * a.max(1e-6).log10() + 40.0) / 40.0).clamp(0.0, 1.0) } else { a.powf(0.6) };
            image.pixels[row * w + c] = if r.field {
                let (cr, cg, cb) = if *v >= 0.0 { (255.0, 170.0, 60.0) } else { (70.0, 150.0, 255.0) };
                Color32::from_rgb((cr * b) as u8, (cg * b) as u8, (cb * b) as u8)
            } else {
                inferno(b)
            };
        }
    }
    XtResult { setup_id: s.id, x0: r.x0, x1: r.x1, t_end: s.t_end, image, millis: start.elapsed().as_secs_f32() * 1000.0 }
}

/// dark purple → red → yellow
pub fn inferno(v: f32) -> Color32 {
    const STOPS: [[f32; 3]; 6] = [
        [0.0, 0.0, 4.0],
        [40.0, 11.0, 84.0],
        [101.0, 21.0, 110.0],
        [178.0, 50.0, 90.0],
        [237.0, 105.0, 37.0],
        [252.0, 255.0, 164.0],
    ];
    let f = v.clamp(0.0, 1.0) * (STOPS.len() - 1) as f32;
    let i = (f.floor() as usize).min(STOPS.len() - 2);
    let a = f - i as f32;
    let c = |k: usize| (STOPS[i][k] * (1.0 - a) + STOPS[i + 1][k] * a) as u8;
    Color32::from_rgb(c(0), c(1), c(2))
}

// ---------------------------------------------------------------- examples

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispPreset {
    NoDispersion,
    PhaseVsGroup,
    Spreading,
    LongPacket,
    DeltaGlass,
    Beats,
    Compression,
    Resonance,
    Cutoff,
    DeepWater,
    Capillary,
    MatterWave,
    Free,
    Thunder,
    Whistler,
}

impl DispPreset {
    pub const ALL: [DispPreset; 15] = [
        DispPreset::NoDispersion,
        DispPreset::PhaseVsGroup,
        DispPreset::Spreading,
        DispPreset::LongPacket,
        DispPreset::DeltaGlass,
        DispPreset::Beats,
        DispPreset::Compression,
        DispPreset::Resonance,
        DispPreset::Cutoff,
        DispPreset::DeepWater,
        DispPreset::Capillary,
        DispPreset::MatterWave,
        DispPreset::Free,
        DispPreset::Thunder,
        DispPreset::Whistler,
    ];

    pub fn label(self) -> &'static str {
        match self {
            DispPreset::NoDispersion => "Gaussian wave packet without dispersion",
            DispPreset::PhaseVsGroup => "Phase velocity ≠ group velocity",
            DispPreset::Spreading => "Group velocity dispersion: the packet spreads and chirps",
            DispPreset::LongPacket => "Nearly one frequency: a long packet hardly spreads",
            DispPreset::DeltaGlass => "Delta pulse through glass: a chirp",
            DispPreset::Beats => "Two frequencies: beats and the group velocity",
            DispPreset::Compression => "Chirped pulse compression",
            DispPreset::Resonance => "Near a resonance: anomalous dispersion and absorption",
            DispPreset::Cutoff => "Waveguide / plasma: cutoff, v_p > c > v_g",
            DispPreset::DeepWater => "Deep-water waves: crests run through the group",
            DispPreset::Capillary => "Capillary ripples: crests run backwards",
            DispPreset::MatterWave => "Matter wave: an electron wave packet spreads",
            DispPreset::Free => "Draw your own n(ω)",
            DispPreset::Thunder => "Sound: thunder near and far",
            DispPreset::Whistler => "Sound: a whistler (lightning heard on a radio)",
        }
    }

    pub fn setup(self) -> (DispParams, String) {
        let mut p = DispParams::default();
        let taylor = |p: &mut DispParams, n_p: f32, n_g: f32, gvd: f32| {
            p.medium.model = Model::Taylor;
            p.medium.n_p = n_p;
            p.medium.n_g = n_g;
            p.medium.gvd = gvd;
            p.medium.tod = 0.0;
        };
        let notes = match self {
            DispPreset::NoDispersion => {
                p.medium.model = Model::Linear;
                p.medium.n0 = 1.5;
                p.pulse.sigma = 2.5;
                "A Gaussian wave packet comes from vacuum (x < 0) into a medium with a constant refractive index \
                 n = 1.5. All frequencies travel at c/n, so the packet keeps its shape. It only gets shorter by \
                 1/n, like the wavelength, because its front is slowed down first. Change n by dragging the handle \
                 in the n(ω) plot. The space–time diagram shows the packet's path: straight lines, slope = speed."
            }
            DispPreset::PhaseVsGroup => {
                taylor(&mut p, 1.0, 1.6, 0.0);
                p.pulse.sigma = 3.0;
                p.picks = vec![0.9, 1.0, 1.1];
                "The phase index (n = 1) and the group index (n_g = 1.6) differ, but n_g is the same for all \
                 frequencies (no GVD). The envelope moves at v_g = c/n_g, the crests at v_p = c/n: they appear at \
                 the back of the packet, run through it and disappear at the front. The coloured curves are three \
                 of the frequency components; the dots ride on their crests. The packet does not spread."
            }
            DispPreset::Spreading => {
                taylor(&mut p, 1.3, 1.4, 0.5);
                p.pulse.sigma = 1.5;
                p.length = 150.0;
                p.show_local_freq = true;
                "Normal dispersion: the group index grows with frequency (GVD > 0), so red travels faster than blue. \
                 A short packet has many frequencies, and they separate: the packet gets longer and chirped, with \
                 the long wavelengths in front (colour = local wavelength). Its duration grows as \
                 σ(L) = σ₀ √(1 + (L/L_D)²) with the dispersion length L_D = σ₀² / k''. Try GVD < 0."
            }
            DispPreset::LongPacket => {
                taylor(&mut p, 1.3, 1.4, 0.5);
                p.pulse.sigma = 20.0;
                p.length = 150.0;
                p.show_local_freq = true;
                "The same medium as in 'the packet spreads', but a long packet: nearly one frequency (a narrow \
                 spectrum, see the n(ω) plot). All its components have almost the same group velocity, so it \
                 hardly spreads. The price is that it is long: Δω · Δt ≥ 1/2. Make σ shorter and watch the \
                 spectrum get wider and the packet spread more."
            }
            DispPreset::DeltaGlass => {
                p.medium.model = Model::Glass;
                p.medium.glass_n = 1.5;
                p.medium.glass_b = 0.04;
                p.pulse.shape = Shape::Delta;
                p.pulse.bandwidth = 3.0;
                p.length = 100.0;
                p.show_local_freq = true;
                "A delta pulse contains all frequencies (up to 3ω₀ here) with equal weight. In glass (normal \
                 dispersion, dispersion exaggerated) each frequency arrives at its own time t = L/v_g(ω): the \
                 pulse becomes a chirp, low frequencies first. Put the observer at the end and listen: a delta \
                 becomes a 'whoop'. With 'no dispersion' the delta would stay a delta."
            }
            DispPreset::Beats => {
                taylor(&mut p, 1.0, 1.5, 0.0);
                p.pulse.shape = Shape::Lines;
                p.picks = vec![0.9, 1.1];
                p.length = 60.0;
                p.observer = 40.0;
                "Only two frequencies, ω₁ and ω₂. Their sum is a beat: a wave with an envelope that moves at \
                 (ω₂ − ω₁)/(k₂ − k₁), the group velocity in the limit of close frequencies. The crests move at \
                 the phase velocity. Click in the n(ω) plot to add or remove frequencies: with more lines the \
                 beats turn into separate packets."
            }
            DispPreset::Compression => {
                taylor(&mut p, 1.3, 1.4, 0.5);
                p.pulse.sigma = 5.0;
                p.pulse.chirp = -5.0;
                p.length = 150.0;
                p.show_local_freq = true;
                "This packet is chirped: blue at the front, red at the back (C = −5). In the medium red is faster \
                 than blue, so the back catches up: the packet gets shorter and higher until, at \
                 L = −Cσ²/((1 + C²) k'') ≈ 60 λ₀, it is as short as its spectrum allows (σ/√(1 + C²)). After \
                 that it spreads again. This is how short laser pulses are compressed (with gratings or prisms)."
            }
            DispPreset::Resonance => {
                p.medium.model = Model::Resonance;
                p.medium.res_w = 1.15;
                p.medium.res_f = 0.08;
                p.medium.res_gamma = 0.006;
                p.pulse.sigma = 5.0;
                p.length = 60.0;
                p.plot = PlotView::Index;
                "An absorption line at ω_r = 1.15 ω₀ (Lorentz oscillator). Below the resonance n grows with ω \
                 (normal dispersion) and the group index is large: slow light. In the absorption band n falls \
                 with ω (anomalous dispersion), the group velocity can exceed c or become negative, but those \
                 frequencies are absorbed (red curve: Im n). Move the carrier towards the resonance, or send a \
                 delta pulse and look for the precursors at the front."
            }
            DispPreset::Cutoff => {
                p.medium.model = Model::Cutoff;
                p.medium.cutoff = 0.8;
                p.pulse.sigma = 3.0;
                p.length = 100.0;
                p.plot = PlotView::OmegaK;
                "A waveguide (or a plasma): n² = 1 − ω_c²/ω². Above the cutoff, n < 1: the phase velocity is \
                 larger than c, but the group velocity is smaller (v_p · v_g = c²). The wavelength in the medium \
                 is longer than in vacuum. Frequencies below ω_c cannot propagate (evanescent). Look at ω(k): the \
                 secant (v_p) is steeper than the light line, the tangent (v_g) flatter."
            }
            DispPreset::DeepWater => {
                p.medium.model = Model::PowerLaw;
                p.medium.power_m = 0.5;
                p.medium.power_n = 1.0;
                p.pulse.sigma = 4.0;
                p.lead_in = false;
                p.length = 80.0;
                p.observer = 60.0;
                p.picks = vec![1.0];
                "Waves on deep water: ω = √(gk), so v_g = v_p / 2. A wave maker at x = 0 makes a group of waves. \
                 Single crests are born at the back of the group, run forward through it and die at the front \
                 (throw a stone into a pond and watch). The dot rides on one crest."
            }
            DispPreset::Capillary => {
                p.medium.model = Model::PowerLaw;
                p.medium.power_m = 1.5;
                p.medium.power_n = 1.0;
                p.pulse.sigma = 4.0;
                p.lead_in = false;
                p.length = 120.0;
                p.observer = 80.0;
                p.picks = vec![1.0];
                "Capillary ripples (short waves held by surface tension): ω² = σk³/ρ, so v_g = 1.5 v_p. Now the \
                 group is faster than the crests: crests appear at the front and fall back through the group."
            }
            DispPreset::MatterWave => {
                p.medium.model = Model::PowerLaw;
                p.medium.power_m = 2.0;
                p.medium.power_n = 1.0;
                p.pulse.sigma = 1.5;
                p.lead_in = false;
                p.length = 120.0;
                p.show_local_freq = true;
                p.observer = 90.0;
                "A free electron: ħω = p²/2m = ħ²k²/2m, so ω ~ k². The group velocity is the particle velocity \
                 ħk/m, twice the phase velocity. The packet spreads: a well localised electron has a wide spread \
                 of momenta (Heisenberg), fast parts in front (short wavelengths), slow parts behind."
            }
            DispPreset::Free => {
                p.medium.model = Model::Free;
                p.medium.free = default_free();
                p.pulse.sigma = 2.0;
                p.show_local_freq = true;
                "Drag the white points in the n(ω) plot to make your own refractive index. A flat line means no \
                 dispersion. A slope makes the group index n_g = n + ω dn/dω different from n, and curvature \
                 makes the packet spread. The dashed line is n_g."
            }
            DispPreset::Thunder => {
                p.tab = Tab::Sound;
                p.sound = SoundParams { example: SoundExample::Thunder, distance_km: 0.5, ..SoundParams::default() };
                "Thunder heard from close by starts with a sudden crack; from far away it is a soft, low rumble. \
                 A sharp crack needs high frequencies. Air absorbs them (roughly like f², the imaginary part of n), so \
                 they die out over kilometres. The rumble comes from the kilometres-long, zig-zag channel: its \
                 parts are at different distances. Real dispersion of sound in air is tiny: switch it on and \
                 exaggerate it to hear what it would do. Change the distance and press play."
            }
            DispPreset::Whistler => {
                p.tab = Tab::Sound;
                p.sound = SoundParams { example: SoundExample::Whistler, ..SoundParams::default() };
                "Lightning also sends a radio click. Part of it travels along the Earth's magnetic field through \
                 the plasma of the magnetosphere to the other hemisphere. There the dispersion is huge \
                 (whistler mode, ω ~ k²): high frequencies arrive first, so a VLF receiver (a long wire and an \
                 amplifier) plays a falling whistle about a second long. The spectrogram traces the dispersion \
                 relation: t(f) = D/√f."
            }
        };
        (p, notes.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup(p: &DispParams) -> Setup {
        Setup::new(&p.phys(), &mut FftPlanner::new())
    }

    /// |ψ(x, t)| on the FFT time grid
    fn trace(s: &Setup, x: f64) -> Vec<f32> {
        let mut buf = s.spectrum_at(x);
        FftPlanner::new().plan_fft_forward(s.n).process(&mut buf);
        buf.iter().map(|v| v.norm()).collect()
    }

    /// mean and rms width of |ψ|² in time
    fn moments(s: &Setup, v: &[f32]) -> (f64, f64) {
        let (mut a, mut b, mut c) = (0.0, 0.0, 0.0);
        for (i, e) in v.iter().enumerate() {
            let w = (*e as f64).powi(2);
            let t = i as f64 * s.dt;
            a += w;
            b += w * t;
            c += w * t * t;
        }
        let m = b / a;
        (m, (c / a - m * m).sqrt())
    }

    fn params(model: Model) -> DispParams {
        let mut p = DispParams::default();
        p.medium.model = model;
        p
    }

    #[test]
    fn without_dispersion_the_pulse_keeps_its_shape() {
        let mut p = params(Model::Linear);
        p.medium.n0 = 1.5;
        let s = setup(&p);
        let shift = (150.0 / s.dt).round() as usize;
        let a = trace(&s, 0.0);
        let b = trace(&s, shift as f64 * s.dt / 1.5);
        let err = (0..s.n - shift).map(|i| (a[i] - b[i + shift]).abs()).fold(0.0f32, f32::max);
        assert!(err < 0.02, "max deviation {err}");
    }

    #[test]
    fn the_envelope_moves_with_the_group_velocity() {
        let mut p = params(Model::Taylor);
        p.medium.n_p = 1.0;
        p.medium.n_g = 1.6;
        p.medium.gvd = 0.0;
        let s = setup(&p);
        let (t0, _) = moments(&s, &trace(&s, 0.0));
        let (t1, _) = moments(&s, &trace(&s, 100.0));
        assert!(((t1 - t0) / 100.0 - 1.6).abs() < 0.01, "group delay per λ₀: {}", (t1 - t0) / 100.0);
        assert!((s.v_phase(W0) - 1.0).abs() < 1e-6);
        assert!((s.v_group(W0) - 1.0 / 1.6).abs() < 1e-3);
    }

    #[test]
    fn gaussian_broadening_follows_the_dispersion_length() {
        let mut p = params(Model::Taylor);
        p.medium.gvd = 0.5;
        p.pulse.sigma = 1.5;
        p.length = 150.0;
        let s = setup(&p);
        let k2 = 0.5 / W0;
        let ld = 1.5f64.powi(2) / k2;
        let (_, w0) = moments(&s, &trace(&s, 0.0));
        for x in [30.0, 80.0, 150.0] {
            let (_, w) = moments(&s, &trace(&s, x));
            let expect = w0 * (1.0 + (x / ld).powi(2)).sqrt();
            assert!((w / expect - 1.0).abs() < 0.02, "x = {x}: width {w}, expected {expect}");
        }
        assert!((s.key.medium.k2(W0) - k2).abs() < 1e-4);
    }

    #[test]
    fn a_chirped_pulse_is_compressed_where_expected() {
        let (p, _) = DispPreset::Compression.setup();
        let s = setup(&p);
        let (sigma, c) = (5.0f64, -5.0f64);
        let k2 = 0.5 / W0;
        let z_min = -c * sigma * sigma / ((1.0 + c * c) * k2);
        let widths: Vec<(f64, f64)> = (0..=60).map(|i| i as f64 * 2.5).map(|x| (x, moments(&s, &trace(&s, x)).1)).collect();
        let (xb, wb) = widths.iter().cloned().fold((0.0, f64::MAX), |a, b| if b.1 < a.1 { b } else { a });
        let (_, w0) = moments(&s, &trace(&s, 0.0));
        assert!((xb - z_min).abs() < 5.0, "shortest at {xb}, expected {z_min}");
        assert!((wb / w0 - 1.0 / (1.0 + c * c).sqrt()).abs() < 0.03, "compressed by {}", wb / w0);
    }

    #[test]
    fn water_and_matter_waves() {
        let mut m = Medium { model: Model::PowerLaw, power_n: 1.0, ..Medium::default() };
        for (mm, ratio) in [(0.5, 0.5), (1.5, 1.5), (2.0, 2.0), (1.0, 1.0)] {
            m.power_m = mm;
            let vp = W0 / m.k(W0).re;
            let vg = 1.0 / m.slowness(W0);
            assert!((vg / vp - ratio).abs() < 1e-3, "m = {mm}: v_g/v_p = {}", vg / vp);
        }
    }

    #[test]
    fn waveguide_phase_times_group_velocity_is_c_squared() {
        let m = Medium { model: Model::Cutoff, cutoff: 0.8, ..Medium::default() };
        for u in [0.9, 1.0, 1.5] {
            let w = u * W0;
            let vp = w / m.k(w).re;
            let vg = 1.0 / m.slowness(w);
            assert!((vp * vg - 1.0).abs() < 1e-3);
        }
        assert!(m.k(0.5 * W0).re.abs() < 1e-9 && m.k(0.5 * W0).im > 0.0, "evanescent below the cutoff");
    }

    #[test]
    fn resonance_absorbs_and_has_anomalous_dispersion() {
        let m = Medium { model: Model::Resonance, ..Medium::default() };
        let wr = m.res_w as f64 * W0;
        assert!(m.k(wr).im > 10.0 * m.k(0.8 * W0).im);
        // n falls with ω at the resonance, rises below it
        assert!(m.n(wr + 0.01).re < m.n(wr - 0.01).re);
        assert!(m.n(0.9 * W0).re > m.n(0.8 * W0).re);
    }

    #[test]
    fn the_free_form_goes_through_its_points() {
        let pts = vec![1.0, 1.2, 1.1, 1.5, 1.4];
        for (i, v) in pts.iter().enumerate() {
            assert!((free_n(&pts, i as f64 * FREE_STEP) - *v as f64).abs() < 1e-9);
        }
        assert!((free_n(&pts, 10.0) - pts[4] as f64).abs() < 1e-9);
    }

    #[test]
    fn the_delta_chirp_does_not_wrap_around() {
        let (p, _) = DispPreset::DeltaGlass.setup();
        let s = setup(&p);
        let v = trace(&s, 100.0);
        // nothing arrives before the fastest frequency could
        // (less the width of the front near the smallest group delay)
        let first = s.t_c + 100.0 * 1.46 - 8.0;
        let early = v.iter().take((first / s.dt) as usize).fold(0.0f32, |a, b| a.max(*b));
        let total = v.iter().fold(0.0f32, |a, b| a.max(*b));
        assert!(early < 0.01 * total, "early {early}, peak {total}");
        // and the last frequencies arrive within the animation
        let late = v.iter().skip((s.t_end / s.dt) as usize).fold(0.0f32, |a, b| a.max(*b));
        assert!(late < 0.02 * total, "late {late}, peak {total}");
    }

    #[test]
    fn field_matches_the_single_components() {
        let (p, _) = DispPreset::Resonance.setup();
        let s = setup(&p);
        let t = 0.6 * s.t_end;
        let mut out = vec![C32::new(0.0, 0.0); 700];
        s.field(t, s.x_min, s.x_max, &mut out);
        for i in [0, 123, 350, 699] {
            let x = s.x_min + (s.x_max - s.x_min) * i as f64 / 699.0;
            let direct: C32 = s.sig.iter().map(|&j| s.component(j, x, t)).sum();
            assert!((direct - out[i]).norm() < 1e-3, "x = {x}: {direct} vs {}", out[i]);
        }
    }

    #[test]
    fn every_preset_makes_a_sensible_grid() {
        for pr in DispPreset::ALL {
            let (p, _) = pr.setup();
            let s = setup(&p);
            assert!(!s.sig.is_empty(), "{}", pr.label());
            assert!(s.n <= 1 << 16 && s.t_end > s.t_c, "{}", pr.label());
            assert!(s.peak.is_finite() && s.peak > 0.1, "{}: peak {}", pr.label(), s.peak);
            let json = serde_json::to_string(&p).unwrap();
            let back: DispParams = serde_json::from_str(&json).unwrap();
            assert_eq!(back, p);
        }
    }

    #[test]
    fn timing() {
        for pr in [DispPreset::DeltaGlass, DispPreset::Spreading] {
            let (p, _) = pr.setup();
            let t = Instant::now();
            let s = Arc::new(setup(&p));
            let t_setup = t.elapsed();
            let t = Instant::now();
            let mut out = vec![C32::new(0.0, 0.0); 5000];
            s.field(0.5 * s.t_end, s.x_min, s.x_max, &mut out);
            let t_field = t.elapsed();
            let r = XtRequest { setup: s.clone(), x0: s.x_min, x1: s.x_max, w: 1000, h: 500, field: false, log: false };
            let xt = space_time(&r, &mut FftPlanner::new());
            eprintln!(
                "{}: n = {}, bins {}, setup {:?}, field {:?}, x–t {:.0} ms",
                pr.label(),
                s.n,
                s.sig.len(),
                t_setup,
                t_field,
                xt.millis
            );
        }
    }
}
