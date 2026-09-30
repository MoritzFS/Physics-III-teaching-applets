//! Sound examples for the dispersion bench: thunder and whistlers.
//!
//! Thunder: every piece of the zig-zag lightning channel sends out the same
//! N-wave (a short shock pulse) at the moment of the flash. A straight piece
//! of length ℓ reaches the listener as a box of length ℓ cos θ / c (θ between
//! the piece and the line of sight): pieces seen side-on give loud claps,
//! pieces seen end-on a smeared rumble. On the way the sound is weakened by
//! 1/r and by the absorption of air, α(f) from ISO 9613-1 (20 °C, 70 %
//! relative humidity). Absorption and delay depend on frequency, so the sum
//! is done in 1/5-octave bands with FFTs. The dispersion of sound in air
//! (from the same relaxation processes of O₂ and N₂ that absorb) is tiny; it
//! can be switched on and exaggerated.
//!
//! Whistler: the radio click of a lightning stroke, dispersed by the
//! magnetospheric plasma. Group delay t(f) = D/√f (Eckersley), or with a
//! "nose" t(f) = D/√f · (1 − f/f_H)^(−3/2), f_H = 4 f_nose.

use std::collections::HashMap;
use std::f64::consts::{PI, TAU};
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use eframe::egui::ColorImage;
use rayon::prelude::*;
use rustfft::num_complex::Complex32 as C32;
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};

use crate::dispersion::inferno;

/// sample rate of the rendered sound
pub const FS: u32 = 48000;
/// speed of sound at 20 °C (m/s)
pub const C0: f64 = 343.2;
/// length of the N-wave sent out by every piece of the channel (s)
const T_NWAVE: f64 = 0.006;
const BANDS_PER_OCTAVE: f64 = 5.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SoundExample {
    #[default]
    Thunder,
    Whistler,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SoundParams {
    pub example: SoundExample,
    /// horizontal distance from the lightning's ground point (km)
    pub distance_km: f32,
    pub absorption: bool,
    /// the long zig-zag channel (off: one point source)
    pub channel: bool,
    pub dispersion: bool,
    /// factor on the real dispersion of sound in air
    pub exaggerate: f32,
    pub seed: u32,
    /// play far thunder quieter (else every sound is normalised)
    pub true_loudness: bool,
    /// whistler dispersion D (s^½)
    pub whistler_d: f32,
    /// nose frequency (kHz), 0 = pure Eckersley law
    pub nose_khz: f32,
    /// listener in the hemisphere of the lightning (click + 2-hop echo)
    pub same_hemisphere: bool,
    pub echoes: bool,
    /// crackle of distant lightning in the background
    pub sferics: bool,
}

impl Default for SoundParams {
    fn default() -> Self {
        SoundParams {
            example: SoundExample::Thunder,
            distance_km: 1.0,
            absorption: true,
            channel: true,
            dispersion: false,
            exaggerate: 1000.0,
            seed: 7,
            true_loudness: false,
            whistler_d: 60.0,
            nose_khz: 0.0,
            same_hemisphere: false,
            echoes: true,
            sferics: true,
        }
    }
}

// ---------------------------------------------------------------- air

/// ISO 9613-1 at 20 °C, 70 % relative humidity, 101.325 kPa
struct Air {
    /// relaxation frequencies of oxygen and nitrogen (Hz)
    f_ro: f64,
    f_rn: f64,
    /// coefficients of the two relaxation terms
    c_o: f64,
    c_n: f64,
}

fn air() -> Air {
    let t = 293.15;
    let c = -6.8346 * (273.16f64 / t).powf(1.261) + 4.6151;
    // molar concentration of water vapour (%)
    let h = 70.0 * 10f64.powf(c);
    Air {
        f_ro: 24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h),
        f_rn: 9.0 + 280.0 * h,
        c_o: 0.01275 * (-2239.1 / t).exp(),
        c_n: 0.1068 * (-3352.0 / t).exp(),
    }
}

/// absorption of sound in air (dB/km)
pub fn absorption_db_per_km(f: f64) -> f64 {
    let a = air();
    let f2 = f * f;
    8.686e3 * f2 * (1.84e-11 + a.c_o / (a.f_ro + f2 / a.f_ro) + a.c_n / (a.f_rn + f2 / a.f_rn))
}

fn absorption_np_per_m(f: f64) -> f64 {
    absorption_db_per_km(f) / 8686.0
}

/// phase speed of sound (m/s), with the relaxation dispersion multiplied by `x`.
/// A relaxation with α λ = π Δ (f/f_r)/(1 + (f/f_r)²) raises the speed by Δ c₀ at high frequency.
pub fn phase_speed(f: f64, x: f64) -> f64 {
    let a = air();
    let g = |fr: f64| (f / fr).powi(2) / (1.0 + (f / fr).powi(2));
    let d_o = a.c_o * C0 / TAU;
    let d_n = a.c_n * C0 / TAU;
    C0 * (1.0 + x * (d_o * g(a.f_ro) + d_n * g(a.f_rn)))
}

/// group slowness dk/dω (s/m)
pub fn group_slowness(f: f64, x: f64) -> f64 {
    let h = f * 1e-4 + 1e-3;
    let k = |f: f64| f / phase_speed(f, x);
    (k(f + h) - k(f - h)) / (2.0 * h)
}

// ---------------------------------------------------------------- lightning channel

struct Rng(u64);

impl Rng {
    fn new(seed: u32) -> Rng {
        Rng(0x9E37_79B9_7F4A_7C15 ^ (seed as u64).wrapping_mul(0xD1B5_4A32_D192_ED03) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let u = self.uniform().max(1e-300);
        (-2.0 * u.ln()).sqrt() * (TAU * self.uniform()).cos()
    }
}

type P3 = [f64; 3];

fn add(a: P3, b: P3, s: f64) -> P3 {
    [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]]
}

fn unit(a: P3) -> P3 {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().max(1e-12);
    [a[0] / l, a[1] / l, a[2] / l]
}

fn dist(a: P3, b: P3) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// a cloud-to-ground flash: the main channel (from the ground point at the
/// origin up into the cloud, then sideways) and a few branches; metres
pub fn lightning(seed: u32) -> Vec<Vec<P3>> {
    let mut rng = Rng::new(seed);
    for _ in 0..4 {
        rng.next();
    }
    let step = 10.0;
    let turn = 0.3;
    let mut main = vec![[0.0, 0.0, 0.0]];
    let mut d: P3 = [0.0, 0.0, 1.0];
    let cloud = 3000.0 + 1500.0 * rng.uniform();
    while main.last().unwrap()[2] < cloud {
        let g = [rng.gauss(), rng.gauss(), rng.gauss()];
        d = unit(add(add(d, g, turn), [0.0, 0.0, 1.0], 0.12));
        if d[2] < 0.15 {
            d = unit([d[0], d[1], 0.15]);
        }
        main.push(add(*main.last().unwrap(), d, step));
    }
    // inside the cloud the channel runs roughly horizontally
    let az = TAU * rng.uniform();
    let dir = [az.cos(), az.sin(), 0.0];
    let n_cloud = 200 + (200.0 * rng.uniform()) as usize;
    for _ in 0..n_cloud {
        let last = *main.last().unwrap();
        let g = [rng.gauss(), rng.gauss(), rng.gauss()];
        let up = ((cloud + 600.0 - last[2]) / 2000.0).clamp(-0.3, 0.3);
        d = add(add(d, g, turn), dir, 0.15);
        d = unit([d[0], d[1], 0.6 * d[2] + up]);
        main.push(add(last, d, step));
    }
    let mut lines = vec![main.clone()];
    let low: Vec<usize> = (0..main.len()).filter(|&i| main[i][2] > 300.0 && main[i][2] < 0.95 * cloud).collect();
    let n_branch = 8 + (rng.uniform() * 7.0) as usize;
    let mut k = 0;
    while k < n_branch && !low.is_empty() {
        let start = main[low[(rng.uniform() * low.len() as f64) as usize % low.len()]];
        let len = 20 + (80.0 * rng.uniform()) as usize;
        let br = branch(&mut rng, start, len, step, turn);
        // side branches of the branch
        for _ in 0..(rng.uniform() * 3.0) as usize {
            let s2 = br[(rng.uniform() * br.len() as f64) as usize % br.len()];
            let len2 = 10 + (30.0 * rng.uniform()) as usize;
            lines.push(branch(&mut rng, s2, len2, step, turn));
        }
        lines.push(br);
        k += 1;
    }
    // zig-zag on the metre scale as well
    lines.iter().map(|l| refine(&mut rng, l, 4, 0.7)).collect()
}

/// a branch growing downwards and outwards from `start`
fn branch(rng: &mut Rng, start: P3, len: usize, step: f64, turn: f64) -> Vec<P3> {
    let mut q = start;
    let mut d = unit([rng.gauss(), rng.gauss(), -0.8]);
    let mut br = vec![q];
    for _ in 0..len {
        let g = [rng.gauss(), rng.gauss(), rng.gauss()];
        d = unit(add(add(d, g, turn), [0.0, 0.0, -1.0], 0.05));
        q = add(q, d, step);
        if q[2] < 30.0 {
            break;
        }
        br.push(q);
    }
    br
}

/// splits every piece into `parts` and moves the new points sideways by about `jitter` metres
fn refine(rng: &mut Rng, l: &[P3], parts: usize, jitter: f64) -> Vec<P3> {
    let mut out = vec![l[0]];
    for w in l.windows(2) {
        for i in 1..=parts {
            let f = i as f64 / parts as f64;
            let p = add(w[0], add(w[1], w[0], -1.0), f);
            out.push(if i == parts { w[1] } else { add(p, [rng.gauss(), rng.gauss(), rng.gauss()], jitter) });
        }
    }
    out
}

// ---------------------------------------------------------------- rendering

pub struct Rendered {
    pub params: SoundParams,
    /// mono, FS, peak ≤ 1
    pub samples: Arc<Vec<f32>>,
    /// time after the lightning of the first sample (s)
    pub t_start: f32,
    pub duration: f32,
    /// loudness (peak) compared with the same lightning heard from 100 m (dB)
    pub level_db: Option<f32>,
    /// min/max of the samples in each of ENVELOPE_COLS columns
    pub envelope: Vec<[f32; 2]>,
    pub spectrogram: ColorImage,
    /// frequency axis of the spectrogram (Hz) and whether it is logarithmic
    pub spec_range: (f32, f32),
    pub spec_log: bool,
    /// predicted arrival times: curves of (f in Hz, t after the lightning in s)
    pub arrivals: Vec<Vec<[f32; 2]>>,
    /// the channel (km)
    pub channel: Vec<Vec<[f32; 3]>>,
    /// nearest and farthest part of the channel (km)
    pub r_range: (f32, f32),
    pub millis: f32,
}

const ENVELOPE_COLS: usize = 1600;

/// peak of the thunder heard from 100 m, for the true loudness
#[derive(Default)]
pub struct RefCache(HashMap<(u32, bool, bool, bool, u32), f32>);

pub fn render(p: &SoundParams, cache: &mut RefCache, planner: &mut FftPlanner<f32>) -> Rendered {
    let start = Instant::now();
    match p.example {
        SoundExample::Thunder => {
            let (raw, t_start, r_range, lines) = thunder(p, p.distance_km as f64, planner);
            let peak = raw.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-20);
            let key = (p.seed, p.channel, p.absorption, p.dispersion, p.exaggerate.to_bits());
            let reference = *cache.0.entry(key).or_insert_with(|| {
                let (r, ..) = thunder(p, 0.1, planner);
                r.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-20)
            });
            let gain = if p.true_loudness { 0.9 / reference } else { 0.9 / peak };
            let mut samples: Vec<f32> = raw.iter().map(|v| soft_clip(v * gain)).collect();
            fade(&mut samples);
            let x = if p.dispersion { p.exaggerate as f64 } else { 0.0 };
            let freqs: Vec<f64> = (0..=60).map(|i| 20.0 * 800f64.powf(i as f64 / 60.0)).collect();
            let arrivals = vec![
                freqs.iter().map(|&f| [f as f32, (r_range.0 as f64 * 1000.0 * group_slowness(f, x)) as f32]).collect(),
                freqs.iter().map(|&f| [f as f32, (r_range.1 as f64 * 1000.0 * group_slowness(f, x)) as f32]).collect(),
            ];
            let channel = lines.iter().map(|l| l.iter().map(|q| [q[0] as f32 / 1000.0, q[1] as f32 / 1000.0, q[2] as f32 / 1000.0]).collect()).collect();
            finish(p, samples, t_start, Some(20.0 * (peak / reference).log10()), (20.0, 16000.0), true, arrivals, channel, r_range, start)
        }
        SoundExample::Whistler => {
            let (samples, t_start, arrivals) = whistler(p, planner);
            finish(p, samples, t_start, None, (0.0, 16000.0), false, arrivals, vec![], (0.0, 0.0), start)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn finish(
    p: &SoundParams,
    samples: Vec<f32>,
    t_start: f64,
    level_db: Option<f32>,
    spec_range: (f32, f32),
    spec_log: bool,
    arrivals: Vec<Vec<[f32; 2]>>,
    channel: Vec<Vec<[f32; 3]>>,
    r_range: (f32, f32),
    start: Instant,
) -> Rendered {
    let n = samples.len();
    let envelope = (0..ENVELOPE_COLS)
        .map(|c| {
            let a = c * n / ENVELOPE_COLS;
            let b = ((c + 1) * n / ENVELOPE_COLS).max(a + 1).min(n);
            samples[a..b].iter().fold([f32::MAX, f32::MIN], |m, v| [m[0].min(*v), m[1].max(*v)])
        })
        .collect();
    let spectrogram = spectrogram(&samples, 900, 240, spec_range, spec_log);
    Rendered {
        params: p.clone(),
        duration: n as f32 / FS as f32,
        samples: Arc::new(samples),
        t_start: t_start as f32,
        level_db,
        envelope,
        spectrogram,
        spec_range,
        spec_log,
        arrivals,
        channel,
        r_range,
        millis: start.elapsed().as_secs_f32() * 1000.0,
    }
}

fn soft_clip(v: f32) -> f32 {
    if v.abs() < 0.9 { v } else { v.signum() * (0.9 + 0.1 * ((v.abs() - 0.9) / 0.1).tanh()) }
}

fn fade(s: &mut [f32]) {
    let n = s.len();
    let a = (0.01 * FS as f64) as usize;
    let b = (0.2 * FS as f64) as usize;
    for i in 0..a.min(n) {
        s[i] *= i as f32 / a as f32;
    }
    for i in 0..b.min(n) {
        s[n - 1 - i] *= i as f32 / b as f32;
    }
}

/// adds `amount` spread evenly over the samples lo..hi (in samples) to the real
/// (slot 0) or imaginary (slot 1) part of buf
fn deposit(buf: &mut [C32], slot: usize, lo: f64, hi: f64, amount: f64) {
    let n = buf.len() as isize;
    let mut put = |i: isize, v: f64| {
        if i >= 0 && i < n {
            let c = &mut buf[i as usize];
            if slot == 0 { c.re += v as f32 } else { c.im += v as f32 }
        }
    };
    if hi - lo < 1.0 {
        let m = 0.5 * (lo + hi);
        let i = m.floor();
        let f = m - i;
        put(i as isize, amount * (1.0 - f));
        put(i as isize + 1, amount * f);
    } else {
        let h = amount / (hi - lo);
        for i in lo.floor() as isize..=hi.floor() as isize {
            let ov = (hi.min(i as f64 + 1.0) - lo.max(i as f64)).max(0.0);
            put(i, h * ov);
        }
    }
}

/// raw thunder at the listener: samples, time of the first sample after the flash,
/// nearest and farthest distance (km), and the channel
fn thunder(p: &SoundParams, distance_km: f64, planner: &mut FftPlanner<f32>) -> (Vec<f32>, f64, (f32, f32), Vec<Vec<P3>>) {
    let lines = if p.channel { lightning(p.seed) } else { vec![vec![[0.0, 0.0, 1500.0], [0.0, 0.0, 1510.0]]] };
    let ear = [distance_km * 1000.0, 0.0, 1.7];
    // (r at both ends, r in the middle, length)
    let segs: Vec<(f64, f64, f64, f64)> = lines
        .iter()
        .flat_map(|l| l.windows(2))
        .map(|w| {
            let mid = add(w[0], w[1], 1.0);
            (dist(w[0], ear), dist(w[1], ear), dist([mid[0] / 2.0, mid[1] / 2.0, mid[2] / 2.0], ear), dist(w[0], w[1]))
        })
        .collect();
    let r_min = segs.iter().map(|s| s.0.min(s.1)).fold(f64::MAX, f64::min);
    let r_max = segs.iter().map(|s| s.0.max(s.1)).fold(0.0, f64::max);

    let bands: Vec<f64> =
        (0..).map(|b| 20.0 * 2f64.powf(b as f64 / BANDS_PER_OCTAVE)).take_while(|f| *f < 0.45 * FS as f64).collect();
    let nb = bands.len();
    let x = if p.dispersion { p.exaggerate as f64 } else { 0.0 };
    // extra delay per metre relative to c₀, and absorption
    let extra: Vec<f64> = bands.iter().map(|&f| group_slowness(f, x) - 1.0 / C0).collect();
    let alpha: Vec<f64> = bands.iter().map(|&f| if p.absorption { absorption_np_per_m(f) } else { 0.0 }).collect();
    let e_min = extra.iter().cloned().fold(0.0, f64::min);
    let e_max = extra.iter().cloned().fold(0.0, f64::max);
    let t_s = r_min * (1.0 / C0 + e_min) - 0.15;
    let t_e = r_max * (1.0 / C0 + e_max) + 1.0;
    let n = (((t_e - t_s) * FS as f64) as usize).next_power_of_two().clamp(1 << 14, 1 << 21);
    let half = n / 2;

    let amp = |b: usize, s: &(f64, f64, f64, f64)| s.3 / s.2 * (-alpha[b] * s.2).exp();
    let loudest: Vec<f64> = (0..nb).map(|b| segs.iter().map(|s| amp(b, s)).fold(0.0, f64::max)).collect();
    let top = loudest.iter().cloned().fold(0.0, f64::max);
    let active: Vec<usize> = (0..nb).filter(|&b| loudest[b] > 1e-5 * top).collect();

    let fft = planner.plan_fft_forward(n);
    let mut nwave = vec![C32::new(0.0, 0.0); n];
    let len = (T_NWAVE * FS as f64) as usize;
    for (i, v) in nwave.iter_mut().take(len).enumerate() {
        v.re = 1.0 - 2.0 * (i as f32 + 0.5) / len as f32;
    }
    fft.process(&mut nwave);

    let df = FS as f64 / n as f64;
    let window = |b: usize, q: usize| -> f32 {
        let f = q as f64 * df;
        let u = if f > 0.0 { BANDS_PER_OCTAVE * (f / bands[b]).log2() } else { f64::NEG_INFINITY };
        if (b == 0 && u <= 0.0) || (b == nb - 1 && u >= 0.0) {
            1.0
        } else if u.abs() >= 1.0 {
            0.0
        } else {
            (0.5 * PI * u).cos().powi(2) as f32
        }
    };
    let bins = |b: usize| -> (usize, usize) {
        let lo = if b == 0 { 0 } else { (bands[b - 1] / df).floor() as usize };
        let hi = if b == nb - 1 { half } else { ((bands[b + 1] / df).ceil() as usize).min(half) };
        (lo, hi)
    };
    let pairs: Vec<Vec<usize>> = active.chunks(2).map(|c| c.to_vec()).collect();
    let spectrum = pairs
        .par_iter()
        .fold(
            || vec![C32::new(0.0, 0.0); half + 1],
            |mut acc, pair| {
                let mut buf = vec![C32::new(0.0, 0.0); n];
                for (slot, &b) in pair.iter().enumerate() {
                    let slow = 1.0 / C0 + extra[b];
                    for s in &segs {
                        let ta = (s.0 * slow - t_s) * FS as f64;
                        let tb = (s.1 * slow - t_s) * FS as f64;
                        deposit(&mut buf, slot, ta.min(tb), ta.max(tb), amp(b, s));
                    }
                }
                fft.process(&mut buf);
                for (slot, &b) in pair.iter().enumerate() {
                    let (lo, hi) = bins(b);
                    for q in lo..=hi {
                        let z = buf[q];
                        let zc = buf[(n - q) % n].conj();
                        let part = if slot == 0 { (z + zc) * 0.5 } else { (z - zc) * C32::new(0.0, -0.5) };
                        acc[q] += part * nwave[q] * window(b, q);
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![C32::new(0.0, 0.0); half + 1],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                a
            },
        );
    let out = hermitian_to_real(&spectrum, n, planner);
    let n_keep = (((t_e - t_s) * FS as f64) as usize).min(n);
    (out[..n_keep].to_vec(), t_s, (r_min as f32 / 1000.0, r_max as f32 / 1000.0), lines)
}

/// real signal from the positive half of its spectrum (bins 0..=n/2)
fn hermitian_to_real(half_spec: &[C32], n: usize, planner: &mut FftPlanner<f32>) -> Vec<f32> {
    let mut full = vec![C32::new(0.0, 0.0); n];
    full[..=n / 2].copy_from_slice(&half_spec[..=n / 2]);
    full[0].im = 0.0;
    full[n / 2].im = 0.0;
    for q in 1..n / 2 {
        full[n - q] = half_spec[q].conj();
    }
    planner.plan_fft_inverse(n).process(&mut full);
    let s = 1.0 / n as f32;
    full.iter().map(|v| v.re * s).collect()
}

/// group delay per hop of a whistler (s)
pub fn whistler_delay(f: f64, d: f64, nose_khz: f64) -> f64 {
    let base = d / f.max(1.0).sqrt();
    if nose_khz > 0.0 {
        let fh = 4000.0 * nose_khz;
        if f >= fh { f64::INFINITY } else { base * (1.0 - f / fh).powf(-1.5) }
    } else {
        base
    }
}

/// hops and their amplitudes; hop 0 is the undispersed click
fn hops(p: &SoundParams) -> Vec<(u32, f32)> {
    let mut h = if p.same_hemisphere { vec![(0, 1.0), (2, 1.0), (4, 0.5), (6, 0.25)] } else { vec![(1, 1.0), (3, 0.45), (5, 0.2)] };
    if !p.echoes {
        h.truncate(if p.same_hemisphere { 2 } else { 1 });
    }
    h
}

const T_FLASH: f64 = 0.4;

fn whistler(p: &SoundParams, planner: &mut FftPlanner<f32>) -> (Vec<f32>, f64, Vec<Vec<[f32; 2]>>) {
    let d = p.whistler_d as f64;
    let nose = p.nose_khz as f64;
    let hops = hops(p);
    let h_max = hops.iter().map(|h| h.0).max().unwrap_or(1).max(1) as f64;
    let t_total = (T_FLASH + h_max * whistler_delay(500.0, d, 0.0) + 0.8).min(40.0);
    let n = ((t_total * FS as f64) as usize).next_power_of_two().min(1 << 21);
    let half = n / 2;
    let df = FS as f64 / n as f64;
    let fh = if nose > 0.0 { 4000.0 * nose } else { f64::INFINITY };
    // phase φ(f) = 2π ∫ t(f) df, starting with 2D√f for the first bin
    let mut phase = vec![0.0; half + 1];
    for q in 1..=half {
        let f = q as f64 * df;
        phase[q] = if q == 1 {
            TAU * 2.0 * d * f.sqrt()
        } else {
            let fa = f - df;
            phase[q - 1] + TAU * df * 0.5 * (whistler_delay(fa, d, nose).min(1e3) + whistler_delay(f, d, nose).min(1e3))
        };
    }
    let mut spec = vec![C32::new(0.0, 0.0); half + 1];
    for (q, v) in spec.iter_mut().enumerate().skip(1) {
        let f = q as f64 * df;
        if f >= 0.98 * fh {
            continue;
        }
        let src = (1.0 - (-(f / 600.0).powi(2)).exp()) * (-(f / 9000.0).powi(2)).exp();
        let mut sum = C32::new(0.0, 0.0);
        for &(h, a) in hops.iter().filter(|h| h.0 > 0) {
            let damp = (-(h as f64) * f / 30000.0).exp();
            let ph = -(h as f64) * phase[q] - TAU * f * T_FLASH;
            sum += C32::from_polar((a as f64 * damp) as f32, ph.rem_euclid(TAU) as f32);
        }
        *v = sum * src as f32;
    }
    let mut out = hermitian_to_real(&spec, n, planner);
    let peak = out.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-20);
    out.iter_mut().for_each(|v| *v *= 0.7 / peak);
    let mut rng = Rng::new(p.seed ^ 0xABCD);
    let click = |out: &mut [f32], t: f64, a: f32, rng: &mut Rng| {
        let f = 4000.0 + 3000.0 * rng.uniform();
        let i0 = (t * FS as f64) as usize;
        for i in 0..(0.003 * FS as f64) as usize {
            if let Some(v) = out.get_mut(i0 + i) {
                let tt = i as f64 / FS as f64;
                *v += a * ((-tt / 0.0004).exp() * (TAU * f * tt).sin()) as f32;
            }
        }
    };
    if hops.iter().any(|h| h.0 == 0) {
        click(&mut out, T_FLASH, 0.9, &mut rng);
    }
    if p.sferics {
        let mut t = 0.0;
        let end = out.len() as f64 / FS as f64;
        loop {
            t += -(rng.uniform().max(1e-9)).ln() / 25.0;
            if t > end {
                break;
            }
            let a = (0.08 / rng.uniform().max(0.05).powf(0.8)).min(0.6) as f32;
            click(&mut out, t, a, &mut rng);
        }
    }
    out.iter_mut().for_each(|v| *v = soft_clip(*v));
    out.truncate(((t_total * FS as f64) as usize).min(n));
    fade(&mut out);
    let top = if nose > 0.0 { (0.95 * fh).min(15000.0) } else { 15000.0 };
    let arrivals = hops
        .iter()
        .filter(|h| h.0 > 0)
        .map(|&(h, _)| {
            (0..=80)
                .map(|i| 250.0 + (top - 250.0) * i as f64 / 80.0)
                .map(|f| [f as f32, (h as f64 * whistler_delay(f, d, nose)) as f32])
                .collect()
        })
        .collect();
    // time axis starts at the flash
    (out, -T_FLASH, arrivals)
}

/// |STFT|² in dB as an image: time to the right, frequency upwards
fn spectrogram(x: &[f32], cols: usize, rows: usize, range: (f32, f32), log: bool) -> ColorImage {
    let win = 2048;
    let n = x.len();
    let hann: Vec<f32> = (0..win).map(|i| (PI * i as f64 / win as f64).sin().powi(2) as f32).collect();
    let hop = n.saturating_sub(win).max(1) as f64 / cols as f64;
    let fft = FftPlanner::<f32>::new().plan_fft_forward(win);
    let (f0, f1) = (range.0.max(1.0) as f64, range.1 as f64);
    let row_bins: Vec<(usize, usize)> = (0..rows)
        .map(|r| {
            let a = (rows - 1 - r) as f64 / rows as f64;
            let b = (rows - r) as f64 / rows as f64;
            let f = |s: f64| if log { f0 * (f1 / f0).powf(s) } else { range.0 as f64 + (f1 - range.0 as f64) * s };
            let k = |f: f64| f * win as f64 / FS as f64;
            let lo = k(f(a)).round() as usize;
            let hi = (k(f(b)).round() as usize).max(lo);
            (lo.min(win / 2), hi.min(win / 2))
        })
        .collect();
    let columns: Vec<Vec<f32>> = (0..cols)
        .into_par_iter()
        .map(|c| {
            let s = (c as f64 * hop) as usize;
            let mut buf: Vec<C32> = (0..win).map(|i| C32::new(x.get(s + i).copied().unwrap_or(0.0) * hann[i], 0.0)).collect();
            fft.process(&mut buf);
            row_bins
                .iter()
                .map(|&(lo, hi)| {
                    let p = buf[lo..=hi].iter().map(|v| v.norm_sqr()).fold(0.0f32, f32::max);
                    10.0 * (p + 1e-20).log10()
                })
                .collect()
        })
        .collect();
    let max = columns.iter().flatten().cloned().fold(f32::MIN, f32::max);
    let mut img = ColorImage::new([cols, rows], vec![Default::default(); cols * rows]);
    for (c, col) in columns.iter().enumerate() {
        for (r, db) in col.iter().enumerate() {
            img.pixels[r * cols + c] = inferno(((db - (max - 70.0)) / 70.0).clamp(0.0, 1.0));
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_absorption_values() {
        // ISO 9613-1 table values for 20 °C, 70 % RH: ~5 dB/km at 1 kHz, ~23 at 4 kHz
        let a1 = absorption_db_per_km(1000.0);
        let a4 = absorption_db_per_km(4000.0);
        assert!((a1 - 5.0).abs() < 0.3, "{a1}");
        assert!((a4 - 23.0).abs() < 2.0, "{a4}");
        assert!(absorption_db_per_km(100.0) < 0.4);
    }

    #[test]
    fn real_air_dispersion_is_tiny() {
        let dc = phase_speed(4000.0, 1.0) - phase_speed(50.0, 1.0);
        assert!(dc > 0.0 && dc < 0.1, "{dc} m/s");
        // over 10 km the arrival times differ by less than 10 ms
        let dt = 1e4 * (group_slowness(50.0, 1.0) - group_slowness(4000.0, 1.0));
        assert!(dt > 0.0 && dt < 0.01, "{dt} s");
    }

    #[test]
    fn the_channel_reaches_the_cloud() {
        let l = lightning(7);
        let top = l[0].iter().map(|p| p[2]).fold(0.0, f64::max);
        assert!(top > 3000.0);
        assert!(l.len() >= 4);
    }

    /// first sample above 1 % of the peak
    fn onset(x: &[f32]) -> usize {
        let peak = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        x.iter().position(|v| v.abs() > 0.01 * peak).unwrap()
    }

    /// energy in 1–4 kHz compared with 50–200 Hz (dB), in the first second of the sound
    fn brightness(x: &[f32]) -> f64 {
        let n = 1 << 15;
        let first = onset(x);
        let mut buf: Vec<C32> = (0..n).map(|i| C32::new(x.get(first + i).copied().unwrap_or(0.0), 0.0)).collect();
        FftPlanner::new().plan_fft_forward(n).process(&mut buf);
        let band = |f0: f64, f1: f64| -> f64 {
            let q = |f: f64| (f * n as f64 / FS as f64) as usize;
            buf[q(f0)..q(f1)].iter().map(|v| v.norm_sqr() as f64).sum()
        };
        10.0 * (band(1000.0, 4000.0) / band(50.0, 200.0)).log10()
    }

    /// loudest moment within 0.1 s after the onset, compared with the loudest moment overall
    fn suddenness(x: &[f32]) -> f64 {
        let first = onset(x);
        let peak = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        (x[first..first + FS as usize / 10].iter().fold(0.0f32, |a, v| a.max(v.abs())) / peak) as f64
    }

    #[test]
    fn near_thunder_is_sudden_and_far_thunder_is_a_rumble() {
        let mut cache = RefCache::default();
        let mut planner = FftPlanner::new();
        let p = |d: f32| SoundParams { distance_km: d, ..SoundParams::default() };
        let near = render(&p(0.3), &mut cache, &mut planner);
        let far = render(&p(8.0), &mut cache, &mut planner);
        let (bn, bf) = (brightness(&near.samples), brightness(&far.samples));
        assert!(bn > bf + 20.0, "high/low frequencies near {bn:.1} dB, far {bf:.1} dB");
        let (sn, sf) = (suddenness(&near.samples), suddenness(&far.samples));
        assert!(sn > 0.5 && sn > 4.0 * sf, "suddenness near {sn}, far {sf}");
        assert!(far.level_db.unwrap() < near.level_db.unwrap() - 20.0);
        eprintln!("near: {:.0} ms, far: {:.0} ms; high/low {bn:.1} / {bf:.1} dB; suddenness {sn:.2} / {sf:.2}", near.millis, far.millis);
    }

    #[test]
    fn whistler_follows_the_eckersley_law() {
        let p = SoundParams { example: SoundExample::Whistler, sferics: false, echoes: false, ..SoundParams::default() };
        let (x, _, _) = whistler(&p, &mut FftPlanner::new());
        // band-pass around f by a short FFT window search: the time of max energy near f
        for f in [1000.0, 4000.0] {
            let win = 1024;
            let k = (f * win as f64 / FS as f64).round() as usize;
            let fft = FftPlanner::<f32>::new().plan_fft_forward(win);
            let mut best = (0.0, 0.0f32);
            let mut s = 0;
            while s + win < x.len() {
                let mut b: Vec<C32> = x[s..s + win].iter().enumerate().map(|(i, v)| C32::new(v * (PI * i as f64 / win as f64).sin().powi(2) as f32, 0.0)).collect();
                fft.process(&mut b);
                let e = b[k].norm_sqr();
                if e > best.1 {
                    best = ((s + win / 2) as f64 / FS as f64, e);
                }
                s += 256;
            }
            let expect = T_FLASH + 60.0 / f.sqrt();
            assert!((best.0 - expect).abs() < 0.05, "f = {f}: {} s, expected {expect}", best.0);
        }
    }
}

#[cfg(test)]
mod debug {
    use super::*;

    /// waveform above the spectrogram, as a PNG, and the sound as a WAV file
    pub fn dump(r: &Rendered, stem: &str) {
        let (w, sh) = (r.spectrogram.size[0], r.spectrogram.size[1]);
        let wh = 160;
        let h = wh + sh;
        let mut px = vec![0u8; w * h * 4];
        for c in 0..w {
            let e = r.envelope[c * r.envelope.len() / w];
            let y0 = ((1.0 - e[1]) * 0.5 * wh as f32) as usize;
            let y1 = ((1.0 - e[0]) * 0.5 * wh as f32) as usize;
            for y in y0.min(wh - 1)..=y1.min(wh - 1) {
                px[(y * w + c) * 4..(y * w + c) * 4 + 4].copy_from_slice(&[200, 220, 255, 255]);
            }
            for y in 0..sh {
                let p = r.spectrogram.pixels[y * w + c];
                px[((y + wh) * w + c) * 4..((y + wh) * w + c) * 4 + 4].copy_from_slice(&[p.r(), p.g(), p.b(), 255]);
            }
        }
        let f = std::fs::File::create(format!("{stem}.png")).unwrap();
        let mut enc = png::Encoder::new(std::io::BufWriter::new(f), w as u32, h as u32);
        enc.set_color(png::ColorType::Rgba);
        enc.write_header().unwrap().write_image_data(&px).unwrap();
        let mut wav = Vec::new();
        let n = r.samples.len() as u32;
        wav.extend(b"RIFF");
        wav.extend((36 + 2 * n).to_le_bytes());
        wav.extend(b"WAVEfmt ");
        wav.extend(16u32.to_le_bytes());
        wav.extend(1u16.to_le_bytes());
        wav.extend(1u16.to_le_bytes());
        wav.extend(FS.to_le_bytes());
        wav.extend((2 * FS).to_le_bytes());
        wav.extend(2u16.to_le_bytes());
        wav.extend(16u16.to_le_bytes());
        wav.extend(b"data");
        wav.extend((2 * n).to_le_bytes());
        for v in r.samples.iter() {
            wav.extend(((v.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
        }
        std::fs::write(format!("{stem}.wav"), wav).unwrap();
    }

    #[test]
    #[ignore]
    fn dump_examples() {
        let dir = std::env::var("DUMP_DIR").unwrap_or_else(|_| ".".into());
        let mut cache = RefCache::default();
        let mut planner = FftPlanner::new();
        for (name, p) in [
            ("thunder_0.3km", SoundParams { distance_km: 0.3, ..SoundParams::default() }),
            ("thunder_1km", SoundParams { distance_km: 1.0, ..SoundParams::default() }),
            ("thunder_3km", SoundParams { distance_km: 3.0, ..SoundParams::default() }),
            ("thunder_10km", SoundParams { distance_km: 10.0, ..SoundParams::default() }),
            ("thunder_3km_disp", SoundParams { distance_km: 3.0, dispersion: true, ..SoundParams::default() }),
            ("thunder_3km_point_disp", SoundParams { distance_km: 3.0, dispersion: true, channel: false, ..SoundParams::default() }),
            ("whistler", SoundParams { example: SoundExample::Whistler, ..SoundParams::default() }),
            ("whistler_same", SoundParams { example: SoundExample::Whistler, same_hemisphere: true, nose_khz: 6.0, ..SoundParams::default() }),
        ] {
            let r = render(&p, &mut cache, &mut planner);
            eprintln!("{name}: {:.1} s from {:.1} s, {:.0} ms, level {:?}", r.duration, r.t_start, r.millis, r.level_db);
            dump(&r, &format!("{dir}/{name}"));
        }
    }
}
