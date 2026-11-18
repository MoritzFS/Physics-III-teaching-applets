//! PS09 38 · The particle in the box, seen through 2D cross-sections.
//!
//! One scene type serves all presets (eigenstates, sloshing, degenerate
//! pair, Gaussian packet at rest, packet with momentum); the variant sets the
//! defaults and the notes. Structure as in the 1D reference scene:
//! parameters → `restart()` → `step()` (exact phases e^(−iEt)) →
//! `central_ui()` (view bar, readouts, map + marginals, thumbnails + ladder,
//! time series and repeat-N results) → `side_ui()` → `notes()`.

use std::f64::consts::PI;

use qb_core::egui::{self, Align2, Color32, Rect, RichText, Sense, Shape, Stroke, StrokeKind, TextureHandle, TextureOptions, pos2, vec2};
use qb_core::num::ShotHistogram;
use qb_core::widgets::{
    Domain, FlashKind, Heatmap2D, HeatmapMode, HistScale, HistogramPlot, MeasureFlash, PanZoom, PlotFrame, PlotTransform, TimeSeries,
    TimeSeriesPlot, TimeWindow, field_image, fmt_num, label_box, plot_dashed, plot_line, readout_row, repeat_button, segmented,
};
use qb_core::{C64, Ctx, Look, QbRng, Scene, c64, wall_time};

use crate::ladder::{GroupedLadder, LadderLevel, LadderState};
use crate::modes::{
    A, Basis, EPS, Geometry, M, Moments2, Plane, Sampler2D, THUMBS, active, apply_window, degenerate_3d, e3_eps, gaussian_2d, group, idx,
    ij, level_probabilities, marginals, normalize, plane_degeneracy, project_level, zero,
};
use crate::view::{Display, arrow, colour_bar, fill_under, marginal_v, re_image, surface, vortices, zero_contour};

/// Which preset the scene was built for (defaults, notes, time scale).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// single eigenstates, the ladder and degeneracies (parts a–c)
    Eigen,
    /// (1,1) + (2,1): density sloshing along x
    Slosh,
    /// (1,2) + i(2,1): degenerate pair with a circulating current
    Degenerate,
    /// localised Gaussian at rest: spreading, interference, revival
    Packet,
    /// Gaussian with momentum: bouncing, spreading, revival
    Moving,
}

/// One eigenstate in the composed superposition: c ∝ amp · e^(iπ·phase).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Member {
    i: usize,
    j: usize,
    amp: f64,
    /// in units of π
    phase: f64,
}

impl Member {
    fn new(i: usize, j: usize, amp: f64, phase: f64) -> Self {
        Member { i, j, amp, phase }
    }
}

/// Where the state at restart comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    Members,
    Packet,
}

/// Gaussian packet parameters (fractions of the box edges for the centre).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Packet {
    /// centre as a fraction of the edges
    fu: f64,
    fv: f64,
    sigma: f64,
    /// |p| (ħ/a) and its direction (degrees from the x axis)
    p: f64,
    angle: f64,
}

/// "Repeat N" positions (drawn as dots building up |ψ|²).
struct RepeatPos {
    pts: Vec<[f64; 2]>,
    /// wall_time when they were taken (build-up animation)
    at: f64,
    mean: [f64; 2],
}

/// "Repeat N" energies.
struct RepeatE {
    hist: ShotHistogram,
    pred: Vec<f64>,
}

/// The tracer markers carried by the probability flow v = j/|ψ|².
struct Tracer {
    p: [f64; 2],
    trail: Vec<[f64; 2]>,
}

const TRACERS: usize = 110;
/// at most this many occupied modes for flow markers (cost per marker ∝ modes)
const FLOW_MAX_MODES: usize = 64;

pub struct BoxScene {
    variant: Variant,
    // parameters
    geom: Geometry,
    source: Source,
    members: Vec<Member>,
    /// thumbnail click adds/removes (true) or shows that state alone (false)
    click_adds: bool,
    packet: Packet,
    /// detector resolution w (σ of the window)
    width: f64,
    repeat_n: usize,
    display: Display,
    /// brightness ∝ |ψ| instead of |ψ|² (phase map)
    bright_amp: bool,
    /// brightness relative to the current maximum (else the fixed scale `top`)
    bright_auto: bool,
    surface: bool,
    azimuth: f32,
    show_nodes: bool,
    show_current: bool,
    show_flow: bool,

    // state
    basis: Basis,
    /// coefficients at time t0 (restart or last collapse)
    c0: Vec<C64>,
    t0: f64,
    t: f64,
    /// c(t) and the time it belongs to
    ct: Vec<C64>,
    ct_time: f64,
    active: Vec<usize>,
    moments: Moments2,
    field: Vec<C64>,
    field_time: f64,
    pu: Vec<f64>,
    pv: Vec<f64>,
    /// |ψ|² of full brightness, tops of the marginals, largest |j| (only grow)
    top: f64,
    top_pu: f64,
    top_pv: f64,
    j_top: f64,
    /// fraction of the state captured by the M × M modes at the last (re)start / collapse
    captured: f64,
    /// description of the state if it does not come from the settings (after a collapse)
    collapsed: Option<String>,
    /// energy range of the ladder when it shows a spectrum (many levels)
    spectrum: Option<[f64; 2]>,

    // time series
    sx: TimeSeries,
    sy: TimeSeries,
    sa: TimeSeries,
    sample_dt: f64,
    events: Vec<(f64, String)>,

    // measurement
    flash: MeasureFlash,
    spot: Option<([f64; 2], f64)>,
    rep_pos: Option<RepeatPos>,
    rep_e: Option<RepeatE>,

    // drawing
    map: Heatmap2D,
    thumbs: Vec<TextureHandle>,
    thumbs_key: Option<(Plane, u64)>,
    tracers: Vec<Tracer>,
    tracer_rng: QbRng,
}

impl BoxScene {
    pub fn new(variant: Variant) -> Self {
        let geom = Geometry::default();
        let (members, source) = match variant {
            Variant::Eigen => (vec![Member::new(2, 3, 1.0, 0.0)], Source::Members),
            Variant::Slosh => (vec![Member::new(1, 1, 1.0, 0.0), Member::new(2, 1, 1.0, 0.0)], Source::Members),
            Variant::Degenerate => (vec![Member::new(1, 2, 1.0, 0.0), Member::new(2, 1, 1.0, 0.5)], Source::Members),
            Variant::Packet | Variant::Moving => (vec![Member::new(1, 1, 1.0, 0.0)], Source::Packet),
        };
        let packet = match variant {
            Variant::Moving => Packet {
                fu: 0.3,
                fv: 0.35,
                sigma: 0.07,
                p: 58.0,
                angle: 31.0,
            },
            _ => Packet {
                fu: 0.3,
                fv: 0.4,
                sigma: 0.05,
                p: 0.0,
                angle: 0.0,
            },
        };
        let packety = matches!(variant, Variant::Packet | Variant::Moving);
        let mut s = BoxScene {
            variant,
            geom,
            source,
            members,
            click_adds: variant != Variant::Eigen,
            packet,
            width: 0.05,
            repeat_n: 1000,
            display: Display::Phase,
            bright_amp: packety,
            bright_auto: packety,
            surface: false,
            azimuth: -0.6,
            show_nodes: true,
            show_current: matches!(variant, Variant::Slosh | Variant::Degenerate),
            show_flow: variant == Variant::Degenerate,
            basis: Basis::new(geom),
            c0: vec![zero(); M * M],
            t0: 0.0,
            t: 0.0,
            ct: vec![],
            ct_time: f64::NAN,
            active: vec![],
            moments: Moments2::default(),
            field: vec![],
            field_time: f64::NAN,
            pu: vec![],
            pv: vec![],
            top: 1.0,
            top_pu: 1.0,
            top_pv: 1.0,
            j_top: 1.0,
            captured: 1.0,
            collapsed: None,
            spectrum: None,
            sx: TimeSeries::new("⟨x⟩"),
            sy: TimeSeries::new("⟨y⟩"),
            sa: TimeSeries::new("|⟨ψ(t₀)|ψ(t)⟩|²"),
            sample_dt: 0.004,
            events: vec![],
            flash: MeasureFlash::new(),
            spot: None,
            rep_pos: None,
            rep_e: None,
            map: Heatmap2D::new(),
            thumbs: vec![],
            thumbs_key: None,
            tracers: vec![],
            tracer_rng: QbRng::seed(11),
        };
        s.restart();
        s
    }

    // ------------------------------------------------------------ state

    /// Back to t = 0 with the current settings.
    fn restart(&mut self) {
        self.c0 = match self.source {
            Source::Members => {
                if self.members.iter().all(|m| m.amp <= 0.0)
                    && let Some(m) = self.members.first_mut()
                {
                    m.amp = 1.0;
                }
                let mut c = vec![zero(); M * M];
                for m in &self.members {
                    c[idx(m.i, m.j)] += C64::from_polar(m.amp, PI * m.phase);
                }
                c
            }
            Source::Packet => {
                let p = self.packet;
                let (lx, ly) = (self.geom.lx(), self.geom.ly());
                let a = p.angle.to_radians();
                gaussian_2d(&self.basis, p.fu * lx, p.fv * ly, p.sigma, p.p * a.cos(), p.p * a.sin())
            }
        };
        self.captured = match self.source {
            Source::Members => 1.0,
            Source::Packet => self.c0.iter().map(|z| z.norm_sqr()).sum(),
        };
        normalize(&mut self.c0);
        self.t = 0.0;
        self.t0 = 0.0;
        self.collapsed = None;
        self.events.clear();
        self.flash.clear();
        self.spot = None;
        self.rep_pos = None;
        self.rep_e = None;
        for s in [&mut self.sx, &mut self.sy, &mut self.sa] {
            s.clear();
        }
        self.sy.name = format!("⟨{}⟩", self.geom.v_axis());
        self.sample_dt = match self.variant {
            Variant::Eigen | Variant::Slosh | Variant::Degenerate => 0.004,
            Variant::Packet => 0.0016,
            Variant::Moving => 0.001,
        };
        self.after_new_state();
    }

    /// After c0 changed (restart or collapse): scales, ladder range, tracers.
    fn after_new_state(&mut self) {
        self.ct_time = f64::NAN;
        self.field_time = f64::NAN;
        self.ensure_field();
        // brightness and marginal scales: largest values over a few times for few-mode states
        let act = active(&self.c0);
        let probe = if act.len() <= 30 { self.probe_period(&act) } else { 0.0 };
        let mut top = 0.0f64;
        let mut tpu = 0.0f64;
        let mut tpv = 0.0f64;
        let mut jt = 0.0f64;
        let samples = if probe > 0.0 { 24 } else { 1 };
        let mut f = vec![];
        for k in 0..samples {
            let dt = probe * k as f64 / samples as f64;
            let c = self.basis.evolve(&self.c0, dt);
            self.basis.eval(&c, &mut f);
            top = top.max(f.iter().map(|z| z.norm_sqr()).fold(0.0, f64::max));
            let (pu, pv) = marginals(&f, self.basis.nx(), self.basis.ny(), self.basis.ax.dx, self.basis.ay.dx);
            tpu = tpu.max(pu.iter().cloned().fold(0.0, f64::max));
            tpv = tpv.max(pv.iter().cloned().fold(0.0, f64::max));
            jt = jt.max(self.max_current(&c));
        }
        self.top = 1.05 * top.max(1e-9);
        self.top_pu = 1.15 * tpu.max(1e-9);
        self.top_pv = 1.15 * tpv.max(1e-9);
        self.j_top = jt.max(1e-9);
        // ladder: a spectrum if the state reaches beyond the thumbnails
        let probs = level_probabilities(&self.geom, &self.c0);
        let detail_hi = self.detail_range()[1];
        let few = probs.len() <= 25 && probs.iter().all(|l| l.0 <= detail_hi);
        self.spectrum = if few { None } else { Some(quantile_range(&probs, 4e-3)) };
        self.seed_tracers();
        self.record();
    }

    /// A time over which the |ψ|² of a few-mode state goes through its pattern.
    fn probe_period(&self, act: &[usize]) -> f64 {
        let mut es: Vec<f64> = act.iter().map(|&k| self.basis.e_eps[k]).collect();
        es.sort_by(f64::total_cmp);
        es.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        let gap = es.windows(2).map(|w| w[1] - w[0]).fold(f64::INFINITY, f64::min);
        if gap.is_finite() && gap > 0.0 {
            2.0 * PI / (EPS * gap)
        } else {
            0.0
        }
    }

    /// Largest |j| on the coarse arrow grid.
    fn max_current(&self, c: &[C64]) -> f64 {
        let act = active(c);
        let (lx, ly) = (self.geom.lx(), self.geom.ly());
        let mut m = 0.0f64;
        for (u, v) in arrow_grid(lx, ly) {
            let (ju, jv, _) = self.basis.current_at(c, &act, u, v);
            m = m.max((ju * ju + jv * jv).sqrt());
        }
        m
    }

    /// Detail range of the ladder (E/ε): the thumbnail states with margin.
    fn detail_range(&self) -> [f64; 2] {
        let g = &self.geom;
        let lo = g.e_eps(1, 1);
        let mut hi = g.e_eps(THUMBS, THUMBS);
        if self.source == Source::Members {
            for m in &self.members {
                hi = hi.max(g.e_eps(m.i, m.j));
            }
        }
        let span = (hi - lo).max(1.0);
        [(lo - 0.04 * span).max(0.0), hi + 0.025 * span]
    }

    /// c(t), moments and the active list for the current time.
    fn ensure_state(&mut self) {
        if self.ct_time == self.t && !self.ct.is_empty() {
            return;
        }
        self.ct = self.basis.evolve(&self.c0, self.t - self.t0);
        self.ct_time = self.t;
        self.active = active(&self.ct);
        self.moments = self.basis.moments(&self.ct);
    }

    /// ψ on the grid and the marginals for the current time.
    fn ensure_field(&mut self) {
        self.ensure_state();
        if self.field_time == self.t && !self.field.is_empty() {
            return;
        }
        self.basis.eval(&self.ct, &mut self.field);
        self.field_time = self.t;
        let b = &self.basis;
        let (pu, pv) = marginals(&self.field, b.nx(), b.ny(), b.ax.dx, b.ay.dx);
        self.pu = pu;
        self.pv = pv;
        let peak = self.field.iter().map(|z| z.norm_sqr()).fold(0.0, f64::max);
        if peak > self.top * 1.02 {
            self.top = 1.1 * peak;
        }
        let m = self.pu.iter().cloned().fold(0.0, f64::max);
        if m > self.top_pu {
            self.top_pu = 1.15 * m;
        }
        let m = self.pv.iter().cloned().fold(0.0, f64::max);
        if m > self.top_pv {
            self.top_pv = 1.15 * m;
        }
    }

    /// Autocorrelation |⟨ψ(t₀)|ψ(t)⟩|² (exact, from the coefficients).
    fn autocorrelation(&self, t: f64) -> f64 {
        let dt = t - self.t0;
        let mut s = zero();
        for (z, &e) in self.c0.iter().zip(&self.basis.e_eps) {
            let p = z.norm_sqr();
            if p > 0.0 {
                s += C64::from_polar(p, -EPS * e * dt);
            }
        }
        s.norm_sqr()
    }

    fn record(&mut self) {
        self.ensure_state();
        let m = self.moments;
        self.sx.push(self.t, m.x);
        self.sy.push(self.t, m.y);
        self.sa.push(self.t, self.autocorrelation(self.t));
    }

    /// Records the series at sample times between the last point and t (exact moments).
    fn record_until(&mut self, t: f64) {
        let last = self.sx.last().map_or(t, |p| p[0]);
        let n = ((t - last) / self.sample_dt).floor() as usize;
        let n = n.min(3000);
        for k in 1..=n {
            let tk = last + (t - last) * k as f64 / n as f64;
            let c = self.basis.evolve(&self.c0, tk - self.t0);
            let m = self.basis.moments(&c);
            self.sx.push(tk, m.x);
            self.sy.push(tk, m.y);
            self.sa.push(tk, self.autocorrelation(tk));
        }
    }

    /// Sets the clock (jump buttons); the series are filled in exactly.
    fn jump(&mut self, t: f64) {
        let t = t.max(self.t0);
        if t > self.t {
            self.record_until(t);
        }
        self.t = t;
        self.ensure_field();
        self.record();
        self.seed_tracers();
    }

    fn set_geometry(&mut self, g: Geometry) {
        if g == self.geom {
            return;
        }
        // keep composed states inside the expansion
        self.geom = g;
        self.basis = Basis::new(g);
        self.restart();
    }

    fn select_single(&mut self, i: usize, j: usize) {
        self.source = Source::Members;
        self.members = vec![Member::new(i, j, 1.0, 0.0)];
        self.restart();
    }

    fn toggle_member(&mut self, i: usize, j: usize) {
        if self.source != Source::Members {
            self.source = Source::Members;
            self.members.clear();
        }
        if let Some(k) = self.members.iter().position(|m| m.i == i && m.j == j) {
            if self.members.len() > 1 {
                self.members.remove(k);
            }
        } else {
            self.members.push(Member::new(i, j, 1.0, 0.0));
        }
        self.restart();
    }

    /// The single occupied state, if the state is one eigenstate.
    fn single(&self) -> Option<(usize, usize)> {
        if self.active.len() == 1 { Some(ij(self.active[0])) } else { None }
    }

    // ------------------------------------------------------------ measurement

    fn measure_position(&mut self, ctx: &mut Ctx) {
        self.ensure_field();
        let b = &self.basis;
        let Some(s) = Sampler2D::new(&self.field, b.nx(), b.ax.dx, b.ay.dx) else {
            return;
        };
        let r0 = s.sample(ctx.rng);
        let w = self.width;
        let mut f = self.field.clone();
        apply_window(&mut f, &self.basis, r0, w);
        let n_grid: f64 = f.iter().map(|z| z.norm_sqr()).sum::<f64>() * b.ax.dx * b.ay.dx;
        let mut c = self.basis.project(&f);
        let n_modes: f64 = c.iter().map(|z| z.norm_sqr()).sum();
        normalize(&mut c);
        self.captured = n_modes / n_grid.max(1e-300);
        self.c0 = c;
        self.t0 = self.t;
        let v = self.geom.v_axis();
        let label = format!("x = {:.2}, {v} = {:.2}", r0[0], r0[1]);
        self.flash.trigger(FlashKind::Position { x: r0[0], width: w }, label.clone());
        self.spot = Some((r0, w));
        self.collapsed = Some(format!(
            "localised by a position measurement at t = {:.3} ({label}, detector w = {w:.2})",
            self.t
        ));
        self.events.push((self.t, "x".into()));
        self.after_new_state();
        ctx.set_status(format!(
            "measured position {label} at t = {:.3}; ψ collapsed onto a spot of width w = {w:.2} (the {M} × {M} modes keep {} of it)",
            self.t,
            fmt_captured(self.captured)
        ));
    }

    fn measure_energy(&mut self, ctx: &mut Ctx) {
        self.ensure_state();
        let probs = level_probabilities(&self.geom, &self.ct);
        let weights: Vec<f64> = probs.iter().map(|l| l.1).collect();
        let Some(k) = ctx.rng.weighted(&weights) else { return };
        let (e, p, ref states) = probs[k];
        let mut c = self.ct.clone();
        project_level(&self.geom, &mut c, e);
        self.c0 = c;
        self.t0 = self.t;
        self.captured = 1.0;
        let names = states
            .iter()
            .map(|&q| {
                let (i, j) = ij(q);
                format!("({i},{j})")
            })
            .collect::<Vec<_>>()
            .join(" ");
        let g = plane_degeneracy(&self.geom, e).len();
        let label = format!("E = {} ε", fmt_e(e));
        self.flash.trigger(FlashKind::Level { index: k, energy: e }, label.clone());
        self.spot = None;
        let what = if states.len() > 1 {
            format!(
                "projected onto the {}-dimensional eigenspace {names} (amplitudes and phases within it kept)",
                states.len()
            )
        } else if g > 1 {
            format!("only {names} of this {g}-fold level was occupied, so ψ is that state")
        } else {
            format!("ψ is now the eigenstate {names}")
        };
        self.collapsed = Some(format!("energy measurement at t = {:.3}: {label}, {what}", self.t));
        self.events.push((self.t, "E".into()));
        self.after_new_state();
        ctx.set_status(format!("measured {label} (probability was {p:.3}) at t = {:.3}: {what}", self.t));
    }

    fn repeat_position(&mut self, n: usize, ctx: &mut Ctx) {
        self.ensure_field();
        let b = &self.basis;
        let Some(s) = Sampler2D::new(&self.field, b.nx(), b.ax.dx, b.ay.dx) else {
            return;
        };
        let pts: Vec<[f64; 2]> = (0..n).map(|_| s.sample(ctx.rng)).collect();
        let mean = [
            pts.iter().map(|p| p[0]).sum::<f64>() / n.max(1) as f64,
            pts.iter().map(|p| p[1]).sum::<f64>() / n.max(1) as f64,
        ];
        let m = self.moments;
        let v = self.geom.v_axis();
        ctx.set_status(format!(
            "{n} fresh copies measured in position: mean (x, {v}) = ({:.3}, {:.3}), Born rule ⟨x⟩, ⟨{v}⟩ = ({:.3}, {:.3})",
            mean[0], mean[1], m.x, m.y
        ));
        self.rep_pos = Some(RepeatPos {
            pts,
            at: wall_time(),
            mean,
        });
    }

    fn repeat_energy(&mut self, n: usize, ctx: &mut Ctx) {
        self.ensure_state();
        let probs = level_probabilities(&self.geom, &self.ct);
        let weights: Vec<f64> = probs.iter().map(|l| l.1).collect();
        let outcomes: Vec<usize> = (0..n).filter_map(|_| ctx.rng.weighted(&weights)).collect();
        let mean: f64 = outcomes.iter().map(|&k| probs[k].0).sum::<f64>() / n.max(1) as f64;
        // bins on the energy axis (E/ε), so that the histogram lines up with the ladder
        let [lo, hi] = match self.spectrum {
            Some(_) => quantile_range(&probs, 1e-4),
            None => {
                let (a, b) = (probs[0].0, probs[probs.len() - 1].0);
                let pad = (0.08 * (b - a)).max(1.0);
                [a - pad, b + pad]
            }
        };
        // few levels: bins of 0.25 ε (near-degenerate levels closer than that share a bin)
        let bins = if self.spectrum.is_some() {
            40
        } else {
            ((hi - lo) / 0.25).round().clamp(8.0, 80.0) as usize
        };
        let mut hist = ShotHistogram::binned(lo, hi, bins);
        outcomes.iter().for_each(|&k| hist.add(probs[k].0));
        let edges = hist.edges();
        let mut pred = vec![0.0; hist.len()];
        for l in &probs {
            let b = edges.partition_point(|&e| e <= l.0);
            if b >= 1 && b <= pred.len() {
                pred[b - 1] += l.1;
            }
        }
        let rep = RepeatE { hist, pred };
        ctx.set_status(format!(
            "{n} fresh copies measured in energy: mean {:.3} ε, Born rule ⟨E⟩ = {:.3} ε (each outcome is a level; the state is not changed)",
            mean, self.moments.e
        ));
        self.rep_e = Some(rep);
    }

    // ------------------------------------------------------------ tracers

    fn flow_possible(&self) -> bool {
        active(&self.c0).len() <= FLOW_MAX_MODES
    }

    fn seed_tracers(&mut self) {
        self.tracers.clear();
        if !self.flow_possible() {
            return;
        }
        self.ensure_field();
        let b = &self.basis;
        if let Some(s) = Sampler2D::new(&self.field, b.nx(), b.ax.dx, b.ay.dx) {
            for _ in 0..TRACERS {
                let p = s.sample(&mut self.tracer_rng);
                self.tracers.push(Tracer { p, trail: vec![p] });
            }
        }
    }

    /// Velocity of the probability flow v = j/|ψ|² at p for coefficients c (capped).
    fn velocity(&self, c: &[C64], act: &[usize], p: [f64; 2]) -> [f64; 2] {
        let (ju, jv, rho) = self.basis.current_at(c, act, p[0], p[1]);
        let floor = 1e-4 * self.top;
        let rho = rho.max(floor);
        let (mut vu, mut vv) = (ju / rho, jv / rho);
        let s = (vu * vu + vv * vv).sqrt();
        let cap = 25.0;
        if s > cap {
            vu *= cap / s;
            vv *= cap / s;
        }
        [vu, vv]
    }

    fn advance_tracers(&mut self, t_start: f64, dt: f64) {
        if !self.show_flow || self.tracers.is_empty() {
            return;
        }
        let steps = (dt / 0.0015).ceil().max(1.0) as usize;
        let h = dt / steps as f64;
        let (lx, ly) = (self.geom.lx(), self.geom.ly());
        let eps = 1e-3;
        let mut tracers = std::mem::take(&mut self.tracers);
        for s in 0..steps {
            let ta = t_start + s as f64 * h;
            let ca = self.basis.evolve(&self.c0, ta - self.t0);
            let cm = self.basis.evolve(&self.c0, ta + 0.5 * h - self.t0);
            let act = active(&ca);
            for tr in tracers.iter_mut() {
                let v1 = self.velocity(&ca, &act, tr.p);
                let mid = [tr.p[0] + 0.5 * h * v1[0], tr.p[1] + 0.5 * h * v1[1]];
                let v2 = self.velocity(&cm, &act, mid);
                tr.p = [
                    (tr.p[0] + h * v2[0]).clamp(eps, lx - eps),
                    (tr.p[1] + h * v2[1]).clamp(eps, ly - eps),
                ];
                let last = *tr.trail.last().unwrap_or(&tr.p);
                if ((tr.p[0] - last[0]).powi(2) + (tr.p[1] - last[1]).powi(2)).sqrt() > 0.006 {
                    tr.trail.push(tr.p);
                    if tr.trail.len() > 10 {
                        tr.trail.remove(0);
                    }
                }
            }
        }
        self.tracers = tracers;
    }

    // ------------------------------------------------------------ drawing

    fn ensure_thumbs(&mut self, ctx: &egui::Context) {
        let key = (self.geom.plane, self.geom.ly().to_bits());
        if self.thumbs_key == Some(key) && self.thumbs.len() == THUMBS * THUMBS {
            return;
        }
        let (lx, ly) = (self.geom.lx(), self.geom.ly());
        let nu = 44usize;
        let nv = ((44.0 * ly / lx).round() as usize).clamp(16, 120);
        self.thumbs.clear();
        for i in 1..=THUMBS {
            for j in 1..=THUMBS {
                let mut f = vec![zero(); nu * nv];
                for iv in 0..nv {
                    let v = (iv as f64 + 0.5) / nv as f64 * ly;
                    for iu in 0..nu {
                        let u = (iu as f64 + 0.5) / nu as f64 * lx;
                        f[iv * nu + iu] = c64((i as f64 * PI * u / lx).sin() * (j as f64 * PI * v / ly).sin(), 0.0);
                    }
                }
                let img = field_image(nu, nv, &f, HeatmapMode::PhaseBrightness, Some(1.0));
                self.thumbs
                    .push(ctx.load_texture(format!("box_thumb_{i}_{j}"), img, TextureOptions::LINEAR));
            }
        }
        self.thumbs_key = Some(key);
    }

    fn thumbs_ui(&mut self, ui: &mut egui::Ui, width: f32, height: f32) {
        let look = Look::of(ui);
        self.ensure_thumbs(ui.ctx());
        let (outer, _) = ui.allocate_exact_size(vec2(width, height), Sense::hover());
        let painter = ui.painter_at(outer);
        let lab = look.px(18.0);
        let cell = ((width - lab - look.px(4.0)) / THUMBS as f32).min((height - lab - look.px(4.0)) / THUMBS as f32);
        let x0 = outer.left() + lab;
        let y_bottom = outer.top() + THUMBS as f32 * cell;
        let (lx, ly) = (self.geom.lx() as f32, self.geom.ly() as f32);
        let pad = look.px(3.0);
        let inner = cell - 2.0 * pad;
        let (iw, ih) = if ly > lx {
            (inner * lx / ly, inner)
        } else {
            (inner, inner * ly / lx)
        };
        // few states: highlight them; many (a packet): a bar of |c|² under each thumbnail
        let few = self.active.len() <= 25;
        let occupied: Vec<(usize, usize)> = if few {
            self.active.iter().map(|&k| ij(k)).collect()
        } else {
            vec![]
        };
        let cmax = self.ct.iter().map(|z| z.norm_sqr()).fold(0.0, f64::max).max(1e-300);
        let mut clicked = None;
        for i in 1..=THUMBS {
            for j in 1..=THUMBS {
                let c = pos2(x0 + (i as f32 - 0.5) * cell, y_bottom - (j as f32 - 0.5) * cell);
                let r = Rect::from_center_size(c, vec2(iw, ih));
                let id = ui.id().with(("box_thumb", i, j));
                let resp = ui.interact(r.expand(pad), id, Sense::click());
                painter.image(
                    self.thumbs[(i - 1) * THUMBS + (j - 1)].id(),
                    r,
                    Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                    Color32::WHITE,
                );
                let occ = occupied.contains(&(i, j));
                let (w, col) = if occ {
                    (2.5, look.accent())
                } else if resp.hovered() {
                    (1.5, look.text())
                } else {
                    (1.0, look.plot_frame())
                };
                painter.rect_stroke(r.expand(1.0), 2.0, look.stroke(w, col), StrokeKind::Outside);
                if !few {
                    let p = (self.ct[idx(i, j)].norm_sqr() / cmax) as f32;
                    let bar = Rect::from_min_size(pos2(r.left(), r.bottom() + 1.5), vec2(p * r.width(), look.px(3.0)));
                    painter.rect_filled(bar, 0.0, look.counts());
                }
                if resp.clicked() {
                    clicked = Some((i, j, ui.input(|inp| inp.modifiers.shift)));
                }
                let e = self.geom.e_eps(i, j);
                let g = plane_degeneracy(&self.geom, e).len();
                resp.on_hover_text(format!(
                    "({i},{j}): E = {} ε{}\nclick: {}  (shift-click: {})",
                    fmt_e(e),
                    if g > 1 { format!(", degenerate g = {g}") } else { String::new() },
                    if self.click_adds { "add / remove" } else { "show alone" },
                    if self.click_adds { "show alone" } else { "add / remove" },
                ));
            }
            painter.text(
                pos2(x0 + (i as f32 - 0.5) * cell, y_bottom + 1.0),
                Align2::CENTER_TOP,
                i.to_string(),
                look.font(10.5),
                look.weak_text(),
            );
        }
        for j in 1..=THUMBS {
            painter.text(
                pos2(x0 - 3.0, y_bottom - (j as f32 - 0.5) * cell),
                Align2::RIGHT_CENTER,
                j.to_string(),
                look.font(10.5),
                look.weak_text(),
            );
        }
        painter.text(
            pos2(x0 + THUMBS as f32 * cell, y_bottom + 1.0),
            Align2::LEFT_TOP,
            " n₁",
            look.font(11.5),
            look.text(),
        );
        painter.text(
            pos2(outer.left(), outer.top()),
            Align2::LEFT_TOP,
            self.geom.v_index(),
            look.font(11.5),
            look.text(),
        );
        if let Some((i, j, shift)) = clicked {
            if self.click_adds != shift {
                self.toggle_member(i, j);
            } else {
                self.select_single(i, j);
            }
        }
    }

    /// Ladder levels for the detail view (all modes up to the range) or the spectrum.
    fn ladder_levels(&self) -> (Vec<LadderLevel>, [f64; 2], bool) {
        let occupied = |k: usize| self.ct.get(k).copied().unwrap_or(zero());
        match self.spectrum {
            None => {
                let range = self.detail_range();
                let mut all = vec![];
                for i in 1..=M {
                    for j in 1..=M {
                        let e = self.geom.e_eps(i, j);
                        if e <= range[1] {
                            all.push((e, (i, j)));
                        }
                    }
                }
                let levels = group(all)
                    .into_iter()
                    .map(|l| {
                        let states: Vec<LadderState> = l
                            .states
                            .iter()
                            .map(|&(i, j)| LadderState {
                                i,
                                j,
                                c: occupied(idx(i, j)),
                                thumb: i <= THUMBS && j <= THUMBS,
                            })
                            .collect();
                        let prob = states.iter().map(|s| s.c.norm_sqr()).sum();
                        LadderLevel {
                            g: plane_degeneracy(&self.geom, l.e).len().max(states.len()),
                            e: l.e,
                            states,
                            prob,
                        }
                    })
                    .collect();
                (levels, range, false)
            }
            Some(range) => {
                let levels = level_probabilities(&self.geom, &self.ct)
                    .into_iter()
                    .map(|(e, p, ks)| LadderLevel {
                        e,
                        g: ks.len(),
                        states: ks
                            .iter()
                            .map(|&k| {
                                let (i, j) = ij(k);
                                LadderState {
                                    i,
                                    j,
                                    c: occupied(k),
                                    thumb: i <= THUMBS && j <= THUMBS,
                                }
                            })
                            .collect(),
                        prob: p,
                    })
                    .collect();
                (levels, range, true)
            }
        }
    }

    /// |ψ| of full brightness: the fixed scale or the current maximum.
    fn scale_abs(&self) -> f64 {
        if self.bright_auto {
            self.field.iter().map(|z| z.norm_sqr()).fold(0.0, f64::max).sqrt().max(1e-12)
        } else {
            self.top.sqrt()
        }
    }

    /// The map with its overlays, P(v) on the right and P(u) below.
    fn map_ui(&mut self, ui: &mut egui::Ui, width: f32, height: f32) {
        let look = Look::of(ui);
        self.ensure_field();
        // current arrows (the scale only grows, like the brightness)
        let arrows: Vec<(f64, f64, f64, f64)> = if self.show_current {
            arrow_grid(self.geom.lx(), self.geom.ly())
                .into_iter()
                .map(|(u, v)| {
                    let (ju, jv, _) = self.basis.current_at(&self.ct, &self.active, u, v);
                    (u, v, ju, jv)
                })
                .collect()
        } else {
            vec![]
        };
        let jm = arrows.iter().map(|a| a.2.hypot(a.3)).fold(0.0, f64::max);
        if jm > self.j_top {
            self.j_top = 1.1 * jm;
        }
        let b = &self.basis;
        let (nx, ny) = (b.nx(), b.ny());
        let max_abs = self.scale_abs();
        let img = match self.display {
            Display::Phase => field_image(
                nx,
                ny,
                &self.field,
                if self.bright_amp {
                    HeatmapMode::PhaseBrightness
                } else {
                    HeatmapMode::PhaseDensity
                },
                Some(max_abs),
            ),
            Display::Real => re_image(nx, ny, &self.field, max_abs, &look),
            Display::Plain => field_image(nx, ny, &self.field, HeatmapMode::Density, Some(max_abs)),
        };
        self.map.legend = self.display == Display::Phase;
        self.map.set_image(ui.ctx(), img);

        let (lx, ly) = (self.geom.lx(), self.geom.ly());
        let v = self.geom.v_axis();
        let (ml, mr, mt, mb) = PlotFrame::margins(&look, false);
        let side_w = look.px(80.0);
        let pu_h = look.px(104.0).min(height * 0.24);
        let gap = look.px(6.0);
        let max_w = width - ml - mr - side_w - gap;
        let max_h = height - pu_h - mt - mb - gap;
        let plot_w = max_w.min(max_h * (lx / ly) as f32).max(60.0);
        let plot_h = plot_w * (ly / lx) as f32;
        let map_w = plot_w + ml + mr;

        let resp = ui
            .horizontal_top(|ui| {
                ui.spacing_mut().item_spacing.x = gap;
                let r = self.map.show(ui, "box_map", [0.0, lx, 0.0, ly], ("x", v), plot_h + mt + mb);
                let single = self.single();
                let analytic: Option<Vec<f64>> =
                    single.map(|(_, j)| b.ay.xs.iter().map(|&y| 2.0 / ly * (j as f64 * PI * y / ly).sin().powi(2)).collect());
                marginal_v(
                    ui,
                    side_w,
                    &r.transform,
                    &b.ay.xs,
                    &self.pv,
                    self.top_pv,
                    analytic.as_deref(),
                    &format!("P({v})"),
                );
                r
            })
            .inner;
        let tr = resp.transform;
        let painter = resp.painter(ui);
        self.overlays(ui, &painter, &tr, &arrows);
        if let Some((x, y)) = resp.hover {
            resp.response.clone().on_hover_text(format!("x = {x:.3}, {v} = {y:.3}"));
        }

        // P(u) below, sharing the x axis
        ui.add_space(gap);
        let single = self.single();
        let analytic: Option<Vec<f64>> =
            single.map(|(i, _)| b.ax.xs.iter().map(|&x| 2.0 / lx * (i as f64 * PI * x / lx).sin().powi(2)).collect());
        let (xs, pu, top) = (&b.ax.xs, &self.pu, self.top_pu);
        let pr = PlotFrame::new("box_pu")
            .x_range(tr.x[0], tr.x[1])
            .y_range(0.0, top)
            .labels("x", "P(x) = ∫|ψ|² d".to_string() + v)
            .width(map_w)
            .height(pu_h)
            .pan_zoom(PanZoom::None)
            .show(ui, |p, ptr| {
                fill_under(p, ptr, xs, pu, look.re().gamma_multiply(0.35));
                plot_line(p, ptr, xs, pu, look.stroke(1.6, look.re()));
                if let Some(a) = &analytic {
                    plot_dashed(p, ptr, xs, a, look.stroke(1.3, look.prediction()), 5.0);
                }
            });
        if matches!(self.flash.kind(), Some(FlashKind::Position { .. })) {
            self.flash.paint(ui, &pr.transform, Domain::Position);
        }
        if let Some((i, j)) = single {
            let p = ui.painter_at(pr.transform.rect);
            label_box(
                &p,
                pr.transform.rect.right_top() + vec2(-look.px(4.0), look.px(4.0)),
                Align2::RIGHT_TOP,
                &format!("dashed: 1D box states n = {i} (x), n = {j} ({v})"),
                look.prediction(),
            );
        }
    }

    fn overlays(&self, ui: &egui::Ui, painter: &egui::Painter, tr: &PlotTransform, arrows: &[(f64, f64, f64, f64)]) {
        let look = Look::of(ui);
        let b = &self.basis;
        let (lx, ly) = (self.geom.lx(), self.geom.ly());
        // walls
        let wall = Rect::from_two_pos(tr.to_screen(0.0, 0.0), tr.to_screen(lx, ly));
        painter.rect_stroke(wall, 0.0, look.stroke(1.5, look.potential()), StrokeKind::Outside);
        let line_col = match self.display {
            Display::Real => look.text().gamma_multiply(0.85),
            _ => Color32::from_white_alpha(210),
        };
        if self.show_nodes {
            // node lines when ψ is real up to a global phase, else the phase vortices
            let (kmax, zmax) = self
                .field
                .iter()
                .enumerate()
                .map(|(k, z)| (k, z.norm()))
                .fold((0, 0.0), |a, b| if b.1 > a.1 { b } else { a });
            let rot = if zmax > 0.0 {
                self.field[kmax].conj() / zmax
            } else {
                c64(1.0, 0.0)
            };
            let im_max = self.field.iter().map(|z| (z * rot).im.abs()).fold(0.0, f64::max);
            if im_max < 1e-7 * zmax {
                let re: Vec<f64> = self.field.iter().map(|z| (z * rot).re).collect();
                for seg in zero_contour(&re, &b.ax.xs, &b.ay.xs, 1e-3 * zmax) {
                    painter.line_segment(
                        [tr.to_screen(seg[0][0], seg[0][1]), tr.to_screen(seg[1][0], seg[1][1])],
                        look.stroke(1.4, line_col),
                    );
                }
            } else if self.active.len() <= 30 {
                let vs = vortices(&self.field, &b.ax.xs, &b.ay.xs, zmax);
                if vs.len() <= 40 {
                    for (u, v, w) in vs {
                        let c = tr.to_screen(u, v);
                        let r = look.px(6.0);
                        painter.circle_stroke(c, r, look.stroke(1.5, line_col));
                        // a small arrow head showing the sense of the phase winding (arg ψ increases)
                        let s = if w > 0 { 1.0 } else { -1.0 };
                        let tip = c + vec2(r, 0.0);
                        painter.add(Shape::convex_polygon(
                            vec![tip + vec2(-3.5, s * 1.0), tip + vec2(3.5, s * 1.0), tip + vec2(0.0, -s * 5.0)],
                            line_col,
                            Stroke::NONE,
                        ));
                    }
                }
            }
        }
        if !arrows.is_empty() {
            let cell = (lx.max(ly)) / ARROWS as f64;
            let jt = self.j_top.max(1e-12);
            for &(u, v, ju, jv) in arrows {
                let s = 0.85 * cell / jt;
                let a = tr.to_screen(u - 0.5 * s * ju, v - 0.5 * s * jv);
                let e = tr.to_screen(u + 0.5 * s * ju, v + 0.5 * s * jv);
                arrow(painter, a, e, look.stroke(1.3, line_col));
            }
        }
        if self.show_flow && !self.tracers.is_empty() {
            let dot = look.px(2.4);
            for t in &self.tracers {
                let n = t.trail.len();
                for k in 1..n {
                    let a = (k as f32 / n as f32) * 0.8;
                    painter.line_segment(
                        [
                            tr.to_screen(t.trail[k - 1][0], t.trail[k - 1][1]),
                            tr.to_screen(t.trail[k][0], t.trail[k][1]),
                        ],
                        Stroke::new(look.px(1.4), Color32::from_white_alpha((a * 200.0) as u8)),
                    );
                }
                let p = tr.to_screen(t.p[0], t.p[1]);
                painter.circle(p, dot, Color32::WHITE, Stroke::new(1.0, Color32::from_black_alpha(200)));
            }
        }
        // the last position measurement: detector spot
        if let Some((r0, w)) = self.spot {
            let c = tr.to_screen(r0[0], r0[1]);
            let rad = (w / tr.dx_per_point()).abs() as f32;
            let age = self.flash.age().unwrap_or(f32::MAX);
            let col = look.measure();
            if age < 0.7 {
                let f = 1.0 - age / 0.7;
                painter.circle_filled(c, rad * 3.0 + look.px(6.0), Color32::WHITE.gamma_multiply(0.35 * f));
            }
            if age < 4.0 {
                painter.circle_stroke(c, rad, look.stroke(1.8, col));
                painter.circle_stroke(c, 2.0 * rad, look.stroke(1.0, col.gamma_multiply(0.7)));
                if let Some(l) = self.flash.label() {
                    label_box(
                        painter,
                        c + vec2(2.0 * rad + look.px(6.0), -look.px(6.0)),
                        Align2::LEFT_BOTTOM,
                        l,
                        col,
                    );
                }
            }
            let s = look.px(5.0);
            painter.line_segment([c - vec2(s, 0.0), c + vec2(s, 0.0)], look.stroke(1.5, col));
            painter.line_segment([c - vec2(0.0, s), c + vec2(0.0, s)], look.stroke(1.5, col));
        }
        if matches!(self.flash.kind(), Some(FlashKind::Level { .. })) {
            self.flash.paint_banner(ui, tr.rect);
        }
        // legend of the brightness / colour scale
        if self.display != Display::Phase {
            colour_bar(painter, tr, self.display, self.bright_amp, &look);
        }
        let what = match self.display {
            Display::Phase if self.bright_amp => "brightness ∝ |ψ|",
            Display::Phase => "brightness ∝ |ψ|²",
            Display::Real => "Re ψ",
            Display::Plain => "|ψ|²",
        };
        let note = format!(
            "{what}, {}",
            if self.bright_auto {
                "scaled to the current maximum"
            } else {
                "fixed scale"
            }
        );
        let g = painter.layout_no_wrap(note, look.font(10.5), look.weak_text());
        let p = tr.rect.left_bottom() + vec2(look.px(5.0), -look.px(4.0) - g.size().y);
        painter.rect_filled(
            Rect::from_min_size(p - vec2(2.0, 1.0), g.size() + vec2(4.0, 2.0)),
            2.0,
            look.plot_bg().gamma_multiply(0.8),
        );
        painter.galley(p, g, look.weak_text());
    }

    fn surface_ui(&mut self, ui: &mut egui::Ui, width: f32, height: f32) {
        let look = Look::of(ui);
        self.ensure_field();
        let (rect, resp) = ui.allocate_exact_size(vec2(width, height), Sense::drag());
        if resp.dragged() && !look.shot {
            self.azimuth += resp.drag_delta().x * 0.01;
        }
        let b = &self.basis;
        let step = 3;
        let (nx, ny) = (b.nx() / step, b.ny() / step);
        let mut f = Vec::with_capacity(nx * ny);
        for iv in 0..ny {
            for iu in 0..nx {
                f.push(self.field[(iv * step + 1) * b.nx() + iu * step + 1]);
            }
        }
        let painter = ui.painter_at(rect);
        surface(
            &painter,
            rect,
            &f,
            nx,
            ny,
            self.geom.lx(),
            self.geom.ly(),
            self.scale_abs().powi(2),
            self.display,
            self.azimuth,
            ("x", self.geom.v_axis()),
            &look,
        );
        if matches!(self.flash.kind(), Some(FlashKind::Level { .. } | FlashKind::Position { .. })) {
            self.flash.paint_banner(ui, rect);
        }
    }

    fn scatter_ui(&self, ui: &mut egui::Ui, rep: &RepeatPos, height: f32) {
        let look = Look::of(ui);
        let (lx, ly) = (self.geom.lx(), self.geom.ly());
        let (ml, mr, mt, mb) = PlotFrame::margins(&look, false);
        let ph = height - mt - mb;
        let pw = ph * (lx / ly) as f32;
        let n_show = if look.shot {
            rep.pts.len()
        } else {
            let age = (wall_time() - rep.at) as f32;
            if age < 1.6 {
                ui.ctx().request_repaint();
            }
            ((age / 1.5).clamp(0.0, 1.0) * rep.pts.len() as f32) as usize
        };
        let pts = &rep.pts[..n_show.min(rep.pts.len())];
        let v = self.geom.v_axis();
        let r = PlotFrame::new("box_scatter")
            .x_range(0.0, lx)
            .y_range(0.0, ly)
            .labels("x", v)
            .width(pw + ml + mr)
            .height(height)
            .grid(false)
            .zero_line(false)
            .pan_zoom(PanZoom::None)
            .show(ui, |p, tr| {
                let rad = if pts.len() > 3000 { 1.0 } else { 1.6 } * look.font_scale();
                let col = look.counts();
                for q in pts {
                    p.circle_filled(tr.to_screen(q[0], q[1]), rad, col);
                }
                label_box(
                    p,
                    tr.rect.right_top() + vec2(-3.0, 3.0),
                    Align2::RIGHT_TOP,
                    &format!("{} detections", pts.len()),
                    look.text(),
                );
            });
        let mean = rep.mean;
        r.response.on_hover_text(format!(
            "{} positions on fresh copies of ψ: they build up |ψ|² (Born rule).\nmean (x, {v}) = ({:.3}, {:.3})",
            rep.pts.len(),
            mean[0],
            mean[1]
        ));
    }

    fn bottom_ui(&mut self, ui: &mut egui::Ui, height: f32) {
        let look = Look::of(ui);
        let w = ui.available_width();
        let (lx, ly) = (self.geom.lx(), self.geom.ly());
        let scatter_w = if self.rep_pos.is_some() {
            let (ml, mr, mt, mb) = PlotFrame::margins(&look, false);
            (height - mt - mb) * (lx / ly) as f32 + ml + mr + look.px(8.0)
        } else {
            0.0
        };
        let hist_w = if self.rep_e.is_some() {
            (w * 0.34).max(look.px(240.0))
        } else {
            0.0
        };
        let ts_w = w - scatter_w - hist_w - if hist_w > 0.0 { look.px(8.0) } else { 0.0 };
        let show_ts = self.variant != Variant::Eigen || self.collapsed.is_some();
        let packet = self.source == Source::Packet || self.active.len() > 30;
        ui.horizontal_top(|ui| {
            if show_ts && ts_w > look.px(160.0) {
                let span = match self.variant {
                    Variant::Eigen | Variant::Degenerate => 0.5,
                    Variant::Slosh => 1.0,
                    Variant::Packet => 0.25,
                    Variant::Moving => 0.15,
                };
                let t_hi = self.t.max(span);
                let mut ts = TimeSeriesPlot::new("box_ts")
                    .series(&self.sx)
                    .series(&self.sy)
                    .window(TimeWindow::Fixed(0.0, t_hi))
                    .now(self.t)
                    .y_range(0.0, lx.max(ly))
                    .labels("t", "")
                    .width(ts_w)
                    .height(height);
                if packet {
                    ts = ts.series(&self.sa);
                }
                if let Some(tr) = self.geom.revival_time()
                    && packet
                {
                    for (f, l) in [(0.25, "T_rev/4"), (0.5, "T_rev/2"), (1.0, "T_rev")] {
                        if self.t >= f * tr * 0.98 {
                            ts = ts.event(self.t0 + f * tr, l);
                        }
                    }
                }
                for (t, l) in &self.events {
                    ts = ts.event(*t, l.clone());
                }
                ts.show(ui);
            } else if ts_w > 0.0 {
                ui.add_space(ts_w);
            }
            if let Some(rep) = &self.rep_pos {
                self.scatter_ui(ui, rep, height);
            }
            if let Some(r) = &self.rep_e {
                HistogramPlot::new("box_hist_e", &r.hist)
                    .predicted(&r.pred)
                    .scale(HistScale::Fraction)
                    .x_label("E / ε (repeat N)")
                    .width(hist_w)
                    .height(height)
                    .show(ui);
            }
        });
    }

    fn readouts(&self, ui: &mut egui::Ui) {
        let look = Look::of(ui);
        let m = self.moments;
        let v = self.geom.v_axis();
        let mut items: Vec<(&str, String, Option<Color32>)> = vec![];
        let e_label;
        if let Some((i, j)) = self.single() {
            let (n1, n2, n3) = self.geom.triple(i, j);
            e_label = format!("E({n1},{n2},{n3})");
            items.push((
                e_label.as_str(),
                format!("{} ε = {:.2}", fmt_e(m.e), m.e * EPS),
                Some(look.accent()),
            ));
        } else {
            items.push(("⟨E⟩", format!("{:.3} ε", m.e), Some(look.accent())));
            items.push(("ΔE", format!("{:.3} ε", m.de), None));
        }
        let vx = format!("⟨{v}⟩");
        items.push(("⟨x⟩", fmt_num(m.x, 3), Some(look.series(0))));
        items.push((vx.as_str(), fmt_num(m.y, 3), Some(look.series(1))));
        items.push(("∫|ψ|²", format!("{:.3}", m.norm), None));
        if self.variant == Variant::Degenerate
            && self.active.len() <= 30
            && let Some(l) = self.basis.angular_momentum(&self.ct)
        {
            items.push(("⟨L⟩ (centre)", format!("{} ħ", fmt_num(l, 3)), Some(look.series(3))));
        }
        if let Some(tr) = self.geom.revival_time()
            && (self.source == Source::Packet || self.active.len() > 30)
        {
            items.push(("(t − t₀)/T_rev", format!("{:.3}", (self.t - self.t0) / tr), Some(look.series(2))));
        }
        readout_row(ui, &items);
    }

    fn view_bar(&mut self, ui: &mut egui::Ui) {
        let look = Look::of(ui);
        ui.horizontal(|ui| {
            ui.label(RichText::new("display").color(look.weak_text()));
            segmented(ui, |ui| {
                for d in Display::ALL {
                    if ui
                        .selectable_label(self.display == d, d.label())
                        .on_hover_text("display mode (V cycles)")
                        .clicked()
                    {
                        self.display = d;
                    }
                }
            });
            ui.label(RichText::new("V").monospace().color(look.weak_text()))
                .on_hover_text("key V cycles the display mode");
            ui.add_space(look.px(12.0));
            ui.label(RichText::new("view").color(look.weak_text()));
            segmented(ui, |ui| {
                if ui.selectable_label(!self.surface, "map").clicked() {
                    self.surface = false;
                }
                if ui
                    .selectable_label(self.surface, "surface")
                    .on_hover_text("|ψ|² as a height surface (drag to turn)")
                    .clicked()
                {
                    self.surface = true;
                }
            });
            ui.add_space(look.px(12.0));
            ui.label(RichText::new("plane").color(look.weak_text()));
            let mut g = self.geom;
            segmented(ui, |ui| {
                if ui
                    .selectable_label(g.plane == Plane::XY, "x–y (a × a)")
                    .on_hover_text("square cross-section: (n₁, n₂) ↔ (n₂, n₁) degenerate")
                    .clicked()
                {
                    g.plane = Plane::XY;
                }
                if ui
                    .selectable_label(g.plane == Plane::XZ, "x–z (a × b)")
                    .on_hover_text("rectangular cross-section: degeneracies depend on b/a")
                    .clicked()
                {
                    g.plane = Plane::XZ;
                }
            });
            self.set_geometry(g);
        });
    }
}

/// Points of the current-arrow grid.
const ARROWS: usize = 13;

fn arrow_grid(lx: f64, ly: f64) -> Vec<(f64, f64)> {
    let cell = lx.max(ly) / ARROWS as f64;
    let nu = (lx / cell).round().max(2.0) as usize;
    let nv = (ly / cell).round().max(2.0) as usize;
    let mut out = Vec::with_capacity(nu * nv);
    for iv in 0..nv {
        for iu in 0..nu {
            out.push(((iu as f64 + 0.5) * lx / nu as f64, (iv as f64 + 0.5) * ly / nv as f64));
        }
    }
    out
}

/// Energy range (E/ε) of the levels without the tails of weight `tail` on either side, padded by 5 %.
fn quantile_range(probs: &[(f64, f64, Vec<usize>)], tail: f64) -> [f64; 2] {
    let mut lo = probs.first().map_or(0.0, |l| l.0);
    let mut hi = probs.last().map_or(1.0, |l| l.0);
    let mut acc = 0.0;
    for l in probs {
        acc += l.1;
        if acc < tail {
            lo = l.0;
        }
    }
    acc = 0.0;
    for l in probs.iter().rev() {
        acc += l.1;
        if acc < tail {
            hi = l.0;
        }
    }
    let pad = 0.05 * (hi - lo).max(1.0);
    [(lo - pad).max(0.0), hi + pad]
}

/// "all but 3e-9" or "99.2 %".
fn fmt_captured(c: f64) -> String {
    if c > 0.9999 {
        format!("all but {:.0e}", (1.0 - c).max(1e-16))
    } else {
        format!("{:.2} %", 100.0 * c)
    }
}

/// E/ε compactly: "5", "13.5", "9.333".
fn fmt_e(e: f64) -> String {
    if (e - e.round()).abs() < 1e-9 {
        format!("{}", e.round() as i64)
    } else {
        let s = format!("{e:.3}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// "1", "i", "−1", "−i", "e^(iπ·0.25)".
fn fmt_phase(phase: f64) -> String {
    let p = phase.rem_euclid(2.0);
    for (v, s) in [(0.0, "1"), (0.5, "i"), (1.0, "−1"), (1.5, "−i"), (2.0, "1")] {
        if (p - v).abs() < 1e-9 {
            return s.into();
        }
    }
    format!("e^(i·{:.2}π)", phase)
}

impl Scene for BoxScene {
    fn time(&self) -> f64 {
        self.t
    }

    fn time_scale(&self) -> f64 {
        match self.variant {
            Variant::Eigen => 0.04,
            Variant::Slosh => 0.08,
            Variant::Degenerate => 0.04,
            Variant::Packet => 0.02,
            Variant::Moving => 0.01,
        }
    }

    fn step(&mut self, dt: f64, _ctx: &mut Ctx) {
        let t_start = self.t;
        self.advance_tracers(t_start, dt);
        let target = self.t + dt;
        if self.sx.last().is_none_or(|p| target - p[0] >= self.sample_dt) {
            self.record_until(target);
        }
        self.t = target;
        self.ct_time = f64::NAN;
    }

    fn reset(&mut self, _ctx: &mut Ctx) {
        self.restart();
    }

    fn measure_label(&self) -> Option<String> {
        Some("Measure position".into())
    }

    fn measure(&mut self, ctx: &mut Ctx) {
        self.measure_position(ctx);
    }

    fn on_key(&mut self, key: egui::Key, _m: egui::Modifiers, ctx: &mut Ctx) -> bool {
        match key {
            egui::Key::V => self.display = self.display.next(),
            egui::Key::E => self.measure_energy(ctx),
            _ => return false,
        }
        true
    }

    fn action(&mut self, action: &str, ctx: &mut Ctx) -> bool {
        let mut words = action.split_whitespace();
        let cmd = words.next().unwrap_or("");
        let args: Vec<&str> = words.collect();
        let num = |k: usize| args.get(k).and_then(|w| w.parse::<f64>().ok());
        let n = num(0).map_or(self.repeat_n, |v| v as usize);
        let on = |k: usize| args.get(k).is_none_or(|w| *w != "off" && *w != "0");
        match cmd {
            "measure" | "measure_x" | "measure_pos" => self.measure_position(ctx),
            "measure_e" => self.measure_energy(ctx),
            "repeat_x" | "repeat_pos" => self.repeat_position(n, ctx),
            "repeat_e" => self.repeat_energy(n, ctx),
            "surface" | "view_surface" => self.surface = true,
            "map" | "view_map" => self.surface = false,
            "plane_xy" | "plane_xz" => {
                let g = Geometry {
                    plane: if cmd == "plane_xy" { Plane::XY } else { Plane::XZ },
                    ..self.geom
                };
                self.set_geometry(g);
            }
            "ratio" => {
                let Some(r) = args.first().and_then(|w| match *w {
                    "sqrt2" => Some(2f64.sqrt()),
                    "sqrt3" => Some(3f64.sqrt()),
                    _ => w.parse().ok(),
                }) else {
                    return false;
                };
                self.set_geometry(Geometry { ratio: r, ..self.geom });
            }
            "hidden" | "n3" => {
                let Some(h) = num(0) else { return false };
                self.set_geometry(Geometry {
                    hidden: (h as usize).clamp(1, 9),
                    ..self.geom
                });
            }
            "state" | "add" => {
                let (Some(i), Some(j)) = (num(0), num(1)) else { return false };
                let (i, j) = ((i as usize).clamp(1, M), (j as usize).clamp(1, M));
                if cmd == "state" {
                    self.select_single(i, j);
                } else {
                    self.toggle_member(i, j);
                }
            }
            "rel_phase" => {
                let ph = match args.first().copied() {
                    Some("1") => 0.0,
                    Some("i") => 0.5,
                    Some("-1") => 1.0,
                    Some("-i") => 1.5,
                    Some(w) => match w.parse::<f64>() {
                        Ok(v) => v,
                        Err(_) => return false,
                    },
                    None => return false,
                };
                if let Some(m) = self.members.get_mut(1) {
                    m.phase = ph;
                }
                self.restart();
            }
            "packet" => {
                self.source = Source::Packet;
                self.restart();
            }
            "jump" => {
                let Some(t) = num(0) else { return false };
                self.jump(t);
            }
            "jump_rev" => {
                let (Some(f), Some(tr)) = (num(0), self.geom.revival_time()) else {
                    return false;
                };
                self.jump(self.t0 + f * tr);
            }
            "current" => self.show_current = on(0),
            "flow" => {
                self.show_flow = on(0);
                self.seed_tracers();
            }
            "nodes" => self.show_nodes = on(0),
            "bright_amp" => self.bright_amp = true,
            "bright_density" => self.bright_amp = false,
            "scale_auto" => self.bright_auto = true,
            "scale_fixed" => self.bright_auto = false,
            "azimuth" => {
                let Some(a) = num(0) else { return false };
                self.azimuth = a as f32;
            }
            "width" => {
                let Some(w) = num(0) else { return false };
                self.width = w.clamp(0.02, 0.3);
            }
            "click_adds" => self.click_adds = on(0),
            _ if cmd.starts_with("style_") => match Display::parse(&cmd[6..]) {
                Some(d) => self.display = d,
                None => return false,
            },
            _ => return false,
        }
        true
    }

    fn central_ui(&mut self, ui: &mut egui::Ui, _ctx: &mut Ctx) {
        let look = Look::of(ui);
        self.view_bar(ui);
        self.ensure_field();
        self.readouts(ui);
        let show_bottom = self.variant != Variant::Eigen || self.collapsed.is_some() || self.rep_pos.is_some() || self.rep_e.is_some();
        let bottom_h = if show_bottom {
            look.px(165.0).min(ui.available_height() * 0.26)
        } else {
            0.0
        };
        let avail = ui.available_size();
        let main_h = (avail.y - bottom_h - if show_bottom { look.px(8.0) } else { 0.0 }).max(200.0);
        // column A as wide as the map needs at full height, column B (thumbnails, ladder) the rest
        let (ml, mr, mt, mb) = PlotFrame::margins(&look, false);
        let map_h = main_h - look.px(104.0).min(main_h * 0.24) - mt - mb - look.px(6.0);
        let want_a = map_h * (self.geom.lx() / self.geom.ly()) as f32 + ml + mr + look.px(80.0) + look.px(6.0) + 2.0;
        let min_b = (avail.x * 0.34).clamp(240.0, 480.0);
        let col_a = want_a.min(avail.x - min_b - look.px(12.0)).max(200.0);
        let col_b = (avail.x - col_a - look.px(12.0)).max(look.px(200.0));
        ui.horizontal_top(|ui| {
            ui.allocate_ui_with_layout(vec2(col_a, main_h), egui::Layout::top_down(egui::Align::Min), |ui| {
                ui.set_width(col_a);
                if self.surface {
                    self.surface_ui(ui, col_a, main_h);
                } else {
                    self.map_ui(ui, col_a, main_h);
                }
            });
            ui.add_space(look.px(8.0));
            ui.allocate_ui_with_layout(vec2(col_b, main_h), egui::Layout::top_down(egui::Align::Min), |ui| {
                ui.set_width(col_b);
                let th = (main_h * 0.4).min(col_b * 0.9);
                self.thumbs_ui(ui, col_b, th);
                ui.add_space(look.px(4.0));
                let (levels, range, compact) = self.ladder_levels();
                let sel = self.single();
                let h = (main_h - th - look.px(8.0)).max(look.px(120.0));
                let title = if compact {
                    "E / ε  (spectrum of the state)".to_string()
                } else {
                    format!(
                        "E / ε   ({} fixed: + {} ε)",
                        self.geom.hidden_index(),
                        fmt_e(self.geom.hidden_e_eps())
                    )
                };
                let out = GroupedLadder::new("box_ladder", &levels, range, vec2(col_b, h))
                    .selected(sel)
                    .compact(compact)
                    .title(title)
                    .show(ui);
                if matches!(self.flash.kind(), Some(FlashKind::Level { .. })) {
                    self.flash.paint(ui, &out.transform, Domain::Position);
                }
                if let Some((i, j)) = out.clicked {
                    if self.click_adds && !compact {
                        self.toggle_member(i, j);
                    } else {
                        self.select_single(i, j);
                    }
                }
            });
        });
        if show_bottom {
            ui.add_space(look.px(8.0));
            self.bottom_ui(ui, bottom_h);
        }
    }

    fn side_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx) {
        let look = ctx.look;
        let weak = |s: String| RichText::new(s).color(look.weak_text());

        // ---------------- the box
        ui.label(RichText::new("Box  0 ≤ x, y ≤ a,  0 ≤ z ≤ b").strong());
        let mut g = self.geom;
        ui.radio_value(&mut g.plane, Plane::XY, "x–y plane (a × a, square: degeneracies)");
        ui.radio_value(&mut g.plane, Plane::XZ, "x–z plane (a × b)");
        // the slider rounds its value to 3 decimals: take it only when really moved, so that the exact √2, √3 of the buttons stay
        let mut r = g.ratio;
        let mut shown = r;
        ui.add(egui::Slider::new(&mut shown, 0.5..=2.5).max_decimals(3).text("b / a"));
        if (shown - r).abs() > 6e-4 {
            r = shown;
        }
        ui.horizontal(|ui| {
            ui.label(weak("special b/a:".into()));
            for (label, v) in [("1", 1.0), ("√2", 2f64.sqrt()), ("√3", 3f64.sqrt()), ("2", 2.0)] {
                if ui.small_button(label).clicked() {
                    r = v;
                }
            }
        });
        g.ratio = r;
        let label = format!("{} ({} direction, fixed)", g.hidden_index(), g.hidden_axis());
        ui.add(egui::Slider::new(&mut g.hidden, 1..=5).text(label));
        self.set_geometry(g);
        let gm = self.geom;
        ui.label(weak(format!(
            "E(n₁,n₂,n₃) = (π²ħ²/2m)[(n₁²+n₂²)/a² + n₃²/b²] = ε(n₁² + n₂² + n₃²a²/b²),  ε = π²ħ²/(2ma²){}",
            if gm.plane == Plane::XY {
                format!(
                    ";  here every level gets + {}²·(a/b)² ε = {} ε from {}",
                    gm.hidden,
                    fmt_e(gm.hidden_e_eps()),
                    gm.hidden_index()
                )
            } else {
                format!(";  here + {}² ε = {} ε from n₂", gm.hidden, fmt_e(gm.hidden_e_eps()))
            }
        )));
        if let Some((i, j)) = self.single() {
            let (n1, n2, n3) = gm.triple(i, j);
            let e = e3_eps(n1, n2, n3, gm.ratio);
            ui.label(
                RichText::new(format!(
                    "E({n1},{n2},{n3}) = ε({} + {} + {}·{:.3}) = {} ε",
                    n1 * n1,
                    n2 * n2,
                    n3 * n3,
                    1.0 / (gm.ratio * gm.ratio),
                    fmt_e(e)
                ))
                .color(look.accent()),
            );
            let gp = plane_degeneracy(&gm, e);
            let d3 = degenerate_3d(e, gm.ratio, 40);
            let list = |v: &[String]| {
                if v.len() > 8 {
                    format!("{} …", v[..8].join(" "))
                } else {
                    v.join(" ")
                }
            };
            let p_names: Vec<String> = gp.iter().map(|(a, b)| format!("({a},{b})")).collect();
            let d_names: Vec<String> = d3.iter().map(|(a, b, c)| format!("({a},{b},{c})")).collect();
            ui.label(weak(format!(
                "in this plane: g = {}  {}\nin the 3D box: g = {}  {}",
                gp.len(),
                list(&p_names),
                d3.len(),
                list(&d_names)
            )));
        }

        // ---------------- the state
        ui.add_space(look.px(8.0));
        ui.label(RichText::new("State").strong());
        ui.horizontal(|ui| {
            if ui.selectable_label(self.source == Source::Members, "eigenstates").clicked() && self.source != Source::Members {
                self.source = Source::Members;
                self.restart();
            }
            if ui.selectable_label(self.source == Source::Packet, "Gaussian packet").clicked() && self.source != Source::Packet {
                self.source = Source::Packet;
                self.restart();
            }
        });
        let mut changed = false;
        match self.source {
            Source::Members => {
                ui.horizontal(|ui| {
                    ui.label(weak("thumbnail click:".into()));
                    ui.selectable_value(&mut self.click_adds, false, "show alone");
                    ui.selectable_value(&mut self.click_adds, true, "add / remove");
                });
                let v_idx = gm.v_index();
                let mut remove = None;
                let n_members = self.members.len();
                for (k, m) in self.members.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("({},{})", m.i, m.j)).monospace().color(look.accent()))
                            .on_hover_text(format!("n₁ = {}, {v_idx} = {}: E = {} ε", m.i, m.j, fmt_e(gm.e_eps(m.i, m.j))));
                        changed |= ui
                            .add(egui::DragValue::new(&mut m.amp).range(0.0..=1.0).speed(0.01).prefix("|c| "))
                            .changed();
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut m.phase)
                                    .range(-1.0..=1.0)
                                    .speed(0.01)
                                    .prefix("phase ")
                                    .suffix(" π"),
                            )
                            .changed();
                        if n_members > 1 && ui.small_button("✕").on_hover_text("remove").clicked() {
                            remove = Some(k);
                        }
                    });
                }
                if let Some(k) = remove {
                    self.members.remove(k);
                    changed = true;
                }
                ui.label(weak(
                    "c ∝ |c| e^(iπ·phase), normalised automatically. Changing the state restarts at t = 0.".into(),
                ));
                if self.members.len() == 2 {
                    ui.horizontal(|ui| {
                        let (a, b) = (self.members[0], self.members[1]);
                        ui.label(weak(format!("relative phase of ({},{}):", b.i, b.j)));
                        for (label, ph) in [("1", 0.0), ("i", 0.5), ("−1", 1.0), ("−i", 1.5)] {
                            let cur = (self.members[1].phase - a.phase).rem_euclid(2.0);
                            if ui.selectable_label((cur - ph).abs() < 1e-9, label).clicked() {
                                self.members[1].phase = a.phase + ph;
                                changed = true;
                            }
                        }
                    });
                    let (a, b) = (self.members[0], self.members[1]);
                    let degenerate = (gm.e_eps(a.i, a.j) - gm.e_eps(b.i, b.j)).abs() < 1e-7;
                    ui.label(weak(if degenerate {
                        format!(
                            "same energy: any combination is stationary; relative phase now {}",
                            fmt_phase(b.phase - a.phase)
                        )
                    } else {
                        let de = (gm.e_eps(a.i, a.j) - gm.e_eps(b.i, b.j)).abs();
                        format!(
                            "ΔE = {} ε: the pattern repeats with T = 2πħ/ΔE = {:.3}",
                            fmt_e(de),
                            2.0 * PI / (EPS * de)
                        )
                    }));
                }
            }
            Source::Packet => {
                let p = &mut self.packet;
                changed |= ui.add(egui::Slider::new(&mut p.fu, 0.15..=0.85).text("x₀ / a")).changed();
                changed |= ui
                    .add(egui::Slider::new(&mut p.fv, 0.15..=0.85).text(format!(
                        "{}₀ / {}",
                        gm.v_axis(),
                        if gm.plane == Plane::XY { "a" } else { "b" }
                    )))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut p.sigma, 0.04..=0.15).text("σ (|ψ|² width)"))
                    .changed();
                changed |= ui.add(egui::Slider::new(&mut p.p, 0.0..=70.0).text("|p| (ħ/a)")).changed();
                changed |= ui
                    .add(egui::Slider::new(&mut p.angle, -180.0..=180.0).text("direction (°)"))
                    .changed();
                ui.label(weak(format!(
                    "n̄ = |p|a/(πħ) ≈ {:.1};  the {M} × {M} modes capture {} of the packet",
                    p.p * A / PI,
                    fmt_captured(self.captured)
                )));
            }
        }
        if changed {
            self.restart();
        }
        if let Some(c) = &self.collapsed {
            ui.label(RichText::new(format!("Now: {c}. R restarts from the settings above.")).color(look.measure()));
        }

        // ---------------- time
        ui.add_space(look.px(8.0));
        ui.label(RichText::new("Time").strong());
        match gm.revival_time() {
            Some(tr) => {
                ui.label(weak(format!(
                    "revival T_rev = {:.4} (= {}·4ma²/(πħ)); t − t₀ = {:.4}",
                    tr,
                    fmt_e(tr / (4.0 / PI)),
                    self.t - self.t0
                )));
                ui.horizontal(|ui| {
                    ui.label(weak("jump to t₀ +".into()));
                    for (label, f) in [("T_rev/4", 0.25), ("T_rev/2", 0.5), ("T_rev", 1.0)] {
                        if ui.button(label).clicked() {
                            self.jump(self.t0 + f * tr);
                        }
                    }
                });
            }
            None => {
                ui.label(weak(
                    "b²/a² is not a ratio of small integers: no common revival time of the x and z motions.".into(),
                ));
            }
        }

        // ---------------- view
        ui.add_space(look.px(8.0));
        ui.label(RichText::new("View").strong());
        ui.checkbox(&mut self.show_nodes, "node lines (real ψ) / phase vortices ○");
        ui.checkbox(&mut self.show_current, "probability current j (arrows)");
        let fp = self.flow_possible();
        ui.add_enabled_ui(fp, |ui| {
            if ui
                .checkbox(&mut self.show_flow, "flow markers (carried by v = j/|ψ|²)")
                .on_disabled_hover_text(format!("only for states of at most {FLOW_MAX_MODES} eigenstates"))
                .changed()
            {
                self.seed_tracers();
            }
        });
        if self.display == Display::Phase {
            ui.horizontal(|ui| {
                ui.label(weak("brightness ∝".into()));
                ui.selectable_value(&mut self.bright_amp, false, "|ψ|²");
                ui.selectable_value(&mut self.bright_amp, true, "|ψ|");
            });
        }
        ui.horizontal(|ui| {
            ui.label(weak("colour scale:".into()));
            ui.selectable_value(&mut self.bright_auto, false, "fixed")
                .on_hover_text("set at the start, only grows: a spreading state gets darker");
            ui.selectable_value(&mut self.bright_auto, true, "current maximum")
                .on_hover_text("rescaled every frame: shows the structure of a spread-out state");
        });

        // ---------------- measure
        ui.add_space(look.px(8.0));
        ui.label(RichText::new("Measure").strong());
        ui.add(
            egui::Slider::new(&mut self.width, 0.02..=0.3)
                .logarithmic(true)
                .text("detector w (position)"),
        );
        ui.horizontal(|ui| {
            if ui.button(RichText::new("Measure position  (M)").color(look.measure())).clicked() {
                self.measure_position(ctx);
            }
            if ui.button(RichText::new("Measure energy  (E)").color(look.measure())).clicked() {
                self.measure_energy(ctx);
            }
        });
        if let Some(n) = repeat_button(ui, "box_rep_x", "Repeat position", &mut self.repeat_n) {
            self.repeat_position(n, ctx);
        }
        if let Some(n) = repeat_button(ui, "box_rep_e", "Repeat energy", &mut self.repeat_n) {
            self.repeat_energy(n, ctx);
        }
        ui.label(
            RichText::new(
                "Single shot: ψ collapses (position: onto a spot of width w; energy: onto the whole eigenspace of the outcome) and keeps evolving. Repeat N: N fresh copies of the current ψ, compared with the Born rule.",
            )
            .color(look.weak_text())
            .size(12.0 * look.font_scale()),
        );
        if let Some(l) = self.flash.label() {
            ui.label(RichText::new(format!("last outcome: {l}")).color(look.measure()));
        }
    }

    fn notes(&self) -> String {
        let main = match self.variant {
            Variant::Eigen => NOTES_EIGEN,
            Variant::Slosh => NOTES_SLOSH,
            Variant::Degenerate => NOTES_DEGENERATE,
            Variant::Packet => NOTES_PACKET,
            Variant::Moving => NOTES_MOVING,
        };
        format!("{main}\n\n{NOTES_MEASURE}")
    }
}

const NOTES_EIGEN: &str = "\
# The exercise
PS09 38: a particle in the box 0 ≤ x, y ≤ a, 0 ≤ z ≤ b with infinitely high walls. Units ħ = m = 1 and a = 1; energies are given in ε = π²ħ²/(2ma²).
- (a) Separation of variables: ψ = X(x)Y(y)Z(z) with X ∝ sin(n₁πx/a), Y ∝ sin(n₂πy/a), Z ∝ sin(n₃πz/b): each factor is a 1D box state, and E = E₁ + E₂ + E₃.
- (b) Normalised: ψ = √(8/(a²b)) sin(n₁πx/a) sin(n₂πy/a) sin(n₃πz/b).
- (c) E(n₁,n₂,n₃) = (π²ħ²/2m)[(n₁² + n₂²)/a² + n₃²/b²] = ε(n₁² + n₂² + n₃²a²/b²).
- The map is a cross-section: in the x–y plane it shows the 2D factor (2/a) sin(n₁πx/a) sin(n₂πy/a); the z factor is one constant on any plane z = const, so only the brightness scale would change. The third quantum number (n₃ here) is fixed in the side panel and enters E.
- Degeneracy: in the square x–y cross-section E depends on n₁² + n₂² only, so (n₁, n₂) and (n₂, n₁) always have the same energy (g ≥ 2 for n₁ ≠ n₂). Some levels have more states, from different sums of two squares: 1² + 7² = 5² + 5² = 50 gives g = 3. In 3D more states can share a level, e.g. for b = a: (1,1,2), (1,2,1), (2,1,1); the side panel lists them.

# Try
- Click the thumbnails (n₁ across, n₂ up): the node lines are straight, n₁ − 1 vertical and n₂ − 1 horizontal ones.
- The side plots P(x) = ∫|ψ|²dy and P(y) = ∫|ψ|²dx are, for an eigenstate, exactly the 1D box densities (2/a) sin²(nπx/a) (dashed): the separation of variables of part (a).
- Play: only the colour turns (the phase e^(−iEt/ħ)); |ψ|² does not change: a stationary state. In Re ψ the pattern oscillates as cos(Et/ħ).
- The ladder: degenerate states share one row (one dash each, g = …); (1,2) and (2,1) are different states (mirror images about the diagonal) with the same energy. Click a dash to show that state.
- Choose the x–z plane (a × b) and move b/a away from 1: the pairs split. At special ratios other, accidental degeneracies appear, e.g. b = 2a: (n₁, n₃) = (1,4) and (2,2); b = √3·a: (1,5), (2,4), (3,1) (buttons in the side panel).
- In the x–y plane, change n₃ or b/a: every level moves by the same n₃²(a/b)²ε, the picture and the degeneracies stay.

# Behind the scenes
Exact eigenfunctions sampled on a 192 × 192 grid; time evolution by the exact phases e^(−iE(n₁,n₂,n₃)t/ħ) (the fixed quantum number only adds a global phase). Energies that agree to 10⁻⁷ (relative) count as degenerate, so a slider value near a special ratio shows nearly, not exactly, degenerate levels; the buttons set the exact ratios.";

const NOTES_SLOSH: &str = "\
# Superposition of two energies: sloshing
ψ = [φ(1,1) + φ(2,1)]/√2 with n₃ fixed (PS09 38, parts a–c). The two states have different energies, so |ψ|² is not stationary.
- |ψ(t)|² = ½[φ₁₁² + φ₂₁² + 2φ₁₁φ₂₁ cos(ΔE t/ħ)] with ΔE = E(2,1) − E(1,1) = 3ε: the density sloshes along x with period T = 2πħ/ΔE = 4ma²/(3πħ) ≈ 0.42.
- ⟨x⟩(t) = a/2 − (16a/9π²) cos(ΔE t/ħ), while y stays in its ground state: the x and y motions separate.
- Arrows: the probability current j = (ħ/m) Im(ψ*∇ψ), along x, reversing every half period (∂|ψ|²/∂t + ∇·j = 0).

# Try
- Set the thumbnail click to add / remove and add (1,2) or (2,2); change amplitudes and phases.
- Measure energy (E): ψ collapses onto φ(1,1) or φ(2,1), each with probability ½, and the sloshing stops.
- Repeat energy: the histogram of outcomes approaches |c|² per level.
- Measure position: the collapsed packet is a superposition of many energies and spreads over the box.";

const NOTES_DEGENERATE: &str = "\
# Degenerate pair: circulation or node line
ψ = [φ(1,2) + e^(iα) φ(2,1)]/√2, here α = π/2 (relative phase i). Both states have E = 5ε plus the fixed n₃ part (PS09 38 c).
- Any combination of degenerate states is again an eigenstate with the same E: |ψ|² is stationary for every α; only the global phase e^(−iEt/ħ) turns.
- Relative phase i: ψ vanishes only at the centre and its phase winds once around it (a vortex, marked ○). As time passes the colours rotate around the centre. The current j = (ħ/m) Im(ψ*∇ψ) circulates (arrows, flow markers): a stationary flow with angular momentum ⟨L⟩ ≠ 0 about the centre (readout; not a multiple of ħ, since ψ is not an eigenstate of L: the square box has no rotational symmetry).
- Relative phase 1: ψ is real up to the global phase, j = 0 everywhere and ψ ∝ sin(πx/a) sin(πy/a)[cos(πx/a) + cos(πy/a)] has a straight diagonal node line x + y = a. Phase −1: the other diagonal, x = y.
- An energy measurement cannot distinguish these states: it projects onto the whole degenerate subspace, so here it changes nothing (P = 1).

# Try
- Switch the relative phase (side panel) between i, 1, −1, −i: circulation one way, node line, the other node line, circulation the other way.
- Play: the density stays, the colour pattern turns.
- Choose the x–z plane with b ≠ a: (1,2) and (2,1) (now (n₁, n₃)) are no longer degenerate, so the relative phase now runs, with period 2πħ/ΔE: the state cycles through circulation, node line, opposite circulation, other node line, and |ψ|² is no longer stationary.
- Measure energy: the outcome is E = 6ε (5ε from n₁, n₂ plus 1ε from n₃ = 1, b = a) with P = 1, and the state is not changed. Then measure position and watch the collapse spread.

# Behind the scenes
Flow markers are moved with the velocity field v = j/|ψ|² of the probability fluid; they visualise the current and are not measured particle paths (speeds are capped close to the vortex, where v grows like 1/r).";

const NOTES_PACKET: &str = "\
# Gaussian packet: spreading, reflection and revival
ψ(x, y, 0) ∝ exp(−|r − r₀|²/(4σ²)) at rest, σ = 0.05a, expanded in 40 × 40 eigenstates of the box (part a); then each coefficient simply turns, c(t) = c e^(−iE(n₁,n₂,n₃)t/ħ) (part c).
- The packet spreads (free-particle time scale 2mσ²/ħ = 0.005), reflects from the walls and interferes with itself: a regular, changing pattern of fringes.
- Revival: E ∝ n₁² + n₂², so at T_rev = 4ma²/(πħ) (= 4/π ≈ 1.27 here) every phase e^(−iEt/ħ) is back to the same value: ψ(T_rev) = ψ(0) up to a global phase, for any state.
- At T_rev/2 the packet reappears point-mirrored through the centre, ψ(x, y) → ψ(a − x, a − y); at T_rev/4 four copies (the original and its three mirror images) overlap.
- Assumptions of T_rev = 4ma²/(πħ): infinitely hard walls and no potential inside, so E ∝ n² exactly (non-relativistic); a square cross-section (for a × b only when b²/a² = p/q is a ratio of integers, then T = p·4ma²/(πħ)). It does not depend on the state or on the truncation to 40 × 40 modes; the fixed third quantum number adds only a global phase.
- The packet is a product g(x)h(y) and H = H_x + H_y, so it stays a product: P(x, t) is exactly a packet in a 1D box.

# Try
- Jump to t₀ + T_rev/4, T_rev/2, T_rev (side panel) and compare with t = 0. The time plot shows |⟨ψ(t₀)|ψ(t)⟩|² returning to 1.
- The ladder shows the spectrum of the packet: many levels, each with P = Σ|c|².
- Measure position at some t: the new packet starts its own revival cycle from t₀ = t.
- Give the packet a momentum (side panel) or load the next preset.

# Behind the scenes
Expansion coefficients by quadrature of the Gaussian along x and y (it is separable); 40 × 40 modes capture all but ~10⁻⁹ of the norm; display grid 192 × 192. Brightness ∝ |ψ| by default so that the spread-out state stays visible.";

const NOTES_MOVING: &str = "\
# Packet with momentum: a quantum billiard
σ = 0.07a, p₀ = 58 ħ/a at 31° (n̄ = p₀a/πħ ≈ 18), expanded in 40 × 40 box eigenstates and evolved exactly.
- At first it moves like a classical particle with v = p₀/m and reflects specularly; ⟨x⟩ and ⟨y⟩ zig-zag between the walls. During a reflection the incoming and reflected parts interfere (fringes at the wall).
- It spreads (σ(t) ≈ ħt/(2mσ) once t ≫ 2mσ²/ħ ≈ 0.01), so after a few crossings it fills the box and the zig-zag of ⟨x⟩ dies out.
- At T_rev = 4ma²/(πħ) ≈ 1.27 (about 30 round trips along x) it is back exactly; at T_rev/2 point-mirrored, moving the other way. Same assumptions as in the previous preset: hard walls, E ∝ n² exactly, square cross-section.

# Try
- Watch the first reflections at speed 1, then jump to T_rev/2 and T_rev.
- Measure position while it is spread out: a new localised packet appears; it keeps the local phase of ψ (the local momentum) and has a momentum spread of about ħ/(2w).
- Change |p| and the direction; with |p| = 0 the packet only spreads.

# Behind the scenes
The highest mode n = 40 has p = 40πħ/a ≈ 126ħ/a, well above p₀ + 4σ_p ≈ 86ħ/a, so the 40 × 40 expansion captures the packet (the side panel shows the fraction).";

const NOTES_MEASURE: &str = "\
# Measurements
- Position (M): an outcome r₀ is drawn from |ψ|² (Born rule), then ψ is multiplied by the detector window exp(−|r − r₀|²/(4w²)) and renormalised: the particle is localised within w and keeps evolving (drawing r₀ from |ψ|² is the ideal-detector limit, as in the 1D tab).
- Energy (E): a level E is drawn with probability Σ|c|² over its states; ψ is projected onto the whole eigenspace of E. For a degenerate level that is the span of all its states, with their amplitudes and phases kept, not one particular (n₁, n₂).
- Repeat N: N fresh copies of the current ψ (ψ itself is not changed). Positions appear as dots building up |ψ|²; energies fill a histogram compared with the Born rule Σ|c|² per level.";
