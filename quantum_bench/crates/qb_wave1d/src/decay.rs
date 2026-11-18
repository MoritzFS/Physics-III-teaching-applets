//! PS09 36 · The decaying particle: V = V₀(x) − iΓ/2 (ħ = 1).
//!
//! The split-step propagator runs with the complex potential, so the total
//! probability P(t) = ∫|ψ|² dx falls as e^(−Γt) whatever V₀ and the state are.
//! P(t) is plotted against e^(−Γt) with τ = 1/Γ; V₀ is a free particle, a
//! harmonic well or a box (with the level ladder, whose bars all shrink at the
//! same rate). Measurement: "is the particle still there?"; repeat N follows N
//! copies and builds the survival curve and the histogram of decay times.

use std::sync::Arc;

use qb_core::egui::{self, Align2, RichText, pos2};
use qb_core::num::{Absorber, C64, EigenBasis, Grid1D, Momentum, ShotHistogram, SplitStep, Wave1D, c64};
use qb_core::widgets::{
    Domain, FlashKind, HistScale, HistogramPlot, MeasureFlash, PanZoom, PlotFrame, ReprView, TimeSeries, WaveCurve, WavePlot, WaveStyle,
    fmt_num, plot_dashed, plot_line, readout_row, repeat_button, segmented, view_bar,
};
use qb_core::{Ctx, Look, Scene};

use crate::common::{self, EnergyView, LevelCurve, grow_top, hint, section};

/// The real part V₀ of the potential.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Well {
    Free,
    Harmonic,
    Box,
}

impl Well {
    const ALL: [Well; 3] = [Well::Free, Well::Harmonic, Well::Box];
    fn label(self) -> &'static str {
        match self {
            Well::Free => "free (V₀ = 0)",
            Well::Harmonic => "harmonic well",
            Well::Box => "box",
        }
    }
    fn parse(s: &str) -> Option<Self> {
        match s {
            "free" => Some(Well::Free),
            "harmonic" | "ho" => Some(Well::Harmonic),
            "box" => Some(Well::Box),
            _ => None,
        }
    }
}

/// Half-width of the box and the height of its walls (finite, so that the
/// propagator and the eigen-solver see the same potential).
const BOX_HALF: f64 = 6.0;
const BOX_WALL: f64 = 80.0;
const LEVELS: usize = 30;
/// width of the level ladder beside the x plot (nominal points)
const LADDER_W: f32 = 170.0;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Init {
    Packet,
    Level(usize),
}

/// N copies prepared at `t0`, each with a random decay time (memoryless: exponential).
struct Survival {
    t0: f64,
    gamma: f64,
    /// decay times measured from t0
    lifetimes: Vec<f64>,
    hist: ShotHistogram,
    predicted: Vec<f64>,
}

impl Survival {
    fn fraction(&self, dt: f64) -> f64 {
        let alive = self.lifetimes.iter().filter(|&&l| l > dt).count();
        alive as f64 / self.lifetimes.len().max(1) as f64
    }
}

pub struct DecayScene {
    // parameters
    gamma: f64,
    well: Well,
    init: Init,
    repeat_n: usize,
    view: ReprView,
    style: WaveStyle,

    // state
    grid: Grid1D,
    xs: Vec<f64>,
    v0: Vec<f64>,
    basis: Option<Arc<EigenBasis>>,
    level_labels: Vec<String>,
    wave: Wave1D,
    mom: Momentum,
    prop: SplitStep,
    t: f64,
    /// time of the last preparation (start, or a "yes" answer that renormalised ψ)
    t_anchor: f64,
    /// false after the answer "no, it has decayed"
    present: bool,
    p_series: TimeSeries,
    events: Vec<(f64, String)>,
    survival: Option<Survival>,
    flash: MeasureFlash,
    top_x: f64,
    top_p: f64,
}

impl DecayScene {
    pub fn new() -> Self {
        Self::with_well(Well::Harmonic)
    }

    pub fn with_well(well: Well) -> Self {
        let grid = Grid1D::new(-20.0, 20.0, 1024);
        let wave = Wave1D::zeros(grid);
        let mom = wave.momentum();
        let mut s = DecayScene {
            gamma: 0.2,
            well,
            init: Init::Packet,
            repeat_n: 1000,
            view: ReprView::X,
            style: WaveStyle::Density,
            grid,
            xs: grid.xs(),
            v0: vec![0.0; grid.n],
            basis: None,
            level_labels: vec![],
            wave,
            mom,
            prop: SplitStep::new(grid, 0.01),
            t: 0.0,
            t_anchor: 0.0,
            present: true,
            p_series: TimeSeries::new("P(t) = ∫|ψ|² dx"),
            events: vec![],
            survival: None,
            flash: MeasureFlash::new(),
            top_x: 0.0,
            top_p: 0.0,
        };
        s.build_well();
        s.restart();
        s
    }

    /// Grid, V₀, eigenbasis and propagator for the chosen well.
    fn build_well(&mut self) {
        self.grid = match self.well {
            Well::Free => Grid1D::new(-200.0, 200.0, 4096),
            _ => Grid1D::new(-20.0, 20.0, 1024),
        };
        let g = self.grid;
        self.xs = g.xs();
        self.v0 = match self.well {
            Well::Free => vec![0.0; g.n],
            Well::Harmonic => self.xs.iter().map(|x| 0.5 * x * x).collect(),
            // walls smoothed over ~0.15: a sharp wall makes the split-step heat spurious high momenta
            Well::Box => self
                .xs
                .iter()
                .map(|x| BOX_WALL * 0.5 * (1.0 + ((x.abs() - BOX_HALF) / 0.15).tanh()))
                .collect(),
        };
        self.basis = match self.well {
            Well::Free => None,
            _ => Some(Arc::new(EigenBasis::numeric(g, &self.v0, LEVELS))),
        };
        let first = if self.well == Well::Box { 1 } else { 0 };
        self.level_labels = (0..LEVELS).map(|k| format!("{}", k + first)).collect();
        self.prop = SplitStep::new(g, if self.well == Well::Free { 0.05 } else { 0.01 });
        if self.well == Well::Free {
            self.prop.set_absorber(Some(Absorber::new(20.0, 2.0)));
        }
        self.set_potential();
    }

    /// V = V₀ − iΓ/2 in the propagator.
    fn set_potential(&mut self) {
        let vc: Vec<C64> = self.v0.iter().map(|&v| c64(v, -0.5 * self.gamma)).collect();
        self.prop.set_complex_potential(&vc);
    }

    fn initial_wave(&self) -> Wave1D {
        if let (Init::Level(n), Some(b)) = (self.init, &self.basis) {
            return b.wave(n.min(b.len() - 1));
        }
        match self.well {
            Well::Free => Wave1D::gaussian(self.grid, -20.0, 1.5, 1.0),
            Well::Harmonic => Wave1D::gaussian(self.grid, 2.5, 0.6, 0.0),
            Well::Box => Wave1D::gaussian(self.grid, -2.0, 0.8, 2.0),
        }
    }

    fn restart(&mut self) {
        self.set_potential();
        self.wave = self.initial_wave();
        self.mom = self.wave.momentum();
        self.t = 0.0;
        self.t_anchor = 0.0;
        self.present = true;
        self.p_series.clear();
        self.p_series.push(0.0, 1.0);
        self.events.clear();
        self.survival = None;
        self.flash.clear();
        self.top_x = 1.2 * common::peak_density(&self.wave.psi);
        self.top_p = 1.2 * common::peak_density(&self.mom.phi);
    }

    /// Advances by exactly dt.
    pub fn advance(&mut self, dt: f64) {
        if self.present {
            self.prop.advance(&mut self.wave, dt);
        }
        self.t += dt;
        self.mom = self.wave.momentum();
        if self.p_series.last().is_none_or(|p| self.t - p[0] >= 0.04) {
            self.p_series.push(self.t, self.norm());
        }
    }

    fn tau(&self) -> f64 {
        if self.gamma > 0.0 { 1.0 / self.gamma } else { f64::INFINITY }
    }

    fn norm(&self) -> f64 {
        self.wave.norm_sqr()
    }

    /// e^(−Γ(t − t₀)) since the last preparation.
    fn predicted(&self, t: f64) -> f64 {
        if t < self.t_anchor || !self.present {
            return f64::NAN;
        }
        (-self.gamma * (t - self.t_anchor)).exp()
    }

    /// "Is the particle still there?": yes with probability P(t).
    fn measure_present(&mut self, ctx: &mut Ctx) {
        if !self.present {
            ctx.set_status("The particle has already decayed: R prepares a new one.");
            return;
        }
        let p = self.norm().min(1.0);
        let yes = ctx.rng.bool(p);
        if yes {
            self.wave.normalize();
            self.t_anchor = self.t;
            self.events.push((self.t, "yes".into()));
            self.flash.trigger(FlashKind::Other, "still there: yes");
            ctx.set_status(format!(
                "t = {:.2}: still there (probability was P = {p:.3}). ψ is renormalised: the decay starts afresh, P(t) = e^(−Γ(t − {:.2})). No memory.",
                self.t, self.t
            ));
        } else {
            self.wave.scale(c64(0.0, 0.0));
            self.present = false;
            self.events.push((self.t, "decayed".into()));
            self.flash.trigger(FlashKind::Other, "decayed");
            ctx.set_status(format!(
                "t = {:.2}: the particle has decayed (probability 1 − P = {:.3}). R prepares a new one.",
                self.t,
                1.0 - p
            ));
        }
        self.p_series.push(self.t, self.norm());
        self.mom = self.wave.momentum();
    }

    /// N copies of the particle as prepared at the last preparation: each decays at a random
    /// time drawn from the Born rule −dP/dt = Γ e^(−Γt).
    fn repeat(&mut self, n: usize, ctx: &mut Ctx) {
        if self.gamma <= 0.0 {
            ctx.set_status("Γ = 0: the particle is stable, every copy is still there.");
            return;
        }
        let lifetimes: Vec<f64> = (0..n).map(|_| ctx.rng.exponential(self.gamma)).collect();
        let tmax = 4.0 * self.tau();
        let mut hist = ShotHistogram::binned(0.0, tmax, 24);
        hist.add_all(&lifetimes);
        let edges = hist.edges();
        let predicted = edges
            .windows(2)
            .map(|w| (-self.gamma * w[0]).exp() - (-self.gamma * w[1]).exp())
            .collect();
        let mean = lifetimes.iter().sum::<f64>() / n as f64;
        ctx.set_status(format!(
            "{n} copies prepared at t = {:.2}: mean lifetime {mean:.2} (τ = 1/Γ = {:.2}); survivors after τ: {:.1} % (e⁻¹ = 36.8 %)",
            self.t_anchor,
            self.tau(),
            100.0 * lifetimes.iter().filter(|&&l| l > self.tau()).count() as f64 / n as f64
        ));
        self.survival = Some(Survival {
            t0: self.t_anchor,
            gamma: self.gamma,
            lifetimes,
            hist,
            predicted,
        });
    }

    fn coeffs(&self) -> Option<Vec<C64>> {
        self.basis.as_ref().map(|b| common::clean(common::project(&b.states, &self.wave)))
    }

    fn plot_x(&mut self, ui: &mut egui::Ui, height: f32) {
        let look = Look::of(ui);
        grow_top(&mut self.top_x, common::peak_density(&self.wave.psi));
        if self.well == Well::Free {
            let (lo, hi) = self.style.y_range(self.top_x);
            let resp = WavePlot::new("decay_free_x")
                .domain(Domain::Position)
                .style(self.style)
                .x_range(-40.0, 40.0)
                .y_range(lo, hi)
                .height(height)
                .curve(WaveCurve::new(&self.xs, &self.wave.psi))
                .show(ui);
            self.flash.paint(ui, &resp.transform, Domain::Position);
            return;
        }
        let basis = self.basis.clone().expect("well has a basis");
        let coeffs = self.coeffs().unwrap_or_default();
        let (e_lo, e_hi) = match self.well {
            Well::Harmonic => (0.0, 8.0),
            _ => (0.0, 5.0),
        };
        let e_mean = if self.present {
            common::mean_energy(&coeffs, &basis.energies)
        } else {
            e_lo
        };
        let scale = common::energy_curve_scale(self.style, self.top_x, 0.35 * (e_hi - e_lo));
        let mut curves = vec![];
        if let Init::Level(n) = self.init {
            let spacing = basis.energies.get(n + 1).map_or(1.0, |e| e - basis.energies[n]);
            let amp = basis.states[n].iter().fold(0.0f64, |m, v| m.max(v.abs()));
            curves.push(LevelCurve {
                psi: &basis.states[n],
                energy: basis.energies[n],
                scale: 0.4 * spacing / amp.max(1e-9),
                label: format!("ψ{} (t = 0)", common::subscript(n + usize::from(self.well == Well::Box))),
                color: look.accent().gamma_multiply(0.7),
                dashed: true,
            });
        }
        let x_half = if self.well == Well::Harmonic { 5.0 } else { 8.0 };
        let (resp, ladder) = EnergyView {
            id: "decay",
            xs: &self.xs,
            v: &self.v0,
            psi: &self.wave.psi,
            style: self.style,
            baseline: e_mean,
            scale,
            x_range: (-x_half, x_half),
            e_range: (e_lo, e_hi),
            height,
            energies: &basis.energies,
            coeffs: Some(&coeffs),
            labels: Some(&self.level_labels),
            selected: match self.init {
                Init::Level(n) => Some(n),
                Init::Packet => None,
            },
            ladder_width: look.px(LADDER_W),
            curves,
            markers: vec![],
            drawable: false,
        }
        .show(ui);
        self.flash.paint(ui, &resp.transform, Domain::Position);
        if let Some(n) = ladder.and_then(|l| l.clicked) {
            self.init = Init::Level(n);
            self.restart();
        }
    }

    fn plot_p(&mut self, ui: &mut egui::Ui, height: f32) {
        grow_top(&mut self.top_p, common::peak_density(&self.mom.phi));
        let (lo, hi) = self.style.y_range(self.top_p);
        let mut plot = WavePlot::new("decay_p")
            .domain(Domain::Momentum)
            .style(self.style)
            .x_range(-6.0, 6.0)
            .y_range(lo, hi)
            .height(height)
            .curve(WaveCurve::new(&self.mom.p, &self.mom.phi));
        // under the x plot with its ladder: same frame width and left edge
        if self.view == ReprView::Both && self.basis.is_some() {
            plot = plot.width(common::width_beside_ladder(ui, Look::of(ui).px(LADDER_W)));
        }
        let resp = plot.show(ui);
        self.flash.paint(ui, &resp.transform, Domain::Momentum);
    }

    /// P(t) against e^(−Γ(t − t₀)), the lifetime τ, and the survival fractions of the last repeat.
    fn plot_survival(&self, ui: &mut egui::Ui, width: f32, height: f32) {
        let look = Look::of(ui);
        let tau = self.tau();
        let t_end = if tau.is_finite() {
            (self.t_anchor + 3.0 * tau).max(self.t)
        } else {
            self.t.max(10.0)
        };
        let t_end = (t_end * 1.05).max(1.0);
        let resp = PlotFrame::new("decay_pt")
            .x_range(0.0, t_end)
            .y_range(0.0, 1.12)
            .labels("t", "P(t)")
            .width(width)
            .height(height)
            .pan_zoom(PanZoom::None)
            .show(ui, |painter, tr| {
                let r = tr.rect;
                // 1/e and τ
                plot_dashed(
                    painter,
                    tr,
                    &[0.0, t_end],
                    &[(-1f64).exp(); 2],
                    look.stroke(1.0, look.weak_text()),
                    5.0,
                );
                painter.text(
                    pos2(r.right() - look.px(4.0), tr.y_to_screen((-1f64).exp()) - 2.0),
                    Align2::RIGHT_BOTTOM,
                    "1/e",
                    look.font(11.0),
                    look.weak_text(),
                );
                if tau.is_finite() && self.present {
                    let ta = self.t_anchor + tau;
                    plot_dashed(painter, tr, &[ta, ta], &[0.0, 1.0], look.stroke(1.0, look.weak_text()), 5.0);
                    painter.text(
                        tr.to_screen(ta, 0.02) + egui::vec2(3.0, 0.0),
                        Align2::LEFT_BOTTOM,
                        format!("t₀ + τ, τ = 1/Γ = {tau:.2}"),
                        look.font(11.0),
                        look.weak_text(),
                    );
                }
                // events
                for (t, l) in &self.events {
                    let sx = tr.x_to_screen(*t);
                    painter.line_segment(
                        [pos2(sx, r.top()), pos2(sx, r.bottom())],
                        look.stroke(1.0, look.measure().gamma_multiply(0.6)),
                    );
                    painter.text(pos2(sx + 2.0, r.top() + 2.0), Align2::LEFT_TOP, l, look.font(10.5), look.measure());
                }
                // prediction e^(−Γ(t − t₀)) from the last preparation
                let n = 200;
                let ts: Vec<f64> = (0..=n)
                    .map(|k| self.t_anchor + (t_end - self.t_anchor) * k as f64 / n as f64)
                    .collect();
                let ys: Vec<f64> = ts.iter().map(|&t| self.predicted(t)).collect();
                plot_dashed(painter, tr, &ts, &ys, look.stroke(1.5, look.prediction()), 6.0);
                // the simulated P(t)
                let xs: Vec<f64> = self.p_series.points.iter().map(|p| p[0]).collect();
                let ys: Vec<f64> = self.p_series.points.iter().map(|p| p[1]).collect();
                plot_line(painter, tr, &xs, &ys, look.stroke(2.0, look.series(0)));
                if let Some(p) = self.p_series.last() {
                    painter.circle_filled(tr.to_screen(p[0], p[1]), look.px(3.0), look.series(0));
                }
                // survival fractions of the N copies, with ±1σ (binomial)
                if let Some(s) = &self.survival {
                    let tau_s = 1.0 / s.gamma;
                    let n_copies = s.lifetimes.len() as f64;
                    for k in 0..=8 {
                        let dt = 0.5 * k as f64 * tau_s;
                        let f = s.fraction(dt);
                        let pr = (-s.gamma * dt).exp();
                        let sd = (pr * (1.0 - pr) / n_copies).sqrt();
                        let t = s.t0 + dt;
                        let c = look.counts();
                        painter.line_segment([tr.to_screen(t, f - sd), tr.to_screen(t, f + sd)], look.stroke(1.5, c));
                        painter.circle_filled(tr.to_screen(t, f), look.px(3.5), c);
                    }
                }
                // legend
                let mut y = r.top() + look.px(6.0);
                let x = r.right() - look.px(8.0);
                let mut items = vec![
                    ("simulated ∫|ψ|² dx".to_string(), look.series(0)),
                    ("e^(−Γ(t − t₀)) (dashed)".to_string(), look.prediction()),
                ];
                if let Some(s) = &self.survival {
                    items.push((format!("survivors of N = {} copies", s.lifetimes.len()), look.counts()));
                }
                for (text, col) in items {
                    let g = painter.layout_no_wrap(text, look.font(11.5), col);
                    let rect = Align2::RIGHT_TOP.anchor_size(pos2(x, y), g.size());
                    painter.rect_filled(rect.expand(2.0), 2.0, look.plot_bg().gamma_multiply(0.85));
                    painter.galley(rect.min, g, col);
                    y += look.px(16.0);
                }
            });
        let _ = resp;
    }
}

impl Default for DecayScene {
    fn default() -> Self {
        Self::new()
    }
}

impl Scene for DecayScene {
    fn time(&self) -> f64 {
        self.t
    }

    fn time_scale(&self) -> f64 {
        2.0
    }

    fn step(&mut self, dt: f64, _ctx: &mut Ctx) {
        self.advance(dt);
    }

    fn reset(&mut self, _ctx: &mut Ctx) {
        self.restart();
    }

    fn measure_label(&self) -> Option<String> {
        Some("Still there?".into())
    }

    fn measure(&mut self, ctx: &mut Ctx) {
        self.measure_present(ctx);
    }

    fn on_key(&mut self, key: egui::Key, _m: egui::Modifiers, _ctx: &mut Ctx) -> bool {
        if key == egui::Key::V {
            self.style = self.style.next();
            return true;
        }
        false
    }

    fn action(&mut self, action: &str, ctx: &mut Ctx) -> bool {
        let mut words = action.split_whitespace();
        let cmd = words.next().unwrap_or("");
        let arg = words.next();
        let n = arg.and_then(|w| w.parse().ok()).unwrap_or(self.repeat_n);
        match cmd {
            "measure" | "still_there" => self.measure_present(ctx),
            "repeat" => self.repeat(n, ctx),
            "gamma" => {
                let Some(v) = common::num_arg(arg) else { return false };
                self.gamma = v.clamp(0.0, 1.0);
                self.restart();
            }
            "well" => {
                let Some(w) = arg.and_then(Well::parse) else { return false };
                self.well = w;
                self.init = Init::Packet;
                self.build_well();
                self.restart();
            }
            "level" => {
                let Some(v) = common::num_arg(arg) else { return false };
                if self.basis.is_some() {
                    self.init = Init::Level(v as usize);
                    self.restart();
                }
            }
            _ => return common::view_style_action(cmd, &mut self.view, &mut self.style),
        }
        true
    }

    fn central_ui(&mut self, ui: &mut egui::Ui, _ctx: &mut Ctx) {
        let look = Look::of(ui);
        view_bar(ui, &mut self.view, &mut self.style);
        let p = self.norm();
        let pred = self.predicted(self.t);
        let tau = self.tau();
        readout_row(
            ui,
            &[
                ("Γ", format!("{:.3}", self.gamma), None),
                ("τ = 1/Γ", if tau.is_finite() { format!("{tau:.2}") } else { "∞".into() }, None),
                ("P(t) = ∫|ψ|²", format!("{p:.4}"), Some(look.series(0))),
                (
                    "e^(−Γ(t − t₀))",
                    if pred.is_finite() { format!("{pred:.4}") } else { "—".into() },
                    Some(look.prediction()),
                ),
                ("t₀", fmt_num(self.t_anchor, 2), None),
                (
                    "half-life τ ln 2",
                    if tau.is_finite() {
                        format!("{:.2}", tau * std::f64::consts::LN_2)
                    } else {
                        "∞".into()
                    },
                    None,
                ),
            ],
        );
        let bottom_h = look.px(200.0);
        let plots_h = (ui.available_height() - bottom_h - look.px(10.0)).max(180.0);
        let view = self.view;
        view.show(ui, plots_h, |ui, d, h| match d {
            Domain::Position => self.plot_x(ui, h),
            Domain::Momentum => self.plot_p(ui, h),
        });
        ui.add_space(look.px(6.0));
        let w = ui.available_width();
        let hist_w = if self.survival.is_some() { (w * 0.36).max(240.0) } else { 0.0 };
        ui.horizontal(|ui| {
            let gap = if hist_w > 0.0 { look.px(8.0) } else { 0.0 };
            self.plot_survival(ui, w - hist_w - gap, bottom_h);
            if let Some(s) = &self.survival {
                HistogramPlot::new("decay_hist", &s.hist)
                    .predicted(&s.predicted)
                    .scale(HistScale::Density)
                    .x_label("decay time t − t₀ (repeat N)")
                    .width(hist_w)
                    .height(bottom_h)
                    .show(ui);
            }
        });
    }

    fn side_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx) {
        let look = ctx.look;
        ui.label(RichText::new("Decay rate").strong());
        if ui
            .add(egui::Slider::new(&mut self.gamma, 0.0..=1.0).step_by(0.01).text("Γ"))
            .changed()
        {
            self.restart();
        }
        hint(
            ui,
            format!(
                "V = V₀(x) − iΓ/2, so P(t) = e^(−Γt) and τ = 1/Γ = {}",
                if self.gamma > 0.0 {
                    format!("{:.2}", self.tau())
                } else {
                    "∞".into()
                }
            ),
        );

        section(ui, "Real potential V₀");
        let mut changed = false;
        segmented(ui, |ui| {
            for w in Well::ALL {
                if ui.selectable_label(self.well == w, w.label()).clicked() && self.well != w {
                    self.well = w;
                    changed = true;
                }
            }
        });
        if changed {
            self.init = Init::Packet;
            self.build_well();
            self.restart();
        }
        if self.basis.is_some() {
            ui.horizontal(|ui| {
                if ui.selectable_label(self.init == Init::Packet, "start: wave packet").clicked() {
                    self.init = Init::Packet;
                    self.restart();
                }
                if ui.selectable_label(matches!(self.init, Init::Level(_)), "eigenstate").clicked() {
                    self.init = Init::Level(0);
                    self.restart();
                }
            });
            hint(
                ui,
                "Click a level in the ladder to start in that eigenstate: |ψ|² keeps its shape and only fades.",
            );
        }

        section(ui, "Measure");
        if ui.button(RichText::new("Still there?  (M)").color(look.measure())).clicked() {
            self.measure_present(ctx);
        }
        if let Some(n) = repeat_button(ui, "decay_rep", "Follow N copies", &mut self.repeat_n) {
            self.repeat(n, ctx);
        }
        hint(
            ui,
            "Single shot: \"yes\" with probability P(t); then ψ is renormalised and the decay starts afresh. Repeat N: N copies prepared at t₀, each decays at a random time; dots = fraction still there at t₀ + kτ/2, histogram = decay times against Γe^(−Γt).",
        );
        if let Some(l) = self.flash.label() {
            ui.label(RichText::new(format!("last answer: {l}")).color(look.measure()));
        }
        if !self.present {
            ui.label(RichText::new("decayed: R prepares a new particle").color(look.measure()));
        }
        section(ui, "View");
        hint(ui, "x / p and the display mode: switches above the plot (V cycles the mode).");
    }

    fn notes(&self) -> String {
        NOTES.into()
    }
}

const NOTES: &str = "\
# The exercise
An unstable particle is modelled by giving the potential an imaginary part, V = V₀ − iħΓ/2 with Γ > 0 (here ħ = m = 1).
- (a) The continuity equation gets an extra term: ∂|Ψ|²/∂t = −∂j/∂x − Γ|Ψ|², so dP/dt = −ΓP for P(t) = ∫|Ψ|² dx.
- (b) P(t) = e^(−Γt): the lifetime is τ = 1/Γ.
- Equivalently: eigenstates ψₙ of H₀ are eigenstates of H with complex energy Eₙ − iΓ/2, so ψₙ(x, t) = ψₙ(x) e^(−iEₙt) e^(−Γt/2) and every |cₙ|² decays as e^(−Γt).

# Try
- Play: the simulated ∫|ψ|² lies on the dashed e^(−Γt) and reaches 1/e at t = τ, whatever V₀ and whatever the packet does.
- Harmonic well or box: all bars of the ladder shrink at the same rate; the packet keeps sloshing while it fades.
- Click a level: an eigenstate keeps its shape and only fades (and its phase turns).
- Still there? (M): yes with probability P(t). After a yes, ψ is renormalised and the decay starts again from 1: the particle does not age (exponential decay has no memory).
- Follow N copies: the fraction still there at t₀ + kτ/2 (dots, ±1σ) follows e^(−Γt); the decay times follow Γe^(−Γt).

# Behind the scenes
Split-step propagation with the complex potential V₀ − iΓ/2. The box has finite, slightly rounded walls (V₀ = 80 beyond |x| = 6) so that the propagator and the level ladder (finite-difference eigenstates) use the same potential. The free packet runs on a wide grid (|x| < 200) with absorbing edges far away; they would also remove probability, but the packet does not reach them in the times shown.
This is a crude model: H is not Hermitian, and the decay products are not described at all; the missing probability simply disappears. A Γ that depends on x would not give a single exponential.";

#[cfg(test)]
mod tests {
    use super::*;

    /// P(t) = e^(−Γt) for every V₀ (PS09 36 b), and the box/oscillator propagation does not heat up.
    #[test]
    fn norm_decays_exponentially_in_every_well() {
        for well in Well::ALL {
            let mut s = DecayScene::with_well(well);
            s.gamma = 0.3;
            s.restart();
            let e0 = s.wave.energy(&s.v0);
            for _ in 0..10 {
                s.advance(0.5);
                let want = survival(0.3, s.t);
                assert!(
                    (s.norm() - want).abs() < 1e-6 * want.max(1e-3) + 1e-9,
                    "{well:?}: {} vs {want}",
                    s.norm()
                );
            }
            let e1 = s.wave.energy(&s.v0);
            assert!((e1 - e0).abs() < 0.01 * e0.abs().max(0.1), "{well:?}: ⟨H₀⟩ {e0} → {e1}");
        }
    }

    use crate::physics::survival;
}
