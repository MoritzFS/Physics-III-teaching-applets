//! PS10 40 · The harmonic oscillator and its coherent states (ħ = m = ω = 1).
//!
//! The state is kept as coefficients cₙ in the oscillator basis and evolved
//! exactly, cₙ(t) = cₙ e^(−i(n+½)t). Number states, superpositions (sloshing
//! ψ₀ + ψ₁, breathing ψ₀ + ψ₂), coherent states |α⟩ and squeezed Gaussians.
//! The ladder shows |cₙ|² and the phasors; clicking a level picks a number
//! state, shift-click adds it to the mix, dragging a bar sets |cₙ|². A
//! phase-space panel shows (⟨x⟩, ⟨p⟩) with the uncertainty ellipse, the
//! classical orbit and (coherent preset) a draggable α. Measurements: E
//! (collapse to a number state) and x.

use std::f64::consts::TAU;
use std::sync::Arc;

use qb_core::egui::{self, Align2, RichText, Sense, pos2, vec2};
use qb_core::num::{C64, EigenBasis, EigenExpansion, Grid1D, Moments, Wave1D, c64, measure_energy, measure_position};
use qb_core::widgets::{
    Domain, FlashKind, MeasureFlash, PanZoom, PlotFrame, ReprView, TimeSeries, TimeSeriesPlot, TimeWindow, WaveCurve, WavePlot, WaveStyle,
    fmt_num, plot_dashed, plot_line, readout_row, repeat_button, segmented, view_bar,
};
use qb_core::{Ctx, Look, Scene};

use crate::common::{self, EnergyView, LevelCurve, Repeated, grow_top, hint, section};
use crate::physics::{number_state_product, poisson};

const LEVELS: usize = 60;
/// width of the level ladder beside the x plot (nominal points)
const LADDER_W: f32 = 230.0;
const SAMPLE_DT: f64 = 0.05;

/// The presets of this scene.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OscPreset {
    /// (ψ₀ + ψ₁)/√2: ⟨x⟩ oscillates at ω
    Sloshing,
    /// (ψ₀ + ψ₂)/√2: the width oscillates at 2ω, ⟨x⟩ = 0
    Breathing,
    /// a single number state ψₙ: stationary, only the phase turns
    Number(usize),
    /// a coherent state |α⟩
    Coherent,
}

/// What the "coherent" preset prepares.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Prep {
    /// |α⟩
    Coherent,
    /// a Gaussian with the same ⟨x⟩, ⟨p⟩ but σₓ = this (σₓ = 1/√2 is |α⟩)
    Squeezed(f64),
}

pub struct OscillatorScene {
    preset: OscPreset,
    // parameters
    alpha: C64,
    prep: Prep,
    squeeze: f64,
    width_x: f64,
    repeat_n: usize,
    view: ReprView,
    style: WaveStyle,
    classical: bool,
    components: bool,

    // state
    grid: Grid1D,
    xs: Vec<f64>,
    v: Vec<f64>,
    basis: Arc<EigenBasis>,
    labels: Vec<String>,
    exp: EigenExpansion,
    wave: Wave1D,
    moments: Moments,
    t: f64,
    /// classical trajectory: (t₀, x₀, p₀) of the last preparation
    classical0: (f64, f64, f64),
    trail: Vec<[f64; 2]>,
    sigma_x: TimeSeries,
    sigma_p: TimeSeries,
    product: TimeSeries,
    mean_x: TimeSeries,
    events: Vec<(f64, String)>,
    repeated: Option<Repeated>,
    flash: MeasureFlash,
    top_x: f64,
    top_p: f64,
    selected: Option<usize>,
    /// level whose bar is being dragged
    dragging: Option<usize>,
    /// fraction of the state the basis captured after the last projection
    captured: f64,
    /// top of the energy axis
    e_top: f64,
}

impl OscillatorScene {
    pub fn new(preset: OscPreset) -> Self {
        // a long cell for a fine momentum grid (dp = 2π/96 ≈ 0.065): smooth |Φ(p)|²
        let grid = Grid1D::new(-48.0, 48.0, 2048);
        let basis = Arc::new(EigenBasis::harmonic(grid, LEVELS, 1.0, 0.0));
        let xs = grid.xs();
        let v = xs.iter().map(|x| 0.5 * x * x).collect();
        let exp = EigenExpansion::from_coeffs(basis.clone(), &[c64(1.0, 0.0)]);
        let wave = exp.wave();
        let mut s = OscillatorScene {
            preset,
            alpha: c64(2.0, 0.0),
            prep: Prep::Coherent,
            squeeze: 0.35,
            width_x: 0.25,
            repeat_n: 1000,
            view: ReprView::X,
            style: WaveStyle::Density,
            classical: true,
            components: preset != OscPreset::Coherent,
            grid,
            xs,
            v,
            basis,
            labels: (0..LEVELS).map(|n| n.to_string()).collect(),
            exp,
            wave,
            moments: Moments::default(),
            t: 0.0,
            classical0: (0.0, 0.0, 0.0),
            trail: vec![],
            sigma_x: TimeSeries::new("σₓ"),
            sigma_p: TimeSeries::new("σₚ"),
            product: TimeSeries::new("σₓσₚ"),
            mean_x: TimeSeries::new("⟨x⟩"),
            events: vec![],
            repeated: None,
            flash: MeasureFlash::new(),
            top_x: 0.0,
            top_p: 0.0,
            selected: None,
            dragging: None,
            captured: 1.0,
            e_top: 7.6,
        };
        s.restart();
        s
    }

    fn is_coherent(&self) -> bool {
        self.preset == OscPreset::Coherent
    }

    /// The preset's coefficients at t = 0.
    fn initial_coeffs(&self) -> Vec<C64> {
        let r = std::f64::consts::FRAC_1_SQRT_2;
        match self.preset {
            OscPreset::Sloshing => vec![c64(r, 0.0), c64(r, 0.0)],
            OscPreset::Breathing => vec![c64(r, 0.0), c64(0.0, 0.0), c64(r, 0.0)],
            OscPreset::Number(n) => {
                let mut c = vec![c64(0.0, 0.0); n + 1];
                c[n] = c64(1.0, 0.0);
                c
            }
            OscPreset::Coherent => match self.prep {
                Prep::Coherent => EigenExpansion::coherent(self.basis.clone(), self.alpha).c0,
                Prep::Squeezed(s) => {
                    let w = Wave1D::gaussian(self.grid, 2f64.sqrt() * self.alpha.re, s, 2f64.sqrt() * self.alpha.im);
                    let e = EigenExpansion::project(self.basis.clone(), &w);
                    e.c0
                }
            },
        }
    }

    /// Back to t = 0 with the preset's (or the user's) initial state.
    fn restart(&mut self) {
        let c = self.initial_coeffs();
        self.exp = EigenExpansion::from_coeffs(self.basis.clone(), &c);
        self.captured = self.exp.captured();
        self.exp.normalize();
        self.t = 0.0;
        self.events.clear();
        self.repeated = None;
        self.flash.clear();
        self.selected = match self.preset {
            OscPreset::Number(n) => Some(n),
            _ => None,
        };
        for s in [&mut self.sigma_x, &mut self.sigma_p, &mut self.product, &mut self.mean_x] {
            s.clear();
        }
        self.refresh();
        self.new_preparation();
        let peak = common::peak_density(&self.wave.psi);
        self.top_x = 1.2 * peak.max(0.3);
        self.top_p = 1.2 * common::peak_density(&self.wave.momentum().phi).max(0.3);
    }

    /// The current expansion changed by hand: same time, new classical start, fresh trail.
    fn new_preparation(&mut self) {
        self.fit_e_top();
        self.classical0 = (self.t, self.moments.x, self.moments.p);
        self.trail.clear();
        self.record();
    }

    /// ψ and the moments from the expansion.
    fn refresh(&mut self) {
        self.wave = self.exp.wave();
        self.moments = self.wave.moments();
    }

    fn record(&mut self) {
        let m = self.moments;
        self.sigma_x.push(self.t, m.sigma_x);
        self.sigma_p.push(self.t, m.sigma_p);
        self.product.push(self.t, m.uncertainty_product());
        self.mean_x.push(self.t, m.x);
        self.trail.push([m.x, m.p]);
        if self.trail.len() > 400 {
            self.trail.remove(0);
        }
    }

    /// Classical particle started at the last preparation with (⟨x⟩, ⟨p⟩).
    fn classical(&self) -> (f64, f64) {
        let (t0, x0, p0) = self.classical0;
        let s = self.t - t0;
        (x0 * s.cos() + p0 * s.sin(), -x0 * s.sin() + p0 * s.cos())
    }

    /// Sets c(t) at the current time (c0 = c(t) e^(+iEₙt)).
    fn set_coeffs_now(&mut self, c_now: &[C64]) {
        let t = self.t;
        let c0: Vec<C64> = c_now
            .iter()
            .zip(&self.basis.energies)
            .map(|(c, e)| c * C64::from_polar(1.0, e * t))
            .collect();
        self.exp = EigenExpansion::from_coeffs(self.basis.clone(), &c0);
        self.exp.set_time(t);
        self.exp.normalize();
        self.refresh();
        self.repeated = None;
    }

    /// |α⟩ (or the squeezed Gaussian at the same centre) prepared at the current time.
    fn set_alpha_now(&mut self, alpha: C64) {
        self.alpha = alpha;
        let c_now = match self.prep {
            Prep::Coherent => EigenExpansion::coherent(self.basis.clone(), alpha).c0,
            Prep::Squeezed(s) => {
                let w = Wave1D::gaussian(self.grid, 2f64.sqrt() * alpha.re, s, 2f64.sqrt() * alpha.im);
                EigenExpansion::project(self.basis.clone(), &w).c0
            }
        };
        self.set_coeffs_now(&c_now);
        self.new_preparation();
    }

    fn measure_e(&mut self, ctx: &mut Ctx) {
        let Some(n) = measure_energy(&mut self.exp, ctx.rng) else { return };
        self.refresh();
        let e = self.basis.energies[n];
        self.flash
            .trigger(FlashKind::Level { index: n, energy: e }, format!("E = (n + ½)ħω, n = {n}"));
        self.events.push((self.t, format!("E{}", common::subscript(n))));
        self.selected = Some(n);
        self.new_preparation();
        self.repeated = None;
        ctx.set_status(format!(
            "measured E = {e:.1} (n = {n}) at t = {:.2}: the state is now ψ{}, stationary; σₓσₚ = (2n + 1)/2 = {:.1}",
            self.t,
            common::subscript(n),
            number_state_product(n)
        ));
    }

    fn measure_x(&mut self, ctx: &mut Ctx) {
        let mut w = self.exp.wave();
        let x = measure_position(&mut w, self.width_x, ctx.rng);
        let t = self.t;
        let mut e = EigenExpansion::project(self.basis.clone(), &w);
        e.set_time(0.0);
        self.captured = e.captured();
        // c(t) of the collapsed state, then back to t = 0 phases
        let c_now = e.c0.clone();
        self.set_coeffs_now(&c_now);
        self.t = t;
        self.flash
            .trigger(FlashKind::Position { x, width: self.width_x }, format!("x = {x:.2}"));
        self.events.push((self.t, "x".into()));
        self.selected = None;
        self.new_preparation();
        grow_top(&mut self.top_x, common::peak_density(&self.wave.psi));
        ctx.set_status(format!(
            "measured x = {x:.3} at t = {:.2} (detector σ = {}): a narrow packet, which now oscillates and breathes (basis holds {:.4} of it)",
            self.t, self.width_x, self.captured
        ));
    }

    fn repeat_e(&mut self, n: usize, ctx: &mut Ctx) {
        let (probs, names) = common::level_outcomes(&self.exp.probabilities());
        let r = common::repeat_discrete(&probs, names, n, ctx.rng, "n (repeat N)");
        let mean = r.hist.mean();
        let var = r.hist.std().powi(2);
        let extra = if self.is_coherent() && self.prep == Prep::Coherent && self.events.is_empty() {
            format!("; Poisson: mean = variance = |α|² = {:.3}", self.alpha.norm_sqr())
        } else {
            String::new()
        };
        ctx.set_status(format!(
            "{n} fresh copies measured in E: mean n = {mean:.3}, variance {var:.3}{extra}"
        ));
        self.repeated = Some(r);
    }

    fn repeat_x(&mut self, n: usize, ctx: &mut Ctx) {
        let r = common::repeat_position(&self.wave, n, ctx.rng);
        ctx.set_status(format!(
            "{n} fresh copies measured in x: mean {:.3}, spread {:.3}",
            r.hist.mean(),
            r.hist.std()
        ));
        self.repeated = Some(r);
    }

    fn e_top(&self) -> f64 {
        self.e_top
    }

    /// Top of the energy axis for the current state: the preset's, or higher if ⟨E⟩ needs it.
    fn fit_e_top(&mut self) {
        let base = if self.is_coherent() {
            let a2 = self.alpha.norm_sqr();
            (a2 + 0.5 + 3.0 * a2.sqrt() + 3.5).max(8.0).ceil()
        } else {
            7.6
        };
        let p = self.exp.probabilities();
        let e = self.exp.mean_energy();
        let var: f64 = p.iter().zip(&self.basis.energies).map(|(q, en)| q * (en - e).powi(2)).sum();
        self.e_top = base.max((e + 2.5 * var.sqrt() + 1.5).ceil());
    }

    /// Number state index if the state is (numerically) one.
    fn number_state(&self) -> Option<usize> {
        let p = self.exp.probabilities();
        p.iter().position(|&q| q > 1.0 - 1e-9)
    }

    fn plot_x(&mut self, ui: &mut egui::Ui, height: f32) {
        let look = Look::of(ui);
        grow_top(&mut self.top_x, common::peak_density(&self.wave.psi));
        let e_top = self.e_top();
        let x_half = (2.0 * e_top).sqrt().max(5.0) + 0.5;
        let coeffs = self.exp.coeffs();
        let probs = self.exp.probabilities();
        let e_mean = self.exp.mean_energy();
        let scale = common::energy_curve_scale(self.style, self.top_x, 0.28 * e_top);
        // the eigenfunctions in the mix, on their levels
        let mut curves = vec![];
        // (only for a few components: a wide mix would bury the picture)
        let few = probs.iter().filter(|&&p| p > 0.03).count() <= 4;
        if self.components && few {
            let mut idx: Vec<usize> = (0..LEVELS).filter(|&n| probs[n] > 0.03).collect();
            idx.sort_by(|a, b| probs[*b].total_cmp(&probs[*a]));
            idx.truncate(6);
            for n in idx {
                let amp = self.basis.states[n].iter().fold(0.0f64, |m, v| m.max(v.abs()));
                curves.push(LevelCurve {
                    psi: &self.basis.states[n],
                    energy: self.basis.energies[n],
                    scale: 0.42 / amp.max(1e-9),
                    label: String::new(),
                    color: look.weak_text().gamma_multiply(0.75),
                    dashed: false,
                });
            }
        }
        let (resp, ladder) = EnergyView {
            id: "osc",
            xs: &self.xs,
            v: &self.v,
            psi: &self.wave.psi,
            style: self.style,
            baseline: e_mean,
            scale,
            x_range: (-x_half, x_half),
            e_range: (0.0, e_top),
            height,
            energies: &self.basis.energies,
            coeffs: Some(&coeffs),
            labels: Some(&self.labels),
            selected: self.selected,
            ladder_width: look.px(LADDER_W),
            curves,
            markers: vec![],
            drawable: false,
        }
        .show(ui);
        let tr = resp.transform;
        let painter = resp.painter(ui);
        if self.classical {
            // a classical particle with the same start, riding on V
            let (xc, _) = self.classical();
            let p = tr.to_screen(xc, 0.5 * xc * xc);
            painter.circle_filled(p, look.px(6.0), look.accent());
            painter.circle_stroke(p, look.px(6.0), look.stroke(1.0, look.plot_bg()));
            painter.text(
                p + vec2(look.px(8.0), -look.px(4.0)),
                Align2::LEFT_BOTTOM,
                "classical",
                look.font(11.0),
                look.accent(),
            );
        }
        // ⟨E⟩ label at the right end of the baseline
        painter.text(
            pos2(tr.rect.right() - look.px(4.0), tr.y_to_screen(e_mean) - 2.0),
            Align2::RIGHT_BOTTOM,
            format!("⟨E⟩ = {e_mean:.3}"),
            look.font(11.5),
            look.text(),
        );
        self.flash.paint(ui, &tr, Domain::Position);
        if let Some(l) = ladder {
            self.flash.paint(ui, &l.transform, Domain::Position);
            self.ladder_input(ui, &l);
        }
    }

    /// Click: number state; shift-click: add/remove a level; drag a bar: set |cₙ|².
    fn ladder_input(&mut self, ui: &mut egui::Ui, l: &qb_core::widgets::LadderResponse) {
        let look = Look::of(ui);
        let tr = l.transform;
        let resp = ui.interact(l.response.rect, egui::Id::new("osc_ladder_drag"), Sense::click_and_drag());
        let energies = self.basis.energies.clone();
        let level_at = |y: f32| -> Option<usize> {
            (0..LEVELS)
                .filter(|&n| energies[n] >= tr.y[0] && energies[n] <= tr.y[1])
                .map(|n| (n, (tr.y_to_screen(energies[n]) - y).abs()))
                .filter(|(_, d)| *d < look.px(9.0))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(n, _)| n)
        };
        // bar geometry of LevelLadder: lines take 42 % of the inner width, then the bars, then the phasors
        let r = tr.rect;
        let gaps: Vec<f32> = (0..LEVELS - 1)
            .filter(|&n| self.basis.energies[n + 1] <= tr.y[1])
            .map(|n| (tr.y_to_screen(self.basis.energies[n]) - tr.y_to_screen(self.basis.energies[n + 1])).abs())
            .collect();
        let min_gap = gaps.iter().cloned().fold(f32::INFINITY, f32::min);
        let radius = (min_gap * 0.42).clamp(3.0, look.px(13.0));
        let bar_x0 = r.left() + r.width() * 0.42 + look.px(8.0);
        let bar_w = (r.right() - bar_x0 - 2.0 * radius - look.px(10.0)).max(10.0);
        if let Some(p) = resp.hover_pos()
            && let Some(n) = level_at(p.y)
        {
            let c = self.exp.coeff(n);
            resp.clone().on_hover_text(format!(
                "n = {n}, E = {:.1}\n|c|² = {:.4}\nclick: ψ{n} alone · shift-click: add/remove · drag the bar: set |cₙ|²",
                self.basis.energies[n],
                c.norm_sqr()
            ));
        }
        if resp.drag_started()
            && let Some(p) = resp.interact_pointer_pos()
        {
            self.dragging = level_at(p.y);
        }
        if resp.dragged()
            && let (Some(n), Some(p)) = (self.dragging, resp.interact_pointer_pos())
        {
            let target = ((p.x - bar_x0) / bar_w).clamp(0.0, 1.0) as f64;
            let mut c = self.exp.coeffs();
            let others: f64 = c.iter().enumerate().filter(|(k, _)| *k != n).map(|(_, z)| z.norm_sqr()).sum();
            let phase_n = if c[n].norm() > 1e-12 { c[n] / c[n].norm() } else { c64(1.0, 0.0) };
            c[n] = phase_n * target.sqrt();
            if others > 1e-12 {
                let f = ((1.0 - target) / others).sqrt();
                for (k, z) in c.iter_mut().enumerate() {
                    if k != n {
                        *z *= f;
                    }
                }
            } else if target < 1.0 {
                let k = if n == 0 { 1 } else { n - 1 };
                c[k] = c64((1.0 - target).sqrt(), 0.0);
            }
            self.set_coeffs_now(&c);
            self.selected = Some(n);
        }
        if resp.drag_stopped() && self.dragging.is_some() {
            self.dragging = None;
            self.new_preparation();
        }
        if resp.clicked()
            && let Some(p) = resp.interact_pointer_pos()
            && let Some(n) = level_at(p.y)
        {
            let shift = ui.input(|i| i.modifiers.shift);
            let mut c = self.exp.coeffs();
            if shift {
                if c[n].norm_sqr() > 1e-6 {
                    c[n] = c64(0.0, 0.0);
                } else {
                    let avg = c.iter().map(|z| z.norm_sqr()).filter(|p| *p > 1e-6).sum::<f64>()
                        / c.iter().filter(|z| z.norm_sqr() > 1e-6).count().max(1) as f64;
                    c[n] = c64(avg.max(0.05).sqrt(), 0.0);
                }
                if c.iter().all(|z| z.norm_sqr() < 1e-12) {
                    c[n] = c64(1.0, 0.0);
                }
            } else {
                c = vec![c64(0.0, 0.0); LEVELS];
                c[n] = c64(1.0, 0.0);
            }
            self.set_coeffs_now(&c);
            self.selected = Some(n);
            self.new_preparation();
        }
    }

    fn plot_p(&mut self, ui: &mut egui::Ui, height: f32) {
        let mom = self.wave.momentum();
        grow_top(&mut self.top_p, common::peak_density(&mom.phi));
        let (lo, hi) = self.style.y_range(self.top_p);
        let w = (2.0 * self.e_top()).sqrt().max(5.0) + 0.5;
        let mut plot = WavePlot::new("osc_p")
            .domain(Domain::Momentum)
            .style(self.style)
            .x_range(-w, w)
            .y_range(lo, hi)
            .height(height)
            .curve(WaveCurve::new(&mom.p, &mom.phi));
        // under the x plot with its ladder: same frame width and left edge
        if self.view == ReprView::Both {
            plot = plot.width(common::width_beside_ladder(ui, Look::of(ui).px(LADDER_W)));
        }
        let resp = plot.show(ui);
        if !matches!(self.flash.kind(), Some(FlashKind::Level { .. })) {
            self.flash.paint(ui, &resp.transform, Domain::Momentum);
        }
    }

    /// (⟨x⟩, ⟨p⟩) with the 1σ ellipse, the trail, the classical orbit; α is draggable for |α⟩.
    fn phase_space(&mut self, ui: &mut egui::Ui, size: f32) {
        let look = Look::of(ui);
        let m = self.moments;
        let r_orbit = (m.x * m.x + m.p * m.p).sqrt();
        let half = (r_orbit + 2.0 * m.sigma_x.max(m.sigma_p) + 0.5).max(3.0).ceil();
        let half = if self.is_coherent() {
            half.max((2f64.sqrt() * 3.6).ceil())
        } else {
            half
        };
        let (xc, pc) = self.classical();
        let trail = self.trail.clone();
        let coherent = self.is_coherent();
        let resp = PlotFrame::new(("osc_phase", half as i64))
            .x_range(-half, half)
            .y_range(-half, half)
            .labels(
                if coherent { "⟨x⟩ = √2 Re α" } else { "⟨x⟩" },
                if coherent { "⟨p⟩ = √2 Im α" } else { "⟨p⟩" },
            )
            .width(size)
            .height(size)
            .pan_zoom(PanZoom::None)
            .left_drag_pans(false)
            .show(ui, |painter, tr| {
                // classical orbit through the start
                let (_, x0, p0) = self.classical0;
                let rc = (x0 * x0 + p0 * p0).sqrt();
                let n = 120;
                let (ox, oy): (Vec<f64>, Vec<f64>) = (0..=n)
                    .map(|k| {
                        let a = TAU * k as f64 / n as f64;
                        (rc * a.cos(), rc * a.sin())
                    })
                    .unzip();
                plot_dashed(painter, tr, &ox, &oy, look.stroke(1.0, look.weak_text()), 5.0);
                // uncertainty ellipse (1σ) from the covariance matrix
                let (sxx, spp, sxp) = (m.sigma_x * m.sigma_x, m.sigma_p * m.sigma_p, m.cov_xp);
                let tr_half = 0.5 * (sxx + spp);
                let det = (0.25 * (sxx - spp).powi(2) + sxp * sxp).sqrt();
                let (l1, l2) = ((tr_half + det).max(0.0).sqrt(), (tr_half - det).max(0.0).sqrt());
                let ang = 0.5 * (2.0 * sxp).atan2(sxx - spp);
                let (ex, ey): (Vec<f64>, Vec<f64>) = (0..=n)
                    .map(|k| {
                        let a = TAU * k as f64 / n as f64;
                        let (u, v) = (l1 * a.cos(), l2 * a.sin());
                        (m.x + u * ang.cos() - v * ang.sin(), m.p + u * ang.sin() + v * ang.cos())
                    })
                    .unzip();
                let pts: Vec<egui::Pos2> = ex.iter().zip(&ey).map(|(x, y)| tr.to_screen(*x, *y)).collect();
                painter.add(egui::Shape::convex_polygon(pts, look.re().gamma_multiply(0.25), egui::Stroke::NONE));
                plot_line(painter, tr, &ex, &ey, look.stroke(1.2, look.re()));
                // trail of ⟨x⟩, ⟨p⟩
                let tx: Vec<f64> = trail.iter().map(|p| p[0]).collect();
                let ty: Vec<f64> = trail.iter().map(|p| p[1]).collect();
                plot_line(painter, tr, &tx, &ty, look.stroke(1.4, look.series(0).gamma_multiply(0.8)));
                painter.circle_filled(tr.to_screen(m.x, m.p), look.px(4.0), look.series(0));
                if self.classical {
                    painter.circle_stroke(tr.to_screen(xc, pc), look.px(7.0), look.stroke(1.5, look.accent()));
                }
                let caption = if coherent {
                    let a = c64(m.x, m.p) / 2f64.sqrt();
                    format!(
                        "α(t) = {} {} {:.2}i",
                        fmt_num(a.re, 2),
                        if a.im < 0.0 { "−" } else { "+" },
                        a.im.abs()
                    )
                } else {
                    "phase space".to_string()
                };
                painter.text(
                    pos2(tr.rect.left() + look.px(6.0), tr.rect.bottom() - look.px(4.0)),
                    Align2::LEFT_BOTTOM,
                    caption,
                    look.font(11.0),
                    look.text(),
                );
            });
        if coherent
            && (resp.response.dragged_by(egui::PointerButton::Primary) || resp.response.clicked())
            && let Some(p) = resp.response.interact_pointer_pos()
        {
            let (x, p) = resp.transform.from_screen(p);
            let a = c64(x, p) / 2f64.sqrt();
            let a = if a.norm() > 3.6 { a / a.norm() * 3.6 } else { a };
            self.set_alpha_now(a);
        }
        if coherent {
            resp.response
                .on_hover_text("drag or click: prepare |α⟩ (or the squeezed state) centred here");
        }
    }

    fn uncertainty_text(&self) -> String {
        let prod = self.moments.uncertainty_product();
        match self.number_state() {
            Some(n) => format!("{prod:.3} = (2n+1)/2, n = {n}"),
            None if prod < 0.5 * 1.002 => format!("{prod:.3} = ħ/2"),
            None => format!("{prod:.3} ≥ ħ/2"),
        }
    }
}

impl Scene for OscillatorScene {
    fn time(&self) -> f64 {
        self.t
    }

    fn step(&mut self, dt: f64, _ctx: &mut Ctx) {
        self.advance(dt);
    }

    fn reset(&mut self, _ctx: &mut Ctx) {
        self.restart();
    }

    fn measure_label(&self) -> Option<String> {
        Some("Measure E".into())
    }

    fn measure(&mut self, ctx: &mut Ctx) {
        self.measure_e(ctx);
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
            "measure_e" => self.measure_e(ctx),
            "measure_x" => self.measure_x(ctx),
            "repeat_e" => self.repeat_e(n, ctx),
            "repeat_x" => self.repeat_x(n, ctx),
            "alpha" => {
                let (Some(re), Some(im)) = (common::num_arg(arg), common::num_arg(words.next())) else {
                    return false;
                };
                self.alpha = c64(re, im);
                self.restart();
            }
            "squeezed" => {
                if let Some(v) = common::num_arg(arg) {
                    self.squeeze = v;
                }
                self.prep = Prep::Squeezed(self.squeeze);
                self.restart();
            }
            "coherent" => {
                self.prep = Prep::Coherent;
                self.restart();
            }
            "level" => {
                let Some(v) = common::num_arg(arg) else { return false };
                let mut c = vec![c64(0.0, 0.0); LEVELS];
                c[(v as usize).min(LEVELS - 1)] = c64(1.0, 0.0);
                self.set_coeffs_now(&c);
                self.new_preparation();
            }
            "classical_off" => self.classical = false,
            _ => return common::view_style_action(cmd, &mut self.view, &mut self.style),
        }
        true
    }

    fn central_ui(&mut self, ui: &mut egui::Ui, _ctx: &mut Ctx) {
        let look = Look::of(ui);
        view_bar(ui, &mut self.view, &mut self.style);
        let m = self.moments;
        let mut items = vec![
            ("⟨E⟩", format!("{:.3}", self.exp.mean_energy()), None),
            ("⟨x⟩", fmt_num(m.x, 3), Some(look.series(3))),
            ("⟨p⟩", fmt_num(m.p, 3), None),
            ("σₓ", fmt_num(m.sigma_x, 3), Some(look.series(0))),
            ("σₚ", fmt_num(m.sigma_p, 3), Some(look.series(1))),
            ("σₓσₚ", self.uncertainty_text(), Some(look.series(2))),
        ];
        if self.is_coherent() {
            let a = self.exp.coeffs();
            let mean_n: f64 = a.iter().enumerate().map(|(n, c)| n as f64 * c.norm_sqr()).sum();
            items.push(("⟨n⟩", format!("{mean_n:.3}"), None));
            if self.prep == Prep::Coherent {
                items.push(("|α|²", format!("{:.3}", self.alpha.norm_sqr()), Some(look.prediction())));
            }
        }
        readout_row(ui, &items);
        let bottom_h = look.px(220.0);
        let plots_h = (ui.available_height() - bottom_h - look.px(10.0)).max(200.0);
        let view = self.view;
        view.show(ui, plots_h, |ui, d, h| match d {
            Domain::Position => self.plot_x(ui, h),
            Domain::Momentum => self.plot_p(ui, h),
        });
        ui.add_space(look.px(6.0));
        let w = ui.available_width();
        let gap = look.px(8.0);
        let hist_w = if self.repeated.is_some() { (0.3 * w).max(240.0) } else { 0.0 };
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = gap;
            self.phase_space(ui, bottom_h);
            let ts_w = ui.available_width() - hist_w - if hist_w > 0.0 { gap } else { 0.0 };
            let mut ts = TimeSeriesPlot::new("osc_ts")
                .series(&self.sigma_x)
                .series(&self.sigma_p)
                .series(&self.product)
                .series(&self.mean_x)
                .hline(0.5, "ħ/2")
                .window(if self.t < 4.0 * TAU {
                    TimeWindow::All
                } else {
                    TimeWindow::Last(4.0 * TAU)
                })
                .now(self.t)
                .labels("t", "")
                .width(ts_w)
                .height(bottom_h);
            for (t, l) in &self.events {
                ts = ts.event(*t, l.clone());
            }
            ts.show(ui);
            if let Some(r) = &self.repeated {
                // Poisson prediction for a coherent state is the Born rule |cₙ|² itself
                r.show(ui, "osc_hist", hist_w, bottom_h);
            }
        });
    }

    fn side_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx) {
        let look = ctx.look;
        if self.is_coherent() {
            ui.label(RichText::new("Coherent state |α⟩").strong());
            let (mut re, mut im) = (self.alpha.re, self.alpha.im);
            let mut changed = false;
            changed |= ui.add(egui::Slider::new(&mut re, -3.5..=3.5).text("Re α")).changed();
            changed |= ui.add(egui::Slider::new(&mut im, -3.5..=3.5).text("Im α")).changed();
            if changed {
                self.alpha = c64(re, im);
                self.restart();
            }
            hint(
                ui,
                format!(
                    "⟨x⟩ = √2 Re α = {:.2}, ⟨p⟩ = √2 Im α = {:.2} at t = 0; |cₙ|² Poisson with mean |α|² = {:.2}. Or drag α in the phase-space panel.",
                    2f64.sqrt() * re,
                    2f64.sqrt() * im,
                    self.alpha.norm_sqr()
                ),
            );
            section(ui, "Compare");
            let mut squeezed = matches!(self.prep, Prep::Squeezed(_));
            segmented(ui, |ui| {
                if ui.selectable_label(!squeezed, "coherent |α⟩").clicked() && squeezed {
                    squeezed = false;
                    self.prep = Prep::Coherent;
                    self.restart();
                }
                if ui.selectable_label(squeezed, "other-width Gaussian").clicked() && !squeezed {
                    squeezed = true;
                    self.prep = Prep::Squeezed(self.squeeze);
                    self.restart();
                }
            });
            if squeezed {
                if ui
                    .add(egui::Slider::new(&mut self.squeeze, 0.2..=2.0).logarithmic(true).text("σₓ(0)"))
                    .changed()
                {
                    self.prep = Prep::Squeezed(self.squeeze);
                    self.restart();
                }
                hint(
                    ui,
                    format!(
                        "same ⟨x⟩, ⟨p⟩ but σₓ(0) = {:.2} ≠ 1/√2: σₓ swings between {:.2} and 1/(2σₓ(0)) = {:.2} twice per period (breathing); |cₙ|² is no longer Poisson.",
                        self.squeeze,
                        self.squeeze,
                        0.5 / self.squeeze
                    ),
                );
            }
        } else {
            ui.label(RichText::new("State").strong());
            hint(
                ui,
                "Ladder: click a level for ψₙ alone, shift-click to add or remove it, drag a bar to set |cₙ|² (the others are rescaled).",
            );
            ui.horizontal_wrapped(|ui| {
                let r = std::f64::consts::FRAC_1_SQRT_2;
                let mut set = None;
                if ui.button("ψ₀ + ψ₁").clicked() {
                    set = Some(vec![c64(r, 0.0), c64(r, 0.0)]);
                }
                if ui.button("ψ₀ + ψ₂").clicked() {
                    set = Some(vec![c64(r, 0.0), c64(0.0, 0.0), c64(r, 0.0)]);
                }
                if ui.button("ψ₁ + ψ₂").clicked() {
                    set = Some(vec![c64(0.0, 0.0), c64(r, 0.0), c64(r, 0.0)]);
                }
                for n in [0usize, 1, 3] {
                    if ui.button(format!("ψ{}", common::subscript(n))).clicked() {
                        let mut c = vec![c64(0.0, 0.0); n + 1];
                        c[n] = c64(1.0, 0.0);
                        set = Some(c);
                    }
                }
                if let Some(c) = set {
                    let mut full = vec![c64(0.0, 0.0); LEVELS];
                    full[..c.len()].copy_from_slice(&c);
                    // these are t = 0 coefficients: start the clock again
                    self.exp = EigenExpansion::from_coeffs(self.basis.clone(), &full);
                    self.t = 0.0;
                    self.events.clear();
                    for s in [&mut self.sigma_x, &mut self.sigma_p, &mut self.product, &mut self.mean_x] {
                        s.clear();
                    }
                    self.refresh();
                    self.selected = self.number_state();
                    self.new_preparation();
                    self.repeated = None;
                }
            });
        }
        section(ui, "View");
        ui.checkbox(&mut self.components, "eigenfunctions ψₙ of the mix on their levels");
        ui.checkbox(&mut self.classical, "classical particle (same start ⟨x⟩, ⟨p⟩)");
        hint(ui, "x / p and the display mode: switches above the plot (V cycles the mode).");

        section(ui, "Measure");
        ui.add(
            egui::Slider::new(&mut self.width_x, 0.05..=1.5)
                .logarithmic(true)
                .text("x detector σ"),
        );
        ui.horizontal(|ui| {
            if ui.button(RichText::new("Measure E  (M)").color(look.measure())).clicked() {
                self.measure_e(ctx);
            }
            if ui.button(RichText::new("Measure x").color(look.measure())).clicked() {
                self.measure_x(ctx);
            }
        });
        if let Some(n) = repeat_button(ui, "osc_rep_e", "Repeat E", &mut self.repeat_n) {
            self.repeat_e(n, ctx);
        }
        if let Some(n) = repeat_button(ui, "osc_rep_x", "Repeat x", &mut self.repeat_n) {
            self.repeat_x(n, ctx);
        }
        hint(
            ui,
            "E: outcome (n + ½)ħω with probability |cₙ|²; the state collapses to ψₙ and stops moving. x: a narrow packet, which then oscillates and breathes.",
        );
        if let Some(l) = self.flash.label() {
            ui.label(RichText::new(format!("last outcome: {l}")).color(look.measure()));
        }
    }

    fn notes(&self) -> String {
        match self.preset {
            OscPreset::Sloshing => format!("{NOTES_HO_INTRO}{SLOSHING}{NOTES_HO}"),
            OscPreset::Breathing => format!("{NOTES_HO_INTRO}{BREATHING}{NOTES_HO}"),
            OscPreset::Number(_) => format!("{NOTES_HO_INTRO}{NUMBER}{NOTES_HO}"),
            OscPreset::Coherent => NOTES_COHERENT.into(),
        }
    }
}

impl OscillatorScene {
    /// Advances by exactly dt (exact evolution of the coefficients).
    pub fn advance(&mut self, dt: f64) {
        self.exp.advance(dt);
        self.t += dt;
        self.refresh();
        if self.sigma_x.last().is_none_or(|p| self.t - p[0] >= SAMPLE_DT) {
            self.record();
        }
    }
}

/// Born-rule weights of a coherent state (the Poisson law) for the tests and readouts.
pub fn coherent_weights(alpha: C64, count: usize) -> Vec<f64> {
    (0..count).map(|n| poisson(n, alpha.norm_sqr())).collect()
}

const NOTES_HO_INTRO: &str = "\
# The oscillator (PS10 40)
V = ½mω²x², units ħ = m = ω = 1, levels Eₙ = (n + ½)ħω. The exercise starts from a fact you can check here: of the stationary states ψₙ only ψ₀ reaches the uncertainty limit; in general σₓσₚ = (2n + 1)ħ/2.

";

const SLOSHING: &str = "\
# This preset: (ψ₀ + ψ₁)/√2
The two components turn at different rates, e^(−it/2) and e^(−i3t/2), so their interference moves: ⟨x⟩ = √2 Re(c₀*c₁ e^(−iωt)) = (1/√2) cos ωt sloshes back and forth at ω (period 2π) while the shape stays the same.

";

const BREATHING: &str = "\
# This preset: (ψ₀ + ψ₂)/√2
Both components are even, so ⟨x⟩ = ⟨p⟩ = 0 at all times, but the width oscillates at the difference frequency 2ω (period π): the packet breathes, σₓ and σₚ swing in antiphase.

";

const NUMBER: &str = "\
# This preset: a number state ψ₃
A single eigenstate is stationary: |ψ|² never changes, only the overall phase turns (watch the colour and the phasor in the ladder). σₓσₚ = (2·3 + 1)/2 = 3.5.

";

const NOTES_HO: &str = "\
# Try
- Click levels to compare σₓσₚ with (2n + 1)/2: 0.5, 1.5, 2.5, …
- Shift-click to mix levels; drag a bar to change |cₙ|². Only superpositions move: the motion is the beating of the phases e^(−iEₙt).
- Measure E (M): one level comes out with probability |cₙ|²; afterwards nothing moves. Repeat E: the histogram follows |cₙ|².
- Measure x with a narrow detector: the packet is narrow, contains many levels, and oscillates and breathes (period 2π for ⟨x⟩, π for the width).
- Phase space (bottom left): (⟨x⟩, ⟨p⟩) moves on the classical orbit (Ehrenfest); the ellipse is the 1σ uncertainty region.

# Behind the scenes
The state is the coefficient vector cₙ on the analytic oscillator states n = 0 … 59 (Hermite functions, textbook sign ψₙ(+∞) > 0), evolved exactly by cₙ(t) = cₙ e^(−iEₙt). After a position measurement the collapsed state is projected back onto these 60 states (the status line says how much they hold).";

const NOTES_COHERENT: &str = "\
# The exercise
Coherent states are eigenstates of the lowering operator, â|α⟩ = α|α⟩, with α any complex number (units ħ = m = ω = 1).
- (a) ⟨x⟩ = √(2ħ/mω) Re α, ⟨p⟩ = √(2ħmω) Im α, σₓ = √(ħ/2mω), σₚ = √(ħmω/2): σₓσₚ = ħ/2, the minimum.
- (b, c) cₙ = αⁿ/√(n!) c₀ with c₀ = e^(−|α|²/2), so |cₙ|² = e^(−|α|²)|α|^(2n)/n!: a Poisson distribution with mean ⟨n⟩ = |α|².
- (d) With ψₙ → e^(−iEₙt)ψₙ the state stays coherent with α(t) = α e^(−iωt) (and a global phase e^(−iωt/2)): ⟨x⟩ and ⟨p⟩ follow the classical motion, and σₓ, σₚ stay constant.

# Try
- Play: the packet oscillates rigidly without spreading; the classical particle (dot on V) moves with ⟨x⟩. In phase space α(t)√2 = (⟨x⟩, ⟨p⟩) runs clockwise on a circle and the uncertainty disk keeps its size.
- σₓ, σₚ and σₓσₚ = ½ stay flat in the time plot.
- Drag α in the phase-space panel or use the sliders: larger |α| means more quanta (⟨n⟩ = |α|²) and a larger amplitude, but the same width.
- Compare: an other-width Gaussian with the same ⟨x⟩, ⟨p⟩ breathes (σₓ oscillates at 2ω, the ellipse rotates and changes shape), and its |cₙ|² is not Poisson. Only σₓ = 1/√2 (the ground-state width) moves rigidly.
- Measure E (M): one n, then the state is ψₙ and stops moving. Repeat E with N = 1000: Poisson statistics, mean ≈ variance ≈ |α|².
- Measure x: the packet becomes narrower, which is no longer a coherent state: it breathes.

# Behind the scenes
cₙ on the analytic oscillator states n = 0 … 59, evolved exactly. The coherent state is built from the formula of (b, c); the squeezed comparison is a Gaussian of the chosen width projected on the same states.";

#[cfg(test)]
mod tests {
    use super::*;

    /// |α⟩ from the formula: Poisson weights, ⟨x⟩ = √2|α|cos(t − φ) without spreading, σₓσₚ = ½.
    #[test]
    fn coherent_state_moves_rigidly() {
        let mut s = OscillatorScene::new(OscPreset::Coherent);
        let alpha = s.alpha;
        let w = coherent_weights(alpha, LEVELS);
        for (p, q) in s.exp.probabilities().iter().zip(&w) {
            assert!((p - q).abs() < 1e-9);
        }
        for k in 0..12 {
            s.advance(0.5);
            let (x, p) = crate::physics::coherent_mean(alpha, 1.0, s.t);
            let m = s.moments;
            assert!((m.x - x).abs() < 1e-6 && (m.p - p).abs() < 1e-6, "k = {k}: ⟨x⟩ {} vs {x}", m.x);
            assert!((m.sigma_x - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-6);
            assert!((m.uncertainty_product() - 0.5).abs() < 1e-6);
            // the classical particle coincides with ⟨x⟩, ⟨p⟩
            let (xc, pc) = s.classical();
            assert!((xc - m.x).abs() < 1e-6 && (pc - m.p).abs() < 1e-6);
        }
    }

    /// σₓσₚ = (2n + 1)/2 for number states; ψ₀ + ψ₂ breathes at 2ω with ⟨x⟩ = 0.
    #[test]
    fn number_states_and_breathing() {
        for n in 0..6 {
            let s = OscillatorScene::new(OscPreset::Number(n));
            assert!((s.moments.uncertainty_product() - number_state_product(n)).abs() < 1e-6, "n = {n}");
        }
        let mut s = OscillatorScene::new(OscPreset::Breathing);
        let w0 = s.moments.sigma_x;
        s.advance(std::f64::consts::PI / 2.0);
        assert!(s.moments.x.abs() < 1e-9);
        assert!((s.moments.sigma_x - w0).abs() > 0.2, "width changes");
        s.advance(std::f64::consts::PI / 2.0);
        assert!((s.moments.sigma_x - w0).abs() < 1e-6, "period π");
        let mut s = OscillatorScene::new(OscPreset::Sloshing);
        s.advance(std::f64::consts::PI);
        assert!((s.moments.x + std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-6, "⟨x⟩(π) = −1/√2");
    }
}
