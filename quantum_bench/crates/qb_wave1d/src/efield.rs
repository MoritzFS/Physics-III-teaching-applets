//! PS12 49 · A charged oscillator in a uniform electric field, H = H₀ − λħωξ.
//!
//! Units: ξ = √(mω/ħ) x, energies in ħω (the code uses ħ = m = ω = 1, so ξ = x).
//! The new potential ½ξ² − λξ = ½(ξ − λ)² − λ²/2 is the old one shifted by λ
//! and lowered by λ²/2. The scene compares exact levels n + ½ − λ²/2 with
//! first-order (no shift) and second-order perturbation theory, the
//! first-order perturbed states with the exact displaced ones, shows
//! ⟨ξ⟩ₙ = λ, and the sudden switch-on: the old ground state is a coherent state
//! of the new well and oscillates about ξ = λ. Part a): a constant energy
//! offset only adds a global phase.

use std::f64::consts::TAU;
use std::sync::Arc;

use qb_core::egui::{self, Align2, RichText, pos2};
use qb_core::num::{C64, EigenBasis, EigenExpansion, Grid1D, Moments, Wave1D, c64, measure_position};
use qb_core::widgets::{
    Domain, FlashKind, MeasureFlash, PanZoom, PlotFrame, ReprView, TimeSeries, TimeSeriesPlot, TimeWindow, WaveCurve, WavePlot, WaveStyle,
    fmt_num, plot_dashed, plot_line, readout_row, repeat_button, segmented, view_bar,
};
use qb_core::{Ctx, Look, Scene};

use crate::common::{self, EnergyView, LevelCurve, Repeated, grow_top, hint, section};
use crate::physics::{first_order_energy, first_order_mean_xi, first_order_state, second_order_energy, shifted_energy};

const LEVELS: usize = 40;
/// width of the level ladder beside the x plot (nominal points)
const LADDER_W: f32 = 220.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Start {
    /// the old ground state: the field is switched on suddenly at t = 0
    Sudden,
    /// the selected exact eigenstate of the new H
    Eigen,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LadderBasis {
    New,
    Old,
}

pub struct EfieldScene {
    // parameters
    lambda: f64,
    offset_on: bool,
    offset: f64,
    start: Start,
    ladder_basis: LadderBasis,
    selected: usize,
    width_x: f64,
    repeat_n: usize,
    view: ReprView,
    style: WaveStyle,

    // derived
    grid: Grid1D,
    xs: Vec<f64>,
    v_old: Vec<f64>,
    v_new: Vec<f64>,
    old: Arc<EigenBasis>,
    new: Arc<EigenBasis>,
    labels: Vec<String>,

    // state
    exp: EigenExpansion,
    wave: Wave1D,
    moments: Moments,
    t: f64,
    /// t of the last preparation (for the ⟨ξ⟩(t) prediction after the switch)
    t_prep: f64,
    mean_xi: TimeSeries,
    sigma_x: TimeSeries,
    events: Vec<(f64, String)>,
    repeated: Option<Repeated>,
    flash: MeasureFlash,
    top_x: f64,
    top_p: f64,
}

impl EfieldScene {
    pub fn new() -> Self {
        // a long cell for a fine momentum grid (dp = 2π/80 ≈ 0.08): smooth |Φ(p)|²
        let grid = Grid1D::new(-40.0, 40.0, 2048);
        let old = Arc::new(EigenBasis::harmonic(grid, LEVELS, 1.0, 0.0));
        let xs = grid.xs();
        let exp = EigenExpansion::from_coeffs(old.clone(), &[c64(1.0, 0.0)]);
        let wave = exp.wave();
        let mut s = EfieldScene {
            lambda: 1.0,
            offset_on: false,
            offset: 3.0,
            start: Start::Sudden,
            ladder_basis: LadderBasis::New,
            selected: 0,
            width_x: 0.25,
            repeat_n: 1000,
            view: ReprView::X,
            style: WaveStyle::Density,
            grid,
            v_old: xs.iter().map(|x| 0.5 * x * x).collect(),
            v_new: vec![],
            xs,
            old: old.clone(),
            new: old,
            labels: (0..LEVELS).map(|n| n.to_string()).collect(),
            exp,
            wave,
            moments: Moments::default(),
            t: 0.0,
            t_prep: 0.0,
            mean_xi: TimeSeries::new("⟨ξ⟩"),
            sigma_x: TimeSeries::new("σ_ξ"),
            events: vec![],
            repeated: None,
            flash: MeasureFlash::new(),
            top_x: 0.0,
            top_p: 0.0,
        };
        s.build();
        s.restart();
        s
    }

    fn c_offset(&self) -> f64 {
        if self.offset_on { self.offset } else { 0.0 }
    }

    /// New potential, basis (exact displaced states, energies n + ½ − λ²/2 + C).
    fn build(&mut self) {
        let (l, c) = (self.lambda, self.c_offset());
        self.v_new = self.xs.iter().map(|x| 0.5 * x * x - l * x + c).collect();
        let mut b = EigenBasis::harmonic(self.grid, LEVELS, 1.0, l);
        b.energies = (0..LEVELS).map(|n| shifted_energy(n, l) + c).collect();
        self.new = Arc::new(b);
    }

    fn restart(&mut self) {
        let c: Vec<C64> = match self.start {
            // the old ground state written in the new basis
            Start::Sudden => common::project(&self.new.states, &self.old.wave(0)),
            Start::Eigen => {
                let mut c = vec![c64(0.0, 0.0); LEVELS];
                c[self.selected] = c64(1.0, 0.0);
                c
            }
        };
        self.exp = EigenExpansion::from_coeffs(self.new.clone(), &c);
        self.exp.normalize();
        self.t = 0.0;
        self.t_prep = 0.0;
        self.mean_xi.clear();
        self.sigma_x.clear();
        self.events.clear();
        self.repeated = None;
        self.flash.clear();
        self.refresh();
        self.record();
        self.top_x = 1.2 * common::peak_density(&self.wave.psi).max(0.3);
        self.top_p = 1.2 * common::peak_density(&self.wave.momentum().phi).max(0.3);
    }

    /// λ or the offset changed: new H, the state continues (the clock restarts for a new switch-on).
    fn rebuild(&mut self) {
        self.build();
        self.restart();
    }

    fn refresh(&mut self) {
        self.wave = self.exp.wave();
        self.moments = self.wave.moments();
    }

    fn record(&mut self) {
        self.mean_xi.push(self.t, self.moments.x);
        self.sigma_x.push(self.t, self.moments.sigma_x);
    }

    /// Advances by exactly dt.
    pub fn advance(&mut self, dt: f64) {
        self.exp.advance(dt);
        self.t += dt;
        self.refresh();
        if self.mean_xi.last().is_none_or(|p| self.t - p[0] >= 0.05) {
            self.record();
        }
    }

    /// Replaces the state by ψ (at the current time): new coefficients in the new basis.
    fn set_wave_now(&mut self, w: &Wave1D) {
        let c_now = common::project(&self.new.states, w);
        let t = self.t;
        let c0: Vec<C64> = c_now
            .iter()
            .zip(&self.new.energies)
            .map(|(c, e)| c * C64::from_polar(1.0, e * t))
            .collect();
        self.exp = EigenExpansion::from_coeffs(self.new.clone(), &c0);
        self.exp.set_time(t);
        self.exp.normalize();
        self.refresh();
        self.t_prep = t;
        self.repeated = None;
    }

    /// Coefficients and energies of the ladder's basis.
    fn ladder_data(&self) -> (Vec<C64>, Vec<f64>) {
        match self.ladder_basis {
            LadderBasis::New => (self.exp.coeffs(), self.new.energies.clone()),
            LadderBasis::Old => (common::project(&self.old.states, &self.wave), self.old.energies.clone()),
        }
    }

    /// Energy measurement in the ladder's basis (new H or old H₀).
    fn measure_e(&mut self, ctx: &mut Ctx) {
        let (c, energies) = self.ladder_data();
        let probs: Vec<f64> = c.iter().map(|z| z.norm_sqr()).collect();
        let Some(n) = ctx.rng.weighted(&probs) else { return };
        let phase = if c[n].norm() > 0.0 { c[n] / c[n].norm() } else { c64(1.0, 0.0) };
        let basis = if self.ladder_basis == LadderBasis::New {
            self.new.clone()
        } else {
            self.old.clone()
        };
        let mut w = basis.wave(n);
        w.scale(phase);
        self.set_wave_now(&w);
        let (which, e) = match self.ladder_basis {
            LadderBasis::New => ("H", energies[n]),
            LadderBasis::Old => ("H₀", energies[n]),
        };
        self.flash
            .trigger(FlashKind::Level { index: n, energy: e }, format!("{which}: n = {n}"));
        self.events.push((self.t, format!("{which} n={n}")));
        self.selected = n;
        let after = if self.ladder_basis == LadderBasis::New {
            "an eigenstate of the actual H: stationary from now on"
        } else {
            "an eigenstate of the old H₀, not of H: it oscillates about ξ = λ"
        };
        ctx.set_status(format!(
            "measured {which} = {e:.3} (n = {n}, probability {:.3}) at t = {:.2}: {after}",
            probs[n] / probs.iter().sum::<f64>(),
            self.t
        ));
        self.record();
    }

    fn measure_x(&mut self, ctx: &mut Ctx) {
        let mut w = self.exp.wave();
        let x = measure_position(&mut w, self.width_x, ctx.rng);
        self.set_wave_now(&w);
        self.flash
            .trigger(FlashKind::Position { x, width: self.width_x }, format!("ξ = {x:.2}"));
        self.events.push((self.t, "ξ".into()));
        ctx.set_status(format!("measured ξ = {x:.3} at t = {:.2} (detector σ = {})", self.t, self.width_x));
        self.record();
    }

    fn repeat_e(&mut self, n: usize, ctx: &mut Ctx) {
        let (c, _) = self.ladder_data();
        let probs: Vec<f64> = c.iter().map(|z| z.norm_sqr()).collect();
        let (probs, names) = common::level_outcomes(&probs);
        let which = if self.ladder_basis == LadderBasis::New {
            "n of H"
        } else {
            "n of H₀"
        };
        let r = common::repeat_discrete(&probs, names, n, ctx.rng, &format!("{which} (repeat N)"));
        ctx.set_status(format!(
            "{n} fresh copies, energy in the {} basis: mean n = {:.3} (after a sudden switch the new-basis weights are Poisson with mean λ²/2 = {:.3})",
            if self.ladder_basis == LadderBasis::New { "new" } else { "old" },
            r.hist.mean(),
            0.5 * self.lambda * self.lambda
        ));
        self.repeated = Some(r);
    }

    /// First-order perturbed state n (normalised) on the grid.
    fn first_order_wave(&self, n: usize) -> Vec<f64> {
        let c = first_order_state(n, self.lambda, LEVELS);
        let norm: f64 = c.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        let mut out = vec![0.0; self.grid.n];
        for (k, ck) in c.iter().enumerate() {
            if ck.norm() > 0.0 {
                for (o, s) in out.iter_mut().zip(&self.old.states[k]) {
                    *o += ck.re / norm * s;
                }
            }
        }
        out
    }

    /// |⟨ψₙ⁽¹⁾|ψₙ(ξ − λ)⟩|² (normalised first-order state against the exact one).
    fn overlap(&self, n: usize) -> f64 {
        let a = self.first_order_wave(n);
        let dx = self.grid.dx;
        let ov: f64 = a.iter().zip(&self.new.states[n]).map(|(x, y)| x * y).sum::<f64>() * dx;
        ov * ov
    }

    fn e_top(&self) -> f64 {
        7.0 + self.c_offset()
    }

    fn plot_x(&mut self, ui: &mut egui::Ui, height: f32) {
        let look = Look::of(ui);
        grow_top(&mut self.top_x, common::peak_density(&self.wave.psi));
        let e_lo = (-0.5 * self.lambda * self.lambda + self.c_offset() - 0.3).min(-0.3);
        let e_hi = self.e_top();
        let (coeffs, energies) = self.ladder_data();
        let e_mean = self.exp.mean_energy();
        let scale = common::energy_curve_scale(self.style, self.top_x, 0.25 * (e_hi - e_lo));
        let n = self.selected.min(LEVELS - 1);
        let first = self.first_order_wave(n);
        let amp = self.new.states[n].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let curve_scale = 0.42 / amp.max(1e-9);
        let curves = vec![
            LevelCurve {
                psi: &self.new.states[n],
                energy: self.new.energies[n],
                scale: curve_scale,
                label: format!("ψ{}(ξ − λ) exact", common::subscript(n)),
                color: look.accent(),
                dashed: false,
            },
            LevelCurve {
                psi: &first,
                energy: self.new.energies[n],
                scale: curve_scale,
                label: String::new(),
                color: look.prediction().gamma_multiply(0.9),
                dashed: true,
            },
        ];
        let (resp, ladder) = EnergyView {
            id: "efield",
            xs: &self.xs,
            v: &self.v_new,
            psi: &self.wave.psi,
            style: self.style,
            baseline: e_mean,
            scale,
            x_range: (-5.5, 5.5),
            e_range: (e_lo, e_hi),
            height,
            energies: &energies,
            coeffs: Some(&coeffs),
            labels: Some(&self.labels),
            selected: Some(n),
            ladder_width: look.px(LADDER_W),
            curves,
            markers: vec![],
            drawable: false,
        }
        .show(ui);
        let tr = resp.transform;
        let painter = resp.painter(ui);
        // the unperturbed potential and its levels, dashed
        plot_dashed(
            &painter,
            &tr,
            &self.xs,
            &self.v_old,
            look.stroke(1.2, look.potential().gamma_multiply(0.7)),
            6.0,
        );
        let other: &[f64] = if self.ladder_basis == LadderBasis::New {
            &self.old.energies
        } else {
            &self.new.energies
        };
        let other_v: &[f64] = if self.ladder_basis == LadderBasis::New {
            &self.v_old
        } else {
            &self.v_new
        };
        common::draw_levels(
            &painter,
            &tr,
            &self.xs,
            other_v,
            other,
            look.stroke(1.0, look.level().gamma_multiply(0.6)),
            true,
        );
        // ξ = λ and ξ = 0
        plot_dashed(
            &painter,
            &tr,
            &[self.lambda, self.lambda],
            &[e_lo, e_hi],
            look.stroke(1.0, look.accent().gamma_multiply(0.6)),
            4.0,
        );
        painter.text(
            tr.to_screen(self.lambda, e_hi) + egui::vec2(3.0, 3.0),
            Align2::LEFT_TOP,
            "ξ = λ",
            look.font(11.0),
            look.accent(),
        );
        // legend
        let items = [
            ("V = ½ξ² − λξ (solid)", look.potential()),
            ("V₀ = ½ξ², old levels (dashed)", look.potential().gamma_multiply(0.7)),
            ("exact ψₙ (solid), first order ψₙ⁽¹⁾ (dashed)", look.prediction()),
        ];
        let mut y = tr.rect.top() + look.px(40.0);
        for (text, col) in items {
            let g = painter.layout_no_wrap(text.to_string(), look.font(11.0), col);
            let r = Align2::RIGHT_TOP.anchor_size(pos2(tr.rect.right() - look.px(6.0), y), g.size());
            painter.rect_filled(r.expand(2.0), 2.0, look.plot_bg().gamma_multiply(0.85));
            painter.galley(r.min, g, col);
            y += look.px(15.0);
        }
        painter.text(
            pos2(tr.rect.left() + look.px(26.0), tr.y_to_screen(e_mean) - 2.0),
            Align2::LEFT_BOTTOM,
            format!("⟨H⟩ = {e_mean:.3}"),
            look.font(11.5),
            look.text(),
        );
        self.flash.paint(ui, &tr, Domain::Position);
        if let Some(l) = ladder {
            self.flash.paint(ui, &l.transform, Domain::Position);
            if let Some(k) = l.clicked {
                self.selected = k;
                if self.start == Start::Eigen {
                    self.restart();
                }
            }
        }
    }

    fn plot_p(&mut self, ui: &mut egui::Ui, height: f32) {
        let mom = self.wave.momentum();
        grow_top(&mut self.top_p, common::peak_density(&mom.phi));
        let (lo, hi) = self.style.y_range(self.top_p);
        let mut plot = WavePlot::new("efield_p")
            .domain(Domain::Momentum)
            .style(self.style)
            .x_range(-5.5, 5.5)
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

    /// Eₙ(λ): exact (= second order) against first order, for n = 0 … 4.
    fn plot_levels_vs_lambda(&self, ui: &mut egui::Ui, width: f32, height: f32) {
        let look = Look::of(ui);
        let c = self.c_offset();
        PlotFrame::new("efield_e_lambda")
            .x_range(-2.0, 2.0)
            .y_range(-2.2 + c, 5.2 + c)
            .labels("λ", "Eₙ / ħω")
            .width(width)
            .height(height)
            .pan_zoom(PanZoom::None)
            .show(ui, |painter, tr| {
                let ls: Vec<f64> = (0..=100).map(|k| -2.0 + 4.0 * k as f64 / 100.0).collect();
                for n in 0..5 {
                    let ex: Vec<f64> = ls.iter().map(|&l| shifted_energy(n, l) + c).collect();
                    let f1: Vec<f64> = ls.iter().map(|&l| first_order_energy(n, l) + c).collect();
                    plot_dashed(painter, tr, &ls, &f1, look.stroke(1.2, look.prediction()), 6.0);
                    plot_line(painter, tr, &ls, &ex, look.stroke(1.8, look.series(0)));
                    painter.text(
                        tr.to_screen(-2.0, shifted_energy(n, -2.0) + c) + egui::vec2(3.0, -2.0),
                        Align2::LEFT_BOTTOM,
                        format!("n = {n}"),
                        look.font(10.0),
                        look.weak_text(),
                    );
                }
                let sx = tr.x_to_screen(self.lambda);
                painter.line_segment(
                    [pos2(sx, tr.rect.top()), pos2(sx, tr.rect.bottom())],
                    look.stroke(1.3, look.accent()),
                );
                let items = [
                    ("exact = 2nd order: n + ½ − λ²/2", look.series(0)),
                    ("1st order: n + ½ (dashed)", look.prediction()),
                ];
                // (bottom centre: the levels bend away from there)
                let mut y = tr.rect.bottom() - look.px(4.0);
                for (text, col) in items.iter().rev() {
                    let g = painter.layout_no_wrap(text.to_string(), look.font(11.0), *col);
                    let r = Align2::CENTER_BOTTOM.anchor_size(pos2(tr.rect.center().x, y), g.size());
                    painter.rect_filled(r.expand(2.0), 2.0, look.plot_bg().gamma_multiply(0.85));
                    painter.galley(r.min, g, *col);
                    y -= look.px(14.0);
                }
            });
    }
}

impl Default for EfieldScene {
    fn default() -> Self {
        Self::new()
    }
}

impl Scene for EfieldScene {
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
        Some(match self.ladder_basis {
            LadderBasis::New => "Measure E (H)".into(),
            LadderBasis::Old => "Measure E₀ (H₀)".into(),
        })
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
            "basis_old" => self.ladder_basis = LadderBasis::Old,
            "basis_new" => self.ladder_basis = LadderBasis::New,
            "lambda" => {
                let Some(v) = common::num_arg(arg) else { return false };
                self.lambda = v.clamp(-2.0, 2.0);
                self.rebuild();
            }
            "offset" => {
                self.offset_on = true;
                if let Some(v) = common::num_arg(arg) {
                    self.offset = v;
                }
                self.rebuild();
            }
            "level" => {
                let Some(v) = common::num_arg(arg) else { return false };
                self.selected = (v as usize).min(LEVELS - 1);
            }
            "eigen" => {
                self.start = Start::Eigen;
                self.restart();
            }
            "sudden" => {
                self.start = Start::Sudden;
                self.restart();
            }
            _ => return common::view_style_action(cmd, &mut self.view, &mut self.style),
        }
        true
    }

    fn central_ui(&mut self, ui: &mut egui::Ui, _ctx: &mut Ctx) {
        let look = Look::of(ui);
        view_bar(ui, &mut self.view, &mut self.style);
        let n = self.selected;
        let l = self.lambda;
        readout_row(
            ui,
            &[
                ("level n", n.to_string(), Some(look.accent())),
                (
                    "exact Eₙ = n + ½ − λ²/2",
                    format!("{:.4}", shifted_energy(n, l)),
                    Some(look.series(0)),
                ),
                ("1st order", format!("{:.4}", first_order_energy(n, l)), Some(look.prediction())),
                ("2nd order", format!("{:.4}", second_order_energy(n, l)), None),
                ("⟨ξ⟩ₙ exact", fmt_num(l, 4), Some(look.accent())),
                ("⟨ξ⟩ in ψₙ⁽¹⁾", fmt_num(first_order_mean_xi(n, l), 4), Some(look.prediction())),
                ("|⟨ψₙ⁽¹⁾|ψₙ⟩|²", format!("{:.4}", self.overlap(n)), None),
            ],
        );
        let m = self.moments;
        let mut row2 = vec![
            ("now: ⟨ξ⟩", fmt_num(m.x, 3), Some(look.series(0))),
            ("σ_ξ", fmt_num(m.sigma_x, 3), Some(look.series(1))),
            ("⟨H⟩", format!("{:.3}", self.exp.mean_energy()), None),
        ];
        if self.offset_on {
            let ph = (-(self.offset) * self.t).rem_euclid(TAU);
            row2.push(("global phase −Ct mod 2π", format!("{ph:.2}"), Some(look.measure())));
        }
        readout_row(ui, &row2);
        let bottom_h = look.px(200.0);
        let plots_h = (ui.available_height() - bottom_h - look.px(10.0)).max(200.0);
        let view = self.view;
        view.show(ui, plots_h, |ui, d, h| match d {
            Domain::Position => self.plot_x(ui, h),
            Domain::Momentum => self.plot_p(ui, h),
        });
        ui.add_space(look.px(6.0));
        let w = ui.available_width();
        let gap = look.px(8.0);
        let hist_w = if self.repeated.is_some() { (0.26 * w).max(220.0) } else { 0.0 };
        let lam_w = 0.32 * (w - hist_w);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = gap;
            self.plot_levels_vs_lambda(ui, lam_w, bottom_h);
            let ts_w = ui.available_width() - hist_w - if hist_w > 0.0 { gap } else { 0.0 };
            let (t0, lam) = (self.t_prep, self.lambda);
            let sudden = self.start == Start::Sudden && self.events.is_empty();
            let pred = move |t: f64| {
                if sudden && t >= t0 {
                    lam * (1.0 - (t - t0).cos())
                } else {
                    f64::NAN
                }
            };
            let mut ts = TimeSeriesPlot::new("efield_ts")
                .series(&self.mean_xi)
                .series(&self.sigma_x)
                .hline(l, "λ")
                .window(if self.t < 4.0 * TAU {
                    TimeWindow::All
                } else {
                    TimeWindow::Last(4.0 * TAU)
                })
                .now(self.t)
                .labels("t", "")
                .width(ts_w)
                .height(bottom_h);
            if sudden {
                ts = ts.function("λ(1 − cos ωt)", look.prediction(), &pred);
            }
            for (t, e) in &self.events {
                ts = ts.event(*t, e.clone());
            }
            ts.show(ui);
            if let Some(r) = &self.repeated {
                r.show(ui, "efield_hist", hist_w, bottom_h);
            }
        });
    }

    fn side_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx) {
        let look = ctx.look;
        ui.label(RichText::new("Field").strong());
        if ui
            .add(egui::Slider::new(&mut self.lambda, -2.0..=2.0).step_by(0.05).text("λ"))
            .changed()
        {
            self.rebuild();
        }
        hint(
            ui,
            format!(
                "H' = −λħω ξ; the new well is centred at ξ = λ = {:.2} and lowered by λ²/2 = {:.3}.",
                self.lambda,
                0.5 * self.lambda * self.lambda
            ),
        );
        section(ui, "Start");
        segmented(ui, |ui| {
            if ui
                .selectable_label(self.start == Start::Sudden, "old ψ₀, field switched on")
                .clicked()
            {
                self.start = Start::Sudden;
                self.restart();
            }
            if ui.selectable_label(self.start == Start::Eigen, "eigenstate n of H").clicked() {
                self.start = Start::Eigen;
                self.restart();
            }
        });
        hint(
            ui,
            "Click a level in the ladder to select n (the exact and first-order states are drawn on it).",
        );

        section(ui, "Ladder and energy measurement in the basis of");
        segmented(ui, |ui| {
            if ui
                .selectable_label(self.ladder_basis == LadderBasis::New, "new H (with field)")
                .clicked()
            {
                self.ladder_basis = LadderBasis::New;
                self.repeated = None;
            }
            if ui.selectable_label(self.ladder_basis == LadderBasis::Old, "old H₀").clicked() {
                self.ladder_basis = LadderBasis::Old;
                self.repeated = None;
            }
        });

        section(ui, "Part a): constant energy offset");
        let mut changed = ui.checkbox(&mut self.offset_on, "add a constant C to H").changed();
        if self.offset_on {
            changed |= ui.add(egui::Slider::new(&mut self.offset, -3.0..=3.0).text("C / ħω")).changed();
        }
        if changed {
            self.rebuild();
        }
        hint(
            ui,
            "Every level moves by C, so ψ(t) only gets the global factor e^(−iCt/ħ): the phase colour changes, |ψ|² and every expectation value do not.",
        );

        section(ui, "Measure");
        ui.add(
            egui::Slider::new(&mut self.width_x, 0.05..=1.5)
                .logarithmic(true)
                .text("ξ detector σ"),
        );
        ui.horizontal(|ui| {
            let label = if self.ladder_basis == LadderBasis::New {
                "Measure E (H)  (M)"
            } else {
                "Measure E₀ (H₀)  (M)"
            };
            if ui.button(RichText::new(label).color(look.measure())).clicked() {
                self.measure_e(ctx);
            }
            if ui.button(RichText::new("Measure ξ").color(look.measure())).clicked() {
                self.measure_x(ctx);
            }
        });
        if let Some(n) = repeat_button(ui, "efield_rep", "Repeat E", &mut self.repeat_n) {
            self.repeat_e(n, ctx);
        }
        hint(
            ui,
            "In the new basis an energy outcome is stationary; an outcome of H₀ is not an eigenstate of H and keeps oscillating.",
        );
        if let Some(l) = self.flash.label() {
            ui.label(RichText::new(format!("last outcome: {l}")).color(look.measure()));
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
A charged oscillator (frequency ω, charge q) in a uniform field E: H' = −qE·x = −λħω·ξ (as operators) with ξ = √(mω/ħ) x = (â† + â)/√2 and λ = (qE/ω)/√(mħω). Here ξ is the horizontal axis and energies are in units of ħω.
- (a) A constant energy offset C only multiplies ψ by the global phase e^(−iCt/ħ): no force, no change in any probability or expectation value (toggle it below: the colours change, nothing else).
- (b) First order: Eₙ⁽¹⁾ = ⟨n|H'|n⟩ = 0, the levels do not move. (Second order gives −λ²/2 for every level, which is already exact.)
- (c) ψₙ⁽¹⁾ = ψₙ + λ√((n+1)/2) ψₙ₊₁ − λ√(n/2) ψₙ₋₁ (dashed on the selected level).
- (d) ⟨ξ⟩ₙ = λ to first order: the field shifts the equilibrium position, as classically. (Normalising ψₙ⁽¹⁾ gives λ/(1 + λ²(2n+1)/2) = λ + O(λ³); the exact displaced state has exactly λ.)
- (e) Completing the square, ½ξ² − λξ = ½(ξ − λ)² − λ²/2: the same oscillator shifted by λ (x' = x − qE/(mω²)), Eₙ = (n + ½)ħω − q²E²/(2mω²) = (n + ½ − λ²/2)ħω. The first-order result misses the −λ²/2, which is small for a weak field.

# Try
- Move λ: the solid parabola moves to ξ = λ and down by λ²/2; the bottom panel compares Eₙ(λ) (exact) with the flat first-order lines.
- Select levels: the dashed first-order state lies on top of the exact displaced state for small λ; the overlap readout drops as λ grows.
- Field switched on suddenly (default): the old ground state is a coherent state of the new well. It oscillates rigidly between ξ = 0 and 2λ, ⟨ξ⟩ = λ(1 − cos ωt), without changing shape.
- Ladder in the new basis: Poisson weights with mean λ²/2, constant in time. In the old basis: the weights change in time (the state is not stationary for H₀).
- Measure E in the new basis: a stationary state of H. Measure E₀ (old basis): an eigenstate of H₀ centred at 0, which then oscillates about ξ = λ.
- Part a): add C and watch the phase colour; |ψ|², ⟨ξ⟩ and the ladder bars do not change.

# Behind the scenes
The state is expanded in the exact displaced oscillator states (n = 0 … 39) and evolved exactly; the old basis is the λ = 0 one. The first-order states are built from the formula of (c) and normalised for drawing.";

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact levels of ½ξ² − λξ from the finite-difference solver: n + ½ − λ²/2;
    /// first-order states against the exact ones; the sudden switch oscillates as λ(1 − cos t).
    #[test]
    fn shifted_oscillator() {
        let g = Grid1D::new(-12.0, 12.0, 2048);
        for &l in &[0.5, 1.3] {
            let v: Vec<f64> = g.xs().iter().map(|x| 0.5 * x * x - l * x).collect();
            let b = EigenBasis::numeric(g, &v, 5);
            for n in 0..5 {
                assert!(
                    (b.energies[n] - shifted_energy(n, l)).abs() < 2e-3,
                    "λ = {l}, n = {n}: {}",
                    b.energies[n]
                );
                assert!((second_order_energy(n, l) - b.energies[n]).abs() < 2e-3);
                // ⟨ξ⟩ = λ in the exact eigenstates
                let mean: f64 = b.states[n].iter().zip(g.xs()).map(|(p, x)| p * p * x).sum::<f64>() * g.dx;
                assert!((mean - l).abs() < 1e-6, "⟨ξ⟩ = {mean}");
            }
        }
        let mut s = EfieldScene::new();
        s.lambda = 0.2;
        s.rebuild();
        for n in 0..4 {
            // 1 − overlap is of order λ⁴: 0.0026 for n = 3 at λ = 0.2
            assert!(s.overlap(n) > 0.99, "small λ: first order ≈ exact (n = {n}: {})", s.overlap(n));
            // ⟨ξ⟩ of the normalised first-order state
            let w = s.first_order_wave(n);
            let mean: f64 = w.iter().zip(&s.xs).map(|(p, x)| p * p * x).sum::<f64>() * s.grid.dx;
            assert!((mean - first_order_mean_xi(n, 0.2)).abs() < 1e-6);
        }
        let mut s = EfieldScene::new();
        s.lambda = 1.2;
        s.rebuild();
        for k in 1..10 {
            s.advance(0.7);
            let want = 1.2 * (1.0 - s.t.cos());
            assert!((s.moments.x - want).abs() < 1e-6, "t = {}: ⟨ξ⟩ {} vs {want}", s.t, s.moments.x);
            assert!(
                (s.moments.sigma_x - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-6,
                "rigid (k = {k})"
            );
        }
        // new-basis weights after the switch: Poisson with mean λ²/2
        let p = s.exp.probabilities();
        for (n, pn) in p.iter().enumerate().take(8) {
            assert!((pn - crate::physics::poisson(n, 0.72)).abs() < 1e-6);
        }
        // a constant offset changes only a global phase
        let mut a = EfieldScene::new();
        let mut b = EfieldScene::new();
        b.offset_on = true;
        b.offset = 2.3;
        b.rebuild();
        a.advance(1.7);
        b.advance(1.7);
        let ov = a.wave.inner(&b.wave);
        assert!((ov.norm() - 1.0).abs() < 1e-9);
        let d = (ov.arg() + 2.3 * 1.7).rem_euclid(TAU);
        assert!(d < 1e-6 || TAU - d < 1e-6, "global phase −Ct");
    }
}
