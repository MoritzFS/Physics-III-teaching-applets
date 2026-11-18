//! PS13 51 · Larmor precession and the Ramsey sequence.
//!
//! An electron in B = B z: H₀ = μ_B B σz = (ħω/2) σz, ω = 2μ_B B/ħ. Time is in
//! ns and B in mT, so the frequencies are the real ones, shown in slow motion
//! (15 ns per second at speed 1). The state is computed exactly from 2 × 2
//! matrices at every moment of the sequence
//!
//!   |↑z⟩ → R_y(π/2) → free precession for τ → R_second(π/2) → measure σz,
//!
//! with the pulses instantaneous in the model and stretched to a few ns of
//! display time (free precession paused meanwhile). The scene clock t equals
//! the exercise's t during the free precession (the first pulse ends at t = 0).

use std::f64::consts::PI;

use qb_core::egui::{self, Align2, Rect, RichText, Sense, Shape, Vec2, pos2, vec2};
use qb_core::widgets::{
    FlashKind, LevelLadder, MeasureFlash, PanZoom, PlotFrame, TimeSeries, TimeSeriesPlot, TimeWindow, plot_dashed, plot_line, readout_row,
    repeat_button, segmented,
};
use qb_core::{Ctx, Look, QbRng, Scene};

use crate::bloch::{AxisChoice, BlochAxis, BlochSphere, SceneAxis, state_panel};
use crate::spinor::{EY, EZ, Mat2, PulseAxis, Spinor, Vec3, dir, ramsey_final_state, ramsey_p_up, units};
use crate::util::{Key, error_point, fmt_pi, legend, num, panel, sci};

/// The presets of this scene.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RamseyVariant {
    /// R_y(π/2), then free precession for ever (part b, c)
    Larmor,
    /// the full sequence with the second pulse about x (b, d, e)
    Ramsey,
    /// the second pulse about y: the fringe shifts by a quarter period
    RamseyY,
    /// B hidden: read it off the fringe (part e)
    DetermineB,
}

impl RamseyVariant {
    pub const ALL: [RamseyVariant; 4] = [
        RamseyVariant::Larmor,
        RamseyVariant::Ramsey,
        RamseyVariant::RamseyY,
        RamseyVariant::DetermineB,
    ];

    pub fn title(self) -> &'static str {
        match self {
            RamseyVariant::Larmor => "PS13 51 · Larmor precession",
            RamseyVariant::Ramsey => "PS13 51 · Ramsey sequence",
            RamseyVariant::RamseyY => "PS13 51 · Ramsey, 2nd pulse about y",
            RamseyVariant::DetermineB => "PS13 51 · Determine B from the fringe",
        }
    }

    pub fn autoplay(self) -> bool {
        self != RamseyVariant::DetermineB
    }
}

/// A random hidden field for "Determine B": 0.40 … 2.00 mT.
fn hidden_b(rng: &mut QbRng) -> f64 {
    0.4 + (160.0 * rng.uniform()).round() / 100.0
}

/// ns of simulated time per wall-clock second at speed 1.
const TIME_SCALE: f64 = 15.0;
/// ns of |↑z⟩ shown before the first pulse
const PREP: f64 = 6.0;
/// wall seconds the end state is held before the next shot in loop mode
const HOLD: f64 = 0.8;

#[derive(Clone, Copy, Debug)]
enum Seg {
    Free { t0: f64, t1: f64 },
    Pulse { t0: f64, t1: f64, axis: Vec3, angle: f64 },
}

/// Outcomes at one τ.
#[derive(Clone, Copy, Debug)]
struct FringePoint {
    tau: f64,
    up: u64,
    n: u64,
}

/// Shots still to be taken: interleaved sweeps over `taus`.
#[derive(Clone, Debug)]
struct ShotJob {
    taus: Vec<f64>,
    per_point: u64,
    sweeps: u64,
    idx: usize,
}

impl ShotJob {
    fn left(&self) -> u64 {
        (self.per_point - self.sweeps) * self.taus.len() as u64 - self.idx as u64
    }
}

pub struct RamseyScene {
    variant: RamseyVariant,
    // parameters
    b_mt: f64,
    tau: f64,
    second: PulseAxis,
    /// display duration of a pulse (ns)
    pulse: f64,
    tau_max: f64,
    scan_points: usize,
    repeat_n: usize,
    loop_shots: bool,
    show_u: bool,
    u_corotating: bool,
    u_offset: f64,
    axis_choice: AxisChoice,
    revealed: bool,
    cursors: [f64; 2],
    periods: u32,
    grab: Option<usize>,

    // state
    t: f64,
    collapsed: Option<(f64, Spinor)>,
    done: bool,
    hold: f64,
    trail: Vec<Vec3>,
    last_sample: f64,
    sx: TimeSeries,
    sy: TimeSeries,
    sz: TimeSeries,
    pu: TimeSeries,
    events: Vec<(f64, String)>,
    fringe: Vec<FringePoint>,
    job: Option<ShotJob>,
    flash: MeasureFlash,
}

impl RamseyScene {
    pub fn new(variant: RamseyVariant, rng: &mut QbRng) -> Self {
        let hidden_b = hidden_b(rng);
        let determine = variant == RamseyVariant::DetermineB;
        let mut s = RamseyScene {
            variant,
            b_mt: if variant == RamseyVariant::DetermineB { hidden_b } else { 1.0 },
            tau: 40.0,
            second: if variant == RamseyVariant::RamseyY {
                PulseAxis::Y
            } else {
                PulseAxis::X
            },
            pulse: 10.0,
            tau_max: if determine { 120.0 } else { 150.0 },
            scan_points: if determine { 121 } else { 61 },
            repeat_n: 100,
            loop_shots: false,
            show_u: variant == RamseyVariant::Larmor,
            u_corotating: true,
            u_offset: 0.0,
            axis_choice: AxisChoice::Z,
            revealed: false,
            cursors: [10.0, 46.0],
            periods: 1,
            grab: None,
            t: 0.0,
            collapsed: None,
            done: false,
            hold: 0.0,
            trail: vec![],
            last_sample: f64::NEG_INFINITY,
            sx: TimeSeries::new("⟨σx⟩"),
            sy: TimeSeries::new("⟨σy⟩"),
            sz: TimeSeries::new("⟨σz⟩"),
            pu: TimeSeries::new("P(↑u)"),
            events: vec![],
            fringe: vec![],
            job: None,
            flash: MeasureFlash::new(),
        };
        if variant == RamseyVariant::Larmor {
            s.axis_choice = AxisChoice::U;
        }
        s.new_shot();
        s
    }

    fn larmor(&self) -> bool {
        self.variant == RamseyVariant::Larmor
    }

    fn hidden(&self) -> bool {
        self.variant == RamseyVariant::DetermineB && !self.revealed
    }

    /// ω in rad/ns.
    fn omega(&self) -> f64 {
        units::omega_per_ns(self.b_mt)
    }

    fn t0(&self) -> f64 {
        -(PREP + self.pulse)
    }

    /// Time of the σz measurement (None: Larmor, no end).
    fn t_end(&self) -> Option<f64> {
        if self.larmor() { None } else { Some(self.tau + self.pulse) }
    }

    fn segments(&self) -> Vec<Seg> {
        let d = self.pulse;
        let mut v = vec![
            Seg::Free { t0: self.t0(), t1: -d },
            Seg::Pulse {
                t0: -d,
                t1: 0.0,
                axis: EY,
                angle: 0.5 * PI,
            },
        ];
        if self.larmor() {
            v.push(Seg::Free {
                t0: 0.0,
                t1: f64::INFINITY,
            });
        } else {
            v.push(Seg::Free { t0: 0.0, t1: self.tau });
            v.push(Seg::Pulse {
                t0: self.tau,
                t1: self.tau + d,
                axis: self.second.dir(),
                angle: 0.5 * PI,
            });
            v.push(Seg::Free {
                t0: self.tau + d,
                t1: f64::INFINITY,
            });
        }
        v
    }

    /// The exercise's time: t during the free precession, frozen during the (instantaneous) pulses.
    fn physics_time(&self, t: f64) -> f64 {
        let d = self.pulse;
        if t < -d {
            t + d
        } else if t < 0.0 {
            0.0
        } else if self.larmor() || t < self.tau {
            t
        } else if t < self.tau + d {
            self.tau
        } else {
            t - d
        }
    }

    /// The state at display time t, from the start (or the last collapse) by exact matrix products.
    fn state_at(&self, t: f64) -> Spinor {
        let omega = self.omega();
        let (ts, mut psi) = match self.collapsed {
            Some((tc, s)) if t >= tc => (tc, s),
            // e^(iω·PREP/2)|↑z⟩, so that the free evolution brings it to exactly |↑z⟩ when the pulse starts
            _ => (self.t0(), Spinor::UP_Z.scaled(qb_core::C64::from_polar(1.0, 0.5 * omega * PREP))),
        };
        for seg in self.segments() {
            let (a, b) = match seg {
                Seg::Free { t0, t1 } | Seg::Pulse { t0, t1, .. } => (t0, t1),
            };
            let (lo, hi) = (a.max(ts), b.min(t));
            if hi <= lo {
                continue;
            }
            psi = match seg {
                Seg::Free { .. } => Mat2::free_evolution(omega, hi - lo) * psi,
                Seg::Pulse { axis, angle, .. } => Mat2::rotation(axis, angle * (hi - lo) / (b - a)) * psi,
            };
        }
        psi
    }

    /// 0 before the first pulse, 1 first pulse, 2 free precession, 3 second pulse, 4 measured.
    fn stage(&self) -> usize {
        let d = self.pulse;
        if self.done {
            4
        } else if self.t < -d {
            0
        } else if self.t < 0.0 {
            1
        } else if self.larmor() || self.t < self.tau {
            2
        } else {
            3
        }
    }

    /// Azimuth φ of the measurement axis u(t) of part c (in the x–y plane), not wrapped:
    /// |↑u⟩ of Eq. 50.3 changes sign when φ goes once around.
    fn u_phi(&self, t: f64) -> f64 {
        if self.u_corotating {
            self.omega() * self.physics_time(t) + self.u_offset
        } else {
            self.u_offset
        }
    }

    fn u_dir(&self, t: f64) -> Vec3 {
        dir(0.5 * PI, self.u_phi(t))
    }

    /// Back to t = t₀ with |↑z⟩ (keeps the fringe data).
    fn new_shot(&mut self) {
        self.t = self.t0();
        self.collapsed = None;
        self.done = false;
        self.hold = 0.0;
        self.events.clear();
        self.flash.clear();
        self.rebuild_history();
    }

    /// Recomputes the trail and the time series from the start up to t.
    fn rebuild_history(&mut self) {
        self.trail.clear();
        for s in [&mut self.sx, &mut self.sy, &mut self.sz, &mut self.pu] {
            s.clear();
        }
        self.last_sample = f64::NEG_INFINITY;
        let dt = self.sample_dt();
        let t_end = self.t;
        let mut t = self.t0();
        let tc = self.collapsed.map(|c| c.0);
        while t < t_end {
            if let Some(tc) = tc
                && t < tc
                && t + dt >= tc
            {
                self.record_at(tc - 1e-9);
                self.trail.push([f64::NAN; 3]);
            }
            self.record_at(t);
            t += dt;
        }
        self.record_at(t_end);
    }

    fn sample_dt(&self) -> f64 {
        (0.08 / self.omega().max(1e-6)).min(0.25)
    }

    fn record_at(&mut self, t: f64) {
        let psi = self.state_at(t);
        let r = psi.bloch();
        self.trail.push(r);
        if self.trail.len() > 700 {
            self.trail.drain(..100);
        }
        self.sx.push(t, r[0]);
        self.sy.push(t, r[1]);
        self.sz.push(t, r[2]);
        if self.show_u {
            self.pu.push(t, psi.p_up(self.u_dir(t)));
        }
        self.last_sample = t;
    }

    fn add_fringe(&mut self, tau: f64, up: bool) {
        match self.fringe.iter_mut().find(|p| (p.tau - tau).abs() < 1e-6) {
            Some(p) => {
                p.n += 1;
                p.up += up as u64;
            }
            None => self.fringe.push(FringePoint { tau, up: up as u64, n: 1 }),
        }
    }

    /// The σz measurement at the end of the sequence.
    fn end_measurement(&mut self, ctx: &mut Ctx) {
        let Some(te) = self.t_end() else { return };
        let psi = self.state_at(te);
        let p = psi.p_up(EZ);
        let up = ctx.rng.bool(p);
        self.collapsed = Some((te, if up { Spinor::UP_Z } else { Spinor::DOWN_Z }));
        self.done = true;
        self.hold = 0.0;
        self.add_fringe(self.tau, up);
        self.trail.push([f64::NAN; 3]);
        self.record_at(te);
        let a = if up { "↑z" } else { "↓z" };
        self.events.push((te, a.into()));
        self.flash.trigger(FlashKind::Other, format!("σz: {a}"));
        let pinfo = if self.hidden() {
            String::new()
        } else {
            format!(" (P(↑z) = {p:.3})")
        };
        ctx.set_status(format!("shot at τ = {:.1} ns: {a}{pinfo}; added to the fringe plot", self.tau));
    }

    /// Key M: measure σz now (collapse), or start a new shot after the end.
    fn measure_now(&mut self, ctx: &mut Ctx) {
        if self.done {
            self.new_shot();
            ctx.playing = true;
            ctx.set_status("new shot");
            return;
        }
        let psi = self.state_at(self.t);
        let p = psi.p_up(EZ);
        let up = ctx.rng.bool(p);
        self.collapse(if up { Spinor::UP_Z } else { Spinor::DOWN_Z });
        let a = if up { "↑z" } else { "↓z" };
        self.events.push((self.t, a.into()));
        self.flash.trigger(FlashKind::Other, format!("σz: {a}"));
        ctx.set_status(format!(
            "measured σz at t = {:.1} ns: {a} (P(↑z) was {p:.3}); not at the end of the sequence, so not added to the fringe",
            self.t
        ));
    }

    /// Part c: measure along u(t).
    fn measure_u(&mut self, ctx: &mut Ctx) {
        let phi = self.u_phi(self.t);
        let psi = self.state_at(self.t);
        let p = psi.p_up(self.u_dir(self.t));
        let up = ctx.rng.bool(p);
        // Eq. 50.3 with the unwrapped φ: for φ = ωt, |↑u⟩ is exactly |φ(t)⟩ of Eq. 51.5
        self.collapse(if up {
            Spinor::up_along(0.5 * PI, phi)
        } else {
            Spinor::down_along(0.5 * PI, phi)
        });
        let a = if up { "↑u" } else { "↓u" };
        self.events.push((self.t, a.into()));
        self.flash.trigger(FlashKind::Other, format!("u(t): {a}"));
        ctx.set_status(format!("measured along u(t) at t = {:.1} ns: {a} (P(↑u) was {p:.3})", self.t));
    }

    fn collapse(&mut self, s: Spinor) {
        self.collapsed = Some((self.t, s));
        self.trail.push([f64::NAN; 3]);
        self.record_at(self.t);
    }

    /// Takes up to `budget` shots of the queued job.
    fn run_job(&mut self, budget: u64, rng: &mut QbRng) {
        let Some(mut job) = self.job.take() else { return };
        let omega = self.omega();
        let mut left = budget;
        while left > 0 && job.sweeps < job.per_point {
            let tau = job.taus[job.idx];
            // explicit matrix product (not the formula), see the tests
            let p = ramsey_final_state(omega, tau, self.second).prob(&Spinor::UP_Z);
            let up = rng.bool(p);
            self.add_fringe(tau, up);
            job.idx += 1;
            if job.idx == job.taus.len() {
                job.idx = 0;
                job.sweeps += 1;
            }
            left -= 1;
        }
        if job.sweeps < job.per_point {
            self.job = Some(job);
        }
    }

    fn queue_here(&mut self, n: usize) {
        self.job = Some(ShotJob {
            taus: vec![self.tau],
            per_point: n as u64,
            sweeps: 0,
            idx: 0,
        });
    }

    fn queue_scan(&mut self, n: usize) {
        let k = self.scan_points.max(2);
        self.job = Some(ShotJob {
            taus: (0..k).map(|i| self.tau_max * i as f64 / (k - 1) as f64).collect(),
            per_point: n as u64,
            sweeps: 0,
            idx: 0,
        });
    }

    fn set_tau(&mut self, tau: f64) {
        self.tau = tau.clamp(0.0, 400.0);
        self.new_shot();
    }

    // ------------------------------------------------------------ drawing

    /// The sequence as a timeline with a playhead; returns a time the user clicked or dragged to.
    fn timeline(&self, ui: &mut egui::Ui, width: f32) -> Option<f64> {
        let look = Look::of(ui);
        let h = look.px(64.0);
        let (rect, resp) = ui.allocate_exact_size(vec2(width, h), Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        panel(&painter, rect);
        let t0 = self.t0();
        let (lo, hi) = match self.t_end() {
            Some(te) => (t0, te + 0.04 * (te - t0) + 3.0),
            None => {
                let span = 130.0;
                let hi = (t0 + span).max(self.t + 15.0);
                (hi - span, hi)
            }
        };
        let inner = rect.shrink2(vec2(look.px(12.0), look.px(5.0)));
        let tx = |t: f64| inner.left() + ((t - lo) / (hi - lo)) as f32 * inner.width();
        let bar_top = inner.top() + look.px(17.0);
        let bar_bot = bar_top + look.px(20.0);
        let block = |a: f64, b: f64, fill: egui::Color32, text: &str, text_col: egui::Color32| {
            let (xa, xb) = (tx(a.max(lo)), tx(b.min(hi)));
            if xb <= xa {
                return;
            }
            let r = Rect::from_min_max(pos2(xa, bar_top), pos2(xb, bar_bot));
            painter.rect_filled(r, 2.0, fill);
            let g = painter.layout_no_wrap(text.to_string(), look.font(11.5), text_col);
            if g.size().x < r.width() - 4.0 {
                painter.galley(r.center() - 0.5 * g.size(), g, text_col);
            }
        };
        let d = self.pulse;
        let pulse_col = look.series(2).gamma_multiply(0.55);
        block(t0, -d, look.grid(), "|↑z⟩", look.weak_text());
        block(-d, 0.0, pulse_col, "R_y(π/2)", look.text());
        let free_end = self.t_end().map_or(hi, |_| self.tau);
        let free_text = if self.larmor() {
            "free precession about z (no second pulse)".to_string()
        } else {
            format!("free precession, τ = {} ns", num(self.tau, 1))
        };
        block(0.0, free_end, look.prediction().gamma_multiply(0.18), &free_text, look.text());
        if let Some(te) = self.t_end() {
            let second = if self.second == PulseAxis::X { "R_x(π/2)" } else { "R_y(π/2)" };
            block(self.tau, te, look.series(3).gamma_multiply(0.5), second, look.text());
            // the σz detector
            let x = tx(te);
            let s = look.px(7.0);
            painter.add(Shape::convex_polygon(
                vec![
                    pos2(x, bar_top - 1.0),
                    pos2(x + 1.6 * s, bar_top - s),
                    pos2(x + 1.6 * s, bar_bot + s),
                ],
                look.measure(),
                egui::Stroke::NONE,
            ));
            painter.text(
                pos2(x + 1.8 * s, 0.5 * (bar_top + bar_bot)),
                Align2::LEFT_CENTER,
                "σz",
                look.font(12.0),
                look.measure(),
            );
        }
        // ticks
        let tick = |t: f64, label: &str| {
            if t < lo || t > hi {
                return;
            }
            let x = tx(t);
            painter.line_segment([pos2(x, bar_bot), pos2(x, bar_bot + look.px(4.0))], look.stroke(1.0, look.axis()));
            painter.text(
                pos2(x, bar_bot + look.px(4.0)),
                Align2::CENTER_TOP,
                label,
                look.font(10.5),
                look.weak_text(),
            );
        };
        let step = if hi - lo > 160.0 { 50.0 } else { 20.0 };
        let mut t = (lo / step).ceil() * step;
        while t <= hi {
            if (t - self.tau).abs() > 0.1 * step || self.larmor() {
                tick(t, &format!("{}", t as i64));
            }
            t += step;
        }
        if !self.larmor() {
            tick(self.tau, "τ");
        }
        painter.text(
            inner.left_top(),
            Align2::LEFT_TOP,
            format!(
                "timeline, t in ns · pulses are instantaneous in the model, drawn {} ns long (display only, precession paused) · drag to scrub",
                num(self.pulse, 0)
            ),
            look.font(10.5),
            look.weak_text(),
        );
        // playhead
        let x = tx(self.t);
        painter.line_segment(
            [pos2(x, bar_top - look.px(4.0)), pos2(x, bar_bot + look.px(2.0))],
            look.stroke(2.0, look.accent()),
        );
        painter.circle_filled(pos2(x, bar_top - look.px(4.0)), look.px(3.5), look.accent());

        if (resp.dragged() || resp.clicked())
            && !look.shot
            && let Some(p) = resp.interact_pointer_pos()
        {
            let t = lo + ((p.x - inner.left()) / inner.width()) as f64 * (hi - lo);
            let max = self.t_end().unwrap_or(hi);
            return Some(t.clamp(t0, max - 1e-6));
        }
        None
    }

    /// The stage-by-stage formulas of the exercise, the current one highlighted.
    fn formulas(&self, ui: &mut egui::Ui) {
        let look = Look::of(ui);
        let stage = self.stage();
        let hidden = self.hidden();
        let omega = self.omega();
        let line = |ui: &mut egui::Ui, i: usize, text: &str| {
            let (mark, col) = if i == stage {
                ("▶ ", look.accent())
            } else {
                ("   ", look.text())
            };
            ui.label(RichText::new(format!("{mark}{text}")).color(col));
        };
        ui.label(RichText::new("The sequence (ω = 2μ_B B/ħ, t in ns)").strong());
        line(ui, 0, "t < 0:  |↑z⟩ = (1, 0)ᵀ, stationary (only its phase turns)");
        line(ui, 1, "t = 0:  |φ(0)⟩ = R_y(π/2)|↑z⟩ = (1, 1)ᵀ/√2 = |↑x⟩   (51.8)");
        line(ui, 2, "|φ(t)⟩ = e^(−iH₀t/ħ)|φ(0)⟩ = (e^(−iωt/2), e^(iωt/2))ᵀ/√2   (51.5)");
        if self.larmor() {
            ui.label(
                RichText::new("   (c) P(↑u|φ(t)) = cos²(φ/2 − ωt/2) = 1 for φ = ωt   (51.12)").color(if self.show_u {
                    look.measure()
                } else {
                    look.text()
                }),
            );
        } else if self.second == PulseAxis::X {
            line(
                ui,
                3,
                "|ϕ(τ)⟩ = R_x(π/2)|φ(τ)⟩ = ½ (e^(−iωτ/2) − i e^(iωτ/2), −i e^(−iωτ/2) + e^(iωτ/2))ᵀ   (51.13)",
            );
            let p = if hidden {
                "?".to_string()
            } else {
                format!("{:.3}", ramsey_p_up(omega, self.tau, PulseAxis::X))
            };
            line(ui, 4, &format!("P(↑z|ϕ(τ)) = (1 + sin ωτ)/2 = {p}   (51.14)"));
        } else {
            line(ui, 3, "|ϕ(τ)⟩ = R_y(π/2)|φ(τ)⟩ = (−i sin(ωτ/2), cos(ωτ/2))ᵀ");
            let p = if hidden {
                "?".to_string()
            } else {
                format!("{:.3}", ramsey_p_up(omega, self.tau, PulseAxis::Y))
            };
            line(ui, 4, &format!("P(↑z|ϕ(τ)) = (1 − cos ωτ)/2 = sin²(ωτ/2) = {p}"));
        }
        let tp = self.physics_time(self.t);
        let now = if hidden {
            format!("now: t = {} ns", num(tp, 1))
        } else {
            format!("now: t = {} ns,  ωt = {}", num(tp, 1), fmt_pi(omega * tp))
        };
        ui.label(RichText::new(now).color(look.weak_text()).monospace());
        if stage == 1 || stage == 3 {
            ui.label(
                RichText::new("pulse: instantaneous in the exercise, shown as a slow rotation (display only)").color(look.weak_text()),
            );
        }
    }

    fn fringe_plot(&mut self, ui: &mut egui::Ui, size: Vec2) {
        let look = Look::of(ui);
        let omega = self.omega();
        let hidden = self.hidden();
        let second = self.second;
        let determine = self.variant == RamseyVariant::DetermineB;
        let tau = self.tau;
        let cursors = self.cursors;
        let fringe = self.fringe.clone();
        let total: u64 = fringe.iter().map(|p| p.n).sum();
        let tau_max = self.tau_max;
        let variant = self.variant;
        let resp = PlotFrame::new(("ramsey_fringe", (tau_max * 10.0) as i64))
            .x_range(0.0, tau_max)
            .y_range(-0.05, 1.12)
            .labels("τ (ns)", "P(↑z) at the end")
            .pan_zoom(PanZoom::None)
            .left_drag_pans(false)
            .width(size.x)
            .height(size.y)
            .show(ui, |painter, tr| {
                let xs: Vec<f64> = (0..=800).map(|i| tau_max * i as f64 / 800.0).collect();
                if !hidden {
                    if variant == RamseyVariant::RamseyY {
                        let ys: Vec<f64> = xs.iter().map(|&t| ramsey_p_up(omega, t, PulseAxis::X)).collect();
                        plot_dashed(painter, tr, &xs, &ys, look.stroke(1.2, look.weak_text()), 5.0);
                    }
                    let ys: Vec<f64> = xs.iter().map(|&t| ramsey_p_up(omega, t, second)).collect();
                    plot_line(painter, tr, &xs, &ys, look.stroke(1.8, look.prediction()));
                }
                if !determine {
                    painter.line_segment(
                        [tr.to_screen(tau, -0.05), tr.to_screen(tau, 1.12)],
                        look.stroke(1.2, look.accent().gamma_multiply(0.8)),
                    );
                    painter.text(
                        tr.to_screen(tau, 1.1) + vec2(3.0, 0.0),
                        Align2::LEFT_TOP,
                        "τ",
                        look.font(12.0),
                        look.accent(),
                    );
                }
                for p in &fringe {
                    let f = p.up as f64 / p.n as f64;
                    let q = if hidden { f } else { ramsey_p_up(omega, p.tau, second) };
                    let sd = (q * (1.0 - q) / p.n as f64).sqrt();
                    error_point(painter, tr, p.tau, f, sd, look.counts());
                }
                if determine {
                    for (i, c) in cursors.iter().enumerate() {
                        let a = tr.to_screen(*c, -0.05);
                        let b = tr.to_screen(*c, 1.12);
                        painter.line_segment([a, b], look.stroke(1.6, look.measure()));
                        // handle and name at the bottom (the top holds the axis label and the legend)
                        let h = pos2(a.x, tr.rect.bottom() - look.px(9.0));
                        painter.circle_filled(h, look.px(5.5), look.measure());
                        painter.text(
                            h + vec2(look.px(8.0), 0.0),
                            Align2::LEFT_CENTER,
                            ["A", "B"][i],
                            look.font(13.0),
                            look.measure(),
                        );
                    }
                }
                let r = tr.rect;
                let mut entries = vec![];
                if !hidden {
                    let label = match second {
                        PulseAxis::X => "Born rule (1 + sin ωτ)/2  (51.14)",
                        PulseAxis::Y => "Born rule (1 − cos ωτ)/2",
                    };
                    entries.push((label.to_string(), look.prediction(), Key::Line));
                    if variant == RamseyVariant::RamseyY {
                        entries.push(("2nd pulse about x (51.14), for comparison".into(), look.weak_text(), Key::Dashed));
                    }
                }
                entries.push(("measured ± √(p(1−p)/N)".into(), look.counts(), Key::Point));
                entries.push((
                    format!("{total} shots at {} values of τ", fringe.len()),
                    look.weak_text(),
                    Key::None,
                ));
                legend(painter, pos2(r.right() - look.px(8.0), r.top() + look.px(6.0)), &entries);
            });
        let r = &resp.response;
        let tr = resp.transform;
        if look.shot {
            return;
        }
        if determine {
            if r.drag_started()
                && let Some(p) = r.interact_pointer_pos()
            {
                let x = tr.x_from_screen(p.x);
                let i = if (x - self.cursors[0]).abs() <= (x - self.cursors[1]).abs() {
                    0
                } else {
                    1
                };
                self.grab = Some(i);
            }
            if r.dragged()
                && let (Some(i), Some(p)) = (self.grab, r.interact_pointer_pos())
            {
                self.cursors[i] = tr.x_from_screen(p.x).clamp(0.0, self.tau_max);
            }
            if r.drag_stopped() {
                self.grab = None;
            }
        } else if r.clicked()
            && let Some(p) = r.interact_pointer_pos()
        {
            let t = (tr.x_from_screen(p.x) * 2.0).round() / 2.0;
            self.set_tau(t.clamp(0.0, self.tau_max));
        }
    }
}

impl Scene for RamseyScene {
    fn time(&self) -> f64 {
        self.t
    }

    fn time_scale(&self) -> f64 {
        TIME_SCALE
    }

    fn step(&mut self, dt: f64, ctx: &mut Ctx) {
        if self.done {
            self.hold += dt / TIME_SCALE;
            if !self.loop_shots || self.hold >= HOLD {
                self.new_shot();
            }
            return;
        }
        let t_new = self.t + dt;
        if let Some(te) = self.t_end()
            && t_new >= te
        {
            self.t = te;
            self.record_at(te);
            self.end_measurement(ctx);
            if !self.loop_shots {
                ctx.playing = false;
            }
            return;
        }
        self.t = t_new;
        if self.t - self.last_sample >= self.sample_dt() {
            self.record_at(self.t);
        }
    }

    fn reset(&mut self, _ctx: &mut Ctx) {
        self.new_shot();
    }

    fn measure_label(&self) -> Option<String> {
        Some(if self.done { "New shot".into() } else { "Measure σz".into() })
    }

    fn measure(&mut self, ctx: &mut Ctx) {
        self.measure_now(ctx);
    }

    fn action(&mut self, action: &str, ctx: &mut Ctx) -> bool {
        let mut words = action.split_whitespace();
        let cmd = words.next().unwrap_or("");
        let arg = words.next();
        let val = arg.and_then(|w| w.parse::<f64>().ok());
        let n = val.map_or(self.repeat_n, |v| v as usize);
        match cmd {
            "measure" => self.measure_now(ctx),
            "measure_u" => self.measure_u(ctx),
            "shots" => {
                self.queue_here(n);
                self.run_job(u64::MAX, ctx.rng);
            }
            "scan" => {
                self.queue_scan(n);
                // optional second argument: only this many sweeps (a scan caught while building up)
                let sweeps: Option<u64> = words.next().and_then(|w| w.parse().ok());
                let budget = sweeps.map_or(u64::MAX, |s| s * self.scan_points as u64);
                self.run_job(budget, ctx.rng);
            }
            "tau" => self.set_tau(val.unwrap_or(self.tau)),
            "b" => {
                self.b_mt = val.unwrap_or(self.b_mt).clamp(0.05, 10.0);
                self.fringe.clear();
                self.new_shot();
            }
            "loop" => self.loop_shots = true,
            "u_on" => {
                self.show_u = true;
                self.rebuild_history();
            }
            "u_fixed" => {
                self.show_u = true;
                self.u_corotating = false;
                self.rebuild_history();
            }
            "reveal" => self.revealed = true,
            "cursors" => {
                let b: Option<f64> = words.next().and_then(|w| w.parse().ok());
                if let (Some(a), Some(b)) = (val, b) {
                    self.cursors = [a, b];
                }
                if let Some(k) = words.next().and_then(|w| w.parse().ok()) {
                    self.periods = k;
                }
            }
            "axis_x" => self.axis_choice = AxisChoice::X,
            "axis_z" => self.axis_choice = AxisChoice::Z,
            _ => return false,
        }
        true
    }

    fn central_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx) {
        let look = Look::of(ui);
        // shots queued by "run N" / "scan": a few thousand per second, interleaved over τ
        if let Some(job) = &self.job {
            let total = job.taus.len() as u64 * job.per_point;
            let rate = (total as f64 / 3.0).clamp(300.0, 40_000.0);
            let budget = if ctx.shot {
                job.left()
            } else {
                (rate * ctx.dt_wall.max(1.0 / 60.0)).ceil() as u64
            };
            self.run_job(budget, ctx.rng);
            ui.ctx().request_repaint();
        }

        let hidden = self.hidden();
        let omega = self.omega();
        let psi = self.state_at(self.t);
        let f_mhz = units::larmor_freq(self.b_mt * 1e-3) / 1e6;
        let mut items: Vec<(&str, String, Option<egui::Color32>)> = vec![];
        if hidden {
            items.push(("B", "hidden".into(), Some(look.measure())));
        } else {
            items.push(("B", format!("{} mT", num(self.b_mt, 2)), None));
            items.push(("ω = 2μ_B B/ħ", format!("{} rad/s", sci(omega * 1e9, 4)), None));
            items.push(("f = ω/2π", format!("{f_mhz:.2} MHz"), Some(look.prediction())));
            items.push(("period", format!("{:.1} ns", 1e3 / f_mhz), None));
        }
        if !self.larmor() {
            items.push(("τ", format!("{} ns", num(self.tau, 1)), Some(look.accent())));
        }
        if !hidden {
            items.push(("P(↑z) now", format!("{:.3}", psi.p_up(EZ)), None));
        }
        readout_row(ui, &items);

        if let Some(t) = self.timeline(ui, ui.available_width()) {
            self.t = t;
            self.collapsed = None;
            self.done = false;
            self.events.clear();
            self.rebuild_history();
        }
        ui.add_space(look.px(4.0));

        let avail = ui.available_size();
        // Larmor: no fringe, the middle row takes the whole height
        let bottom_h = if self.larmor() {
            0.0
        } else if hidden {
            avail.y * 0.62
        } else {
            (avail.y * 0.38).max(look.px(170.0))
        };
        let mid_h = avail.y - bottom_h - if self.larmor() { 0.0 } else { look.px(8.0) };
        let u = self.u_dir(self.t);
        let stage = self.stage();
        ui.horizontal(|ui| {
            ui.set_height(mid_h);
            let frac = if self.larmor() { 0.4 } else { 0.34 };
            let sph = mid_h.min(ui.available_width() * frac);
            if hidden {
                let (rect, _) = ui.allocate_exact_size(vec2(sph, mid_h), Sense::hover());
                let painter = ui.painter_at(rect);
                panel(&painter, rect);
                painter.text(
                    rect.center(),
                    Align2::CENTER_CENTER,
                    "state hidden:\nin the lab you only get\nthe σz outcomes.\nRead B off the fringe.",
                    look.font(14.0),
                    look.weak_text(),
                );
            } else {
                let caption = match stage {
                    0 => "|↑z⟩: stationary".to_string(),
                    1 => "first pulse R_y(π/2)".to_string(),
                    2 => "free precession about B ∥ z at ω".to_string(),
                    3 => format!("second pulse R_{}(π/2)", self.second.name()),
                    _ => "measured σz: collapsed".to_string(),
                };
                let mut s = BlochSphere::new("ramsey_bloch", sph)
                    .size(sph, mid_h)
                    .state(psi.bloch())
                    .trail(&self.trail)
                    .title("Bloch sphere")
                    .caption(caption);
                if stage == 1 {
                    s = s.axis(BlochAxis::rotation(EY, "R_y", look.series(2)));
                }
                if stage == 3 {
                    s = s.axis(BlochAxis::rotation(
                        self.second.dir(),
                        format!("R_{}", self.second.name()),
                        look.series(3),
                    ));
                }
                if stage >= 3 && !self.larmor() {
                    s = s.axis(BlochAxis::measurement(EZ, "", "", look.measure()));
                }
                if self.show_u {
                    // drawn as an axis through the sphere: the state arrow lies along it when P(↑u) = 1
                    s = s.axis(BlochAxis::measurement(u, "u(t)", "", look.measure()));
                }
                let resp = s.show(ui);
                self.flash
                    .paint_banner(ui, Rect::from_min_max(resp.rect.min + vec2(0.0, look.px(18.0)), resp.rect.max));
            }

            ui.vertical(|ui| {
                let w = look.px(262.0);
                ui.set_width(w);
                if hidden {
                    ui.label(RichText::new("(column vector and levels hidden)").color(look.weak_text()));
                } else {
                    let u_opt = self.show_u.then_some(SceneAxis {
                        dir: u,
                        name: "u",
                        button: "u(t)",
                    });
                    state_panel(ui, &psi, &mut self.axis_choice, u_opt, w);
                    ui.add_space(look.px(4.0));
                    ui.label(RichText::new("levels of H₀ = μ_B B σz (E in ħω)").color(look.weak_text()));
                    let c = [psi.down, psi.up];
                    let labels = ["|↓z⟩  −ħω/2".to_string(), "|↑z⟩  +ħω/2".to_string()];
                    let lh = (ui.available_height() - look.px(4.0)).max(look.px(90.0));
                    LevelLadder::new("ramsey_ladder", &[-0.5, 0.5])
                        .coeffs(&c)
                        .labels(&labels)
                        .energy_range(-0.9, 0.9)
                        .size(vec2(w, lh))
                        .show(ui);
                }
            });

            ui.vertical(|ui| {
                self.formulas(ui);
                if !hidden {
                    ui.add_space(look.px(4.0));
                    let h = ui.available_height();
                    if h > look.px(80.0) {
                        let t0 = self.t0();
                        let window = match self.t_end() {
                            Some(te) => TimeWindow::Fixed(t0, te),
                            None => {
                                let lo = t0.max(self.t - 120.0);
                                TimeWindow::Fixed(lo, lo + 120.0)
                            }
                        };
                        let mut ts = TimeSeriesPlot::new("ramsey_ts")
                            .series(&self.sx)
                            .series(&self.sy)
                            .series(&self.sz)
                            .window(window)
                            .y_range(-1.15, 1.35)
                            .now(self.t)
                            .labels("t (ns)", "")
                            .height(h);
                        if self.show_u {
                            ts = ts.series(&self.pu);
                        }
                        for (t, l) in &self.events {
                            ts = ts.event(*t, l.clone());
                        }
                        ts.show(ui);
                    }
                }
            });
        });
        if !self.larmor() {
            ui.add_space(look.px(6.0));
            let size = vec2(ui.available_width(), ui.available_height());
            self.fringe_plot(ui, size);
        }
    }

    fn side_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx) {
        let look = ctx.look;
        let hidden = self.hidden();
        ui.label(RichText::new("Field").strong());
        if hidden {
            ui.label(RichText::new("B is hidden (0.4 … 2 mT): find it from the fringe period.").color(look.measure()));
        } else {
            if ui
                .add(egui::Slider::new(&mut self.b_mt, 0.1..=5.0).logarithmic(true).text("B (mT)"))
                .changed()
            {
                self.fringe.clear();
                self.job = None;
                self.new_shot();
            }
            let f = units::larmor_freq(self.b_mt * 1e-3);
            ui.label(
                RichText::new(format!(
                    "ω = 2μ_B B/ħ = 2π · {:.2} MHz,  ħω = 2μ_B B = {:.3} μeV\n(1 mT ↔ 27.99 MHz; g_s ≈ 2 as in Eq. 51.1)",
                    f / 1e6,
                    units::splitting_ev(self.b_mt * 1e-3) * 1e6
                ))
                .color(look.weak_text()),
            );
        }

        if !self.larmor() {
            ui.add_space(look.px(8.0));
            ui.label(RichText::new("Sequence").strong());
            let mut tau = self.tau;
            if ui
                .add(egui::Slider::new(&mut tau, 0.0..=self.tau_max).text("τ (ns)").step_by(0.5))
                .changed()
            {
                self.set_tau(tau);
            }
            ui.horizontal(|ui| {
                ui.label(RichText::new("second pulse").color(look.weak_text()));
                let mut changed = false;
                segmented(ui, |ui| {
                    changed |= ui.selectable_value(&mut self.second, PulseAxis::X, "R_x(π/2)").changed();
                    changed |= ui.selectable_value(&mut self.second, PulseAxis::Y, "R_y(π/2)").changed();
                });
                if changed {
                    self.fringe.clear();
                    self.job = None;
                    self.new_shot();
                }
            });
            ui.checkbox(&mut self.loop_shots, "loop: repeat the sequence (one shot each)");
        }
        if ui
            .add(egui::Slider::new(&mut self.pulse, 2.0..=30.0).text("pulse display length (ns)"))
            .changed()
        {
            self.new_shot();
        }

        ui.add_space(look.px(8.0));
        ui.label(RichText::new("Measure").strong());
        ui.horizontal(|ui| {
            let label = if self.done { "New shot  (M)" } else { "Measure σz now  (M)" };
            if ui.button(RichText::new(label).color(look.measure())).clicked() {
                self.measure_now(ctx);
            }
            if self.show_u && ui.button(RichText::new("Measure along u(t)").color(look.measure())).clicked() {
                self.measure_u(ctx);
            }
        });
        if !self.larmor() {
            if let Some(n) = repeat_button(ui, "ramsey_here", "Run N shots at this τ", &mut self.repeat_n) {
                self.queue_here(n);
            }
            ui.horizontal(|ui| {
                if ui.button(format!("Scan τ: {} values × N", self.scan_points)).clicked() {
                    self.queue_scan(self.repeat_n);
                }
                if ui.button("Clear").clicked() {
                    self.fringe.clear();
                    self.job = None;
                }
            });
            ui.add(egui::Slider::new(&mut self.tau_max, 40.0..=400.0).text("scan up to τ (ns)"));
            if let Some(j) = &self.job {
                ui.label(RichText::new(format!("taking shots … {} left", j.left())).color(look.counts()));
            }
            ui.label(
                RichText::new(
                    "One shot = one electron through the whole sequence, one σz outcome. Each point is the ↑ fraction of its shots; the bars are the projection noise √(p(1−p)/N). Click the plot to choose τ.",
                )
                .color(look.weak_text())
                .size(12.0 * look.font_scale()),
            );
        }

        if !hidden {
            ui.add_space(look.px(8.0));
            ui.label(RichText::new("Part c: axis u(t) in the x–y plane").strong());
            if ui.checkbox(&mut self.show_u, "show u(t) and P(↑u)").changed() {
                self.rebuild_history();
            }
        }
        if self.show_u && !hidden {
            let mut changed = false;
            segmented(ui, |ui| {
                changed |= ui.selectable_value(&mut self.u_corotating, true, "turning at ω").changed();
                changed |= ui.selectable_value(&mut self.u_corotating, false, "fixed").changed();
            });
            let mut off = self.u_offset.to_degrees();
            changed |= ui
                .add(egui::Slider::new(&mut off, -180.0..=180.0).text("azimuth offset").suffix("°"))
                .changed();
            self.u_offset = off.to_radians();
            if changed {
                self.rebuild_history();
            }
            let tp = self.physics_time(self.t);
            let phi_u = if self.u_corotating {
                self.omega() * tp + self.u_offset
            } else {
                self.u_offset
            };
            ui.label(
                RichText::new(format!(
                    "P(↑u) = cos²(φ/2 − ωt/2) = {:.3}",
                    (0.5 * phi_u - 0.5 * self.omega() * tp).cos().powi(2)
                ))
                .color(look.weak_text()),
            );
        }

        if self.variant == RamseyVariant::DetermineB {
            ui.add_space(look.px(8.0));
            ui.label(RichText::new("Determine B (part e)").strong());
            let dt = (self.cursors[1] - self.cursors[0]).abs();
            ui.horizontal(|ui| {
                ui.label(format!("cursors A–B: Δτ = {} ns =", num(dt, 1)));
                ui.add(egui::DragValue::new(&mut self.periods).range(1..=30));
                ui.label("periods");
            });
            let period = dt / self.periods.max(1) as f64;
            if period > 0.5 {
                let f = 1e3 / period;
                let b = units::H * (f * 1e6) / (2.0 * units::MU_B) * 1e3;
                ui.label(
                    RichText::new(format!(
                        "T = {} ns → f = 1/T = {} MHz → B = h f/(2μ_B) = {:.3} mT",
                        num(period, 2),
                        num(f, 2),
                        b
                    ))
                    .color(look.accent()),
                );
                if self.revealed {
                    ui.label(
                        RichText::new(format!(
                            "true B = {:.3} mT: your estimate is off by {:+.1} %",
                            self.b_mt,
                            100.0 * (b / self.b_mt - 1.0)
                        ))
                        .color(look.prediction()),
                    );
                }
            }
            ui.horizontal(|ui| {
                if !self.revealed && ui.button("Reveal B").clicked() {
                    self.revealed = true;
                }
                if ui.button("New hidden B").clicked() {
                    self.b_mt = hidden_b(ctx.rng);
                    self.revealed = false;
                    self.fringe.clear();
                    self.job = None;
                    self.new_shot();
                }
            });
            ui.label(
                RichText::new("Drag the two cursors on the fringe plot onto matching points (e.g. two maxima) and enter the number of periods between them.")
                    .color(look.weak_text())
                    .size(12.0 * look.font_scale()),
            );
        }
        if let Some(l) = self.flash.label() {
            ui.label(RichText::new(format!("last outcome: {l}")).color(look.measure()));
        }
    }

    fn notes(&self) -> String {
        let specific = match self.variant {
            RamseyVariant::Larmor => NOTES_LARMOR,
            RamseyVariant::Ramsey => NOTES_RAMSEY,
            RamseyVariant::RamseyY => NOTES_RAMSEY_Y,
            RamseyVariant::DetermineB => NOTES_DETERMINE,
        };
        format!("{specific}\n\n{NOTES_BEHIND}")
    }
}

const NOTES_LARMOR: &str = "\
# The exercise (PS13 51 b, c)
An electron in B = B z has H₀ = −μ̂·B ≈ μ_B B σz (g_s ≈ 2, Eq. 51.1) with energies ±μ_B B: |↑z⟩ (upper) and |↓z⟩ (lower) are the stationary states, split by ħω = 2μ_B B. The spin starts in |↑z⟩; at t = 0 the pulse R_y(π/2) turns it into |φ(0)⟩ = (1, 1)ᵀ/√2 = |↑x⟩ (Eq. 51.8), which then evolves freely.
- (b) e^(−iH₀t/ħ) = e^(−iωtσz/2) with ω = 2μ_B B/ħ, so |φ(t)⟩ = (e^(−iωt/2), e^(iωt/2))ᵀ/√2 (Eq. 51.5): the column vector shows exactly these amplitudes.
- (c) For u in the x–y plane at azimuth φ (Eq. 50.3a with θ = π/2): P(↑u|φ(t)) = cos²(φ/2 − ωt/2). It is 1 at all times if u turns with φ = ωt: the state is |↑u(t)⟩, its Bloch vector precesses about z with the Larmor frequency ω = 2μ_B B/ħ.
- Numbers: ω/2π = 2μ_B B/h ≈ 28.0 MHz per mT; ħω ≈ 0.116 μeV at 1 mT.

# Try
- The level ladder: each energy eigenstate only picks up a phase e^(−iEt/ħ), and the relative phase of the two is the azimuth of the Bloch vector. |c|², and so ⟨σz⟩, stay constant.
- u(t) is on: \"Measure along u(t)\" gives ↑u every time (P = 1). Switch u to \"fixed\": P(↑u) = cos²(φ/2 − ωt/2) oscillates (the P(↑u) curve).
- Press M (σz) during the precession: ↑ or ↓ with 1/2 each, and the precession stops, since |↑z⟩ and |↓z⟩ are stationary states.
- Change B: the precession speeds up in proportion.";

const NOTES_RAMSEY: &str = "\
# The exercise (PS13 51 b, d, e)
Start in |↑z⟩; at t = 0 R_y(π/2) gives |φ(0)⟩ = (1, 1)ᵀ/√2 (Eq. 51.8); free precession for τ gives |φ(τ)⟩ = (e^(−iωτ/2), e^(iωτ/2))ᵀ/√2 (Eq. 51.5); then R_x(π/2) = (1/√2)[[1, −i], [−i, 1]] and a measurement of σz.
- (d) |ϕ(τ)⟩ = R_x(π/2)|φ(τ)⟩ = ½ (e^(−iωτ/2) − i e^(iωτ/2), −i e^(−iωτ/2) + e^(iωτ/2))ᵀ and P(↑z|ϕ(τ)) = |e^(−iωτ/2) − i e^(iωτ/2)|²/4 = (1 + sin ωτ)/2 (Eq. 51.14).
- On the Bloch sphere: the first pulse turns +z to +x, the precession turns it to (cos ωτ, sin ωτ, 0), R_x(π/2) turns that to (cos ωτ, 0, sin ωτ), so ⟨σz⟩ = sin ωτ. The second pulse makes the precession angle measurable, which σz alone cannot see (P(↑z) = 1/2 throughout the free precession).
- (e) P(↑z) against τ oscillates with period 2π/ω: fitting the frequency of the fringe gives B = ħω/(2μ_B).

# Try
- Play one shot: the sequence runs along the timeline and ends with a σz measurement (collapse), which adds one outcome at this τ to the fringe plot. \"loop\" repeats it.
- \"Run N shots at this τ\" and \"Scan τ\" (N shots at each of 61 values, interleaved, so the fringe builds up shot by shot). The error bars are the projection noise √(p(1−p)/N): zero where p = 0 or 1, largest at p = 1/2.
- Click the fringe plot to choose τ, drag the playhead on the timeline.
- Press M during the free precession: σz gives 50/50 there and that atom's fringe is lost.";

const NOTES_RAMSEY_Y: &str = "\
# Second pulse about y
The same sequence with R_y(π/2) as the second pulse: |ϕ(τ)⟩ = R_y(π/2)|φ(τ)⟩ = (−i sin(ωτ/2), cos(ωτ/2))ᵀ and P(↑z|ϕ(τ)) = (1 − cos ωτ)/2 = sin²(ωτ/2). Same period, but the fringe is shifted by a quarter period (π/2 in ωτ) against (1 + sin ωτ)/2 of the exercise (dashed): the axis of the second pulse sets the phase of the fringe.
- At τ = 0 both pulses together are R_y(π), which flips |↑z⟩ to |↓z⟩: P = 0 (with an x pulse P(τ = 0) = 1/2).
- On the Bloch sphere: R_y(π/2) takes (cos ωτ, sin ωτ, 0) to (0, sin ωτ, −cos ωτ), so ⟨σz⟩ = −cos ωτ.

# Try
- Scan τ and compare the points with both curves; switch the second pulse back to x in the side panel.";

const NOTES_DETERMINE: &str = "\
# Determine B (PS13 51 e)
B is hidden (somewhere between 0.4 and 2 mT), and so are the state, the analytic curve and everything that would give ω away. You can only do what an experiment does: run the Ramsey sequence at a chosen τ and record σz outcomes.
- Scan τ (121 values, 1 ns apart) to build up the fringe P(↑z) = (1 + sin ωτ)/2.
- Put the cursors A and B on matching points of the fringe (e.g. two maxima) and enter the number of periods between them: T = Δτ/k and ω = 2π/T.
- Then B = ħω/(2μ_B) = h/(2μ_B T), i.e. B ≈ 35.7 mT·ns / T.
- Reveal B to compare. More shots per point make the points sharper; more periods between the cursors make the estimate better.";

const NOTES_BEHIND: &str = "\
# Behind the scenes
Time in ns, B in mT, ω = 2μ_B B/ħ from the CODATA constants with g_s = 2 as in Eq. 51.1 (the electron's g_s = 2.0023 would give 0.1 % more). The state is computed exactly with 2 × 2 matrices; rotations are R_u(α) = e^(−iα u·σ/2) = cos(α/2) I − i sin(α/2) u·σ (Eq. 51.3). Pulses are instantaneous in the exercise; here they are stretched over a few ns of display time, with the free precession paused meanwhile, so the scene clock equals the exercise's t during the free precession. Before t = 0 the state is e^(−iωt/2)|↑z⟩ (stationary: only its phase turns). The shots of the fringe use the same matrices, outcomes drawn with the Born rule from the seeded random generator. Shown at 15 ns per second at speed 1.";
