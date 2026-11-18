//! PS10 39 · Tunnelling through a rectangular barrier.
//!
//! A Gaussian packet with mean energy E hits a barrier of height V and width a
//! (absorbing edges far away). Readouts: the reflected and transmitted
//! probabilities against time (including what the edges have absorbed on each
//! side). A second panel shows T(E) and R(E) from the stationary transfer-matrix
//! solver with the analytic formula (Eq. 39.4 and its E > V continuation), the
//! packet's energy distribution, its mean energy, and the resonances k₂a = nπ.
//! Optionally the x panel shows the stationary scattering state at energy E.
//! Measurement: "which side?" (transmitted / reflected), repeat N against the
//! Born rule and against the packet-averaged T.

use std::f64::consts::PI;

use qb_core::egui::{self, Align2, RichText, pos2};
use qb_core::num::{Absorber, C64, Grid1D, Momentum, Piecewise, SplitStep, Wave1D, c64, rect_barrier_transmission};
use qb_core::widgets::{
    Domain, FlashKind, Marker, MeasureFlash, PanZoom, PlotFrame, ReprView, TimeSeries, TimeSeriesPlot, TimeWindow, WaveCurve, WavePlot,
    WaveStyle, label_box, plot_dashed, plot_line, readout_row, repeat_button, segmented, view_bar,
};
use qb_core::{Ctx, Look, Scene};

use crate::common::{self, Repeated, grow_top, hint, section};
use crate::physics::{barrier_resonances, energy_density, packet_transmission};

/// dx = 0.1 and dt ≤ 0.01 keep p_max²·dt/2 ≈ 5: with the sharp barrier edges a larger
/// kinetic phase per step pumps spurious high-momentum noise (⟨H⟩ drifts) although
/// the transmitted fraction stays right (see the tests).
const N: usize = 4096;
const DX: f64 = 0.1;
const DT_MAX: f64 = 0.01;
const ABSORB_W: f64 = 30.0;

/// The four presets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarrierPreset {
    /// E = V/2 < V, a = 1 (the numbers of PS10 41 b)
    Tunnelling,
    /// E > V with k₂a = π: T = 1
    Resonance,
    /// E > V with k₂a = π/2: strong reflection although E > V
    OffResonance,
    /// a = 3, E = V/2: T ≈ 1 %
    Thick,
}

/// What the x panel shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shown {
    Packet,
    Stationary,
}

pub struct BarrierScene {
    preset: BarrierPreset,
    // parameters
    v0: f64,
    a: f64,
    /// mean energy ⟨E⟩ of the packet
    energy: f64,
    sigma_x: f64,
    repeat_n: usize,
    view: ReprView,
    style: WaveStyle,
    shown: Shown,

    // state
    grid: Grid1D,
    xs: Vec<f64>,
    v: Vec<f64>,
    /// absorption rate 2W(x) of the edge layers
    absorb_rate: Vec<f64>,
    wave: Wave1D,
    mom: Momentum,
    /// Φ(p) of the packet at t = 0 (free, before it meets the barrier)
    mom0: Momentum,
    prop: SplitStep,
    t: f64,
    absorbed_left: f64,
    absorbed_right: f64,
    series_r: TimeSeries,
    series_t: TimeSeries,
    events: Vec<(f64, String)>,
    /// packet-averaged transmission of the initial packet
    t_avg: f64,
    repeated: Option<Repeated>,
    flash: MeasureFlash,
    top_x: f64,
    top_p: f64,
}

impl BarrierScene {
    pub fn new(preset: BarrierPreset) -> Self {
        let (v0, a, energy, sigma_x) = match preset {
            BarrierPreset::Tunnelling => (1.0, 1.0, 0.5, 5.0),
            BarrierPreset::Resonance => (1.0, 3.0, 1.0 + 0.5 * (PI / 3.0).powi(2), 8.0),
            BarrierPreset::OffResonance => (1.0, 3.0, 1.0 + 0.5 * (PI / 6.0).powi(2), 8.0),
            BarrierPreset::Thick => (1.0, 3.0, 0.5, 5.0),
        };
        let x_min = -(N as f64 / 2.0) * DX;
        let grid = Grid1D::new(x_min, x_min + N as f64 * DX, N);
        let absorber = Absorber::new(ABSORB_W, 2.0);
        let absorb_rate = grid.sample(|x| 2.0 * absorber.profile(&grid, x));
        let wave = Wave1D::zeros(grid);
        let mom = wave.momentum();
        let mut s = BarrierScene {
            preset,
            v0,
            a,
            energy,
            sigma_x,
            repeat_n: 1000,
            view: ReprView::X,
            style: WaveStyle::Density,
            shown: Shown::Packet,
            grid,
            xs: grid.xs(),
            v: vec![0.0; grid.n],
            absorb_rate,
            wave,
            mom: mom.clone(),
            mom0: mom,
            prop: SplitStep::new(grid, DT_MAX).with_absorber(absorber),
            t: 0.0,
            absorbed_left: 0.0,
            absorbed_right: 0.0,
            series_r: TimeSeries::new("R(t)"),
            series_t: TimeSeries::new("T(t)"),
            events: vec![],
            t_avg: 0.0,
            repeated: None,
            flash: MeasureFlash::new(),
            top_x: 0.0,
            top_p: 0.0,
        };
        s.restart();
        s
    }

    fn barrier(&self) -> Piecewise {
        Piecewise::barrier(0.0, self.a, self.v0)
    }

    /// σₚ = 1/(2σₓ) and the carrier k₀ from ⟨E⟩ = (k₀² + σₚ²)/2.
    fn k0(&self) -> f64 {
        let sp = 0.5 / self.sigma_x;
        (2.0 * self.energy - sp * sp).max(1e-4).sqrt()
    }

    fn x_start(&self) -> f64 {
        -(4.0 * self.sigma_x + 12.0)
    }

    fn restart(&mut self) {
        // V sampled with the overlap of each cell with [0, a], so the barrier is exactly a wide on average
        let g = self.grid;
        self.v = self
            .xs
            .iter()
            .map(|&x| {
                let (lo, hi) = (x - 0.5 * g.dx, x + 0.5 * g.dx);
                let overlap = (hi.min(self.a) - lo.max(0.0)).max(0.0) / g.dx;
                self.v0 * overlap
            })
            .collect();
        self.prop.set_potential(&self.v);
        self.wave = Wave1D::gaussian(g, self.x_start(), self.sigma_x, self.k0());
        self.mom0 = self.wave.momentum();
        self.mom = self.mom0.clone();
        let (v0, a) = (self.v0, self.a);
        self.t_avg = packet_transmission(&self.mom0, |e| rect_barrier_transmission(e, v0, a));
        self.t = 0.0;
        self.absorbed_left = 0.0;
        self.absorbed_right = 0.0;
        self.series_r.clear();
        self.series_t.clear();
        self.events.clear();
        self.repeated = None;
        self.flash.clear();
        self.record();
        self.top_x = 1.2 * common::peak_density(&self.wave.psi);
        self.top_p = 1.2 * common::peak_density(&self.mom.phi);
    }

    /// (R, T, inside): probability left of the barrier, right of it, inside it,
    /// with what the absorbing edges have taken on each side.
    fn sides(&self) -> (f64, f64, f64) {
        let left = self.wave.prob_between(self.grid.x_min - 1.0, 0.0) + self.absorbed_left;
        let right = self.wave.prob_between(self.a, self.grid.x_max() + 1.0) + self.absorbed_right;
        let inside = self.wave.prob_between(0.0, self.a);
        (left, right, inside)
    }

    /// Probabilities of the "which side?" detector, split at the middle of the barrier.
    fn detector_probs(&self) -> (f64, f64) {
        let mid = 0.5 * self.a;
        let left = self.wave.prob_between(self.grid.x_min - 1.0, mid) + self.absorbed_left;
        let right = self.wave.prob_between(mid, self.grid.x_max() + 1.0) + self.absorbed_right;
        let s = (left + right).max(1e-300);
        (left / s, right / s)
    }

    fn record(&mut self) {
        let (r, t, _) = self.sides();
        self.series_r.push(self.t, r);
        self.series_t.push(self.t, t);
    }

    fn measure_side(&mut self, ctx: &mut Ctx) {
        let (_, p_t) = self.detector_probs();
        let transmitted = ctx.rng.bool(p_t);
        let mid = 0.5 * self.a;
        let g = self.grid;
        for (i, c) in self.wave.psi.iter_mut().enumerate() {
            if (g.x(i) > mid) != transmitted {
                *c = c64(0.0, 0.0);
            }
        }
        // renormalise: what is left on that side (on the grid and absorbed) becomes probability 1
        let (abs_kept, other) = if transmitted {
            (&mut self.absorbed_right, &mut self.absorbed_left)
        } else {
            (&mut self.absorbed_left, &mut self.absorbed_right)
        };
        *other = 0.0;
        let total = self.wave.norm_sqr() + *abs_kept;
        if total > 0.0 {
            self.wave.scale(c64(1.0 / total.sqrt(), 0.0));
            *abs_kept /= total;
        }
        self.mom = self.wave.momentum();
        let label = if transmitted { "transmitted" } else { "reflected" };
        self.flash.trigger(
            FlashKind::Region {
                a: if transmitted { mid } else { g.x_min },
                b: if transmitted { g.x_max() } else { mid },
                yes: true,
            },
            label,
        );
        self.events.push((self.t, label.into()));
        self.record();
        ctx.set_status(format!(
            "t = {:.1}: found {label} (probability was {:.3}); packet-averaged T = {:.3}, T(E) at the mean energy = {:.3}",
            self.t,
            if transmitted { p_t } else { 1.0 - p_t },
            self.t_avg,
            rect_barrier_transmission(self.energy, self.v0, self.a)
        ));
    }

    fn repeat(&mut self, n: usize, ctx: &mut Ctx) {
        let (p_r, p_t) = self.detector_probs();
        let names = vec!["reflected".to_string(), "transmitted".to_string()];
        let r = common::repeat_discrete(&[p_r, p_t], names, n, ctx.rng, "which side? (repeat N)");
        let k = r.hist.counts[1];
        ctx.set_status(format!(
            "{n} fresh copies at t = {:.1}: {k} transmitted ({:.1} %); Born rule now {:.1} %, packet-averaged T = {:.1} %",
            self.t,
            100.0 * k as f64 / n as f64,
            100.0 * p_t,
            100.0 * self.t_avg
        ));
        self.repeated = Some(r);
    }

    /// Stationary scattering state ψ_E(x) e^(−iEt) for incidence from the left at the packet's energy.
    fn stationary(&self) -> Vec<C64> {
        let s = self.barrier().scatter(self.energy);
        let phase = C64::from_polar(1.0, -self.energy * self.t);
        self.xs.iter().map(|&x| s.psi(x) * phase).collect()
    }

    fn v_axis_top(&self) -> f64 {
        1.6 * self.v0.max(self.energy)
    }

    fn plot_x(&mut self, ui: &mut egui::Ui, height: f32) {
        let look = Look::of(ui);
        let (x_lo, x_hi) = (-60.0, 60.0);
        let stationary = self.shown == Shown::Stationary;
        let psi_stat;
        let (psi, top): (&[C64], f64) = if stationary {
            psi_stat = self.stationary();
            let peak = self
                .xs
                .iter()
                .zip(&psi_stat)
                .filter(|(x, _)| **x > x_lo && **x < x_hi)
                .map(|(_, z)| z.norm_sqr())
                .fold(0.0, f64::max);
            (&psi_stat, 1.15 * peak.max(1e-9))
        } else {
            grow_top(&mut self.top_x, common::peak_density(&self.wave.psi));
            (&self.wave.psi, self.top_x)
        };
        let (lo, hi) = self.style.y_range(top);
        let v_top = self.v_axis_top();
        // E on the potential's scale
        let y_e = lo + self.energy / v_top * (hi - lo);
        let resp = WavePlot::new("barrier_x")
            .domain(Domain::Position)
            .style(self.style)
            .x_range(x_lo, x_hi)
            .y_range(lo, hi)
            .height(height)
            .potential(&self.xs, &self.v, [0.0, v_top])
            .curve(WaveCurve::new(&self.xs, psi))
            .y_label(if stationary {
                format!("{} (stationary ψ_E, incoming amplitude 1)", self.style.axis_label(Domain::Position))
            } else {
                self.style.axis_label(Domain::Position)
            })
            .show(ui);
        let tr = resp.transform;
        let painter = resp.painter(ui);
        plot_dashed(&painter, &tr, &[x_lo, x_hi], &[y_e, y_e], look.stroke(1.2, look.accent()), 6.0);
        label_box(
            &painter,
            pos2(tr.x_to_screen(self.a) + look.px(10.0), tr.y_to_screen(y_e) - look.px(3.0)),
            Align2::LEFT_BOTTOM,
            &format!("⟨E⟩ = {:.3} = {:.2} V", self.energy, self.energy / self.v0),
            look.accent(),
        );
        if !stationary {
            self.flash.paint(ui, &tr, Domain::Position);
        }
    }

    fn plot_p(&mut self, ui: &mut egui::Ui, height: f32) {
        grow_top(&mut self.top_p, common::peak_density(&self.mom.phi));
        let (lo, hi) = self.style.y_range(self.top_p);
        let w = (1.6 * self.k0()).max(3.0).ceil();
        let resp = WavePlot::new("barrier_p")
            .domain(Domain::Momentum)
            .style(self.style)
            .x_range(-w, w)
            .y_range(lo, hi)
            .height(height)
            .curve(WaveCurve::new(&self.mom.p, &self.mom.phi))
            .marker(Marker::vline(self.k0(), "k₀").color(Look::of(ui).weak_text()).dashed())
            .show(ui);
        self.flash.paint(ui, &resp.transform, Domain::Momentum);
    }

    /// T(E), R(E): transfer matrix (solid), analytic Eq. 39.4 (dashed), packet spectrum, resonances.
    fn plot_te(&self, ui: &mut egui::Ui, width: f32, height: f32) {
        let look = Look::of(ui);
        let (v0, a) = (self.v0, self.a);
        let e_max = (2.6 * v0).max(1.6 * self.energy).max(self.energy + 4.0 * self.energy_spread());
        let n = 400;
        let es: Vec<f64> = (1..=n).map(|k| e_max * k as f64 / n as f64).collect();
        let barrier = self.barrier();
        let t_num: Vec<f64> = es.iter().map(|&e| barrier.scatter(e).transmission).collect();
        let r_num: Vec<f64> = es.iter().map(|&e| barrier.scatter(e).reflection).collect();
        let t_ana: Vec<f64> = es.iter().map(|&e| rect_barrier_transmission(e, v0, a)).collect();
        let spec = energy_density(&self.mom0, &es);
        let spec_max = spec.iter().cloned().fold(0.0, f64::max).max(1e-12);
        let spec_y: Vec<f64> = spec.iter().map(|s| 0.85 * s / spec_max).collect();
        let res = barrier_resonances(v0, a, e_max);
        let t_mean = rect_barrier_transmission(self.energy, v0, a);
        PlotFrame::new("barrier_te")
            .x_range(0.0, e_max)
            .y_range(0.0, 1.28)
            .labels("E", "T, R")
            .width(width)
            .height(height)
            .pan_zoom(PanZoom::X)
            .show(ui, |painter, tr| {
                let r = tr.rect;
                // packet's energy distribution (filled, scaled)
                let mut mesh = egui::Mesh::default();
                let fill = look.accent().gamma_multiply(0.18);
                for (k, (&e, &y)) in es.iter().zip(&spec_y).enumerate() {
                    mesh.colored_vertex(tr.to_screen(e, y), fill);
                    mesh.colored_vertex(tr.to_screen(e, 0.0), fill);
                    if k > 0 {
                        let i = 2 * k as u32;
                        mesh.add_triangle(i - 2, i - 1, i);
                        mesh.add_triangle(i - 1, i, i + 1);
                    }
                }
                painter.add(egui::Shape::mesh(mesh));
                // E = V
                let sv = tr.x_to_screen(v0);
                painter.line_segment([pos2(sv, r.top()), pos2(sv, r.bottom())], look.stroke(1.2, look.potential()));
                painter.text(
                    pos2(sv + 3.0, r.bottom() - 3.0),
                    Align2::LEFT_BOTTOM,
                    "E = V",
                    look.font(11.0),
                    look.potential(),
                );
                // resonances k₂a = nπ
                for (k, &e) in res.iter().enumerate() {
                    plot_dashed(painter, tr, &[e, e], &[0.0, 1.0], look.stroke(1.0, look.series(2)), 4.0);
                    painter.text(
                        tr.to_screen(e, 1.0) + egui::vec2(2.0, -1.0),
                        Align2::LEFT_BOTTOM,
                        if k == 0 {
                            "k₂a = π".to_string()
                        } else {
                            format!("k₂a = {}π", k + 1)
                        },
                        look.font(10.5),
                        look.series(2),
                    );
                }
                plot_line(painter, tr, &es, &r_num, look.stroke(1.5, look.series(1)));
                plot_line(painter, tr, &es, &t_num, look.stroke(2.0, look.series(0)));
                plot_dashed(painter, tr, &es, &t_ana, look.stroke(1.4, look.prediction()), 7.0);
                // the packet's mean energy
                let se = tr.x_to_screen(self.energy);
                painter.line_segment([pos2(se, r.top()), pos2(se, r.bottom())], look.stroke(1.5, look.accent()));
                painter.circle_filled(tr.to_screen(self.energy, t_mean), look.px(4.0), look.accent());
                label_box(
                    painter,
                    tr.to_screen(self.energy, t_mean.clamp(0.15, 0.8)) + egui::vec2(look.px(8.0), 0.0),
                    Align2::LEFT_CENTER,
                    &format!("T(⟨E⟩) = {t_mean:.3}"),
                    look.accent(),
                );
                // legend: one row along the top, right-aligned
                let items = [
                    ("T", look.series(0)),
                    ("Eq. 39.4 (dashed)", look.prediction()),
                    ("R", look.series(1)),
                    ("packet's energies", look.accent()),
                ];
                let mut x = r.right() - look.px(6.0);
                for (text, col) in items.iter().rev() {
                    let g = painter.layout_no_wrap(text.to_string(), look.font(11.0), *col);
                    let rect = Align2::RIGHT_TOP.anchor_size(pos2(x, r.top() + look.px(3.0)), g.size());
                    painter.galley(rect.min, g, *col);
                    x = rect.left() - look.px(10.0);
                }
            });
    }

    /// Advances by exactly dt, counting what the absorbing edges take on each side.
    pub fn advance(&mut self, dt: f64) {
        let n = (dt / DT_MAX).ceil().max(1.0) as usize;
        let h = dt / n as f64;
        let mid = self.grid.n / 2;
        for _ in 0..n {
            self.prop.step(&mut self.wave, h);
            // what the edge layers absorb in this step, by side
            let dx = self.grid.dx;
            let mut l = 0.0;
            let mut r = 0.0;
            for (i, (c, rate)) in self.wave.psi.iter().zip(&self.absorb_rate).enumerate() {
                if *rate > 0.0 {
                    let d = c.norm_sqr() * rate * dx * h;
                    if i < mid {
                        l += d;
                    } else {
                        r += d;
                    }
                }
            }
            self.absorbed_left += l;
            self.absorbed_right += r;
        }
        self.t += dt;
        self.mom = self.wave.momentum();
        if self.series_r.last().is_none_or(|p| self.t - p[0] >= 0.2) {
            self.record();
        }
    }

    /// Standard deviation of the packet's kinetic energy, ≈ k₀σₚ.
    fn energy_spread(&self) -> f64 {
        let sp = 0.5 / self.sigma_x;
        (self.k0() * self.k0() * sp * sp + 0.5 * sp.powi(4)).sqrt()
    }
}

impl Scene for BarrierScene {
    fn time(&self) -> f64 {
        self.t
    }

    fn time_scale(&self) -> f64 {
        6.0
    }

    fn step(&mut self, dt: f64, _ctx: &mut Ctx) {
        self.advance(dt);
    }

    fn reset(&mut self, _ctx: &mut Ctx) {
        self.restart();
    }

    fn measure_label(&self) -> Option<String> {
        Some("Which side?".into())
    }

    fn measure(&mut self, ctx: &mut Ctx) {
        self.measure_side(ctx);
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
            "measure" | "which_side" => self.measure_side(ctx),
            "repeat" => self.repeat(n, ctx),
            "stationary" => self.shown = Shown::Stationary,
            "packet" => self.shown = Shown::Packet,
            "v" | "a" | "e" | "sigma" => {
                let Some(v) = common::num_arg(arg) else { return false };
                match cmd {
                    "v" => self.v0 = v,
                    "a" => self.a = v,
                    "e" => self.energy = v,
                    _ => self.sigma_x = v,
                }
                self.restart();
            }
            _ => return common::view_style_action(cmd, &mut self.view, &mut self.style),
        }
        true
    }

    fn central_ui(&mut self, ui: &mut egui::Ui, _ctx: &mut Ctx) {
        let look = Look::of(ui);
        view_bar(ui, &mut self.view, &mut self.style);
        let (r, t, inside) = self.sides();
        let t_e = rect_barrier_transmission(self.energy, self.v0, self.a);
        readout_row(
            ui,
            &[
                ("⟨E⟩/V", format!("{:.3}", self.energy / self.v0), Some(look.accent())),
                ("T(t)", format!("{t:.3}"), Some(look.series(0))),
                ("R(t)", format!("{r:.3}"), Some(look.series(1))),
                ("in barrier", format!("{inside:.3}"), None),
                ("T(⟨E⟩)", format!("{t_e:.4}"), Some(look.prediction())),
                ("⟨T⟩ packet", format!("{:.4}", self.t_avg), Some(look.prediction())),
                ("R + T + inside", format!("{:.3}", r + t + inside), None),
            ],
        );
        let bottom_h = look.px(210.0);
        let plots_h = (ui.available_height() - bottom_h - look.px(10.0)).max(180.0);
        let view = self.view;
        view.show(ui, plots_h, |ui, d, h| match d {
            Domain::Position => self.plot_x(ui, h),
            Domain::Momentum => self.plot_p(ui, h),
        });
        ui.add_space(look.px(6.0));
        let w = ui.available_width();
        let gap = look.px(8.0);
        let hist_w = if self.repeated.is_some() { (0.22 * w).max(200.0) } else { 0.0 };
        let te_w = 0.46 * (w - hist_w);
        let ts_w = w - hist_w - te_w - gap * if hist_w > 0.0 { 2.0 } else { 1.0 };
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = gap;
            self.plot_te(ui, te_w, bottom_h);
            let t_avg = self.t_avg;
            let f_avg = move |_t: f64| t_avg;
            // a small T would be invisible next to R ≈ 1: then plot T(t) alone, zoomed
            let small = self.t_avg < 0.08;
            let mut ts = TimeSeriesPlot::new(("barrier_rt", small)).series(&self.series_t);
            if !small {
                ts = ts.series(&self.series_r);
            }
            ts = ts
                .function("⟨T⟩ packet", look.prediction(), &f_avg)
                .window(TimeWindow::All)
                .now(self.t)
                .y_range(0.0, if small { 2.5 * t_avg.max(1e-6) } else { 1.15 })
                .labels("t", if small { "T(t), zoomed (R ≈ 1 − T)" } else { "" })
                .width(ts_w)
                .height(bottom_h);
            for (t, l) in &self.events {
                ts = ts.event(*t, l.clone());
            }
            ts.show(ui);
            if let Some(rep) = &self.repeated {
                rep.show(ui, "barrier_hist", hist_w, bottom_h);
            }
        });
    }

    fn side_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx) {
        let look = ctx.look;
        ui.label(RichText::new("Barrier").strong());
        let mut changed = false;
        changed |= ui.add(egui::Slider::new(&mut self.v0, 0.2..=3.0).text("height V")).changed();
        changed |= ui.add(egui::Slider::new(&mut self.a, 0.2..=5.0).text("width a")).changed();
        section(ui, "Packet");
        let e_max = 4.0 * self.v0;
        changed |= ui
            .add(egui::Slider::new(&mut self.energy, 0.05..=e_max).text("mean energy ⟨E⟩"))
            .changed();
        changed |= ui.add(egui::Slider::new(&mut self.sigma_x, 1.5..=15.0).text("width σₓ")).changed();
        hint(
            ui,
            format!(
                "k₀ = {:.3}, σₚ = 1/(2σₓ) = {:.3}, energy spread σ_E ≈ k₀σₚ = {:.3}",
                self.k0(),
                0.5 / self.sigma_x,
                self.energy_spread()
            ),
        );
        if changed {
            self.restart();
        }
        ui.horizontal(|ui| {
            if ui.button("E at resonance k₂a = π").clicked() {
                self.energy = self.v0 + 0.5 * (PI / self.a).powi(2);
                self.restart();
            }
            if ui.button("E = V/2").clicked() {
                self.energy = 0.5 * self.v0;
                self.restart();
            }
        });

        section(ui, "x panel shows");
        segmented(ui, |ui| {
            if ui.selectable_label(self.shown == Shown::Packet, "wave packet").clicked() {
                self.shown = Shown::Packet;
            }
            if ui
                .selectable_label(self.shown == Shown::Stationary, "stationary state at ⟨E⟩")
                .clicked()
            {
                self.shown = Shown::Stationary;
                if self.style == WaveStyle::Density {
                    self.style = WaveStyle::Real;
                }
            }
        });
        if self.shown == Shown::Stationary {
            hint(
                ui,
                "ψ_E e^(−iEt) with e^(ik₁x) incoming: left a standing-wave pattern (incoming + reflected), inside a decaying (E < V) or oscillating part, right a pure transmitted wave of amplitude |t| = √T.",
            );
        }

        section(ui, "Measure");
        if ui.button(RichText::new("Which side?  (M)").color(look.measure())).clicked() {
            self.measure_side(ctx);
        }
        if let Some(n) = repeat_button(ui, "barrier_rep", "Repeat", &mut self.repeat_n) {
            self.repeat(n, ctx);
        }
        hint(
            ui,
            "A detector on each side of the barrier (split at its middle; what the edges absorbed counts on its side). Measure after the packet has split: the outcome collapses ψ onto one side. Repeat N: fresh copies, compared with the Born rule now and with the packet-averaged T.",
        );
        if let Some(l) = self.flash.label() {
            ui.label(RichText::new(format!("last outcome: {l}")).color(look.measure()));
        }
        section(ui, "View");
        hint(
            ui,
            "x / p and the display mode: switches above the plot (V cycles the mode). In p, the reflected part appears at −k₀.",
        );
    }

    fn notes(&self) -> String {
        let intro = match self.preset {
            BarrierPreset::Tunnelling => {
                "# This preset\nE = V/2 below the barrier, a = 1 (the numbers of PS10 41 b): T(E) = 1/(1 + sinh²(κa)) = 0.42 with κ = √(2m(V − E))/ħ = 1. Classically the particle would always be reflected.\n\n"
            }
            BarrierPreset::Resonance => {
                "# This preset\nE > V with k₂a = π (k₂ = √(2m(E − V))/ħ): sin(k₂a) = 0, so T = 1: no reflection at all (part d). The waves reflected at the two edges cancel.\n\n"
            }
            BarrierPreset::OffResonance => {
                "# This preset\nE > V but k₂a = π/2, half-way to the first resonance: T = 0.38, more than half of the packet is reflected although classically it would pass with certainty.\n\n"
            }
            BarrierPreset::Thick => {
                "# This preset\nE = V/2 and a thick barrier a = 3: T ≈ 16E(V − E)/V² · e^(−2κa) is about 1 %. Here the energy spread matters most (see below).\n\n"
            }
        };
        format!("{intro}{NOTES}")
    }
}

const NOTES: &str = "\
# The exercise
A particle from the left meets V(x) = V for 0 < x < a (units ħ = m = 1).
- (a) Region III has no wave coming from the right, so B_III = 0. In each region ψ = A e^(ikx) + B e^(−ikx) with k₁ = √(2mE)/ħ outside and k₂ = √(2m(E − V))/ħ inside (imaginary for E < V: a decaying exponential).
- (b, c) Matching ψ and ψ' at 0 and a: T = |A_III/A_I|² = 1/(1 + V² sinh²(k₂'a)/(4E(V − E))) with k₂' = √(2m(V − E))/ħ, and R = 1 − T.
- (d) For E < V, sinh(k₂'a) > 0: no transmission resonance. For E > V the sinh becomes a sine, T = 1/(1 + V² sin²(k₂a)/(4E(E − V))), and T = 1 whenever k₂a = nπ (dashed lines in the T(E) panel).

# Try
- Play: part of the packet appears beyond the barrier, part goes back. R(t) and T(t) settle at their final values.
- Compare the final T(t) with T(⟨E⟩) and with ⟨T⟩ packet (dashed line in the time plot).
- Switch the x panel to the stationary state at ⟨E⟩ in Re ψ: the standing wave on the left, the decaying part inside, the transmitted wave on the right.
- Show p: the reflected part shows up at p ≈ −k₀.
- Make σₓ larger: the energy spread shrinks and the packet's T approaches T(⟨E⟩).
- Which side? (M) after the split, then Repeat N: the fractions scatter around the Born-rule value (shaded band: ±1σ for N shots).

# Energy spread (why the packet's T differs from T(⟨E⟩))
A packet is not an energy eigenstate: it is a superposition of plane waves with a spread of energies σ_E ≈ ħk₀σₚ/m (the shaded curve in the T(E) panel). Each energy component is transmitted with its own T(E), so the transmitted fraction is the average ⟨T⟩ = ∫|Φ(p)|² T(p²/2m) dp over the packet, not T at the mean energy. Below the barrier T(E) grows roughly exponentially with E, so the faster components dominate and ⟨T⟩ > T(⟨E⟩) (largest effect for the thick barrier); near a resonance the average is a little below 1. This is an effect of the packet, not a correction to Eq. 39.4, which is exact for a plane wave.

# Behind the scenes
Split-step propagation on 4096 points (dx = 0.1, |x| < 205, time step 0.01) with absorbing layers at the ends; what they absorb is counted on its side, so R + T + inside stays 1. The barrier is sampled with the overlap of each grid cell, so its width is exactly a. The T(E) panel uses the exact transfer-matrix solution of the stationary problem (solid) and the formula of the exercise (dashed); they lie on top of each other.";

#[cfg(test)]
mod tests {
    use super::*;

    /// The packet's final transmitted fraction is the packet-averaged T (not T at ⟨E⟩),
    /// ⟨H⟩ is conserved, and R + T + inside = 1 including what the edges absorbed.
    #[test]
    fn packet_transmission_matches_energy_average() {
        for preset in [
            BarrierPreset::Tunnelling,
            BarrierPreset::Resonance,
            BarrierPreset::OffResonance,
            BarrierPreset::Thick,
        ] {
            let mut s = BarrierScene::new(preset);
            let e0 = s.wave.energy(&s.v);
            let t_end = 2.0 * s.x_start().abs() / s.k0() + 25.0;
            s.advance(t_end);
            let (r, t, inside) = s.sides();
            assert!(
                (r + t + inside - 1.0).abs() < 2e-3,
                "{preset:?}: R + T + inside = {}",
                r + t + inside
            );
            assert!(
                (t - s.t_avg).abs() < 0.01 * s.t_avg.max(0.05),
                "{preset:?}: packet T = {t}, ⟨T⟩ = {}",
                s.t_avg
            );
            let e1 = s.wave.energy(&s.v);
            // (the part absorbed at the edges is gone; what is left must not have heated up)
            assert!(e1 < e0 * 1.3 + 0.05, "{preset:?}: ⟨H⟩ {e0} → {e1}");
        }
        // the energy spread makes a difference for the thick barrier: ⟨T⟩ > T(⟨E⟩)
        let s = BarrierScene::new(BarrierPreset::Thick);
        let t_mean = rect_barrier_transmission(s.energy, s.v0, s.a);
        assert!(s.t_avg > 1.1 * t_mean, "⟨T⟩ = {}, T(⟨E⟩) = {t_mean}", s.t_avg);
        // and the presets are what their names say
        let s = BarrierScene::new(BarrierPreset::Resonance);
        assert!((rect_barrier_transmission(s.energy, s.v0, s.a) - 1.0).abs() < 1e-12);
        let s = BarrierScene::new(BarrierPreset::Tunnelling);
        assert!((rect_barrier_transmission(s.energy, s.v0, s.a) - 1.0 / (1.0 + 1f64.sinh().powi(2))).abs() < 1e-12);
    }

    /// Transfer matrix (core) and Eq. 39.4 agree for E < V and E > V.
    #[test]
    fn transfer_matrix_matches_formula() {
        let b = Piecewise::barrier(0.0, 1.7, 1.3);
        for k in 1..60 {
            let e = 0.05 * k as f64;
            if (e - 1.3).abs() < 1e-6 {
                continue;
            }
            let s = b.scatter(e);
            assert!((s.transmission - rect_barrier_transmission(e, 1.3, 1.7)).abs() < 1e-9, "E = {e}");
            assert!((s.transmission + s.reflection - 1.0).abs() < 1e-9);
        }
    }
}
