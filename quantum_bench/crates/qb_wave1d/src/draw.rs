//! PS10 41 · Draw your own potential.
//!
//! V(x) is drawn with the mouse on the energy-axis plot (left-drag) or picked
//! from quick shapes. For any V the scene shows the numerical bound states
//! (finite-difference eigen-solver) with the level ladder, T(E) and R(E) by
//! Numerov integration (as in the exercise) for incidence from the left, and a
//! wave packet (x₀, k₀, σ) or eigenstates evolving in it. Measurements: x and E.

use std::f64::consts::TAU;
use std::sync::Arc;

use qb_core::egui::{self, Align2, RichText, pos2};
use qb_core::num::{Absorber, C64, EigenBasis, Grid1D, Momentum, SplitStep, Wave1D, c64, measure_position, numerov_scatter};
use qb_core::widgets::{
    Domain, FlashKind, MeasureFlash, PanZoom, PlotFrame, ReprView, TimeSeries, TimeSeriesPlot, TimeWindow, WaveCurve, WavePlot, WaveStyle,
    label_box, plot_dashed, plot_line, readout_row, repeat_button, view_bar,
};
use qb_core::{Ctx, Look, Scene};

use crate::common::{self, EnergyView, LevelCurve, Repeated, grow_top, hint, section};

/// The drawable region and its control points (V is constant beyond it).
const CX_MIN: f64 = -20.0;
const CX_MAX: f64 = 20.0;
const NC: usize = 2001;
const LEVELS: usize = 14;
/// width of the level ladder beside the x plot (nominal points)
const LADDER_W: f32 = 220.0;
/// dx = 0.1 and dt ≤ 0.01: small enough kinetic phase per step for sharp drawn edges.
const DT_MAX: f64 = 0.01;

/// The three presets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawPreset {
    DoubleWell,
    DoubleBarrier,
    Step,
}

/// Quick shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Flat,
    Step,
    Barrier,
    Parabolic,
    DoubleBarrier,
    FiniteWell,
    DoubleWell,
    Ramp,
    WellBump,
}

impl Shape {
    const ALL: [Shape; 9] = [
        Shape::Step,
        Shape::Barrier,
        Shape::Parabolic,
        Shape::DoubleBarrier,
        Shape::FiniteWell,
        Shape::DoubleWell,
        Shape::Ramp,
        Shape::WellBump,
        Shape::Flat,
    ];

    fn label(self) -> &'static str {
        match self {
            Shape::Flat => "flat",
            Shape::Step => "step",
            Shape::Barrier => "barrier (41 b)",
            Shape::Parabolic => "parabolic (41 c)",
            Shape::DoubleBarrier => "double barrier",
            Shape::FiniteWell => "finite well",
            Shape::DoubleWell => "double well",
            Shape::Ramp => "ramp",
            Shape::WellBump => "well with a bump",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        Shape::ALL
            .into_iter()
            .find(|sh| sh.label().split(' ').next() == Some(s))
            .or(match s {
                "double_barrier" => Some(Shape::DoubleBarrier),
                "double_well" => Some(Shape::DoubleWell),
                "finite_well" => Some(Shape::FiniteWell),
                "bump" => Some(Shape::WellBump),
                _ => None,
            })
    }

    fn v(self, x: f64) -> f64 {
        match self {
            Shape::Flat => 0.0,
            Shape::Step => {
                if x >= 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            // PS10 41 b: V = 1 on [0, a), a = 1
            Shape::Barrier => {
                if (0.0..1.0).contains(&x) {
                    1.0
                } else {
                    0.0
                }
            }
            // PS10 41 c: V = 4(x/a − x²/a²) on [0, a], a = 1
            Shape::Parabolic => {
                if (0.0..=1.0).contains(&x) {
                    4.0 * (x - x * x)
                } else {
                    0.0
                }
            }
            Shape::DoubleBarrier => {
                if (2.0..2.8).contains(&x.abs()) {
                    1.0
                } else {
                    0.0
                }
            }
            Shape::FiniteWell => {
                if x.abs() < 3.0 {
                    -2.0
                } else {
                    0.0
                }
            }
            Shape::DoubleWell => {
                let u = x / 1.5;
                (2.0 * (u * u - 1.0).powi(2)).min(4.0)
            }
            Shape::Ramp => (1.5 * (x + 4.0) / 8.0).clamp(0.0, 1.5),
            Shape::WellBump => {
                if x.abs() < 0.6 {
                    -1.5
                } else if x.abs() < 4.0 {
                    -3.0
                } else {
                    0.0
                }
            }
        }
    }

    /// Visible x range for this shape.
    fn x_view(self) -> (f64, f64) {
        match self {
            Shape::DoubleWell => (-5.0, 5.0),
            Shape::FiniteWell | Shape::WellBump => (-9.0, 9.0),
            _ => (-30.0, 30.0),
        }
    }

    /// Energy axis for this shape.
    fn e_range(self) -> (f64, f64) {
        match self {
            Shape::Flat => (-1.0, 3.0),
            Shape::Step | Shape::Ramp => (-0.2, 2.6),
            Shape::Barrier | Shape::Parabolic | Shape::DoubleBarrier => (-0.2, 2.0),
            Shape::FiniteWell => (-2.5, 1.5),
            Shape::DoubleWell => (-0.3, 4.6),
            Shape::WellBump => (-3.5, 1.2),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Init {
    Packet,
    Level(usize),
    /// (ψ₀ + ψ₁)/√2: left-localised in a symmetric double well
    Pair,
}

/// T(E), R(E) for incidence from the left.
struct Transmission {
    es: Vec<f64>,
    t: Vec<f64>,
    r: Vec<f64>,
}

pub struct DrawScene {
    preset: DrawPreset,
    // parameters
    ctrl: Vec<f64>,
    shape: Option<Shape>,
    e_range: (f64, f64),
    x_view: (f64, f64),
    x0: f64,
    k0: f64,
    sigma: f64,
    init: Init,
    width_x: f64,
    repeat_n: usize,
    view: ReprView,
    style: WaveStyle,

    // derived
    grid: Grid1D,
    xs: Vec<f64>,
    v: Vec<f64>,
    absorb_rate: Vec<f64>,
    prop: SplitStep,
    /// bound states only (E below both asymptotic values of V)
    basis: Option<Arc<EigenBasis>>,
    labels: Vec<String>,
    trans: Option<Transmission>,

    // state
    wave: Wave1D,
    mom: Momentum,
    t: f64,
    absorbed: (f64, f64),
    series_l: TimeSeries,
    series_r: TimeSeries,
    events: Vec<(f64, String)>,
    /// two-level prediction P_left(t) = a + b cos(ΔE t) after a (ψ₀ + ψ₁)/√2 start
    pair_prediction: Option<(f64, f64, f64)>,
    repeated: Option<Repeated>,
    flash: MeasureFlash,
    top_x: f64,
    top_p: f64,
    /// last x of the pointer while drawing
    drawing: Option<f64>,
}

impl DrawScene {
    pub fn new(preset: DrawPreset) -> Self {
        let grid = Grid1D::new(-102.4, 102.4, 2048);
        let absorber = Absorber::new(15.0, 2.0);
        let wave = Wave1D::zeros(grid);
        let mom = wave.momentum();
        let (shape, init, x0, k0, sigma) = match preset {
            DrawPreset::DoubleWell => (Shape::DoubleWell, Init::Pair, -10.0, 1.0, 2.0),
            // the packet's energy on the second transmission resonance (E ≈ 0.645, found with the T(E) panel)
            DrawPreset::DoubleBarrier => (Shape::DoubleBarrier, Init::Packet, -24.0, (2.0f64 * 0.645).sqrt(), 6.0),
            DrawPreset::Step => (Shape::Step, Init::Packet, -20.0, 3f64.sqrt(), 4.0),
        };
        let mut s = DrawScene {
            preset,
            ctrl: vec![0.0; NC],
            shape: Some(shape),
            e_range: shape.e_range(),
            x_view: shape.x_view(),
            x0,
            k0,
            sigma,
            init,
            width_x: 0.3,
            repeat_n: 1000,
            view: ReprView::X,
            style: WaveStyle::Density,
            grid,
            xs: grid.xs(),
            v: vec![0.0; grid.n],
            absorb_rate: grid.sample(|x| 2.0 * absorber.profile(&grid, x)),
            prop: SplitStep::new(grid, DT_MAX).with_absorber(absorber),
            basis: None,
            labels: vec![],
            trans: None,
            wave,
            mom,
            t: 0.0,
            absorbed: (0.0, 0.0),
            series_l: TimeSeries::new("P(x < 0)"),
            series_r: TimeSeries::new("P(x > 0)"),
            events: vec![],
            pair_prediction: None,
            repeated: None,
            flash: MeasureFlash::new(),
            top_x: 0.0,
            top_p: 0.0,
            drawing: None,
        };
        s.set_shape(shape);
        s.restart();
        s
    }

    fn cx(i: usize) -> f64 {
        CX_MIN + (CX_MAX - CX_MIN) * i as f64 / (NC - 1) as f64
    }

    /// V(x) from the control points (linear, constant beyond the drawable region).
    fn v_at(ctrl: &[f64], x: f64) -> f64 {
        let f = (x - CX_MIN) / (CX_MAX - CX_MIN) * (NC - 1) as f64;
        if f <= 0.0 {
            return ctrl[0];
        }
        if f >= (NC - 1) as f64 {
            return ctrl[NC - 1];
        }
        let i = f.floor() as usize;
        let t = f - i as f64;
        ctrl[i] * (1.0 - t) + ctrl[i + 1] * t
    }

    fn set_shape(&mut self, shape: Shape) {
        self.shape = Some(shape);
        self.e_range = shape.e_range();
        self.x_view = shape.x_view();
        self.ctrl = (0..NC).map(|i| shape.v(Self::cx(i))).collect();
        self.rebuild();
    }

    /// V on the grid, the propagator, the bound states and T(E) after V changed.
    fn rebuild(&mut self) {
        let ctrl = self.ctrl.clone();
        self.v = self.xs.iter().map(|&x| Self::v_at(&ctrl, x)).collect();
        self.prop.set_potential(&self.v);
        let v_asym = ctrl[0].min(ctrl[NC - 1]);
        let all = EigenBasis::numeric(self.grid, &self.v, LEVELS);
        let bound = all.energies.iter().take_while(|&&e| e < v_asym - 1e-3).count();
        self.basis = (bound > 0).then(|| {
            Arc::new(EigenBasis {
                grid: all.grid,
                energies: all.energies[..bound].to_vec(),
                states: all.states[..bound].to_vec(),
            })
        });
        self.labels = (0..bound).map(|n| n.to_string()).collect();
        // T(E) by Numerov for E above the left asymptote, up to the top of the energy axis
        let (v_l, v_r) = (ctrl[0], ctrl[NC - 1]);
        let e_lo = v_l.max(self.e_range.0) + 1e-3;
        let e_hi = self.e_range.1;
        self.trans = (e_hi > e_lo + 0.05).then(|| {
            let n = 260;
            let es: Vec<f64> = (0..=n).map(|k| e_lo + (e_hi - e_lo) * k as f64 / n as f64).collect();
            let ctrl = &ctrl;
            let res: Vec<(f64, f64)> = qb_core::par::map(&es, |&e| {
                let s = numerov_scatter(CX_MIN - 0.5, CX_MAX + 0.5, 0.01, |x| Self::v_at(ctrl, x), e);
                (s.transmission, if e > v_r { s.reflection } else { 1.0 })
            });
            Transmission {
                es,
                t: res.iter().map(|r| r.0).collect(),
                r: res.iter().map(|r| r.1).collect(),
            }
        });
        if self.init != Init::Packet && self.basis.as_ref().is_none_or(|b| b.len() < 2 && self.init == Init::Pair) {
            self.init = Init::Packet;
        }
    }

    fn initial_wave(&self) -> Wave1D {
        match (self.init, &self.basis) {
            (Init::Level(n), Some(b)) => b.wave(n.min(b.len() - 1)),
            (Init::Pair, Some(b)) if b.len() >= 2 => {
                let mut w = b.wave(0);
                w.add_scaled(c64(1.0, 0.0), &b.wave(1));
                w.normalize();
                w
            }
            _ => Wave1D::gaussian(self.grid, self.x0, self.sigma, self.k0),
        }
    }

    fn restart(&mut self) {
        self.wave = self.initial_wave();
        self.mom = self.wave.momentum();
        self.t = 0.0;
        self.absorbed = (0.0, 0.0);
        self.series_l.clear();
        self.series_r.clear();
        self.events.clear();
        self.repeated = None;
        self.flash.clear();
        self.pair_prediction = match (self.init, &self.basis) {
            (Init::Pair, Some(b)) if b.len() >= 2 => {
                let mid = self.grid.nearest(0.0);
                let dx = self.grid.dx;
                let (s0, s1) = (&b.states[0], &b.states[1]);
                let l00: f64 = s0[..mid].iter().map(|v| v * v).sum::<f64>() * dx;
                let l11: f64 = s1[..mid].iter().map(|v| v * v).sum::<f64>() * dx;
                let l01: f64 = s0[..mid].iter().zip(&s1[..mid]).map(|(a, b)| a * b).sum::<f64>() * dx;
                Some((0.5 * (l00 + l11), l01, b.energies[1] - b.energies[0]))
            }
            _ => None,
        };
        self.record();
        self.top_x = 1.2 * common::peak_density(&self.wave.psi);
        self.top_p = 1.2 * common::peak_density(&self.mom.phi);
    }

    fn sides(&self) -> (f64, f64) {
        let g = self.grid;
        (
            self.wave.prob_between(g.x_min - 1.0, 0.0) + self.absorbed.0,
            self.wave.prob_between(0.0, g.x_max() + 1.0) + self.absorbed.1,
        )
    }

    fn record(&mut self) {
        let (l, r) = self.sides();
        self.series_l.push(self.t, l);
        self.series_r.push(self.t, r);
    }

    fn coeffs(&self) -> Vec<C64> {
        self.basis
            .as_ref()
            .map_or(vec![], |b| common::clean(common::project(&b.states, &self.wave)))
    }

    fn after_measurement(&mut self, label: String) {
        self.mom = self.wave.momentum();
        self.absorbed = (0.0, 0.0);
        self.pair_prediction = None;
        self.events.push((self.t, label));
        self.record();
    }

    fn measure_x(&mut self, ctx: &mut Ctx) {
        // the absorbed part counts as "gone": measure the part still on the grid
        self.wave.normalize();
        let x = measure_position(&mut self.wave, self.width_x, ctx.rng);
        self.flash
            .trigger(FlashKind::Position { x, width: self.width_x }, format!("x = {x:.2}"));
        self.after_measurement("x".into());
        ctx.set_status(format!("measured x = {x:.3} at t = {:.2} (detector σ = {})", self.t, self.width_x));
    }

    /// Energy: a bound level n with probability |cₙ|², or "unbound" with the rest.
    fn measure_e(&mut self, ctx: &mut Ctx) {
        let Some(b) = self.basis.clone() else {
            ctx.set_status("No bound states in this potential: the energy spectrum is continuous here.");
            return;
        };
        self.wave.normalize();
        let c = common::project(&b.states, &self.wave);
        let mut probs: Vec<f64> = c.iter().map(|z| z.norm_sqr()).collect();
        let bound: f64 = probs.iter().sum();
        probs.push((1.0 - bound).max(0.0));
        let k = ctx.rng.weighted(&probs).unwrap_or(0);
        if k < b.len() {
            let phase = if c[k].norm() > 0.0 { c[k] / c[k].norm() } else { c64(1.0, 0.0) };
            let mut w = b.wave(k);
            w.scale(phase);
            self.wave = w;
            let e = b.energies[k];
            self.flash.trigger(
                FlashKind::Level { index: k, energy: e },
                format!("E = E{} = {e:.3}", common::subscript(k)),
            );
            self.after_measurement(format!("E{}", common::subscript(k)));
            ctx.set_status(format!(
                "measured E = {e:.4} (level {k}, probability {:.3}): ψ is now ψ{} and only its phase turns",
                probs[k],
                common::subscript(k)
            ));
        } else {
            for (cn, s) in c.iter().zip(&b.states) {
                for (p, v) in self.wave.psi.iter_mut().zip(s) {
                    *p -= cn * *v;
                }
            }
            self.wave.normalize();
            self.flash.trigger(FlashKind::Other, "E above the bound levels (unbound)");
            self.after_measurement("E free".into());
            ctx.set_status(format!(
                "measured an unbound energy (probability {:.3}): the bound components are projected out",
                probs[k]
            ));
        }
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

    fn repeat_e(&mut self, n: usize, ctx: &mut Ctx) {
        let Some(b) = &self.basis else {
            ctx.set_status("No bound states: nothing discrete to histogram.");
            return;
        };
        let norm = self.wave.norm_sqr().max(1e-300);
        let mut probs: Vec<f64> = common::project(&b.states, &self.wave).iter().map(|z| z.norm_sqr() / norm).collect();
        let bound: f64 = probs.iter().sum();
        probs.push((1.0 - bound).max(0.0));
        let mut names: Vec<String> = (0..b.len()).map(|k| format!("E{}", common::subscript(k))).collect();
        names.push("free".into());
        let r = common::repeat_discrete(&probs, names, n, ctx.rng, "energy (repeat N)");
        ctx.set_status(format!(
            "{n} fresh copies measured in E (bound levels and \"free\" for the continuum)"
        ));
        self.repeated = Some(r);
    }

    /// Left-drag on the plot sets V at the pointer (between the last and the current x).
    fn handle_drawing(&mut self, resp: &qb_core::widgets::PlotResponse) {
        let r = &resp.response;
        if r.dragged_by(egui::PointerButton::Primary)
            && let Some(p) = r.interact_pointer_pos()
        {
            let (x, e) = resp.transform.from_screen(p);
            let e = e.clamp(self.e_range.0, self.e_range.1);
            let x = x.clamp(CX_MIN, CX_MAX);
            let from = self.drawing.unwrap_or(x);
            let (a, b) = (from.min(x), from.max(x));
            let ia = ((a - CX_MIN) / (CX_MAX - CX_MIN) * (NC - 1) as f64).round() as usize;
            let ib = ((b - CX_MIN) / (CX_MAX - CX_MIN) * (NC - 1) as f64).round() as usize;
            let e_from = if self.drawing.is_some() { Self::v_at(&self.ctrl, from) } else { e };
            for i in ia.min(NC - 1)..=ib.min(NC - 1) {
                // straight line from the previous pointer position
                let xi = Self::cx(i);
                let t = if b > a { (xi - from).abs() / (b - a) } else { 1.0 };
                self.ctrl[i] = e_from + (e - e_from) * t.clamp(0.0, 1.0);
            }
            self.drawing = Some(x);
            self.shape = None;
            // live: V on the grid follows (the expensive parts wait for the release)
            let ctrl = &self.ctrl;
            self.v = self.xs.iter().map(|&x| Self::v_at(ctrl, x)).collect();
        }
        if r.drag_stopped() && self.drawing.is_some() {
            self.drawing = None;
            self.rebuild();
        }
    }

    /// Advances by exactly dt, counting what the absorbing edges take on each side.
    pub fn advance(&mut self, dt: f64) {
        let n = (dt / DT_MAX).ceil().max(1.0) as usize;
        let h = dt / n as f64;
        let mid = self.grid.n / 2;
        let dx = self.grid.dx;
        for _ in 0..n {
            self.prop.step(&mut self.wave, h);
            for (i, (c, rate)) in self.wave.psi.iter().zip(&self.absorb_rate).enumerate() {
                if *rate > 0.0 {
                    let d = c.norm_sqr() * rate * dx * h;
                    if i < mid {
                        self.absorbed.0 += d;
                    } else {
                        self.absorbed.1 += d;
                    }
                }
            }
        }
        self.t += dt;
        self.mom = self.wave.momentum();
        if self.series_l.last().is_none_or(|p| self.t - p[0] >= 0.1) {
            self.record();
        }
    }

    fn plot_x(&mut self, ui: &mut egui::Ui, height: f32) {
        let look = Look::of(ui);
        grow_top(&mut self.top_x, common::peak_density(&self.wave.psi));
        let coeffs = self.coeffs();
        let basis = self.basis.clone();
        let energies: Vec<f64> = basis.as_ref().map_or(vec![], |b| b.energies.clone());
        let e_mean = self.wave.energy(&self.v);
        let (e_lo, e_hi) = self.e_range;
        let scale = common::energy_curve_scale(self.style, self.top_x, 0.3 * (e_hi - e_lo));
        let mut curves = vec![];
        if let (Init::Level(n), Some(b)) = (self.init, &basis)
            && n < b.len()
        {
            let amp = b.states[n].iter().fold(0.0f64, |m, v| m.max(v.abs()));
            curves.push(LevelCurve {
                psi: &b.states[n],
                energy: b.energies[n],
                scale: 0.08 * (e_hi - e_lo) / amp.max(1e-9),
                label: String::new(),
                color: look.accent().gamma_multiply(0.7),
                dashed: true,
            });
        }
        let selected = match self.init {
            Init::Level(n) => Some(n),
            _ => None,
        };
        let (resp, ladder) = EnergyView {
            id: "draw",
            xs: &self.xs,
            v: &self.v,
            psi: &self.wave.psi,
            style: self.style,
            baseline: e_mean.clamp(e_lo, e_hi),
            scale,
            x_range: self.x_view,
            e_range: (e_lo, e_hi),
            height,
            energies: &energies,
            coeffs: basis.as_ref().map(|_| coeffs.as_slice()),
            labels: Some(&self.labels),
            selected,
            ladder_width: look.px(LADDER_W),
            curves,
            markers: vec![],
            drawable: true,
        }
        .show(ui);
        let tr = resp.transform;
        let painter = resp.painter(ui);
        // the drawable region
        for x in [CX_MIN, CX_MAX] {
            plot_dashed(
                &painter,
                &tr,
                &[x, x],
                &[e_lo, e_hi],
                look.stroke(1.0, look.weak_text().gamma_multiply(0.5)),
                3.0,
            );
        }
        label_box(
            &painter,
            pos2(tr.rect.right() - look.px(50.0), tr.rect.top() + look.px(42.0)),
            Align2::RIGHT_TOP,
            &format!("⟨H⟩ = {e_mean:.3}"),
            look.text(),
        );
        self.flash.paint(ui, &tr, Domain::Position);
        self.handle_drawing(&resp);
        if let Some(n) = ladder.and_then(|l| l.clicked) {
            self.init = Init::Level(n);
            self.restart();
        }
        if let Some((x, e)) = resp.hover {
            resp.response.on_hover_text(format!(
                "x = {x:.2}, E = {e:.2}\nleft-drag: draw V · right-drag: pan · scroll: zoom"
            ));
        }
    }

    fn plot_p(&mut self, ui: &mut egui::Ui, height: f32) {
        grow_top(&mut self.top_p, common::peak_density(&self.mom.phi));
        let (lo, hi) = self.style.y_range(self.top_p);
        let mut plot = WavePlot::new("draw_p")
            .domain(Domain::Momentum)
            .style(self.style)
            .x_range(-5.0, 5.0)
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

    fn plot_te(&self, ui: &mut egui::Ui, width: f32, height: f32) {
        let look = Look::of(ui);
        let (e_lo, e_hi) = self.e_range;
        let e_packet = self.wave.energy(&self.v);
        PlotFrame::new("draw_te")
            .x_range(e_lo, e_hi)
            .y_range(0.0, 1.25)
            .labels("E", "T, R (Numerov)")
            .width(width)
            .height(height)
            .pan_zoom(PanZoom::X)
            .show(ui, |painter, tr| {
                let r = tr.rect;
                match &self.trans {
                    Some(tt) => {
                        plot_line(painter, tr, &tt.es, &tt.r, look.stroke(1.4, look.series(1)));
                        plot_line(painter, tr, &tt.es, &tt.t, look.stroke(2.0, look.series(0)));
                    }
                    None => {
                        painter.text(
                            r.center(),
                            Align2::CENTER_CENTER,
                            "no scattering states in this energy range",
                            look.font(12.0),
                            look.weak_text(),
                        );
                    }
                }
                if let Some(b) = &self.basis {
                    for &e in &b.energies {
                        let sx = tr.x_to_screen(e);
                        painter.line_segment(
                            [pos2(sx, r.bottom() - look.px(10.0)), pos2(sx, r.bottom())],
                            look.stroke(1.5, look.level()),
                        );
                    }
                }
                if self.init == Init::Packet && self.events.is_empty() {
                    let se = tr.x_to_screen(e_packet);
                    painter.line_segment([pos2(se, r.top()), pos2(se, r.bottom())], look.stroke(1.3, look.accent()));
                    painter.text(
                        pos2(se + 3.0, r.bottom() - look.px(14.0)),
                        Align2::LEFT_BOTTOM,
                        "packet ⟨H⟩",
                        look.font(10.5),
                        look.accent(),
                    );
                }
                let items = [("T", look.series(0)), ("R", look.series(1)), ("bound levels (ticks)", look.level())];
                let mut x = r.right() - look.px(6.0);
                for (text, col) in items.iter().rev() {
                    let g = painter.layout_no_wrap(text.to_string(), look.font(11.0), *col);
                    let rect = Align2::RIGHT_TOP.anchor_size(pos2(x, r.top() + look.px(3.0)), g.size());
                    painter.galley(rect.min, g, *col);
                    x = rect.left() - look.px(10.0);
                }
            });
    }
}

impl Scene for DrawScene {
    fn time(&self) -> f64 {
        self.t
    }

    fn time_scale(&self) -> f64 {
        4.0
    }

    fn step(&mut self, dt: f64, _ctx: &mut Ctx) {
        self.advance(dt);
    }

    fn reset(&mut self, _ctx: &mut Ctx) {
        self.restart();
    }

    fn measure_label(&self) -> Option<String> {
        Some("Measure x".into())
    }

    fn measure(&mut self, ctx: &mut Ctx) {
        self.measure_x(ctx);
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
            "measure_x" => self.measure_x(ctx),
            "measure_e" => self.measure_e(ctx),
            "repeat_x" => self.repeat_x(n, ctx),
            "repeat_e" => self.repeat_e(n, ctx),
            "shape" => {
                let Some(s) = arg.and_then(Shape::parse) else { return false };
                self.set_shape(s);
                self.restart();
            }
            "level" => {
                let Some(v) = common::num_arg(arg) else { return false };
                self.init = Init::Level(v as usize);
                self.restart();
            }
            "packet" => {
                self.init = Init::Packet;
                self.restart();
            }
            "pair" => {
                self.init = Init::Pair;
                self.restart();
            }
            "x0" | "k0" | "sigma" => {
                let Some(v) = common::num_arg(arg) else { return false };
                match cmd {
                    "x0" => self.x0 = v,
                    "k0" => self.k0 = v,
                    _ => self.sigma = v,
                }
                self.init = Init::Packet;
                self.restart();
            }
            _ => return common::view_style_action(cmd, &mut self.view, &mut self.style),
        }
        true
    }

    fn central_ui(&mut self, ui: &mut egui::Ui, _ctx: &mut Ctx) {
        let look = Look::of(ui);
        view_bar(ui, &mut self.view, &mut self.style);
        let (pl, pr) = self.sides();
        let mut items = vec![("⟨H⟩", format!("{:.3}", self.wave.energy(&self.v)), None)];
        match &self.basis {
            Some(b) => {
                items.push(("bound states", b.len().to_string(), None));
                items.push(("E₀", format!("{:.4}", b.energies[0]), Some(look.level())));
                if b.len() >= 2 {
                    let de = b.energies[1] - b.energies[0];
                    items.push(("E₁", format!("{:.4}", b.energies[1]), Some(look.level())));
                    items.push(("ΔE = E₁ − E₀", format!("{de:.4}"), Some(look.accent())));
                    items.push(("period 2π/ΔE", format!("{:.1}", TAU / de), Some(look.accent())));
                }
            }
            None => items.push(("bound states", "none".into(), None)),
        }
        items.push(("P(x < 0)", format!("{pl:.3}"), Some(look.series(0))));
        items.push(("P(x > 0)", format!("{pr:.3}"), Some(look.series(1))));
        readout_row(ui, &items);
        let bottom_h = look.px(190.0);
        let plots_h = (ui.available_height() - bottom_h - look.px(10.0)).max(180.0);
        let view = self.view;
        view.show(ui, plots_h, |ui, d, h| match d {
            Domain::Position => self.plot_x(ui, h),
            Domain::Momentum => self.plot_p(ui, h),
        });
        ui.add_space(look.px(6.0));
        let w = ui.available_width();
        let gap = look.px(8.0);
        let hist_w = if self.repeated.is_some() { (0.26 * w).max(220.0) } else { 0.0 };
        let te_w = 0.45 * (w - hist_w);
        let ts_w = w - hist_w - te_w - gap * if hist_w > 0.0 { 2.0 } else { 1.0 };
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = gap;
            self.plot_te(ui, te_w, bottom_h);
            let pred = self.pair_prediction;
            let f = move |t: f64| pred.map_or(f64::NAN, |(a, b, de)| a + b * (de * t).cos());
            let mut ts = TimeSeriesPlot::new("draw_lr")
                .series(&self.series_l)
                .series(&self.series_r)
                .window(TimeWindow::All)
                .now(self.t)
                .y_range(0.0, 1.15)
                .labels("t", "")
                .width(ts_w)
                .height(bottom_h);
            if pred.is_some() {
                ts = ts.function("two-level prediction", look.prediction(), &f);
            }
            for (t, l) in &self.events {
                ts = ts.event(*t, l.clone());
            }
            ts.show(ui);
            if let Some(r) = &self.repeated {
                r.show(ui, "draw_hist", hist_w, bottom_h);
            }
        });
    }

    fn side_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx) {
        let look = ctx.look;
        ui.label(RichText::new("Potential").strong());
        hint(
            ui,
            "Left-drag on the plot to draw V(x) (between the dotted lines; constant outside). Right-drag pans, scroll zooms, double-click resets the view.",
        );
        ui.horizontal_wrapped(|ui| {
            for s in Shape::ALL {
                if ui.selectable_label(self.shape == Some(s), s.label()).clicked() {
                    self.set_shape(s);
                    self.restart();
                }
            }
        });

        section(ui, "Start with");
        let has_pair = self.basis.as_ref().is_some_and(|b| b.len() >= 2);
        ui.horizontal(|ui| {
            if ui.selectable_label(self.init == Init::Packet, "wave packet").clicked() {
                self.init = Init::Packet;
                self.restart();
            }
            if ui
                .add_enabled(has_pair, egui::Button::selectable(self.init == Init::Pair, "(ψ₀ + ψ₁)/√2"))
                .on_hover_text("left-localised in a symmetric double well")
                .clicked()
            {
                self.init = Init::Pair;
                self.restart();
            }
        });
        hint(ui, "Click a level in the ladder to start in that bound state.");
        let mut changed = false;
        changed |= ui.add(egui::Slider::new(&mut self.x0, -40.0..=40.0).text("packet x₀")).changed();
        changed |= ui.add(egui::Slider::new(&mut self.k0, -3.0..=3.0).text("k₀")).changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.sigma, 0.3..=10.0).logarithmic(true).text("σₓ"))
            .changed();
        hint(
            ui,
            format!(
                "kinetic energy ≈ (k₀² + σₚ²)/2 = {:.3}, σₚ = 1/(2σₓ) = {:.3}",
                0.5 * (self.k0 * self.k0 + 0.25 / (self.sigma * self.sigma)),
                0.5 / self.sigma
            ),
        );
        if ui.button("Launch packet").clicked() || changed {
            self.init = Init::Packet;
            self.restart();
        }

        section(ui, "Measure");
        ui.add(
            egui::Slider::new(&mut self.width_x, 0.05..=3.0)
                .logarithmic(true)
                .text("x detector σ"),
        );
        ui.horizontal(|ui| {
            if ui.button(RichText::new("Measure x  (M)").color(look.measure())).clicked() {
                self.measure_x(ctx);
            }
            if ui
                .add_enabled(
                    self.basis.is_some(),
                    egui::Button::new(RichText::new("Measure E").color(look.measure())),
                )
                .clicked()
            {
                self.measure_e(ctx);
            }
        });
        if let Some(n) = repeat_button(ui, "draw_rep_x", "Repeat x", &mut self.repeat_n) {
            self.repeat_x(n, ctx);
        }
        if self.basis.is_some()
            && let Some(n) = repeat_button(ui, "draw_rep_e", "Repeat E", &mut self.repeat_n)
        {
            self.repeat_e(n, ctx);
        }
        hint(
            ui,
            "E: a bound level with probability |cₙ|², or \"free\" (the continuum above the bound levels, which this scene does not resolve further).",
        );
        if let Some(l) = self.flash.label() {
            ui.label(RichText::new(format!("last outcome: {l}")).color(look.measure()));
        }
        section(ui, "View");
        hint(ui, "x / p and the display mode: switches above the plot (V cycles the mode).");
    }

    fn notes(&self) -> String {
        let intro = match self.preset {
            DrawPreset::DoubleWell => {
                "# This preset: a double well\nTwo wells separated by a barrier. The lowest two states are the symmetric ψ₀ and the antisymmetric ψ₁ with a small splitting ΔE = E₁ − E₀. The start (ψ₀ + ψ₁)/√2 sits in the left well; then ψ(t) ∝ ψ₀ + e^(−iΔE t)ψ₁ moves to the right well after half a period and back: tunnelling oscillation with period 2π/ΔE (ħ = 1). The dashed curve is the two-level prediction P_left(t) = ½(∫_L ψ₀² + ∫_L ψ₁²) + (∫_L ψ₀ψ₁) cos(ΔE t), integrals over x < 0.\n- Try: a higher or wider barrier (draw it) makes ΔE exponentially smaller and the period longer.\n\n"
            }
            DrawPreset::DoubleBarrier => {
                "# This preset: resonant tunnelling\nTwo barriers (V = 1, width 0.8, 4 apart). T(E) has sharp peaks with T ≈ 1 below the barrier height: at these energies a quasi-bound state between the barriers is in resonance, the waves reflected at both barriers cancel. The packet is launched at the second peak (E ≈ 0.645). The first peak (E ≈ 0.15) is so narrow that a packet would need a far smaller energy spread (a much wider packet) to use it.\n- Try: a packet off resonance (change k₀) and compare the transmitted fraction; part of the packet stays trapped between the barriers for a while and leaks out on both sides.\n\n"
            }
            DrawPreset::Step => {
                "# This preset: a potential step\nV = 0 for x < 0, V = 1 for x > 0, packet with E ≈ 1.5 > V. Even above the step part of the packet is reflected: T(E) = 4k₁k₂/(k₁ + k₂)² with k₁ = √(2E), k₂ = √(2(E − V)). Below the step (E < V) T = 0, but the wave penetrates a little into the step.\n- Try: the ramp shape: a smooth step reflects much less at the same energies.\n\n"
            }
        };
        // the preset text below is about the preset's potential: say so when V is now something else
        let preset_shape = match self.preset {
            DrawPreset::DoubleWell => Shape::DoubleWell,
            DrawPreset::DoubleBarrier => Shape::DoubleBarrier,
            DrawPreset::Step => Shape::Step,
        };
        let current = match self.shape {
            Some(s) if s == preset_shape => String::new(),
            Some(s) => format!(
                "# Current potential: {} (quick shape)\nThe preset section below describes the {}; the exercise and the rest apply to any V.\n\n",
                s.label(),
                preset_shape.label()
            ),
            None => format!(
                "# Current potential: hand-drawn\nThe preset section below describes the {}; the exercise and the rest apply to any V.\n\n",
                preset_shape.label()
            ),
        };
        format!("{current}{intro}{NOTES}")
    }
}

const NOTES: &str = "\
# The exercise (bonus)
Implement Numerov's method for scattering on an arbitrary V(x): start with the transmitted wave C e^(iqx), C = 1, on the right, integrate ψ'' = −k(x)ψ with k = 2m(E − V)/ħ² backwards to the left, match to A e^(iqx) + B e^(−iqx) there, and get R = |B|²/|A|², T = 1/|A|² (times k_R/k_L if V differs on the two sides).
- (b) The rectangular barrier V = 1 on [0, 1] (quick shape \"barrier\"): compare with the analytic T; at E = V/2, T = 1/(1 + sinh²1) = 0.42.
- (c) The parabolic barrier 4(x/a − x²/a²) (quick shape \"parabolic\"), same height: it transmits more than the rectangle of the same height and width, because it is thinner near its top.

# Try
- Draw any V: the T(E) panel, the bound states and the ladder update when you release the mouse.
- Click a level: a bound state only turns its phase. Mix states with (ψ₀ + ψ₁)/√2 or by measuring.
- Launch packets at different k₀ and compare the transmitted fraction (P(x > 0) at late times) with T(E) near the packet's ⟨H⟩.
- Measure E: in a well you get one bound level (then nothing moves) or \"free\" (unbound part).

# Behind the scenes
V is drawn on 2001 control points in −20 < x < 20 and kept constant beyond. Bound states: finite differences on 2048 points in −102 < x < 102 (the lowest levels below both ends of V); errors O(dx²) with dx = 0.1. T(E): Numerov with dx = 0.01 across the drawn region, incoming amplitude normalised to 1. Packets: split-step propagation with absorbing edges (counted on their side in P(x < 0), P(x > 0)).";

#[cfg(test)]
mod tests {
    use super::*;

    /// (ψ₀ + ψ₁)/√2 in the double well: in the right well after half a period 2π/ΔE, back after a full one.
    #[test]
    fn double_well_tunnelling_period() {
        let mut s = DrawScene::new(DrawPreset::DoubleWell);
        let b = s.basis.clone().unwrap();
        let period = TAU / (b.energies[1] - b.energies[0]);
        assert!(period > 20.0 && period < 80.0, "period {period}");
        let e0 = s.wave.energy(&s.v);
        assert!(s.sides().0 > 0.95, "starts left: {:?}", s.sides());
        s.advance(0.5 * period);
        assert!(s.sides().1 > 0.95, "right after T/2: {:?}", s.sides());
        s.advance(0.5 * period);
        assert!(s.sides().0 > 0.95, "back after T: {:?}", s.sides());
        assert!((s.wave.energy(&s.v) - e0).abs() < 1e-3);
    }

    /// Numerov T(E) of the rectangular barrier quick shape (41 b) against Eq. 39.4, and the
    /// parabolic barrier (41 c) transmits more at the same height and width.
    #[test]
    fn numerov_transmission_of_quick_shapes() {
        let mut s = DrawScene::new(DrawPreset::Step);
        s.set_shape(Shape::Barrier);
        let tt = s.trans.as_ref().unwrap();
        for (e, t) in tt.es.iter().zip(&tt.t) {
            if *e > 0.05 && (e - 1.0).abs() > 0.02 {
                let want = qb_core::num::rect_barrier_transmission(*e, 1.0, 1.0);
                assert!((t - want).abs() < 5e-3, "E = {e}: {t} vs {want}");
            }
        }
        let rect = tt.t.clone();
        s.set_shape(Shape::Parabolic);
        let para = &s.trans.as_ref().unwrap().t;
        for (k, (a, b)) in rect.iter().zip(para).enumerate() {
            if k > 5 {
                assert!(b + 1e-9 >= *a, "parabolic transmits more");
            }
        }
        // step: T = 4k₁k₂/(k₁ + k₂)² above the step
        s.set_shape(Shape::Step);
        let tt = s.trans.as_ref().unwrap();
        for (e, t) in tt.es.iter().zip(&tt.t) {
            if *e > 1.05 {
                let (k1, k2) = ((2.0 * e).sqrt(), (2.0 * (e - 1.0)).sqrt());
                let want = 4.0 * k1 * k2 / (k1 + k2).powi(2);
                assert!((t - want).abs() < 1e-3, "step E = {e}: {t} vs {want}");
            } else if *e < 0.95 {
                assert!(*t < 1e-9);
            }
        }
    }
}
