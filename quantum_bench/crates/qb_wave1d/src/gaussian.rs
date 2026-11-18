//! PS10 38 · The Gaussian wave packet: the reference scene.
//!
//! A free Gaussian Ψ(x, 0) = A e^(−ax²) (optionally with momentum p₀) evolves
//! with the split-step propagator. The scene shows ψ(x) and/or Φ(p) with
//! phase colours, ⟨x⟩ and ±σ markers, live σₓ, σₚ, σₓσₚ with their time
//! series against the free-particle prediction, and position / momentum
//! measurements (single shot with collapse, and "repeat N" histograms).
//!
//! It is the template for the other 1D scenes: parameters → `restart()` →
//! `step()` advances the propagator → `central_ui()` draws with the qb_core
//! widgets (view bar with the x/p switch and the display modes on top, key V
//! cycles the modes) → `side_ui()` holds the controls.
//!
//! Two presets use it: the exercise (packet at rest, |ψ|² + phase) and a
//! moving packet k₀ = 3 shown as Re ψ, where phase and group velocity differ.

use qb_core::egui::{self, RichText};
use qb_core::num::{
    Absorber, Grid1D, Moments, ShotHistogram, SplitStep, Wave1D, born_samples_p, born_samples_x, measure_momentum, measure_position,
};
use qb_core::widgets::{
    Domain, FlashKind, HistScale, HistogramPlot, Marker, MeasureFlash, ReprView, TimeSeries, TimeSeriesPlot, TimeWindow, WaveCurve,
    WavePlot, WaveStyle, fmt_num, readout_row, repeat_button, view_bar,
};
use qb_core::{Ctx, Look, Scene};

/// Grid: wide enough that the packet spreads for a while before it reaches the absorbing edges.
const X_MIN: f64 = -60.0;
const X_MAX: f64 = 60.0;
const N: usize = 2048;
/// largest split-step time step
const DT_MAX: f64 = 0.02;
/// time between recorded points of the time series
const SAMPLE_DT: f64 = 0.05;

/// The last "repeat N" run.
struct Repeated {
    hist: ShotHistogram,
    /// Born-rule probability of each bin (for the state that was measured)
    predicted: Vec<f64>,
    domain: Domain,
}

/// Which preset the scene was built for (sets defaults and notes).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Variant {
    /// PS10 38 as posed: Ψ(x, 0) = A e^(−ax²), at rest
    Exercise,
    /// the same packet with a carrier k₀ = p₀ ≠ 0, shown as Re ψ
    Moving,
}

pub struct GaussianScene {
    variant: Variant,
    // parameters
    /// σₓ at t = 0 (the exercise's a = 1/(4σ₀²))
    sigma0: f64,
    /// carrier wave number k₀ = p₀ (ħ = 1)
    p0: f64,
    x0: f64,
    /// detector resolutions (σ of the collapse window) in x and in p
    width_x: f64,
    width_p: f64,
    repeat_n: usize,
    view: ReprView,
    style: WaveStyle,
    markers: bool,

    // state
    grid: Grid1D,
    xs: Vec<f64>,
    wave: Wave1D,
    prop: SplitStep,
    t: f64,
    moments: Moments,
    /// moments and time from which the dashed free-particle prediction of σₓ starts
    anchor: (f64, Moments),
    sigma_x: TimeSeries,
    sigma_p: TimeSeries,
    product: TimeSeries,
    events: Vec<(f64, String)>,
    repeated: Option<Repeated>,
    flash: MeasureFlash,
    /// y-axis tops (only grow, so that the flattening of |ψ|² stays visible)
    top_x: f64,
    top_p: f64,
}

impl GaussianScene {
    /// PS10 38 as posed: a packet at rest with σ₀ = 1 (a = 1/4), shown as |ψ|² + phase.
    pub fn new() -> Self {
        Self::build(Variant::Exercise)
    }

    /// A packet with carrier k₀ = 3 starting at x₀ = −15, shown as Re ψ:
    /// the crests move at v_phase = k₀/2, the envelope at v_group = k₀.
    pub fn moving() -> Self {
        Self::build(Variant::Moving)
    }

    fn build(variant: Variant) -> Self {
        let grid = Grid1D::new(X_MIN, X_MAX, N);
        let moving = variant == Variant::Moving;
        let mut s = GaussianScene {
            variant,
            sigma0: if moving { 2.0 } else { 1.0 },
            p0: if moving { 3.0 } else { 0.0 },
            x0: if moving { -15.0 } else { 0.0 },
            width_x: 0.3,
            width_p: 0.1,
            repeat_n: 1000,
            view: ReprView::X,
            style: if moving { WaveStyle::Real } else { WaveStyle::Density },
            markers: !moving,
            grid,
            xs: grid.xs(),
            wave: Wave1D::zeros(grid),
            prop: SplitStep::new(grid, DT_MAX).with_absorber(Absorber::new(8.0, 3.0)),
            t: 0.0,
            moments: Moments::default(),
            anchor: (0.0, Moments::default()),
            sigma_x: TimeSeries::new("σₓ"),
            sigma_p: TimeSeries::new("σₚ"),
            product: TimeSeries::new("σₓσₚ"),
            events: vec![],
            repeated: None,
            flash: MeasureFlash::new(),
            top_x: 0.0,
            top_p: 0.0,
        };
        s.restart();
        s
    }

    /// Back to t = 0 with the current parameters.
    fn restart(&mut self) {
        self.wave = Wave1D::gaussian(self.grid, self.x0, self.sigma0, self.p0);
        self.t = 0.0;
        self.moments = self.wave.moments();
        self.anchor = (0.0, self.moments);
        self.events.clear();
        self.repeated = None;
        self.flash.clear();
        for s in [&mut self.sigma_x, &mut self.sigma_p, &mut self.product] {
            s.clear();
        }
        self.record();
        // peak of |ψ|² = 1/(√(2π)σ₀), peak of |Φ|² = 1/(√(2π)σₚ) with σₚ = 1/(2σ₀)
        let tau = std::f64::consts::TAU.sqrt();
        self.top_x = 1.2 / (tau * self.sigma0);
        self.top_p = 1.2 * 2.0 * self.sigma0 / tau;
    }

    fn record(&mut self) {
        let m = &self.moments;
        self.sigma_x.push(self.t, m.sigma_x);
        self.sigma_p.push(self.t, m.sigma_p);
        self.product.push(self.t, m.uncertainty_product());
    }

    /// After a collapse: new moments, new start of the prediction, an event line.
    fn after_measurement(&mut self, label: &str) {
        self.moments = self.wave.moments();
        self.anchor = (self.t, self.moments);
        self.events.push((self.t, label.into()));
        self.record();
    }

    fn measure_x(&mut self, ctx: &mut Ctx) {
        let x = measure_position(&mut self.wave, self.width_x, ctx.rng);
        self.flash
            .trigger(FlashKind::Position { x, width: self.width_x }, format!("x = {x:.2}"));
        self.after_measurement("x");
        ctx.set_status(format!("measured x = {x:.3} at t = {:.2} (detector σ = {})", self.t, self.width_x));
    }

    fn measure_p(&mut self, ctx: &mut Ctx) {
        let p = measure_momentum(&mut self.wave, self.width_p, ctx.rng);
        self.flash
            .trigger(FlashKind::Momentum { p, width: self.width_p }, format!("p = {p:.2}"));
        self.after_measurement("p");
        if !self.view.shows_p() {
            self.view = ReprView::Both;
        }
        ctx.set_status(format!("measured p = {p:.3} at t = {:.2} (detector σ = {})", self.t, self.width_p));
    }

    /// "Repeat N": measure N fresh copies of the current state (ψ is not changed).
    fn repeat(&mut self, domain: Domain, n: usize, ctx: &mut Ctx) {
        let m = self.moments;
        let (center, sigma) = match domain {
            Domain::Position => (m.x, m.sigma_x),
            Domain::Momentum => (m.p, m.sigma_p),
        };
        let mut hist = ShotHistogram::binned(center - 4.0 * sigma, center + 4.0 * sigma, 40);
        let predicted = match domain {
            Domain::Position => {
                hist.add_all(&born_samples_x(&self.wave, n, ctx.rng));
                hist.predict_from_density(&self.xs, &self.wave.density())
            }
            Domain::Momentum => {
                hist.add_all(&born_samples_p(&self.wave, n, ctx.rng));
                let mom = self.wave.momentum();
                hist.predict_from_density(&mom.p, &mom.density())
            }
        };
        ctx.set_status(format!(
            "{n} copies measured in {}: mean {:.3} (⟨{0}⟩ = {center:.3}), spread {:.3} (σ = {sigma:.3})",
            domain.var(),
            hist.mean(),
            hist.std()
        ));
        self.repeated = Some(Repeated { hist, predicted, domain });
    }

    fn plot(&mut self, ui: &mut egui::Ui, domain: Domain, height: f32) {
        let look = Look::of(ui);
        let m = self.moments;
        let (xs, psi, center, sigma) = match domain {
            Domain::Position => (self.xs.clone(), self.wave.psi.clone(), m.x, m.sigma_x),
            Domain::Momentum => {
                let mom = self.wave.momentum();
                (mom.p, mom.phi, m.p, m.sigma_p)
            }
        };
        // y range: fixed per start, grows if the peak would be clipped
        let peak = psi.iter().map(|z| z.norm_sqr()).fold(0.0, f64::max);
        let top = match domain {
            Domain::Position => &mut self.top_x,
            Domain::Momentum => &mut self.top_p,
        };
        if peak * 1.05 > *top {
            *top = peak * 1.2;
        }
        // one fixed scale per start for all display modes (no autoscale jitter)
        let (lo, hi) = self.style.y_range(*top);
        let (x_lo, x_hi) = match domain {
            Domain::Position => (-25.0, 25.0),
            Domain::Momentum => {
                let w = (self.p0.abs() + 6.0 / (2.0 * self.sigma0)).max(6.0).ceil();
                (-w, w)
            }
        };
        let v = domain.var();
        let mut plot = WavePlot::new(("gauss", domain))
            .domain(domain)
            .style(self.style)
            .x_range(x_lo, x_hi)
            .y_range(lo, hi)
            .height(height)
            .curve(WaveCurve::new(&xs, &psi));
        if self.markers {
            let y_bracket = if self.style.is_quadratic() { 0.62 * hi } else { 0.8 * hi };
            plot = plot.marker(Marker::vline(center, format!("⟨{v}⟩"))).marker(
                Marker::bracket(
                    center - sigma,
                    center + sigma,
                    y_bracket,
                    format!("2σ{} = {:.2}", sub(v), 2.0 * sigma),
                )
                .color(look.text()),
            );
        }
        let resp = plot.show(ui);
        self.flash.paint(ui, &resp.transform, domain);
        if let Some((x, _)) = resp.hover {
            resp.response.on_hover_text(format!("{v} = {x:.2}"));
        }
    }
}

impl Default for GaussianScene {
    fn default() -> Self {
        Self::new()
    }
}

/// "x" → "ₓ", "p" → "ₚ"
fn sub(v: &str) -> &'static str {
    if v == "x" { "ₓ" } else { "ₚ" }
}

impl Scene for GaussianScene {
    fn time(&self) -> f64 {
        self.t
    }

    fn step(&mut self, dt: f64, _ctx: &mut Ctx) {
        self.prop.advance(&mut self.wave, dt);
        self.t += dt;
        self.moments = self.wave.moments();
        if self.sigma_x.last().is_none_or(|p| self.t - p[0] >= SAMPLE_DT) {
            self.record();
        }
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
        // V: the display mode, as in every 1D scene
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
            "measure_p" => self.measure_p(ctx),
            "repeat_x" => self.repeat(Domain::Position, n, ctx),
            "repeat_p" => self.repeat(Domain::Momentum, n, ctx),
            "view_x" | "view_p" | "view_both" => self.view = ReprView::parse(&cmd[5..]).unwrap_or_default(),
            _ if cmd.starts_with("style_") => match WaveStyle::parse(&cmd[6..]) {
                Some(st) => self.style = st,
                None => return false,
            },
            "p0" | "sigma0" => {
                let Some(v) = arg.and_then(|w| w.parse::<f64>().ok()) else {
                    return false;
                };
                if cmd == "p0" {
                    self.p0 = v
                } else {
                    self.sigma0 = v
                }
                self.restart();
            }
            _ => return false,
        }
        true
    }

    fn central_ui(&mut self, ui: &mut egui::Ui, _ctx: &mut Ctx) {
        let look = Look::of(ui);
        view_bar(ui, &mut self.view, &mut self.style);
        let m = self.moments;
        let near_limit = m.uncertainty_product() < 0.5 * 1.02;
        readout_row(
            ui,
            &[
                ("⟨x⟩", fmt_num(m.x, 2), None),
                ("⟨p⟩", fmt_num(m.p, 2), None),
                ("σₓ", fmt_num(m.sigma_x, 3), Some(look.series(0))),
                ("σₚ", fmt_num(m.sigma_p, 3), Some(look.series(1))),
                (
                    "σₓσₚ",
                    format!("{:.3} {}", m.uncertainty_product(), if near_limit { "= ħ/2" } else { "≥ ħ/2" }),
                    Some(look.series(2)),
                ),
                ("∫|ψ|²", format!("{:.3}", m.norm), None),
            ],
        );
        let bottom_h = look.px(170.0);
        let plots_h = (ui.available_height() - bottom_h - look.px(10.0)).max(160.0);
        let view = self.view;
        view.show(ui, plots_h, |ui, d, h| self.plot(ui, d, h));
        ui.add_space(look.px(6.0));

        // time series and the last histogram, side by side
        let w = ui.available_width();
        let hist_w = if self.repeated.is_some() { (w * 0.42).max(260.0) } else { 0.0 };
        ui.horizontal(|ui| {
            let (t_a, anchor) = self.anchor;
            let prediction = move |t: f64| if t >= t_a { anchor.free_sigma_x(t - t_a) } else { f64::NAN };
            let mut ts = TimeSeriesPlot::new("gauss_sigma")
                .series(&self.sigma_x)
                .series(&self.sigma_p)
                .series(&self.product)
                .function("free prediction σₓ", look.prediction(), &prediction)
                .hline(0.5, "ħ/2")
                .window(TimeWindow::All)
                .now(self.t)
                .labels("t", "")
                .width(w - hist_w - if hist_w > 0.0 { look.px(8.0) } else { 0.0 })
                .height(bottom_h);
            for (t, l) in &self.events {
                ts = ts.event(*t, l.clone());
            }
            ts.show(ui);
            if let Some(r) = &self.repeated {
                HistogramPlot::new("gauss_hist", &r.hist)
                    .predicted(&r.predicted)
                    .scale(HistScale::Density)
                    .x_label(format!("{} (repeat N)", r.domain.var()))
                    .width(hist_w)
                    .height(bottom_h)
                    .show(ui);
            }
        });
    }

    fn side_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx) {
        let look = ctx.look;
        ui.label(RichText::new("Initial state").strong());
        let mut changed = false;
        changed |= ui.add(egui::Slider::new(&mut self.sigma0, 0.3..=3.0).text("σ₀ = σₓ(0)")).changed();
        ui.label(
            RichText::new(format!(
                "a = 1/(4σ₀²) = {:.3},   spreading time 2σ₀² = {:.2}",
                0.25 / (self.sigma0 * self.sigma0),
                2.0 * self.sigma0 * self.sigma0
            ))
            .color(look.weak_text()),
        );
        changed |= ui.add(egui::Slider::new(&mut self.p0, -4.0..=4.0).text("k₀ = p₀")).changed();
        if self.p0 != 0.0 {
            ui.label(
                RichText::new(format!(
                    "crests: v_phase = k₀/2 = {:.2},  envelope: v_group = k₀ = {:.2}",
                    0.5 * self.p0,
                    self.p0
                ))
                .color(look.weak_text()),
            );
        }
        changed |= ui.add(egui::Slider::new(&mut self.x0, -15.0..=15.0).text("x₀")).changed();
        if changed {
            self.restart();
        }

        ui.add_space(look.px(8.0));
        ui.label(RichText::new("View").strong());
        ui.label(RichText::new("x / p and the display mode: switches above the plot (V cycles the mode).").color(look.weak_text()));
        ui.checkbox(&mut self.markers, "⟨x⟩ and width markers");

        ui.add_space(look.px(8.0));
        ui.label(RichText::new("Measure").strong());
        ui.add(
            egui::Slider::new(&mut self.width_x, 0.05..=3.0)
                .logarithmic(true)
                .text("x detector σ"),
        );
        ui.add(
            egui::Slider::new(&mut self.width_p, 0.02..=2.0)
                .logarithmic(true)
                .text("p detector σ"),
        );
        ui.horizontal(|ui| {
            if ui.button(RichText::new("Measure x  (M)").color(look.measure())).clicked() {
                self.measure_x(ctx);
            }
            if ui.button(RichText::new("Measure p").color(look.measure())).clicked() {
                self.measure_p(ctx);
            }
        });
        if let Some(n) = repeat_button(ui, "gauss_rep_x", "Repeat x", &mut self.repeat_n) {
            self.repeat(Domain::Position, n, ctx);
        }
        if let Some(n) = repeat_button(ui, "gauss_rep_p", "Repeat p", &mut self.repeat_n) {
            self.repeat(Domain::Momentum, n, ctx);
        }
        ui.label(
            RichText::new(
                "Single shot: ψ collapses and keeps evolving. Repeat N: N fresh copies of the current ψ, compared with the Born rule.",
            )
            .color(look.weak_text())
            .size(12.0 * look.font_scale()),
        );
        if let Some(l) = self.flash.label() {
            ui.label(RichText::new(format!("last outcome: {l}")).color(look.measure()));
        }
    }

    fn notes(&self) -> String {
        match self.variant {
            Variant::Exercise => NOTES.to_string(),
            Variant::Moving => NOTES_MOVING.to_string(),
        }
    }
}

const NOTES: &str = "\
# The exercise
A free particle starts as Ψ(x, 0) = A e^(−ax²). In this applet σ₀ = σₓ(0) = 1/(2√a), so the default σ₀ = 1 is a = 1/4. Units ħ = m = 1.
- (a) normalise: A = (2a/π)^(1/4)
- (b, c) Ψ(x, t) stays a Gaussian; |Ψ|² has the width σₓ(t) = σ₀ √(1 + (t/(2σ₀²))²), i.e. ω² = a/(1 + (2at)²) in the exercise's notation
- (d) σₚ = 1/(2σ₀) does not change; σₓσₚ = ħ/2 only at t = 0, the closest approach to the limit
- (e) |Ψ|² flattens and broadens because the momentum components travel at different speeds

# Try
- Play and watch σₓ(t) follow the dashed prediction while σₚ stays flat.
- Colour is the phase: the packet develops a chirp, the faster components run ahead.
- Show x + p: |Φ(p)|² does not change at all, only its phase turns.
- Make σ₀ smaller: the packet spreads faster (time scale 2σ₀²).
- Give it a momentum p₀: it moves with ⟨p⟩ = p₀ and spreads in the same way.
- Measure x with a narrow detector: ψ collapses, σₚ jumps up and the packet then spreads much faster. Measuring p does the opposite.
- Repeat N: measure many fresh copies and compare the histogram with |ψ|² (Born rule). The shaded band is the expected scatter for N shots.

# Behind the scenes
Split-step Fourier propagation on 2048 points in −60 < x < 60, with absorbing edges. The dashed curve is σₓ²(t) = σₓ² + 2C·(t − t₀) + σₚ²·(t − t₀)² from the moments at the start or the last measurement (exact for any free state; C is the x–p covariance).";

const NOTES_MOVING: &str = "\
# A moving packet: phase and group velocity
The same Gaussian with a carrier: ψ(x, 0) ∝ e^(−(x − x₀)²/(4σ₀²)) e^(ik₀x), here k₀ = 3, σ₀ = 2, x₀ = −15. Shown as Re ψ: the oscillation is the carrier, the faint band ±|ψ| the envelope.
- Free particle: ω(k) = ħk²/(2m), so the crests move with v_phase = ω/k = k₀/2 and the envelope with v_group = dω/dk = k₀.
- Play: the crests slide backwards through the envelope, which moves twice as fast.
- The packet still spreads (time scale 2σ₀² = 8); the carrier then gets a chirp: short wavelengths at the front, long ones at the back.
- Switch the display (V): |ψ|² + phase shows the same chirp as colour bands; Re ψ, Im ψ shows the 90° shift between the two parts.
- Show x + p: |Φ(p)|² is centred on k₀ and does not change.
";
