//! PS09 37 · The truncated plane wave Ψ = e^(i2πx/λ)/√(2nλ) on (−nλ, nλ).
//!
//! x and p side by side (default x + p): the sliders n and λ change both
//! panels live, with the widths wₓ, wₚ between the zeros either side of the
//! main peak and their product. Other initial shapes (Gaussian envelope, two
//! boxes, a one-wavelength spike) build Fourier intuition. Free evolution:
//! |Φ(p)|² stays put, only its phase winds; in x the sharp edges spread.

use std::f64::consts::{PI, TAU};

use qb_core::egui::{self, RichText};
use qb_core::num::{Absorber, Grid1D, Moments, Momentum, SplitStep, Wave1D, c64, measure_momentum, measure_position};
use qb_core::widgets::{
    Domain, FlashKind, Marker, MeasureFlash, ReprView, WaveCurve, WavePlot, WaveStyle, fmt_num, label_box, readout_row, repeat_button,
    segmented, view_bar,
};
use qb_core::{Ctx, Look, Scene};

use crate::common::{self, Repeated, grow_top, hint, section};
use crate::physics::{truncated_widths, zeros_around_peak};

/// dx = 0.05 with the grid points at odd multiples of 0.025: the box edges ±nλ
/// (λ in steps of 0.1) fall exactly half-way between two points, so the sampled
/// box has exactly the width 2nλ.
const N: usize = 8192;
const DX: f64 = 0.05;
const X_MIN: f64 = -(N as f64 / 2.0 + 0.5) * DX;
const X_MAX: f64 = X_MIN + N as f64 * DX;
const DT_MAX: f64 = 0.02;

/// The initial envelope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// the exercise: a sharp box of half-width nλ
    Box,
    /// a Gaussian envelope with the same σₓ = nλ/√3 as the box
    Gauss,
    /// the box with its middle third removed: two boxes (a double slit)
    TwoBoxes,
    /// a box only one wavelength wide
    Spike,
}

impl Shape {
    const ALL: [Shape; 4] = [Shape::Box, Shape::Gauss, Shape::TwoBoxes, Shape::Spike];

    fn label(self) -> &'static str {
        match self {
            Shape::Box => "box (exercise)",
            Shape::Gauss => "Gaussian",
            Shape::TwoBoxes => "two boxes",
            Shape::Spike => "spike",
        }
    }

    fn hover(self) -> &'static str {
        match self {
            Shape::Box => "e^(i2πx/λ) on −nλ < x < nλ: the exercise",
            Shape::Gauss => "e^(i2πx/λ) with a Gaussian envelope of the same σₓ = nλ/√3: no zeros, no side lobes",
            Shape::TwoBoxes => "two boxes nλ/3 < |x| < nλ: fringes in p with spacing 2π/d",
            Shape::Spike => "a box only one wavelength wide: narrow in x, very wide in p",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "box" => Some(Shape::Box),
            "gauss" | "gaussian" => Some(Shape::Gauss),
            "two" | "two_boxes" => Some(Shape::TwoBoxes),
            "spike" => Some(Shape::Spike),
            _ => None,
        }
    }
}

/// Measured widths between the zeros around the main peaks (None: no zero on that side).
#[derive(Clone, Copy, Default)]
struct Widths {
    x: (Option<f64>, Option<f64>),
    p: (Option<f64>, Option<f64>),
    p_peak: f64,
}

impl Widths {
    fn wx(&self) -> Option<f64> {
        Some(self.x.1? - self.x.0?)
    }
    fn wp(&self) -> Option<f64> {
        Some(self.p.1? - self.p.0?)
    }
}

pub struct TruncatedScene {
    // parameters
    n: u32,
    lambda: f64,
    shape: Shape,
    width_x: f64,
    width_p: f64,
    repeat_n: usize,
    view: ReprView,
    style: WaveStyle,
    markers: bool,
    follow: bool,

    // state
    grid: Grid1D,
    xs: Vec<f64>,
    wave: Wave1D,
    mom: Momentum,
    prop: SplitStep,
    t: f64,
    moments: Moments,
    widths: Widths,
    repeated: Option<Repeated>,
    flash: MeasureFlash,
    top_x: f64,
    top_p: f64,
    /// a measurement has changed the state since t = 0
    measured: bool,
}

/// The grid of this scene (box edges half-way between grid points).
pub fn grid() -> Grid1D {
    Grid1D::new(X_MIN, X_MAX, N)
}

impl TruncatedScene {
    pub fn new() -> Self {
        let grid = grid();
        let wave = Wave1D::zeros(grid);
        let mom = wave.momentum();
        let mut s = TruncatedScene {
            n: 3,
            lambda: 1.0,
            shape: Shape::Box,
            width_x: 0.4,
            width_p: 0.05,
            repeat_n: 1000,
            view: ReprView::Both,
            style: WaveStyle::Density,
            markers: true,
            follow: true,
            grid,
            xs: grid.xs(),
            wave,
            mom,
            prop: SplitStep::new(grid, DT_MAX).with_absorber(Absorber::new(30.0, 20.0)),
            t: 0.0,
            moments: Moments::default(),
            widths: Widths::default(),
            repeated: None,
            flash: MeasureFlash::new(),
            top_x: 0.0,
            top_p: 0.0,
            measured: false,
        };
        s.restart();
        s
    }

    /// Half-width L = nλ of the box.
    fn half(&self) -> f64 {
        self.n as f64 * self.lambda
    }

    /// wₓ (zeros of |ψ|²) is the exercise's quantity for the initial state only.
    fn wx_valid(&self) -> bool {
        self.t == 0.0 && !self.measured
    }

    /// |Φ(p)|² does not change in free flight, so wₚ stays valid until a measurement.
    fn wp_valid(&self) -> bool {
        !self.measured
    }

    /// Carrier momentum p₀ = 2πħ/λ = h/λ.
    fn p0(&self) -> f64 {
        TAU / self.lambda
    }

    fn initial_wave(&self) -> Wave1D {
        let (l, p0) = (self.half(), self.p0());
        let g = self.grid;
        let mut w = match self.shape {
            Shape::Box => Wave1D::plane_wave_truncated(g, 0.0, l, p0),
            Shape::Gauss => Wave1D::gaussian(g, 0.0, l / 3f64.sqrt(), p0),
            Shape::TwoBoxes => Wave1D::from_fn(g, |x| {
                if x.abs() > l / 3.0 && x.abs() < l {
                    c64(0.0, p0 * x).exp()
                } else {
                    c64(0.0, 0.0)
                }
            }),
            Shape::Spike => Wave1D::plane_wave_truncated(g, 0.0, 0.5 * self.lambda, p0),
        };
        w.normalize();
        w
    }

    /// Back to t = 0 with the current parameters.
    fn restart(&mut self) {
        self.wave = self.initial_wave();
        self.t = 0.0;
        self.measured = false;
        self.repeated = None;
        self.flash.clear();
        self.update_derived();
        self.top_x = 1.2 * common::peak_density(&self.wave.psi);
        self.top_p = 1.2 * common::peak_density(&self.mom.phi);
    }

    fn update_derived(&mut self) {
        self.mom = self.wave.momentum();
        self.moments = common::quick_moments(&self.wave, &self.mom);
        let x = zeros_around_peak(&self.xs, &self.wave.psi);
        let p = zeros_around_peak(&self.mom.p, &self.mom.phi);
        self.widths = Widths {
            x: (x.0, x.1),
            p: (p.0, p.1),
            p_peak: self.mom.p[p.2],
        };
    }

    fn measure_x(&mut self, ctx: &mut Ctx) {
        let x = measure_position(&mut self.wave, self.width_x, ctx.rng);
        self.measured = true;
        self.update_derived();
        self.flash
            .trigger(FlashKind::Position { x, width: self.width_x }, format!("x = {x:.2}"));
        ctx.set_status(format!(
            "measured x = {x:.3} at t = {:.2} (detector σ = {}): now narrow in x, so wide in p (σₚ = {:.2})",
            self.t, self.width_x, self.moments.sigma_p
        ));
    }

    fn measure_p(&mut self, ctx: &mut Ctx) {
        let p = measure_momentum(&mut self.wave, self.width_p, ctx.rng);
        self.measured = true;
        self.update_derived();
        self.flash
            .trigger(FlashKind::Momentum { p, width: self.width_p }, format!("p = {p:.2}"));
        if !self.view.shows_p() {
            self.view = ReprView::Both;
        }
        ctx.set_status(format!(
            "measured p = {p:.3} (h/λ = {:.3}) at t = {:.2} (detector σ = {}): now narrow in p, so wide in x (σₓ = {:.1})",
            self.p0(),
            self.t,
            self.width_p,
            self.moments.sigma_x
        ));
    }

    fn repeat(&mut self, domain: Domain, n: usize, ctx: &mut Ctx) {
        let r = match domain {
            Domain::Position => common::repeat_position(&self.wave, n, ctx.rng),
            Domain::Momentum => common::repeat_momentum(&self.wave, n, ctx.rng),
        };
        let v = domain.var();
        ctx.set_status(format!(
            "{n} fresh copies measured in {v}: mean {:.3}, spread {:.3}; bars against the Born rule |{}|²",
            r.hist.mean(),
            r.hist.std(),
            domain.func()
        ));
        self.repeated = Some(r);
    }

    /// Default visible x range: wide enough for the largest box, fixed while n changes
    /// inside the usual range (so that the box visibly grows).
    fn x_view(&self) -> (f64, f64) {
        let half = (1.35 * self.half()).max(16.0);
        let half = if half > 16.0 { (half / 10.0).ceil() * 10.0 } else { half };
        let c = if self.follow { self.moments.x } else { 0.0 };
        (c - half, c + half)
    }

    /// Default visible p range: [−4, 16] unless the main lobe needs more.
    fn p_view(&self) -> (f64, f64) {
        let p0 = self.p0();
        let lobe = match self.shape {
            Shape::Spike => TAU / (0.5 * self.lambda),
            _ => TAU / self.half(),
        };
        let lo = (-4.0f64).min(((p0 - 1.3 * lobe) / 2.0).floor() * 2.0);
        let hi = 16.0f64.max(((p0 + 1.3 * lobe) / 2.0).ceil() * 2.0);
        (lo, hi)
    }

    fn plot(&mut self, ui: &mut egui::Ui, domain: Domain, height: f32) {
        let look = Look::of(ui);
        let (xs, psi) = match domain {
            Domain::Position => (&self.xs, &self.wave.psi),
            Domain::Momentum => (&self.mom.p, &self.mom.phi),
        };
        let top = match domain {
            Domain::Position => &mut self.top_x,
            Domain::Momentum => &mut self.top_p,
        };
        grow_top(top, common::peak_density(psi));
        let (lo, hi) = self.style.y_range(*top);
        let (x_lo, x_hi) = match domain {
            Domain::Position => self.x_view(),
            Domain::Momentum => self.p_view(),
        };
        let mut plot = WavePlot::new(("trunc", domain))
            .domain(domain)
            .style(self.style)
            .x_range(x_lo, x_hi)
            .y_range(lo, hi)
            .height(height)
            .curve(WaveCurve::new(xs, psi));
        let zeros = match domain {
            Domain::Position => self.wx_valid().then_some(self.widths.x),
            Domain::Momentum => self.wp_valid().then_some(self.widths.p),
        };
        if self.markers {
            let name = if domain == Domain::Position { "wₓ" } else { "wₚ" };
            if let Some(zeros) = zeros {
                for z in [zeros.0, zeros.1].into_iter().flatten() {
                    plot = plot.marker(Marker::vline(z, "").color(look.accent()).dashed());
                }
                // (the bracket only over |ψ|²: Re ψ fills the plot height)
                if let (Some(a), Some(b)) = zeros
                    && self.style.is_quadratic()
                {
                    plot = plot.marker(Marker::bracket(a, b, 0.9 * hi, format!("{name} = {:.3}", b - a)).color(look.text()));
                }
            }
            if domain == Domain::Momentum {
                plot = plot.marker(Marker::vline(self.p0(), "").color(look.prediction()));
            }
        }
        let resp = plot.show(ui);
        if self.markers && domain == Domain::Momentum {
            // "h/λ" right of the main lobe, half-way up
            let tr = resp.transform;
            let right = zeros.and_then(|z| z.1).unwrap_or(self.p0());
            label_box(
                &resp.painter(ui),
                egui::pos2(tr.x_to_screen(right) + look.px(6.0), tr.y_to_screen(0.5 * (lo + hi))),
                egui::Align2::LEFT_CENTER,
                &format!("peak at h/λ = {:.3}", self.p0()),
                look.prediction(),
            );
        }
        self.flash.paint(ui, &resp.transform, domain);
        if let Some((x, _)) = resp.hover {
            resp.response.on_hover_text(format!("{} = {x:.3}", domain.var()));
        }
    }
}

impl Default for TruncatedScene {
    fn default() -> Self {
        Self::new()
    }
}

impl Scene for TruncatedScene {
    fn time(&self) -> f64 {
        self.t
    }

    /// The packet moves at v = h/λ ≈ 6 for λ = 1: half speed keeps it on screen a while.
    fn time_scale(&self) -> f64 {
        0.5
    }

    fn step(&mut self, dt: f64, _ctx: &mut Ctx) {
        self.prop.advance(&mut self.wave, dt);
        self.t += dt;
        self.update_derived();
    }

    fn reset(&mut self, _ctx: &mut Ctx) {
        self.restart();
    }

    fn measure_label(&self) -> Option<String> {
        Some("Measure p".into())
    }

    fn measure(&mut self, ctx: &mut Ctx) {
        self.measure_p(ctx);
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
            "measure_p" => self.measure_p(ctx),
            "repeat_x" => self.repeat(Domain::Position, n, ctx),
            "repeat_p" => self.repeat(Domain::Momentum, n, ctx),
            "n" => {
                let Some(v) = common::num_arg(arg) else { return false };
                self.n = (v.round() as u32).clamp(1, 12);
                self.restart();
            }
            "lambda" => {
                let Some(v) = common::num_arg(arg) else { return false };
                self.lambda = (v.clamp(0.5, 4.0) * 10.0).round() / 10.0;
                self.restart();
            }
            "shape" => {
                let Some(s) = arg.and_then(Shape::parse) else { return false };
                self.shape = s;
                self.restart();
            }
            "follow" => self.follow = true,
            "markers_off" => self.markers = false,
            _ => return common::view_style_action(cmd, &mut self.view, &mut self.style),
        }
        true
    }

    fn central_ui(&mut self, ui: &mut egui::Ui, _ctx: &mut Ctx) {
        let look = Look::of(ui);
        view_bar(ui, &mut self.view, &mut self.style);
        let m = self.moments;
        let fmt_w = |w: Option<f64>| w.map_or("—".to_string(), |w| format!("{w:.3}"));
        let wx = self.widths.wx().filter(|_| self.wx_valid());
        let wp = self.widths.wp().filter(|_| self.wp_valid());
        let product = match (wx, wp) {
            (Some(a), Some(b)) => format!("{:.2} = {:.2}·h", a * b, a * b / TAU),
            _ if !self.wx_valid() => "— (t > 0)".into(),
            _ => "— (no zeros)".into(),
        };
        readout_row(
            ui,
            &[
                ("h/λ", format!("{:.3}", self.p0()), Some(look.prediction())),
                ("|Φ|² peak at p", format!("{:.3}", self.widths.p_peak), None),
                ("wₓ", fmt_w(wx), Some(look.series(0))),
                ("wₚ", fmt_w(wp), Some(look.series(1))),
                ("wₓwₚ", product, Some(look.series(2))),
                ("σₓ", fmt_num(m.sigma_x, 3), None),
                ("σₚ", fmt_num(m.sigma_p, 2), None),
                ("⟨p⟩", fmt_num(m.p, 3), None),
                ("∫|ψ|²", format!("{:.3}", m.norm), None),
            ],
        );
        let bottom_h = if self.repeated.is_some() { look.px(160.0) } else { 0.0 };
        let plots_h = (ui.available_height() - bottom_h - look.px(10.0)).max(200.0);
        let view = self.view;
        view.show(ui, plots_h, |ui, d, h| self.plot(ui, d, h));
        if let Some(r) = &self.repeated {
            ui.add_space(look.px(6.0));
            let w = ui.available_width();
            r.show(ui, "trunc_hist", w, bottom_h);
        }
    }

    fn side_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx) {
        let look = ctx.look;
        ui.label(RichText::new("Initial state").strong());
        let mut changed = false;
        segmented(ui, |ui| {
            for s in Shape::ALL {
                if ui.selectable_label(self.shape == s, s.label()).on_hover_text(s.hover()).clicked() && self.shape != s {
                    self.shape = s;
                    changed = true;
                }
            }
        });
        let spike = self.shape == Shape::Spike;
        changed |= ui.add_enabled(!spike, egui::Slider::new(&mut self.n, 1..=12).text("n")).changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.lambda, 0.5..=4.0).step_by(0.1).text("λ"))
            .changed();
        let (wx, wp) = truncated_widths(self.n as f64, self.lambda);
        hint(
            ui,
            match self.shape {
                Shape::Box => format!(
                    "exercise: wₓ = 2nλ = {wx:.3},  wₚ = h/(nλ) = {wp:.3},  wₓwₚ = 2h = 4πħ = {:.2}",
                    4.0 * PI
                ),
                Shape::Gauss => format!(
                    "σₓ = nλ/√3 = {:.3} (as the box), σₚ = 1/(2σₓ) = {:.3}: σₓσₚ = ħ/2, no zeros",
                    self.half() / 3f64.sqrt(),
                    3f64.sqrt() / (2.0 * self.half())
                ),
                Shape::TwoBoxes => {
                    let d = 4.0 * self.half() / 3.0;
                    format!(
                        "boxes of width 2nλ/3 = {:.2}, centres d = 4nλ/3 = {d:.2} apart: fringes in p every 2π/d = {:.3}",
                        2.0 * self.half() / 3.0,
                        TAU / d
                    )
                }
                Shape::Spike => format!(
                    "a box one wavelength wide: wₓ = λ = {:.2}, wₚ = 4π/λ = {:.2} (n is not used)",
                    self.lambda,
                    2.0 * TAU / self.lambda
                ),
            },
        );
        if changed {
            self.restart();
        }

        section(ui, "View");
        ui.checkbox(&mut self.markers, "zeros, widths wₓ, wₚ and h/λ");
        ui.checkbox(&mut self.follow, "x axis follows the packet (v = h/λ)");
        hint(ui, "x / p and the display mode: switches above the plots (V cycles the mode).");

        section(ui, "Measure");
        ui.add(
            egui::Slider::new(&mut self.width_x, 0.05..=5.0)
                .logarithmic(true)
                .text("x detector σ"),
        );
        ui.add(
            egui::Slider::new(&mut self.width_p, 0.01..=2.0)
                .logarithmic(true)
                .text("p detector σ"),
        );
        ui.horizontal(|ui| {
            if ui.button(RichText::new("Measure p  (M)").color(look.measure())).clicked() {
                self.measure_p(ctx);
            }
            if ui.button(RichText::new("Measure x").color(look.measure())).clicked() {
                self.measure_x(ctx);
            }
        });
        if let Some(n) = repeat_button(ui, "trunc_rep_p", "Repeat p", &mut self.repeat_n) {
            self.repeat(Domain::Momentum, n, ctx);
        }
        if let Some(n) = repeat_button(ui, "trunc_rep_x", "Repeat x", &mut self.repeat_n) {
            self.repeat(Domain::Position, n, ctx);
        }
        hint(
            ui,
            "Single shot: ψ collapses and keeps evolving (narrow detector in x ⇒ wide in p, and vice versa). Repeat N: N fresh copies of the current ψ against the Born rule.",
        );
        if let Some(l) = self.flash.label() {
            ui.label(RichText::new(format!("last outcome: {l}")).color(look.measure()));
        }
        if self.measured {
            hint(ui, "R restarts from the initial shape.");
        }
    }

    fn notes(&self) -> String {
        NOTES.into()
    }
}

const NOTES: &str = "\
# The exercise
Ψ(x, 0) = e^(i2πx/λ)/√(2nλ) for −nλ < x < nλ, 0 outside: a plane wave cut to n wavelengths on each side. Units ħ = m = 1, so h = 2π and h/λ = 2π/λ.
- (a) Φ(p) = √(ħλ/(nπ)) · sin(npλ/ħ)/(pλ − 2πħ): a sinc centred on p = h/λ (the de Broglie momentum of the carrier).
- (b) |Ψ|² is flat, its zeros are the edges: wₓ = 2nλ. |Φ|² peaks at p = h/λ, its first zeros are at h/λ·(1 ± 1/(2n)): wₚ = h/(nλ) = 2πħ/(nλ).
- (c) n → ∞: wₓ → ∞, wₚ → 0. The momentum becomes sharp, the position completely uncertain.
- (d) wₓ·wₚ = 4πħ = 2h ≥ ħ/2 for every n and λ. These are full widths between zeros, not standard deviations, so the number differs from the σₓσₚ ≥ ħ/2 bound but the uncertainty principle is satisfied.
- (e) Ψ is normalised, but Ψ and ∂Ψ/∂x jump at x = ±nλ. A physical wavefunction must be continuous (with a continuous derivative where V is finite), so this Ψ is not a valid wavefunction: it is an idealisation, like the states of an infinitely deep well. One sign of it here: |Φ|² falls off only like 1/p², so ⟨p²⟩ (the kinetic energy) is infinite. The σₚ readout is finite only because the grid has a largest momentum.

# Try
- Move n: the box in x grows, the sinc in p narrows; the product wₓwₚ stays 12.57 = 2h.
- Move λ: the peak of |Φ|² follows h/λ, and wₚ = h/(nλ) shrinks too.
- Switch to Re ψ (V): the carrier has exactly 2n wavelengths inside the box.
- Gaussian: same σₓ, but |Φ|² has no side lobes and no zeros (σₓσₚ = ħ/2, the minimum). Sharp edges cause the side lobes.
- Two boxes: |Φ|² gets fringes, like a double slit, under the envelope of one box.
- Spike: one wavelength in x gives a very wide |Φ|²; momentum is then hardly defined.
- Play: |Φ(p)|² does not change (free particle), only its phase colour winds as e^(−ip²t/2ħ). In x the packet moves at v = h/(mλ) and the sharp edges smear out first (the high-momentum side lobes run away).
- Measure x with a narrow detector: the state becomes narrow in x and wide in p. Measure p (M) with a narrow detector: narrow in p, very wide in x.
- Repeat p: the outcomes follow |Φ(p)|², including the side lobes.

# Behind the scenes
Split-step Fourier propagation on 8192 points (dx = 0.05) in −205 < x < 205 with absorbing edges (beyond |x| ≈ 175). The box edges fall half-way between grid points, so the sampled box is exactly 2nλ wide. Φ(p) is the FFT of the sampled Ψ; it agrees with the analytic sinc above to the grid accuracy. The measured widths come from the plotted curves: an edge counts as a zero at the midpoint of the jump, a minimum is refined with a parabola.";
