//! Pieces shared by the 1D scenes: "repeat N" results, the stable vertical
//! scale, the view/style actions, energy levels drawn between their turning
//! points, and a level ladder lined up with an energy-axis plot.

use qb_core::egui::{self, Align2, RichText, pos2};
use qb_core::num::{C64, Sampler, ShotHistogram, Wave1D, born_samples_p, born_samples_x};
use qb_core::widgets::{Domain, HistScale, HistogramPlot, LadderResponse, LevelLadder, PlotFrame, PlotTransform, ReprView, WaveStyle};
use qb_core::{Look, QbRng};

/// The last "repeat N" run of a scene: counts, Born-rule prediction, labels.
pub struct Repeated {
    pub hist: ShotHistogram,
    /// Born-rule probability of each bin
    pub predicted: Vec<f64>,
    pub x_label: String,
    /// names of discrete outcomes
    pub labels: Option<Vec<String>>,
    pub scale: HistScale,
}

impl Repeated {
    /// Binned outcomes of a continuous variable, drawn as a density.
    pub fn binned(hist: ShotHistogram, predicted: Vec<f64>, x_label: impl Into<String>) -> Self {
        Repeated {
            hist,
            predicted,
            x_label: x_label.into(),
            labels: None,
            scale: HistScale::Density,
        }
    }

    /// Discrete outcomes (levels, yes/no), drawn as fractions.
    pub fn discrete(hist: ShotHistogram, predicted: Vec<f64>, x_label: impl Into<String>, labels: Vec<String>) -> Self {
        Repeated {
            hist,
            predicted,
            x_label: x_label.into(),
            labels: Some(labels),
            scale: HistScale::Fraction,
        }
    }

    pub fn show(&self, ui: &mut egui::Ui, id: &str, width: f32, height: f32) {
        let mut h = HistogramPlot::new(id, &self.hist)
            .predicted(&self.predicted)
            .scale(self.scale)
            .x_label(self.x_label.clone())
            .width(width)
            .height(height);
        if let Some(l) = &self.labels {
            h = h.labels(l);
        }
        h.show(ui);
    }
}

/// The range that holds all but `tail` of a density on each side, padded by 10 %.
pub fn quantile_range(xs: &[f64], density: &[f64], tail: f64) -> (f64, f64) {
    let total: f64 = density.iter().map(|d| d.max(0.0)).sum();
    if total <= 0.0 || xs.len() < 2 {
        return (-1.0, 1.0);
    }
    let mut acc = 0.0;
    let (mut lo, mut hi) = (xs[0], xs[xs.len() - 1]);
    let mut found_lo = false;
    for (x, d) in xs.iter().zip(density) {
        acc += d.max(0.0) / total;
        if !found_lo && acc >= tail {
            lo = *x;
            found_lo = true;
        }
        if acc >= 1.0 - tail {
            hi = *x;
            break;
        }
    }
    let pad = 0.1 * (hi - lo).max(xs[1] - xs[0]) + (xs[1] - xs[0]);
    (lo - pad, hi + pad)
}

/// "Repeat N" of a position measurement on fresh copies of `wave`.
pub fn repeat_position(wave: &Wave1D, n: usize, rng: &mut QbRng) -> Repeated {
    let xs = wave.xs();
    let d = wave.density();
    let (lo, hi) = quantile_range(&xs, &d, 0.002);
    let mut hist = ShotHistogram::binned(lo, hi, 48);
    hist.add_all(&born_samples_x(wave, n, rng));
    let pred = hist.predict_from_density(&xs, &d);
    Repeated::binned(hist, pred, "x (repeat N)")
}

/// "Repeat N" of a momentum measurement on fresh copies of `wave`.
pub fn repeat_momentum(wave: &Wave1D, n: usize, rng: &mut QbRng) -> Repeated {
    let m = wave.momentum();
    let d = m.density();
    let (lo, hi) = quantile_range(&m.p, &d, 0.002);
    let mut hist = ShotHistogram::binned(lo, hi, 48);
    hist.add_all(&born_samples_p(wave, n, rng));
    let pred = hist.predict_from_density(&m.p, &d);
    Repeated::binned(hist, pred, "p (repeat N)")
}

/// "Repeat N" of an energy measurement with probabilities `probs` (|c_n|²);
/// outcome k is labelled by `names[k]`.
pub fn repeat_discrete(probs: &[f64], names: Vec<String>, n: usize, rng: &mut QbRng, x_label: &str) -> Repeated {
    let total: f64 = probs.iter().sum();
    let mut hist = ShotHistogram::discrete(probs.len());
    for _ in 0..n {
        if let Some(k) = rng.weighted(probs) {
            hist.add_index(k);
        }
    }
    let pred = probs.iter().map(|p| p / total.max(1e-300)).collect();
    Repeated::discrete(hist, pred, x_label, names)
}

/// Draws one value from a density on a uniform grid.
pub fn sample_density(xs: &[f64], density: &[f64], rng: &mut QbRng) -> Option<f64> {
    Sampler::new(xs, density).map(|s| s.sample(rng))
}

/// Largest |ψ|².
pub fn peak_density(psi: &[C64]) -> f64 {
    psi.iter().map(|z| z.norm_sqr()).fold(0.0, f64::max)
}

/// The stable vertical scale: `top` only grows, when the curve would be clipped.
pub fn grow_top(top: &mut f64, peak: f64) {
    if peak * 1.05 > *top {
        *top = peak * 1.2;
    }
}

/// Factor that maps the plotted quantity of `style` (|ψ|² or Re/Im ψ) onto
/// `height` energy units, for a state whose |ψ|² never exceeds `top`.
pub fn energy_curve_scale(style: WaveStyle, top: f64, height: f64) -> f64 {
    if style.is_quadratic() {
        height / top.max(1e-12)
    } else {
        // ±√top fills ±height/2 … height: keep the same visual size as |ψ|²
        0.5 * height / top.max(1e-12).sqrt()
    }
}

/// The actions every 1D scene understands: view_x|view_p|view_both,
/// style_density|style_re|style_reim|style_plain. Returns true if handled.
pub fn view_style_action(cmd: &str, view: &mut ReprView, style: &mut WaveStyle) -> bool {
    if let Some(v) = cmd.strip_prefix("view_")
        && let Some(r) = ReprView::parse(v)
    {
        *view = r;
        return true;
    }
    if let Some(s) = cmd.strip_prefix("style_")
        && let Some(st) = WaveStyle::parse(s)
    {
        *style = st;
        return true;
    }
    false
}

/// Parses the numeric argument of an action ("lambda 0.8").
pub fn num_arg(arg: Option<&str>) -> Option<f64> {
    arg.and_then(|w| w.trim().parse::<f64>().ok())
}

/// "x" → "ₓ", "p" → "ₚ"
pub fn sub(v: &str) -> &'static str {
    if v == "x" { "ₓ" } else { "ₚ" }
}

/// Small grey help text in the side panel.
pub fn hint(ui: &mut egui::Ui, text: impl Into<String>) {
    let look = Look::of(ui);
    ui.label(RichText::new(text.into()).color(look.weak_text()).size(12.0 * look.font_scale()));
}

/// Bold section title in the side panel, with a little space above.
pub fn section(ui: &mut egui::Ui, title: &str) {
    let look = Look::of(ui);
    ui.add_space(look.px(8.0));
    ui.label(RichText::new(title).strong());
}

/// Classical turning points of V at energy e around the minimum of V
/// (the outermost crossings left and right of the lowest point).
pub fn turning_points(xs: &[f64], v: &[f64], e: f64) -> Option<(f64, f64)> {
    if xs.len() < 2 {
        return None;
    }
    let imin = v.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i)?;
    if v[imin] > e {
        return None;
    }
    let cross = |i: usize, j: usize| {
        // linear interpolation between points i and j where V − e changes sign
        let (a, b) = (v[i] - e, v[j] - e);
        let t = if (b - a).abs() > 1e-300 { a / (a - b) } else { 0.5 };
        xs[i] + t * (xs[j] - xs[i])
    };
    let mut lo = xs[0];
    for i in (1..=imin).rev() {
        if v[i - 1] > e {
            lo = cross(i - 1, i);
            break;
        }
    }
    let mut hi = xs[xs.len() - 1];
    for i in imin..xs.len() - 1 {
        if v[i + 1] > e {
            hi = cross(i, i + 1);
            break;
        }
    }
    Some((lo, hi))
}

/// Draws energy levels as horizontal lines between their turning points of V
/// (the whole width where V stays below the level).
pub fn draw_levels(
    painter: &egui::Painter,
    tr: &PlotTransform,
    xs: &[f64],
    v: &[f64],
    energies: &[f64],
    stroke: egui::Stroke,
    dashed: bool,
) {
    for &e in energies {
        if e < tr.y[0] || e > tr.y[1] {
            continue;
        }
        let (a, b) = turning_points(xs, v, e).unwrap_or((tr.x[0], tr.x[1]));
        let pts = [tr.to_screen(a.max(tr.x[0]), e), tr.to_screen(b.min(tr.x[1]), e)];
        if dashed {
            painter.extend(egui::Shape::dashed_line(&pts, stroke, 5.0, 4.0));
        } else {
            painter.line_segment(pts, stroke);
        }
    }
}

/// Energy range for a [`LevelLadder`] of the same outer height placed next to
/// an energy-axis plot (same row), so that its levels line up with the plot's.
pub fn ladder_range_for(tr: &PlotTransform, look: &Look) -> (f64, f64) {
    let (_, _, mt, mb) = PlotFrame::margins(look, false);
    let outer_top = tr.rect.top() - mt;
    let outer_bottom = tr.rect.bottom() + mb;
    // LevelLadder's own margins: 8 points (scaled) above and below
    let lo = tr.y_from_screen(outer_bottom - look.px(8.0));
    let hi = tr.y_from_screen(outer_top + look.px(8.0));
    (lo, hi)
}

/// A caption in the top-left corner under the y label of a plot.
pub fn plot_caption(painter: &egui::Painter, tr: &PlotTransform, line: usize, text: &str, color: egui::Color32) {
    let look = Look::from_ctx(painter.ctx());
    let y = tr.rect.top() + look.px(22.0) + line as f32 * look.px(16.0);
    let g = painter.layout_no_wrap(text.to_string(), look.font(12.0), color);
    let r = Align2::LEFT_TOP.anchor_size(pos2(tr.rect.left() + look.px(6.0), y), g.size());
    painter.rect_filled(r.expand(2.0), 2.0, look.plot_bg().gamma_multiply(0.85));
    painter.galley(r.min, g, color);
}

/// Domain-specific "x"/"p" helpers for status lines.
pub fn var(d: Domain) -> &'static str {
    d.var()
}

/// Expectation value ⟨E⟩ = Σ|c|²E / Σ|c|² from coefficients and energies.
pub fn mean_energy(coeffs: &[C64], energies: &[f64]) -> f64 {
    let s: f64 = coeffs.iter().map(|c| c.norm_sqr()).sum();
    if s <= 0.0 {
        return 0.0;
    }
    coeffs.iter().zip(energies).map(|(c, e)| c.norm_sqr() * e).sum::<f64>() / s
}

/// Sets coefficients with |c|² below 10⁻⁷ to zero (numerical leakage between
/// finite-difference eigenstates and the split-step propagation), so that the
/// ladder does not draw phasors for levels that are not really occupied.
pub fn clean(mut c: Vec<C64>) -> Vec<C64> {
    for z in &mut c {
        if z.norm_sqr() < 1e-7 {
            *z = C64::new(0.0, 0.0);
        }
    }
    c
}

/// Projections c_n = ⟨ψ_n|ψ⟩ of a wave on real basis states.
pub fn project(states: &[Vec<f64>], wave: &Wave1D) -> Vec<C64> {
    let dx = wave.grid.dx;
    states
        .iter()
        .map(|s| s.iter().zip(&wave.psi).map(|(a, b)| b * *a).sum::<C64>() * dx)
        .collect()
}

/// ⟨x⟩, σₓ from ψ and ⟨p⟩, σₚ from an already computed Φ (no extra FFT;
/// the x–p covariance is left at 0).
pub fn quick_moments(wave: &Wave1D, mom: &qb_core::num::Momentum) -> qb_core::num::Moments {
    let g = wave.grid;
    let (mut s0, mut s1, mut s2) = (0.0, 0.0, 0.0);
    for (i, c) in wave.psi.iter().enumerate() {
        let w = c.norm_sqr();
        let x = g.x(i);
        s0 += w;
        s1 += w * x;
        s2 += w * x * x;
    }
    let (x, x2) = if s0 > 0.0 { (s1 / s0, s2 / s0) } else { (0.0, 0.0) };
    let (_, p, p2) = mom.moments();
    qb_core::num::Moments {
        norm: s0 * g.dx,
        x,
        x2,
        sigma_x: (x2 - x * x).max(0.0).sqrt(),
        p,
        p2,
        sigma_p: (p2 - p * p).max(0.0).sqrt(),
        cov_xp: 0.0,
    }
}

/// One eigenfunction drawn on its level (real ψₙ, as a line).
pub struct LevelCurve<'a> {
    pub psi: &'a [f64],
    pub energy: f64,
    /// energy units per unit of ψ
    pub scale: f64,
    pub label: String,
    pub color: egui::Color32,
    pub dashed: bool,
}

/// An energy-axis plot of ψ(x) in a potential, with the levels between their
/// turning points, the state drawn on the baseline ⟨E⟩, eigenfunctions on
/// their levels, and (optionally) a [`LevelLadder`] lined up on the right.
pub struct EnergyView<'a> {
    pub id: &'a str,
    pub xs: &'a [f64],
    pub v: &'a [f64],
    pub psi: &'a [C64],
    pub style: WaveStyle,
    /// where the state sits (usually ⟨E⟩)
    pub baseline: f64,
    /// factor for the plotted quantity of `style` (see [`energy_curve_scale`])
    pub scale: f64,
    pub x_range: (f64, f64),
    pub e_range: (f64, f64),
    pub height: f32,
    /// levels drawn in the plot (between turning points) and listed in the ladder
    pub energies: &'a [f64],
    /// c_n(t) for the ladder (None: no ladder)
    pub coeffs: Option<&'a [C64]>,
    pub labels: Option<&'a [String]>,
    pub selected: Option<usize>,
    pub ladder_width: f32,
    pub curves: Vec<LevelCurve<'a>>,
    pub markers: Vec<qb_core::widgets::Marker>,
    /// left-drag is left to the scene (drawing V); pan with the right button
    pub drawable: bool,
}

impl<'a> EnergyView<'a> {
    pub fn show(self, ui: &mut egui::Ui) -> (qb_core::widgets::PlotResponse, Option<LadderResponse>) {
        use qb_core::widgets::{WaveCurve, WavePlot, plot_dashed, plot_line};
        let look = Look::of(ui);
        let gap = look.px(6.0);
        let ladder_w = if self.coeffs.is_some() { self.ladder_width } else { 0.0 };
        let mut out = None;
        let mut ladder = None;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = gap;
            let w = ui.available_width() - ladder_w - if ladder_w > 0.0 { gap } else { 0.0 };
            let mut plot = WavePlot::new((self.id, "energy"))
                .domain(Domain::Position)
                .style(self.style)
                .energy_axis(true)
                .x_range(self.x_range.0, self.x_range.1)
                .y_range(self.e_range.0, self.e_range.1)
                .potential(self.xs, self.v, [self.e_range.0, self.e_range.1])
                .height(self.height)
                .width(w.max(100.0))
                .left_drag_pans(!self.drawable)
                .curve(WaveCurve::new(self.xs, self.psi).baseline(self.baseline).scale(self.scale));
            for m in self.markers {
                plot = plot.marker(m);
            }
            let resp = plot.show(ui);
            let tr = resp.transform;
            let painter = resp.painter(ui);
            draw_levels(
                &painter,
                &tr,
                self.xs,
                self.v,
                self.energies,
                look.stroke(1.0, look.level().gamma_multiply(0.8)),
                false,
            );
            for c in &self.curves {
                // only where ψₙ is not negligible (its flat tails would look like extra levels)
                let max = c.psi.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                let ys: Vec<f64> = c
                    .psi
                    .iter()
                    .map(|p| if p.abs() > 0.01 * max { c.energy + c.scale * p } else { f64::NAN })
                    .collect();
                let stroke = look.stroke(1.6, c.color);
                if c.dashed {
                    plot_dashed(&painter, &tr, self.xs, &ys, stroke, 6.0);
                } else {
                    plot_line(&painter, &tr, self.xs, &ys, stroke);
                }
                if !c.label.is_empty() {
                    painter.text(
                        pos2(tr.rect.left() + look.px(6.0), tr.y_to_screen(c.energy) - 2.0),
                        Align2::LEFT_BOTTOM,
                        &c.label,
                        look.font(12.0),
                        c.color,
                    );
                }
            }
            if let Some(coeffs) = self.coeffs {
                // same outer height and row as the plot, energy range matched to its axis
                let (lo, hi) = ladder_range_for(&tr, &look);
                let (_, _, mt, mb) = PlotFrame::margins(&look, false);
                let mut l = LevelLadder::new(format!("{}_ladder", self.id), self.energies)
                    .coeffs(coeffs)
                    .selected(self.selected)
                    .energy_range(lo, hi)
                    .size(egui::vec2(ladder_w, tr.rect.height() + mt + mb));
                if let Some(lb) = self.labels {
                    l = l.labels(lb);
                }
                ladder = Some(l.show(ui));
            }
            out = Some(resp);
        });
        (out.expect("plot shown"), ladder)
    }
}

/// Width for a plot stacked under an [`EnergyView`] with a ladder of width
/// `ladder_width` (x + p view), so that both plot frames have the same width and
/// left edge; the strip beside it under the ladder stays empty.
pub fn width_beside_ladder(ui: &egui::Ui, ladder_width: f32) -> f32 {
    let look = Look::of(ui);
    (ui.available_width() - ladder_width - look.px(6.0)).max(100.0)
}

/// Subscript digits for level labels: 12 → "₁₂".
pub fn subscript(n: usize) -> String {
    const D: [char; 10] = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
    n.to_string().chars().map(|c| D[c.to_digit(10).unwrap_or(0) as usize]).collect()
}

/// Level probabilities cut where the remaining tail is below 10⁻⁴ (at least 5 levels),
/// with labels n (only every 5th when there are many, so that they stay readable).
pub fn level_outcomes(probs: &[f64]) -> (Vec<f64>, Vec<String>) {
    let total: f64 = probs.iter().sum();
    let mut acc = 0.0;
    let mut count = probs.len();
    for (k, p) in probs.iter().enumerate() {
        acc += p;
        if acc >= (1.0 - 1e-4) * total {
            count = k + 1;
            break;
        }
    }
    let count = count.max(5).min(probs.len());
    let sparse = count > 16;
    let names = (0..count)
        .map(|k| if !sparse || k % 5 == 0 { k.to_string() } else { String::new() })
        .collect();
    (probs[..count].to_vec(), names)
}
