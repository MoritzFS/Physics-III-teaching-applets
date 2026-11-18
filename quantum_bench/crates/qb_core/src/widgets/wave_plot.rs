//! `WavePlot`: a wavefunction (and the potential) in position or momentum space.

use eframe::egui::{self, Align2, Color32, Pos2, Shape, Stroke, pos2, vec2};
use serde::{Deserialize, Serialize};

use super::color::{phase_color_alpha, phase_wheel};
use super::plot::{PanZoom, PlotFrame, PlotResponse, PlotTransform, label_box, plot_line};
use crate::{C64, Look};

/// Position or momentum representation (sets the axis labels).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Domain {
    Position,
    Momentum,
}

impl Domain {
    /// "x" / "p"
    pub fn var(self) -> &'static str {
        match self {
            Domain::Position => "x",
            Domain::Momentum => "p",
        }
    }

    /// "ψ" / "Φ"
    pub fn func(self) -> &'static str {
        match self {
            Domain::Position => "ψ",
            Domain::Momentum => "Φ",
        }
    }
}

/// How a wavefunction is drawn: the display mode every 1D scene offers, for x
/// and p alike. Key V cycles through them ([`WaveStyle::next`]); show the
/// switch with [`crate::widgets::view_bar`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WaveStyle {
    /// |ψ|² filled, coloured by the phase arg ψ (the default)
    #[default]
    Density,
    /// Re ψ alone: the actual wave, carrier and group envelope ±|ψ| (faint)
    Real,
    /// Re ψ and Im ψ together, with the envelope ±|ψ|
    ReIm,
    /// |ψ|² filled in one plain colour, no phase
    Plain,
}

impl WaveStyle {
    pub const ALL: [WaveStyle; 4] = [WaveStyle::Density, WaveStyle::Real, WaveStyle::ReIm, WaveStyle::Plain];

    /// Short label for the switch, e.g. "|ψ|² + phase".
    pub fn label(self, d: Domain) -> String {
        let f = d.func();
        match self {
            WaveStyle::Density => format!("|{f}|² + phase"),
            WaveStyle::Real => format!("Re {f}"),
            WaveStyle::ReIm => format!("Re {f}, Im {f}"),
            WaveStyle::Plain => format!("|{f}|²"),
        }
    }

    /// The y-axis label for this style and domain.
    pub fn axis_label(self, d: Domain) -> String {
        let (f, v) = (d.func(), d.var());
        match self {
            WaveStyle::Density | WaveStyle::Plain => format!("|{f}({v})|²"),
            WaveStyle::Real => format!("Re {f}({v})"),
            WaveStyle::ReIm => format!("{f}({v})"),
        }
    }

    /// The plotted quantity is |ψ|² (true) or linear in ψ (false).
    pub fn is_quadratic(self) -> bool {
        matches!(self, WaveStyle::Density | WaveStyle::Plain)
    }

    /// The y range for this style, given the top of the |ψ|² axis: [0, top]
    /// for |ψ|², [−√top, √top] for Re/Im. Keep `top` fixed (or only growing)
    /// so that the scale does not jump between modes or while a packet spreads.
    pub fn y_range(self, density_top: f64) -> (f64, f64) {
        if self.is_quadratic() {
            (0.0, density_top)
        } else {
            (-density_top.sqrt(), density_top.sqrt())
        }
    }

    /// The next mode (key V).
    pub fn next(self) -> Self {
        let i = Self::ALL.iter().position(|s| *s == self).unwrap_or(0);
        Self::ALL[(i + 1) % Self::ALL.len()]
    }

    /// Parses "density", "re", "reim", "plain" (for `QB_ACTION=style_re` etc.).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "density" | "phase" => Some(WaveStyle::Density),
            "re" | "real" => Some(WaveStyle::Real),
            "reim" | "re_im" => Some(WaveStyle::ReIm),
            "plain" | "abs" => Some(WaveStyle::Plain),
            _ => None,
        }
    }

    /// The modes as a segmented switch. Returns true if the mode changed.
    pub fn selector(&mut self, ui: &mut egui::Ui, d: Domain) -> bool {
        let mut changed = false;
        segmented(ui, |ui| {
            for s in WaveStyle::ALL {
                if ui
                    .selectable_label(*self == s, s.label(d))
                    .on_hover_text("display mode (V cycles)")
                    .clicked()
                {
                    changed |= *self != s;
                    *self = s;
                }
            }
        });
        changed
    }
}

/// Lays out `add` as one group of adjacent buttons (a segmented control).
pub fn segmented<R>(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::same(2))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                add(ui)
            })
            .inner
        })
        .inner
}

/// The prominent display switch of the 1D scenes, for the top of the central
/// area: representation (x | p | x + p) and display mode (|ψ|² + phase | Re ψ |
/// Re ψ, Im ψ | |ψ|²). Returns true if anything changed.
pub fn view_bar(ui: &mut egui::Ui, view: &mut super::ReprView, style: &mut WaveStyle) -> bool {
    let look = Look::of(ui);
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("show").color(look.weak_text()));
        changed |= view.segmented(ui);
        ui.add_space(look.px(12.0));
        ui.label(egui::RichText::new("display").color(look.weak_text()));
        changed |= style.selector(ui, Domain::Position);
        ui.label(egui::RichText::new("V").monospace().color(look.weak_text()))
            .on_hover_text("key V cycles the display mode");
    });
    changed
}

/// One wavefunction to draw.
pub struct WaveCurve<'a> {
    pub xs: &'a [f64],
    pub psi: &'a [C64],
    /// vertical offset (e.g. E_n, to sit on an energy level)
    pub baseline: f64,
    /// factor applied to the plotted quantity (|ψ|², |ψ| or Re/Im)
    pub scale: f64,
    pub label: Option<String>,
    /// style for this curve (default: the plot's)
    pub style: Option<WaveStyle>,
    /// fill opacity for the phase-coloured styles
    pub alpha: f32,
    /// outline colour (default: the look's |ψ| colour)
    pub color: Option<Color32>,
}

impl<'a> WaveCurve<'a> {
    pub fn new(xs: &'a [f64], psi: &'a [C64]) -> Self {
        WaveCurve {
            xs,
            psi,
            baseline: 0.0,
            scale: 1.0,
            label: None,
            style: None,
            alpha: 0.9,
            color: None,
        }
    }
    pub fn baseline(mut self, b: f64) -> Self {
        self.baseline = b;
        self
    }
    pub fn scale(mut self, s: f64) -> Self {
        self.scale = s;
        self
    }
    pub fn label(mut self, l: impl Into<String>) -> Self {
        self.label = Some(l.into());
        self
    }
    pub fn style(mut self, s: WaveStyle) -> Self {
        self.style = Some(s);
        self
    }
    pub fn alpha(mut self, a: f32) -> Self {
        self.alpha = a;
        self
    }
    pub fn color(mut self, c: Color32) -> Self {
        self.color = Some(c);
        self
    }
}

#[derive(Clone, Debug)]
enum MarkerKind {
    VLine { x: f64 },
    HLine { y: f64 },
    Span { x0: f64, x1: f64 },
    Bracket { x0: f64, x1: f64, y: f64 },
    Point { x: f64, y: f64 },
}

/// Annotations on a plot: ⟨x⟩ lines, ±σ bands, width brackets, levels, points.
#[derive(Clone, Debug)]
pub struct Marker {
    kind: MarkerKind,
    label: String,
    color: Option<Color32>,
    dashed: bool,
}

impl Marker {
    fn new(kind: MarkerKind, label: impl Into<String>) -> Self {
        Marker {
            kind,
            label: label.into(),
            color: None,
            dashed: false,
        }
    }
    /// Vertical line at x (e.g. "⟨x⟩").
    pub fn vline(x: f64, label: impl Into<String>) -> Self {
        Self::new(MarkerKind::VLine { x }, label)
    }
    /// Horizontal line at y (e.g. an energy level "E₀").
    pub fn hline(y: f64, label: impl Into<String>) -> Self {
        Self::new(MarkerKind::HLine { y }, label)
    }
    /// Shaded band x0 … x1 (e.g. "⟨x⟩ ± σₓ", a detector region).
    pub fn span(x0: f64, x1: f64, label: impl Into<String>) -> Self {
        Self::new(MarkerKind::Span { x0, x1 }, label)
    }
    /// Width bracket |←→| from x0 to x1 at height y (data units), label above.
    pub fn bracket(x0: f64, x1: f64, y: f64, label: impl Into<String>) -> Self {
        Self::new(MarkerKind::Bracket { x0, x1, y }, label)
    }
    /// A dot at (x, y).
    pub fn point(x: f64, y: f64, label: impl Into<String>) -> Self {
        Self::new(MarkerKind::Point { x, y }, label)
    }
    pub fn color(mut self, c: Color32) -> Self {
        self.color = Some(c);
        self
    }
    pub fn dashed(mut self) -> Self {
        self.dashed = true;
        self
    }

    /// Draws the marker with a plot's transform (also usable on custom plots).
    pub fn paint(&self, painter: &egui::Painter, tr: &PlotTransform) {
        let look = Look::from_ctx(painter.ctx());
        let col = self.color.unwrap_or(look.accent());
        let r = tr.rect;
        let stroke = look.stroke(1.3, col);
        let font = look.font(12.0);
        let line = |a: Pos2, b: Pos2| {
            if self.dashed {
                painter.extend(Shape::dashed_line(&[a, b], stroke, 6.0, 4.0));
            } else {
                painter.line_segment([a, b], stroke);
            }
        };
        match self.kind {
            MarkerKind::VLine { x } => {
                let sx = tr.x_to_screen(x);
                line(pos2(sx, r.top()), pos2(sx, r.bottom()));
                if !self.label.is_empty() {
                    painter.text(pos2(sx + 3.0, r.top() + 3.0), Align2::LEFT_TOP, &self.label, font, col);
                }
            }
            MarkerKind::HLine { y } => {
                let sy = tr.y_to_screen(y);
                line(pos2(r.left(), sy), pos2(r.right(), sy));
                if !self.label.is_empty() {
                    painter.text(pos2(r.right() - 4.0, sy - 2.0), Align2::RIGHT_BOTTOM, &self.label, font, col);
                }
            }
            MarkerKind::Span { x0, x1 } => {
                let (a, b) = (tr.x_to_screen(x0.min(x1)), tr.x_to_screen(x0.max(x1)));
                let band = egui::Rect::from_min_max(pos2(a, r.top()), pos2(b, r.bottom()));
                painter.rect_filled(band, 0.0, col.gamma_multiply(0.12));
                for sx in [a, b] {
                    painter.extend(Shape::dashed_line(
                        &[pos2(sx, r.top()), pos2(sx, r.bottom())],
                        look.stroke(1.0, col.gamma_multiply(0.7)),
                        4.0,
                        4.0,
                    ));
                }
                if !self.label.is_empty() {
                    painter.text(pos2(0.5 * (a + b), r.bottom() - 3.0), Align2::CENTER_BOTTOM, &self.label, font, col);
                }
            }
            MarkerKind::Bracket { x0, x1, y } => {
                let sy = tr.y_to_screen(y);
                let (a, b) = (tr.x_to_screen(x0.min(x1)), tr.x_to_screen(x0.max(x1)));
                let h = look.px(5.0);
                painter.line_segment([pos2(a, sy), pos2(b, sy)], stroke);
                painter.line_segment([pos2(a, sy - h), pos2(a, sy + h)], stroke);
                painter.line_segment([pos2(b, sy - h), pos2(b, sy + h)], stroke);
                if !self.label.is_empty() {
                    label_box(painter, pos2(0.5 * (a + b), sy - h - 1.0), Align2::CENTER_BOTTOM, &self.label, col);
                }
            }
            MarkerKind::Point { x, y } => {
                let p = tr.to_screen(x, y);
                painter.circle_filled(p, look.px(3.5), col);
                if !self.label.is_empty() {
                    painter.text(p + vec2(5.0, -5.0), Align2::LEFT_BOTTOM, &self.label, font, col);
                }
            }
        }
    }
}

/// Plot of ψ(x) or Φ(p) with the potential, phase colouring, markers and pan/zoom.
///
/// ```ignore
/// let xs = wave.xs();
/// let resp = WavePlot::new("packet_x")
///     .domain(Domain::Position)
///     .style(WaveStyle::Density)
///     .x_range(-20.0, 20.0)
///     .y_range(0.0, 0.5)
///     .potential(&xs, &v, [0.0, 2.0])          // V from 0 … 2 spans the plot height (right axis)
///     .curve(WaveCurve::new(&xs, &wave.psi))
///     .marker(Marker::vline(m.x, "⟨x⟩"))
///     .marker(Marker::span(m.x - m.sigma_x, m.x + m.sigma_x, "±σₓ"))
///     .height(300.0)
///     .show(ui);
/// ```
///
/// With [`WavePlot::energy_axis`] the y axis is energy: V is drawn in its own
/// units and each curve sits on its `baseline` (e.g. ψ_n on E_n, scaled by `scale`).
pub struct WavePlot<'a> {
    frame: PlotFrame,
    domain: Domain,
    style: WaveStyle,
    y: [f64; 2],
    potential: Option<(&'a [f64], &'a [f64], [f64; 2])>,
    energy_axis: bool,
    curves: Vec<WaveCurve<'a>>,
    markers: Vec<Marker>,
    phase_legend: bool,
    y_label: Option<String>,
    x_label: Option<String>,
}

impl<'a> WavePlot<'a> {
    pub fn new(id: impl std::hash::Hash + std::fmt::Debug) -> Self {
        WavePlot {
            frame: PlotFrame::new(id),
            domain: Domain::Position,
            style: WaveStyle::Density,
            y: [0.0, 1.0],
            potential: None,
            energy_axis: false,
            curves: vec![],
            markers: vec![],
            phase_legend: true,
            y_label: None,
            x_label: None,
        }
    }
    pub fn domain(mut self, d: Domain) -> Self {
        self.domain = d;
        self
    }
    pub fn style(mut self, s: WaveStyle) -> Self {
        self.style = s;
        self
    }
    pub fn x_range(mut self, lo: f64, hi: f64) -> Self {
        self.frame = self.frame.x_range(lo, hi);
        self
    }
    pub fn y_range(mut self, lo: f64, hi: f64) -> Self {
        self.y = [lo, hi];
        self.frame = self.frame.y_range(lo, hi);
        self
    }
    pub fn height(mut self, h: f32) -> Self {
        self.frame = self.frame.height(h);
        self
    }
    pub fn width(mut self, w: f32) -> Self {
        self.frame = self.frame.width(w);
        self
    }
    pub fn pan_zoom(mut self, pz: PanZoom) -> Self {
        self.frame = self.frame.pan_zoom(pz);
        self
    }
    /// false: left-drag is free for the tab (e.g. drawing V); pan with the right button.
    pub fn left_drag_pans(mut self, yes: bool) -> Self {
        self.frame = self.frame.left_drag_pans(yes);
        self
    }
    /// V(x) sampled at `xs`. Without an energy axis, `range` = [V_lo, V_hi]
    /// is mapped onto the plot's y range and shown on a right-hand axis.
    pub fn potential(mut self, xs: &'a [f64], v: &'a [f64], range: [f64; 2]) -> Self {
        self.potential = Some((xs, v, range));
        self
    }
    /// The y axis is energy: V is drawn in its own units, curves on their baselines.
    pub fn energy_axis(mut self, yes: bool) -> Self {
        self.energy_axis = yes;
        self
    }
    pub fn curve(mut self, c: WaveCurve<'a>) -> Self {
        self.curves.push(c);
        self
    }
    pub fn marker(mut self, m: Marker) -> Self {
        self.markers.push(m);
        self
    }
    pub fn markers(mut self, ms: impl IntoIterator<Item = Marker>) -> Self {
        self.markers.extend(ms);
        self
    }
    /// Show the phase colour wheel (default true for the phase-coloured styles).
    pub fn phase_legend(mut self, yes: bool) -> Self {
        self.phase_legend = yes;
        self
    }
    /// Override the automatic y-axis label.
    pub fn y_label(mut self, l: impl Into<String>) -> Self {
        self.y_label = Some(l.into());
        self
    }
    /// Override the automatic x-axis label.
    pub fn x_label(mut self, l: impl Into<String>) -> Self {
        self.x_label = Some(l.into());
        self
    }

    pub fn show(self, ui: &mut egui::Ui) -> PlotResponse {
        let look = Look::of(ui);
        let WavePlot {
            mut frame,
            domain,
            style,
            y,
            potential,
            energy_axis,
            curves,
            markers,
            phase_legend,
            y_label,
            x_label,
        } = self;
        let y_label = y_label.unwrap_or_else(|| if energy_axis { "E".into() } else { style.axis_label(domain) });
        frame = frame.labels(x_label.unwrap_or_else(|| domain.var().into()), y_label);
        if let Some((_, _, range)) = potential
            && !energy_axis
        {
            frame = frame.right_axis("V", range);
        }
        let curves_drawn: Vec<WaveStyle> = curves.iter().map(|c| c.style.unwrap_or(style)).collect();
        frame.show(ui, |painter, tr| {
            // potential
            if let Some((xs, v, range)) = potential {
                let map = |v: f64| {
                    if energy_axis {
                        v
                    } else {
                        y[0] + (v - range[0]) / (range[1] - range[0]) * (y[1] - y[0])
                    }
                };
                let ys: Vec<f64> = v.iter().map(|&v| map(v)).collect();
                fill_below(painter, tr, xs, &ys, look.potential_fill());
                plot_line(painter, tr, xs, &ys, look.stroke(1.4, look.potential()));
            }
            // curves
            for (c, st) in curves.iter().zip(&curves_drawn) {
                draw_curve(painter, tr, c, *st, &look);
            }
            for m in &markers {
                m.paint(painter, tr);
            }
            // legends
            let r = tr.rect;
            if phase_legend && !curves.is_empty() && curves_drawn.contains(&WaveStyle::Density) {
                let rad = look.px(13.0);
                let c = pos2(r.right() - rad - look.px(14.0), r.top() + rad + look.px(6.0));
                phase_wheel(painter, c, rad, Some(&format!("arg {}", domain.func())));
            }
            let lines = curves_drawn.iter().find(|s| matches!(s, WaveStyle::Real | WaveStyle::ReIm));
            if let Some(st) = lines {
                let f = domain.func();
                let mut items = vec![(format!("±|{f}|"), look.weak_text())];
                if *st == WaveStyle::ReIm {
                    items.push((format!("Im {f}"), look.im()));
                }
                items.push((format!("Re {f}"), look.re()));
                let mut p = pos2(r.right() - look.px(8.0), r.top() + look.px(6.0));
                for (text, col) in items {
                    let rect = painter.text(p, Align2::RIGHT_TOP, text, look.font(12.5), col);
                    p.x = rect.left() - look.px(12.0);
                }
            }
            // curve labels at the left end of their baseline (energy plots)
            for c in &curves {
                if let Some(l) = &c.label {
                    let sy = tr.y_to_screen(c.baseline);
                    painter.text(
                        pos2(r.left() + 4.0, sy - 2.0),
                        Align2::LEFT_BOTTOM,
                        l,
                        look.font(11.5),
                        look.weak_text(),
                    );
                }
            }
        })
    }
}

/// Fills between the bottom of the plot and the curve.
fn fill_below(painter: &egui::Painter, tr: &PlotTransform, xs: &[f64], ys: &[f64], color: Color32) {
    let base = tr.rect.bottom();
    let mut mesh = egui::Mesh::default();
    for (&x, &y) in xs.iter().zip(ys) {
        if !(x.is_finite() && y.is_finite()) {
            continue;
        }
        let sx = tr.x_to_screen(x);
        let sy = tr.y_to_screen(y).clamp(tr.rect.top() - 10.0, tr.rect.bottom() + 10.0);
        let i = mesh.vertices.len() as u32;
        mesh.colored_vertex(pos2(sx, sy), color);
        mesh.colored_vertex(pos2(sx, base), color);
        if i >= 2 {
            mesh.add_triangle(i - 2, i - 1, i);
            mesh.add_triangle(i - 1, i, i + 1);
        }
    }
    painter.add(Shape::mesh(mesh));
}

/// Draws one wavefunction curve in the given style.
pub fn draw_curve(painter: &egui::Painter, tr: &PlotTransform, c: &WaveCurve, style: WaveStyle, look: &Look) {
    let n = c.xs.len().min(c.psi.len());
    if n < 2 {
        return;
    }
    // only the visible part (plus one point on each side)
    let (lo, hi) = (tr.x[0].min(tr.x[1]), tr.x[0].max(tr.x[1]));
    let first = c.xs[..n].partition_point(|&x| x < lo).saturating_sub(1);
    let last = (c.xs[..n].partition_point(|&x| x <= hi) + 1).min(n);
    let (xs, psi) = (&c.xs[first..last], &c.psi[first..last]);
    match style {
        WaveStyle::Density | WaveStyle::Plain => {
            let base_y = tr.y_to_screen(c.baseline);
            let plain = look.re().gamma_multiply(0.55 * c.alpha);
            let mut mesh = egui::Mesh::default();
            let mut outline = Vec::with_capacity(xs.len());
            for (k, (&x, z)) in xs.iter().zip(psi).enumerate() {
                let sx = tr.x_to_screen(x);
                let sy = tr
                    .y_to_screen(c.baseline + c.scale * z.norm_sqr())
                    .clamp(tr.rect.top() - 10.0, tr.rect.bottom() + 10.0);
                let col = if style == WaveStyle::Density {
                    phase_color_alpha(z.arg() as f32, c.alpha)
                } else {
                    plain
                };
                mesh.colored_vertex(pos2(sx, sy), col);
                mesh.colored_vertex(pos2(sx, base_y), col);
                if k > 0 {
                    let i = 2 * k as u32;
                    mesh.add_triangle(i - 2, i - 1, i);
                    mesh.add_triangle(i - 1, i, i + 1);
                }
                outline.push(pos2(sx, sy));
            }
            painter.add(Shape::mesh(mesh));
            let col = match style {
                WaveStyle::Plain => c.color.unwrap_or(look.re()),
                _ => c.color.unwrap_or(look.abs()).gamma_multiply(0.85),
            };
            painter.add(Shape::line(
                outline,
                look.stroke(if style == WaveStyle::Plain { 1.6 } else { 1.0 }, col),
            ));
        }
        WaveStyle::Real | WaveStyle::ReIm => {
            // faint band between −|ψ| and +|ψ|: the group envelope
            let band = look.weak_text().gamma_multiply(0.13);
            let mut mesh = egui::Mesh::default();
            for (k, (&x, z)) in xs.iter().zip(psi).enumerate() {
                let sx = tr.x_to_screen(x);
                let a = c.scale * z.norm();
                let clamp = |y: f32| y.clamp(tr.rect.top() - 10.0, tr.rect.bottom() + 10.0);
                mesh.colored_vertex(pos2(sx, clamp(tr.y_to_screen(c.baseline + a))), band);
                mesh.colored_vertex(pos2(sx, clamp(tr.y_to_screen(c.baseline - a))), band);
                if k > 0 {
                    let i = 2 * k as u32;
                    mesh.add_triangle(i - 2, i - 1, i);
                    mesh.add_triangle(i - 1, i, i + 1);
                }
            }
            painter.add(Shape::mesh(mesh));
            let ys = |f: &dyn Fn(&C64) -> f64| -> Vec<f64> { psi.iter().map(|z| c.baseline + c.scale * f(z)).collect() };
            let env = look.stroke(1.0, look.weak_text().gamma_multiply(0.7));
            plot_line(painter, tr, xs, &ys(&|z| z.norm()), env);
            plot_line(painter, tr, xs, &ys(&|z| -z.norm()), env);
            if c.baseline != 0.0 {
                let sy = tr.y_to_screen(c.baseline);
                painter.line_segment(
                    [pos2(tr.rect.left(), sy), pos2(tr.rect.right(), sy)],
                    Stroke::new(1.0, look.level()),
                );
            }
            if style == WaveStyle::ReIm {
                plot_line(painter, tr, xs, &ys(&|z| z.im), look.stroke(1.6, look.im()));
            }
            plot_line(painter, tr, xs, &ys(&|z| z.re), look.stroke(1.8, c.color.unwrap_or(look.re())));
        }
    }
}
