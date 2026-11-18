//! `TimeSeries` + `TimeSeriesPlot`: quantities against time (σ_x(t), P(t), fringes, …).

use eframe::egui::{self, Align2, Color32, pos2};

use super::plot::{PanZoom, PlotFrame, PlotResponse, plot_dashed, plot_line};
use crate::Look;

/// A recorded series of (t, y) points.
#[derive(Clone, Debug, Default)]
pub struct TimeSeries {
    pub name: String,
    /// colour (default: the look's series palette by position)
    pub color: Option<Color32>,
    pub points: Vec<[f64; 2]>,
    /// oldest points are dropped beyond this (0 = unlimited)
    pub max_points: usize,
}

impl TimeSeries {
    pub fn new(name: impl Into<String>) -> Self {
        TimeSeries {
            name: name.into(),
            color: None,
            points: vec![],
            max_points: 20_000,
        }
    }
    pub fn with_color(mut self, c: Color32) -> Self {
        self.color = Some(c);
        self
    }
    /// Appends a point. A time earlier than the last point (e.g. after a reset) clears the series first.
    pub fn push(&mut self, t: f64, y: f64) {
        if let Some(last) = self.points.last()
            && t < last[0]
        {
            self.points.clear();
        }
        self.points.push([t, y]);
        if self.max_points > 0 && self.points.len() > self.max_points {
            let extra = self.points.len() - self.max_points;
            self.points.drain(..extra);
        }
    }
    pub fn clear(&mut self) {
        self.points.clear();
    }
    pub fn last(&self) -> Option<[f64; 2]> {
        self.points.last().copied()
    }
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// Which time window is shown.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TimeWindow {
    /// from the first to the last point (or `now`)
    All,
    /// the last `w` time units (scrolling)
    Last(f64),
    /// a fixed interval
    Fixed(f64, f64),
}

/// A named, coloured reference function y = f(t).
type RefFn<'a> = (String, Color32, &'a dyn Fn(f64) -> f64);

/// Line plot of [`TimeSeries`], with optional reference functions (dashed,
/// e.g. the analytic σ(t)), horizontal reference lines (e.g. ħ/2) and event
/// lines (e.g. measurement times).
///
/// ```ignore
/// TimeSeriesPlot::new("sigma_t")
///     .series(&self.sigma_x)
///     .function("analytic", look.prediction(), &|t| free_gaussian_sigma(s0, t))
///     .hline(0.5, "ħ/2")
///     .window(TimeWindow::All)
///     .now(self.t)
///     .labels("t", "σ")
///     .height(150.0)
///     .show(ui);
/// ```
pub struct TimeSeriesPlot<'a> {
    id: egui::Id,
    series: Vec<&'a TimeSeries>,
    functions: Vec<RefFn<'a>>,
    hlines: Vec<(f64, String)>,
    events: Vec<(f64, String)>,
    window: TimeWindow,
    y_range: Option<[f64; 2]>,
    now: Option<f64>,
    x_label: String,
    y_label: String,
    height: Option<f32>,
    width: Option<f32>,
}

impl<'a> TimeSeriesPlot<'a> {
    pub fn new(id: impl std::hash::Hash + std::fmt::Debug) -> Self {
        TimeSeriesPlot {
            id: egui::Id::new(id),
            series: vec![],
            functions: vec![],
            hlines: vec![],
            events: vec![],
            window: TimeWindow::All,
            y_range: None,
            now: None,
            x_label: "t".into(),
            y_label: String::new(),
            height: None,
            width: None,
        }
    }
    pub fn series(mut self, s: &'a TimeSeries) -> Self {
        self.series.push(s);
        self
    }
    /// A reference curve y = f(t), drawn dashed over the visible window.
    pub fn function(mut self, name: impl Into<String>, color: Color32, f: &'a dyn Fn(f64) -> f64) -> Self {
        self.functions.push((name.into(), color, f));
        self
    }
    /// Horizontal reference line.
    pub fn hline(mut self, y: f64, label: impl Into<String>) -> Self {
        self.hlines.push((y, label.into()));
        self
    }
    /// Vertical event line at time t (e.g. "measured").
    pub fn event(mut self, t: f64, label: impl Into<String>) -> Self {
        self.events.push((t, label.into()));
        self
    }
    pub fn window(mut self, w: TimeWindow) -> Self {
        self.window = w;
        self
    }
    /// Fixed y range (default: fitted to the data, reference curves and lines).
    pub fn y_range(mut self, lo: f64, hi: f64) -> Self {
        self.y_range = Some([lo, hi]);
        self
    }
    /// The current time: extends the window to it and draws a marker.
    pub fn now(mut self, t: f64) -> Self {
        self.now = Some(t);
        self
    }
    pub fn labels(mut self, x: impl Into<String>, y: impl Into<String>) -> Self {
        self.x_label = x.into();
        self.y_label = y.into();
        self
    }
    pub fn height(mut self, h: f32) -> Self {
        self.height = Some(h);
        self
    }
    pub fn width(mut self, w: f32) -> Self {
        self.width = Some(w);
        self
    }

    pub fn show(self, ui: &mut egui::Ui) -> PlotResponse {
        let look = Look::of(ui);
        let t_first = self
            .series
            .iter()
            .filter_map(|s| s.points.first())
            .map(|p| p[0])
            .fold(f64::INFINITY, f64::min);
        let mut t_last = self
            .series
            .iter()
            .filter_map(|s| s.points.last())
            .map(|p| p[0])
            .fold(f64::NEG_INFINITY, f64::max);
        if let Some(n) = self.now {
            t_last = t_last.max(n);
        }
        let (t0, t1) = match self.window {
            TimeWindow::Fixed(a, b) => (a, b),
            TimeWindow::Last(w) => (t_last - w, t_last),
            TimeWindow::All => {
                let a = if t_first.is_finite() { t_first.min(0.0) } else { 0.0 };
                (a, if t_last.is_finite() { t_last.max(a + 1.0) } else { a + 1.0 })
            }
        };
        let (t0, t1) = if t1 > t0 { (t0, t1) } else { (t0, t0 + 1.0) };
        let samples = 200;
        let fvals: Vec<Vec<[f64; 2]>> = self
            .functions
            .iter()
            .map(|(_, _, f)| {
                (0..=samples)
                    .map(|k| t0 + (t1 - t0) * k as f64 / samples as f64)
                    .map(|t| [t, f(t)])
                    .collect()
            })
            .collect();
        let [ylo, yhi] = self.y_range.unwrap_or_else(|| {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            let mut add = |y: f64| {
                if y.is_finite() {
                    lo = lo.min(y);
                    hi = hi.max(y);
                }
            };
            for s in &self.series {
                s.points.iter().filter(|p| p[0] >= t0 && p[0] <= t1).for_each(|p| add(p[1]));
            }
            fvals.iter().flatten().for_each(|p| add(p[1]));
            self.hlines.iter().for_each(|(y, _)| add(*y));
            if !lo.is_finite() {
                return [0.0, 1.0];
            }
            // extra room at the top for the legend
            let pad = 0.1 * (hi - lo).max(1e-9 + 0.05 * hi.abs());
            [if lo >= 0.0 && lo - pad < 0.0 { 0.0 } else { lo - pad }, hi + 2.5 * pad]
        });
        let mut frame = PlotFrame::new(self.id)
            .x_range(t0, t1)
            .y_range(ylo, yhi)
            .labels(self.x_label.clone(), self.y_label.clone())
            .pan_zoom(PanZoom::None);
        if let Some(h) = self.height {
            frame = frame.height(h);
        }
        if let Some(w) = self.width {
            frame = frame.width(w);
        }
        frame.show(ui, |painter, tr| {
            let r = tr.rect;
            for (t, label) in &self.events {
                let sx = tr.x_to_screen(*t);
                painter.line_segment(
                    [pos2(sx, r.top()), pos2(sx, r.bottom())],
                    look.stroke(1.0, look.measure().gamma_multiply(0.6)),
                );
                if !label.is_empty() {
                    painter.text(
                        pos2(sx + 2.0, r.bottom() - 2.0),
                        Align2::LEFT_BOTTOM,
                        label,
                        look.font(10.0),
                        look.measure(),
                    );
                }
            }
            for (y, label) in &self.hlines {
                let sy = tr.y_to_screen(*y);
                plot_dashed(painter, tr, &[t0, t1], &[*y, *y], look.stroke(1.0, look.weak_text()), 5.0);
                painter.text(
                    pos2(r.left() + 4.0, sy - 1.0),
                    Align2::LEFT_BOTTOM,
                    label,
                    look.font(11.0),
                    look.weak_text(),
                );
            }
            for (i, s) in self.series.iter().enumerate() {
                let col = s.color.unwrap_or(look.series(i));
                let xs: Vec<f64> = s.points.iter().map(|p| p[0]).collect();
                let ys: Vec<f64> = s.points.iter().map(|p| p[1]).collect();
                plot_line(painter, tr, &xs, &ys, look.stroke(1.7, col));
                if let Some(p) = s.points.last() {
                    painter.circle_filled(tr.to_screen(p[0], p[1]), look.px(2.8), col);
                }
            }
            for ((_, col, _), pts) in self.functions.iter().zip(&fvals) {
                let xs: Vec<f64> = pts.iter().map(|p| p[0]).collect();
                let ys: Vec<f64> = pts.iter().map(|p| p[1]).collect();
                plot_dashed(painter, tr, &xs, &ys, look.stroke(1.3, *col), 6.0);
            }
            if let Some(n) = self.now {
                let sx = tr.x_to_screen(n);
                painter.line_segment([pos2(sx, r.top()), pos2(sx, r.bottom())], egui::Stroke::new(1.0, look.axis()));
            }
            // legend
            let mut x = r.right() - look.px(8.0);
            let y = r.top() + look.px(5.0);
            let items: Vec<(String, Color32)> = self
                .series
                .iter()
                .enumerate()
                .map(|(i, s)| (s.name.clone(), s.color.unwrap_or(look.series(i))))
                .chain(self.functions.iter().map(|(n, c, _)| (format!("{n} (dashed)"), *c)))
                .collect();
            // lay out right to left, then draw on a translucent box
            let mut placed = vec![];
            for (name, col) in items.iter().rev() {
                if name.is_empty() {
                    continue;
                }
                let g = painter.layout_no_wrap(name.clone(), look.font(11.5), *col);
                let rect = Align2::RIGHT_TOP.anchor_size(pos2(x, y), g.size());
                x = rect.left() - look.px(12.0);
                placed.push((rect, g, *col));
            }
            if let Some(all) = placed.iter().map(|p| p.0).reduce(|a, b| a.union(b)) {
                painter.rect_filled(all.expand(look.px(3.0)), 3.0, look.plot_bg().gamma_multiply(0.85));
            }
            for (rect, g, col) in placed {
                painter.galley(rect.min, g, col);
            }
        })
    }
}
