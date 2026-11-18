//! `HistogramPlot`: measured counts against the predicted probabilities.

use eframe::egui::{self, Align2, Rect, Shape, pos2, vec2};

use super::plot::{PanZoom, PlotFrame, PlotResponse};
use crate::Look;
use crate::num::{Bins, ShotHistogram};

/// How the bars are scaled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HistScale {
    /// counts / N (sums to 1)
    #[default]
    Fraction,
    /// counts / (N · bin width): comparable with a probability density |ψ|²
    Density,
    /// raw counts
    Counts,
}

/// Bars of a [`ShotHistogram`] with the prediction on top.
///
/// The prediction is the probability of each bin (e.g. from
/// [`ShotHistogram::predict_from_density`] or |c_n|²). It is drawn as a line
/// (binned) or as markers (discrete), with a faint band of ±1 standard
/// deviation √(N p (1 − p)) of the counts, so students see what "agreement" means.
///
/// ```ignore
/// let pred = hist.predict_from_density(&wave.xs(), &wave.density());
/// HistogramPlot::new("x_hist", &hist).predicted(&pred).x_label("x").height(160.0).show(ui);
/// ```
pub struct HistogramPlot<'a> {
    id: egui::Id,
    hist: &'a ShotHistogram,
    predicted: Option<&'a [f64]>,
    labels: Option<&'a [String]>,
    x_label: String,
    scale: HistScale,
    height: Option<f32>,
    width: Option<f32>,
    y_max: Option<f64>,
}

impl<'a> HistogramPlot<'a> {
    pub fn new(id: impl std::hash::Hash + std::fmt::Debug, hist: &'a ShotHistogram) -> Self {
        HistogramPlot {
            id: egui::Id::new(id),
            hist,
            predicted: None,
            labels: None,
            x_label: String::new(),
            scale: HistScale::Fraction,
            height: None,
            width: None,
            y_max: None,
        }
    }
    /// Predicted probability of each bin (same length as the histogram).
    pub fn predicted(mut self, p: &'a [f64]) -> Self {
        self.predicted = Some(p);
        self
    }
    /// Labels of discrete outcomes (e.g. "↑", "↓" or "n = 0" …).
    pub fn labels(mut self, l: &'a [String]) -> Self {
        self.labels = Some(l);
        self
    }
    pub fn x_label(mut self, l: impl Into<String>) -> Self {
        self.x_label = l.into();
        self
    }
    pub fn scale(mut self, s: HistScale) -> Self {
        self.scale = s;
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
    /// Fixed top of the y axis (default: automatic).
    pub fn y_max(mut self, y: f64) -> Self {
        self.y_max = Some(y);
        self
    }

    pub fn show(self, ui: &mut egui::Ui) -> PlotResponse {
        let look = Look::of(ui);
        let h = self.hist;
        let n = h.total() as f64;
        let edges = h.edges();
        let widths: Vec<f64> = edges.windows(2).map(|w| w[1] - w[0]).collect();
        let factor = |i: usize| -> f64 {
            match self.scale {
                HistScale::Fraction => 1.0 / n.max(1.0),
                HistScale::Density => 1.0 / (n.max(1.0) * widths[i]),
                HistScale::Counts => 1.0,
            }
        };
        // prediction in the same units as the bars
        let pred_scale = |i: usize| -> f64 {
            match self.scale {
                HistScale::Fraction => 1.0,
                HistScale::Density => 1.0 / widths[i],
                HistScale::Counts => n,
            }
        };
        let bars: Vec<f64> = (0..h.len()).map(|i| h.counts[i] as f64 * factor(i)).collect();
        let preds: Option<Vec<f64>> = self
            .predicted
            .map(|p| (0..h.len()).map(|i| p.get(i).copied().unwrap_or(0.0) * pred_scale(i)).collect());
        let top = self.y_max.unwrap_or_else(|| {
            let m = bars
                .iter()
                .cloned()
                .fold(0.0, f64::max)
                .max(preds.as_ref().map_or(0.0, |p| p.iter().cloned().fold(0.0, f64::max)));
            if m > 0.0 { m * 1.25 } else { 1.0 }
        });
        let y_label = match self.scale {
            HistScale::Fraction => "fraction",
            HistScale::Density => "counts / (N·Δ)",
            HistScale::Counts => "counts",
        };
        let discrete = matches!(h.bins, Bins::Discrete { .. });
        let mut frame = PlotFrame::new(self.id)
            .x_range(edges[0], *edges.last().unwrap())
            .y_range(0.0, top)
            .labels(self.x_label.clone(), y_label)
            .pan_zoom(if discrete { PanZoom::None } else { PanZoom::X });
        if let Some(hh) = self.height {
            frame = frame.height(hh);
        }
        if let Some(w) = self.width {
            frame = frame.width(w);
        }
        let labels = self.labels;
        frame.show(ui, |painter, tr| {
            let gap = if discrete { 0.18 } else { 0.0 };
            for i in 0..h.len() {
                let (a, b) = (edges[i] + gap * widths[i], edges[i + 1] - gap * widths[i]);
                let r = Rect::from_min_max(tr.to_screen(a, bars[i]), tr.to_screen(b, 0.0));
                painter.rect_filled(r, 0.0, look.counts().gamma_multiply(0.8));
            }
            if let Some(p) = &preds {
                let pc = look.prediction();
                // ±1σ band of the counts for N shots
                if n > 0.0 {
                    for i in 0..h.len() {
                        let prob = self.predicted.unwrap().get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                        let sd = (n * prob * (1.0 - prob)).sqrt() * factor(i);
                        let (a, b) = (edges[i] + gap * widths[i], edges[i + 1] - gap * widths[i]);
                        let r = Rect::from_min_max(tr.to_screen(a, p[i] + sd), tr.to_screen(b, (p[i] - sd).max(0.0)));
                        painter.rect_filled(r, 0.0, pc.gamma_multiply(0.14));
                    }
                }
                if discrete {
                    for i in 0..h.len() {
                        let (a, b) = (edges[i] + 0.1 * widths[i], edges[i + 1] - 0.1 * widths[i]);
                        painter.line_segment([tr.to_screen(a, p[i]), tr.to_screen(b, p[i])], look.stroke(2.2, pc));
                    }
                } else {
                    let mut pts = Vec::with_capacity(2 * h.len());
                    for i in 0..h.len() {
                        pts.push(tr.to_screen(edges[i], p[i]));
                        pts.push(tr.to_screen(edges[i + 1], p[i]));
                    }
                    painter.add(Shape::line(pts, look.stroke(1.8, pc)));
                }
            }
            if discrete {
                for i in 0..h.len() {
                    let text = labels.and_then(|l| l.get(i).cloned()).unwrap_or_else(|| i.to_string());
                    let c = tr.to_screen(0.5 * (edges[i] + edges[i + 1]), 0.0);
                    painter.text(c - vec2(0.0, 3.0), Align2::CENTER_BOTTOM, text, look.font(12.0), look.text());
                }
            }
            // legend + N
            let r = tr.rect;
            let mut y = r.top() + look.px(6.0);
            let x = r.right() - look.px(8.0);
            let text = format!("N = {}", h.total());
            painter.text(pos2(x, y), Align2::RIGHT_TOP, text, look.font(12.0), look.text());
            y += look.px(16.0);
            painter.text(pos2(x, y), Align2::RIGHT_TOP, "measured", look.font(11.0), look.counts());
            if preds.is_some() {
                y += look.px(14.0);
                painter.text(pos2(x, y), Align2::RIGHT_TOP, "Born rule", look.font(11.0), look.prediction());
            }
            if h.outside > 0 {
                y += look.px(14.0);
                painter.text(
                    pos2(x, y),
                    Align2::RIGHT_TOP,
                    format!("{} outside", h.outside),
                    look.font(10.5),
                    look.weak_text(),
                );
            }
        })
    }
}
