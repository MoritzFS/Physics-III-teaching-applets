//! `LevelLadder`: energy levels with |c_n|² bars and rotating phasors c_n(t).

use eframe::egui::{self, Align2, Id, Rect, Sense, Stroke, StrokeKind, pos2, vec2};

use super::color::phase_color;
use super::plot::{PlotTransform, nice_ticks, tick_label};
use crate::{C64, Look};

/// What happened on the ladder this frame.
pub struct LadderResponse {
    pub response: egui::Response,
    /// level clicked this frame
    pub clicked: Option<usize>,
    /// level under the mouse
    pub hovered: Option<usize>,
    /// energy ↔ screen mapping (x of the transform is 0 … 1 across the widget)
    pub transform: PlotTransform,
}

/// Clickable energy levels; optional |c_n|² bars and phasors for c_n(t).
///
/// ```ignore
/// let r = LevelLadder::new("ho_levels", &basis.energies)
///     .coeffs(&expansion.coeffs())     // c_n(t): bar = |c_n|², arrow = arg c_n
///     .selected(Some(n))
///     .energy_range(0.0, 10.0)         // same as the WavePlot's energy axis, so they line up
///     .size(vec2(180.0, 300.0))
///     .show(ui);
/// if let Some(n) = r.clicked { … }
/// ```
pub struct LevelLadder<'a> {
    id: Id,
    energies: &'a [f64],
    coeffs: Option<&'a [C64]>,
    labels: Option<&'a [String]>,
    selected: Option<usize>,
    range: Option<[f64; 2]>,
    size: Option<egui::Vec2>,
    phasors: bool,
}

impl<'a> LevelLadder<'a> {
    pub fn new(id: impl std::hash::Hash + std::fmt::Debug, energies: &'a [f64]) -> Self {
        LevelLadder {
            id: Id::new(id),
            energies,
            coeffs: None,
            labels: None,
            selected: None,
            range: None,
            size: None,
            phasors: true,
        }
    }
    /// c_n at the current time: bar length |c_n|², phasor angle arg c_n.
    pub fn coeffs(mut self, c: &'a [C64]) -> Self {
        self.coeffs = Some(c);
        self
    }
    /// Per-level labels (default "n = 0", "n = 1", …).
    pub fn labels(mut self, l: &'a [String]) -> Self {
        self.labels = Some(l);
        self
    }
    pub fn selected(mut self, s: Option<usize>) -> Self {
        self.selected = s;
        self
    }
    /// Energy axis range (default: the levels with some margin).
    pub fn energy_range(mut self, lo: f64, hi: f64) -> Self {
        self.range = Some([lo, hi]);
        self
    }
    pub fn size(mut self, s: egui::Vec2) -> Self {
        self.size = Some(s);
        self
    }
    /// Draw the rotating phasors (default true when coefficients are given).
    pub fn phasors(mut self, yes: bool) -> Self {
        self.phasors = yes;
        self
    }

    pub fn show(self, ui: &mut egui::Ui) -> LadderResponse {
        let look = Look::of(ui);
        let size = self.size.unwrap_or_else(|| ui.available_size_before_wrap());
        let (outer, _) = ui.allocate_exact_size(size, Sense::hover());
        let response = ui.interact(outer, self.id, Sense::click());
        let (ml, mt, mb) = (look.px(40.0), look.px(8.0), look.px(8.0));
        let rect = Rect::from_min_max(
            pos2(outer.left() + ml, outer.top() + mt),
            pos2(outer.right() - 4.0, outer.bottom() - mb),
        );
        let [lo, hi] = self.range.unwrap_or_else(|| {
            let lo = self.energies.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = self.energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let pad = 0.08 * (hi - lo).max(1e-9);
            [lo - pad, hi + pad]
        });
        let tr = PlotTransform {
            rect,
            x: [0.0, 1.0],
            y: [lo, hi],
        };
        let painter = ui.painter();
        painter.rect_filled(rect.expand(2.0), 2.0, look.plot_bg());
        painter.rect_stroke(rect.expand(2.0), 2.0, Stroke::new(1.0, look.plot_frame()), StrokeKind::Inside);
        let (ticks, step) = nice_ticks(lo, hi, (rect.height() / look.px(45.0)).max(2.0) as usize);
        for t in ticks {
            let sy = tr.y_to_screen(t);
            painter.text(
                pos2(rect.left() - 4.0, sy),
                Align2::RIGHT_CENTER,
                tick_label(t, step),
                look.font(11.0),
                look.weak_text(),
            );
        }
        painter.text(
            rect.left_top() + vec2(3.0, 1.0),
            Align2::LEFT_TOP,
            "E",
            look.font(12.5),
            look.text(),
        );

        let hover_y = response.hover_pos().map(|p| p.y);
        let visible: Vec<usize> = (0..self.energies.len())
            .filter(|&n| self.energies[n] >= lo && self.energies[n] <= hi)
            .collect();
        // nearest level to the mouse, within a few points
        let hovered = hover_y.and_then(|y| {
            visible
                .iter()
                .map(|&n| (n, (tr.y_to_screen(self.energies[n]) - y).abs()))
                .filter(|(_, d)| *d < look.px(8.0))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(n, _)| n)
        });
        let clicked = if response.clicked() { hovered } else { None };

        let max_c = self
            .coeffs
            .map_or(1.0, |c| c.iter().map(|z| z.norm()).fold(0.0, f64::max).max(1e-12));
        // phasor radius from the level spacing
        let min_gap = visible
            .windows(2)
            .map(|w| (tr.y_to_screen(self.energies[w[0]]) - tr.y_to_screen(self.energies[w[1]])).abs())
            .fold(f32::INFINITY, f32::min);
        let radius = (min_gap * 0.42).clamp(3.0, look.px(13.0));
        let line_w = rect.width() * 0.42;
        let bar_x0 = rect.left() + line_w + look.px(8.0);
        let bar_w = (rect.right() - bar_x0 - 2.0 * radius - look.px(10.0)).max(10.0);
        for &n in &visible {
            let sy = tr.y_to_screen(self.energies[n]);
            let sel = self.selected == Some(n);
            let hov = hovered == Some(n);
            let col = if sel {
                look.accent()
            } else if hov {
                look.text()
            } else {
                look.level()
            };
            painter.line_segment(
                [pos2(rect.left() + 4.0, sy), pos2(rect.left() + line_w, sy)],
                look.stroke(if sel { 2.5 } else { 1.5 }, col),
            );
            if radius > 5.0 || sel || hov {
                let label = self.labels.and_then(|l| l.get(n).cloned()).unwrap_or_else(|| format!("{n}"));
                painter.text(pos2(rect.left() + 6.0, sy - 1.0), Align2::LEFT_BOTTOM, label, look.font(10.5), col);
            }
            if let Some(cs) = self.coeffs
                && let Some(c) = cs.get(n)
            {
                let p = c.norm_sqr();
                let w = p as f32 * bar_w;
                let h = (radius * 1.1).max(3.0);
                let bar = Rect::from_min_size(pos2(bar_x0, sy - 0.5 * h), vec2(w.max(0.0), h));
                painter.rect_filled(bar, 1.0, look.counts().gamma_multiply(0.85));
                if self.phasors && c.norm() > 1e-6 {
                    let center = pos2(bar_x0 + bar_w + radius + look.px(6.0), sy);
                    painter.circle_stroke(center, radius, Stroke::new(1.0, look.plot_frame()));
                    let len = radius * (c.norm() / max_c).sqrt() as f32;
                    let a = c.arg() as f32;
                    let tip = center + vec2(a.cos(), -a.sin()) * len;
                    painter.line_segment([center, tip], look.stroke(1.8, phase_color(a, 1.0)));
                    painter.circle_filled(tip, look.px(1.8), phase_color(a, 1.0));
                }
            }
        }
        if let Some(n) = hovered {
            let mut text = format!("E = {:.4}", self.energies[n]);
            if let Some(c) = self.coeffs.and_then(|c| c.get(n)) {
                text += &format!("\n|c|² = {:.4}\narg c = {:.2}π", c.norm_sqr(), c.arg() / std::f64::consts::PI);
            }
            response.clone().on_hover_text(text);
        }
        LadderResponse {
            response,
            clicked,
            hovered,
            transform: tr,
        }
    }
}
