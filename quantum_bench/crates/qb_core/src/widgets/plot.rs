//! The plot frame shared by all plot widgets: axes, ticks, labels, pan/zoom,
//! and the data ↔ screen transform that tabs use for their own drawing and
//! mouse interaction.

use eframe::egui::{self, Align2, Color32, Id, Pos2, Rect, Response, Sense, Shape, Stroke, StrokeKind, Vec2, pos2, vec2};

use crate::Look;

/// Maps data coordinates (x, y) to screen points inside `rect` (y up).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlotTransform {
    /// the plot area on screen (without the axis margins)
    pub rect: Rect,
    /// visible data range in x: [left, right]
    pub x: [f64; 2],
    /// visible data range in y: [bottom, top]
    pub y: [f64; 2],
}

impl PlotTransform {
    pub fn x_to_screen(&self, x: f64) -> f32 {
        self.rect.left() + ((x - self.x[0]) / (self.x[1] - self.x[0])) as f32 * self.rect.width()
    }

    pub fn y_to_screen(&self, y: f64) -> f32 {
        self.rect.bottom() - ((y - self.y[0]) / (self.y[1] - self.y[0])) as f32 * self.rect.height()
    }

    pub fn to_screen(&self, x: f64, y: f64) -> Pos2 {
        pos2(self.x_to_screen(x), self.y_to_screen(y))
    }

    pub fn x_from_screen(&self, sx: f32) -> f64 {
        self.x[0] + ((sx - self.rect.left()) / self.rect.width()) as f64 * (self.x[1] - self.x[0])
    }

    pub fn y_from_screen(&self, sy: f32) -> f64 {
        self.y[0] + ((self.rect.bottom() - sy) / self.rect.height()) as f64 * (self.y[1] - self.y[0])
    }

    /// (x, y) in data coordinates of a screen point.
    pub fn from_screen(&self, p: Pos2) -> (f64, f64) {
        (self.x_from_screen(p.x), self.y_from_screen(p.y))
    }

    /// Data units per screen point in x.
    pub fn dx_per_point(&self) -> f64 {
        (self.x[1] - self.x[0]) / self.rect.width() as f64
    }

    /// Data units per screen point in y.
    pub fn dy_per_point(&self) -> f64 {
        (self.y[1] - self.y[0]) / self.rect.height() as f64
    }

    /// true if x is inside the visible x range.
    pub fn contains_x(&self, x: f64) -> bool {
        x >= self.x[0].min(self.x[1]) && x <= self.x[0].max(self.x[1])
    }
}

/// What the user may do with the view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanZoom {
    /// fixed view
    None,
    /// scroll / pinch zooms x around the cursor, drag pans x, double-click resets
    X,
    /// the same in x and y
    XY,
}

/// Result of showing a plot.
pub struct PlotResponse {
    /// hover, click and drag on the plot area
    pub response: Response,
    /// data ↔ screen mapping of the plot area
    pub transform: PlotTransform,
    /// data coordinates under the mouse
    pub hover: Option<(f64, f64)>,
}

impl PlotResponse {
    /// A painter clipped to the plot area, for custom overlays.
    pub fn painter(&self, ui: &egui::Ui) -> egui::Painter {
        ui.painter_at(self.transform.rect)
    }
}

/// Axis tick positions: about `target` "nice" values (1, 2, 5 × 10ⁿ) in [lo, hi].
pub fn nice_ticks(lo: f64, hi: f64, target: usize) -> (Vec<f64>, f64) {
    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    let range = (hi - lo).max(1e-300);
    let raw = range / target.max(1) as f64;
    let mag = 10f64.powf(raw.log10().floor());
    let step = [1.0, 2.0, 5.0, 10.0]
        .iter()
        .map(|m| m * mag)
        .find(|s| *s >= raw)
        .unwrap_or(10.0 * mag);
    let first = (lo / step).ceil() as i64;
    let last = (hi / step).floor() as i64;
    let ticks = (first..=last).map(|k| k as f64 * step).collect();
    (ticks, step)
}

/// Formats a tick value with as many decimals as the step needs.
pub fn tick_label(v: f64, step: f64) -> String {
    let decimals = (-step.log10().floor()).max(0.0) as usize;
    let v = if v.abs() < step * 1e-6 { 0.0 } else { v };
    if v.abs() >= 1e5 || (v != 0.0 && v.abs() < 1e-3 && decimals > 4) {
        format!("{v:.1e}")
    } else {
        format!("{v:.decimals$}")
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ViewState {
    defaults: ([f64; 2], [f64; 2]),
    x: [f64; 2],
    y: [f64; 2],
}

/// A plot area with axes. All plot widgets of qb_core are built on it, and
/// tabs can use it directly for custom plots:
///
/// ```ignore
/// let resp = PlotFrame::new("my_plot").x_range(0.0, 10.0).y_range(-1.0, 1.0)
///     .labels("t", "P(t)").height(200.0)
///     .show(ui, |painter, tr| {
///         painter.line_segment([tr.to_screen(0.0, 0.0), tr.to_screen(10.0, 1.0)], look.stroke(1.5, look.accent()));
///     });
/// if resp.response.dragged() { /* use resp.transform.from_screen(..) */ }
/// ```
///
/// The visible range starts at the given ranges and can be changed by the user
/// (see [`PanZoom`]); it resets when the given ranges change (e.g. a new preset)
/// or on double-click.
pub struct PlotFrame {
    id: Id,
    x: [f64; 2],
    y: [f64; 2],
    x_label: String,
    y_label: String,
    /// right-hand axis: label and the data range that maps onto [y0, y1]
    y2: Option<(String, [f64; 2])>,
    height: Option<f32>,
    width: Option<f32>,
    pan_zoom: PanZoom,
    left_drag_pans: bool,
    y_ticks: bool,
    grid: bool,
    zero_line: bool,
}

impl PlotFrame {
    pub fn new(id: impl std::hash::Hash + std::fmt::Debug) -> Self {
        PlotFrame {
            id: Id::new(id),
            x: [0.0, 1.0],
            y: [0.0, 1.0],
            x_label: String::new(),
            y_label: String::new(),
            y2: None,
            height: None,
            width: None,
            pan_zoom: PanZoom::X,
            left_drag_pans: true,
            y_ticks: true,
            grid: true,
            zero_line: true,
        }
    }

    /// Default visible x range.
    pub fn x_range(mut self, lo: f64, hi: f64) -> Self {
        self.x = [lo, if hi > lo { hi } else { lo + 1.0 }];
        self
    }

    /// Default visible y range.
    pub fn y_range(mut self, lo: f64, hi: f64) -> Self {
        self.y = [lo, if hi > lo { hi } else { lo + 1.0 }];
        self
    }

    /// Axis labels (x at the bottom right, y at the top left inside the plot).
    pub fn labels(mut self, x: impl Into<String>, y: impl Into<String>) -> Self {
        self.x_label = x.into();
        self.y_label = y.into();
        self
    }

    /// A second y axis on the right: `range` maps onto the default y range.
    pub fn right_axis(mut self, label: impl Into<String>, range: [f64; 2]) -> Self {
        self.y2 = Some((label.into(), range));
        self
    }

    /// Total height in points including the axes (default: all available height).
    pub fn height(mut self, h: f32) -> Self {
        self.height = Some(h);
        self
    }

    /// Total width in points (default: all available width).
    pub fn width(mut self, w: f32) -> Self {
        self.width = Some(w);
        self
    }

    pub fn pan_zoom(mut self, pz: PanZoom) -> Self {
        self.pan_zoom = pz;
        self
    }

    /// false: left-drag is left to the caller (e.g. for drawing a potential);
    /// panning then needs the right or middle mouse button.
    pub fn left_drag_pans(mut self, yes: bool) -> Self {
        self.left_drag_pans = yes;
        self
    }

    pub fn y_ticks(mut self, yes: bool) -> Self {
        self.y_ticks = yes;
        self
    }

    pub fn grid(mut self, yes: bool) -> Self {
        self.grid = yes;
        self
    }

    /// Draw the line y = 0 (default true).
    pub fn zero_line(mut self, yes: bool) -> Self {
        self.zero_line = yes;
        self
    }

    /// Margins around the plot area: (left, right, top, bottom).
    pub fn margins(look: &Look, right_axis: bool) -> (f32, f32, f32, f32) {
        let f = look.font_scale();
        (48.0 * f, if right_axis { 44.0 * f } else { 10.0 }, 6.0, 22.0 * f)
    }

    /// Lays out the plot, handles pan/zoom, draws the frame, then calls
    /// `content(painter, transform)` with a painter clipped to the plot area.
    pub fn show(self, ui: &mut egui::Ui, content: impl FnOnce(&egui::Painter, &PlotTransform)) -> PlotResponse {
        let look = Look::of(ui);
        let avail = ui.available_size_before_wrap();
        let size = vec2(self.width.unwrap_or(avail.x).max(60.0), self.height.unwrap_or(avail.y).max(40.0));
        let (outer, _) = ui.allocate_exact_size(size, Sense::hover());
        let (ml, mr, mt, mb) = Self::margins(&look, self.y2.is_some());
        let rect = Rect::from_min_max(
            pos2(outer.left() + ml, outer.top() + mt),
            pos2(outer.right() - mr, outer.bottom() - mb),
        );
        let rect = if rect.width() < 10.0 || rect.height() < 10.0 { outer } else { rect };
        let response = ui.interact(rect, self.id, Sense::click_and_drag());

        // view state
        let defaults = (self.x, self.y);
        let mut view = ui
            .data(|d| d.get_temp::<ViewState>(self.id))
            .filter(|v| v.defaults == defaults)
            .unwrap_or(ViewState {
                defaults,
                x: self.x,
                y: self.y,
            });
        // (no mouse input in screenshots: the window pops up under the user's pointer)
        if self.pan_zoom != PanZoom::None && !look.shot {
            if response.double_clicked() {
                view.x = self.x;
                view.y = self.y;
            }
            let pan_drag = response.dragged_by(egui::PointerButton::Secondary)
                || response.dragged_by(egui::PointerButton::Middle)
                || (self.left_drag_pans && response.dragged_by(egui::PointerButton::Primary));
            if pan_drag {
                let d = response.drag_delta();
                let sx = (view.x[1] - view.x[0]) / rect.width() as f64;
                view.x[0] -= d.x as f64 * sx;
                view.x[1] -= d.x as f64 * sx;
                if self.pan_zoom == PanZoom::XY {
                    let sy = (view.y[1] - view.y[0]) / rect.height() as f64;
                    view.y[0] += d.y as f64 * sy;
                    view.y[1] += d.y as f64 * sy;
                }
            }
            if let Some(hover) = response.hover_pos() {
                let (scroll, zoom) = ui.input(|i| (i.smooth_scroll_delta, i.zoom_delta()));
                let factor = zoom as f64 * (scroll.y as f64 * 0.003).exp();
                if (factor - 1.0).abs() > 1e-6 {
                    let tr = PlotTransform {
                        rect,
                        x: view.x,
                        y: view.y,
                    };
                    let (cx, cy) = tr.from_screen(hover);
                    view.x = [cx + (view.x[0] - cx) / factor, cx + (view.x[1] - cx) / factor];
                    if self.pan_zoom == PanZoom::XY {
                        view.y = [cy + (view.y[0] - cy) / factor, cy + (view.y[1] - cy) / factor];
                    }
                }
                if scroll.x != 0.0 {
                    let sx = (view.x[1] - view.x[0]) / rect.width() as f64;
                    view.x[0] -= scroll.x as f64 * sx;
                    view.x[1] -= scroll.x as f64 * sx;
                }
            }
        }
        ui.data_mut(|d| d.insert_temp(self.id, view));
        let tr = PlotTransform {
            rect,
            x: view.x,
            y: view.y,
        };

        // frame, grid, ticks
        let painter = ui.painter();
        painter.rect_filled(rect, 2.0, look.plot_bg());
        let font = look.font(11.0);
        let tick_col = look.weak_text();
        // the x-axis label sits at the bottom right, in the row of the tick labels
        let x_label = (!self.x_label.is_empty()).then(|| {
            let g = painter.layout_no_wrap(self.x_label.clone(), look.font(12.5), look.text());
            let r = Rect::from_min_size(pos2(rect.right() - g.size().x, rect.bottom() + 2.0), g.size());
            (g, r)
        });
        let (xt, xs) = nice_ticks(tr.x[0], tr.x[1], (rect.width() / look.px(80.0)).max(2.0) as usize);
        for &x in &xt {
            let sx = tr.x_to_screen(x);
            if self.grid {
                painter.line_segment([pos2(sx, rect.top()), pos2(sx, rect.bottom())], Stroke::new(1.0, look.grid()));
            }
            let g = painter.layout_no_wrap(tick_label(x, xs), font.clone(), tick_col);
            let r = Align2::CENTER_TOP.anchor_size(pos2(sx, rect.bottom() + 2.0), g.size());
            if x_label.as_ref().is_none_or(|(_, lr)| !lr.expand(4.0).intersects(r)) {
                painter.galley(r.min, g, tick_col);
            }
        }
        let (yt, ys) = nice_ticks(tr.y[0], tr.y[1], (rect.height() / look.px(36.0)).max(3.0) as usize);
        for &y in &yt {
            let sy = tr.y_to_screen(y);
            if self.grid {
                painter.line_segment([pos2(rect.left(), sy), pos2(rect.right(), sy)], Stroke::new(1.0, look.grid()));
            }
            if self.y_ticks {
                painter.text(
                    pos2(rect.left() - 4.0, sy),
                    Align2::RIGHT_CENTER,
                    tick_label(y, ys),
                    font.clone(),
                    tick_col,
                );
            }
        }
        if self.zero_line && tr.y[0] < 0.0 && tr.y[1] > 0.0 {
            let sy = tr.y_to_screen(0.0);
            painter.line_segment([pos2(rect.left(), sy), pos2(rect.right(), sy)], Stroke::new(1.0, look.axis()));
        }
        if let Some((label, r2)) = &self.y2 {
            // the right axis follows the visible y range
            let map = |y: f64| r2[0] + (y - self.y[0]) / (self.y[1] - self.y[0]) * (r2[1] - r2[0]);
            let (lo2, hi2) = (map(tr.y[0]), map(tr.y[1]));
            let (t2, s2) = nice_ticks(lo2, hi2, (rect.height() / look.px(45.0)).max(2.0) as usize);
            for &v in &t2 {
                let y = self.y[0] + (v - r2[0]) / (r2[1] - r2[0]) * (self.y[1] - self.y[0]);
                let sy = tr.y_to_screen(y);
                painter.line_segment(
                    [pos2(rect.right(), sy), pos2(rect.right() + 3.0, sy)],
                    Stroke::new(1.0, look.axis()),
                );
                painter.text(
                    pos2(rect.right() + 5.0, sy),
                    Align2::LEFT_CENTER,
                    tick_label(v, s2),
                    font.clone(),
                    tick_col,
                );
            }
            painter.text(
                pos2(rect.right() + 5.0, rect.top()),
                Align2::LEFT_TOP,
                label,
                look.font(11.5),
                look.potential(),
            );
        }

        let clipped = ui.painter_at(rect);
        content(&clipped, &tr);

        painter.rect_stroke(rect, 2.0, Stroke::new(1.0, look.plot_frame()), StrokeKind::Inside);
        if let Some((g, r)) = x_label {
            painter.galley(r.min, g, look.text());
        }
        if !self.y_label.is_empty() {
            let g = painter.layout_no_wrap(self.y_label.clone(), look.font(12.5), look.text());
            let pos = rect.left_top() + vec2(5.0, 3.0);
            painter.rect_filled(
                Rect::from_min_size(pos - vec2(2.0, 1.0), g.size() + vec2(4.0, 2.0)),
                2.0,
                look.plot_bg().gamma_multiply(0.85),
            );
            painter.galley(pos, g, look.text());
        }
        let hover = response.hover_pos().map(|p| tr.from_screen(p));
        PlotResponse {
            response,
            transform: tr,
            hover,
        }
    }
}

/// Draws a polyline of data points (skips non-finite values).
pub fn plot_line(painter: &egui::Painter, tr: &PlotTransform, xs: &[f64], ys: &[f64], stroke: Stroke) {
    let mut pts: Vec<Pos2> = Vec::with_capacity(xs.len());
    let flush = |pts: &mut Vec<Pos2>| {
        if pts.len() >= 2 {
            painter.add(Shape::line(std::mem::take(pts), stroke));
        }
        pts.clear();
    };
    for (&x, &y) in xs.iter().zip(ys) {
        if x.is_finite() && y.is_finite() {
            let p = tr.to_screen(x, y);
            // keep coordinates sane far outside the view
            pts.push(pos2(p.x.clamp(-1e5, 1e5), p.y.clamp(-1e5, 1e5)));
        } else {
            flush(&mut pts);
        }
    }
    flush(&mut pts);
}

/// Draws a dashed polyline.
pub fn plot_dashed(painter: &egui::Painter, tr: &PlotTransform, xs: &[f64], ys: &[f64], stroke: Stroke, dash: f32) {
    let pts: Vec<Pos2> = xs
        .iter()
        .zip(ys)
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(&x, &y)| tr.to_screen(x, y))
        .collect();
    painter.extend(Shape::dashed_line(&pts, stroke, dash, dash * 0.8));
}

/// Text with a translucent background box, for labels on top of plots.
pub fn label_box(painter: &egui::Painter, pos: Pos2, anchor: Align2, text: &str, color: Color32) -> Rect {
    let look = Look::from_ctx(painter.ctx());
    let g = painter.layout_no_wrap(text.to_string(), look.font(12.0), color);
    let rect = anchor.anchor_size(pos, g.size() + Vec2::splat(6.0));
    painter.rect_filled(rect, 3.0, look.plot_bg().gamma_multiply(0.85));
    painter.rect_stroke(rect, 3.0, Stroke::new(1.0, color.gamma_multiply(0.6)), StrokeKind::Inside);
    painter.galley(rect.min + Vec2::splat(3.0), g, color);
    rect
}
