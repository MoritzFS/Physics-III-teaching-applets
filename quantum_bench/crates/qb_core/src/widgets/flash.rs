//! `MeasureFlash`: the shared look of a collapse event, so measuring looks the
//! same in every tab: a detector marker with its resolution window, a short
//! bright flash, and the outcome label. After the flash a small marker stays
//! at the outcome until the next measurement or `clear()`.

use eframe::egui::{self, Align2, Color32, Pos2, Rect, Shape, Stroke, pos2, vec2};

use super::Domain;
use super::plot::{PlotTransform, label_box};
use crate::{Look, wall_time};

/// What was measured (in data coordinates of the plot it is painted on).
#[derive(Clone, Debug, PartialEq)]
pub enum FlashKind {
    /// position x with detector resolution `width` (σ of the window)
    Position { x: f64, width: f64 },
    /// momentum p with resolution `width`
    Momentum { p: f64, width: f64 },
    /// "is it in [a, b]?" with the answer
    Region { a: f64, b: f64, yes: bool },
    /// energy level `index` at energy `energy` (painted as a horizontal line)
    Level { index: usize, energy: f64 },
    /// anything else: only the label and the flash (e.g. spin up/down)
    Other,
}

#[derive(Clone, Debug)]
struct Event {
    kind: FlashKind,
    label: String,
    /// [`wall_time`] of the event
    start: f64,
}

/// Holds the latest measurement event of a view. Keep one in the scene,
/// call [`MeasureFlash::trigger`] when measuring and [`MeasureFlash::paint`]
/// on every plot after drawing it.
#[derive(Clone, Debug, Default)]
pub struct MeasureFlash {
    event: Option<Event>,
}

/// Flash duration and label duration in seconds.
const FLASH: f32 = 0.7;
const LABEL: f32 = 4.0;

impl MeasureFlash {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a measurement; `label` is the outcome text, e.g. "x = 1.23".
    pub fn trigger(&mut self, kind: FlashKind, label: impl Into<String>) {
        self.event = Some(Event {
            kind,
            label: label.into(),
            start: wall_time(),
        });
    }

    /// Forgets the event (e.g. on reset).
    pub fn clear(&mut self) {
        self.event = None;
    }

    pub fn kind(&self) -> Option<&FlashKind> {
        self.event.as_ref().map(|e| &e.kind)
    }

    /// The outcome text of the latest measurement.
    pub fn label(&self) -> Option<&str> {
        self.event.as_ref().map(|e| e.label.as_str())
    }

    /// Seconds since the event ([`wall_time`]: in screenshots the virtual clock).
    pub fn age(&self) -> Option<f32> {
        self.event.as_ref().map(|e| (wall_time() - e.start) as f32)
    }

    /// Paints on a plot of the given domain (position events on x plots,
    /// momentum events on p plots; `Level` and `Other` on any). Requests
    /// repaints while animating.
    pub fn paint(&self, ui: &egui::Ui, tr: &PlotTransform, domain: Domain) {
        let Some(e) = &self.event else { return };
        let look = Look::of(ui);
        let age = self.age().unwrap_or(f32::MAX);
        if age < LABEL + 0.5 {
            ui.ctx().request_repaint();
        }
        let painter = ui.painter_at(tr.rect);
        let r = tr.rect;
        let col = look.measure();
        let flash = (1.0 - age / FLASH).clamp(0.0, 1.0);
        let label_alpha = if age < LABEL {
            1.0
        } else {
            (1.0 - (age - LABEL) / 0.5).clamp(0.0, 1.0)
        };
        let mut label_at: Option<Pos2> = None;
        // while the label shows: the detector window and a bright line; afterwards only a small marker
        let fresh = label_alpha;
        let at = |v: f64, width: f64| {
            let sx = tr.x_to_screen(v);
            let w = (width / tr.dx_per_point()).abs() as f32;
            if fresh > 0.0 {
                // the detector window exp(−(x−x0)²/(2w²)) as a filled bump
                let n = 64;
                let mut pts = Vec::with_capacity(n + 2);
                let h = r.height() * 0.85;
                for k in 0..=n {
                    let u = -3.0 + 6.0 * k as f32 / n as f32;
                    pts.push(pos2(sx + u * w.max(0.5), r.bottom() - h * (-0.5 * u * u).exp()));
                }
                let fill = col.gamma_multiply((0.08 + 0.25 * flash) * fresh);
                let mut mesh = egui::Mesh::default();
                for p in &pts {
                    let i = mesh.vertices.len() as u32;
                    mesh.colored_vertex(*p, fill);
                    mesh.colored_vertex(pos2(p.x, r.bottom()), fill);
                    if i >= 2 {
                        mesh.add_triangle(i - 2, i - 1, i);
                        mesh.add_triangle(i - 1, i, i + 1);
                    }
                }
                painter.add(Shape::mesh(mesh));
                painter.add(Shape::line(pts, look.stroke(1.0, col.gamma_multiply(0.8 * fresh))));
            }
            if flash > 0.0 {
                let glow = Rect::from_center_size(pos2(sx, r.center().y), vec2(look.px(8.0) + w * 2.0, r.height()));
                painter.rect_filled(glow, 0.0, Color32::WHITE.gamma_multiply(0.35 * flash));
            }
            let line_col = col.gamma_multiply(0.35 + 0.65 * fresh);
            let ends = [pos2(sx, r.top()), pos2(sx, r.bottom())];
            if fresh > 0.5 {
                painter.line_segment(ends, look.stroke(1.5, line_col));
            } else {
                painter.extend(Shape::dashed_line(&ends, look.stroke(1.2, line_col), 6.0, 4.0));
            }
            // detector symbol at the bottom
            let s = look.px(6.0);
            painter.add(Shape::convex_polygon(
                vec![pos2(sx, r.bottom() - s * 1.6), pos2(sx - s, r.bottom()), pos2(sx + s, r.bottom())],
                col,
                Stroke::NONE,
            ));
            pos2(sx, r.top() + look.px(42.0))
        };
        match (&e.kind, domain) {
            (FlashKind::Position { x, width }, Domain::Position) => label_at = Some(at(*x, *width)),
            (FlashKind::Momentum { p, width }, Domain::Momentum) => label_at = Some(at(*p, *width)),
            (FlashKind::Region { a, b, yes }, Domain::Position) => {
                let (sa, sb) = (tr.x_to_screen(a.min(*b)), tr.x_to_screen(a.max(*b)));
                let band = Rect::from_min_max(pos2(sa, r.top()), pos2(sb, r.bottom()));
                let c = if *yes { col } else { look.weak_text() };
                painter.rect_filled(band, 0.0, c.gamma_multiply(0.10 + 0.4 * flash));
                painter.rect_stroke(band, 0.0, look.stroke(1.5, c), egui::StrokeKind::Inside);
                label_at = Some(pos2(0.5 * (sa + sb), r.top() + look.px(46.0)));
            }
            (FlashKind::Level { energy, .. }, _) => {
                let sy = tr.y_to_screen(*energy);
                if flash > 0.0 {
                    let glow = Rect::from_center_size(pos2(r.center().x, sy), vec2(r.width(), look.px(14.0)));
                    painter.rect_filled(glow, 0.0, Color32::WHITE.gamma_multiply(0.5 * flash));
                }
                painter.line_segment([pos2(r.left(), sy), pos2(r.right(), sy)], look.stroke(2.0, col));
                label_at = Some(pos2(r.center().x, sy - look.px(14.0)));
            }
            (FlashKind::Other, _) => {
                if flash > 0.0 {
                    painter.rect_filled(r, 0.0, Color32::WHITE.gamma_multiply(0.25 * flash));
                }
                label_at = Some(pos2(r.center().x, r.top() + look.px(46.0)));
            }
            _ => {}
        }
        if let Some(p) = label_at
            && label_alpha > 0.0
        {
            // beside the marker line, clear of the ⟨x⟩ label above it
            let (pos, anchor) = if p.x > r.right() - look.px(110.0) {
                (p - vec2(look.px(6.0), 0.0), Align2::RIGHT_TOP)
            } else {
                (p + vec2(look.px(6.0), 0.0), Align2::LEFT_TOP)
            };
            label_box(&painter, pos, anchor, &e.label, col.gamma_multiply(label_alpha));
        }
    }

    /// The outcome as a prominent banner at the top of `rect` (for views
    /// without a matching axis, e.g. a Stern–Gerlach screen).
    pub fn paint_banner(&self, ui: &egui::Ui, rect: Rect) {
        let Some(e) = &self.event else { return };
        let look = Look::of(ui);
        let age = self.age().unwrap_or(f32::MAX);
        if age < LABEL + 0.5 {
            ui.ctx().request_repaint();
        }
        let a = if age < LABEL {
            1.0
        } else {
            (1.0 - (age - LABEL) / 0.5).clamp(0.0, 1.0)
        };
        if a <= 0.0 {
            return;
        }
        let painter = ui.painter_at(rect);
        let flash = (1.0 - age / FLASH).clamp(0.0, 1.0);
        if flash > 0.0 {
            painter.rect_filled(rect, 0.0, Color32::WHITE.gamma_multiply(0.2 * flash));
        }
        label_box(
            &painter,
            pos2(rect.center().x, rect.top() + look.px(20.0)),
            Align2::CENTER_CENTER,
            &e.label,
            look.measure().gamma_multiply(a),
        );
    }
}
