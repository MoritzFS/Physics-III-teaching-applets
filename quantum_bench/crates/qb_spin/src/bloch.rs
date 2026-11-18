//! The Bloch sphere widget and the state panel (column vector in the z basis
//! and P(↑), P(↓) bars along a chosen axis), drawn with egui painters.
//!
//! The sphere is an orthographic view from the direction (az, el); drag to
//! turn it, double-click to reset. Depth cues: everything on the far side is
//! drawn first and then covered by the translucent, shaded ball, so the back
//! hemisphere appears dimmed; the near side is drawn on top.
//!
//! ```ignore
//! BlochSphere::new("ramsey_bloch", 320.0)
//!     .state(psi.bloch())
//!     .trail(&self.trail)
//!     .axis(BlochAxis::measurement(EZ, "↑z", "↓z", look.measure()))
//!     .show(ui);
//! ```

use qb_core::Look;
use qb_core::egui::{self, Align2, Color32, FontId, Id, Mesh, Pos2, Rect, Sense, Shape, Stroke, Vec2, pos2, vec2};
use qb_core::widgets::{phase_color, segmented};

use crate::spinor::{EX, EY, EZ, Spinor, Vec3, norm, scale};
use crate::util::{fmt_amp, fmt_cartesian};

/// View direction of a Bloch sphere: azimuth and elevation of the viewer (radians).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub az: f32,
    pub el: f32,
}

/// The textbook view: x towards the viewer (lower left), y to the right, z up.
pub const DEFAULT_CAMERA: Camera = Camera { az: 0.42, el: 0.36 };

impl Camera {
    /// (right, up, towards viewer) unit vectors.
    fn basis(&self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let (sa, ca) = self.az.sin_cos();
        let (se, ce) = self.el.sin_cos();
        ([-sa, ca, 0.0], [-se * ca, -se * sa, ce], [ce * ca, ce * sa, se])
    }

    /// Screen offset in units of the radius (x right, y down) and depth (> 0: near side).
    pub fn project(&self, p: Vec3) -> (Vec2, f32) {
        let (r, u, v) = self.basis();
        let p = [p[0] as f32, p[1] as f32, p[2] as f32];
        let d = |a: [f32; 3]| a[0] * p[0] + a[1] * p[1] + a[2] * p[2];
        (vec2(d(r), -d(u)), d(v))
    }
}

/// A line through the sphere (a measurement axis, a rotation axis) with labels at both ends.
#[derive(Clone, Debug)]
pub struct BlochAxis {
    pub dir: Vec3,
    pub color: Color32,
    pub plus: String,
    pub minus: String,
    pub dashed: bool,
}

impl BlochAxis {
    /// A measurement axis u: solid, labels "↑u" at +u and "↓u" at −u.
    pub fn measurement(dir: Vec3, plus: impl Into<String>, minus: impl Into<String>, color: Color32) -> Self {
        BlochAxis {
            dir,
            color,
            plus: plus.into(),
            minus: minus.into(),
            dashed: false,
        }
    }

    /// A rotation axis (drawn dashed).
    pub fn rotation(dir: Vec3, label: impl Into<String>, color: Color32) -> Self {
        BlochAxis {
            dir,
            color,
            plus: label.into(),
            minus: String::new(),
            dashed: true,
        }
    }
}

/// An extra arrow from the centre (e.g. a rotating measurement direction).
#[derive(Clone, Debug)]
pub struct BlochArrow {
    pub dir: Vec3,
    pub color: Color32,
    pub label: String,
}

/// What `BlochSphere::show` returns.
pub struct BlochResponse {
    pub response: egui::Response,
    /// the whole widget
    pub rect: Rect,
    pub center: Pos2,
    pub radius: f32,
}

/// The Bloch sphere widget.
pub struct BlochSphere<'a> {
    id: Id,
    camera_id: Id,
    default_camera: Camera,
    size: Vec2,
    state: Option<Vec3>,
    state_color: Option<Color32>,
    state_label: Option<String>,
    trail: &'a [Vec3],
    axes: Vec<BlochAxis>,
    arrows: Vec<BlochArrow>,
    cloud: &'a [Vec3],
    title: Option<String>,
    caption: Option<String>,
    xyz_labels: bool,
    pole_labels: bool,
    compact: bool,
    interactive: bool,
    frame: bool,
}

impl<'a> BlochSphere<'a> {
    /// `id` identifies the widget and (by default) its camera; `size` is the side of the square.
    pub fn new(id: impl std::hash::Hash + std::fmt::Debug, size: f32) -> Self {
        let id = Id::new(id);
        BlochSphere {
            id,
            camera_id: id.with("camera"),
            default_camera: DEFAULT_CAMERA,
            size: vec2(size, size),
            state: None,
            state_color: None,
            state_label: None,
            trail: &[],
            axes: vec![],
            arrows: vec![],
            cloud: &[],
            title: None,
            caption: None,
            xyz_labels: true,
            pole_labels: true,
            compact: false,
            interactive: true,
            frame: true,
        }
    }

    /// Width and height (the sphere is centred, its radius set by the smaller side).
    pub fn size(mut self, w: f32, h: f32) -> Self {
        self.size = vec2(w, h);
        self
    }

    /// The initial view (also restored by a double-click).
    pub fn default_camera(mut self, cam: Camera) -> Self {
        self.default_camera = cam;
        self
    }

    /// Several spheres with the same camera key turn together.
    pub fn shared_camera(mut self, key: impl std::hash::Hash + std::fmt::Debug) -> Self {
        self.camera_id = Id::new(key).with("camera");
        self
    }

    /// The Bloch vector of the state (|r| ≤ 1).
    pub fn state(mut self, r: Vec3) -> Self {
        self.state = Some(r);
        self
    }

    pub fn state_color(mut self, c: Color32) -> Self {
        self.state_color = Some(c);
        self
    }

    pub fn state_label(mut self, l: impl Into<String>) -> Self {
        self.state_label = Some(l.into());
        self
    }

    /// Earlier Bloch vectors, oldest first (a non-finite entry breaks the line).
    pub fn trail(mut self, t: &'a [Vec3]) -> Self {
        self.trail = t;
        self
    }

    pub fn axis(mut self, a: BlochAxis) -> Self {
        self.axes.push(a);
        self
    }

    pub fn arrow(mut self, dir: Vec3, color: Color32, label: impl Into<String>) -> Self {
        self.arrows.push(BlochArrow {
            dir,
            color,
            label: label.into(),
        });
        self
    }

    /// Faint dots (e.g. the directions of atoms from an unpolarised oven).
    pub fn cloud(mut self, pts: &'a [Vec3]) -> Self {
        self.cloud = pts;
        self
    }

    pub fn title(mut self, t: impl Into<String>) -> Self {
        self.title = Some(t.into());
        self
    }

    /// A line of text at the bottom.
    pub fn caption(mut self, t: impl Into<String>) -> Self {
        self.caption = Some(t.into());
        self
    }

    /// Show the x, y, z letters (default true).
    pub fn xyz_labels(mut self, yes: bool) -> Self {
        self.xyz_labels = yes;
        self
    }

    /// Show |↑z⟩, |↓z⟩ at the poles (default true).
    pub fn pole_labels(mut self, yes: bool) -> Self {
        self.pole_labels = yes;
        self
    }

    /// Small version: thinner lines, smaller text, no pole labels.
    pub fn compact(mut self, yes: bool) -> Self {
        self.compact = yes;
        if yes {
            self.pole_labels = false;
        }
        self
    }

    pub fn interactive(mut self, yes: bool) -> Self {
        self.interactive = yes;
        self
    }

    /// Draw the panel background and frame (default true).
    pub fn frame(mut self, yes: bool) -> Self {
        self.frame = yes;
        self
    }

    pub fn show(self, ui: &mut egui::Ui) -> BlochResponse {
        let look = Look::of(ui);
        let (rect, _) = ui.allocate_exact_size(self.size, Sense::hover());
        let sense = if self.interactive {
            Sense::click_and_drag()
        } else {
            Sense::hover()
        };
        let response = ui.interact(rect, self.id, sense);
        let mut cam = ui.data(|d| d.get_temp::<Camera>(self.camera_id)).unwrap_or(self.default_camera);
        if self.interactive && !look.shot {
            if response.dragged() {
                let d = response.drag_delta();
                cam.az -= d.x * 0.008;
                cam.el = (cam.el + d.y * 0.008).clamp(-1.52, 1.52);
            }
            if response.double_clicked() {
                cam = self.default_camera;
            }
        }
        ui.data_mut(|d| d.insert_temp(self.camera_id, cam));
        let response = if self.interactive {
            response.on_hover_text("drag to turn the sphere, double-click to reset the view")
        } else {
            response
        };

        let painter = ui.painter_at(rect);
        if self.frame {
            painter.rect_filled(rect, 3.0, look.plot_bg());
            painter.rect_stroke(rect, 3.0, Stroke::new(1.0, look.plot_frame()), egui::StrokeKind::Inside);
        }
        let k = if self.compact { 0.75 } else { 1.0 };
        let top_pad = if self.title.is_some() { look.px(16.0 * k) } else { 0.0 };
        let bottom_pad = if self.caption.is_some() { look.px(16.0 * k) } else { 0.0 };
        let avail = vec2(rect.width(), rect.height() - top_pad - bottom_pad);
        let label_room = if self.xyz_labels || !self.axes.is_empty() { 1.42 } else { 1.12 };
        let radius = (0.5 * avail.x.min(avail.y) / label_room).max(10.0);
        let center = pos2(rect.center().x, rect.top() + top_pad + 0.5 * avail.y);

        let mut d = Draw {
            cam,
            center,
            radius,
            back: vec![],
            front: vec![],
            back_text: vec![],
            front_text: vec![],
        };
        let line_k = if self.compact { 0.75 } else { 1.0 };
        let font_small = look.font(if self.compact { 10.0 } else { 13.0 });

        // great circles: equator stronger, two meridians faint
        let circle = |f: &dyn Fn(f64) -> Vec3| -> Vec<Vec3> { (0..=96).map(|i| f(i as f64 / 96.0 * std::f64::consts::TAU)).collect() };
        let eq = circle(&|t| [t.cos(), t.sin(), 0.0]);
        let mxz = circle(&|t| [t.cos(), 0.0, t.sin()]);
        let myz = circle(&|t| [0.0, t.cos(), t.sin()]);
        d.polyline(&mxz, look.stroke(0.9 * line_k, look.plot_frame()));
        d.polyline(&myz, look.stroke(0.9 * line_k, look.plot_frame()));
        d.polyline(&eq, look.stroke(1.3 * line_k, look.axis()));

        // coordinate axes
        let axis_col = look.weak_text();
        for (e, name) in [(EX, "x"), (EY, "y"), (EZ, "z")] {
            d.polyline(&[scale(e, -1.0), e], look.stroke(0.9 * line_k, axis_col.gamma_multiply(0.75)));
            let tip = scale(e, 1.2);
            d.arrow3(e, tip, look.stroke(1.1 * line_k, axis_col), look.px(6.0 * k));
            if self.xyz_labels {
                d.label(scale(e, 1.33), Align2::CENTER_CENTER, name, font_small.clone(), axis_col);
            }
        }
        if self.pole_labels {
            let f = look.font(12.0);
            let (p, _) = d.to_screen(EZ);
            let (q, dq) = d.to_screen(scale(EZ, -1.0));
            d.front_text.push((
                p + vec2(look.px(7.0), -look.px(2.0)),
                Align2::LEFT_BOTTOM,
                "|↑z⟩".into(),
                f.clone(),
                look.text(),
            ));
            let col = if dq < 0.0 { look.text().gamma_multiply(0.6) } else { look.text() };
            d.front_text
                .push((q + vec2(look.px(7.0), look.px(2.0)), Align2::LEFT_TOP, "|↓z⟩".into(), f, col));
        }

        // cloud of points (mixtures)
        for p in self.cloud {
            let (s, depth) = d.to_screen(*p);
            let col = look.counts();
            let shape = Shape::circle_filled(s, look.px(1.8 * k), col.gamma_multiply(0.8));
            if depth >= 0.0 { d.front.push(shape) } else { d.back.push(shape) }
        }

        // axes through the sphere (measurement u, rotation axes)
        for a in &self.axes {
            let u = crate::spinor::normalize(a.dir);
            let stroke = look.stroke(1.8 * line_k, a.color);
            if a.dashed {
                d.dashed(scale(u, -1.15), scale(u, 1.15), look.stroke(1.6 * line_k, a.color));
                if !a.plus.is_empty() {
                    d.label(scale(u, 1.3), Align2::CENTER_CENTER, &a.plus, look.font(12.0 * k), a.color);
                }
            } else {
                d.polyline(&[scale(u, -1.0), [0.0; 3], u], stroke);
                d.arrow3(u, scale(u, 1.22), stroke, look.px(8.0 * k));
                d.polyline(&[scale(u, -1.0), scale(u, -1.1)], stroke);
                if !a.plus.is_empty() {
                    d.label(scale(u, 1.36), Align2::CENTER_CENTER, &a.plus, look.font(13.0 * k), a.color);
                }
                if !a.minus.is_empty() {
                    d.label(scale(u, -1.24), Align2::CENTER_CENTER, &a.minus, look.font(12.0 * k), a.color);
                }
            }
        }

        // trail of the state
        let state_col = self.state_color.unwrap_or(look.accent());
        let n = self.trail.len();
        for i in 1..n {
            let (a, b) = (self.trail[i - 1], self.trail[i]);
            if !(a.iter().chain(&b).all(|v| v.is_finite())) {
                continue;
            }
            let age = (i as f32 / n as f32).powf(0.8);
            d.segment(a, b, look.stroke(1.6 * line_k, state_col.gamma_multiply(0.15 + 0.7 * age)));
        }

        // extra arrows
        for a in &self.arrows {
            d.vector(a.dir, look.stroke(2.0 * line_k, a.color), look.px(9.0 * k), look.px(2.6 * k));
            if !a.label.is_empty() {
                let r = norm(a.dir).max(0.3);
                d.label(
                    scale(a.dir, (r + 0.17) / r),
                    Align2::CENTER_CENTER,
                    &a.label,
                    look.font(12.0 * k),
                    a.color,
                );
            }
        }

        // the state: drop line to the equator plane, arrow, label
        if let Some(r) = self.state {
            let len = norm(r);
            if len > 0.05 && r[2].abs() < 0.995 * len {
                let foot = [r[0], r[1], 0.0];
                let weak = look.weak_text().gamma_multiply(0.8);
                d.dashed(r, foot, look.stroke(1.0 * line_k, weak));
                d.dashed([0.0; 3], foot, look.stroke(0.8 * line_k, weak.gamma_multiply(0.7)));
            }
            if len > 0.02 {
                d.vector(r, look.stroke(2.8 * line_k, state_col), look.px(11.0 * k), look.px(3.6 * k));
            } else {
                // the maximally mixed state: r = 0
                d.front.push(Shape::circle_filled(center, look.px(4.0 * k), state_col));
            }
            if let Some(l) = &self.state_label {
                let at = if len > 0.02 {
                    scale(r, (len + 0.2) / len)
                } else {
                    [0.0, 0.0, -0.22]
                };
                d.label(at, Align2::CENTER_CENTER, l, look.font(13.0 * k), state_col);
            }
        }

        // paint: far side, the translucent ball, near side
        for s in d.back.drain(..) {
            painter.add(s);
        }
        for (p, a, t, f, c) in d.back_text.drain(..) {
            painter.text(p, a, t, f, c);
        }
        ball(&painter, &look, center, radius);
        for s in d.front.drain(..) {
            painter.add(s);
        }
        for (p, a, t, f, c) in d.front_text.drain(..) {
            painter.text(p, a, t, f, c);
        }
        if let Some(t) = &self.title {
            painter.text(
                rect.left_top() + vec2(look.px(6.0), look.px(4.0)),
                Align2::LEFT_TOP,
                t,
                look.font(12.5 * k),
                look.text(),
            );
        }
        if let Some(t) = &self.caption {
            painter.text(
                pos2(rect.center().x, rect.bottom() - look.px(3.0)),
                Align2::CENTER_BOTTOM,
                t,
                look.font(12.0 * k),
                look.text(),
            );
        }
        BlochResponse {
            response,
            rect,
            center,
            radius,
        }
    }
}

type TextItem = (Pos2, Align2, String, FontId, Color32);

/// Collects shapes into a far and a near layer.
struct Draw {
    cam: Camera,
    center: Pos2,
    radius: f32,
    back: Vec<Shape>,
    front: Vec<Shape>,
    back_text: Vec<TextItem>,
    front_text: Vec<TextItem>,
}

impl Draw {
    fn to_screen(&self, p: Vec3) -> (Pos2, f32) {
        let (v, d) = self.cam.project(p);
        (self.center + v * self.radius, d)
    }

    fn push(&mut self, front: bool, s: Shape) {
        if front { self.front.push(s) } else { self.back.push(s) }
    }

    /// A polyline split into near and far parts by the depth of each piece.
    fn polyline(&mut self, pts: &[Vec3], stroke: Stroke) {
        let mut cur: Vec<Pos2> = vec![];
        let mut cur_front: Option<bool> = None;
        for w in pts.windows(2) {
            let (a, da) = self.to_screen(w[0]);
            let (b, db) = self.to_screen(w[1]);
            let front = 0.5 * (da + db) >= -1e-4;
            if cur_front != Some(front) {
                if cur.len() >= 2 {
                    let line = std::mem::take(&mut cur);
                    self.push(cur_front.unwrap(), Shape::line(line, stroke));
                }
                cur.clear();
                cur.push(a);
                cur_front = Some(front);
            }
            cur.push(b);
        }
        if cur.len() >= 2 {
            self.push(cur_front.unwrap(), Shape::line(cur, stroke));
        }
    }

    fn segment(&mut self, a: Vec3, b: Vec3, stroke: Stroke) {
        let (pa, da) = self.to_screen(a);
        let (pb, db) = self.to_screen(b);
        self.push(0.5 * (da + db) >= 0.0, Shape::line_segment([pa, pb], stroke));
    }

    fn dashed(&mut self, a: Vec3, b: Vec3, stroke: Stroke) {
        // split at the depth sign change so each half lands in its layer
        let (pa, da) = self.to_screen(a);
        let (pb, db) = self.to_screen(b);
        let mut parts = vec![];
        if da * db < 0.0 {
            let f = da / (da - db);
            let pm = pa + (pb - pa) * f;
            parts.push((pa, pm, da >= 0.0));
            parts.push((pm, pb, db >= 0.0));
        } else {
            parts.push((pa, pb, da + db >= 0.0));
        }
        for (p, q, front) in parts {
            if (q - p).length() > 0.5 {
                for s in Shape::dashed_line(&[p, q], stroke, 4.0, 3.0) {
                    self.push(front, s);
                }
            }
        }
    }

    /// Line from a to b with an arrow head at b.
    fn arrow3(&mut self, a: Vec3, b: Vec3, stroke: Stroke, head: f32) {
        let (pa, da) = self.to_screen(a);
        let (pb, db) = self.to_screen(b);
        let front = 0.5 * (da + db) >= 0.0;
        let mut shapes = vec![];
        head_shapes(&mut shapes, pa, pb, stroke, head);
        for s in shapes {
            self.push(front, s);
        }
    }

    /// Arrow from the centre to r with a tip dot.
    fn vector(&mut self, r: Vec3, stroke: Stroke, head: f32, dot_r: f32) {
        let (pa, _) = self.to_screen([0.0; 3]);
        let (pb, db) = self.to_screen(r);
        let front = db >= 0.0;
        let mut shapes = vec![];
        if (pb - pa).length() > head * 1.2 {
            head_shapes(&mut shapes, pa, pb, stroke, head);
        } else {
            // pointing almost at (or away from) the viewer: a dot with a ring
            shapes.push(Shape::line_segment([pa, pb], stroke));
            shapes.push(Shape::circle_stroke(pb, dot_r * 2.2, Stroke::new(stroke.width * 0.6, stroke.color)));
        }
        shapes.push(Shape::circle_filled(pb, dot_r, stroke.color));
        for s in shapes {
            self.push(front, s);
        }
    }

    fn label(&mut self, at: Vec3, align: Align2, text: &str, font: FontId, color: Color32) {
        let (p, depth) = self.to_screen(at);
        // labels behind the ball are dimmed with it; labels outside its outline stay sharp
        let inside = (p - self.center).length() < self.radius;
        let item = (p, align, text.to_string(), font, color);
        if depth < 0.0 && inside {
            self.back_text.push(item);
        } else {
            self.front_text.push(item);
        }
    }
}

fn head_shapes(out: &mut Vec<Shape>, a: Pos2, b: Pos2, stroke: Stroke, head: f32) {
    let d = b - a;
    let len = d.length();
    if len < 1e-3 {
        return;
    }
    let dir = d / len;
    let head = head.min(0.7 * len);
    let base = b - dir * head;
    let perp = vec2(-dir.y, dir.x) * head * 0.45;
    out.push(Shape::line_segment([a, base + dir * 1.0], stroke));
    out.push(Shape::convex_polygon(vec![b, base + perp, base - perp], stroke.color, Stroke::NONE));
}

/// The translucent, softly shaded ball (covers and so dims the far side).
fn ball(painter: &egui::Painter, look: &Look, center: Pos2, radius: f32) {
    let (inner, outer) = if look.dark {
        (
            Color32::from_rgba_unmultiplied(78, 88, 108, 120),
            Color32::from_rgba_unmultiplied(22, 25, 32, 185),
        )
    } else {
        (
            Color32::from_rgba_unmultiplied(255, 255, 255, 150),
            Color32::from_rgba_unmultiplied(200, 206, 218, 190),
        )
    };
    let hl = center + vec2(-0.35, -0.38) * radius;
    let mut mesh = Mesh::default();
    mesh.colored_vertex(hl, inner);
    let n = 72;
    for i in 0..=n {
        let a = i as f32 / n as f32 * std::f32::consts::TAU;
        mesh.colored_vertex(center + vec2(a.cos(), a.sin()) * radius, outer);
    }
    for i in 0..n as u32 {
        mesh.add_triangle(0, i + 1, i + 2);
    }
    painter.add(Shape::mesh(mesh));
    painter.circle_stroke(center, radius, look.stroke(1.3, look.axis()));
}

// ------------------------------------------------------------------ state panel

/// The axis along which P(↑), P(↓) are shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AxisChoice {
    Z,
    X,
    Y,
    /// the scene's measurement axis u
    U,
}

impl AxisChoice {
    pub fn dir(self, u: Option<Vec3>) -> Vec3 {
        match self {
            AxisChoice::Z => EZ,
            AxisChoice::X => EX,
            AxisChoice::Y => EY,
            AxisChoice::U => u.unwrap_or(EZ),
        }
    }

    pub fn name(self, u_name: &str) -> &str {
        match self {
            AxisChoice::Z => "z",
            AxisChoice::X => "x",
            AxisChoice::Y => "y",
            AxisChoice::U => u_name,
        }
    }
}

/// The state as a column vector in the z basis: |ψ⟩ = (a, b)ᵀ, each amplitude with a
/// phasor in the phase colour (angle arg c, length |c|) and in polar form.
pub fn state_column(ui: &mut egui::Ui, label: &str, s: &Spinor) -> egui::Response {
    let look = Look::of(ui);
    let font = look.font(15.0);
    let painter = ui.painter().clone();
    let rows = [fmt_amp(s.up), fmt_amp(s.down)];
    let g_label = painter.layout_no_wrap(label.to_string(), font.clone(), look.text());
    let g_rows: Vec<_> = rows
        .iter()
        .map(|r| painter.layout_no_wrap(r.clone(), font.clone(), look.text()))
        .collect();
    let tags = ["|↑z⟩", "|↓z⟩"];
    let g_tags: Vec<_> = tags
        .iter()
        .map(|t| painter.layout_no_wrap(t.to_string(), look.font(11.0), look.weak_text()))
        .collect();
    let row_h = g_rows[0].size().y.max(look.px(18.0)) + look.px(5.0);
    let r = 0.36 * row_h;
    let text_w = g_rows.iter().map(|g| g.size().x).fold(look.px(60.0), f32::max);
    let tag_w = g_tags.iter().map(|g| g.size().x).fold(0.0, f32::max);
    let paren = look.px(7.0);
    let w = g_label.size().x + look.px(8.0) + paren + 2.0 * r + look.px(8.0) + text_w + look.px(6.0) + paren + look.px(6.0) + tag_w;
    let h = 2.0 * row_h + look.px(6.0);
    let (rect, response) = ui.allocate_exact_size(vec2(w, h), Sense::hover());
    let cy = rect.center().y;
    painter.galley(pos2(rect.left(), cy - 0.5 * g_label.size().y), g_label.clone(), look.text());
    let x0 = rect.left() + g_label.size().x + look.px(8.0);
    let x1 = x0 + paren + 2.0 * r + look.px(8.0) + text_w + look.px(6.0);
    let stroke = look.stroke(1.4, look.text());
    paren_shape(&painter, x0, rect.top() + 1.0, rect.bottom() - 1.0, paren, true, stroke);
    paren_shape(&painter, x1, rect.top() + 1.0, rect.bottom() - 1.0, paren, false, stroke);
    for (i, (c, g)) in [s.up, s.down].iter().zip(&g_rows).enumerate() {
        let y = rect.top() + look.px(3.0) + (i as f32 + 0.5) * row_h;
        let pc = pos2(x0 + paren + r, y);
        painter.circle_stroke(pc, r, Stroke::new(1.0, look.plot_frame()));
        if c.norm() > 1e-3 {
            let a = c.arg() as f32;
            let col = phase_color(a, 1.0);
            let tip = pc + vec2(a.cos(), -a.sin()) * r * c.norm() as f32;
            painter.line_segment([pc, tip], look.stroke(1.8, col));
            painter.circle_filled(tip, look.px(1.8), col);
        }
        painter.galley(pos2(pc.x + r + look.px(8.0), y - 0.5 * g.size().y), g.clone(), look.text());
        painter.galley(
            pos2(x1 + paren + look.px(4.0), y - 0.5 * g_tags[i].size().y),
            g_tags[i].clone(),
            look.weak_text(),
        );
    }
    response.on_hover_text(format!(
        "z basis: (⟨↑z|ψ⟩, ⟨↓z|ψ⟩)\n⟨↑z|ψ⟩ = {}\n⟨↓z|ψ⟩ = {}\nphasor: angle = arg, length = |amplitude| (phase colour)",
        fmt_cartesian(s.up),
        fmt_cartesian(s.down)
    ))
}

fn paren_shape(painter: &egui::Painter, x: f32, top: f32, bottom: f32, w: f32, left: bool, stroke: Stroke) {
    let n = 16;
    let pts: Vec<Pos2> = (0..=n)
        .map(|i| {
            let t = i as f32 / n as f32;
            let bulge = (std::f32::consts::PI * t).sin() * w * 0.8;
            let px = if left { x + w - bulge } else { x + bulge };
            pos2(px, top + (bottom - top) * t)
        })
        .collect();
    painter.add(Shape::line(pts, stroke));
}

/// P(↑u) and P(↓u) as two horizontal bars (Born rule, |⟨↑u|ψ⟩|²).
pub fn prob_bars(ui: &mut egui::Ui, s: &Spinor, n: Vec3, name: &str, width: f32) -> egui::Response {
    let look = Look::of(ui);
    let row_h = look.px(19.0);
    let (rect, response) = ui.allocate_exact_size(vec2(width, 2.0 * row_h + 2.0), Sense::hover());
    let painter = ui.painter_at(rect.expand(2.0));
    let p_up = s.p_up(n).clamp(0.0, 1.0);
    let label_w = look.px(58.0);
    let value_w = look.px(46.0);
    let bar_x0 = rect.left() + label_w;
    let bar_w = (rect.width() - label_w - value_w).max(20.0);
    for (i, (lab, p)) in [(format!("P(↑{name})"), p_up), (format!("P(↓{name})"), 1.0 - p_up)]
        .iter()
        .enumerate()
    {
        let y = rect.top() + i as f32 * row_h;
        painter.text(
            pos2(rect.left(), y + 0.5 * row_h),
            Align2::LEFT_CENTER,
            lab,
            look.font(12.5),
            look.text(),
        );
        let bg = Rect::from_min_size(pos2(bar_x0, y + 3.0), vec2(bar_w, row_h - 6.0));
        painter.rect_filled(bg, 2.0, look.grid());
        let fill = Rect::from_min_size(bg.min, vec2(bar_w * *p as f32, bg.height()));
        painter.rect_filled(fill, 2.0, look.prediction().gamma_multiply(0.85));
        painter.text(
            pos2(rect.right(), y + 0.5 * row_h),
            Align2::RIGHT_CENTER,
            format!("{p:.3}"),
            look.mono(13.0),
            look.text(),
        );
    }
    response.on_hover_text("Born rule: P(↑u) = |⟨↑u|ψ⟩|² with |↑u⟩ from Eq. 50.3a")
}

/// The scene's own axis for the probability bars: direction, the name used in
/// "P(↑name)", and the label of its switch button.
#[derive(Clone, Copy, Debug)]
pub struct SceneAxis<'a> {
    pub dir: Vec3,
    pub name: &'a str,
    pub button: &'a str,
}

/// A z | x | y (| scene axis) switch for the axis of the probability bars.
pub fn axis_selector(ui: &mut egui::Ui, axis: &mut AxisChoice, button: Option<&str>) {
    let look = Look::of(ui);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("P along").color(look.weak_text()));
        segmented(ui, |ui| {
            for (a, l) in [(AxisChoice::Z, "z"), (AxisChoice::X, "x"), (AxisChoice::Y, "y")] {
                ui.selectable_value(axis, a, l);
            }
            if let Some(b) = button {
                ui.selectable_value(axis, AxisChoice::U, b);
            }
        });
    });
    if button.is_none() && *axis == AxisChoice::U {
        *axis = AxisChoice::Z;
    }
}

/// Column vector, axis switch and probability bars, stacked.
pub fn state_panel(ui: &mut egui::Ui, s: &Spinor, axis: &mut AxisChoice, u: Option<SceneAxis>, width: f32) {
    ui.vertical(|ui| {
        ui.set_width(width);
        state_column(ui, "|ψ⟩ =", s);
        ui.add_space(Look::of(ui).px(4.0));
        axis_selector(ui, axis, u.map(|x| x.button));
        let dir = axis.dir(u.map(|x| x.dir));
        let name = axis.name(u.map_or("u", |x| x.name));
        prob_bars(ui, s, dir, name, width);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_is_right_handed() {
        let cam = DEFAULT_CAMERA;
        let (r, u, v) = cam.basis();
        let c = [r[1] * u[2] - r[2] * u[1], r[2] * u[0] - r[0] * u[2], r[0] * u[1] - r[1] * u[0]];
        for k in 0..3 {
            assert!((c[k] - v[k]).abs() < 1e-6);
        }
        // the textbook view: +x towards the viewer and to the lower left, +y right, +z up
        let (x, dx) = cam.project(EX);
        let (y, _) = cam.project(EY);
        let (z, _) = cam.project(EZ);
        assert!(dx > 0.5 && x.x < 0.0 && x.y > 0.0);
        assert!(y.x > 0.8);
        assert!(z.y < -0.8);
    }
}
