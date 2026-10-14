//! User interface of the polarisation bench (Saleh & Teich, Sec. 6.1): the
//! wave along z, the x–y plane looking into the beam and the Poincaré sphere
//! (top), the Jones calculus (middle), and the light and the elements (bottom).

use std::f32::consts::PI;

use eframe::egui::{self, pos2, vec2, Align2, Color32, FontId, Pos2, Rect, Sense, Shape, Stroke};

use crate::interferometer_ui::{arrow_head, tag};
use crate::polarisation::{
    amplitudes, angles, describe, factor_phase, factor_phase_vec, field, fmt_angle, fmt_complex, fmt_phase, from_amplitudes, from_sphere, intensity, normal_form, sphere, Element, Jones, Kind, Mat, PolParams, PolPreset, DEG,
};

const BG: Color32 = Color32::from_gray(16);
const LABEL: Color32 = Color32::from_gray(170);
const AXIS: Color32 = Color32::from_gray(120);
const FIELD: Color32 = Color32::from_rgb(240, 95, 55);
const INPUT: Color32 = Color32::from_rgb(235, 235, 235);
const PICK: Color32 = Color32::from_rgb(255, 220, 80);
const EX: Color32 = Color32::from_rgb(90, 170, 255);
const EY: Color32 = Color32::from_rgb(120, 220, 120);
const MAX_ELEMENTS: usize = 5;

fn kind_color(k: Kind) -> Color32 {
    match k {
        Kind::Polariser => Color32::from_rgb(190, 190, 190),
        Kind::HalfWave => Color32::from_rgb(240, 170, 80),
        Kind::QuarterWave => Color32::from_rgb(110, 210, 230),
        Kind::Retarder => Color32::from_rgb(220, 210, 110),
        Kind::Rotator => Color32::from_rgb(190, 150, 255),
    }
}

/// the colour of the state after element k (k = 0: the input)
fn stage_color(p: &PolParams, k: usize) -> Color32 {
    if k == 0 { INPUT } else { kind_color(p.elements[k - 1].kind) }
}

enum Drag {
    Wave,
    Sphere,
    SpherePoint,
}

/// An orthographic camera: turned by `yaw` about the vertical, then tilted by `pitch`.
struct Cam {
    yaw: f32,
    pitch: f32,
    scale: f32,
    centre: Pos2,
}

impl Cam {
    /// screen position and depth (positive towards the viewer) of a point (x right, y up, z towards the viewer)
    fn project(&self, p: [f32; 3]) -> (Pos2, f32) {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let x1 = p[0] * cy + p[2] * sy;
        let z1 = -p[0] * sy + p[2] * cy;
        let y2 = p[1] * cp - z1 * sp;
        let z2 = p[1] * sp + z1 * cp;
        (self.centre + vec2(x1, -y2) * self.scale, z2)
    }

    fn at(&self, p: [f32; 3]) -> Pos2 {
        self.project(p).0
    }

    /// the point on the front of the unit sphere under a screen position
    fn unproject_sphere(&self, q: Pos2) -> [f32; 3] {
        let v = (q - self.centre) / self.scale;
        let (mut x2, mut y2) = (v.x, -v.y);
        let r2 = x2 * x2 + y2 * y2;
        if r2 > 1.0 {
            let r = r2.sqrt();
            x2 /= r;
            y2 /= r;
        }
        let z2 = (1.0 - x2 * x2 - y2 * y2).max(0.0).sqrt();
        // undo the pitch, then the yaw
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let y = y2 * cp + z2 * sp;
        let z1 = -y2 * sp + z2 * cp;
        let x = x2 * cy - z1 * sy;
        let z = x2 * sy + z1 * cy;
        [x, y, z]
    }
}

/// sphere coordinates (u1, u2, u3) in the camera's world: u2 to the right, u3 up, u1 towards the viewer
fn su(u: [f64; 3]) -> [f32; 3] {
    [u[1] as f32, u[2] as f32, u[0] as f32]
}

fn world_to_u(w: [f32; 3]) -> [f64; 3] {
    [w[2] as f64, w[0] as f64, w[1] as f64]
}

pub struct PolUi {
    pub params: PolParams,
    pub notes: String,
    /// ωt of the animation
    wt: f64,
    playing: bool,
    /// periods per second
    speed: f64,
    wave_cam: (f32, f32),
    sphere_cam: (f32, f32),
    drag: Option<Drag>,
}

impl Default for PolUi {
    fn default() -> Self {
        let (params, notes) = PolPreset::Elliptical.setup();
        PolUi { params, notes, wt: 0.0, playing: true, speed: 0.4, wave_cam: (70.0, 15.0), sphere_cam: (-28.0, 18.0), drag: None }
    }
}

impl PolUi {
    pub fn load_preset(&mut self, p: PolPreset) {
        let (params, notes) = p.setup();
        self.params = params;
        self.notes = notes;
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let win = ui.ctx().content_rect();
        if self.playing {
            let dt = ui.input(|i| i.stable_dt).min(0.1) as f64;
            self.wt = (self.wt + dt * self.speed * std::f64::consts::TAU) % (1000.0 * std::f64::consts::TAU);
            ui.ctx().request_repaint();
        }
        egui::Panel::top("p_views")
            .resizable(true)
            .default_size(win.height() * 0.46)
            .min_size(220.0)
            .max_size(win.height() * 0.7)
            .show(ui, |ui| {
                ui.add_space(4.0);
                let full = ui.available_rect_before_wrap();
                let w = full.width();
                let a = full.left() + w * 0.46;
                let b = full.left() + w * 0.71;
                let builder = |r: Rect| egui::UiBuilder::new().max_rect(r).layout(egui::Layout::top_down(egui::Align::Min));
                self.wave_panel(&mut ui.new_child(builder(Rect::from_min_max(full.min, pos2(a - 4.0, full.bottom())))));
                self.xy_panel(&mut ui.new_child(builder(Rect::from_min_max(pos2(a + 4.0, full.top()), pos2(b - 4.0, full.bottom())))));
                self.sphere_panel(&mut ui.new_child(builder(Rect::from_min_max(pos2(b + 4.0, full.top()), full.max))));
                ui.allocate_rect(full, Sense::hover());
            });
        egui::Panel::bottom("p_controls")
            .resizable(true)
            .default_size(250.0)
            .min_size(120.0)
            .max_size(win.height() * 0.45)
            .show(ui, |ui| self.controls_ui(ui));
        egui::CentralPanel::default().show(ui, |ui| self.jones_ui(ui));
    }

    // ------------------------------------------------------------ the wave along z

    fn wave_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.strong("WAVE");
            ui.weak("along z · drag to turn");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button(if self.playing { "⏸" } else { "▶" }).on_hover_text("let time run / stop it").clicked() {
                    self.playing = !self.playing;
                }
                if ui.small_button("reset view").clicked() {
                    self.wave_cam = (70.0, 15.0);
                }
            });
        });
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3.0, BG);
        if resp.drag_started() {
            self.drag = Some(Drag::Wave);
        }
        if matches!(self.drag, Some(Drag::Wave)) {
            let d = resp.drag_delta();
            self.wave_cam.0 = (self.wave_cam.0 + d.x * 0.4).clamp(-180.0, 180.0);
            self.wave_cam.1 = (self.wave_cam.1 + d.y * 0.4).clamp(-80.0, 80.0);
            if !resp.dragged() {
                self.drag = None;
            }
        }
        let p = &self.params;
        let len = p.wavelengths.clamp(0.5, 6.0) as f32;
        let amp = 0.5f32;
        // the field is drawn with z towards the right: world (x, y, z) = (E_x, E_y, z − len/2)
        let scale = (rect.width() / (len + 1.0)).min(rect.height() / (3.8 * amp + 0.4));
        let cam = Cam { yaw: self.wave_cam.0 * PI / 180.0, pitch: self.wave_cam.1 * PI / 180.0, scale, centre: rect.center() };
        let w = |x: f32, y: f32, z: f32| cam.at([x, y, z - 0.5 * len]);
        let states = p.states();
        let n = p.elements.len();
        let zs: Vec<f32> = (0..n).map(|k| len * (k as f32 + 1.0) / (n as f32 + 1.0)).collect();
        let stage_at = |z: f32| zs.iter().filter(|&&zk| zk <= z).count();
        // axes
        let thin = Stroke::new(1.0, AXIS);
        painter.line_segment([w(0.0, 0.0, -0.3), w(0.0, 0.0, len + 0.35)], thin);
        arrow_head(&painter, w(0.0, 0.0, len + 0.35), (w(0.0, 0.0, len + 0.35) - w(0.0, 0.0, len)).normalized(), 7.0, AXIS);
        painter.text(w(0.0, 0.0, len + 0.45), Align2::LEFT_CENTER, "z", FontId::proportional(13.0), LABEL);
        for (e, name) in [([1.6 * amp, 0.0], "x"), ([0.0, 1.6 * amp], "y")] {
            let tip = w(e[0], e[1], -0.3);
            painter.line_segment([w(0.0, 0.0, -0.3), tip], thin);
            arrow_head(&painter, tip, (tip - w(0.0, 0.0, -0.3)).normalized(), 7.0, AXIS);
            painter.text(tip + (tip - w(0.0, 0.0, -0.3)).normalized() * 10.0, Align2::CENTER_CENTER, name, FontId::proportional(13.0), LABEL);
        }
        // the polarisation ellipse at the start and end of every stretch (the "cylinder")
        let ellipse = |j: &Jones, z: f32| -> Vec<Pos2> {
            (0..=64)
                .map(|k| {
                    let (ex, ey) = field(j, 0.0, k as f64 / 64.0 * std::f64::consts::TAU);
                    w(ex as f32 * amp, ey as f32 * amp, z)
                })
                .collect()
        };
        let mut edges = vec![0.0f32];
        edges.extend(&zs);
        edges.push(len);
        for k in 0..=n {
            if intensity(&states[k]) < 1e-9 {
                continue;
            }
            let col = Color32::from_gray(75);
            painter.add(Shape::closed_line(ellipse(&states[k], edges[k]), Stroke::new(1.0, col)));
            painter.add(Shape::closed_line(ellipse(&states[k], edges[k + 1]), Stroke::new(1.0, col)));
        }
        // the elements as plates across the beam, with their axis
        for (k, e) in p.elements.iter().enumerate() {
            let z = zs[k];
            let s = 1.45 * amp;
            let col = kind_color(e.kind);
            let fill = Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), if e.on { 40 } else { 12 });
            let corners = vec![w(-s, -s, z), w(s, -s, z), w(s, s, z), w(-s, s, z)];
            painter.add(Shape::convex_polygon(corners.clone(), fill, Stroke::new(1.0, col.gamma_multiply(if e.on { 0.9 } else { 0.3 }))));
            let (sa, ca) = ((e.angle_deg * DEG) as f32).sin_cos();
            if e.kind != Kind::Rotator {
                let a = w(-ca * s, -sa * s, z);
                let b = w(ca * s, sa * s, z);
                painter.line_segment([a, b], Stroke::new(2.0, col.gamma_multiply(if e.on { 1.0 } else { 0.3 })));
            }
            let top = w(0.0, s + 0.12, z);
            painter.text(top, Align2::CENTER_BOTTOM, format!("{} {}", e.kind.short(), fmt_angle(e.angle_deg)), FontId::proportional(11.0), col);
        }
        // the field: its tip at this moment, and arrows from the axis
        let samples = 720;
        let mut seg: Vec<Pos2> = vec![];
        let mut last_stage = 0;
        for i in 0..=samples {
            let z = len * i as f32 / samples as f32;
            let st = stage_at(z);
            if st != last_stage {
                painter.add(Shape::line(std::mem::take(&mut seg), Stroke::new(2.2, FIELD)));
                last_stage = st;
            }
            let (ex, ey) = field(&states[st], std::f64::consts::TAU * z as f64, self.wt);
            let tip = w(ex as f32 * amp, ey as f32 * amp, z);
            if i % 12 == 0 {
                painter.line_segment([w(0.0, 0.0, z), tip], Stroke::new(1.0, FIELD.gamma_multiply(0.45)));
            }
            seg.push(tip);
        }
        painter.add(Shape::line(seg, Stroke::new(2.2, FIELD)));
        if p.components {
            for (c, col) in [(0usize, EX), (1, EY)] {
                let mut line: Vec<Pos2> = vec![];
                let mut last = 0;
                for i in 0..=samples {
                    let z = len * i as f32 / samples as f32;
                    let st = stage_at(z);
                    if st != last {
                        painter.add(Shape::line(std::mem::take(&mut line), Stroke::new(1.2, col)));
                        last = st;
                    }
                    let (ex, ey) = field(&states[st], std::f64::consts::TAU * z as f64, self.wt);
                    let v = if c == 0 { ex } else { ey } as f32 * amp;
                    line.push(if c == 0 { w(v, 0.0, z) } else { w(0.0, v, z) });
                }
                painter.add(Shape::line(line, Stroke::new(1.2, col)));
            }
        }
        // λ, as in the book's figure
        let y = rect.top() + 16.0;
        let (a, b) = (w(0.0, 0.0, 0.0).x, w(0.0, 0.0, 1.0).x);
        if (b - a).abs() > 20.0 {
            painter.line_segment([pos2(a, y), pos2(b, y)], Stroke::new(1.0, LABEL));
            arrow_head(&painter, pos2(a, y), vec2(-1.0, 0.0), 6.0, LABEL);
            arrow_head(&painter, pos2(b, y), vec2(1.0, 0.0), 6.0, LABEL);
            painter.text(pos2(0.5 * (a + b), y - 2.0), Align2::CENTER_BOTTOM, "λ", FontId::proportional(13.0), LABEL);
        }
        let mut leg = vec![("tip of the field at this moment", FIELD)];
        if p.components {
            leg.push(("E_x", EX));
            leg.push(("E_y", EY));
        }
        let mut ly = rect.bottom() - 6.0;
        for (t, c) in leg.into_iter().rev() {
            let r = tag(&painter, pos2(rect.left() + 6.0, ly), Align2::LEFT_BOTTOM, t, c, 10.0);
            ly = r.top() - 2.0;
        }
    }

    // ------------------------------------------------------------ the x–y plane

    fn xy_panel(&mut self, ui: &mut egui::Ui) {
        let p = &self.params;
        let states = p.states();
        let k = p.shown_stage();
        let j = states[k];
        ui.horizontal(|ui| {
            ui.strong("x–y PLANE");
            ui.weak(if k == 0 { "input".to_string() } else { format!("after element {k}") });
        });
        let (rect, _) = ui.allocate_exact_size(ui.available_size(), Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3.0, BG);
        let c = rect.center() + vec2(0.0, 6.0);
        let r = (rect.width().min(rect.height() - 40.0) * 0.40).max(20.0);
        let to = |x: f64, y: f64| c + vec2(x as f32, -y as f32) * r;
        let thin = Stroke::new(1.0, AXIS);
        for (d, name) in [(vec2(1.0, 0.0), "x"), (vec2(0.0, -1.0), "y")] {
            painter.line_segment([c - d * r * 1.15, c + d * r * 1.15], thin);
            arrow_head(&painter, c + d * r * 1.15, d, 7.0, AXIS);
            painter.text(c + d * r * 1.15 + d * 9.0 + vec2(2.0, 0.0), Align2::CENTER_CENTER, name, FontId::proportional(13.0), LABEL);
        }
        let total = intensity(&j);
        if total < 1e-9 {
            painter.text(c, Align2::CENTER_CENTER, "no light", FontId::proportional(13.0), LABEL);
            return;
        }
        // amplitudes relative to the input (intensity 1)
        let path = |j: &Jones| -> Vec<Pos2> {
            (0..=96)
                .map(|n| {
                    let (x, y) = field(j, 0.0, n as f64 / 96.0 * std::f64::consts::TAU);
                    to(x, y)
                })
                .collect()
        };
        if k > 0 && intensity(&states[0]) > 1e-9 {
            painter.extend(Shape::dashed_line(&path(&states[0]), Stroke::new(1.0, Color32::from_gray(90)), 4.0, 4.0));
        }
        // the rectangle 2a_x × 2a_y and the angles ψ and χ (Fig. 6.1-5)
        let (ax, ay, _) = amplitudes(&j);
        let rect_pts = [to(-ax, -ay), to(ax, -ay), to(ax, ay), to(-ax, ay), to(-ax, -ay)];
        painter.extend(Shape::dashed_line(&rect_pts, Stroke::new(1.0, Color32::from_gray(80)), 3.0, 3.0));
        let (psi, chi) = angles(&j);
        let amp = total.sqrt();
        let (major, minor) = (amp * chi.cos(), amp * chi.sin().abs());
        let (sp, cp) = psi.sin_cos();
        // for circular light the orientation ψ is undefined
        let circular = chi.abs() > 44.5 * DEG;
        if !circular {
            painter.line_segment([to(-major * cp, -major * sp), to(major * cp, major * sp)], Stroke::new(1.0, Color32::from_gray(150)));
            let diag = (psi + chi.abs()).sin_cos();
            let dlen = (major * major + minor * minor).sqrt();
            painter.line_segment([c, to(dlen * diag.1, dlen * diag.0)], Stroke::new(1.0, Color32::from_gray(110)));
        }
        let arc = |a0: f64, a1: f64, rr: f32, col: Color32, text: &str| {
            let pts: Vec<Pos2> = (0..=20).map(|i| a0 + (a1 - a0) * i as f64 / 20.0).map(|a| c + vec2(a.cos() as f32, -a.sin() as f32) * rr).collect();
            painter.add(Shape::line(pts, Stroke::new(1.0, col)));
            let m = 0.5 * (a0 + a1);
            painter.text(c + vec2(m.cos() as f32, -m.sin() as f32) * (rr + 10.0), Align2::CENTER_CENTER, text, FontId::proportional(11.0), col);
        };
        if psi.abs() > 1.0 * DEG && !circular {
            arc(0.0, psi, r * 0.32, PICK, "ψ");
        }
        if chi.abs() > 1.0 * DEG && !circular {
            arc(psi, psi + chi.abs(), r * 0.5, EX, "χ");
        }
        // the ellipse, with the sense of rotation
        let pts = path(&j);
        painter.add(Shape::closed_line(pts.clone(), Stroke::new(2.0, FIELD)));
        if chi.abs() > 1.5 * DEG {
            for n in [10usize, 58] {
                let d = (pts[n + 1] - pts[n]).normalized();
                arrow_head(&painter, pts[n + 1], d, 8.0, FIELD);
            }
        } else {
            for s in [-1.0f64, 1.0] {
                let tip = to(s * major * cp, s * major * sp);
                arrow_head(&painter, tip, vec2((s * cp) as f32, -(s * sp) as f32), 8.0, FIELD);
            }
        }
        // the field now and a short trail
        let (ex, ey) = field(&j, 0.0, self.wt);
        let trail: Vec<Pos2> = (0..=20).map(|n| field(&j, 0.0, self.wt - n as f64 * 0.05)).map(|(x, y)| to(x, y)).collect();
        painter.add(Shape::line(trail, Stroke::new(3.0, PICK.gamma_multiply(0.5))));
        painter.line_segment([c, to(ex, ey)], Stroke::new(2.0, PICK));
        painter.circle_filled(to(ex, ey), 3.5, PICK);
        painter.text(rect.left_bottom() + vec2(5.0, -4.0), Align2::LEFT_BOTTOM, "looking into the oncoming beam (z towards you)", FontId::proportional(10.0), LABEL);
        painter.text(rect.left_top() + vec2(5.0, 4.0), Align2::LEFT_TOP, describe(&j), FontId::proportional(11.0), PICK);
        if k > 0 {
            painter.text(rect.right_top() + vec2(-5.0, 4.0), Align2::RIGHT_TOP, "dashed: input", FontId::proportional(10.0), Color32::from_gray(110));
        }
    }

    // ------------------------------------------------------------ the Poincaré sphere

    fn sphere_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.strong("POINCARÉ SPHERE");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("reset view").clicked() {
                    self.sphere_cam = (-28.0, 18.0);
                }
            });
        });
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3.0, BG);
        let radius = (rect.width().min(rect.height()) * 0.36).max(20.0);
        let cam = Cam { yaw: self.sphere_cam.0 * PI / 180.0, pitch: self.sphere_cam.1 * PI / 180.0, scale: radius, centre: rect.center() + vec2(0.0, 8.0) };
        let p = self.params.clone();
        let states = p.states();
        let input_at = cam.project(su(sphere(&states[0])));
        // drag the input point, or turn the sphere
        if resp.drag_started() {
            let near = resp.interact_pointer_pos().is_some_and(|q| (q - input_at.0).length() < 14.0 && input_at.1 > -0.05);
            self.drag = Some(if near { Drag::SpherePoint } else { Drag::Sphere });
        }
        match self.drag {
            Some(Drag::Sphere) => {
                let d = resp.drag_delta();
                self.sphere_cam.0 = (self.sphere_cam.0 + d.x * 0.5).clamp(-180.0, 180.0);
                self.sphere_cam.1 = (self.sphere_cam.1 + d.y * 0.5).clamp(-85.0, 85.0);
                if !resp.dragged() {
                    self.drag = None;
                }
            }
            Some(Drag::SpherePoint) => {
                if let Some(q) = resp.interact_pointer_pos() {
                    let u = world_to_u(cam.unproject_sphere(q));
                    self.params.set_input(&from_sphere(u));
                }
                if !resp.dragged() {
                    self.drag = None;
                }
            }
            _ => {}
        }
        if resp.hovered() && (resp.hover_pos().is_some_and(|q| (q - input_at.0).length() < 14.0) || matches!(self.drag, Some(Drag::SpherePoint))) {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
        }
        let c = cam.centre;
        painter.circle(c, radius, Color32::from_rgba_premultiplied(40, 44, 52, 120), Stroke::new(1.2, Color32::from_gray(110)));
        // a great circle, solid in front and dashed behind
        let circle = |f: &dyn Fn(f64) -> [f64; 3], col: Color32| {
            let mut front: Vec<Pos2> = vec![];
            for i in 0..=96 {
                let t = i as f64 / 96.0 * std::f64::consts::TAU;
                let (q, depth) = cam.project(su(f(t)));
                if depth >= 0.0 {
                    front.push(q);
                } else {
                    if front.len() > 1 {
                        painter.add(Shape::line(std::mem::take(&mut front), Stroke::new(1.0, col)));
                    }
                    front.clear();
                    painter.circle_filled(q, 0.6, col.gamma_multiply(0.6));
                }
            }
            if front.len() > 1 {
                painter.add(Shape::line(front, Stroke::new(1.0, col)));
            }
        };
        circle(&|t| [t.cos(), t.sin(), 0.0], Color32::from_gray(130));
        circle(&|t| [t.cos(), 0.0, t.sin()], Color32::from_gray(70));
        circle(&|t| [0.0, t.cos(), t.sin()], Color32::from_gray(70));
        // axes and the named points
        for (u, name, pt) in [
            ([1.0, 0.0, 0.0], "u1", "H"),
            ([0.0, 1.0, 0.0], "u2", "+45°"),
            ([0.0, 0.0, 1.0], "u3", "R ↻"),
        ] {
            let tip = cam.at(su([u[0] * 1.3, u[1] * 1.3, u[2] * 1.3]));
            painter.line_segment([c, tip], Stroke::new(1.0, AXIS));
            arrow_head(&painter, tip, (tip - c).normalized(), 7.0, AXIS);
            painter.text(tip + (tip - c).normalized() * 10.0, Align2::CENTER_CENTER, name, FontId::proportional(12.0), LABEL);
            let on = cam.at(su(u));
            painter.circle_filled(on, 2.0, LABEL);
            painter.text(on + vec2(5.0, -5.0), Align2::LEFT_BOTTOM, pt, FontId::proportional(10.0), LABEL);
        }
        for (u, pt) in [([-1.0, 0.0, 0.0], "V"), ([0.0, -1.0, 0.0], "−45°"), ([0.0, 0.0, -1.0], "L ↺")] {
            let (on, depth) = cam.project(su(u));
            painter.circle_filled(on, 2.0, LABEL.gamma_multiply(if depth >= 0.0 { 1.0 } else { 0.5 }));
            painter.text(on + vec2(5.0, 5.0), Align2::LEFT_TOP, pt, FontId::proportional(10.0), LABEL.gamma_multiply(if depth >= 0.0 { 1.0 } else { 0.6 }));
        }
        // the axes of the elements and the paths of the state through them
        for (k, e) in p.elements.iter().enumerate() {
            if !e.on {
                continue;
            }
            let col = kind_color(e.kind);
            let a = 2.0 * e.angle_deg * DEG;
            match e.kind {
                Kind::Polariser => {
                    let (q, _) = cam.project(su([a.cos(), a.sin(), 0.0]));
                    painter.circle_stroke(q, 5.0, Stroke::new(1.5, col));
                }
                Kind::Rotator => {}
                _ => {
                    let (q1, _) = cam.project(su([a.cos() * 1.15, a.sin() * 1.15, 0.0]));
                    let (q2, _) = cam.project(su([-a.cos() * 1.15, -a.sin() * 1.15, 0.0]));
                    painter.extend(Shape::dashed_line(&[q2, q1], Stroke::new(1.0, col.gamma_multiply(0.7)), 4.0, 3.0));
                }
            }
            if intensity(&states[k]) < 1e-9 {
                continue;
            }
            let path = e.sphere_path(&states[k]);
            let pts: Vec<(Pos2, f32)> = path.iter().map(|u| cam.project(su(*u))).collect();
            if e.kind == Kind::Polariser {
                if pts.len() == 2 {
                    painter.extend(Shape::dashed_line(&[pts[0].0, pts[1].0], Stroke::new(1.5, col), 5.0, 3.0));
                }
            } else {
                for w in pts.windows(2) {
                    let front = w[0].1 >= 0.0 && w[1].1 >= 0.0;
                    painter.line_segment([w[0].0, w[1].0], Stroke::new(if front { 2.0 } else { 1.0 }, col.gamma_multiply(if front { 1.0 } else { 0.5 })));
                }
                if let (Some(a), Some(b)) = (pts.get(pts.len().saturating_sub(4)), pts.last()) {
                    arrow_head(&painter, b.0, (b.0 - a.0).normalized(), 7.0, col);
                }
            }
        }
        // the angles 2ψ and 2χ of the shown state
        let k = p.shown_stage();
        let shown = states[k];
        if intensity(&shown) > 1e-9 {
            let (psi, chi) = angles(&shown);
            let (sp2, cp2) = (2.0 * psi).sin_cos();
            let base = cam.at(su([cp2, sp2, 0.0]));
            let point = cam.at(su(sphere(&shown)));
            painter.line_segment([c, point], Stroke::new(1.0, PICK.gamma_multiply(0.7)));
            painter.extend(Shape::dashed_line(&[c, base], Stroke::new(1.0, PICK.gamma_multiply(0.5)), 3.0, 3.0));
            let arc_eq: Vec<Pos2> = (0..=24).map(|i| 2.0 * psi * i as f64 / 24.0).map(|a| cam.at(su([0.35 * a.cos(), 0.35 * a.sin(), 0.0]))).collect();
            // at the poles the longitude 2ψ is undefined
            if psi.abs() > 1.0 * DEG && chi.abs() < 44.5 * DEG {
                painter.add(Shape::line(arc_eq.clone(), Stroke::new(1.2, PICK)));
                painter.text(*arc_eq.last().expect("arc") + vec2(4.0, 6.0), Align2::LEFT_TOP, "2ψ", FontId::proportional(11.0), PICK);
            }
            if chi.abs() > 1.0 * DEG {
                let arc_m: Vec<Pos2> = (0..=24)
                    .map(|i| 2.0 * chi * i as f64 / 24.0)
                    .map(|b| cam.at(su([0.55 * b.cos() * cp2, 0.55 * b.cos() * sp2, 0.55 * b.sin()])))
                    .collect();
                painter.add(Shape::line(arc_m.clone(), Stroke::new(1.2, EX)));
                painter.text(arc_m[12] + vec2(6.0, 0.0), Align2::LEFT_CENTER, "2χ", FontId::proportional(11.0), EX);
            }
        }
        // the states: input and after each element
        for (k, s) in states.iter().enumerate() {
            if intensity(s) < 1e-9 {
                continue;
            }
            let (q, depth) = cam.project(su(sphere(s)));
            let col = stage_color(&p, k);
            let r = if k == 0 { 6.0 } else { 4.5 };
            if depth >= 0.0 {
                painter.circle(q, r, col, Stroke::new(1.0, Color32::BLACK));
            } else {
                painter.circle_stroke(q, r, Stroke::new(1.5, col));
            }
        }
        painter.text(
            rect.left_bottom() + vec2(5.0, -4.0),
            Align2::LEFT_BOTTOM,
            "drag the white point to choose the light; drag elsewhere to turn",
            FontId::proportional(10.0),
            LABEL,
        );
        painter.text(rect.left_top() + vec2(5.0, 4.0), Align2::LEFT_TOP, "upper half: right-handed", FontId::proportional(10.0), LABEL);
    }

    // ------------------------------------------------------------ Jones calculus

    fn jones_ui(&mut self, ui: &mut egui::Ui) {
        let p = self.params.clone();
        let states = p.states();
        ui.horizontal(|ui| {
            ui.strong("JONES CALCULUS");
            ui.weak("E(z, t) = Re[J e^{i(kz − ωt)}] as in the problem sets · J_out = M_n … M₂ M₁ J_in · vectors up to a common phase");
        });
        egui::ScrollArea::vertical().id_salt("p_jones").auto_shrink([false, false]).show(ui, |ui| {
            let font = FontId::proportional(13.0);
            // one row per step: what it is, the matrix, the vector after it, and in words
            let row_h = 46.0;
            let mut rows: Vec<(String, Option<Mat>, Jones, Color32)> = vec![("input light".into(), None, states[0], INPUT)];
            for (k, e) in p.elements.iter().enumerate() {
                let sub = subscript(k + 1);
                let name = if e.on { format!("M{sub}: {}", e.label()) } else { format!("M{sub}: {} (off)", e.label()) };
                rows.push((name, Some(e.matrix()), states[k + 1], kind_color(e.kind)));
            }
            for (k, (name, m, j, col)) in rows.iter().enumerate() {
                let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), row_h), Sense::click());
                if resp.clicked() {
                    self.params.stage = if k == p.elements.len() { 99 } else { k };
                }
                let painter = ui.painter_at(rect);
                if k == p.shown_stage() {
                    painter.rect_filled(rect, 3.0, Color32::from_rgba_premultiplied(45, 40, 15, 60));
                }
                let mut x = rect.left() + 6.0;
                let mid = rect.center().y;
                let g = painter.layout_no_wrap(name.clone(), FontId::proportional(12.0), *col);
                let label_w = 250.0f32.max(g.size().x + 10.0).min(rect.width() * 0.32);
                painter.galley(pos2(x, mid - g.size().y * 0.5), g, *col);
                x += label_w;
                if let Some(m) = m {
                    let (alpha, mm) = factor_phase(m);
                    let entries = [[fmt_complex(mm[0][0]), fmt_complex(mm[0][1])], [fmt_complex(mm[1][0]), fmt_complex(mm[1][1])]];
                    x = draw_bracket(&painter, pos2(x, mid), &fmt_phase(alpha), &entries, font.clone(), LABEL) + 18.0;
                    painter.text(pos2(x - 9.0, mid), Align2::CENTER_CENTER, "→", FontId::monospace(13.0), LABEL);
                }
                let sub = if k == 0 { "_in".to_string() } else { subscript(k) };
                let tl = painter.layout_no_wrap(format!("J{sub} ="), font.clone(), *col);
                painter.galley(pos2(x, mid - tl.size().y * 0.5), tl.clone(), *col);
                x += tl.size().x + 6.0;
                let i = intensity(j);
                if i < 1e-12 {
                    painter.text(pos2(x, mid), Align2::LEFT_CENTER, "0  (no light)", font.clone(), LABEL);
                    continue;
                }
                // the vector: the common phase in front, then the components
                let (common, jj) = factor_phase_vec(j);
                let comps = [[fmt_complex(jj[0])], [fmt_complex(jj[1])]];
                x = draw_bracket(&painter, pos2(x, mid), &fmt_phase(common), &comps, font.clone(), LABEL) + 16.0;
                painter.text(pos2(x, mid), Align2::LEFT_CENTER, format!("{}   intensity {:.3}", describe(j), i), FontId::proportional(12.0), *col);
            }
            if p.elements.len() > 1 {
                let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), row_h), Sense::hover());
                let painter = ui.painter_at(rect);
                let mid = rect.center().y;
                let names: Vec<String> = (1..=p.elements.len()).rev().map(|k| format!("M{}", subscript(k))).collect();
                let label = format!("together: M = {}", names.join(""));
                let g = painter.layout_no_wrap(label, FontId::proportional(12.0), PICK);
                let x0 = rect.left() + 6.0;
                painter.galley(pos2(x0, mid - g.size().y * 0.5), g.clone(), PICK);
                let (alpha, mm) = factor_phase(&p.total());
                let entries = [[fmt_complex(mm[0][0]), fmt_complex(mm[0][1])], [fmt_complex(mm[1][0]), fmt_complex(mm[1][1])]];
                let x = x0 + 250.0f32.max(g.size().x + 10.0).min(rect.width() * 0.32);
                draw_bracket(&painter, pos2(x, mid), &fmt_phase(alpha), &entries, font.clone(), PICK);
            }
            ui.add_space(4.0);
            ui.weak(
                "Click a row to show its state in the x–y plane. A λ/4 plate with its fast axis vertical is \
                 e^{iπ/4}[[1, 0], [0, −i]], M_QWP of PS03. In Saleh & Teich's convention e^{j(ωt − kz)} every number \
                 here appears complex conjugated.",
            );
        });
    }

    // ------------------------------------------------------------ controls

    fn controls_ui(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        if !self.notes.trim().is_empty() {
            ui.horizontal_wrapped(|ui| {
                ui.strong("About this setup:");
                ui.label(&self.notes);
            });
            ui.separator();
        }
        egui::ScrollArea::vertical().id_salt("pol_controls").auto_shrink([false, false]).show(ui, |ui| {
            ui.columns(3, |cols| {
                self.input_ui(&mut cols[0]);
                self.elements_ui(&mut cols[1]);
                self.view_ui(&mut cols[2]);
            });
        });
    }

    fn input_ui(&mut self, ui: &mut egui::Ui) {
        let p = &mut self.params;
        ui.strong("INPUT LIGHT");
        ui.horizontal_wrapped(|ui| {
            for (t, psi, chi, hint) in [
                ("H", 0.0, 0.0, "linear, horizontal (along x)"),
                ("V", 90.0, 0.0, "linear, vertical (along y)"),
                ("+45°", 45.0, 0.0, "linear at +45°"),
                ("−45°", -45.0, 0.0, "linear at −45°"),
                ("R ↻", 0.0, 45.0, "right circular, (1, −i)/√2: clockwise looking into the beam"),
                ("L ↺", 0.0, -45.0, "left circular, (1, i)/√2: counter-clockwise looking into the beam"),
            ] {
                let on = (p.psi_deg - psi).abs() < 1e-6 && (p.chi_deg - chi).abs() < 1e-6;
                if ui.selectable_label(on, t).on_hover_text(hint).clicked() {
                    p.psi_deg = psi;
                    p.chi_deg = chi;
                }
            }
        });
        egui::Grid::new("p_input").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
            ui.label("orientation ψ");
            ui.add(egui::Slider::new(&mut p.psi_deg, -90.0..=90.0).suffix("°").fixed_decimals(1)).on_hover_text("angle of the major axis from x");
            ui.end_row();
            ui.label("ellipticity χ");
            ui.add(egui::Slider::new(&mut p.chi_deg, -45.0..=45.0).suffix("°").fixed_decimals(1))
                .on_hover_text("tan χ = minor/major axis; χ > 0 right-handed (↻), χ < 0 left-handed (↺), 0 linear, ±45° circular");
            ui.end_row();
            // the same light as two components
            let j = p.input();
            let (ax, _, phi) = amplitudes(&j);
            let mut ax2 = ax;
            let mut phi_deg = phi / DEG;
            ui.label("a_x  (a_y = √(1 − a_x²))");
            // clamped only on edits: these are derived from ψ and χ and must not be rounded every frame
            let a = ui.add(egui::Slider::new(&mut ax2, 0.0..=1.0).fixed_decimals(3).clamping(egui::SliderClamping::Edits));
            ui.end_row();
            ui.label("phase φ = φ_y − φ_x");
            let b = ui
                .add(egui::Slider::new(&mut phi_deg, -180.0..=180.0).suffix("°").fixed_decimals(1).clamping(egui::SliderClamping::Edits))
                .on_hover_text("in this convention φ = +90° with a_x = a_y is left circular; in Saleh & Teich's it is right circular");
            ui.end_row();
            if a.changed() || b.changed() {
                let ay2 = (1.0 - ax2 * ax2).max(0.0).sqrt();
                p.set_input(&from_amplitudes(ax2, ay2, phi_deg * DEG));
            }
        });
        let j = normal_form(&p.input());
        let u = sphere(&j);
        ui.weak(format!("J_in = ({}, {})   u = ({:.2}, {:.2}, {:.2})", fmt_complex(j[0]), fmt_complex(j[1]), u[0], u[1], u[2]));
        ui.weak(describe(&j));
    }

    fn elements_ui(&mut self, ui: &mut egui::Ui) {
        let p = &mut self.params;
        ui.strong("OPTICAL ELEMENTS");
        ui.weak("in the order the light meets them");
        let mut remove: Option<usize> = None;
        let mut swap: Option<usize> = None;
        let n = p.elements.len();
        for (k, e) in p.elements.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.colored_label(kind_color(e.kind), format!("M{}", subscript(k + 1)));
                ui.checkbox(&mut e.on, "").on_hover_text("in the beam");
                egui::ComboBox::from_id_salt(("p_kind", k)).selected_text(e.kind.short()).width(54.0).show_ui(ui, |ui| {
                    for kind in Kind::ALL {
                        ui.selectable_value(&mut e.kind, kind, kind.label());
                    }
                });
                let what = match e.kind {
                    Kind::Polariser => "axis",
                    Kind::Rotator => "by",
                    _ => "fast axis",
                };
                ui.label(what);
                ui.add(egui::DragValue::new(&mut e.angle_deg).range(-180.0..=180.0).speed(0.5).suffix("°"));
                if e.kind == Kind::Retarder {
                    ui.label("Γ");
                    ui.add(egui::DragValue::new(&mut e.retard_deg).range(0.0..=360.0).speed(1.0).suffix("°"));
                }
                if k > 0 && ui.small_button("⏶").on_hover_text("earlier in the beam").clicked() {
                    swap = Some(k - 1);
                }
                if k + 1 < n && ui.small_button("⏷").on_hover_text("later in the beam").clicked() {
                    swap = Some(k);
                }
                if ui.small_button("✖").on_hover_text("remove").clicked() {
                    remove = Some(k);
                }
            });
        }
        if let Some(k) = swap {
            p.elements.swap(k, k + 1);
        }
        if let Some(k) = remove {
            p.elements.remove(k);
        }
        if p.elements.is_empty() {
            ui.weak("none: the light goes straight through");
        }
        ui.horizontal_wrapped(|ui| {
            for (t, kind, angle) in [("+ polariser", Kind::Polariser, 0.0), ("+ λ/2", Kind::HalfWave, 0.0), ("+ λ/4", Kind::QuarterWave, 0.0), ("+ waveplate", Kind::Retarder, 0.0), ("+ rotator", Kind::Rotator, 30.0)] {
                if ui.add_enabled(p.elements.len() < MAX_ELEMENTS, egui::Button::new(t)).on_hover_text(kind.label()).clicked() {
                    p.elements.push(Element::new(kind, angle));
                    p.stage = 99;
                }
            }
        });
    }

    fn view_ui(&mut self, ui: &mut egui::Ui) {
        ui.strong("VIEW");
        let n = self.params.elements.len();
        ui.horizontal_wrapped(|ui| {
            ui.label("x–y plane and angles:");
            let shown = self.params.shown_stage();
            if ui.selectable_label(shown == 0, "input").clicked() {
                self.params.stage = 0;
            }
            for k in 1..=n {
                if ui.selectable_label(shown == k, format!("after M{}", subscript(k))).clicked() {
                    self.params.stage = if k == n { 99 } else { k };
                }
            }
        });
        egui::Grid::new("p_view").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
            ui.label("time");
            ui.horizontal(|ui| {
                if ui.button(if self.playing { "⏸ stop" } else { "▶ run" }).clicked() {
                    self.playing = !self.playing;
                }
                ui.add(egui::Slider::new(&mut self.speed, 0.05..=2.0).logarithmic(true).suffix(" periods/s").fixed_decimals(2));
            });
            ui.end_row();
            ui.label("3D view");
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.params.components, "E_x and E_y");
                ui.add(egui::Slider::new(&mut self.params.wavelengths, 1.0..=5.0).suffix(" λ").fixed_decimals(1));
            });
            ui.end_row();
        });
        ui.weak(
            "Pause the time to see the helix at one moment; let it run to see the field turn in the x–y plane. \
             The angles 2ψ and 2χ on the sphere belong to the state shown in the x–y plane.",
        );
    }
}

fn subscript(n: usize) -> String {
    n.to_string().chars().map(|c| char::from_u32(0x2080 + c.to_digit(10).unwrap_or(0)).unwrap_or(c)).collect()
}

/// A column vector or matrix in brackets, with a phase factor e^{…} in front.
/// Returns the right edge.
fn draw_bracket<const N: usize>(painter: &egui::Painter, left_mid: Pos2, phase: &str, entries: &[[String; N]; 2], font: FontId, col: Color32) -> f32 {
    let mut x = left_mid.x;
    let mid = left_mid.y;
    if !phase.is_empty() {
        // "e^{iπ/4}": the exponent raised and smaller
        let exp = phase.trim_start_matches("e^{").trim_end_matches('}').to_string();
        let ge = painter.layout_no_wrap("e".into(), font.clone(), col);
        let gx = painter.layout_no_wrap(exp, FontId::proportional(font.size * 0.72), col);
        painter.galley(pos2(x, mid - ge.size().y * 0.5), ge.clone(), col);
        painter.galley(pos2(x + ge.size().x, mid - ge.size().y * 0.5 - gx.size().y * 0.45), gx.clone(), col);
        x += ge.size().x + gx.size().x + 4.0;
    }
    let galleys: Vec<Vec<_>> = entries.iter().map(|row| row.iter().map(|t| painter.layout_no_wrap(t.clone(), font.clone(), col)).collect()).collect();
    let col_w: Vec<f32> = (0..N).map(|c| galleys.iter().map(|r| r[c].size().x).fold(0.0, f32::max)).collect();
    let line_h = galleys[0][0].size().y + 2.0;
    let h = 2.0 * line_h;
    let top = mid - 0.5 * h;
    let gap = 14.0;
    let width: f32 = col_w.iter().sum::<f32>() + gap * (N as f32 - 1.0) + 12.0;
    let stroke = Stroke::new(1.2, col);
    let bracket = |xb: f32, dir: f32| {
        painter.line_segment([pos2(xb, top), pos2(xb, top + h)], stroke);
        painter.line_segment([pos2(xb, top), pos2(xb + dir * 4.0, top)], stroke);
        painter.line_segment([pos2(xb, top + h), pos2(xb + dir * 4.0, top + h)], stroke);
    };
    bracket(x, 1.0);
    bracket(x + width, -1.0);
    for (r, row) in galleys.into_iter().enumerate() {
        let mut cx = x + 6.0;
        for (c, g) in row.into_iter().enumerate() {
            let w = g.size().x;
            painter.galley(pos2(cx + 0.5 * (col_w[c] - w), top + r as f32 * line_h + 1.0), g, col);
            cx += col_w[c] + gap;
        }
    }
    x + width
}
