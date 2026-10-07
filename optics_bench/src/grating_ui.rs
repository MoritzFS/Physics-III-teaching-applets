//! User interface of the grating bench: the spectrometer seen from above and
//! the resolution (top), the spectrum on the camera or in all directions
//! (middle), and the lamps, the grating and the spectrometer (bottom).

use std::f64::consts::PI;

use eframe::egui::{self, pos2, vec2, Align2, Color32, FontId, Pos2, Rect, Sense, Shape, Stroke, StrokeKind};

use crate::dispersion_ui::{fmt_tick, readout, ticks};
use crate::grating::{
    calibrate, drop_side_lobes, far_field, find_peaks, record, Axis, Calibration, FarAxis, FarField, GratingParams, GratingPreset, Groove, Lamp, Recording, DEG,
};
use crate::interferometer_ui::{arrow_head, dashed, polyline, tag};
use crate::worker::Worker;

const BG: Color32 = Color32::from_gray(16);
const LABEL: Color32 = Color32::from_gray(170);
const PICK: Color32 = Color32::from_rgb(255, 220, 80);
const SOURCE: Color32 = Color32::from_rgb(110, 205, 255);
const REFERENCE: Color32 = Color32::from_rgb(255, 165, 70);
const LENS: Color32 = Color32::from_rgb(150, 190, 245);
const SILVER: Color32 = Color32::from_rgb(205, 210, 220);
const SLIT: Color32 = Color32::from_rgb(230, 70, 60);
const CAMERA: Color32 = Color32::from_rgb(110, 225, 120);

#[derive(Clone, PartialEq)]
struct Request {
    params: GratingParams,
    far_cols: usize,
}

struct Response {
    rec: Recording,
    far: Option<FarField>,
    params: GratingParams,
}

/// what the recording depends on (not the view)
fn record_key(p: &GratingParams) -> GratingParams {
    let mut q = p.clone();
    q.axis = Axis::Exact;
    q.log = false;
    q.view = (0.0, 1.0);
    q.phasor_frac = 0.0;
    q.fit_degree = 0;
    if !q.far_field {
        q.far_view = (0.0, 0.0);
        q.far_view_rad = (0.0, 0.0);
        q.far_axis = FarAxis::Path;
        q.compare_few = false;
    }
    q
}

enum Drag {
    Grating { start_angle: f64, start_psi: f64 },
    Pan { last: f32 },
}

pub struct GratingUi {
    pub params: GratingParams,
    pub notes: String,
    worker: Option<Worker<Request, Response>>,
    rec: Option<Recording>,
    far: Option<FarField>,
    /// the parameters of the shown recording
    shown: Option<GratingParams>,
    drag: Option<Drag>,
    scan: bool,
}

impl Default for GratingUi {
    fn default() -> Self {
        let (params, notes) = GratingPreset::HowItWorks.setup();
        GratingUi { params, notes, worker: None, rec: None, far: None, shown: None, drag: None, scan: false }
    }
}

fn lamp_color(nm: f64) -> Color32 {
    let c = crate::fourier::wavelength_rgb(nm.clamp(380.0, 780.0) as f32);
    let k = if nm > 780.0 { 0.45 } else { 1.0 };
    Color32::from_rgb((c[0] * k * 255.0) as u8, ((c[1] * k) * 255.0) as u8, ((c[2] * k) * 255.0) as u8)
}

/// a tick label with as many decimals as the tick step needs
fn fmt_step(v: f64, step: f64) -> String {
    let dec = (-(step.abs().max(1e-12)).log10().floor()).max(0.0) as usize;
    if dec == 0 { fmt_tick(v) } else { format!("{v:.dec$}") }
}

fn tick_step(t: &[f64]) -> f64 {
    if t.len() >= 2 { t[1] - t[0] } else { 1.0 }
}

fn fmt_nm(v: f64) -> String {
    if v >= 10.0 {
        format!("{v:.2} nm")
    } else if v >= 0.1 {
        format!("{v:.3} nm")
    } else {
        format!("{:.1} pm", v * 1e3)
    }
}

fn fmt_mm(v: f64) -> String {
    if v.abs() >= 1.0 {
        format!("{v:.2} mm")
    } else {
        format!("{:.1} µm", v * 1e3)
    }
}

impl GratingUi {
    pub fn load_preset(&mut self, p: GratingPreset) {
        let (params, notes) = p.setup();
        self.params = params;
        self.notes = notes;
        self.scan = false;
    }

    pub fn update(&mut self, ctx: &egui::Context) {
        let worker = self.worker.get_or_insert_with(|| {
            let c = ctx.clone();
            Worker::new(
                "grating",
                |q: Request| Response {
                    rec: record(&q.params),
                    far: q.params.far_field.then(|| far_field(&q.params, q.far_cols)),
                    params: q.params,
                },
                move || c.request_repaint(),
            )
        });
        worker.request(&Request { params: record_key(&self.params), far_cols: 1600 });
        if let Some(r) = worker.poll() {
            self.rec = Some(r.rec);
            self.far = r.far;
            self.shown = Some(r.params);
        }
        if worker.busy() {
            ctx.request_repaint();
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let win = ui.ctx().content_rect();
        if self.scan {
            // turn the grating slowly: the spectrum moves across the camera
            let dt = ui.input(|i| i.stable_dt).min(0.1) as f64;
            self.params.angle_deg += dt * 1.5;
            if self.params.angle_deg > 40.0 {
                self.params.angle_deg = -40.0;
            }
            ui.ctx().request_repaint();
        }
        egui::Panel::top("g_views")
            .resizable(true)
            .default_size(win.height() * 0.42)
            .min_size(200.0)
            .max_size(win.height() * 0.65)
            .show(ui, |ui| {
                ui.add_space(4.0);
                let full = ui.available_rect_before_wrap();
                let split = full.left() + full.width() * 0.56;
                let left = Rect::from_min_max(full.min, pos2(split - 4.0, full.bottom()));
                let right = Rect::from_min_max(pos2(split + 4.0, full.top()), full.max);
                let builder = |r: Rect| egui::UiBuilder::new().max_rect(r).layout(egui::Layout::top_down(egui::Align::Min));
                self.layout_panel(&mut ui.new_child(builder(left)));
                self.resolution_panel(&mut ui.new_child(builder(right)));
                ui.painter().line_segment([pos2(split, full.top()), pos2(split, full.bottom())], Stroke::new(1.0, Color32::from_gray(50)));
                ui.allocate_rect(full, Sense::hover());
            });
        egui::Panel::bottom("g_controls")
            .resizable(true)
            .default_size(250.0)
            .min_size(120.0)
            .max_size(win.height() * 0.45)
            .show(ui, |ui| self.controls_ui(ui));
        egui::CentralPanel::default().show(ui, |ui| self.spectrum_panel(ui));
    }

    // ------------------------------------------------------------ layout

    fn layout_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.strong("SPECTROMETER");
            ui.weak("seen from above · drag the grating to turn it");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button(if self.scan { "⏸ stop" } else { "▶ turn" }).on_hover_text("turn the grating slowly").clicked() {
                    self.scan = !self.scan;
                }
            });
        });
        let avail = ui.available_size();
        let (rect, resp) = ui.allocate_exact_size(avail, Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3.0, BG);
        let p = &self.params;
        let (f1, f2) = (p.f1_mm, p.f2_mm);
        let a_arms = p.arms_deg * DEG;
        // directions from the grating (y up): collimator arm to the left, camera arm A below it
        let u_c = (-1.0f64, 0.0f64);
        let u_cam = (-(a_arms.cos()), -(a_arms.sin()));
        let normal = PI + 0.5 * a_arms + p.angle_deg * DEG;
        let at = |o: (f64, f64), u: (f64, f64), d: f64| (o.0 + u.0 * d, o.1 + u.1 * d);
        let g = (0.0, 0.0);
        let l1 = at(g, u_c, 0.5 * f1);
        let iris = at(g, u_c, 0.18 * f1);
        let slit = at(l1, u_c, f1);
        let bs = at(slit, u_c, 0.32 * f1);
        let l0 = at(bs, u_c, 0.22 * f1);
        let lamp_s = at(l0, u_c, 0.22 * f1);
        let up = (0.0, 1.0);
        let lr = at(bs, up, 0.22 * f1);
        let lamp_r = at(lr, up, 0.22 * f1);
        let l2 = at(g, u_cam, 0.5 * f2);
        let det = at(l2, u_cam, f2);
        let half_det = 0.5 * p.camera_mm();
        let det_dir = (-u_cam.1, u_cam.0);
        // fit everything into the panel
        let pts = [g, slit, lamp_s, lamp_r, det, at(det, det_dir, half_det), at(det, det_dir, -half_det), (g.0 + 0.15 * f1, 0.0)];
        let (mut x0, mut x1, mut y0, mut y1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for q in pts {
            x0 = x0.min(q.0);
            x1 = x1.max(q.0);
            y0 = y0.min(q.1);
            y1 = y1.max(q.1);
        }
        let margin = 0.08 * (x1 - x0).max(y1 - y0);
        let (x0, x1, y0, y1) = (x0 - margin, x1 + margin, y0 - margin, y1 + margin);
        let k = ((rect.width() as f64) / (x1 - x0)).min((rect.height() as f64) / (y1 - y0));
        let cx = rect.center().x as f64 - 0.5 * (x0 + x1) * k;
        let cy = rect.center().y as f64 + 0.5 * (y0 + y1) * k;
        let s = |q: (f64, f64)| pos2((cx + q.0 * k) as f32, (cy - q.1 * k) as f32);
        let dirv = |a: f64| vec2(a.cos() as f32, -a.sin() as f32);
        let w = p.grooves() * p.d_mm();
        let beam_half = (0.5 * w * k).max(1.0) as f32;
        let thin = Stroke::new(1.0, Color32::from_white_alpha(70));
        // source and reference into the slit
        let lens = |c: (f64, f64), axis: (f64, f64), h: f64| {
            let n = vec2(-axis.1 as f32, -axis.0 as f32);
            let c = s(c);
            let hh = (h * k) as f32;
            painter.line_segment([c - n * hh, c + n * hh], Stroke::new(2.0, LENS));
            painter.circle_filled(c - n * hh, 1.5, LENS);
            painter.circle_filled(c + n * hh, 1.5, LENS);
        };
        let lamp_box = |c: (f64, f64), text: &str, col: Color32, on: bool| {
            let r = Rect::from_center_size(s(c), vec2(34.0, 22.0));
            painter.rect(r, 3.0, if on { Color32::from_gray(70) } else { Color32::from_gray(35) }, Stroke::new(1.0, col), StrokeKind::Inside);
            painter.text(r.center(), Align2::CENTER_CENTER, text, FontId::proportional(10.0), col);
        };
        let src_names: Vec<&str> = p.sources.iter().map(|l| l.short()).collect();
        let ref_names: Vec<&str> = p.references.iter().map(|l| l.short()).collect();
        let cone = 0.06 * f1;
        for (from, via, on) in [(lamp_s, l0, !p.sources.is_empty()), (lamp_r, lr, !p.references.is_empty())] {
            if !on {
                continue;
            }
            let dir = ((via.0 - from.0) / (0.22 * f1), (via.1 - from.1) / (0.22 * f1));
            let n = (-dir.1, dir.0);
            for sgn in [-1.0, 1.0] {
                let e = at(via, n, sgn * cone);
                painter.line_segment([s(from), s(e)], thin);
                if via == lr {
                    // reflected at the beam splitter
                    let on_bs = at(bs, u_c, -sgn * cone * 0.5);
                    painter.line_segment([s(e), s(on_bs)], thin);
                    painter.line_segment([s(on_bs), s(slit)], thin);
                } else {
                    painter.line_segment([s(e), s(slit)], thin);
                }
            }
        }
        lens(l0, u_c, 0.08 * f1);
        lens(lr, up, 0.08 * f1);
        lamp_box(lamp_s, &if src_names.is_empty() { "source".into() } else { src_names.join("+") }, SOURCE, !p.sources.is_empty());
        lamp_box(lamp_r, &if ref_names.is_empty() { "ref.".into() } else { ref_names.join("+") }, REFERENCE, !p.references.is_empty());
        painter.text(s(lamp_s) + vec2(0.0, 14.0), Align2::CENTER_TOP, "source", FontId::proportional(10.0), LABEL);
        painter.text(s(lamp_r) + vec2(20.0, 0.0), Align2::LEFT_CENTER, "reference", FontId::proportional(10.0), LABEL);
        // beam splitter at 45°
        let bsd = vec2(0.7, 0.7) * (0.05 * f1 * k) as f32;
        painter.line_segment([s(bs) - bsd, s(bs) + bsd], Stroke::new(2.0, Color32::from_gray(150)));
        // slit
        let sl = s(slit);
        painter.line_segment([sl - vec2(0.0, 22.0), sl - vec2(0.0, 2.5)], Stroke::new(2.5, SLIT));
        painter.line_segment([sl + vec2(0.0, 2.5), sl + vec2(0.0, 22.0)], Stroke::new(2.5, SLIT));
        painter.text(sl + vec2(0.0, 24.0), Align2::CENTER_TOP, format!("slit {:.0} µm", p.slit_um), FontId::proportional(10.0), SLIT);
        // slit to collimator to grating
        let l1s = s(l1);
        let up_v = vec2(0.0, 1.0);
        let ap = beam_half.max(3.0);
        painter.line_segment([sl, l1s - up_v * ap], Stroke::new(1.0, Color32::from_white_alpha(120)));
        painter.line_segment([sl, l1s + up_v * ap], Stroke::new(1.0, Color32::from_white_alpha(120)));
        let gs = s(g);
        let band = |a: Pos2, b: Pos2, half: f32, col: Color32| {
            painter.add(Shape::convex_polygon(vec![a - up_v * half, b - up_v * half, b + up_v * half, a + up_v * half], col, Stroke::NONE));
        };
        band(l1s, gs, beam_half, Color32::from_white_alpha(40));
        lens(l1, u_c, (0.5 * w).max(0.03 * f1) + 0.01 * f1);
        painter.text(l1s + vec2(0.0, ap + 4.0), Align2::CENTER_TOP, format!("f₁ = {:.0} mm", f1), FontId::proportional(10.0), LENS);
        // iris
        let is = s(iris);
        painter.line_segment([is - up_v * (beam_half + 14.0), is - up_v * beam_half], Stroke::new(2.0, Color32::from_gray(120)));
        painter.line_segment([is + up_v * beam_half, is + up_v * (beam_half + 14.0)], Stroke::new(2.0, Color32::from_gray(120)));
        painter.text(is - up_v * (beam_half + 15.0), Align2::CENTER_BOTTOM, format!("iris W = {}", fmt_mm(w)), FontId::proportional(10.0), LABEL);
        // the diffracted beams: every line of the lamps, every order
        let alpha = p.alpha();
        let lam_lines = p.lines(&p.sources.iter().chain(&p.references).copied().collect::<Vec<_>>());
        let white = p.sources.contains(&Lamp::White) || p.references.contains(&Lamp::White);
        let mut shown_lines: Vec<f64> = lam_lines.iter().map(|l| l.0).collect();
        if white {
            shown_lines.extend((0..9).map(|i| 400.0 + 37.5 * i as f64));
        }
        shown_lines.truncate(60);
        let ray_len = 0.55 * f2;
        let l2_half = 0.12 * f2;
        let mut labelled = std::collections::HashSet::new();
        for &nm in &shown_lines {
            for m in -4..=4 {
                let Some(beta) = p.beta_of(nm, m, alpha) else { continue };
                let dir_a = normal - beta;
                let d = dirv(dir_a);
                let col = if m == 0 { Color32::from_white_alpha(140) } else { lamp_color(nm).gamma_multiply(0.85) };
                // does it go through the camera lens?
                let off = (beta - p.beta_c()).tan() * 0.5 * f2;
                if off.abs() < l2_half {
                    let hit = at(l2, det_dir, -off);
                    painter.line_segment([gs, s(hit)], Stroke::new(1.0, col));
                    if let Some(x) = p.x_of_beta(beta) {
                        let on = at(det, det_dir, -x);
                        painter.line_segment([s(hit), s(on)], Stroke::new(1.0, col));
                    }
                } else {
                    let end = gs + d * (ray_len * k) as f32;
                    painter.line_segment([gs, end], Stroke::new(1.0, col.gamma_multiply(0.6)));
                    if labelled.insert(m) {
                        let t = if m == 0 { "m = 0".to_string() } else { format!("m = {m:+}") };
                        painter.text(end + d * 8.0, Align2::CENTER_CENTER, t, FontId::proportional(10.0), LABEL);
                    }
                }
            }
        }
        // camera lens and camera
        lens(l2, u_cam, l2_half);
        let ds = s(det);
        let dv = vec2(det_dir.0 as f32, -det_dir.1 as f32);
        let hd = (half_det * k) as f32;
        // the camera shows what it records, in colour
        if let Some(rec) = &self.rec {
            let n = rec.source.len();
            let max = rec.source.iter().zip(&rec.reference).map(|(a, b)| a + b).fold(0.0f32, f32::max).max(1e-30);
            let segs = 160;
            for i in 0..segs {
                let (a, b) = (i * n / segs, ((i + 1) * n / segs).max(i * n / segs + 1));
                let mut v = 0.0f32;
                let mut c = [0.0f32; 3];
                for j in a..b.min(n) {
                    let t = rec.source[j] + rec.reference[j];
                    if t > v {
                        v = t;
                        c = rec.rgb[j];
                    }
                }
                let lvl = (v / max).powf(0.5);
                let col = Color32::from_rgb((c[0] * lvl * 255.0) as u8, (c[1] * lvl * 255.0) as u8, (c[2] * lvl * 255.0) as u8);
                // pixel 0 is at x = −L/2, drawn at −(−L/2) along det_dir
                let t0 = 1.0 - 2.0 * i as f32 / segs as f32;
                let t1 = 1.0 - 2.0 * (i + 1) as f32 / segs as f32;
                painter.line_segment([ds + dv * hd * t0, ds + dv * hd * t1], Stroke::new(7.0, col));
            }
        }
        painter.line_segment([ds - dv * (hd + 2.0), ds + dv * (hd + 2.0)], Stroke::new(1.0, CAMERA));
        painter.text(ds + vec2(0.0, 10.0) + dv * hd, Align2::LEFT_TOP, "camera", FontId::proportional(10.0), CAMERA);
        painter.text(s(l2) + vec2(10.0, 0.0), Align2::LEFT_CENTER, format!("f₂ = {:.0} mm", f2), FontId::proportional(10.0), LENS);
        // the grating: a mirror with a sawtooth
        let gl = (0.5 * w.max(0.12 * f1)) * 1.15;
        let along = normal + PI / 2.0;
        let ga = dirv(along) * (gl * k) as f32;
        let nrm = dirv(normal);
        painter.line_segment([gs - ga, gs + ga], Stroke::new(3.0, SILVER));
        let teeth = 9;
        for i in 0..teeth {
            let t = -1.0 + 2.0 * (i as f32 + 0.5) / teeth as f32;
            let q = gs + ga * t;
            painter.line_segment([q, q - nrm * 4.0 + ga.normalized() * 3.0], Stroke::new(1.0, SILVER));
        }
        dashed(&painter, gs, gs + nrm * 40.0, Stroke::new(1.0, Color32::from_gray(110)));
        painter.text(gs - nrm * 10.0 + vec2(8.0, 0.0), Align2::LEFT_CENTER, format!("grating {:.0} /mm\nturned {:+.2}°", p.lines_per_mm, p.angle_deg), FontId::proportional(10.0), SILVER);
        // drag the grating to turn it
        let near_grating = resp.hover_pos().is_some_and(|q| (q - gs).length() < (gl * k) as f32 + 25.0);
        if resp.drag_started() && near_grating
            && let Some(q) = resp.interact_pointer_pos()
        {
            let v = q - gs;
            self.drag = Some(Drag::Grating { start_angle: (-v.y as f64).atan2(v.x as f64), start_psi: self.params.angle_deg });
            self.scan = false;
        }
        if let Some(Drag::Grating { start_angle, start_psi }) = self.drag {
            if let Some(q) = resp.interact_pointer_pos() {
                let v = q - gs;
                let mut da = (-v.y as f64).atan2(v.x as f64) - start_angle;
                while da > PI {
                    da -= 2.0 * PI;
                }
                while da < -PI {
                    da += 2.0 * PI;
                }
                let fine = if ui.input(|i| i.modifiers.shift) { 0.1 } else { 1.0 };
                self.params.angle_deg = (start_psi + da / DEG * fine).clamp(-80.0, 80.0);
            }
            if !resp.dragged() {
                self.drag = None;
            }
        }
        if near_grating || matches!(self.drag, Some(Drag::Grating { .. })) {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
        }
        let m0 = self.params.main_order();
        let centre = if m0 != 0 { format!("centre of the camera: {:.1} nm in order {m0:+}", self.params.lambda_at(0.0, m0)) } else { "the camera sees the mirror reflection (order 0)".into() };
        tag(&painter, rect.left_top() + vec2(6.0, 6.0), Align2::LEFT_TOP, &centre, PICK, 11.0);
    }

    // ------------------------------------------------------------ resolution

    fn resolution_panel(&mut self, ui: &mut egui::Ui) {
        let p = &self.params;
        let m0 = p.main_order();
        let lam = if m0 != 0 { self.view_centre_nm() } else { 550.0 };
        let r = p.resolution(lam);
        ui.horizontal(|ui| {
            ui.strong("RESOLUTION");
            ui.weak(format!("at {lam:.1} nm, order {}", r.m));
        });
        let r = crate::grating::Resolution { theta_i: if r.theta_i.abs() < 5e-7 { 0.0 } else { r.theta_i }, ..r };
        let text = if r.m == 0 {
            "Order 0 (the mirror reflection): all colours leave the same way; no spectrum. Turn the grating.".to_string()
        } else {
            let mm = r.m.abs() as f64;
            let limit = if r.dl_diff >= r.dl_slit && r.dl_diff >= r.dl_pixel {
                "diffraction (the number of grooves)"
            } else if r.dl_slit >= r.dl_pixel {
                "the slit"
            } else {
                "the pixels"
            };
            format!(
                "lit grooves N = W/d = {:.0}    θ_i = {:.1}° ({:.3} rad), θ_m = {:.1}°\n\
                 path difference across the grating Δ = Nd(sin θ_m − sin θ_i) = {} = {:.0} λ\n\
                 resolving power λ/δλ = mN = Δ/λ = {:.0}, so δλ = {}\n\
                 limit for any order and angle: Δ ≤ 2Nd, so λ/δλ ≤ 2W/λ = {:.0}\n\
                 slit: δλ = {}   pixels (2 per line): δλ = {}\n\
                 this setting is limited by {}",
                r.n,
                r.theta_i / DEG,
                r.theta_i,
                r.theta_m / DEG,
                fmt_mm(r.path_mm.abs()),
                r.path_mm.abs() * 1e6 / lam,
                mm * r.n,
                fmt_nm(r.dl_diff),
                r.r_max,
                fmt_nm(r.dl_slit),
                fmt_nm(r.dl_pixel),
                limit
            )
        };
        ui.label(egui::RichText::new(text).size(11.5).color(LABEL));
        ui.add_space(4.0);
        let avail = ui.available_size();
        let slider_h = 24.0;
        let (rect, _) = ui.allocate_exact_size(vec2(avail.x, (avail.y - slider_h).max(60.0)), Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3.0, BG);
        let left = Rect::from_min_max(rect.min, pos2(rect.center().x - 2.0, rect.bottom()));
        let right = Rect::from_min_max(pos2(rect.center().x + 2.0, rect.top()), rect.max);
        self.wavefronts(&painter, left, &r);
        self.phasors(&painter, right, &r);
        ui.horizontal(|ui| {
            ui.label("second wavelength λ + δλ, δλ =");
            ui.add(egui::Slider::new(&mut self.params.phasor_frac, 0.0..=2.5).fixed_decimals(2).suffix(" × λ/mN"));
        });
    }

    /// the wavefront arriving at and leaving the lit grating, and the path difference Δ
    fn wavefronts(&self, painter: &egui::Painter, rect: Rect, r: &crate::grating::Resolution) {
        painter.text(rect.left_top() + vec2(5.0, 4.0), Align2::LEFT_TOP, "path difference across the grating", FontId::proportional(10.0), LABEL);
        if r.m == 0 {
            return;
        }
        let c = rect.center() + vec2(0.0, rect.height() * 0.22);
        let half = rect.width().min(rect.height() * 1.5) * 0.36;
        // grating horizontal, normal up; θ measured from the normal
        let ti = r.theta_i as f32;
        let tm = r.theta_m as f32;
        let a = c - vec2(half, 0.0);
        let b = c + vec2(half, 0.0);
        painter.line_segment([a, b], Stroke::new(2.5, SILVER));
        let teeth = 12;
        for i in 0..=teeth {
            let q = a + (b - a) * (i as f32 / teeth as f32);
            painter.line_segment([q, q + vec2(3.0, 4.0)], Stroke::new(1.0, SILVER));
        }
        // incoming direction of travel: tangential component sin θ_i, going down
        let din = vec2(ti.sin(), ti.cos());
        let dout = vec2(tm.sin(), -tm.cos());
        let len = rect.height() * 0.6;
        let inc = Color32::from_gray(200);
        let out = PICK;
        for q in [a, b] {
            painter.line_segment([q - din * len, q], Stroke::new(1.0, inc));
            painter.line_segment([q, q + dout * len], Stroke::new(1.0, out));
        }
        arrow_head(painter, a - din * len * 0.4, din, 6.0, inc);
        arrow_head(painter, b + dout * len * 0.6, dout, 6.0, out);
        // wavefronts: perpendicular to the rays, through the first groove on the way in
        let perp_in = vec2(din.y, -din.x);
        let perp_out = vec2(dout.y, -dout.x);
        // the incoming wavefront through a meets the ray to b at b − din·(extra)
        let extra_in = (b - a).dot(din);
        let extra_out = -(b - a).dot(dout);
        let wf = |p0: Pos2, p1: Pos2| {
            let d = (p1 - p0).normalized();
            painter.extend(Shape::dashed_line(&[p0 - d * 18.0, p1 + d * 18.0], Stroke::new(1.0, Color32::from_gray(150)), 4.0, 3.0));
        };
        let _ = (perp_in, perp_out);
        // the groove that the incoming wavefront reaches last gets the extra path before it
        let (late_in, early_in, ein) = if extra_in < 0.0 { (a, b, -extra_in) } else { (b, a, extra_in) };
        let start_in = late_in - din * ein;
        painter.line_segment([start_in, late_in], Stroke::new(3.0, SOURCE));
        wf(start_in, early_in);
        // on the way out, the groove behind the outgoing wavefront
        let (late_out, early_out, eout) = if extra_out > 0.0 { (b, a, extra_out) } else { (a, b, -extra_out) };
        let end_out = late_out + dout * eout;
        painter.line_segment([late_out, end_out], Stroke::new(3.0, SOURCE));
        wf(end_out, early_out);
        dashed(painter, c, c - vec2(0.0, len * 0.9), Stroke::new(1.0, Color32::from_gray(90)));
        painter.text(c - vec2(0.0, len * 0.9 + 2.0), Align2::CENTER_BOTTOM, "normal", FontId::proportional(9.5), Color32::from_gray(120));
        painter.text(pos2(c.x, a.y + 10.0), Align2::CENTER_TOP, format!("lit width W = Nd = {}", fmt_mm(r.n * self.params.d_mm())), FontId::proportional(10.0), SILVER);
        painter.text(
            rect.left_bottom() + vec2(5.0, -4.0),
            Align2::LEFT_BOTTOM,
            format!("blue: extra path, Δ = W(sin θ_m − sin θ_i) = mNλ = {:.0} λ", r.path_mm.abs() * 1e6 / r.lambda),
            FontId::proportional(10.0),
            SOURCE,
        );
    }

    /// N phasors, one per groove, for λ (in a line) and for λ + δλ (curling up)
    fn phasors(&self, painter: &egui::Painter, rect: Rect, r: &crate::grating::Resolution) {
        painter.text(rect.left_top() + vec2(5.0, 4.0), Align2::LEFT_TOP, "phasors of the N grooves where λ has its maximum", FontId::proportional(10.0), LABEL);
        if r.m == 0 {
            return;
        }
        let n = r.n;
        let frac = self.params.phasor_frac;
        // extra phase per groove for λ + δλ: 2π m δλ/λ with δλ = frac·λ/(mN): 2π frac/N
        let dphi = 2.0 * PI * frac / n;
        let count = n.min(400.0) as usize;
        let step = n / count as f64;
        let area = rect.shrink2(vec2(14.0, 24.0));
        // λ + δλ: each groove turned a little more
        let mut chain = vec![(0.0, 0.0)];
        let (mut x, mut y) = (0.0, 0.0);
        for k in 0..count {
            let ph = dphi * (k as f64 + 0.5) * step;
            x += ph.cos() * step;
            y += ph.sin() * step;
            chain.push((x, y));
        }
        // fit both chains into the area
        let (mut x0, mut x1, mut y0, mut y1) = (0.0f64, n, 0.0f64, 0.0f64);
        for &(u, v) in &chain {
            x0 = x0.min(u);
            x1 = x1.max(u);
            y0 = y0.min(v);
            y1 = y1.max(v);
        }
        let scale = (area.width() as f64 / (x1 - x0).max(1e-9)).min(area.height() as f64 / (y1 - y0).max(1e-9));
        let ox = area.center().x as f64 - 0.5 * (x0 + x1) * scale;
        let oy = area.center().y as f64 + 0.5 * (y0 + y1) * scale;
        let to = |u: f64, v: f64| pos2((ox + u * scale) as f32, (oy - v * scale) as f32);
        // λ: all in phase
        painter.line_segment([to(0.0, 0.0), to(n, 0.0)], Stroke::new(2.0, Color32::from_gray(200)));
        arrow_head(painter, to(n, 0.0), vec2(1.0, 0.0), 7.0, Color32::from_gray(200));
        let pts: Vec<Pos2> = chain.iter().map(|&(u, v)| to(u, v)).collect();
        let col = SOURCE;
        if count <= 60 {
            for w in pts.windows(2) {
                painter.line_segment([w[0], w[1]], Stroke::new(1.5, col));
                let d = (w[1] - w[0]).normalized();
                arrow_head(painter, w[1], d, 4.0, col);
            }
        } else {
            painter.add(Shape::line(pts.clone(), Stroke::new(1.5, col)));
        }
        let end = *pts.last().expect("phasors");
        painter.line_segment([to(0.0, 0.0), end], Stroke::new(2.5, PICK));
        let amp = (x * x + y * y).sqrt() / n;
        painter.text(
            rect.left_bottom() + vec2(5.0, -4.0),
            Align2::LEFT_BOTTOM,
            format!("grey: λ, sum N   ·   blue: λ + δλ, sum {:.2} N, intensity {:.2}", amp, amp * amp),
            FontId::proportional(10.0),
            LABEL,
        );
        if (frac - 1.0).abs() < 0.02 {
            painter.text(rect.right_top() + vec2(-5.0, 18.0), Align2::RIGHT_TOP, "δλ = λ/mN: the chain closes, zero (Rayleigh)", FontId::proportional(10.0), PICK);
        }
    }

    /// the wavelength in the middle of the shown part of the camera
    fn view_centre_nm(&self) -> f64 {
        let p = &self.params;
        let m0 = p.main_order();
        if m0 == 0 {
            return 550.0;
        }
        let x = (0.5 * (p.view.0 + p.view.1) - 0.5) * p.camera_mm();
        p.lambda_at(x, m0)
    }

    // ------------------------------------------------------------ spectrum

    fn spectrum_panel(&mut self, ui: &mut egui::Ui) {
        let cal = match (&self.rec, &self.shown) {
            (Some(rec), Some(sp)) => {
                let mut q = sp.clone();
                q.fit_degree = self.params.fit_degree;
                calibrate(&q, rec)
            }
            _ => None,
        };
        let cal_ok = cal.as_ref().is_some_and(|c| !c.coeffs.is_empty());
        ui.horizontal(|ui| {
            ui.strong("SPECTRUM");
            ui.selectable_value(&mut self.params.far_field, false, "on the camera");
            ui.selectable_value(&mut self.params.far_field, true, "in all directions").on_hover_text("the light leaving the grating, against the path difference between neighbouring grooves");
            ui.separator();
            if !self.params.far_field {
                ui.label("axis");
                egui::ComboBox::from_id_salt("g_axis")
                    .selected_text(match self.params.axis {
                        Axis::Pixels => "pixels",
                        Axis::Calibrated => "λ from the calibration",
                        Axis::Exact => "λ from the grating equation",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.params.axis, Axis::Pixels, "pixels");
                        ui.selectable_value(&mut self.params.axis, Axis::Calibrated, "λ from the calibration")
                            .on_hover_text("a polynomial λ(pixel) through the reference lines");
                        ui.selectable_value(&mut self.params.axis, Axis::Exact, "λ from the grating equation")
                            .on_hover_text("as if the angles and focal lengths were known exactly");
                    });
            }
            if self.params.far_field {
                ui.label("against");
                ui.selectable_value(&mut self.params.far_axis, FarAxis::Path, "path difference").on_hover_text("d(sin θ_m − sin θ_i): the orders at mλ");
                ui.selectable_value(&mut self.params.far_axis, FarAxis::Angle, "angle θ_m");
                ui.checkbox(&mut self.params.compare_few, "compare N = 1, 2").on_hover_text("the same grating with only one and two grooves lit");
                ui.separator();
            }
            ui.checkbox(&mut self.params.log, "log");
            if ui.small_button("whole range").on_hover_text("or double-click the plot").clicked() {
                self.reset_zoom();
            }
            ui.weak("scroll to zoom, drag to move");
            if let Some(r) = &self.rec {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.weak(format!("{:.0} ms", r.millis));
                });
            }
        });
        let avail = ui.available_size();
        let list_w = if self.params.far_field { 0.0 } else { (avail.x * 0.24).clamp(180.0, 300.0) };
        let (rect, resp) = ui.allocate_exact_size(vec2(avail.x - list_w, avail.y), Sense::click_and_drag());
        if self.params.far_field {
            self.far_plot(ui, rect, &resp);
        } else {
            self.camera_plot(ui, rect, &resp, cal.as_ref().filter(|_| cal_ok));
            let list = Rect::from_min_max(pos2(rect.right() + 6.0, rect.top()), pos2(rect.right() + list_w, rect.bottom()));
            self.lines_list(ui, list, cal.as_ref());
        }
    }

    fn reset_zoom(&mut self) {
        let p = &mut self.params;
        if !p.far_field {
            p.view = (0.0, 1.0);
        } else if p.far_axis == FarAxis::Path {
            let d = p.d_mm() * 1e6;
            p.far_view = (-2.0 * d, 2.0 * d);
        } else {
            p.far_view_rad = (-1.5, 1.5);
        }
    }

    /// the shown range of the active plot
    fn current_range(&self) -> (f64, f64) {
        let p = &self.params;
        match (p.far_field, p.far_axis) {
            (false, _) => p.view,
            (true, FarAxis::Path) => p.far_view,
            (true, FarAxis::Angle) => p.far_view_rad,
        }
    }

    /// scroll to zoom and drag to pan a range (a, b) shown across `rect`
    fn zoom_pan(&mut self, ui: &egui::Ui, rect: Rect, resp: &egui::Response, range: (f64, f64), limits: (f64, f64), min_span: f64) -> (f64, f64) {
        let (mut a, mut b) = range;
        if resp.double_clicked() {
            self.reset_zoom();
            return self.current_range();
        }
        if resp.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0
                && let Some(q) = resp.hover_pos()
            {
                let t = ((q.x - rect.left()) / rect.width()) as f64;
                let at = a + t * (b - a);
                let f = (-scroll as f64 * 0.003).exp();
                a = at - (at - a) * f;
                b = at + (b - at) * f;
            }
        }
        if resp.drag_started() {
            self.drag = Some(Drag::Pan { last: resp.interact_pointer_pos().map_or(0.0, |q| q.x) });
        }
        if let Some(Drag::Pan { last }) = &mut self.drag {
            if let Some(q) = resp.interact_pointer_pos() {
                let dx = ((q.x - *last) / rect.width()) as f64 * (b - a);
                a -= dx;
                b -= dx;
                *last = q.x;
            }
            if !resp.dragged() {
                self.drag = None;
            }
        }
        if b - a < min_span {
            let c = 0.5 * (a + b);
            a = c - 0.5 * min_span;
            b = c + 0.5 * min_span;
        }
        let span = (b - a).min(limits.1 - limits.0);
        if a < limits.0 {
            a = limits.0;
            b = a + span;
        }
        if b > limits.1 {
            b = limits.1;
            a = b - span;
        }
        (a, b)
    }

    fn camera_plot(&mut self, ui: &mut egui::Ui, rect: Rect, resp: &egui::Response, cal: Option<&Calibration>) {
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3.0, BG);
        let (Some(rec), Some(sp)) = (self.rec.clone(), self.shown.clone()) else {
            painter.text(rect.center(), Align2::CENTER_CENTER, "computing…", FontId::proportional(13.0), LABEL);
            return;
        };
        let npx = rec.source.len() as f64;
        let view = self.zoom_pan(ui, rect, resp, self.params.view, (0.0, 1.0), 8.0 / npx.max(8.0));
        self.params.view = view;
        let (pa, pb) = (view.0 * npx, view.1 * npx);
        let strip_h = 26.0;
        let plot = Rect::from_min_max(rect.min + vec2(46.0, strip_h + 10.0), rect.max - vec2(10.0, 34.0));
        let sx = |px: f64| plot.left() + ((px - pa) / (pb - pa)) as f32 * plot.width();
        let m0 = sp.main_order();
        // the axis: pixels, or wavelength from the fit or from the grating equation
        let axis = match self.params.axis {
            Axis::Calibrated if cal.is_none() => Axis::Exact,
            a => a,
        };
        let axis = if m0 == 0 && axis != Axis::Pixels { Axis::Pixels } else { axis };
        let lam_of = |px: f64| -> f64 {
            match axis {
                Axis::Calibrated => cal.map_or(0.0, |c| c.lambda(px)),
                _ => sp.lambda_at(sp.x_of_pixel(px), m0),
            }
        };
        let px_of = |nm: f64| -> f64 {
            match axis {
                Axis::Calibrated => cal.map_or(0.0, |c| c.pixel(nm)),
                _ => sp.x_of(nm, m0).map_or(f64::NAN, |x| sp.pixel_of_x(x)),
            }
        };
        let font = FontId::proportional(10.0);
        let grid = Stroke::new(1.0, Color32::from_gray(38));
        let xlabel = match axis {
            Axis::Pixels => "camera pixel".to_string(),
            Axis::Calibrated => format!("wavelength (nm), from the calibration (order {m0:+})"),
            Axis::Exact => format!("wavelength (nm), from the grating equation (order {m0:+})"),
        };
        if axis == Axis::Pixels {
            let t = ticks(pa, pb, (plot.width() / 80.0).max(2.0) as f64);
            for &v in &t {
                let x = sx(v);
                painter.line_segment([pos2(x, plot.top()), pos2(x, plot.bottom())], grid);
                painter.text(pos2(x, plot.bottom() + 3.0), Align2::CENTER_TOP, fmt_step(v, tick_step(&t)), font.clone(), LABEL);
            }
        } else {
            let (la, lb) = (lam_of(pa), lam_of(pb));
            let t = ticks(la.min(lb), la.max(lb), (plot.width() / 80.0).max(2.0) as f64);
            for &v in &t {
                let x = sx(px_of(v));
                if x.is_finite() && (plot.left()..=plot.right()).contains(&x) {
                    painter.line_segment([pos2(x, plot.top()), pos2(x, plot.bottom())], grid);
                    painter.text(pos2(x, plot.bottom() + 3.0), Align2::CENTER_TOP, fmt_step(v, tick_step(&t)), font.clone(), LABEL);
                }
            }
        }
        painter.text(pos2(plot.center().x, rect.bottom() - 3.0), Align2::CENTER_BOTTOM, xlabel, font.clone(), LABEL);
        painter.rect_stroke(plot, 0.0, Stroke::new(1.0, Color32::from_gray(70)), StrokeKind::Inside);
        // the camera image as a colour strip
        let (i0, i1) = (pa.floor().max(0.0) as usize, (pb.ceil() as usize).min(rec.source.len()));
        let total: Vec<f32> = rec.source.iter().zip(&rec.reference).map(|(a, b)| a + b).collect();
        let vmax = total[i0..i1].iter().cloned().fold(0.0f32, f32::max).max(1e-30);
        let strip = Rect::from_min_max(pos2(plot.left(), rect.top() + 6.0), pos2(plot.right(), rect.top() + 6.0 + strip_h - 4.0));
        painter.rect_filled(strip, 0.0, Color32::BLACK);
        let cols = (strip.width() as usize).max(1);
        for c in 0..cols {
            let (qa, qb) = (pa + (pb - pa) * c as f64 / cols as f64, pa + (pb - pa) * (c + 1) as f64 / cols as f64);
            let (ja, jb) = (qa.floor().max(0.0) as usize, (qb.ceil() as usize).min(total.len()).max(qa.floor().max(0.0) as usize + 1));
            let mut v = 0.0f32;
            let mut col = [0.0f32; 3];
            for (t, c) in total.iter().zip(&rec.rgb).take(jb).skip(ja) {
                if *t > v {
                    v = *t;
                    col = *c;
                }
            }
            let lvl = (v / vmax).powf(0.45);
            let x = strip.left() + c as f32;
            painter.line_segment(
                [pos2(x, strip.top()), pos2(x, strip.bottom())],
                Stroke::new(1.0, Color32::from_rgb((col[0] * lvl * 255.0) as u8, (col[1] * lvl * 255.0) as u8, (col[2] * lvl * 255.0) as u8)),
            );
        }
        painter.text(strip.right_top() + vec2(-3.0, 1.0), Align2::RIGHT_TOP, "what the camera sees", FontId::proportional(9.5), Color32::from_gray(150));
        // the two recordings, scaled to the highest in view
        let log = self.params.log;
        let lmin = (vmax as f64 * 1e-4).log10();
        let sy = |v: f64| {
            let t = if log { ((v.max(1e-300).log10() - lmin) / ((vmax as f64).log10() - lmin)).clamp(0.0, 1.0) } else { v / (vmax as f64 * 1.08) };
            plot.bottom() - t as f32 * plot.height()
        };
        for (data, col) in [(&rec.reference, REFERENCE), (&rec.source, SOURCE)] {
            let xs: Vec<f64> = (i0..i1).map(|k| k as f64 + 0.5).collect();
            let ys: Vec<f64> = data[i0..i1].iter().map(|&v| v as f64).collect();
            if ys.iter().all(|&v| v <= 0.0) {
                continue;
            }
            // pixels as steps when they are wide on screen
            let pts = if plot.width() / (i1 - i0).max(1) as f32 > 4.0 {
                let mut v = vec![];
                for (k, &y) in ys.iter().enumerate() {
                    let x = i0 as f64 + k as f64;
                    v.push(pos2(sx(x), sy(y)));
                    v.push(pos2(sx(x + 1.0), sy(y)));
                }
                v
            } else {
                polyline(&xs, &ys, |x, y| pos2(sx(x), sy(y)), plot.width())
            };
            ui.painter_at(plot.expand(1.0)).add(Shape::line(pts, Stroke::new(1.3, col)));
        }
        // labels: reference lines with their known wavelength, source lines as measured
        let pp = ui.painter_at(plot);
        let mut used: Vec<Rect> = vec![];
        let mut put = |x: f32, y: f32, text: String, col: Color32| {
            let g = pp.layout_no_wrap(text, FontId::proportional(9.5), col);
            let mut r = Align2::CENTER_BOTTOM.anchor_size(pos2(x, y - 3.0), g.size()).expand(1.5);
            for _ in 0..8 {
                if used.iter().any(|u| u.intersects(r)) {
                    r = r.translate(vec2(0.0, -r.height() - 1.0));
                }
            }
            if r.top() < plot.top() {
                r = r.translate(vec2(0.0, plot.top() - r.top()));
            }
            used.push(r);
            pp.rect_filled(r, 2.0, Color32::from_black_alpha(160));
            pp.galley(r.min + vec2(1.5, 1.5), g, col);
        };
        if let Some(c) = cal {
            for mt in &c.matches {
                if mt.px < pa || mt.px > pb {
                    continue;
                }
                let k = (mt.px as usize).min(rec.reference.len() - 1);
                let txt = if mt.m == c.m { format!("{} {:.2}", mt.lamp.short(), mt.lambda) } else { format!("{} {:.1} (order {})", mt.lamp.short(), mt.lambda, mt.m) };
                put(sx(mt.px), sy(rec.reference[k] as f64), txt, REFERENCE);
            }
        }
        let peaks = drop_side_lobes(find_peaks(&rec.source[i0..i1], 0.02), sp.line_width_px());
        let mut best: Vec<_> = peaks.iter().map(|q| (q.px + i0 as f64, q.height)).collect();
        best.sort_by(|a, b| b.1.total_cmp(&a.1));
        for &(px, h) in best.iter().take(24) {
            let txt = if axis == Axis::Pixels { format!("{px:.1}") } else { format!("{:.2}", lam_of(px)) };
            put(sx(px), sy(h), txt, SOURCE);
        }
        // hover: the value under the pointer
        if let Some(q) = resp.hover_pos()
            && plot.contains(q)
            && self.drag.is_none()
        {
            let px = pa + ((q.x - plot.left()) / plot.width()) as f64 * (pb - pa);
            pp.line_segment([pos2(q.x, plot.top()), pos2(q.x, plot.bottom())], Stroke::new(1.0, Color32::from_gray(110)));
            let mut t = format!("pixel {px:.1}");
            if m0 != 0 {
                t += &format!("\nλ = {:.3} nm (grating equation)", sp.lambda_at(sp.x_of_pixel(px), m0));
                if let Some(c) = cal {
                    t += &format!("\nλ = {:.3} nm (calibration)", c.lambda(px));
                }
            }
            readout(&pp, q + vec2(12.0, 12.0), Align2::LEFT_TOP, t);
        }
        let legend = [("source", SOURCE), ("reference", REFERENCE)];
        let mut y = plot.top() + 4.0;
        for (n, c) in legend {
            let r = tag(&pp, pos2(plot.right() - 4.0, y), Align2::RIGHT_TOP, n, c, 10.0);
            y = r.bottom() + 2.0;
        }
    }

    fn lines_list(&self, ui: &mut egui::Ui, rect: Rect, cal: Option<&Calibration>) {
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(rect));
        let ui = &mut child;
        egui::ScrollArea::vertical().id_salt("g_lines").auto_shrink([false, false]).show(ui, |ui| {
            ui.strong("CALIBRATION");
            match cal {
                None if self.params.references.is_empty() => {
                    ui.weak("No reference lamp: switch one on to calibrate the camera.");
                }
                None => {
                    ui.weak("The camera sees no order to calibrate.");
                }
                Some(c) if c.coeffs.is_empty() => {
                    ui.weak(format!("{} reference line(s) found in order {:+}; at least 2 are needed.", c.matches.iter().filter(|m| m.m == c.m).count(), c.m));
                }
                Some(c) => {
                    ui.weak(format!(
                        "λ(pixel): degree {} through {} lines, rms {}",
                        c.coeffs.len() - 1,
                        c.matches.iter().filter(|m| m.residual.is_some()).count(),
                        fmt_nm(c.rms_nm)
                    ));
                }
            }
            if let Some(c) = cal {
                egui::Grid::new("g_cal").num_columns(3).spacing([8.0, 1.0]).striped(true).show(ui, |ui| {
                    let mut ms = c.matches.clone();
                    ms.sort_by(|a, b| a.px.total_cmp(&b.px));
                    for m in ms {
                        ui.label(egui::RichText::new(format!("{} {:.3}", m.lamp.short(), m.lambda)).size(11.0).color(REFERENCE));
                        if m.m != c.m {
                            ui.label(egui::RichText::new(format!("order {}", m.m)).size(11.0).color(LABEL));
                            ui.label(egui::RichText::new("not used").size(11.0).color(LABEL));
                        } else {
                            ui.label(egui::RichText::new(format!("px {:.1}", m.px)).size(11.0).color(LABEL));
                            ui.label(egui::RichText::new(m.residual.map_or("–".into(), |r| format!("{r:+.3}"))).size(11.0).color(LABEL));
                        }
                        ui.end_row();
                    }
                });
            }
            ui.add_space(6.0);
            ui.strong("SOURCE LINES");
            let (Some(rec), Some(sp)) = (&self.rec, &self.shown) else { return };
            let m0 = sp.main_order();
            let peaks = drop_side_lobes(find_peaks(&rec.source, 0.02), sp.line_width_px());
            if peaks.is_empty() {
                ui.weak("none on the camera");
            }
            let cal_ok = cal.filter(|c| !c.coeffs.is_empty());
            egui::Grid::new("g_src").num_columns(2).spacing([8.0, 1.0]).striped(true).show(ui, |ui| {
                for q in peaks.iter().take(40) {
                    let measured = match cal_ok {
                        Some(c) => format!("{:.3} nm", c.lambda(q.px)),
                        None if m0 != 0 => format!("{:.3} nm*", sp.lambda_at(sp.x_of_pixel(q.px), m0)),
                        None => format!("px {:.1}", q.px),
                    };
                    ui.label(egui::RichText::new(measured).size(11.0).color(SOURCE));
                    ui.label(egui::RichText::new(format!("px {:.1}", q.px)).size(11.0).color(LABEL));
                    ui.end_row();
                }
            });
            if cal_ok.is_none() && m0 != 0 {
                ui.weak("* from the grating equation, without calibration");
            }
        });
    }

    fn far_plot(&mut self, ui: &mut egui::Ui, rect: Rect, resp: &egui::Response) {
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3.0, BG);
        let d = self.params.d_mm() * 1e6;
        let angle = self.params.far_axis == FarAxis::Angle;
        if angle {
            self.params.far_view_rad = self.zoom_pan(ui, rect, resp, self.params.far_view_rad, (-1.55, 1.55), 1e-4);
        } else {
            self.params.far_view = self.zoom_pan(ui, rect, resp, self.params.far_view, (-2.0 * d, 2.0 * d), 5.0);
        }
        let fv = self.current_range();
        let Some(f) = self.far.clone() else {
            painter.text(rect.center(), Align2::CENTER_CENTER, "computing…", FontId::proportional(13.0), LABEL);
            return;
        };
        let plot = Rect::from_min_max(rect.min + vec2(46.0, 10.0), rect.max - vec2(10.0, 34.0));
        let (a, b) = (f.x.first().copied().unwrap_or(fv.0), f.x.last().copied().unwrap_or(fv.1));
        let span = (b - a).max(1e-9);
        let sx = |x: f64| plot.left() + ((x - a) / span) as f32 * plot.width();
        let vmax = f.inten.iter().chain(f.few.iter().flat_map(|(_, v)| v.iter())).cloned().fold(0.0f32, f32::max).max(1e-30) as f64;
        let log = self.params.log;
        let sy = |v: f64| {
            let t = if log { ((v.max(1e-300).log10() + 4.0 - vmax.log10()) / 4.0).clamp(0.0, 1.0) } else { v / (vmax * 1.08) };
            plot.bottom() - t as f32 * plot.height()
        };
        let font = FontId::proportional(10.0);
        let grid = Stroke::new(1.0, Color32::from_gray(38));
        let t = ticks(a, b, (plot.width() / 90.0).max(2.0) as f64);
        for &v in &t {
            let x = sx(v);
            painter.line_segment([pos2(x, plot.top()), pos2(x, plot.bottom())], grid);
            painter.text(pos2(x, plot.bottom() + 3.0), Align2::CENTER_TOP, fmt_step(v, tick_step(&t)), font.clone(), LABEL);
        }
        let p = &self.params;
        let xlabel = if angle {
            let ti = p.theta_i() + 0.0;
            let ti = if ti.abs() < 5e-7 { 0.0 } else { ti };
            format!("angle θ_m from the grating normal (rad); the light arrives at θ_i = {ti:.3} rad ({:.2}°)", ti / DEG)
        } else {
            "path difference between neighbouring grooves d(sin θ_m − sin θ_i) (nm): order m of λ at mλ".to_string()
        };
        painter.text(pos2(plot.center().x, rect.bottom() - 3.0), Align2::CENTER_BOTTOM, xlabel, font.clone(), LABEL);
        // the part the camera sees
        let half = 0.5 * p.camera_mm();
        let cam = |x: f64| if angle { p.beta_at(x) } else { d * (p.alpha().sin() + p.beta_at(x).sin()) };
        let (c0, c1) = (cam(-half), cam(half));
        let cr = Rect::from_min_max(pos2(sx(c0.min(c1)), plot.top()), pos2(sx(c0.max(c1)), plot.bottom())).intersect(plot);
        if cr.width() > 0.0 {
            painter.rect_filled(cr, 0.0, Color32::from_rgba_premultiplied(14, 30, 14, 25));
            painter.text(cr.center_bottom() - vec2(0.0, 3.0), Align2::CENTER_BOTTOM, "seen by the camera", font.clone(), CAMERA);
        }
        // coloured columns
        let pp = ui.painter_at(plot);
        for (k, &v) in f.inten.iter().enumerate() {
            if v <= 0.0 {
                continue;
            }
            let x = sx(f.x[k]);
            let c = f.rgb[k];
            let mx = c[0].max(c[1]).max(c[2]).max(1e-6);
            let col = Color32::from_rgb((c[0] / mx * 230.0) as u8, (c[1] / mx * 230.0) as u8, (c[2] / mx * 230.0) as u8);
            pp.line_segment([pos2(x, plot.bottom()), pos2(x, sy(v as f64))], Stroke::new((plot.width() / f.inten.len() as f32).max(1.0), col.gamma_multiply(0.75)));
        }
        let ys: Vec<f64> = f.inten.iter().map(|&v| v as f64).collect();
        pp.add(Shape::line(polyline(&f.x, &ys, |x, y| pos2(sx(x), sy(y)), plot.width()), Stroke::new(1.2, Color32::from_gray(225))));
        // the same grating with one and two grooves
        let mut legend: Vec<(String, Color32)> = vec![];
        for (n, v) in &f.few {
            let col = if *n == 1.0 { Color32::from_gray(150) } else { Color32::from_rgb(190, 150, 255) };
            let ys: Vec<f64> = v.iter().map(|&v| v as f64).collect();
            pp.add(Shape::line(polyline(&f.x, &ys, |x, y| pos2(sx(x), sy(y)), plot.width()), Stroke::new(1.5, col)));
            legend.push((if *n == 1.0 { "N = 1 (one groove or slit)".into() } else { format!("N = {n:.0}") }, col));
        }
        if !legend.is_empty() {
            legend.push((format!("N = {:.0}", p.grooves()), Color32::from_gray(225)));
        }
        let mut ly = plot.top() + 40.0;
        for (t, c) in legend {
            let r = tag(&pp, pos2(plot.right() - 4.0, ly), Align2::RIGHT_TOP, &t, c, 10.0);
            ly = r.bottom() + 2.0;
        }
        // orders of the first line
        let lamps: Vec<Lamp> = p.sources.iter().chain(&p.references).copied().collect();
        if let Some(&(nm, _, _)) = p.lines(&lamps).first() {
            let (pa, pb) = p.far_range().1;
            let m0 = (pa / nm).ceil() as i64;
            let m1 = (pb / nm).floor() as i64;
            if m1 - m0 < 30 {
                for m in m0..=m1 {
                    let pos = if angle {
                        let sb = m as f64 * nm / d - p.alpha().sin();
                        if sb.abs() > 1.0 {
                            continue;
                        }
                        sb.asin()
                    } else {
                        m as f64 * nm
                    };
                    painter.text(pos2(sx(pos), plot.top() + 22.0), Align2::CENTER_TOP, format!("m = {m}"), font.clone(), LABEL);
                }
            }
        }
        let n = p.grooves();
        tag(&pp, plot.left_top() + vec2(4.0, 4.0), Align2::LEFT_TOP, &format!("N = {n:.0} grooves: principal maxima N² high, {:.0} weak maxima between them", (n - 2.0).max(0.0)), LABEL, 10.0);
        painter.rect_stroke(plot, 0.0, Stroke::new(1.0, Color32::from_gray(70)), StrokeKind::Inside);
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
        egui::ScrollArea::vertical().id_salt("grating_controls").auto_shrink([false, false]).show(ui, |ui| {
            ui.columns(4, |cols| {
                self.lamps_ui(&mut cols[0]);
                self.grating_ui(&mut cols[1]);
                self.optics_ui(&mut cols[2]);
                self.view_ui(&mut cols[3]);
            });
        });
    }

    fn lamps_ui(&mut self, ui: &mut egui::Ui) {
        let p = &mut self.params;
        ui.strong("LAMPS");
        egui::Grid::new("g_lamps").num_columns(3).spacing([6.0, 2.0]).show(ui, |ui| {
            ui.weak("");
            ui.colored_label(SOURCE, "source");
            ui.colored_label(REFERENCE, "reference");
            ui.end_row();
            for lamp in Lamp::ALL {
                ui.label(lamp.label());
                for list in [&mut p.sources, &mut p.references] {
                    let mut on = list.contains(&lamp);
                    if ui.checkbox(&mut on, "").changed() {
                        if on {
                            list.push(lamp);
                        } else {
                            list.retain(|&l| l != lamp);
                        }
                    }
                }
                ui.end_row();
            }
        });
        if p.sources.contains(&Lamp::Doublet) || p.references.contains(&Lamp::Doublet) {
            ui.horizontal(|ui| {
                ui.label("test lines at");
                ui.add(egui::DragValue::new(&mut p.doublet_nm).range(380.0..=1000.0).speed(0.5).suffix(" nm"));
                ui.label("apart");
                ui.add(egui::DragValue::new(&mut p.doublet_sep_nm).range(0.001..=50.0).speed(0.01).max_decimals(3).suffix(" nm"));
            });
        }
    }

    fn grating_ui(&mut self, ui: &mut egui::Ui) {
        let p = &mut self.params;
        ui.strong("GRATING");
        egui::Grid::new("g_grating").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
            ui.label("lines per mm");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut p.lines_per_mm).range(1.0..=3600.0).speed(1.0));
                for v in [100.0, 300.0, 600.0, 1200.0, 1800.0] {
                    if ui.small_button(format!("{v:.0}")).clicked() {
                        p.lines_per_mm = v;
                    }
                }
            });
            ui.end_row();
            ui.label("");
            ui.weak(format!("d = {:.3} µm", p.d_mm() * 1e3));
            ui.end_row();
            ui.label("lit width W");
            ui.add(egui::Slider::new(&mut p.width_mm, 0.005..=50.0).logarithmic(true).suffix(" mm").custom_formatter(|v, _| if v < 1.0 { format!("{v:.3}") } else { format!("{v:.1}") }))
                .on_hover_text("the iris in front of the grating");
            ui.end_row();
            ui.label("");
            ui.weak(format!("N = W/d = {:.0} grooves", p.grooves()));
            ui.end_row();
            ui.label("grooves");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut p.groove, Groove::Blazed, "blazed").on_hover_text("sawtooth facets tilted by the blaze angle");
                ui.selectable_value(&mut p.groove, Groove::Strips, "strips").on_hover_text("flat reflecting strips, like the slits of a multi-slit");
            });
            ui.end_row();
            match p.groove {
                Groove::Blazed => {
                    ui.label("blaze angle");
                    ui.add(egui::Slider::new(&mut p.blaze_deg, 0.0..=45.0).suffix("°").fixed_decimals(1));
                    ui.end_row();
                    let lb = 2.0 * p.d_mm() * (p.blaze_deg * DEG).sin() * 1e6;
                    ui.label("");
                    ui.weak(format!("most light at {lb:.0} nm in order 1 (Littrow)"));
                    ui.end_row();
                }
                Groove::Strips => {
                    ui.label("strip width");
                    ui.add(egui::Slider::new(&mut p.fill, 0.05..=1.0).fixed_decimals(2).suffix(" d"));
                    ui.end_row();
                }
            }
            ui.label("turned by");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut p.angle_deg).range(-80.0..=80.0).speed(0.05).max_decimals(3).suffix("°"));
            });
            ui.end_row();
            ui.label("light arrives at θ_i");
            ui.horizontal(|ui| {
                let mut ti = p.theta_i();
                if ti.abs() < 5e-9 {
                    ti = 0.0;
                }
                if ui
                    .add(egui::DragValue::new(&mut ti).range(-1.5..=1.5).speed(0.001).max_decimals(4).suffix(" rad"))
                    .on_hover_text("angle of incidence from the grating normal (turns the grating)")
                    .changed()
                {
                    p.set_theta_i(ti);
                }
                ui.weak(format!("= {:.2}°", ti / DEG));
                if ui.small_button("0").on_hover_text("normal incidence").clicked() {
                    p.set_theta_i(0.0);
                }
            });
            ui.end_row();
        });
        let mut target = self.view_centre_nm().round();
        let mut order = self.params.main_order().max(1);
        ui.horizontal(|ui| {
            ui.label("centre");
            let a = ui.add(egui::DragValue::new(&mut target).range(200.0..=1500.0).speed(1.0).suffix(" nm"));
            ui.label("order");
            let b = ui.add(egui::DragValue::new(&mut order).range(1..=10));
            // an unreachable wavelength leaves the grating as it is
            if (a.changed() || b.changed()) && self.params.set_center(target, order) {
                self.params.view = (0.0, 1.0);
            }
        });
    }

    fn optics_ui(&mut self, ui: &mut egui::Ui) {
        let p = &mut self.params;
        ui.strong("SPECTROMETER");
        egui::Grid::new("g_optics").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
            ui.label("slit width");
            ui.add(egui::Slider::new(&mut p.slit_um, 1.0..=500.0).logarithmic(true).suffix(" µm").fixed_decimals(0));
            ui.end_row();
            ui.label("focal lengths");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut p.f1_mm).range(20.0..=2000.0).speed(1.0).prefix("f₁ ").suffix(" mm"));
                ui.add(egui::DragValue::new(&mut p.f2_mm).range(20.0..=2000.0).speed(1.0).prefix("f₂ ").suffix(" mm"));
            });
            ui.end_row();
            ui.label("angle between arms");
            ui.add(egui::Slider::new(&mut p.arms_deg, 0.0..=90.0).suffix("°").fixed_decimals(1))
                .on_hover_text("between the collimator and the camera; 0 = Littrow (light goes back the way it came)");
            ui.end_row();
            ui.label("camera");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut p.pixels).range(64..=8192).speed(8.0).suffix(" pixels"));
                ui.add(egui::DragValue::new(&mut p.pixel_um).range(1.0..=50.0).speed(0.1).suffix(" µm"));
            });
            ui.end_row();
        });
    }

    fn view_ui(&mut self, ui: &mut egui::Ui) {
        let p = &mut self.params;
        ui.strong("CALIBRATION AND VIEW");
        egui::Grid::new("g_view").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
            ui.label("fit λ(pixel)");
            ui.horizontal(|ui| {
                for (d, t) in [(1, "straight line"), (2, "degree 2"), (3, "degree 3")] {
                    ui.selectable_value(&mut p.fit_degree, d, t);
                }
            });
            ui.end_row();
        });
        ui.weak(
            "The reference lines calibrate the camera: their known wavelengths and where they land give λ(pixel). \
             Lines in other orders are listed but not used.",
        );
    }
}
