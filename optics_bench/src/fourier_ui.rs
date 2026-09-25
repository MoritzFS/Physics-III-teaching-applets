//! User interface of the 4f Fourier-optics bench.

use std::f32::consts::PI;

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, TextureHandle, TextureOptions};
use serde::{Deserialize, Serialize};

use crate::fourier::{
    wavelength_rgb, Beam, Engine, Filter, FourierParams, FourierPreset, FourierResult, InputView, Object,
};

/// which bench is shown
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mode {
    #[default]
    Ray,
    Fourier,
}

/// everything of a result except the images (those live in textures)
struct Meta {
    params: FourierParams,
    side_intensity: Vec<f32>,
    side_nz: usize,
    side_m: usize,
    side_half_mm: f32,
    row_phase: Vec<f32>,
    row_amp: Vec<f32>,
    fourier_row: Vec<f32>,
    image_row: Vec<f32>,
    passed: f32,
    millis: f32,
    side_aliased: bool,
}

pub struct FourierUi {
    pub params: FourierParams,
    pub notes: String,
    engine: Option<Engine>,
    tex: [Option<TextureHandle>; 4],
    meta: Option<Meta>,
    /// observation plane in the side view, as a fraction of the total length
    z_marker: f32,
}

impl Default for FourierUi {
    fn default() -> Self {
        let (params, notes) = FourierPreset::DoubleSlit.setup();
        FourierUi { params, notes, engine: None, tex: [None, None, None, None], meta: None, z_marker: 0.5 }
    }
}

const PLOT_H: f32 = 96.0;
const T_INPUT: usize = 0;
const T_FOURIER: usize = 1;
const T_IMAGE: usize = 2;
const T_SIDE: usize = 3;

fn nice_step(range: f32, target_ticks: f32) -> f32 {
    let raw = range / target_ticks;
    let mag = 10f32.powf(raw.log10().floor());
    for m in [1.0, 2.0, 5.0, 10.0] {
        if raw <= m * mag {
            return m * mag;
        }
    }
    10.0 * mag
}

fn fmt_mm(v: f32) -> String {
    if v.abs() < 1.0 { format!("{:.0} µm", v * 1000.0) } else { format!("{v:.1} mm") }
}

impl FourierUi {
    pub fn load_preset(&mut self, p: FourierPreset) {
        let (params, notes) = p.setup();
        self.params = params;
        self.notes = notes;
        self.z_marker = 0.5;
    }

    /// talk to the worker thread; call once per frame while the 4f bench is shown
    pub fn update(&mut self, ctx: &egui::Context) {
        let engine = self.engine.get_or_insert_with(|| {
            let c = ctx.clone();
            Engine::new(move || c.request_repaint())
        });
        // zoom is only a display setting: do not recompute for it
        let mut physics = self.params.clone();
        physics.zoom = [1.0; 3];
        engine.request(&physics);
        if let Some(r) = engine.poll() {
            self.accept(ctx, r);
        }
        if self.engine.as_ref().is_some_and(|e| e.busy()) {
            ctx.request_repaint();
        }
    }

    fn accept(&mut self, ctx: &egui::Context, r: FourierResult) {
        let FourierResult {
            params,
            input,
            fourier,
            image,
            side,
            side_intensity,
            side_nz,
            side_m,
            side_half_mm,
            row_phase,
            row_amp,
            fourier_row,
            image_row,
            passed,
            millis,
            side_aliased,
        } = r;
        for (i, img) in [input, fourier, image, side].into_iter().enumerate() {
            match &mut self.tex[i] {
                Some(t) => t.set(img, TextureOptions::LINEAR),
                None => self.tex[i] = Some(ctx.load_texture(format!("fourier{i}"), img, TextureOptions::LINEAR)),
            }
        }
        self.meta = Some(Meta {
            params,
            side_intensity,
            side_nz,
            side_m,
            side_half_mm,
            row_phase,
            row_amp,
            fourier_row,
            image_row,
            passed,
            millis,
            side_aliased,
        });
    }

    // ------------------------------------------------------------ top: the three planes

    pub fn planes_ui(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.columns(3, |cols| {
            self.input_panel(&mut cols[0]);
            self.fourier_panel(&mut cols[1]);
            self.image_panel(&mut cols[2]);
        });
    }

    /// square image area filling the width (or height) of the column
    fn square(ui: &mut egui::Ui, reserve_below: f32) -> (Rect, egui::Response) {
        let avail = ui.available_size() - egui::vec2(0.0, reserve_below);
        let side = avail.x.min(avail.y).max(60.0);
        let (outer, resp) = ui.allocate_exact_size(egui::vec2(avail.x, side), Sense::click_and_drag());
        (Rect::from_center_size(outer.center(), egui::vec2(side, side)), resp)
    }

    fn draw_tex(&self, ui: &egui::Ui, which: usize, rect: Rect) {
        let painter = ui.painter_at(rect.expand(1.0));
        painter.rect_filled(rect, 0.0, Color32::BLACK);
        let z = self.params.zoom[which].max(1.0);
        if let Some(t) = &self.tex[which] {
            let h = 0.5 / z;
            painter.image(t.id(), rect, Rect::from_min_max(egui::pos2(0.5 - h, 0.5 - h), egui::pos2(0.5 + h, 0.5 + h)), Color32::WHITE);
        }
        if z > 1.01 {
            painter.text(rect.left_top() + egui::vec2(6.0, 6.0), Align2::LEFT_TOP, format!("×{z:.1}"), FontId::proportional(12.0), Color32::WHITE);
        }
    }

    /// scroll to zoom, double click to reset
    fn zoom_input(&mut self, ui: &egui::Ui, which: usize, resp: &egui::Response) {
        if resp.hovered() {
            let s = ui.input(|i| i.smooth_scroll_delta.y);
            if s != 0.0 {
                self.params.zoom[which] = (self.params.zoom[which] * (s * 0.004).exp()).clamp(1.0, 32.0);
            }
        }
        if resp.double_clicked() {
            self.params.zoom[which] = 1.0;
        }
    }

    /// the part of a centre-line profile that is visible at the given zoom
    fn visible(row: &[f32], zoom: f32) -> &[f32] {
        let n = row.len();
        let k = ((n as f32 / (2.0 * zoom.max(1.0))) as usize).clamp(2, n / 2);
        &row[n / 2 - k..n / 2 + k]
    }

    fn scale_bar(painter: &egui::Painter, rect: Rect, width_mm: f32) {
        let len = nice_step(width_mm * 0.25, 1.0);
        let px = len / width_mm * rect.width();
        let a = rect.left_bottom() + egui::vec2(10.0, -10.0);
        let b = a + egui::vec2(px, 0.0);
        let st = Stroke::new(2.0, Color32::WHITE);
        painter.line_segment([a, b], st);
        painter.line_segment([a, a - egui::vec2(0.0, 4.0)], st);
        painter.line_segment([b, b - egui::vec2(0.0, 4.0)], st);
        painter.text((a + b.to_vec2()) / 2.0 - egui::vec2(0.0, 5.0), Align2::CENTER_BOTTOM, fmt_mm(len), FontId::proportional(11.0), Color32::WHITE);
    }

    /// physical coordinates (mm, y up) of a point in a panel of the given width
    fn coords(rect: Rect, p: Pos2, width_mm: f32) -> (f32, f32) {
        let s = width_mm / rect.width();
        ((p.x - rect.center().x) * s, (rect.center().y - p.y) * s)
    }

    fn readout(painter: &egui::Painter, rect: Rect, text: String) {
        let pos = rect.right_top() + egui::vec2(-6.0, 6.0);
        let galley = painter.layout_no_wrap(text, FontId::monospace(11.0), Color32::WHITE);
        let r = Align2::RIGHT_TOP.anchor_size(pos, galley.size()).expand(3.0);
        painter.rect_filled(r, 3.0, Color32::from_black_alpha(160));
        painter.galley(r.min + egui::vec2(3.0, 3.0), galley, Color32::WHITE);
    }

    fn input_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.strong("INPUT FIELD");
            ui.weak("right after the object");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                egui::ComboBox::from_id_salt("input_view")
                    .selected_text(match self.params.input_view {
                        InputView::Complex => "amplitude + phase",
                        InputView::Amplitude => "amplitude",
                        InputView::Phase => "phase",
                        InputView::Intensity => "intensity",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.params.input_view, InputView::Complex, "amplitude + phase");
                        ui.selectable_value(&mut self.params.input_view, InputView::Amplitude, "amplitude");
                        ui.selectable_value(&mut self.params.input_view, InputView::Phase, "phase");
                        ui.selectable_value(&mut self.params.input_view, InputView::Intensity, "intensity");
                    });
            });
        });
        let plot_h = PLOT_H;
        let (rect, resp) = Self::square(ui, plot_h + 6.0);
        self.draw_tex(ui, T_INPUT, rect);
        let painter = ui.painter_at(rect);
        self.zoom_input(ui, T_INPUT, &resp);
        let w = self.params.window_mm / self.params.zoom[T_INPUT];
        Self::scale_bar(&painter, rect, w);
        if matches!(self.params.input_view, InputView::Complex | InputView::Phase) {
            phase_wheel(&painter, rect.right_bottom() + egui::vec2(-24.0, -24.0), 16.0);
        }
        // the centre line used for the wavefront plot and the side view
        painter.line_segment(
            [egui::pos2(rect.left(), rect.center().y), egui::pos2(rect.right(), rect.center().y)],
            Stroke::new(1.0, Color32::from_white_alpha(50)),
        );
        if let Some(p) = resp.hover_pos() {
            let (x, y) = Self::coords(rect, p, w);
            Self::readout(&painter, rect, format!("x = {}\ny = {}", fmt_mm(x), fmt_mm(y)));
        }
        ui.add_space(6.0);
        let (plot, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), plot_h), Sense::hover());
        self.wavefront_plot(ui, Rect::from_x_y_ranges(rect.x_range(), plot.y_range()));
    }

    fn wavefront_plot(&self, ui: &egui::Ui, rect: Rect) {
        let painter = ui.painter_at(rect.expand(2.0));
        painter.rect_filled(rect, 3.0, Color32::from_gray(22));
        let Some(m) = &self.meta else { return };
        let zoom = self.params.zoom[T_INPUT];
        let row_phase = Self::visible(&m.row_phase, zoom);
        let row_amp = Self::visible(&m.row_amp, zoom);
        let n = row_phase.len();
        let x_of = |i: usize| rect.left() + (i as f32 + 0.5) / n as f32 * rect.width();
        // amplitude as a grey area
        let base = rect.bottom() - 4.0;
        let h = rect.height() - 22.0;
        let mut area = vec![egui::pos2(rect.left(), base)];
        area.extend((0..n).map(|i| egui::pos2(x_of(i), base - row_amp[i] * h * 0.35)));
        area.push(egui::pos2(rect.right(), base));
        painter.add(egui::Shape::line(area, Stroke::new(1.0, Color32::from_gray(110))));
        // phase in waves
        let (lo, hi) = row_phase
            .iter()
            .filter(|v| v.is_finite())
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| (a.min(*v), b.max(*v)));
        if !lo.is_finite() {
            return;
        }
        let (lo, hi) = if hi - lo < 1.0 { ((lo + hi) * 0.5 - 0.5, (lo + hi) * 0.5 + 0.5) } else { (lo, hi) };
        let top = rect.top() + 16.0;
        let y_of = |v: f32| base - 4.0 - (v - lo) / (hi - lo) * (base - 4.0 - top);
        let col = Color32::from_rgb(255, 210, 90);
        let mut seg: Vec<Pos2> = vec![];
        for i in 0..n {
            let v = row_phase[i];
            if v.is_finite() {
                seg.push(egui::pos2(x_of(i), y_of(v)));
            } else if seg.len() > 1 {
                painter.add(egui::Shape::line(std::mem::take(&mut seg), Stroke::new(1.5, col)));
            } else {
                seg.clear();
            }
        }
        if seg.len() > 1 {
            painter.add(egui::Shape::line(seg, Stroke::new(1.5, col)));
        }
        // grid lines every wave (or more)
        let step = nice_step(hi - lo, 4.0).max(0.25);
        let mut v = (lo / step).ceil() * step;
        while v <= hi {
            let y = y_of(v);
            painter.line_segment([egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)], Stroke::new(1.0, Color32::from_white_alpha(20)));
            painter.text(egui::pos2(rect.right() - 3.0, y), Align2::RIGHT_BOTTOM, format!("{v:+.2} λ"), FontId::monospace(9.0), Color32::from_gray(150));
            v += step;
        }
        painter.text(
            rect.left_top() + egui::vec2(4.0, 2.0),
            Align2::LEFT_TOP,
            "wavefront along the centre line: phase (yellow, in wavelengths), amplitude (grey)",
            FontId::proportional(10.0),
            Color32::from_gray(170),
        );
    }

    fn fourier_panel(&mut self, ui: &mut egui::Ui) {
        let passed = self.meta.as_ref().map(|m| m.passed);
        ui.horizontal(|ui| {
            ui.strong("FOURIER PLANE");
            ui.weak("|U|² behind L1");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.checkbox(&mut self.params.log_fourier, "log");
                if let (Some(pw), true) = (passed, self.params.filter != Filter::None) {
                    ui.weak(format!("filter passes {:.1} %", pw * 100.0));
                }
            });
        });
        let (rect, resp) = Self::square(ui, PLOT_H + 6.0);
        self.draw_tex(ui, T_FOURIER, rect);
        let painter = ui.painter_at(rect);
        self.zoom_input(ui, T_FOURIER, &resp);
        let wf = self.params.fourier_window() / self.params.zoom[T_FOURIER];
        Self::scale_bar(&painter, rect, wf);
        self.draw_filter(&painter, rect, wf);
        // drag in the pattern to size the filter
        if resp.dragged() {
            if let Some(p) = resp.interact_pointer_pos() {
                let (u, v) = Self::coords(rect, p, wf);
                let a = self.params.filter_angle_deg.to_radians();
                let ur = u * a.cos() + v * a.sin();
                let r = (u * u + v * v).sqrt();
                let secondary = resp.dragged_by(egui::PointerButton::Secondary);
                match self.params.filter {
                    Filter::LowPass | Filter::HighPass | Filter::Zernike => self.params.filter_mm = r.max(0.002),
                    Filter::BandPass if secondary => self.params.filter2_mm = r.max(self.params.filter_mm),
                    Filter::BandPass => self.params.filter_mm = r.min(self.params.filter2_mm),
                    Filter::Slit => self.params.filter_mm = (2.0 * ur.abs()).max(0.002),
                    Filter::KnifeEdge => self.params.filter_mm = ur,
                    _ => {}
                }
            }
        }
        if let Some(p) = resp.hover_pos() {
            let (u, v) = Self::coords(rect, p, wf);
            let lf = self.params.lambda_mm() * self.params.f1_mm;
            let r = (u * u + v * v).sqrt();
            Self::readout(
                &painter,
                rect,
                format!(
                    "u = {}   v = {}\nr = {}\nspatial freq. {:.2} lines/mm\nangle {:.2} mrad",
                    fmt_mm(u),
                    fmt_mm(v),
                    fmt_mm(r),
                    r / lf,
                    r / self.params.f1_mm * 1000.0
                ),
            );
            resp.on_hover_text(if matches!(self.params.filter, Filter::None | Filter::Spiral) {
                "scroll to zoom, double-click to reset"
            } else {
                "drag to change the size of the filter · scroll to zoom, double-click to reset"
            });
        }
        ui.add_space(6.0);
        let (plot, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), PLOT_H), Sense::hover());
        if let Some(m) = &self.meta {
            let vis = Self::visible(&m.fourier_row, self.params.zoom[T_FOURIER]);
            let row: Vec<f32> = if self.params.log_fourier {
                vis.iter().map(|v| ((v.max(1e-12).log10() + 5.0) / 5.0).max(0.0)).collect()
            } else {
                vis.to_vec()
            };
            let label = if self.params.log_fourier { "intensity along the centre line (log, 5 decades)" } else { "intensity along the centre line" };
            profile_plot(ui, Rect::from_x_y_ranges(rect.x_range(), plot.y_range()), &row, label, m.params.wavelength_nm);
        }
    }

    fn draw_filter(&self, painter: &egui::Painter, rect: Rect, wf: f32) {
        let p = &self.params;
        let s = rect.width() / wf;
        let c = rect.center();
        let edge = Stroke::new(1.5, Color32::from_rgb(120, 220, 255));
        let shade = Color32::from_rgba_unmultiplied(40, 60, 80, 110);
        let a = p.filter_angle_deg.to_radians();
        let dir = egui::vec2(a.cos(), -a.sin());
        let nrm = egui::vec2(a.sin(), a.cos());
        let far = rect.width() * 1.5;
        match p.filter {
            Filter::None => {}
            Filter::LowPass => {
                painter.circle_stroke(c, p.filter_mm * s, edge);
            }
            Filter::HighPass => {
                painter.circle(c, p.filter_mm * s, shade, edge);
            }
            Filter::BandPass => {
                painter.circle(c, p.filter_mm * s, shade, edge);
                painter.circle_stroke(c, p.filter2_mm * s, edge);
            }
            Filter::Slit => {
                let h = p.filter_mm * 0.5 * s;
                for sgn in [-1.0, 1.0] {
                    let o = c + dir * (sgn * h);
                    painter.line_segment([o - nrm * far, o + nrm * far], edge);
                    let q = c + dir * (sgn * (h + far * 0.5));
                    let pts = vec![q - nrm * far - dir * far * 0.5, q + nrm * far - dir * far * 0.5, q + nrm * far + dir * far * 0.5, q - nrm * far + dir * far * 0.5];
                    painter.add(egui::Shape::convex_polygon(pts, shade, Stroke::NONE));
                }
            }
            Filter::KnifeEdge => {
                let o = c + dir * (p.filter_mm * s);
                painter.line_segment([o - nrm * far, o + nrm * far], edge);
                let q = o - dir * far * 0.5;
                let pts = vec![q - nrm * far - dir * far * 0.5, q + nrm * far - dir * far * 0.5, q + nrm * far + dir * far * 0.5, q - nrm * far + dir * far * 0.5];
                painter.add(egui::Shape::convex_polygon(pts, shade, Stroke::NONE));
            }
            Filter::Zernike => {
                painter.circle_stroke(c, (p.filter_mm * s).max(2.0), Stroke::new(1.5, Color32::from_rgb(255, 150, 255)));
            }
            Filter::Spiral => {
                phase_wheel(painter, c, 14.0);
            }
        }
    }

    fn image_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.strong("IMAGE PLANE");
            ui.weak(format!("|U|² behind L2 (m = −{:.2})", self.params.magnification()));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.checkbox(&mut self.params.log_image, "log");
            });
        });
        let (rect, resp) = Self::square(ui, PLOT_H + 6.0);
        self.draw_tex(ui, T_IMAGE, rect);
        let painter = ui.painter_at(rect);
        self.zoom_input(ui, T_IMAGE, &resp);
        let wi = self.params.window_mm * self.params.magnification() / self.params.zoom[T_IMAGE];
        Self::scale_bar(&painter, rect, wi);
        if let Some(p) = resp.hover_pos() {
            let (x, y) = Self::coords(rect, p, wi);
            Self::readout(&painter, rect, format!("x = {}\ny = {}", fmt_mm(x), fmt_mm(y)));
        }
        ui.add_space(6.0);
        let (plot, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), PLOT_H), Sense::hover());
        if let Some(m) = &self.meta {
            let vis = Self::visible(&m.image_row, self.params.zoom[T_IMAGE]);
            let row: Vec<f32> = if self.params.log_image {
                vis.iter().map(|v| ((v.max(1e-12).log10() + 5.0) / 5.0).max(0.0)).collect()
            } else {
                vis.to_vec()
            };
            profile_plot(ui, Rect::from_x_y_ranges(rect.x_range(), plot.y_range()), &row, "intensity along the centre line", m.params.wavelength_nm);
        }
    }

    // ------------------------------------------------------------ middle: propagation

    pub fn side_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.strong("PROPAGATION");
            ui.weak("intensity in the x–z plane through the centre line of the object (1D model)");
            ui.checkbox(&mut self.params.log_side, "log")
                .on_hover_text("off: every plane normalised to its own maximum; on: one common logarithmic scale");
            if let Some(m) = &self.meta {
                if m.side_aliased {
                    ui.colored_label(Color32::from_rgb(230, 150, 80), "side view undersampled: use a smaller window or longer focal lengths");
                }
            }
        });
        let avail = ui.available_size();
        let profile_w = (avail.x * 0.18).clamp(120.0, 260.0);
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(avail.x - profile_w - 8.0, avail.y), Sense::click_and_drag());
        let prof_rect = Rect::from_min_size(egui::pos2(rect.right() + 8.0, rect.top()), egui::vec2(profile_w, rect.height()));
        let img = Rect::from_min_max(rect.min + egui::vec2(46.0, 18.0), rect.max - egui::vec2(0.0, 20.0));
        let painter = ui.painter_at(rect);
        painter.rect_filled(img, 0.0, Color32::BLACK);
        if let Some(t) = &self.tex[T_SIDE] {
            painter.image(t.id(), img, Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
        }
        let p = &self.params;
        let total = 2.0 * p.f1_mm + 2.0 * p.f2_mm;
        let zx = |z: f32| img.left() + z / total * img.width();
        let half = self.meta.as_ref().map_or(p.window_mm * 0.55, |m| m.side_half_mm);
        let xy = |x: f32| img.center().y - x / half * img.height() * 0.5;
        // elements
        let label_col = Color32::from_gray(200);
        let planes = [
            (0.0, "object"),
            (p.f1_mm, "L1"),
            (2.0 * p.f1_mm, "Fourier plane"),
            (2.0 * p.f1_mm + p.f2_mm, "L2"),
            (total, "image"),
        ];
        for (z, name) in planes {
            let x = zx(z).clamp(img.left() + 1.0, img.right() - 1.0);
            let is_lens = name.starts_with('L');
            painter.line_segment(
                [egui::pos2(x, img.top()), egui::pos2(x, img.bottom())],
                Stroke::new(1.0, if is_lens { Color32::from_rgba_unmultiplied(150, 210, 255, 140) } else { Color32::from_white_alpha(40) }),
            );
            let anchor = if z == 0.0 { Align2::LEFT_BOTTOM } else if z == total { Align2::RIGHT_BOTTOM } else { Align2::CENTER_BOTTOM };
            painter.text(egui::pos2(x, img.top() - 2.0), anchor, name, FontId::proportional(11.0), label_col);
            if is_lens {
                for s in [-1.0f32, 1.0] {
                    let tip = egui::pos2(x, if s > 0.0 { img.top() + 2.0 } else { img.bottom() - 2.0 });
                    painter.line_segment([tip, tip + egui::vec2(-5.0, s * 6.0)], Stroke::new(1.5, Color32::from_rgb(150, 210, 255)));
                    painter.line_segment([tip, tip + egui::vec2(5.0, s * 6.0)], Stroke::new(1.5, Color32::from_rgb(150, 210, 255)));
                }
            }
        }
        // filter: blocked parts of the Fourier plane
        if !matches!(p.filter, Filter::None | Filter::Zernike | Filter::Spiral) {
            let x = zx(2.0 * p.f1_mm);
            let rows = 120;
            for r in 0..rows {
                let xa = half - (r as f32) / rows as f32 * 2.0 * half;
                let xb = half - (r as f32 + 1.0) / rows as f32 * 2.0 * half;
                let u = 0.5 * (xa + xb);
                let a = p.filter_angle_deg.to_radians();
                let ur = u * a.cos();
                let blocked = match p.filter {
                    Filter::LowPass => u.abs() > p.filter_mm,
                    Filter::HighPass => u.abs() < p.filter_mm,
                    Filter::BandPass => u.abs() < p.filter_mm || u.abs() > p.filter2_mm,
                    Filter::Slit => ur.abs() > p.filter_mm * 0.5,
                    Filter::KnifeEdge => ur < p.filter_mm,
                    _ => false,
                };
                if blocked {
                    painter.rect_filled(Rect::from_min_max(egui::pos2(x - 3.0, xy(xa)), egui::pos2(x + 3.0, xy(xb))), 0.0, Color32::from_gray(140));
                }
            }
        }
        // axes
        let step = nice_step(total, 8.0);
        let mut z = 0.0;
        while z <= total + 1e-3 {
            let x = zx(z);
            painter.line_segment([egui::pos2(x, img.bottom()), egui::pos2(x, img.bottom() + 4.0)], Stroke::new(1.0, Color32::GRAY));
            painter.text(egui::pos2(x, img.bottom() + 5.0), Align2::CENTER_TOP, format!("{z:.0}"), FontId::monospace(9.0), Color32::GRAY);
            z += step;
        }
        painter.text(img.right_bottom() + egui::vec2(0.0, 5.0), Align2::RIGHT_TOP, "z / mm", FontId::monospace(9.0), Color32::GRAY);
        let xstep = nice_step(2.0 * half, 6.0);
        let mut xv = -(half / xstep).floor() * xstep;
        while xv <= half {
            let y = xy(xv);
            painter.line_segment([egui::pos2(img.left() - 4.0, y), egui::pos2(img.left(), y)], Stroke::new(1.0, Color32::GRAY));
            painter.text(egui::pos2(img.left() - 6.0, y), Align2::RIGHT_CENTER, format!("{xv:.1}"), FontId::monospace(9.0), Color32::GRAY);
            xv += xstep;
        }
        painter.text(egui::pos2(img.left() - 6.0, img.top()), Align2::RIGHT_BOTTOM, "x / mm", FontId::monospace(9.0), Color32::GRAY);
        // observation plane
        if resp.dragged() || resp.clicked() {
            if let Some(q) = resp.interact_pointer_pos() {
                self.z_marker = ((q.x - img.left()) / img.width()).clamp(0.0, 1.0);
            }
        }
        let zm = self.z_marker * total;
        let mx = zx(zm);
        painter.line_segment([egui::pos2(mx, img.top()), egui::pos2(mx, img.bottom())], Stroke::new(1.5, Color32::from_rgb(255, 220, 80)));
        let where_ = if zm < p.f1_mm {
            format!("{zm:.0} mm behind the object")
        } else if zm < 2.0 * p.f1_mm {
            format!("{:.0} mm behind L1", zm - p.f1_mm)
        } else if zm < 2.0 * p.f1_mm + p.f2_mm {
            format!("{:.0} mm behind the Fourier plane", zm - 2.0 * p.f1_mm)
        } else {
            format!("{:.0} mm behind L2", zm - 2.0 * p.f1_mm - p.f2_mm)
        };
        painter.text(
            egui::pos2(mx + 4.0, img.bottom() - 4.0),
            Align2::LEFT_BOTTOM,
            format!("z = {zm:.0} mm ({where_})"),
            FontId::proportional(11.0),
            Color32::from_rgb(255, 220, 80),
        );
        if resp.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
        }
        self.profile(ui, prof_rect, img);
    }

    /// intensity across the beam at the observation plane, drawn sideways next to the side view
    fn profile(&self, ui: &egui::Ui, rect: Rect, img: Rect) {
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3.0, Color32::from_gray(22));
        painter.text(rect.left_top() + egui::vec2(4.0, 2.0), Align2::LEFT_TOP, "profile at z (yellow line)", FontId::proportional(10.0), Color32::from_gray(170));
        let Some(m) = &self.meta else { return };
        let c = ((self.z_marker * m.side_nz as f32) as usize).min(m.side_nz - 1);
        let col = &m.side_intensity[c * m.side_m..(c + 1) * m.side_m];
        let max = col.iter().cloned().fold(0.0f32, f32::max).max(1e-30);
        let tint = wavelength_rgb(m.params.wavelength_nm);
        let colr = Color32::from_rgb((tint[0] * 255.0) as u8, (tint[1] * 255.0) as u8, (tint[2] * 255.0) as u8);
        let pts: Vec<Pos2> = col
            .iter()
            .enumerate()
            .map(|(r, v)| {
                let y = img.top() + (r as f32 + 0.5) / m.side_m as f32 * img.height();
                egui::pos2(rect.left() + 6.0 + v / max * (rect.width() - 12.0), y)
            })
            .collect();
        painter.add(egui::Shape::line(pts, Stroke::new(1.5, colr.gamma_multiply(1.3))));
        painter.line_segment([egui::pos2(rect.left() + 6.0, img.top()), egui::pos2(rect.left() + 6.0, img.bottom())], Stroke::new(1.0, Color32::from_gray(80)));
    }

    // ------------------------------------------------------------ bottom: controls

    pub fn controls_ui(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        if !self.notes.trim().is_empty() {
            ui.horizontal_wrapped(|ui| {
                ui.strong("About this setup:");
                ui.label(&self.notes);
            });
            ui.separator();
        }
        egui::ScrollArea::vertical().id_salt("fourier_controls").show(ui, |ui| {
            ui.columns(4, |cols| {
                self.light_controls(&mut cols[0]);
                self.object_controls(&mut cols[1]);
                self.filter_controls(&mut cols[2]);
                self.system_controls(&mut cols[3]);
            });
        });
    }

    fn light_controls(&mut self, ui: &mut egui::Ui) {
        let p = &mut self.params;
        ui.strong("LIGHT");
        egui::Grid::new("light").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
            ui.label("wavelength");
            ui.horizontal(|ui| {
                ui.add(egui::Slider::new(&mut p.wavelength_nm, 400.0..=700.0).suffix(" nm").step_by(1.0));
                let t = wavelength_rgb(p.wavelength_nm);
                let (r, _) = ui.allocate_exact_size(egui::vec2(14.0, 14.0), Sense::hover());
                ui.painter().circle_filled(r.center(), 6.0, Color32::from_rgb((t[0] * 255.0) as u8, (t[1] * 255.0) as u8, (t[2] * 255.0) as u8));
            });
            ui.end_row();
            ui.label("beam");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut p.beam, Beam::Plane, "plane wave");
                ui.selectable_value(&mut p.beam, Beam::Gaussian, "Gaussian");
                ui.selectable_value(&mut p.beam, Beam::TopHat, "top hat");
            });
            ui.end_row();
            if p.beam != Beam::Plane {
                ui.label(if p.beam == Beam::Gaussian { "waist w₀" } else { "diameter" });
                ui.add(egui::Slider::new(&mut p.beam_mm, 0.1..=8.0).logarithmic(true).suffix(" mm"));
                ui.end_row();
            }
            let max_tilt = p.max_tilt_mrad();
            ui.label("tilt x / y");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut p.tilt_x_mrad).speed(0.05).range(-max_tilt..=max_tilt).suffix(" mrad"));
                ui.add(egui::DragValue::new(&mut p.tilt_y_mrad).speed(0.05).range(-max_tilt..=max_tilt).suffix(" mrad"));
            })
            .response
            .on_hover_text("A tilted plane wave has a linear phase ramp; it shifts the Fourier pattern by f·θ.");
            ui.end_row();
            ui.label("point source");
            ui.horizontal(|ui| {
                let mut on = p.source_mm != 0.0;
                if ui.checkbox(&mut on, "").changed() {
                    p.source_mm = if on { 1000.0 } else { 0.0 };
                }
                if on {
                    ui.add(egui::Slider::new(&mut p.source_mm, -5000.0..=5000.0).suffix(" mm"))
                        .on_hover_text("Distance of a point source in front of the object: a curved (spherical) wavefront. Negative: converging.");
                } else {
                    ui.weak("off (plane wavefront)");
                }
            });
            ui.end_row();
        });
    }

    fn object_controls(&mut self, ui: &mut egui::Ui) {
        let p = &mut self.params;
        ui.strong("OBJECT");
        egui::ComboBox::from_id_salt("object").width(220.0).selected_text(p.object.label()).show_ui(ui, |ui| {
            for o in Object::ALL {
                ui.selectable_value(&mut p.object, o, o.label());
            }
        });
        let (size, period, duty, angle, phase) = p.object.uses();
        egui::Grid::new("object").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
            if size {
                ui.label(match p.object {
                    Object::SingleSlit | Object::DoubleSlit => "slit width",
                    Object::LetterF | Object::MeshF | Object::PhaseF => "letter height",
                    Object::Square => "side",
                    _ => "diameter",
                });
                ui.add(egui::Slider::new(&mut p.size_mm, 0.01..=8.0).logarithmic(true).suffix(" mm"));
                ui.end_row();
            }
            if matches!(p.object, Object::SingleSlit | Object::DoubleSlit) {
                ui.label("slit length");
                ui.add(egui::Slider::new(&mut p.length_mm, 0.05..=8.0).logarithmic(true).suffix(" mm"));
                ui.end_row();
            }
            if period {
                ui.label(if p.object == Object::DoubleSlit { "slit distance" } else { "period" });
                ui.add(egui::Slider::new(&mut p.period_mm, 0.02..=4.0).logarithmic(true).suffix(" mm"));
                ui.end_row();
            }
            if duty {
                ui.label("open fraction");
                ui.add(egui::Slider::new(&mut p.duty, 0.05..=0.95));
                ui.end_row();
            }
            if angle {
                ui.label("rotation");
                ui.add(egui::Slider::new(&mut p.angle_deg, -90.0..=90.0).suffix("°"));
                ui.end_row();
            }
            if phase {
                if p.object == Object::Vortex {
                    ui.label("charge ℓ");
                    ui.add(egui::Slider::new(&mut p.phase, -5.0..=5.0).step_by(1.0));
                } else {
                    ui.label("phase delay");
                    let mut waves = p.phase / (2.0 * PI);
                    if ui.add(egui::Slider::new(&mut waves, 0.0..=1.0).suffix(" λ")).changed() {
                        p.phase = waves * 2.0 * PI;
                    }
                }
                ui.end_row();
            }
        });
    }

    fn filter_controls(&mut self, ui: &mut egui::Ui) {
        let p = &mut self.params;
        ui.strong("FILTER IN THE FOURIER PLANE");
        egui::ComboBox::from_id_salt("filter").width(220.0).selected_text(p.filter.label()).show_ui(ui, |ui| {
            for f in Filter::ALL {
                ui.selectable_value(&mut p.filter, f, f.label());
            }
        });
        let lf = p.lambda_mm() * p.f1_mm;
        let wf = p.fourier_window();
        egui::Grid::new("filter").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
            let radius = |ui: &mut egui::Ui, label: &str, v: &mut f32| {
                ui.label(label);
                ui.add(egui::Slider::new(v, 0.002..=wf * 0.5).logarithmic(true).suffix(" mm"))
                    .on_hover_text(format!("{:.2} lines/mm", *v / lf));
                ui.end_row();
            };
            match p.filter {
                Filter::None => {}
                Filter::LowPass | Filter::HighPass => radius(ui, "radius", &mut p.filter_mm),
                Filter::BandPass => {
                    radius(ui, "inner radius", &mut p.filter_mm);
                    radius(ui, "outer radius", &mut p.filter2_mm);
                }
                Filter::Slit => {
                    radius(ui, "slit width", &mut p.filter_mm);
                    ui.label("rotation");
                    ui.add(egui::Slider::new(&mut p.filter_angle_deg, -90.0..=90.0).suffix("°"));
                    ui.end_row();
                }
                Filter::KnifeEdge => {
                    ui.label("edge position");
                    ui.add(egui::Slider::new(&mut p.filter_mm, -wf * 0.2..=wf * 0.2).suffix(" mm"));
                    ui.end_row();
                    ui.label("rotation");
                    ui.add(egui::Slider::new(&mut p.filter_angle_deg, -180.0..=180.0).suffix("°"));
                    ui.end_row();
                }
                Filter::Zernike => {
                    radius(ui, "dot radius", &mut p.filter_mm);
                    ui.label("phase shift");
                    let mut waves = p.filter_phase / (2.0 * PI);
                    if ui.add(egui::Slider::new(&mut waves, -0.5..=0.5).suffix(" λ")).changed() {
                        p.filter_phase = waves * 2.0 * PI;
                    }
                    ui.end_row();
                }
                Filter::Spiral => {
                    ui.label("charge");
                    ui.add(egui::Slider::new(&mut p.filter_phase, -3.0..=3.0).step_by(1.0));
                    ui.end_row();
                }
            }
        });
        if !matches!(p.filter, Filter::None | Filter::Spiral) {
            ui.weak("Tip: drag inside the Fourier plane to size the filter (band pass: right-drag for the outer radius).");
        }
    }

    fn system_controls(&mut self, ui: &mut egui::Ui) {
        let p = &mut self.params;
        ui.strong("LENSES & SIMULATION");
        egui::Grid::new("system").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
            ui.label("f₁");
            ui.add(egui::Slider::new(&mut p.f1_mm, 20.0..=1000.0).logarithmic(true).suffix(" mm"));
            ui.end_row();
            ui.label("f₂");
            ui.add(egui::Slider::new(&mut p.f2_mm, 20.0..=1000.0).logarithmic(true).suffix(" mm"));
            ui.end_row();
            ui.label("object window");
            ui.add(egui::Slider::new(&mut p.window_mm, 1.0..=40.0).logarithmic(true).suffix(" mm"));
            ui.end_row();
            ui.label("grid");
            ui.horizontal(|ui| {
                for n in [256, 512, 1024] {
                    ui.selectable_value(&mut p.n, n, format!("{n}²"));
                }
            });
            ui.end_row();
        });
        ui.add_space(4.0);
        ui.weak(format!(
            "pixel: {} (object), {} (Fourier plane)\nFourier plane: {} wide = ±{:.1} lines/mm\nmagnification −f₂/f₁ = −{:.2}",
            fmt_mm(p.dx()),
            fmt_mm(p.du()),
            fmt_mm(p.fourier_window()),
            p.fourier_window() * 0.5 / (p.lambda_mm() * p.f1_mm),
            p.magnification()
        ));
        if let Some(m) = &self.meta {
            ui.weak(format!("computed in {:.0} ms", m.millis));
        }
    }
}

/// a normalised 1D curve (values 0..1) under an image panel
fn profile_plot(ui: &egui::Ui, rect: Rect, row: &[f32], label: &str, wavelength_nm: f32) {
    let painter = ui.painter_at(rect.expand(2.0));
    painter.rect_filled(rect, 3.0, Color32::from_gray(22));
    let n = row.len().max(1);
    let t = wavelength_rgb(wavelength_nm);
    let col = Color32::from_rgb((t[0] * 255.0) as u8, (t[1] * 255.0) as u8, (t[2] * 255.0) as u8);
    let base = rect.bottom() - 4.0;
    let h = rect.height() - 20.0;
    let pts: Vec<Pos2> = row
        .iter()
        .enumerate()
        .map(|(i, v)| egui::pos2(rect.left() + (i as f32 + 0.5) / n as f32 * rect.width(), base - v.clamp(0.0, 1.0) * h))
        .collect();
    painter.add(egui::Shape::line(pts, Stroke::new(1.3, col.gamma_multiply(1.4))));
    painter.text(rect.left_top() + egui::vec2(4.0, 2.0), Align2::LEFT_TOP, label, FontId::proportional(10.0), Color32::from_gray(170));
}

/// small colour wheel explaining the phase colours
fn phase_wheel(painter: &egui::Painter, c: Pos2, r: f32) {
    let n = 36;
    for i in 0..n {
        let a0 = i as f32 / n as f32 * 2.0 * PI;
        let a1 = (i + 1) as f32 / n as f32 * 2.0 * PI;
        let col = {
            let hs = i as f32 / n as f32 * 6.0;
            let x = 1.0 - (hs % 2.0 - 1.0).abs();
            let (r_, g_, b_) = match hs as i32 {
                0 => (1.0, x, 0.0),
                1 => (x, 1.0, 0.0),
                2 => (0.0, 1.0, x),
                3 => (0.0, x, 1.0),
                4 => (x, 0.0, 1.0),
                _ => (1.0, 0.0, x),
            };
            Color32::from_rgb((r_ * 230.0) as u8, (g_ * 230.0) as u8, (b_ * 230.0) as u8)
        };
        let pts = vec![c, c + egui::vec2(a0.cos(), -a0.sin()) * r, c + egui::vec2(a1.cos(), -a1.sin()) * r];
        painter.add(egui::Shape::convex_polygon(pts, col, Stroke::NONE));
    }
    painter.circle_stroke(c, r, Stroke::new(1.0, Color32::from_black_alpha(150)));
    painter.text(c + egui::vec2(r + 2.0, 0.0), Align2::LEFT_CENTER, "0", FontId::monospace(9.0), Color32::WHITE);
    painter.text(c + egui::vec2(-r - 2.0, 0.0), Align2::RIGHT_CENTER, "π", FontId::monospace(9.0), Color32::WHITE);
}
