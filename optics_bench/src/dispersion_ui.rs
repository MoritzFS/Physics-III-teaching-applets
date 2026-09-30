//! User interface of the dispersion bench: the wave in the medium (top), its
//! path in the x–t plane with the signal at an observer (middle), and the
//! dispersion relation with the controls (bottom). The second tab plays
//! thunder and whistlers (see sound_ui.rs).

use std::f64::consts::TAU;
use std::sync::Arc;

use eframe::egui::{self, pos2, vec2, Align2, Color32, FontId, Pos2, Rect, Sense, Shape, Stroke, TextureHandle, TextureOptions};
use rustfft::num_complex::Complex32 as C32;
use rustfft::FftPlanner;

use crate::audio::{self, Audio};
use crate::dispersion::{
    default_free, space_time, DispParams, DispPreset, Model, PlotView, Setup, Shape as PulseShape, Tab, XtRequest, XtResult,
    FREE_POINTS, FREE_STEP, W0,
};
use crate::fourier::wavelength_rgb;
use crate::sound::FS;
use crate::sound_ui::SoundUi;
use crate::worker::Worker;

/// left margin for the axis labels of the wave and space–time views
const ML: f32 = 50.0;
const AXIS_H: f32 = 18.0;

const WAVE: Color32 = Color32::from_rgb(140, 210, 255);
const PHASE: Color32 = Color32::from_rgb(255, 160, 80);
const GROUP: Color32 = Color32::from_rgb(110, 230, 160);
const OBSERVER: Color32 = Color32::from_rgb(255, 220, 80);
const INDEX: Color32 = Color32::from_rgb(235, 235, 235);
const GROUP_INDEX: Color32 = Color32::from_rgb(110, 190, 255);
const ABSORB: Color32 = Color32::from_rgb(240, 100, 90);
const PICKS: [Color32; 6] = [
    Color32::from_rgb(230, 120, 255),
    Color32::from_rgb(90, 225, 230),
    Color32::from_rgb(175, 230, 90),
    Color32::from_rgb(255, 130, 160),
    Color32::from_rgb(250, 195, 80),
    Color32::from_rgb(130, 150, 255),
];
const MAX_PICKS: usize = 6;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Handle {
    N0,
    Np,
    Ng,
    Gvd,
    GlassN,
    GlassB,
    ResW,
    Cutoff,
    PowerN,
    Free(usize),
}

#[derive(Clone, Copy, PartialEq)]
enum Drag {
    Handle(Handle),
    Observer,
    Pan,
    Scrub,
}

/// the observer signal that is being played
struct Listen {
    id: u64,
    /// simulated time per second of sound (T₀/s)
    rate: f64,
}

/// maps x (λ₀) to the screen
#[derive(Clone, Copy)]
struct XMap {
    left: f32,
    right: f32,
    x0: f64,
    x1: f64,
}

impl XMap {
    fn sx(&self, x: f64) -> f32 {
        self.left + ((x - self.x0) / (self.x1 - self.x0)) as f32 * (self.right - self.left)
    }
    fn xs(&self, px: f32) -> f64 {
        self.x0 + ((px - self.left) / (self.right - self.left)) as f64 * (self.x1 - self.x0)
    }
}

/// a 2D plot area
#[derive(Clone, Copy)]
struct Axes {
    r: Rect,
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    log_x: bool,
}

impl Axes {
    fn new(r: Rect, x: (f64, f64), y: (f64, f64)) -> Axes {
        Axes { r, x0: x.0, x1: x.1, y0: y.0, y1: y.1, log_x: false }
    }
    fn fx(&self, x: f64) -> f64 {
        if self.log_x { (x.max(1e-9).ln() - self.x0.ln()) / (self.x1.ln() - self.x0.ln()) } else { (x - self.x0) / (self.x1 - self.x0) }
    }
    fn p(&self, x: f64, y: f64) -> Pos2 {
        pos2(
            self.r.left() + self.fx(x) as f32 * self.r.width(),
            self.r.bottom() - ((y - self.y0) / (self.y1 - self.y0)) as f32 * self.r.height(),
        )
    }
    fn inv(&self, p: Pos2) -> (f64, f64) {
        let fx = ((p.x - self.r.left()) / self.r.width()) as f64;
        let x = if self.log_x { (self.x0.ln() + fx * (self.x1.ln() - self.x0.ln())).exp() } else { self.x0 + fx * (self.x1 - self.x0) };
        (x, self.y0 + ((self.r.bottom() - p.y) / self.r.height()) as f64 * (self.y1 - self.y0))
    }
    /// background, grid, tick labels
    fn frame(&self, painter: &egui::Painter, xlabel: &str, ylabel: &str) {
        painter.rect_filled(self.r, 3.0, Color32::from_gray(22));
        let grid = Stroke::new(1.0, Color32::from_white_alpha(14));
        let font = FontId::monospace(9.0);
        let col = Color32::from_gray(150);
        if self.log_x {
            let mut d = 10f64.powf(self.x0.log10().floor());
            while d <= self.x1 {
                for m in [1.0, 2.0, 5.0] {
                    let v = d * m;
                    if v >= self.x0 && v <= self.x1 {
                        let x = self.p(v, self.y0).x;
                        painter.line_segment([pos2(x, self.r.top()), pos2(x, self.r.bottom())], grid);
                        painter.text(pos2(x, self.r.bottom() + 2.0), Align2::CENTER_TOP, fmt_hz(v), font.clone(), col);
                    }
                }
                d *= 10.0;
            }
        } else {
            for v in ticks(self.x0, self.x1, 6.0) {
                let x = self.p(v, self.y0).x;
                painter.line_segment([pos2(x, self.r.top()), pos2(x, self.r.bottom())], grid);
                painter.text(pos2(x, self.r.bottom() + 2.0), Align2::CENTER_TOP, fmt_tick(v), font.clone(), col);
            }
        }
        for v in ticks(self.y0, self.y1, 4.0) {
            let y = self.p(self.x0, v).y;
            painter.line_segment([pos2(self.r.left(), y), pos2(self.r.right(), y)], grid);
            painter.text(pos2(self.r.left() - 4.0, y), Align2::RIGHT_CENTER, fmt_tick(v), font.clone(), col);
        }
        painter.text(self.r.right_bottom() + vec2(0.0, 12.0), Align2::RIGHT_TOP, xlabel, FontId::proportional(11.0), col);
        painter.text(self.r.left_top() + vec2(-6.0, -2.0), Align2::RIGHT_BOTTOM, ylabel, FontId::proportional(11.0), col);
    }
    /// a curve; points that are not finite break it
    fn curve(&self, painter: &egui::Painter, pts: &[(f64, f64)], stroke: Stroke, dashed: bool) {
        let mut seg: Vec<Pos2> = vec![];
        let lim = 20.0 * (self.y1 - self.y0).abs();
        let flush = |seg: &mut Vec<Pos2>| {
            if seg.len() > 1 {
                if dashed {
                    painter.extend(Shape::dashed_line(seg, stroke, 6.0, 4.0));
                } else {
                    painter.add(Shape::line(std::mem::take(seg), stroke));
                }
            }
            seg.clear();
        };
        for &(x, y) in pts {
            if x.is_finite() && y.is_finite() && (y - self.y0).abs() < lim {
                seg.push(self.p(x, y));
            } else {
                flush(&mut seg);
            }
        }
        flush(&mut seg);
    }
}

pub fn nice_step(range: f64, target: f64) -> f64 {
    let raw = (range / target).abs().max(1e-12);
    let mag = 10f64.powf(raw.log10().floor());
    for m in [1.0, 2.0, 5.0, 10.0] {
        if raw <= m * mag {
            return m * mag;
        }
    }
    10.0 * mag
}

pub fn ticks(a: f64, b: f64, target: f64) -> Vec<f64> {
    let step = nice_step(b - a, target);
    let mut v = (a / step).ceil() * step;
    let mut out = vec![];
    while v <= b + 1e-9 * step && out.len() < 50 {
        out.push(if v.abs() < 1e-9 * step { 0.0 } else { v });
        v += step;
    }
    out
}

pub fn fmt_tick(v: f64) -> String {
    let a = v.abs();
    if a == 0.0 {
        "0".into()
    } else if a >= 100.0 || (a >= 1.0 && v.fract().abs() < 1e-9) {
        format!("{v:.0}")
    } else if a >= 1.0 {
        format!("{v:.1}")
    } else if a >= 0.01 {
        format!("{v:.2}")
    } else {
        format!("{v:.0e}")
    }
}

pub fn fmt_hz(f: f64) -> String {
    if f >= 1000.0 { format!("{}k", fmt_tick(f / 1000.0)) } else { fmt_tick(f) }
}

/// a box with monospace text in a corner
pub fn readout(painter: &egui::Painter, pos: Pos2, align: Align2, text: String) {
    let galley = painter.layout_no_wrap(text, FontId::monospace(11.0), Color32::WHITE);
    let r = align.anchor_size(pos, galley.size()).expand(3.0);
    painter.rect_filled(r, 3.0, Color32::from_black_alpha(170));
    painter.galley(r.min + vec2(3.0, 3.0), galley, Color32::WHITE);
}

/// coloured lines of text on a dark box in the top left corner of a plot
fn legend_box(painter: &egui::Painter, plot: Rect, items: &[(&str, Color32)]) {
    let galleys: Vec<_> = items.iter().map(|(t, c)| painter.layout_no_wrap(t.to_string(), FontId::proportional(11.0), *c)).collect();
    let w = galleys.iter().map(|g| g.size().x).fold(0.0, f32::max);
    let h: f32 = galleys.iter().map(|g| g.size().y).sum();
    let r = Rect::from_min_size(plot.left_top() + vec2(4.0, 4.0), vec2(w + 8.0, h + 6.0));
    painter.rect_filled(r, 3.0, Color32::from_black_alpha(170));
    let mut y = r.top() + 3.0;
    for (g, (_, c)) in galleys.into_iter().zip(items) {
        let dy = g.size().y;
        painter.galley(pos2(r.left() + 4.0, y), g, *c);
        y += dy;
    }
}

/// a label on a dark box
fn tag(painter: &egui::Painter, pos: Pos2, align: Align2, text: String, col: Color32) {
    let g = painter.layout_no_wrap(text, FontId::proportional(10.0), col);
    let r = align.anchor_size(pos, g.size()).expand(2.0);
    painter.rect_filled(r, 2.0, Color32::from_black_alpha(170));
    painter.galley(r.min + vec2(2.0, 2.0), g, col);
}

fn dashed_v(painter: &egui::Painter, x: f32, top: f32, bottom: f32, stroke: Stroke) {
    painter.extend(Shape::dashed_line(&[pos2(x, top), pos2(x, bottom)], stroke, 5.0, 4.0));
}

fn strip_w(width: f32) -> f32 {
    (width * 0.16).clamp(150.0, 240.0)
}

/// colour of a local wavenumber relative to the carrier: long waves red, short ones blue
fn local_colour(ratio: f64) -> Color32 {
    let t = wavelength_rgb((550.0 / ratio.max(0.1)).clamp(410.0, 690.0) as f32);
    let c = |v: f32| (v * 255.0).min(255.0) as u8;
    Color32::from_rgb(c(t[0] * 1.2), c(t[1] * 1.2), c(t[2] * 1.2))
}

pub struct DispUi {
    pub params: DispParams,
    /// notes of the wave-packet tab and of the sound tab
    pub notes: String,
    pub sound_notes: String,
    planner: FftPlanner<f32>,
    setup: Option<Arc<Setup>>,
    /// simulated time (T₀)
    t: f64,
    playing: bool,
    xt: Option<Worker<XtRequest, XtResult>>,
    xt_tex: Option<TextureHandle>,
    /// setup, x range and t_end of the space–time image shown
    xt_meta: Option<(u64, f64, f64, f64, f32)>,
    /// ψ(x_obs, t) on the FFT grid, for (setup id, x_obs)
    trace: Option<(u64, f32, Vec<C32>)>,
    field: Vec<C32>,
    drag: Option<Drag>,
    /// y range of the n(ω) plot, kept while a handle is dragged
    frozen_y: Option<(f64, f64)>,
    audio: Audio,
    listen: Option<Listen>,
    sound: SoundUi,
    /// which crest each marker dot follows (carrier, then the picks), for (setup id, number of picks)
    crests: (u64, usize, Vec<Option<f64>>),
    crest_t: f64,
}

impl Default for DispUi {
    fn default() -> Self {
        let (params, notes) = DispPreset::Spreading.setup();
        DispUi {
            params,
            notes,
            sound_notes: DispPreset::Thunder.setup().1,
            planner: FftPlanner::new(),
            setup: None,
            t: 0.0,
            playing: false,
            xt: None,
            xt_tex: None,
            xt_meta: None,
            trace: None,
            field: vec![],
            drag: None,
            frozen_y: None,
            audio: Audio::default(),
            listen: None,
            sound: SoundUi::default(),
            crests: (0, 0, vec![]),
            crest_t: 0.0,
        }
    }
}

impl DispUi {
    pub fn load_preset(&mut self, p: DispPreset) {
        let (params, notes) = p.setup();
        if params.tab == Tab::Sound {
            self.sound_notes = notes;
        } else {
            self.notes = notes;
        }
        self.playing = params.tab == Tab::Waves;
        self.params = params;
        self.t = 0.0;
        self.audio.stop();
        self.listen = None;
    }

    /// shows the wave at a fraction of the whole run and stops (for screenshots)
    pub fn set_time_fraction(&mut self, f: f64) {
        self.refresh();
        if let Some(s) = &self.setup {
            self.t = f.clamp(0.0, 1.0) * s.t_end;
            self.playing = false;
        }
    }

    fn refresh(&mut self) {
        let key = self.params.phys();
        if self.setup.as_ref().is_some_and(|s| s.key == key) {
            return;
        }
        let s = Setup::new(&key, &mut self.planner);
        self.t = self.t.min(s.t_end);
        self.params.observer = self.params.observer.clamp(s.x_min as f32, s.x_max as f32);
        self.setup = Some(Arc::new(s));
    }

    fn toggle_play(&mut self) {
        if let Some(s) = &self.setup {
            if !self.playing && self.t >= s.t_end - 1e-9 {
                self.t = 0.0;
            }
        }
        self.playing = !self.playing;
        if self.listen.take().is_some() {
            self.audio.stop();
        }
    }

    fn view(&self, s: &Setup) -> (f64, f64) {
        let span = s.x_max - s.x_min;
        (s.x_min + self.params.zoom[0] as f64 * span, s.x_min + self.params.zoom[1] as f64 * span)
    }

    /// call every frame while the dispersion bench is shown
    pub fn update(&mut self, ctx: &egui::Context) {
        if !ctx.egui_wants_keyboard_input() && ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Space)) {
            match self.params.tab {
                Tab::Waves => self.toggle_play(),
                Tab::Sound => self.sound.toggle(&self.params.sound, &mut self.audio),
            }
        }
        if self.params.tab == Tab::Sound {
            self.sound.update(ctx, &self.params.sound, &mut self.audio);
            return;
        }
        self.refresh();
        let Some(s) = self.setup.clone() else { return };
        if let Some(l) = &self.listen {
            match self.audio.playing() {
                Some((id, pos)) if id == l.id => {
                    self.t = (pos as f64 * l.rate).min(s.t_end);
                    self.playing = false;
                    ctx.request_repaint();
                }
                _ => self.listen = None,
            }
        }
        if self.playing {
            let dt = ctx.input(|i| i.stable_dt).min(0.1) as f64;
            self.t += dt * s.t_end / 12.0 * self.params.speed as f64;
            if self.t >= s.t_end {
                if self.params.looping {
                    self.t = 0.0;
                } else {
                    self.t = s.t_end;
                    self.playing = false;
                }
            }
            ctx.request_repaint();
        }
        let worker = self.xt.get_or_insert_with(|| {
            let c = ctx.clone();
            let mut planner = FftPlanner::new();
            Worker::new("space-time", move |r: XtRequest| space_time(&r, &mut planner), move || c.request_repaint())
        });
        if let Some(r) = worker.poll() {
            match &mut self.xt_tex {
                Some(t) => t.set(r.image, TextureOptions::LINEAR),
                None => self.xt_tex = Some(ctx.load_texture("dispersion_xt", r.image, TextureOptions::LINEAR)),
            }
            self.xt_meta = Some((r.setup_id, r.x0, r.x1, r.t_end, r.millis));
        }
        if worker.busy() {
            ctx.request_repaint();
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        if self.params.tab == Tab::Sound {
            self.sound.ui(ui, &mut self.params.sound, &mut self.audio, &self.sound_notes);
            return;
        }
        let win_h = ui.ctx().content_rect().height();
        egui::Panel::top("d_wave")
            .resizable(true)
            .default_size(win_h * 0.3)
            .min_size(150.0)
            .max_size(win_h * 0.55)
            .show(ui, |ui| self.wave_ui(ui));
        egui::Panel::bottom("d_controls")
            .resizable(true)
            .default_size(win_h * 0.37)
            .min_size(200.0)
            .max_size(win_h * 0.6)
            .show(ui, |ui| self.bottom_ui(ui));
        egui::CentralPanel::default().show(ui, |ui| self.space_time_ui(ui));
    }

    // ------------------------------------------------------------ zoom, pan, observer

    fn zoom_input(&mut self, ui: &egui::Ui, resp: &egui::Response, xm: &XMap, s: &Setup) {
        let span = s.x_max - s.x_min;
        let [z0, z1] = self.params.zoom;
        if resp.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                if let Some(p) = resp.hover_pos() {
                    let c = z0 + (p.x - xm.left) / (xm.right - xm.left) * (z1 - z0);
                    let w = ((z1 - z0) * (-scroll * 0.003).exp()).clamp((5.0 / span) as f32, 1.0);
                    let a = (c - (c - z0) * w / (z1 - z0)).clamp(0.0, 1.0 - w);
                    self.params.zoom = [a, a + w];
                }
            }
        }
        if resp.double_clicked() {
            self.params.zoom = [0.0, 1.0];
        }
    }

    fn pan(&mut self, dx_px: f32, xm: &XMap) {
        let [z0, z1] = self.params.zoom;
        let w = z1 - z0;
        let a = (z0 - dx_px / (xm.right - xm.left) * w).clamp(0.0, 1.0 - w);
        self.params.zoom = [a, a + w];
    }

    fn observer_input(&mut self, ui: &egui::Ui, resp: &egui::Response, xm: &XMap, s: &Setup, scrub: Option<Rect>) {
        let obs_x = xm.sx(self.params.observer as f64);
        let near_obs = |p: Pos2| (p.x - obs_x).abs() < 8.0;
        if resp.drag_started() {
            let o = ui.input(|i| i.pointer.press_origin()).unwrap_or_default();
            self.drag = Some(if near_obs(o) {
                Drag::Observer
            } else if scrub.is_some() {
                Drag::Scrub
            } else {
                Drag::Pan
            });
        }
        if resp.dragged() {
            match self.drag {
                Some(Drag::Observer) => {
                    if let Some(p) = resp.interact_pointer_pos() {
                        self.params.observer = xm.xs(p.x).clamp(s.x_min, s.x_max) as f32;
                    }
                }
                Some(Drag::Pan) => self.pan(resp.drag_delta().x, xm),
                _ => {}
            }
        }
        if let (Some(r), true) = (scrub, resp.clicked() || (resp.dragged() && self.drag == Some(Drag::Scrub))) {
            if let Some(p) = resp.interact_pointer_pos() {
                self.t = (((p.y - r.top()) / r.height()) as f64 * s.t_end).clamp(0.0, s.t_end);
                self.playing = false;
            }
        }
        if resp.drag_stopped() {
            self.drag = None;
        }
        if let Some(p) = resp.hover_pos() {
            if near_obs(p) || self.drag == Some(Drag::Observer) {
                ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
            }
        }
    }

    fn draw_observer(&self, painter: &egui::Painter, xm: &XMap, top: f32, bottom: f32, label: bool) {
        let x = xm.sx(self.params.observer as f64);
        dashed_v(painter, x, top, bottom, Stroke::new(1.5, OBSERVER));
        if label {
            let tri = vec![pos2(x - 6.0, top), pos2(x + 6.0, top), pos2(x, top + 8.0)];
            painter.add(Shape::convex_polygon(tri, OBSERVER, Stroke::NONE));
            painter.text(pos2(x + 8.0, top), Align2::LEFT_TOP, "observer", FontId::proportional(11.0), OBSERVER);
        }
    }

    fn x_axis(painter: &egui::Painter, xm: &XMap, y: f32) {
        let col = Color32::from_gray(150);
        for v in ticks(xm.x0, xm.x1, ((xm.right - xm.left) / 90.0) as f64) {
            let x = xm.sx(v);
            painter.line_segment([pos2(x, y), pos2(x, y + 4.0)], Stroke::new(1.0, col));
            if x < xm.right - 40.0 {
                painter.text(pos2(x, y + 4.0), Align2::CENTER_TOP, fmt_tick(v), FontId::monospace(9.0), col);
            }
        }
        painter.text(pos2(xm.right, y + 4.0), Align2::RIGHT_TOP, "x / λ₀", FontId::proportional(10.0), col);
    }

    // ------------------------------------------------------------ top: the wave

    fn wave_ui(&mut self, ui: &mut egui::Ui) {
        let Some(s) = self.setup.clone() else { return };
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            ui.strong("WAVE");
            ui.weak("u(x, t) = Re ψ");
            ui.separator();
            if ui.button(if self.playing { "⏸ pause" } else { "▶ play" }).on_hover_text("space bar").clicked() {
                self.toggle_play();
            }
            if ui.button("⏮").on_hover_text("back to t = 0").clicked() {
                self.t = 0.0;
            }
            ui.spacing_mut().slider_width = (ui.available_width() * 0.4).clamp(120.0, 480.0);
            let r = ui.add(egui::Slider::new(&mut self.t, 0.0..=s.t_end).text("t / T₀").fixed_decimals(1));
            if r.dragged() {
                self.playing = false;
            }
            ui.spacing_mut().slider_width = 80.0;
            ui.add(egui::Slider::new(&mut self.params.speed, 0.1..=5.0).logarithmic(true).text("speed"));
            ui.checkbox(&mut self.params.looping, "loop");
        });
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let sw = strip_w(rect.width());
        let plot = Rect::from_min_max(pos2(rect.left() + ML, rect.top() + 4.0), pos2(rect.right() - sw - 10.0, rect.bottom() - AXIS_H));
        let strip = Rect::from_min_max(pos2(rect.right() - sw, rect.top() + 4.0), rect.max);
        if plot.width() < 20.0 || plot.height() < 20.0 {
            return;
        }
        let (x0, x1) = self.view(&s);
        let xm = XMap { left: plot.left(), right: plot.right(), x0, x1 };
        self.zoom_input(ui, &resp, &xm, &s);
        self.observer_input(ui, &resp, &xm, &s, None);
        let painter = ui.painter_at(rect);
        painter.rect_filled(plot, 3.0, Color32::from_gray(20));
        // vacuum and medium
        let x_in = xm.sx(0.0).clamp(plot.left(), plot.right());
        painter.rect_filled(Rect::from_min_max(pos2(x_in, plot.top()), plot.max), 3.0, Color32::from_rgb(24, 32, 44));
        if x0 < 0.0 {
            painter.text(plot.left_top() + vec2(4.0, 2.0), Align2::LEFT_TOP, "vacuum", FontId::proportional(11.0), Color32::from_gray(120));
            painter.line_segment([pos2(x_in, plot.top()), pos2(x_in, plot.bottom())], Stroke::new(1.0, Color32::from_rgb(90, 130, 180)));
        }
        if x1 > 0.0 {
            painter.text(pos2(x_in + 4.0, plot.top() + 2.0), Align2::LEFT_TOP, format!("medium: {}", s.key.medium.model.label()), FontId::proportional(11.0), Color32::from_gray(140));
        }
        Self::x_axis(&painter, &xm, plot.bottom());

        let picks: Vec<f32> = if self.params.show_picks { self.params.picks.clone() } else { vec![] };
        let main_h = if picks.is_empty() { plot.height() } else { plot.height() * 0.62 };
        let main = Rect::from_min_size(plot.min, vec2(plot.width(), main_h));
        let yc = main.center().y + 6.0;
        let amp = (main.height() * 0.5 - 14.0).max(4.0) / s.peak;
        painter.line_segment([pos2(plot.left(), yc), pos2(plot.right(), yc)], Stroke::new(1.0, Color32::from_white_alpha(25)));

        // the field
        let n = ((plot.width() * 3.0) as usize).clamp(600, 5000);
        self.field.resize(n, C32::new(0.0, 0.0));
        s.field(self.t, x0, x1, &mut self.field);
        let f = &self.field;
        let px = |i: usize| plot.left() + i as f32 / (n - 1) as f32 * plot.width();
        if self.params.show_envelope {
            for sign in [-1.0, 1.0] {
                let pts: Vec<Pos2> = (0..n).map(|i| pos2(px(i), yc - sign * f[i].norm() * amp)).collect();
                painter.add(Shape::line(pts, Stroke::new(1.0, Color32::from_gray(105))));
            }
        }
        if self.params.show_local_freq {
            let dx = (x1 - x0) / (n - 1) as f64;
            let k_med = s.key.medium.k(s.wc).re;
            let mut run: Vec<Pos2> = vec![];
            let mut run_col = Color32::GRAY;
            for i in 0..n {
                let p = pos2(px(i), yc - f[i].re * amp);
                let col = if i + 1 < n && f[i].norm() > 0.03 * s.peak {
                    let k = (f[i + 1] * f[i].conj()).arg() as f64 / dx;
                    let x = x0 + i as f64 * dx;
                    let kref = if x < 0.0 { s.wc } else { k_med };
                    if (k * dx).abs() < 1.2 { local_colour(k / kref) } else { Color32::GRAY }
                } else {
                    Color32::from_gray(150)
                };
                // quantise so that runs are long
                let col = Color32::from_rgb(col.r() & 0xF0, col.g() & 0xF0, col.b() & 0xF0);
                if col != run_col && !run.is_empty() {
                    run.push(p);
                    painter.add(Shape::line(std::mem::take(&mut run), Stroke::new(1.6, run_col)));
                }
                run_col = col;
                run.push(p);
            }
            painter.add(Shape::line(run, Stroke::new(1.6, run_col)));
        } else {
            let pts: Vec<Pos2> = (0..n).map(|i| pos2(px(i), yc - f[i].re * amp)).collect();
            painter.add(Shape::line(pts, Stroke::new(1.6, WAVE)));
        }
        let psi_at = |x: f64| -> C32 {
            let fi = ((x - x0) / (x1 - x0) * (n - 1) as f64).clamp(0.0, (n - 1) as f64);
            f[fi as usize]
        };
        // centre and length of the packet in view, for the marker dots
        let dxv = (x1 - x0) / (n - 1) as f64;
        let (mut wsum, mut xsum, mut x2sum, mut fmax) = (0.0, 0.0, 0.0, 0.0f32);
        for (i, v) in f.iter().enumerate() {
            let w = v.norm_sqr() as f64;
            let x = x0 + i as f64 * dxv;
            wsum += w;
            xsum += w * x;
            x2sum += w * x * x;
            fmax = fmax.max(v.norm());
        }
        let has_packet = fmax > 0.02 * s.peak;
        let xc = if wsum > 0.0 { xsum / wsum } else { 0.5 * (x0 + x1) };
        let xl = if wsum > 0.0 { (x2sum / wsum - xc * xc).max(0.0).sqrt() } else { 1.0 };
        // which crest each dot follows; start again for a new setup or when the time went back
        let nd = 1 + picks.len();
        if self.crests.0 != s.id || self.crests.1 != nd || self.t < self.crest_t {
            self.crests = (s.id, nd, vec![None; nd]);
        }
        self.crest_t = self.t;
        let t = self.t;
        let v_env = s.group_speed().unwrap_or(1.0);
        let m_near = |w: f64, phase0: f64, x: f64| -> f64 {
            let k = if x < 0.0 { w } else { s.key.medium.k(w).re };
            ((k * x - w * t + phase0) / TAU).round()
        };
        let visible = |x: f64| x >= x0 && x <= x1 && x >= s.x_min;
        // group and phase markers
        let carrier_pulse = matches!(s.key.pulse.shape, PulseShape::Gaussian | PulseShape::Rect);
        if self.params.show_guides && has_packet {
            if s.group_speed().is_some() {
                let xg = s.group_position(t);
                if visible(xg) {
                    let x = xm.sx(xg);
                    let b = main.bottom() - 2.0;
                    painter.add(Shape::convex_polygon(vec![pos2(x, b - 10.0), pos2(x - 6.0, b), pos2(x + 6.0, b)], GROUP, Stroke::NONE));
                    let name = if carrier_pulse { "v_g" } else { "Δω/Δk" };
                    painter.text(pos2(x + 8.0, b), Align2::LEFT_BOTTOM, name, FontId::proportional(11.0), GROUP);
                }
            }
            if carrier_pulse {
                // a crest of the carrier; when it has run out of the packet, take the next one coming in
                let (w, phase0) = (s.wc, s.wc * s.t_c);
                let inside = |x: f64| x >= x0 && x <= x1 && psi_at(x).norm() > 0.3 * fmax;
                let mut m = self.crests.2[0].unwrap_or_else(|| m_near(w, phase0, xc));
                if !inside(s.crest(w, phase0, m, t)) {
                    let dir = (s.v_phase(w) - v_env).signum();
                    m = m_near(w, phase0, xc - dir * 1.2 * xl);
                }
                self.crests.2[0] = Some(m);
                let xp = s.crest(w, phase0, m, t);
                if visible(xp) {
                    let y = yc - psi_at(xp).norm() * amp;
                    painter.circle(pos2(xm.sx(xp), y), 4.5, PHASE, Stroke::new(1.0, Color32::BLACK));
                    painter.text(pos2(xm.sx(xp) + 7.0, y - 3.0), Align2::LEFT_BOTTOM, "v_p", FontId::proportional(11.0), PHASE);
                }
            }
        }
        // frequency components, one lane each
        if !picks.is_empty() {
            let lanes = Rect::from_min_max(pos2(plot.left(), main.bottom()), plot.max);
            let lane_h = lanes.height() / picks.len() as f32;
            for (i, &pk) in picks.iter().enumerate() {
                let j = s.bin(pk as f64 * W0);
                let w = s.omega(j);
                let lane = Rect::from_min_size(pos2(lanes.left(), lanes.top() + i as f32 * lane_h), vec2(lanes.width(), lane_h));
                painter.line_segment([lane.left_top(), lane.right_top()], Stroke::new(1.0, Color32::from_white_alpha(18)));
                let ly = lane.center().y;
                let a = lane_h * 0.36;
                let norm = s.spec[j].norm();
                let weight = norm / s.max_spec.max(1e-30);
                let col = PICKS[i % PICKS.len()];
                let col = if weight > 0.01 { col } else { col.gamma_multiply(0.4) };
                if norm > 0.0 {
                    let m = plot.width().ceil() as usize;
                    let pts: Vec<Pos2> = (0..=m)
                        .map(|k| {
                            let sx = plot.left() + k as f32;
                            let x = xm.xs(sx);
                            let c = if x < s.x_min { C32::new(0.0, 0.0) } else { s.component(j, x, t) / norm };
                            pos2(sx, ly - c.re * a)
                        })
                        .collect();
                    painter.add(Shape::line(pts, Stroke::new(1.2, col)));
                    // a crest near the packet; it drifts with v_p − v_g and is replaced when it gets too far away
                    let phase0 = s.spec[j].arg() as f64;
                    let mut mm = self.crests.2[i + 1].unwrap_or_else(|| m_near(w, phase0, xc));
                    let reach = 1.5 * xl + 3.0;
                    if has_packet && (s.crest(w, phase0, mm, t) - xc).abs() > reach {
                        let dir = (s.v_phase(w) - v_env).signum();
                        mm = m_near(w, phase0, xc - dir * 0.9 * reach);
                    }
                    self.crests.2[i + 1] = Some(mm);
                    let xd = s.crest(w, phase0, mm, t);
                    if visible(xd) && self.params.show_guides {
                        painter.circle(pos2(xm.sx(xd), ly - a), 4.0, col, Stroke::new(1.0, Color32::BLACK));
                    }
                }
                tag(
                    &painter,
                    lane.left_top() + vec2(3.0, 2.0),
                    Align2::LEFT_TOP,
                    format!("ω = {:.2} ω₀   v_p = {:.2} c   weight {:.0} %", w / W0, s.v_phase(w), weight * 100.0),
                    col,
                );
            }
        }
        self.draw_observer(&painter, &xm, plot.top(), plot.bottom(), true);
        if let Some(p) = resp.hover_pos() {
            if plot.contains(p) && self.drag.is_none() {
                let x = xm.xs(p.x);
                readout(&painter, plot.right_top() + vec2(-4.0, 4.0), Align2::RIGHT_TOP, format!("x = {x:.1} λ₀\nu = {:+.3}", psi_at(x).re));
                if self.params.zoom != [0.0, 1.0] {
                    resp.on_hover_text("scroll to zoom, drag to move, double-click to see everything");
                } else {
                    resp.on_hover_text("scroll to zoom · drag the yellow line to move the observer");
                }
            }
        }
        self.wave_readouts(&painter, strip, &s);
    }

    fn wave_readouts(&self, painter: &egui::Painter, strip: Rect, s: &Setup) {
        painter.rect_filled(strip, 3.0, Color32::from_gray(22));
        let m = &s.key.medium;
        let wc = s.wc;
        let mut t = String::new();
        use std::fmt::Write;
        match s.key.pulse.shape {
            PulseShape::Delta => {
                let _ = writeln!(t, "delta pulse: all ω up\nto {:.1} ω₀\n", s.key.pulse.bandwidth);
            }
            PulseShape::Lines => {
                let _ = writeln!(t, "{} frequencies only", s.key.picks.len());
                if let Some(v) = s.group_speed() {
                    let _ = writeln!(t, "beats move with\n  Δω/Δk = {v:.3} c");
                }
                t.push('\n');
            }
            _ => {
                let k = m.k(wc);
                let _ = writeln!(t, "at the carrier ω = {:.2} ω₀", wc / W0);
                let _ = writeln!(t, "phase velocity");
                let _ = writeln!(t, "  v_p = ω/k   = {:.3} c", wc / k.re);
                let _ = writeln!(t, "group velocity");
                let _ = writeln!(t, "  v_g = dω/dk = {:.3} c", 1.0 / m.slowness(wc));
                let _ = writeln!(t, "n = {:.3}   n_g = {:.3}", k.re / wc, m.slowness(wc));
                let _ = writeln!(t, "k'' = {:+.3} T₀²/λ₀", m.k2(wc));
                if k.im > 1e-6 {
                    let _ = writeln!(t, "amplitude 1/e after {:.1} λ₀", 1.0 / k.im);
                }
                if s.key.pulse.shape == PulseShape::Gaussian && m.k2(wc).abs() > 1e-6 {
                    let sig = s.key.pulse.sigma as f64;
                    let _ = writeln!(t, "L_D = σ²/|k''| = {:.0} λ₀", sig * sig / m.k2(wc).abs());
                }
                t.push('\n');
            }
        }
        let _ = writeln!(t, "t = {:.1} T₀", self.t);
        // size of the packet inside the medium
        let (x0, x1) = self.view(s);
        let n = self.field.len();
        if n > 1 {
            let (mut a, mut b, mut c, mut top) = (0.0, 0.0, 0.0, 0.0f32);
            for (i, v) in self.field.iter().enumerate() {
                let x = x0 + (x1 - x0) * i as f64 / (n - 1) as f64;
                if x >= 0.0 {
                    let w = v.norm_sqr() as f64;
                    a += w;
                    b += w * x;
                    c += w * x * x;
                    top = top.max(v.norm());
                }
            }
            if a > 1e-9 * n as f64 && top > 1e-3 {
                let m = b / a;
                let _ = writeln!(t, "in the medium (in view):");
                let _ = writeln!(t, "  rms length {:.1} λ₀", (c / a - m * m).max(0.0).sqrt());
                let _ = writeln!(t, "  height {:.2} × input", top);
            }
        }
        painter.text(strip.left_top() + vec2(6.0, 6.0), Align2::LEFT_TOP, t, FontId::monospace(11.0), Color32::from_gray(210));
    }

    // ------------------------------------------------------------ middle: space–time

    fn observer_trace(&mut self, s: &Setup) -> &[C32] {
        let x = self.params.observer;
        if !self.trace.as_ref().is_some_and(|(id, xo, _)| *id == s.id && *xo == x) {
            let mut buf = s.spectrum_at(x as f64);
            self.planner.plan_fft_forward(s.n).process(&mut buf);
            self.trace = Some((s.id, x, buf));
        }
        &self.trace.as_ref().unwrap().2
    }

    fn listen(&mut self) {
        let Some(s) = self.setup.clone() else { return };
        // simulated time per second of sound: the whole run takes about 2.5 s
        let rate = (s.t_end / 2.5).clamp(150.0, 1200.0);
        let period = s.n as f64 * s.dt;
        let n = ((period * FS as f64 / rate) as usize).next_power_of_two().max(s.n);
        let rate = period * FS as f64 / n as f64;
        let mut buf = s.spectrum_at(self.params.observer as f64);
        buf.resize(n, C32::new(0.0, 0.0));
        self.planner.plan_fft_forward(n).process(&mut buf);
        let keep = ((s.t_end / rate * FS as f64) as usize).min(n);
        let peak = buf[..keep].iter().fold(0.0f32, |a, v| a.max(v.re.abs())).max(1e-9);
        let fade = (0.01 * FS as f64) as usize;
        let samples: Vec<f32> = (0..keep)
            .map(|i| {
                let e = (i.min(keep - 1 - i) as f32 / fade as f32).min(1.0);
                buf[i].re / peak * 0.8 * e
            })
            .collect();
        let id = audio::clip_id();
        self.audio.play(Arc::new(samples), FS, id, 0.0);
        self.listen = Some(Listen { id, rate });
        self.playing = false;
    }

    fn space_time_ui(&mut self, ui: &mut egui::Ui) {
        let Some(s) = self.setup.clone() else { return };
        let listening = self.listen.is_some();
        ui.horizontal(|ui| {
            ui.strong("SPACE–TIME");
            ui.weak(if self.params.xt_field { "colour = u(x, t)" } else { "brightness = |ψ(x, t)|" });
            ui.weak("· time runs down · click to jump in time");
            ui.selectable_value(&mut self.params.xt_field, false, "envelope");
            ui.selectable_value(&mut self.params.xt_field, true, "field");
            ui.checkbox(&mut self.params.xt_log, "log").on_hover_text("logarithmic brightness (40 dB): shows weak tails and precursors");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let label = if listening { "■ stop" } else { "▶ listen" };
                if ui
                    .button(label)
                    .on_hover_text("play the signal at the observer as sound (the whole run in about 2.5 s)")
                    .clicked()
                {
                    if listening {
                        self.audio.stop();
                        self.listen = None;
                    } else {
                        self.listen();
                    }
                }
                ui.strong("AT THE OBSERVER");
            });
        });
        if let Some(e) = &self.audio.error {
            ui.colored_label(Color32::from_rgb(230, 130, 90), e);
        }
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let sw = strip_w(rect.width());
        let plot = Rect::from_min_max(pos2(rect.left() + ML, rect.top() + 2.0), pos2(rect.right() - sw - 10.0, rect.bottom() - AXIS_H));
        let strip = Rect::from_min_max(pos2(rect.right() - sw, plot.top()), pos2(rect.right(), plot.bottom()));
        if plot.width() < 20.0 || plot.height() < 20.0 {
            return;
        }
        let (x0, x1) = self.view(&s);
        let xm = XMap { left: plot.left(), right: plot.right(), x0, x1 };
        self.zoom_input(ui, &resp, &xm, &s);
        self.observer_input(ui, &resp, &xm, &s, Some(plot));
        if let Some(w) = &mut self.xt {
            w.request(&XtRequest {
                setup: s.clone(),
                x0,
                x1,
                w: (plot.width() as usize).clamp(64, 1400),
                h: (plot.height() as usize).clamp(32, 900),
                field: self.params.xt_field,
                log: self.params.xt_log,
            });
        }
        let painter = ui.painter_at(rect);
        let ty = |t: f64| plot.top() + (t / s.t_end) as f32 * plot.height();
        {
            let clip = ui.painter_at(plot);
            clip.rect_filled(plot, 0.0, Color32::BLACK);
            if let (Some(tex), Some((_, ix0, ix1, it_end, _))) = (&self.xt_tex, self.xt_meta) {
                let r = Rect::from_min_max(pos2(xm.sx(ix0), ty(0.0)), pos2(xm.sx(ix1), ty(it_end)));
                clip.image(tex.id(), r, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
            }
            if s.x_min < 0.0 {
                let xi = xm.sx(0.0);
                clip.line_segment([pos2(xi, plot.top()), pos2(xi, plot.bottom())], Stroke::new(1.0, Color32::from_rgba_unmultiplied(120, 170, 230, 90)));
            }
            let carrier_pulse = matches!(s.key.pulse.shape, PulseShape::Gaussian | PulseShape::Rect);
            if self.params.show_guides && s.group_speed().is_some() {
                let p = |x: f64, t: f64| pos2(xm.sx(x), ty(t));
                if s.x_min < 0.0 {
                    clip.extend(Shape::dashed_line(&[p(s.x_min, s.t_c + s.x_min), p(0.0, s.t_c)], Stroke::new(1.2, Color32::from_gray(200)), 6.0, 5.0));
                }
                let group_name = if carrier_pulse { "v_g" } else { "Δω/Δk" };
                let mut lines = vec![(s.group_speed().unwrap_or(0.0), GROUP, group_name, true)];
                if carrier_pulse {
                    lines.push((s.v_phase(s.wc), PHASE, "v_p", false));
                }
                for (v, col, name, below) in lines {
                    if v.is_finite() && v > 1e-3 {
                        let t_end = s.t_c + s.x_max / v;
                        clip.extend(Shape::dashed_line(&[p(0.0, s.t_c), p(s.x_max, t_end)], Stroke::new(1.4, col), 8.0, 5.0));
                        let (xl, tl) = if t_end > s.t_end { (v * (s.t_end - s.t_c) * 0.9, s.t_c + (s.t_end - s.t_c) * 0.9) } else { (s.x_max * 0.9, s.t_c + s.x_max * 0.9 / v) };
                        let (off, align) = if below { (vec2(-4.0, 6.0), Align2::RIGHT_TOP) } else { (vec2(4.0, -6.0), Align2::LEFT_BOTTOM) };
                        clip.text(p(xl, tl) + off, align, name, FontId::proportional(12.0), col);
                    }
                }
            }
            let yt = ty(self.t);
            clip.line_segment([pos2(plot.left(), yt), pos2(plot.right(), yt)], Stroke::new(1.2, Color32::from_white_alpha(200)));
            self.draw_observer(&clip, &xm, plot.top(), plot.bottom(), false);
        }
        Self::x_axis(&painter, &xm, plot.bottom());
        let col = Color32::from_gray(150);
        for v in ticks(0.0, s.t_end, (plot.height() / 50.0) as f64) {
            let y = ty(v);
            painter.line_segment([pos2(plot.left() - 4.0, y), pos2(plot.left(), y)], Stroke::new(1.0, col));
            if y > plot.top() + 16.0 {
                painter.text(pos2(plot.left() - 6.0, y), Align2::RIGHT_CENTER, fmt_tick(v), FontId::monospace(9.0), col);
            }
        }
        painter.text(pos2(plot.left() - 6.0, plot.top()), Align2::RIGHT_TOP, "t / T₀", FontId::proportional(10.0), col);
        if let Some(p) = resp.hover_pos() {
            if plot.contains(p) {
                let t = ((p.y - plot.top()) / plot.height()) as f64 * s.t_end;
                readout(&painter, plot.right_top() + vec2(-4.0, 4.0), Align2::RIGHT_TOP, format!("x = {:.1} λ₀\nt = {t:.1} T₀", xm.xs(p.x)));
            }
        }
        if let Some((_, _, _, _, ms)) = self.xt_meta {
            painter.text(plot.left_bottom() + vec2(4.0, -3.0), Align2::LEFT_BOTTOM, format!("{ms:.0} ms"), FontId::monospace(9.0), Color32::from_gray(90));
        }
        self.trace_strip(&painter, strip, &s, &ty);
    }

    /// the signal at the observer, with time running down like in the space–time view
    fn trace_strip(&mut self, painter: &egui::Painter, strip: Rect, s: &Setup, ty: &dyn Fn(f64) -> f32) {
        painter.rect_filled(strip, 3.0, Color32::from_gray(22));
        let cx = strip.center().x;
        let a = strip.width() * 0.44 / s.peak;
        painter.line_segment([pos2(cx, strip.top()), pos2(cx, strip.bottom())], Stroke::new(1.0, Color32::from_white_alpha(25)));
        let dt = s.dt;
        let t_end = s.t_end;
        let tr = self.observer_trace(s).to_vec();
        let rows = strip.height().max(1.0) as usize;
        let mut wave = Vec::with_capacity(2 * rows);
        let mut env_l = Vec::with_capacity(rows);
        let mut env_r = Vec::with_capacity(rows);
        for r in 0..rows {
            let ta = r as f64 / rows as f64 * t_end;
            let tb = (r + 1) as f64 / rows as f64 * t_end;
            let ia = (ta / dt) as usize;
            let ib = ((tb / dt) as usize).max(ia).min(tr.len() - 1);
            let (mut lo, mut hi, mut e) = (f32::MAX, f32::MIN, 0.0f32);
            for v in &tr[ia..=ib] {
                lo = lo.min(v.re);
                hi = hi.max(v.re);
                e = e.max(v.norm());
            }
            let y = ty(0.5 * (ta + tb));
            wave.push(pos2(cx + lo * a, y));
            wave.push(pos2(cx + hi * a, y));
            env_l.push(pos2(cx - e * a, y));
            env_r.push(pos2(cx + e * a, y));
        }
        if self.params.show_envelope {
            painter.add(Shape::line(env_l, Stroke::new(1.0, Color32::from_gray(100))));
            painter.add(Shape::line(env_r, Stroke::new(1.0, Color32::from_gray(100))));
        }
        painter.add(Shape::line(wave, Stroke::new(1.0, WAVE)));
        let yt = ty(self.t);
        painter.line_segment([pos2(strip.left(), yt), pos2(strip.right(), yt)], Stroke::new(1.2, Color32::from_white_alpha(200)));
        painter.text(strip.left_top() + vec2(4.0, 2.0), Align2::LEFT_TOP, format!("u(t) at x = {:.1} λ₀", self.params.observer), FontId::proportional(10.0), OBSERVER);
    }

    // ------------------------------------------------------------ bottom: dispersion relation and controls

    fn bottom_ui(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        if !self.notes.trim().is_empty() {
            ui.horizontal_wrapped(|ui| {
                ui.strong("About this setup:");
                ui.label(&self.notes);
            });
            ui.separator();
        }
        let avail = ui.available_rect_before_wrap();
        let plot_w = (avail.width() * 0.4).clamp(300.0, 760.0);
        let left = Rect::from_min_size(avail.min, vec2(plot_w, avail.height()));
        let right = Rect::from_min_max(pos2(avail.left() + plot_w + 14.0, avail.top()), avail.max);
        ui.painter().line_segment(
            [pos2(avail.left() + plot_w + 7.0, avail.top()), pos2(avail.left() + plot_w + 7.0, avail.bottom())],
            ui.visuals().widgets.noninteractive.bg_stroke,
        );
        ui.scope_builder(egui::UiBuilder::new().max_rect(left), |ui| self.dispersion_plot(ui));
        ui.scope_builder(egui::UiBuilder::new().max_rect(right), |ui| {
            egui::ScrollArea::vertical().id_salt("disp_controls").auto_shrink([false, false]).show(ui, |ui| {
                ui.columns(3, |cols| {
                    self.pulse_controls(&mut cols[0]);
                    self.medium_controls(&mut cols[1]);
                    self.show_controls(&mut cols[2]);
                });
            });
        });
    }

    fn u_max(&self, s: &Setup) -> f64 {
        let pk = self.params.picks.iter().cloned().fold(0.0f32, f32::max) as f64;
        (s.band.1 / W0 * 1.05).max(3.0).max(pk + 0.2).min(6.0)
    }

    fn handles(&self, s: &Setup) -> Vec<(Handle, f64, f64)> {
        let m = &s.key.medium;
        let n_at = |u: f64| m.n(u * W0).re;
        match m.model {
            Model::Linear => vec![(Handle::N0, s.wc / W0, m.n0 as f64)],
            Model::Taylor => vec![
                (Handle::Np, 1.0, m.n_p as f64),
                (Handle::Ng, 0.75, m.slowness(0.75 * W0)),
                (Handle::Gvd, 1.5, m.slowness(1.5 * W0)),
            ],
            Model::Glass => vec![(Handle::GlassN, 1.0, m.glass_n as f64), (Handle::GlassB, 2.0, n_at(2.0))],
            Model::Resonance => vec![(Handle::ResW, m.res_w as f64, n_at(m.res_w as f64))],
            Model::Cutoff => vec![(Handle::Cutoff, m.cutoff as f64, 0.0)],
            Model::PowerLaw => vec![(Handle::PowerN, 1.0, m.power_n as f64)],
            Model::Free => m.free.iter().enumerate().map(|(i, v)| (Handle::Free(i), i as f64 * FREE_STEP, *v as f64)).collect(),
        }
    }

    fn apply_handle(&mut self, h: Handle, u: f64, y: f64) {
        let m = &mut self.params.medium;
        let y = y as f32;
        match h {
            Handle::N0 => m.n0 = y.clamp(0.2, 5.0),
            Handle::Np => m.n_p = y.clamp(0.2, 5.0),
            // n_g(ω) = n_g + GVD d + TOD d²/2 with d = ω/ω₀ − 1
            Handle::Ng => m.n_g = (y + 0.25 * m.gvd - m.tod / 32.0).clamp(-2.0, 6.0),
            Handle::Gvd => m.gvd = ((y - m.n_g - m.tod / 8.0) / 0.5).clamp(-6.0, 6.0),
            Handle::GlassN => m.glass_n = y.clamp(0.5, 4.0),
            Handle::GlassB => m.glass_b = ((y - m.glass_n) / 3.0).clamp(-0.3, 0.5),
            Handle::ResW => m.res_w = (u as f32).clamp(0.2, 4.0),
            Handle::Cutoff => m.cutoff = (u as f32).clamp(0.02, 3.0),
            Handle::PowerN => m.power_n = y.clamp(0.2, 5.0),
            Handle::Free(i) => {
                if let Some(v) = m.free.get_mut(i) {
                    *v = y.clamp(-1.0, 6.0);
                }
            }
        }
    }

    fn toggle_pick(&mut self, u: f64, u_max: f64) {
        if u <= 0.02 || u > u_max {
            return;
        }
        let tol = 0.025 * u_max;
        if let Some(i) = self.params.picks.iter().position(|p| (*p as f64 - u).abs() < tol) {
            self.params.picks.remove(i);
        } else if self.params.picks.len() < MAX_PICKS {
            self.params.picks.push(((u * 100.0).round() / 100.0) as f32);
            self.params.show_picks = true;
        }
    }

    fn dispersion_plot(&mut self, ui: &mut egui::Ui) {
        let Some(s) = self.setup.clone() else { return };
        ui.horizontal(|ui| {
            ui.strong("DISPERSION RELATION");
            ui.selectable_value(&mut self.params.plot, PlotView::Index, "n(ω)").on_hover_text("refractive index, group index and absorption");
            ui.selectable_value(&mut self.params.plot, PlotView::OmegaK, "ω(k)").on_hover_text("frequency against wavenumber");
            ui.selectable_value(&mut self.params.plot, PlotView::Velocity, "v(ω)").on_hover_text("phase and group velocity");
        });
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let plot = Rect::from_min_max(rect.min + vec2(44.0, 14.0), rect.max - vec2(8.0, 24.0));
        if plot.width() < 40.0 || plot.height() < 40.0 {
            return;
        }
        let painter = ui.painter_at(rect);
        let u_max = self.u_max(&s);
        let m = s.key.medium.clone();
        let us: Vec<f64> = (1..=500).map(|i| u_max * i as f64 / 500.0).collect();
        let absorbing = us.iter().any(|&u| m.n(u * W0).im > 1e-3);
        match self.params.plot {
            PlotView::Index => {
                let n: Vec<(f64, f64)> = us.iter().map(|&u| (u, m.n(u * W0).re)).collect();
                let ng: Vec<(f64, f64)> = us.iter().map(|&u| (u, m.slowness(u * W0))).collect();
                let kappa: Vec<(f64, f64)> = us.iter().map(|&u| (u, m.n(u * W0).im)).collect();
                let y = self.frozen_y.unwrap_or_else(|| {
                    let mut lo: f64 = 1.0;
                    let mut hi: f64 = 1.0;
                    let mut take = |v: f64| {
                        if v.is_finite() {
                            lo = lo.min(v);
                            hi = hi.max(v);
                        }
                    };
                    for i in 0..us.len() {
                        if us[i] > 0.12 * u_max {
                            take(n[i].1);
                            take(ng[i].1);
                            if absorbing {
                                take(kappa[i].1);
                            }
                        }
                    }
                    for h in self.handles(&s) {
                        take(h.2);
                    }
                    let (lo, hi) = (lo.max(-2.0), hi.min(6.0));
                    let pad = ((hi - lo) * 0.12).max(0.15);
                    (lo - pad, hi + pad)
                });
                let ax = Axes::new(plot, (0.0, u_max), y);
                ax.frame(&painter, "ω / ω₀", "n");
                self.spectrum_fill(&painter, &ax, &s, false);
                painter.line_segment([ax.p(0.0, 1.0), ax.p(u_max, 1.0)], Stroke::new(1.0, Color32::from_white_alpha(40)));
                if absorbing {
                    ax.curve(&painter, &kappa, Stroke::new(1.5, ABSORB), false);
                }
                ax.curve(&painter, &ng, Stroke::new(1.5, GROUP_INDEX), true);
                ax.curve(&painter, &n, Stroke::new(2.0, INDEX), false);
                self.carrier_and_picks(&painter, &ax, &s, |u| m.n(u * W0).re);
                let mut legend = vec![("n = c k/ω  (phase index)", INDEX), ("n_g = c dk/dω  (group index)", GROUP_INDEX)];
                if absorbing {
                    legend.push(("κ = Im n  (absorption)", ABSORB));
                }
                legend_box(&painter, plot, &legend);
                // handles
                let handles = self.handles(&s);
                let hover = resp.hover_pos();
                let near = |p: Pos2| handles.iter().map(|h| (h.0, ax.p(h.1, h.2).distance(p))).filter(|h| h.1 < 10.0).min_by(|a, b| a.1.total_cmp(&b.1)).map(|h| h.0);
                for &(h, u, v) in &handles {
                    let hot = hover.and_then(near) == Some(h) || self.drag == Some(Drag::Handle(h));
                    painter.circle(ax.p(u, v), if hot { 7.0 } else { 5.0 }, Color32::WHITE, Stroke::new(1.5, Color32::BLACK));
                }
                if resp.drag_started() {
                    let o = ui.input(|i| i.pointer.press_origin()).unwrap_or_default();
                    if let Some(h) = near(o) {
                        self.drag = Some(Drag::Handle(h));
                        self.frozen_y = Some(y);
                    }
                }
                if let (Some(Drag::Handle(h)), true) = (self.drag, resp.dragged()) {
                    if let Some(p) = resp.interact_pointer_pos() {
                        let (u, v) = ax.inv(p);
                        self.apply_handle(h, u, v);
                    }
                }
                if resp.drag_stopped() {
                    self.drag = None;
                    self.frozen_y = None;
                }
                if hover.and_then(near).is_some() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                }
                if resp.clicked() {
                    if let Some(p) = resp.interact_pointer_pos() {
                        if near(p).is_none() {
                            self.toggle_pick(ax.inv(p).0, u_max);
                        }
                    }
                }
                if let Some(p) = hover {
                    if plot.contains(p) && self.drag.is_none() {
                        self.hover_readout(&painter, &ax, &m, ax.inv(p).0, plot);
                    }
                }
                resp.on_hover_text("drag the white points to change n(ω) · click to pick a frequency component");
            }
            PlotView::OmegaK => {
                let pts: Vec<(f64, f64)> = us.iter().map(|&u| (m.k(u * W0).re / W0, u)).collect();
                let im: Vec<(f64, f64)> = us.iter().map(|&u| (m.k(u * W0).im / W0, u)).collect();
                let k_max = pts.iter().map(|p| p.0).filter(|v| v.is_finite()).fold(u_max, f64::max).min(8.0 * u_max);
                let ax = Axes::new(plot, (0.0, k_max * 1.05), (0.0, u_max));
                ax.frame(&painter, "k / k₀   (k₀ = ω₀/c)", "ω / ω₀");
                self.spectrum_fill(&painter, &ax, &s, true);
                ax.curve(&painter, &[(0.0, 0.0), (u_max, u_max)], Stroke::new(1.0, Color32::from_gray(110)), true);
                painter.text(ax.p(u_max.min(k_max) * 0.9, u_max.min(k_max) * 0.9), Align2::RIGHT_BOTTOM, "vacuum ω = ck ", FontId::proportional(10.0), Color32::from_gray(130));
                if absorbing {
                    ax.curve(&painter, &im, Stroke::new(1.5, ABSORB), false);
                }
                ax.curve(&painter, &pts, Stroke::new(2.0, INDEX), false);
                let uc = s.wc / W0;
                let kc = m.k(s.wc).re / W0;
                if matches!(s.key.pulse.shape, PulseShape::Gaussian | PulseShape::Rect) && kc > 0.0 {
                    // secant: slope ω/k = v_p; tangent: slope dω/dk = v_g
                    let far = 2.0 * k_max;
                    ax.curve(&painter, &[(0.0, 0.0), (far, uc / kc * far)], Stroke::new(1.4, PHASE), true);
                    let vg = 1.0 / m.slowness(s.wc);
                    ax.curve(&painter, &[(kc - far, uc - vg * far), (kc + far, uc + vg * far)], Stroke::new(1.4, GROUP), true);
                    painter.circle(ax.p(kc, uc), 5.0, PHASE, Stroke::new(1.0, Color32::BLACK));
                    legend_box(&painter, plot, &[("secant: v_p = ω/k", PHASE), ("tangent: v_g = dω/dk", GROUP)]);
                }
                for (i, &pk) in self.params.picks.iter().enumerate() {
                    let u = pk as f64;
                    let col = PICKS[i % PICKS.len()];
                    painter.line_segment([ax.p(0.0, u), ax.p(k_max * 1.05, u)], Stroke::new(1.0, col.gamma_multiply(0.6)));
                    painter.circle_filled(ax.p(m.k(u * W0).re / W0, u), 4.0, col);
                }
                if resp.clicked() {
                    if let Some(p) = resp.interact_pointer_pos() {
                        self.toggle_pick(ax.inv(p).1, u_max);
                    }
                }
                resp.on_hover_text("click to pick a frequency component");
            }
            PlotView::Velocity => {
                let vp: Vec<(f64, f64)> = us.iter().map(|&u| (u, u * W0 / m.k(u * W0).re)).collect();
                let vg: Vec<(f64, f64)> = us.iter().map(|&u| (u, 1.0 / m.slowness(u * W0))).collect();
                let mut lo: f64 = 0.0;
                let mut hi: f64 = 1.2;
                for (i, &u) in us.iter().enumerate() {
                    if u > 0.12 * u_max {
                        for v in [vp[i].1, vg[i].1] {
                            if v.is_finite() {
                                lo = lo.min(v);
                                hi = hi.max(v);
                            }
                        }
                    }
                }
                let (lo, hi) = (lo.max(-2.0), hi.min(4.0));
                let ax = Axes::new(plot, (0.0, u_max), (lo - 0.05 * (hi - lo), hi + 0.08 * (hi - lo)));
                ax.frame(&painter, "ω / ω₀", "v / c");
                self.spectrum_fill(&painter, &ax, &s, false);
                painter.line_segment([ax.p(0.0, 1.0), ax.p(u_max, 1.0)], Stroke::new(1.0, Color32::from_white_alpha(60)));
                painter.text(ax.p(u_max, 1.0), Align2::RIGHT_BOTTOM, "c ", FontId::proportional(10.0), Color32::from_gray(150));
                ax.curve(&painter, &vp, Stroke::new(2.0, PHASE), false);
                ax.curve(&painter, &vg, Stroke::new(2.0, GROUP), false);
                self.carrier_and_picks(&painter, &ax, &s, |u| u * W0 / m.k(u * W0).re);
                legend_box(&painter, plot, &[("phase velocity v_p = ω/k", PHASE), ("group velocity v_g = dω/dk", GROUP)]);
                if resp.clicked() {
                    if let Some(p) = resp.interact_pointer_pos() {
                        self.toggle_pick(ax.inv(p).0, u_max);
                    }
                }
                if let Some(p) = resp.hover_pos() {
                    if plot.contains(p) {
                        self.hover_readout(&painter, &ax, &m, ax.inv(p).0, plot);
                    }
                }
                resp.on_hover_text("click to pick a frequency component");
            }
        }
    }

    fn hover_readout(&self, painter: &egui::Painter, ax: &Axes, m: &crate::dispersion::Medium, u: f64, plot: Rect) {
        if u <= 0.0 {
            return;
        }
        let w = u * W0;
        let k = m.k(w);
        let x = ax.p(u, ax.y0).x;
        painter.line_segment([pos2(x, plot.top()), pos2(x, plot.bottom())], Stroke::new(1.0, Color32::from_white_alpha(60)));
        let mut t = format!(
            "ω = {u:.2} ω₀\nn = {:.3}  n_g = {:.3}\nv_p = {:.3} c\nv_g = {:.3} c",
            k.re / w,
            m.slowness(w),
            w / k.re,
            1.0 / m.slowness(w)
        );
        if k.im > 1e-6 {
            t += &format!("\nκ = {:.3}", k.im / w);
        }
        readout(painter, plot.right_top() + vec2(-4.0, 4.0), Align2::RIGHT_TOP, t);
    }

    /// |S(ω)|² of the pulse (filled) and after the medium (outline)
    fn spectrum_fill(&self, painter: &egui::Painter, ax: &Axes, s: &Setup, vertical: bool) {
        let cols = if vertical { ax.r.height() } else { ax.r.width() } as usize;
        let u_max = if vertical { ax.y1 } else { ax.x1 };
        let max2 = (s.max_spec as f64).powi(2).max(1e-60);
        let mut v_in = vec![0.0f64; cols];
        let mut v_out = vec![0.0f64; cols];
        for &j in &s.sig {
            let u = s.omega(j) / W0;
            let c = ((u / u_max) * cols as f64) as usize;
            if c < cols {
                let a = s.spec[j].norm_sqr() as f64 / max2;
                v_in[c] = v_in[c].max(a);
                v_out[c] = v_out[c].max(a * (-2.0 * s.k[j].im * s.x_max).exp());
            }
        }
        let size = 0.3 * if vertical { ax.r.width() } else { ax.r.height() };
        let fill = Color32::from_rgba_unmultiplied(110, 170, 255, 45);
        let pos = |c: usize, v: f64| -> (Pos2, Pos2) {
            if vertical {
                let y = ax.r.bottom() - (c as f32 + 0.5);
                (pos2(ax.r.left(), y), pos2(ax.r.left() + v as f32 * size, y))
            } else {
                let x = ax.r.left() + c as f32 + 0.5;
                (pos2(x, ax.r.bottom()), pos2(x, ax.r.bottom() - v as f32 * size))
            }
        };
        for (c, v) in v_in.iter().enumerate() {
            if *v > 1e-4 {
                let (a, b) = pos(c, *v);
                painter.line_segment([a, b], Stroke::new(1.0, fill));
            }
        }
        if s.key.medium.is_absorbing() {
            let pts: Vec<Pos2> = v_out.iter().enumerate().map(|(c, v)| pos(c, *v).1).collect();
            painter.add(Shape::line(pts, Stroke::new(1.0, Color32::from_rgba_unmultiplied(240, 120, 110, 160))));
        }
        let label = if s.key.medium.is_absorbing() { "spectrum of the pulse (red: after the medium)" } else { "spectrum of the pulse" };
        let at = if vertical { ax.r.left_bottom() + vec2(4.0, -3.0) } else { ax.r.left_bottom() + vec2(4.0, -3.0) };
        painter.text(at, Align2::LEFT_BOTTOM, label, FontId::proportional(10.0), Color32::from_rgb(140, 180, 240));
    }

    fn carrier_and_picks(&self, painter: &egui::Painter, ax: &Axes, s: &Setup, y_of: impl Fn(f64) -> f64) {
        if matches!(s.key.pulse.shape, PulseShape::Gaussian | PulseShape::Rect) {
            let x = ax.p(s.wc / W0, 0.0).x;
            dashed_v(painter, x, ax.r.top(), ax.r.bottom(), Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 210, 120, 120)));
            painter.text(pos2(x + 3.0, ax.r.top() + 2.0), Align2::LEFT_TOP, "carrier", FontId::proportional(10.0), Color32::from_rgb(230, 190, 120));
        }
        for (i, &pk) in self.params.picks.iter().enumerate() {
            let u = pk as f64;
            let col = PICKS[i % PICKS.len()];
            let x = ax.p(u, 0.0).x;
            painter.line_segment([pos2(x, ax.r.top()), pos2(x, ax.r.bottom())], Stroke::new(1.0, col.gamma_multiply(0.6)));
            let y = y_of(u);
            if y.is_finite() {
                painter.circle_filled(ax.p(u, y), 4.0, col);
            }
        }
    }

    fn pulse_controls(&mut self, ui: &mut egui::Ui) {
        let p = &mut self.params.pulse;
        ui.strong("PULSE");
        ui.weak("the signal that arrives at x = 0");
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut p.shape, PulseShape::Gaussian, "Gaussian");
            ui.selectable_value(&mut p.shape, PulseShape::Delta, "delta");
            ui.selectable_value(&mut p.shape, PulseShape::Rect, "switched on/off");
            ui.selectable_value(&mut p.shape, PulseShape::Lines, "picked frequencies");
        });
        if p.shape == PulseShape::Lines && self.params.picks.is_empty() {
            self.params.picks = vec![0.9, 1.1];
        }
        egui::Grid::new("pulse").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
            if matches!(p.shape, PulseShape::Gaussian | PulseShape::Rect) {
                ui.label("carrier ω_c");
                ui.add(egui::Slider::new(&mut p.carrier, 0.3..=3.0).suffix(" ω₀").fixed_decimals(2));
                ui.end_row();
            }
            match p.shape {
                PulseShape::Gaussian => {
                    ui.label("duration σ");
                    ui.add(egui::Slider::new(&mut p.sigma, 0.3..=40.0).logarithmic(true).suffix(" T₀"))
                        .on_hover_text("rms duration of the envelope exp(−t²/2σ²); the spectrum has width 1/σ");
                    ui.end_row();
                    ui.label("");
                    ui.horizontal(|ui| {
                        if ui.button("short").on_hover_text("a few cycles: a wide spectrum").clicked() {
                            p.sigma = 1.0;
                        }
                        if ui.button("long: nearly one frequency").clicked() {
                            p.sigma = 25.0;
                        }
                    });
                    ui.end_row();
                    ui.label("chirp C");
                    ui.add(egui::Slider::new(&mut p.chirp, -8.0..=8.0).fixed_decimals(1))
                        .on_hover_text("exp(−(1 + iC) t²/2σ²): C > 0 the frequency rises during the pulse, C < 0 it falls");
                    ui.end_row();
                }
                PulseShape::Rect => {
                    ui.label("duration");
                    ui.add(egui::Slider::new(&mut p.length, 2.0..=150.0).logarithmic(true).suffix(" T₀"));
                    ui.end_row();
                }
                PulseShape::Delta => {
                    ui.label("up to");
                    ui.add(egui::Slider::new(&mut p.bandwidth, 0.5..=5.0).suffix(" ω₀"))
                        .on_hover_text("a delta pulse contains all frequencies; here those up to this one");
                    ui.end_row();
                }
                PulseShape::Lines => {
                    ui.label("frequencies");
                    ui.horizontal(|ui| {
                        for (n, label) in [(2, "2"), (3, "3"), (5, "5")] {
                            if ui.button(label).on_hover_text(format!("{n} frequencies around the carrier")).clicked() {
                                let c = p.carrier;
                                self.params.picks = (0..n).map(|i| c + 0.1 * (i as f32 - (n - 1) as f32 / 2.0)).collect();
                            }
                        }
                    });
                    ui.end_row();
                }
            }
        });
        if p.shape == PulseShape::Lines {
            ui.weak("Click in the dispersion plot to add or remove frequencies.");
        }
    }

    fn medium_controls(&mut self, ui: &mut egui::Ui) {
        let wc = self.params.pulse.carrier as f64 * W0;
        let m = &mut self.params.medium;
        ui.strong("MEDIUM");
        egui::ComboBox::from_id_salt("disp_model").width(240.0).selected_text(m.model.label()).show_ui(ui, |ui| {
            for md in Model::ALL {
                ui.selectable_value(&mut m.model, md, md.label());
            }
        });
        egui::Grid::new("medium").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
            let row = |ui: &mut egui::Ui, label: &str, v: &mut f32, range: std::ops::RangeInclusive<f32>, hint: &str| {
                ui.label(label);
                let r = ui.add(egui::Slider::new(v, range));
                if !hint.is_empty() {
                    r.on_hover_text(hint);
                }
                ui.end_row();
            };
            match m.model {
                Model::Linear => row(ui, "n", &mut m.n0, 0.3..=4.0, "the same for all frequencies"),
                Model::Taylor => {
                    row(ui, "phase index n", &mut m.n_p, 0.3..=3.0, "n = ck/ω at ω₀: v_p = c/n");
                    row(ui, "group index n_g", &mut m.n_g, 0.2..=4.0, "n_g = c dk/dω at ω₀: v_g = c/n_g");
                    row(ui, "GVD", &mut m.gvd, -3.0..=3.0, "c ω₀ d²k/dω²: how fast n_g changes with ω (spreading)");
                    row(ui, "TOD", &mut m.tod, -6.0..=6.0, "third order: makes the spreading asymmetric");
                }
                Model::Glass => {
                    row(ui, "n at ω₀", &mut m.glass_n, 1.0..=3.0, "");
                    row(ui, "B", &mut m.glass_b, -0.2..=0.3, "n = A + B(ω/ω₀)²; real glass: B ≈ 0.005 (here exaggerated)");
                }
                Model::Resonance => {
                    row(ui, "resonance ω_r", &mut m.res_w, 0.3..=3.0, "in units of ω₀");
                    row(ui, "strength f", &mut m.res_f, 0.0..=0.6, "ε = 1 + f ω_r²/(ω_r² − ω² − iγω)");
                    ui.label("damping γ");
                    ui.add(egui::Slider::new(&mut m.res_gamma, 0.002..=0.5).logarithmic(true)).on_hover_text("width of the absorption line (ω₀)");
                    ui.end_row();
                }
                Model::Cutoff => row(ui, "cutoff ω_c", &mut m.cutoff, 0.05..=2.5, "n² = 1 − ω_c²/ω²; below ω_c no wave propagates"),
                Model::PowerLaw => {
                    row(ui, "exponent m", &mut m.power_m, 0.3..=3.0, "ω ~ k^m, so v_g = m v_p");
                    row(ui, "phase index at ω₀", &mut m.power_n, 0.3..=3.0, "");
                }
                Model::Free => {}
            }
        });
        match m.model {
            Model::PowerLaw => {
                ui.horizontal_wrapped(|ui| {
                    ui.label("m for");
                    for (v, name, hint) in [
                        (0.5, "deep water", "ω = √(gk)"),
                        (1.0, "linear", "no dispersion"),
                        (1.5, "ripples", "capillary waves ω² = σk³/ρ"),
                        (2.0, "matter wave", "ħω = ħ²k²/2m"),
                    ] {
                        if ui.small_button(name).on_hover_text(hint).clicked() {
                            m.power_m = v;
                        }
                    }
                });
            }
            Model::Free => {
                ui.horizontal_wrapped(|ui| {
                    ui.label("start from");
                    if ui.small_button("flat").clicked() {
                        m.free = vec![1.5; FREE_POINTS];
                    }
                    if ui.small_button("normal").clicked() {
                        m.free = default_free();
                    }
                    if ui.small_button("anomalous").clicked() {
                        m.free = (0..FREE_POINTS).map(|i| 1.7 - 0.03 * (i as f32 * FREE_STEP as f32).powi(2)).collect();
                    }
                    if ui.small_button("bumpy").clicked() {
                        m.free = (0..FREE_POINTS).map(|i| 1.4 + 0.12 * ((i as f32) * 1.3).sin()).collect();
                    }
                });
                ui.weak("drag the white points in the n(ω) plot");
            }
            _ => {}
        }
        if ui
            .button("make it linear (no dispersion)")
            .on_hover_text("replace the medium by a constant n, the one it has at the carrier")
            .clicked()
        {
            let n = m.n(wc).re;
            m.n0 = if n.is_finite() { (n as f32).clamp(0.3, 4.0) } else { 1.0 };
            m.model = Model::Linear;
        }
        egui::Grid::new("medium2").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
            ui.label("length L");
            ui.add(egui::Slider::new(&mut self.params.length, 10.0..=1000.0).logarithmic(true).suffix(" λ₀"));
            ui.end_row();
        });
        ui.checkbox(&mut self.params.lead_in, "vacuum in front (x < 0)")
            .on_hover_text("the pulse comes from vacuum; off: it is sent into the medium at x = 0 (e.g. by a wave maker)");
    }

    fn show_controls(&mut self, ui: &mut egui::Ui) {
        let p = &mut self.params;
        ui.strong("SHOW");
        ui.checkbox(&mut p.show_envelope, "envelope |ψ|");
        ui.checkbox(&mut p.show_local_freq, "colour = local wavelength").on_hover_text("red: longer waves, blue: shorter waves than at the carrier");
        ui.checkbox(&mut p.show_guides, "v_p and v_g markers and lines");
        ui.checkbox(&mut p.show_picks, "picked frequency components").on_hover_text("each one is a plane wave moving with its own phase velocity");
        ui.horizontal_wrapped(|ui| {
            for (i, pk) in p.picks.clone().iter().enumerate() {
                if ui.small_button(egui::RichText::new(format!("{pk:.2} ×")).color(PICKS[i % PICKS.len()])).on_hover_text("remove").clicked() {
                    p.picks.remove(i);
                    break;
                }
            }
        });
        ui.horizontal_wrapped(|ui| {
            if ui.small_button("pick 3 across the spectrum").clicked() {
                if let Some(s) = &self.setup {
                    let (lo, hi) = spectrum_range(s);
                    p.picks = (0..3).map(|i| ((lo + (hi - lo) * (0.2 + 0.3 * i as f64)) * 100.0).round() as f32 / 100.0).collect();
                    p.show_picks = true;
                }
            }
            if ui.small_button("clear").clicked() {
                p.picks.clear();
            }
        });
        egui::Grid::new("show").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
            if let Some(s) = &self.setup {
                ui.label("observer at");
                ui.add(egui::Slider::new(&mut p.observer, s.x_min as f32..=s.x_max as f32).suffix(" λ₀").fixed_decimals(1));
                ui.end_row();
            }
        });
        ui.add_space(4.0);
        ui.weak("Click in the dispersion plot to pick frequencies, drag its white points to change the medium. Scroll in the wave or space–time view to zoom. Space: play/pause.");
    }
}

/// ω/ω₀ where the spectrum of the pulse is above 5 % of its peak
fn spectrum_range(s: &Setup) -> (f64, f64) {
    let lim = 0.05 * s.max_spec;
    let js: Vec<usize> = s.sig.iter().cloned().filter(|&j| s.spec[j].norm() > lim).collect();
    match (js.first(), js.last()) {
        (Some(&a), Some(&b)) => (s.omega(a) / W0, s.omega(b) / W0),
        _ => (0.9, 1.1),
    }
}
