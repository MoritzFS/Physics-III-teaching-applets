//! User interface of the interferometer bench: the optical table (centre),
//! the spectrum of a sweep and the switch-on in time (right), and the
//! toolbox, the selected part, the laser and the sweep (bottom).

use std::f32::consts::{FRAC_1_SQRT_2, TAU};

use eframe::egui::{self, pos2, vec2, Align2, Color32, FontId, Pos2, Rect, Sense, Shape, Stroke, StrokeKind, Vec2};
use rustfft::num_complex::Complex64 as C64;

use crate::dispersion_ui::{fmt_tick, readout, ticks};
use crate::fourier::wavelength_rgb;
use crate::interferometer::{
    describe_polarisation, ellipse, fmt_percent, peaks, power, steady, sweep, Dir, IfoParams, IfoPreset, Jones, Kind,
    Part, Spectrum, Steady, SweepResult, SweepVar, TimeSim, BACK, BAR, C_LIGHT, DASH, HISTORY_MAX, SLASH,
};
use crate::worker::Worker;

const BG: Color32 = Color32::from_gray(16);
const TABLE: Color32 = Color32::from_rgb(30, 32, 36);
const HOLE: Color32 = Color32::from_rgb(52, 55, 60);
const LABEL: Color32 = Color32::from_gray(170);
const PICK: Color32 = Color32::from_rgb(255, 220, 80);
const SELECT: Color32 = Color32::from_rgb(255, 220, 80);
const SILVER: Color32 = Color32::from_rgb(205, 210, 220);
const GLASS: Color32 = Color32::from_rgba_premultiplied(70, 105, 140, 120);
const DETECTOR: Color32 = Color32::from_rgb(110, 225, 120);
const TRACE: [Color32; 5] = [
    Color32::from_rgb(90, 200, 255),
    Color32::from_rgb(255, 165, 70),
    Color32::from_rgb(230, 120, 230),
    Color32::from_rgb(130, 225, 120),
    Color32::from_rgb(240, 230, 110),
];
const BACK_TRACE: Color32 = Color32::from_gray(150);

/// seconds for the animated sweep to cross the range
const SWEEP_SECONDS: f64 = 8.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tool {
    Laser,
    Mirror,
    Splitter,
    CavityMirror,
    Pbs,
    HalfWave,
    QuarterWave,
    Polarizer,
    Phase,
    Faraday,
    Detector,
    Block,
}

impl Tool {
    const ALL: [Tool; 12] = [
        Tool::Laser,
        Tool::Mirror,
        Tool::Splitter,
        Tool::CavityMirror,
        Tool::Pbs,
        Tool::HalfWave,
        Tool::QuarterWave,
        Tool::Polarizer,
        Tool::Phase,
        Tool::Faraday,
        Tool::Detector,
        Tool::Block,
    ];

    fn label(self) -> &'static str {
        match self {
            Tool::Laser => "Laser",
            Tool::Mirror => "Mirror",
            Tool::Splitter => "Beam splitter",
            Tool::CavityMirror => "Cavity mirror",
            Tool::Pbs => "PBS",
            Tool::HalfWave => "λ/2 plate",
            Tool::QuarterWave => "λ/4 plate",
            Tool::Polarizer => "Polariser",
            Tool::Phase => "Phase shifter",
            Tool::Faraday => "Faraday rotator",
            Tool::Detector => "Detector",
            Tool::Block => "Beam block",
        }
    }

    fn hint(self) -> &'static str {
        match self {
            Tool::Laser => "single-frequency laser; only one on the table. Light coming back into it is absorbed (and shown as 'back into the laser')",
            Tool::Mirror => "R = 100 %, at 45°; right-click to turn it",
            Tool::Splitter => "50:50 beam splitter, at 45°",
            Tool::CavityMirror => "R = 90 %, facing the beam: for cavities",
            Tool::Pbs => "polarising beam splitter: passes H (in the table plane), reflects V",
            Tool::HalfWave => "half-wave plate: turns linear polarisation by twice its axis angle",
            Tool::QuarterWave => "quarter-wave plate: linear at 45° to its axis ↔ circular",
            Tool::Polarizer => "passes the polarisation along its axis, absorbs the rest",
            Tool::Phase => "adds a phase to light passing either way (a glass plate, an electro-optic modulator)",
            Tool::Faraday => "turns the polarisation about the magnetic field: in the same sense in the lab for both directions",
            Tool::Detector => "photodiode: measures the power arriving from any side",
            Tool::Block => "absorbs the light",
        }
    }

    fn part(self, x: i32, y: i32) -> Part {
        let p = |k: Kind, turn: u8| Part::new(k, x, y, turn);
        match self {
            Tool::Laser => p(Kind::Laser, 0),
            Tool::Mirror => p(Kind::Mirror, SLASH),
            Tool::Splitter => Part { reflect: 0.5, ..p(Kind::Mirror, SLASH) },
            Tool::CavityMirror => Part { reflect: 0.9, ..p(Kind::Mirror, BAR) },
            Tool::Pbs => p(Kind::Pbs, SLASH),
            Tool::HalfWave => Part { retard: 0.5, ..p(Kind::Waveplate, 0) },
            Tool::QuarterWave => Part { retard: 0.25, angle_deg: 45.0, ..p(Kind::Waveplate, 0) },
            Tool::Polarizer => p(Kind::Polarizer, 0),
            Tool::Phase => p(Kind::Phase, 0),
            Tool::Faraday => p(Kind::Faraday, 0),
            Tool::Detector => p(Kind::Detector, 0),
            Tool::Block => p(Kind::Block, 0),
        }
    }
}

enum Drag {
    Part { id: u32, moved: bool },
    Marker,
}

/// what the sweep depends on: everything but the view and the swept value itself
fn sweep_key(p: &IfoParams) -> IfoParams {
    let mut q = p.clone();
    q.show_pol = false;
    q.show_power = false;
    q.sweep.in_fsr = false;
    q.sweep.log = false;
    q.sweep.show_back = false;
    match q.sweep.var {
        SweepVar::Frequency => q.laser.detuning_mhz = 0.0,
        SweepVar::Part(id) => {
            if let Some(part) = q.parts.iter_mut().find(|x| x.id == id) {
                part.set_sweep_param(0.0);
            }
        }
    }
    q
}

/// what the table and the switch-on depend on: everything but the view and the sweep
fn optics_key(p: &IfoParams) -> IfoParams {
    let mut q = p.clone();
    q.show_pol = false;
    q.show_power = false;
    q.sweep = Default::default();
    q
}

pub struct IfoUi {
    pub params: IfoParams,
    pub notes: String,
    worker: Option<Worker<IfoParams, SweepResult>>,
    result: Option<SweepResult>,
    steady: Option<(IfoParams, Steady)>,
    selected: Option<u32>,
    tool: Option<Tool>,
    drag: Option<Drag>,
    time: Option<TimeSim>,
    playing: bool,
    /// time steps per second of real time
    speed: f64,
    step_carry: f64,
    animate: bool,
    animate_dir: f64,
    hover_text: Option<(Pos2, String)>,
}

impl Default for IfoUi {
    fn default() -> Self {
        let (params, notes) = IfoPreset::FabryPerot.setup();
        IfoUi {
            params,
            notes,
            worker: None,
            result: None,
            steady: None,
            selected: None,
            tool: None,
            drag: None,
            time: None,
            playing: false,
            speed: 300.0,
            step_carry: 0.0,
            animate: false,
            animate_dir: 1.0,
            hover_text: None,
        }
    }
}

// ---------------------------------------------------------------- formatting

pub fn fmt_freq(hz: f64) -> String {
    let a = hz.abs();
    if a >= 1e9 {
        format!("{:.3} GHz", hz * 1e-9)
    } else if a >= 1e6 {
        format!("{:.3} MHz", hz * 1e-6).replace(".000 MHz", " MHz")
    } else if a >= 1e3 {
        format!("{:.1} kHz", hz * 1e-3)
    } else {
        format!("{hz:.0} Hz")
    }
}

fn fmt_power(mw: f64) -> String {
    let a = mw.abs();
    if a >= 1000.0 {
        format!("{:.2} W", mw * 1e-3)
    } else if a >= 10.0 {
        format!("{mw:.1} mW")
    } else if a >= 0.01 {
        format!("{mw:.3} mW")
    } else if a >= 1e-5 {
        format!("{:.2} µW", mw * 1e3)
    } else {
        "0".into()
    }
}

fn fmt_len(m: f64) -> String {
    if m >= 1.0 {
        format!("{m:.2} m")
    } else if m >= 0.01 {
        format!("{:.1} cm", m * 100.0)
    } else {
        format!("{:.2} mm", m * 1000.0)
    }
}

fn kind_name(p: &Part) -> &'static str {
    match p.kind {
        Kind::Laser => "Laser",
        Kind::Mirror if p.reflect >= 0.9999 => "Mirror",
        Kind::Mirror if p.turn % 2 == 1 => "Beam splitter",
        Kind::Mirror => "Partly transmitting mirror",
        Kind::Pbs => "Polarising beam splitter",
        Kind::Waveplate => "Waveplate",
        Kind::Polarizer => "Polariser",
        Kind::Phase => "Phase shifter",
        Kind::Faraday => "Faraday rotator",
        Kind::Detector => "Detector",
        Kind::Block => "Beam block",
    }
}

fn dir_vec(d: Dir) -> Vec2 {
    let (x, y) = d.step();
    vec2(x as f32, y as f32)
}

/// to the right of the direction of travel, on the screen (y down)
pub(crate) fn right_of(d: Vec2) -> Vec2 {
    vec2(-d.y, d.x)
}

fn laser_rgb(p: &IfoParams) -> [f32; 3] {
    let c = wavelength_rgb(p.laser.wavelength_nm.clamp(380.0, 780.0) as f32);
    // infrared lasers: draw them dark red
    if p.laser.wavelength_nm > 780.0 { [0.75, 0.12, 0.12] } else { [c[0].max(0.15), c[1], c[2]] }
}

/// a stroke width and colour
type Pen = (f32, Color32);

/// width and colour of a beam of power `ratio` × the laser power, and its glow
fn beam_style(ratio: f64, rgb: [f32; 3], s: f32) -> Option<(f32, Color32, Option<Pen>)> {
    if ratio < 1e-5 {
        return None;
    }
    let k = (s / 34.0).clamp(0.6, 1.5);
    let width = (1.3 + 2.6 * (1.0 + ratio).log10() as f32) * k;
    let alpha = (ratio.min(1.0) as f32).powf(0.4).max(0.12);
    // strong beams turn whitish
    let white = (((ratio.max(1.0)).log10() as f32) / 2.5).clamp(0.0, 0.6);
    let c = |v: f32| ((v * (1.0 - white) + white) * 255.0 * alpha) as u8;
    let core = Color32::from_rgba_premultiplied(c(rgb[0]), c(rgb[1]), c(rgb[2]), (255.0 * alpha) as u8);
    let glow = (ratio > 1.0).then(|| {
        let a = (0.12 + 0.1 * ratio.log10() as f32).min(0.35);
        let g = |v: f32| (v * 255.0 * a) as u8;
        (width * 2.8, Color32::from_rgba_premultiplied(g(rgb[0]), g(rgb[1]), g(rgb[2]), (255.0 * a) as u8))
    });
    Some((width, core, glow))
}

pub(crate) fn dashed(painter: &egui::Painter, a: Pos2, b: Pos2, stroke: Stroke) {
    painter.extend(Shape::dashed_line(&[a, b], stroke, 5.0, 4.0));
}

pub(crate) fn arrow_head(painter: &egui::Painter, tip: Pos2, d: Vec2, size: f32, col: Color32) {
    let n = right_of(d);
    painter.add(Shape::convex_polygon(vec![tip, tip - d * size + n * size * 0.55, tip - d * size - n * size * 0.55], col, Stroke::NONE));
}

/// a label on a dark box
pub(crate) fn tag(painter: &egui::Painter, pos: Pos2, align: Align2, text: &str, col: Color32, size: f32) -> Rect {
    let g = painter.layout_no_wrap(text.to_string(), FontId::proportional(size), col);
    let r = align.anchor_size(pos, g.size()).expand(2.0);
    painter.rect_filled(r, 2.0, Color32::from_black_alpha(170));
    painter.galley(r.min + vec2(2.0, 2.0), g, col);
    r
}

/// unit and scale for the sweep axis (values are in MHz, nm or degrees)
fn axis_unit(unit: &str, span: f64) -> (&'static str, f64) {
    match unit {
        "MHz" if span >= 5000.0 => ("GHz", 1e-3),
        "MHz" => ("MHz", 1.0),
        "nm" if span >= 2e7 => ("mm", 1e-6),
        "nm" if span >= 2e4 => ("µm", 1e-3),
        "nm" => ("nm", 1.0),
        _ => ("°", 1.0),
    }
}

/// a polyline through many points, reduced to min/max per pixel column
pub(crate) fn polyline(xs: &[f64], ys: &[f64], to_screen: impl Fn(f64, f64) -> Pos2, width_px: f32) -> Vec<Pos2> {
    let n = xs.len().min(ys.len());
    let pts: Vec<Pos2> = (0..n).map(|i| to_screen(xs[i], ys[i])).collect();
    if (n as f32) < 3.0 * width_px {
        return pts;
    }
    let mut out: Vec<Pos2> = vec![];
    let mut i = 0;
    while i < n {
        let col = pts[i].x.floor();
        let (mut lo, mut hi) = (pts[i], pts[i]);
        let first = pts[i];
        while i < n && pts[i].x.floor() == col {
            if pts[i].y < lo.y {
                lo = pts[i];
            }
            if pts[i].y > hi.y {
                hi = pts[i];
            }
            i += 1;
        }
        out.push(first);
        out.push(lo);
        out.push(hi);
    }
    out
}

/// axes with ticks; returns the plot rectangle inside the margins
pub(crate) fn axes(painter: &egui::Painter, rect: Rect, x: (f64, f64), y: (f64, f64), xlabel: &str, ylabel: &str, log: bool) -> Rect {
    let plot = Rect::from_min_max(rect.min + vec2(46.0, 8.0), rect.max - vec2(10.0, 30.0));
    painter.rect_filled(rect, 3.0, BG);
    let sx = |v: f64| plot.left() + ((v - x.0) / (x.1 - x.0)) as f32 * plot.width();
    let sy = |v: f64| plot.bottom() - ((v - y.0) / (y.1 - y.0)) as f32 * plot.height();
    let grid = Stroke::new(1.0, Color32::from_gray(38));
    let font = FontId::proportional(10.0);
    for v in ticks(x.0, x.1, (plot.width() / 80.0).max(2.0) as f64) {
        let px = sx(v);
        painter.line_segment([pos2(px, plot.top()), pos2(px, plot.bottom())], grid);
        painter.text(pos2(px, plot.bottom() + 3.0), Align2::CENTER_TOP, fmt_tick(v), font.clone(), LABEL);
    }
    if log {
        let mut d = y.0.ceil();
        while d <= y.1 {
            let py = sy(d);
            painter.line_segment([pos2(plot.left(), py), pos2(plot.right(), py)], grid);
            painter.text(pos2(plot.left() - 4.0, py), Align2::RIGHT_CENTER, format!("1e{d:.0}"), font.clone(), LABEL);
            d += 1.0;
        }
    } else {
        for v in ticks(y.0, y.1, (plot.height() / 40.0).max(2.0) as f64) {
            let py = sy(v);
            painter.line_segment([pos2(plot.left(), py), pos2(plot.right(), py)], grid);
            painter.text(pos2(plot.left() - 4.0, py), Align2::RIGHT_CENTER, fmt_tick(v), font.clone(), LABEL);
        }
    }
    painter.rect_stroke(plot, 0.0, Stroke::new(1.0, Color32::from_gray(70)), StrokeKind::Inside);
    painter.text(pos2(plot.center().x, rect.bottom() - 2.0), Align2::CENTER_BOTTOM, xlabel, font.clone(), LABEL);
    painter.text(pos2(rect.left() + 3.0, plot.top()), Align2::LEFT_TOP, ylabel, font, LABEL);
    plot
}

impl IfoUi {
    pub fn load_preset(&mut self, p: IfoPreset) {
        let (params, notes) = p.setup();
        self.params = params;
        self.notes = notes;
        self.selected = None;
        self.tool = None;
        self.time = None;
        self.playing = false;
        self.animate = false;
        if p == IfoPreset::SwitchOn {
            self.switch_on();
        }
    }

    /// talk to the worker; call once per frame while the bench is shown
    pub fn update(&mut self, ctx: &egui::Context) {
        let worker = self.worker.get_or_insert_with(|| {
            let c = ctx.clone();
            // milliseconds; the browser computes in the frame, the desktop on a thread
            let budget = if cfg!(target_arch = "wasm32") { 60.0 } else { 150.0 };
            Worker::new("interferometer", move |q: IfoParams| sweep(&q, budget), move || c.request_repaint())
        });
        worker.request(&sweep_key(&self.params));
        if let Some(r) = worker.poll() {
            self.result = Some(r);
        }
        if worker.busy() {
            ctx.request_repaint();
        }
    }

    fn steady(&mut self) -> &Steady {
        let key = optics_key(&self.params);
        if self.steady.as_ref().is_none_or(|(k, _)| *k != key) {
            let s = steady(&self.params);
            self.steady = Some((key, s));
        }
        &self.steady.as_ref().expect("steady state").1
    }

    fn switch_on(&mut self) {
        self.time = Some(TimeSim::new(&self.params));
        self.playing = true;
        self.step_carry = 0.0;
    }

    fn selected_index(&self) -> Option<usize> {
        let id = self.selected?;
        self.params.parts.iter().position(|p| p.id == id)
    }

    fn rotate(&mut self, i: usize, back: bool) {
        let p = &mut self.params.parts[i];
        let step = match p.kind {
            Kind::Mirror => 1,
            Kind::Pbs | Kind::Laser | Kind::Faraday => {
                if p.kind == Kind::Pbs {
                    2
                } else {
                    1
                }
            }
            _ => 0,
        };
        if step > 0 {
            p.turn = if back { (p.turn + 4 - step) % 4 } else { (p.turn + step) % 4 };
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let win = ui.ctx().content_rect();
        self.advance_time(ui.ctx());
        self.animate_sweep(ui.ctx());
        egui::Panel::bottom("i_controls")
            .resizable(true)
            .default_size(250.0)
            .min_size(120.0)
            .max_size(win.height() * 0.45)
            .show(ui, |ui| self.controls_ui(ui));
        egui::Panel::right("i_plots")
            .resizable(true)
            .default_size(win.width() * 0.4)
            .min_size(300.0)
            .max_size(win.width() * 0.6)
            .show(ui, |ui| {
                let h = ui.available_height();
                ui.allocate_ui(vec2(ui.available_width(), h * 0.56), |ui| self.spectrum_ui(ui));
                ui.separator();
                self.time_ui(ui);
            });
        egui::CentralPanel::default().show(ui, |ui| self.table_ui(ui));
    }

    // ------------------------------------------------------------ time and animation

    fn advance_time(&mut self, ctx: &egui::Context) {
        // rebuild the switch-on when the optics changed
        if let Some(t) = &self.time
            && optics_key(&t.built_for) != optics_key(&self.params)
        {
            let on = t.laser_on;
            self.time = Some(TimeSim::new(&self.params));
            self.step_carry = 0.0;
            if !on {
                self.playing = false;
                self.time = None;
            }
        }
        let Some(t) = &mut self.time else { return };
        if !self.playing {
            return;
        }
        let dt = ctx.input(|i| i.stable_dt).min(0.1) as f64;
        self.step_carry += self.speed * dt;
        let n = (self.step_carry.floor() as usize).min(20_000);
        self.step_carry -= n as f64;
        self.step_carry = self.step_carry.min(1.0);
        t.advance(n);
        if t.history.len() >= HISTORY_MAX {
            self.playing = false;
        }
        ctx.request_repaint();
    }

    fn animate_sweep(&mut self, ctx: &egui::Context) {
        if !self.animate {
            return;
        }
        let (a, b, _) = self.params.sweep_axis();
        let dt = ctx.input(|i| i.stable_dt).min(0.1) as f64;
        let mut v = self.marker_value();
        v += self.animate_dir * (b - a) * dt / SWEEP_SECONDS;
        let (lo, hi) = (a.min(b), a.max(b));
        if v >= hi {
            v = hi;
            self.animate_dir = -self.animate_dir;
        } else if v <= lo {
            v = lo;
            self.animate_dir = -self.animate_dir;
        }
        self.set_marker_value(v);
        ctx.request_repaint();
    }

    /// the current value of the swept quantity (laser detuning or the part's parameter)
    fn marker_value(&self) -> f64 {
        match self.params.sweep.var {
            SweepVar::Frequency => self.params.laser.detuning_mhz,
            SweepVar::Part(id) => self.params.by_id(id).and_then(|p| p.sweep_param()).map_or(0.0, |s| s.2),
        }
    }

    fn set_marker_value(&mut self, v: f64) {
        match self.params.sweep.var {
            SweepVar::Frequency => self.params.laser.detuning_mhz = v,
            SweepVar::Part(id) => {
                if let Some(p) = self.params.parts.iter_mut().find(|p| p.id == id) {
                    p.set_sweep_param(v);
                }
            }
        }
    }

    // ------------------------------------------------------------ the table

    fn table_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.strong("OPTICAL TABLE");
            ui.weak(format!("{} × {} squares of {} mm", self.params.cols, self.params.rows, self.params.cell_mm));
            if let Some(tool) = self.tool {
                ui.colored_label(PICK, format!("click a square to place: {}", tool.label()));
                if ui.small_button("cancel").clicked() {
                    self.tool = None;
                }
            } else {
                ui.weak("drag parts to move · right-click or R turns · Del removes");
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if self.time.is_some() {
                    ui.colored_label(PICK, "showing the switch-on in time");
                } else {
                    ui.weak("steady state");
                }
            });
        });
        let avail = ui.available_size();
        let (rect, resp) = ui.allocate_exact_size(avail, Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3.0, BG);
        let p = &self.params;
        let s = ((rect.width() - 16.0) / p.cols as f32).min((rect.height() - 16.0) / p.rows as f32).max(4.0);
        let size = vec2(s * p.cols as f32, s * p.rows as f32);
        let origin = rect.center() - size * 0.5;
        let board = Rect::from_min_size(origin, size);
        let centre = |x: i32, y: i32| origin + vec2((x as f32 + 0.5) * s, (y as f32 + 0.5) * s);
        let square_at = |q: Pos2| -> Option<(i32, i32)> {
            let v = (q - origin) / s;
            let (x, y) = (v.x.floor() as i32, v.y.floor() as i32);
            p.inside(x, y).then_some((x, y))
        };
        painter.rect_filled(board, 4.0, TABLE);
        if s >= 10.0 {
            for y in 0..p.rows {
                for x in 0..p.cols {
                    painter.circle_filled(centre(x, y), (s * 0.05).max(1.0), HOLE);
                }
            }
        }
        let hover = resp.hover_pos();
        let hover_sq = hover.and_then(square_at);

        // ---- beams
        let rgb = laser_rgb(&self.params);
        let pref = self.params.laser.power_mw.max(1e-12);
        let show_pol = self.params.show_pol;
        let show_power = self.params.show_power;
        let mut segments: Vec<(Pos2, Pos2, String)> = vec![];
        let st_power: Vec<[f64; 4]>;
        let st_field: Vec<[Jones; 4]>;
        let st_arrived: Vec<f64>;
        {
            let st = self.steady();
            st_power = st.power.clone();
            st_field = st.field.clone();
            st_arrived = st.arrived.clone();
        }
        let p = &self.params;
        let lane = (s * 0.07).max(1.5);
        let laser_pol = p.laser.jones();
        let net_links = crate::interferometer::Network::new(p);
        for (i, part) in p.parts.iter().enumerate() {
            let Some(ni) = net_links.parts.iter().position(|q| q.id == part.id) else { continue };
            let c0 = centre(part.x, part.y);
            for d in Dir::ALL {
                let link = net_links.links[ni][d.index()];
                let dv = dir_vec(d);
                let off = right_of(dv) * lane;
                let c1 = c0 + dv * link.cells as f32 * s;
                let (a, b) = (c0 + off, c1 + off);
                if let Some(t) = &self.time {
                    // the field square by square, as it is now
                    let Some(k) = t.part_index(part.id) else { continue };
                    let Some(cells) = t.beam(k, d) else { continue };
                    let total = link.cells as f32;
                    for (j, v) in cells.iter().enumerate() {
                        let from = j as f32;
                        if from >= total {
                            break;
                        }
                        let to = (j as f32 + 1.0).min(total);
                        let pw = power(v);
                        if let Some((w, col, glow)) = beam_style(pw / pref, rgb, s) {
                            let (u, v2) = (a + (b - a) * (from / total), a + (b - a) * (to / total));
                            if let Some((gw, gc)) = glow {
                                painter.line_segment([u, v2], Stroke::new(gw, gc));
                            }
                            painter.line_segment([u, v2], Stroke::new(w, col));
                        }
                    }
                    continue;
                }
                let pw = st_power[i][d.index()];
                let Some((w, col, glow)) = beam_style(pw / pref, rgb, s) else { continue };
                if let Some((gw, gc)) = glow {
                    painter.line_segment([a, b], Stroke::new(gw, gc));
                }
                painter.line_segment([a, b], Stroke::new(w, col));
                // direction of travel, in the middle of the first square
                let len = link.cells as f32 * s;
                if len > s * 0.9 {
                    let tip = a + dv * (s * 0.62).min(len * 0.5);
                    arrow_head(&painter, tip, dv, (w + 4.0).min(s * 0.22), col.gamma_multiply(0.9));
                }
                let v = st_field[i][d.index()];
                let mut info = format!(
                    "{} {}  {}  ({:.3} × laser)\n{}",
                    part.title(),
                    d.arrow(),
                    fmt_power(pw),
                    pw / pref,
                    if self.params.laser.spectrum == Spectrum::Single { describe_polarisation(&v) } else { "several lines: polarisation of the first".into() }
                );
                if let Some(to) = link.to {
                    info += &format!("\n{} squares = {} to the {}", link.cells, fmt_len(link.cells * p.cell_m()), net_links.parts[to].title());
                } else {
                    info += "\nleaves the table";
                }
                segments.push((a, b, info));
                if show_pol && pw / pref > 0.01 && len > s * 1.4 && (part.kind == Kind::Laser || !same_polarisation(&v, &laser_pol)) {
                    let at = a + (b - a) * if len > 3.0 * s { 0.5 } else { 0.55 } + right_of(dv) * s * 0.3;
                    pol_glyph(&painter, at, (s * 0.2).clamp(6.0, 12.0), &v);
                }
                if show_power && pw / pref > 1e-3 && len > s * 1.4 {
                    let at = a + (b - a) * 0.5 - right_of(dv) * s * 0.32;
                    tag(&painter, at, Align2::CENTER_CENTER, &fmt_power(pw), LABEL, 9.5);
                } else if pw / pref >= 1.5 && d.forward() && len > s * 2.5 {
                    // the field has built up: say by how much
                    let at = a + (b - a) * 0.3 - right_of(dv) * (w * 0.5 + s * 0.3);
                    tag(&painter, at, Align2::CENTER_CENTER, &format!("{:.0} × laser power", pw / pref), PICK, 10.0);
                }
            }
        }

        // ---- parts
        let selected = self.selected;
        // in the switch-on, the detectors show what arrives now
        let now: Option<Vec<f32>> = self.time.as_ref().and_then(|t| t.history.last().cloned());
        let dets = p.detectors();
        let shown = |i: usize| -> f64 {
            match (&now, dets.iter().position(|&j| j == i)) {
                (Some(row), Some(k)) => row.get(k).copied().unwrap_or(0.0) as f64,
                (Some(row), None) => row.last().copied().unwrap_or(0.0) as f64,
                (None, _) => st_arrived[i],
            }
        };
        for (i, part) in p.parts.iter().enumerate() {
            if !p.inside(part.x, part.y) {
                continue;
            }
            let c = centre(part.x, part.y);
            let axis_h = st_power[i][Dir::E.index()] + st_power[i][Dir::W.index()] >= st_power[i][Dir::N.index()] + st_power[i][Dir::S.index()];
            draw_part(&painter, part, c, s, axis_h, rgb, false);
            if selected == Some(part.id) {
                painter.rect_stroke(Rect::from_center_size(c, vec2(s, s) * 0.96), 3.0, Stroke::new(1.5, SELECT), StrokeKind::Inside);
            }
            if s >= 20.0 {
                let mut text = match part.kind {
                    Kind::Detector => format!("{}  {}", p.detector_name(i), fmt_power(shown(i))),
                    Kind::Laser => format!("laser {}", fmt_power(p.laser.power_mw)),
                    _ if !part.name.is_empty() => format!("{} {}", part.name, part.label()),
                    _ => part.label(),
                };
                if part.kind == Kind::Mirror && part.shift_nm != 0.0 {
                    text += &format!(" {:+.0} nm", part.shift_nm);
                }
                tag(&painter, c + vec2(0.0, s * 0.47), Align2::CENTER_TOP, &text, LABEL, 9.5);
            }
        }
        if let Some(l) = p.parts.iter().position(|q| q.kind == Kind::Laser)
            && shown(l) > 1e-4 * pref
            && s >= 20.0
        {
            let q = &p.parts[l];
            tag(&painter, centre(q.x, q.y) - vec2(0.0, s * 0.47), Align2::CENTER_BOTTOM, &format!("back: {}", fmt_power(shown(l))), BACK_TRACE, 9.5);
        }

        // ---- placing a new part
        if let Some(tool) = self.tool {
            if let Some((x, y)) = hover_sq {
                let free = p.at(x, y).is_none();
                let ghost = tool.part(x, y);
                draw_part(&painter, &ghost, centre(x, y), s, true, rgb, true);
                if !free {
                    painter.rect_stroke(Rect::from_center_size(centre(x, y), vec2(s, s)), 2.0, Stroke::new(1.5, Color32::from_rgb(230, 90, 80)), StrokeKind::Inside);
                }
                if resp.clicked() && free {
                    let mut part = tool.part(x, y);
                    if tool == Tool::Laser {
                        self.params.parts.retain(|q| q.kind != Kind::Laser);
                    }
                    if part.kind == Kind::Waveplate || part.kind == Kind::Polarizer {
                        part.turn = 0;
                    }
                    if matches!(part.kind, Kind::Pbs | Kind::Waveplate | Kind::Polarizer | Kind::Faraday) {
                        self.params.show_pol = true;
                    }
                    let id = self.params.add(part);
                    self.selected = Some(id);
                    if !ui.input(|i| i.modifiers.shift) {
                        self.tool = None;
                    }
                }
            }
            if resp.secondary_clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                self.tool = None;
            }
            ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
        } else {
            self.table_interaction(ui, &resp, hover_sq);
        }

        // ---- hover info on beams
        self.hover_text = None;
        if self.drag.is_none()
            && self.tool.is_none()
            && let Some(q) = hover
            && hover_sq.is_none_or(|(x, y)| self.params.at(x, y).is_none())
        {
            let mut best: Option<(f32, String)> = None;
            for (a, b, info) in &segments {
                let ab = *b - *a;
                let t = ((q - *a).dot(ab) / ab.length_sq().max(1e-6)).clamp(0.0, 1.0);
                let dist = (q - (*a + ab * t)).length();
                if dist < 7.0 && best.as_ref().is_none_or(|(d, _)| dist < *d) {
                    best = Some((dist, info.clone()));
                }
            }
            if let Some((_, info)) = best {
                self.hover_text = Some((q, info));
            }
        }
        if let Some((q, text)) = &self.hover_text {
            readout(&painter, *q + vec2(14.0, 10.0), Align2::LEFT_TOP, text.clone());
        }
        if self.params.parts.iter().all(|q| q.kind != Kind::Laser) {
            tag(&painter, board.center(), Align2::CENTER_CENTER, "Put a laser on the table (toolbox below)", PICK, 13.0);
        }
        if show_pol && s >= 20.0 {
            painter.text(
                board.left_bottom() + vec2(4.0, -3.0),
                Align2::LEFT_BOTTOM,
                "circles: the laser's polarisation, and wherever it has changed; seen looking into the beam (H = in the table plane, V = up)",
                FontId::proportional(10.0),
                Color32::from_gray(120),
            );
        }
    }

    fn table_interaction(&mut self, ui: &egui::Ui, resp: &egui::Response, hover_sq: Option<(i32, i32)>) {
        let part_at = |p: &IfoParams, sq: Option<(i32, i32)>| sq.and_then(|(x, y)| p.at(x, y)).map(|i| p.parts[i].id);
        if resp.drag_started_by(egui::PointerButton::Primary)
            && let Some(id) = part_at(&self.params, hover_sq)
        {
            self.selected = Some(id);
            self.drag = Some(Drag::Part { id, moved: false });
        }
        if let Some(Drag::Part { id, moved }) = &mut self.drag {
            if let Some((x, y)) = hover_sq
                && let Some(i) = self.params.parts.iter().position(|p| p.id == *id)
                && (self.params.parts[i].x, self.params.parts[i].y) != (x, y)
                && self.params.at(x, y).is_none()
            {
                self.params.parts[i].x = x;
                self.params.parts[i].y = y;
                *moved = true;
            }
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
            if !resp.dragged() {
                self.drag = None;
            }
        } else if part_at(&self.params, hover_sq).is_some() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
        }
        if resp.clicked() {
            self.selected = part_at(&self.params, hover_sq);
        }
        if resp.secondary_clicked()
            && let Some(id) = part_at(&self.params, hover_sq)
            && let Some(i) = self.params.parts.iter().position(|p| p.id == id)
        {
            self.selected = Some(id);
            self.rotate(i, false);
        }
        // keys act on the selected part
        if ui.ctx().egui_wants_keyboard_input() {
            return;
        }
        let Some(i) = self.selected_index() else { return };
        let (del, rot, shift, mv) = ui.input(|inp| {
            let mut mv = (0, 0);
            if inp.key_pressed(egui::Key::ArrowLeft) {
                mv.0 -= 1;
            }
            if inp.key_pressed(egui::Key::ArrowRight) {
                mv.0 += 1;
            }
            if inp.key_pressed(egui::Key::ArrowUp) {
                mv.1 -= 1;
            }
            if inp.key_pressed(egui::Key::ArrowDown) {
                mv.1 += 1;
            }
            (inp.key_pressed(egui::Key::Delete) || inp.key_pressed(egui::Key::Backspace), inp.key_pressed(egui::Key::R), inp.modifiers.shift, mv)
        });
        if del {
            self.params.parts.remove(i);
            self.selected = None;
            return;
        }
        if rot {
            self.rotate(i, shift);
        }
        if mv != (0, 0) {
            let (x, y) = (self.params.parts[i].x + mv.0, self.params.parts[i].y + mv.1);
            if self.params.inside(x, y) && self.params.at(x, y).is_none() {
                self.params.parts[i].x = x;
                self.params.parts[i].y = y;
            }
        }
    }

    // ------------------------------------------------------------ spectrum

    fn spectrum_ui(&mut self, ui: &mut egui::Ui) {
        let (a, b, unit) = self.params.sweep_axis();
        let what = match self.params.sweep.var {
            SweepVar::Frequency => "laser frequency".to_string(),
            SweepVar::Part(id) => match self.params.by_id(id) {
                Some(p) => p.sweep_name(),
                None => "(part removed)".into(),
            },
        };
        let cavity = self.steady().cavity.clone();
        ui.horizontal(|ui| {
            ui.strong("SWEEP");
            ui.weak(format!("detector power vs {what}"));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if let Some(r) = &self.result {
                    ui.weak(format!("{} points, {:.0} ms", r.xs.len(), r.millis));
                }
                if ui
                    .button(if self.animate { "⏸ stop" } else { "▶ sweep" })
                    .on_hover_text("move the yellow line back and forth: watch the table as the laser goes through the resonances")
                    .clicked()
                {
                    self.animate = !self.animate;
                }
            });
        });
        let avail = ui.available_size();
        let readout_h = 50.0;
        let (rect, resp) = ui.allocate_exact_size(vec2(avail.x, (avail.y - readout_h).max(80.0)), Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        let Some(r) = self.result.clone() else {
            painter.rect_filled(rect, 3.0, BG);
            painter.text(rect.center(), Align2::CENTER_CENTER, "computing…", FontId::proportional(13.0), LABEL);
            return;
        };
        let span = (b - a).abs();
        let fsr_mhz = cavity.as_ref().map(|c| c.fsr_hz * 1e-6);
        let in_fsr = self.params.sweep.in_fsr && self.params.sweep.var == SweepVar::Frequency && fsr_mhz.is_some();
        let (uname, uscale) = if in_fsr { ("FSR", 1.0 / fsr_mhz.unwrap_or(1.0)) } else { axis_unit(unit, span) };
        let xr = (a * uscale, b * uscale);
        let show: Vec<usize> = (0..r.traces.len()).filter(|&t| t + 1 < r.traces.len() || self.params.sweep.show_back).collect();
        let ymax = show.iter().flat_map(|&t| r.traces[t].iter()).cloned().fold(0.0f64, f64::max).max(1e-9);
        let log = self.params.sweep.log;
        let yr = if log { ((ymax * 1e-6).log10().floor(), ymax.log10().ceil().max((ymax * 1e-6).log10().floor() + 1.0)) } else { (0.0, ymax * 1.08) };
        let xlabel = match self.params.sweep.var {
            SweepVar::Frequency if in_fsr => "laser detuning Δν / FSR".to_string(),
            SweepVar::Frequency => format!("laser detuning Δν ({uname})"),
            SweepVar::Part(_) => format!("{what} ({uname})"),
        };
        let plot = axes(&painter, rect, xr, yr, &xlabel, "mW", log);
        let painter = ui.painter_at(plot.expand(1.0));
        let sx = |v: f64| plot.left() + ((v * uscale - xr.0) / (xr.1 - xr.0)) as f32 * plot.width();
        let sy = |v: f64| {
            let y = if log { v.max(1e-30).log10().max(yr.0) } else { v };
            plot.bottom() - ((y - yr.0) / (yr.1 - yr.0)) as f32 * plot.height()
        };
        // predicted resonances of the cavity
        if self.params.sweep.var == SweepVar::Frequency
            && let (Some(c), Some(fsr)) = (&cavity, fsr_mhz)
        {
            for m in &c.modes {
                let first = ((a / fsr - m.offset_fsr).ceil()) as i64;
                let last = ((b / fsr - m.offset_fsr).floor()) as i64;
                for q in first..=last.min(first + 400) {
                    let x = sx((q as f64 + m.offset_fsr) * fsr);
                    painter.line_segment([pos2(x, plot.bottom()), pos2(x, plot.bottom() - 6.0)], Stroke::new(1.5, Color32::from_rgb(120, 200, 140)));
                }
            }
        }
        for &t in show.iter().rev() {
            let back = t + 1 == r.traces.len();
            let col = if back { BACK_TRACE } else { TRACE[t % TRACE.len()] };
            let pts = polyline(&r.xs, &r.traces[t], |x, y| pos2(sx(x), sy(y)), plot.width());
            let stroke = Stroke::new(if back { 1.0 } else { 1.5 }, col);
            painter.add(Shape::line(pts, stroke));
        }
        // FSR and linewidth measured on the first detector trace
        let mut measured = String::new();
        if let Some(ys) = r.traces.first().filter(|_| r.traces.len() > 1) {
            let pk = peaks(&r.xs, ys, 1e-9 * self.params.laser.power_mw);
            if pk.len() >= 2 {
                let mid = 0.5 * (a + b);
                let k = (0..pk.len() - 1).min_by(|&i, &j| (pk[i].0 - mid).abs().total_cmp(&(pk[j].0 - mid).abs())).unwrap_or(0);
                let (x0, x1) = (pk[k].0, pk[k + 1].0);
                let y = sy(pk[k].1.max(pk[k + 1].1) * 0.5);
                let ya = plot.top() + 14.0;
                let col = Color32::from_rgb(120, 200, 140);
                painter.line_segment([pos2(sx(x0), ya), pos2(sx(x1), ya)], Stroke::new(1.2, col));
                arrow_head(&painter, pos2(sx(x0), ya), vec2(-1.0, 0.0), 6.0, col);
                arrow_head(&painter, pos2(sx(x1), ya), vec2(1.0, 0.0), 6.0, col);
                dashed(&painter, pos2(sx(x0), ya), pos2(sx(x0), sy(pk[k].1)), Stroke::new(1.0, col));
                dashed(&painter, pos2(sx(x1), ya), pos2(sx(x1), sy(pk[k + 1].1)), Stroke::new(1.0, col));
                let fmt = |v: f64| match self.params.sweep.var {
                    SweepVar::Frequency => fmt_freq(v * 1e6),
                    SweepVar::Part(_) => format!("{} {unit}", fmt_tick(v)),
                };
                let gap = x1 - x0;
                tag(&painter, pos2(0.5 * (sx(x0) + sx(x1)), ya - 2.0), Align2::CENTER_BOTTOM, &format!("peak spacing {}", fmt(gap)), col, 10.0);
                // full width at half maximum of the left peak
                let w = pk[k].2;
                let (wl, wr) = (sx(x0 - 0.5 * w), sx(x0 + 0.5 * w));
                if wr - wl > 1.0 {
                    painter.line_segment([pos2(wl, y), pos2(wr, y)], Stroke::new(1.5, PICK));
                }
                measured = format!("measured on D1: peak spacing {}, width (FWHM) {}, ratio {:.1}", fmt(gap), fmt(w), gap / w.max(1e-12));
            }
        }
        // the current value: drag it
        let mv = self.marker_value();
        let xm = sx(mv);
        if (plot.left() - 1.0..=plot.right() + 1.0).contains(&xm) {
            painter.line_segment([pos2(xm, plot.top()), pos2(xm, plot.bottom())], Stroke::new(1.5, PICK));
        }
        if resp.drag_started() || resp.clicked() {
            self.drag = Some(Drag::Marker);
            self.animate = false;
        }
        if matches!(self.drag, Some(Drag::Marker)) {
            if let Some(q) = resp.interact_pointer_pos() {
                let v = (xr.0 + (q.x - plot.left()) as f64 / plot.width() as f64 * (xr.1 - xr.0)) / uscale;
                self.set_marker_value(v.clamp(a.min(b), a.max(b)));
            }
            if !resp.dragged() {
                self.drag = None;
            }
        }
        if resp.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
        }
        // legend
        let mut items: Vec<(String, Color32)> = r.names.iter().enumerate().filter(|(t, _)| show.contains(t)).map(|(t, n)| {
            let back = t + 1 == r.traces.len();
            (n.clone(), if back { BACK_TRACE } else { TRACE[t % TRACE.len()] })
        }).collect();
        if r.traces.len() <= 1 {
            items.push(("no detector on the table".into(), LABEL));
        }
        let mut y = plot.top() + 4.0;
        for (n, c) in items {
            let rr = tag(&painter, pos2(plot.right() - 4.0, y), Align2::RIGHT_TOP, &n, c, 10.0);
            y = rr.bottom() + 2.0;
        }
        // readout below the plot
        let mut text = String::new();
        if let Some(c) = &cavity {
            text += &format!(
                "Cavity: round trip {} squares = {}, FSR = c/L = {}",
                c.cells,
                fmt_len(c.round_trip_m),
                fmt_freq(c.fsr_hz)
            );
            for (k, m) in c.modes.iter().enumerate() {
                if m.finesse.is_finite() {
                    text += &format!(
                        "\n{}round-trip loss {}: finesse π√ρ/(1−ρ) = {:.1}, linewidth FSR/F = {}{}",
                        if c.modes.len() > 1 { format!("mode {}: ", k + 1) } else { String::new() },
                        fmt_percent(1.0 - m.gain),
                        m.finesse,
                        fmt_freq(c.fsr_hz / m.finesse),
                        if m.offset_fsr > 1e-6 { format!(", resonances shifted by {:.3} FSR", m.offset_fsr) } else { String::new() }
                    );
                }
            }
        } else {
            text += "No closed light path (cavity) on the table.";
        }
        let escaped = self.steady().escaped;
        if escaped > 1e-4 * self.params.laser.power_mw {
            text += &format!("  {} leaves the table at its edge.", fmt_power(escaped));
        }
        if !measured.is_empty() {
            text += &format!("\n{measured}");
        }
        let (lr, _) = ui.allocate_exact_size(vec2(avail.x, readout_h), Sense::hover());
        ui.painter_at(lr).text(lr.left_top() + vec2(2.0, 4.0), Align2::LEFT_TOP, text, FontId::proportional(11.0), LABEL);
    }

    // ------------------------------------------------------------ switch-on

    fn time_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.strong("SWITCH-ON");
            if ui.button("▶ switch on").on_hover_text("start the laser at t = 0 and follow the light square by square").clicked() {
                self.switch_on();
            }
            if let Some(t) = &mut self.time {
                if ui.button(if self.playing { "⏸" } else { "▶" }).on_hover_text("pause / go on").clicked() {
                    self.playing = !self.playing;
                }
                let on = t.laser_on;
                if ui.button(if on { "laser off" } else { "laser on" }).on_hover_text("switch the laser off to see the light leave the cavity (ring-down)").clicked() {
                    t.set_laser(!on);
                    self.playing = true;
                }
                if ui.button("■ steady state").on_hover_text("back to the steady state").clicked() {
                    self.time = None;
                    self.playing = false;
                }
            }
            ui.label("speed");
            ui.add(egui::Slider::new(&mut self.speed, 3.0..=100_000.0).logarithmic(true).custom_formatter(|v, _| format!("{v:.0} steps/s")))
                .on_hover_text(format!("one step = one square = {:.0} ps", self.params.cell_m() / C_LIGHT * 1e12));
        });
        let avail = ui.available_size();
        let ph = avail.y.min(avail.x * 0.4).max(60.0);
        let (rect, _) = ui.allocate_exact_size(vec2(avail.x, avail.y.max(60.0)), Sense::hover());
        let trace_rect = Rect::from_min_max(rect.min, pos2(rect.right() - ph - 6.0, rect.bottom()));
        let phasor_rect = Rect::from_min_max(pos2(rect.right() - ph, rect.top()), rect.max);
        let painter = ui.painter_at(rect);
        let st_levels: Vec<f64>;
        let probe_field: Option<Jones>;
        {
            let p = self.params.clone();
            let st = self.steady();
            let dets = p.detectors();
            let mut lv: Vec<f64> = dets.iter().map(|&i| st.arrived[i]).collect();
            lv.push(p.parts.iter().position(|q| q.kind == Kind::Laser).map_or(0.0, |l| st.arrived[l]));
            st_levels = lv;
            probe_field = dets.first().map(|&i| st.at_part[i]);
        }
        let Some(t) = &self.time else {
            painter.rect_filled(rect, 3.0, BG);
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                "Press ▶ switch on to start the laser at t = 0\nand watch the light fill the setup.",
                FontId::proportional(12.0),
                LABEL,
            );
            return;
        };
        let tau_ns = t.tau * 1e9;
        let n = t.history.len();
        let t_end = (n.max(10) as f64) * tau_ns;
        let nt = st_levels.len();
        let mut ymax = st_levels.iter().cloned().fold(0.0f64, f64::max);
        for row in &t.history {
            for v in row {
                ymax = ymax.max(*v as f64);
            }
        }
        let ymax = ymax.max(1e-9) * 1.1;
        let rt = self.steady.as_ref().and_then(|(_, s)| s.cavity.as_ref().map(|c| c.cells));
        let xlabel = match rt {
            Some(c) => format!("time (ns)   ·   round trip = {} steps = {:.2} ns   ·   t = {} round trips", c, c * tau_ns, fmt_tick((n as f64 / c).floor())),
            None => "time (ns)".into(),
        };
        let plot = axes(&painter, trace_rect, (0.0, t_end), (0.0, ymax), &xlabel, "mW", false);
        let pp = ui.painter_at(plot.expand(1.0));
        let sx = |v: f64| plot.left() + (v / t_end) as f32 * plot.width();
        let sy = |v: f64| plot.bottom() - (v / ymax) as f32 * plot.height();
        let xs: Vec<f64> = (0..n).map(|k| k as f64 * tau_ns).collect();
        for k in (0..nt).rev() {
            let back = k + 1 == nt;
            if back && !self.params.sweep.show_back {
                continue;
            }
            let col = if back { BACK_TRACE } else { TRACE[k % TRACE.len()] };
            let ys: Vec<f64> = t.history.iter().map(|r| r.get(k).copied().unwrap_or(0.0) as f64).collect();
            pp.add(Shape::line(polyline(&xs, &ys, |x, y| pos2(sx(x), sy(y)), plot.width()), Stroke::new(if back { 1.0 } else { 1.5 }, col)));
            let y = sy(st_levels[k]);
            dashed(&pp, pos2(plot.left(), y), pos2(plot.right(), y), Stroke::new(1.0, col.gamma_multiply(0.6)));
        }
        if t.switched_at > 0 {
            let x = sx(t.switched_at as f64 * tau_ns);
            dashed(&pp, pos2(x, plot.top()), pos2(x, plot.bottom()), Stroke::new(1.0, PICK));
        }
        tag(&pp, plot.left_top() + vec2(4.0, 4.0), Align2::LEFT_TOP, "dashed: steady state", LABEL, 10.0);

        // phasor of the field at D1
        let painter = ui.painter_at(phasor_rect);
        painter.rect_filled(phasor_rect, 3.0, BG);
        let Some(target) = probe_field else {
            painter.text(phasor_rect.center(), Align2::CENTER_CENTER, "no detector", FontId::proportional(11.0), LABEL);
            return;
        };
        let comp = if target[0].norm_sqr() >= target[1].norm_sqr() { 0 } else { 1 };
        let pts: Vec<C64> = t.phasor.iter().map(|v| v[comp]).collect();
        let r_max = pts.iter().map(|z| z.norm()).fold(target[comp].norm(), f64::max).max(1e-9) * 1.15;
        let c = phasor_rect.center() + vec2(0.0, 6.0);
        let rad = (phasor_rect.width().min(phasor_rect.height() - 16.0) * 0.46).max(10.0);
        let to = |z: C64| c + vec2((z.re / r_max) as f32, -(z.im / r_max) as f32) * rad;
        painter.line_segment([c - vec2(rad, 0.0), c + vec2(rad, 0.0)], Stroke::new(1.0, Color32::from_gray(50)));
        painter.line_segment([c - vec2(0.0, rad), c + vec2(0.0, rad)], Stroke::new(1.0, Color32::from_gray(50)));
        // the field jumps whenever light from one more round trip arrives: those are the phasors of the chain
        let mut chain: Vec<C64> = vec![C64::new(0.0, 0.0)];
        for &z in &pts {
            if (z - *chain.last().expect("chain")).norm() > 1e-4 * r_max {
                chain.push(z);
            }
        }
        let stride = (chain.len() / 600).max(1);
        let line: Vec<Pos2> = chain.iter().step_by(stride).map(|&z| to(z)).collect();
        painter.add(Shape::line(line.clone(), Stroke::new(1.2, TRACE[0])));
        if line.len() < 120 {
            for q in &line {
                painter.circle_filled(*q, 1.8, TRACE[0]);
            }
        }
        if let Some(&z) = pts.last() {
            painter.line_segment([c, to(z)], Stroke::new(1.5, Color32::WHITE));
            painter.circle_filled(to(z), 3.0, Color32::WHITE);
        }
        painter.circle_stroke(to(target[comp]), 4.0, Stroke::new(1.2, PICK));
        painter.text(phasor_rect.left_top() + vec2(4.0, 3.0), Align2::LEFT_TOP, format!("field at D1 ({})", if comp == 0 { "H" } else { "V" }), FontId::proportional(10.0), LABEL);
        painter.text(phasor_rect.left_bottom() + vec2(4.0, -3.0), Align2::LEFT_BOTTOM, "○ steady state", FontId::proportional(10.0), PICK);
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
        egui::ScrollArea::vertical().id_salt("ifo_controls").auto_shrink([false, false]).show(ui, |ui| {
            ui.columns(4, |cols| {
                self.toolbox_ui(&mut cols[0]);
                self.inspector_ui(&mut cols[1]);
                self.laser_ui(&mut cols[2]);
                self.sweep_controls(&mut cols[3]);
            });
        });
    }

    fn toolbox_ui(&mut self, ui: &mut egui::Ui) {
        ui.strong("TOOLBOX");
        ui.weak("pick a part, then click a square (shift: several)");
        let has_laser = self.params.parts.iter().any(|p| p.kind == Kind::Laser);
        ui.horizontal_wrapped(|ui| {
            for t in Tool::ALL {
                let on = self.tool == Some(t);
                let enabled = t != Tool::Laser || !has_laser;
                let r = ui.add_enabled(enabled, egui::Button::new(t.label()).selected(on)).on_hover_text(t.hint());
                if r.clicked() {
                    self.tool = if on { None } else { Some(t) };
                }
            }
        });
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui.button("Clear the table").clicked() {
                self.params.parts.retain(|p| p.kind == Kind::Laser);
                self.selected = None;
                self.time = None;
            }
        });
    }

    fn inspector_ui(&mut self, ui: &mut egui::Ui) {
        ui.strong("SELECTED");
        let Some(i) = self.selected_index() else {
            ui.weak("Click a part on the table to change it.");
            return;
        };
        let lambda = self.params.laser.wavelength_nm;
        let st_arrived = self.steady().arrived.get(i).copied().unwrap_or(0.0);
        let mut sweep_this = false;
        let mut remove = false;
        let mut turn: Option<bool> = None;
        let id = self.params.parts[i].id;
        let swept = self.params.sweep.var == SweepVar::Part(id);
        let p = &mut self.params.parts[i];
        ui.horizontal(|ui| {
            ui.label(kind_name(p));
            ui.weak(format!("square ({}, {})", p.x, p.y));
        });
        ui.horizontal(|ui| {
            let turns = matches!(p.kind, Kind::Mirror | Kind::Pbs | Kind::Laser | Kind::Faraday);
            if turns {
                if ui.button("⟲ turn").on_hover_text("R (or right-click on the table)").clicked() {
                    turn = Some(false);
                }
                if ui.button("⟳").on_hover_text("shift+R").clicked() {
                    turn = Some(true);
                }
            }
            if ui.button("remove").on_hover_text("Delete key").clicked() {
                remove = true;
            }
            if p.sweep_param().is_some() && !swept && ui.button("sweep this").on_hover_text("scan this part's setting in the plot").clicked() {
                sweep_this = true;
            }
        });
        egui::Grid::new("ifo_part").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| match p.kind {
            Kind::Mirror => {
                ui.label("reflectivity R");
                ui.horizontal(|ui| {
                    let mut pct = p.reflect * 100.0;
                    if ui.add(egui::DragValue::new(&mut pct).range(0.0..=100.0).speed(0.05).max_decimals(3).suffix(" %")).changed() {
                        p.reflect = (pct / 100.0).clamp(0.0, 1.0);
                    }
                    for v in [0.5, 0.9, 0.99, 0.999, 1.0] {
                        if ui.small_button(fmt_percent(v).replace(" %", "")).clicked() {
                            p.reflect = v;
                        }
                    }
                });
                p.loss = p.loss.min(1.0 - p.reflect);
                ui.end_row();
                ui.label("loss");
                let mut loss = p.loss * 100.0;
                if ui
                    .add(egui::DragValue::new(&mut loss).range(0.0..=100.0).speed(0.02).max_decimals(3).suffix(" %"))
                    .on_hover_text("absorbed or scattered per pass; T = 1 − R − loss")
                    .changed()
                {
                    p.loss = (loss / 100.0).clamp(0.0, 1.0);
                }
                ui.end_row();
                p.reflect = p.reflect.min(1.0 - p.loss);
                ui.label("");
                ui.weak(format!("T = 1 − R − loss = {}", fmt_percent(p.transmit())));
                ui.end_row();
                ui.label("position");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut p.shift_nm).speed(1.0).suffix(" nm"))
                        .on_hover_text(format!("moved along its normal (a piezo); one wavelength is {lambda} nm"));
                    for (t, f) in [("−λ/8", -0.125), ("+λ/8", 0.125)] {
                        if ui.small_button(t).clicked() {
                            p.shift_nm += f * lambda;
                        }
                    }
                    if ui.small_button("0").clicked() {
                        p.shift_nm = 0.0;
                    }
                });
                ui.end_row();
            }
            Kind::Waveplate => {
                ui.label("retardance");
                ui.horizontal(|ui| {
                    for (v, t) in [(0.5, "λ/2"), (0.25, "λ/4"), (0.125, "λ/8")] {
                        if ui.selectable_label((p.retard - v).abs() < 1e-9, t).clicked() {
                            p.retard = v;
                        }
                    }
                    ui.add(egui::DragValue::new(&mut p.retard).range(0.0..=1.0).speed(0.002).max_decimals(3).suffix(" λ"));
                });
                ui.end_row();
                ui.label("fast axis");
                angle_row(ui, &mut p.angle_deg, &[0.0, 22.5, 45.0, 90.0]);
                ui.end_row();
            }
            Kind::Polarizer => {
                ui.label("axis");
                angle_row(ui, &mut p.angle_deg, &[0.0, 45.0, 90.0]);
                ui.end_row();
            }
            Kind::Phase => {
                ui.label("phase");
                ui.horizontal(|ui| {
                    ui.add(egui::Slider::new(&mut p.phase_deg, 0.0..=360.0).suffix("°"));
                    for v in [0.0, 90.0, 180.0] {
                        if ui.small_button(format!("{v:.0}")).clicked() {
                            p.phase_deg = v;
                        }
                    }
                });
                ui.end_row();
            }
            Kind::Faraday => {
                ui.label("rotation");
                angle_row(ui, &mut p.angle_deg, &[0.0, 45.0]);
                ui.end_row();
                ui.label("");
                ui.weak(format!("magnetic field points {} (on the screen)", p.dir().word()));
                ui.end_row();
            }
            Kind::Detector => {
                ui.label("name");
                ui.add(egui::TextEdit::singleline(&mut p.name).desired_width(120.0));
                ui.end_row();
                ui.label("power");
                ui.strong(fmt_power(st_arrived));
                ui.end_row();
            }
            Kind::Laser => {
                ui.label("");
                ui.weak("its settings are under LASER");
                ui.end_row();
            }
            Kind::Pbs => {
                ui.label("");
                ui.weak("passes H (in the table plane), reflects V");
                ui.end_row();
            }
            Kind::Block => {}
        });
        if remove {
            self.params.parts.remove(i);
            self.selected = None;
            return;
        }
        if let Some(back) = turn {
            self.rotate(i, back);
        }
        if sweep_this {
            let p = &self.params.parts[i];
            let (from, to) = match p.kind {
                Kind::Mirror => (p.shift_nm, p.shift_nm + 2.0 * lambda),
                Kind::Phase => (0.0, 720.0),
                _ => (0.0, 180.0),
            };
            self.params.sweep.var = SweepVar::Part(id);
            self.params.sweep.from = from;
            self.params.sweep.to = to;
        }
    }

    fn laser_ui(&mut self, ui: &mut egui::Ui) {
        let l = &mut self.params.laser;
        ui.strong("LASER");
        egui::Grid::new("ifo_laser").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
            ui.label("wavelength");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut l.wavelength_nm).range(380.0..=1600.0).speed(1.0).suffix(" nm"));
                for (v, t) in [(532.0, "532"), (633.0, "633"), (1064.0, "1064")] {
                    if ui.small_button(t).clicked() {
                        l.wavelength_nm = v;
                    }
                }
            });
            ui.end_row();
            ui.label("detuning Δν");
            let span = self.params.sweep.span_mhz.max(1.0);
            ui.horizontal(|ui| {
                ui.add(egui::Slider::new(&mut l.detuning_mhz, -span..=span).suffix(" MHz").fixed_decimals(1))
                    .on_hover_text("from the reference frequency, at which every square holds a whole number of wavelengths");
                if ui.small_button("0").clicked() {
                    l.detuning_mhz = 0.0;
                }
            });
            ui.end_row();
            ui.label("power");
            ui.add(egui::DragValue::new(&mut l.power_mw).range(0.001..=10000.0).speed(0.01).suffix(" mW"));
            ui.end_row();
            ui.label("polarisation");
            ui.horizontal(|ui| {
                for (t, a, e, hint) in [
                    ("H", 0.0, 0.0, "horizontal: in the plane of the table"),
                    ("V", 90.0, 0.0, "vertical"),
                    ("45°", 45.0, 0.0, "diagonal"),
                    ("↺", 0.0, 45.0, "left circular, (1, i)/√2 as in PS03: turns counter-clockwise seen looking into the beam"),
                    ("↻", 0.0, -45.0, "right circular, (1, −i)/√2: turns clockwise seen looking into the beam"),
                ] {
                    let on = (l.pol_deg - a).abs() < 1e-6 && (l.ellip_deg - e).abs() < 1e-6;
                    if ui.selectable_label(on, t).on_hover_text(hint).clicked() {
                        l.pol_deg = a;
                        l.ellip_deg = e;
                    }
                }
            });
            ui.end_row();
            ui.label("");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut l.pol_deg).range(-90.0..=180.0).speed(0.5).suffix("°")).on_hover_text("axis from horizontal");
                ui.add(egui::DragValue::new(&mut l.ellip_deg).range(-45.0..=45.0).speed(0.5).prefix("ellipticity ").suffix("°"))
                    .on_hover_text("0 = linear, ±45° = circular");
            });
            ui.end_row();
            ui.label("spectrum");
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt("ifo_spectrum").selected_text(l.spectrum.label()).show_ui(ui, |ui| {
                    for s in Spectrum::ALL {
                        ui.selectable_value(&mut l.spectrum, s, s.label());
                    }
                });
                match l.spectrum {
                    Spectrum::Single => {}
                    Spectrum::Two => {
                        ui.add(egui::DragValue::new(&mut l.line_sep_ghz).range(0.001..=100_000.0).speed(0.01).prefix("apart ").suffix(" GHz"));
                    }
                    Spectrum::Broad => {
                        ui.add(egui::DragValue::new(&mut l.width_ghz).range(0.001..=100_000.0).speed(0.1).prefix("width ").suffix(" GHz"));
                    }
                }
            });
            ui.end_row();
        });
        if l.spectrum != Spectrum::Single {
            ui.weak("several lines: their powers add (they do not interfere with each other). The switch-on uses one line.");
        }
    }

    fn sweep_controls(&mut self, ui: &mut egui::Ui) {
        ui.strong("SWEEP AND VIEW");
        let parts: Vec<(u32, String)> = self
            .params
            .parts
            .iter()
            .filter_map(|p| p.sweep_param().map(|s| (p.id, format!("{} ({})", p.sweep_name(), s.1))))
            .collect();
        let s = &mut self.params.sweep;
        egui::Grid::new("ifo_sweep").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
            ui.label("sweep");
            let current = match s.var {
                SweepVar::Frequency => "laser frequency".to_string(),
                SweepVar::Part(id) => parts.iter().find(|p| p.0 == id).map_or("(removed)".into(), |p| p.1.clone()),
            };
            egui::ComboBox::from_id_salt("ifo_sweep_var").selected_text(current).width(190.0).show_ui(ui, |ui| {
                ui.selectable_value(&mut s.var, SweepVar::Frequency, "laser frequency");
                for (id, name) in &parts {
                    ui.selectable_value(&mut s.var, SweepVar::Part(*id), name);
                }
            });
            ui.end_row();
            match s.var {
                SweepVar::Frequency => {
                    ui.label("range");
                    ui.add(egui::DragValue::new(&mut s.span_mhz).range(1.0..=1e7).speed(10.0).prefix("± ").custom_formatter(|v, _| format!("{:.0}", v / 2.0)).custom_parser(|t| t.trim().parse::<f64>().ok().map(|v| v * 2.0)).suffix(" MHz"));
                    ui.end_row();
                }
                SweepVar::Part(_) => {
                    ui.label("from … to");
                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut s.from).speed(1.0));
                        ui.add(egui::DragValue::new(&mut s.to).speed(1.0));
                    });
                    ui.end_row();
                }
            }
            ui.label("");
            ui.horizontal(|ui| {
                ui.checkbox(&mut s.show_back, "back into the laser");
                ui.checkbox(&mut s.log, "log");
                if s.var == SweepVar::Frequency {
                    ui.checkbox(&mut s.in_fsr, "in FSR").on_hover_text("frequency axis in units of the free spectral range of the cavity");
                }
            });
            ui.end_row();
        });
        ui.separator();
        let p = &mut self.params;
        egui::Grid::new("ifo_view").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
            ui.label("square");
            ui.add(egui::DragValue::new(&mut p.cell_mm).range(1.0..=1000.0).speed(0.5).suffix(" mm"))
                .on_hover_text("the size of a square sets all lengths (and the FSR)");
            ui.end_row();
            ui.label("table");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut p.cols).range(8..=60).suffix(" wide"));
                ui.add(egui::DragValue::new(&mut p.rows).range(6..=40).suffix(" deep"));
            });
            ui.end_row();
            ui.label("show");
            ui.horizontal(|ui| {
                ui.checkbox(&mut p.show_pol, "polarisation");
                ui.checkbox(&mut p.show_power, "powers");
            });
            ui.end_row();
        });
    }
}

fn angle_row(ui: &mut egui::Ui, v: &mut f64, quick: &[f64]) {
    ui.horizontal(|ui| {
        ui.add(egui::DragValue::new(v).range(-180.0..=180.0).speed(0.5).suffix("°"));
        for &q in quick {
            if ui.small_button(format!("{q}")).clicked() {
                *v = q;
            }
        }
    });
}

/// the same polarisation state (up to phase and power)
fn same_polarisation(a: &Jones, b: &Jones) -> bool {
    let (pa, pb) = (power(a), power(b));
    if pa <= 0.0 || pb <= 0.0 {
        return false;
    }
    let overlap = (a[0].conj() * b[0] + a[1].conj() * b[1]).norm_sqr() / (pa * pb);
    overlap > 0.9999
}

/// the polarisation ellipse as seen looking into the beam (H to the right, V up)
fn pol_glyph(painter: &egui::Painter, c: Pos2, r: f32, v: &Jones) {
    let p = power(v);
    if p <= 0.0 {
        return;
    }
    painter.circle(c, r, Color32::from_black_alpha(180), Stroke::new(1.0, Color32::from_gray(90)));
    let amp = p.sqrt();
    let at = |t: f64| {
        let e = C64::from_polar(1.0, -t);
        let (x, y) = ((v[0] * e).re / amp, (v[1] * e).re / amp);
        c + vec2(x as f32, -y as f32) * (r * 0.82)
    };
    let pts: Vec<Pos2> = (0..=40).map(|k| at(k as f64 / 40.0 * TAU as f64)).collect();
    let col = Color32::from_rgb(235, 235, 235);
    painter.add(Shape::line(pts, Stroke::new(1.3, col)));
    let (angle, e) = ellipse(v);
    if e.abs() > 3.0 {
        let a = at(0.0);
        let b = at(0.25);
        let d = (b - a).normalized();
        arrow_head(painter, a + d * 1.5, d, 4.0, col);
    } else {
        // linear: arrow heads at both ends
        let (sn, cs) = (angle.to_radians() as f32).sin_cos();
        let d = vec2(cs, -sn);
        arrow_head(painter, c + d * r * 0.85, d, 3.5, col);
        arrow_head(painter, c - d * r * 0.85, -d, 3.5, col);
    }
}

/// a part drawn on the table; `axis_h`: the beams through it run left–right
fn draw_part(painter: &egui::Painter, p: &Part, c: Pos2, s: f32, axis_h: bool, rgb: [f32; 3], ghost: bool) {
    let a = if ghost { 0.55 } else { 1.0 };
    let fade = |col: Color32| col.gamma_multiply(a);
    let line_dir = |turn: u8| -> Vec2 {
        match turn % 4 {
            BAR => vec2(0.0, 1.0),
            BACK => vec2(FRAC_1_SQRT_2, FRAC_1_SQRT_2),
            DASH => vec2(1.0, 0.0),
            _ => vec2(FRAC_1_SQRT_2, -FRAC_1_SQRT_2),
        }
    };
    // plates stand across the beam
    let across = if axis_h { vec2(0.0, 1.0) } else { vec2(1.0, 0.0) };
    let along = if axis_h { vec2(1.0, 0.0) } else { vec2(0.0, 1.0) };
    let plate = |w: f32, l: f32| {
        let corners = [c + along * w + across * l, c + along * w - across * l, c - along * w - across * l, c - along * w + across * l];
        corners.to_vec()
    };
    match p.kind {
        Kind::Laser => {
            let d = vec2(p.dir().step().0 as f32, p.dir().step().1 as f32);
            let n = right_of(d);
            let back = c - d * s * 0.46;
            let front = c + d * s * 0.3;
            let w = s * 0.24;
            let body = vec![back + n * w, front + n * w, front - n * w, back - n * w];
            painter.add(Shape::convex_polygon(body, fade(Color32::from_rgb(70, 74, 82)), Stroke::new(1.2, fade(Color32::from_gray(150)))));
            let col = Color32::from_rgb((rgb[0] * 255.0) as u8, (rgb[1] * 255.0) as u8, (rgb[2] * 255.0) as u8);
            painter.line_segment([back + d * s * 0.12, front - d * s * 0.08], Stroke::new((s * 0.06).max(1.5), fade(col)));
            painter.circle_filled(front, (s * 0.06).max(2.0), fade(col));
        }
        Kind::Mirror => {
            let u = line_dir(p.turn);
            let l = s * 0.42;
            if p.reflect >= 0.9999 {
                painter.line_segment([c - u * l, c + u * l], Stroke::new((s * 0.11).max(2.5), fade(SILVER)));
            } else {
                // glass with a partly reflecting coating
                let n = right_of(u);
                let w = (s * 0.07).max(2.0);
                painter.add(Shape::convex_polygon(vec![c - u * l + n * w, c + u * l + n * w, c + u * l - n * w, c - u * l - n * w], fade(GLASS), Stroke::NONE));
                let alpha = (0.25 + 0.75 * p.reflect) as f32;
                painter.line_segment([c - u * l, c + u * l], Stroke::new((s * 0.04).max(1.2), fade(SILVER.gamma_multiply(alpha))));
            }
        }
        Kind::Pbs => {
            let h = s * 0.32;
            let r = Rect::from_center_size(c, vec2(2.0 * h, 2.0 * h));
            painter.rect(r, 1.0, fade(Color32::from_rgba_premultiplied(60, 90, 130, 110)), Stroke::new(1.0, fade(Color32::from_rgb(150, 180, 220))), StrokeKind::Inside);
            let u = line_dir(p.turn);
            painter.line_segment([c - u * h * 1.41, c + u * h * 1.41], Stroke::new(1.5, fade(SILVER)));
        }
        Kind::Waveplate => {
            let col = if (p.retard - 0.5).abs() < 1e-9 {
                Color32::from_rgb(240, 170, 80)
            } else if (p.retard - 0.25).abs() < 1e-9 {
                Color32::from_rgb(110, 210, 230)
            } else {
                Color32::from_rgb(210, 210, 120)
            };
            painter.add(Shape::convex_polygon(plate(s * 0.07, s * 0.36), fade(col.gamma_multiply(0.75)), Stroke::new(1.0, fade(col))));
        }
        Kind::Polarizer => {
            painter.add(Shape::convex_polygon(plate(s * 0.06, s * 0.38), fade(Color32::from_gray(90)), Stroke::new(1.0, fade(Color32::from_gray(190)))));
            for k in -3..=3 {
                let q = c + across * (k as f32 * s * 0.1);
                painter.line_segment([q - along * s * 0.06, q + along * s * 0.06], Stroke::new(1.0, fade(Color32::from_gray(30))));
            }
        }
        Kind::Phase => {
            painter.add(Shape::convex_polygon(plate(s * 0.13, s * 0.3), fade(Color32::from_rgba_premultiplied(70, 120, 170, 140)), Stroke::new(1.0, fade(Color32::from_rgb(150, 200, 250)))));
        }
        Kind::Faraday => {
            let d = vec2(p.dir().step().0 as f32, p.dir().step().1 as f32);
            let n = right_of(d);
            let (l, w) = (s * 0.4, s * 0.22);
            painter.add(Shape::convex_polygon(
                vec![c - d * l + n * w, c + d * l + n * w, c + d * l - n * w, c - d * l - n * w],
                fade(Color32::from_rgb(95, 70, 130)),
                Stroke::new(1.0, fade(Color32::from_rgb(190, 160, 230))),
            ));
            painter.line_segment([c - d * l * 0.6, c + d * l * 0.45], Stroke::new(1.2, fade(Color32::from_rgb(230, 210, 255))));
            arrow_head(painter, c + d * l * 0.7, d, 5.0, fade(Color32::from_rgb(230, 210, 255)));
        }
        Kind::Detector => {
            let r = Rect::from_center_size(c, vec2(s * 0.56, s * 0.56));
            painter.rect(r, 3.0, fade(DETECTOR.gamma_multiply(0.75)), Stroke::new(1.2, fade(DETECTOR)), StrokeKind::Inside);
        }
        Kind::Block => {
            let r = Rect::from_center_size(c, vec2(s * 0.5, s * 0.5));
            painter.rect(r, 1.0, fade(Color32::from_gray(20)), Stroke::new(1.2, fade(Color32::from_rgb(170, 70, 60))), StrokeKind::Inside);
        }
    }
}
