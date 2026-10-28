//! User interface of examples 4 and 5: many modes (masses on springs or
//! strings in their fundamental mode), their energy distribution, the mean
//! energy in time and, for quantised modes, ⟨E⟩ and C against k_BT/ε.

use std::f64::consts::{PI, TAU};

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Sense, Shape, Stroke, pos2, vec2};

use crate::oscillators::{GUITAR, MAX_MODES, Modes, OscPreset, mean_quanta, quantum_heat_capacity, quantum_probability};
use crate::plot::{self, BG, CLASSICAL, LABEL, NOW, Plot, SIM, THEORY, fmt};
use crate::stats::{Avg, Hist, Moments, Trace};
use crate::widgets;

const KT_MIN: f64 = 0.05;
const KT_MAX: f64 = 5.0;
const EPS_MIN: f64 = 0.2;
const EPS_MAX: f64 = 5.0;
/// at most this many masses or strings are drawn
const MAX_MASSES: usize = 200;
const MAX_STRINGS: usize = 24;
const NOTE_NAMES: [&str; 6] = ["E", "A", "D", "G", "B", "e"];
/// drawn swings per second of the lowest pitch
const SWING_HZ: f64 = 1.0;
/// drawn amplitude per √(energy unit), as a fraction of a cell
const AMP: f32 = 0.22;
/// the most quanta a histogram bar is shown for
const MAX_BARS: usize = 60;
const SINGLE: Color32 = Color32::from_rgb(150, 160, 190);

pub struct OscUi {
    quantum: bool,
    example: OscPreset,
    m: usize,
    kt: f64,
    /// the step ε of the quantised energies
    eps: f64,
    strings: bool,
    guitar: bool,
    notes: &'static str,
    modes: Modes,
    running: bool,
    speed: f64,
    avg: Avg,
    /// the drawn time (s), for the swinging
    t_draw: f64,
    hist: Hist,
    energy_m: Moments,
    trace: Trace,
    time: f64,
    ranges_for: (f64, f64),
}

impl OscUi {
    pub fn new(quantum: bool) -> Self {
        let first = if quantum { OscPreset::QuantumEqual } else { OscPreset::Masses };
        let mut ui = OscUi {
            quantum,
            example: first,
            m: 0,
            kt: 1.0,
            eps: 1.0,
            strings: false,
            guitar: false,
            notes: "",
            modes: Modes::new(quantum, 0, 1.0, 1.0),
            running: true,
            speed: 1.0,
            avg: Avg::Now,
            t_draw: 0.0,
            hist: Hist::new(0.0, 1.0, 1),
            energy_m: Moments::default(),
            trace: Trace::new(30.0),
            time: 0.0,
            ranges_for: (0.0, 0.0),
        };
        ui.load_preset(first);
        ui
    }

    pub fn load_preset(&mut self, p: OscPreset) {
        let s = p.setup();
        self.example = p;
        self.m = s.m;
        self.kt = s.kt;
        self.eps = s.eps;
        self.strings = s.strings;
        self.guitar = s.guitar;
        self.notes = s.notes;
        self.avg = if s.long_average { Avg::Long } else { Avg::Now };
        self.modes = Modes::new(self.quantum, s.m, s.kt, s.eps);
        self.trace.clear();
        self.clear_stats();
    }

    fn clear_stats(&mut self) {
        self.hist = if self.quantum {
            let bars = ((6.0 * self.kt / self.eps).ceil() as usize + 2).clamp(6, MAX_BARS);
            Hist::new(-0.5, bars as f64 - 0.5, bars)
        } else {
            Hist::new(0.0, 6.0 * self.kt, 30)
        };
        self.energy_m.clear();
        self.ranges_for = (self.kt, self.eps);
    }

    /// the mean energy and C/k_B of one mode
    fn theory(&self) -> (f64, f64) {
        if self.quantum { (self.eps * mean_quanta(self.kt, self.eps), quantum_heat_capacity(self.kt, self.eps)) } else { (self.kt, 1.0) }
    }

    fn pitch(&self, k: usize) -> f64 {
        if self.guitar { GUITAR[k % 6] } else { 1.0 }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let dt = widgets::frame_dt(ui);
        if self.running {
            self.modes.step(self.kt, self.eps, dt * self.speed);
            self.t_draw += dt * self.speed;
            self.time += dt;
            self.record(dt);
            ui.ctx().request_repaint();
        }
        let win_h = ui.ctx().content_rect().height();
        let id = if self.quantum { "q" } else { "c" };
        egui::Panel::left(format!("{id}_side")).resizable(true).default_size(340.0).min_size(260.0).show(ui, |ui| self.side_ui(ui));
        egui::Panel::bottom(format!("{id}_plots"))
            .resizable(true)
            .default_size(win_h * 0.36)
            .min_size(150.0)
            .max_size(win_h * 0.6)
            .show(ui, |ui| self.plots_ui(ui));
        egui::CentralPanel::default().show(ui, |ui| self.view_ui(ui));
    }

    fn record(&mut self, dt: f64) {
        let tau = self.avg.tau();
        self.hist.fade(dt, tau);
        self.energy_m.fade(dt, tau);
        let m = self.modes.len();
        let mut sum = 0.0;
        for k in 0..m {
            let e = self.modes.energy(k, self.eps);
            self.hist.add(if self.quantum { self.modes.n[k] as f64 } else { e });
            self.energy_m.add(e);
            sum += e;
        }
        let first = if m > 0 { self.modes.energy(0, self.eps) } else { f64::NAN };
        self.trace.push(self.time, [sum / m.max(1) as f64, first, f64::NAN]);
    }

    fn side_ui(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            if self.quantum {
                widgets::heading(ui, "5 · Modes with quantised energy", "E_n = n ε,   Z₁ = 1/(1 − e^(−ε/k_BT))");
                if let Some(p) = widgets::examples(ui, "q_example", self.example, &OscPreset::QUANTUM, OscPreset::label) {
                    self.load_preset(p);
                }
            } else {
                widgets::heading(ui, "4 · Modes with continuous energy", "Z₁ ∝ β^(−1),   ⟨E⟩ = k_BT,   C = k_B");
                if let Some(p) = widgets::examples(ui, "c_example", self.example, &OscPreset::CLASSICAL, OscPreset::label) {
                    self.load_preset(p);
                }
            }

            widgets::section(ui, "SETTINGS");
            let mut changed = widgets::kt_slider(ui, &mut self.kt, (if self.quantum { 0.02 } else { KT_MIN })..=KT_MAX);
            if self.quantum {
                changed |=
                    ui.add(egui::Slider::new(&mut self.eps, EPS_MIN..=EPS_MAX).logarithmic(true).text("step ε").custom_formatter(|v, _| fmt(v))).changed();
                ui.weak(format!("k_BT/ε = {}", fmt(self.kt / self.eps)));
            }
            let mut m = self.m;
            if ui.add(egui::Slider::new(&mut m, 1..=MAX_MODES).logarithmic(true).text("modes M")).changed() {
                self.m = m;
                self.modes.resize(m, self.kt, self.eps);
                self.trace.clear();
                self.energy_m.clear();
                self.hist.clear();
            }
            ui.horizontal(|ui| {
                ui.label("show as:");
                ui.selectable_value(&mut self.strings, false, "masses on springs");
                ui.selectable_value(&mut self.strings, true, "strings");
            });
            if !self.quantum {
                ui.horizontal(|ui| {
                    ui.label("pitches:");
                    ui.selectable_value(&mut self.guitar, false, "all the same");
                    ui.selectable_value(&mut self.guitar, true, "a guitar's (E A D G B E)");
                });
            }
            widgets::play_row(ui, &mut self.running, &mut self.speed, 0.1..=5.0, "speed");
            ui.horizontal(|ui| {
                if ui.button("take all energy out").on_hover_text("every mode at rest (n = 0): watch the bath heat them up").clicked() {
                    self.modes.freeze();
                }
                if ui.button("equilibrium now").clicked() {
                    self.modes.equilibrate(self.kt, self.eps);
                }
            });
            if widgets::avg_row(ui, &mut self.avg) {
                self.clear_stats();
            }
            if changed && (self.kt, self.eps) != self.ranges_for {
                self.clear_stats();
            }

            widgets::section(ui, "NUMBERS (per mode)");
            let (mean, c) = self.theory();
            let kt = self.kt;
            let m = self.modes.len().max(1) as f64;
            let mut rows = vec![];
            if self.quantum {
                rows.push((
                    "⟨n⟩",
                    Some(mean_quanta(kt, self.eps)),
                    self.energy_m.mean().map(|e| e / self.eps),
                    "1/(e^(ε/k_BT) − 1), the mean number of quanta",
                ));
                rows.push(("P(n = 0)", Some(quantum_probability(0, kt, self.eps)), (self.hist.total > 0.0).then(|| self.hist.fraction(0)), "1 − e^(−ε/k_BT)"));
            }
            rows.push(("⟨E⟩", Some(mean), self.energy_m.mean(), if self.quantum { "ε ⟨n⟩" } else { "k_BT: equipartition" }));
            rows.push(("spread of E", Some(c.sqrt() * kt), self.energy_m.var().map(f64::sqrt), "the standard deviation of one mode's energy"));
            rows.push(("C/k_B", Some(c), self.energy_m.var().map(|v| v / (kt * kt)), "simulation: from the spread, C = Var(E)/k_BT²"));
            rows.push(("spread of the mean of M", Some(c.sqrt() * kt / m.sqrt()), None, "σ/√M: how much the mean over all M modes fluctuates"));
            widgets::numbers(ui, if self.quantum { "q_numbers" } else { "c_numbers" }, &rows);
            widgets::notes(ui, self.notes);
        });
    }

    fn view_ui(&mut self, ui: &mut egui::Ui) {
        let full = ui.available_rect_before_wrap();
        ui.allocate_rect(full, Sense::hover());
        let painter = ui.painter_at(full);
        painter.rect_filled(full, 3.0, BG);
        let m = self.modes.len();
        let shown = m.min(if self.strings { MAX_STRINGS } else { MAX_MASSES });
        let title = if self.strings { "STRINGS · fundamental mode, amplitude ∝ √E" } else { "MASSES ON SPRINGS · amplitude ∝ √E" };
        painter.text(full.min + vec2(8.0, 5.0), Align2::LEFT_TOP, title, FontId::proportional(13.0), LABEL);
        let more = if shown < m { format!("showing {shown} of {m} modes  ·  ") } else { String::new() };
        painter.text(full.right_top() + vec2(-8.0, 5.0), Align2::RIGHT_TOP, format!("{more}colour: energy (0 → 4 k_BT)"), FontId::proportional(12.0), LABEL);
        let inner = Rect::from_min_max(full.min + vec2(14.0, 28.0), full.max - vec2(14.0, 10.0));
        if inner.width() < 40.0 || inner.height() < 40.0 || shown == 0 {
            return;
        }
        if self.strings {
            self.draw_strings(&painter, inner, shown);
        } else {
            self.draw_masses(&painter, inner, shown);
        }
    }

    fn color(&self, k: usize) -> Color32 {
        plot::heat((self.modes.energy(k, self.eps) / (4.0 * self.kt)) as f32)
    }

    /// the drawn phase of mode k
    fn theta(&self, k: usize) -> f64 {
        TAU * SWING_HZ * self.pitch(k) * self.t_draw
    }

    fn draw_masses(&self, painter: &egui::Painter, inner: Rect, shown: usize) {
        let aspect = inner.width() / inner.height();
        let cols = ((shown as f32 * aspect).sqrt().ceil() as usize).clamp(1, shown);
        let rows = shown.div_ceil(cols);
        let (cw, ch) = (inner.width() / cols as f32, inner.height() / rows as f32);
        let base_r = (0.2 * ch).min(0.11 * cw).max(1.5);
        for k in 0..shown {
            let cell = Rect::from_min_size(inner.min + vec2((k % cols) as f32 * cw, (k / cols) as f32 * ch), vec2(cw, ch));
            let (cx, cy) = (cell.center().x, cell.center().y);
            let wall_x = cell.left() + 0.06 * cw;
            let x = cx + (self.modes.displacement(k, self.theta(k), self.eps) as f32 * AMP * cw).clamp(-0.36 * cw, 0.4 * cw);
            // a heavier mass (lower pitch) is drawn larger: m ∝ 1/ω² for the same spring
            let r = base_r * (self.pitch(k) as f32).powf(-2.0 / 3.0);
            if cw > 14.0 && ch > 8.0 {
                painter.line_segment([pos2(wall_x, cy - 0.3 * ch), pos2(wall_x, cy + 0.3 * ch)], Stroke::new(1.5, Color32::from_gray(110)));
                painter.add(spring(pos2(wall_x, cy), pos2(x - r, cy), 0.12 * ch, Stroke::new(1.0, Color32::from_gray(150))));
            }
            if self.quantum && cw > 34.0 {
                // the allowed amplitudes, √(n ε)
                for n in 1..8 {
                    let a = ((n as f64 * self.eps).sqrt() as f32 * AMP * cw).min(1e3);
                    if a > 0.36 * cw {
                        break;
                    }
                    for s in [-1.0, 1.0] {
                        let tx = cx + s * a;
                        painter.line_segment([pos2(tx, cy + r + 2.0), pos2(tx, cy + r + 5.0)], Stroke::new(1.0, Color32::from_gray(110)));
                    }
                }
            }
            painter.circle_filled(pos2(x, cy), r, self.color(k));
            if self.quantum && cw > 40.0 && ch > 26.0 {
                painter.text(pos2(cell.right() - 3.0, cell.top() + 2.0), Align2::RIGHT_TOP, format!("{}", self.modes.n[k]), FontId::proportional(10.0), LABEL);
            }
        }
    }

    fn draw_strings(&self, painter: &egui::Painter, inner: Rect, shown: usize) {
        let row = inner.height() / shown as f32;
        let (x0, x1) = (inner.left() + 34.0, inner.right() - 12.0);
        for k in 0..shown {
            let y = inner.top() + (k as f32 + 0.5) * row;
            let name = if self.guitar { NOTE_NAMES[k % 6].to_string() } else { format!("{}", k + 1) };
            painter.text(pos2(inner.left() + 4.0, y), Align2::LEFT_CENTER, name, FontId::proportional(12.0), LABEL);
            for x in [x0, x1] {
                painter.circle_filled(pos2(x, y), 2.5, Color32::from_gray(160));
            }
            let scale = 1.4 * AMP * row;
            let clamp = 0.48 * row;
            let shape = |a: f32| -> Vec<Pos2> {
                (0..=48)
                    .map(|i| {
                        let s = i as f32 / 48.0;
                        pos2(x0 + s * (x1 - x0), y - (a * (PI as f32 * s).sin()).clamp(-clamp, clamp))
                    })
                    .collect()
            };
            // the envelope: the amplitude now, or the allowed ones
            let envelopes: Vec<f32> = if self.quantum {
                (1..4).map(|n| (n as f64 * self.eps).sqrt() as f32 * scale).filter(|a| *a < clamp).collect()
            } else {
                vec![self.modes.energy(k, self.eps).sqrt() as f32 * scale]
            };
            for a in envelopes {
                for s in [-1.0, 1.0] {
                    painter.add(Shape::line(shape(s * a), Stroke::new(1.0, Color32::from_gray(60))));
                }
            }
            let a = self.modes.displacement(k, self.theta(k), self.eps) as f32 * scale;
            painter.add(Shape::line(shape(a), Stroke::new(2.0, self.color(k))));
        }
    }

    fn plots_ui(&mut self, ui: &mut egui::Ui) {
        let full = ui.available_rect_before_wrap();
        ui.allocate_rect(full, Sense::hover());
        let cols = plot::columns(full.shrink2(vec2(0.0, 4.0)), if self.quantum { 4 } else { 3 }, 8.0);
        let (mean, c) = self.theory();
        let kt = self.kt;

        // the energy distribution
        let h = &self.hist;
        if self.quantum {
            let eps = self.eps;
            let bars = h.bins.len();
            let fr: Vec<f64> = (0..bars).map(|k| h.fraction(k)).collect();
            let p0 = quantum_probability(0, kt, eps);
            let y_max = p0.max(fr.iter().cloned().fold(0.0, f64::max)) * 1.2;
            let e_top = (bars as f64 - 0.5) * eps;
            let p = Plot::new(ui, cols[0], "P(n): the fraction of modes with n quanta", "energy E = n ε", (-0.5 * eps, e_top), (0.0, y_max), false);
            for (n, &f) in fr.iter().enumerate() {
                let e = n as f64 * eps;
                p.bar(e - 0.3 * eps, e + 0.3 * eps, f, SIM.gamma_multiply(0.8));
            }
            // P(n) = e^{−E_n/k_BT}/Z₁ lies on the Boltzmann factor
            p.curve(&plot::lin_grid(0.0, e_top, 200, |e| p0 * (-e / kt).exp()), Stroke::new(1.0, THEORY.gamma_multiply(0.5)));
            for n in 0..bars {
                p.dot(n as f64 * eps, quantum_probability(n as u32, kt, eps), 3.5, THEORY);
            }
            p.dashed_vline(mean, Stroke::new(1.0, NOW));
            p.label(mean, y_max * 0.97, Align2::LEFT_TOP, " ⟨E⟩", NOW);
            p.legend(&[("simulation", SIM, false), ("(1 − e^(−ε/k_BT)) e^(−nε/k_BT)", THEORY, false)]);
        } else {
            let dens: Vec<f64> = (0..h.bins.len()).map(|k| h.density(k)).collect();
            let y_max = (1.0 / kt).max(dens.iter().cloned().fold(0.0, f64::max)) * 1.15;
            let p = Plot::new(ui, cols[0], "energy distribution P(E)", "energy E of one mode", (h.lo, h.hi), (0.0, y_max), false);
            for (k, &v) in dens.iter().enumerate() {
                p.bar(h.lo + k as f64 * h.width(), h.lo + (k + 1) as f64 * h.width(), v, SIM.gamma_multiply(0.8));
            }
            p.curve(&plot::lin_grid(h.lo, h.hi, 200, |e| (-e / kt).exp() / kt), Stroke::new(2.0, THEORY));
            p.dashed_vline(kt, Stroke::new(1.0, NOW));
            p.label(kt, y_max * 0.97, Align2::LEFT_TOP, " ⟨E⟩ = k_BT", NOW);
            p.legend(&[("simulation", SIM, false), ("e^(−E/k_BT)/k_BT", THEORY, false)]);
        }

        // the mean energy in time, and one mode's
        let m = self.modes.len().max(1) as f64;
        let sd = c.sqrt() * kt / m.sqrt();
        let y_max = (3.0 * mean).max(mean + 3.0 * sd).max(self.trace.max(0)).max(0.05 * kt) * 1.1;
        let p = Plot::new(ui, cols[1], "energy per mode", "time (s)", (-self.trace.window, 0.0), (0.0, y_max), false);
        p.band(mean - sd, mean + sd, NOW.gamma_multiply(0.18));
        let single: Vec<_> = self.trace.points.iter().map(|&(t, v)| (t - self.time, v[1])).collect();
        p.curve(&single, Stroke::new(1.0, SINGLE));
        p.hline(mean, Stroke::new(1.5, THEORY));
        let pts: Vec<_> = self.trace.points.iter().map(|&(t, v)| (t - self.time, v[0])).collect();
        p.curve(&pts, Stroke::new(2.0, SIM));
        p.legend(&[("mean of all M modes", SIM, false), ("mode 1 alone", SINGLE, false), ("⟨E⟩ ± σ/√M", THEORY, false)]);

        // ⟨E⟩ (and C) against the temperature
        let measured = self.energy_m.mean();
        if self.quantum {
            let eps = self.eps;
            let t = kt / eps;
            let t_max = (1.15 * t).max(3.0);
            let quantum = plot::lin_grid(0.005, t_max, 240, |x| mean_quanta(x, 1.0));
            let p = Plot::new(ui, cols[2], "⟨E⟩/ε per mode", "k_BT/ε", (0.0, t_max), (0.0, t_max * 1.05), false);
            p.dashed(&[(0.0, 0.0), (t_max, t_max)], Stroke::new(1.5, CLASSICAL));
            p.curve(&quantum, Stroke::new(2.0, THEORY));
            p.vline(t, Stroke::new(1.0, NOW));
            p.dot(t, mean / eps, 5.0, NOW);
            if let Some(e) = measured {
                p.dot(t, e / eps, 3.5, SIM);
            }
            p.legend(&[("quantum: 1/(e^(ε/k_BT) − 1)", THEORY, false), ("classical: k_BT/ε", CLASSICAL, true)]);
            let p = Plot::new(ui, cols[3], "heat capacity C/k_B per mode", "k_BT/ε", (0.0, t_max), (0.0, 1.25), false);
            p.dashed(&[(0.0, 1.0), (t_max, 1.0)], Stroke::new(1.5, CLASSICAL));
            p.curve(&plot::lin_grid(0.005, t_max, 240, |x| quantum_heat_capacity(x, 1.0)), Stroke::new(2.0, THEORY));
            p.vline(t, Stroke::new(1.0, NOW));
            p.dot(t, c, 5.0, NOW);
            if let Some(v) = self.energy_m.var() {
                p.dot(t, v / (kt * kt), 3.5, SIM);
            }
            p.legend(&[("quantum", THEORY, false), ("classical", CLASSICAL, true), ("from the spread", SIM, false)]);
        } else {
            let p = Plot::new(ui, cols[2], "⟨E⟩ per mode: equipartition", "k_BT", (0.0, KT_MAX), (0.0, KT_MAX * 1.05), false);
            p.curve(&[(0.0, 0.0), (KT_MAX, KT_MAX)], Stroke::new(2.0, THEORY));
            p.vline(kt, Stroke::new(1.0, NOW));
            p.dot(kt, kt, 5.0, NOW);
            if let Some(e) = measured {
                p.dot(kt, e, 3.5, SIM);
            }
            p.legend_left(&[("⟨E⟩ = k_BT, whatever the frequency", THEORY, false), ("simulation", SIM, false)]);
        }
    }
}

/// a zigzag spring from a to b (horizontal)
fn spring(a: Pos2, b: Pos2, width: f32, stroke: Stroke) -> Shape {
    let turns = 5;
    let mut pts = vec![a];
    let len = b.x - a.x;
    let lead = 0.12 * len;
    pts.push(pos2(a.x + lead, a.y));
    for i in 0..2 * turns {
        let s = (i as f32 + 0.5) / (2 * turns) as f32;
        let side = if i % 2 == 0 { -1.0 } else { 1.0 };
        pts.push(pos2(a.x + lead + s * (len - 2.0 * lead), a.y + side * width));
    }
    pts.push(pos2(b.x - lead, b.y));
    pts.push(b);
    Shape::line(pts, stroke)
}
