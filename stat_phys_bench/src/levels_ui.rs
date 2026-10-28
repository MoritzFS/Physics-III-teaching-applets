//! User interface of example 1: the levels with their particles (drag a
//! level up or down), the occupations next to them on the same energy axis,
//! and ⟨E⟩ and C against k_BT.

use eframe::egui::{self, Align2, Color32, CursorIcon, FontId, Rect, RichText, Sense, Shape, Stroke, pos2, vec2};

use crate::levels::{E_TOP, Level, LevelPreset, MAX_G, MAX_LEVELS, MAX_PARTICLES, Particles, populations, thermo};
use crate::plot::{self, BG, LABEL, NOW, Plot, SIM, THEORY, fmt};
use crate::stats::{Smooth, Trace};
use crate::widgets;

const KT_MIN: f64 = 0.02;
const KT_MAX: f64 = 100.0;
const DOT: Color32 = Color32::from_rgb(140, 190, 255);

pub struct LevelsUi {
    pub example: LevelPreset,
    levels: Vec<Level>,
    n: usize,
    kt: f64,
    per_box: bool,
    notes: &'static str,
    parts: Particles,
    running: bool,
    /// tries per particle per second
    rate: f64,
    drag: Option<usize>,
    /// the top of the drawn energy axis (kept while a level is dragged)
    e_view: f64,
    trace: Trace,
    time: f64,
    mean: Smooth,
    c: Smooth,
}

impl Default for LevelsUi {
    fn default() -> Self {
        Self::with(LevelPreset::TwoLevels)
    }
}

impl LevelsUi {
    fn with(p: LevelPreset) -> Self {
        let s = p.setup();
        LevelsUi {
            example: p,
            parts: Particles::new(s.n, &s.levels, s.kt),
            levels: s.levels,
            n: s.n,
            kt: s.kt,
            per_box: s.per_box,
            notes: s.notes,
            running: true,
            rate: 2.0,
            drag: None,
            e_view: E_TOP,
            trace: Trace::new(30.0),
            time: 0.0,
            mean: Smooth::default(),
            c: Smooth::default(),
        }
    }

    pub fn load_preset(&mut self, p: LevelPreset) {
        let (rate, running) = (self.rate, self.running);
        *self = Self::with(p);
        self.rate = rate;
        self.running = running;
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let dt = widgets::frame_dt(ui);
        if self.running {
            self.parts.step(&self.levels, self.kt, self.rate * dt);
            self.time += dt;
            ui.ctx().request_repaint();
        }
        let (mean, var) = self.parts.energy_stats(&self.levels);
        self.mean.update(mean, dt, 1.0);
        self.c.update(var / (self.kt * self.kt), dt, 1.0);
        if self.running {
            self.trace.push(self.time, [mean, f64::NAN, f64::NAN]);
        }
        let win_h = ui.ctx().content_rect().height();
        egui::Panel::left("lv_side").resizable(true).default_size(340.0).min_size(260.0).show(ui, |ui| self.side_ui(ui));
        egui::Panel::bottom("lv_plots").resizable(true).default_size(win_h * 0.33).min_size(150.0).max_size(win_h * 0.6).show(ui, |ui| self.plots_ui(ui));
        egui::CentralPanel::default().show(ui, |ui| self.diagram_ui(ui));
    }

    fn set_level_count(&mut self, count: usize) {
        while self.levels.len() < count {
            let top = self.levels.iter().map(|l| l.e).fold(0.0, f64::max);
            self.levels.push(Level { e: (top + 1.0).min(E_TOP), g: 1 });
        }
        self.levels.truncate(count);
        self.parts.fit(&self.levels);
    }

    fn side_ui(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            widgets::heading(ui, "1 · Boxes of fixed energy", "Z₁ = Σ g_i e^(−E_i/k_BT)");
            if let Some(p) = widgets::examples(ui, "lv_example", self.example, &LevelPreset::ALL, LevelPreset::label) {
                self.load_preset(p);
            }
            widgets::section(ui, "SETTINGS");
            widgets::kt_slider(ui, &mut self.kt, KT_MIN..=KT_MAX);
            let mut n = self.n;
            if ui.add(egui::Slider::new(&mut n, 1..=MAX_PARTICLES).logarithmic(true).text("particles N")).changed() {
                self.n = n;
                self.parts.resize(n, &self.levels, self.kt);
                self.trace.clear();
            }
            let mut count = self.levels.len();
            if ui.add(egui::Slider::new(&mut count, 1..=MAX_LEVELS).text("levels")).changed() {
                self.set_level_count(count);
            }
            widgets::play_row(ui, &mut self.running, &mut self.rate, 0.1..=30.0, "hops per particle per s");
            ui.horizontal(|ui| {
                ui.label("bars:");
                ui.selectable_value(&mut self.per_box, false, "per level");
                ui.selectable_value(&mut self.per_box, true, "per box");
            });
            ui.horizontal(|ui| {
                if ui.button("all to the top level").on_hover_text("start far from equilibrium and watch the bath sort the particles").clicked() {
                    let top = (0..self.levels.len()).max_by(|&a, &b| self.levels[a].e.total_cmp(&self.levels[b].e)).unwrap_or(0);
                    let g = self.levels[top].g;
                    for k in 0..self.parts.level.len() {
                        self.parts.level[k] = top;
                        self.parts.slot[k] = k as u32 % g;
                    }
                }
                if ui.button("equilibrium now").clicked() {
                    self.parts.equilibrate(&self.levels, self.kt);
                }
            });

            widgets::section(ui, "LEVELS (drag them in the diagram)");
            let counts = self.parts.counts(self.levels.len());
            let mut boxes_changed = false;
            egui::Grid::new("lv_levels").num_columns(4).spacing([12.0, 2.0]).striped(true).show(ui, |ui| {
                ui.weak("level");
                ui.weak("energy E");
                ui.weak("boxes g");
                ui.weak("particles");
                ui.end_row();
                for (i, l) in self.levels.iter_mut().enumerate() {
                    ui.label(format!("{i}"));
                    ui.add(egui::DragValue::new(&mut l.e).range(0.0..=E_TOP).speed(0.01).fixed_decimals(2));
                    boxes_changed |= ui.add(egui::DragValue::new(&mut l.g).range(1..=MAX_G).speed(0.05)).changed();
                    ui.label(RichText::new(format!("{}", counts[i])).color(SIM));
                    ui.end_row();
                }
            });
            if boxes_changed {
                self.parts.fit(&self.levels);
            }

            widgets::section(ui, "NUMBERS (per particle)");
            let th = thermo(&self.levels, self.kt);
            widgets::numbers(
                ui,
                "lv_numbers",
                &[
                    ("Z₁", Some(th.z), None, "the sum of the Boltzmann factors of all boxes"),
                    ("⟨E⟩", Some(th.mean), self.mean.get(), "simulation: the mean energy of the particles now"),
                    (
                        "C/k_B",
                        Some(th.c),
                        self.c.get(),
                        "simulation: from the spread of the particles' energies, C = Var(E)/k_BT² (fluctuations and heat capacity are the same thing)",
                    ),
                    ("boxes Σ g_i", Some(self.levels.iter().map(|l| l.g as f64).sum()), None, "the hot limit of Z₁"),
                ],
            );
            widgets::notes(ui, self.notes);
        });
    }

    fn diagram_ui(&mut self, ui: &mut egui::Ui) {
        let full = ui.available_rect_before_wrap();
        let resp = ui.allocate_rect(full, Sense::click_and_drag());
        let painter = ui.painter_at(full);
        painter.rect_filled(full, 3.0, BG);
        painter.text(full.min + vec2(8.0, 5.0), Align2::LEFT_TOP, "LEVELS AND PARTICLES · drag a level up or down", FontId::proportional(13.0), LABEL);
        let split = full.left() + full.width() * 0.62;
        let dia = Rect::from_min_max(full.min + vec2(52.0, 44.0), pos2(split - 34.0, full.bottom() - 34.0));
        let occ = Rect::from_min_max(pos2(split + 10.0, dia.top()), pos2(full.right() - 16.0, dia.bottom()));
        if dia.width() < 40.0 || dia.height() < 40.0 {
            return;
        }
        // the energy axis reaches a bit above the highest level; it stays put while a level is dragged
        if self.drag.is_none() {
            let top = self.levels.iter().map(|l| l.e).fold(0.5, f64::max) * 1.25;
            let step = plot::nice_step(top, 5.0);
            self.e_view = ((top / step).ceil() * step).min(E_TOP);
        }
        let e_view = self.e_view;
        let ey = |e: f64| dia.bottom() - (e / e_view) as f32 * dia.height();
        let font = FontId::proportional(11.0);

        // energy axis, shared by both parts
        let step = plot::nice_step(e_view, (dia.height() / 40.0).clamp(2.0, 10.0) as f64);
        for e in plot::ticks(0.0, e_view, step) {
            let y = ey(e);
            for r in [dia, occ] {
                painter.line_segment([pos2(r.left(), y), pos2(r.right(), y)], Stroke::new(1.0, plot::GRID));
            }
            painter.text(pos2(dia.left() - 8.0, y), Align2::RIGHT_CENTER, fmt(e), font.clone(), LABEL);
        }
        painter.text(pos2(dia.left() - 8.0, dia.top() - 16.0), Align2::RIGHT_CENTER, "E", FontId::proportional(13.0), LABEL);

        // k_BT as a length on the energy axis
        let x_kt = dia.left() - 30.0;
        let top = ey(self.kt.min(e_view));
        painter.line_segment([pos2(x_kt, ey(0.0)), pos2(x_kt, top)], Stroke::new(2.5, NOW));
        if self.kt <= e_view {
            painter.line_segment([pos2(x_kt - 5.0, top), pos2(x_kt + 5.0, top)], Stroke::new(2.5, NOW));
        } else {
            painter.add(Shape::convex_polygon(vec![pos2(x_kt, top - 7.0), pos2(x_kt - 5.0, top), pos2(x_kt + 5.0, top)], NOW, Stroke::NONE));
        }
        painter.text(pos2(x_kt, ey(0.0) + 4.0), Align2::CENTER_TOP, "k_BT", font.clone(), NOW);

        // dragging a level
        let near = |y: f32, levels: &[Level]| -> Option<usize> {
            levels.iter().enumerate().map(|(i, l)| (i, (ey(l.e) - y).abs())).filter(|&(_, d)| d < 9.0).min_by(|a, b| a.1.total_cmp(&b.1)).map(|(i, _)| i)
        };
        let pointer = resp.hover_pos().filter(|p| p.x >= dia.left() - 4.0 && p.x <= dia.right() + 4.0);
        if resp.drag_started() {
            self.drag = resp.interact_pointer_pos().and_then(|p| near(p.y, &self.levels));
        }
        if let Some(i) = self.drag {
            if resp.dragged() {
                if let Some(p) = resp.interact_pointer_pos() {
                    let e = ((dia.bottom() - p.y) / dia.height()) as f64 * e_view;
                    self.levels[i].e = (e.clamp(0.0, e_view) * 100.0).round() / 100.0;
                }
            } else {
                self.drag = None;
            }
        }
        let hot = self.drag.or_else(|| pointer.and_then(|p| near(p.y, &self.levels)));
        if hot.is_some() {
            ui.ctx().set_cursor_icon(CursorIcon::ResizeVertical);
        }

        // the levels, split into their boxes
        let w = dia.width();
        let box_x = |b: u32, g: u32| -> (f32, f32) {
            let bw = w / g as f32;
            let gap = (bw * 0.1).min(8.0);
            (dia.left() + b as f32 * bw + 0.5 * gap, dia.left() + (b + 1) as f32 * bw - 0.5 * gap)
        };
        for (i, l) in self.levels.iter().enumerate() {
            let y = ey(l.e);
            let (width, color) = if hot == Some(i) { (3.5, Color32::WHITE) } else { (2.0, Color32::from_gray(200)) };
            for b in 0..l.g {
                let (x0, x1) = box_x(b, l.g);
                painter.line_segment([pos2(x0, y), pos2(x1, y)], Stroke::new(width, color));
            }
            if l.g > 1 {
                painter.text(pos2(dia.right() + 6.0, y), Align2::LEFT_CENTER, format!("×{}", l.g), font.clone(), LABEL);
            }
        }

        // the particles sit on their level, each at its own spot in its box
        let n = self.parts.level.len();
        let r = if n <= 100 {
            3.0
        } else if n <= 400 {
            2.2
        } else {
            1.6
        };
        let band = (dia.height() / 30.0).clamp(6.0, 14.0);
        for k in 0..n {
            let l = self.levels[self.parts.level[k]];
            let (x0, x1) = box_x(self.parts.slot[k], l.g);
            let [sx, sy] = self.parts.spot[k];
            let x = x0 + r + (x1 - x0 - 2.0 * r).max(0.0) * sx;
            let y = ey(l.e) - r - 1.5 - band * sy;
            painter.circle_filled(pos2(x, y), r, DOT);
        }

        // occupations on the same energy axis
        let counts = self.parts.counts(self.levels.len());
        let pop = populations(&self.levels, self.kt);
        let per = |i: usize| if self.per_box { self.levels[i].g as f64 } else { 1.0 };
        let theory: Vec<f64> = (0..self.levels.len()).map(|i| pop[i] / per(i)).collect();
        let shown: Vec<f64> = (0..self.levels.len()).map(|i| counts[i] as f64 / (n.max(1) as f64 * per(i))).collect();
        let x_step = plot::nice_step(theory.iter().cloned().fold(0.0, f64::max) * 1.25, 3.0);
        let x_max = ((theory.iter().cloned().fold(0.0, f64::max) * 1.25 / x_step).ceil() * x_step).max(x_step).min(1.0);
        let fx = |v: f64| occ.left() + (v / x_max) as f32 * occ.width();
        for t in plot::ticks(0.0, x_max, x_step) {
            let x = fx(t);
            painter.line_segment([pos2(x, occ.top()), pos2(x, occ.bottom())], Stroke::new(1.0, plot::GRID));
            painter.text(pos2(x, occ.bottom() + 3.0), Align2::CENTER_TOP, fmt(t), font.clone(), LABEL);
        }
        painter.line_segment([occ.left_top(), occ.left_bottom()], Stroke::new(1.0, plot::AXIS));
        let what = if self.per_box { "fraction of the particles per box" } else { "fraction of the particles per level" };
        painter.text(pos2(occ.center().x, occ.bottom() + 17.0), Align2::CENTER_TOP, what, FontId::proportional(12.0), LABEL);
        let clip = painter.with_clip_rect(occ.expand2(vec2(0.0, 10.0)));
        if self.per_box {
            // every box ends on e^{−E/k_BT}/Z₁
            let e_min = self.levels.iter().map(|l| l.e).fold(f64::INFINITY, f64::min);
            let sum: f64 = self.levels.iter().map(|l| l.g as f64 * (-(l.e - e_min) / self.kt).exp()).sum();
            let pts: Vec<_> =
                plot::lin_grid(0.0, e_view, 200, |e| (-(e - e_min) / self.kt).exp() / sum).iter().map(|&(e, v)| pos2(fx(v.min(10.0 * x_max)), ey(e))).collect();
            clip.add(Shape::line(pts, Stroke::new(1.5, THEORY.gamma_multiply(0.6))));
        }
        for i in 0..self.levels.len() {
            let y = ey(self.levels[i].e);
            clip.rect_filled(Rect::from_min_max(pos2(occ.left(), y - 4.0), pos2(fx(shown[i]), y + 4.0)), 0.0, SIM);
            let x = fx(theory[i]);
            clip.line_segment([pos2(x, y - 8.0), pos2(x, y + 8.0)], Stroke::new(2.5, THEORY));
        }
        // legend
        let lx = occ.right() - 4.0;
        let formula = if self.per_box { "e^(−E/k_BT)/Z₁" } else { "g e^(−E/k_BT)/Z₁" };
        painter.text(pos2(lx, occ.top() + 2.0), Align2::RIGHT_TOP, "particles now", font.clone(), SIM);
        painter.text(pos2(lx, occ.top() + 16.0), Align2::RIGHT_TOP, formula, font, THEORY);
    }

    fn plots_ui(&mut self, ui: &mut egui::Ui) {
        let full = ui.available_rect_before_wrap();
        ui.allocate_rect(full, Sense::hover());
        let cols = plot::columns(full.shrink2(vec2(0.0, 4.0)), 3, 8.0);
        let th = thermo(&self.levels, self.kt);
        let n = self.parts.level.len().max(1) as f64;

        // the mean energy of the particles in time
        let sd = (th.c / n).sqrt() * self.kt;
        let y_max = (th.mean + 3.0 * sd).max(self.trace.max(0)).max(0.05) * 1.2;
        let window = self.trace.window;
        let p = Plot::new(ui, cols[0], "mean energy per particle", "time (s)", (-window, 0.0), (0.0, y_max), false);
        p.band(th.mean - sd, th.mean + sd, NOW.gamma_multiply(0.18));
        p.hline(th.mean, Stroke::new(1.5, THEORY));
        let pts: Vec<_> = self.trace.points.iter().map(|&(t, v)| (t - self.time, v[0])).collect();
        p.curve(&pts, Stroke::new(1.5, SIM));
        p.legend(&[("particles now", SIM, false), ("⟨E⟩ ± σ/√N", THEORY, false)]);

        // ⟨E⟩ and C against k_BT
        let mean_curve = plot::log_grid(KT_MIN, KT_MAX, 240, |t| thermo(&self.levels, t).mean);
        let c_curve = plot::log_grid(KT_MIN, KT_MAX, 240, |t| thermo(&self.levels, t).c);
        let e_max = mean_curve.iter().map(|p| p.1).fold(0.0, f64::max).max(0.05) * 1.15;
        let p = Plot::new(ui, cols[1], "⟨E⟩ per particle", "k_BT", (KT_MIN, KT_MAX), (0.0, e_max), true);
        p.curve(&mean_curve, Stroke::new(2.0, THEORY));
        p.vline(self.kt, Stroke::new(1.0, NOW));
        p.dot(self.kt, th.mean, 5.0, NOW);
        if let Some(m) = self.mean.get() {
            p.dot(self.kt, m, 3.5, SIM);
        }
        let c_max = c_curve.iter().map(|p| p.1).fold(0.0, f64::max).max(0.2) * 1.25;
        let p = Plot::new(ui, cols[2], "heat capacity C/k_B per particle", "k_BT", (KT_MIN, KT_MAX), (0.0, c_max), true);
        p.curve(&c_curve, Stroke::new(2.0, THEORY));
        p.vline(self.kt, Stroke::new(1.0, NOW));
        p.dot(self.kt, th.c, 5.0, NOW);
        if let Some(c) = self.c.get() {
            p.dot(self.kt, c, 3.5, SIM);
        }
        p.legend(&[("formula", THEORY, false), ("from the spread now", SIM, false)]);
    }
}
