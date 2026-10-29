//! User interface of example 6: the modes of the box as points in the octant
//! of k-space (drag to turn, coloured by their energy), the density of
//! states, the energy per mode, the spectrum (Rayleigh–Jeans or Planck) and
//! the total energy against the cutoff.

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Sense, Shape, Stroke, pos2, vec2};

use crate::cavity::{Cavity, CavityPreset, MAX_R, WIEN_X, density_of_states, modes_below, planck_mean, planck_total_below, stefan_boltzmann};
use crate::plot::{self, BG, CLASSICAL, LABEL, NOW, Plot, SIM, THEORY, fmt};
use crate::stats::Avg;
use crate::widgets;

const KT_MIN: f64 = 0.3;
const KT_MAX: f64 = 8.0;
/// points up to this |n| are drawn
const R_DRAW: f64 = 18.0;
/// bin width of the plots (units of ε₁)
const BIN: f64 = 1.0;

/// An orthographic camera: turned by `yaw` about the vertical, then tilted by `pitch`.
struct Cam {
    yaw: f32,
    pitch: f32,
    scale: f32,
    centre: Pos2,
}

impl Cam {
    /// screen position and depth of a point (x right, y up, z towards the viewer)
    fn project(&self, p: [f32; 3]) -> (Pos2, f32) {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let x1 = p[0] * cy + p[2] * sy;
        let z1 = -p[0] * sy + p[2] * cy;
        let y2 = p[1] * cp - z1 * sp;
        let z2 = p[1] * sp + z1 * cp;
        (self.centre + vec2(x1, -y2) * self.scale, z2)
    }
}

pub struct CavityUi {
    example: CavityPreset,
    quantum: bool,
    kt: f64,
    cutoff: f64,
    /// the inner radius of the outlined shell (units of ε₁)
    shell: f64,
    notes: &'static str,
    cav: Cavity,
    running: bool,
    speed: f64,
    avg: Avg,
    cam: (f32, f32),
    /// time-averaged energy in each bin of ε, and the weight of the average
    bin_energy: Vec<f64>,
    weight: f64,
    /// modes in each bin
    bin_modes: Vec<f64>,
}

impl Default for CavityUi {
    fn default() -> Self {
        let mut ui = CavityUi {
            example: CavityPreset::Counting,
            quantum: true,
            kt: 3.0,
            cutoff: 16.0,
            shell: 10.0,
            notes: "",
            cav: Cavity::new(true, 16.0, 3.0),
            running: true,
            speed: 1.0,
            avg: Avg::Short,
            cam: (0.75, 0.45),
            bin_energy: vec![],
            weight: 0.0,
            bin_modes: vec![],
        };
        ui.load_preset(CavityPreset::Counting);
        ui
    }
}

impl CavityUi {
    pub fn load_preset(&mut self, p: CavityPreset) {
        let s = p.setup();
        self.example = p;
        self.quantum = s.quantum;
        self.kt = s.kt;
        self.cutoff = s.cutoff;
        self.notes = s.notes;
        self.shell = (WIEN_X * s.kt).round().clamp(2.0, s.cutoff - 1.0);
        self.cav = Cavity::new(s.quantum, s.cutoff, s.kt);
        self.clear_stats();
    }

    fn bins(&self) -> usize {
        (MAX_R / BIN).ceil() as usize + 1
    }

    fn clear_stats(&mut self) {
        let nb = self.bins();
        self.bin_energy = vec![0.0; nb];
        self.weight = 0.0;
        self.bin_modes = vec![0.0; nb];
        for k in 0..self.cav.active {
            self.bin_modes[(self.cav.eps[k] / BIN) as usize] += 1.0;
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let dt = widgets::frame_dt(ui);
        if self.running {
            self.cav.step(self.kt, dt * self.speed);
            self.record(dt);
            ui.ctx().request_repaint();
        }
        let win_h = ui.ctx().content_rect().height();
        egui::Panel::left("cav_side").resizable(true).default_size(340.0).min_size(260.0).show(ui, |ui| self.side_ui(ui));
        egui::Panel::bottom("cav_plots").resizable(true).default_size(win_h * 0.4).min_size(150.0).max_size(win_h * 0.65).show(ui, |ui| self.plots_ui(ui));
        egui::CentralPanel::default().show(ui, |ui| self.octant_ui(ui));
    }

    fn record(&mut self, dt: f64) {
        let tau = self.avg.tau();
        let keep = if tau <= 0.0 { 0.0 } else { (-dt / tau).exp() };
        self.bin_energy.iter_mut().for_each(|b| *b *= keep);
        self.weight = self.weight * keep + 1.0;
        for k in 0..self.cav.active {
            self.bin_energy[(self.cav.eps[k] / BIN) as usize] += self.cav.energy(k);
        }
    }

    /// the simulation's average energy in bin b
    fn sim_bin(&self, b: usize) -> f64 {
        if self.weight > 0.0 { self.bin_energy[b] / self.weight } else { 0.0 }
    }

    fn side_ui(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            widgets::heading(ui, "6 · Light in a box", "g(ε) = π ε²/ε₁³,   ε = hν = ε₁|n|");
            if let Some(p) = widgets::examples(ui, "cav_example", self.example, &CavityPreset::ALL, CavityPreset::label) {
                self.load_preset(p);
            }
            widgets::section(ui, "SETTINGS (energies in units of ε₁ = hc/2L)");
            let old = (self.kt, self.cutoff, self.quantum);
            widgets::kt_slider(ui, &mut self.kt, KT_MIN..=KT_MAX);
            ui.horizontal(|ui| {
                ui.label("mode energies:");
                ui.selectable_value(&mut self.quantum, false, "continuous").on_hover_text("classical modes: Rayleigh–Jeans");
                ui.selectable_value(&mut self.quantum, true, "quantised, steps hν").on_hover_text("Planck");
            });
            ui.add(egui::Slider::new(&mut self.cutoff, 4.0..=MAX_R).text("modes up to |n| =").custom_formatter(|v, _| fmt(v)))
                .on_hover_text("only modes up to this |n| are in the simulation; classically the total energy grows with it");
            ui.add(egui::Slider::new(&mut self.shell, 1.0..=MAX_R - BIN).text("outlined shell at ε").custom_formatter(|v, _| fmt(v)));
            widgets::play_row(ui, &mut self.running, &mut self.speed, 0.1..=5.0, "speed");
            ui.horizontal(|ui| {
                if ui.button("take all energy out").clicked() {
                    self.cav.freeze();
                }
                if ui.button("equilibrium now").clicked() {
                    self.cav.equilibrate(self.kt);
                }
            });
            if widgets::avg_row(ui, &mut self.avg) {
                self.clear_stats();
            }
            if (self.kt, self.cutoff, self.quantum) != old {
                if self.quantum != old.2 {
                    self.cav.quantum = self.quantum;
                    self.cav.equilibrate(self.kt);
                }
                self.cav.set_cutoff(self.cutoff, self.kt);
                self.clear_stats();
            }

            widgets::section(ui, "NUMBERS");
            let kt = self.kt;
            let n = self.cav.active as f64;
            let total: f64 = (0..self.bins()).map(|b| self.sim_bin(b)).sum();
            let lo = (self.shell / BIN).floor() as usize;
            let rows = [
                ("modes in the box", Some(modes_below(self.cutoff)), Some(n), "formula: π ε³/3ε₁³ (continuum); simulation: counted, with polarisations"),
                (
                    "energy of these modes",
                    Some(self.cav.expected_total(kt)),
                    (self.weight > 0.0).then_some(total),
                    "k_BT per mode, or Planck's ⟨E⟩ per mode, summed",
                ),
                ("Rayleigh–Jeans total", Some(kt * n), None, "k_BT × the number of modes: grows without end with the cutoff"),
                ("Stefan–Boltzmann (all modes)", Some(stefan_boltzmann(kt)), None, "π⁵ (k_BT)⁴/15 ε₁³: Planck, summed over infinitely many modes"),
                ("spectrum peak (Wien)", Some(WIEN_X * kt), None, "ε = 2.82 k_BT"),
                ("modes in the shell", Some(density_of_states(lo as f64 * BIN + 0.5 * BIN) * BIN), Some(self.bin_modes[lo]), "π ε² dε/ε₁³ against the count"),
            ];
            widgets::numbers(ui, "cav_numbers", &rows);
            widgets::notes(ui, self.notes);
        });
    }

    fn octant_ui(&mut self, ui: &mut egui::Ui) {
        let full = ui.available_rect_before_wrap();
        let resp = ui.allocate_rect(full, Sense::drag());
        let painter = ui.painter_at(full);
        painter.rect_filled(full, 3.0, BG);
        painter.text(full.min + vec2(8.0, 5.0), Align2::LEFT_TOP, "MODES · one point per (n_x, n_y, n_z), drag to turn", FontId::proportional(13.0), LABEL);
        painter.text(
            full.right_top() + vec2(-8.0, 5.0),
            Align2::RIGHT_TOP,
            format!("colour: energy of the mode (0 → 3 k_BT){}", if self.cutoff > R_DRAW { format!(" · drawn up to |n| = {R_DRAW}") } else { String::new() }),
            FontId::proportional(12.0),
            LABEL,
        );
        if resp.dragged() {
            let d = resp.drag_delta();
            self.cam.0 += d.x * 0.008;
            self.cam.1 = (self.cam.1 + d.y * 0.008).clamp(-1.4, 1.4);
        }
        let inner = full.shrink2(vec2(20.0, 30.0));
        if inner.width() < 40.0 || inner.height() < 40.0 {
            return;
        }
        let r_draw = self.cutoff.min(R_DRAW);
        let cam = Cam { yaw: self.cam.0, pitch: self.cam.1, scale: inner.width().min(inner.height()) / (1.9 * r_draw as f32), centre: inner.center() };
        // n_x to the right, n_z up, n_y towards the viewer (before turning); centred on the middle of the octant
        let c = 0.5 * r_draw as f32;
        let w = |n: [f32; 3]| [n[0] - c, n[2] - c, n[1] - c];
        let axis = Stroke::new(1.5, Color32::from_gray(140));
        for (k, name) in ["n_x", "n_y", "n_z"].iter().enumerate() {
            let mut tip = [0.0f32; 3];
            tip[k] = 1.1 * r_draw as f32;
            let (a, _) = cam.project(w([0.0; 3]));
            let (b, _) = cam.project(w(tip));
            painter.line_segment([a, b], axis);
            painter.text(b, Align2::CENTER_CENTER, *name, FontId::proportional(13.0), LABEL);
        }
        // the shell between ε and ε + dε: quarter circles in the three coordinate planes
        let arc = |r: f64, i: usize, j: usize| -> Vec<Pos2> {
            (0..=24)
                .map(|s| {
                    let t = s as f64 / 24.0 * std::f64::consts::FRAC_PI_2;
                    let mut p = [0.0f32; 3];
                    p[i] = (r * t.cos()) as f32;
                    p[j] = (r * t.sin()) as f32;
                    cam.project(w(p)).0
                })
                .collect()
        };
        let lo = (self.shell / BIN).floor() * BIN;
        for r in [lo, lo + BIN] {
            if r <= r_draw {
                for (i, j) in [(0, 1), (1, 2), (0, 2)] {
                    painter.add(Shape::line(arc(r, i, j), Stroke::new(1.2, NOW.gamma_multiply(0.8))));
                }
            }
        }
        // the points, far ones first; a point's energy is that of its polarisations together, per mode
        let mut pts: Vec<(Pos2, f32, f64, bool)> = vec![];
        let mut k = 0;
        while k < self.cav.active && self.cav.eps[k] <= r_draw {
            let n = self.cav.n[k];
            let mut e = self.cav.energy(k);
            let mut count = 1.0;
            if k + 1 < self.cav.active && self.cav.n[k + 1] == n {
                e += self.cav.energy(k + 1);
                count = 2.0;
                k += 1;
            }
            let (p, z) = cam.project(w([n[0] as f32, n[1] as f32, n[2] as f32]));
            let eps = self.cav.eps[k];
            pts.push((p, z, e / count, eps >= lo && eps < lo + BIN));
            k += 1;
        }
        pts.sort_by(|a, b| a.1.total_cmp(&b.1));
        let r0 = (cam.scale * 0.22).clamp(1.2, 5.0);
        for (p, _, e, in_shell) in pts {
            let t = (e / (3.0 * self.kt)) as f32;
            let r = r0 * (0.45 + 0.55 * t.min(1.0));
            painter.circle_filled(p, r, plot::heat(t).gamma_multiply(0.35 + 0.65 * t.min(1.0)));
            if in_shell {
                painter.circle_stroke(p, r + 1.5, Stroke::new(1.0, NOW));
            }
        }
        painter.text(
            pos2(full.left() + 8.0, full.bottom() - 6.0),
            Align2::LEFT_BOTTOM,
            format!(
                "{} modes up to |n| = {} (two polarisations per point, one if an n_i = 0)  ·  orange: the shell from ε = {} to {}",
                self.cav.active,
                fmt(self.cutoff),
                fmt(lo),
                fmt(lo + BIN)
            ),
            FontId::proportional(12.0),
            LABEL,
        );
    }

    fn plots_ui(&mut self, ui: &mut egui::Ui) {
        let full = ui.available_rect_before_wrap();
        ui.allocate_rect(full, Sense::hover());
        let cols = plot::columns(full.shrink2(vec2(0.0, 4.0)), 4, 8.0);
        let kt = self.kt;
        let top = self.cutoff.ceil();
        let nb = (top / BIN).ceil() as usize;
        let lo = (self.shell / BIN).floor() as usize;

        // density of states: the count against the continuum
        let g_max = density_of_states(top) * 1.15;
        let p = Plot::new(ui, cols[0], "density of states g(ε): modes per ε₁", "photon energy ε = hν", (0.0, top), (0.0, g_max), false);
        for b in 0..nb {
            let color = if b == lo { NOW } else { SIM.gamma_multiply(0.8) };
            p.bar(b as f64 * BIN, (b + 1) as f64 * BIN, self.bin_modes[b] / BIN, color);
        }
        p.curve(&plot::lin_grid(0.0, top, 200, density_of_states), Stroke::new(2.0, THEORY));
        p.legend_left(&[("modes in the box, counted", SIM, false), ("π ε²/ε₁³  (8πν²V/c³)", THEORY, false)]);

        // energy per mode
        let p = Plot::new(ui, cols[1], "energy per mode ⟨E⟩", "photon energy ε = hν", (0.0, top), (0.0, 1.35 * kt), false);
        for b in 0..nb {
            if self.bin_modes[b] > 0.0 {
                p.bar(b as f64 * BIN, (b + 1) as f64 * BIN, self.sim_bin(b) / self.bin_modes[b], SIM.gamma_multiply(0.8));
            }
        }
        p.dashed(&[(0.0, kt), (top, kt)], Stroke::new(1.5, CLASSICAL));
        p.curve(&plot::lin_grid(0.0, top, 200, |e| planck_mean(e, kt)), Stroke::new(2.0, THEORY));
        p.legend(&[("simulation", SIM, false), ("classical: k_BT", CLASSICAL, true), ("Planck: ε/(e^(ε/k_BT) − 1)", THEORY, false)]);

        // the spectrum
        let planck = |e: f64| density_of_states(e) * planck_mean(e, kt);
        let peak = planck(WIEN_X * kt);
        let y_max = peak * 1.8;
        let p = Plot::new(ui, cols[2], "spectrum: energy per ε₁, g(ε)⟨E⟩", "photon energy ε = hν", (0.0, top), (0.0, y_max), false);
        for b in 0..nb {
            p.bar(b as f64 * BIN, (b + 1) as f64 * BIN, self.sim_bin(b) / BIN, SIM.gamma_multiply(0.8));
        }
        p.dashed(&plot::lin_grid(0.0, top, 200, |e| density_of_states(e) * kt), Stroke::new(1.5, CLASSICAL));
        p.curve(&plot::lin_grid(0.0, top, 200, planck), Stroke::new(2.0, THEORY));
        p.dashed_vline(WIEN_X * kt, Stroke::new(1.0, NOW));
        p.label(WIEN_X * kt, y_max * 0.62, Align2::LEFT_TOP, " 2.82 k_BT (Wien)", NOW);
        p.legend(&[("simulation", SIM, false), ("Rayleigh–Jeans: g k_BT → ∞", CLASSICAL, true), ("Planck", THEORY, false)]);

        // total energy against the cutoff
        let sb = stefan_boltzmann(kt);
        let y_max = sb * 1.6;
        let p = Plot::new(ui, cols[3], "total energy of the modes below ε", "cutoff ε (|n| of the last mode)", (0.0, MAX_R), (0.0, y_max), false);
        p.dashed(&plot::lin_grid(0.0, MAX_R, 200, |e| kt * modes_below(e)), Stroke::new(1.5, CLASSICAL));
        p.curve(&plot::lin_grid(0.0, MAX_R, 120, |e| planck_total_below(e, kt)), Stroke::new(2.0, THEORY));
        p.hline(sb, Stroke::new(1.0, THEORY.gamma_multiply(0.4)));
        p.vline(self.cutoff, Stroke::new(1.0, NOW));
        if self.weight > 0.0 {
            let total: f64 = (0..self.bins()).map(|b| self.sim_bin(b)).sum();
            if total < y_max {
                p.dot(self.cutoff, total, 4.5, SIM);
            } else {
                p.label(self.cutoff, y_max * 0.75, Align2::RIGHT_TOP, &format!("simulation: {} ↑ ", fmt(total)), SIM);
            }
        }
        p.legend_left(&[("classical: k_BT per mode", CLASSICAL, true), ("Planck → Stefan–Boltzmann", THEORY, false), ("simulation", SIM, false)]);
    }
}
