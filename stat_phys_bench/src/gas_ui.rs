//! User interface of examples 2 and 3: the box with the gas (drag to turn
//! it in 3D), the speed and energy distributions, the energy in time and, for
//! the two-level atoms, the heat capacity against k_BT.

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, pos2, vec2};

use crate::gas::{AtomPreset, Gas, GasParams, GasPreset, MAX_ATOMS, TwoLevel, Walls, component_density, energy_density, speed_density, speeds};
use crate::plot::{self, BG, CLASSICAL, EXCITED, GROUND, LABEL, NOW, Plot, SIM, THEORY, fmt};
use crate::stats::{Avg, Hist, Moments, Smooth, Trace};
use crate::widgets;

const KT_MIN: f64 = 0.05;
const KT_MAX: f64 = 10.0;
const BINS: usize = 24;
/// the box colour for a heat bath and for insulating walls
const BATH_WALL: Color32 = Color32::from_rgb(200, 120, 60);
const PLAIN_WALL: Color32 = Color32::from_gray(150);

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
}

/// what one example sets up
type Setup = (GasParams, bool, &'static str);

pub struct GasUi {
    /// example 3: atoms with a ground and an excited level
    atoms: bool,
    gas_example: GasPreset,
    atom_example: AtomPreset,
    p: GasParams,
    notes: &'static str,
    gas: Gas,
    running: bool,
    /// simulation time per second
    speed: f64,
    avg: Avg,
    show_vx: bool,
    arrows: bool,
    cam: (f32, f32),
    /// |v| of ground (or all) atoms, of excited atoms, v_x, kinetic energy
    hist_v: Hist,
    hist_v_exc: Hist,
    hist_vx: Hist,
    hist_e: Hist,
    speed_m: Moments,
    kinetic_m: Moments,
    /// kinetic + internal energy per atom, for C from the spread
    energy_m: Moments,
    excited_m: Moments,
    /// the mean kinetic energy per atom, smoothed: the gas's own temperature
    kinetic_now: Smooth,
    trace: Trace,
    time: f64,
    /// the (k_BT, m, d) the histogram ranges were made for
    ranges_for: (f64, f64, usize),
}

impl GasUi {
    pub fn new(atoms: bool) -> Self {
        let setup = if atoms { AtomPreset::Schottky.setup() } else { GasPreset::Maxwell.setup() };
        let p = setup.0;
        let mut ui = GasUi {
            atoms,
            gas_example: GasPreset::Maxwell,
            atom_example: AtomPreset::Schottky,
            p,
            notes: "",
            gas: Gas::new(&p, if atoms { 12 } else { 11 }),
            running: true,
            speed: 0.5,
            avg: Avg::Long,
            show_vx: false,
            arrows: false,
            cam: (-0.6, 0.35),
            hist_v: Hist::new(0.0, 1.0, BINS),
            hist_v_exc: Hist::new(0.0, 1.0, BINS),
            hist_vx: Hist::new(-1.0, 1.0, BINS),
            hist_e: Hist::new(0.0, 1.0, BINS),
            speed_m: Moments::default(),
            kinetic_m: Moments::default(),
            energy_m: Moments::default(),
            excited_m: Moments::default(),
            kinetic_now: Smooth::default(),
            trace: Trace::new(30.0),
            time: 0.0,
            ranges_for: (0.0, 0.0, 0),
        };
        ui.apply(setup);
        ui
    }

    pub fn load_gas_preset(&mut self, p: GasPreset) {
        self.gas_example = p;
        self.apply(p.setup());
    }

    pub fn load_atom_preset(&mut self, p: AtomPreset) {
        self.atom_example = p;
        self.apply(p.setup());
    }

    fn apply(&mut self, (p, same_speed, notes): Setup) {
        self.p = p;
        self.notes = notes;
        self.gas = Gas::new(&p, if self.atoms { 12 } else { 11 });
        if same_speed {
            self.gas.same_speed(&p);
        }
        self.avg = if same_speed { Avg::Short } else { Avg::Long };
        self.trace.clear();
        self.kinetic_now.clear();
        self.clear_stats();
    }

    fn clear_stats(&mut self) {
        let s = self.p.sigma();
        let v_top = 4.5 * s;
        self.hist_v = Hist::new(0.0, v_top, BINS);
        self.hist_v_exc = Hist::new(0.0, v_top, BINS);
        self.hist_vx = Hist::new(-v_top, v_top, BINS);
        self.hist_e = Hist::new(0.0, 6.0 * self.p.kt, BINS);
        self.speed_m.clear();
        self.kinetic_m.clear();
        self.energy_m.clear();
        self.excited_m.clear();
        self.ranges_for = (self.p.kt, self.p.mass, self.p.dim);
    }

    /// the temperature the formulas use: the bath's, or the gas's own when it is isolated
    fn kt_theory(&self) -> f64 {
        match self.p.walls {
            Walls::Bath => self.p.kt,
            Walls::Insulating => self.kinetic_now.get().map_or(self.p.kt, |e| 2.0 * e / self.p.dim as f64),
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let dt = widgets::frame_dt(ui);
        if self.running {
            let p = self.p;
            self.gas.step(&p, dt * self.speed);
            self.time += dt;
            self.record(dt);
            ui.ctx().request_repaint();
        }
        let win_h = ui.ctx().content_rect().height();
        egui::Panel::left(if self.atoms { "at_side" } else { "gas_side" }).resizable(true).default_size(340.0).min_size(260.0).show(ui, |ui| self.side_ui(ui));
        egui::Panel::bottom(if self.atoms { "at_plots" } else { "gas_plots" })
            .resizable(true)
            .default_size(win_h * 0.36)
            .min_size(150.0)
            .max_size(win_h * 0.6)
            .show(ui, |ui| self.plots_ui(ui));
        egui::CentralPanel::default().show(ui, |ui| self.box_ui(ui));
    }

    /// adds this frame's atoms to the statistics
    fn record(&mut self, dt: f64) {
        let tau = self.avg.tau();
        for h in [&mut self.hist_v, &mut self.hist_v_exc, &mut self.hist_vx, &mut self.hist_e] {
            h.fade(dt, tau);
        }
        for m in [&mut self.speed_m, &mut self.kinetic_m, &mut self.energy_m, &mut self.excited_m] {
            m.fade(dt, tau);
        }
        let delta = self.p.internal.map_or(0.0, |tl| tl.delta);
        let (mut kin, mut int) = (0.0, 0.0);
        for a in &self.gas.atoms {
            let v = a.v.iter().map(|c| c * c).sum::<f64>().sqrt();
            let e = self.gas.kinetic(a, self.p.mass);
            (if a.excited { &mut self.hist_v_exc } else { &mut self.hist_v }).add(v);
            self.hist_vx.add(a.v[0]);
            self.hist_e.add(e);
            self.speed_m.add(v);
            self.kinetic_m.add(e);
            let e_int = if a.excited { delta } else { 0.0 };
            self.energy_m.add(e + e_int);
            self.excited_m.add(if a.excited { 1.0 } else { 0.0 });
            kin += e;
            int += e_int;
        }
        let n = self.gas.atoms.len().max(1) as f64;
        self.kinetic_now.update(kin / n, dt, 2.0);
        let v = if self.atoms { [kin / n, int / n, (kin + int) / n] } else { [kin / n, f64::NAN, f64::NAN] };
        self.trace.push(self.time, v);
    }

    fn side_ui(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            let d = self.p.dim;
            if self.atoms {
                widgets::heading(ui, "3 · Motion plus internal levels", "Z₁ = (V/λ³) × (1 + g e^(−Δ/k_BT))");
                if let Some(p) = widgets::examples(ui, "at_example", self.atom_example, &AtomPreset::ALL, AtomPreset::label) {
                    self.load_atom_preset(p);
                }
            } else {
                let formula = match d {
                    1 => "Z₁ = L/λ ∝ β^(−1/2),   ⟨E⟩ = ½ k_BT",
                    2 => "Z₁ = A/λ² ∝ β^(−1),   ⟨E⟩ = k_BT",
                    _ => "Z₁ = V/λ³ ∝ β^(−3/2),   ⟨E⟩ = 3/2 k_BT",
                };
                widgets::heading(ui, "2 · A free particle in a box", formula);
                if let Some(p) = widgets::examples(ui, "gas_example", self.gas_example, &GasPreset::ALL, GasPreset::label) {
                    self.load_gas_preset(p);
                }
            }

            widgets::section(ui, "SETTINGS");
            let old = self.p;
            let label = match self.p.walls {
                Walls::Bath => "k_BT of the walls",
                Walls::Insulating => "k_BT (sets the speeds now)",
            };
            ui.add(egui::Slider::new(&mut self.p.kt, KT_MIN..=KT_MAX).logarithmic(true).text(label).custom_formatter(|v, _| fmt(v)));
            ui.add(egui::Slider::new(&mut self.p.mass, 0.25..=4.0).logarithmic(true).text("mass m").custom_formatter(|v, _| fmt(v)));
            ui.add(egui::Slider::new(&mut self.p.n, 1..=MAX_ATOMS).text("particles N"));
            ui.horizontal(|ui| {
                ui.label("dimensions:");
                for k in 1..=3 {
                    ui.selectable_value(&mut self.p.dim, k, format!("{k}D"));
                }
            });
            ui.horizontal(|ui| {
                ui.label("walls:");
                ui.selectable_value(&mut self.p.walls, Walls::Bath, "heat bath")
                    .on_hover_text("a particle that hits a wall leaves with a new random velocity (and internal state) at the wall's temperature");
                ui.selectable_value(&mut self.p.walls, Walls::Insulating, "insulating")
                    .on_hover_text("the walls only reflect: the energy of the gas stays the same");
            });
            ui.add(egui::Slider::new(&mut self.p.radius, 0.0..=0.06).text("particle radius").custom_formatter(|v, _| fmt(v)))
                .on_hover_text("0: the particles pass through each other (no collisions). In 1D there are no collisions anyway.");
            if let Some(mut tl) = self.p.internal {
                ui.add(egui::Slider::new(&mut tl.delta, 0.1..=5.0).logarithmic(true).text("excitation energy Δ").custom_formatter(|v, _| fmt(v)));
                ui.add(egui::Slider::new(&mut tl.g, 1..=5).text("states in the upper level g"));
                self.p.internal = Some(tl);
            }
            widgets::play_row(ui, &mut self.running, &mut self.speed, 0.05..=3.0, "speed");
            ui.horizontal(|ui| {
                if ui.button("new Maxwell velocities").on_hover_text("velocities and internal states drawn at k_BT").clicked() {
                    self.gas.thermalise(&self.p);
                    self.clear_stats();
                }
                if ui.button("same speed").on_hover_text("every particle gets the same speed, in a random direction (and the ground state)").clicked() {
                    self.gas.same_speed(&self.p);
                    self.clear_stats();
                }
            });
            if !self.atoms {
                ui.horizontal(|ui| {
                    ui.label("left plot:");
                    ui.selectable_value(&mut self.show_vx, false, "speed |v|");
                    ui.selectable_value(&mut self.show_vx, true, "one component v_x");
                });
            }
            if widgets::avg_row(ui, &mut self.avg) {
                self.clear_stats();
            }
            ui.checkbox(&mut self.arrows, "show the velocities");
            self.react(old);

            widgets::section(ui, "NUMBERS (per particle)");
            let kt = self.kt_theory();
            let half_d = 0.5 * d as f64;
            let c_int = self.p.internal.map_or(0.0, |tl| tl.heat_capacity(kt));
            let sp = speeds(d, (kt / self.p.mass).sqrt());
            let kin = self.kinetic_m.mean();
            let mut rows = vec![
                ("⟨E_kin⟩", Some(half_d * kt), kin, "d/2 k_BT: one half k_BT per quadratic term"),
                ("C/k_B", Some(half_d + c_int), self.energy_m.var().map(|v| v / (kt * kt)), "simulation: from the spread of the energies, C = Var(E)/k_BT²"),
                ("most probable speed", Some(sp[0]), None, "√((d − 1) k_BT/m)"),
                ("mean speed ⟨v⟩", Some(sp[1]), self.speed_m.mean(), ""),
                ("rms speed", Some(sp[2]), self.kinetic_m.mean().map(|e| (2.0 * e / self.p.mass).sqrt()), "√(d k_BT/m)"),
            ];
            if let Some(tl) = self.p.internal {
                rows.push(("excited fraction", Some(tl.excited_fraction(kt)), self.excited_m.mean(), "g e^(−Δ/k_BT) / (1 + g e^(−Δ/k_BT))"));
                rows.push(("⟨E_int⟩", Some(tl.delta * tl.excited_fraction(kt)), self.excited_m.mean().map(|f| f * tl.delta), ""));
            }
            rows.push((
                "k_BT of the motion",
                (self.p.walls == Walls::Bath).then_some(self.p.kt),
                kin.map(|e| e / half_d),
                "2⟨E_kin⟩/d: the temperature you would read off the motion",
            ));
            widgets::numbers(ui, if self.atoms { "at_numbers" } else { "gas_numbers" }, &rows);
            if self.p.walls == Walls::Insulating {
                ui.weak("Insulating walls: the formulas use the gas's own temperature, 2⟨E_kin⟩/d.");
            }
            widgets::notes(ui, self.notes);
        });
    }

    /// Adjusts the gas after a setting changed.
    fn react(&mut self, old: GasParams) {
        let p = self.p;
        if p == old {
            return;
        }
        if p.n != old.n {
            self.gas.resize(&p);
        }
        if p.dim != old.dim {
            self.gas.fit_dimension(&p);
            self.gas.thermalise(&p);
            self.trace.clear();
        }
        if p.mass != old.mass {
            // keep the energies: v ∝ 1/√m
            self.gas.scale_velocities((old.mass / p.mass).sqrt());
        }
        if p.kt != old.kt && p.walls == Walls::Insulating {
            // an isolated gas only changes temperature when we change its energy
            let e = self.gas.atoms.iter().map(|a| self.gas.kinetic(a, p.mass)).sum::<f64>() / p.n.max(1) as f64;
            let want = 0.5 * p.dim as f64 * p.kt;
            if e > 0.0 {
                self.gas.scale_velocities((want / e).sqrt());
            }
            self.kinetic_now.clear();
        }
        if (p.kt, p.mass, p.dim) != self.ranges_for || p.internal != old.internal || p.walls != old.walls || p.n != old.n {
            self.clear_stats();
        }
    }

    fn box_ui(&mut self, ui: &mut egui::Ui) {
        let full = ui.available_rect_before_wrap();
        let resp = ui.allocate_rect(full, Sense::drag());
        let painter = ui.painter_at(full);
        painter.rect_filled(full, 3.0, BG);
        let d = self.p.dim;
        let title = match d {
            3 => "THE BOX · drag to turn it",
            2 => "THE BOX · seen from above",
            _ => "THE BOX · one dimension: each particle in its own lane",
        };
        painter.text(full.min + vec2(8.0, 5.0), Align2::LEFT_TOP, title, FontId::proportional(13.0), LABEL);
        let wall = if self.p.walls == Walls::Bath { BATH_WALL } else { PLAIN_WALL };
        let walls_text = match self.p.walls {
            Walls::Bath => format!("walls: heat bath at k_BT = {}", fmt(self.p.kt)),
            Walls::Insulating => "walls: insulating (they only reflect)".to_string(),
        };
        let kt_motion = self.kinetic_now.get().map_or(String::new(), |e| format!("  ·  k_BT of the motion = 2⟨E_kin⟩/d = {}", fmt(2.0 * e / d as f64)));
        painter.text(
            pos2(full.left() + 8.0, full.bottom() - 6.0),
            Align2::LEFT_BOTTOM,
            format!("{walls_text}  ·  collisions so far: {}{kt_motion}", self.gas.collisions),
            FontId::proportional(12.0),
            LABEL,
        );
        if self.atoms {
            painter.text(full.right_top() + vec2(-8.0, 5.0), Align2::RIGHT_TOP, "ground", FontId::proportional(12.0), GROUND);
            painter.text(full.right_top() + vec2(-8.0, 20.0), Align2::RIGHT_TOP, "excited (Δ)", FontId::proportional(12.0), EXCITED);
        } else {
            painter.text(full.right_top() + vec2(-8.0, 5.0), Align2::RIGHT_TOP, "colour: speed (slow → fast)", FontId::proportional(12.0), LABEL);
        }
        let inner = full.shrink2(vec2(20.0, 30.0));
        if inner.width() < 40.0 || inner.height() < 40.0 {
            return;
        }
        if d == 3 && resp.dragged() {
            let dd = resp.drag_delta();
            self.cam.0 += dd.x * 0.008;
            self.cam.1 = (self.cam.1 + dd.y * 0.008).clamp(-1.4, 1.4);
        }
        let v_ref = (d as f64 * self.p.kt / self.p.mass).sqrt();
        let atoms = &self.gas.atoms;
        let two_level = self.atoms;
        let color = |k: usize| -> Color32 {
            let a = &atoms[k];
            if two_level {
                if a.excited { EXCITED } else { GROUND }
            } else {
                plot::heat((a.v.iter().map(|c| c * c).sum::<f64>().sqrt() / (2.0 * v_ref)) as f32)
            }
        };
        let arrow_len = 0.12 / v_ref.max(1e-9);
        match d {
            3 => {
                let cam = Cam { yaw: self.cam.0, pitch: self.cam.1, scale: inner.width().min(inner.height()) / 1.85, centre: inner.center() };
                let w = |r: [f64; 3]| [(r[0] - 0.5) as f32, (r[1] - 0.5) as f32, (r[2] - 0.5) as f32];
                // edges, the far ones dimmer
                let corners: Vec<[f32; 3]> = (0..8).map(|c| [(c & 1) as f32 - 0.5, ((c >> 1) & 1) as f32 - 0.5, ((c >> 2) & 1) as f32 - 0.5]).collect();
                let edges = (0..8).flat_map(|a| [1, 2, 4].map(|b| (a, a | b))).filter(|(a, b)| a != b);
                let mut back = vec![];
                let mut front = vec![];
                for (a, b) in edges {
                    let (pa, za) = cam.project(corners[a]);
                    let (pb, zb) = cam.project(corners[b]);
                    if za + zb < 0.0 { back.push([pa, pb]) } else { front.push([pa, pb]) }
                }
                for e in back {
                    painter.line_segment(e, Stroke::new(1.0, wall.gamma_multiply(0.45)));
                }
                let mut order: Vec<(usize, Pos2, f32)> = (0..atoms.len())
                    .map(|k| {
                        let (p, z) = cam.project(w(atoms[k].r));
                        (k, p, z)
                    })
                    .collect();
                order.sort_by(|a, b| a.2.total_cmp(&b.2));
                let radius = ((self.p.radius as f32) * cam.scale).max(3.0);
                for (k, p, z) in order {
                    let shade = (0.6 + 0.4 * (z + 0.87) / 1.74).clamp(0.4, 1.0);
                    if self.atoms && atoms[k].excited {
                        painter.circle_filled(p, radius * 1.9, EXCITED.gamma_multiply(0.25));
                    }
                    painter.circle_filled(p, radius, color(k).gamma_multiply(shade));
                    if self.arrows {
                        let tip = atoms[k].r.iter().zip(atoms[k].v).map(|(r, v)| r + v * arrow_len).collect::<Vec<_>>();
                        let (q, _) = cam.project(w([tip[0], tip[1], tip[2]]));
                        painter.line_segment([p, q], Stroke::new(1.2, Color32::from_gray(220)));
                    }
                }
                for e in front {
                    painter.line_segment(e, Stroke::new(2.0, wall));
                }
            }
            2 => {
                let side = inner.width().min(inner.height());
                let sq = Rect::from_center_size(inner.center(), vec2(side, side));
                let at = |r: &[f64; 3]| pos2(sq.left() + r[0] as f32 * side, sq.bottom() - r[1] as f32 * side);
                painter.rect_stroke(sq, 0.0, Stroke::new(2.5, wall), egui::StrokeKind::Outside);
                let radius = ((self.p.radius as f32) * side).max(3.0);
                for (k, a) in atoms.iter().enumerate() {
                    let p = at(&a.r);
                    if self.atoms && a.excited {
                        painter.circle_filled(p, radius * 1.9, EXCITED.gamma_multiply(0.25));
                    }
                    painter.circle_filled(p, radius, color(k));
                    if self.arrows {
                        let tip = [a.r[0] + a.v[0] * arrow_len, a.r[1] + a.v[1] * arrow_len, 0.0];
                        painter.line_segment([p, at(&tip)], Stroke::new(1.2, Color32::from_gray(220)));
                    }
                }
            }
            _ => {
                let n = atoms.len().max(1);
                let lane = (inner.height() / n as f32).min(22.0);
                let h = lane * n as f32;
                let r = Rect::from_center_size(inner.center(), vec2(inner.width(), h));
                painter.line_segment([r.left_top(), r.left_bottom()], Stroke::new(3.0, wall));
                painter.line_segment([r.right_top(), r.right_bottom()], Stroke::new(3.0, wall));
                let radius = (lane * 0.35).clamp(1.5, 6.0);
                for (k, a) in atoms.iter().enumerate() {
                    let y = r.top() + (k as f32 + 0.5) * lane;
                    painter.line_segment([pos2(r.left(), y), pos2(r.right(), y)], Stroke::new(1.0, plot::GRID));
                    let x = r.left() + a.r[0] as f32 * r.width();
                    if self.atoms && a.excited {
                        painter.circle_filled(pos2(x, y), radius * 1.9, EXCITED.gamma_multiply(0.25));
                    }
                    painter.circle_filled(pos2(x, y), radius, color(k));
                    if self.arrows {
                        let dx = (a.v[0] * arrow_len) as f32 * r.width();
                        painter.line_segment([pos2(x, y), pos2(x + dx, y)], Stroke::new(1.2, Color32::from_gray(220)));
                    }
                }
            }
        }
    }

    fn plots_ui(&mut self, ui: &mut egui::Ui) {
        let full = ui.available_rect_before_wrap();
        ui.allocate_rect(full, Sense::hover());
        let cols = plot::columns(full.shrink2(vec2(0.0, 4.0)), 3, 8.0);
        let d = self.p.dim;
        let kt = self.kt_theory();
        let s = (kt / self.p.mass).sqrt();
        let n = self.gas.atoms.len().max(1) as f64;

        // speeds (or one velocity component)
        if self.show_vx && !self.atoms {
            let h = &self.hist_vx;
            let dens: Vec<f64> = (0..BINS).map(|k| h.density(k)).collect();
            let y_max = (component_density(0.0, s) * 1.3).max(dens.iter().cloned().fold(0.0, f64::max) * 1.1);
            let p = Plot::new(ui, cols[0], "one velocity component: f(v_x)", "v_x", (h.lo, h.hi), (0.0, y_max), false);
            for (k, &v) in dens.iter().enumerate() {
                p.bar(h.lo + k as f64 * h.width(), h.lo + (k + 1) as f64 * h.width(), v, SIM.gamma_multiply(0.8));
            }
            p.curve(&plot::lin_grid(h.lo, h.hi, 200, |v| component_density(v, s)), Stroke::new(2.0, THEORY));
            p.legend(&[("simulation", SIM, false), ("Gaussian, width √(k_BT/m)", THEORY, false)]);
        } else {
            let (h, he) = (&self.hist_v, &self.hist_v_exc);
            let theory = plot::lin_grid(h.lo, h.hi, 200, |v| speed_density(v, d, s));
            let peak = theory.iter().map(|p| p.1).fold(0.0, f64::max);
            let title = if self.atoms { "speed distribution of ground and excited atoms" } else { "speed distribution f(v)" };
            if self.atoms {
                // each normalised by itself: the same shape means the motion ignores the internal state
                let norm = |h: &Hist, k: usize| {
                    let t: f64 = h.bins.iter().sum();
                    if t > 0.0 { h.bins[k] / (t * h.width()) } else { 0.0 }
                };
                let g: Vec<f64> = (0..BINS).map(|k| norm(h, k)).collect();
                let e: Vec<f64> = (0..BINS).map(|k| norm(he, k)).collect();
                let y_max = (peak * 1.35).max(g.iter().chain(&e).cloned().fold(0.0, f64::max) * 1.05);
                let p = Plot::new(ui, cols[0], title, "speed v", (h.lo, h.hi), (0.0, y_max), false);
                p.steps(h.lo, h.width(), &g, Stroke::new(2.0, GROUND));
                p.steps(h.lo, h.width(), &e, Stroke::new(2.0, EXCITED));
                p.curve(&theory, Stroke::new(2.0, THEORY));
                p.legend(&[("ground", GROUND, false), ("excited", EXCITED, false), ("Maxwell–Boltzmann", THEORY, false)]);
            } else {
                let dens: Vec<f64> = (0..BINS).map(|k| h.density(k)).collect();
                let y_max = (peak * 1.3).max(dens.iter().cloned().fold(0.0, f64::max) * 1.05);
                let p = Plot::new(ui, cols[0], title, "speed v", (h.lo, h.hi), (0.0, y_max), false);
                for (k, &v) in dens.iter().enumerate() {
                    p.bar(h.lo + k as f64 * h.width(), h.lo + (k + 1) as f64 * h.width(), v, SIM.gamma_multiply(0.8));
                }
                p.curve(&theory, Stroke::new(2.0, THEORY));
                let names = ["v_p", "⟨v⟩", "v_rms"];
                for (k, v) in speeds(d, s).into_iter().enumerate() {
                    if d == 1 && k == 0 {
                        continue;
                    }
                    p.dashed_vline(v, Stroke::new(1.0, NOW));
                    p.label(v, y_max * (0.97 - 0.09 * k as f64), Align2::LEFT_TOP, &format!(" {}", names[k]), NOW);
                }
                p.legend(&[("simulation", SIM, false), ("Maxwell–Boltzmann", THEORY, false)]);
            }
        }

        if self.atoms {
            self.energy_trace(ui, cols[1], kt);
            self.heat_capacity_plot(ui, cols[2], kt);
            return;
        }

        // kinetic energy
        let h = &self.hist_e;
        let dens: Vec<f64> = (0..BINS).map(|k| h.density(k)).collect();
        let y_max = (0..BINS).map(|k| energy_density(h.centre(k), d, kt).max(dens[k])).fold(0.0, f64::max) * 1.15;
        let p = Plot::new(ui, cols[1], "energy distribution P(E)", "kinetic energy E", (h.lo, h.hi), (0.0, y_max), false);
        for (k, &v) in dens.iter().enumerate() {
            p.bar(h.lo + k as f64 * h.width(), h.lo + (k + 1) as f64 * h.width(), v, SIM.gamma_multiply(0.8));
        }
        p.curve(&plot::lin_grid(h.lo + 1e-3 * h.hi, h.hi, 300, |e| energy_density(e, d, kt)), Stroke::new(2.0, THEORY));
        let mean = 0.5 * d as f64 * kt;
        p.dashed_vline(mean, Stroke::new(1.0, NOW));
        p.label(mean, y_max * 0.97, Align2::LEFT_TOP, " ⟨E⟩", NOW);
        let formula = match d {
            1 => "∝ E^(−1/2) e^(−E/k_BT)",
            2 => "e^(−E/k_BT)/k_BT",
            _ => "∝ E^(1/2) e^(−E/k_BT)",
        };
        p.legend(&[("simulation", SIM, false), (formula, THEORY, false)]);

        // mean kinetic energy in time
        let mean = 0.5 * d as f64 * kt;
        let sd = (0.5 * d as f64 / n).sqrt() * kt;
        let y_max = (mean + 3.0 * sd).max(self.trace.max(0)).max(0.05) * 1.2;
        let p = Plot::new(ui, cols[2], "mean kinetic energy per particle", "time (s)", (-self.trace.window, 0.0), (0.0, y_max), false);
        if self.p.walls == Walls::Bath {
            p.band(mean - sd, mean + sd, NOW.gamma_multiply(0.18));
        }
        p.hline(mean, Stroke::new(1.5, THEORY));
        let pts: Vec<_> = self.trace.points.iter().map(|&(t, v)| (t - self.time, v[0])).collect();
        p.curve(&pts, Stroke::new(1.5, SIM));
        p.legend(&[("particles now", SIM, false), ("d/2 k_BT ± σ/√N", THEORY, false)]);
    }

    /// example 3: where the energy is, in time
    fn energy_trace(&self, ui: &egui::Ui, rect: Rect, kt: f64) {
        let d = self.p.dim as f64;
        let tl = self.p.internal.unwrap_or(TwoLevel { delta: 1.0, g: 1 });
        let (kin, int) = (0.5 * d * kt, tl.delta * tl.excited_fraction(kt));
        // room above the lines for the legend
        let y_max = (self.trace.max(2)).max(kin + int).max(0.05) * 1.5;
        let p = Plot::new(ui, rect, "energy per atom", "time (s)", (-self.trace.window, 0.0), (0.0, y_max), false);
        for (k, color) in [(0, GROUND), (1, EXCITED), (2, THEORY)] {
            let pts: Vec<_> = self.trace.points.iter().map(|&(t, v)| (t - self.time, v[k])).collect();
            p.curve(&pts, Stroke::new(1.5, color));
        }
        p.dashed(&[(-self.trace.window, kin), (0.0, kin)], Stroke::new(1.0, GROUND));
        p.dashed(&[(-self.trace.window, int), (0.0, int)], Stroke::new(1.0, EXCITED));
        p.legend(&[("motion", GROUND, false), ("internal", EXCITED, false), ("total", THEORY, false), ("formulas at k_BT", CLASSICAL, true)]);
    }

    /// example 3: C = d/2 k_B + the Schottky bump
    fn heat_capacity_plot(&self, ui: &egui::Ui, rect: Rect, kt: f64) {
        let half_d = 0.5 * self.p.dim as f64;
        let tl = self.p.internal.unwrap_or(TwoLevel { delta: 1.0, g: 1 });
        let curve = plot::log_grid(KT_MIN, KT_MAX, 240, |t| half_d + tl.heat_capacity(t));
        let y_max = curve.iter().map(|p| p.1).fold(0.0, f64::max) * 1.3;
        let p = Plot::new(ui, rect, "heat capacity C/k_B per atom", "k_BT", (KT_MIN, KT_MAX), (0.0, y_max), true);
        p.dashed(&[(KT_MIN, half_d), (KT_MAX, half_d)], Stroke::new(1.0, CLASSICAL));
        p.curve(&curve, Stroke::new(2.0, THEORY));
        p.vline(kt, Stroke::new(1.0, NOW));
        p.dot(kt, half_d + tl.heat_capacity(kt), 5.0, NOW);
        if let Some(v) = self.energy_m.var() {
            p.dot(kt, v / (kt * kt), 3.5, SIM);
        }
        p.legend(&[("d/2 + C_int", THEORY, false), ("motion only, d/2", CLASSICAL, true), ("from the spread", SIM, false)]);
    }
}
