//! PS13 50 · Sequences of Stern–Gerlach boxes.
//!
//! Up to four boxes along a beam line, each with an axis (z, x, y or u(θ, φ))
//! and a choice of the output that continues (↑, ↓, or both recombined
//! without which-path information, which is no measurement). Atoms fly
//! through one at a time or as a stream; each measuring box draws the outcome
//! with the Born rule and the spin collapses onto |↑u⟩ or |↓u⟩ (Eq. 50.3).
//! Counters at every port, measured fractions against the predictions, the
//! state after every box on small Bloch spheres, the latest atom on a large
//! sphere with the energy levels of the box it is in.

use std::collections::VecDeque;
use std::f64::consts::PI;

use qb_core::egui::{self, Align2, Color32, Pos2, Rect, RichText, Sense, Shape, Stroke, Vec2, pos2, vec2};
use qb_core::num::ShotHistogram;
use qb_core::widgets::{
    FlashKind, HistScale, HistogramPlot, LevelLadder, MeasureFlash, PanZoom, PlotFrame, label_box, plot_line, readout_row, repeat_button,
    segmented,
};
use qb_core::{Ctx, Look, QbRng, Scene, wall_time};

use crate::bloch::{AxisChoice, BlochAxis, BlochSphere, SceneAxis, state_panel};
use crate::spinor::{EX, EY, EZ, Mat2, Spinor, Vec3, dir, measure, random_state};
use crate::util::{Key, deg, error_point, fmt_amp, legend, panel, smooth};

/// The presets of the Examples menu.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SgPreset {
    ZX,
    ZXZ,
    ZZ,
    ZTheta,
    Oven,
}

impl SgPreset {
    pub const ALL: [SgPreset; 5] = [SgPreset::ZX, SgPreset::ZXZ, SgPreset::ZZ, SgPreset::ZTheta, SgPreset::Oven];

    pub fn title(self) -> &'static str {
        match self {
            SgPreset::ZX => "PS13 50 · Stern–Gerlach z, then x",
            SgPreset::ZXZ => "PS13 50 · Stern–Gerlach z–x–z",
            SgPreset::ZZ => "PS13 50 · Stern–Gerlach z–z",
            SgPreset::ZTheta => "PS13 50 · z, then an axis at angle θ",
            SgPreset::Oven => "PS13 50 · Unpolarised oven (a mixture)",
        }
    }
}

/// Axis of a box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AxisSel {
    Z,
    X,
    Y,
    /// u(θ, φ) of Eq. 50.2
    Angle,
}

/// Which output continues to the next box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Port {
    Up,
    Down,
    /// both beams recombined without which-path information: no measurement
    Both,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SgBox {
    pub axis: AxisSel,
    /// θ, φ in radians (for `AxisSel::Angle`)
    pub theta: f64,
    pub phi: f64,
    pub port: Port,
}

impl SgBox {
    fn new(axis: AxisSel) -> Self {
        SgBox {
            axis,
            theta: 60f64.to_radians(),
            phi: 0.0,
            port: Port::Up,
        }
    }

    pub fn dir(&self) -> Vec3 {
        match self.axis {
            AxisSel::Z => EZ,
            AxisSel::X => EX,
            AxisSel::Y => EY,
            AxisSel::Angle => dir(self.theta, self.phi),
        }
    }

    /// "z", "x", "y" or "u".
    pub fn name(&self) -> &'static str {
        match self.axis {
            AxisSel::Z => "z",
            AxisSel::X => "x",
            AxisSel::Y => "y",
            AxisSel::Angle => "u",
        }
    }

    /// "z", or "u (60°, 0°)" for an axis at angles.
    fn short(&self) -> String {
        match self.axis {
            AxisSel::Angle => format!("u ({:.0}°, {:.0}°)", self.theta.to_degrees(), self.phi.to_degrees()).replace('-', "−"),
            _ => self.name().to_string(),
        }
    }

    fn describe(&self) -> String {
        match self.axis {
            AxisSel::Angle => format!("u: θ = {}, φ = {}", deg(self.theta), deg(self.phi)),
            _ => self.name().to_string(),
        }
    }

    pub fn measures(&self) -> bool {
        self.port != Port::Both
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    Oven,
    Polarised,
}

#[derive(Clone, Copy, Debug, Default)]
struct BoxStats {
    arrived: u64,
    up: u64,
    down: u64,
    /// wall time of the last ↑ / ↓ hit (for the counter flash)
    hit: [f64; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Leg {
    ToBox(usize),
    InBox(usize),
    ToStop(usize),
    ToDetector,
    Done,
}

#[derive(Clone, Debug)]
struct Atom {
    id: u64,
    spin: Spinor,
    u: f64,
    leg: Leg,
    /// outcome in the current box (↑?)
    outcome: bool,
    /// outcome of the box just left (None after a recombined box or the oven)
    prev: Option<bool>,
}

/// Positions along the beam line (0 … 1 across the drawing).
struct Geom {
    a: Vec<f64>,
    b: Vec<f64>,
    f: Vec<f64>,
    det: f64,
}

const SOURCE_EXIT: f64 = 0.065;
/// beam-line units per second
const SPEED: f64 = 0.3;
/// turns of the display precession inside a box
const TURNS: f64 = 1.5;
/// deflection inside a box (fraction of the port offset)
const D_BOX: f64 = 0.42;

fn geom(n: usize) -> Geom {
    let (start, end) = (0.095, 0.9);
    let w = (end - start) / n as f64;
    let mut g = Geom {
        a: vec![],
        b: vec![],
        f: vec![],
        det: 0.955,
    };
    for k in 0..n {
        let a = start + k as f64 * w + 0.04 * w;
        let b = a + 0.44 * w;
        g.a.push(a);
        g.b.push(b);
        g.f.push(b + 0.2 * w);
    }
    g
}

/// Measured fraction of ↑ at one angle of the last box.
#[derive(Clone, Copy, Debug)]
struct ScanPoint {
    theta: f64,
    up: u64,
    n: u64,
}

pub struct SgScene {
    preset: SgPreset,
    source: Source,
    /// direction of a polarised source
    theta0: f64,
    phi0: f64,
    boxes: Vec<SgBox>,
    stream: bool,
    rate: f64,
    repeat_n: usize,
    axis_choice: AxisChoice,

    t: f64,
    atoms: Vec<Atom>,
    next_id: u64,
    spawn_acc: f64,
    stats: Vec<BoxStats>,
    sent: u64,
    /// the atom followed on the big sphere
    tracked: Option<u64>,
    tracked_spin: Spinor,
    tracked_box: usize,
    tracked_in_field: bool,
    /// what became of the followed atom once it is gone
    tracked_fate: Option<String>,
    trail: Vec<Vec3>,
    /// directions of recent atoms from the oven
    cloud: VecDeque<Vec3>,
    scan: Vec<ScanPoint>,
    scan_key: Option<(Option<[i64; 3]>, i64)>,
    flash: MeasureFlash,
}

impl SgScene {
    pub fn new(preset: SgPreset) -> Self {
        let up_z = SgBox::new(AxisSel::Z);
        let last = |axis| {
            let mut b = SgBox::new(axis);
            b.port = Port::Up;
            b
        };
        let boxes = match preset {
            SgPreset::ZX => vec![up_z, last(AxisSel::X)],
            SgPreset::ZXZ => vec![up_z, SgBox::new(AxisSel::X), last(AxisSel::Z)],
            SgPreset::ZZ => vec![up_z, last(AxisSel::Z)],
            SgPreset::ZTheta => vec![up_z, last(AxisSel::Angle)],
            SgPreset::Oven => vec![last(AxisSel::Angle)],
        };
        let mut s = SgScene {
            preset,
            source: Source::Oven,
            theta0: 0.0,
            phi0: 0.0,
            boxes,
            stream: true,
            rate: 3.0,
            repeat_n: 1000,
            axis_choice: AxisChoice::U,
            t: 0.0,
            atoms: vec![],
            next_id: 0,
            spawn_acc: 0.6,
            stats: vec![],
            sent: 0,
            tracked: None,
            tracked_spin: Spinor::UP_Z,
            tracked_box: 0,
            tracked_in_field: false,
            tracked_fate: None,
            trail: vec![],
            cloud: VecDeque::new(),
            scan: vec![],
            scan_key: None,
            flash: MeasureFlash::new(),
        };
        s.restart();
        s
    }

    fn restart(&mut self) {
        self.t = 0.0;
        self.atoms.clear();
        self.spawn_acc = 0.6;
        self.stats = vec![BoxStats::default(); self.boxes.len()];
        self.sent = 0;
        self.tracked = None;
        self.tracked_box = 0;
        self.tracked_in_field = false;
        self.tracked_fate = None;
        self.trail.clear();
        self.flash.clear();
        self.check_scan_key();
    }

    /// Counters refer to one configuration: clear them (and the atoms in flight) when it changes.
    fn chain_changed(&mut self) {
        self.restart();
    }

    fn last(&self) -> usize {
        self.boxes.len() - 1
    }

    fn scanning(&self) -> bool {
        let b = &self.boxes[self.last()];
        b.axis == AxisSel::Angle && b.measures()
    }

    /// The scan plot is valid for one input state of the last box and one φ.
    fn check_scan_key(&mut self) {
        let last = self.last();
        let input = self.ideal_inputs()[last].map(|s| {
            let r = s.bloch();
            [
                (r[0] * 1e6).round() as i64,
                (r[1] * 1e6).round() as i64,
                (r[2] * 1e6).round() as i64,
            ]
        });
        let key = Some((input, (self.boxes[last].phi * 1e6).round() as i64));
        if key != self.scan_key {
            self.scan.clear();
            self.scan_key = key;
        }
    }

    fn source_name(&self) -> &'static str {
        match self.source {
            Source::Oven => "oven",
            Source::Polarised => "polarised",
        }
    }

    fn source_state(&mut self, rng: &mut QbRng) -> Spinor {
        match self.source {
            Source::Oven => {
                let s = random_state(rng);
                self.cloud.push_back(s.bloch());
                while self.cloud.len() > 60 {
                    self.cloud.pop_front();
                }
                s
            }
            Source::Polarised => Spinor::up_along(self.theta0, self.phi0),
        }
    }

    /// The ideal input state of every box (None = the unpolarised mixture).
    fn ideal_inputs(&self) -> Vec<Option<Spinor>> {
        let mut cur = match self.source {
            Source::Oven => None,
            Source::Polarised => Some(Spinor::up_along(self.theta0, self.phi0)),
        };
        let mut out = vec![];
        for b in &self.boxes {
            out.push(cur);
            if b.measures() {
                cur = Some(if b.port == Port::Down {
                    Spinor::down_dir(b.dir())
                } else {
                    Spinor::up_dir(b.dir())
                });
            }
        }
        out
    }

    /// Born-rule P(↑) of box k for its ideal input.
    fn predicted(&self, k: usize) -> f64 {
        self.ideal_inputs()[k].map_or(0.5, |s| s.p_up(self.boxes[k].dir()))
    }

    fn record_scan(&mut self, up: bool) {
        let theta = self.boxes[self.last()].theta.to_degrees();
        match self.scan.iter_mut().find(|p| (p.theta - theta).abs() < 0.05) {
            Some(p) => {
                p.n += 1;
                p.up += up as u64;
            }
            None => self.scan.push(ScanPoint {
                theta,
                up: up as u64,
                n: 1,
            }),
        }
    }

    // ------------------------------------------------------------ atoms

    fn spawn(&mut self, ctx: &mut Ctx) {
        let spin = self.source_state(ctx.rng);
        let id = self.next_id;
        self.next_id += 1;
        self.sent += 1;
        self.atoms.push(Atom {
            id,
            spin,
            u: SOURCE_EXIT,
            leg: Leg::ToBox(0),
            outcome: true,
            prev: None,
        });
        let tracked_alive = self.tracked.is_some_and(|t| self.atoms.iter().any(|a| a.id == t));
        if !tracked_alive {
            self.tracked = Some(id);
            self.tracked_spin = spin;
            self.tracked_box = 0;
            self.tracked_in_field = false;
            self.tracked_fate = None;
            self.trail = vec![spin.bloch()];
        }
    }

    /// Key M / button: one atom (starts the transport if paused).
    fn send_one(&mut self, ctx: &mut Ctx) {
        self.spawn(ctx);
        ctx.playing = true;
        ctx.set_status(format!("atom {} leaves the {}", self.next_id, self.source_name()));
    }

    /// N atoms through the chain at once (no animation).
    fn batch(&mut self, n: usize, record_stats: bool, ctx: &mut Ctx) {
        let last = self.last();
        let scanning = self.scanning();
        for _ in 0..n {
            let mut spin = self.source_state(ctx.rng);
            if record_stats {
                self.sent += 1;
            }
            for k in 0..self.boxes.len() {
                let b = self.boxes[k];
                if record_stats {
                    self.stats[k].arrived += 1;
                }
                if !b.measures() {
                    continue;
                }
                let (up, s) = measure(&spin, b.dir(), ctx.rng);
                spin = s;
                if record_stats {
                    if up { self.stats[k].up += 1 } else { self.stats[k].down += 1 }
                }
                if k == last {
                    if scanning {
                        self.record_scan(up);
                    }
                    break;
                }
                let goes_on = (up && b.port == Port::Up) || (!up && b.port == Port::Down);
                if !goes_on {
                    break;
                }
            }
        }
    }

    fn batch_send(&mut self, n: usize, ctx: &mut Ctx) {
        self.batch(n, true, ctx);
        let last = self.last();
        let st = self.stats[last];
        let done = st.up + st.down;
        ctx.set_status(if self.boxes[last].measures() && done > 0 {
            format!(
                "{n} atoms sent at once: the last box counted {done} in all, ↑ fraction {:.3} (Born rule {:.3})",
                st.up as f64 / done as f64,
                self.predicted(last)
            )
        } else {
            format!("{n} atoms sent at once")
        });
    }

    /// N atoms at each of 19 angles θ = 0°, 10°, … 180° of the last box (for the plot only).
    fn scan_theta(&mut self, n: usize, ctx: &mut Ctx) {
        if !self.scanning() {
            return;
        }
        let last = self.last();
        let keep = self.boxes[last].theta;
        for i in 0..=18 {
            self.boxes[last].theta = (i as f64 * 10.0).to_radians();
            self.batch(n, false, ctx);
        }
        self.boxes[last].theta = keep;
        ctx.set_status(format!("θ scan: {n} atoms at each of 19 angles of the last box"));
    }

    fn step_atoms(&mut self, dt: f64, ctx: &mut Ctx) {
        let g = geom(self.boxes.len());
        let last = self.last();
        let mut atoms = std::mem::take(&mut self.atoms);
        for atom in atoms.iter_mut() {
            let mut rem = dt;
            for _ in 0..16 {
                if rem <= 0.0 || atom.leg == Leg::Done {
                    break;
                }
                let target = match atom.leg {
                    Leg::ToBox(k) => g.a[k],
                    Leg::InBox(k) => g.b[k],
                    Leg::ToStop(k) => g.f[k],
                    Leg::ToDetector => g.det,
                    Leg::Done => break,
                };
                let need = ((target - atom.u) / SPEED).max(0.0);
                let reached = need <= rem;
                let go = if reached { need } else { rem };
                if let Leg::InBox(k) = atom.leg
                    && self.boxes[k].measures()
                {
                    // precession about the box field (display rate)
                    let omega = TURNS * 2.0 * PI * SPEED / (g.b[k] - g.a[k]);
                    atom.spin = Mat2::rotation(self.boxes[k].dir(), omega * go) * atom.spin;
                }
                atom.u += go * SPEED;
                rem -= go;
                if reached {
                    atom.u = target;
                    self.event(atom, last, ctx);
                }
            }
            if Some(atom.id) == self.tracked {
                self.tracked_spin = atom.spin;
                self.tracked_box = match atom.leg {
                    Leg::ToBox(k) | Leg::InBox(k) | Leg::ToStop(k) => k,
                    Leg::ToDetector => last,
                    Leg::Done => self.tracked_box,
                };
                self.tracked_in_field = matches!(atom.leg, Leg::InBox(k) if self.boxes[k].measures());
                let r = atom.spin.bloch();
                if self
                    .trail
                    .last()
                    .is_none_or(|p| (p[0] - r[0]).abs() + (p[1] - r[1]).abs() + (p[2] - r[2]).abs() > 1e-3)
                {
                    self.trail.push(r);
                    if self.trail.len() > 600 {
                        self.trail.drain(..100);
                    }
                }
            }
        }
        atoms.retain(|a| a.leg != Leg::Done);
        self.atoms = atoms;
    }

    /// An atom reached the end of its current leg.
    fn event(&mut self, atom: &mut Atom, last: usize, ctx: &mut Ctx) {
        match atom.leg {
            Leg::ToBox(k) => {
                self.stats[k].arrived += 1;
                if self.boxes[k].measures() {
                    // the outcome is drawn now (so the dot can curve the right way); precession about u
                    // inside the box does not change P(↑u), the collapse is shown at the exit
                    atom.outcome = ctx.rng.bool(atom.spin.p_up(self.boxes[k].dir()));
                }
                atom.leg = Leg::InBox(k);
            }
            Leg::InBox(k) => {
                let b = self.boxes[k];
                if b.measures() {
                    let before = atom.spin.p_up(b.dir());
                    let up = atom.outcome;
                    atom.spin = if up { Spinor::up_dir(b.dir()) } else { Spinor::down_dir(b.dir()) };
                    let st = &mut self.stats[k];
                    if up {
                        st.up += 1
                    } else {
                        st.down += 1
                    }
                    st.hit[if up { 0 } else { 1 }] = wall_time();
                    if k == last && self.scanning() {
                        self.record_scan(up);
                    }
                    if Some(atom.id) == self.tracked {
                        let arrow = if up { "↑" } else { "↓" };
                        self.flash.trigger(FlashKind::Other, format!("SG {}: {arrow}{}", k + 1, b.name()));
                        self.trail.push([f64::NAN; 3]);
                        ctx.set_status(format!(
                            "atom {}: SG {} ({}) gave {arrow}{} (P was {before:.3}); its spin is now |{arrow}{}⟩",
                            atom.id + 1,
                            k + 1,
                            b.describe(),
                            b.name(),
                            b.name()
                        ));
                    }
                    atom.prev = Some(up);
                    atom.leg = if k == last {
                        Leg::ToDetector
                    } else if (up && b.port == Port::Up) || (!up && b.port == Port::Down) {
                        Leg::ToBox(k + 1)
                    } else {
                        Leg::ToStop(k)
                    };
                } else {
                    atom.prev = None;
                    atom.leg = if k == last { Leg::ToDetector } else { Leg::ToBox(k + 1) };
                }
            }
            Leg::ToStop(k) => {
                if Some(atom.id) == self.tracked {
                    self.tracked_fate = Some(format!("absorbed by the beam stop after SG {}", k + 1));
                }
                atom.leg = Leg::Done;
            }
            Leg::ToDetector => {
                if Some(atom.id) == self.tracked {
                    self.tracked_fate = Some("reached the detector".into());
                }
                atom.leg = Leg::Done;
            }
            Leg::Done => {}
        }
    }

    /// Vertical offset (units of the port offset, + up) of an atom; the bool is true
    /// inside a recombining box (drawn as two half-dots).
    fn atom_y(&self, a: &Atom, g: &Geom) -> (f64, bool) {
        let o = |up: bool| if up { 1.0 } else { -1.0 };
        let last = self.last();
        match a.leg {
            Leg::InBox(k) => {
                let s = ((a.u - g.a[k]) / (g.b[k] - g.a[k])).clamp(0.0, 1.0);
                if self.boxes[k].measures() {
                    (o(a.outcome) * D_BOX * s * s, false)
                } else {
                    (0.6 * (PI * s).sin(), true)
                }
            }
            Leg::ToBox(k) => match (k, a.prev) {
                (k, Some(up)) if k > 0 => {
                    let (b, f) = (g.b[k - 1], g.f[k - 1]);
                    if a.u < f {
                        (o(up) * (D_BOX + (1.0 - D_BOX) * (a.u - b) / (f - b)), false)
                    } else {
                        let s = ((a.u - f) / (g.a[k] - f)) as f32;
                        (o(up) * (1.0 - smooth(s) as f64), false)
                    }
                }
                _ => (0.0, false),
            },
            Leg::ToStop(k) => {
                let up = a.prev.unwrap_or(true);
                let (b, f) = (g.b[k], g.f[k]);
                (o(up) * (D_BOX + (1.0 - D_BOX) * ((a.u - b) / (f - b)).clamp(0.0, 1.0)), false)
            }
            Leg::ToDetector => match a.prev {
                Some(up) => {
                    let b = g.b[last];
                    (o(up) * (D_BOX + (1.0 - D_BOX) * ((a.u - b) / (g.det - b)).clamp(0.0, 1.0)), false)
                }
                None => (0.0, false),
            },
            Leg::Done => (0.0, false),
        }
    }

    // ------------------------------------------------------------ drawing

    fn beam_ui(&self, ui: &mut egui::Ui, size: Vec2) -> Rect {
        let look = Look::of(ui);
        let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
        let painter = ui.painter_at(rect);
        panel(&painter, rect);
        let g = geom(self.boxes.len());
        let last = self.last();
        let x = |u: f64| rect.left() + u as f32 * rect.width();
        let yc = rect.center().y + look.px(10.0);
        let big_y = rect.height() * 0.3;
        let y = |off: f64| yc - off as f32 * big_y;
        let beam = look.stroke(4.0, look.weak_text().gamma_multiply(0.28));
        let beam_line = |pts: Vec<Pos2>| {
            painter.add(Shape::line(pts, beam));
        };
        let now = wall_time();

        // source
        let oven = Rect::from_min_max(pos2(x(0.008), yc - look.px(20.0)), pos2(x(0.055), yc + look.px(20.0)));
        painter.rect_filled(oven, 4.0, look.series(1).gamma_multiply(0.35));
        painter.rect_stroke(oven, 4.0, look.stroke(1.2, look.series(1)), egui::StrokeKind::Inside);
        painter.text(
            pos2(oven.center().x, oven.top() - look.px(3.0)),
            Align2::CENTER_BOTTOM,
            match self.source {
                Source::Oven => "oven",
                Source::Polarised => "|↑n₀⟩",
            },
            look.font(12.5),
            look.text(),
        );
        beam_line(vec![pos2(oven.right(), yc), pos2(x(g.a[0]), yc)]);

        for (k, b) in self.boxes.iter().enumerate() {
            let (xa, xb) = (x(g.a[k]), x(g.b[k]));
            let n = 24;
            // beams inside the box
            for sgn in [1.0, -1.0] {
                let pts: Vec<Pos2> = (0..=n)
                    .map(|i| {
                        let s = i as f64 / n as f64;
                        let off = if b.measures() {
                            sgn * D_BOX * s * s
                        } else {
                            sgn * 0.6 * (PI * s).sin()
                        };
                        pos2(x(g.a[k] + s * (g.b[k] - g.a[k])), y(off))
                    })
                    .collect();
                beam_line(pts);
            }
            // magnet: pointed upper pole, flat lower pole with a groove (schematic)
            let top = y(0.95);
            let bot = y(-0.95);
            let frame = Rect::from_min_max(pos2(xa, top), pos2(xb, bot));
            let pole = look.plot_frame().gamma_multiply(0.55);
            let mid = y(0.0);
            painter.add(Shape::convex_polygon(
                vec![
                    pos2(xa, top),
                    pos2(xb, top),
                    pos2(xb, mid - 0.62 * big_y),
                    pos2(xa, mid - 0.62 * big_y),
                ],
                pole,
                Stroke::NONE,
            ));
            painter.add(Shape::convex_polygon(
                vec![
                    pos2(xa, mid - 0.62 * big_y),
                    pos2(xb, mid - 0.62 * big_y),
                    pos2(xb, mid - 0.5 * big_y),
                    pos2(xa, mid - 0.5 * big_y),
                ],
                pole.gamma_multiply(0.6),
                Stroke::NONE,
            ));
            painter.add(Shape::convex_polygon(
                vec![
                    pos2(xa, mid + 0.55 * big_y),
                    pos2(xb, mid + 0.55 * big_y),
                    pos2(xb, bot),
                    pos2(xa, bot),
                ],
                pole,
                Stroke::NONE,
            ));
            let accent = if Some(k) == self.tracked.map(|_| self.tracked_box) && self.tracked_in_field {
                look.accent()
            } else {
                look.plot_frame()
            };
            painter.rect_stroke(frame, 3.0, look.stroke(1.2, accent), egui::StrokeKind::Inside);
            painter.text(
                pos2(0.5 * (xa + xb), top - look.px(4.0)),
                Align2::CENTER_BOTTOM,
                format!("SG {} · {}", k + 1, b.name()),
                look.font(13.5),
                look.text(),
            );
            if b.axis == AxisSel::Angle {
                painter.text(
                    pos2(0.5 * (xa + xb), bot + look.px(3.0)),
                    Align2::CENTER_TOP,
                    format!("θ = {}, φ = {}", deg(b.theta), deg(b.phi)),
                    look.font(11.0),
                    look.weak_text(),
                );
            }
            if !b.measures() {
                painter.text(
                    pos2(0.5 * (xa + xb), mid),
                    Align2::CENTER_CENTER,
                    "recombined",
                    look.font(11.0),
                    look.weak_text(),
                );
            }

            // outputs
            let st = &self.stats[k];
            let port_label = |pos: Pos2, align: Align2, up: bool, extra: &str, col: Color32| {
                let count = if up { st.up } else { st.down };
                let arrow = if up { "↑" } else { "↓" };
                label_box(&painter, pos, align, &format!("{arrow}{} {count}{extra}", b.name()), col);
            };
            let glow = |p: Pos2, which: usize| {
                let age = (now - st.hit[which]) as f32;
                if age < 0.35 {
                    painter.circle_filled(
                        p,
                        look.px(10.0) * (1.0 + age * 2.0),
                        look.measure().gamma_multiply(0.6 * (1.0 - age / 0.35)),
                    );
                    ui.ctx().request_repaint();
                }
            };
            if k == last {
                let xd = x(g.det);
                if b.measures() {
                    for (up, sgn) in [(true, 1.0), (false, -1.0)] {
                        beam_line(vec![pos2(xb, y(sgn * D_BOX)), pos2(xd, y(sgn))]);
                        let spot = pos2(xd, y(sgn));
                        glow(spot, if up { 0 } else { 1 });
                        let total = (st.up + st.down).max(1) as f64;
                        let frac = if up { st.up } else { st.down } as f64 / total;
                        painter.circle_filled(spot, look.px(3.0 + 7.0 * frac as f32), look.counts());
                        let lp = pos2(xd - look.px(10.0), y(sgn) - sgn as f32 * look.px(16.0));
                        port_label(
                            lp,
                            Align2::RIGHT_CENTER,
                            up,
                            &if st.up + st.down > 0 {
                                format!(" ({:.1} %)", 100.0 * frac)
                            } else {
                                String::new()
                            },
                            look.counts(),
                        );
                    }
                } else {
                    beam_line(vec![pos2(xb, yc), pos2(xd, yc)]);
                    painter.circle_filled(pos2(xd, yc), look.px(6.0), look.counts());
                    label_box(
                        &painter,
                        pos2(xd - look.px(10.0), yc - look.px(16.0)),
                        Align2::RIGHT_CENTER,
                        &format!("{} atoms, no spin recorded", st.arrived),
                        look.counts(),
                    );
                }
                // the detector screen
                painter.line_segment(
                    [pos2(xd, y(1.25)), pos2(xd, y(-1.25))],
                    look.stroke(3.0, look.counts().gamma_multiply(0.7)),
                );
                // its name, written upwards beside it (the ports' labels sit on its left)
                let g = painter.layout_no_wrap("detector".into(), look.font(11.0), look.weak_text());
                let at = pos2(xd + look.px(6.0), yc + 0.5 * g.size().x);
                painter.add(egui::epaint::TextShape::new(at, g, look.weak_text()).with_angle(-0.5 * PI as f32));
            } else if b.measures() {
                let (xf, xn) = (x(g.f[k]), x(g.a[k + 1]));
                for (up, sgn) in [(true, 1.0f64), (false, -1.0)] {
                    let goes_on = (up && b.port == Port::Up) || (!up && b.port == Port::Down);
                    let first = vec![pos2(xb, y(sgn * D_BOX)), pos2(xf, y(sgn))];
                    let end = pos2(xf, y(sgn));
                    glow(end, if up { 0 } else { 1 });
                    if goes_on {
                        let mut pts = first;
                        for i in 1..=16 {
                            let s = i as f32 / 16.0;
                            pts.push(pos2(xf + (xn - xf) * s, y(sgn * (1.0 - smooth(s) as f64))));
                        }
                        beam_line(pts);
                        port_label(
                            end + vec2(0.0, -(sgn as f32) * look.px(13.0)),
                            if up { Align2::CENTER_BOTTOM } else { Align2::CENTER_TOP },
                            up,
                            "",
                            look.text(),
                        );
                    } else {
                        beam_line(first);
                        // beam stop
                        let stop = Rect::from_center_size(end, vec2(look.px(7.0), look.px(16.0)));
                        painter.rect_filled(stop, 1.0, look.weak_text());
                        port_label(end + vec2(look.px(7.0), 0.0), Align2::LEFT_CENTER, up, " stopped", look.weak_text());
                    }
                }
            } else {
                beam_line(vec![pos2(xb, yc), pos2(x(g.a[k + 1]), yc)]);
            }
        }

        // atoms
        for a in &self.atoms {
            let (off, split) = self.atom_y(a, &g);
            let p = pos2(x(a.u), y(off));
            let r = look.px(4.0);
            let col = look.accent();
            if split {
                painter.circle_filled(p, r * 0.85, col.gamma_multiply(0.55));
                painter.circle_filled(pos2(p.x, y(-off)), r * 0.85, col.gamma_multiply(0.55));
            } else {
                painter.circle_filled(p, r, col);
            }
            if Some(a.id) == self.tracked {
                painter.circle_stroke(p, r * 2.0, look.stroke(1.5, look.text()));
                if split {
                    painter.circle_stroke(pos2(p.x, y(-off)), r * 2.0, look.stroke(1.5, look.text()));
                }
            }
        }
        painter.text(
            rect.left_bottom() + vec2(look.px(8.0), -look.px(4.0)),
            Align2::LEFT_BOTTOM,
            "schematic: in every box the ↑u beam is drawn going up, ↓u down",
            look.font(11.0),
            look.weak_text(),
        );
        rect
    }

    /// Small Bloch spheres with the state leaving the source and every box.
    fn spheres_row(&self, ui: &mut egui::Ui, beam: Rect, height: f32) {
        let look = Look::of(ui);
        let (row, _) = ui.allocate_exact_size(vec2(beam.width(), height), Sense::hover());
        let g = geom(self.boxes.len());
        let last = self.last();
        let x = |u: f64| beam.left() + u as f32 * beam.width();
        let size = height.min(look.px(118.0));
        let inputs = self.ideal_inputs();
        let cloud: Vec<Vec3> = self.cloud.iter().copied().collect();
        let place = |ui: &mut egui::Ui, cx: f32| {
            let cx = cx.clamp(row.left() + 0.5 * size, row.right() - 0.5 * size);
            let r = Rect::from_center_size(pos2(cx, row.center().y), vec2(size, height));
            ui.new_child(egui::UiBuilder::new().max_rect(r))
        };
        // the source
        {
            let mut child = place(ui, x(0.035));
            let mut s = BlochSphere::new("sg_small_src", size)
                .size(size, height)
                .shared_camera("sg_small")
                .compact(true);
            s = match self.source {
                Source::Oven => s.cloud(&cloud).state([0.0; 3]).caption("oven: ρ = I/2"),
                Source::Polarised => s.state(Spinor::up_along(self.theta0, self.phi0).bloch()).caption("source |↑n₀⟩"),
            };
            s.show(&mut child);
        }
        for (k, b) in self.boxes.iter().enumerate() {
            let cx = if k == last {
                x(0.5 * (g.b[k] + g.det)) - look.px(30.0)
            } else {
                x(0.5 * (g.b[k] + g.a[k + 1]))
            };
            let mut child = place(ui, cx);
            let mut s = BlochSphere::new(("sg_small", k), size)
                .size(size, height)
                .shared_camera("sg_small")
                .compact(true)
                .axis(BlochAxis::measurement(b.dir(), "", "", look.measure().gamma_multiply(0.8)));
            let n = b.name();
            s = if b.measures() {
                if k == last {
                    s.state(b.dir())
                        .arrow(crate::spinor::scale(b.dir(), -1.0), look.accent().gamma_multiply(0.6), "")
                        .caption(format!("|↑{n}⟩ or |↓{n}⟩"))
                } else if b.port == Port::Down {
                    s.state(crate::spinor::scale(b.dir(), -1.0))
                        .caption(format!("after SG {}: |↓{n}⟩", k + 1))
                } else {
                    s.state(b.dir()).caption(format!("after SG {}: |↑{n}⟩", k + 1))
                }
            } else {
                match inputs[k] {
                    Some(st) => s.state(st.bloch()).caption("unchanged"),
                    None => s.cloud(&cloud).state([0.0; 3]).caption("still ρ = I/2"),
                }
            };
            s.show(&mut child);
        }
    }

    /// Table of the counters: measured fraction ↑ against the Born rule.
    fn table_ui(&self, ui: &mut egui::Ui) {
        let look = Look::of(ui);
        let inputs = self.ideal_inputs();
        egui::Grid::new("sg_table")
            .striped(true)
            .spacing(vec2(look.px(10.0), look.px(3.0)))
            .show(ui, |ui| {
                // (in presentation mode the axis column is left out: the boxes show it)
                for h in ["box", "axis", "input", "N", "measured P(↑)", "Born P(↑)"] {
                    if h == "axis" && look.present {
                        continue;
                    }
                    ui.label(RichText::new(h).color(look.weak_text()));
                }
                ui.end_row();
                for (k, b) in self.boxes.iter().enumerate() {
                    let st = &self.stats[k];
                    ui.label(format!("SG {}", k + 1));
                    if !look.present {
                        ui.label(b.short());
                    }
                    let input = match inputs[k] {
                        None => "ρ = I/2".to_string(),
                        Some(s) => ket_name(&s, &self.boxes[..k], self.source, (self.theta0, self.phi0)),
                    };
                    ui.label(input);
                    ui.label(RichText::new(format!("{}", st.arrived)).monospace());
                    if b.measures() {
                        let n = st.up + st.down;
                        let p = self.predicted(k);
                        if n > 0 {
                            let f = st.up as f64 / n as f64;
                            let sd = (p * (1.0 - p) / n as f64).sqrt();
                            ui.label(RichText::new(format!("{f:.3} ± {sd:.3}")).monospace().color(look.counts()));
                        } else {
                            ui.label("–");
                        }
                        ui.label(RichText::new(format!("{p:.3}")).monospace().color(look.prediction()));
                    } else {
                        ui.label(RichText::new("no measurement").color(look.weak_text()));
                        ui.label("");
                    }
                    ui.end_row();
                }
            });
        ui.label(
            RichText::new("± is the expected spread √(p(1−p)/N) of the fraction for N atoms.")
                .color(look.weak_text())
                .size(11.5 * look.font_scale()),
        );
        // the eigenvectors of the boxes in the z basis (Eq. 50.3; for x this is part a)
        let mut seen: Vec<String> = vec![];
        for b in &self.boxes {
            let key = b.describe();
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
            let (up, dn) = (Spinor::up_dir(b.dir()), Spinor::down_dir(b.dir()));
            let n = b.name();
            ui.label(
                RichText::new(format!(
                    "Eq. 50.3: |↑{n}⟩ = ({}, {})ᵀ,  |↓{n}⟩ = ({}, {})ᵀ",
                    fmt_amp(up.up),
                    fmt_amp(up.down),
                    fmt_amp(dn.up),
                    fmt_amp(dn.down)
                ))
                .color(look.text())
                .size(12.0 * look.font_scale()),
            );
        }
    }

    /// Measured ↑ fraction of the last box against θ.
    fn scan_plot(&self, ui: &mut egui::Ui, size: Vec2) {
        let look = Look::of(ui);
        let last = self.last();
        let b = self.boxes[last];
        let input = self.ideal_inputs()[last];
        let born = |theta_deg: f64| input.map_or(0.5, |s| s.p_up(dir(theta_deg.to_radians(), b.phi)));
        let label = match input {
            Some(s) if (s.bloch()[2] - 1.0).abs() < 1e-9 => "Born rule cos²(θ/2)".to_string(),
            Some(_) => "Born rule |⟨↑u|ψ⟩|²".to_string(),
            None => "Born rule 1/2 (mixture)".to_string(),
        };
        PlotFrame::new("sg_scan")
            .x_range(0.0, 180.0)
            .y_range(-0.04, 1.08)
            .labels("θ of the last box (°)", "P(↑u)")
            .pan_zoom(PanZoom::None)
            .width(size.x)
            .height(size.y)
            .show(ui, |painter, tr| {
                let xs: Vec<f64> = (0..=180).map(|i| i as f64).collect();
                let ys: Vec<f64> = xs.iter().map(|&t| born(t)).collect();
                plot_line(painter, tr, &xs, &ys, look.stroke(1.8, look.prediction()));
                let th = b.theta.to_degrees();
                painter.line_segment(
                    [tr.to_screen(th, -0.04), tr.to_screen(th, 1.08)],
                    look.stroke(1.0, look.accent().gamma_multiply(0.7)),
                );
                for p in &self.scan {
                    let f = p.up as f64 / p.n as f64;
                    let q = born(p.theta);
                    let sd = (q * (1.0 - q) / p.n as f64).sqrt();
                    error_point(painter, tr, p.theta, f, sd, look.counts());
                }
                let r = tr.rect;
                let n: u64 = self.scan.iter().map(|p| p.n).sum();
                legend(
                    painter,
                    pos2(r.right() - look.px(8.0), r.top() + look.px(6.0)),
                    &[
                        (label.clone(), look.prediction(), Key::Line),
                        ("measured ± √(p(1−p)/N)".into(), look.counts(), Key::Point),
                        (format!("{n} atoms at {} angles", self.scan.len()), look.weak_text(), Key::None),
                    ],
                );
            });
    }
}

/// A readable name of an ideal input state.
fn ket_name(s: &Spinor, before: &[SgBox], source: Source, n0: (f64, f64)) -> String {
    if let Some(b) = before.iter().rev().find(|b| b.measures()) {
        let arrow = if b.port == Port::Down { "↓" } else { "↑" };
        return format!("|{arrow}{}⟩", b.name());
    }
    let _ = s;
    match source {
        Source::Polarised => format!("|↑n₀⟩ ({}, {})", deg(n0.0), deg(n0.1)),
        Source::Oven => "ρ = I/2".into(),
    }
}

impl Scene for SgScene {
    fn time(&self) -> f64 {
        self.t
    }

    fn step(&mut self, dt: f64, ctx: &mut Ctx) {
        self.t += dt;
        if self.stream {
            self.spawn_acc += self.rate * dt;
            while self.spawn_acc >= 1.0 {
                self.spawn_acc -= 1.0;
                self.spawn(ctx);
            }
        }
        self.step_atoms(dt, ctx);
    }

    fn reset(&mut self, _ctx: &mut Ctx) {
        self.restart();
    }

    fn measure_label(&self) -> Option<String> {
        Some("Send one atom".into())
    }

    fn measure(&mut self, ctx: &mut Ctx) {
        self.send_one(ctx);
    }

    fn action(&mut self, action: &str, ctx: &mut Ctx) -> bool {
        let mut words = action.split_whitespace();
        let cmd = words.next().unwrap_or("");
        let arg = words.next();
        let n = arg.and_then(|w| w.parse().ok()).unwrap_or(self.repeat_n);
        let last = self.last();
        match cmd {
            "send" | "measure" => self.send_one(ctx),
            "batch" | "repeat" => self.batch_send(n, ctx),
            "scan" => self.scan_theta(n, ctx),
            "stream_off" => self.stream = false,
            "stream_on" => self.stream = true,
            "rate" => self.rate = arg.and_then(|w| w.parse().ok()).unwrap_or(self.rate),
            "theta" => {
                let Some(v) = arg.and_then(|w| w.parse::<f64>().ok()) else {
                    return false;
                };
                self.boxes[last].theta = v.to_radians();
                self.chain_changed();
            }
            "polarised" => {
                self.source = Source::Polarised;
                self.chain_changed();
            }
            "oven" => {
                self.source = Source::Oven;
                self.chain_changed();
            }
            "recombine" => {
                // recombine box k (1-based)
                let k: usize = arg.and_then(|w| w.parse().ok()).unwrap_or(2);
                if k >= 1 && k <= self.boxes.len() {
                    self.boxes[k - 1].port = Port::Both;
                    self.chain_changed();
                }
            }
            _ => return false,
        }
        true
    }

    fn central_ui(&mut self, ui: &mut egui::Ui, _ctx: &mut Ctx) {
        let look = Look::of(ui);
        let last = self.last();
        let st = self.stats[last];
        let done = st.up + st.down;
        let mut items = vec![
            ("atoms sent", format!("{}", self.sent), None),
            ("in flight", format!("{}", self.atoms.len()), None),
        ];
        if self.boxes[last].measures() {
            let p = self.predicted(last);
            let name = self.boxes[last].name();
            items.push((
                "last box: measured P(↑)",
                if done > 0 {
                    format!("{:.3} ± {:.3}", st.up as f64 / done as f64, (p * (1.0 - p) / done as f64).sqrt())
                } else {
                    "–".into()
                },
                Some(look.counts()),
            ));
            items.push(("Born P(↑)", format!("{p:.3}  (↑{name})"), Some(look.prediction())));
        }
        readout_row(ui, &items.iter().map(|(a, b, c)| (*a, b.clone(), *c)).collect::<Vec<_>>());

        let avail = ui.available_size();
        let beam_h = (avail.y * 0.3).clamp(look.px(170.0), look.px(270.0));
        let row_h = look.px(112.0).min(avail.y * 0.15);
        let beam = self.beam_ui(ui, vec2(avail.x, beam_h));
        self.spheres_row(ui, beam, row_h);
        ui.add_space(look.px(4.0));

        let bottom_h = ui.available_height();
        let tracked_box = self.tracked_box.min(last);
        let b = self.boxes[tracked_box];
        let name = b.name();
        let total_w = ui.available_width();
        let gap = ui.spacing().item_spacing.x;
        let sph = bottom_h.min(look.px(330.0)).min(total_w * if look.present { 0.25 } else { 0.3 });
        let mid_w = if look.present { look.px(215.0) } else { look.px(250.0) };
        let right_w = (total_w - sph - mid_w - 2.0 * gap).max(look.px(200.0));
        ui.horizontal(|ui| {
            let caption = if self.tracked.is_none() {
                "send an atom (M)".to_string()
            } else if let Some(f) = &self.tracked_fate {
                f.clone()
            } else if self.tracked_in_field {
                format!("in SG {}: precessing about {name}", tracked_box + 1)
            } else {
                format!("next / last box: SG {} ({name})", tracked_box + 1)
            };
            let resp = BlochSphere::new("sg_main", sph)
                .size(sph, bottom_h)
                .state(self.tracked_spin.bloch())
                .trail(&self.trail)
                .axis(BlochAxis::measurement(
                    b.dir(),
                    format!("↑{name}"),
                    format!("↓{name}"),
                    look.measure(),
                ))
                .pole_labels(false)
                .title("latest atom")
                .caption(caption)
                .show(ui);
            self.flash
                .paint_banner(ui, Rect::from_min_max(resp.rect.min + vec2(0.0, look.px(18.0)), resp.rect.max));

            ui.vertical(|ui| {
                let w = mid_w;
                ui.set_width(w);
                let button = format!("SG {}", tracked_box + 1);
                let axis = SceneAxis {
                    dir: b.dir(),
                    name,
                    button: &button,
                };
                state_panel(ui, &self.tracked_spin, &mut self.axis_choice, Some(axis), w);
                ui.add_space(look.px(4.0));
                ui.label(RichText::new(format!("levels in the field of SG {} (E in ħω)", tracked_box + 1)).color(look.weak_text()));
                let c = [
                    Spinor::down_dir(b.dir()).inner(&self.tracked_spin),
                    Spinor::up_dir(b.dir()).inner(&self.tracked_spin),
                ];
                let labels = [format!("|↓{name}⟩"), format!("|↑{name}⟩")];
                let lh = (ui.available_height() - look.px(4.0)).max(look.px(90.0));
                LevelLadder::new("sg_ladder", &[-0.5, 0.5])
                    .coeffs(&c)
                    .labels(&labels)
                    .energy_range(-0.9, 0.9)
                    .size(vec2(w, lh))
                    .show(ui);
            });

            ui.vertical(|ui| {
                ui.set_width(right_w);
                egui::ScrollArea::horizontal()
                    .id_salt("sg_table_scroll")
                    .max_width(right_w)
                    .show(ui, |ui| self.table_ui(ui));
                ui.add_space(look.px(4.0));
                let h = ui.available_height();
                let w = right_w;
                if self.scanning() {
                    self.scan_plot(ui, vec2(w, h));
                } else if self.boxes[last].measures() {
                    let mut hist = ShotHistogram::discrete(2);
                    hist.counts = vec![st.up, st.down];
                    let p = self.predicted(last);
                    let labels = [format!("↑{}", self.boxes[last].name()), format!("↓{}", self.boxes[last].name())];
                    HistogramPlot::new("sg_hist", &hist)
                        .predicted(&[p, 1.0 - p])
                        .labels(&labels)
                        .x_label(format!("SG {} outcome", last + 1))
                        .scale(HistScale::Fraction)
                        .y_max(1.05)
                        .width(w)
                        .height(h)
                        .show(ui);
                }
            });
        });
    }

    fn side_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx) {
        let look = ctx.look;
        let mut changed = false;
        ui.label(RichText::new("Source").strong());
        segmented(ui, |ui| {
            changed |= ui.selectable_value(&mut self.source, Source::Oven, "oven (unpolarised)").changed();
            changed |= ui
                .selectable_value(&mut self.source, Source::Polarised, "polarised |↑n₀⟩")
                .changed();
        });
        if self.source == Source::Polarised {
            let mut t = self.theta0.to_degrees();
            let mut p = self.phi0.to_degrees();
            changed |= ui.add(egui::Slider::new(&mut t, 0.0..=180.0).text("θ₀").suffix("°")).changed();
            changed |= ui.add(egui::Slider::new(&mut p, -180.0..=180.0).text("φ₀").suffix("°")).changed();
            self.theta0 = t.to_radians();
            self.phi0 = p.to_radians();
        }

        ui.add_space(look.px(8.0));
        ui.horizontal(|ui| {
            ui.label(RichText::new("Boxes").strong());
            if ui.add_enabled(self.boxes.len() > 1, egui::Button::new("−")).clicked() {
                self.boxes.pop();
                changed = true;
            }
            ui.label(format!("{}", self.boxes.len()));
            if ui.add_enabled(self.boxes.len() < 4, egui::Button::new("+")).clicked() {
                let mut b = SgBox::new(AxisSel::Z);
                b.port = Port::Up;
                self.boxes.push(b);
                changed = true;
            }
        });
        let n = self.boxes.len();
        for k in 0..n {
            let b = &mut self.boxes[k];
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("SG {}", k + 1)).color(look.accent()));
                segmented(ui, |ui| {
                    for (a, l) in [(AxisSel::Z, "z"), (AxisSel::X, "x"), (AxisSel::Y, "y"), (AxisSel::Angle, "u(θ, φ)")] {
                        changed |= ui.selectable_value(&mut b.axis, a, l).changed();
                    }
                });
            });
            if b.axis == AxisSel::Angle {
                let mut t = b.theta.to_degrees();
                let mut p = b.phi.to_degrees();
                changed |= ui.add(egui::Slider::new(&mut t, 0.0..=180.0).text("θ").suffix("°")).changed();
                changed |= ui.add(egui::Slider::new(&mut p, -180.0..=180.0).text("φ").suffix("°")).changed();
                b.theta = t.to_radians();
                b.phi = p.to_radians();
            }
            ui.horizontal(|ui| {
                if k + 1 < n {
                    ui.label(RichText::new("continue").color(look.weak_text()));
                    segmented(ui, |ui| {
                        changed |= ui.selectable_value(&mut b.port, Port::Up, "↑").changed();
                        changed |= ui.selectable_value(&mut b.port, Port::Down, "↓").changed();
                        changed |= ui
                            .selectable_value(&mut b.port, Port::Both, "both, recombined")
                            .on_hover_text("no which-path information: no measurement, the spin passes unchanged")
                            .changed();
                    });
                } else {
                    ui.label(RichText::new("outputs").color(look.weak_text()));
                    let mut rec = b.port == Port::Both;
                    segmented(ui, |ui| {
                        changed |= ui.selectable_value(&mut rec, false, "detect ↑ and ↓").changed();
                        changed |= ui
                            .selectable_value(&mut rec, true, "recombined")
                            .on_hover_text("no which-path information: no measurement")
                            .changed();
                    });
                    b.port = if rec {
                        Port::Both
                    } else if b.port == Port::Both {
                        Port::Up
                    } else {
                        b.port
                    };
                }
            });
        }
        if changed {
            self.chain_changed();
        }

        ui.add_space(look.px(8.0));
        ui.label(RichText::new("Atoms").strong());
        ui.horizontal(|ui| {
            if ui.button(RichText::new("Send one atom  (M)").color(look.measure())).clicked() {
                self.send_one(ctx);
            }
            ui.checkbox(&mut self.stream, "stream");
        });
        if self.stream {
            ui.add(egui::Slider::new(&mut self.rate, 0.5..=30.0).logarithmic(true).text("atoms / s"));
        }
        if let Some(n) = repeat_button(ui, "sg_batch", "Send N at once", &mut self.repeat_n) {
            self.batch_send(n, ctx);
        }
        if self.scanning() && ui.button(format!("Scan θ: N = {} at 19 angles", self.repeat_n)).clicked() {
            self.scan_theta(self.repeat_n, ctx);
        }
        ui.horizontal(|ui| {
            if ui.button("Clear counters").clicked() {
                self.restart();
            }
            if self.scanning() && ui.button("Clear θ plot").clicked() {
                self.scan.clear();
            }
        });
        ui.label(
            RichText::new(
                "The stream only flows while playing (Space). Send N: N atoms through the chain at once, counted at every port. Changing the chain clears the counters.",
            )
            .color(look.weak_text())
            .size(12.0 * look.font_scale()),
        );
        if let Some(l) = self.flash.label() {
            ui.label(RichText::new(format!("latest atom: {l}")).color(look.measure()));
        }
    }

    fn notes(&self) -> String {
        let specific = match self.preset {
            SgPreset::ZX => NOTES_ZX,
            SgPreset::ZXZ => NOTES_ZXZ,
            SgPreset::ZZ => NOTES_ZZ,
            SgPreset::ZTheta => NOTES_ZTHETA,
            SgPreset::Oven => NOTES_OVEN,
        };
        format!("{specific}\n\n{NOTES_COMMON}")
    }
}

const NOTES_ZX: &str = "\
# The exercise (PS13 50 a, b)
Silver atoms (one unpaired electron: spin ½) leave the oven with random spins. Box 1 measures along z and only its ↑ beam goes on, so every atom entering box 2 is in |↑z⟩ = (1, 0)ᵀ. Box 2 measures along x.
- (a) Eq. 50.3 with θ = π/2, φ = 0: |↑x⟩ = (|↑z⟩ + |↓z⟩)/√2 = (1, 1)ᵀ/√2 and |↓x⟩ = −(|↑z⟩ − |↓z⟩)/√2 = (−1, 1)ᵀ/√2. Check: σx|↑x⟩ = +|↑x⟩, σx|↓x⟩ = −|↓x⟩.
- (b) P(↑x|↑z) = |⟨↑x|↑z⟩|² = |1/√2|² = 1/2: half of the atoms go up in box 2, half down.

# Try
- Play: atoms fly through one by one. Watch the latest atom on the big sphere: random direction from the oven, collapse onto ±z in box 1, onto ±x in box 2.
- Send N = 10 000 at once and compare the fractions with 0.500 in the table.
- Let box 1 pass ↓z: box 2 still gives 50/50. Set box 2 to y: also 50/50 (|⟨↑y|↑z⟩|² = 1/2).";

const NOTES_ZXZ: &str = "\
# The exercise (PS13 50 c)
After box 1 (z, ↑ only) and box 2 (x, ↑ only) the atoms are in |↑x⟩. Box 3 measures z again and finds ↑z with P(↑z|↑x) = |⟨↑z|↑x⟩|² = 1/2, although every one of these atoms was ↑z before box 2.
- \"It was ↑z, so it must still be ↑z\" fails because box 2 measured: it projected the state onto |↑x⟩, which is not an eigenstate of σz. Without box 2 the atoms stay in |↑z⟩ and box 3 gives ↑z every time (z–z preset).
- Of all atoms leaving the oven, 1/2 · 1/2 · 1/2 = 1/8 end on the ↑z detector of box 3.

# Try
- Set box 2 to \"both, recombined\": no which-path information, no measurement. Box 3 now gives ↑z every time.
- Let box 2 pass ↓x instead: still 50 % ↑z at box 3.";

const NOTES_ZZ: &str = "\
# z, then z again
Box 2 measures the same observable as box 1. The atoms arrive in |↑z⟩, an eigenstate of σz, so P(↑z) = |⟨↑z|↑z⟩|² = 1: a repeated measurement repeats its result.

# Try
- Let box 1 pass ↓z: box 2 then gives ↓z every time.
- Switch box 2 to u(θ, φ) and tilt it: P(↑u|↑z) = cos²(θ/2).
- Put an x box in between (+, then choose x for box 2 and z for box 3): the certainty is gone (z–x–z preset).";

const NOTES_ZTHETA: &str = "\
# z, then an axis at angle θ
After box 1 the atoms are in |↑z⟩. Box 2 measures along u = (sin θ cos φ, sin θ sin φ, cos θ). With Eq. 50.3a, ⟨↑u|↑z⟩ = cos(θ/2) e^(iφ/2), so P(↑u) = cos²(θ/2) for every φ: 1 at θ = 0, 1/2 at θ = 90° (any axis in the x–y plane, e.g. x), 0 at θ = 180°.
- On the Bloch sphere cos²(θ/2) = (1 + cos θ)/2 = (1 + n·u)/2: only the angle between the state's Bloch vector n and the axis u matters.

# Try
- Move θ while the stream runs: the measured points fill in at each angle. Or press \"Scan θ\" (N atoms at each of 19 angles).
- The error bars are the expected spread √(p(1−p)/N) for N atoms: zero at θ = 0° and 180°, largest at 90°.
- Change φ: the curve does not change.";

const NOTES_OVEN: &str = "\
# An unpolarised beam is a mixture
Each atom leaves the oven in some pure spin state, but nobody knows which: a statistical mixture, described by the density matrix ρ = I/2 (Bloch vector r = 0, the dot in the centre). It is not a superposition.
- Every axis gives 50/50: P(↑u) = Tr(ρ |↑u⟩⟨↑u|) = 1/2 for all u. No pure state does that: for |↑n⟩ the axis u = n gives ↑ every time.
- Here every atom gets a random direction, uniform on the sphere. For any measurement this is the same as ρ = I/2, and also the same as a 50/50 mix of |↑z⟩ and |↓z⟩: no experiment can tell these preparations apart.

# Try
- Scan θ: flat at 1/2 within the error bars.
- Switch the source to \"polarised |↑n₀⟩\": now the result depends on the axis, cos²(α/2) with α the angle between n₀ and u.
- Add a z box in front (+, then make box 1 z with ↑ only): after it the beam is pure, |↑z⟩.";

const NOTES_COMMON: &str = "\
# How the picture works
- In every measuring box the outcome is drawn with the Born rule P(↑u) = |⟨↑u|ψ⟩|²; leaving the box the atom is in |↑u⟩ or |↓u⟩ of Eq. 50.3 (the collapse; the flash on the big sphere). Counters sit at every port; atoms on a blocked port end on a beam stop.
- \"both, recombined\": the two beams are brought together again without recording the path. That is no measurement and the spin leaves as it entered (an idealisation: for atoms such an interferometer is extremely hard to build).
- Schematic: the ↑u beam is always drawn going up. A real x box deflects sideways.
- Energy levels: the strong field B along u of a box makes |↑u⟩, |↓u⟩ the energy eigenstates (H = μ_B B σ_u, splitting ħω = 2μ_B B). In the field the phasors turn (the spin precesses about u; here slowed down to 1.5 turns per box, really ω/2π ≈ 28 GHz per tesla), but |c|² stays fixed. The field gradient pushes the two energy eigenstates apart (force −∇E), which sorts them into the two beams.
- The small spheres show the state leaving each box; the big one follows one atom at a time.";

#[cfg(test)]
mod tests {
    use super::*;

    fn run(preset: SgPreset, n: usize, seed: u64) -> SgScene {
        let mut s = SgScene::new(preset);
        let mut rng = QbRng::seed(seed);
        // a Ctx is not constructible outside qb_core: use the batch logic directly
        let last = s.last();
        for _ in 0..n {
            let mut spin = s.source_state(&mut rng);
            for k in 0..s.boxes.len() {
                let b = s.boxes[k];
                s.stats[k].arrived += 1;
                if !b.measures() {
                    continue;
                }
                let (up, st) = measure(&spin, b.dir(), &mut rng);
                spin = st;
                if up {
                    s.stats[k].up += 1
                } else {
                    s.stats[k].down += 1
                }
                if k == last || !((up && b.port == Port::Up) || (!up && b.port == Port::Down)) {
                    break;
                }
            }
        }
        s
    }

    fn frac(s: &SgScene, k: usize) -> (f64, u64) {
        let st = s.stats[k];
        (st.up as f64 / (st.up + st.down) as f64, st.up + st.down)
    }

    #[test]
    fn predicted_probabilities() {
        assert!((SgScene::new(SgPreset::ZX).predicted(1) - 0.5).abs() < 1e-12);
        assert!((SgScene::new(SgPreset::ZXZ).predicted(2) - 0.5).abs() < 1e-12);
        assert!((SgScene::new(SgPreset::ZZ).predicted(1) - 1.0).abs() < 1e-12);
        assert!((SgScene::new(SgPreset::Oven).predicted(0) - 0.5).abs() < 1e-12);
        let mut s = SgScene::new(SgPreset::ZTheta);
        for i in 0..=18 {
            let th = (i as f64 * 10.0).to_radians();
            s.boxes[1].theta = th;
            s.boxes[1].phi = 0.3 * i as f64;
            assert!((s.predicted(1) - (0.5 * th).cos().powi(2)).abs() < 1e-12);
        }
        // recombining box 2 of z–x–z: box 3 gets |↑z⟩ back
        let mut s = SgScene::new(SgPreset::ZXZ);
        s.boxes[1].port = Port::Both;
        assert!((s.predicted(2) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn simulated_fractions() {
        for (preset, k, p) in [
            (SgPreset::ZX, 1, 0.5),
            (SgPreset::ZXZ, 2, 0.5),
            (SgPreset::ZZ, 1, 1.0),
            (SgPreset::Oven, 0, 0.5),
        ] {
            let s = run(preset, 40_000, 11);
            let (f, n) = frac(&s, k);
            let sd = (p * (1.0 - p) / n as f64).sqrt().max(1e-9);
            assert!((f - p).abs() <= 5.0 * sd, "{preset:?}: {f} vs {p} (N = {n})");
        }
        // z–x–z: about 1/8 of the oven's atoms reach the ↑z detector of box 3
        let s = run(SgPreset::ZXZ, 80_000, 5);
        let f = s.stats[2].up as f64 / 80_000.0;
        assert!((f - 0.125).abs() < 0.006, "{f}");
    }

    #[test]
    fn atom_path_geometry() {
        let g = geom(3);
        for k in 0..3 {
            assert!(g.a[k] < g.b[k] && g.b[k] < g.f[k]);
            if k + 1 < 3 {
                assert!(g.f[k] < g.a[k + 1]);
            }
        }
        assert!(g.b[2] < g.det && g.det < 1.0);
    }
}
