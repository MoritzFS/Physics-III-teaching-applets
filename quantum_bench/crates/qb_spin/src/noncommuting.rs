//! PS09 35 · Measurement of non-commuting observables.
//!
//! A two-state system with the eigenbasis {ψ₁, ψ₂} of Â and {φ₁, φ₂} of B̂,
//! ψ₁ = cos β φ₁ + sin β φ₂, ψ₂ = sin β φ₁ − cos β φ₂ (the exercise: cos β = 3/5).
//! All amplitudes are real, so the state is drawn as a unit vector in a plane
//! with both bases as rotated pairs of axes: a measurement projects onto one
//! axis of the basis (probability = squared projection) and the state jumps
//! there. The Bloch sphere is a secondary view. No time evolution.

use qb_core::egui::{self, Align2, Color32, Pos2, RichText, Sense, Shape, Vec2, pos2, vec2};
use qb_core::num::ShotHistogram;
use qb_core::widgets::{FlashKind, HistScale, HistogramPlot, MeasureFlash, label_box, readout_row, repeat_button};
use qb_core::{Ctx, Look, QbRng, Scene, wall_time};

use crate::bloch::{BlochAxis, BlochSphere, Camera};
use crate::spinor::Spinor;
use crate::util::{arrow, deg, num, panel, smooth};

/// Which observable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Obs {
    A,
    B,
}

impl Obs {
    fn outcome(self, k: usize) -> &'static str {
        match (self, k) {
            (Obs::A, 0) => "a₁",
            (Obs::A, _) => "a₂",
            (Obs::B, 0) => "b₁",
            (Obs::B, _) => "b₂",
        }
    }
    fn vector(self, k: usize) -> &'static str {
        match (self, k) {
            (Obs::A, 0) => "ψ₁",
            (Obs::A, _) => "ψ₂",
            (Obs::B, 0) => "φ₁",
            (Obs::B, _) => "φ₂",
        }
    }
    fn name(self) -> &'static str {
        match self {
            Obs::A => "A",
            Obs::B => "B",
        }
    }
}

/// The exercise's mixing angle: cos β = 3/5, sin β = 4/5 (β ≈ 53.13°).
pub fn beta_345() -> f64 {
    4f64.atan2(3.0)
}

/// Eigenvector k of the observable in the φ basis (components along φ₁, φ₂).
pub fn eigvec(obs: Obs, k: usize, beta: f64) -> [f64; 2] {
    let (s, c) = beta.sin_cos();
    match (obs, k) {
        (Obs::B, 0) => [1.0, 0.0],
        (Obs::B, _) => [0.0, 1.0],
        (Obs::A, 0) => [c, s],
        (Obs::A, _) => [s, -c],
    }
}

fn dot2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

/// One Born-rule measurement: (outcome index, state after it = P̂ψ/‖P̂ψ‖, probability of that outcome).
pub fn sample(obs: Obs, state: [f64; 2], beta: f64, rng: &mut QbRng) -> (usize, [f64; 2], f64) {
    let amp = [dot2(eigvec(obs, 0, beta), state), dot2(eigvec(obs, 1, beta), state)];
    let p0 = amp[0] * amp[0] / (amp[0] * amp[0] + amp[1] * amp[1]);
    let k = if rng.bool(p0) { 0 } else { 1 };
    let e = eigvec(obs, k, beta);
    let sign = if amp[k] < 0.0 { -1.0 } else { 1.0 };
    (k, [sign * e[0], sign * e[1]], if k == 0 { p0 } else { 1.0 - p0 })
}

/// Born probabilities of the four paths of the exercise, starting from ψ₁ and
/// measuring B then A: [b₁a₁, b₁a₂, b₂a₁, b₂a₂].
pub fn path_probabilities(beta: f64) -> [f64; 4] {
    let psi1 = eigvec(Obs::A, 0, beta);
    let mut out = [0.0; 4];
    for kb in 0..2 {
        let phi = eigvec(Obs::B, kb, beta);
        let pb = dot2(phi, psi1).powi(2);
        for ka in 0..2 {
            out[2 * kb + ka] = pb * dot2(eigvec(Obs::A, ka, beta), phi).powi(2);
        }
    }
    out
}

struct LogEntry {
    obs: Obs,
    k: usize,
    p: f64,
    negative: bool,
}

/// The collapse animation: the tip drops onto the axis, then grows to unit length.
struct Anim {
    obs: Obs,
    from: [f64; 2],
    foot: [f64; 2],
    to: [f64; 2],
    start: f64,
}

const DROP: f64 = 0.45;
const GROW: f64 = 0.35;

struct Tree {
    beta: f64,
    paths: [u64; 4],
    hist: ShotHistogram,
}

impl Tree {
    fn runs(&self) -> u64 {
        self.paths.iter().sum()
    }
}

pub struct NonCommutingScene {
    beta: f64,
    /// angle of the prepared start state χ from φ₁ (radians)
    prep: f64,
    /// start (and reset) in ψ₁, as after part a) of the exercise, instead of in χ
    start_psi1: bool,
    state: [f64; 2],
    last: Option<Obs>,
    log: Vec<LogEntry>,
    anim: Option<Anim>,
    flash: MeasureFlash,
    tree: Option<Tree>,
    repeat_n: usize,
}

impl Default for NonCommutingScene {
    fn default() -> Self {
        Self::new()
    }
}

impl NonCommutingScene {
    pub fn new() -> Self {
        let mut s = NonCommutingScene {
            beta: beta_345(),
            prep: 75f64.to_radians(),
            start_psi1: true,
            state: [0.0, 1.0],
            last: None,
            log: vec![],
            anim: None,
            flash: MeasureFlash::new(),
            tree: None,
            repeat_n: 1000,
        };
        s.restart();
        s
    }

    /// Back to the chosen start state: ψ₁ (the exercise) or the prepared χ.
    fn restart(&mut self) {
        if self.start_psi1 {
            self.set_psi1();
        } else {
            self.set_chi();
        }
    }

    fn set_chi(&mut self) {
        self.state = [self.prep.cos(), self.prep.sin()];
        self.last = None;
        self.log.clear();
        self.anim = None;
        self.flash.clear();
    }

    /// Start as after part a): A gave a₁, the state is ψ₁.
    fn set_psi1(&mut self) {
        self.state = eigvec(Obs::A, 0, self.beta);
        self.last = Some(Obs::A);
        self.log.clear();
        self.log.push(LogEntry {
            obs: Obs::A,
            k: 0,
            p: 1.0,
            negative: false,
        });
        self.anim = None;
        self.flash.clear();
    }

    fn is_345(&self) -> bool {
        (self.beta - beta_345()).abs() < 1e-9
    }

    /// The observable the walk measures next (A, B, A, B, …).
    fn next_obs(&self) -> Obs {
        if self.last == Some(Obs::A) { Obs::B } else { Obs::A }
    }

    fn measure(&mut self, obs: Obs, ctx: &mut Ctx) {
        let from = self.state;
        let (k, to, p) = sample(obs, from, self.beta, ctx.rng);
        let e = eigvec(obs, k, self.beta);
        let a = dot2(e, from);
        let negative = to[0] * e[0] + to[1] * e[1] < 0.0;
        self.anim = Some(Anim {
            obs,
            from,
            foot: [a * e[0], a * e[1]],
            to,
            start: wall_time(),
        });
        self.state = to;
        self.last = Some(obs);
        self.log.push(LogEntry { obs, k, p, negative });
        let v = format!("{}{}", if negative { "−" } else { "" }, obs.vector(k));
        self.flash
            .trigger(FlashKind::Other, format!("{} → {}: state {v}", obs.name(), obs.outcome(k)));
        ctx.set_status(format!(
            "measured {}: {} (probability {:.3}); the state is now {v}{}",
            obs.name(),
            obs.outcome(k),
            p,
            if negative {
                format!(" (= {} up to an overall sign)", obs.vector(k))
            } else {
                String::new()
            }
        ));
    }

    /// N fresh systems prepared in ψ₁, each measured B then A.
    fn repeat(&mut self, n: usize, ctx: &mut Ctx) {
        let beta = self.beta;
        let tree = match &mut self.tree {
            Some(t) if t.beta == beta => t,
            _ => {
                self.tree = Some(Tree {
                    beta,
                    paths: [0; 4],
                    hist: ShotHistogram::discrete(4),
                });
                self.tree.as_mut().unwrap()
            }
        };
        let psi1 = eigvec(Obs::A, 0, beta);
        for _ in 0..n {
            let (kb, s, _) = sample(Obs::B, psi1, beta, ctx.rng);
            let (ka, _, _) = sample(Obs::A, s, beta, ctx.rng);
            tree.paths[2 * kb + ka] += 1;
            tree.hist.add_index(2 * kb + ka);
        }
        let runs = tree.runs();
        let a1 = tree.paths[0] + tree.paths[2];
        let pred = path_probabilities(beta);
        ctx.set_status(format!(
            "{n} more sequences ψ₁ → B → A ({runs} in all): a₁ at the end {:.3}, Born rule {:.4}",
            a1 as f64 / runs as f64,
            pred[0] + pred[2]
        ));
    }

    /// The state drawn now (follows the collapse animation).
    fn shown_state(&self, ui: &egui::Ui) -> [f64; 2] {
        let Some(a) = &self.anim else { return self.state };
        let age = wall_time() - a.start;
        if age < DROP + GROW {
            ui.ctx().request_repaint();
        }
        let lerp = |p: [f64; 2], q: [f64; 2], t: f32| {
            let t = smooth(t) as f64;
            [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]
        };
        if age < DROP {
            lerp(a.from, a.foot, (age / DROP) as f32)
        } else if age < DROP + GROW {
            lerp(a.foot, a.to, ((age - DROP) / GROW) as f32)
        } else {
            self.state
        }
    }

    fn colour(look: &Look, obs: Obs) -> Color32 {
        match obs {
            Obs::A => look.series(0),
            Obs::B => look.series(1),
        }
    }

    /// The plane of real states with both bases.
    fn plane(&self, ui: &mut egui::Ui, size: Vec2) {
        let look = Look::of(ui);
        let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
        let painter = ui.painter_at(rect);
        panel(&painter, rect);
        let c = rect.center() + vec2(0.0, look.px(6.0));
        let r = 0.37 * rect.width().min(rect.height());
        let to = |v: [f64; 2]| c + vec2(v[0] as f32, -(v[1] as f32)) * r;
        painter.text(
            rect.left_top() + vec2(look.px(8.0), look.px(6.0)),
            Align2::LEFT_TOP,
            "real state vector in the plane (φ₁, φ₂)",
            look.font(12.5),
            look.text(),
        );
        painter.circle_stroke(c, r, look.stroke(1.0, look.plot_frame()));

        let shown = self.shown_state(ui);
        let animating = self.anim.as_ref().is_some_and(|a| wall_time() - a.start < DROP + GROW + 0.6);
        let proj_obs = match &self.anim {
            Some(a) if animating => a.obs,
            _ => self.next_obs(),
        };

        // the two bases: full axes, unit eigenvectors as arrows, labels
        for obs in [Obs::B, Obs::A] {
            let col = Self::colour(&look, obs);
            let hi = obs == proj_obs;
            for k in 0..2 {
                let e = eigvec(obs, k, self.beta);
                let far = |s: f64| to([s * e[0], s * e[1]]);
                painter.extend(Shape::dashed_line(
                    &[far(-1.18), c],
                    look.stroke(1.0, col.gamma_multiply(0.6)),
                    5.0,
                    4.0,
                ));
                painter.line_segment([c, far(1.18)], look.stroke(1.0, col.gamma_multiply(0.6)));
                arrow(&painter, c, far(1.0), look.stroke(if hi { 2.6 } else { 1.6 }, col), look.px(10.0));
                let label = format!("{} ({})", obs.vector(k), obs.outcome(k));
                let lp = far(1.3);
                painter.text(lp, Align2::CENTER_CENTER, label, look.font(if hi { 15.0 } else { 13.0 }), col);
            }
        }
        // the angle β between φ₁ and ψ₁
        let arc: Vec<Pos2> = (0..=24)
            .map(|i| {
                let a = self.beta * i as f64 / 24.0;
                to([0.24 * a.cos(), 0.24 * a.sin()])
            })
            .collect();
        painter.add(Shape::line(arc, look.stroke(1.2, look.weak_text())));
        let mid = 0.5 * self.beta;
        painter.text(
            to([0.33 * mid.cos(), 0.33 * mid.sin()]),
            Align2::CENTER_CENTER,
            format!("β = {}", deg(self.beta)),
            look.font(12.0),
            look.weak_text(),
        );

        // projections of the state onto the basis measured next (or just measured)
        let col = Self::colour(&look, proj_obs);
        let s_len = (shown[0] * shown[0] + shown[1] * shown[1]).sqrt().max(1e-9);
        let unit = [shown[0] / s_len, shown[1] / s_len];
        if !animating || self.anim.as_ref().is_some_and(|a| wall_time() - a.start < DROP) {
            let src = if animating { self.anim.as_ref().unwrap().from } else { unit };
            for k in 0..2 {
                let e = eigvec(proj_obs, k, self.beta);
                let a = dot2(e, src);
                let foot = [a * e[0], a * e[1]];
                painter.extend(Shape::dashed_line(
                    &[to(src), to(foot)],
                    look.stroke(1.2, look.text().gamma_multiply(0.7)),
                    4.0,
                    3.0,
                ));
                painter.line_segment([c, to(foot)], look.stroke(5.0, col.gamma_multiply(0.45)));
                painter.circle_filled(to(foot), look.px(3.5), col);
                if !animating {
                    // the amplitude beside the foot point, on the side of the axis away from the state
                    let perp = [-e[1], e[0]];
                    let side = if dot2([src[0] - foot[0], src[1] - foot[1]], perp) > 0.0 {
                        -1.0
                    } else {
                        1.0
                    };
                    let n = vec2((side * perp[0]) as f32, -(side * perp[1]) as f32); // screen direction
                    let align = if n.x >= 0.0 { Align2::LEFT_CENTER } else { Align2::RIGHT_CENTER };
                    painter.text(to(foot) + n * look.px(9.0), align, num(a, 3), look.font(13.0), col);
                }
            }
            if !animating {
                // the projections in words, in the empty lower-left corner
                let mut y = rect.bottom() - look.px(8.0);
                for k in (0..2).rev() {
                    let a = dot2(eigvec(proj_obs, k, self.beta), src);
                    let txt = format!(
                        "⟨{}|ψ⟩ = {}  →  P({}) = {:.3}",
                        proj_obs.vector(k),
                        num(a, 3),
                        proj_obs.outcome(k),
                        a * a
                    );
                    let r = label_box(&painter, pos2(rect.left() + look.px(8.0), y), Align2::LEFT_BOTTOM, &txt, col);
                    y = r.top() - look.px(3.0);
                }
                painter.text(
                    pos2(rect.left() + look.px(8.0), y),
                    Align2::LEFT_BOTTOM,
                    format!("measuring {} next: projections (Born rule)", proj_obs.name()),
                    look.font(12.0),
                    look.weak_text(),
                );
            }
        }
        // the state
        let acc = look.accent();
        arrow(&painter, c, to(shown), look.stroke(3.0, acc), look.px(13.0));
        let tip = to([shown[0] * 1.12, shown[1] * 1.12]);
        painter.text(tip, Align2::CENTER_CENTER, "ψ", look.font(16.0), acc);
        if let Some(a) = &self.anim {
            let age = (wall_time() - a.start) as f32;
            if age < 0.5 {
                painter.circle_filled(
                    to(a.foot),
                    look.px(14.0) * (1.0 + age),
                    look.measure().gamma_multiply(0.5 * (1.0 - 2.0 * age)),
                );
            }
        }
        self.flash.paint_banner(ui, rect);
    }

    /// The guided walk through the exercise and the outcome log.
    fn walk_ui(&self, ui: &mut egui::Ui) {
        let look = Look::of(ui);
        let (s, c) = self.beta.sin_cos();
        let (c2, s2) = (c * c, s * s);
        let f = self.is_345();
        let fr = |x: f64, exact: &str| if f { format!("{exact} = {x:.3}") } else { format!("{x:.3}") };
        let psi1 = if f {
            "ψ₁ = (3φ₁ + 4φ₂)/5".to_string()
        } else {
            format!("ψ₁ = {:.3} φ₁ + {:.3} φ₂", c, s)
        };
        // which step is next
        let stage = match self.log.last() {
            Some(e) if e.obs == Obs::A && e.k == 0 => 1,
            Some(e) if e.obs == Obs::B => 2,
            _ => 0,
        };
        let line = |ui: &mut egui::Ui, i: usize, text: String| {
            let (mark, col) = if i == stage {
                ("▶ ", look.accent())
            } else {
                ("   ", look.text())
            };
            ui.label(RichText::new(format!("{mark}{text}")).color(col));
        };
        ui.label(RichText::new("The exercise").strong());
        line(ui, 0, format!("(a) A gives a₁ → state {psi1}"));
        line(ui, 1, format!("(b) B: b₁ with {},  b₂ with {}", fr(c2, "9/25"), fr(s2, "16/25")));
        line(
            ui,
            2,
            format!(
                "(c) A again: P(a₁) = ({})² + ({})² = {}",
                if f { "9/25".to_string() } else { format!("{c2:.3}") },
                if f { "16/25".to_string() } else { format!("{s2:.3}") },
                fr(c2 * c2 + s2 * s2, "337/625")
            ),
        );
        if stage == 0 && self.log.last().is_some_and(|e| e.obs == Obs::A && e.k == 1) {
            ui.label(RichText::new("A gave a₂: the exercise assumes a₁. Press R to start again in ψ₁, or measure on.").color(look.weak_text()));
        }
        ui.add_space(look.px(6.0));
        ui.label(RichText::new("Your outcomes").strong());
        if self.log.is_empty() {
            ui.label(RichText::new("none yet: press M (A, B, A, … in turn). The start state χ is not part of the exercise.").color(look.weak_text()));
        }
        let skip = self.log.len().saturating_sub(7);
        for (i, e) in self.log.iter().enumerate().skip(skip) {
            let text = if i == 0 && e.p == 1.0 && e.obs == Obs::A && e.k == 0 && !e.negative {
                format!("{}. A → a₁ (given): state ψ₁", i + 1)
            } else {
                format!(
                    "{}. {} → {}  (P = {:.3}): state {}{}",
                    i + 1,
                    e.obs.name(),
                    e.obs.outcome(e.k),
                    e.p,
                    if e.negative { "−" } else { "" },
                    e.obs.vector(e.k)
                )
            };
            ui.label(RichText::new(text).color(Self::colour(&look, e.obs)).monospace());
        }
    }

    /// The tree of outcome paths of the repeated sequence.
    fn tree_ui(&self, ui: &mut egui::Ui, size: Vec2) {
        let look = Look::of(ui);
        let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
        let painter = ui.painter_at(rect);
        panel(&painter, rect);
        let Some(t) = &self.tree else {
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                "\"Repeat sequence\" (side panel): N fresh systems in ψ₁ are measured B, then A.\nThe four outcome paths build up a tree here.",
                look.font(13.0),
                look.weak_text(),
            );
            return;
        };
        let pred = path_probabilities(t.beta);
        let f = (t.beta - beta_345()).abs() < 1e-9;
        let n = t.runs().max(1) as f64;
        let inner = rect.shrink2(vec2(look.px(10.0), look.px(10.0)));
        let w = inner.width();
        let h = inner.height() - look.px(22.0);
        let at = |x: f32, y: f32| pos2(inner.left() + x * w, inner.top() + y * h);
        let root = at(0.07, 0.5);
        let b_nodes = [at(0.32, 0.25), at(0.32, 0.75)];
        let leaves = [at(0.56, 0.1), at(0.56, 0.4), at(0.56, 0.6), at(0.56, 0.9)];
        let b_counts = [t.paths[0] + t.paths[1], t.paths[2] + t.paths[3]];
        let pb = [pred[0] + pred[1], pred[2] + pred[3]];
        let fr_b = ["9/25", "16/25"];
        let fr_p = ["81/625", "144/625", "256/625", "144/625"];
        let edge = |a: Pos2, b: Pos2, frac: f64, col: Color32| {
            painter.line_segment([a, b], look.stroke(1.0 + 12.0 * frac as f32, col.gamma_multiply(0.55)));
        };
        let colb = look.series(1);
        let cola = look.series(0);
        for kb in 0..2 {
            edge(root, b_nodes[kb], b_counts[kb] as f64 / n, colb);
            for ka in 0..2 {
                edge(b_nodes[kb], leaves[2 * kb + ka], t.paths[2 * kb + ka] as f64 / n, cola);
            }
        }
        let node = |p: Pos2, text: &str, col: Color32| {
            label_box(&painter, p, Align2::CENTER_CENTER, text, col);
        };
        node(root, &format!("ψ₁  (N = {})", t.runs()), look.text());
        for kb in 0..2 {
            let m = b_counts[kb] as f64 / n;
            let born = if f {
                format!("{} = {:.3}", fr_b[kb], pb[kb])
            } else {
                format!("{:.3}", pb[kb])
            };
            node(
                b_nodes[kb],
                &format!("b{}: {} ({m:.3}) · Born {born}", ["₁", "₂"][kb], b_counts[kb]),
                colb,
            );
            for ka in 0..2 {
                let i = 2 * kb + ka;
                let m = t.paths[i] as f64 / n;
                let born = if f {
                    format!("{} = {:.3}", fr_p[i], pred[i])
                } else {
                    format!("{:.3}", pred[i])
                };
                let p = leaves[i];
                label_box(
                    &painter,
                    p,
                    Align2::LEFT_CENTER,
                    &format!("a{}: {} ({m:.3}) · Born {born}", ["₁", "₂"][ka], t.paths[i]),
                    cola,
                );
            }
        }
        let a1 = (t.paths[0] + t.paths[2]) as f64 / n;
        let p_a1 = pred[0] + pred[2];
        let sd = (p_a1 * (1.0 - p_a1) / n).sqrt();
        painter.text(
            pos2(inner.left(), inner.bottom()),
            Align2::LEFT_BOTTOM,
            format!(
                "a₁ at the end: {a1:.3} measured (expected scatter ± {sd:.3}),  Born rule {}",
                if f {
                    format!("337/625 = {p_a1:.4}")
                } else {
                    format!("cos⁴β + sin⁴β = {p_a1:.4}")
                }
            ),
            look.font(13.5),
            look.prediction(),
        );
    }
}

impl Scene for NonCommutingScene {
    fn time(&self) -> f64 {
        0.0
    }

    fn step(&mut self, _dt: f64, _ctx: &mut Ctx) {}

    fn reset(&mut self, _ctx: &mut Ctx) {
        self.restart();
    }

    fn measure_label(&self) -> Option<String> {
        Some(format!("Measure {}", self.next_obs().name()))
    }

    fn measure(&mut self, ctx: &mut Ctx) {
        let obs = self.next_obs();
        self.measure(obs, ctx);
    }

    fn action(&mut self, action: &str, ctx: &mut Ctx) -> bool {
        let mut words = action.split_whitespace();
        let cmd = words.next().unwrap_or("");
        let arg = words.next();
        match cmd {
            "measure" => {
                let o = self.next_obs();
                self.measure(o, ctx)
            }
            "measure_a" => self.measure(Obs::A, ctx),
            "measure_b" => self.measure(Obs::B, ctx),
            "psi1" => {
                self.start_psi1 = true;
                self.restart();
            }
            "chi" => {
                self.start_psi1 = false;
                self.restart();
            }
            // end the collapse animation (this scene's clock does not move, so a screenshot
            // would always catch it a quarter second in)
            "settle" => {
                self.anim = None;
                self.flash.clear();
            }
            "repeat" => {
                let n = arg.and_then(|w| w.parse().ok()).unwrap_or(self.repeat_n);
                self.repeat(n, ctx)
            }
            "beta" => {
                let Some(v) = arg.and_then(|w| w.parse::<f64>().ok()) else {
                    return false;
                };
                self.beta = v.to_radians();
                self.tree = None;
                self.restart();
            }
            "prep" => {
                let Some(v) = arg.and_then(|w| w.parse::<f64>().ok()) else {
                    return false;
                };
                self.prep = v.to_radians();
                self.start_psi1 = false;
                self.restart();
            }
            _ => return false,
        }
        true
    }

    fn central_ui(&mut self, ui: &mut egui::Ui, _ctx: &mut Ctx) {
        let look = Look::of(ui);
        let s = self.state;
        let amp = |o: Obs, k: usize| dot2(eigvec(o, k, self.beta), s);
        readout_row(
            ui,
            &[
                ("⟨φ₁|ψ⟩", num(amp(Obs::B, 0), 3), Some(look.series(1))),
                ("⟨φ₂|ψ⟩", num(amp(Obs::B, 1), 3), Some(look.series(1))),
                ("⟨ψ₁|ψ⟩", num(amp(Obs::A, 0), 3), Some(look.series(0))),
                ("⟨ψ₂|ψ⟩", num(amp(Obs::A, 1), 3), Some(look.series(0))),
                ("P(b₁)", format!("{:.3}", amp(Obs::B, 0).powi(2)), None),
                ("P(a₁)", format!("{:.3}", amp(Obs::A, 0).powi(2)), None),
                ("β", deg(self.beta), None),
            ],
        );
        let avail = ui.available_size();
        let bottom_h = look.px(230.0).min(avail.y * 0.36);
        let top_h = avail.y - bottom_h - look.px(10.0);
        let plane_w = (avail.x * 0.56).min(top_h * 1.25);
        ui.horizontal(|ui| {
            self.plane(ui, vec2(plane_w, top_h));
            ui.add_space(look.px(6.0));
            ui.vertical(|ui| {
                let w = ui.available_width();
                let sph = w.min(top_h * 0.6);
                let a_dir = Spinor::real(self.beta.cos(), self.beta.sin()).bloch();
                let shown = self.shown_state(ui);
                let resp = BlochSphere::new("ps09_bloch", sph)
                    .size(w, sph)
                    .default_camera(Camera { az: -1.25, el: 0.3 })
                    .state(Spinor::real(shown[0], shown[1]).normalized().bloch())
                    .state_label("ψ")
                    .axis(BlochAxis::measurement([0.0, 0.0, 1.0], "φ₁", "φ₂", look.series(1)))
                    .axis(BlochAxis::measurement(a_dir, "ψ₁", "ψ₂", look.series(0)))
                    .xyz_labels(false)
                    .pole_labels(false)
                    .title("Bloch sphere (secondary view): φ₁ at the north pole")
                    .caption(format!("angles double: ψ₁ is 2β = {} from φ₁", deg(2.0 * self.beta)))
                    .show(ui);
                let _ = resp;
                ui.add_space(look.px(4.0));
                egui::ScrollArea::vertical()
                    .id_salt("ps09_walk")
                    .max_height(top_h - sph - look.px(10.0))
                    .show(ui, |ui| self.walk_ui(ui));
            });
        });
        ui.add_space(look.px(6.0));
        ui.horizontal(|ui| {
            let w = ui.available_width();
            let tree_w = if self.tree.is_some() { w * 0.62 } else { w };
            self.tree_ui(ui, vec2(tree_w, bottom_h));
            if let Some(t) = &self.tree {
                let pred = path_probabilities(t.beta);
                let labels: Vec<String> = ["b₁a₁", "b₁a₂", "b₂a₁", "b₂a₂"].iter().map(|s| s.to_string()).collect();
                HistogramPlot::new("ps09_hist", &t.hist)
                    .predicted(&pred)
                    .labels(&labels)
                    .x_label("path")
                    .scale(HistScale::Fraction)
                    .width(ui.available_width())
                    .height(bottom_h)
                    .show(ui);
            }
        });
    }

    fn side_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx) {
        let look = ctx.look;
        ui.label(RichText::new("The two bases").strong());
        let mut b = self.beta.to_degrees();
        if ui
            .add(egui::Slider::new(&mut b, 0.0..=90.0).text("mixing angle β").suffix("°"))
            .changed()
        {
            self.beta = b.to_radians();
            self.tree = None;
            self.restart();
        }
        if ui.button("β of the exercise (cos β = 3/5)").clicked() {
            self.beta = beta_345();
            self.tree = None;
            self.restart();
        }
        let (s, c) = self.beta.sin_cos();
        ui.label(
            RichText::new(format!(
                "ψ₁ = cos β φ₁ + sin β φ₂ = {} φ₁ + {} φ₂\nψ₂ = sin β φ₁ − cos β φ₂ = {} φ₁ − {} φ₂",
                num(c, 3),
                num(s, 3),
                num(s, 3),
                num(c, 3)
            ))
            .color(look.weak_text()),
        );

        ui.add_space(look.px(8.0));
        ui.label(RichText::new("Start state").strong());
        let mut p = self.prep.to_degrees();
        if ui
            .add(egui::Slider::new(&mut p, 0.0..=180.0).text("χ: angle from φ₁").suffix("°"))
            .changed()
        {
            self.prep = p.to_radians();
            self.start_psi1 = false;
            self.restart();
        }
        ui.horizontal(|ui| {
            if ui.selectable_label(self.start_psi1, "ψ₁ (after a₁, the exercise)").clicked() {
                self.start_psi1 = true;
                self.restart();
            }
            if ui.selectable_label(!self.start_psi1, "χ (any state)").clicked() {
                self.start_psi1 = false;
                self.restart();
            }
        });

        ui.add_space(look.px(8.0));
        ui.label(RichText::new("Measure").strong());
        let next = self.next_obs();
        ui.horizontal(|ui| {
            for obs in [Obs::A, Obs::B] {
                let text = if obs == next {
                    format!("Measure {}  (M)", obs.name())
                } else {
                    format!("Measure {}", obs.name())
                };
                if ui.button(RichText::new(text).color(look.measure())).clicked() {
                    self.measure(obs, ctx);
                }
            }
        });
        if let Some(n) = repeat_button(ui, "ps09_rep", "Repeat sequence ψ₁ → B → A", &mut self.repeat_n) {
            self.repeat(n, ctx);
        }
        if self.tree.is_some() && ui.button("Clear the tree").clicked() {
            self.tree = None;
        }
        ui.label(
            RichText::new(
                "Single shot: the state collapses onto an axis of the measured basis. Repeat: N fresh systems prepared in ψ₁ (the displayed state is not touched); counts add up until cleared.",
            )
            .color(look.weak_text())
            .size(12.0 * look.font_scale()),
        );
        if let Some(l) = self.flash.label() {
            ui.label(RichText::new(format!("last outcome: {l}")).color(look.measure()));
        }
    }

    fn notes(&self) -> String {
        NOTES.into()
    }
}

const NOTES: &str = "\
# The exercise
Â has the orthonormal eigenstates ψ₁, ψ₂ (eigenvalues a₁, a₂), B̂ has φ₁, φ₂ (b₁, b₂), related by ψ₁ = (3φ₁ + 4φ₂)/5 and ψ₂ = (4φ₁ − 3φ₂)/5.
- (a) A is measured and gives a₁: right after the measurement the state is ψ₁.
- (b) B is measured next: b₁ with |⟨φ₁|ψ₁⟩|² = 9/25, b₂ with |⟨φ₂|ψ₁⟩|² = 16/25.
- (c) A again: a₁ with (9/25)·(9/25) + (16/25)·(16/25) = 337/625 ≈ 0.539.
Why not 1: Â and B̂ do not commute (they have no common eigenbasis). The B measurement collapses ψ₁ onto φ₁ or φ₂, and neither is an eigenstate of Â: the earlier result a₁ is erased, and the second A measurement starts from φ₁ or φ₂.

# The picture
With real amplitudes a two-state system is a plane: the state is a unit vector, each observable's eigenbasis is a pair of perpendicular axes (blue: A, orange: B; here ψ₁ is turned by β = 53.13° from φ₁, cos β = 3/5). A measurement projects the state onto the axes of that basis; the squared projections are the probabilities (Born rule) and the state jumps onto the axis of the outcome (collapse).
- A minus sign (−φ₁) is the same physical state as φ₁: an overall factor is not observable.
- Bloch sphere: φ₁ and φ₂ are the poles. Angles double there, so orthogonal states sit opposite each other and P = cos²(angle/2).

# Try
- The scene starts in ψ₁, right after A gave a₁ (part a). Press M: B is measured, b₁ with 9/25 and b₂ with 16/25. Press M again: A, and a₁ comes back with 9/25 after b₁ or 16/25 after b₂. R starts again in ψ₁.
- Start state χ (side panel): any other state, to see the first A measurement collapse it onto ψ₁ or ψ₂.
- Repeat the sequence N = 10 000 times: the tree shows the four paths with 81, 144, 256 and 144 out of 625, and a₁ at the end → 337/625.
- Change β: P(a₁ again) = cos⁴β + sin⁴β = 1 − ½ sin²2β. It is 1 only for β = 0° or 90° (same eigenbasis: commuting observables) and smallest, 1/2, at 45°.
- Measure A twice in a row: the second result always repeats the first.

# Behind the scenes
Outcomes are drawn with the Born rule from the seeded random generator. The state after a measurement is P̂ψ/‖P̂ψ‖ (projection postulate), which is why it can come out as −φ₁. No time evolution in this scene (no Hamiltonian is given).";

#[cfg(test)]
mod tests {
    use super::*;

    /// PS09 35: 9/25, 16/25 and 337/625.
    #[test]
    fn exercise_numbers() {
        let beta = beta_345();
        assert!((beta.cos() - 0.6).abs() < 1e-15 && (beta.sin() - 0.8).abs() < 1e-15);
        // the bases as in the exercise
        let psi1 = eigvec(Obs::A, 0, beta);
        let psi2 = eigvec(Obs::A, 1, beta);
        assert!((psi1[0] - 0.6).abs() < 1e-15 && (psi1[1] - 0.8).abs() < 1e-15);
        assert!((psi2[0] - 0.8).abs() < 1e-15 && (psi2[1] + 0.6).abs() < 1e-15);
        assert!(dot2(psi1, psi2).abs() < 1e-15);
        let p = path_probabilities(beta);
        assert!((p[0] + p[1] - 9.0 / 25.0).abs() < 1e-14);
        assert!((p[2] + p[3] - 16.0 / 25.0).abs() < 1e-14);
        assert!((p[0] - 81.0 / 625.0).abs() < 1e-14);
        assert!((p[2] - 256.0 / 625.0).abs() < 1e-14);
        assert!((p[0] + p[2] - 337.0 / 625.0).abs() < 1e-14);
        // the same with complex spinors (φ₁ = |↑z⟩, φ₂ = |↓z⟩)
        let s1 = Spinor::real(0.6, 0.8);
        let pb1 = s1.prob(&Spinor::UP_Z);
        let pa1_after = pb1 * Spinor::UP_Z.prob(&s1) + (1.0 - pb1) * Spinor::DOWN_Z.prob(&s1);
        assert!((pa1_after - 337.0 / 625.0).abs() < 1e-14);
    }

    #[test]
    fn general_angle_and_sampling() {
        for i in 0..=18 {
            let beta = i as f64 * 5f64.to_radians();
            let p = path_probabilities(beta);
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-14);
            let want = beta.cos().powi(4) + beta.sin().powi(4);
            assert!((p[0] + p[2] - want).abs() < 1e-14);
        }
        // Monte Carlo of the sequence ψ₁ → B → A
        let mut rng = QbRng::seed(3);
        let beta = beta_345();
        let n = 50_000;
        let mut a1 = 0;
        for _ in 0..n {
            let (_, s, _) = sample(Obs::B, eigvec(Obs::A, 0, beta), beta, &mut rng);
            let (ka, _, _) = sample(Obs::A, s, beta, &mut rng);
            a1 += (ka == 0) as usize;
        }
        let p = 337.0 / 625.0;
        let f = a1 as f64 / n as f64;
        assert!((f - p).abs() < 5.0 * (p * (1.0 - p) / n as f64).sqrt(), "{f}");
        // a repeated measurement repeats its result
        let (k, s, _) = sample(Obs::A, [0.3, 0.9539392014169456], beta, &mut rng);
        for _ in 0..100 {
            assert_eq!(sample(Obs::A, s, beta, &mut rng).0, k);
        }
    }
}
