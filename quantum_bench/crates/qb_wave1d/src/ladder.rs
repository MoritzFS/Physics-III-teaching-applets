//! Lecture script 6.8 · The harmonic oscillator through its ladder operators.
//!
//! Three linked panels: the wavefunction ψ(x, t) in the potential (x / p),
//! a bar chart of the Fock populations |cₙ|² coloured by arg cₙ, and an
//! operator panel whose buttons apply â, â†, N̂ = â†â, x̂, p̂ to the current
//! state (phases included), show the norm of the result and renormalise.
//! Units ħ = m = ω = 1, so the script's χ = √(mω/ħ) x is x here and
//! â = (∂/∂χ + χ)/√2, â† = (−∂/∂χ + χ)/√2, H = ħω(â†â + ½).

use std::f64::consts::TAU;
use std::sync::Arc;

use qb_core::egui::{self, Align2, RichText, pos2, vec2};
use qb_core::num::{C64, EigenBasis, EigenExpansion, Grid1D, Moments, Wave1D, c64, measure_energy, measure_position};
use qb_core::widgets::{
    Domain, FlashKind, MeasureFlash, PanZoom, PlotFrame, ReprView, TimeSeries, TimeSeriesPlot, TimeWindow, WaveCurve, WavePlot, WaveStyle,
    fmt_num, label_box, phase_color, readout_row, repeat_button, view_bar,
};
use qb_core::{Ctx, Look, Scene, wall_time};

use crate::common::{self, EnergyView, LevelCurve, Repeated, grow_top, hint, section};
use crate::physics::poisson;

/// Fock states n = 0 … LEVELS − 1 (the truncated basis).
pub const LEVELS: usize = 40;

// ------------------------------------------------------------ the operators on coefficient vectors

/// â on coefficients: (âc)ₙ = √(n+1) cₙ₊₁, since â|n⟩ = √n |n−1⟩ (script 6.225).
pub fn lower(c: &[C64]) -> Vec<C64> {
    let n = c.len();
    (0..n)
        .map(|k| {
            if k + 1 < n {
                c[k + 1] * ((k + 1) as f64).sqrt()
            } else {
                c64(0.0, 0.0)
            }
        })
        .collect()
}

/// â† on coefficients: (â†c)ₙ = √n cₙ₋₁, since â†|n⟩ = √(n+1) |n+1⟩ (script 6.224).
/// Returns the result and the squared norm that would land on n = len, beyond
/// the truncated basis (dropped).
pub fn raise(c: &[C64]) -> (Vec<C64>, f64) {
    let n = c.len();
    let out = (0..n)
        .map(|k| if k >= 1 { c[k - 1] * (k as f64).sqrt() } else { c64(0.0, 0.0) })
        .collect();
    let lost = c.last().map_or(0.0, |z| z.norm_sqr() * n as f64);
    (out, lost)
}

/// N̂ = â†â: cₙ → n cₙ.
pub fn number(c: &[C64]) -> Vec<C64> {
    c.iter().enumerate().map(|(k, z)| z * k as f64).collect()
}

/// x̂ = √(ħ/2mω)(â† + â) = (â† + â)/√2 (script 6.230), and the dropped norm².
pub fn position(c: &[C64]) -> (Vec<C64>, f64) {
    let (up, lost) = raise(c);
    let down = lower(c);
    let s = std::f64::consts::FRAC_1_SQRT_2;
    (up.iter().zip(&down).map(|(a, b)| (a + b) * s).collect(), 0.5 * lost)
}

/// p̂ = i√(mħω/2)(â† − â) = i(â† − â)/√2 (script 6.231), and the dropped norm².
pub fn momentum(c: &[C64]) -> (Vec<C64>, f64) {
    let (up, lost) = raise(c);
    let down = lower(c);
    let s = c64(0.0, std::f64::consts::FRAC_1_SQRT_2);
    (up.iter().zip(&down).map(|(a, b)| (a - b) * s).collect(), 0.5 * lost)
}

/// Σ|cₙ|².
pub fn norm2(c: &[C64]) -> f64 {
    c.iter().map(|z| z.norm_sqr()).sum()
}

/// ‖[â, â†]c − c‖ = ‖(ââ† − â†â)c − c‖ with the same functions as the buttons.
/// It vanishes unless the top level of the truncated basis is populated.
pub fn commutator_residual(c: &[C64]) -> f64 {
    let (up, _) = raise(c);
    let a_ad = lower(&up);
    let ad_a = raise(&lower(c)).0;
    a_ad.iter()
        .zip(&ad_a)
        .zip(c)
        .map(|((x, y), z)| (x - y - z).norm_sqr())
        .sum::<f64>()
        .sqrt()
}

/// ⟨n⟩ and Δn of normalised or unnormalised coefficients.
pub fn n_stats(c: &[C64]) -> (f64, f64) {
    let s = norm2(c).max(1e-300);
    let m: f64 = c.iter().enumerate().map(|(k, z)| k as f64 * z.norm_sqr()).sum::<f64>() / s;
    let m2: f64 = c.iter().enumerate().map(|(k, z)| (k * k) as f64 * z.norm_sqr()).sum::<f64>() / s;
    (m, (m2 - m * m).max(0.0).sqrt())
}

/// α̃ = ⟨ψ|â|ψ⟩ and the residual ‖âψ − α̃ψ‖ (0 for an eigenstate of â, i.e. a coherent state).
pub fn lowering_eigen_check(c: &[C64]) -> (C64, f64) {
    let s = norm2(c).max(1e-300);
    let a = lower(c);
    let alpha = c.iter().zip(&a).map(|(x, y)| x.conj() * y).sum::<C64>() / s;
    let res = a.iter().zip(c).map(|(y, x)| (y - alpha * x).norm_sqr()).sum::<f64>().sqrt() / s.sqrt();
    (alpha, res)
}

/// The single occupied level, if the state is a Fock state.
fn fock_index(c: &[C64]) -> Option<usize> {
    let s = norm2(c);
    c.iter().position(|z| z.norm_sqr() > (1.0 - 1e-9) * s)
}

// ------------------------------------------------------------ the scene

/// The presets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LadderPreset {
    /// |0⟩: climb the ladder with â†
    Ground,
    /// |3⟩
    Three,
    /// coherent state α = 1.5
    Coherent,
    /// (|0⟩ + |1⟩)/√2
    Superposition,
}

/// One operator button.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Lower,
    Raise,
    Number,
    X,
    P,
}

impl Op {
    fn symbol(self) -> &'static str {
        match self {
            Op::Lower => "â",
            Op::Raise => "â†",
            // (egui does not draw combining accents: N̂ is written â†â, the hats of x̂, p̂ are painted on the buttons)
            Op::Number => "â†â",
            Op::X => "x",
            Op::P => "p",
        }
    }
}

pub struct LadderScene {
    preset: LadderPreset,
    // parameters
    alpha: C64,
    width_x: f64,
    repeat_n: usize,
    view: ReprView,
    style: WaveStyle,
    classical: bool,
    poisson_overlay: bool,

    // state
    xs: Vec<f64>,
    v: Vec<f64>,
    basis: Arc<EigenBasis>,
    exp: EigenExpansion,
    wave: Wave1D,
    moments: Moments,
    t: f64,
    /// how the current chain started, e.g. "|0⟩"
    start_label: String,
    /// operators applied since then, oldest first
    history: Vec<Op>,
    /// c(t = 0 phases) before each operator, for Undo
    undo: Vec<Vec<C64>>,
    /// text describing the last operator's result
    result: String,
    /// animation of the bars after an operator: (wall time, populations before)
    bar_anim: Option<(f64, Vec<f64>)>,
    /// classical start (t₀, x₀, p₀)
    classical0: (f64, f64, f64),
    n_series: TimeSeries,
    x_series: TimeSeries,
    events: Vec<(f64, String)>,
    repeated: Option<Repeated>,
    flash: MeasureFlash,
    top_x: f64,
    top_p: f64,
    e_top: f64,
}

impl LadderScene {
    pub fn new(preset: LadderPreset) -> Self {
        // a long cell for a fine momentum grid (dp = 2π/128 ≈ 0.05): smooth |Φ(p)|²
        let grid = Grid1D::new(-64.0, 64.0, 2048);
        let basis = Arc::new(EigenBasis::harmonic(grid, LEVELS, 1.0, 0.0));
        let xs = grid.xs();
        let exp = EigenExpansion::from_coeffs(basis.clone(), &[c64(1.0, 0.0)]);
        let wave = exp.wave();
        let mut s = LadderScene {
            preset,
            alpha: c64(1.5, 0.0),
            width_x: 0.25,
            repeat_n: 1000,
            view: if preset == LadderPreset::Superposition {
                ReprView::Both
            } else {
                ReprView::X
            },
            style: WaveStyle::Density,
            classical: preset == LadderPreset::Coherent,
            poisson_overlay: preset == LadderPreset::Coherent,
            v: xs.iter().map(|x| 0.5 * x * x).collect(),
            xs,
            basis,
            exp,
            wave,
            moments: Moments::default(),
            t: 0.0,
            start_label: String::new(),
            history: vec![],
            undo: vec![],
            result: String::new(),
            bar_anim: None,
            classical0: (0.0, 0.0, 0.0),
            n_series: TimeSeries::new("⟨n⟩"),
            x_series: TimeSeries::new("⟨x⟩"),
            events: vec![],
            repeated: None,
            flash: MeasureFlash::new(),
            top_x: 0.0,
            top_p: 0.0,
            e_top: 6.5,
        };
        s.restart();
        s
    }

    fn initial(&self) -> (Vec<C64>, String) {
        let mut c = vec![c64(0.0, 0.0); LEVELS];
        let r = std::f64::consts::FRAC_1_SQRT_2;
        match self.preset {
            LadderPreset::Ground => {
                c[0] = c64(1.0, 0.0);
                (c, "|0⟩".into())
            }
            LadderPreset::Three => {
                c[3] = c64(1.0, 0.0);
                (c, "|3⟩".into())
            }
            LadderPreset::Coherent => (
                EigenExpansion::coherent(self.basis.clone(), self.alpha).c0,
                format!("|α = {}⟩", fmt_alpha(self.alpha)),
            ),
            LadderPreset::Superposition => {
                c[0] = c64(r, 0.0);
                c[1] = c64(r, 0.0);
                (c, "(|0⟩ + |1⟩)/√2".into())
            }
        }
    }

    fn restart(&mut self) {
        let (c, label) = self.initial();
        self.start_with(c, label);
        self.t = 0.0;
        self.exp.set_time(0.0);
        self.n_series.clear();
        self.x_series.clear();
        self.events.clear();
        self.refresh();
        self.new_preparation();
        self.top_x = 1.2 * common::peak_density(&self.wave.psi).max(0.3);
        self.top_p = 1.2 * common::peak_density(&self.wave.momentum().phi).max(0.3);
    }

    /// A new chain: coefficients at the current time, a start label, no history.
    fn start_with(&mut self, c_now: Vec<C64>, label: String) {
        self.set_now(&c_now);
        self.start_label = label;
        self.history.clear();
        self.undo.clear();
        self.result.clear();
        self.repeated = None;
        self.bar_anim = None;
    }

    /// Sets c(t) at the current time (stored as c₀ = c(t) e^(+iEₙt)), normalised.
    fn set_now(&mut self, c_now: &[C64]) {
        let t = self.t;
        let s = norm2(c_now).max(1e-300).sqrt();
        let c0: Vec<C64> = c_now
            .iter()
            .zip(&self.basis.energies)
            .map(|(c, e)| c / s * C64::from_polar(1.0, e * t))
            .collect();
        self.exp = EigenExpansion::from_coeffs(self.basis.clone(), &c0);
        self.exp.set_time(t);
        self.refresh();
    }

    fn refresh(&mut self) {
        self.wave = self.exp.wave();
        self.moments = self.wave.moments();
    }

    fn new_preparation(&mut self) {
        let p = self.exp.probabilities();
        let e = self.exp.mean_energy();
        let var: f64 = p.iter().zip(&self.basis.energies).map(|(q, en)| q * (en - e).powi(2)).sum();
        self.e_top = 6.5f64.max((e + 2.5 * var.sqrt() + 1.5).ceil());
        self.classical0 = (self.t, self.moments.x, self.moments.p);
        self.record();
    }

    fn record(&mut self) {
        let (n, _) = n_stats(&self.exp.c0);
        self.n_series.push(self.t, n);
        self.x_series.push(self.t, self.moments.x);
    }

    /// Advances by exactly dt (exact phases e^(−iEₙt)).
    pub fn advance(&mut self, dt: f64) {
        self.exp.advance(dt);
        self.t += dt;
        self.refresh();
        if self.n_series.last().is_none_or(|p| self.t - p[0] >= 0.05) {
            self.record();
        }
    }

    /// Applies an operator to the current state, reports the norm, renormalises.
    pub fn apply(&mut self, op: Op, ctx: Option<&mut Ctx>) {
        let c = self.exp.coeffs();
        let (n_mean, _) = n_stats(&c);
        let fock = fock_index(&c);
        let (r, lost) = match op {
            Op::Lower => (lower(&c), 0.0),
            Op::Raise => raise(&c),
            Op::Number => (number(&c), 0.0),
            Op::X => position(&c),
            Op::P => momentum(&c),
        };
        let norm = norm2(&r).sqrt();
        let sym = op.symbol();
        if norm < 1e-10 {
            self.result = match (op, fock) {
                (Op::Lower, Some(0)) => {
                    "â|0⟩ = 0: the ground state is annihilated (âψ₀ = 0, script 6.206). Nothing to normalise: the state is kept.".into()
                }
                (Op::Number, Some(0)) => "â†â|0⟩ = 0·|0⟩ = 0 (eigenvalue 0). Nothing to normalise: the state is kept.".into(),
                _ => format!("{sym}ψ = 0: nothing to normalise, the state is kept."),
            };
            self.bar_anim = Some((wall_time(), self.exp.probabilities()));
            return;
        }
        let before = self.exp.probabilities();
        self.undo.push(self.exp.c0.clone());
        self.history.push(op);
        // the eigenstate check of â before normalising (coherent states)
        let (alpha, res) = lowering_eigen_check(&c);
        let text = match (op, fock) {
            (Op::Lower, Some(n)) => format!("â|{n}⟩ = √{n} |{}⟩   (‖âψ‖ = {norm:.4})", n - 1),
            (Op::Raise, Some(n)) if n + 1 < LEVELS => format!("â†|{n}⟩ = √{} |{}⟩   (‖â†ψ‖ = {norm:.4})", n + 1, n + 1),
            (Op::Number, Some(n)) => format!("â†â|{n}⟩ = {n}|{n}⟩: the same state (eigenvalue {n})"),
            (Op::X, Some(0)) => format!("x|0⟩ = |1⟩/√2   (‖xψ‖ = {norm:.4})"),
            (Op::X, Some(n)) => format!("x|{n}⟩ = (√{n} |{}⟩ + √{} |{}⟩)/√2   (‖xψ‖ = {norm:.4})", n - 1, n + 1, n + 1),
            (Op::P, Some(0)) => format!("p|0⟩ = i|1⟩/√2   (‖pψ‖ = {norm:.4})"),
            (Op::P, Some(n)) => format!("p|{n}⟩ = i(√{} |{}⟩ − √{n} |{}⟩)/√2   (‖pψ‖ = {norm:.4})", n + 1, n + 1, n - 1),
            (Op::Lower, None) if res < 1e-6 => format!(
                "âψ = αψ with α = {}: an eigenstate of â (coherent state). After normalising only the global phase arg α = {:.2} rad is new; |ψ|² is unchanged.",
                fmt_alpha(alpha),
                alpha.arg()
            ),
            (Op::Lower, _) => format!("‖âψ‖ = {norm:.4} = √⟨n⟩ with ⟨n⟩ = {n_mean:.4} before"),
            (Op::Raise, _) if lost > 1e-12 => format!(
                "‖â†ψ‖ = {norm:.4} = √(⟨n⟩ + 1 − {lost:.1e}) with ⟨n⟩ = {n_mean:.4} before (the part beyond n = {} is dropped)",
                LEVELS - 1
            ),
            (Op::Raise, _) => format!("‖â†ψ‖ = {norm:.4} = √(⟨n⟩ + 1) with ⟨n⟩ = {n_mean:.4} before"),
            (Op::Number, _) => format!("‖â†âψ‖ = {norm:.4} = √⟨n²⟩: each cₙ multiplied by n (the |0⟩ part is removed)"),
            (Op::X, _) => format!("‖xψ‖ = {norm:.4} = √⟨x²⟩"),
            (Op::P, _) => format!("‖pψ‖ = {norm:.4} = √⟨p²⟩"),
        };
        self.result = text;
        self.set_now(&r);
        self.bar_anim = Some((wall_time(), before));
        self.repeated = None;
        self.events.push((self.t, sym.to_string()));
        self.new_preparation();
        grow_top(&mut self.top_x, common::peak_density(&self.wave.psi));
        if let Some(ctx) = ctx {
            if lost > 1e-12 {
                ctx.set_status(format!(
                    "{sym}: the basis ends at n = {}; a part of norm² {lost:.2e} would go to n = {LEVELS} and was dropped (truncation).",
                    LEVELS - 1
                ));
            } else {
                ctx.set_status(format!("{sym} applied at t = {:.2}: {}", self.t, self.result));
            }
        }
    }

    fn undo(&mut self) {
        if let Some(c0) = self.undo.pop() {
            let before = self.exp.probabilities();
            self.exp = EigenExpansion::from_coeffs(self.basis.clone(), &c0);
            self.exp.set_time(self.t);
            self.exp.normalize();
            self.history.pop();
            self.refresh();
            self.result = "undone".into();
            self.bar_anim = Some((wall_time(), before));
            self.new_preparation();
        }
    }

    fn reset_ground(&mut self) {
        let mut c = vec![c64(0.0, 0.0); LEVELS];
        c[0] = c64(1.0, 0.0);
        let before = self.exp.probabilities();
        self.start_with(c, "|0⟩".into());
        self.bar_anim = Some((wall_time(), before));
        self.new_preparation();
    }

    fn set_fock(&mut self, n: usize, toggle: bool) {
        let before = self.exp.probabilities();
        if toggle {
            let mut c = self.exp.coeffs();
            if c[n].norm_sqr() > 1e-6 {
                c[n] = c64(0.0, 0.0);
            } else {
                c[n] = c64(
                    norm2(&c).max(1e-6).sqrt() / (c.iter().filter(|z| z.norm_sqr() > 1e-6).count().max(1) as f64).sqrt(),
                    0.0,
                );
            }
            if norm2(&c) < 1e-12 {
                c[n] = c64(1.0, 0.0);
            }
            self.start_with(c, "ψ (mixed by hand)".into());
        } else {
            let mut c = vec![c64(0.0, 0.0); LEVELS];
            c[n] = c64(1.0, 0.0);
            self.start_with(c, format!("|{n}⟩"));
        }
        self.bar_anim = Some((wall_time(), before));
        self.new_preparation();
    }

    fn measure_e(&mut self, ctx: &mut Ctx) {
        let Some(n) = measure_energy(&mut self.exp, ctx.rng) else { return };
        self.refresh();
        let e = self.basis.energies[n];
        self.flash
            .trigger(FlashKind::Level { index: n, energy: e }, format!("E = (n + ½)ħω, n = {n}"));
        self.events.push((self.t, format!("E→{n}")));
        // a measurement cannot be undone: a new chain starts here
        self.start_label = format!("|{n}⟩ (measured)");
        self.history.clear();
        self.undo.clear();
        self.result = format!("measured E = {e:.1}ħω: the state is now the Fock state |{n}⟩ (a measurement cannot be undone)");
        self.repeated = None;
        self.new_preparation();
        ctx.set_status(format!("measured E = {e:.1} (n = {n}) at t = {:.2}", self.t));
    }

    fn measure_x(&mut self, ctx: &mut Ctx) {
        let mut w = self.exp.wave();
        let x = measure_position(&mut w, self.width_x, ctx.rng);
        let c = common::project(&self.basis.states, &w);
        let captured = norm2(&c);
        self.set_now(&c);
        self.flash
            .trigger(FlashKind::Position { x, width: self.width_x }, format!("x = {x:.2}"));
        self.events.push((self.t, "x".into()));
        self.start_label = format!("ψ after x = {x:.2}");
        self.history.clear();
        self.undo.clear();
        self.result = String::new();
        self.repeated = None;
        self.new_preparation();
        grow_top(&mut self.top_x, common::peak_density(&self.wave.psi));
        ctx.set_status(format!(
            "measured x = {x:.3} at t = {:.2} (detector σ = {}); the {LEVELS} Fock states hold {captured:.4} of the collapsed state",
            self.t, self.width_x
        ));
    }

    fn repeat_e(&mut self, n: usize, ctx: &mut Ctx) {
        let (probs, names) = common::level_outcomes(&self.exp.probabilities());
        let r = common::repeat_discrete(&probs, names, n, ctx.rng, "n (repeat N)");
        ctx.set_status(format!(
            "{n} fresh copies measured in E: mean n = {:.3}, Δn = {:.3}",
            r.hist.mean(),
            r.hist.std()
        ));
        self.repeated = Some(r);
    }

    fn repeat_x(&mut self, n: usize, ctx: &mut Ctx) {
        let r = common::repeat_position(&self.wave, n, ctx.rng);
        ctx.set_status(format!(
            "{n} fresh copies measured in x: mean {:.3}, spread {:.3}",
            r.hist.mean(),
            r.hist.std()
        ));
        self.repeated = Some(r);
    }

    fn classical(&self) -> (f64, f64) {
        let (t0, x0, p0) = self.classical0;
        let s = self.t - t0;
        (x0 * s.cos() + p0 * s.sin(), -x0 * s.sin() + p0 * s.cos())
    }

    fn history_text(&self) -> String {
        let ops: Vec<&str> = self.history.iter().rev().map(|o| o.symbol()).collect();
        if ops.is_empty() {
            self.start_label.clone()
        } else {
            format!("{} {}", ops.join(" "), self.start_label)
        }
    }

    // ------------------------------------------------------------ panels

    fn plot_x(&mut self, ui: &mut egui::Ui, height: f32) {
        let look = Look::of(ui);
        grow_top(&mut self.top_x, common::peak_density(&self.wave.psi));
        let e_top = self.e_top;
        let x_half = (2.0 * e_top).sqrt().max(4.5) + 0.5;
        let e_mean = self.exp.mean_energy();
        let scale = common::energy_curve_scale(self.style, self.top_x, 0.28 * e_top);
        let probs = self.exp.probabilities();
        let levels: Vec<f64> = self.basis.energies.iter().cloned().filter(|e| *e < e_top).collect();
        let mut curves = vec![];
        if let Some(n) = fock_index(&self.exp.c0) {
            let amp = self.basis.states[n].iter().fold(0.0f64, |m, v| m.max(v.abs()));
            curves.push(LevelCurve {
                psi: &self.basis.states[n],
                energy: self.basis.energies[n],
                scale: 0.42 / amp.max(1e-9),
                label: format!("ψ{}", common::subscript(n)),
                color: look.weak_text(),
                dashed: false,
            });
        }
        let _ = probs;
        let (resp, _) = EnergyView {
            id: "ladder",
            xs: &self.xs,
            v: &self.v,
            psi: &self.wave.psi,
            style: self.style,
            baseline: e_mean,
            scale,
            x_range: (-x_half, x_half),
            e_range: (0.0, e_top),
            height,
            energies: &levels,
            coeffs: None,
            labels: None,
            selected: None,
            ladder_width: 0.0,
            curves,
            markers: vec![],
            drawable: false,
        }
        .show(ui);
        let tr = resp.transform;
        let painter = resp.painter(ui);
        if self.classical {
            let (xc, _) = self.classical();
            let p = tr.to_screen(xc, 0.5 * xc * xc);
            painter.circle_filled(p, look.px(6.0), look.accent());
            painter.text(
                p + vec2(look.px(8.0), -look.px(4.0)),
                Align2::LEFT_BOTTOM,
                "classical",
                look.font(11.0),
                look.accent(),
            );
        }
        painter.text(
            pos2(tr.rect.right() - look.px(4.0), tr.y_to_screen(e_mean) - 2.0),
            Align2::RIGHT_BOTTOM,
            format!("⟨E⟩ = (⟨n⟩ + ½)ħω = {e_mean:.3}"),
            look.font(11.5),
            look.text(),
        );
        self.flash.paint(ui, &tr, Domain::Position);
    }

    fn plot_p(&mut self, ui: &mut egui::Ui, height: f32) {
        let mom = self.wave.momentum();
        grow_top(&mut self.top_p, common::peak_density(&mom.phi));
        let (lo, hi) = self.style.y_range(self.top_p);
        let w = (2.0 * self.e_top).sqrt().max(4.5) + 0.5;
        let resp = WavePlot::new("ladder_p")
            .domain(Domain::Momentum)
            .style(self.style)
            .x_range(-w, w)
            .y_range(lo, hi)
            .height(height)
            .curve(WaveCurve::new(&mom.p, &mom.phi))
            .show(ui);
        if !matches!(self.flash.kind(), Some(FlashKind::Level { .. })) {
            self.flash.paint(ui, &resp.transform, Domain::Momentum);
        }
    }

    /// The Fock populations |cₙ|² as bars coloured by arg cₙ, with phasors, ⟨n⟩ and Δn.
    fn fock_chart(&mut self, ui: &mut egui::Ui, width: f32, height: f32) {
        let look = Look::of(ui);
        let c = self.exp.coeffs();
        let probs: Vec<f64> = c.iter().map(|z| z.norm_sqr()).collect();
        let highest = probs.iter().rposition(|p| *p > 1e-4).unwrap_or(0);
        let n_max = (highest + 2).clamp(15, LEVELS - 1);
        let (mean, dn) = n_stats(&c);
        // bars grow from the previous populations for a moment after an operator
        let anim = self.bar_anim.as_ref().and_then(|(t0, before)| {
            let u = ((wall_time() - t0) / 0.45) as f32;
            (u < 1.0).then(|| (u.clamp(0.0, 1.0), before.clone()))
        });
        if anim.is_some() {
            ui.ctx().request_repaint();
        }
        let shown: Vec<f64> = match &anim {
            Some((u, before)) => {
                let e = 1.0 - (1.0 - u).powi(3);
                probs.iter().zip(before).map(|(p, b)| b + (p - b) * e as f64).collect()
            }
            None => probs.clone(),
        };
        // the scale follows the tallest bar (and the bars before an operator, while they move)
        let tallest = probs
            .iter()
            .chain(anim.as_ref().map_or(&[][..], |(_, b)| &b[..]))
            .cloned()
            .fold(0.0, f64::max);
        let y_top = (1.32 * tallest).clamp(0.3, 1.32);
        let resp = PlotFrame::new(("ladder_fock", n_max, (y_top * 100.0) as i64))
            .x_range(-0.6, n_max as f64 + 0.6)
            .y_range(0.0, y_top)
            .labels("n", "|cₙ|² of Fock state |n⟩")
            .width(width)
            .height(height)
            .pan_zoom(PanZoom::None)
            .left_drag_pans(false)
            .show(ui, |painter, tr| {
                let r = tr.rect;
                let bar_px = tr.rect.width() / (n_max as f32 + 1.2);
                // Poisson with the same ⟨n⟩ (what a coherent state would have)
                if self.poisson_overlay {
                    for n in 0..=n_max {
                        let p = poisson(n, mean);
                        let (a, b) = (tr.x_to_screen(n as f64 - 0.42), tr.x_to_screen(n as f64 + 0.42));
                        let y = tr.y_to_screen(p);
                        painter.extend(egui::Shape::dashed_line(
                            &[pos2(a, y), pos2(b, y)],
                            look.stroke(1.6, look.prediction()),
                            4.0,
                            3.0,
                        ));
                    }
                }
                for n in 0..=n_max {
                    let p = shown[n];
                    let col = phase_color(c[n].arg() as f32, 0.95);
                    let rect = egui::Rect::from_min_max(tr.to_screen(n as f64 - 0.36, p), tr.to_screen(n as f64 + 0.36, 0.0));
                    painter.rect_filled(rect, 1.0, if probs[n] > 1e-6 { col } else { look.grid() });
                    // phasor of cₙ in a row along the top
                    if probs[n] > 1e-4 {
                        let rad = (bar_px * 0.36).clamp(3.0, look.px(11.0));
                        let center = pos2(tr.x_to_screen(n as f64), tr.y_to_screen(0.93 * y_top));
                        painter.circle_stroke(center, rad, look.stroke(1.0, look.plot_frame()));
                        let a = c[n].arg() as f32;
                        let len = rad * (probs[n] / probs.iter().cloned().fold(0.0, f64::max)).sqrt() as f32;
                        let tip = center + vec2(a.cos(), -a.sin()) * len.max(rad * 0.35);
                        painter.line_segment([center, tip], look.stroke(2.0, phase_color(a, 1.0)));
                    }
                    if p > 0.04 && bar_px > look.px(22.0) {
                        painter.text(
                            tr.to_screen(n as f64, p) - vec2(0.0, 2.0),
                            Align2::CENTER_BOTTOM,
                            format!("{p:.2}"),
                            look.font(11.0),
                            look.text(),
                        );
                    }
                }
                // ⟨n⟩ and ±Δn
                let sx = tr.x_to_screen(mean);
                painter.line_segment(
                    [pos2(sx, r.top() + look.px(28.0)), pos2(sx, r.bottom())],
                    look.stroke(1.6, look.accent()),
                );
                let y = tr.y_to_screen(0.83 * y_top);
                let (a, b) = (tr.x_to_screen(mean - dn), tr.x_to_screen(mean + dn));
                if dn > 1e-6 {
                    painter.line_segment([pos2(a, y), pos2(b, y)], look.stroke(1.4, look.accent()));
                    for xx in [a, b] {
                        painter.line_segment([pos2(xx, y - 4.0), pos2(xx, y + 4.0)], look.stroke(1.4, look.accent()));
                    }
                }
                label_box(
                    painter,
                    pos2(sx + look.px(6.0), y + look.px(6.0)),
                    Align2::LEFT_TOP,
                    &format!("⟨n⟩ = {mean:.3},  Δn = {dn:.3}"),
                    look.accent(),
                );
                // the last operator's result, while the bars move
                if let Some((u, _)) = &anim
                    && !self.result.is_empty()
                {
                    let alpha = (1.0 - u).clamp(0.0, 1.0);
                    painter.rect_filled(r, 0.0, look.accent().gamma_multiply(0.06 * alpha));
                }
            });
        if let Some((x, _)) = resp.hover {
            let n = x.round().clamp(0.0, (LEVELS - 1) as f64) as usize;
            resp.response.clone().on_hover_text(format!(
                "|{n}⟩: |cₙ|² = {:.4}, arg cₙ = {:.2}π, Eₙ = {:.1}ħω\nclick: |{n}⟩ alone · shift-click: add/remove it",
                probs[n],
                c[n].arg() / std::f64::consts::PI,
                n as f64 + 0.5
            ));
        }
        if resp.response.clicked()
            && let Some(p) = resp.response.interact_pointer_pos()
        {
            let (x, _) = resp.transform.from_screen(p);
            let n = x.round().clamp(0.0, (LEVELS - 1) as f64) as usize;
            let shift = ui.input(|i| i.modifiers.shift);
            self.set_fock(n, shift);
        }
    }

    /// The big operator buttons and the algebra readouts.
    fn operator_panel(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx, width: f32) {
        let look = Look::of(ui);
        ui.allocate_ui(vec2(width, 0.0), |ui| {
            ui.set_width(width);
            ui.horizontal_wrapped(|ui| {
                let big = |text: &str| RichText::new(text).size(24.0 * look.font_scale()).strong();
                let size = vec2(look.px(64.0), look.px(44.0));
                for (op, hover) in [
                    (Op::Lower, "annihilation operator â = (∂/∂χ + χ)/√2: â|n⟩ = √n |n−1⟩"),
                    (Op::Raise, "creation operator â† = (−∂/∂χ + χ)/√2: â†|n⟩ = √(n+1) |n+1⟩"),
                    (Op::Number, "number operator N = â†â: â†â|n⟩ = n|n⟩"),
                    (Op::X, "position operator x = √(ħ/2mω)(â† + â) (script 6.230)"),
                    (Op::P, "momentum operator p = i√(mħω/2)(â† − â) (script 6.231)"),
                ] {
                    let resp = ui.add(egui::Button::new(big(op.symbol()).color(look.accent())).min_size(size));
                    if matches!(op, Op::X | Op::P) {
                        // the operator hat (egui does not draw the combining circumflex)
                        let fs = 24.0 * look.font_scale();
                        let c = resp.rect.center();
                        let (apex, foot, half) = (c.y - 0.42 * fs, c.y - 0.27 * fs, 0.17 * fs);
                        let stroke = look.stroke(1.6, look.accent());
                        ui.painter().line_segment([pos2(c.x - half, foot), pos2(c.x, apex)], stroke);
                        ui.painter().line_segment([pos2(c.x, apex), pos2(c.x + half, foot)], stroke);
                    }
                    if resp.on_hover_text(hover).clicked() {
                        self.apply(op, Some(&mut *ctx));
                    }
                }
                ui.add_space(look.px(10.0));
                if ui
                    .add_enabled(
                        !self.undo.is_empty(),
                        egui::Button::new(big("Undo").size(16.0 * look.font_scale())).min_size(size),
                    )
                    .on_disabled_hover_text("nothing to undo (a measurement cannot be undone)")
                    .clicked()
                {
                    self.undo();
                }
                if ui
                    .add(egui::Button::new(big("reset |0⟩").size(16.0 * look.font_scale())).min_size(size))
                    .clicked()
                {
                    self.reset_ground();
                }
            });
            ui.add_space(look.px(4.0));
            let line = |ui: &mut egui::Ui, label: &str, text: String, col: egui::Color32| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new(label).color(look.weak_text()).size(14.0 * look.font_scale()));
                    ui.label(RichText::new(text).color(col).size(15.0 * look.font_scale()));
                });
            };
            line(ui, "state ∝", self.history_text(), look.text());
            if !self.result.is_empty() {
                line(ui, "last step:", self.result.clone(), look.accent());
            }
            let c = self.exp.coeffs();
            let res = commutator_residual(&c);
            let top = c.last().map_or(0.0, |z| z.norm_sqr());
            line(
                ui,
                "[â, â†]ψ = (ââ† − â†â)ψ = ψ ?",
                if res < 1e-9 {
                    format!("yes: ‖[â, â†]ψ − ψ‖ = {res:.1e}")
                } else {
                    format!(
                        "‖[â, â†]ψ − ψ‖ = {res:.2e}: the top level n = {} holds |c|² = {top:.1e} (truncated basis)",
                        LEVELS - 1
                    )
                },
                look.series(2),
            );
            let (alpha, res_a) = lowering_eigen_check(&c);
            line(
                ui,
                "eigenstate of â ?",
                if res_a < 1e-6 {
                    format!("yes: âψ = αψ with α = {} (|α|² = {:.3} = ⟨n⟩)", fmt_alpha(alpha), alpha.norm_sqr())
                } else {
                    format!("no: ‖âψ − ⟨â⟩ψ‖ = {res_a:.3} with ⟨â⟩ = {}", fmt_alpha(alpha))
                },
                look.series(0),
            );
        });
    }

    fn uncertainty_text(&self) -> String {
        let prod = self.moments.uncertainty_product();
        match fock_index(&self.exp.c0) {
            Some(n) => format!("{prod:.3} = (2n+1)/2, n = {n}"),
            None if prod < 0.5 * 1.002 => format!("{prod:.3} = ħ/2"),
            None => format!("{prod:.3} ≥ ħ/2"),
        }
    }
}

fn fmt_alpha(a: C64) -> String {
    format!("{} {} {:.2}i", fmt_num(a.re, 2), if a.im < 0.0 { "−" } else { "+" }, a.im.abs())
}

impl Scene for LadderScene {
    fn time(&self) -> f64 {
        self.t
    }

    fn step(&mut self, dt: f64, _ctx: &mut Ctx) {
        self.advance(dt);
    }

    fn reset(&mut self, _ctx: &mut Ctx) {
        self.restart();
    }

    fn measure_label(&self) -> Option<String> {
        Some("Measure E".into())
    }

    fn measure(&mut self, ctx: &mut Ctx) {
        self.measure_e(ctx);
    }

    fn on_key(&mut self, key: egui::Key, _m: egui::Modifiers, _ctx: &mut Ctx) -> bool {
        if key == egui::Key::V {
            self.style = self.style.next();
            return true;
        }
        false
    }

    fn action(&mut self, action: &str, ctx: &mut Ctx) -> bool {
        let mut words = action.split_whitespace();
        let cmd = words.next().unwrap_or("");
        let arg = words.next();
        let n = arg.and_then(|w| w.parse().ok()).unwrap_or(self.repeat_n);
        match cmd {
            "lower" | "a" => self.apply(Op::Lower, Some(ctx)),
            "raise" | "ad" => self.apply(Op::Raise, Some(ctx)),
            "number" => self.apply(Op::Number, Some(ctx)),
            "xop" => self.apply(Op::X, Some(ctx)),
            "pop" => self.apply(Op::P, Some(ctx)),
            "undo" => self.undo(),
            "ground" => self.reset_ground(),
            "fock" => {
                let Some(v) = common::num_arg(arg) else { return false };
                self.set_fock((v as usize).min(LEVELS - 1), false);
            }
            "measure_e" => self.measure_e(ctx),
            "measure_x" => self.measure_x(ctx),
            "repeat_e" => self.repeat_e(n, ctx),
            "repeat_x" => self.repeat_x(n, ctx),
            _ => return common::view_style_action(cmd, &mut self.view, &mut self.style),
        }
        true
    }

    fn central_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx) {
        let look = Look::of(ui);
        view_bar(ui, &mut self.view, &mut self.style);
        let m = self.moments;
        let (n_mean, dn) = n_stats(&self.exp.c0);
        readout_row(
            ui,
            &[
                ("⟨n⟩", format!("{n_mean:.3}"), Some(look.accent())),
                ("Δn", format!("{dn:.3}"), None),
                ("⟨E⟩ = (⟨n⟩ + ½)ħω", format!("{:.3}", n_mean + 0.5), None),
                ("⟨x⟩", fmt_num(m.x, 3), Some(look.series(1))),
                ("⟨p⟩", fmt_num(m.p, 3), None),
                ("σₓσₚ", self.uncertainty_text(), Some(look.series(2))),
            ],
        );
        let gap = look.px(8.0);
        let bottom_h = look.px(215.0);
        let top_h = (ui.available_height() - bottom_h - look.px(10.0)).max(220.0);
        let w = ui.available_width();
        let left_w = (0.56 * w).max(320.0);
        let right_w = w - left_w - gap;
        // row 1: wavefunction | Fock populations (top-aligned: a plain horizontal row
        // would centre its items in all the remaining height)
        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing.x = gap;
            // (a top-down column: inside the row a plain child would put x and p side by side)
            ui.allocate_ui_with_layout(vec2(left_w, top_h), egui::Layout::top_down(egui::Align::Min), |ui| {
                ui.set_width(left_w);
                let view = self.view;
                view.show(ui, top_h, |ui, d, h| match d {
                    Domain::Position => self.plot_x(ui, h),
                    Domain::Momentum => self.plot_p(ui, h),
                });
            });
            self.fock_chart(ui, right_w, top_h);
        });
        ui.add_space(look.px(6.0));
        // row 2: operators | ⟨n⟩(t) or the last histogram
        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing.x = gap;
            ui.allocate_ui_with_layout(vec2(left_w, bottom_h), egui::Layout::top_down(egui::Align::Min), |ui| {
                ui.set_width(left_w);
                ui.vertical(|ui| self.operator_panel(ui, ctx, left_w));
            });
            if let Some(r) = &self.repeated {
                r.show(ui, "ladder_hist", right_w, bottom_h);
            } else {
                // a fixed range (all-zero series, e.g. |0⟩, would otherwise give a 1e-10 axis)
                let (lo, hi) = self
                    .n_series
                    .points
                    .iter()
                    .chain(&self.x_series.points)
                    .fold((-1.0f64, 1.0f64), |(lo, hi), p| (lo.min(p[1]), hi.max(p[1])));
                let mut ts = TimeSeriesPlot::new("ladder_ts")
                    .series(&self.n_series)
                    .series(&self.x_series)
                    .y_range(lo.floor() - 0.2, hi.ceil() + 0.6)
                    .window(if self.t < 4.0 * TAU {
                        TimeWindow::All
                    } else {
                        TimeWindow::Last(4.0 * TAU)
                    })
                    .now(self.t)
                    .labels("t", "")
                    .width(right_w)
                    .height(bottom_h);
                for (t, l) in &self.events {
                    ts = ts.event(*t, l.clone());
                }
                ts.show(ui);
            }
        });
    }

    fn side_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx) {
        let look = ctx.look;
        ui.label(RichText::new("Start state").strong());
        ui.horizontal_wrapped(|ui| {
            for (p, label) in [
                (LadderPreset::Ground, "|0⟩"),
                (LadderPreset::Three, "|3⟩"),
                (LadderPreset::Coherent, "coherent |α⟩"),
                (LadderPreset::Superposition, "(|0⟩ + |1⟩)/√2"),
            ] {
                if ui.selectable_label(self.preset == p, label).clicked() {
                    self.preset = p;
                    self.restart();
                }
            }
        });
        if self.preset == LadderPreset::Coherent {
            let (mut re, mut im) = (self.alpha.re, self.alpha.im);
            let mut changed = ui.add(egui::Slider::new(&mut re, -3.0..=3.0).text("Re α")).changed();
            changed |= ui.add(egui::Slider::new(&mut im, -3.0..=3.0).text("Im α")).changed();
            if changed {
                self.alpha = c64(re, im);
                self.restart();
            }
        }
        hint(ui, "Fock chart: click a bar for |n⟩ alone, shift-click to add or remove it.");

        section(ui, "Operators");
        hint(
            ui,
            "The buttons under the plots apply â, â†, the number operator â†â, or the position and momentum operators x and p to the current state (with its current phases), show the norm of the result and normalise it. This is algebra on the state, not a time evolution or a measurement.",
        );

        section(ui, "View");
        ui.checkbox(&mut self.classical, "classical particle (same start ⟨x⟩, ⟨p⟩)");
        ui.checkbox(&mut self.poisson_overlay, "Poisson with the same ⟨n⟩ (dashed)");
        hint(ui, "x / p and the display mode: switches above the plots (V cycles the mode).");

        section(ui, "Measure");
        ui.add(
            egui::Slider::new(&mut self.width_x, 0.05..=1.5)
                .logarithmic(true)
                .text("x detector σ"),
        );
        ui.horizontal(|ui| {
            if ui.button(RichText::new("Measure E  (M)").color(look.measure())).clicked() {
                self.measure_e(ctx);
            }
            if ui.button(RichText::new("Measure x").color(look.measure())).clicked() {
                self.measure_x(ctx);
            }
        });
        if let Some(n) = repeat_button(ui, "ladder_rep_e", "Repeat E", &mut self.repeat_n) {
            self.repeat_e(n, ctx);
        }
        if let Some(n) = repeat_button(ui, "ladder_rep_x", "Repeat x", &mut self.repeat_n) {
            self.repeat_x(n, ctx);
        }
        hint(
            ui,
            "E: Fock state n with probability |cₙ|²; the state collapses to |n⟩ (see the chart) and only its phase turns afterwards.",
        );
        if let Some(l) = self.flash.label() {
            ui.label(RichText::new(format!("last outcome: {l}")).color(look.measure()));
        }
    }

    fn notes(&self) -> String {
        let intro = match self.preset {
            LadderPreset::Ground => {
                "# This preset: |0⟩\nPress â† a few times: |0⟩ → |1⟩ → |2⟩ → |3⟩, with the factors √1, √2, √3 shown before normalising; ⟨E⟩ climbs by ħω per step. Press â on |0⟩: the result is 0.\n\n"
            }
            LadderPreset::Three => {
                "# This preset: |3⟩\nA Fock state: stationary, σₓσₚ = (2·3 + 1)/2 = 3.5. â gives √3 |2⟩, â† gives √4 |4⟩, â†â gives 3|3⟩ (the same state). The position and momentum operators x and p give superpositions of |2⟩ and |4⟩.\n\n"
            }
            LadderPreset::Coherent => {
                "# This preset: a coherent state α = 1.5\nThe bars follow the Poisson law with ⟨n⟩ = |α|² = 2.25 (dashed). Press â: âψ_α = αψ_α, so after normalising the state is the same (only a global phase); the eigenstate line shows α, which turns as α e^(−iωt). Press â†: not an eigenstate; ⟨n⟩ jumps by more than one and the bars are no longer Poisson.\n\n"
            }
            LadderPreset::Superposition => {
                "# This preset: (|0⟩ + |1⟩)/√2\nThe packet sloshes at ω. â gives |0⟩ (√1·|0⟩/√2, normalised); â† gives (|1⟩ + √2|2⟩)/√3, which sloshes and breathes.\n\n"
            }
        };
        format!("{intro}{NOTES}")
    }
}

const NOTES: &str = "\
# Ladder operators (lecture script 6.8)
Units ħ = m = ω = 1, so the script's dimensionless χ = √(mω/ħ) x is simply x here.
- (6.191, 6.192) â = (∂/∂χ + χ)/√2 is the annihilation operator, â† = (−∂/∂χ + χ)/√2 the creation operator. Their commutator is [â, â†] = 1 (6.194).
- (6.193) Ĥ = ħω(â†â + ½): the energy eigenstates are the eigenstates of the number operator â†â.
- (6.201, 6.204) If â†âψₙ = nψₙ, then âψₙ is an eigenstate with n − 1 and â†ψₙ one with n + 1: â lowers and â† raises the energy by ħω.
- (6.206 – 6.215) The ladder must end at the bottom: âψ₀ = 0, i.e. dψ₀/dχ = −χψ₀, so ψ₀ = π^(−1/4) e^(−χ²/2), with E₀ = ħω/2: the zero-point energy.
- (6.224, 6.225) â†ψₙ = √(n+1) ψₙ₊₁ and âψₙ = √n ψₙ₋₁, so ψₙ = (â†)ⁿψ₀/√(n!) ∝ Hₙ(χ) e^(−χ²/2) and Eₙ = (n + ½)ħω.
- (6.230, 6.231) The position and momentum operators are x = √(ħ/2mω)(â† + â) and p = i√(mħω/2)(â† − â) (hats left out in this panel); for ψₙ: ⟨x⟩ = ⟨p⟩ = 0, σₓσₚ = (2n + 1)ħ/2 (6.239).
- PS10 40: coherent states are the eigenstates of â, âψ_α = αψ_α, with Poisson weights |cₙ|² = e^(−|α|²)|α|^(2n)/n!.

# What the buttons do
They apply the operator to the current state cₙ(t), show the norm of the (unnormalised) result and normalise it. This is algebra on the state, not a physical time evolution or a measurement: â, â†, x and p are not unitary.
- ‖âψ‖² = ⟨ψ|â†â|ψ⟩ = ⟨n⟩ and ‖â†ψ‖² = ⟨ââ†⟩ = ⟨n⟩ + 1: the \"last step\" line shows both sides.
- The commutator line applies ââ† − â†â with the same code as the buttons and checks that it returns ψ.
- The basis holds n = 0 … 39. If â† pushes population beyond n = 39 the status line says how much was dropped, and the commutator check then fails at the top level.
- Undo goes back one operator; a measurement cannot be undone.

# Try
- The Fock chart: bar height |cₙ|², colour arg cₙ (the phasors turn as e^(−iEₙt)); ⟨n⟩ and Δn are marked. A Fock state has Δn = 0.
- â†â on |n⟩ leaves the state unchanged (eigenvalue n); on a superposition it reweights cₙ → n cₙ.
- Measure E (M): one Fock state with probability |cₙ|²; Repeat E: the histogram follows the bars.

# Behind the scenes
The state is the coefficient vector cₙ on the analytic oscillator states n = 0 … 39 (textbook sign, as in the script: ψ₁ ∝ +χ e^(−χ²/2)), evolved exactly as cₙ e^(−iEₙt). The operators act on this vector with the matrix elements above; the tests check them against χ − d/dχ, x and −i d/dx applied to ψ on the grid.";

#[cfg(test)]
mod tests {
    use super::*;
    use qb_core::QbRng;
    use qb_core::num::ho_psi;

    fn random_state(rng: &mut QbRng, top: usize) -> Vec<C64> {
        let mut c: Vec<C64> = (0..LEVELS)
            .map(|k| if k < top { c64(rng.normal(), rng.normal()) } else { c64(0.0, 0.0) })
            .collect();
        let s = norm2(&c).sqrt();
        c.iter_mut().for_each(|z| *z /= s);
        c
    }

    #[test]
    fn number_operator_and_commutator() {
        // â†â|n⟩ = n|n⟩
        for n in 0..LEVELS {
            let mut c = vec![c64(0.0, 0.0); LEVELS];
            c[n] = c64(1.0, 0.0);
            let nc = raise(&lower(&c)).0;
            for (k, z) in nc.iter().enumerate() {
                let want = if k == n { n as f64 } else { 0.0 };
                assert!((z - c64(want, 0.0)).norm() < 1e-12, "â†â|{n}⟩ at {k}");
            }
            assert_eq!(number(&c)[n], c64(n as f64, 0.0));
        }
        // [â, â†] = 1 on random states away from the top of the basis
        let mut rng = QbRng::seed(11);
        for _ in 0..20 {
            let c = random_state(&mut rng, LEVELS - 2);
            assert!(commutator_residual(&c) < 1e-12);
            // ‖âψ‖² = ⟨n⟩, ‖â†ψ‖² = ⟨n⟩ + 1
            let (n, _) = n_stats(&c);
            assert!((norm2(&lower(&c)) - n).abs() < 1e-10);
            let (up, lost) = raise(&c);
            assert!(lost == 0.0 && (norm2(&up) - n - 1.0).abs() < 1e-10);
        }
        // the truncation edge is reported, not hidden
        let mut c = vec![c64(0.0, 0.0); LEVELS];
        c[LEVELS - 1] = c64(1.0, 0.0);
        let (up, lost) = raise(&c);
        assert!((lost - LEVELS as f64).abs() < 1e-12 && norm2(&up) < 1e-30);
        assert!(commutator_residual(&c) > 1.0);
    }

    #[test]
    fn coherent_state_is_an_eigenstate_of_a() {
        let s = LadderScene::new(LadderPreset::Coherent);
        let c = s.exp.coeffs();
        let (alpha, res) = lowering_eigen_check(&c);
        assert!((alpha - c64(1.5, 0.0)).norm() < 1e-9 && res < 1e-9, "α = {alpha}, residual {res}");
        let a = lower(&c);
        for (x, y) in a.iter().zip(&c) {
            assert!((x - y * 1.5).norm() < 1e-9);
        }
        // â† on it is not an eigenstate of â, and ⟨n⟩ grows by 2 − 1/(⟨n⟩ + 1)
        let (up, _) = raise(&c);
        let (_, res) = lowering_eigen_check(&up);
        assert!(res > 0.1);
        let (n0, _) = n_stats(&c);
        let (n1, _) = n_stats(&up);
        assert!((n1 - n0 - (2.0 - 1.0 / (n0 + 1.0))).abs() < 1e-9);
        // ‖âψ_α‖ = |α|; the scene keeps the state up to the phase of α
        let mut s = LadderScene::new(LadderPreset::Coherent);
        s.advance(0.7);
        let before = s.wave.density();
        s.apply(Op::Lower, None);
        let after = s.wave.density();
        let d: f64 = before.iter().zip(&after).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(d < 1e-9, "|ψ|² unchanged by â on |α⟩");
    }

    /// (â†)ⁿψ₀/√n! with â† = (χ − d/dχ)/√2 applied on the grid gives the analytic ψₙ.
    #[test]
    fn creation_operator_on_the_grid_builds_psi_n() {
        // a coarse grid (p_max = 33): each spectral derivative multiplies round-off noise at
        // the highest momenta by p_max, which 8 derivatives on a fine grid would blow up
        let g = Grid1D::new(-12.0, 12.0, 256);
        let mut w = Wave1D::from_real_fn(g, |x| ho_psi(0, x, 1.0));
        let mut fact = 1.0;
        for n in 1..=8 {
            // −d/dχ ψ = −i p̂ψ
            let pp = w.p_psi();
            let next: Vec<C64> = w
                .psi
                .iter()
                .zip(&pp)
                .enumerate()
                .map(|(i, (psi, dp))| (psi * g.x(i) - c64(0.0, 1.0) * dp) * std::f64::consts::FRAC_1_SQRT_2)
                .collect();
            w = Wave1D::from_samples(g, next);
            fact *= n as f64;
            let err = w
                .psi
                .iter()
                .enumerate()
                .map(|(i, z)| (z / fact.sqrt() - c64(ho_psi(n, g.x(i), 1.0), 0.0)).norm())
                .fold(0.0, f64::max);
            assert!(err < 1e-6, "n = {n}: max error {err}");
        }
    }

    /// x̂ and p̂ as matrices on cₙ agree with x·ψ and −i dψ/dx on the grid.
    #[test]
    fn position_and_momentum_matrices_match_the_grid() {
        let s = LadderScene::new(LadderPreset::Ground);
        let mut rng = QbRng::seed(5);
        let c = random_state(&mut rng, 12);
        let build = |c: &[C64]| -> Wave1D {
            let e = EigenExpansion::from_coeffs(s.basis.clone(), c);
            e.wave()
        };
        let psi = build(&c);
        let xpsi: Vec<C64> = psi.psi.iter().enumerate().map(|(i, z)| z * s.basis.grid.x(i)).collect();
        let ppsi = psi.p_psi();
        let from_x = build(&position(&c).0);
        let from_p = build(&momentum(&c).0);
        let ex = from_x.psi.iter().zip(&xpsi).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max);
        let ep = from_p.psi.iter().zip(&ppsi).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max);
        assert!(ex < 1e-8 && ep < 1e-8, "x̂: {ex}, p̂: {ep}");
    }

    #[test]
    fn buttons_on_fock_states() {
        let mut s = LadderScene::new(LadderPreset::Ground);
        s.apply(Op::Lower, None);
        assert!(s.result.starts_with("â|0⟩ = 0"));
        assert_eq!(fock_index(&s.exp.c0), Some(0));
        for k in 1..=3 {
            s.apply(Op::Raise, None);
            assert_eq!(fock_index(&s.exp.c0), Some(k));
        }
        assert_eq!(s.history_text(), "â† â† â† |0⟩");
        assert!((s.moments.uncertainty_product() - 3.5).abs() < 1e-6);
        s.undo();
        assert_eq!(fock_index(&s.exp.c0), Some(2));
        let mut s = LadderScene::new(LadderPreset::Superposition);
        s.apply(Op::Raise, None);
        let p = s.exp.probabilities();
        assert!((p[1] - 1.0 / 3.0).abs() < 1e-12 && (p[2] - 2.0 / 3.0).abs() < 1e-12);
    }
}
