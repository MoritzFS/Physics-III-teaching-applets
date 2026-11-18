//! The hydrogen scene: one struct for the five presets (they share the 3D
//! view, the 2D slice, the radial panel, the level diagram and the
//! measurements; each preset has its own state controls and notes).
//!
//! parameters → `restart()` (prepares the state at t = 0) → `step()` (phases
//! e^(−iEt)) → `central_ui()` (3D view, level diagram or Zeeman diagram,
//! slice, radial panel, time series, histogram) → `side_ui()` (state, view,
//! measurements) → `notes()`.

use std::f64::consts::{FRAC_1_SQRT_2, PI, TAU};

use qb_core::egui::{self, Align, Layout, Rect, RichText, Sense, UiBuilder, pos2, vec2};
use qb_core::glam::DVec3;
use qb_core::num::ShotHistogram;
use qb_core::widgets::{
    FlashKind, HeatmapMode, Heatmap2D, HistScale, HistogramPlot, MeasureFlash, TimeSeries, TimeSeriesPlot, TimeWindow, fmt_num,
    readout_row, repeat_button, segmented,
};
use qb_core::{C64, Ctx, Look, Scene, c64};

use crate::notes;
use crate::orbital::{
    AU_FIELD_T, AU_TIME_FS, AngOp, BASIS, Coeffs, HARTREE_EV, HC_EV_NM, Nlm, active, apply_ladder, born_sample, closed, energy_ev, expect,
    expect_r, expect_xyz, ladder_factor, radial_density, signed, sub, tables,
};
use crate::panels::{LevelDiagram, density_scale, level_diagram, prob_bars, radial_panel, slice_field};
use crate::state::{AtomState, Obs, mean_and_spread, measure, outcomes, zero};
use crate::view3d::{Mode3D, Plane, RenderInput, View3D};
use crate::zeeman;

/// Which preset the scene was built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// PS11 43: quantum numbers, ladder operators
    Numbers,
    /// PS11 44: energy and radius
    Radius,
    /// PS11 45: (ψ₂₁₁ + ψ₂₁₋₁)/√2 and l_y
    Ly,
    /// 1s + 2p: oscillating dipole
    Dipole,
    /// PS12 47: normal Zeeman effect
    Zeeman,
}

/// The last "repeat N" run.
struct Repeated {
    obs: Obs,
    hist: ShotHistogram,
    predicted: Vec<f64>,
    labels: Vec<String>,
    /// measured positions (position measurements only)
    points: Vec<[f64; 3]>,
    /// state version and time it was measured at (repeats on the same state accumulate)
    version: u64,
    t: f64,
}

const SLICE_N: usize = 150;
const LIVE_POINTS_MAX: usize = 20_000;

pub struct HydrogenScene {
    variant: Variant,
    // ---- preparation (parameters)
    sel: Nlm,
    /// |α|² of ψ₂₁₁ (PS11 45)
    ly_w: f64,
    /// relative phase δ of ψ₂₁₋₁ (radians)
    ly_phase: f64,
    /// 2p sublevel and |c₂|², phase δ (dipole scene)
    dip_m: i32,
    dip_w: f64,
    dip_phase: f64,
    /// Zeeman: field (T), selected transition (m lower, m upper), observation along B
    zee_b: f64,
    zee_sel: (i32, i32),
    zee_along: bool,
    /// changes when the preparation changes (dots of single shots belong to one preparation)
    prep_version: u64,

    // ---- dynamics
    t: f64,
    state: AtomState,
    /// after a position measurement: (centre, σ) of the localized packet
    localized: Option<([f64; 3], f64)>,
    /// changes whenever the state changes other than by time evolution
    version: u64,
    radius: f64,
    /// brightness scale ρ_ref of the views (see `panels::density_scale`)
    rho_max: f64,
    /// magnification of the ⟨r⟩ arrow
    dipole_scale: f64,

    // ---- views
    view: View3D,
    slice: Heatmap2D,
    live_points: Vec<[f64; 3]>,
    n_points: usize,
    points_version: Option<u64>,
    sx: TimeSeries,
    sy: TimeSeries,
    sz: TimeSeries,
    events: Vec<(f64, String)>,

    // ---- measurement
    flash: MeasureFlash,
    det_sigma: f64,
    repeat_n: usize,
    repeat_obs: Obs,
    repeated: Option<Repeated>,
    dots: Vec<[f64; 3]>,
    ladder_msg: Option<String>,
    /// a ladder button was pressed in the side panel (applied with the Ctx)
    pending_ladder: Option<bool>,
}

impl HydrogenScene {
    pub fn new(variant: Variant) -> Self {
        let mut view = View3D::default();
        let sel = match variant {
            Variant::Numbers => Nlm::new(4, 2, 1),
            Variant::Radius => Nlm::new(1, 0, 0),
            _ => Nlm::new(2, 1, 1),
        };
        match variant {
            Variant::Numbers | Variant::Radius => {
                view.cut = true;
                view.plane = Plane::XZ;
            }
            Variant::Ly => {
                view.plane = Plane::XY;
                view.cam.pitch = 38f64.to_radians();
            }
            Variant::Dipole | Variant::Zeeman => {
                view.plane = Plane::XZ;
            }
        }
        if variant == Variant::Numbers {
            view.mode = Mode3D::Iso;
        }
        let mut s = HydrogenScene {
            variant,
            sel,
            ly_w: 0.5,
            ly_phase: 0.0,
            dip_m: 0,
            dip_w: 0.5,
            dip_phase: 0.0,
            zee_b: zeeman::B_DEFAULT,
            zee_sel: (0, 0),
            zee_along: false,
            prep_version: 0,
            t: 0.0,
            state: AtomState::eigen(sel, 0.0),
            localized: None,
            version: 0,
            radius: 10.0,
            rho_max: 1.0,
            dipole_scale: 1.0,
            view,
            slice: Heatmap2D::new(),
            live_points: vec![],
            n_points: 4000,
            points_version: None,
            sx: TimeSeries::new("⟨x⟩"),
            sy: TimeSeries::new("⟨y⟩"),
            sz: TimeSeries::new("⟨z⟩"),
            events: vec![],
            flash: MeasureFlash::new(),
            det_sigma: 0.4,
            repeat_n: 1000,
            repeat_obs: Obs::Position,
            repeated: None,
            dots: vec![],
            ladder_msg: None,
            pending_ladder: None,
        };
        s.repeat_obs = match variant {
            Variant::Ly => Obs::Lz,
            Variant::Dipole | Variant::Zeeman => Obs::Energy,
            _ => Obs::Position,
        };
        s.restart();
        s
    }

    /// The prepared coefficients at t = 0.
    fn prepared(&self) -> Coeffs {
        let mut c = zero();
        match self.variant {
            Variant::Numbers | Variant::Radius => c[self.sel.index()] = c64(1.0, 0.0),
            Variant::Ly => {
                c[Nlm::new(2, 1, 1).index()] = c64(self.ly_w.sqrt(), 0.0);
                c[Nlm::new(2, 1, -1).index()] = C64::from_polar((1.0 - self.ly_w).max(0.0).sqrt(), self.ly_phase);
            }
            Variant::Dipole => {
                c[0] = c64((1.0 - self.dip_w).max(0.0).sqrt(), 0.0);
                c[Nlm::new(2, 1, self.dip_m).index()] = C64::from_polar(self.dip_w.sqrt(), self.dip_phase);
            }
            Variant::Zeeman => {
                let (m1, m2) = self.zee_sel;
                c[Nlm::new(zeeman::N_LOWER, 1, m1).index()] = c64(FRAC_1_SQRT_2, 0.0);
                c[Nlm::new(zeeman::N_UPPER, 2, m2).index()] = c64(FRAC_1_SQRT_2, 0.0);
            }
        }
        c
    }

    /// Back to t = 0 with the current parameters.
    fn restart(&mut self) {
        self.t = 0.0;
        self.state = AtomState::new(self.prepared(), 0.0);
        if self.variant == Variant::Zeeman {
            self.state.b_au = self.zee_b / AU_FIELD_T;
        }
        self.localized = None;
        self.flash.clear();
        self.events.clear();
        self.repeated = None;
        self.ladder_msg = None;
        for s in [&mut self.sx, &mut self.sy, &mut self.sz] {
            s.clear();
        }
        self.state_changed();
        self.record();
    }

    /// A parameter of the preparation changed: new state, forget the old dots.
    fn prepare(&mut self) {
        self.prep_version += 1;
        self.dots.clear();
        self.restart();
    }

    /// After anything but time evolution changed the state.
    fn state_changed(&mut self) {
        self.version += 1;
        let c = self.state.coeffs(self.t);
        let t = tables();
        let r = active(&c)
            .iter()
            .map(|&k| t.radial_cdf[BASIS[k].nl()].quantile(0.99) * 1.1)
            .fold(4.0, f64::max);
        self.radius = r.ceil();
        self.dipole_scale = self.dipole_scale_for();
        self.rho_max = density_scale(&c, self.radius).1;
        self.view.offset = self.view.offset.clamp(-self.radius, self.radius);
        self.points_version = None;
    }

    /// A round magnification so that the largest ⟨r⟩ over one period is about 0.4 of the view radius.
    fn dipole_scale_for(&self) -> f64 {
        let w = self.state.max_bohr_frequency();
        if w == 0.0 {
            return 1.0;
        }
        let amp = (0..16)
            .map(|k| {
                let ck = self.state.coeffs(self.t + TAU / w * k as f64 / 16.0);
                let [x, y, z] = expect_xyz(&ck);
                (x * x + y * y + z * z).sqrt()
            })
            .fold(0.0, f64::max);
        if amp < 1e-9 {
            return 1.0;
        }
        let want = 0.4 * self.radius / amp;
        [1.0, 2.0, 5.0, 10.0, 20.0, 50.0].into_iter().filter(|v| *v <= want).fold(1.0, f64::max)
    }

    fn coeffs(&self) -> Coeffs {
        self.state.coeffs(self.t)
    }

    fn record(&mut self) {
        let [x, y, z] = if self.localized.is_some() { [f64::NAN; 3] } else { expect_xyz(&self.coeffs()) };
        self.sx.push(self.t, x);
        self.sy.push(self.t, y);
        self.sz.push(self.t, z);
    }

    /// One Bohr period of the slowest phase that matters for the display.
    fn display_period(&self) -> f64 {
        match self.variant {
            Variant::Dipole => TAU / (crate::orbital::energy(2) - crate::orbital::energy(1)),
            Variant::Zeeman => TAU / (crate::orbital::energy(zeeman::N_UPPER) - crate::orbital::energy(zeeman::N_LOWER)),
            _ => {
                // the global phase of the (first) component
                let n = self.state.active().first().map_or(1, |&k| BASIS[k].n);
                TAU / crate::orbital::energy(n).abs()
            }
        }
    }

    /// Born samples for the point-cloud mode (regenerated when the state changed).
    fn ensure_points(&mut self, ctx: &mut Ctx) {
        if self.points_version == Some(self.version) && self.live_points.len() == self.n_points {
            return;
        }
        let c = self.coeffs();
        let act = active(&c);
        self.live_points = (0..self.n_points).filter_map(|_| born_sample(&c, &act, ctx.rng)).collect();
        self.points_version = Some(self.version);
    }

    // ------------------------------------------------------------------ measurements

    fn main_obs(&self) -> Obs {
        match self.variant {
            Variant::Numbers | Variant::Radius => Obs::Position,
            Variant::Ly => Obs::Lz,
            Variant::Dipole | Variant::Zeeman => Obs::Energy,
        }
    }

    fn measure_obs(&mut self, obs: Obs, ctx: &mut Ctx) {
        if self.localized.is_some() {
            ctx.set_status("The state is localized after the position measurement: reset (R) to continue.");
            return;
        }
        if obs == Obs::Position {
            return self.measure_position(ctx);
        }
        let c = self.coeffs();
        let e = self.state.energies();
        let Some((o, after)) = measure(&c, obs, &e, ctx.rng) else { return };
        let certain = o.prob > 1.0 - 1e-9;
        self.state.set(after, self.t);
        let kind = match obs {
            Obs::Energy => FlashKind::Level {
                index: 0,
                energy: o.value * HARTREE_EV,
            },
            _ => FlashKind::Other,
        };
        self.flash.trigger(kind, format!("{}: {}", obs.symbol(), o.label));
        self.events.push((self.t, obs.symbol().into()));
        ctx.set_status(format!(
            "measured {} → {} (probability {:.2}){}",
            obs.symbol(),
            o.label,
            o.prob,
            if certain {
                "; certain outcome, the state is unchanged".to_string()
            } else {
                format!("; the state is projected onto that eigenspace: {}", self.state_text())
            }
        ));
        self.state_changed();
        self.record();
    }

    fn measure_position(&mut self, ctx: &mut Ctx) {
        let c = self.coeffs();
        let act = active(&c);
        let Some(p) = born_sample(&c, &act, ctx.rng) else { return };
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        self.dots.push(p);
        self.localized = Some((p, self.det_sigma));
        self.flash
            .trigger(FlashKind::Other, format!("r = ({:.1}, {:.1}, {:.1}) a₀", p[0], p[1], p[2]));
        self.events.push((self.t, "r".into()));
        ctx.set_status(format!(
            "position measured: (x, y, z) = ({:.2}, {:.2}, {:.2}) a₀, |r| = {r:.2} a₀. The state is now localized (detector σ = {} a₀); reset (R) to continue. Dots from earlier preparations are kept.",
            p[0], p[1], p[2], self.det_sigma
        ));
        ctx.playing = false;
        self.points_version = None;
        self.record();
    }

    /// The outcome bins of a discrete observable: (values, labels).
    fn bins(&self, obs: Obs, c: &Coeffs) -> (Vec<f64>, Vec<String>) {
        let e = self.state.energies();
        let lmax = active(c).iter().map(|&k| BASIS[k].l).max().unwrap_or(1).max(1) as i32;
        match obs {
            Obs::Energy if self.state.b_au == 0.0 => (
                (1..=4).map(crate::orbital::energy).collect(),
                (1..=4).map(|n| format!("n = {n}")).collect(),
            ),
            Obs::Energy => {
                let o = outcomes(c, obs, &e);
                (o.iter().map(|o| o.value).collect(), o.iter().map(|o| o.label.clone()).collect())
            }
            Obs::L2 => ((0..4).map(|l| (l * (l + 1)) as f64).collect(), (0..4).map(|l| format!("l = {l}")).collect()),
            Obs::Lz | Obs::Ly => (
                (-lmax..=lmax).map(|m| m as f64).collect(),
                (-lmax..=lmax).map(|m| format!("{}ħ", signed(m))).collect(),
            ),
            Obs::Position => (vec![], vec![]),
        }
    }

    /// "Repeat N": N fresh copies of the current state.
    fn repeat(&mut self, obs: Obs, n: usize, ctx: &mut Ctx) {
        if self.localized.is_some() {
            ctx.set_status("The state is localized after the position measurement: reset (R) to continue.");
            return;
        }
        let c = self.coeffs();
        let accumulate = self
            .repeated
            .as_ref()
            .is_some_and(|r| r.obs == obs && r.version == self.version && (r.t - self.t).abs() < 1e-9);
        if !accumulate {
            let (hist, predicted, labels) = if obs == Obs::Position {
                let bins = 40;
                let hist = ShotHistogram::binned(0.0, 1.25 * self.radius, bins);
                let m = 2000;
                let xs: Vec<f64> = (0..m).map(|i| 1.25 * self.radius * (i as f64 + 0.5) / m as f64).collect();
                let dens: Vec<f64> = xs.iter().map(|&r| radial_density(&c, r)).collect();
                let pred = hist.predict_from_density(&xs, &dens);
                (hist, pred, vec![])
            } else {
                let (vals, labels) = self.bins(obs, &c);
                let outs = outcomes(&c, obs, &self.state.energies());
                let pred = vals
                    .iter()
                    .map(|v| outs.iter().filter(|o| (o.value - v).abs() < 1e-9).map(|o| o.prob).sum())
                    .collect();
                (ShotHistogram::discrete(vals.len()), pred, labels)
            };
            self.repeated = Some(Repeated {
                obs,
                hist,
                predicted,
                labels,
                points: vec![],
                version: self.version,
                t: self.t,
            });
        }
        let (vals, _) = self.bins(obs, &c);
        let e = self.state.energies();
        let rep = self.repeated.as_mut().unwrap();
        if obs == Obs::Position {
            let act = active(&c);
            let pts: Vec<[f64; 3]> = (0..n).filter_map(|_| born_sample(&c, &act, ctx.rng)).collect();
            let rs: Vec<f64> = pts.iter().map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()).collect();
            rep.hist.add_all(&rs);
            rep.points.extend(pts);
            let extra = rep.points.len().saturating_sub(LIVE_POINTS_MAX);
            rep.points.drain(..extra);
            self.view.mode = Mode3D::Points;
            ctx.set_status(format!(
                "{} copies measured in position: mean |r| = {:.3} a₀ (⟨r⟩ = {:.3} a₀); the 3D view shows the measured positions",
                rep.hist.total(),
                rep.hist.mean(),
                expect_r(&c, 1)
            ));
        } else {
            let outs = outcomes(&c, obs, &e);
            let probs: Vec<f64> = outs.iter().map(|o| o.prob).collect();
            for _ in 0..n {
                if let Some(k) = ctx.rng.weighted(&probs)
                    && let Some(bin) = vals.iter().position(|v| (v - outs[k].value).abs() < 1e-9)
                {
                    rep.hist.add_index(bin);
                }
            }
            let summary: Vec<String> = outs.iter().map(|o| format!("{}: {:.2}", o.label, o.prob)).collect();
            ctx.set_status(format!(
                "{} copies measured in {}; Born rule: {}",
                rep.hist.total(),
                obs.symbol(),
                summary.join(", ")
            ));
        }
    }

    // ------------------------------------------------------------------ texts

    /// The state as a formula, e.g. "0.71 ψ₂₁₁ + 0.71 ψ₂₁₋₁".
    fn state_text(&self) -> String {
        if let Some((p, _)) = self.localized {
            return format!("localized at ({:.1}, {:.1}, {:.1}) a₀", p[0], p[1], p[2]);
        }
        let c = self.state.coeffs(0.0 + self.t);
        let act = active(&c);
        if let [k] = act.as_slice() {
            return BASIS[*k].name();
        }
        // phases relative to the first component
        let ph0 = c[act[0]].arg();
        let mut s = String::new();
        for (i, &k) in act.iter().enumerate() {
            let z = c[k] * C64::from_polar(1.0, -ph0);
            let mag = z.norm();
            let arg = z.arg();
            let coef = if arg.abs() < 1e-6 {
                format!("{mag:.2}")
            } else if (arg.abs() - PI).abs() < 1e-6 {
                format!("−{mag:.2}")
            } else {
                if arg < 0.0 { format!("{mag:.2}e^(−i·{:.0}°)", -arg.to_degrees()) } else { format!("{mag:.2}e^(i·{:.0}°)", arg.to_degrees()) }
            };
            let coef = if i > 0 && !coef.starts_with('−') { format!(" + {coef}") } else if i > 0 { format!(" {coef}") } else { coef };
            s += &format!("{coef} {}", BASIS[k].name());
        }
        s
    }

    /// Title in the 3D view.
    fn title(&self) -> String {
        match self.variant {
            Variant::Ly if self.localized.is_none() && self.ly_w == 0.5 && self.ly_phase == 0.0 && self.state.active().len() == 2 => {
                "Ψ = (ψ₂₁₁ + ψ₂₁₋₁)/√2 ∝ −i R₂₁ sin θ sin φ   (p_y)".into()
            }
            _ => format!("ψ = {}", self.state_text()),
        }
    }

    // ------------------------------------------------------------------ UI pieces

    fn view_bar(&mut self, ui: &mut egui::Ui) {
        let look = Look::of(ui);
        ui.horizontal(|ui| {
            ui.label(RichText::new("3D").color(look.weak_text()));
            segmented(ui, |ui| {
                for m in [Mode3D::Cloud, Mode3D::Iso, Mode3D::Points] {
                    if ui.selectable_label(self.view.mode == m, m.label()).on_hover_text("display mode (V cycles)").clicked() {
                        self.view.mode = m;
                    }
                }
            });
            ui.label(RichText::new("V").monospace().color(look.weak_text()));
            ui.add_space(look.px(12.0));
            ui.checkbox(&mut self.view.cut, "cut open");
            ui.label(RichText::new("plane").color(look.weak_text()));
            segmented(ui, |ui| {
                for (p, name) in [(Plane::XZ, "x–z"), (Plane::XY, "x–y")] {
                    if ui.selectable_label(self.view.plane == p, name).on_hover_text("plane of the 2D slice and of the 3D cut").clicked() {
                        self.view.plane = p;
                    }
                }
            });
            let (_, _, nax) = self.view.plane.axes();
            let r = self.radius;
            ui.add(egui::Slider::new(&mut self.view.offset, -r..=r).text(format!("{nax} (a₀)")).fixed_decimals(1));
        });
    }

    fn readouts(&self, ui: &mut egui::Ui) {
        let look = Look::of(ui);
        let c = self.coeffs();
        if let Some((p, _)) = self.localized {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            readout_row(
                ui,
                &[(
                    "state",
                    format!("localized at |r| = {r:.2} a₀ by the position measurement · reset (R) to continue"),
                    Some(look.measure()),
                )],
            );
            return;
        }
        let e = self.state.energies();
        let (em, de) = mean_and_spread(&c, Obs::Energy, &e);
        let e_text = if de < 1e-12 {
            format!("{:.3} eV", em * HARTREE_EV)
        } else {
            format!("{:.3} eV (ΔE = {:.2} eV)", em * HARTREE_EV, de * HARTREE_EV)
        };
        let mut items: Vec<(&str, String, Option<egui::Color32>)> = vec![("⟨E⟩", e_text, None)];
        match self.variant {
            Variant::Numbers => {
                if let Some(q) = self.state.single() {
                    items.push(("l²", format!("{}ħ²", q.l * (q.l + 1)), Some(look.series(0))));
                    items.push(("l_z", format!("{}ħ", signed(q.m)), Some(look.series(1))));
                }
                items.push(("⟨r⟩", format!("{:.2} a₀", expect_r(&c, 1)), Some(look.accent())));
            }
            Variant::Radius => {
                let rinv = expect_r(&c, -1);
                items.push(("⟨Ê⟩ = −½⟨1/r⟩", format!("{:.3} eV", -0.5 * rinv * HARTREE_EV), Some(look.series(0))));
                items.push(("⟨r⟩", format!("{:.3} a₀", expect_r(&c, 1)), Some(look.accent())));
                items.push(("⟨r²⟩", format!("{:.3} a₀²", expect_r(&c, 2)), None));
                let [x, _, _] = expect_xyz(&c);
                items.push(("⟨x⟩", fmt_num(x, 3), None));
                items.push(("⟨x²⟩", format!("{:.3} a₀²", expect(&c, 2, AngOp::XX).re), None));
            }
            Variant::Ly => {
                let (lz, dlz) = mean_and_spread(&c, Obs::Lz, &e);
                let (ly, dly) = mean_and_spread(&c, Obs::Ly, &e);
                items.push(("⟨l_z⟩", format!("{}ħ", fmt_num(lz, 2)), Some(look.series(1))));
                items.push(("Δl_z", format!("{}ħ", fmt_num(dlz, 2)), Some(look.series(1))));
                items.push(("⟨l_y⟩", format!("{}ħ", fmt_num(ly, 2)), Some(look.series(2))));
                items.push((
                    "Δl_y",
                    format!("{}ħ{}", fmt_num(dly, 2), if dly < 1e-9 { "  (l_y eigenstate)" } else { "" }),
                    Some(look.series(2)),
                ));
            }
            Variant::Dipole | Variant::Zeeman => {
                let [x, y, z] = expect_xyz(&c);
                items.push(("⟨x⟩", fmt_num(x, 3), Some(look.series(0))));
                items.push(("⟨y⟩", fmt_num(y, 3), Some(look.series(1))));
                items.push(("⟨z⟩", fmt_num(z, 3), Some(look.series(2))));
                let w = self.state.max_bohr_frequency();
                if w > 0.0 {
                    items.push((
                        "Bohr period",
                        format!("{:.3} fs ({:.1} nm)", TAU / w * AU_TIME_FS, HC_EV_NM / (w * HARTREE_EV)),
                        None,
                    ));
                } else {
                    items.push(("Bohr period", "— (stationary state: no oscillation)".into(), None));
                }
            }
        }
        readout_row(ui, &items);
    }

    /// Right column: level diagram (or Zeeman diagram) above the 2D slice / spectrum.
    fn right_column(&mut self, ui: &mut egui::Ui, rect: Rect) {
        let look = Look::of(ui);
        let gap = look.px(6.0);
        let c = if self.localized.is_some() { zero() } else { self.coeffs() };
        if self.variant == Variant::Zeeman {
            let top = Rect::from_min_size(rect.min, vec2(rect.width(), rect.height() * 0.56));
            if let Some(t) = zeeman::zeeman_diagram(ui, top, self.zee_b, self.zee_sel) {
                self.zee_sel = t;
                self.prepare();
            }
            let bottom = Rect::from_min_max(pos2(rect.left(), top.bottom() + gap), rect.max);
            let mut child = ui.new_child(UiBuilder::new().max_rect(bottom).layout(Layout::top_down(Align::Min)));
            let h = bottom.height() - look.px(22.0);
            zeeman::spectrum(&mut child, vec2(bottom.width(), h), self.zee_b, self.zee_along, self.zee_sel.1 - self.zee_sel.0);
            return;
        }
        let top = Rect::from_min_size(rect.min, vec2(rect.width(), rect.height() * 0.47));
        let clickable = matches!(self.variant, Variant::Numbers | Variant::Radius | Variant::Dipole);
        let selected = match self.variant {
            Variant::Numbers | Variant::Radius => Some(self.sel),
            Variant::Dipole => Some(Nlm::new(2, 1, self.dip_m)),
            _ => None,
        };
        let clicked = level_diagram(
            ui,
            top,
            &LevelDiagram {
                coeffs: &c,
                selected,
                lyman: matches!(self.variant, Variant::Radius | Variant::Dipole),
                clickable,
            },
            &self.flash,
        );
        if let Some(q) = clicked {
            match self.variant {
                Variant::Numbers | Variant::Radius => {
                    self.sel = q;
                    self.prepare();
                }
                Variant::Dipole if q.n == 2 && q.l == 1 => {
                    self.dip_m = q.m;
                    self.prepare();
                }
                _ => {}
            }
        }
        let bottom = Rect::from_min_max(pos2(rect.left(), top.bottom() + gap), rect.max);
        let mut child = ui.new_child(UiBuilder::new().max_rect(bottom).layout(Layout::top_down(Align::Center)));
        self.slice_view(&mut child, bottom.height());
    }

    fn slice_view(&mut self, ui: &mut egui::Ui, height: f32) {
        let look = Look::of(ui);
        let c = self.coeffs();
        let l = self.radius;
        let field = slice_field(&c, self.localized, self.view.plane, self.view.offset, l, SLICE_N);
        let max_abs = if self.localized.is_some() { 1.0 } else { self.rho_max.sqrt() };
        self.slice
            .set_field(ui.ctx(), SLICE_N, SLICE_N, &field, HeatmapMode::PhaseBrightness, Some(max_abs));
        let (a, b, nax) = self.view.plane.axes();
        let resp = self.slice.show(ui, ("hy_slice", self.view.plane as u8), [-l, l, -l, l], (a, b), height);
        let painter = resp.painter(ui);
        let tr = resp.transform;
        // radial nodes of a single (n, l) as dashed circles where they cut the plane
        if self.localized.is_none()
            && let Some((n, ll)) = self.state.single_nl()
        {
            for rn in crate::orbital::radial_nodes(n, ll) {
                if rn > self.view.offset.abs() {
                    let rc = (rn * rn - self.view.offset * self.view.offset).sqrt();
                    let pts: Vec<egui::Pos2> = (0..=72)
                        .map(|k| {
                            let t = k as f64 / 72.0 * TAU;
                            tr.to_screen(rc * t.cos(), rc * t.sin())
                        })
                        .collect();
                    painter.add(egui::Shape::dashed_line(&pts, look.stroke(1.0, look.weak_text()), 4.0, 4.0));
                }
            }
        }
        // the measured position, if near the plane
        if let Some((p, _)) = self.localized {
            let v = DVec3::from_array(p);
            let (pa, pb, pn) = match self.view.plane {
                Plane::XZ => (v.x, v.z, v.y),
                Plane::XY => (v.x, v.y, v.z),
            };
            let s = tr.to_screen(pa, pb);
            painter.circle_stroke(s, look.px(6.0), look.stroke(1.5, look.measure()));
            painter.text(
                s + vec2(look.px(8.0), 0.0),
                egui::Align2::LEFT_CENTER,
                format!("{nax} = {pn:.1}"),
                look.font(11.0),
                look.measure(),
            );
        }
        qb_core::widgets::label_box(
            &painter,
            tr.rect.left_top() + vec2(look.px(5.0), look.px(22.0)),
            egui::Align2::LEFT_TOP,
            &format!("slice {nax} = {:.1} a₀", self.view.offset),
            look.weak_text(),
        );
        if resp.response.hovered() && !look.shot {
            resp.response.on_hover_text("ψ in the slice plane: hue = arg ψ, brightness = |ψ| (fixed scale); dashed: radial nodes");
        }
    }

    /// Bottom row: radial panel / slice, time series or |c_m|² bars, histogram.
    fn bottom_row(&mut self, ui: &mut egui::Ui, rect: Rect) {
        let look = Look::of(ui);
        let gap = look.px(8.0);
        let c = self.coeffs();
        let hist = self.repeated.is_some();
        let middle = matches!(self.variant, Variant::Ly | Variant::Dipole | Variant::Zeeman);
        let n_items = 1 + middle as usize + hist as usize;
        let w_hist = if hist { (rect.width() * if n_items == 3 { 0.28 } else { 0.4 }).max(look.px(230.0)) } else { 0.0 };
        let w_mid = if middle { if n_items == 3 { rect.width() * 0.3 } else { rect.width() * 0.45 } } else { 0.0 };
        let w_first = rect.width() - w_hist - w_mid - gap * (n_items as f32 - 1.0);
        let h = rect.height();
        let mut x = rect.left();
        let mut next = |w: f32| {
            let r = Rect::from_min_size(pos2(x, rect.top()), vec2(w, h));
            x += w + gap;
            r
        };
        // first: radial panel (slice in the Zeeman scene: the right column has the spectrum)
        let r1 = next(w_first);
        let mut ui1 = ui.new_child(UiBuilder::new().max_rect(r1).layout(Layout::top_down(Align::Min)));
        if self.variant == Variant::Zeeman {
            ui1.with_layout(Layout::top_down(Align::Center), |ui| self.slice_view(ui, h));
        } else {
            let c_rad = if self.localized.is_some() { self.state.coeffs(self.t) } else { c };
            let measured = self.localized.map(|(p, _)| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt());
            radial_panel(&mut ui1, "main", &c_rad, self.radius, measured, vec2(r1.width(), h));
        }
        if middle {
            let r2 = next(w_mid);
            let mut ui2 = ui.new_child(UiBuilder::new().max_rect(r2).layout(Layout::top_down(Align::Min)));
            if self.variant == Variant::Ly {
                let e = self.state.energies();
                let half = (r2.width() - gap) / 2.0;
                ui2.horizontal(|ui| {
                    for (obs, title, col, id) in [
                        (Obs::Lz, "P(l_z = mħ)", look.series(1), "lz"),
                        (Obs::Ly, "P(l_y = m_yħ)", look.series(2), "ly"),
                    ] {
                        let outs = if self.localized.is_some() { vec![] } else { outcomes(&c, obs, &e) };
                        let ms: Vec<i32> = (-1..=1).collect();
                        let probs: Vec<f64> = ms
                            .iter()
                            .map(|m| outs.iter().filter(|o| (o.value - *m as f64).abs() < 1e-9).map(|o| o.prob).sum())
                            .collect();
                        let labels: Vec<String> = ms.iter().map(|m| format!("{}ħ", signed(*m))).collect();
                        prob_bars(ui, id, title, &labels, &probs, vec2(half, h), col);
                    }
                });
            } else {
                let period = self.display_period();
                TimeSeriesPlot::new("hy_dipole_t")
                    .series(&self.sx)
                    .series(&self.sy)
                    .series(&self.sz)
                    .window(TimeWindow::Fixed((self.t - 2.5 * period).max(0.0), self.t.max(2.5 * period)))
                    .now(self.t)
                    .labels("t (ħ/E_h)", "⟨r⟩ (a₀)")
                    .width(r2.width())
                    .height(h)
                    .show(&mut ui2);
            }
        }
        if let Some(rep) = &self.repeated {
            let r3 = next(w_hist);
            let mut ui3 = ui.new_child(UiBuilder::new().max_rect(r3).layout(Layout::top_down(Align::Min)));
            let mut plot = HistogramPlot::new("hy_hist", &rep.hist).predicted(&rep.predicted).width(r3.width()).height(h);
            if rep.obs == Obs::Position {
                plot = plot.scale(HistScale::Density).x_label("|r| (a₀), repeat N");
            } else {
                plot = plot.labels(&rep.labels).x_label(format!("{} (repeat N)", rep.obs.symbol()));
            }
            plot.show(&mut ui3);
        }
    }

    // ------------------------------------------------------------------ side panel pieces

    fn side_state(&mut self, ui: &mut egui::Ui) {
        let look = Look::of(ui);
        let weak = |s: String| RichText::new(s).color(look.weak_text()).size(12.0 * look.font_scale());
        match self.variant {
            Variant::Numbers | Variant::Radius => {
                if self.variant == Variant::Radius {
                    ui.horizontal(|ui| {
                        for q in [Nlm::new(1, 0, 0), Nlm::new(2, 1, 0)] {
                            if ui.selectable_label(self.sel == q, q.name()).clicked() && self.sel != q {
                                self.sel = q;
                                self.prepare();
                            }
                        }
                        ui.label(weak("or click a level".into()));
                    });
                }
                let mut q = self.sel;
                ui.horizontal(|ui| {
                    ui.label("n");
                    segmented(ui, |ui| {
                        for n in 1..=4u32 {
                            if ui.selectable_label(q.n == n, n.to_string()).clicked() {
                                q.n = n;
                            }
                        }
                    });
                });
                q.l = q.l.min(q.n - 1);
                ui.horizontal(|ui| {
                    ui.label("l");
                    segmented(ui, |ui| {
                        for l in 0..q.n {
                            if ui.selectable_label(q.l == l, format!("{l} ({})", crate::orbital::l_letter(l))).clicked() {
                                q.l = l;
                            }
                        }
                    });
                });
                q.m = q.m.clamp(-(q.l as i32), q.l as i32);
                ui.horizontal(|ui| {
                    ui.label("m");
                    segmented(ui, |ui| {
                        for m in -(q.l as i32)..=(q.l as i32) {
                            if ui.selectable_label(q.m == m, signed(m)).clicked() {
                                q.m = m;
                            }
                        }
                    });
                });
                if q != self.sel {
                    self.sel = q;
                    self.prepare();
                }
                if self.variant == Variant::Numbers {
                    self.ladder_ui(ui);
                    self.table_ui(ui);
                    egui::CollapsingHeader::new("Expectation values (44)")
                        .id_salt("hy_expect_43")
                        .default_open(false)
                        .show(ui, |ui| self.expectation_table(ui));
                } else {
                    self.expectation_table(ui);
                }
            }
            Variant::Ly => {
                ui.horizontal_wrapped(|ui| {
                    for (name, w, ph, tip) in [
                        ("(ψ₂₁₁ + ψ₂₁₋₁)/√2", 0.5, 0.0, "the l_y eigenstate of part (a): p_y"),
                        ("(ψ₂₁₋₁ − ψ₂₁₁)/√2", 0.5, PI, "p_x, an l_x eigenstate"),
                        ("ψ₂₁₁", 1.0, 0.0, "a single l_z eigenstate: torus around z"),
                    ] {
                        if ui
                            .selectable_label((self.ly_w - w).abs() < 1e-9 && (self.ly_phase - ph).abs() < 1e-9, name)
                            .on_hover_text(tip)
                            .clicked()
                        {
                            self.ly_w = w;
                            self.ly_phase = ph;
                            self.prepare();
                        }
                    }
                });
                let mut changed = ui.add(egui::Slider::new(&mut self.ly_w, 0.0..=1.0).text("|α|² (ψ₂₁₁)")).changed();
                let mut deg = self.ly_phase.to_degrees();
                if ui.add(egui::Slider::new(&mut deg, 0.0..=360.0).text("δ (°), β = |β|e^(iδ)")).changed() {
                    self.ly_phase = deg.to_radians();
                    changed = true;
                }
                if changed {
                    self.prepare();
                }
                ui.label(weak(format!(
                    "Ψ = αψ₂₁₁ + βψ₂₁₋₁, α = {:.3}, β = {:.3}·e^(i{:.0}°): the lobe axis is at φ = 90° + δ/2 = {:.0}°",
                    self.ly_w.sqrt(),
                    (1.0 - self.ly_w).sqrt(),
                    self.ly_phase.to_degrees(),
                    (90.0 + 0.5 * self.ly_phase.to_degrees()) % 180.0
                )));
            }
            Variant::Dipole => {
                ui.horizontal(|ui| {
                    ui.label("2p sublevel m");
                    segmented(ui, |ui| {
                        for m in [-1, 0, 1] {
                            if ui.selectable_label(self.dip_m == m, signed(m)).clicked() && self.dip_m != m {
                                self.dip_m = m;
                                self.prepare();
                            }
                        }
                    });
                });
                let mut changed = ui.add(egui::Slider::new(&mut self.dip_w, 0.0..=1.0).text("|c₂|² (2p weight)")).changed();
                let mut deg = self.dip_phase.to_degrees();
                if ui.add(egui::Slider::new(&mut deg, 0.0..=360.0).text("δ (°), c₂ = |c₂|e^(iδ)")).changed() {
                    self.dip_phase = deg.to_radians();
                    changed = true;
                }
                if changed {
                    self.prepare();
                }
                let z = 128.0 * 2f64.sqrt() / 243.0;
                let c1c2 = 2.0 * (self.dip_w * (1.0 - self.dip_w)).sqrt();
                let text = if self.dip_m == 0 {
                    format!("⟨z⟩ oscillates with amplitude 2|c₁c₂|·⟨1s|z|2p₀⟩ = {c1c2:.2}·{z:.3} a₀ = {:.3} a₀ (along z: π)", c1c2 * z)
                } else {
                    format!(
                        "⟨r⟩ turns about z on a circle of radius 2|c₁c₂|·{z:.3}/√2 a₀ = {:.3} a₀ ({}: σ)",
                        c1c2 * z * FRAC_1_SQRT_2,
                        if self.dip_m > 0 { "counter-clockwise seen from +z" } else { "clockwise seen from +z" }
                    )
                };
                ui.label(weak(format!(
                    "Ψ = c₁ψ₁₀₀ + c₂ψ₂₁{}: {text}; ω = (E₂ − E₁)/ħ, λ = 121.6 nm",
                    sub(self.dip_m as i64),
                )));
            }
            Variant::Zeeman => {
                if ui
                    .add(egui::Slider::new(&mut self.zee_b, 0.0..=zeeman::B_MAX).text("B (T) along z"))
                    .changed()
                {
                    self.state.b_au = self.zee_b / AU_FIELD_T;
                }
                ui.label(RichText::new("Transitions (m of l = 1 → m of l = 2)").color(look.text()));
                egui::Grid::new("hy_zee_grid").spacing(vec2(look.px(4.0), look.px(2.0))).show(ui, |ui| {
                    for dm in [1, 0, -1] {
                        ui.label(RichText::new(zeeman::pol_name(dm)).color(zeeman::pol_color(&look, dm)));
                        for m1 in -1..=1 {
                            let t = (m1, m1 + dm);
                            if ui
                                .selectable_label(self.zee_sel == t, format!("{}→{}", signed(t.0), signed(t.1)))
                                .clicked()
                                && self.zee_sel != t
                            {
                                self.zee_sel = t;
                                self.prepare();
                            }
                        }
                        ui.end_row();
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("spectrum seen");
                    segmented(ui, |ui| {
                        if ui.selectable_label(!self.zee_along, "⊥ B").clicked() {
                            self.zee_along = false;
                        }
                        if ui.selectable_label(self.zee_along, "∥ B").clicked() {
                            self.zee_along = true;
                        }
                    });
                });
                ui.label(weak(format!(
                    "3D: (ψ₂₁{} + ψ₃₂{})/√2; Δm = {} ({}). μ_B B = {:.1} µeV, Δν = {:.1} GHz",
                    sub(self.zee_sel.0 as i64),
                    sub(self.zee_sel.1 as i64),
                    signed(self.zee_sel.1 - self.zee_sel.0),
                    zeeman::pol_name(self.zee_sel.1 - self.zee_sel.0),
                    crate::orbital::MU_B_EV_PER_T * self.zee_b * 1e6,
                    crate::orbital::MU_B_GHZ_PER_T * self.zee_b
                )));
            }
        }
    }

    fn apply_ladder(&mut self, raise: bool, ctx: &mut Ctx) {
        let q = self.sel;
        let f = ladder_factor(q.l, q.m, raise);
        let op = if raise { "l₊" } else { "l₋" };
        let msg = if f < 1e-12 {
            format!("{op}{} = 0: the ladder ends at m = {} (state kept)", q.name(), signed(q.m))
        } else {
            let q2 = Nlm::new(q.n, q.l, q.m + if raise { 1 } else { -1 });
            let mut c = zero();
            c[q.index()] = c64(1.0, 0.0);
            debug_assert!((apply_ladder(&c, raise)[q2.index()].re - f).abs() < 1e-12);
            self.sel = q2;
            self.prepare();
            format!(
                "{op}{} = ħ√(l(l+1) − m(m{}1)) {} = ħ√({} − {}) {} = {:.3}ħ {}",
                q.name(),
                if raise { "+" } else { "−" },
                q2.name(),
                q.l * (q.l + 1),
                q.m * (q.m + if raise { 1 } else { -1 }),
                q2.name(),
                f,
                q2.name()
            )
        };
        self.ladder_msg = Some(msg.clone());
        ctx.set_status(msg);
    }

    fn ladder_ui(&mut self, ui: &mut egui::Ui) {
        let look = Look::of(ui);
        let q = self.sel;
        ui.add_space(look.px(4.0));
        ui.label(RichText::new("Ladder operators (43 b)").strong());
        let mut act = None;
        ui.horizontal(|ui| {
            let fp = ladder_factor(q.l, q.m, true);
            let fm = ladder_factor(q.l, q.m, false);
            if ui
                .button(format!("l₊  (×{fp:.3}ħ)"))
                .on_hover_text("l₊ψ_nlm = ħ√(l(l+1) − m(m+1)) ψ_nl,m+1")
                .clicked()
            {
                act = Some(true);
            }
            if ui
                .button(format!("l₋  (×{fm:.3}ħ)"))
                .on_hover_text("l₋ψ_nlm = ħ√(l(l+1) − m(m−1)) ψ_nl,m−1")
                .clicked()
            {
                act = Some(false);
            }
        });
        if let Some(raise) = act {
            self.pending_ladder = Some(raise);
        }
        if let Some(m) = &self.ladder_msg {
            ui.label(RichText::new(m).color(look.accent()).size(12.0 * look.font_scale()));
        }
    }

    fn table_ui(&mut self, ui: &mut egui::Ui) {
        let look = Look::of(ui);
        let n = self.sel.n;
        ui.add_space(look.px(4.0));
        ui.label(RichText::new(format!("Allowed l, m for n = {n} (43 a)")).strong());
        egui::Grid::new("hy_lm_table").striped(true).spacing(vec2(look.px(10.0), look.px(2.0))).show(ui, |ui| {
            ui.label(RichText::new("l").color(look.weak_text()));
            ui.label(RichText::new("m").color(look.weak_text()));
            ui.label(RichText::new("l²").color(look.weak_text()));
            ui.label(RichText::new("l_z").color(look.weak_text()));
            ui.end_row();
            for l in 0..n {
                let col = if l == self.sel.l { look.accent() } else { look.text() };
                ui.label(RichText::new(format!("{l} ({})", crate::orbital::l_letter(l))).color(col));
                let ms: Vec<String> = (-(l as i32)..=(l as i32)).map(signed).collect();
                ui.label(RichText::new(ms.join(", ")).color(col));
                ui.label(RichText::new(format!("{}ħ²", l * (l + 1))).color(col));
                ui.label(RichText::new(format!("{}ħ … {}ħ", signed(-(l as i32)), signed(l as i32))).color(col));
                ui.end_row();
            }
        });
        ui.label(
            RichText::new(format!("{} states, all with E{} = {:.2} eV", n * n, sub(n as i64), energy_ev(n)))
                .color(look.weak_text())
                .size(12.0 * look.font_scale()),
        );
    }

    fn expectation_table(&self, ui: &mut egui::Ui) {
        let look = Look::of(ui);
        let c = self.coeffs();
        ui.add_space(look.px(4.0));
        ui.label(RichText::new("Expectation values (numerical | closed form)").strong());
        let nl = self.state.single_nl();
        let cf = |f: &dyn Fn(u32, u32) -> f64| nl.map_or("—".to_string(), |(n, l)| format!("{:.4}", f(n, l)));
        let rows: Vec<(&str, f64, String)> = vec![
            ("⟨r⟩ (a₀)", expect_r(&c, 1), cf(&closed::r1)),
            ("⟨r²⟩ (a₀²)", expect_r(&c, 2), cf(&closed::r2)),
            ("⟨1/r⟩ (1/a₀)", expect_r(&c, -1), cf(&|n, _| closed::rinv(n))),
            ("⟨1/r²⟩ (1/a₀²)", expect_r(&c, -2), cf(&closed::rinv2)),
            ("⟨Ê⟩ = −½⟨1/r⟩ (eV)", -0.5 * expect_r(&c, -1) * HARTREE_EV, cf(&|n, _| energy_ev(n))),
            ("⟨x⟩ (a₀)", expect_xyz(&c)[0], if nl.is_some_and(|x| x.1 == 0) { "0".into() } else { "—".into() }),
            (
                "⟨x²⟩ (a₀²)",
                expect(&c, 2, AngOp::XX).re,
                if nl.is_some_and(|x| x.1 == 0) { cf(&|n, l| closed::r2(n, l) / 3.0) } else { "—".into() },
            ),
        ];
        egui::Grid::new("hy_expect").striped(true).spacing(vec2(look.px(10.0), look.px(2.0))).show(ui, |ui| {
            for (name, num, cl) in rows {
                ui.label(name);
                ui.label(RichText::new(fmt_num(num, 4)).monospace());
                ui.label(RichText::new(cl).monospace().color(look.prediction()));
                ui.end_row();
            }
        });
        ui.label(
            RichText::new("closed forms: ⟨r⟩ = (3n² − l(l+1))/2, ⟨r²⟩ = n²(5n²+1−3l(l+1))/2, ⟨1/r⟩ = 1/n², ⟨1/r²⟩ = 1/((l+½)n³); for l = 0: ⟨x²⟩ = ⟨r²⟩/3")
                .color(look.weak_text())
                .size(11.5 * look.font_scale()),
        );
    }

    fn side_view(&mut self, ui: &mut egui::Ui) {
        let look = Look::of(ui);
        ui.label(RichText::new("3D view").strong()).on_hover_text(
            "Brightness scale ρ_ref per state: the geometric mean of the largest |ψ|² and of the density above which 25 % of the probability lies, so that 1s and 4f, and the outer shells of ns states, are all visible (dense parts saturate). Fixed while the state evolves.",
        );
        match self.view.mode {
            Mode3D::Cloud => {
                ui.add(egui::Slider::new(&mut self.view.exposure, 0.1..=6.0).logarithmic(true).text("exposure"));
                ui.add(egui::Slider::new(&mut self.view.gamma, 0.2..=1.5).text("γ (opacity ∝ ρ^γ)"));
            }
            Mode3D::Iso => {
                ui.add(
                    egui::Slider::new(&mut self.view.iso, 0.002..=0.7)
                        .logarithmic(true)
                        .text("level (× ρ_ref)"),
                );
                ui.add(egui::Slider::new(&mut self.view.gamma, 0.2..=1.5).text("γ of the cut face"));
            }
            Mode3D::Points => {
                let mut n = self.n_points;
                if ui
                    .add(egui::Slider::new(&mut n, 500..=LIVE_POINTS_MAX).logarithmic(true).text("points (Born samples)"))
                    .changed()
                {
                    self.n_points = n;
                    self.points_version = None;
                }
                ui.checkbox(&mut self.view.faint_cloud, "faint cloud behind");
            }
        }
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.view.cap, "ψ on the cut face");
            ui.checkbox(&mut self.view.axes, "axes");
            if ui.button("default view").clicked() {
                self.view.cam = Default::default();
            }
        });
        ui.label(
            RichText::new("drag: rotate · scroll: zoom · cut open: removes the half towards you")
                .color(look.weak_text())
                .size(11.5 * look.font_scale()),
        );
    }

    fn side_measure(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx) {
        let look = Look::of(ui);
        ui.label(RichText::new("Measure").strong());
        let main = self.main_obs();
        let mut obs_list = vec![Obs::Energy, Obs::L2, Obs::Lz];
        if self.variant == Variant::Ly {
            obs_list.push(Obs::Ly);
        }
        obs_list.push(Obs::Position);
        ui.horizontal_wrapped(|ui| {
            for &o in &obs_list {
                let label = if o == main { format!("{}  (M)", o.symbol()) } else { o.symbol().to_string() };
                if ui.button(RichText::new(label).color(look.measure())).clicked() {
                    self.measure_obs(o, ctx);
                }
            }
        });
        ui.add(
            egui::Slider::new(&mut self.det_sigma, 0.1..=2.0)
                .logarithmic(true)
                .text("position detector σ (a₀)"),
        );
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("hy_rep_obs")
                .selected_text(self.repeat_obs.symbol())
                .width(look.px(80.0))
                .show_ui(ui, |ui| {
                    for &o in &obs_list {
                        ui.selectable_value(&mut self.repeat_obs, o, o.symbol());
                    }
                });
            let obs = self.repeat_obs;
            if let Some(n) = repeat_button(ui, "hy_rep", "Repeat", &mut self.repeat_n) {
                self.repeat(obs, n, ctx);
            }
        });
        if !self.dots.is_empty() && ui.button(format!("clear {} position dots", self.dots.len())).clicked() {
            self.dots.clear();
        }
        ui.label(
            RichText::new(
                "E, l², l_z, l_y: projection onto the eigenspace of the outcome (the state keeps evolving). Position: one dot, the state becomes a localized packet (not evolved further: reset R to continue; dots from earlier preparations stay). Repeat N: N fresh copies, compared with the Born rule.",
            )
            .color(look.weak_text())
            .size(11.5 * look.font_scale()),
        );
        if let Some(l) = self.flash.label() {
            ui.label(RichText::new(format!("last outcome: {l}")).color(look.measure()));
        }
    }
}

impl Scene for HydrogenScene {
    fn time(&self) -> f64 {
        self.t
    }

    fn time_scale(&self) -> f64 {
        // one display period ≈ 3 s (superpositions) or 6 s (global phase of a stationary state)
        let p = self.display_period();
        match self.variant {
            Variant::Dipole | Variant::Zeeman => p / 3.0,
            _ => p / 6.0,
        }
    }

    fn step(&mut self, dt: f64, ctx: &mut Ctx) {
        if self.localized.is_some() {
            ctx.playing = false;
            return;
        }
        self.t += dt;
        let period = self.display_period();
        if self.sz.last().is_none_or(|p| self.t - p[0] >= period / 60.0) {
            self.record();
        }
        // a time-dependent density: refresh a share of the Born samples
        if self.view.mode == Mode3D::Points && !self.state.stationary() && self.repeated.as_ref().is_none_or(|r| r.obs != Obs::Position) {
            self.ensure_points(ctx);
            let w = self.state.max_bohr_frequency();
            let frac = (dt * w / TAU * 25.0).min(1.0);
            let k = ((self.live_points.len() as f64) * frac).ceil() as usize;
            let c = self.coeffs();
            let act = active(&c);
            let n = self.live_points.len();
            if n > 0 {
                for _ in 0..k {
                    let i = ctx.rng.below(n);
                    if let Some(p) = born_sample(&c, &act, ctx.rng) {
                        self.live_points[i] = p;
                    }
                }
            }
        }
    }

    fn reset(&mut self, _ctx: &mut Ctx) {
        self.restart();
    }

    fn measure_label(&self) -> Option<String> {
        Some(match self.main_obs() {
            Obs::Position => "Measure position".into(),
            o => format!("Measure {}", o.symbol()),
        })
    }

    fn measure(&mut self, ctx: &mut Ctx) {
        let o = self.main_obs();
        self.measure_obs(o, ctx);
    }

    fn on_key(&mut self, key: egui::Key, _m: egui::Modifiers, _ctx: &mut Ctx) -> bool {
        if key == egui::Key::V {
            self.view.mode = self.view.mode.next();
            return true;
        }
        false
    }

    fn action(&mut self, action: &str, ctx: &mut Ctx) -> bool {
        let mut words = action.split_whitespace();
        let cmd = words.next().unwrap_or("");
        let args: Vec<&str> = words.collect();
        let num = |i: usize| args.get(i).and_then(|w| w.parse::<f64>().ok());
        match cmd {
            "mode_cloud" => self.view.mode = Mode3D::Cloud,
            "mode_iso" => self.view.mode = Mode3D::Iso,
            "mode_points" => self.view.mode = Mode3D::Points,
            "cut_on" => self.view.cut = true,
            "cut_off" => self.view.cut = false,
            "cap_off" => self.view.cap = false,
            "plane_xz" => self.view.plane = Plane::XZ,
            "plane_xy" => self.view.plane = Plane::XY,
            "offset" => self.view.offset = num(0).unwrap_or(0.0),
            "yaw" => self.view.cam.yaw = num(0).unwrap_or(-58.0).to_radians(),
            "pitch" => self.view.cam.pitch = num(0).unwrap_or(22.0).to_radians(),
            "zoom" => self.view.cam.zoom = num(0).unwrap_or(1.0),
            "exposure" => self.view.exposure = num(0).unwrap_or(1.2),
            "gamma" => self.view.gamma = num(0).unwrap_or(0.8),
            "iso" => self.view.iso = num(0).unwrap_or(0.06),
            "points" => {
                self.n_points = num(0).unwrap_or(4000.0) as usize;
                self.points_version = None;
            }
            "state" => {
                let (Some(n), Some(l), Some(m)) = (num(0), num(1), num(2)) else { return false };
                let q = Nlm::new(n as u32, l as u32, m as i32);
                if !q.is_valid() {
                    return false;
                }
                match self.variant {
                    Variant::Dipole if q.n == 2 && q.l == 1 => self.dip_m = q.m,
                    _ => self.sel = q,
                }
                self.prepare();
            }
            "lplus" => self.apply_ladder(true, ctx),
            "lminus" => self.apply_ladder(false, ctx),
            "ly_w" => {
                self.ly_w = num(0).unwrap_or(0.5).clamp(0.0, 1.0);
                self.prepare();
            }
            "ly_phase" => {
                self.ly_phase = num(0).unwrap_or(0.0).to_radians();
                self.prepare();
            }
            "dip_m" => {
                self.dip_m = num(0).unwrap_or(0.0) as i32;
                self.prepare();
            }
            "dip_w" => {
                self.dip_w = num(0).unwrap_or(0.5).clamp(0.0, 1.0);
                self.prepare();
            }
            "zeeman" => {
                let (Some(a), Some(b)) = (num(0), num(1)) else { return false };
                self.zee_sel = (a as i32, b as i32);
                self.prepare();
            }
            "B" => {
                self.zee_b = num(0).unwrap_or(1.0).clamp(0.0, zeeman::B_MAX);
                self.state.b_au = self.zee_b / AU_FIELD_T;
            }
            "view_par" => self.zee_along = true,
            "view_perp" => self.zee_along = false,
            _ => {
                if let Some(o) = cmd.strip_prefix("measure_").and_then(Obs::parse) {
                    self.measure_obs(o, ctx);
                } else if let Some(o) = cmd.strip_prefix("repeat_").and_then(Obs::parse) {
                    let n = num(0).map_or(self.repeat_n, |v| v as usize);
                    self.repeat_obs = o;
                    self.repeat(o, n, ctx);
                } else {
                    return false;
                }
            }
        }
        true
    }

    fn central_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx) {
        let look = Look::of(ui);
        if let Some(raise) = self.pending_ladder.take() {
            self.apply_ladder(raise, ctx);
        }
        if self.view.mode == Mode3D::Points {
            self.ensure_points(ctx);
        }
        self.view_bar(ui);
        self.readouts(ui);
        let avail = ui.available_rect_before_wrap();
        ui.allocate_rect(avail, Sense::hover());
        let gap = look.px(8.0);
        let bottom_h = match self.variant {
            Variant::Radius => (avail.height() * 0.36).max(look.px(200.0)),
            _ => (avail.height() * 0.27).max(look.px(170.0)),
        };
        let main_h = avail.height() - bottom_h - gap;
        let left_w = (avail.width() * 0.6).max(look.px(300.0));
        let r3d = Rect::from_min_size(avail.min, vec2(left_w, main_h));
        let right = Rect::from_min_max(pos2(r3d.right() + gap, avail.top()), pos2(avail.right(), r3d.bottom()));
        let bottom = Rect::from_min_max(pos2(avail.left(), r3d.bottom() + gap), avail.max);

        // 3D view
        let c = self.coeffs();
        let title = self.title();
        let (points, points_phase): (&[[f64; 3]], bool) = match &self.repeated {
            Some(r) if r.obs == Obs::Position && !r.points.is_empty() && r.version == self.version => (&r.points, false),
            _ => (&self.live_points, true),
        };
        let dipole = match self.variant {
            Variant::Dipole | Variant::Zeeman if self.localized.is_none() => {
                let d = expect_xyz(&c);
                Some((d, self.dipole_scale))
            }
            _ => None,
        };
        let input = RenderInput {
            coeffs: &c,
            packet: self.localized,
            radius: self.radius,
            rho_max: if self.localized.is_some() { 1.0 } else { self.rho_max },
            points,
            points_phase,
            dots: &self.dots,
            dipole,
            b_arrow: self.variant == Variant::Zeeman && self.zee_b > 0.0,
            title: &title,
        };
        self.view.show(ui, ctx, r3d, &input, &self.flash);

        self.right_column(ui, right);
        self.bottom_row(ui, bottom);
    }

    fn side_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx) {
        let look = ctx.look;
        ui.label(RichText::new("State").strong());
        self.side_state(ui);
        ui.add_space(look.px(8.0));
        self.side_view(ui);
        ui.add_space(look.px(8.0));
        self.side_measure(ui, ctx);
        if let Some(raise) = self.pending_ladder.take() {
            self.apply_ladder(raise, ctx);
        }
        ui.label(
            RichText::new("Atomic units: a₀ = 1, E_n = −1/(2n²) E_h = −13.6 eV/n², t in ħ/E_h = 24.2 as (slowed down).")
                .color(look.weak_text())
                .size(11.5 * look.font_scale()),
        );
    }

    fn notes(&self) -> String {
        match self.variant {
            Variant::Numbers => notes::NUMBERS,
            Variant::Radius => notes::RADIUS,
            Variant::Ly => notes::LY,
            Variant::Dipole => notes::DIPOLE,
            Variant::Zeeman => notes::ZEEMAN,
        }
        .to_string()
    }
}
