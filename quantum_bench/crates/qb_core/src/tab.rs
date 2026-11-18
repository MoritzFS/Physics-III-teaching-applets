//! What a tab implements: [`Tab`] (directly), or a list of [`Scene`]s wrapped
//! in a [`SceneTab`] (one scene per preset / exercise).

use eframe::egui;
use eframe::egui_wgpu::RenderState;

use crate::{Look, QbRng};

/// An entry of a tab's "Examples" menu.
#[derive(Clone, Debug, PartialEq)]
pub struct PresetInfo {
    /// Shown in the menu, e.g. "PS10 39 · Barrier tunnelling"
    /// (problem set, exercise number, middle dot, short title).
    pub name: String,
    /// start playing when the preset is loaded (default true)
    pub autoplay: bool,
    /// speed multiplier set when the preset is loaded (default 1)
    pub speed: f64,
}

impl PresetInfo {
    pub fn new(name: impl Into<String>) -> Self {
        PresetInfo {
            name: name.into(),
            autoplay: true,
            speed: 1.0,
        }
    }

    /// Builder: load paused.
    pub fn paused(mut self) -> Self {
        self.autoplay = false;
        self
    }

    /// Builder: initial speed multiplier.
    pub fn speed(mut self, speed: f64) -> Self {
        self.speed = speed;
        self
    }
}

/// What the shell hands to a tab on every call.
pub struct Ctx<'a> {
    /// the transport state; a tab may set it (e.g. pause when a sequence ends)
    pub playing: bool,
    /// speed multiplier of the transport (0.05 … 20); a tab may set it
    pub speed: f64,
    /// wall-clock seconds since the previous frame (0 in screenshot set-up)
    pub dt_wall: f64,
    /// this tab's seeded random number generator (`QB_SEED` in shot mode)
    pub rng: &'a mut QbRng,
    /// colours and presentation scaling (also available as `Look::of(ui)`)
    pub look: Look,
    /// wgpu device, queue, target format, egui renderer (None without wgpu)
    pub render_state: Option<&'a RenderState>,
    /// true while a screenshot is being prepared (`QB_SHOT`)
    pub shot: bool,
    pub(crate) status: Option<String>,
}

impl<'a> Ctx<'a> {
    pub(crate) fn new(rng: &'a mut QbRng, look: Look, render_state: Option<&'a RenderState>) -> Self {
        Ctx {
            playing: false,
            speed: 1.0,
            dt_wall: 0.0,
            rng,
            look,
            render_state,
            shot: look.shot,
            status: None,
        }
    }

    /// One line in the status bar at the bottom (kept until replaced or the preset changes).
    pub fn set_status(&mut self, text: impl Into<String>) {
        self.status = Some(text.into());
    }

    /// true in the browser build.
    pub fn is_web(&self) -> bool {
        cfg!(target_arch = "wasm32")
    }
}

/// One tab of the app ("1D waves", "2D box", "Hydrogen", "Spin").
///
/// The shell owns the transport (play/pause, speed, reset, presets) and calls
/// these methods; the tab owns its state and its simulation clock.
pub trait Tab {
    /// Stable lowercase id, used by `QB_TAB` and the URL hash: "wave1d", "box2d", "hydrogen", "spin".
    fn id(&self) -> &'static str;

    /// Name in the tab selector, e.g. "1D waves".
    fn name(&self) -> &str;

    /// The "Examples" menu, one entry per exercise. Keys 1–9 load the first nine.
    fn presets(&self) -> Vec<PresetInfo>;

    /// Loads preset `index` (0-based, < presets().len()) and restarts at t = 0.
    fn load_preset(&mut self, index: usize, ctx: &mut Ctx);

    /// Key R: restart the current scenario at t = 0, keeping the user's
    /// current parameter settings.
    fn reset(&mut self, ctx: &mut Ctx);

    /// Current simulation time (shown in the top bar, used by `QB_T`).
    fn time(&self) -> f64;

    /// Simulation time units per wall-clock second at speed 1×.
    fn time_scale(&self) -> f64 {
        1.0
    }

    /// Advances the simulation by `dt` time units (> 0). The shell passes
    /// `dt = dt_wall · speed · time_scale` while playing and exact steps in
    /// screenshot mode; sub-step internally if `dt` is too large for accuracy.
    fn step(&mut self, dt: f64, ctx: &mut Ctx);

    /// The step button / key "." (default: 1/30 s of wall time at the current speed).
    fn single_step(&mut self, ctx: &mut Ctx) {
        let dt = self.time_scale() * ctx.speed / 30.0;
        self.step(dt, ctx);
    }

    /// The main area (plots, views).
    fn central_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx);

    /// Controls in the right-hand panel (above the notes).
    fn side_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx);

    /// Notes for the current preset: what the exercise asks, what to try.
    /// Lines starting with "# " are headings, "- " bullets, blank lines separate paragraphs.
    fn notes(&self) -> String;

    /// Label of the top-bar measure button (key M), e.g. "Measure x"; `None` hides it.
    fn measure_label(&self) -> Option<String> {
        None
    }

    /// Key M / the measure button: one single-shot measurement (the tab decides what).
    fn measure(&mut self, _ctx: &mut Ctx) {}

    /// Any key the shell does not use itself. Return true if handled.
    fn on_key(&mut self, _key: egui::Key, _modifiers: egui::Modifiers, _ctx: &mut Ctx) -> bool {
        false
    }

    /// Scripted action (`QB_ACTION`, e.g. "measure_x", "view_both", "repeat_x 500").
    /// Return false if unknown.
    fn action(&mut self, _action: &str, _ctx: &mut Ctx) -> bool {
        false
    }

    /// Called once when the wgpu render state is available (create pipelines here).
    fn init_gpu(&mut self, _render_state: &RenderState) {}
}

/// One preset's content inside a [`SceneTab`]. Same methods as [`Tab`]
/// without the preset handling.
pub trait Scene {
    fn time(&self) -> f64;
    fn time_scale(&self) -> f64 {
        1.0
    }
    fn step(&mut self, dt: f64, ctx: &mut Ctx);
    fn single_step(&mut self, ctx: &mut Ctx) {
        let dt = self.time_scale() * ctx.speed / 30.0;
        self.step(dt, ctx);
    }
    fn reset(&mut self, ctx: &mut Ctx);
    fn central_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx);
    fn side_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx);
    fn notes(&self) -> String;
    fn measure_label(&self) -> Option<String> {
        None
    }
    fn measure(&mut self, _ctx: &mut Ctx) {}
    fn on_key(&mut self, _key: egui::Key, _modifiers: egui::Modifiers, _ctx: &mut Ctx) -> bool {
        false
    }
    fn action(&mut self, _action: &str, _ctx: &mut Ctx) -> bool {
        false
    }
}

type SceneMaker = Box<dyn Fn(&mut Ctx) -> Box<dyn Scene>>;

/// A [`Tab`] made of one [`Scene`] per preset:
///
/// ```ignore
/// SceneTab::new("wave1d", "1D waves")
///     .preset(PresetInfo::new("PS10 38 · Gaussian wave packet"), |_ctx| Box::new(GaussianScene::new()))
///     .preset(PresetInfo::new("PS10 39 · Barrier tunnelling"), |_ctx| Box::new(BarrierScene::new()))
/// ```
///
/// Loading a preset builds a fresh scene (so settings reset to the preset's).
pub struct SceneTab {
    id: &'static str,
    name: String,
    entries: Vec<(PresetInfo, SceneMaker)>,
    current: Option<Box<dyn Scene>>,
    index: usize,
}

impl SceneTab {
    pub fn new(id: &'static str, name: impl Into<String>) -> Self {
        SceneTab {
            id,
            name: name.into(),
            entries: vec![],
            current: None,
            index: 0,
        }
    }

    /// Builder: adds a preset and the function that builds its scene.
    pub fn preset(mut self, info: PresetInfo, make: impl Fn(&mut Ctx) -> Box<dyn Scene> + 'static) -> Self {
        self.entries.push((info, Box::new(make)));
        self
    }

    /// Index of the loaded preset.
    pub fn current_index(&self) -> usize {
        self.index
    }

    pub fn scene(&self) -> Option<&dyn Scene> {
        self.current.as_deref()
    }

    pub fn scene_mut(&mut self) -> Option<&mut (dyn Scene + 'static)> {
        self.current.as_deref_mut()
    }
}

impl Tab for SceneTab {
    fn id(&self) -> &'static str {
        self.id
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn presets(&self) -> Vec<PresetInfo> {
        self.entries.iter().map(|(p, _)| p.clone()).collect()
    }
    fn load_preset(&mut self, index: usize, ctx: &mut Ctx) {
        if let Some((_, make)) = self.entries.get(index) {
            self.current = Some(make(ctx));
            self.index = index;
        }
    }
    fn reset(&mut self, ctx: &mut Ctx) {
        if let Some(s) = &mut self.current {
            s.reset(ctx);
        }
    }
    fn time(&self) -> f64 {
        self.current.as_ref().map_or(0.0, |s| s.time())
    }
    fn time_scale(&self) -> f64 {
        self.current.as_ref().map_or(1.0, |s| s.time_scale())
    }
    fn step(&mut self, dt: f64, ctx: &mut Ctx) {
        if let Some(s) = &mut self.current {
            s.step(dt, ctx);
        }
    }
    fn single_step(&mut self, ctx: &mut Ctx) {
        if let Some(s) = &mut self.current {
            s.single_step(ctx);
        }
    }
    fn central_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx) {
        if let Some(s) = &mut self.current {
            s.central_ui(ui, ctx);
        }
    }
    fn side_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx) {
        if let Some(s) = &mut self.current {
            s.side_ui(ui, ctx);
        }
    }
    fn notes(&self) -> String {
        self.current.as_ref().map(|s| s.notes()).unwrap_or_default()
    }
    fn measure_label(&self) -> Option<String> {
        self.current.as_ref().and_then(|s| s.measure_label())
    }
    fn measure(&mut self, ctx: &mut Ctx) {
        if let Some(s) = &mut self.current {
            s.measure(ctx);
        }
    }
    fn on_key(&mut self, key: egui::Key, modifiers: egui::Modifiers, ctx: &mut Ctx) -> bool {
        self.current.as_mut().is_some_and(|s| s.on_key(key, modifiers, ctx))
    }
    fn action(&mut self, action: &str, ctx: &mut Ctx) -> bool {
        self.current.as_mut().is_some_and(|s| s.action(action, ctx))
    }
}

/// A scene that only shows a title and a sentence (for tabs under construction).
pub struct PlaceholderScene {
    pub title: String,
    pub text: String,
}

impl PlaceholderScene {
    pub fn new(title: impl Into<String>, text: impl Into<String>) -> Self {
        PlaceholderScene {
            title: title.into(),
            text: text.into(),
        }
    }
}

impl Scene for PlaceholderScene {
    fn time(&self) -> f64 {
        0.0
    }
    fn step(&mut self, _dt: f64, _ctx: &mut Ctx) {}
    fn reset(&mut self, _ctx: &mut Ctx) {}
    fn central_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx) {
        let look = ctx.look;
        ui.add_space(look.px(40.0));
        ui.vertical_centered(|ui| {
            ui.label(egui::RichText::new(&self.title).font(look.font(26.0)).color(look.text()));
            ui.add_space(look.px(8.0));
            ui.label(egui::RichText::new(&self.text).font(look.font(15.0)).color(look.weak_text()));
        });
    }
    fn side_ui(&mut self, ui: &mut egui::Ui, _ctx: &mut Ctx) {
        ui.weak("No controls yet.");
    }
    fn notes(&self) -> String {
        format!("# {}\n{}", self.title, self.text)
    }
}
