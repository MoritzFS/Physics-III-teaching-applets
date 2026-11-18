//! The app shell: top bar (tabs, examples, transport, presentation, theme),
//! side panel (tab controls + notes), status bar, keyboard shortcuts,
//! screenshot mode (desktop) and URL-hash deep links (web).

use eframe::egui::{self, Align, Key, Layout, Modifiers, RichText};
use eframe::egui_wgpu::RenderState;

use crate::{Ctx, Look, QbRng, Tab};

/// Window title and size for [`run_app_with`].
#[derive(Clone, Debug)]
pub struct AppConfig {
    pub title: String,
    /// initial window size in points (desktop)
    pub size: [f32; 2],
}

impl AppConfig {
    pub fn new(title: impl Into<String>) -> Self {
        AppConfig {
            title: title.into(),
            size: [1500.0, 950.0],
        }
    }
}

/// Runs the app with these tabs (desktop window or browser canvas).
///
/// With one tab (the `*_solo` binaries) the window is titled "Quantum Bench · <name>".
pub fn run_app(tabs: Vec<Box<dyn Tab>>) {
    let title = match tabs.as_slice() {
        [one] => format!("Quantum Bench · {}", one.name()),
        _ => "Quantum Bench".to_string(),
    };
    run_app_with(AppConfig::new(title), tabs);
}

/// [`run_app`] with an explicit title / window size.
pub fn run_app_with(config: AppConfig, tabs: Vec<Box<dyn Tab>>) {
    assert!(!tabs.is_empty(), "run_app needs at least one tab");
    #[cfg(not(target_arch = "wasm32"))]
    native::run(config, tabs);
    #[cfg(target_arch = "wasm32")]
    {
        let title = config.title.clone();
        crate::web::start(title, Box::new(move |cc| Ok(Box::new(Shell::new(cc, config, tabs)))));
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;
    use crate::shot::ShotPlan;

    pub(super) fn run(config: AppConfig, tabs: Vec<Box<dyn Tab>>) {
        let _ = env_logger::try_init();
        // a double-clicked app has no terminal: keep panic messages in ~/Library/Logs/Quantum Bench.log
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if let Some(home) = std::env::var_os("HOME") {
                let path = std::path::Path::new(&home).join("Library/Logs/Quantum Bench.log");
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                    use std::io::Write;
                    let bt = std::backtrace::Backtrace::force_capture();
                    let _ = writeln!(f, "{:?}: {info}\n{bt}\n", std::time::SystemTime::now());
                }
            }
            default_hook(info);
        }));
        let shot = ShotPlan::from_env();
        let size = shot.as_ref().map_or(config.size, |s| s.size);
        let mut viewport = egui::ViewportBuilder::default().with_title(&config.title).with_inner_size(size);
        if shot.is_none() {
            viewport = viewport.with_min_inner_size([960.0, 640.0]);
        }
        let mut options = eframe::NativeOptions {
            viewport,
            renderer: eframe::Renderer::Wgpu,
            ..Default::default()
        };
        if shot.is_some() {
            // screenshots must not read or overwrite the user's saved state
            options.persistence_path = Some(std::env::temp_dir().join("quantum_bench_shot_state"));
            options.persist_window = false;
        }
        let result = eframe::run_native(
            "Quantum Bench",
            options,
            Box::new(move |cc| Ok(Box::new(Shell::new(cc, config, tabs, shot)))),
        );
        if let Err(e) = result {
            eprintln!("Quantum Bench could not start: {e}");
            std::process::exit(1);
        }
    }
}

/// Key of the shell state in eframe's storage. In the browser that is the
/// local storage of the whole site, so use a name no other applet uses.
#[cfg(not(target_arch = "wasm32"))]
const STATE_KEY: &str = eframe::APP_KEY;
#[cfg(target_arch = "wasm32")]
const STATE_KEY: &str = "quantum_bench/state";

#[derive(serde::Serialize, serde::Deserialize)]
struct Persist {
    dark: bool,
    present: bool,
    tab: String,
}

struct Slot {
    tab: Box<dyn Tab>,
    rng: QbRng,
    playing: bool,
    speed: f64,
    preset: usize,
    status: Option<String>,
}

enum Cmd {
    SelectTab(usize),
    Preset(usize),
    TogglePlay,
    Step,
    Reset,
    Measure,
    TogglePresent,
    ToggleDark,
    Key(Key, Modifiers),
}

pub(crate) struct Shell {
    slots: Vec<Slot>,
    active: usize,
    look: Look,
    applied: Option<Look>,
    gpu_ready: bool,
    #[cfg(not(target_arch = "wasm32"))]
    shot: Option<crate::shot::ShotRun>,
    #[cfg(target_arch = "wasm32")]
    hash: String,
}

impl Shell {
    fn new(
        cc: &eframe::CreationContext<'_>,
        _config: AppConfig,
        tabs: Vec<Box<dyn Tab>>,
        #[cfg(not(target_arch = "wasm32"))] shot: Option<crate::shot::ShotPlan>,
    ) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let (in_shot, seed) = (
            shot.is_some(),
            shot.as_ref().map_or_else(|| QbRng::from_time().next_u64(), |s| s.seed),
        );
        #[cfg(target_arch = "wasm32")]
        let (in_shot, seed) = (false, QbRng::from_time().next_u64());
        let slots = tabs
            .into_iter()
            .enumerate()
            .map(|(i, tab)| Slot {
                tab,
                rng: QbRng::seed(seed.wrapping_add((i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15))),
                playing: false,
                speed: 1.0,
                preset: 0,
                status: None,
            })
            .collect();
        let mut shell = Shell {
            slots,
            active: 0,
            look: Look::default(),
            applied: None,
            gpu_ready: false,
            #[cfg(not(target_arch = "wasm32"))]
            shot: shot.map(crate::shot::ShotRun::new),
            #[cfg(target_arch = "wasm32")]
            hash: String::new(),
        };
        if !in_shot && let Some(p) = cc.storage.and_then(|s| eframe::get_value::<Persist>(s, STATE_KEY)) {
            shell.look.dark = p.dark;
            shell.look.present = p.present;
            if let Some(i) = shell.slots.iter().position(|s| s.tab.id() == p.tab) {
                shell.active = i;
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(run) = &shell.shot {
            shell.look = Look {
                dark: !run.plan.light,
                present: run.plan.present,
                shot: true,
            };
            crate::clock::use_virtual();
        }
        crate::fonts::install(&cc.egui_ctx);
        let rs = cc.wgpu_render_state.as_ref();
        if let Some(rs) = rs {
            for s in &mut shell.slots {
                s.tab.init_gpu(rs);
            }
            shell.gpu_ready = true;
        }
        for i in 0..shell.slots.len() {
            shell.load_preset(i, 0, rs);
        }
        #[cfg(target_arch = "wasm32")]
        {
            let hash = crate::web::location_hash();
            shell.apply_hash(&hash, rs);
            shell.sync_hash();
        }
        shell
    }

    /// Runs `f` with the tab `i` and a fresh [`Ctx`]; writes back transport changes.
    fn with_tab<R>(&mut self, i: usize, rs: Option<&RenderState>, dt_wall: f64, f: impl FnOnce(&mut dyn Tab, &mut Ctx) -> R) -> R {
        let look = self.look;
        let slot = &mut self.slots[i];
        let mut ctx = Ctx::new(&mut slot.rng, look, rs);
        ctx.playing = slot.playing;
        ctx.speed = slot.speed;
        ctx.dt_wall = dt_wall;
        let r = f(slot.tab.as_mut(), &mut ctx);
        slot.playing = ctx.playing;
        slot.speed = ctx.speed.clamp(0.01, 100.0);
        if let Some(s) = ctx.status.take() {
            slot.status = Some(s);
        }
        r
    }

    fn load_preset(&mut self, i: usize, preset: usize, rs: Option<&RenderState>) {
        let Some(info) = self.slots[i].tab.presets().get(preset).cloned() else {
            return;
        };
        self.slots[i].preset = preset;
        self.slots[i].status = None;
        self.with_tab(i, rs, 0.0, |tab, ctx| {
            ctx.playing = info.autoplay;
            ctx.speed = info.speed;
            tab.load_preset(preset, ctx);
        });
    }

    fn exec(&mut self, cmd: Cmd, rs: Option<&RenderState>) {
        let a = self.active;
        match cmd {
            Cmd::SelectTab(i) => self.active = i.min(self.slots.len() - 1),
            Cmd::Preset(p) => self.load_preset(a, p, rs),
            Cmd::TogglePlay => self.slots[a].playing = !self.slots[a].playing,
            Cmd::Step => {
                self.slots[a].playing = false;
                self.with_tab(a, rs, 0.0, |tab, ctx| tab.single_step(ctx));
            }
            Cmd::Reset => self.with_tab(a, rs, 0.0, |tab, ctx| tab.reset(ctx)),
            Cmd::Measure => self.with_tab(a, rs, 0.0, |tab, ctx| tab.measure(ctx)),
            Cmd::TogglePresent => self.look.present = !self.look.present,
            Cmd::ToggleDark => self.look.dark = !self.look.dark,
            Cmd::Key(k, m) => {
                self.with_tab(a, rs, 0.0, |tab, ctx| tab.on_key(k, m, ctx));
            }
        }
    }

    /// Shell shortcuts (consumed) and other key presses (passed to the tab).
    fn keys(&self, ctx: &egui::Context, cmds: &mut Vec<Cmd>) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        ctx.input_mut(|i| {
            let none = Modifiers::NONE;
            if i.consume_key(none, Key::Space) {
                cmds.push(Cmd::TogglePlay);
            }
            if i.consume_key(none, Key::R) {
                cmds.push(Cmd::Reset);
            }
            if i.consume_key(none, Key::M) {
                cmds.push(Cmd::Measure);
            }
            if i.consume_key(none, Key::P) {
                cmds.push(Cmd::TogglePresent);
            }
            if i.consume_key(none, Key::Period) {
                cmds.push(Cmd::Step);
            }
            const NUMS: [Key; 9] = [
                Key::Num1,
                Key::Num2,
                Key::Num3,
                Key::Num4,
                Key::Num5,
                Key::Num6,
                Key::Num7,
                Key::Num8,
                Key::Num9,
            ];
            for (n, k) in NUMS.iter().enumerate() {
                if i.consume_key(none, *k) {
                    cmds.push(Cmd::Preset(n));
                }
            }
            for e in &i.events {
                if let egui::Event::Key {
                    key,
                    pressed: true,
                    repeat: false,
                    modifiers,
                    ..
                } = e
                    && !modifiers.command
                    && !modifiers.ctrl
                {
                    cmds.push(Cmd::Key(*key, *modifiers));
                }
            }
        });
    }

    fn apply_look(&mut self, ctx: &egui::Context) {
        self.look.install(ctx);
        if self.applied != Some(self.look) {
            let style = self.look.egui_style();
            let theme = if self.look.dark { egui::Theme::Dark } else { egui::Theme::Light };
            ctx.set_style_of(theme, style);
            // (the web backend does not implement the window-theme command that set_theme sends)
            #[cfg(not(target_arch = "wasm32"))]
            ctx.set_theme(theme);
            #[cfg(target_arch = "wasm32")]
            ctx.options_mut(|o| o.theme_preference = theme.into());
            self.applied = Some(self.look);
        }
    }

    fn top_bar(&mut self, ui: &mut egui::Ui, cmds: &mut Vec<Cmd>) {
        let look = self.look;
        let a = self.active;
        let presets = self.slots[a].tab.presets();
        let current = self.slots[a].preset;
        let playing = self.slots[a].playing;
        let time = self.slots[a].tab.time();
        let measure = self.slots[a].tab.measure_label();
        // right: Present / theme / Help; left: everything else, wrapping onto a
        // second row when the window is narrow (e.g. presentation mode on a projector)
        ui.add_space(2.0);
        // (inside `horizontal`, so that the layout starts one row high and only grows when it wraps)
        ui.horizontal(|ui| {
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.menu_button("Help", help_ui);
                let theme = if look.dark { "Light" } else { "Dark" };
                if ui.button(theme).on_hover_text("light / dark").clicked() {
                    cmds.push(Cmd::ToggleDark);
                }
                if ui
                    .selectable_label(look.present, "Present")
                    .on_hover_text("presentation mode (P)")
                    .clicked()
                {
                    cmds.push(Cmd::TogglePresent);
                }
                ui.separator();
                ui.with_layout(Layout::left_to_right(Align::Center).with_main_wrap(true), |ui| {
                    ui.label(RichText::new("Quantum Bench").strong());
                    if self.slots.len() > 1 {
                        ui.separator();
                        for (i, s) in self.slots.iter().enumerate() {
                            if ui.selectable_label(i == a, s.tab.name()).clicked() {
                                cmds.push(Cmd::SelectTab(i));
                            }
                        }
                    }
                    ui.separator();
                    ui.menu_button("Examples", |ui| {
                        ui.set_min_width(look.px(320.0));
                        for (k, p) in presets.iter().enumerate() {
                            let key = if k < 9 { format!("{}   ", k + 1) } else { "     ".into() };
                            if ui.selectable_label(k == current, format!("{key}{}", p.name)).clicked() {
                                cmds.push(Cmd::Preset(k));
                                ui.close();
                            }
                        }
                    });
                    if let Some(p) = presets.get(current) {
                        ui.label(RichText::new(&p.name).color(look.accent()));
                    }
                    ui.separator();
                    let play = if playing { "⏸ Pause" } else { "⏵ Play" };
                    if ui.button(play).on_hover_text("Space").clicked() {
                        cmds.push(Cmd::TogglePlay);
                    }
                    if ui.button("⏭").on_hover_text("one step (.)").clicked() {
                        cmds.push(Cmd::Step);
                    }
                    if ui.button("⟲ Reset").on_hover_text("restart at t = 0 (R)").clicked() {
                        cmds.push(Cmd::Reset);
                    }
                    ui.spacing_mut().slider_width = look.px(90.0);
                    let mut speed = self.slots[a].speed;
                    let r = ui.add(
                        egui::Slider::new(&mut speed, 0.05..=20.0)
                            .logarithmic(true)
                            .max_decimals(2)
                            .text("speed"),
                    );
                    if r.changed() {
                        self.slots[a].speed = speed;
                    }
                    if r.double_clicked() {
                        self.slots[a].speed = 1.0;
                    }
                    ui.label(RichText::new(format!("t = {time:.2}")).monospace());
                    if let Some(m) = measure {
                        ui.separator();
                        let b = egui::Button::new(RichText::new(format!("{m}  (M)")).color(look.measure()).strong());
                        if ui.add(b).clicked() {
                            cmds.push(Cmd::Measure);
                        }
                    }
                });
            })
        });
        ui.add_space(2.0);
    }

    fn status_bar(&self, ui: &mut egui::Ui) {
        let slot = &self.slots[self.active];
        ui.horizontal(|ui| {
            if let Some(s) = &slot.status {
                ui.label(s);
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.weak("Space play/pause · . step · R reset · M measure · 1–9 examples · P present");
            });
        });
    }

    /// Applies a deep link "id" or "id/preset".
    #[cfg(target_arch = "wasm32")]
    fn apply_hash(&mut self, hash: &str, rs: Option<&RenderState>) {
        let mut parts = hash.split('/');
        let id = parts.next().unwrap_or("").trim().to_lowercase();
        if id.is_empty() {
            return;
        }
        let Some(i) = self.slots.iter().position(|s| s.tab.id() == id) else {
            return;
        };
        self.active = i;
        if let Some(p) = parts.next().and_then(|p| p.trim().parse::<usize>().ok())
            && p != self.slots[i].preset
        {
            self.load_preset(i, p, rs);
        }
    }

    /// Writes the current tab and preset into the URL ("#wave1d/0").
    #[cfg(target_arch = "wasm32")]
    fn sync_hash(&mut self) {
        let s = &self.slots[self.active];
        let hash = format!("{}/{}", s.tab.id(), s.preset);
        if hash != self.hash {
            crate::web::set_location_hash(&hash);
            self.hash = hash;
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn taking_shot(&self) -> bool {
        self.shot.is_some()
    }

    #[cfg(target_arch = "wasm32")]
    fn taking_shot(&self) -> bool {
        false
    }

    /// Screenshot mode: set up on the first frame, shoot once the layout has settled.
    #[cfg(not(target_arch = "wasm32"))]
    fn shot_frame(&mut self, ctx: &egui::Context, rs: Option<&RenderState>) {
        let Some(run) = &mut self.shot else { return };
        ctx.request_repaint();
        if !run.set_up {
            run.set_up = true;
            let plan = run.plan.clone();
            ctx.send_viewport_cmd(egui::ViewportCommand::WindowLevel(egui::WindowLevel::AlwaysOnTop));
            if let Some(name) = &plan.tab {
                let name = name.to_lowercase();
                let found = name.parse::<usize>().ok().filter(|&i| i < self.slots.len()).or_else(|| {
                    self.slots
                        .iter()
                        .position(|s| s.tab.id() == name || s.tab.name().to_lowercase().starts_with(&name))
                });
                match found {
                    Some(i) => self.active = i,
                    None => eprintln!("QB_TAB={name}: no such tab"),
                }
            }
            let a = self.active;
            self.load_preset(a, plan.preset, rs);
            self.slots[a].playing = false;
            let mut events: Vec<(f64, String)> = plan
                .actions
                .iter()
                .map(|(t, s)| (t.or(plan.t).unwrap_or(f64::NEG_INFINITY), s.clone()))
                .collect();
            events.sort_by(|x, y| x.0.total_cmp(&y.0));
            for (t, action) in events {
                self.advance_to(a, t, rs);
                let ok = self.with_tab(a, rs, 0.0, |tab, c| tab.action(&action, c));
                if !ok {
                    eprintln!("QB_ACTION: tab '{}' does not know '{action}'", self.slots[a].tab.id());
                }
            }
            if let Some(t) = plan.t {
                self.advance_to(a, t, rs);
            }
            // a flash triggered right at QB_T is shown a quarter second into its animation
            crate::clock::advance_virtual(0.25);
            self.slots[a].playing = false;
            if let Some(run) = &mut self.shot {
                run.ready_at = Some(crate::Instant::now());
            }
            return;
        }
        run.frames += 1;
        let waited = run.ready_at.map_or(0.0, |t| t.elapsed().as_secs_f64());
        if !run.requested && run.frames >= 6 && waited >= run.plan.wait {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
            run.requested = true;
        }
        let image = ctx.input(|i| {
            i.raw.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        if let Some(img) = image {
            let path = run.plan.path.clone();
            match crate::shot::save_png(&img, &path) {
                Ok(()) => eprintln!(
                    "screenshot {} ({}×{} px), tab {}, t = {:.3}",
                    path.display(),
                    img.size[0],
                    img.size[1],
                    self.slots[self.active].tab.id(),
                    self.slots[self.active].tab.time()
                ),
                Err(e) => eprintln!("could not write {}: {e}", path.display()),
            }
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }

    /// Steps tab `i` in fixed chunks until its time reaches `t` (deterministic).
    #[cfg(not(target_arch = "wasm32"))]
    fn advance_to(&mut self, i: usize, t: f64, rs: Option<&RenderState>) {
        if !t.is_finite() {
            return;
        }
        let chunk = self.slots[i].tab.time_scale() / 60.0;
        for _ in 0..10_000_000 {
            let now = self.slots[i].tab.time();
            let left = t - now;
            if left <= 1e-12 {
                break;
            }
            let dt = left.min(chunk.max(1e-9));
            self.with_tab(i, rs, 0.0, |tab, c| tab.step(dt, c));
            // animations see the time a viewer at speed 1 would have seen
            crate::clock::advance_virtual(dt / self.slots[i].tab.time_scale().max(1e-12));
            if self.slots[i].tab.time() <= now {
                break; // this tab's clock does not move
            }
        }
    }
}

fn help_ui(ui: &mut egui::Ui) {
    ui.set_max_width(420.0);
    ui.label(RichText::new("Keys").strong());
    for (k, what) in [
        ("Space", "play / pause"),
        (".", "one step"),
        ("R", "restart the example at t = 0"),
        ("M", "measure (what is measured depends on the tab)"),
        ("1 … 9", "load an example"),
        ("P", "presentation mode (larger text and lines)"),
    ] {
        ui.horizontal(|ui| {
            ui.label(RichText::new(k).monospace().strong());
            ui.label(what);
        });
    }
    ui.separator();
    ui.label("Units: ħ = m = 1. Colour of a wavefunction = its phase arg ψ (see the colour wheel), height or brightness = |ψ|².");
    ui.label("Measurements: a single shot collapses the state, which then keeps evolving. \"Repeat N\" measures N fresh copies of the current state and compares the counts with the Born rule.");
}

/// Renders tab notes: "# " headings, "- " bullets, blank lines between paragraphs.
pub fn notes_ui(ui: &mut egui::Ui, text: &str) {
    let look = Look::of(ui);
    let mut para = String::new();
    let flush = |ui: &mut egui::Ui, para: &mut String| {
        if !para.is_empty() {
            ui.label(RichText::new(para.trim()).color(look.text()));
            para.clear();
        }
    };
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() {
            flush(ui, &mut para);
            ui.add_space(look.px(4.0));
        } else if let Some(h) = l.strip_prefix("# ") {
            flush(ui, &mut para);
            ui.add_space(look.px(2.0));
            ui.label(RichText::new(h).strong().color(look.text()).size(15.0 * look.font_scale()));
        } else if let Some(b) = l.strip_prefix("- ").or_else(|| l.strip_prefix("• ")) {
            flush(ui, &mut para);
            ui.horizontal_top(|ui| {
                ui.label(RichText::new("•").color(look.accent()));
                ui.add(egui::Label::new(RichText::new(b).color(look.text())).wrap());
            });
        } else {
            if !para.is_empty() {
                para.push(' ');
            }
            para.push_str(l);
        }
    }
    flush(ui, &mut para);
}

impl eframe::App for Shell {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let rs = frame.wgpu_render_state();
        if !self.gpu_ready
            && let Some(rs) = rs
        {
            for s in &mut self.slots {
                s.tab.init_gpu(rs);
            }
            self.gpu_ready = true;
        }
        self.apply_look(&ctx);
        #[cfg(target_arch = "wasm32")]
        {
            let hash = crate::web::location_hash();
            if hash != self.hash {
                self.apply_hash(&hash, rs);
                self.hash = hash;
                self.sync_hash();
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.shot_frame(&ctx, rs);

        let mut cmds = vec![];
        if !self.taking_shot() {
            self.keys(&ctx, &mut cmds);
        }
        for c in cmds.drain(..) {
            self.exec(c, rs);
        }

        // simulation
        let a = self.active;
        let dt_wall = (ctx.input(|i| i.stable_dt) as f64).clamp(0.0, 0.1);
        if self.slots[a].playing && !self.taking_shot() {
            let dt = dt_wall * self.slots[a].speed * self.slots[a].tab.time_scale();
            if dt > 0.0 {
                self.with_tab(a, rs, dt_wall, |tab, c| tab.step(dt, c));
            }
            ctx.request_repaint();
        }

        egui::Panel::top("qb_top").show(ui, |ui| self.top_bar(ui, &mut cmds));
        for c in cmds.drain(..) {
            self.exec(c, rs);
        }
        self.apply_look(&ctx);
        #[cfg(target_arch = "wasm32")]
        self.sync_hash();

        egui::Panel::bottom("qb_status").show(ui, |ui| self.status_bar(ui));

        let look = self.look;
        let a = self.active;
        let width = ctx.content_rect().width();
        let (side_id, side_w) = if look.present {
            ("qb_side_present", 430.0)
        } else {
            ("qb_side", 340.0)
        };
        let preset_name = self.slots[a]
            .tab
            .presets()
            .get(self.slots[a].preset)
            .map(|p| p.name.clone())
            .unwrap_or_default();
        self.with_tab(a, rs, dt_wall, |tab, c| {
            egui::Panel::right(side_id)
                .resizable(true)
                .default_size(side_w)
                .min_size(240.0)
                .max_size((width * 0.5).max(260.0))
                .show(ui, |ui| {
                    egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
                        ui.add_space(4.0);
                        ui.label(RichText::new(&preset_name).strong().color(look.text()));
                        ui.separator();
                        tab.side_ui(ui, c);
                        ui.add_space(look.px(10.0));
                        egui::CollapsingHeader::new(RichText::new("Notes").strong())
                            .id_salt("qb_notes")
                            .default_open(true)
                            .show(ui, |ui| notes_ui(ui, &tab.notes()));
                    });
                });
            egui::CentralPanel::default().show(ui, |ui| tab.central_ui(ui, c));
        });
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if self.taking_shot() {
            return;
        }
        let p = Persist {
            dark: self.look.dark,
            present: self.look.present,
            tab: self.slots[self.active].tab.id().into(),
        };
        eframe::set_value(storage, STATE_KEY, &p);
    }

    /// In the browser eframe only saves periodically (not reliably on close).
    fn auto_save_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(if cfg!(target_arch = "wasm32") { 5 } else { 30 })
    }

    /// In the browser all applets of the site share one local storage: keep egui's memory out of it.
    fn persist_egui_memory(&self) -> bool {
        !cfg!(target_arch = "wasm32")
    }
}
