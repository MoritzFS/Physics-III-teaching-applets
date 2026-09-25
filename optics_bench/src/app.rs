//! The egui application: eye + screen views on top, the bench in the middle,
//! the toolbox (and inspector) at the bottom.

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, StrokeKind};
use eframe::egui_wgpu;
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::configs::{self, ConfigFile, SavedView};
use crate::fourier::{FourierParams, FourierPreset};
use crate::fourier_ui::{FourierUi, Mode};
use crate::gpu::{self, FrameData, Globals, Gpu, ViewUniform, T_EYE, T_OVERVIEW, T_SCREEN};
use crate::scene::{
    build_world, default_aperture, default_eye, default_lens, default_prism, default_screen, facing,
    min_deviation, prism_index, right_of, Element, Kind, ObjKind, Preset, Scene, StopParams, World, LAMBDA_C, LAMBDA_D, LAMBDA_F,
    RETINA, VIS_OVERVIEW, VIS_PHYSICAL,
};
use crate::trace::{focus_along, Event, Surf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tool {
    Stickman,
    Tree,
    Sign,
    SlitLamp,
    Checker,
    Converging,
    Diverging,
    Aperture,
    Stop,
    Prism,
    Screen,
    Eye,
}

impl Tool {
    const ALL: [Tool; 12] = [
        Tool::Stickman,
        Tool::Tree,
        Tool::Sign,
        Tool::SlitLamp,
        Tool::Checker,
        Tool::Converging,
        Tool::Diverging,
        Tool::Aperture,
        Tool::Stop,
        Tool::Prism,
        Tool::Screen,
        Tool::Eye,
    ];
    fn label(self) -> &'static str {
        match self {
            Tool::Stickman => "Stickman",
            Tool::Tree => "Tree",
            Tool::Sign => "F-sign",
            Tool::SlitLamp => "Slit lamp",
            Tool::Checker => "Checkerboard",
            Tool::Converging => "Converging lens",
            Tool::Diverging => "Diverging lens",
            Tool::Aperture => "Aperture",
            Tool::Stop => "Central stop",
            Tool::Prism => "Prism",
            Tool::Screen => "Screen",
            Tool::Eye => "Eye",
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// render pixels per UI point
    pub render_scale: f32,
    pub spp: u32,
    pub max_samples: u32,
    pub show_rays: bool,
    pub show_focal: bool,
    pub show_virtual: bool,
    pub show_sight: bool,
    pub retinal: bool,
    pub ruler: bool,
    /// show the screen as seen from behind (translucent screen / ground glass)
    pub screen_behind: bool,
    /// only light that passed all optics in front of the screen reaches it
    pub screen_tube: bool,
    pub bench_3d: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            render_scale: 1.0,
            spp: 2,
            max_samples: 1024,
            show_rays: true,
            show_focal: true,
            show_virtual: false,
            show_sight: true,
            retinal: false,
            ruler: true,
            screen_behind: false,
            screen_tube: true,
            bench_3d: false,
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
struct TopCam {
    center: Vec2,
    /// pixels per metre
    zoom: f32,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
struct Orbit {
    target: Vec3,
    yaw: f32,
    pitch: f32,
    dist: f32,
}

impl Orbit {
    fn frame(&self) -> (Vec3, Vec3, Vec3, Vec3) {
        let dir = Vec3::new(self.pitch.cos() * self.yaw.cos(), self.pitch.sin(), self.pitch.cos() * self.yaw.sin());
        let pos = self.target + dir * self.dist;
        let f = -dir;
        let r = f.cross(Vec3::Y).normalize();
        let u = r.cross(f);
        (pos, r, u, f)
    }
}

/// Field of view of the 3D overview camera: 48° vertically, at most 84° horizontally.
fn overview_tan(w: f32, h: f32) -> (f32, f32) {
    let aspect = (w / h.max(1.0)).max(0.1);
    let tx = (0.445 * aspect).min(0.9);
    (tx, tx / aspect)
}

#[derive(Serialize, Deserialize)]
struct Persist {
    scene: Scene,
    settings: Settings,
    top: TopCam,
    orbit: Orbit,
    #[serde(default)]
    mode: Mode,
    #[serde(default)]
    fourier: Option<(FourierParams, String)>,
}

enum Drag {
    Move { id: u32, offset: Vec2 },
    Rotate { id: u32 },
    Pan,
    Orbit,
    Move3d { id: u32, offset: Vec2, plane_y: f32 },
    Look,
}

pub struct OpticsApp {
    /// ray-optics bench or 4f Fourier bench
    mode: Mode,
    four: FourierUi,
    scene: Scene,
    settings: Settings,
    selected: Option<u32>,
    top: TopCam,
    orbit: Orbit,
    gpu: Option<Gpu>,
    drag: Option<Drag>,
    world: World,
    eye_status: (String, bool),
    screen_status: String,
    sight_path: Vec<Vec3>,
    /// bit mask of the lenses (and apertures) on the screen's axis
    screen_train: u32,
    /// how the screen samples the light, cached by a hash of the scene
    entrance: (u64, Option<Entrance>),
    /// debug hook: OPTICS_SHOT=file.png saves a screenshot after some frames and quits
    shot: Option<Shot>,
    /// fit the top view to the scene as soon as the bench size is known
    pending_fit: bool,
    /// the save / load window
    configs: ConfigWindow,
}

#[derive(Default)]
struct ConfigWindow {
    open: bool,
    name: String,
    entries: Vec<configs::Entry>,
    confirm_delete: Option<std::path::PathBuf>,
    message: Option<(String, bool)>,
    focus_name: bool,
}

struct Shot {
    path: String,
    secs: f32,
    count: u32,
    start: std::time::Instant,
}

fn default_top() -> TopCam {
    TopCam { center: Vec2::new(2.5, 0.0), zoom: 90.0 }
}

fn default_orbit() -> Orbit {
    Orbit { target: Vec3::new(2.5, 0.8, 0.0), yaw: 1.9, pitch: 0.42, dist: 9.0 }
}

impl OpticsApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut app = OpticsApp {
            mode: Mode::Ray,
            four: FourierUi::default(),
            scene: Scene::default(),
            settings: Settings::default(),
            selected: None,
            top: default_top(),
            orbit: default_orbit(),
            gpu: None,
            drag: None,
            world: World::default(),
            eye_status: (String::new(), true),
            screen_status: String::new(),
            sight_path: vec![],
            screen_train: 0,
            entrance: (0, None),
            pending_fit: true,
            configs: ConfigWindow::default(),
            shot: std::env::var("OPTICS_SHOT").ok().map(|path| Shot {
                path,
                secs: std::env::var("OPTICS_SHOT_SECS").ok().and_then(|f| f.parse().ok()).unwrap_or(5.0),
                count: 0,
                start: std::time::Instant::now(),
            }),
        };
        if let Some(storage) = cc.storage.filter(|_| app.shot.is_none()) {
            if let Some(p) = eframe::get_value::<Persist>(storage, eframe::APP_KEY) {
                app.scene = p.scene;
                app.settings = p.settings;
                app.top = p.top;
                app.orbit = p.orbit;
                app.pending_fit = false;
                app.mode = p.mode;
                if let Some((params, notes)) = p.fourier {
                    app.four.params = params;
                    app.four.notes = notes;
                }
            }
        }
        if let Some(rs) = cc.wgpu_render_state.as_ref() {
            app.gpu = Some(Gpu::new(rs));
        }
        if app.shot.is_some() {
            if let Some(i) = std::env::var("OPTICS_PRESET").ok().and_then(|p| p.parse::<usize>().ok()) {
                app.load_preset(Preset::ALL[i.min(Preset::ALL.len() - 1)]);
            }
            app.settings.bench_3d = std::env::var("OPTICS_BENCH3D").is_ok_and(|v| v == "1");
            if let Some(i) = std::env::var("OPTICS_FOURIER").ok().and_then(|p| p.parse::<usize>().ok()) {
                app.mode = Mode::Fourier;
                app.four.load_preset(FourierPreset::ALL[i.min(FourierPreset::ALL.len() - 1)]);
            }
            if let Ok(path) = std::env::var("OPTICS_LOAD") {
                app.load_config(std::path::Path::new(&path));
            }
            if let Some(i) = std::env::var("OPTICS_SELECT").ok().and_then(|p| p.parse::<usize>().ok()) {
                app.selected = app.scene.elements.get(i).map(|e| e.id);
            }
        }
        app
    }

    fn load_preset(&mut self, p: Preset) {
        self.scene = Scene::preset(p);
        self.selected = None;
        self.pending_fit = true;
    }

    fn fit_view(&mut self, bench: egui::Vec2) {
        if self.scene.elements.is_empty() {
            self.top = default_top();
            return;
        }
        let _ = bench;
        let mut lo = Vec2::splat(f32::INFINITY);
        let mut hi = Vec2::splat(f32::NEG_INFINITY);
        for e in &self.scene.elements {
            lo = lo.min(e.pos);
            hi = hi.max(e.pos);
        }
        // do not zoom out forever for far away telescope targets
        self.top.center = (lo + hi) * 0.5;
        let span = (hi - lo).max(Vec2::new(4.0, 3.0)) + Vec2::new(2.5, 2.0);
        self.top.zoom = (bench.x / span.x).min(bench.y / span.y).clamp(8.0, 400.0);
        let c = Vec3::new(self.top.center.x, 0.6, self.top.center.y);
        self.orbit = Orbit { target: c, yaw: -2.05, pitch: 0.55, dist: (span.x.max(span.y) * 1.0 + 2.0).clamp(3.0, 60.0) };
    }

    fn add_tool(&mut self, tool: Tool, at: Vec2) {
        let obj = |kind| Kind::Object { kind, scale: 1.0 };
        let (kind, yaw) = match tool {
            Tool::Stickman => (obj(ObjKind::Stickman), 0.0),
            Tool::Tree => (obj(ObjKind::Tree), 0.0),
            Tool::Sign => (obj(ObjKind::Sign), 0.0),
            Tool::SlitLamp => (obj(ObjKind::SlitLamp), 0.0),
            Tool::Checker => (obj(ObjKind::Checker), 0.0),
            Tool::Converging => (Kind::Lens(default_lens(1.0)), 0.0),
            Tool::Diverging => (Kind::Lens(default_lens(-1.0)), 0.0),
            Tool::Aperture => (Kind::Aperture(default_aperture()), 0.0),
            Tool::Stop => (Kind::Stop(StopParams { diameter: 0.2 }), 0.0),
            Tool::Prism => (Kind::Prism(default_prism()), 0.0),
            Tool::Screen => {
                // only one screen: move the existing one
                if let Some(id) = self.scene.screen().map(|e| e.id) {
                    self.scene.get_mut(id).unwrap().pos = at;
                    self.selected = Some(id);
                    return;
                }
                (Kind::Screen(default_screen()), 180.0)
            }
            Tool::Eye => {
                if let Some(id) = self.scene.eye().map(|e| e.id) {
                    self.scene.get_mut(id).unwrap().pos = at;
                    self.selected = Some(id);
                    return;
                }
                (Kind::Eye(default_eye()), 180.0)
            }
        };
        let at = (at * 100.0).round() / 100.0;
        let id = self.scene.add(kind, at.x, at.y, yaw);
        self.selected = Some(id);
    }

    // ------------------------------------------------------------ optics status

    fn update_focus(&mut self) {
        self.sight_path.clear();
        let world = &self.world;
        if let Some(eye) = self.scene.eye_mut() {
            let (c, _, _, f) = eye.eye_frame().unwrap();
            let fi = focus_along(world, c, f, false);
            self.sight_path = fi.path.points.clone();
            let Kind::Eye(p) = &mut eye.kind else { unreachable!() };
            let needed = p.prescription - fi.vergence;
            if p.auto_focus && needed.is_finite() {
                p.accommodation = (needed.clamp(0.0, p.max_accommodation) * 1000.0).round() / 1000.0;
            }
            let what = match fi.end {
                Surf::Object(id) => self.scene_name(id),
                Surf::ScreenFront | Surf::ScreenBack => "the screen".into(),
                Surf::Ground => "the grass".into(),
                Surf::Rim(i) if self.world.lenses[i].aperture => "an aperture plate".into(),
                Surf::Rim(_) => "a lens holder".into(),
                _ => "the far distance".into(),
            };
            let Kind::Eye(p) = &self.scene.eye().unwrap().kind else { unreachable!() };
            let light = if fi.vergence.abs() < 0.02 {
                "parallel light".to_string()
            } else if fi.vergence < 0.0 {
                format!("light diverging from {:.2} m", -1.0 / fi.vergence)
            } else {
                format!("light converging to {:.2} m behind the eye", 1.0 / fi.vergence)
            };
            let err = (p.accommodation - needed).abs();
            let sharp = err < 0.12;
            let verdict = if sharp {
                format!("sharp, accommodating {:.2} D", p.accommodation)
            } else if needed > p.max_accommodation {
                format!("blurred: needs {:.1} D, can do {:.1} D", needed, p.max_accommodation)
            } else if needed < 0.0 {
                format!("blurred: would need {:.1} D (cannot relax that far)", needed)
            } else {
                format!("defocused by {:.2} D", err)
            };
            self.eye_status = (format!("Looking at {what} · {light} · {verdict}"), sharp);
        }
        self.screen_status.clear();
        self.screen_train = 0;
        if let Some(s) = self.world.screen {
            let start = s.center + s.normal * s.ht;
            let fi = focus_along(&self.world, start, s.normal, true);
            self.screen_train = fi
                .path
                .events
                .iter()
                .filter_map(|e| match e {
                    Event::Lens(i) | Event::Stop(i) if *i < 32 => Some(1u32 << i),
                    _ => None,
                })
                .fold(0, |a, b| a | b);
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                bytemuck::cast_slice::<_, u8>(&self.world.prims).hash(&mut h);
                bytemuck::cast_slice::<_, u8>(&self.world.gpu_lenses()).hash(&mut h);
                bytemuck::cast_slice::<_, u8>(&self.world.gpu_prisms()).hash(&mut h);
                format!("{:?}", s).hash(&mut h);
                (self.screen_train, self.settings.screen_tube).hash(&mut h);
                h.finish()
            };
            if key != self.entrance.0 {
                self.entrance = (key, compute_entrance(&self.world, self.screen_train, self.settings.screen_tube));
            }
            let pinhole = !fi.path.events.iter().any(|e| matches!(e, Event::Lens(_)))
                && fi.path.events.iter().any(|e| matches!(e, Event::Stop(_)));
            if pinhole {
                self.screen_status = "pinhole camera: sharp at any distance, blurred by the size of the hole".into();
            } else if self.entrance.1.is_some() {
                if let (Some(v_after), Some(d)) = (fi.vergence_after_first_lens, fi.first_lens_distance) {
                    if v_after > 1e-3 {
                        let b = 1.0 / v_after;
                        let delta = b - d;
                        self.screen_status = if delta.abs() < 0.01 {
                            format!("sharp image {:.0} cm behind the lens: in focus", b * 100.0)
                        } else {
                            format!(
                                "sharp image {:.0} cm behind the lens, screen at {:.0} cm ({} {:.0} cm)",
                                b * 100.0,
                                d * 100.0,
                                if delta > 0.0 { "move away by" } else { "move closer by" },
                                delta.abs() * 100.0
                            )
                        };
                    } else {
                        self.screen_status = "no real image: the light diverges behind the lens".into();
                    }
                } else {
                    self.screen_status = "the screen's axis misses the lens".into();
                }
            } else {
                self.screen_status = "nothing in front of the screen: it is lit evenly".into();
            }
        }
    }

    fn scene_name(&self, id: u32) -> String {
        self.scene.get(id).map_or("?".into(), |e| match &e.kind {
            Kind::Object { kind, .. } => format!("the {}", kind.label()),
            Kind::Lens(_) => "a lens holder".into(),
            Kind::Aperture(_) => "an aperture plate".into(),
            Kind::Stop(_) => "a central stop".into(),
            Kind::Prism(_) => "a prism stand".into(),
            Kind::Eye(_) => "the eye".into(),
            Kind::Screen(_) => "the screen".into(),
        })
    }

    // ------------------------------------------------------------ GPU frame data

    fn frame_data(&self, want: [bool; 3]) -> (Globals, [Option<ViewUniform>; 3]) {
        let sd = self.scene.sun_dir();
        let mut flags = 0u32;
        if self.world.dispersive() {
            flags |= 1;
        }
        let mut g = Globals {
            sun_dir: [sd.x, sd.y, sd.z, 0.02],
            sun_col: [2.5, 2.35, 2.1, 1.0],
            counts: [self.world.objs.len() as u32, self.world.lenses.len() as u32, 0, 0],
            misc: [0.0035, 0.07, 0.014, 0.0],
            counts2: [self.world.prisms.len() as u32, 0, 0, 0],
            ..Default::default()
        };
        let mut views: [Option<ViewUniform>; 3] = [None, None, None];
        let gpu = self.gpu.as_ref();
        if let (Some(s), Some(el)) = (self.world.screen, self.scene.screen()) {
            let Kind::Screen(sp) = &el.kind else { unreachable!() };
            flags |= 2;
            g.scr_center = [s.center.x, s.center.y, s.center.z, s.hw];
            g.scr_right = [s.right.x, s.right.y, s.right.z, s.hh];
            g.scr_up = [s.up.x, s.up.y, s.up.z, s.ht];
            g.scr_normal = [s.normal.x, s.normal.y, s.normal.z, sp.exposure];
            if want[T_SCREEN] {
                let o = s.center + s.normal * s.ht;
                let mut u = ViewUniform {
                    info: [gpu::MODE_SCREEN, 0, 0, 0],
                    info2: [0, 0, VIS_PHYSICAL, 0],
                    origin: [o.x, o.y, o.z, sp.exposure],
                    right: v4(s.right),
                    up: v4(s.up),
                    fwd: v4(s.normal),
                    p0: [s.hw, s.hh, 1.0, 0.0],
                    p1: [0.0; 4],
                    p2: [0.0; 4],
                };
                if let Some(en) = self.entrance.1 {
                    let d = en.center - o;
                    let d0 = d.length();
                    let dir = d / d0;
                    let norm = d0 * d0 / (dir.dot(s.normal).max(0.05) * dir.dot(en.axis).abs().max(0.05));
                    u.p0[2] = norm;
                    u.p1 = [en.center.x, en.center.y, en.center.z, en.radius];
                    u.p2 = [en.axis.x, en.axis.y, en.axis.z, en.mult];
                    if self.settings.screen_tube {
                        u.info2[3] = self.screen_train;
                    }
                }
                views[T_SCREEN] = Some(u);
            }
        }
        g.counts[2] = flags;
        if let (Some(eye), true) = (self.scene.eye(), want[T_EYE]) {
            let Kind::Eye(p) = &eye.kind else { unreachable!() };
            let (c, r, u, f) = eye.eye_frame().unwrap();
            let (w, h) = gpu.and_then(|g| g.targets[T_EYE].as_ref()).map_or((4, 3), |t| (t.width, t.height));
            let tx = (p.fov_deg.to_radians() * 0.5).tan();
            let ty = tx * h as f32 / w as f32;
            let power = 1.0 / RETINA + p.accommodation - p.prescription;
            views[T_EYE] = Some(ViewUniform {
                info: [gpu::MODE_EYE, 0, 0, 0],
                info2: [0, 0, VIS_PHYSICAL, 0],
                origin: [c.x, c.y, c.z, p.exposure],
                right: v4(r),
                up: v4(u),
                fwd: v4(f),
                p0: [tx, ty, p.pupil_mm * 0.0005, power],
                p1: [RETINA, 0.0, 0.0, 0.0],
                p2: [0.0; 4],
            });
        }
        if want[T_OVERVIEW] {
            let (pos, r, u, f) = self.orbit.frame();
            let (w, h) = gpu.and_then(|g| g.targets[T_OVERVIEW].as_ref()).map_or((4, 3), |t| (t.width, t.height));
            let (tx, ty) = overview_tan(w as f32, h as f32);
            views[T_OVERVIEW] = Some(ViewUniform {
                info: [gpu::MODE_OVERVIEW, 0, 0, 0],
                info2: [0, 0, VIS_PHYSICAL | VIS_OVERVIEW, 0],
                origin: [pos.x, pos.y, pos.z, 1.0],
                right: v4(r),
                up: v4(u),
                fwd: v4(f),
                p0: [tx, ty, 0.0, 0.0],
                p1: [0.0; 4],
                p2: [0.0; 4],
            });
        }
        (g, views)
    }

    // ------------------------------------------------------------ top: eye & screen

    fn views_ui(&mut self, ui: &mut egui::Ui, rs: Option<&egui_wgpu::RenderState>, want: &mut [bool; 3]) {
        ui.add_space(4.0);
        ui.columns(2, |cols| {
            self.eye_view_ui(&mut cols[0], rs, want);
            self.screen_view_ui(&mut cols[1], rs, want);
        });
    }

    fn render_px(&self, ui: &egui::Ui, size: egui::Vec2) -> (u32, u32) {
        let s = ui.ctx().pixels_per_point().min(self.settings.render_scale.max(0.25));
        ((size.x * s).round().max(8.0) as u32, (size.y * s).round().max(8.0) as u32)
    }

    fn eye_view_ui(&mut self, ui: &mut egui::Ui, rs: Option<&egui_wgpu::RenderState>, want: &mut [bool; 3]) {
        ui.horizontal(|ui| {
            ui.strong("EYE");
            if self.scene.eye().is_some() {
                ui.checkbox(&mut self.settings.retinal, "show retinal image (inverted)")
                    .on_hover_text("The image on the retina is upside down; the brain turns it around.");
                ui.checkbox(&mut self.settings.show_sight, "line of sight");
                if let Some(g) = &self.gpu {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.weak(format!("{} spp", g.samples(T_EYE)));
                    });
                }
            }
        });
        let (rect, status_rect) = split_status(ui);
        let resp = ui.allocate_rect(rect, Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, Color32::from_gray(20));
        if self.scene.eye().is_none() {
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                "Drag an eye from the toolbox onto the bench",
                FontId::proportional(15.0),
                Color32::GRAY,
            );
            return;
        }
        want[T_EYE] = true;
        let (w, h) = self.render_px(ui, rect.size());
        if let (Some(gpu), Some(rs)) = (self.gpu.as_mut(), rs) {
            let tex = gpu.target(rs, T_EYE, w, h);
            let uv = if self.settings.retinal {
                Rect::from_min_max(egui::pos2(1.0, 1.0), egui::pos2(0.0, 0.0))
            } else {
                Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0))
            };
            painter.image(tex, rect, uv, Color32::WHITE);
        }
        // fixation cross
        let c = rect.center();
        let col = Color32::from_rgba_unmultiplied(255, 255, 255, 110);
        painter.line_segment([c - egui::vec2(8.0, 0.0), c + egui::vec2(8.0, 0.0)], Stroke::new(1.0, col));
        painter.line_segment([c - egui::vec2(0.0, 8.0), c + egui::vec2(0.0, 8.0)], Stroke::new(1.0, col));
        // drag to look around
        if resp.drag_started() {
            self.drag = Some(Drag::Look);
        }
        if resp.dragged() && matches!(self.drag, Some(Drag::Look)) {
            let d = resp.drag_delta();
            if let Some(e) = self.scene.eye_mut() {
                if let Kind::Eye(p) = &mut e.kind {
                    let k = p.fov_deg.to_radians() / rect.width();
                    e.yaw -= d.x * k;
                    p.pitch_deg = (p.pitch_deg + (d.y * k).to_degrees()).clamp(-60.0, 60.0);
                }
            }
        }
        if resp.drag_stopped() {
            self.drag = None;
        }
        if resp.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                if let Some(Element { kind: Kind::Eye(p), .. }) = self.scene.eye_mut() {
                    p.fov_deg = (p.fov_deg * (-scroll * 0.002).exp()).clamp(5.0, 120.0);
                }
            }
        }
        resp.on_hover_text("Drag to look around, scroll to change the field of view");
        let (text, sharp) = &self.eye_status;
        let col = if *sharp { Color32::from_rgb(120, 200, 120) } else { Color32::from_rgb(230, 170, 90) };
        status_text(ui, status_rect, text, col);
    }

    fn screen_view_ui(&mut self, ui: &mut egui::Ui, rs: Option<&egui_wgpu::RenderState>, want: &mut [bool; 3]) {
        ui.horizontal(|ui| {
            ui.strong("SCREEN");
            if self.scene.screen().is_some() {
                ui.checkbox(&mut self.settings.ruler, "ruler");
                ui.checkbox(&mut self.settings.screen_tube, "light-tight tube").on_hover_text(
                    "Only light that went through all the lenses and apertures in front of the screen reaches it, \
                     as in a camera or a telescope tube. Off: stray light that passes beside the optics \
                     also falls on the screen.",
                );
                ui.checkbox(&mut self.settings.screen_behind, "view from behind").on_hover_text(
                    "Front: what you see standing next to the lens (like a projection screen).\n\
                     Behind: what you see through a translucent screen (like a camera's ground glass).",
                );
                if let Some(g) = &self.gpu {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.weak(format!("{} spp", g.samples(T_SCREEN)));
                    });
                }
            }
        });
        let (outer, status_rect) = split_status(ui);
        ui.allocate_rect(outer, Sense::hover());
        let painter = ui.painter_at(outer);
        painter.rect_filled(outer, 4.0, Color32::from_gray(20));
        let Some(Element { kind: Kind::Screen(sp), .. }) = self.scene.screen().cloned() else {
            painter.text(
                outer.center(),
                Align2::CENTER_CENTER,
                "Drag a screen from the toolbox onto the bench",
                FontId::proportional(15.0),
                Color32::GRAY,
            );
            return;
        };
        want[T_SCREEN] = true;
        let aspect = sp.width / sp.height;
        let pad = if self.settings.ruler { 22.0 } else { 4.0 };
        let inner = outer.shrink(pad);
        let (w, h) = if inner.width() / inner.height() > aspect {
            (inner.height() * aspect, inner.height())
        } else {
            (inner.width(), inner.width() / aspect)
        };
        let rect = Rect::from_center_size(inner.center(), egui::vec2(w, h));
        let (pw, ph) = self.render_px(ui, rect.size());
        if let (Some(gpu), Some(rs)) = (self.gpu.as_mut(), rs) {
            let tex = gpu.target(rs, T_SCREEN, pw, ph);
            let uv = if self.settings.screen_behind {
                Rect::from_min_max(egui::pos2(1.0, 0.0), egui::pos2(0.0, 1.0))
            } else {
                Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0))
            };
            painter.image(tex, rect, uv, Color32::WHITE);
        }
        painter.rect_stroke(rect, 0.0, Stroke::new(1.0, Color32::from_gray(90)), StrokeKind::Outside);
        if self.settings.ruler {
            draw_ruler(&painter, rect, sp.width, sp.height);
        }
        let text = format!("{:.0} × {:.0} cm · {}", sp.width * 100.0, sp.height * 100.0, self.screen_status);
        status_text(ui, status_rect, &text, ui.visuals().text_color());
    }

    // ------------------------------------------------------------ bench

    fn bench_ui(&mut self, ui: &mut egui::Ui, rs: Option<&egui_wgpu::RenderState>, want: &mut [bool; 3]) {
        ui.horizontal(|ui| {
            ui.strong("BENCH");
            ui.selectable_value(&mut self.settings.bench_3d, false, "Top view");
            ui.selectable_value(&mut self.settings.bench_3d, true, "3D view");
            ui.separator();
            if ui.button("Fit").on_hover_text("Show the whole scene").clicked() {
                self.pending_fit = true;
            }
            if self.settings.bench_3d {
                ui.weak("drag things to move them · drag the lawn to orbit · right-drag to pan · scroll to zoom");
            } else {
                ui.checkbox(&mut self.settings.show_rays, "rays");
                ui.checkbox(&mut self.settings.show_virtual, "virtual rays");
                ui.checkbox(&mut self.settings.show_focal, "focal points");
                ui.weak("drag to move · drag the ring to rotate · Shift = no snapping · scroll to zoom");
            }
        });
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        if self.pending_fit {
            self.pending_fit = false;
            self.fit_view(rect.size());
        }
        if self.settings.bench_3d {
            self.bench_3d_ui(ui, rect, &resp, rs, want);
        } else {
            self.bench_top_ui(ui, rect, &resp);
        }
        // dropping things from the toolbox
        if let Some(tool) = resp.dnd_hover_payload::<Tool>() {
            if let Some(p) = ui.ctx().pointer_hover_pos() {
                let painter = ui.painter_at(rect);
                painter.circle_stroke(p, 14.0, Stroke::new(2.0, Color32::WHITE));
                painter.text(p + egui::vec2(18.0, 0.0), Align2::LEFT_CENTER, tool.label(), FontId::proportional(13.0), Color32::WHITE);
            }
        }
        if let Some(tool) = resp.dnd_release_payload::<Tool>() {
            if let Some(p) = ui.ctx().pointer_interact_pos() {
                let at = if self.settings.bench_3d {
                    self.ground_point_3d(rect, p, 0.0).unwrap_or(Vec2::new(self.orbit.target.x, self.orbit.target.z))
                } else {
                    self.s2w(rect, p)
                };
                self.add_tool(*tool, at);
            }
        }
    }

    fn w2s(&self, rect: Rect, p: Vec2) -> Pos2 {
        rect.center() + egui::vec2((p.x - self.top.center.x) * self.top.zoom, (p.y - self.top.center.y) * self.top.zoom)
    }

    fn s2w(&self, rect: Rect, s: Pos2) -> Vec2 {
        let d = s - rect.center();
        self.top.center + Vec2::new(d.x, d.y) / self.top.zoom
    }

    fn handle_pos(&self, rect: Rect, e: &Element) -> Pos2 {
        let f = facing(e.yaw);
        let base = self.w2s(rect, e.pos);
        let r = self.pick_radius(e) * self.top.zoom + 22.0;
        base + egui::vec2(f.x, f.z) * r
    }

    fn pick_radius(&self, e: &Element) -> f32 {
        match &e.kind {
            Kind::Object { kind, scale } => kind.half_width() * scale,
            Kind::Lens(l) => l.diameter * 0.5,
            Kind::Aperture(a) => a.plate * 0.5,
            Kind::Stop(st) => st.diameter * 0.5,
            Kind::Prism(p) => p.side * 0.6,
            Kind::Screen(s) => s.width * 0.5,
            Kind::Eye(_) => 0.08,
        }
    }

    fn element_at(&self, rect: Rect, s: Pos2) -> Option<u32> {
        let p = self.s2w(rect, s);
        let tol = 7.0 / self.top.zoom;
        let mut best: Option<(f32, u32)> = None;
        for e in &self.scene.elements {
            let d = match &e.kind {
                Kind::Lens(_) | Kind::Screen(_) | Kind::Aperture(_) | Kind::Stop(_) => {
                    // distance to the segment across the element
                    let r3 = right_of(e.yaw);
                    let r = Vec2::new(r3.x, r3.z);
                    let half = self.pick_radius(e);
                    let q = p - e.pos;
                    let along = q.dot(r).clamp(-half, half);
                    (q - r * along).length() - 0.05
                }
                Kind::Eye(_) => (p - e.pos).length() - 0.1,
                Kind::Prism(pp) => (p - e.pos).length() - pp.side * 0.45,
                Kind::Object { kind, scale } => {
                    let r = match kind {
                        ObjKind::Tree => 0.9,
                        _ => kind.half_width() * 0.8,
                    } * scale;
                    (p - e.pos).length() - r
                }
            };
            if d < tol && best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, e.id));
            }
        }
        best.map(|b| b.1)
    }

    fn snap_position(&self, id: u32, mut p: Vec2, free: bool) -> Vec2 {
        if free {
            return p;
        }
        p = (p * 100.0).round() / 100.0;
        let tol = 8.0 / self.top.zoom;
        let mut best_z: Option<f32> = None;
        let mut best_x: Option<f32> = None;
        for e in self.scene.elements.iter().filter(|e| e.id != id) {
            if (e.pos.y - p.y).abs() < tol && best_z.is_none_or(|b| (e.pos.y - p.y).abs() < (b - p.y).abs()) {
                best_z = Some(e.pos.y);
            }
            if (e.pos.x - p.x).abs() < tol && best_x.is_none_or(|b| (e.pos.x - p.x).abs() < (b - p.x).abs()) {
                best_x = Some(e.pos.x);
            }
        }
        if let Some(z) = best_z {
            p.y = z;
        }
        if let Some(x) = best_x {
            p.x = x;
        }
        p
    }

    fn bench_top_ui(&mut self, ui: &mut egui::Ui, rect: Rect, resp: &egui::Response) {
        let shift = ui.input(|i| i.modifiers.shift);
        // ---- interaction
        if resp.hovered() {
            let (scroll, zoom) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let factor = (scroll * 0.0025).exp() * zoom;
            if factor != 1.0 {
                if let Some(mp) = resp.hover_pos() {
                    let before = self.s2w(rect, mp);
                    self.top.zoom = (self.top.zoom * factor).clamp(5.0, 2000.0);
                    let after = self.s2w(rect, mp);
                    self.top.center += before - after;
                }
            }
        }
        if resp.drag_started_by(egui::PointerButton::Primary) {
            if let Some(p) = resp.interact_pointer_pos() {
                let on_handle = self
                    .selected
                    .and_then(|id| self.scene.get(id))
                    .filter(|e| (self.handle_pos(rect, e) - p).length() < 10.0)
                    .map(|e| e.id);
                self.drag = if let Some(id) = on_handle {
                    Some(Drag::Rotate { id })
                } else if let Some(id) = self.element_at(rect, p) {
                    self.selected = Some(id);
                    let e = self.scene.get(id).unwrap();
                    Some(Drag::Move { id, offset: e.pos - self.s2w(rect, p) })
                } else {
                    Some(Drag::Pan)
                };
            }
        } else if resp.drag_started() {
            self.drag = Some(Drag::Pan);
        }
        if resp.dragged() {
            let pointer = resp.interact_pointer_pos();
            match (&self.drag, pointer) {
                (Some(Drag::Move { id, offset }), Some(p)) => {
                    let (id, offset) = (*id, *offset);
                    let target = self.snap_position(id, self.s2w(rect, p) + offset, shift);
                    if let Some(e) = self.scene.get_mut(id) {
                        e.pos = target;
                    }
                }
                (Some(Drag::Rotate { id }), Some(p)) => {
                    let id = *id;
                    if let Some(e) = self.scene.get_mut(id) {
                        let c = e.pos;
                        let w = self.top.center + Vec2::new(p.x - rect.center().x, p.y - rect.center().y) / self.top.zoom;
                        let d = w - c;
                        let mut a = d.y.atan2(d.x).to_degrees();
                        if !shift {
                            a = (a / 5.0).round() * 5.0;
                        }
                        e.yaw = a.to_radians();
                    }
                }
                (Some(Drag::Pan), _) => {
                    let d = resp.drag_delta();
                    self.top.center -= Vec2::new(d.x, d.y) / self.top.zoom;
                }
                _ => {}
            }
        }
        if resp.drag_stopped() {
            self.drag = None;
        }
        if resp.clicked() {
            if let Some(p) = resp.interact_pointer_pos() {
                self.selected = self.element_at(rect, p);
            }
        }
        let hovered = resp.hover_pos().and_then(|p| self.element_at(rect, p));
        if hovered.is_some() && self.drag.is_none() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
        }
        if matches!(self.drag, Some(Drag::Move { .. })) {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
        }

        // ---- drawing
        let painter = ui.painter_at(rect);
        self.draw_lawn(&painter, rect);
        if self.settings.show_rays {
            self.draw_rays(&painter, rect);
        }
        self.draw_objects_top(&painter, rect);
        for e in &self.scene.elements {
            match &e.kind {
                Kind::Lens(l) => self.draw_lens_top(&painter, rect, e, l.focal, l.diameter),
                Kind::Aperture(a) => self.draw_aperture_top(&painter, rect, e, a.hole, a.plate),
                Kind::Stop(st) => self.draw_aperture_top(&painter, rect, e, 0.0, st.diameter),
                Kind::Prism(_) => self.draw_prism_top(&painter, rect, e),
                Kind::Screen(s) => self.draw_screen_top(&painter, rect, e, s.width),
                Kind::Eye(p) => self.draw_eye_top(&painter, rect, e, p.fov_deg),
                _ => {}
            }
        }
        if self.settings.show_sight && self.sight_path.len() >= 2 {
            let pts: Vec<Pos2> = self.sight_path.iter().map(|p| self.w2s(rect, Vec2::new(p.x, p.z))).collect();
            painter.extend(egui::Shape::dashed_line(&pts, Stroke::new(1.5, Color32::from_rgb(255, 230, 90)), 6.0, 4.0));
        }
        // hover + selection
        if let Some(e) = hovered.and_then(|id| self.scene.get(id)) {
            let p = self.w2s(rect, e.pos);
            painter.text(
                p + egui::vec2(0.0, -(self.pick_radius(e) * self.top.zoom + 14.0)),
                Align2::CENTER_BOTTOM,
                e.name(),
                FontId::proportional(12.0),
                Color32::WHITE,
            );
        }
        if let Some(e) = self.selected.and_then(|id| self.scene.get(id)) {
            let p = self.w2s(rect, e.pos);
            let r = self.pick_radius(e) * self.top.zoom + 10.0;
            painter.circle_stroke(p, r, Stroke::new(1.5, Color32::from_rgba_unmultiplied(255, 255, 255, 160)));
            let h = self.handle_pos(rect, e);
            let f = facing(e.yaw);
            painter.line_segment([p + egui::vec2(f.x, f.z) * r, h], Stroke::new(1.5, Color32::WHITE));
            painter.circle_filled(h, 6.0, Color32::from_rgb(255, 220, 80));
            painter.circle_stroke(h, 6.0, Stroke::new(1.0, Color32::BLACK));
            painter.text(
                rect.left_bottom() + egui::vec2(8.0, -8.0),
                Align2::LEFT_BOTTOM,
                format!("{}   x = {:.2} m   z = {:.2} m   {:.0}°", e.name(), e.pos.x, e.pos.y, e.yaw.to_degrees()),
                FontId::monospace(12.0),
                Color32::WHITE,
            );
        }
        self.draw_scale_bar(&painter, rect);
    }

    fn draw_lawn(&self, painter: &egui::Painter, rect: Rect) {
        painter.rect_filled(rect, 0.0, Color32::from_rgb(70, 112, 46));
        let lo = self.s2w(rect, rect.left_top());
        let hi = self.s2w(rect, rect.right_bottom());
        // mowing stripes, same as in the 3D rendering
        let mut i = (lo.x / 1.5).floor() as i32;
        while (i as f32) * 1.5 < hi.x {
            if i & 1 == 0 {
                let a = self.w2s(rect, Vec2::new(i as f32 * 1.5, lo.y));
                let b = self.w2s(rect, Vec2::new((i + 1) as f32 * 1.5, hi.y));
                painter.rect_filled(Rect::from_min_max(a, b), 0.0, Color32::from_rgb(80, 124, 52));
            }
            i += 1;
        }
        let step = if self.top.zoom > 60.0 { 0.5 } else if self.top.zoom > 20.0 { 1.0 } else { 5.0 };
        let grid = Color32::from_rgba_unmultiplied(255, 255, 255, 22);
        let major = Color32::from_rgba_unmultiplied(255, 255, 255, 45);
        let mut x = (lo.x / step).floor() * step;
        while x < hi.x {
            let s = self.w2s(rect, Vec2::new(x, 0.0));
            let is_major = (x / (step * 2.0)).fract().abs() < 1e-3;
            painter.line_segment(
                [egui::pos2(s.x, rect.top()), egui::pos2(s.x, rect.bottom())],
                Stroke::new(1.0, if is_major { major } else { grid }),
            );
            if is_major {
                painter.text(egui::pos2(s.x + 2.0, rect.top() + 2.0), Align2::LEFT_TOP, format!("{x:.0}"), FontId::monospace(10.0), major);
            }
            x += step;
        }
        let mut z = (lo.y / step).floor() * step;
        while z < hi.y {
            let s = self.w2s(rect, Vec2::new(0.0, z));
            let is_major = (z / (step * 2.0)).fract().abs() < 1e-3;
            painter.line_segment(
                [egui::pos2(rect.left(), s.y), egui::pos2(rect.right(), s.y)],
                Stroke::new(1.0, if is_major { major } else { grid }),
            );
            z += step;
        }
    }

    fn draw_scale_bar(&self, painter: &egui::Painter, rect: Rect) {
        let mut len = 1.0;
        for l in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0] {
            len = l;
            if l * self.top.zoom > 70.0 {
                break;
            }
        }
        let px = len * self.top.zoom;
        let a = rect.right_bottom() + egui::vec2(-16.0 - px, -14.0);
        let b = a + egui::vec2(px, 0.0);
        let st = Stroke::new(2.0, Color32::WHITE);
        painter.line_segment([a, b], st);
        painter.line_segment([a, a - egui::vec2(0.0, 5.0)], st);
        painter.line_segment([b, b - egui::vec2(0.0, 5.0)], st);
        let label = if len < 1.0 { format!("{:.0} cm", len * 100.0) } else { format!("{len:.0} m") };
        painter.text((a + b.to_vec2()) / 2.0 - egui::vec2(0.0, 6.0), Align2::CENTER_BOTTOM, label, FontId::proportional(11.0), Color32::WHITE);
    }

    /// Objects seen from above: every primitive drawn as its silhouette, lowest first.
    fn draw_objects_top(&self, painter: &egui::Painter, rect: Rect) {
        let z = self.top.zoom;
        let mut items: Vec<(f32, usize)> = vec![];
        for ob in &self.world.objs {
            let start = ob.range[0] as usize;
            for i in start..start + ob.range[1] as usize {
                let p = &self.world.prims[i];
                if p.meta[2] & VIS_PHYSICAL == 0 {
                    continue;
                }
                let top = match p.meta[0] {
                    0 => p.a[1] + p.a[3],
                    1 => p.a[1].max(p.b[1]) + p.a[3],
                    _ => p.a[1] + p.b[1],
                };
                items.push((top, i));
            }
        }
        items.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (_, i) in items {
            let p = &self.world.prims[i];
            let col = top_color(p.meta[1]);
            let edge = Stroke::new(1.0, Color32::from_black_alpha(90));
            let a = self.w2s(rect, Vec2::new(p.a[0], p.a[2]));
            match p.meta[0] {
                0 => {
                    painter.circle(a, (p.a[3] * z).max(1.0), col, edge);
                }
                1 => {
                    let b = self.w2s(rect, Vec2::new(p.b[0], p.b[2]));
                    let w = (2.0 * p.a[3] * z).max(1.0);
                    painter.line_segment([a, b], Stroke::new(w, col));
                    painter.circle_filled(a, w * 0.5, col);
                    painter.circle_filled(b, w * 0.5, col);
                }
                _ => {
                    let r = egui::vec2(p.c[0], p.c[1]) * p.b[0] * z;
                    let f = egui::vec2(p.c[2], p.c[3]) * p.b[2] * z;
                    let pts = vec![a + r + f, a - r + f, a - r - f, a + r - f];
                    painter.add(egui::Shape::convex_polygon(pts, col, edge));
                }
            }
        }
    }

    fn draw_lens_top(&self, painter: &egui::Painter, rect: Rect, e: &Element, focal: f32, diameter: f32) {
        let f3 = facing(e.yaw);
        let r3 = right_of(e.yaw);
        let f = egui::vec2(f3.x, f3.z);
        let r = egui::vec2(r3.x, r3.z);
        let c = self.w2s(rect, e.pos);
        let z = self.top.zoom;
        // optical axis
        let axis_col = Color32::from_rgba_unmultiplied(255, 255, 255, 60);
        let reach = 2.5 * focal.abs().max(1.0) * z;
        painter.extend(egui::Shape::dashed_line(&[c - f * reach, c + f * reach], Stroke::new(1.0, axis_col), 8.0, 5.0));
        draw_lens_shape(painter, c, r, f, diameter * 0.5 * z, focal, z);
        if self.settings.show_focal {
            for (k, label) in [(1.0, "F"), (-1.0, "F"), (2.0, "2F"), (-2.0, "2F")] {
                let p = c + f * (k * focal * z);
                let is_f = label == "F";
                painter.circle_filled(p, if is_f { 3.5 } else { 2.5 }, Color32::from_rgb(255, 240, 150));
                painter.text(
                    p + r * 8.0 + egui::vec2(0.0, -2.0),
                    Align2::CENTER_BOTTOM,
                    label,
                    FontId::proportional(10.0),
                    Color32::from_rgb(255, 240, 150),
                );
            }
        }
    }

    fn draw_screen_top(&self, painter: &egui::Painter, rect: Rect, e: &Element, width: f32) {
        let f3 = facing(e.yaw);
        let r3 = right_of(e.yaw);
        let f = egui::vec2(f3.x, f3.z);
        let r = egui::vec2(r3.x, r3.z);
        let c = self.w2s(rect, e.pos);
        let z = self.top.zoom;
        let hw = width * 0.5 * z;
        let t = (0.03 * z).max(3.0);
        let pts = vec![c + r * hw, c - r * hw, c - r * hw - f * t, c + r * hw - f * t];
        painter.add(egui::Shape::convex_polygon(pts, Color32::from_gray(70), Stroke::NONE));
        painter.line_segment([c + r * hw, c - r * hw], Stroke::new(2.5, Color32::WHITE));
        // little arrow showing where the screen faces
        painter.arrow(c, f * 18.0, Stroke::new(1.5, Color32::WHITE));
    }

    fn draw_eye_top(&self, painter: &egui::Painter, rect: Rect, e: &Element, fov: f32) {
        let f3 = facing(e.yaw);
        let f = egui::vec2(f3.x, f3.z);
        let n = egui::vec2(-f.y, f.x);
        let c = self.w2s(rect, e.pos);
        // field of view wedge
        let half = (fov * 0.5).to_radians();
        let len = 220.0;
        let rot = |v: egui::Vec2, a: f32| egui::vec2(v.x * a.cos() - v.y * a.sin(), v.x * a.sin() + v.y * a.cos());
        let wedge = Color32::from_rgba_unmultiplied(255, 255, 255, 18);
        let pts: Vec<Pos2> = std::iter::once(c)
            .chain((0..=16).map(|i| c + rot(f, -half + 2.0 * half * i as f32 / 16.0) * len))
            .collect();
        painter.add(egui::Shape::convex_polygon(pts, wedge, Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 50))));
        // almond shaped eye
        let s = 13.0;
        let back = c - f * s * 1.6;
        let mut outline = vec![];
        for i in 0..=12 {
            let t = i as f32 / 12.0;
            let a = (t - 0.5) * std::f32::consts::PI * 0.9;
            outline.push(back + f * (a.cos() * s * 1.6) + n * (a.sin() * s));
        }
        painter.add(egui::Shape::convex_polygon(outline.clone(), Color32::WHITE, Stroke::new(1.5, Color32::BLACK)));
        painter.circle(c - f * 5.0, 6.0, Color32::from_rgb(40, 110, 200), Stroke::NONE);
        painter.circle_filled(c - f * 3.5, 3.0, Color32::BLACK);
    }

    fn draw_aperture_top(&self, painter: &egui::Painter, rect: Rect, e: &Element, hole: f32, plate: f32) {
        let f3 = facing(e.yaw);
        let r3 = right_of(e.yaw);
        let f = egui::vec2(f3.x, f3.z);
        let r = egui::vec2(r3.x, r3.z);
        let c = self.w2s(rect, e.pos);
        let z = self.top.zoom;
        let axis_col = Color32::from_rgba_unmultiplied(255, 255, 255, 45);
        painter.extend(egui::Shape::dashed_line(&[c - f * 2.0 * z, c + f * 2.0 * z], Stroke::new(1.0, axis_col), 8.0, 5.0));
        let h = hole * 0.5 * z;
        let o = (plate * 0.5 * z).max(h + 3.0);
        let st = Stroke::new((0.03 * z).clamp(3.0, 8.0), Color32::from_gray(20));
        painter.line_segment([c + r * h, c + r * o], st);
        painter.line_segment([c - r * h, c - r * o], st);
        // mark the edges of the hole
        if hole > 0.0 {
            for s in [-1.0, 1.0] {
                painter.circle_filled(c + r * (s * h), 2.0, Color32::from_rgb(255, 220, 80));
            }
        }
    }

    fn draw_prism_top(&self, painter: &egui::Painter, rect: Rect, e: &Element) {
        let Some(p) = self.world.prisms.iter().find(|p| p.id == e.id) else { return };
        let pts: Vec<Pos2> = p
            .corners
            .iter()
            .map(|c| {
                let w = p.to_world(*c);
                self.w2s(rect, Vec2::new(w.x, w.z))
            })
            .collect();
        painter.add(egui::Shape::convex_polygon(
            pts.clone(),
            Color32::from_rgba_unmultiplied(190, 225, 255, 130),
            Stroke::new(1.3, Color32::from_rgb(20, 60, 110)),
        ));
        // the base is frosted: draw it a bit thicker
        painter.line_segment([pts[1], pts[2]], Stroke::new(2.5, Color32::from_rgb(20, 60, 110)));
    }

    /// Fans of rays in the horizontal plane. Normally from the left (red) and right
    /// (blue) edge of the object in front of each lens, aperture or prism. When the
    /// light passes dispersive glass, one fan per colour from the object's centre.
    fn draw_rays(&self, painter: &egui::Painter, rect: Rect) {
        /// points across an opening that the rays aim at
        struct Target {
            id: u32,
            center: Vec3,
            across: Vec3,
            half: f32,
            /// direction in which the light should arrive
            incoming: Vec3,
            two_sided: bool,
        }
        let mut targets: Vec<Target> = vec![];
        for l in self.world.lenses.iter().filter(|l| !l.obstruction) {
            targets.push(Target {
                id: l.id,
                center: l.center,
                across: l.axis.cross(Vec3::Y).normalize(),
                half: 0.9 * l.radius,
                incoming: l.axis,
                two_sided: true,
            });
        }
        for p in &self.world.prisms {
            // entrance face at minimum deviation
            let delta = min_deviation(self.prism_apex(p.id), p.n_d).unwrap_or(0.7);
            let incoming = p.fwd * (delta * 0.5).cos() + p.right * (delta * 0.5).sin();
            let a = p.to_world(p.corners[0]);
            let b = p.to_world(p.corners[1]);
            targets.push(Target {
                id: p.id,
                center: (a + b) * 0.5,
                across: (b - a).normalize(),
                half: 0.4 * (b - a).length(),
                incoming,
                two_sided: false,
            });
        }
        let flat = |e: &Element, t: &Target| Vec3::new(e.pos.x, t.center.y, e.pos.y);
        let accept = |e: &Element, t: &Target| {
            let d = (t.center - flat(e, t)).normalize_or_zero();
            let c = d.dot(t.incoming);
            if t.two_sided { c.abs() } else { c }
        };
        let dist = |e: &Element, t: &Target| (flat(e, t) - t.center).length();
        let selected = self
            .selected
            .and_then(|id| self.scene.get(id))
            .filter(|e| matches!(e.kind, Kind::Object { .. }));
        let mut pairs: Vec<(&Element, &Target)> = vec![];
        if let Some(e) = selected {
            if let Some(t) = targets
                .iter()
                .filter(|t| accept(e, t) > 0.7 && dist(e, t) < 40.0)
                .min_by(|a, b| dist(e, a).total_cmp(&dist(e, b)))
            {
                pairs.push((e, t));
            }
        } else {
            for t in &targets {
                let ok = |e: &&Element| matches!(e.kind, Kind::Object { .. }) && accept(e, t) > 0.9 && dist(e, t) < 40.0;
                let Some(e) = self.scene.elements.iter().filter(ok).min_by(|a, b| dist(a, t).total_cmp(&dist(b, t)))
                else {
                    continue;
                };
                // skip openings whose light comes through another one (e.g. an eyepiece)
                let blocked = targets.iter().any(|o| o.id != t.id && accept(e, o) > 0.9 && dist(e, o) < dist(e, t));
                if !blocked && !pairs.iter().any(|(pe, _)| pe.id == e.id) {
                    pairs.push((e, t));
                }
            }
        }
        let spectrum = [
            (450.0, Color32::from_rgb(90, 120, 255)),
            (530.0, Color32::from_rgb(70, 210, 90)),
            (650.0, Color32::from_rgb(255, 70, 50)),
        ];
        for (e, t) in pairs {
            let Kind::Object { kind, scale } = &e.kind else { continue };
            let center = flat(e, t);
            let reach = dist(e, t) + 12.0;
            let n = 9;
            let aim = |k: usize| t.center + t.across * ((k as f32 / (n - 1) as f32 - 0.5) * 2.0 * t.half);
            let probe = self.world.trace_path_ex(center, (t.center - center).normalize(), reach, false, Some(e.id), LAMBDA_D);
            // (source, colour, wavelength) of each fan
            let fans: Vec<(Vec3, Color32, f32)> = if probe.dispersed {
                spectrum.iter().map(|(l, c)| (center, *c, *l)).collect()
            } else {
                let r = right_of(e.yaw);
                [(-1.0, Color32::from_rgb(255, 90, 70)), (1.0, Color32::from_rgb(90, 150, 255))]
                    .iter()
                    .map(|(side, c)| (center + r * (side * kind.half_width() * scale * 0.9), *c, LAMBDA_D))
                    .collect()
            };
            for (src, col, lambda) in fans {
                for k in 0..n {
                    let dir = (aim(k) - src).normalize();
                    let path = self.world.trace_path_ex(src, dir, reach, false, Some(e.id), lambda);
                    if path.events.is_empty() {
                        continue;
                    }
                    let pts: Vec<Pos2> = path.points.iter().map(|p| self.w2s(rect, Vec2::new(p.x, p.z))).collect();
                    painter.add(egui::Shape::line(pts, Stroke::new(1.2, col.gamma_multiply(0.75))));
                    if let (true, Some(k)) = (self.settings.show_virtual, path.last_refraction()) {
                        let last = path.points[k + 1];
                        let d = path.dirs[k + 1];
                        let a = self.w2s(rect, Vec2::new(last.x, last.z));
                        let b = self.w2s(rect, Vec2::new(last.x - d.x * 8.0, last.z - d.z * 8.0));
                        painter.extend(egui::Shape::dashed_line(&[a, b], Stroke::new(1.0, col.gamma_multiply(0.45)), 5.0, 4.0));
                    }
                }
            }
        }
    }

    fn prism_apex(&self, id: u32) -> f32 {
        match self.scene.get(id).map(|e| &e.kind) {
            Some(Kind::Prism(p)) => p.apex_deg,
            _ => 60.0,
        }
    }

    // ------------------------------------------------------------ 3D bench

    fn camera_ray(&self, rect: Rect, p: Pos2) -> (Vec3, Vec3) {
        let (pos, r, u, f) = self.orbit.frame();
        let (tx, ty) = overview_tan(rect.width(), rect.height());
        let sx = (p.x - rect.left()) / rect.width() * 2.0 - 1.0;
        let sy = 1.0 - (p.y - rect.top()) / rect.height() * 2.0;
        (pos, (f + r * (sx * tx) + u * (sy * ty)).normalize())
    }

    fn ground_point_3d(&self, rect: Rect, p: Pos2, y: f32) -> Option<Vec2> {
        let (ro, rd) = self.camera_ray(rect, p);
        if rd.y.abs() < 1e-5 {
            return None;
        }
        let t = (y - ro.y) / rd.y;
        if t <= 0.0 {
            return None;
        }
        let q = ro + rd * t;
        Some(Vec2::new(q.x, q.z))
    }

    fn pick_3d(&self, rect: Rect, p: Pos2) -> Option<u32> {
        let (ro, rd) = self.camera_ray(rect, p);
        let (_, s) = self.world.intersect(ro, rd, VIS_PHYSICAL | VIS_OVERVIEW, false);
        match s {
            Surf::Object(id) => Some(id),
            Surf::Lens(i) | Surf::Rim(i) => Some(self.world.lenses[i].id),
            Surf::Prism(i) => Some(self.world.prisms[i].id),
            Surf::ScreenFront | Surf::ScreenBack => self.world.screen.map(|s| s.id),
            _ => None,
        }
    }

    fn bench_3d_ui(
        &mut self,
        ui: &mut egui::Ui,
        rect: Rect,
        resp: &egui::Response,
        rs: Option<&egui_wgpu::RenderState>,
        want: &mut [bool; 3],
    ) {
        want[T_OVERVIEW] = true;
        let painter = ui.painter_at(rect);
        let (w, h) = self.render_px(ui, rect.size());
        if let (Some(gpu), Some(rs)) = (self.gpu.as_mut(), rs) {
            let tex = gpu.target(rs, T_OVERVIEW, w, h);
            painter.image(tex, rect, Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
        }
        if resp.drag_started_by(egui::PointerButton::Primary) {
            if let Some(p) = resp.interact_pointer_pos() {
                self.drag = match self.pick_3d(rect, p) {
                    Some(id) => {
                        self.selected = Some(id);
                        let e = self.scene.get(id).unwrap();
                        let plane_y = match e.kind {
                            Kind::Object { .. } => 0.0,
                            _ => e.height,
                        };
                        let g = self.ground_point_3d(rect, p, plane_y).unwrap_or(e.pos);
                        Some(Drag::Move3d { id, offset: e.pos - g, plane_y })
                    }
                    None => Some(Drag::Orbit),
                };
            }
        } else if resp.drag_started() {
            self.drag = Some(Drag::Pan);
        }
        if resp.dragged() {
            let d = resp.drag_delta();
            match &self.drag {
                Some(Drag::Orbit) => {
                    self.orbit.yaw += d.x * 0.008;
                    self.orbit.pitch = (self.orbit.pitch + d.y * 0.006).clamp(0.02, 1.5);
                }
                Some(Drag::Pan) => {
                    let (_, r, u, _) = self.orbit.frame();
                    let k = self.orbit.dist * 0.0018;
                    self.orbit.target += (-r * d.x + u * d.y) * k;
                    self.orbit.target.y = self.orbit.target.y.clamp(0.0, 10.0);
                }
                Some(Drag::Move3d { id, offset, plane_y }) => {
                    let (id, offset, plane_y) = (*id, *offset, *plane_y);
                    if let Some(p) = resp.interact_pointer_pos() {
                        if let Some(g) = self.ground_point_3d(rect, p, plane_y) {
                            let shift = ui.input(|i| i.modifiers.shift);
                            let target = self.snap_position(id, g + offset, shift);
                            if let Some(e) = self.scene.get_mut(id) {
                                e.pos = target;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        if resp.drag_stopped() {
            self.drag = None;
        }
        if resp.clicked() {
            if let Some(p) = resp.interact_pointer_pos() {
                self.selected = self.pick_3d(rect, p);
            }
        }
        if resp.hovered() {
            let (scroll, zoom) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let factor = (-scroll * 0.002).exp() / zoom;
            self.orbit.dist = (self.orbit.dist * factor).clamp(0.5, 200.0);
        }
        if let Some(e) = self.selected.and_then(|id| self.scene.get(id)) {
            painter.text(
                rect.left_bottom() + egui::vec2(8.0, -8.0),
                Align2::LEFT_BOTTOM,
                format!("selected: {}   x = {:.2} m   z = {:.2} m", e.name(), e.pos.x, e.pos.y),
                FontId::monospace(12.0),
                Color32::WHITE,
            );
        }
        if let Some(g) = &self.gpu {
            painter.text(
                rect.right_top() + egui::vec2(-8.0, 6.0),
                Align2::RIGHT_TOP,
                format!("{} spp", g.samples(T_OVERVIEW)),
                FontId::proportional(11.0),
                Color32::from_white_alpha(140),
            );
        }
    }

    // ------------------------------------------------------------ toolbox + inspector

    fn toolbox_ui(&mut self, ui: &mut egui::Ui) {
        let w = (ui.available_width() * 0.5).max(360.0);
        egui::Panel::right("inspector").resizable(true).default_size(w).min_size(300.0).show(ui, |ui| {
            egui::ScrollArea::vertical().id_salt("inspector").auto_shrink([false, false]).show(ui, |ui| {
                ui.add_space(4.0);
                self.inspector_ui(ui);
            });
        });
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.strong("TOOLBOX");
                ui.weak("drag onto the bench (or click)");
            });
            ui.add_space(4.0);
            egui::ScrollArea::vertical().id_salt("tiles").show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    for tool in Tool::ALL {
                        self.tool_tile(ui, tool);
                    }
                });
            });
        });
    }

    fn tool_tile(&mut self, ui: &mut egui::Ui, tool: Tool) {
        let size = egui::vec2(84.0, 84.0);
        // allocated directly in the wrapping layout, so the tiles wrap onto new rows
        let (rect, resp) = ui.allocate_exact_size(size, Sense::click_and_drag());
        resp.dnd_set_drag_payload(tool);
        let visuals = ui.style().interact(&resp);
        let painter = ui.painter();
        painter.rect_filled(rect, 8.0, visuals.bg_fill.gamma_multiply(0.9));
        painter.rect_stroke(rect, 8.0, visuals.bg_stroke, StrokeKind::Inside);
        let icon = Rect::from_center_size(rect.center() - egui::vec2(0.0, 9.0), egui::vec2(52.0, 52.0));
        draw_tool_icon(painter, icon, tool);
        painter.text(
            rect.center_bottom() - egui::vec2(0.0, 5.0),
            Align2::CENTER_BOTTOM,
            tool.label(),
            FontId::proportional(10.5),
            visuals.text_color(),
        );
        if resp.dragged() {
            // a small copy of the tile follows the mouse
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
            if let Some(p) = ui.ctx().pointer_interact_pos() {
                let layer = egui::LayerId::new(egui::Order::Tooltip, egui::Id::new(("tool drag", tool as u32)));
                let painter = ui.ctx().layer_painter(layer);
                let r = Rect::from_center_size(p, egui::vec2(56.0, 56.0));
                painter.rect_filled(r, 8.0, visuals.bg_fill.gamma_multiply(0.8));
                draw_tool_icon(&painter, r.shrink(6.0), tool);
            }
        } else if resp.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
        }
        if resp.clicked() {
            let c = self.top.center;
            let at = match tool {
                Tool::Converging | Tool::Diverging | Tool::Aperture | Tool::Stop | Tool::Prism | Tool::Screen | Tool::Eye => c,
                _ => c - Vec2::new(2.0, 0.0),
            };
            self.add_tool(tool, at);
        }
    }

    fn inspector_ui(&mut self, ui: &mut egui::Ui) {
        let Some(id) = self.selected else {
            if !self.scene.notes.trim().is_empty() {
                ui.strong("ABOUT THIS SCENE");
                ui.label(&self.scene.notes);
                ui.add_space(8.0);
            }
            ui.strong("INSPECTOR");
            ui.label("Select something on the bench to change its properties.");
            ui.add_space(8.0);
            ui.weak("Keys: Delete removes the selection, arrow keys nudge it by 1 cm (Shift: 10 cm), Cmd+S saves the scene.");
            return;
        };
        let lenses: Vec<(Vec2, f32)> = self
            .scene
            .elements
            .iter()
            .filter_map(|e| if let Kind::Lens(l) = &e.kind { Some((e.pos, l.focal)) } else { None })
            .collect();
        let objects: Vec<(String, Vec2)> = self
            .scene
            .elements
            .iter()
            .filter(|e| matches!(e.kind, Kind::Object { .. }))
            .map(|e| (e.name(), e.pos))
            .collect();
        let Some(e) = self.scene.get_mut(id) else {
            self.selected = None;
            return;
        };
        let mut delete = false;
        let mut duplicate = false;
        ui.horizontal(|ui| {
            ui.strong(e.name());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                delete = ui.button("Delete").clicked();
                if !matches!(e.kind, Kind::Screen(_) | Kind::Eye(_)) {
                    duplicate = ui.button("Duplicate").clicked();
                }
            });
        });
        egui::Grid::new("inspector_grid").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
            ui.label("position x");
            ui.add(egui::DragValue::new(&mut e.pos.x).speed(0.01).suffix(" m").fixed_decimals(2));
            ui.end_row();
            ui.label("position z");
            ui.add(egui::DragValue::new(&mut e.pos.y).speed(0.01).suffix(" m").fixed_decimals(2));
            ui.end_row();
            ui.label("rotation");
            let mut deg = e.yaw.to_degrees();
            if ui.add(egui::DragValue::new(&mut deg).speed(0.5).suffix("°").fixed_decimals(1)).changed() {
                e.yaw = deg.to_radians();
            }
            ui.end_row();
            if !matches!(e.kind, Kind::Object { .. }) {
                ui.label("height");
                ui.add(egui::Slider::new(&mut e.height, 0.1..=3.0).suffix(" m"));
                ui.end_row();
            }
            match &mut e.kind {
                Kind::Object { scale, .. } => {
                    ui.label("size");
                    ui.add(egui::Slider::new(scale, 0.1..=3.0).suffix("×"));
                    ui.end_row();
                }
                Kind::Lens(l) => {
                    ui.label("type");
                    ui.horizontal(|ui| {
                        let mut conv = l.focal >= 0.0;
                        if ui.radio_value(&mut conv, true, "converging").changed()
                            || ui.radio_value(&mut conv, false, "diverging").changed()
                        {
                            l.focal = if conv { l.focal.abs() } else { -l.focal.abs() };
                        }
                    });
                    ui.end_row();
                    ui.label("focal length |f|");
                    let mut af = l.focal.abs();
                    if ui.add(egui::Slider::new(&mut af, 0.05..=10.0).logarithmic(true).suffix(" m")).changed() {
                        l.focal = af.copysign(l.focal);
                    }
                    ui.end_row();
                    ui.label("power");
                    ui.label(format!("{:+.2} dpt", 1.0 / l.focal));
                    ui.end_row();
                    ui.label("diameter");
                    ui.add(egui::Slider::new(&mut l.diameter, 0.02..=2.0).logarithmic(true).suffix(" m"));
                    ui.end_row();
                    ui.label("spherical aberration");
                    ui.add(egui::Slider::new(&mut l.spherical, 0.0..=1.0))
                        .on_hover_text("Relative extra power at the rim of the lens");
                    ui.end_row();
                    ui.label("chromatic aberration");
                    ui.add(egui::Slider::new(&mut l.chromatic, 0.0..=0.15))
                        .on_hover_text("Relative power difference between blue (486 nm) and yellow (588 nm) light, ≈ 1/V");
                    ui.end_row();
                }
                Kind::Aperture(a) => {
                    ui.label("hole diameter");
                    ui.add(egui::Slider::new(&mut a.hole, 0.002..=1.0).logarithmic(true).suffix(" m"));
                    ui.end_row();
                    ui.label("plate diameter");
                    ui.add(egui::Slider::new(&mut a.plate, 0.1..=3.0).suffix(" m"));
                    ui.end_row();
                    if let Some((pos, f)) = lenses
                        .iter()
                        .filter(|(p, _)| (*p - e.pos).length() < 0.5)
                        .min_by(|x, y| (x.0 - e.pos).length().total_cmp(&(y.0 - e.pos).length()))
                    {
                        let _ = pos;
                        ui.label("f-number");
                        ui.label(format!("N = f/D = {:.1}", f.abs() / a.hole));
                        ui.end_row();
                    }
                }
                Kind::Stop(st) => {
                    ui.label("disk diameter");
                    ui.add(egui::Slider::new(&mut st.diameter, 0.01..=1.5).logarithmic(true).suffix(" m"))
                        .on_hover_text("An opaque disk: it blocks the centre of the beam and lets the outer ring pass.");
                    ui.end_row();
                }
                Kind::Prism(p) => {
                    ui.label("apex angle");
                    ui.add(egui::Slider::new(&mut p.apex_deg, 10.0..=90.0).suffix("°"));
                    ui.end_row();
                    ui.label("face length");
                    ui.add(egui::Slider::new(&mut p.side, 0.1..=1.2).suffix(" m"));
                    ui.end_row();
                    ui.label("prism height");
                    ui.add(egui::Slider::new(&mut p.height, 0.1..=1.5).suffix(" m"));
                    ui.end_row();
                    ui.label("glass");
                    ui.horizontal_wrapped(|ui| {
                        for (name, n, v) in
                            [("crown", 1.52, 59.0), ("flint", 1.62, 36.0), ("dense flint", 1.75, 27.0), ("exaggerated", 1.62, 10.0)]
                        {
                            if ui.selectable_label(p.n_d == n && p.abbe == v, name).clicked() {
                                p.n_d = n;
                                p.abbe = v;
                            }
                        }
                    });
                    ui.end_row();
                    ui.label("refractive index n_d");
                    ui.add(egui::Slider::new(&mut p.n_d, 1.3..=2.0));
                    ui.end_row();
                    ui.label("Abbe number V");
                    ui.add(egui::Slider::new(&mut p.abbe, 5.0..=90.0).logarithmic(true))
                        .on_hover_text("V = (n_d − 1)/(n_F − n_C). Small V = strong dispersion.");
                    ui.end_row();
                    ui.label("n (blue / yellow / red)");
                    ui.label(format!(
                        "{:.3} / {:.3} / {:.3}",
                        prism_index(p, LAMBDA_F),
                        prism_index(p, LAMBDA_D),
                        prism_index(p, LAMBDA_C)
                    ));
                    ui.end_row();
                    ui.label("min. deviation");
                    let dev = |l: f32| min_deviation(p.apex_deg, prism_index(p, l)).map(|d| d.to_degrees());
                    ui.label(match (dev(LAMBDA_F), dev(LAMBDA_C)) {
                        (Some(b), Some(r)) => format!("{:.1}° (blue) – {:.1}° (red)", b, r),
                        _ => "total internal reflection".into(),
                    });
                    ui.end_row();
                }
                Kind::Screen(s) => {
                    ui.label("width");
                    ui.add(egui::Slider::new(&mut s.width, 0.1..=4.0).suffix(" m"));
                    ui.end_row();
                    ui.label("height");
                    ui.add(egui::Slider::new(&mut s.height, 0.1..=4.0).suffix(" m"));
                    ui.end_row();
                    ui.label("brightness");
                    ui.add(egui::Slider::new(&mut s.exposure, 0.1..=10.0).logarithmic(true));
                    ui.end_row();
                }
                Kind::Eye(p) => {
                    ui.label("auto focus");
                    ui.checkbox(&mut p.auto_focus, "accommodate on the centre of view");
                    ui.end_row();
                    ui.label("accommodation");
                    ui.add_enabled(
                        !p.auto_focus,
                        egui::Slider::new(&mut p.accommodation, 0.0..=p.max_accommodation.max(0.01)).suffix(" D"),
                    );
                    ui.end_row();
                    ui.label("max. accommodation");
                    ui.add(egui::Slider::new(&mut p.max_accommodation, 0.0..=12.0).suffix(" D"))
                        .on_hover_text("4 D: near point at 25 cm; a child has ~12 D, a 60 year old < 1 D");
                    ui.end_row();
                    ui.label("glasses needed");
                    ui.add(egui::Slider::new(&mut p.prescription, -6.0..=6.0).suffix(" D"))
                        .on_hover_text("Negative: short-sighted (myopic) eye, positive: far-sighted");
                    ui.end_row();
                    ui.label("pupil");
                    ui.add(egui::Slider::new(&mut p.pupil_mm, 0.5..=20.0).logarithmic(true).suffix(" mm"))
                        .on_hover_text("Real pupils are 2–8 mm. Larger values exaggerate the blur.");
                    ui.end_row();
                    ui.label("field of view");
                    ui.add(egui::Slider::new(&mut p.fov_deg, 5.0..=120.0).suffix("°"));
                    ui.end_row();
                    ui.label("look up/down");
                    ui.add(egui::Slider::new(&mut p.pitch_deg, -60.0..=60.0).suffix("°"));
                    ui.end_row();
                    ui.label("brightness");
                    ui.add(egui::Slider::new(&mut p.exposure, 0.1..=10.0).logarithmic(true));
                    ui.end_row();
                }
            }
        });
        if let Kind::Lens(l) = &e.kind {
            let ax = facing(e.yaw);
            let ax = Vec2::new(ax.x, ax.z);
            let lines: Vec<String> = objects
                .iter()
                .filter_map(|(name, p)| {
                    let g = (*p - e.pos).dot(ax).abs();
                    if g < 0.01 {
                        return None;
                    }
                    let inv_b = 1.0 / l.focal - 1.0 / g;
                    Some(if inv_b.abs() < 1e-4 {
                        format!("{name}: g = {g:.2} m → image at infinity")
                    } else {
                        let b = 1.0 / inv_b;
                        let m = -b / g;
                        format!(
                            "{name}: g = {g:.2} m → b = {b:+.2} m ({}), m = {m:+.2}",
                            if b > 0.0 { "real" } else { "virtual" }
                        )
                    })
                })
                .collect();
            if !lines.is_empty() {
                ui.add_space(6.0);
                ui.weak("Thin lens equation 1/f = 1/g + 1/b for this lens alone:");
                for l in lines {
                    ui.monospace(l);
                }
            }
        }
        if delete {
            self.scene.remove(id);
            self.selected = None;
        } else if duplicate {
            let mut copy = self.scene.get(id).unwrap().clone();
            copy.id = self.scene.next_id;
            self.scene.next_id += 1;
            copy.pos += Vec2::new(0.3, 0.3);
            self.selected = Some(copy.id);
            self.scene.elements.push(copy);
        }
    }

    fn menu_ui(&mut self, ui: &mut egui::Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("Scene", |ui| {
                ui.menu_button("Examples: ray optics", |ui| {
                    for p in Preset::ALL {
                        if ui.button(p.label()).clicked() {
                            self.load_preset(p);
                            self.mode = Mode::Ray;
                            ui.close();
                        }
                    }
                });
                ui.menu_button("Examples: Fourier optics (4f)", |ui| {
                    for p in FourierPreset::ALL {
                        if ui.button(p.label()).clicked() {
                            self.four.load_preset(p);
                            self.mode = Mode::Fourier;
                            ui.close();
                        }
                    }
                });
                let saved = configs::list();
                ui.add_enabled_ui(!saved.is_empty(), |ui| {
                    ui.menu_button("My configurations", |ui| {
                        for entry in &saved {
                            if ui.button(&entry.name).clicked() {
                                self.load_config(&entry.path.clone());
                                ui.close();
                            }
                        }
                    });
                });
                ui.separator();
                if ui.button("Save / manage configurations…   Cmd+S").clicked() {
                    self.open_config_window();
                    ui.close();
                }
                ui.separator();
                if ui.button("Empty lawn").clicked() {
                    self.scene = Scene::empty();
                    self.selected = None;
                    ui.close();
                }
            });
            ui.menu_button("Rendering", |ui| {
                ui.add(egui::Slider::new(&mut self.settings.render_scale, 0.25..=2.0).text("resolution (px per point)"));
                ui.add(egui::Slider::new(&mut self.settings.spp, 1..=16).text("samples per frame"));
                ui.add(egui::Slider::new(&mut self.settings.max_samples, 16..=8192).logarithmic(true).text("max samples"));
                ui.separator();
                ui.add(egui::Slider::new(&mut self.scene.sun_elevation_deg, 5.0..=85.0).text("sun elevation"));
                ui.add(egui::Slider::new(&mut self.scene.sun_azimuth_deg, 0.0..=360.0).text("sun azimuth"));
            });
            ui.menu_button("Help", |ui| {
                ui.set_max_width(420.0);
                ui.label("Everything is ray traced on the GPU: ideal thin lenses, apertures, and glass prisms with Snell's law and wavelength-dependent refraction.");
                ui.label("• EYE: pixels are points on the retina. Rays go through the pupil, the eye lens (which accommodates) and all optics on the bench.");
                ui.label("• SCREEN: every point of the screen collects the light that comes through the lens, aperture or prism in front of it, like in a dark room.");
                ui.label("• FOURIER OPTICS (4f): switch in the menu bar. Scalar wave optics: input field (colour = phase, brightness = amplitude, plus the wavefront along the centre line), Fraunhofer pattern in the Fourier plane with a filter, filtered image, and the propagation through the whole system. Scroll in a panel to zoom, drag in the Fourier plane to size the filter.");
                ui.label("• Save scenes (with notes for students) via Scene → Save / manage configurations. They are JSON files in the folder 'Optics Bench configs' next to the app; drop one onto the window to open it.");
                ui.label("• Stickman: red = his left arm and leg, blue = his right. The F-sign and the apples on the tree are asymmetric too, so you can see how images are flipped.");
                ui.label("• Top view: red/blue ray fans start at the left/right edge of each object (select an object to show only its rays).");
            });
            ui.separator();
            ui.selectable_value(&mut self.mode, Mode::Ray, "Ray optics bench");
            ui.selectable_value(&mut self.mode, Mode::Fourier, "Fourier optics (4f)");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.weak("Optics Bench");
            });
        });
    }

    fn handle_keys(&mut self, ctx: &egui::Context) {
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::S)) {
            self.open_config_window();
        }
        if ctx.egui_wants_keyboard_input() || self.mode != Mode::Ray {
            return;
        }
        let Some(id) = self.selected else { return };
        let (del, dx, dz, shift) = ctx.input(|i| {
            let del = i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace);
            let mut dx = 0.0;
            let mut dz = 0.0;
            if i.key_pressed(egui::Key::ArrowLeft) {
                dx -= 1.0;
            }
            if i.key_pressed(egui::Key::ArrowRight) {
                dx += 1.0;
            }
            if i.key_pressed(egui::Key::ArrowUp) {
                dz -= 1.0;
            }
            if i.key_pressed(egui::Key::ArrowDown) {
                dz += 1.0;
            }
            (del, dx, dz, i.modifiers.shift)
        });
        if del {
            self.scene.remove(id);
            self.selected = None;
            return;
        }
        if dx != 0.0 || dz != 0.0 {
            let step = if shift { 0.1 } else { 0.01 };
            if let Some(e) = self.scene.get_mut(id) {
                e.pos += Vec2::new(dx, dz) * step;
            }
        }
    }
}

impl OpticsApp {
    fn open_config_window(&mut self) {
        self.configs.open = true;
        self.configs.focus_name = true;
        self.configs.entries = configs::list();
        self.configs.confirm_delete = None;
    }

    fn saved_view(&self) -> SavedView {
        let o = &self.orbit;
        SavedView {
            top_center: [self.top.center.x, self.top.center.y],
            top_zoom: self.top.zoom,
            orbit: [o.target.x, o.target.y, o.target.z, o.yaw, o.pitch, o.dist],
            bench_3d: self.settings.bench_3d,
        }
    }

    fn save_config(&mut self) {
        let name = self.configs.name.trim().to_string();
        let cfg = ConfigFile {
            name: name.clone(),
            scene: self.scene.clone(),
            view: Some(self.saved_view()),
            mode: self.mode,
            fourier: Some(self.four.params.clone()),
            fourier_notes: self.four.notes.clone(),
        };
        self.configs.message = Some(match configs::save(&cfg) {
            Ok(path) => (format!("Saved '{name}' ({})", path.file_name().unwrap_or_default().to_string_lossy()), true),
            Err(e) => (format!("Could not save: {e}"), false),
        });
        self.configs.entries = configs::list();
    }

    fn load_config(&mut self, path: &std::path::Path) {
        match configs::load(path) {
            Ok(cfg) => {
                self.scene = cfg.scene;
                self.selected = None;
                match cfg.view {
                    Some(v) => {
                        self.top = TopCam { center: Vec2::from(v.top_center), zoom: v.top_zoom };
                        let o = v.orbit;
                        self.orbit = Orbit { target: Vec3::new(o[0], o[1], o[2]), yaw: o[3], pitch: o[4], dist: o[5] };
                        self.settings.bench_3d = v.bench_3d;
                        self.pending_fit = false;
                    }
                    None => self.pending_fit = true,
                }
                self.mode = cfg.mode;
                if let Some(fp) = cfg.fourier {
                    self.four.params = fp;
                    self.four.notes = cfg.fourier_notes;
                }
                self.configs.message = Some((format!("Opened '{}'", cfg.name), true));
                self.configs.name = cfg.name;
            }
            Err(e) => self.configs.message = Some((format!("Could not open: {e}"), false)),
        }
    }

    fn config_window_ui(&mut self, ctx: &egui::Context) {
        if !self.configs.open {
            return;
        }
        let mut open = true;
        let mut to_load: Option<std::path::PathBuf> = None;
        let mut to_delete: Option<std::path::PathBuf> = None;
        let mut save = false;
        egui::Window::new("Configurations")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_width(460.0)
            .default_pos(ctx.content_rect().center() - egui::vec2(230.0, 200.0))
            .show(ctx, |ui| {
                ui.strong("Save the current scene");
                ui.horizontal(|ui| {
                    ui.label("Name");
                    let r = ui.add(
                        egui::TextEdit::singleline(&mut self.configs.name)
                            .hint_text("e.g. PS03 exercise 2")
                            .desired_width(260.0),
                    );
                    if self.configs.focus_name {
                        r.request_focus();
                        self.configs.focus_name = false;
                    }
                    let valid = !self.configs.name.trim().is_empty();
                    let exists = valid && configs::path_for(&self.configs.name).exists();
                    let enter = r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    let clicked = ui.add_enabled(valid, egui::Button::new(if exists { "Overwrite" } else { "Save" })).clicked();
                    save = valid && (clicked || enter);
                });
                ui.label("Notes, shown when the scene is opened (e.g. a task for the students):");
                let notes = if self.mode == Mode::Fourier { &mut self.four.notes } else { &mut self.scene.notes };
                ui.add(egui::TextEdit::multiline(notes).desired_rows(3).desired_width(f32::INFINITY));
                if let Some((msg, ok)) = &self.configs.message {
                    let col = if *ok { Color32::from_rgb(110, 190, 110) } else { Color32::from_rgb(230, 110, 90) };
                    ui.colored_label(col, msg);
                }
                ui.separator();
                ui.horizontal(|ui| {
                    ui.strong("Saved configurations");
                    if ui.small_button("Show in Finder").clicked() {
                        let dir = configs::dir();
                        let _ = std::fs::create_dir_all(&dir);
                        let _ = std::process::Command::new("open").arg(&dir).spawn();
                    }
                    if ui.small_button("Refresh").clicked() {
                        self.configs.entries = configs::list();
                    }
                });
                ui.weak(configs::dir().display().to_string());
                egui::ScrollArea::vertical().max_height(260.0).show(ui, |ui| {
                    if self.configs.entries.is_empty() {
                        ui.weak("Nothing saved yet.");
                    }
                    for entry in &self.configs.entries {
                        ui.horizontal(|ui| {
                            ui.label(&entry.name);
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if self.configs.confirm_delete.as_ref() == Some(&entry.path) {
                                    if ui.small_button("No").clicked() {
                                        self.configs.confirm_delete = None;
                                    }
                                    if ui.small_button("Yes, delete").clicked() {
                                        to_delete = Some(entry.path.clone());
                                    }
                                    ui.colored_label(Color32::from_rgb(230, 110, 90), "Delete?");
                                } else {
                                    if ui.small_button("Delete").clicked() {
                                        self.configs.confirm_delete = Some(entry.path.clone());
                                    }
                                    if ui.small_button("Open").clicked() {
                                        to_load = Some(entry.path.clone());
                                    }
                                }
                            });
                        });
                    }
                });
                ui.add_space(4.0);
                ui.weak("Tip: drop a .json configuration onto the window to open it.");
            });
        if save {
            self.save_config();
        }
        if let Some(p) = to_load {
            self.load_config(&p);
        }
        if let Some(p) = to_delete {
            self.configs.message = Some(match configs::delete(&p) {
                Ok(()) => ("Deleted.".into(), true),
                Err(e) => (format!("Could not delete: {e}"), false),
            });
            self.configs.confirm_delete = None;
            self.configs.entries = configs::list();
        }
        if !open {
            self.configs.open = false;
        }
    }

    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped: Vec<std::path::PathBuf> =
            ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).collect());
        for p in dropped.iter().filter(|p| p.extension().is_some_and(|e| e == "json")) {
            self.load_config(p);
        }
    }
}

impl eframe::App for OpticsApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let rs = frame.wgpu_render_state().cloned();
        if self.gpu.is_none() {
            if let Some(rs) = &rs {
                self.gpu = Some(Gpu::new(rs));
            }
        }
        let ctx = ui.ctx().clone();
        self.handle_keys(&ctx);
        self.world = build_world(&self.scene);
        self.update_focus();

        let mut want = [false; 3];
        egui::Panel::top("menu").show(ui, |ui| self.menu_ui(ui));
        if self.mode == Mode::Fourier {
            let win_h = ctx.content_rect().height();
            self.four.update(&ctx);
            egui::Panel::top("f_planes")
                .resizable(true)
                .default_size(win_h * 0.48)
                .min_size(200.0)
                .max_size(win_h * 0.7)
                .show(ui, |ui| self.four.planes_ui(ui));
            egui::Panel::bottom("f_controls")
                .resizable(true)
                .default_size(245.0)
                .min_size(120.0)
                .max_size(win_h * 0.45)
                .show(ui, |ui| self.four.controls_ui(ui));
            egui::CentralPanel::default().show(ui, |ui| self.four.side_ui(ui));
            self.config_window_ui(&ctx);
            self.handle_dropped_files(&ctx);
            self.debug_shot(&ctx);
            return;
        }
        let win_h = ctx.content_rect().height();
        egui::Panel::top("views")
            .resizable(true)
            .default_size(win_h * 0.44)
            .min_size(180.0)
            .max_size((win_h * 0.62).max(200.0))
            .show(ui, |ui| self.views_ui(ui, rs.as_ref(), &mut want));
        egui::Panel::bottom("toolbox")
            .resizable(true)
            .default_size(200.0)
            .min_size(120.0)
            .max_size((win_h * 0.35).max(150.0))
            .show(ui, |ui| self.toolbox_ui(ui));
        egui::CentralPanel::default().show(ui, |ui| self.bench_ui(ui, rs.as_ref(), &mut want));
        self.config_window_ui(&ctx);
        self.handle_dropped_files(&ctx);

        // the UI may have changed the scene: rebuild before rendering
        self.world = build_world(&self.scene);
        let (globals, views) = self.frame_data(want);
        if let (Some(gpu), Some(rs)) = (self.gpu.as_mut(), rs.as_ref()) {
            let lenses = self.world.gpu_lenses();
            let prisms = self.world.gpu_prisms();
            let data = FrameData {
                prims: &self.world.prims,
                objs: &self.world.objs,
                lenses: &lenses,
                prisms: &prisms,
                globals,
                views,
                spp: self.settings.spp,
                max_samples: self.settings.max_samples,
            };
            if gpu.render(rs, &data) {
                ctx.request_repaint();
            }
        }
        self.debug_shot(&ctx);
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if self.shot.is_some() {
            return;
        }
        eframe::set_value(
            storage,
            eframe::APP_KEY,
            &Persist {
                scene: self.scene.clone(),
                settings: self.settings.clone(),
                top: self.top,
                orbit: self.orbit,
                mode: self.mode,
                fourier: Some((self.four.params.clone(), self.four.notes.clone())),
            },
        );
    }
}

impl OpticsApp {
    fn debug_shot(&mut self, ctx: &egui::Context) {
        let Some(shot) = &mut self.shot else { return };
        shot.count += 1;
        if shot.count == 1 {
            ctx.send_viewport_cmd(egui::ViewportCommand::WindowLevel(egui::WindowLevel::AlwaysOnTop));
            ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
        }
        ctx.request_repaint();
        let t = shot.start.elapsed().as_secs_f32();
        if t > shot.secs && shot.count % 10 == 0 {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
        }
        let image = ctx.input(|i| {
            i.raw.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        if let Some(img) = image {
            eprintln!(
                "screenshot after {} frames, {:.1}s, samples {:?}",
                shot.count,
                shot.start.elapsed().as_secs_f32(),
                self.gpu.as_ref().map(|g| [g.samples(T_SCREEN), g.samples(T_EYE), g.samples(T_OVERVIEW)])
            );
            let bytes: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
            let file = std::fs::File::create(&shot.path).expect("screenshot file");
            let mut enc = png::Encoder::new(std::io::BufWriter::new(file), img.size[0] as u32, img.size[1] as u32);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            enc.write_header().and_then(|mut w| w.write_image_data(&bytes)).expect("png");
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}

// ---------------------------------------------------------------- helpers

const STATUS_H: f32 = 22.0;

/// Splits the remaining space into an image area and a one-line status bar
/// without letting the content grow the (resizable) panel.
fn split_status(ui: &mut egui::Ui) -> (Rect, Rect) {
    let full = ui.available_rect_before_wrap();
    let h = (full.height() - STATUS_H).max(40.0);
    let rect = Rect::from_min_size(full.min, egui::vec2(full.width(), h));
    let status = Rect::from_min_max(egui::pos2(full.min.x, full.min.y + h), egui::pos2(full.max.x, full.min.y + h + STATUS_H));
    ui.allocate_rect(status, Sense::hover());
    (rect, status)
}

fn status_text(ui: &egui::Ui, rect: Rect, text: &str, color: Color32) {
    ui.painter_at(rect).text(rect.left_center() + egui::vec2(2.0, 0.0), Align2::LEFT_CENTER, text, FontId::proportional(13.0), color);
}

fn v4(v: Vec3) -> [f32; 4] {
    [v.x, v.y, v.z, 0.0]
}

/// The opening a screen looks through: the nearest lens, aperture or prism in front of it.
#[derive(Clone, Copy, Debug)]
pub struct Entrance {
    pub center: Vec3,
    pub axis: Vec3,
    /// radius of the disk that is sampled
    pub radius: f32,
    /// brightness normalisation (1 / fraction of the sampled rays that get through)
    pub mult: f32,
}

/// Finds the opening in front of the screen and the part of it through which
/// light actually arrives (other lenses, apertures and the tube cut the bundle).
/// Sampling only that part keeps the image bright and noise free.
pub fn compute_entrance(world: &World, train: u32, tube: bool) -> Option<Entrance> {
    let s = world.screen?;
    let o = s.center + s.normal * s.ht;
    let in_front = |c: Vec3| {
        let d = c - o;
        d.length() > 1e-3 && d.normalize().dot(s.normal) > 0.26
    };
    let mut best: Option<(f32, Entrance)> = None;
    for l in world.lenses.iter().filter(|l| !l.obstruction) {
        let dist = (l.center - o).length();
        if in_front(l.center) && best.as_ref().is_none_or(|(bd, _)| dist < *bd) {
            best = Some((dist, Entrance { center: l.center, axis: l.axis, radius: l.radius, mult: 1.0 }));
        }
    }
    for p in &world.prisms {
        let dist = (p.center - o).length();
        if in_front(p.center) && best.as_ref().is_none_or(|(bd, _)| dist < *bd) {
            let axis = (o - p.center).normalize();
            best = Some((dist, Entrance { center: p.center, axis, radius: p.inradius.min(p.half_h) * 0.9, mult: 1.0 }));
        }
    }
    let base = best?.1;
    let e1 = base.axis.cross(Vec3::Y).normalize_or(Vec3::X);
    let e2 = e1.cross(base.axis);
    let disk = |rho: f32, phi: f32| base.center + e1 * (rho * phi.cos()) + e2 * (rho * phi.sin());
    let passes = |p: Vec3, q: Vec3| {
        let path = world.trace_path_ex(p, (q - p).normalize(), 1000.0, true, None, LAMBDA_D);
        if matches!(path.end, Surf::Rim(_)) {
            return false;
        }
        let got = path
            .events
            .iter()
            .filter_map(|e| match e {
                Event::Lens(i) | Event::Stop(i) if *i < 32 => Some(1u32 << i),
                _ => None,
            })
            .fold(0, |a, b| a | b);
        !tube || got & train == train
    };
    // how far out on the opening does light still get through, seen from points all over the screen?
    let mut max_r: f32 = 0.0;
    for (u, v) in [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5)] {
        let p = o + s.right * (u * s.hw) + s.up * (v * s.hh);
        for i in 0..=8 {
            let rho = base.radius * i as f32 / 8.0;
            for j in 0..(if i == 0 { 1 } else { 12 }) {
                let phi = j as f32 / 12.0 * std::f32::consts::TAU;
                if rho > max_r && passes(p, disk(rho, phi)) {
                    max_r = rho;
                }
            }
        }
    }
    if max_r == 0.0 {
        return Some(base);
    }
    let radius = (max_r + base.radius / 8.0 + 0.005).min(base.radius);
    // fraction of rays from the screen centre that pass, within the new radius
    let (mut n, mut k) = (0, 0);
    for i in 0..10 {
        for j in 0..12 {
            let rho = radius * ((i as f32 + 0.5) / 10.0).sqrt();
            let phi = (j as f32 + 0.5 * (i % 2) as f32) / 12.0 * std::f32::consts::TAU;
            n += 1;
            if passes(o, disk(rho, phi)) {
                k += 1;
            }
        }
    }
    let frac = if k > 0 { k as f32 / n as f32 } else { 1.0 };
    Some(Entrance { radius, mult: 1.0 / frac.max(0.03), ..base })
}

/// Albedo (linear, packed) -> a colour for the top view drawing.
fn top_color(packed: u32) -> Color32 {
    let ch = |s: u32| {
        let lin = ((packed >> s) & 255) as f32 / 255.0;
        let lit = (lin * 2.6).min(1.0);
        (lit.powf(1.0 / 2.2) * 255.0) as u8
    };
    Color32::from_rgb(ch(0), ch(8), ch(16))
}

/// Draws a lens cross section: `r` along the lens, `f` along the axis (screen space unit vectors).
fn draw_lens_shape(painter: &egui::Painter, c: Pos2, r: egui::Vec2, f: egui::Vec2, half_len: f32, focal: f32, px_per_m: f32) {
    let n = 24;
    let thick = ((0.1 / focal.abs().max(0.05)).sqrt() * 0.07 * px_per_m).clamp(3.0, (0.45 * half_len).max(3.0));
    let thin = (thick * 0.25).max(1.5);
    let mut top = vec![];
    let mut bot = vec![];
    for i in 0..=n {
        let s = i as f32 / n as f32 * 2.0 - 1.0;
        let w = if focal >= 0.0 { thin + (thick - thin) * (1.0 - s * s) } else { thin + (thick - thin) * s * s };
        let p = c + r * (s * half_len);
        top.push(p + f * (w * 0.5));
        bot.push(p - f * (w * 0.5));
    }
    let fill = Color32::from_rgba_unmultiplied(170, 220, 255, 150);
    let mut mesh = egui::Mesh::default();
    for i in 0..=n {
        mesh.colored_vertex(top[i], fill);
        mesh.colored_vertex(bot[i], fill);
    }
    for i in 0..n as u32 {
        let a = 2 * i;
        mesh.add_triangle(a, a + 1, a + 2);
        mesh.add_triangle(a + 1, a + 3, a + 2);
    }
    painter.add(mesh);
    let mut outline = top.clone();
    outline.extend(bot.iter().rev());
    painter.add(egui::Shape::closed_line(outline, Stroke::new(1.2, Color32::from_rgb(20, 60, 110))));
    // holder at both ends
    for s in [-1.0, 1.0] {
        let p = c + r * (s * half_len);
        painter.line_segment([p, p + r * (s * 4.0)], Stroke::new(3.0, Color32::from_gray(30)));
    }
}

fn draw_tool_icon(painter: &egui::Painter, rect: Rect, tool: Tool) {
    let c = rect.center();
    let s = rect.width() / 52.0;
    let ink = Color32::from_gray(25);
    let red = Color32::from_rgb(220, 60, 50);
    let blue = Color32::from_rgb(60, 110, 230);
    let st = |w: f32, c: Color32| Stroke::new(w * s, c);
    match tool {
        Tool::Stickman => {
            let p = |x: f32, y: f32| c + egui::vec2(x, y) * s;
            painter.line_segment([p(0.0, -10.0), p(0.0, 6.0)], st(3.0, ink));
            painter.line_segment([p(0.0, -7.0), p(-9.0, -12.0)], st(3.0, red));
            painter.line_segment([p(-9.0, -12.0), p(-11.0, -22.0)], st(3.0, red));
            painter.line_segment([p(0.0, -7.0), p(10.0, 4.0)], st(3.0, blue));
            painter.line_segment([p(0.0, 6.0), p(-7.0, 22.0)], st(3.0, red));
            painter.line_segment([p(0.0, 6.0), p(7.0, 22.0)], st(3.0, blue));
            painter.circle(p(0.0, -16.0), 6.0 * s, Color32::from_rgb(235, 190, 140), st(1.0, ink));
            painter.rect_filled(Rect::from_center_size(p(0.0, -22.5), egui::vec2(12.0, 3.5) * s), 1.0, Color32::from_rgb(235, 190, 30));
        }
        Tool::Tree => {
            let p = |x: f32, y: f32| c + egui::vec2(x, y) * s;
            painter.line_segment([p(0.0, 24.0), p(0.0, 0.0)], st(5.0, Color32::from_rgb(110, 70, 35)));
            painter.circle_filled(p(-8.0, -4.0), 11.0 * s, Color32::from_rgb(50, 130, 45));
            painter.circle_filled(p(8.0, -6.0), 11.0 * s, Color32::from_rgb(40, 110, 40));
            painter.circle_filled(p(0.0, -14.0), 12.0 * s, Color32::from_rgb(60, 150, 50));
            for (x, y) in [(9.0, -2.0), (12.0, -12.0), (4.0, -18.0)] {
                painter.circle_filled(p(x, y), 2.5 * s, red);
            }
        }
        Tool::Sign => {
            let p = |x: f32, y: f32| c + egui::vec2(x, y) * s;
            painter.line_segment([p(0.0, 24.0), p(0.0, 6.0)], st(3.0, Color32::GRAY));
            let board = Rect::from_center_size(p(0.0, -6.0), egui::vec2(30.0, 34.0) * s);
            painter.rect_filled(board, 2.0, Color32::from_gray(225));
            painter.text(board.center(), Align2::CENTER_CENTER, "F", FontId::proportional(26.0 * s), red);
        }
        Tool::SlitLamp => {
            let r = Rect::from_center_size(c, egui::vec2(40.0, 36.0) * s);
            painter.rect_filled(r, 2.0, Color32::from_gray(15));
            painter.rect_filled(Rect::from_center_size(c, egui::vec2(6.0, 24.0) * s), 1.0, Color32::from_rgba_unmultiplied(255, 255, 220, 60));
            painter.rect_filled(Rect::from_center_size(c, egui::vec2(2.5, 22.0) * s), 0.0, Color32::WHITE);
            for x in [-14.0, 14.0] {
                painter.line_segment([c + egui::vec2(x, 18.0) * s, c + egui::vec2(x, 26.0) * s], st(2.0, Color32::GRAY));
            }
        }
        Tool::Checker => {
            let n = 4;
            let cell = 9.0 * s;
            let origin = c - egui::vec2(cell * 2.0, cell * 2.0 + 3.0 * s);
            for i in 0..n {
                for j in 0..n {
                    let col = if (i + j) % 2 == 0 { Color32::from_gray(235) } else { Color32::from_gray(20) };
                    let r = Rect::from_min_size(origin + egui::vec2(i as f32 * cell, j as f32 * cell), egui::vec2(cell, cell));
                    painter.rect_filled(r, 0.0, col);
                }
            }
            painter.rect_stroke(Rect::from_min_size(origin, egui::vec2(cell * 4.0, cell * 4.0)), 0.0, st(1.0, Color32::GRAY), StrokeKind::Outside);
        }
        Tool::Aperture => {
            painter.circle_filled(c, 22.0 * s, Color32::from_gray(35));
            painter.circle_filled(c, 7.0 * s, Color32::from_rgb(200, 225, 245));
            painter.circle_stroke(c, 22.0 * s, st(1.0, Color32::from_gray(90)));
        }
        Tool::Stop => {
            // a beam (light) with its centre blocked by a dark disk
            painter.circle_filled(c, 22.0 * s, Color32::from_rgb(200, 225, 245));
            painter.circle_filled(c, 13.0 * s, Color32::from_gray(35));
            painter.line_segment([c + egui::vec2(0.0, 13.0) * s, c + egui::vec2(0.0, 26.0) * s], st(2.0, Color32::from_gray(35)));
        }
        Tool::Prism => {
            let p = |x: f32, y: f32| c + egui::vec2(x, y) * s;
            let tri = vec![p(-6.0, -18.0), p(10.0, 12.0), p(-22.0, 12.0)];
            // a white beam in, a spectrum out
            painter.line_segment([p(-26.0, -8.0), p(-12.0, -2.0)], st(2.0, Color32::WHITE));
            for (i, col) in [Color32::from_rgb(230, 50, 40), Color32::from_rgb(240, 200, 40), Color32::from_rgb(60, 200, 80), Color32::from_rgb(60, 110, 240)]
                .iter()
                .enumerate()
            {
                let dy = i as f32 * 3.0;
                painter.line_segment([p(2.0, -1.0), p(24.0, 4.0 + dy * 1.6)], st(2.0, *col));
            }
            painter.add(egui::Shape::convex_polygon(tri, Color32::from_rgba_unmultiplied(190, 225, 255, 170), st(1.2, Color32::from_rgb(20, 60, 110))));
        }
        Tool::Converging | Tool::Diverging => {
            let focal = if tool == Tool::Converging { 0.2 } else { -0.2 };
            draw_lens_shape(painter, c, egui::vec2(0.0, 1.0), egui::vec2(1.0, 0.0), 22.0 * s, focal, 120.0 * s);
        }
        Tool::Screen => {
            let r = Rect::from_center_size(c, egui::vec2(34.0, 40.0) * s);
            painter.rect_filled(r.translate(egui::vec2(3.0, 3.0) * s), 1.0, Color32::from_gray(60));
            painter.rect_filled(r, 1.0, Color32::from_gray(235));
            painter.line_segment([c + egui::vec2(0.0, 20.0) * s, c + egui::vec2(0.0, 26.0) * s], st(3.0, Color32::GRAY));
            // a small upside-down stickman as the "image"
            let p = |x: f32, y: f32| c + egui::vec2(x, -y) * s * 0.6;
            painter.line_segment([p(0.0, -10.0), p(0.0, 6.0)], st(1.5, ink));
            painter.line_segment([p(0.0, 6.0), p(-7.0, 22.0)], st(1.5, blue));
            painter.line_segment([p(0.0, 6.0), p(7.0, 22.0)], st(1.5, red));
            painter.circle_filled(p(0.0, -16.0), 3.5 * s, Color32::from_rgb(235, 190, 140));
        }
        Tool::Eye => {
            let mut pts = vec![];
            for i in 0..=16 {
                let t = i as f32 / 16.0 * std::f32::consts::TAU;
                pts.push(c + egui::vec2(t.cos() * 22.0, t.sin() * 11.0 * (1.0 - 0.25 * t.cos().abs())) * s);
            }
            painter.add(egui::Shape::convex_polygon(pts, Color32::WHITE, st(1.5, ink)));
            painter.circle_filled(c, 9.0 * s, Color32::from_rgb(40, 110, 200));
            painter.circle_filled(c, 4.0 * s, Color32::BLACK);
            painter.circle_filled(c + egui::vec2(-2.5, -2.5) * s, 1.5 * s, Color32::WHITE);
        }
    }
}

fn draw_ruler(painter: &egui::Painter, rect: Rect, width_m: f32, height_m: f32) {
    let col = Color32::from_gray(170);
    let font = FontId::monospace(9.0);
    let px_per_m = rect.width() / width_m;
    let step = if px_per_m * 0.1 > 30.0 { 0.1 } else if px_per_m * 0.2 > 30.0 { 0.2 } else { 0.5 };
    let minor = step / 5.0;
    // bottom edge, x measured from the centre
    let n = (width_m * 0.5 / minor).floor() as i32;
    for i in -n..=n {
        let x = rect.center().x + i as f32 * minor * px_per_m;
        let major = i % 5 == 0;
        let len = if major { 7.0 } else { 3.5 };
        painter.line_segment([egui::pos2(x, rect.bottom() + 2.0), egui::pos2(x, rect.bottom() + 2.0 + len)], Stroke::new(1.0, col));
        if major {
            painter.text(
                egui::pos2(x, rect.bottom() + 10.0),
                Align2::CENTER_TOP,
                format!("{:.0}", i as f32 * minor * 100.0),
                font.clone(),
                col,
            );
        }
    }
    let n = (height_m * 0.5 / minor).floor() as i32;
    for i in -n..=n {
        let y = rect.center().y - i as f32 * minor * px_per_m;
        let major = i % 5 == 0;
        let len = if major { 7.0 } else { 3.5 };
        painter.line_segment([egui::pos2(rect.left() - 2.0, y), egui::pos2(rect.left() - 2.0 - len, y)], Stroke::new(1.0, col));
        if major {
            painter.text(
                egui::pos2(rect.left() - 11.0, y),
                Align2::RIGHT_CENTER,
                format!("{:.0}", i as f32 * minor * 100.0),
                font.clone(),
                col,
            );
        }
    }
    painter.text(rect.right_bottom() + egui::vec2(0.0, 10.0), Align2::RIGHT_TOP, "cm", font, col);
}

