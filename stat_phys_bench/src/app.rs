//! The app: one bench per example of the notes "Partition functions: five
//! worked examples", switched in the menu bar.

use eframe::egui;

use crate::cavity::CavityPreset;
use crate::cavity_ui::CavityUi;
use crate::gas::{AtomPreset, GasPreset};
use crate::gas_ui::GasUi;
use crate::levels::LevelPreset;
use crate::levels_ui::LevelsUi;
use crate::oscillators::OscPreset;
use crate::oscillators_ui::OscUi;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tab {
    Levels,
    Gas,
    Atoms,
    Classical,
    Quantum,
    Cavity,
}

impl Tab {
    pub const ALL: [Tab; 6] = [Tab::Levels, Tab::Gas, Tab::Atoms, Tab::Classical, Tab::Quantum, Tab::Cavity];

    pub fn label(self) -> &'static str {
        match self {
            Tab::Levels => "1 Energy levels",
            Tab::Gas => "2 Gas in a box",
            Tab::Atoms => "3 Two-level atoms",
            Tab::Classical => "4 Classical oscillators",
            Tab::Quantum => "5 Quantum oscillators",
            Tab::Cavity => "6 Light in a box",
        }
    }

    /// the name in links (`…/stat-phys-bench/#quantum`)
    pub fn slug(self) -> &'static str {
        match self {
            Tab::Levels => "levels",
            Tab::Gas => "gas",
            Tab::Atoms => "atoms",
            Tab::Classical => "classical",
            Tab::Quantum => "quantum",
            Tab::Cavity => "cavity",
        }
    }

    pub fn from_slug(s: &str) -> Option<Tab> {
        Tab::ALL.into_iter().find(|t| t.slug() == s)
    }
}

pub struct StatApp {
    tab: Tab,
    levels: LevelsUi,
    gas: GasUi,
    atoms: GasUi,
    classical: OscUi,
    quantum: OscUi,
    cavity: CavityUi,
    #[cfg(target_arch = "wasm32")]
    url_tab: Option<Tab>,
    #[cfg(not(target_arch = "wasm32"))]
    shot: Option<Shot>,
}

/// a screenshot taken after a few seconds (STAT_SHOT=out.png), for checking the drawing
#[cfg(not(target_arch = "wasm32"))]
struct Shot {
    path: String,
    secs: f32,
    start: std::time::Instant,
    frames: u64,
}

impl StatApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // egui's text font lacks ⟨ ⟩ ∝ → and ⁻; Hack (bundled with egui) has them
        let mut fonts = egui::FontDefinitions::default();
        fonts.families.entry(egui::FontFamily::Proportional).or_default().push("Hack".to_owned());
        cc.egui_ctx.set_fonts(fonts);
        cc.egui_ctx.set_theme(egui::Theme::Dark);
        let mut app = StatApp {
            tab: Tab::Levels,
            levels: LevelsUi::default(),
            gas: GasUi::new(false),
            atoms: GasUi::new(true),
            classical: OscUi::new(false),
            quantum: OscUi::new(true),
            cavity: CavityUi::default(),
            #[cfg(target_arch = "wasm32")]
            url_tab: None,
            #[cfg(not(target_arch = "wasm32"))]
            shot: std::env::var("STAT_SHOT").ok().map(|path| Shot {
                path,
                secs: std::env::var("STAT_SHOT_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(4.0),
                start: std::time::Instant::now(),
                frames: 0,
            }),
        };
        // a link to one bench (`#quantum`) or STAT_BENCH=quantum opens it
        #[cfg(target_arch = "wasm32")]
        let wanted = crate::web::url_hash();
        #[cfg(not(target_arch = "wasm32"))]
        let wanted = std::env::var("STAT_BENCH").ok();
        if let Some(t) = wanted.and_then(|h| Tab::from_slug(&h)) {
            app.tab = t;
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(k) = std::env::var("STAT_EXAMPLE").ok().and_then(|s| s.parse::<usize>().ok()) {
            app.load_example(app.tab, k);
        }
        app
    }

    /// Loads example k (counted from 0) of a bench.
    pub fn load_example(&mut self, tab: Tab, k: usize) {
        let pick = |n: usize| k.min(n - 1);
        match tab {
            Tab::Levels => self.levels.load_preset(LevelPreset::ALL[pick(LevelPreset::ALL.len())]),
            Tab::Gas => self.gas.load_gas_preset(GasPreset::ALL[pick(GasPreset::ALL.len())]),
            Tab::Atoms => self.atoms.load_atom_preset(AtomPreset::ALL[pick(AtomPreset::ALL.len())]),
            Tab::Classical => self.classical.load_preset(OscPreset::CLASSICAL[pick(OscPreset::CLASSICAL.len())]),
            Tab::Quantum => self.quantum.load_preset(OscPreset::QUANTUM[pick(OscPreset::QUANTUM.len())]),
            Tab::Cavity => self.cavity.load_preset(CavityPreset::ALL[pick(CavityPreset::ALL.len())]),
        }
    }

    fn menu_ui(&mut self, ui: &mut egui::Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("Help", |ui| {
                ui.set_max_width(440.0);
                ui.label("Six benches: one for each example of the notes 'Partition functions: five worked examples', and light in a box (lecture notes 4.6: cavity modes, Rayleigh–Jeans, Planck). Each runs a small simulation in contact with a heat bath (or isolated), shows the distribution it reaches, and puts the formulas of the notes next to it.");
                ui.label("• Colours: white = the formula from the notes, blue = the simulation, orange = the current temperature, grey = the classical result for comparison.");
                ui.label("• Energies and k_BT are measured in the same unit, which is arbitrary. Masses are in units of a reference mass, speeds in units of √(energy/mass).");
                ui.label("• 'histograms' sets how long the histograms remember: only now, about 5 s, or about a minute. Few particles need a long memory.");
                ui.label("• C from the spread: in equilibrium the energy of one particle fluctuates by Var(E) = k_BT² C. The simulation's spread gives its heat capacity.");
                ui.label("• Each bench has a menu of examples, each with notes.");
            });
            ui.separator();
            for t in Tab::ALL {
                ui.selectable_value(&mut self.tab, t, t.label());
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.weak("Stat Phys Bench");
            });
        });
    }

    #[cfg(target_arch = "wasm32")]
    fn debug_shot(&mut self, _ctx: &egui::Context) {}

    #[cfg(not(target_arch = "wasm32"))]
    fn debug_shot(&mut self, ctx: &egui::Context) {
        let Some(shot) = &mut self.shot else { return };
        shot.frames += 1;
        ctx.request_repaint();
        if shot.start.elapsed().as_secs_f32() > shot.secs && shot.frames % 10 == 0 {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
        }
        let image = ctx.input(|i| {
            i.raw.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        if let Some(img) = image {
            let bytes: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
            let file = std::fs::File::create(&shot.path).expect("screenshot file");
            let mut enc = png::Encoder::new(std::io::BufWriter::new(file), img.size[0] as u32, img.size[1] as u32);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            enc.write_header().and_then(|mut w| w.write_image_data(&bytes)).expect("png");
            eprintln!("screenshot after {} frames", shot.frames);
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}

impl eframe::App for StatApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("menu").show(ui, |ui| self.menu_ui(ui));
        // keep the page address on the shown bench, so that it can be shared; a changed address switches the bench
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(t) = crate::web::url_hash().and_then(|h| Tab::from_slug(&h))
                && self.url_tab.is_some_and(|u| u != t)
            {
                self.tab = t;
                self.url_tab = Some(t);
            }
            if self.url_tab != Some(self.tab) {
                self.url_tab = Some(self.tab);
                crate::web::set_url_hash(self.tab.slug());
            }
        }
        match self.tab {
            Tab::Levels => self.levels.ui(ui),
            Tab::Gas => self.gas.ui(ui),
            Tab::Atoms => self.atoms.ui(ui),
            Tab::Classical => self.classical.ui(ui),
            Tab::Quantum => self.quantum.ui(ui),
            Tab::Cavity => self.cavity.ui(ui),
        }
        self.debug_shot(&ui.ctx().clone());
    }
}
