//! Quantum Bench tab "Hydrogen": orbitals ψ_nlm, superpositions, 3D density
//! (PS11, PS12). STUB: to be built by the hydrogen agent.
//!
//! The placeholder scene already exercises the GPU path the hydrogen views
//! will need: a `qb_core::gpu::ShaderRect` fragment shader drawn inside egui
//! (here the phase-wheel demo, turning with the simulation time).

use qb_core::egui::{self, RichText};
use qb_core::gpu::PhaseWheelDemo;
use qb_core::{Ctx, Look, PresetInfo, Scene, SceneTab, Tab};

/// The "Hydrogen" tab (the app calls `HydrogenTab::new()`).
pub struct HydrogenTab;

impl HydrogenTab {
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> impl Tab {
        SceneTab::new("hydrogen", "Hydrogen").preset(
            PresetInfo::new("PS11 45 · Superposition of ψ₂₁₁ and ψ₂₁₋₁"),
            |_| Box::new(GpuCheckScene::default()),
        )
    }
}

/// Placeholder with a working GPU paint callback.
#[derive(Default)]
struct GpuCheckScene {
    t: f64,
    wheel: Option<PhaseWheelDemo>,
}

impl Scene for GpuCheckScene {
    fn time(&self) -> f64 {
        self.t
    }

    fn step(&mut self, dt: f64, _ctx: &mut Ctx) {
        self.t += dt;
    }

    fn reset(&mut self, _ctx: &mut Ctx) {
        self.t = 0.0;
    }

    fn central_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx) {
        let look = Look::of(ui);
        ui.add_space(look.px(30.0));
        ui.vertical_centered(|ui| {
            ui.label(RichText::new("Hydrogen: under construction").font(look.font(26.0)));
            ui.add_space(look.px(6.0));
            ui.label(
                RichText::new("GPU check: this wheel is drawn by a WGSL fragment shader (qb_core::gpu::ShaderRect) with qb_phase_color.")
                    .color(look.weak_text()),
            );
        });
        ui.add_space(look.px(10.0));
        let size = (ui.available_height() - look.px(20.0)).min(ui.available_width()).clamp(80.0, 360.0);
        let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), size), egui::Sense::hover());
        let rect = egui::Rect::from_center_size(rect.center(), egui::vec2(size, size));
        match (&self.wheel, ctx.render_state) {
            (Some(w), _) => w.paint(ui.painter(), rect, self.t as f32),
            (None, Some(rs)) => {
                let w = PhaseWheelDemo::new(rs);
                w.paint(ui.painter(), rect, self.t as f32);
                self.wheel = Some(w);
            }
            (None, None) => {
                ui.painter().text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "no wgpu render state",
                    look.font(14.0),
                    look.weak_text(),
                );
            }
        }
    }

    fn side_ui(&mut self, ui: &mut egui::Ui, _ctx: &mut Ctx) {
        ui.weak("No controls yet.");
    }

    fn notes(&self) -> String {
        "# Hydrogen (under construction)\nOrbitals ψ_nlm = R_nl Y_lm, superpositions such as (ψ₂₁₁ ± ψ₂₁₋₁)/√2, densities in 3D.".into()
    }
}
