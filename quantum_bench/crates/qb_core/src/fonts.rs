//! Fonts: egui's defaults plus DejaVu Sans as a fallback for the physics
//! symbols they lack (⟨ ⟩, arrows, sub/superscripts like ₓ ₚ ⁿ, ⊗, …).
//! The same file is embedded on the desktop and on the web, so text renders
//! identically. License: `fonts/LICENSE_DEJAVU` (Bitstream Vera / public domain).

use std::sync::Arc;

use eframe::egui::{self, FontData, FontDefinitions, FontFamily};

static DEJAVU: &[u8] = include_bytes!("../fonts/DejaVuSans.ttf");

pub(crate) fn install(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    fonts.font_data.insert("DejaVuSans".into(), Arc::new(FontData::from_static(DEJAVU)));
    // after the primary font (Ubuntu / Hack), before the emoji fonts
    for family in [FontFamily::Proportional, FontFamily::Monospace] {
        if let Some(list) = fonts.families.get_mut(&family) {
            list.insert(1.min(list.len()), "DejaVuSans".into());
        }
    }
    ctx.set_fonts(fonts);
}
