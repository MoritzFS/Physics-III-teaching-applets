//! Readouts: "σₓ = 1.234" chips for live values, in one consistent style.

use eframe::egui::{self, Color32, RichText};

use crate::Look;

/// One labelled value, e.g. `readout(ui, "σₓ", &fmt_num(s, 3), None)`.
/// The label is weak, the value monospace and larger; `color` tints the value.
pub fn readout(ui: &mut egui::Ui, label: &str, value: &str, color: Option<Color32>) -> egui::Response {
    ui.horizontal(|ui| readout_parts(ui, label, value, color)).response
}

/// The two labels of a readout, added to the current layout.
fn readout_parts(ui: &mut egui::Ui, label: &str, value: &str, color: Option<Color32>) {
    let look = Look::of(ui);
    let gap = ui.spacing().item_spacing.x;
    ui.spacing_mut().item_spacing.x = look.px(4.0);
    ui.label(RichText::new(label).color(look.weak_text()).size(13.0 * look.font_scale()));
    ui.label(
        RichText::new(value)
            .monospace()
            .size(15.0 * look.font_scale())
            .color(color.unwrap_or(look.text())),
    );
    ui.spacing_mut().item_spacing.x = gap;
}

/// `v` with `decimals` decimals, never "-0.00" (values that round to zero print as 0).
pub fn fmt_num(v: f64, decimals: usize) -> String {
    let v = if v.abs() < 0.5 * 10f64.powi(-(decimals as i32)) { 0.0 } else { v };
    format!("{v:.decimals$}")
}

/// A row of readouts separated by spacing: `readout_row(ui, &[("⟨x⟩", "1.20".into(), None), …])`.
pub fn readout_row(ui: &mut egui::Ui, items: &[(&str, String, Option<Color32>)]) {
    let look = Look::of(ui);
    ui.horizontal_wrapped(|ui| {
        for (label, value, color) in items {
            readout_parts(ui, label, value, *color);
            ui.add_space(look.px(14.0));
        }
    });
}

/// "Repeat N" control used by all tabs: a button and a choice of N.
/// Returns Some(N) when clicked.
pub fn repeat_button(ui: &mut egui::Ui, id: impl std::hash::Hash + std::fmt::Debug, label: &str, n: &mut usize) -> Option<usize> {
    let mut clicked = None;
    ui.horizontal(|ui| {
        if ui.button(label).clicked() {
            clicked = Some(*n);
        }
        egui::ComboBox::from_id_salt(id)
            .selected_text(format!("N = {n}"))
            .width(90.0)
            .show_ui(ui, |ui| {
                for v in [10, 100, 1000, 10_000] {
                    ui.selectable_value(n, v, format!("N = {v}"));
                }
            });
    });
    clicked
}
