//! Controls shared by the benches: headings, example menus, the temperature
//! slider, play/pause, averaging, the table of numbers and the notes.

use std::ops::RangeInclusive;

use eframe::egui::{self, Color32, RichText};

use crate::plot::{self, fmt};
use crate::stats::Avg;

/// the bench's title and its key formula
pub fn heading(ui: &mut egui::Ui, title: &str, formula: &str) {
    ui.add_space(4.0);
    ui.label(RichText::new(title).heading().strong());
    ui.label(RichText::new(formula).color(plot::THEORY).size(15.0));
    ui.add_space(2.0);
}

/// A menu of examples; returns the one chosen.
pub fn examples<T: Copy + PartialEq>(ui: &mut egui::Ui, id: &str, current: T, all: &[T], label: impl Fn(T) -> &'static str) -> Option<T> {
    let mut chosen = None;
    ui.horizontal(|ui| {
        ui.label("Example:");
        egui::ComboBox::from_id_salt(id).selected_text(label(current)).width(ui.available_width() - 8.0).show_ui(ui, |ui| {
            for &p in all {
                if ui.selectable_label(p == current, label(p)).clicked() {
                    chosen = Some(p);
                }
            }
        });
    });
    chosen
}

pub fn kt_slider(ui: &mut egui::Ui, kt: &mut f64, range: RangeInclusive<f64>) -> bool {
    ui.add(egui::Slider::new(kt, range).logarithmic(true).text("k_BT").custom_formatter(|v, _| fmt(v))).changed()
}

/// ⏸/▶ and a speed slider
pub fn play_row(ui: &mut egui::Ui, running: &mut bool, speed: &mut f64, range: RangeInclusive<f64>, what: &str) {
    ui.horizontal(|ui| {
        if ui.button(if *running { "⏸" } else { "▶" }).on_hover_text("run / pause").clicked() {
            *running = !*running;
        }
        ui.add(egui::Slider::new(speed, range).logarithmic(true).text(what).custom_formatter(|v, _| fmt(v)));
    });
}

/// How long the histograms remember. Returns true when "clear" was pressed.
pub fn avg_row(ui: &mut egui::Ui, avg: &mut Avg) -> bool {
    let mut clear = false;
    ui.horizontal(|ui| {
        ui.label("histograms:").on_hover_text("how long the histograms remember: only now, about 5 s, or about a minute");
        for a in Avg::ALL {
            ui.selectable_value(avg, a, a.label());
        }
        clear = ui.small_button("clear").clicked();
    });
    clear
}

pub fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(6.0);
    ui.label(RichText::new(title).small().strong().color(Color32::from_gray(150)));
}

/// A table of numbers: what, the formula's value (white) and the simulation's (blue).
pub fn numbers(ui: &mut egui::Ui, id: &str, rows: &[(&str, Option<f64>, Option<f64>, &str)]) {
    egui::Grid::new(id).num_columns(3).spacing([14.0, 3.0]).striped(true).show(ui, |ui| {
        ui.weak("");
        ui.label(RichText::new("formula").small().color(plot::THEORY));
        ui.label(RichText::new("simulation").small().color(plot::SIM));
        ui.end_row();
        for &(name, theory, sim, hover) in rows {
            let r = ui.label(name);
            if !hover.is_empty() {
                r.on_hover_text(hover);
            }
            ui.label(RichText::new(theory.map_or("".into(), fmt)).color(plot::THEORY));
            ui.label(RichText::new(sim.map_or("".into(), fmt)).color(plot::SIM));
            ui.end_row();
        }
    });
}

pub fn notes(ui: &mut egui::Ui, text: &str) {
    section(ui, "NOTES");
    ui.label(RichText::new(text).size(13.5));
    ui.add_space(8.0);
}

/// the time step of this frame in seconds (bounded, so that a stalled tab does not jump)
pub fn frame_dt(ui: &egui::Ui) -> f64 {
    ui.input(|i| i.stable_dt).clamp(0.0, 0.1) as f64
}
