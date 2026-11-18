//! The x / p / both switch that every 1D scene has, and the layout helper that
//! stacks the position plot above the momentum plot.

use eframe::egui;
use serde::{Deserialize, Serialize};

use super::Domain;
use crate::Look;

/// Which representation(s) to show.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReprView {
    /// position space only
    #[default]
    X,
    /// momentum space only
    P,
    /// x above p
    Both,
}

impl ReprView {
    pub const ALL: [ReprView; 3] = [ReprView::X, ReprView::P, ReprView::Both];

    pub fn label(self) -> &'static str {
        match self {
            ReprView::X => "x",
            ReprView::P => "p",
            ReprView::Both => "x + p",
        }
    }

    /// Parses "x", "p", "both" (as used by `QB_ACTION=view_both`).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "x" => Some(ReprView::X),
            "p" => Some(ReprView::P),
            "both" | "xp" | "x+p" => Some(ReprView::Both),
            _ => None,
        }
    }

    pub fn shows_x(self) -> bool {
        self != ReprView::P
    }

    pub fn shows_p(self) -> bool {
        self != ReprView::X
    }

    /// Selectable labels "show  x | p | x + p". Returns true if the view changed.
    pub fn selector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("show");
            changed = self.segmented(ui);
        });
        changed
    }

    /// Just the segmented switch "x | p | x + p".
    pub fn segmented(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        super::wave_plot::segmented(ui, |ui| {
            for v in ReprView::ALL {
                let hover = match v {
                    ReprView::X => "position space ψ(x)",
                    ReprView::P => "momentum space Φ(p)",
                    ReprView::Both => "both, x above p",
                };
                if ui.selectable_label(*self == v, v.label()).on_hover_text(hover).clicked() {
                    changed |= *self != v;
                    *self = v;
                }
            }
        });
        changed
    }

    /// Lays out the plots in `height` points: calls `plot(ui, domain, h)` for x
    /// and/or p, x above p (55 % / 45 % of the height when both are shown).
    /// Use the same style and widths for both so their axes line up.
    pub fn show(self, ui: &mut egui::Ui, height: f32, mut plot: impl FnMut(&mut egui::Ui, Domain, f32)) {
        let gap = Look::of(ui).px(6.0);
        match self {
            ReprView::X => plot(ui, Domain::Position, height),
            ReprView::P => plot(ui, Domain::Momentum, height),
            ReprView::Both => {
                let hx = ((height - gap) * 0.55).max(60.0);
                let hp = (height - gap - hx).max(60.0);
                plot(ui, Domain::Position, hx);
                ui.add_space(gap);
                plot(ui, Domain::Momentum, hp);
            }
        }
    }
}
