//! The look of Quantum Bench: light/dark colours and presentation-mode scaling.
//!
//! The shell stores the current [`Look`] in the egui context every frame, so any
//! widget gets it with `Look::of(ui)` without it being passed around:
//!
//! ```ignore
//! let look = Look::of(ui);
//! painter.line_segment([a, b], look.stroke(1.5, look.accent()));
//! painter.text(p, Align2::LEFT_TOP, "⟨x⟩", look.font(13.0), look.text());
//! ```
//!
//! Always size text with `look.font(..)` and lines with `look.stroke(..)` /
//! `look.px(..)`: in presentation mode (key P) fonts grow by
//! [`Look::font_scale`] and lines by [`Look::stroke_scale`].

use eframe::egui::{self, Color32, FontId, Stroke};

/// Current visual settings (copied, cheap).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Look {
    pub dark: bool,
    /// presentation (projector) mode: larger fonts, thicker lines
    pub present: bool,
    /// screenshot mode: animations are frozen at a representative moment
    #[serde(skip)]
    pub shot: bool,
}

impl Default for Look {
    fn default() -> Self {
        Look {
            dark: true,
            present: false,
            shot: false,
        }
    }
}

const LOOK_ID: &str = "qb_core::look";

const fn gray(v: u8) -> Color32 {
    Color32::from_rgb(v, v, v)
}

impl Look {
    /// The look the shell installed for this frame.
    pub fn of(ui: &egui::Ui) -> Look {
        Self::from_ctx(ui.ctx())
    }

    pub fn from_ctx(ctx: &egui::Context) -> Look {
        ctx.data(|d| d.get_temp(egui::Id::new(LOOK_ID))).unwrap_or_default()
    }

    /// Stores the look in the context (done by the shell every frame).
    pub fn install(self, ctx: &egui::Context) {
        ctx.data_mut(|d| d.insert_temp(egui::Id::new(LOOK_ID), self));
    }

    /// Text size factor: 1, or 1.4 in presentation mode.
    pub fn font_scale(&self) -> f32 {
        if self.present { 1.4 } else { 1.0 }
    }

    /// Line width factor: 1, or 1.8 in presentation mode.
    pub fn stroke_scale(&self) -> f32 {
        if self.present { 1.8 } else { 1.0 }
    }

    /// Proportional font of nominal size `size` (scaled).
    pub fn font(&self, size: f32) -> FontId {
        FontId::proportional(size * self.font_scale())
    }

    /// Monospace font of nominal size `size` (scaled).
    pub fn mono(&self, size: f32) -> FontId {
        FontId::monospace(size * self.font_scale())
    }

    /// Stroke of nominal width `width` (scaled).
    pub fn stroke(&self, width: f32, color: Color32) -> Stroke {
        Stroke::new(width * self.stroke_scale(), color)
    }

    /// A length for markers, tick sizes, paddings (scaled like text).
    pub fn px(&self, size: f32) -> f32 {
        size * self.font_scale()
    }

    // ---------------------------------------------------------------- colours

    /// Picks the dark or the light variant.
    pub fn pick(&self, dark: Color32, light: Color32) -> Color32 {
        if self.dark { dark } else { light }
    }

    pub fn text(&self) -> Color32 {
        self.pick(gray(225), gray(25))
    }

    pub fn weak_text(&self) -> Color32 {
        self.pick(gray(150), gray(95))
    }

    /// background of plot areas
    pub fn plot_bg(&self) -> Color32 {
        self.pick(Color32::from_rgb(18, 20, 24), Color32::from_rgb(252, 252, 250))
    }

    pub fn plot_frame(&self) -> Color32 {
        self.pick(gray(70), gray(180))
    }

    pub fn grid(&self) -> Color32 {
        self.pick(gray(42), gray(228))
    }

    /// the zero line / axes
    pub fn axis(&self) -> Color32 {
        self.pick(gray(95), gray(150))
    }

    /// the potential V(x): line
    pub fn potential(&self) -> Color32 {
        self.pick(gray(200), gray(70))
    }

    /// the potential V(x): fill below the line
    pub fn potential_fill(&self) -> Color32 {
        self.pick(
            Color32::from_rgba_unmultiplied(200, 200, 210, 34),
            Color32::from_rgba_unmultiplied(60, 60, 80, 30),
        )
    }

    /// Re ψ
    pub fn re(&self) -> Color32 {
        self.pick(Color32::from_rgb(95, 175, 255), Color32::from_rgb(20, 95, 200))
    }

    /// Im ψ
    pub fn im(&self) -> Color32 {
        self.pick(Color32::from_rgb(255, 145, 75), Color32::from_rgb(210, 90, 10))
    }

    /// |ψ| and other neutral curves
    pub fn abs(&self) -> Color32 {
        self.pick(gray(235), gray(20))
    }

    /// markers such as ⟨x⟩, the selected item, draggable handles
    pub fn accent(&self) -> Color32 {
        self.pick(Color32::from_rgb(255, 214, 90), Color32::from_rgb(190, 120, 0))
    }

    /// detectors and collapse events
    pub fn measure(&self) -> Color32 {
        self.pick(Color32::from_rgb(255, 95, 150), Color32::from_rgb(205, 30, 100))
    }

    /// measured counts (histogram bars)
    pub fn counts(&self) -> Color32 {
        self.pick(Color32::from_rgb(110, 165, 235), Color32::from_rgb(70, 125, 205))
    }

    /// predictions (Born rule, analytic curves)
    pub fn prediction(&self) -> Color32 {
        self.pick(Color32::from_rgb(255, 214, 90), Color32::from_rgb(200, 90, 0))
    }

    /// energy levels in ladders and plots
    pub fn level(&self) -> Color32 {
        self.pick(gray(120), gray(150))
    }

    /// Colour `i` of a short qualitative palette for several series.
    pub fn series(&self, i: usize) -> Color32 {
        const DARK: [Color32; 6] = [
            Color32::from_rgb(110, 190, 255),
            Color32::from_rgb(255, 160, 80),
            Color32::from_rgb(120, 220, 140),
            Color32::from_rgb(230, 125, 255),
            Color32::from_rgb(250, 210, 90),
            Color32::from_rgb(255, 120, 140),
        ];
        const LIGHT: [Color32; 6] = [
            Color32::from_rgb(25, 105, 200),
            Color32::from_rgb(215, 100, 10),
            Color32::from_rgb(30, 145, 60),
            Color32::from_rgb(150, 50, 190),
            Color32::from_rgb(170, 130, 0),
            Color32::from_rgb(205, 40, 70),
        ];
        if self.dark { DARK[i % 6] } else { LIGHT[i % 6] }
    }

    /// The egui style for this look (used by the shell).
    pub fn egui_style(&self) -> egui::Style {
        let mut style = egui::Style {
            visuals: if self.dark { egui::Visuals::dark() } else { egui::Visuals::light() },
            ..Default::default()
        };
        let f = self.font_scale();
        for font in style.text_styles.values_mut() {
            font.size *= f;
        }
        if self.present {
            style.spacing.interact_size.y *= 1.25;
            style.spacing.item_spacing *= 1.2;
            style.spacing.button_padding *= 1.3;
            style.spacing.slider_width *= 1.3;
            style.spacing.icon_width *= 1.3;
            style.spacing.icon_width_inner *= 1.3;
            style.visuals.widgets.noninteractive.fg_stroke.width *= 1.5;
        }
        style
    }
}
