//! `Heatmap2D`: a complex 2D field as a texture (phase hue, brightness from
//! the density), with axes and the phase legend.

use eframe::egui::{self, Color32, ColorImage, Rect, TextureHandle, TextureOptions, pos2};

use super::color::{density_color, phase_color, phase_wheel};
use super::plot::{PanZoom, PlotFrame, PlotResponse};
use crate::{C64, Look};

/// How complex values become colours.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HeatmapMode {
    /// hue = arg ψ, brightness = |ψ|/max|ψ| (i.e. √ of the density)
    #[default]
    PhaseBrightness,
    /// hue = arg ψ, brightness = |ψ|²/max|ψ|²
    PhaseDensity,
    /// |ψ|²/max with the sequential map [`density_color`] (no phase)
    Density,
    /// hue = arg ψ at full brightness where |ψ| > 1 % of max (phase only)
    PhaseOnly,
}

/// Converts a row-major `nx × ny` field into an image. Data row `iy = 0` is
/// y_min and is put at the **bottom** of the image (y up). `max_abs` fixes the
/// brightness scale (None = the field's maximum, so brightness is relative).
pub fn field_image(nx: usize, ny: usize, field: &[C64], mode: HeatmapMode, max_abs: Option<f64>) -> ColorImage {
    assert_eq!(field.len(), nx * ny, "field_image: wrong size");
    let max = max_abs
        .unwrap_or_else(|| field.iter().map(|z| z.norm()).fold(0.0, f64::max))
        .max(1e-300);
    let mut pixels = vec![Color32::BLACK; nx * ny];
    crate::par::chunks_mut(&mut pixels, nx, |row, out| {
        let iy = ny - 1 - row;
        for ix in 0..nx {
            let z = field[iy * nx + ix];
            let a = (z.norm() / max) as f32;
            out[ix] = match mode {
                HeatmapMode::PhaseBrightness => phase_color(z.arg() as f32, a),
                HeatmapMode::PhaseDensity => phase_color(z.arg() as f32, a * a),
                HeatmapMode::Density => density_color(a * a),
                HeatmapMode::PhaseOnly => {
                    if a > 0.01 {
                        phase_color(z.arg() as f32, 1.0)
                    } else {
                        Color32::BLACK
                    }
                }
            };
        }
    });
    ColorImage::new([nx, ny], pixels)
}

/// A texture of a 2D field plus the plot that shows it. Keep one per view in
/// your tab state and call [`Heatmap2D::set_field`] when the field changes.
///
/// ```ignore
/// self.map.set_field(ui.ctx(), nx, ny, &psi, HeatmapMode::PhaseBrightness, None);
/// let resp = self.map.show(ui, "box_xy", [0.0, a, 0.0, b], ("x", "y"), 400.0);
/// // resp.transform maps (x, y) ↔ screen: overlay nodal lines, detectors, …
/// ```
#[derive(Default)]
pub struct Heatmap2D {
    texture: Option<TextureHandle>,
    size: [usize; 2],
    /// draw the phase wheel (true unless the mode is `Density`)
    pub legend: bool,
    mode: HeatmapMode,
}

impl Heatmap2D {
    pub fn new() -> Self {
        Heatmap2D {
            legend: true,
            ..Default::default()
        }
    }

    /// Updates the texture from a complex field (see [`field_image`]).
    pub fn set_field(&mut self, ctx: &egui::Context, nx: usize, ny: usize, field: &[C64], mode: HeatmapMode, max_abs: Option<f64>) {
        self.mode = mode;
        self.set_image(ctx, field_image(nx, ny, field, mode, max_abs));
    }

    /// Updates the texture from any image (e.g. one you coloured yourself).
    pub fn set_image(&mut self, ctx: &egui::Context, image: ColorImage) {
        self.size = image.size;
        match &mut self.texture {
            Some(t) => t.set(image, TextureOptions::LINEAR),
            None => self.texture = Some(ctx.load_texture("qb_heatmap", image, TextureOptions::LINEAR)),
        }
    }

    /// Image size in pixels.
    pub fn size(&self) -> [usize; 2] {
        self.size
    }

    /// Shows the field over `extent = [x0, x1, y0, y1]` with equal aspect ratio,
    /// at most `height` points tall (and as wide as available).
    pub fn show(
        &self,
        ui: &mut egui::Ui,
        id: impl std::hash::Hash + std::fmt::Debug,
        extent: [f64; 4],
        labels: (&str, &str),
        height: f32,
    ) -> PlotResponse {
        let look = Look::of(ui);
        let [x0, x1, y0, y1] = extent;
        let (ml, mr, mt, mb) = PlotFrame::margins(&look, false);
        let avail_w = ui.available_width();
        let aspect = ((x1 - x0) / (y1 - y0)).abs().max(1e-6) as f32;
        let plot_h = (height - mt - mb).max(20.0);
        let plot_w = (plot_h * aspect).min(avail_w - ml - mr).max(20.0);
        let plot_h = plot_w / aspect;
        let legend = self.legend && self.mode != HeatmapMode::Density;
        let frame = PlotFrame::new(id)
            .x_range(x0, x1)
            .y_range(y0, y1)
            .labels(labels.0, labels.1)
            .width(plot_w + ml + mr)
            .height(plot_h + mt + mb)
            .grid(false)
            .zero_line(false)
            .pan_zoom(PanZoom::XY);
        let tex = self.texture.as_ref().map(|t| t.id());
        frame.show(ui, |painter, tr| {
            if let Some(tex) = tex {
                let r = Rect::from_two_pos(tr.to_screen(x0, y0), tr.to_screen(x1, y1));
                painter.image(tex, r, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
            }
            if legend {
                let rad = look.px(13.0);
                let c = pos2(tr.rect.right() - rad - look.px(14.0), tr.rect.top() + rad + look.px(6.0));
                phase_wheel(painter, c, rad, Some("arg ψ"));
            }
        })
    }
}
