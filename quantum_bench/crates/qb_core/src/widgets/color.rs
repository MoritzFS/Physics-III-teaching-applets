//! The phase colour used everywhere: hue = arg ψ, brightness = intensity.
//!
//! arg = 0 is red, π/3 yellow, 2π/3 green, π cyan, 4π/3 blue, 5π/3 magenta.
//! The CPU function [`phase_color`] and the WGSL function `qb_phase_color`
//! in [`PHASE_WGSL`] use the same formula, so GPU and CPU pictures match.

use std::f32::consts::TAU;

use eframe::egui::{self, Color32, Pos2, Shape, Stroke};

use crate::C64;
use crate::Look;

/// RGB in 0..1 of the fully bright phase colour.
#[inline]
pub fn phase_rgb(arg: f32) -> [f32; 3] {
    let h = (arg / TAU).rem_euclid(1.0) * 6.0;
    [
        ((h - 3.0).abs() - 1.0).clamp(0.0, 1.0),
        (2.0 - (h - 2.0).abs()).clamp(0.0, 1.0),
        (2.0 - (h - 4.0).abs()).clamp(0.0, 1.0),
    ]
}

/// Hue = `arg` (radians), brightness = `intensity` (0 = black … 1 = full colour).
#[inline]
pub fn phase_color(arg: f32, intensity: f32) -> Color32 {
    let [r, g, b] = phase_rgb(arg);
    let v = intensity.clamp(0.0, 1.0) * 255.0;
    Color32::from_rgb((r * v + 0.5) as u8, (g * v + 0.5) as u8, (b * v + 0.5) as u8)
}

/// Full-brightness phase colour with opacity `alpha` (0..1), for fills over a background.
#[inline]
pub fn phase_color_alpha(arg: f32, alpha: f32) -> Color32 {
    let [r, g, b] = phase_rgb(arg);
    Color32::from_rgba_unmultiplied(
        (r * 255.0 + 0.5) as u8,
        (g * 255.0 + 0.5) as u8,
        (b * 255.0 + 0.5) as u8,
        (alpha.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
    )
}

/// Colour of a complex value: hue = arg z, brightness = |z|/max_abs.
#[inline]
pub fn complex_color(z: C64, max_abs: f64) -> Color32 {
    let i = if max_abs > 0.0 { (z.norm() / max_abs) as f32 } else { 0.0 };
    phase_color(z.arg() as f32, i)
}

/// Sequential colour map for densities (black → purple → orange → pale yellow), t in 0..1.
pub fn density_color(t: f32) -> Color32 {
    let [r, g, b] = density_rgb(t);
    Color32::from_rgb((r * 255.0 + 0.5) as u8, (g * 255.0 + 0.5) as u8, (b * 255.0 + 0.5) as u8)
}

/// RGB in 0..1 of [`density_color`] (same formula as `qb_density_color` in WGSL).
pub fn density_rgb(t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    [
        (1.6 * t).min(1.0) * (0.15 + 0.85 * t.sqrt()),
        (1.9 * t - 0.75).clamp(0.0, 1.0) * 0.85 + 0.15 * t,
        (0.55 * (TAU * 0.6 * t).sin().max(0.0) + 0.9 * (t - 0.8).max(0.0) * 4.0).min(1.0),
    ]
}

/// WGSL with `qb_phase_color(arg, intensity) -> vec3<f32>` and
/// `qb_density_color(t) -> vec3<f32>` (sRGB-encoded 0..1, like `Color32`).
/// Paste it into your shader source: `format!("{PHASE_WGSL}\n{my_shader}")`.
pub const PHASE_WGSL: &str = r#"
// --- qb_core phase colours (keep in sync with widgets/color.rs) ---
fn qb_phase_rgb(arg: f32) -> vec3<f32> {
    let h = fract(arg / 6.2831853) * 6.0;
    return vec3<f32>(
        clamp(abs(h - 3.0) - 1.0, 0.0, 1.0),
        clamp(2.0 - abs(h - 2.0), 0.0, 1.0),
        clamp(2.0 - abs(h - 4.0), 0.0, 1.0),
    );
}
fn qb_phase_color(arg: f32, intensity: f32) -> vec3<f32> {
    return qb_phase_rgb(arg) * clamp(intensity, 0.0, 1.0);
}
fn qb_density_color(t_in: f32) -> vec3<f32> {
    let t = clamp(t_in, 0.0, 1.0);
    return vec3<f32>(
        min(1.6 * t, 1.0) * (0.15 + 0.85 * sqrt(t)),
        clamp(1.9 * t - 0.75, 0.0, 1.0) * 0.85 + 0.15 * t,
        min(0.55 * max(sin(6.2831853 * 0.6 * t), 0.0) + 0.9 * max(t - 0.8, 0.0) * 4.0, 1.0),
    );
}
"#;

/// The phase legend: a colour wheel with "0" on the right and "π" on the left,
/// and an optional caption (e.g. "arg ψ") underneath.
pub fn phase_wheel(painter: &egui::Painter, center: Pos2, radius: f32, caption: Option<&str>) {
    let look = Look::from_ctx(painter.ctx());
    let n = 48;
    let mut mesh = egui::Mesh::default();
    mesh.colored_vertex(center, Color32::from_gray(if look.dark { 25 } else { 245 }));
    for i in 0..=n {
        let a = i as f32 / n as f32 * TAU;
        // counter-clockwise on screen (y down ⇒ −sin)
        mesh.colored_vertex(center + egui::vec2(a.cos(), -a.sin()) * radius, phase_color(a, 1.0));
    }
    for i in 0..n as u32 {
        mesh.add_triangle(0, i + 1, i + 2);
    }
    painter.add(Shape::mesh(mesh));
    painter.circle_stroke(center, radius, Stroke::new(1.0, look.plot_frame()));
    let f = look.font(9.5);
    let c = look.weak_text();
    painter.text(center + egui::vec2(radius + 3.0, 0.0), egui::Align2::LEFT_CENTER, "0", f.clone(), c);
    painter.text(
        center - egui::vec2(radius + 3.0, 0.0),
        egui::Align2::RIGHT_CENTER,
        "π",
        f.clone(),
        c,
    );
    if let Some(cap) = caption {
        painter.text(center + egui::vec2(0.0, radius + 2.0), egui::Align2::CENTER_TOP, cap, f, c);
    }
}
