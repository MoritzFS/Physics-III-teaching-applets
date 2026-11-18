//! The 3D density view: a GPU ray marcher of |ψ|² (qb_core::gpu::ShaderRect,
//! analytic ψ in the shader, see orbital.wgsl) with an orbit camera, a cut
//! plane, axes, the B direction, Born-sampled points and measured positions
//! drawn on top with egui.

use qb_core::egui::{self, Align2, Color32, Pos2, Rect, Sense, Shape, Stroke, pos2, vec2};
use qb_core::gpu::ShaderRect;
use qb_core::glam::DVec3;
use qb_core::widgets::{MeasureFlash, label_box, phase_color, phase_color_alpha, phase_wheel};
use qb_core::{C64, Ctx, Look};

use crate::orbital::{Coeffs, active, polys};

/// WGSL of the ray marcher.
pub const ORBITAL_WGSL: &str = include_str!("orbital.wgsl");

/// Maximal number of terms the shader takes (uniform array of 4·32 vec4).
pub const MAX_TERMS: usize = 32;
const N_HEADER: usize = 8;

/// How the density is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode3D {
    /// glowing cloud: brightness/opacity from |ψ|², hue = phase
    Cloud,
    /// surface |ψ|² = level, lit, coloured by phase
    Iso,
    /// Born-sampled positions
    Points,
}

impl Mode3D {
    pub fn next(self) -> Self {
        match self {
            Mode3D::Cloud => Mode3D::Iso,
            Mode3D::Iso => Mode3D::Points,
            Mode3D::Points => Mode3D::Cloud,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Mode3D::Cloud => "cloud",
            Mode3D::Iso => "isosurface",
            Mode3D::Points => "point cloud",
        }
    }
}

/// The plane of the 2D slice and of the 3D cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plane {
    /// the x–z plane at y = d
    XZ,
    /// the x–y plane at z = d
    XY,
}

impl Plane {
    pub fn normal(self) -> DVec3 {
        match self {
            Plane::XZ => DVec3::Y,
            Plane::XY => DVec3::Z,
        }
    }
    /// (horizontal axis, vertical axis, normal axis) names
    pub fn axes(self) -> (&'static str, &'static str, &'static str) {
        match self {
            Plane::XZ => ("x", "z", "y"),
            Plane::XY => ("x", "y", "z"),
        }
    }
    /// The 3D point of plane coordinates (a, b) at offset d.
    pub fn point(self, a: f64, b: f64, d: f64) -> [f64; 3] {
        match self {
            Plane::XZ => [a, d, b],
            Plane::XY => [a, b, d],
        }
    }
}

/// Orbit camera around the origin (z up).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    /// azimuth of the eye (radians, from +x towards +y)
    pub yaw: f64,
    /// elevation of the eye above the x–y plane (radians)
    pub pitch: f64,
    /// distance factor (1 = the bounding sphere fills the view)
    pub zoom: f64,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            yaw: (-58f64).to_radians(),
            pitch: 22f64.to_radians(),
            zoom: 1.0,
        }
    }
}

/// Camera frame for one picture.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub eye: DVec3,
    pub fwd: DVec3,
    pub right: DVec3,
    pub up: DVec3,
    pub tan_half: f64,
    pub aspect: f64,
    pub rect: Rect,
}

const FOV_DEG: f64 = 32.0;

impl Camera {
    pub fn frame(&self, radius: f64, rect: Rect) -> Frame {
        let tan_half = (0.5 * FOV_DEG).to_radians().tan();
        let aspect = (rect.width() / rect.height().max(1.0)) as f64;
        // distance so that the sphere fits the shorter side
        let fit = radius / (0.5 * FOV_DEG).to_radians().sin() / aspect.clamp(0.3, 1.0);
        let dist = fit * 1.07 * self.zoom;
        let dir = DVec3::new(self.pitch.cos() * self.yaw.cos(), self.pitch.cos() * self.yaw.sin(), self.pitch.sin());
        let eye = dir * dist;
        let fwd = -dir;
        let right = fwd.cross(DVec3::Z).normalize_or(DVec3::X);
        let up = right.cross(fwd);
        Frame {
            eye,
            fwd,
            right,
            up,
            tan_half,
            aspect,
            rect,
        }
    }
}

impl Frame {
    /// Screen position and depth of a 3D point (None behind the eye).
    pub fn project(&self, p: DVec3) -> Option<(Pos2, f64)> {
        let v = p - self.eye;
        let z = v.dot(self.fwd);
        if z <= 1e-9 {
            return None;
        }
        let sx = v.dot(self.right) / (z * self.tan_half * self.aspect);
        let sy = v.dot(self.up) / (z * self.tan_half);
        let r = self.rect;
        Some((
            pos2(
                r.center().x + sx as f32 * 0.5 * r.width(),
                r.center().y - sy as f32 * 0.5 * r.height(),
            ),
            z,
        ))
    }
}

/// What to draw this frame.
pub struct RenderInput<'a> {
    /// ψ = Σ c ψ_nlm at the current time (ignored if `packet` is set)
    pub coeffs: &'a Coeffs,
    /// a localized Gaussian packet (centre, σ) instead of the bound state
    pub packet: Option<([f64; 3], f64)>,
    /// radius of the view sphere (a₀)
    pub radius: f64,
    /// brightness scale ρ_ref (fixed while the state evolves)
    pub rho_max: f64,
    /// Born-sampled positions to draw as a point cloud (Points mode)
    pub points: &'a [[f64; 3]],
    /// colour points by the phase of ψ (else the counts colour)
    pub points_phase: bool,
    /// outcomes of single position measurements (dots); the last one is highlighted
    pub dots: &'a [[f64; 3]],
    /// the dipole ⟨r⟩ (drawn as an arrow, magnified by the given factor)
    pub dipole: Option<([f64; 3], f64)>,
    /// draw the B arrow along z
    pub b_arrow: bool,
    /// text in the top-left corner (e.g. the state)
    pub title: &'a str,
}

/// The 3D view and its settings.
pub struct View3D {
    pub cam: Camera,
    pub mode: Mode3D,
    /// opacity scale of the cloud
    pub exposure: f64,
    /// brightness = (|ψ|²/ρ_max)^γ
    pub gamma: f64,
    /// isosurface level as a fraction of ρ_max
    pub iso: f64,
    pub cut: bool,
    pub plane: Plane,
    /// offset of the cut / slice plane (a₀)
    pub offset: f64,
    /// show ψ on the cut face
    pub cap: bool,
    /// faint cloud behind the points
    pub faint_cloud: bool,
    pub axes: bool,
    /// ray-march steps across the sphere diameter
    pub steps: f32,
    shader: Option<ShaderRect>,
    gpu_failed: bool,
}

impl Default for View3D {
    fn default() -> Self {
        View3D {
            cam: Camera::default(),
            mode: Mode3D::Cloud,
            exposure: 1.2,
            gamma: 0.8,
            iso: 0.06,
            cut: false,
            plane: Plane::XZ,
            offset: 0.0,
            cap: true,
            faint_cloud: true,
            axes: true,
            steps: 200.0,
            shader: None,
            gpu_failed: false,
        }
    }
}

fn v4(v: DVec3, w: f64) -> [f32; 4] {
    [v.x as f32, v.y as f32, v.z as f32, w as f32]
}

fn srgb(c: Color32) -> [f32; 3] {
    [c.r() as f32 / 255.0, c.g() as f32 / 255.0, c.b() as f32 / 255.0]
}

impl View3D {
    /// The uniform block for the shader.
    pub fn uniforms(&self, input: &RenderInput, frame: &Frame, look: &Look) -> Vec<[f32; 4]> {
        let mut u = vec![[0f32; 4]; N_HEADER + 4 * MAX_TERMS];
        let mode = match self.mode {
            Mode3D::Cloud => 0.0,
            Mode3D::Iso => 1.0,
            Mode3D::Points => 0.0,
        };
        let exposure = match self.mode {
            Mode3D::Points if !self.faint_cloud => 0.0,
            Mode3D::Points => 0.18 * self.exposure,
            _ => self.exposure,
        };
        u[0] = v4(frame.eye, input.radius);
        u[1] = v4(frame.fwd, frame.tan_half);
        u[2] = v4(frame.right, self.steps as f64);
        u[3] = v4(frame.up, mode);
        u[4] = [(1.0 / input.rho_max.max(1e-300)) as f32, exposure as f32, self.gamma as f32, self.iso as f32];
        u[5] = v4(self.plane.normal(), self.offset);
        let bg = srgb(look.plot_bg());
        u[7] = [bg[0], bg[1], bg[2], 0.0];
        let mut nt = 0;
        if let Some((c, s)) = input.packet {
            u[N_HEADER] = [1.0, 0.0, 0.0, 1.0];
            u[N_HEADER + 1] = [c[0] as f32, c[1] as f32, c[2] as f32, s as f32];
            nt = 1;
        } else {
            let ps = polys();
            for k in active(input.coeffs).into_iter().take(MAX_TERMS) {
                let o = &ps[k];
                let c: C64 = input.coeffs[k] * (o.rnorm * o.anorm);
                let b = N_HEADER + 4 * nt;
                u[b] = [c.re as f32, c.im as f32, o.inv_n as f32, 0.0];
                u[b + 1] = [o.q.l as f32, o.q.m as f32, 0.0, 0.0];
                u[b + 2] = o.rad.map(|v| v as f32);
                u[b + 3] = o.ang.map(|v| v as f32);
                nt += 1;
            }
        }
        // the painted cut face (not in the point-cloud mode: the points are the picture there)
        let cap = self.cap && self.mode != Mode3D::Points;
        u[6] = [if self.cut { 1.0 } else { 0.0 }, if cap { 1.0 } else { 0.0 }, nt as f32, 0.9];
        u
    }

    /// Is the point in the half-space removed by the cut (seen from the eye)?
    pub fn cut_away(&self, p: DVec3, eye: DVec3) -> bool {
        if !self.cut {
            return false;
        }
        let n = self.plane.normal();
        let side = eye.dot(n) - self.offset;
        let sgn = if side >= 0.0 { 1.0 } else { -1.0 };
        (p.dot(n) - self.offset) * sgn > 0.0
    }

    /// Draws the view into `rect` and handles orbit/zoom.
    pub fn show(&mut self, ui: &mut egui::Ui, ctx: &Ctx, rect: Rect, input: &RenderInput, flash: &MeasureFlash) -> egui::Response {
        let look = Look::of(ui);
        let response = ui.interact(rect, ui.id().with("hydrogen_3d"), Sense::click_and_drag());
        if !look.shot {
            if response.dragged_by(egui::PointerButton::Primary) || response.dragged_by(egui::PointerButton::Secondary) {
                let d = response.drag_delta();
                self.cam.yaw -= d.x as f64 * 0.008;
                self.cam.pitch = (self.cam.pitch + d.y as f64 * 0.008).clamp(-1.55, 1.55);
            }
            if response.hovered() {
                let (scroll, zoom) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
                let f = (-(scroll as f64) * 0.002).exp() / zoom as f64;
                if (f - 1.0).abs() > 1e-6 {
                    self.cam.zoom = (self.cam.zoom * f).clamp(0.25, 4.0);
                }
            }
            if response.double_clicked() {
                let keep = self.cam;
                self.cam = Camera::default();
                if keep == self.cam {
                    self.cam.zoom = 1.0;
                }
            }
        }
        let frame = self.cam.frame(input.radius, rect);
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 2.0, look.plot_bg());

        // the GPU volume
        let mut drawn = false;
        if let Some(rs) = ctx.render_state {
            if self.shader.is_none() && !self.gpu_failed {
                self.shader = Some(ShaderRect::new(
                    rs,
                    "hydrogen orbital",
                    ORBITAL_WGSL,
                    (N_HEADER + 4 * MAX_TERMS) * 16,
                    None,
                ));
            }
            if let Some(sh) = &self.shader {
                let u = self.uniforms(input, &frame, &look);
                sh.paint(&painter, rect, bytemuck::cast_slice(&u));
                drawn = true;
            }
        }
        if !drawn && self.mode != Mode3D::Points {
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                "no GPU: use the point cloud mode",
                look.font(14.0),
                look.weak_text(),
            );
        }

        self.overlays(&painter, &look, &frame, input);
        // frame
        painter.rect_stroke(rect, 2.0, Stroke::new(1.0, look.plot_frame()), egui::StrokeKind::Inside);
        flash.paint_banner(ui, rect);
        if response.hovered() && !look.shot {
            response.clone().on_hover_text("drag: rotate · scroll: zoom · double-click: default view");
        }
        response
    }

    fn overlays(&self, painter: &egui::Painter, look: &Look, f: &Frame, input: &RenderInput) {
        let l = input.radius;
        let rect = f.rect;
        let proj = |p: DVec3| f.project(p).map(|(s, _)| s);
        // cut-plane outline: the circle where the plane meets the sphere
        if self.cut && self.offset.abs() < l {
            let rc = (l * l - self.offset * self.offset).sqrt();
            let (a, b) = match self.plane {
                Plane::XZ => (DVec3::X, DVec3::Z),
                Plane::XY => (DVec3::X, DVec3::Y),
            };
            let c0 = self.plane.normal() * self.offset;
            let pts: Vec<Pos2> = (0..=96)
                .filter_map(|k| {
                    let t = k as f64 / 96.0 * std::f64::consts::TAU;
                    proj(c0 + a * (rc * t.cos()) + b * (rc * t.sin()))
                })
                .collect();
            painter.add(Shape::dashed_line(&pts, look.stroke(1.0, look.weak_text().gamma_multiply(0.7)), 5.0, 4.0));
        }
        // axes
        if self.axes {
            let axis_col = look.axis();
            for (dir, name) in [(DVec3::X, "x"), (DVec3::Y, "y"), (DVec3::Z, "z")] {
                let toward = dir.dot(f.eye) > 0.0;
                let (Some(o), Some(p), Some(n)) = (proj(DVec3::ZERO), proj(dir * l * 0.98), proj(-dir * l * 0.98)) else {
                    continue;
                };
                // the half pointing away from the eye lies (mostly) behind the cloud: fainter
                let (front, back) = if toward { (p, n) } else { (n, p) };
                painter.line_segment([o, back], Stroke::new(look.stroke_scale(), axis_col.gamma_multiply(0.45)));
                painter.line_segment([o, front], Stroke::new(1.2 * look.stroke_scale(), axis_col.gamma_multiply(0.9)));
                arrow_head(painter, n, p, look.px(7.0), axis_col);
                let lab = p + (p - o).normalized() * look.px(10.0);
                painter.text(lab, Align2::CENTER_CENTER, name, look.font(13.0), look.text());
            }
        }
        if input.b_arrow
            && let (Some(o), Some(zt)) = (proj(DVec3::ZERO), proj(DVec3::Z * l))
        {
            // B along z, drawn beside the view (screen direction of the z axis)
            let d = (zt - o).normalized();
            let col = look.series(0);
            let a = rect.left_top() + vec2(look.px(34.0), look.px(120.0));
            let b = a + d * look.px(60.0);
            painter.line_segment([a, b], look.stroke(3.0, col));
            arrow_head(painter, a, b, look.px(11.0), col);
            painter.text(b + vec2(look.px(8.0), look.px(6.0)), Align2::LEFT_TOP, "B ∥ z", look.font(14.0), col);
        }
        // Born-sampled points
        if self.mode == Mode3D::Points && !input.points.is_empty() {
            let act = active(input.coeffs);
            let ps = polys();
            let r_px = look.px(1.5).max(1.0);
            let dmid = f.eye.length();
            for p in input.points {
                let v = DVec3::from_array(*p);
                if self.cut_away(v, f.eye) {
                    continue;
                }
                let Some((s, depth)) = f.project(v) else { continue };
                if !rect.contains(s) {
                    continue;
                }
                let col = if input.points_phase && input.packet.is_none() {
                    let g = crate::orbital::Geo::new(*p);
                    let psi: C64 = act.iter().map(|&k| input.coeffs[k] * ps[k].eval_geo(&g)).sum();
                    let a = (0.95 - 0.35 * ((depth - dmid) / l).clamp(-1.0, 1.0)) as f32;
                    phase_color_alpha(psi.arg() as f32, a.clamp(0.3, 1.0))
                } else {
                    look.counts()
                };
                painter.circle_filled(s, r_px, col);
            }
        }
        // the dipole ⟨r⟩
        if let Some((d, scale)) = input.dipole {
            let v = DVec3::from_array(d) * scale;
            if let (Some(o), Some(p)) = (proj(DVec3::ZERO), proj(v))
                && (p - o).length() > 2.0
            {
                let col = look.accent();
                painter.line_segment([o, p], look.stroke(2.5, col));
                arrow_head(painter, o, p, look.px(9.0), col);
                painter.text(
                    p + vec2(look.px(6.0), -look.px(6.0)),
                    Align2::LEFT_BOTTOM,
                    format!("⟨r⟩ ×{scale:.0}"),
                    look.font(12.0),
                    col,
                );
            }
        }
        // single-shot position outcomes
        let n = input.dots.len();
        for (i, p) in input.dots.iter().enumerate() {
            let Some(s) = proj(DVec3::from_array(*p)) else { continue };
            let last = i + 1 == n;
            let r = look.px(if last { 4.5 } else { 2.6 });
            painter.circle_filled(s, r, look.measure());
            if last {
                painter.circle_stroke(s, r + look.px(2.0), Stroke::new(look.stroke_scale(), Color32::WHITE.gamma_multiply(0.8)));
            }
        }
        // legend, title, scale
        let rad = look.px(13.0);
        let c = pos2(rect.right() - rad - look.px(14.0), rect.top() + rad + look.px(8.0));
        phase_wheel(painter, c, rad, Some("arg ψ"));
        if !input.title.is_empty() {
            label_box(
                painter,
                rect.left_top() + vec2(look.px(6.0), look.px(6.0)),
                Align2::LEFT_TOP,
                input.title,
                look.text(),
            );
        }
        let info = match self.mode {
            Mode3D::Cloud => format!("cloud: opacity ∝ (|ψ|²/ρ_ref)^{:.2}", self.gamma),
            Mode3D::Iso => format!("isosurface |ψ|² = {:.3} ρ_ref", self.iso),
            Mode3D::Points => format!("{} positions drawn from |ψ|²", input.points.len()),
        };
        painter.text(
            rect.left_bottom() + vec2(look.px(6.0), -look.px(5.0)),
            Align2::LEFT_BOTTOM,
            format!("{info}   ·   sphere radius {:.0} a₀", input.radius),
            look.font(11.0),
            look.weak_text(),
        );
        let _ = phase_color;
    }
}

fn arrow_head(painter: &egui::Painter, from: Pos2, to: Pos2, size: f32, col: Color32) {
    let d = to - from;
    if d.length() < 1e-3 {
        return;
    }
    let d = d.normalized();
    let n = vec2(-d.y, d.x);
    painter.add(Shape::convex_polygon(
        vec![to, to - d * size + n * size * 0.45, to - d * size - n * size * 0.45],
        col,
        Stroke::NONE,
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(src: &str) {
        let module = match naga::front::wgsl::parse_str(src) {
            Ok(m) => m,
            Err(e) => panic!("{}", e.emit_to_string(src)),
        };
        let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::default());
        if let Err(e) = v.validate(&module) {
            panic!("{}", e.emit_to_string(src));
        }
    }

    #[test]
    fn orbital_shader_validates() {
        for srgb in [false, true] {
            validate(&qb_core::gpu::shader_source(ORBITAL_WGSL, srgb));
        }
    }

    #[test]
    fn uniform_block_size_matches_the_wgsl_struct() {
        // 8 header vec4 + 128 term vec4
        assert_eq!((N_HEADER + 4 * MAX_TERMS) * 16, 2176);
        assert!(ORBITAL_WGSL.contains("array<vec4<f32>, 128>"));
    }

    #[test]
    fn projection_matches_the_shader_ray() {
        let cam = Camera::default();
        let rect = Rect::from_min_size(pos2(10.0, 20.0), vec2(400.0, 300.0));
        let f = cam.frame(10.0, rect);
        // a point on the ray through uv = (0.3, 0.6) must project back onto that pixel
        let (u, v) = (0.3f64, 0.6f64);
        let sx = (2.0 * u - 1.0) * f.aspect * f.tan_half;
        let sy = (1.0 - 2.0 * v) * f.tan_half;
        let dir = (f.fwd + f.right * sx + f.up * sy).normalize();
        let p = f.eye + dir * 17.0;
        let (s, _) = f.project(p).unwrap();
        assert!((s.x - (rect.left() + 0.3 * rect.width())).abs() < 1e-3);
        assert!((s.y - (rect.top() + 0.6 * rect.height())).abs() < 1e-3);
        // z is up on screen, right-handed: x to the right when looking along +y
        let side = Camera {
            yaw: -std::f64::consts::FRAC_PI_2,
            pitch: 0.0,
            zoom: 1.0,
        }
        .frame(10.0, rect);
        assert!((side.right - DVec3::X).length() < 1e-12 && (side.up - DVec3::Z).length() < 1e-12);
    }
}
