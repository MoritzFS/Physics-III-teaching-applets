//! Drawing helpers of the 2D box: colour maps, the vertical marginal plot,
//! node lines (marching squares), phase vortices, current arrows and the
//! height-surface (pseudo-3D) view.

use qb_core::egui::{self, Align2, Color32, ColorImage, Pos2, Rect, Shape, Stroke, StrokeKind, pos2, vec2};
use qb_core::widgets::{PlotTransform, density_color, nice_ticks, phase_rgb, tick_label};
use qb_core::{C64, Look};

/// How ψ is coloured on the map (the 2D counterpart of the 1D display modes; key V cycles).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Display {
    /// hue = arg ψ, brightness from |ψ|² (or |ψ|)
    #[default]
    Phase,
    /// Re ψ with a diverging map (blue < 0 < red)
    Real,
    /// |ψ|² with a sequential map, no phase
    Plain,
}

impl Display {
    pub const ALL: [Display; 3] = [Display::Phase, Display::Real, Display::Plain];

    pub fn label(self) -> &'static str {
        match self {
            Display::Phase => "|ψ|² + phase",
            Display::Real => "Re ψ",
            Display::Plain => "|ψ|²",
        }
    }

    pub fn next(self) -> Self {
        let i = Self::ALL.iter().position(|d| *d == self).unwrap_or(0);
        Self::ALL[(i + 1) % Self::ALL.len()]
    }

    /// "density"/"phase", "re", "plain" (the 1D action names).
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "density" | "phase" => Some(Display::Phase),
            "re" | "real" | "reim" => Some(Display::Real),
            "plain" | "abs" => Some(Display::Plain),
            _ => None,
        }
    }
}

/// Diverging colour for t ∈ [−1, 1]: blue (negative) … background (0) … red (positive).
pub fn diverging(t: f32, look: &Look) -> Color32 {
    let bg = look.plot_bg();
    let end = if t >= 0.0 {
        look.pick(Color32::from_rgb(255, 105, 70), Color32::from_rgb(200, 40, 20))
    } else {
        look.pick(Color32::from_rgb(80, 160, 255), Color32::from_rgb(20, 80, 205))
    };
    let a = t.abs().clamp(0.0, 1.0);
    let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * a + 0.5) as u8;
    Color32::from_rgb(mix(bg.r(), end.r()), mix(bg.g(), end.g()), mix(bg.b(), end.b()))
}

/// Re ψ as an image (row 0 of the data = bottom of the image).
pub fn re_image(nx: usize, ny: usize, field: &[C64], max: f64, look: &Look) -> ColorImage {
    let mut pixels = vec![Color32::BLACK; nx * ny];
    let inv = 1.0 / max.max(1e-300);
    for (row, out) in pixels.chunks_mut(nx).enumerate() {
        let iv = ny - 1 - row;
        for (iu, o) in out.iter_mut().enumerate() {
            *o = diverging((field[iv * nx + iu].re * inv) as f32, look);
        }
    }
    ColorImage::new([nx, ny], pixels)
}

/// A small colour bar in the top-right corner of a map: Re ψ (diverging) or |ψ|² (sequential).
pub fn colour_bar(painter: &egui::Painter, tr: &PlotTransform, display: Display, brightness_amp: bool, look: &Look) {
    let w = look.px(90.0);
    let h = look.px(9.0);
    let r = tr.rect;
    let bar = Rect::from_min_size(pos2(r.right() - w - look.px(12.0), r.top() + look.px(10.0)), vec2(w, h));
    let bg = Rect::from_min_max(
        bar.min - vec2(look.px(6.0), look.px(4.0)),
        bar.max + vec2(look.px(6.0), look.px(18.0)),
    );
    painter.rect_filled(bg, 3.0, look.plot_bg().gamma_multiply(0.85));
    let n = 32;
    let mut mesh = egui::Mesh::default();
    for k in 0..=n {
        let s = k as f32 / n as f32;
        let col = match display {
            Display::Real => diverging(2.0 * s - 1.0, look),
            Display::Plain => density_color(s),
            Display::Phase => {
                let [r, g, b] = phase_rgb(0.0);
                let v = if brightness_amp { s.sqrt() } else { s };
                Color32::from_rgb((r * v * 255.0) as u8, (g * v * 255.0) as u8, (b * v * 255.0) as u8)
            }
        };
        let x = bar.left() + s * w;
        mesh.colored_vertex(pos2(x, bar.top()), col);
        mesh.colored_vertex(pos2(x, bar.bottom()), col);
        if k > 0 {
            let i = 2 * k as u32;
            mesh.add_triangle(i - 2, i - 1, i);
            mesh.add_triangle(i - 1, i, i + 1);
        }
    }
    painter.add(Shape::mesh(mesh));
    painter.rect_stroke(bar, 0.0, Stroke::new(1.0, look.plot_frame()), StrokeKind::Outside);
    let f = look.font(10.0);
    let c = look.weak_text();
    let y = bar.bottom() + 2.0;
    let (l, m, rr) = match display {
        Display::Real => ("−", "Re ψ", "+"),
        _ => ("0", "|ψ|²", "max"),
    };
    painter.text(pos2(bar.left(), y), Align2::LEFT_TOP, l, f.clone(), c);
    painter.text(pos2(bar.center().x, y), Align2::CENTER_TOP, m, f.clone(), c);
    painter.text(pos2(bar.right(), y), Align2::RIGHT_TOP, rr, f, c);
}

/// Fills between the line u = 0 (or v = 0) and a curve, for the marginal plots.
fn fill(painter: &egui::Painter, pts: &[(Pos2, Pos2)], color: Color32) {
    let mut mesh = egui::Mesh::default();
    for (k, (a, b)) in pts.iter().enumerate() {
        mesh.colored_vertex(*a, color);
        mesh.colored_vertex(*b, color);
        if k > 0 {
            let i = 2 * k as u32;
            mesh.add_triangle(i - 2, i - 1, i);
            mesh.add_triangle(i - 1, i, i + 1);
        }
    }
    painter.add(Shape::mesh(mesh));
}

/// The marginal P(v) to the right of the map, sharing its vertical axis
/// (`map` is the map's transform; the plot uses its rows). `analytic` is drawn dashed.
#[allow(clippy::too_many_arguments)]
pub fn marginal_v(
    ui: &mut egui::Ui,
    width: f32,
    map: &PlotTransform,
    vs: &[f64],
    pv: &[f64],
    top: f64,
    analytic: Option<&[f64]>,
    label: &str,
) -> PlotTransform {
    let look = Look::of(ui);
    let h = map.rect.bottom() - ui.cursor().top() + look.px(22.0);
    let (outer, _) = ui.allocate_exact_size(vec2(width, h.max(map.rect.height())), egui::Sense::hover());
    let rect = Rect::from_min_max(
        pos2(outer.left() + look.px(4.0), map.rect.top()),
        pos2(outer.right() - look.px(8.0), map.rect.bottom()),
    );
    let tr = PlotTransform {
        rect,
        x: [0.0, top],
        y: map.y,
    };
    let painter = ui.painter_at(outer);
    painter.rect_filled(rect, 2.0, look.plot_bg());
    let (ticks, step) = nice_ticks(0.0, top, 2);
    for t in ticks {
        let sx = tr.x_to_screen(t);
        painter.line_segment([pos2(sx, rect.top()), pos2(sx, rect.bottom())], Stroke::new(1.0, look.grid()));
        painter.text(
            pos2(sx, rect.bottom() + 2.0),
            Align2::CENTER_TOP,
            tick_label(t, step),
            look.font(11.0),
            look.weak_text(),
        );
    }
    let clip = ui.painter_at(rect);
    let pts: Vec<(Pos2, Pos2)> = vs
        .iter()
        .zip(pv)
        .map(|(&v, &p)| (tr.to_screen(0.0, v), tr.to_screen(p.min(top * 1.5), v)))
        .collect();
    fill(&clip, &pts, look.re().gamma_multiply(0.35));
    clip.add(Shape::line(pts.iter().map(|p| p.1).collect(), look.stroke(1.6, look.re())));
    if let Some(a) = analytic {
        let line: Vec<Pos2> = vs.iter().zip(a).map(|(&v, &p)| tr.to_screen(p, v)).collect();
        clip.extend(Shape::dashed_line(&line, look.stroke(1.3, look.prediction()), 5.0, 4.0));
    }
    painter.rect_stroke(rect, 2.0, Stroke::new(1.0, look.plot_frame()), StrokeKind::Inside);
    let g = painter.layout_no_wrap(label.to_string(), look.font(12.0), look.text());
    let p = rect.left_top() + vec2(4.0, 3.0);
    painter.rect_filled(
        Rect::from_min_size(p - vec2(2.0, 1.0), g.size() + vec2(4.0, 2.0)),
        2.0,
        look.plot_bg().gamma_multiply(0.85),
    );
    painter.galley(p, g, look.text());
    tr
}

/// Fills the area under a horizontal marginal curve (P(u) below the map).
pub fn fill_under(painter: &egui::Painter, tr: &PlotTransform, xs: &[f64], ys: &[f64], color: Color32) {
    let pts: Vec<(Pos2, Pos2)> = xs
        .iter()
        .zip(ys)
        .map(|(&x, &y)| (tr.to_screen(x, y), tr.to_screen(x, 0.0)))
        .collect();
    fill(painter, &pts, color);
}

/// Zero contour of a real field (marching squares): segments in data coordinates.
/// Cells where all |f| < `floor` are skipped (sign changes of numerical noise in the tails).
pub fn zero_contour(f: &[f64], xs: &[f64], ys: &[f64], floor: f64) -> Vec<[[f64; 2]; 2]> {
    let (nx, ny) = (xs.len(), ys.len());
    let mut out = vec![];
    for iv in 0..ny - 1 {
        for iu in 0..nx - 1 {
            // corners: 0 (u, v), 1 (u+1, v), 2 (u+1, v+1), 3 (u, v+1)
            let c = [
                f[iv * nx + iu],
                f[iv * nx + iu + 1],
                f[(iv + 1) * nx + iu + 1],
                f[(iv + 1) * nx + iu],
            ];
            if c.iter().all(|v| v.abs() < floor) {
                continue;
            }
            let p = [
                [xs[iu], ys[iv]],
                [xs[iu + 1], ys[iv]],
                [xs[iu + 1], ys[iv + 1]],
                [xs[iu], ys[iv + 1]],
            ];
            let mut cross: Vec<[f64; 2]> = vec![];
            for e in 0..4 {
                let (a, b) = (e, (e + 1) % 4);
                if (c[a] > 0.0) != (c[b] > 0.0) {
                    let t = c[a] / (c[a] - c[b]);
                    cross.push([p[a][0] + (p[b][0] - p[a][0]) * t, p[a][1] + (p[b][1] - p[a][1]) * t]);
                }
            }
            match cross.len() {
                2 => out.push([cross[0], cross[1]]),
                4 => {
                    // saddle: decide with the centre value
                    let centre = 0.25 * (c[0] + c[1] + c[2] + c[3]);
                    if (centre > 0.0) == (c[0] > 0.0) {
                        out.push([cross[0], cross[3]]);
                        out.push([cross[1], cross[2]]);
                    } else {
                        out.push([cross[0], cross[1]]);
                        out.push([cross[2], cross[3]]);
                    }
                }
                _ => {}
            }
        }
    }
    out
}

/// Phase vortices (points where ψ = 0 and arg ψ winds by ±2π): (u, v, winding).
pub fn vortices(field: &[C64], xs: &[f64], ys: &[f64], max_abs: f64) -> Vec<(f64, f64, i32)> {
    let (nx, ny) = (xs.len(), ys.len());
    let args: Vec<f64> = field.iter().map(|z| z.arg()).collect();
    let wrap = |d: f64| {
        let tau = std::f64::consts::TAU;
        (d + std::f64::consts::PI).rem_euclid(tau) - std::f64::consts::PI
    };
    let floor = 1e-3 * max_abs;
    let mut out = vec![];
    for iv in 0..ny - 1 {
        for iu in 0..nx - 1 {
            let k = [iv * nx + iu, iv * nx + iu + 1, (iv + 1) * nx + iu + 1, (iv + 1) * nx + iu];
            if k.iter().all(|&q| field[q].norm() < floor) {
                continue;
            }
            let mut w = 0.0;
            for e in 0..4 {
                w += wrap(args[k[(e + 1) % 4]] - args[k[e]]);
            }
            let n = (w / std::f64::consts::TAU).round() as i32;
            if n != 0 {
                out.push((0.5 * (xs[iu] + xs[iu + 1]), 0.5 * (ys[iv] + ys[iv + 1]), n));
            }
        }
    }
    out
}

/// An arrow from `a` to `b` (screen points).
pub fn arrow(painter: &egui::Painter, a: Pos2, b: Pos2, stroke: Stroke) {
    let d = b - a;
    let len = d.length();
    if len < 1.5 {
        return;
    }
    painter.line_segment([a, b], stroke);
    let dir = d / len;
    let n = vec2(-dir.y, dir.x);
    let head = (len * 0.35).min(stroke.width * 4.0 + 3.0);
    painter.add(Shape::convex_polygon(
        vec![b, b - dir * head + n * head * 0.5, b - dir * head - n * head * 0.5],
        stroke.color,
        Stroke::NONE,
    ));
}

/// Height surface of |ψ|² (pseudo-3D, painter's algorithm), coloured like the map.
/// `field` is a coarse nx × ny sampling over [0, lx] × [0, ly]; `top` is the |ψ|² of full height.
#[allow(clippy::too_many_arguments)]
pub fn surface(
    painter: &egui::Painter,
    rect: Rect,
    field: &[C64],
    nx: usize,
    ny: usize,
    lx: f64,
    ly: f64,
    top: f64,
    display: Display,
    azimuth: f32,
    labels: (&str, &str),
    look: &Look,
) {
    let elev: f32 = 0.55;
    let (ca, sa) = (azimuth.cos(), azimuth.sin());
    let (ce, se) = (elev.cos(), elev.sin());
    let span = lx.max(ly) as f32;
    // world: box centred, unit = span; height scaled to 0.55 span
    let hz = 0.55f32;
    let project = |u: f32, v: f32, h: f32| -> (f32, f32, f32) {
        let x = (u - 0.5 * lx as f32) / span;
        let y = (v - 0.5 * ly as f32) / span;
        let xr = x * ca - y * sa;
        let yr = x * sa + y * ca;
        // screen x, screen y (up), depth (larger = farther)
        (xr, h * hz * ce + yr * se, yr * ce - h * hz * se)
    };
    // fit the box + height into the rect
    let corners = [
        project(0.0, 0.0, 0.0),
        project(lx as f32, 0.0, 0.0),
        project(0.0, ly as f32, 0.0),
        project(lx as f32, ly as f32, 0.0),
        project(0.0, 0.0, 1.0),
        project(lx as f32, ly as f32, 1.0),
        project(lx as f32, 0.0, 1.0),
        project(0.0, ly as f32, 1.0),
    ];
    let (mut x0, mut x1, mut y0, mut y1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for c in corners {
        x0 = x0.min(c.0);
        x1 = x1.max(c.0);
        y0 = y0.min(c.1);
        y1 = y1.max(c.1);
    }
    let scale = (rect.width() / (x1 - x0)).min(rect.height() / (y1 - y0)) * 0.92;
    let cx = rect.center().x - 0.5 * (x0 + x1) * scale;
    let cy = rect.center().y + 0.5 * (y0 + y1) * scale;
    let to_screen = |p: (f32, f32, f32)| pos2(cx + p.0 * scale, cy - p.1 * scale);

    painter.rect_filled(rect, 2.0, look.plot_bg());
    // floor outline
    let floor = [(0.0, 0.0), (lx as f32, 0.0), (lx as f32, ly as f32), (0.0, ly as f32), (0.0, 0.0)];
    let pts: Vec<Pos2> = floor.iter().map(|&(u, v)| to_screen(project(u, v, 0.0))).collect();
    painter.add(Shape::line(pts, look.stroke(1.2, look.axis())));
    let inv_top = 1.0 / top.max(1e-300);
    let max_abs = top.sqrt();
    let h_at = |iu: usize, iv: usize| (field[iv * nx + iu].norm_sqr() * inv_top).min(1.6) as f32;
    let uv = |iu: usize, iv: usize| ((iu as f32 + 0.5) / nx as f32 * lx as f32, (iv as f32 + 0.5) / ny as f32 * ly as f32);
    // quads sorted back to front
    let mut quads: Vec<(f32, usize, usize)> = Vec::with_capacity((nx - 1) * (ny - 1));
    for iv in 0..ny - 1 {
        for iu in 0..nx - 1 {
            let (u, v) = uv(iu, iv);
            let d = project(u, v, 0.0).2;
            quads.push((d, iu, iv));
        }
    }
    quads.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut mesh = egui::Mesh::default();
    let light = glam_light();
    for (_, iu, iv) in quads {
        let corners = [(iu, iv), (iu + 1, iv), (iu + 1, iv + 1), (iu, iv + 1)];
        let p: Vec<(f32, f32, f32)> = corners
            .iter()
            .map(|&(a, b)| {
                let (u, v) = uv(a, b);
                project(u, v, h_at(a, b))
            })
            .collect();
        // shading from the surface normal in world coordinates
        let (u0, v0) = uv(iu, iv);
        let (u1, v1) = uv(iu + 1, iv + 1);
        let dzdu = 0.5 * (h_at(iu + 1, iv) - h_at(iu, iv) + h_at(iu + 1, iv + 1) - h_at(iu, iv + 1)) * hz / ((u1 - u0) / span);
        let dzdv = 0.5 * (h_at(iu, iv + 1) - h_at(iu, iv) + h_at(iu + 1, iv + 1) - h_at(iu + 1, iv)) * hz / ((v1 - v0) / span);
        let nrm = {
            let (a, b, c) = (-dzdu, -dzdv, 1.0f32);
            let l = (a * a + b * b + c * c).sqrt();
            (a / l, b / l, c / l)
        };
        let shade = (0.35 + 0.65 * (nrm.0 * light.0 + nrm.1 * light.1 + nrm.2 * light.2).max(0.0)).min(1.0);
        let z = field[iv * nx + iu];
        let hval = h_at(iu, iv).min(1.0);
        let base = match display {
            Display::Phase => {
                let [r, g, b] = phase_rgb(z.arg() as f32);
                [r, g, b]
            }
            Display::Plain => {
                let c = density_color(0.25 + 0.75 * hval);
                [c.r() as f32 / 255.0, c.g() as f32 / 255.0, c.b() as f32 / 255.0]
            }
            Display::Real => {
                let c = diverging((z.re / max_abs).clamp(-1.0, 1.0) as f32, look);
                [c.r() as f32 / 255.0, c.g() as f32 / 255.0, c.b() as f32 / 255.0]
            }
        };
        // the flat floor (|ψ| ≈ 0) is a neutral grey; colour appears with the height
        let w = (4.0 * hval.sqrt()).min(1.0);
        let floor_c = look.pick(Color32::from_gray(48), Color32::from_gray(200));
        let fl = [floor_c.r() as f32 / 255.0, floor_c.g() as f32 / 255.0, floor_c.b() as f32 / 255.0];
        let mix = |k: usize| (fl[k] * (1.0 - w) + base[k] * w) * shade;
        let col = Color32::from_rgb((mix(0) * 255.0) as u8, (mix(1) * 255.0) as u8, (mix(2) * 255.0) as u8);
        let i = mesh.vertices.len() as u32;
        for q in &p {
            mesh.colored_vertex(to_screen(*q), col);
        }
        mesh.add_triangle(i, i + 1, i + 2);
        mesh.add_triangle(i, i + 2, i + 3);
    }
    painter.add(Shape::mesh(mesh));
    // axis labels at the floor edges
    let f = look.font(12.5);
    // axis names at the middle of the two edges through the origin, just outside the floor
    let lu = to_screen(project(0.5 * lx as f32, -0.09 * span, 0.0));
    let lv = to_screen(project(-0.09 * span, 0.5 * ly as f32, 0.0));
    let o = to_screen(project(0.0, 0.0, 0.0));
    painter.text(lu, Align2::CENTER_CENTER, labels.0, f.clone(), look.text());
    painter.text(lv, Align2::CENTER_CENTER, labels.1, f.clone(), look.text());
    painter.text(
        o + vec2(0.0, look.px(4.0)),
        Align2::CENTER_TOP,
        "0",
        look.font(11.0),
        look.weak_text(),
    );
    painter.text(
        rect.left_top() + vec2(look.px(6.0), look.px(4.0)),
        Align2::LEFT_TOP,
        "height = |ψ|²   (drag to turn)",
        look.font(12.0),
        look.weak_text(),
    );
    painter.rect_stroke(rect, 2.0, Stroke::new(1.0, look.plot_frame()), StrokeKind::Inside);
}

fn glam_light() -> (f32, f32, f32) {
    let v = qb_core::glam::Vec3::new(-0.4, -0.6, 0.7).normalize();
    (v.x, v.y, v.z)
}
