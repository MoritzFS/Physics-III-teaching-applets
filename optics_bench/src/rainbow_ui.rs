//! User interface of the rainbow bench (PS02, exercise 8): the rays in a drop,
//! the deviation δ(θ) and the observer in a side view (top), the sky (middle)
//! and the controls (bottom).

use std::f32::consts::{PI, TAU};
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use eframe::egui::{self, pos2, vec2, Align2, Color32, ColorImage, FontId, Pos2, Rect, Sense, Shape, Stroke, TextureHandle, TextureOptions, Vec2};

use crate::dispersion_ui::{readout, ticks};
use crate::rainbow::{
    circle, descartes, line_color, order_power, path, path_power, polarisation, rays_at, reflect_power, sky_image, DropRays, Medium,
    RainbowParams, RainbowPreset, SkyGeom, Spectrum, Tables, DEG, K_MAX,
};
use crate::worker::Worker;

const PICK: Color32 = Color32::from_rgb(255, 220, 80);
const SUN: Color32 = Color32::from_rgb(255, 205, 100);
const DROP_EDGE: Color32 = Color32::from_rgb(150, 190, 245);
const BG: Color32 = Color32::from_gray(16);
const LABEL: Color32 = Color32::from_gray(170);

/// seconds for the swept ray to cross the drop from the centre to the edge
const SWEEP_SECONDS: f32 = 6.0;
/// where no ray of the order goes: in the δ(θ) plot and as a wedge at the exit
const FORBIDDEN: Color32 = Color32::from_rgba_premultiplied(60, 18, 16, 60);
const WEDGE: Color32 = Color32::from_rgba_premultiplied(130, 38, 32, 120);

const ORDER_LABEL: [&str; K_MAX + 1] =
    ["k = 0: straight through", "k = 1: primary bow", "k = 2: secondary bow", "k = 3: 3rd order", "k = 4: 4th order"];

#[derive(Clone, PartialEq)]
struct Request {
    params: RainbowParams,
    geom: SkyGeom,
}

struct Response {
    tables: Arc<Tables>,
    sky: ColorImage,
    geom: SkyGeom,
    millis: f32,
}

#[derive(Clone, Copy, PartialEq)]
enum DevDrag {
    Ray,
    Pick,
}

pub struct RainbowUi {
    pub params: RainbowParams,
    pub notes: String,
    worker: Option<Worker<Request, Response>>,
    tables: Option<Arc<Tables>>,
    sky: Option<(TextureHandle, SkyGeom)>,
    sky_millis: f32,
    dev_drag: Option<DevDrag>,
    /// the picked ray moves across the drop by itself
    sweep: bool,
    sweep_dir: f32,
    /// the most extreme δ of the reddest shown colour since the sweep started (degrees)
    sweep_extreme: Option<f64>,
}

impl Default for RainbowUi {
    fn default() -> Self {
        let (params, notes) = RainbowPreset::Exercise.setup();
        RainbowUi {
            params,
            notes,
            worker: None,
            tables: None,
            sky: None,
            sky_millis: 0.0,
            dev_drag: None,
            sweep: false,
            sweep_dir: 1.0,
            sweep_extreme: None,
        }
    }
}

/// only what the sky image depends on, so that moving the ray does not redraw the sky
fn sky_request(p: &RainbowParams) -> RainbowParams {
    RainbowParams { b: 0.0, rays: 0, drop_k: 0, drop_rays: DropRays::Picked, pick_deg: 0.0, plot_b: false, ..p.clone() }
}

fn dashed(painter: &egui::Painter, a: Pos2, b: Pos2, stroke: Stroke) {
    painter.extend(Shape::dashed_line(&[a, b], stroke, 5.0, 4.0));
}

/// an arc from the screen angle a0 to a1 the short way round; returns the middle angle
fn arc(painter: &egui::Painter, c: Pos2, r: f32, a0: f32, a1: f32, stroke: Stroke) -> f32 {
    let mut d = a1 - a0;
    while d > PI {
        d -= TAU;
    }
    while d < -PI {
        d += TAU;
    }
    let pts: Vec<Pos2> = (0..=16).map(|i| a0 + d * i as f32 / 16.0).map(|a| c + vec2(a.cos(), a.sin()) * r).collect();
    painter.add(Shape::line(pts, stroke));
    a0 + 0.5 * d
}

fn arrow(painter: &egui::Painter, from: Pos2, d: Vec2, len: f32, col: Color32) {
    let tip = from + d * len;
    painter.line_segment([from, tip], Stroke::new(1.5, col));
    painter.add(Shape::convex_polygon(vec![tip + d * 2.0, tip - d * 5.0 + d.rot90() * 3.5, tip - d * 5.0 - d.rot90() * 3.5], col, Stroke::NONE));
}

fn dir(a: f32) -> Vec2 {
    vec2(a.cos(), a.sin())
}

fn fmt_pct(v: f64) -> String {
    if v >= 10.0 {
        format!("{v:.0} %")
    } else if v >= 0.1 {
        format!("{v:.1} %")
    } else {
        format!("{v:.2} %")
    }
}

/// the points where a ray of order k meets the surface (entry, reflections, exit) and the direction it leaves in (y down)
fn drop_path(c: Pos2, r: f32, b: f64, n: f64, k: u32) -> (Vec<Pos2>, Vec2) {
    let pa = path(b, n, k);
    let pt = |a: f64| c + vec2(a.cos() as f32, -a.sin() as f32) * r;
    let mut a = std::f64::consts::PI - pa.theta;
    let mut pts = vec![pt(a)];
    for _ in 0..=k {
        a -= std::f64::consts::PI - 2.0 * pa.phi;
        pts.push(pt(a));
    }
    (pts, vec2(pa.dev.cos() as f32, pa.dev.sin() as f32))
}

/// the orders drawn in the δ(θ) plot: the enabled ones and the one in the drop
fn plot_orders(p: &RainbowParams) -> Vec<u32> {
    (0..=K_MAX as u32).filter(|&k| p.orders[k as usize] || k == p.main_order()).collect()
}

/// wavelength at which the bow of order k lies at δ (degrees), or None
fn cutoff_wavelength(p: &RainbowParams, k: u32, delta: f64) -> Option<f64> {
    let f = |l: f64| p.bow_deg(l, k).map(|d| d - delta);
    let (mut lo, mut hi) = (400.0, 700.0);
    let (flo, fhi) = (f(lo)?, f(hi)?);
    if flo * fhi > 0.0 {
        return None;
    }
    for _ in 0..40 {
        let m = 0.5 * (lo + hi);
        if (f(m)? < 0.0) == (flo < 0.0) {
            lo = m;
        } else {
            hi = m;
        }
    }
    Some(0.5 * (lo + hi))
}

/// what the observer sees at δ from the antisolar point
fn describe(p: &RainbowParams, delta: f64) -> String {
    if delta > 90.0 {
        return "on the sun's side of the sky".into();
    }
    let ls = p.wavelengths();
    let range = |k: u32| -> Option<(f64, f64)> {
        let a = p.bow_deg(ls[0], k)?;
        let b = p.bow_deg(*ls.last().unwrap(), k)?;
        Some((a.min(b), a.max(b)))
    };
    let colours = |k: u32| {
        if p.spectrum == Spectrum::White {
            return match cutoff_wavelength(p, k, delta) {
                Some(l) => format!("only λ > {l:.0} nm reaches you"),
                None => "only part of the spectrum reaches you".into(),
            };
        }
        // the lines whose bow lies beyond δ (primary) or inside it (secondary)
        let reach: Vec<String> = ls
            .iter()
            .filter(|&&l| p.bow_deg(l, k).is_some_and(|d| if k == 2 { d <= delta } else { d >= delta }))
            .map(|l| format!("{l:.0}"))
            .collect();
        format!("only {} nm reach{} you", reach.join(", "), if reach.len() == 1 { "es" } else { "" })
    };
    let r1 = if p.orders[1] { range(1) } else { None };
    let r2 = if p.orders[2] { range(2) } else { None };
    if let Some((lo, hi)) = r1 {
        if delta < lo {
            return if ls.len() > 1 { "inside the bow: all colours arrive and mix".into() } else { "inside the bow: lit".into() };
        }
        if delta <= hi {
            return format!("on the primary bow: {}", colours(1));
        }
    }
    if let Some((lo, hi)) = r2 {
        if delta > hi {
            return "outside the secondary bow: all colours arrive".into();
        }
        if delta >= lo {
            return format!("on the secondary bow: {}", colours(2));
        }
        if r1.is_some() {
            return "between the bows: Alexander's dark band".into();
        }
        return "inside the secondary bow: dark".into();
    }
    if r1.is_some() { "outside the bow: no light from it".into() } else { String::new() }
}

impl RainbowUi {
    pub fn load_preset(&mut self, p: RainbowPreset) {
        let (params, notes) = p.setup();
        self.params = params;
        self.notes = notes;
    }

    /// talk to the worker thread; call once per frame while the rainbow bench is shown
    pub fn update(&mut self, ctx: &egui::Context) {
        let Some(worker) = &mut self.worker else { return };
        if let Some(r) = worker.poll() {
            match &mut self.sky {
                Some((t, g)) => {
                    t.set(r.sky, TextureOptions::LINEAR);
                    *g = r.geom;
                }
                None => self.sky = Some((ctx.load_texture("rainbow_sky", r.sky, TextureOptions::LINEAR), r.geom)),
            }
            self.tables = Some(r.tables);
            self.sky_millis = r.millis;
        }
        if worker.busy() {
            ctx.request_repaint();
        }
    }

    fn request(&mut self, ctx: &egui::Context, geom: SkyGeom) {
        let worker = self.worker.get_or_insert_with(|| {
            let c = ctx.clone();
            let mut cache: Option<Arc<Tables>> = None;
            Worker::new(
                "rainbow",
                move |q: Request| {
                    let t0 = Instant::now();
                    let key = q.params.table_key();
                    let tables = match &cache {
                        Some(t) if t.key == key => t.clone(),
                        _ => Arc::new(Tables::new(key)),
                    };
                    cache = Some(tables.clone());
                    let sky = sky_image(&tables, &q.params, &q.geom);
                    Response { tables, sky, geom: q.geom, millis: t0.elapsed().as_secs_f32() * 1000.0 }
                },
                move || c.request_repaint(),
            )
        });
        worker.request(&Request { params: sky_request(&self.params), geom });
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let win_h = ui.ctx().content_rect().height();
        egui::Panel::top("r_views")
            .resizable(true)
            .default_size(win_h * 0.4)
            .min_size(200.0)
            .max_size(win_h * 0.65)
            .show(ui, |ui| {
                ui.add_space(4.0);
                ui.columns(3, |cols| {
                    self.drop_panel(&mut cols[0]);
                    self.deviation_panel(&mut cols[1]);
                    self.observer_panel(&mut cols[2]);
                });
            });
        egui::Panel::bottom("r_controls")
            .resizable(true)
            .default_size(255.0)
            .min_size(120.0)
            .max_size(win_h * 0.45)
            .show(ui, |ui| self.controls_ui(ui));
        egui::CentralPanel::default().show(ui, |ui| self.sky_panel(ui));
    }

    // ------------------------------------------------------------ drop

    fn drop_panel(&mut self, ui: &mut egui::Ui) {
        let p = &mut self.params;
        ui.horizontal(|ui| {
            ui.strong("DROP");
            ui.weak("reflections inside");
            for k in 0..=K_MAX as u32 {
                if ui.selectable_value(&mut p.drop_k, k, k.to_string()).on_hover_text(ORDER_LABEL[k as usize]).clicked() {
                    p.snap_to_descartes();
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                egui::ComboBox::from_id_salt("r_drop_rays")
                    .selected_text(match p.drop_rays {
                        DropRays::Picked => "picked ray",
                        DropRays::ToEye => "rays to your eye",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut p.drop_rays, DropRays::Picked, "picked ray").on_hover_text("the ray at the height b: drag in the drop");
                        ui.selectable_value(&mut p.drop_rays, DropRays::ToEye, "rays to your eye")
                            .on_hover_text("the rays that leave at the picked angle δ, i.e. reach your eye from the picked drop");
                    });
            });
        });
        let avail = ui.available_size();
        let (rect, resp) = ui.allocate_exact_size(vec2(avail.x, (avail.y - 28.0).max(40.0)), Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3.0, BG);
        // room around the drop for the wedge at the exit
        let r = (rect.height().min(rect.width()) * 0.34).max(10.0);
        let c = pos2(rect.left() + rect.width() * 0.58, rect.center().y);
        if resp.double_clicked() {
            p.snap_to_descartes();
            self.sweep = false;
        } else if (resp.dragged() || resp.clicked())
            && let Some(q) = resp.interact_pointer_pos()
        {
            p.b = ((c.y - q.y) / r).clamp(0.0, 0.999);
            p.drop_rays = DropRays::Picked;
            self.sweep = false;
        }
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui
                .button(if self.sweep { "⏸ stop" } else { "▶ sweep" })
                .on_hover_text("move the ray from the centre of the drop to its edge and back")
                .clicked()
            {
                self.sweep = !self.sweep;
                self.sweep_extreme = None;
                p.drop_rays = DropRays::Picked;
            }
            ui.label("height b");
            if ui.add(egui::Slider::new(&mut p.b, 0.0..=0.999).custom_formatter(|v, _| format!("{v:.3}"))).changed() {
                p.drop_rays = DropRays::Picked;
                self.sweep = false;
            }
        });
        if self.sweep {
            let dt = ui.input(|i| i.stable_dt).min(0.1);
            p.b += self.sweep_dir * dt / SWEEP_SECONDS;
            if p.b >= 0.999 {
                p.b = 0.999;
                self.sweep_dir = -1.0;
            } else if p.b <= 0.0 {
                p.b = 0.0;
                self.sweep_dir = 1.0;
            }
            ui.ctx().request_repaint();
        }
        if resp.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
        }
        let k = p.main_order();
        let lambda_mid = if p.spectrum == Spectrum::Single { p.wavelength_nm as f64 } else { 589.0 };
        let n_mid = p.n(lambda_mid);
        let big = 3.0 * rect.width().max(rect.height());

        painter.circle(c, r, Color32::from_rgba_unmultiplied(90, 140, 220, 28), Stroke::new(1.2, DROP_EDGE));
        // the fan
        for i in 0..p.rays {
            let b = (i as f64 + 0.5) / p.rays as f64;
            let (pts, out) = drop_path(c, r, b, n_mid, k);
            let mut line = vec![pos2(rect.left(), pts[0].y)];
            line.extend(&pts);
            line.push(*pts.last().unwrap() + out * big);
            painter.add(Shape::line(line, Stroke::new(1.0, Color32::from_white_alpha(45))));
        }
        let shown = p.shown();
        match p.drop_rays {
            DropRays::Picked => {
                let b = p.b as f64;
                for &l in &shown {
                    let (pts, out) = drop_path(c, r, b, p.n(l), k);
                    let mut line = pts.clone();
                    line.push(*pts.last().unwrap() + out * big);
                    painter.add(Shape::line(line, Stroke::new(1.6, line_color(l))));
                }
                let (pts, _) = drop_path(c, r, b, n_mid, k);
                painter.line_segment([pos2(rect.left(), pts[0].y), pts[0]], Stroke::new(1.8, Color32::WHITE));
                self_lost_light(&painter, c, r, b, n_mid, k, big);
                limit_marks(&painter, c, r, b, p, k);
                angle_marks(&painter, c, r, b, n_mid, k);
            }
            DropRays::ToEye => {
                let delta = p.pick_deg as f64 * DEG;
                for &l in &shown {
                    let n = p.n(l);
                    for b in rays_at(n, k, delta) {
                        let (pts, out) = drop_path(c, r, b, n, k);
                        let mut line = vec![pos2(rect.left(), pts[0].y)];
                        line.extend(&pts);
                        line.push(*pts.last().unwrap() + out * big);
                        painter.add(Shape::line(line, Stroke::new(1.6, line_color(l))));
                    }
                }
            }
        }
        // the Descartes ray
        if let Some(d) = descartes(n_mid, k) {
            let y = c.y - d.theta.sin() as f32 * r;
            let x = rect.left() + 3.0;
            painter.add(Shape::convex_polygon(vec![pos2(x, y - 5.0), pos2(x + 8.0, y), pos2(x, y + 5.0)], PICK, Stroke::NONE));
            painter.text(pos2(x + 11.0, y), Align2::LEFT_CENTER, "Descartes ray", FontId::proportional(10.0), PICK);
        }
        let g = painter.text(rect.left_bottom() + vec2(6.0, -6.0), Align2::LEFT_BOTTOM, "sunlight", FontId::proportional(11.0), SUN);
        arrow(&painter, pos2(g.right() + 4.0, g.center().y), vec2(1.0, 0.0), 18.0, SUN);
        // readout
        let text = match p.drop_rays {
            DropRays::Picked => {
                let b = p.b as f64;
                let ends = [shown[0], *shown.last().unwrap()];
                let pa: Vec<_> = ends.iter().map(|&l| path(b, p.n(l), k)).collect();
                let (s, pp) = path_power(b, n_mid, k);
                let mut t = format!("b = {b:.3}   θ = {:.1}°", pa[0].theta / DEG);
                if shown.len() > 1 {
                    t += &format!(
                        "\nφ = {:.2}° … {:.2}°\nδ = {:.2}° … {:.2}°   ({:.0} … {:.0} nm)",
                        pa[0].phi / DEG,
                        pa[1].phi / DEG,
                        pa[0].delta / DEG,
                        pa[1].delta / DEG,
                        ends[0],
                        ends[1]
                    );
                } else {
                    t += &format!("\nφ = {:.2}°\nδ = {:.2}°", pa[0].phi / DEG, pa[0].delta / DEG);
                }
                t += &format!("\nthis path: {} of the ray", fmt_pct(100.0 * (s + pp)));
                // the bow of the reddest colour: no ray goes beyond it
                let red = *shown.last().unwrap();
                if let Some((d, is_max)) = extreme(p.n(red), k) {
                    let word = if is_max { "largest" } else { "smallest" };
                    t += &format!("\n{word} possible δ: {:.2}° ({red:.0} nm)", d / DEG);
                    if self.sweep {
                        let now = path(b, p.n(red), k).delta;
                        let e = self.sweep_extreme.map_or(now, |x| if is_max { x.max(now) } else { x.min(now) });
                        self.sweep_extreme = Some(e);
                        t += &format!("\n{word} so far:    {:.2}°", e / DEG);
                    }
                }
                t
            }
            DropRays::ToEye => {
                let d = p.pick_deg as f64 * DEG;
                let counts: Vec<usize> = shown.iter().map(|&l| rays_at(p.n(l), k, d).len()).collect();
                if counts.iter().all(|&c| c == 0) {
                    format!("no light of order {k} leaves at δ = {:.1}°", p.pick_deg)
                } else if counts.iter().all(|&c| c == counts[0]) {
                    format!("δ = {:.1}°: {} ray{} per colour", p.pick_deg, counts[0], if counts[0] == 1 { "" } else { "s" })
                } else {
                    format!("δ = {:.1}°: some colours only", p.pick_deg)
                }
            }
        };
        // a corner the rays of this order leave free
        if k >= 3 {
            readout(&painter, rect.right_bottom() - vec2(6.0, 6.0), Align2::RIGHT_BOTTOM, text);
        } else {
            readout(&painter, rect.right_top() + vec2(-6.0, 6.0), Align2::RIGHT_TOP, text);
        }
    }

    // ------------------------------------------------------------ deviation

    fn deviation_panel(&mut self, ui: &mut egui::Ui) {
        let tables = self.tables.clone();
        let p = &mut self.params;
        ui.horizontal(|ui| {
            ui.strong("DEVIATION δ(θ)");
            ui.weak("angle from the antisolar point");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.checkbox(&mut p.plot_b, "against b").on_hover_text("plot against the height b = sin θ: the rays are evenly spaced in b");
            });
        });
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let strip_w = 58.0;
        let plot = Rect::from_min_max(rect.min + vec2(38.0, 12.0), rect.max - vec2(strip_w + 8.0, 22.0));
        let strip = Rect::from_min_max(pos2(plot.right() + 8.0, plot.top()), pos2(rect.right() - 2.0, plot.bottom()));
        if plot.width() < 20.0 || plot.height() < 20.0 {
            return;
        }
        let orders = plot_orders(p);
        let shown = p.shown();
        let mut y_max: f64 = 70.0;
        for &k in &orders {
            if k == 0 {
                y_max = 180.0;
            }
            for &l in &shown {
                if let Some(d) = p.bow_deg(l, k) {
                    y_max = y_max.max(d + 12.0);
                }
            }
        }
        y_max = y_max.max(p.pick_deg as f64 + 8.0).min(180.0);
        let plot_b = p.plot_b;
        let fx = |b: f64| plot.left() + (if plot_b { b } else { b.clamp(0.0, 1.0).asin() / (0.5 * std::f64::consts::PI) }) as f32 * plot.width();
        let bx = |x: f32| -> f64 {
            let f = ((x - plot.left()) / plot.width()).clamp(0.0, 1.0) as f64;
            if plot_b { f.min(0.999) } else { (f * 0.5 * std::f64::consts::PI).sin().min(0.999) }
        };
        let fy = |d: f64| plot.bottom() - (d / y_max) as f32 * plot.height();
        let yd = |y: f32| ((plot.bottom() - y) / plot.height()) as f64 * y_max;
        let b_of = |i: usize, n: usize| {
            let f = i as f64 / n as f64;
            (if plot_b { f } else { (f * 0.5 * std::f64::consts::PI).sin() }).min(1.0 - 1e-9)
        };

        // input
        if resp.drag_started() {
            let near = resp.interact_pointer_pos().is_some_and(|q| (q.y - fy(p.pick_deg as f64)).abs() < 8.0);
            self.dev_drag = Some(if near { DevDrag::Pick } else { DevDrag::Ray });
        }
        if resp.drag_stopped() {
            self.dev_drag = None;
        }
        if let Some(q) = resp.interact_pointer_pos().filter(|_| resp.dragged() || resp.clicked()) {
            match self.dev_drag {
                Some(DevDrag::Pick) => p.pick_deg = yd(q.y).clamp(0.0, 180.0) as f32,
                _ => {
                    p.b = bx(q.x) as f32;
                    p.drop_rays = DropRays::Picked;
                }
            }
        }
        if let Some(q) = resp.hover_pos() {
            let near = (q.y - fy(p.pick_deg as f64)).abs() < 8.0;
            ui.ctx().set_cursor_icon(if near || self.dev_drag == Some(DevDrag::Pick) {
                egui::CursorIcon::ResizeVertical
            } else {
                egui::CursorIcon::ResizeHorizontal
            });
        }

        let painter = ui.painter_at(rect);
        painter.rect_filled(plot, 3.0, Color32::from_gray(22));
        let grid = Stroke::new(1.0, Color32::from_white_alpha(14));
        let font = FontId::monospace(9.0);
        for v in ticks(0.0, y_max, 5.0) {
            let y = fy(v);
            painter.line_segment([pos2(plot.left(), y), pos2(plot.right(), y)], grid);
            painter.text(pos2(plot.left() - 4.0, y), Align2::RIGHT_CENTER, format!("{v:.0}°"), font.clone(), LABEL);
        }
        let xt: Vec<(f64, String)> = if plot_b {
            (0..=5).map(|i| (i as f64 * 0.2, format!("{:.1}", i as f64 * 0.2))).collect()
        } else {
            (0..=6).map(|i| ((i as f64 * 15.0 * DEG).sin(), format!("{}°", i * 15))).collect()
        };
        for (b, t) in xt {
            let x = fx(b);
            painter.line_segment([pos2(x, plot.top()), pos2(x, plot.bottom())], grid);
            painter.text(pos2(x, plot.bottom() + 2.0), Align2::CENTER_TOP, t, font.clone(), LABEL);
        }
        painter.text(plot.right_bottom() + vec2(0.0, 12.0), Align2::RIGHT_TOP, if plot_b { "b = sin θ" } else { "θ" }, FontId::proportional(11.0), LABEL);
        painter.text(plot.left_top() + vec2(-6.0, -2.0), Align2::RIGHT_BOTTOM, "δ", FontId::proportional(11.0), LABEL);

        let clip = ui.painter_at(plot);
        const N: usize = 300;
        // the spread of the colours
        if shown.len() > 1 {
            let (n0, n1) = (p.n(shown[0]), p.n(*shown.last().unwrap()));
            for &k in &orders {
                for i in 0..N {
                    let (b0, b1) = (b_of(i, N), b_of(i + 1, N));
                    let (a0, a1) = (path(b0, n0, k).delta / DEG, path(b1, n0, k).delta / DEG);
                    let (z0, z1) = (path(b0, n1, k).delta / DEG, path(b1, n1, k).delta / DEG);
                    let quad = vec![pos2(fx(b0), fy(a0)), pos2(fx(b1), fy(a1)), pos2(fx(b1), fy(z1)), pos2(fx(b0), fy(z0))];
                    clip.add(Shape::convex_polygon(quad, Color32::from_white_alpha(18), Stroke::NONE));
                }
            }
        }
        if p.reflection {
            let pts: Vec<Pos2> = (0..=N).map(|i| b_of(i, N)).map(|b| pos2(fx(b), fy(2.0 * b.asin() / DEG))).collect();
            clip.extend(Shape::dashed_line(&pts, Stroke::new(1.2, Color32::from_gray(150)), 5.0, 4.0));
        }
        for &k in &orders {
            for &l in &shown {
                let n = p.n(l);
                let pts: Vec<Pos2> = (0..=N).map(|i| b_of(i, N)).map(|b| pos2(fx(b), fy(path(b, n, k).delta / DEG))).collect();
                clip.add(Shape::line(pts, Stroke::new(1.5, line_color(l))));
            }
        }
        // the bows: extreme of δ
        let ends = if shown.len() > 1 { vec![shown[0], *shown.last().unwrap()] } else { shown.clone() };
        for &k in orders.iter().filter(|&&k| k > 0) {
            for &l in &shown {
                if let Some(d) = descartes(p.n(l), k) {
                    let q = pos2(fx(d.theta.sin()), fy(d.delta / DEG));
                    clip.circle_filled(q, 3.0, line_color(l));
                    if ends.contains(&l) && q.y > plot.top() + 6.0 {
                        let up = l == ends[ends.len() - 1] && ends.len() > 1;
                        clip.text(
                            q + vec2(6.0, if up { -7.0 } else { 7.0 }),
                            Align2::LEFT_CENTER,
                            format!("{:.1}°", d.delta / DEG),
                            FontId::proportional(11.0),
                            line_color(l),
                        );
                    }
                }
            }
        }
        // the bow of the order in the drop: no ray of it goes beyond this line
        let k_main = p.main_order();
        if let Some((d, is_max)) = extreme(p.n(*shown.last().unwrap()), k_main) {
            let y = fy(d / DEG);
            let col = Color32::from_rgb(235, 130, 120);
            if orders.len() == 1 {
                let far = if is_max { plot.top() } else { plot.bottom() };
                clip.rect_filled(Rect::from_x_y_ranges(plot.x_range(), egui::Rangef::new(y.min(far), y.max(far))), 0.0, FORBIDDEN);
            }
            clip.line_segment([pos2(plot.left(), y), pos2(plot.right(), y)], Stroke::new(1.0, col));
            let (anchor, dy) = if is_max { (Align2::LEFT_BOTTOM, -2.0) } else { (Align2::LEFT_TOP, 2.0) };
            let word = if is_max { "above" } else { "below" };
            clip.text(pos2(plot.left() + 4.0, y + dy), anchor, format!("no ray of order {k_main} {word} {:.1}°", d / DEG), FontId::proportional(10.0), col);
        }
        // picked ray and picked angle
        if p.drop_rays == DropRays::Picked {
            let x = fx(p.b as f64);
            dashed(&clip, pos2(x, plot.top()), pos2(x, plot.bottom()), Stroke::new(1.0, Color32::from_white_alpha(150)));
            // where the picked ray is on each curve
            for &l in &shown {
                let q = pos2(x, fy(path(p.b as f64, p.n(l), k_main).delta / DEG));
                clip.circle(q, 4.0, line_color(l), Stroke::new(1.2, Color32::WHITE));
            }
        }
        let yp = fy(p.pick_deg as f64);
        dashed(&clip, pos2(plot.left(), yp), pos2(plot.right(), yp), Stroke::new(1.4, PICK));
        clip.text(pos2(plot.right() - 4.0, yp - 3.0), Align2::RIGHT_BOTTOM, format!("δ = {:.1}°", p.pick_deg), FontId::proportional(11.0), PICK);
        for &k in &orders {
            for &l in &shown {
                for b in rays_at(p.n(l), k, p.pick_deg as f64 * DEG) {
                    clip.circle(pos2(fx(b), yp), 3.5, line_color(l), Stroke::new(1.0, Color32::WHITE));
                }
            }
        }
        // what the sky looks like at each δ, and how bright it is
        painter.text(strip.center_top() - vec2(0.0, 2.0), Align2::CENTER_BOTTOM, "sky", FontId::proportional(10.0), LABEL);
        if let Some(t) = &tables {
            let e = p.exposure(t);
            let mut prof = vec![];
            let mut y = strip.top();
            while y < strip.bottom() {
                let d = yd(y + 0.5);
                painter.rect_filled(Rect::from_min_size(pos2(strip.left(), y), vec2(14.0, 1.0)), 0.0, p.color_at(t, d));
                let v = 1.0 - (-t.luminance(d) * e).exp();
                prof.push(pos2(strip.left() + 18.0 + v * (strip.width() - 20.0), y + 0.5));
                y += 1.0;
            }
            painter.add(Shape::line(prof, Stroke::new(1.0, Color32::from_white_alpha(170))));
        }
        // hover readout
        if let Some(q) = resp.hover_pos().filter(|q| plot.contains(*q)) {
            let b = bx(q.x);
            let mut t = format!("θ = {:.1}°   b = {b:.3}", b.asin() / DEG);
            for &k in &orders {
                let ds: Vec<f64> = ends.iter().map(|&l| path(b, p.n(l), k).delta / DEG).collect();
                t += &if ds.len() > 1 { format!("\nk = {k}: δ = {:.2}° … {:.2}°", ds[0], ds[1]) } else { format!("\nk = {k}: δ = {:.2}°", ds[0]) };
            }
            readout(&painter, plot.left_top() + vec2(4.0, 4.0), Align2::LEFT_TOP, t);
        }
    }

    // ------------------------------------------------------------ observer

    fn observer_panel(&mut self, ui: &mut egui::Ui) {
        let tables = self.tables.clone();
        let p = &mut self.params;
        ui.horizontal(|ui| {
            ui.strong("OBSERVER");
            ui.weak("side view: sun behind you, rain in front");
        });
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        if rect.width() < 40.0 || rect.height() < 60.0 {
            return;
        }
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3.0, BG);
        let ground_y = rect.bottom() - 22.0;
        painter.rect_filled(Rect::from_min_max(pos2(rect.left(), ground_y), rect.max), 0.0, Color32::from_rgb(26, 36, 24));
        let eye = pos2(rect.left() + 34.0, ground_y - 32.0);
        let e = p.sun_elevation_deg * PI / 180.0;
        if (resp.dragged() || resp.clicked()) && resp.interact_pointer_pos().is_some() {
            let q = resp.interact_pointer_pos().unwrap();
            let a = (eye.y - q.y).atan2((q.x - eye.x).max(1.0)) / PI * 180.0;
            p.pick_deg = (a + p.sun_elevation_deg).clamp(0.0, 90.0);
        }
        if resp.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
        }
        // rain
        let rain_x0 = rect.left() + rect.width() * 0.55;
        let frac = |v: f32| v - v.floor();
        for i in 0..260 {
            let x = rain_x0 + frac(i as f32 * 0.618_034) * (rect.right() - rain_x0 - 4.0);
            let y = rect.top() + frac(i as f32 * 0.754_877 + 0.3) * (ground_y - rect.top());
            painter.line_segment([pos2(x, y), pos2(x - 1.5, y + 6.0)], Stroke::new(1.0, Color32::from_rgba_unmultiplied(140, 175, 230, 60)));
        }
        // sunlight from behind: parallel rays going down at the sun's elevation
        let sd = dir(e);
        for i in 0..6 {
            let start = pos2(rect.left(), rect.top() + 10.0 + 16.0 * i as f32);
            painter.line_segment([start, start + sd * 2.0 * rect.width()], Stroke::new(1.0, SUN.gamma_multiply(0.35)));
            let mid = start + sd * (40.0 + 25.0 * i as f32);
            painter.add(Shape::convex_polygon(
                vec![mid + sd * 6.0, mid + sd.rot90() * 3.0, mid - sd.rot90() * 3.0],
                SUN.gamma_multiply(0.7),
                Stroke::NONE,
            ));
        }
        painter.text(rect.left_top() + vec2(4.0, 2.0), Align2::LEFT_TOP, format!("sunlight, {:.0}° up", p.sun_elevation_deg), FontId::proportional(10.0), SUN);
        // towards the antisolar point
        let to_ground = if sd.y > 0.01 { (ground_y - eye.y) / sd.y } else { rect.width() };
        let anti_end = eye + sd * to_ground.min(rect.width() * 0.45);
        dashed(&painter, eye, anti_end, Stroke::new(1.0, SUN.gamma_multiply(0.8)));
        painter.text(anti_end + vec2(4.0, -2.0), Align2::LEFT_BOTTOM, "to the antisolar point", FontId::proportional(10.0), SUN.gamma_multiply(0.8));

        let ls = p.wavelengths();
        let ends = if ls.len() > 1 { vec![ls[0], *ls.last().unwrap()] } else { ls.clone() };
        let cone_orders: Vec<u32> = [1, 2].into_iter().filter(|&k| p.orders[k as usize]).collect();
        // the directions from the eye in which the bows are seen
        for &k in &cone_orders {
            for &l in &ends {
                if let Some(d) = p.bow_deg(l, k) {
                    let a = (d as f32 - p.sun_elevation_deg) * PI / 180.0;
                    painter.line_segment([eye, eye + vec2(a.cos(), -a.sin()) * 2.0 * rect.width()], Stroke::new(1.0, line_color(l).gamma_multiply(0.45)));
                }
            }
        }
        // the picked drop
        let elev = (p.pick_deg - p.sun_elevation_deg) * PI / 180.0;
        let status = describe(p, p.pick_deg as f64);
        if p.pick_deg <= 90.0 && elev.cos() > 0.02 {
            let u = vec2(elev.cos(), -elev.sin());
            let mut t = (rect.left() + rect.width() * 0.75 - eye.x) / u.x;
            if eye.y + t * u.y < rect.top() + 10.0 {
                t = (eye.y - rect.top() - 10.0) / -u.y;
            }
            if eye.y + t * u.y > ground_y - 3.0 {
                t = (ground_y - 3.0 - eye.y) / u.y;
            }
            let drop = eye + u * t;
            let back = PI - e;
            let len = t * 1.25;
            for &k in &cone_orders {
                if k == 1
                    && let Some(d) = p.bow_deg(*ends.last().unwrap(), 1).into_iter().chain(p.bow_deg(ends[0], 1)).reduce(f64::max)
                {
                    let a1 = back + d as f32 * PI / 180.0;
                    painter.add(Shape::convex_polygon(
                        vec![drop, drop + vec2(back.cos(), -back.sin()) * len, drop + vec2(a1.cos(), -a1.sin()) * len],
                        Color32::from_white_alpha(10),
                        Stroke::NONE,
                    ));
                }
                let lines = if k == 1 { p.shown() } else { ends.clone() };
                for &l in &lines {
                    if let Some(d) = p.bow_deg(l, k) {
                        let a = back + d as f32 * PI / 180.0;
                        let end = drop + vec2(a.cos(), -a.sin()) * len;
                        let st = Stroke::new(1.3, line_color(l));
                        if k == 1 {
                            painter.line_segment([drop, end], st);
                        } else {
                            dashed(&painter, drop, end, st);
                        }
                    }
                }
            }
            dashed(&painter, eye, drop, Stroke::new(1.0, Color32::from_white_alpha(120)));
            painter.circle(drop, 4.5, Color32::from_rgb(160, 200, 255), Stroke::new(1.0, PICK));
            painter.text(
                drop + vec2(-8.0, 8.0),
                Align2::RIGHT_TOP,
                format!("{:.1}° up, {:.1}° from the antisolar point", p.pick_deg - p.sun_elevation_deg, p.pick_deg),
                FontId::proportional(10.0),
                PICK,
            );
        } else {
            painter.text(rect.center(), Align2::CENTER_CENTER, "the picked angle is on the sun's side (see the sky)", FontId::proportional(11.0), LABEL);
        }
        // the observer
        let st = Stroke::new(1.6, Color32::from_gray(220));
        painter.circle_stroke(eye + vec2(0.0, -2.0), 5.0, st);
        painter.line_segment([eye + vec2(0.0, 3.0), pos2(eye.x, ground_y - 12.0)], st);
        painter.line_segment([pos2(eye.x, ground_y - 12.0), pos2(eye.x - 6.0, ground_y)], st);
        painter.line_segment([pos2(eye.x, ground_y - 12.0), pos2(eye.x + 6.0, ground_y)], st);
        painter.line_segment([eye + vec2(-7.0, 10.0), eye + vec2(7.0, 10.0)], st);
        // what reaches the eye from there
        let sw = Rect::from_min_size(pos2(rect.right() - 36.0, rect.bottom() - 18.0), vec2(30.0, 13.0));
        if let Some(t) = &tables {
            painter.rect_filled(sw, 2.0, p.color_at(t, p.pick_deg as f64));
            painter.rect_stroke(sw, 2.0, Stroke::new(1.0, Color32::from_white_alpha(90)), egui::StrokeKind::Outside);
        }
        painter.text(sw.left_center() - vec2(6.0, 0.0), Align2::RIGHT_CENTER, status, FontId::proportional(11.0), Color32::WHITE);
    }

    // ------------------------------------------------------------ sky

    fn sky_panel(&mut self, ui: &mut egui::Ui) {
        let p = &mut self.params;
        ui.horizontal(|ui| {
            ui.strong("SKY");
            ui.weak(if p.toward_sun { "looking towards the sun" } else { "what you see with the sun behind you" });
            ui.checkbox(&mut p.toward_sun, "look towards the sun");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.weak(format!("{:.0} ms", self.sky_millis)).on_hover_text("time to compute the brightness tables and the sky");
            });
        });
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click());
        if rect.width() < 40.0 || rect.height() < 40.0 {
            return;
        }
        // the browser computes on one thread: fewer pixels there
        let ppp = ui.ctx().pixels_per_point().min(if cfg!(target_arch = "wasm32") { 1.0 } else { 1.5 });
        let geom = SkyGeom::new((rect.width() * ppp) as usize, (rect.height() * ppp) as usize, &self.params);
        self.request(ui.ctx(), geom);
        let p = &mut self.params;
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::BLACK);
        let Some((tex, g)) = &self.sky else { return };
        let g = *g;
        painter.image(tex.id(), rect, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
        let to_screen = |az: f64, el: f64| {
            let (x, y) = g.xy(az, el);
            rect.min + vec2(x as f32 / g.w as f32 * rect.width(), y as f32 / g.h as f32 * rect.height())
        };
        let from_screen = |q: Pos2| g.az_el(((q.x - rect.left()) / rect.width() * g.w as f32) as f64, ((q.y - rect.top()) / rect.height() * g.h as f32) as f64);
        let sun_el = p.sun_elevation_deg as f64;
        let anti = crate::rainbow::antisolar_dir(sun_el);
        let delta_at = |az: f64, el: f64| {
            let v = g.dir(az, el);
            (v[0] * anti[0] + v[1] * anti[1] + v[2] * anti[2]).clamp(-1.0, 1.0).acos() / DEG
        };
        // horizon and azimuth ticks
        let hy = to_screen(0.0, 0.0).y;
        painter.line_segment([pos2(rect.left(), hy), pos2(rect.right(), hy)], Stroke::new(1.0, Color32::from_white_alpha(70)));
        painter.text(pos2(rect.right() - 6.0, hy - 3.0), Align2::RIGHT_BOTTOM, "horizon", FontId::proportional(10.0), LABEL);
        let half = 0.5 * g.w as f64 * g.scale;
        let mut az = -(half / 30.0).floor() * 30.0;
        while az <= half {
            let q = to_screen(az, 0.0);
            painter.line_segment([q, q + vec2(0.0, 5.0)], Stroke::new(1.0, Color32::from_white_alpha(90)));
            if az != 0.0 && az.abs() < 179.0 {
                painter.text(q + vec2(0.0, 6.0), Align2::CENTER_TOP, format!("{az:.0}°"), FontId::monospace(9.0), LABEL);
            }
            az += 30.0;
        }
        // antisolar point or sun
        if g.toward_sun {
            let q = to_screen(0.0, sun_el);
            let r = ((p.sun_diameter_deg as f64 * 0.5 / g.scale) as f32 / g.h as f32 * rect.height()).max(4.0);
            painter.circle_filled(q, r * 3.0, Color32::from_white_alpha(25));
            painter.circle_filled(q, r, Color32::WHITE);
            painter.text(q + vec2(r + 4.0, -r - 2.0), Align2::LEFT_BOTTOM, "sun", FontId::proportional(11.0), SUN);
        } else {
            let q = to_screen(0.0, -sun_el);
            if q.y < rect.bottom() - 4.0 {
                let st = Stroke::new(1.5, SUN);
                painter.line_segment([q - vec2(6.0, 0.0), q + vec2(6.0, 0.0)], st);
                painter.line_segment([q - vec2(0.0, 6.0), q + vec2(0.0, 6.0)], st);
                painter.text(q + vec2(8.0, -3.0), Align2::LEFT_BOTTOM, "antisolar point", FontId::proportional(10.0), SUN);
            } else {
                let x = rect.center().x;
                painter.text(pos2(x, rect.bottom() - 4.0), Align2::CENTER_BOTTOM, format!("antisolar point: {sun_el:.0}° below the horizon"), FontId::proportional(10.0), SUN);
            }
        }
        // the picked angle: a circle around the antisolar point
        for arc_pts in circle(&g, sun_el, p.pick_deg as f64) {
            let pts: Vec<Pos2> = arc_pts.iter().map(|&(az, el)| to_screen(az, el)).collect();
            painter.extend(Shape::dashed_line(&pts, Stroke::new(1.0, PICK.gamma_multiply(0.8)), 6.0, 5.0));
        }
        // status
        let mut status = describe(p, p.pick_deg as f64);
        if p.orders[1]
            && !g.toward_sun
            && let Some(d) = p.bow_deg(700.0, 1)
            && d < sun_el
        {
            status = format!("the sun is higher than {d:.0}°: the primary bow is below the horizon");
        }
        if !status.is_empty() {
            let galley = painter.layout_no_wrap(format!("picked δ = {:.1}°: {status}", p.pick_deg), FontId::proportional(12.0), PICK);
            let r = Rect::from_min_size(rect.left_top() + vec2(6.0, 6.0), galley.size()).expand(3.0);
            painter.rect_filled(r, 3.0, Color32::from_black_alpha(150));
            painter.galley(r.min + vec2(3.0, 3.0), galley, PICK);
        }
        // hover and click
        if let Some(q) = resp.hover_pos() {
            let (az, el) = from_screen(q);
            let d = delta_at(az, el);
            readout(
                &painter,
                rect.right_top() + vec2(-6.0, 6.0),
                Align2::RIGHT_TOP,
                format!("{d:.1}° from the antisolar point\n{:.1}° from the sun\nelevation {el:.1}°", 180.0 - d),
            );
            ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
            if resp.clicked() {
                p.pick_deg = d as f32;
            }
        }
    }

    // ------------------------------------------------------------ controls

    fn controls_ui(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        if !self.notes.trim().is_empty() {
            ui.horizontal_wrapped(|ui| {
                ui.strong("About this setup:");
                ui.label(&self.notes);
            });
            ui.separator();
        }
        egui::ScrollArea::vertical().id_salt("rainbow_controls").auto_shrink([false, false]).show(ui, |ui| {
            ui.columns(4, |cols| {
                self.light_controls(&mut cols[0]);
                self.drop_controls(&mut cols[1]);
                self.path_controls(&mut cols[2]);
                self.view_controls(&mut cols[3]);
            });
        });
    }

    fn light_controls(&mut self, ui: &mut egui::Ui) {
        let p = &mut self.params;
        ui.strong("LIGHT");
        egui::Grid::new("r_light").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
            ui.label("spectrum");
            egui::ComboBox::from_id_salt("r_spectrum").selected_text(p.spectrum.label()).show_ui(ui, |ui| {
                for s in Spectrum::ALL {
                    ui.selectable_value(&mut p.spectrum, s, s.label());
                }
            });
            ui.end_row();
            if p.spectrum == Spectrum::Single {
                ui.label("wavelength");
                ui.horizontal(|ui| {
                    ui.add(egui::Slider::new(&mut p.wavelength_nm, 400.0..=700.0).suffix(" nm").step_by(1.0));
                    let (r, _) = ui.allocate_exact_size(vec2(14.0, 14.0), Sense::hover());
                    ui.painter().circle_filled(r.center(), 6.0, line_color(p.wavelength_nm as f64));
                });
                ui.end_row();
            }
            ui.label("sun diameter");
            ui.horizontal(|ui| {
                ui.add(egui::Slider::new(&mut p.sun_diameter_deg, 0.0..=3.0).suffix("°").fixed_decimals(2))
                    .on_hover_text("0 = a point source, as in the exercise");
                if ui.small_button("real").on_hover_text("the sun: 0.53°").clicked() {
                    p.sun_diameter_deg = 0.53;
                }
                if ui.small_button("point").clicked() {
                    p.sun_diameter_deg = 0.0;
                }
            });
            ui.end_row();
            ui.label("sun elevation");
            ui.add(egui::Slider::new(&mut p.sun_elevation_deg, 0.0..=60.0).suffix("°").step_by(1.0));
            ui.end_row();
            ui.checkbox(&mut p.polarizer, "polariser");
            ui.add_enabled(p.polarizer, egui::Slider::new(&mut p.polarizer_deg, 0.0..=180.0).suffix("°").step_by(5.0))
                .on_hover_text("transmission axis: 0° = horizontal, 90° = vertical");
            ui.end_row();
        });
    }

    fn drop_controls(&mut self, ui: &mut egui::Ui) {
        let p = &mut self.params;
        ui.strong("DROP");
        egui::Grid::new("r_drop").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
            ui.label("medium");
            let mut m = p.medium;
            egui::ComboBox::from_id_salt("r_medium").selected_text(m.label()).show_ui(ui, |ui| {
                for x in Medium::ALL.into_iter().filter(|&x| x != Medium::Custom) {
                    ui.selectable_value(&mut m, x, x.label());
                }
            });
            if m != p.medium {
                p.set_medium(m);
            }
            ui.end_row();
            ui.label("n at 589 nm");
            let before = p.n_d;
            ui.add(egui::Slider::new(&mut p.n_d, 1.05..=2.4).custom_formatter(|v, _| format!("{v:.4}")));
            if (p.n_d - before).abs() > 1e-6 {
                p.medium = Medium::Custom;
            }
            ui.end_row();
            ui.label("dispersion");
            ui.add(egui::Slider::new(&mut p.dispersion, 0.0..=10.0).suffix(" ×").fixed_decimals(1))
                .on_hover_text("1 = real; 0 = no dispersion: every colour makes the same bow");
            ui.end_row();
            ui.label("");
            ui.checkbox(&mut p.fresnel, "Fresnel losses").on_hover_text("reflection and transmission at every surface, separately for s and p; off: every path equally bright");
            ui.end_row();
        });
        let ls = p.wavelengths();
        let (l0, l1) = (ls[0], *ls.last().unwrap());
        let mut info = if ls.len() > 1 {
            format!("n = {:.4} ({l0:.0} nm) … {:.4} ({l1:.0} nm)", p.n(l0), p.n(l1))
        } else {
            format!("n = {:.4} ({l0:.0} nm)", p.n(l0))
        };
        for (k, name) in [(1, "primary"), (2, "secondary")] {
            match (p.bow_deg(l1, k), p.bow_deg(l0, k)) {
                (Some(a), Some(b)) if ls.len() > 1 => info += &format!("\n{name} bow: {a:.1}° ({l1:.0} nm) … {b:.1}° ({l0:.0} nm)"),
                (Some(a), _) => info += &format!("\n{name} bow: {a:.1}°"),
                _ => info += &format!("\n{name} bow: none (n too large)"),
            }
        }
        if let (Some(d), Some(pol)) = (descartes(p.n(589.0), 1), polarisation(p.n(589.0), 1)) {
            info += &format!("\nprimary Descartes ray: θ = {:.1}°, {:.0} % polarised", d.theta / DEG, 100.0 * pol);
        }
        ui.add_space(4.0);
        ui.weak(info);
    }

    fn path_controls(&mut self, ui: &mut egui::Ui) {
        let p = &mut self.params;
        ui.strong("LIGHT PATHS");
        let n = p.n(589.0);
        ui.horizontal(|ui| {
            ui.checkbox(&mut p.reflection, "reflected off the surface");
            ui.weak(fmt_pct(100.0 * order_power(n, None)));
        });
        for k in 0..=K_MAX {
            ui.horizontal(|ui| {
                ui.checkbox(&mut p.orders[k], ORDER_LABEL[k]);
                let mut t = fmt_pct(100.0 * order_power(n, Some(k as u32)));
                if let Some(d) = p.bow_deg(589.0, k as u32) {
                    t += &if d > 90.0 { format!(", {:.0}° from the sun", 180.0 - d) } else { format!(", bow at {d:.1}°") };
                }
                ui.weak(t);
            });
        }
        ui.weak("share of the light that hits a drop (589 nm)");
    }

    fn view_controls(&mut self, ui: &mut egui::Ui) {
        let p = &mut self.params;
        ui.strong("VIEW");
        egui::Grid::new("r_view").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
            ui.label("picked angle δ");
            ui.add(egui::Slider::new(&mut p.pick_deg, 0.0..=180.0).suffix("°").fixed_decimals(1))
                .on_hover_text("angle from the antisolar point: the yellow line and circle");
            ui.end_row();
            ui.label("ray height b");
            ui.horizontal(|ui| {
                if ui.add(egui::Slider::new(&mut p.b, 0.0..=0.999).custom_formatter(|v, _| format!("{v:.3}"))).changed() {
                    p.drop_rays = DropRays::Picked;
                }
                if ui.small_button("Descartes").on_hover_text("the ray of extreme deviation, which makes the bow").clicked() {
                    p.snap_to_descartes();
                    p.drop_rays = DropRays::Picked;
                }
            });
            ui.end_row();
            ui.label("rays in the drop");
            ui.add(egui::Slider::new(&mut p.rays, 0..=80));
            ui.end_row();
            ui.label("brightness");
            ui.add(egui::Slider::new(&mut p.brightness, 0.1..=30.0).logarithmic(true).fixed_decimals(1));
            ui.end_row();
        });
        ui.add_space(4.0);
        ui.weak(
            "Drag in the drop to move the ray (double-click: Descartes ray). To pick an angle, drag the yellow line in δ(θ) \
             or the drop in the side view, or click in the sky.",
        );
    }
}

/// the light that leaves the picked ray's path at each surface, with its share
fn self_lost_light(painter: &egui::Painter, c: Pos2, r: f32, b: f64, n: f64, k: u32, big: f32) {
    let (pts, _) = drop_path(c, r, b, n, k);
    let faint = |share: f64| Stroke::new(1.0, Color32::from_white_alpha((30.0 + 170.0 * share.sqrt()).min(200.0) as u8));
    let font = FontId::proportional(10.0);
    // reflected off the surface at the entry
    let (rs, rp) = reflect_power(b, n);
    let nrm = (pts[0] - c).normalized();
    let refl = vec2(1.0, 0.0) - 2.0 * vec2(1.0, 0.0).dot(nrm) * nrm;
    painter.line_segment([pts[0], pts[0] + refl * big], faint(rs + rp));
    painter.text(pts[0] + refl * r * 0.45, Align2::CENTER_BOTTOM, fmt_pct(100.0 * (rs + rp)), font.clone(), LABEL);
    // leaving at each internal reflection: that is the light of the lower orders
    for j in 1..=k as usize {
        let pa = path(b, n, j as u32 - 1);
        let (s, p) = path_power(b, n, j as u32 - 1);
        let d = vec2(pa.dev.cos() as f32, pa.dev.sin() as f32);
        painter.line_segment([pts[j], pts[j] + d * big], faint(s + p));
        painter.text(pts[j] + d * r * 0.4, Align2::CENTER_CENTER, fmt_pct(100.0 * (s + p)), font.clone(), LABEL);
    }
}

/// the bow of order k: its angle δ (rad) and whether no ray goes beyond it
/// (a maximum of δ, as for k = 1) or none below it (a minimum, as for k = 2)
fn extreme(n: f64, k: u32) -> Option<(f64, bool)> {
    let d = descartes(n, k)?;
    let b = d.theta.sin();
    let side = path((b - 0.02).max(0.0), n, k).delta;
    Some((d.delta, side < d.delta))
}

/// At the exit point: the directions of the Descartes rays (dashed) and the
/// wedge beyond them, where no ray of this order ever goes. The total
/// deviation D is smallest for the Descartes ray, for every k ≥ 1.
fn limit_marks(painter: &egui::Painter, c: Pos2, r: f32, b: f64, p: &RainbowParams, k: u32) {
    let shown = p.shown();
    let ends = if shown.len() > 1 { vec![shown[0], *shown.last().unwrap()] } else { shown };
    let limits: Vec<(f64, f64)> = ends.iter().filter_map(|&l| descartes(p.n(l), k).map(|d| (l, d.dev))).collect();
    let Some(&(_, d_min)) = limits.iter().min_by(|a, b| a.1.total_cmp(&b.1)) else { return };
    let (pts, _) = drop_path(c, r, b, p.n(589.0), k);
    let pe = *pts.last().unwrap();
    let len = r * 0.42;
    let width = 35f32.to_radians();
    let a0 = d_min as f32;
    let mut wedge = vec![pe];
    wedge.extend((0..=12).map(|i| pe + dir(a0 - width * i as f32 / 12.0) * len));
    painter.add(Shape::convex_polygon(wedge, WEDGE, Stroke::NONE));
    painter.text(pe + dir(a0 - 0.5 * width) * (len * 0.72), Align2::CENTER_CENTER, "no ray", FontId::proportional(10.0), Color32::from_rgb(255, 200, 190));
    for (l, d) in limits {
        let st = Stroke::new(1.2, line_color(l));
        painter.extend(Shape::dashed_line(&[pe, pe + dir(d as f32) * len * 1.25], st, 4.0, 3.0));
    }
}

/// θ and φ at the entry, δ at the exit, as in figure 8 of the exercise
fn angle_marks(painter: &egui::Painter, c: Pos2, r: f32, b: f64, n: f64, k: u32) {
    if b < 0.12 {
        return;
    }
    let (pts, out) = drop_path(c, r, b, n, k);
    let p0 = pts[0];
    let u = (p0 - c).normalized();
    let st = Stroke::new(1.0, Color32::from_white_alpha(120));
    dashed(painter, c, p0 + u * r * 0.35, st);
    let font = FontId::proportional(12.0);
    let col = Color32::WHITE;
    let a_out = u.y.atan2(u.x);
    let m = arc(painter, p0, 18.0, a_out, PI, st);
    painter.text(p0 + dir(m) * 28.0, Align2::CENTER_CENTER, "θ", font.clone(), col);
    let chord = (pts[1] - p0).normalized();
    let a_in = (-u.y).atan2(-u.x);
    let m = arc(painter, p0, 22.0, a_in, chord.y.atan2(chord.x), st);
    painter.text(p0 + dir(m) * 32.0, Align2::CENTER_CENTER, "φ", font.clone(), col);
    let pe = *pts.last().unwrap();
    dashed(painter, pe, pe - vec2(r * 0.6, 0.0), st);
    let m = arc(painter, pe, 26.0, PI, out.y.atan2(out.x), st);
    painter.text(pe + dir(m) * 36.0, Align2::CENTER_CENTER, "δ", font, col);
}
