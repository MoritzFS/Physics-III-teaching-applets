//! The 2D panels shared by the hydrogen scenes: level diagram (n rows, l
//! columns, m sublevels with |c|² bars), radial panel (R_nl and r²|R|²),
//! probability bars, and the 2D slice field.

use qb_core::egui::{self, Align2, Color32, Rect, Sense, Shape, Stroke, StrokeKind, pos2, vec2};
use qb_core::widgets::{Domain, MeasureFlash, PanZoom, PlotFrame, PlotTransform, label_box, nice_ticks, phase_color, plot_dashed, plot_line, tick_label};
use qb_core::{C64, Look, c64, par};

use crate::orbital::{BASIS, Coeffs, HC_EV_NM, N_MAX, Nlm, active, energy_ev, l_letter, polys, radial, radial_density, radial_nodes};
use crate::view3d::Plane;

// ------------------------------------------------------------------ level diagram

/// Options of the level diagram.
pub struct LevelDiagram<'a> {
    pub coeffs: &'a Coeffs,
    /// highlighted state (selected by the user)
    pub selected: Option<Nlm>,
    /// draw the Lyman-α transition 1s ↔ 2p
    pub lyman: bool,
    /// levels react to clicks
    pub clickable: bool,
}

/// Draws the level diagram into `rect`; returns the clicked state.
pub fn level_diagram(ui: &mut egui::Ui, rect: Rect, d: &LevelDiagram, flash: &MeasureFlash) -> Option<Nlm> {
    let look = Look::of(ui);
    let response = ui.interact(rect, ui.id().with("hy_levels"), if d.clickable { Sense::click() } else { Sense::hover() });
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, look.plot_bg());
    let (ml, mr, mt, mb) = (look.px(44.0), look.px(70.0), look.px(30.0), look.px(10.0));
    let inner = Rect::from_min_max(pos2(rect.left() + ml, rect.top() + mt), pos2(rect.right() - mr, rect.bottom() - mb));
    let (lo, hi) = (-14.6, 0.9);
    let tr = PlotTransform {
        rect: inner,
        x: [0.0, 1.0],
        y: [lo, hi],
    };
    // energy axis
    let (ticks, step) = nice_ticks(lo, hi, (inner.height() / look.px(40.0)).max(2.0) as usize);
    for t in ticks {
        let sy = tr.y_to_screen(t);
        painter.line_segment([pos2(inner.left() - 3.0, sy), pos2(inner.left(), sy)], Stroke::new(1.0, look.axis()));
        painter.text(pos2(inner.left() - 5.0, sy), Align2::RIGHT_CENTER, tick_label(t, step), look.font(10.5), look.weak_text());
    }
    painter.line_segment([inner.left_top(), inner.left_bottom()], Stroke::new(1.0, look.axis()));
    painter.text(rect.left_top() + vec2(4.0, 3.0), Align2::LEFT_TOP, "E (eV)", look.font(11.5), look.text());
    // ionisation threshold
    let s0 = tr.y_to_screen(0.0);
    plot_dashed(&painter, &tr, &[0.0, 1.0], &[0.0, 0.0], look.stroke(1.0, look.weak_text()), 5.0);
    // columns
    let wc = inner.width() / 4.0;
    // "p  (l = 1)" where there is room, else "p" over "l = 1"
    let wide = painter.layout_no_wrap("p  (l = 1)".into(), look.font(11.0), look.text()).size().x + look.px(8.0) < wc;
    for l in 0..4u32 {
        let cx = inner.left() + (l as f32 + 0.5) * wc;
        let text = if wide { format!("{}  (l = {l})", l_letter(l)) } else { format!("{}\nl = {l}", l_letter(l)) };
        painter.text(
            pos2(cx, rect.top() + look.px(if wide { 4.0 } else { 1.0 })),
            Align2::CENTER_TOP,
            text,
            look.font(if wide { 11.0 } else { 10.0 }),
            look.text(),
        );
    }
    // geometry of the sublevel ticks
    let tick_x = |q: Nlm| -> (f32, f32) {
        let cx = inner.left() + (q.l as f32 + 0.5) * wc;
        let k = (2 * q.l + 1) as f32;
        let slot = (wc * 0.86 / k).min(look.px(26.0));
        let x = cx + (q.m as f32) * slot;
        (x, slot * 0.78)
    };
    let px_per_ev = inner.height() / (hi - lo) as f32;
    let max_bar = |n: u32| -> f32 {
        let gap = if n < N_MAX { energy_ev(n + 1) - energy_ev(n) } else { -energy_ev(n) } as f32 * px_per_ev;
        (gap * 0.8).min(look.px(42.0))
    };
    let hover = response.hover_pos();
    let mut hovered: Option<Nlm> = None;
    for q in BASIS {
        let sy = tr.y_to_screen(energy_ev(q.n));
        let (x, w) = tick_x(q);
        let hit = Rect::from_center_size(pos2(x, sy), vec2(w + 2.0, look.px(9.0)));
        if hover.is_some_and(|h| hit.contains(h)) {
            hovered = Some(q);
        }
    }
    for q in BASIS {
        let sy = tr.y_to_screen(energy_ev(q.n));
        let (x, w) = tick_x(q);
        let c = d.coeffs[q.index()];
        let p = c.norm_sqr();
        if p > 1e-6 {
            let h = (p as f32 * max_bar(q.n)).max(1.5);
            let bar = Rect::from_min_max(pos2(x - 0.5 * w, sy - h - 1.5), pos2(x + 0.5 * w, sy - 1.5));
            painter.rect_filled(bar, 1.0, phase_color(c.arg() as f32, 0.95));
        }
        let sel = d.selected == Some(q);
        let hov = hovered == Some(q);
        let col = if sel {
            look.accent()
        } else if hov {
            look.text()
        } else {
            look.level()
        };
        painter.line_segment(
            [pos2(x - 0.5 * w, sy), pos2(x + 0.5 * w, sy)],
            look.stroke(if sel || hov { 3.0 } else { 2.0 }, col),
        );
    }
    // level labels on the right
    // (pushed apart where the levels crowd near the threshold)
    let gap = look.px(11.0);
    let mut y_prev = f32::INFINITY;
    let mut labels: Vec<(f32, String)> = (1..=N_MAX)
        .map(|n| (tr.y_to_screen(energy_ev(n)), format!("n = {n}  {:.2}", energy_ev(n))))
        .collect();
    labels.push((s0, "E = 0".into()));
    for (y, _) in labels.iter_mut() {
        *y = y.min(y_prev - gap);
        y_prev = *y;
    }
    // if the top label was pushed out of the frame, shift the crowded ones down again
    let over = (rect.top() + look.px(4.0)) - labels.last().map_or(0.0, |l| l.0 - 0.5 * gap);
    if over > 0.0 {
        for (y, _) in labels.iter_mut().skip(1) {
            *y += over;
        }
    }
    for (y, text) in labels {
        painter.text(pos2(inner.right() + 4.0, y), Align2::LEFT_CENTER, text, look.font(10.0), look.weak_text());
    }
    if d.lyman {
        let a = tick_x(Nlm::new(1, 0, 0));
        let b = tick_x(Nlm::new(2, 1, 0));
        let pa = pos2(a.0 + look.px(12.0), tr.y_to_screen(energy_ev(1)) - 2.0);
        let pb = pos2(b.0 - look.px(4.0), tr.y_to_screen(energy_ev(2)) + 3.0);
        let col = look.series(3);
        painter.line_segment([pa, pb], look.stroke(1.8, col));
        let dir = (pb - pa).normalized();
        let nrm = vec2(-dir.y, dir.x);
        for (tip, sgn) in [(pb, 1.0), (pa, -1.0)] {
            let s = look.px(7.0);
            painter.add(Shape::convex_polygon(
                vec![tip, tip - dir * s * sgn + nrm * s * 0.45, tip - dir * s * sgn - nrm * s * 0.45],
                col,
                Stroke::NONE,
            ));
        }
        let de = energy_ev(2) - energy_ev(1);
        let mid = pa + (pb - pa) * 0.55;
        label_box(
            &painter,
            mid + vec2(look.px(8.0), 0.0),
            Align2::LEFT_CENTER,
            &format!("Lyman α: {de:.1} eV, λ = {:.1} nm", HC_EV_NM / de),
            col,
        );
    }
    painter.rect_stroke(rect, 2.0, Stroke::new(1.0, look.plot_frame()), StrokeKind::Inside);
    if matches!(flash.kind(), Some(qb_core::widgets::FlashKind::Level { .. })) {
        flash.paint(ui, &tr, Domain::Position);
    }
    if let Some(q) = hovered {
        let c = d.coeffs[q.index()];
        let mut text = format!(
            "{}  ({}, m = {})\nE{} = {:.3} eV  (degenerate: {} states with n = {})",
            q.name(),
            q.shell(),
            crate::orbital::signed(q.m),
            crate::orbital::sub(q.n as i64),
            energy_ev(q.n),
            q.n * q.n,
            q.n
        );
        if c.norm_sqr() > 1e-9 {
            text += &format!("\n|c|² = {:.3}, arg c = {:.2}π", c.norm_sqr(), c.arg() / std::f64::consts::PI);
        }
        if d.clickable {
            text += "\nclick: prepare this state";
        }
        response.clone().on_hover_text(text);
    }
    if response.clicked() { hovered } else { None }
}

// ------------------------------------------------------------------ radial panel

/// Radial functions R_nl of the components and the radial probability density
/// P(r) = r² Σ_lm |Σ_n c R_nl|² of the state, with ⟨r⟩ and the most probable r.
pub fn radial_panel(ui: &mut egui::Ui, id: &str, c: &Coeffs, r_max: f64, measured: Option<f64>, size: egui::Vec2) {
    let look = Look::of(ui);
    let mut comps: Vec<(u32, u32)> = vec![];
    for k in active(c) {
        let q = BASIS[k];
        if !comps.contains(&(q.n, q.l)) {
            comps.push((q.n, q.l));
        }
    }
    let n_pts = 600;
    let rs: Vec<f64> = (0..=n_pts).map(|i| r_max * i as f64 / n_pts as f64).collect();
    let dens: Vec<f64> = rs.iter().map(|&r| radial_density(c, r)).collect();
    let curves: Vec<Vec<f64>> = comps.iter().map(|&(n, l)| rs.iter().map(|&r| radial(n, l, r)).collect()).collect();
    let r_top = curves.iter().flatten().cloned().fold(0.0, f64::max).max(1e-9) * 1.3;
    let r_min = curves.iter().flatten().cloned().fold(0.0, f64::min);
    let r_bot = (1.18 * r_min).min(-0.12 * r_top);
    let p_top = dens.iter().cloned().fold(0.0, f64::max).max(1e-9) * 1.3;
    let p_bot = p_top * r_bot / r_top;
    // ⟨r⟩ and the most probable r
    let mean_r = crate::orbital::expect_r(c, 1);
    let (imax, _) = dens.iter().enumerate().fold((0, 0.0), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc });
    let r_mp = if imax > 0 && imax < n_pts {
        let (a, b, c) = (dens[imax - 1], dens[imax], dens[imax + 1]);
        let den = a - 2.0 * b + c;
        let h = rs[1] - rs[0];
        if den.abs() > 0.0 { rs[imax] + 0.5 * h * (a - c) / den } else { rs[imax] }
    } else {
        rs[imax]
    };
    let nodes: Vec<f64> = if comps.len() == 1 { radial_nodes(comps[0].0, comps[0].1) } else { vec![] };
    PlotFrame::new(("hy_radial", id))
        .x_range(0.0, r_max)
        .y_range(p_bot, p_top)
        .labels("r / a₀", "r²|R|²")
        .right_axis("R(r)", [r_bot, r_top])
        .width(size.x)
        .height(size.y)
        .pan_zoom(PanZoom::X)
        .show(ui, |painter, tr| {
            // P(r) filled
            let mut mesh = egui::Mesh::default();
            let fill = look.abs().gamma_multiply(0.16);
            for (i, (&r, &p)) in rs.iter().zip(&dens).enumerate() {
                mesh.colored_vertex(tr.to_screen(r, p), fill);
                mesh.colored_vertex(tr.to_screen(r, 0.0), fill);
                if i > 0 {
                    let k = 2 * i as u32;
                    mesh.add_triangle(k - 2, k - 1, k);
                    mesh.add_triangle(k - 1, k, k + 1);
                }
            }
            painter.add(Shape::mesh(mesh));
            plot_line(painter, tr, &rs, &dens, look.stroke(1.8, look.abs()));
            // R_nl on the right axis
            let to_left = |v: f64| p_bot + (v - r_bot) / (r_top - r_bot) * (p_top - p_bot);
            for (i, ((n, l), ys)) in comps.iter().zip(&curves).enumerate() {
                let col = look.series(i);
                let mapped: Vec<f64> = ys.iter().map(|&v| to_left(v)).collect();
                plot_dashed(painter, tr, &rs, &mapped, look.stroke(1.5, col), 5.0);
                let label = format!("R{}{}", crate::orbital::sub(*n as i64), crate::orbital::sub(*l as i64));
                // on the outer flank of R, past its outermost extremum (clear of the markers)
                let ext = (1..n_pts).rev().find(|&j| (ys[j] - ys[j - 1]) * (ys[j + 1] - ys[j]) <= 0.0).unwrap_or(0);
                let k = (ext..=n_pts).find(|&j| ys[j].abs() < 0.45 * ys[ext].abs()).unwrap_or(n_pts / 2).min(n_pts);
                painter.text(
                    tr.to_screen(rs[k], mapped[k]) + vec2(4.0, -4.0),
                    Align2::LEFT_BOTTOM,
                    label,
                    look.font(11.5),
                    col,
                );
            }
            // markers: ⟨r⟩ (solid) and the most probable r (dashed), labels on opposite sides
            let r = tr.rect;
            let sx = tr.x_to_screen(mean_r);
            let sm = tr.x_to_screen(r_mp);
            painter.line_segment([pos2(sx, r.top()), pos2(sx, r.bottom())], look.stroke(1.5, look.accent()));
            painter.extend(Shape::dashed_line(
                &[pos2(sm, r.top()), pos2(sm, r.bottom())],
                look.stroke(1.2, look.text().gamma_multiply(0.7)),
                4.0,
                3.0,
            ));
            let mean_right = mean_r >= r_mp;
            let y_lab = r.top() + look.px(22.0);
            // each label beside its line (preferred side if it fits), ⟨r⟩ in the first row,
            // the most probable r in the second, so that they never overlap
            let place = |x: f32, prefer_right: bool, text: &str, row: f32| {
                let w = painter.layout_no_wrap(text.to_string(), look.font(12.0), look.text()).size().x + look.px(10.0);
                let fits_right = x + 4.0 + w < r.right();
                let fits_left = x - 4.0 - w > r.left();
                let right = if prefer_right { fits_right || !fits_left } else { !fits_left && fits_right };
                let y = y_lab + row * look.px(22.0);
                if right {
                    (pos2(x + 4.0, y), Align2::LEFT_TOP)
                } else {
                    (pos2(x - 4.0, y), Align2::RIGHT_TOP)
                }
            };
            let t1 = format!("⟨r⟩ = {mean_r:.2}");
            let t2 = format!("most probable r = {r_mp:.2}");
            let (p1, a1) = place(sx, mean_right, &t1, 0.0);
            label_box(painter, p1, a1, &t1, look.accent());
            let (p2, a2) = place(sm, !mean_right, &t2, 1.0);
            label_box(painter, p2, a2, &t2, look.text());
            if let Some(rm) = measured {
                let x = tr.x_to_screen(rm);
                painter.line_segment([pos2(x, r.top()), pos2(x, r.bottom())], look.stroke(2.0, look.measure()));
                label_box(
                    painter,
                    pos2(x + 4.0, r.bottom() - look.px(24.0)),
                    Align2::LEFT_BOTTOM,
                    &format!("measured |r| = {rm:.2}"),
                    look.measure(),
                );
            }
            for rn in &nodes {
                let p = tr.to_screen(*rn, 0.0);
                painter.circle_stroke(p, look.px(3.5), look.stroke(1.2, look.weak_text()));
            }
            if !nodes.is_empty() {
                let p = tr.to_screen(nodes[0], 0.0);
                painter.text(p + vec2(4.0, 4.0), Align2::LEFT_TOP, "radial node", look.font(10.0), look.weak_text());
            }
        });
}

// ------------------------------------------------------------------ probability bars

/// Bars of probabilities (e.g. |c_m|² over m) in a small panel of its own.
pub fn prob_bars(ui: &mut egui::Ui, id: &str, title: &str, labels: &[String], probs: &[f64], size: egui::Vec2, color: Color32) {
    let look = Look::of(ui);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, look.plot_bg());
    let inner = Rect::from_min_max(
        pos2(rect.left() + look.px(8.0), rect.top() + look.px(42.0)),
        pos2(rect.right() - look.px(8.0), rect.bottom() - look.px(22.0)),
    );
    painter.text(rect.left_top() + vec2(look.px(6.0), look.px(4.0)), Align2::LEFT_TOP, title, look.font(12.5), look.text());
    let n = probs.len().max(1);
    let w = inner.width() / n as f32;
    let y = |p: f64| inner.bottom() - (p.clamp(0.0, 1.0) as f32) * inner.height();
    painter.line_segment([inner.left_bottom(), inner.right_bottom()], Stroke::new(1.0, look.axis()));
    // the level p = 1
    painter.extend(Shape::dashed_line(
        &[pos2(inner.left(), y(1.0)), pos2(inner.right(), y(1.0))],
        Stroke::new(1.0, look.grid()),
        4.0,
        4.0,
    ));
    for (i, p) in probs.iter().enumerate() {
        let cx = inner.left() + (i as f32 + 0.5) * w;
        let bar = Rect::from_min_max(pos2(cx - 0.3 * w, y(*p)), pos2(cx + 0.3 * w, inner.bottom()));
        painter.rect_filled(bar, 1.0, color.gamma_multiply(0.85));
        if *p > 0.0005 {
            painter.text(pos2(cx, y(*p) - 2.0), Align2::CENTER_BOTTOM, format!("{p:.2}"), look.font(11.5), look.text());
        }
        if let Some(l) = labels.get(i) {
            painter.text(pos2(cx, inner.bottom() + 3.0), Align2::CENTER_TOP, l, look.font(11.5), look.text());
        }
    }
    painter.rect_stroke(rect, 2.0, Stroke::new(1.0, look.plot_frame()), StrokeKind::Inside);
    let _ = (id, resp);
}

// ------------------------------------------------------------------ 2D slice

/// ψ on the plane (row-major, row 0 = lowest vertical coordinate), `n`×`n` points over [−L, L]².
pub fn slice_field(c: &Coeffs, packet: Option<([f64; 3], f64)>, plane: Plane, offset: f64, l: f64, n: usize) -> Vec<C64> {
    let act = active(c);
    let ps = polys();
    let rows: Vec<Vec<C64>> = par::map_range(n, |j| {
        let b = -l + 2.0 * l * (j as f64 + 0.5) / n as f64;
        (0..n)
            .map(|i| {
                let a = -l + 2.0 * l * (i as f64 + 0.5) / n as f64;
                let p = plane.point(a, b, offset);
                if let Some((p0, s)) = packet {
                    let d2 = (p[0] - p0[0]).powi(2) + (p[1] - p0[1]).powi(2) + (p[2] - p0[2]).powi(2);
                    return c64((-d2 / (4.0 * s * s)).exp(), 0.0);
                }
                let g = crate::orbital::Geo::new(p);
                act.iter().map(|&k| c[k] * ps[k].eval_geo(&g)).sum()
            })
            .collect()
    });
    rows.into_iter().flatten().collect()
}

/// Brightness scale of a state: (ρ_bound, ρ_ref).
///
/// ρ_bound bounds |ψ|² at all times: the maximum over a grid of (Σ_k |c_k||ψ_k|)²
/// (exact for one term, reached when the phases line up). ρ_ref, the scale the
/// views use, is the geometric mean of ρ_bound and ρ₂₅, the density above which 25 %
/// of the probability lies: for states whose probability sits far from the density
/// maximum (ns orbitals: the maximum is at the nucleus, the probability in the outer
/// shell) the outer parts stay visible and the densest parts saturate.
pub fn density_scale(c: &Coeffs, l: f64) -> (f64, f64) {
    let act = active(c);
    let ps = polys();
    let n = 45;
    let cols = par::map_range(n, |iz| {
        let z = -l + 2.0 * l * iz as f64 / (n - 1) as f64;
        let mut m: f64 = 0.0;
        let mut dens = Vec::with_capacity(n * n);
        for iy in 0..n {
            let y = -l + 2.0 * l * iy as f64 / (n - 1) as f64;
            for ix in 0..n {
                let x = -l + 2.0 * l * ix as f64 / (n - 1) as f64;
                let g = crate::orbital::Geo::new([x, y, z]);
                let mut s = 0.0;
                let mut psi = c64(0.0, 0.0);
                for &k in &act {
                    let v = ps[k].eval_geo(&g);
                    s += c[k].norm() * v.norm();
                    psi += c[k] * v;
                }
                m = m.max(s * s);
                if x * x + y * y + z * z <= l * l {
                    dens.push(psi.norm_sqr());
                }
            }
        }
        (m, dens)
    });
    let mut dens: Vec<f64> = cols.iter().flat_map(|c| c.1.iter().copied()).collect();
    let vals: Vec<f64> = cols.iter().map(|c| c.0).collect();
    dens.sort_by(|a, b| b.total_cmp(a));
    let total: f64 = dens.iter().sum();
    let mut acc = 0.0;
    let mut rho25 = dens.first().copied().unwrap_or(1.0);
    for d in &dens {
        acc += d;
        rho25 = *d;
        if acc >= 0.25 * total {
            break;
        }
    }
    // the maximum of |ψ_nlm|² of each single component, sampled finely along its peak direction
    let mut single: f64 = 0.0;
    for &k in &act {
        let o = &ps[k];
        let mut best: f64 = 0.0;
        for ir in 0..400 {
            let r = l * ir as f64 / 400.0;
            for it in 0..60 {
                let u = -1.0 + 2.0 * it as f64 / 59.0;
                let s = (1.0 - u * u).max(0.0).sqrt();
                let v = (o.rnorm * o.anorm).abs()
                    * (-r * o.inv_n).exp()
                    * r.powi(o.q.l as i32)
                    * (((o.rad[3] * r + o.rad[2]) * r + o.rad[1]) * r + o.rad[0]).abs()
                    * s.powi(o.q.m.abs())
                    * (((o.ang[3] * u + o.ang[2]) * u + o.ang[1]) * u + o.ang[0]).abs();
                best = best.max(v);
            }
        }
        single = single.max(c[k].norm_sqr() * best * best);
    }
    let bound = vals.into_iter().fold(single, f64::max).max(1e-300);
    (bound, (bound * rho25.max(1e-300)).sqrt().min(bound))
}
