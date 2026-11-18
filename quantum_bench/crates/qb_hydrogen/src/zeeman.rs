//! PS12 47: the normal Zeeman effect (orbital only, no spin) of an l = 1 and
//! an l = 2 manifold: level diagram with the nine Δm = 0, ±1 transitions and
//! the resulting three-line spectrum.
//!
//! Convention: Δm = m(upper, l = 2) − m(lower, l = 1); σ⁺: Δm = +1 (photon
//! energy hν₀ + μ_B B), π: Δm = 0 (hν₀), σ⁻: Δm = −1 (hν₀ − μ_B B), with the
//! electron's Zeeman energy E = E_n + μ_B B m (B along z).

use qb_core::egui::{self, Align2, Color32, Pos2, Rect, Sense, Shape, Stroke, StrokeKind, pos2, vec2};
use qb_core::widgets::{PanZoom, PlotFrame, label_box, plot_line};
use qb_core::Look;

use crate::orbital::{HC_EV_NM, MU_B_EV_PER_T, MU_B_GHZ_PER_T, energy_ev};

/// The two manifolds (hydrogen: 2p below, 3d above, i.e. Balmer α).
pub const N_LOWER: u32 = 2;
pub const N_UPPER: u32 = 3;
/// Largest field on the slider (T).
pub const B_MAX: f64 = 3.0;
/// Default field (T).
pub const B_DEFAULT: f64 = 2.0;

/// All dipole-allowed transitions (m_lower, m_upper) with |Δm| ≤ 1: nine.
pub fn transitions() -> Vec<(i32, i32)> {
    let mut v = vec![];
    for dm in [1i32, 0, -1] {
        for m1 in -1..=1 {
            let m2 = m1 + dm;
            if m2.abs() <= 2 {
                v.push((m1, m2));
            }
        }
    }
    v
}

/// "σ⁺", "π", "σ⁻"
pub fn pol_name(dm: i32) -> &'static str {
    match dm {
        1 => "σ⁺",
        0 => "π",
        _ => "σ⁻",
    }
}

/// Colour of a polarisation.
pub fn pol_color(look: &Look, dm: i32) -> Color32 {
    match dm {
        1 => look.series(0),
        0 => look.series(2),
        _ => look.series(1),
    }
}

/// λ₀ of the field-free line (E_n = −13.6 eV/n²).
pub fn lambda0_nm() -> f64 {
    HC_EV_NM / (energy_ev(N_UPPER) - energy_ev(N_LOWER))
}

/// The diagram into `rect`; returns a clicked transition.
pub fn zeeman_diagram(ui: &mut egui::Ui, rect: Rect, b: f64, selected: (i32, i32)) -> Option<(i32, i32)> {
    let look = Look::of(ui);
    let response = ui.interact(rect, ui.id().with("hy_zeeman"), Sense::click());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, look.plot_bg());
    let h = rect.height();
    let top = rect.top() + look.px(32.0);
    let bottom = rect.bottom() - look.px(38.0);
    let hh = bottom - top;
    let y_up = top + 0.24 * hh;
    let y_lo = top + 0.80 * hh;
    // exaggerated splitting: ±2 steps of the l = 2 manifold fill ±18 % of the height at B_MAX
    let step = (0.12 * hh) * (b / B_MAX) as f32;
    let x0 = rect.left() + look.px(10.0);
    let x_split0 = rect.left() + 0.30 * rect.width();
    let x_fan = x_split0 + look.px(26.0);
    let x_right = rect.right() - look.px(46.0);
    let lw = look.stroke(2.5, look.level());
    painter.text(
        pos2(0.5 * (x0 + x_split0), rect.top() + look.px(4.0)),
        Align2::CENTER_TOP,
        "B = 0",
        look.font(11.5),
        look.text(),
    );
    let head = format!("B = {b:.2} T along z (splitting exaggerated)");
    let fits = painter.layout_no_wrap(head.clone(), look.font(11.5), look.text()).size().x < (x_right - x_fan) + look.px(30.0);
    painter.text(
        pos2(0.5 * (x_fan + x_right), rect.top() + look.px(4.0)),
        Align2::CENTER_TOP,
        if fits { head } else { format!("B = {b:.2} T along z\n(splitting exaggerated)") },
        look.font(if fits { 11.5 } else { 10.5 }),
        look.text(),
    );
    for (y, l, n, name) in [(y_up, 2i32, N_UPPER, "3d"), (y_lo, 1, N_LOWER, "2p")] {
        painter.line_segment([pos2(x0, y), pos2(x_split0, y)], lw);
        painter.text(
            pos2(x0, y - 3.0),
            Align2::LEFT_BOTTOM,
            format!("{name}: n = {n}, l = {l}"),
            look.font(11.0),
            look.text(),
        );
        painter.text(
            pos2(x0, y + 3.0),
            Align2::LEFT_TOP,
            format!("E = {:.2} eV", energy_ev(n)),
            look.font(10.0),
            look.weak_text(),
        );
        for m in -l..=l {
            let ym = y - m as f32 * step;
            painter.line_segment([pos2(x_split0, y), pos2(x_fan, ym)], Stroke::new(1.0, look.level().gamma_multiply(0.6)));
            painter.line_segment([pos2(x_fan, ym), pos2(x_right, ym)], look.stroke(1.8, look.level()));
            // all labels when there is room, else only the outermost ones
            if step >= look.px(11.0) || m.abs() == l || (l == 1 && step >= look.px(6.0)) {
                painter.text(
                    pos2(x_right + 4.0, ym),
                    Align2::LEFT_CENTER,
                    format!("m = {}", crate::orbital::signed(m)),
                    look.font(10.0),
                    look.weak_text(),
                );
            }
        }
    }
    // the gap is not to scale either
    let gx = 0.5 * (x0 + x_split0);
    painter.text(
        pos2(gx, 0.5 * (y_up + y_lo)),
        Align2::CENTER_CENTER,
        format!("λ₀ = {:.0} nm\n(Hα, {:.2} eV)", lambda0_nm(), energy_ev(N_UPPER) - energy_ev(N_LOWER)),
        look.font(10.5),
        look.weak_text(),
    );
    // transitions, grouped σ⁺ | π | σ⁻ as in the solution's figure
    let group_w = (x_right - x_fan) / 3.0;
    let wide_groups = painter.layout_no_wrap("Δm = +1".into(), look.font(11.0), look.text()).size().x + look.px(10.0) < group_w;
    let mut segs: Vec<((i32, i32), Pos2, Pos2)> = vec![];
    for (gi, dm) in [1i32, 0, -1].into_iter().enumerate() {
        let gx0 = x_fan + gi as f32 * group_w;
        let mut k = 0;
        for m1 in -1..=1 {
            let m2 = m1 + dm;
            if m2.abs() > 2 {
                continue;
            }
            let x = gx0 + group_w * (0.25 + 0.25 * k as f32);
            k += 1;
            let a = pos2(x, y_lo - m1 as f32 * step);
            let bpt = pos2(x, y_up - m2 as f32 * step);
            segs.push(((m1, m2), a, bpt));
        }
        painter.text(
            pos2(gx0 + 0.5 * group_w, bottom + look.px(4.0)),
            Align2::CENTER_TOP,
            if wide_groups {
                format!("{}\nΔm = {}", pol_name(dm), crate::orbital::signed(dm))
            } else {
                format!("{}\n{}", pol_name(dm), crate::orbital::signed(dm))
            },
            look.font(11.0),
            pol_color(&look, dm),
        );
    }
    let hover = response.hover_pos();
    let near = |p: Pos2| -> Option<(i32, i32)> {
        segs.iter()
            .map(|(t, a, b)| (*t, dist_to_segment(p, *a, *b)))
            .filter(|(_, d)| *d < look.px(8.0))
            .min_by(|x, y| x.1.total_cmp(&y.1))
            .map(|(t, _)| t)
    };
    let hovered = hover.and_then(near);
    for (t, a, bpt) in &segs {
        let dm = t.1 - t.0;
        let col = pol_color(&look, dm);
        let sel = *t == selected;
        let w = if sel { 3.2 } else if hovered == Some(*t) { 2.4 } else { 1.4 };
        if sel {
            painter.line_segment([*a, *bpt], look.stroke(8.0, col.gamma_multiply(0.25)));
        }
        painter.line_segment([*a, *bpt], look.stroke(w, col));
        let s = look.px(if sel { 8.0 } else { 6.0 });
        painter.add(Shape::convex_polygon(
            vec![*bpt, *bpt + vec2(-0.45 * s, s), *bpt + vec2(0.45 * s, s)],
            col,
            Stroke::NONE,
        ));
    }
    if let Some((_, a, bpt)) = segs.iter().find(|(t, _, _)| *t == selected) {
        let mid = pos2(a.x, 0.5 * (a.y + bpt.y));
        let (off, anchor) = if a.x > rect.center().x + 0.2 * rect.width() {
            (vec2(-look.px(6.0), 0.0), Align2::RIGHT_CENTER)
        } else {
            (vec2(look.px(6.0), 0.0), Align2::LEFT_CENTER)
        };
        label_box(
            &painter,
            mid + off,
            anchor,
            &format!("{} → {}", crate::orbital::signed(selected.0), crate::orbital::signed(selected.1)),
            pol_color(&look, selected.1 - selected.0),
        );
    }
    let _ = h;
    painter.rect_stroke(rect, 2.0, Stroke::new(1.0, look.plot_frame()), StrokeKind::Inside);
    if let Some(t) = hovered {
        response.clone().on_hover_text(format!(
            "{}: m = {} (l = 1) ↔ m = {} (l = 2)\nphoton energy hν₀ {} μ_B B\nclick: show this superposition in 3D",
            pol_name(t.1 - t.0),
            crate::orbital::signed(t.0),
            crate::orbital::signed(t.1),
            match t.1 - t.0 {
                1 => "+",
                0 => "+ 0 ·",
                _ => "−",
            }
        ));
    }
    if response.clicked() { hovered } else { None }
}

fn dist_to_segment(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_sq().max(1e-6)).clamp(0.0, 1.0);
    (a + ab * t - p).length()
}

/// The three lines: (Δm, offset Δν in GHz, relative intensity) for observation ⊥ B or ∥ B.
///
/// Classical dipole radiation of the Δm components with equal total power
/// (isotropic population): seen ⊥ B, π : σ⁺ : σ⁻ = 2 : 1 : 1 (all linearly
/// polarised: π ∥ B, σ ⊥ B); seen ∥ B, π is absent and σ± are circular.
pub fn lines(b: f64, along_b: bool) -> Vec<(i32, f64, f64)> {
    [1, 0, -1]
        .into_iter()
        .map(|dm| {
            let nu = dm as f64 * MU_B_GHZ_PER_T * b;
            let i = match (along_b, dm) {
                (false, 0) => 1.0,
                (false, _) => 0.5,
                (true, 0) => 0.0,
                (true, _) => 1.0,
            };
            (dm, nu, i)
        })
        .collect()
}

/// The spectrum: Δν = ν − ν₀ in GHz.
pub fn spectrum(ui: &mut egui::Ui, size: egui::Vec2, b: f64, along_b: bool, selected_dm: i32) {
    let look = Look::of(ui);
    let span = 1.5 * MU_B_GHZ_PER_T * B_MAX;
    let ls = lines(b, along_b);
    // display line shape (Lorentzian, FWHM 4 GHz): only to draw the lines, not a physical width
    let gamma = 2.0;
    let xs: Vec<f64> = (0..=400).map(|i| -span + 2.0 * span * i as f64 / 400.0).collect();
    let prof: Vec<f64> = xs
        .iter()
        .map(|&x| ls.iter().map(|(_, nu, i)| i * gamma * gamma / ((x - nu).powi(2) + gamma * gamma)).sum())
        .collect();
    PlotFrame::new("hy_zeeman_spectrum")
        .x_range(-span, span)
        .y_range(0.0, 1.65)
        .labels("Δν = ν − ν₀ (GHz)", if along_b { "intensity, seen ∥ B" } else { "intensity, seen ⊥ B" })
        .width(size.x)
        .height(size.y)
        .y_ticks(false)
        .pan_zoom(PanZoom::X)
        .show(ui, |painter, tr| {
            plot_line(painter, tr, &xs, &prof, look.stroke(1.4, look.abs().gamma_multiply(0.6)));
            for (dm, nu, i) in &ls {
                let col = pol_color(&look, *dm);
                let sel = *dm == selected_dm;
                let base = tr.to_screen(*nu, 0.0);
                let tip = tr.to_screen(*nu, *i);
                if *i > 0.0 {
                    painter.line_segment([base, tip], look.stroke(if sel { 4.0 } else { 2.5 }, col));
                }
                let pol = match (along_b, dm) {
                    (true, 0) => "absent",
                    (false, 0) => "lin. ∥ B",
                    (true, _) => "circular",
                    (false, _) => "lin. ⊥ B",
                };
                // just above each line (π: above its peak; an absent π line: at mid height)
                let y_lab = if *i > 0.0 { i + 0.08 } else { 0.5 };
                let dx = if b < 0.3 { *dm as f32 * look.px(34.0) } else { 0.0 };
                painter.text(
                    tr.to_screen(*nu, y_lab) + vec2(dx, 0.0),
                    Align2::CENTER_BOTTOM,
                    format!("{}  {pol}", pol_name(*dm)),
                    look.font(10.5),
                    col,
                );
            }
        });
    let dnu = MU_B_GHZ_PER_T * b;
    let lam = lambda0_nm();
    let dlam_pm = lam * lam * dnu * 1e9 / 2.997_924_58e17 * 1e3;
    ui.label(
        egui::RichText::new(format!(
            "μ_B B = {:.1} µeV,  Δν = ±{dnu:.1} GHz,  Δλ = ∓{dlam_pm:.1} pm at λ₀ = {lam:.0} nm",
            MU_B_EV_PER_T * b * 1e6
        ))
        .color(look.weak_text())
        .size(11.5 * look.font_scale()),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nine_transitions_three_lines() {
        let t = transitions();
        assert_eq!(t.len(), 9);
        for dm in [-1, 0, 1] {
            assert_eq!(t.iter().filter(|(a, b)| b - a == dm).count(), 3);
        }
        let mut freqs: Vec<i64> = t.iter().map(|(a, b)| (((b - a) as f64 * MU_B_GHZ_PER_T * 1.0) * 1000.0).round() as i64).collect();
        freqs.sort();
        freqs.dedup();
        assert_eq!(freqs.len(), 3, "normal Zeeman effect: three lines");
        // μ_B B/h = 14.0 GHz at 1 T, Hα at 656 nm with E_n = −13.6 eV/n²
        assert!((MU_B_GHZ_PER_T - 13.996).abs() < 1e-3);
        assert!((lambda0_nm() - 656.4).abs() < 0.1, "{}", lambda0_nm());
    }
}
