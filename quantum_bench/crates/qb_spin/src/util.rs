//! Small formatting and drawing helpers shared by the spin scenes.

use std::f64::consts::PI;

use qb_core::C64;
use qb_core::Look;
use qb_core::egui::{self, Align2, Color32, Pos2, Rect, Stroke, pos2, vec2};
use qb_core::widgets::{PlotTransform, fmt_num};

/// "-" → "−" (the typographic minus).
pub fn minus(s: String) -> String {
    s.replace('-', "−")
}

/// `fmt_num` with a typographic minus.
pub fn num(v: f64, decimals: usize) -> String {
    minus(fmt_num(v, decimals))
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a.abs().max(1) } else { gcd(b, a % b) }
}

/// An angle as a multiple of π: "π/4", "−3π/4", "π", or "0.37π" when it is not a simple fraction.
pub fn fmt_pi(x: f64) -> String {
    let k = x / PI;
    for den in [1i64, 2, 3, 4, 6, 8, 12] {
        let numer = (k * den as f64).round();
        if (k * den as f64 - numer).abs() < 2e-3 {
            let numer = numer as i64;
            let g = gcd(numer, den);
            let (n, d) = (numer / g, den / g);
            let sign = if n < 0 { "−" } else { "" };
            let n = n.abs();
            return match (n, d) {
                (0, _) => "0".into(),
                (1, 1) => format!("{sign}π"),
                (n, 1) => format!("{sign}{n}π"),
                (1, d) => format!("{sign}π/{d}"),
                (n, d) => format!("{sign}{n}π/{d}"),
            };
        }
    }
    format!("{}π", num(k, 2))
}

/// A complex amplitude in polar form, e.g. "0.707", "−0.600", "0.707 e^(−iπ/4)", "0.5 e^(i·0.37π)".
pub fn fmt_amp(c: C64) -> String {
    let r = c.norm();
    if r < 5e-4 {
        return "0".into();
    }
    let a = c.arg();
    if a.abs() < 1e-3 {
        return format!("{r:.3}");
    }
    if (a.abs() - PI).abs() < 1e-3 {
        return format!("−{r:.3}");
    }
    let mag = fmt_pi(a.abs());
    let sign = if a < 0.0 { "−" } else { "" };
    if mag.ends_with("π") && mag.contains('.') {
        format!("{r:.3} e^({sign}i·{mag})")
    } else {
        format!("{r:.3} e^({sign}i{mag})")
    }
}

/// An amplitude in Cartesian form "0.500 − 0.500i".
pub fn fmt_cartesian(c: C64) -> String {
    let re = num(c.re, 3);
    let im = c.im;
    if im.abs() < 5e-4 {
        return re;
    }
    let sign = if im < 0.0 { "−" } else { "+" };
    format!("{re} {sign} {:.3}i", im.abs())
}

/// Scientific notation with superscript exponent: "1.759·10⁸".
pub fn sci(v: f64, digits: usize) -> String {
    if v == 0.0 || !v.is_finite() {
        return format!("{v}");
    }
    let e = v.abs().log10().floor() as i32;
    let m = v / 10f64.powi(e);
    let sup: String = e
        .to_string()
        .chars()
        .map(|c| match c {
            '-' => '⁻',
            '0' => '⁰',
            '1' => '¹',
            '2' => '²',
            '3' => '³',
            '4' => '⁴',
            '5' => '⁵',
            '6' => '⁶',
            '7' => '⁷',
            '8' => '⁸',
            _ => '⁹',
        })
        .collect();
    format!("{}·10{sup}", num(m, digits - 1))
}

/// Degrees from radians, formatted "53.1°".
pub fn deg(x: f64) -> String {
    format!("{}°", num(x.to_degrees(), 1))
}

/// A measured point with a vertical error bar ±sd (data units) on a plot.
pub fn error_point(painter: &egui::Painter, tr: &PlotTransform, x: f64, y: f64, sd: f64, col: Color32) {
    let look = Look::from_ctx(painter.ctx());
    let p = tr.to_screen(x, y);
    if sd > 0.0 {
        let a = tr.to_screen(x, y - sd);
        let b = tr.to_screen(x, y + sd);
        let w = look.px(2.5);
        painter.line_segment([a, b], look.stroke(1.1, col));
        painter.line_segment([pos2(a.x - w, a.y), pos2(a.x + w, a.y)], look.stroke(1.1, col));
        painter.line_segment([pos2(b.x - w, b.y), pos2(b.x + w, b.y)], look.stroke(1.1, col));
    }
    painter.circle_filled(p, look.px(2.8), col);
}

/// Text with a coloured key line in front, for legends drawn by hand.
pub fn legend_entry(painter: &egui::Painter, at: Pos2, text: &str, col: Color32, dashed: bool) -> Rect {
    let look = Look::from_ctx(painter.ctx());
    let l = look.px(18.0);
    let y = at.y + look.px(7.0);
    if dashed {
        painter.extend(egui::Shape::dashed_line(
            &[pos2(at.x, y), pos2(at.x + l, y)],
            look.stroke(1.5, col),
            4.0,
            3.0,
        ));
    } else {
        painter.line_segment([pos2(at.x, y), pos2(at.x + l, y)], look.stroke(2.0, col));
    }
    let r = painter.text(pos2(at.x + l + look.px(5.0), at.y), Align2::LEFT_TOP, text, look.font(11.5), col);
    Rect::from_min_max(at, r.max)
}

/// Key of a legend entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    Line,
    Dashed,
    /// a point with an error bar
    Point,
    /// text only
    None,
}

/// A legend in a translucent box whose top-right corner is `top_right`.
pub fn legend(painter: &egui::Painter, top_right: Pos2, entries: &[(String, Color32, Key)]) {
    let look = Look::from_ctx(painter.ctx());
    let key_w = look.px(18.0) + look.px(5.0);
    let row = look.px(16.0);
    let text_w = entries
        .iter()
        .map(|(t, c, _)| painter.layout_no_wrap(t.clone(), look.font(11.5), *c).size().x)
        .fold(0.0, f32::max);
    let w = key_w + text_w;
    let x0 = top_right.x - w;
    let bg = Rect::from_min_max(
        pos2(x0 - look.px(5.0), top_right.y - look.px(3.0)),
        pos2(top_right.x + look.px(4.0), top_right.y + row * entries.len() as f32 + look.px(1.0)),
    );
    painter.rect_filled(bg, 3.0, look.plot_bg().gamma_multiply(0.88));
    for (i, (t, c, k)) in entries.iter().enumerate() {
        let at = pos2(x0, top_right.y + i as f32 * row);
        match k {
            Key::Line => {
                legend_entry(painter, at, t, *c, false);
            }
            Key::Dashed => {
                legend_entry(painter, at, t, *c, true);
            }
            Key::Point => {
                let p = pos2(at.x + look.px(9.0), at.y + look.px(7.0));
                painter.line_segment([p - vec2(0.0, look.px(5.0)), p + vec2(0.0, look.px(5.0))], look.stroke(1.1, *c));
                painter.circle_filled(p, look.px(2.8), *c);
                painter.text(pos2(at.x + key_w, at.y), Align2::LEFT_TOP, t, look.font(11.5), *c);
            }
            Key::None => {
                painter.text(pos2(at.x + key_w, at.y), Align2::LEFT_TOP, t, look.font(11.5), *c);
            }
        }
    }
}

/// A filled panel (plot background with a frame) for custom drawings.
pub fn panel(painter: &egui::Painter, rect: Rect) {
    let look = Look::from_ctx(painter.ctx());
    painter.rect_filled(rect, 3.0, look.plot_bg());
    painter.rect_stroke(rect, 3.0, Stroke::new(1.0, look.plot_frame()), egui::StrokeKind::Inside);
}

/// An arrow from a to b (screen points) with a filled head.
pub fn arrow(painter: &egui::Painter, a: Pos2, b: Pos2, stroke: Stroke, head: f32) {
    let d = b - a;
    let len = d.length();
    if len < 1e-3 {
        return;
    }
    let dir = d / len;
    let head = head.min(0.6 * len);
    let base = b - dir * head;
    let perp = vec2(-dir.y, dir.x) * head * 0.5;
    painter.line_segment([a, base], stroke);
    painter.add(egui::Shape::convex_polygon(
        vec![b, base + perp, base - perp],
        stroke.color,
        Stroke::NONE,
    ));
}

/// Smooth step 0 → 1 on [0, 1].
pub fn smooth(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use qb_core::c64;

    #[test]
    fn pi_fractions() {
        assert_eq!(fmt_pi(PI / 4.0), "π/4");
        assert_eq!(fmt_pi(-3.0 * PI / 4.0), "−3π/4");
        assert_eq!(fmt_pi(PI), "π");
        assert_eq!(fmt_pi(0.0), "0");
        assert_eq!(fmt_pi(2.0 * PI / 3.0), "2π/3");
        assert_eq!(fmt_pi(0.37 * PI), "0.37π");
        let r = std::f64::consts::FRAC_1_SQRT_2;
        assert_eq!(fmt_amp(C64::from_polar(r, -PI / 4.0)), "0.707 e^(−iπ/4)");
        assert_eq!(fmt_amp(c64(-0.6, 0.0)), "−0.600");
        assert_eq!(fmt_amp(c64(0.0, 0.0)), "0");
        assert_eq!(fmt_cartesian(c64(0.5, -0.5)), "0.500 − 0.500i");
    }
}
