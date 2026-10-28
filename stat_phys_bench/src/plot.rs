//! Small plots drawn with egui's painter: axes with ticks, curves, bars and
//! markers, plus the colours shared by all benches.

use eframe::egui::{self, Align2, Color32, FontId, Painter, Pos2, Rect, Shape, Stroke, pos2, vec2};

pub const BG: Color32 = Color32::from_gray(16);
pub const GRID: Color32 = Color32::from_gray(34);
pub const AXIS: Color32 = Color32::from_gray(110);
pub const LABEL: Color32 = Color32::from_gray(170);
/// the formula of the notes
pub const THEORY: Color32 = Color32::from_gray(235);
/// what the simulation shows
pub const SIM: Color32 = Color32::from_rgb(80, 150, 255);
/// the current temperature
pub const NOW: Color32 = Color32::from_rgb(255, 170, 60);
/// the classical result, for comparison
pub const CLASSICAL: Color32 = Color32::from_gray(125);
pub const GROUND: Color32 = Color32::from_rgb(90, 150, 255);
pub const EXCITED: Color32 = Color32::from_rgb(255, 140, 50);

pub struct Plot {
    pub area: Rect,
    /// draws inside the plot area only
    pub data: Painter,
    frame: Painter,
    x: (f64, f64),
    y: (f64, f64),
    logx: bool,
}

impl Plot {
    /// Draws the background, title, grid and tick labels. The title should say what is plotted upwards.
    pub fn new(ui: &egui::Ui, outer: Rect, title: &str, xlabel: &str, x: (f64, f64), y: (f64, f64), logx: bool) -> Plot {
        let frame = ui.painter_at(outer);
        frame.rect_filled(outer, 3.0, BG);
        let min = outer.min + vec2(46.0, 24.0);
        let area = Rect::from_min_max(min, (outer.max - vec2(12.0, 32.0)).max(min + vec2(10.0, 10.0)));
        let y = if y.1 > y.0 { y } else { (y.0, y.0 + 1.0) };
        let plot = Plot { area, data: frame.with_clip_rect(area.expand(1.5)), frame, x, y, logx };
        plot.frame.text(outer.min + vec2(8.0, 5.0), Align2::LEFT_TOP, title, FontId::proportional(13.0), LABEL);
        plot.axes(xlabel);
        plot
    }

    fn axes(&self, xlabel: &str) {
        let (a, f) = (self.area, &self.frame);
        let font = FontId::proportional(11.0);
        let n_y = (a.height() / 36.0).clamp(2.0, 8.0) as f64;
        let step = nice_step(self.y.1 - self.y.0, n_y);
        for t in ticks(self.y.0, self.y.1, step) {
            let y = self.sy(t);
            f.line_segment([pos2(a.left(), y), pos2(a.right(), y)], Stroke::new(1.0, GRID));
            f.text(pos2(a.left() - 5.0, y), Align2::RIGHT_CENTER, fmt_tick(t, step), font.clone(), LABEL);
        }
        if self.logx {
            let (d0, d1) = (self.x.0.log10().floor() as i32, self.x.1.log10().ceil() as i32);
            let few = d1 - d0 <= 2;
            for d in d0..=d1 {
                for m in 1..10 {
                    let v = m as f64 * 10f64.powi(d);
                    if v < self.x.0 * 0.999 || v > self.x.1 * 1.001 {
                        continue;
                    }
                    let x = self.sx(v);
                    let major = m == 1;
                    f.line_segment([pos2(x, a.top()), pos2(x, a.bottom())], Stroke::new(1.0, if major { GRID } else { Color32::from_gray(24) }));
                    if major || (few && (m == 2 || m == 5)) {
                        f.text(pos2(x, a.bottom() + 3.0), Align2::CENTER_TOP, fmt(v), font.clone(), LABEL);
                    }
                }
            }
        } else {
            let n_x = (a.width() / 60.0).clamp(2.0, 10.0) as f64;
            let step = nice_step(self.x.1 - self.x.0, n_x);
            for t in ticks(self.x.0, self.x.1, step) {
                let x = self.sx(t);
                f.line_segment([pos2(x, a.top()), pos2(x, a.bottom())], Stroke::new(1.0, GRID));
                f.text(pos2(x, a.bottom() + 3.0), Align2::CENTER_TOP, fmt_tick(t, step), font.clone(), LABEL);
            }
        }
        f.line_segment([a.left_bottom(), a.right_bottom()], Stroke::new(1.0, AXIS));
        f.line_segment([a.left_bottom(), a.left_top()], Stroke::new(1.0, AXIS));
        f.text(pos2(a.center().x, a.bottom() + 16.0), Align2::CENTER_TOP, xlabel, FontId::proportional(12.0), LABEL);
    }

    pub fn sx(&self, x: f64) -> f32 {
        let t = if self.logx { (x.max(1e-300).ln() - self.x.0.ln()) / (self.x.1.ln() - self.x.0.ln()) } else { (x - self.x.0) / (self.x.1 - self.x.0) };
        self.area.left() + (t.clamp(-10.0, 10.0) as f32) * self.area.width()
    }

    pub fn sy(&self, y: f64) -> f32 {
        let t = (y - self.y.0) / (self.y.1 - self.y.0);
        self.area.bottom() - (t.clamp(-10.0, 10.0) as f32) * self.area.height()
    }

    pub fn at(&self, x: f64, y: f64) -> Pos2 {
        pos2(self.sx(x), self.sy(y))
    }

    pub fn curve(&self, pts: &[(f64, f64)], stroke: Stroke) {
        let p: Vec<Pos2> = pts.iter().filter(|(x, y)| x.is_finite() && y.is_finite()).map(|&(x, y)| self.at(x, y)).collect();
        if p.len() >= 2 {
            self.data.add(Shape::line(p, stroke));
        }
    }

    pub fn dashed(&self, pts: &[(f64, f64)], stroke: Stroke) {
        let p: Vec<Pos2> = pts.iter().filter(|(x, y)| x.is_finite() && y.is_finite()).map(|&(x, y)| self.at(x, y)).collect();
        if p.len() >= 2 {
            self.data.extend(Shape::dashed_line(&p, stroke, 6.0, 4.0));
        }
    }

    /// a bar from 0 up to y between x0 and x1
    pub fn bar(&self, x0: f64, x1: f64, y: f64, fill: Color32) {
        if y > 0.0 {
            let r = Rect::from_min_max(pos2(self.sx(x0) + 0.5, self.sy(y)), pos2(self.sx(x1) - 0.5, self.sy(0.0)));
            self.data.rect_filled(r, 0.0, fill);
        }
    }

    /// the outline of a histogram whose bins start at lo and are `width` wide
    pub fn steps(&self, lo: f64, width: f64, values: &[f64], stroke: Stroke) {
        let mut p = vec![];
        for (k, &v) in values.iter().enumerate() {
            let x0 = lo + k as f64 * width;
            p.push(self.at(x0, v));
            p.push(self.at(x0 + width, v));
        }
        if p.len() >= 2 {
            self.data.add(Shape::line(p, stroke));
        }
    }

    pub fn vline(&self, x: f64, stroke: Stroke) {
        let x = self.sx(x);
        self.data.line_segment([pos2(x, self.area.top()), pos2(x, self.area.bottom())], stroke);
    }

    pub fn dashed_vline(&self, x: f64, stroke: Stroke) {
        let x = self.sx(x);
        self.data.extend(Shape::dashed_line(&[pos2(x, self.area.top()), pos2(x, self.area.bottom())], stroke, 4.0, 3.0));
    }

    pub fn hline(&self, y: f64, stroke: Stroke) {
        let y = self.sy(y);
        self.data.line_segment([pos2(self.area.left(), y), pos2(self.area.right(), y)], stroke);
    }

    /// a horizontal band between y0 and y1
    pub fn band(&self, y0: f64, y1: f64, fill: Color32) {
        let r = Rect::from_min_max(pos2(self.area.left(), self.sy(y1)), pos2(self.area.right(), self.sy(y0)));
        self.data.rect_filled(r, 0.0, fill);
    }

    pub fn dot(&self, x: f64, y: f64, r: f32, color: Color32) {
        if x.is_finite() && y.is_finite() {
            self.data.circle_filled(self.at(x, y), r, color);
            self.data.circle_stroke(self.at(x, y), r, Stroke::new(1.0, BG));
        }
    }

    /// text at a point of the plot (in plot coordinates), kept inside the plot area
    pub fn label(&self, x: f64, y: f64, anchor: Align2, text: &str, color: Color32) {
        self.data.text(self.at(x, y), anchor, text, FontId::proportional(11.0), color);
    }

    /// a legend in the top right corner: (text, colour, dashed)
    pub fn legend(&self, items: &[(&str, Color32, bool)]) {
        self.legend_at(items, false);
    }

    /// a legend in the top left corner
    pub fn legend_left(&self, items: &[(&str, Color32, bool)]) {
        self.legend_at(items, true);
    }

    fn legend_at(&self, items: &[(&str, Color32, bool)], left: bool) {
        let font = FontId::proportional(11.0);
        let mut y = self.area.top() + 8.0;
        let right = self.area.right() - 6.0;
        for &(text, color, dashed) in items {
            let galley = self.frame.layout_no_wrap(text.to_owned(), font.clone(), LABEL);
            let x_text = if left { self.area.left() + 36.0 } else { right - galley.size().x };
            let line = [pos2(x_text - 24.0, y), pos2(x_text - 6.0, y)];
            if dashed {
                self.frame.extend(Shape::dashed_line(&line, Stroke::new(2.0, color), 4.0, 3.0));
            } else {
                self.frame.line_segment(line, Stroke::new(2.5, color));
            }
            self.frame.galley(pos2(x_text, y - galley.size().y / 2.0), galley, LABEL);
            y += 15.0;
        }
    }
}

/// a round step for about n ticks over span: 1, 2 or 5 times a power of ten
pub fn nice_step(span: f64, n: f64) -> f64 {
    let raw = (span / n).max(1e-12);
    let p = 10f64.powf(raw.log10().floor());
    let m = raw / p;
    p * if m < 1.5 {
        1.0
    } else if m < 3.5 {
        2.0
    } else if m < 7.5 {
        5.0
    } else {
        10.0
    }
}

pub fn ticks(lo: f64, hi: f64, step: f64) -> Vec<f64> {
    let first = (lo / step - 1e-9).ceil() as i64;
    let last = (hi / step + 1e-9).floor() as i64;
    (first..=last.min(first + 100)).map(|k| k as f64 * step).collect()
}

fn fmt_tick(v: f64, step: f64) -> String {
    let decimals = (-step.log10().floor()).max(0.0) as usize;
    let s = format!("{:.*}", decimals, v);
    if s.trim_start_matches('-').chars().all(|c| c == '0' || c == '.') { "0".into() } else { s }
}

/// a number with three significant digits; very large or small ones as a × 10ⁿ
pub fn fmt(v: f64) -> String {
    if !v.is_finite() {
        return "∞".into();
    }
    let a = v.abs();
    if a == 0.0 {
        return "0".into();
    }
    if !(1e-3..1e5).contains(&a) {
        let e = a.log10().floor() as i32;
        let m = v / 10f64.powi(e);
        return format!("{} × 10{}", trim(format!("{m:.2}")), superscript(e));
    }
    let decimals = (2 - a.log10().floor() as i32).max(0) as usize;
    trim(format!("{v:.decimals$}"))
}

fn trim(s: String) -> String {
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
}

fn superscript(e: i32) -> String {
    e.to_string()
        .chars()
        .map(|c| match c {
            '-' => '⁻',
            d => ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'][d.to_digit(10).unwrap_or(0) as usize],
        })
        .collect()
}

/// a colour scale for energies and speeds: dark blue → cyan → yellow → red (t in [0, 1])
pub fn heat(t: f32) -> Color32 {
    const STOPS: [(f32, [f32; 3]); 5] =
        [(0.0, [40.0, 60.0, 170.0]), (0.3, [40.0, 170.0, 230.0]), (0.55, [120.0, 220.0, 120.0]), (0.8, [250.0, 210.0, 60.0]), (1.0, [240.0, 70.0, 50.0])];
    let t = t.clamp(0.0, 1.0);
    for w in STOPS.windows(2) {
        let ((t0, c0), (t1, c1)) = (w[0], w[1]);
        if t <= t1 {
            let s = (t - t0) / (t1 - t0);
            let c = |k: usize| (c0[k] + s * (c1[k] - c0[k])) as u8;
            return Color32::from_rgb(c(0), c(1), c(2));
        }
    }
    Color32::from_rgb(240, 70, 50)
}

/// Splits a rectangle into n columns with gaps.
pub fn columns(rect: Rect, n: usize, gap: f32) -> Vec<Rect> {
    let w = (rect.width() - gap * (n as f32 - 1.0)) / n as f32;
    (0..n).map(|k| Rect::from_min_size(pos2(rect.left() + k as f32 * (w + gap), rect.top()), vec2(w, rect.height()))).collect()
}

/// points of a function on a logarithmic grid
pub fn log_grid(lo: f64, hi: f64, n: usize, f: impl Fn(f64) -> f64) -> Vec<(f64, f64)> {
    (0..n)
        .map(|k| {
            let x = lo * (hi / lo).powf(k as f64 / (n - 1) as f64);
            (x, f(x))
        })
        .collect()
}

/// points of a function on a linear grid
pub fn lin_grid(lo: f64, hi: f64, n: usize, f: impl Fn(f64) -> f64) -> Vec<(f64, f64)> {
    (0..n)
        .map(|k| {
            let x = lo + (hi - lo) * k as f64 / (n - 1) as f64;
            (x, f(x))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers() {
        assert_eq!(fmt(0.44), "0.44");
        assert_eq!(fmt(1.3679), "1.37");
        assert_eq!(fmt(0.047426), "0.0474");
        assert_eq!(fmt(4.54e-5), "4.54 × 10⁻⁵");
        assert_eq!(fmt(150.0), "150");
        assert_eq!(fmt(-2.5), "-2.5");
        assert_eq!(fmt_tick(0.30000000000000004, 0.1), "0.3");
        assert_eq!(ticks(0.0, 1.0, 0.25), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }
}
