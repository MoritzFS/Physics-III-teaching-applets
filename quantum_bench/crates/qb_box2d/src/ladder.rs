//! An energy ladder that groups degenerate states: one row per energy level,
//! one short dash per state of that level, a degeneracy label, a bar for the
//! probability of the level and the phasors c(t) of the occupied states.
//!
//! (qb_core's `LevelLadder` draws one full-width line per state, so states of
//! equal energy would sit on top of each other; hence this local widget.)

use qb_core::egui::{self, Align2, Id, Rect, Sense, Stroke, StrokeKind, pos2, vec2};
use qb_core::widgets::{PlotTransform, nice_ticks, phase_color, tick_label};
use qb_core::{C64, Look};

use crate::modes::EPS;

/// One state of a level.
#[derive(Clone, Debug)]
pub struct LadderState {
    pub i: usize,
    pub j: usize,
    /// c(t) of this state (zero if not occupied)
    pub c: C64,
    /// inside the thumbnail range (n ≤ 5): drawn brighter
    pub thumb: bool,
}

/// One energy level (E/ε) with its states.
#[derive(Clone, Debug)]
pub struct LadderLevel {
    pub e: f64,
    pub states: Vec<LadderState>,
    /// probability of the level, Σ|c|² over its states
    pub prob: f64,
    /// degeneracy in the plane (all n, also beyond the states listed)
    pub g: usize,
}

/// What happened on the ladder.
pub struct LadderOut {
    pub response: egui::Response,
    /// state (i, j) clicked this frame
    pub clicked: Option<(usize, usize)>,
    /// E/ε ↔ screen (x of the transform is 0 … 1 across the widget)
    pub transform: PlotTransform,
}

/// The ladder. `compact` = many levels (a spectrum): no labels, bars relative to the largest.
pub struct GroupedLadder<'a> {
    id: Id,
    levels: &'a [LadderLevel],
    range: [f64; 2],
    size: egui::Vec2,
    selected: Option<(usize, usize)>,
    compact: bool,
    title: String,
}

impl<'a> GroupedLadder<'a> {
    pub fn new(id: impl std::hash::Hash + std::fmt::Debug, levels: &'a [LadderLevel], range: [f64; 2], size: egui::Vec2) -> Self {
        GroupedLadder {
            id: Id::new(id),
            levels,
            range,
            size,
            selected: None,
            compact: false,
            title: "E / ε".into(),
        }
    }

    pub fn selected(mut self, s: Option<(usize, usize)>) -> Self {
        self.selected = s;
        self
    }

    pub fn compact(mut self, yes: bool) -> Self {
        self.compact = yes;
        self
    }

    pub fn title(mut self, t: impl Into<String>) -> Self {
        self.title = t.into();
        self
    }

    pub fn show(self, ui: &mut egui::Ui) -> LadderOut {
        let look = Look::of(ui);
        let (outer, _) = ui.allocate_exact_size(self.size, Sense::hover());
        let response = ui.interact(outer, self.id, Sense::click());
        let rect = Rect::from_min_max(
            pos2(outer.left() + look.px(34.0), outer.top() + look.px(22.0)),
            pos2(outer.right() - 4.0, outer.bottom() - look.px(8.0)),
        );
        let [lo, hi] = self.range;
        let tr = PlotTransform {
            rect,
            x: [0.0, 1.0],
            y: [lo, hi],
        };
        let painter = ui.painter_at(outer);
        painter.rect_filled(rect.expand(2.0), 2.0, look.plot_bg());
        painter.rect_stroke(rect.expand(2.0), 2.0, Stroke::new(1.0, look.plot_frame()), StrokeKind::Inside);
        let (ticks, step) = nice_ticks(lo, hi, (rect.height() / look.px(42.0)).max(2.0) as usize);
        for t in ticks {
            let sy = tr.y_to_screen(t);
            painter.line_segment([pos2(rect.left() - 3.0, sy), pos2(rect.left(), sy)], Stroke::new(1.0, look.axis()));
            painter.text(
                pos2(rect.left() - 5.0, sy),
                Align2::RIGHT_CENTER,
                tick_label(t, step),
                look.font(11.0),
                look.weak_text(),
            );
        }
        painter.text(
            pos2(rect.left(), outer.top() + 2.0),
            Align2::LEFT_TOP,
            &self.title,
            look.font(12.5),
            look.text(),
        );

        let visible: Vec<usize> = (0..self.levels.len())
            .filter(|&k| self.levels[k].e >= lo && self.levels[k].e <= hi)
            .collect();
        let gmax = visible.iter().map(|&k| self.levels[k].states.len()).max().unwrap_or(1).clamp(1, 6);
        let dash_w = if self.compact { look.px(30.0) } else { look.px(15.0) };
        let gap = look.px(3.0);
        let x0 = rect.left() + look.px(6.0);
        let dash_col = if self.compact { dash_w } else { gmax as f32 * (dash_w + gap) };
        let font = look.font(11.0);
        let font_h = look.px(12.5);

        // hovered level and state
        let hover = response.hover_pos();
        let hovered_level = hover.and_then(|p| {
            visible
                .iter()
                .map(|&k| (k, (tr.y_to_screen(self.levels[k].e) - p.y).abs()))
                .filter(|(_, d)| *d < look.px(7.0))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(k, _)| k)
        });
        let state_x = |s: usize| x0 + s as f32 * (dash_w + gap);
        let hovered_state = hovered_level.map(|k| {
            let lv = &self.levels[k];
            let px = hover.map_or(0.0, |p| p.x);
            if self.compact {
                return (lv.states[0].i, lv.states[0].j);
            }
            let s = (0..lv.states.len())
                .min_by(|&a, &b| {
                    let da = (state_x(a) + 0.5 * dash_w - px).abs();
                    let db = (state_x(b) + 0.5 * dash_w - px).abs();
                    da.total_cmp(&db)
                })
                .unwrap_or(0);
            (lv.states[s].i, lv.states[s].j)
        });
        let clicked = if response.clicked() { hovered_state } else { None };

        // phasor size from the spacing to the neighbouring levels
        let rmax = look.px(9.0);
        let radius_of = |pos: usize| -> f32 {
            let sy = tr.y_to_screen(self.levels[visible[pos]].e);
            let mut gap = f32::INFINITY;
            if pos > 0 {
                gap = gap.min((tr.y_to_screen(self.levels[visible[pos - 1]].e) - sy).abs());
            }
            if pos + 1 < visible.len() {
                gap = gap.min((tr.y_to_screen(self.levels[visible[pos + 1]].e) - sy).abs());
            }
            (gap * 0.45).clamp(3.0, rmax)
        };
        let radius = rmax;
        let n_ph = if self.compact {
            0
        } else {
            visible
                .iter()
                .map(|&k| self.levels[k].states.iter().filter(|s| s.c.norm_sqr() > 1e-6).count())
                .max()
                .unwrap_or(0)
                .min(3)
        };
        // labels: one column right of the dashes; where levels are closer than a
        // line of text the labels are pushed apart and joined to their level by a thin line.
        // Right part: bar of P(level), then the phasors; the bar gives way to the labels.
        let ph_w = n_ph as f32 * (2.0 * radius + look.px(4.0));
        let lab_x = x0 + dash_col + look.px(16.0);
        let width_of = |t: &str| painter.layout_no_wrap(t.to_string(), font.clone(), look.text()).size().x;
        let mut full: Vec<(String, String)> = vec![(String::new(), String::new()); self.levels.len()];
        let mut max_w = 0.0f32;
        if !self.compact {
            for &k in &visible {
                let lv = &self.levels[k];
                let names: Vec<String> = lv.states.iter().map(|s| format!("({},{})", s.i, s.j)).collect();
                let g = if lv.g > 1 {
                    format!("  g = {}{}", lv.g, if lv.g > lv.states.len() { "*" } else { "" })
                } else {
                    String::new()
                };
                max_w = max_w.max(width_of(&(names.join(" ") + &g)));
                full[k] = (names.join(" "), g);
            }
        }
        let room = rect.right() - look.px(6.0) - ph_w - lab_x - max_w - look.px(8.0);
        let bar_w = room.clamp(look.px(24.0), (rect.width() * 0.22).max(look.px(24.0)));
        let bar_x0 = rect.right() - look.px(6.0) - ph_w - bar_w;
        let lab_w = (bar_x0 - look.px(8.0) - lab_x).max(look.px(30.0));
        let mut label_y: Vec<Option<f32>> = vec![None; self.levels.len()];
        let mut texts: Vec<String> = vec![String::new(); self.levels.len()];
        if !self.compact && !visible.is_empty() {
            for &k in &visible {
                let (names, g) = &full[k];
                // all states and g; else the states only (the dashes count them); else shortened
                let mut t = format!("{names}{g}");
                if width_of(&t) > lab_w {
                    t = names.clone();
                    if width_of(&t) > lab_w {
                        let first = names.split(' ').next().unwrap_or("");
                        t = format!("{first} …{g}");
                    }
                }
                texts[k] = t;
            }
            let mut ys: Vec<f32> = visible.iter().map(|&k| tr.y_to_screen(self.levels[k].e)).collect();
            // bottom up (visible is sorted by energy: screen y decreases)
            for p in 1..ys.len() {
                ys[p] = ys[p].min(ys[p - 1] - font_h);
            }
            // top down: stay inside the plot
            let top = rect.top() + 0.5 * font_h;
            let last = ys.len() - 1;
            ys[last] = ys[last].max(top);
            for p in (0..last).rev() {
                ys[p] = ys[p].max(ys[p + 1] + font_h);
            }
            let fits = ys[0] <= rect.bottom() + 0.5 * font_h;
            if fits {
                for (p, &k) in visible.iter().enumerate() {
                    label_y[k] = Some(ys[p]);
                }
            } else {
                // too many levels for all labels: those that do not collide, and the occupied ones
                let mut last_y = f32::INFINITY;
                for &k in &visible {
                    let sy = tr.y_to_screen(self.levels[k].e);
                    if last_y - sy >= font_h || self.levels[k].prob > 1e-9 {
                        label_y[k] = Some(sy);
                        last_y = sy;
                    }
                }
            }
        }

        let pmax = if self.compact {
            visible.iter().map(|&k| self.levels[k].prob).fold(0.0, f64::max).max(1e-12)
        } else {
            1.0
        };

        for (pos, &k) in visible.iter().enumerate() {
            let lv = &self.levels[k];
            let sy = tr.y_to_screen(lv.e);
            let hov = hovered_level == Some(k);
            let r_here = radius_of(pos);
            if self.compact {
                let occupied = lv.prob > 1e-9;
                let col = if hov {
                    look.text()
                } else if occupied {
                    look.level()
                } else {
                    look.level().gamma_multiply(0.5)
                };
                painter.line_segment([pos2(x0, sy), pos2(x0 + dash_w, sy)], Stroke::new(1.0, col));
            } else {
                for (s, st) in lv.states.iter().enumerate() {
                    let sx = state_x(s);
                    let sel = self.selected == Some((st.i, st.j));
                    let occupied = st.c.norm_sqr() > 1e-6;
                    let col = if sel {
                        look.accent()
                    } else if occupied || hovered_state == Some((st.i, st.j)) && hov {
                        look.text()
                    } else if st.thumb {
                        look.level()
                    } else {
                        look.level().gamma_multiply(0.55)
                    };
                    let w = if sel || occupied { 3.0 } else { 1.6 };
                    painter.line_segment([pos2(sx, sy), pos2(sx + dash_w, sy)], look.stroke(w, col));
                }
                if let Some(ly) = label_y[k] {
                    let col = if lv.g > 1 { look.text() } else { look.weak_text() };
                    if (ly - sy).abs() > 1.5 {
                        let x_end = x0 + lv.states.len() as f32 * (dash_w + gap);
                        painter.line_segment(
                            [pos2(x_end, sy), pos2(lab_x - look.px(2.0), ly)],
                            Stroke::new(1.0, look.level().gamma_multiply(0.6)),
                        );
                    }
                    painter.text(pos2(lab_x, ly), Align2::LEFT_CENTER, &texts[k], font.clone(), col);
                }
            }
            if lv.prob > 1e-9 {
                let w = (lv.prob / pmax) as f32 * bar_w;
                let h = if self.compact {
                    2.0f32.max(look.px(2.0))
                } else {
                    (r_here * 1.2).max(4.0)
                };
                let x = if self.compact { x0 + dash_w + look.px(6.0) } else { bar_x0 };
                let bw = if self.compact {
                    (lv.prob / pmax) as f32 * (rect.right() - x - look.px(6.0))
                } else {
                    w
                };
                painter.rect_filled(
                    Rect::from_min_size(pos2(x, sy - 0.5 * h), vec2(bw.max(1.0), h)),
                    1.0,
                    look.counts().gamma_multiply(0.85),
                );
            }
            if !self.compact {
                let mut px = bar_x0 + bar_w + look.px(6.0) + radius;
                for st in lv.states.iter().filter(|s| s.c.norm_sqr() > 1e-6).take(3) {
                    let center = pos2(px, sy);
                    painter.circle_stroke(center, r_here, Stroke::new(1.0, look.plot_frame()));
                    let len = r_here * (st.c.norm() / lv.prob.sqrt().max(1e-12)).min(1.0) as f32;
                    let a = st.c.arg() as f32;
                    let tip = center + vec2(a.cos(), -a.sin()) * len;
                    painter.line_segment([center, tip], look.stroke(1.8, phase_color(a, 1.0)));
                    painter.circle_filled(tip, look.px(1.6), phase_color(a, 1.0));
                    px += 2.0 * radius + look.px(4.0);
                }
            }
        }
        if !self.compact && visible.iter().any(|&k| self.levels[k].g > self.levels[k].states.len()) {
            painter.text(
                pos2(rect.right() - 4.0, rect.bottom() - 3.0),
                Align2::RIGHT_BOTTOM,
                "* more states of this energy outside the range shown",
                look.font(10.0),
                look.weak_text(),
            );
        }
        if self.compact {
            painter.text(
                pos2(rect.right() - 4.0, rect.top() + 3.0),
                Align2::RIGHT_TOP,
                "bars: P(E) / largest",
                look.font(10.5),
                look.weak_text(),
            );
        }
        if let Some(k) = hovered_level {
            let lv = &self.levels[k];
            let states = lv
                .states
                .iter()
                .map(|s| format!("({},{})", s.i, s.j))
                .collect::<Vec<_>>()
                .join(", ");
            let mut text = format!("E = {:.4} ε = {:.3} ħ²/(ma²)\n{states}", lv.e, lv.e * EPS);
            if lv.g > 1 {
                text += &format!("\ndegenerate: g = {} in this plane", lv.g);
            }
            text += &format!("\nP = {:.4}", lv.prob);
            if !self.compact {
                text += "\nclick a dash to select that state";
            }
            response.clone().on_hover_text(text);
        }
        LadderOut {
            response,
            clicked,
            transform: tr,
        }
    }
}
