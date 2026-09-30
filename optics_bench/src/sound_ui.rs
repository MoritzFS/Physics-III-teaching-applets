//! The sound tab of the dispersion bench: thunder heard from different
//! distances, and whistlers.

use eframe::egui::{self, pos2, vec2, Align2, Color32, FontId, Pos2, Rect, Sense, Shape, Stroke, TextureHandle, TextureOptions};
use rustfft::FftPlanner;

use crate::audio::{self, Audio};
use crate::dispersion_ui::{fmt_hz, fmt_tick, readout, ticks};
use crate::sound::{
    absorption_db_per_km, group_slowness, phase_speed, render, whistler_delay, RefCache, Rendered, SoundExample, SoundParams, C0, FS,
};
use crate::worker::Worker;

const PLAYHEAD: Color32 = Color32::from_rgb(255, 220, 80);
const FREQ_COLS: [(f64, Color32); 4] = [
    (8000.0, Color32::from_rgb(130, 150, 255)),
    (4000.0, Color32::from_rgb(90, 225, 230)),
    (2000.0, Color32::from_rgb(175, 230, 90)),
    (1000.0, Color32::from_rgb(255, 150, 90)),
];

#[derive(Default)]
pub struct SoundUi {
    worker: Option<Worker<SoundParams, Rendered>>,
    current: Option<Rendered>,
    tex: Option<TextureHandle>,
    /// id of the clip last played
    clip: u64,
    /// play as soon as the sound for the current parameters is ready
    autoplay: bool,
    /// view of the scene (x range, top) kept while the listener is dragged
    frozen: Option<(f64, f64, f64)>,
}

impl SoundUi {
    pub fn update(&mut self, ctx: &egui::Context, p: &SoundParams, audio: &mut Audio) {
        let worker = self.worker.get_or_insert_with(|| {
            let c = ctx.clone();
            let mut cache = RefCache::default();
            let mut planner = FftPlanner::new();
            Worker::new("sound", move |q: SoundParams| render(&q, &mut cache, &mut planner), move || c.request_repaint())
        });
        worker.request(p);
        let busy = worker.busy();
        if let Some(r) = worker.poll() {
            match &mut self.tex {
                Some(t) => t.set(r.spectrogram.clone(), TextureOptions::LINEAR),
                None => self.tex = Some(ctx.load_texture("sound_spectrogram", r.spectrogram.clone(), TextureOptions::LINEAR)),
            }
            let fresh = r.params == *p;
            let was_playing = self.position(audio).is_some();
            self.current = Some(r);
            if fresh && self.autoplay {
                self.autoplay = false;
                self.play(audio, 0.0);
            } else if was_playing {
                audio.stop();
            }
        }
        if busy || audio.playing().is_some() {
            ctx.request_repaint();
        }
    }

    fn ready(&self, p: &SoundParams) -> bool {
        self.current.as_ref().is_some_and(|r| r.params == *p)
    }

    fn play(&mut self, audio: &mut Audio, from: f32) {
        if let Some(r) = &self.current {
            self.clip = audio::clip_id();
            audio.play(r.samples.clone(), FS, self.clip, from);
        }
    }

    /// play (as soon as it is computed), or stop
    pub fn toggle(&mut self, p: &SoundParams, audio: &mut Audio) {
        if self.position(audio).is_some() {
            audio.stop();
        } else if self.ready(p) {
            self.play(audio, 0.0);
        } else {
            self.autoplay = true;
        }
    }

    fn play_when_ready(&mut self, p: &SoundParams, audio: &mut Audio) {
        audio.stop();
        if self.ready(p) {
            self.play(audio, 0.0);
        } else {
            self.autoplay = true;
        }
    }

    /// seconds into the current clip, while it plays
    fn position(&self, audio: &Audio) -> Option<f32> {
        audio.playing().filter(|(id, _)| *id == self.clip).map(|(_, pos)| pos)
    }

    /// time after the lightning that is playing now
    fn now(&self, audio: &Audio) -> Option<f64> {
        let r = self.current.as_ref()?;
        self.position(audio).map(|pos| (r.t_start + pos) as f64)
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, p: &mut SoundParams, audio: &mut Audio, notes: &str) {
        let win_h = ui.ctx().content_rect().height();
        egui::Panel::top("s_scene")
            .resizable(true)
            .default_size(win_h * 0.33)
            .min_size(150.0)
            .max_size(win_h * 0.6)
            .show(ui, |ui| match p.example {
                SoundExample::Thunder => self.thunder_scene(ui, p, audio),
                SoundExample::Whistler => self.whistler_scene(ui, p, audio),
            });
        egui::Panel::bottom("s_controls")
            .resizable(true)
            .default_size(win_h * 0.34)
            .min_size(180.0)
            .max_size(win_h * 0.6)
            .show(ui, |ui| self.controls_ui(ui, p, audio, notes));
        egui::CentralPanel::default().show(ui, |ui| self.hear_ui(ui, p, audio));
    }

    // ------------------------------------------------------------ top: the scene

    fn thunder_scene(&mut self, ui: &mut egui::Ui, p: &mut SoundParams, audio: &mut Audio) {
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            ui.strong("THE LIGHTNING AND YOU");
            ui.weak("side view · drag yourself to another place");
        });
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        // right after switching from the whistler, its sound (without a channel) is still the current one
        let Some(r) = self.current.as_ref().filter(|r| r.params.example == SoundExample::Thunder && !r.channel.is_empty()) else {
            ui.painter().text(rect.center(), Align2::CENTER_CENTER, "computing…", FontId::proportional(14.0), Color32::GRAY);
            return;
        };
        let painter = ui.painter_at(rect);
        let d = p.distance_km as f64;
        let (x_lo, x_hi, z_hi) = self.frozen.unwrap_or_else(|| {
            let xs = r.channel.iter().flatten().map(|q| q[0] as f64);
            let (lo, hi) = xs.fold((0.0f64, d), |(a, b), x| (a.min(x), b.max(x)));
            let top = r.channel.iter().flatten().map(|q| q[2] as f64).fold(1.0, f64::max);
            (lo - 0.4, hi.max(d) + 0.6, top + 0.7)
        });
        let ground_px = 26.0;
        let scale = ((rect.width() as f64 - 20.0) / (x_hi - x_lo)).min((rect.height() as f64 - ground_px as f64 - 6.0) / z_hi);
        let off = (rect.width() as f64 - scale * (x_hi - x_lo)) / 2.0;
        let gy = rect.bottom() - ground_px;
        let sp = |x: f64, z: f64| pos2(rect.left() + (off + (x - x_lo) * scale) as f32, gy - (z * scale) as f32);
        // sky, ground, clouds
        painter.rect_filled(rect, 3.0, Color32::from_rgb(18, 22, 34));
        painter.rect_filled(Rect::from_min_max(pos2(rect.left(), gy), rect.max), 0.0, Color32::from_rgb(30, 40, 26));
        // a cloud deck above the lower end of the part of the channel that runs sideways
        let top = r.channel[0].iter().map(|q| q[2]).fold(0.0f32, f32::max) as f64;
        let base = (top - 1.0).max(0.5);
        let cloud = Color32::from_rgba_unmultiplied(110, 115, 135, 40);
        let deck = sp(0.0, base + 0.3).y;
        painter.rect_filled(Rect::from_min_max(rect.min, pos2(rect.right(), deck)), 0.0, cloud);
        let mut k = 0u32;
        let mut x = rect.left() - 10.0;
        while x < rect.right() + 30.0 {
            let r = 16.0 + 12.0 * ((k * 7919 % 13) as f32 / 13.0);
            painter.circle_filled(pos2(x, deck), r, cloud);
            x += 26.0;
            k += 1;
        }
        // the sound front: everything at the distance c·t from the channel
        if let Some(t) = self.now(audio) {
            let rad = C0 * t.max(0.0) / 1000.0 * scale;
            let stroke = Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 22));
            for (i, q) in r.channel.iter().flatten().enumerate() {
                if i % 60 == 0 {
                    painter.circle_stroke(sp(q[0] as f64, q[2] as f64), rad as f32, stroke);
                }
            }
        }
        for (i, l) in r.channel.iter().enumerate() {
            let pts: Vec<Pos2> = l.iter().map(|q| sp(q[0] as f64, q[2] as f64)).collect();
            let w = if i == 0 { 1.6 } else { 1.0 };
            painter.add(Shape::line(pts.clone(), Stroke::new(w * 4.0, Color32::from_rgba_unmultiplied(170, 140, 255, 40))));
            painter.add(Shape::line(pts, Stroke::new(w, Color32::from_rgb(235, 230, 255))));
        }
        // nearest and farthest part of the channel
        let ear = [d as f32, 0.0, 0.0017];
        let dist = |q: &[f32; 3]| ((q[0] - ear[0]).powi(2) + (q[1] - ear[1]).powi(2) + (q[2] - ear[2]).powi(2)).sqrt();
        let all: Vec<&[f32; 3]> = r.channel.iter().flatten().collect();
        let you = sp(d, 0.0);
        if let (Some(near), Some(far)) = (
            all.iter().min_by(|a, b| dist(a).total_cmp(&dist(b))),
            all.iter().max_by(|a, b| dist(a).total_cmp(&dist(b))),
        ) {
            for (q, name, col) in [(near, "nearest", Color32::from_rgb(120, 230, 160)), (far, "farthest", Color32::from_rgb(240, 150, 110))] {
                let a = sp(q[0] as f64, q[2] as f64);
                painter.extend(Shape::dashed_line(&[a, you + vec2(0.0, -14.0)], Stroke::new(1.0, col), 5.0, 4.0));
                let rr = dist(q) as f64;
                painter.text(
                    a + vec2(6.0, if name == "nearest" { -2.0 } else { 12.0 }),
                    Align2::LEFT_BOTTOM,
                    format!("{name} part: {rr:.2} km, heard after {:.1} s", rr * 1000.0 / C0),
                    FontId::proportional(11.0),
                    col,
                );
            }
        }
        // you
        let col = PLAYHEAD;
        painter.circle_filled(you + vec2(0.0, -20.0), 4.0, col);
        painter.line_segment([you + vec2(0.0, -16.0), you + vec2(0.0, -7.0)], Stroke::new(2.0, col));
        painter.line_segment([you + vec2(0.0, -7.0), you + vec2(-4.0, 0.0)], Stroke::new(2.0, col));
        painter.line_segment([you + vec2(0.0, -7.0), you + vec2(4.0, 0.0)], Stroke::new(2.0, col));
        painter.line_segment([you + vec2(-5.0, -13.0), you + vec2(5.0, -13.0)], Stroke::new(2.0, col));
        painter.text(you + vec2(0.0, 4.0), Align2::CENTER_TOP, format!("you, {:.2} km", d), FontId::proportional(12.0), col);
        // scale bar
        let km = {
            // the largest 1, 2 or 5 × 10ⁿ below a quarter of the width
            let target = (x_hi - x_lo) * 0.25;
            let mag = 10f64.powf(target.log10().floor());
            [5.0, 2.0, 1.0].iter().map(|m| m * mag).find(|v| *v <= target).unwrap_or(mag)
        };
        let a = pos2(rect.left() + 12.0, rect.top() + 14.0);
        let b = a + vec2((km * scale) as f32, 0.0);
        painter.line_segment([a, b], Stroke::new(2.0, Color32::WHITE));
        painter.text(a + vec2(0.0, 3.0), Align2::LEFT_TOP, format!("{} km", fmt_tick(km)), FontId::proportional(11.0), Color32::WHITE);
        // drag the listener
        if resp.drag_started() {
            self.frozen = Some((x_lo, x_hi, z_hi));
        }
        if resp.dragged() || resp.clicked() {
            if let Some(q) = resp.interact_pointer_pos() {
                let x = x_lo + (q.x - rect.left()) as f64 / scale - off / scale;
                p.distance_km = x.clamp(0.1, 20.0) as f32;
            }
        }
        if resp.drag_stopped() || resp.clicked() {
            self.frozen = None;
            self.play_when_ready(p, audio);
        }
        if resp.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
        }
    }

    fn whistler_scene(&mut self, ui: &mut egui::Ui, p: &mut SoundParams, audio: &mut Audio) {
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            ui.strong("LIGHTNING, MAGNETOSPHERE AND YOU");
            ui.weak("the radio click travels along a magnetic field line (not to scale in time)");
        });
        let (rect, _) = ui.allocate_exact_size(ui.available_size(), Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3.0, Color32::from_rgb(12, 14, 26));
        // Earth radii; dipole field lines r = L cos²λ
        let scale = (rect.width() / 6.0).min(rect.height() / 3.8);
        let c = pos2(rect.left() + 1.6 * scale, rect.center().y);
        let sp = |x: f64, y: f64| c + vec2(x as f32 * scale, -y as f32 * scale);
        let field_line = |l: f64| -> Vec<Pos2> {
            let l0 = (1.0 / l).sqrt().acos();
            (0..=100).map(|i| -l0 + 2.0 * l0 * i as f64 / 100.0).map(|lat: f64| sp(l * lat.cos().powi(2) * lat.cos(), l * lat.cos().powi(2) * lat.sin())).collect()
        };
        for l in [2.0, 3.0, 5.0] {
            painter.add(Shape::line(field_line(l), Stroke::new(1.0, Color32::from_rgba_unmultiplied(120, 140, 220, 50))));
        }
        let main = field_line(4.0);
        painter.add(Shape::line(main.clone(), Stroke::new(1.5, Color32::from_rgb(140, 160, 240))));
        painter.circle_filled(c, scale, Color32::from_rgb(40, 80, 140));
        painter.text(c, Align2::CENTER_CENTER, "Earth", FontId::proportional(12.0), Color32::from_gray(210));
        painter.text(c + vec2(0.0, -scale - 4.0), Align2::CENTER_BOTTOM, "N", FontId::proportional(11.0), Color32::from_gray(160));
        painter.text(sp(4.05, 0.0), Align2::LEFT_CENTER, "plasma along the field line: n² ≈ ω_p²/(ω Ω)", FontId::proportional(11.0), Color32::from_rgb(150, 170, 240));
        let south = *main.first().unwrap();
        let north = *main.last().unwrap();
        // the lightning
        let bolt = [vec2(0.0, 0.0), vec2(-6.0, 9.0), vec2(1.0, 10.0), vec2(-5.0, 22.0)];
        let bolt: Vec<Pos2> = bolt.iter().map(|v| south + *v + vec2(4.0, -24.0)).collect();
        painter.add(Shape::line(bolt, Stroke::new(2.5, Color32::from_rgb(255, 240, 150))));
        painter.text(south + vec2(10.0, 4.0), Align2::LEFT_TOP, "lightning", FontId::proportional(12.0), Color32::from_rgb(255, 240, 150));
        let you = if p.same_hemisphere { south + vec2(-18.0, 6.0) } else { north };
        painter.circle_filled(you, 5.0, PLAYHEAD);
        painter.text(you + vec2(10.0, if p.same_hemisphere { 10.0 } else { -4.0 }), Align2::LEFT_BOTTOM, "you, with a VLF radio", FontId::proportional(12.0), PLAYHEAD);
        // groups of different frequencies travelling along the line
        if let Some(t) = self.now(audio) {
            let hops_max = if p.echoes { if p.same_hemisphere { 6 } else { 5 } } else if p.same_hemisphere { 2 } else { 1 };
            for (f, col) in FREQ_COLS {
                let per_hop = whistler_delay(f, p.whistler_d as f64, p.nose_khz as f64);
                if t < 0.0 || !per_hop.is_finite() {
                    continue;
                }
                let phi = t / per_hop;
                let hop = phi.floor() as usize;
                if hop >= hops_max {
                    continue;
                }
                let frac = phi - hop as f64;
                let f_along = if hop % 2 == 0 { frac } else { 1.0 - frac };
                let i = ((f_along * 100.0) as usize).min(100);
                painter.circle_filled(main[i], 5.0, col);
                painter.text(main[i] + vec2(7.0, 0.0), Align2::LEFT_CENTER, format!("{} Hz", fmt_hz(f)), FontId::proportional(10.0), col);
            }
        }
    }

    // ------------------------------------------------------------ middle: what you hear

    fn hear_ui(&mut self, ui: &mut egui::Ui, p: &SoundParams, audio: &mut Audio) {
        let playing = self.position(audio).is_some();
        ui.horizontal(|ui| {
            ui.strong("WHAT YOU HEAR");
            if ui.add(egui::Button::new(if playing { "■ stop" } else { "▶ play" }).min_size(vec2(70.0, 0.0))).on_hover_text("space bar").clicked() {
                self.toggle(p, audio);
            }
            if !self.ready(p) {
                ui.spinner();
                ui.weak("computing…");
            }
            if let Some(r) = &self.current {
                if let Some(db) = r.level_db {
                    ui.weak(format!("loudest moment: {db:+.0} dB compared with 100 m"));
                    if !p.true_loudness {
                        ui.weak("(played normalised)");
                    }
                }
            }
            ui.weak("· click in the picture to play from there");
            if let Some(r) = &self.current {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.weak(format!("computed in {:.0} ms", r.millis));
                });
            }
        });
        if let Some(e) = &audio.error {
            ui.colored_label(Color32::from_rgb(230, 130, 90), e);
        }
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click());
        let Some(r) = &self.current else { return };
        let painter = ui.painter_at(rect);
        let area = Rect::from_min_max(rect.min + vec2(46.0, 2.0), rect.max - vec2(8.0, 20.0));
        if area.height() < 60.0 {
            return;
        }
        let wave = Rect::from_min_size(area.min, vec2(area.width(), area.height() * 0.28));
        let spec = Rect::from_min_max(pos2(area.left(), wave.bottom() + 6.0), area.max);
        let (t0, t1) = (r.t_start as f64, (r.t_start + r.duration) as f64);
        let tx = |t: f64| area.left() + ((t - t0) / (t1 - t0)) as f32 * area.width();
        // waveform
        painter.rect_filled(wave, 3.0, Color32::from_gray(20));
        let cols = wave.width().max(1.0) as usize;
        let yc = wave.center().y;
        let a = wave.height() * 0.48;
        let mut pts = Vec::with_capacity(2 * cols);
        for c in 0..cols {
            let i0 = c * r.envelope.len() / cols;
            let i1 = ((c + 1) * r.envelope.len() / cols).max(i0 + 1).min(r.envelope.len());
            let (lo, hi) = r.envelope[i0..i1].iter().fold((f32::MAX, f32::MIN), |(l, h), e| (l.min(e[0]), h.max(e[1])));
            let x = wave.left() + c as f32 + 0.5;
            pts.push(pos2(x, yc - hi * a));
            pts.push(pos2(x, yc - lo * a));
        }
        painter.add(Shape::line(pts, Stroke::new(1.0, Color32::from_rgb(140, 210, 255))));
        painter.text(wave.left_top() + vec2(4.0, 2.0), Align2::LEFT_TOP, "pressure", FontId::proportional(10.0), Color32::from_gray(150));
        // spectrogram
        if let Some(tex) = &self.tex {
            painter.image(tex.id(), spec, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
        }
        let (f0, f1) = (r.spec_range.0 as f64, r.spec_range.1 as f64);
        let fy = |f: f64| {
            let s = if r.spec_log { (f.max(f0).ln() - f0.max(1.0).ln()) / (f1.ln() - f0.max(1.0).ln()) } else { (f - f0) / (f1 - f0) };
            spec.bottom() - s as f32 * spec.height()
        };
        let grey = Color32::from_gray(150);
        let freqs: Vec<f64> = if r.spec_log { vec![50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0] } else { ticks(f0, f1, 8.0) };
        for f in freqs {
            let y = fy(f);
            painter.line_segment([pos2(spec.left() - 4.0, y), pos2(spec.left(), y)], Stroke::new(1.0, grey));
            painter.text(pos2(spec.left() - 6.0, y), Align2::RIGHT_CENTER, fmt_hz(f), FontId::monospace(9.0), grey);
        }
        painter.text(pos2(spec.left() - 6.0, spec.top() - 2.0), Align2::RIGHT_BOTTOM, "f / Hz", FontId::proportional(10.0), grey);
        // predicted arrival times of each frequency
        let clip = ui.painter_at(spec);
        for (i, curve) in r.arrivals.iter().enumerate() {
            let pts: Vec<Pos2> = curve.iter().map(|q| pos2(tx(q[1] as f64), fy(q[0] as f64))).collect();
            let col = if p.example == SoundExample::Thunder && i == 1 {
                Color32::from_rgba_unmultiplied(240, 150, 110, 160)
            } else {
                Color32::from_rgba_unmultiplied(120, 230, 160, 160)
            };
            clip.extend(Shape::dashed_line(&pts, Stroke::new(1.2, col), 6.0, 5.0));
        }
        // time axis: seconds after the flash
        for v in ticks(t0, t1, (area.width() / 80.0) as f64) {
            let x = tx(v);
            painter.line_segment([pos2(x, area.bottom()), pos2(x, area.bottom() + 4.0)], Stroke::new(1.0, grey));
            if x < area.right() - 170.0 {
                painter.text(pos2(x, area.bottom() + 4.0), Align2::CENTER_TOP, fmt_tick(v), FontId::monospace(9.0), grey);
            }
        }
        painter.text(area.right_bottom() + vec2(0.0, 4.0), Align2::RIGHT_TOP, "time after the lightning / s", FontId::proportional(10.0), grey);
        if let Some(t) = self.now(audio) {
            let x = tx(t);
            painter.line_segment([pos2(x, area.top()), pos2(x, area.bottom())], Stroke::new(1.5, PLAYHEAD));
        }
        if let Some(q) = resp.hover_pos() {
            if spec.contains(q) {
                let t = t0 + ((q.x - area.left()) / area.width()) as f64 * (t1 - t0);
                let s = ((spec.bottom() - q.y) / spec.height()) as f64;
                let f = if r.spec_log { f0.max(1.0) * (f1 / f0.max(1.0)).powf(s) } else { f0 + s * (f1 - f0) };
                readout(&painter, spec.right_top() + vec2(-4.0, 4.0), Align2::RIGHT_TOP, format!("t = {t:.2} s\nf = {} Hz", fmt_hz(f.round())));
            }
        }
        if resp.clicked() {
            if let Some(q) = resp.interact_pointer_pos() {
                let from = ((q.x - area.left()) / area.width()).clamp(0.0, 1.0) * r.duration;
                self.play(audio, from);
            }
        }
    }

    // ------------------------------------------------------------ bottom: medium and controls

    fn controls_ui(&mut self, ui: &mut egui::Ui, p: &mut SoundParams, audio: &mut Audio, notes: &str) {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.strong("EXAMPLE");
            let before = p.example;
            ui.selectable_value(&mut p.example, SoundExample::Thunder, "thunder");
            ui.selectable_value(&mut p.example, SoundExample::Whistler, "whistler");
            if p.example != before {
                audio.stop();
            }
        });
        if !notes.trim().is_empty() {
            ui.horizontal_wrapped(|ui| {
                ui.strong("About this:");
                ui.label(notes);
            });
        }
        ui.separator();
        let avail = ui.available_rect_before_wrap();
        let plot_w = (avail.width() * 0.4).clamp(300.0, 760.0);
        let left = Rect::from_min_size(avail.min, vec2(plot_w, avail.height()));
        let right = Rect::from_min_max(pos2(avail.left() + plot_w + 14.0, avail.top()), avail.max);
        ui.painter().line_segment(
            [pos2(avail.left() + plot_w + 7.0, avail.top()), pos2(avail.left() + plot_w + 7.0, avail.bottom())],
            ui.visuals().widgets.noninteractive.bg_stroke,
        );
        ui.scope_builder(egui::UiBuilder::new().max_rect(left), |ui| match p.example {
            SoundExample::Thunder => self.air_plot(ui, p),
            SoundExample::Whistler => self.whistler_plot(ui, p),
        });
        ui.scope_builder(egui::UiBuilder::new().max_rect(right), |ui| {
            egui::ScrollArea::vertical().id_salt("sound_controls").auto_shrink([false, false]).show(ui, |ui| {
                ui.columns(2, |cols| match p.example {
                    SoundExample::Thunder => {
                        self.where_controls(&mut cols[0], p, audio);
                        self.air_controls(&mut cols[1], p);
                    }
                    SoundExample::Whistler => self.whistler_controls(cols, p),
                });
            });
        });
    }

    fn where_controls(&mut self, ui: &mut egui::Ui, p: &mut SoundParams, audio: &mut Audio) {
        ui.strong("WHERE YOU ARE");
        ui.horizontal(|ui| {
            ui.label("distance");
            ui.add(egui::Slider::new(&mut p.distance_km, 0.1..=20.0).logarithmic(true).suffix(" km").fixed_decimals(2));
        });
        ui.horizontal_wrapped(|ui| {
            for d in [0.2, 0.5, 1.0, 3.0, 10.0] {
                if ui.button(format!("▶ {} km", fmt_tick(d))).clicked() {
                    p.distance_km = d as f32;
                    self.play_when_ready(p, audio);
                }
            }
        });
        if ui.button("new lightning").on_hover_text("another random channel").clicked() {
            p.seed = p.seed.wrapping_add(1);
        }
        ui.add_space(4.0);
        ui.weak(format!("Sound needs {:.1} s per km. Count the seconds between flash and thunder and divide by 3.", 1000.0 / C0));
    }

    fn air_controls(&mut self, ui: &mut egui::Ui, p: &mut SoundParams) {
        ui.strong("WHAT THE AIR DOES");
        ui.checkbox(&mut p.absorption, "absorption: high frequencies die out")
            .on_hover_text("ISO 9613-1, 20 °C, 70 % humidity: about 5 dB/km at 1 kHz, 23 dB/km at 4 kHz, 0.2 dB/km at 100 Hz");
        ui.checkbox(&mut p.channel, "long zig-zag channel (off: one point)")
            .on_hover_text("the parts of the channel are at different distances: that makes the rumble");
        ui.checkbox(&mut p.dispersion, "dispersion: speed depends on frequency");
        ui.add_enabled_ui(p.dispersion, |ui| {
            ui.horizontal(|ui| {
                ui.label("exaggerated");
                ui.add(egui::Slider::new(&mut p.exaggerate, 1.0..=5000.0).logarithmic(true).prefix("×").fixed_decimals(0));
            });
        });
        let r_km = self.current.as_ref().map_or(p.distance_km as f64, |r| r.r_range.0 as f64);
        let spread = |x: f64| r_km * 1000.0 * (group_slowness(50.0, x) - group_slowness(5000.0, x));
        ui.weak(format!(
            "Real air: over {:.1} km, 50 Hz arrives {:.1} ms after 5 kHz.{}",
            r_km,
            spread(1.0) * 1000.0,
            if p.dispersion && p.exaggerate > 1.0 { format!(" ×{:.0}: {:.2} s.", p.exaggerate, spread(p.exaggerate as f64)) } else { String::new() }
        ));
        ui.checkbox(&mut p.true_loudness, "real loudness (far thunder is quiet)");
    }

    fn whistler_controls(&mut self, cols: &mut [egui::Ui], p: &mut SoundParams) {
        let ui = &mut cols[0];
        ui.strong("MAGNETOSPHERE");
        ui.horizontal(|ui| {
            ui.label("dispersion D");
            ui.add(egui::Slider::new(&mut p.whistler_d, 10.0..=150.0).suffix(" √s")).on_hover_text("group delay t(f) = D/√f; larger D: longer path or denser plasma");
        });
        let mut nose = p.nose_khz > 0.0;
        if ui.checkbox(&mut nose, "nose frequency").on_hover_text("near the electron gyro frequency the delay rises again: a 'nose whistler'").changed() {
            p.nose_khz = if nose { 6.0 } else { 0.0 };
        }
        if nose {
            ui.add(egui::Slider::new(&mut p.nose_khz, 2.0..=15.0).suffix(" kHz"));
        }
        ui.weak(format!("1 kHz arrives after {:.1} s, 4 kHz after {:.1} s.", whistler_delay(1000.0, p.whistler_d as f64, p.nose_khz as f64), whistler_delay(4000.0, p.whistler_d as f64, p.nose_khz as f64)));
        let ui = &mut cols[1];
        ui.strong("RECEIVER");
        ui.radio_value(&mut p.same_hemisphere, false, "other hemisphere (1 hop)");
        ui.radio_value(&mut p.same_hemisphere, true, "near the lightning (click, then 2 hops)");
        ui.checkbox(&mut p.echoes, "echoes (bouncing between the hemispheres)");
        ui.checkbox(&mut p.sferics, "crackle of other lightning (sferics)");
        if ui.button("new").on_hover_text("other random crackle").clicked() {
            p.seed = p.seed.wrapping_add(1);
        }
    }

    /// what is left of each frequency after the distance, and the speed of sound
    fn air_plot(&mut self, ui: &mut egui::Ui, p: &SoundParams) {
        let r_km = self.current.as_ref().map_or(p.distance_km as f64, |r| r.r_range.0 as f64);
        ui.horizontal(|ui| {
            ui.strong("AIR");
            ui.weak(format!("the dispersion relation of sound, over {r_km:.2} km"));
        });
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::hover());
        let painter = ui.painter_at(rect);
        let area = Rect::from_min_max(rect.min + vec2(46.0, 12.0), rect.max - vec2(8.0, 22.0));
        if area.height() < 60.0 {
            return;
        }
        let top = Rect::from_min_size(area.min, vec2(area.width(), area.height() * 0.55 - 8.0));
        let bot = Rect::from_min_max(pos2(area.left(), top.bottom() + 20.0), area.max);
        let (fa, fb) = (20.0f64, 20000.0f64);
        let fx = |f: f64, r: Rect| r.left() + ((f.ln() - fa.ln()) / (fb.ln() - fa.ln())) as f32 * r.width();
        let freqs: Vec<f64> = (0..=300).map(|i| fa * (fb / fa).powf(i as f64 / 300.0)).collect();
        let grey = Color32::from_gray(150);
        let grid = Stroke::new(1.0, Color32::from_white_alpha(14));
        let font = FontId::monospace(9.0);
        for r in [top, bot] {
            painter.rect_filled(r, 3.0, Color32::from_gray(22));
            for f in [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0] {
                let x = fx(f, r);
                painter.line_segment([pos2(x, r.top()), pos2(x, r.bottom())], grid);
                if r == bot {
                    painter.text(pos2(x, r.bottom() + 2.0), Align2::CENTER_TOP, fmt_hz(f), font.clone(), grey);
                }
            }
        }
        painter.text(bot.right_bottom() + vec2(0.0, 12.0), Align2::RIGHT_TOP, "f / Hz", FontId::proportional(10.0), grey);
        // left over after r: 0 … −80 dB
        let ty = |db: f64| top.top() + (-db / 80.0).clamp(0.0, 1.0) as f32 * top.height();
        for db in [0.0, -20.0, -40.0, -60.0, -80.0] {
            let y = ty(db);
            painter.line_segment([pos2(top.left(), y), pos2(top.right(), y)], grid);
            painter.text(pos2(top.left() - 4.0, y), Align2::RIGHT_CENTER, format!("{db:.0}"), font.clone(), grey);
        }
        painter.text(pos2(top.left() - 4.0, top.top() - 2.0), Align2::RIGHT_BOTTOM, "dB", FontId::proportional(10.0), grey);
        let loss: Vec<Pos2> = freqs.iter().map(|&f| pos2(fx(f, top), ty(if p.absorption { -absorption_db_per_km(f) * r_km } else { 0.0 }))).collect();
        painter.add(Shape::line(loss, Stroke::new(2.0, Color32::from_rgb(240, 110, 100))));
        painter.text(top.left_top() + vec2(6.0, 4.0), Align2::LEFT_TOP, "absorption (Im k): what is left of each frequency", FontId::proportional(11.0), Color32::from_rgb(240, 110, 100));
        // speed of sound
        let x = if p.dispersion { p.exaggerate as f64 } else { 1.0 };
        let dc: Vec<f64> = freqs.iter().map(|&f| phase_speed(f, x) - C0).collect();
        let hi = dc.iter().cloned().fold(1e-6, f64::max) * 1.15;
        let by = |v: f64| bot.bottom() - (v / hi) as f32 * bot.height();
        for v in ticks(0.0, hi, 3.0) {
            let y = by(v);
            painter.line_segment([pos2(bot.left(), y), pos2(bot.right(), y)], grid);
            painter.text(pos2(bot.left() - 4.0, y), Align2::RIGHT_CENTER, fmt_tick(v), font.clone(), grey);
        }
        painter.text(pos2(bot.left() - 4.0, bot.top() - 2.0), Align2::RIGHT_BOTTOM, "m/s", FontId::proportional(10.0), grey);
        let pts: Vec<Pos2> = freqs.iter().zip(&dc).map(|(&f, &v)| pos2(fx(f, bot), by(v))).collect();
        painter.add(Shape::line(pts, Stroke::new(2.0, Color32::from_rgb(110, 190, 255))));
        let label = if p.dispersion && x > 1.0 {
            format!("speed of sound − {C0:.1} m/s (Re k), exaggerated ×{x:.0}")
        } else {
            format!("speed of sound − {C0:.1} m/s (Re k): real air, tiny")
        };
        painter.text(bot.left_top() + vec2(6.0, 4.0), Align2::LEFT_TOP, label, FontId::proportional(11.0), Color32::from_rgb(110, 190, 255));
        if let Some(q) = resp.hover_pos() {
            if area.contains(q) {
                let f = (fa.ln() + ((q.x - area.left()) / area.width()) as f64 * (fb.ln() - fa.ln())).exp();
                readout(
                    &painter,
                    area.right_top() + vec2(-4.0, 4.0),
                    Align2::RIGHT_TOP,
                    format!("f = {} Hz\nα = {:.2} dB/km\nc − c₀ = {:.4} m/s", fmt_hz(f.round()), absorption_db_per_km(f), phase_speed(f, x) - C0),
                );
            }
        }
    }

    fn whistler_plot(&mut self, ui: &mut egui::Ui, p: &SoundParams) {
        ui.horizontal(|ui| {
            ui.strong("MAGNETOSPHERE");
            ui.weak("group delay of one hop: the dispersion relation shows up as t(f)");
        });
        let (rect, _) = ui.allocate_exact_size(ui.available_size(), Sense::hover());
        let painter = ui.painter_at(rect);
        let area = Rect::from_min_max(rect.min + vec2(46.0, 12.0), rect.max - vec2(8.0, 22.0));
        if area.height() < 40.0 {
            return;
        }
        painter.rect_filled(area, 3.0, Color32::from_gray(22));
        let (fa, fb) = (0.0, 15000.0);
        let d = p.whistler_d as f64;
        let t_max = whistler_delay(400.0, d, 0.0);
        let px = |f: f64| area.left() + ((f - fa) / (fb - fa)) as f32 * area.width();
        let py = |t: f64| area.bottom() - (t / t_max) as f32 * area.height();
        let grey = Color32::from_gray(150);
        let grid = Stroke::new(1.0, Color32::from_white_alpha(14));
        for f in ticks(fa, fb, 6.0) {
            painter.line_segment([pos2(px(f), area.top()), pos2(px(f), area.bottom())], grid);
            painter.text(pos2(px(f), area.bottom() + 2.0), Align2::CENTER_TOP, fmt_hz(f), FontId::monospace(9.0), grey);
        }
        for t in ticks(0.0, t_max, 4.0) {
            painter.line_segment([pos2(area.left(), py(t)), pos2(area.right(), py(t))], grid);
            painter.text(pos2(area.left() - 4.0, py(t)), Align2::RIGHT_CENTER, fmt_tick(t), FontId::monospace(9.0), grey);
        }
        painter.text(area.right_bottom() + vec2(0.0, 12.0), Align2::RIGHT_TOP, "f / Hz", FontId::proportional(10.0), grey);
        painter.text(pos2(area.left() - 4.0, area.top() - 2.0), Align2::RIGHT_BOTTOM, "t / s", FontId::proportional(10.0), grey);
        let pts: Vec<Pos2> = (1..=300)
            .map(|i| 400.0 + (fb - 400.0) * i as f64 / 300.0)
            .map(|f| (f, whistler_delay(f, d, p.nose_khz as f64)))
            .filter(|(_, t)| t.is_finite() && *t < 2.0 * t_max)
            .map(|(f, t)| pos2(px(f), py(t)))
            .collect();
        ui.painter_at(area).add(Shape::line(pts, Stroke::new(2.0, Color32::from_rgb(120, 230, 160))));
        painter.text(
            area.left_top() + vec2(6.0, 4.0),
            Align2::LEFT_TOP,
            if p.nose_khz > 0.0 { "t(f) = D/√f · (1 − f/f_H)^(−3/2)" } else { "t(f) = D/√f   (whistler mode: ω ~ k², v_g = 2 v_p)" },
            FontId::proportional(11.0),
            Color32::from_rgb(120, 230, 160),
        );
    }
}
