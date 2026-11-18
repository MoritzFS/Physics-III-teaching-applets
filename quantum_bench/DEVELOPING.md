# Developing Quantum Bench

Guide for the scene agents: crate layout, the `Tab` API, numerics, widgets,
screenshot mode, the web rules and the conventions every tab follows.

## 1. Ground rules

* **Frozen:** `crates/qb_core/**`, every `Cargo.toml`, `Cargo.lock` policy
  (do not add dependencies), `.cargo/config.toml`, `Trunk.toml`,
  `index.html`, `app/`, `shot.sh`, `bundle.sh`, `rustfmt.toml`. If something
  is missing in qb_core, work around it in your crate (copy and adapt) and
  report it; do not edit qb_core.
* **Yours:** only your tab crate, `crates/qb_<tab>/` (`src/**`, `tests/**`).
  The four tab crates are built in parallel, so never touch another tab's crate.
* The app calls exactly `qb_wave1d::Wave1dTab::new()`, `qb_box2d::Box2dTab::new()`,
  `qb_hydrogen::HydrogenTab::new()`, `qb_spin::SpinTab::new()`, each boxed as
  `Box<dyn Tab>`. Keep that constructor; what it returns is up to you (any type
  implementing `Tab`).
* The owner teaches with this. Physics statements in notes and labels must be
  correct; say when something is an approximation.
* Before you finish: `cargo build --release` and `cargo clippy --release -p <crate>`
  without warnings, `cargo test --release -p <crate>` green, the **wasm check**
  (section 6) passes, and you have looked at screenshots of every preset.

## 2. Layout

```
quantum_bench/                 cargo workspace (run everything from here)
├ Cargo.toml                   workspace + [workspace.dependencies]; profiles release (fast) and web (LTO)
├ .cargo/config.toml           target-dir = "target.nosync" (iCloud must not sync it)
├ crates/qb_core/              shared, FROZEN
│  ├ src/tab.rs                Tab, Scene, SceneTab, Ctx, PresetInfo, PlaceholderScene
│  ├ src/shell.rs              run_app: top bar, side panel + notes, keys, screenshot mode, URL hash
│  ├ src/look.rs               Look: colours (dark/light), presentation scaling
│  ├ src/num/                  1D numerics (grid, wave, FFT, propagator, eigen, expansion, scattering, measurement)
│  ├ src/widgets/              PlotFrame, WavePlot, ReprView/view_bar, LevelLadder, HistogramPlot,
│  │                           TimeSeriesPlot, Heatmap2D, MeasureFlash, phase colour, readouts
│  ├ src/gpu.rs                ShaderRect (fragment shader in a UI rectangle), PhaseWheelDemo
│  ├ src/rng.rs, par.rs, clock.rs   QbRng, rayon-or-sequential helpers, wall_time()
│  ├ src/web.rs, shot.rs       browser start-up / desktop screenshots
│  ├ fonts/                    DejaVu Sans fallback (⟨ ⟩, ₓ, ₚ, arrows …) + its license
│  └ tests/                    numerics.rs (physics checks), wgsl.rs (naga validation)
├ crates/qb_wave1d/            "1D waves"  (bin wave1d_solo)   reference scene: src/gaussian.rs
├ crates/qb_box2d/             "2D box"    (bin box2d_solo)    stub
├ crates/qb_hydrogen/          "Hydrogen"  (bin hydrogen_solo) stub + GPU check
├ crates/qb_spin/              "Spin"      (bin spin_solo)     stub
├ app/                         bin quantum_bench (all four tabs)
├ index.html, Trunk.toml       web build (trunk, from this folder)
├ shot.sh, bundle.sh
└ shots/                       screenshots
```

Dependencies available in every tab crate: `qb_core`, `eframe` (egui, egui_wgpu,
wgpu via `qb_core::{egui, egui_wgpu, wgpu}`), `bytemuck`, `glam`,
`num-complex`, `rustfft`, `serde`, `serde_json`, `log`, `web-time`; `rayon`
(native only, see section 6); dev: `naga`. There is **no `rand`**: use `QbRng`.

## 3. Commands

```bash
cargo build --release                         # everything (native)
cargo run --release                           # full app
cargo run --release -p qb_spin                # one tab alone (bin spin_solo)
cargo test --release -p qb_core               # numerics + WGSL checks
cargo test --release                          # all tests (what CI runs)
./shot.sh shots/x.png spin 2 1.5 "measure"    # screenshot (section 5)
BIN=spin_solo ./shot.sh shots/x.png           # screenshot of a solo binary
# web (rustup's cargo; Homebrew's cannot build wasm):
PATH="$HOME/.cargo/bin:$PATH" cargo clippy --release --target wasm32-unknown-unknown -p qb_spin
PATH="$HOME/.cargo/bin:$PATH" trunk build --release      # → dist/
PATH="$HOME/.cargo/bin:$PATH" trunk serve                # http://127.0.0.1:8080/#spin
```

CI (GitHub Pages) runs from this folder: `cargo test --release`, then
`trunk build --release --public-url /Physics-III-teaching-applets/quantum-bench/ --dist ../_site/quantum-bench`.

## 4. Writing a tab

### 4.1 `Scene` + `SceneTab` (recommended)

One scene struct per exercise; the tab is a list of presets:

```rust
use qb_core::{Ctx, PresetInfo, Scene, SceneTab, Tab, egui};

pub struct SpinTab;
impl SpinTab {
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> impl Tab {
        SceneTab::new("spin", "Spin")
            .preset(PresetInfo::new("PS13 50 · Stern–Gerlach sequence"), |_ctx| Box::new(SgScene::new()))
            .preset(PresetInfo::new("PS13 51 · Ramsey experiment").paused().speed(0.5), |_ctx| Box::new(RamseyScene::new()))
    }
}

impl Scene for SgScene {
    fn time(&self) -> f64 { self.t }
    fn step(&mut self, dt: f64, ctx: &mut Ctx) { /* advance by exactly dt */ }
    fn reset(&mut self, ctx: &mut Ctx) { /* t = 0, keep the user's parameters */ }
    fn central_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx) { /* plots */ }
    fn side_ui(&mut self, ui: &mut egui::Ui, ctx: &mut Ctx) { /* controls */ }
    fn notes(&self) -> String { NOTES.into() }
    // optional: time_scale, single_step, measure_label, measure, on_key, action
}
```

Loading a preset builds a fresh scene (parameters back to the preset's).
`src/gaussian.rs` in qb_wave1d is the complete reference; copy its structure:
parameters → `restart()` → `step()` → `central_ui()` (view bar, readouts,
plots, time series, histogram) → `side_ui()` (parameters, measurement) →
`notes()`.

### 4.2 The `Tab` trait (if you need state shared across presets)

| method | meaning |
|---|---|
| `id() -> &'static str` | "wave1d", "box2d", "hydrogen", "spin": used by `QB_TAB` and the URL hash |
| `name()` | text in the tab selector |
| `presets() -> Vec<PresetInfo>` | the Examples menu; keys 1–9 load the first nine |
| `load_preset(i, ctx)` | load preset i (0-based) and restart at t = 0 |
| `reset(ctx)` | key R: restart at t = 0 **keeping** the user's current parameters |
| `time()` | simulation time, shown in the top bar, used by `QB_T` |
| `time_scale()` | simulation time units per wall second at speed 1 (default 1) |
| `step(dt, ctx)` | advance by exactly `dt` (the shell passes `dt_wall·speed·time_scale`); sub-step yourself |
| `single_step(ctx)` | key "." (default: 1/30 s of wall time at the current speed) |
| `central_ui(ui, ctx)` / `side_ui(ui, ctx)` | main area / right panel above the notes |
| `notes()` | notes of the current preset (markup below) |
| `measure_label()` / `measure(ctx)` | top-bar button + key M (None hides the button) |
| `on_key(key, mods, ctx) -> bool` | any key the shell does not use |
| `action(&str, ctx) -> bool` | scripted actions for screenshots (`QB_ACTION`) |
| `init_gpu(render_state)` | once, when wgpu is available (SceneTab does not forward it: scenes use `ctx.render_state` lazily, see the hydrogen stub) |

`Ctx` fields: `playing` and `speed` (read/write: e.g. pause when a sequence
ends), `dt_wall`, `rng: &mut QbRng`, `look: Look`, `render_state:
Option<&RenderState>`, `shot` (true while preparing a screenshot);
`ctx.set_status("…")` writes the status line (keep it for measurement outcomes).

Keys used by the shell: Space, R, M, P, `.`, 1–9. **V is reserved in every 1D
scene for cycling the display mode** (`style = style.next()`). Other keys are free.

### 4.3 Presets, notes

* Preset names: `"PS<set> <exercise> · <short title>"`, e.g.
  "PS10 39 · Barrier tunnelling". One preset per exercise; a second preset
  for the same exercise only for a clearly different situation
  (e.g. "PS10 38 · Moving packet, k₀ = 3 (Re ψ)").
* `PresetInfo::new(name)` autoplays at speed 1; `.paused()`, `.speed(x)`.
* Notes markup: `# Heading`, `- bullet`, blank line = new paragraph. Structure:
  "# The exercise" (what it asks, with the formulas the applet shows),
  "# Try" (concrete things to do and what to watch), optionally
  "# Behind the scenes" (method, grid, approximations).
* Glyphs: ⟨ ⟩ ψ Φ σ ħ ₀ ₓ ₚ ² √ ∫ ± → ↑ ↓ ⊗ all render (DejaVu Sans fallback is embedded).

## 5. Screenshot mode (desktop)

| variable | meaning |
|---|---|
| `QB_SHOT=out.png` | take a screenshot and quit |
| `QB_TAB=spin` | tab by id, name prefix or index (default: first) |
| `QB_PRESET=2` | preset, 0-based (default 0) |
| `QB_T=3.5` | advance to simulation time 3.5 in exact chunks of `time_scale/60` (not wall clock) |
| `QB_ACTION="view_both;measure_x@2;repeat_x 500"` | `Tab::action` calls; `@t` runs one at time t, the others at QB_T |
| `QB_PRESENT=1`, `QB_LIGHT=1` | presentation mode, light theme |
| `QB_SEED=7` | seed of every tab's `QbRng` (default 1) |
| `QB_SIZE=1600x1000`, `QB_WAIT=1` | window size in points; extra seconds before the shot (GPU warm-up) |

`./shot.sh out.png [tab] [preset] [t] [actions]` sets these (plus `PRESENT=1`,
`LIGHT=1`, `SEED=`, `SIZE=`, `WAIT=`, `BIN=<x>_solo`). The window pops up for a
moment. The simulation is paused for the shot; the state, saved settings and
mouse are ignored.

For reproducible shots: draw all randomness from `ctx.rng`; time animations
with `qb_core::wall_time()` (in shot mode it is a virtual clock that advances
with the simulation, one second per `time_scale` units, so a flash triggered at
`@2` is gone in a shot at t = 5, and one triggered at QB_T is shown a quarter
second into its animation); never depend on frame timing in `step`.

Implement `action` for everything a slide may need: at least `measure…`,
`repeat… N`, `view_x|view_p|view_both`, `style_density|style_re|style_reim|style_plain`
in 1D scenes, and parameter setters you find useful. Unknown actions are
reported on stderr.

## 6. Web build: rules for tab crates

The same code runs in the browser (wasm32, single thread, WebGPU or WebGL2).
Forbidden in tab crates, with the replacement:

| don't | use |
|---|---|
| `rayon`, `std::thread` (no threads on the web) | `qb_core::par::{map, map_range, for_each_mut, chunks_mut, sum_range}`; if you really need rayon, `#[cfg(not(target_arch = "wasm32"))]` with a sequential `cfg(target_arch = "wasm32")` twin |
| `std::time::Instant`, `SystemTime` (panics on wasm) | `qb_core::Instant` (web-time) or `qb_core::wall_time()` for animations |
| `std::fs`, `std::env`, `std::process` | nothing (screenshot mode is the shell's job) |
| `rand`, `getrandom` | `ctx.rng` (`QbRng`: `uniform, range, below, bool, normal, gauss, exponential, poisson, binomial, weighted, shuffle, fork`) |
| compute shaders, storage buffers/textures | `qb_core::gpu::ShaderRect` (render pipeline, uniforms ≤ 16 KiB, one 3D `Rgba32Float` texture read with `textureLoad`) |
| heavy work per frame | budget for one thread: 1D grids of 1024–2048 points, 2D up to 256² per frame, precompute tables once |

Check before finishing (Homebrew's cargo cannot build wasm, use rustup's):

```bash
PATH="$HOME/.cargo/bin:$PATH" cargo clippy --release --target wasm32-unknown-unknown -p qb_<tab>   # must be clean
PATH="$HOME/.cargo/bin:$PATH" trunk build --release    # optional: full web build into dist/
```

Deep links work for every tab without extra code: `…/#spin` opens the tab,
`…/#spin/2` its preset 2 (0-based); the shell keeps the hash up to date. In the
browser `?webgl` forces WebGL2 (e.g. `http://127.0.0.1:8080/?webgl#hydrogen`).

## 7. Conventions

* **Units** ħ = m = 1 (1D numerics). Hydrogen: atomic units (a₀, Hartree or
  Rydberg, say which in the notes); spin: ħ = 1, energies in units of ħω.
* **Phase colour**: hue = arg ψ (0 red, π/3 yellow, 2π/3 green, π cyan,
  4π/3 blue, 5π/3 magenta), brightness or height = |ψ| or |ψ|². Use
  `phase_color(arg, intensity)` on the CPU and `qb_phase_color` (in
  `PHASE_WGSL`) on the GPU; always show the legend (`phase_wheel`; WavePlot and
  Heatmap2D draw it for you).
* **Display modes of a wavefunction (1D, every scene, x and p alike)**:
  `WaveStyle::Density` "|ψ|² + phase" (default), `Real` "Re ψ" (the actual
  wave with a faint ±|ψ| envelope band: shows carrier and group envelope, phase
  vs group velocity), `ReIm` "Re ψ, Im ψ", `Plain` "|ψ|²" without phase. Put
  `view_bar(ui, &mut self.view, &mut self.style)` at the top of the central
  area (the x | p | x + p switch and the mode switch in one row), handle key V
  with `self.style = self.style.next()`, and actions `style_<density|re|reim|plain>`.
* **Stable vertical scale**: keep one "top of |ψ|²" per plot, set at
  (re)start from the initial state and only grown when the curve would be
  clipped (`if peak > top { top = 1.2·peak }`), and get the range with
  `style.y_range(top)` ([0, top] or ±√top). Never autoscale per frame: the
  flattening of a spreading packet must stay visible.
* **Measurement UX** (same in every tab): a single shot (key M, top-bar
  button via `measure_label`) gives one outcome, collapses the displayed state,
  shows `MeasureFlash`, writes the outcome to the status line, and the state
  keeps evolving. "Repeat N" (`repeat_button`) measures N **fresh copies** of
  the current state (the displayed state is untouched) into a `ShotHistogram`,
  drawn with `HistogramPlot` against the Born-rule prediction. Detector
  resolution is a slider where it makes sense.
* **Presentation mode** (P): use `Look` for every size and colour:
  `look.font(13.0)`, `look.stroke(1.5, c)`, `look.px(8.0)`, `look.accent()` …
  never hard-coded `FontId`/`Stroke` widths or `Color32` greys, so the
  projector view and light mode work. Check a `PRESENT=1` and a `LIGHT=1`
  screenshot of each scene.
* **Colours** (`Look`): `text, weak_text, plot_bg, grid, axis, potential,
  potential_fill, re, im, abs, accent (⟨x⟩, handles), measure (detectors,
  collapse), counts (histogram bars), prediction (Born rule, analytic curves),
  level, series(i)`.
* **Readouts**: `readout_row(ui, &[("σₓ", fmt_num(s, 3), Some(look.series(0))), …])`;
  `fmt_num` avoids "-0.00".

## 8. Numerics (`qb_core::num`, ħ = m = 1)

```rust
use qb_core::num::*;
let g = Grid1D::new(-60.0, 60.0, 2048);        // periodic, n even; g.x(i), g.xs(), g.dx, g.ps(), g.dp(), g.p_max()
let mut w = Wave1D::gaussian(g, x0, sigma, p0); // σ = σ_x; PS10 38's A e^(−ax²) is sigma = 1/(2√a)
let w2 = Wave1D::plane_wave_truncated(g, 0.0, 5.0, 3.0);   // PS09 37
let w3 = Wave1D::from_fn(g, |x| c64(x.exp(), 0.0));         // then w3.normalize()
w.norm_sqr(); w.density(); w.prob_between(a, b); w.inner(&other); w.at(x); w.current();
let m = w.moments();     // m.norm, m.x, m.sigma_x, m.p, m.sigma_p, m.cov_xp, m.uncertainty_product()
m.free_sigma_x(dt);      // exact free-particle σ_x after dt, for any state
let phi = w.momentum();  // phi.p (ascending), phi.phi, phi.dp; phi.density(), phi.moments(), phi.prob_between()
let back = Wave1D::from_momentum(&phi);
// Φ(p) = (2π)^(−1/2) ∫ψ e^(−ipx) dx with absolute x; Σ|Φ|²dp = Σ|ψ|²dx exactly.
```

**Time evolution** (split-step, Strang, error O(dt³)/step; V may be complex):

```rust
let mut prop = SplitStep::new(g, 0.01)                 // dt_max
    .with_potential(&v)                                // real V on the grid
    .with_absorber(Absorber::new(8.0, 3.0));           // optional absorbing edges (width, strength ~ p²/2)
prop.set_complex_potential(&vc);                       // V0 − iΓ/2 ⇒ ∫|ψ|² = e^(−Γt) (PS09 36)
prop.advance(&mut w, dt);                              // exactly dt in ⌈dt/dt_max⌉ equal steps
```

**Bound states and expansions**:

```rust
let basis = EigenBasis::numeric(g, &v, 20);            // FD + Sturm bisection + inverse iteration; first lobe positive
let ho = EigenBasis::harmonic(g, 40, omega, x0);       // analytic, textbook sign (ψ_n(+∞) > 0) = (−1)ⁿ × numeric
let well = EigenBasis::infinite_well(g, 10, a, b);     // index k ↔ n = k + 1
ho_psi(n, x, omega); ho_energy(n, omega); well_psi(n, x, a, b); well_energy(n, l);
let basis = std::sync::Arc::new(basis);
let mut e = EigenExpansion::project(basis.clone(), &w); // c_n = ⟨ψ_n|ψ⟩; e.captured() = Σ|c_n|²
let coh = EigenExpansion::coherent(ho_arc, c64(2.0, 0.0)); // PS10 40
e.advance(dt); e.coeffs(); e.probabilities(); e.mean_energy(); let psi_t = e.wave();
```

**Scattering** (incoming amplitude 1, T = (k_R/k_L)|t|², R = |r|²):

```rust
let barrier = Piecewise::barrier(0.0, 1.0, 1.0);       // or Piecewise::new(edges, values), Piecewise::step(x0, v0)
let s = barrier.scatter(e);                            // s.transmission, s.reflection, s.psi(x), s.sample(&xs)
rect_barrier_transmission(e, v0, a);                   // PS10 39 analytic formula
let n = numerov_scatter(-5.0, 5.0, 0.01, |x| v(x), e); // PS10 41: arbitrary V; n.transmission, n.xs, n.psi
```

**Measurement** (`rng = ctx.rng`):

```rust
let x0 = measure_position(&mut w, width, rng);   // x0 from |ψ|², then ψ·exp(−(x−x0)²/(4w²)), renormalised
let p0 = measure_momentum(&mut w, width_p, rng); // the same in p space
let yes = measure_region(&mut w, a, b, rng);     // projective "in [a, b]?", sharp projector
let n = measure_energy(&mut expansion, rng);     // Option<usize>, collapses onto ψ_n (keeps the phase)
collapse_position(&mut w, x0, width);            // deterministic part only
let xs = born_samples_x(&w, 1000, rng);          // "repeat N" on fresh copies (also born_samples_p)
let mut h = ShotHistogram::binned(lo, hi, 40);   // or ShotHistogram::discrete(n_outcomes)
h.add_all(&xs); h.add_index(k);
let pred = h.predict_from_density(&w.xs(), &w.density());   // Born-rule probability per bin
Sampler::new(&grid_xs, &density)                  // reusable sampler for any 1D density
```

**FFT** (`qb_core::num::fft`, cached plans, unnormalised): `fft::forward(&mut buf)`,
`fft::inverse(&mut buf)`, `fft::fft2(&mut data, nx, ny, inverse)` (row-major,
rows in parallel on the desktop) for the 2D box.

**Tested** (`cargo test -p qb_core`): Parseval and Φ(p) of a Gaussian (modulus
and phase), free σ_x(t) and ψ(x, t) against the analytic packet, norm and
energy conservation, e^(−Γt) decay, absorber, HO levels n + ½ and
orthonormality, double-well splitting, infinite-well levels, coherent state
⟨x⟩ = √2|α|cos t with σ_x = 1/√2 (expansion and propagator), energy
measurement statistics, rectangular barrier T(E) (transfer matrix and
Numerov) including the resonance, position/momentum/region measurements and
histogram predictions, the moment-based spreading prediction after a collapse.

## 9. Widgets (`qb_core::widgets`)

All plots return `PlotResponse { response, transform, hover }`;
`transform: PlotTransform` maps data ↔ screen (`to_screen(x, y)`,
`from_screen(pos)`, `x_to_screen`, `rect` …) for your own overlays and mouse
interaction; `resp.painter(ui)` gives a painter clipped to the plot.
Pan/zoom: scroll/pinch zooms around the cursor, drag pans, double-click
resets; the view resets when the default ranges you pass change.

**PlotFrame**: axes + ticks + labels for custom plots:
`PlotFrame::new("id").x_range(a, b).y_range(c, d).labels("t", "P").height(200.0).pan_zoom(PanZoom::None).show(ui, |painter, tr| { … })`.
Also `.right_axis(label, range)`, `.left_drag_pans(false)` (keep left-drag for
yourself, e.g. drawing a potential), `.grid(false)`. Helpers: `plot_line`,
`plot_dashed`, `label_box`, `nice_ticks`.

**WavePlot**:

```rust
let resp = WavePlot::new(("my_scene", domain))
    .domain(domain)                       // Domain::Position | Domain::Momentum (labels x/ψ or p/Φ)
    .style(self.style)                    // WaveStyle: Density | Real | ReIm | Plain
    .x_range(-25.0, 25.0).y_range(lo, hi) // (lo, hi) = self.style.y_range(self.top)
    .height(h)
    .potential(&xs, &v, [0.0, 2.0])       // V drawn behind, [V_lo, V_hi] mapped onto the plot (right axis "V")
    .curve(WaveCurve::new(&xs, &psi))     // .baseline(E_n).scale(s).label("ψ₂").color(c).style(s)
    .marker(Marker::vline(m.x, "⟨x⟩"))
    .marker(Marker::bracket(m.x - m.sigma_x, m.x + m.sigma_x, y, "2σₓ = 1.0"))
    .show(ui);
self.flash.paint(ui, &resp.transform, domain);
```

Markers: `vline, hline, span (shaded band), bracket, point`, with `.color(c)`,
`.dashed()`; `marker.paint(painter, tr)` works on any plot. With
`.energy_axis(true)` the y axis is energy, V is drawn in its own units and
curves sit on their baselines (ψ_n on E_n, scaled by `scale`).

**ReprView + view_bar**: `view_bar(ui, &mut self.view, &mut self.style)`
(top of the central area), then
`self.view.show(ui, height, |ui, domain, h| self.plot(ui, domain, h))` stacks
x above p (55 / 45 %) with aligned axes. `ReprView::parse("both")`,
`WaveStyle::parse("re")`, `style.next()`, `style.label(domain)`.

**LevelLadder**: `LevelLadder::new("id", &energies).coeffs(&c_t).selected(Some(n)).energy_range(lo, hi).size(vec2(180.0, h)).show(ui)`
→ `LadderResponse { clicked, hovered, transform, response }`. Bars = |c_n|²,
phasors = arg c_n(t) in the phase colour (rotate as c_n(t) = c_n e^(−iE_n t)).
Use the same energy range as an energy-axis WavePlot so the levels line up.

**HistogramPlot**: `HistogramPlot::new("id", &hist).predicted(&pred).labels(&names).x_label("x").scale(HistScale::Density).height(160.0).show(ui)`;
bars = measured, line/markers = Born rule, shaded = ±1 standard deviation of
the counts for N shots; "N = …" and "k outside" in the corner.
`HistScale::{Fraction, Density, Counts}`.

**TimeSeries / TimeSeriesPlot**: keep `TimeSeries::new("σₓ")` in the scene,
`ts.push(t, y)` (a smaller t than the last clears it, e.g. after a reset);
`TimeSeriesPlot::new("id").series(&a).series(&b).function("analytic", look.prediction(), &|t| f(t)).hline(0.5, "ħ/2").event(t_m, "x").window(TimeWindow::All).now(t).labels("t", "σ").height(h).show(ui)`.
Functions return NaN where undefined.

**Heatmap2D** (2D box, hydrogen slices): keep a `Heatmap2D::new()` in the
scene; `map.set_field(ui.ctx(), nx, ny, &field, HeatmapMode::PhaseBrightness, None)`
when the field changes (row-major, row 0 = y_min, drawn at the bottom), then
`map.show(ui, "id", [x0, x1, y0, y1], ("x", "y"), height)` (equal aspect,
legend, pan/zoom). Modes: `PhaseBrightness` (|ψ| → brightness), `PhaseDensity`,
`Density` (sequential map), `PhaseOnly`. `field_image(..)` gives the `ColorImage`.
`max_abs: Some(m)` fixes the brightness scale across frames (do that while animating).

**MeasureFlash**: `self.flash.trigger(FlashKind::Position { x, width }, format!("x = {x:.2}"))`
(also `Momentum { p, width }`, `Region { a, b, yes }`, `Level { index, energy }`,
`Other`); call `self.flash.paint(ui, &resp.transform, domain)` after every plot
it belongs on, or `paint_banner(ui, rect)` for views without axes. `clear()` on
reset. Shows the detector window, a short flash and the outcome for 4 s, then a
small marker stays.

**Colour**: `phase_color(arg, intensity)`, `phase_color_alpha`, `phase_rgb`,
`complex_color(z, max_abs)`, `density_color(t)`, `phase_wheel(painter, center, r, Some("arg ψ"))`,
`PHASE_WGSL` (`qb_phase_color`, `qb_density_color` for shaders).

**Readouts**: `readout(ui, label, value, color)`, `readout_row`, `fmt_num`,
`repeat_button(ui, "id", "Repeat x", &mut self.n) -> Option<N>`, `segmented(ui, |ui| …)`.

Not provided: a Bloch sphere (the spin agent builds it in its crate, with
`PlotTransform`-style projection and `Look` colours).

## 10. GPU (`qb_core::gpu`)

`ShaderRect` draws a WGSL fragment shader into a rectangle of the UI via an
`egui_wgpu` paint callback (render pipeline, premultiplied alpha, sRGB handled).
Your WGSL defines `struct Uniforms { … }` (use `vec4<f32>` fields: a `vec3`
after a scalar is 16-byte aligned in WGSL but not in Rust) and
`fn shade(uv: vec2<f32>) -> vec4<f32>` (uv ∈ [0,1]², y down; return sRGB colour,
straight alpha). Bindings: `qb_frame.rect_px` (pixels), `u`, `qb_data`
(`texture_3d<f32>`, `Rgba32Float`, read with `textureLoad`, interpolate yourself).

```rust
let view = ShaderRect::new(rs, "orbital", MY_WGSL, std::mem::size_of::<MyUniforms>(), Some([64, 64, 64]));
view.write_data(rs, &texels);                                  // [f32; 4] per texel, x fastest
view.paint(ui.painter(), rect, bytemuck::bytes_of(&uniforms)); // once per frame per ShaderRect
```

`PhaseWheelDemo` (used by the hydrogen stub) is the minimal working example;
it is verified natively (Metal), in the browser with WebGPU and with WebGL2
(`?webgl`). Validate your WGSL in a test with naga, as in `crates/qb_core/tests/wgsl.rs`:
`validate(&qb_core::gpu::shader_source(MY_WGSL, false))`.
Get the render state from `ctx.render_state` (Option; it is None only without wgpu).

## 11. Checklist for a new scene

1. Struct with parameters + state; `new()` sets the preset's defaults and calls `restart()`.
2. `step(dt)`: exact advance (sub-steps), update cached moments/series.
3. `central_ui`: `view_bar` (1D), readouts, plots with markers, `flash.paint`,
   time series / histogram strip.
4. `side_ui`: parameters (changing one restarts), measurement controls
   (single shot + repeat N), short help line.
5. `notes()`: exercise, try, behind the scenes.
6. `measure_label`/`measure`, `on_key` (V), `action` for screenshots.
7. Screenshots: every preset at t = 0 and later, after a measurement,
   `PRESENT=1`, `LIGHT=1`; look at them.
8. Native build + clippy clean, tests, wasm clippy clean.
