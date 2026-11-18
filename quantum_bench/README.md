# Quantum Bench

Interactive quantum wave mechanics for Physics III (problem sets 9–13):
wave packets, measurement and collapse, tunnelling, the harmonic oscillator,
the particle in a box, the hydrogen atom and spin. Written in Rust with
egui/eframe and wgpu; runs as a desktop app and in the browser.

```
┌ top bar: tabs · Examples · play/pause, step, reset, speed, t · Measure · Present, Light/Dark ┐
├ central area: the scene (plots, views)          ├ side panel: controls + notes ┤
└ status line: last measurement, keys ───────────────────────────────────────────┘
```

Tabs: **1D waves**, **2D box**, **Hydrogen**, **Spin**. Each tab has an
"Examples" menu with one entry per exercise (e.g. "PS10 38 · Gaussian wave
packet") and notes on what the exercise asks and what to try.

| tab | examples |
|---|---|
| **1D waves** | PS10 38 Gaussian wave packet (also as a moving wave, Re ψ) · PS09 36 decaying particle (complex potential) · PS09 37 truncated plane wave in x and p · PS10 39 barrier tunnelling, T(E) and resonances · PS10 40 harmonic oscillator and coherent states · ladder operators â, â† acting on the state, with the Fock-state populations (script 6.8) · PS10 41 draw your own potential (Numerov) · PS12 49 oscillator in an electric field |
| **2D box** | PS09 38 eigenstates and degeneracy, sloshing and circulating superpositions, a Gaussian packet and its revival |
| **Hydrogen** | PS11 43 quantum numbers and ladder operators · PS11 44 energy and radius · PS11 45 ψ₂₁₁ and ψ₂₁₋₁ · 1s + 2p oscillating dipole · PS12 47 Zeeman effect; with a 3D density view (cloud, isosurface, point cloud, cut plane) |
| **Spin** | PS09 35 non-commuting observables · PS13 50 Stern–Gerlach sequences · PS13 51 Larmor precession and the Ramsey sequence |

Every scene shows the time evolution, the energy eigenstates and what a
measurement does: a single shot collapses the state, which then keeps
evolving; "repeat N" compares many shots with the Born rule.

## Run

```bash
./bundle.sh                 # builds "../Quantum Bench.app" (double-clickable)
cargo run --release         # or, during development: the full app
cargo run --release -p qb_spin   # one tab alone (also qb_wave1d, qb_box2d, qb_hydrogen)
```

The build output goes to `target.nosync/` so that iCloud doesn't sync it.

Web version (needs rustup's Rust with the `wasm32-unknown-unknown` target, and
[Trunk](https://trunkrs.dev)):

```bash
trunk serve                 # http://127.0.0.1:8080, also …/#wave1d/1, #hydrogen, #spin
trunk build --release       # into dist/
```

It uses WebGPU and falls back to WebGL2 (add `?webgl` to the URL to force WebGL2).

## Using it

* **Keys**: Space play/pause · `.` one step · R restart the example · M
  measure · 1–9 load an example · P presentation mode (bigger text and lines,
  for the projector) · V cycle the display mode of a wavefunction.
* **Wavefunctions**: colour = phase arg ψ (colour wheel in the corner),
  height = |ψ|². The display switch above the plot also offers Re ψ (the
  actual wave with its envelope ±|ψ|), Re ψ and Im ψ together, and plain |ψ|².
  "show x / p / x + p" switches between position and momentum space.
* **Plots**: scroll or pinch to zoom, drag to pan, double-click to reset.
* **Measurements**: "Measure" is one shot: the state collapses (with the
  detector's resolution) and keeps evolving. "Repeat N" measures N fresh
  copies of the current state and compares the histogram with the Born rule.
* Units: ħ = m = 1.

## Screenshots (for slides)

```bash
./shot.sh out.png wave1d 0 4 "view_both;measure_x@2"     # tab, preset, time, actions
PRESENT=1 LIGHT=1 ./shot.sh out.png wave1d 1 3
BIN=spin_solo ./shot.sh out.png                          # a single tab
```

See `shot.sh` and DEVELOPING.md for all options.

## For developers

DEVELOPING.md describes the crate layout, the `Tab` trait, the numerics and
widget API, screenshot mode, the web rules and the conventions.
`cargo test --release` runs the physics checks of the numerics.
