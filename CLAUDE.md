# Notes for Claude

The current task is in **HANDOFF.md**: port `optics_bench/` to the web
(WebAssembly + WebGPU) and deploy it with GitHub Pages. Read it first.

## Layout

- `optics_bench/`: a Rust app (egui/eframe + wgpu).
  - `src/shader.wgsl`: GPU ray tracer.
  - `src/gpu.rs`: compute pipeline.
  - `src/scene.rs`: scene model, 3D objects and ray-optics examples.
  - `src/trace.rs`: CPU tracer (auto focus, ray fans, picking).
  - `src/app.rs`: user interface.
  - `src/configs.rs`: saving configurations as JSON.
  - `src/fourier.rs`: 4f wave-optics engine and its examples.
  - `src/fourier_ui.rs`: 4f user interface.
  - `optics_bench/README.md` explains features and physics.
- One folder per applet; the repo root holds the landing page and CI.

## Working on it

- Build and test from `optics_bench/`: `cargo build --release`,
  `cargo test --release`.
  - The tests check the 4f physics and the saving format.
  - `tests/shader.rs` validates the WGSL with naga. Run it after every shader
    change.
- On Linux, the native build needs an eframe windowing feature (`x11` or
  `wayland`); see HANDOFF.md, task 1.
- The native app can't be run without a display. Visual checks were done on
  macOS with the screenshot hook (`OPTICS_SHOT=...`, see `shot.sh`).
- The owner teaches with this. Keep physics statements in notes and UI text
  correct, and say so when something is an approximation.
- Match the surrounding code style: short doc comments and descriptive names.
  Don't commit build output (`target/`, `dist/`, `*.app`).
