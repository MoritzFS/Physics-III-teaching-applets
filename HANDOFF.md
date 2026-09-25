# Handoff: put Optics Bench on the web (GitHub Pages)

**Goal:** the app in `optics_bench/` runs in the browser at
`https://moritzfs.github.io/Physics-III-teaching-applets/optics-bench/`, and
the repo root has a small landing page at
`https://moritzfs.github.io/Physics-III-teaching-applets/`. Every push to
`main` rebuilds and redeploys. The macOS desktop app must keep working from
the same code.

**Status (2026-09-25):** tasks 1–7 are done and deployed from `main`. See
[section 7](#7-web-port-what-was-done) for what was verified, the decisions
made beyond this plan, and open points. Sections 1–6 are the original plan.

---

## 1. What the app is

A Rust app (eframe/egui 0.36, wgpu 30) for teaching optics. It has two
benches, switched in the menu bar.

- **Ray optics bench.** Objects on a lawn (stickman, tree, F-sign, slit lamp,
  checkerboard), lenses, apertures, central stops and glass prisms. It shows
  what a placed *eye* and a placed *screen* see.
  - Everything is traced on the GPU by a WGSL **compute shader**
    (`src/shader.wgsl`) with progressive accumulation.
  - Top-view editor with ray fans, a 3D view, and 12 examples.
  - Configurations can be saved as JSON with notes for students.
- **Fourier optics (4f) bench.** Scalar wave optics on the CPU with `rustfft`
  and `rayon`, on a background thread (`src/fourier.rs`). It shows:
  - the input field (phase and amplitude, plus the wavefront plot);
  - the Fourier plane with a filter;
  - the image;
  - an x–z propagation view.

  There are 11 examples.

`optics_bench/README.md` has the full feature and physics description.
`cargo test --release` covers the 4f physics (λf/d fringes, the 1.22 λf/D
Airy zero, the Talbot length, inverted imaging), the JSON config format, and
WGSL validation (`tests/shader.rs`).

## 2. Constraints and decisions already made

- **WebGPU only.** The renderer relies on compute shaders and a write-only
  `rgba8unorm` storage texture, which WebGL2 can't do. Browsers without
  WebGPU get a friendly message instead of the app. Desktop remains the
  fallback.
- **No threads on the web.** GitHub Pages can't send the COOP/COEP headers
  needed for `SharedArrayBuffer`, so there are no wasm threads.
  - `rayon` already falls back to the current thread on wasm (rayon-core
    1.13 does this). Keep it.
  - The `std::thread::spawn` in `fourier::Engine` has to go on wasm.
- **Build tool:** [Trunk](https://trunkrs.dev).
- **Deploy:** GitHub Actions to Pages.
- **URL path:** `/Physics-III-teaching-applets/optics-bench/`, which is also
  Trunk's `--public-url`.
- The repo is **public** and meant for students. Keep the README for
  students; put developer notes here or in `CLAUDE.md`.

## 3. Tasks

### Task 1: make the native build work on Linux (needed for CI tests)
`Cargo.toml` uses `eframe` with `default-features = false` and no windowing
backend. On Linux, winit then fails with *"The platform you're compiling for
is not supported by winit"*. Add a Linux-only feature, for example:

```toml
[target.'cfg(target_os = "linux")'.dependencies]
eframe = { version = "0.36.2", default-features = false, features = ["wgpu", "default_fonts", "persistence", "x11"] }
```

Do not add it for macOS. Check that `cargo test --release` passes on Linux.
Some system packages may be needed (`libxkbcommon-dev`, `libx11-dev`), or
`winit`'s dlopen means none are.

### Task 2: split the entry points
- Move the app into `src/lib.rs`, keeping the modules as they are.
- `src/main.rs` stays the native entry point: `eframe::run_native`,
  `env_logger`, and the `OPTICS_SHOT` screenshot mode. Mark it
  `#[cfg(not(target_arch = "wasm32"))]`.
- The web entry is `#[cfg(target_arch = "wasm32")]`. Use
  `eframe::WebRunner::new().start(canvas, web_options, …)` inside
  `wasm_bindgen_futures::spawn_local`, and `eframe::WebLogger` for logging.
  The canvas comes from `index.html`.
- In `WebOptions.wgpu_options`, ask for the **BROWSER_WEBGPU** backend only
  (`egui_wgpu::WgpuSetup::CreateNew`; see `egui-wgpu-0.36/src/setup.rs`).
- Dependencies:
  - native only: `env_logger`, `png`
  - wasm only: `wasm-bindgen-futures`, `web-sys` (features `Window`,
    `Document`, `HtmlCanvasElement`), `web-time`

### Task 3: remove the parts that only work on the desktop

| file | what | on the web |
|---|---|---|
| `fourier.rs` | `Engine` spawns a `std::thread`; `compute()` uses `std::time::Instant` | Compute synchronously in `Engine::request`/`poll` when parameters change (at most one per frame, newest wins). Use `web_time::Instant`. Default grid **256²** on the web. Expect about 3–4× native time (native: 15 ms at 512², 35 ms at 1024²). |
| `configs.rs` | `std::fs` folder next to the app | Keep saved configurations in browser storage (eframe persistence / `localStorage`). Add **Download .json** / **Open .json** (for example with `rfd`, which supports wasm, or `web-sys` Blob and file input). |
| `app.rs` → `handle_dropped_files` | uses `DroppedFile::path()` | Use `DroppedFile::bytes()` on the web |
| `app.rs` → `config_window_ui` | "Show in Finder" (`std::process::Command`) | Hide on the web |
| `app.rs` → `OpticsApp::new`, `debug_shot` | `OPTICS_*` env vars, `png`, `std::fs::File` | Native only |
| `app.rs` → menu "My configurations" | `configs::list()` reads a folder | List from browser storage on the web |

`gpu.rs`, `shader.wgsl`, `scene.rs`, `trace.rs` and `fourier_ui.rs` should
compile unchanged.

### Task 4: WebGPU details
- **Limits:** bind group 0 has 6 read-only storage buffers and bind group 1
  has one read-write storage buffer, 7 in total. The WebGPU default is 8 per
  stage. The storage textures are fine.
- **Shader:** the browser compiles WGSL with its own compiler (Tint in
  Chrome), which can be stricter than naga. Fix any errors it reports.
- **Web defaults:** keep `render_scale` at 1.0 px per point. Consider lower
  `spp` and `max_samples` for weak integrated GPUs.
- **`index.html`:** check `navigator.gpu` before starting. If it's missing,
  show "This app needs WebGPU (Chrome/Edge, Safari 26+, or Firefox with
  WebGPU). The desktop app works everywhere" instead of the canvas.

### Task 5: Trunk build
- Create `optics_bench/index.html` (full-window canvas, loading text, WebGPU
  check) and `optics_bench/Trunk.toml`.
- For size and speed, use a release profile with `lto = true` and
  `codegen-units = 1`, and let Trunk run `wasm-opt`.
- Check it with:
  `trunk build --release --public-url /Physics-III-teaching-applets/optics-bench/`

### Task 6: landing page and GitHub Actions
- Add `site/index.html` at the repo root: a small page listing the applets
  and linking to `optics-bench/`.
- Add `.github/workflows/pages.yml`. Sketch (bump action versions to current
  majors):

```yaml
name: pages
on: { push: { branches: [main] }, workflow_dispatch: {} }
permissions: { contents: read, pages: write, id-token: write }
concurrency: { group: pages, cancel-in-progress: true }
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with: { targets: wasm32-unknown-unknown }
      - name: tests (native)
        working-directory: optics_bench
        run: cargo test --release
      - uses: jetli/trunk-action@v0.5.0
      - name: build web app
        working-directory: optics_bench
        run: trunk build --release --public-url /Physics-III-teaching-applets/optics-bench/ --dist ../_site/optics-bench
      - run: cp site/index.html _site/
      - uses: actions/upload-pages-artifact@v3
        with: { path: _site }
  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment: { name: github-pages, url: "${{ steps.d.outputs.page_url }}" }
    steps:
      - { id: d, uses: actions/deploy-pages@v4 }
```

- **Pages is already turned on** (source: GitHub Actions, 2026-09-25). The
  site URL is https://moritzfs.github.io/Physics-III-teaching-applets/ and
  stays empty until the workflow above has run once.

### Task 7: document
- In `optics_bench/README.md`: add the web link and a "Web version" section
  covering browser requirements and what differs (no threads, configurations
  in the browser).
- Update the table in the root README.

## 4. Acceptance criteria
- CI is green: native tests on Linux, plus the wasm build and deploy.
- The site loads in Chrome/Edge and Safari 26+ on macOS; the owner tests on
  their Mac.
- **Ray bench:** eye and screen views render and converge, dragging from the
  toolbox works, the top view and 3D view work, and the examples load.
- **4f bench:** the examples load; changing parameters updates within about
  0.2 s at 256².
- **Configurations:** save, download, open and drop all work in the browser.
- A browser without WebGPU shows the explanation message, not a blank page.
- `cargo build --release` and `./bundle.sh` still work on macOS.

## 5. Verifying from the cloud
There is no GPU or display in the cloud session. What can be checked there:

- `cargo test --release` (Linux, after task 1).
- `cargo build --target wasm32-unknown-unknown --release` and `trunk build`.
- Optionally, headless Chromium with WebGPU flags (for example Playwright,
  `--enable-unsafe-webgpu`, SwiftShader/Vulkan) to catch start-up and WGSL
  errors in the console. This may not work in the container; don't
  over-invest.

Final visual checking is done by the owner in their browser. Say clearly in
the summary what was and wasn't verified.

## 6. Background
- The repo lives locally at `~/Documents/phd_local/Physics-III-teaching-applets`.
  The older working copy in the owner's iCloud folder
  (`…/PhD/teaching/2026_PH3/applets/optics_bench`) is stale. This repo is the
  source of truth.
- The owner prefers short, concrete explanations. Physics text in the app is
  used in teaching, so it must be correct.
- Known limitations of the app, all documented in its README:
  - Lenses are ideal thin lenses; aberrations are model terms.
  - The ray tracer has no diffraction; that's why the 4f bench exists.
  - The 4f propagation view is a 1D model of the object's centre line.

## 7. Web port: what was done

**Verified**
- In the cloud (Linux):
  - `cargo test --release` passes (10 tests).
  - The native build has no warnings, and `cargo check` / `cargo clippy`
    pass for `wasm32-unknown-unknown`.
  - `trunk build --release` succeeds. The wasm is 9.6 MB, 3.5 MB gzipped.
- In headless Chromium with a software WebGPU adapter (SwiftShader):
  - The app starts, the WGSL compiles in Tint, and the 4f bench runs.
  - Without `navigator.gpu`, or without an adapter, the WebGPU message shows.
  - Headless screenshots don't capture WebGPU canvases, so there was no
    visual check there.
- The owner ran it locally with `trunk serve` on the Mac.

**Not verified yet**
- The deployed site in Chrome and Safari.
- Download, Open .json and drop in the page.
- A 4f timing measured in the browser (native single-thread: about 30 ms at
  256² and 120 ms at 512²).
- `./bundle.sh` on macOS after the change.

**Decisions beyond the plan**
- The web entry lives in `src/main.rs` behind `cfg(target_arch = "wasm32")`,
  as in eframe's template. `src/web.rs` holds the browser helpers (local
  storage, download, file dialog, async file reads).
- `configs.rs` has one API with two stores. On the desktop the files are
  the same as before; in the browser each configuration is a local-storage
  key `optics_bench/config/<name>`.
- In the browser:
  - The app state lives under `optics_bench/state` instead of eframe's
    `app`, because all of `moritzfs.github.io` shares one local storage.
  - The state is saved every 5 s. eframe 0.36 registers its save-on-close
    listener as `"onbeforeunload"`, which never fires.
  - Rendering defaults are 1 sample per frame and at most 512 samples.
  - A panic shows a message in the page (a hook in `main.rs`, which eframe's
    own panic hook chains to).
- The save shortcut label comes from `format_shortcut` (⌘S on the Mac,
  Ctrl+S elsewhere).
- The release profile has `lto = true` and `codegen-units = 1`. CI turns
  both off for the test step only, to save time.
- The workflow also runs the tests and the build (without deploying) on
  pull requests.

**Working on the web version**
- Building for wasm needs rustup's Rust: Homebrew's `rust` cannot add the
  `wasm32-unknown-unknown` target. Then run
  `rustup target add wasm32-unknown-unknown`.
- `trunk serve` (run from `optics_bench/`) serves the app at
  http://127.0.0.1:8080. Localhost counts as secure, so WebGPU works.
- Check the wasm side with
  `cargo clippy --target wasm32-unknown-unknown`.

**Possible follow-ups**
- Speed up the 4f bench on the web with rustfft's `wasm_simd` feature and
  `-C target-feature=+simd128`. All browsers with WebGPU support wasm SIMD.
- Shrink the wasm: eframe's `wgpu` feature also compiles wgpu's WebGL
  backend, which is never used.
- If a second egui applet goes onto the same site, both would share eframe's
  `egui_memory_ron` key. Override `persist_egui_memory` on the web.
