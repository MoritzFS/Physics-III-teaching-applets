# Optics Bench

An interactive optics bench on a lawn, for teaching geometrical optics.
Written in Rust. egui draws the interface and wgpu (Metal on macOS, WebGPU in
the browser) runs the GPU ray tracer.

**Open it in the browser:**
<https://moritzfs.github.io/Physics-III-teaching-applets/optics-bench/>
(needs WebGPU, see [Web version](#web-version)).

```
┌ EYE ───────────────────────┬ SCREEN ────────────────────┐
│ what the placed eye sees   │ image formed on the screen │
├ BENCH (top view | 3D view) ┴────────────────────────────┤
│ drag objects, lenses, screen and eye around the lawn    │
├ TOOLBOX ─────────────────────┬ INSPECTOR ───────────────┤
│ stickman, tree, F-sign, slit │ focal length, diameter,  │
│ lamp, checkerboard, ± lens,  │ hole size, glass, …      │
│ aperture, central stop,      │ + notes about the scene  │
│ prism, screen, eye           │                          │
└──────────────────────────────┴──────────────────────────┘
```

## Run

```bash
./bundle.sh            # builds "../Optics Bench.app" (double-clickable)
```

or, during development:

```bash
cargo run --release
```

`bundle.sh` puts the app bundle next to this folder, in the repository root. It is git-ignored.

The web version is built with [Trunk](https://trunkrs.dev): `trunk serve`
runs it locally at <http://127.0.0.1:8080>.

## Web version

<https://moritzfs.github.io/Physics-III-teaching-applets/optics-bench/> runs
the same app in the browser (WebAssembly and WebGPU).

* **Browsers**: it needs WebGPU, which Chrome and Edge, Safari 26 or newer,
  and Firefox with WebGPU have. Other browsers show a message instead. The
  desktop app works everywhere.
* **4f bench**: browsers only allow threads on sites that send special
  headers, which GitHub Pages cannot. The wave optics is therefore computed
  on one thread, between two frames. The grid starts at 256²; 512² and
  1024² work, but the controls lag while the picture updates.
* **Configurations** are kept in this browser's storage for the site, not in
  files. **Download** saves one as a `.json` file, to keep it or to share it.
  **Open .json…** opens one, and so does dropping it onto the page. The files
  are the same as those of the desktop app. Clearing the site data in the
  browser deletes the saved configurations.
* **Rendering** starts with 1 sample per frame and at most 512 samples, to be
  gentle on laptop GPUs. Both can be raised in the Rendering menu.

## Using it

* **Toolbox**: drag an item onto the bench, or click it. There is one screen
  and one eye; dropping another one moves the existing one.
* **Top view**: drag to move (it snaps to 1 cm and to the line of other
  elements). Drag the yellow handle to rotate (5° steps). Shift turns snapping
  off. Scroll to zoom and drag the lawn to pan. Delete or Backspace removes
  the selection, and the arrow keys nudge it.
* **Rays** (top view): red and blue fans start at the left and right edge of
  the object in front of each lens. Select an object to show only its rays.
  "virtual rays" extends the outgoing rays backwards as dashed lines, which
  shows where virtual images are. F and 2F are marked on every lens axis.
* **3D view**: drag things to move them. Drag the lawn to orbit, right-drag to
  pan, scroll to zoom.
* **Eye view**: drag to look around and scroll to change the field of view.
  The status line says what the eye looks at, the vergence of the light
  arriving at the eye, and whether the eye can accommodate to it.
* **Scene → Examples**, each with a short explanation in the inspector:
  * real image on a screen
  * magnifying glass
  * aperture and depth of field
  * central stop and ring-shaped blur
  * pinhole camera
  * Kepler and Galilei telescopes: each has one row ending in an eye and one
    projecting onto a screen
  * diverging lens
  * short-sighted eye with glasses
  * spherical and chromatic aberration
  * looking through a prism
  * prism spectrometer
* **Screen options**:
  * "light-tight tube": only light that passed all the optics in front of the
    screen reaches it, as in a camera or telescope tube.
  * "view from behind": shows the picture as seen through a ground-glass
    screen.

## Fourier optics (4f bench)

Switch with **Fourier optics (4f)** in the menu bar. This is a separate
scalar wave-optics simulation of

```
object ──f₁── L1 ──f₁── Fourier plane (filter) ──f₂── L2 ──f₂── image
```

* **Input field**: the complex field right behind the object. Colour is the
  phase and brightness the amplitude, with a colour wheel as legend. Below it,
  the *wavefront* along the centre line: unwrapped phase in wavelengths, plus
  amplitude.
* **Fourier plane**: |U|² in the back focal plane of L1, i.e. the Fraunhofer
  pattern, optionally on a log scale. The filter is drawn on top; drag inside
  the pattern to resize it. Hovering shows position, spatial frequency and
  angle.
* **Image plane**: |U|² behind L2, inverted and magnified by −f₂/f₁. All three
  panels have a profile along the centre line underneath. Scroll in a panel to
  zoom, double-click to reset.
* **Propagation**: the intensity in the x–z plane through the whole system,
  computed by angular-spectrum propagation of the object's centre line. It
  shows Fresnel diffraction and Talbot carpets, the focus in the Fourier plane
  and the image. Drag the yellow observation plane; its profile is shown on
  the right.
* **Light**: wavelength; plane, Gaussian or top-hat beam; tilt (a linear phase
  ramp); a point source at finite distance (a curved wavefront).
* **Objects**: slits, double slit, holes, binary and sinusoidal gratings, mesh,
  letter F (also behind a mesh), checkerboard, phase grating, transparent
  phase F, spiral phase plate.
* **Filters**: low pass, high pass (central stop), band pass, slit, knife edge
  (schlieren), Zernike phase dot, spiral phase.
* **Examples** (Scene → Examples: Fourier optics):
  * double slit
  * Airy disk
  * Talbot carpet
  * Abbe–Porter
  * low pass
  * high pass (edge enhancement)
  * Zernike phase contrast
  * schlieren
  * spiral phase filter
  * tilted and curved wavefronts
  * optical vortex

The 2D planes are exact discrete Fourier transforms (paraxial, scalar). Unit
tests check the fringe spacing λf/d, the first Airy zero 1.22 λf/D, the Talbot
length 2p²/λ, and that the unfiltered image is the inverted object. One update
takes about 15 ms at 512² and 35 ms at 1024² on a background thread (desktop
app; the web version is slower, see [Web version](#web-version)).

## Saving configurations

**Scene → Save / manage configurations…** (or Cmd+S; Ctrl+S on Windows and
Linux) saves the current scene,
including the camera views, the 4f bench settings, which bench is open, and a
free-text note. The note is shown in the
inspector when the scene is opened, so it can hold a task for the students.

Configurations are plain `.json` files in the folder `Optics Bench configs`
next to the app. Copy them to share them. You can open them from
**Scene → My configurations**, or by dropping a file onto the window. In the
browser they are stored in the browser instead; see
[Web version](#web-version).

The current scene and settings are also kept automatically when the app quits
(in the browser: every few seconds).

## Physics model

* **Apertures** are opaque plates with a round hole. A small hole in front of
  a lens gives a larger f-number (N = f/D, shown in the inspector) and more
  depth of field. A hole alone is a pinhole camera.
* **Central stops** ("anti-apertures") are opaque disks that block the centre
  of a beam. In ray optics this makes out-of-focus points into rings (as with
  the secondary mirror of a reflecting telescope) and gives dark-field /
  schlieren effects. It does not filter spatial frequencies: that needs
  diffraction, which a ray tracer does not model.
* **Prisms** are real glass bodies: Snell's law at every face, including total
  internal reflection. The index follows Cauchy's formula
  *n(λ) = A + B/λ²*, set by n_d and the Abbe number V. There are presets for
  crown glass, flint glass and "exaggerated".
* **Colour**: when a prism or a lens with chromatic aberration is on the
  bench, every sample gets a random wavelength (400–700 nm). Only paths that
  actually pass dispersive glass are treated spectrally, using the CIE colour
  matching functions. Everything else stays plain RGB, so white stays white
  and there is no extra colour noise.
* **Lenses** are ideal thin lenses in slope form: a ray hitting the lens at
  height *h* leaves with slope *t′ = t − h·P*. This images planes perfectly.
  Two optional terms can be added:
  * spherical aberration: *P(h) = P (1 + k h²/R²)*
  * chromatic aberration: *P(λ) = P (1 + c·s(λ))*, where *s* = 0 at 588 nm
    and 1 at 486 nm, following the same Cauchy shape as glass.
* **Screen**: each pixel is a point on a diffuse screen. It collects the light
  that arrives through the lens, aperture or prism in front of it, as in a
  dark room. The irradiance estimate includes the cos·cos/r² factors, so
  vignetting is real.
  * Only the part of the opening that light actually passes is sampled. The
    CPU probes this whenever the scene changes, for example when a small
    aperture or a telescope objective limits the bundle.
  * Brightness is normalised to that part, like automatic exposure.
* **Eye**: reduced-eye model with a thin lens and the retina 17 mm behind it.
  The pupil is sampled, so defocus blur is physical. The eye's power is
  *1/17 mm + A − Rx*: *A* is accommodation, limited to 0…A_max, and *Rx* is
  the prescription the eye would need (negative = short-sighted). Auto focus
  traces the line of sight through all lenses and propagates the vergence
  paraxially.
* **Rendering**: progressive Monte-Carlo accumulation on the GPU, with sun,
  sky, soft shadows and ambient occlusion. The grass is 3D shell-traced
  blades. The image converges in about a second and the GPU goes idle once
  it has reached the maximum number of samples.

Objects are built to be asymmetric so you can see how images are flipped.
The stickman's left arm and leg are red, his right ones blue, and his left
hand waves. The F-sign is unreadable when mirrored, and the apples on the
tree are mostly on one side.

## Files

| file | content |
|---|---|
| `src/shader.wgsl` | GPU ray tracer (screen / eye / overview camera) |
| `src/gpu.rs` | wgpu compute pipeline, accumulation buffers |
| `src/scene.rs` | scene model, 3D objects, example presets |
| `src/trace.rs` | CPU tracer: auto focus, ray fans, picking |
| `src/app.rs` | user interface |
| `src/configs.rs` | saving and loading configurations (JSON files, or browser storage) |
| `src/fourier.rs` | wave optics of the 4f system (FFT, angular spectrum), 4f examples |
| `src/fourier_ui.rs` | user interface of the 4f bench |
| `src/main.rs`, `src/lib.rs` | start-up on the desktop and in the browser |
| `src/web.rs` | browser helpers: storage, downloading and opening files |
| `index.html`, `Trunk.toml` | the web page and its build |
| `shot.sh` | saves a screenshot of an example or a saved configuration, e.g. for slides |
