# Optics Bench

An interactive optics bench on a lawn, for teaching geometrical optics,
with a Fourier-optics (4f) bench and a dispersion bench (wave packets, thunder).
Written in Rust. egui draws the interface and wgpu (Metal on macOS) runs the
GPU ray tracer.

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

The build output goes to `target.nosync/` so that iCloud doesn't sync it.

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
takes about 15 ms at 512² and 35 ms at 1024² on a background thread.

## Dispersion

Switch with **Dispersion** in the menu bar. It has two tabs.

### Wave packets

A pulse travels through a medium with refractive index n(ω). Everything is
computed exactly (1D, linear, scalar): the pulse is the signal s(t) arriving at
the entrance x = 0, and every frequency travels with its own k(ω) = n(ω) ω/c:

```
ψ(x, t) = Σ_ω S(ω) exp(i (k(ω) x − ω t)),      u = Re ψ
```

In front of the medium (x < 0, optional) is vacuum; reflections at the entrance
are left out. A complex n(ω) also gives absorption. Units: the period T₀ and the
vacuum wavelength λ₀ of the reference frequency ω₀ (c = 1).

* **Wave** (top): u(x, t) with its envelope |ψ|. Optional colour = local
  wavelength (red: longer than at the carrier), a green triangle moving with
  v_g and an orange dot riding on a crest of the carrier (v_p). Picked
  frequency components are drawn in lanes underneath, each a plane wave with
  its own phase velocity. Play/pause with the space bar; scroll to zoom, drag
  to pan, double-click to see everything. Drag the yellow line to move the
  observer.
* **Space–time** (middle): |ψ(x, t)| (or u) in the x–t plane, time running
  down, with the v_g and v_p lines. Click to jump in time. On the right, the
  signal u(t) at the observer; **listen** plays it as sound (the whole run in
  about 2.5 s) while the time line follows.
* **Dispersion relation** (bottom left): n(ω) with the group index
  n_g = c dk/dω and the absorption Im n, or ω(k) with the secant (v_p) and the
  tangent (v_g) at the carrier, or v_p(ω) and v_g(ω). The spectrum of the pulse
  is shaded underneath. Drag the white points to change the medium; click to
  pick (or remove) frequency components.
* **Pulse**: Gaussian (duration, carrier, chirp; buttons for a few-cycle pulse
  and a long, nearly monochromatic one), delta (all frequencies up to a
  limit), a switched-on/off wave, or only the picked frequencies (beats).
* **Medium**: no dispersion; Taylor expansion (phase index, group index, GVD,
  TOD, each set independently); glass (Cauchy); a resonance (Lorentz
  oscillator, with absorption); a cutoff (plasma, waveguide); a power law
  ω ~ k^m (deep water m = ½, capillary ripples 3/2, matter waves 2); or a free
  form drawn by hand. "Make it linear" replaces the medium by a constant n.
* **Examples** (Scene → Examples: dispersion): packet without dispersion,
  v_p ≠ v_g, spreading and chirp, a long packet, delta pulse through glass,
  two-frequency beats, chirped pulse compression, near a resonance, cutoff,
  deep-water waves, capillary ripples, matter wave, draw your own n(ω).

Unit tests check that without dispersion the pulse keeps its shape, that the
envelope moves with dω/dk, the Gaussian broadening σ√(1 + (L/L_D)²), the
position and depth of chirped-pulse compression, v_g/v_p for the power laws,
v_p·v_g = c² for the waveguide, and that nothing wraps around in time.

### Sound: thunder and whistlers

* **Thunder**: the lightning channel is a random zig-zag (about 10 km with
  branches, tortuous down to the metre scale). Every piece sends out the same
  6 ms N-wave at the moment of the flash; pieces seen side-on arrive together
  (claps), pieces seen end-on are smeared out (rumble). On the way the sound
  falls off as 1/r and is absorbed by air as in ISO 9613-1 (20 °C, 70 %
  humidity: ~0.2 dB/km at 100 Hz, 5 dB/km at 1 kHz, 23 dB/km at 4 kHz). Close
  by the thunder starts with a sudden, bright crack; far away the high
  frequencies are gone and it is a soft, low rumble. Drag yourself in the side
  view or use the distance buttons; the spectrogram shows what you hear, with
  the arrival of the nearest and farthest part of the channel. The same
  relaxation processes of O₂ and N₂ that absorb also make sound dispersive,
  but only by ~0.02 m/s: over 10 km that is a few milliseconds. The
  dispersion can be switched on and exaggerated (×1 … ×5000) to hear what it
  would do (each clap becomes a falling chirp). So the honest answer to "why
  is far thunder a rumble" is absorption plus the long channel, not
  dispersion.
* **Whistler**: the radio click of lightning, dispersed on its way along a
  magnetic field line through the magnetosphere (whistler mode, ω ~ k²):
  group delay t(f) = D/√f, optionally with a nose frequency, echoes between
  the hemispheres and background crackle (sferics). This is real dispersion
  that you can hear with a VLF receiver.

To save the sounds as WAV files (with a PNG of waveform and spectrogram):

```bash
DUMP_DIR=/some/folder cargo test --release -- sound::debug --ignored
```

## Rainbow

Switch with **Rainbow** in the menu bar. Sunlight in spherical drops, as in
exercise 8 of PS02. A ray hits a drop at the height b (radius 1), so
sin θ = b and inside sin θ = n sin φ. After k internal reflections it leaves
turned by D = 2(θ − φ) + k(π − 2φ), and you see it at the angle δ from the
antisolar point (for k = 1: δ = 4φ − 2θ). Where δ is extreme (the Descartes ray,
cos²θ = (n² − 1)/((k+1)² − 1)) the rays pile up: that is the bow.

```
┌ DROP ─────────────┬ DEVIATION δ(θ) ──────┬ OBSERVER (side view) ─┐
├ SKY ──────────────┴──────────────────────┴───────────────────────┤
├ LIGHT ──────┬ DROP ──────┬ LIGHT PATHS ───────┬ VIEW ────────────┤
```

* **Drop**: the picked ray split into its colours, a fan of rays evenly spaced
  in b, the light lost at each surface (with its share of the power) and the
  angles θ, φ, δ as in figure 8 of the exercise. Drag to move the ray,
  double-click for the Descartes ray. "Rays to your eye" shows the rays that
  leave at the picked angle instead: two per colour below the bow, none above.
* **Deviation δ(θ)** for every colour and the chosen orders, with the maxima
  (42.4° / 40.5° for the exercise's red and blue). The shaded band is the
  spread of the colours: zero for the central ray, 1.9° at the Descartes ray,
  2.5° for grazing rays. On the right, the colour and brightness of the sky at
  each δ. The yellow line is the picked angle (drag it); its dots are the rays
  that leave at that angle. "against b" plots against b = sin θ.
* **Observer**: you, the sunlight from behind, the rain, the directions of the
  bows, and the picked drop with its cone of light. Drag the drop to pick an
  angle. The swatch shows what reaches your eye from there.
* **Sky**: a panorama (azimuth and elevation linear) away from the sun or
  towards it. Click to pick an angle; all drops on the dashed circle look the
  same.
* **Light**: 405 + 707 nm (the exercise), one wavelength, 8 lines or white
  sunlight; the sun's diameter (0 = a point as in the exercise, 0.53° real);
  its elevation; a polariser.
* **Drop**: water at 10 °C (Cauchy fit through the exercise's
  n(707 nm) = 1.331 and n(405 nm) = 1.344), sea water, glass beads
  (n = 1.52, 1.90) or any n; dispersion ×0 … ×10; Fresnel losses on/off.
* **Light paths**: reflection off the surface and k = 0 … 4 internal
  reflections, each with its share of the light that hits a drop (88 %, 4.1 %,
  0.6 %, 0.2 %, 0.1 % for k = 0 … 3, 6.6 % reflected).
* **Examples** (Scene → Examples: rainbow): exercise 8, why a bow (caustic),
  dispersion as a function of the angle, which drop sends which colour,
  secondary bow and Alexander's dark band, the real sun, polarisation, sun too
  high, sea spray, glass beads and retroreflectors, 3rd and 4th order towards
  the sun, two rays in one direction (outlook to interference).

The brightness is the power per solid angle: every path carries the Fresnel
factors (1 − R)² Rᵏ for s and p, the drop is filled evenly (weight 2b db), and
the result is divided by sin δ and convolved with the sun's disk. White light
uses the CIE colour matching functions. It is ray optics: supernumerary bows
and fogbows (Airy theory) are not included. Unit tests check 42.37° / 40.51°
and θ_max from the exercise, the Descartes formula against a brute-force
search, energy conservation over all orders, the ~92 % polarisation, and that
the sky is brighter inside the bow and dark in Alexander's band.

## Saving configurations

**Scene → Save / manage configurations…** (or Cmd+S) saves the current scene,
including the camera views, the 4f bench settings, which bench is open, and a
free-text note. The note is shown in the
inspector when the scene is opened, so it can hold a task for the students.

Configurations are plain `.json` files in the folder `Optics Bench configs`
next to the app. Copy them to share them. You can open them from
**Scene → My configurations**, or by dropping a file onto the window.

The current scene and settings are also kept automatically when the app quits.

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
| `src/configs.rs` | saving and loading configurations (JSON) |
| `src/fourier.rs` | wave optics of the 4f system (FFT, angular spectrum), 4f examples |
| `src/fourier_ui.rs` | user interface of the 4f bench |
| `src/dispersion.rs` | pulse propagation in a dispersive medium, media, dispersion examples |
| `src/dispersion_ui.rs` | user interface of the dispersion bench (wave packets) |
| `src/sound.rs` | thunder and whistler synthesis, air absorption (ISO 9613-1) |
| `src/sound_ui.rs` | user interface of the sound tab |
| `src/audio.rs` | sound output (cpal) |
| `src/rainbow.rs` | rays in a drop, Fresnel weights, sky brightness, rainbow examples |
| `src/rainbow_ui.rs` | user interface of the rainbow bench |
| `src/worker.rs` | background thread for the newest request |
| `shot.sh` | saves a screenshot of an example or a saved configuration, e.g. for slides (`d3` = dispersion example 3, `r0` = rainbow example 0) |
