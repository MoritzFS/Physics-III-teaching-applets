# Optics Bench

An interactive optics bench on a lawn, for teaching geometrical optics,
with a Fourier-optics (4f) bench, a dispersion bench (wave packets, thunder),
a rainbow bench, an interferometer bench (cavities, beam splitters,
polarisation) and a grating spectrometer. Written in Rust. egui draws the interface and wgpu (Metal
on macOS, WebGPU in the browser) runs the GPU ray tracer.

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

The build output goes to `target.nosync/` so that iCloud doesn't sync it.

The web version is built with [Trunk](https://trunkrs.dev): `trunk serve`
runs it locally at <http://127.0.0.1:8080>.

## Web version

<https://moritzfs.github.io/Physics-III-teaching-applets/optics-bench/> runs
the same app in the browser (WebAssembly and WebGPU).

* **Browsers**: it needs WebGPU, which Chrome and Edge, Safari 26 or newer,
  and Firefox with WebGPU have. Other browsers show a message instead. The
  desktop app works everywhere.
* **4f bench**: computing on several threads in the browser needs shared
  memory, which browsers only enable for sites that send special headers
  (COOP/COEP). GitHub Pages cannot send them, so the wave optics is computed
  on one thread, between two frames. The grid starts at 256²; 512² and
  1024² work, but the controls lag while the picture updates.
* **Dispersion, sound and rainbow benches** also compute on one thread, in
  the frame after a change. Changing the light or the drop of the rainbow
  bench takes a moment; white light uses 10 nm steps instead of 5 nm. Sound
  plays through the browser's Web Audio, after you have clicked on the page.
* **Configurations** are kept in this browser's storage for the site, not in
  files. **Download** saves one as a `.json` file, to keep it or to share it.
  **Open .json…** opens one, and so does dropping it onto the page. The files
  are the same as those of the desktop app. Clearing the site data in the
  browser deletes the saved configurations, and Safari deletes them by
  itself when the site has not been used for about a week of browsing, so
  download the ones you want to keep.
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
  double-click for the Descartes ray, or press **▶ sweep** to move it from
  the centre to the edge and back. At the exit, the dashed lines are the
  directions of the Descartes rays and the red wedge beyond them is where no
  ray of this order goes; the readout shows the largest δ reached so far. "Rays to your eye" shows the rays that
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

## Interferometers

Switch with **Interferometers** in the menu bar. An optical table with a square
grid: a laser, mirrors, beam splitters, polarisation optics and detectors, to
build cavities and interferometers.

```
┌ OPTICAL TABLE ───────────────────────────┬ SWEEP ─────────────────────┐
│ parts on the squares, beams between      │ detector power vs laser    │
│ them (width and brightness = power)      │ frequency (or a setting)   │
│                                          ├ SWITCH-ON ─────────────────┤
│                                          │ power vs time, phasor      │
├ TOOLBOX ──────┬ SELECTED ─────┬ LASER ───┴─────┬ SWEEP AND VIEW ──────┤
```

* **Parts**: laser (one per table), mirror, beam splitter and cavity mirror
  (all the same part: any reflectivity R, a loss, and a shift along its normal,
  as by a piezo), PBS (passes H, reflects V), λ/2 and λ/4 plates (any
  retardance and axis), polariser, phase shifter, Faraday rotator, detector
  and beam block. Pick a part in the toolbox and click a square. Drag parts to
  move them; right-click or R turns them (mirrors in 45° steps), Delete
  removes them, the arrow keys move them by a square.
* **Table**: the width and brightness of a beam show its power relative to the
  laser. Inside a resonant cavity the beam is stronger than the laser
  (labelled "× laser power"), the more so the better the mirrors. Beams going
  both ways are drawn side by side. Hover a beam for its power and
  polarisation. The circles show the polarisation (looking into the beam,
  H = in the plane of the table) at the laser and wherever it has changed.
* **Sweep**: the detector powers, and the light coming back into the laser, as
  the laser frequency is swept, or as one part's setting is scanned (a
  mirror's position, a phase, a waveplate or polariser angle, a Faraday
  rotation). The peak spacing and width are measured on D1. The readout gives
  the round trip of the cavity on the table, FSR = c/L and the finesse
  π√ρ/(1 − ρ), where ρ is the field left after one round trip; green ticks mark
  the predicted resonances. "in FSR" puts the frequency axis in units of the
  FSR. Drag the yellow line (or press **▶ sweep**) to tune the laser through a
  resonance and watch the table.
* **Switch-on**: the laser starts at t = 0 and the light moves one square per
  time step (83 ps for 25 mm squares). Watch it fill the setup and the field in
  a cavity grow round trip by round trip; "laser off" shows the ring-down. The
  plot shows the detector powers against time (dashed: the steady state) and
  the field at D1 in the complex plane: a chain of phasors, one per round trip.
* **Laser**: wavelength (it sets the colour and the scale of mirror shifts),
  power, detuning Δν, polarisation, and the spectrum: one line, two lines or
  a broad band. Different lines do not interfere; their powers add.
* **Examples** (Scene → Examples: interferometers): Fabry–Pérot resonances and
  FSR, better mirrors (finesse and build-up), switch-on, unequal mirrors
  (impedance matching), ring cavity, measuring the reflection with PBS and
  λ/4, coupled cavities, a waveplate in a cavity (two polarisation modes),
  Michelson (λ/2 per fringe), Michelson with unequal arms (frequency
  dependence), Michelson with two lines (resolving them), Mach–Zehnder,
  Sagnac, λ/2 plate and PBS, PS03 exercise 13 (polariser and λ/4 plate in
  both orders), optical isolator. Circular polarisation is named as in PS03:
  (1, i)/√2 is left circular, turning counter-clockwise seen looking into the
  beam (↺).

The fields are Jones vectors (H in the plane of the table, V vertical). Each
part maps the fields arriving on its four sides to the fields leaving them.
Mirrors reflect H with +r and V with −r, as an ideal metal mirror does, and
transmit with i·t; this keeps every beam splitter lossless. Real dielectric
coatings have other phases, different for s and p. At the reference frequency
every square holds a whole number of wavelengths, so a detuning Δν gives the
phase 2πΔνL/c over a length L. The steady state is the exact solution of the
linear system of all beams: every round trip of every cavity is included. The
beams are plane waves: there are no transverse modes, no divergence and no
misalignment, so every cavity is perfectly aligned and mode-matched. The
switch-on uses the main line only and treats mirror shifts as phases. Unit
tests check the Airy transmission, FSR and finesse, the build-up T/(1 − R)²,
energy conservation, the complementary outputs of the Michelson and
Mach–Zehnder, the dark port of the Sagnac, PBS + λ/4, the isolator, the
splitting of the polarisation modes, and that the switch-on settles to the
steady state.

## Gratings

Switch with **Gratings** in the menu bar. A grating spectrometer as in the
lecture notes: a source and a reference lamp are combined at a beam splitter
and focused onto the entrance slit; a lens (f₁) makes the light parallel, an
iris sets the lit width W of a reflection grating, and a second lens (f₂)
focuses every direction onto a line camera.

```
┌ SPECTROMETER (seen from above) ─────┬ RESOLUTION ──────────────────────┐
│ lamps, slit, lenses, grating, the   │ N, θ_i, θ_m, Δ = Nd(sin θ_m −    │
│ orders, the camera                  │ sin θ_i), λ/δλ = mN; phasors      │
├ SPECTRUM (on the camera | in all directions) ──────────┬ CALIBRATION ─┤
├ LAMPS ────────┬ GRATING ───────┬ SPECTROMETER ──────────┬ VIEW ───────┤
```

* **Spectrometer**: the setup with its real angles. Every line of the lamps
  leaves the grating in all its orders; the beams that reach the camera lens
  are focused onto the camera, whose colour strip shows what it records. Drag
  the grating to turn it (shift: finely), or press **▶ turn**.
* **Resolution**: the number of lit grooves N = W/d, the angles θ_i and θ_m
  (from the grating normal, d(sin θ_m − sin θ_i) = mλ as in the notes), the
  path difference across the lit grating Δ = Nd(sin θ_m − sin θ_i) = mNλ, the
  resolving power λ/δλ = mN = Δ/λ, and its limit λ/δλ ≤ 2W/λ, because Δ can
  never exceed 2Nd. It also gives the limits set by the slit and the pixels,
  and which of the three limits the current setting. The drawings show the
  extra path of the outermost rays (blue), and the phasors of the N grooves at
  the maximum of λ: for λ + δλ every groove adds 2π·mδλ/λ, the chain curls up,
  and at δλ = λ/(mN) it closes into a circle (Rayleigh's criterion).
* **Spectrum on the camera**: what the camera records with the source lamps
  (blue) and with the reference lamps (orange). The wavelength axis comes from
  the calibration (a polynomial λ(pixel) through the reference lines it finds,
  with a table of the residuals), from the grating equation, or is in pixels.
  Lines of a reference lamp in another order are listed but not used. Scroll
  to zoom, drag to move, double-click for the whole camera.
* **Spectrum in all directions**: the intensity against the path difference
  between neighbouring grooves, d(sin θ_m − sin θ_i), so the orders of λ sit
  at mλ, with N − 2 weak maxima between them; or against the angle θ_m. It can
  show the same grating with one and two grooves for comparison, each
  normalised to its 0th order as in PS03, exercise 11. The green band is what
  the camera sees.
* **Lamps**: mercury, neon, sodium, hydrogen (Balmer), helium, cadmium, HeNe
  and green lasers, a white lamp (2900 K), and a test pair of lines with any
  separation. Each can be a source or a reference. The wavelengths are the
  standard values in air; the relative strengths are rough, as they differ
  from lamp to lamp.
* **Grating**: lines per mm, lit width W, blazed (sawtooth facets at the blaze
  angle) or flat reflecting strips (any width), and the angle it is turned by
  (or the angle of incidence θ_i).
  **Spectrometer**: slit width, f₁ and f₂, the angle between the arms
  (0 = Littrow), the camera's pixels. **Calibration**: the degree of the fit.
* **Examples** (Scene → Examples: grating spectrometer): how it works,
  PS03 exercise 11 (one slit, two slits and 20 slits against the angle, and
  oblique incidence), calibrating with reference lamps, the sodium doublet (how many grooves),
  second order, Rayleigh's criterion and the phasors, slit and pixels, the
  limit δλ/λ ≥ λ/2Nd, a few grooves (the N-slit pattern), white light and
  the blaze, ghost lines from overlapping orders.

The light of a line is the Fraunhofer pattern of N evenly lit grooves,
sin²(Nφ/2)/sin²(φ/2) with φ = 2πd(sin θ_m − sin θ_i)/λ, times the pattern of a
single groove, a tilted facet or a strip of width b:
sinc²(πb(sin α′ + sin β′)/λ), with the angles measured from its own normal.
This is scalar theory without polarisation or shadowing, so the efficiencies
of the orders are approximate. The lenses are ideal. The camera position is
x = f₂ tan(β − β_c), so λ(pixel) is not linear; the pattern is smeared by the
image of the slit (its width times f₂/f₁ · cos θ_i/cos θ_m) and summed over
each pixel. Unit tests check the grating equation, that a line keeps its
power on the camera, that the sodium doublet needs about 990 grooves in the
first order and is resolved with 600 in the second, the 81 % (8/π²) dip of
two lines at Rayleigh's distance, the calibration residuals, the
second-order ghosts and the N − 2 weak maxima.

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
| `src/dispersion.rs` | pulse propagation in a dispersive medium, media, dispersion examples |
| `src/dispersion_ui.rs` | user interface of the dispersion bench (wave packets) |
| `src/sound.rs` | thunder and whistler synthesis, air absorption (ISO 9613-1) |
| `src/sound_ui.rs` | user interface of the sound tab |
| `src/audio.rs` | sound output (cpal) |
| `src/rainbow.rs` | rays in a drop, Fresnel weights, sky brightness, rainbow examples |
| `src/rainbow_ui.rs` | user interface of the rainbow bench |
| `src/interferometer.rs` | the optical table: Jones matrices of the parts, steady state, sweeps, switch-on in time, interferometer examples |
| `src/interferometer_ui.rs` | user interface of the interferometer bench |
| `src/grating.rs` | grating spectrometer: lamps, line shapes on the camera, far field, calibration, grating examples |
| `src/grating_ui.rs` | user interface of the grating bench |
| `src/worker.rs` | background thread for the newest request (computes in place in the browser) |
| `src/main.rs`, `src/lib.rs` | start-up on the desktop and in the browser |
| `src/web.rs` | browser helpers: storage, downloading and opening files |
| `index.html`, `Trunk.toml` | the web page and its build |
| `shot.sh` | saves a screenshot of an example or a saved configuration, e.g. for slides (`d3` = dispersion example 3, `r0` = rainbow example 0, `i2` = interferometer example 2, `g4` = grating example 4) |
