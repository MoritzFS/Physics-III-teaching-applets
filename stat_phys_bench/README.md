# Stat Phys Bench

The five worked examples of the notes *Partition functions: five worked
examples* as small simulations. In each, particles or modes are in contact
with a heat bath (or isolated), and you can watch them reach the
distribution that the partition function predicts. The formulas of the notes
are drawn next to them. Written in Rust with egui.

**Open it in the browser:**
<https://moritzfs.github.io/Physics-III-teaching-applets/stat-phys-bench/>
(any current browser, WebGL 2).

```
┌ SETTINGS ───────┬ SIMULATION ─────────────────────────────────┐
│ example, k_BT,  │ levels with particles, a gas in a box,      │
│ N, …            │ masses on springs or strings                │
│                 ├ PLOTS ──────────────────────────────────────┤
│ numbers, notes  │ distribution · energy in time · ⟨E⟩, C(T)   │
└─────────────────┴─────────────────────────────────────────────┘
```

Colours everywhere: **white** = the formula from the notes, **blue** = the
simulation, **orange** = the current temperature, **grey** = the classical
result for comparison. Each bench has a menu of examples, each with notes.
A link can open one bench directly: `…/stat-phys-bench/#levels`, `#gas`,
`#atoms`, `#classical` or `#quantum`.

## Run

In the browser, follow the link above. On the desktop:

```bash
./bundle.sh            # macOS: builds "../Stat Phys Bench.app" (double-clickable)
```

or, during development:

```bash
cargo run --release
```

`cargo test --release` checks the physics: the tables of the notes, the
Schottky peak, Maxwell–Boltzmann, and that every simulation reaches the
Boltzmann distribution. The web version is built with
[Trunk](https://trunkrs.dev): `trunk serve` runs it at <http://127.0.0.1:8080>.

## Units

Energies, k_BT, the level energies, Δ and ε are all in the same unit, which
is arbitrary. Masses are in units of a reference mass and speeds in units of
√(energy/mass). The box has side 1.

## 1 · Boxes of fixed energy

Up to 20 levels and up to 1000 particles. Level i has the energy E_i and
g_i boxes (its degeneracy), drawn as g_i segments. **Drag a level up or
down** in the diagram, or type its energy and number of boxes in the list.

* **The bath.** Each particle now and then tries to hop into a random box.
  A hop down in energy is always accepted, a hop up by ΔE with probability
  e^(−ΔE/k_BT) (the Metropolis rule). That is all; the occupations then
  settle to n_i/N = g_i e^(−E_i/k_BT)/Z₁ and keep fluctuating around it.
* **Next to the levels**, on the same energy axis: the fraction of the
  particles on each level (blue) and the formula (white ticks). With *per
  box*, every box ends on one curve, e^(−E/k_BT)/Z₁.
* **Plots**: the mean energy per particle in time (with the band ±σ/√N),
  ⟨E⟩ and C against k_BT (log scale) with the simulation's values.
* **Examples**: two levels (C peaks at 0.44 k_B at k_BT ≈ 0.42 Δ), an upper
  level with three boxes, many boxes higher up, an equally spaced ladder, a
  rotating molecule (E_l = l(l + 1)B with 2l + 1 states) and random levels.

## 2 · A free particle in a box

Up to 50 hard spheres in a box, in 1, 2 or 3 dimensions (3D: drag to turn
the box). The colour is the speed.

* **The walls** are a heat bath: a particle that hits a wall leaves with a
  new random velocity at the wall's temperature. Or they are *insulating*
  and only reflect; then the energy of the gas stays fixed, and the
  temperature is the gas's own, k_BT = 2⟨E_kin⟩/d.
* **Collisions** between the particles are elastic. They share energy
  between particles, so even an isolated gas that starts with all speeds
  equal ends up with the Maxwell–Boltzmann distribution (example *All start
  with the same speed*). With radius 0 the particles pass through each
  other. In 1D each particle runs in its own lane: equal masses colliding
  head-on would only swap their velocities.
* **Plots**: the speed distribution f(v) ∝ v^(d−1) e^(−mv²/2k_BT) with the
  most probable, mean and rms speeds (or one component, v_x, a Gaussian); the
  energy distribution P(E) ∝ E^(d/2 − 1) e^(−E/k_BT); and the mean kinetic
  energy in time, around d/2 k_BT.
* With 50 particles a snapshot is noisy, so the histograms remember about a
  minute by default (*histograms: now / ~5 s / ~1 min*).

## 3 · Motion plus internal levels

The same gas, but each atom also has a ground level and an excited level Δ
(with g states), drawn orange. Z₁ = (V/λ³)(1 + g e^(−Δ/k_BT)) factorises, so
C = 3/2 k_B + the Schottky bump of example 1.

* The bath walls set the internal state too.
* **In collisions** one of the two atoms may change its level, paid for by
  the kinetic energy of the pair along the line of centres: excitation needs
  at least Δ there. Energy is conserved, and the rule obeys detailed balance,
  so an isolated gas settles to the Boltzmann distribution at its own
  temperature. Example *Hot gas, no bath*: the gas cools while its atoms
  get excited.
* **Plots**: the speed distributions of ground and excited atoms (the same
  shape: the motion ignores the internal state), the kinetic, internal and
  total energy per atom in time, and C against k_BT.

## 4 · Modes with continuous energy

Up to 10 000 modes of fixed frequency, drawn as masses on springs or as
strings in their fundamental mode (up to 200 masses or 24 strings are
drawn; all modes count in the plots). The amplitude is √E; the colour is
the energy.

* **The bath** is a weak friction plus random kicks. It changes each mode's
  amplitude slowly; the energy spreads as P(E) = e^(−E/k_BT)/k_BT with mean
  k_BT, whatever the frequency (try *pitches: a guitar's*).
* **Plots**: P(E), the energy of one mode and the mean over all M modes in
  time (the band is ±k_BT/√M: the mean becomes sharp for many modes, one
  mode never does), and ⟨E⟩ = k_BT.

## 5 · Modes with quantised energy

The same modes with E_n = n ε (step ε, n = 0, 1, 2, …). The bath adds or
removes one quantum at a time, with rates whose ratio is e^(−ε/k_BT).

* **Plots**: P(n) = (1 − e^(−ε/k_BT)) e^(−nε/k_BT), the energy in time, and
  ⟨E⟩/ε and C/k_B against k_BT/ε, quantum (white) and classical (grey).
  Cold, the modes freeze out (C → 0); hot, they become classical (C → k_B).
* **Drawing**: a mode with n quanta is drawn as a classical vibration with
  the energy n ε, so only certain amplitudes appear. A real quantum state has
  no definite amplitude; the picture is only for comparing with example 4.
* **Zero-point energy**: the notes add ε/2 to every level, E_n = (n + ½) ε.
  It is left out here. It shifts ⟨E⟩ by ε/2 and changes neither P(n) nor C.
  So hot, ⟨E⟩ → k_BT − ε/2 here, and k_BT in the notes.

## Heat capacity from the spread

Every bench also shows C *from the spread*: in equilibrium the energy of one
particle fluctuates by Var(E) = k_BT² C, which is the same second derivative
of ln Z. The simulation's spread of energies therefore measures its heat
capacity, without changing the temperature.

## Physics model and its limits

* All baths are exact in the sense that the Boltzmann distribution is
  stationary (the rules obey detailed balance). How fast the simulations get
  there is arbitrary: only the distributions are physics.
* The gas has at most 50 particles. With insulating walls (fixed total
  energy) small deviations from the Boltzmann form remain.
* In example 3 an excited atom keeps its energy until a collision or a wall
  takes it; there is no spontaneous emission.
* The oscillator baths are stepped exactly over each frame: the classical
  one is an Ornstein–Uhlenbeck process, the quantum one a birth–death
  process with rates γ n̄ (n + 1) and γ (n̄ + 1) n.

## Files

```
src/main.rs            entry points: desktop window and browser canvas
src/app.rs             the menu bar and the five benches
src/levels.rs          example 1: levels, Z₁, Metropolis bath     (+ levels_ui.rs)
src/gas.rs             examples 2 and 3: hard spheres, walls, two-level atoms (+ gas_ui.rs)
src/oscillators.rs     examples 4 and 5: classical and quantum modes (+ oscillators_ui.rs)
src/plot.rs            axes, curves and bars drawn with egui
src/stats.rs           histograms and time traces
src/rng.rs             the random number generator
shot.sh                screenshot of one example (e.g. ./shot.sh out.png quantum 1)
```
