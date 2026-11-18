//! Split-step Fourier propagator for H = p²/2 + V(x) (ħ = m = 1).
//!
//! One step of length dt is the symmetric (Strang) splitting
//!
//! ```text
//! ψ ← e^(−iV dt/2) · F⁻¹ e^(−ip² dt/2) F · e^(−iV dt/2) ψ        error O(dt³) per step
//! ```
//!
//! V may be complex: V = V0 − iΓ/2 makes the norm decay as e^(−Γt) (PS09 36).
//! An optional absorbing layer (a smooth negative imaginary potential near both
//! ends of the periodic grid) swallows outgoing waves instead of letting them
//! wrap around.

use super::{C64, Grid1D, Wave1D, fft};

/// Absorbing boundary layer: −i·strength·(d/width)² inside the last `width`
/// of the grid at each end (d = depth into the layer).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Absorber {
    pub width: f64,
    pub strength: f64,
}

impl Absorber {
    /// A layer of the given width. `strength` of about the kinetic energy p²/2
    /// of the waves to absorb (1–10 typically) works well; too strong reflects.
    pub fn new(width: f64, strength: f64) -> Self {
        Absorber { width, strength }
    }

    /// Γ(x)/2 ≥ 0 of the layer at x (the imaginary potential is −i times this).
    pub fn profile(&self, grid: &Grid1D, x: f64) -> f64 {
        let d = (grid.x_min + self.width - x)
            .max(x - (grid.x_max() - grid.dx - self.width))
            .max(0.0);
        let s = (d / self.width).min(1.0);
        self.strength * s * s
    }
}

/// Split-step propagator. Holds V, the absorber and cached phase factors.
#[derive(Clone, Debug)]
pub struct SplitStep {
    grid: Grid1D,
    /// potential without the absorber
    v: Vec<C64>,
    absorber: Option<Absorber>,
    /// largest time step used by [`Self::advance`]
    dt_max: f64,
    /// dt for which the caches below were computed (NaN = invalid)
    cached_dt: f64,
    half_v: Vec<C64>,
    kinetic: Vec<C64>,
}

impl SplitStep {
    /// Free propagation (V = 0) with steps of at most `dt_max`.
    pub fn new(grid: Grid1D, dt_max: f64) -> Self {
        SplitStep {
            grid,
            v: vec![C64::new(0.0, 0.0); grid.n],
            absorber: None,
            dt_max: dt_max.max(1e-6),
            cached_dt: f64::NAN,
            half_v: vec![],
            kinetic: vec![],
        }
    }

    /// Builder: real potential sampled on the grid.
    pub fn with_potential(mut self, v: &[f64]) -> Self {
        self.set_potential(v);
        self
    }

    /// Builder: absorbing layer at both ends.
    pub fn with_absorber(mut self, absorber: Absorber) -> Self {
        self.set_absorber(Some(absorber));
        self
    }

    /// Replaces V by a real potential (length `grid.n`).
    pub fn set_potential(&mut self, v: &[f64]) {
        assert_eq!(v.len(), self.grid.n, "SplitStep: potential has the wrong length");
        self.v = v.iter().map(|&v| C64::new(v, 0.0)).collect();
        self.cached_dt = f64::NAN;
    }

    /// Replaces V by a complex potential, e.g. V0 − iΓ/2.
    pub fn set_complex_potential(&mut self, v: &[C64]) {
        assert_eq!(v.len(), self.grid.n, "SplitStep: potential has the wrong length");
        self.v = v.to_vec();
        self.cached_dt = f64::NAN;
    }

    /// Sets or removes the absorbing layer.
    pub fn set_absorber(&mut self, absorber: Option<Absorber>) {
        self.absorber = absorber;
        self.cached_dt = f64::NAN;
    }

    pub fn absorber(&self) -> Option<Absorber> {
        self.absorber
    }

    /// Changes the largest step of [`Self::advance`].
    pub fn set_dt_max(&mut self, dt_max: f64) {
        self.dt_max = dt_max.max(1e-6);
    }

    pub fn dt_max(&self) -> f64 {
        self.dt_max
    }

    pub fn grid(&self) -> Grid1D {
        self.grid
    }

    /// The potential (without the absorber).
    pub fn potential(&self) -> &[C64] {
        &self.v
    }

    /// The total potential including the absorber, V(x) − iΓ_abs(x)/2.
    pub fn total_potential(&self) -> Vec<C64> {
        let g = self.grid;
        (0..g.n)
            .map(|i| match self.absorber {
                Some(a) => self.v[i] - C64::new(0.0, a.profile(&g, g.x(i))),
                None => self.v[i],
            })
            .collect()
    }

    fn prepare(&mut self, dt: f64) {
        if self.cached_dt == dt {
            return;
        }
        let g = self.grid;
        let total = self.total_potential();
        // e^(−i V dt/2) for complex V: |factor| = e^(Im V · dt/2) ≤ 1 when Im V ≤ 0
        self.half_v = total.iter().map(|v| (C64::new(0.0, -0.5 * dt) * v).exp()).collect();
        let mut kin = vec![C64::new(0.0, 0.0); g.n];
        for k in 0..g.n {
            let p = g.p(k);
            kin[g.fft_index(k)] = C64::from_polar(1.0 / g.n as f64, -0.5 * p * p * dt);
        }
        self.kinetic = kin;
        self.cached_dt = dt;
    }

    /// One Strang step of length `dt`.
    pub fn step(&mut self, wave: &mut Wave1D, dt: f64) {
        debug_assert_eq!(wave.grid, self.grid, "SplitStep: wave on a different grid");
        self.prepare(dt);
        let psi = &mut wave.psi;
        psi.iter_mut().zip(&self.half_v).for_each(|(a, b)| *a *= b);
        fft::forward(psi);
        psi.iter_mut().zip(&self.kinetic).for_each(|(a, b)| *a *= b);
        fft::inverse(psi);
        psi.iter_mut().zip(&self.half_v).for_each(|(a, b)| *a *= b);
    }

    /// Advances by exactly `t` (≥ 0) in ⌈t/dt_max⌉ equal steps.
    pub fn advance(&mut self, wave: &mut Wave1D, t: f64) {
        if t <= 0.0 {
            return;
        }
        let n = (t / self.dt_max).ceil().max(1.0) as usize;
        let dt = t / n as f64;
        for _ in 0..n {
            self.step(wave, dt);
        }
    }
}
