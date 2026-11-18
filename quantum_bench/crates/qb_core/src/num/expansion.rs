//! A state written in an eigenbasis, ψ(t) = Σ c_n e^(−iE_n t) ψ_n, evolved exactly.

use std::sync::Arc;

use super::{C64, EigenBasis, Wave1D};

/// Coefficients c_n (at t = 0) in an eigenbasis, plus the current time.
///
/// The basis is shared (`Arc`), so cloning an expansion is cheap.
#[derive(Clone, Debug)]
pub struct EigenExpansion {
    pub basis: Arc<EigenBasis>,
    /// c_n at t = 0
    pub c0: Vec<C64>,
    /// current time
    pub t: f64,
}

impl EigenExpansion {
    /// c_n = ⟨ψ_n|ψ⟩ (t = 0). A truncated basis captures only part of ψ:
    /// see [`Self::captured`].
    pub fn project(basis: Arc<EigenBasis>, wave: &Wave1D) -> Self {
        let dx = basis.grid.dx;
        let c0 = basis
            .states
            .iter()
            .map(|s| s.iter().zip(&wave.psi).map(|(a, b)| b * *a).sum::<C64>() * dx)
            .collect();
        EigenExpansion { basis, c0, t: 0.0 }
    }

    /// Given coefficients (missing ones are 0, extra ones ignored), t = 0.
    pub fn from_coeffs(basis: Arc<EigenBasis>, coeffs: &[C64]) -> Self {
        let mut c0 = vec![C64::new(0.0, 0.0); basis.len()];
        for (c, v) in c0.iter_mut().zip(coeffs) {
            *c = *v;
        }
        EigenExpansion { basis, c0, t: 0.0 }
    }

    /// Coherent state |α⟩ in an oscillator basis (PS10 40):
    /// c_n = e^(−|α|²/2) αⁿ/√(n!), textbook sign convention of [`EigenBasis::harmonic`].
    pub fn coherent(basis: Arc<EigenBasis>, alpha: C64) -> Self {
        let mut c = Vec::with_capacity(basis.len());
        let mut term = C64::new((-0.5 * alpha.norm_sqr()).exp(), 0.0);
        for n in 0..basis.len() {
            if n > 0 {
                term = term * alpha / (n as f64).sqrt();
            }
            c.push(term);
        }
        EigenExpansion { basis, c0: c, t: 0.0 }
    }

    pub fn len(&self) -> usize {
        self.c0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.c0.is_empty()
    }

    /// c_n(t) = c_n e^(−iE_n t) at the current time.
    pub fn coeff(&self, n: usize) -> C64 {
        self.c0[n] * C64::from_polar(1.0, -self.basis.energies[n] * self.t)
    }

    /// All c_n(t) at the current time.
    pub fn coeffs(&self) -> Vec<C64> {
        (0..self.len()).map(|n| self.coeff(n)).collect()
    }

    /// |c_n|².
    pub fn probabilities(&self) -> Vec<f64> {
        self.c0.iter().map(|c| c.norm_sqr()).collect()
    }

    /// Σ|c_n|²: how much of the state the (truncated) basis holds.
    pub fn captured(&self) -> f64 {
        self.c0.iter().map(|c| c.norm_sqr()).sum()
    }

    /// Scales the coefficients to Σ|c_n|² = 1.
    pub fn normalize(&mut self) {
        let s = self.captured();
        if s > 0.0 {
            let f = 1.0 / s.sqrt();
            self.c0.iter_mut().for_each(|c| *c *= f);
        }
    }

    /// ⟨E⟩ = Σ|c_n|² E_n / Σ|c_n|².
    pub fn mean_energy(&self) -> f64 {
        let s = self.captured();
        if s <= 0.0 {
            return 0.0;
        }
        self.c0.iter().zip(&self.basis.energies).map(|(c, e)| c.norm_sqr() * e).sum::<f64>() / s
    }

    pub fn set_time(&mut self, t: f64) {
        self.t = t;
    }

    pub fn advance(&mut self, dt: f64) {
        self.t += dt;
    }

    /// ψ(x, t) = Σ c_n(t) ψ_n(x) at the current time.
    pub fn wave(&self) -> Wave1D {
        let g = self.basis.grid;
        let mut psi = vec![C64::new(0.0, 0.0); g.n];
        for (n, s) in self.basis.states.iter().enumerate() {
            let c = self.coeff(n);
            if c.norm_sqr() < 1e-30 {
                continue;
            }
            for (p, v) in psi.iter_mut().zip(s) {
                *p += c * *v;
            }
        }
        Wave1D { grid: g, psi }
    }

    /// Collapses onto ψ_n at the current time, keeping the phase of c_n(t).
    pub fn collapse_to(&mut self, n: usize) {
        let phase = {
            let c = self.coeff(n);
            if c.norm() > 0.0 { c / c.norm() } else { C64::new(1.0, 0.0) }
        };
        let back = C64::from_polar(1.0, self.basis.energies[n] * self.t);
        self.c0.iter_mut().for_each(|c| *c = C64::new(0.0, 0.0));
        self.c0[n] = phase * back;
    }
}
