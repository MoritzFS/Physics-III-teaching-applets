//! 1D numerics in units ħ = m = 1 on a uniform periodic grid.
//!
//! | type / fn | what |
//! |---|---|
//! | [`Grid1D`] | x grid and its momentum grid |
//! | [`Wave1D`], [`Momentum`], [`Moments`] | ψ(x), Φ(p), ⟨x⟩ σ_x ⟨p⟩ σ_p |
//! | [`SplitStep`], [`Absorber`] | time evolution for H = p²/2 + V(x), V complex allowed |
//! | [`EigenBasis`], [`ho_psi`], [`well_psi`] | bound states (numerical and analytic) |
//! | [`EigenExpansion`] | c_n, exact evolution by phases, reconstruction |
//! | [`Piecewise`], [`numerov_scatter`] | stationary scattering T(E), R(E), ψ_E(x) |
//! | [`measure_position`] … [`ShotHistogram`] | Born rule, collapse, repeated shots |
//! | [`fft`] | cached 1D/2D FFTs (also for the 2D tabs) |

pub mod eigen;
pub mod expansion;
pub mod fft;
pub mod grid;
pub mod measure;
pub mod propagate;
pub mod scatter;
pub mod wave1d;

/// Complex number used everywhere (f64).
pub type C64 = num_complex::Complex64;

/// Shorthand for `C64::new(re, im)`.
#[inline]
pub const fn c64(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

pub use eigen::{EigenBasis, fix_sign_first_lobe, ho_energy, ho_psi, ho_psi_all, tridiagonal_lowest, well_energy, well_psi};
pub use expansion::EigenExpansion;
pub use grid::Grid1D;
pub use measure::{
    Bins, Sampler, ShotHistogram, born_sample_x, born_samples_p, born_samples_x, collapse_momentum, collapse_position, measure_energy,
    measure_momentum, measure_position, measure_region,
};
pub use propagate::{Absorber, SplitStep};
pub use scatter::{NumerovScattering, Piecewise, PiecewiseScattering, numerov_scatter, rect_barrier_transmission};
pub use wave1d::{Moments, Momentum, Wave1D, free_gaussian_psi, free_gaussian_sigma};
