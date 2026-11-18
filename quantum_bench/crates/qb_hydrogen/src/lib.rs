//! Quantum Bench tab "Hydrogen": orbitals ψ_nlm = R_nl Y_lm for n ≤ 4,
//! superpositions, a GPU ray-marched 3D density, measurements (PS11, PS12).
//!
//! * [`orbital`]: the eigenfunctions (associated Laguerre / Legendre, Condon–Shortley
//!   phases as in PS11 Eq. 44.4), integrals, angular momentum operators, Born sampling;
//! * [`state`]: time evolution and projective measurements (E, l², l_z, l_y);
//! * [`view3d`] + `orbital.wgsl`: the 3D view (cloud, isosurface, point cloud, cut plane);
//! * [`panels`], [`zeeman`]: level diagram, radial panel, slice, Zeeman diagram and spectrum;
//! * [`scene`]: the five presets.

pub mod notes;
pub mod orbital;
pub mod panels;
pub mod scene;
pub mod state;
pub mod view3d;
pub mod zeeman;

use qb_core::{PresetInfo, SceneTab, Tab};
use scene::{HydrogenScene, Variant};

/// The "Hydrogen" tab (the app calls `HydrogenTab::new()`).
pub struct HydrogenTab;

impl HydrogenTab {
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> impl Tab {
        SceneTab::new("hydrogen", "Hydrogen")
            .preset(PresetInfo::new("PS11 43 · Quantum numbers n, l, m"), |_| {
                Box::new(HydrogenScene::new(Variant::Numbers))
            })
            .preset(PresetInfo::new("PS11 44 · Energy and radius"), |_| Box::new(HydrogenScene::new(Variant::Radius)))
            .preset(PresetInfo::new("PS11 45 · ψ₂₁₁ and ψ₂₁₋₁ superposition"), |_| {
                Box::new(HydrogenScene::new(Variant::Ly))
            })
            .preset(PresetInfo::new("Superposition 1s + 2p (oscillating dipole)"), |_| {
                Box::new(HydrogenScene::new(Variant::Dipole))
            })
            .preset(PresetInfo::new("PS12 47 · Zeeman effect"), |_| Box::new(HydrogenScene::new(Variant::Zeeman)))
    }
}
