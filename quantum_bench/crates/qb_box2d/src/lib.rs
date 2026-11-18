//! Quantum Bench tab "2D box": the particle in a box of PS09 38 seen through
//! 2D cross-sections: eigenstates and degeneracies, superpositions with exact
//! time evolution, Gaussian packets and their revival, measurements.
//!
//! * [`modes`]: the numerics (separable sine modes, energies of Eq. 38.2,
//!   degenerate levels, projection, moments, current, Gaussian expansion);
//! * [`ladder`]: an energy ladder that groups degenerate states;
//! * [`view`]: colour maps, marginal plot, node lines, vortices, surface view;
//! * [`scene`]: the scene ([`scene::BoxScene`]) used by all presets.

pub mod ladder;
pub mod modes;
pub mod scene;
pub mod view;

use qb_core::{PresetInfo, SceneTab, Tab};
use scene::{BoxScene, Variant};

/// The "2D box" tab (the app calls `Box2dTab::new()`).
pub struct Box2dTab;

impl Box2dTab {
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> impl Tab {
        SceneTab::new("box2d", "2D box")
            .preset(PresetInfo::new("PS09 38 · Eigenstates and degeneracy"), |_| {
                Box::new(BoxScene::new(Variant::Eigen))
            })
            .preset(PresetInfo::new("PS09 38 · Sloshing: (1,1) + (2,1)"), |_| {
                Box::new(BoxScene::new(Variant::Slosh))
            })
            .preset(PresetInfo::new("PS09 38 · Degenerate pair: (1,2) + i·(2,1)"), |_| {
                Box::new(BoxScene::new(Variant::Degenerate))
            })
            .preset(PresetInfo::new("PS09 38 · Gaussian packet and revival"), |_| {
                Box::new(BoxScene::new(Variant::Packet))
            })
            .preset(PresetInfo::new("PS09 38 · Packet with momentum (billiard)"), |_| {
                Box::new(BoxScene::new(Variant::Moving))
            })
    }
}
