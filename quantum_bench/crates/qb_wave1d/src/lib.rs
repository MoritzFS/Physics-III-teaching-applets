//! Quantum Bench tab "1D waves": one scene per exercise (PS09 36/37, PS10 38–41, …).
//!
//! Each preset of the "Examples" menu builds its own [`qb_core::Scene`];
//! [`gaussian::GaussianScene`] (PS10 38) is the reference scene to copy from.

pub mod gaussian;

use qb_core::{PresetInfo, SceneTab, Tab};

/// The "1D waves" tab (the app calls `Wave1dTab::new()`).
pub struct Wave1dTab;

impl Wave1dTab {
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> impl Tab {
        SceneTab::new("wave1d", "1D waves")
            .preset(PresetInfo::new("PS10 38 · Gaussian wave packet"), |_| {
                Box::new(gaussian::GaussianScene::new())
            })
            .preset(PresetInfo::new("PS10 38 · Moving packet, k₀ = 3 (Re ψ)"), |_| {
                Box::new(gaussian::GaussianScene::moving())
            })
    }
}
